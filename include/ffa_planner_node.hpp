// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <atomic>
#include <cstdint>
#include <cmath>
#include <condition_variable>
#include <thread>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <std_msgs/msg/int8.hpp>

#include "astar_vdb.hpp"
#include "connect_astar.hpp"
#include "fov_astar.hpp"
#include "occlusion_analyzer.hpp"
#include "observe_astar.hpp"
#include "traj_optimizer/traj_optimizer.hpp"

#include "pid_path_tracker/msg/tracking_point_state.hpp"
#include "rclcpp/rclcpp.hpp"
#include <openvdb/io/Stream.h>
#include <vdb_edt/vdbmap.h>

// Parallel State Machine FFA Planner
//
// Global planner: dedicated thread, event-driven (new goal / replan request).
// Local planner:  high-freq timer (~10 Hz), perception-aware navigation (TODO).
//
// Shared data flows from Global to Local through a read-write lock.

class FFAPlannerNode
{
public:
    FFAPlannerNode();
    ~FFAPlannerNode();

    rclcpp::Node::SharedPtr get_node_handle() const;

private:
    rclcpp::Node::SharedPtr node_handle_;

    // State Machines
    enum class GlobalStatus : uint8_t { IDLE, PLANNING, READY };
    enum class LocalStatus  : uint8_t { IDLE, NAVIGATING, RECOVERY };
    // Published on /goal_reach_status: 0 in-progress, 1 reached, 2 failed.
    enum class GoalReachStatus : int8_t { IN_PROGRESS = 0, REACHED = 1, FAILED = 2 };
    enum class VisLayer : uint8_t { ROOT, SUBGOAL };
    enum class DebugTreeKind : uint8_t { GUIDANCE, FOV, OBSERVE, CONNECT };
    enum class SubgoalType : uint8_t { ROOT_GOAL, FOV_CLEAR, OBSERVE_UNKNOWN };
    enum class SubgoalStatus : uint8_t { ACTIVE, EXECUTING };

    struct HitpointKey
    {
        openvdb::Coord coord{0, 0, 0};

        bool operator==(const HitpointKey &other) const noexcept
        {
            return coord == other.coord;
        }
    };

    struct HitpointKeyHash
    {
        std::size_t operator()(const HitpointKey &key) const noexcept
        {
            return CoordHash{}(key.coord);
        }
    };

    std::atomic<GlobalStatus> global_status_{GlobalStatus::IDLE};
    std::atomic<LocalStatus>  local_status_{LocalStatus::IDLE};
    // Starts "reached"; flips to IN_PROGRESS on new goal, REACHED on arrival,
    // FAILED when the global planner can't reach a freshly received goal.
    std::atomic<GoalReachStatus> goal_reach_status_{GoalReachStatus::REACHED};

    static const char *globalStatusStr(GlobalStatus s);
    static const char *localStatusStr(LocalStatus s);
    static const char *subgoalTypeStr(SubgoalType t);

    struct SafeEdgeHitResult
    {
        bool path_blocked = false;
        bool safe_edge_reached = false;
        size_t hit_idx = 0;
        openvdb::Vec3d hit_pt{0, 0, 0};
        openvdb::Coord hit_ijk{0, 0, 0};
        int hit_inflate_val = 0;
        // Inflate value of the voxel one below the hit, probed at the same
        // instant: tells a DDA phase-skip over a stably occupied voxel apart
        // from a genuine map change between two calls.
        int below_inflate_val = 0;
        bool has_search_start = false;
        openvdb::Coord search_start_ijk{0, 0, 0};
        openvdb::Vec3d search_start_world{0, 0, 0};
    };

    enum class FovCandidateState : uint8_t
    {
        INVALID,
        CERTIFIED,
        CONTAMINATED
    };

    enum class CandidateSetVisKind : uint8_t
    {
        FOV,
        OBSERVE
    };

    struct CandidateSetSnapshot
    {
        std::vector<SearchGoalCandidate> certified_candidates;
        std::vector<SearchGoalCandidate> contaminated_candidates;
        openvdb::Coord selected_coord{0, 0, 0};
        ReachLabel selected_label = ReachLabel::CERTIFIED;
        bool has_selected = false;
    };

    struct PreviewProgressWindow
    {
        bool valid = false;
        openvdb::Vec3d window_start{0, 0, 0};
        openvdb::Vec3d window_end{0, 0, 0};
        openvdb::Vec3d predicted_clear_point{0, 0, 0};
        double predicted_clear_distance = 0.0;
        std::vector<openvdb::Vec3d> samples;
        std::vector<double> sample_progress;
    };

    struct PreviewSegmentPlan
    {
        bool valid = false;
        openvdb::Vec3d last_clearable_point{0, 0, 0};
        openvdb::Vec3d hypothetical_hit_point{0, 0, 0};
        openvdb::Vec3d viewpoint{0, 0, 0};
        openvdb::Vec3d stop_point{0, 0, 0};
        double target_yaw = 0.0;
        double fov_search_ms = 0.0;
        std::vector<openvdb::Vec3d> fov_path_raw;
        std::vector<openvdb::Vec3d> fov_path_short;
        std::vector<openvdb::Vec3d> guidance_raw;
        std::vector<openvdb::Vec3d> guidance_short;
        std::vector<openvdb::Vec3d> exec_path;
        std::string failure_reason;
        PreviewProgressWindow progress;
    };

    struct PreviewTrajectoryExecutionResult
    {
        bool published = false;
        bool used_visibility_optimization = false;
        size_t stitched_path_size = 0;
        openvdb::Vec3d active_preview_anchor{0, 0, 0};
        double active_preview_yaw = 0.0;
        // Direct handle for activateStitchMonitoring(). A zero trajectory
        // stamp or anchor_time < 0 mean the publish path did not produce a
        // usable anchor (e.g. fallback into interpolateAndPublishTrajectory)
        // and stitch monitoring should NOT be armed.
        builtin_interfaces::msg::Time trajectory_stamp;
        double anchor_time = -1.0;
    };

    // Bookkeeping returned by every publish function (optimizer-based and
    // linear-interpolation alike). published=false means the publisher elected
    // not to publish (e.g. invalid input, optimization failure with no
    // fallback). anchor_time < 0 means the published plan has no anchor
    // (single-stage trajectories). trajectory_stamp is copied from the
    // MultiDOFJointTrajectory header and matched against path-tracker
    // TrackingPointState messages.
    struct PublishResult
    {
        bool published = false;
        uint64_t traj_id = 0;
        builtin_interfaces::msg::Time trajectory_stamp;
        double total_duration = 0.0;
        double anchor_time = -1.0;
    };

    // Snapshot of the most recently published MultiDOFJointTrajectory.
    // Stored in a sparse form (per-waypoint positions + yaws + arclength +
    // time_from_start). For piecewise-linear publishers (straight-line
    // interpolate, stitched, yaw-in-place) the dense message published to
    // the controller is mathematically equivalent to linearly interpolating
    // these sparse points, so the cache is lossless for lookahead sampling.
    //
    // cruise_speed == 0 signals a stationary (yaw-only) cache, in which case
    // positions[] are all identical and only yaws[] / times_from_start[] vary.
    struct PublishedTrajectory
    {
        uint64_t traj_id = 0;
        rclcpp::Time t_published;

        std::vector<Eigen::Vector3d> positions;
        std::vector<Eigen::Vector3d> velocities;
        std::vector<Eigen::Vector3d> accelerations;
        std::vector<double> yaws;
        std::vector<double> cum_arclen;
        std::vector<double> times_from_start;

        double cruise_speed = 0.0;
        bool valid = false;
    };

    struct SubgoalNode
    {
        uint64_t node_id = 0;
        SubgoalType type = SubgoalType::ROOT_GOAL;
        SubgoalStatus status = SubgoalStatus::ACTIVE;

        // Static identity: anchor_point is the upstream hit_pt that justified
        // this node's existence (used by the top-down necessity check). For
        // ROOT_GOAL it is the user-provided goal position. goal_pos / goal_yaw
        // are where the parent guidance lands this node before its own local
        // FOV / observe phase begins.
        openvdb::Vec3d anchor_point{0, 0, 0};
        openvdb::Vec3d goal_pos{0, 0, 0};
        double goal_yaw = 0.0;

        std::vector<openvdb::Vec3d> guidance_path_raw;
        std::vector<openvdb::Vec3d> guidance_path_short;

        // Active local plan state (current FOV-clear / observe cycle).
        bool has_active_plan = false;
        bool has_preview_plan = false;
        bool goal_traj_published = false;
        openvdb::Vec3d active_viewpoint{0, 0, 0};
        openvdb::Vec3d active_anchor_seed{0, 0, 0};
        openvdb::Vec3d active_plan_hit_point{0, 0, 0};
        bool active_plan_has_search_start = false;
        openvdb::Coord active_plan_search_start_ijk{0, 0, 0};
        openvdb::Vec3d active_plan_search_start_world{0, 0, 0};
        bool has_certified_guidance_start = false;
        openvdb::Coord certified_guidance_start_ijk{0, 0, 0};
        openvdb::Vec3d certified_guidance_start_world{0, 0, 0};
        // Frozen guidance origin for this subgoal's optimistic guidance: the
        // plan start (lookahead; robot pose before any trajectory exists) at
        // CREATION time, inherited by descendants so a whole subtree shares
        // one stable polyline anchor (DDA-phase reproducibility), re-frozen
        // only on explicit blocked-replan events. Distinct from
        // certified_guidance_start (= parent a(h)), which stays the
        // viewpoint-search fallback anchor.
        bool has_guidance_origin = false;
        openvdb::Coord guidance_origin_ijk{0, 0, 0};
        openvdb::Vec3d guidance_origin_world{0, 0, 0};
        double active_target_yaw = 0.0;
        struct PendingNoGood
        {
            std::unordered_set<openvdb::Coord, CoordHash> rejected_viewpoint_voxels;
            std::unordered_set<HitpointKey, HitpointKeyHash> exhausted_hitpoints;
        };

        std::unordered_set<HitpointKey, HitpointKeyHash> ancestor_hitpoints;
        // Context-local rejection state. rejected_viewpoint_voxels is Qrej(S):
        // contaminated viewpoint voxels that cannot be certified as subgoals
        // under this context. exhausted_hitpoints is Hrej(S). Child contexts
        // inherit both sets; child-local no-good memory is propagated back only
        // after its dependency hitpoint has entered the parent Hrej(S).
        std::unordered_set<openvdb::Coord, CoordHash> rejected_viewpoint_voxels;
        std::unordered_set<HitpointKey, HitpointKeyHash> exhausted_hitpoints;
        std::unordered_map<HitpointKey, PendingNoGood, HitpointKeyHash> pending_no_good_by_hitpoint;

        bool has_guidance_progress_floor = false;
        openvdb::Vec3d guidance_progress_floor{0, 0, 0};
        bool has_receding_preview_state = false;
        openvdb::Vec3d preview_next_anchor_seed{0, 0, 0};
        double preview_next_anchor_yaw = 0.0;
        PreviewProgressWindow active_anchor_progress;
        PreviewProgressWindow next_anchor_progress;
        HitpointKey last_anchor_hitpoint;
        bool has_last_anchor_hitpoint = false;

        // Observe-phase lock (engineering failsafe, see handleObserveDwell):
        // the unknown voxel currently being observed stays fixed until it
        // leaves U. Viewpoints from which the robot arrived but did not
        // observe it are excluded for this u. Cleared with the execution
        // state; inherited by OBSERVE_UNKNOWN children.
        bool has_locked_unknown = false;
        openvdb::Coord locked_unknown{0, 0, 0};
        HitpointKey locked_unknown_hitpoint;
        std::unordered_set<openvdb::Coord, CoordHash> observe_failed_voxels;
        int observe_arrival_ticks = 0;
        bool observe_dwell_valid = false;
        int observe_dwell_start_count = 0;
        int observe_failure_count = 0;

        std::unique_ptr<SubgoalNode> child;
    };

    // Map
    std::shared_ptr<VDBMap> map_manager_;

    // Planners
    Astar global_astar_;
    Astar subgoal_guidance_astar_;
    FovAstar fov_astar_;
    SensorFovConfig fov_cfg_;
    FovClearProfile fov_standard_profile_;
    FovClearProfile fov_tight_profile_;
    OcclusionAnalyzer occlusion_analyzer_;
    ObserveAstar observe_astar_;
    ConnectAstar connect_astar_;
    std::unique_ptr<traj_optimizer::TrajOptimizer> traj_optimizer_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    bool getRobotPose(Eigen::Vector3d &pos_out, double &yaw_out) const;

    // Callback Groups
    rclcpp::CallbackGroup::SharedPtr cbg_local_;
    rclcpp::CallbackGroup::SharedPtr cbg_subs_;

    // Global planner thread (event-driven, sleeps on CV)
    std::thread global_thread_;
    std::mutex global_cv_mtx_;
    std::condition_variable global_cv_;
    std::atomic<bool> shutdown_{false};
    std::optional<openvdb::Coord> pending_global_replan_start_override_;

    void globalPlannerLoop();
    void wakeGlobalPlanner();

    // Local timer
    rclcpp::TimerBase::SharedPtr local_timer_;
    rclcpp::TimerBase::SharedPtr goal_reach_status_timer_;

    // Shared Data (Global -> Local)
    mutable std::shared_mutex guidance_path_mtx_;
    std::vector<openvdb::Vec3d> shared_guidance_path_;       // A* raw waypoints
    std::vector<openvdb::Vec3d> shared_guidance_path_short_; // shortened path

    std::atomic<bool> need_global_replan_{false};
    std::unique_ptr<SubgoalNode> root_subgoal_;
    uint64_t next_subgoal_id_ = 1;
    // Guards context-local refutation sets stored on SubgoalNode objects.
    mutable std::recursive_mutex refute_state_mtx_;
    // Completeness backstop cooldown (see localTimerCallback): allows at most
    // one memory-reset retry per window when root guidance fails.
    double last_refutation_backstop_sec_ = -1.0;

    // Goal
    std::mutex goal_mtx_;
    Eigen::Vector3d curr_goal_xyz_{0, 0, 0};
    double curr_goal_yaw_ = 0.0;
    std::atomic<bool> has_new_goal_{false};

    // Tracking Point (from downstream controller)
    mutable std::shared_mutex tp_mtx_;
    bool has_tracking_point_ = false;
    geometry_msgs::msg::PoseStamped current_tracking_point_;

    // Cache of last published trajectory (for lookahead-point computation).
    // Writers: interpolateAndPublishTrajectory / interpolateAndPublishStitchedTrajectory
    //          / publishYawOnlyDegenerateTrajectory (after successful publish).
    // Invalidator: resetLocalPlanState (new goal / LOCAL IDLE transitions).
    // Access is guarded by last_traj_mtx_ (shared_mutex: writers take unique,
    // readers in sampleLastPublishedTrajectory take shared).
    mutable std::shared_mutex last_traj_mtx_;
    PublishedTrajectory last_published_traj_;
    std::atomic<uint64_t> next_traj_id_{1};

    // Lookahead point sampled from the last published trajectory.
    // Refreshed in tracking_point_state_callback (primary) using the
    // tracker-reported progress (so it advances only as fast as the vehicle
    // actually tracks, not open-loop with wall-clock):
    //   t_eval = time_from_start + lookahead_buffer_time_s_
    // Readers: planner callsites that replace `robot_pos` as search start
    // grab this via getLookaheadState().
    struct LookaheadState
    {
        Eigen::Vector3d pos{0.0, 0.0, 0.0};
        Eigen::Vector3d vel{0.0, 0.0, 0.0};
        Eigen::Vector3d acc{0.0, 0.0, 0.0};
        double yaw = 0.0;
        double t_eval = 0.0;   // seconds from t_published used for sampling
        uint64_t traj_id = 0;  // source cache id; 0 means invalid
        bool valid = false;
        bool at_end = false;   // clamped to trajectory end
    };

    mutable std::shared_mutex lookahead_mtx_;
    LookaheadState cached_lookahead_;

    // Set true whenever the most recent tracking_point diverges from where
    // the cached trajectory says the controller should be right now.
    std::atomic<bool> tp_divergent_{false};

    // Stitched FOV plan: vp1 arrival detection via tracking point
    std::mutex stitch_monitor_mtx_;
    bool stitch_monitoring_active_ = false;
    builtin_interfaces::msg::Time stitch_monitored_trajectory_stamp_;
    // Time-from-start (seconds along the published trajectory) at which the
    // vp1 / observation anchor is reached. Supplied directly by the optimizer
    // / publish path -- no 3D projection needed.
    double stitch_monitored_anchor_time_ = 0.0;
    bool stitch_vp1_reached_ = false;

    // Parameters
    std::string world_frame_id_;
    std::string robot_frame_id_;
    double voxel_size_ = 0.2;
    bool vis_debug_tree_ = false;
    bool vis_fov_debug_tree_ = false;
    bool vis_traj_optimizer_ = true;
    // FFA-search ablation: when true, skip the SFC optimizer in both publish
    // paths and emit the searched waypoints via the linear interpolator.
    bool disable_traj_opt_ = false;

    double local_timer_period_ = 0.1;

    // A* config (loaded from yaml, fed to AstarParams)
    int global_astar_allocate_num_ = 5000000;
    double global_astar_max_search_time_ = 0.0;
    double global_astar_lambda_heu_ = 1.0;
    double clearance_soft_cost_multiplier_ = 1.0;

    int fov_astar_allocate_num_ = 500000;
    double fov_astar_max_search_time_ = 0.1;
    double fov_astar_lambda_heu_ = 2.0;

    int observe_astar_allocate_num_ = 500000;
    double observe_astar_max_search_time_ = 0.2;
    double observe_astar_lambda_heu_ = 2.0;

    int connect_astar_allocate_num_ = 500000;
    double connect_astar_max_search_time_ = 0.2;
    double connect_astar_lambda_heu_ = 1.0;
    double preview_connect_max_search_time_ = 0.05;

    // FOV / Observe A* anchor-bias shaping (passed to both fov_astar_ and observe_astar_)
    double anchor_bias_away_weight_ = 0.1;
    double anchor_bias_max_multiplier_ = 1.1;

    // Subscribers
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_tracking_point_;
    rclcpp::Subscription<pid_path_tracker::msg::TrackingPointState>::SharedPtr sub_tracking_point_state_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr sub_goal_pose_;

    // Publishers
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_guidance_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_guidance_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_debug_tree_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr pub_trajectory_;
    rclcpp::Publisher<std_msgs::msg::Int8>::SharedPtr pub_goal_reach_status_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_safe_edge_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_fov_viewpoint_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_fov_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_fov_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_fov_candidate_set_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_fov_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_observe_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_observe_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_observe_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_observe_candidate_set_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_observe_frustum_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_connect_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_connect_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_connect_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_fov_frustum_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_guidance_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_guidance_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_subgoal_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_safe_edge_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_fov_viewpoint_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_fov_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_fov_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_subgoal_fov_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_subgoal_observe_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_observe_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_observe_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub_subgoal_connect_debug_tree_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_connect_path_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_connect_short_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_subgoal_fov_frustum_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_traj_optimizer_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_traj_sfc_vis_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_lookahead_vis_;

    // Local Timer Callback
    void localTimerCallback();
    void goalReachStatusTimerCallback();

    // Subscription Callbacks
    void tracking_point_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void tracking_point_state_callback(const pid_path_tracker::msg::TrackingPointState::SharedPtr msg);
    void goal_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg);

    bool needs_escape_corridor_ = true;
    double yaw_reach_tol_ = 0.15;  // ~8.6 deg

    // Trajectory interpolation params
    double interpolate_step_ = 0.2;
    double cruise_speed_ = 1.0;
    double viewpoint_reach_tol_ = 0.5;
    double fov_stitch_lookahead_dist_ = 1.5;
    double tight_additional_radius_xy_ = 0.0;
    double tight_additional_height_z_ = 0.0;
    double visibility_cell_roi_xy_ = 1.5;
    double visibility_cell_roi_z_ = 0.8;
    int visibility_cell_erode_iters_ = 1;
    double visibility_cell_min_radius_ = 0.1;
    double visibility_cell_shrink_m_ = 0.1;
    int visibility_cell_max_carve_ = 3;
    bool visibility_cell_fixed_yaw_ = true;
    double fov_theta_h_ = -1.0;

    // Observe-phase failsafe params (arrived at observe VP, u not resolved)
    int observe_dwell_map_updates_ = 3;      // map updates to wait at a geometrically valid pose
    int observe_arrival_max_ticks_ = 10;     // ticks within reach_tol without passing geometry
    double observe_resume_budget_ = 0.2;     // s, extra budget per resumed observe search
    double observe_graze_penalty_vox_ = 5.0; // connect seed cost for grazing candidates (voxels)
    int observe_graze_skip_vox_ = 2;         // ray voxels next to the target ignored by grazing test

    // Lookahead params
    double lookahead_buffer_time_s_ = 0.2;    // traj-time to sample ahead of tracker progress
    bool lookahead_progress_driven_ = true;   // false = benchmark-era wall-clock lookahead
    double tp_divergence_threshold_m_ = 0.5;  // |tp_pos - wall-clock scheduled pos| above this flags lag

    // Helpers
    void initialize();
    void setup_parameters();
    void resetLocalPlanState();
    void resetRefutationState();
    void clearRefutationStateRecursive(SubgoalNode &node);
    void resetSubgoalChain();
    void clearSubgoalVisualization();
    SubgoalNode *ensureRootSubgoal();
    void refreshRootSubgoalGoal();
    bool refreshRootSubgoalGuidance();
    bool planGuidanceToSubgoal(SubgoalNode &node,
                               const Eigen::Vector3d &robot_pos);
    void updateNodeGuidance(SubgoalNode &node,
                            const std::vector<openvdb::Vec3d> &raw_path,
                            const std::vector<openvdb::Vec3d> &short_path);
    void clearNodeExecutionState(SubgoalNode &node);
    SubgoalNode *createOrUpdateChildSubgoal(SubgoalNode &parent,
                                            SubgoalType type,
                                            const openvdb::Vec3d &anchor_point,
                                            const openvdb::Vec3d &goal_pos,
                                            double goal_yaw);
    void pruneSubgoalChildren(SubgoalNode &node);
    bool pruneSubgoalNodeById(SubgoalNode &node, uint64_t node_id);
    SubgoalNode *findParentSubgoalByChildId(SubgoalNode &node, uint64_t child_id);
    SubgoalNode *getDeepestActiveSubgoal();
    void publishGuidancePathVis(const std::vector<openvdb::Vec3d> &raw_path,
                                const std::vector<openvdb::Vec3d> &short_path,
                                VisLayer layer);
    void publishDebugTreeVis(const visualization_msgs::msg::Marker &tree_marker,
                             DebugTreeKind kind,
                             VisLayer layer);
    SafeEdgeHitResult findFirstSafeEdgeHit(const std::vector<openvdb::Vec3d> &path_world) const;
    bool isHitPointCleared(const openvdb::Vec3d &hit_pt) const;
    PublishResult interpolateAndPublishTrajectory(const std::vector<openvdb::Vec3d> &waypoints,
                                                  double start_yaw, double final_yaw,
                                                  bool allow_optimization = true);
    PublishResult interpolateAndPublishStitchedTrajectory(
        const std::vector<openvdb::Vec3d> &waypoints,
        size_t split_index,
        double start_yaw,
        double split_yaw,
        double final_yaw,
        bool allow_optimization = true);
    // Degenerate-input fallback used only by interpolateAndPublishTrajectory
    // when total path length collapses to zero. Not a planner-level branch.
    PublishResult publishYawOnlyDegenerateTrajectory(const Eigen::Vector3d &pos,
                                                     double start_yaw,
                                                     double target_yaw);

    // Wrap a yaw value (or a yaw delta) to (-pi, pi]. Use whenever a yaw is
    // consumed at a trajectory boundary (plan start, target yaw, yaw delta
    // between two unrelated frames, etc.) so cached unwrapped values do not
    // leak into downstream representations and cause the optimizer to pick
    // multi-turn equivalence classes.
    static inline double wrapToPi(double yaw)
    {
        return std::remainder(yaw, 2.0 * M_PI);
    }

    // Cache the sparse representation of a trajectory we just handed to the
    // controller. times_from_start[0] must be 0 and strictly increasing.
    // cruise_speed == 0 marks a stationary (yaw-in-place) cache.
    // Returns the assigned internal traj_id (0 if caching was skipped due to
    // invalid input). This id is only for the lookahead cache; stitch
    // monitoring uses the published trajectory header stamp.
    // Thread-safe: takes last_traj_mtx_ in unique mode internally.
    uint64_t cachePublishedTrajectory(std::vector<Eigen::Vector3d> positions,
                                      std::vector<Eigen::Vector3d> velocities,
                                      std::vector<Eigen::Vector3d> accelerations,
                                      std::vector<double> yaws,
                                      std::vector<double> cum_arclen,
                                      std::vector<double> times_from_start,
                                      double cruise_speed,
                                      const rclcpp::Time &t_published);

    // Mark the cached trajectory invalid. Call on new-goal / IDLE transitions
    // where the controller is no longer expected to follow the last plan.
    void invalidateLastPublishedTrajectory();

    // Sample the cached trajectory at a wall-clock offset (seconds since
    // t_published). Returns false when the cache is invalid. Clamps to the
    // two endpoints when t_eval is out-of-range and sets at_end=true at the
    // trailing clamp. Thread-safe: takes last_traj_mtx_ in shared mode.
    bool sampleLastPublishedTrajectory(double t_eval,
                                       Eigen::Vector3d &pos,
                                       Eigen::Vector3d &vel,
                                       Eigen::Vector3d &acc,
                                       double &yaw,
                                       bool &at_end,
                                       uint64_t &traj_id_out) const;
    bool projectPointOntoLastPublishedTrajectory(const Eigen::Vector3d &query_pos,
                                                 double &time_from_start,
                                                 double &arclen,
                                                 uint64_t &traj_id_out,
                                                 double *distance_to_traj_out = nullptr) const;

    // Refresh cached_lookahead_ from tracking_point_state_callback. Samples
    // the cached trajectory at the tracker-reported progress (time_from_start
    // + lookahead_buffer_time_s_) after verifying trajectory_stamp matches the
    // cache. Also runs the lag check (tp_pos vs wall-clock scheduled pos at
    // t_elapsed), updates tp_divergent_, and publishes the lookahead marker.
    void refreshLookaheadFromTracking(const rclcpp::Time &now,
                                      const Eigen::Vector3d &tp_pos,
                                      double tp_time,
                                      const builtin_interfaces::msg::Time &trajectory_stamp);

    // Thread-safe accessor for planner-side callers.
    LookaheadState getLookaheadState() const;

    // Preferred start pose for all A* search callsites: lookahead when valid,
    // else falls back to the current TF (getRobotPose). Returns false only
    // when lookahead is invalid AND TF is not yet ready. WARNs (throttled)
    // whenever fallback is taken so we can see how often it happens.
    // NOTE: arrival/distance checks should keep using getRobotPose directly.
    bool getPlanStartState(Eigen::Vector3d &pos, double &yaw) const;

    // Publish a small MarkerArray (sphere + yaw arrow + connector to tp)
    // showing the current lookahead sample in RViz.
    void publishLookaheadVis(const LookaheadState &state,
                             const Eigen::Vector3d &tp_pos);
    void clearLookaheadVis();
    void publishSafeEdgeVis(const SafeEdgeHitResult &result, VisLayer layer);
    void publishActiveFrustumVis(const openvdb::Vec3d &viewpoint,
                                 const openvdb::Vec3d &target,
                                 VisLayer layer,
                                 bool tight_primary = false);
    void publishFovPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                           const std::vector<openvdb::Vec3d> &short_path,
                           const openvdb::Vec3d &viewpoint,
                           const openvdb::Vec3d &target,
                           VisLayer layer,
                           bool tight_primary = false);
    void publishSecondaryFovPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                                    const std::vector<openvdb::Vec3d> &short_path,
                                    const openvdb::Vec3d &viewpoint,
                                    const openvdb::Vec3d &target,
                                    VisLayer layer);
    void clearSecondaryFovPlanVis(VisLayer layer);
    void publishCandidateSetVis(CandidateSetVisKind kind,
                                const std::vector<SearchGoalCandidate> &certified_candidates,
                                const std::vector<SearchGoalCandidate> &contaminated_candidates,
                                const openvdb::Coord &selected_coord,
                                bool has_selected,
                                ReachLabel selected_label);
    void clearCandidateSetVis(CandidateSetVisKind kind);
    void publishObservePlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                               const std::vector<openvdb::Vec3d> &short_path,
                               const openvdb::Vec3d &viewpoint,
                               const openvdb::Vec3d &target,
                               VisLayer layer);
    void publishObserveFrustumVis(const openvdb::Vec3d &viewpoint,
                                  const openvdb::Vec3d &target,
                                  VisLayer layer);
    void clearObserveFrustumVis();
    void publishConnectPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                               const std::vector<openvdb::Vec3d> &short_path,
                               VisLayer layer);
    void publishOptimizedTrajectoryVis(
        const traj_optimizer::OptimizedPlan::PositionTrajectory &traj,
        const traj_optimizer::DebugInfo *debug_info = nullptr);
    void clearOptimizedTrajectoryVis();
    void publishSfcDebugVis(const traj_optimizer::DebugInfo &debug_info);
    void clearSfcDebugVis();
    traj_optimizer::BoundaryState makeOptimizerStartBoundaryState(
        const openvdb::Vec3d &start_pos) const;
    PublishResult tryPublishOptimizedTrajectory(const std::vector<openvdb::Vec3d> &waypoints,
                                                double start_yaw,
                                                double final_yaw);
    PublishResult tryPublishOptimizedStitchedTrajectory(
        const std::vector<openvdb::Vec3d> &waypoints,
        size_t split_index,
        double start_yaw,
        double split_yaw,
        double final_yaw);
    PublishResult tryPublishOptimizedVisibilityStitchedTrajectory(
        const std::vector<openvdb::Vec3d> &path_in,
        const traj_optimizer::VisibilityCell &visibility_cell,
        const std::vector<openvdb::Vec3d> &path_out,
        double start_yaw,
        double observation_yaw,
        double final_yaw);
    bool executePreviewTrajectoryPlan(std::string_view log_prefix,
                                      const openvdb::Vec3d &plan_start_world,
                                      const std::vector<openvdb::Vec3d> &entry_exec_path_short,
                                      const std::vector<openvdb::Vec3d> &path_prefix,
                                      const openvdb::Vec3d &observation_viewpoint,
                                      double observation_yaw,
                                      const openvdb::Vec3d &visibility_target_world,
                                      const PreviewSegmentPlan &preview_plan,
                                      double start_yaw,
                                      bool allow_optimization,
                                      PreviewTrajectoryExecutionResult &result);
    PublishResult publishOptimizedPlanTrajectory(const traj_optimizer::OptimizedPlan &plan);

    void handleOccludedFrontier(SubgoalNode &leaf, VisLayer layer);
    // Observe failsafe pieces (see handleObserveDwell in the .cpp).
    void handleObserveDwell(SubgoalNode &leaf, VisLayer layer, const Eigen::Vector3d &robot_pos);
    void handleObserveFailure(SubgoalNode &leaf, VisLayer layer, const char *reason);
    void releaseObserveLock(SubgoalNode &leaf);
    void resetObserveArrivalCounters(SubgoalNode &leaf);
    bool observeGeometryOkFromPose(const Eigen::Vector3d &robot_pos,
                                   const openvdb::Coord &unknown_ijk);
    bool isVoxelUnknown(const openvdb::Coord &ijk) const;
    bool selectAndPublishObserveViewpoint(SubgoalNode &leaf,
                                          VisLayer layer,
                                          const Eigen::Vector3d &plan_start_pos,
                                          double plan_start_yaw,
                                          double obs_ms);
    void setActivePlanHitFromSafeEdge(SubgoalNode &node,
                                      const SafeEdgeHitResult &hit);
    void clearActivePlanHitSearchStart(SubgoalNode &node);
    bool isStrictFreeVoxel(const openvdb::Coord &coord) const;
    openvdb::Coord chooseTargetCentricSearchStart(const openvdb::Vec3d &target_hit,
                                                  const openvdb::Vec3d &search_anchor) const;
    bool selectGoalFromSet(const openvdb::Vec3d &start_world,
                           const std::vector<SearchGoalCandidate> &goal_set,
                           openvdb::Vec3d &selected_goal_world,
                           double *elapsed_ms_out = nullptr,
                           std::string *failure_reason_out = nullptr);
    // Same as selectGoalFromSet but seeds grazing candidates with
    // graze_penalty_vox and skips `excluded` voxels: returns argmin over
    // candidates of (seed + certified path cost).
    bool selectGoalFromSeeds(const openvdb::Vec3d &start_world,
                             const std::vector<SearchGoalCandidate> &goal_set,
                             double graze_penalty_vox,
                             const std::unordered_set<openvdb::Coord, CoordHash> &excluded,
                             openvdb::Vec3d &selected_goal_world,
                             double *elapsed_ms_out = nullptr,
                             std::string *failure_reason_out = nullptr);
    bool selectLowestCostGoalFromSet(const std::vector<SearchGoalCandidate> &goal_set,
                                     openvdb::Coord &selected_goal_ijk) const;
    HitpointKey canonicalizeHitpoint(const openvdb::Vec3d &hit_pt) const;
    bool isViewpointRejected(const SubgoalNode &context, const openvdb::Coord &viewpoint_ijk) const;
    bool markViewpointRejected(SubgoalNode &context, const openvdb::Coord &viewpoint_ijk);
    bool markParentViewpointRejected(SubgoalNode &child);
    void markHitpointExhausted(SubgoalNode &leaf,
                               const HitpointKey &hitpoint,
                               const char *reason);
    void stashChildNoGood(SubgoalNode &parent, const SubgoalNode &child);
    void propagatePendingNoGoodForHitpoint(SubgoalNode &context, const HitpointKey &hitpoint);
    bool isHitpointKeyCleared(const HitpointKey &key) const;
    bool invalidateClearedRejections(SubgoalNode &node);
    bool hasRefutationMemory() const;
    std::unordered_set<openvdb::Coord, CoordHash> buildRejectedHitpointBlockedSet(
        const SubgoalNode *context) const;
    std::unordered_set<openvdb::Coord, CoordHash> buildGuidanceBlockedSet(
        const SubgoalNode &node) const;
    FovCandidateState classifyCandidateState(const openvdb::Vec3d &viewpoint) const;
    bool findGuidanceLookAheadTarget(const std::vector<openvdb::Vec3d> &path,
                                     size_t anchor_idx,
                                     const openvdb::Vec3d &anchor_pt,
                                     double look_ahead_dist,
                                     openvdb::Vec3d &target_out) const;
    bool buildTargetCentricFovPlan(const SubgoalNode &context,
                                   const openvdb::Vec3d &target_hit,
                                   const openvdb::Vec3d &search_anchor,
                                   const openvdb::Coord &search_start_ijk,
                                   FovClearProfileKind profile_kind,
                                   bool allow_contaminated_traversal,
                                   std::vector<openvdb::Vec3d> &raw_path_out,
                                   std::vector<openvdb::Vec3d> &short_path_out,
                                   openvdb::Vec3d &viewpoint_out,
                                   double &target_yaw_out,
                                   double &elapsed_ms_out,
                                   FovCandidateState &viewpoint_state_out,
                                   VisLayer layer,
                                   bool publish_debug_tree,
                                   CandidateSetSnapshot *candidate_snapshot_out = nullptr);
    bool buildCertifiedAccessPath(const openvdb::Vec3d &start_world,
                                  const openvdb::Vec3d &goal_world,
                                  std::vector<openvdb::Vec3d> &raw_path_out,
                                  std::vector<openvdb::Vec3d> &short_path_out,
                                  double *elapsed_ms_out = nullptr,
                                  std::string *failure_reason_out = nullptr);
    bool buildStartupEscapePrefix(const Eigen::Vector3d &plan_start_pos,
                                  double plan_start_yaw,
                                  std::vector<openvdb::Vec3d> &prefix_out,
                                  Eigen::Vector3d &virtual_start_out);
    bool buildOptimisticGuidancePath(const openvdb::Vec3d &start_world,
                                     const openvdb::Vec3d &goal_world,
                                     std::vector<openvdb::Vec3d> &raw_path_out,
                                     std::vector<openvdb::Vec3d> &short_path_out,
                                     double *elapsed_ms_out = nullptr);
    bool rayTraceClearKnownOcc(const openvdb::Vec3d &sensor_world,
                               const openvdb::Vec3d &target_world) const;
    bool isTargetClearableFromAnchor(const openvdb::Vec3d &anchor_body_world,
                                     const openvdb::Vec3d &target_world,
                                     FovClearProfileKind profile_kind) const;
    bool isTargetClearableFromAnchorYaw(const openvdb::Vec3d &anchor_body_world,
                                        const openvdb::Vec3d &target_world,
                                        FovClearProfileKind profile_kind,
                                        double psi_body,
                                        bool check_horizontal) const;
    bool buildVisibilityAnchorCell(const openvdb::Vec3d &witness_viewpoint,
                                   const openvdb::Vec3d &target_world,
                                   traj_optimizer::VisibilityCell &cell_out) const;
    double yawTowardTarget(const openvdb::Vec3d &from,
                           const openvdb::Vec3d &target) const;
    double projectProgressAlongPath(const std::vector<openvdb::Vec3d> &path,
                                    const openvdb::Vec3d &point) const;
    std::vector<openvdb::Vec3d> localGuidancePath(const SubgoalNode &node) const;
    bool isAheadOfGuidanceProgressFloor(const SubgoalNode &node,
                                        const openvdb::Vec3d &point,
                                        double eps = 1.0e-4) const;
    void advanceGuidanceProgressFloor(SubgoalNode &node,
                                      const openvdb::Vec3d &point);
    bool buildPreviewProgressWindow(const openvdb::Vec3d &anchor_viewpoint,
                                    const std::vector<openvdb::Vec3d> &guidance_path,
                                    const openvdb::Vec3d &window_start,
                                    double lookahead_dist,
                                    FovClearProfileKind profile_kind,
                                    PreviewProgressWindow &window_out) const;
    bool buildPreviewProgressWindowToTarget(const openvdb::Vec3d &anchor_viewpoint,
                                            const std::vector<openvdb::Vec3d> &guidance_path,
                                            const openvdb::Vec3d &window_start,
                                            const openvdb::Vec3d &window_end,
                                            FovClearProfileKind profile_kind,
                                            PreviewProgressWindow &window_out) const;
    bool findFreshCertifiedWindowJump(const openvdb::Vec3d &anchor_viewpoint,
                                      const std::vector<openvdb::Vec3d> &guidance_path,
                                      const openvdb::Vec3d &window_start,
                                      openvdb::Vec3d &jump_point_out,
                                      std::vector<openvdb::Vec3d> *connect_path_out = nullptr);
    bool buildCertifiedPreviewSegment(const SubgoalNode &context,
                                      const openvdb::Vec3d &anchor_viewpoint,
                                      const std::vector<openvdb::Vec3d> &guidance_path,
                                      const openvdb::Vec3d &window_start,
                                      const openvdb::Coord &search_start_ijk,
                                      VisLayer layer,
                                      PreviewSegmentPlan &plan_out,
                                      const openvdb::Vec3d *fixed_window_end = nullptr);
    bool publishRecedingPreviewFromAnchor(SubgoalNode &leaf,
                                          VisLayer layer,
                                          const openvdb::Vec3d &plan_start_world,
                                          const std::vector<openvdb::Vec3d> &entry_path_short,
                                          const std::vector<openvdb::Vec3d> &path_prefix,
                                          const openvdb::Vec3d &anchor_seed,
                                          const PreviewProgressWindow &anchor_progress,
                                          const PreviewSegmentPlan &next_preview,
                                          double start_yaw,
                                          bool allow_optimization,
                                          const char *log_prefix);
    void promotePreviewAnchorWithoutExtension(SubgoalNode &leaf,
                                              VisLayer layer,
                                              const openvdb::Vec3d &promoted_anchor,
                                              const openvdb::Vec3d &visibility_target);
    bool handleRecedingPreviewAnchorReached(SubgoalNode &leaf,
                                            VisLayer layer,
                                            const Eigen::Vector3d &plan_start_pos,
                                            double plan_start_yaw);
    std::vector<openvdb::Vec3d> prependPathPrefix(const std::vector<openvdb::Vec3d> &prefix,
                                                  const std::vector<openvdb::Vec3d> &suffix) const;
    bool activateFovClearSubgoal(SubgoalNode &parent,
                                 const SafeEdgeHitResult *target_hit_context,
                                 const openvdb::Vec3d &target_hit,
                                 const openvdb::Vec3d &viewpoint,
                                 double target_yaw,
                                 const Eigen::Vector3d &plan_start_pos,
                                 double plan_start_yaw);
    double pathLengthWorld(const std::vector<openvdb::Vec3d> &path) const;
    openvdb::Vec3d samplePointAlongPath(const std::vector<openvdb::Vec3d> &path,
                                        double distance_along_path) const;
    std::vector<openvdb::Vec3d> trimPathFromPoint(const std::vector<openvdb::Vec3d> &path,
                                                  const openvdb::Vec3d &point) const;
    std::vector<openvdb::Vec3d> prefixPathToPoint(const std::vector<openvdb::Vec3d> &path,
                                                  const openvdb::Vec3d &point) const;
    // Arm stitch monitoring using the explicit (trajectory_stamp, anchor_time)
    // handle returned by the publish path. The path tracker republishes the
    // same stamp with each TrackingPointState, so execution progress is read
    // directly from the tracker's time_from_start instead of geometric
    // projection.
    // Both arguments must be valid (stamp nonzero && anchor_time >= 0);
    // otherwise the call is a no-op (monitoring left disarmed) and a WARN
    // is logged.
    void activateStitchMonitoring(const builtin_interfaces::msg::Time &trajectory_stamp,
                                  double anchor_time);
    void deactivateStitchMonitoring();
    bool startup_escape_completed_ = false;
    bool startup_escape_traj_active_ = false;
    openvdb::Vec3d startup_escape_endpoint_{0, 0, 0};
};
