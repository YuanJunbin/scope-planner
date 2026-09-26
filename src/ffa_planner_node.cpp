// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/ffa_planner_node.hpp"
#include <vdb_edt/ray_trace_util.h>
#include "../include/vis_tools.hpp"
#include "../include/visibility_cell_builder.hpp"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <type_traits>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/create_timer_ros.h>



namespace
{
Eigen::Vector3d worldDirection(const openvdb::Vec3d &from,
                               const openvdb::Vec3d &to)
{
    Eigen::Vector3d dir(to.x() - from.x(), to.y() - from.y(), to.z() - from.z());
    const double norm = dir.norm();
    if (norm > 1.0e-6)
    {
        return (dir / norm).eval();
    }
    return Eigen::Vector3d::Zero();
}

traj_optimizer::BoundaryState makeBoundaryState(const openvdb::Vec3d &pos,
                                                const Eigen::Vector3d &vel = Eigen::Vector3d::Zero(),
                                                const Eigen::Vector3d &acc = Eigen::Vector3d::Zero())
{
    traj_optimizer::BoundaryState out;
    out.pos = Eigen::Vector3d(pos.x(), pos.y(), pos.z());
    out.vel = vel;
    out.acc = acc;
    return out;
}

double durationToSeconds(const builtin_interfaces::msg::Duration &duration)
{
    return static_cast<double>(duration.sec) +
           static_cast<double>(duration.nanosec) * 1.0e-9;
}

bool stampIsZero(const builtin_interfaces::msg::Time &stamp)
{
    return stamp.sec == 0 && stamp.nanosec == 0;
}

bool stampsEqual(const builtin_interfaces::msg::Time &a,
                 const builtin_interfaces::msg::Time &b)
{
    return a.sec == b.sec && a.nanosec == b.nanosec;
}

} // namespace

// Status -> String helpers

const char *FFAPlannerNode::globalStatusStr(GlobalStatus s)
{
    switch (s)
    {
    case GlobalStatus::IDLE:
        return "GLOBAL_IDLE";
    case GlobalStatus::PLANNING:
        return "GLOBAL_PLANNING";
    case GlobalStatus::READY:
        return "GLOBAL_READY";
    }
    return "UNKNOWN";
}

const char *FFAPlannerNode::localStatusStr(LocalStatus s)
{
    switch (s)
    {
    case LocalStatus::IDLE:
        return "LOCAL_IDLE";
    case LocalStatus::NAVIGATING:
        return "LOCAL_NAVIGATING";
    case LocalStatus::RECOVERY:
        return "LOCAL_RECOVERY";
    }
    return "UNKNOWN";
}

const char *FFAPlannerNode::subgoalTypeStr(SubgoalType t)
{
    switch (t)
    {
    case SubgoalType::ROOT_GOAL:
        return "ROOT_GOAL";
    case SubgoalType::FOV_CLEAR:
        return "FOV_CLEAR";
    case SubgoalType::OBSERVE_UNKNOWN:
        return "OBSERVE_UNKNOWN";
    }
    return "UNKNOWN";
}

// Construction & Destruction

FFAPlannerNode::FFAPlannerNode()
{
    node_handle_ = std::make_shared<rclcpp::Node>("ffa_plan_node");
    initialize();
}

FFAPlannerNode::~FFAPlannerNode()
{
    shutdown_.store(true);
    {
        std::lock_guard<std::mutex> lk(global_cv_mtx_);
    }
    global_cv_.notify_all();
    if (global_thread_.joinable())
    {
        global_thread_.join();
    }
}

void FFAPlannerNode::initialize()
{
    // TF
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node_handle_->get_clock());
    tf_buffer_->setCreateTimerInterface(
        std::make_shared<tf2_ros::CreateTimerROS>(
            node_handle_->get_node_base_interface(),
            node_handle_->get_node_timers_interface()));
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, node_handle_);

    // VDB Map
    map_manager_ = std::make_shared<VDBMap>(node_handle_, tf_buffer_, tf_listener_);
    RCLCPP_INFO(node_handle_->get_logger(), "Initialized VDB grids");

    // Parameters (must come before planner init)
    setup_parameters();


    traj_optimizer_ = std::make_unique<traj_optimizer::TrajOptimizer>();
    if (!traj_optimizer_->initialize(node_handle_, map_manager_))
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "Trajectory optimizer failed to initialize; debug visualization disabled");
        traj_optimizer_.reset();
    }

    // A* Planners
    AstarParams gp;
    gp.allocate_num = global_astar_allocate_num_;
    gp.max_search_time = global_astar_max_search_time_;
    gp.lambda_heu = global_astar_lambda_heu_;
    global_astar_.initialize(map_manager_, gp);
    subgoal_guidance_astar_.initialize(map_manager_, gp);
    // The guidance A*s run with an unlimited budget by design (exhaustion of
    // G_opt is the machine-checkable infeasibility certificate), so a sealed
    // target means a multi-second flood: report progress + a decimated
    // debug-tree snapshot every second instead of going silent.
    global_astar_.setProgressCallback(
        [this](double elapsed_s, int used_nodes, size_t open_size)
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Global] guidance A* still searching: %.1f s, closed %d, open %zu",
                        elapsed_s, used_nodes, open_size);
            auto tree_mk = global_astar_.getDebugTreeMarker(world_frame_id_, 60000);
            tree_mk.header.stamp = node_handle_->now();
            publishDebugTreeVis(tree_mk, DebugTreeKind::GUIDANCE, VisLayer::ROOT);
        },
        1.0);
    subgoal_guidance_astar_.setProgressCallback(
        [this](double elapsed_s, int used_nodes, size_t open_size)
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Subgoal] guidance A* still searching: %.1f s, closed %d, open %zu",
                        elapsed_s, used_nodes, open_size);
            auto tree_mk = subgoal_guidance_astar_.getDebugTreeMarker(world_frame_id_, 60000);
            tree_mk.header.stamp = node_handle_->now();
            publishDebugTreeVis(tree_mk, DebugTreeKind::GUIDANCE, VisLayer::SUBGOAL);
        },
        1.0);

    AstarParams lfp;
    lfp.allocate_num = fov_astar_allocate_num_;
    lfp.max_search_time = fov_astar_max_search_time_;
    lfp.lambda_heu = fov_astar_lambda_heu_;
    fov_astar_.initialize(map_manager_, lfp);
    fov_astar_.setFovConfig(fov_cfg_);
    fov_astar_.setClearProfile(FovClearProfileKind::STANDARD, fov_standard_profile_);
    fov_astar_.setClearProfile(FovClearProfileKind::TIGHT, fov_tight_profile_);
    fov_astar_.setActiveClearProfile(FovClearProfileKind::STANDARD);
    fov_astar_.setAnchorBias(anchor_bias_away_weight_, anchor_bias_max_multiplier_);

    AstarParams obp;
    obp.allocate_num = observe_astar_allocate_num_;
    obp.max_search_time = observe_astar_max_search_time_;
    obp.lambda_heu = observe_astar_lambda_heu_;
    observe_astar_.initialize(map_manager_, obp);
    observe_astar_.setFovConfig(fov_cfg_);
    observe_astar_.setAnchorBias(anchor_bias_away_weight_, anchor_bias_max_multiplier_);
    observe_astar_.setOccShellSkipVoxels(observe_graze_skip_vox_);

    AstarParams cap;
    cap.allocate_num = connect_astar_allocate_num_;
    cap.max_search_time = connect_astar_max_search_time_;
    cap.lambda_heu = connect_astar_lambda_heu_;
    connect_astar_.initialize(map_manager_, cap);

    // Soft standoff from occ-inflated space on the geometry-shaping searches
    // (fov/observe/connect). Guidance A*s stay at 1.0: their result is
    // topological and the unlimited floods must stay cheap per node.
    fov_astar_.setClearanceSoftCost(clearance_soft_cost_multiplier_);
    observe_astar_.setClearanceSoftCost(clearance_soft_cost_multiplier_);
    connect_astar_.setClearanceSoftCost(clearance_soft_cost_multiplier_);

    occlusion_analyzer_.initialize(node_handle_, map_manager_, fov_cfg_, world_frame_id_);

    RCLCPP_INFO(node_handle_->get_logger(),
                "Global A*: pool=%d, time_limit=%.3fs | FOV A*: pool=%d, time_limit=%.3fs | "
                "Observe A*: pool=%d, time_limit=%.3fs | Connect A*: pool=%d, time_limit=%.3fs",
                global_astar_allocate_num_, global_astar_max_search_time_,
                fov_astar_allocate_num_, fov_astar_max_search_time_,
                observe_astar_allocate_num_, observe_astar_max_search_time_,
                connect_astar_allocate_num_, connect_astar_max_search_time_);

    // Callback Groups
    cbg_local_ = node_handle_->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cbg_subs_ = node_handle_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    // Subscriptions
    rclcpp::SubscriptionOptions sub_opt;
    sub_opt.callback_group = cbg_subs_;

    sub_tracking_point_ = node_handle_->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/tracking_point", rclcpp::QoS(10),
        std::bind(&FFAPlannerNode::tracking_point_callback, this, std::placeholders::_1),
        sub_opt);

    sub_tracking_point_state_ = node_handle_->create_subscription<pid_path_tracker::msg::TrackingPointState>(
        "/tracking_point_state", rclcpp::QoS(10),
        std::bind(&FFAPlannerNode::tracking_point_state_callback, this, std::placeholders::_1),
        sub_opt);

    rclcpp::QoS goal_qos(10);
    goal_qos.reliable();
    goal_qos.transient_local();
    sub_goal_pose_ = node_handle_->create_subscription<geometry_msgs::msg::Pose>(
        "/goal_point", goal_qos,
        std::bind(&FFAPlannerNode::goal_pose_callback, this, std::placeholders::_1),
        sub_opt);

    // Publishers
    pub_guidance_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/guidance_path_astar_vis", 10);
    pub_guidance_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/guidance_path_short_vis", 10);
    pub_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/astar_debug_tree", 10);
    pub_trajectory_ = node_handle_->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
        "/cmd_trajectory", 10);
    pub_safe_edge_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/safe_edge_vis", 10);
    pub_fov_viewpoint_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/fov_viewpoint_vis", 10);
    pub_fov_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/fov_path_astar_vis", 10);
    pub_fov_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/fov_path_short_vis", 10);
    pub_fov_candidate_set_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/fov_candidate_set_vis", 10);
    pub_fov_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/fov_debug_tree", 10);
    pub_observe_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/observe_debug_tree", 10);
    pub_observe_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/observe_path_astar_vis", 10);
    pub_observe_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/observe_path_short_vis", 10);
    pub_observe_candidate_set_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/observe_candidate_set_vis", 10);
    pub_observe_frustum_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/observe_frustum_vis", 10);
    pub_connect_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/connect_debug_tree", 10);
    pub_connect_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/connect_path_astar_vis", 10);
    pub_connect_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/connect_path_short_vis", 10);
    pub_fov_frustum_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/fov_frustum_vis", 10);
    pub_subgoal_guidance_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_guidance_path_astar_vis", 10);
    pub_subgoal_guidance_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_guidance_path_short_vis", 10);
    pub_subgoal_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/subgoal_guidance_debug_tree", 10);
    pub_subgoal_safe_edge_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_safe_edge_vis", 10);
    pub_subgoal_fov_viewpoint_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_fov_viewpoint_vis", 10);
    pub_subgoal_fov_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_fov_path_astar_vis", 10);
    pub_subgoal_fov_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_fov_path_short_vis", 10);
    pub_subgoal_fov_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/subgoal_fov_debug_tree", 10);
    pub_subgoal_observe_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/subgoal_observe_debug_tree", 10);
    pub_subgoal_observe_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_observe_path_astar_vis", 10);
    pub_subgoal_observe_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_observe_path_short_vis", 10);
    pub_subgoal_connect_debug_tree_ = node_handle_->create_publisher<visualization_msgs::msg::Marker>(
        "/subgoal_connect_debug_tree", 10);
    pub_subgoal_connect_path_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_connect_path_astar_vis", 10);
    pub_subgoal_connect_short_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_connect_path_short_vis", 10);
    pub_subgoal_fov_frustum_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/subgoal_fov_frustum_vis", 10);
    pub_traj_optimizer_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/traj_optimizer_path_vis", 10);
    pub_traj_sfc_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/traj_optimizer_sfc_vis", 10);
    pub_lookahead_vis_ = node_handle_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "/lookahead_point_vis", 10);

    // Local timer
    local_timer_ = rclcpp::create_timer(
        node_handle_, node_handle_->get_clock(),
        rclcpp::Duration::from_seconds(local_timer_period_),
        std::bind(&FFAPlannerNode::localTimerCallback, this),
        cbg_local_);

    // Goal-reach status heartbeat (0 in-progress / 1 reached / 2 failed), 2 s.
    pub_goal_reach_status_ = node_handle_->create_publisher<std_msgs::msg::Int8>(
        "/goal_reach_status", 10);
    goal_reach_status_timer_ = rclcpp::create_timer(
        node_handle_, node_handle_->get_clock(),
        rclcpp::Duration::from_seconds(2.0),
        std::bind(&FFAPlannerNode::goalReachStatusTimerCallback, this),
        cbg_local_);

    // Global planner thread (sleeps until woken by goal or replan request)
    global_thread_ = std::thread(&FFAPlannerNode::globalPlannerLoop, this);

    RCLCPP_INFO(node_handle_->get_logger(),
                "FFAPlannerNode ready.  Global: event-driven thread  |  Local: %.1f Hz",
                1.0 / local_timer_period_);
}

void FFAPlannerNode::setup_parameters()
{
    auto log = node_handle_->get_logger();

    auto to_str = [](const auto &v) -> std::string
    {
        using T = std::decay_t<decltype(v)>;
        if constexpr (std::is_same_v<T, std::string>)
        {
            return v;
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            return v ? "true" : "false";
        }
        else
        {
            return std::to_string(v);
        }
    };

    auto param_set = [&](std::string_view name, const auto &def, auto &out)
    {
        using T = std::decay_t<decltype(out)>;
        const std::string key{name};
        T val{};

        if (!node_handle_->has_parameter(key))
        {
            val = node_handle_->template declare_parameter<T>(key, static_cast<T>(def));
        }
        else
        {
            node_handle_->get_parameter(key, val);
        }
        out = val;

        if (val == static_cast<T>(def))
        {
            RCLCPP_INFO(log, "  %s = %s (default)", key.c_str(), to_str(val).c_str());
        }
        else
        {
            RCLCPP_INFO(log, "  %s = %s (config)", key.c_str(), to_str(val).c_str());
        }
    };

    RCLCPP_INFO(log, "Loading parameters");

    param_set("world_frame_id", "map", world_frame_id_);
    param_set("robot_frame_id", "base_link", robot_frame_id_);
    param_set("vox_size", 0.2, voxel_size_);
    param_set("vis_debug_tree", false, vis_debug_tree_);
    param_set("vis_fov_debug_tree", false, vis_fov_debug_tree_);
    param_set("vis_traj_optimizer", true, vis_traj_optimizer_);

    param_set("local_timer_period", 0.5, local_timer_period_);

    param_set("global_astar_allocate_num", 5000000, global_astar_allocate_num_);
    param_set("global_astar_max_search_time", 0.0, global_astar_max_search_time_);
    param_set("global_astar_lambda_heu", 1.0, global_astar_lambda_heu_);
    param_set("clearance_soft_cost_multiplier", 1.0, clearance_soft_cost_multiplier_);

    param_set("body_sensor_x", 0.1, fov_cfg_.body_x);
    param_set("body_sensor_y", 0.0, fov_cfg_.body_y);
    param_set("body_sensor_z", 0.0, fov_cfg_.body_z);
    param_set("body_sensor_roll", 0.0, fov_cfg_.body_roll);
    param_set("body_sensor_pitch", 0.0, fov_cfg_.body_pitch);
    param_set("body_sensor_yaw", 0.0, fov_cfg_.body_yaw);
    param_set("fov_theta_u", 0.3, fov_cfg_.theta_u);
    param_set("fov_theta_d", -0.3, fov_cfg_.theta_d);
    param_set("fov_min_range", 0.5, fov_cfg_.fov_min_range);
    param_set("observe_min_range", 0.5, fov_cfg_.observe_min_range);
    param_set("soft_frontier_cost_multiplier", 3.0, fov_cfg_.soft_frontier_cost_multiplier);
    param_set("anchor_bias_away_weight", 0.1, anchor_bias_away_weight_);
    param_set("anchor_bias_max_multiplier", 1.1, anchor_bias_max_multiplier_);

    double sensor_range = 10.0, safe_rxy = 0.6, safe_rz = 0.2;
    double fov_range = -1.0;
    param_set("sensor_range", 10.0, sensor_range);
    param_set("fov_range", -1.0, fov_range);
    param_set("safe_robot_radius_xy", 0.6, safe_rxy);
    param_set("safe_robot_height_z", 0.2, safe_rz);
    param_set("tight_additional_radius_xy", 0.0, tight_additional_radius_xy_);
    param_set("tight_additional_height_z", 0.0, tight_additional_height_z_);
    fov_cfg_.range = (fov_range > 0.0) ? fov_range : sensor_range;
    fov_standard_profile_.inflate_rxy = safe_rxy;
    fov_standard_profile_.inflate_rz = safe_rz;
    fov_tight_profile_.inflate_rxy = safe_rxy + std::max(0.0, tight_additional_radius_xy_);
    fov_tight_profile_.inflate_rz = safe_rz + std::max(0.0, tight_additional_height_z_);


    param_set("fov_astar_allocate_num", 500000, fov_astar_allocate_num_);
    param_set("fov_astar_max_search_time", 0.1, fov_astar_max_search_time_);
    param_set("fov_astar_lambda_heu", 2.0, fov_astar_lambda_heu_);

    param_set("observe_astar_allocate_num", 500000, observe_astar_allocate_num_);
    param_set("observe_astar_max_search_time", 0.2, observe_astar_max_search_time_);
    param_set("observe_astar_lambda_heu", 2.0, observe_astar_lambda_heu_);

    param_set("connect_astar_allocate_num", 500000, connect_astar_allocate_num_);
    param_set("connect_astar_max_search_time", 0.2, connect_astar_max_search_time_);
    param_set("connect_astar_lambda_heu", 1.0, connect_astar_lambda_heu_);
    param_set("preview_connect_max_search_time", 0.05, preview_connect_max_search_time_);

    param_set("interpolate_step", 0.2, interpolate_step_);
    param_set("cruise_speed", 1.0, cruise_speed_);
    // FFA-search ablation: bypass the SFC trajectory optimizer entirely and
    // publish the searched waypoint path via the constant-cruise-speed linear
    // interpolator (the optimizer's own fallback) — the same densify style the
    // baselines use. Search/FSM/yaw semantics are untouched.
    param_set("ffa_disable_traj_opt", false, disable_traj_opt_);
    param_set("viewpoint_reach_tol", 0.5, viewpoint_reach_tol_);
    param_set("fov_stitch_lookahead_dist", 1.5, fov_stitch_lookahead_dist_);
    param_set("visibility_cell_roi_xy", 1.5, visibility_cell_roi_xy_);
    param_set("visibility_cell_roi_z", 0.8, visibility_cell_roi_z_);
    param_set("visibility_cell_erode_iters", 1, visibility_cell_erode_iters_);
    param_set("visibility_cell_min_radius", 0.1, visibility_cell_min_radius_);
    param_set("visibility_cell_shrink_m", 0.1, visibility_cell_shrink_m_);
    param_set("visibility_cell_max_carve", 3, visibility_cell_max_carve_);
    param_set("visibility_cell_fixed_yaw", true, visibility_cell_fixed_yaw_);
    // Azimuth half-FOV bound used only by fixed-yaw validation; <=0 falls
    // back to theta_u (conservative when horizontal FOV >= vertical FOV).
    param_set("fov_theta_h", -1.0, fov_theta_h_);

    // Observe-phase failsafe (arrived at observe viewpoint but the locked
    // unknown was not resolved). See handleObserveDwell.
    param_set("observe_dwell_map_updates", 3, observe_dwell_map_updates_);
    param_set("observe_arrival_max_ticks", 10, observe_arrival_max_ticks_);
    param_set("observe_resume_budget", 0.2, observe_resume_budget_);
    param_set("observe_graze_penalty_vox", 5.0, observe_graze_penalty_vox_);
    param_set("observe_graze_skip_vox", 2, observe_graze_skip_vox_);

    param_set("lookahead_buffer_time_s", 0.2, lookahead_buffer_time_s_);
    // true = anchor lookahead to tracker-reported progress (real-robot fix,
    // fc93f2c); false = benchmark-era wall-clock schedule.
    param_set("lookahead_progress_driven", true, lookahead_progress_driven_);
    param_set("tp_divergence_threshold_m", 0.5, tp_divergence_threshold_m_);
}

rclcpp::Node::SharedPtr FFAPlannerNode::get_node_handle() const
{
    return node_handle_;
}

// TF Helper

bool FFAPlannerNode::getRobotPose(Eigen::Vector3d &pos_out, double &yaw_out) const
{
    try
    {
        auto ts = tf_buffer_->lookupTransform(world_frame_id_, robot_frame_id_, rclcpp::Time(0));

        pos_out.x() = ts.transform.translation.x;
        pos_out.y() = ts.transform.translation.y;
        pos_out.z() = ts.transform.translation.z;

        tf2::Quaternion q;
        tf2::fromMsg(ts.transform.rotation, q);
        q.normalize();
        double roll, pitch;
        tf2::Matrix3x3(q).getRPY(roll, pitch, yaw_out);

        return true;
    }
    catch (tf2::TransformException &ex)
    {
        RCLCPP_ERROR_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 3000,
                              "Robot TF not available: %s", ex.what());
        return false;
    }
}

// Preferred planning start: lookahead if available, else current TF.
// We keep the logic deliberately simple (no staleness / divergence checks)
// so it's easy to reason about. The only fallback case in practice is the
// very first plan after startup, before any trajectory has been published.
bool FFAPlannerNode::getPlanStartState(Eigen::Vector3d &pos, double &yaw) const
{
    const LookaheadState s = getLookaheadState();
    if (s.valid)
    {
        pos = s.pos;
        // Wrap to (-pi, pi]. The lookahead cache stores yaw as a monotonically
        // unwrapped, possibly drifting accumulator (good for sampling
        // continuity inside one trajectory). Leaking that drift into a new
        // plan's start-yaw boundary lets gcopter's unwrapAngleNear(start,
        // target) pick a multi-turn equivalence class for `target` — the
        // optimizer then commands an extra ~2pi rotation. Normalizing here
        // keeps every new plan's start in the same frame as freshly computed
        // target yaws (atan2 outputs).
        yaw = wrapToPi(s.yaw);
        return true;
    }

    RCLCPP_WARN_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 2000,
                         "[Lookahead] cache not yet valid, falling back to TF "
                         "(expected only on first plan after startup)");
    const bool ok = getRobotPose(pos, yaw);
    if (ok)
    {
        yaw = wrapToPi(yaw);
    }
    return ok;
}

// State Reset

void FFAPlannerNode::resetLocalPlanState()
{
    needs_escape_corridor_ = true;
    occlusion_analyzer_.reset();
    deactivateStitchMonitoring();
    startup_escape_completed_ = false;
    startup_escape_traj_active_ = false;
    startup_escape_endpoint_ = openvdb::Vec3d(0, 0, 0);
    // NOTE: we intentionally do NOT invalidate last_published_traj_ on new
    // goals. The robot is still physically tracking the old trajectory for
    // ~0.1-0.3s after replan is requested, so the cached lookahead point
    // (a short buffer ahead of now) remains a valid, absolute-safe start
    // for the next search. needs_escape_corridor_ still flips true so the
    // first plan after each reset allows frontier-traversal near start.
    resetSubgoalChain();
    resetRefutationState();
}

void FFAPlannerNode::resetRefutationState()
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    if (root_subgoal_)
    {
        clearRefutationStateRecursive(*root_subgoal_);
    }
}

void FFAPlannerNode::clearRefutationStateRecursive(SubgoalNode &node)
{
    node.rejected_viewpoint_voxels.clear();
    node.exhausted_hitpoints.clear();
    node.pending_no_good_by_hitpoint.clear();
    if (node.child)
    {
        clearRefutationStateRecursive(*node.child);
    }
}

void FFAPlannerNode::resetSubgoalChain()
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    root_subgoal_.reset();
    next_subgoal_id_ = 1;
    clearSubgoalVisualization();
}


FFAPlannerNode::SubgoalNode *FFAPlannerNode::ensureRootSubgoal()
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    if (!root_subgoal_)
    {
        root_subgoal_ = std::make_unique<SubgoalNode>();
        root_subgoal_->node_id = next_subgoal_id_++;
        root_subgoal_->type = SubgoalType::ROOT_GOAL;
        root_subgoal_->status = SubgoalStatus::ACTIVE;
    }

    return root_subgoal_.get();
}

void FFAPlannerNode::refreshRootSubgoalGoal()
{
    SubgoalNode *root = ensureRootSubgoal();
    std::lock_guard<std::mutex> lk(goal_mtx_);
    root->anchor_point = openvdb::Vec3d(curr_goal_xyz_.x(), curr_goal_xyz_.y(), curr_goal_xyz_.z());
    root->goal_pos = root->anchor_point;
    root->goal_yaw = curr_goal_yaw_;
}

bool FFAPlannerNode::refreshRootSubgoalGuidance()
{
    SubgoalNode *root = ensureRootSubgoal();

    std::shared_lock<std::shared_mutex> lk(guidance_path_mtx_);
    if (shared_guidance_path_short_.empty())
    {
        root->guidance_path_raw.clear();
        root->guidance_path_short.clear();
        return false;
    }

    updateNodeGuidance(*root, shared_guidance_path_, shared_guidance_path_short_);
    return true;
}

bool FFAPlannerNode::planGuidanceToSubgoal(SubgoalNode &node,
                                           const Eigen::Vector3d &robot_pos)
{
    openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
    openvdb::Coord start_ijk = openvdb::Coord::round(
        grid_tf->worldToIndex(openvdb::Vec3d(robot_pos.x(), robot_pos.y(), robot_pos.z())));
    // Subgoal guidance anchors at the frozen creation-time lookahead (shared
    // down the subtree), NOT the parent's a(h): a boundary-hugging origin
    // makes the child's first march step re-hit the same frontier pocket and
    // chain grandchildren (take-025 class). Falls back to the live plan
    // start when the frozen origin was invalidated by map updates.
    if (node.has_guidance_origin &&
        isStrictFreeVoxel(node.guidance_origin_ijk))
    {
        start_ijk = node.guidance_origin_ijk;
    }
    openvdb::Coord goal_ijk = openvdb::Coord::round(grid_tf->worldToIndex(node.goal_pos));

    auto blocked = buildGuidanceBlockedSet(node);
    blocked.erase(start_ijk);
    subgoal_guidance_astar_.setExtraBlockedVoxels(blocked);
    subgoal_guidance_astar_.reset();
    const int sg_result = subgoal_guidance_astar_.search(start_ijk, goal_ijk);
    if (sg_result != Astar::PATH_FOUND)
    {
        subgoal_guidance_astar_.clearExtraBlockedVoxels();
        node.guidance_path_raw.clear();
        node.guidance_path_short.clear();
        return false;
    }

    std::vector<openvdb::Vec3d> raw = subgoal_guidance_astar_.getPathAstar();
    std::vector<openvdb::Vec3d> shortened;
    subgoal_guidance_astar_.pathShorten(shortened);
    subgoal_guidance_astar_.clearExtraBlockedVoxels();
    if (shortened.empty())
    {
        shortened = raw;
    }

    updateNodeGuidance(node, raw, shortened);
    node.has_guidance_progress_floor = false;
    node.guidance_progress_floor = openvdb::Vec3d(0, 0, 0);
    clearSubgoalVisualization();
    publishGuidancePathVis(node.guidance_path_raw, node.guidance_path_short, VisLayer::SUBGOAL);
    if (vis_debug_tree_)
    {
        auto tree_mk = subgoal_guidance_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::GUIDANCE, VisLayer::SUBGOAL);
    }
    return true;
}

void FFAPlannerNode::updateNodeGuidance(SubgoalNode &node,
                                        const std::vector<openvdb::Vec3d> &raw_path,
                                        const std::vector<openvdb::Vec3d> &short_path)
{
    node.guidance_path_raw = raw_path;
    node.guidance_path_short = short_path;
}

void FFAPlannerNode::clearNodeExecutionState(SubgoalNode &node)
{
    node.has_active_plan = false;
    node.has_preview_plan = false;
    node.active_viewpoint = openvdb::Vec3d(0, 0, 0);
    node.active_anchor_seed = openvdb::Vec3d(0, 0, 0);
    node.active_plan_hit_point = openvdb::Vec3d(0, 0, 0);
    clearActivePlanHitSearchStart(node);
    node.active_target_yaw = 0.0;
    node.goal_traj_published = false;
    node.status = SubgoalStatus::ACTIVE;
    node.has_receding_preview_state = false;
    node.preview_next_anchor_seed = openvdb::Vec3d(0, 0, 0);
    node.preview_next_anchor_yaw = 0.0;
    node.active_anchor_progress = PreviewProgressWindow{};
    node.next_anchor_progress = PreviewProgressWindow{};
    node.last_anchor_hitpoint = HitpointKey{};
    node.has_last_anchor_hitpoint = false;
    releaseObserveLock(node);
    deactivateStitchMonitoring();
}

void FFAPlannerNode::setActivePlanHitFromSafeEdge(SubgoalNode &node,
                                                  const SafeEdgeHitResult &hit)
{
    node.active_plan_hit_point = hit.hit_pt;
    // Certification is monotone (observation never regresses): a strict-free
    // search start, once established for this context, stays valid. A hit
    // marched from the trimmed guidance suffix can legitimately carry no
    // strict-free prefix (hit at idx 0 with the progress floor advanced into
    // the contested tail); in that case KEEP the previously certified start
    // instead of clearing it — clearing would strand the observe phase with
    // no search seed (defer livelock).
    if (hit.has_search_start)
    {
        node.active_plan_has_search_start = true;
        node.active_plan_search_start_ijk = hit.search_start_ijk;
        node.active_plan_search_start_world = hit.search_start_world;
        node.has_certified_guidance_start = true;
        node.certified_guidance_start_ijk = hit.search_start_ijk;
        node.certified_guidance_start_world = hit.search_start_world;
    }
}

void FFAPlannerNode::clearActivePlanHitSearchStart(SubgoalNode &node)
{
    node.active_plan_has_search_start = false;
    node.active_plan_search_start_ijk = openvdb::Coord(0, 0, 0);
    node.active_plan_search_start_world = openvdb::Vec3d(0, 0, 0);
}

FFAPlannerNode::SubgoalNode *FFAPlannerNode::createOrUpdateChildSubgoal(
    SubgoalNode &parent,
    SubgoalType type,
    const openvdb::Vec3d &anchor_point,
    const openvdb::Vec3d &goal_pos,
    double goal_yaw)
{
    clearNodeExecutionState(parent);

    if (!parent.child || parent.child->type != type)
    {
        parent.child = std::make_unique<SubgoalNode>();
        parent.child->node_id = next_subgoal_id_++;
        parent.child->type = type;
    }

    SubgoalNode *child = parent.child.get();
    child->status = SubgoalStatus::ACTIVE;
    child->anchor_point = anchor_point;
    child->goal_pos = goal_pos;
    child->goal_yaw = goal_yaw;
    child->ancestor_hitpoints = parent.ancestor_hitpoints;
    child->ancestor_hitpoints.insert(canonicalizeHitpoint(anchor_point));
    {
        std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
        child->rejected_viewpoint_voxels = parent.rejected_viewpoint_voxels;
        child->exhausted_hitpoints = parent.exhausted_hitpoints;
        child->pending_no_good_by_hitpoint.clear();
    }
    clearNodeExecutionState(*child);
    child->has_certified_guidance_start = false;
    child->certified_guidance_start_ijk = openvdb::Coord(0, 0, 0);
    child->certified_guidance_start_world = openvdb::Vec3d(0, 0, 0);
    child->has_guidance_progress_floor = false;
    child->guidance_progress_floor = openvdb::Vec3d(0, 0, 0);
    child->child.reset();
    return child;
}

void FFAPlannerNode::pruneSubgoalChildren(SubgoalNode &node)
{
    if (node.child)
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Subgoal] Pruning child chain under node %llu (%s)",
                    static_cast<unsigned long long>(node.node_id), subgoalTypeStr(node.type));
        clearSubgoalVisualization();
    }
    node.child.reset();
}

bool FFAPlannerNode::pruneSubgoalNodeById(SubgoalNode &node, uint64_t node_id)
{
    if (!node.child)
    {
        return false;
    }

    if (node.child->node_id == node_id)
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Subgoal] Removing node %llu under parent %llu (%s -> %s)",
                    static_cast<unsigned long long>(node_id),
                    static_cast<unsigned long long>(node.node_id),
                    subgoalTypeStr(node.type),
                    subgoalTypeStr(node.child->type));
        clearSubgoalVisualization();
        node.child.reset();
        return true;
    }

    return pruneSubgoalNodeById(*node.child, node_id);
}

FFAPlannerNode::SubgoalNode *
FFAPlannerNode::findParentSubgoalByChildId(SubgoalNode &node, uint64_t child_id)
{
    if (!node.child)
    {
        return nullptr;
    }

    if (node.child->node_id == child_id)
    {
        return &node;
    }

    return findParentSubgoalByChildId(*node.child, child_id);
}

FFAPlannerNode::SubgoalNode *FFAPlannerNode::getDeepestActiveSubgoal()
{
    SubgoalNode *node = root_subgoal_.get();
    SubgoalNode *leaf = node;

    while (node)
    {
        if (node->status == SubgoalStatus::ACTIVE ||
            node->status == SubgoalStatus::EXECUTING)
        {
            leaf = node;
        }
        node = node->child.get();
    }

    return leaf;
}

// Subscription Callbacks

void FFAPlannerNode::tracking_point_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
    std::unique_lock<std::shared_mutex> lk(tp_mtx_);
    has_tracking_point_ = true;
    current_tracking_point_ = *msg;
}

void FFAPlannerNode::tracking_point_state_callback(
    const pid_path_tracker::msg::TrackingPointState::SharedPtr msg)
{
    const double tp_time = durationToSeconds(msg->time_from_start);
    {
        std::lock_guard<std::mutex> slk(stitch_monitor_mtx_);
        if (stitch_monitoring_active_ && !stitch_vp1_reached_ &&
            stampsEqual(msg->trajectory_stamp, stitch_monitored_trajectory_stamp_))
        {
            if (tp_time + 1.0e-3 >= stitch_monitored_anchor_time_)
            {
                stitch_vp1_reached_ = true;
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Stitch] Tracking point reached vp1 milestone at t=%.2fs on traj stamp %d.%09u",
                            tp_time,
                            msg->trajectory_stamp.sec,
                            msg->trajectory_stamp.nanosec);
            }
        }
    }

    const Eigen::Vector3d tp_pos(msg->pose.pose.position.x,
                                 msg->pose.pose.position.y,
                                 msg->pose.pose.position.z);
    refreshLookaheadFromTracking(node_handle_->now(), tp_pos, tp_time,
                                 msg->trajectory_stamp);
}

void FFAPlannerNode::activateStitchMonitoring(const builtin_interfaces::msg::Time &trajectory_stamp,
                                              double anchor_time)
{
    if (stampIsZero(trajectory_stamp) || !std::isfinite(anchor_time) || anchor_time < 0.0)
    {
        std::lock_guard<std::mutex> lk(stitch_monitor_mtx_);
        stitch_monitoring_active_ = false;
        stitch_monitored_trajectory_stamp_ = builtin_interfaces::msg::Time{};
        stitch_monitored_anchor_time_ = 0.0;
        stitch_vp1_reached_ = false;
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Stitch] refusing to arm vp1 monitoring: invalid handle "
                    "(trajectory_stamp=%d.%09u anchor_time=%.3f)",
                    trajectory_stamp.sec, trajectory_stamp.nanosec, anchor_time);
        return;
    }

    std::lock_guard<std::mutex> lk(stitch_monitor_mtx_);
    stitch_monitoring_active_ = true;
    stitch_monitored_trajectory_stamp_ = trajectory_stamp;
    stitch_monitored_anchor_time_ = anchor_time;
    stitch_vp1_reached_ = anchor_time <= 1.0e-3;
    RCLCPP_INFO(node_handle_->get_logger(),
                "[Stitch] Armed vp1 monitoring on traj stamp %d.%09u at t=%.2fs%s",
                trajectory_stamp.sec, trajectory_stamp.nanosec, anchor_time,
                stitch_vp1_reached_ ? " (already reached)" : "");
}

void FFAPlannerNode::deactivateStitchMonitoring()
{
    std::lock_guard<std::mutex> lk(stitch_monitor_mtx_);
    stitch_monitoring_active_ = false;
    stitch_monitored_trajectory_stamp_ = builtin_interfaces::msg::Time{};
    stitch_monitored_anchor_time_ = 0.0;
    stitch_vp1_reached_ = false;
}

void FFAPlannerNode::goal_pose_callback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
    Eigen::Vector3d goal_pos(msg->position.x, msg->position.y, msg->position.z);
    double goal_yaw = 0.0;
    {
        std::lock_guard<std::mutex> lk(goal_mtx_);
        curr_goal_xyz_ = goal_pos;

        tf2::Quaternion q;
        tf2::fromMsg(msg->orientation, q);
        q.normalize();
        double roll, pitch;
        tf2::Matrix3x3(q).getRPY(roll, pitch, goal_yaw);
        curr_goal_yaw_ = goal_yaw;
    }


    has_new_goal_.store(true);
    goal_reach_status_.store(GoalReachStatus::IN_PROGRESS);
    resetLocalPlanState();

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Goal] Received: [%.2f, %.2f, %.2f]  yaw=%.2f",
                msg->position.x, msg->position.y, msg->position.z, curr_goal_yaw_);

    wakeGlobalPlanner();
}

// Wake the global planner thread from its CV wait

void FFAPlannerNode::wakeGlobalPlanner()
{
    {
        std::lock_guard<std::mutex> lk(global_cv_mtx_);
    }
    global_cv_.notify_one();
}

// Fixed-interval heartbeat of the current goal-reach state on /goal_reach_status.
void FFAPlannerNode::goalReachStatusTimerCallback()
{
    std_msgs::msg::Int8 msg;
    msg.data = static_cast<int8_t>(goal_reach_status_.load());
    pub_goal_reach_status_->publish(msg);
}

// Global Planner Thread
// Sleeps until woken by new goal or replan request, runs A*, then goes back to sleep.

void FFAPlannerNode::globalPlannerLoop()
{
    RCLCPP_INFO(node_handle_->get_logger(), "[Global] Thread started, waiting for events");

    while (!shutdown_.load())
    {
        // Sleep until signalled
        {
            std::unique_lock<std::mutex> lk(global_cv_mtx_);
            global_cv_.wait(lk, [this]()
                            { return shutdown_.load() || has_new_goal_.load() || need_global_replan_.load(); });
        }

        if (shutdown_.load())
        {
            break;
        }

        // Consume flags
        bool new_goal = has_new_goal_.exchange(false);
        bool replan = need_global_replan_.exchange(false);
        std::optional<openvdb::Coord> start_override;
        {
            std::lock_guard<std::mutex> lk(global_cv_mtx_);
            if (!new_goal && replan)
            {
                start_override = pending_global_replan_start_override_;
            }
            pending_global_replan_start_override_.reset();
        }

        if (!new_goal && !replan)
        {
            continue;
        }

        GlobalStatus prev = global_status_.load();
        global_status_.store(GlobalStatus::PLANNING);

        if (new_goal)
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Global] %s -> PLANNING (new goal)", globalStatusStr(prev));
        }
        else
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Global] %s -> PLANNING (local requested replan)", globalStatusStr(prev));
        }

        // Plan start pose — prefers the cached lookahead point (a short
        // buffer ahead on the currently executing traj) so new plans stitch
        // smoothly onto what the controller is already following. Falls back
        // to TF when no trajectory has been published yet. Retries on
        // transient failure instead of dropping the task (the flags have
        // already been consumed above).
        Eigen::Vector3d start_pos;
        double start_yaw = 0.0;
        std::optional<openvdb::Vec3d> cached_root_guidance_start;
        std::optional<openvdb::Coord> cached_root_guidance_start_ijk;
        if (!new_goal)
        {
            std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
            if (root_subgoal_ &&
                root_subgoal_->has_certified_guidance_start)
            {
                cached_root_guidance_start_ijk =
                    root_subgoal_->certified_guidance_start_ijk;
                cached_root_guidance_start =
                    root_subgoal_->certified_guidance_start_world;
            }
        }
        if (start_override)
        {
            openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
            const openvdb::Vec3d start_world = grid_tf->indexToWorld(*start_override);
            start_pos = Eigen::Vector3d(start_world.x(), start_world.y(), start_world.z());
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Global] Using virtual guidance start [%d,%d,%d] -> [%.2f, %.2f, %.2f]",
                        start_override->x(), start_override->y(), start_override->z(),
                        start_pos.x(), start_pos.y(), start_pos.z());
        }
        else if (cached_root_guidance_start &&
                 cached_root_guidance_start_ijk &&
                 isStrictFreeVoxel(*cached_root_guidance_start_ijk))
        {
            const openvdb::Vec3d start_world = *cached_root_guidance_start;
            start_pos = Eigen::Vector3d(start_world.x(), start_world.y(), start_world.z());
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Global] Using cached certified a(h) as root replan start "
                        "[%.2f, %.2f, %.2f]",
                        start_pos.x(), start_pos.y(), start_pos.z());
        }
        else if (!getPlanStartState(start_pos, start_yaw))
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Global] plan-start pose not ready, will retry...");
            if (new_goal)
            {
                has_new_goal_.store(true);
            }
            if (replan)
            {
                need_global_replan_.store(true);
            }
            global_status_.store(GlobalStatus::IDLE);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        Eigen::Vector3d goal;
        {
            std::lock_guard<std::mutex> lk(goal_mtx_);
            goal = curr_goal_xyz_;
        }

        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Global] Planning  start=[%.2f, %.2f, %.2f]  goal=[%.2f, %.2f, %.2f]",
                    start_pos.x(), start_pos.y(), start_pos.z(),
                    goal.x(), goal.y(), goal.z());

        openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
        openvdb::Vec3d s_w(start_pos.x(), start_pos.y(), start_pos.z());
        openvdb::Vec3d g_w(goal.x(), goal.y(), goal.z());
        openvdb::Coord s_ijk(openvdb::Coord::round(grid_tf->worldToIndex(s_w)));
        openvdb::Coord g_ijk(openvdb::Coord::round(grid_tf->worldToIndex(g_w)));

        // Goal pre-check: a goal voxel inside occupied inflation is unreachable
        // in G_opt by definition, and the guidance A* would otherwise flood the
        // (unbounded) unknown space until its budget runs out. Frontier
        // inflation and unknown are fine (optimistic target). No snapping:
        // fail plainly so the cause is visible.
        {
            int goal_inflate_val = 0;
            const bool goal_active =
                map_manager_->query_is_inflated_at_index(g_ijk, goal_inflate_val);
            const bool goal_in_occ =
                goal_active && (goal_inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
            if (goal_in_occ)
            {
                RCLCPP_ERROR(node_handle_->get_logger(),
                             "[Global] Goal [%.2f, %.2f, %.2f] (ijk %d,%d,%d) lies inside occupied "
                             "inflation; unreachable, returning to IDLE",
                             goal.x(), goal.y(), goal.z(), g_ijk.x(), g_ijk.y(), g_ijk.z());
                goal_reach_status_.store(GoalReachStatus::FAILED);
                global_status_.store(GlobalStatus::IDLE);
                continue;
            }
        }

        std::unordered_set<openvdb::Coord, CoordHash> blocked;
        {
            std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
            if (root_subgoal_)
            {
                for (const HitpointKey &hitpoint : root_subgoal_->exhausted_hitpoints)
                {
                    blocked.insert(hitpoint.coord);
                }
            }
        }
        blocked.erase(s_ijk);
        global_astar_.setExtraBlockedVoxels(blocked);
        global_astar_.reset();

        auto t0 = std::chrono::steady_clock::now();
        int result = global_astar_.search(s_ijk, g_ijk);
        double elapsed_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

        if (result == Astar::PATH_FOUND)
        {
            std::vector<openvdb::Vec3d> raw = global_astar_.getPathAstar();
            std::vector<openvdb::Vec3d> shortened;
            global_astar_.pathShorten(shortened);
            global_astar_.clearExtraBlockedVoxels();

            // Degenerate success: start cell == goal cell, so the shortened path
            // is empty and there is nothing to navigate. The robot is already at
            // the goal -> mark reached and go IDLE. Going READY here instead would
            // livelock the local planner flipping NAVIGATING<->IDLE forever on an
            // empty guidance path.
            if (shortened.empty())
            {
                {
                    std::unique_lock<std::shared_mutex> lk(guidance_path_mtx_);
                    shared_guidance_path_.clear();
                    shared_guidance_path_short_.clear();
                }
                goal_reach_status_.store(GoalReachStatus::REACHED);
                global_status_.store(GlobalStatus::IDLE);
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Global] Start already at goal (%.3f ms, A* %zu pts); marking reached",
                            elapsed_ms, raw.size());
                continue;
            }

            {
                std::unique_lock<std::shared_mutex> lk(guidance_path_mtx_);
                shared_guidance_path_ = raw;
                shared_guidance_path_short_ = shortened;
            }

            global_status_.store(GlobalStatus::READY);

            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Global] Planning took %.3f ms  (A* %zu pts, short %zu pts)",
                        elapsed_ms, raw.size(), shortened.size());

            publishGuidancePathVis(raw, shortened, VisLayer::ROOT);
            if (vis_debug_tree_)
            {
                auto tree_mk = global_astar_.getDebugTreeMarker(world_frame_id_);
                tree_mk.header.stamp = node_handle_->now();
                publishDebugTreeVis(tree_mk, DebugTreeKind::GUIDANCE, VisLayer::ROOT);
            }
        }
        else
        {
            global_astar_.clearExtraBlockedVoxels();
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Global] No path found (%.1f ms), returning to IDLE", elapsed_ms);
            // A goal we were actively pursuing is unreachable (e.g. occupied).
            // Fires on the initial plan AND on execution-time replans, since a
            // replan that can no longer find a path means the goal is lost.
            goal_reach_status_.store(GoalReachStatus::FAILED);
            global_status_.store(GlobalStatus::IDLE);
        }
    }

    RCLCPP_INFO(node_handle_->get_logger(), "[Global] Thread exiting");
}

// Local Timer
// IDLE -[guidance ready]-> NAVIGATING -[stuck/occluded]-> RECOVERY

void FFAPlannerNode::localTimerCallback()
{
    LocalStatus ls = local_status_.load();
    GlobalStatus gs = global_status_.load();

    switch (ls)
    {
    case LocalStatus::IDLE:
    {
        if (gs == GlobalStatus::READY)
        {
            refreshRootSubgoalGoal();
            refreshRootSubgoalGuidance();
            local_status_.store(LocalStatus::NAVIGATING);
            clearNodeExecutionState(*ensureRootSubgoal());
            RCLCPP_INFO(node_handle_->get_logger(), "[Local] IDLE -> NAVIGATING");
        }
        else
        {
            RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 3000,
                                 "[Local] IDLE, hovering  (Global: %s)", globalStatusStr(gs));
        }
        break;
    }

    case LocalStatus::NAVIGATING:
    {
        if (gs != GlobalStatus::READY)
        {
            local_status_.store(LocalStatus::IDLE);
            resetSubgoalChain();
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local] NAVIGATING -> IDLE (guidance path lost, Global: %s)",
                        globalStatusStr(gs));
            break;
        }

        refreshRootSubgoalGoal();
        if (!refreshRootSubgoalGuidance())
        {
            // Completeness backstop: a root-level guidance failure while
            // rejection memory is present may be caused by stale no-goods.
            // Reset the memory once (rate-limited) and retry on the next
            // tick; a repeated failure with empty memory falls through to
            // IDLE as before.
            const double now_sec = node_handle_->get_clock()->now().seconds();
            constexpr double kBackstopCooldownSec = 5.0;
            if (hasRefutationMemory() &&
                (last_refutation_backstop_sec_ < 0.0 ||
                 now_sec - last_refutation_backstop_sec_ > kBackstopCooldownSec))
            {
                last_refutation_backstop_sec_ = now_sec;
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Local] Root guidance failed with rejection memory present; "
                            "resetting refutation memory as a completeness backstop and retrying");
                resetRefutationState();
                break;
            }
            local_status_.store(LocalStatus::IDLE);
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Local] NAVIGATING -> IDLE (guidance path is empty)");
            break;
        }

        // Top-down necessity check: walk root -> ... -> deepest child and
        // drop the first subtree whose justification is no longer valid.
        // A non-root subgoal exists because some upstream hit_pt needed to
        // be cleared (FOV_CLEAR) or observed (OBSERVE_UNKNOWN). That hit
        // is recorded on the child as `anchor_point`. If perception
        // collected while flying the chain has already cleared that
        // anchor, the child and everything below it become redundant and
        // we should immediately fall back to the parent for replanning.
        if (root_subgoal_)
        {
            SubgoalNode *node = root_subgoal_.get();
            while (node && node->child)
            {
                SubgoalNode *child = node->child.get();
                if (isHitPointCleared(child->anchor_point))
                {
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Subgoal] Top-down validation: anchor [%.2f, %.2f, %.2f] of "
                                "child %llu (%s) cleared, pruning subtree under parent %llu (%s) "
                                "and replanning from parent",
                                child->anchor_point.x(), child->anchor_point.y(), child->anchor_point.z(),
                                static_cast<unsigned long long>(child->node_id),
                                subgoalTypeStr(child->type),
                                static_cast<unsigned long long>(node->node_id),
                                subgoalTypeStr(node->type));
                    {
                        std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
                        node->pending_no_good_by_hitpoint.erase(canonicalizeHitpoint(child->anchor_point));
                    }
                    pruneSubgoalChildren(*node);
                    clearNodeExecutionState(*node);
                    occlusion_analyzer_.reset();
                    break;
                }
                node = child;
            }
        }

        // Dependency-triggered invalidation of the rejection memory: walk the
        // chain and drop the memory of any context one of whose recorded
        // hitpoints has been cleared by new observations (paper Sec. IV-D).
        if (root_subgoal_)
        {
            SubgoalNode *node = root_subgoal_.get();
            while (node)
            {
                if (invalidateClearedRejections(*node))
                {
                    occlusion_analyzer_.reset();
                    break; // descendants were pruned together with the reset
                }
                node = node->child.get();
            }
        }

        SubgoalNode *leaf = getDeepestActiveSubgoal();
        if (!leaf)
        {
            local_status_.store(LocalStatus::IDLE);
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Local] NAVIGATING -> IDLE (subgoal chain is empty)");
            break;
        }

        if (leaf->type == SubgoalType::ROOT_GOAL)
        {
            clearSubgoalVisualization();
        }

        // Real current pose (from TF) — used only for arrival / distance
        // checks and logging below. Search starts and published-traj start
        // yaws use plan_start_pos/plan_start_yaw instead, so that waypoint[0]
        // is a certified-safe point ahead of the robot rather than the
        // possibly-in-inflation current TF voxel.
        Eigen::Vector3d robot_pos;
        double robot_yaw;
        if (!getRobotPose(robot_pos, robot_yaw))
        {
            break;
        }

        Eigen::Vector3d plan_start_pos;
        double plan_start_yaw = 0.0;
        if (!getPlanStartState(plan_start_pos, plan_start_yaw))
        {
            break;
        }

        if (startup_escape_traj_active_)
        {
            const Eigen::Vector3d escape_goal(startup_escape_endpoint_.x(),
                                              startup_escape_endpoint_.y(),
                                              startup_escape_endpoint_.z());
            const double escape_dist = (robot_pos - escape_goal).norm();
            if (escape_dist > viewpoint_reach_tol_)
            {
                RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 1000,
                                     "[StartupEscape] En route to corridor head [%.2f, %.2f, %.2f] (dist %.2f m)",
                                     startup_escape_endpoint_.x(),
                                     startup_escape_endpoint_.y(),
                                     startup_escape_endpoint_.z(),
                                     escape_dist);
                break;
            }

            startup_escape_traj_active_ = false;
            startup_escape_completed_ = true;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[StartupEscape] Reached corridor head [%.2f, %.2f, %.2f], resuming normal subgoal planning",
                        startup_escape_endpoint_.x(),
                        startup_escape_endpoint_.y(),
                        startup_escape_endpoint_.z());
        }

        std::vector<openvdb::Vec3d> startup_escape_prefix;
        Eigen::Vector3d effective_plan_start_pos = plan_start_pos;
        if (!startup_escape_completed_)
        {
            int inflate_val = 0;
            const bool start_inflated =
                map_manager_->query_is_inflated_at_world(plan_start_pos, inflate_val);
            if (!start_inflated)
            {
                startup_escape_completed_ = true;
            }
            else if (buildStartupEscapePrefix(plan_start_pos, plan_start_yaw,
                                              startup_escape_prefix, effective_plan_start_pos))
            {
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[StartupEscape] Using body+x corridor prefix (%zu pts) from [%.2f, %.2f, %.2f] "
                            "to virtual free start [%.2f, %.2f, %.2f]",
                            startup_escape_prefix.size(),
                            plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z(),
                            effective_plan_start_pos.x(), effective_plan_start_pos.y(), effective_plan_start_pos.z());
            }
        }

        if (leaf->type != SubgoalType::ROOT_GOAL &&
            (leaf->guidance_path_raw.empty() || leaf->guidance_path_short.empty()))
        {
            if (!planGuidanceToSubgoal(*leaf, effective_plan_start_pos))
            {
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] Leaf node %llu has no valid optimistic guidance, pruning it and returning to parent",
                            static_cast<unsigned long long>(leaf->node_id));
                if (root_subgoal_)
                {
                    markParentViewpointRejected(*leaf);
                    pruneSubgoalNodeById(*root_subgoal_, leaf->node_id);
                }
                break;
            }
        }

        // --- Monitor active FOV plan ---
        if (leaf->has_active_plan)
        {
            const VisLayer vis_layer =
                (leaf->type == SubgoalType::ROOT_GOAL) ? VisLayer::ROOT : VisLayer::SUBGOAL;
            if (leaf->has_preview_plan)
            {
                bool anchor_reached;
                {
                    std::lock_guard<std::mutex> slk(stitch_monitor_mtx_);
                    anchor_reached = stitch_vp1_reached_;
                }

                if (!anchor_reached)
                {
                    RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 1500,
                                         "[Local] Flying preview plan for %s node %llu, waiting for anchor arrival",
                                         subgoalTypeStr(leaf->type),
                                         static_cast<unsigned long long>(leaf->node_id));
                    break;
                }

                if (handleRecedingPreviewAnchorReached(*leaf,
                                                       vis_layer,
                                                       plan_start_pos,
                                                       plan_start_yaw))
                {
                    break;
                }

                if (isHitPointCleared(leaf->active_plan_hit_point))
                {
                    const LookaheadState la = getLookaheadState();
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Local] Anchor hit point [%.2f, %.2f, %.2f] cleared after preview anchor reached, "
                                "continuing immediate replan from current lookahead [%.2f, %.2f, %.2f] "
                                "(valid=%d, traj=%lu, t=%.2fs)",
                                leaf->active_plan_hit_point.x(),
                                leaf->active_plan_hit_point.y(),
                                leaf->active_plan_hit_point.z(),
                                la.pos.x(), la.pos.y(), la.pos.z(),
                                la.valid ? 1 : 0,
                                static_cast<unsigned long>(la.traj_id),
                                la.t_eval);
                    clearSecondaryFovPlanVis(vis_layer);
                    clearNodeExecutionState(*leaf);
                    occlusion_analyzer_.reset();
                    // Fall through to fresh hit-point search / replanning.
                }
            }

            if (leaf->has_active_plan)
            {
                if (isHitPointCleared(leaf->active_plan_hit_point))
                {
                    const uint64_t leaf_id = leaf->node_id;
                    const bool remove_leaf = (leaf->type != SubgoalType::ROOT_GOAL);
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Local] Hit point [%.2f, %.2f, %.2f] cleared, re-planning",
                                leaf->active_plan_hit_point.x(), leaf->active_plan_hit_point.y(), leaf->active_plan_hit_point.z());
                    clearSecondaryFovPlanVis(vis_layer);
                    pruneSubgoalChildren(*leaf);
                    clearNodeExecutionState(*leaf);
                    occlusion_analyzer_.reset();
                    if (remove_leaf && root_subgoal_)
                    {
                        pruneSubgoalNodeById(*root_subgoal_, leaf_id);
                        break;
                    }
                    // No break — fall through to findFirstSafeEdgeHit() which
                    // re-walks the guidance path and updates hit point.
                }
                else
                {
                    Eigen::Vector3d vp(leaf->active_viewpoint.x(), leaf->active_viewpoint.y(), leaf->active_viewpoint.z());
                    double dist = (robot_pos - vp).norm();
                    if (dist < viewpoint_reach_tol_)
                    {
                        double yaw_err = wrapToPi(robot_yaw - leaf->active_target_yaw);
                        if (std::fabs(yaw_err) > yaw_reach_tol_)
                        {
                            RCLCPP_INFO_THROTTLE(
                                node_handle_->get_logger(),
                                *node_handle_->get_clock(), 1000,
                                "[Local] At viewpoint but yaw not settled "
                                "(cur=%.2f, tgt=%.2f, err=%.2f°)",
                                robot_yaw, leaf->active_target_yaw,
                                yaw_err * 180.0 / M_PI);
                            break;
                        }

                        // pos+yaw converged, hit point still not cleared.
                        // First arrival: pick + lock an unknown and plan the
                        // observe viewpoint. Later arrivals: dwell on the locked
                        // unknown and escalate on failure (handleObserveDwell).
                        handleObserveDwell(*leaf, vis_layer, robot_pos);
                        break;
                    }
                    RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 2000,
                                         "[Local] En route to viewpoint for %s node %llu (dist %.2f m)",
                                         subgoalTypeStr(leaf->type),
                                         static_cast<unsigned long long>(leaf->node_id),
                                         dist);
                    break;
                }
            }
        }

        // --- No active plan: find safe edge and plan ---
        const std::vector<openvdb::Vec3d> local_guidance_path =
            localGuidancePath(*leaf);
        SafeEdgeHitResult hit = findFirstSafeEdgeHit(local_guidance_path);

        if (hit.path_blocked)
        {
            leaf->has_guidance_progress_floor = false;
            leaf->guidance_progress_floor = openvdb::Vec3d(0, 0, 0);

            const openvdb::Vec3d plan_start_world(plan_start_pos.x(),
                                                  plan_start_pos.y(),
                                                  plan_start_pos.z());
            const openvdb::Coord virtual_start_ijk =
                chooseTargetCentricSearchStart(hit.hit_pt, plan_start_world);
            const bool have_virtual_start = isStrictFreeVoxel(virtual_start_ijk);
            const auto grid_tf = map_manager_->get_grid_transform();
            const openvdb::Vec3d virtual_start_world =
                grid_tf->indexToWorld(virtual_start_ijk);

            if (leaf->type == SubgoalType::ROOT_GOAL)
            {
                if (have_virtual_start)
                {
                    {
                        std::lock_guard<std::mutex> lk(global_cv_mtx_);
                        pending_global_replan_start_override_ = virtual_start_ijk;
                    }
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Local] Path blocked at idx %zu [%.2f, %.2f, %.2f], requesting root guidance replan "
                                "from virtual start [%.2f, %.2f, %.2f]",
                                hit.hit_idx, hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                                virtual_start_world.x(), virtual_start_world.y(), virtual_start_world.z());
                }
                else
                {
                    {
                        std::lock_guard<std::mutex> lk(global_cv_mtx_);
                        pending_global_replan_start_override_.reset();
                    }
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Local] Path blocked at idx %zu [%.2f, %.2f, %.2f], virtual guidance start unavailable; "
                                "requesting lookahead-based replan",
                                hit.hit_idx, hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z());
                }
                need_global_replan_.store(true);
                wakeGlobalPlanner();
            }
            else
            {
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] Guidance to %s node %llu blocked at idx %zu [%.2f, %.2f, %.2f], replanning subgoal guidance",
                            subgoalTypeStr(leaf->type),
                            static_cast<unsigned long long>(leaf->node_id),
                            hit.hit_idx, hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z());
                const Eigen::Vector3d subgoal_guidance_start =
                    have_virtual_start
                        ? Eigen::Vector3d(virtual_start_world.x(),
                                          virtual_start_world.y(),
                                          virtual_start_world.z())
                        : plan_start_pos;
                if (have_virtual_start)
                {
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Subgoal] Replanning guidance from virtual start [%.2f, %.2f, %.2f]",
                                virtual_start_world.x(), virtual_start_world.y(), virtual_start_world.z());
                }
                // Blocked replan is the one event that re-freezes the subtree
                // guidance origin: staleness resets exactly when a new
                // polyline is being cut anyway, never mid-march.
                leaf->has_guidance_origin = true;
                leaf->guidance_origin_world = openvdb::Vec3d(subgoal_guidance_start.x(),
                                                             subgoal_guidance_start.y(),
                                                             subgoal_guidance_start.z());
                leaf->guidance_origin_ijk = openvdb::Coord::round(
                    map_manager_->get_grid_transform()->worldToIndex(leaf->guidance_origin_world));
                if (!planGuidanceToSubgoal(*leaf, subgoal_guidance_start))
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Subgoal] Replan failed for %s node %llu, pruning it and returning to parent",
                                subgoalTypeStr(leaf->type),
                                static_cast<unsigned long long>(leaf->node_id));
                    if (root_subgoal_)
                    {
                        markParentViewpointRejected(*leaf);
                        pruneSubgoalNodeById(*root_subgoal_, leaf->node_id);
                    }
                }
                else
                {
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Subgoal] Replanned guidance to %s node %llu (raw %zu, short %zu)",
                                subgoalTypeStr(leaf->type),
                                static_cast<unsigned long long>(leaf->node_id),
                                leaf->guidance_path_raw.size(),
                                leaf->guidance_path_short.size());
                }
            }
            break;
        }

        if (hit.safe_edge_reached)
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local] Safe edge at idx %zu [%.2f, %.2f, %.2f] | ijk (%d,%d,%d) val=%d "
                        "below_val=%d | path front [%.2f, %.2f, %.2f] pts %zu",
                        hit.hit_idx, hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                        hit.hit_ijk.x(), hit.hit_ijk.y(), hit.hit_ijk.z(),
                        hit.hit_inflate_val, hit.below_inflate_val,
                        local_guidance_path.empty() ? 0.0 : local_guidance_path.front().x(),
                        local_guidance_path.empty() ? 0.0 : local_guidance_path.front().y(),
                        local_guidance_path.empty() ? 0.0 : local_guidance_path.front().z(),
                        local_guidance_path.size());
            const VisLayer vis_layer =
                (leaf->type == SubgoalType::ROOT_GOAL) ? VisLayer::ROOT : VisLayer::SUBGOAL;
            publishSafeEdgeVis(hit, vis_layer);
            if (hit.has_search_start)
            {
                leaf->has_certified_guidance_start = true;
                leaf->certified_guidance_start_ijk = hit.search_start_ijk;
                leaf->certified_guidance_start_world = hit.search_start_world;
            }

            const openvdb::Vec3d plan_start_world(effective_plan_start_pos.x(),
                                                  effective_plan_start_pos.y(),
                                                  effective_plan_start_pos.z());
            const auto fov_candidate_state_str = [](FovCandidateState s) -> const char *
            {
                switch (s)
                {
                case FovCandidateState::CERTIFIED:
                    return "CERTIFIED";
                case FovCandidateState::CONTAMINATED:
                    return "CONTAMINATED";
                case FovCandidateState::INVALID:
                default:
                    return "INVALID";
                }
            };
            const auto maybe_publish_startup_escape = [&]()
            {
                if (startup_escape_prefix.size() <= 1)
                {
                    return;
                }
                // Slew yaw during the escape so the sensor lands pointing at
                // the current hit point. By the time the robot reaches the
                // corridor head, FOV is already clearing the hit, dramatically
                // improving vp1 search success on the next planning cycle.
                const openvdb::Vec3d &endp = startup_escape_prefix.back();
                const double dx = hit.hit_pt.x() - endp.x();
                const double dy = hit.hit_pt.y() - endp.y();
                double end_yaw = plan_start_yaw;
                if (dx * dx + dy * dy > 1.0e-6)
                {
                    end_yaw = std::atan2(dy, dx) - fov_cfg_.body_yaw;
                }
                interpolateAndPublishTrajectory(startup_escape_prefix,
                                                plan_start_yaw, end_yaw, false);
                startup_escape_traj_active_ = true;
                startup_escape_endpoint_ = startup_escape_prefix.back();
            };

            std::vector<openvdb::Vec3d> vp1_path;
            std::vector<openvdb::Vec3d> vp1_short;
            openvdb::Vec3d vp1_viewpoint;
            double vp1_target_yaw = 0.0;
            double vp1_ms = 0.0;
            FovCandidateState vp1_state = FovCandidateState::INVALID;
            CandidateSetSnapshot vp1_candidate_snapshot;
            auto publish_fov_candidate_snapshot = [this](const CandidateSetSnapshot &snapshot)
            {
                publishCandidateSetVis(CandidateSetVisKind::FOV,
                                       snapshot.certified_candidates,
                                       snapshot.contaminated_candidates,
                                       snapshot.selected_coord,
                                       snapshot.has_selected,
                                       snapshot.selected_label);
            };
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local][FOV] Cycle start for %s node %llu | start [%.2f, %.2f, %.2f] yaw %.2f | hit [%.2f, %.2f, %.2f] idx %zu",
                        subgoalTypeStr(leaf->type),
                        static_cast<unsigned long long>(leaf->node_id),
                        plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z(),
                        plan_start_yaw,
                        hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                        hit.hit_idx);
            openvdb::Coord fov_search_start_ijk;
            if (hit.has_search_start)
            {
                fov_search_start_ijk = hit.search_start_ijk;
            }
            else
            {
                if (startup_escape_prefix.size() > 1)
                {
                    const auto grid_tf = map_manager_->get_grid_transform();
                    fov_search_start_ijk = openvdb::Coord::round(
                        grid_tf->worldToIndex(plan_start_world));
                    if (!isStrictFreeVoxel(fov_search_start_ijk))
                    {
                        RCLCPP_WARN(node_handle_->get_logger(),
                                    "[StartupEscape] Virtual free start [%.2f, %.2f, %.2f] is not strict-free; "
                                    "cannot seed FOV search",
                                    plan_start_world.x(), plan_start_world.y(), plan_start_world.z());
                        clearNodeExecutionState(*leaf);
                        break;
                    }
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[StartupEscape] Using corridor endpoint [%.2f, %.2f, %.2f] as FOV search start "
                                "because hitpoint is at guidance idx 0",
                                plan_start_world.x(), plan_start_world.y(), plan_start_world.z());
                }
                else if (leaf->has_certified_guidance_start &&
                         isStrictFreeVoxel(leaf->certified_guidance_start_ijk))
                {
                    fov_search_start_ijk = leaf->certified_guidance_start_ijk;
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[FOV] Safe-edge hit [%.2f, %.2f, %.2f] has no path-local predecessor; "
                                "using cached certified a(h) [%.2f, %.2f, %.2f] as FOV search start",
                                hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                                leaf->certified_guidance_start_world.x(),
                                leaf->certified_guidance_start_world.y(),
                                leaf->certified_guidance_start_world.z());
                }
                else
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[FOV] Safe-edge hit [%.2f, %.2f, %.2f] has no strict-free predecessor; "
                                "guidance invariant is broken",
                                hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z());
                    clearNodeExecutionState(*leaf);
                    if (leaf->type == SubgoalType::ROOT_GOAL)
                    {
                        need_global_replan_.store(true);
                        wakeGlobalPlanner();
                    }
                    else if (root_subgoal_)
                    {
                        markParentViewpointRejected(*leaf);
                        pruneSubgoalNodeById(*root_subgoal_, leaf->node_id);
                    }
                    break;
                }
            }
            auto cache_active_hit_search_start = [&](SubgoalNode &node)
            {
                const auto grid_tf = map_manager_->get_grid_transform();
                node.active_plan_hit_point = hit.hit_pt;
                node.active_plan_has_search_start = true;
                node.active_plan_search_start_ijk = fov_search_start_ijk;
                node.active_plan_search_start_world =
                    grid_tf->indexToWorld(fov_search_start_ijk);
                node.has_certified_guidance_start = true;
                node.certified_guidance_start_ijk = node.active_plan_search_start_ijk;
                node.certified_guidance_start_world = node.active_plan_search_start_world;
            };
            const bool vp1_result = buildTargetCentricFovPlan(
                *leaf,
                hit.hit_pt,
                plan_start_world,
                fov_search_start_ijk,
                FovClearProfileKind::STANDARD,
                true,
                vp1_path,
                vp1_short,
                vp1_viewpoint,
                vp1_target_yaw,
                vp1_ms,
                vp1_state,
                vis_layer,
                true,
                &vp1_candidate_snapshot);

            if (!vp1_result)
            {
                clearCandidateSetVis(CandidateSetVisKind::FOV);
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Local] Target-centric VP1 search failed for hitpoint [%.2f, %.2f, %.2f] (%.1f ms)",
                            hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(), vp1_ms);
                const GoalSearchStopReason stop_reason = fov_astar_.getLastGoalSearchStopReason();
                if (stop_reason == GoalSearchStopReason::EXHAUSTED_NO_GOAL)
                {
                    cache_active_hit_search_start(*leaf);
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Local] FOV clear candidates exhausted for hitpoint "
                                "[%.2f, %.2f, %.2f]; falling back to observe a witness unknown",
                                hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z());
                    handleOccludedFrontier(*leaf, vis_layer);
                }
                else
                {
                    clearNodeExecutionState(*leaf);
                }
                break;
            }

            pruneSubgoalChildren(*leaf);
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local][FOV] vp1 standard success | hit [%.2f, %.2f, %.2f] -> viewpoint [%.2f, %.2f, %.2f] "
                        "yaw %.2f | label=%s | raw %zu short %zu | %.1f ms",
                        hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                        vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z(),
                        vp1_target_yaw,
                        fov_candidate_state_str(vp1_state),
                        vp1_path.size(), vp1_short.size(), vp1_ms);

            clearSecondaryFovPlanVis(vis_layer);

            bool degenerate_subgoal_goal = false;
            if (leaf->type != SubgoalType::ROOT_GOAL)
            {
                auto grid_tf = map_manager_->get_grid_transform();
                const openvdb::Coord vp1_ijk =
                    openvdb::Coord::round(grid_tf->worldToIndex(vp1_viewpoint));
                const openvdb::Coord leaf_goal_ijk =
                    openvdb::Coord::round(grid_tf->worldToIndex(leaf->goal_pos));
                degenerate_subgoal_goal = (vp1_ijk == leaf_goal_ijk);
            }

            if (degenerate_subgoal_goal)
            {
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] Degenerate recursion detected for %s node %llu: "
                            "vp1 [%.2f, %.2f, %.2f] falls in the same goal voxel as current leaf goal "
                            "[%.2f, %.2f, %.2f]; switching directly to observe for hit [%.2f, %.2f, %.2f]",
                            subgoalTypeStr(leaf->type),
                            static_cast<unsigned long long>(leaf->node_id),
                            vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z(),
                            leaf->goal_pos.x(), leaf->goal_pos.y(), leaf->goal_pos.z(),
                            hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z());
                cache_active_hit_search_start(*leaf);
                clearCandidateSetVis(CandidateSetVisKind::FOV);
                handleOccludedFrontier(*leaf, vis_layer);
                needs_escape_corridor_ = false;
                break;
            }

            if (vp1_state != FovCandidateState::CERTIFIED)
            {
                publish_fov_candidate_snapshot(vp1_candidate_snapshot);
                if (!activateFovClearSubgoal(*leaf, &hit, hit.hit_pt, vp1_viewpoint, vp1_target_yaw,
                                             effective_plan_start_pos, plan_start_yaw))
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] Contaminated VP1 branch [%.2f, %.2f, %.2f] found but no optimistic guidance path from robot",
                            vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z());
                    pruneSubgoalChildren(*leaf);
                    clearNodeExecutionState(*leaf);
                    clearCandidateSetVis(CandidateSetVisKind::FOV);
                    break;
                }

                SubgoalNode *child = leaf->child.get();
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] VP1 branch is contaminated; created FOV_CLEAR child node %llu "
                            "from hitpoint [%.2f, %.2f, %.2f] to viewpoint [%.2f, %.2f, %.2f] "
                            "(guidance %zu pts, %.1f ms)",
                            child ? static_cast<unsigned long long>(child->node_id) : 0ULL,
                            hit.hit_pt.x(), hit.hit_pt.y(), hit.hit_pt.z(),
                            vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z(),
                            child ? child->guidance_path_short.size() : 0u,
                            vp1_ms);
                if (child)
                {
                    child->status = SubgoalStatus::ACTIVE;
                }
                maybe_publish_startup_escape();
                needs_escape_corridor_ = false;
                break;
            }

            std::vector<openvdb::Vec3d> primary_exec_path_raw;
            std::vector<openvdb::Vec3d> primary_exec_path_short;
            double primary_access_ms = 0.0;
            std::string primary_access_failure;
            if (!buildCertifiedAccessPath(plan_start_world, vp1_viewpoint,
                                          primary_exec_path_raw, primary_exec_path_short,
                                          &primary_access_ms, &primary_access_failure))
            {
                if (!activateFovClearSubgoal(*leaf, &hit, hit.hit_pt, vp1_viewpoint, vp1_target_yaw,
                                             effective_plan_start_pos, plan_start_yaw))
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Subgoal] Certified VP1 candidate [%.2f, %.2f, %.2f] found but neither certified connect nor optimistic guidance succeeded "
                                "(certified-connect reason=%s)",
                                vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z(),
                                primary_access_failure.c_str());
                    pruneSubgoalChildren(*leaf);
                    clearNodeExecutionState(*leaf);
                    clearCandidateSetVis(CandidateSetVisKind::FOV);
                    break;
                }
                publish_fov_candidate_snapshot(vp1_candidate_snapshot);

                SubgoalNode *child = leaf->child.get();
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Subgoal] VP1 is certified as a candidate but requires contaminated refinement; created FOV_CLEAR child node %llu "
                            "for viewpoint [%.2f, %.2f, %.2f] (guidance %zu pts, certified-connect failed after %.1f ms, reason=%s)",
                            child ? static_cast<unsigned long long>(child->node_id) : 0ULL,
                            vp1_viewpoint.x(), vp1_viewpoint.y(), vp1_viewpoint.z(),
                            child ? child->guidance_path_short.size() : 0u,
                            primary_access_ms,
                            primary_access_failure.c_str());
                if (child)
                {
                    child->status = SubgoalStatus::ACTIVE;
                }
                maybe_publish_startup_escape();
                needs_escape_corridor_ = false;
                break;
            }

            std::vector<openvdb::Vec3d> primary_fov_path = vp1_path;
            std::vector<openvdb::Vec3d> primary_fov_short = vp1_short;
            openvdb::Vec3d primary_viewpoint = vp1_viewpoint;
            double primary_target_yaw = vp1_target_yaw;
            double primary_ms = vp1_ms;
            bool using_tight_primary = false;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local][FOV] using standard primary anchor [%.2f, %.2f, %.2f]; "
                        "tight anchor search is bypassed by receding-preview mode",
                        primary_viewpoint.x(), primary_viewpoint.y(), primary_viewpoint.z());

            publish_fov_candidate_snapshot(vp1_candidate_snapshot);

            PreviewSegmentPlan preview_plan;
            if (hit.has_search_start)
            {
                advanceGuidanceProgressFloor(*leaf, hit.search_start_world);
            }

            const bool have_preview = buildCertifiedPreviewSegment(*leaf,
                                                                   primary_viewpoint,
                                                                   local_guidance_path,
                                                                   hit.hit_pt,
                                                                   fov_search_start_ijk,
                                                                   vis_layer,
                                                                   preview_plan);
            if (have_preview)
            {
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Local][FOV] vp2 preview candidate | preview target [%.2f, %.2f, %.2f] "
                            "-> viewpoint [%.2f, %.2f, %.2f] yaw %.2f | stop [%.2f, %.2f, %.2f] "
                            "| fov raw %zu short %zu | preview exec %zu",
                            preview_plan.hypothetical_hit_point.x(),
                            preview_plan.hypothetical_hit_point.y(),
                            preview_plan.hypothetical_hit_point.z(),
                            preview_plan.viewpoint.x(),
                            preview_plan.viewpoint.y(),
                            preview_plan.viewpoint.z(),
                            preview_plan.target_yaw,
                            preview_plan.stop_point.x(),
                            preview_plan.stop_point.y(),
                            preview_plan.stop_point.z(),
                            preview_plan.fov_path_raw.size(),
                            preview_plan.fov_path_short.size(),
                            preview_plan.exec_path.size());
            }
            else
            {
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Local][FOV] vp2 preview unavailable from primary [%.2f, %.2f, %.2f] "
                            "(reason=%s); falling back to single-stage clear",
                            primary_viewpoint.x(), primary_viewpoint.y(), primary_viewpoint.z(),
                            preview_plan.failure_reason.c_str());
            }

            // Always route execution through the optimizer. Short / degenerate
            // path cases (e.g. plan_start already at viewpoint, requiring mostly
            // yaw change) are valid optimizer inputs: connect A* still produces
            // a 1-2 point path, and the optimizer handles short-distance +
            // large-yaw-change inputs by tuning piece durations. Keeping a
            // separate "already at viewpoint → publish yaw-in-place" branch
            // means two different reach-tolerance standards (tp progress vs TF
            // distance) that can deadlock against each other.
            if (have_preview && preview_plan.exec_path.size() > 1)
            {
                std::vector<openvdb::Vec3d> primary_exec_path_for_preview =
                    (primary_exec_path_short.size() >= 2)
                        ? primary_exec_path_short
                        : std::vector<openvdb::Vec3d>{plan_start_world, primary_viewpoint};

                PreviewProgressWindow primary_anchor_progress;
                buildPreviewProgressWindow(primary_viewpoint,
                                           local_guidance_path,
                                           hit.hit_pt,
                                           fov_stitch_lookahead_dist_,
                                           FovClearProfileKind::STANDARD,
                                           primary_anchor_progress);

                if (!publishRecedingPreviewFromAnchor(*leaf,
                                                       vis_layer,
                                                       plan_start_world,
                                                       primary_exec_path_for_preview,
                                                       startup_escape_prefix,
                                                       primary_viewpoint,
                                                       primary_anchor_progress,
                                                       preview_plan,
                                                       plan_start_yaw,
                                                       startup_escape_prefix.empty(),
                                                       "Local"))
                {
                    clearSecondaryFovPlanVis(vis_layer);
                    break;
                }

                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Local] Preview FOV clear: primary anchor [%.2f, %.2f, %.2f] (%s) -> vp2 [%.2f, %.2f, %.2f] "
                            "for preview target [%.2f, %.2f, %.2f] (primary %.1f ms, preview %.1f ms, path %zu pts)",
                            primary_viewpoint.x(), primary_viewpoint.y(), primary_viewpoint.z(),
                            using_tight_primary ? "tight" : "standard",
                            preview_plan.viewpoint.x(), preview_plan.viewpoint.y(), preview_plan.viewpoint.z(),
                            preview_plan.hypothetical_hit_point.x(),
                            preview_plan.hypothetical_hit_point.y(),
                            preview_plan.hypothetical_hit_point.z(),
                            primary_ms, preview_plan.fov_search_ms, preview_plan.exec_path.size());

                publishFovPlanVis(primary_fov_path, primary_fov_short, primary_viewpoint, hit.hit_pt, vis_layer,
                                  using_tight_primary);

                cache_active_hit_search_start(*leaf);
                leaf->last_anchor_hitpoint = canonicalizeHitpoint(hit.hit_pt);
                leaf->has_last_anchor_hitpoint = true;
                needs_escape_corridor_ = false;
                break;
            }

            publishFovPlanVis(primary_fov_path, primary_fov_short, primary_viewpoint, hit.hit_pt, vis_layer,
                              using_tight_primary);
            clearSecondaryFovPlanVis(vis_layer);

            std::vector<openvdb::Vec3d> primary_exec_path =
                (primary_exec_path_short.size() > 1) ? primary_exec_path_short : primary_exec_path_raw;
            if (primary_exec_path.size() <= 1)
            {
                primary_exec_path = {plan_start_world, primary_viewpoint};
            }
            primary_exec_path = prependPathPrefix(startup_escape_prefix, primary_exec_path);
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Local] FOV primary viewpoint [%.2f, %.2f, %.2f] (%s) "
                        "(%.1f ms total, fov raw %zu, fov short %zu, exec %zu)",
                        primary_viewpoint.x(), primary_viewpoint.y(), primary_viewpoint.z(),
                        using_tight_primary ? "tight" : "standard",
                        primary_ms, primary_fov_path.size(), primary_fov_short.size(), primary_exec_path.size());
            interpolateAndPublishTrajectory(primary_exec_path, plan_start_yaw, primary_target_yaw,
                                            startup_escape_prefix.empty());

            leaf->active_viewpoint = primary_viewpoint;
            leaf->active_anchor_seed = primary_viewpoint;
            cache_active_hit_search_start(*leaf);
            leaf->active_target_yaw = primary_target_yaw;
            leaf->has_active_plan = true;
            leaf->goal_traj_published = false;
            needs_escape_corridor_ = false;
            break;
        }

        // Path clear all the way to goal
        if (!leaf->goal_traj_published)
        {
            if (leaf->guidance_path_short.empty())
            {
                if (leaf->type == SubgoalType::ROOT_GOAL)
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Local] Root guidance is clear but shortened path is empty, requesting global replan");
                    need_global_replan_.store(true);
                    wakeGlobalPlanner();
                }
                else
                {
                    const uint64_t leaf_id = leaf->node_id;
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Subgoal] %s node %llu has clear guidance but empty shortened path, pruning it",
                                subgoalTypeStr(leaf->type),
                                static_cast<unsigned long long>(leaf_id));
                    if (root_subgoal_)
                    {
                        pruneSubgoalNodeById(*root_subgoal_, leaf_id);
                    }
                }
                break;
            }

            if (!local_guidance_path.empty())
            {
                pruneSubgoalChildren(*leaf);

                const openvdb::Vec3d goal_world = local_guidance_path.back();
                openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
                openvdb::Coord plan_start_ijk = openvdb::Coord::round(
                    grid_tf->worldToIndex(openvdb::Vec3d(effective_plan_start_pos.x(),
                                                         effective_plan_start_pos.y(),
                                                         effective_plan_start_pos.z())));
                openvdb::Coord goal_ijk = openvdb::Coord::round(grid_tf->worldToIndex(goal_world));

                auto t_conn = std::chrono::steady_clock::now();
                int conn_result = connect_astar_.search(plan_start_ijk, goal_ijk);
                double conn_ms = std::chrono::duration<double, std::milli>(
                                     std::chrono::steady_clock::now() - t_conn)
                                     .count();

                const VisLayer vis_layer =
                    (leaf->type == SubgoalType::ROOT_GOAL) ? VisLayer::ROOT : VisLayer::SUBGOAL;

                if (conn_result != Astar::PATH_FOUND)
                {
                    RCLCPP_WARN(node_handle_->get_logger(),
                                "[Local] Path clear but connect A* to goal [%.2f, %.2f, %.2f] failed "
                                "(result %d, %.1f ms), falling back to guidance path",
                                goal_world.x(), goal_world.y(), goal_world.z(),
                                conn_result, conn_ms);
                    std::vector<openvdb::Vec3d> fallback_path = prependPathPrefix(
                        startup_escape_prefix,
                        trimPathFromPoint(local_guidance_path,
                                          openvdb::Vec3d(effective_plan_start_pos.x(),
                                                         effective_plan_start_pos.y(),
                                                         effective_plan_start_pos.z())));
                    interpolateAndPublishTrajectory(
                        fallback_path,
                        plan_start_yaw,
                        (leaf->type == SubgoalType::ROOT_GOAL) ? curr_goal_yaw_ : leaf->goal_yaw,
                        startup_escape_prefix.empty());
                    leaf->goal_traj_published = true;
                    leaf->status = SubgoalStatus::EXECUTING;
                    break;
                }

                std::vector<openvdb::Vec3d> conn_path = connect_astar_.getPathAstar();
                std::vector<openvdb::Vec3d> conn_short;
                connect_astar_.pathShortenStrict(conn_short);
                if (conn_short.empty())
                {
                    conn_short = conn_path;
                }

                double goal_yaw;
                {
                    std::lock_guard<std::mutex> lk(goal_mtx_);
                    goal_yaw = (leaf->type == SubgoalType::ROOT_GOAL) ? curr_goal_yaw_ : leaf->goal_yaw;
                }

                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Local] Path clear to %s node %llu, connect A* to goal "
                            "[%.2f, %.2f, %.2f] (%.1f ms, %zu pts)",
                            subgoalTypeStr(leaf->type),
                            static_cast<unsigned long long>(leaf->node_id),
                            goal_world.x(), goal_world.y(), goal_world.z(),
                            conn_ms, conn_short.size());

                publishConnectPlanVis(conn_path, conn_short, vis_layer);
                interpolateAndPublishTrajectory(
                    prependPathPrefix(startup_escape_prefix, conn_short),
                    plan_start_yaw, goal_yaw, startup_escape_prefix.empty());
                leaf->goal_traj_published = true;
                leaf->status = SubgoalStatus::EXECUTING;
            }
        }
        else
        {
            if (leaf->type == SubgoalType::ROOT_GOAL && leaf->guidance_path_short.empty())
            {
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Local] Root guidance execution lost shortened path, requesting global replan");
                leaf->goal_traj_published = false;
                need_global_replan_.store(true);
                wakeGlobalPlanner();
                break;
            }

            const openvdb::Vec3d target_pos =
                (leaf->type == SubgoalType::ROOT_GOAL) ? leaf->guidance_path_short.back() : leaf->goal_pos;
            Eigen::Vector3d target_world(target_pos.x(), target_pos.y(), target_pos.z());
            const double dist_to_goal = (robot_pos - target_world).norm();

            if (leaf->type == SubgoalType::ROOT_GOAL &&
                dist_to_goal < viewpoint_reach_tol_)
            {
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Local] Reached ROOT_GOAL node %llu [%.2f, %.2f, %.2f] "
                            "(dist %.2f m), returning to IDLE",
                            static_cast<unsigned long long>(leaf->node_id),
                            target_world.x(), target_world.y(), target_world.z(),
                            dist_to_goal);

                {
                    std::unique_lock<std::shared_mutex> lk(guidance_path_mtx_);
                    shared_guidance_path_.clear();
                    shared_guidance_path_short_.clear();
                }

                resetSubgoalChain();
                occlusion_analyzer_.reset();
                deactivateStitchMonitoring();
                clearCandidateSetVis(CandidateSetVisKind::FOV);
                clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
                clearObserveFrustumVis();
                goal_reach_status_.store(GoalReachStatus::REACHED);
                global_status_.store(GlobalStatus::IDLE);
                local_status_.store(LocalStatus::IDLE);
                break;
            }

            if (leaf->type != SubgoalType::ROOT_GOAL)
            {
                if (dist_to_goal < viewpoint_reach_tol_)
                {
                    RCLCPP_INFO(node_handle_->get_logger(),
                                "[Subgoal] Reached node %llu staging pose [%.2f, %.2f, %.2f], activating local clear/observe phase",
                                static_cast<unsigned long long>(leaf->node_id),
                                leaf->goal_pos.x(), leaf->goal_pos.y(), leaf->goal_pos.z());
                    leaf->active_viewpoint = leaf->goal_pos;
                    leaf->active_target_yaw = leaf->goal_yaw;
                    leaf->active_plan_hit_point = leaf->anchor_point;
                    leaf->has_active_plan = true;
                    // The optimistic guidance-to-staging trajectory for this
                    // subgoal has already been executed. Keep this latched so
                    // that transient local-phase state drops do not republish
                    // the same short staging trajectory.
                    leaf->goal_traj_published = true;
                    leaf->status = SubgoalStatus::ACTIVE;
                    break;
                }
            }

            RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 2000,
                                 "[Local] Following %s node %llu guidance (dist %.2f m)",
                                 subgoalTypeStr(leaf->type),
                                 static_cast<unsigned long long>(leaf->node_id),
                                 dist_to_goal);
        }
        break;
    }

    case LocalStatus::RECOVERY:
    {
        // TODO: yaw-scan recovery
        RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 3000,
                             "[Local] RECOVERY, stub hovering");
        break;
    }
    }
}

// Hit Point Clearance Check

bool FFAPlannerNode::isHitPointCleared(const openvdb::Vec3d &hit_pt) const
{
    openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
    openvdb::Coord ijk = openvdb::Coord::round(grid_tf->worldToIndex(hit_pt));

    int val = 0;
    bool active = map_manager_->query_is_inflated_at_index(ijk, val);
    return !(active && val > 0);
}

// Occluded Frontier Handler

// ---------------------------------------------------------------------------
// Observe phase (unknown-observation plan, paper Sec. IV-B/IV-E) with the
// arrival failsafe.
//
// Engineering failsafe, not part of the analyzed planner: the paper's model
// assumes an executed observation q |= {u} resolves u (Lemma 5). Physically it
// can fail (sensor edge effects, a ray grazing a known obstacle edge, map
// flicker). We then keep u locked, exclude the viewpoint the robot actually
// stood at, and re-select from the (resumed) candidate search with a small
// connect penalty on candidates whose ray grazes the occ-shell. Certified
// candidates are consumed first; contaminated ones (subgoals) only after
// that, matching Alg. 1's order. Physical failures therefore end in H_exh
// exactly like planning refutations once every candidate for u has been
// tried; the semantic gap versus the paper's Exh is accepted (see
// markHitpointExhausted callers).
// ---------------------------------------------------------------------------

bool FFAPlannerNode::isVoxelUnknown(const openvdb::Coord &ijk) const
{
    float ll = 0.0f;
    return !map_manager_->query_log_odds_at_index(ijk, ll);
}

void FFAPlannerNode::releaseObserveLock(SubgoalNode &leaf)
{
    leaf.has_locked_unknown = false;
    leaf.locked_unknown = openvdb::Coord(0, 0, 0);
    leaf.locked_unknown_hitpoint = HitpointKey{};
    leaf.observe_failed_voxels.clear();
    leaf.observe_failure_count = 0;
    resetObserveArrivalCounters(leaf);
}

void FFAPlannerNode::resetObserveArrivalCounters(SubgoalNode &leaf)
{
    leaf.observe_arrival_ticks = 0;
    leaf.observe_dwell_valid = false;
    leaf.observe_dwell_start_count = 0;
}

bool FFAPlannerNode::observeGeometryOkFromPose(const Eigen::Vector3d &robot_pos,
                                               const openvdb::Coord &unknown_ijk)
{
    // Pure FOV geometry (range + vertical angle, yaw assumed aligned to the
    // target) evaluated at the REAL robot position instead of the planned
    // voxel center. No line-of-sight test: that part is what the dwell is
    // for. Reuses the observe search's own predicate for consistency.
    observe_astar_.setTargetUnknown(unknown_ijk);
    return observe_astar_.fovSatisfiedAt(openvdb::Vec3d(robot_pos.x(), robot_pos.y(), robot_pos.z()));
}

void FFAPlannerNode::handleObserveDwell(SubgoalNode &leaf, VisLayer layer,
                                        const Eigen::Vector3d &robot_pos)
{
    if (!leaf.has_locked_unknown)
    {
        // First arrival at a clearing/observe viewpoint with the hit still
        // uncleared: choose and lock an unknown, plan its viewpoint.
        handleOccludedFrontier(leaf, layer);
        return;
    }

    const HitpointKey current_hit = canonicalizeHitpoint(leaf.active_plan_hit_point);
    if (!(leaf.locked_unknown_hitpoint == current_hit))
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe] Active hitpoint changed; dropping locked unknown [%d,%d,%d]",
                    leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z());
        releaseObserveLock(leaf);
        handleOccludedFrontier(leaf, layer);
        return;
    }

    if (!isVoxelUnknown(leaf.locked_unknown))
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe] Locked unknown [%d,%d,%d] resolved after %d failure(s); "
                    "releasing lock and picking the next unknown",
                    leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z(),
                    leaf.observe_failure_count);
        releaseObserveLock(leaf);
        handleOccludedFrontier(leaf, layer);
        return;
    }

    ++leaf.observe_arrival_ticks;
    const int map_count = map_manager_->get_occu_update_count();
    const bool geom_ok = observeGeometryOkFromPose(robot_pos, leaf.locked_unknown);
    if (geom_ok)
    {
        if (!leaf.observe_dwell_valid)
        {
            leaf.observe_dwell_valid = true;
            leaf.observe_dwell_start_count = map_count;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Observe] At viewpoint [%.2f, %.2f, %.2f] for unknown [%d,%d,%d]; "
                        "geometry ok from real pose, dwelling for %d map update(s)",
                        leaf.active_viewpoint.x(), leaf.active_viewpoint.y(), leaf.active_viewpoint.z(),
                        leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z(),
                        observe_dwell_map_updates_);
            return;
        }
        const int updates = map_count - leaf.observe_dwell_start_count;
        if (updates < observe_dwell_map_updates_)
        {
            RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 1000,
                                 "[Observe] Dwelling: %d/%d map updates, unknown [%d,%d,%d] still unknown",
                                 updates, observe_dwell_map_updates_,
                                 leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z());
            return;
        }
        handleObserveFailure(leaf, layer, "observe");
        return;
    }

    if (leaf.observe_arrival_ticks < observe_arrival_max_ticks_)
    {
        RCLCPP_INFO_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 1000,
                             "[Observe] Within reach tol but real pose fails FOV geometry for "
                             "unknown [%d,%d,%d] (tick %d/%d)",
                             leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z(),
                             leaf.observe_arrival_ticks, observe_arrival_max_ticks_);
        return;
    }
    handleObserveFailure(leaf, layer, "pose");
}

void FFAPlannerNode::handleObserveFailure(SubgoalNode &leaf, VisLayer layer, const char *reason)
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord vp_ijk = openvdb::Coord::round(tf->worldToIndex(leaf.active_viewpoint));
    leaf.observe_failed_voxels.insert(vp_ijk);
    ++leaf.observe_failure_count;
    resetObserveArrivalCounters(leaf);

    RCLCPP_WARN(node_handle_->get_logger(),
                "[Observe][Failsafe] unknown [%d,%d,%d] not resolved at viewpoint "
                "[%.2f, %.2f, %.2f] (class=%s, failure #%d, excluded voxels %zu); reselecting",
                leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z(),
                leaf.active_viewpoint.x(), leaf.active_viewpoint.y(), leaf.active_viewpoint.z(),
                reason, leaf.observe_failure_count, leaf.observe_failed_voxels.size());

    Eigen::Vector3d plan_start_pos;
    double plan_start_yaw = 0.0;
    if (!getPlanStartState(plan_start_pos, plan_start_yaw))
    {
        return;
    }
    if (!leaf.active_plan_has_search_start)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Observe][Failsafe] no cached strict-free search start; deferring");
        return;
    }
    const openvdb::Coord start_ijk = leaf.active_plan_search_start_ijk;
    const openvdb::Coord u = leaf.locked_unknown;

    auto predicate = [this, &leaf](const SearchGoalCandidate &candidate, ReachLabel label)
    {
        if (leaf.observe_failed_voxels.count(candidate.coord) > 0)
        {
            return false;
        }
        if (label == ReachLabel::CERTIFIED)
        {
            return true;
        }
        return !isViewpointRejected(leaf, candidate.coord);
    };

    int result = Astar::PATH_NOT_FOUND;
    double obs_ms = 0.0;
    bool used_resume = false;
    observe_astar_.setTargetUnknown(u);
    observe_astar_.setSearchAnchor(openvdb::Vec3d(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z()));
    observe_astar_.clearEscapeCorridor();
    observe_astar_.setAllowSoftFrontierTraversal(true);

    if (observe_astar_.sessionMatches(start_ijk, u))
    {
        const size_t removed = observe_astar_.revalidateCandidates(leaf.observe_failed_voxels);
        const bool untried_certified = !observe_astar_.getCertifiedGoalCandidates().empty();
        if (untried_certified || !observe_astar_.openSetEmpty())
        {
            observe_astar_.setGoalCandidatePredicate(predicate);
            const auto t0 = std::chrono::steady_clock::now();
            result = observe_astar_.resume(observe_resume_budget_);
            obs_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            observe_astar_.clearGoalCandidatePredicate();
            used_resume = true;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Observe][Failsafe] resumed search (removed %zu stale, certified %zu, "
                        "contaminated %zu, open %s, %.1f ms)",
                        removed,
                        observe_astar_.getCertifiedGoalCandidates().size(),
                        observe_astar_.getContaminatedGoalCandidates().size(),
                        observe_astar_.openSetEmpty() ? "empty" : "nonempty",
                        obs_ms);
        }
    }

    const bool resume_gave_nothing =
        used_resume &&
        observe_astar_.getCertifiedGoalCandidates().empty() &&
        observe_astar_.getContaminatedGoalCandidates().empty();
    if (!used_resume || resume_gave_nothing)
    {
        // Fresh search: with the failed voxels in the predicate, previously
        // failed certified candidates no longer set certified_found, so
        // contaminated branches get enumerated (Alg. 1 order).
        observe_astar_.setGoalCandidatePredicate(predicate);
        const auto t0 = std::chrono::steady_clock::now();
        result = observe_astar_.search(start_ijk, u);
        obs_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        observe_astar_.clearGoalCandidatePredicate();
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe][Failsafe] fresh search (certified %zu, contaminated %zu, %.1f ms)",
                    observe_astar_.getCertifiedGoalCandidates().size(),
                    observe_astar_.getContaminatedGoalCandidates().size(),
                    obs_ms);
        if (result != Astar::PATH_FOUND)
        {
            clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
            clearObserveFrustumVis();
            if (observe_astar_.getLastGoalSearchStopReason() == GoalSearchStopReason::EXHAUSTED_NO_GOAL)
            {
                // Every admissible candidate for the locked u has been tried
                // (or none exists): treat like a planning refutation.
                markHitpointExhausted(leaf, canonicalizeHitpoint(leaf.active_plan_hit_point),
                                      "locked unknown: all observe viewpoints failed or none");
            }
            else
            {
                RCLCPP_WARN(node_handle_->get_logger(),
                            "[Observe][Failsafe] no viewpoint for unknown [%d,%d,%d] within budget; "
                            "will retry next tick",
                            u.x(), u.y(), u.z());
            }
            return;
        }
    }
    else if (result != Astar::PATH_FOUND)
    {
        // Resume hit the budget without adding candidates but old ones remain.
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe][Failsafe] resume returned %d; selecting among remaining candidates",
                    result);
    }

    if (layer == VisLayer::ROOT ? (pub_observe_debug_tree_->get_subscription_count() > 0)
                                : (pub_subgoal_observe_debug_tree_->get_subscription_count() > 0))
    {
        auto tree_mk = observe_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::OBSERVE, layer);
    }

    selectAndPublishObserveViewpoint(leaf, layer, plan_start_pos, plan_start_yaw, obs_ms);
}

void FFAPlannerNode::handleOccludedFrontier(SubgoalNode &leaf, VisLayer layer)
{
    const openvdb::Vec3d hit_pt = leaf.active_plan_hit_point;
    const HitpointKey hitpoint = canonicalizeHitpoint(hit_pt);
    occlusion_analyzer_.analyzeTick(hit_pt);

    Eigen::Vector3d plan_start_pos;
    double plan_start_yaw = 0.0;
    if (!getPlanStartState(plan_start_pos, plan_start_yaw))
    {
        return;
    }

    const auto &unknowns = occlusion_analyzer_.getCurrentUnknowns();
    if (unknowns.empty())
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe] No unknowns remain near hit point; clearing active observe plan");
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        const uint64_t leaf_id = leaf.node_id;
        const bool remove_leaf = (leaf.type != SubgoalType::ROOT_GOAL);
        pruneSubgoalChildren(leaf);
        clearNodeExecutionState(leaf);
        occlusion_analyzer_.reset();
        if (remove_leaf && root_subgoal_)
        {
            pruneSubgoalNodeById(*root_subgoal_, leaf_id);
        }
        return;
    }

    if (!leaf.active_plan_has_search_start)
    {
        RCLCPP_WARN_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 1000,
                             "[Observe] Active hit [%.2f, %.2f, %.2f] has no cached strict-free search start; "
                             "deferring observe",
                             hit_pt.x(), hit_pt.y(), hit_pt.z());
        return;
    }
    const openvdb::Coord start_ijk = leaf.active_plan_search_start_ijk;

    // Lock the observation target: keep the previously locked unknown while
    // it is still unknown and belongs to this hitpoint; otherwise sample a
    // fresh one from Omega(h) (Alg. 1 line 26).
    if (leaf.has_locked_unknown &&
        leaf.locked_unknown_hitpoint == hitpoint &&
        isVoxelUnknown(leaf.locked_unknown))
    {
        // keep
    }
    else
    {
        releaseObserveLock(leaf);
        const size_t idx = static_cast<size_t>(std::rand()) % unknowns.size();
        leaf.locked_unknown = unknowns[idx];
        leaf.locked_unknown_hitpoint = hitpoint;
        leaf.has_locked_unknown = true;
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Observe] Locked unknown [%d,%d,%d] (of %zu) for hit [%.2f, %.2f, %.2f]",
                    leaf.locked_unknown.x(), leaf.locked_unknown.y(), leaf.locked_unknown.z(),
                    unknowns.size(), hit_pt.x(), hit_pt.y(), hit_pt.z());
    }
    const openvdb::Coord target_unknown = leaf.locked_unknown;

    observe_astar_.setTargetUnknown(target_unknown);
    observe_astar_.setSearchAnchor(openvdb::Vec3d(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z()));
    observe_astar_.clearEscapeCorridor();
    observe_astar_.setAllowSoftFrontierTraversal(true);
    observe_astar_.setGoalCandidatePredicate(
        [this, &leaf](const SearchGoalCandidate &candidate, ReachLabel label)
        {
            if (leaf.observe_failed_voxels.count(candidate.coord) > 0)
            {
                return false;
            }
            if (label == ReachLabel::CERTIFIED)
            {
                return true;
            }
            return !isViewpointRejected(leaf, candidate.coord);
        });

    auto t0 = std::chrono::steady_clock::now();
    int result = observe_astar_.search(start_ijk, target_unknown);
    double obs_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0)
                        .count();
    observe_astar_.clearGoalCandidatePredicate();

    if (layer == VisLayer::ROOT ? (pub_observe_debug_tree_->get_subscription_count() > 0)
                                : (pub_subgoal_observe_debug_tree_->get_subscription_count() > 0))
    {
        auto tree_mk = observe_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::OBSERVE, layer);
    }

    if (result != Astar::PATH_FOUND)
    {
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        const GoalSearchStopReason stop_reason = observe_astar_.getLastGoalSearchStopReason();
        if (stop_reason == GoalSearchStopReason::EXHAUSTED_NO_GOAL)
        {
            markHitpointExhausted(leaf, hitpoint, "sampled witness unknown exhausted");
        }
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Observe] No viewpoint found for unknown [%d,%d,%d] "
                    "(label-aware %.1fms)",
                    target_unknown.x(), target_unknown.y(), target_unknown.z(),
                    obs_ms);
        return;
    }

    selectAndPublishObserveViewpoint(leaf, layer, plan_start_pos, plan_start_yaw, obs_ms);
}

bool FFAPlannerNode::selectAndPublishObserveViewpoint(SubgoalNode &leaf,
                                                      VisLayer layer,
                                                      const Eigen::Vector3d &plan_start_pos,
                                                      double plan_start_yaw,
                                                      double obs_ms)
{
    const openvdb::Vec3d hit_pt = leaf.active_plan_hit_point;
    const openvdb::Coord target_unknown = leaf.locked_unknown;
    auto grid_tf = map_manager_->get_grid_transform();
    const openvdb::Vec3d target_unknown_world = grid_tf->indexToWorld(target_unknown);
    const auto &certified_candidates = observe_astar_.getCertifiedGoalCandidates();
    const auto &contaminated_candidates = observe_astar_.getContaminatedGoalCandidates();
    const bool failsafe_active = leaf.observe_failure_count > 0;

    bool selected_candidate_path = false;
    ReachLabel selected_label = observe_astar_.getLastPathLabel();
    if (!certified_candidates.empty())
    {
        openvdb::Vec3d selected_goal_world;
        double select_ms = 0.0;
        std::string select_fail;
        const openvdb::Vec3d start_world(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z());
        const bool selected =
            failsafe_active
                ? selectGoalFromSeeds(start_world, certified_candidates,
                                      observe_graze_penalty_vox_, leaf.observe_failed_voxels,
                                      selected_goal_world, &select_ms, &select_fail)
                : selectGoalFromSet(start_world, certified_candidates,
                                    selected_goal_world, &select_ms, &select_fail);
        if (selected)
        {
            const openvdb::Coord selected_goal_ijk =
                openvdb::Coord::round(grid_tf->worldToIndex(selected_goal_world));
            if (observe_astar_.extractPathToGoal(selected_goal_ijk, ReachLabel::CERTIFIED))
            {
                selected_candidate_path = true;
                selected_label = ReachLabel::CERTIFIED;
            }
        }
        else
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Observe] %zu certified VPs unreachable from plan start "
                        "[%.2f, %.2f, %.2f] (%s, %.1f ms); falling back to contaminated set",
                        certified_candidates.size(),
                        plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z(),
                        select_fail.c_str(), select_ms);
        }
    }
    if (!selected_candidate_path && !contaminated_candidates.empty())
    {
        openvdb::Coord selected_goal_ijk;
        if (selectLowestCostGoalFromSet(contaminated_candidates, selected_goal_ijk) &&
            observe_astar_.extractPathToGoal(selected_goal_ijk, ReachLabel::CONTAMINATED))
        {
            selected_candidate_path = true;
            selected_label = ReachLabel::CONTAMINATED;
        }
    }

    std::vector<openvdb::Vec3d> obs_path = observe_astar_.getPathAstar();
    if (obs_path.empty())
    {
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Observe] Search reached end but returned empty path for unknown [%d,%d,%d]",
                    target_unknown.x(), target_unknown.y(), target_unknown.z());
        return false;
    }
    openvdb::Vec3d viewpoint = obs_path.back();
    const double yaw_dx = target_unknown_world.x() - viewpoint.x();
    const double yaw_dy = target_unknown_world.y() - viewpoint.y();
    double target_yaw = 0.0;
    if (yaw_dx * yaw_dx + yaw_dy * yaw_dy > 1.0e-6)
    {
        target_yaw = std::atan2(yaw_dy, yaw_dx) - fov_cfg_.body_yaw;
    }
    const openvdb::Coord viewpoint_ijk = openvdb::Coord::round(grid_tf->worldToIndex(viewpoint));
    if (selected_label == ReachLabel::CONTAMINATED &&
        isViewpointRejected(leaf, viewpoint_ijk))
    {
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Observe] Selected contaminated viewpoint voxel [%d,%d,%d] for unknown [%d,%d,%d] is rejected; skipping",
                    viewpoint_ijk.x(), viewpoint_ijk.y(), viewpoint_ijk.z(),
                    target_unknown.x(), target_unknown.y(), target_unknown.z());
        return false;
    }
    if (leaf.observe_failed_voxels.count(viewpoint_ijk) > 0)
    {
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Observe][Failsafe] Selected viewpoint voxel [%d,%d,%d] already failed for unknown "
                    "[%d,%d,%d]; skipping",
                    viewpoint_ijk.x(), viewpoint_ijk.y(), viewpoint_ijk.z(),
                    target_unknown.x(), target_unknown.y(), target_unknown.z());
        return false;
    }

    publishCandidateSetVis(CandidateSetVisKind::OBSERVE,
                           certified_candidates,
                           contaminated_candidates,
                           viewpoint_ijk,
                           true,
                           selected_label);
    RCLCPP_INFO(node_handle_->get_logger(),
                "[Observe] VP candidate set for unknown [%d,%d,%d]: certified %zu, contaminated %zu, selected=%s%s",
                target_unknown.x(), target_unknown.y(), target_unknown.z(),
                certified_candidates.size(), contaminated_candidates.size(),
                selected_label == ReachLabel::CERTIFIED ? "certified" : "contaminated",
                failsafe_active ? " (failsafe seeded)" : "");

    if (selected_label == ReachLabel::CONTAMINATED)
    {
        SubgoalNode *parent = &leaf;
        const bool parent_has_search_start = leaf.active_plan_has_search_start;
        const openvdb::Coord parent_search_start_ijk = leaf.active_plan_search_start_ijk;
        const openvdb::Vec3d parent_search_start_world = leaf.active_plan_search_start_world;
        // Snapshot the observe lock: createOrUpdateChildSubgoal clears the
        // parent's execution state (and thus the lock); the child inherits
        // it and the parent keeps it as data for when it becomes leaf again.
        const bool lock_valid = leaf.has_locked_unknown;
        const openvdb::Coord lock_u = leaf.locked_unknown;
        const HitpointKey lock_hit = leaf.locked_unknown_hitpoint;
        const auto lock_failed = leaf.observe_failed_voxels;
        const int lock_failures = leaf.observe_failure_count;
        SubgoalNode *child = createOrUpdateChildSubgoal(
            *parent, SubgoalType::OBSERVE_UNKNOWN, hit_pt, viewpoint, target_yaw);
        for (SubgoalNode *n : {parent, child})
        {
            n->has_locked_unknown = lock_valid;
            n->locked_unknown = lock_u;
            n->locked_unknown_hitpoint = lock_hit;
            n->observe_failed_voxels = lock_failed;
            n->observe_failure_count = lock_failures;
        }
        child->active_plan_hit_point = hit_pt;
        child->active_plan_has_search_start = parent_has_search_start;
        child->active_plan_search_start_ijk =
            parent_has_search_start ? parent_search_start_ijk : openvdb::Coord(0, 0, 0);
        child->active_plan_search_start_world =
            parent_has_search_start ? parent_search_start_world : openvdb::Vec3d(0, 0, 0);
        if (parent_has_search_start)
        {
            child->has_certified_guidance_start = true;
            child->certified_guidance_start_ijk = parent_search_start_ijk;
            child->certified_guidance_start_world = parent_search_start_world;
        }
        // Freeze the guidance origin (inherit subtree anchor, else the
        // creation-time plan start / lookahead).
        child->has_guidance_origin = true;
        if (parent->has_guidance_origin)
        {
            child->guidance_origin_ijk = parent->guidance_origin_ijk;
            child->guidance_origin_world = parent->guidance_origin_world;
        }
        else
        {
            const openvdb::Vec3d origin_w(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z());
            child->guidance_origin_world = origin_w;
            child->guidance_origin_ijk =
                openvdb::Coord::round(grid_tf->worldToIndex(origin_w));
        }

        if (!planGuidanceToSubgoal(*child, plan_start_pos))
        {
            clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
            clearObserveFrustumVis();
            markViewpointRejected(*parent, viewpoint_ijk);
            stashChildNoGood(*parent, *child);
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Subgoal] Label-contaminated observe fallback found viewpoint [%.2f, %.2f, %.2f] but no optimistic guidance path from robot",
                        viewpoint.x(), viewpoint.y(), viewpoint.z());
            pruneSubgoalChildren(*parent);
            return false;
        }

        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Subgoal] Created OBSERVE_UNKNOWN child node %llu for unknown [%d,%d,%d] "
                    "from anchor [%.2f, %.2f, %.2f] to viewpoint [%.2f, %.2f, %.2f] "
                    "(label-aware %.1f ms, guidance %zu pts; "
                    "child will clear along this guidance before approaching the goal)",
                    static_cast<unsigned long long>(child->node_id),
                    target_unknown.x(), target_unknown.y(), target_unknown.z(),
                    hit_pt.x(), hit_pt.y(), hit_pt.z(),
                    viewpoint.x(), viewpoint.y(), viewpoint.z(),
                    obs_ms, child->guidance_path_short.size());

        publishObservePlanVis(obs_path, obs_path, viewpoint, target_unknown_world, layer);
        child->status = SubgoalStatus::ACTIVE;
        return true;
    }

    std::vector<openvdb::Vec3d> obs_short;
    observe_astar_.pathShortenStrict(obs_short);
    pruneSubgoalChildren(leaf);

    RCLCPP_WARN(node_handle_->get_logger(),
                "[Observe] From safe edge [%.2f,%.2f,%.2f], found viewpoint "
                "[%.2f,%.2f,%.2f] yaw=%.2f for unknown [%d,%d,%d] "
                "(certified path %zu pts, short %zu pts, %.1fms)",
                hit_pt.x(), hit_pt.y(), hit_pt.z(),
                viewpoint.x(), viewpoint.y(), viewpoint.z(),
                target_yaw,
                target_unknown.x(), target_unknown.y(), target_unknown.z(),
                obs_path.size(), obs_short.size(), obs_ms);

    publishObservePlanVis(obs_path, obs_short, viewpoint, target_unknown_world, layer);

    openvdb::Coord plan_start_ijk = openvdb::Coord::round(
        grid_tf->worldToIndex(openvdb::Vec3d(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z())));

    auto t1 = std::chrono::steady_clock::now();
    int connect_result = connect_astar_.search(plan_start_ijk, viewpoint_ijk);
    double connect_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t1)
                            .count();

    if (layer == VisLayer::ROOT ? (pub_connect_debug_tree_->get_subscription_count() > 0)
                                : (pub_subgoal_connect_debug_tree_->get_subscription_count() > 0))
    {
        auto tree_mk = connect_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::CONNECT, layer);
    }

    if (connect_result != Astar::PATH_FOUND)
    {
        clearCandidateSetVis(CandidateSetVisKind::OBSERVE);
        clearObserveFrustumVis();
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Connect] No path from plan-start [%.2f,%.2f,%.2f] to viewpoint "
                    "[%.2f,%.2f,%.2f] (%.1fms)",
                    plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z(),
                    viewpoint.x(), viewpoint.y(), viewpoint.z(),
                    connect_ms);
        return false;
    }

    std::vector<openvdb::Vec3d> connect_path = connect_astar_.getPathAstar();
    std::vector<openvdb::Vec3d> connect_short;
    connect_astar_.pathShortenStrict(connect_short);
    if (connect_short.empty())
    {
        connect_short = connect_path;
    }

    RCLCPP_WARN(node_handle_->get_logger(),
                "[Connect] Found path to observe viewpoint "
                "[%.2f,%.2f,%.2f] yaw=%.2f (raw %zu, short %zu, %.1fms)",
                viewpoint.x(), viewpoint.y(), viewpoint.z(),
                target_yaw, connect_path.size(), connect_short.size(), connect_ms);

    publishConnectPlanVis(connect_path, connect_short, layer);
    clearSecondaryFovPlanVis(layer);
    RCLCPP_INFO(node_handle_->get_logger(),
                "[Observe] Preview stitch disabled; executing precise observe viewpoint only");

    // Always route through the optimizer-aware publisher. When the connect A*
    // result is degenerate (<=1 point, i.e. plan_start already at the observe
    // viewpoint) we synthesize a 2-point [plan_start, viewpoint] path so the
    // optimizer / linear interpolator can still produce a real trajectory
    // (mostly a yaw change). This avoids maintaining a separate yaw-in-place
    // publishing branch with its own reach-tolerance semantics.
    std::vector<openvdb::Vec3d> exec_path = connect_short;
    if (exec_path.size() < 2)
    {
        exec_path = {openvdb::Vec3d(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z()),
                     viewpoint};
    }
    interpolateAndPublishTrajectory(exec_path, plan_start_yaw, target_yaw);

    leaf.active_viewpoint = viewpoint;
    leaf.active_anchor_seed = viewpoint;
    leaf.active_target_yaw = target_yaw;
    leaf.active_plan_hit_point = hit_pt;
    leaf.has_active_plan = true;
    leaf.has_preview_plan = false;
    leaf.has_receding_preview_state = false;
    leaf.goal_traj_published = true;
    resetObserveArrivalCounters(leaf);
    needs_escape_corridor_ = true;
    return true;
}

bool FFAPlannerNode::isStrictFreeVoxel(const openvdb::Coord &coord) const
{
    int inflate_val = 0;
    const bool active = map_manager_->query_is_inflated_at_index(coord, inflate_val);
    if (!active)
    {
        return true;
    }

    const bool has_occ = (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
    const bool has_frontier = inflate_val >= VDBMap::FRONTIER_INFLATION_DELTA;
    return !has_occ && !has_frontier;
}

openvdb::Coord FFAPlannerNode::chooseTargetCentricSearchStart(const openvdb::Vec3d &target_hit,
                                                              const openvdb::Vec3d &search_anchor) const
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord target_ijk = openvdb::Coord::round(tf->worldToIndex(target_hit));
    if (isStrictFreeVoxel(target_ijk))
    {
        return target_ijk;
    }

    openvdb::Vec3d dir = search_anchor - target_hit;
    const double dir_norm = dir.length();
    if (dir_norm <= 1e-6)
    {
        return target_ijk;
    }

    dir /= dir_norm;
    const double voxel_size = tf->voxelSize()[0];
    const int max_steps = std::max(2, static_cast<int>(std::ceil(dir_norm / voxel_size)) + 1);
    openvdb::Coord last_ijk = target_ijk;
    for (int step = 1; step <= max_steps; ++step)
    {
        const openvdb::Vec3d sample_world = target_hit + dir * (voxel_size * static_cast<double>(step));
        const openvdb::Coord sample_ijk = openvdb::Coord::round(tf->worldToIndex(sample_world));
        if (sample_ijk == last_ijk)
        {
            continue;
        }
        last_ijk = sample_ijk;
        if (isStrictFreeVoxel(sample_ijk))
        {
            return sample_ijk;
        }
    }

    return last_ijk;
}

bool FFAPlannerNode::selectGoalFromSet(const openvdb::Vec3d &start_world,
                                       const std::vector<SearchGoalCandidate> &goal_set,
                                       openvdb::Vec3d &selected_goal_world,
                                       double *elapsed_ms_out,
                                       std::string *failure_reason_out)
{
    if (goal_set.empty())
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "empty goal set";
        }
        return false;
    }

    std::vector<openvdb::Coord> goal_coords;
    goal_coords.reserve(goal_set.size());
    for (const SearchGoalCandidate &candidate : goal_set)
    {
        goal_coords.push_back(candidate.coord);
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord start_ijk = openvdb::Coord::round(tf->worldToIndex(start_world));

    connect_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = connect_astar_.searchToGoalSet(start_ijk, goal_coords);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
    if (elapsed_ms_out)
    {
        *elapsed_ms_out = elapsed_ms;
    }

    if (result != Astar::PATH_FOUND)
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "no free path to certified goal set";
        }
        return false;
    }

    const std::vector<openvdb::Vec3d> connect_path = connect_astar_.getPathAstar();
    if (connect_path.empty())
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "empty connect path to goal set";
        }
        return false;
    }

    selected_goal_world = connect_path.back();
    return true;
}

bool FFAPlannerNode::selectGoalFromSeeds(const openvdb::Vec3d &start_world,
                                         const std::vector<SearchGoalCandidate> &goal_set,
                                         double graze_penalty_vox,
                                         const std::unordered_set<openvdb::Coord, CoordHash> &excluded,
                                         openvdb::Vec3d &selected_goal_world,
                                         double *elapsed_ms_out,
                                         std::string *failure_reason_out)
{
    std::vector<ConnectAstar::GoalSeed> seeds;
    seeds.reserve(goal_set.size());
    size_t n_grazing = 0;
    for (const SearchGoalCandidate &candidate : goal_set)
    {
        if (excluded.count(candidate.coord) > 0)
        {
            continue;
        }
        ConnectAstar::GoalSeed seed;
        seed.coord = candidate.coord;
        seed.init_cost = candidate.grazing ? std::max(0.0, graze_penalty_vox) : 0.0;
        n_grazing += candidate.grazing ? 1 : 0;
        seeds.push_back(seed);
    }
    if (seeds.empty())
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "goal set empty after exclusion";
        }
        return false;
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord start_ijk = openvdb::Coord::round(tf->worldToIndex(start_world));

    connect_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = connect_astar_.searchToGoalSet(start_ijk, seeds);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
    if (elapsed_ms_out)
    {
        *elapsed_ms_out = elapsed_ms;
    }
    RCLCPP_INFO(node_handle_->get_logger(),
                "[Observe][Failsafe] seeded connect over %zu candidates (%zu grazing, penalty %.1f vox): %s",
                seeds.size(), n_grazing, graze_penalty_vox,
                result == Astar::PATH_FOUND ? "ok" : "no path");

    if (result != Astar::PATH_FOUND)
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "no free path to seeded goal set";
        }
        return false;
    }

    const std::vector<openvdb::Vec3d> connect_path = connect_astar_.getPathAstar();
    if (connect_path.empty())
    {
        if (failure_reason_out)
        {
            *failure_reason_out = "empty connect path to seeded goal set";
        }
        return false;
    }

    selected_goal_world = connect_path.back();
    return true;
}

bool FFAPlannerNode::selectLowestCostGoalFromSet(const std::vector<SearchGoalCandidate> &goal_set,
                                                 openvdb::Coord &selected_goal_ijk) const
{
    if (goal_set.empty())
    {
        return false;
    }

    const SearchGoalCandidate *best = &goal_set.front();
    for (const SearchGoalCandidate &candidate : goal_set)
    {
        if (candidate.g_score < best->g_score)
        {
            best = &candidate;
        }
    }
    selected_goal_ijk = best->coord;
    return true;
}

FFAPlannerNode::HitpointKey
FFAPlannerNode::canonicalizeHitpoint(const openvdb::Vec3d &hit_pt) const
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    return HitpointKey{openvdb::Coord::round(tf->worldToIndex(hit_pt))};
}

bool FFAPlannerNode::isViewpointRejected(const SubgoalNode &context,
                                         const openvdb::Coord &viewpoint_ijk) const
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    return context.rejected_viewpoint_voxels.find(viewpoint_ijk) !=
           context.rejected_viewpoint_voxels.end();
}

bool FFAPlannerNode::markViewpointRejected(SubgoalNode &context,
                                           const openvdb::Coord &viewpoint_ijk)
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    return context.rejected_viewpoint_voxels.insert(viewpoint_ijk).second;
}

bool FFAPlannerNode::markParentViewpointRejected(SubgoalNode &child)
{
    if (!root_subgoal_)
    {
        return false;
    }

    SubgoalNode *parent = findParentSubgoalByChildId(*root_subgoal_, child.node_id);
    if (!parent)
    {
        return false;
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord viewpoint_ijk =
        openvdb::Coord::round(tf->worldToIndex(child.goal_pos));

    const bool inserted = markViewpointRejected(*parent, viewpoint_ijk);
    stashChildNoGood(*parent, child);
    return inserted;
}

void FFAPlannerNode::stashChildNoGood(SubgoalNode &parent, const SubgoalNode &child)
{
    const HitpointKey dependency = canonicalizeHitpoint(child.anchor_point);
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    SubgoalNode::PendingNoGood &pending = parent.pending_no_good_by_hitpoint[dependency];
    pending.rejected_viewpoint_voxels.insert(child.rejected_viewpoint_voxels.begin(),
                                            child.rejected_viewpoint_voxels.end());
    pending.exhausted_hitpoints.insert(child.exhausted_hitpoints.begin(),
                                       child.exhausted_hitpoints.end());
}

void FFAPlannerNode::propagatePendingNoGoodForHitpoint(SubgoalNode &context,
                                                       const HitpointKey &hitpoint)
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    auto it = context.pending_no_good_by_hitpoint.find(hitpoint);
    if (it == context.pending_no_good_by_hitpoint.end())
    {
        return;
    }

    context.rejected_viewpoint_voxels.insert(it->second.rejected_viewpoint_voxels.begin(),
                                            it->second.rejected_viewpoint_voxels.end());
    context.exhausted_hitpoints.insert(it->second.exhausted_hitpoints.begin(),
                                       it->second.exhausted_hitpoints.end());
    context.pending_no_good_by_hitpoint.erase(it);
}

bool FFAPlannerNode::isHitpointKeyCleared(const HitpointKey &key) const
{
    int val = 0;
    const bool active = map_manager_->query_is_inflated_at_index(key.coord, val);
    return !(active && val > 0);
}

// Dependency-triggered invalidation (paper Sec. IV-D). Rejection entries are
// failure certificates derived while some recorded hitpoint was blocked; once
// that hitpoint is observed cleared, the entries of the context that recorded
// it and of all descendant contexts are stale. Stashed child memory keyed by
// a cleared hitpoint is dropped individually; if one of the context's own
// exhausted hitpoints is cleared, the context memory is reset as a whole and
// the descendants are pruned (over-invalidation only costs recomputation).
bool FFAPlannerNode::invalidateClearedRejections(SubgoalNode &node)
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);

    for (auto it = node.pending_no_good_by_hitpoint.begin();
         it != node.pending_no_good_by_hitpoint.end();)
    {
        if (isHitpointKeyCleared(it->first))
        {
            it = node.pending_no_good_by_hitpoint.erase(it);
        }
        else
        {
            ++it;
        }
    }

    bool own_hitpoint_cleared = false;
    for (const HitpointKey &hitpoint : node.exhausted_hitpoints)
    {
        if (isHitpointKeyCleared(hitpoint))
        {
            own_hitpoint_cleared = true;
            break;
        }
    }
    if (!own_hitpoint_cleared)
    {
        return false;
    }

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Refute] An exhausted hitpoint of context node %llu was cleared by new "
                "observations; resetting the context rejection memory and pruning descendants",
                static_cast<unsigned long long>(node.node_id));

    node.rejected_viewpoint_voxels.clear();
    node.exhausted_hitpoints.clear();
    node.pending_no_good_by_hitpoint.clear();
    pruneSubgoalChildren(node);
    clearNodeExecutionState(node);
    return true;
}

bool FFAPlannerNode::hasRefutationMemory() const
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    const SubgoalNode *node = root_subgoal_.get();
    while (node)
    {
        if (!node->rejected_viewpoint_voxels.empty() ||
            !node->exhausted_hitpoints.empty() ||
            !node->pending_no_good_by_hitpoint.empty())
        {
            return true;
        }
        node = node->child.get();
    }
    return false;
}

void FFAPlannerNode::markHitpointExhausted(SubgoalNode &leaf,
                                           const HitpointKey &hitpoint,
                                           const char *reason)
{
    bool inserted = false;
    {
        std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
        inserted = leaf.exhausted_hitpoints.insert(hitpoint).second;
    }

    propagatePendingNoGoodForHitpoint(leaf, hitpoint);

    if (!inserted)
    {
        return;
    }

    RCLCPP_WARN(node_handle_->get_logger(),
                "[Refute] Exhausted hitpoint class [%d,%d,%d] in context node %llu (%s); "
                "future guidance in this context will avoid it",
                hitpoint.coord.x(), hitpoint.coord.y(), hitpoint.coord.z(),
                static_cast<unsigned long long>(leaf.node_id),
                reason ? reason : "unspecified");

    leaf.has_guidance_progress_floor = false;
    leaf.guidance_progress_floor = openvdb::Vec3d(0, 0, 0);
    pruneSubgoalChildren(leaf);
    clearNodeExecutionState(leaf);
    occlusion_analyzer_.reset();
    if (leaf.type == SubgoalType::ROOT_GOAL)
    {
        need_global_replan_.store(true);
        wakeGlobalPlanner();
    }
    else
    {
        leaf.guidance_path_raw.clear();
        leaf.guidance_path_short.clear();
    }
}

std::unordered_set<openvdb::Coord, CoordHash>
FFAPlannerNode::buildRejectedHitpointBlockedSet(const SubgoalNode *context) const
{
    std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
    std::unordered_set<openvdb::Coord, CoordHash> blocked;
    if (!context)
    {
        return blocked;
    }
    for (const HitpointKey &hitpoint : context->exhausted_hitpoints)
    {
        blocked.insert(hitpoint.coord);
    }
    return blocked;
}

std::unordered_set<openvdb::Coord, CoordHash>
FFAPlannerNode::buildGuidanceBlockedSet(const SubgoalNode &node) const
{
    std::unordered_set<openvdb::Coord, CoordHash> blocked = buildRejectedHitpointBlockedSet(&node);
    for (const HitpointKey &hitpoint : node.ancestor_hitpoints)
    {
        blocked.insert(hitpoint.coord);
    }
    return blocked;
}

FFAPlannerNode::FovCandidateState
FFAPlannerNode::classifyCandidateState(const openvdb::Vec3d &viewpoint) const
{
    int inflate_val = 0;
    const Eigen::Vector3d viewpoint_world(viewpoint.x(), viewpoint.y(), viewpoint.z());
    const bool active = map_manager_->query_is_inflated_at_world(viewpoint_world, inflate_val);
    if (!active)
    {
        return FovCandidateState::CERTIFIED;
    }

    const bool has_occ = (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
    const bool has_frontier = inflate_val >= VDBMap::FRONTIER_INFLATION_DELTA;
    if (has_occ)
    {
        return FovCandidateState::INVALID;
    }
    if (has_frontier)
    {
        return FovCandidateState::CONTAMINATED;
    }
    return FovCandidateState::CERTIFIED;
}

bool FFAPlannerNode::findGuidanceLookAheadTarget(const std::vector<openvdb::Vec3d> &path,
                                                 size_t anchor_idx,
                                                 const openvdb::Vec3d &anchor_pt,
                                                 double look_ahead_dist,
                                                 openvdb::Vec3d &target_out) const
{
    if (path.empty())
    {
        return false;
    }

    if (look_ahead_dist <= 1e-3)
    {
        target_out = anchor_pt;
        return true;
    }

    openvdb::Vec3d prev = anchor_pt;
    for (size_t i = std::min(anchor_idx + 1, path.size() - 1); i < path.size(); ++i)
    {
        const openvdb::Vec3d &curr = path[i];
        const double seg_len = (curr - prev).length();
        if (seg_len < 1e-6)
        {
            prev = curr;
            continue;
        }

        if (look_ahead_dist <= seg_len)
        {
            const double ratio = look_ahead_dist / seg_len;
            target_out = prev + (curr - prev) * ratio;
            return true;
        }

        look_ahead_dist -= seg_len;
        prev = curr;
    }

    target_out = path.back();
    return true;
}

bool FFAPlannerNode::buildTargetCentricFovPlan(const SubgoalNode &context,
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
                                               CandidateSetSnapshot *candidate_snapshot_out)
{
    if (candidate_snapshot_out)
    {
        *candidate_snapshot_out = CandidateSetSnapshot{};
    }

    auto grid_tf = map_manager_->get_grid_transform();
    const openvdb::Coord target_ijk = openvdb::Coord::round(grid_tf->worldToIndex(target_hit));

    fov_astar_.setActiveClearProfile(profile_kind);
    fov_astar_.setTarget(target_hit);
    fov_astar_.setSearchAnchor(search_anchor);
    fov_astar_.setAllowSoftFrontierTraversal(allow_contaminated_traversal);
    fov_astar_.clearEscapeCorridor();
    fov_astar_.setGoalCandidatePredicate(
        [this, &context](const SearchGoalCandidate &candidate, ReachLabel label)
        {
            if (label == ReachLabel::CERTIFIED)
            {
                return true;
            }
            return !isViewpointRejected(context, candidate.coord);
        });

    fov_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = fov_astar_.search(search_start_ijk, target_ijk);
    elapsed_ms_out = std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
    fov_astar_.clearGoalCandidatePredicate();

    if (publish_debug_tree && vis_fov_debug_tree_)
    {
        auto tree_mk = fov_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::FOV, layer);
    }

    if (result != Astar::PATH_FOUND)
    {
        raw_path_out.clear();
        short_path_out.clear();
        viewpoint_out = openvdb::Vec3d(0, 0, 0);
        target_yaw_out = 0.0;
        viewpoint_state_out = FovCandidateState::INVALID;
        return false;
    }

    const auto &certified_candidates = fov_astar_.getCertifiedGoalCandidates();
    const auto &contaminated_candidates = fov_astar_.getContaminatedGoalCandidates();

    bool selected_candidate_path = false;
    ReachLabel selected_label = fov_astar_.getLastPathLabel();
    if (!certified_candidates.empty())
    {
        openvdb::Vec3d selected_goal_world;
        if (selectGoalFromSet(search_anchor, certified_candidates, selected_goal_world))
        {
            const openvdb::Coord selected_goal_ijk =
                openvdb::Coord::round(grid_tf->worldToIndex(selected_goal_world));
            if (fov_astar_.extractPathToGoal(selected_goal_ijk, ReachLabel::CERTIFIED))
            {
                selected_candidate_path = true;
                selected_label = ReachLabel::CERTIFIED;
            }
        }
    }
    if (!selected_candidate_path && !contaminated_candidates.empty())
    {
        openvdb::Coord selected_goal_ijk;
        if (selectLowestCostGoalFromSet(contaminated_candidates, selected_goal_ijk) &&
            fov_astar_.extractPathToGoal(selected_goal_ijk, ReachLabel::CONTAMINATED))
        {
            selected_candidate_path = true;
            selected_label = ReachLabel::CONTAMINATED;
        }
    }

    raw_path_out = fov_astar_.getPathAstar();
    if (raw_path_out.empty())
    {
        short_path_out.clear();
        viewpoint_out = openvdb::Vec3d(0, 0, 0);
        target_yaw_out = 0.0;
        viewpoint_state_out = FovCandidateState::INVALID;
        return false;
    }

    fov_astar_.pathShortenStrict(short_path_out);
    if (short_path_out.empty())
    {
        short_path_out = raw_path_out;
    }

    viewpoint_out = raw_path_out.back();
    const double dx = target_hit.x() - viewpoint_out.x();
    const double dy = target_hit.y() - viewpoint_out.y();
    target_yaw_out = std::atan2(dy, dx) - fov_cfg_.body_yaw;
    const openvdb::Coord viewpoint_ijk = openvdb::Coord::round(grid_tf->worldToIndex(viewpoint_out));
    if (selected_label == ReachLabel::CONTAMINATED &&
        isViewpointRejected(context, viewpoint_ijk))
    {
        raw_path_out.clear();
        short_path_out.clear();
        viewpoint_out = openvdb::Vec3d(0, 0, 0);
        target_yaw_out = 0.0;
        viewpoint_state_out = FovCandidateState::INVALID;
        return false;
    }

    if (selected_label == ReachLabel::CERTIFIED)
    {
        viewpoint_state_out = FovCandidateState::CERTIFIED;
    }
    else
    {
        viewpoint_state_out = classifyCandidateState(viewpoint_out);
        if (viewpoint_state_out == FovCandidateState::CERTIFIED)
        {
            viewpoint_state_out = FovCandidateState::CONTAMINATED;
        }
    }
    const bool valid_viewpoint = (viewpoint_state_out != FovCandidateState::INVALID);
    if (valid_viewpoint && candidate_snapshot_out)
    {
        candidate_snapshot_out->certified_candidates = certified_candidates;
        candidate_snapshot_out->contaminated_candidates = contaminated_candidates;
        candidate_snapshot_out->selected_coord = viewpoint_ijk;
        candidate_snapshot_out->selected_label = selected_label;
        candidate_snapshot_out->has_selected = true;
    }
    return valid_viewpoint;
}

bool FFAPlannerNode::buildCertifiedAccessPath(const openvdb::Vec3d &start_world,
                                              const openvdb::Vec3d &goal_world,
                                              std::vector<openvdb::Vec3d> &raw_path_out,
                                              std::vector<openvdb::Vec3d> &short_path_out,
                                              double *elapsed_ms_out,
                                              std::string *failure_reason_out)
{
    auto grid_tf = map_manager_->get_grid_transform();
    const openvdb::Coord start_ijk = openvdb::Coord::round(grid_tf->worldToIndex(start_world));
    const openvdb::Coord goal_ijk = openvdb::Coord::round(grid_tf->worldToIndex(goal_world));

    connect_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = connect_astar_.search(start_ijk, goal_ijk);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
    if (elapsed_ms_out)
    {
        *elapsed_ms_out = elapsed_ms;
    }

    if (result != Astar::PATH_FOUND)
    {
        if (failure_reason_out)
        {
            *failure_reason_out = connect_astar_.getLastFailureReasonString();
        }
        raw_path_out.clear();
        short_path_out.clear();
        return false;
    }

    if (failure_reason_out)
    {
        failure_reason_out->clear();
    }

    raw_path_out = connect_astar_.getPathAstar();
    if (raw_path_out.empty())
    {
        short_path_out.clear();
        return false;
    }

    connect_astar_.pathShortenStrict(short_path_out);
    if (short_path_out.empty())
    {
        short_path_out = raw_path_out;
    }
    return true;
}

bool FFAPlannerNode::buildStartupEscapePrefix(const Eigen::Vector3d &plan_start_pos,
                                              double plan_start_yaw,
                                              std::vector<openvdb::Vec3d> &prefix_out,
                                              Eigen::Vector3d &virtual_start_out)
{
    prefix_out.clear();
    virtual_start_out = plan_start_pos;

    int start_inflate_val = 0;
    const bool start_inflated =
        map_manager_->query_is_inflated_at_world(plan_start_pos, start_inflate_val);
    if (!start_inflated)
    {
        startup_escape_completed_ = true;
        return false;
    }

    const bool start_has_occ = (start_inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
    if (start_has_occ)
    {
        return false;
    }

    prefix_out.emplace_back(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z());

    const double step = std::max(voxel_size_, 1.0e-2);
    const int max_steps = std::max(1, static_cast<int>(std::ceil(1.0 / step)));
    const double cos_yaw = std::cos(plan_start_yaw);
    const double sin_yaw = std::sin(plan_start_yaw);

    for (int i = 1; i <= max_steps; ++i)
    {
        const Eigen::Vector3d sample =
            plan_start_pos + Eigen::Vector3d(step * i * cos_yaw,
                                             step * i * sin_yaw,
                                             0.0);

        int inflate_val = 0;
        const bool active = map_manager_->query_is_inflated_at_world(sample, inflate_val);
        const bool has_occ = active && ((inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0);
        if (has_occ)
        {
            prefix_out.clear();
            virtual_start_out = plan_start_pos;
            return false;
        }

        prefix_out.emplace_back(sample.x(), sample.y(), sample.z());
        if (!active)
        {
            virtual_start_out = sample;
            return true;
        }
    }

    prefix_out.clear();
    virtual_start_out = plan_start_pos;
    return false;
}

bool FFAPlannerNode::buildOptimisticGuidancePath(const openvdb::Vec3d &start_world,
                                                 const openvdb::Vec3d &goal_world,
                                                 std::vector<openvdb::Vec3d> &raw_path_out,
                                                 std::vector<openvdb::Vec3d> &short_path_out,
                                                 double *elapsed_ms_out)
{
    auto grid_tf = map_manager_->get_grid_transform();
    const openvdb::Coord start_ijk = openvdb::Coord::round(grid_tf->worldToIndex(start_world));
    const openvdb::Coord goal_ijk = openvdb::Coord::round(grid_tf->worldToIndex(goal_world));

    std::unordered_set<openvdb::Coord, CoordHash> blocked;
    {
        std::lock_guard<std::recursive_mutex> lk(refute_state_mtx_);
        if (root_subgoal_)
        {
            for (const HitpointKey &hitpoint : root_subgoal_->exhausted_hitpoints)
            {
                blocked.insert(hitpoint.coord);
            }
        }
    }
    blocked.erase(start_ijk);
    subgoal_guidance_astar_.setExtraBlockedVoxels(blocked);
    subgoal_guidance_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = subgoal_guidance_astar_.search(start_ijk, goal_ijk);
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
    if (elapsed_ms_out)
    {
        *elapsed_ms_out = elapsed_ms;
    }

    if (result != Astar::PATH_FOUND)
    {
        subgoal_guidance_astar_.clearExtraBlockedVoxels();
        raw_path_out.clear();
        short_path_out.clear();
        return false;
    }

    raw_path_out = subgoal_guidance_astar_.getPathAstar();
    if (raw_path_out.empty())
    {
        subgoal_guidance_astar_.clearExtraBlockedVoxels();
        short_path_out.clear();
        return false;
    }

    subgoal_guidance_astar_.pathShorten(short_path_out);
    subgoal_guidance_astar_.clearExtraBlockedVoxels();
    if (short_path_out.empty())
    {
        short_path_out = raw_path_out;
    }
    return true;
}

std::vector<openvdb::Vec3d> FFAPlannerNode::prependPathPrefix(const std::vector<openvdb::Vec3d> &prefix,
                                                              const std::vector<openvdb::Vec3d> &suffix) const
{
    if (prefix.empty())
    {
        return suffix;
    }
    if (suffix.empty())
    {
        return prefix;
    }

    std::vector<openvdb::Vec3d> combined = prefix;
    const openvdb::Vec3d &tail = combined.back();
    const openvdb::Vec3d &head = suffix.front();
    const bool duplicate_join = (tail - head).length() <= 1.0e-6;
    combined.insert(combined.end(),
                    suffix.begin() + (duplicate_join ? 1 : 0),
                    suffix.end());
    return combined;
}

bool FFAPlannerNode::activateFovClearSubgoal(SubgoalNode &parent,
                                             const SafeEdgeHitResult *target_hit_context,
                                             const openvdb::Vec3d &target_hit,
                                             const openvdb::Vec3d &viewpoint,
                                             double target_yaw,
                                             const Eigen::Vector3d &plan_start_pos,
                                             double /*plan_start_yaw*/)
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord viewpoint_ijk = openvdb::Coord::round(tf->worldToIndex(viewpoint));
    if (isViewpointRejected(parent, viewpoint_ijk))
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Subgoal] Refusing rejected FOV_CLEAR viewpoint voxel [%d,%d,%d] "
                    "for hit [%.2f, %.2f, %.2f] -> viewpoint [%.2f, %.2f, %.2f]",
                    viewpoint_ijk.x(), viewpoint_ijk.y(), viewpoint_ijk.z(),
                    target_hit.x(), target_hit.y(), target_hit.z(),
                    viewpoint.x(), viewpoint.y(), viewpoint.z());
        return false;
    }

    SubgoalNode *child = createOrUpdateChildSubgoal(
        parent, SubgoalType::FOV_CLEAR, target_hit, viewpoint, target_yaw);
    if (target_hit_context && target_hit_context->has_search_start)
    {
        child->has_certified_guidance_start = true;
        child->certified_guidance_start_ijk = target_hit_context->search_start_ijk;
        child->certified_guidance_start_world = target_hit_context->search_start_world;
    }
    // Freeze the guidance origin: inherit the subtree anchor, or freeze the
    // current plan start (lookahead) at creation.
    child->has_guidance_origin = true;
    if (parent.has_guidance_origin)
    {
        child->guidance_origin_ijk = parent.guidance_origin_ijk;
        child->guidance_origin_world = parent.guidance_origin_world;
    }
    else
    {
        const openvdb::Vec3d origin_w(plan_start_pos.x(), plan_start_pos.y(), plan_start_pos.z());
        child->guidance_origin_world = origin_w;
        child->guidance_origin_ijk = openvdb::Coord::round(tf->worldToIndex(origin_w));
    }
    if (!planGuidanceToSubgoal(*child, plan_start_pos))
    {
        markViewpointRejected(parent, viewpoint_ijk);
        stashChildNoGood(parent, *child);
        return false;
    }
    if (target_hit_context)
    {
        setActivePlanHitFromSafeEdge(*child, *target_hit_context);
    }
    return true;
}

double FFAPlannerNode::pathLengthWorld(const std::vector<openvdb::Vec3d> &path) const
{
    double total = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        total += (path[i + 1] - path[i]).length();
    }
    return total;
}

openvdb::Vec3d FFAPlannerNode::samplePointAlongPath(const std::vector<openvdb::Vec3d> &path,
                                                    double distance_along_path) const
{
    if (path.empty())
    {
        return openvdb::Vec3d(0, 0, 0);
    }
    if (distance_along_path <= 0.0)
    {
        return path.front();
    }

    double remaining = distance_along_path;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const openvdb::Vec3d seg = path[i + 1] - path[i];
        const double seg_len = seg.length();
        if (seg_len < 1e-6)
        {
            continue;
        }
        if (remaining <= seg_len)
        {
            return path[i] + seg * (remaining / seg_len);
        }
        remaining -= seg_len;
    }

    return path.back();
}

std::vector<openvdb::Vec3d> FFAPlannerNode::trimPathFromPoint(
    const std::vector<openvdb::Vec3d> &path,
    const openvdb::Vec3d &point) const
{
    if (path.size() < 2)
    {
        return path;
    }

    double best_dist_sq = std::numeric_limits<double>::infinity();
    double best_progress = 0.0;
    double accum = 0.0;

    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const openvdb::Vec3d ab = path[i + 1] - path[i];
        const double seg_len = ab.length();
        if (seg_len < 1e-6)
        {
            continue;
        }
        double t = (point - path[i]).dot(ab) / (seg_len * seg_len);
        t = std::clamp(t, 0.0, 1.0);
        const openvdb::Vec3d proj = path[i] + ab * t;
        const double dist_sq = (point - proj).lengthSqr();
        if (dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_progress = accum + t * seg_len;
        }
        accum += seg_len;
    }

    std::vector<openvdb::Vec3d> trimmed;
    trimmed.push_back(samplePointAlongPath(path, best_progress));
    double seg_accum = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        seg_accum += (path[i + 1] - path[i]).length();
        if (seg_accum > best_progress + 1e-4)
        {
            trimmed.push_back(path[i + 1]);
        }
    }
    if (trimmed.size() == 1 && (trimmed.back() - path.back()).length() > 1e-6)
    {
        trimmed.push_back(path.back());
    }
    return trimmed;
}

std::vector<openvdb::Vec3d> FFAPlannerNode::prefixPathToPoint(
    const std::vector<openvdb::Vec3d> &path,
    const openvdb::Vec3d &point) const
{
    if (path.size() < 2)
    {
        return path;
    }

    double best_dist_sq = std::numeric_limits<double>::infinity();
    double best_progress = 0.0;
    double accum = 0.0;

    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const openvdb::Vec3d ab = path[i + 1] - path[i];
        const double seg_len = ab.length();
        if (seg_len < 1e-6)
        {
            continue;
        }
        double t = (point - path[i]).dot(ab) / (seg_len * seg_len);
        t = std::clamp(t, 0.0, 1.0);
        const openvdb::Vec3d proj = path[i] + ab * t;
        const double dist_sq = (point - proj).lengthSqr();
        if (dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_progress = accum + t * seg_len;
        }
        accum += seg_len;
    }

    std::vector<openvdb::Vec3d> prefix;
    prefix.push_back(path.front());
    double seg_accum = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const double seg_len = (path[i + 1] - path[i]).length();
        seg_accum += seg_len;
        if (seg_accum + 1.0e-4 < best_progress)
        {
            prefix.push_back(path[i + 1]);
            continue;
        }

        prefix.push_back(samplePointAlongPath(path, best_progress));
        break;
    }

    if (prefix.size() == 1 && (prefix.back() - point).length() > 1.0e-6)
    {
        prefix.push_back(point);
    }
    return prefix;
}

bool FFAPlannerNode::rayTraceClearKnownOcc(const openvdb::Vec3d &sensor_world,
                                           const openvdb::Vec3d &target_world) const
{
    // Exact Amanatides-Woo traversal (see ray_trace_util.h): the former
    // max-axis sampler could skip thin inclined slabs the segment crosses.
    auto grid_tf = map_manager_->get_grid_transform();
    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();

    return ffa::rayTraceVisitClear(
        *grid_tf, sensor_world, target_world,
        [&](const openvdb::Coord &vox) {
            float logodds = 0.0f;
            const bool active =
                map_manager_->query_log_odds_at_index(vox, logodds, occ_acc);
            return active && logodds > 0.0f;
        });
}

bool FFAPlannerNode::isTargetClearableFromAnchor(const openvdb::Vec3d &anchor_body_world,
                                                 const openvdb::Vec3d &target_world,
                                                 FovClearProfileKind profile_kind) const
{
    const double dx = target_world.x() - anchor_body_world.x();
    const double dy = target_world.y() - anchor_body_world.y();
    double psi_body = 0.0;
    if (dx * dx + dy * dy >= 1e-12)
    {
        psi_body = std::atan2(dy, dx) - fov_cfg_.body_yaw;
    }
    return isTargetClearableFromAnchorYaw(anchor_body_world, target_world,
                                          profile_kind, psi_body,
                                          /*check_horizontal=*/false);
}

bool FFAPlannerNode::isTargetClearableFromAnchorYaw(const openvdb::Vec3d &anchor_body_world,
                                                    const openvdb::Vec3d &target_world,
                                                    FovClearProfileKind profile_kind,
                                                    double psi_body,
                                                    bool check_horizontal) const
{
    const FovClearProfile &profile =
        (profile_kind == FovClearProfileKind::STANDARD) ? fov_standard_profile_ : fov_tight_profile_;

    // Worst-case horizontal half-extent of the world-axis-aligned inflation
    // box (45-degree azimuth diagonal); must stay in sync with
    // FovAstar::updateActiveProfileCache so preview/visibility checks agree
    // with the search goal test.
    const double inflate_rxy = std::sqrt(2.0) * profile.inflate_rxy;
    const double inflate_rz = profile.inflate_rz;
    const double margin_R = std::sqrt(inflate_rxy * inflate_rxy + inflate_rz * inflate_rz);
    const double safe_range = std::max(fov_cfg_.range - margin_R, 0.0);
    const double safe_min_range = std::max(fov_cfg_.fov_min_range + margin_R, 0.0);

    const double cos_psi = std::cos(psi_body);
    const double sin_psi = std::sin(psi_body);

    const Eigen::Vector3d t_bs(fov_cfg_.body_x, fov_cfg_.body_y, fov_cfg_.body_z);
    const Eigen::Matrix3d R_bs =
        (Eigen::AngleAxisd(fov_cfg_.body_yaw, Eigen::Vector3d::UnitZ()) *
         Eigen::AngleAxisd(fov_cfg_.body_pitch, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(fov_cfg_.body_roll, Eigen::Vector3d::UnitX()))
            .toRotationMatrix();

    const openvdb::Vec3d sensor_world(
        anchor_body_world.x() + t_bs.x() * cos_psi - t_bs.y() * sin_psi,
        anchor_body_world.y() + t_bs.x() * sin_psi + t_bs.y() * cos_psi,
        anchor_body_world.z() + t_bs.z());

    Eigen::Matrix3d R_yaw;
    R_yaw << cos_psi, -sin_psi, 0,
        sin_psi, cos_psi, 0,
        0, 0, 1;
    const Eigen::Matrix3d R_ws = R_yaw * R_bs;

    const Eigen::Vector3d d_w(target_world.x() - sensor_world.x(),
                              target_world.y() - sensor_world.y(),
                              target_world.z() - sensor_world.z());
    const Eigen::Vector3d d_s = R_ws.transpose() * d_w;

    const double d_h = d_s.x();
    const double d_norm = d_s.norm();
    const double theta_1 = std::atan2(d_s.z() + inflate_rz, d_h - inflate_rxy);
    const double theta_2 = std::atan2(d_s.z() - inflate_rz, d_h - inflate_rxy);

    const bool range_ok = (d_norm < safe_range) && (safe_range > 0.0);
    const bool min_range_ok = d_norm > safe_min_range;
    const bool theta1_ok = theta_1 < fov_cfg_.theta_u;
    const bool theta2_ok = theta_2 > fov_cfg_.theta_d;

    if (!(range_ok && min_range_ok && theta1_ok && theta2_ok))
    {
        return false;
    }

    if (check_horizontal)
    {
        // With a caller-fixed yaw the target is no longer on the optical
        // axis; require azimuth containment. Default bound reuses theta_u,
        // conservative for sensors whose horizontal FOV >= vertical FOV.
        const double theta_h = (fov_theta_h_ > 0.0) ? fov_theta_h_ : fov_cfg_.theta_u;
        const double az = std::atan2(std::abs(d_s.y()), d_h);
        if (!(az < theta_h))
        {
            return false;
        }
    }

    return rayTraceClearKnownOcc(sensor_world, target_world);
}

bool FFAPlannerNode::buildVisibilityAnchorCell(const openvdb::Vec3d &witness_viewpoint,
                                               const openvdb::Vec3d &target_world,
                                               traj_optimizer::VisibilityCell &cell_out) const
{
    VisibilityCellBuilder::Config config;
    config.roi_xy = visibility_cell_roi_xy_;
    config.roi_z = visibility_cell_roi_z_;
    config.erode_iters = visibility_cell_erode_iters_;
    config.min_radius = visibility_cell_min_radius_;
    config.voxel_size = voxel_size_;
    config.body_sensor_yaw = fov_cfg_.body_yaw;
    config.shrink_m = visibility_cell_shrink_m_;
    config.max_carve_iters = visibility_cell_max_carve_;

    VisibilityCellBuilder builder(
        map_manager_,
        config,
        [this](const openvdb::Vec3d &anchor_body_world,
               const openvdb::Vec3d &target_world_cb,
               FovClearProfileKind profile_kind)
        {
            return isTargetClearableFromAnchor(anchor_body_world,
                                               target_world_cb,
                                               profile_kind);
        },
        node_handle_->get_logger());
    if (visibility_cell_fixed_yaw_)
    {
        builder.setFixedYawValidator(
            [this](const openvdb::Vec3d &anchor_body_world,
                   const openvdb::Vec3d &target_world_cb,
                   FovClearProfileKind profile_kind,
                   double psi_body)
            {
                return isTargetClearableFromAnchorYaw(anchor_body_world,
                                                      target_world_cb,
                                                      profile_kind,
                                                      psi_body,
                                                      /*check_horizontal=*/true);
            });
    }
    return builder.build(witness_viewpoint, target_world, cell_out);
}

double FFAPlannerNode::yawTowardTarget(const openvdb::Vec3d &from,
                                       const openvdb::Vec3d &target) const
{
    const double dx = target.x() - from.x();
    const double dy = target.y() - from.y();
    if (dx * dx + dy * dy <= 1.0e-6)
    {
        return 0.0;
    }
    return std::atan2(dy, dx) - fov_cfg_.body_yaw;
}

double FFAPlannerNode::projectProgressAlongPath(
    const std::vector<openvdb::Vec3d> &path,
    const openvdb::Vec3d &point) const
{
    if (path.size() < 2)
    {
        return 0.0;
    }

    double best_dist_sq = std::numeric_limits<double>::infinity();
    double best_progress = 0.0;
    double accum = 0.0;
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        const openvdb::Vec3d ab = path[i + 1] - path[i];
        const double seg_len = ab.length();
        if (seg_len < 1.0e-6)
        {
            continue;
        }

        double t = (point - path[i]).dot(ab) / (seg_len * seg_len);
        t = std::clamp(t, 0.0, 1.0);
        const openvdb::Vec3d proj = path[i] + ab * t;
        const double dist_sq = (point - proj).lengthSqr();
        if (dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_progress = accum + t * seg_len;
        }
        accum += seg_len;
    }
    return best_progress;
}

std::vector<openvdb::Vec3d> FFAPlannerNode::localGuidancePath(
    const SubgoalNode &node) const
{
    if (!node.has_guidance_progress_floor)
    {
        return node.guidance_path_short;
    }

    return trimPathFromPoint(node.guidance_path_short,
                             node.guidance_progress_floor);
}

bool FFAPlannerNode::isAheadOfGuidanceProgressFloor(
    const SubgoalNode &node,
    const openvdb::Vec3d &point,
    double eps) const
{
    if (!node.has_guidance_progress_floor)
    {
        return true;
    }

    const double floor_progress =
        projectProgressAlongPath(node.guidance_path_short,
                                 node.guidance_progress_floor);
    const double point_progress =
        projectProgressAlongPath(node.guidance_path_short, point);
    return point_progress > floor_progress + eps;
}

void FFAPlannerNode::advanceGuidanceProgressFloor(
    SubgoalNode &node,
    const openvdb::Vec3d &point)
{
    if (!isAheadOfGuidanceProgressFloor(node, point, -1.0e-4))
    {
        return;
    }

    node.guidance_progress_floor = point;
    node.has_guidance_progress_floor = true;
}

bool FFAPlannerNode::buildPreviewProgressWindow(
    const openvdb::Vec3d &anchor_viewpoint,
    const std::vector<openvdb::Vec3d> &guidance_path,
    const openvdb::Vec3d &window_start,
    double lookahead_dist,
    FovClearProfileKind profile_kind,
    PreviewProgressWindow &window_out) const
{
    window_out = PreviewProgressWindow{};
    if (guidance_path.size() < 2)
    {
        return false;
    }

    std::vector<openvdb::Vec3d> tail = trimPathFromPoint(guidance_path, window_start);
    if (tail.size() < 2)
    {
        return false;
    }

    const double total_len = pathLengthWorld(tail);
    if (total_len <= 1.0e-6)
    {
        return false;
    }

    const double step = std::max(voxel_size_, 1.0e-3);
    const double window_len = std::min(std::max(lookahead_dist, step), total_len);
    const openvdb::Vec3d window_end = samplePointAlongPath(tail, window_len);
    std::vector<openvdb::Vec3d> window_path = prefixPathToPoint(tail, window_end);
    const double actual_len = pathLengthWorld(window_path);
    if (actual_len <= 1.0e-6)
    {
        return false;
    }

    window_out.window_start = window_path.front();
    window_out.window_end = window_path.back();
    window_out.predicted_clear_point = window_out.window_start;
    window_out.predicted_clear_distance = 0.0;
    window_out.valid = true;

    for (double d = step; d < actual_len; d += step)
    {
        window_out.samples.push_back(samplePointAlongPath(window_path, d));
        window_out.sample_progress.push_back(d);
    }
    window_out.samples.push_back(window_out.window_end);
    window_out.sample_progress.push_back(actual_len);

    for (size_t rev = window_out.samples.size(); rev > 0; --rev)
    {
        const size_t i = rev - 1;
        if (isTargetClearableFromAnchor(anchor_viewpoint,
                                        window_out.samples[i],
                                        profile_kind))
        {
            window_out.predicted_clear_point = window_out.samples[i];
            window_out.predicted_clear_distance = window_out.sample_progress[i];
            return true;
        }
    }

    return true;
}

bool FFAPlannerNode::buildPreviewProgressWindowToTarget(
    const openvdb::Vec3d &anchor_viewpoint,
    const std::vector<openvdb::Vec3d> &guidance_path,
    const openvdb::Vec3d &window_start,
    const openvdb::Vec3d &window_end,
    FovClearProfileKind profile_kind,
    PreviewProgressWindow &window_out) const
{
    window_out = PreviewProgressWindow{};
    if (guidance_path.size() < 2)
    {
        return false;
    }

    std::vector<openvdb::Vec3d> tail = trimPathFromPoint(guidance_path, window_start);
    if (tail.size() < 2)
    {
        return false;
    }

    const double end_progress = projectProgressAlongPath(tail, window_end);
    if (end_progress <= 1.0e-6)
    {
        return false;
    }

    std::vector<openvdb::Vec3d> window_path =
        prefixPathToPoint(tail, samplePointAlongPath(tail, end_progress));
    const double actual_len = pathLengthWorld(window_path);
    if (actual_len <= 1.0e-6)
    {
        return false;
    }

    window_out.window_start = window_path.front();
    window_out.window_end = window_path.back();
    window_out.predicted_clear_point = window_out.window_start;
    window_out.predicted_clear_distance = 0.0;
    window_out.valid = true;

    const double step = std::max(voxel_size_, 1.0e-3);
    for (double d = step; d < actual_len; d += step)
    {
        window_out.samples.push_back(samplePointAlongPath(window_path, d));
        window_out.sample_progress.push_back(d);
    }
    window_out.samples.push_back(window_out.window_end);
    window_out.sample_progress.push_back(actual_len);

    for (size_t rev = window_out.samples.size(); rev > 0; --rev)
    {
        const size_t i = rev - 1;
        if (isTargetClearableFromAnchor(anchor_viewpoint,
                                        window_out.samples[i],
                                        profile_kind))
        {
            window_out.predicted_clear_point = window_out.samples[i];
            window_out.predicted_clear_distance = window_out.sample_progress[i];
            return true;
        }
    }

    return true;
}

bool FFAPlannerNode::findFreshCertifiedWindowJump(
    const openvdb::Vec3d &anchor_viewpoint,
    const std::vector<openvdb::Vec3d> &guidance_path,
    const openvdb::Vec3d &window_start,
    openvdb::Vec3d &jump_point_out,
    std::vector<openvdb::Vec3d> *connect_path_out)
{
    if (connect_path_out)
    {
        connect_path_out->clear();
    }
    if (guidance_path.size() < 2)
    {
        return false;
    }

    std::vector<openvdb::Vec3d> tail = trimPathFromPoint(guidance_path, window_start);
    if (tail.size() < 2)
    {
        return false;
    }

    const double total_len = pathLengthWorld(tail);
    if (total_len <= 1.0e-6)
    {
        return false;
    }

    const double step = std::max(voxel_size_, 1.0e-3);
    const double window_len = std::min(std::max(fov_stitch_lookahead_dist_, step), total_len);

    std::vector<openvdb::Coord> goal_coords;
    goal_coords.reserve(static_cast<size_t>(std::ceil(window_len / step)) + 1);
    std::unordered_set<openvdb::Coord, CoordHash> seen_goals;

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    for (double d = step; d < window_len; d += step)
    {
        const openvdb::Coord sample_ijk =
            openvdb::Coord::round(tf->worldToIndex(samplePointAlongPath(tail, d)));
        if (seen_goals.insert(sample_ijk).second && isStrictFreeVoxel(sample_ijk))
        {
            goal_coords.push_back(sample_ijk);
        }
    }

    const openvdb::Coord end_ijk =
        openvdb::Coord::round(tf->worldToIndex(samplePointAlongPath(tail, window_len)));
    if (seen_goals.insert(end_ijk).second && isStrictFreeVoxel(end_ijk))
    {
        goal_coords.push_back(end_ijk);
    }

    if (goal_coords.empty())
    {
        return false;
    }

    const openvdb::Coord anchor_ijk =
        openvdb::Coord::round(tf->worldToIndex(anchor_viewpoint));
    const double old_timeout = connect_astar_.max_search_time_;
    connect_astar_.max_search_time_ = std::max(0.0, preview_connect_max_search_time_);
    connect_astar_.reset();
    const int result = connect_astar_.searchToGoalSet(anchor_ijk, goal_coords);
    connect_astar_.max_search_time_ = old_timeout;
    if (result != Astar::PATH_FOUND)
    {
        return false;
    }

    std::vector<openvdb::Vec3d> raw = connect_astar_.getPathAstar();
    if (raw.empty())
    {
        return false;
    }
    std::vector<openvdb::Vec3d> short_path;
    connect_astar_.pathShortenStrict(short_path);
    if (short_path.empty())
    {
        short_path = raw;
    }

    jump_point_out = raw.back();
    if (connect_path_out)
    {
        *connect_path_out = short_path;
    }
    return true;
}

bool FFAPlannerNode::buildCertifiedPreviewSegment(
    const SubgoalNode &,
    const openvdb::Vec3d &anchor_viewpoint,
    const std::vector<openvdb::Vec3d> &guidance_path,
    const openvdb::Vec3d &window_start,
    const openvdb::Coord &search_start_ijk,
    VisLayer layer,
    PreviewSegmentPlan &plan_out,
    const openvdb::Vec3d *fixed_window_end)
{
    plan_out = PreviewSegmentPlan{};

    PreviewProgressWindow search_window;
    const bool have_window =
        fixed_window_end
            ? buildPreviewProgressWindowToTarget(anchor_viewpoint,
                                                 guidance_path,
                                                 window_start,
                                                 *fixed_window_end,
                                                 FovClearProfileKind::STANDARD,
                                                 search_window)
            : buildPreviewProgressWindow(anchor_viewpoint,
                                         guidance_path,
                                         window_start,
                                         fov_stitch_lookahead_dist_,
                                         FovClearProfileKind::STANDARD,
                                         search_window);
    if (!have_window)
    {
        plan_out.failure_reason = "preview_window_unavailable";
        return false;
    }

    auto grid_tf = map_manager_->get_grid_transform();
    const openvdb::Coord target_ijk =
        openvdb::Coord::round(grid_tf->worldToIndex(search_window.window_end));

    fov_astar_.setActiveClearProfile(FovClearProfileKind::STANDARD);
    fov_astar_.setTarget(search_window.window_end);
    fov_astar_.setSearchAnchor(anchor_viewpoint);
    fov_astar_.setAllowSoftFrontierTraversal(false);
    fov_astar_.clearEscapeCorridor();
    fov_astar_.setPreviewProgressSamples(search_window.samples,
                                         search_window.sample_progress);
    fov_astar_.setGoalCandidatePredicate(
        [](const SearchGoalCandidate & /*candidate*/, ReachLabel label)
        {
            return label == ReachLabel::CERTIFIED;
        });

    fov_astar_.reset();
    const auto t0 = std::chrono::steady_clock::now();
    const int result = fov_astar_.search(search_start_ijk, target_ijk);
    plan_out.fov_search_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - t0)
                                 .count();
    fov_astar_.clearGoalCandidatePredicate();

    if (vis_fov_debug_tree_)
    {
        auto tree_mk = fov_astar_.getDebugTreeMarker(world_frame_id_);
        tree_mk.header.stamp = node_handle_->now();
        publishDebugTreeVis(tree_mk, DebugTreeKind::FOV, layer);
    }

    const std::vector<std::vector<SearchGoalCandidate>> progress_buckets =
        fov_astar_.getPreviewProgressBuckets();
    const PreviewProgressStats progress_stats =
        fov_astar_.getPreviewProgressStats();
    fov_astar_.clearPreviewProgressSamples();

    struct Incumbent
    {
        bool valid = false;
        openvdb::Coord coord{0, 0, 0};
        openvdb::Vec3d viewpoint{0, 0, 0};
        openvdb::Vec3d clear_point{0, 0, 0};
        double clear_distance = 0.0;
        double target_yaw = 0.0;
        std::vector<openvdb::Vec3d> access_raw;
        std::vector<openvdb::Vec3d> access_short;
        std::vector<openvdb::Vec3d> fov_path;
    } best;

    size_t connect_searches = 0;
    size_t connect_goal_candidates = 0;
    const openvdb::Coord anchor_ijk =
        openvdb::Coord::round(grid_tf->worldToIndex(anchor_viewpoint));

    struct ScopedConnectTimeout
    {
        ConnectAstar &astar;
        double old_timeout;

        ScopedConnectTimeout(ConnectAstar &astar_in, double timeout)
            : astar(astar_in), old_timeout(astar_in.max_search_time_)
        {
            astar.max_search_time_ = timeout;
        }

        ~ScopedConnectTimeout()
        {
            astar.max_search_time_ = old_timeout;
        }
    } scoped_preview_timeout(connect_astar_,
                             std::max(0.0, preview_connect_max_search_time_));

    for (int bucket_idx = static_cast<int>(progress_buckets.size()) - 1;
         bucket_idx >= 0; --bucket_idx)
    {
        const std::vector<SearchGoalCandidate> &bucket =
            progress_buckets[static_cast<size_t>(bucket_idx)];
        if (bucket.empty())
        {
            continue;
        }

        const openvdb::Vec3d &clear_point =
            search_window.samples[static_cast<size_t>(bucket_idx)];
        const double clear_distance =
            search_window.sample_progress[static_cast<size_t>(bucket_idx)];

        std::vector<openvdb::Coord> goal_coords;
        goal_coords.reserve(bucket.size());
        for (const SearchGoalCandidate &candidate : bucket)
        {
            if (candidate.coord == anchor_ijk)
            {
                continue;
            }
            goal_coords.push_back(candidate.coord);
        }
        if (goal_coords.empty())
        {
            continue;
        }
        connect_goal_candidates += goal_coords.size();

        connect_astar_.reset();
        ++connect_searches;
        const int connect_result = connect_astar_.searchToGoalSet(anchor_ijk, goal_coords);
        if (connect_result != Astar::PATH_FOUND)
        {
            plan_out.failure_reason = "preview_connect_failed_at_farthest_bucket";
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Preview] farthest progress bucket connect failed "
                        "(bucket=%d, clear_progress=%.2fm, candidates=%zu, "
                        "result=%d, preview_timeout=%.3fs); abandoning preview extension",
                        bucket_idx,
                        clear_distance,
                        goal_coords.size(),
                        connect_result,
                        connect_astar_.max_search_time_);
            return false;
        }

        std::vector<openvdb::Vec3d> access_raw = connect_astar_.getPathAstar();
        if (access_raw.empty())
        {
            plan_out.failure_reason = "preview_connect_empty_path_at_farthest_bucket";
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Preview] farthest progress bucket connect returned empty path "
                        "(bucket=%d, clear_progress=%.2fm, candidates=%zu); "
                        "abandoning preview extension",
                        bucket_idx,
                        clear_distance,
                        goal_coords.size());
            return false;
        }
        std::vector<openvdb::Vec3d> access_short;
        connect_astar_.pathShortenStrict(access_short);
        if (access_short.empty())
        {
            access_short = access_raw;
        }

        const openvdb::Vec3d selected_viewpoint = access_raw.back();
        const openvdb::Coord selected_coord =
            openvdb::Coord::round(grid_tf->worldToIndex(selected_viewpoint));
        const double selected_yaw =
            yawTowardTarget(selected_viewpoint, clear_point);

        std::vector<openvdb::Vec3d> fov_path;
        if (fov_astar_.extractPathToClosedState(selected_coord,
                                                ReachLabel::CERTIFIED))
        {
            fov_path = fov_astar_.getPathAstar();
        }
        if (fov_path.empty())
        {
            fov_path = {clear_point, selected_viewpoint};
        }

        best.valid = true;
        best.coord = selected_coord;
        best.viewpoint = selected_viewpoint;
        best.clear_point = clear_point;
        best.clear_distance = clear_distance;
        best.target_yaw = selected_yaw;
        best.access_raw = std::move(access_raw);
        best.access_short = std::move(access_short);
        best.fov_path = std::move(fov_path);
        break;
    }

    if (!best.valid)
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Preview] no connectable incumbent from search-time progress buckets "
                    "(search_result=%d, closed_certified=%zu, ray_checks=%zu, best_sample=%d, "
                    "connect_searches=%zu, connect_goal_candidates=%zu)",
                    result,
                    progress_stats.closed_certified,
                    progress_stats.ray_checks,
                    progress_stats.best_sample_index,
                    connect_searches,
                    connect_goal_candidates);
        plan_out.failure_reason = "preview_no_connectable_progress_candidate";
        return false;
    }

    plan_out.valid = true;
    plan_out.last_clearable_point = best.clear_point;
    plan_out.hypothetical_hit_point = search_window.window_end;
    plan_out.viewpoint = best.viewpoint;
    plan_out.stop_point = best.viewpoint;
    plan_out.target_yaw = best.target_yaw;
    plan_out.fov_path_raw = best.fov_path;
    plan_out.fov_path_short = best.fov_path;
    if (best.access_short.size() >= 2)
    {
        plan_out.exec_path = best.access_short;
    }
    else
    {
        plan_out.exec_path = best.access_raw;
    }
    plan_out.guidance_raw = best.access_raw;
    plan_out.guidance_short = plan_out.exec_path;
    plan_out.progress = search_window;
    plan_out.progress.predicted_clear_point = best.clear_point;
    plan_out.progress.predicted_clear_distance = best.clear_distance;
    RCLCPP_INFO(node_handle_->get_logger(),
                "[Preview] search-time incumbent selected progress %.2fm/%.2fm "
                "from bucketed certified VPs (closed_certified=%zu, ray_checks=%zu, "
                "best_sample=%d, connect_searches=%zu, connect_goal_candidates=%zu)",
                best.clear_distance,
                search_window.sample_progress.empty() ? 0.0 : search_window.sample_progress.back(),
                progress_stats.closed_certified,
                progress_stats.ray_checks,
                progress_stats.best_sample_index,
                connect_searches,
                connect_goal_candidates);
    return plan_out.exec_path.size() >= 2;
}

bool FFAPlannerNode::publishRecedingPreviewFromAnchor(
    SubgoalNode &leaf,
    VisLayer layer,
    const openvdb::Vec3d &plan_start_world,
    const std::vector<openvdb::Vec3d> &entry_path_short,
    const std::vector<openvdb::Vec3d> &path_prefix,
    const openvdb::Vec3d &anchor_seed,
    const PreviewProgressWindow &anchor_progress,
    const PreviewSegmentPlan &next_preview,
    double start_yaw,
    bool allow_optimization,
    const char *log_prefix)
{
    if (!next_preview.valid || entry_path_short.empty() ||
        next_preview.exec_path.size() < 2)
    {
        return false;
    }

    const openvdb::Vec3d visibility_target =
        anchor_progress.valid ? anchor_progress.predicted_clear_point
                              : next_preview.progress.window_start;
    const double anchor_yaw = yawTowardTarget(anchor_seed, visibility_target);

    PreviewTrajectoryExecutionResult preview_exec;
    if (!executePreviewTrajectoryPlan(log_prefix ? std::string_view(log_prefix)
                                                 : std::string_view("Receding"),
                                      plan_start_world,
                                      entry_path_short,
                                      path_prefix,
                                      anchor_seed,
                                      anchor_yaw,
                                      visibility_target,
                                      next_preview,
                                      start_yaw,
                                      allow_optimization,
                                      preview_exec))
    {
        clearSecondaryFovPlanVis(layer);
        return false;
    }

    publishSecondaryFovPlanVis(next_preview.fov_path_raw,
                               next_preview.fov_path_short,
                               next_preview.viewpoint,
                               next_preview.progress.window_end,
                               layer);
    publishActiveFrustumVis(preview_exec.active_preview_anchor,
                            visibility_target,
                            layer);

    leaf.active_anchor_seed = preview_exec.active_preview_anchor;
    leaf.active_viewpoint = preview_exec.active_preview_anchor;
    leaf.active_target_yaw = preview_exec.active_preview_yaw;
    leaf.has_preview_plan = true;
    leaf.has_active_plan = true;
    leaf.goal_traj_published = false;
    leaf.has_receding_preview_state = true;
    leaf.preview_next_anchor_seed = next_preview.viewpoint;
    leaf.preview_next_anchor_yaw = next_preview.target_yaw;
    leaf.active_anchor_progress = anchor_progress;
    leaf.next_anchor_progress = next_preview.progress;
    activateStitchMonitoring(preview_exec.trajectory_stamp, preview_exec.anchor_time);

    RCLCPP_INFO(node_handle_->get_logger(),
                "[%s][Preview] published receding stitch: anchor [%.2f, %.2f, %.2f] "
                "-> next [%.2f, %.2f, %.2f], predicted progress %.2fm",
                log_prefix ? log_prefix : "Receding",
                anchor_seed.x(), anchor_seed.y(), anchor_seed.z(),
                next_preview.viewpoint.x(), next_preview.viewpoint.y(), next_preview.viewpoint.z(),
                next_preview.progress.predicted_clear_distance);
    return true;
}

void FFAPlannerNode::promotePreviewAnchorWithoutExtension(
    SubgoalNode &leaf,
    VisLayer layer,
    const openvdb::Vec3d &promoted_anchor,
    const openvdb::Vec3d &visibility_target)
{
    clearSecondaryFovPlanVis(layer);
    leaf.active_anchor_seed = promoted_anchor;
    leaf.active_viewpoint = promoted_anchor;
    leaf.active_target_yaw = yawTowardTarget(promoted_anchor, visibility_target);
    leaf.has_active_plan = true;
    leaf.has_preview_plan = false;
    leaf.goal_traj_published = false;
    leaf.has_receding_preview_state = false;
    leaf.preview_next_anchor_seed = openvdb::Vec3d(0, 0, 0);
    leaf.preview_next_anchor_yaw = 0.0;
    leaf.active_anchor_progress = PreviewProgressWindow{};
    leaf.next_anchor_progress = PreviewProgressWindow{};
    deactivateStitchMonitoring();
    publishActiveFrustumVis(promoted_anchor, visibility_target, layer);
}

bool FFAPlannerNode::handleRecedingPreviewAnchorReached(
    SubgoalNode &leaf,
    VisLayer layer,
    const Eigen::Vector3d &plan_start_pos,
    double plan_start_yaw)
{
    if (!leaf.has_receding_preview_state)
    {
        return false;
    }

    const openvdb::Vec3d plan_start_world(plan_start_pos.x(),
                                          plan_start_pos.y(),
                                          plan_start_pos.z());

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Preview] Anchor reached; extending receding preview from lookahead "
                "[%.2f, %.2f, %.2f]",
                plan_start_world.x(), plan_start_world.y(), plan_start_world.z());

    const std::vector<openvdb::Vec3d> active_guidance_path = localGuidancePath(leaf);
    const openvdb::Vec3d fresh_window_start =
        leaf.has_guidance_progress_floor && !active_guidance_path.empty()
            ? active_guidance_path.front()
            : leaf.active_plan_hit_point;
    openvdb::Vec3d jump_point;
    std::vector<openvdb::Vec3d> jump_connect_path;
    const bool have_jump =
        findFreshCertifiedWindowJump(leaf.active_anchor_seed,
                                     active_guidance_path,
                                     fresh_window_start,
                                     jump_point,
                                     &jump_connect_path) &&
        isAheadOfGuidanceProgressFloor(leaf, jump_point);

    SafeEdgeHitResult latest_hit;
    bool hitpoint_progressed = false;
    bool have_extension_hit = false;
    bool stop_receding_preview_for_stuck_hit = false;
    openvdb::Vec3d extension_hit_point(0, 0, 0);
    if (have_jump)
    {
        advanceGuidanceProgressFloor(leaf, jump_point);
        latest_hit = findFirstSafeEdgeHit(localGuidancePath(leaf));
        if (latest_hit.safe_edge_reached)
        {
            const HitpointKey latest_key = canonicalizeHitpoint(latest_hit.hit_pt);
            if (leaf.has_last_anchor_hitpoint &&
                latest_key == leaf.last_anchor_hitpoint)
            {
                stop_receding_preview_for_stuck_hit = true;
            }
            else
            {
                hitpoint_progressed = true;
            }
            setActivePlanHitFromSafeEdge(leaf, latest_hit);
            publishSafeEdgeVis(latest_hit, layer);
            leaf.last_anchor_hitpoint = latest_key;
            leaf.has_last_anchor_hitpoint = true;
            have_extension_hit = true;
            extension_hit_point = latest_hit.hit_pt;
        }
        else if (!latest_hit.path_blocked)
        {
            leaf.active_plan_hit_point = jump_point;
            clearActivePlanHitSearchStart(leaf);
        }
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Preview] Anchor [%.2f, %.2f, %.2f] found fresh certified window jump "
                    "[%.2f, %.2f, %.2f] (connect pts %zu)",
                    leaf.active_anchor_seed.x(), leaf.active_anchor_seed.y(), leaf.active_anchor_seed.z(),
                    jump_point.x(), jump_point.y(), jump_point.z(),
                    jump_connect_path.size());
        if (latest_hit.path_blocked)
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Preview] Guidance after jump point is blocked near "
                        "[%.2f, %.2f, %.2f]; dropping preview state so the next "
                        "cycle replans guidance",
                        latest_hit.hit_pt.x(), latest_hit.hit_pt.y(), latest_hit.hit_pt.z());
            clearSecondaryFovPlanVis(layer);
            clearNodeExecutionState(leaf);
            return true;
        }
        if (!latest_hit.safe_edge_reached)
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Preview] Jump point clears the remaining guidance tail; "
                        "dropping preview state");
            clearSecondaryFovPlanVis(layer);
            clearNodeExecutionState(leaf);
            occlusion_analyzer_.reset();
            return true;
        }
    }
    else
    {
        latest_hit = findFirstSafeEdgeHit(active_guidance_path);
        if (latest_hit.safe_edge_reached)
        {
            const HitpointKey latest_key = canonicalizeHitpoint(latest_hit.hit_pt);
            if (leaf.has_last_anchor_hitpoint &&
                latest_key == leaf.last_anchor_hitpoint)
            {
                stop_receding_preview_for_stuck_hit = true;
            }
            else
            {
                hitpoint_progressed = true;
            }
            leaf.last_anchor_hitpoint = latest_key;
            leaf.has_last_anchor_hitpoint = true;
            setActivePlanHitFromSafeEdge(leaf, latest_hit);
            have_extension_hit = true;
            extension_hit_point = latest_hit.hit_pt;
        }
        else if (!latest_hit.path_blocked)
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Preview] Guidance tail is clear after anchor; dropping preview state");
            clearSecondaryFovPlanVis(layer);
            clearNodeExecutionState(leaf);
            occlusion_analyzer_.reset();
            return true;
        }
        else
        {
            RCLCPP_WARN(node_handle_->get_logger(),
                        "[Preview] Guidance became blocked near [%.2f, %.2f, %.2f] "
                        "and no fresh certified window jump was available; dropping preview "
                        "state so the next cycle replans guidance | ijk (%d,%d,%d) val=%d "
                        "below_val=%d | path front [%.2f, %.2f, %.2f] pts %zu",
                        latest_hit.hit_pt.x(), latest_hit.hit_pt.y(), latest_hit.hit_pt.z(),
                        latest_hit.hit_ijk.x(), latest_hit.hit_ijk.y(), latest_hit.hit_ijk.z(),
                        latest_hit.hit_inflate_val, latest_hit.below_inflate_val,
                        active_guidance_path.empty() ? 0.0 : active_guidance_path.front().x(),
                        active_guidance_path.empty() ? 0.0 : active_guidance_path.front().y(),
                        active_guidance_path.empty() ? 0.0 : active_guidance_path.front().z(),
                        active_guidance_path.size());
            clearSecondaryFovPlanVis(layer);
            // Keep the guidance progress floor: the next cycle must re-march
            // the exact suffix this check marched, so the blocked verdict
            // reproduces and the cycle-start blocked branch replans guidance.
            // Resetting the floor changes the marched polyline (DDA phase)
            // and can flip the verdict to safe-edge, starving the replan.
            clearNodeExecutionState(leaf);
            return true;
        }

        if (hitpoint_progressed)
        {
            if (latest_hit.has_search_start)
            {
                advanceGuidanceProgressFloor(leaf, latest_hit.search_start_world);
            }
        }
    }

    if (stop_receding_preview_for_stuck_hit)
    {
        const openvdb::Vec3d promoted_anchor = leaf.preview_next_anchor_seed;
        const openvdb::Vec3d visibility_target =
            leaf.next_anchor_progress.valid
                ? leaf.next_anchor_progress.predicted_clear_point
                : latest_hit.hit_pt;
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Preview] Hitpoint [%.2f, %.2f, %.2f] did not advance after anchor arrival; "
                    "stopping receding extension and keeping current preview endpoint "
                    "[%.2f, %.2f, %.2f] as the terminal sensing pose",
                    latest_hit.hit_pt.x(), latest_hit.hit_pt.y(), latest_hit.hit_pt.z(),
                    promoted_anchor.x(), promoted_anchor.y(), promoted_anchor.z());
        promotePreviewAnchorWithoutExtension(leaf, layer, promoted_anchor, visibility_target);
        return true;
    }

    const openvdb::Vec3d promoted_anchor = leaf.preview_next_anchor_seed;
    PreviewProgressWindow promoted_anchor_progress = leaf.next_anchor_progress;
    const std::vector<openvdb::Vec3d> preview_guidance_path =
        localGuidancePath(leaf);
    if (!promoted_anchor_progress.valid)
    {
        buildPreviewProgressWindow(promoted_anchor,
                                   preview_guidance_path,
                                   have_jump ? jump_point : leaf.active_plan_hit_point,
                                   fov_stitch_lookahead_dist_,
                                   FovClearProfileKind::STANDARD,
                                   promoted_anchor_progress);
    }

    const std::vector<openvdb::Vec3d> &comparison_guidance_path =
        leaf.guidance_path_short.empty() ? preview_guidance_path : leaf.guidance_path_short;

    const double preview_eps = std::max(0.5 * voxel_size_, 1.0e-4);
    const bool have_comparison_path = comparison_guidance_path.size() >= 2;
    auto progress_of = [&](const openvdb::Vec3d &point) -> double
    {
        return projectProgressAlongPath(comparison_guidance_path, point);
    };
    auto is_ahead = [&](const openvdb::Vec3d &candidate,
                        const openvdb::Vec3d &reference) -> bool
    {
        return have_comparison_path &&
               progress_of(candidate) > progress_of(reference) + preview_eps;
    };

    openvdb::Vec3d extension_start =
        promoted_anchor_progress.valid
            ? promoted_anchor_progress.predicted_clear_point
            : (latest_hit.safe_edge_reached ? latest_hit.hit_pt : leaf.active_plan_hit_point);

    // The receding preview should advance from the promoted anchor's actual
    // certified clear progress, not from the previous preview window end.
    // Jump/search-start progress can only move that base forward.
    if (have_jump && (!have_comparison_path || is_ahead(jump_point, extension_start)))
    {
        extension_start = jump_point;
    }
    if (latest_hit.has_search_start &&
        is_ahead(latest_hit.search_start_world, extension_start))
    {
        extension_start = latest_hit.search_start_world;
    }

    std::optional<openvdb::Vec3d> fixed_preview_target;
    if (have_comparison_path)
    {
        const double base_progress = progress_of(extension_start);
        const double total_progress = pathLengthWorld(comparison_guidance_path);
        const double default_target_progress =
            std::min(base_progress + fov_stitch_lookahead_dist_,
                     total_progress);
        openvdb::Vec3d default_preview_target =
            samplePointAlongPath(comparison_guidance_path,
                                 default_target_progress);

        if (have_extension_hit)
        {
            const double hit_progress = progress_of(extension_hit_point);
            if (hit_progress > default_target_progress + preview_eps)
            {
                fixed_preview_target = extension_hit_point;
                RCLCPP_INFO(node_handle_->get_logger(),
                            "[Preview] Latest hitpoint [%.2f, %.2f, %.2f] is ahead of promoted-clear "
                            "preview target [%.2f, %.2f, %.2f]; targeting the hitpoint directly "
                            "(base progress %.2f, default %.2f, hit %.2f)",
                            extension_hit_point.x(), extension_hit_point.y(), extension_hit_point.z(),
                            default_preview_target.x(), default_preview_target.y(), default_preview_target.z(),
                            base_progress, default_target_progress, hit_progress);
            }
        }

        if (!fixed_preview_target)
        {
            fixed_preview_target = default_preview_target;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[Preview] Rolling preview target from promoted clear base "
                        "[%.2f, %.2f, %.2f] to [%.2f, %.2f, %.2f] "
                        "(base progress %.2f, target %.2f / %.2f)",
                        extension_start.x(), extension_start.y(), extension_start.z(),
                        fixed_preview_target->x(), fixed_preview_target->y(), fixed_preview_target->z(),
                        base_progress, default_target_progress, total_progress);
        }
    }

    PreviewSegmentPlan next_preview;
    if (!leaf.active_plan_has_search_start)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Preview] Promoted anchor has no cached strict-free search start after "
                    "hit [%.2f, %.2f, %.2f]; keeping promoted anchor "
                    "[%.2f, %.2f, %.2f] active",
                    leaf.active_plan_hit_point.x(),
                    leaf.active_plan_hit_point.y(),
                    leaf.active_plan_hit_point.z(),
                    promoted_anchor.x(), promoted_anchor.y(), promoted_anchor.z());
        promotePreviewAnchorWithoutExtension(leaf, layer, promoted_anchor, extension_start);
        return true;
    }
    const openvdb::Coord preview_search_start_ijk = leaf.active_plan_search_start_ijk;
    if (!buildCertifiedPreviewSegment(leaf,
                                      promoted_anchor,
                                      preview_guidance_path,
                                      extension_start,
                                      preview_search_start_ijk,
                                      layer,
                                      next_preview,
                                      fixed_preview_target ? &(*fixed_preview_target) : nullptr))
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Preview] Failed to extend from promoted anchor [%.2f, %.2f, %.2f] "
                    "(reason=%s); keeping promoted anchor active",
                    promoted_anchor.x(), promoted_anchor.y(), promoted_anchor.z(),
                    next_preview.failure_reason.c_str());
        promotePreviewAnchorWithoutExtension(leaf, layer, promoted_anchor, extension_start);
        return true;
    }

    std::vector<openvdb::Vec3d> entry_raw;
    std::vector<openvdb::Vec3d> entry_short;
    if (!buildCertifiedAccessPath(plan_start_world, promoted_anchor,
                                  entry_raw, entry_short))
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Preview] Cannot connect current lookahead to promoted anchor "
                    "[%.2f, %.2f, %.2f]; keeping promoted anchor active",
                    promoted_anchor.x(), promoted_anchor.y(), promoted_anchor.z());
        promotePreviewAnchorWithoutExtension(leaf, layer, promoted_anchor, extension_start);
        return true;
    }

    if (entry_short.empty())
    {
        entry_short = entry_raw;
    }

    const bool published =
        publishRecedingPreviewFromAnchor(leaf,
                                         layer,
                                         plan_start_world,
                                         entry_short,
                                         {},
                                         promoted_anchor,
                                         promoted_anchor_progress,
                                         next_preview,
                                         plan_start_yaw,
                                         true,
                                         "Receding");
    if (!published)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Preview] Failed to publish extension from promoted anchor "
                    "[%.2f, %.2f, %.2f]; keeping promoted anchor active",
                    promoted_anchor.x(), promoted_anchor.y(), promoted_anchor.z());
        promotePreviewAnchorWithoutExtension(leaf, layer, promoted_anchor, extension_start);
    }
    return true;
}

traj_optimizer::BoundaryState FFAPlannerNode::makeOptimizerStartBoundaryState(
    const openvdb::Vec3d &start_pos) const
{
    const LookaheadState la = getLookaheadState();
    if (la.valid)
    {
        return makeBoundaryState(start_pos, la.vel, la.acc);
    }

    return makeBoundaryState(start_pos);
}

FFAPlannerNode::PublishResult FFAPlannerNode::tryPublishOptimizedTrajectory(
    const std::vector<openvdb::Vec3d> &waypoints,
    double start_yaw,
    double final_yaw)
{
    PublishResult result;
    // FFA-search ablation: source-level gate so NO optimizer path can publish
    // (the visibility-cell pipeline does not route through the interpolate
    // wrappers, so gating only there is insufficient).
    if (disable_traj_opt_ || !traj_optimizer_ || waypoints.size() < 2)
    {
        return result;
    }

    traj_optimizer::BoundaryState start_state = makeOptimizerStartBoundaryState(waypoints.front());
    traj_optimizer::BoundaryState goal_state = makeBoundaryState(waypoints.back());

    traj_optimizer::OptimizedPlan plan;
    traj_optimizer::DebugInfo debug_info;
    if (!traj_optimizer_->optimizePath(waypoints, start_state, goal_state,
                                       start_yaw, final_yaw, plan, &debug_info))
    {
        clearOptimizedTrajectoryVis();
        if (!debug_info.corridor_hpolys.empty() || !debug_info.inflated_points.empty())
        {
            publishSfcDebugVis(debug_info);
        }
        else
        {
            clearSfcDebugVis();
        }
        return result;
    }

    publishSfcDebugVis(debug_info);
    publishOptimizedTrajectoryVis(plan.position_traj, &debug_info);
    return publishOptimizedPlanTrajectory(plan);
}

FFAPlannerNode::PublishResult FFAPlannerNode::tryPublishOptimizedStitchedTrajectory(
    const std::vector<openvdb::Vec3d> &waypoints,
    size_t split_index,
    double start_yaw,
    double split_yaw,
    double final_yaw)
{
    PublishResult result;
    if (disable_traj_opt_ || !traj_optimizer_ || waypoints.size() < 3 ||
        split_index == 0 || split_index >= waypoints.size() - 1)
    {
        return result;
    }

    const traj_optimizer::BoundaryState stitched_start = makeOptimizerStartBoundaryState(waypoints.front());
    const traj_optimizer::BoundaryState stitched_goal = makeBoundaryState(waypoints.back());

    traj_optimizer::OptimizedPlan stitched_plan;
    traj_optimizer::DebugInfo stitched_debug_info;
    if (traj_optimizer_->optimizeStitchedPath(waypoints, split_index,
                                              stitched_start, stitched_goal,
                                              start_yaw, split_yaw, final_yaw,
                                              stitched_plan, &stitched_debug_info))
    {
        publishSfcDebugVis(stitched_debug_info);
        publishOptimizedTrajectoryVis(stitched_plan.position_traj, &stitched_debug_info);
        return publishOptimizedPlanTrajectory(stitched_plan);
    }

    std::vector<openvdb::Vec3d> seg1(waypoints.begin(), waypoints.begin() + split_index + 1);
    std::vector<openvdb::Vec3d> seg2(waypoints.begin() + split_index, waypoints.end());
    if (seg1.size() < 2 || seg2.size() < 2)
    {
        return result;
    }

    const traj_optimizer::BoundaryState shared_state = makeBoundaryState(
        waypoints[split_index], Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    const traj_optimizer::BoundaryState seg1_start = makeOptimizerStartBoundaryState(seg1.front());
    const traj_optimizer::BoundaryState seg2_goal = makeBoundaryState(seg2.back());

    traj_optimizer::OptimizedPlan plan1;
    traj_optimizer::DebugInfo debug_info1;
    if (!traj_optimizer_->optimizePath(seg1, seg1_start, shared_state,
                                       start_yaw, split_yaw, plan1, &debug_info1))
    {
        clearOptimizedTrajectoryVis();
        if (!debug_info1.corridor_hpolys.empty() || !debug_info1.inflated_points.empty())
        {
            publishSfcDebugVis(debug_info1);
        }
        else
        {
            clearSfcDebugVis();
        }
        return result;
    }

    traj_optimizer::OptimizedPlan plan2;
    traj_optimizer::DebugInfo debug_info2;
    if (!traj_optimizer_->optimizePath(seg2, shared_state, seg2_goal,
                                       split_yaw, final_yaw, plan2, &debug_info2))
    {
        clearOptimizedTrajectoryVis();
        if (!debug_info2.corridor_hpolys.empty() || !debug_info2.inflated_points.empty())
        {
            publishSfcDebugVis(debug_info2);
        }
        else
        {
            clearSfcDebugVis();
        }
        return result;
    }

    // Two-piece stitch fallback: tag plan1's end as the anchor before
    // appending, so the stitched plan inherits a meaningful anchor_time.
    // (append() promotes plan1.anchor_time onto the joined timeline.)
    plan1.anchor_time = plan1.getTotalDuration();
    traj_optimizer::OptimizedPlan stitched = plan1;
    stitched.append(plan2);
    publishSfcDebugVis(debug_info2);
    publishOptimizedTrajectoryVis(stitched.position_traj, &debug_info2);
    return publishOptimizedPlanTrajectory(stitched);
}

FFAPlannerNode::PublishResult FFAPlannerNode::tryPublishOptimizedVisibilityStitchedTrajectory(
    const std::vector<openvdb::Vec3d> &path_in,
    const traj_optimizer::VisibilityCell &visibility_cell,
    const std::vector<openvdb::Vec3d> &path_out,
    double start_yaw,
    double observation_yaw,
    double final_yaw)
{
    PublishResult result;
    if (disable_traj_opt_ || !traj_optimizer_ || !visibility_cell.valid ||
        path_in.size() < 2 || path_out.size() < 2)
    {
        return result;
    }

    const traj_optimizer::BoundaryState stitched_start =
        makeOptimizerStartBoundaryState(path_in.front());
    const traj_optimizer::BoundaryState stitched_goal =
        makeBoundaryState(path_out.back());

    traj_optimizer::ObservationHint observation_hint;
    observation_hint.cell_index = -1; // resolved inside traj optimizer to T_in.size()
    observation_hint.yaw_ref = observation_yaw;
    observation_hint.yaw_weight = 1.0e4;

    traj_optimizer::OptimizedPlan stitched_plan;
    traj_optimizer::DebugInfo stitched_debug_info;
    if (traj_optimizer_->optimizeThroughVisibilityCell(path_in,
                                                       visibility_cell,
                                                       path_out,
                                                       stitched_start,
                                                       stitched_goal,
                                                       start_yaw,
                                                       final_yaw,
                                                       observation_hint,
                                                       stitched_plan,
                                                       &stitched_debug_info))
    {
        publishSfcDebugVis(stitched_debug_info);
        publishOptimizedTrajectoryVis(stitched_plan.position_traj, &stitched_debug_info);
        return publishOptimizedPlanTrajectory(stitched_plan);
    }

    clearOptimizedTrajectoryVis();
    if (!stitched_debug_info.corridor_hpolys.empty() || !stitched_debug_info.inflated_points.empty())
    {
        publishSfcDebugVis(stitched_debug_info);
    }
    else
    {
        clearSfcDebugVis();
    }

    return result;
}

bool FFAPlannerNode::executePreviewTrajectoryPlan(
    std::string_view log_prefix,
    const openvdb::Vec3d &plan_start_world,
    const std::vector<openvdb::Vec3d> &entry_exec_path_short,
    const std::vector<openvdb::Vec3d> &path_prefix,
    const openvdb::Vec3d &observation_viewpoint,
    double observation_yaw,
    const openvdb::Vec3d &visibility_target_world,
    const PreviewSegmentPlan &preview_plan,
    double start_yaw,
    bool allow_optimization,
    PreviewTrajectoryExecutionResult &result)
{
    result = PreviewTrajectoryExecutionResult();
    result.active_preview_anchor = observation_viewpoint;
    result.active_preview_yaw = observation_yaw;

    if (entry_exec_path_short.empty() || preview_plan.exec_path.size() <= 1)
    {
        return false;
    }

    auto vecDistance = [](const openvdb::Vec3d &a, const openvdb::Vec3d &b)
    {
        const Eigen::Vector3d ea(a.x(), a.y(), a.z());
        const Eigen::Vector3d eb(b.x(), b.y(), b.z());
        return (ea - eb).norm();
    };
    auto pathLength = [&](const std::vector<openvdb::Vec3d> &path)
    {
        double length = 0.0;
        for (size_t i = 0; i + 1 < path.size(); ++i)
        {
            length += vecDistance(path[i], path[i + 1]);
        }
        return length;
    };

    const double entry_length = pathLength(entry_exec_path_short);
    if (entry_exec_path_short.size() < 2 || entry_length < 1.0e-6)
    {
        const double start_anchor_dist = vecDistance(plan_start_world, observation_viewpoint);
        const double direct_tail_tol = std::max(0.1 * voxel_size_, 1.0e-3);
        if (start_anchor_dist <= direct_tail_tol)
        {
            std::vector<openvdb::Vec3d> direct_path = preview_plan.exec_path;
            direct_path.front() = plan_start_world;
            const PublishResult direct_pub =
                interpolateAndPublishTrajectory(direct_path, start_yaw,
                                                preview_plan.target_yaw,
                                                allow_optimization);
            if (!direct_pub.published || stampIsZero(direct_pub.trajectory_stamp))
            {
                return false;
            }

            result.stitched_path_size = direct_path.size();
            result.trajectory_stamp = direct_pub.trajectory_stamp;
            result.anchor_time = 0.0;
            result.published = true;
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[%.*s][Preview] current lookahead is already at anchor "
                        "[%.2f, %.2f, %.2f]; publishing direct tail to next preview",
                        static_cast<int>(log_prefix.size()), log_prefix.data(),
                        observation_viewpoint.x(), observation_viewpoint.y(), observation_viewpoint.z());
            return true;
        }

        std::vector<openvdb::Vec3d> synthetic_entry{plan_start_world, observation_viewpoint};
        std::vector<openvdb::Vec3d> stitched =
            prependPathPrefix(path_prefix, synthetic_entry);
        const size_t split_index = stitched.size() - 1;
        stitched.insert(stitched.end(),
                        preview_plan.exec_path.begin() + 1,
                        preview_plan.exec_path.end());
        const PublishResult seam_pub = interpolateAndPublishStitchedTrajectory(
            stitched,
            split_index,
            start_yaw,
            observation_yaw,
            preview_plan.target_yaw,
            allow_optimization);
        if (!seam_pub.published || stampIsZero(seam_pub.trajectory_stamp) ||
            seam_pub.anchor_time < 0.0)
        {
            return false;
        }

        result.stitched_path_size = stitched.size();
        result.trajectory_stamp = seam_pub.trajectory_stamp;
        result.anchor_time = seam_pub.anchor_time;
        result.published = true;
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[%.*s][Preview] recovered degenerate entry with synthetic "
                    "%.3fm segment to anchor [%.2f, %.2f, %.2f]",
                    static_cast<int>(log_prefix.size()), log_prefix.data(),
                    start_anchor_dist,
                    observation_viewpoint.x(), observation_viewpoint.y(),
                    observation_viewpoint.z());
        return true;
    }

    std::vector<openvdb::Vec3d> stitched =
        prependPathPrefix(path_prefix, entry_exec_path_short);
    if (stitched.size() < 2)
    {
        return false;
    }

    const size_t split_index = stitched.size() - 1;
    stitched.insert(stitched.end(),
                    preview_plan.exec_path.begin() + 1,
                    preview_plan.exec_path.end());
    result.stitched_path_size = stitched.size();

    traj_optimizer::VisibilityCell visibility_cell;
    std::vector<openvdb::Vec3d> visibility_in_raw;
    std::vector<openvdb::Vec3d> visibility_in_short;
    std::vector<openvdb::Vec3d> visibility_out_raw;
    std::vector<openvdb::Vec3d> visibility_out_short;
    bool have_visibility_opt = false;

    if (buildVisibilityAnchorCell(observation_viewpoint, visibility_target_world, visibility_cell))
    {
        const openvdb::Vec3d vis_center(visibility_cell.center.x(),
                                        visibility_cell.center.y(),
                                        visibility_cell.center.z());
        std::string vis_in_reason;
        std::string vis_out_reason;
        if (buildCertifiedAccessPath(plan_start_world, vis_center,
                                     visibility_in_raw, visibility_in_short,
                                     nullptr, &vis_in_reason) &&
            buildCertifiedAccessPath(vis_center, preview_plan.stop_point,
                                     visibility_out_raw, visibility_out_short,
                                     nullptr, &vis_out_reason) &&
            visibility_in_short.size() >= 2 &&
            visibility_out_short.size() >= 2)
        {
            const PublishResult vis_pub =
                tryPublishOptimizedVisibilityStitchedTrajectory(
                    visibility_in_short,
                    visibility_cell,
                    visibility_out_short,
                    start_yaw,
                    visibility_cell.certified_yaw,
                    preview_plan.target_yaw);
            have_visibility_opt = vis_pub.published;
            if (have_visibility_opt)
            {
                result.used_visibility_optimization = true;
                result.trajectory_stamp = vis_pub.trajectory_stamp;
                result.anchor_time = vis_pub.anchor_time;
            }
        }
        else
        {
            RCLCPP_INFO(node_handle_->get_logger(),
                        "[%.*s][VisCell] transition to/from V_anchor unavailable | in_reason=%s out_reason=%s",
                        static_cast<int>(log_prefix.size()), log_prefix.data(),
                        vis_in_reason.c_str(),
                        vis_out_reason.c_str());
        }
    }
    else
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[%.*s][VisCell] failed to build V_anchor around viewpoint [%.2f, %.2f, %.2f]",
                    static_cast<int>(log_prefix.size()), log_prefix.data(),
                    observation_viewpoint.x(), observation_viewpoint.y(), observation_viewpoint.z());
    }

    if (!have_visibility_opt)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[%.*s][VisCell] visibility-cell optimize failed, falling back to seam-stitched optimizer/interpolator",
                    static_cast<int>(log_prefix.size()), log_prefix.data());
        const PublishResult seam_pub = interpolateAndPublishStitchedTrajectory(
            stitched,
            split_index,
            start_yaw,
            observation_yaw,
            preview_plan.target_yaw,
            allow_optimization);
        // Visibility-cell anchor is unused on the seam fallback; the seam
        // itself doubles as the vp1 anchor for monitoring purposes.
        if (seam_pub.published)
        {
            result.trajectory_stamp = seam_pub.trajectory_stamp;
            result.anchor_time = seam_pub.anchor_time;
        }
    }

    if (stampIsZero(result.trajectory_stamp) || result.anchor_time < 0.0)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[%.*s][Stitch] preview trajectory published no usable anchor handle "
                    "(stamp=%d.%09u anchor_time=%.3f)",
                    static_cast<int>(log_prefix.size()), log_prefix.data(),
                    result.trajectory_stamp.sec, result.trajectory_stamp.nanosec,
                    result.anchor_time);
        return false;
    }

    result.published = true;
    return true;
}

FFAPlannerNode::PublishResult FFAPlannerNode::publishOptimizedPlanTrajectory(
    const traj_optimizer::OptimizedPlan &plan)
{
    PublishResult result;
    if (!plan.valid || plan.position_traj.getPieceNum() <= 0)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
        return result;
    }

    const double total_duration = plan.getTotalDuration();
    if (total_duration <= 1.0e-6)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
        return result;
    }

    const double sample_dt =
        (cruise_speed_ > 1.0e-6 && interpolate_step_ > 1.0e-6) ? (interpolate_step_ / cruise_speed_) : 0.1;

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = node_handle_->now();
    traj.header.frame_id = world_frame_id_;
    traj.joint_names.push_back(robot_frame_id_);

    std::vector<Eigen::Vector3d> cache_positions;
    std::vector<Eigen::Vector3d> cache_velocities;
    std::vector<Eigen::Vector3d> cache_accelerations;
    std::vector<double> cache_yaws;
    std::vector<double> cache_times;

    auto append_point = [&](double t_eval)
    {
        const Eigen::Vector3d pos = plan.position_traj.getPos(t_eval);
        const Eigen::Vector3d vel = plan.position_traj.getVel(t_eval);
        const Eigen::Vector3d acc = plan.position_traj.getAcc(t_eval);
        const double yaw = plan.sampleYaw(t_eval);
        const double yaw_rate = plan.sampleYawRate(t_eval);

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
        geometry_msgs::msg::Transform tf;
        tf.translation.x = pos.x();
        tf.translation.y = pos.y();
        tf.translation.z = pos.z();
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        tf.rotation = tf2::toMsg(q);
        pt.transforms.push_back(tf);

        geometry_msgs::msg::Twist tw;
        tw.linear.x = vel.x();
        tw.linear.y = vel.y();
        tw.linear.z = vel.z();
        tw.angular.z = yaw_rate;
        pt.velocities.push_back(tw);
        pt.time_from_start = rclcpp::Duration::from_seconds(t_eval);
        traj.points.push_back(pt);

        cache_positions.push_back(pos);
        cache_velocities.push_back(vel);
        cache_accelerations.push_back(acc);
        cache_yaws.push_back(yaw);
        cache_times.push_back(t_eval);
    };

    for (double t_eval = 0.0; t_eval < total_duration; t_eval += sample_dt)
    {
        append_point(t_eval);
    }
    append_point(total_duration);

    std::vector<double> cache_arclen;
    cache_arclen.reserve(cache_positions.size());
    double accum_len = 0.0;
    for (size_t i = 0; i < cache_positions.size(); ++i)
    {
        if (i > 0)
        {
            accum_len += (cache_positions[i] - cache_positions[i - 1]).norm();
        }
        cache_arclen.push_back(accum_len);
    }

    if (accum_len < 1.0e-4)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Local] Rejecting optimized trajectory with collapsed spatial length "
                    "(len=%.6f m, duration=%.4fs); falling back to non-optimized publisher",
                    accum_len, total_duration);
        return result;
    }

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Local] Publishing optimized trajectory: %zu points, %.1f s, yaw %.2f -> %.2f",
                traj.points.size(), total_duration,
                cache_yaws.front(), cache_yaws.back());
    const rclcpp::Time t_pub(traj.header.stamp);
    pub_trajectory_->publish(traj);

    const double effective_speed = accum_len / std::max(total_duration, 1.0e-6);
    const uint64_t traj_id = cachePublishedTrajectory(std::move(cache_positions),
                                                      std::move(cache_velocities),
                                                      std::move(cache_accelerations),
                                                      std::move(cache_yaws),
                                                      std::move(cache_arclen),
                                                      std::move(cache_times),
                                                      effective_speed, t_pub);
    result.published = true;
    result.traj_id = traj_id;
    result.trajectory_stamp = traj.header.stamp;
    result.total_duration = total_duration;
    result.anchor_time = plan.anchor_time;
    return result;
}

// Trajectory Interpolation & Publishing

FFAPlannerNode::PublishResult FFAPlannerNode::interpolateAndPublishTrajectory(
    const std::vector<openvdb::Vec3d> &waypoints,
    double start_yaw, double final_yaw,
    bool allow_optimization)
{
    PublishResult result;
    if (waypoints.size() < 2)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
        return result;
    }
    allow_optimization = allow_optimization && !disable_traj_opt_;
    if (allow_optimization)
    {
        PublishResult opt_result = tryPublishOptimizedTrajectory(waypoints, start_yaw, final_yaw);
        if (opt_result.published)
        {
            return opt_result;
        }
    }
    if (!allow_optimization)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
    }
    if (interpolate_step_ <= 0.0 || cruise_speed_ <= 0.0)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Local] interpolate_step or cruise_speed <= 0, skipping");
        return result;
    }

    // Compute total path length for yaw interpolation fraction
    double total_len = 0.0;
    for (size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        Eigen::Vector3d a(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        Eigen::Vector3d b(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
        total_len += (b - a).norm();
    }

    // Normalize yaw difference to (-pi, pi].
    double yaw_diff = wrapToPi(final_yaw - start_yaw);

    // Degenerate input: path collapsed to a single point in position. Publish
    // a yaw-only trajectory (mostly relevant when the optimizer above failed
    // on the same degenerate path). Without this, we'd silently return and
    // leave the controller chasing the previous trajectory while planner
    // state machine believes a fresh one was published.
    if (total_len < 1e-6)
    {
        const Eigen::Vector3d pos(waypoints.front().x(),
                                  waypoints.front().y(),
                                  waypoints.front().z());
        return publishYawOnlyDegenerateTrajectory(pos, start_yaw, start_yaw + yaw_diff);
    }

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = node_handle_->now();
    traj.header.frame_id = world_frame_id_;
    traj.joint_names.push_back(robot_frame_id_);

    auto make_point = [](const Eigen::Vector3d &pos, double yaw,
                         const Eigen::Vector3d &vel,
                         double t) -> trajectory_msgs::msg::MultiDOFJointTrajectoryPoint
    {
        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;

        geometry_msgs::msg::Transform tf;
        tf.translation.x = pos.x();
        tf.translation.y = pos.y();
        tf.translation.z = pos.z();
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        tf.rotation = tf2::toMsg(q);
        pt.transforms.push_back(tf);

        geometry_msgs::msg::Twist tw;
        tw.linear.x = vel.x();
        tw.linear.y = vel.y();
        tw.linear.z = vel.z();
        pt.velocities.push_back(tw);

        pt.time_from_start = rclcpp::Duration::from_seconds(t);
        return pt;
    };

    double t = 0.0;
    double cum_len = 0.0;
    Eigen::Vector3d last_pos(waypoints[0].x(), waypoints[0].y(), waypoints[0].z());

    for (size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        Eigen::Vector3d p0(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        Eigen::Vector3d p1(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
        Eigen::Vector3d seg = p1 - p0;
        double seg_len = seg.norm();
        if (seg_len < 1e-6)
        {
            continue;
        }

        Eigen::Vector3d dir = seg / seg_len;
        Eigen::Vector3d vel = dir * cruise_speed_;

        if (i == 0)
        {
            double frac = cum_len / total_len;
            traj.points.push_back(make_point(p0, start_yaw + yaw_diff * frac, vel, t));
        }

        double dt_step = interpolate_step_ / cruise_speed_;
        for (double d = interpolate_step_; d < seg_len; d += interpolate_step_)
        {
            t += dt_step;
            double frac = (cum_len + d) / total_len;
            Eigen::Vector3d pos = p0 + dir * d;
            traj.points.push_back(make_point(pos, start_yaw + yaw_diff * frac, vel, t));
            last_pos = pos;
        }

        double ds = (p1 - last_pos).norm();
        t += ds / cruise_speed_;
        cum_len += seg_len;
        last_pos = p1;

        bool is_last = (i + 2 >= waypoints.size());
        double frac = cum_len / total_len;
        Eigen::Vector3d end_vel = is_last ? Eigen::Vector3d::Zero() : vel;
        traj.points.push_back(make_point(p1, start_yaw + yaw_diff * frac, end_vel, t));
    }

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Local] Publishing trajectory: %zu points, %.1f s, yaw %.2f -> %.2f",
                traj.points.size(), t, start_yaw, final_yaw);
    const rclcpp::Time t_pub(traj.header.stamp);
    pub_trajectory_->publish(traj);

    // Build a sparse snapshot matching what the controller just received.
    // Dense message is piecewise-linear between these waypoints, so linear
    // interpolation here reproduces the published traj exactly.
    std::vector<Eigen::Vector3d> cache_positions;
    std::vector<Eigen::Vector3d> cache_velocities;
    std::vector<Eigen::Vector3d> cache_accelerations;
    std::vector<double> cache_yaws, cache_arclen, cache_times;
    cache_positions.reserve(waypoints.size());
    cache_velocities.reserve(waypoints.size());
    cache_accelerations.reserve(waypoints.size());
    cache_yaws.reserve(waypoints.size());
    cache_arclen.reserve(waypoints.size());
    cache_times.reserve(waypoints.size());
    double acc_len = 0.0;
    for (size_t i = 0; i < waypoints.size(); ++i)
    {
        Eigen::Vector3d p(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        if (i > 0)
        {
            Eigen::Vector3d prev(waypoints[i - 1].x(), waypoints[i - 1].y(), waypoints[i - 1].z());
            acc_len += (p - prev).norm();
        }
        cache_positions.push_back(p);
        if (i + 1 < waypoints.size())
        {
            Eigen::Vector3d next(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
            const Eigen::Vector3d delta = next - p;
            const Eigen::Vector3d dir = delta.norm() > 1.0e-6 ? (delta / delta.norm()).eval() : Eigen::Vector3d::Zero();
            cache_velocities.push_back(dir * cruise_speed_);
        }
        else
        {
            cache_velocities.push_back(Eigen::Vector3d::Zero());
        }
        cache_accelerations.push_back(Eigen::Vector3d::Zero());
        cache_arclen.push_back(acc_len);
        cache_times.push_back(acc_len / cruise_speed_);
        const double frac = total_len > 1e-6 ? (acc_len / total_len) : 0.0;
        cache_yaws.push_back(start_yaw + yaw_diff * frac);
    }
    const uint64_t traj_id = cachePublishedTrajectory(std::move(cache_positions),
                                                      std::move(cache_velocities),
                                                      std::move(cache_accelerations),
                                                      std::move(cache_yaws),
                                                      std::move(cache_arclen),
                                                      std::move(cache_times),
                                                      cruise_speed_, t_pub);
    result.published = true;
    result.traj_id = traj_id;
    result.trajectory_stamp = traj.header.stamp;
    result.total_duration = t;
    // Single-stage trajectory has no anchor.
    return result;
}

FFAPlannerNode::PublishResult FFAPlannerNode::interpolateAndPublishStitchedTrajectory(
    const std::vector<openvdb::Vec3d> &waypoints,
    size_t split_index,
    double start_yaw,
    double split_yaw,
    double final_yaw,
    bool allow_optimization)
{
    PublishResult result;
    if (waypoints.size() < 2)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
        return result;
    }
    allow_optimization = allow_optimization && !disable_traj_opt_;
    if (allow_optimization)
    {
        PublishResult opt_result =
            tryPublishOptimizedStitchedTrajectory(waypoints, split_index,
                                                  start_yaw, split_yaw, final_yaw);
        if (opt_result.published)
        {
            return opt_result;
        }
    }
    if (!allow_optimization)
    {
        clearOptimizedTrajectoryVis();
        clearSfcDebugVis();
    }
    if (split_index == 0 || split_index >= waypoints.size())
    {
        return interpolateAndPublishTrajectory(waypoints, start_yaw, final_yaw, allow_optimization);
    }
    if (interpolate_step_ <= 0.0 || cruise_speed_ <= 0.0)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Local] interpolate_step or cruise_speed <= 0, skipping stitched traj");
        return result;
    }

    std::vector<double> prefix_len(waypoints.size(), 0.0);
    for (size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        const Eigen::Vector3d a(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        const Eigen::Vector3d b(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
        prefix_len[i + 1] = prefix_len[i] + (b - a).norm();
    }

    const double total_len = prefix_len.back();
    const double split_len = prefix_len[split_index];
    if (total_len < 1e-6 || split_len < 1e-6)
    {
        return interpolateAndPublishTrajectory(waypoints, start_yaw, final_yaw, allow_optimization);
    }

    // Unwrap the split and final yaws relative to start_yaw so the yaw
    // schedule below is a single monotonic function of arclength. Without this
    // unwrap, the second segment would restart from the wrapped split_yaw and
    // create a ~2pi discontinuity at the seam whenever |split_yaw - start_yaw|
    // crosses +/-pi, which makes the lookahead vis arrow appear to spin a full
    // turn on the segment containing the seam.
    const double yaw_diff_1 = wrapToPi(split_yaw - start_yaw);
    const double split_yaw_unwrapped = start_yaw + yaw_diff_1;
    const double yaw_diff_2 = wrapToPi(final_yaw - split_yaw_unwrapped);

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = node_handle_->now();
    traj.header.frame_id = world_frame_id_;
    traj.joint_names.push_back(robot_frame_id_);

    auto yaw_at_progress = [&](double s)
    {
        if (s <= split_len)
        {
            const double frac = split_len > 1e-6 ? (s / split_len) : 1.0;
            return start_yaw + yaw_diff_1 * frac;
        }

        const double tail = std::max(total_len - split_len, 1e-6);
        const double frac = (s - split_len) / tail;
        return split_yaw_unwrapped + yaw_diff_2 * frac;
    };

    auto make_point = [&](const Eigen::Vector3d &pos, double yaw,
                          const Eigen::Vector3d &vel,
                          double t) -> trajectory_msgs::msg::MultiDOFJointTrajectoryPoint
    {
        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;

        geometry_msgs::msg::Transform tf;
        tf.translation.x = pos.x();
        tf.translation.y = pos.y();
        tf.translation.z = pos.z();
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        tf.rotation = tf2::toMsg(q);
        pt.transforms.push_back(tf);

        geometry_msgs::msg::Twist tw;
        tw.linear.x = vel.x();
        tw.linear.y = vel.y();
        tw.linear.z = vel.z();
        pt.velocities.push_back(tw);

        pt.time_from_start = rclcpp::Duration::from_seconds(t);
        return pt;
    };

    double t = 0.0;
    double last_s = 0.0;
    Eigen::Vector3d last_pos(waypoints.front().x(), waypoints.front().y(), waypoints.front().z());

    for (size_t i = 0; i + 1 < waypoints.size(); ++i)
    {
        const Eigen::Vector3d p0(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        const Eigen::Vector3d p1(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
        const Eigen::Vector3d seg = p1 - p0;
        const double seg_len = seg.norm();
        if (seg_len < 1e-6)
        {
            continue;
        }

        const Eigen::Vector3d dir = seg / seg_len;
        const Eigen::Vector3d vel = dir * cruise_speed_;

        if (i == 0)
        {
            traj.points.push_back(make_point(p0, yaw_at_progress(0.0), vel, t));
        }

        const double dt_step = interpolate_step_ / cruise_speed_;
        for (double d = interpolate_step_; d < seg_len; d += interpolate_step_)
        {
            t += dt_step;
            const Eigen::Vector3d pos = p0 + dir * d;
            const double s = prefix_len[i] + d;
            traj.points.push_back(make_point(pos, yaw_at_progress(s), vel, t));
            last_pos = pos;
            last_s = s;
        }

        const double ds = (p1 - last_pos).norm();
        t += ds / cruise_speed_;
        last_pos = p1;
        last_s = prefix_len[i + 1];

        const bool is_last = (i + 2 >= waypoints.size());
        const Eigen::Vector3d end_vel = is_last ? Eigen::Vector3d::Zero() : vel;
        traj.points.push_back(make_point(p1, yaw_at_progress(last_s), end_vel, t));
    }

    RCLCPP_INFO(node_handle_->get_logger(),
                "[Local] Publishing stitched trajectory: %zu points, %.1f s, yaw %.2f -> %.2f -> %.2f",
                traj.points.size(), t, start_yaw, split_yaw, final_yaw);
    const rclcpp::Time t_pub(traj.header.stamp);
    pub_trajectory_->publish(traj);

    // Sparse snapshot: per-waypoint (pos, yaw, arclen, time). Yaw uses the
    // same piecewise-linear law as yaw_at_progress() so lookahead sampling
    // matches the published stitched profile exactly.
    std::vector<Eigen::Vector3d> cache_positions;
    std::vector<Eigen::Vector3d> cache_velocities;
    std::vector<Eigen::Vector3d> cache_accelerations;
    std::vector<double> cache_yaws, cache_arclen, cache_times;
    cache_positions.reserve(waypoints.size());
    cache_velocities.reserve(waypoints.size());
    cache_accelerations.reserve(waypoints.size());
    cache_yaws.reserve(waypoints.size());
    cache_arclen.reserve(waypoints.size());
    cache_times.reserve(waypoints.size());
    for (size_t i = 0; i < waypoints.size(); ++i)
    {
        Eigen::Vector3d p(waypoints[i].x(), waypoints[i].y(), waypoints[i].z());
        const double s = prefix_len[i];
        cache_positions.push_back(p);
        if (i + 1 < waypoints.size())
        {
            Eigen::Vector3d next(waypoints[i + 1].x(), waypoints[i + 1].y(), waypoints[i + 1].z());
            const Eigen::Vector3d delta = next - p;
            const Eigen::Vector3d dir = delta.norm() > 1.0e-6 ? (delta / delta.norm()).eval() : Eigen::Vector3d::Zero();
            cache_velocities.push_back(dir * cruise_speed_);
        }
        else
        {
            cache_velocities.push_back(Eigen::Vector3d::Zero());
        }
        cache_accelerations.push_back(Eigen::Vector3d::Zero());
        cache_arclen.push_back(s);
        cache_times.push_back(s / cruise_speed_);
        cache_yaws.push_back(yaw_at_progress(s));
    }
    const uint64_t traj_id = cachePublishedTrajectory(std::move(cache_positions),
                                                      std::move(cache_velocities),
                                                      std::move(cache_accelerations),
                                                      std::move(cache_yaws),
                                                      std::move(cache_arclen),
                                                      std::move(cache_times),
                                                      cruise_speed_, t_pub);
    // Anchor time on the linearly-interpolated profile equals the seam's
    // arclength divided by the (constant) cruise speed -- by construction
    // it is exactly the time_from_start value cached at index split_index.
    result.published = true;
    result.traj_id = traj_id;
    result.trajectory_stamp = traj.header.stamp;
    result.total_duration = t;
    result.anchor_time = split_len / cruise_speed_;
    return result;
}

// Degenerate-input fallback: publish a stationary, yaw-only trajectory at
// `pos`. Called from interpolateAndPublishTrajectory when total path length
// collapses to zero (start ≈ goal in position). Not a planner-level branch:
// reach decisions are still driven by the standard optimizer-published cache
// + TF-based viewpoint checks elsewhere. Keeping this purely as a
// "something must be published so the controller does not chase stale
// trajectory" safety net.
FFAPlannerNode::PublishResult FFAPlannerNode::publishYawOnlyDegenerateTrajectory(
    const Eigen::Vector3d &pos,
    double start_yaw,
    double target_yaw)
{
    clearOptimizedTrajectoryVis();
    clearSfcDebugVis();

    double yaw_diff = wrapToPi(target_yaw - start_yaw);
    double yaw_rate = 1.0; // rad/s
    double duration = std::fabs(yaw_diff) / yaw_rate;
    duration = std::max(duration, 0.1);

    trajectory_msgs::msg::MultiDOFJointTrajectory traj;
    traj.header.stamp = node_handle_->now();
    traj.header.frame_id = world_frame_id_;
    traj.joint_names.push_back(robot_frame_id_);

    auto make_pt = [&](double yaw, double t)
    {
        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
        geometry_msgs::msg::Transform tf;
        tf.translation.x = pos.x();
        tf.translation.y = pos.y();
        tf.translation.z = pos.z();
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, yaw);
        tf.rotation = tf2::toMsg(q);
        pt.transforms.push_back(tf);

        geometry_msgs::msg::Twist tw; // zero velocity
        pt.velocities.push_back(tw);
        pt.time_from_start = rclcpp::Duration::from_seconds(t);
        return pt;
    };

    int n_steps = std::max(static_cast<int>(duration / 0.1), 2);
    for (int i = 0; i <= n_steps; ++i)
    {
        double frac = static_cast<double>(i) / n_steps;
        double yaw = start_yaw + yaw_diff * frac;
        double t = duration * frac;
        traj.points.push_back(make_pt(yaw, t));
    }

    RCLCPP_WARN(node_handle_->get_logger(),
                "[Local] Publishing yaw-only degenerate trajectory (path collapsed in position): "
                "pos [%.2f, %.2f, %.2f] yaw %.2f -> %.2f (%.1fs)",
                pos.x(), pos.y(), pos.z(), start_yaw, target_yaw, duration);
    const rclcpp::Time t_pub(traj.header.stamp);
    pub_trajectory_->publish(traj);

    // Stationary cache: two identical positions with linearly interpolated
    // yaw over [0, duration]. cruise_speed=0 flags this as yaw-only so that
    // lookahead sampling interpolates via time rather than arclength.
    std::vector<Eigen::Vector3d> cache_positions{pos, pos};
    std::vector<Eigen::Vector3d> cache_velocities{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    std::vector<Eigen::Vector3d> cache_accelerations{Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    std::vector<double> cache_yaws{start_yaw, start_yaw + yaw_diff};
    std::vector<double> cache_arclen{0.0, 0.0};
    std::vector<double> cache_times{0.0, duration};
    const uint64_t traj_id = cachePublishedTrajectory(std::move(cache_positions),
                                                      std::move(cache_velocities),
                                                      std::move(cache_accelerations),
                                                      std::move(cache_yaws),
                                                      std::move(cache_arclen),
                                                      std::move(cache_times),
                                                      0.0, t_pub);
    PublishResult result;
    result.published = true;
    result.traj_id = traj_id;
    result.trajectory_stamp = traj.header.stamp;
    result.total_duration = duration;
    // No anchor on a degenerate yaw-only fallback.
    return result;
}

// Cache the sparse trajectory we just handed to the controller so the
// lookahead-point sampler (to be added) can reproduce (pos, yaw, vel, acc)
// at arbitrary wall-clock times without touching the dense controller-facing
// MultiDOFJointTrajectory.  Writers call this under no other lock; readers
// take last_traj_mtx_ in shared mode.
uint64_t FFAPlannerNode::cachePublishedTrajectory(
    std::vector<Eigen::Vector3d> positions,
    std::vector<Eigen::Vector3d> velocities,
    std::vector<Eigen::Vector3d> accelerations,
    std::vector<double> yaws,
    std::vector<double> cum_arclen,
    std::vector<double> times_from_start,
    double cruise_speed,
    const rclcpp::Time &t_published)
{
    if (positions.size() < 2 ||
        positions.size() != velocities.size() ||
        positions.size() != accelerations.size() ||
        positions.size() != yaws.size() ||
        positions.size() != cum_arclen.size() ||
        positions.size() != times_from_start.size())
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[LookaheadCache] skipping cache update: inconsistent sizes "
                    "(pos=%zu vel=%zu acc=%zu yaws=%zu arclen=%zu times=%zu)",
                    positions.size(), velocities.size(), accelerations.size(),
                    yaws.size(), cum_arclen.size(), times_from_start.size());
        return 0;
    }

    // Defensive unwrap: ensure the cached yaw sequence is continuous so that
    // sampleLastPublishedTrajectory's plain lerp does not synthesize a
    // ~2*pi sweep across any seam. Each adjacent pair is brought within
    // [-pi, pi] of its predecessor; the absolute value drifts but the
    // published quaternions and the lookahead arrow remain consistent.
    for (size_t i = 1; i < yaws.size(); ++i)
    {
        const double delta = wrapToPi(yaws[i] - yaws[i - 1]);
        yaws[i] = yaws[i - 1] + delta;
    }

    const uint64_t traj_id = next_traj_id_.fetch_add(1, std::memory_order_relaxed);
    const double duration = times_from_start.back();
    const double length = cum_arclen.back();

    {
        std::unique_lock<std::shared_mutex> lk(last_traj_mtx_);
        last_published_traj_.traj_id = traj_id;
        last_published_traj_.t_published = t_published;
        last_published_traj_.positions = std::move(positions);
        last_published_traj_.velocities = std::move(velocities);
        last_published_traj_.accelerations = std::move(accelerations);
        last_published_traj_.yaws = std::move(yaws);
        last_published_traj_.cum_arclen = std::move(cum_arclen);
        last_published_traj_.times_from_start = std::move(times_from_start);
        last_published_traj_.cruise_speed = cruise_speed;
        last_published_traj_.valid = true;
    }

    RCLCPP_DEBUG(node_handle_->get_logger(),
                 "[LookaheadCache] updated id=%lu stamp=%.3f dur=%.2fs len=%.2fm cruise=%.2f",
                 static_cast<unsigned long>(traj_id),
                 t_published.seconds(), duration, length, cruise_speed);
    return traj_id;
}

void FFAPlannerNode::invalidateLastPublishedTrajectory()
{
    {
        std::unique_lock<std::shared_mutex> lk(last_traj_mtx_);
        if (last_published_traj_.valid)
        {
            last_published_traj_.valid = false;
            last_published_traj_.positions.clear();
            last_published_traj_.velocities.clear();
            last_published_traj_.accelerations.clear();
            last_published_traj_.yaws.clear();
            last_published_traj_.cum_arclen.clear();
            last_published_traj_.times_from_start.clear();
            RCLCPP_DEBUG(node_handle_->get_logger(),
                         "[LookaheadCache] invalidated (prev id=%lu)",
                         static_cast<unsigned long>(last_published_traj_.traj_id));
        }
    }
    {
        std::unique_lock<std::shared_mutex> lk(lookahead_mtx_);
        cached_lookahead_ = LookaheadState{};
    }
    tp_divergent_.store(false, std::memory_order_relaxed);
    clearLookaheadVis();
}

bool FFAPlannerNode::sampleLastPublishedTrajectory(double t_eval,
                                                   Eigen::Vector3d &pos,
                                                   Eigen::Vector3d &vel,
                                                   Eigen::Vector3d &acc,
                                                   double &yaw,
                                                   bool &at_end,
                                                   uint64_t &traj_id_out) const
{
    std::shared_lock<std::shared_mutex> lk(last_traj_mtx_);
    const PublishedTrajectory &c = last_published_traj_;
    if (!c.valid || c.positions.size() < 2)
    {
        return false;
    }

    traj_id_out = c.traj_id;
    at_end = false;

    const auto &times = c.times_from_start;
    const auto &P = c.positions;
    const auto &V = c.velocities;
    const auto &A = c.accelerations;
    const auto &Y = c.yaws;
    const double t_end = times.back();

    if (t_eval <= times.front())
    {
        pos = P.front();
        yaw = Y.front();
        vel = V.front();
        acc = A.front();
        return true;
    }
    if (t_eval >= t_end)
    {
        pos = P.back();
        yaw = Y.back();
        vel = V.back();
        acc = A.back();
        at_end = true;
        return true;
    }

    // Segment [i, i+1] with times[i] <= t_eval < times[i+1].
    auto it = std::upper_bound(times.begin(), times.end(), t_eval);
    const size_t i1 = static_cast<size_t>(it - times.begin());
    const size_t i0 = i1 - 1;
    const double t0 = times[i0];
    const double t1 = times[i1];
    const double dt = std::max(t1 - t0, 1e-9);
    const double frac = (t_eval - t0) / dt;

    pos = P[i0] + frac * (P[i1] - P[i0]);
    // Yaws stored as the pre-unwrapped monotonic sweep, so plain lerp is fine.
    yaw = Y[i0] + frac * (Y[i1] - Y[i0]);
    if (c.cruise_speed <= 0.0)
    {
        // Yaw-in-place cache: stationary, zero velocity.
        vel = V[i0] + frac * (V[i1] - V[i0]);
        acc = A[i0] + frac * (A[i1] - A[i0]);
    }
    else
    {
        vel = V[i0] + frac * (V[i1] - V[i0]);
        acc = A[i0] + frac * (A[i1] - A[i0]);
    }
    return true;
}

bool FFAPlannerNode::projectPointOntoLastPublishedTrajectory(const Eigen::Vector3d &query_pos,
                                                             double &time_from_start,
                                                             double &arclen,
                                                             uint64_t &traj_id_out,
                                                             double *distance_to_traj_out) const
{
    std::shared_lock<std::shared_mutex> lk(last_traj_mtx_);
    const PublishedTrajectory &c = last_published_traj_;
    if (!c.valid || c.positions.size() < 2 ||
        c.positions.size() != c.times_from_start.size() ||
        c.positions.size() != c.cum_arclen.size())
    {
        return false;
    }

    const auto &P = c.positions;
    const auto &times = c.times_from_start;
    const auto &S = c.cum_arclen;

    double best_dist_sq = std::numeric_limits<double>::infinity();
    double best_time = 0.0;
    double best_s = 0.0;
    bool found = false;

    for (size_t i = 0; i + 1 < P.size(); ++i)
    {
        const Eigen::Vector3d seg = P[i + 1] - P[i];
        const double seg_len_sq = seg.squaredNorm();
        if (seg_len_sq < 1.0e-12)
        {
            continue;
        }

        const double frac = std::clamp((query_pos - P[i]).dot(seg) / seg_len_sq, 0.0, 1.0);
        const Eigen::Vector3d proj = P[i] + frac * seg;
        const double dist_sq = (query_pos - proj).squaredNorm();
        if (dist_sq < best_dist_sq)
        {
            best_dist_sq = dist_sq;
            best_time = times[i] + frac * (times[i + 1] - times[i]);
            best_s = S[i] + frac * (S[i + 1] - S[i]);
            found = true;
        }
    }

    if (!found)
    {
        return false;
    }

    time_from_start = best_time;
    arclen = best_s;
    traj_id_out = c.traj_id;
    if (distance_to_traj_out)
    {
        *distance_to_traj_out = std::sqrt(best_dist_sq);
    }
    return true;
}

void FFAPlannerNode::refreshLookaheadFromTracking(const rclcpp::Time &now,
                                                  const Eigen::Vector3d &tp_pos,
                                                  double tp_time,
                                                  const builtin_interfaces::msg::Time &trajectory_stamp)
{
    // Snapshot cache metadata we need outside the shared lock scope.
    rclcpp::Time t_published;
    bool cache_valid = false;
    uint64_t cache_id = 0;
    {
        std::shared_lock<std::shared_mutex> lk(last_traj_mtx_);
        cache_valid = last_published_traj_.valid;
        cache_id = last_published_traj_.traj_id;
        if (cache_valid)
        {
            t_published = last_published_traj_.t_published;
        }
    }

    if (!cache_valid)
    {
        {
            std::unique_lock<std::shared_mutex> lk(lookahead_mtx_);
            cached_lookahead_ = LookaheadState{};
        }
        tp_divergent_.store(false, std::memory_order_relaxed);
        clearLookaheadVis();
        return;
    }

    const double t_elapsed = std::max(0.0, (now - t_published).seconds());
    double t_eval = t_elapsed + lookahead_buffer_time_s_;
    if (lookahead_progress_driven_)
    {
        // Progress-driven lookahead (real-robot semantics): anchor t_eval to
        // the tracker-reported time_from_start so the reference never runs
        // open-loop ahead of a lagging controller. Only trust tp_time when
        // it refers to the trajectory we cached. A brief mismatch right
        // after each publish is expected (the tracker echoes the old stamp
        // for a tick or two, and the previous lookahead stays valid); a
        // persistent mismatch means the tracker never received the new
        // trajectory.
        if (!stampsEqual(trajectory_stamp, t_published))
        {
            if ((now - t_published).seconds() > 0.5)
            {
                RCLCPP_WARN_THROTTLE(node_handle_->get_logger(), *node_handle_->get_clock(), 2000,
                                     "[Lookahead] tracker still executing traj stamp %d.%09u but cache "
                                     "was published at %.3f; skipping lookahead refresh",
                                     trajectory_stamp.sec, trajectory_stamp.nanosec,
                                     t_published.seconds());
            }
            return;
        }
        t_eval = std::max(0.0, tp_time) + lookahead_buffer_time_s_;
    }
    // else: wall-clock schedule (benchmark-era semantics) — the reference
    // advances with published-trajectory time regardless of tracker progress.

    // Sample once at t_elapsed (where the wall-clock schedule says the
    // controller should be, used only as a lag indicator) and once at t_eval
    // (tracker progress + buffer) for the lookahead output.
    Eigen::Vector3d now_pos, now_vel, now_acc, la_pos, la_vel, la_acc;
    double now_yaw = 0.0, la_yaw = 0.0;
    bool now_at_end = false, la_at_end = false;
    uint64_t id_check = 0, id_check2 = 0;
    const bool ok_now = sampleLastPublishedTrajectory(t_elapsed, now_pos, now_vel, now_acc,
                                                      now_yaw, now_at_end, id_check);
    const bool ok_la = sampleLastPublishedTrajectory(t_eval, la_pos, la_vel, la_acc,
                                                     la_yaw, la_at_end, id_check2);

    if (!ok_now || !ok_la || id_check != cache_id || id_check2 != cache_id)
    {
        // Cache got invalidated between snapshots; skip this update.
        return;
    }

    const double div = (tp_pos - now_pos).norm();
    const bool divergent = div > tp_divergence_threshold_m_;
    const bool prev_divergent = tp_divergent_.exchange(divergent, std::memory_order_relaxed);
    if (divergent && !prev_divergent)
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[Lookahead] tp lags wall-clock schedule: |tp - scheduled|=%.3fm, "
                    "time lag %.2fs (threshold %.2fm, id=%lu)",
                    div, t_elapsed - tp_time, tp_divergence_threshold_m_,
                    static_cast<unsigned long>(cache_id));
    }

    LookaheadState st;
    st.pos = la_pos;
    st.vel = la_vel;
    st.acc = la_acc;
    st.yaw = la_yaw;
    st.t_eval = t_eval;
    st.traj_id = cache_id;
    st.valid = true;
    st.at_end = la_at_end;

    {
        std::unique_lock<std::shared_mutex> lk(lookahead_mtx_);
        cached_lookahead_ = st;
    }

    publishLookaheadVis(st, tp_pos);
}

FFAPlannerNode::LookaheadState FFAPlannerNode::getLookaheadState() const
{
    std::shared_lock<std::shared_mutex> lk(lookahead_mtx_);
    return cached_lookahead_;
}

void FFAPlannerNode::publishLookaheadVis(const LookaheadState &state,
                                         const Eigen::Vector3d &tp_pos)
{
    if (!pub_lookahead_vis_)
    {
        return;
    }
    visualization_msgs::msg::MarkerArray arr;
    const auto stamp = node_handle_->now();
    const bool divergent = tp_divergent_.load(std::memory_order_relaxed);

    // Colors: yellow = healthy lookahead, red = divergent, magenta = at_end.
    std_msgs::msg::ColorRGBA col;
    col.a = 1.0;
    if (divergent)
    {
        col.r = 1.0; col.g = 0.2; col.b = 0.2;
    }
    else if (state.at_end)
    {
        col.r = 1.0; col.g = 0.0; col.b = 1.0;
    }
    else
    {
        col.r = 1.0; col.g = 0.9; col.b = 0.1;
    }

    // 1) sphere at lookahead position
    visualization_msgs::msg::Marker sphere;
    sphere.header.stamp = stamp;
    sphere.header.frame_id = world_frame_id_;
    sphere.ns = "lookahead";
    sphere.id = 0;
    sphere.type = visualization_msgs::msg::Marker::SPHERE;
    sphere.action = visualization_msgs::msg::Marker::ADD;
    sphere.pose.position.x = state.pos.x();
    sphere.pose.position.y = state.pos.y();
    sphere.pose.position.z = state.pos.z();
    sphere.pose.orientation.w = 1.0;
    sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.25;
    sphere.color = col;
    arr.markers.push_back(sphere);

    // 2) arrow showing lookahead yaw
    visualization_msgs::msg::Marker yaw_arrow;
    yaw_arrow.header.stamp = stamp;
    yaw_arrow.header.frame_id = world_frame_id_;
    yaw_arrow.ns = "lookahead";
    yaw_arrow.id = 1;
    yaw_arrow.type = visualization_msgs::msg::Marker::ARROW;
    yaw_arrow.action = visualization_msgs::msg::Marker::ADD;
    yaw_arrow.pose.position.x = state.pos.x();
    yaw_arrow.pose.position.y = state.pos.y();
    yaw_arrow.pose.position.z = state.pos.z();
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, state.yaw);
    yaw_arrow.pose.orientation = tf2::toMsg(q);
    yaw_arrow.scale.x = 0.6;   // length
    yaw_arrow.scale.y = 0.06;  // shaft diameter
    yaw_arrow.scale.z = 0.06;  // head diameter
    yaw_arrow.color = col;
    arr.markers.push_back(yaw_arrow);

    // 3) thin line from tracking_point to lookahead (shows buffer length)
    visualization_msgs::msg::Marker line;
    line.header.stamp = stamp;
    line.header.frame_id = world_frame_id_;
    line.ns = "lookahead";
    line.id = 2;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.pose.orientation.w = 1.0;
    line.scale.x = 0.04;
    line.color = col;
    geometry_msgs::msg::Point p0;
    p0.x = tp_pos.x(); p0.y = tp_pos.y(); p0.z = tp_pos.z();
    geometry_msgs::msg::Point p1;
    p1.x = state.pos.x(); p1.y = state.pos.y(); p1.z = state.pos.z();
    line.points.push_back(p0);
    line.points.push_back(p1);
    arr.markers.push_back(line);

    pub_lookahead_vis_->publish(arr);
}

void FFAPlannerNode::clearLookaheadVis()
{
    if (!pub_lookahead_vis_)
    {
        return;
    }
    visualization_msgs::msg::MarkerArray arr;
    visualization_msgs::msg::Marker del;
    del.header.stamp = node_handle_->now();
    del.header.frame_id = world_frame_id_;
    del.ns = "lookahead";
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);
    pub_lookahead_vis_->publish(arr);
}

// Frontier Intersection Detection
// Walk guidance path from robot position, return the first inflated-frontier
// or occ-blocked voxel plus the last strict-free voxel immediately before it.
// Copies the path under lock, then checks the inflated grid under a separate
// read lock so neither is held for long.

FFAPlannerNode::SafeEdgeHitResult
FFAPlannerNode::findFirstSafeEdgeHit(const std::vector<openvdb::Vec3d> &path_copy) const
{
    SafeEdgeHitResult result;
    if (path_copy.empty())
    {
        return result;
    }

    openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();

    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();

    bool have_last_strict_free = false;
    openvdb::Coord last_strict_free_ijk(0, 0, 0);
    auto attach_search_start = [&]()
    {
        if (!have_last_strict_free)
        {
            return;
        }
        result.has_search_start = true;
        result.search_start_ijk = last_strict_free_ijk;
        result.search_start_world = grid_tf->indexToWorld(last_strict_free_ijk);
    };

    for (size_t seg = 0; seg + 1 < path_copy.size(); ++seg)
    {
        openvdb::Coord c0 = openvdb::Coord::round(grid_tf->worldToIndex(path_copy[seg]));
        openvdb::Coord c1 = openvdb::Coord::round(grid_tf->worldToIndex(path_copy[seg + 1]));

        const int dx = c1.x() - c0.x();
        const int dy = c1.y() - c0.y();
        const int dz = c1.z() - c0.z();
        const int steps = std::max({std::abs(dx), std::abs(dy), std::abs(dz)});

        const int t_end = (seg + 2 < path_copy.size()) ? steps - 1 : steps;

        const double inv = (steps > 0) ? 1.0 / steps : 0.0;
        const double sx = dx * inv;
        const double sy = dy * inv;
        const double sz = dz * inv;

        for (int t = 0; t <= t_end; ++t)
        {
            const openvdb::Coord vox(
                c0.x() + static_cast<int>(std::round(sx * t)),
                c0.y() + static_cast<int>(std::round(sy * t)),
                c0.z() + static_cast<int>(std::round(sz * t)));

            int inflate_val = 0;
            bool active = map_manager_->query_is_inflated_at_index(vox, inflate_val, inf_acc);
            bool has_occ = active && (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
            bool has_frontier = active && inflate_val >= VDBMap::FRONTIER_INFLATION_DELTA;

            auto fill_hit_probe = [&]()
            {
                result.hit_idx = seg;
                result.hit_pt = grid_tf->indexToWorld(vox);
                result.hit_ijk = vox;
                result.hit_inflate_val = active ? inflate_val : 0;
                int below_val = 0;
                const bool below_active = map_manager_->query_is_inflated_at_index(
                    vox.offsetBy(0, 0, -1), below_val, inf_acc);
                result.below_inflate_val = below_active ? below_val : 0;
                attach_search_start();
            };

            if (has_occ)
            {
                result.path_blocked = true;
                fill_hit_probe();
                return result;
            }

            if (has_frontier)
            {
                result.safe_edge_reached = true;
                fill_hit_probe();
                return result;
            }

            have_last_strict_free = true;
            last_strict_free_ijk = vox;
        }
    }

    return result;
}


// main

#include <rclcpp/executors/multi_threaded_executor.hpp>

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto planner = std::make_shared<FFAPlannerNode>();
    auto node = planner->get_node_handle();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}
