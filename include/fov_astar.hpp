// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include "astar_vdb.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

struct SensorFovConfig
{
    // Body-to-sensor translation (body frame: x=forward, y=left, z=up)
    double body_x = 0.0;
    double body_y = 0.0;
    double body_z = 0.0;

    // Body-to-sensor rotation RPY (rotations about body x, y, z)
    double body_roll = 0.0;
    double body_pitch = 0.0;
    double body_yaw = 0.0;

    // Vertical FOV limits (radians, relative to sensor optical axis)
    double theta_u = 0.3;
    double theta_d = -0.3;

    double range = 10.0;
    double fov_min_range = 0.5;
    double observe_min_range = 0.5;
    // Multiplier applied to any step whose resulting search label is
    // CONTAMINATED. We keep the legacy parameter name for compatibility.
    double soft_frontier_cost_multiplier = 3.0;
};

enum class FovClearProfileKind : uint8_t
{
    STANDARD,
    TIGHT
};

struct FovClearProfile
{
    double inflate_rxy = 0.6;
    double inflate_rz = 0.2;
};

struct SearchGoalCandidate
{
    openvdb::Coord coord{0, 0, 0};
    double g_score = 0.0;
    // Observe-only: the sensor->target ray passed through the occ-shell
    // (within occ_shell_radius of a known-occupied voxel) somewhere before
    // the last few voxels at the target. Such a viewpoint is physically
    // marginal (edge grazing). Always false when the shell grid is disabled.
    bool grazing = false;
};

struct PreviewProgressStats
{
    size_t closed_certified = 0;
    size_t ray_checks = 0;
    int best_sample_index = -1;
};

enum class ReachLabel : uint8_t
{
    CERTIFIED = 0,
    CONTAMINATED = 1
};

enum class GoalSearchStopReason : uint8_t
{
    NONE = 0,
    TIME_LIMIT,
    EXHAUSTED,
    EXHAUSTED_NO_GOAL
};

enum class TraversalClass : uint8_t
{
    BLOCKED_OCC = 0,
    STRICT_FREE = 1,
    SOFT = 2
};

// A* search that finds the nearest free-space viewpoint from which a given
// target point P can be safely observed through the sensor FOV.
//
// Differences from base Astar:
//   isGoal   — FOV constraint satisfaction instead of coordinate match
//   heuristic — FOV deficit distance instead of diagonal grid distance
//   isBlocked — blocks ALL inflated voxels (occ + frontier), not just occ
//
// Escape corridor: at startup the robot may sit inside the inflated-frontier
// zone.  buildEscapeCorridor() carves a 10-voxel corridor along body +x so
// the search can reach guaranteed-free space.  Corridor nodes bypass
// isBlocked and use h=0 to be expanded first.
class FovAstar : public Astar
{
public:
    using GoalCandidatePredicate = std::function<bool(const SearchGoalCandidate &, ReachLabel)>;

    int search(const openvdb::Coord &start_pt, const openvdb::Coord &end_pt);
    void setFovConfig(const SensorFovConfig &cfg);
    void setClearProfile(FovClearProfileKind kind, const FovClearProfile &profile);
    void setActiveClearProfile(FovClearProfileKind kind);
    void setTarget(const openvdb::Vec3d &target_world);
    void setAllowSoftFrontierTraversal(bool enable) { allow_soft_frontier_traversal_ = enable; }
    void setSearchAnchor(const openvdb::Vec3d &anchor_world)
    {
        search_anchor_world_ = anchor_world;
        has_search_anchor_ = true;
    }
    void clearSearchAnchor() { has_search_anchor_ = false; }
    void setAnchorBias(double away_weight, double max_multiplier)
    {
        anchor_bias_away_weight_ = std::max(0.0, away_weight);
        anchor_bias_max_multiplier_ = std::max(1.0, max_multiplier);
    }
    void setGoalCandidatePredicate(GoalCandidatePredicate predicate)
    {
        goal_candidate_predicate_ = std::move(predicate);
    }
    void clearGoalCandidatePredicate()
    {
        goal_candidate_predicate_ = nullptr;
    }
    void setPreviewProgressSamples(const std::vector<openvdb::Vec3d> &samples,
                                   const std::vector<double> &sample_progress);
    void clearPreviewProgressSamples();
    const std::vector<std::vector<SearchGoalCandidate>> &getPreviewProgressBuckets() const
    {
        return preview_progress_buckets_;
    }
    PreviewProgressStats getPreviewProgressStats() const
    {
        return preview_progress_stats_;
    }

    // Soft standoff: edges leaving a node with any occ-inflated 26-neighbor
    // cost mult x (1 = off). Feasibility/admissibility unchanged.
    void setClearanceSoftCost(double mult) { clearance_soft_cost_mult_ = std::max(mult, 1.0); }

    void buildEscapeCorridor(const openvdb::Coord &start_ijk, double body_yaw);
    void clearEscapeCorridor();

    void pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out);
    bool extractPathToGoal(const openvdb::Coord &goal_ijk, ReachLabel label);
    bool extractPathToClosedState(const openvdb::Coord &ijk, ReachLabel label);
    ReachLabel getLastPathLabel() const { return last_path_label_; }
    GoalSearchStopReason getLastGoalSearchStopReason() const { return last_goal_search_stop_reason_; }
    const std::vector<SearchGoalCandidate> &getCertifiedGoalCandidates() const
    {
        return certified_goal_candidates_;
    }
    const std::vector<SearchGoalCandidate> &getContaminatedGoalCandidates() const
    {
        return contaminated_goal_candidates_;
    }
    std::vector<SearchGoalCandidate> getClosedCandidates(ReachLabel label) const;
    visualization_msgs::msg::Marker getDebugTreeMarker(const std::string &frame_id) const;

protected:
    bool isGoal(const openvdb::Coord &current, const openvdb::Coord &target) const override;
    double computeHeuristic(const openvdb::Coord &from, const openvdb::Coord &target) const override;
    bool isBlocked(const openvdb::Coord &coord, int inflate_val, bool is_active) const override;
    double getTraversalCostMultiplier(const openvdb::Coord &coord,
                                      int inflate_val,
                                      bool is_active) const override;
    double getEdgeCostMultiplier(const openvdb::Coord &current,
                                 const openvdb::Coord &next,
                                 int inflate_val,
                                 bool is_active) const override;

private:
    struct LabelSearchNode
    {
        openvdb::Coord ijk;
        ReachLabel label = ReachLabel::CERTIFIED;
        double f_score = 0.0;
        double g_score = 0.0;
        LabelSearchNode *parent = nullptr;
    };

    struct DualLabelRecord
    {
        LabelSearchNode *open_nodes[2] = {nullptr, nullptr};
        const LabelSearchNode *closed_nodes[2] = {nullptr, nullptr};
        bool closed[2] = {false, false};
        double best_g[2] = {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
    };

    void updateActiveProfileCache();
    void backtrackLabeled(const LabelSearchNode *end_node);
    TraversalClass classifyTraversal(const openvdb::Coord &coord,
                                     int inflate_val,
                                     bool is_active) const;
    ReachLabel advanceLabel(ReachLabel current_label,
                            TraversalClass traversal) const;
    int labelIndex(ReachLabel label) const;
    struct FovCheckResult
    {
        bool satisfied = false;
        double deficit_range = 0.0;
        double deficit_min_range = 0.0;
        double deficit_theta1 = 0.0;
        double deficit_theta2 = 0.0;
    };

    FovCheckResult evaluateFov(const openvdb::Vec3d &body_world) const;
    FovCheckResult evaluateFov(const openvdb::Vec3d &body_world,
                               const openvdb::Vec3d &target_world) const;
    openvdb::Vec3d sensorWorldForTarget(const openvdb::Vec3d &body_world,
                                        const openvdb::Vec3d &target_world) const;
    bool isGoalWithAccessor(const openvdb::Coord &current,
                            const openvdb::Vec3d &target_world,
                            openvdb::FloatGrid::ConstAccessor &occ_acc) const;
    bool rayTraceClear(const openvdb::Vec3d &sensor_world,
                       const openvdb::Vec3d &target_world) const;
    bool rayTraceClearWithAccessor(const openvdb::Vec3d &sensor_world,
                                   const openvdb::Vec3d &target_world,
                                   openvdb::FloatGrid::ConstAccessor &occ_acc) const;
    void resetPreviewProgressResults();
    void updatePreviewProgressBuckets(const LabelSearchNode *node,
                                      openvdb::FloatGrid::ConstAccessor &occ_acc);
    // Cheap geometric "stay near search anchor" bias applied on every step
    // (no FOV/LOS evaluation). Intended for tie-breaking in the bulk search.
    double computeAnchorBiasMultiplier(const openvdb::Coord &current,
                                       const openvdb::Coord &next) const;

    SensorFovConfig cfg_;
    FovClearProfile standard_profile_;
    FovClearProfile tight_profile_;
    FovClearProfileKind active_profile_kind_ = FovClearProfileKind::STANDARD;
    openvdb::Vec3d target_world_{0, 0, 0};

    // Precomputed from config
    Eigen::Matrix3d R_bs_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_bs_ = Eigen::Vector3d::Zero();
    double psi_off_ = 0.0;
    double active_inflate_rxy_ = 0.0;
    double active_inflate_rz_ = 0.0;
    double active_margin_R_ = 0.0;
    double active_safe_range_ = 0.0;
    double active_safe_min_range_ = 0.0;
    bool allow_soft_frontier_traversal_ = false;
    openvdb::Vec3d search_anchor_world_{0, 0, 0};
    bool has_search_anchor_ = false;
    double anchor_bias_away_weight_ = 0.1;
    double anchor_bias_max_multiplier_ = 1.1;

    // Startup escape corridor (non-occupied voxels along body +x)
    std::unordered_set<openvdb::Coord, CoordHash> escape_set_;
    double clearance_soft_cost_mult_ = 1.0;

    std::unordered_map<openvdb::Coord, DualLabelRecord, CoordHash> state_table_;
    std::vector<std::unique_ptr<LabelSearchNode>> search_nodes_;
    std::unordered_map<openvdb::Coord, const LabelSearchNode *, CoordHash> certified_goal_nodes_;
    std::unordered_map<openvdb::Coord, const LabelSearchNode *, CoordHash> contaminated_goal_nodes_;
    std::vector<SearchGoalCandidate> certified_goal_candidates_;
    std::vector<SearchGoalCandidate> contaminated_goal_candidates_;
    ReachLabel last_path_label_ = ReachLabel::CERTIFIED;
    GoalSearchStopReason last_goal_search_stop_reason_ = GoalSearchStopReason::NONE;
    GoalCandidatePredicate goal_candidate_predicate_;
    std::vector<openvdb::Vec3d> preview_progress_samples_;
    std::vector<double> preview_sample_progress_;
    std::vector<std::vector<SearchGoalCandidate>> preview_progress_buckets_;
    PreviewProgressStats preview_progress_stats_;
};
