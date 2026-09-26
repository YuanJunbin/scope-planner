// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include "astar_vdb.hpp"
#include "fov_astar.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <queue>
#include <unordered_set>

// A* search that finds the nearest safe viewpoint from which a given
// unknown voxel can be observed by the sensor.
//
// Differences from FovAstar:
//   - Target is a single unknown voxel (not a hit point with inflation box)
//   - FOV check has NO inflation box shift — just point-in-frustum
//   - Goal additionally requires ray-trace verification (logocc grid,
//     only real occupied blocks)
//   - isBlocked is the same (blocks ALL inflated voxels)
//   - Escape corridor reused from FovAstar pattern
class ObserveAstar : public Astar
{
public:
    // Fresh search: resets all session state and runs up to max_search_time_.
    int search(const openvdb::Coord &start_pt, const openvdb::Coord &end_pt);
    // Continue the previous search() session (same start / target) for
    // another `budget_s` seconds, keeping open/closed sets and the candidate
    // lists. Returns PATH_NOT_FOUND if no session is active. Never reports
    // EXHAUSTED_NO_GOAL: an exhausted resume that yields no candidate reports
    // EXHAUSTED, because contaminated branches may have been pruned once a
    // certified candidate was seen; the caller decides whether to restart.
    int resume(double budget_s);
    bool hasSession() const { return session_valid_; }
    bool sessionMatches(const openvdb::Coord &start_pt,
                        const openvdb::Coord &target_unknown) const
    {
        return session_valid_ && session_start_ == start_pt &&
               session_target_ == target_unknown;
    }
    bool openSetEmpty() const { return label_open_set_.empty(); }
    // Drop candidates that are no longer valid under the current map
    // (voxel occ-inflated, or the sensor->target ray blocked) or that are
    // in `failed`, refreshing the grazing flag of the survivors. FOV
    // geometry is NOT re-evaluated: it is constant for a fixed target.
    // Returns the number of candidates removed.
    size_t revalidateCandidates(const std::unordered_set<openvdb::Coord, CoordHash> &failed);
    // Number of ray voxels adjacent to the target that are ignored by the
    // grazing test (the target's own surroundings are shared by all
    // viewpoints and carry no discriminative information).
    void setOccShellSkipVoxels(int n) { occ_shell_skip_vox_ = std::max(0, n); }
    // Pure FOV geometry test (range + vertical angle, yaw aligned to the
    // current target) at an arbitrary body position; no line-of-sight test.
    bool fovSatisfiedAt(const openvdb::Vec3d &body_world) const
    {
        return evaluateObserveFov(body_world).satisfied;
    }
    void setFovConfig(const SensorFovConfig &cfg);
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
    void setGoalCandidatePredicate(FovAstar::GoalCandidatePredicate predicate)
    {
        goal_candidate_predicate_ = std::move(predicate);
    }
    void clearGoalCandidatePredicate()
    {
        goal_candidate_predicate_ = nullptr;
    }

    void setTargetUnknown(const openvdb::Coord &unknown_ijk);

    // Soft standoff: edges leaving a node with any occ-inflated 26-neighbor
    // cost mult x (1 = off). Feasibility/admissibility unchanged.
    void setClearanceSoftCost(double mult) { clearance_soft_cost_mult_ = std::max(mult, 1.0); }

    void buildEscapeCorridor(const openvdb::Coord &start_ijk, double body_yaw);
    void clearEscapeCorridor();

    void pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out);
    bool extractPathToGoal(const openvdb::Coord &goal_ijk, ReachLabel label);

    double getTargetYaw() const { return target_yaw_; }
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
        bool closed[2] = {false, false};
        double best_g[2] = {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()};
    };

    struct ObserveFovResult
    {
        bool satisfied = false;
        double deficit_range = 0.0;
        double deficit_min_range = 0.0;
        double deficit_theta = 0.0;
    };

    ObserveFovResult evaluateObserveFov(const openvdb::Vec3d &body_world) const;
    void backtrackLabeled(const LabelSearchNode *end_node);
    TraversalClass classifyTraversal(const openvdb::Coord &coord,
                                     int inflate_val,
                                     bool is_active) const;
    ReachLabel advanceLabel(ReachLabel current_label,
                            TraversalClass traversal) const;
    int labelIndex(ReachLabel label) const;
    bool rayTraceClear(const openvdb::Vec3d &sensor_world,
                       const openvdb::Vec3d &target_world) const;
    // Lock-free variant used inside the search loop / revalidation where the
    // caller already holds the map read lock and owns the accessors.
    // shell_acc may be null (shell grid disabled) -> grazing stays false.
    bool rayTraceClearWithAcc(const openvdb::Vec3d &sensor_world,
                              const openvdb::Vec3d &target_world,
                              openvdb::FloatGrid::ConstAccessor &occ_acc,
                              openvdb::Int32Grid::ConstAccessor *shell_acc,
                              bool &grazing_out) const;
    // Full goal test (FOV geometry + LOS) with accessor reuse; also reports
    // grazing. Equivalent to isGoal() when the FOV part passes.
    bool evaluateGoalWithAcc(const openvdb::Coord &current,
                             openvdb::FloatGrid::ConstAccessor &occ_acc,
                             openvdb::Int32Grid::ConstAccessor *shell_acc,
                             bool &grazing_out) const;
    openvdb::Vec3d sensorWorldForBody(const openvdb::Vec3d &body_world, double psi) const;

    void beginSearch(const openvdb::Coord &start_ijk);
    int runBudget(double budget_s, bool is_resume);
    void recordGoalCandidate(LabelSearchNode *cur, bool grazing);
    void refreshBestGoalPointers();

    struct LabelNodeCmp
    {
        bool operator()(const LabelSearchNode *lhs, const LabelSearchNode *rhs) const
        {
            return lhs->f_score > rhs->f_score;
        }
    };
    std::priority_queue<LabelSearchNode *,
                        std::vector<LabelSearchNode *>,
                        LabelNodeCmp>
        label_open_set_;
    bool session_valid_ = false;
    openvdb::Coord session_start_{0, 0, 0};
    openvdb::Coord session_target_{0, 0, 0};
    bool certified_found_ = false;
    LabelSearchNode *best_certified_goal_ = nullptr;
    LabelSearchNode *best_contaminated_goal_ = nullptr;
    int occ_shell_skip_vox_ = 2;
    // Cheap geometric "stay near search anchor" bias applied on every step
    // (no FOV/LOS evaluation). Intended for tie-breaking in the bulk search.
    double computeAnchorBiasMultiplier(const openvdb::Coord &current,
                                       const openvdb::Coord &next) const;

    SensorFovConfig cfg_;
    openvdb::Coord target_ijk_{0, 0, 0};
    openvdb::Vec3d target_world_{0, 0, 0};

    Eigen::Matrix3d R_bs_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_bs_ = Eigen::Vector3d::Zero();
    double psi_off_ = 0.0;
    double safe_range_ = 0.0;
    double min_range_ = 0.0;
    bool allow_soft_frontier_traversal_ = false;
    openvdb::Vec3d search_anchor_world_{0, 0, 0};
    bool has_search_anchor_ = false;
    double anchor_bias_away_weight_ = 0.1;
    double anchor_bias_max_multiplier_ = 1.1;

    mutable double target_yaw_ = 0.0;

    std::unordered_set<openvdb::Coord, CoordHash> escape_set_;
    double clearance_soft_cost_mult_ = 1.0;
    std::unordered_map<openvdb::Coord, DualLabelRecord, CoordHash> state_table_;
    std::vector<std::unique_ptr<LabelSearchNode>> search_nodes_;
    std::unordered_map<openvdb::Coord, LabelSearchNode *, CoordHash> certified_goal_nodes_;
    std::unordered_map<openvdb::Coord, LabelSearchNode *, CoordHash> contaminated_goal_nodes_;
    std::vector<SearchGoalCandidate> certified_goal_candidates_;
    std::vector<SearchGoalCandidate> contaminated_goal_candidates_;
    ReachLabel last_path_label_ = ReachLabel::CERTIFIED;
    GoalSearchStopReason last_goal_search_stop_reason_ = GoalSearchStopReason::NONE;
    FovAstar::GoalCandidatePredicate goal_candidate_predicate_;
};
