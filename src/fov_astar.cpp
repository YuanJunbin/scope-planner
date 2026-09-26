// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/fov_astar.hpp"
#include <vdb_edt/ray_trace_util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>

void FovAstar::setFovConfig(const SensorFovConfig &cfg)
{
    cfg_ = cfg;

    t_bs_ = Eigen::Vector3d(cfg.body_x, cfg.body_y, cfg.body_z);

    R_bs_ = (Eigen::AngleAxisd(cfg.body_yaw, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(cfg.body_pitch, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(cfg.body_roll, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();

    psi_off_ = cfg.body_yaw;
    updateActiveProfileCache();
}

void FovAstar::setClearProfile(FovClearProfileKind kind,
                               const FovClearProfile &profile)
{
    if (kind == FovClearProfileKind::STANDARD)
    {
        standard_profile_ = profile;
    }
    else
    {
        tight_profile_ = profile;
    }

    if (kind == active_profile_kind_)
    {
        updateActiveProfileCache();
    }
}

void FovAstar::setActiveClearProfile(FovClearProfileKind kind)
{
    active_profile_kind_ = kind;
    updateActiveProfileCache();
}

void FovAstar::updateActiveProfileCache()
{
    const FovClearProfile &profile =
        (active_profile_kind_ == FovClearProfileKind::STANDARD) ? standard_profile_ : tight_profile_;
    // The inflation box around the target is world-axis-aligned, so its
    // horizontal half-extent as seen along an arbitrary azimuth reaches
    // sqrt(2) * rxy at the 45-degree diagonal. Using that worst case here
    // makes margin_R the full 3D half-diagonal sqrt(2*rxy^2 + rz^2) and
    // shifts the vertical-angle near face accordingly, so the whole box is
    // guaranteed inside the frustum regardless of viewing azimuth.
    active_inflate_rxy_ = std::sqrt(2.0) * profile.inflate_rxy;
    active_inflate_rz_ = profile.inflate_rz;
    active_margin_R_ = std::sqrt(active_inflate_rxy_ * active_inflate_rxy_ +
                                 active_inflate_rz_ * active_inflate_rz_);
    active_safe_range_ = std::max(cfg_.range - active_margin_R_, 0.0);
    active_safe_min_range_ = std::max(cfg_.fov_min_range + active_margin_R_, 0.0);
}

void FovAstar::setTarget(const openvdb::Vec3d &target_world)
{
    target_world_ = target_world;
}

void FovAstar::setPreviewProgressSamples(const std::vector<openvdb::Vec3d> &samples,
                                         const std::vector<double> &sample_progress)
{
    preview_progress_samples_.clear();
    preview_sample_progress_.clear();
    preview_progress_buckets_.clear();
    preview_progress_stats_ = PreviewProgressStats{};

    if (samples.empty() || samples.size() != sample_progress.size())
    {
        return;
    }

    preview_progress_samples_ = samples;
    preview_sample_progress_ = sample_progress;
    preview_progress_buckets_.resize(preview_progress_samples_.size());
}

void FovAstar::clearPreviewProgressSamples()
{
    preview_progress_samples_.clear();
    preview_sample_progress_.clear();
    preview_progress_buckets_.clear();
    preview_progress_stats_ = PreviewProgressStats{};
}

void FovAstar::resetPreviewProgressResults()
{
    for (auto &bucket : preview_progress_buckets_)
    {
        bucket.clear();
    }
    preview_progress_stats_ = PreviewProgressStats{};
}

void FovAstar::buildEscapeCorridor(const openvdb::Coord &start_ijk, double body_yaw)
{
    escape_set_.clear();

    auto grid_tf = map_manager_->get_grid_transform();
    openvdb::Vec3d start_world = grid_tf->indexToWorld(start_ijk);

    const double cos_yaw = std::cos(body_yaw);
    const double sin_yaw = std::sin(body_yaw);
    const double vox_size = grid_tf->voxelSize()[0];

    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();

    for (int i = 0; i <= 10; ++i)
    {
        openvdb::Vec3d wp(
            start_world.x() + i * vox_size * cos_yaw,
            start_world.y() + i * vox_size * sin_yaw,
            start_world.z());
        openvdb::Coord ijk = openvdb::Coord::round(grid_tf->worldToIndex(wp));

        int val = 0;
        bool active = map_manager_->query_is_inflated_at_index(ijk, val, inf_acc);
        bool is_occ = active && (val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
        if (is_occ)
        {
            break;
        }
        escape_set_.insert(ijk);
    }
}

void FovAstar::clearEscapeCorridor()
{
    escape_set_.clear();
}

int FovAstar::labelIndex(ReachLabel label) const
{
    return (label == ReachLabel::CERTIFIED) ? 0 : 1;
}

ReachLabel FovAstar::advanceLabel(ReachLabel current_label,
                                  TraversalClass traversal) const
{
    if (current_label == ReachLabel::CONTAMINATED)
    {
        return ReachLabel::CONTAMINATED;
    }

    return (traversal == TraversalClass::STRICT_FREE)
               ? ReachLabel::CERTIFIED
               : ReachLabel::CONTAMINATED;
}

TraversalClass FovAstar::classifyTraversal(const openvdb::Coord &coord,
                                           int inflate_val,
                                           bool is_active) const
{
    if (!escape_set_.empty() && escape_set_.count(coord))
    {
        return TraversalClass::STRICT_FREE;
    }

    if (!is_active)
    {
        return TraversalClass::STRICT_FREE;
    }

    const bool has_occ = (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
    const bool has_frontier = inflate_val >= VDBMap::FRONTIER_INFLATION_DELTA;
    if (has_occ)
    {
        return TraversalClass::BLOCKED_OCC;
    }

    if (has_frontier)
    {
        return allow_soft_frontier_traversal_
                   ? TraversalClass::SOFT
                   : TraversalClass::BLOCKED_OCC;
    }

    return TraversalClass::STRICT_FREE;
}

void FovAstar::backtrackLabeled(const LabelSearchNode *end_node)
{
    path_nodes_.clear();
    const LabelSearchNode *cur = end_node;
    while (cur != nullptr)
    {
        path_nodes_.push_back(cur->ijk);
        cur = cur->parent;
    }
    std::reverse(path_nodes_.begin(), path_nodes_.end());
}

bool FovAstar::extractPathToGoal(const openvdb::Coord &goal_ijk, ReachLabel label)
{
    const auto &goal_map =
        (label == ReachLabel::CERTIFIED) ? certified_goal_nodes_ : contaminated_goal_nodes_;
    auto it = goal_map.find(goal_ijk);
    if (it == goal_map.end() || it->second == nullptr)
    {
        return false;
    }

    backtrackLabeled(it->second);
    last_path_label_ = label;
    return true;
}

bool FovAstar::extractPathToClosedState(const openvdb::Coord &ijk, ReachLabel label)
{
    auto it = state_table_.find(ijk);
    if (it == state_table_.end())
    {
        return false;
    }

    const int idx = labelIndex(label);
    const LabelSearchNode *node = it->second.closed_nodes[idx];
    if (node == nullptr)
    {
        return false;
    }

    backtrackLabeled(node);
    last_path_label_ = label;
    return true;
}

std::vector<SearchGoalCandidate> FovAstar::getClosedCandidates(ReachLabel label) const
{
    std::vector<SearchGoalCandidate> out;
    const int idx = labelIndex(label);
    out.reserve(state_table_.size());
    for (const auto &kv : state_table_)
    {
        const DualLabelRecord &rec = kv.second;
        if (rec.closed[idx] && rec.closed_nodes[idx] != nullptr &&
            std::isfinite(rec.best_g[idx]))
        {
            out.push_back(SearchGoalCandidate{kv.first, rec.best_g[idx]});
        }
    }
    return out;
}

int FovAstar::search(const openvdb::Coord &start_ijk,
                     const openvdb::Coord &end_ijk)
{
    Astar::reset();
    state_table_.clear();
    search_nodes_.clear();
    certified_goal_nodes_.clear();
    contaminated_goal_nodes_.clear();
    certified_goal_candidates_.clear();
    contaminated_goal_candidates_.clear();
    last_path_label_ = ReachLabel::CERTIFIED;
    last_goal_search_stop_reason_ = GoalSearchStopReason::NONE;
    resetPreviewProgressResults();

    if (!map_manager_)
    {
        return PATH_NOT_FOUND;
    }

    const auto node_cmp = [](const LabelSearchNode *lhs, const LabelSearchNode *rhs)
    {
        return lhs->f_score > rhs->f_score;
    };
    std::priority_queue<LabelSearchNode *,
                        std::vector<LabelSearchNode *>,
                        decltype(node_cmp)>
        open_set(node_cmp);

    auto make_node = [&](const openvdb::Coord &ijk,
                         ReachLabel label,
                         double g_score,
                         double f_score,
                         LabelSearchNode *parent) -> LabelSearchNode *
    {
        search_nodes_.push_back(std::make_unique<LabelSearchNode>());
        LabelSearchNode *node = search_nodes_.back().get();
        node->ijk = ijk;
        node->label = label;
        node->g_score = g_score;
        node->f_score = f_score;
        node->parent = parent;
        return node;
    };

    const ReachLabel start_label = ReachLabel::CERTIFIED;
    DualLabelRecord &start_rec = state_table_[start_ijk];
    LabelSearchNode *start_node = make_node(
        start_ijk,
        start_label,
        0.0,
        lambda_heu_ * computeHeuristic(start_ijk, end_ijk),
        nullptr);
    start_rec.open_nodes[labelIndex(start_label)] = start_node;
    start_rec.best_g[labelIndex(start_label)] = 0.0;
    open_set.push(start_node);

    LabelSearchNode *best_contaminated_goal = nullptr;
    LabelSearchNode *best_certified_goal = nullptr;
    bool certified_found = false;
    auto t0 = std::chrono::steady_clock::now();

    std::shared_lock<std::shared_mutex> map_lock(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();

    while (!open_set.empty())
    {
        LabelSearchNode *cur = open_set.top();
        open_set.pop();

        DualLabelRecord &cur_rec = state_table_[cur->ijk];
        const int cur_idx = labelIndex(cur->label);
        if (cur_rec.open_nodes[cur_idx] != cur)
        {
            continue;
        }
        cur_rec.open_nodes[cur_idx] = nullptr;

        if (cur_rec.closed[cur_idx])
        {
            continue;
        }
        cur_rec.closed[cur_idx] = true;
        cur_rec.closed_nodes[cur_idx] = cur;

        if (cur->label == ReachLabel::CERTIFIED)
        {
            updatePreviewProgressBuckets(cur, occ_acc);
        }

        if (isGoalWithAccessor(cur->ijk, target_world_, occ_acc))
        {
            const SearchGoalCandidate candidate{cur->ijk, cur->g_score};
            const bool candidate_accepted =
                !goal_candidate_predicate_ || goal_candidate_predicate_(candidate, cur->label);
            if (!candidate_accepted)
            {
                continue;
            }

            if (cur->label == ReachLabel::CERTIFIED)
            {
                if (!certified_found)
                {
                    certified_found = true;
                    best_certified_goal = cur;
                }

                if (certified_goal_nodes_.find(cur->ijk) == certified_goal_nodes_.end())
                {
                    certified_goal_nodes_[cur->ijk] = cur;
                    certified_goal_candidates_.push_back(candidate);
                }
                else if (cur->g_score < certified_goal_nodes_[cur->ijk]->g_score)
                {
                    certified_goal_nodes_[cur->ijk] = cur;
                    for (SearchGoalCandidate &candidate : certified_goal_candidates_)
                    {
                        if (candidate.coord == cur->ijk)
                        {
                            candidate.g_score = cur->g_score;
                            break;
                        }
                    }
                }
                continue;
            }

            if (best_contaminated_goal == nullptr ||
                cur->g_score < best_contaminated_goal->g_score)
            {
                best_contaminated_goal = cur;
            }
            if (contaminated_goal_nodes_.find(cur->ijk) == contaminated_goal_nodes_.end())
            {
                contaminated_goal_nodes_[cur->ijk] = cur;
                contaminated_goal_candidates_.push_back(candidate);
            }
            continue;
        }

        if (max_search_time_ > 0.0)
        {
            const auto dt = std::chrono::steady_clock::now() - t0;
            const double elapsed =
                std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
            if (elapsed > max_search_time_)
            {
                last_goal_search_stop_reason_ = GoalSearchStopReason::TIME_LIMIT;
                if (best_certified_goal != nullptr)
                {
                    backtrackLabeled(best_certified_goal);
                    last_path_label_ = ReachLabel::CERTIFIED;
                    return PATH_FOUND;
                }
                if (best_contaminated_goal != nullptr)
                {
                    backtrackLabeled(best_contaminated_goal);
                    last_path_label_ = ReachLabel::CONTAMINATED;
                    return PATH_FOUND;
                }
                return PATH_NOT_FOUND;
            }
        }

        // Soft standoff pre-scan (see setClearanceSoftCost).
        double clearance_mult = 1.0;
        if (clearance_soft_cost_mult_ > 1.0)
        {
            for (int dx = -1; dx <= 1 && clearance_mult == 1.0; ++dx)
            {
                for (int dy = -1; dy <= 1 && clearance_mult == 1.0; ++dy)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            continue;
                        }
                        int v = 0;
                        const bool a = map_manager_->query_is_inflated_at_index(
                            openvdb::Coord(cur->ijk.x() + dx, cur->ijk.y() + dy,
                                           cur->ijk.z() + dz),
                            v, inf_acc);
                        if (a && (v % VDBMap::FRONTIER_INFLATION_DELTA) > 0)
                        {
                            clearance_mult = clearance_soft_cost_mult_;
                            break;
                        }
                    }
                }
            }
        }

        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dz = -1; dz <= 1; ++dz)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                    {
                        continue;
                    }

                    const openvdb::Coord nbr_ijk(cur->ijk.x() + dx,
                                                 cur->ijk.y() + dy,
                                                 cur->ijk.z() + dz);

                    int inflate_val = 0;
                    const bool is_inflated =
                        map_manager_->query_is_inflated_at_index(nbr_ijk, inflate_val, inf_acc);
                    const TraversalClass traversal =
                        classifyTraversal(nbr_ijk, inflate_val, is_inflated);
                    if (traversal == TraversalClass::BLOCKED_OCC)
                    {
                        continue;
                    }

                    const ReachLabel next_label = advanceLabel(cur->label, traversal);
                    if (certified_found && next_label == ReachLabel::CONTAMINATED)
                    {
                        continue;
                    }
                    const int next_idx = labelIndex(next_label);
                    DualLabelRecord &nbr_rec = state_table_[nbr_ijk];

                    double step_cost = std::sqrt(double(dx * dx + dy * dy + dz * dz));
                    step_cost *= clearance_mult;
                    step_cost *= getEdgeCostMultiplier(cur->ijk, nbr_ijk, inflate_val, is_inflated);
                    if (next_label == ReachLabel::CONTAMINATED)
                    {
                        step_cost *= std::max(cfg_.soft_frontier_cost_multiplier, 1.0);
                    }

                    const double tentative_g = cur->g_score + step_cost;
                    if (next_label == ReachLabel::CONTAMINATED &&
                        tentative_g >= nbr_rec.best_g[labelIndex(ReachLabel::CERTIFIED)])
                    {
                        continue;
                    }
                    if (tentative_g >= nbr_rec.best_g[next_idx])
                    {
                        continue;
                    }
                    if (nbr_rec.closed[next_idx])
                    {
                        continue;
                    }

                    nbr_rec.best_g[next_idx] = tentative_g;
                    LabelSearchNode *nbr_node = nbr_rec.open_nodes[next_idx];
                    const double tentative_f =
                        tentative_g + lambda_heu_ * computeHeuristic(nbr_ijk, end_ijk);
                    if (nbr_node == nullptr)
                    {
                        nbr_node = make_node(nbr_ijk, next_label, tentative_g, tentative_f, cur);
                        nbr_rec.open_nodes[next_idx] = nbr_node;
                    }
                    else
                    {
                        nbr_node->g_score = tentative_g;
                        nbr_node->f_score = tentative_f;
                        nbr_node->parent = cur;
                    }
                    open_set.push(nbr_node);
                }
            }
        }
    }

    if (best_certified_goal != nullptr)
    {
        if (last_goal_search_stop_reason_ == GoalSearchStopReason::NONE)
        {
            last_goal_search_stop_reason_ = GoalSearchStopReason::EXHAUSTED;
        }
        backtrackLabeled(best_certified_goal);
        last_path_label_ = ReachLabel::CERTIFIED;
        return PATH_FOUND;
    }

    if (best_contaminated_goal != nullptr)
    {
        if (last_goal_search_stop_reason_ == GoalSearchStopReason::NONE)
        {
            last_goal_search_stop_reason_ = GoalSearchStopReason::EXHAUSTED;
        }
        backtrackLabeled(best_contaminated_goal);
        last_path_label_ = ReachLabel::CONTAMINATED;
        return PATH_FOUND;
    }

    if (last_goal_search_stop_reason_ == GoalSearchStopReason::NONE)
    {
        last_goal_search_stop_reason_ = GoalSearchStopReason::EXHAUSTED_NO_GOAL;
    }
    return PATH_NOT_FOUND;
}

void FovAstar::pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out)
{
    sparse_path_out.clear();
    if (path_nodes_.size() < 2)
    {
        return;
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    std::vector<openvdb::Coord> sparse;

    openvdb::Coord anchor = path_nodes_.front();
    sparse.push_back(anchor);

    for (size_t i = 1; i < path_nodes_.size(); ++i)
    {
        bool clear = map_manager_->ray_all_inflated_clear_index_banded(anchor, path_nodes_[i]);

        if (clear)
        {
            if (i == path_nodes_.size() - 1)
            {
                sparse.push_back(path_nodes_[i]);
            }
            continue;
        }

        anchor = path_nodes_[i - 1];
        sparse.push_back(anchor);

        if (i == path_nodes_.size() - 1)
        {
            sparse.push_back(path_nodes_[i]);
        }
    }

    for (auto &c : sparse)
    {
        sparse_path_out.push_back(tf->indexToWorld(c));
    }
}

// Core FOV evaluation: given a candidate body position (world frame),
// compute sensor pose (yaw aligned to target), transform target to sensor
// frame, and check the three safety constraints.  Returns per-constraint
// deficit distances (world metres) when violated.
FovAstar::FovCheckResult
FovAstar::evaluateFov(const openvdb::Vec3d &body_world) const
{
    return evaluateFov(body_world, target_world_);
}

FovAstar::FovCheckResult
FovAstar::evaluateFov(const openvdb::Vec3d &body_world,
                      const openvdb::Vec3d &target_world) const
{
    FovCheckResult result;

    const double dx = target_world.x() - body_world.x();
    const double dy = target_world.y() - body_world.y();
    const double d_xy_body = std::sqrt(dx * dx + dy * dy);

    double psi_body = 0.0;
    if (d_xy_body >= 1e-6)
    {
        psi_body = std::atan2(dy, dx) - psi_off_;
    }
    const double cos_psi = std::cos(psi_body);
    const double sin_psi = std::sin(psi_body);

    Eigen::Matrix3d R_yaw;
    R_yaw << cos_psi, -sin_psi, 0,
        sin_psi, cos_psi, 0,
        0, 0, 1;
    const Eigen::Matrix3d R_ws = R_yaw * R_bs_;
    const openvdb::Vec3d sensor_world =
        sensorWorldForTarget(body_world, target_world);

    // Target in sensor frame
    const Eigen::Vector3d d_w(target_world.x() - sensor_world.x(),
                              target_world.y() - sensor_world.y(),
                              target_world.z() - sensor_world.z());
    const Eigen::Vector3d d_s = R_ws.transpose() * d_w;

    // Frustum depth model: yaw alignment makes d_s.y ≈ 0, so depth ≈ d_s.x
    const double d_h = d_s.x();
    const double d_norm = d_s.norm();

    // Angular constraints with box-specific margins (Rz vertical, Rxy horizontal)
    const double theta_1 = std::atan2(d_s.z() + active_inflate_rz_, d_h - active_inflate_rxy_);
    const double theta_2 = std::atan2(d_s.z() - active_inflate_rz_, d_h - active_inflate_rxy_);

    const bool range_ok = (d_norm < active_safe_range_) && (active_safe_range_ > 0.0);
    const bool min_range_ok = d_norm > active_safe_min_range_;
    const bool theta1_ok = theta_1 < cfg_.theta_u;
    const bool theta2_ok = theta_2 > cfg_.theta_d;

    result.satisfied = range_ok && min_range_ok && theta1_ok && theta2_ok;

    if (!range_ok)
    {
        result.deficit_range = (active_safe_range_ > 0.0)
                                   ? (d_norm - active_safe_range_)
                                   : d_norm;
    }
    if (!min_range_ok)
    {
        result.deficit_min_range = active_safe_min_range_ - d_norm;
    }
    if (!theta1_ok)
    {
        result.deficit_theta1 = d_norm * std::sin(theta_1 - cfg_.theta_u);
    }
    if (!theta2_ok)
    {
        result.deficit_theta2 = d_norm * std::sin(cfg_.theta_d - theta_2);
    }

    return result;
}

openvdb::Vec3d FovAstar::sensorWorldForTarget(const openvdb::Vec3d &body_world,
                                              const openvdb::Vec3d &target_world) const
{
    const double dx = target_world.x() - body_world.x();
    const double dy = target_world.y() - body_world.y();
    const double d_xy_body = std::sqrt(dx * dx + dy * dy);

    double psi_body = 0.0;
    if (d_xy_body >= 1e-6)
    {
        psi_body = std::atan2(dy, dx) - psi_off_;
    }
    const double cos_psi = std::cos(psi_body);
    const double sin_psi = std::sin(psi_body);

    return openvdb::Vec3d(
        body_world.x() + t_bs_.x() * cos_psi - t_bs_.y() * sin_psi,
        body_world.y() + t_bs_.x() * sin_psi + t_bs_.y() * cos_psi,
        body_world.z() + t_bs_.z());
}

bool FovAstar::rayTraceClear(const openvdb::Vec3d &sensor_world,
                             const openvdb::Vec3d &target_world) const
{
    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();
    return rayTraceClearWithAccessor(sensor_world, target_world, occ_acc);
}

bool FovAstar::rayTraceClearWithAccessor(
    const openvdb::Vec3d &sensor_world,
    const openvdb::Vec3d &target_world,
    openvdb::FloatGrid::ConstAccessor &occ_acc) const
{
    // Exact Amanatides-Woo traversal (see ray_trace_util.h): the former
    // max-axis sampler could skip thin inclined slabs the segment crosses.
    auto grid_tf = map_manager_->get_grid_transform();
    return ffa::rayTraceVisitClear(
        *grid_tf, sensor_world, target_world,
        [&](const openvdb::Coord &vox) {
            float logodds = 0.0f;
            const bool active =
                map_manager_->query_log_odds_at_index(vox, logodds, occ_acc);
            return active && logodds > 0.0f;
        });
}

bool FovAstar::isGoalWithAccessor(
    const openvdb::Coord &current,
    const openvdb::Vec3d &target_world,
    openvdb::FloatGrid::ConstAccessor &occ_acc) const
{
    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(current);

    const FovCheckResult fov = evaluateFov(body_world, target_world);
    if (!fov.satisfied)
    {
        return false;
    }

    const openvdb::Vec3d sensor_world =
        sensorWorldForTarget(body_world, target_world);
    return rayTraceClearWithAccessor(sensor_world, target_world, occ_acc);
}

void FovAstar::updatePreviewProgressBuckets(
    const LabelSearchNode *node,
    openvdb::FloatGrid::ConstAccessor &occ_acc)
{
    if (node == nullptr || preview_progress_samples_.empty() ||
        preview_progress_samples_.size() != preview_sample_progress_.size() ||
        preview_progress_buckets_.size() != preview_progress_samples_.size())
    {
        return;
    }

    ++preview_progress_stats_.closed_certified;

    for (size_t rev = preview_progress_samples_.size(); rev > 0; --rev)
    {
        const size_t i = rev - 1;
        ++preview_progress_stats_.ray_checks;
        if (!isGoalWithAccessor(node->ijk, preview_progress_samples_[i], occ_acc))
        {
            continue;
        }

        preview_progress_buckets_[i].push_back(
            SearchGoalCandidate{node->ijk, node->g_score});
        if (preview_progress_stats_.best_sample_index < 0 ||
            static_cast<int>(i) > preview_progress_stats_.best_sample_index)
        {
            preview_progress_stats_.best_sample_index = static_cast<int>(i);
        }
        return;
    }
}

bool FovAstar::isGoal(const openvdb::Coord &current,
                      const openvdb::Coord & /*target*/) const
{
    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(current);
    const FovCheckResult fov = evaluateFov(body_world);
    if (!fov.satisfied)
    {
        return false;
    }
    return rayTraceClear(sensorWorldForTarget(body_world, target_world_),
                         target_world_);
}

double FovAstar::computeHeuristic(const openvdb::Coord &from,
                                  const openvdb::Coord & /*target*/) const
{
    if (!escape_set_.empty() && escape_set_.count(from))
    {
        return 0.0;
    }

    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(from);
    const FovCheckResult fov = evaluateFov(body_world);

    if (fov.satisfied)
    {
        return 0.0;
    }

    const double max_deficit =
        std::max({fov.deficit_range, fov.deficit_min_range,
                  fov.deficit_theta1, fov.deficit_theta2});
    const double voxel_size = map_manager_->get_grid_transform()->voxelSize()[0];

    return max_deficit / voxel_size;
}

bool FovAstar::isBlocked(const openvdb::Coord &coord,
                         int inflate_val, bool is_active) const
{
    return classifyTraversal(coord, inflate_val, is_active) == TraversalClass::BLOCKED_OCC;
}

double FovAstar::getTraversalCostMultiplier(const openvdb::Coord &coord,
                                            int inflate_val, bool is_active) const
{
    return 1.0;
}

double FovAstar::computeAnchorBiasMultiplier(const openvdb::Coord &current,
                                             const openvdb::Coord &next) const
{
    if (!has_search_anchor_ || anchor_bias_away_weight_ <= 0.0)
    {
        return 1.0;
    }
    if (!escape_set_.empty() && escape_set_.count(next))
    {
        return 1.0;
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Vec3d body_world = tf->indexToWorld(current);
    const openvdb::Vec3d next_world = tf->indexToWorld(next);

    Eigen::Vector3d toward_anchor(search_anchor_world_.x() - body_world.x(),
                                  search_anchor_world_.y() - body_world.y(),
                                  search_anchor_world_.z() - body_world.z());
    Eigen::Vector3d step_dir(next_world.x() - body_world.x(),
                             next_world.y() - body_world.y(),
                             next_world.z() - body_world.z());
    const double toward_norm = toward_anchor.norm();
    const double step_norm = step_dir.norm();
    if (toward_norm <= 1.0e-9 || step_norm <= 1.0e-9)
    {
        return 1.0;
    }

    toward_anchor /= toward_norm;
    step_dir /= step_norm;
    const double align = toward_anchor.dot(step_dir);
    const double away = std::max(0.0, -align);
    return std::clamp(1.0 + anchor_bias_away_weight_ * away,
                      1.0,
                      anchor_bias_max_multiplier_);
}

double FovAstar::getEdgeCostMultiplier(const openvdb::Coord &current,
                                       const openvdb::Coord &next,
                                       int inflate_val,
                                       bool is_active) const
{
    const double traversal_mult = getTraversalCostMultiplier(next, inflate_val, is_active);
    const double anchor_mult = computeAnchorBiasMultiplier(current, next);
    return traversal_mult * anchor_mult;
}

visualization_msgs::msg::Marker FovAstar::getDebugTreeMarker(const std::string &frame_id) const
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();

    visualization_msgs::msg::Marker tree_mk;
    tree_mk.header.frame_id = frame_id;
    tree_mk.ns = "astar_tree";
    tree_mk.id = 0;
    tree_mk.type = visualization_msgs::msg::Marker::LINE_LIST;
    tree_mk.action = visualization_msgs::msg::Marker::ADD;
    tree_mk.pose.orientation.w = 1.0;
    tree_mk.scale.x = 0.02;
    tree_mk.color.r = 0.0f;
    tree_mk.color.g = 1.0f;
    tree_mk.color.b = 1.0f;
    tree_mk.color.a = 0.3f;
    tree_mk.points.reserve(search_nodes_.size() * 2);

    for (const auto &node_ptr : search_nodes_)
    {
        const LabelSearchNode *node = node_ptr.get();
        if (node == nullptr || node->parent == nullptr)
        {
            continue;
        }

        const openvdb::Vec3d w_child = tf->indexToWorld(node->ijk);
        const openvdb::Vec3d w_parent = tf->indexToWorld(node->parent->ijk);

        geometry_msgs::msg::Point p1, p2;
        p1.x = w_child.x();
        p1.y = w_child.y();
        p1.z = w_child.z();
        p2.x = w_parent.x();
        p2.y = w_parent.y();
        p2.z = w_parent.z();
        tree_mk.points.push_back(p1);
        tree_mk.points.push_back(p2);
    }

    return tree_mk;
}
