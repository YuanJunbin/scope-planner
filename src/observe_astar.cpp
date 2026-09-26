// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/observe_astar.hpp"
#include <vdb_edt/ray_trace_util.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <queue>

void ObserveAstar::setFovConfig(const SensorFovConfig &cfg)
{
    cfg_ = cfg;

    t_bs_ = Eigen::Vector3d(cfg.body_x, cfg.body_y, cfg.body_z);

    R_bs_ = (Eigen::AngleAxisd(cfg.body_yaw, Eigen::Vector3d::UnitZ()) *
             Eigen::AngleAxisd(cfg.body_pitch, Eigen::Vector3d::UnitY()) *
             Eigen::AngleAxisd(cfg.body_roll, Eigen::Vector3d::UnitX()))
                .toRotationMatrix();

    psi_off_ = cfg.body_yaw;
    safe_range_ = cfg.range;
    min_range_ = std::max(cfg.observe_min_range, 0.0);
}

void ObserveAstar::setTargetUnknown(const openvdb::Coord &unknown_ijk)
{
    target_ijk_ = unknown_ijk;
    target_world_ = map_manager_->get_grid_transform()->indexToWorld(unknown_ijk);
}

void ObserveAstar::buildEscapeCorridor(const openvdb::Coord &start_ijk, double body_yaw)
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

void ObserveAstar::clearEscapeCorridor()
{
    escape_set_.clear();
}

int ObserveAstar::labelIndex(ReachLabel label) const
{
    return (label == ReachLabel::CERTIFIED) ? 0 : 1;
}

ReachLabel ObserveAstar::advanceLabel(ReachLabel current_label,
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

TraversalClass ObserveAstar::classifyTraversal(const openvdb::Coord &coord,
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

void ObserveAstar::backtrackLabeled(const LabelSearchNode *end_node)
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

bool ObserveAstar::extractPathToGoal(const openvdb::Coord &goal_ijk, ReachLabel label)
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

int ObserveAstar::search(const openvdb::Coord &start_ijk,
                         const openvdb::Coord & /*end_ijk*/)
{
    beginSearch(start_ijk);
    if (!map_manager_)
    {
        session_valid_ = false;
        return PATH_NOT_FOUND;
    }
    return runBudget(max_search_time_, /*is_resume=*/false);
}

int ObserveAstar::resume(double budget_s)
{
    if (!session_valid_ || !map_manager_)
    {
        return PATH_NOT_FOUND;
    }
    path_nodes_.clear();
    last_goal_search_stop_reason_ = GoalSearchStopReason::NONE;
    return runBudget(budget_s, /*is_resume=*/true);
}

void ObserveAstar::beginSearch(const openvdb::Coord &start_ijk)
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
    {
        std::priority_queue<LabelSearchNode *, std::vector<LabelSearchNode *>, LabelNodeCmp> empty;
        label_open_set_.swap(empty);
    }
    certified_found_ = false;
    best_certified_goal_ = nullptr;
    best_contaminated_goal_ = nullptr;
    session_valid_ = true;
    session_start_ = start_ijk;
    session_target_ = target_ijk_;

    search_nodes_.push_back(std::make_unique<LabelSearchNode>());
    LabelSearchNode *start_node = search_nodes_.back().get();
    start_node->ijk = start_ijk;
    start_node->label = ReachLabel::CERTIFIED;
    start_node->g_score = 0.0;
    start_node->f_score = lambda_heu_ * computeHeuristic(start_ijk, target_ijk_);
    start_node->parent = nullptr;

    DualLabelRecord &start_rec = state_table_[start_ijk];
    start_rec.open_nodes[labelIndex(ReachLabel::CERTIFIED)] = start_node;
    start_rec.best_g[labelIndex(ReachLabel::CERTIFIED)] = 0.0;
    label_open_set_.push(start_node);
}

void ObserveAstar::recordGoalCandidate(LabelSearchNode *cur, bool grazing)
{
    SearchGoalCandidate candidate{cur->ijk, cur->g_score};
    candidate.grazing = grazing;
    if (cur->label == ReachLabel::CERTIFIED)
    {
        if (!certified_found_)
        {
            certified_found_ = true;
        }
        if (best_certified_goal_ == nullptr)
        {
            best_certified_goal_ = cur;
        }

        auto it = certified_goal_nodes_.find(cur->ijk);
        if (it == certified_goal_nodes_.end())
        {
            certified_goal_nodes_[cur->ijk] = cur;
            certified_goal_candidates_.push_back(candidate);
        }
        else if (cur->g_score < it->second->g_score)
        {
            it->second = cur;
            for (SearchGoalCandidate &c : certified_goal_candidates_)
            {
                if (c.coord == cur->ijk)
                {
                    c.g_score = cur->g_score;
                    break;
                }
            }
        }
    }
    else
    {
        if (best_contaminated_goal_ == nullptr ||
            cur->g_score < best_contaminated_goal_->g_score)
        {
            best_contaminated_goal_ = cur;
        }
        if (contaminated_goal_nodes_.find(cur->ijk) == contaminated_goal_nodes_.end())
        {
            contaminated_goal_nodes_[cur->ijk] = cur;
            contaminated_goal_candidates_.push_back(candidate);
        }
    }
}

int ObserveAstar::runBudget(double budget_s, bool is_resume)
{
    auto t0 = std::chrono::steady_clock::now();

    std::shared_lock<std::shared_mutex> map_lock(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();
    std::unique_ptr<openvdb::Int32Grid::ConstAccessor> shell_acc;
    if (map_manager_->has_occ_shell_map())
    {
        shell_acc = std::make_unique<openvdb::Int32Grid::ConstAccessor>(
            map_manager_->get_occ_shell_accessor());
    }

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

    auto finish_with_best = [&](GoalSearchStopReason reason) -> int
    {
        last_goal_search_stop_reason_ = reason;
        if (best_certified_goal_ != nullptr)
        {
            backtrackLabeled(best_certified_goal_);
            last_path_label_ = ReachLabel::CERTIFIED;
            return PATH_FOUND;
        }
        if (best_contaminated_goal_ != nullptr)
        {
            backtrackLabeled(best_contaminated_goal_);
            last_path_label_ = ReachLabel::CONTAMINATED;
            return PATH_FOUND;
        }
        return PATH_NOT_FOUND;
    };

    while (!label_open_set_.empty())
    {
        LabelSearchNode *cur = label_open_set_.top();
        label_open_set_.pop();

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

        if (is_resume && cur->parent != nullptr)
        {
            // Map may have changed since this node was pushed: a voxel that
            // is now occ-blocked must not be expanded (or accepted as goal).
            int cur_inflate = 0;
            const bool cur_active =
                map_manager_->query_is_inflated_at_index(cur->ijk, cur_inflate, inf_acc);
            if (classifyTraversal(cur->ijk, cur_inflate, cur_active) == TraversalClass::BLOCKED_OCC)
            {
                continue;
            }
        }

        bool grazing = false;
        if (evaluateGoalWithAcc(cur->ijk, occ_acc, shell_acc.get(), grazing))
        {
            const SearchGoalCandidate probe{cur->ijk, cur->g_score};
            const bool candidate_accepted =
                !goal_candidate_predicate_ || goal_candidate_predicate_(probe, cur->label);
            if (candidate_accepted)
            {
                recordGoalCandidate(cur, grazing);
            }
            // Keep expanding from goal states so a start-near-goal observe
            // query still enumerates a useful viewpoint set within the budget.
        }

        if (budget_s > 0.0)
        {
            const auto dt = std::chrono::steady_clock::now() - t0;
            const double elapsed =
                std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
            if (elapsed > budget_s)
            {
                return finish_with_best(GoalSearchStopReason::TIME_LIMIT);
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
                    if (certified_found_ && next_label == ReachLabel::CONTAMINATED)
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
                        tentative_g + lambda_heu_ * computeHeuristic(nbr_ijk, target_ijk_);
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
                    label_open_set_.push(nbr_node);
                }
            }
        }
    }

    // Open set exhausted.
    const int rc = finish_with_best(GoalSearchStopReason::EXHAUSTED);
    if (rc == PATH_NOT_FOUND && !is_resume)
    {
        last_goal_search_stop_reason_ = GoalSearchStopReason::EXHAUSTED_NO_GOAL;
    }
    return rc;
}

void ObserveAstar::refreshBestGoalPointers()
{
    best_certified_goal_ = nullptr;
    best_contaminated_goal_ = nullptr;
    for (const SearchGoalCandidate &c : certified_goal_candidates_)
    {
        auto it = certified_goal_nodes_.find(c.coord);
        if (it != certified_goal_nodes_.end() && it->second != nullptr)
        {
            best_certified_goal_ = it->second;
            break;
        }
    }
    for (const SearchGoalCandidate &c : contaminated_goal_candidates_)
    {
        auto it = contaminated_goal_nodes_.find(c.coord);
        if (it == contaminated_goal_nodes_.end() || it->second == nullptr)
        {
            continue;
        }
        if (best_contaminated_goal_ == nullptr ||
            it->second->g_score < best_contaminated_goal_->g_score)
        {
            best_contaminated_goal_ = it->second;
        }
    }
}

size_t ObserveAstar::revalidateCandidates(
    const std::unordered_set<openvdb::Coord, CoordHash> &failed)
{
    if (!session_valid_ || !map_manager_)
    {
        return 0;
    }

    std::shared_lock<std::shared_mutex> map_lock(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();
    std::unique_ptr<openvdb::Int32Grid::ConstAccessor> shell_acc;
    if (map_manager_->has_occ_shell_map())
    {
        shell_acc = std::make_unique<openvdb::Int32Grid::ConstAccessor>(
            map_manager_->get_occ_shell_accessor());
    }
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();

    size_t removed = 0;
    auto filter = [&](std::vector<SearchGoalCandidate> &list,
                      std::unordered_map<openvdb::Coord, LabelSearchNode *, CoordHash> &nodes)
    {
        std::vector<SearchGoalCandidate> kept;
        kept.reserve(list.size());
        for (SearchGoalCandidate &c : list)
        {
            bool drop = failed.count(c.coord) > 0;
            if (!drop)
            {
                int inflate_val = 0;
                const bool active =
                    map_manager_->query_is_inflated_at_index(c.coord, inflate_val, inf_acc);
                drop = active && (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
            }
            if (!drop)
            {
                const openvdb::Vec3d body_world = tf->indexToWorld(c.coord);
                const double dx = target_world_.x() - body_world.x();
                const double dy = target_world_.y() - body_world.y();
                const double psi = (dx * dx + dy * dy >= 1e-12) ? (std::atan2(dy, dx) - psi_off_) : 0.0;
                bool grazing = false;
                drop = !rayTraceClearWithAcc(sensorWorldForBody(body_world, psi), target_world_,
                                             occ_acc, shell_acc.get(), grazing);
                c.grazing = grazing;
            }
            if (drop)
            {
                nodes.erase(c.coord);
                ++removed;
            }
            else
            {
                kept.push_back(c);
            }
        }
        list.swap(kept);
    };

    filter(certified_goal_candidates_, certified_goal_nodes_);
    filter(contaminated_goal_candidates_, contaminated_goal_nodes_);
    refreshBestGoalPointers();
    return removed;
}

void ObserveAstar::pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out)
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

// FOV check for a single unknown point (no inflation box shift).
// Computes sensor position at candidate body position (yaw aligned to target),
// transforms target to sensor frame, checks range + vertical angle.
ObserveAstar::ObserveFovResult
ObserveAstar::evaluateObserveFov(const openvdb::Vec3d &body_world) const
{
    ObserveFovResult result;

    const double dx = target_world_.x() - body_world.x();
    const double dy = target_world_.y() - body_world.y();
    const double d_xy_body = std::sqrt(dx * dx + dy * dy);

    double psi_body = 0.0;
    if (d_xy_body >= 1e-6)
    {
        psi_body = std::atan2(dy, dx) - psi_off_;
    }
    const double cos_psi = std::cos(psi_body);
    const double sin_psi = std::sin(psi_body);

    const Eigen::Vector3d xs(
        body_world.x() + t_bs_.x() * cos_psi - t_bs_.y() * sin_psi,
        body_world.y() + t_bs_.x() * sin_psi + t_bs_.y() * cos_psi,
        body_world.z() + t_bs_.z());

    Eigen::Matrix3d R_yaw;
    R_yaw << cos_psi, -sin_psi, 0,
        sin_psi, cos_psi, 0,
        0, 0, 1;
    const Eigen::Matrix3d R_ws = R_yaw * R_bs_;

    const Eigen::Vector3d d_w(target_world_.x() - xs.x(),
                              target_world_.y() - xs.y(),
                              target_world_.z() - xs.z());
    const Eigen::Vector3d d_s = R_ws.transpose() * d_w;

    const double d_h = d_s.x();
    const double d_norm = d_s.norm();

    // No inflation box shift — just the point itself
    const double theta = std::atan2(d_s.z(), d_h);

    const bool range_ok = (d_norm < safe_range_);
    const bool min_range_ok = (d_norm > min_range_);
    const bool theta_u_ok = theta < cfg_.theta_u;
    const bool theta_d_ok = theta > cfg_.theta_d;

    result.satisfied = range_ok && min_range_ok && theta_u_ok && theta_d_ok;

    if (!range_ok)
    {
        result.deficit_range = (safe_range_ > 0.0)
                                   ? (d_norm - safe_range_)
                                   : d_norm;
    }
    if (!min_range_ok)
    {
        result.deficit_min_range = min_range_ - d_norm;
    }
    if (!theta_u_ok)
    {
        result.deficit_theta = d_norm * std::sin(theta - cfg_.theta_u);
    }
    else if (!theta_d_ok)
    {
        result.deficit_theta = d_norm * std::sin(cfg_.theta_d - theta);
    }

    if (result.satisfied)
    {
        target_yaw_ = psi_body;
    }

    return result;
}

// Exact ray trace on logocc grid (Amanatides-Woo via ray_trace_util.h):
// returns true if no occupied voxel between sensor and target (unknowns are
// transparent). The former max-axis sampler could skip thin inclined slabs
// the segment actually crosses and certify physically blind viewpoints.
bool ObserveAstar::rayTraceClear(const openvdb::Vec3d &sensor_world,
                                 const openvdb::Vec3d &target_world) const
{
    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();
    bool grazing = false;
    return rayTraceClearWithAcc(sensor_world, target_world, occ_acc, nullptr, grazing);
}

bool ObserveAstar::rayTraceClearWithAcc(const openvdb::Vec3d &sensor_world,
                                        const openvdb::Vec3d &target_world,
                                        openvdb::FloatGrid::ConstAccessor &occ_acc,
                                        openvdb::Int32Grid::ConstAccessor *shell_acc,
                                        bool &grazing_out) const
{
    grazing_out = false;
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const openvdb::Coord target_ijk = openvdb::Coord::round(tf->worldToIndex(target_world));
    const int skip = occ_shell_skip_vox_;

    return ffa::rayTraceVisitClear(
        *tf, sensor_world, target_world,
        [&](const openvdb::Coord &vox) {
            float logodds = 0.0f;
            const bool active =
                map_manager_->query_log_odds_at_index(vox, logodds, occ_acc);
            if (active && logodds > 0.0f)
            {
                return true; // blocked
            }
            if (shell_acc != nullptr && !grazing_out)
            {
                const int cheb = std::max({std::abs(vox.x() - target_ijk.x()),
                                           std::abs(vox.y() - target_ijk.y()),
                                           std::abs(vox.z() - target_ijk.z())});
                if (cheb > skip && map_manager_->query_is_occ_shell_at_index(vox, *shell_acc))
                {
                    grazing_out = true;
                }
            }
            return false;
        });
}

openvdb::Vec3d ObserveAstar::sensorWorldForBody(const openvdb::Vec3d &body_world, double psi) const
{
    const double cos_psi = std::cos(psi);
    const double sin_psi = std::sin(psi);
    return openvdb::Vec3d(
        body_world.x() + t_bs_.x() * cos_psi - t_bs_.y() * sin_psi,
        body_world.y() + t_bs_.x() * sin_psi + t_bs_.y() * cos_psi,
        body_world.z() + t_bs_.z());
}

bool ObserveAstar::evaluateGoalWithAcc(const openvdb::Coord &current,
                                       openvdb::FloatGrid::ConstAccessor &occ_acc,
                                       openvdb::Int32Grid::ConstAccessor *shell_acc,
                                       bool &grazing_out) const
{
    grazing_out = false;
    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(current);

    ObserveFovResult fov = evaluateObserveFov(body_world);
    if (!fov.satisfied)
    {
        return false;
    }

    // FOV geometry satisfied — verify line of sight via ray trace
    return rayTraceClearWithAcc(sensorWorldForBody(body_world, target_yaw_), target_world_,
                                occ_acc, shell_acc, grazing_out);
}

bool ObserveAstar::isGoal(const openvdb::Coord &current,
                          const openvdb::Coord & /*target*/) const
{
    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(current);

    ObserveFovResult fov = evaluateObserveFov(body_world);
    if (!fov.satisfied)
    {
        return false;
    }

    return rayTraceClear(sensorWorldForBody(body_world, target_yaw_), target_world_);
}

double ObserveAstar::computeHeuristic(const openvdb::Coord &from,
                                      const openvdb::Coord & /*target*/) const
{
    if (!escape_set_.empty() && escape_set_.count(from))
    {
        return 0.0;
    }

    const openvdb::Vec3d body_world =
        map_manager_->get_grid_transform()->indexToWorld(from);
    const ObserveFovResult fov = evaluateObserveFov(body_world);

    if (fov.satisfied)
    {
        return 0.0;
    }

    const double max_deficit = std::max({fov.deficit_range,
                                         fov.deficit_min_range,
                                         fov.deficit_theta});
    const double voxel_size = map_manager_->get_grid_transform()->voxelSize()[0];

    return max_deficit / voxel_size;
}

bool ObserveAstar::isBlocked(const openvdb::Coord &coord,
                             int inflate_val, bool is_active) const
{
    return classifyTraversal(coord, inflate_val, is_active) == TraversalClass::BLOCKED_OCC;
}

double ObserveAstar::getTraversalCostMultiplier(const openvdb::Coord &coord,
                                                int inflate_val, bool is_active) const
{
    return 1.0;
}

double ObserveAstar::computeAnchorBiasMultiplier(const openvdb::Coord &current,
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

double ObserveAstar::getEdgeCostMultiplier(const openvdb::Coord &current,
                                           const openvdb::Coord &next,
                                           int inflate_val,
                                           bool is_active) const
{
    const double traversal_mult = getTraversalCostMultiplier(next, inflate_val, is_active);
    const double anchor_mult = computeAnchorBiasMultiplier(current, next);
    return traversal_mult * anchor_mult;
}

visualization_msgs::msg::Marker ObserveAstar::getDebugTreeMarker(const std::string &frame_id) const
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
