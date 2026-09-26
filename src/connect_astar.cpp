// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/connect_astar.hpp"
#include <cmath>

#include <chrono>
#include <queue>

void ConnectAstar::backtrackToGoalSet(const SearchNode *end_node)
{
    path_nodes_.clear();
    const SearchNode *cur = end_node;
    while (cur != nullptr)
    {
        path_nodes_.push_back(cur->ijk);
        cur = cur->parent;
    }
}

int ConnectAstar::searchToGoalSet(const openvdb::Coord &start_ijk,
                                  const std::vector<openvdb::Coord> &goal_set)
{
    std::vector<GoalSeed> seeds;
    seeds.reserve(goal_set.size());
    for (const openvdb::Coord &c : goal_set)
    {
        seeds.push_back(GoalSeed{c, 0.0});
    }
    return searchToGoalSet(start_ijk, seeds);
}

int ConnectAstar::searchToGoalSet(const openvdb::Coord &start_ijk,
                                  const std::vector<GoalSeed> &goal_set)
{
    reset();
    state_table_.clear();
    search_nodes_.clear();

    if (!map_manager_ || goal_set.empty())
    {
        return PATH_NOT_FOUND;
    }

    const auto node_cmp = [](const SearchNode *lhs, const SearchNode *rhs)
    {
        return lhs->f_score > rhs->f_score;
    };
    std::priority_queue<SearchNode *,
                        std::vector<SearchNode *>,
                        decltype(node_cmp)>
        open_set(node_cmp);

    auto make_node = [&](const openvdb::Coord &ijk,
                         double g_score,
                         double f_score,
                         SearchNode *parent) -> SearchNode *
    {
        search_nodes_.push_back(std::make_unique<SearchNode>());
        SearchNode *node = search_nodes_.back().get();
        node->ijk = ijk;
        node->g_score = g_score;
        node->f_score = f_score;
        node->parent = parent;
        return node;
    };

    std::shared_lock<std::shared_mutex> map_lock(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();

    for (const GoalSeed &seed : goal_set)
    {
        const openvdb::Coord &goal_ijk = seed.coord;
        const double g0 = std::max(0.0, seed.init_cost);
        if (!std::isfinite(g0))
        {
            continue; // infinite seed cost == excluded goal
        }
        int inflate_val = 0;
        const bool is_inflated =
            map_manager_->query_is_inflated_at_index(goal_ijk, inflate_val, inf_acc);
        if (isBlocked(goal_ijk, inflate_val, is_inflated))
        {
            continue;
        }

        SearchRecord &rec = state_table_[goal_ijk];
        if (g0 >= rec.best_g)
        {
            continue;
        }

        const double f_score = g0 + lambda_heu_ * computeHeuristic(goal_ijk, start_ijk);
        SearchNode *node = rec.open_node;
        if (node == nullptr)
        {
            node = make_node(goal_ijk, g0, f_score, nullptr);
            rec.open_node = node;
        }
        else
        {
            node->g_score = g0;
            node->f_score = f_score;
            node->parent = nullptr;
        }
        rec.best_g = g0;
        open_set.push(node);
    }

    if (open_set.empty())
    {
        return PATH_NOT_FOUND;
    }

    const auto t0 = std::chrono::steady_clock::now();

    while (!open_set.empty())
    {
        SearchNode *cur = open_set.top();
        open_set.pop();

        SearchRecord &cur_rec = state_table_[cur->ijk];
        if (cur_rec.open_node != cur)
        {
            continue;
        }
        cur_rec.open_node = nullptr;

        if (cur_rec.closed)
        {
            continue;
        }
        cur_rec.closed = true;

        if (cur->ijk == start_ijk)
        {
            backtrackToGoalSet(cur);
            return PATH_FOUND;
        }

        if (max_search_time_ > 0.0)
        {
            const auto dt = std::chrono::steady_clock::now() - t0;
            const double elapsed =
                std::chrono::duration_cast<std::chrono::duration<double>>(dt).count();
            if (elapsed > max_search_time_)
            {
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
                    if (isBlocked(nbr_ijk, inflate_val, is_inflated))
                    {
                        continue;
                    }

                    SearchRecord &nbr_rec = state_table_[nbr_ijk];
                    if (nbr_rec.closed)
                    {
                        continue;
                    }

                    double step_cost = std::sqrt(double(dx * dx + dy * dy + dz * dz));
                    step_cost *= clearance_mult;
                    step_cost *= getEdgeCostMultiplier(cur->ijk, nbr_ijk, inflate_val, is_inflated);
                    const double tentative_g = cur->g_score + step_cost;
                    if (tentative_g >= nbr_rec.best_g)
                    {
                        continue;
                    }

                    nbr_rec.best_g = tentative_g;
                    const double tentative_f =
                        tentative_g + lambda_heu_ * computeHeuristic(nbr_ijk, start_ijk);
                    SearchNode *nbr_node = nbr_rec.open_node;
                    if (nbr_node == nullptr)
                    {
                        nbr_node = make_node(nbr_ijk, tentative_g, tentative_f, cur);
                        nbr_rec.open_node = nbr_node;
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

    return PATH_NOT_FOUND;
}

void ConnectAstar::pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out)
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

bool ConnectAstar::isBlocked(const openvdb::Coord & /*coord*/,
                             int inflate_val, bool is_active) const
{
    return is_active && inflate_val > 0;
}
