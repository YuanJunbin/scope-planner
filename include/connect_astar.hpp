// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include "astar_vdb.hpp"
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

// A* search that connects the robot to a selected observation viewpoint
// through voxels that are entirely free of both occupied and frontier inflation.
class ConnectAstar : public Astar
{
public:
    // Goal-set entry with an initial (seed) cost. The reverse multi-source
    // search seeds each goal at g = init_cost, so the returned goal is
    // argmin over goals of (init_cost + certified path cost). init_cost = 0
    // reproduces plain nearest-goal selection.
    struct GoalSeed
    {
        openvdb::Coord coord{0, 0, 0};
        double init_cost = 0.0;
    };

    int searchToGoalSet(const openvdb::Coord &start_pt,
                        const std::vector<openvdb::Coord> &goal_set);
    int searchToGoalSet(const openvdb::Coord &start_pt,
                        const std::vector<GoalSeed> &goal_seeds);
    void pathShortenStrict(std::vector<openvdb::Vec3d> &sparse_path_out);

protected:
    bool isBlocked(const openvdb::Coord &coord, int inflate_val, bool is_active) const override;

private:
    struct SearchNode
    {
        openvdb::Coord ijk;
        double f_score = 0.0;
        double g_score = 0.0;
        SearchNode *parent = nullptr;
    };

    struct SearchRecord
    {
        SearchNode *open_node = nullptr;
        bool closed = false;
        double best_g = std::numeric_limits<double>::infinity();
    };

    void backtrackToGoalSet(const SearchNode *end_node);

    std::unordered_map<openvdb::Coord, SearchRecord, CoordHash> state_table_;
    std::vector<std::unique_ptr<SearchNode>> search_nodes_;
};
