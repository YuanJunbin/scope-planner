// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

// Grid A* over the VDB-EDT inflated occupancy map.
//
// The planner searches on integer voxel coordinates with a 26-connected
// neighbourhood, an octile-distance heuristic weighted by lambda_heu, and an
// optional wall-clock budget. Traversability, goal test, heuristic and edge
// costs are virtual hooks so the FOV / observe / connect searches can
// specialise them without touching the loop.
//
// Nodes live in a flat arena (std::vector) addressed by index; parent links
// and the open heap store indices, never pointers, so the arena may grow on
// demand without invalidating anything.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <openvdb/openvdb.h>
#include <visualization_msgs/msg/marker.hpp>

#include <vdb_edt/vdbmap.h>

// Search configuration. Field names double as the YAML parameter names in
// the planner configs (<prefix>_astar_allocate_num, ...).
struct AstarParams
{
    int allocate_num = 1000000;    // hard cap on nodes created per search
    double tie_breaker = 1.001;    // multiplier on the heuristic to prefer deeper nodes on ties
    double lambda_heu = 1.0;       // heuristic weight (1.0 = plain A*, >1 = weighted)
    double max_search_time = 0.05; // seconds; <= 0 means unlimited
};

// 64-bit mixing hash for voxel coordinates. Each component is multiplied by
// its own large odd constant and rotated to a different lane before the
// final avalanche, so neighbouring voxels land in unrelated buckets.
struct CoordHash
{
    std::size_t operator()(const openvdb::Coord &c) const noexcept
    {
        auto rotl = [](std::uint64_t v, unsigned r) noexcept
        {
            return (v << r) | (v >> (64u - r));
        };
        std::uint64_t h = static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.x())) * 0x9E3779B97F4A7C15ull;
        h ^= rotl(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.y())) * 0xC2B2AE3D27D4EB4Full, 23u);
        h ^= rotl(static_cast<std::uint64_t>(static_cast<std::uint32_t>(c.z())) * 0x165667B19E3779F9ull, 46u);
        h ^= h >> 32;
        h *= 0xD6E8FEB86659FD93ull;
        h ^= h >> 29;
        return static_cast<std::size_t>(h);
    }
};

class Astar
{
public:
    Astar();
    virtual ~Astar();

    // The arena and the heap comparator refer to this object's storage, so
    // copies would alias each other. The planner keeps one instance per role.
    Astar(const Astar &) = delete;
    Astar &operator=(const Astar &) = delete;

    enum
    {
        PATH_FOUND = 1,
        PATH_NOT_FOUND = 2
    };

    enum class SearchFailureReason
    {
        NONE = 0,
        MAP_UNAVAILABLE,
        UNINITIALIZED,
        TIME_LIMIT,
        NODE_POOL_EXHAUSTED,
        OPEN_SET_EMPTY
    };

    void initialize(std::shared_ptr<VDBMap> &map_manager,
                    const AstarParams &params = AstarParams{});
    void reset();

    // Voxels treated as blocked in addition to the map's occupancy inflation.
    void setExtraBlockedVoxels(const std::unordered_set<openvdb::Coord, CoordHash> &blocked);
    void clearExtraBlockedVoxels();

    // Runs a fresh search. Returns PATH_FOUND or PATH_NOT_FOUND; on failure
    // getLastFailureReasonString() says why.
    int search(const openvdb::Coord &start_pt, const openvdb::Coord &end_pt);

    std::vector<openvdb::Vec3d> getPathAstar();
    std::vector<openvdb::Coord> getVisited() const;
    double getEarlyTerminateCost() const;
    const char *getLastFailureReasonString() const;
    bool lastSearchTimedOut() const;

    // Public so the derived searches can read the configured weights.
    double lambda_heu_;
    double max_search_time_;

    // Greedy line-of-sight shortening of the last path (banded inflation
    // check between anchors, plus the extra blocked voxels).
    void pathShorten(std::vector<openvdb::Vec3d> &sparse_path_out);

    visualization_msgs::msg::Marker getDebugTreeMarker(const std::string &frame_id,
                                                       int max_edges = 0) const;

    using ProgressCallback =
        std::function<void(double elapsed_s, int used_nodes, size_t open_size)>;
    void setProgressCallback(ProgressCallback cb, double interval_s = 1.0);

    // Steps taken from a voxel adjacent to occupied inflation cost this much
    // more (>= 1.0; 1.0 disables).
    void setClearanceSoftCost(double mult) { clearance_soft_cost_mult_ = std::max(mult, 1.0); }

protected:
    virtual bool isGoal(const openvdb::Coord &current, const openvdb::Coord &target) const;
    virtual double computeHeuristic(const openvdb::Coord &from, const openvdb::Coord &target) const;
    virtual bool isBlocked(int inflate_val, bool is_active) const;
    virtual bool isBlocked(const openvdb::Coord &coord, int inflate_val, bool is_active) const;
    virtual double getTraversalCostMultiplier(const openvdb::Coord &coord,
                                              int inflate_val,
                                              bool is_active) const;
    virtual double getEdgeCostMultiplier(const openvdb::Coord &current,
                                         const openvdb::Coord &next,
                                         int inflate_val,
                                         bool is_active) const;

    std::shared_ptr<VDBMap> map_manager_;
    std::vector<openvdb::Coord> path_nodes_;
    ProgressCallback progress_cb_;
    double progress_interval_s_ = 1.0;
    double clearance_soft_cost_mult_ = 1.0;

private:
    struct GridNode
    {
        openvdb::Coord ijk;
        double g = 0.0;
        double f = 0.0;
        int parent = -1; // arena index, -1 for the root
    };

    // Orders arena indices by the node's CURRENT f value. A node whose cost
    // improves is pushed again; both heap entries then see the new f.
    struct HeapOrder
    {
        const std::vector<GridNode> *nodes = nullptr;
        bool operator()(int a, int b) const noexcept
        {
            return (*nodes)[a].f > (*nodes)[b].f;
        }
    };
    using OpenHeap = std::priority_queue<int, std::vector<int>, HeapOrder>;

    static constexpr int kNoNode = -1;

    // Appends a node to the arena; returns kNoNode when the cap is reached.
    int allocNode(const openvdb::Coord &ijk, double g, double f, int parent);
    // 3-D octile distance in voxel units, scaled by the tie-breaker.
    double octileDistance(const openvdb::Coord &a, const openvdb::Coord &b) const;
    void backtrack(int end_index);
    void logStartDiagnostic(const char *reason,
                            const openvdb::Coord &start_ijk,
                            const openvdb::Coord &end_ijk,
                            openvdb::Int32Grid::ConstAccessor &inf_acc) const;

    std::vector<GridNode> nodes_;
    OpenHeap open_heap_;
    std::unordered_map<openvdb::Coord, int, CoordHash> frontier_by_coord_; // open set: coord -> arena index
    std::unordered_set<openvdb::Coord, CoordHash> expanded_;               // closed set (also holds blocked voxels)
    std::unordered_set<openvdb::Coord, CoordHash> extra_blocked_voxels_;

    int node_cap_ = 0;
    int expansions_ = 0;
    double tie_break_ = 1.0;
    double cutoff_cost_estimate_ = 0.0;
    SearchFailureReason last_failure_reason_ = SearchFailureReason::NONE;
};
