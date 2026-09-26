// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/astar_vdb.hpp"

#include <algorithm>
#include <iostream>
#include <shared_mutex>

namespace
{

const char *failureReasonName(Astar::SearchFailureReason reason)
{
    switch (reason)
    {
    case Astar::SearchFailureReason::NONE:
        return "none";
    case Astar::SearchFailureReason::MAP_UNAVAILABLE:
        return "map_unavailable";
    case Astar::SearchFailureReason::UNINITIALIZED:
        return "uninitialized";
    case Astar::SearchFailureReason::TIME_LIMIT:
        return "time_limit";
    case Astar::SearchFailureReason::NODE_POOL_EXHAUSTED:
        return "node_pool_exhausted";
    case Astar::SearchFailureReason::OPEN_SET_EMPTY:
        return "open_set_empty";
    }
    return "unknown";
}

// Occupancy channel of the packed inflation counter.
inline bool occupancyInflated(int inflate_val)
{
    return (inflate_val % VDBMap::FRONTIER_INFLATION_DELTA) > 0;
}

// Initial arena reservation: enough for typical local searches without
// committing the full cap up front (the global search cap is millions).
constexpr int kInitialArenaReserve = 1 << 16;

} // namespace

Astar::Astar()
    : lambda_heu_(1.0),
      max_search_time_(0.0),
      open_heap_(HeapOrder{&nodes_})
{
}

Astar::~Astar() = default;

void Astar::initialize(std::shared_ptr<VDBMap> &map_manager,
                       const AstarParams &params)
{
    map_manager_ = map_manager;

    node_cap_ = params.allocate_num;
    tie_break_ = params.tie_breaker;
    lambda_heu_ = params.lambda_heu;
    max_search_time_ = params.max_search_time;

    nodes_.reserve(static_cast<size_t>(std::max(0, std::min(node_cap_, kInitialArenaReserve))));
    reset();
}

void Astar::reset()
{
    nodes_.clear();
    frontier_by_coord_.clear();
    expanded_.clear();
    path_nodes_.clear();
    open_heap_ = OpenHeap(HeapOrder{&nodes_});

    expansions_ = 0;
    cutoff_cost_estimate_ = 0.0;
    last_failure_reason_ = SearchFailureReason::NONE;
}

void Astar::setExtraBlockedVoxels(const std::unordered_set<openvdb::Coord, CoordHash> &blocked)
{
    extra_blocked_voxels_ = blocked;
}

void Astar::clearExtraBlockedVoxels()
{
    extra_blocked_voxels_.clear();
}

int Astar::allocNode(const openvdb::Coord &ijk, double g, double f, int parent)
{
    if (static_cast<int>(nodes_.size()) >= node_cap_)
    {
        return kNoNode;
    }
    GridNode node;
    node.ijk = ijk;
    node.g = g;
    node.f = f;
    node.parent = parent;
    nodes_.push_back(node);
    return static_cast<int>(nodes_.size()) - 1;
}

std::vector<openvdb::Vec3d> Astar::getPathAstar()
{
    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    std::vector<openvdb::Vec3d> world_path;
    world_path.reserve(path_nodes_.size());
    for (const openvdb::Coord &c : path_nodes_)
    {
        world_path.push_back(tf->indexToWorld(c));
    }
    return world_path;
}

std::vector<openvdb::Coord> Astar::getVisited() const
{
    std::vector<openvdb::Coord> visited;
    visited.reserve(nodes_.size());
    for (const GridNode &node : nodes_)
    {
        visited.push_back(node.ijk);
    }
    return visited;
}

double Astar::getEarlyTerminateCost() const
{
    return cutoff_cost_estimate_;
}

const char *Astar::getLastFailureReasonString() const
{
    return failureReasonName(last_failure_reason_);
}

bool Astar::lastSearchTimedOut() const
{
    return last_failure_reason_ == SearchFailureReason::TIME_LIMIT;
}

double Astar::octileDistance(const openvdb::Coord &a, const openvdb::Coord &b) const
{
    // Sort the absolute axis deltas: d0 <= d1 <= d2. The cheapest 26-connected
    // walk takes d0 full-diagonal steps, then (d1 - d0) planar-diagonal steps,
    // then (d2 - d1) axis steps.
    double d0 = std::abs(static_cast<double>(a.x() - b.x()));
    double d1 = std::abs(static_cast<double>(a.y() - b.y()));
    double d2 = std::abs(static_cast<double>(a.z() - b.z()));
    if (d0 > d1)
    {
        std::swap(d0, d1);
    }
    if (d1 > d2)
    {
        std::swap(d1, d2);
    }
    if (d0 > d1)
    {
        std::swap(d0, d1);
    }
    const double h = std::sqrt(3.0) * d0 + std::sqrt(2.0) * (d1 - d0) + (d2 - d1);
    return tie_break_ * h;
}

bool Astar::isGoal(const openvdb::Coord &current,
                   const openvdb::Coord &target) const
{
    return current == target;
}

double Astar::computeHeuristic(const openvdb::Coord &from,
                                const openvdb::Coord &target) const
{
    return octileDistance(from, target);
}

bool Astar::isBlocked(int inflate_val, bool is_active) const
{
    return is_active && occupancyInflated(inflate_val);
}

bool Astar::isBlocked(const openvdb::Coord &coord, int inflate_val, bool is_active) const
{
    if (extra_blocked_voxels_.count(coord) != 0)
    {
        return true;
    }
    return isBlocked(inflate_val, is_active);
}

double Astar::getTraversalCostMultiplier(const openvdb::Coord & /*coord*/,
                                         int /*inflate_val*/, bool /*is_active*/) const
{
    return 1.0;
}

double Astar::getEdgeCostMultiplier(const openvdb::Coord & /*current*/,
                                    const openvdb::Coord &next,
                                    int inflate_val,
                                    bool is_active) const
{
    return getTraversalCostMultiplier(next, inflate_val, is_active);
}

void Astar::logStartDiagnostic(const char *reason,
                               const openvdb::Coord &start_ijk,
                               const openvdb::Coord &end_ijk,
                               openvdb::Int32Grid::ConstAccessor &inf_acc) const
{
    int start_inflate_val = 0;
    const bool start_active =
        map_manager_->query_is_inflated_at_index(start_ijk, start_inflate_val, inf_acc);

    int free_neighbors = 0;
    int blocked_neighbors = 0;
    int occ_blocked_neighbors = 0;
    int frontier_blocked_neighbors = 0;
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
                const openvdb::Coord nbr = start_ijk.offsetBy(dx, dy, dz);
                int inflate_val = 0;
                const bool is_active = map_manager_->query_is_inflated_at_index(nbr, inflate_val, inf_acc);
                if (!isBlocked(nbr, inflate_val, is_active))
                {
                    ++free_neighbors;
                    continue;
                }
                ++blocked_neighbors;
                if (is_active)
                {
                    if (occupancyInflated(inflate_val))
                    {
                        ++occ_blocked_neighbors;
                    }
                    else if (inflate_val >= VDBMap::FRONTIER_INFLATION_DELTA)
                    {
                        ++frontier_blocked_neighbors;
                    }
                }
            }
        }
    }

    std::cerr << "[AstarVDB] " << reason
              << " start=[" << start_ijk.x() << "," << start_ijk.y() << "," << start_ijk.z() << "]"
              << " goal=[" << end_ijk.x() << "," << end_ijk.y() << "," << end_ijk.z() << "]"
              << " start_inflated=" << (start_active ? 1 : 0)
              << " start_inflate_val=" << start_inflate_val
              << " free_neighbors=" << free_neighbors
              << " blocked_neighbors=" << blocked_neighbors
              << " occ_blocked_neighbors=" << occ_blocked_neighbors
              << " frontier_blocked_neighbors=" << frontier_blocked_neighbors
              << std::endl;
}

int Astar::search(const openvdb::Coord &start_ijk,
                  const openvdb::Coord &end_ijk)
{
    using Clock = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;

    reset();

    if (!map_manager_)
    {
        std::cerr << "[AstarVDB] no map attached; call initialize() first." << std::endl;
        last_failure_reason_ = SearchFailureReason::MAP_UNAVAILABLE;
        return PATH_NOT_FOUND;
    }
    if (node_cap_ <= 0)
    {
        std::cerr << "[AstarVDB] node cap is " << node_cap_ << "; call initialize() first." << std::endl;
        last_failure_reason_ = SearchFailureReason::UNINITIALIZED;
        return PATH_NOT_FOUND;
    }

    const int root = allocNode(start_ijk, 0.0,
                               lambda_heu_ * computeHeuristic(start_ijk, end_ijk),
                               kNoNode);
    open_heap_.push(root);
    frontier_by_coord_.emplace(start_ijk, root);

    std::shared_lock<std::shared_mutex> map_lock(map_manager_->get_map_mutex());
    openvdb::Int32Grid::ConstAccessor inf_acc = map_manager_->get_inflated_accessor();

    const auto t0 = Clock::now();
    const auto progress_period =
        std::chrono::duration_cast<Clock::duration>(Seconds(progress_interval_s_));
    auto next_progress = t0 + progress_period;
    int progress_countdown = 4096;

    while (!open_heap_.empty())
    {
        const int cur = open_heap_.top();

        if (progress_cb_ && --progress_countdown <= 0)
        {
            progress_countdown = 4096;
            const auto now = Clock::now();
            if (now >= next_progress)
            {
                next_progress = now + progress_period;
                progress_cb_(Seconds(now - t0).count(),
                             static_cast<int>(nodes_.size()),
                             open_heap_.size());
            }
        }

        if (isGoal(nodes_[cur].ijk, end_ijk))
        {
            backtrack(cur);
            return PATH_FOUND;
        }

        if (max_search_time_ > 0.0)
        {
            const double elapsed = Seconds(Clock::now() - t0).count();
            if (elapsed > max_search_time_)
            {
                cutoff_cost_estimate_ =
                    nodes_[cur].g + lambda_heu_ * computeHeuristic(nodes_[cur].ijk, end_ijk);
                last_failure_reason_ = SearchFailureReason::TIME_LIMIT;
                return PATH_NOT_FOUND;
            }
        }

        open_heap_.pop();
        const openvdb::Coord cur_ijk = nodes_[cur].ijk;
        const double cur_g = nodes_[cur].g;
        frontier_by_coord_.erase(cur_ijk);
        expanded_.insert(cur_ijk);
        ++expansions_;

        // Soft clearance penalty: any occupancy-inflated neighbour makes every
        // step out of this voxel more expensive.
        double clearance_mult = 1.0;
        if (clearance_soft_cost_mult_ > 1.0)
        {
            bool near_occupancy = false;
            for (int dx = -1; dx <= 1 && !near_occupancy; ++dx)
            {
                for (int dy = -1; dy <= 1 && !near_occupancy; ++dy)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        if (dx == 0 && dy == 0 && dz == 0)
                        {
                            continue;
                        }
                        int v = 0;
                        const bool a = map_manager_->query_is_inflated_at_index(
                            cur_ijk.offsetBy(dx, dy, dz), v, inf_acc);
                        if (a && occupancyInflated(v))
                        {
                            near_occupancy = true;
                            break;
                        }
                    }
                }
            }
            if (near_occupancy)
            {
                clearance_mult = clearance_soft_cost_mult_;
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
                    const openvdb::Coord nbr_ijk = cur_ijk.offsetBy(dx, dy, dz);
                    if (expanded_.count(nbr_ijk) != 0)
                    {
                        continue;
                    }

                    int inflate_val = 0;
                    const bool is_active =
                        map_manager_->query_is_inflated_at_index(nbr_ijk, inflate_val, inf_acc);
                    if (isBlocked(nbr_ijk, inflate_val, is_active))
                    {
                        // Remember blocked voxels so they are queried once.
                        expanded_.insert(nbr_ijk);
                        continue;
                    }

                    double step_cost = std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
                    step_cost *= clearance_mult;
                    step_cost *= getEdgeCostMultiplier(cur_ijk, nbr_ijk, inflate_val, is_active);

                    const double tentative_g = cur_g + step_cost;
                    const double tentative_f =
                        tentative_g + lambda_heu_ * computeHeuristic(nbr_ijk, end_ijk);

                    const auto seen = frontier_by_coord_.find(nbr_ijk);
                    if (seen == frontier_by_coord_.end())
                    {
                        const int idx = allocNode(nbr_ijk, tentative_g, tentative_f, cur);
                        if (idx == kNoNode)
                        {
                            std::cerr << "[AstarVDB] node cap " << node_cap_
                                      << " reached, giving up." << std::endl;
                            last_failure_reason_ = SearchFailureReason::NODE_POOL_EXHAUSTED;
                            return PATH_NOT_FOUND;
                        }
                        open_heap_.push(idx);
                        frontier_by_coord_.emplace(nbr_ijk, idx);
                    }
                    else
                    {
                        GridNode &known = nodes_[seen->second];
                        if (tentative_g < known.g)
                        {
                            known.g = tentative_g;
                            known.f = tentative_f;
                            known.parent = cur;
                            open_heap_.push(seen->second);
                        }
                    }
                }
            }
        }
    }

    logStartDiagnostic("frontier exhausted without reaching the goal.", start_ijk, end_ijk, inf_acc);
    last_failure_reason_ = SearchFailureReason::OPEN_SET_EMPTY;
    return PATH_NOT_FOUND;
}

void Astar::backtrack(int end_index)
{
    path_nodes_.clear();
    for (int idx = end_index; idx != kNoNode; idx = nodes_[idx].parent)
    {
        path_nodes_.push_back(nodes_[idx].ijk);
    }
    std::reverse(path_nodes_.begin(), path_nodes_.end());
}

void Astar::pathShorten(std::vector<openvdb::Vec3d> &sparse_path_out)
{
    if (path_nodes_.size() < 2)
    {
        return;
    }

    const openvdb::math::Transform::ConstPtr tf = map_manager_->get_grid_transform();
    const std::vector<openvdb::Coord> &dense = path_nodes_;

    // A segment is rejected when any extra blocked voxel lies within 0.75
    // voxels of it (endpoints excluded).
    const auto segmentTouchesExtraBlocked = [this](const openvdb::Coord &a,
                                                   const openvdb::Coord &b) -> bool
    {
        if (extra_blocked_voxels_.empty())
        {
            return false;
        }
        const openvdb::Vec3d av(a.x(), a.y(), a.z());
        const openvdb::Vec3d bv(b.x(), b.y(), b.z());
        const openvdb::Vec3d ab = bv - av;
        const double len2 = ab.lengthSqr();
        for (const openvdb::Coord &blocked : extra_blocked_voxels_)
        {
            if (blocked == a || blocked == b)
            {
                continue;
            }
            const openvdb::Vec3d cv(blocked.x(), blocked.y(), blocked.z());
            double t = 0.0;
            if (len2 > 1.0e-9)
            {
                t = std::clamp((cv - av).dot(ab) / len2, 0.0, 1.0);
            }
            const openvdb::Vec3d closest = av + ab * t;
            if ((cv - closest).lengthSqr() <= 0.75 * 0.75)
            {
                return true;
            }
        }
        return false;
    };

    std::vector<openvdb::Coord> sparse;
    openvdb::Coord anchor = dense.front();
    sparse.push_back(anchor);

    const size_t last = dense.size() - 1;
    for (size_t i = 1; i <= last; ++i)
    {
        const bool clear = map_manager_->ray_inflated_clear_index_banded(anchor, dense[i]) &&
                           !segmentTouchesExtraBlocked(anchor, dense[i]);
        if (!clear)
        {
            // Line of sight broke: the previous voxel becomes the next anchor.
            anchor = dense[i - 1];
            sparse.push_back(anchor);
        }
        if (i == last)
        {
            sparse.push_back(dense[i]);
        }
    }

    sparse_path_out.clear();
    sparse_path_out.reserve(sparse.size());
    for (const openvdb::Coord &c : sparse)
    {
        sparse_path_out.push_back(tf->indexToWorld(c));
    }
}

void Astar::setProgressCallback(ProgressCallback cb, double interval_s)
{
    progress_cb_ = std::move(cb);
    progress_interval_s_ = (interval_s > 0.0) ? interval_s : 1.0;
}

visualization_msgs::msg::Marker Astar::getDebugTreeMarker(const std::string &frame_id,
                                                          int max_edges) const
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

    const int count = static_cast<int>(nodes_.size());
    const int stride = (max_edges > 0 && count > max_edges) ? (count + max_edges - 1) / max_edges : 1;
    tree_mk.points.reserve(static_cast<size_t>(count / stride + 1) * 2);

    const auto toPoint = [&tf](const openvdb::Coord &c)
    {
        const openvdb::Vec3d w = tf->indexToWorld(c);
        geometry_msgs::msg::Point p;
        p.x = w.x();
        p.y = w.y();
        p.z = w.z();
        return p;
    };

    for (int i = 0; i < count; i += stride)
    {
        const GridNode &node = nodes_[i];
        if (node.parent == kNoNode)
        {
            continue;
        }
        tree_mk.points.push_back(toPoint(node.ijk));
        tree_mk.points.push_back(toPoint(nodes_[node.parent].ijk));
    }

    return tree_mk;
}
