// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "traj_optimizer/traj_optimizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace
{
using PolyhedraH = gcopter::GCOPTER_PolytopeSFC::PolyhedraH;
using CoordKey = std::tuple<int, int, int>;

CoordKey coordKey(const openvdb::Coord &coord)
{
    return std::make_tuple(coord.x(), coord.y(), coord.z());
}

PolyhedraH flattenCorridorSequence(const traj_optimizer::CorridorSequence &seq)
{
    PolyhedraH hpolys;
    hpolys.reserve(seq.cells.size());
    for (const auto &cell : seq.cells)
    {
        hpolys.push_back(cell.hpoly);
    }
    return hpolys;
}

std::string vecToStr(const Eigen::Vector3d &v)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
    return oss.str();
}

std::vector<Eigen::Vector3d> filterPointsNearPath(
    const std::vector<Eigen::Vector3d> &path,
    const std::vector<Eigen::Vector3d> &points,
    double keep_radius)
{
    if (path.size() < 2 || points.empty() || keep_radius <= 0.0)
    {
        return points;
    }

    struct SegmentAabb
    {
        Eigen::Vector3d lo;
        Eigen::Vector3d hi;
    };

    std::vector<SegmentAabb> segments;
    segments.reserve(path.size() - 1);
    const Eigen::Vector3d pad = Eigen::Vector3d::Constant(keep_radius);
    for (size_t i = 0; i + 1 < path.size(); ++i)
    {
        SegmentAabb seg;
        seg.lo = path[i].cwiseMin(path[i + 1]) - pad;
        seg.hi = path[i].cwiseMax(path[i + 1]) + pad;
        segments.push_back(seg);
    }

    // Keep every point inside any per-segment AABB padded by keep_radius on
    // each axis. This must dominate every FIRI domain built downstream:
    // convexCover boxes are per-axis padded AABBs of sub-chunks of these
    // segments, so a Euclidean-distance cut here would clear obstacle points
    // out of the box corners and let the polytope swallow them.
    std::vector<Eigen::Vector3d> filtered;
    filtered.reserve(points.size());
    for (const auto &p : points)
    {
        bool keep = false;
        for (const auto &seg : segments)
        {
            if ((p.array() >= seg.lo.array()).all() &&
                (p.array() <= seg.hi.array()).all())
            {
                keep = true;
                break;
            }
        }

        if (keep)
        {
            filtered.push_back(p);
        }
    }

    return filtered;
}

openvdb::Coord worldToRoundedCoord(const openvdb::math::Transform &tf,
                                   const Eigen::Vector3d &p)
{
    const openvdb::Vec3d ijk = tf.worldToIndex(openvdb::Vec3d(p.x(), p.y(), p.z()));
    return openvdb::Coord::round(ijk);
}

std::set<CoordKey> findDiagonalSideVoxelKeys(
    const std::vector<Eigen::Vector3d> &path,
    const openvdb::math::Transform &tf)
{
    std::set<CoordKey> center_only;
    if (path.size() < 2)
    {
        return center_only;
    }

    const auto coord_at = [&](const openvdb::Coord &c, int axis) -> int
    {
        return axis == 0 ? c.x() : (axis == 1 ? c.y() : c.z());
    };
    const auto offset_axis = [](openvdb::Coord &c, int axis, int delta)
    {
        if (axis == 0)
        {
            c.setX(c.x() + delta);
        }
        else if (axis == 1)
        {
            c.setY(c.y() + delta);
        }
        else
        {
            c.setZ(c.z() + delta);
        }
    };

    for (size_t seg_idx = 0; seg_idx + 1 < path.size(); ++seg_idx)
    {
        const openvdb::Coord c0 = worldToRoundedCoord(tf, path[seg_idx]);
        const openvdb::Coord c1 = worldToRoundedCoord(tf, path[seg_idx + 1]);
        const int dx = c1.x() - c0.x();
        const int dy = c1.y() - c0.y();
        const int dz = c1.z() - c0.z();
        const int steps = std::max({std::abs(dx), std::abs(dy), std::abs(dz)});
        if (steps <= 0)
        {
            continue;
        }

        const double inv = 1.0 / static_cast<double>(steps);
        const double sx = dx * inv;
        const double sy = dy * inv;
        const double sz = dz * inv;
        openvdb::Coord prev = c0;
        for (int i = 1; i <= steps; ++i)
        {
            const openvdb::Coord curr(
                c0.x() + static_cast<int>(std::round(sx * i)),
                c0.y() + static_cast<int>(std::round(sy * i)),
                c0.z() + static_cast<int>(std::round(sz * i)));
            if (curr == prev)
            {
                continue;
            }

            std::vector<int> axes;
            std::array<int, 3> delta{0, 0, 0};
            for (int axis = 0; axis < 3; ++axis)
            {
                const int d = coord_at(curr, axis) - coord_at(prev, axis);
                if (d != 0)
                {
                    axes.push_back(axis);
                    delta[axis] = d;
                }
            }

            if (axes.size() >= 2)
            {
                const int full_mask = (1 << static_cast<int>(axes.size())) - 1;
                for (int mask = 1; mask < full_mask; ++mask)
                {
                    openvdb::Coord side = prev;
                    for (int bit = 0; bit < static_cast<int>(axes.size()); ++bit)
                    {
                        if (mask & (1 << bit))
                        {
                            const int axis = axes[bit];
                            offset_axis(side, axis, delta[axis]);
                        }
                    }
                    center_only.insert(coordKey(side));
                }
            }
            prev = curr;
        }
    }
    return center_only;
}

std::vector<Eigen::Vector3d> expandInflatedCentersForSfc(
    const std::vector<Eigen::Vector3d> &centers,
    double voxel_size,
    const openvdb::math::Transform &tf,
    const std::set<CoordKey> &center_only_keys,
    std::vector<Eigen::Vector3d> *center_only_centers = nullptr)
{
    if (centers.empty() || voxel_size <= 0.0)
    {
        return centers;
    }

    const double half = 0.5 * voxel_size;
    const double inv_half = 1.0 / half;
    const std::array<Eigen::Vector3d, 26> surface_offsets = {
        Eigen::Vector3d(-half, -half, -half),
        Eigen::Vector3d(-half, -half, +half),
        Eigen::Vector3d(-half, +half, -half),
        Eigen::Vector3d(-half, +half, +half),
        Eigen::Vector3d(+half, -half, -half),
        Eigen::Vector3d(+half, -half, +half),
        Eigen::Vector3d(+half, +half, -half),
        Eigen::Vector3d(+half, +half, +half),
        Eigen::Vector3d(-half, 0.0, 0.0),
        Eigen::Vector3d(+half, 0.0, 0.0),
        Eigen::Vector3d(0.0, -half, 0.0),
        Eigen::Vector3d(0.0, +half, 0.0),
        Eigen::Vector3d(0.0, 0.0, -half),
        Eigen::Vector3d(0.0, 0.0, +half),
        Eigen::Vector3d(-half, -half, 0.0),
        Eigen::Vector3d(-half, +half, 0.0),
        Eigen::Vector3d(+half, -half, 0.0),
        Eigen::Vector3d(+half, +half, 0.0),
        Eigen::Vector3d(-half, 0.0, -half),
        Eigen::Vector3d(-half, 0.0, +half),
        Eigen::Vector3d(+half, 0.0, -half),
        Eigen::Vector3d(+half, 0.0, +half),
        Eigen::Vector3d(0.0, -half, -half),
        Eigen::Vector3d(0.0, -half, +half),
        Eigen::Vector3d(0.0, +half, -half),
        Eigen::Vector3d(0.0, +half, +half),
    };

    std::vector<Eigen::Vector3d> points;
    points.reserve(centers.size() * 27);
    std::set<std::tuple<long long, long long, long long>> unique_quantized;

    auto try_insert_point = [&](const Eigen::Vector3d &p)
    {
        const auto key = std::make_tuple(
            static_cast<long long>(std::llround(p.x() * inv_half)),
            static_cast<long long>(std::llround(p.y() * inv_half)),
            static_cast<long long>(std::llround(p.z() * inv_half)));
        if (unique_quantized.insert(key).second)
        {
            points.push_back(p);
        }
    };

    for (const auto &center : centers)
    {
        try_insert_point(center);
        const bool center_only =
            center_only_keys.find(coordKey(worldToRoundedCoord(tf, center))) != center_only_keys.end();
        if (center_only)
        {
            if (center_only_centers)
            {
                center_only_centers->push_back(center);
            }
            continue;
        }
        for (const auto &offset : surface_offsets)
        {
            try_insert_point(center + offset);
        }
    }
    return points;
}

void fillDebugInfo(traj_optimizer::DebugInfo *debug_out,
                   const std::vector<Eigen::Vector3d> &path,
                   const openvdb::CoordBBox &bbox,
                   const std::vector<Eigen::Vector3d> &inflated_points,
                   const PolyhedraH &corridor)
{
    if (!debug_out)
    {
        return;
    }

    debug_out->input_path = path;
    debug_out->local_bbox = bbox;
    debug_out->inflated_points = inflated_points;
    debug_out->corridor_hpolys = corridor;
    debug_out->corridor_seq.cells.clear();
    debug_out->piece_idx.clear();
    debug_out->observation_cell_index = -1;
    debug_out->observation_knot_index = -1;
}

void fillDebugInfo(traj_optimizer::DebugInfo *debug_out,
                   const std::vector<Eigen::Vector3d> &path,
                   const openvdb::CoordBBox &bbox,
                   const std::vector<Eigen::Vector3d> &inflated_points,
                   const traj_optimizer::CorridorSequence &corridor_seq)
{
    if (!debug_out)
    {
        return;
    }

    debug_out->input_path = path;
    debug_out->local_bbox = bbox;
    debug_out->inflated_points = inflated_points;
    debug_out->corridor_seq = corridor_seq;
    debug_out->corridor_hpolys = flattenCorridorSequence(corridor_seq);
}

std::string diagnoseCorridorForGcopter(const PolyhedraH &corridor)
{
    if (corridor.empty())
    {
        return "safeCorridor is empty";
    }

    Eigen::Matrix3Xd vertices;
    for (size_t i = 0; i < corridor.size(); ++i)
    {
        if (!geo_utils::enumerateVs(corridor[i], vertices))
        {
            std::ostringstream oss;
            oss << "poly[" << i << "] has no valid interior / vertex enumeration failed";
            return oss.str();
        }
        if (vertices.cols() <= 0)
        {
            std::ostringstream oss;
            oss << "poly[" << i << "] enumerated zero vertices";
            return oss.str();
        }
    }

    for (size_t i = 0; i + 1 < corridor.size(); ++i)
    {
        Eigen::MatrixX4d overlap(corridor[i].rows() + corridor[i + 1].rows(), 4);
        overlap.topRows(corridor[i].rows()) = corridor[i];
        overlap.bottomRows(corridor[i + 1].rows()) = corridor[i + 1];
        if (!geo_utils::enumerateVs(overlap, vertices))
        {
            std::ostringstream oss;
            oss << "overlap poly[" << i << "]<->poly[" << (i + 1)
                << "] has no valid interior / vertex enumeration failed";
            return oss.str();
        }
        if (vertices.cols() <= 1)
        {
            std::ostringstream oss;
            oss << "overlap poly[" << i << "]<->poly[" << (i + 1)
                << "] is degenerate (<=1 vertex)";
            return oss.str();
        }
    }

    return "corridor passes pre-check; setup likely failed in shortest-path or piece allocation";
}

template <typename Derived>
std::string arrayToStr(const Eigen::MatrixBase<Derived> &arr)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << "[";
    for (int i = 0; i < arr.size(); ++i)
    {
        if (i > 0)
        {
            oss << ", ";
        }
        oss << arr.derived()(i);
    }
    oss << "]";
    return oss.str();
}

std::string sanitizeForPath(std::string text)
{
    for (char &c : text)
    {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' && c != '-')
        {
            c = '_';
        }
    }
    while (text.find("__") != std::string::npos)
    {
        text.replace(text.find("__"), 2, "_");
    }
    if (text.empty())
    {
        text = "snapshot";
    }
    return text;
}

std::string snapshotStamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        now.time_since_epoch())
                        .count();
    return std::to_string(ns);
}

std::string jsonNumber(double value)
{
    if (!std::isfinite(value))
    {
        return "null";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

double normalizedHpolyViolation(const Eigen::MatrixX4d &hpoly,
                                const Eigen::Vector3d &pos);

void writeVectorCsv(const std::filesystem::path &path,
                    const std::vector<Eigen::Vector3d> &points)
{
    std::ofstream out(path);
    out << "idx,x,y,z\n";
    out << std::setprecision(17);
    for (size_t i = 0; i < points.size(); ++i)
    {
        out << i << "," << points[i].x() << "," << points[i].y() << "," << points[i].z() << "\n";
    }
}

void writeMatrixPointsCsv(const std::filesystem::path &path,
                          const gcopter::DebugSnapshot &snapshot)
{
    std::ofstream out(path);
    out << "kind,idx,x,y,z\n";
    out << std::setprecision(17);
    auto write_cols = [&](std::string_view kind, const Eigen::Matrix3Xd &mat)
    {
        for (int i = 0; i < mat.cols(); ++i)
        {
            out << kind << "," << i << "," << mat(0, i) << "," << mat(1, i) << "," << mat(2, i) << "\n";
        }
    };
    write_cols("short_path", snapshot.shortPath);
    write_cols("initial_points", snapshot.initialPoints);
    write_cols("final_points", snapshot.finalPoints);
}

void writeSfcCsv(const std::filesystem::path &path,
                 const PolyhedraH &corridor,
                 const std::vector<int> &cell_types)
{
    std::ofstream out(path);
    out << "cell,row,cell_type,nx,ny,nz,d\n";
    out << std::setprecision(17);
    for (size_t cell = 0; cell < corridor.size(); ++cell)
    {
        const int type = cell < cell_types.size() ? cell_types[cell] : -1;
        const Eigen::MatrixX4d &hpoly = corridor[cell];
        for (int r = 0; r < hpoly.rows(); ++r)
        {
            out << cell << "," << r << "," << type << ","
                << hpoly(r, 0) << "," << hpoly(r, 1) << ","
                << hpoly(r, 2) << "," << hpoly(r, 3) << "\n";
        }
    }
}

double yawAtPieceTime(const Eigen::VectorXd &yaw_knots,
                      int piece_idx,
                      double piece_time,
                      double piece_duration)
{
    if (piece_idx < 0 || yaw_knots.size() <= piece_idx + 1)
    {
        return 0.0;
    }
    const double frac = piece_duration > 1.0e-9
                            ? std::clamp(piece_time / piece_duration, 0.0, 1.0)
                            : 1.0;
    return yaw_knots(piece_idx) + frac * (yaw_knots(piece_idx + 1) - yaw_knots(piece_idx));
}

void writeTrajectorySamplesCsv(
    const std::filesystem::path &path,
    const traj_optimizer::OptimizedPlan::PositionTrajectory *traj,
    const Eigen::VectorXd &yaw_knots,
    const Eigen::VectorXi &hpoly_idx,
    const PolyhedraH &corridor,
    double dense_dt,
    const std::shared_ptr<VDBMap> &map_manager = nullptr)
{
    std::ofstream out(path);
    out << "piece,t,x,y,z,vx,vy,vz,ax,ay,az,yaw,hpoly_idx,corridor_violation,inflated_active,inflate_value\n";
    out << std::setprecision(17);
    if (!traj || traj->getPieceNum() <= 0)
    {
        return;
    }

    dense_dt = std::max(dense_dt, 1.0e-3);
    double t_global = 0.0;
    for (int i = 0; i < traj->getPieceNum(); ++i)
    {
        const auto &piece = (*traj)[i];
        const double duration = piece.getDuration();
        const int steps = std::max(1, static_cast<int>(std::ceil(duration / dense_dt)));
        const int hidx = i < hpoly_idx.size() ? hpoly_idx(i) : -1;
        for (int j = 0; j <= steps; ++j)
        {
            if (i > 0 && j == 0)
            {
                continue;
            }
            const double local_t = duration * static_cast<double>(j) / static_cast<double>(steps);
            const Eigen::Vector3d pos = piece.getPos(local_t);
            const Eigen::Vector3d vel = piece.getVel(local_t);
            const Eigen::Vector3d acc = piece.getAcc(local_t);
            const double violation =
                (hidx >= 0 && hidx < static_cast<int>(corridor.size()))
                    ? normalizedHpolyViolation(corridor[hidx], pos)
                    : 0.0;
            int inflate_value = 0;
            const bool inflated_active =
                map_manager &&
                map_manager->query_is_inflated_at_world(pos, inflate_value) &&
                inflate_value > 0;
            out << i << "," << (t_global + local_t) << ","
                << pos.x() << "," << pos.y() << "," << pos.z() << ","
                << vel.x() << "," << vel.y() << "," << vel.z() << ","
                << acc.x() << "," << acc.y() << "," << acc.z() << ","
                << yawAtPieceTime(yaw_knots, i, local_t, duration) << ","
                << hidx << "," << violation << ","
                << (inflated_active ? 1 : 0) << "," << inflate_value << "\n";
        }
        t_global += duration;
    }
}

double chainLength(const Eigen::Vector3d &start,
                   const Eigen::Matrix3Xd &inner_points,
                   const Eigen::Vector3d &goal)
{
    double length = 0.0;
    Eigen::Vector3d prev = start;
    for (int i = 0; i < inner_points.cols(); ++i)
    {
        length += (inner_points.col(i) - prev).norm();
        prev = inner_points.col(i);
    }
    length += (goal - prev).norm();
    return length;
}

double polylineLength(const std::vector<Eigen::Vector3d> &points)
{
    double length = 0.0;
    for (size_t i = 0; i + 1 < points.size(); ++i)
    {
        length += (points[i + 1] - points[i]).norm();
    }
    return length;
}

struct DurationStats
{
    double total = 0.0;
    double min = 0.0;
    double mean = 0.0;
    double max = 0.0;
};

DurationStats computeDurationStats(const Eigen::VectorXd &durations)
{
    DurationStats out;
    if (durations.size() <= 0)
    {
        return out;
    }
    out.total = durations.sum();
    out.min = durations.minCoeff();
    out.max = durations.maxCoeff();
    out.mean = out.total / static_cast<double>(durations.size());
    return out;
}

struct OptimizerLogConfig
{
    double time_weight = 0.0;
    double jerk_weight = 1.0;
    double length_per_piece = 0.0;
    double init_alloc_speed_ratio = 0.0;
    double init_max_acc = 0.0;
    double init_time_margin = 1.0;
    double max_yaw_rate = 0.0;
    Eigen::VectorXd magnitude_bounds;
    Eigen::VectorXd physical_params;
};

struct PieceSpeedDiag
{
    int index = -1;
    int hpoly_index = -1;
    int cell_type = -1;
    double duration = 0.0;
    double length = 0.0;
    double avg_speed = 0.0;
    double max_corridor_violation = 0.0;
    double max_vel = 0.0;
    double max_tilt = 0.0;
    double max_thrust = 0.0;
    Eigen::Vector3d start = Eigen::Vector3d::Zero();
    Eigen::Vector3d end = Eigen::Vector3d::Zero();
};

struct TrajectoryDiag
{
    double length = 0.0;
    double chord = 0.0;
    double avg_speed = 0.0;
    double straightness = 1.0;
    double max_vel = 0.0;
    double max_acc = 0.0;
    double max_omg = 0.0;
    double max_tilt = 0.0;
    double min_thrust = std::numeric_limits<double>::infinity();
    double max_thrust = 0.0;
    double max_yaw_rate = 0.0;
    double max_corridor_violation = 0.0;
    int samples = 0;
    int low_speed_samples = 0;
    int vel_violations = 0;
    int omg_violations = 0;
    int tilt_violations = 0;
    int thrust_violations = 0;
    int yaw_rate_violations = 0;
    int corridor_violations = 0;
    int corridor_violations_after_first_piece = 0;
    double max_corridor_violation_after_first_piece = 0.0;
    std::vector<PieceSpeedDiag> slow_pieces;
};

struct InflatedCollisionDiag
{
    bool collision = false;
    int piece_index = -1;
    int sample_index = -1;
    int inflate_value = 0;
    double global_time = 0.0;
    double local_time = 0.0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    int samples_checked = 0;
};

double normalizedHpolyViolation(const Eigen::MatrixX4d &hpoly,
                                const Eigen::Vector3d &pos)
{
    double max_violation = -std::numeric_limits<double>::infinity();
    for (int r = 0; r < hpoly.rows(); ++r)
    {
        const double norm = hpoly.row(r).head<3>().norm();
        if (norm <= 1.0e-12)
        {
            continue;
        }
        const double violation = (hpoly.row(r).head<3>().dot(pos) + hpoly(r, 3)) / norm;
        max_violation = std::max(max_violation, violation);
    }
    return std::max(max_violation, 0.0);
}

TrajectoryDiag evaluateTrajectory(
    const traj_optimizer::OptimizedPlan::PositionTrajectory &traj,
    const Eigen::VectorXd &yaw_knots,
    const gcopter::DebugSnapshot &snapshot,
    const PolyhedraH &corridor,
    const OptimizerLogConfig &config)
{
    TrajectoryDiag diag;
    const int piece_num = traj.getPieceNum();
    if (piece_num <= 0)
    {
        return diag;
    }

    flatness::FlatnessMap flatmap;
    if (config.physical_params.size() >= 6)
    {
        flatmap.reset(config.physical_params(0),
                      config.physical_params(1),
                      config.physical_params(2),
                      config.physical_params(3),
                      config.physical_params(4),
                      config.physical_params(5));
    }

    const double vel_limit = config.magnitude_bounds.size() > 0 ? config.magnitude_bounds(0) : INFINITY;
    const double omg_limit = config.magnitude_bounds.size() > 1 ? config.magnitude_bounds(1) : INFINITY;
    const double tilt_limit = config.magnitude_bounds.size() > 2 ? config.magnitude_bounds(2) : INFINITY;
    const double thrust_min = config.magnitude_bounds.size() > 3 ? config.magnitude_bounds(3) : -INFINITY;
    const double thrust_max = config.magnitude_bounds.size() > 4 ? config.magnitude_bounds(4) : INFINITY;
    const double low_speed_threshold = std::isfinite(vel_limit) ? 0.1 * vel_limit : 0.1;

    std::vector<PieceSpeedDiag> piece_diags;
    piece_diags.reserve(piece_num);
    std::vector<Eigen::Vector3d> sampled_positions;
    sampled_positions.reserve(static_cast<size_t>(piece_num) * 9 + 1);

    for (int i = 0; i < piece_num; ++i)
    {
        const auto &piece = traj[i];
        const double dt = std::max(piece.getDuration(), 1.0e-9);
        const int hidx = (i < snapshot.hPolyIdx.size()) ? snapshot.hPolyIdx(i) : -1;
        const int cell_type =
            (hidx >= 0 && hidx < static_cast<int>(snapshot.cellTypes.size())) ? snapshot.cellTypes[hidx] : -1;
        const double yaw_rate =
            (yaw_knots.size() == piece_num + 1) ? (yaw_knots(i + 1) - yaw_knots(i)) / dt : 0.0;

        PieceSpeedDiag piece_diag;
        piece_diag.index = i;
        piece_diag.hpoly_index = hidx;
        piece_diag.cell_type = cell_type;
        piece_diag.duration = dt;
        piece_diag.start = piece.getPos(0.0);
        piece_diag.end = piece.getPos(piece.getDuration());
        diag.max_yaw_rate = std::max(diag.max_yaw_rate, std::abs(yaw_rate));
        if (config.max_yaw_rate > 0.0 && std::abs(yaw_rate) > config.max_yaw_rate + 1.0e-6)
        {
            ++diag.yaw_rate_violations;
        }

        const int samples_per_piece = 8;
        Eigen::Vector3d prev = piece_diag.start;
        for (int j = 0; j <= samples_per_piece; ++j)
        {
            const double alpha = static_cast<double>(j) / static_cast<double>(samples_per_piece);
            const double t = alpha * piece.getDuration();
            const Eigen::Vector3d pos = piece.getPos(t);
            const Eigen::Vector3d vel = piece.getVel(t);
            const Eigen::Vector3d acc = piece.getAcc(t);
            const Eigen::Vector3d jer = piece.getJer(t);
            if (j > 0)
            {
                const double ds = (pos - prev).norm();
                diag.length += ds;
                piece_diag.length += ds;
            }
            prev = pos;

            if (i == 0 || j > 0)
            {
                sampled_positions.push_back(pos);
            }
            ++diag.samples;
            const double vel_norm = vel.norm();
            const double acc_norm = acc.norm();
            diag.max_vel = std::max(diag.max_vel, vel_norm);
            diag.max_acc = std::max(diag.max_acc, acc_norm);
            piece_diag.max_vel = std::max(piece_diag.max_vel, vel_norm);
            if (vel_norm < low_speed_threshold)
            {
                ++diag.low_speed_samples;
            }
            if (vel_norm > vel_limit + 1.0e-6)
            {
                ++diag.vel_violations;
            }

            double thrust = 0.0;
            Eigen::Vector4d quat = Eigen::Vector4d::Zero();
            Eigen::Vector3d omg = Eigen::Vector3d::Zero();
            flatmap.forward(vel, acc, jer, 0.0, 0.0, thrust, quat, omg);
            const double omg_norm = omg.norm();
            const double cos_tilt = std::clamp(1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2)),
                                               -1.0, 1.0);
            const double tilt = std::acos(cos_tilt);
            diag.max_omg = std::max(diag.max_omg, omg_norm);
            diag.max_tilt = std::max(diag.max_tilt, tilt);
            diag.min_thrust = std::min(diag.min_thrust, thrust);
            diag.max_thrust = std::max(diag.max_thrust, thrust);
            piece_diag.max_tilt = std::max(piece_diag.max_tilt, tilt);
            piece_diag.max_thrust = std::max(piece_diag.max_thrust, thrust);
            if (omg_norm > omg_limit + 1.0e-6)
            {
                ++diag.omg_violations;
            }
            if (tilt > tilt_limit + 1.0e-6)
            {
                ++diag.tilt_violations;
            }
            if (thrust < thrust_min - 1.0e-6 || thrust > thrust_max + 1.0e-6)
            {
                ++diag.thrust_violations;
            }

            if (hidx >= 0 && hidx < static_cast<int>(corridor.size()))
            {
                const double violation = normalizedHpolyViolation(corridor[hidx], pos);
                diag.max_corridor_violation = std::max(diag.max_corridor_violation, violation);
                piece_diag.max_corridor_violation = std::max(piece_diag.max_corridor_violation, violation);
                if (violation > 1.0e-6)
                {
                    ++diag.corridor_violations;
                    if (i > 0)
                    {
                        ++diag.corridor_violations_after_first_piece;
                        diag.max_corridor_violation_after_first_piece =
                            std::max(diag.max_corridor_violation_after_first_piece, violation);
                    }
                }
            }
        }
        piece_diag.avg_speed = piece_diag.length / dt;
        piece_diags.push_back(piece_diag);
    }

    const double total_time = traj.getTotalDuration();
    diag.avg_speed = diag.length / std::max(total_time, 1.0e-9);
    if (!sampled_positions.empty())
    {
        diag.chord = (sampled_positions.back() - sampled_positions.front()).norm();
    }
    diag.straightness = diag.chord > 1.0e-6 ? diag.length / diag.chord : 1.0;
    if (!std::isfinite(diag.min_thrust))
    {
        diag.min_thrust = 0.0;
    }

    std::sort(piece_diags.begin(), piece_diags.end(),
              [](const PieceSpeedDiag &a, const PieceSpeedDiag &b)
              {
                  return a.avg_speed < b.avg_speed;
              });
    const size_t keep = std::min<size_t>(5, piece_diags.size());
    diag.slow_pieces.assign(piece_diags.begin(), piece_diags.begin() + keep);
    return diag;
}

InflatedCollisionDiag checkTrajectoryInflatedCollision(
    const traj_optimizer::OptimizedPlan::PositionTrajectory &traj,
    const std::shared_ptr<VDBMap> &map_manager,
    double dense_dt = 0.05)
{
    InflatedCollisionDiag diag;
    if (!map_manager || traj.getPieceNum() <= 0)
    {
        return diag;
    }

    dense_dt = std::max(dense_dt, 1.0e-3);
    double global_time = 0.0;
    for (int i = 0; i < traj.getPieceNum(); ++i)
    {
        const auto &piece = traj[i];
        const double duration = piece.getDuration();
        const int steps = std::max(1, static_cast<int>(std::ceil(duration / dense_dt)));
        for (int j = 0; j <= steps; ++j)
        {
            if (i > 0 && j == 0)
            {
                continue;
            }

            const double local_time =
                duration * static_cast<double>(j) / static_cast<double>(steps);
            const Eigen::Vector3d pos = piece.getPos(local_time);
            int inflate_value = 0;
            ++diag.samples_checked;
            if (map_manager->query_is_inflated_at_world(pos, inflate_value) && inflate_value > 0)
            {
                diag.collision = true;
                diag.piece_index = i;
                diag.sample_index = j;
                diag.inflate_value = inflate_value;
                diag.global_time = global_time + local_time;
                diag.local_time = local_time;
                diag.position = pos;
                return diag;
            }
        }
        global_time += duration;
    }

    return diag;
}

std::string slowPiecesToStr(const std::vector<PieceSpeedDiag> &pieces)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    for (size_t i = 0; i < pieces.size(); ++i)
    {
        if (i > 0)
        {
            oss << " | ";
        }
        const PieceSpeedDiag &p = pieces[i];
        oss << "#" << p.index
            << " dt=" << p.duration
            << " len=" << p.length
            << " avg_v=" << p.avg_speed
            << " h=" << p.hpoly_index
            << " type=" << p.cell_type
            << " pos=" << vecToStr(p.start) << "->" << vecToStr(p.end)
            << " max(v/tilt/thr/corr)=" << p.max_vel << "/" << p.max_tilt
            << "/" << p.max_thrust << "/" << p.max_corridor_violation;
    }
    return oss.str();
}

bool writeTrajOptSnapshot(const rclcpp::Logger &logger,
                          const std::string &snapshot_dir,
                          const char *tag,
                          const std::string &trigger,
                          const char *timing,
                          const std::vector<Eigen::Vector3d> &path,
                          const std::vector<Eigen::Vector3d> &inflated_points,
                          const PolyhedraH &corridor,
                          const traj_optimizer::BoundaryState &start_state,
                          const traj_optimizer::BoundaryState &goal_state,
                          double start_yaw,
                          double goal_yaw,
                          const gcopter::DebugSnapshot &snapshot,
                          const OptimizerLogConfig &config,
                          double opt_cost,
                          const traj_optimizer::OptimizedPlan::PositionTrajectory *traj = nullptr,
                          const std::shared_ptr<VDBMap> &map_manager = nullptr,
                          const std::vector<Eigen::Vector3d> *inflated_voxel_centers = nullptr,
                          const std::vector<Eigen::Vector3d> *center_only_voxels = nullptr)
{
    try
    {
        const std::filesystem::path root(snapshot_dir.empty()
                                             ? "robot/ros_ws/src/autonomy/4_global/b_planners/ffa_planner/log_debug"
                                             : snapshot_dir);
        const std::string clean_tag = sanitizeForPath(tag ? tag : "trajopt");
        const std::string clean_trigger = sanitizeForPath(trigger);
        const std::filesystem::path dir =
            root / (snapshotStamp() + "_" + clean_tag + "_" + clean_trigger);
        std::filesystem::create_directories(dir);

        writeVectorCsv(dir / "input_path.csv", path);
        writeVectorCsv(dir / "inflated_points.csv", inflated_points);
        if (inflated_voxel_centers)
        {
            writeVectorCsv(dir / "inflated_voxels.csv", *inflated_voxel_centers);
        }
        if (center_only_voxels)
        {
            writeVectorCsv(dir / "center_only_voxels.csv", *center_only_voxels);
        }
        writeSfcCsv(dir / "sfc_cells.csv", corridor, snapshot.cellTypes);
        writeMatrixPointsCsv(dir / "optimizer_points.csv", snapshot);
        writeTrajectorySamplesCsv(dir / "trajectory_samples.csv",
                                  traj,
                                  snapshot.finalYawKnots,
                                  snapshot.hPolyIdx,
                                  corridor,
                                  0.05,
                                  map_manager);

        std::ofstream manifest(dir / "manifest.json");
        manifest << std::setprecision(17);
        manifest << "{\n";
        manifest << "  \"tag\": \"" << (tag ? tag : "") << "\",\n";
        manifest << "  \"trigger\": \"" << trigger << "\",\n";
        manifest << "  \"timing\": \"" << (timing ? timing : "") << "\",\n";
        manifest << "  \"input_path_points\": " << path.size() << ",\n";
        manifest << "  \"inflated_points\": " << inflated_points.size() << ",\n";
        manifest << "  \"inflated_voxels\": " << (inflated_voxel_centers ? inflated_voxel_centers->size() : 0) << ",\n";
        manifest << "  \"center_only_voxels\": " << (center_only_voxels ? center_only_voxels->size() : 0) << ",\n";
        double voxel_size = 0.0;
        if (map_manager && map_manager->get_grid_transform())
        {
            voxel_size = map_manager->get_grid_transform()->voxelSize()[0];
        }
        manifest << "  \"voxel_size\": " << jsonNumber(voxel_size) << ",\n";
        manifest << "  \"corridor_cells\": " << corridor.size() << ",\n";
        manifest << "  \"polyN\": " << snapshot.polyNum << ",\n";
        manifest << "  \"pieceN\": " << snapshot.pieceNum << ",\n";
        manifest << "  \"pieceIdx\": \"" << arrayToStr(snapshot.pieceIdx) << "\",\n";
        manifest << "  \"hPolyIdx\": \"" << arrayToStr(snapshot.hPolyIdx) << "\",\n";
        manifest << "  \"observationCellIndex\": " << snapshot.observationCellIndex << ",\n";
        manifest << "  \"observationKnotIndex\": " << snapshot.observationKnotIndex << ",\n";
        manifest << "  \"seamPieceIndex\": " << snapshot.seamPieceIndex << ",\n";
        manifest << "  \"hasSeamAnchor\": " << (snapshot.hasSeamAnchor ? "true" : "false") << ",\n";
        manifest << "  \"start\": {\"pos\": [" << start_state.pos.x() << ", " << start_state.pos.y() << ", " << start_state.pos.z()
                 << "], \"vel\": [" << start_state.vel.x() << ", " << start_state.vel.y() << ", " << start_state.vel.z()
                 << "], \"acc\": [" << start_state.acc.x() << ", " << start_state.acc.y() << ", " << start_state.acc.z()
                 << "], \"yaw\": " << start_yaw << "},\n";
        manifest << "  \"goal\": {\"pos\": [" << goal_state.pos.x() << ", " << goal_state.pos.y() << ", " << goal_state.pos.z()
                 << "], \"vel\": [" << goal_state.vel.x() << ", " << goal_state.vel.y() << ", " << goal_state.vel.z()
                 << "], \"acc\": [" << goal_state.acc.x() << ", " << goal_state.acc.y() << ", " << goal_state.acc.z()
                 << "], \"yaw\": " << goal_yaw << "},\n";
        manifest << "  \"params\": {\"time_weight\": " << config.time_weight
                 << ", \"jerk_weight\": " << config.jerk_weight
                 << ", \"length_per_piece\": " << config.length_per_piece
                 << ", \"init_alloc_speed_ratio\": " << config.init_alloc_speed_ratio
                 << ", \"init_max_acc\": " << config.init_max_acc
                 << ", \"init_time_margin\": " << config.init_time_margin
                 << ", \"max_yaw_rate\": " << config.max_yaw_rate << "},\n";
        manifest << "  \"costs\": {\"total\": " << jsonNumber(opt_cost)
                 << ", \"jerk\": " << jsonNumber(snapshot.costEnergy)
                 << ", \"time\": " << jsonNumber(snapshot.costTime)
                 << ", \"pos\": " << jsonNumber(snapshot.costPos)
                 << ", \"vel\": " << jsonNumber(snapshot.costVel)
                 << ", \"omg\": " << jsonNumber(snapshot.costOmg)
                 << ", \"tilt\": " << jsonNumber(snapshot.costTheta)
                 << ", \"thrust\": " << jsonNumber(snapshot.costThrust)
                 << ", \"yaw_smooth\": " << jsonNumber(snapshot.costYawSmooth)
                 << ", \"yaw_rate\": " << jsonNumber(snapshot.costYawRateLimit)
                 << ", \"seam_pos\": " << jsonNumber(snapshot.costSeamPos)
                 << ", \"seam_yaw\": " << jsonNumber(snapshot.costSeamYaw)
                 << ", \"obs_yaw\": " << jsonNumber(snapshot.costObservationYaw) << "}\n";
        manifest << "}\n";

        RCLCPP_WARN(logger,
                    "%s snapshot saved: %s",
                    tag ? tag : "[TrajOpt]",
                    dir.string().c_str());
        return true;
    }
    catch (const std::exception &e)
    {
        RCLCPP_WARN(logger,
                    "%s snapshot write failed: %s",
                    tag ? tag : "[TrajOpt]",
                    e.what());
        return false;
    }
}

void logInflatedCollisionAndSnapshot(
    const rclcpp::Logger &logger,
    const std::string &snapshot_dir,
    bool snapshot_enable,
    const char *tag,
    const std::string &timing,
    const InflatedCollisionDiag &collision,
    const std::vector<Eigen::Vector3d> &path,
    const std::vector<Eigen::Vector3d> &inflated_points,
    const PolyhedraH &corridor,
    const traj_optimizer::BoundaryState &start_state,
    const traj_optimizer::BoundaryState &goal_state,
    double start_yaw,
    double goal_yaw,
    const gcopter::DebugSnapshot &debug_snapshot,
    const OptimizerLogConfig &log_config,
    double opt_cost,
    const traj_optimizer::OptimizedPlan::PositionTrajectory &traj,
    const std::shared_ptr<VDBMap> &map_manager,
    bool reject_trajectory,
    const std::vector<Eigen::Vector3d> *inflated_voxel_centers = nullptr,
    const std::vector<Eigen::Vector3d> *center_only_voxels = nullptr)
{
    std::ostringstream collision_timing;
    collision_timing << timing
                     << " collision_piece=" << collision.piece_index
                     << " sample=" << collision.sample_index
                     << " t=" << std::fixed << std::setprecision(3) << collision.global_time
                     << " pos=" << vecToStr(collision.position)
                     << " inflate=" << collision.inflate_value
                     << " checked=" << collision.samples_checked;

    if (snapshot_enable)
    {
        writeTrajOptSnapshot(logger,
                             snapshot_dir,
                             tag,
                             "inflated_collision",
                             collision_timing.str().c_str(),
                             path,
                             inflated_points,
                             corridor,
                             start_state,
                             goal_state,
                             start_yaw,
                             goal_yaw,
                             debug_snapshot,
                             log_config,
                             opt_cost,
                             &traj,
                             map_manager,
                             inflated_voxel_centers,
                             center_only_voxels);
    }

    RCLCPP_WARN(logger,
                "%s sampled optimized trajectory hit inflated map; %s | "
                "piece=%d sample=%d t=%.3f local_t=%.3f pos=%s inflate=%d checked=%d | %s",
                tag,
                reject_trajectory ? "rejecting optimized trajectory" : "keeping optimized trajectory",
                collision.piece_index,
                collision.sample_index,
                collision.global_time,
                collision.local_time,
                vecToStr(collision.position).c_str(),
                collision.inflate_value,
                collision.samples_checked,
                timing.c_str());
}

void logOptimizerDiagnostics(const rclcpp::Logger &logger,
                             const char *tag,
                             const char *timing,
                             const std::vector<Eigen::Vector3d> &path,
                             const PolyhedraH &corridor,
                             const traj_optimizer::BoundaryState &start_state,
                             const traj_optimizer::BoundaryState &goal_state,
                             double start_yaw,
                             double goal_yaw,
                             const gcopter::DebugSnapshot &snapshot,
                             const OptimizerLogConfig &config,
                             double opt_cost,
                             const traj_optimizer::OptimizedPlan::PositionTrajectory *traj = nullptr)
{
    const double input_path_len = polylineLength(path);
    const double chord_len = path.size() >= 2 ? (path.back() - path.front()).norm() : 0.0;
    const DurationStats init_dt = computeDurationStats(snapshot.initialTimes);
    const DurationStats final_dt = computeDurationStats(snapshot.finalTimes);
    const double init_len = chainLength(start_state.pos, snapshot.initialPoints, goal_state.pos);
    const double init_avg_v = init_len / std::max(init_dt.total, 1.0e-9);
    const int yaw_knots = snapshot.finalYawKnots.size();

    RCLCPP_INFO(logger,
                "%s setup | %s | input_len=%.2f chord=%.2f cells=%d pieces=%d yaw_knots=%d "
                "pieceIdx=%s obs(cell/knot)=%d/%d seam_piece=%d | "
                "start p/v/a=%s/%s/%s goal p/v/a=%s/%s/%s yaw=%.2f->%.2f | "
                "params time_w=%.3f jerk_w=%.3f len_piece=%.3f init(speed_ratio/acc/margin)=%.2f/%.2f/%.2f",
                tag,
                timing ? timing : "",
                input_path_len,
                chord_len,
                snapshot.polyNum,
                snapshot.pieceNum,
                yaw_knots,
                arrayToStr(snapshot.pieceIdx).c_str(),
                snapshot.observationCellIndex,
                snapshot.observationKnotIndex,
                snapshot.seamPieceIndex,
                vecToStr(start_state.pos).c_str(),
                vecToStr(start_state.vel).c_str(),
                vecToStr(start_state.acc).c_str(),
                vecToStr(goal_state.pos).c_str(),
                vecToStr(goal_state.vel).c_str(),
                vecToStr(goal_state.acc).c_str(),
                start_yaw,
                goal_yaw,
                config.time_weight,
                config.jerk_weight,
                config.length_per_piece,
                config.init_alloc_speed_ratio,
                config.init_max_acc,
                config.init_time_margin);

    if (!traj || traj->getPieceNum() <= 0)
    {
        RCLCPP_WARN(logger,
                    "%s no valid final trajectory | init_time=%.2f init_len=%.2f init_avg_v=%.2f "
                    "cost total=%.3f jerk=%.3f time=%.3f pos=%.3f vel=%.3f omg=%.3f tilt=%.3f thrust=%.3f yaw_smooth=%.3f yaw_rate=%.3f seam=%.3f/%.3f obs=%.3f",
                    tag,
                    init_dt.total,
                    init_len,
                    init_avg_v,
                    opt_cost,
                    snapshot.costEnergy,
                    snapshot.costTime,
                    snapshot.costPos,
                    snapshot.costVel,
                    snapshot.costOmg,
                    snapshot.costTheta,
                    snapshot.costThrust,
                    snapshot.costYawSmooth,
                    snapshot.costYawRateLimit,
                    snapshot.costSeamPos,
                    snapshot.costSeamYaw,
                    snapshot.costObservationYaw);
        return;
    }

    const TrajectoryDiag final_diag =
        evaluateTrajectory(*traj, snapshot.finalYawKnots, snapshot, corridor, config);
    const double low_speed_ratio =
        final_diag.samples > 0 ? static_cast<double>(final_diag.low_speed_samples) / final_diag.samples : 0.0;
    const double vel_limit = config.magnitude_bounds.size() > 0 ? config.magnitude_bounds(0) : 0.0;
    const double omg_limit = config.magnitude_bounds.size() > 1 ? config.magnitude_bounds(1) : 0.0;
    const double tilt_limit = config.magnitude_bounds.size() > 2 ? config.magnitude_bounds(2) : 0.0;
    const double thrust_min = config.magnitude_bounds.size() > 3 ? config.magnitude_bounds(3) : 0.0;
    const double thrust_max = config.magnitude_bounds.size() > 4 ? config.magnitude_bounds(4) : 0.0;
    const double nominal_min_time =
        (vel_limit > 1.0e-6) ? final_diag.length / vel_limit : std::numeric_limits<double>::infinity();
    const bool suspicious_slow =
        (vel_limit > 1.0e-6 && final_diag.avg_speed < 0.3 * vel_limit) ||
        (std::isfinite(nominal_min_time) && final_dt.total > 2.0 * nominal_min_time) ||
        (init_dt.total > 1.0e-6 && final_dt.total > 1.2 * init_dt.total);
    const bool suspicious_shape = final_diag.straightness > 1.5;
    const bool has_violation =
        final_diag.vel_violations + final_diag.omg_violations + final_diag.tilt_violations +
            final_diag.thrust_violations + final_diag.yaw_rate_violations + final_diag.corridor_violations >
        0;

    RCLCPP_INFO(logger,
                "%s time | init T=%.2f len=%.2f avg_v=%.2f dt[min/mean/max]=%.2f/%.2f/%.2f | "
                "final T=%.2f len=%.2f chord=%.2f straight=%.2f avg_v=%.2f low_v=%.0f%% dt[min/mean/max]=%.2f/%.2f/%.2f",
                tag,
                init_dt.total,
                init_len,
                init_avg_v,
                init_dt.min,
                init_dt.mean,
                init_dt.max,
                final_dt.total,
                final_diag.length,
                final_diag.chord,
                final_diag.straightness,
                final_diag.avg_speed,
                100.0 * low_speed_ratio,
                final_dt.min,
                final_dt.mean,
                final_dt.max);

    RCLCPP_INFO(logger,
                "%s limits | max_v=%.2f/%.2f max_acc=%.2f max_omg=%.2f/%.2f max_tilt=%.2f/%.2f "
                "thrust=%.2f..%.2f/[%.2f,%.2f] max_yaw_rate=%.2f/%.2f corr=max %.4f | "
                "viol samples v/omg/tilt/thr/yaw/corr=%d/%d/%d/%d/%d/%d of %d",
                tag,
                final_diag.max_vel,
                vel_limit,
                final_diag.max_acc,
                final_diag.max_omg,
                omg_limit,
                final_diag.max_tilt,
                tilt_limit,
                final_diag.min_thrust,
                final_diag.max_thrust,
                thrust_min,
                thrust_max,
                final_diag.max_yaw_rate,
                config.max_yaw_rate,
                final_diag.max_corridor_violation,
                final_diag.vel_violations,
                final_diag.omg_violations,
                final_diag.tilt_violations,
                final_diag.thrust_violations,
                final_diag.yaw_rate_violations,
                final_diag.corridor_violations,
                final_diag.samples);

    RCLCPP_INFO(logger,
                "%s costs | total=%.3f jerk=%.3f time=%.3f pos=%.3f vel=%.3f omg=%.3f tilt=%.3f thrust=%.3f "
                "yaw_smooth=%.3f yaw_rate=%.3f seam=%.3f/%.3f obs=%.3f",
                tag,
                opt_cost,
                snapshot.costEnergy,
                snapshot.costTime,
                snapshot.costPos,
                snapshot.costVel,
                snapshot.costOmg,
                snapshot.costTheta,
                snapshot.costThrust,
                snapshot.costYawSmooth,
                snapshot.costYawRateLimit,
                snapshot.costSeamPos,
                snapshot.costSeamYaw,
                snapshot.costObservationYaw);

    if (suspicious_slow || suspicious_shape || has_violation)
    {
        RCLCPP_WARN(logger,
                    "%s suspect | slow=%d shape=%d violation=%d | slowest_pieces: %s",
                    tag,
                    static_cast<int>(suspicious_slow),
                    static_cast<int>(suspicious_shape),
                    static_cast<int>(has_violation),
                    slowPiecesToStr(final_diag.slow_pieces).c_str());
    }
    // SFC excursions are useful optimizer diagnostics, but MINCO trajectories
    // are not guaranteed to stay inside the constructed corridor. Snapshot and
    // fallback decisions are handled by the inflated-map collision check.
}
} // namespace

namespace traj_optimizer
{

bool TrajOptimizer::initialize(const rclcpp::Node::SharedPtr &node_handle,
                               const std::shared_ptr<VDBMap> &map_manager)
{
    node_handle_ = node_handle;
    map_manager_ = map_manager;

    if (!node_handle_ || !map_manager_)
    {
        initialized_ = false;
        return false;
    }

    initialized_ = loadParameters();
    return initialized_;
}

bool TrajOptimizer::optimizePath(const std::vector<openvdb::Vec3d> &path_world,
                                 const BoundaryState &start_state,
                                 const BoundaryState &goal_state,
                                 double start_yaw,
                                 double goal_yaw,
                                 OptimizedPlan &plan_out,
                                 DebugInfo *debug_out) const
{
    plan_out.clear();

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto ms_since = [](const clock::time_point &from) -> double
    {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };
    const auto elapsed_ms = [&]() -> double { return ms_since(t0); };

    // Per-stage timing (ms). Any stage not reached stays at 0.
    double t_convert_ms = 0.0;
    double t_start_check_ms = 0.0;
    double t_bbox_ms = 0.0;
    double t_extract_ms = 0.0;
    double t_corridor_ms = 0.0;
    double t_setup_ms = 0.0;
    double t_optimize_ms = 0.0;

    const auto fail = [&](const char *reason) -> bool
    {
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt] failed: %s | total=%.2f ms",
                     reason, elapsed_ms());
        plan_out.clear();
        return false;
    };
    const auto make_log_config = [&]() -> OptimizerLogConfig
    {
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        return log_config;
    };

    if (!initialized_ || !node_handle_ || !map_manager_)
    {
        return fail("not initialized");
    }

    auto stage_t = clock::now();
    const std::vector<Eigen::Vector3d> path_eigen = convertPath(path_world);
    t_convert_ms = ms_since(stage_t);
    if (path_eigen.size() < 2)
    {
        return fail("path has fewer than 2 points");
    }

    stage_t = clock::now();
    int inflate_val = 0;
    const bool start_inflated =
        map_manager_->query_is_inflated_at_world(path_eigen.front(), inflate_val) && inflate_val > 0;
    t_start_check_ms = ms_since(stage_t);
    if (start_inflated)
    {
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt]",
                                 "start_inside_inflated",
                                 "",
                                 path_eigen,
                                 {},
                                 {},
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr);
        }
        return fail("path start is inside inflated region");
    }

    stage_t = clock::now();
    openvdb::CoordBBox local_bbox;
    const bool bbox_ok = computeLocalBBoxFromPath(path_eigen, local_bbox);
    t_bbox_ms = ms_since(stage_t);
    if (!bbox_ok)
    {
        return fail("failed to compute local bbox");
    }

    stage_t = clock::now();
    std::vector<Eigen::Vector3d> inflated_points;
    map_manager_->extractInflatedSurfacePointsInBox(local_bbox, inflated_points);
    const size_t raw_inflated_points = inflated_points.size();
    inflated_points = filterPointsNearPath(path_eigen, inflated_points, corridor_range_);
    const size_t filtered_center_points = inflated_points.size();
    const std::vector<Eigen::Vector3d> inflated_voxel_centers = inflated_points;
    const auto tf = map_manager_->get_grid_transform();
    const double voxel_size = tf->voxelSize()[0];
    const std::set<CoordKey> center_only_keys = findDiagonalSideVoxelKeys(path_eigen, *tf);
    std::vector<Eigen::Vector3d> center_only_voxels;
    inflated_points = expandInflatedCentersForSfc(inflated_voxel_centers,
                                                  voxel_size,
                                                  *tf,
                                                  center_only_keys,
                                                  &center_only_voxels);
    t_extract_ms = ms_since(stage_t);
    RCLCPP_DEBUG(node_handle_->get_logger(),
                 "[TrajOpt][Tube] raw_surface_pts=%zu filtered_center_pts=%zu center_only=%zu sfc_pts=%zu keep_radius=%.2f voxel=%.3f",
                 raw_inflated_points,
                 filtered_center_points,
                 center_only_voxels.size(),
                 inflated_points.size(),
                 corridor_range_,
                 voxel_size);

    stage_t = clock::now();
    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor;
    const bool corridor_ok = buildCorridor(path_eigen, local_bbox, inflated_points, corridor);
    t_corridor_ms = ms_since(stage_t);
    if (!corridor_ok)
    {
        fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt]",
                                 "corridor_construction_failed",
                                 "",
                                 path_eigen,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt] failed: corridor construction failed | total=%.2f ms | "
                     "convert=%.2f start_chk=%.2f bbox=%.2f extract=%.2f corridor=%.2f | "
                     "path_pts=%zu inflated_pts=%zu",
                     elapsed_ms(),
                     t_convert_ms, t_start_check_ms, t_bbox_ms, t_extract_ms, t_corridor_ms,
                     path_eigen.size(), inflated_points.size());
        return false;
    }

    stage_t = clock::now();
    gcopter::GCOPTER_PolytopeSFC optimizer;
    const Eigen::Matrix3d initial_pva = makeBoundaryPVA(start_state);
    const Eigen::Matrix3d terminal_pva = makeBoundaryPVA(goal_state);
    const double length_per_piece =
        (length_per_piece_ > 0.0) ? length_per_piece_ : std::numeric_limits<double>::infinity();
    const gcopter::SetupOptions setup_options;

    const bool setup_ok = optimizer.setup(time_weight_,
                                          jerk_weight_,
                                          initial_pva,
                                          terminal_pva,
                                          corridor,
                                          length_per_piece,
                                          smoothing_eps_,
                                          integral_resolution_,
                                          makeMagnitudeBounds(),
                                          makePenaltyWeights(),
                                          makePhysicalParams(),
                                          init_alloc_speed_ratio_,
                                          init_max_acc_,
                                          init_time_margin_,
                                          start_yaw,
                                          goal_yaw,
                                          yaw_smooth_weight_,
                                          max_yaw_rate_,
                                          penalty_yaw_rate_,
                                          setup_options);
    t_setup_ms = ms_since(stage_t);
    if (!setup_ok)
    {
        fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor);
        const std::string corridor_diag = diagnoseCorridorForGcopter(corridor);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt]",
                                 "setup_failed",
                                 corridor_diag.c_str(),
                                 path_eigen,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt] failed: optimizer setup failed | total=%.2f ms | "
                     "convert=%.2f start_chk=%.2f bbox=%.2f extract=%.2f corridor=%.2f setup=%.2f | "
                     "corridor_polys=%zu | diag=%s",
                     elapsed_ms(),
                     t_convert_ms, t_start_check_ms, t_bbox_ms, t_extract_ms, t_corridor_ms, t_setup_ms,
                     corridor.size(),
                     corridor_diag.c_str());
        return false;
    }

    stage_t = clock::now();
    Eigen::VectorXd yaw_knots;
    traj_optimizer::OptimizedPlan::PositionTrajectory traj;
    const double opt_cost = optimizer.optimize(traj, yaw_knots, rel_cost_tol_);
    const gcopter::DebugSnapshot debug_snapshot = optimizer.getDebugSnapshot();
    t_optimize_ms = ms_since(stage_t);
    if (!std::isfinite(opt_cost) || traj.getPieceNum() <= 0)
    {
        plan_out.clear();
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(2)
               << "total=" << elapsed_ms() << "ms convert=" << t_convert_ms
               << " start_chk=" << t_start_check_ms << " bbox=" << t_bbox_ms
               << " extract=" << t_extract_ms << " corridor=" << t_corridor_ms
               << " setup=" << t_setup_ms << " optimize=" << t_optimize_ms;
        logOptimizerDiagnostics(node_handle_->get_logger(),
                                "[TrajOpt]",
                                timing.str().c_str(),
                                path_eigen,
                                corridor,
                                start_state,
                                goal_state,
                                start_yaw,
                                goal_yaw,
                                debug_snapshot,
                                log_config,
                                opt_cost,
                                &traj);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt]",
                                 "optimize_invalid",
                                 timing.str().c_str(),
                                 path_eigen,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 debug_snapshot,
                                 log_config,
                                 opt_cost,
                                 &traj,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt] failed: optimize returned invalid trajectory | total=%.2f ms | "
                     "convert=%.2f start_chk=%.2f bbox=%.2f extract=%.2f corridor=%.2f setup=%.2f optimize=%.2f | "
                     "cost=%f pieces=%d",
                     elapsed_ms(),
                     t_convert_ms, t_start_check_ms, t_bbox_ms, t_extract_ms,
                     t_corridor_ms, t_setup_ms, t_optimize_ms,
                     opt_cost, traj.getPieceNum());
        return false;
    }

    plan_out.position_traj = traj;
    plan_out.valid = true;
    const Eigen::VectorXd durs = traj.getDurations();
    plan_out.piece_durations.assign(durs.data(), durs.data() + durs.size());
    plan_out.yaw_knots.assign(yaw_knots.data(), yaw_knots.data() + yaw_knots.size());

    fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor);

    OptimizerLogConfig log_config;
    log_config.time_weight = time_weight_;
    log_config.jerk_weight = jerk_weight_;
    log_config.length_per_piece = length_per_piece_;
    log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
    log_config.init_max_acc = init_max_acc_;
    log_config.init_time_margin = init_time_margin_;
    log_config.max_yaw_rate = max_yaw_rate_;
    log_config.magnitude_bounds = makeMagnitudeBounds();
    log_config.physical_params = makePhysicalParams();
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(2)
           << "total=" << elapsed_ms() << "ms convert=" << t_convert_ms
           << " start_chk=" << t_start_check_ms << " bbox=" << t_bbox_ms
           << " extract=" << t_extract_ms << " corridor=" << t_corridor_ms
           << " setup=" << t_setup_ms << " optimize=" << t_optimize_ms;
    logOptimizerDiagnostics(node_handle_->get_logger(),
                            "[TrajOpt]",
                            timing.str().c_str(),
                            path_eigen,
                            corridor,
                            start_state,
                            goal_state,
                            start_yaw,
                            goal_yaw,
                            debug_snapshot,
                            log_config,
                            opt_cost,
                            &traj);

    const InflatedCollisionDiag collision =
        checkTrajectoryInflatedCollision(traj, map_manager_);
    if (collision.collision)
    {
        logInflatedCollisionAndSnapshot(node_handle_->get_logger(),
                                        snapshot_dir_,
                                        snapshot_enable_,
                                        "[TrajOpt]",
                                        timing.str(),
                                        collision,
                                        path_eigen,
                                        inflated_points,
                                        corridor,
                                        start_state,
                                        goal_state,
                                        start_yaw,
                                        goal_yaw,
                                        debug_snapshot,
                                        log_config,
                                        opt_cost,
                                        traj,
                                        map_manager_,
                                        strict_traj_,
                                        &inflated_voxel_centers,
                                        &center_only_voxels);
        if (strict_traj_)
        {
            plan_out.clear();
            return false;
        }
    }

    return true;
}

bool TrajOptimizer::optimizeStitchedPath(const std::vector<openvdb::Vec3d> &path_world,
                                         size_t split_index,
                                         const BoundaryState &start_state,
                                         const BoundaryState &goal_state,
                                         double start_yaw,
                                         double split_yaw,
                                         double goal_yaw,
                                         OptimizedPlan &plan_out,
                                         DebugInfo *debug_out) const
{
    plan_out.clear();

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto ms_since = [](const clock::time_point &from) -> double
    {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };
    const auto elapsed_ms = [&]() -> double { return ms_since(t0); };

    double t_convert_ms = 0.0;
    double t_start_check_ms = 0.0;
    double t_bbox_ms = 0.0;
    double t_extract_ms = 0.0;
    double t_corridor1_ms = 0.0;
    double t_corridor2_ms = 0.0;
    double t_setup_ms = 0.0;
    double t_optimize_ms = 0.0;

    const auto fail = [&](const char *reason) -> bool
    {
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt][Stitch] failed: %s | total=%.2f ms",
                     reason, elapsed_ms());
        plan_out.clear();
        return false;
    };
    const auto make_log_config = [&]() -> OptimizerLogConfig
    {
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        return log_config;
    };

    if (!initialized_ || !node_handle_ || !map_manager_)
    {
        return fail("not initialized");
    }

    auto stage_t = clock::now();
    const std::vector<Eigen::Vector3d> path_eigen = convertPath(path_world);
    t_convert_ms = ms_since(stage_t);
    if (path_eigen.size() < 3)
    {
        return fail("path has fewer than 3 points");
    }
    if (split_index == 0 || split_index >= path_eigen.size() - 1)
    {
        return fail("split index is out of range");
    }

    stage_t = clock::now();
    int inflate_val = 0;
    const bool start_inflated =
        map_manager_->query_is_inflated_at_world(path_eigen.front(), inflate_val) && inflate_val > 0;
    t_start_check_ms = ms_since(stage_t);
    if (start_inflated)
    {
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Stitch]",
                                 "start_inside_inflated",
                                 "",
                                 path_eigen,
                                 {},
                                 {},
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr);
        }
        return fail("path start is inside inflated region");
    }

    stage_t = clock::now();
    openvdb::CoordBBox local_bbox;
    const bool bbox_ok = computeLocalBBoxFromPath(path_eigen, local_bbox);
    t_bbox_ms = ms_since(stage_t);
    if (!bbox_ok)
    {
        return fail("failed to compute local bbox");
    }

    stage_t = clock::now();
    std::vector<Eigen::Vector3d> inflated_points;
    map_manager_->extractInflatedSurfacePointsInBox(local_bbox, inflated_points);
    const size_t raw_inflated_points = inflated_points.size();
    inflated_points = filterPointsNearPath(path_eigen, inflated_points, corridor_range_);
    const size_t filtered_center_points = inflated_points.size();
    const std::vector<Eigen::Vector3d> inflated_voxel_centers = inflated_points;
    const auto tf = map_manager_->get_grid_transform();
    const double voxel_size = tf->voxelSize()[0];
    const std::set<CoordKey> center_only_keys = findDiagonalSideVoxelKeys(path_eigen, *tf);
    std::vector<Eigen::Vector3d> center_only_voxels;
    inflated_points = expandInflatedCentersForSfc(inflated_voxel_centers,
                                                  voxel_size,
                                                  *tf,
                                                  center_only_keys,
                                                  &center_only_voxels);
    t_extract_ms = ms_since(stage_t);
    RCLCPP_DEBUG(node_handle_->get_logger(),
                 "[TrajOpt][Tube] raw_surface_pts=%zu filtered_center_pts=%zu center_only=%zu sfc_pts=%zu keep_radius=%.2f voxel=%.3f",
                 raw_inflated_points,
                 filtered_center_points,
                 center_only_voxels.size(),
                 inflated_points.size(),
                 corridor_range_,
                 voxel_size);

    std::vector<Eigen::Vector3d> seg1(path_eigen.begin(), path_eigen.begin() + split_index + 1);
    std::vector<Eigen::Vector3d> seg2(path_eigen.begin() + split_index, path_eigen.end());

    stage_t = clock::now();
    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor1;
    const bool corridor1_ok = buildCorridor(seg1, local_bbox, inflated_points, corridor1);
    t_corridor1_ms = ms_since(stage_t);
    if (!corridor1_ok)
    {
        fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor1);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Stitch]",
                                 "corridor1_construction_failed",
                                 "",
                                 path_eigen,
                                 inflated_points,
                                 corridor1,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        return fail("first stitched corridor construction failed");
    }

    stage_t = clock::now();
    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor2;
    const bool corridor2_ok = buildCorridor(seg2, local_bbox, inflated_points, corridor2);
    t_corridor2_ms = ms_since(stage_t);
    if (!corridor2_ok)
    {
        fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor1);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Stitch]",
                                 "corridor2_construction_failed",
                                 "",
                                 path_eigen,
                                 inflated_points,
                                 corridor1,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        return fail("second stitched corridor construction failed");
    }

    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor = corridor1;
    corridor.insert(corridor.end(), corridor2.begin(), corridor2.end());

    stage_t = clock::now();
    gcopter::GCOPTER_PolytopeSFC optimizer;
    const Eigen::Matrix3d initial_pva = makeBoundaryPVA(start_state);
    const Eigen::Matrix3d terminal_pva = makeBoundaryPVA(goal_state);
    const double length_per_piece =
        (length_per_piece_ > 0.0) ? length_per_piece_ : std::numeric_limits<double>::infinity();

    gcopter::SetupOptions setup_options;
    setup_options.seamAnchor.enabled = true;
    setup_options.seamAnchor.splitPolyIndex = static_cast<int>(corridor1.size());
    setup_options.seamAnchor.position = path_eigen[split_index];
    setup_options.seamAnchor.yaw = split_yaw;
    setup_options.seamAnchor.positionWeight = stitched_seam_pos_weight_;
    setup_options.seamAnchor.yawWeight = stitched_seam_yaw_weight_;

    const bool setup_ok = optimizer.setup(time_weight_,
                                          jerk_weight_,
                                          initial_pva,
                                          terminal_pva,
                                          corridor,
                                          length_per_piece,
                                          smoothing_eps_,
                                          integral_resolution_,
                                          makeMagnitudeBounds(),
                                          makePenaltyWeights(),
                                          makePhysicalParams(),
                                          init_alloc_speed_ratio_,
                                          init_max_acc_,
                                          init_time_margin_,
                                          start_yaw,
                                          goal_yaw,
                                          yaw_smooth_weight_,
                                          max_yaw_rate_,
                                          penalty_yaw_rate_,
                                          setup_options);
    t_setup_ms = ms_since(stage_t);
    if (!setup_ok)
    {
        fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor);
        const std::string corridor_diag = diagnoseCorridorForGcopter(corridor);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Stitch]",
                                 "setup_failed",
                                 corridor_diag.c_str(),
                                 path_eigen,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt][Stitch] failed: optimizer setup failed | total=%.2f ms | "
                     "convert=%.2f start_chk=%.2f bbox=%.2f extract=%.2f corridor1=%.2f corridor2=%.2f setup=%.2f | "
                     "corridor1_polys=%zu corridor2_polys=%zu | diag=%s",
                     elapsed_ms(),
                     t_convert_ms, t_start_check_ms, t_bbox_ms, t_extract_ms,
                     t_corridor1_ms, t_corridor2_ms, t_setup_ms,
                     corridor1.size(), corridor2.size(),
                     corridor_diag.c_str());
        return false;
    }

    stage_t = clock::now();
    Eigen::VectorXd yaw_knots;
    traj_optimizer::OptimizedPlan::PositionTrajectory traj;
    const double opt_cost = optimizer.optimize(traj, yaw_knots, rel_cost_tol_);
    const gcopter::DebugSnapshot debug_snapshot = optimizer.getDebugSnapshot();
    t_optimize_ms = ms_since(stage_t);
    if (!std::isfinite(opt_cost) || traj.getPieceNum() <= 0)
    {
        plan_out.clear();
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(2)
               << "total=" << elapsed_ms() << "ms convert=" << t_convert_ms
               << " start_chk=" << t_start_check_ms << " bbox=" << t_bbox_ms
               << " extract=" << t_extract_ms << " corridor1=" << t_corridor1_ms
               << " corridor2=" << t_corridor2_ms << " setup=" << t_setup_ms
               << " optimize=" << t_optimize_ms;
        logOptimizerDiagnostics(node_handle_->get_logger(),
                                "[TrajOpt][Stitch]",
                                timing.str().c_str(),
                                path_eigen,
                                corridor,
                                start_state,
                                goal_state,
                                start_yaw,
                                goal_yaw,
                                debug_snapshot,
                                log_config,
                                opt_cost,
                                &traj);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Stitch]",
                                 "optimize_invalid",
                                 timing.str().c_str(),
                                 path_eigen,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 debug_snapshot,
                                 log_config,
                                 opt_cost,
                                 &traj,
                                 map_manager_,
                                 &inflated_voxel_centers,
                                 &center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt][Stitch] failed: optimize returned invalid trajectory | total=%.2f ms | "
                     "convert=%.2f start_chk=%.2f bbox=%.2f extract=%.2f corridor1=%.2f corridor2=%.2f setup=%.2f optimize=%.2f | "
                     "cost=%f pieces=%d",
                     elapsed_ms(),
                     t_convert_ms, t_start_check_ms, t_bbox_ms, t_extract_ms,
                     t_corridor1_ms, t_corridor2_ms, t_setup_ms, t_optimize_ms,
                     opt_cost, traj.getPieceNum());
        return false;
    }

    plan_out.position_traj = traj;
    plan_out.valid = true;
    const Eigen::VectorXd durs = traj.getDurations();
    plan_out.piece_durations.assign(durs.data(), durs.data() + durs.size());
    plan_out.yaw_knots.assign(yaw_knots.data(), yaw_knots.data() + yaw_knots.size());
    // seamPieceIndex is the *first* piece index of the post-seam segment.
    // The seam (vp1 anchor) therefore lives at the end of piece
    // (seamPieceIndex - 1), i.e. at the sum of the first seamPieceIndex
    // piece durations. Negative / out-of-range => no usable anchor.
    if (debug_snapshot.seamPieceIndex > 0 &&
        debug_snapshot.seamPieceIndex <= static_cast<int>(plan_out.piece_durations.size()))
    {
        double anchor_t = 0.0;
        for (int i = 0; i < debug_snapshot.seamPieceIndex; ++i)
        {
            anchor_t += plan_out.piece_durations[i];
        }
        plan_out.anchor_time = anchor_t;
    }

    fillDebugInfo(debug_out, path_eigen, local_bbox, inflated_points, corridor);

    OptimizerLogConfig log_config;
    log_config.time_weight = time_weight_;
    log_config.jerk_weight = jerk_weight_;
    log_config.length_per_piece = length_per_piece_;
    log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
    log_config.init_max_acc = init_max_acc_;
    log_config.init_time_margin = init_time_margin_;
    log_config.max_yaw_rate = max_yaw_rate_;
    log_config.magnitude_bounds = makeMagnitudeBounds();
    log_config.physical_params = makePhysicalParams();
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(2)
           << "total=" << elapsed_ms() << "ms convert=" << t_convert_ms
           << " start_chk=" << t_start_check_ms << " bbox=" << t_bbox_ms
           << " extract=" << t_extract_ms << " corridor1=" << t_corridor1_ms
           << " corridor2=" << t_corridor2_ms << " setup=" << t_setup_ms
           << " optimize=" << t_optimize_ms;
    logOptimizerDiagnostics(node_handle_->get_logger(),
                            "[TrajOpt][Stitch]",
                            timing.str().c_str(),
                            path_eigen,
                            corridor,
                            start_state,
                            goal_state,
                            start_yaw,
                            goal_yaw,
                            debug_snapshot,
                            log_config,
                            opt_cost,
                            &traj);

    const InflatedCollisionDiag collision =
        checkTrajectoryInflatedCollision(traj, map_manager_);
    if (collision.collision)
    {
        logInflatedCollisionAndSnapshot(node_handle_->get_logger(),
                                        snapshot_dir_,
                                        snapshot_enable_,
                                        "[TrajOpt][Stitch]",
                                        timing.str(),
                                        collision,
                                        path_eigen,
                                        inflated_points,
                                        corridor,
                                        start_state,
                                        goal_state,
                                        start_yaw,
                                        goal_yaw,
                                        debug_snapshot,
                                        log_config,
                                        opt_cost,
                                        traj,
                                        map_manager_,
                                        strict_traj_,
                                        &inflated_voxel_centers,
                                        &center_only_voxels);
        if (strict_traj_)
        {
            plan_out.clear();
            return false;
        }
    }

    return true;
}

bool TrajOptimizer::loadParameters()
{
    auto log = node_handle_->get_logger();

    auto param_set = [&](std::string_view name, const auto &def, auto &out)
    {
        using T = std::decay_t<decltype(out)>;
        const std::string key{name};
        T val{};

        if (!node_handle_->has_parameter(key))
        {
            val = node_handle_->declare_parameter<T>(key, static_cast<T>(def));
        }
        else
        {
            node_handle_->get_parameter(key, val);
        }
        out = val;
    };

    param_set("traj_opt.corridor_progress", 1.0, corridor_progress_);
    param_set("traj_opt.corridor_range", 1.5, corridor_range_);
    param_set("traj_opt.corridor_padding_vox", 2, corridor_padding_vox_);
    param_set("traj_opt.strict_traj", false, strict_traj_);
    param_set("traj_opt.snapshot.enable", false, snapshot_enable_);
    param_set("traj_opt.snapshot.dir",
              std::string("robot/ros_ws/src/autonomy/4_global/b_planners/ffa_planner/log_debug"),
              snapshot_dir_);

    param_set("traj_opt.time_weight", 20.0, time_weight_);
    param_set("traj_opt.jerk_weight", 1.0, jerk_weight_);
    param_set("traj_opt.length_per_piece", 0.0, length_per_piece_);
    param_set("traj_opt.smoothing_eps", 1.0e-2, smoothing_eps_);
    param_set("traj_opt.integral_resolution", 16, integral_resolution_);
    param_set("traj_opt.rel_cost_tol", 1.0e-5, rel_cost_tol_);
    param_set("traj_opt.yaw_smooth_weight", 1.0, yaw_smooth_weight_);
    param_set("traj_opt.max_yaw_rate", 1.0, max_yaw_rate_);
    param_set("traj_opt.penalty_yaw_rate", 1.0e4, penalty_yaw_rate_);
    param_set("traj_opt.init_alloc_speed_ratio", 0.5, init_alloc_speed_ratio_);
    param_set("traj_opt.init_max_acc", 1.0, init_max_acc_);
    param_set("traj_opt.init_time_margin", 1.2, init_time_margin_);
    param_set("traj_opt.stitched_seam_pos_weight", 1.0e4, stitched_seam_pos_weight_);
    param_set("traj_opt.stitched_seam_yaw_weight", 1.0e4, stitched_seam_yaw_weight_);
    param_set("traj_opt.visibility_overlap_min_radius", 0.05, visibility_overlap_min_radius_);
    param_set("traj_opt.visibility_min_pieces", 2, visibility_min_pieces_);

    param_set("traj_opt.max_vel", 4.0, max_vel_);
    param_set("traj_opt.max_body_rate", 2.1, max_body_rate_);
    param_set("traj_opt.max_tilt", 1.05, max_tilt_);
    param_set("traj_opt.min_thrust", 2.0, min_thrust_);
    param_set("traj_opt.max_thrust", 12.0, max_thrust_);

    param_set("traj_opt.vehicle_mass", 0.61, vehicle_mass_);
    param_set("traj_opt.gravity", 9.8, gravity_);
    param_set("traj_opt.horiz_drag", 0.70, horiz_drag_);
    param_set("traj_opt.vert_drag", 0.80, vert_drag_);
    param_set("traj_opt.paras_drag", 0.01, paras_drag_);
    param_set("traj_opt.speed_eps", 1.0e-4, speed_eps_);

    param_set("traj_opt.penalty_pos", 1.0e4, penalty_pos_);
    param_set("traj_opt.penalty_vel", 1.0e4, penalty_vel_);
    param_set("traj_opt.penalty_omg", 1.0e4, penalty_omg_);
    param_set("traj_opt.penalty_tilt", 1.0e4, penalty_tilt_);
    param_set("traj_opt.penalty_thrust", 1.0e5, penalty_thrust_);

    if (corridor_progress_ <= 0.0 || corridor_range_ <= 0.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameters invalid: corridor_progress and corridor_range must be > 0");
        return false;
    }
    if (integral_resolution_ <= 0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: integral_resolution must be > 0");
        return false;
    }
    if (jerk_weight_ < 0.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: jerk_weight must be >= 0");
        return false;
    }
    if (init_alloc_speed_ratio_ <= 0.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: init_alloc_speed_ratio must be > 0");
        return false;
    }
    if (init_max_acc_ <= 0.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: init_max_acc must be > 0");
        return false;
    }
    if (init_time_margin_ < 1.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: init_time_margin must be >= 1");
        return false;
    }
    if (visibility_overlap_min_radius_ < 0.0)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: visibility_overlap_min_radius must be >= 0");
        return false;
    }
    if (visibility_min_pieces_ < 2)
    {
        RCLCPP_ERROR(log, "traj_optimizer parameter invalid: visibility_min_pieces must be >= 2");
        return false;
    }
    return true;
}

std::vector<Eigen::Vector3d> TrajOptimizer::convertPath(
    const std::vector<openvdb::Vec3d> &path_world) const
{
    std::vector<Eigen::Vector3d> out;
    out.reserve(path_world.size());
    for (const auto &p : path_world)
    {
        out.emplace_back(p.x(), p.y(), p.z());
    }
    return out;
}

bool TrajOptimizer::computeLocalBBoxFromPath(const std::vector<Eigen::Vector3d> &path_world,
                                             openvdb::CoordBBox &bbox_out) const
{
    const auto tf = map_manager_->get_grid_transform();
    if (!tf || path_world.empty())
    {
        return false;
    }

    const auto world_to_coord = [&](const Eigen::Vector3d &p)
    {
        const openvdb::Vec3d ijk = tf->worldToIndex(openvdb::Vec3d(p.x(), p.y(), p.z()));
        return openvdb::Coord::round(ijk);
    };

    bbox_out = openvdb::CoordBBox(world_to_coord(path_world.front()),
                                  world_to_coord(path_world.front()));
    for (const auto &p : path_world)
    {
        bbox_out.expand(world_to_coord(p));
    }

    const openvdb::Vec3d voxel_size = tf->voxelSize();
    const double voxel_size_max = std::max({std::abs(voxel_size.x()),
                                            std::abs(voxel_size.y()),
                                            std::abs(voxel_size.z()),
                                            1.0e-6});
    const int pad_vox = std::max(
        corridor_padding_vox_,
        static_cast<int>(std::ceil(corridor_range_ / voxel_size_max)));
    bbox_out.expand(openvdb::Coord(pad_vox, pad_vox, pad_vox));

    return true;
}

bool TrajOptimizer::computeChebyshevBall(const Eigen::MatrixX4d &hpoly,
                                         Eigen::Vector3d &center_out,
                                         double &radius_out) const
{
    center_out.setZero();
    radius_out = -1.0;
    if (hpoly.rows() <= 0)
    {
        return false;
    }

    const int m = hpoly.rows();
    Eigen::MatrixX4d A(m, 4);
    Eigen::VectorXd b(m);
    Eigen::Vector4d c;
    Eigen::Vector4d x;
    const Eigen::ArrayXd h_norm = hpoly.leftCols<3>().rowwise().norm();
    if ((h_norm <= 1.0e-12).any())
    {
        return false;
    }
    A.leftCols<3>() = hpoly.leftCols<3>().array().colwise() / h_norm;
    A.rightCols<1>().setConstant(1.0);
    b = -hpoly.rightCols<1>().array() / h_norm;
    c.setZero();
    c(3) = -1.0;

    const double minmaxsd = sdlp::linprog<4>(c, A, b, x);
    if (!(minmaxsd < 0.0) || std::isinf(minmaxsd))
    {
        return false;
    }

    center_out = x.head<3>();
    radius_out = x(3);
    return radius_out > 1.0e-6;
}

bool TrajOptimizer::checkAdjacentOverlapThickness(const Eigen::MatrixX4d &a,
                                                  const Eigen::MatrixX4d &b,
                                                  double &radius_out,
                                                  double min_radius) const
{
    radius_out = -1.0;
    if (a.rows() <= 0 || b.rows() <= 0)
    {
        return false;
    }

    Eigen::MatrixX4d overlap(a.rows() + b.rows(), 4);
    overlap.topRows(a.rows()) = a;
    overlap.bottomRows(b.rows()) = b;

    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    if (!computeChebyshevBall(overlap, center, radius_out))
    {
        return false;
    }

    if (min_radius < 0.0)
    {
        min_radius = visibility_overlap_min_radius_;
    }
    return radius_out >= min_radius;
}

bool TrajOptimizer::buildMotionCorridor(const std::vector<Eigen::Vector3d> &path_world,
                                        const openvdb::CoordBBox &bbox,
                                        const std::vector<Eigen::Vector3d> &inflated_points,
                                        std::vector<CorridorCell> &cells_out,
                                        bool do_shortcut) const
{
    cells_out.clear();
    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor_out;
    if (!buildCorridor(path_world, bbox, inflated_points, corridor_out))
    {
        return false;
    }

    if (!do_shortcut)
    {
        // buildCorridor currently always shortcuts; keep the switch in the
        // interface so future motion blocks can bypass it without changing
        // the higher-level API.
    }

    cells_out.reserve(corridor_out.size());
    for (const auto &poly : corridor_out)
    {
        CorridorCell cell;
        cell.hpoly = poly;
        cell.type = CorridorCellType::MOTION;
        cell.locked = false;
        cell.min_pieces = 1;
        cells_out.push_back(std::move(cell));
    }

    return !cells_out.empty();
}

bool TrajOptimizer::buildConnectorCell(const Eigen::Vector3d &joint_world,
                                       const openvdb::CoordBBox &bbox,
                                       const std::vector<Eigen::Vector3d> &inflated_points,
                                       CorridorCell &cell_out) const
{
    cell_out = CorridorCell{};

    const auto tf = map_manager_ ? map_manager_->get_grid_transform() : nullptr;
    if (!tf)
    {
        return false;
    }

    const openvdb::Vec3d voxel_size = tf->voxelSize();
    const Eigen::Vector3d half_voxel(0.5 * voxel_size.x(),
                                     0.5 * voxel_size.y(),
                                     0.5 * voxel_size.z());

    const openvdb::Vec3d wmin = tf->indexToWorld(bbox.min());
    const openvdb::Vec3d wmax = tf->indexToWorld(bbox.max());
    const Eigen::Vector3d low_corner(wmin.x(), wmin.y(), wmin.z());
    const Eigen::Vector3d high_corner(wmax.x(), wmax.y(), wmax.z());

    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
    bd(0, 0) = 1.0;
    bd(1, 0) = -1.0;
    bd(2, 1) = 1.0;
    bd(3, 1) = -1.0;
    bd(4, 2) = 1.0;
    bd(5, 2) = -1.0;

    const Eigen::Vector3d lo = (joint_world - Eigen::Vector3d::Constant(corridor_range_))
                                   .cwiseMax(low_corner - half_voxel);
    const Eigen::Vector3d hi = (joint_world + Eigen::Vector3d::Constant(corridor_range_))
                                   .cwiseMin(high_corner + half_voxel);
    if ((hi.array() <= lo.array()).any())
    {
        return false;
    }

    bd(0, 3) = -hi.x();
    bd(1, 3) = lo.x();
    bd(2, 3) = -hi.y();
    bd(3, 3) = lo.y();
    bd(4, 3) = -hi.z();
    bd(5, 3) = lo.z();

    std::vector<Eigen::Vector3d> valid_pc;
    valid_pc.reserve(inflated_points.size());
    for (const auto &p : inflated_points)
    {
        if ((bd.leftCols<3>() * p + bd.rightCols<1>()).maxCoeff() < 0.0)
        {
            valid_pc.push_back(p);
        }
    }

    Eigen::Matrix3Xd pc(3, static_cast<Eigen::Index>(valid_pc.size()));
    for (size_t i = 0; i < valid_pc.size(); ++i)
    {
        pc.col(static_cast<Eigen::Index>(i)) = valid_pc[i];
    }

    Eigen::MatrixX4d gap;
    if (!firi::firi(bd, pc, joint_world, joint_world, gap, 1, 1.0e-6))
    {
        return false;
    }
    if (gap.rows() <= 0)
    {
        return false;
    }

    cell_out.hpoly = gap;
    cell_out.type = CorridorCellType::MOTION;
    cell_out.locked = false;
    cell_out.min_pieces = 1;
    return true;
}

bool TrajOptimizer::optimizeCorridorSequence(const std::vector<Eigen::Vector3d> &full_path_world,
                                             const openvdb::CoordBBox &bbox,
                                             const std::vector<Eigen::Vector3d> &inflated_points,
                                             const CorridorSequence &corridor_seq,
                                             const BoundaryState &start_state,
                                             const BoundaryState &goal_state,
                                             double start_yaw,
                                             double goal_yaw,
                                             const ObservationHint &observation_hint,
                                             OptimizedPlan &plan_out,
                                             DebugInfo *debug_out,
                                             const std::vector<Eigen::Vector3d> *inflated_voxel_centers,
                                             const std::vector<Eigen::Vector3d> *center_only_voxels) const
{
    plan_out.clear();
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto ms_since = [](const clock::time_point &from) -> double
    {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };
    const auto elapsed_ms = [&]() -> double { return ms_since(t0); };
    double t_setup_ms = 0.0;
    double t_optimize_ms = 0.0;
    const auto make_log_config = [&]() -> OptimizerLogConfig
    {
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        return log_config;
    };

    if (!initialized_ || !node_handle_ || !map_manager_ || corridor_seq.cells.empty())
    {
        return false;
    }

    const PolyhedraH corridor = flattenCorridorSequence(corridor_seq);
    gcopter::GCOPTER_PolytopeSFC optimizer;
    const Eigen::Matrix3d initial_pva = makeBoundaryPVA(start_state);
    const Eigen::Matrix3d terminal_pva = makeBoundaryPVA(goal_state);
    const double length_per_piece =
        (length_per_piece_ > 0.0) ? length_per_piece_ : std::numeric_limits<double>::infinity();

    gcopter::SetupOptions setup_options;
    setup_options.cellMeta.reserve(corridor_seq.cells.size());
    for (const auto &cell : corridor_seq.cells)
    {
        gcopter::CorridorCellMeta meta;
        meta.type = (cell.type == CorridorCellType::VISIBILITY)
                        ? gcopter::CellType::VISIBILITY
                        : gcopter::CellType::MOTION;
        meta.locked = cell.locked;
        meta.minPieces = std::max(cell.min_pieces, 1);
        setup_options.cellMeta.push_back(meta);
    }
    setup_options.observationCellIndex = observation_hint.cell_index;
    setup_options.observationYaw = observation_hint.yaw_ref;
    setup_options.observationYawWeight = observation_hint.yaw_weight;

    auto stage_t = clock::now();
    const bool setup_ok = optimizer.setup(time_weight_,
                                          jerk_weight_,
                                          initial_pva,
                                          terminal_pva,
                                          corridor,
                                          length_per_piece,
                                          smoothing_eps_,
                                          integral_resolution_,
                                          makeMagnitudeBounds(),
                                          makePenaltyWeights(),
                                          makePhysicalParams(),
                                          init_alloc_speed_ratio_,
                                          init_max_acc_,
                                          init_time_margin_,
                                          start_yaw,
                                          goal_yaw,
                                          yaw_smooth_weight_,
                                          max_yaw_rate_,
                                          penalty_yaw_rate_,
                                          setup_options);
    t_setup_ms = ms_since(stage_t);
    if (!setup_ok)
    {
        fillDebugInfo(debug_out, full_path_world, bbox, inflated_points, corridor_seq);
        const std::string corridor_diag = diagnoseCorridorForGcopter(corridor);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Seq]",
                                 "setup_failed",
                                 corridor_diag.c_str(),
                                 full_path_world,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 gcopter::DebugSnapshot{},
                                 make_log_config(),
                                 std::numeric_limits<double>::infinity(),
                                 nullptr);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt][Seq] failed: optimizer setup failed | corridor_polys=%zu | diag=%s",
                     corridor.size(),
                     corridor_diag.c_str());
        return false;
    }

    stage_t = clock::now();
    Eigen::VectorXd yaw_knots;
    traj_optimizer::OptimizedPlan::PositionTrajectory traj;
    const double opt_cost = optimizer.optimize(traj, yaw_knots, rel_cost_tol_);
    const gcopter::DebugSnapshot debug_snapshot = optimizer.getDebugSnapshot();
    t_optimize_ms = ms_since(stage_t);
    if (!std::isfinite(opt_cost) || traj.getPieceNum() <= 0)
    {
        fillDebugInfo(debug_out, full_path_world, bbox, inflated_points, corridor_seq);
        if (debug_out)
        {
            debug_out->piece_idx.assign(debug_snapshot.pieceIdx.data(),
                                        debug_snapshot.pieceIdx.data() + debug_snapshot.pieceIdx.size());
            debug_out->observation_cell_index = debug_snapshot.observationCellIndex;
            debug_out->observation_knot_index = debug_snapshot.observationKnotIndex;
        }
        OptimizerLogConfig log_config;
        log_config.time_weight = time_weight_;
        log_config.jerk_weight = jerk_weight_;
        log_config.length_per_piece = length_per_piece_;
        log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
        log_config.init_max_acc = init_max_acc_;
        log_config.init_time_margin = init_time_margin_;
        log_config.max_yaw_rate = max_yaw_rate_;
        log_config.magnitude_bounds = makeMagnitudeBounds();
        log_config.physical_params = makePhysicalParams();
        std::ostringstream timing;
        timing << std::fixed << std::setprecision(2)
               << "total=" << elapsed_ms() << "ms setup=" << t_setup_ms
               << " optimize=" << t_optimize_ms;
        logOptimizerDiagnostics(node_handle_->get_logger(),
                                "[TrajOpt][Seq]",
                                timing.str().c_str(),
                                full_path_world,
                                corridor,
                                start_state,
                                goal_state,
                                start_yaw,
                                goal_yaw,
                                debug_snapshot,
                                log_config,
                                opt_cost,
                                &traj);
        if (snapshot_enable_)
        {
            writeTrajOptSnapshot(node_handle_->get_logger(),
                                 snapshot_dir_,
                                 "[TrajOpt][Seq]",
                                 "optimize_invalid",
                                 timing.str().c_str(),
                                 full_path_world,
                                 inflated_points,
                                 corridor,
                                 start_state,
                                 goal_state,
                                 start_yaw,
                                 goal_yaw,
                                 debug_snapshot,
                                 log_config,
                                 opt_cost,
                                 &traj,
                                 map_manager_,
                                 inflated_voxel_centers,
                                 center_only_voxels);
        }
        RCLCPP_ERROR(node_handle_->get_logger(),
                     "[TrajOpt][Seq] failed: optimize returned invalid trajectory | cost=%f pieces=%d",
                     opt_cost,
                     traj.getPieceNum());
        return false;
    }

    plan_out.position_traj = traj;
    plan_out.valid = true;
    const Eigen::VectorXd durs = traj.getDurations();
    plan_out.piece_durations.assign(durs.data(), durs.data() + durs.size());
    plan_out.yaw_knots.assign(yaw_knots.data(), yaw_knots.data() + yaw_knots.size());
    // observationKnotIndex is the knot index at which the observation
    // anchor (vp1 equivalent) lives. That knot sits at the end of piece
    // (observationKnotIndex - 1), i.e. at the sum of the first
    // observationKnotIndex piece durations. Negative => no anchor.
    if (debug_snapshot.observationKnotIndex > 0 &&
        debug_snapshot.observationKnotIndex <= static_cast<int>(plan_out.piece_durations.size()))
    {
        double anchor_t = 0.0;
        for (int i = 0; i < debug_snapshot.observationKnotIndex; ++i)
        {
            anchor_t += plan_out.piece_durations[i];
        }
        plan_out.anchor_time = anchor_t;
    }

    fillDebugInfo(debug_out, full_path_world, bbox, inflated_points, corridor_seq);
    if (debug_out)
    {
        debug_out->piece_idx.assign(debug_snapshot.pieceIdx.data(),
                                    debug_snapshot.pieceIdx.data() + debug_snapshot.pieceIdx.size());
        debug_out->observation_cell_index = debug_snapshot.observationCellIndex;
        debug_out->observation_knot_index = debug_snapshot.observationKnotIndex;
    }

    OptimizerLogConfig log_config;
    log_config.time_weight = time_weight_;
    log_config.jerk_weight = jerk_weight_;
    log_config.length_per_piece = length_per_piece_;
    log_config.init_alloc_speed_ratio = init_alloc_speed_ratio_;
    log_config.init_max_acc = init_max_acc_;
    log_config.init_time_margin = init_time_margin_;
    log_config.max_yaw_rate = max_yaw_rate_;
    log_config.magnitude_bounds = makeMagnitudeBounds();
    log_config.physical_params = makePhysicalParams();
    std::ostringstream timing;
    timing << std::fixed << std::setprecision(2)
           << "total=" << elapsed_ms() << "ms setup=" << t_setup_ms
           << " optimize=" << t_optimize_ms;
    logOptimizerDiagnostics(node_handle_->get_logger(),
                            "[TrajOpt][Seq]",
                            timing.str().c_str(),
                            full_path_world,
                            corridor,
                            start_state,
                            goal_state,
                            start_yaw,
                            goal_yaw,
                            debug_snapshot,
                            log_config,
                            opt_cost,
                            &traj);

    const InflatedCollisionDiag collision =
        checkTrajectoryInflatedCollision(traj, map_manager_);
    if (collision.collision)
    {
        logInflatedCollisionAndSnapshot(node_handle_->get_logger(),
                                        snapshot_dir_,
                                        snapshot_enable_,
                                        "[TrajOpt][Seq]",
                                        timing.str(),
                                        collision,
                                        full_path_world,
                                        inflated_points,
                                        corridor,
                                        start_state,
                                        goal_state,
                                        start_yaw,
                                        goal_yaw,
                                        debug_snapshot,
                                        log_config,
                                        opt_cost,
                                        traj,
                                        map_manager_,
                                        strict_traj_,
                                        inflated_voxel_centers,
                                        center_only_voxels);
        if (strict_traj_)
        {
            plan_out.clear();
            return false;
        }
    }
    return true;
}

bool TrajOptimizer::optimizeThroughVisibilityCell(const std::vector<openvdb::Vec3d> &path_in_world,
                                                  const VisibilityCell &visibility_cell,
                                                  const std::vector<openvdb::Vec3d> &path_out_world,
                                                  const BoundaryState &start_state,
                                                  const BoundaryState &goal_state,
                                                  double start_yaw,
                                                  double goal_yaw,
                                                  const ObservationHint &observation_hint,
                                                  OptimizedPlan &plan_out,
                                                  DebugInfo *debug_out) const
{
    plan_out.clear();
    if (!initialized_ || !node_handle_ || !map_manager_ || !visibility_cell.valid)
    {
        return false;
    }

    const std::vector<Eigen::Vector3d> path_in = convertPath(path_in_world);
    const std::vector<Eigen::Vector3d> path_out = convertPath(path_out_world);
    if (path_in.size() < 2 || path_out.size() < 2)
    {
        return false;
    }

    std::vector<Eigen::Vector3d> full_path = path_in;
    if (!path_out.empty())
    {
        full_path.insert(full_path.end(), path_out.begin() + 1, path_out.end());
    }

    openvdb::CoordBBox local_bbox;
    if (!computeLocalBBoxFromPath(full_path, local_bbox))
    {
        return false;
    }

    std::vector<Eigen::Vector3d> inflated_points;
    map_manager_->extractInflatedSurfacePointsInBox(local_bbox, inflated_points);
    // The connector cells inflate FIRI boxes centered on the visibility-cell
    // seed, which may sit off the motion path; include it in the filter
    // polyline so their obstacle points survive the pre-cull.
    std::vector<Eigen::Vector3d> filter_path = full_path;
    filter_path.push_back(visibility_cell.center);
    inflated_points = filterPointsNearPath(filter_path, inflated_points, corridor_range_);
    const std::vector<Eigen::Vector3d> inflated_voxel_centers = inflated_points;
    const auto tf = map_manager_->get_grid_transform();
    const double voxel_size = tf->voxelSize()[0];
    const std::set<CoordKey> center_only_keys = findDiagonalSideVoxelKeys(full_path, *tf);
    std::vector<Eigen::Vector3d> center_only_voxels;
    inflated_points = expandInflatedCentersForSfc(inflated_voxel_centers,
                                                  voxel_size,
                                                  *tf,
                                                  center_only_keys,
                                                  &center_only_voxels);

    std::vector<CorridorCell> in_cells;
    if (!buildMotionCorridor(path_in, local_bbox, inflated_points, in_cells, true))
    {
        fillDebugInfo(debug_out, full_path, local_bbox, inflated_points, CorridorSequence{});
        return false;
    }

    std::vector<CorridorCell> out_cells;
    if (!buildMotionCorridor(path_out, local_bbox, inflated_points, out_cells, true))
    {
        fillDebugInfo(debug_out, full_path, local_bbox, inflated_points, CorridorSequence{});
        return false;
    }

    double entry_radius = -1.0;
    double exit_radius = -1.0;
    if (in_cells.empty() || out_cells.empty())
    {
        fillDebugInfo(debug_out, full_path, local_bbox, inflated_points, CorridorSequence{});
        return false;
    }

    // Small (box-tier) cells compete with the zero-radius point-seam: any
    // solid intersection beats it, so the comfort threshold scales down to a
    // numerical floor instead of rejecting a buildable cell.
    const double overlap_min =
        (visibility_cell.chebyshev_radius < 2.0 * visibility_overlap_min_radius_)
            ? std::min(visibility_overlap_min_radius_, 0.01)
            : visibility_overlap_min_radius_;
    bool entry_ok = checkAdjacentOverlapThickness(in_cells.back().hpoly, visibility_cell.hpoly, entry_radius, overlap_min);
    bool exit_ok = checkAdjacentOverlapThickness(visibility_cell.hpoly, out_cells.front().hpoly, exit_radius, overlap_min);

    std::optional<CorridorCell> entry_connector;
    std::optional<CorridorCell> exit_connector;

    if (!entry_ok)
    {
        CorridorCell connector;
        double motion_gap_radius = -1.0;
        double vis_gap_radius = -1.0;
        if (buildConnectorCell(visibility_cell.center, local_bbox, inflated_points, connector) &&
            checkAdjacentOverlapThickness(in_cells.back().hpoly, connector.hpoly, motion_gap_radius, overlap_min) &&
            checkAdjacentOverlapThickness(connector.hpoly, visibility_cell.hpoly, vis_gap_radius, overlap_min))
        {
            entry_connector = std::move(connector);
            entry_ok = true;
            entry_radius = std::min(motion_gap_radius, vis_gap_radius);
        }
    }

    if (!exit_ok)
    {
        CorridorCell connector;
        double vis_gap_radius = -1.0;
        double motion_gap_radius = -1.0;
        if (buildConnectorCell(visibility_cell.center, local_bbox, inflated_points, connector) &&
            checkAdjacentOverlapThickness(visibility_cell.hpoly, connector.hpoly, vis_gap_radius, overlap_min) &&
            checkAdjacentOverlapThickness(connector.hpoly, out_cells.front().hpoly, motion_gap_radius, overlap_min))
        {
            exit_connector = std::move(connector);
            exit_ok = true;
            exit_radius = std::min(vis_gap_radius, motion_gap_radius);
        }
    }

    CorridorSequence corridor_seq;
    corridor_seq.cells.reserve(in_cells.size() +
                               (entry_connector ? 1 : 0) +
                               1 +
                               (exit_connector ? 1 : 0) +
                               out_cells.size());
    corridor_seq.cells.insert(corridor_seq.cells.end(), in_cells.begin(), in_cells.end());
    if (entry_connector)
    {
        corridor_seq.cells.push_back(*entry_connector);
    }

    CorridorCell visibility_seq_cell;
    visibility_seq_cell.hpoly = visibility_cell.hpoly;
    visibility_seq_cell.type = CorridorCellType::VISIBILITY;
    visibility_seq_cell.locked = true;
    visibility_seq_cell.min_pieces = std::max(visibility_min_pieces_, 2);
    corridor_seq.cells.push_back(visibility_seq_cell);

    if (exit_connector)
    {
        corridor_seq.cells.push_back(*exit_connector);
    }
    corridor_seq.cells.insert(corridor_seq.cells.end(), out_cells.begin(), out_cells.end());

    if (!entry_ok || !exit_ok)
    {
        fillDebugInfo(debug_out, full_path, local_bbox, inflated_points, corridor_seq);
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[TrajOpt][Seq] visibility-cell overlap too thin after connector attempt | entry_r=%.3f exit_r=%.3f min=%.3f entry_conn=%d exit_conn=%d",
                    entry_radius,
                    exit_radius,
                    visibility_overlap_min_radius_,
                    static_cast<int>(entry_connector.has_value()),
                    static_cast<int>(exit_connector.has_value()));
        return false;
    }

    ObservationHint resolved_hint = observation_hint;
    if (resolved_hint.cell_index < 0)
    {
        resolved_hint.cell_index =
            static_cast<int>(in_cells.size() + (entry_connector ? 1 : 0));
    }

    return optimizeCorridorSequence(full_path,
                                    local_bbox,
                                    inflated_points,
                                    corridor_seq,
                                    start_state,
                                    goal_state,
                                    start_yaw,
                                    goal_yaw,
                                    resolved_hint,
                                    plan_out,
                                    debug_out,
                                    &inflated_voxel_centers,
                                    &center_only_voxels);
}

bool TrajOptimizer::buildCorridor(const std::vector<Eigen::Vector3d> &path_world,
                                  const openvdb::CoordBBox &bbox,
                                  const std::vector<Eigen::Vector3d> &inflated_points,
                                  gcopter::GCOPTER_PolytopeSFC::PolyhedraH &corridor_out) const
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const auto ms_since = [](const clock::time_point &from) -> double
    {
        return std::chrono::duration<double, std::milli>(clock::now() - from).count();
    };

    const auto tf = map_manager_->get_grid_transform();
    if (!tf)
    {
        return false;
    }

    const openvdb::Vec3d voxel_size = tf->voxelSize();
    const Eigen::Vector3d half_voxel(0.5 * voxel_size.x(),
                                     0.5 * voxel_size.y(),
                                     0.5 * voxel_size.z());

    const openvdb::Vec3d wmin = tf->indexToWorld(bbox.min());
    const openvdb::Vec3d wmax = tf->indexToWorld(bbox.max());
    const Eigen::Vector3d low_corner(wmin.x(), wmin.y(), wmin.z());
    const Eigen::Vector3d high_corner(wmax.x(), wmax.y(), wmax.z());

    auto stage_t = clock::now();
    sfc_gen::convexCover(path_world,
                         inflated_points,
                         low_corner - half_voxel,
                         high_corner + half_voxel,
                         corridor_progress_,
                         corridor_range_,
                         corridor_out,
                         1.0e-6);
    const double convex_cover_ms = ms_since(stage_t);
    const size_t corridor_before_shortcut = corridor_out.size();

    if (corridor_out.empty())
    {
        RCLCPP_WARN(node_handle_->get_logger(),
                    "[TrajOpt][SFC] convex cover empty | total=%.2f ms convex=%.2f | "
                    "path_pts=%zu inflated_pts=%zu bbox_min=(%d,%d,%d) bbox_max=(%d,%d,%d)",
                    ms_since(t0),
                    convex_cover_ms,
                    path_world.size(),
                    inflated_points.size(),
                    bbox.min().x(), bbox.min().y(), bbox.min().z(),
                    bbox.max().x(), bbox.max().y(), bbox.max().z());
        return false;
    }

    stage_t = clock::now();
    sfc_gen::shortCut(corridor_out);
    const double shortcut_ms = ms_since(stage_t);
    const bool ok = !corridor_out.empty();

    RCLCPP_DEBUG(node_handle_->get_logger(),
                 "[TrajOpt][SFC] total=%.2f ms | convex=%.2f shortcut=%.2f | "
                 "path_pts=%zu inflated_pts=%zu corridor_before=%zu corridor_after=%zu",
                 ms_since(t0),
                 convex_cover_ms,
                 shortcut_ms,
                 path_world.size(),
                 inflated_points.size(),
                 corridor_before_shortcut,
                 corridor_out.size());

    return ok;
}

Eigen::Matrix3d TrajOptimizer::makeBoundaryPVA(const BoundaryState &state) const
{
    Eigen::Matrix3d pva;
    pva.col(0) = state.pos;
    pva.col(1) = state.vel;
    pva.col(2) = state.acc;
    return pva;
}

Eigen::VectorXd TrajOptimizer::makeMagnitudeBounds() const
{
    Eigen::VectorXd out(5);
    out << max_vel_, max_body_rate_, max_tilt_, min_thrust_, max_thrust_;
    return out;
}

Eigen::VectorXd TrajOptimizer::makePenaltyWeights() const
{
    Eigen::VectorXd out(5);
    out << penalty_pos_, penalty_vel_, penalty_omg_, penalty_tilt_, penalty_thrust_;
    return out;
}

Eigen::VectorXd TrajOptimizer::makePhysicalParams() const
{
    Eigen::VectorXd out(6);
    out << vehicle_mass_, gravity_, horiz_drag_, vert_drag_, paras_drag_, speed_eps_;
    return out;
}

} // namespace traj_optimizer
