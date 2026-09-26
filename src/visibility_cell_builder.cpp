// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/visibility_cell_builder.hpp"

#include "traj_optimizer/firi.hpp"
#include "traj_optimizer/geo_utils.hpp"
#include "traj_optimizer/sdlp.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <set>

namespace
{
double targetFacingYaw(const openvdb::Vec3d &body_world,
                       const openvdb::Vec3d &target_world,
                       double body_sensor_yaw)
{
    const double dx = target_world.x() - body_world.x();
    const double dy = target_world.y() - body_world.y();
    if (dx * dx + dy * dy <= 1.0e-12)
    {
        return 0.0;
    }
    return std::atan2(dy, dx) - body_sensor_yaw;
}

std::vector<Eigen::Vector3d> expandVoxelCentersToCorners(const std::vector<Eigen::Vector3d> &points,
                                                         double voxel_size)
{
    if (points.empty() || voxel_size <= 0.0)
    {
        return points;
    }

    const double half = 0.5 * voxel_size;
    const std::array<Eigen::Vector3d, 8> offsets = {
        Eigen::Vector3d(-half, -half, -half),
        Eigen::Vector3d(-half, -half, +half),
        Eigen::Vector3d(-half, +half, -half),
        Eigen::Vector3d(-half, +half, +half),
        Eigen::Vector3d(+half, -half, -half),
        Eigen::Vector3d(+half, -half, +half),
        Eigen::Vector3d(+half, +half, -half),
        Eigen::Vector3d(+half, +half, +half),
    };

    std::vector<Eigen::Vector3d> expanded;
    expanded.reserve(points.size() * offsets.size());
    for (const auto &p : points)
    {
        for (const auto &offset : offsets)
        {
            expanded.push_back(p + offset);
        }
    }
    return expanded;
}

bool computeChebyshevBall(const Eigen::MatrixX4d &hpoly,
                          Eigen::Vector3d &center_out,
                          double &radius_out)
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
} // namespace

VisibilityCellBuilder::VisibilityCellBuilder(std::shared_ptr<VDBMap> map_manager,
                                             Config config,
                                             ClearableFn clearable_fn,
                                             rclcpp::Logger logger)
    : map_manager_(std::move(map_manager)),
      config_(config),
      clearable_fn_(std::move(clearable_fn)),
      logger_(std::move(logger))
{
}

bool VisibilityCellBuilder::build(const openvdb::Vec3d &witness_viewpoint,
                                  const openvdb::Vec3d &target_world,
                                  traj_optimizer::VisibilityCell &cell_out) const
{
    cell_out = traj_optimizer::VisibilityCell{};

    if (!map_manager_ || !clearable_fn_)
    {
        return false;
    }

    const auto tf = map_manager_->get_grid_transform();
    if (!tf)
    {
        return false;
    }

    // Search, preview windows, and runtime arrival checks all certify under
    // the STANDARD profile; the former TIGHT probe built the cell under a
    // stricter predicate no downstream consumer requires, thinning the
    // certified-visible cluster exactly where it could afford to be fat.
    const FovClearProfileKind profile_kind = FovClearProfileKind::STANDARD;

    const openvdb::Coord witness_coord =
        openvdb::Coord::round(tf->worldToIndex(witness_viewpoint));
    const openvdb::Vec3d vox = tf->voxelSize();
    const int roi_xy = std::max(1, static_cast<int>(std::ceil(
                                     config_.roi_xy /
                                     std::max({std::abs(vox.x()), std::abs(vox.y()), 1.0e-6}))));
    const int roi_z = std::max(1, static_cast<int>(std::ceil(
                                    config_.roi_z /
                                    std::max(std::abs(vox.z()), 1.0e-6))));

    const openvdb::Coord min_c(witness_coord.x() - roi_xy,
                               witness_coord.y() - roi_xy,
                               witness_coord.z() - roi_z);
    const openvdb::Coord max_c(witness_coord.x() + roi_xy,
                               witness_coord.y() + roi_xy,
                               witness_coord.z() + roi_z);
    const openvdb::CoordBBox bbox(min_c, max_c);

    const int nx = bbox.max().x() - bbox.min().x() + 1;
    const int ny = bbox.max().y() - bbox.min().y() + 1;
    const int nz = bbox.max().z() - bbox.min().z() + 1;
    if (nx <= 0 || ny <= 0 || nz <= 0)
    {
        return false;
    }

    const size_t total = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
    if (total == 0)
    {
        return false;
    }

    auto to_linear = [&](const openvdb::Coord &c) -> int
    {
        return ((c.z() - bbox.min().z()) * ny + (c.y() - bbox.min().y())) * nx +
               (c.x() - bbox.min().x());
    };
    auto to_coord = [&](int idx) -> openvdb::Coord
    {
        const int x = idx % nx;
        const int yz = idx / nx;
        const int y = yz % ny;
        const int z = yz / ny;
        return openvdb::Coord(bbox.min().x() + x,
                              bbox.min().y() + y,
                              bbox.min().z() + z);
    };

    const std::array<openvdb::Coord, 6> nbrs = {
        openvdb::Coord(1, 0, 0), openvdb::Coord(-1, 0, 0),
        openvdb::Coord(0, 1, 0), openvdb::Coord(0, -1, 0),
        openvdb::Coord(0, 0, 1), openvdb::Coord(0, 0, -1)};

    auto coord_visible = [&](const openvdb::Coord &coord) -> bool
    {
        const openvdb::Vec3d body_world = tf->indexToWorld(coord);
        const Eigen::Vector3d body_eig(body_world.x(), body_world.y(), body_world.z());

        float logodds = 0.0f;
        if (!map_manager_->query_log_odds_at_world(body_eig, logodds))
        {
            return false;
        }
        if (logodds > 0.0f)
        {
            return false;
        }

        int inflate_val = 0;
        if (map_manager_->query_is_inflated_at_world(body_eig, inflate_val) && inflate_val > 0)
        {
            return false;
        }

        return clearable_fn_(body_world, target_world, profile_kind);
    };

    std::vector<uint8_t> visible(total, 0);
    for (int z = bbox.min().z(); z <= bbox.max().z(); ++z)
    {
        for (int y = bbox.min().y(); y <= bbox.max().y(); ++y)
        {
            for (int x = bbox.min().x(); x <= bbox.max().x(); ++x)
            {
                const openvdb::Coord c(x, y, z);
                visible[static_cast<size_t>(to_linear(c))] = coord_visible(c) ? 1U : 0U;
            }
        }
    }

    int witness_idx = to_linear(witness_coord);
    if (witness_idx < 0 || static_cast<size_t>(witness_idx) >= total || !visible[static_cast<size_t>(witness_idx)])
    {
        // The witness was certified at search time but the map may have
        // flickered since. Before giving up, re-seed from the nearest
        // visible voxel within a small ring — the cell (not the witness
        // pose) is what execution enforces, so any visible root targeting
        // the same tau serves.
        int reseed_idx = -1;
        for (int r = 1; r <= 2 && reseed_idx < 0; ++r)
        {
            for (int dz = -r; dz <= r && reseed_idx < 0; ++dz)
            {
                for (int dy = -r; dy <= r && reseed_idx < 0; ++dy)
                {
                    for (int dx = -r; dx <= r; ++dx)
                    {
                        if (std::max({std::abs(dx), std::abs(dy), std::abs(dz)}) != r)
                        {
                            continue;
                        }
                        const openvdb::Coord nb = witness_coord.offsetBy(dx, dy, dz);
                        if (!bbox.isInside(nb))
                        {
                            continue;
                        }
                        const int nb_idx = to_linear(nb);
                        if (nb_idx >= 0 && static_cast<size_t>(nb_idx) < total &&
                            visible[static_cast<size_t>(nb_idx)])
                        {
                            reseed_idx = nb_idx;
                            break;
                        }
                    }
                }
            }
        }
        if (reseed_idx < 0)
        {
            RCLCPP_WARN(logger_,
                        "[Local][VisCell] witness viewpoint [%.2f, %.2f, %.2f] is not locally visible under chosen profile",
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
            return false;
        }
        RCLCPP_INFO(logger_,
                    "[Local][VisCell] witness [%.2f, %.2f, %.2f] not visible (flicker); re-seeded from nearby visible voxel",
                    witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
        witness_idx = reseed_idx;
    }

    std::vector<uint8_t> component(total, 0);
    std::queue<int> bfs;
    bfs.push(witness_idx);
    component[static_cast<size_t>(witness_idx)] = 1U;
    while (!bfs.empty())
    {
        const int curr = bfs.front();
        bfs.pop();
        const openvdb::Coord c = to_coord(curr);
        for (const auto &delta : nbrs)
        {
            const openvdb::Coord nb = c + delta;
            if (!bbox.isInside(nb))
            {
                continue;
            }
            const int nb_idx = to_linear(nb);
            if (nb_idx < 0 || static_cast<size_t>(nb_idx) >= total)
            {
                continue;
            }
            if (!visible[static_cast<size_t>(nb_idx)] || component[static_cast<size_t>(nb_idx)])
            {
                continue;
            }
            component[static_cast<size_t>(nb_idx)] = 1U;
            bfs.push(nb_idx);
        }
    }

    // Erosion below only serves interior-seed selection. Blockers, the FIRI
    // bounding box, and the leakage pre-gate all use the FULL certified
    // cluster: treating the erosion shell (certified-visible voxels) as
    // obstacles taxed every face by a voxel; delta-shrink plus the exact
    // continuous validation now own the discrete-to-continuous margin.
    const std::vector<uint8_t> full_component = component;

    if (config_.erode_iters > 0)
    {
        std::vector<uint8_t> eroded = component;
        for (int iter = 0; iter < config_.erode_iters; ++iter)
        {
            std::vector<uint8_t> next = eroded;
            for (size_t idx = 0; idx < eroded.size(); ++idx)
            {
                if (!eroded[idx])
                {
                    continue;
                }
                const openvdb::Coord c = to_coord(static_cast<int>(idx));
                bool boundary = false;
                for (const auto &delta : nbrs)
                {
                    const openvdb::Coord nb = c + delta;
                    if (!bbox.isInside(nb))
                    {
                        boundary = true;
                        break;
                    }
                    const int nb_idx = to_linear(nb);
                    if (!eroded[static_cast<size_t>(nb_idx)])
                    {
                        boundary = true;
                        break;
                    }
                }
                if (boundary)
                {
                    next[idx] = 0U;
                }
            }

            if (!next[static_cast<size_t>(witness_idx)])
            {
                break;
            }
            eroded.swap(next);
        }
        if (eroded[static_cast<size_t>(witness_idx)])
        {
            component.swap(eroded);
        }
    }

    std::vector<int> dist(total, -1);
    std::queue<int> boundary_q;
    for (size_t idx = 0; idx < component.size(); ++idx)
    {
        if (!component[idx])
        {
            continue;
        }
        const openvdb::Coord c = to_coord(static_cast<int>(idx));
        bool boundary = false;
        for (const auto &delta : nbrs)
        {
            const openvdb::Coord nb = c + delta;
            if (!bbox.isInside(nb))
            {
                boundary = true;
                break;
            }
            if (!component[static_cast<size_t>(to_linear(nb))])
            {
                boundary = true;
                break;
            }
        }
        if (boundary)
        {
            dist[idx] = 0;
            boundary_q.push(static_cast<int>(idx));
        }
    }

    while (!boundary_q.empty())
    {
        const int curr = boundary_q.front();
        boundary_q.pop();
        const openvdb::Coord c = to_coord(curr);
        for (const auto &delta : nbrs)
        {
            const openvdb::Coord nb = c + delta;
            if (!bbox.isInside(nb))
            {
                continue;
            }
            const int nb_idx = to_linear(nb);
            if (!component[static_cast<size_t>(nb_idx)] || dist[static_cast<size_t>(nb_idx)] >= 0)
            {
                continue;
            }
            dist[static_cast<size_t>(nb_idx)] = dist[static_cast<size_t>(curr)] + 1;
            boundary_q.push(nb_idx);
        }
    }

    int best_idx = witness_idx;
    for (size_t idx = 0; idx < component.size(); ++idx)
    {
        if (!component[idx])
        {
            continue;
        }
        if (dist[idx] > dist[static_cast<size_t>(best_idx)])
        {
            best_idx = static_cast<int>(idx);
        }
    }

    const openvdb::Coord best_coord = to_coord(best_idx);
    const openvdb::Vec3d center_world = tf->indexToWorld(best_coord);

    // Yaw commanded at the observation knot: target-facing from the FIRI
    // seed (also published as certified_yaw). When the fixed-yaw validator
    // is wired, acceptance samples are checked under this yaw — validating
    // what will be executed rather than a per-sample ideal yaw.
    const double psi_cmd =
        targetFacingYaw(center_world, target_world, config_.body_sensor_yaw);

    const auto sample_visible_yaw = [&](const Eigen::Vector3d &q, double psi) -> bool
    {
        float logodds = 0.0f;
        if (!map_manager_->query_log_odds_at_world(q, logodds) || logodds > 0.0f)
        {
            return false;
        }
        int inflate_val = 0;
        if (map_manager_->query_is_inflated_at_world(q, inflate_val) && inflate_val > 0)
        {
            return false;
        }
        const openvdb::Vec3d q_w(q.x(), q.y(), q.z());
        if (clearable_yaw_fn_)
        {
            return clearable_yaw_fn_(q_w, target_world, profile_kind, psi);
        }
        return clearable_fn_(q_w, target_world, profile_kind);
    };
    const auto sample_visible = [&](const Eigen::Vector3d &q) -> bool
    {
        return sample_visible_yaw(q, psi_cmd);
    };

    // Memoize passed samples (mm quantization) so carve iterations only pay
    // the predicate for vertices they have not certified yet. Valid only for
    // psi_cmd-yaw samples; the box tier validates under its own yaw and
    // bypasses the memo.
    std::set<std::array<long long, 3>> passed_samples;
    const auto check_sample = [&](const Eigen::Vector3d &q) -> bool
    {
        const std::array<long long, 3> key = {
            static_cast<long long>(std::llround(q.x() * 1.0e3)),
            static_cast<long long>(std::llround(q.y() * 1.0e3)),
            static_cast<long long>(std::llround(q.z() * 1.0e3))};
        if (passed_samples.count(key) != 0)
        {
            return true;
        }
        if (!sample_visible(q))
        {
            return false;
        }
        passed_samples.insert(key);
        return true;
    };

    // Leakage pre-gate: a vertex whose containing voxel is outside the
    // certified cluster escaped the discrete certification by construction —
    // carve it without paying the ray trace.
    const auto voxel_in_component = [&](const Eigen::Vector3d &q) -> bool
    {
        const openvdb::Coord c = openvdb::Coord::round(
            tf->worldToIndex(openvdb::Vec3d(q.x(), q.y(), q.z())));
        if (!bbox.isInside(c))
        {
            return false;
        }
        const int q_idx = to_linear(c);
        if (q_idx < 0 || static_cast<size_t>(q_idx) >= total)
        {
            return false;
        }
        return full_component[static_cast<size_t>(q_idx)] != 0U;
    };

    // Box-tier fallback: when the FIRI cell cannot be certified, degrade to
    // the maximal axis-aligned box of certified-visible voxels grown from
    // the witness instead of aborting to the point-seam. Every point of such
    // a box lies inside some voxel whose center passed the observation
    // predicate — the same discretization slack the seam's exact-point
    // enforcement rests on — while giving the optimizer a native hard
    // set-membership constraint plus local freedom.
    const auto try_box_tier = [&]() -> bool
    {
        const openvdb::Coord root = to_coord(witness_idx);
        openvdb::Coord bmin = root;
        openvdb::Coord bmax = root;
        const int axis_cap = 5; // growth per direction, voxels

        const auto slab_ok = [&](const openvdb::Coord &lo, const openvdb::Coord &hi) -> bool
        {
            for (int z = lo.z(); z <= hi.z(); ++z)
            {
                for (int y = lo.y(); y <= hi.y(); ++y)
                {
                    for (int x = lo.x(); x <= hi.x(); ++x)
                    {
                        const openvdb::Coord c(x, y, z);
                        if (!bbox.isInside(c) ||
                            !full_component[static_cast<size_t>(to_linear(c))])
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        bool grew = true;
        while (grew)
        {
            grew = false;
            for (int axis = 0; axis < 3; ++axis)
            {
                for (int sgn = -1; sgn <= 1; sgn += 2)
                {
                    openvdb::Coord lo = bmin;
                    openvdb::Coord hi = bmax;
                    if (sgn < 0)
                    {
                        if (root[axis] - bmin[axis] >= axis_cap)
                        {
                            continue;
                        }
                        lo[axis] = bmin[axis] - 1;
                        hi[axis] = lo[axis];
                    }
                    else
                    {
                        if (bmax[axis] - root[axis] >= axis_cap)
                        {
                            continue;
                        }
                        lo[axis] = bmax[axis] + 1;
                        hi[axis] = lo[axis];
                    }
                    if (!slab_ok(lo, hi))
                    {
                        continue;
                    }
                    if (sgn < 0)
                    {
                        bmin[axis] -= 1;
                    }
                    else
                    {
                        bmax[axis] += 1;
                    }
                    grew = true;
                }
            }
        }

        const auto make_box_cell = [&](const Eigen::Vector3d &lo,
                                       const Eigen::Vector3d &hi,
                                       bool validate) -> bool
        {
            if (((hi - lo).array() <= 1.0e-6).any())
            {
                return false;
            }
            const Eigen::Vector3d box_center = 0.5 * (lo + hi);
            const openvdb::Vec3d box_center_w(box_center.x(), box_center.y(), box_center.z());
            const double psi_box =
                targetFacingYaw(box_center_w, target_world, config_.body_sensor_yaw);

            if (validate)
            {
                if (!sample_visible_yaw(box_center, psi_box))
                {
                    return false;
                }
                for (int ix = 0; ix < 2; ++ix)
                {
                    for (int iy = 0; iy < 2; ++iy)
                    {
                        for (int iz = 0; iz < 2; ++iz)
                        {
                            const Eigen::Vector3d v(ix ? hi.x() : lo.x(),
                                                    iy ? hi.y() : lo.y(),
                                                    iz ? hi.z() : lo.z());
                            if (!sample_visible_yaw(v, psi_box) ||
                                !sample_visible_yaw(0.5 * (v + box_center), psi_box))
                            {
                                return false;
                            }
                        }
                    }
                }
            }

            Eigen::MatrixX4d box(6, 4);
            box << 1, 0, 0, -hi.x(),
                -1, 0, 0, lo.x(),
                0, 1, 0, -hi.y(),
                0, -1, 0, lo.y(),
                0, 0, 1, -hi.z(),
                0, 0, -1, lo.z();
            cell_out.hpoly = box;
            cell_out.center = box_center;
            cell_out.target =
                Eigen::Vector3d(target_world.x(), target_world.y(), target_world.z());
            cell_out.certified_yaw = psi_box;
            cell_out.chebyshev_radius = 0.5 * (hi - lo).minCoeff();
            cell_out.valid = true;
            return true;
        };

        const double half = 0.5 * config_.voxel_size;
        const double s = std::max(config_.shrink_m, 0.0);
        const openvdb::Vec3d wmin = tf->indexToWorld(bmin);
        const openvdb::Vec3d wmax = tf->indexToWorld(bmax);
        const Eigen::Vector3d lo_full(wmin.x() - half + s, wmin.y() - half + s, wmin.z() - half + s);
        const Eigen::Vector3d hi_full(wmax.x() + half - s, wmax.y() + half - s, wmax.z() + half - s);
        if (make_box_cell(lo_full, hi_full, true))
        {
            RCLCPP_INFO(logger_,
                        "[Local][VisCell] box-tier accepted (validated %dx%dx%d vox, radius=%.2f) around witness [%.2f, %.2f, %.2f]",
                        bmax.x() - bmin.x() + 1, bmax.y() - bmin.y() + 1, bmax.z() - bmin.z() + 1,
                        cell_out.chebyshev_radius,
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
            return true;
        }

        // Last box tier: the witness voxel alone, discrete certification only
        // (its center passed the sweep-time predicate). This is the guarantee
        // level of the former point-seam, expressed as a hard constraint.
        const openvdb::Vec3d rw = tf->indexToWorld(root);
        const double s0 = std::min(0.02, 0.5 * half);
        const Eigen::Vector3d lo_root(rw.x() - half + s0, rw.y() - half + s0, rw.z() - half + s0);
        const Eigen::Vector3d hi_root(rw.x() + half - s0, rw.y() + half - s0, rw.z() + half - s0);
        if (make_box_cell(lo_root, hi_root, false))
        {
            RCLCPP_INFO(logger_,
                        "[Local][VisCell] box-tier accepted (witness voxel, discrete-certified) around witness [%.2f, %.2f, %.2f]",
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
            return true;
        }
        return false;
    };

    std::vector<Eigen::Vector3d> blocker_centers;
    blocker_centers.reserve(full_component.size());
    std::vector<uint8_t> blocker_mark(total, 0);
    for (size_t idx = 0; idx < full_component.size(); ++idx)
    {
        if (!full_component[idx])
        {
            continue;
        }
        const openvdb::Coord c = to_coord(static_cast<int>(idx));
        // 26-connected shell: under the former 6-connected collection the
        // edge/corner-diagonal outside voxels contributed no obstacle
        // points, so the convex hull could slip diagonally past the eroded
        // set by a voxel or more and fail vertex certification.
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                    {
                        continue;
                    }
                    const openvdb::Coord nb = c.offsetBy(dx, dy, dz);
                    if (!bbox.isInside(nb))
                    {
                        continue;
                    }
                    const int nb_idx = to_linear(nb);
                    if (full_component[static_cast<size_t>(nb_idx)] ||
                        blocker_mark[static_cast<size_t>(nb_idx)])
                    {
                        continue;
                    }
                    blocker_mark[static_cast<size_t>(nb_idx)] = 1U;
                    const openvdb::Vec3d w = tf->indexToWorld(nb);
                    blocker_centers.emplace_back(w.x(), w.y(), w.z());
                }
            }
        }
    }

    const std::vector<Eigen::Vector3d> blocker_points =
        expandVoxelCentersToCorners(blocker_centers, config_.voxel_size);
    Eigen::Matrix3Xd pc(3, static_cast<Eigen::Index>(blocker_points.size()));
    for (size_t i = 0; i < blocker_points.size(); ++i)
    {
        pc.col(static_cast<Eigen::Index>(i)) = blocker_points[i];
    }

    // Bound FIRI by the eroded component's AABB (+half voxel), not the ROI:
    // the ROI box let the hull balloon into space the discrete certification
    // never covered; local concavities are handled by the blocker shell.
    openvdb::Coord comp_min = bbox.max();
    openvdb::Coord comp_max = bbox.min();
    for (size_t idx = 0; idx < full_component.size(); ++idx)
    {
        if (!full_component[idx])
        {
            continue;
        }
        const openvdb::Coord c = to_coord(static_cast<int>(idx));
        comp_min = openvdb::Coord(std::min(comp_min.x(), c.x()),
                                  std::min(comp_min.y(), c.y()),
                                  std::min(comp_min.z(), c.z()));
        comp_max = openvdb::Coord(std::max(comp_max.x(), c.x()),
                                  std::max(comp_max.y(), c.y()),
                                  std::max(comp_max.z(), c.z()));
    }
    const openvdb::Vec3d min_world = tf->indexToWorld(comp_min);
    const openvdb::Vec3d max_world = tf->indexToWorld(comp_max);
    const Eigen::Vector3d low_corner(min_world.x() - 0.5 * config_.voxel_size,
                                     min_world.y() - 0.5 * config_.voxel_size,
                                     min_world.z() - 0.5 * config_.voxel_size);
    const Eigen::Vector3d high_corner(max_world.x() + 0.5 * config_.voxel_size,
                                      max_world.y() + 0.5 * config_.voxel_size,
                                      max_world.z() + 0.5 * config_.voxel_size);
    Eigen::Matrix<double, 6, 4> bd = Eigen::Matrix<double, 6, 4>::Zero();
    bd(0, 0) = 1.0;
    bd(1, 0) = -1.0;
    bd(2, 1) = 1.0;
    bd(3, 1) = -1.0;
    bd(4, 2) = 1.0;
    bd(5, 2) = -1.0;
    bd(0, 3) = -high_corner.x();
    bd(1, 3) = low_corner.x();
    bd(2, 3) = -high_corner.y();
    bd(3, 3) = low_corner.y();
    bd(4, 3) = -high_corner.z();
    bd(5, 3) = low_corner.z();

    Eigen::MatrixX4d hpoly;
    const Eigen::Vector3d seed(center_world.x(), center_world.y(), center_world.z());
    if (!firi::firi(bd, pc, seed, seed, hpoly))
    {
        RCLCPP_WARN(logger_,
                    "[Local][VisCell] FIRI failed around witness [%.2f, %.2f, %.2f]",
                    witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
        return try_box_tier();
    }

    // Normalize face rows so the offset column is metric, then shrink every
    // face by shrink_m. FIRI is volume-maximizing: raw vertices sit exactly
    // ON the certified/uncertified boundary, so re-validating them with the
    // zero-slack predicate was a coin flip per vertex. The shrink moves all
    // validation samples >= shrink_m inside that boundary and gives the
    // executed knot the same real position margin, under an unchanged
    // acceptance predicate.
    {
        const Eigen::ArrayXd face_norms = hpoly.leftCols<3>().rowwise().norm();
        if ((face_norms <= 1.0e-12).any())
        {
            return try_box_tier();
        }
        hpoly.array().colwise() /= face_norms;
        if (config_.shrink_m > 0.0)
        {
            hpoly.rightCols<1>().array() += config_.shrink_m;
        }
    }

    Eigen::Vector3d cheb_center = Eigen::Vector3d::Zero();
    double cheb_radius = -1.0;
    Eigen::Matrix3Xd vertices;
    const int max_carves = std::max(config_.max_carve_iters, 0);
    const double carve_eps = std::max(0.25 * config_.voxel_size, 1.0e-3);
    int carve_count = 0;

    // Bounded, monotone carve-repair loop: instead of rejecting the whole
    // cell on one bad vertex (all-or-nothing collapses multiplicatively) or
    // re-running FIRI in another direction (expensive and non-monotone), cut
    // the offending vertex off with a half-space and re-validate only what
    // is new. Terminates via the carve budget or the radius floor; any abort
    // falls back to the seam-stitched formulation upstream.
    while (true)
    {
        if (!computeChebyshevBall(hpoly, cheb_center, cheb_radius) ||
            cheb_radius < config_.min_radius)
        {
            RCLCPP_WARN(logger_,
                        "[Local][VisCell] polytope too thin around witness [%.2f, %.2f, %.2f] | radius=%.3f carves=%d",
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z(),
                        cheb_radius, carve_count);
            return try_box_tier();
        }
        if (!check_sample(seed) || !check_sample(cheb_center))
        {
            RCLCPP_WARN(logger_,
                        "[Local][VisCell] certification failed at seed/center around witness [%.2f, %.2f, %.2f] | carves=%d",
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z(),
                        carve_count);
            return try_box_tier();
        }
        if (!geo_utils::enumerateVs(hpoly, vertices))
        {
            return try_box_tier();
        }

        // Collect ALL failing vertices of this round: leakage comes in
        // clusters (a whole face region bulging past the eroded set), so
        // cutting one vertex per round burned the budget without clearing
        // the cluster. One round = one budget unit, any number of cuts.
        std::vector<int> bad_vertices;
        for (int i = 0; i < vertices.cols(); ++i)
        {
            const Eigen::Vector3d v = vertices.col(i);
            if (!voxel_in_component(v) || !check_sample(v) ||
                !check_sample(0.5 * (v + seed)))
            {
                bad_vertices.push_back(i);
            }
        }
        if (bad_vertices.empty())
        {
            break;
        }
        if (carve_count >= max_carves)
        {
            RCLCPP_WARN(logger_,
                        "[Local][VisCell] certification failed at %zu vertices around witness [%.2f, %.2f, %.2f] | carve budget %d exhausted, radius=%.3f",
                        bad_vertices.size(),
                        witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z(),
                        max_carves, cheb_radius);
            return try_box_tier();
        }

        const int rows = static_cast<int>(hpoly.rows());
        hpoly.conservativeResize(rows + static_cast<int>(bad_vertices.size()), 4);
        int cut_row = rows;
        for (const int idx : bad_vertices)
        {
            const Eigen::Vector3d bad_v = vertices.col(idx);
            Eigen::Vector3d cut_dir = bad_v - seed;
            const double cut_dist = cut_dir.norm();
            if (cut_dist <= carve_eps)
            {
                RCLCPP_WARN(logger_,
                            "[Local][VisCell] failing vertex coincides with seed around witness [%.2f, %.2f, %.2f]",
                            witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
                return try_box_tier();
            }
            cut_dir /= cut_dist;

            // Cut at the certified boundary, not a fixed depth behind the
            // vertex: march seed->vertex through the full cluster and place
            // the plane just inside the first exit. A fixed 0.25-voxel cut
            // converged at ~5 cm/round and burned the budget on deep
            // bulges. Vertices whose whole segment stays in-cluster (map
            // level predicate failures) keep the shallow vertex-local cut.
            double cut_at = cut_dist - carve_eps;
            const double march_step = 0.5 * config_.voxel_size;
            for (double s = march_step; s < cut_dist; s += march_step)
            {
                const Eigen::Vector3d p = seed + s * cut_dir;
                if (!voxel_in_component(p))
                {
                    cut_at = s - carve_eps;
                    break;
                }
            }
            if (cut_at <= carve_eps)
            {
                RCLCPP_WARN(logger_,
                            "[Local][VisCell] carve would collapse to seed around witness [%.2f, %.2f, %.2f]",
                            witness_viewpoint.x(), witness_viewpoint.y(), witness_viewpoint.z());
                return try_box_tier();
            }
            const Eigen::Vector3d cut_point = seed + cut_at * cut_dir;
            hpoly.row(cut_row++) << cut_dir.x(), cut_dir.y(), cut_dir.z(),
                -cut_dir.dot(cut_point);
        }
        ++carve_count;
    }

    if (carve_count > 0)
    {
        RCLCPP_INFO(logger_,
                    "[Local][VisCell] accepted after %d carve(s) | radius=%.3f vertices=%d",
                    carve_count, cheb_radius, static_cast<int>(vertices.cols()));
    }

    // Keep the FIRI seed as the representative center while it remains
    // inside the (shrunk, carved) cell; otherwise fall back to the validated
    // Chebyshev center so downstream access paths target a point the
    // corridor cell actually contains.
    Eigen::Vector3d cell_center = seed;
    if (((hpoly.leftCols<3>() * seed + hpoly.rightCols<1>()).array() > 1.0e-9).any())
    {
        cell_center = cheb_center;
    }
    cell_out.hpoly = hpoly;
    cell_out.center = cell_center;
    cell_out.target = Eigen::Vector3d(target_world.x(), target_world.y(), target_world.z());
    // certified_yaw must stay the yaw the samples were validated under
    // (seed-facing psi_cmd), even when the representative center switched
    // to the Chebyshev center above.
    cell_out.certified_yaw = psi_cmd;
    cell_out.chebyshev_radius = cheb_radius;
    cell_out.valid = true;
    return true;
}
