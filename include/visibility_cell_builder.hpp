// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include "fov_astar.hpp"
#include "traj_optimizer/traj_optimizer.hpp"

#include <rclcpp/rclcpp.hpp>

#include <openvdb/openvdb.h>
#include <vdb_edt/vdbmap.h>

#include <functional>
#include <memory>

class VisibilityCellBuilder
{
public:
    using ClearableFn = std::function<bool(const openvdb::Vec3d &,
                                           const openvdb::Vec3d &,
                                           FovClearProfileKind)>;
    // Fixed-yaw variant: (anchor, target, profile, psi_body). When provided,
    // the continuous acceptance samples are validated under the yaw that
    // will actually be commanded at the observation knot instead of the
    // per-sample target-facing yaw. Cluster growth keeps ClearableFn.
    using ClearableYawFn = std::function<bool(const openvdb::Vec3d &,
                                              const openvdb::Vec3d &,
                                              FovClearProfileKind,
                                              double)>;

    struct Config
    {
        double roi_xy = 0.0;
        double roi_z = 0.0;
        int erode_iters = 0;
        double min_radius = 0.0;
        double voxel_size = 0.0;
        double body_sensor_yaw = 0.0;
        // Metric shrink applied to every FIRI face before validation; buys
        // the executed knot a real position margin and moves the validation
        // samples off the zero-slack boundary. 0 = legacy behavior.
        double shrink_m = 0.0;
        // Carve-repair budget: a failing vertex is cut off by a half-space
        // (monotone, sub-ms) instead of rejecting the whole cell; abort to
        // the seam fallback once exhausted. 0 = legacy all-or-nothing.
        int max_carve_iters = 0;
    };

    VisibilityCellBuilder(std::shared_ptr<VDBMap> map_manager,
                          Config config,
                          ClearableFn clearable_fn,
                          rclcpp::Logger logger);

    void setFixedYawValidator(ClearableYawFn fn) { clearable_yaw_fn_ = std::move(fn); }

    bool build(const openvdb::Vec3d &witness_viewpoint,
               const openvdb::Vec3d &target_world,
               traj_optimizer::VisibilityCell &cell_out) const;

private:
    std::shared_ptr<VDBMap> map_manager_;
    Config config_;
    ClearableFn clearable_fn_;
    ClearableYawFn clearable_yaw_fn_;
    rclcpp::Logger logger_;
};
