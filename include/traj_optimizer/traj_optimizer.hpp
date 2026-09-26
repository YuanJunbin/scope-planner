// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#ifndef TRAJ_OPTIMIZER_TRAJ_OPTIMIZER_HPP
#define TRAJ_OPTIMIZER_TRAJ_OPTIMIZER_HPP

#include "traj_optimizer/gcopter.hpp"
#include "traj_optimizer/sfc_gen.hpp"

#include <rclcpp/rclcpp.hpp>

#include <vdb_edt/vdbmap.h>

#include <openvdb/openvdb.h>

#include <Eigen/Core>

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace traj_optimizer
{

enum class CorridorCellType : uint8_t
{
    MOTION = 0,
    VISIBILITY = 1,
};

struct CorridorCell
{
    Eigen::MatrixX4d hpoly;
    CorridorCellType type = CorridorCellType::MOTION;
    bool locked = false;
    int min_pieces = 1;
};

struct CorridorSequence
{
    std::vector<CorridorCell> cells;
};

struct VisibilityCell
{
    Eigen::MatrixX4d hpoly;
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
    double certified_yaw = 0.0;
    double chebyshev_radius = 0.0;
    bool valid = false;
};

struct ObservationHint
{
    int cell_index = -1;
    double yaw_ref = 0.0;
    double yaw_weight = 0.0;
};

struct BoundaryState
{
    Eigen::Vector3d pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d vel = Eigen::Vector3d::Zero();
    Eigen::Vector3d acc = Eigen::Vector3d::Zero();
};

struct DebugInfo
{
    std::vector<Eigen::Vector3d> input_path;
    openvdb::CoordBBox local_bbox;
    std::vector<Eigen::Vector3d> inflated_points;
    gcopter::GCOPTER_PolytopeSFC::PolyhedraH corridor_hpolys;
    CorridorSequence corridor_seq;
    std::vector<int> piece_idx;
    int observation_cell_index = -1;
    int observation_knot_index = -1;
};

struct OptimizedPlan
{
    using PositionTrajectory = ::Trajectory<5>;

    PositionTrajectory position_traj;
    std::vector<double> piece_durations;
    std::vector<double> yaw_knots;
    bool valid = false;
    // Time-from-start at which the stitched plan's anchor knot (vp1 seam for
    // seam-stitched plans, or the observation-cell knot for visibility-cell
    // plans) is reached. Negative => no anchor (single-stage plan). Filled by
    // the optimizer / publisher path; callers use it directly to arm stitch
    // monitoring instead of re-projecting a 3D anchor onto the published
    // polyline (which fails on degenerate / zero-length segments).
    double anchor_time = -1.0;

    void clear()
    {
        position_traj.clear();
        piece_durations.clear();
        yaw_knots.clear();
        valid = false;
        anchor_time = -1.0;
    }

    double getTotalDuration() const
    {
        double total = 0.0;
        for (double dt : piece_durations)
        {
            total += dt;
        }
        return total;
    }

    double sampleYaw(double t) const
    {
        if (piece_durations.empty() || yaw_knots.size() != piece_durations.size() + 1)
        {
            return 0.0;
        }

        if (t <= 0.0)
        {
            return yaw_knots.front();
        }

        double remaining = t;
        for (size_t i = 0; i < piece_durations.size(); ++i)
        {
            const double dt = piece_durations[i];
            if (remaining <= dt || i + 1 == piece_durations.size())
            {
                const double frac = (dt > 1.0e-9) ? std::clamp(remaining / dt, 0.0, 1.0) : 1.0;
                return yaw_knots[i] + frac * (yaw_knots[i + 1] - yaw_knots[i]);
            }
            remaining -= dt;
        }

        return yaw_knots.back();
    }

    double sampleYawRate(double t) const
    {
        if (piece_durations.empty() || yaw_knots.size() != piece_durations.size() + 1)
        {
            return 0.0;
        }

        if (t <= 0.0)
        {
            const double dt = piece_durations.front();
            return dt > 1.0e-9 ? (yaw_knots[1] - yaw_knots[0]) / dt : 0.0;
        }

        double remaining = t;
        for (size_t i = 0; i < piece_durations.size(); ++i)
        {
            const double dt = piece_durations[i];
            if (remaining <= dt || i + 1 == piece_durations.size())
            {
                return dt > 1.0e-9 ? (yaw_knots[i + 1] - yaw_knots[i]) / dt : 0.0;
            }
            remaining -= dt;
        }

        return 0.0;
    }

    void append(const OptimizedPlan &other)
    {
        if (!other.valid)
        {
            return;
        }

        if (!valid)
        {
            *this = other;
            return;
        }

        // The append() seam itself is treated as an anchor when the first
        // plan has none and the caller does not override anchor_time after
        // the call. Pre-set anchors on either side are preserved on the
        // first plan; an anchor that sat inside `other` is shifted by the
        // current plan's duration to remain meaningful on the stitched
        // timeline.
        if (anchor_time < 0.0)
        {
            if (other.anchor_time >= 0.0)
            {
                anchor_time = getTotalDuration() + other.anchor_time;
            }
            else
            {
                anchor_time = getTotalDuration();
            }
        }

        position_traj.append(other.position_traj);
        piece_durations.insert(piece_durations.end(),
                               other.piece_durations.begin(),
                               other.piece_durations.end());
        if (!other.yaw_knots.empty())
        {
            double prev_yaw = yaw_knots.empty() ? other.yaw_knots.front() : yaw_knots.back();
            for (size_t i = 1; i < other.yaw_knots.size(); ++i)
            {
                const double yaw =
                    prev_yaw + std::remainder(other.yaw_knots[i] - prev_yaw, 2.0 * M_PI);
                yaw_knots.push_back(yaw);
                prev_yaw = yaw;
            }
        }
        valid = true;
    }
};

class TrajOptimizer
{
public:
    TrajOptimizer() = default;

    bool initialize(const rclcpp::Node::SharedPtr &node_handle,
                    const std::shared_ptr<VDBMap> &map_manager);

    bool optimizePath(const std::vector<openvdb::Vec3d> &path_world,
                      const BoundaryState &start_state,
                      const BoundaryState &goal_state,
                      double start_yaw,
                      double goal_yaw,
                      OptimizedPlan &plan_out,
                      DebugInfo *debug_out = nullptr) const;

    bool optimizeStitchedPath(const std::vector<openvdb::Vec3d> &path_world,
                              size_t split_index,
                              const BoundaryState &start_state,
                              const BoundaryState &goal_state,
                              double start_yaw,
                              double split_yaw,
                              double goal_yaw,
                              OptimizedPlan &plan_out,
                              DebugInfo *debug_out = nullptr) const;

    bool optimizeThroughVisibilityCell(const std::vector<openvdb::Vec3d> &path_in_world,
                                       const VisibilityCell &visibility_cell,
                                       const std::vector<openvdb::Vec3d> &path_out_world,
                                       const BoundaryState &start_state,
                                       const BoundaryState &goal_state,
                                       double start_yaw,
                                       double goal_yaw,
                                       const ObservationHint &observation_hint,
                                       OptimizedPlan &plan_out,
                                       DebugInfo *debug_out = nullptr) const;

private:
    bool loadParameters();

    std::vector<Eigen::Vector3d> convertPath(
        const std::vector<openvdb::Vec3d> &path_world) const;

    bool computeLocalBBoxFromPath(const std::vector<Eigen::Vector3d> &path_world,
                                  openvdb::CoordBBox &bbox_out) const;

    bool buildCorridor(const std::vector<Eigen::Vector3d> &path_world,
                       const openvdb::CoordBBox &bbox,
                       const std::vector<Eigen::Vector3d> &inflated_points,
                       gcopter::GCOPTER_PolytopeSFC::PolyhedraH &corridor_out) const;

    bool buildMotionCorridor(const std::vector<Eigen::Vector3d> &path_world,
                             const openvdb::CoordBBox &bbox,
                             const std::vector<Eigen::Vector3d> &inflated_points,
                             std::vector<CorridorCell> &cells_out,
                             bool do_shortcut = true) const;

    bool buildConnectorCell(const Eigen::Vector3d &joint_world,
                            const openvdb::CoordBBox &bbox,
                            const std::vector<Eigen::Vector3d> &inflated_points,
                            CorridorCell &cell_out) const;

    bool optimizeCorridorSequence(const std::vector<Eigen::Vector3d> &full_path_world,
                                  const openvdb::CoordBBox &bbox,
                                  const std::vector<Eigen::Vector3d> &inflated_points,
                                  const CorridorSequence &corridor_seq,
                                  const BoundaryState &start_state,
                                  const BoundaryState &goal_state,
                                  double start_yaw,
                                  double goal_yaw,
                                  const ObservationHint &observation_hint,
                                  OptimizedPlan &plan_out,
                                  DebugInfo *debug_out = nullptr,
                                  const std::vector<Eigen::Vector3d> *inflated_voxel_centers = nullptr,
                                  const std::vector<Eigen::Vector3d> *center_only_voxels = nullptr) const;

    bool computeChebyshevBall(const Eigen::MatrixX4d &hpoly,
                              Eigen::Vector3d &center_out,
                              double &radius_out) const;

    bool checkAdjacentOverlapThickness(const Eigen::MatrixX4d &a,
                                       const Eigen::MatrixX4d &b,
                                       double &radius_out,
                                       double min_radius = -1.0) const;

    Eigen::Matrix3d makeBoundaryPVA(const BoundaryState &state) const;
    Eigen::VectorXd makeMagnitudeBounds() const;
    Eigen::VectorXd makePenaltyWeights() const;
    Eigen::VectorXd makePhysicalParams() const;

private:
    rclcpp::Node::SharedPtr node_handle_;
    std::shared_ptr<VDBMap> map_manager_;
    bool initialized_ = false;
    bool snapshot_enable_ = false;
    bool strict_traj_ = false;
    std::string snapshot_dir_ = "robot/ros_ws/src/autonomy/4_global/b_planners/ffa_planner/log_debug";

    double corridor_progress_ = 1.0;
    double corridor_range_ = 1.5;
    int corridor_padding_vox_ = 2;

    double time_weight_ = 20.0;
    double jerk_weight_ = 1.0;
    double length_per_piece_ = 0.0;
    double smoothing_eps_ = 1.0e-2;
    int integral_resolution_ = 16;
    double rel_cost_tol_ = 1.0e-5;
    double yaw_smooth_weight_ = 1.0;
    double max_yaw_rate_ = 1.0;
    double penalty_yaw_rate_ = 1.0e4;
    double init_alloc_speed_ratio_ = 0.5;
    double init_max_acc_ = 1.0;
    double init_time_margin_ = 1.2;
    double stitched_seam_pos_weight_ = 1.0e4;
    double stitched_seam_yaw_weight_ = 1.0e4;
    double visibility_overlap_min_radius_ = 0.05;
    int visibility_min_pieces_ = 2;

    double max_vel_ = 4.0;
    double max_body_rate_ = 2.1;
    double max_tilt_ = 1.05;
    double min_thrust_ = 2.0;
    double max_thrust_ = 12.0;

    double vehicle_mass_ = 0.61;
    double gravity_ = 9.8;
    double horiz_drag_ = 0.70;
    double vert_drag_ = 0.80;
    double paras_drag_ = 0.01;
    double speed_eps_ = 1.0e-4;

    double penalty_pos_ = 1.0e4;
    double penalty_vel_ = 1.0e4;
    double penalty_omg_ = 1.0e4;
    double penalty_tilt_ = 1.0e4;
    double penalty_thrust_ = 1.0e5;
};

} // namespace traj_optimizer

#endif
