// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#pragma once

#include "fov_astar.hpp"

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vdb_edt/vdbmap.h>

// Enumerates unknown (unobserved) voxels within the inflation kernel of a
// target hit point. Publishes them as PointCloud2 on /occlusion_unknowns.
//
// Viewpoint selection for clearing these unknowns is handled by ObserveAstar.
class OcclusionAnalyzer
{
public:
    void initialize(rclcpp::Node::SharedPtr node_handle,
                    std::shared_ptr<VDBMap> &map_manager,
                    const SensorFovConfig &cfg,
                    const std::string &world_frame_id);

    void analyzeTick(const openvdb::Vec3d &hit_pt);

    void reset();

    std::vector<openvdb::Coord> enumerateUnknownVoxels(const openvdb::Vec3d &hit_pt) const;

    const std::vector<openvdb::Coord> &getCurrentUnknowns() const { return curr_unknowns_; }

private:
    void publishUnknownCloud();
    void publishEmptyUnknownCloud();

    rclcpp::Node::SharedPtr node_handle_;
    std::shared_ptr<VDBMap> map_manager_;
    SensorFovConfig cfg_;
    std::string world_frame_id_;

    std::vector<openvdb::Coord> curr_unknowns_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_unknown_cloud_;
};
