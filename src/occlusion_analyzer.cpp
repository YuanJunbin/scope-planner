// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/occlusion_analyzer.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

void OcclusionAnalyzer::initialize(rclcpp::Node::SharedPtr node_handle,
                                   std::shared_ptr<VDBMap> &map_manager,
                                   const SensorFovConfig &cfg,
                                   const std::string &world_frame_id)
{
    node_handle_ = node_handle;
    map_manager_ = map_manager;
    cfg_ = cfg;
    world_frame_id_ = world_frame_id;

    pub_unknown_cloud_ =
        node_handle_->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/occlusion_unknowns", 10);
}

std::vector<openvdb::Coord>
OcclusionAnalyzer::enumerateUnknownVoxels(const openvdb::Vec3d &hit_pt) const
{
    std::vector<openvdb::Coord> unknowns;
    if (!map_manager_)
    {
        return unknowns;
    }

    openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();
    openvdb::Coord center = openvdb::Coord::round(grid_tf->worldToIndex(hit_pt));

    const auto &kernel = map_manager_->get_inflation_kernel();

    std::shared_lock<std::shared_mutex> map_lk(map_manager_->get_map_mutex());
    openvdb::FloatGrid::ConstAccessor occ_acc = map_manager_->get_logocc_accessor();

    for (const auto &offset : kernel)
    {
        openvdb::Coord ijk = center + offset;
        float logodds = 0.0f;
        bool observed = map_manager_->query_log_odds_at_index(ijk, logodds, occ_acc);
        if (!observed)
        {
            unknowns.push_back(ijk);
        }
    }

    return unknowns;
}

void OcclusionAnalyzer::reset()
{
    curr_unknowns_.clear();
    publishEmptyUnknownCloud();
}

void OcclusionAnalyzer::analyzeTick(const openvdb::Vec3d &hit_pt)
{
    curr_unknowns_ = enumerateUnknownVoxels(hit_pt);

    if (curr_unknowns_.empty())
    {
        RCLCPP_INFO(node_handle_->get_logger(),
                    "[Occlusion] No unknown voxels in inflation kernel");
        publishEmptyUnknownCloud();
        return;
    }

    RCLCPP_INFO_THROTTLE(node_handle_->get_logger(),
                         *node_handle_->get_clock(), 2000,
                         "[Occlusion] %zu unknowns in inflation kernel",
                         curr_unknowns_.size());

    publishUnknownCloud();
}

void OcclusionAnalyzer::publishUnknownCloud()
{
    if (pub_unknown_cloud_->get_subscription_count() == 0)
    {
        return;
    }

    openvdb::math::Transform::ConstPtr grid_tf = map_manager_->get_grid_transform();

    pcl::PointCloud<pcl::PointXYZ> unknown_pcl;
    unknown_pcl.reserve(curr_unknowns_.size());
    for (auto &ijk : curr_unknowns_)
    {
        openvdb::Vec3d w = grid_tf->indexToWorld(ijk);
        unknown_pcl.emplace_back(
            static_cast<float>(w.x()),
            static_cast<float>(w.y()),
            static_cast<float>(w.z()));
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(unknown_pcl, msg);
    msg.header.frame_id = world_frame_id_;
    msg.header.stamp = node_handle_->now();
    pub_unknown_cloud_->publish(msg);
}

void OcclusionAnalyzer::publishEmptyUnknownCloud()
{
    if (pub_unknown_cloud_->get_subscription_count() == 0)
    {
        return;
    }

    pcl::PointCloud<pcl::PointXYZ> empty_pcl;

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(empty_pcl, msg);
    msg.header.frame_id = world_frame_id_;
    msg.header.stamp = node_handle_->now();
    pub_unknown_cloud_->publish(msg);
}
