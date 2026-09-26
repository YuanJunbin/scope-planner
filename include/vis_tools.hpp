// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#ifndef VIS_TOOLS_H
#define VIS_TOOLS_H

#include <geometry_msgs/msg/point.hpp>
#include <openvdb/openvdb.h>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <vector>

visualization_msgs::msg::MarkerArray path_to_marker_array(
    const std::vector<openvdb::Vec3d> &path_world,
    double alpha = 1.0);

#endif
