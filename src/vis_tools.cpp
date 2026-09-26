// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 Carnegie Mellon University

#include "../include/ffa_planner_node.hpp"
#include "../include/vis_tools.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

namespace
{
struct Rgb
{
    float r;
    float g;
    float b;
};

void setMarkerRgb(visualization_msgs::msg::Marker &marker, const Rgb &rgb, float alpha)
{
    marker.color.r = rgb.r;
    marker.color.g = rgb.g;
    marker.color.b = rgb.b;
    marker.color.a = alpha;
}

visualization_msgs::msg::Marker makeDeleteAllMarker(const std::string &frame_id,
                                                    const rclcpp::Time &stamp,
                                                    const std::string &ns)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker.pose.orientation.w = 1.0;
    return marker;
}

visualization_msgs::msg::Marker makeDeleteMarker(const std::string &frame_id,
                                                 const rclcpp::Time &stamp,
                                                 const std::string &ns,
                                                 int id)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::msg::Marker::DELETE;
    marker.pose.orientation.w = 1.0;
    return marker;
}

geometry_msgs::msg::Point toRosPoint(const Eigen::Vector3d &p)
{
    geometry_msgs::msg::Point ros_p;
    ros_p.x = p.x();
    ros_p.y = p.y();
    ros_p.z = p.z();
    return ros_p;
}

visualization_msgs::msg::MarkerArray corridorToMarkerArray(
    const gcopter::GCOPTER_PolytopeSFC::PolyhedraH &corridor_hpolys,
    const std::string &frame_id,
    const rclcpp::Time &stamp)
{
    visualization_msgs::msg::MarkerArray arr;
    if (corridor_hpolys.empty())
    {
        return arr;
    }

    visualization_msgs::msg::Marker faces;
    faces.header.frame_id = frame_id;
    faces.header.stamp = stamp;
    faces.ns = "traj_sfc_corridor_faces";
    faces.id = 2;
    faces.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    faces.action = visualization_msgs::msg::Marker::ADD;
    faces.pose.orientation.w = 1.0;
    faces.scale.x = 1.0;
    faces.scale.y = 1.0;
    faces.scale.z = 1.0;
    setMarkerRgb(faces, {0.20f, 0.55f, 1.00f}, 0.14f);

    visualization_msgs::msg::Marker edges;
    edges.header.frame_id = frame_id;
    edges.header.stamp = stamp;
    edges.ns = "traj_sfc_corridor_edges";
    edges.id = 3;
    edges.type = visualization_msgs::msg::Marker::LINE_LIST;
    edges.action = visualization_msgs::msg::Marker::ADD;
    edges.pose.orientation.w = 1.0;
    edges.scale.x = 0.02;
    setMarkerRgb(edges, {0.10f, 0.35f, 0.95f}, 0.60f);

    for (const auto &hpoly : corridor_hpolys)
    {
        Eigen::Matrix3Xd v_poly;
        if (!geo_utils::enumerateVs(hpoly, v_poly) || v_poly.cols() < 4)
        {
            continue;
        }

        quickhull::QuickHull<double> qh;
        const auto hull = qh.getConvexHull(v_poly.data(),
                                           static_cast<size_t>(v_poly.cols()),
                                           true,
                                           false,
                                           quickhull::defaultEps<double>());
        const auto &indices = hull.getIndexBuffer();
        const auto &vertices = hull.getVertexBuffer();
        if (indices.size() < 3)
        {
            continue;
        }

        std::set<std::pair<size_t, size_t>> unique_edges;
        for (size_t i = 0; i + 2 < indices.size(); i += 3)
        {
            const size_t i0 = indices[i + 0];
            const size_t i1 = indices[i + 1];
            const size_t i2 = indices[i + 2];

            const Eigen::Vector3d p0(vertices[i0].x, vertices[i0].y, vertices[i0].z);
            const Eigen::Vector3d p1(vertices[i1].x, vertices[i1].y, vertices[i1].z);
            const Eigen::Vector3d p2(vertices[i2].x, vertices[i2].y, vertices[i2].z);

            faces.points.push_back(toRosPoint(p0));
            faces.points.push_back(toRosPoint(p1));
            faces.points.push_back(toRosPoint(p2));

            unique_edges.emplace(std::min(i0, i1), std::max(i0, i1));
            unique_edges.emplace(std::min(i1, i2), std::max(i1, i2));
            unique_edges.emplace(std::min(i2, i0), std::max(i2, i0));
        }

        for (const auto &[a, b] : unique_edges)
        {
            const Eigen::Vector3d pa(vertices[a].x, vertices[a].y, vertices[a].z);
            const Eigen::Vector3d pb(vertices[b].x, vertices[b].y, vertices[b].z);
            edges.points.push_back(toRosPoint(pa));
            edges.points.push_back(toRosPoint(pb));
        }
    }

    if (!faces.points.empty())
    {
        arr.markers.push_back(std::move(faces));
    }
    if (!edges.points.empty())
    {
        arr.markers.push_back(std::move(edges));
    }
    return arr;
}

visualization_msgs::msg::Marker makeCorridorPointCloudMarker(
    const std::vector<Eigen::Vector3d> &points,
    const std::string &frame_id,
    const rclcpp::Time &stamp)
{
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "traj_sfc_corridor_points";
    marker.id = 4;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.06;
    marker.scale.y = 0.06;
    marker.scale.z = 0.06;
    setMarkerRgb(marker, {1.00f, 0.20f, 0.10f}, 0.75f);
    marker.points.reserve(points.size());
    for (const auto &p : points)
    {
        marker.points.push_back(toRosPoint(p));
    }
    return marker;
}
} // namespace

visualization_msgs::msg::MarkerArray path_to_marker_array(
    const std::vector<openvdb::Vec3d> &path_world, double alpha)
{
    visualization_msgs::msg::MarkerArray arr;
    if (path_world.empty())
    {
        return arr;
    }

    visualization_msgs::msg::Marker spheres;
    spheres.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    spheres.action = visualization_msgs::msg::Marker::ADD;
    spheres.ns = "astar_path";
    spheres.id = 0;
    spheres.pose.orientation.w = 1.0;
    spheres.scale.x = 0.1;
    spheres.scale.y = 0.1;
    spheres.scale.z = 0.1;
    spheres.color.r = 1.0f;
    spheres.color.g = 0.2f;
    spheres.color.b = 0.2f;
    spheres.color.a = alpha;

    visualization_msgs::msg::Marker line;
    line.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line.action = visualization_msgs::msg::Marker::ADD;
    line.ns = "astar_path";
    line.id = 1;
    line.pose.orientation.w = 1.0;
    line.scale.x = 0.05;
    line.color.r = 0.2f;
    line.color.g = 0.8f;
    line.color.b = 1.0f;
    line.color.a = alpha;

    spheres.points.reserve(path_world.size());
    line.points.reserve(path_world.size());

    for (const auto &w : path_world)
    {
        geometry_msgs::msg::Point p;
        p.x = w.x();
        p.y = w.y();
        p.z = w.z();
        spheres.points.push_back(p);
        line.points.push_back(p);
    }

    arr.markers.push_back(std::move(spheres));
    arr.markers.push_back(std::move(line));
    return arr;
}

void FFAPlannerNode::clearSubgoalVisualization()
{
    auto stamp = node_handle_->now();
    auto publish_array_delete = [&](const auto &pub, const std::vector<std::pair<std::string, int>> &markers)
    {
        if (!pub)
        {
            return;
        }

        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(makeDeleteAllMarker(world_frame_id_, stamp, "subgoal_clear"));
        for (const auto &[ns, id] : markers)
        {
            arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, ns, id));
        }
        pub->publish(arr);
    };

    auto publish_marker_delete = [&](const auto &pub, const char *ns)
    {
        if (!pub)
        {
            return;
        }

        pub->publish(makeDeleteAllMarker(world_frame_id_, stamp, "subgoal_clear"));
        pub->publish(makeDeleteMarker(world_frame_id_, stamp, ns, 0));
    };

    publish_array_delete(pub_subgoal_guidance_path_vis_, {{"subgoal_guidance_astar", 0}, {"subgoal_guidance_astar", 1}});
    publish_array_delete(pub_subgoal_guidance_short_vis_, {{"subgoal_guidance_short", 0}, {"subgoal_guidance_short", 1}});
    publish_marker_delete(pub_subgoal_debug_tree_, "subgoal_guidance_tree");
    publish_array_delete(pub_subgoal_safe_edge_vis_, {{"subgoal_safe_edge", 0}});
    publish_array_delete(pub_subgoal_fov_viewpoint_vis_, {{"subgoal_active_viewpoint", 0}});
    publish_array_delete(pub_subgoal_fov_path_vis_, {{"subgoal_fov_astar", 0}, {"subgoal_fov_astar", 1}});
    publish_array_delete(pub_subgoal_fov_short_vis_, {{"subgoal_fov_short", 0}, {"subgoal_fov_short", 1}});
    publish_array_delete(pub_subgoal_fov_viewpoint_vis_, {{"subgoal_lookahead_viewpoint", 0}});
    publish_array_delete(pub_subgoal_fov_path_vis_, {{"subgoal_lookahead_fov_astar", 0}, {"subgoal_lookahead_fov_astar", 1}});
    publish_array_delete(pub_subgoal_fov_short_vis_, {{"subgoal_lookahead_fov_short", 0}, {"subgoal_lookahead_fov_short", 1}});
    publish_marker_delete(pub_subgoal_fov_debug_tree_, "subgoal_fov_tree");
    publish_marker_delete(pub_subgoal_observe_debug_tree_, "subgoal_observe_tree");
    publish_array_delete(pub_subgoal_observe_path_vis_, {{"subgoal_observe_astar", 0}, {"subgoal_observe_astar", 1}});
    publish_array_delete(pub_subgoal_observe_short_vis_, {{"subgoal_observe_short", 0}, {"subgoal_observe_short", 1}});
    publish_array_delete(pub_observe_frustum_vis_, {{"subgoal_observe_viewpoint", 0},
                                                    {"subgoal_observe_frustum", 0},
                                                    {"subgoal_observe_target", 0}});
    publish_marker_delete(pub_subgoal_connect_debug_tree_, "subgoal_connect_tree");
    publish_array_delete(pub_subgoal_connect_path_vis_, {{"subgoal_connect_astar", 0}, {"subgoal_connect_astar", 1}});
    publish_array_delete(pub_subgoal_connect_short_vis_, {{"subgoal_connect_short", 0}, {"subgoal_connect_short", 1}});
    publish_array_delete(pub_subgoal_fov_frustum_vis_, {{"subgoal_active_frustum", 0}, {"subgoal_active_target", 0}});
    publish_array_delete(pub_subgoal_fov_frustum_vis_, {{"subgoal_lookahead_frustum", 0}, {"subgoal_lookahead_target", 0}});
}

void FFAPlannerNode::publishOptimizedTrajectoryVis(
    const traj_optimizer::OptimizedPlan::PositionTrajectory &traj,
    const traj_optimizer::DebugInfo * /*debug_info*/)
{
    if (!pub_traj_optimizer_vis_ || pub_traj_optimizer_vis_->get_subscription_count() == 0)
    {
        return;
    }
    if (traj.getPieceNum() <= 0)
    {
        clearOptimizedTrajectoryVis();
        return;
    }

    const double total_duration = traj.getTotalDuration();
    if (total_duration <= 1e-6)
    {
        clearOptimizedTrajectoryVis();
        return;
    }

    const double sample_dt =
        (cruise_speed_ > 1e-6 && interpolate_step_ > 1e-6) ? (interpolate_step_ / cruise_speed_) : 0.1;

    std::vector<openvdb::Vec3d> sampled_path;
    for (double t = 0.0; t < total_duration; t += sample_dt)
    {
        const Eigen::Vector3d p = traj.getPos(t);
        sampled_path.emplace_back(p.x(), p.y(), p.z());
    }
    const Eigen::Vector3d p_end = traj.getPos(total_duration);
    sampled_path.emplace_back(p_end.x(), p_end.y(), p_end.z());

    auto markers = path_to_marker_array(sampled_path, 0.95);
    const auto stamp = node_handle_->now();
    for (auto &m : markers.markers)
    {
        m.header.frame_id = world_frame_id_;
        m.header.stamp = stamp;
        m.ns = "traj_optimizer_candidate";
        if (m.id == 0)
        {
            m.scale.x = 0.08;
            m.scale.y = 0.08;
            m.scale.z = 0.08;
            setMarkerRgb(m, {0.15f, 0.82f, 0.22f}, 0.98f);
        }
        else
        {
            m.scale.x = 0.07;
            setMarkerRgb(m, {0.05f, 0.72f, 0.12f}, 0.98f);
        }
    }
    pub_traj_optimizer_vis_->publish(markers);
}

void FFAPlannerNode::clearOptimizedTrajectoryVis()
{
    if (!pub_traj_optimizer_vis_)
    {
        return;
    }
    visualization_msgs::msg::MarkerArray arr;
    const auto stamp = node_handle_->now();
    arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, "traj_optimizer_candidate", 0));
    arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, "traj_optimizer_candidate", 1));
    pub_traj_optimizer_vis_->publish(arr);
}

void FFAPlannerNode::publishSfcDebugVis(const traj_optimizer::DebugInfo &debug_info)
{
    if (!pub_traj_sfc_vis_ || pub_traj_sfc_vis_->get_subscription_count() == 0)
    {
        return;
    }

    const auto stamp = node_handle_->now();
    visualization_msgs::msg::MarkerArray markers =
        corridorToMarkerArray(debug_info.corridor_hpolys, world_frame_id_, stamp);
    if (!debug_info.inflated_points.empty())
    {
        markers.markers.push_back(
            makeCorridorPointCloudMarker(debug_info.inflated_points, world_frame_id_, stamp));
    }
    pub_traj_sfc_vis_->publish(markers);
}

void FFAPlannerNode::clearSfcDebugVis()
{
    if (!pub_traj_sfc_vis_)
    {
        return;
    }
    visualization_msgs::msg::MarkerArray arr;
    const auto stamp = node_handle_->now();
    arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, "traj_sfc_corridor_faces", 2));
    arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, "traj_sfc_corridor_edges", 3));
    arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, "traj_sfc_corridor_points", 4));
    pub_traj_sfc_vis_->publish(arr);
}

void FFAPlannerNode::publishSafeEdgeVis(const SafeEdgeHitResult &result, VisLayer layer)
{
    auto pub = (layer == VisLayer::ROOT) ? pub_safe_edge_vis_ : pub_subgoal_safe_edge_vis_;
    if (pub->get_subscription_count() == 0)
    {
        return;
    }

    visualization_msgs::msg::MarkerArray markers;
    auto stamp = node_handle_->now();
    const char *ns = (layer == VisLayer::ROOT) ? "root_safe_edge" : "subgoal_safe_edge";

    visualization_msgs::msg::Marker m;
    m.header.frame_id = world_frame_id_;
    m.header.stamp = stamp;
    m.ns = ns;
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose.position.x = result.hit_pt.x();
    m.pose.position.y = result.hit_pt.y();
    m.pose.position.z = result.hit_pt.z();
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 0.6;
    if (result.path_blocked)
    {
        setMarkerRgb(m, {1.0f, 0.0f, 0.0f}, 0.85f);
    }
    else if (layer == VisLayer::ROOT)
    {
        setMarkerRgb(m, {1.0f, 1.0f, 0.0f}, 0.85f);
    }
    else
    {
        setMarkerRgb(m, {0.15f, 0.85f, 1.0f}, 0.85f);
    }
    markers.markers.push_back(m);
    pub->publish(markers);
}

void FFAPlannerNode::publishGuidancePathVis(const std::vector<openvdb::Vec3d> &raw_path,
                                            const std::vector<openvdb::Vec3d> &short_path,
                                            VisLayer layer)
{
    auto path_pub = (layer == VisLayer::ROOT) ? pub_guidance_path_vis_ : pub_subgoal_guidance_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_guidance_short_vis_ : pub_subgoal_guidance_short_vis_;
    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_guidance_astar" : "subgoal_guidance_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_guidance_short" : "subgoal_guidance_short";
    const Rgb raw_rgb = (layer == VisLayer::ROOT) ? Rgb{0.18f, 0.75f, 1.0f} : Rgb{0.62f, 0.40f, 1.0f};
    const Rgb short_rgb = (layer == VisLayer::ROOT) ? Rgb{0.05f, 0.95f, 0.85f} : Rgb{1.0f, 0.45f, 0.95f};

    auto stamp = node_handle_->now();

    if (!raw_path.empty() && path_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(raw_path, 0.4);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = raw_ns;
            setMarkerRgb(m, raw_rgb, 0.4f);
        }
        path_pub->publish(markers);
    }

    if (!short_path.empty() && short_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(short_path, 1.0);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = short_ns;
            setMarkerRgb(m, short_rgb, 1.0f);
        }
        short_pub->publish(markers);
    }
}

void FFAPlannerNode::publishDebugTreeVis(const visualization_msgs::msg::Marker &tree_marker,
                                         DebugTreeKind kind,
                                         VisLayer layer)
{
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr pub;
    const char *ns = "";
    Rgb rgb{1.0f, 1.0f, 1.0f};
    float alpha = 0.25f;

    if (layer == VisLayer::ROOT)
    {
        switch (kind)
        {
        case DebugTreeKind::GUIDANCE:
            pub = pub_debug_tree_;
            ns = "root_guidance_tree";
            rgb = {0.20f, 0.70f, 1.0f};
            alpha = 0.25f;
            break;
        case DebugTreeKind::FOV:
            pub = pub_fov_debug_tree_;
            ns = "root_fov_tree";
            rgb = {0.85f, 0.25f, 1.0f};
            alpha = 0.20f;
            break;
        case DebugTreeKind::OBSERVE:
            pub = pub_observe_debug_tree_;
            ns = "root_observe_tree";
            rgb = {0.20f, 1.0f, 0.35f};
            alpha = 0.30f;
            break;
        case DebugTreeKind::CONNECT:
            pub = pub_connect_debug_tree_;
            ns = "root_connect_tree";
            rgb = {0.20f, 0.70f, 1.0f};
            alpha = 0.35f;
            break;
        }
    }
    else
    {
        switch (kind)
        {
        case DebugTreeKind::GUIDANCE:
            pub = pub_subgoal_debug_tree_;
            ns = "subgoal_guidance_tree";
            rgb = {0.70f, 0.45f, 1.0f};
            alpha = 0.30f;
            break;
        case DebugTreeKind::FOV:
            pub = pub_subgoal_fov_debug_tree_;
            ns = "subgoal_fov_tree";
            rgb = {0.30f, 0.80f, 1.0f};
            alpha = 0.25f;
            break;
        case DebugTreeKind::OBSERVE:
            pub = pub_subgoal_observe_debug_tree_;
            ns = "subgoal_observe_tree";
            rgb = {0.25f, 1.0f, 0.75f};
            alpha = 0.30f;
            break;
        case DebugTreeKind::CONNECT:
            pub = pub_subgoal_connect_debug_tree_;
            ns = "subgoal_connect_tree";
            rgb = {0.60f, 0.85f, 1.0f};
            alpha = 0.35f;
            break;
        }
    }

    if (!pub || pub->get_subscription_count() == 0)
    {
        return;
    }

    auto marker = tree_marker;
    marker.header.frame_id = world_frame_id_;
    marker.header.stamp = node_handle_->now();
    marker.ns = ns;
    setMarkerRgb(marker, rgb, alpha);
    pub->publish(marker);
}

void FFAPlannerNode::publishActiveFrustumVis(const openvdb::Vec3d &viewpoint,
                                             const openvdb::Vec3d &target,
                                             VisLayer layer,
                                             bool tight_primary)
{
    auto vp_pub = (layer == VisLayer::ROOT) ? pub_fov_viewpoint_vis_ : pub_subgoal_fov_viewpoint_vis_;
    auto fr_pub = (layer == VisLayer::ROOT) ? pub_fov_frustum_vis_ : pub_subgoal_fov_frustum_vis_;
    if (vp_pub->get_subscription_count() == 0 && fr_pub->get_subscription_count() == 0)
    {
        return;
    }

    const char *vp_ns = (layer == VisLayer::ROOT) ? "root_active_viewpoint" : "subgoal_active_viewpoint";
    const char *fr_ns = (layer == VisLayer::ROOT) ? "root_active_frustum" : "subgoal_active_frustum";
    const char *tgt_ns = (layer == VisLayer::ROOT) ? "root_active_target" : "subgoal_active_target";
    const Rgb viewpoint_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.82f, 0.10f} : Rgb{0.20f, 0.82f, 1.0f};
    const Rgb frustum_rgb = (layer == VisLayer::ROOT)
                                ? (tight_primary ? Rgb{0.10f, 0.92f, 0.18f} : Rgb{1.0f, 0.82f, 0.10f})
                                : Rgb{0.20f, 0.82f, 1.0f};
    const Rgb target_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.45f, 0.05f} : Rgb{0.10f, 0.55f, 1.0f};

    auto stamp = node_handle_->now();
    visualization_msgs::msg::MarkerArray vp_markers;
    visualization_msgs::msg::MarkerArray fr_markers;

    {
        visualization_msgs::msg::Marker m;
        m.header.frame_id = world_frame_id_;
        m.header.stamp = stamp;
        m.ns = vp_ns;
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = viewpoint.x();
        m.pose.position.y = viewpoint.y();
        m.pose.position.z = viewpoint.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.5;
        setMarkerRgb(m, viewpoint_rgb, 0.9f);
        vp_markers.markers.push_back(m);
    }

    const double dx = target.x() - viewpoint.x();
    const double dy = target.y() - viewpoint.y();
    const double dz = target.z() - viewpoint.z();
    const double d_xy = std::sqrt(dx * dx + dy * dy);
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    const double yaw = (d_xy > 1e-6) ? std::atan2(dy, dx) : 0.0;
    const double cos_y = std::cos(yaw);
    const double sin_y = std::sin(yaw);
    const double fov_len = std::min(dist, fov_cfg_.range);

    auto corner = [&](double theta) -> geometry_msgs::msg::Point
    {
        const double fwd = fov_len * std::cos(theta);
        const double up = fov_len * std::sin(theta);
        geometry_msgs::msg::Point p;
        p.x = viewpoint.x() + fwd * cos_y;
        p.y = viewpoint.y() + fwd * sin_y;
        p.z = viewpoint.z() + up;
        return p;
    };

    geometry_msgs::msg::Point origin;
    origin.x = viewpoint.x();
    origin.y = viewpoint.y();
    origin.z = viewpoint.z();

    geometry_msgs::msg::Point tgt;
    tgt.x = target.x();
    tgt.y = target.y();
    tgt.z = target.z();

    auto top = corner(fov_cfg_.theta_u);
    auto bot = corner(fov_cfg_.theta_d);

    {
        visualization_msgs::msg::Marker cone;
        cone.header.frame_id = world_frame_id_;
        cone.header.stamp = stamp;
        cone.ns = fr_ns;
        cone.id = 0;
        cone.type = visualization_msgs::msg::Marker::LINE_LIST;
        cone.action = visualization_msgs::msg::Marker::ADD;
        cone.pose.orientation.w = 1.0;
        cone.scale.x = 0.06;
        setMarkerRgb(cone, frustum_rgb, 0.75f);

        cone.points.push_back(origin);
        cone.points.push_back(top);
        cone.points.push_back(origin);
        cone.points.push_back(bot);
        cone.points.push_back(top);
        cone.points.push_back(bot);
        cone.points.push_back(origin);
        cone.points.push_back(tgt);
        fr_markers.markers.push_back(cone);
    }

    {
        visualization_msgs::msg::Marker ts;
        ts.header.frame_id = world_frame_id_;
        ts.header.stamp = stamp;
        ts.ns = tgt_ns;
        ts.id = 0;
        ts.type = visualization_msgs::msg::Marker::SPHERE;
        ts.action = visualization_msgs::msg::Marker::ADD;
        ts.pose.position = tgt;
        ts.pose.orientation.w = 1.0;
        ts.scale.x = ts.scale.y = ts.scale.z = 0.4;
        setMarkerRgb(ts, target_rgb, 0.9f);
        fr_markers.markers.push_back(ts);
    }

    if (vp_pub->get_subscription_count() > 0)
    {
        vp_pub->publish(vp_markers);
    }
    if (fr_pub->get_subscription_count() > 0)
    {
        fr_pub->publish(fr_markers);
    }
}

void FFAPlannerNode::publishFovPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                                       const std::vector<openvdb::Vec3d> &short_path,
                                       const openvdb::Vec3d &viewpoint,
                                       const openvdb::Vec3d &target,
                                       VisLayer layer,
                                       bool tight_primary)
{
    auto path_pub = (layer == VisLayer::ROOT) ? pub_fov_path_vis_ : pub_subgoal_fov_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_fov_short_vis_ : pub_subgoal_fov_short_vis_;
    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_fov_astar" : "subgoal_fov_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_fov_short" : "subgoal_fov_short";
    const Rgb raw_rgb = (layer == VisLayer::ROOT) ? Rgb{0.35f, 1.0f, 0.30f} : Rgb{0.30f, 0.95f, 1.0f};
    const Rgb short_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.90f, 0.10f} : Rgb{0.55f, 1.0f, 1.0f};

    auto stamp = node_handle_->now();

    if (!raw_path.empty() && path_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(raw_path, 0.4);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = raw_ns;
            setMarkerRgb(m, raw_rgb, 0.4f);
        }
        path_pub->publish(markers);
    }

    if (!short_path.empty() && short_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(short_path, 1.0);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = short_ns;
            setMarkerRgb(m, short_rgb, 1.0f);
        }
        short_pub->publish(markers);
    }

    publishActiveFrustumVis(viewpoint, target, layer, tight_primary);
}

void FFAPlannerNode::publishSecondaryFovPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                                                const std::vector<openvdb::Vec3d> &short_path,
                                                const openvdb::Vec3d &viewpoint,
                                                const openvdb::Vec3d &target,
                                                VisLayer layer)
{
    auto path_pub = (layer == VisLayer::ROOT) ? pub_fov_path_vis_ : pub_subgoal_fov_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_fov_short_vis_ : pub_subgoal_fov_short_vis_;
    auto vp_pub = (layer == VisLayer::ROOT) ? pub_fov_viewpoint_vis_ : pub_subgoal_fov_viewpoint_vis_;
    auto fr_pub = (layer == VisLayer::ROOT) ? pub_fov_frustum_vis_ : pub_subgoal_fov_frustum_vis_;

    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_lookahead_fov_astar" : "subgoal_lookahead_fov_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_lookahead_fov_short" : "subgoal_lookahead_fov_short";
    const char *vp_ns = (layer == VisLayer::ROOT) ? "root_lookahead_viewpoint" : "subgoal_lookahead_viewpoint";
    const char *fr_ns = (layer == VisLayer::ROOT) ? "root_lookahead_frustum" : "subgoal_lookahead_frustum";
    const char *tgt_ns = (layer == VisLayer::ROOT) ? "root_lookahead_target" : "subgoal_lookahead_target";
    const Rgb base_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.25f, 0.15f} : Rgb{0.15f, 0.95f, 0.55f};
    const Rgb raw_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.35f, 0.20f} : Rgb{0.15f, 0.95f, 0.65f};
    const Rgb short_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.65f, 0.25f} : Rgb{0.55f, 1.0f, 0.65f};
    const Rgb target_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.10f, 0.10f} : Rgb{0.10f, 1.0f, 0.45f};

    auto stamp = node_handle_->now();

    if (!raw_path.empty() && path_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(raw_path, 0.35);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = raw_ns;
            setMarkerRgb(m, raw_rgb, 0.45f);
        }
        path_pub->publish(markers);
    }

    if (!short_path.empty() && short_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(short_path, 0.9);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = short_ns;
            setMarkerRgb(m, short_rgb, 0.95f);
        }
        short_pub->publish(markers);
    }

    if (vp_pub->get_subscription_count() > 0)
    {
        visualization_msgs::msg::MarkerArray vp_markers;
        visualization_msgs::msg::Marker m;
        m.header.frame_id = world_frame_id_;
        m.header.stamp = stamp;
        m.ns = vp_ns;
        m.id = 0;
        m.type = visualization_msgs::msg::Marker::SPHERE;
        m.action = visualization_msgs::msg::Marker::ADD;
        m.pose.position.x = viewpoint.x();
        m.pose.position.y = viewpoint.y();
        m.pose.position.z = viewpoint.z();
        m.pose.orientation.w = 1.0;
        m.scale.x = m.scale.y = m.scale.z = 0.45;
        setMarkerRgb(m, base_rgb, 0.95f);
        vp_markers.markers.push_back(m);
        vp_pub->publish(vp_markers);
    }

    if (fr_pub->get_subscription_count() > 0)
    {
        visualization_msgs::msg::MarkerArray fr_markers;

        const double dx = target.x() - viewpoint.x();
        const double dy = target.y() - viewpoint.y();
        const double dz = target.z() - viewpoint.z();
        const double d_xy = std::sqrt(dx * dx + dy * dy);
        const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

        const double yaw = (d_xy > 1e-6) ? std::atan2(dy, dx) : 0.0;
        const double cos_y = std::cos(yaw);
        const double sin_y = std::sin(yaw);
        const double fov_len = std::min(dist, fov_cfg_.range);

        auto corner = [&](double theta) -> geometry_msgs::msg::Point
        {
            const double fwd = fov_len * std::cos(theta);
            const double up = fov_len * std::sin(theta);
            geometry_msgs::msg::Point p;
            p.x = viewpoint.x() + fwd * cos_y;
            p.y = viewpoint.y() + fwd * sin_y;
            p.z = viewpoint.z() + up;
            return p;
        };

        geometry_msgs::msg::Point origin;
        origin.x = viewpoint.x();
        origin.y = viewpoint.y();
        origin.z = viewpoint.z();

        geometry_msgs::msg::Point tgt;
        tgt.x = target.x();
        tgt.y = target.y();
        tgt.z = target.z();

        auto top = corner(fov_cfg_.theta_u);
        auto bot = corner(fov_cfg_.theta_d);

        visualization_msgs::msg::Marker cone;
        cone.header.frame_id = world_frame_id_;
        cone.header.stamp = stamp;
        cone.ns = fr_ns;
        cone.id = 0;
        cone.type = visualization_msgs::msg::Marker::LINE_LIST;
        cone.action = visualization_msgs::msg::Marker::ADD;
        cone.pose.orientation.w = 1.0;
        cone.scale.x = 0.05;
        setMarkerRgb(cone, base_rgb, 0.8f);
        cone.points.push_back(origin);
        cone.points.push_back(top);
        cone.points.push_back(origin);
        cone.points.push_back(bot);
        cone.points.push_back(top);
        cone.points.push_back(bot);
        cone.points.push_back(origin);
        cone.points.push_back(tgt);
        fr_markers.markers.push_back(cone);

        visualization_msgs::msg::Marker ts;
        ts.header.frame_id = world_frame_id_;
        ts.header.stamp = stamp;
        ts.ns = tgt_ns;
        ts.id = 0;
        ts.type = visualization_msgs::msg::Marker::SPHERE;
        ts.action = visualization_msgs::msg::Marker::ADD;
        ts.pose.position = tgt;
        ts.pose.orientation.w = 1.0;
        ts.scale.x = ts.scale.y = ts.scale.z = 0.34;
        setMarkerRgb(ts, target_rgb, 0.95f);
        fr_markers.markers.push_back(ts);

        fr_pub->publish(fr_markers);
    }
}

void FFAPlannerNode::publishCandidateSetVis(CandidateSetVisKind kind,
                                            const std::vector<SearchGoalCandidate> &certified_candidates,
                                            const std::vector<SearchGoalCandidate> &contaminated_candidates,
                                            const openvdb::Coord &selected_coord,
                                            bool has_selected,
                                            ReachLabel selected_label)
{
    auto pub = (kind == CandidateSetVisKind::FOV) ? pub_fov_candidate_set_vis_ : pub_observe_candidate_set_vis_;
    if (!pub)
    {
        return;
    }

    const char *ns = (kind == CandidateSetVisKind::FOV) ? "fov_candidate_set" : "observe_candidate_set";
    const auto stamp = node_handle_->now();
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(makeDeleteAllMarker(world_frame_id_, stamp, ns));

    auto grid_tf = map_manager_->get_grid_transform();
    auto add_sphere_list = [&](const std::vector<SearchGoalCandidate> &candidates,
                               int id,
                               const Rgb &rgb,
                               float alpha,
                               double scale)
    {
        if (candidates.empty())
        {
            return;
        }

        visualization_msgs::msg::Marker cloud;
        cloud.header.frame_id = world_frame_id_;
        cloud.header.stamp = stamp;
        cloud.ns = ns;
        cloud.id = id;
        cloud.type = visualization_msgs::msg::Marker::SPHERE_LIST;
        cloud.action = visualization_msgs::msg::Marker::ADD;
        cloud.pose.orientation.w = 1.0;
        cloud.scale.x = cloud.scale.y = cloud.scale.z = scale;
        setMarkerRgb(cloud, rgb, alpha);

        cloud.points.reserve(candidates.size());
        for (const auto &candidate : candidates)
        {
            const openvdb::Vec3d p = grid_tf->indexToWorld(candidate.coord);
            geometry_msgs::msg::Point ros_p;
            ros_p.x = p.x();
            ros_p.y = p.y();
            ros_p.z = p.z();
            cloud.points.push_back(ros_p);
        }

        markers.markers.push_back(cloud);
    };

    bool selected_set_is_certified = !certified_candidates.empty();
    if (kind == CandidateSetVisKind::OBSERVE)
    {
        add_sphere_list(certified_candidates, 0, Rgb{0.05f, 0.85f, 1.00f}, 0.95f, 0.18);
        add_sphere_list(contaminated_candidates, 1, Rgb{0.90f, 0.35f, 1.00f}, 0.78f, 0.18);
        selected_set_is_certified = (selected_label == ReachLabel::CERTIFIED);
    }
    else
    {
        bool show_certified = !certified_candidates.empty();
        if (has_selected)
        {
            if (selected_label == ReachLabel::CONTAMINATED && !contaminated_candidates.empty())
            {
                show_certified = false;
            }
            else if (selected_label == ReachLabel::CERTIFIED && !certified_candidates.empty())
            {
                show_certified = true;
            }
        }
        const auto &visible_candidates = show_certified ? certified_candidates : contaminated_candidates;
        add_sphere_list(visible_candidates,
                        0,
                        show_certified ? Rgb{0.10f, 0.95f, 0.25f} : Rgb{1.00f, 0.22f, 0.10f},
                        show_certified ? 0.85f : 0.70f,
                        0.12);
        selected_set_is_certified = show_certified;
    }

    if (markers.markers.size() == 1)
    {
        clearCandidateSetVis(kind);
        return;
    }

    if (has_selected)
    {
        const openvdb::Vec3d selected = grid_tf->indexToWorld(selected_coord);
        visualization_msgs::msg::Marker chosen;
        chosen.header.frame_id = world_frame_id_;
        chosen.header.stamp = stamp;
        chosen.ns = ns;
        chosen.id = (kind == CandidateSetVisKind::OBSERVE) ? 2 : 1;
        chosen.type = visualization_msgs::msg::Marker::SPHERE;
        chosen.action = visualization_msgs::msg::Marker::ADD;
        chosen.pose.position.x = selected.x();
        chosen.pose.position.y = selected.y();
        chosen.pose.position.z = selected.z();
        chosen.pose.orientation.w = 1.0;
        chosen.scale.x = chosen.scale.y = chosen.scale.z =
            (kind == CandidateSetVisKind::OBSERVE) ? 0.34 : 0.26;
        const Rgb selected_rgb =
            (kind == CandidateSetVisKind::OBSERVE)
                ? Rgb{1.00f, 1.00f, 1.00f}
                : (selected_set_is_certified ? Rgb{0.00f, 0.85f, 1.00f} : Rgb{0.80f, 0.20f, 1.00f});
        setMarkerRgb(chosen, selected_rgb, 1.0f);
        markers.markers.push_back(chosen);
    }

    pub->publish(markers);
}

void FFAPlannerNode::clearCandidateSetVis(CandidateSetVisKind kind)
{
    auto pub = (kind == CandidateSetVisKind::FOV) ? pub_fov_candidate_set_vis_ : pub_observe_candidate_set_vis_;
    if (!pub)
    {
        return;
    }

    const char *ns = (kind == CandidateSetVisKind::FOV) ? "fov_candidate_set" : "observe_candidate_set";
    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(makeDeleteAllMarker(world_frame_id_, node_handle_->now(), ns));
    pub->publish(markers);
}

void FFAPlannerNode::clearSecondaryFovPlanVis(VisLayer layer)
{
    const auto stamp = node_handle_->now();
    auto path_pub = (layer == VisLayer::ROOT) ? pub_fov_path_vis_ : pub_subgoal_fov_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_fov_short_vis_ : pub_subgoal_fov_short_vis_;
    auto vp_pub = (layer == VisLayer::ROOT) ? pub_fov_viewpoint_vis_ : pub_subgoal_fov_viewpoint_vis_;
    auto fr_pub = (layer == VisLayer::ROOT) ? pub_fov_frustum_vis_ : pub_subgoal_fov_frustum_vis_;

    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_lookahead_fov_astar" : "subgoal_lookahead_fov_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_lookahead_fov_short" : "subgoal_lookahead_fov_short";
    const char *vp_ns = (layer == VisLayer::ROOT) ? "root_lookahead_viewpoint" : "subgoal_lookahead_viewpoint";
    const char *fr_ns = (layer == VisLayer::ROOT) ? "root_lookahead_frustum" : "subgoal_lookahead_frustum";
    const char *tgt_ns = (layer == VisLayer::ROOT) ? "root_lookahead_target" : "subgoal_lookahead_target";

    if (path_pub)
    {
        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, raw_ns, 0));
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, raw_ns, 1));
        path_pub->publish(arr);
    }
    if (short_pub)
    {
        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, short_ns, 0));
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, short_ns, 1));
        short_pub->publish(arr);
    }
    if (vp_pub)
    {
        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, vp_ns, 0));
        vp_pub->publish(arr);
    }
    if (fr_pub)
    {
        visualization_msgs::msg::MarkerArray arr;
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, fr_ns, 0));
        arr.markers.push_back(makeDeleteMarker(world_frame_id_, stamp, tgt_ns, 0));
        fr_pub->publish(arr);
    }
}

void FFAPlannerNode::publishObservePlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                                           const std::vector<openvdb::Vec3d> &short_path,
                                           const openvdb::Vec3d &viewpoint,
                                           const openvdb::Vec3d &target,
                                           VisLayer layer)
{
    auto path_pub = (layer == VisLayer::ROOT) ? pub_observe_path_vis_ : pub_subgoal_observe_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_observe_short_vis_ : pub_subgoal_observe_short_vis_;
    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_observe_astar" : "subgoal_observe_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_observe_short" : "subgoal_observe_short";
    const Rgb raw_rgb = (layer == VisLayer::ROOT) ? Rgb{0.20f, 1.0f, 0.60f} : Rgb{0.15f, 0.95f, 0.80f};
    const Rgb short_rgb = (layer == VisLayer::ROOT) ? Rgb{1.0f, 0.55f, 0.05f} : Rgb{0.55f, 0.95f, 1.0f};

    auto stamp = node_handle_->now();

    if (!raw_path.empty() && path_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(raw_path, 0.5);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = raw_ns;
            setMarkerRgb(m, raw_rgb, 0.5f);
        }
        path_pub->publish(markers);
    }

    if (!short_path.empty() && short_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(short_path, 1.0);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = short_ns;
            setMarkerRgb(m, short_rgb, 1.0f);
        }
        short_pub->publish(markers);
    }

    publishObserveFrustumVis(viewpoint, target, layer);
}

void FFAPlannerNode::publishObserveFrustumVis(const openvdb::Vec3d &viewpoint,
                                              const openvdb::Vec3d &target,
                                              VisLayer layer)
{
    if (!pub_observe_frustum_vis_ || pub_observe_frustum_vis_->get_subscription_count() == 0)
    {
        return;
    }

    const char *vp_ns = (layer == VisLayer::ROOT) ? "root_observe_viewpoint" : "subgoal_observe_viewpoint";
    const char *fr_ns = (layer == VisLayer::ROOT) ? "root_observe_frustum" : "subgoal_observe_frustum";
    const char *tgt_ns = (layer == VisLayer::ROOT) ? "root_observe_target" : "subgoal_observe_target";
    const Rgb viewpoint_rgb = (layer == VisLayer::ROOT) ? Rgb{0.65f, 0.25f, 1.00f} : Rgb{0.45f, 0.30f, 1.00f};
    const Rgb frustum_rgb = (layer == VisLayer::ROOT) ? Rgb{0.72f, 0.28f, 1.00f} : Rgb{0.55f, 0.35f, 1.00f};
    const Rgb target_rgb = (layer == VisLayer::ROOT) ? Rgb{0.10f, 0.45f, 1.00f} : Rgb{0.20f, 0.35f, 1.00f};

    const auto stamp = node_handle_->now();
    visualization_msgs::msg::MarkerArray markers;

    visualization_msgs::msg::Marker vp;
    vp.header.frame_id = world_frame_id_;
    vp.header.stamp = stamp;
    vp.ns = vp_ns;
    vp.id = 0;
    vp.type = visualization_msgs::msg::Marker::SPHERE;
    vp.action = visualization_msgs::msg::Marker::ADD;
    vp.pose.position.x = viewpoint.x();
    vp.pose.position.y = viewpoint.y();
    vp.pose.position.z = viewpoint.z();
    vp.pose.orientation.w = 1.0;
    vp.scale.x = vp.scale.y = vp.scale.z = 0.46;
    setMarkerRgb(vp, viewpoint_rgb, 0.95f);
    markers.markers.push_back(vp);

    const double dx = target.x() - viewpoint.x();
    const double dy = target.y() - viewpoint.y();
    const double dz = target.z() - viewpoint.z();
    const double d_xy = std::sqrt(dx * dx + dy * dy);
    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);

    const double yaw = (d_xy > 1e-6) ? std::atan2(dy, dx) : 0.0;
    const double cos_y = std::cos(yaw);
    const double sin_y = std::sin(yaw);
    const double fov_len = std::min(dist, fov_cfg_.range);

    auto corner = [&](double theta) -> geometry_msgs::msg::Point
    {
        const double fwd = fov_len * std::cos(theta);
        const double up = fov_len * std::sin(theta);
        geometry_msgs::msg::Point p;
        p.x = viewpoint.x() + fwd * cos_y;
        p.y = viewpoint.y() + fwd * sin_y;
        p.z = viewpoint.z() + up;
        return p;
    };

    geometry_msgs::msg::Point origin;
    origin.x = viewpoint.x();
    origin.y = viewpoint.y();
    origin.z = viewpoint.z();

    geometry_msgs::msg::Point tgt;
    tgt.x = target.x();
    tgt.y = target.y();
    tgt.z = target.z();

    const auto top = corner(fov_cfg_.theta_u);
    const auto bot = corner(fov_cfg_.theta_d);

    visualization_msgs::msg::Marker cone;
    cone.header.frame_id = world_frame_id_;
    cone.header.stamp = stamp;
    cone.ns = fr_ns;
    cone.id = 0;
    cone.type = visualization_msgs::msg::Marker::LINE_LIST;
    cone.action = visualization_msgs::msg::Marker::ADD;
    cone.pose.orientation.w = 1.0;
    cone.scale.x = 0.055;
    setMarkerRgb(cone, frustum_rgb, 0.85f);
    cone.points.push_back(origin);
    cone.points.push_back(top);
    cone.points.push_back(origin);
    cone.points.push_back(bot);
    cone.points.push_back(top);
    cone.points.push_back(bot);
    cone.points.push_back(origin);
    cone.points.push_back(tgt);
    markers.markers.push_back(cone);

    visualization_msgs::msg::Marker ts;
    ts.header.frame_id = world_frame_id_;
    ts.header.stamp = stamp;
    ts.ns = tgt_ns;
    ts.id = 0;
    ts.type = visualization_msgs::msg::Marker::SPHERE;
    ts.action = visualization_msgs::msg::Marker::ADD;
    ts.pose.position = tgt;
    ts.pose.orientation.w = 1.0;
    ts.scale.x = ts.scale.y = ts.scale.z = 0.34;
    setMarkerRgb(ts, target_rgb, 0.95f);
    markers.markers.push_back(ts);

    pub_observe_frustum_vis_->publish(markers);
}

void FFAPlannerNode::clearObserveFrustumVis()
{
    if (!pub_observe_frustum_vis_)
    {
        return;
    }

    visualization_msgs::msg::MarkerArray markers;
    markers.markers.push_back(makeDeleteAllMarker(world_frame_id_,
                                                  node_handle_->now(),
                                                  "observe_frustum"));
    pub_observe_frustum_vis_->publish(markers);
}

void FFAPlannerNode::publishConnectPlanVis(const std::vector<openvdb::Vec3d> &raw_path,
                                           const std::vector<openvdb::Vec3d> &short_path,
                                           VisLayer layer)
{
    auto path_pub = (layer == VisLayer::ROOT) ? pub_connect_path_vis_ : pub_subgoal_connect_path_vis_;
    auto short_pub = (layer == VisLayer::ROOT) ? pub_connect_short_vis_ : pub_subgoal_connect_short_vis_;
    const char *raw_ns = (layer == VisLayer::ROOT) ? "root_connect_astar" : "subgoal_connect_astar";
    const char *short_ns = (layer == VisLayer::ROOT) ? "root_connect_short" : "subgoal_connect_short";
    const Rgb raw_rgb = (layer == VisLayer::ROOT) ? Rgb{0.20f, 0.70f, 1.0f} : Rgb{0.70f, 0.70f, 1.0f};
    const Rgb short_rgb = (layer == VisLayer::ROOT) ? Rgb{0.00f, 0.90f, 1.0f} : Rgb{0.95f, 0.70f, 1.0f};

    auto stamp = node_handle_->now();

    if (!raw_path.empty() && path_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(raw_path, 0.45);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = raw_ns;
            setMarkerRgb(m, raw_rgb, 0.45f);
        }
        path_pub->publish(markers);
    }

    if (!short_path.empty() && short_pub->get_subscription_count() > 0)
    {
        auto markers = path_to_marker_array(short_path, 1.0);
        for (auto &m : markers.markers)
        {
            m.header.frame_id = world_frame_id_;
            m.header.stamp = stamp;
            m.ns = short_ns;
            setMarkerRgb(m, short_rgb, 1.0f);
        }
        short_pub->publish(markers);
    }
}
