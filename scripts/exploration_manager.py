#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Carnegie Mellon University
"""Frontier exploration manager for SCOPE.

Drives the planner goal by goal from the ranked frontier viewpoints that
vdb_edt publishes:

    /frontier_ranked_viewpoints (PoseArray, best viewpoint per cluster, score order)
        -> pick the best one that is not blacklisted
        -> /goal_point (Pose, latched)
        -> watch /goal_reach_status (Int8: 0 in progress, 1 reached, 2 failed)
        -> reached: next; failed / timed out: blacklist it, next

A std_srvs/Trigger service (~/toggle) switches exploration on and off; in
AirStack it is remapped to behavior/global_plan_toggle. Exploration ends by
itself when no viewpoint has been available for `done_grace_s`.
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Int8, String
from std_srvs.srv import Trigger

IN_PROGRESS, REACHED, FAILED = 0, 1, 2


def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class ExplorationManager(Node):
    def __init__(self):
        super().__init__("exploration_manager")
        p = self.declare_parameters("", [
            ("ranked_topic", "/frontier_ranked_viewpoints"),
            ("goal_topic", "/goal_point"),
            ("status_topic", "/goal_reach_status"),
            ("auto_start", False),
            ("blacklist_radius_m", 1.0),   # a failed viewpoint blocks candidates this close
            ("blacklist_ttl_s", 120.0),    # ... for this long
            ("accept_timeout_s", 15.0),    # planner never reported IN_PROGRESS -> count as failed
            ("goal_timeout_s", 120.0),     # goal not reached in time -> count as failed
            ("status_settle_s", 3.0),      # a REACHED/FAILED older than this after the send is ours even without a 0
            ("retarget_if_gone", True),    # drop the goal when its viewpoint left the ranked list
            ("gone_match_radius_m", 1.0),
            ("gone_confirmations", 5),
            ("done_grace_s", 20.0),        # no viewpoints for this long -> exploration complete
            ("tick_period_s", 0.5),
            # exploration bounds (world frame): viewpoints outside are never chosen
            ("bounds_min", [-1.0e9, -1.0e9, -1.0e9]),
            ("bounds_max", [1.0e9, 1.0e9, 1.0e9]),
        ])
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.goal_pub = self.create_publisher(Pose, g("goal_topic"), latched)
        self.state_pub = self.create_publisher(String, "~/state", 1)
        self.current_goal_pub = self.create_publisher(PoseStamped, "~/current_goal", 1)
        self.create_subscription(PoseArray, g("ranked_topic"), self._on_ranked, 5)
        self.create_subscription(Int8, g("status_topic"), self._on_status, 10)
        self.create_service(Trigger, "~/toggle", self._on_toggle)

        self.blacklist_radius = float(g("blacklist_radius_m"))
        self.blacklist_ttl = float(g("blacklist_ttl_s"))
        self.accept_timeout = float(g("accept_timeout_s"))
        self.goal_timeout = float(g("goal_timeout_s"))
        self.status_settle = float(g("status_settle_s"))
        self.retarget_if_gone = bool(g("retarget_if_gone"))
        self.gone_radius = float(g("gone_match_radius_m"))
        self.gone_confirmations = int(g("gone_confirmations"))
        self.done_grace = float(g("done_grace_s"))
        self.bounds_min = [float(v) for v in g("bounds_min")]
        self.bounds_max = [float(v) for v in g("bounds_max")]

        self.exploring = bool(g("auto_start"))
        self.ranked = []                 # latest PoseArray poses
        self.ranked_stamp = None
        self.last_candidate_stamp = None   # last time an in-bounds, non-blacklisted viewpoint existed
        self.goal = None                 # Pose currently sent
        self.goal_sent_at = None
        self.saw_in_progress = False
        self.gone_count = 0
        self.blacklist = []              # [(Pose, expiry_time)]
        self.stats = {"sent": 0, "reached": 0, "failed": 0, "timeout": 0, "retargeted": 0}
        self.last_status = None
        self.last_status_at = None

        self.create_timer(float(g("tick_period_s")), self._tick)
        self.get_logger().info("exploration manager ready (%s)" % ("exploring" if self.exploring else "idle"))

    # ---------------------------------------------------------------- inputs
    def _on_ranked(self, msg: PoseArray):
        self.ranked = list(msg.poses)
        self.ranked_stamp = time.monotonic()

    def _on_status(self, msg: Int8):
        self.last_status = int(msg.data)
        self.last_status_at = time.monotonic()
        if self.goal is not None and self.last_status == IN_PROGRESS:
            self.saw_in_progress = True

    def _on_toggle(self, _req, resp):
        self.exploring = not self.exploring
        if self.exploring:
            self.last_candidate_stamp = time.monotonic()  # fresh grace period
        resp.success = True
        resp.message = "exploration %s" % ("on" if self.exploring else "off")
        self.get_logger().info(resp.message)
        return resp

    # ------------------------------------------------------------------ loop
    def _tick(self):
        now = time.monotonic()
        self.blacklist = [(p, t) for p, t in self.blacklist if t > now]

        if not self.exploring:
            self._publish_state("idle")
            return

        if self.goal is not None:
            self._check_goal(now)

        if self.goal is None:
            self._pick_next(now)

        self._publish_state("exploring" if self.goal is not None else "waiting")

    def _check_goal(self, now):
        elapsed = now - self.goal_sent_at
        status_is_ours = self.last_status_at is not None and self.last_status_at > self.goal_sent_at and (
            self.saw_in_progress or elapsed > self.status_settle)

        if status_is_ours and self.last_status == REACHED:
            self.stats["reached"] += 1
            self.get_logger().info("goal reached after %.0f s" % elapsed)
            self._clear_goal()
            return
        if status_is_ours and self.last_status == FAILED:
            self.stats["failed"] += 1
            self.get_logger().warn("planner reported FAILED; blacklisting viewpoint")
            self._blacklist(self.goal, now)
            self._clear_goal()
            return
        if not self.saw_in_progress and elapsed > self.accept_timeout:
            self.stats["failed"] += 1
            self.get_logger().warn("planner never accepted the goal; blacklisting viewpoint")
            self._blacklist(self.goal, now)
            self._clear_goal()
            return
        if elapsed > self.goal_timeout:
            self.stats["timeout"] += 1
            self.get_logger().warn("goal timed out after %.0f s; blacklisting viewpoint" % elapsed)
            self._blacklist(self.goal, now)
            self._clear_goal()
            return
        if self.retarget_if_gone and self.ranked_stamp is not None and self.ranked_stamp > self.goal_sent_at:
            still_there = any(dist(p.position, self.goal.position) < self.gone_radius for p in self.ranked)
            self.gone_count = 0 if still_there else self.gone_count + 1
            if self.gone_count >= self.gone_confirmations:
                self.stats["retargeted"] += 1
                self.get_logger().info("viewpoint no longer offered (frontier observed); retargeting")
                self._clear_goal()

    def _pick_next(self, now):
        candidates = [p for p in self.ranked
                      if self._in_bounds(p.position)
                      and not any(dist(p.position, b.position) < self.blacklist_radius for b, _ in self.blacklist)]
        if candidates:
            self.last_candidate_stamp = now
        if not candidates:
            if self.last_candidate_stamp is None or now - self.last_candidate_stamp > self.done_grace:
                self.exploring = False
                self.get_logger().info("no frontier viewpoints for %.0f s: exploration complete (%s)"
                                       % (self.done_grace, self.stats))
                self._publish_state("done")
            return
        best = candidates[0]
        self.goal = best
        self.goal_sent_at = now
        self.saw_in_progress = False
        self.gone_count = 0
        self.stats["sent"] += 1
        self.goal_pub.publish(best)
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "map"
        ps.pose = best
        self.current_goal_pub.publish(ps)
        self.get_logger().info("goal #%d -> (%.2f, %.2f, %.2f), %d candidates, %d blacklisted"
                               % (self.stats["sent"], best.position.x, best.position.y, best.position.z,
                                  len(candidates), len(self.blacklist)))

    def _in_bounds(self, pos):
        return (self.bounds_min[0] <= pos.x <= self.bounds_max[0] and
                self.bounds_min[1] <= pos.y <= self.bounds_max[1] and
                self.bounds_min[2] <= pos.z <= self.bounds_max[2])

    def _blacklist(self, pose, now):
        self.blacklist.append((pose, now + self.blacklist_ttl))

    def _clear_goal(self):
        self.goal = None
        self.goal_sent_at = None
        self.saw_in_progress = False
        self.gone_count = 0

    def _publish_state(self, state):
        msg = String()
        msg.data = "%s sent=%d reached=%d failed=%d timeout=%d retargeted=%d" % (
            state, self.stats["sent"], self.stats["reached"], self.stats["failed"],
            self.stats["timeout"], self.stats["retargeted"])
        self.state_pub.publish(msg)


def main():
    rclpy.init()
    node = ExplorationManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
