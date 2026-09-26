# scope-planner

SCOPE: a perception-aware point-to-point planner for aerial robots on a VDB
occupancy map. ROS 2 package `ffa_planner`.

- `ffa_planner_node`: takes a goal on `/goal_point`, publishes the trajectory
  on `/cmd_trajectory` and progress on `/goal_reach_status`.
- `exploration_manager.py`: optional frontier exploration, feeds the planner
  viewpoints from `vdb_edt`; start and stop with `~/toggle`.

## License

BSD 3-Clause. `include/traj_optimizer/` is adapted from
[GCOPTER](https://github.com/ZJU-FAST-Lab/GCOPTER) (MIT) and keeps its
original notices.
