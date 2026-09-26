cmake_minimum_required(VERSION 3.5)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules" ${CMAKE_MODULE_PATH})
project(ffa_planner)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(OpenVDB REQUIRED)
find_package(vdb_edt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)
find_package(pid_path_tracker REQUIRED)

set(FFA_ROS_DEPS
  rclcpp
  std_msgs
  std_srvs
  sensor_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
  trajectory_msgs
  tf2
  tf2_ros
  pcl_conversions
  tf2_eigen
  tf2_geometry_msgs
  message_filters
  vdb_edt
)

###########
## Build ##
###########

# Trajectory optimizer backend (MINCO / GCOPTER-style header-only third-party
# code under include/traj_optimizer/ plus our wrapper in src/traj_optimizer.cpp)
add_library(traj_optimizer_lib
  src/traj_optimizer.cpp)

target_include_directories(traj_optimizer_lib
  PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(traj_optimizer_lib
  OpenVDB::openvdb
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES})

ament_target_dependencies(traj_optimizer_lib ${FFA_ROS_DEPS})

# The SCOPE / FFA planner node
add_executable(ffa_planner_node
  src/ffa_planner_node.cpp
  src/visibility_cell_builder.cpp
  src/vis_tools.cpp
  src/astar_vdb.cpp
  src/connect_astar.cpp
  src/fov_astar.cpp
  src/observe_astar.cpp
  src/occlusion_analyzer.cpp)

target_include_directories(ffa_planner_node
  PRIVATE
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(ffa_planner_node
  PRIVATE
  FFA_PLANNER_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(ffa_planner_node
  traj_optimizer_lib
  OpenVDB::openvdb
  ${EIGEN3_LIBRARIES}
  ${PCL_LIBRARIES})

ament_target_dependencies(ffa_planner_node ${FFA_ROS_DEPS} pid_path_tracker)

#############
## Install ##
#############

install(TARGETS
  ffa_planner_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/exploration_manager.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
