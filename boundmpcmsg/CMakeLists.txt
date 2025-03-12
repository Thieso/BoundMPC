cmake_minimum_required(VERSION 3.10.0)

# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

project(bound_mpc_msg)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Vector.msg"
  "msg/MPCData.msg"
  "msg/HandoverLocation.msg"
  "msg/HandoverData.msg"
  "msg/MujocoData.msg"
  "srv/MPCParams.srv"
  "srv/Trajectory.srv"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

## Installation
ament_package()
