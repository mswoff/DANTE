cmake_minimum_required(VERSION 2.8.3)
project(fformation_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  message_generation
  visualization_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

#add_service_files(
#  FILES
#)

add_message_files(
  FILES
  TrackedPersonMarkers.msg
  TrackedPersonsMarkers.msg
  Group.msg
  Groups.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  visualization_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
   CATKIN_DEPENDS message_runtime std_msgs sensor_msgs
)

