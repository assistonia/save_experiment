#!/bin/bash
# Warehouse PedSim + Gazebo + Robot Simulation (with Navigation)
# This is a SEPARATE version - does not affect the pedestrian-only version

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(dirname "$SCRIPT_DIR")/navigation"

# Use different container name to avoid conflict
CONTAINER_NAME="gdae_pedsim_robot"

# Stop and remove existing container (only this one)
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

# Run container with GUI support (using navigation-enabled image)
docker run -it \
  --name $CONTAINER_NAME \
  --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ${SCRIPT_DIR}:/environment:rw \
  -v ${NAV_DIR}:/navigation:rw \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:latest \
  bash -c "
    # Copy configuration files to ROS packages
    cp /environment/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # Launch simulation with robot
    roslaunch pedsim_simulator warehouse_with_robot.launch scene_file:=/environment/scenarios/scenario_block_heavy.xml world_file:=/environment/worlds/warehouse.world
  "
