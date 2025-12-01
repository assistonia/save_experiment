#!/bin/bash
# Warehouse PedSim + Gazebo Simulation Runner
# Usage: ./run_simulation.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stop and remove existing container
docker stop gdae_pedsim 2>/dev/null
docker rm gdae_pedsim 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

# Run container with GUI support
docker run -it \
  --name gdae_pedsim \
  --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ${SCRIPT_DIR}:/environment:rw \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  11namminseok/gdae:latest \
  bash -c "
    # Copy configuration files to ROS packages
    cp /environment/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/launch/warehouse_simulation.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # Launch simulation
    roslaunch pedsim_simulator warehouse_simulation.launch scene_file:=/environment/scenarios/warehouse_pedsim.xml world_file:=/environment/worlds/warehouse.world
  "
