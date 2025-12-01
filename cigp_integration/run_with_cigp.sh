#!/bin/bash
# Warehouse PedSim + Robot + CIGP Global Planner
# 기존 환경 코드를 수정하지 않고 CIGP 플래너를 추가로 실행

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"

CONTAINER_NAME="gdae_pedsim_cigp"

# Stop and remove existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "=================================================="
echo " Warehouse Simulation with CIGP Global Planner"
echo "=================================================="
echo ""
echo "ROS Topics published by CIGP:"
echo "  /cigp/global_path      - Human-aware global path"
echo "  /cigp/next_waypoint    - Next waypoint for local planner"
echo "  /cigp/social_cost_map  - Social cost visualization"
echo ""
echo "Subscribe to /move_base_simple/goal to set navigation goal"
echo ""

# Run container
docker run -it \
  --name $CONTAINER_NAME \
  --gpus all \
  --privileged \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONPATH="/sicnav-test:${PYTHONPATH}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v ${ENV_DIR}:/environment:rw \
  -v ${NAV_DIR}:/navigation:rw \
  -v ${SCRIPT_DIR}:/cigp_integration:rw \
  -v ${SICNAV_DIR}:/sicnav-test:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:latest \
  bash -c "
    # Copy configuration files to ROS packages
    cp /environment/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # Set PYTHONPATH for CIGP
    export PYTHONPATH=/sicnav-test:\${PYTHONPATH}

    # Launch simulation with robot (background)
    echo 'Starting PedSim + Gazebo simulation...'
    roslaunch pedsim_simulator warehouse_with_robot.launch \
      scene_file:=/environment/scenarios/warehouse_pedsim.xml \
      world_file:=/environment/worlds/warehouse.world &

    # Wait for simulation to start
    sleep 10

    # Start CIGP Bridge Node
    echo 'Starting CIGP Global Planner...'
    python3 /cigp_integration/cigp_bridge_node.py
  "
