#!/bin/bash
# Warehouse PedSim + Robot + CIGP Global Planner
# 가제보에서 사람 위치를 받아서 CIGP로 글로벌 패스 생성

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAV_DIR="$(dirname "$SCRIPT_DIR")/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"

CONTAINER_NAME="gdae_pedsim_cigp"

# 시나리오 선택 (기본: scenario_block_heavy.xml)
SCENARIO=${1:-"scenario_block_heavy.xml"}

# Stop and remove existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "=================================================="
echo " Warehouse + Robot + CIGP Global Planner"
echo "=================================================="
echo ""
echo " Scenario: $SCENARIO"
echo ""
echo " CIGP Topics:"
echo "   /cigp/global_path     - Human-aware path (Path)"
echo "   /cigp/waypoints       - RViz markers (MarkerArray)"
echo "   /cigp/costmap_image   - Visualization image"
echo ""
echo " Set goal (use /cigp/goal, NOT /move_base_simple/goal):"
echo "   rostopic pub -1 /cigp/goal geometry_msgs/PoseStamped \\"
echo "     '{header: {frame_id: \"odom\"}, pose: {position: {x: 9, y: -9, z: 0}, orientation: {w: 1}}}'"
echo ""
echo " Logs saved to: /environment/cigp_logs/"
echo " Costmap image: /environment/cigp_logs/costmap_latest.png"
echo "=================================================="

# Run container
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
  -v ${SICNAV_DIR}:/sicnav-test:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:latest \
  bash -c "
    # Copy configuration files
    cp /environment/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # PYTHONPATH for CIGP
    export PYTHONPATH=/sicnav-test:\${PYTHONPATH}

    # Create log directory
    mkdir -p /environment/cigp_logs

    echo ''
    echo 'Starting simulation...'
    echo ''

    # Launch simulation (background)
    roslaunch pedsim_simulator warehouse_with_robot.launch \
      scene_file:=/environment/scenarios/${SCENARIO} \
      world_file:=/environment/worlds/warehouse.world &

    # Wait for simulation to fully start
    sleep 10

    echo ''
    echo '============================================'
    echo 'Starting cmd_vel inverter...'
    echo '============================================'
    echo ''

    # Start cmd_vel inverter (fixes left/right reversal)
    python3 /navigation/scripts/cmd_vel_inverter.py &
    sleep 2

    echo ''
    echo '============================================'
    echo 'Starting move_base (DWA local planner)...'
    echo '============================================'
    echo ''

    # Launch move_base for DWA local planning (background)
    roslaunch /navigation/launch/move_base.launch &
    sleep 5

    echo ''
    echo '============================================'
    echo 'Starting CIGP Global Planner...'
    echo '============================================'
    echo ''
    echo 'CIGP will plan human-aware path and send waypoints to move_base'
    echo ''

    # Run CIGP planner (foreground)
    python3 /environment/cigp_global_planner.py
  "
