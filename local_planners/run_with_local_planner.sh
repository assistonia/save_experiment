#!/bin/bash
# Warehouse PedSim + CIGP + Local Planner 통합 실행
# 기존 환경 코드 수정 없이 독립적으로 로컬 플래너 테스트
#
# Usage:
#   ./run_with_local_planner.sh [planner] [scenario]
#
# Examples:
#   ./run_with_local_planner.sh dwa
#   ./run_with_local_planner.sh drl_vo scenario_block_heavy.xml
#   ./run_with_local_planner.sh orca scenario_congestion_all.xml
#   ./run_with_local_planner.sh sfm scenario_busy_warehouse.xml
#   ./run_with_local_planner.sh teb

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"
WITH_ROBOT_DIR="${ENV_DIR}/with_robot"

# 인자 처리
PLANNER=${1:-"dwa"}
SCENARIO=${2:-"warehouse_pedsim.xml"}

CONTAINER_NAME="gdae_pedsim_local_planner"

# Stop and remove existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "=================================================="
echo " Warehouse + CIGP + Local Planner"
echo "=================================================="
echo ""
echo " Planner:  $PLANNER"
echo " Scenario: $SCENARIO"
echo ""
echo " Available planners:"
echo "   dwa   - Dynamic Window Approach"
echo "   drl_vo - DRL-VO (PPO + ResNet)"
echo "   orca  - Optimal Reciprocal Collision Avoidance"
echo "   sfm   - Social Force Model"
echo "   teb   - Timed Elastic Band"
echo ""
echo " Topics:"
echo "   /cigp/global_path    - Global path from CIGP"
echo "   /cigp/next_waypoint  - Next waypoint"
echo "   /p3dx/cmd_vel        - Velocity command (output)"
echo ""
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
  -v ${ENV_DIR}:/environment:rw \
  -v ${NAV_DIR}:/navigation:rw \
  -v ${SICNAV_DIR}:/sicnav-test:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:latest \
  bash -c "
    # Copy configuration files
    cp /environment/with_robot/scenarios/${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # PYTHONPATH
    export PYTHONPATH=/sicnav-test:/environment/local_planners:\${PYTHONPATH}

    echo ''
    echo 'Starting simulation...'
    echo ''

    # 1. Launch PedSim + Gazebo (background)
    roslaunch pedsim_simulator warehouse_with_robot.launch \
      scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
      world_file:=/environment/with_robot/worlds/warehouse.world &

    sleep 12

    echo ''
    echo '============================================'
    echo 'Starting CIGP Global Planner...'
    echo '============================================'
    echo ''

    # 2. Start CIGP Bridge Node (background)
    python3 /environment/cigp_integration/cigp_bridge_node.py &

    sleep 5

    echo ''
    echo '============================================'
    echo 'Starting Local Planner: ${PLANNER}'
    echo '============================================'
    echo ''

    # 3. Start Local Planner (foreground)
    python3 /environment/local_planners/local_planner_node.py _planner:=${PLANNER}
  "
