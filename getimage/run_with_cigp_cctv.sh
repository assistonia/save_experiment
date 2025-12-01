#!/bin/bash
# Warehouse PedSim + Robot + CIGP Global Planner (CCTV Vision-based)
# CCTV 이미지에서 사람 감지 → CIGP로 Human-aware 경로 생성
#
# 기존 run_with_cigp.sh는 GT(Ground Truth) 사용
# 이 스크립트는 CCTV 감지 결과 사용

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${PARENT_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"
SINGULAR_DIR="/home/pyongjoo/Desktop/newstart/SingularTrajectory"
WITH_ROBOT_DIR="${PARENT_DIR}/with_robot"

CONTAINER_NAME="gdae_pedsim_cigp_cctv"

# 시나리오 선택 (기본: scenario_block_heavy.xml)
SCENARIO=${1:-"scenario_block_heavy.xml"}

# Stop and remove existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "=================================================="
echo " Warehouse + Robot + CIGP (CCTV Vision-based)"
echo "=================================================="
echo ""
echo " Scenario: $SCENARIO"
echo ""
echo " [CCTV Mode] Uses camera detection instead of GT"
echo ""
echo " Topics:"
echo "   /cctv/detected_agents   - CCTV detected humans"
echo "   /cigp_cctv/global_path  - Human-aware path"
echo "   /cigp_cctv/waypoints    - RViz markers"
echo "   /cigp_cctv/costmap_image - Visualization"
echo ""
echo " Set goal:"
echo "   rostopic pub -1 /cigp_cctv/goal geometry_msgs/PoseStamped \\"
echo "     '{header: {frame_id: \"odom\"}, pose: {position: {x: 9, y: -9, z: 0}, orientation: {w: 1}}}'"
echo ""
echo " Logs: /environment/cigp_cctv_logs/"
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
  -v ${PARENT_DIR}:/environment:rw \
  -v ${NAV_DIR}:/navigation:rw \
  -v ${SICNAV_DIR}:/sicnav-test:ro \
  -v ${SINGULAR_DIR}:/SingularTrajectory:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:yolo \
  bash -c "
    # Copy configuration files (기존 with_robot에서 복사)
    cp /environment/with_robot/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # PYTHONPATH for CIGP, getimage, and SingularTrajectory modules
    export PYTHONPATH=/sicnav-test:/environment/getimage:/SingularTrajectory:/environment/trajectory_prediction:\${PYTHONPATH}

    # Create log directory
    mkdir -p /environment/cigp_cctv_logs

    echo ''
    echo 'Starting simulation...'
    echo ''

    # Launch simulation (background)
    roslaunch pedsim_simulator warehouse_with_robot.launch \
      scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
      world_file:=/environment/with_robot/worlds/warehouse.world &

    # Wait for simulation to fully start
    sleep 10

    echo ''
    echo '============================================'
    echo 'Starting cmd_vel inverter...'
    echo '============================================'
    echo ''

    # Start cmd_vel inverter
    python3 /navigation/scripts/cmd_vel_inverter.py &
    sleep 2

    echo ''
    echo '============================================'
    echo 'Starting move_base (DWA local planner)...'
    echo '============================================'
    echo ''

    # Launch move_base for DWA local planning
    roslaunch /navigation/launch/move_base.launch &
    sleep 5

    echo ''
    echo '============================================'
    echo 'Starting CCTV Human Publisher...'
    echo '============================================'
    echo ''

    # CCTV 감지 → ROS 토픽 발행 (background)
    python3 /environment/getimage/cctv_human_publisher.py &
    sleep 3

    echo ''
    echo '============================================'
    echo 'Starting CIGP CCTV Planner...'
    echo '============================================'
    echo ''
    echo 'CIGP will use CCTV detections for human-aware planning'
    echo ''

    # Run CIGP CCTV planner (foreground)
    python3 /environment/getimage/cigp_cctv_planner.py
  "
