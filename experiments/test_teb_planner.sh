#!/bin/bash
# ==============================================================================
# TEB Local Planner Test
# Quick test to see if TEB works with crossing scenario
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
SCENARIO="scenario_crossing_sparse.xml"
CONTAINER_NAME="gdae_pedsim_teb_test"

echo "=============================================================="
echo " TEB Local Planner Test"
echo "=============================================================="
echo " Scenario: $SCENARIO"
echo "=============================================================="

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
sleep 2

# xhost 설정
xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

# Docker 실행
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e QT_QPA_PLATFORM=offscreen \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${ENV_DIR}:/environment:rw \
    -v ${ENV_DIR}/navigation:/navigation:rw \
    -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
    --network host \
    gdae_with_navigation:yolo \
    bash -c '
        # Install TEB
        echo "Installing TEB local planner..."
        apt-get update -qq && apt-get install -y -qq ros-noetic-teb-local-planner

        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

        # Headless mode
        export QT_QPA_PLATFORM=offscreen
        Xvfb :99 -screen 0 1024x768x24 &
        export DISPLAY=:99
        sleep 2

        # Copy scenario
        cp /environment/with_robot/scenarios/'${SCENARIO}' /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        # Start PedSim + Gazebo
        echo "Starting PedSim + Gazebo..."
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/'${SCENARIO}' \
            world_file:=/environment/with_robot/worlds/warehouse.world \
            gui:=false &

        sleep 15

        # cmd_vel inverter
        python3 /navigation/scripts/cmd_vel_inverter.py &
        sleep 2

        # Start move_base with TEB
        echo "Starting move_base with TEB local planner..."
        roslaunch /navigation/launch/move_base_teb.launch &
        sleep 5

        echo "TEB planner ready!"

        # Keep container running
        tail -f /dev/null
    '

echo "Waiting for simulation to initialize (40s)..."
sleep 40

# Check container
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "ERROR: Container not running!"
    docker logs $CONTAINER_NAME 2>&1 | tail -50
    exit 1
fi

echo ""
echo "Checking TEB planner status..."
docker exec $CONTAINER_NAME bash -c "source /opt/ros/noetic/setup.bash && rostopic list | grep -E 'move_base|teb'" || true

echo ""
echo "Testing goal navigation..."
docker exec $CONTAINER_NAME bash -c "
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # Send a test goal
    rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{
        header: {frame_id: \"odom\"},
        pose: {position: {x: 0.0, y: 10.0, z: 0.0}, orientation: {w: 1.0}}
    }'
"

echo ""
echo "Waiting for robot to navigate (30s)..."
sleep 30

# Check robot position
echo ""
echo "Current robot position:"
docker exec $CONTAINER_NAME bash -c "
    source /opt/ros/noetic/setup.bash
    rostopic echo -n 1 /p3dx/odom | head -10
"

echo ""
echo "Stopping container..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Done!"
