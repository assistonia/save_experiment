#!/bin/bash
# ==============================================================================
# DRL Local Planner Test
# TD3 강화학습 모델을 Local Planner로 테스트
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
SCENARIO="scenario_block_dynamic.xml"
CONTAINER_NAME="gdae_pedsim_drl_test"

echo "=============================================================="
echo " DRL (TD3) Local Planner Test"
echo "=============================================================="
echo " Scenario: $SCENARIO"
echo "=============================================================="

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
sleep 2

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

        # Start simulation
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/'${SCENARIO}' \
            world_file:=/environment/with_robot/worlds/warehouse.world \
            gui:=false &

        sleep 15

        # DRL Local Planner 실행 (직접 /p3dx/cmd_vel에 발행, inverter 필요 없음)
        echo "Starting DRL (TD3) Local Planner..."
        python3 /navigation/scripts/drl_local_planner.py &
        DRL_PID=$!
        sleep 5

        # DRL 프로세스 상태 확인
        if ps -p $DRL_PID > /dev/null; then
            echo "DRL planner running (PID: $DRL_PID)"
        else
            echo "ERROR: DRL planner failed to start"
            wait $DRL_PID
        fi

        echo "DRL simulation ready!"
        tail -f /dev/null
    '

echo "Waiting for simulation to initialize (30s)..."
sleep 30

if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "ERROR: Container not running!"
    docker logs $CONTAINER_NAME 2>&1 | tail -30
    exit 1
fi

echo ""
echo "Checking DRL planner status..."
docker exec $CONTAINER_NAME bash -c "source /opt/ros/noetic/setup.bash && rostopic list | grep -E 'cmd_vel|goal'" || true

echo ""
echo "Sending test goal (far from start position)..."
# 여러 번 발송하여 확실히 수신되도록 함
for i in 1 2 3; do
    docker exec $CONTAINER_NAME bash -c "
        source /opt/ros/noetic/setup.bash
        rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped '{
            header: {frame_id: \"odom\"},
            pose: {position: {x: 0.0, y: 5.0, z: 0.0}, orientation: {w: 1.0}}
        }'
    " &
    sleep 1
done
wait
sleep 2

echo ""
echo "Checking DRL node status..."
docker exec $CONTAINER_NAME bash -c "
    source /opt/ros/noetic/setup.bash
    echo '=== ROS nodes ==='
    rosnode list 2>/dev/null || echo 'Cannot list nodes'
    echo ''
    echo '=== Processes ==='
    ps aux | grep -E 'drl_local|python3' | grep -v grep | head -5
    echo ''
    echo '=== Topic publishers for /p3dx/cmd_vel ==='
    rostopic info /p3dx/cmd_vel 2>/dev/null || echo 'Cannot get topic info'
" || true

echo ""
echo "Watching robot for 30s..."
for i in {1..6}; do
    sleep 5
    echo "--- $((i*5))s ---"
    docker exec $CONTAINER_NAME bash -c "
        source /opt/ros/noetic/setup.bash
        rostopic echo -n 1 /p3dx/odom 2>/dev/null | grep -A 2 'position:' | head -4
        echo 'cmd_vel:'
        timeout 1 rostopic echo -n 1 /p3dx/cmd_vel 2>/dev/null | head -5
    " || true
done

echo ""
echo "Container logs (DRL planner output):"
docker logs $CONTAINER_NAME 2>&1 | grep -E "\[DRL\]|Error|error" | tail -30

echo ""
echo "Stopping..."
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Done!"
