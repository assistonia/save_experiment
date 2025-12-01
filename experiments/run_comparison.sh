#!/bin/bash
# ==============================================================================
# Comparison Experiment - 동일 조건에서 DWA vs CIGP 비교
#
# 같은 시뮬레이션 환경에서 시작하여 공정 비교
# 시뮬레이션 리셋 후 동일 조건에서 테스트
# ==============================================================================

set -e

SCENARIO="${1:-scenario_congestion_all.xml}"
EPISODES="${2:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="${SCRIPT_DIR}/results/comparison_${TIMESTAMP}"
CONTAINER_NAME="comparison_test_${TIMESTAMP}"

mkdir -p "${RESULTS_BASE}"

echo "=============================================================="
echo " COMPARISON EXPERIMENT - DWA vs CIGP"
echo "=============================================================="
echo ""
echo " Scenario: $SCENARIO"
echo " Episodes: $EPISODES"
echo " Results:  $RESULTS_BASE"
echo ""
echo "=============================================================="

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

# Docker 실행
docker run -i \
    --name $CONTAINER_NAME \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TIMESTAMP=$TIMESTAMP \
    -e EPISODES=$EPISODES \
    -e SCENARIO=$SCENARIO \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${ENV_DIR}:/environment:rw \
    -v ${NAV_DIR}:/navigation:rw \
    -v ${SICNAV_DIR}:/sicnav-test:ro \
    -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
    --network host \
    gdae_with_navigation:yolo \
    bash -c '
        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
        export PYTHONPATH=/sicnav-test:/environment:${PYTHONPATH}

        # 시나리오 복사
        cp /environment/with_robot/scenarios/${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        mkdir -p /environment/experiments/results/comparison_${TIMESTAMP}/dwa/episodes
        mkdir -p /environment/experiments/results/comparison_${TIMESTAMP}/cigp/episodes

        echo ""
        echo "Starting simulation..."
        echo ""

        # PedSim + Gazebo 실행 (headless)
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world \
            gui:=false &
        sleep 15

        # cmd_vel inverter
        python3 /navigation/scripts/cmd_vel_inverter.py &
        sleep 2

        # move_base
        roslaunch /navigation/launch/move_base.launch &
        sleep 5

        # 비교 실험 Python 스크립트 실행
        python3 /environment/experiments/comparison_runner.py \
            --results-dir /environment/experiments/results/comparison_${TIMESTAMP} \
            --episodes ${EPISODES}
    '

echo ""
echo "=============================================================="
echo " COMPARISON COMPLETED"
echo " Results: ${RESULTS_BASE}"
echo "=============================================================="

# 시각화
cd ${SCRIPT_DIR}
python3 visualize_comparison.py ${RESULTS_BASE} 2>/dev/null || echo "Visualization skipped"
