#!/bin/bash
# ==============================================================================
# Experiment Runner Script
#
# Local Planner Only vs Local Planner + CIGP 비교 실험
# 논문: "CCTV-Informed Human-Aware Robot Navigation"
#
# Usage:
#   ./run_experiment.sh [OPTIONS]
#
# Options:
#   --planner PLANNER     Local planner (dwa, drl_vo, teb, orca, sfm)
#   --cigp                Enable CIGP global planner
#   --scenario SCENARIO   Scenario file
#   --episodes N          Number of episodes (default: 100)
#   --all                 Run all conditions
#
# Examples:
#   # DWA only (baseline)
#   ./run_experiment.sh --planner dwa --scenario warehouse_pedsim.xml --episodes 10
#
#   # DWA + CIGP
#   ./run_experiment.sh --planner dwa --cigp --scenario warehouse_pedsim.xml --episodes 10
#
#   # All conditions
#   ./run_experiment.sh --all --episodes 10
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"
WITH_ROBOT_DIR="${ENV_DIR}/with_robot"

# 기본값
PLANNER="dwa"
USE_CIGP=false
SCENARIO="warehouse_pedsim.xml"
EPISODES=100
RUN_ALL=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --planner)
            PLANNER="$2"
            shift 2
            ;;
        --cigp)
            USE_CIGP=true
            shift
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        -h|--help)
            head -n 30 "$0" | tail -n 28
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 결과 디렉토리 (타임스탬프 포함)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/environment/experiments/results/experiment_${TIMESTAMP}"

CONTAINER_NAME="experiment_runner_${TIMESTAMP}"

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# X11 허용
xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

echo "=============================================================="
echo " EXPERIMENT: Local Planner vs CIGP + Local Planner"
echo "=============================================================="
echo ""
echo " Configuration:"
if [ "$RUN_ALL" = true ]; then
    echo "   Mode:      ALL CONDITIONS"
else
    echo "   Planner:   $PLANNER"
    echo "   CIGP:      $USE_CIGP"
    echo "   Scenario:  $SCENARIO"
fi
echo "   Episodes:  $EPISODES"
echo "   Results:   $RESULTS_DIR"
echo ""
echo "=============================================================="

# CIGP 옵션 문자열
CIGP_ARG=""
if [ "$USE_CIGP" = true ]; then
    CIGP_ARG="--use-cigp"
fi

# 실험 명령어 구성
if [ "$RUN_ALL" = true ]; then
    EXPERIMENT_CMD="python3 /environment/experiments/experiment_runner.py --all --episodes $EPISODES --results-dir $RESULTS_DIR"
else
    EXPERIMENT_CMD="python3 /environment/experiments/experiment_runner.py --planner $PLANNER $CIGP_ARG --scenario $SCENARIO --episodes $EPISODES --results-dir $RESULTS_DIR"
fi

# Docker 실행
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
        # 시나리오 파일 복사
        cp /environment/with_robot/scenarios/${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        # ROS 환경 설정
        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

        # PYTHONPATH 설정
        export PYTHONPATH=/sicnav-test:/environment/local_planners:/environment/experiments:\${PYTHONPATH}

        echo ''
        echo 'Starting simulation environment...'
        echo ''

        # 1. PedSim + Gazebo 시작 (백그라운드)
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world &

        sleep 15

        # 2. CIGP 시작 (조건부)
        if [ '${USE_CIGP}' = 'true' ] || [ '${RUN_ALL}' = 'true' ]; then
            echo ''
            echo 'Starting CIGP Global Planner...'
            echo ''
            python3 /environment/cigp_integration/cigp_bridge_node.py &
            sleep 5
        fi

        # 3. Local Planner 시작
        echo ''
        echo 'Starting Local Planner: ${PLANNER}'
        echo ''
        python3 /environment/local_planners/local_planner_node.py _planner:=${PLANNER} &
        sleep 3

        # 4. 실험 러너 시작
        echo ''
        echo '============================================'
        echo 'Starting Experiment Runner...'
        echo '============================================'
        echo ''

        ${EXPERIMENT_CMD}
    "

echo ""
echo "=============================================================="
echo " EXPERIMENT COMPLETED"
echo " Results saved to: ${ENV_DIR}/experiments/results/experiment_${TIMESTAMP}"
echo "=============================================================="
