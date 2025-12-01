#!/bin/bash
# Predictive Planning 실행 스크립트
#
# 사용법:
#   ./run_predictive_planning.sh              # 기본 실행
#   ./run_predictive_planning.sh --mock       # Mock 모드 (예측기 없이 테스트)
#   ./run_predictive_planning.sh --standalone # ROS 없이 독립 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"

# Python 경로 설정
export PYTHONPATH="$ENV_DIR:$PYTHONPATH"

# SingularTrajectory 경로 (도커/호스트 자동 감지)
if [ -d "/SingularTrajectory" ]; then
    export PYTHONPATH="/SingularTrajectory:$PYTHONPATH"
else
    export PYTHONPATH="/home/pyongjoo/Desktop/newstart/SingularTrajectory:$PYTHONPATH"
fi

# CUDA 설정
export CUDA_VISIBLE_DEVICES=0

echo "=========================================="
echo "  Predictive Planning Module"
echo "=========================================="
echo "  Script dir: $SCRIPT_DIR"
echo "  Env dir: $ENV_DIR"
echo "  Arguments: $@"
echo "=========================================="

# 인자 처리
USE_MOCK=""
STANDALONE=""

for arg in "$@"; do
    case $arg in
        --mock)
            USE_MOCK="--mock"
            echo "  Mode: Mock (no real predictor)"
            ;;
        --standalone)
            STANDALONE="--standalone"
            echo "  Mode: Standalone (no ROS)"
            ;;
    esac
done

# ROS 환경 설정 (standalone이 아닌 경우)
if [ -z "$STANDALONE" ]; then
    if [ -f "/opt/ros/noetic/setup.bash" ]; then
        source /opt/ros/noetic/setup.bash
        echo "  ROS: Noetic sourced"
    elif [ -f "/opt/ros/melodic/setup.bash" ]; then
        source /opt/ros/melodic/setup.bash
        echo "  ROS: Melodic sourced"
    fi

    # Workspace 설정
    if [ -f "$ENV_DIR/../catkin_ws/devel/setup.bash" ]; then
        source "$ENV_DIR/../catkin_ws/devel/setup.bash"
    fi
fi

echo "=========================================="

# 실행
cd "$SCRIPT_DIR"
python3 src/predictive_planning_bridge.py $USE_MOCK $STANDALONE
