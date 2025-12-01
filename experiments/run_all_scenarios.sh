#!/bin/bash
# ==============================================================================
# Full Experiment Script - All Scenarios
#
# 모든 시나리오에서 2가지 글로벌 플래너 비교 실험
# - 4 Local Planners: DWA, DRL-VO, TEB, SFM
# - 2 Global Planners: None (Baseline), CIGP
# - 3 Scenarios: baseline, congestion, circulation
#
# 총 조건: 4 × 2 × 3 = 24 조건
#
# Usage:
#   ./run_all_scenarios.sh [EPISODES_PER_CONDITION]
#
# Example:
#   ./run_all_scenarios.sh 10   # 각 조건당 10 에피소드 (총 360 에피소드)
#   ./run_all_scenarios.sh 100  # 각 조건당 100 에피소드 (총 3600 에피소드)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"
SINGULAR_DIR="/home/pyongjoo/Desktop/newstart/SingularTrajectory"

# 에피소드 수
EPISODES=${1:-10}

# 타임스탬프
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_BASE="/environment/experiments/results/full_experiment_${TIMESTAMP}"

# 설정 (Predictive Planning 제외)
LOCAL_PLANNERS=("dwa" "drl_vo" "teb" "sfm")
GLOBAL_PLANNERS=("none" "cigp")
SCENARIOS=("warehouse_pedsim.xml" "scenario_block_heavy.xml" "scenario_congestion_all.xml")
SCENARIO_NAMES=("baseline" "congestion" "circulation")

CONTAINER_NAME="full_experiment_${TIMESTAMP}"

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# X11 허용
xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

# 총 조건 수 계산
TOTAL_CONDITIONS=$((${#LOCAL_PLANNERS[@]} * ${#GLOBAL_PLANNERS[@]} * ${#SCENARIOS[@]}))
TOTAL_EPISODES=$((TOTAL_CONDITIONS * EPISODES))

echo "=============================================================="
echo " FULL EXPERIMENT - ALL SCENARIOS"
echo "=============================================================="
echo ""
echo " Local Planners:  ${LOCAL_PLANNERS[*]}"
echo " Global Planners: ${GLOBAL_PLANNERS[*]}"
echo " Scenarios:       ${SCENARIO_NAMES[*]}"
echo ""
echo " Episodes per condition: $EPISODES"
echo " Total conditions:       $TOTAL_CONDITIONS"
echo " Total episodes:         $TOTAL_EPISODES"
echo ""
echo " Results: $RESULTS_BASE"
echo ""
echo " Estimated time: ~$((TOTAL_EPISODES * 2 / 60)) hours"
echo " (Predictive Planning 제외 - Local vs CIGP 비교)"
echo ""
echo "=============================================================="
echo ""
read -p "Press Enter to start or Ctrl+C to cancel..."

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
    -v ${SINGULAR_DIR}:/SingularTrajectory:ro \
    -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
    --network host \
    gdae_with_navigation:yolo \
    bash -c "
        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
        export PYTHONPATH=/SingularTrajectory:/sicnav-test:/environment/local_planners:/environment/experiments:/environment/predictive_planning:/environment/trajectory_prediction:\${PYTHONPATH}

        mkdir -p ${RESULTS_BASE}

        # 시나리오별 실험 (Predictive Planning 제외)
        SCENARIOS=(warehouse_pedsim.xml scenario_block_heavy.xml scenario_congestion_all.xml)
        SCENARIO_NAMES=(baseline congestion circulation)
        LOCAL_PLANNERS=(dwa drl_vo teb sfm)
        GLOBAL_PLANNERS=(none cigp)

        CONDITION_COUNT=0
        TOTAL_CONDITIONS=\$((4 * 2 * 3))

        for scenario_idx in 0 1 2; do
            SCENARIO=\${SCENARIOS[\$scenario_idx]}
            SCENARIO_NAME=\${SCENARIO_NAMES[\$scenario_idx]}

            echo ''
            echo '################################################################'
            echo \"# SCENARIO: \$SCENARIO_NAME (\$SCENARIO)\"
            echo '################################################################'
            echo ''

            # 시나리오 파일 복사
            cp /environment/with_robot/scenarios/\${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
            cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
            cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

            for local_planner in \${LOCAL_PLANNERS[@]}; do
                for global_planner in \${GLOBAL_PLANNERS[@]}; do
                    CONDITION_COUNT=\$((CONDITION_COUNT + 1))

                    echo ''
                    echo '================================================'
                    echo \"Condition \$CONDITION_COUNT/\$TOTAL_CONDITIONS\"
                    echo \"Local: \$local_planner, Global: \$global_planner\"
                    echo \"Scenario: \$SCENARIO_NAME\"
                    echo '================================================'
                    echo ''

                    # PedSim + Gazebo 시작
                    roslaunch pedsim_simulator warehouse_with_robot.launch \
                        scene_file:=/environment/with_robot/scenarios/\${SCENARIO} \
                        world_file:=/environment/with_robot/worlds/warehouse.world &
                    PEDSIM_PID=\$!
                    sleep 15

                    # Global Planner 시작 (CIGP만)
                    if [ \"\$global_planner\" = \"cigp\" ]; then
                        python3 /environment/cigp_integration/cigp_bridge_node.py &
                        GLOBAL_PID=\$!
                        sleep 5
                    else
                        GLOBAL_PID=0
                    fi

                    # Local Planner 시작
                    python3 /environment/local_planners/local_planner_node.py _planner:=\${local_planner} &
                    LOCAL_PID=\$!
                    sleep 3

                    # 실험 실행
                    python3 /environment/experiments/experiment_runner.py \
                        --planner \$local_planner \
                        --global-planner \$global_planner \
                        --scenario \$SCENARIO \
                        --episodes ${EPISODES} \
                        --results-dir ${RESULTS_BASE}

                    # 프로세스 종료
                    kill \$LOCAL_PID 2>/dev/null || true
                    [ \$GLOBAL_PID -ne 0 ] && kill \$GLOBAL_PID 2>/dev/null || true
                    kill \$PEDSIM_PID 2>/dev/null || true
                    sleep 5
                done
            done
        done

        echo ''
        echo '################################################################'
        echo '# EXPERIMENT COMPLETE'
        echo '################################################################'
        echo ''
        echo 'Generating final analysis and images...'

        # 최종 분석 및 이미지 생성
        python3 -c \"
import sys
sys.path.insert(0, '/environment/experiments')
from data_logger import load_results, ResultsAnalyzer
from visualizer import TrajectoryVisualizer

logger = load_results('${RESULTS_BASE}')
analyzer = ResultsAnalyzer(logger)
analyzer.save_analysis(generate_images=True)
\"

        echo ''
        echo 'Results saved to: ${RESULTS_BASE}'
        echo ''

        # 결과 요약 출력
        if [ -f ${RESULTS_BASE}/analysis/summary.txt ]; then
            cat ${RESULTS_BASE}/analysis/summary.txt
        fi
    "

echo ""
echo "=============================================================="
echo " FULL EXPERIMENT COMPLETED"
echo " Results: ${ENV_DIR}/experiments/results/full_experiment_${TIMESTAMP}"
echo "=============================================================="
