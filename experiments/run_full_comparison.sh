#!/bin/bash
# ==============================================================================
# Full Comparison Script (3 Global Planners)
#
# 3가지 조건 비교:
#   1. Local Only (Baseline)
#   2. CIGP + Local
#   3. Predictive Planning + Local
#
# Usage:
#   ./run_full_comparison.sh [PLANNER] [SCENARIO] [EPISODES]
#
# Examples:
#   ./run_full_comparison.sh              # DWA, baseline, 10 episodes
#   ./run_full_comparison.sh drl_vo       # DRL-VO, baseline, 10 episodes
#   ./run_full_comparison.sh dwa congestion 20
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"
SINGULAR_DIR="/home/pyongjoo/Desktop/newstart/SingularTrajectory"

# 인자
PLANNER=${1:-"dwa"}
SCENARIO_TYPE=${2:-"baseline"}
EPISODES=${3:-10}

# 시나리오 매핑
case $SCENARIO_TYPE in
    baseline|basic)
        SCENARIO="warehouse_pedsim.xml"
        ;;
    congestion|heavy)
        SCENARIO="scenario_block_heavy.xml"
        ;;
    circulation|all)
        SCENARIO="scenario_congestion_all.xml"
        ;;
    *)
        SCENARIO="${SCENARIO_TYPE}"
        ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/environment/experiments/results/full_comparison_${PLANNER}_${TIMESTAMP}"

CONTAINER_NAME="full_comparison_${PLANNER}_${TIMESTAMP}"

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# X11 허용
xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

echo "=============================================================="
echo " FULL COMPARISON EXPERIMENT (3 Global Planners)"
echo "=============================================================="
echo ""
echo " Local Planner: $PLANNER"
echo " Scenario:      $SCENARIO"
echo " Episodes:      $EPISODES (per condition)"
echo ""
echo " Conditions:"
echo "   1. ${PLANNER^^} only          (Baseline)"
echo "   2. CIGP-${PLANNER^^}          (CCTV-Informed)"
echo "   3. PRED-${PLANNER^^}          (Predictive Planning)"
echo ""
echo " Total episodes: $((EPISODES * 3))"
echo " Results: $RESULTS_DIR"
echo ""
echo "=============================================================="

# SingularTrajectory 체크
if [ ! -d "$SINGULAR_DIR" ]; then
    echo "WARNING: SingularTrajectory not found at $SINGULAR_DIR"
    echo "Predictive Planning will not work!"
fi

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
        # 파일 복사
        cp /environment/with_robot/scenarios/${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
        export PYTHONPATH=/SingularTrajectory:/sicnav-test:/environment/local_planners:/environment/experiments:/environment/predictive_planning:/environment/trajectory_prediction:\${PYTHONPATH}

        mkdir -p ${RESULTS_DIR}

        # ============================================================
        # Phase 1: Local Only (Baseline)
        # ============================================================
        echo ''
        echo '================================================'
        echo ' Phase 1/3: ${PLANNER^^} Only (Baseline)'
        echo '================================================'
        echo ''

        # PedSim + Gazebo 시작
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world &
        PEDSIM_PID=\$!
        sleep 15

        # Local Planner만 시작 (Global Planner 없음)
        python3 /environment/local_planners/local_planner_node.py _planner:=${PLANNER} &
        LOCAL_PID=\$!
        sleep 3

        # 실험 실행
        python3 /environment/experiments/experiment_runner.py \
            --planner ${PLANNER} \
            --global-planner none \
            --scenario ${SCENARIO} \
            --episodes ${EPISODES} \
            --results-dir ${RESULTS_DIR}

        # 프로세스 종료
        kill \$LOCAL_PID 2>/dev/null || true
        kill \$PEDSIM_PID 2>/dev/null || true
        sleep 5

        # ============================================================
        # Phase 2: CIGP + Local
        # ============================================================
        echo ''
        echo '================================================'
        echo ' Phase 2/3: CIGP-${PLANNER^^}'
        echo '================================================'
        echo ''

        # PedSim + Gazebo 재시작
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world &
        PEDSIM_PID=\$!
        sleep 15

        # CIGP 시작
        python3 /environment/cigp_integration/cigp_bridge_node.py &
        CIGP_PID=\$!
        sleep 5

        # Local Planner 시작
        python3 /environment/local_planners/local_planner_node.py _planner:=${PLANNER} &
        LOCAL_PID=\$!
        sleep 3

        # 실험 실행
        python3 /environment/experiments/experiment_runner.py \
            --planner ${PLANNER} \
            --global-planner cigp \
            --scenario ${SCENARIO} \
            --episodes ${EPISODES} \
            --results-dir ${RESULTS_DIR}

        # 프로세스 종료
        kill \$LOCAL_PID 2>/dev/null || true
        kill \$CIGP_PID 2>/dev/null || true
        kill \$PEDSIM_PID 2>/dev/null || true
        sleep 5

        # ============================================================
        # Phase 3: Predictive Planning + Local
        # ============================================================
        echo ''
        echo '================================================'
        echo ' Phase 3/3: PRED-${PLANNER^^} (Predictive Planning)'
        echo '================================================'
        echo ''

        # PedSim + Gazebo 재시작
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world &
        PEDSIM_PID=\$!
        sleep 15

        # Predictive Planning 시작
        cd /environment/predictive_planning
        python3 src/predictive_planning_bridge.py _use_direct_control:=false &
        PRED_PID=\$!
        sleep 8

        # Local Planner 시작
        cd /environment/local_planners
        python3 local_planner_node.py _planner:=${PLANNER} &
        LOCAL_PID=\$!
        sleep 3

        # 실험 실행
        python3 /environment/experiments/experiment_runner.py \
            --planner ${PLANNER} \
            --global-planner predictive \
            --scenario ${SCENARIO} \
            --episodes ${EPISODES} \
            --results-dir ${RESULTS_DIR}

        # 프로세스 종료
        kill \$LOCAL_PID 2>/dev/null || true
        kill \$PRED_PID 2>/dev/null || true
        kill \$PEDSIM_PID 2>/dev/null || true

        # ============================================================
        # 완료
        # ============================================================
        echo ''
        echo '================================================'
        echo ' FULL COMPARISON COMPLETE'
        echo '================================================'
        echo ''
        echo 'Results saved to: ${RESULTS_DIR}'
        echo ''

        # 결과 요약 출력
        if [ -f ${RESULTS_DIR}/analysis/summary.txt ]; then
            cat ${RESULTS_DIR}/analysis/summary.txt
        fi
    "

echo ""
echo "=============================================================="
echo " FULL COMPARISON COMPLETED"
echo " Results: ${ENV_DIR}/experiments/results/full_comparison_${PLANNER}_${TIMESTAMP}"
echo "=============================================================="
