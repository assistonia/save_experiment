#!/bin/bash
# Warehouse PedSim + Trajectory Prediction + Predictive Planning
#
# 기존 run_with_prediction.sh 복제 + predictive_planning 추가
# 기존 환경/컨테이너 손상 없음
#
# 로그 저장:
#   - /environment/trajectory_prediction_logs/  (예측 로그)
#   - /environment/predictive_planning/logs/    (계획 로그)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$ENV_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SINGULAR_DIR="${PARENT_DIR}/SingularTrajectory"
WITH_ROBOT_DIR="${ENV_DIR}/with_robot"

# 새 컨테이너 이름 (기존 컨테이너와 충돌 안 함)
CONTAINER_NAME="gdae_pedsim_predictive_planning"

# 시나리오 선택 (기본: scenario_block_heavy.xml)
SCENARIO=${1:-"scenario_block_heavy.xml"}

# 로컬 플래너 선택: dwa, drl_vo, sfm, orca (기본: dwa)
LOCAL_PLANNER=${2:-"dwa"}

# 직접 제어 모드 (true/false) - false면 로컬 플래너 사용
USE_DIRECT=${3:-"true"}

# Stop and remove only OUR container (기존 것은 건드리지 않음)
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "============================================================"
echo " Warehouse + Predictive Planning"
echo "============================================================"
echo ""
echo " Container: $CONTAINER_NAME (새 컨테이너, 기존 것 유지)"
echo " Scenario: $SCENARIO"
echo " Local Planner: $LOCAL_PLANNER"
echo " Direct Control: $USE_DIRECT"
echo ""
echo " [Predictive Planning Mode]"
echo " - SingularTrajectory: 8 frames → 12 frames × 20 samples"
echo " - Predictive A*: σ×0.8, 확률적 희석(/20)"
echo ""
echo " Topics (Input):"
echo "   /pedsim_simulator/simulated_agents  - Pedestrian GT"
echo "   /p3dx/odom                          - Robot odometry"
echo "   /predictive_planning/goal           - Navigation goal (별도 토픽)"
echo ""
echo " Topics (Output):"
echo "   /move_base_simple/goal              - Waypoint to move_base (CIGP 방식)"
echo "   /predictive_planning/global_path    - Global path visualization"
echo "   /predictive_planning/markers        - Visualization"
echo ""
echo " Logs:"
echo "   /environment/predictive_planning/logs/"
echo "============================================================"

# Check if SingularTrajectory exists
if [ ! -d "$SINGULAR_DIR" ]; then
    echo "ERROR: SingularTrajectory not found at $SINGULAR_DIR"
    exit 1
fi

# Run container (기존 이미지 사용, 새 컨테이너 이름)
docker run -i \
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
  -v ${SINGULAR_DIR}:/SingularTrajectory:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:latest \
  bash -c "
    # Copy configuration files (기존과 동일)
    cp /environment/with_robot/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # PYTHONPATH (기존 + predictive_planning 추가)
    export PYTHONPATH=/SingularTrajectory:/environment:/environment/trajectory_prediction:/environment/predictive_planning:\${PYTHONPATH}

    # Create log directories
    mkdir -p /environment/trajectory_prediction_logs
    mkdir -p /environment/predictive_planning/logs

    echo ''
    echo 'Starting simulation...'
    echo ''

    # Launch simulation (background) - 기존과 동일
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

    # Start cmd_vel inverter (CIGP와 동일 - fixes left/right reversal)
    python3 /navigation/scripts/cmd_vel_inverter.py &
    sleep 2

    echo ''
    echo '============================================'
    echo 'Starting move_base (DWA local planner)...'
    echo '============================================'
    echo ''

    # Launch move_base for DWA local planning (CIGP와 동일)
    roslaunch /navigation/launch/move_base.launch &
    sleep 5

    echo ''
    echo '============================================'
    echo 'Starting Predictive Planning Node...'
    echo '============================================'
    echo ''
    echo 'Pipeline:'
    echo '  PedSim GT → SingularTrajectory (20 samples)'
    echo '           → Predictive Cost (σ×0.8, /20)'
    echo '           → Predictive A*'
    echo '           → move_base/DWA (CIGP와 동일)'
    echo ''

    # Run predictive planning (background)
    cd /environment/predictive_planning
    python3 src/predictive_planning_bridge.py >> /environment/predictive_planning/logs/bridge.log 2>&1 &

    echo ''
    echo 'Predictive Planning started. Sending waypoints to move_base.'
    echo ''
    echo 'To set a goal, publish to /predictive_planning/goal:'
    echo '  rostopic pub -1 /predictive_planning/goal geometry_msgs/PoseStamped \"{header: {frame_id: odom}, pose: {position: {x: 5.0, y: 5.0, z: 0.0}, orientation: {w: 1.0}}}\"'
    echo ''

    # Wait for all background processes
    wait
  "
