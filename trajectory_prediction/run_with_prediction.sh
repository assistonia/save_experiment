#!/bin/bash
# Warehouse PedSim + Trajectory Prediction (SingularTrajectory)
# GT 데이터 → SingularTrajectory 모델 → 경로 예측 시각화
#
# 로그 저장: /environment/trajectory_prediction_logs/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
PARENT_DIR="$(dirname "$ENV_DIR")"
NAV_DIR="${ENV_DIR}/navigation"
SINGULAR_DIR="${PARENT_DIR}/SingularTrajectory"
WITH_ROBOT_DIR="${ENV_DIR}/with_robot"

CONTAINER_NAME="gdae_pedsim_trajectory_prediction"

# 시나리오 선택 (기본: scenario_block_heavy.xml)
SCENARIO=${1:-"scenario_block_heavy.xml"}

# Stop and remove existing container
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

# Allow X11 access
xhost +local:root
xhost +local:docker

echo "=================================================="
echo " Warehouse + Trajectory Prediction"
echo "=================================================="
echo ""
echo " Scenario: $SCENARIO"
echo ""
echo " [Trajectory Prediction Mode]"
echo " - Uses SingularTrajectory model (CVPR 2024)"
echo " - Observes 8 frames → Predicts 12 frames"
echo ""
echo " Topics:"
echo "   /pedsim_simulator/simulated_agents  - Input (GT)"
echo "   /trajectory_prediction/predicted_paths - Predicted trajectories"
echo "   /trajectory_prediction/agent_markers   - Agent visualization"
echo ""
echo " Logs: /environment/trajectory_prediction_logs/"
echo "=================================================="

# Check if SingularTrajectory exists
if [ ! -d "$SINGULAR_DIR" ]; then
    echo "ERROR: SingularTrajectory not found at $SINGULAR_DIR"
    exit 1
fi

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
  -v ${SINGULAR_DIR}:/SingularTrajectory:ro \
  -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
  --network host \
  gdae_with_navigation:yolo \
  bash -c "
    # Copy configuration files
    cp /environment/with_robot/scenarios/warehouse_pedsim.xml /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
    cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
    cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

    # Source ROS
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

    # PYTHONPATH for modules
    export PYTHONPATH=/SingularTrajectory:/environment/trajectory_prediction:\${PYTHONPATH}

    # Create log directory
    mkdir -p /environment/trajectory_prediction_logs

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
    echo 'Starting Trajectory Prediction Node...'
    echo '============================================'
    echo ''
    echo 'Model: SingularTrajectory (CVPR 2024)'
    echo 'Obs: 8 frames, Pred: 12 frames'
    echo ''

    # Run trajectory prediction (foreground)
    python3 /environment/trajectory_prediction/prediction_bridge_node.py
  "
