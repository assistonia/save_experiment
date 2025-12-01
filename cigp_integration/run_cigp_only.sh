#!/bin/bash
# CIGP Bridge Node만 단독 실행
# 이미 시뮬레이션이 실행 중일 때 사용

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SICNAV_DIR="/home/pyongjoo/Desktop/newstart/sicnav-test"

CONTAINER_NAME="gdae_pedsim_robot"  # 기존 시뮬레이션 컨테이너

echo "=================================================="
echo " CIGP Global Planner (Standalone)"
echo "=================================================="
echo ""
echo "Note: Make sure the simulation is already running!"
echo "      (run_with_robot_scenario.sh should be active)"
echo ""

# Check if simulation container is running
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "Error: Simulation container '$CONTAINER_NAME' is not running."
    echo "Please start the simulation first:"
    echo "  cd /home/pyongjoo/Desktop/newstart/environment/with_robot"
    echo "  ./run_with_robot_scenario.sh"
    exit 1
fi

# Execute CIGP in the running container
docker exec -it \
  -e PYTHONPATH="/sicnav-test:${PYTHONPATH}" \
  $CONTAINER_NAME \
  bash -c "
    # Mount sicnav-test if not already mounted
    if [ ! -d '/sicnav-test' ]; then
      echo 'Warning: sicnav-test not mounted. CIGP may not work.'
    fi

    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
    export PYTHONPATH=/sicnav-test:\${PYTHONPATH}

    echo 'Starting CIGP Global Planner...'
    python3 /environment/cigp_integration/cigp_bridge_node.py
  "
