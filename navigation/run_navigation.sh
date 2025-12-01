#!/bin/bash
# Run DWA Navigation for robot
# This script runs INSIDE the gdae_pedsim_robot container
# Usage: ./run_navigation.sh [goal_x] [goal_y]

GOAL_X=${1:-9.0}
GOAL_Y=${2:--10.0}

echo "==================================="
echo "DWA Navigation"
echo "Goal: ($GOAL_X, $GOAL_Y)"
echo "==================================="

# Source ROS
source /opt/ros/noetic/setup.bash
source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

# Launch move_base in background
echo "Starting move_base..."
roslaunch /navigation/launch/move_base.launch &
MOVE_BASE_PID=$!

# Wait for move_base to initialize
sleep 5

# Send goal
echo "Sending navigation goal..."
python3 /navigation/scripts/send_goal.py $GOAL_X $GOAL_Y

# Cleanup
kill $MOVE_BASE_PID 2>/dev/null
echo "Navigation complete"
