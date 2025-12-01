#!/bin/bash
# Start navigation with cmd_vel inverter
# Run this INSIDE the container after simulation is running

source /opt/ros/noetic/setup.bash
source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash

echo "Starting cmd_vel inverter (fixes left/right reversal)..."
python3 /navigation/scripts/cmd_vel_inverter.py &
sleep 2

echo "Starting move_base..."
roslaunch /navigation/launch/move_base.launch &
sleep 5

echo ""
echo "Navigation started!"
echo ""
echo "To send goal (9, -10):"
echo "rostopic pub -1 /move_base_simple/goal geometry_msgs/PoseStamped \"header: {frame_id: 'odom'}\" \"pose: {position: {x: 9.0, y: -10.0}, orientation: {w: 1.0}}\""
echo ""
