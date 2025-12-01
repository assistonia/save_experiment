#!/usr/bin/env python3
"""
Send navigation goal to move_base
Usage: python3 send_goal.py [x] [y]
Default goal: (9, -10) from warehouse.json
"""

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
import sys

def send_goal(x, y):
    rospy.init_node('send_nav_goal', anonymous=True)

    # Create action client
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base server")

    # Create goal
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "odom"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0

    rospy.loginfo(f"Sending goal: ({x}, {y})")
    client.send_goal(goal)

    # Wait for result
    wait = client.wait_for_result(rospy.Duration(120.0))
    if wait:
        result = client.get_state()
        if result == 3:  # SUCCEEDED
            rospy.loginfo("Goal reached!")
            return True
        else:
            rospy.logwarn(f"Goal failed with state: {result}")
            return False
    else:
        rospy.logwarn("Timeout waiting for goal")
        client.cancel_goal()
        return False

if __name__ == '__main__':
    # Default goal from warehouse.json
    goal_x = 9.0
    goal_y = -10.0

    if len(sys.argv) >= 3:
        goal_x = float(sys.argv[1])
        goal_y = float(sys.argv[2])

    try:
        send_goal(goal_x, goal_y)
    except rospy.ROSInterruptException:
        pass
