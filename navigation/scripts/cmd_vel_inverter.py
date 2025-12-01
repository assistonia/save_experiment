#!/usr/bin/env python3
"""
Inverts angular.z in cmd_vel to fix left/right reversal issue
"""
import rospy
from geometry_msgs.msg import Twist

pub = None

def cmd_vel_callback(msg):
    inverted = Twist()
    inverted.linear.x = msg.linear.x
    inverted.linear.y = msg.linear.y
    inverted.linear.z = msg.linear.z
    inverted.angular.x = msg.angular.x
    inverted.angular.y = msg.angular.y
    inverted.angular.z = -msg.angular.z  # Invert rotation
    pub.publish(inverted)

if __name__ == '__main__':
    rospy.init_node('cmd_vel_inverter')
    pub = rospy.Publisher('/p3dx/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/cmd_vel_raw', Twist, cmd_vel_callback)
    rospy.loginfo("cmd_vel_inverter: Inverting angular.z from /cmd_vel_raw to /p3dx/cmd_vel")
    rospy.spin()
