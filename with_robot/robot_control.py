#!/usr/bin/env python3
"""
Simple Robot Keyboard Control
WASD + QE for control
"""
import rospy
from geometry_msgs.msg import Twist
import sys
import termios
import tty

msg = """
Robot Control (Pioneer3DX)
---------------------------
   w : 전진
   s : 후진
   a : 좌회전
   d : 우회전
   q : 전진+좌회전
   e : 전진+우회전
space : 정지
   x : 종료

r/f : 속도 증가/감소
---------------------------
"""

def get_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return key

def main():
    rospy.init_node('robot_teleop')
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    speed = 0.5
    turn = 1.0

    print(msg)
    print(f"Speed: {speed}, Turn: {turn}")

    try:
        while not rospy.is_shutdown():
            key = get_key()
            twist = Twist()

            if key == 'w':
                twist.linear.x = speed
            elif key == 's':
                twist.linear.x = -speed
            elif key == 'a':
                twist.angular.z = -turn  # 좌회전 (반전)
            elif key == 'd':
                twist.angular.z = turn   # 우회전 (반전)
            elif key == 'q':
                twist.linear.x = speed
                twist.angular.z = -turn  # 전진+좌회전 (반전)
            elif key == 'e':
                twist.linear.x = speed
                twist.angular.z = turn   # 전진+우회전 (반전)
            elif key == ' ':
                twist.linear.x = 0
                twist.angular.z = 0
            elif key == 'r':
                speed += 0.1
                print(f"Speed: {speed:.1f}")
            elif key == 'f':
                speed = max(0.1, speed - 0.1)
                print(f"Speed: {speed:.1f}")
            elif key == 'x' or key == '\x03':
                break

            pub.publish(twist)

    except Exception as e:
        print(e)
    finally:
        pub.publish(Twist())
        print("\nStopped")

if __name__ == '__main__':
    main()
