#!/usr/bin/env python3
"""Capture images from all CCTV cameras"""
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

rospy.init_node('capture_cctv', anonymous=True)
bridge = CvBridge()

for i in range(0, 4):
    topic = f'/cctv_{i}/image_raw'
    try:
        msg = rospy.wait_for_message(topic, Image, timeout=5)
        img = bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imwrite(f'/environment/cctv_{i}.jpg', img)
        print(f'Saved cctv_{i}.jpg')
    except Exception as e:
        print(f'Failed {topic}: {e}')

print('Done! Images saved to /environment/')
