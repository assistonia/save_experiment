#!/usr/bin/env python3
"""
Capture single image from each CCTV camera
Usage: python3 capture_once.py
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class SingleCapture:
    def __init__(self, save_dir="/environment/getimage"):
        self.bridge = CvBridge()
        self.save_dir = save_dir
        self.camera_ids = [0, 1, 2, 3]
        self.captured = {i: False for i in self.camera_ids}

        os.makedirs(save_dir, exist_ok=True)

        # Subscribe to each camera
        for cam_id in self.camera_ids:
            topic = f"/cctv_{cam_id}/image_raw"
            rospy.Subscriber(topic, Image, self.callback, callback_args=cam_id)
            print(f"Waiting for {topic}...")

    def callback(self, msg, cam_id):
        if self.captured[cam_id]:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cctv_{cam_id}_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, filename)

            cv2.imwrite(filepath, cv_image)
            self.captured[cam_id] = True
            print(f"Saved: {filepath}")

        except Exception as e:
            print(f"Error: {e}")

    def all_captured(self):
        return all(self.captured.values())

def main():
    rospy.init_node('single_capture', anonymous=True)

    save_dir = rospy.get_param('~save_dir', '/environment/getimage')
    capturer = SingleCapture(save_dir)

    rate = rospy.Rate(10)
    timeout = rospy.Time.now() + rospy.Duration(10)

    while not rospy.is_shutdown() and rospy.Time.now() < timeout:
        if capturer.all_captured():
            print("All cameras captured!")
            break
        rate.sleep()

    if not capturer.all_captured():
        missing = [i for i, v in capturer.captured.items() if not v]
        print(f"Timeout. Missing cameras: {missing}")

if __name__ == '__main__':
    main()
