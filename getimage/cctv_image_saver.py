#!/usr/bin/env python3
"""
CCTV Image Saver - ROS node to save images from CCTV cameras
Saves images to /environment/getimage/cctv_X/ directories
"""

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class CCTVImageSaver:
    def __init__(self, save_dir="/environment/getimage"):
        self.bridge = CvBridge()
        self.save_dir = save_dir
        self.camera_ids = [0, 1, 2, 3]
        self.image_count = {i: 0 for i in self.camera_ids}

        # Create directories for each camera
        for cam_id in self.camera_ids:
            cam_dir = os.path.join(self.save_dir, f"cctv_{cam_id}")
            os.makedirs(cam_dir, exist_ok=True)

        # Subscribe to each camera topic
        self.subscribers = []
        for cam_id in self.camera_ids:
            topic = f"/cctv_{cam_id}/image_raw"
            sub = rospy.Subscriber(
                topic,
                Image,
                self.image_callback,
                callback_args=cam_id
            )
            self.subscribers.append(sub)
            rospy.loginfo(f"Subscribed to {topic}")

    def image_callback(self, msg, cam_id):
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cctv_{cam_id}_{timestamp}.jpg"
            filepath = os.path.join(self.save_dir, f"cctv_{cam_id}", filename)

            # Save image
            cv2.imwrite(filepath, cv_image)
            self.image_count[cam_id] += 1

            if self.image_count[cam_id] % 10 == 0:
                rospy.loginfo(f"CCTV {cam_id}: Saved {self.image_count[cam_id]} images")

        except Exception as e:
            rospy.logerr(f"Error saving image from CCTV {cam_id}: {e}")

def main():
    rospy.init_node('cctv_image_saver', anonymous=True)

    save_dir = rospy.get_param('~save_dir', '/environment/getimage')
    saver = CCTVImageSaver(save_dir)

    rospy.loginfo("CCTV Image Saver started. Saving to: " + save_dir)
    rospy.spin()

if __name__ == '__main__':
    main()
