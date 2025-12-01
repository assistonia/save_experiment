#!/usr/bin/env python3
"""
실시간 CCTV 검출 테스트

CCTV 이미지를 받아서:
1. YOLO로 사람 검출
2. Homography로 월드 좌표 변환
3. Ground Truth와 비교
4. 결과 시각화 및 저장
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import cv2
import time
from typing import List, Tuple, Dict

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from pedsim_msgs.msg import AgentStates
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Error: ROS not available")
    sys.exit(1)

from warehouse_config import CCTV_CONFIGS, is_in_valid_region, SHELVES, AISLES
from detector import PersonDetector
from homography import HomographyManager
from human_state_extractor import HumanStateExtractor, HumanState
from visualizer import WarehouseVisualizer, compute_detection_metrics


class DetectionTestNode:
    """실시간 검출 테스트 노드"""

    def __init__(self):
        rospy.init_node('detection_test_node', anonymous=True)

        self.bridge = CvBridge()
        self.num_cctvs = 4

        # 출력 디렉토리
        self.output_dir = os.path.join(SCRIPT_DIR, 'detection_output')
        os.makedirs(self.output_dir, exist_ok=True)

        # YOLO 검출기
        print("Loading YOLO model...")
        self.detector = PersonDetector(
            model_path='yolov8n.pt',
            confidence_threshold=0.3
        )
        print("YOLO model loaded")

        # Homography - 기존 캘리브레이션 로드
        print("Loading calibration...")
        self.homography_manager = HomographyManager()
        self.homography_manager.load_all()
        print("Calibration ready")

        # 상태 추출기
        self.extractor = HumanStateExtractor(history_length=8, dt=0.4)

        # 시각화
        self.visualizer = WarehouseVisualizer()

        # 데이터 저장
        self.latest_images: Dict[int, np.ndarray] = {}
        self.ground_truth: List[HumanState] = []
        self.frame_count = 0

        # ROS 설정
        self._setup_ros()

        print("Detection Test Node initialized")

    def _setup_ros(self):
        """ROS subscriber 설정"""
        for cctv_id in range(self.num_cctvs):
            topic = f'/cctv_{cctv_id}/image_raw'
            rospy.Subscriber(
                topic, Image,
                self._image_callback,
                callback_args=cctv_id,
                queue_size=1
            )
            rospy.loginfo(f"Subscribed to {topic}")

        rospy.Subscriber(
            '/pedsim_simulator/simulated_agents',
            AgentStates,
            self._gt_callback,
            queue_size=1
        )
        rospy.loginfo("Subscribed to Ground Truth")

        self.timer = rospy.Timer(rospy.Duration(1.0), self._process_callback)

    def _image_callback(self, msg: Image, cctv_id: int):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_images[cctv_id] = cv_image
        except Exception as e:
            rospy.logerr(f"Image conversion error (CCTV {cctv_id}): {e}")

    def _gt_callback(self, msg: AgentStates):
        gt_list = []
        for agent in msg.agent_states:
            gt_list.append(HumanState(
                id=agent.id,
                px=agent.pose.position.x,
                py=agent.pose.position.y,
                vx=agent.twist.linear.x,
                vy=agent.twist.linear.y,
                radius=0.3
            ))
        self.ground_truth = gt_list

    def _process_callback(self, event):
        if len(self.latest_images) < self.num_cctvs:
            rospy.loginfo(f"Waiting for all CCTVs... ({len(self.latest_images)}/{self.num_cctvs})")
            return

        self.frame_count += 1
        rospy.loginfo(f"\n{'='*50}")
        rospy.loginfo(f"Processing frame {self.frame_count}")

        # 각 CCTV에서 검출
        all_detections = {}
        all_raw_pixels = {}

        for cctv_id, image in self.latest_images.items():
            foot_pixels = self.detector.detect_feet_only(image)
            all_raw_pixels[cctv_id] = foot_pixels

            rospy.loginfo(f"  CCTV {cctv_id}: {len(foot_pixels)} detections")

            if foot_pixels:
                try:
                    world_coords = self.homography_manager.transform(cctv_id, foot_pixels)
                    valid_coords = []
                    for (px, py), (wx, wy) in zip(foot_pixels, world_coords):
                        if is_in_valid_region(wx, wy):
                            valid_coords.append((wx, wy))
                            rospy.loginfo(f"    pixel ({px:.0f}, {py:.0f}) -> world ({wx:.2f}, {wy:.2f}) [valid]")
                        else:
                            rospy.loginfo(f"    pixel ({px:.0f}, {py:.0f}) -> world ({wx:.2f}, {wy:.2f}) [INVALID]")
                    all_detections[cctv_id] = valid_coords
                except Exception as e:
                    rospy.logerr(f"    Homography error: {e}")
                    all_detections[cctv_id] = []
            else:
                all_detections[cctv_id] = []

        # 검출 결과 융합
        merged = self._merge_detections(all_detections)
        rospy.loginfo(f"  Merged detections: {len(merged)}")

        # Ground Truth 정보
        rospy.loginfo(f"  Ground Truth: {len(self.ground_truth)} humans")
        for gt in self.ground_truth:
            rospy.loginfo(f"    GT {gt.id}: ({gt.px:.2f}, {gt.py:.2f})")

        # 지표 계산
        if self.ground_truth and merged:
            metrics = compute_detection_metrics(self.ground_truth, merged)
            rospy.loginfo(f"\n  === Metrics ===")
            rospy.loginfo(f"  Precision: {metrics['precision']:.3f}")
            rospy.loginfo(f"  Recall:    {metrics['recall']:.3f}")
            rospy.loginfo(f"  F1 Score:  {metrics['f1_score']:.3f}")
            rospy.loginfo(f"  Mean Error: {metrics['mean_position_error']:.3f}m")
            rospy.loginfo(f"  TP={metrics['true_positives']}, FP={metrics['false_positives']}, FN={metrics['false_negatives']}")

        # 시각화 저장
        self._save_visualization(all_raw_pixels, all_detections, merged)
        self._save_cctv_images(all_raw_pixels)

    def _merge_detections(self, all_detections: Dict[int, List[Tuple[float, float]]],
                          distance_threshold: float = 0.5) -> List[Tuple[float, float]]:
        all_points = []
        for coords in all_detections.values():
            all_points.extend(coords)

        if not all_points:
            return []

        merged = []
        used = [False] * len(all_points)

        for i, (x1, y1) in enumerate(all_points):
            if used[i]:
                continue
            cluster = [(x1, y1)]
            used[i] = True

            for j, (x2, y2) in enumerate(all_points):
                if used[j]:
                    continue
                dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                if dist < distance_threshold:
                    cluster.append((x2, y2))
                    used[j] = True

            cx = np.mean([p[0] for p in cluster])
            cy = np.mean([p[1] for p in cluster])
            merged.append((cx, cy))

        return merged

    def _save_visualization(self, raw_pixels, all_detections, merged):
        self.visualizer.create_figure()

        if self.ground_truth:
            self.visualizer.draw_ground_truth(
                self.ground_truth, color='blue', label='Ground Truth'
            )

        if merged:
            self.visualizer.draw_detections(
                merged, color='red', marker='x', label='Detection'
            )

        self.visualizer.add_legend()

        title = f"Frame {self.frame_count} | GT: {len(self.ground_truth)} | Det: {len(merged)}"
        if self.ground_truth and merged:
            metrics = compute_detection_metrics(self.ground_truth, merged)
            title += f" | P={metrics['precision']:.2f} R={metrics['recall']:.2f}"
        self.visualizer.ax.set_title(title)

        filepath = os.path.join(self.output_dir, f'frame_{self.frame_count:04d}.png')
        self.visualizer.save(filepath, dpi=100)
        self.visualizer.close()

    def _save_cctv_images(self, all_raw_pixels):
        for cctv_id, image in self.latest_images.items():
            img_copy = image.copy()
            pixels = all_raw_pixels.get(cctv_id, [])
            for px, py in pixels:
                cv2.circle(img_copy, (int(px), int(py)), 5, (0, 0, 255), -1)
                cv2.circle(img_copy, (int(px), int(py)), 10, (0, 255, 0), 2)

            cv2.putText(img_copy, f'CCTV {cctv_id}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img_copy, f'Detections: {len(pixels)}', (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            filepath = os.path.join(self.output_dir, f'cctv_{cctv_id}_frame_{self.frame_count:04d}.jpg')
            cv2.imwrite(filepath, img_copy)

    def run(self):
        rospy.loginfo("Detection Test Node running...")
        rospy.loginfo(f"Output directory: {self.output_dir}")
        rospy.spin()


def main():
    if not ROS_AVAILABLE:
        print("ROS not available")
        return

    try:
        node = DetectionTestNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
