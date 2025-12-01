#!/usr/bin/env python3
"""
자동 Homography 캘리브레이션

GT 사람 위치와 YOLO 검출을 매칭해서 Homography를 자동 계산.
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import cv2
import json
import time
from typing import List, Tuple, Dict
from collections import defaultdict

try:
    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from pedsim_msgs.msg import AgentStates
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available")
    sys.exit(1)

from warehouse_config import CCTV_CONFIGS, is_in_valid_region
from detector import PersonDetector


class AutoCalibrator:
    """GT와 검출 매칭으로 자동 캘리브레이션"""

    def __init__(self, num_samples=50):
        rospy.init_node('auto_calibrator', anonymous=True)

        self.bridge = CvBridge()
        self.num_cctvs = 4
        self.num_samples = num_samples

        # YOLO
        print("Loading YOLO...")
        self.detector = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.4)

        # 데이터 저장
        self.latest_images: Dict[int, np.ndarray] = {}
        self.ground_truth = []

        # 캘리브레이션 데이터: {cctv_id: [(pixel, world), ...]}
        self.calib_data: Dict[int, List[Tuple[Tuple, Tuple]]] = defaultdict(list)

        self._setup_ros()
        print("AutoCalibrator ready")

    def _setup_ros(self):
        for cctv_id in range(self.num_cctvs):
            rospy.Subscriber(
                f'/cctv_{cctv_id}/image_raw', Image,
                self._image_callback, callback_args=cctv_id, queue_size=1
            )

        rospy.Subscriber(
            '/pedsim_simulator/simulated_agents',
            AgentStates, self._gt_callback, queue_size=1
        )

    def _image_callback(self, msg, cctv_id):
        try:
            self.latest_images[cctv_id] = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            pass

    def _gt_callback(self, msg):
        self.ground_truth = []
        for agent in msg.agent_states:
            self.ground_truth.append({
                'id': agent.id,
                'x': agent.pose.position.x,
                'y': agent.pose.position.y,
            })

    def _get_cctv_visible_region(self, cctv_id):
        """각 CCTV가 볼 수 있는 대략적인 영역"""
        regions = {
            0: {'x': (-12, -5), 'y': (-6, 10)},   # Aisle 1
            1: {'x': (-7, 0), 'y': (-10, 6)},     # Aisle 2
            2: {'x': (-2, 5), 'y': (-6, 10)},     # Aisle 3
            3: {'x': (3, 12), 'y': (-10, 6)},     # Aisle 4
        }
        return regions.get(cctv_id)

    def _is_in_cctv_view(self, cctv_id, x, y):
        """해당 좌표가 CCTV 시야 안에 있는지"""
        region = self._get_cctv_visible_region(cctv_id)
        if region is None:
            return False
        return (region['x'][0] <= x <= region['x'][1] and
                region['y'][0] <= y <= region['y'][1])

    def _match_detection_to_gt(self, cctv_id, detections, gt_list):
        """
        검출과 GT 매칭.

        CCTV 시야 내의 GT만 고려하고, 가장 가까운 검출과 매칭.
        """
        matches = []

        # CCTV 시야 내 GT만 필터링
        visible_gt = [g for g in gt_list
                      if self._is_in_cctv_view(cctv_id, g['x'], g['y'])
                      and is_in_valid_region(g['x'], g['y'])]

        if not visible_gt or not detections:
            return matches

        # 간단한 매칭: 검출 수 == GT 수일 때만 (신뢰도 높음)
        if len(detections) == len(visible_gt) and len(detections) <= 3:
            # x좌표 기준 정렬해서 매칭
            det_sorted = sorted(detections, key=lambda p: p[0])
            gt_sorted = sorted(visible_gt, key=lambda g: g['x'])

            for det, gt in zip(det_sorted, gt_sorted):
                matches.append((det, (gt['x'], gt['y'])))

        return matches

    def collect_samples(self):
        """캘리브레이션 샘플 수집"""
        print(f"\nCollecting {self.num_samples} samples per CCTV...")
        print("Move people around in the simulation for better coverage.\n")

        rate = rospy.Rate(2)  # 2 Hz
        sample_count = 0

        while not rospy.is_shutdown():
            if len(self.latest_images) < self.num_cctvs:
                rate.sleep()
                continue

            for cctv_id, image in self.latest_images.items():
                # YOLO 검출
                foot_pixels = self.detector.detect_feet_only(image)

                if not foot_pixels:
                    continue

                # GT와 매칭
                matches = self._match_detection_to_gt(cctv_id, foot_pixels, self.ground_truth)

                for pixel, world in matches:
                    self.calib_data[cctv_id].append((pixel, world))

            # 진행 상황 출력
            sample_count += 1
            if sample_count % 10 == 0:
                print(f"Sample {sample_count}:")
                for cctv_id in range(self.num_cctvs):
                    count = len(self.calib_data[cctv_id])
                    print(f"  CCTV {cctv_id}: {count} pairs")

            # 충분한 샘플 수집 확인
            min_samples = min(len(self.calib_data[i]) for i in range(self.num_cctvs))
            if min_samples >= self.num_samples:
                print(f"\nCollected enough samples!")
                break

            if sample_count > self.num_samples * 5:
                print(f"\nMax iterations reached")
                break

            rate.sleep()

    def compute_homography(self):
        """수집된 데이터로 Homography 계산"""
        print("\n=== Computing Homography ===")

        calibration_dir = os.path.join(SCRIPT_DIR, 'calibration')
        os.makedirs(calibration_dir, exist_ok=True)

        results = {}

        for cctv_id in range(self.num_cctvs):
            pairs = self.calib_data[cctv_id]
            print(f"\nCCTV {cctv_id}: {len(pairs)} pairs")

            if len(pairs) < 4:
                print(f"  Not enough pairs (need >= 4)")
                continue

            # 중복 제거 및 정리
            unique_pairs = list(set(pairs))
            print(f"  Unique pairs: {len(unique_pairs)}")

            # numpy 배열로 변환
            pixels = np.array([p[0] for p in unique_pairs], dtype=np.float32)
            worlds = np.array([p[1] for p in unique_pairs], dtype=np.float32)

            # Homography 계산
            H, status = cv2.findHomography(pixels, worlds, cv2.RANSAC, 5.0)

            if H is None:
                print(f"  Failed to compute homography")
                continue

            # 저장
            npy_path = os.path.join(calibration_dir, f'cctv_{cctv_id}.npy')
            np.save(npy_path, H)
            print(f"  Saved: {npy_path}")

            # JSON으로 포인트도 저장
            json_path = os.path.join(calibration_dir, f'cctv_{cctv_id}_points.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'pixel_points': [[float(x) for x in p[0]] for p in unique_pairs],
                    'world_points': [[float(x) for x in p[1]] for p in unique_pairs],
                }, f, indent=2)
            print(f"  Saved: {json_path}")

            # 테스트
            test_error = self._test_homography(H, unique_pairs[:10])
            print(f"  Test error: {test_error:.3f}m")

            results[cctv_id] = {
                'num_pairs': len(unique_pairs),
                'error': test_error
            }

        return results

    def _test_homography(self, H, pairs):
        """Homography 테스트"""
        errors = []
        for pixel, world_gt in pairs:
            # 변환
            pt = np.array([pixel[0], pixel[1], 1], dtype=np.float32)
            world_pred = H @ pt
            world_pred = world_pred[:2] / world_pred[2]

            # 오차
            error = np.sqrt((world_pred[0] - world_gt[0])**2 +
                           (world_pred[1] - world_gt[1])**2)
            errors.append(error)

        return np.mean(errors) if errors else 999

    def run(self):
        print("=== Auto Calibration ===")
        print("Waiting for data...")
        time.sleep(2)

        self.collect_samples()
        results = self.compute_homography()

        print("\n=== Results ===")
        for cctv_id, res in results.items():
            print(f"CCTV {cctv_id}: {res['num_pairs']} pairs, error={res['error']:.3f}m")


def main():
    try:
        calibrator = AutoCalibrator(num_samples=30)
        calibrator.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
