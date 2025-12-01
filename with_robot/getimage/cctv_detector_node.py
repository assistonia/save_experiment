#!/usr/bin/env python3
"""
CCTV Detection ROS Node

CCTV 이미지를 받아서:
1. YOLO로 사람 검출
2. Homography로 월드 좌표 변환
3. History Buffer 관리
4. HumanState 퍼블리시
5. Ground Truth와 비교 (검증용)

Subscribe:
    - /cctv_X/image_raw: CCTV 이미지
    - /pedsim_simulator/simulated_agents: Ground Truth (검증용)

Publish:
    - /cctv_detection/human_states: 검출된 HumanState
    - /cctv_detection/visualization: 시각화 이미지
"""

import sys
import os

# 현재 디렉토리를 path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import time
import threading

try:
    import rospy
    from sensor_msgs.msg import Image
    from geometry_msgs.msg import Point
    from std_msgs.msg import Header
    from cv_bridge import CvBridge
    from pedsim_msgs.msg import AgentStates
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS not available. Running in standalone mode.")

from warehouse_config import CCTV_CONFIGS, is_in_valid_region
from detector import PersonDetector
from homography import HomographyManager
from human_state_extractor import HumanStateExtractor, HumanState
from visualizer import WarehouseVisualizer, compute_detection_metrics


class CCTVDetectorNode:
    """
    CCTV 검출 ROS 노드.

    4개 CCTV 이미지를 받아서 사람 위치를 검출하고 융합.
    """

    def __init__(self):
        if ROS_AVAILABLE:
            rospy.init_node('cctv_detector_node', anonymous=True)

        # 설정
        self.num_cctvs = 4
        self.confidence_threshold = 0.5
        self.visualization_enabled = True
        self.save_visualization = True
        self.viz_save_dir = os.path.join(SCRIPT_DIR, 'viz_output')
        os.makedirs(self.viz_save_dir, exist_ok=True)

        # 컴포넌트 초기화
        self.detector = PersonDetector(
            model_path='yolov8n.pt',
            confidence_threshold=self.confidence_threshold
        )

        self.homography_manager = HomographyManager(
            calibration_dir=os.path.join(SCRIPT_DIR, 'calibration')
        )
        self.homography_manager.load_all()

        self.extractor = HumanStateExtractor(
            history_length=8,
            dt=0.4,
            max_association_dist=1.0
        )

        self.visualizer = WarehouseVisualizer()

        # 상태
        self.latest_images: Dict[int, np.ndarray] = {}
        self.latest_detections: Dict[int, List[Tuple[float, float]]] = {}
        self.ground_truth: List[HumanState] = []
        self.frame_count = 0

        # 스레드 안전
        self.lock = threading.Lock()

        # ROS 설정
        if ROS_AVAILABLE:
            self.bridge = CvBridge()
            self._setup_ros()

        print("CCTV Detector Node initialized")

    def _setup_ros(self):
        """ROS subscriber/publisher 설정"""
        # CCTV 이미지 구독
        self.image_subs = []
        for cctv_id in range(self.num_cctvs):
            topic = f'/cctv_{cctv_id}/image_raw'
            sub = rospy.Subscriber(
                topic, Image,
                self._image_callback,
                callback_args=cctv_id,
                queue_size=1
            )
            self.image_subs.append(sub)
            rospy.loginfo(f"Subscribed to {topic}")

        # Ground Truth 구독 (검증용)
        rospy.Subscriber(
            '/pedsim_simulator/simulated_agents',
            AgentStates,
            self._gt_callback,
            queue_size=1
        )

        # 타이머 (메인 처리 루프)
        self.timer = rospy.Timer(rospy.Duration(0.1), self._process_callback)

    def _image_callback(self, msg: Image, cctv_id: int):
        """CCTV 이미지 콜백"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.lock:
                self.latest_images[cctv_id] = cv_image
        except Exception as e:
            rospy.logerr(f"Image conversion error (CCTV {cctv_id}): {e}")

    def _gt_callback(self, msg: AgentStates):
        """Ground Truth 콜백"""
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
        with self.lock:
            self.ground_truth = gt_list

    def _process_callback(self, event):
        """메인 처리 루프"""
        self.process_frame()

    def detect_from_image(self, cctv_id: int, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        단일 CCTV 이미지에서 사람 검출 → 월드 좌표.

        Args:
            cctv_id: CCTV ID
            image: BGR 이미지

        Returns:
            [(x, y), ...] 월드 좌표 리스트
        """
        # 1. YOLO 검출 → 발 픽셀 좌표
        foot_pixels = self.detector.detect_feet_only(image)

        if not foot_pixels:
            return []

        # 2. Homography 변환 → 월드 좌표
        try:
            world_coords = self.homography_manager.transform(cctv_id, foot_pixels)
        except ValueError as e:
            print(f"CCTV {cctv_id}: Homography error - {e}")
            return []

        # 3. 유효 영역 필터링
        valid_coords = []
        for x, y in world_coords:
            if is_in_valid_region(x, y):
                valid_coords.append((x, y))
            else:
                print(f"CCTV {cctv_id}: Invalid detection at ({x:.2f}, {y:.2f})")

        return valid_coords

    def merge_detections(self, all_detections: Dict[int, List[Tuple[float, float]]],
                         distance_threshold: float = 0.5) -> List[Tuple[float, float]]:
        """
        여러 CCTV 검출 결과 융합 (중복 제거).

        Args:
            all_detections: {cctv_id: [(x, y), ...]}
            distance_threshold: 중복 판정 거리

        Returns:
            융합된 좌표 리스트
        """
        all_points = []
        for cctv_id, coords in all_detections.items():
            all_points.extend(coords)

        if not all_points:
            return []

        # 간단한 거리 기반 클러스터링
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

            # 클러스터 중심
            cx = np.mean([p[0] for p in cluster])
            cy = np.mean([p[1] for p in cluster])
            merged.append((cx, cy))

        return merged

    def process_frame(self) -> List[HumanState]:
        """
        한 프레임 처리.

        Returns:
            검출된 HumanState 리스트
        """
        with self.lock:
            images = dict(self.latest_images)
            gt = list(self.ground_truth)

        self.frame_count += 1
        timestamp = time.time()

        # 1. 각 CCTV에서 검출
        all_detections = {}
        for cctv_id, image in images.items():
            detections = self.detect_from_image(cctv_id, image)
            all_detections[cctv_id] = detections
            self.latest_detections[cctv_id] = detections

        # 2. 검출 결과 융합
        merged_detections = self.merge_detections(all_detections)

        # 3. History Buffer 업데이트
        self.extractor.update(merged_detections, timestamp)

        # 4. HumanState 추출
        human_states = self.extractor.get_human_states()

        # 5. 검증 및 시각화
        if self.visualization_enabled and self.frame_count % 10 == 0:
            self._visualize(gt, human_states, merged_detections)

        return human_states

    def _visualize(self, gt: List[HumanState],
                   detected: List[HumanState],
                   raw_detections: List[Tuple[float, float]]):
        """시각화 및 저장"""
        self.visualizer.create_figure()

        # GT 그리기
        if gt:
            self.visualizer.draw_ground_truth(gt, color='blue', label='Ground Truth')

        # 검출 결과 그리기
        if raw_detections:
            self.visualizer.draw_detections(raw_detections, color='red', label='Raw Detection')

        if detected:
            self.visualizer.draw_detection_states(detected, color='green', label='Tracked')

        # 오차 표시
        if gt and detected:
            self.visualizer.draw_errors(gt, detected)

        self.visualizer.add_legend()

        # 지표 계산
        if gt and raw_detections:
            metrics = compute_detection_metrics(gt, raw_detections)
            title = f"Frame {self.frame_count} | "
            title += f"P={metrics['precision']:.2f} R={metrics['recall']:.2f} "
            title += f"Err={metrics['mean_position_error']:.2f}m"
            self.visualizer.ax.set_title(title)

        # 저장
        if self.save_visualization:
            filepath = os.path.join(self.viz_save_dir, f'frame_{self.frame_count:05d}.png')
            self.visualizer.save(filepath, dpi=100)

        self.visualizer.close()

    def get_trajectory_tensor(self):
        """SingularTrajectory 입력 텐서 반환"""
        return self.extractor.get_trajectory_tensor()

    def run(self):
        """노드 실행"""
        if ROS_AVAILABLE:
            rospy.loginfo("CCTV Detector Node running...")
            rospy.spin()
        else:
            print("ROS not available. Use process_frame() manually.")


class StandaloneTest:
    """
    ROS 없이 테스트용 클래스.

    저장된 이미지 또는 더미 데이터로 테스트.
    """

    def __init__(self):
        self.node = CCTVDetectorNode()

    def test_with_dummy_data(self):
        """더미 데이터로 테스트"""
        print("=== Standalone Test with Dummy Data ===")

        # 더미 GT
        gt_humans = [
            HumanState(id=0, px=-8.5, py=2, vx=0, vy=-0.5),
            HumanState(id=1, px=1.5, py=0, vx=0.3, vy=0),
        ]
        self.node.ground_truth = gt_humans

        # 더미 검출 (GT 근처 + 약간 오차)
        dummy_detections = [
            (-8.3, 2.1),
            (1.6, 0.15),
        ]

        # 10프레임 시뮬레이션
        for frame in range(10):
            print(f"\n--- Frame {frame} ---")

            # 이동 시뮬레이션
            detections = [
                (dummy_detections[0][0], dummy_detections[0][1] - 0.05 * frame),
                (dummy_detections[1][0] + 0.03 * frame, dummy_detections[1][1]),
            ]

            # extractor 업데이트
            self.node.extractor.update(detections, timestamp=frame * 0.4)

            # 결과
            states = self.node.extractor.get_human_states()
            for s in states:
                print(f"  ID {s.id}: ({s.px:.2f}, {s.py:.2f}), v=({s.vx:.2f}, {s.vy:.2f})")

        # 최종 시각화
        self.node._visualize(gt_humans, states, detections)
        print(f"\nVisualization saved to: {self.node.viz_save_dir}")


def main():
    if ROS_AVAILABLE:
        try:
            node = CCTVDetectorNode()
            node.run()
        except rospy.ROSInterruptException:
            pass
    else:
        # 스탠드얼론 테스트
        test = StandaloneTest()
        test.test_with_dummy_data()


if __name__ == '__main__':
    main()
