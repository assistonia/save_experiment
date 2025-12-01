"""
YOLO-based Person Detector

CCTV 이미지에서 사람을 검출하고 발 위치(픽셀)를 반환.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


@dataclass
class Detection:
    """검출 결과"""
    bbox: Tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max)
    confidence: float
    foot_pixel: Tuple[float, float]  # 발 위치 (bbox 하단 중앙)


class PersonDetector:
    """
    YOLO 기반 사람 검출기.

    BBox 하단 중앙을 발 위치로 추정.
    """

    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Args:
            model_path: YOLO 모델 경로 (기본: yolov8n.pt 경량 모델)
            confidence_threshold: 검출 신뢰도 임계값
        """
        self.confidence_threshold = confidence_threshold

        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
            print(f"YOLO model loaded: {model_path}")
        else:
            self.model = None
            print("YOLO not available. Using dummy detector.")

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        이미지에서 사람 검출.

        Args:
            image: BGR 이미지 (H, W, 3)

        Returns:
            Detection 리스트
        """
        if self.model is None:
            return []

        # YOLO 추론 (class 0 = person)
        results = self.model(image, classes=[0], verbose=False)

        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue

                # BBox 추출
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()

                # 발 위치 = BBox 하단 중앙
                foot_x = (x_min + x_max) / 2
                foot_y = y_max  # 하단

                detections.append(Detection(
                    bbox=(x_min, y_min, x_max, y_max),
                    confidence=conf,
                    foot_pixel=(foot_x, foot_y)
                ))

        return detections

    def detect_feet_only(self, image: np.ndarray) -> List[Tuple[float, float]]:
        """
        발 위치(픽셀)만 반환.

        Args:
            image: BGR 이미지

        Returns:
            [(foot_x, foot_y), ...] 픽셀 좌표 리스트
        """
        detections = self.detect(image)
        return [d.foot_pixel for d in detections]


class DummyDetector:
    """
    테스트용 더미 검출기.

    Ground Truth 좌표를 받아서 검출 결과처럼 반환.
    (YOLO 없이 Homography 테스트용)
    """

    def __init__(self):
        pass

    def set_ground_truth(self, world_coords: List[Tuple[float, float]],
                         homography_inverse: np.ndarray):
        """
        GT 월드 좌표를 픽셀 좌표로 변환해서 저장.

        Args:
            world_coords: [(x, y), ...] 월드 좌표
            homography_inverse: 월드→픽셀 변환 행렬 (H^-1)
        """
        self.pixel_coords = []
        for wx, wy in world_coords:
            # 월드 → 픽셀 역변환
            world_pt = np.array([[wx, wy]], dtype=np.float32).reshape(-1, 1, 2)
            pixel_pt = cv2.perspectiveTransform(world_pt, homography_inverse)
            px, py = pixel_pt[0, 0]
            self.pixel_coords.append((px, py))

    def detect_feet_only(self, image: np.ndarray = None) -> List[Tuple[float, float]]:
        """저장된 픽셀 좌표 반환"""
        return self.pixel_coords if hasattr(self, 'pixel_coords') else []


if __name__ == '__main__':
    import cv2

    # 테스트
    print("=== Person Detector Test ===")

    # 더미 이미지로 테스트
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = PersonDetector(confidence_threshold=0.5)
    detections = detector.detect(dummy_image)

    print(f"Detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"  {i}: bbox={det.bbox}, conf={det.confidence:.2f}, foot={det.foot_pixel}")
