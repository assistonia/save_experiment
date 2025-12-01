"""
Homography Transform

CCTV 이미지 픽셀 좌표 → 월드 좌표 변환.
각 CCTV마다 별도의 Homography 행렬 필요.
"""

import numpy as np
import cv2
import os
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from warehouse_config import CCTV_CONFIGS, CCTVConfig


@dataclass
class CalibrationPoint:
    """캘리브레이션 대응점"""
    pixel: Tuple[float, float]   # (u, v) 픽셀 좌표
    world: Tuple[float, float]   # (x, y) 월드 좌표


class HomographyTransformer:
    """
    Homography 기반 좌표 변환기.

    픽셀 좌표 (u, v) → 월드 좌표 (x, y)
    """

    def __init__(self, cctv_id: int, H: np.ndarray = None):
        """
        Args:
            cctv_id: CCTV ID
            H: 3x3 Homography 행렬 (픽셀→월드)
        """
        self.cctv_id = cctv_id
        self.H = H
        self.H_inv = np.linalg.inv(H) if H is not None else None
        self.config = CCTV_CONFIGS.get(cctv_id)

    def pixel_to_world(self, pixel_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        픽셀 좌표 → 월드 좌표 변환.

        Args:
            pixel_coords: [(u, v), ...] 픽셀 좌표 리스트

        Returns:
            [(x, y), ...] 월드 좌표 리스트
        """
        if self.H is None:
            raise ValueError("Homography matrix not set. Run calibration first.")

        if not pixel_coords:
            return []

        # OpenCV 형식으로 변환
        pts = np.array(pixel_coords, dtype=np.float32).reshape(-1, 1, 2)

        # Homography 변환
        world_pts = cv2.perspectiveTransform(pts, self.H)

        return [(float(pt[0, 0]), float(pt[0, 1])) for pt in world_pts]

    def world_to_pixel(self, world_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        월드 좌표 → 픽셀 좌표 역변환 (검증용).

        Args:
            world_coords: [(x, y), ...] 월드 좌표 리스트

        Returns:
            [(u, v), ...] 픽셀 좌표 리스트
        """
        if self.H_inv is None:
            raise ValueError("Homography matrix not set.")

        if not world_coords:
            return []

        pts = np.array(world_coords, dtype=np.float32).reshape(-1, 1, 2)
        pixel_pts = cv2.perspectiveTransform(pts, self.H_inv)

        return [(float(pt[0, 0]), float(pt[0, 1])) for pt in pixel_pts]

    def save(self, filepath: str):
        """Homography 행렬 저장"""
        np.save(filepath, self.H)
        print(f"Saved homography to {filepath}")

    @classmethod
    def load(cls, cctv_id: int, filepath: str) -> 'HomographyTransformer':
        """저장된 Homography 행렬 로드"""
        H = np.load(filepath)
        return cls(cctv_id, H)

    @classmethod
    def from_points(cls, cctv_id: int,
                    pixel_points: List[Tuple[float, float]],
                    world_points: List[Tuple[float, float]]) -> 'HomographyTransformer':
        """
        대응점으로부터 Homography 계산.

        Args:
            cctv_id: CCTV ID
            pixel_points: 4개 이상의 픽셀 좌표
            world_points: 대응하는 월드 좌표

        Returns:
            HomographyTransformer 인스턴스
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            raise ValueError("At least 4 corresponding points required")

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)

        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            raise ValueError("Failed to compute homography")

        return cls(cctv_id, H)


class HomographyManager:
    """
    모든 CCTV의 Homography 관리.
    """

    def __init__(self, calibration_dir: str = None):
        """
        Args:
            calibration_dir: 캘리브레이션 파일 저장 디렉토리
        """
        self.calibration_dir = calibration_dir or os.path.join(
            os.path.dirname(__file__), 'calibration'
        )
        os.makedirs(self.calibration_dir, exist_ok=True)

        self.transformers: Dict[int, HomographyTransformer] = {}

    def get_transformer(self, cctv_id: int) -> Optional[HomographyTransformer]:
        """특정 CCTV의 transformer 반환"""
        return self.transformers.get(cctv_id)

    def set_transformer(self, cctv_id: int, transformer: HomographyTransformer):
        """transformer 설정"""
        self.transformers[cctv_id] = transformer

    def calibrate(self, cctv_id: int,
                  pixel_points: List[Tuple[float, float]],
                  world_points: List[Tuple[float, float]]):
        """
        특정 CCTV 캘리브레이션.

        Args:
            cctv_id: CCTV ID
            pixel_points: 픽셀 좌표 리스트
            world_points: 월드 좌표 리스트
        """
        transformer = HomographyTransformer.from_points(
            cctv_id, pixel_points, world_points
        )
        self.transformers[cctv_id] = transformer

        # 저장
        filepath = os.path.join(self.calibration_dir, f'cctv_{cctv_id}.npy')
        transformer.save(filepath)

    def load_all(self):
        """모든 CCTV 캘리브레이션 로드"""
        for cctv_id in CCTV_CONFIGS.keys():
            filepath = os.path.join(self.calibration_dir, f'cctv_{cctv_id}.npy')
            if os.path.exists(filepath):
                self.transformers[cctv_id] = HomographyTransformer.load(cctv_id, filepath)
                print(f"Loaded calibration for CCTV {cctv_id}")
            else:
                print(f"No calibration found for CCTV {cctv_id}")

    def save_calibration_points(self, cctv_id: int,
                                 pixel_points: List[Tuple[float, float]],
                                 world_points: List[Tuple[float, float]]):
        """캘리브레이션 대응점 JSON으로 저장 (재현용)"""
        data = {
            'cctv_id': cctv_id,
            'pixel_points': pixel_points,
            'world_points': world_points
        }
        filepath = os.path.join(self.calibration_dir, f'cctv_{cctv_id}_points.json')
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved calibration points to {filepath}")

    def transform(self, cctv_id: int,
                  pixel_coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        특정 CCTV의 픽셀→월드 변환.

        Args:
            cctv_id: CCTV ID
            pixel_coords: 픽셀 좌표 리스트

        Returns:
            월드 좌표 리스트
        """
        transformer = self.transformers.get(cctv_id)
        if transformer is None:
            raise ValueError(f"No calibration for CCTV {cctv_id}")

        return transformer.pixel_to_world(pixel_coords)


def create_default_calibration():
    """
    수동 캘리브레이션 데이터 사용.

    선반 모서리 바닥 접점을 기준으로 캘리브레이션.
    """
    manager = HomographyManager()

    # CCTV 0: (-8.5, 10, 3), 남쪽(아래)을 봄
    # 이미지 상단 = 멀리(y가 작은 쪽), 하단 = 가까이(y가 큰 쪽)
    # 보이는 선반: shelf_1 (왼쪽), shelf_2 (오른쪽)
    cctv0_pixel = [
        (195, 400),   # shelf_1 오른쪽, 이미지 하단(가까이) → y=4
        (85, 155),    # shelf_1 오른쪽, 이미지 상단(멀리) → y=-5
        (445, 400),   # shelf_2 왼쪽, 이미지 하단(가까이) → y=4
        (545, 155),   # shelf_2 왼쪽, 이미지 상단(멀리) → y=-5
    ]
    cctv0_world = [
        (-10, 4),     # 하단 = 가까이 = y=4
        (-10, -5),    # 상단 = 멀리 = y=-5
        (-7, 4),
        (-7, -5),
    ]

    # CCTV 1: (-3.5, -10, 3), 북쪽(위)을 봄
    # 이미지 상단 = 멀리(y가 큰 쪽), 하단 = 가까이(y가 작은 쪽)
    # 보이는 선반: shelf_2 (왼쪽), shelf_3 (오른쪽)
    cctv1_pixel = [
        (185, 390),   # shelf_2 오른쪽, 이미지 하단(가까이) → y=-5
        (75, 140),    # shelf_2 오른쪽, 이미지 상단(멀리) → y=4
        (455, 390),   # shelf_3 왼쪽, 이미지 하단(가까이) → y=-5
        (560, 140),   # shelf_3 왼쪽, 이미지 상단(멀리) → y=4
    ]
    cctv1_world = [
        (-5, -5),     # 하단 = 가까이 = y=-5
        (-5, 4),      # 상단 = 멀리 = y=4
        (-2, -5),
        (-2, 4),
    ]

    # CCTV 2: (1.5, 10, 3), 남쪽(아래)을 봄 (CCTV 0과 같은 방향)
    # 보이는 선반: shelf_3 (왼쪽), shelf_4 (오른쪽)
    cctv2_pixel = [
        (190, 395),   # shelf_3 오른쪽, 이미지 하단(가까이) → y=4
        (80, 145),    # shelf_3 오른쪽, 이미지 상단(멀리) → y=-5
        (450, 395),   # shelf_4 왼쪽, 이미지 하단(가까이) → y=4
        (555, 145),   # shelf_4 왼쪽, 이미지 상단(멀리) → y=-5
    ]
    cctv2_world = [
        (0, 4),
        (0, -5),
        (3, 4),
        (3, -5),
    ]

    # CCTV 3: (7.5, -10, 3), 북쪽(위)을 봄 (CCTV 1과 같은 방향)
    # 보이는 선반: shelf_4 (왼쪽), shelf_5 (오른쪽)
    cctv3_pixel = [
        (175, 385),   # shelf_4 오른쪽, 이미지 하단(가까이) → y=-5
        (70, 130),    # shelf_4 오른쪽, 이미지 상단(멀리) → y=4
        (460, 385),   # shelf_5 왼쪽, 이미지 하단(가까이) → y=-5
        (565, 130),   # shelf_5 왼쪽, 이미지 상단(멀리) → y=4
    ]
    cctv3_world = [
        (5, -5),
        (5, 4),
        (10, -5),
        (10, 4),
    ]

    calibrations = [
        (0, cctv0_pixel, cctv0_world),
        (1, cctv1_pixel, cctv1_world),
        (2, cctv2_pixel, cctv2_world),
        (3, cctv3_pixel, cctv3_world),
    ]

    for cctv_id, pixel_pts, world_pts in calibrations:
        try:
            manager.calibrate(cctv_id, pixel_pts, world_pts)
            manager.save_calibration_points(cctv_id, pixel_pts, world_pts)
            print(f"CCTV {cctv_id}: Calibration created")
        except Exception as e:
            print(f"CCTV {cctv_id}: Calibration failed - {e}")

    return manager


if __name__ == '__main__':
    print("=== Homography Test ===")

    # 기본 캘리브레이션 생성
    manager = create_default_calibration()

    # 테스트: 픽셀 → 월드 변환
    test_pixels = [(320, 250)]  # 이미지 중앙

    for cctv_id in range(4):
        try:
            world_coords = manager.transform(cctv_id, test_pixels)
            print(f"CCTV {cctv_id}: pixel {test_pixels[0]} → world {world_coords[0]}")
        except Exception as e:
            print(f"CCTV {cctv_id}: Error - {e}")
