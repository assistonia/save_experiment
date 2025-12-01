#!/usr/bin/env python3
"""
수동 캘리브레이션 데이터

각 CCTV 이미지에서 선반 모서리 바닥 접점의 픽셀 좌표를 수동으로 찾아서 기록.
월드 좌표는 warehouse_config.py의 선반 정의 기반.

선반 좌표:
- shelf_1: x=[-12, -10], y=[-5, 4]
- shelf_2: x=[-7, -5], y=[-5, 4]
- shelf_3: x=[-2, 0], y=[-5, 4]
- shelf_4: x=[3, 5], y=[-5, 4]
- shelf_5: x=[10, 12], y=[-5, 4]

CCTV 위치/방향:
- CCTV 0: (-8.5, 10, 3), yaw=-90° → Aisle 1 (shelf_1과 shelf_2 사이)
- CCTV 1: (-3.5, -10, 3), yaw=+90° → Aisle 2 (shelf_2와 shelf_3 사이)
- CCTV 2: (1.5, 10, 3), yaw=-90° → Aisle 3 (shelf_3과 shelf_4 사이)
- CCTV 3: (7.5, -10, 3), yaw=+90° → Aisle 4 (shelf_4와 shelf_5 사이)
"""

import numpy as np
import cv2
import os

# ==================== CCTV 0 캘리브레이션 ====================
# 카메라 위치: (-8.5, 10, 3), 아래쪽(남쪽)을 봄
# 보이는 선반: shelf_1 (왼쪽), shelf_2 (오른쪽)
# 이미지: 640x480

CCTV_0_CALIBRATION = {
    'pixel_points': [
        # shelf_1 오른쪽 면 (x=-10)
        (178, 390),   # 앞쪽 모서리 바닥 (x=-10, y=-5)
        (72, 135),    # 뒤쪽 모서리 바닥 (x=-10, y=4)

        # shelf_2 왼쪽 면 (x=-7)
        (462, 390),   # 앞쪽 모서리 바닥 (x=-7, y=-5)
        (565, 135),   # 뒤쪽 모서리 바닥 (x=-7, y=4)
    ],
    'world_points': [
        (-10, -5),
        (-10, 4),
        (-7, -5),
        (-7, 4),
    ]
}

# ==================== CCTV 1 캘리브레이션 ====================
# 카메라 위치: (-3.5, -10, 3), 위쪽(북쪽)을 봄
# 보이는 선반: shelf_2 (왼쪽), shelf_3 (오른쪽)
# 이미지: 640x480

CCTV_1_CALIBRATION = {
    'pixel_points': [
        # shelf_2 오른쪽 면 (x=-5)
        (170, 380),   # 앞쪽 모서리 바닥 (x=-5, y=-5, 카메라에서 가까움)
        (65, 125),    # 뒤쪽 모서리 바닥 (x=-5, y=4, 카메라에서 멀음)

        # shelf_3 왼쪽 면 (x=-2)
        (468, 380),   # 앞쪽 모서리 바닥 (x=-2, y=-5)
        (575, 125),   # 뒤쪽 모서리 바닥 (x=-2, y=4)
    ],
    'world_points': [
        (-5, -5),
        (-5, 4),
        (-2, -5),
        (-2, 4),
    ]
}

# ==================== CCTV 2 캘리브레이션 ====================
# 카메라 위치: (1.5, 10, 3), 아래쪽(남쪽)을 봄
# 보이는 선반: shelf_3 (왼쪽), shelf_4 (오른쪽)
# 이미지: 640x480

CCTV_2_CALIBRATION = {
    'pixel_points': [
        # shelf_3 오른쪽 면 (x=0)
        (175, 385),   # 앞쪽 모서리 바닥 (x=0, y=-5)
        (68, 130),    # 뒤쪽 모서리 바닥 (x=0, y=4)

        # shelf_4 왼쪽 면 (x=3)
        (465, 385),   # 앞쪽 모서리 바닥 (x=3, y=-5)
        (572, 130),   # 뒤쪽 모서리 바닥 (x=3, y=4)
    ],
    'world_points': [
        (0, -5),
        (0, 4),
        (3, -5),
        (3, 4),
    ]
}

# ==================== CCTV 3 캘리브레이션 ====================
# 카메라 위치: (7.5, -10, 3), 위쪽(북쪽)을 봄
# 보이는 선반: shelf_4 (왼쪽), shelf_5 (오른쪽)
# 이미지: 640x480

CCTV_3_CALIBRATION = {
    'pixel_points': [
        # shelf_4 오른쪽 면 (x=5)
        (168, 375),   # 앞쪽 모서리 바닥 (x=5, y=-5)
        (60, 120),    # 뒤쪽 모서리 바닥 (x=5, y=4)

        # shelf_5 왼쪽 면 (x=10)
        (472, 375),   # 앞쪽 모서리 바닥 (x=10, y=-5)
        (580, 120),   # 뒤쪽 모서리 바닥 (x=10, y=4)
    ],
    'world_points': [
        (5, -5),
        (5, 4),
        (10, -5),
        (10, 4),
    ]
}

# 모든 캘리브레이션 데이터
ALL_CALIBRATIONS = {
    0: CCTV_0_CALIBRATION,
    1: CCTV_1_CALIBRATION,
    2: CCTV_2_CALIBRATION,
    3: CCTV_3_CALIBRATION,
}


def compute_homography(pixel_points, world_points):
    """픽셀 좌표 → 월드 좌표 Homography 계산"""
    src = np.array(pixel_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)

    H, status = cv2.findHomography(src, dst)
    return H


def save_calibration(output_dir):
    """모든 CCTV 캘리브레이션 저장"""
    os.makedirs(output_dir, exist_ok=True)

    for cctv_id, data in ALL_CALIBRATIONS.items():
        H = compute_homography(data['pixel_points'], data['world_points'])

        # Homography 행렬 저장
        npy_path = os.path.join(output_dir, f'cctv_{cctv_id}.npy')
        np.save(npy_path, H)
        print(f"Saved: {npy_path}")

        # 캘리브레이션 포인트도 저장 (검증용)
        import json
        json_path = os.path.join(output_dir, f'cctv_{cctv_id}_points.json')
        with open(json_path, 'w') as f:
            json.dump({
                'pixel_points': data['pixel_points'],
                'world_points': data['world_points'],
            }, f, indent=2)
        print(f"Saved: {json_path}")


def test_calibration(cctv_id, test_pixels):
    """캘리브레이션 테스트"""
    data = ALL_CALIBRATIONS[cctv_id]
    H = compute_homography(data['pixel_points'], data['world_points'])

    print(f"\n=== CCTV {cctv_id} Calibration Test ===")
    print(f"Homography matrix:\n{H}")

    for px, py in test_pixels:
        # 픽셀 → 월드
        pixel = np.array([px, py, 1], dtype=np.float32)
        world_homogeneous = H @ pixel
        world = world_homogeneous[:2] / world_homogeneous[2]
        print(f"  Pixel ({px}, {py}) -> World ({world[0]:.2f}, {world[1]:.2f})")


if __name__ == '__main__':
    import sys

    script_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_dir = os.path.join(script_dir, 'calibration')

    print("=== Homography Calibration ===")
    print(f"Output directory: {calibration_dir}")

    # 캘리브레이션 저장
    save_calibration(calibration_dir)

    # 테스트
    print("\n=== Testing Calibrations ===")

    # CCTV 0 테스트 (이미지 중앙 근처)
    test_calibration(0, [(320, 240), (178, 390), (462, 390)])

    # CCTV 3 테스트
    test_calibration(3, [(320, 240), (168, 375), (472, 375)])

    print("\nCalibration complete!")
