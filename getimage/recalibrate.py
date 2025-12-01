#!/usr/bin/env python3
"""
캘리브레이션 포인트 JSON에서 Homography 매트릭스 재계산.

Usage:
    python3 recalibrate.py
"""

import os
import json
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(SCRIPT_DIR, 'calibration')


def recalibrate_from_json():
    """JSON 포인트 파일에서 Homography 매트릭스 재계산"""

    for cctv_id in range(4):
        json_path = os.path.join(CALIBRATION_DIR, f'cctv_{cctv_id}_points.json')
        npy_path = os.path.join(CALIBRATION_DIR, f'cctv_{cctv_id}.npy')

        if not os.path.exists(json_path):
            print(f"CCTV {cctv_id}: No points file found at {json_path}")
            continue

        # JSON 로드
        with open(json_path, 'r') as f:
            data = json.load(f)

        pixel_points = data['pixel_points']
        world_points = data['world_points']

        print(f"\nCCTV {cctv_id}:")
        print(f"  Pixel points: {len(pixel_points)}")
        print(f"  World points: {len(world_points)}")

        if len(pixel_points) < 4:
            print(f"  ERROR: Need at least 4 points, got {len(pixel_points)}")
            continue

        # Homography 계산
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)

        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if H is None:
            print(f"  ERROR: Failed to compute homography")
            continue

        # 저장
        np.save(npy_path, H)
        print(f"  Saved homography to {npy_path}")

        # 검증: 픽셀 → 월드 변환 테스트
        print(f"  Verification:")
        for i, (px, wx) in enumerate(zip(pixel_points, world_points)):
            pt = np.array([[px]], dtype=np.float32)
            transformed = cv2.perspectiveTransform(pt, H)
            tx, ty = transformed[0, 0, 0], transformed[0, 0, 1]
            error = np.sqrt((tx - wx[0])**2 + (ty - wx[1])**2)
            print(f"    Point {i}: pixel {px} → world ({tx:.2f}, {ty:.2f}), expected {wx}, error: {error:.3f}m")


if __name__ == '__main__':
    print("=== Recalibrating Homography from JSON points ===")
    recalibrate_from_json()
    print("\n=== Done ===")
