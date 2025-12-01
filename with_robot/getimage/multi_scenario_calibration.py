#!/usr/bin/env python3
"""
다중 시나리오 캘리브레이션

각 Aisle별 시나리오를 실행해서 해당 CCTV의 캘리브레이션 데이터를 수집.
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
import subprocess
from typing import List, Tuple, Dict
from collections import defaultdict

# 시나리오별 타겟 CCTV
SCENARIO_CCTV_MAP = {
    'scenario_work_in_aisle1.xml': 0,
    'scenario_work_in_aisle2.xml': 1,
    'scenario_work_in_aisle3.xml': 2,
    'scenario_work_in_aisle4.xml': 3,
    'scenario_busy_warehouse.xml': None,  # 모든 CCTV
    'warehouse_pedsim.xml': None,
}

# 각 CCTV가 보는 영역 (더 정확하게)
CCTV_REGIONS = {
    0: {'x': (-12, -5), 'y': (-6, 8)},   # Aisle 1 (S1-S2 사이)
    1: {'x': (-7, 0), 'y': (-8, 6)},     # Aisle 2 (S2-S3 사이)
    2: {'x': (-2, 5), 'y': (-6, 8)},     # Aisle 3 (S3-S4 사이)
    3: {'x': (3, 12), 'y': (-8, 6)},     # Aisle 4 (S4-S5 사이)
}


def collect_from_scenario(scenario_name, duration=30):
    """특정 시나리오에서 캘리브레이션 데이터 수집"""

    # ROS import (함수 내에서)
    try:
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        from pedsim_msgs.msg import AgentStates
    except ImportError:
        print("ROS not available")
        return {}

    from warehouse_config import is_in_valid_region
    from detector import PersonDetector

    print(f"\n{'='*50}")
    print(f"Collecting from: {scenario_name}")
    print(f"Duration: {duration}s")
    print(f"{'='*50}")

    # 초기화
    if not rospy.core.is_initialized():
        rospy.init_node('multi_scenario_calib', anonymous=True)

    bridge = CvBridge()
    detector = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.4)

    # 데이터 저장
    latest_images = {}
    ground_truth = []
    calib_data = defaultdict(list)

    def image_callback(msg, cctv_id):
        try:
            latest_images[cctv_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def gt_callback(msg):
        nonlocal ground_truth
        ground_truth = []
        for agent in msg.agent_states:
            ground_truth.append({
                'id': agent.id,
                'x': agent.pose.position.x,
                'y': agent.pose.position.y,
            })

    # Subscriber 설정
    subs = []
    for cctv_id in range(4):
        sub = rospy.Subscriber(
            f'/cctv_{cctv_id}/image_raw', Image,
            image_callback, callback_args=cctv_id, queue_size=1
        )
        subs.append(sub)

    gt_sub = rospy.Subscriber(
        '/pedsim_simulator/simulated_agents',
        AgentStates, gt_callback, queue_size=1
    )
    subs.append(gt_sub)

    # 데이터 수집
    rate = rospy.Rate(2)
    start_time = time.time()
    sample_count = 0

    while not rospy.is_shutdown() and (time.time() - start_time) < duration:
        if len(latest_images) < 4:
            rate.sleep()
            continue

        for cctv_id, image in latest_images.items():
            # CCTV 시야 내 GT 필터링
            region = CCTV_REGIONS.get(cctv_id)
            if region is None:
                continue

            visible_gt = [
                g for g in ground_truth
                if (region['x'][0] <= g['x'] <= region['x'][1] and
                    region['y'][0] <= g['y'] <= region['y'][1] and
                    is_in_valid_region(g['x'], g['y']))
            ]

            if not visible_gt:
                continue

            # YOLO 검출
            foot_pixels = detector.detect_feet_only(image)

            if not foot_pixels:
                continue

            # 간단한 매칭: 수가 같고 적을 때만
            if len(foot_pixels) == len(visible_gt) and len(foot_pixels) <= 3:
                det_sorted = sorted(foot_pixels, key=lambda p: p[0])
                gt_sorted = sorted(visible_gt, key=lambda g: g['x'])

                for det, gt in zip(det_sorted, gt_sorted):
                    calib_data[cctv_id].append((det, (gt['x'], gt['y'])))

        sample_count += 1
        if sample_count % 20 == 0:
            print(f"  {int(time.time() - start_time)}s: ", end="")
            for cctv_id in range(4):
                print(f"C{cctv_id}={len(calib_data[cctv_id])} ", end="")
            print()

        rate.sleep()

    # Subscriber 해제
    for sub in subs:
        sub.unregister()

    print(f"\nCollected:")
    for cctv_id in range(4):
        print(f"  CCTV {cctv_id}: {len(calib_data[cctv_id])} pairs")

    return dict(calib_data)


def compute_and_save_homography(all_calib_data):
    """모든 데이터로 Homography 계산"""

    print(f"\n{'='*50}")
    print("Computing Homography")
    print(f"{'='*50}")

    calibration_dir = os.path.join(SCRIPT_DIR, 'calibration')
    os.makedirs(calibration_dir, exist_ok=True)

    results = {}

    for cctv_id in range(4):
        pairs = all_calib_data.get(cctv_id, [])
        print(f"\nCCTV {cctv_id}: {len(pairs)} total pairs")

        if len(pairs) < 4:
            print(f"  Not enough pairs")
            continue

        # 중복 제거
        unique_pairs = list(set(pairs))
        print(f"  Unique: {len(unique_pairs)}")

        # Homography 계산
        pixels = np.array([p[0] for p in unique_pairs], dtype=np.float32)
        worlds = np.array([p[1] for p in unique_pairs], dtype=np.float32)

        H, status = cv2.findHomography(pixels, worlds, cv2.RANSAC, 3.0)

        if H is None:
            print(f"  Failed")
            continue

        # 저장
        npy_path = os.path.join(calibration_dir, f'cctv_{cctv_id}.npy')
        np.save(npy_path, H)

        json_path = os.path.join(calibration_dir, f'cctv_{cctv_id}_points.json')
        with open(json_path, 'w') as f:
            json.dump({
                'pixel_points': [[float(x) for x in p[0]] for p in unique_pairs],
                'world_points': [[float(x) for x in p[1]] for p in unique_pairs],
            }, f, indent=2)

        # 테스트
        errors = []
        for pixel, world_gt in unique_pairs[:20]:
            pt = np.array([pixel[0], pixel[1], 1], dtype=np.float32)
            world_pred = H @ pt
            world_pred = world_pred[:2] / world_pred[2]
            error = np.sqrt((world_pred[0] - world_gt[0])**2 +
                           (world_pred[1] - world_gt[1])**2)
            errors.append(error)

        mean_error = np.mean(errors) if errors else 999
        print(f"  Saved! Error: {mean_error:.3f}m")

        results[cctv_id] = {
            'pairs': len(unique_pairs),
            'error': mean_error
        }

    return results


def main():
    """메인 함수"""

    print("="*60)
    print("Multi-Scenario Calibration")
    print("="*60)

    # 현재 실행 중인 시나리오에서 데이터 수집
    # (시나리오 전환은 수동으로 해야 함)

    all_calib_data = defaultdict(list)

    # 여러 번 수집
    for i in range(3):
        print(f"\n>>> Collection round {i+1}/3")
        data = collect_from_scenario("current", duration=40)

        for cctv_id, pairs in data.items():
            all_calib_data[cctv_id].extend(pairs)

    # Homography 계산
    results = compute_and_save_homography(dict(all_calib_data))

    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    for cctv_id, res in results.items():
        print(f"CCTV {cctv_id}: {res['pairs']} pairs, error={res['error']:.3f}m")


if __name__ == '__main__':
    main()
