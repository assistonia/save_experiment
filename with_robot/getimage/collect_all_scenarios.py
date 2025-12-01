#!/usr/bin/env python3
"""
5개 시나리오에서 캘리브레이션 데이터 수집 및 최종 Homography 계산
"""

import sys
import os
import subprocess
import time
import json
import numpy as np
import cv2
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 5개 시나리오
SCENARIOS = [
    "scenario_work_in_aisle1.xml",
    "scenario_work_in_aisle2.xml",
    "scenario_work_in_aisle3.xml",
    "scenario_work_in_aisle4.xml",
    "scenario_busy_warehouse.xml",
]

CCTV_REGIONS = {
    0: {'x': (-12, -5), 'y': (-6, 8)},
    1: {'x': (-7, 0), 'y': (-8, 6)},
    2: {'x': (-2, 5), 'y': (-6, 8)},
    3: {'x': (3, 12), 'y': (-8, 6)},
}

def run_docker_command(cmd):
    """Docker 명령 실행"""
    full_cmd = f'docker exec gdae_pedsim_robot bash -c "source /opt/ros/noetic/setup.bash && source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash && {cmd}"'
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

def start_scenario(scenario_name):
    """시나리오 시작"""
    print(f"  Starting: {scenario_name}")

    # 기존 roslaunch 종료
    run_docker_command("pkill -f roslaunch || true")
    time.sleep(3)

    # 새 시나리오 시작
    cmd = f"roslaunch /environment/with_robot.launch scenario:={scenario_name} &"
    subprocess.Popen(
        f'docker exec -d gdae_pedsim_robot bash -c "source /opt/ros/noetic/setup.bash && source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash && {cmd}"',
        shell=True
    )

    print(f"  Waiting for simulation to start...")
    time.sleep(20)

def collect_calibration_data(duration=40):
    """현재 시나리오에서 캘리브레이션 데이터 수집"""

    import rospy
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    from pedsim_msgs.msg import AgentStates
    from warehouse_config import is_in_valid_region
    from detector import PersonDetector

    # ROS 초기화
    if not rospy.core.is_initialized():
        rospy.init_node('calib_collector', anonymous=True)

    bridge = CvBridge()
    detector = PersonDetector(model_path='yolov8n.pt', confidence_threshold=0.4)

    latest_images = {}
    ground_truth = []
    calib_data = defaultdict(list)

    def image_cb(msg, cctv_id):
        try:
            latest_images[cctv_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def gt_cb(msg):
        nonlocal ground_truth
        ground_truth = [{'id': a.id, 'x': a.pose.position.x, 'y': a.pose.position.y}
                       for a in msg.agent_states]

    # Subscribers
    subs = []
    for cctv_id in range(4):
        subs.append(rospy.Subscriber(f'/cctv_{cctv_id}/image_raw', Image, image_cb, callback_args=cctv_id, queue_size=1))
    subs.append(rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, gt_cb, queue_size=1))

    # 수집
    rate = rospy.Rate(2)
    start = time.time()

    while not rospy.is_shutdown() and (time.time() - start) < duration:
        if len(latest_images) < 4:
            rate.sleep()
            continue

        for cctv_id, image in latest_images.items():
            region = CCTV_REGIONS.get(cctv_id)
            if not region:
                continue

            visible_gt = [g for g in ground_truth
                         if region['x'][0] <= g['x'] <= region['x'][1]
                         and region['y'][0] <= g['y'] <= region['y'][1]
                         and is_in_valid_region(g['x'], g['y'])]

            if not visible_gt:
                continue

            foot_pixels = detector.detect_feet_only(image)
            if not foot_pixels:
                continue

            # 매칭
            if len(foot_pixels) == len(visible_gt) and len(foot_pixels) <= 3:
                det_sorted = sorted(foot_pixels, key=lambda p: p[0])
                gt_sorted = sorted(visible_gt, key=lambda g: g['x'])
                for det, gt in zip(det_sorted, gt_sorted):
                    calib_data[cctv_id].append((det, (gt['x'], gt['y'])))

        rate.sleep()

    # Unsubscribe
    for sub in subs:
        sub.unregister()

    return dict(calib_data)


def compute_final_homography(all_data):
    """최종 Homography 계산"""

    calibration_dir = os.path.join(SCRIPT_DIR, 'calibration')
    os.makedirs(calibration_dir, exist_ok=True)

    results = {}

    for cctv_id in range(4):
        pairs = all_data.get(cctv_id, [])
        print(f"\nCCTV {cctv_id}: {len(pairs)} pairs")

        if len(pairs) < 4:
            print(f"  Not enough pairs")
            continue

        unique_pairs = list(set(pairs))
        print(f"  Unique: {len(unique_pairs)}")

        pixels = np.array([p[0] for p in unique_pairs], dtype=np.float32)
        worlds = np.array([p[1] for p in unique_pairs], dtype=np.float32)

        H, _ = cv2.findHomography(pixels, worlds, cv2.RANSAC, 3.0)

        if H is None:
            print(f"  Failed")
            continue

        # 저장
        np.save(os.path.join(calibration_dir, f'cctv_{cctv_id}.npy'), H)

        with open(os.path.join(calibration_dir, f'cctv_{cctv_id}_points.json'), 'w') as f:
            json.dump({
                'pixel_points': [[float(x) for x in p[0]] for p in unique_pairs],
                'world_points': [[float(x) for x in p[1]] for p in unique_pairs],
            }, f, indent=2)

        # 오차 계산
        errors = []
        for pixel, world_gt in unique_pairs[:20]:
            pt = np.array([pixel[0], pixel[1], 1], dtype=np.float32)
            pred = H @ pt
            pred = pred[:2] / pred[2]
            errors.append(np.sqrt((pred[0]-world_gt[0])**2 + (pred[1]-world_gt[1])**2))

        mean_error = np.mean(errors)
        print(f"  Saved! Error: {mean_error:.3f}m")
        results[cctv_id] = {'pairs': len(unique_pairs), 'error': mean_error}

    return results


def main():
    print("="*60)
    print("5-Scenario Calibration Collection")
    print("="*60)

    all_calib_data = defaultdict(list)

    for i, scenario in enumerate(SCENARIOS):
        print(f"\n>>> [{i+1}/5] {scenario}")
        print("-"*40)

        # 시나리오 시작
        start_scenario(scenario)

        # 데이터 수집
        print(f"  Collecting data for 40s...")
        data = collect_calibration_data(duration=40)

        for cctv_id, pairs in data.items():
            all_calib_data[cctv_id].extend(pairs)
            print(f"    CCTV {cctv_id}: +{len(pairs)} pairs (total: {len(all_calib_data[cctv_id])})")

    # 최종 Homography 계산
    print("\n" + "="*60)
    print("Computing Final Homography")
    print("="*60)

    results = compute_final_homography(dict(all_calib_data))

    print("\n" + "="*60)
    print("Final Results")
    print("="*60)
    for cctv_id, res in results.items():
        print(f"CCTV {cctv_id}: {res['pairs']} pairs, error={res['error']:.3f}m")


if __name__ == '__main__':
    main()
