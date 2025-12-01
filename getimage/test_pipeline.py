#!/usr/bin/env python3
"""
파이프라인 테스트 스크립트

1. ROS 없이 더미 데이터로 테스트
2. ROS로 실제 시뮬레이션 데이터 테스트
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import argparse


def test_config():
    """warehouse_config 테스트"""
    print("\n" + "="*50)
    print("TEST 1: warehouse_config")
    print("="*50)

    from warehouse_config import (
        SHELVES, AISLES, CCTV_CONFIGS,
        is_in_valid_region, is_in_shelf, get_aisle_for_position
    )

    print(f"\n선반 {len(SHELVES)}개:")
    for name, bounds in SHELVES.items():
        print(f"  {name}: x=[{bounds[0]}, {bounds[1]}], y=[{bounds[2]}, {bounds[3]}]")

    print(f"\n통로 {len(AISLES)}개:")
    for name, bounds in AISLES.items():
        print(f"  {name}: x=[{bounds[0]}, {bounds[1]}]")

    print(f"\nCCTV {len(CCTV_CONFIGS)}개:")
    for cctv_id, cfg in CCTV_CONFIGS.items():
        print(f"  CCTV {cctv_id}: pos={cfg.position}, target={cfg.target_aisle}")

    # 유효 영역 테스트
    test_points = [
        (-8.5, 0, "Aisle 1 중앙"),
        (-3.5, 0, "Aisle 2 중앙"),
        (1.5, 0, "Aisle 3 중앙"),
        (7.5, 0, "Aisle 4 중앙"),
        (-11, 0, "선반 1 내부"),
        (-6, 0, "선반 2 내부"),
        (0, 8, "상단 통로"),
        (0, -8, "하단 통로"),
    ]

    print("\n유효 영역 테스트:")
    for x, y, desc in test_points:
        valid = is_in_valid_region(x, y)
        shelf = is_in_shelf(x, y)
        aisle = get_aisle_for_position(x, y)
        status = "✓" if valid else "✗"
        print(f"  {status} ({x:5.1f}, {y:5.1f}) {desc:15} → valid={valid}, shelf={shelf}, aisle={aisle}")

    print("\n✓ Config 테스트 완료")


def test_extractor():
    """HumanStateExtractor 테스트"""
    print("\n" + "="*50)
    print("TEST 2: HumanStateExtractor")
    print("="*50)

    from human_state_extractor import HumanStateExtractor

    extractor = HumanStateExtractor(history_length=8, dt=0.4)

    print("\n시뮬레이션: 2명이 이동하는 상황")
    print("  사람 0: Aisle 1에서 아래로 이동")
    print("  사람 1: Aisle 3에서 오른쪽으로 이동")

    for frame in range(12):
        t = frame * 0.4

        # 사람 0: (-8.5, 3) → (-8.5, -3) 이동
        # 사람 1: (1, 0) → (3, 0) 이동
        detections = [
            (-8.5, 3 - 0.5 * frame),   # 사람 0
            (1 + 0.17 * frame, 0),      # 사람 1
        ]

        extractor.update(detections, timestamp=t)

        states = extractor.get_human_states()
        ready = extractor.get_ready_count()

        print(f"\nFrame {frame:2d} (t={t:.1f}s) - Ready: {ready}/{len(states)}")
        for s in states:
            print(f"    ID {s.id}: pos=({s.px:6.2f}, {s.py:6.2f}), vel=({s.vx:5.2f}, {s.vy:5.2f}), speed={s.speed:.2f}")

    # 텐서 출력
    tensor, ids = extractor.get_trajectory_tensor()
    print(f"\n최종 텐서:")
    print(f"  Shape: {tensor.shape}")
    print(f"  IDs: {ids}")

    print("\n✓ Extractor 테스트 완료")


def test_visualizer():
    """Visualizer 테스트"""
    print("\n" + "="*50)
    print("TEST 3: Visualizer")
    print("="*50)

    from human_state_extractor import HumanState
    from visualizer import WarehouseVisualizer, compute_detection_metrics

    # GT 데이터
    gt_humans = [
        HumanState(id=0, px=-8.5, py=2, vx=0, vy=-0.5),
        HumanState(id=1, px=-3.5, py=-1, vx=0, vy=0.5),
        HumanState(id=2, px=1.5, py=0, vx=0.3, vy=0),
        HumanState(id=3, px=7.5, py=1, vx=0, vy=-0.3),
    ]

    # 검출 결과 (약간 오차 + 1개 FP)
    det_positions = [
        (-8.3, 2.1),    # GT 0 근처 (오차 0.22m)
        (-3.6, -0.8),   # GT 1 근처 (오차 0.22m)
        (1.7, 0.1),     # GT 2 근처 (오차 0.22m)
        (7.4, 1.2),     # GT 3 근처 (오차 0.22m)
        (-6, 0),        # FP - 선반 안 (invalid)
    ]

    # 시각화
    viz = WarehouseVisualizer()
    viz.create_figure()
    viz.draw_ground_truth(gt_humans, color='blue', label='Ground Truth')
    viz.draw_detections(det_positions, color='red', label='Detection')
    viz.add_legend()

    # 저장
    save_path = os.path.join(SCRIPT_DIR, 'test_output.png')
    viz.save(save_path)
    viz.close()

    # 지표 계산
    metrics = compute_detection_metrics(gt_humans, det_positions)
    print("\n검출 지표:")
    print(f"  True Positives:  {metrics['true_positives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  Precision:       {metrics['precision']:.3f}")
    print(f"  Recall:          {metrics['recall']:.3f}")
    print(f"  F1 Score:        {metrics['f1_score']:.3f}")
    print(f"  Mean Error:      {metrics['mean_position_error']:.3f}m")
    print(f"  Invalid Dets:    {metrics['invalid_detections']}")

    print(f"\n✓ 시각화 저장됨: {save_path}")


def test_homography():
    """Homography 테스트 (기본 캘리브레이션)"""
    print("\n" + "="*50)
    print("TEST 4: Homography (추정값)")
    print("="*50)

    from homography import create_default_calibration

    print("\n기본 캘리브레이션 생성 중...")
    manager = create_default_calibration()

    # 테스트: 이미지 중앙 → 월드 좌표
    test_pixel = [(320, 250)]  # 이미지 중앙

    print("\n픽셀 (320, 250) → 월드 좌표 변환:")
    for cctv_id in range(4):
        try:
            world = manager.transform(cctv_id, test_pixel)
            print(f"  CCTV {cctv_id}: {test_pixel[0]} → ({world[0][0]:.2f}, {world[0][1]:.2f})")
        except Exception as e:
            print(f"  CCTV {cctv_id}: Error - {e}")

    print("\n⚠️  이 값은 추정값입니다. 실제 캘리브레이션 필요!")
    print("✓ Homography 테스트 완료")


def test_ros_connection():
    """ROS 연결 테스트"""
    print("\n" + "="*50)
    print("TEST 5: ROS 연결")
    print("="*50)

    try:
        import rospy
        from sensor_msgs.msg import Image
        from pedsim_msgs.msg import AgentStates

        rospy.init_node('test_node', anonymous=True)

        print("\n토픽 확인 중... (5초 대기)")

        # CCTV 토픽 확인
        cctv_topics = [f'/cctv_{i}/image_raw' for i in range(4)]
        for topic in cctv_topics:
            try:
                msg = rospy.wait_for_message(topic, Image, timeout=2.0)
                print(f"  ✓ {topic}: {msg.width}x{msg.height}")
            except:
                print(f"  ✗ {topic}: 없음")

        # GT 토픽 확인
        try:
            msg = rospy.wait_for_message('/pedsim_simulator/simulated_agents', AgentStates, timeout=2.0)
            print(f"  ✓ /pedsim_simulator/simulated_agents: {len(msg.agent_states)}명")
        except:
            print(f"  ✗ /pedsim_simulator/simulated_agents: 없음")

        print("\n✓ ROS 연결 테스트 완료")

    except ImportError:
        print("\n✗ ROS를 사용할 수 없습니다")
        print("  시뮬레이션을 먼저 실행하세요:")
        print("  cd /home/pyongjoo/Desktop/newstart/environment/with_robot")
        print("  ./run_with_robot_scenario.sh")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "#"*50)
    print("# CCTV Detection Pipeline Test")
    print("#"*50)

    test_config()
    test_extractor()
    test_visualizer()
    test_homography()

    print("\n" + "="*50)
    print("모든 오프라인 테스트 완료!")
    print("="*50)
    print("\nROS 테스트를 하려면:")
    print("  1. 시뮬레이션 실행")
    print("  2. python3 test_pipeline.py --ros")


def main():
    parser = argparse.ArgumentParser(description='CCTV Detection Pipeline Test')
    parser.add_argument('--ros', action='store_true', help='ROS 연결 테스트')
    parser.add_argument('--config', action='store_true', help='Config만 테스트')
    parser.add_argument('--extractor', action='store_true', help='Extractor만 테스트')
    parser.add_argument('--visualizer', action='store_true', help='Visualizer만 테스트')
    parser.add_argument('--homography', action='store_true', help='Homography만 테스트')

    args = parser.parse_args()

    if args.ros:
        test_ros_connection()
    elif args.config:
        test_config()
    elif args.extractor:
        test_extractor()
    elif args.visualizer:
        test_visualizer()
    elif args.homography:
        test_homography()
    else:
        run_all_tests()


if __name__ == '__main__':
    main()
