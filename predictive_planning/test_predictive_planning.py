#!/usr/bin/env python3
"""
Predictive Planning 테스트 스크립트

ROS 없이 독립적으로 모듈을 테스트.
Mock 예측기를 사용하여 기본 동작 확인.
"""

import sys
import os
import numpy as np
import time

# 경로 설정
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.dirname(MODULE_PATH)
if ENV_PATH not in sys.path:
    sys.path.insert(0, ENV_PATH)

from predictive_planning.src.config import PredictivePlanningConfig
from predictive_planning.src.predicted_trajectory import (
    PredictedTrajectory,
    PredictedTrajectoryArray
)
from predictive_planning.src.predictive_cost_calculator import PredictiveCostCalculator
from predictive_planning.src.predictive_global_planner import (
    PredictiveGlobalPlanner,
    SimpleGlobalPlanner
)
from predictive_planning.src.prediction_receiver import MockPredictionReceiver


def test_config():
    """설정 테스트"""
    print("\n" + "="*50)
    print("1. Configuration Test")
    print("="*50)

    config = PredictivePlanningConfig()

    print(f"  X range: {config.x_range}")
    print(f"  Y range: {config.y_range}")
    print(f"  Resolution: {config.resolution}")
    print(f"  Grid size: {config.grid_width} x {config.grid_height}")
    print(f"  Sigma scale: {config.sigma_scale}")
    print(f"  Sigma min: {config.sigma_min}")
    print(f"  Static obstacles: {len(config.static_obstacles)}")

    # 좌표 변환 테스트
    wx, wy = 0.0, 0.0
    gx, gy = config.world_to_grid(wx, wy)
    wx2, wy2 = config.grid_to_world(gx, gy)
    print(f"  Coord test: world({wx},{wy}) -> grid({gx},{gy}) -> world({wx2:.2f},{wy2:.2f})")

    # 장애물 테스트
    test_points = [(0, 0), (-11, 0), (4, 0), (0, 10)]
    for px, py in test_points:
        in_obs = config.is_in_obstacle(px, py)
        print(f"  Point ({px},{py}) in obstacle: {in_obs}")

    print("  [PASS] Configuration test completed")


def test_predicted_trajectory():
    """예측 궤적 데이터 구조 테스트"""
    print("\n" + "="*50)
    print("2. Predicted Trajectory Test")
    print("="*50)

    # 샘플 데이터 생성
    samples = np.zeros((20, 12, 2))
    for k in range(20):
        for t in range(12):
            # 약간의 노이즈와 함께 직선 이동
            samples[k, t, 0] = 0.0 + 0.5 * (t + 1) + np.random.normal(0, 0.1 * (t + 1))
            samples[k, t, 1] = 0.0 + 0.3 * (t + 1) + np.random.normal(0, 0.1 * (t + 1))

    traj = PredictedTrajectory(
        agent_id=0,
        current_x=0.0,
        current_y=0.0,
        current_vx=0.5,
        current_vy=0.3,
        samples=samples
    )

    print(f"  Agent ID: {traj.agent_id}")
    print(f"  Current pos: ({traj.current_x}, {traj.current_y})")
    print(f"  Samples shape: {traj.samples.shape}")

    # 시간별 위치
    for t in [0.4, 2.0, 4.0]:
        pos = traj.get_position_at_time(t)
        vel = traj.get_velocity_at_time(t)
        print(f"  t={t}s: pos_mean=({pos.mean(axis=0)[0]:.2f}, {pos.mean(axis=0)[1]:.2f}), "
              f"vel_mean={vel.mean():.2f}")

    # Flat 변환 테스트
    sx, sy = traj.to_flat_arrays()
    print(f"  Flat arrays: len_x={len(sx)}, len_y={len(sy)}")

    # 복원 테스트
    traj2 = PredictedTrajectory.from_flat_arrays(
        0, 0.0, 0.0, 0.5, 0.3, sx, sy
    )
    print(f"  Restored samples shape: {traj2.samples.shape}")

    print("  [PASS] Predicted trajectory test completed")


def test_cost_calculator():
    """비용 계산기 테스트"""
    print("\n" + "="*50)
    print("3. Cost Calculator Test")
    print("="*50)

    config = PredictivePlanningConfig()
    calculator = PredictiveCostCalculator(config)

    # Mock 예측 생성
    predictions = PredictedTrajectoryArray()

    # 에이전트 1: (5, 5)에서 (-1, -1) 방향으로 이동
    samples1 = np.zeros((20, 12, 2))
    for k in range(20):
        for t in range(12):
            samples1[k, t, 0] = 5.0 - 0.5 * (t + 1) + np.random.normal(0, 0.05 * (t + 1))
            samples1[k, t, 1] = 5.0 - 0.5 * (t + 1) + np.random.normal(0, 0.05 * (t + 1))

    traj1 = PredictedTrajectory(
        agent_id=0, current_x=5.0, current_y=5.0,
        current_vx=-0.5, current_vy=-0.5, samples=samples1
    )
    predictions.add_trajectory(traj1)

    calculator.update_predictions(predictions)
    calculator.update_robot_state(-5.0, -5.0)

    # 다양한 위치에서 비용 계산
    test_points = [
        (0.0, 0.0, 7.0),   # 원점, g_cost=7m (약 8.75초 소요)
        (3.0, 3.0, 4.0),   # 에이전트 경로 근처, g_cost=4m (약 5초)
        (5.0, 5.0, 2.0),   # 에이전트 시작점 근처
        (-8.0, -8.0, 1.0), # 멀리 떨어진 곳
    ]

    print(f"  Robot at (-5, -5), Agent at (5, 5) moving to (-∞, -∞)")
    print(f"  Config: sigma_scale={config.sigma_scale}, sigma_min={config.sigma_min}")

    for px, py, g_cost in test_points:
        cost = calculator.calculate_cost(px, py, g_cost)
        breakdown = calculator.calculate_cost_detailed(px, py, g_cost)
        print(f"  Point ({px:5.1f},{py:5.1f}), g={g_cost:.1f}m, t_arr={g_cost/0.8:.2f}s: "
              f"cost={cost:.4f}, hits={breakdown.hit_counts}")

    print("  [PASS] Cost calculator test completed")


def test_planner():
    """경로 계획기 테스트"""
    print("\n" + "="*50)
    print("4. Planner Test")
    print("="*50)

    config = PredictivePlanningConfig()

    # 예측 기반 플래너
    planner = PredictiveGlobalPlanner(config)

    # 단순 플래너 (비교용)
    simple_planner = SimpleGlobalPlanner(config)

    # Mock 예측 생성
    predictions = PredictedTrajectoryArray()

    # 에이전트: 경로 중간에 있음
    samples = np.zeros((20, 12, 2))
    for k in range(20):
        for t in range(12):
            samples[k, t, 0] = 0.0 + 0.3 * (t + 1) + np.random.normal(0, 0.1)
            samples[k, t, 1] = 6.0 - 0.2 * (t + 1) + np.random.normal(0, 0.1)

    traj = PredictedTrajectory(
        agent_id=0, current_x=0.0, current_y=6.0,
        current_vx=0.3, current_vy=-0.2, samples=samples
    )
    predictions.add_trajectory(traj)

    planner.update_predictions(predictions)

    # 시작/목표
    start = (-9.0, 9.0)
    goal = (9.0, -9.0)

    print(f"  Start: {start}")
    print(f"  Goal: {goal}")
    print(f"  Agent at (0, 6) moving to (+∞, -∞)")

    # 단순 A*
    print("\n  --- Simple A* (no prediction) ---")
    result_simple = simple_planner.plan(start, goal)
    print(f"  Success: {result_simple.success}")
    print(f"  Path length: {result_simple.path_length:.2f}m")
    print(f"  Waypoints: {len(result_simple.path)}")
    print(f"  Time: {result_simple.planning_time*1000:.1f}ms")

    # 예측 기반 A*
    print("\n  --- Predictive A* (with 20 samples) ---")
    result_pred = planner.plan(start, goal)
    print(f"  Success: {result_pred.success}")
    print(f"  Path length: {result_pred.path_length:.2f}m")
    print(f"  Total cost: {result_pred.total_cost:.2f}")
    print(f"  Waypoints: {len(result_pred.path)}")
    print(f"  Time: {result_pred.planning_time*1000:.1f}ms")
    print(f"  Iterations: {result_pred.iterations}")

    # 경로 비교
    if result_simple.success and result_pred.success:
        diff = result_pred.path_length - result_simple.path_length
        print(f"\n  Path length difference: {diff:+.2f}m")
        if diff > 0:
            print(f"  -> Predictive path is {diff:.2f}m longer (avoiding predicted positions)")

    print("  [PASS] Planner test completed")


def test_full_pipeline():
    """전체 파이프라인 테스트"""
    print("\n" + "="*50)
    print("5. Full Pipeline Test")
    print("="*50)

    config = PredictivePlanningConfig()
    receiver = MockPredictionReceiver(config)
    calculator = PredictiveCostCalculator(config)
    planner = PredictiveGlobalPlanner(config, calculator)

    # 시뮬레이션 데이터
    robot_pos = (-9.0, 9.0)
    goal = (9.0, -9.0)

    # 여러 프레임 시뮬레이션
    print(f"  Simulating {10} frames...")

    for frame in range(10):
        timestamp = frame * 0.4

        # 에이전트 업데이트 (이동)
        agents = [
            {'id': 0, 'x': 0.0 + 0.3 * frame, 'y': 6.0 - 0.2 * frame,
             'vx': 0.3, 'vy': -0.2},
            {'id': 1, 'x': -5.0 + 0.2 * frame, 'y': -5.0 + 0.4 * frame,
             'vx': 0.2, 'vy': 0.4},
        ]

        receiver.update_agents(agents, timestamp)

        # 8프레임 이후 예측 가능
        if frame >= 2:
            predictions = receiver.get_predictions()

            if predictions:
                calculator.update_predictions(predictions)
                calculator.update_robot_state(robot_pos[0], robot_pos[1])
                planner.update_predictions(predictions)

                result = planner.plan(robot_pos, goal)

                if frame == 9:  # 마지막 프레임만 상세 출력
                    print(f"\n  Frame {frame}: {len(predictions)} agents predicted")
                    print(f"  Planning result: success={result.success}, "
                          f"path_len={result.path_length:.2f}m, "
                          f"cost={result.total_cost:.2f}")

    print("  [PASS] Full pipeline test completed")


def test_visualization():
    """시각화 테스트 (이미지 저장)"""
    print("\n" + "="*50)
    print("6. Visualization Test")
    print("="*50)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    config = PredictivePlanningConfig()
    calculator = PredictiveCostCalculator(config)

    # Mock 예측 생성
    predictions = PredictedTrajectoryArray()

    for agent_id in range(3):
        samples = np.zeros((20, 12, 2))
        base_x = -6.0 + agent_id * 6.0
        base_y = 6.0

        for k in range(20):
            for t in range(12):
                samples[k, t, 0] = base_x + 0.4 * (t + 1) + np.random.normal(0, 0.1 * (t + 1))
                samples[k, t, 1] = base_y - 0.3 * (t + 1) + np.random.normal(0, 0.1 * (t + 1))

        traj = PredictedTrajectory(
            agent_id=agent_id, current_x=base_x, current_y=base_y,
            current_vx=0.4, current_vy=-0.3, samples=samples
        )
        predictions.add_trajectory(traj)

    calculator.update_predictions(predictions)
    calculator.update_robot_state(-9.0, 9.0)

    # 비용 맵 생성
    cost_grid, metadata = calculator.get_cost_map(resolution=0.3)

    # 시각화
    fig, ax = plt.subplots(figsize=(10, 10))

    extent = [config.x_range[0], config.x_range[1],
              config.y_range[1], config.y_range[0]]
    im = ax.imshow(cost_grid, extent=extent, cmap='hot_r', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Predictive Social Cost')

    # 장애물
    for obs in config.static_obstacles:
        rect = Rectangle(
            (obs['x_min'], obs['y_min']),
            obs['x_max'] - obs['x_min'],
            obs['y_max'] - obs['y_min'],
            facecolor='gray', edgecolor='white', alpha=0.8
        )
        ax.add_patch(rect)

    # 예측 궤적
    colors = ['cyan', 'lime', 'yellow']
    for i, traj in enumerate(predictions):
        ax.scatter(traj.current_x, traj.current_y, c=colors[i], s=100, zorder=10)
        mean = traj.get_mean_trajectory()
        ax.plot(mean[:, 0], mean[:, 1], c=colors[i], linewidth=2, alpha=0.8)

    # 로봇
    ax.scatter(-9.0, 9.0, c='blue', s=200, marker='o', zorder=15)

    ax.set_xlim(config.x_range[0] - 1, config.x_range[1] + 1)
    ax.set_ylim(config.y_range[1] + 1, config.y_range[0] - 1)
    ax.set_aspect('equal')
    ax.set_title('Predictive Cost Map Test\n(sigma_scale=0.8, 20 samples)', fontsize=12)

    # 저장
    output_path = os.path.join(MODULE_PATH, 'test_visualization.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    print("  [PASS] Visualization test completed")


def main():
    """메인 테스트 함수"""
    print("\n" + "#"*60)
    print("  PREDICTIVE PLANNING MODULE TEST")
    print("#"*60)

    test_config()
    test_predicted_trajectory()
    test_cost_calculator()
    test_planner()
    test_full_pipeline()
    test_visualization()

    print("\n" + "="*60)
    print("  ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
