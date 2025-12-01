#!/usr/bin/env python3
"""
Standalone Trajectory Prediction Visualizer

ROS 없이 Ground Truth 파일을 읽어서
SingularTrajectory 예측 결과를 실시간 애니메이션으로 시각화.

Usage:
    python standalone_visualizer.py [--data_path PATH] [--output OUTPUT.mp4]
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from collections import defaultdict

# 모듈 경로
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.dirname(MODULE_PATH)
if ENV_PATH not in sys.path:
    sys.path.insert(0, ENV_PATH)

from trajectory_prediction.predictor import TrajectoryPredictor
from trajectory_prediction.prediction_config import PredictionConfig


class StandaloneVisualizer:
    """스탠드얼론 시각화 도구"""

    def __init__(self, config: PredictionConfig):
        self.config = config
        self.predictor = TrajectoryPredictor(config)

        # 색상 팔레트
        self.colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
        ]

    def load_data(self, data_path: str) -> dict:
        """
        Ground Truth 데이터 로드

        Format: frame_id \t ped_id \t x \t y
        """
        data = defaultdict(list)  # frame -> [(ped_id, x, y), ...]

        with open(data_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    frame = int(parts[0])
                    ped_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    data[frame].append((ped_id, x, y))

        return dict(sorted(data.items()))

    def run_simulation(self, data: dict) -> list:
        """
        프레임별로 시뮬레이션 실행

        Returns:
            각 프레임의 (agents, predictions) 리스트
        """
        print("[Visualizer] Loading model...")
        self.predictor.load_model()

        frames = sorted(data.keys())
        results = []

        print(f"[Visualizer] Processing {len(frames)} frames...")

        for i, frame in enumerate(frames):
            # 현재 프레임의 에이전트 위치
            agents = data[frame]
            timestamp = frame * self.config.dt

            # 에이전트 업데이트
            for ped_id, x, y in agents:
                self.predictor.update_agent(ped_id, x, y, timestamp)

            # 예측 실행
            predictions = self.predictor.predict()

            results.append({
                'frame': frame,
                'timestamp': timestamp,
                'agents': agents,
                'predictions': predictions
            })

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(frames)} frames")

        print(f"[Visualizer] Done processing")
        return results

    def create_animation(self, results: list, output_path: str = None,
                         use_map: bool = True):
        """애니메이션 생성"""
        print("[Visualizer] Creating animation...")

        # Figure 설정
        fig, ax = plt.subplots(figsize=(14, 12))

        # 맵 이미지 로드 (있으면)
        map_img = None
        if use_map and os.path.exists(self.config.map_image_path):
            try:
                map_img = plt.imread(self.config.map_image_path)
            except:
                pass

        def animate(frame_idx):
            ax.clear()

            if frame_idx >= len(results):
                return []

            result = results[frame_idx]
            frame = result['frame']
            timestamp = result['timestamp']
            agents = result['agents']
            predictions = result['predictions']

            # 배경
            if map_img is not None:
                ax.imshow(map_img)
                ax.set_xlim(self.config.plot_bounds['left'] - 50,
                           self.config.plot_bounds['right'] + 50)
                ax.set_ylim(self.config.plot_bounds['bottom'] + 50,
                           self.config.plot_bounds['top'] - 50)
                world_to_pixel = self.config.world_to_pixel
            else:
                # 장애물 그리기
                for obs in self.config.obstacles:
                    rect = Rectangle(
                        (obs['x_min'], obs['y_min']),
                        obs['x_max'] - obs['x_min'],
                        obs['y_max'] - obs['y_min'],
                        linewidth=1, edgecolor='gray',
                        facecolor='lightgray', alpha=0.5
                    )
                    ax.add_patch(rect)

                ax.set_xlim(self.config.x_range[0] - 1, self.config.x_range[1] + 1)
                ax.set_ylim(self.config.y_range[0] - 1, self.config.y_range[1] + 1)
                ax.set_aspect('equal')
                world_to_pixel = lambda x, y: (x, y)

            ax.axis('off')

            # 에이전트 및 예측 그리기
            for ped_id, x, y in agents:
                color = self.colors[ped_id % len(self.colors)]
                px, py = world_to_pixel(x, y)

                # 현재 위치
                ax.scatter(px, py, c=color, s=300, zorder=10,
                          edgecolors='white', linewidths=2)
                ax.text(px, py - 40, f'P{ped_id}', fontsize=10, ha='center',
                       color=color, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                # 예측이 있으면 그리기
                if ped_id in predictions:
                    pred_data = predictions[ped_id]
                    obs_traj = pred_data['obs_traj']
                    pred_best = pred_data['pred_best']
                    pred_samples = pred_data['pred_samples']

                    # 관측 궤적 (회색)
                    obs_px = np.array([world_to_pixel(p[0], p[1]) for p in obs_traj])
                    ax.plot(obs_px[:, 0], obs_px[:, 1], color='gray',
                           linewidth=2, alpha=0.5)

                    # 예측 궤적 (Best)
                    pred_px = np.array([world_to_pixel(p[0], p[1]) for p in pred_best])
                    ax.plot(pred_px[:, 0], pred_px[:, 1], color=color,
                           linewidth=3, alpha=0.9)

                    # 예측 끝점
                    ax.scatter(pred_px[-1, 0], pred_px[-1, 1], c=color, s=150,
                              marker='*', zorder=9, edgecolors='white', linewidths=1)

                    # 다른 샘플들 (얇게)
                    for sample_idx in range(1, min(3, len(pred_samples))):
                        sample = pred_samples[sample_idx]
                        sample_px = np.array([world_to_pixel(p[0], p[1]) for p in sample])
                        ax.plot(sample_px[:, 0], sample_px[:, 1], color=color,
                               linewidth=1, alpha=0.2)

            # 제목
            valid_preds = len(predictions)
            total_agents = len(agents)
            ax.set_title(
                f'Frame {frame} | t={timestamp:.1f}s | '
                f'Predicting {valid_preds}/{total_agents} agents',
                fontsize=14, fontweight='bold'
            )

            return []

        # 애니메이션 생성
        total_frames = len(results)
        ani = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=int(self.config.dt * 1000),  # ms
            blit=False
        )

        if output_path:
            print(f"[Visualizer] Saving to {output_path}...")
            ani.save(output_path, writer='ffmpeg', fps=int(1/self.config.dt), dpi=100)
            print(f"[Visualizer] Saved!")
        else:
            plt.show()

        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Trajectory Prediction Visualizer')
    parser.add_argument('--data_path', type=str,
                       default='/home/pyongjoo/Desktop/newstart/trajectory_collector/warehouse_test.txt',
                       help='Path to ground truth data file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (if not set, show interactive)')
    parser.add_argument('--scenario', type=str, default='warehouse',
                       help='Scenario name')
    parser.add_argument('--no_map', action='store_true',
                       help='Do not use map image background')

    args = parser.parse_args()

    # 설정 및 시각화기 생성
    config = PredictionConfig(args.scenario)
    visualizer = StandaloneVisualizer(config)

    # 데이터 로드
    print(f"[Visualizer] Loading data from {args.data_path}")
    data = visualizer.load_data(args.data_path)
    print(f"[Visualizer] Loaded {len(data)} frames")

    # 시뮬레이션 실행
    results = visualizer.run_simulation(data)

    # 애니메이션 생성
    visualizer.create_animation(
        results,
        output_path=args.output,
        use_map=not args.no_map
    )


if __name__ == '__main__':
    main()
