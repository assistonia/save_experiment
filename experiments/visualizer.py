#!/usr/bin/env python3
"""
Trajectory Visualizer

실험 결과 궤적을 시각화하고 이미지로 저장.
논문 Figure 6, 7 스타일의 시각화.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class WarehouseMap:
    """Warehouse 맵 설정"""
    x_range: Tuple[float, float] = (-12.0, 12.0)
    y_range: Tuple[float, float] = (-12.0, 12.0)

    # 선반 위치 (warehouse 기준)
    shelves: List[Tuple[float, float, float, float]] = None  # (x_min, y_min, x_max, y_max)

    def __post_init__(self):
        if self.shelves is None:
            # warehouse_pedsim.xml 기준 선반
            self.shelves = [
                (-12, -5, -10, 4),   # Shelf 1 (좌측)
                (-7, -5, -5, 4),     # Shelf 2
                (-2, -5, 0, 4),      # Shelf 3
                (3, -5, 5, 4),       # Shelf 4
                (10, -5, 12, 4),     # Shelf 5 (우측)
            ]


class TrajectoryVisualizer:
    """궤적 시각화"""

    def __init__(self, map_config: WarehouseMap = None):
        self.map_config = map_config or WarehouseMap()

        # 색상 설정
        self.colors = {
            'robot': '#1f77b4',      # 파란색
            'human': '#ff7f0e',      # 주황색
            'goal': '#2ca02c',       # 초록색
            'start': '#d62728',      # 빨간색
            'path': '#9467bd',       # 보라색
            'collision': '#e41a1c',  # 진한 빨간색
            'shelf': '#7f7f7f',      # 회색
        }

        # 방법별 색상
        self.method_colors = {
            'DWA': '#1f77b4',
            'DRL_VO': '#ff7f0e',
            'TEB': '#2ca02c',
            'SFM': '#9467bd',
            'CIGP-DWA': '#17becf',
            'CIGP-DRL_VO': '#bcbd22',
            'CIGP-TEB': '#e377c2',
            'CIGP-SFM': '#8c564b',
            'PRED-DWA': '#d62728',
            'PRED-DRL_VO': '#ff9896',
            'PRED-TEB': '#98df8a',
            'PRED-SFM': '#c5b0d5',
        }

    def _draw_warehouse(self, ax):
        """Warehouse 맵 그리기"""
        # 외곽선
        ax.set_xlim(self.map_config.x_range)
        ax.set_ylim(self.map_config.y_range)

        # 선반 그리기
        for shelf in self.map_config.shelves:
            x_min, y_min, x_max, y_max = shelf
            rect = patches.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor=self.colors['shelf'],
                facecolor=self.colors['shelf'],
                alpha=0.5
            )
            ax.add_patch(rect)

        # 그리드
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_aspect('equal')

    def _draw_start_goal(self, ax, start: Tuple[float, float], goal: Tuple[float, float]):
        """시작/목표 표시"""
        # 시작점 (S)
        ax.plot(start[0], start[1], 'o', color=self.colors['start'],
                markersize=12, markeredgecolor='black', markeredgewidth=2, zorder=10)
        ax.annotate('S', (start[0], start[1]), fontsize=10, ha='center', va='center',
                    color='white', fontweight='bold', zorder=11)

        # 목표점 (★)
        ax.plot(goal[0], goal[1], '*', color=self.colors['goal'],
                markersize=20, markeredgecolor='black', markeredgewidth=1, zorder=10)

    def _draw_trajectory(self, ax, trajectory: List[Dict], color: str = None,
                         label: str = None, show_timesteps: bool = True,
                         timestep_interval: int = 30):
        """궤적 그리기"""
        if not trajectory:
            return

        color = color or self.colors['robot']

        # 위치 추출
        positions = [(t['robot_pos'][0], t['robot_pos'][1]) for t in trajectory
                     if 'robot_pos' in t]

        if not positions:
            return

        x = [p[0] for p in positions]
        y = [p[1] for p in positions]

        # 궤적 선
        ax.plot(x, y, '-', color=color, linewidth=2, alpha=0.8, label=label, zorder=5)

        # 타임스텝 마커
        if show_timesteps:
            for i in range(0, len(positions), timestep_interval):
                ax.plot(x[i], y[i], 'o', color=color, markersize=6,
                        markeredgecolor='white', markeredgewidth=1, zorder=6)

    def _draw_humans(self, ax, humans: List[Dict], show_trajectory: bool = True):
        """보행자 그리기"""
        for human in humans:
            if 'trajectory' in human and show_trajectory:
                # 보행자 궤적
                traj = human['trajectory']
                x = [p[0] for p in traj]
                y = [p[1] for p in traj]
                ax.plot(x, y, '--', color=self.colors['human'], linewidth=1, alpha=0.5)

            if 'pos' in human:
                # 현재 위치
                ax.plot(human['pos'][0], human['pos'][1], 'o',
                        color=self.colors['human'], markersize=8, alpha=0.7)

    def plot_single_episode(self,
                            trajectory: List[Dict],
                            start: Tuple[float, float],
                            goal: Tuple[float, float],
                            method_name: str,
                            episode_id: int,
                            success: bool,
                            collision: bool,
                            humans: List[Dict] = None,
                            save_path: str = None,
                            show: bool = False) -> str:
        """
        단일 에피소드 시각화

        Returns:
            저장된 이미지 경로
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # 맵 그리기
        self._draw_warehouse(ax)

        # 시작/목표
        self._draw_start_goal(ax, start, goal)

        # 보행자
        if humans:
            self._draw_humans(ax, humans)

        # 로봇 궤적
        color = self.method_colors.get(method_name, self.colors['robot'])
        self._draw_trajectory(ax, trajectory, color=color, label=method_name)

        # 충돌 표시
        if collision:
            # 마지막 위치에 X 표시
            if trajectory:
                last_pos = trajectory[-1].get('robot_pos', start)
                ax.plot(last_pos[0], last_pos[1], 'X', color=self.colors['collision'],
                        markersize=20, markeredgewidth=3, zorder=15)

        # 제목
        status = "SUCCESS" if success else ("COLLISION" if collision else "TIMEOUT")
        title = f"{method_name} - Episode {episode_id} [{status}]"
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right')

        plt.tight_layout()

        # 저장
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_comparison(self,
                        trajectories: Dict[str, List[Dict]],
                        start: Tuple[float, float],
                        goal: Tuple[float, float],
                        title: str = "Method Comparison",
                        save_path: str = None,
                        show: bool = False) -> str:
        """
        여러 방법 비교 시각화 (논문 Figure 6 스타일)

        Args:
            trajectories: {method_name: trajectory_data}
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # 맵 그리기
        self._draw_warehouse(ax)

        # 시작/목표
        self._draw_start_goal(ax, start, goal)

        # 각 방법 궤적
        for method_name, traj in trajectories.items():
            color = self.method_colors.get(method_name, '#333333')
            self._draw_trajectory(ax, traj, color=color, label=method_name,
                                  timestep_interval=50)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def plot_scenario_summary(self,
                              results: List[Dict],
                              scenario_name: str,
                              save_path: str = None,
                              show: bool = False) -> str:
        """
        시나리오별 요약 시각화 (성공률, 평균 속도 등)
        """
        # 방법별 그룹화
        methods = {}
        for r in results:
            method = r.get('method_name', 'unknown')
            if method not in methods:
                methods[method] = []
            methods[method].append(r)

        # 메트릭 계산
        method_names = list(methods.keys())
        success_rates = []
        avg_velocities = []
        social_distances = []

        for method in method_names:
            eps = methods[method]
            sr = sum(1 for e in eps if e.get('success')) / len(eps) * 100
            vavg = np.mean([e.get('avg_velocity', 0) for e in eps])
            sd = np.mean([e.get('avg_social_distance', 0) for e in eps])

            success_rates.append(sr)
            avg_velocities.append(vavg)
            social_distances.append(sd)

        # 플롯
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 성공률
        colors = [self.method_colors.get(m, '#333333') for m in method_names]
        axes[0].bar(method_names, success_rates, color=colors)
        axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate (SR)')
        axes[0].set_ylim(0, 100)
        axes[0].tick_params(axis='x', rotation=45)

        # 평균 속도
        axes[1].bar(method_names, avg_velocities, color=colors)
        axes[1].set_ylabel('Velocity (m/s)')
        axes[1].set_title('Average Velocity (Vavg)')
        axes[1].tick_params(axis='x', rotation=45)

        # 사회적 거리
        axes[2].bar(method_names, social_distances, color=colors)
        axes[2].set_ylabel('Distance (m)')
        axes[2].set_title('Social Distance (SD)')
        axes[2].tick_params(axis='x', rotation=45)

        fig.suptitle(f'Scenario: {scenario_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

        return save_path

    def generate_all_images(self, results_dir: str):
        """
        결과 디렉토리에서 모든 이미지 생성

        Args:
            results_dir: 실험 결과 디렉토리
        """
        images_dir = os.path.join(results_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        # 에피소드 결과 로드
        episodes_dir = os.path.join(results_dir, 'episodes')
        trajectories_dir = os.path.join(results_dir, 'trajectories')

        if not os.path.exists(episodes_dir):
            print(f"Episodes directory not found: {episodes_dir}")
            return

        # 각 에피소드 이미지 생성
        episode_files = sorted([f for f in os.listdir(episodes_dir) if f.endswith('.json')])

        for ep_file in episode_files:
            ep_path = os.path.join(episodes_dir, ep_file)
            with open(ep_path, 'r') as f:
                ep_data = json.load(f)

            # 궤적 로드
            traj_file = ep_file.replace('ep_', 'traj_')
            traj_path = os.path.join(trajectories_dir, traj_file)
            trajectory = []
            if os.path.exists(traj_path):
                with open(traj_path, 'r') as f:
                    trajectory = json.load(f)

            # 이미지 생성
            start = tuple(ep_data.get('start_pos', (0, 0)))
            goal = tuple(ep_data.get('goal_pos', (0, 0)))
            method_name = ep_data.get('method_name', 'unknown')
            episode_id = ep_data.get('episode_id', 0)
            success = ep_data.get('success', False)
            collision = ep_data.get('collision', False)

            save_path = os.path.join(images_dir, 'episodes', f'{ep_file.replace(".json", ".png")}')
            self.plot_single_episode(
                trajectory=trajectory,
                start=start,
                goal=goal,
                method_name=method_name,
                episode_id=episode_id,
                success=success,
                collision=collision,
                save_path=save_path
            )

        print(f"Generated episode images in: {images_dir}/episodes/")

        # 전체 결과 로드
        all_episodes_path = os.path.join(results_dir, 'all_episodes.json')
        if os.path.exists(all_episodes_path):
            with open(all_episodes_path, 'r') as f:
                all_results = json.load(f)

            # 시나리오별 요약 이미지
            scenarios = set(r.get('scenario', '') for r in all_results)
            for scenario in scenarios:
                scenario_results = [r for r in all_results if r.get('scenario') == scenario]
                save_path = os.path.join(images_dir, f'summary_{scenario.replace(".xml", "")}.png')
                self.plot_scenario_summary(
                    results=scenario_results,
                    scenario_name=scenario,
                    save_path=save_path
                )

            print(f"Generated summary images in: {images_dir}/")


def main():
    """테스트"""
    import tempfile

    viz = TrajectoryVisualizer()

    # 테스트 궤적 생성
    trajectory = []
    for i in range(100):
        t = i / 100.0
        x = -10 + t * 20
        y = -10 + t * 20 + np.sin(t * 10) * 2
        trajectory.append({
            'robot_pos': (x, y),
            'timestamp': i * 0.1
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_trajectory.png')
        viz.plot_single_episode(
            trajectory=trajectory,
            start=(-10, -10),
            goal=(10, 10),
            method_name='DWA',
            episode_id=0,
            success=True,
            collision=False,
            save_path=save_path,
            show=False
        )
        print(f"Test image saved to: {save_path}")


if __name__ == '__main__':
    main()
