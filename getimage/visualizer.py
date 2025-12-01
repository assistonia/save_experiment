"""
Visualizer

Ground Truth와 검출 결과를 지도에 시각화.
검증 및 디버깅용.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless mode - GUI 없이 이미지 저장
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyArrow
from typing import List, Tuple, Dict, Optional
import os

from warehouse_config import (
    SHELVES, AISLES, CORRIDORS, CCTV_CONFIGS,
    MAP_X_RANGE, MAP_Y_RANGE, is_in_valid_region
)
from human_state_extractor import HumanState


class WarehouseVisualizer:
    """
    Warehouse 맵에 사람 위치 시각화.
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 12)):
        """
        Args:
            figsize: Figure 크기
        """
        self.figsize = figsize
        self.fig = None
        self.ax = None

    def create_figure(self):
        """새 Figure 생성"""
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._draw_warehouse()
        return self.fig, self.ax

    def _draw_warehouse(self):
        """Warehouse 배경 그리기"""
        ax = self.ax

        # 맵 범위
        ax.set_xlim(MAP_X_RANGE[0] - 1, MAP_X_RANGE[1] + 1)
        ax.set_ylim(MAP_Y_RANGE[0] - 1, MAP_Y_RANGE[1] + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # 외벽
        outer_wall = Rectangle(
            (MAP_X_RANGE[0], MAP_Y_RANGE[0]),
            MAP_X_RANGE[1] - MAP_X_RANGE[0],
            MAP_Y_RANGE[1] - MAP_Y_RANGE[0],
            fill=False, edgecolor='black', linewidth=2
        )
        ax.add_patch(outer_wall)

        # 선반 (갈색)
        for shelf_name, (x_min, x_max, y_min, y_max) in SHELVES.items():
            shelf = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor='#8B4513',  # 갈색
                edgecolor='black',
                alpha=0.7,
                label=shelf_name
            )
            ax.add_patch(shelf)
            # 라벨
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            ax.text(cx, cy, shelf_name.replace('shelf_', 'S'),
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')

        # 통로 영역 (연한 녹색 배경)
        for aisle_name, (x_min, x_max, y_min, y_max) in AISLES.items():
            aisle = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor='#90EE90',  # 연녹색
                edgecolor='none',
                alpha=0.3
            )
            ax.add_patch(aisle)

        # CCTV 위치
        for cctv_id, config in CCTV_CONFIGS.items():
            x, y, z = config.position
            # 카메라 마커
            ax.plot(x, y, 'k^', markersize=12, label=f'CCTV {cctv_id}' if cctv_id == 0 else '')
            ax.text(x, y + 0.8, f'C{cctv_id}', ha='center', va='bottom', fontsize=9)

            # 시야 방향 화살표
            arrow_len = 3
            dx = arrow_len * np.cos(config.yaw)
            dy = arrow_len * np.sin(config.yaw)
            ax.arrow(x, y, dx, dy, head_width=0.5, head_length=0.3,
                     fc='gray', ec='gray', alpha=0.5)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Warehouse Map')

    def draw_ground_truth(self, humans: List[HumanState], color='blue', label='Ground Truth'):
        """
        Ground Truth 사람 위치 표시.

        Args:
            humans: HumanState 리스트
            color: 마커 색상
            label: 범례 라벨
        """
        for i, h in enumerate(humans):
            circle = Circle((h.px, h.py), h.radius, facecolor=color,
                            edgecolor='black', alpha=0.6,
                            label=label if i == 0 else '')
            self.ax.add_patch(circle)

            # 속도 벡터
            if h.speed > 0.1:
                self.ax.arrow(h.px, h.py, h.vx * 0.5, h.vy * 0.5,
                              head_width=0.15, head_length=0.1,
                              fc=color, ec=color, alpha=0.8)

            # ID 텍스트
            self.ax.text(h.px, h.py + h.radius + 0.2, f'GT{h.id}',
                        ha='center', va='bottom', fontsize=8, color=color)

    def draw_detections(self, detections: List[Tuple[float, float]],
                        color='red', label='Detection', marker='x'):
        """
        검출 결과 표시.

        Args:
            detections: [(x, y), ...] 좌표 리스트
            color: 마커 색상
            label: 범례 라벨
            marker: 마커 스타일
        """
        if not detections:
            return

        xs, ys = zip(*detections)
        self.ax.scatter(xs, ys, c=color, marker=marker, s=100,
                        label=label, zorder=5)

        # 유효 영역 체크 표시
        for i, (x, y) in enumerate(detections):
            if not is_in_valid_region(x, y):
                # 무효 영역이면 빨간 X 추가
                self.ax.scatter([x], [y], c='red', marker='X', s=200,
                                edgecolors='black', linewidths=2, zorder=6)

    def draw_detection_states(self, humans: List[HumanState],
                               color='green', label='Detected'):
        """
        검출된 HumanState 표시 (GT와 비교용).

        Args:
            humans: 검출된 HumanState 리스트
            color: 색상
            label: 라벨
        """
        for i, h in enumerate(humans):
            circle = Circle((h.px, h.py), h.radius, facecolor='none',
                            edgecolor=color, linewidth=2, linestyle='--',
                            alpha=0.8, label=label if i == 0 else '')
            self.ax.add_patch(circle)

            # 속도 벡터
            if h.speed > 0.1:
                self.ax.arrow(h.px, h.py, h.vx * 0.5, h.vy * 0.5,
                              head_width=0.15, head_length=0.1,
                              fc=color, ec=color, alpha=0.8)

            # ID 텍스트
            self.ax.text(h.px, h.py - h.radius - 0.3, f'D{h.id}',
                        ha='center', va='top', fontsize=8, color=color)

    def draw_trajectories(self, trajectories: Dict[int, List[Tuple[float, float]]],
                          color='purple', alpha=0.5):
        """
        궤적 표시.

        Args:
            trajectories: {human_id: [(x, y), ...]}
        """
        for human_id, traj in trajectories.items():
            if len(traj) < 2:
                continue

            xs, ys = zip(*traj)
            self.ax.plot(xs, ys, '-', color=color, alpha=alpha, linewidth=2)
            # 시작점
            self.ax.plot(xs[0], ys[0], 'o', color=color, markersize=5)
            # 끝점 (현재)
            self.ax.plot(xs[-1], ys[-1], 's', color=color, markersize=7)

    def draw_errors(self, gt_humans: List[HumanState],
                    det_humans: List[HumanState]):
        """
        GT와 검출 사이의 오차 표시.

        Args:
            gt_humans: GT HumanState 리스트
            det_humans: 검출 HumanState 리스트
        """
        # 간단한 거리 기반 매칭
        for gt in gt_humans:
            best_det = None
            best_dist = float('inf')

            for det in det_humans:
                dist = np.sqrt((gt.px - det.px)**2 + (gt.py - det.py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_det = det

            if best_det is not None and best_dist < 2.0:
                # 오차 선
                self.ax.plot([gt.px, best_det.px], [gt.py, best_det.py],
                             'r--', linewidth=1, alpha=0.5)
                # 오차 텍스트
                mid_x = (gt.px + best_det.px) / 2
                mid_y = (gt.py + best_det.py) / 2
                self.ax.text(mid_x, mid_y, f'{best_dist:.2f}m',
                            fontsize=7, color='red')

    def add_legend(self):
        """범례 추가"""
        self.ax.legend(loc='upper right')

    def save(self, filepath: str, dpi: int = 150):
        """이미지 저장"""
        self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")

    def show(self):
        """화면에 표시"""
        plt.show()

    def close(self):
        """Figure 닫기"""
        plt.close(self.fig)


def compute_detection_metrics(gt_humans: List[HumanState],
                               det_positions: List[Tuple[float, float]],
                               distance_threshold: float = 2.0) -> Dict:
    """
    검출 성능 지표 계산.

    Args:
        gt_humans: GT HumanState 리스트
        det_positions: 검출 좌표 리스트
        distance_threshold: 매칭 거리 임계값

    Returns:
        지표 딕셔너리
    """
    gt_positions = [(h.px, h.py) for h in gt_humans]

    # True Positives, False Positives, False Negatives
    tp = 0
    matched_gt = set()
    matched_det = set()
    position_errors = []

    for det_idx, (dx, dy) in enumerate(det_positions):
        best_gt_idx = None
        best_dist = distance_threshold

        for gt_idx, (gx, gy) in enumerate(gt_positions):
            if gt_idx in matched_gt:
                continue

            dist = np.sqrt((dx - gx)**2 + (dy - gy)**2)
            if dist < best_dist:
                best_dist = dist
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_det.add(det_idx)
            position_errors.append(best_dist)

    fp = len(det_positions) - tp  # False Positives (잘못된 검출)
    fn = len(gt_humans) - tp      # False Negatives (놓친 GT)

    # 지표 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_error = np.mean(position_errors) if position_errors else 0

    # 유효 영역 체크
    invalid_detections = sum(1 for x, y in det_positions if not is_in_valid_region(x, y))

    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_position_error': mean_error,
        'invalid_detections': invalid_detections,
        'total_detections': len(det_positions),
        'total_gt': len(gt_humans),
    }


if __name__ == '__main__':
    print("=== Visualizer Test ===")

    # 테스트 데이터
    gt_humans = [
        HumanState(id=0, px=-8.5, py=2, vx=0, vy=-0.5),   # Aisle 1
        HumanState(id=1, px=-3.5, py=-2, vx=0, vy=0.5),   # Aisle 2
        HumanState(id=2, px=1.5, py=0, vx=0.3, vy=0),     # Aisle 3
        HumanState(id=3, px=7.5, py=3, vx=0, vy=-0.3),    # Aisle 4
    ]

    # 검출 결과 (약간의 오차 포함)
    det_positions = [
        (-8.3, 2.1),    # GT 0 근처
        (-3.7, -1.8),   # GT 1 근처
        (1.6, 0.2),     # GT 2 근처
        (7.3, 3.2),     # GT 3 근처
        (-6, 0),        # FP (선반 위 - invalid)
    ]

    # 시각화
    viz = WarehouseVisualizer()
    viz.create_figure()
    viz.draw_ground_truth(gt_humans)
    viz.draw_detections(det_positions)
    viz.add_legend()

    # 지표 계산
    metrics = compute_detection_metrics(gt_humans, det_positions)
    print("\n=== Detection Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # 저장
    save_path = '/home/pyongjoo/Desktop/newstart/environment/getimage/test_visualization.png'
    viz.save(save_path)
    print(f"\nSaved to: {save_path}")

    viz.close()
