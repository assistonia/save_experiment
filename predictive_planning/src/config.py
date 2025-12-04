#!/usr/bin/env python3
"""
Predictive Planning Configuration

예측 기반 경로 계획 모듈의 설정 관리.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any


@dataclass
class PredictivePlanningConfig:
    """예측 기반 경로 계획 설정"""

    # === 맵 설정 ===
    x_range: Tuple[float, float] = (-12.0, 12.0)
    y_range: Tuple[float, float] = (-12.0, 12.0)
    resolution: float = 0.2  # A* 그리드 해상도 (m)

    # === 예측 파라미터 ===
    time_step: float = 0.4       # 예측 시간 간격 (초)
    num_samples: int = 20        # 예측 샘플 수
    pred_horizon: int = 12       # 예측 시점 수 (12 * 0.4 = 4.8초)

    # === 로봇 파라미터 ===
    robot_radius: float = 0.25   # 로봇 반경 (m)
    robot_velocity: float = 0.5  # 로봇 평균 속도 (m/s) - Pioneer P3-DX 특성 반영

    # === 비용 계산 파라미터 (핵심!) ===
    sigma_scale: float = 0.8     # 개별 샘플 반경 축소 계수 [합의된 값]
    sigma_min: float = 0.3       # 최소 반경 (m)
    sigma_base: float = 0.5      # 기본 반경 (속도 0일 때)

    # === CIGP 방향성 파라미터 ===
    use_direction_cost: bool = True    # tau 방향성 비용 사용 여부
    gamma1: float = 0.5                # 개인 공간 가중치

    # === A* 파라미터 ===
    heuristic_weight: float = 1.0      # 휴리스틱 가중치
    social_cost_weight: float = 2.0    # 소셜 비용 가중치
    max_iterations: int = 10000        # 최대 반복 횟수

    # === 장애물 설정 (warehouse) ===
    static_obstacles: List[Dict[str, float]] = field(default_factory=list)

    def __post_init__(self):
        """초기화 후 처리"""
        if not self.static_obstacles:
            self.static_obstacles = self._get_warehouse_obstacles()

    def _get_warehouse_obstacles(self) -> List[Dict[str, float]]:
        """Warehouse 장애물 리스트"""
        return [
            {'x_min': -12, 'x_max': -10, 'y_min': -5, 'y_max': 4},  # Shelf 1
            {'x_min': -7, 'x_max': -5, 'y_min': -5, 'y_max': 4},   # Shelf 2
            {'x_min': -2, 'x_max': 0, 'y_min': -5, 'y_max': 4},    # Shelf 3
            {'x_min': 3, 'x_max': 5, 'y_min': -5, 'y_max': 4},     # Shelf 4
            {'x_min': 10, 'x_max': 12, 'y_min': -5, 'y_max': 4},   # Shelf 5
        ]

    @property
    def walls(self) -> Dict[str, float]:
        """벽 경계"""
        return {
            'x_min': self.x_range[0],
            'x_max': self.x_range[1],
            'y_min': self.y_range[0],
            'y_max': self.y_range[1]
        }

    @property
    def grid_width(self) -> int:
        """그리드 너비"""
        return int((self.x_range[1] - self.x_range[0]) / self.resolution)

    @property
    def grid_height(self) -> int:
        """그리드 높이"""
        return int((self.y_range[1] - self.y_range[0]) / self.resolution)

    @property
    def max_prediction_time(self) -> float:
        """최대 예측 시간 (초)"""
        return self.pred_horizon * self.time_step

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """월드 좌표 -> 그리드 좌표"""
        gx = int((x - self.x_range[0]) / self.resolution)
        gy = int((y - self.y_range[0]) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """그리드 좌표 -> 월드 좌표"""
        x = self.x_range[0] + (gx + 0.5) * self.resolution
        y = self.y_range[0] + (gy + 0.5) * self.resolution
        return x, y

    def is_in_obstacle(self, x: float, y: float, margin: float = 0.0) -> bool:
        """해당 위치가 장애물 내부인지 확인"""
        # 벽 체크
        if (x <= self.walls['x_min'] + margin or
            x >= self.walls['x_max'] - margin or
            y <= self.walls['y_min'] + margin or
            y >= self.walls['y_max'] - margin):
            return True

        # 장애물 체크
        for obs in self.static_obstacles:
            if (obs['x_min'] - margin <= x <= obs['x_max'] + margin and
                obs['y_min'] - margin <= y <= obs['y_max'] + margin):
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'x_range': self.x_range,
            'y_range': self.y_range,
            'resolution': self.resolution,
            'time_step': self.time_step,
            'num_samples': self.num_samples,
            'pred_horizon': self.pred_horizon,
            'robot_radius': self.robot_radius,
            'robot_velocity': self.robot_velocity,
            'sigma_scale': self.sigma_scale,
            'sigma_min': self.sigma_min,
            'social_cost_weight': self.social_cost_weight,
        }


# 기본 설정 인스턴스
DEFAULT_CONFIG = PredictivePlanningConfig()
