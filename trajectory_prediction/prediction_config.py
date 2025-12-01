#!/usr/bin/env python3
"""
Trajectory Prediction Configuration

예측 모델 및 시각화 관련 설정 관리.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class PredictionConfig:
    """경로 예측 설정 클래스"""

    def __init__(self, scenario: str = 'warehouse'):
        self.scenario = scenario
        self._load_scenario(scenario)

    def _load_scenario(self, scenario: str):
        if scenario == 'warehouse':
            self._load_warehouse()
        else:
            self._load_default()

    def _load_warehouse(self):
        """Warehouse 시나리오 설정"""
        import os

        # SingularTrajectory 모델 경로 (도커/호스트 자동 감지)
        # 도커: /SingularTrajectory, 호스트: /home/pyongjoo/Desktop/newstart/SingularTrajectory
        if os.path.exists('/SingularTrajectory'):
            self.model_base_path = '/SingularTrajectory'
        else:
            self.model_base_path = '/home/pyongjoo/Desktop/newstart/SingularTrajectory'

        self.config_path = f'{self.model_base_path}/config/stochastic/singulartrajectory-transformerdiffusion-eth.json'
        self.checkpoint_dir = f'{self.model_base_path}/checkpoints/SingularTrajectory-stochastic/eth/'

        # 예측 파라미터
        self.obs_len = 8          # 관측 프레임 수
        self.pred_len = 12        # 예측 프레임 수
        self.num_samples = 20     # 예측 샘플 수 (stochastic)
        self.dt = 0.4             # 시간 간격 (초)

        # 맵 범위
        self.x_range = (-12.0, 12.0)
        self.y_range = (-12.0, 12.0)

        # 장애물 정의
        self.obstacles = self._get_warehouse_obstacles()
        self.walls = {'x_min': -12, 'x_max': 12, 'y_min': -12, 'y_max': 12}

        # 시각화 설정 (도커/호스트 자동 감지)
        if os.path.exists('/environment/maps/warehouse.jpg'):
            self.map_image_path = '/environment/maps/warehouse.jpg'
        else:
            self.map_image_path = '/home/pyongjoo/Desktop/newstart/environment/maps/warehouse.jpg'

        self.plot_bounds = {
            'left': 180, 'right': 1600,
            'top': 120, 'bottom': 1650
        }

        # 업데이트 주기
        self.update_rate = 2.5    # Hz (0.4초 간격)
        self.publish_rate = 10.0  # Hz

    def _load_default(self):
        """기본 설정"""
        self._load_warehouse()  # 기본값으로 warehouse 사용

    def _get_warehouse_obstacles(self) -> List[Dict[str, float]]:
        """Warehouse 장애물 리스트 (warehouse_pedsim.xml 기준)"""
        return [
            {'x_min': -12, 'x_max': -10, 'y_min': -5, 'y_max': 4},  # Shelf 1
            {'x_min': -7, 'x_max': -5, 'y_min': -5, 'y_max': 4},   # Shelf 2
            {'x_min': -2, 'x_max': 0, 'y_min': -5, 'y_max': 4},    # Shelf 3
            {'x_min': 3, 'x_max': 5, 'y_min': -5, 'y_max': 4},     # Shelf 4
            {'x_min': 10, 'x_max': 12, 'y_min': -5, 'y_max': 4},   # Shelf 5
        ]

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """월드 좌표를 픽셀 좌표로 변환"""
        b = self.plot_bounds
        x_range = self.x_range[1] - self.x_range[0]
        y_range = self.y_range[1] - self.y_range[0]

        px = b['left'] + (x - self.x_range[0]) / x_range * (b['right'] - b['left'])
        py = b['bottom'] - (y - self.y_range[0]) / y_range * (b['bottom'] - b['top'])
        return px, py

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """픽셀 좌표를 월드 좌표로 변환"""
        b = self.plot_bounds
        x_range = self.x_range[1] - self.x_range[0]
        y_range = self.y_range[1] - self.y_range[0]

        x = self.x_range[0] + (px - b['left']) / (b['right'] - b['left']) * x_range
        y = self.y_range[0] + (b['bottom'] - py) / (b['bottom'] - b['top']) * y_range
        return x, y

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'scenario': self.scenario,
            'obs_len': self.obs_len,
            'pred_len': self.pred_len,
            'num_samples': self.num_samples,
            'dt': self.dt,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'update_rate': self.update_rate,
        }


# 기본 warehouse 설정
WAREHOUSE_CONFIG = PredictionConfig('warehouse')
