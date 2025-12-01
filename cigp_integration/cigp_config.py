#!/usr/bin/env python3
"""
CIGP Configuration Manager

시나리오별 CIGP 설정 관리.
warehouse, mall 등 다양한 환경에 맞는 설정 제공.
"""

import numpy as np
from typing import List, Tuple, Dict, Any


class CIGPConfig:
    """CIGP 설정 클래스"""

    def __init__(self, scenario: str = 'warehouse'):
        """
        Args:
            scenario: 시나리오 이름 ('warehouse', 'mall', 'custom')
        """
        self.scenario = scenario
        self._load_scenario(scenario)

    def _load_scenario(self, scenario: str):
        """시나리오별 설정 로드"""
        if scenario == 'warehouse':
            self._load_warehouse()
        elif scenario == 'mall':
            self._load_mall()
        else:
            self._load_default()

    def _load_warehouse(self):
        """Warehouse 시나리오 설정"""
        # 맵 범위
        self.x_range = (-12.0, 12.0)
        self.y_range = (-12.0, 12.0)

        # CIGP 파라미터
        self.resolution = 0.1
        self.gamma1 = 0.5
        self.robot_radius = 0.25
        self.robot_fov = np.pi  # 180도
        self.robot_range = 10.0

        # 업데이트
        self.update_rate = 10.0
        self.replan_interval = 2

        # CCTV 설정
        self.n_cctvs = 4
        self.cctv_fov = np.pi / 2
        self.cctv_range = 15.0

        # 정적 장애물 (warehouse_pedsim.xml 기반)
        self.static_obstacles = self._get_warehouse_obstacles()

        # CCTV 위치 (수동 설정 시 사용)
        self.cctv_configs = [
            {'id': 0, 'position': (-10, 10), 'orientation': -np.pi/4, 'fov': np.pi/2, 'max_range': 15.0},
            {'id': 1, 'position': (10, 10), 'orientation': -3*np.pi/4, 'fov': np.pi/2, 'max_range': 15.0},
            {'id': 2, 'position': (-10, -10), 'orientation': np.pi/4, 'fov': np.pi/2, 'max_range': 15.0},
            {'id': 3, 'position': (10, -10), 'orientation': 3*np.pi/4, 'fov': np.pi/2, 'max_range': 15.0},
        ]

    def _load_mall(self):
        """Mall 시나리오 설정 (예시)"""
        self.x_range = (-20.0, 20.0)
        self.y_range = (-15.0, 15.0)
        self.resolution = 0.1
        self.gamma1 = 0.6
        self.robot_radius = 0.3
        self.robot_fov = np.pi
        self.robot_range = 8.0
        self.update_rate = 10.0
        self.replan_interval = 2
        self.n_cctvs = 6
        self.cctv_fov = np.pi / 2
        self.cctv_range = 20.0
        self.static_obstacles = []
        self.cctv_configs = []

    def _load_default(self):
        """기본 설정"""
        self.x_range = (-10.0, 10.0)
        self.y_range = (-10.0, 10.0)
        self.resolution = 0.1
        self.gamma1 = 0.5
        self.robot_radius = 0.25
        self.robot_fov = np.pi
        self.robot_range = 10.0
        self.update_rate = 10.0
        self.replan_interval = 2
        self.n_cctvs = 4
        self.cctv_fov = np.pi / 2
        self.cctv_range = 15.0
        self.static_obstacles = []
        self.cctv_configs = []

    def _get_warehouse_obstacles(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Warehouse 장애물 리스트"""
        return [
            # Outer walls
            ((-12.0, -12.0), (-12.0, 12.0)),
            ((12.0, -12.0), (12.0, 12.0)),
            ((-12.0, 12.0), (12.0, 12.0)),
            ((-12.0, -12.0), (12.0, -12.0)),

            # Shelf 1 (leftmost)
            ((-12, 4), (-12, -5)),
            ((-12, -5), (-10, -5)),
            ((-10, -5), (-10, 4)),
            ((-10, 4), (-12, 4)),

            # Shelf 2
            ((-7, 4), (-7, -5)),
            ((-7, -5), (-5, -5)),
            ((-5, -5), (-5, 4)),
            ((-5, 4), (-7, 4)),

            # Shelf 3
            ((-2, 4), (-2, -5)),
            ((-2, -5), (0, -5)),
            ((0, -5), (0, 4)),
            ((0, 4), (-2, 4)),

            # Shelf 4
            ((3, 4), (3, -5)),
            ((3, -5), (5, -5)),
            ((5, -5), (5, 4)),
            ((5, 4), (3, 4)),

            # Shelf 5 (rightmost)
            ((10, 4), (10, -5)),
            ((10, -5), (12, -5)),
            ((12, -5), (12, 4)),
            ((12, 4), (10, 4)),
        ]

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'scenario': self.scenario,
            'x_range': self.x_range,
            'y_range': self.y_range,
            'resolution': self.resolution,
            'gamma1': self.gamma1,
            'robot_radius': self.robot_radius,
            'robot_fov': self.robot_fov,
            'robot_range': self.robot_range,
            'update_rate': self.update_rate,
            'replan_interval': self.replan_interval,
            'n_cctvs': self.n_cctvs,
            'cctv_fov': self.cctv_fov,
            'cctv_range': self.cctv_range,
        }

    def to_ros_params(self) -> Dict[str, Any]:
        """ROS 파라미터 형식으로 변환"""
        return {
            'x_min': self.x_range[0],
            'x_max': self.x_range[1],
            'y_min': self.y_range[0],
            'y_max': self.y_range[1],
            'resolution': self.resolution,
            'gamma1': self.gamma1,
            'robot_radius': self.robot_radius,
            'robot_fov': self.robot_fov,
            'robot_range': self.robot_range,
            'update_rate': self.update_rate,
            'replan_interval': self.replan_interval,
            'n_cctvs': self.n_cctvs,
            'cctv_fov': self.cctv_fov,
            'cctv_range': self.cctv_range,
        }


def get_config(scenario: str = 'warehouse') -> CIGPConfig:
    """설정 인스턴스 반환"""
    return CIGPConfig(scenario)


# 기본 warehouse 설정
WAREHOUSE_CONFIG = CIGPConfig('warehouse')
