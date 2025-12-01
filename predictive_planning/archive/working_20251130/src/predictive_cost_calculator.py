#!/usr/bin/env python3
"""
Predictive Cost Calculator

합의된 "20개 샘플 신뢰 + 개별 sigma 축소" 방식 구현.

핵심 아이디어:
1. 각 샘플의 위험 반경(sigma)을 0.8배로 축소
2. 20개 샘플 중 실제로 겹치는 비율로 확률적 희석
3. CIGP의 방향성(tau) 개념 적용
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .predicted_trajectory import PredictedTrajectory, PredictedTrajectoryArray
from .config import PredictivePlanningConfig


@dataclass
class CostBreakdown:
    """비용 분석 결과"""
    total_cost: float
    agent_costs: Dict[int, float]
    hit_counts: Dict[int, int]
    time_index: int
    arrival_time: float


class PredictiveCostCalculator:
    """
    예측 기반 확률적 비용 계산기

    Args:
        config: 설정 객체
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None):
        self.config = config or PredictivePlanningConfig()

        # 현재 예측 데이터
        self._predictions: Optional[PredictedTrajectoryArray] = None

        # 로봇 상태
        self._robot_x: float = 0.0
        self._robot_y: float = 0.0
        self._robot_vx: float = 0.0
        self._robot_vy: float = 0.0

    def update_predictions(self, predictions: PredictedTrajectoryArray):
        """예측 데이터 업데이트"""
        self._predictions = predictions

    def update_robot_state(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """로봇 상태 업데이트"""
        self._robot_x = x
        self._robot_y = y
        self._robot_vx = vx
        self._robot_vy = vy

    def calculate_cost(self, node_x: float, node_y: float, g_cost: float) -> float:
        """
        특정 노드에서의 예측 기반 소셜 비용 계산

        Args:
            node_x, node_y: 평가할 위치
            g_cost: 시작점에서 해당 노드까지의 거리 (m)

        Returns:
            소셜 비용 (0.0 ~ 무한대)
        """
        if self._predictions is None or len(self._predictions) == 0:
            return 0.0

        # 1. 도착 예정 시간 계산
        t_arrival = g_cost / self.config.robot_velocity
        time_index = int(t_arrival / self.config.time_step)

        # 예측 범위 벗어나면 비용 0
        if time_index >= self.config.pred_horizon:
            return 0.0

        total_cost = 0.0

        # 2. 모든 예측된 에이전트에 대해 비용 계산
        for traj in self._predictions:
            agent_cost = self._calculate_agent_cost(
                node_x, node_y, t_arrival, time_index, traj
            )
            total_cost += agent_cost

        return total_cost

    def calculate_cost_detailed(self, node_x: float, node_y: float,
                                g_cost: float) -> CostBreakdown:
        """
        상세 비용 분석 (디버깅용)
        """
        if self._predictions is None or len(self._predictions) == 0:
            return CostBreakdown(
                total_cost=0.0,
                agent_costs={},
                hit_counts={},
                time_index=0,
                arrival_time=0.0
            )

        t_arrival = g_cost / self.config.robot_velocity
        time_index = int(t_arrival / self.config.time_step)

        if time_index >= self.config.pred_horizon:
            return CostBreakdown(
                total_cost=0.0,
                agent_costs={},
                hit_counts={},
                time_index=time_index,
                arrival_time=t_arrival
            )

        agent_costs = {}
        hit_counts = {}
        total_cost = 0.0

        for traj in self._predictions:
            cost, hits = self._calculate_agent_cost_detailed(
                node_x, node_y, t_arrival, time_index, traj
            )
            agent_costs[traj.agent_id] = cost
            hit_counts[traj.agent_id] = hits
            total_cost += cost

        return CostBreakdown(
            total_cost=total_cost,
            agent_costs=agent_costs,
            hit_counts=hit_counts,
            time_index=time_index,
            arrival_time=t_arrival
        )

    def _calculate_agent_cost(self, node_x: float, node_y: float,
                              t_arrival: float, time_index: int,
                              traj: PredictedTrajectory) -> float:
        """
        단일 에이전트에 대한 비용 계산

        핵심 수식:
        1. sigma_k = max(0.8 * v_k, sigma_min)  # 개별 반경 축소
        2. cost_k = 0.5 * (1 - tau)  if dist < sigma_k else 0
        3. agent_cost = sum(cost_k) / num_samples  # 확률적 희석
        """
        # 해당 시간의 모든 샘플 위치
        future_positions = traj.get_position_at_time(t_arrival)  # (20, 2)

        # 해당 시간의 속도 (sigma 계산용)
        velocities = traj.get_velocity_at_time(t_arrival)  # (20,)

        # 방향 벡터 (tau 계산용)
        velocity_vectors = traj.get_velocity_vector_at_time(t_arrival)  # (20, 2)

        sample_costs = 0.0
        num_samples = traj.num_samples

        for k in range(num_samples):
            # A. 개별 샘플의 위험 반경 (0.8배 축소)
            sigma_k = max(
                self.config.sigma_scale * velocities[k],
                self.config.sigma_min
            )

            # B. 거리 계산
            dx = node_x - future_positions[k, 0]
            dy = node_y - future_positions[k, 1]
            dist = np.sqrt(dx**2 + dy**2)

            # 거리가 sigma보다 크면 비용 0
            if dist >= sigma_k:
                continue

            # C. 방향성(tau) 계산 (CIGP 스타일)
            if self.config.use_direction_cost:
                tau = self._calculate_tau(node_x, node_y, future_positions[k],
                                         velocity_vectors[k])
                # 방향이 반대(위협적)일수록 비용 증가
                # tau = 1이면 같은 방향 (안전), tau = -1이면 반대 방향 (위험)
                direction_factor = 0.5 * (1.0 - tau)
            else:
                direction_factor = 1.0

            # D. 거리 기반 감쇠 (가우시안 스타일)
            distance_factor = np.exp(-0.5 * (dist / sigma_k) ** 2)

            # 샘플 비용 누적
            sample_costs += direction_factor * distance_factor

        # E. 확률적 희석: 20개 샘플 평균
        agent_cost = sample_costs / num_samples

        return agent_cost

    def _calculate_agent_cost_detailed(self, node_x: float, node_y: float,
                                       t_arrival: float, time_index: int,
                                       traj: PredictedTrajectory) -> Tuple[float, int]:
        """상세 비용 계산 (hit count 포함)"""
        future_positions = traj.get_position_at_time(t_arrival)
        velocities = traj.get_velocity_at_time(t_arrival)
        velocity_vectors = traj.get_velocity_vector_at_time(t_arrival)

        sample_costs = 0.0
        hit_count = 0
        num_samples = traj.num_samples

        for k in range(num_samples):
            sigma_k = max(
                self.config.sigma_scale * velocities[k],
                self.config.sigma_min
            )

            dx = node_x - future_positions[k, 0]
            dy = node_y - future_positions[k, 1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist >= sigma_k:
                continue

            hit_count += 1

            if self.config.use_direction_cost:
                tau = self._calculate_tau(node_x, node_y, future_positions[k],
                                         velocity_vectors[k])
                direction_factor = 0.5 * (1.0 - tau)
            else:
                direction_factor = 1.0

            distance_factor = np.exp(-0.5 * (dist / sigma_k) ** 2)
            sample_costs += direction_factor * distance_factor

        agent_cost = sample_costs / num_samples

        return agent_cost, hit_count

    def _calculate_tau(self, node_x: float, node_y: float,
                       human_pos: np.ndarray, human_vel: np.ndarray) -> float:
        """
        방향성 계수(tau) 계산 (CIGP 논문 기반)

        tau = cos(theta) where theta = angle between robot_direction and human_direction
        tau = 1: 같은 방향 (안전)
        tau = -1: 반대 방향 (위험)
        """
        # 로봇 이동 방향 (현재 위치 -> 노드)
        robot_vec = np.array([node_x - self._robot_x, node_y - self._robot_y])
        robot_norm = np.linalg.norm(robot_vec)

        if robot_norm < 1e-6:
            return 0.0

        robot_vec = robot_vec / robot_norm

        # 사람 이동 방향
        human_norm = np.linalg.norm(human_vel)
        if human_norm < 1e-6:
            return 0.0

        human_dir = human_vel / human_norm

        # 내적 = cos(theta)
        tau = np.dot(robot_vec, human_dir)

        return float(tau)

    def get_cost_map(self, resolution: float = 0.5) -> Tuple[np.ndarray, dict]:
        """
        전체 맵의 비용 그리드 생성 (시각화용)

        Args:
            resolution: 그리드 해상도

        Returns:
            (cost_grid, metadata)
        """
        x_range = self.config.x_range
        y_range = self.config.y_range

        width = int((x_range[1] - x_range[0]) / resolution)
        height = int((y_range[1] - y_range[0]) / resolution)

        cost_grid = np.zeros((height, width))

        # 로봇 현재 위치에서 각 셀까지의 대략적인 g_cost 계산
        for gy in range(height):
            for gx in range(width):
                wx = x_range[0] + (gx + 0.5) * resolution
                wy = y_range[0] + (gy + 0.5) * resolution

                # 간단한 유클리드 거리를 g_cost로 사용
                g_cost = np.sqrt((wx - self._robot_x)**2 + (wy - self._robot_y)**2)

                cost = self.calculate_cost(wx, wy, g_cost)
                cost_grid[gy, gx] = cost

        metadata = {
            'resolution': resolution,
            'x_range': x_range,
            'y_range': y_range,
            'robot_x': self._robot_x,
            'robot_y': self._robot_y
        }

        return cost_grid, metadata


class SimpleCostCalculator:
    """
    단순화된 비용 계산기 (비교용)

    기존 CIGP 방식: 현재 위치만 사용, 예측 없음
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None):
        self.config = config or PredictivePlanningConfig()
        self._humans: List[Dict] = []

    def update_humans(self, humans: List[Dict]):
        """사람 위치 업데이트 (현재 시점만)"""
        self._humans = humans

    def calculate_cost(self, node_x: float, node_y: float) -> float:
        """현재 위치 기반 비용 계산 (시간 무시)"""
        total_cost = 0.0

        for human in self._humans:
            hx = human['x']
            hy = human['y']
            vx = human.get('vx', 0)
            vy = human.get('vy', 0)

            # 속도 기반 반경
            speed = np.sqrt(vx**2 + vy**2)
            sigma = max(speed * 0.5, 0.5)

            # 거리
            dist = np.sqrt((node_x - hx)**2 + (node_y - hy)**2)

            if dist < sigma:
                cost = np.exp(-0.5 * (dist / sigma) ** 2)
                total_cost += cost

        return total_cost
