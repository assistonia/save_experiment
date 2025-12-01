#!/usr/bin/env python3
"""
Predicted Trajectory Data Structures

ROS 메시지 대신 사용할 순수 Python 데이터 클래스.
ROS 환경 없이도 테스트 가능하도록 설계.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import time


@dataclass
class PredictedTrajectory:
    """
    단일 에이전트의 예측 궤적

    Attributes:
        agent_id: 에이전트 고유 ID
        current_x, current_y: 현재 위치
        current_vx, current_vy: 현재 속도
        samples: (num_samples, pred_horizon, 2) 형태의 예측 샘플
        time_step: 시간 간격 (기본 0.4초)
    """
    agent_id: int
    current_x: float
    current_y: float
    current_vx: float = 0.0
    current_vy: float = 0.0
    samples: np.ndarray = field(default_factory=lambda: np.zeros((20, 12, 2)))
    time_step: float = 0.4
    num_samples: int = 20
    pred_horizon: int = 12

    def get_position_at_time(self, t: float) -> np.ndarray:
        """
        특정 시간에서의 모든 샘플 위치 반환

        Args:
            t: 현재로부터의 시간 (초)

        Returns:
            (num_samples, 2) 형태의 위치 배열
        """
        time_index = int(t / self.time_step)
        time_index = min(time_index, self.pred_horizon - 1)
        return self.samples[:, time_index, :]

    def get_velocity_at_time(self, t: float) -> np.ndarray:
        """
        특정 시간에서의 모든 샘플 속도 반환 (차분 기반)

        Args:
            t: 현재로부터의 시간 (초)

        Returns:
            (num_samples,) 형태의 속도 크기 배열
        """
        time_index = int(t / self.time_step)
        time_index = min(time_index, self.pred_horizon - 1)

        if time_index == 0:
            # 첫 시점: 현재 속도 사용
            return np.full(self.num_samples,
                          np.sqrt(self.current_vx**2 + self.current_vy**2))

        # 차분으로 속도 계산
        prev_pos = self.samples[:, time_index - 1, :]
        curr_pos = self.samples[:, time_index, :]
        velocity = np.linalg.norm(curr_pos - prev_pos, axis=1) / self.time_step
        return velocity

    def get_velocity_vector_at_time(self, t: float) -> np.ndarray:
        """
        특정 시간에서의 모든 샘플 속도 벡터 반환

        Args:
            t: 현재로부터의 시간 (초)

        Returns:
            (num_samples, 2) 형태의 속도 벡터 배열
        """
        time_index = int(t / self.time_step)
        time_index = min(time_index, self.pred_horizon - 1)

        if time_index == 0:
            return np.tile([self.current_vx, self.current_vy], (self.num_samples, 1))

        prev_pos = self.samples[:, time_index - 1, :]
        curr_pos = self.samples[:, time_index, :]
        return (curr_pos - prev_pos) / self.time_step

    def get_mean_trajectory(self) -> np.ndarray:
        """평균 예측 궤적 반환 (pred_horizon, 2)"""
        return np.mean(self.samples, axis=0)

    def get_std_trajectory(self) -> np.ndarray:
        """예측 궤적의 표준편차 반환 (pred_horizon, 2)"""
        return np.std(self.samples, axis=0)

    def get_sigma_at_time(self, t: float) -> float:
        """
        특정 시간에서의 sigma (표준편차 크기) 반환

        시그마 계산: σ_k = max(0.8 × v_k, 0.3)
        여기서 v_k는 해당 시간에서의 평균 속도

        Args:
            t: 현재로부터의 시간 (초)

        Returns:
            sigma 값 (최소 0.3)
        """
        # 해당 시간의 속도 계산
        velocity = self.get_velocity_at_time(t)
        mean_velocity = np.mean(velocity)

        # σ_k = max(0.8 × v_k, 0.3)
        sigma = max(0.8 * mean_velocity, 0.3)
        return sigma

    def to_flat_arrays(self) -> tuple:
        """ROS 메시지 호환을 위한 flat 배열 변환"""
        samples_x = self.samples[:, :, 0].flatten().tolist()
        samples_y = self.samples[:, :, 1].flatten().tolist()
        return samples_x, samples_y

    @classmethod
    def from_flat_arrays(cls, agent_id: int, current_x: float, current_y: float,
                        current_vx: float, current_vy: float,
                        samples_x: List[float], samples_y: List[float],
                        time_step: float = 0.4, num_samples: int = 20,
                        pred_horizon: int = 12):
        """flat 배열에서 객체 생성"""
        samples_x_np = np.array(samples_x).reshape(num_samples, pred_horizon)
        samples_y_np = np.array(samples_y).reshape(num_samples, pred_horizon)
        samples = np.stack([samples_x_np, samples_y_np], axis=-1)

        return cls(
            agent_id=agent_id,
            current_x=current_x,
            current_y=current_y,
            current_vx=current_vx,
            current_vy=current_vy,
            samples=samples,
            time_step=time_step,
            num_samples=num_samples,
            pred_horizon=pred_horizon
        )


@dataclass
class PredictedTrajectoryArray:
    """
    모든 에이전트의 예측 궤적 배열

    Attributes:
        trajectories: 에이전트별 예측 궤적 딕셔너리
        timestamp: 예측 시간
        prediction_timestamp: 시뮬레이션 시간
    """
    trajectories: Dict[int, PredictedTrajectory] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    prediction_timestamp: float = 0.0
    total_agents: int = 0
    predicted_agents: int = 0

    def add_trajectory(self, traj: PredictedTrajectory):
        """예측 궤적 추가"""
        self.trajectories[traj.agent_id] = traj
        self.predicted_agents = len(self.trajectories)

    def get_trajectory(self, agent_id: int) -> Optional[PredictedTrajectory]:
        """특정 에이전트의 예측 궤적 반환"""
        return self.trajectories.get(agent_id)

    def get_all_positions_at_time(self, t: float) -> Dict[int, np.ndarray]:
        """
        특정 시간에서 모든 에이전트의 샘플 위치 반환

        Returns:
            {agent_id: (num_samples, 2) 배열}
        """
        return {
            agent_id: traj.get_position_at_time(t)
            for agent_id, traj in self.trajectories.items()
        }

    def get_agent_ids(self) -> List[int]:
        """예측된 에이전트 ID 목록"""
        return list(self.trajectories.keys())

    def __len__(self):
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories.values())

    def __getitem__(self, index: int) -> PredictedTrajectory:
        """인덱스로 접근 (순서는 딕셔너리 삽입 순서)"""
        keys = list(self.trajectories.keys())
        if 0 <= index < len(keys):
            return self.trajectories[keys[index]]
        raise IndexError(f"Index {index} out of range for {len(keys)} trajectories")
