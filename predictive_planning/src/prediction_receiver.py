#!/usr/bin/env python3
"""
Prediction Receiver

기존 trajectory_prediction 모듈의 출력을 읽어서
PredictedTrajectoryArray 형태로 변환.

기존 trajectory_prediction 코드는 수정하지 않고,
그 출력을 구독하여 사용.
"""

import sys
import os
import numpy as np
from typing import Dict, Optional, Callable
from collections import defaultdict
import threading
import time

# 상위 경로 추가
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.dirname(MODULE_PATH)
if ENV_PATH not in sys.path:
    sys.path.insert(0, ENV_PATH)

from predictive_planning.src.predicted_trajectory import (
    PredictedTrajectory,
    PredictedTrajectoryArray
)
from predictive_planning.src.config import PredictivePlanningConfig


class PredictionReceiver:
    """
    예측 결과 수신기

    trajectory_prediction.predictor.TrajectoryPredictor의 출력을
    직접 호출하여 PredictedTrajectoryArray로 변환.

    ROS 토픽 대신 직접 모듈 호출 방식 사용 (더 간단하고 지연 없음).
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None):
        """
        Args:
            config: 설정 객체 (None이면 기본값 사용)
        """
        self.config = config or PredictivePlanningConfig()

        # 예측기 인스턴스 (lazy loading)
        self._predictor = None
        self._predictor_lock = threading.Lock()

        # 최신 예측 결과 캐시
        self._latest_predictions: Optional[PredictedTrajectoryArray] = None
        self._last_update_time: float = 0.0

        # 콜백 등록
        self._callbacks: list = []

    def _load_predictor(self):
        """TrajectoryPredictor 로드 (lazy loading)"""
        if self._predictor is not None:
            return True

        with self._predictor_lock:
            if self._predictor is not None:
                return True

            try:
                # trajectory_prediction 모듈 임포트
                from trajectory_prediction.predictor import TrajectoryPredictor
                from trajectory_prediction.prediction_config import PredictionConfig

                pred_config = PredictionConfig('warehouse')
                self._predictor = TrajectoryPredictor(pred_config)

                if self._predictor.load_model():
                    print("[PredictionReceiver] TrajectoryPredictor loaded successfully")
                    return True
                else:
                    print("[PredictionReceiver] Failed to load TrajectoryPredictor model")
                    self._predictor = None
                    return False

            except ImportError as e:
                print(f"[PredictionReceiver] Failed to import trajectory_prediction: {e}")
                return False
            except Exception as e:
                print(f"[PredictionReceiver] Error loading predictor: {e}")
                return False

    def update_agents(self, agents: list, timestamp: float = None):
        """
        에이전트 위치 업데이트

        Args:
            agents: [{'id': int, 'x': float, 'y': float, 'vx': float, 'vy': float}, ...]
            timestamp: 현재 시뮬레이션 시간 (None이면 자동)
        """
        if not self._load_predictor():
            return

        if timestamp is None:
            timestamp = time.time()

        # 에이전트 업데이트
        for agent in agents:
            self._predictor.update_agent(
                agent_id=agent['id'],
                x=agent['x'],
                y=agent['y'],
                timestamp=timestamp
            )

    def get_predictions(self) -> Optional[PredictedTrajectoryArray]:
        """
        현재 에이전트들에 대한 예측 수행 및 반환

        Returns:
            PredictedTrajectoryArray 또는 None (예측 불가 시)
        """
        if not self._load_predictor():
            return None

        # 예측 수행
        raw_predictions = self._predictor.predict()

        if not raw_predictions:
            return None

        # PredictedTrajectoryArray로 변환
        result = PredictedTrajectoryArray(
            prediction_timestamp=self._last_update_time,
            total_agents=self._predictor.get_agent_count(),
            predicted_agents=len(raw_predictions)
        )

        for agent_id, pred_data in raw_predictions.items():
            # pred_data 구조:
            # - obs_traj: (obs_len, 2)
            # - pred_samples: (num_samples, pred_len, 2)
            # - pred_best: (pred_len, 2)
            # - pred_mean: (pred_len, 2)

            obs_traj = pred_data['obs_traj']
            pred_samples = pred_data['pred_samples']

            # 현재 위치/속도 추출 (관측 마지막 2점에서)
            current_x = obs_traj[-1, 0]
            current_y = obs_traj[-1, 1]

            if len(obs_traj) >= 2:
                dt = self.config.time_step
                current_vx = (obs_traj[-1, 0] - obs_traj[-2, 0]) / dt
                current_vy = (obs_traj[-1, 1] - obs_traj[-2, 1]) / dt
            else:
                current_vx = 0.0
                current_vy = 0.0

            # PredictedTrajectory 생성
            traj = PredictedTrajectory(
                agent_id=agent_id,
                current_x=current_x,
                current_y=current_y,
                current_vx=current_vx,
                current_vy=current_vy,
                samples=pred_samples,
                time_step=self.config.time_step,
                num_samples=pred_samples.shape[0],
                pred_horizon=pred_samples.shape[1]
            )

            result.add_trajectory(traj)

        self._latest_predictions = result
        self._last_update_time = time.time()

        # 콜백 호출
        for callback in self._callbacks:
            callback(result)

        return result

    def get_latest_predictions(self) -> Optional[PredictedTrajectoryArray]:
        """캐시된 최신 예측 결과 반환"""
        return self._latest_predictions

    def register_callback(self, callback: Callable[[PredictedTrajectoryArray], None]):
        """예측 결과 콜백 등록"""
        self._callbacks.append(callback)

    def clear(self):
        """예측기 상태 초기화"""
        if self._predictor:
            self._predictor.clear_all()
        self._latest_predictions = None


class MockPredictionReceiver:
    """
    테스트용 Mock 예측 수신기

    실제 모델 없이 간단한 선형 외삽으로 예측 생성.
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None):
        self.config = config or PredictivePlanningConfig()
        self._agent_history: Dict[int, list] = defaultdict(list)
        self._latest_predictions: Optional[PredictedTrajectoryArray] = None

    def update_agents(self, agents: list, timestamp: float = None):
        """에이전트 위치 업데이트"""
        for agent in agents:
            history = self._agent_history[agent['id']]
            history.append({
                'x': agent['x'],
                'y': agent['y'],
                'vx': agent.get('vx', 0),
                'vy': agent.get('vy', 0),
                'timestamp': timestamp or time.time()
            })

            # 최대 8개 유지
            if len(history) > 8:
                self._agent_history[agent['id']] = history[-8:]

    def get_predictions(self) -> Optional[PredictedTrajectoryArray]:
        """선형 외삽 기반 mock 예측 생성"""
        result = PredictedTrajectoryArray(
            prediction_timestamp=time.time(),
            total_agents=len(self._agent_history),
            predicted_agents=0
        )

        for agent_id, history in self._agent_history.items():
            if len(history) < 2:
                continue

            # 현재 상태
            current = history[-1]
            prev = history[-2]

            current_x = current['x']
            current_y = current['y']

            # 속도 계산
            dt = self.config.time_step
            vx = (current['x'] - prev['x']) / dt
            vy = (current['y'] - prev['y']) / dt

            # 20개 샘플 생성 (노이즈 추가)
            samples = np.zeros((self.config.num_samples, self.config.pred_horizon, 2))

            for k in range(self.config.num_samples):
                # 약간의 랜덤 노이즈
                noise_vx = vx + np.random.normal(0, 0.1)
                noise_vy = vy + np.random.normal(0, 0.1)

                for t in range(self.config.pred_horizon):
                    time_offset = (t + 1) * dt
                    samples[k, t, 0] = current_x + noise_vx * time_offset
                    samples[k, t, 1] = current_y + noise_vy * time_offset

                    # 추가 노이즈 (시간에 따라 증가)
                    samples[k, t, 0] += np.random.normal(0, 0.05 * (t + 1))
                    samples[k, t, 1] += np.random.normal(0, 0.05 * (t + 1))

            traj = PredictedTrajectory(
                agent_id=agent_id,
                current_x=current_x,
                current_y=current_y,
                current_vx=vx,
                current_vy=vy,
                samples=samples,
                time_step=dt,
                num_samples=self.config.num_samples,
                pred_horizon=self.config.pred_horizon
            )

            result.add_trajectory(traj)

        self._latest_predictions = result
        return result

    def get_latest_predictions(self) -> Optional[PredictedTrajectoryArray]:
        return self._latest_predictions

    def clear(self):
        self._agent_history.clear()
        self._latest_predictions = None
