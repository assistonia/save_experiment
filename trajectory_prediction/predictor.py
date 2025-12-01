#!/usr/bin/env python3
"""
SingularTrajectory Model Wrapper

실시간 경로 예측을 위한 SingularTrajectory 모델 래퍼.
기존 모델 코드를 수정하지 않고 독립적으로 동작.
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import threading

# SingularTrajectory 경로 추가 (도커/호스트 자동 감지)
if os.path.exists('/SingularTrajectory'):
    SINGULAR_PATH = '/SingularTrajectory'
else:
    SINGULAR_PATH = '/home/pyongjoo/Desktop/newstart/SingularTrajectory'

if SINGULAR_PATH not in sys.path:
    sys.path.insert(0, SINGULAR_PATH)


class TrajectoryPredictor:
    """실시간 경로 예측기"""

    def __init__(self, config=None):
        """
        Args:
            config: PredictionConfig 인스턴스 또는 None (기본값 사용)
        """
        if config is None:
            from .prediction_config import PredictionConfig
            config = PredictionConfig('warehouse')

        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 관련
        self.model = None
        self.hyper_params = None
        self._model_loaded = False
        self._model_lock = threading.Lock()

        # 궤적 히스토리 (agent_id -> list of (x, y, timestamp))
        self.trajectory_history: Dict[int, List[Tuple[float, float, float]]] = defaultdict(list)
        self.max_history_len = config.obs_len * 3  # 여유 있게 저장

        # 장애물 정보
        self.obstacles = config.obstacles
        self.walls = config.walls

    def load_model(self) -> bool:
        """모델 로드"""
        if self._model_loaded:
            return True

        with self._model_lock:
            if self._model_loaded:
                return True

            try:
                from SingularTrajectory import SingularTrajectory
                import baseline
                from utils import DotDict, get_exp_config

                # 설정 로드
                self.hyper_params = get_exp_config(self.config.config_path)
                self.hyper_params.s = self.hyper_params.num_samples

                # 베이스라인 모듈
                baseline_module = getattr(baseline, self.hyper_params.baseline)

                # Diffusion 설정
                diff_cfg = DotDict({
                    'scheduler': 'ddim',
                    'steps': 10,
                    'beta_start': 1.e-4,
                    'beta_end': 5.e-2,
                    'beta_schedule': 'linear',
                    'k': self.hyper_params.k,
                    's': self.hyper_params.num_samples
                })

                # 모델 생성
                PredictorModel = baseline_module.TrajectoryPredictor(diff_cfg)
                hook_func = DotDict({
                    "model_forward_pre_hook": baseline_module.model_forward_pre_hook,
                    "model_forward": baseline_module.model_forward,
                    "model_forward_post_hook": baseline_module.model_forward_post_hook
                })

                self.model = SingularTrajectory(
                    baseline_model=PredictorModel,
                    hook_func=hook_func,
                    hyper_params=self.hyper_params
                ).to(self.device)

                # 체크포인트 로드
                model_path = os.path.join(self.config.checkpoint_dir, 'model_best.pth')
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()

                self._model_loaded = True
                print(f"[TrajectoryPredictor] Model loaded on {self.device}")
                return True

            except Exception as e:
                print(f"[TrajectoryPredictor] Failed to load model: {e}")
                import traceback
                traceback.print_exc()
                return False

    def update_agent(self, agent_id: int, x: float, y: float, timestamp: float):
        """에이전트 위치 업데이트 (0.4초 간격으로 샘플링)"""
        history = self.trajectory_history[agent_id]

        # 최소 시간 간격 체크 (SingularTrajectory는 0.4초 간격 데이터 기대)
        min_interval = self.config.dt * 0.8  # 0.32초 (약간 여유)
        if history and (timestamp - history[-1][2]) < min_interval:
            return

        history.append((x, y, timestamp))

        # 히스토리 길이 제한
        if len(history) > self.max_history_len:
            self.trajectory_history[agent_id] = history[-self.max_history_len:]

    def update_agents_batch(self, agents: List[Dict]):
        """
        여러 에이전트 일괄 업데이트

        Args:
            agents: [{'id': int, 'x': float, 'y': float, 'timestamp': float}, ...]
        """
        for agent in agents:
            self.update_agent(
                agent_id=agent['id'],
                x=agent['x'],
                y=agent['y'],
                timestamp=agent['timestamp']
            )

    def get_observation(self, agent_id: int) -> Optional[np.ndarray]:
        """
        에이전트의 관측 궤적 반환 (obs_len 만큼)

        Returns:
            (obs_len, 2) 배열 또는 None (데이터 부족 시)
        """
        history = self.trajectory_history.get(agent_id, [])

        if len(history) < self.config.obs_len:
            return None

        # 최근 obs_len개 추출
        recent = history[-self.config.obs_len:]
        obs = np.array([[p[0], p[1]] for p in recent])
        return obs

    def get_all_valid_observations(self) -> Tuple[List[int], np.ndarray]:
        """
        모든 유효한 에이전트의 관측 궤적 반환

        Returns:
            (agent_ids, observations) - agent_ids는 리스트, observations는 (N, obs_len, 2) 배열
        """
        agent_ids = []
        observations = []

        for agent_id in self.trajectory_history.keys():
            obs = self.get_observation(agent_id)
            if obs is not None:
                agent_ids.append(agent_id)
                observations.append(obs)

        if not observations:
            return [], np.array([])

        return agent_ids, np.array(observations)

    def predict(self, agent_ids: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        경로 예측 수행

        Args:
            agent_ids: 예측할 에이전트 ID 리스트 (None이면 모든 유효 에이전트)

        Returns:
            {agent_id: {
                'obs_traj': (obs_len, 2),
                'pred_samples': (num_samples, pred_len, 2),
                'pred_best': (pred_len, 2),
                'pred_mean': (pred_len, 2)
            }}
        """
        if not self._model_loaded:
            if not self.load_model():
                return {}

        # 관측 데이터 수집
        if agent_ids is None:
            valid_ids, observations = self.get_all_valid_observations()
        else:
            valid_ids = []
            observations = []
            for aid in agent_ids:
                obs = self.get_observation(aid)
                if obs is not None:
                    valid_ids.append(aid)
                    observations.append(obs)
            observations = np.array(observations) if observations else np.array([])

        if len(valid_ids) == 0:
            return {}

        # 최소 에이전트 수 체크 (KMeans 등에서 필요)
        if len(valid_ids) < 2:
            return {}

        # 추론
        try:
            predictions = self._run_inference(observations)
        except Exception as e:
            print(f"[TrajectoryPredictor] Inference error: {e}")
            return {}

        # 결과 구성
        results = {}
        for i, agent_id in enumerate(valid_ids):
            pred_samples = predictions[i]  # (num_samples, pred_len, 2)

            results[agent_id] = {
                'obs_traj': observations[i],
                'pred_samples': pred_samples,
                'pred_best': pred_samples[0],  # 첫 번째 샘플
                'pred_mean': np.mean(pred_samples, axis=0)  # 평균
            }

        return results

    def _run_inference(self, observations: np.ndarray) -> np.ndarray:
        """
        모델 추론 실행

        Args:
            observations: (N, obs_len, 2) 배열

        Returns:
            (N, num_samples, pred_len, 2) 배열
        """
        with self._model_lock:
            n_ped = observations.shape[0]
            pred_len = self.config.pred_len
            k = self.hyper_params.k
            s = self.hyper_params.num_samples  # 원래 값 (20)

            # 텐서 변환
            obs_traj = torch.tensor(observations, dtype=torch.float32).to(self.device)

            # === Singular space 초기화 (최초 1회만) ===
            # calculate_parameters는 V_trunc와 C_anchor를 초기화함.
            # trajectory_collector처럼 실제 데이터 기반으로 초기화 필요.
            if not hasattr(self, '_params_initialized') or not self._params_initialized:
                from utils import augment_trajectory

                # 최소 샘플 수 확보를 위해 복제 (KMeans n_clusters=20)
                min_samples = s  # 20
                if n_ped < min_samples:
                    repeat_factor = (min_samples // n_ped) + 1
                    obs_for_init = obs_traj.repeat(repeat_factor, 1, 1)[:min_samples]
                    obs_for_init = obs_for_init + torch.randn_like(obs_for_init) * 0.01
                else:
                    obs_for_init = obs_traj[:min_samples]

                # 선형 외삽으로 pred 생성 (obs 방향 유지)
                pred_for_init = self._create_linear_extrapolation(obs_for_init, pred_len)

                # augment_trajectory로 데이터 증강
                obs_aug, pred_aug = augment_trajectory(obs_for_init.cpu(), pred_for_init.cpu())
                self.model.calculate_parameters(obs_aug, pred_aug)
                self._params_initialized = True
                print(f"[TrajectoryPredictor] Singular space initialized with {obs_for_init.shape[0]} samples")

            # Adaptive anchor
            adaptive_anchor = torch.zeros((n_ped, k, s), dtype=torch.float32).to(self.device)

            # Scene mask (모든 에이전트 동일 씬)
            scene_mask = torch.ones((n_ped, n_ped), dtype=torch.bool).to(self.device)

            # 추론
            with torch.no_grad():
                addl_info = {"scene_mask": scene_mask, "num_samples": s}
                output = self.model(obs_traj, adaptive_anchor, addl_info=addl_info)

            # 결과: (num_samples, n_ped, pred_len, 2) -> (n_ped, num_samples, pred_len, 2)
            pred_traj_recon = output['recon_traj'].permute(1, 0, 2, 3).cpu().numpy()

            # 후처리: 방향 보정 + 위치 보정 + 스무딩 + 장애물 회피
            pred_final = self._postprocess(observations, pred_traj_recon)

            return pred_final

    def _create_linear_extrapolation(self, obs_traj: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        관측 궤적에서 선형 외삽으로 예측 궤적 생성 (augment용)

        Args:
            obs_traj: (N, obs_len, 2)
            pred_len: 예측 길이

        Returns:
            (N, pred_len, 2)
        """
        # 마지막 두 점에서 속도 계산
        velocity = obs_traj[:, -1, :] - obs_traj[:, -2, :]  # (N, 2)
        last_pos = obs_traj[:, -1, :]  # (N, 2)

        pred = []
        for t in range(1, pred_len + 1):
            pred.append(last_pos + velocity * t)

        return torch.stack(pred, dim=1).to(obs_traj.device)  # (N, pred_len, 2)

    def _postprocess(self, observations: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """
        예측 후처리: 방향 보정 + 위치 보정 + 스무딩 + 장애물 회피

        모델 singular space가 잘못 초기화되면 예측 방향이 반대로 나올 수 있음.
        observation 방향과 비교하여 필요시 예측을 반전.

        Args:
            observations: (N, obs_len, 2)
            predictions: (N, num_samples, pred_len, 2)

        Returns:
            (N, num_samples, pred_len, 2)
        """
        n_ped, num_samples, pred_len, _ = predictions.shape
        result = np.zeros_like(predictions)

        for i in range(n_ped):
            obs_i = observations[i]
            last_obs = obs_i[-1]  # 관측 마지막 위치

            # 관측 방향 (전체 이동 방향)
            obs_direction = obs_i[-1] - obs_i[0]
            obs_dir_norm = np.linalg.norm(obs_direction)

            for sample_idx in range(num_samples):
                pred_sample = predictions[i, sample_idx].copy()

                # 예측 방향
                pred_direction = pred_sample[-1] - pred_sample[0]
                pred_dir_norm = np.linalg.norm(pred_direction)

                # 방향 일치 여부 확인 (내적이 음수면 반대 방향)
                if obs_dir_norm > 0.1 and pred_dir_norm > 0.1:
                    dot_product = np.dot(obs_direction, pred_direction)
                    if dot_product < 0:
                        # 예측이 반대 방향이면, 모델 예측의 "형태"는 유지하면서 방향만 반전
                        # pred_sample 상대 좌표를 구해서 방향 반전
                        pred_relative = pred_sample - pred_sample[0]  # 첫 점 기준 상대 좌표
                        pred_relative = -pred_relative  # 방향 반전
                        pred_sample = last_obs + pred_relative  # 마지막 obs 기준으로 재배치

                # 예측 시작점을 관측 마지막 위치로 맞추기
                pred_start = pred_sample[0]
                offset = last_obs - pred_start
                pred_sample = pred_sample + offset

                # 스무딩 적용 (moving average)
                smoothed = self._smooth_trajectory(pred_sample, window_size=3)

                # 장애물 회피
                corrected = self._apply_obstacle_avoidance(smoothed)

                result[i, sample_idx] = corrected

        return result

    def _smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 3) -> np.ndarray:
        """
        궤적 스무딩 (moving average)

        Args:
            trajectory: (pred_len, 2)
            window_size: 스무딩 윈도우 크기

        Returns:
            (pred_len, 2) 스무딩된 궤적
        """
        if len(trajectory) < window_size:
            return trajectory

        smoothed = np.zeros_like(trajectory)
        half_window = window_size // 2

        for i in range(len(trajectory)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(trajectory), i + half_window + 1)
            smoothed[i] = np.mean(trajectory[start_idx:end_idx], axis=0)

        return smoothed

    def _apply_obstacle_avoidance(self, trajectory: np.ndarray, margin: float = 0.3) -> np.ndarray:
        """궤적에 장애물 회피 적용"""
        corrected = trajectory.copy()

        for i in range(len(corrected)):
            x, y = corrected[i]

            # 벽 경계 체크
            x = max(self.walls['x_min'] + margin, min(self.walls['x_max'] - margin, x))
            y = max(self.walls['y_min'] + margin, min(self.walls['y_max'] - margin, y))

            # 장애물 체크
            for obs in self.obstacles:
                if (obs['x_min'] - margin < x < obs['x_max'] + margin and
                    obs['y_min'] - margin < y < obs['y_max'] + margin):
                    # 가장 가까운 외부로 밀어냄
                    x, y = self._push_out_of_obstacle(x, y, obs, margin)
                    break

            corrected[i] = [x, y]

        return corrected

    def _push_out_of_obstacle(self, x: float, y: float, obs: Dict, margin: float) -> Tuple[float, float]:
        """장애물 내부의 점을 가장 가까운 외부로 밀어냄"""
        dist_left = x - (obs['x_min'] - margin)
        dist_right = (obs['x_max'] + margin) - x
        dist_bottom = y - (obs['y_min'] - margin)
        dist_top = (obs['y_max'] + margin) - y

        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)

        if min_dist == dist_left:
            return obs['x_min'] - margin, y
        elif min_dist == dist_right:
            return obs['x_max'] + margin, y
        elif min_dist == dist_bottom:
            return x, obs['y_min'] - margin
        else:
            return x, obs['y_max'] + margin

    def clear_agent(self, agent_id: int):
        """특정 에이전트 히스토리 삭제"""
        if agent_id in self.trajectory_history:
            del self.trajectory_history[agent_id]

    def clear_all(self):
        """모든 히스토리 삭제"""
        self.trajectory_history.clear()

    def get_agent_count(self) -> int:
        """현재 추적 중인 에이전트 수"""
        return len(self.trajectory_history)

    def get_valid_agent_count(self) -> int:
        """예측 가능한 에이전트 수 (충분한 히스토리 보유)"""
        count = 0
        for agent_id in self.trajectory_history:
            if len(self.trajectory_history[agent_id]) >= self.config.obs_len:
                count += 1
        return count
