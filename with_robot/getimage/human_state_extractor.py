"""
Human State Extractor

검출된 사람 좌표를 받아서:
1. History Buffer 관리 (8프레임 = 3.2초)
2. 속도 계산
3. HumanState 생성
4. SingularTrajectory 모델 입력 텐서 생성
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time


@dataclass
class HumanState:
    """사람의 상태 정보 (CIGP 호환)"""
    id: int
    px: float  # x 위치
    py: float  # y 위치
    vx: float  # x 속도
    vy: float  # y 속도
    radius: float = 0.3

    @property
    def position(self) -> np.ndarray:
        return np.array([self.px, self.py])

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy])

    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)


@dataclass
class TrajectoryPoint:
    """궤적 포인트"""
    timestamp: float
    x: float
    y: float


class HumanStateExtractor:
    """
    검출 결과에서 HumanState 추출.

    - History Buffer로 과거 위치 저장
    - 연속 좌표로 속도 계산
    - SingularTrajectory 입력 텐서 생성
    """

    def __init__(self,
                 history_length: int = 8,
                 dt: float = 0.4,
                 max_association_dist: float = 1.0):
        """
        Args:
            history_length: 저장할 과거 프레임 수 (8 = 3.2초)
            dt: 프레임 간격 (초)
            max_association_dist: ID 연결 최대 거리 (m)
        """
        self.history_length = history_length
        self.dt = dt
        self.max_association_dist = max_association_dist

        # {human_id: deque([TrajectoryPoint, ...])}
        self.history_buffer: Dict[int, deque] = {}

        # ID 관리
        self.next_id = 0
        self.last_positions: Dict[int, Tuple[float, float]] = {}
        self.last_update_time: Dict[int, float] = {}

        # ID 타임아웃 (초)
        self.id_timeout = 2.0

    def update(self, detections: List[Tuple[float, float]], timestamp: float = None):
        """
        새로운 검출 결과로 업데이트.

        Args:
            detections: [(x, y), ...] 월드 좌표 리스트
            timestamp: 현재 시간 (None이면 자동)
        """
        if timestamp is None:
            timestamp = time.time()

        # 1. 기존 ID와 검출 결과 매칭
        matched_ids = self._associate_detections(detections)

        # 2. History Buffer 업데이트
        for det_idx, human_id in enumerate(matched_ids):
            x, y = detections[det_idx]

            # 새 ID면 버퍼 생성
            if human_id not in self.history_buffer:
                self.history_buffer[human_id] = deque(maxlen=self.history_length)

            # 포인트 추가
            self.history_buffer[human_id].append(
                TrajectoryPoint(timestamp=timestamp, x=x, y=y)
            )

            # 마지막 위치/시간 업데이트
            self.last_positions[human_id] = (x, y)
            self.last_update_time[human_id] = timestamp

        # 3. 오래된 ID 제거
        self._cleanup_old_ids(timestamp)

    def _associate_detections(self, detections: List[Tuple[float, float]]) -> List[int]:
        """
        검출 결과와 기존 ID 매칭 (거리 기반).

        Args:
            detections: [(x, y), ...] 검출 좌표

        Returns:
            각 검출에 할당된 ID 리스트
        """
        if not detections:
            return []

        matched_ids = []
        used_ids = set()

        for x, y in detections:
            best_id = None
            best_dist = self.max_association_dist

            # 가장 가까운 기존 ID 찾기
            for human_id, (last_x, last_y) in self.last_positions.items():
                if human_id in used_ids:
                    continue

                dist = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best_id = human_id

            # 매칭 성공
            if best_id is not None:
                matched_ids.append(best_id)
                used_ids.add(best_id)
            else:
                # 새 ID 할당
                matched_ids.append(self.next_id)
                self.next_id += 1

        return matched_ids

    def _cleanup_old_ids(self, current_time: float):
        """오래된 ID 제거"""
        to_remove = []
        for human_id, last_time in self.last_update_time.items():
            if current_time - last_time > self.id_timeout:
                to_remove.append(human_id)

        for human_id in to_remove:
            del self.history_buffer[human_id]
            del self.last_positions[human_id]
            del self.last_update_time[human_id]

    def _compute_velocity(self, human_id: int) -> Tuple[float, float]:
        """
        History Buffer에서 속도 계산.

        최근 2개 포인트로 계산.
        """
        buffer = self.history_buffer.get(human_id)
        if buffer is None or len(buffer) < 2:
            return (0.0, 0.0)

        # 최근 2개 포인트
        p1 = buffer[-2]
        p2 = buffer[-1]

        dt = p2.timestamp - p1.timestamp
        if dt < 1e-6:
            return (0.0, 0.0)

        vx = (p2.x - p1.x) / dt
        vy = (p2.y - p1.y) / dt

        return (vx, vy)

    def get_human_states(self) -> List[HumanState]:
        """
        현재 HumanState 리스트 반환 (CIGP용).

        Returns:
            HumanState 리스트
        """
        states = []
        for human_id, buffer in self.history_buffer.items():
            if len(buffer) == 0:
                continue

            # 최신 위치
            latest = buffer[-1]

            # 속도 계산
            vx, vy = self._compute_velocity(human_id)

            states.append(HumanState(
                id=human_id,
                px=latest.x,
                py=latest.y,
                vx=vx,
                vy=vy
            ))

        return states

    def get_trajectory_tensor(self, min_history: int = None) -> Tuple[np.ndarray, List[int]]:
        """
        SingularTrajectory 모델 입력용 텐서 반환.

        Args:
            min_history: 최소 히스토리 길이 (None이면 history_length)

        Returns:
            (tensor, human_ids) 튜플
            - tensor: shape (N, history_length, 2)
            - human_ids: 각 행에 해당하는 사람 ID
        """
        if min_history is None:
            min_history = self.history_length

        trajectories = []
        human_ids = []

        for human_id, buffer in self.history_buffer.items():
            if len(buffer) < min_history:
                continue

            # 최근 history_length개 추출
            points = list(buffer)[-self.history_length:]

            # (x, y) 배열로 변환
            traj = np.array([[p.x, p.y] for p in points])

            # 길이가 부족하면 패딩
            if len(traj) < self.history_length:
                pad_length = self.history_length - len(traj)
                # 첫 좌표로 패딩
                padding = np.tile(traj[0], (pad_length, 1))
                traj = np.vstack([padding, traj])

            trajectories.append(traj)
            human_ids.append(human_id)

        if not trajectories:
            return np.array([]).reshape(0, self.history_length, 2), []

        tensor = np.stack(trajectories, axis=0)  # (N, 8, 2)
        return tensor, human_ids

    def get_ready_count(self) -> int:
        """History가 꽉 찬 사람 수 반환"""
        return sum(1 for buf in self.history_buffer.values()
                   if len(buf) >= self.history_length)

    def get_all_ids(self) -> List[int]:
        """현재 추적 중인 모든 ID 반환"""
        return list(self.history_buffer.keys())

    def get_history(self, human_id: int) -> List[TrajectoryPoint]:
        """특정 사람의 히스토리 반환"""
        buffer = self.history_buffer.get(human_id)
        return list(buffer) if buffer else []

    def reset(self):
        """모든 상태 초기화"""
        self.history_buffer.clear()
        self.last_positions.clear()
        self.last_update_time.clear()
        self.next_id = 0


if __name__ == '__main__':
    print("=== HumanStateExtractor Test ===")

    extractor = HumanStateExtractor(history_length=8, dt=0.4)

    # 시뮬레이션: 사람이 이동하는 상황
    # 사람 1: (0, 0) → (2, 0) 이동 (초속 0.5m)
    # 사람 2: (5, 5) → (5, 3) 이동 (초속 0.5m)

    for frame in range(10):
        t = frame * 0.4

        detections = [
            (0 + 0.2 * frame, 0),        # 사람 1
            (5, 5 - 0.2 * frame),        # 사람 2
        ]

        extractor.update(detections, timestamp=t)

        print(f"\nFrame {frame} (t={t:.1f}s):")
        states = extractor.get_human_states()
        for s in states:
            print(f"  ID {s.id}: pos=({s.px:.2f}, {s.py:.2f}), vel=({s.vx:.2f}, {s.vy:.2f})")

        # 텐서 상태
        ready = extractor.get_ready_count()
        print(f"  Ready for prediction: {ready}/{len(states)}")

    # 최종 텐서 출력
    tensor, ids = extractor.get_trajectory_tensor()
    print(f"\n=== Trajectory Tensor ===")
    print(f"Shape: {tensor.shape}")
    print(f"IDs: {ids}")
