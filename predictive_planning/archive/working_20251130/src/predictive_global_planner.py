#!/usr/bin/env python3
"""
Predictive Global Planner

시간 인식 A* 알고리즘 구현.
각 노드 평가 시 "로봇이 해당 노드에 도착할 시간"을 계산하고,
그 시간에 맞는 예측 비용을 적용.

핵심 차이점 (기존 A* vs 예측 A*):
- 기존: f(n) = g(n) + h(n)
- 예측: f(n) = g(n) + h(n) + w * social_cost(n, t_arrival)
"""

import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
import time

from .predicted_trajectory import PredictedTrajectoryArray
from .predictive_cost_calculator import PredictiveCostCalculator
from .config import PredictivePlanningConfig


@dataclass(order=True)
class AStarNode:
    """A* 노드"""
    f_cost: float
    g_cost: float = field(compare=False)
    h_cost: float = field(compare=False)
    social_cost: float = field(compare=False)
    x: float = field(compare=False)
    y: float = field(compare=False)
    gx: int = field(compare=False)
    gy: int = field(compare=False)
    parent: Optional['AStarNode'] = field(compare=False, default=None)


@dataclass
class PlanningResult:
    """경로 계획 결과"""
    success: bool
    path: List[Tuple[float, float]]
    path_length: float
    total_cost: float
    planning_time: float
    iterations: int
    nodes_expanded: int


class PredictiveGlobalPlanner:
    """
    예측 기반 글로벌 경로 계획기

    Args:
        config: 설정 객체
        cost_calculator: 비용 계산기 (None이면 자동 생성)
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None,
                 cost_calculator: Optional[PredictiveCostCalculator] = None):
        self.config = config or PredictivePlanningConfig()
        self.cost_calculator = cost_calculator or PredictiveCostCalculator(self.config)

        # 그리드 초기화
        self._init_grid()

        # 8방향 이동 (대각선 포함)
        self._directions = [
            (1, 0, 1.0),    # 오른쪽
            (-1, 0, 1.0),   # 왼쪽
            (0, 1, 1.0),    # 위
            (0, -1, 1.0),   # 아래
            (1, 1, 1.414),  # 대각선
            (1, -1, 1.414),
            (-1, 1, 1.414),
            (-1, -1, 1.414),
        ]

    def _init_grid(self):
        """장애물 그리드 초기화"""
        width = self.config.grid_width
        height = self.config.grid_height

        self._obstacle_grid = np.zeros((height, width), dtype=bool)

        # 장애물 마킹
        for gy in range(height):
            for gx in range(width):
                wx, wy = self.config.grid_to_world(gx, gy)
                if self.config.is_in_obstacle(wx, wy, margin=self.config.robot_radius):
                    self._obstacle_grid[gy, gx] = True

    def update_predictions(self, predictions: PredictedTrajectoryArray):
        """예측 데이터 업데이트"""
        self.cost_calculator.update_predictions(predictions)

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> PlanningResult:
        """
        경로 계획 수행

        Args:
            start: 시작 위치 (x, y)
            goal: 목표 위치 (x, y)

        Returns:
            PlanningResult
        """
        start_time = time.time()

        # 로봇 상태 업데이트
        self.cost_calculator.update_robot_state(start[0], start[1])

        # 그리드 좌표 변환
        start_gx, start_gy = self.config.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.config.world_to_grid(goal[0], goal[1])

        # 유효성 검사
        if not self._is_valid_grid(start_gx, start_gy):
            return PlanningResult(
                success=False, path=[], path_length=0,
                total_cost=float('inf'), planning_time=0, iterations=0, nodes_expanded=0
            )

        if not self._is_valid_grid(goal_gx, goal_gy):
            return PlanningResult(
                success=False, path=[], path_length=0,
                total_cost=float('inf'), planning_time=0, iterations=0, nodes_expanded=0
            )

        # A* 탐색
        result = self._astar_search(start_gx, start_gy, goal_gx, goal_gy)

        result.planning_time = time.time() - start_time

        return result

    def _astar_search(self, start_gx: int, start_gy: int,
                      goal_gx: int, goal_gy: int) -> PlanningResult:
        """A* 탐색 알고리즘"""
        # 시작 노드
        start_x, start_y = self.config.grid_to_world(start_gx, start_gy)
        h_cost = self._heuristic(start_gx, start_gy, goal_gx, goal_gy)

        start_node = AStarNode(
            f_cost=h_cost,
            g_cost=0.0,
            h_cost=h_cost,
            social_cost=0.0,
            x=start_x,
            y=start_y,
            gx=start_gx,
            gy=start_gy,
            parent=None
        )

        # Open/Closed 리스트
        open_heap: List[AStarNode] = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        g_costs: Dict[Tuple[int, int], float] = {(start_gx, start_gy): 0.0}

        iterations = 0
        nodes_expanded = 0

        while open_heap and iterations < self.config.max_iterations:
            iterations += 1

            # 최소 f_cost 노드 추출
            current = heapq.heappop(open_heap)

            # 이미 방문한 노드면 스킵
            if (current.gx, current.gy) in closed_set:
                continue

            closed_set.add((current.gx, current.gy))
            nodes_expanded += 1

            # 목표 도달 체크
            if current.gx == goal_gx and current.gy == goal_gy:
                path = self._reconstruct_path(current)
                return PlanningResult(
                    success=True,
                    path=path,
                    path_length=current.g_cost,
                    total_cost=current.f_cost,
                    planning_time=0,
                    iterations=iterations,
                    nodes_expanded=nodes_expanded
                )

            # 이웃 노드 탐색
            for dx, dy, move_cost in self._directions:
                next_gx = current.gx + dx
                next_gy = current.gy + dy

                # 유효성 검사
                if not self._is_valid_grid(next_gx, next_gy):
                    continue

                if (next_gx, next_gy) in closed_set:
                    continue

                # 월드 좌표
                next_x, next_y = self.config.grid_to_world(next_gx, next_gy)

                # g_cost 계산
                step_distance = move_cost * self.config.resolution
                tentative_g = current.g_cost + step_distance

                # 기존 g_cost보다 크면 스킵
                if (next_gx, next_gy) in g_costs:
                    if tentative_g >= g_costs[(next_gx, next_gy)]:
                        continue

                g_costs[(next_gx, next_gy)] = tentative_g

                # 소셜 비용 계산 (핵심: 도착 시간 기반)
                social_cost = self.cost_calculator.calculate_cost(
                    next_x, next_y, tentative_g
                )

                # h_cost
                h_cost = self._heuristic(next_gx, next_gy, goal_gx, goal_gy)

                # f_cost = g + h + weighted_social
                f_cost = (tentative_g +
                         self.config.heuristic_weight * h_cost +
                         self.config.social_cost_weight * social_cost)

                next_node = AStarNode(
                    f_cost=f_cost,
                    g_cost=tentative_g,
                    h_cost=h_cost,
                    social_cost=social_cost,
                    x=next_x,
                    y=next_y,
                    gx=next_gx,
                    gy=next_gy,
                    parent=current
                )

                heapq.heappush(open_heap, next_node)

        # 경로 찾기 실패
        return PlanningResult(
            success=False,
            path=[],
            path_length=0,
            total_cost=float('inf'),
            planning_time=0,
            iterations=iterations,
            nodes_expanded=nodes_expanded
        )

    def _heuristic(self, gx1: int, gy1: int, gx2: int, gy2: int) -> float:
        """휴리스틱 함수 (유클리드 거리)"""
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        return self.config.resolution * np.sqrt(dx**2 + dy**2)

    def _is_valid_grid(self, gx: int, gy: int) -> bool:
        """그리드 좌표 유효성 검사"""
        if gx < 0 or gx >= self.config.grid_width:
            return False
        if gy < 0 or gy >= self.config.grid_height:
            return False
        if self._obstacle_grid[gy, gx]:
            return False
        return True

    def _reconstruct_path(self, end_node: AStarNode) -> List[Tuple[float, float]]:
        """경로 재구성"""
        path = []
        current = end_node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        path.reverse()
        return path

    def get_next_waypoint(self, path: List[Tuple[float, float]],
                          current_pos: Tuple[float, float],
                          lookahead: float = 0.5) -> Optional[Tuple[float, float]]:
        """
        현재 위치에서 lookahead 거리의 웨이포인트 반환

        Args:
            path: 경로 웨이포인트 리스트
            current_pos: 현재 위치
            lookahead: 전방 주시 거리

        Returns:
            다음 웨이포인트 또는 None
        """
        if not path:
            return None

        # 경로상에서 현재 위치와 가장 가까운 점 찾기
        min_dist = float('inf')
        closest_idx = 0

        for i, wp in enumerate(path):
            dist = np.sqrt((wp[0] - current_pos[0])**2 +
                          (wp[1] - current_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # lookahead 거리만큼 전진
        accumulated_dist = 0.0

        for i in range(closest_idx, len(path) - 1):
            segment_dist = np.sqrt(
                (path[i+1][0] - path[i][0])**2 +
                (path[i+1][1] - path[i][1])**2
            )
            accumulated_dist += segment_dist

            if accumulated_dist >= lookahead:
                return path[i+1]

        # lookahead 이내에 목표 있으면 목표 반환
        return path[-1]


class SimpleGlobalPlanner:
    """
    단순 A* (비교용)

    예측 비용 없이 순수 거리 기반 A*.
    """

    def __init__(self, config: Optional[PredictivePlanningConfig] = None):
        self.config = config or PredictivePlanningConfig()
        self._init_grid()

        self._directions = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414),
        ]

    def _init_grid(self):
        width = self.config.grid_width
        height = self.config.grid_height
        self._obstacle_grid = np.zeros((height, width), dtype=bool)

        for gy in range(height):
            for gx in range(width):
                wx, wy = self.config.grid_to_world(gx, gy)
                if self.config.is_in_obstacle(wx, wy, margin=self.config.robot_radius):
                    self._obstacle_grid[gy, gx] = True

    def plan(self, start: Tuple[float, float],
             goal: Tuple[float, float]) -> PlanningResult:
        """예측 없는 순수 A* 경로 계획"""
        start_time = time.time()

        start_gx, start_gy = self.config.world_to_grid(start[0], start[1])
        goal_gx, goal_gy = self.config.world_to_grid(goal[0], goal[1])

        if not self._is_valid(start_gx, start_gy) or not self._is_valid(goal_gx, goal_gy):
            return PlanningResult(False, [], 0, float('inf'), 0, 0, 0)

        # 간단한 A* 구현
        open_heap = [(0, start_gx, start_gy)]
        came_from = {}
        g_score = {(start_gx, start_gy): 0}
        iterations = 0

        while open_heap and iterations < self.config.max_iterations:
            iterations += 1
            _, cx, cy = heapq.heappop(open_heap)

            if cx == goal_gx and cy == goal_gy:
                path = self._reconstruct(came_from, (cx, cy))
                return PlanningResult(
                    True, path, g_score[(cx, cy)],
                    g_score[(cx, cy)], time.time() - start_time,
                    iterations, len(came_from)
                )

            for dx, dy, cost in self._directions:
                nx, ny = cx + dx, cy + dy
                if not self._is_valid(nx, ny):
                    continue

                ng = g_score[(cx, cy)] + cost * self.config.resolution
                if (nx, ny) not in g_score or ng < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = ng
                    f = ng + self._heuristic(nx, ny, goal_gx, goal_gy)
                    heapq.heappush(open_heap, (f, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

        return PlanningResult(False, [], 0, float('inf'),
                            time.time() - start_time, iterations, len(came_from))

    def _is_valid(self, gx, gy):
        if gx < 0 or gx >= self.config.grid_width:
            return False
        if gy < 0 or gy >= self.config.grid_height:
            return False
        return not self._obstacle_grid[gy, gx]

    def _heuristic(self, gx1, gy1, gx2, gy2):
        return self.config.resolution * np.sqrt((gx2-gx1)**2 + (gy2-gy1)**2)

    def _reconstruct(self, came_from, current):
        path = []
        while current in came_from:
            wx, wy = self.config.grid_to_world(current[0], current[1])
            path.append((wx, wy))
            current = came_from[current]
        wx, wy = self.config.grid_to_world(current[0], current[1])
        path.append((wx, wy))
        path.reverse()
        return path
