#!/usr/bin/env python3
"""
Experiment Configuration

3가지 조건 비교 실험:
1. Local Planner Only (Baseline)
2. CIGP + Local Planner (CCTV-Informed Global Planner)
3. Predictive Planning + Local Planner (Time-aware A*)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import os


class GlobalPlannerType(Enum):
    """글로벌 플래너 타입"""
    NONE = "none"           # Local Planner Only (Baseline)
    CIGP = "cigp"           # CCTV-Informed Global Planner
    PREDICTIVE = "predictive"  # Predictive Planning (Time-aware A*)


@dataclass
class ExperimentConfig:
    """실험 설정"""

    # 실험 식별
    experiment_name: str = "local_vs_cigp_vs_predictive"
    experiment_id: str = ""  # 자동 생성

    # 실험 조건 (5가지 로컬 플래너)
    planners: List[str] = field(default_factory=lambda: ["dwa", "drl_vo", "teb", "sfm"])

    # 글로벌 플래너 옵션 (3가지)
    global_planners: List[str] = field(default_factory=lambda: [
        "none",        # Local Only (Baseline)
        "cigp",        # CIGP
        "predictive"   # Predictive Planning
    ])

    # 이전 버전 호환용
    use_cigp_options: List[bool] = field(default_factory=lambda: [False, True])

    scenarios: List[str] = field(default_factory=lambda: [
        "warehouse_pedsim.xml",        # 기본 (8명)
        "scenario_block_heavy.xml",    # 혼잡 (15명+)
        "scenario_congestion_all.xml"  # 매우 혼잡 (20명+)
    ])

    # 반복 횟수
    num_episodes: int = 100  # 논문: 100 episodes per condition
    max_timesteps: int = 500  # 논문: 500 timesteps max

    # 로봇 설정
    robot_radius: float = 0.25
    robot_max_speed: float = 0.8  # m/s
    robot_max_yaw_rate: float = 1.0  # rad/s

    # 목표 설정
    goal_threshold: float = 0.3  # 도착 판정 거리 (논문: 0.3m)

    # 타임아웃
    episode_timeout: float = 120.0  # seconds

    # 경로 설정
    base_dir: str = "/home/pyongjoo/Desktop/newstart/environment/experiments"
    results_dir: str = ""  # 자동 생성

    def __post_init__(self):
        """초기화 후 처리"""
        import datetime
        if not self.experiment_id:
            self.experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.results_dir = os.path.join(
            self.base_dir,
            "results",
            f"{self.experiment_name}_{self.experiment_id}"
        )


@dataclass
class MetricsConfig:
    """평가 메트릭 설정 (논문 기준)"""

    # Navigation Quality Metrics
    # SR: Success Rate - 성공률
    # Vavg: Average Velocity - 평균 속도
    # ωavg: Heading Change Smoothness - 방향 변화 부드러움

    # Social Awareness Metrics
    # ITR: Intrusion Time Ratio - 개인 공간 침범 비율
    # SD: Social Distance - 평균 사회적 거리

    # 개인 공간 (Individual Space) 파라미터
    personal_space_radius: float = 0.5  # 기본 개인 공간 반경
    intimate_space_radius: float = 0.3  # 친밀 공간 반경

    # 충돌 판정
    collision_threshold: float = 0.35  # 로봇 반경 + 사람 반경

    # 샘플링 주기
    sample_rate: float = 10.0  # Hz


@dataclass
class ScenarioConfig:
    """시나리오별 설정"""

    # 시나리오 이름 -> 설명
    scenario_info: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "warehouse_pedsim.xml": {
            "name": "baseline",
            "description": "Basic warehouse (8 pedestrians)",
            "num_pedestrians": 8,
            "difficulty": "easy"
        },
        "scenario_block_heavy.xml": {
            "name": "congestion",
            "description": "Heavy blocking (15+ pedestrians)",
            "num_pedestrians": 15,
            "difficulty": "medium"
        },
        "scenario_congestion_all.xml": {
            "name": "circulation",
            "description": "Everywhere congestion (20+ pedestrians)",
            "num_pedestrians": 20,
            "difficulty": "hard"
        }
    })

    # 위쪽 위치 (y=10)
    top_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-10.0, 10.0),
        (-5.0, 10.0),
        (0.0, 10.0),
        (5.0, 10.0),
        (10.0, 10.0),
    ])

    # 아래쪽 위치 (y=-10)
    bottom_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-10.0, -10.0),
        (-5.0, -10.0),
        (0.0, -10.0),
        (5.0, -10.0),
        (10.0, -10.0),
    ])

    # 테스트 방향
    # "bottom_to_top": 아래 → 위
    # "top_to_bottom": 위 → 아래
    test_directions: List[str] = field(default_factory=lambda: [
        "bottom_to_top",
        "top_to_bottom"
    ])

    # 시작/목표 위치 (이전 버전 호환)
    spawn_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-10.0, -10.0), (-5.0, -10.0), (0.0, -10.0), (5.0, -10.0), (10.0, -10.0),  # 아래쪽
        (-10.0, 10.0), (-5.0, 10.0), (0.0, 10.0), (5.0, 10.0), (10.0, 10.0),       # 위쪽
    ])

    goal_positions: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-10.0, 10.0), (-5.0, 10.0), (0.0, 10.0), (5.0, 10.0), (10.0, 10.0),       # 위쪽
        (-10.0, -10.0), (-5.0, -10.0), (0.0, -10.0), (5.0, -10.0), (10.0, -10.0),  # 아래쪽
    ])

    def get_test_pairs(self, direction: str) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        테스트 시작/목표 쌍 생성

        Args:
            direction: "bottom_to_top" 또는 "top_to_bottom"

        Returns:
            [(start, goal), ...] 리스트
        """
        pairs = []
        if direction == "bottom_to_top":
            # 아래쪽 출발 → 위쪽 도착 (같은 x좌표)
            for i in range(5):
                start = self.bottom_positions[i]
                goal = self.top_positions[i]
                pairs.append((start, goal))
        elif direction == "top_to_bottom":
            # 위쪽 출발 → 아래쪽 도착 (같은 x좌표)
            for i in range(5):
                start = self.top_positions[i]
                goal = self.bottom_positions[i]
                pairs.append((start, goal))
        return pairs

    def get_all_test_pairs(self) -> List[Dict[str, Any]]:
        """모든 테스트 쌍 생성"""
        all_pairs = []
        for direction in self.test_directions:
            pairs = self.get_test_pairs(direction)
            for i, (start, goal) in enumerate(pairs):
                all_pairs.append({
                    "direction": direction,
                    "pair_id": i,
                    "start": start,
                    "goal": goal,
                    "name": f"{direction}_x{int(start[0])}"
                })
        return all_pairs


# 기본 설정 인스턴스
DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()
DEFAULT_METRICS_CONFIG = MetricsConfig()
DEFAULT_SCENARIO_CONFIG = ScenarioConfig()


def get_method_name(planner: str, global_planner: str) -> str:
    """
    방법 이름 생성

    Args:
        planner: 로컬 플래너 (dwa, drl_vo, etc.)
        global_planner: 글로벌 플래너 (none, cigp, predictive)

    Returns:
        방법 이름 (예: "DWA", "CIGP-DWA", "PRED-DWA")
    """
    planner_upper = planner.upper()

    if global_planner == "none":
        return planner_upper
    elif global_planner == "cigp":
        return f"CIGP-{planner_upper}"
    elif global_planner == "predictive":
        return f"PRED-{planner_upper}"
    else:
        return f"{global_planner.upper()}-{planner_upper}"


def get_method_name_legacy(planner: str, use_cigp: bool) -> str:
    """방법 이름 생성 (이전 버전 호환)"""
    global_planner = "cigp" if use_cigp else "none"
    return get_method_name(planner, global_planner)


def get_all_conditions(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """모든 실험 조건 조합 생성 (3가지 글로벌 플래너)"""
    conditions = []

    for planner in config.planners:
        for global_planner in config.global_planners:
            for scenario in config.scenarios:
                conditions.append({
                    "planner": planner,
                    "global_planner": global_planner,
                    "scenario": scenario,
                    "method_name": get_method_name(planner, global_planner),
                    # 이전 버전 호환
                    "use_cigp": global_planner == "cigp",
                    "use_predictive": global_planner == "predictive"
                })

    return conditions


def get_all_conditions_legacy(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """모든 실험 조건 조합 생성 (이전 버전 - CIGP만)"""
    conditions = []

    for planner in config.planners:
        for use_cigp in config.use_cigp_options:
            for scenario in config.scenarios:
                conditions.append({
                    "planner": planner,
                    "use_cigp": use_cigp,
                    "global_planner": "cigp" if use_cigp else "none",
                    "scenario": scenario,
                    "method_name": get_method_name_legacy(planner, use_cigp)
                })

    return conditions


if __name__ == "__main__":
    # 설정 테스트
    config = ExperimentConfig()
    print(f"Experiment: {config.experiment_name}")
    print(f"ID: {config.experiment_id}")
    print(f"Results dir: {config.results_dir}")
    print()

    conditions = get_all_conditions(config)
    print(f"Total conditions: {len(conditions)}")
    print()

    # 글로벌 플래너별로 그룹화해서 출력
    for gp in config.global_planners:
        print(f"[{gp.upper()}]")
        for cond in conditions:
            if cond['global_planner'] == gp:
                print(f"  - {cond['method_name']} / {cond['scenario']}")
        print()
