#!/usr/bin/env python3
"""
Data Logger and Analyzer

실험 결과를 저장하고 분석하는 모듈.
논문의 Table I, II 형식으로 결과를 정리.
"""

import os
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class AggregatedMetrics:
    """집계된 메트릭 (논문 Table 형식)"""

    method_name: str = ""
    scenario: str = ""
    num_episodes: int = 0

    # Navigation Quality
    success_rate: float = 0.0  # SR (%)
    avg_velocity: float = 0.0  # Vavg (m/s)
    avg_angular_velocity: float = 0.0  # ωavg (rad/s)

    # Social Awareness
    intrusion_time_ratio: float = 0.0  # ITR
    social_distance: float = 0.0  # SD (m)

    # Additional
    avg_navigation_time: float = 0.0  # (s)
    collision_rate: float = 0.0  # (%)
    timeout_rate: float = 0.0  # (%)


class DataLogger:
    """데이터 로거"""

    def __init__(self, results_dir: str):
        """
        Args:
            results_dir: 결과 저장 디렉토리
        """
        self.results_dir = results_dir
        self._ensure_dirs()

        # 에피소드 결과 저장소
        self.episode_results: List[Dict] = []

        # 메타데이터
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "results_dir": results_dir
        }

    def _ensure_dirs(self):
        """디렉토리 생성"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "episodes"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "trajectories"), exist_ok=True)
        os.makedirs(os.path.join(self.results_dir, "analysis"), exist_ok=True)

    def log_episode(self, episode_metrics: Dict, trajectory: Optional[List[Dict]] = None):
        """
        에피소드 결과 저장

        Args:
            episode_metrics: 에피소드 메트릭 딕셔너리
            trajectory: 궤적 데이터 (선택)
        """
        self.episode_results.append(episode_metrics)

        # 개별 에피소드 JSON 저장
        episode_id = episode_metrics.get('episode_id', len(self.episode_results))
        method_name = episode_metrics.get('method_name', 'unknown')
        scenario = episode_metrics.get('scenario', 'unknown').replace('.xml', '')

        filename = f"ep_{episode_id:04d}_{method_name}_{scenario}.json"
        filepath = os.path.join(self.results_dir, "episodes", filename)

        with open(filepath, 'w') as f:
            json.dump(episode_metrics, f, indent=2, default=str)

        # 궤적 저장 (선택)
        if trajectory:
            traj_filename = f"traj_{episode_id:04d}_{method_name}_{scenario}.json"
            traj_filepath = os.path.join(self.results_dir, "trajectories", traj_filename)
            with open(traj_filepath, 'w') as f:
                json.dump(trajectory, f, default=str)

    def save_all_episodes(self):
        """모든 에피소드 결과 저장"""
        filepath = os.path.join(self.results_dir, "all_episodes.json")
        with open(filepath, 'w') as f:
            json.dump(self.episode_results, f, indent=2, default=str)

        # CSV로도 저장
        if self.episode_results:
            csv_filepath = os.path.join(self.results_dir, "all_episodes.csv")
            keys = self.episode_results[0].keys()
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.episode_results)

    def get_aggregated_metrics(self, method_name: str, scenario: str) -> AggregatedMetrics:
        """특정 조건의 집계 메트릭 계산"""
        filtered = [
            ep for ep in self.episode_results
            if ep.get('method_name') == method_name and ep.get('scenario') == scenario
        ]

        if not filtered:
            return AggregatedMetrics(method_name=method_name, scenario=scenario)

        num_episodes = len(filtered)
        successes = [ep for ep in filtered if ep.get('success', False)]
        collisions = [ep for ep in filtered if ep.get('collision', False)]
        timeouts = [ep for ep in filtered if ep.get('timeout', False)]

        return AggregatedMetrics(
            method_name=method_name,
            scenario=scenario,
            num_episodes=num_episodes,

            # Navigation Quality
            success_rate=len(successes) / num_episodes * 100,
            avg_velocity=np.mean([ep.get('avg_velocity', 0) for ep in filtered]),
            avg_angular_velocity=np.mean([ep.get('avg_angular_velocity', 0) for ep in filtered]),

            # Social Awareness
            intrusion_time_ratio=np.mean([ep.get('intrusion_time_ratio', 0) for ep in filtered]),
            social_distance=np.mean([ep.get('avg_social_distance', 0) for ep in filtered]),

            # Additional
            avg_navigation_time=np.mean([ep.get('duration', 0) for ep in successes]) if successes else 0,
            collision_rate=len(collisions) / num_episodes * 100,
            timeout_rate=len(timeouts) / num_episodes * 100
        )


class ResultsAnalyzer:
    """결과 분석기"""

    def __init__(self, logger: DataLogger):
        self.logger = logger

    def generate_summary_table(self) -> str:
        """논문 Table I, II 형식의 요약 테이블 생성"""
        # 모든 조건 추출
        conditions = set()
        for ep in self.logger.episode_results:
            method = ep.get('method_name', '')
            scenario = ep.get('scenario', '')
            conditions.add((method, scenario))

        # 시나리오별로 그룹화
        scenarios = sorted(set(c[1] for c in conditions))
        methods = sorted(set(c[0] for c in conditions))

        lines = []
        lines.append("=" * 100)
        lines.append("EXPERIMENT RESULTS SUMMARY")
        lines.append("=" * 100)
        lines.append("")

        for scenario in scenarios:
            scenario_name = scenario.replace('.xml', '').replace('_', ' ').title()
            lines.append(f"\n{'='*80}")
            lines.append(f"Scenario: {scenario_name}")
            lines.append(f"{'='*80}")
            lines.append("")

            # 헤더
            header = f"{'Method':<20} {'SR(%)':<10} {'Vavg':<10} {'ωavg':<10} {'ITR':<10} {'SD(m)':<10}"
            lines.append(header)
            lines.append("-" * 80)

            for method in methods:
                metrics = self.logger.get_aggregated_metrics(method, scenario)
                if metrics.num_episodes == 0:
                    continue

                row = (
                    f"{method:<20} "
                    f"{metrics.success_rate:<10.1f} "
                    f"{metrics.avg_velocity:<10.2f} "
                    f"{metrics.avg_angular_velocity:<10.2f} "
                    f"{metrics.intrusion_time_ratio:<10.2f} "
                    f"{metrics.social_distance:<10.2f}"
                )
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def generate_comparison_table(self) -> Dict[str, Any]:
        """Local Only vs CIGP vs Predictive 비교 테이블 생성 (3가지 조건)"""
        comparisons = {}

        # 모든 조건 추출
        planners = set()
        scenarios = set()
        for ep in self.logger.episode_results:
            planners.add(ep.get('planner', ''))
            scenarios.add(ep.get('scenario', ''))

        for planner in planners:
            for scenario in scenarios:
                local_only = f"{planner.upper()}"
                with_cigp = f"CIGP-{planner.upper()}"
                with_pred = f"PRED-{planner.upper()}"

                local_metrics = self.logger.get_aggregated_metrics(local_only, scenario)
                cigp_metrics = self.logger.get_aggregated_metrics(with_cigp, scenario)
                pred_metrics = self.logger.get_aggregated_metrics(with_pred, scenario)

                if (local_metrics.num_episodes == 0 and
                    cigp_metrics.num_episodes == 0 and
                    pred_metrics.num_episodes == 0):
                    continue

                key = f"{planner}_{scenario}"
                comparisons[key] = {
                    "planner": planner,
                    "scenario": scenario,
                    "local_only": asdict(local_metrics),
                    "with_cigp": asdict(cigp_metrics),
                    "with_predictive": asdict(pred_metrics),
                    "cigp_improvement": {
                        "success_rate": cigp_metrics.success_rate - local_metrics.success_rate,
                        "avg_velocity": cigp_metrics.avg_velocity - local_metrics.avg_velocity,
                        "social_distance": cigp_metrics.social_distance - local_metrics.social_distance,
                        "intrusion_time_ratio": local_metrics.intrusion_time_ratio - cigp_metrics.intrusion_time_ratio
                    },
                    "pred_improvement": {
                        "success_rate": pred_metrics.success_rate - local_metrics.success_rate,
                        "avg_velocity": pred_metrics.avg_velocity - local_metrics.avg_velocity,
                        "social_distance": pred_metrics.social_distance - local_metrics.social_distance,
                        "intrusion_time_ratio": local_metrics.intrusion_time_ratio - pred_metrics.intrusion_time_ratio
                    }
                }

        return comparisons

    def generate_paper_table(self) -> str:
        """논문 Table I, II 형식의 LaTeX/Markdown 테이블 생성"""
        lines = []
        lines.append("# Navigation Results (Paper Table Format)")
        lines.append("")

        # 모든 조건 추출
        scenarios = sorted(set(ep.get('scenario', '') for ep in self.logger.episode_results))
        methods = sorted(set(ep.get('method_name', '') for ep in self.logger.episode_results))

        for scenario in scenarios:
            scenario_name = scenario.replace('.xml', '').replace('_', ' ').title()
            lines.append(f"## {scenario_name}")
            lines.append("")
            lines.append("| Method | SR(%)↑ | Vavg(m/s)↑ | ωavg(rad/s)↓ | ITR↓ | SD(m)↑ |")
            lines.append("|--------|--------|------------|--------------|------|--------|")

            for method in methods:
                metrics = self.logger.get_aggregated_metrics(method, scenario)
                if metrics.num_episodes == 0:
                    continue

                # 최고 성능 표시를 위해 볼드 처리 가능
                row = (
                    f"| {method} | "
                    f"{metrics.success_rate:.1f} | "
                    f"{metrics.avg_velocity:.2f} | "
                    f"{metrics.avg_angular_velocity:.2f} | "
                    f"{metrics.intrusion_time_ratio:.2f} | "
                    f"{metrics.social_distance:.2f} |"
                )
                lines.append(row)

            lines.append("")

        return "\n".join(lines)

    def save_analysis(self, generate_images: bool = True):
        """분석 결과 저장"""
        analysis_dir = os.path.join(self.logger.results_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # 요약 테이블 저장
        summary = self.generate_summary_table()
        with open(os.path.join(analysis_dir, "summary.txt"), 'w') as f:
            f.write(summary)

        # 논문 형식 테이블 저장
        paper_table = self.generate_paper_table()
        with open(os.path.join(analysis_dir, "paper_table.md"), 'w') as f:
            f.write(paper_table)

        # 비교 결과 저장
        comparisons = self.generate_comparison_table()
        with open(os.path.join(analysis_dir, "comparisons.json"), 'w') as f:
            json.dump(comparisons, f, indent=2)

        # 집계 메트릭 저장
        all_metrics = []
        conditions = set()
        for ep in self.logger.episode_results:
            method = ep.get('method_name', '')
            scenario = ep.get('scenario', '')
            conditions.add((method, scenario))

        for method, scenario in conditions:
            metrics = self.logger.get_aggregated_metrics(method, scenario)
            all_metrics.append(asdict(metrics))

        with open(os.path.join(analysis_dir, "aggregated_metrics.json"), 'w') as f:
            json.dump(all_metrics, f, indent=2)

        # CSV로도 저장
        if all_metrics:
            csv_filepath = os.path.join(analysis_dir, "aggregated_metrics.csv")
            keys = all_metrics[0].keys()
            with open(csv_filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(all_metrics)

        # 이미지 생성
        if generate_images:
            try:
                from visualizer import TrajectoryVisualizer
                viz = TrajectoryVisualizer()
                viz.generate_all_images(self.logger.results_dir)
            except ImportError:
                print("Warning: visualizer module not found, skipping image generation")
            except Exception as e:
                print(f"Warning: image generation failed: {e}")

        print(f"Analysis saved to: {analysis_dir}")
        print(summary)


def load_results(results_dir: str) -> DataLogger:
    """기존 결과 로드"""
    logger = DataLogger(results_dir)

    # all_episodes.json 로드
    all_episodes_path = os.path.join(results_dir, "all_episodes.json")
    if os.path.exists(all_episodes_path):
        with open(all_episodes_path, 'r') as f:
            logger.episode_results = json.load(f)

    return logger


if __name__ == "__main__":
    # 테스트
    import tempfile

    # 임시 디렉토리에 테스트
    with tempfile.TemporaryDirectory() as tmpdir:
        logger = DataLogger(tmpdir)

        # 테스트 데이터 추가
        for i in range(10):
            logger.log_episode({
                "episode_id": i,
                "method_name": "DWA" if i < 5 else "CIGP-DWA",
                "scenario": "warehouse_pedsim.xml",
                "planner": "dwa",
                "use_cigp": i >= 5,
                "success": i % 3 != 0,
                "collision": i % 3 == 0,
                "timeout": False,
                "duration": 30.0 + i,
                "avg_velocity": 0.5 + i * 0.01,
                "avg_angular_velocity": 0.3 + i * 0.01,
                "intrusion_time_ratio": 0.1 - i * 0.01,
                "avg_social_distance": 1.0 + i * 0.1
            })

        logger.save_all_episodes()

        analyzer = ResultsAnalyzer(logger)
        analyzer.save_analysis()
