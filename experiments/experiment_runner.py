#!/usr/bin/env python3
"""
Experiment Runner

Docker 컨테이너 내에서 실행되는 실험 러너.
3가지 조건 비교:
  1. Local Planner Only (Baseline)
  2. CIGP + Local Planner
  3. Predictive Planning + Local Planner

기존 환경 코드 수정 없이 독립적으로 동작.
"""

import sys
import os
import time
import json
import random
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import asdict

# 실험 모듈 경로 추가
EXPERIMENT_PATH = '/environment/experiments'
if EXPERIMENT_PATH not in sys.path:
    sys.path.insert(0, EXPERIMENT_PATH)

from config import ExperimentConfig, MetricsConfig, ScenarioConfig, get_method_name
from metrics_collector import MetricsCollector, EpisodeMetrics
from data_logger import DataLogger, ResultsAnalyzer


class ExperimentRunner:
    """실험 러너 (ROS 노드)"""

    def __init__(self, config: ExperimentConfig):
        import rospy
        from geometry_msgs.msg import Twist, PoseStamped, PointStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Empty, String
        from pedsim_msgs.msg import AgentStates
        from tf.transformations import euler_from_quaternion
        from std_srvs.srv import Empty as EmptySrv

        self.rospy = rospy
        self.config = config
        self.euler_from_quaternion = euler_from_quaternion

        # ROS 초기화
        rospy.init_node('experiment_runner', anonymous=True)

        # 토픽 파라미터
        self.odom_topic = rospy.get_param('~odom_topic', '/p3dx/odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/p3dx/cmd_vel')
        self.agents_topic = rospy.get_param('~agents_topic', '/pedsim_simulator/simulated_agents')

        # 메트릭 수집기
        metrics_config = MetricsConfig()
        self.metrics_collector = MetricsCollector(
            goal_threshold=config.goal_threshold,
            collision_threshold=metrics_config.collision_threshold,
            personal_space_radius=metrics_config.personal_space_radius,
            robot_radius=config.robot_radius
        )

        # 데이터 로거
        self.logger = DataLogger(config.results_dir)

        # 시나리오 설정
        self.scenario_config = ScenarioConfig()

        # 상태
        self.robot_pos: Optional[Tuple[float, float]] = None
        self.robot_vel: Optional[Tuple[float, float]] = None
        self.robot_theta: float = 0.0
        self.robot_omega: float = 0.0
        self.goal_pos: Optional[Tuple[float, float]] = None
        self.humans: List[Dict] = []

        # Subscribers
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(self.agents_topic, AgentStates, self._agents_callback, queue_size=1)

        # Publishers
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.experiment_status_pub = rospy.Publisher('/experiment/status', String, queue_size=1)

        # 실험 상태
        self.current_condition: Optional[Dict] = None
        self.episode_count = 0

        rospy.loginfo("[ExperimentRunner] Initialized")
        rospy.loginfo(f"[ExperimentRunner] Results dir: {config.results_dir}")

    def _odom_callback(self, msg):
        """오도메트리 콜백"""
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.robot_omega = msg.twist.twist.angular.z

        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = self.euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def _agents_callback(self, msg):
        """보행자 콜백"""
        self.humans = []
        for agent in msg.agent_states:
            self.humans.append({
                'id': agent.id,
                'pos': [agent.pose.position.x, agent.pose.position.y],
                'vel': [agent.twist.linear.x, agent.twist.linear.y],
                'radius': 0.3
            })

    def _publish_goal(self, goal: Tuple[float, float]):
        """목표 발행"""
        from geometry_msgs.msg import PoseStamped

        msg = PoseStamped()
        msg.header.stamp = self.rospy.Time.now()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = goal[0]
        msg.pose.position.y = goal[1]
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        self.goal_pub.publish(msg)
        self.goal_pos = goal

    def _stop_robot(self):
        """로봇 정지"""
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def _wait_for_topics(self, timeout: float = 10.0) -> bool:
        """토픽 수신 대기"""
        start_time = time.time()
        rate = self.rospy.Rate(10)

        while time.time() - start_time < timeout:
            if self.robot_pos is not None:
                return True
            rate.sleep()

        return False

    def _select_start_goal(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """시작/목표 위치 선택"""
        spawns = self.scenario_config.spawn_positions
        goals = self.scenario_config.goal_positions

        start = random.choice(spawns)
        # 시작 위치와 다른 목표 선택
        available_goals = [g for g in goals if np.sqrt((g[0]-start[0])**2 + (g[1]-start[1])**2) > 5.0]
        if not available_goals:
            available_goals = goals

        goal = random.choice(available_goals)

        return start, goal

    def run_episode(self,
                    episode_id: int,
                    method_name: str,
                    scenario: str,
                    planner: str,
                    global_planner: str) -> EpisodeMetrics:
        """
        단일 에피소드 실행

        Args:
            episode_id: 에피소드 번호
            method_name: 방법 이름 (예: "CIGP-DWA", "PRED-DWA")
            scenario: 시나리오 파일명
            planner: 로컬 플래너 타입
            global_planner: 글로벌 플래너 타입 (none, cigp, predictive)

        Returns:
            EpisodeMetrics
        """
        self.rospy.loginfo(f"[ExperimentRunner] Starting episode {episode_id}: {method_name}")

        # 시작/목표 선택
        start_pos, goal_pos = self._select_start_goal()

        # 메트릭 수집 시작
        self.metrics_collector.start_episode(
            episode_id=episode_id,
            method_name=method_name,
            scenario=scenario,
            planner=planner,
            use_cigp=(global_planner == "cigp"),
            start_pos=start_pos,
            goal_pos=goal_pos
        )

        # 목표 발행
        time.sleep(0.5)  # 안정화 대기
        self._publish_goal(goal_pos)

        # 상태 발행
        status_msg = json.dumps({
            "status": "running",
            "episode_id": episode_id,
            "method_name": method_name,
            "global_planner": global_planner,
            "start_pos": start_pos,
            "goal_pos": goal_pos
        })
        self.experiment_status_pub.publish(status_msg)

        # 에피소드 루프
        rate = self.rospy.Rate(10)  # 10 Hz
        timesteps = 0

        while not self.rospy.is_shutdown():
            timesteps += 1

            # 데이터 수신 체크
            if self.robot_pos is None or self.robot_vel is None:
                rate.sleep()
                continue

            # 메트릭 업데이트
            self.metrics_collector.update(
                robot_pos=self.robot_pos,
                robot_vel=self.robot_vel,
                robot_omega=self.robot_omega,
                goal_pos=goal_pos,
                humans=self.humans
            )

            # 종료 조건 체크
            terminated, success, collision, timeout = self.metrics_collector.check_termination(
                robot_pos=self.robot_pos,
                goal_pos=goal_pos,
                max_timesteps=self.config.max_timesteps
            )

            if terminated:
                break

            rate.sleep()

        # 로봇 정지
        self._stop_robot()

        # 에피소드 종료
        episode_result = self.metrics_collector.end_episode(success, collision, timeout)

        self.rospy.loginfo(
            f"[ExperimentRunner] Episode {episode_id} finished: "
            f"success={success}, collision={collision}, timeout={timeout}, "
            f"duration={episode_result.duration:.1f}s"
        )

        return episode_result

    def run_condition(self,
                      planner: str,
                      global_planner: str,
                      scenario: str,
                      num_episodes: int) -> List[EpisodeMetrics]:
        """
        하나의 실험 조건 실행

        Args:
            planner: 로컬 플래너 타입
            global_planner: 글로벌 플래너 타입 (none, cigp, predictive)
            scenario: 시나리오 파일명
            num_episodes: 에피소드 수

        Returns:
            에피소드 결과 리스트
        """
        method_name = get_method_name(planner, global_planner)

        self.rospy.loginfo(f"\n{'='*60}")
        self.rospy.loginfo(f"Running condition: {method_name} - {scenario}")
        self.rospy.loginfo(f"Global Planner: {global_planner}")
        self.rospy.loginfo(f"Episodes: {num_episodes}")
        self.rospy.loginfo(f"{'='*60}\n")

        results = []

        for ep_id in range(num_episodes):
            # 에피소드 실행
            episode_result = self.run_episode(
                episode_id=self.episode_count,
                method_name=method_name,
                scenario=scenario,
                planner=planner,
                global_planner=global_planner
            )

            results.append(episode_result)

            # 로깅 (global_planner 정보 추가)
            episode_dict = asdict(episode_result)
            episode_dict['global_planner'] = global_planner
            self.logger.log_episode(
                episode_dict,
                self.metrics_collector.get_trajectory()
            )

            self.episode_count += 1

            # 에피소드 간 대기
            time.sleep(1.0)

        return results

    def run_all_experiments(self):
        """모든 실험 조건 실행 (3가지 글로벌 플래너)"""
        self.rospy.loginfo("\n" + "=" * 80)
        self.rospy.loginfo("STARTING ALL EXPERIMENTS (3 Global Planners)")
        self.rospy.loginfo("=" * 80 + "\n")

        # 토픽 대기
        if not self._wait_for_topics():
            self.rospy.logerr("[ExperimentRunner] Failed to receive topics")
            return

        # 모든 조건 실행
        for planner in self.config.planners:
            for global_planner in self.config.global_planners:
                for scenario in self.config.scenarios:
                    self.run_condition(
                        planner=planner,
                        global_planner=global_planner,
                        scenario=scenario,
                        num_episodes=self.config.num_episodes
                    )

        # 결과 저장
        self.logger.save_all_episodes()

        # 분석
        analyzer = ResultsAnalyzer(self.logger)
        analyzer.save_analysis()

        self.rospy.loginfo("\n" + "=" * 80)
        self.rospy.loginfo("ALL EXPERIMENTS COMPLETED")
        self.rospy.loginfo(f"Results saved to: {self.config.results_dir}")
        self.rospy.loginfo("=" * 80 + "\n")


class SingleConditionRunner:
    """단일 조건 실험 러너 (독립 실행용)"""

    def __init__(self,
                 planner: str,
                 global_planner: str,
                 scenario: str,
                 num_episodes: int = 100,
                 results_dir: str = "/environment/experiments/results"):
        import rospy

        self.rospy = rospy
        self.planner = planner
        self.global_planner = global_planner
        self.scenario = scenario
        self.num_episodes = num_episodes

        # 설정 생성
        config = ExperimentConfig(
            planners=[planner],
            global_planners=[global_planner],
            scenarios=[scenario],
            num_episodes=num_episodes
        )
        config.results_dir = results_dir

        # 러너 생성
        self.runner = ExperimentRunner(config)

    def run(self):
        """실험 실행"""
        # 토픽 대기
        if not self.runner._wait_for_topics():
            self.rospy.logerr("[SingleConditionRunner] Failed to receive topics")
            return

        # 조건 실행
        self.runner.run_condition(
            planner=self.planner,
            global_planner=self.global_planner,
            scenario=self.scenario,
            num_episodes=self.num_episodes
        )

        # 저장
        self.runner.logger.save_all_episodes()

        # 분석
        analyzer = ResultsAnalyzer(self.runner.logger)
        analyzer.save_analysis()


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Experiment Runner')
    parser.add_argument('--planner', type=str, default='dwa',
                        help='Local planner type (dwa, drl_vo, teb, orca, sfm)')
    parser.add_argument('--global-planner', type=str, default='none',
                        choices=['none', 'cigp', 'predictive'],
                        help='Global planner type (none, cigp, predictive)')
    # 이전 버전 호환
    parser.add_argument('--use-cigp', action='store_true',
                        help='Use CIGP global planner (legacy, use --global-planner instead)')
    parser.add_argument('--use-predictive', action='store_true',
                        help='Use Predictive Planning (legacy, use --global-planner instead)')
    parser.add_argument('--scenario', type=str, default='warehouse_pedsim.xml',
                        help='Scenario file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes')
    parser.add_argument('--results-dir', type=str,
                        default='/environment/experiments/results',
                        help='Results directory')
    parser.add_argument('--all', action='store_true',
                        help='Run all experiment conditions')

    args = parser.parse_args()

    # 이전 버전 호환 처리
    global_planner = args.global_planner
    if args.use_cigp:
        global_planner = 'cigp'
    elif args.use_predictive:
        global_planner = 'predictive'

    try:
        if args.all:
            # 모든 조건 실행
            config = ExperimentConfig(num_episodes=args.episodes)
            config.results_dir = args.results_dir
            runner = ExperimentRunner(config)
            runner.run_all_experiments()
        else:
            # 단일 조건 실행
            runner = SingleConditionRunner(
                planner=args.planner,
                global_planner=global_planner,
                scenario=args.scenario,
                num_episodes=args.episodes,
                results_dir=args.results_dir
            )
            runner.run()

    except Exception as e:
        print(f"[ExperimentRunner] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
