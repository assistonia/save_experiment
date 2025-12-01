#!/usr/bin/env python3
"""
Metrics Collector Node

논문의 평가 메트릭을 실시간으로 수집하는 ROS 노드.
기존 환경 코드 수정 없이 독립적으로 동작.

Metrics (논문 기준):
- SR (Success Rate): 성공률
- Vavg (Average Velocity): 평균 속도
- ωavg (Heading Change Smoothness): 방향 변화 부드러움
- ITR (Intrusion Time Ratio): 개인 공간 침범 비율
- SD (Social Distance): 평균 사회적 거리
- Navigation Time: 네비게이션 소요 시간
- Collision Count: 충돌 횟수

Subscribe:
    - /p3dx/odom: 로봇 위치/속도
    - /pedsim_simulator/simulated_agents: 보행자 상태
    - /move_base_simple/goal: 목표 위치

Publish:
    - /experiment/metrics: 실시간 메트릭
    - /experiment/episode_result: 에피소드 결과
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import time


@dataclass
class EpisodeMetrics:
    """에피소드별 메트릭"""

    # 식별 정보
    episode_id: int = 0
    method_name: str = ""
    scenario: str = ""
    planner: str = ""
    use_cigp: bool = False

    # 시간 정보
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0

    # 결과
    success: bool = False
    collision: bool = False
    timeout: bool = False

    # Navigation Quality
    total_distance: float = 0.0
    avg_velocity: float = 0.0  # Vavg
    avg_angular_velocity: float = 0.0  # ωavg (heading change smoothness)

    # Social Awareness
    intrusion_time_ratio: float = 0.0  # ITR
    avg_social_distance: float = 0.0  # SD
    min_human_distance: float = float('inf')

    # 상세
    collision_count: int = 0
    timesteps: int = 0

    # 시작/목표 위치
    start_pos: tuple = (0.0, 0.0)
    goal_pos: tuple = (0.0, 0.0)


@dataclass
class RealTimeMetrics:
    """실시간 메트릭 (매 timestep)"""

    timestamp: float = 0.0

    # 로봇 상태
    robot_x: float = 0.0
    robot_y: float = 0.0
    robot_theta: float = 0.0
    robot_vx: float = 0.0
    robot_vy: float = 0.0
    robot_omega: float = 0.0

    # 목표까지 거리
    distance_to_goal: float = 0.0

    # 가장 가까운 사람까지 거리
    min_human_distance: float = float('inf')

    # 개인 공간 침범 여부
    in_personal_space: bool = False


class MetricsCollector:
    """메트릭 수집기 (ROS 독립 버전)"""

    def __init__(self,
                 goal_threshold: float = 0.3,
                 collision_threshold: float = 0.35,
                 personal_space_radius: float = 0.5,
                 robot_radius: float = 0.25):
        """
        Args:
            goal_threshold: 도착 판정 거리 (m)
            collision_threshold: 충돌 판정 거리 (m)
            personal_space_radius: 개인 공간 반경 (m)
            robot_radius: 로봇 반경 (m)
        """
        self.goal_threshold = goal_threshold
        self.collision_threshold = collision_threshold
        self.personal_space_radius = personal_space_radius
        self.robot_radius = robot_radius

        # 에피소드 상태
        self.current_episode = EpisodeMetrics()
        self.episode_active = False

        # 누적 데이터
        self._velocities: List[float] = []
        self._angular_velocities: List[float] = []
        self._human_distances: List[float] = []
        self._intrusion_steps: int = 0
        self._total_steps: int = 0
        self._last_pos: Optional[tuple] = None

        # 궤적 기록
        self._trajectory: List[Dict] = []

    def start_episode(self,
                      episode_id: int,
                      method_name: str,
                      scenario: str,
                      planner: str,
                      use_cigp: bool,
                      start_pos: tuple,
                      goal_pos: tuple):
        """에피소드 시작"""
        self.current_episode = EpisodeMetrics(
            episode_id=episode_id,
            method_name=method_name,
            scenario=scenario,
            planner=planner,
            use_cigp=use_cigp,
            start_time=time.time(),
            start_pos=start_pos,
            goal_pos=goal_pos
        )

        # 누적 데이터 초기화
        self._velocities = []
        self._angular_velocities = []
        self._human_distances = []
        self._intrusion_steps = 0
        self._total_steps = 0
        self._last_pos = start_pos
        self._trajectory = []

        self.episode_active = True

    def update(self,
               robot_pos: tuple,
               robot_vel: tuple,
               robot_omega: float,
               goal_pos: tuple,
               humans: List[Dict]) -> RealTimeMetrics:
        """
        매 timestep 업데이트

        Args:
            robot_pos: (x, y) 로봇 위치
            robot_vel: (vx, vy) 로봇 속도
            robot_omega: 각속도
            goal_pos: (gx, gy) 목표 위치
            humans: [{'pos': [x, y], 'vel': [vx, vy], 'radius': r}, ...]

        Returns:
            RealTimeMetrics
        """
        if not self.episode_active:
            return RealTimeMetrics()

        self._total_steps += 1

        # 속도 계산
        speed = np.sqrt(robot_vel[0]**2 + robot_vel[1]**2)
        self._velocities.append(speed)
        self._angular_velocities.append(abs(robot_omega))

        # 이동 거리
        if self._last_pos is not None:
            dist = np.sqrt(
                (robot_pos[0] - self._last_pos[0])**2 +
                (robot_pos[1] - self._last_pos[1])**2
            )
            self.current_episode.total_distance += dist
        self._last_pos = robot_pos

        # 목표까지 거리
        distance_to_goal = np.sqrt(
            (robot_pos[0] - goal_pos[0])**2 +
            (robot_pos[1] - goal_pos[1])**2
        )

        # 사람들과의 거리
        min_human_dist = float('inf')
        in_personal_space = False

        for human in humans:
            h_pos = human['pos']
            h_radius = human.get('radius', 0.3)

            dist = np.sqrt(
                (robot_pos[0] - h_pos[0])**2 +
                (robot_pos[1] - h_pos[1])**2
            )
            # 실제 거리 = 중심 거리 - 반경들
            actual_dist = dist - self.robot_radius - h_radius

            min_human_dist = min(min_human_dist, actual_dist)

            # 개인 공간 침범 체크
            if actual_dist < self.personal_space_radius:
                in_personal_space = True

            # 충돌 체크
            if actual_dist < 0:
                self.current_episode.collision_count += 1

        if min_human_dist < float('inf'):
            self._human_distances.append(min_human_dist)

        if in_personal_space:
            self._intrusion_steps += 1

        # 최소 거리 업데이트
        if min_human_dist < self.current_episode.min_human_distance:
            self.current_episode.min_human_distance = min_human_dist

        # 궤적 기록
        self._trajectory.append({
            'timestamp': time.time(),
            'robot_pos': robot_pos,
            'robot_vel': robot_vel,
            'distance_to_goal': distance_to_goal,
            'min_human_distance': min_human_dist
        })

        return RealTimeMetrics(
            timestamp=time.time(),
            robot_x=robot_pos[0],
            robot_y=robot_pos[1],
            robot_vx=robot_vel[0],
            robot_vy=robot_vel[1],
            robot_omega=robot_omega,
            distance_to_goal=distance_to_goal,
            min_human_distance=min_human_dist,
            in_personal_space=in_personal_space
        )

    def check_termination(self,
                          robot_pos: tuple,
                          goal_pos: tuple,
                          max_timesteps: int) -> tuple:
        """
        종료 조건 체크

        Returns:
            (terminated, success, collision, timeout)
        """
        # 도착 체크
        distance_to_goal = np.sqrt(
            (robot_pos[0] - goal_pos[0])**2 +
            (robot_pos[1] - goal_pos[1])**2
        )
        if distance_to_goal < self.goal_threshold:
            return True, True, False, False

        # 충돌 체크
        if self.current_episode.collision_count > 0:
            return True, False, True, False

        # 타임아웃 체크
        if self._total_steps >= max_timesteps:
            return True, False, False, True

        return False, False, False, False

    def end_episode(self, success: bool, collision: bool, timeout: bool) -> EpisodeMetrics:
        """에피소드 종료 및 메트릭 계산"""
        self.current_episode.end_time = time.time()
        self.current_episode.duration = (
            self.current_episode.end_time - self.current_episode.start_time
        )
        self.current_episode.success = success
        self.current_episode.collision = collision
        self.current_episode.timeout = timeout
        self.current_episode.timesteps = self._total_steps

        # 평균 속도 (Vavg) = 총 거리 / 소요 시간 (논문 기준)
        if self.current_episode.duration > 0:
            self.current_episode.avg_velocity = (
                self.current_episode.total_distance / self.current_episode.duration
            )

        # 평균 각속도 (ωavg) - heading change smoothness
        if self._angular_velocities:
            self.current_episode.avg_angular_velocity = np.mean(self._angular_velocities)

        # 개인 공간 침범 비율 (ITR) = (침범 시간 / 총 시간) × 100 (%)
        if self._total_steps > 0:
            self.current_episode.intrusion_time_ratio = (
                self._intrusion_steps / self._total_steps * 100
            )

        # 평균 사회적 거리 (SD)
        if self._human_distances:
            self.current_episode.avg_social_distance = np.mean(self._human_distances)

        self.episode_active = False

        return self.current_episode

    def get_trajectory(self) -> List[Dict]:
        """궤적 데이터 반환"""
        return self._trajectory


class MetricsCollectorNode:
    """ROS 노드 래퍼"""

    def __init__(self):
        import rospy
        from nav_msgs.msg import Odometry
        from geometry_msgs.msg import PoseStamped
        from pedsim_msgs.msg import AgentStates
        from std_msgs.msg import String
        from tf.transformations import euler_from_quaternion

        self.rospy = rospy
        self.euler_from_quaternion = euler_from_quaternion

        rospy.init_node('metrics_collector_node', anonymous=True)

        # 파라미터
        self.goal_threshold = rospy.get_param('~goal_threshold', 0.3)
        self.collision_threshold = rospy.get_param('~collision_threshold', 0.35)
        self.personal_space_radius = rospy.get_param('~personal_space_radius', 0.5)
        self.robot_radius = rospy.get_param('~robot_radius', 0.25)
        self.max_timesteps = rospy.get_param('~max_timesteps', 500)

        self.odom_topic = rospy.get_param('~odom_topic', '/p3dx/odom')
        self.agents_topic = rospy.get_param('~agents_topic', '/pedsim_simulator/simulated_agents')

        # 메트릭 수집기
        self.collector = MetricsCollector(
            goal_threshold=self.goal_threshold,
            collision_threshold=self.collision_threshold,
            personal_space_radius=self.personal_space_radius,
            robot_radius=self.robot_radius
        )

        # 상태
        self.robot_pos = None
        self.robot_vel = None
        self.robot_omega = 0.0
        self.goal_pos = None
        self.humans = []

        # Subscribers
        rospy.Subscriber(self.odom_topic, Odometry, self._odom_callback, queue_size=1)
        rospy.Subscriber(self.agents_topic, AgentStates, self._agents_callback, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self._goal_callback, queue_size=1)

        # Publishers
        self.metrics_pub = rospy.Publisher('/experiment/metrics', String, queue_size=1)
        self.result_pub = rospy.Publisher('/experiment/episode_result', String, queue_size=1)

        # 타이머
        self.timer = rospy.Timer(rospy.Duration(0.1), self._update_callback)  # 10 Hz

        rospy.loginfo("[MetricsCollector] Node initialized")

    def _odom_callback(self, msg):
        """오도메트리 콜백"""
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.robot_omega = msg.twist.twist.angular.z

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

    def _goal_callback(self, msg):
        """목표 콜백"""
        self.goal_pos = (msg.pose.position.x, msg.pose.position.y)

    def _update_callback(self, event):
        """업데이트 콜백"""
        if not self.collector.episode_active:
            return

        if self.robot_pos is None or self.goal_pos is None:
            return

        # 메트릭 업데이트
        rt_metrics = self.collector.update(
            robot_pos=self.robot_pos,
            robot_vel=self.robot_vel,
            robot_omega=self.robot_omega,
            goal_pos=self.goal_pos,
            humans=self.humans
        )

        # 실시간 메트릭 발행
        self.metrics_pub.publish(json.dumps(asdict(rt_metrics)))

        # 종료 조건 체크
        terminated, success, collision, timeout = self.collector.check_termination(
            robot_pos=self.robot_pos,
            goal_pos=self.goal_pos,
            max_timesteps=self.max_timesteps
        )

        if terminated:
            episode_result = self.collector.end_episode(success, collision, timeout)
            self.result_pub.publish(json.dumps(asdict(episode_result)))
            self.rospy.loginfo(
                f"[MetricsCollector] Episode ended: "
                f"success={success}, collision={collision}, timeout={timeout}"
            )

    def start_episode(self, episode_id: int, method_name: str, scenario: str,
                      planner: str, use_cigp: bool):
        """에피소드 시작 (외부에서 호출)"""
        if self.robot_pos is None or self.goal_pos is None:
            self.rospy.logwarn("[MetricsCollector] Robot pos or goal not set")
            return

        self.collector.start_episode(
            episode_id=episode_id,
            method_name=method_name,
            scenario=scenario,
            planner=planner,
            use_cigp=use_cigp,
            start_pos=self.robot_pos,
            goal_pos=self.goal_pos
        )
        self.rospy.loginfo(f"[MetricsCollector] Episode {episode_id} started")

    def run(self):
        """노드 실행"""
        self.rospy.spin()


if __name__ == '__main__':
    try:
        node = MetricsCollectorNode()
        node.run()
    except Exception as e:
        print(f"[MetricsCollector] Error: {e}")
        import traceback
        traceback.print_exc()
