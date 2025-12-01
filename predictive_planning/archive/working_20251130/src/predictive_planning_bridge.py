#!/usr/bin/env python3
"""
Predictive Planning ROS Bridge Node

모든 컴포넌트를 통합하는 ROS 노드.
- PedSim 에이전트 데이터 수신
- SingularTrajectory 예측 수행
- 예측 기반 A* 경로 계획
- 결과 시각화 및 로깅

기존 환경 코드를 수정하지 않고 독립적으로 동작.
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import threading
import time

# Matplotlib (선택적 - 시각화용)
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
    from matplotlib.collections import LineCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[PredictivePlanning] matplotlib not available, visualization disabled")

# 모듈 경로 설정
MODULE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.dirname(MODULE_PATH)
SICNAV_PATH = '/home/pyongjoo/Desktop/newstart/sicnav-test'
if ENV_PATH not in sys.path:
    sys.path.insert(0, ENV_PATH)
if SICNAV_PATH not in sys.path:
    sys.path.insert(0, SICNAV_PATH)

# ROS 임포트 (선택적)
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Point
    from nav_msgs.msg import Odometry, Path
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray
    from pedsim_msgs.msg import AgentStates
    from tf.transformations import euler_from_quaternion
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[PredictivePlanning] ROS not available, running in standalone mode")

from predictive_planning.src.predicted_trajectory import (
    PredictedTrajectory,
    PredictedTrajectoryArray
)
from predictive_planning.src.prediction_receiver import (
    PredictionReceiver,
    MockPredictionReceiver
)
from predictive_planning.src.predictive_cost_calculator import PredictiveCostCalculator
from predictive_planning.src.predictive_global_planner import (
    PredictiveGlobalPlanner,
    SimpleGlobalPlanner,
    PlanningResult
)
from predictive_planning.src.config import PredictivePlanningConfig


class PredictivePlanningBridge:
    """예측 기반 경로 계획 ROS 브릿지"""

    def __init__(self, use_ros: bool = True, use_mock: bool = False):
        """
        Args:
            use_ros: ROS 사용 여부
            use_mock: Mock 예측기 사용 여부 (테스트용)
        """
        self.use_ros = use_ros and ROS_AVAILABLE
        self.use_mock = use_mock

        # 설정 로드
        self.config = PredictivePlanningConfig()

        # 컴포넌트 초기화
        if use_mock:
            self.prediction_receiver = MockPredictionReceiver(self.config)
        else:
            self.prediction_receiver = PredictionReceiver(self.config)

        self.cost_calculator = PredictiveCostCalculator(self.config)
        self.planner = PredictiveGlobalPlanner(self.config, self.cost_calculator)
        self.simple_planner = SimpleGlobalPlanner(self.config)  # 비교용

        # 상태 변수
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.goal: Optional[Tuple[float, float]] = None
        self.current_path: List[Tuple[float, float]] = []
        self.current_agents: List[Dict] = []

        # 예측 결과 캐시
        self.latest_predictions: Optional[PredictedTrajectoryArray] = None

        # 타이밍
        self.last_prediction_time = 0.0
        self.last_planning_time = 0.0
        self.prediction_interval = 0.4  # 예측 주기 (초)
        self.planning_interval = 1.0    # 재계획 주기 (초)

        # 로깅
        self.enable_logging = True
        self.log_dir = None
        self.frames_dir = None
        self.log_data = []
        self.frame_count = 0

        if self.enable_logging:
            self._setup_logging()

        # 웨이포인트 추적
        self.current_waypoint_idx = 0
        self.waypoint_reach_dist = 0.8  # 웨이포인트 도달 판정 거리
        self.last_sent_waypoint = None  # 마지막으로 발행한 웨이포인트 (중복 발행 방지)

        # ROS 초기화
        if self.use_ros:
            self._init_ros()

        print("[PredictivePlanning] Bridge initialized (CIGP-style move_base control)")
        print(f"  Config: resolution={self.config.resolution}, "
              f"sigma_scale={self.config.sigma_scale}")
        print(f"  Use ROS: {self.use_ros}, Use Mock: {self.use_mock}")

    def _setup_logging(self):
        """로깅 디렉토리 설정"""
        log_base = os.path.join(MODULE_PATH, 'logs')
        os.makedirs(log_base, exist_ok=True)

        # 다음 run 번호 찾기
        existing = [d for d in os.listdir(log_base)
                   if d.startswith('run_') and os.path.isdir(os.path.join(log_base, d))]
        if existing:
            max_num = max([int(d.split('_')[1]) for d in existing])
            run_num = max_num + 1
        else:
            run_num = 1

        self.log_dir = os.path.join(log_base, f'run_{run_num:03d}')
        self.frames_dir = os.path.join(self.log_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)

        # 메타데이터 저장
        meta = {
            'start_time': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'use_mock': self.use_mock
        }
        with open(os.path.join(self.log_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"[PredictivePlanning] Logging to: {self.log_dir}")

    def _init_ros(self):
        """ROS 노드 초기화"""
        rospy.init_node('predictive_planning_bridge', anonymous=True)

        # Publishers (CIGP와 동일)
        self.path_pub = rospy.Publisher(
            '/predictive_planning/global_path', Path, queue_size=1
        )
        self.marker_pub = rospy.Publisher(
            '/predictive_planning/markers', MarkerArray, queue_size=1
        )
        # move_base에 웨이포인트 발행 (CIGP와 동일)
        self.goal_pub = rospy.Publisher(
            '/move_base_simple/goal', PoseStamped, queue_size=1
        )
        rospy.loginfo("[PredictivePlanning] Sending waypoints to /move_base_simple/goal (like CIGP)")

        # Subscribers
        rospy.Subscriber(
            '/pedsim_simulator/simulated_agents',
            AgentStates, self._agents_callback, queue_size=1
        )
        # /odom 또는 /p3dx/odom 중 사용 가능한 토픽 구독
        rospy.Subscriber('/p3dx/odom', Odometry, self._odom_callback, queue_size=1)
        # 별도 goal 토픽 사용 (CIGP와 동일 패턴: /predictive_planning/goal)
        rospy.Subscriber(
            '/predictive_planning/goal', PoseStamped, self._goal_callback, queue_size=1
        )

        # 타이머
        self.update_timer = rospy.Timer(
            rospy.Duration(0.1), self._update_callback
        )

        rospy.on_shutdown(self._on_shutdown)

    def _agents_callback(self, msg):
        """에이전트 상태 콜백"""
        timestamp = msg.header.stamp.to_sec()

        agents = []
        for agent in msg.agent_states:
            agents.append({
                'id': agent.id,
                'x': agent.pose.position.x,
                'y': agent.pose.position.y,
                'vx': agent.twist.linear.x,
                'vy': agent.twist.linear.y,
                'timestamp': timestamp
            })

        self.current_agents = agents
        self.prediction_receiver.update_agents(agents, timestamp)

    def _odom_callback(self, msg):
        """오도메트리 콜백"""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y

        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

        self.cost_calculator.update_robot_state(
            self.robot_x, self.robot_y, self.robot_vx, self.robot_vy
        )

    def _goal_callback(self, msg):
        """목표 콜백"""
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        self.current_waypoint_idx = 0  # 웨이포인트 인덱스 초기화
        rospy.loginfo(f"[PredictivePlanning] New goal: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        rospy.loginfo(f"[PredictivePlanning] Robot at: ({self.robot_x:.2f}, {self.robot_y:.2f})")

        # 목표 변경 시 즉시 재계획
        self._do_planning()

    def _update_callback(self, event):
        """주기적 업데이트"""
        current_time = time.time()

        # 예측 업데이트
        if current_time - self.last_prediction_time >= self.prediction_interval:
            self._do_prediction()
            self.last_prediction_time = current_time

        # 경로 재계획
        if (self.goal is not None and
            current_time - self.last_planning_time >= self.planning_interval):
            self._do_planning()
            self.last_planning_time = current_time

        # 로봇 이동
        self._move_robot()

    def _move_robot(self):
        """웨이포인트 추적 - move_base로 전달 (CIGP와 동일)"""
        # 현재 경로 없으면 반환
        if not self.current_path or len(self.current_path) < 2:
            return

        # 목표 도달 체크
        if self.goal:
            dist_to_goal = np.sqrt(
                (self.robot_x - self.goal[0])**2 +
                (self.robot_y - self.goal[1])**2
            )
            if dist_to_goal < 0.5:  # 목표 도달
                rospy.loginfo("[PredictivePlanning] Goal reached!")
                self.goal = None
                self.current_path = []
                self.last_sent_waypoint = None
                return

        target_wp = None
        waypoint_changed = False

        # 현재 웨이포인트까지 거리 체크
        if self.current_waypoint_idx < len(self.current_path):
            target_wp = self.current_path[self.current_waypoint_idx]
            dist_to_wp = np.sqrt(
                (self.robot_x - target_wp[0])**2 +
                (self.robot_y - target_wp[1])**2
            )

            # 웨이포인트 도달 시 다음으로
            if dist_to_wp < self.waypoint_reach_dist:
                self.current_waypoint_idx += 1
                waypoint_changed = True
                if self.current_waypoint_idx < len(self.current_path):
                    target_wp = self.current_path[self.current_waypoint_idx]
                    rospy.loginfo(f"[PredictivePlanning] Waypoint {self.current_waypoint_idx}/{len(self.current_path)}")
                else:
                    target_wp = self.goal
                    rospy.loginfo(f"[PredictivePlanning] Heading to FINAL GOAL")
        else:
            target_wp = self.goal

        # 웨이포인트가 변경되었거나 처음 발행할 때만 move_base에 전달
        if target_wp and (waypoint_changed or self.last_sent_waypoint is None):
            self._send_waypoint_to_movebase(target_wp)
            self.last_sent_waypoint = target_wp
            rospy.loginfo(f"[PredictivePlanning] Sent waypoint to move_base: ({target_wp[0]:.2f}, {target_wp[1]:.2f})")

    def _send_waypoint_to_movebase(self, waypoint):
        """웨이포인트를 move_base goal로 전송 (CIGP와 동일)"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'odom'
        goal_msg.pose.position.x = waypoint[0]
        goal_msg.pose.position.y = waypoint[1]
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)

    def _find_closest_waypoint_idx(self) -> int:
        """로봇 위치에서 적절한 웨이포인트 인덱스 찾기

        move_base의 xy_goal_tolerance (0.5m) 보다 먼 웨이포인트를 선택해야
        move_base가 즉시 "Goal reached"라고 하지 않음
        """
        if not self.current_path:
            return 0

        min_lookahead = 1.0  # move_base xy_goal_tolerance (0.5m) 보다 큰 값

        # 최소 lookahead 거리보다 먼 첫 번째 웨이포인트 찾기
        for i, wp in enumerate(self.current_path):
            dist = np.sqrt((self.robot_x - wp[0])**2 + (self.robot_y - wp[1])**2)
            if dist > min_lookahead:
                return i

        # 모든 웨이포인트가 가까우면 마지막 것 반환
        return len(self.current_path) - 1

    def _do_prediction(self):
        """예측 수행"""
        predictions = self.prediction_receiver.get_predictions()

        if predictions and len(predictions) > 0:
            self.latest_predictions = predictions
            self.cost_calculator.update_predictions(predictions)
            self.planner.update_predictions(predictions)

    def _do_planning(self):
        """경로 계획 수행"""
        if self.goal is None:
            return

        start = (self.robot_x, self.robot_y)

        # 예측 기반 경로 계획
        result = self.planner.plan(start, self.goal)

        if result.success:
            self.current_path = result.path
            # 경로 재계획 시 웨이포인트 인덱스 리셋 (로봇 위치에서 가장 가까운 웨이포인트부터)
            self.current_waypoint_idx = self._find_closest_waypoint_idx()
            self.last_sent_waypoint = None  # 새 경로이므로 웨이포인트 재발행 필요
            print(f"[PredictivePlanning] Path found: {len(result.path)} waypoints, "
                  f"cost={result.total_cost:.2f}, time={result.planning_time*1000:.1f}ms")

            if self.use_ros:
                self._publish_path(result.path)

            # 로깅
            if self.enable_logging:
                self._log_planning_result(result)
        else:
            print("[PredictivePlanning] Path planning failed")

    def _publish_path(self, waypoints: List[Tuple[float, float]]):
        """경로 발행"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'odom'

        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def _log_planning_result(self, result: PlanningResult):
        """계획 결과 로깅"""
        self.frame_count += 1

        log_entry = {
            'frame': self.frame_count,
            'timestamp': time.time(),
            'robot': {
                'x': self.robot_x,
                'y': self.robot_y,
                'theta': self.robot_theta
            },
            'goal': self.goal,
            'planning': {
                'success': result.success,
                'path_length': result.path_length,
                'total_cost': result.total_cost,
                'planning_time_ms': result.planning_time * 1000,
                'iterations': result.iterations,
                'nodes_expanded': result.nodes_expanded
            },
            'agents': len(self.current_agents),
            'predicted_agents': (len(self.latest_predictions)
                               if self.latest_predictions else 0)
        }

        self.log_data.append(log_entry)

        # 시각화 저장
        self._save_visualization()

    def _save_visualization(self):
        """시각화 이미지 저장"""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
            fig.patch.set_facecolor('white')

            # 좌측: 경로 + 예측
            self._draw_path_view(axes[0])

            # 우측: 비용 맵
            self._draw_cost_map(axes[1])

            plt.tight_layout()

            # 저장
            frame_path = os.path.join(self.frames_dir, f'frame_{self.frame_count:05d}.png')
            plt.savefig(frame_path, dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

            latest_path = os.path.join(self.log_dir, 'planning_latest.png')
            plt.savefig(latest_path, dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

            plt.close(fig)
            print(f"[Visualization] Saved frame {self.frame_count}")

            # 예측 전용 이미지도 저장
            self._save_prediction_only_visualization()

        except Exception as e:
            print(f"[Visualization] Error saving: {e}")
            import traceback
            traceback.print_exc()

    def _save_prediction_only_visualization(self):
        """예측 전용 시각화 이미지 저장"""
        if not MATPLOTLIB_AVAILABLE:
            return
        try:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10), facecolor='white')
            fig.patch.set_facecolor('white')

            # 배경 설정 (ROS 좌표계: Y축 정방향)
            ax.set_xlim(self.config.x_range[0] - 1, self.config.x_range[1] + 1)
            ax.set_ylim(self.config.y_range[0] - 1, self.config.y_range[1] + 1)
            ax.set_aspect('equal')
            ax.set_facecolor('#f0f0f0')
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)

            # 장애물
            for obs in self.config.static_obstacles:
                rect = Rectangle(
                    (obs['x_min'], obs['y_min']),
                    obs['x_max'] - obs['x_min'],
                    obs['y_max'] - obs['y_min'],
                    linewidth=1, edgecolor='#555', facecolor='#888', alpha=0.5
                )
                ax.add_patch(rect)

            # 벽
            walls = self.config.walls
            ax.plot([walls['x_min'], walls['x_max']], [walls['y_min'], walls['y_min']], 'k-', lw=2)
            ax.plot([walls['x_min'], walls['x_max']], [walls['y_max'], walls['y_max']], 'k-', lw=2)
            ax.plot([walls['x_min'], walls['x_min']], [walls['y_min'], walls['y_max']], 'k-', lw=2)
            ax.plot([walls['x_max'], walls['x_max']], [walls['y_min'], walls['y_max']], 'k-', lw=2)

            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            predicted_ids = set()

            # 예측 궤적 (SingularTrajectory 출력)
            if self.latest_predictions and len(self.latest_predictions) > 0:
                for traj in self.latest_predictions:
                    color = colors[traj.agent_id % 10]
                    predicted_ids.add(traj.agent_id)

                    # 현재 위치 (큰 원)
                    ax.scatter(traj.current_x, traj.current_y, c=[color], s=200,
                              zorder=10, edgecolors='black', linewidths=2)
                    ax.text(traj.current_x, traj.current_y + 0.7, f'Agent {traj.agent_id}',
                           fontsize=10, ha='center', color=color, fontweight='bold')

                    # 모든 예측 샘플 (20개)
                    for k in range(traj.num_samples):
                        sample = traj.samples[k]  # shape: (pred_horizon, 2)
                        ax.plot(sample[:, 0], sample[:, 1],
                               color=color, alpha=0.15, linewidth=1.5)
                        # 샘플 끝점 표시
                        ax.scatter(sample[-1, 0], sample[-1, 1],
                                  c=[color], s=20, alpha=0.3, zorder=8)

                    # 평균 예측 (두꺼운 선)
                    mean_traj = traj.get_mean_trajectory()
                    ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                           color=color, linewidth=3, alpha=0.9, zorder=9)

                    # 평균 궤적 끝점
                    ax.scatter(mean_traj[-1, 0], mean_traj[-1, 1],
                              c=[color], s=100, marker='x', zorder=9, linewidths=3)

                    # 불확실성 원 (sigma)
                    for t_idx in range(0, len(mean_traj), 3):  # 3스텝마다 표시
                        sigma = traj.get_sigma_at_time(t_idx)
                        circle = plt.Circle(
                            (mean_traj[t_idx, 0], mean_traj[t_idx, 1]),
                            sigma, fill=False, color=color, alpha=0.3, linestyle='--'
                        )
                        ax.add_patch(circle)

            # 예측 없는 에이전트 (회색)
            for agent in self.current_agents:
                if agent['id'] not in predicted_ids:
                    ax.scatter(agent['x'], agent['y'], c='gray', s=120, alpha=0.7,
                              zorder=8, edgecolors='black', linewidths=1)
                    ax.text(agent['x'], agent['y'] + 0.5, f"A{agent['id']}",
                           fontsize=9, ha='center', color='gray')
                    # 속도 벡터
                    vx, vy = agent.get('vx', 0), agent.get('vy', 0)
                    if abs(vx) > 0.01 or abs(vy) > 0.01:
                        ax.arrow(agent['x'], agent['y'], vx, vy,
                                head_width=0.2, head_length=0.15,
                                fc='red', ec='red', alpha=0.7)

            # 로봇 위치
            robot_circle = Circle((self.robot_x, self.robot_y), 0.35,
                                 color='blue', alpha=0.9, zorder=15)
            ax.add_patch(robot_circle)
            ax.text(self.robot_x, self.robot_y + 0.7, 'Robot',
                   fontsize=10, ha='center', fontweight='bold', color='blue')

            # 목표
            if self.goal:
                ax.scatter(self.goal[0], self.goal[1], c='green', s=300,
                          marker='*', zorder=15, edgecolors='white', linewidths=2)

            # 제목
            pred_count = len(self.latest_predictions) if self.latest_predictions else 0
            num_samples = self.latest_predictions[0].num_samples if pred_count > 0 else 0
            ax.set_title(
                f'SingularTrajectory Prediction | Frame {self.frame_count}\n'
                f'Agents: {len(self.current_agents)}, Predicted: {pred_count}, '
                f'Samples/Agent: {num_samples}',
                fontsize=12, fontweight='bold'
            )

            plt.tight_layout()

            # 저장
            pred_path = os.path.join(self.frames_dir, f'pred_{self.frame_count:05d}.png')
            plt.savefig(pred_path, dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

            latest_pred_path = os.path.join(self.log_dir, 'prediction_latest.png')
            plt.savefig(latest_pred_path, dpi=100, bbox_inches='tight',
                       facecolor='white', edgecolor='none')

            plt.close(fig)
            print(f"[Visualization] Saved prediction frame {self.frame_count} (agents: {len(self.current_agents)}, pred: {pred_count})")

        except Exception as e:
            print(f"[Visualization] Error saving prediction: {e}")
            import traceback
            traceback.print_exc()

    def _draw_path_view(self, ax):
        """경로 뷰 그리기"""
        ax.set_xlim(self.config.x_range[0] - 1, self.config.x_range[1] + 1)
        ax.set_ylim(self.config.y_range[0] - 1, self.config.y_range[1] + 1)  # ROS 좌표계
        ax.set_aspect('equal')
        ax.set_facecolor('#e8e8e8')
        ax.grid(True, linestyle='--', alpha=0.5, color='gray')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        # 장애물
        for obs in self.config.static_obstacles:
            rect = Rectangle(
                (obs['x_min'], obs['y_min']),
                obs['x_max'] - obs['x_min'],
                obs['y_max'] - obs['y_min'],
                linewidth=1, edgecolor='#555', facecolor='#888', alpha=0.6
            )
            ax.add_patch(rect)

        # 벽
        walls = self.config.walls
        ax.plot([walls['x_min'], walls['x_max']], [walls['y_min'], walls['y_min']], 'k-', lw=2)
        ax.plot([walls['x_min'], walls['x_max']], [walls['y_max'], walls['y_max']], 'k-', lw=2)
        ax.plot([walls['x_min'], walls['x_min']], [walls['y_min'], walls['y_max']], 'k-', lw=2)
        ax.plot([walls['x_max'], walls['x_max']], [walls['y_min'], walls['y_max']], 'k-', lw=2)

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        predicted_ids = set()

        # 예측 궤적
        if self.latest_predictions and len(self.latest_predictions) > 0:
            for traj in self.latest_predictions:
                color = colors[traj.agent_id % 10]
                predicted_ids.add(traj.agent_id)

                # 현재 위치
                ax.scatter(traj.current_x, traj.current_y, c=[color], s=100, zorder=10)
                ax.text(traj.current_x, traj.current_y + 0.5, f'P{traj.agent_id}',
                       fontsize=8, ha='center', color=color)

                # 예측 샘플 (일부만)
                for k in range(min(5, traj.num_samples)):
                    sample = traj.samples[k]
                    ax.plot(sample[:, 0], sample[:, 1],
                           color=color, alpha=0.2, linewidth=1)

                # 평균 예측
                mean_traj = traj.get_mean_trajectory()
                ax.plot(mean_traj[:, 0], mean_traj[:, 1],
                       color=color, linewidth=2, alpha=0.8)

        # 예측 없는 현재 에이전트도 표시 (회색 원)
        for agent in self.current_agents:
            if agent['id'] not in predicted_ids:
                ax.scatter(agent['x'], agent['y'], c='gray', s=80, alpha=0.7, zorder=9)
                ax.text(agent['x'], agent['y'] + 0.4, f"A{agent['id']}",
                       fontsize=7, ha='center', color='gray')
                # 속도 벡터 표시
                if abs(agent.get('vx', 0)) > 0.01 or abs(agent.get('vy', 0)) > 0.01:
                    ax.arrow(agent['x'], agent['y'],
                            agent['vx'] * 0.5, agent['vy'] * 0.5,
                            head_width=0.15, head_length=0.1,
                            fc='gray', ec='gray', alpha=0.5)

        # 로봇
        robot_circle = Circle((self.robot_x, self.robot_y), 0.3,
                             color='blue', alpha=0.9, zorder=15)
        ax.add_patch(robot_circle)
        ax.text(self.robot_x, self.robot_y + 0.6, 'Robot',
               fontsize=9, ha='center', fontweight='bold', color='blue')

        # 목표
        if self.goal:
            ax.scatter(self.goal[0], self.goal[1], c='green', s=200,
                      marker='*', zorder=15, edgecolors='white', linewidths=2)
            ax.text(self.goal[0], self.goal[1] + 0.6, 'Goal',
                   fontsize=9, ha='center', color='green')

        # 경로
        if self.current_path:
            path_x = [p[0] for p in self.current_path]
            path_y = [p[1] for p in self.current_path]
            ax.plot(path_x, path_y, 'b-', linewidth=2.5, alpha=0.8, zorder=12)
            ax.scatter(path_x, path_y, c='blue', s=20, alpha=0.5, zorder=12)

        # 디버그 정보
        pred_count = len(self.latest_predictions) if self.latest_predictions else 0
        ax.set_title(f'Predictive Path Planning | Frame {self.frame_count}\n'
                    f'Agents: {len(self.current_agents)} (Predicted: {pred_count}), '
                    f'Path: {len(self.current_path)} waypoints',
                    fontsize=11, fontweight='bold')

    def _draw_cost_map(self, ax):
        """비용 맵 그리기 - 예측 샘플 기반 시각화"""
        resolution = 0.5
        x_range = self.config.x_range
        y_range = self.config.y_range
        width = int((x_range[1] - x_range[0]) / resolution)
        height = int((y_range[1] - y_range[0]) / resolution)
        cost_grid = np.zeros((height, width))

        # 비용 맵 표시 - ROS 좌표계: Y축 아래에서 위로
        extent = [
            self.config.x_range[0], self.config.x_range[1],
            self.config.y_range[0], self.config.y_range[1]  # ROS 좌표계: [x_min, x_max, y_min, y_max]
        ]

        # 예측 데이터가 있으면 예측 샘플 기반 cost map 생성 (CIGP 스타일)
        if self.latest_predictions and len(self.latest_predictions) > 0:
            # CIGP와 유사하게 더 큰 sigma 사용
            base_sigma = 2.5

            for traj in self.latest_predictions:
                # 더 많은 시간 스텝 사용 (0 ~ 6, 총 7개 시점)
                for t_idx in range(min(7, traj.pred_horizon)):
                    t = t_idx * self.config.time_step
                    positions = traj.get_position_at_time(t)  # (20, 2)

                    # 평균 위치 계산 (샘플들의 중심)
                    mean_pos = np.mean(positions, axis=0)

                    # 샘플 분산 (불확실성)
                    std_pos = np.std(positions, axis=0)
                    uncertainty = np.linalg.norm(std_pos)

                    # 속도 기반 sigma 조절 (CIGP 스타일)
                    velocities = traj.get_velocity_at_time(t)
                    mean_velocity = np.mean(velocities)

                    # sigma = base + velocity * scale + uncertainty
                    sigma = base_sigma + 0.5 * mean_velocity + 0.8 * uncertainty

                    # 시간에 따른 감쇠 (CIGP: 먼 미래일수록 약해짐)
                    time_decay = np.exp(-0.15 * t_idx)

                    # 평균 위치에 큰 가우시안 그리기
                    for gy in range(height):
                        for gx in range(width):
                            wx = x_range[0] + (gx + 0.5) * resolution
                            wy = y_range[0] + (gy + 0.5) * resolution
                            dist = np.sqrt((wx - mean_pos[0])**2 + (wy - mean_pos[1])**2)
                            if dist < sigma * 3:
                                cost_grid[gy, gx] += np.exp(-0.5 * (dist / sigma)**2) * time_decay

                    # 개별 샘플도 약하게 추가 (불확실성 표현)
                    for sample_idx in range(traj.num_samples):
                        sx, sy = positions[sample_idx]
                        sample_sigma = sigma * 0.5  # 샘플은 더 작은 sigma

                        for gy in range(height):
                            for gx in range(width):
                                wx = x_range[0] + (gx + 0.5) * resolution
                                wy = y_range[0] + (gy + 0.5) * resolution
                                dist = np.sqrt((wx - sx)**2 + (wy - sy)**2)
                                if dist < sample_sigma * 2:
                                    cost_grid[gy, gx] += np.exp(-0.5 * (dist / sample_sigma)**2) * time_decay * 0.1 / traj.num_samples

        # 예측 데이터가 없으면 현재 에이전트 위치 기반 fallback
        elif self.current_agents:
            for agent in self.current_agents:
                ax_pos = agent['x']
                ay_pos = agent['y']
                sigma = 1.5  # 기본 반경

                for gy in range(height):
                    for gx in range(width):
                        wx = x_range[0] + (gx + 0.5) * resolution
                        wy = y_range[0] + (gy + 0.5) * resolution
                        dist = np.sqrt((wx - ax_pos)**2 + (wy - ay_pos)**2)
                        if dist < sigma * 2:
                            cost_grid[gy, gx] += np.exp(-0.5 * (dist / sigma)**2)

        max_cost = np.max(cost_grid)
        vmax = max(np.percentile(cost_grid, 95) + 0.1, 0.5)

        im = ax.imshow(cost_grid, extent=extent, cmap='hot_r',
                      alpha=0.7, vmin=0, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax, label='Social Cost')

        # 장애물 오버레이
        for obs in self.config.static_obstacles:
            rect = Rectangle(
                (obs['x_min'], obs['y_min']),
                obs['x_max'] - obs['x_min'],
                obs['y_max'] - obs['y_min'],
                linewidth=1, edgecolor='white', facecolor='gray', alpha=0.8
            )
            ax.add_patch(rect)

        # 현재 에이전트 표시 (cost map 위)
        for agent in self.current_agents:
            ax.scatter(agent['x'], agent['y'], c='yellow', s=60,
                      marker='o', zorder=12, edgecolors='red', linewidths=1.5)

        # 로봇, 목표
        ax.scatter(self.robot_x, self.robot_y, c='cyan', s=150,
                  marker='o', zorder=15, edgecolors='white', linewidths=2)
        if self.goal:
            ax.scatter(self.goal[0], self.goal[1], c='lime', s=200,
                      marker='*', zorder=15, edgecolors='white', linewidths=2)

        ax.set_xlim(self.config.x_range[0] - 1, self.config.x_range[1] + 1)
        ax.set_ylim(self.config.y_range[0] - 1, self.config.y_range[1] + 1)  # ROS 좌표계
        ax.set_aspect('equal')

        # 디버그: 예측 상태 표시
        pred_status = "Predictions ON" if (self.latest_predictions and len(self.latest_predictions) > 0) else "No Predictions"
        ax.set_title(f'Social Cost Map ({pred_status})\n'
                    f'Max Cost: {max_cost:.3f}',
                    fontsize=11, fontweight='bold')

    def _on_shutdown(self):
        """종료 시 로그 저장"""
        if self.enable_logging and self.log_data:
            log_path = os.path.join(self.log_dir, 'log.json')
            with open(log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            print(f"[PredictivePlanning] Saved {len(self.log_data)} log entries")

    def run(self):
        """노드 실행"""
        if self.use_ros:
            print("[PredictivePlanning] Running with ROS...")
            rospy.spin()
        else:
            print("[PredictivePlanning] Running standalone (press Ctrl+C to exit)...")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                self._on_shutdown()

    # === Standalone API (ROS 없이 사용) ===

    def update_agents_standalone(self, agents: List[Dict], timestamp: float = None):
        """에이전트 업데이트 (standalone)"""
        self.current_agents = agents
        self.prediction_receiver.update_agents(agents, timestamp or time.time())

    def update_robot_standalone(self, x: float, y: float, theta: float = 0.0,
                                vx: float = 0.0, vy: float = 0.0):
        """로봇 상태 업데이트 (standalone)"""
        self.robot_x = x
        self.robot_y = y
        self.robot_theta = theta
        self.robot_vx = vx
        self.robot_vy = vy
        self.cost_calculator.update_robot_state(x, y, vx, vy)

    def set_goal_standalone(self, goal: Tuple[float, float]):
        """목표 설정 (standalone)"""
        self.goal = goal

    def plan_standalone(self) -> PlanningResult:
        """경로 계획 (standalone)"""
        # 예측 업데이트
        self._do_prediction()

        # 계획
        if self.goal is None:
            return PlanningResult(False, [], 0, float('inf'), 0, 0, 0)

        start = (self.robot_x, self.robot_y)
        result = self.planner.plan(start, self.goal)

        if result.success:
            self.current_path = result.path

        # 로깅
        if self.enable_logging:
            self._log_planning_result(result)

        return result


def main():
    """메인 함수"""
    # 인자 파싱
    use_mock = '--mock' in sys.argv

    try:
        bridge = PredictivePlanningBridge(use_ros=True, use_mock=use_mock)
        bridge.run()
    except Exception as e:
        print(f"[PredictivePlanning] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
