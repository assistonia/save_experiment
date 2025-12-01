#!/usr/bin/env python3
"""
DRL-VO Local Planner (Standalone Module)

move_base 대체용 로컬 플래너.
CIGP 글로벌 경로를 따라가면서 DRL-VO 정책으로 cmd_vel 생성.

기존 환경 코드 수정 없이 독립적으로 동작.

Usage:
    from local_planners.drl_vo.drl_vo_planner import DRLVOPlanner

    planner = DRLVOPlanner(model_path="path/to/drl_vo.zip")
    action = planner.compute_action(robot_state, scan_data, ped_positions, goal)
"""

import os
import sys
import numpy as np
import numpy.matlib
from typing import Tuple, List, Optional
from dataclasses import dataclass

# GPU 가속 설정 (RTX 4080)
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class DRLVOConfig:
    """DRL-VO 설정"""
    # 속도 범위
    vx_min: float = 0.0
    vx_max: float = 0.5
    vz_min: float = -2.0
    vz_max: float = 2.0

    # 입력 스케일링
    ped_v_min: float = -2.0
    ped_v_max: float = 2.0
    scan_min: float = 0.0
    scan_max: float = 30.0
    goal_min: float = -2.0
    goal_max: float = 2.0

    # 안전 마진
    goal_margin: float = 0.9  # 목표 도달 판정 거리
    obstacle_margin: float = 0.4  # 장애물 긴급 회피 거리

    # LiDAR 설정
    scan_points: int = 720  # 원본 스캔 포인트 수
    scan_frames: int = 10   # 시간 프레임 수


@dataclass
class DRLVOAction:
    """DRL-VO 출력 액션"""
    linear_x: float
    angular_z: float


class DRLVOPlanner:
    """
    DRL-VO 로컬 플래너

    PPO + ResNet CNN 기반 정책으로 속도 명령 생성.
    보행자 위치 맵 + LiDAR 스캔 + 목표 위치를 입력으로 받음.
    """

    def __init__(
        self,
        model_path: str = None,
        config: DRLVOConfig = None,
        use_gpu: bool = True
    ):
        """
        Args:
            model_path: DRL-VO 모델 파일 경로 (.zip)
            config: DRL-VO 설정
            use_gpu: GPU 사용 여부 (RTX 4080 권장)
        """
        self.config = config or DRLVOConfig()
        self.device = device if use_gpu and torch.cuda.is_available() else torch.device("cpu")

        # 기본 모델 경로
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__),
                "model", "drl_vo.zip"
            )

        self.model = None
        self.model_path = model_path

        # 상태 버퍼 (시간 프레임용)
        self.scan_history: List[np.ndarray] = []
        self.ped_history: List[np.ndarray] = []

        print(f"[DRL-VO] Device: {self.device}")

    def load_model(self):
        """모델 로드 (lazy loading)"""
        if self.model is not None:
            return

        try:
            from stable_baselines3 import PPO

            # Custom CNN 등록 필요
            from .custom_cnn import CustomCNN

            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=256),
            )

            print(f"[DRL-VO] Loading model from {self.model_path}")
            self.model = PPO.load(self.model_path, device=self.device)
            print(f"[DRL-VO] Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"[DRL-VO] Error loading model: {e}")
            raise

    def reset(self):
        """상태 리셋"""
        self.scan_history = []
        self.ped_history = []

    def preprocess_scan(self, scan: np.ndarray) -> np.ndarray:
        """
        LiDAR 스캔 전처리

        Args:
            scan: 원본 스캔 데이터 (720 points)

        Returns:
            처리된 스캔 (6400,)
        """
        cfg = self.config

        # 히스토리 업데이트
        self.scan_history.append(scan.copy())
        if len(self.scan_history) > cfg.scan_frames:
            self.scan_history.pop(0)

        # 패딩 (프레임 부족시)
        while len(self.scan_history) < cfg.scan_frames:
            self.scan_history.insert(0, scan.copy())

        # 스캔 다운샘플링 및 통계
        scan_avg = np.zeros((20, 80))
        for n in range(cfg.scan_frames):
            scan_tmp = self.scan_history[n][:cfg.scan_points]
            for i in range(80):
                scan_avg[2*n, i] = np.min(scan_tmp[i*9:(i+1)*9])
                scan_avg[2*n+1, i] = np.mean(scan_tmp[i*9:(i+1)*9])

        scan_avg = scan_avg.reshape(1600)
        scan_avg_map = np.matlib.repmat(scan_avg, 1, 4)
        scan_processed = scan_avg_map.reshape(6400)

        # 스케일링 [-1, 1]
        scan_processed = 2 * (scan_processed - cfg.scan_min) / (cfg.scan_max - cfg.scan_min) + (-1)

        return scan_processed.astype(np.float32)

    def preprocess_pedestrians(self, ped_positions: np.ndarray) -> np.ndarray:
        """
        보행자 위치 전처리

        Args:
            ped_positions: 보행자 위치 맵 (80x80x2) 또는 (12800,)

        Returns:
            처리된 보행자 맵 (12800,)
        """
        cfg = self.config

        if ped_positions.shape != (12800,):
            ped_positions = ped_positions.flatten()

        # 스케일링 [-1, 1]
        ped_scaled = 2 * (ped_positions - cfg.ped_v_min) / (cfg.ped_v_max - cfg.ped_v_min) + (-1)

        return ped_scaled.astype(np.float32)

    def preprocess_goal(self, goal: np.ndarray) -> np.ndarray:
        """
        목표 위치 전처리 (로봇 기준 상대 좌표)

        Args:
            goal: 목표 위치 [x, y] (로봇 기준)

        Returns:
            처리된 목표 (2,)
        """
        cfg = self.config

        # 스케일링 [-1, 1]
        goal_scaled = 2 * (goal - cfg.goal_min) / (cfg.goal_max - cfg.goal_min) + (-1)

        return goal_scaled.astype(np.float32)

    def compute_action(
        self,
        scan: np.ndarray,
        ped_positions: np.ndarray,
        goal_relative: np.ndarray,
        min_scan_dist: float = None
    ) -> DRLVOAction:
        """
        DRL-VO 액션 계산

        Args:
            scan: LiDAR 스캔 데이터 (720 points)
            ped_positions: 보행자 위치 맵 (12800,) 또는 (80,80,2)
            goal_relative: 로봇 기준 상대 목표 위치 [x, y]
            min_scan_dist: 최소 스캔 거리 (None이면 자동 계산)

        Returns:
            DRLVOAction: linear_x, angular_z
        """
        # 모델 로드 (최초 호출시)
        if self.model is None:
            self.load_model()

        cfg = self.config

        # 최소 스캔 거리 계산
        if min_scan_dist is None:
            scan_front = scan[len(scan)//4:3*len(scan)//4]  # 전방 180도
            scan_valid = scan_front[scan_front > 0]
            min_scan_dist = np.min(scan_valid) if len(scan_valid) > 0 else 10.0

        # 목표 근처 도달
        goal_dist = np.linalg.norm(goal_relative)
        if goal_dist <= cfg.goal_margin:
            return DRLVOAction(linear_x=0.0, angular_z=0.0)

        # 긴급 장애물 회피
        if min_scan_dist <= cfg.obstacle_margin:
            return DRLVOAction(linear_x=0.0, angular_z=0.7)

        # 입력 전처리
        ped_processed = self.preprocess_pedestrians(ped_positions)
        scan_processed = self.preprocess_scan(scan)
        goal_processed = self.preprocess_goal(goal_relative)

        # 관측 벡터 생성
        observation = np.concatenate([ped_processed, scan_processed, goal_processed])

        # 정책 추론
        action, _ = self.model.predict(observation, deterministic=True)

        # 액션 스케일링 ([-1, 1] -> 실제 범위)
        linear_x = (action[0] + 1) * (cfg.vx_max - cfg.vx_min) / 2 + cfg.vx_min
        angular_z = (action[1] + 1) * (cfg.vz_max - cfg.vz_min) / 2 + cfg.vz_min

        # NaN 체크
        if np.isnan(linear_x) or np.isnan(angular_z):
            return DRLVOAction(linear_x=0.0, angular_z=0.0)

        return DRLVOAction(linear_x=float(linear_x), angular_z=float(angular_z))

    def compute_action_simple(
        self,
        robot_pos: Tuple[float, float],
        robot_theta: float,
        goal_world: Tuple[float, float],
        scan: np.ndarray,
        humans: List[dict] = None
    ) -> DRLVOAction:
        """
        간단한 인터페이스로 액션 계산

        Args:
            robot_pos: 로봇 월드 좌표 (x, y)
            robot_theta: 로봇 방향 (rad)
            goal_world: 목표 월드 좌표 (x, y)
            scan: LiDAR 스캔 데이터
            humans: 보행자 리스트 [{'pos': [x,y], 'vel': [vx,vy]}, ...]

        Returns:
            DRLVOAction
        """
        # 목표를 로봇 기준 상대 좌표로 변환
        dx = goal_world[0] - robot_pos[0]
        dy = goal_world[1] - robot_pos[1]

        cos_t = np.cos(-robot_theta)
        sin_t = np.sin(-robot_theta)

        goal_relative = np.array([
            dx * cos_t - dy * sin_t,
            dx * sin_t + dy * cos_t
        ])

        # 보행자 위치 맵 생성 (간단 버전 - 실제로는 더 정교한 처리 필요)
        ped_map = np.zeros((80, 80, 2), dtype=np.float32)

        if humans:
            for h in humans:
                # 로봇 기준 상대 좌표
                hx = h['pos'][0] - robot_pos[0]
                hy = h['pos'][1] - robot_pos[1]

                # 로봇 프레임으로 변환
                hx_rel = hx * cos_t - hy * sin_t
                hy_rel = hx * sin_t + hy * cos_t

                # 맵 좌표 ([-4, 4] -> [0, 80])
                map_x = int((hx_rel + 4) / 8 * 80)
                map_y = int((hy_rel + 4) / 8 * 80)

                if 0 <= map_x < 80 and 0 <= map_y < 80:
                    # 속도 정보
                    hvx = h.get('vel', [0, 0])[0]
                    hvy = h.get('vel', [0, 0])[1]

                    ped_map[map_y, map_x, 0] = hvx * cos_t - hvy * sin_t
                    ped_map[map_y, map_x, 1] = hvx * sin_t + hvy * cos_t

        return self.compute_action(scan, ped_map.flatten(), goal_relative)


class DRLVOPlannerROS:
    """
    DRL-VO ROS 노드 래퍼

    CIGP 글로벌 경로를 구독하고 cmd_vel 발행.
    기존 환경 수정 없이 독립 노드로 동작.
    """

    def __init__(self):
        import rospy
        from geometry_msgs.msg import Twist, PoseStamped
        from nav_msgs.msg import Path, Odometry
        from sensor_msgs.msg import LaserScan
        from pedsim_msgs.msg import AgentStates
        from tf.transformations import euler_from_quaternion

        self.rospy = rospy

        # DRL-VO 플래너
        self.planner = DRLVOPlanner()

        # 상태
        self.robot_pos = None
        self.robot_theta = None
        self.current_goal = None
        self.current_path = []
        self.humans = []
        self.latest_scan = None

        # ROS 설정
        rospy.init_node('drl_vo_local_planner', anonymous=True)

        # 토픽 파라미터 (환경에 맞게 설정 가능)
        self.odom_topic = rospy.get_param('~odom_topic', '/p3dx/odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/p3dx/cmd_vel')
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.agents_topic = rospy.get_param('~agents_topic', '/pedsim_simulator/simulated_agents')

        rospy.loginfo(f"[DRL-VO] odom: {self.odom_topic}, cmd_vel: {self.cmd_vel_topic}")

        # Subscribers
        rospy.Subscriber('/cigp/global_path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/cigp/next_waypoint', PointStamped, self.waypoint_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.agents_topic, AgentStates, self.agents_callback, queue_size=1)

        # Publisher
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)

        # 제어 루프
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("[DRL-VO] Local planner node initialized")

    def path_callback(self, msg):
        """글로벌 경로 콜백"""
        self.current_path = [(p.pose.position.x, p.pose.position.y)
                            for p in msg.poses]

    def waypoint_callback(self, msg):
        """다음 웨이포인트 콜백"""
        self.current_goal = (msg.point.x, msg.point.y)

    def odom_callback(self, msg):
        """오도메트리 콜백"""
        from tf.transformations import euler_from_quaternion

        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)

        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def scan_callback(self, msg):
        """LiDAR 스캔 콜백"""
        self.latest_scan = np.array(msg.ranges)

    def agents_callback(self, msg):
        """보행자 상태 콜백"""
        self.humans = []
        for agent in msg.agent_states:
            self.humans.append({
                'pos': [agent.pose.position.x, agent.pose.position.y],
                'vel': [agent.twist.linear.x, agent.twist.linear.y]
            })

    def run(self):
        """메인 루프"""
        while not self.rospy.is_shutdown():
            if (self.robot_pos is not None and
                self.current_goal is not None and
                self.latest_scan is not None):

                # 액션 계산
                action = self.planner.compute_action_simple(
                    robot_pos=self.robot_pos,
                    robot_theta=self.robot_theta,
                    goal_world=self.current_goal,
                    scan=self.latest_scan,
                    humans=self.humans
                )

                # cmd_vel 발행
                cmd = Twist()
                cmd.linear.x = action.linear_x
                cmd.angular.z = action.angular_z
                self.cmd_pub.publish(cmd)

            self.rate.sleep()


def main():
    """ROS 노드 실행"""
    try:
        node = DRLVOPlannerROS()
        node.run()
    except Exception as e:
        print(f"[DRL-VO] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
