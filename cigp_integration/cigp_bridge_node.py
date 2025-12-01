#!/usr/bin/env python3
"""
CIGP ROS Bridge Node

ROS PedSim 시뮬레이터와 CIGP 모듈을 연결하는 브릿지 노드.
기존 환경 코드를 수정하지 않고 독립적으로 동작.

Subscribe:
    - /pedsim_simulator/simulated_agents (AgentStates): 보행자 상태
    - /odom (Odometry): 로봇 위치/속도
    - /move_base_simple/goal (PoseStamped): 네비게이션 목표

Publish:
    - /cigp/global_path (Path): Human-aware 글로벌 경로
    - /cigp/next_waypoint (PointStamped): 다음 웨이포인트
    - /cigp/cmd_vel (Twist): DWA 속도 명령 (옵션)
    - /cigp/social_cost_map (OccupancyGrid): 시각화용 소셜 코스트 맵
"""

import sys
import os
import numpy as np

# CIGP 모듈 경로 추가
CIGP_PATH = '/home/pyongjoo/Desktop/newstart/sicnav-test'
if CIGP_PATH not in sys.path:
    sys.path.insert(0, CIGP_PATH)

import rospy
from geometry_msgs.msg import Twist, PoseStamped, PointStamped, Point
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from std_msgs.msg import Header
from pedsim_msgs.msg import AgentStates
import tf
from tf.transformations import euler_from_quaternion

from cigp.integration import CIGPNavigator
from cigp.individual_space import HumanState
from cigp.social_cost import RobotState
from cigp.dwa_local_planner import DWALocalPlanner, DWAConfig


class FullState:
    """로봇의 전체 상태 (CIGP 호환용)"""
    def __init__(self, px, py, vx, vy, gx, gy, theta=0.0, radius=0.25):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.theta = theta
        self.radius = radius


class CIGPBridgeNode:
    """CIGP ROS 브릿지 노드"""

    def __init__(self):
        rospy.init_node('cigp_bridge_node', anonymous=True)

        # 파라미터 로드
        self.load_params()

        # CIGP Navigator 초기화
        self.navigator = CIGPNavigator(
            x_range=self.x_range,
            y_range=self.y_range,
            resolution=self.resolution,
            gamma1=self.gamma1,
            robot_radius=self.robot_radius,
            robot_fov=self.robot_fov,
            robot_range=self.robot_range
        )

        # CCTV 네트워크 설정
        self.setup_cctvs()

        # 정적 장애물 설정 (warehouse 시나리오)
        self.setup_static_obstacles()

        # DWA Local Planner (옵션)
        if self.use_dwa:
            self.dwa = DWALocalPlanner(DWAConfig(
                max_speed=self.max_speed,
                max_yaw_rate=self.max_yaw_rate
            ))
        else:
            self.dwa = None

        # 상태 변수
        self.robot_state = None
        self.goal = None
        self.humans = []
        self.last_odom_time = None

        # TF Listener
        self.tf_listener = tf.TransformListener()

        # Publishers
        self.path_pub = rospy.Publisher('/cigp/global_path', Path, queue_size=1)
        self.waypoint_pub = rospy.Publisher('/cigp/next_waypoint', PointStamped, queue_size=1)
        self.cost_map_pub = rospy.Publisher('/cigp/social_cost_map', OccupancyGrid, queue_size=1)

        if self.use_dwa:
            self.cmd_vel_pub = rospy.Publisher('/cigp/cmd_vel', Twist, queue_size=1)

        # Subscribers
        rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates,
                         self.agents_callback, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)

        # 타이머 (메인 루프)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.update_callback)

        rospy.loginfo("CIGP Bridge Node initialized")
        rospy.loginfo(f"  Map range: x={self.x_range}, y={self.y_range}")
        rospy.loginfo(f"  Resolution: {self.resolution}m")
        rospy.loginfo(f"  Gamma1: {self.gamma1}")
        rospy.loginfo(f"  DWA enabled: {self.use_dwa}")

    def load_params(self):
        """ROS 파라미터 로드"""
        # 맵 범위 (warehouse: -12 ~ 12)
        self.x_range = (
            rospy.get_param('~x_min', -12.0),
            rospy.get_param('~x_max', 12.0)
        )
        self.y_range = (
            rospy.get_param('~y_min', -12.0),
            rospy.get_param('~y_max', 12.0)
        )

        # CIGP 파라미터
        self.resolution = rospy.get_param('~resolution', 0.1)
        self.gamma1 = rospy.get_param('~gamma1', 0.5)
        self.robot_radius = rospy.get_param('~robot_radius', 0.25)
        self.robot_fov = rospy.get_param('~robot_fov', np.pi)  # 180도
        self.robot_range = rospy.get_param('~robot_range', 10.0)

        # 업데이트 주기
        self.update_rate = rospy.get_param('~update_rate', 10.0)
        self.replan_interval = rospy.get_param('~replan_interval', 2)

        # DWA 옵션
        self.use_dwa = rospy.get_param('~use_dwa', False)
        self.max_speed = rospy.get_param('~max_speed', 0.8)
        self.max_yaw_rate = rospy.get_param('~max_yaw_rate', 1.0)

        # CCTV 설정
        self.n_cctvs = rospy.get_param('~n_cctvs', 4)
        self.cctv_fov = rospy.get_param('~cctv_fov', np.pi / 2)
        self.cctv_range = rospy.get_param('~cctv_range', 15.0)

        # 퍼블리시 옵션
        self.publish_cost_map = rospy.get_param('~publish_cost_map', True)

    def setup_cctvs(self):
        """CCTV 네트워크 설정"""
        self.navigator.setup_cctvs_auto(
            n_cctvs=self.n_cctvs,
            fov=self.cctv_fov,
            max_range=self.cctv_range
        )
        rospy.loginfo(f"CCTV network configured: {self.n_cctvs} cameras")

    def setup_static_obstacles(self):
        """정적 장애물 설정 (warehouse 시나리오 기반)"""
        # warehouse_pedsim.xml에서 추출한 장애물
        static_obstacles = [
            # Outer walls
            ((-12.0, -12.0), (-12.0, 12.0)),
            ((12.0, -12.0), (12.0, 12.0)),
            ((-12.0, 12.0), (12.0, 12.0)),
            ((-12.0, -12.0), (12.0, -12.0)),

            # Shelf 1 (leftmost)
            ((-12, 4), (-12, -5)),
            ((-12, -5), (-10, -5)),
            ((-10, -5), (-10, 4)),
            ((-10, 4), (-12, 4)),

            # Shelf 2
            ((-7, 4), (-7, -5)),
            ((-7, -5), (-5, -5)),
            ((-5, -5), (-5, 4)),
            ((-5, 4), (-7, 4)),

            # Shelf 3
            ((-2, 4), (-2, -5)),
            ((-2, -5), (0, -5)),
            ((0, -5), (0, 4)),
            ((0, 4), (-2, 4)),

            # Shelf 4
            ((3, 4), (3, -5)),
            ((3, -5), (5, -5)),
            ((5, -5), (5, 4)),
            ((5, 4), (3, 4)),

            # Shelf 5 (rightmost)
            ((10, 4), (10, -5)),
            ((10, -5), (12, -5)),
            ((12, -5), (12, 4)),
            ((12, 4), (10, 4)),
        ]

        self.navigator.set_static_obstacles(static_obstacles)
        rospy.loginfo(f"Static obstacles configured: {len(static_obstacles)} segments")

    def agents_callback(self, msg):
        """보행자 상태 콜백"""
        self.humans = []
        for agent in msg.agent_states:
            human = HumanState(
                id=agent.id,
                px=agent.pose.position.x,
                py=agent.pose.position.y,
                vx=agent.twist.linear.x,
                vy=agent.twist.linear.y,
                radius=0.3  # 기본 보행자 반경
            )
            self.humans.append(human)

    def odom_callback(self, msg):
        """로봇 오도메트리 콜백"""
        # 위치
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y

        # 속도
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y

        # 방향 (쿼터니언 -> 오일러)
        orientation = msg.pose.pose.orientation
        _, _, theta = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

        # 목표 (설정되지 않았으면 현재 위치)
        gx = self.goal[0] if self.goal else px
        gy = self.goal[1] if self.goal else py

        self.robot_state = FullState(
            px=px, py=py,
            vx=vx, vy=vy,
            gx=gx, gy=gy,
            theta=theta,
            radius=self.robot_radius
        )

        self.last_odom_time = rospy.Time.now()

    def goal_callback(self, msg):
        """네비게이션 목표 콜백"""
        self.goal = (msg.pose.position.x, msg.pose.position.y)
        rospy.loginfo(f"New goal received: ({self.goal[0]:.2f}, {self.goal[1]:.2f})")

        # 목표 변경 시 즉시 재계획
        if self.robot_state:
            self.robot_state.gx = self.goal[0]
            self.robot_state.gy = self.goal[1]

    def update_callback(self, event):
        """메인 업데이트 루프"""
        if self.robot_state is None or self.goal is None:
            return

        # CIGP 업데이트
        self.navigator.update(self.robot_state, self.humans)

        # 경로 계획
        path = self.navigator.plan_path()

        if path:
            # 글로벌 경로 퍼블리시
            self.publish_path(path)

            # 다음 웨이포인트 퍼블리시
            next_wp = self.navigator.get_next_waypoint(lookahead=0.5)
            if next_wp:
                self.publish_waypoint(next_wp)

            # DWA 속도 명령 (옵션)
            if self.use_dwa and self.dwa and next_wp:
                self.publish_cmd_vel(next_wp)

        # 소셜 코스트 맵 퍼블리시 (옵션)
        if self.publish_cost_map:
            self.publish_social_cost_map()

    def publish_path(self, waypoints):
        """글로벌 경로 퍼블리시"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'odom'

        for wp in waypoints:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def publish_waypoint(self, waypoint):
        """다음 웨이포인트 퍼블리시"""
        wp_msg = PointStamped()
        wp_msg.header = Header()
        wp_msg.header.stamp = rospy.Time.now()
        wp_msg.header.frame_id = 'odom'
        wp_msg.point.x = waypoint[0]
        wp_msg.point.y = waypoint[1]
        wp_msg.point.z = 0.0

        self.waypoint_pub.publish(wp_msg)

    def publish_cmd_vel(self, next_waypoint):
        """DWA 속도 명령 퍼블리시"""
        if not self.dwa:
            return

        robot = RobotState(
            px=self.robot_state.px,
            py=self.robot_state.py,
            vx=self.robot_state.vx,
            vy=self.robot_state.vy,
            gx=self.robot_state.gx,
            gy=self.robot_state.gy,
            radius=self.robot_state.radius
        )

        action = self.dwa.compute(
            robot_state=robot,
            humans=self.humans,
            next_waypoint=next_waypoint
        )

        if action:
            cmd = Twist()
            cmd.linear.x = action.v
            cmd.angular.z = action.omega
            self.cmd_vel_pub.publish(cmd)

    def publish_social_cost_map(self):
        """소셜 코스트 맵 퍼블리시 (시각화용)"""
        if not self.humans:
            return

        # 그리드 해상도 (coarse for visualization)
        viz_resolution = 0.5

        width = int((self.x_range[1] - self.x_range[0]) / viz_resolution)
        height = int((self.y_range[1] - self.y_range[0]) / viz_resolution)

        # OccupancyGrid 메시지
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = 'odom'

        grid_msg.info.resolution = viz_resolution
        grid_msg.info.width = width
        grid_msg.info.height = height
        grid_msg.info.origin.position.x = self.x_range[0]
        grid_msg.info.origin.position.y = self.y_range[0]
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        # 코스트 맵 계산
        data = []
        for gy in range(height):
            for gx in range(width):
                wx = self.x_range[0] + (gx + 0.5) * viz_resolution
                wy = self.y_range[0] + (gy + 0.5) * viz_resolution

                cost = self.navigator.get_social_cost_at(wx, wy)
                # 0-100 스케일로 변환
                cost_scaled = int(min(cost * 100, 100))
                data.append(cost_scaled)

        grid_msg.data = data
        self.cost_map_pub.publish(grid_msg)

    def run(self):
        """노드 실행"""
        rospy.loginfo("CIGP Bridge Node running...")
        rospy.spin()


def main():
    try:
        node = CIGPBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
