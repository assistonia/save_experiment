#!/usr/bin/env python3
"""
통합 로컬 플래너 노드

다양한 로컬 플래너를 선택적으로 사용 가능.
기존 환경 코드 수정 없이 독립적으로 동작.

Usage:
    # DRL-VO 사용
    rosrun local_planners local_planner_node.py _planner:=drl_vo

    # TEB 사용
    rosrun local_planners local_planner_node.py _planner:=teb

    # DWA 사용 (sicnav-test)
    rosrun local_planners local_planner_node.py _planner:=dwa

    # ORCA 사용
    rosrun local_planners local_planner_node.py _planner:=orca

    # SFM 사용
    rosrun local_planners local_planner_node.py _planner:=sfm
"""

import sys
import os
import numpy as np
from typing import Tuple, Optional

# CIGP 모듈 경로 추가
SICNAV_PATH = '/home/pyongjoo/Desktop/newstart/sicnav-test'
if SICNAV_PATH not in sys.path:
    sys.path.insert(0, SICNAV_PATH)


class LocalPlannerNode:
    """통합 로컬 플래너 노드"""

    def __init__(self):
        import rospy
        from geometry_msgs.msg import Twist, PoseStamped, PointStamped
        from nav_msgs.msg import Path, Odometry
        from sensor_msgs.msg import LaserScan
        from pedsim_msgs.msg import AgentStates
        from tf.transformations import euler_from_quaternion

        self.rospy = rospy
        self.Twist = Twist  # run() 메서드에서 사용하기 위해 저장

        # ROS 초기화
        rospy.init_node('local_planner_node', anonymous=True)

        # 파라미터
        self.planner_type = rospy.get_param('~planner', 'dwa')
        self.max_speed = rospy.get_param('~max_speed', 0.8)
        self.robot_radius = rospy.get_param('~robot_radius', 0.3)

        rospy.loginfo(f"[LocalPlanner] Using planner: {self.planner_type}")

        # 토픽 파라미터 (환경에 맞게 설정 가능)
        self.odom_topic = rospy.get_param('~odom_topic', '/p3dx/odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/p3dx/cmd_vel')
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.agents_topic = rospy.get_param('~agents_topic', '/pedsim_simulator/simulated_agents')

        rospy.loginfo(f"[LocalPlanner] odom: {self.odom_topic}, cmd_vel: {self.cmd_vel_topic}")

        # 플래너 초기화
        self.planner = self._init_planner()

        # 상태
        self.robot_pos = None
        self.robot_theta = None
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.robot_omega = 0.0
        self.current_goal = None
        self.current_path = []
        self.humans = []
        self.latest_scan = None
        self.obstacles = []

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

        rospy.loginfo(f"[LocalPlanner] Node initialized with {self.planner_type}")

    def _init_planner(self):
        """플래너 초기화"""
        if self.planner_type == 'drl_vo':
            from local_planners.drl_vo.drl_vo_planner import DRLVOPlanner
            return DRLVOPlanner()

        elif self.planner_type == 'teb':
            # TEB는 별도 노드로 실행 권장
            self.rospy.logwarn("[LocalPlanner] TEB: Use teb_planner.py directly for full features")
            return None

        elif self.planner_type == 'dwa':
            from cigp.dwa_local_planner import DWALocalPlanner
            return DWALocalPlanner(
                robot_radius=self.robot_radius,
                max_speed=self.max_speed
            )

        elif self.planner_type == 'orca':
            try:
                import rvo2
                self.rospy.loginfo("[LocalPlanner] ORCA: Using RVO2 library")
                return self._create_orca_planner()
            except ImportError:
                self.rospy.logwarn("[LocalPlanner] ORCA: RVO2 not found, using fallback")
                return self._create_orca_fallback()

        elif self.planner_type == 'sfm':
            return self._create_sfm_planner()

        else:
            self.rospy.logwarn(f"[LocalPlanner] Unknown planner: {self.planner_type}, using DWA")
            from cigp.dwa_local_planner import DWALocalPlanner
            return DWALocalPlanner(robot_radius=self.robot_radius, max_speed=self.max_speed)

    def _create_orca_planner(self):
        """ORCA 플래너 생성"""
        import rvo2

        class ORCAPlanner:
            def __init__(self, max_speed=0.8, robot_radius=0.3):
                self.max_speed = max_speed
                self.robot_radius = robot_radius
                self.sim = None

            def compute(self, robot_pos, robot_vel, goal, humans):
                # RVO2 시뮬레이터 초기화
                if self.sim is None or len(humans) + 1 != self.sim.getNumAgents():
                    self.sim = rvo2.PyRVOSimulator(
                        0.25,  # timeStep
                        10.0,  # neighborDist
                        10,    # maxNeighbors
                        2.0,   # timeHorizon
                        0.5,   # timeHorizonObst
                        self.robot_radius,
                        self.max_speed
                    )

                    # 로봇 추가
                    self.sim.addAgent(tuple(robot_pos))

                    # 사람들 추가
                    for h in humans:
                        self.sim.addAgent(tuple(h['pos']))

                # 상태 업데이트
                self.sim.setAgentPosition(0, tuple(robot_pos))
                self.sim.setAgentVelocity(0, tuple(robot_vel))

                for i, h in enumerate(humans):
                    self.sim.setAgentPosition(i + 1, tuple(h['pos']))
                    self.sim.setAgentVelocity(i + 1, tuple(h['vel']))

                # 선호 속도 설정
                goal_vec = np.array(goal) - np.array(robot_pos)
                goal_dist = np.linalg.norm(goal_vec)
                if goal_dist > 0.1:
                    pref_vel = goal_vec / goal_dist * self.max_speed
                else:
                    pref_vel = (0, 0)

                self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

                for i, h in enumerate(humans):
                    self.sim.setAgentPrefVelocity(i + 1, tuple(h['vel']))

                # 시뮬레이션 스텝
                self.sim.doStep()

                return self.sim.getAgentVelocity(0)

        return ORCAPlanner(self.max_speed, self.robot_radius)

    def _create_orca_fallback(self):
        """ORCA fallback (RVO2 없을 때)"""
        class ORCAFallback:
            def __init__(self, max_speed=0.8, robot_radius=0.3):
                self.max_speed = max_speed
                self.robot_radius = robot_radius

            def compute(self, robot_pos, robot_vel, goal, humans):
                # 목표 방향
                goal_vec = np.array(goal) - np.array(robot_pos)
                goal_dist = np.linalg.norm(goal_vec)

                if goal_dist < 0.1:
                    return (0.0, 0.0)

                pref_vel = goal_vec / goal_dist * self.max_speed

                # 사람들과의 반발력
                vx, vy = pref_vel[0], pref_vel[1]
                for h in humans:
                    diff = np.array(robot_pos) - np.array(h['pos'])
                    dist = np.linalg.norm(diff)
                    safe_dist = self.robot_radius + h.get('radius', 0.3) + 1.0

                    if dist < safe_dist and dist > 0.01:
                        repulsion = (safe_dist - dist) / safe_dist * 2.0
                        direction = diff / dist
                        vx += direction[0] * repulsion
                        vy += direction[1] * repulsion

                return (vx, vy)

        return ORCAFallback(self.max_speed, self.robot_radius)

    def _create_sfm_planner(self):
        """SFM 플래너 생성"""
        class SFMPlanner:
            def __init__(self, max_speed=0.8, robot_radius=0.3):
                self.max_speed = max_speed
                self.robot_radius = robot_radius
                self.tau = 0.5  # relaxation time
                self.A_human = 2.0
                self.B_human = 0.8
                self.A_obs = 3.0
                self.B_obs = 0.5

            def compute(self, robot_pos, robot_vel, goal, humans, obstacles=None):
                pos = np.array(robot_pos)
                vel = np.array(robot_vel)
                goal = np.array(goal)

                # Driving force
                direction = goal - pos
                dist = np.linalg.norm(direction)
                if dist < 0.1:
                    return (0.0, 0.0)

                desired_vel = direction / dist * self.max_speed
                f_goal = (desired_vel - vel) / self.tau

                # Human repulsion
                f_humans = np.array([0.0, 0.0])
                for h in humans:
                    diff = pos - np.array(h['pos'])
                    d = np.linalg.norm(diff)
                    if d < 0.01:
                        continue
                    r_ij = self.robot_radius + h.get('radius', 0.3)
                    magnitude = self.A_human * np.exp((r_ij - d) / self.B_human)
                    f_humans += magnitude * diff / d

                # Total force
                f_total = f_goal + f_humans

                # Velocity update
                new_vel = vel + f_total * 0.25  # dt

                # Speed limit
                speed = np.linalg.norm(new_vel)
                if speed > self.max_speed:
                    new_vel = new_vel / speed * self.max_speed

                return (new_vel[0], new_vel[1])

        return SFMPlanner(self.max_speed, self.robot_radius)

    def path_callback(self, msg):
        """글로벌 경로 콜백"""
        self.current_path = [(p.pose.position.x, p.pose.position.y)
                            for p in msg.poses]

        # DWA에 경로 설정
        if self.planner_type == 'dwa' and hasattr(self.planner, 'set_global_path'):
            self.planner.set_global_path(self.current_path)

    def waypoint_callback(self, msg):
        """다음 웨이포인트 콜백"""
        self.current_goal = (msg.point.x, msg.point.y)

    def odom_callback(self, msg):
        """오도메트리 콜백"""
        from tf.transformations import euler_from_quaternion

        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vx = msg.twist.twist.linear.x
        self.robot_vy = msg.twist.twist.linear.y
        self.robot_omega = msg.twist.twist.angular.z

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
                'id': agent.id,
                'pos': [agent.pose.position.x, agent.pose.position.y],
                'vel': [agent.twist.linear.x, agent.twist.linear.y],
                'radius': 0.3
            })

    def compute_action(self) -> Tuple[float, float]:
        """액션 계산"""
        if self.robot_pos is None or self.current_goal is None:
            return 0.0, 0.0

        if self.planner_type == 'drl_vo':
            if self.latest_scan is None:
                return 0.0, 0.0

            action = self.planner.compute_action_simple(
                robot_pos=self.robot_pos,
                robot_theta=self.robot_theta,
                goal_world=self.current_goal,
                scan=self.latest_scan,
                humans=self.humans
            )
            return action.linear_x, action.angular_z

        elif self.planner_type == 'dwa':
            from cigp.social_cost import RobotState

            robot_state = RobotState(
                px=self.robot_pos[0],
                py=self.robot_pos[1],
                vx=self.robot_vx,
                vy=self.robot_vy,
                gx=self.current_goal[0],
                gy=self.current_goal[1],
                radius=self.robot_radius
            )

            # HumanState 변환
            from cigp.individual_space import HumanState
            human_states = [
                HumanState(
                    id=h['id'],
                    px=h['pos'][0],
                    py=h['pos'][1],
                    vx=h['vel'][0],
                    vy=h['vel'][1],
                    radius=h['radius']
                ) for h in self.humans
            ]

            action = self.planner.compute_action_from_state(
                robot_state=robot_state,
                goal=self.current_goal,
                humans=human_states
            )
            return action.v, action.omega

        elif self.planner_type in ['orca', 'sfm']:
            robot_vel = (self.robot_vx, self.robot_vy)
            vx, vy = self.planner.compute(
                robot_pos=self.robot_pos,
                robot_vel=robot_vel,
                goal=self.current_goal,
                humans=self.humans
            )

            # vx, vy -> v, omega 변환
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.01:
                target_theta = np.arctan2(vy, vx)
                angle_error = target_theta - self.robot_theta
                while angle_error > np.pi:
                    angle_error -= 2 * np.pi
                while angle_error < -np.pi:
                    angle_error += 2 * np.pi

                if abs(angle_error) > 0.5:
                    return 0.2, np.clip(angle_error * 2, -1.0, 1.0)
                else:
                    return speed, np.clip(angle_error, -1.0, 1.0)
            return 0.0, 0.0

        return 0.0, 0.0

    def run(self):
        """메인 루프"""
        self.rospy.loginfo(f"[LocalPlanner] Running with {self.planner_type}...")

        while not self.rospy.is_shutdown():
            # 액션 계산
            linear_x, angular_z = self.compute_action()

            # cmd_vel 발행
            cmd = self.Twist()
            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.cmd_pub.publish(cmd)

            self.rate.sleep()


def main():
    """노드 실행"""
    try:
        from geometry_msgs.msg import Twist
        node = LocalPlannerNode()
        node.run()
    except Exception as e:
        print(f"[LocalPlanner] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
