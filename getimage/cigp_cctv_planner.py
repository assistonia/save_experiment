#!/usr/bin/env python3
"""
CIGP Global Planner - CCTV 기반 버전

기존 cigp_global_planner.py와 동일한 로직이지만,
/pedsim_simulator/simulated_agents (GT) 대신
/cctv/detected_agents (CCTV 감지 결과)를 사용.

Subscribe:
    - /cctv/detected_agents: CCTV 감지 사람 위치
    - /p3dx/odom: 로봇 위치 (Pioneer3DX)
    - /cigp_cctv/goal: 목표 위치

Publish:
    - /cigp_cctv/global_path: Human-aware 글로벌 경로
    - /cigp_cctv/waypoints: 웨이포인트 마커 (RViz 시각화)
    - /cigp_cctv/costmap_image: 코스트맵 이미지 (시각화)
"""

import sys
import os
import numpy as np
import json
import time
from datetime import datetime

# CIGP 모듈 경로 추가
CIGP_PATH = '/sicnav-test'
if CIGP_PATH not in sys.path:
    sys.path.insert(0, CIGP_PATH)

import rospy
from geometry_msgs.msg import PoseStamped, Point, Twist
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from pedsim_msgs.msg import AgentStates
from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
import cv2

from cigp.integration import CIGPNavigator
from cigp.individual_space import HumanState


class RobotFullState:
    """로봇 전체 상태"""
    def __init__(self, px=0, py=0, vx=0, vy=0, gx=0, gy=0, theta=0, radius=0.25):
        self.px = px
        self.py = py
        self.vx = vx
        self.vy = vy
        self.gx = gx
        self.gy = gy
        self.theta = theta
        self.radius = radius


class CIGPCCTVPlanner:
    """CIGP 글로벌 플래너 - CCTV 기반"""

    def __init__(self):
        rospy.init_node('cigp_cctv_planner', anonymous=True)

        # Warehouse 맵 설정 (-12 ~ 12)
        self.x_range = (-12.0, 12.0)
        self.y_range = (-12.0, 12.0)

        # CIGP Navigator 초기화
        self.navigator = CIGPNavigator(
            x_range=self.x_range,
            y_range=self.y_range,
            resolution=0.2,
            gamma1=0.5,
            robot_radius=0.25,
            robot_fov=np.pi,
            robot_range=10.0
        )

        # CCTV 자동 배치 (4개)
        self.navigator.setup_cctvs_auto(n_cctvs=4, fov=np.pi/2, max_range=15.0)

        # 정적 장애물 설정 (warehouse)
        self._setup_warehouse_obstacles()

        # 상태 변수
        self.robot_state = RobotFullState()
        self.goal = None
        self.humans = []
        self.path = []
        self.got_odom = False

        # 로그 저장용
        self.base_log_dir = '/environment/cigp_cctv_logs'
        self.log_dir = self._create_run_folder()
        self.log_data = []
        self.start_time = time.time()
        self.frame_count = 0

        # CV Bridge for image
        self.bridge = CvBridge()

        # Publishers - CCTV 버전은 다른 토픽 사용
        self.path_pub = rospy.Publisher('/cigp_cctv/global_path', Path, queue_size=1)
        self.marker_pub = rospy.Publisher('/cigp_cctv/waypoints', MarkerArray, queue_size=1)
        self.costmap_pub = rospy.Publisher('/cigp_cctv/costmap_image', Image, queue_size=1)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)

        # 웨이포인트 추적
        self.current_waypoint_idx = 0
        self.waypoint_reach_dist = 0.8

        # Subscribers
        # ★ 핵심 변경: GT 대신 CCTV 감지 결과 사용
        rospy.Subscriber('/cctv/detected_agents', AgentStates,
                         self.agents_callback, queue_size=1)
        rospy.Subscriber('/p3dx/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/cigp_cctv/goal', PoseStamped, self.goal_callback, queue_size=1)

        # 메인 루프 타이머 (5Hz)
        self.timer = rospy.Timer(rospy.Duration(0.2), self.planning_loop)

        # 시각화 타이머 (1Hz)
        self.viz_timer = rospy.Timer(rospy.Duration(1.0), self.visualize_costmap)

        rospy.loginfo("=" * 60)
        rospy.loginfo("CIGP CCTV Planner Started (Vision-based)")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Subscribed to:")
        rospy.loginfo("  - /cctv/detected_agents (CCTV detections)")
        rospy.loginfo("  - /p3dx/odom (robot)")
        rospy.loginfo("  - /cigp_cctv/goal (goal - use this!)")
        rospy.loginfo("Publishing to:")
        rospy.loginfo("  - /cigp_cctv/global_path (Path)")
        rospy.loginfo("  - /cigp_cctv/waypoints (MarkerArray)")
        rospy.loginfo("  - /cigp_cctv/costmap_image (Image)")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Logs: %s", self.log_dir)
        rospy.loginfo("=" * 60)

    def _create_run_folder(self):
        """실행마다 새 폴더 생성"""
        os.makedirs(self.base_log_dir, exist_ok=True)

        existing = [d for d in os.listdir(self.base_log_dir) if d.startswith('run_')]
        if existing:
            nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
            next_num = max(nums) + 1 if nums else 1
        else:
            next_num = 1

        run_dir = os.path.join(self.base_log_dir, f'run_{next_num:03d}')
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'frames'), exist_ok=True)

        return run_dir

    def _setup_warehouse_obstacles(self):
        """Warehouse 정적 장애물 설정"""
        occ_map = self.navigator.planner.occ_map

        obstacles = [
            # Outer walls
            ((-12.0, -12.0), (-12.0, 12.0)),
            ((12.0, -12.0), (12.0, 12.0)),
            ((-12.0, 12.0), (12.0, 12.0)),
            ((-12.0, -12.0), (12.0, -12.0)),

            # Shelf 1
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

            # Shelf 5
            ((10, 4), (10, -5)),
            ((10, -5), (12, -5)),
            ((12, -5), (12, 4)),
            ((12, 4), (10, 4)),
        ]

        for obs in obstacles:
            occ_map.add_line_obstacle(obs[0], obs[1], thickness=0.1, inflation=0.3)

        rospy.loginfo(f"Loaded {len(obstacles)} obstacles")

    def agents_callback(self, msg):
        """CCTV 감지 사람 위치 수신"""
        self.humans = []
        for agent in msg.agent_states:
            human = HumanState(
                id=agent.id,
                px=agent.pose.position.x,
                py=agent.pose.position.y,
                vx=agent.twist.linear.x,
                vy=agent.twist.linear.y,
                radius=0.3
            )
            self.humans.append(human)

    def odom_callback(self, msg):
        """로봇 오도메트리 수신"""
        self.robot_state.px = msg.pose.pose.position.x
        self.robot_state.py = msg.pose.pose.position.y
        self.robot_state.vx = msg.twist.twist.linear.x
        self.robot_state.vy = msg.twist.twist.linear.y

        q = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.robot_state.theta = yaw

        if not self.got_odom:
            self.got_odom = True
            rospy.loginfo(f"Got first odom: ({self.robot_state.px:.2f}, {self.robot_state.py:.2f})")

    def goal_callback(self, msg):
        """목표 위치 수신"""
        new_goal = (msg.pose.position.x, msg.pose.position.y)

        if self.goal and abs(new_goal[0] - self.goal[0]) < 0.1 and abs(new_goal[1] - self.goal[1]) < 0.1:
            return

        self.goal = new_goal
        self.robot_state.gx = self.goal[0]
        self.robot_state.gy = self.goal[1]
        self.current_waypoint_idx = 0

        rospy.loginfo(f"=" * 40)
        rospy.loginfo(f"NEW GOAL (CCTV): ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        rospy.loginfo(f"Robot: ({self.robot_state.px:.2f}, {self.robot_state.py:.2f})")
        rospy.loginfo(f"Detected humans: {len(self.humans)}")
        rospy.loginfo(f"=" * 40)

        self._plan_path()

    def planning_loop(self, event):
        """메인 계획 루프"""
        if self.goal is None or not self.got_odom:
            return

        dist_to_goal = np.sqrt(
            (self.robot_state.px - self.goal[0])**2 +
            (self.robot_state.py - self.goal[1])**2
        )
        if dist_to_goal < 0.5:
            if self.path:
                rospy.loginfo("=" * 40)
                rospy.loginfo("GOAL REACHED (CCTV mode)!")
                rospy.loginfo("=" * 40)
                self._save_log()
                self.path = []
                self.goal = None
            return

        self._plan_path()
        self._follow_waypoints()

    def _plan_path(self):
        """CIGP로 경로 계획"""
        if self.goal is None or not self.got_odom:
            return

        self.robot_state.gx = self.goal[0]
        self.robot_state.gy = self.goal[1]

        try:
            self.navigator.update(self.robot_state, self.humans)
            self.path = self.navigator.plan_path(force_replan=True)

            if self.path:
                rospy.loginfo(f"Path OK: {len(self.path)} waypoints | {len(self.humans)} humans (CCTV)")
                self._publish_path()
                self._publish_markers()
                self._log_state()
            else:
                rospy.logwarn(f"Path FAILED")

        except Exception as e:
            rospy.logerr(f"Planning error: {e}")

    def _follow_waypoints(self):
        """웨이포인트 순차 추적"""
        if not self.path or self.current_waypoint_idx >= len(self.path):
            return

        target_wp = self.path[self.current_waypoint_idx]

        dist = np.sqrt(
            (self.robot_state.px - target_wp[0])**2 +
            (self.robot_state.py - target_wp[1])**2
        )

        if dist < self.waypoint_reach_dist:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < len(self.path):
                target_wp = self.path[self.current_waypoint_idx]
            else:
                target_wp = self.goal

        self._send_waypoint_to_movebase(target_wp)

    def _send_waypoint_to_movebase(self, waypoint):
        """웨이포인트를 move_base goal로 전송"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'odom'
        goal_msg.pose.position.x = waypoint[0]
        goal_msg.pose.position.y = waypoint[1]
        goal_msg.pose.position.z = 0.0
        goal_msg.pose.orientation.w = 1.0

        self.goal_pub.publish(goal_msg)

    def _publish_path(self):
        """경로 퍼블리시"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = rospy.Time.now()
        path_msg.header.frame_id = 'odom'

        for wp in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wp[0]
            pose.pose.position.y = wp[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    def _publish_markers(self):
        """웨이포인트 마커 퍼블리시"""
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        # 경로 라인 (노란색 - CCTV 버전 구분용)
        line_marker = Marker()
        line_marker.header.frame_id = 'odom'
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = 'cigp_cctv_path'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.15
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.pose.orientation.w = 1.0

        for wp in self.path:
            p = Point()
            p.x = wp[0]
            p.y = wp[1]
            p.z = 0.2
            line_marker.points.append(p)

        marker_array.markers.append(line_marker)

        # 목표점
        if self.goal:
            goal_marker = Marker()
            goal_marker.header.frame_id = 'odom'
            goal_marker.header.stamp = rospy.Time.now()
            goal_marker.ns = 'cigp_cctv_goal'
            goal_marker.id = 2
            goal_marker.type = Marker.SPHERE
            goal_marker.action = Marker.ADD
            goal_marker.pose.position.x = self.goal[0]
            goal_marker.pose.position.y = self.goal[1]
            goal_marker.pose.position.z = 0.3
            goal_marker.pose.orientation.w = 1.0
            goal_marker.scale.x = 0.5
            goal_marker.scale.y = 0.5
            goal_marker.scale.z = 0.5
            goal_marker.color.r = 1.0
            goal_marker.color.g = 0.0
            goal_marker.color.b = 0.0
            goal_marker.color.a = 1.0
            marker_array.markers.append(goal_marker)

        # 사람 위치 마커 (보라색 - CCTV 감지)
        for i, human in enumerate(self.humans):
            human_marker = Marker()
            human_marker.header.frame_id = 'odom'
            human_marker.header.stamp = rospy.Time.now()
            human_marker.ns = 'cctv_humans'
            human_marker.id = 100 + i
            human_marker.type = Marker.CYLINDER
            human_marker.action = Marker.ADD
            human_marker.pose.position.x = human.px
            human_marker.pose.position.y = human.py
            human_marker.pose.position.z = 0.5
            human_marker.pose.orientation.w = 1.0
            human_marker.scale.x = 0.5
            human_marker.scale.y = 0.5
            human_marker.scale.z = 1.0
            human_marker.color.r = 0.8
            human_marker.color.g = 0.0
            human_marker.color.b = 0.8
            human_marker.color.a = 0.8
            marker_array.markers.append(human_marker)

        self.marker_pub.publish(marker_array)

    def visualize_costmap(self, event):
        """코스트맵 시각화"""
        if not self.got_odom:
            return

        try:
            img_size = 480
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

            x_min, x_max = self.x_range
            y_min, y_max = self.y_range

            def world_to_img(wx, wy):
                ix = int((wx - x_min) / (x_max - x_min) * img_size)
                iy = int((y_max - wy) / (y_max - y_min) * img_size)
                return np.clip(ix, 0, img_size-1), np.clip(iy, 0, img_size-1)

            img[:] = (40, 40, 40)

            # 장애물
            occ_grid = self.navigator.planner.occ_map.grid
            for gy in range(occ_grid.shape[0]):
                for gx in range(occ_grid.shape[1]):
                    if occ_grid[gy, gx] > 0:
                        wx, wy = self.navigator.planner.occ_map.grid_to_world(gx, gy)
                        ix, iy = world_to_img(wx, wy)
                        cv2.circle(img, (ix, iy), 1, (100, 100, 100), -1)

            # Individual Space
            if self.humans:
                vis_resolution = 0.2
                x_vals = np.arange(x_min, x_max, vis_resolution)
                y_vals = np.arange(y_min, y_max, vis_resolution)
                x_grid, y_grid = np.meshgrid(x_vals, y_vals)

                is_map = np.zeros_like(x_grid)
                for human in self.humans:
                    from cigp.individual_space import IndividualSpace
                    ind_space = IndividualSpace(human)
                    is_vals = ind_space.evaluate_grid(x_grid, y_grid)
                    is_map = np.maximum(is_map, is_vals)

                for iy_grid in range(is_map.shape[0]):
                    for ix_grid in range(is_map.shape[1]):
                        is_val = is_map[iy_grid, ix_grid]
                        if is_val > 0.05:
                            wx = x_vals[ix_grid]
                            wy = y_vals[iy_grid]
                            ix, iy = world_to_img(wx, wy)
                            intensity = int(is_val * 200)
                            # 보라색 계열 (CCTV 버전 구분)
                            color = (min(255, 50 + intensity), 0, min(255, 50 + intensity))
                            cv2.circle(img, (ix, iy), 2, color, -1)

                for human in self.humans:
                    hx, hy = world_to_img(human.px, human.py)
                    cv2.circle(img, (hx, hy), 8, (255, 0, 255), -1)  # 보라색

            # 경로 (노란색)
            if self.path and len(self.path) > 1:
                pts = [world_to_img(wp[0], wp[1]) for wp in self.path]
                for i in range(len(pts) - 1):
                    cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)

            # 로봇
            rx, ry = world_to_img(self.robot_state.px, self.robot_state.py)
            cv2.circle(img, (rx, ry), 12, (255, 100, 0), -1)

            # 목표
            if self.goal:
                gx, gy = world_to_img(self.goal[0], self.goal[1])
                cv2.circle(img, (gx, gy), 12, (0, 255, 255), -1)

            # 텍스트
            cv2.putText(img, "CIGP CCTV Mode (Vision-based)", (5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(img, f"Detected: {len(self.humans)} humans", (5, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # 발행
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
            img_msg.header.stamp = rospy.Time.now()
            self.costmap_pub.publish(img_msg)

            # 저장
            frame_path = os.path.join(self.log_dir, 'frames', f'frame_{self.frame_count:05d}.png')
            cv2.imwrite(frame_path, img)
            self.frame_count += 1
            cv2.imwrite(os.path.join(self.log_dir, 'costmap_latest.png'), img)

        except Exception as e:
            rospy.logwarn(f"Visualization error: {e}")

    def _log_state(self):
        """현재 상태 로그"""
        log_entry = {
            'timestamp': time.time() - self.start_time,
            'robot': {
                'x': self.robot_state.px,
                'y': self.robot_state.py,
                'theta': self.robot_state.theta
            },
            'goal': self.goal,
            'humans': [{'id': h.id, 'x': h.px, 'y': h.py, 'vx': h.vx, 'vy': h.vy} for h in self.humans],
            'path_length': len(self.path),
            'source': 'cctv'
        }
        self.log_data.append(log_entry)

    def _save_log(self):
        """로그 저장"""
        if not self.log_data:
            return

        log_file = os.path.join(self.log_dir, 'log.json')
        with open(log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)

        rospy.loginfo(f"Log saved: {log_file}")
        self.log_data = []

    def run(self):
        """노드 실행"""
        rospy.on_shutdown(self._save_log)
        rospy.spin()


def main():
    try:
        planner = CIGPCCTVPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
