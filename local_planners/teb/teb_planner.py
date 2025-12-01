#!/usr/bin/env python3
"""
TEB (Timed Elastic Band) Local Planner Wrapper

ROS teb_local_planner를 독립 모듈로 래핑.
CIGP 글로벌 경로를 따라가면서 TEB로 cmd_vel 생성.

기존 환경 코드 수정 없이 독립적으로 동작.

Requirements:
    - ros-noetic-teb-local-planner
    - sudo apt install ros-noetic-teb-local-planner

Usage:
    from local_planners.teb.teb_planner import TEBPlannerROS

    node = TEBPlannerROS()
    node.run()
"""

import os
import sys
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class TEBConfig:
    """TEB 설정 파라미터"""
    # 로봇 설정
    max_vel_x: float = 0.8
    max_vel_x_backwards: float = 0.2
    max_vel_theta: float = 1.0
    acc_lim_x: float = 0.5
    acc_lim_theta: float = 1.0

    # 로봇 형상
    robot_radius: float = 0.3
    footprint_padding: float = 0.01

    # 목표 허용 오차
    xy_goal_tolerance: float = 0.3
    yaw_goal_tolerance: float = 0.2

    # 궤적 설정
    teb_autosize: bool = True
    dt_ref: float = 0.3
    dt_hysteresis: float = 0.1
    min_samples: int = 3
    global_plan_overwrite_orientation: bool = True
    global_plan_viapoint_sep: float = 0.5

    # 장애물 설정
    min_obstacle_dist: float = 0.3
    inflation_dist: float = 0.5
    include_costmap_obstacles: bool = True
    costmap_obstacles_behind_robot_dist: float = 1.0

    # 최적화
    no_inner_iterations: int = 5
    no_outer_iterations: int = 4
    optimization_activate: bool = True
    optimization_verbose: bool = False

    # 보행자 (human-aware)
    include_dynamic_obstacles: bool = True
    dynamic_obstacle_inflation_dist: float = 0.6


def generate_teb_params_yaml(config: TEBConfig, output_path: str):
    """TEB 파라미터 YAML 파일 생성"""
    yaml_content = f"""
# TEB Local Planner Parameters
# Auto-generated for CIGP integration

TebLocalPlannerROS:
  odom_topic: odom

  # Robot Configuration
  max_vel_x: {config.max_vel_x}
  max_vel_x_backwards: {config.max_vel_x_backwards}
  max_vel_y: 0.0
  max_vel_theta: {config.max_vel_theta}
  acc_lim_x: {config.acc_lim_x}
  acc_lim_y: 0.0
  acc_lim_theta: {config.acc_lim_theta}

  # Footprint
  footprint_model:
    type: "circular"
    radius: {config.robot_radius}

  # Goal Tolerance
  xy_goal_tolerance: {config.xy_goal_tolerance}
  yaw_goal_tolerance: {config.yaw_goal_tolerance}
  free_goal_vel: False

  # Trajectory
  teb_autosize: {str(config.teb_autosize).lower()}
  dt_ref: {config.dt_ref}
  dt_hysteresis: {config.dt_hysteresis}
  min_samples: {config.min_samples}
  global_plan_overwrite_orientation: {str(config.global_plan_overwrite_orientation).lower()}
  global_plan_viapoint_sep: {config.global_plan_viapoint_sep}
  max_global_plan_lookahead_dist: 3.0
  force_reinit_new_goal_dist: 1.0
  feasibility_check_no_poses: 5
  publish_feedback: False
  shrink_horizon_backup: True
  allow_init_with_backwards_motion: True
  exact_arc_length: False
  shrink_horizon_min_duration: 10

  # Obstacles
  min_obstacle_dist: {config.min_obstacle_dist}
  inflation_dist: {config.inflation_dist}
  include_costmap_obstacles: {str(config.include_costmap_obstacles).lower()}
  costmap_obstacles_behind_robot_dist: {config.costmap_obstacles_behind_robot_dist}
  obstacle_poses_affected: 30
  legacy_obstacle_association: False
  obstacle_association_cutoff_factor: 5.0
  obstacle_association_force_inclusion_factor: 1.5

  # Dynamic Obstacles (Human-aware)
  include_dynamic_obstacles: {str(config.include_dynamic_obstacles).lower()}
  dynamic_obstacle_inflation_dist: {config.dynamic_obstacle_inflation_dist}

  # Optimization
  no_inner_iterations: {config.no_inner_iterations}
  no_outer_iterations: {config.no_outer_iterations}
  optimization_activate: {str(config.optimization_activate).lower()}
  optimization_verbose: {str(config.optimization_verbose).lower()}
  penalty_epsilon: 0.1
  weight_max_vel_x: 2.0
  weight_max_vel_theta: 1.0
  weight_acc_lim_x: 1.0
  weight_acc_lim_theta: 1.0
  weight_kinematics_nh: 1000.0
  weight_kinematics_forward_drive: 100.0
  weight_kinematics_turning_radius: 1.0
  weight_optimaltime: 1.0
  weight_obstacle: 50.0
  weight_dynamic_obstacle: 50.0
  weight_dynamic_obstacle_inflation: 0.1
  weight_viapoint: 1.0
  weight_adapt_factor: 2.0

  # Homotopy Class Planner
  enable_homotopy_class_planning: True
  enable_multithreading: True
  simple_exploration: False
  max_number_classes: 4
  selection_cost_hysteresis: 1.0
  selection_obst_cost_scale: 100.0
  selection_viapoint_cost_scale: 1.0
  selection_alternative_time_cost: False
  roadmap_graph_no_samples: 15
  roadmap_graph_area_width: 6
  h_signature_prescaler: 1.0
  h_signature_threshold: 0.1
  obstacle_heading_threshold: 0.45
  visualize_hc_graph: False
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content)

    return output_path


class TEBPlannerROS:
    """
    TEB ROS 노드 래퍼

    CIGP 글로벌 경로를 구독하고 TEB로 cmd_vel 발행.
    기존 환경 수정 없이 독립 노드로 동작.

    move_base를 사용하지 않고 직접 TEB를 호출.
    """

    def __init__(self, config: TEBConfig = None):
        import rospy
        from geometry_msgs.msg import Twist, PoseStamped, Point
        from nav_msgs.msg import Path, Odometry
        from sensor_msgs.msg import LaserScan
        from pedsim_msgs.msg import AgentStates
        from costmap_2d.msg import ObstacleArrayMsg, ObstacleMsg
        from tf.transformations import euler_from_quaternion
        import tf2_ros

        self.rospy = rospy
        self.config = config or TEBConfig()

        # 상태
        self.robot_pos = None
        self.robot_theta = None
        self.robot_vel = (0.0, 0.0)
        self.current_goal = None
        self.current_path = []
        self.humans = []
        self.latest_scan = None

        # ROS 초기화
        rospy.init_node('teb_local_planner_node', anonymous=True)

        # 토픽 파라미터 (환경에 맞게 설정 가능)
        self.odom_topic = rospy.get_param('~odom_topic', '/p3dx/odom')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/p3dx/cmd_vel')
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.agents_topic = rospy.get_param('~agents_topic', '/pedsim_simulator/simulated_agents')

        rospy.loginfo(f"[TEB] odom: {self.odom_topic}, cmd_vel: {self.cmd_vel_topic}")

        # TEB 파라미터 설정
        self._setup_teb_params()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Subscribers
        rospy.Subscriber('/cigp/global_path', Path, self.path_callback, queue_size=1)
        rospy.Subscriber('/cigp/next_waypoint', PoseStamped, self.waypoint_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber(self.agents_topic, AgentStates, self.agents_callback, queue_size=1)

        # Publishers
        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.local_plan_pub = rospy.Publisher('/teb_local_plan', Path, queue_size=1)
        self.obstacles_pub = rospy.Publisher('/teb_obstacles', ObstacleArrayMsg, queue_size=1)

        # TEB Planner 초기화
        self.teb_planner = None
        self._init_teb_planner()

        # 제어 루프
        self.rate = rospy.Rate(10)  # 10 Hz

        rospy.loginfo("[TEB] Local planner node initialized")

    def _setup_teb_params(self):
        """TEB ROS 파라미터 설정"""
        cfg = self.config

        # Robot configuration
        self.rospy.set_param('~max_vel_x', cfg.max_vel_x)
        self.rospy.set_param('~max_vel_x_backwards', cfg.max_vel_x_backwards)
        self.rospy.set_param('~max_vel_theta', cfg.max_vel_theta)
        self.rospy.set_param('~acc_lim_x', cfg.acc_lim_x)
        self.rospy.set_param('~acc_lim_theta', cfg.acc_lim_theta)

        # Goal tolerance
        self.rospy.set_param('~xy_goal_tolerance', cfg.xy_goal_tolerance)
        self.rospy.set_param('~yaw_goal_tolerance', cfg.yaw_goal_tolerance)

        # Obstacles
        self.rospy.set_param('~min_obstacle_dist', cfg.min_obstacle_dist)
        self.rospy.set_param('~inflation_dist', cfg.inflation_dist)
        self.rospy.set_param('~include_dynamic_obstacles', cfg.include_dynamic_obstacles)
        self.rospy.set_param('~dynamic_obstacle_inflation_dist', cfg.dynamic_obstacle_inflation_dist)

    def _init_teb_planner(self):
        """TEB Planner 초기화"""
        try:
            # teb_local_planner Python API 사용 시도
            # 참고: teb_local_planner는 주로 C++ 라이브러리이므로
            # 여기서는 간단한 래퍼 또는 서비스 호출 방식 사용

            # 방법 1: move_base 없이 직접 사용 (제한적)
            # 방법 2: move_base 사용 (권장)
            # 방법 3: 간단한 Python 구현

            self.use_simple_teb = True  # 간단한 구현 사용
            self.rospy.loginfo("[TEB] Using simplified TEB implementation")

        except Exception as e:
            self.rospy.logwarn(f"[TEB] Could not initialize full TEB: {e}")
            self.use_simple_teb = True

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
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

        orientation = msg.pose.pose.orientation
        _, _, self.robot_theta = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])

    def scan_callback(self, msg):
        """LiDAR 스캔 콜백"""
        self.latest_scan = np.array(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_increment = msg.angle_increment

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

        # 동적 장애물 발행
        self._publish_dynamic_obstacles()

    def _publish_dynamic_obstacles(self):
        """동적 장애물 (보행자) 발행"""
        from costmap_2d.msg import ObstacleArrayMsg, ObstacleMsg
        from geometry_msgs.msg import Point32

        obs_msg = ObstacleArrayMsg()
        obs_msg.header.stamp = self.rospy.Time.now()
        obs_msg.header.frame_id = "odom"

        for h in self.humans:
            obs = ObstacleMsg()
            obs.id = h['id']
            obs.radius = h['radius']

            # 중심점
            pt = Point32()
            pt.x = h['pos'][0]
            pt.y = h['pos'][1]
            pt.z = 0.0
            obs.polygon.points.append(pt)

            # 속도
            obs.velocities.twist.linear.x = h['vel'][0]
            obs.velocities.twist.linear.y = h['vel'][1]

            obs_msg.obstacles.append(obs)

        self.obstacles_pub.publish(obs_msg)

    def compute_velocity_simple(self) -> Tuple[float, float]:
        """
        간단한 TEB 스타일 속도 계산

        경로 추종 + 장애물 회피의 단순화된 구현.
        실제 TEB는 최적화 기반이지만, 여기서는 휴리스틱 사용.
        """
        cfg = self.config

        if self.robot_pos is None or not self.current_path:
            return 0.0, 0.0

        # 현재 위치에서 가장 가까운 경로점 + lookahead
        closest_idx = 0
        min_dist = float('inf')
        for i, wp in enumerate(self.current_path):
            dist = np.sqrt((self.robot_pos[0] - wp[0])**2 + (self.robot_pos[1] - wp[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Lookahead 포인트 선택
        lookahead_idx = min(closest_idx + 3, len(self.current_path) - 1)
        target = self.current_path[lookahead_idx]

        # 목표 방향
        dx = target[0] - self.robot_pos[0]
        dy = target[1] - self.robot_pos[1]
        target_dist = np.sqrt(dx**2 + dy**2)
        target_angle = np.arctan2(dy, dx)

        # 각도 오차
        angle_error = target_angle - self.robot_theta
        while angle_error > np.pi:
            angle_error -= 2 * np.pi
        while angle_error < -np.pi:
            angle_error += 2 * np.pi

        # 목표 도달
        if target_dist < cfg.xy_goal_tolerance:
            if abs(angle_error) < cfg.yaw_goal_tolerance:
                return 0.0, 0.0
            else:
                return 0.0, np.sign(angle_error) * cfg.max_vel_theta * 0.5

        # 장애물 회피 (LiDAR 기반)
        obstacle_factor = 1.0
        if self.latest_scan is not None:
            # 전방 스캔
            n_points = len(self.latest_scan)
            front_start = int(n_points * 0.25)
            front_end = int(n_points * 0.75)
            front_scan = self.latest_scan[front_start:front_end]
            front_valid = front_scan[front_scan > 0.1]

            if len(front_valid) > 0:
                min_front = np.min(front_valid)
                if min_front < cfg.min_obstacle_dist:
                    obstacle_factor = 0.0  # 정지
                elif min_front < cfg.inflation_dist:
                    obstacle_factor = (min_front - cfg.min_obstacle_dist) / (cfg.inflation_dist - cfg.min_obstacle_dist)

        # 보행자 회피
        for h in self.humans:
            dist_to_human = np.sqrt(
                (self.robot_pos[0] - h['pos'][0])**2 +
                (self.robot_pos[1] - h['pos'][1])**2
            )
            safe_dist = cfg.robot_radius + h['radius'] + cfg.dynamic_obstacle_inflation_dist

            if dist_to_human < safe_dist:
                human_factor = max(0, (dist_to_human - cfg.robot_radius - h['radius']) / cfg.dynamic_obstacle_inflation_dist)
                obstacle_factor = min(obstacle_factor, human_factor)

        # 속도 계산
        if abs(angle_error) > 0.5:  # 큰 각도 오차면 회전 우선
            linear_x = cfg.max_vel_x * 0.3 * obstacle_factor
            angular_z = np.clip(angle_error * 2.0, -cfg.max_vel_theta, cfg.max_vel_theta)
        else:
            linear_x = cfg.max_vel_x * obstacle_factor * (1.0 - abs(angle_error) / np.pi)
            angular_z = np.clip(angle_error * 1.5, -cfg.max_vel_theta, cfg.max_vel_theta)

        return linear_x, angular_z

    def run(self):
        """메인 루프"""
        while not self.rospy.is_shutdown():
            if self.robot_pos is not None and self.current_path:
                # TEB 속도 계산
                linear_x, angular_z = self.compute_velocity_simple()

                # cmd_vel 발행
                cmd = Twist()
                cmd.linear.x = linear_x
                cmd.angular.z = angular_z
                self.cmd_pub.publish(cmd)

            self.rate.sleep()


class TEBWithMoveBase:
    """
    move_base + TEB 조합 사용

    기존 move_base 설정을 TEB로 교체하는 launch 파일 생성.
    """

    @staticmethod
    def generate_launch_file(output_path: str, config: TEBConfig = None):
        """TEB move_base launch 파일 생성"""
        config = config or TEBConfig()

        launch_content = f"""<?xml version="1.0"?>
<launch>
  <!-- TEB Local Planner with CIGP Global Path -->
  <!-- 기존 환경 수정 없이 독립 실행 -->

  <arg name="cmd_vel_topic" default="/cmd_vel"/>
  <arg name="odom_topic" default="/odom"/>
  <arg name="map_frame" default="odom"/>

  <!-- move_base with TEB -->
  <node pkg="move_base" type="move_base" name="move_base_teb" output="screen">
    <!-- 글로벌 플래너: CIGP 경로 사용 -->
    <param name="base_global_planner" value="global_planner/GlobalPlanner"/>
    <param name="planner_frequency" value="0.0"/>  <!-- CIGP에서 경로 받음 -->

    <!-- 로컬 플래너: TEB -->
    <param name="base_local_planner" value="teb_local_planner/TebLocalPlannerROS"/>
    <param name="controller_frequency" value="10.0"/>

    <!-- 토픽 리매핑 -->
    <remap from="cmd_vel" to="$(arg cmd_vel_topic)"/>
    <remap from="odom" to="$(arg odom_topic)"/>

    <!-- TEB 파라미터 -->
    <rosparam>
      TebLocalPlannerROS:
        odom_topic: $(arg odom_topic)
        map_frame: $(arg map_frame)

        # Robot
        max_vel_x: {config.max_vel_x}
        max_vel_x_backwards: {config.max_vel_x_backwards}
        max_vel_theta: {config.max_vel_theta}
        acc_lim_x: {config.acc_lim_x}
        acc_lim_theta: {config.acc_lim_theta}

        # Footprint
        footprint_model:
          type: circular
          radius: {config.robot_radius}

        # Goal Tolerance
        xy_goal_tolerance: {config.xy_goal_tolerance}
        yaw_goal_tolerance: {config.yaw_goal_tolerance}

        # Obstacles
        min_obstacle_dist: {config.min_obstacle_dist}
        inflation_dist: {config.inflation_dist}
        include_dynamic_obstacles: true
        dynamic_obstacle_inflation_dist: {config.dynamic_obstacle_inflation_dist}

        # Optimization
        no_inner_iterations: {config.no_inner_iterations}
        no_outer_iterations: {config.no_outer_iterations}
        weight_obstacle: 50.0
        weight_dynamic_obstacle: 50.0

        # Homotopy
        enable_homotopy_class_planning: true
        max_number_classes: 4
    </rosparam>
  </node>

  <!-- CIGP 경로를 move_base에 전달하는 브릿지 -->
  <node pkg="topic_tools" type="relay" name="cigp_to_move_base"
        args="/cigp/global_path /move_base_teb/GlobalPlanner/plan"/>

</launch>
"""
        with open(output_path, 'w') as f:
            f.write(launch_content)

        return output_path


def main():
    """ROS 노드 실행"""
    try:
        node = TEBPlannerROS()
        node.run()
    except Exception as e:
        print(f"[TEB] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
