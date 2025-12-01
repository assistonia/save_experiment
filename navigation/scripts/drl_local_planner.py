#!/usr/bin/env python3
"""
DRL Local Planner Bridge
TD3 강화학습 모델을 Local Planner로 사용하기 위한 브릿지

입력: Velodyne PointCloud2 + Goal 위치
출력: cmd_vel (linear.x, angular.z)
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

# 즉시 출력을 위한 설정
sys.stdout = sys.stderr

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from actionlib_msgs.msg import GoalStatusArray


class Actor(nn.Module):
    """TD3 Actor Network"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class TD3Agent:
    """TD3 Agent for inference"""
    def __init__(self, state_dim, action_dim, device):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.device = device

    def get_action(self, state):
        state = torch.Tensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load(f"{directory}/{filename}_actor.pth", map_location=self.device)
        )
        self.actor.eval()


class DRLLocalPlanner:
    def __init__(self):
        print("[DRL] Initializing ROS node...", flush=True)
        rospy.init_node('drl_local_planner', anonymous=True)
        print("[DRL] ROS node initialized", flush=True)

        # Parameters
        self.environment_dim = 20  # LiDAR 샘플 수
        self.robot_dim = 4  # [distance_to_goal, angle_to_goal, linear_vel, angular_vel]
        self.state_dim = self.environment_dim + self.robot_dim
        self.action_dim = 2  # [linear_vel, angular_vel]

        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.0
        self.goal_reached_dist = 0.5

        # State variables
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.goal_x = None
        self.goal_y = None
        self.lidar_data = np.ones(self.environment_dim) * 10.0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"[DRL] Using device: {self.device}")

        # Load model
        self.agent = TD3Agent(self.state_dim, self.action_dim, self.device)
        model_path = rospy.get_param("~model_path", "/root/DRL-robot-navigation/TD3/pytorch_models")
        model_name = rospy.get_param("~model_name", "TD3_velodyne")

        print(f"[DRL] Loading model from {model_path}/{model_name}", flush=True)
        try:
            self.agent.load(model_name, model_path)
            print(f"[DRL] Model loaded successfully!", flush=True)
        except Exception as e:
            print(f"[DRL] Failed to load model: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise

        # Publishers - 직접 /p3dx/cmd_vel로 발행 (좌우 반전 적용)
        self.cmd_pub = rospy.Publisher('/p3dx/cmd_vel', Twist, queue_size=1)

        # Subscribers - Velodyne PointCloud2 사용 (DRL 학습 환경과 동일)
        rospy.Subscriber('/p3dx/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/velodyne_points', PointCloud2, self.velodyne_callback, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        # Control rate
        self.rate = rospy.Rate(10)

        print("[DRL] Local Planner initialized", flush=True)

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

        # Quaternion to yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.linear_vel = msg.twist.twist.linear.x
        self.angular_vel = msg.twist.twist.angular.z

    def velodyne_callback(self, msg):
        """Velodyne PointCloud2를 거리 데이터로 변환 (DRL 학습 환경과 동일)"""
        data = list(pc2.read_points(msg, skip_nans=False, field_names=("x", "y", "z")))

        # 각 각도별 최소 거리 계산
        velodyne_data = np.ones(self.environment_dim) * 10.0

        for point in data:
            if len(point) < 3:
                continue
            x, y, z = point[0], point[1], point[2]

            # z 높이 필터링 (로봇 높이 범위만)
            if z > -0.2 and z < 1.0:
                dist = math.sqrt(x*x + y*y)
                angle = math.atan2(y, x)

                # 각도를 인덱스로 변환 (-pi ~ pi -> 0 ~ environment_dim-1)
                idx = int((angle + math.pi) / (2 * math.pi) * self.environment_dim)
                idx = max(0, min(self.environment_dim - 1, idx))

                # 최소 거리 저장
                if dist < velodyne_data[idx]:
                    velodyne_data[idx] = dist

        self.lidar_data = velodyne_data

    def goal_callback(self, msg):
        self.goal_x = msg.pose.position.x
        self.goal_y = msg.pose.position.y
        print(f"[DRL] New goal received: ({self.goal_x:.2f}, {self.goal_y:.2f})", flush=True)

    def get_state(self):
        """현재 상태 벡터 생성"""
        if self.goal_x is None or self.goal_y is None:
            return None

        # Distance and angle to goal
        dx = self.goal_x - self.odom_x
        dy = self.goal_y - self.odom_y
        distance = math.sqrt(dx*dx + dy*dy)

        # Angle to goal (relative to robot heading)
        goal_angle = math.atan2(dy, dx)
        angle_diff = goal_angle - self.odom_yaw

        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Robot state: [distance, angle, linear_vel, angular_vel]
        robot_state = np.array([
            distance,
            angle_diff,
            self.linear_vel,
            self.angular_vel
        ])

        # Full state
        state = np.concatenate([self.lidar_data, robot_state])
        return state

    def is_goal_reached(self):
        if self.goal_x is None or self.goal_y is None:
            return False
        dx = self.goal_x - self.odom_x
        dy = self.goal_y - self.odom_y
        return math.sqrt(dx*dx + dy*dy) < self.goal_reached_dist

    def run(self):
        print("[DRL] Starting control loop", flush=True)
        loop_count = 0

        while not rospy.is_shutdown():
            loop_count += 1
            state = self.get_state()

            if state is None:
                # No goal, stop
                if loop_count % 50 == 0:  # 5초마다 한 번씩 출력
                    print(f"[DRL] Waiting for goal... (goal_x={self.goal_x}, goal_y={self.goal_y})", flush=True)
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            if self.is_goal_reached():
                rospy.loginfo("[DRL] Goal reached!")
                self.goal_x = None
                self.goal_y = None
                cmd = Twist()
                self.cmd_pub.publish(cmd)
                self.rate.sleep()
                continue

            # Get action from TD3
            action = self.agent.get_action(state)

            # Debug: 상태와 액션 출력
            distance = state[-4]
            angle = state[-3]
            rospy.loginfo(f"[DRL] dist={distance:.2f}, angle={angle:.2f}rad, action=[{action[0]:.3f}, {action[1]:.3f}]")

            # Convert action to cmd_vel
            # action[0]: linear velocity [-1, 1] -> [0, max_linear_vel]
            # action[1]: angular velocity [-1, 1] -> [-max_angular_vel, max_angular_vel]
            cmd = Twist()
            cmd.linear.x = (action[0] + 1) / 2 * self.max_linear_vel
            # p3dx 로봇은 angular.z가 이미 반전됨 (+ = 시계방향)
            # DRL 모델은 표준 ROS 컨벤션으로 학습됨 (+ = 반시계방향)
            # 따라서 반전 없이 그대로 사용
            cmd.angular.z = action[1] * self.max_angular_vel

            self.cmd_pub.publish(cmd)
            self.rate.sleep()


if __name__ == '__main__':
    print("[DRL] Starting DRL Local Planner script...", flush=True)
    try:
        planner = DRLLocalPlanner()
        planner.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        print(f"[DRL] Fatal error: {e}", flush=True)
        import traceback
        traceback.print_exc()
