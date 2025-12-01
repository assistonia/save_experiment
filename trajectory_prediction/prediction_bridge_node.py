#!/usr/bin/env python3
"""
Trajectory Prediction ROS Bridge Node

ROS PedSim 시뮬레이터의 Ground Truth 데이터를 받아
SingularTrajectory 모델로 실시간 예측 후 시각화.

기존 환경 코드를 수정하지 않고 독립적으로 동작.

Subscribe:
    - /pedsim_simulator/simulated_agents (AgentStates): 보행자 GT 상태

Publish:
    - /trajectory_prediction/predicted_paths (MarkerArray): 예측 경로 시각화
    - /trajectory_prediction/observation_paths (MarkerArray): 관측 경로 시각화
    - /trajectory_prediction/agent_markers (MarkerArray): 에이전트 마커

Logs:
    - trajectory_prediction_logs/run_XXX/frames/frame_XXXXX.png
    - trajectory_prediction_logs/run_XXX/log.json
    - trajectory_prediction_logs/run_XXX/prediction_latest.png
"""

import sys
import os
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# trajectory_prediction 모듈 경로
MODULE_PATH = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.dirname(MODULE_PATH)
if ENV_PATH not in sys.path:
    sys.path.insert(0, ENV_PATH)

import rospy
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from pedsim_msgs.msg import AgentStates

from trajectory_prediction.predictor import TrajectoryPredictor
from trajectory_prediction.prediction_config import PredictionConfig


class TrajectoryPredictionBridgeNode:
    """경로 예측 ROS 브릿지 노드"""

    def __init__(self):
        rospy.init_node('trajectory_prediction_bridge', anonymous=True)

        # 파라미터 로드
        self.load_params()

        # 설정 및 예측기 초기화
        self.config = PredictionConfig(self.scenario)
        self.predictor = TrajectoryPredictor(self.config)

        # 모델 로드 (시작 시)
        rospy.loginfo("[TrajPred] Loading model...")
        if self.predictor.load_model():
            rospy.loginfo("[TrajPred] Model loaded successfully!")
        else:
            rospy.logwarn("[TrajPred] Failed to load model. Will retry on first prediction.")

        # 상태 변수
        self.current_time = 0.0
        self.last_prediction_time = 0.0
        self.frame_count = 0
        self.prediction_count = 0
        self.current_agents = []

        # 색상 팔레트 (에이전트별)
        self.colors = [
            (0.12, 0.47, 0.71, 1.0),  # Blue
            (1.0, 0.5, 0.05, 1.0),    # Orange
            (0.17, 0.63, 0.17, 1.0),  # Green
            (0.84, 0.15, 0.16, 1.0),  # Red
            (0.58, 0.40, 0.74, 1.0),  # Purple
            (0.55, 0.34, 0.29, 1.0),  # Brown
            (0.89, 0.47, 0.76, 1.0),  # Pink
            (0.50, 0.50, 0.50, 1.0),  # Gray
        ]

        # 로깅 설정
        self.log_data = []
        self.log_dir = None
        self.frames_dir = None
        if self.enable_logging:
            self.setup_logging()

        # Publishers
        self.pred_path_pub = rospy.Publisher(
            '/trajectory_prediction/predicted_paths',
            MarkerArray, queue_size=1
        )
        self.obs_path_pub = rospy.Publisher(
            '/trajectory_prediction/observation_paths',
            MarkerArray, queue_size=1
        )
        self.agent_marker_pub = rospy.Publisher(
            '/trajectory_prediction/agent_markers',
            MarkerArray, queue_size=1
        )

        # Subscribers
        rospy.Subscriber(
            '/pedsim_simulator/simulated_agents',
            AgentStates,
            self.agents_callback,
            queue_size=1
        )

        # 타이머 (예측 및 시각화)
        self.prediction_timer = rospy.Timer(
            rospy.Duration(1.0 / self.prediction_rate),
            self.prediction_callback
        )

        # 종료 시 로그 저장
        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo(f"[TrajPred] Node initialized")
        rospy.loginfo(f"  Scenario: {self.scenario}")
        rospy.loginfo(f"  Prediction rate: {self.prediction_rate} Hz")
        rospy.loginfo(f"  Obs len: {self.config.obs_len}, Pred len: {self.config.pred_len}")
        rospy.loginfo(f"  Logging: {self.enable_logging}")
        if self.enable_logging:
            rospy.loginfo(f"  Log dir: {self.log_dir}")

    def load_params(self):
        """ROS 파라미터 로드"""
        self.scenario = rospy.get_param('~scenario', 'warehouse')
        self.prediction_rate = rospy.get_param('~prediction_rate', 2.5)
        self.show_samples = rospy.get_param('~show_samples', 5)  # 표시할 샘플 수
        self.z_height = rospy.get_param('~z_height', 0.1)  # 마커 높이

        # 로깅 파라미터
        self.enable_logging = rospy.get_param('~enable_logging', True)
        # 기본 경로: 모듈 위치 기준 (도커/호스트 모두 호환)
        default_log_dir = os.path.join(ENV_PATH, 'trajectory_prediction_logs')
        self.log_base_dir = rospy.get_param('~log_base_dir', default_log_dir)
        self.save_every_n_frames = rospy.get_param('~save_every_n_frames', 1)

    def setup_logging(self):
        """로깅 디렉토리 설정"""
        # 기존 run 폴더 확인하여 다음 번호 결정
        os.makedirs(self.log_base_dir, exist_ok=True)

        existing_runs = [d for d in os.listdir(self.log_base_dir)
                        if d.startswith('run_') and os.path.isdir(os.path.join(self.log_base_dir, d))]

        if existing_runs:
            max_num = max([int(d.split('_')[1]) for d in existing_runs])
            run_num = max_num + 1
        else:
            run_num = 1

        self.log_dir = os.path.join(self.log_base_dir, f'run_{run_num:03d}')
        self.frames_dir = os.path.join(self.log_dir, 'frames')

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.frames_dir, exist_ok=True)

        rospy.loginfo(f"[TrajPred] Logging to: {self.log_dir}")

        # 메타데이터 저장
        meta = {
            'start_time': datetime.now().isoformat(),
            'scenario': self.scenario,
            'prediction_rate': self.prediction_rate,
            'obs_len': self.config.obs_len,
            'pred_len': self.config.pred_len,
            'num_samples': self.config.num_samples
        }
        with open(os.path.join(self.log_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def agents_callback(self, msg: AgentStates):
        """보행자 상태 콜백 (Ground Truth)"""
        self.current_time = msg.header.stamp.to_sec()
        self.frame_count += 1

        # 에이전트 위치 업데이트
        agents = []
        for agent in msg.agent_states:
            agents.append({
                'id': agent.id,
                'x': agent.pose.position.x,
                'y': agent.pose.position.y,
                'vx': agent.twist.linear.x,
                'vy': agent.twist.linear.y,
                'timestamp': self.current_time
            })

        self.current_agents = agents
        self.predictor.update_agents_batch(agents)

        # 에이전트 마커 발행
        self.publish_agent_markers(agents)

    def prediction_callback(self, event):
        """주기적 예측 및 시각화"""
        # 디버그: 현재 상태 출력
        total_agents = self.predictor.get_agent_count()
        valid_agents = self.predictor.get_valid_agent_count()

        if total_agents > 0:
            rospy.loginfo_throttle(3.0,
                f"[TrajPred] Agents: {total_agents} total, {valid_agents} valid (need {self.config.obs_len} frames)")

        # 예측 실행
        predictions = self.predictor.predict()

        if not predictions:
            return

        self.prediction_count += 1

        # 시각화 발행
        self.publish_predictions(predictions)
        self.publish_observations(predictions)

        # 로깅
        if self.enable_logging:
            self.log_prediction(predictions)

        # 로그
        valid_count = len(predictions)
        total_count = self.predictor.get_agent_count()
        rospy.loginfo_throttle(5.0,
            f"[TrajPred] Predicting {valid_count}/{total_count} agents"
        )

    def log_prediction(self, predictions: dict):
        """예측 결과 로깅"""
        # 로그 데이터 저장
        log_entry = {
            'timestamp': self.current_time,
            'frame': self.frame_count,
            'prediction_id': self.prediction_count,
            'agents': [],
            'predictions': {}
        }

        for agent in self.current_agents:
            log_entry['agents'].append({
                'id': agent['id'],
                'x': agent['x'],
                'y': agent['y'],
                'vx': agent.get('vx', 0),
                'vy': agent.get('vy', 0)
            })

        for agent_id, pred_data in predictions.items():
            log_entry['predictions'][str(agent_id)] = {
                'obs_traj': pred_data['obs_traj'].tolist(),
                'pred_best': pred_data['pred_best'].tolist(),
                'pred_mean': pred_data['pred_mean'].tolist()
            }

        self.log_data.append(log_entry)

        # 이미지 저장
        if self.prediction_count % self.save_every_n_frames == 0:
            self.save_frame_image(predictions)

    def save_frame_image(self, predictions: dict):
        """프레임 이미지 저장"""
        fig, ax = plt.subplots(figsize=(12, 12))

        # 배경 설정
        ax.set_xlim(self.config.x_range[0] - 1, self.config.x_range[1] + 1)
        # Y축 반전: 위가 y_min (-12), 아래가 y_max (12) - 탑다운 카메라 뷰 기준
        ax.set_ylim(self.config.y_range[1] + 1, self.config.y_range[0] - 1)
        ax.set_aspect('equal')
        ax.set_facecolor('#f0f0f0')

        # 그리드
        ax.grid(True, linestyle='--', alpha=0.3)

        # 장애물 그리기
        for obs in self.config.obstacles:
            rect = Rectangle(
                (obs['x_min'], obs['y_min']),
                obs['x_max'] - obs['x_min'],
                obs['y_max'] - obs['y_min'],
                linewidth=1, edgecolor='#555555',
                facecolor='#888888', alpha=0.6
            )
            ax.add_patch(rect)

        # 벽 그리기
        walls = self.config.walls
        ax.plot([walls['x_min'], walls['x_max']], [walls['y_min'], walls['y_min']], 'k-', linewidth=2)
        ax.plot([walls['x_min'], walls['x_max']], [walls['y_max'], walls['y_max']], 'k-', linewidth=2)
        ax.plot([walls['x_min'], walls['x_min']], [walls['y_min'], walls['y_max']], 'k-', linewidth=2)
        ax.plot([walls['x_max'], walls['x_max']], [walls['y_min'], walls['y_max']], 'k-', linewidth=2)

        # 에이전트 및 예측 그리기
        for agent in self.current_agents:
            agent_id = agent['id']
            color = self.colors[agent_id % len(self.colors)]
            x, y = agent['x'], agent['y']

            # 현재 위치
            circle = Circle((x, y), 0.3, color=color, alpha=0.9, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y + 0.5, f'P{agent_id}', fontsize=9, ha='center',
                   fontweight='bold', color=color)

            # 예측이 있으면 그리기
            if agent_id in predictions:
                pred_data = predictions[agent_id]
                obs_traj = pred_data['obs_traj']
                pred_best = pred_data['pred_best']
                pred_samples = pred_data['pred_samples']

                # 관측 궤적 (회색, 과거→현재)
                if len(obs_traj) > 1:
                    ax.plot(obs_traj[:, 0], obs_traj[:, 1], color='gray',
                           linewidth=2, alpha=0.5, linestyle='--')
                    # 관측 시작점 (과거) - 작은 원
                    ax.scatter(obs_traj[0, 0], obs_traj[0, 1], c='gray', s=30,
                              marker='o', alpha=0.5, zorder=8)
                    # 관측→예측 연결 화살표 (현재 위치에서 예측 시작)
                    if len(pred_best) > 0:
                        dx = pred_best[0, 0] - obs_traj[-1, 0]
                        dy = pred_best[0, 1] - obs_traj[-1, 1]
                        ax.annotate('', xy=(pred_best[0, 0], pred_best[0, 1]),
                                   xytext=(obs_traj[-1, 0], obs_traj[-1, 1]),
                                   arrowprops=dict(arrowstyle='->', color=color, lw=1.5, alpha=0.7))

                # 다른 샘플들 (얇게, 먼저 그리기)
                for sample_idx in range(1, min(self.show_samples, len(pred_samples))):
                    sample = pred_samples[sample_idx]
                    ax.plot(sample[:, 0], sample[:, 1], color=color,
                           linewidth=1, alpha=0.15)

                # Best 예측 궤적 (굵게)
                ax.plot(pred_best[:, 0], pred_best[:, 1], color=color,
                       linewidth=2.5, alpha=0.9)

                # 예측 끝점 (미래)
                ax.scatter(pred_best[-1, 0], pred_best[-1, 1], c=[color[:3]], s=100,
                          marker='*', zorder=11, edgecolors='white', linewidths=1)

        # 제목
        ax.set_title(
            f'Trajectory Prediction | Frame {self.frame_count} | t={self.current_time:.2f}s\n'
            f'Predicting {len(predictions)} agents (obs={self.config.obs_len}, pred={self.config.pred_len})',
            fontsize=12, fontweight='bold'
        )

        # 범례
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='Observation'),
            Line2D([0], [0], color='blue', linewidth=2.5, label='Prediction (best)'),
            Line2D([0], [0], color='blue', linewidth=1, alpha=0.3, label='Prediction (samples)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        # 저장
        frame_path = os.path.join(self.frames_dir, f'frame_{self.prediction_count:05d}.png')
        plt.savefig(frame_path, dpi=100, bbox_inches='tight', facecolor='white')

        # 최신 이미지도 저장
        latest_path = os.path.join(self.log_dir, 'prediction_latest.png')
        plt.savefig(latest_path, dpi=100, bbox_inches='tight', facecolor='white')

        plt.close(fig)

    def on_shutdown(self):
        """종료 시 로그 저장"""
        if self.enable_logging and self.log_data:
            log_path = os.path.join(self.log_dir, 'log.json')
            with open(log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            rospy.loginfo(f"[TrajPred] Saved {len(self.log_data)} log entries to {log_path}")

            # 비디오 생성 시도
            self.create_video()

    def create_video(self):
        """프레임들을 비디오로 합성"""
        try:
            import subprocess
            video_path = os.path.join(self.log_dir, 'prediction_video.mp4')
            frame_pattern = os.path.join(self.frames_dir, 'frame_%05d.png')

            cmd = [
                'ffmpeg', '-y',
                '-framerate', str(int(self.prediction_rate)),
                '-i', frame_pattern,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                video_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                rospy.loginfo(f"[TrajPred] Created video: {video_path}")
            else:
                rospy.logwarn(f"[TrajPred] Failed to create video: {result.stderr}")
        except Exception as e:
            rospy.logwarn(f"[TrajPred] Video creation failed: {e}")

    def publish_predictions(self, predictions: dict):
        """예측 경로 시각화 발행"""
        marker_array = MarkerArray()
        marker_id = 0

        for agent_id, pred_data in predictions.items():
            color = self.get_agent_color(agent_id)
            pred_samples = pred_data['pred_samples']  # (num_samples, pred_len, 2)
            pred_best = pred_data['pred_best']        # (pred_len, 2)
            obs_traj = pred_data['obs_traj']          # (obs_len, 2)

            # Best prediction (굵은 선)
            marker = self.create_line_marker(
                marker_id=marker_id,
                points=pred_best,
                color=(*color[:3], 0.9),
                scale=0.15,
                ns='pred_best'
            )
            marker.header.stamp = rospy.Time.now()
            marker_array.markers.append(marker)
            marker_id += 1

            # 연결선 (관측 끝 -> 예측 시작)
            if len(obs_traj) > 0:
                connection = np.array([obs_traj[-1], pred_best[0]])
                conn_marker = self.create_line_marker(
                    marker_id=marker_id,
                    points=connection,
                    color=(*color[:3], 0.5),
                    scale=0.08,
                    ns='pred_connection'
                )
                marker_array.markers.append(conn_marker)
                marker_id += 1

            # Other samples (얇은 선)
            for sample_idx in range(1, min(self.show_samples, len(pred_samples))):
                sample = pred_samples[sample_idx]
                marker = self.create_line_marker(
                    marker_id=marker_id,
                    points=sample,
                    color=(*color[:3], 0.2),
                    scale=0.05,
                    ns='pred_samples'
                )
                marker_array.markers.append(marker)
                marker_id += 1

            # 예측 끝점 마커
            end_marker = self.create_sphere_marker(
                marker_id=marker_id,
                position=pred_best[-1],
                color=(*color[:3], 0.9),
                scale=0.25,
                ns='pred_endpoint'
            )
            marker_array.markers.append(end_marker)
            marker_id += 1

        self.pred_path_pub.publish(marker_array)

    def publish_observations(self, predictions: dict):
        """관측 경로 시각화 발행"""
        marker_array = MarkerArray()
        marker_id = 0

        for agent_id, pred_data in predictions.items():
            color = self.get_agent_color(agent_id)
            obs_traj = pred_data['obs_traj']  # (obs_len, 2)

            # 관측 경로 (점선 효과를 위해 약간 투명하게)
            marker = self.create_line_marker(
                marker_id=marker_id,
                points=obs_traj,
                color=(0.5, 0.5, 0.5, 0.7),  # 회색
                scale=0.08,
                ns='observation'
            )
            marker_array.markers.append(marker)
            marker_id += 1

            # Trail dots
            for i, point in enumerate(obs_traj):
                alpha = 0.3 + 0.5 * (i / len(obs_traj))
                dot_marker = self.create_sphere_marker(
                    marker_id=marker_id,
                    position=point,
                    color=(0.5, 0.5, 0.5, alpha),
                    scale=0.1,
                    ns='obs_trail'
                )
                marker_array.markers.append(dot_marker)
                marker_id += 1

        self.obs_path_pub.publish(marker_array)

    def publish_agent_markers(self, agents: list):
        """에이전트 현재 위치 마커 발행"""
        marker_array = MarkerArray()

        for i, agent in enumerate(agents):
            agent_id = agent['id']
            color = self.get_agent_color(agent_id)

            # 에이전트 원형 마커
            marker = Marker()
            marker.header = Header()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = 'odom'
            marker.ns = 'agents'
            marker.id = agent_id
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = agent['x']
            marker.pose.position.y = agent['y']
            marker.pose.position.z = self.z_height
            marker.pose.orientation.w = 1.0
            marker.scale = Vector3(x=0.4, y=0.4, z=0.3)
            marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=0.8)
            marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(marker)

            # 에이전트 ID 텍스트
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = 'agent_labels'
            text_marker.id = agent_id + 1000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = agent['x']
            text_marker.pose.position.y = agent['y']
            text_marker.pose.position.z = self.z_height + 0.5
            text_marker.scale.z = 0.3
            text_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text_marker.text = f"P{agent_id}"
            text_marker.lifetime = rospy.Duration(0.5)
            marker_array.markers.append(text_marker)

        self.agent_marker_pub.publish(marker_array)

    def create_line_marker(self, marker_id: int, points: np.ndarray,
                           color: tuple, scale: float, ns: str) -> Marker:
        """라인 마커 생성"""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'odom'
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        marker.lifetime = rospy.Duration(1.0 / self.prediction_rate * 2)

        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = self.z_height
            marker.points.append(p)

        return marker

    def create_sphere_marker(self, marker_id: int, position: np.ndarray,
                             color: tuple, scale: float, ns: str) -> Marker:
        """구 마커 생성"""
        marker = Marker()
        marker.header = Header()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = 'odom'
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = self.z_height
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=scale, y=scale, z=scale)
        marker.color = ColorRGBA(r=color[0], g=color[1], b=color[2], a=color[3])
        marker.lifetime = rospy.Duration(1.0 / self.prediction_rate * 2)

        return marker

    def get_agent_color(self, agent_id: int) -> tuple:
        """에이전트 ID에 따른 색상 반환"""
        return self.colors[agent_id % len(self.colors)]

    def run(self):
        """노드 실행"""
        rospy.loginfo("[TrajPred] Node running...")
        rospy.spin()


def main():
    try:
        node = TrajectoryPredictionBridgeNode()
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
