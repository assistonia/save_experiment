#!/usr/bin/env python3
"""
CCTV Human Publisher for ROS

CCTV 이미지에서 사람을 감지하고 Homography로 월드 좌표 변환 후
AgentStates 메시지로 발행하여 CIGP에서 사용할 수 있게 함.

Subscribe:
    - /cctv_0/image_raw ~ /cctv_3/image_raw: CCTV 이미지

Publish:
    - /cctv/detected_agents: 감지된 사람 위치 (AgentStates 형식)
"""

import sys
import os
import numpy as np
import time
from collections import defaultdict

# 현재 디렉토리를 path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Twist
from pedsim_msgs.msg import AgentState, AgentStates
from cv_bridge import CvBridge

from detector import PersonDetector
from homography import HomographyManager
from warehouse_config import is_in_valid_region


class CCTVHumanPublisher:
    """CCTV 기반 사람 감지 및 ROS 발행"""

    def __init__(self):
        rospy.init_node('cctv_human_publisher', anonymous=True)

        # YOLO 감지기
        rospy.loginfo("Loading YOLO detector...")
        self.detector = PersonDetector()

        # Homography 매니저 (캘리브레이션 로드)
        rospy.loginfo("Loading Homography calibration...")
        self.homography = HomographyManager()
        self.homography.load_all()

        # Warehouse 유효 영역 체크 함수
        self.is_valid_position = is_in_valid_region

        # CV Bridge
        self.bridge = CvBridge()

        # 각 CCTV별 최신 감지 결과
        self.cctv_detections = {0: [], 1: [], 2: [], 3: []}
        self.cctv_timestamps = {0: 0, 1: 0, 2: 0, 3: 0}

        # 히스토리 (속도 계산용)
        self.position_history = defaultdict(list)  # id -> [(time, x, y), ...]
        self.history_max_len = 10  # 더 긴 히스토리

        # 다음 사람 ID
        self.next_person_id = 0

        # 안정화: 마지막으로 발행한 사람들 (사라짐 지연용)
        self.last_published_agents = {}  # id -> {'x', 'y', 'vx', 'vy', 'last_seen', 'confidence'}
        self.disappear_delay = 2.0  # 2초간 유지

        # 속도 스무딩
        self.velocity_history = defaultdict(list)  # id -> [(vx, vy), ...]
        self.velocity_smooth_len = 5

        # Publisher
        self.agents_pub = rospy.Publisher('/cctv/detected_agents', AgentStates, queue_size=1)

        # Subscribers (4개 CCTV)
        for i in range(4):
            rospy.Subscriber(f'/cctv_{i}/image_raw', Image,
                           self.image_callback, callback_args=i, queue_size=1)

        # 메인 루프 타이머 (10Hz)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.publish_agents)

        rospy.loginfo("=" * 60)
        rospy.loginfo("CCTV Human Publisher Started")
        rospy.loginfo("=" * 60)
        rospy.loginfo("Subscribed to: /cctv_0~3/image_raw")
        rospy.loginfo("Publishing to: /cctv/detected_agents")
        rospy.loginfo("=" * 60)

    def image_callback(self, msg, cctv_id):
        """CCTV 이미지 수신 및 처리"""
        try:
            # ROS Image -> OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # YOLO 감지
            detections = self.detector.detect(cv_image)

            # 월드 좌표로 변환
            world_positions = []
            for det in detections:
                # Detection은 dataclass - 속성으로 접근
                foot_pixel = det.foot_pixel

                # Homography 변환 (리스트로 전달)
                world_coords = self.homography.transform(cctv_id, [foot_pixel])
                if world_coords and len(world_coords) > 0:
                    wx, wy = world_coords[0]
                    # 유효 영역 체크 (선반 내부 제외)
                    if self.is_valid_position(wx, wy):
                        world_positions.append({
                            'x': wx,
                            'y': wy,
                            'confidence': det.confidence
                        })

            # 저장
            self.cctv_detections[cctv_id] = world_positions
            self.cctv_timestamps[cctv_id] = time.time()

        except Exception as e:
            rospy.logwarn(f"CCTV {cctv_id} processing error: {e}")

    def merge_detections(self):
        """여러 CCTV 감지 결과 병합 (중복 제거)"""
        current_time = time.time()
        all_detections = []

        # 최근 0.5초 이내 감지만 사용
        for cctv_id in range(4):
            if current_time - self.cctv_timestamps[cctv_id] < 0.5:
                for det in self.cctv_detections[cctv_id]:
                    all_detections.append(det)

        if not all_detections:
            return []

        # 거리 기반 클러스터링 (1.0m 이내는 같은 사람)
        merged = []
        used = [False] * len(all_detections)

        for i, det in enumerate(all_detections):
            if used[i]:
                continue

            cluster = [det]
            used[i] = True

            for j, other in enumerate(all_detections):
                if used[j]:
                    continue
                dist = np.sqrt((det['x'] - other['x'])**2 + (det['y'] - other['y'])**2)
                if dist < 1.0:
                    cluster.append(other)
                    used[j] = True

            # 클러스터 중심 계산 (confidence 가중 평균)
            total_conf = sum(d['confidence'] for d in cluster)
            avg_x = sum(d['x'] * d['confidence'] for d in cluster) / total_conf
            avg_y = sum(d['y'] * d['confidence'] for d in cluster) / total_conf

            merged.append({
                'x': avg_x,
                'y': avg_y,
                'confidence': max(d['confidence'] for d in cluster)
            })

        return merged

    def assign_ids_and_compute_velocity(self, detections):
        """ID 할당 및 속도 계산"""
        current_time = time.time()
        results = []

        # 이전 프레임 위치와 매칭 (간단한 nearest neighbor)
        # 실제로는 Hungarian algorithm 등 사용하면 더 좋음

        for det in detections:
            # 가장 가까운 기존 ID 찾기
            best_id = None
            best_dist = 2.0  # 2m 이내만 같은 사람으로 판단

            for person_id, history in self.position_history.items():
                if history:
                    last_time, last_x, last_y = history[-1]
                    if current_time - last_time < 1.0:  # 1초 이내
                        dist = np.sqrt((det['x'] - last_x)**2 + (det['y'] - last_y)**2)
                        if dist < best_dist:
                            best_dist = dist
                            best_id = person_id

            # 새 ID 할당
            if best_id is None:
                best_id = self.next_person_id
                self.next_person_id += 1

            # 히스토리 업데이트
            self.position_history[best_id].append((current_time, det['x'], det['y']))
            if len(self.position_history[best_id]) > self.history_max_len:
                self.position_history[best_id].pop(0)

            # 속도 계산 (여러 프레임 평균 사용)
            vx, vy = 0.0, 0.0
            history = self.position_history[best_id]
            if len(history) >= 3:
                # 최근 3개 프레임으로 평균 속도 계산
                velocities = []
                for i in range(len(history) - 1, max(0, len(history) - 4), -1):
                    t1, x1, y1 = history[i - 1]
                    t2, x2, y2 = history[i]
                    dt = t2 - t1
                    if dt > 0.01:
                        vel_x = (x2 - x1) / dt
                        vel_y = (y2 - y1) / dt
                        speed = np.sqrt(vel_x**2 + vel_y**2)
                        if speed <= 2.0:  # 2m/s 이하만 유효 (사람 보행속도)
                            velocities.append((vel_x, vel_y))

                if velocities:
                    vx = sum(v[0] for v in velocities) / len(velocities)
                    vy = sum(v[1] for v in velocities) / len(velocities)

            # 속도 스무딩 히스토리에 추가
            self.velocity_history[best_id].append((vx, vy))
            if len(self.velocity_history[best_id]) > self.velocity_smooth_len:
                self.velocity_history[best_id].pop(0)

            # 스무딩된 속도 계산
            if self.velocity_history[best_id]:
                smooth_vx = sum(v[0] for v in self.velocity_history[best_id]) / len(self.velocity_history[best_id])
                smooth_vy = sum(v[1] for v in self.velocity_history[best_id]) / len(self.velocity_history[best_id])
            else:
                smooth_vx, smooth_vy = 0.0, 0.0

            results.append({
                'id': best_id,
                'x': det['x'],
                'y': det['y'],
                'vx': smooth_vx,
                'vy': smooth_vy
            })

        # 오래된 히스토리 정리
        stale_ids = []
        for person_id, history in self.position_history.items():
            if history and current_time - history[-1][0] > 2.0:
                stale_ids.append(person_id)
        for pid in stale_ids:
            del self.position_history[pid]

        return results

    def publish_agents(self, event):
        """감지 결과를 AgentStates로 발행 (사라짐 지연 포함)"""
        current_time = time.time()

        # 감지 병합
        merged = self.merge_detections()

        # ID 할당 및 속도 계산
        agents_data = self.assign_ids_and_compute_velocity(merged)

        # 현재 감지된 ID들
        current_ids = set()

        # 현재 감지된 사람들 업데이트
        for agent in agents_data:
            agent_id = agent['id']
            current_ids.add(agent_id)
            self.last_published_agents[agent_id] = {
                'x': agent['x'],
                'y': agent['y'],
                'vx': agent['vx'],
                'vy': agent['vy'],
                'last_seen': current_time,
                'confidence': 1.0
            }

        # 사라진 사람들 처리 (지연 후 제거)
        stale_agents = []
        for agent_id, agent_data in self.last_published_agents.items():
            if agent_id not in current_ids:
                time_since_seen = current_time - agent_data['last_seen']
                if time_since_seen > self.disappear_delay:
                    stale_agents.append(agent_id)
                else:
                    # 아직 지연 시간 내 - 속도를 점점 줄임 (멈춤 상태로)
                    decay = max(0, 1.0 - time_since_seen / self.disappear_delay)
                    agent_data['vx'] *= decay
                    agent_data['vy'] *= decay
                    agent_data['confidence'] = decay

        for agent_id in stale_agents:
            del self.last_published_agents[agent_id]
            # 관련 히스토리도 정리
            if agent_id in self.velocity_history:
                del self.velocity_history[agent_id]

        # AgentStates 메시지 생성 (현재 + 지연 중인 사람 모두 포함)
        agents_msg = AgentStates()
        agents_msg.header.stamp = rospy.Time.now()
        agents_msg.header.frame_id = 'odom'

        for agent_id, agent_data in self.last_published_agents.items():
            state = AgentState()
            state.id = agent_id
            state.type = 0  # person

            state.pose.position.x = agent_data['x']
            state.pose.position.y = agent_data['y']
            state.pose.position.z = 0.0
            state.pose.orientation.w = 1.0

            state.twist.linear.x = agent_data['vx']
            state.twist.linear.y = agent_data['vy']
            state.twist.linear.z = 0.0

            agents_msg.agent_states.append(state)

        # 발행
        self.agents_pub.publish(agents_msg)

        if self.last_published_agents:
            rospy.loginfo_throttle(2.0, f"Published {len(self.last_published_agents)} humans (detected: {len(agents_data)}, delayed: {len(self.last_published_agents) - len(agents_data)})")

    def run(self):
        """노드 실행"""
        rospy.spin()


def main():
    try:
        publisher = CCTVHumanPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
