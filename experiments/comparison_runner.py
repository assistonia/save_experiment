#!/usr/bin/env python3
"""
Comparison Runner - DWA vs CIGP 공정 비교
같은 시뮬레이션 시간에서 시작하여 공정 비교
"""

import rospy
import json
import time
import math
import argparse
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import subprocess


# 출발/목적지
TOP = [(-10, 10), (-5, 10), (0, 10), (5, 10), (10, 10)]
BOTTOM = [(-10, -10), (-5, -10), (0, -10), (5, -10), (10, -10)]


class ExperimentRunner:
    def __init__(self, goal_topic):
        self.goal_pub = rospy.Publisher(goal_topic, PoseStamped, queue_size=1)
        self.robot_pos = None
        self.robot_vel = None
        self.humans = []

        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/p3dx/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, self.agents_cb)
        time.sleep(2)

    def odom_cb(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)

    def agents_cb(self, msg):
        self.humans = [(a.pose.position.x, a.pose.position.y) for a in msg.agent_states]

    def teleport_robot(self, x, y, yaw=0.0):
        state = ModelState()
        state.model_name = 'p3dx'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.01
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        state.reference_frame = 'world'
        try:
            self.set_model_state(state)
            time.sleep(1)
            return True
        except:
            return False

    def send_goal(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def wait_for_goal(self, goal, timeout=120):
        start_time = time.time()
        rate = rospy.Rate(10)
        trajectory = []
        velocities = []
        min_human_dist_list = []

        while not rospy.is_shutdown() and (time.time() - start_time) < timeout:
            if self.robot_pos is None:
                rate.sleep()
                continue

            trajectory.append(self.robot_pos)
            if self.robot_vel:
                velocities.append(np.sqrt(self.robot_vel[0]**2 + self.robot_vel[1]**2))
            if self.humans:
                dists = [np.sqrt((self.robot_pos[0]-h[0])**2 + (self.robot_pos[1]-h[1])**2) for h in self.humans]
                min_human_dist_list.append(min(dists))

            dist_to_goal = np.sqrt((self.robot_pos[0]-goal[0])**2 + (self.robot_pos[1]-goal[1])**2)
            if dist_to_goal < 0.5:
                return {
                    'success': True,
                    'time': time.time() - start_time,
                    'trajectory': trajectory,
                    'avg_velocity': np.mean(velocities) if velocities else 0,
                    'min_human_dist': min(min_human_dist_list) if min_human_dist_list else float('inf'),
                    'avg_human_dist': np.mean(min_human_dist_list) if min_human_dist_list else float('inf')
                }
            rate.sleep()

        return {
            'success': False,
            'time': timeout,
            'trajectory': trajectory,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'min_human_dist': min(min_human_dist_list) if min_human_dist_list else float('inf'),
            'avg_human_dist': np.mean(min_human_dist_list) if min_human_dist_list else float('inf')
        }

    def run_episode(self, start, goal, episode_id):
        print(f'\n[Episode {episode_id}] {start} -> {goal}')

        yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
        self.teleport_robot(start[0], start[1], yaw)
        time.sleep(2)

        self.send_goal(goal[0], goal[1])
        result = self.wait_for_goal(goal)
        result['episode_id'] = episode_id
        result['start'] = start
        result['goal'] = goal

        status = 'SUCCESS' if result['success'] else 'TIMEOUT'
        print(f'  Result: {status}, Time: {result["time"]:.1f}s, Vel: {result["avg_velocity"]:.2f}m/s')

        return result


def reset_simulation():
    """시뮬레이션 리셋"""
    try:
        rospy.wait_for_service('/gazebo/reset_simulation', timeout=5)
        reset = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        reset()
        time.sleep(3)
        print('[Simulation] Reset complete')
        return True
    except Exception as e:
        print(f'[Simulation] Reset failed: {e}')
        return False


def save_results(results, module_name, results_dir):
    """결과 저장"""
    # 에피소드별 저장
    for r in results:
        ep_file = f'{results_dir}/{module_name}/episodes/episode_{r["episode_id"]:03d}.json'
        save_r = r.copy()
        save_r['trajectory'] = [list(p) for p in r['trajectory']]
        with open(ep_file, 'w') as f:
            json.dump(save_r, f, indent=2)

    # Summary
    summary = {
        'module': module_name,
        'total_episodes': len(results),
        'success_count': sum(1 for r in results if r['success']),
        'success_rate': sum(1 for r in results if r['success']) / len(results) * 100 if results else 0,
        'avg_time': np.mean([r['time'] for r in results if r['success']]) if any(r['success'] for r in results) else 0,
        'avg_velocity': np.mean([r['avg_velocity'] for r in results]),
        'avg_human_dist': np.mean([r['avg_human_dist'] for r in results])
    }
    with open(f'{results_dir}/{module_name}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()

    rospy.init_node('comparison_runner', anonymous=True)

    print('\n' + '='*60)
    print('PHASE 1: Testing DWA (move_base only)')
    print('='*60)

    # 시뮬레이션 리셋
    reset_simulation()

    # DWA 테스트
    dwa_runner = ExperimentRunner('/move_base_simple/goal')
    dwa_results = []

    for i, start in enumerate(BOTTOM[:args.episodes]):
        goal = TOP[0]
        result = dwa_runner.run_episode(start, goal, i + 1)
        dwa_results.append(result)
        time.sleep(3)

    dwa_summary = save_results(dwa_results, 'dwa', args.results_dir)
    print(f'\n[DWA] Success Rate: {dwa_summary["success_rate"]:.1f}%')

    print('\n' + '='*60)
    print('PHASE 2: Testing CIGP')
    print('='*60)

    # 시뮬레이션 리셋 (동일 시작 조건)
    reset_simulation()

    # CIGP 플래너 시작
    cigp_process = subprocess.Popen(
        ['python3', '/environment/with_robot/cigp_global_planner.py'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(5)

    # CIGP 테스트
    cigp_runner = ExperimentRunner('/cigp/goal')
    cigp_results = []

    for i, start in enumerate(BOTTOM[:args.episodes]):
        goal = TOP[0]
        result = cigp_runner.run_episode(start, goal, i + 1)
        cigp_results.append(result)
        time.sleep(3)

    cigp_summary = save_results(cigp_results, 'cigp', args.results_dir)
    print(f'\n[CIGP] Success Rate: {cigp_summary["success_rate"]:.1f}%')

    # CIGP 종료
    cigp_process.terminate()

    # 비교 결과 출력
    print('\n' + '='*60)
    print('COMPARISON RESULTS')
    print('='*60)
    print(f'{"Metric":<20} {"DWA":<15} {"CIGP":<15}')
    print('-'*60)
    print(f'{"Success Rate":<20} {dwa_summary["success_rate"]:.1f}%{"":<10} {cigp_summary["success_rate"]:.1f}%')
    print(f'{"Avg Time (s)":<20} {dwa_summary["avg_time"]:.1f}{"":<12} {cigp_summary["avg_time"]:.1f}')
    print(f'{"Avg Velocity (m/s)":<20} {dwa_summary["avg_velocity"]:.2f}{"":<12} {cigp_summary["avg_velocity"]:.2f}')
    print('='*60)

    # 비교 결과 저장
    comparison = {
        'dwa': dwa_summary,
        'cigp': cigp_summary
    }
    with open(f'{args.results_dir}/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)


if __name__ == '__main__':
    main()
