#!/bin/bash
# ==============================================================================
# TEB Local Planner Experiment Runner
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
SCENARIO="${1:-scenario_crossing_sparse.xml}"
EPISODES="${2:-3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results/exp_teb_${TIMESTAMP}"
CONTAINER_NAME="gdae_pedsim_teb"

mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/episodes"

echo "=============================================================="
echo " TEB Local Planner Experiment"
echo "=============================================================="
echo " Scenario: $SCENARIO"
echo " Episodes: $EPISODES"
echo " Results:  $RESULTS_DIR"
echo "=============================================================="

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true
sleep 2

xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

# Docker 실행
docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e QT_QPA_PLATFORM=offscreen \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${ENV_DIR}:/environment:rw \
    -v ${ENV_DIR}/navigation:/navigation:rw \
    -v /home/pyongjoo/Desktop/newstart/sicnav-test:/sicnav-test:ro \
    -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
    --network host \
    gdae_with_navigation:yolo \
    bash -c '
        # Install TEB
        echo "Installing TEB local planner..."
        apt-get update -qq && apt-get install -y -qq ros-noetic-teb-local-planner 2>/dev/null

        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
        export PYTHONPATH=/sicnav-test:/environment:${PYTHONPATH}

        # Headless mode
        export QT_QPA_PLATFORM=offscreen
        Xvfb :99 -screen 0 1024x768x24 &
        export DISPLAY=:99
        sleep 2

        # Copy scenario
        cp /environment/with_robot/scenarios/'${SCENARIO}' /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        # Start simulation
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/'${SCENARIO}' \
            world_file:=/environment/with_robot/worlds/warehouse.world \
            gui:=false &

        sleep 15

        # cmd_vel inverter
        python3 /navigation/scripts/cmd_vel_inverter.py &
        sleep 2

        # move_base with TEB
        roslaunch /navigation/launch/move_base_teb.launch &
        sleep 5

        echo "TEB simulation ready!"
        tail -f /dev/null
    '

echo "Waiting for simulation to initialize (40s)..."
sleep 40

if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "ERROR: Container not running!"
    exit 1
fi

echo "Starting experiment..."

# 실험 실행
docker exec $CONTAINER_NAME bash -c "
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
    export PYTHONPATH=/environment/experiments:/environment:/sicnav-test:\${PYTHONPATH}

    python3 << 'PYTHON_SCRIPT'
import rospy
import json
import time as time_module
import os
from datetime import datetime
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
import numpy as np
import math

RESULTS_DIR = '/environment/experiments/results/exp_teb_${TIMESTAMP}'
EPISODES = ${EPISODES}

TOP = [(-10,10), (-5,10), (0,10), (5,10), (10,10)]
BOTTOM = [(-10,-10), (-5,-10), (0,-10), (5,-10), (10,-10)]

class TebExperiment:
    def __init__(self):
        rospy.init_node('teb_experiment', anonymous=True)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        self.robot_pos = None
        self.robot_vel = None
        self.robot_angular_vel = 0.0
        self.humans = []

        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/p3dx/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, self.agents_cb)

        print('[TebExperiment] Waiting for PedSim...')
        try:
            msg = rospy.wait_for_message('/pedsim_simulator/simulated_agents', AgentStates, timeout=15.0)
            self.humans = [(a.pose.position.x, a.pose.position.y) for a in msg.agent_states]
            print(f'[TebExperiment] Got {len(self.humans)} humans')
        except:
            print('[TebExperiment] No humans detected')

        time_module.sleep(2)
        print('[TebExperiment] Ready!')

    def odom_cb(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.robot_angular_vel = msg.twist.twist.angular.z

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
            time_module.sleep(1)
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
        start_time = time_module.time()
        rate = rospy.Rate(10)
        trajectory = []
        velocities = []
        min_dists = []
        collision_cnt = 0
        COLLISION_THRESH = 0.6
        MOVEMENT_THRESH = 0.05
        started = False

        while not rospy.is_shutdown() and (time_module.time() - start_time) < timeout:
            if self.robot_pos is None:
                rate.sleep()
                continue

            speed = 0
            if self.robot_vel:
                speed = np.sqrt(self.robot_vel[0]**2 + self.robot_vel[1]**2)

            if not started and speed > MOVEMENT_THRESH:
                started = True
                start_time = time_module.time()

            if started:
                trajectory.append(self.robot_pos)
                velocities.append(speed)

                for h in self.humans:
                    dist = np.sqrt((self.robot_pos[0]-h[0])**2 + (self.robot_pos[1]-h[1])**2)
                    min_dists.append(dist)
                    if dist < COLLISION_THRESH:
                        collision_cnt += 1

            dist_to_goal = np.sqrt((self.robot_pos[0]-goal[0])**2 + (self.robot_pos[1]-goal[1])**2)
            if dist_to_goal < 0.5:
                path_len = sum(np.sqrt((trajectory[i][0]-trajectory[i-1][0])**2 +
                              (trajectory[i][1]-trajectory[i-1][1])**2) for i in range(1,len(trajectory)))
                return {
                    'success': True,
                    'nav_time': time_module.time() - start_time,
                    'trajectory': trajectory,
                    'avg_velocity': np.mean(velocities) if velocities else 0,
                    'path_length': path_len,
                    'min_human_dist': min(min_dists) if min_dists else float('inf'),
                    'collision_count': collision_cnt
                }
            rate.sleep()

        return {
            'success': False,
            'nav_time': timeout,
            'trajectory': trajectory,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'path_length': 0,
            'min_human_dist': min(min_dists) if min_dists else float('inf'),
            'collision_count': collision_cnt
        }

    def run(self):
        results = []
        pairs = [(BOTTOM[i], TOP[i]) for i in range(min(EPISODES, 5))]

        for i, (start, goal) in enumerate(pairs):
            ep = i + 1
            print(f'[Episode {ep}] {start} -> {goal}')
            yaw = math.atan2(goal[1]-start[1], goal[0]-start[0])
            self.teleport_robot(start[0], start[1], yaw)
            time_module.sleep(2)
            self.send_goal(goal[0], goal[1])
            result = self.wait_for_goal(goal)
            result['episode'] = ep
            result['start'] = start
            result['goal'] = goal
            results.append(result)
            status = 'SUCCESS' if result['success'] else 'TIMEOUT'
            t = result['nav_time']
            c = result['collision_count']
            print(f'  {status}: time={t:.1f}s, collisions={c}')

            with open(f'{RESULTS_DIR}/episodes/ep_{ep:02d}.json', 'w') as f:
                r = result.copy()
                r['trajectory'] = [list(p) for p in result['trajectory']]
                json.dump(r, f, indent=2)
            time_module.sleep(3)

        successful = [r for r in results if r['success']]
        summary = {
            'planner': 'TEB',
            'total': len(results),
            'success_count': len(successful),
            'success_rate': len(successful)/len(results)*100 if results else 0,
            'avg_time': np.mean([r['nav_time'] for r in successful]) if successful else 0,
            'avg_velocity': np.mean([r['avg_velocity'] for r in results]),
            'total_collisions': sum(r['collision_count'] for r in results),
            'min_human_dist': min(r['min_human_dist'] for r in results)
        }
        with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        sr = summary['success_rate']
        at = summary['avg_time']
        tc = summary['total_collisions']
        md = summary['min_human_dist']
        print('\\n========== TEB RESULTS ==========')
        print(f'Success Rate: {sr:.1f}%')
        print(f'Avg Time: {at:.1f}s')
        print(f'Total Collisions: {tc}')
        print(f'Min Human Dist: {md:.2f}m')

if __name__ == '__main__':
    try:
        exp = TebExperiment()
        exp.run()
    except rospy.ROSInterruptException:
        pass
PYTHON_SCRIPT
"

echo ""
echo "Experiment completed!"
echo "Results: $RESULTS_DIR"

# 결과 출력
cat "${RESULTS_DIR}/summary.json" 2>/dev/null || echo "No summary"

# 컨테이너 정리
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo "Done!"
