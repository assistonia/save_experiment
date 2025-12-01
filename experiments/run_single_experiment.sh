#!/bin/bash
# ==============================================================================
# Single Experiment Runner
#
# 기존 모듈들(CIGP, Predictive 등)을 호출하여 실험 실행
# 맵/시나리오 선택 가능, 결과는 experiments/results/에 저장
#
# Usage:
#   ./run_single_experiment.sh [OPTIONS]
#
# Options:
#   --module        : cigp, predictive, robot (default: cigp)
#   --local-planner : dwa, teb, mpc (default: dwa)
#   --scenario      : 시나리오 파일 (default: scenario_congestion_all.xml)
#   --episodes      : 에피소드 수 (default: 5)
#   --headless      : GUI 없이 실행
#
# Examples:
#   ./run_single_experiment.sh --module cigp --scenario scenario_block_heavy.xml
#   ./run_single_experiment.sh --module predictive --local-planner teb --episodes 10
#   ./run_single_experiment.sh --module robot --local-planner mpc --headless
# ==============================================================================

set -e

# 기본값
MODULE="cigp"
LOCAL_PLANNER="dwa"
SCENARIO="scenario_congestion_all.xml"
EPISODES=5
HEADLESS=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --module)
            MODULE="$2"
            shift 2
            ;;
        --local-planner)
            LOCAL_PLANNER="$2"
            shift 2
            ;;
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --headless)
            HEADLESS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(dirname "$SCRIPT_DIR")"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results/exp_${MODULE}_${LOCAL_PLANNER}_${TIMESTAMP}"

# 결과 폴더 생성
mkdir -p "${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}/episodes"
mkdir -p "${RESULTS_DIR}/trajectories"

# 로그 파일
LOG_FILE="${RESULTS_DIR}/logs/experiment.log"

echo "==============================================================" | tee "$LOG_FILE"
echo " Single Experiment Runner" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo " Module:        $MODULE" | tee -a "$LOG_FILE"
echo " Local Planner: $LOCAL_PLANNER" | tee -a "$LOG_FILE"
echo " Scenario:      $SCENARIO" | tee -a "$LOG_FILE"
echo " Episodes:      $EPISODES" | tee -a "$LOG_FILE"
echo " Headless:      $HEADLESS" | tee -a "$LOG_FILE"
echo " Results:       $RESULTS_DIR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "==============================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 실험 설정 저장
cat > "${RESULTS_DIR}/config.json" << EOF
{
    "module": "$MODULE",
    "local_planner": "$LOCAL_PLANNER",
    "scenario": "$SCENARIO",
    "episodes": $EPISODES,
    "headless": $HEADLESS,
    "timestamp": "$TIMESTAMP",
    "results_dir": "$RESULTS_DIR"
}
EOF

# 모듈별 실행 스크립트 선택
case $MODULE in
    "cigp")
        RUN_SCRIPT="${ENV_DIR}/with_robot/run_with_cigp.sh"
        CONTAINER_NAME="gdae_pedsim_cigp"
        GOAL_TOPIC="/cigp/goal"
        ;;
    "predictive")
        RUN_SCRIPT="${ENV_DIR}/predictive_planning/run_with_predictive_planning.sh"
        CONTAINER_NAME="gdae_pedsim_predictive_planning"
        GOAL_TOPIC="/predictive_planning/goal"
        ;;
    "robot")
        RUN_SCRIPT="${ENV_DIR}/with_robot/run_with_robot.sh"
        CONTAINER_NAME="gdae_pedsim_robot"
        GOAL_TOPIC="/move_base_simple/goal"
        ;;
    "cigp_cctv")
        RUN_SCRIPT="${ENV_DIR}/getimage/run_with_cigp_cctv.sh"
        CONTAINER_NAME="gdae_pedsim_cigp_cctv"
        GOAL_TOPIC="/cigp_cctv/goal"
        ;;
    *)
        echo "Unknown module: $MODULE"
        exit 1
        ;;
esac

echo "Using script: $RUN_SCRIPT" | tee -a "$LOG_FILE"
echo "Container: $CONTAINER_NAME" | tee -a "$LOG_FILE"
echo "Goal topic: $GOAL_TOPIC" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# GUI 모드 설정 (Docker 실행 전에 설정해야 함)
if [ "$HEADLESS" = true ]; then
    GUI_MODE="false"
else
    GUI_MODE="true"
fi

# 기존 컨테이너 정리
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# 시뮬레이션 시작 (Docker detached 모드로 직접 실행)
echo "Starting simulation..." | tee -a "$LOG_FILE"
echo "GUI Mode: $GUI_MODE" | tee -a "$LOG_FILE"

# 기존 스크립트 대신 직접 Docker 실행 (detached)
xhost +local:root 2>/dev/null || true
xhost +local:docker 2>/dev/null || true

docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --privileged \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e QT_QPA_PLATFORM=offscreen \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e GUI_MODE=$GUI_MODE \
    -e MODULE=$MODULE \
    -e LOCAL_PLANNER=$LOCAL_PLANNER \
    -e SCENARIO=$SCENARIO \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${ENV_DIR}:/environment:rw \
    -v ${ENV_DIR}/navigation:/navigation:rw \
    -v /home/pyongjoo/Desktop/newstart/sicnav-test:/sicnav-test:ro \
    -v /home/pyongjoo/.Xauthority:/root/.Xauthority:rw \
    --network host \
    gdae_with_navigation:yolo \
    bash -c '
        source /opt/ros/noetic/setup.bash
        source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
        export PYTHONPATH=/sicnav-test:/environment:${PYTHONPATH}

        # Headless Qt support - xvfb for pedsim_simulator
        if [ "$GUI_MODE" = "false" ]; then
            export QT_QPA_PLATFORM=offscreen
            Xvfb :99 -screen 0 1024x768x24 &
            export DISPLAY=:99
            sleep 2
        fi

        # 시나리오 복사
        cp /environment/with_robot/scenarios/${SCENARIO} /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/scenarios/
        cp /environment/with_robot/worlds/warehouse.world /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_gazebo_plugin/worlds/
        cp /environment/with_robot/launch/warehouse_with_robot.launch /root/DRL-robot-navigation/catkin_ws/src/pedsim_ros_with_gazebo/pedsim_simulator/launch/

        # PedSim + Gazebo 실행
        roslaunch pedsim_simulator warehouse_with_robot.launch \
            scene_file:=/environment/with_robot/scenarios/${SCENARIO} \
            world_file:=/environment/with_robot/worlds/warehouse.world \
            gui:=${GUI_MODE} &

        sleep 15

        # cmd_vel inverter (좌우 반전 수정)
        python3 /navigation/scripts/cmd_vel_inverter.py &
        sleep 2

        # Local Planner 설치 및 실행
        case $LOCAL_PLANNER in
            teb)
                echo "Installing TEB local planner..."
                apt-get update -qq && apt-get install -y -qq ros-noetic-teb-local-planner 2>/dev/null
                echo "Starting move_base with TEB..."
                roslaunch /navigation/launch/move_base_teb.launch &
                ;;
            mpc)
                echo "Installing MPC local planner..."
                apt-get update -qq && apt-get install -y -qq ros-noetic-mpc-local-planner 2>/dev/null
                echo "Starting move_base with MPC..."
                roslaunch /navigation/launch/move_base_mpc.launch &
                ;;
            *)
                echo "Starting move_base with DWA..."
                roslaunch /navigation/launch/move_base.launch &
                ;;
        esac
        sleep 5

        # 모듈별 글로벌 플래너 실행
        case $MODULE in
            cigp)
                echo "Starting CIGP Global Planner..."
                python3 /environment/with_robot/cigp_global_planner.py &
                ;;
            predictive)
                echo "Starting Predictive Planning..."
                cd /environment/predictive_planning
                python3 src/predictive_planning_bridge.py _use_direct_control:=false &
                ;;
            robot)
                echo "Using move_base only (no global planner)..."
                ;;
        esac

        sleep 3

        # 무한 대기 (컨테이너 유지)
        tail -f /dev/null
    '

# 시뮬레이션 초기화 대기
echo "Waiting for simulation to initialize (30s)..." | tee -a "$LOG_FILE"
sleep 30

# 컨테이너 상태 확인
if ! docker ps | grep -q $CONTAINER_NAME; then
    echo "ERROR: Container $CONTAINER_NAME not running!" | tee -a "$LOG_FILE"
    exit 1
fi

echo "Simulation ready!" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 출발/목적지 위치 정의
TOP_POSITIONS=(
    "-10.0 10.0"
    "-5.0 10.0"
    "0.0 10.0"
    "5.0 10.0"
    "10.0 10.0"
)

BOTTOM_POSITIONS=(
    "-10.0 -10.0"
    "-5.0 -10.0"
    "0.0 -10.0"
    "5.0 -10.0"
    "10.0 -10.0"
)

# 실험 러너 실행 (Docker 내부)
echo "Starting experiment runner inside container..." | tee -a "$LOG_FILE"

docker exec $CONTAINER_NAME bash -c "
    source /opt/ros/noetic/setup.bash
    source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash
    export PYTHONPATH=/environment/experiments:/environment:/sicnav-test:\${PYTHONPATH}

    python3 << 'PYTHON_SCRIPT'
import rospy
import json
import time
import os
from datetime import datetime
from geometry_msgs.msg import PoseStamped, Pose, Twist as TwistMsg
from nav_msgs.msg import Odometry
from pedsim_msgs.msg import AgentStates
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from std_srvs.srv import Empty
import numpy as np

# 설정
RESULTS_DIR = '/environment/experiments/results/exp_${MODULE}_${LOCAL_PLANNER}_${TIMESTAMP}'
GOAL_TOPIC = '${GOAL_TOPIC}'
EPISODES = ${EPISODES}

# 출발/목적지
TOP = [(-10,10), (-5,10), (0,10), (5,10), (10,10)]
BOTTOM = [(-10,-10), (-5,-10), (0,-10), (5,-10), (10,-10)]

class ExperimentRunner:
    def __init__(self):
        rospy.init_node('experiment_runner', anonymous=True)

        self.goal_pub = rospy.Publisher(GOAL_TOPIC, PoseStamped, queue_size=1)
        self.robot_pos = None
        self.robot_vel = None
        self.robot_angular_vel = 0.0  # 각속도 초기화
        self.humans = []
        self.episode_data = []

        # Gazebo 서비스
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10)
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.Subscriber('/p3dx/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/pedsim_simulator/simulated_agents', AgentStates, self.agents_cb)

        # Wait for PedSim agents data (Fix for Min Human Dist: inf)
        print('[ExperimentRunner] Waiting for PedSim agents data...')
        try:
            msg = rospy.wait_for_message('/pedsim_simulator/simulated_agents', AgentStates, timeout=15.0)
            self.humans = [(a.pose.position.x, a.pose.position.y) for a in msg.agent_states]
            print(f'[ExperimentRunner] PedSim agents detected: {len(self.humans)} humans')
        except rospy.ROSException:
            print('[ExperimentRunner] WARNING: PedSim agents NOT detected after 15s!')

        time.sleep(2)
        print('[ExperimentRunner] Initialized')

    def teleport_robot(self, x, y, yaw=0.0):
        \"\"\"로봇을 지정 위치로 텔레포트\"\"\"
        import math
        state = ModelState()
        state.model_name = 'p3dx'
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.01
        # yaw to quaternion
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        state.reference_frame = 'world'

        try:
            self.set_model_state(state)
            print(f'[ExperimentRunner] Robot teleported to ({x}, {y})')
            time.sleep(1)  # 안정화 대기
            return True
        except Exception as e:
            print(f'[ExperimentRunner] Teleport failed: {e}')
            return False

    def odom_cb(self, msg):
        self.robot_pos = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        self.robot_vel = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.robot_angular_vel = msg.twist.twist.angular.z  # 각속도 추가

    def agents_cb(self, msg):
        self.humans = [(a.pose.position.x, a.pose.position.y) for a in msg.agent_states]
        self._agents_callback_count = getattr(self, '_agents_callback_count', 0) + 1
        if len(self.humans) > 0 and not hasattr(self, '_agents_logged'):
            print(f'[ExperimentRunner] Receiving {len(self.humans)} humans from pedsim (callback #{self._agents_callback_count})')
            self._agents_logged = True
        # 주기적으로 상태 출력 (100번마다)
        if self._agents_callback_count % 100 == 0:
            print(f'[ExperimentRunner] PedSim callback #{self._agents_callback_count}: {len(self.humans)} humans')

    def send_goal(self, x, y):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'odom'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)
        print(f'[ExperimentRunner] Goal sent: ({x}, {y})')

    def wait_for_goal(self, goal, timeout=120):
        goal_sent_time = time.time()
        start_time = None  # 로봇이 움직이기 시작한 시점
        rate = rospy.Rate(10)

        trajectory = []
        min_human_dist_list = []
        velocities = []
        angular_velocities = []  # 각속도 추가
        collisions = []  # 충돌 위치 기록
        collision_count = 0
        intrusion_time = 0  # ITR용 침입 시간
        sample_count = 0
        human_positions_all = []  # 사람 위치 전체 기록

        ROBOT_RADIUS = 0.3
        HUMAN_RADIUS = 0.3
        COLLISION_THRESHOLD = ROBOT_RADIUS + HUMAN_RADIUS  # 0.6m 이내면 충돌
        INTRUSION_THRESHOLD = 1.2  # 1.2m 이내 = 개인 공간 침입 (사회적 거리)
        SAMPLE_DT = 0.1  # 10Hz
        MOVEMENT_THRESHOLD = 0.05  # 0.05m/s 이상이면 움직임 시작으로 판정

        while not rospy.is_shutdown() and (time.time() - goal_sent_time) < timeout:
            if self.robot_pos is None:
                rate.sleep()
                continue

            sample_count += 1

            # 현재 속도 계산
            current_speed = 0.0
            if self.robot_vel:
                current_speed = np.sqrt(self.robot_vel[0]**2 + self.robot_vel[1]**2)

            # 움직임 시작 감지 (계산 완료 후 실제 이동 시작)
            if start_time is None and current_speed > MOVEMENT_THRESHOLD:
                start_time = time.time()
                planning_time = start_time - goal_sent_time
                print(f'  [Motion started] Planning took {planning_time:.2f}s')

            # 궤적 기록 (움직이기 시작한 후부터)
            if start_time is not None:
                trajectory.append(self.robot_pos)
                velocities.append(current_speed)
                angular_velocities.append(abs(self.robot_angular_vel))

                # 사람 위치 기록 (빈 리스트라도 기록)
                human_positions_all.append([list(h) for h in self.humans])

            # 사람과의 거리 및 충돌/침입 체크
            if self.humans:
                frame_min_dist = float('inf')
                for h in self.humans:
                    dist = np.sqrt((self.robot_pos[0]-h[0])**2 + (self.robot_pos[1]-h[1])**2)
                    min_human_dist_list.append(dist)
                    frame_min_dist = min(frame_min_dist, dist)

                    # 충돌 체크 (움직이기 시작한 후에만 기록)
                    if dist < COLLISION_THRESHOLD and start_time is not None:
                        collision_count += 1
                        collisions.append({
                            'robot_pos': self.robot_pos,
                            'human_pos': h,
                            'distance': dist,
                            'time': time.time() - start_time
                        })
                        print(f'  [COLLISION] dist={dist:.2f}m at ({self.robot_pos[0]:.1f}, {self.robot_pos[1]:.1f})')

                # 침입 시간 계산 (ITR) - 사회적 거리 침입
                if frame_min_dist < INTRUSION_THRESHOLD:
                    intrusion_time += SAMPLE_DT

            # 목표 도달 체크
            dist_to_goal = np.sqrt((self.robot_pos[0]-goal[0])**2 + (self.robot_pos[1]-goal[1])**2)
            if dist_to_goal < 0.5:
                # 움직임 시작 시점부터 시간 측정 (공정한 비교)
                if start_time is not None:
                    nav_time = time.time() - start_time
                    planning_time = start_time - goal_sent_time
                else:
                    nav_time = time.time() - goal_sent_time
                    planning_time = 0

                # 경로 길이 계산
                path_length = 0
                for i in range(1, len(trajectory)):
                    dx = trajectory[i][0] - trajectory[i-1][0]
                    dy = trajectory[i][1] - trajectory[i-1][1]
                    path_length += np.sqrt(dx**2 + dy**2)

                return {
                    'success': True,
                    'time': nav_time,  # 움직인 시간만 측정
                    'planning_time': planning_time,  # 계획 시간 별도 기록
                    'total_time': time.time() - goal_sent_time,  # 전체 시간
                    'trajectory': trajectory,
                    'human_positions': human_positions_all,  # 사람 위치 기록
                    'avg_velocity': np.mean(velocities) if velocities else 0,
                    'avg_angular_velocity': np.mean(angular_velocities) if angular_velocities else 0,
                    'path_length': path_length,
                    'min_human_dist': min(min_human_dist_list) if min_human_dist_list else float('inf'),
                    'avg_human_dist': np.mean(min_human_dist_list) if min_human_dist_list else float('inf'),
                    'collision_count': collision_count,
                    'collisions': collisions,
                    'intrusion_time': intrusion_time,
                    'itr': intrusion_time / nav_time if nav_time > 0 else 0  # Intrusion Time Ratio
                }

            rate.sleep()

        # Timeout - 경로 길이 계산
        path_length = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i][0] - trajectory[i-1][0]
            dy = trajectory[i][1] - trajectory[i-1][1]
            path_length += np.sqrt(dx**2 + dy**2)

        if start_time is not None:
            nav_time = time.time() - start_time
            planning_time = start_time - goal_sent_time
        else:
            nav_time = timeout
            planning_time = 0

        return {
            'success': False,
            'time': nav_time,
            'planning_time': planning_time,
            'total_time': time.time() - goal_sent_time,
            'trajectory': trajectory,
            'human_positions': human_positions_all,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'avg_angular_velocity': np.mean(angular_velocities) if angular_velocities else 0,
            'path_length': path_length,
            'min_human_dist': min(min_human_dist_list) if min_human_dist_list else float('inf'),
            'avg_human_dist': np.mean(min_human_dist_list) if min_human_dist_list else float('inf'),
            'collision_count': collision_count,
            'collisions': collisions,
            'intrusion_time': intrusion_time,
            'itr': intrusion_time / nav_time if nav_time > 0 else 0
        }

    def run(self):
        results = []
        episode_id = 0
        import math

        # 출발지-목적지 쌍 생성
        # EPISODES로 총 쌍 수 제한 (5x5=25 중 선택)
        pairs = []
        for start in BOTTOM:
            for goal in TOP:
                pairs.append((start, goal))

        # EPISODES 수 만큼만 테스트 (균등 분포로 선택)
        if EPISODES < len(pairs):
            step = len(pairs) // EPISODES
            pairs = pairs[::step][:EPISODES]

        print(f'[ExperimentRunner] Testing {len(pairs)} start-goal pairs')

        for start, goal in pairs:
            episode_id += 1
            print(f'\\n[Episode {episode_id}] {start} -> {goal}')
            print(f'  [PedSim Status] Currently tracking {len(self.humans)} humans')

            # 로봇을 시작 위치로 텔레포트
            yaw = math.atan2(goal[1] - start[1], goal[0] - start[0])
            self.teleport_robot(start[0], start[1], yaw)
            time.sleep(2)  # 안정화

            # 에피소드 시작 전 사람 데이터 재확인
            if len(self.humans) == 0:
                print('  [WARNING] No humans detected! Waiting for PedSim data...')
                try:
                    msg = rospy.wait_for_message('/pedsim_simulator/simulated_agents', AgentStates, timeout=5.0)
                    self.humans = [(a.pose.position.x, a.pose.position.y) for a in msg.agent_states]
                    print(f'  [PedSim] Got {len(self.humans)} humans')
                except:
                    print('  [PedSim] Still no humans detected')

            self.send_goal(goal[0], goal[1])
            result = self.wait_for_goal(goal)
            result['episode_id'] = episode_id
            result['start'] = start
            result['goal'] = goal
            result['direction'] = 'bottom_to_top'
            results.append(result)

            status = 'SUCCESS' if result['success'] else 'TIMEOUT'
            collisions = result.get('collision_count', 0)
            print(f'  Result: {status}, Time: {result[\"time\"]:.1f}s, Vel: {result[\"avg_velocity\"]:.2f}m/s, Collisions: {collisions}')

            # 에피소드 데이터 저장
            ep_file = f'{RESULTS_DIR}/episodes/episode_{episode_id:03d}.json'
            with open(ep_file, 'w') as f:
                save_result = result.copy()
                save_result['trajectory'] = [list(p) for p in result['trajectory']]
                # collisions 내부의 튜플도 변환
                if 'collisions' in save_result:
                    for c in save_result['collisions']:
                        c['robot_pos'] = list(c['robot_pos'])
                        c['human_pos'] = list(c['human_pos'])
                json.dump(save_result, f, indent=2)

            time.sleep(3)  # 다음 에피소드 전 대기

        # 전체 결과 저장 - 새로운 메트릭 포함
        total_collisions = sum(r.get('collision_count', 0) for r in results)
        successful = [r for r in results if r['success']]

        summary = {
            'total_episodes': len(results),
            'success_count': len(successful),
            'success_rate': len(successful) / len(results) * 100 if results else 0,
            'avg_time': np.mean([r['time'] for r in successful]) if successful else 0,
            'avg_planning_time': np.mean([r.get('planning_time', 0) for r in results]),  # 계획 시간
            'avg_total_time': np.mean([r.get('total_time', r['time']) for r in successful]) if successful else 0,  # 전체 시간
            'avg_velocity': np.mean([r['avg_velocity'] for r in results]),
            'avg_angular_velocity': np.mean([r.get('avg_angular_velocity', 0) for r in results]),
            'avg_path_length': np.mean([r.get('path_length', 0) for r in successful]) if successful else 0,
            'avg_human_dist': np.mean([r['avg_human_dist'] for r in results]),
            'min_human_dist': min([r.get('min_human_dist', float('inf')) for r in results]),
            'total_collisions': total_collisions,
            'collision_rate': total_collisions / len(results) if results else 0,
            'avg_itr': np.mean([r.get('itr', 0) for r in results]),  # Intrusion Time Ratio
            'total_intrusion_time': sum(r.get('intrusion_time', 0) for r in results)
        }

        with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f'\\n========== SUMMARY ==========')
        print(f'Success Rate: {summary[\"success_rate\"]:.1f}%')
        print(f'Avg Navigation Time: {summary[\"avg_time\"]:.1f}s (planning excluded)')
        print(f'Avg Planning Time: {summary[\"avg_planning_time\"]:.2f}s')
        print(f'Avg Total Time: {summary[\"avg_total_time\"]:.1f}s')
        print(f'Avg Velocity: {summary[\"avg_velocity\"]:.2f}m/s')
        print(f'Avg Angular Vel: {summary[\"avg_angular_velocity\"]:.2f}rad/s')
        print(f'Avg Path Length: {summary[\"avg_path_length\"]:.1f}m')
        print(f'Min Human Dist: {summary[\"min_human_dist\"]:.2f}m')
        print(f'Avg ITR: {summary[\"avg_itr\"]:.3f}')
        print(f'Total Collisions: {summary[\"total_collisions\"]}')
        print(f'Results saved to: {RESULTS_DIR}')

if __name__ == '__main__':
    try:
        runner = ExperimentRunner()
        runner.run()
    except rospy.ROSInterruptException:
        pass
PYTHON_SCRIPT
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Experiment completed!" | tee -a "$LOG_FILE"
echo "Results: $RESULTS_DIR" | tee -a "$LOG_FILE"

# 시뮬레이션 종료
echo "Stopping simulation..." | tee -a "$LOG_FILE"
docker stop $CONTAINER_NAME 2>/dev/null || true

echo "Done!" | tee -a "$LOG_FILE"
