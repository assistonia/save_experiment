# Local Planners Module

기존 환경 수정 없이 독립적으로 동작하는 로컬 플래너 모듈.

## 구조

```
local_planners/
├── __init__.py
├── local_planner_node.py    # 통합 노드 (DWA, ORCA, SFM)
├── drl_vo/
│   ├── __init__.py
│   ├── custom_cnn.py        # DRL-VO 네트워크
│   ├── drl_vo_planner.py    # DRL-VO 플래너
│   └── model/
│       └── drl_vo.zip       # 학습된 모델 (PPO)
├── teb/
│   ├── __init__.py
│   └── teb_planner.py       # TEB 플래너
└── launch/
    └── local_planner.launch # 통합 런처
```

## 사용 가능한 플래너

| 플래너 | 설명 | 특징 |
|-------|------|------|
| `drl_vo` | DRL-VO (PPO + ResNet CNN) | 학습된 정책, 보행자 맵 + LiDAR 사용 |
| `teb` | Timed Elastic Band | 시간 최적 궤적, ROS 기본 지원 |
| `dwa` | Dynamic Window Approach | 속도-조향 공간 탐색 |
| `orca` | Optimal Reciprocal Collision Avoidance | 상호 충돌 회피 (rvo2 필요) |
| `sfm` | Social Force Model | Force 기반 자연스러운 회피 |

## 사용법

### ROS Launch

```bash
# DRL-VO 사용
roslaunch local_planners local_planner.launch planner:=drl_vo

# TEB 사용
roslaunch local_planners local_planner.launch planner:=teb

# DWA 사용
roslaunch local_planners local_planner.launch planner:=dwa

# ORCA 사용
roslaunch local_planners local_planner.launch planner:=orca

# SFM 사용
roslaunch local_planners local_planner.launch planner:=sfm
```

### Python API

```python
# DRL-VO
from local_planners.drl_vo.drl_vo_planner import DRLVOPlanner

planner = DRLVOPlanner()
action = planner.compute_action_simple(
    robot_pos=(x, y),
    robot_theta=theta,
    goal_world=(gx, gy),
    scan=lidar_data,
    humans=[{'pos': [hx, hy], 'vel': [vx, vy]}]
)
print(f"v={action.linear_x}, w={action.angular_z}")

# TEB
from local_planners.teb.teb_planner import TEBPlannerROS

node = TEBPlannerROS()
node.run()
```

## 토픽

### 입력 (구독)
- `/cigp/global_path` (nav_msgs/Path): CIGP 글로벌 경로
- `/cigp/next_waypoint` (geometry_msgs/PointStamped): 다음 웨이포인트
- `/odom` (nav_msgs/Odometry): 로봇 오도메트리
- `/scan` (sensor_msgs/LaserScan): LiDAR 스캔
- `/pedsim_simulator/simulated_agents` (pedsim_msgs/AgentStates): 보행자 상태

### 출력 (발행)
- `/cmd_vel` (geometry_msgs/Twist): 속도 명령

## 의존성

### DRL-VO
- stable_baselines3
- torch (GPU 권장)

### TEB
- ros-noetic-teb-local-planner
```bash
sudo apt install ros-noetic-teb-local-planner
```

### ORCA
- python-rvo2 (옵션)
```bash
pip install rvo2
```

## 기존 환경과의 연동

이 모듈은 기존 `cigp_bridge_node.py`와 **독립적**으로 동작합니다.

```
CIGP Global Planner (cigp_bridge_node.py)
         │
         ▼
    /cigp/global_path
         │
         ▼
Local Planner (이 모듈) ──► /cmd_vel ──► 로봇
```

기존 환경의 어떤 코드도 수정하지 않고, 단순히 토픽을 구독/발행합니다.
