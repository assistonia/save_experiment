# CIGP Integration for ROS PedSim

CIGP(CCTV-Informed Global Planner)를 ROS PedSim 환경과 통합하는 모듈.

## 구조

```
cigp_integration/
├── cigp_bridge_node.py      # ROS-CIGP 브릿지 노드 (메인)
├── cigp_config.py           # 시나리오별 설정 관리
├── __init__.py
├── launch/
│   ├── cigp_planner.launch         # CIGP 단독 런치
│   └── warehouse_with_cigp.launch  # 시뮬레이션 + CIGP 통합 런치
├── run_with_cigp.sh         # 전체 실행 스크립트
└── run_cigp_only.sh         # CIGP만 단독 실행
```

## 사용법

### 방법 1: 전체 시뮬레이션 + CIGP 함께 실행
```bash
cd /home/pyongjoo/Desktop/newstart/environment/cigp_integration
./run_with_cigp.sh
```

### 방법 2: 기존 시뮬레이션에 CIGP 추가
```bash
# 터미널 1: 기존 시뮬레이션 실행
cd /home/pyongjoo/Desktop/newstart/environment/with_robot
./run_with_robot_scenario.sh

# 터미널 2: CIGP 플래너 추가 실행
cd /home/pyongjoo/Desktop/newstart/environment/cigp_integration
./run_cigp_only.sh
```

## ROS Topics

### Subscribe
- `/pedsim_simulator/simulated_agents` - 보행자 상태
- `/odom` - 로봇 오도메트리
- `/move_base_simple/goal` - 네비게이션 목표

### Publish
- `/cigp/global_path` (nav_msgs/Path) - Human-aware 글로벌 경로
- `/cigp/next_waypoint` (geometry_msgs/PointStamped) - 다음 웨이포인트
- `/cigp/social_cost_map` (nav_msgs/OccupancyGrid) - 소셜 코스트 시각화

## 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| x_min/x_max | -12.0/12.0 | 맵 X 범위 |
| y_min/y_max | -12.0/12.0 | 맵 Y 범위 |
| resolution | 0.1 | 그리드 해상도 (m) |
| gamma1 | 0.5 | Social Cost 가중치 |
| robot_radius | 0.25 | 로봇 반경 |
| n_cctvs | 4 | CCTV 수 |
| use_dwa | false | DWA 로컬 플래너 사용 여부 |

## 기존 코드 수정 없음

이 모듈은 **기존 환경 코드를 전혀 수정하지 않습니다**.
- ROS 토픽을 통해 독립적으로 통신
- 별도 컨테이너/프로세스로 실행 가능
- 언제든 활성화/비활성화 가능
