# Predictive Planning Module

SingularTrajectory 예측 결과를 활용한 **시간 인식 확률적 경로 계획** 모듈.

## 핵심 아이디어

기존 CIGP와의 차이점:
- **기존**: 사람의 **현재 위치**만 고려
- **본 모듈**: 사람의 **미래 예측 위치 20개 샘플**을 시간 기반으로 고려

### 합의된 수식

1. **개별 샘플 반경 축소**:
   ```
   sigma_k = max(0.8 * v_k, 0.3)
   ```
   - `0.8`: 샘플이 퍼졌을 때 맵이 꽉 막히지 않도록 축소
   - `0.3`: 정지 시에도 최소 반경 유지

2. **확률적 희석**:
   ```
   agent_cost = sum(cost_k) / 20
   ```
   - 20개 샘플 중 실제로 겹치는 비율로 자연스럽게 희석
   - 불확실한 예측 → 낮은 비용 → 로봇이 과감해짐

3. **시간 매핑**:
   ```
   t_arrival = g_cost / robot_velocity
   time_index = int(t_arrival / 0.4)
   ```
   - A*의 g_cost를 도착 시간으로 변환
   - 해당 시간의 예측 샘플만 사용

## 폴더 구조

```
predictive_planning/
├── src/
│   ├── predicted_trajectory.py      # 예측 데이터 구조
│   ├── prediction_receiver.py       # 예측 결과 수신기
│   ├── predictive_cost_calculator.py # 확률적 비용 계산 (핵심)
│   ├── predictive_global_planner.py  # 시간 인식 A*
│   ├── predictive_planning_bridge.py # ROS 브릿지
│   └── config.py                     # 설정 관리
├── msg/
│   ├── PredictedTrajectory.msg
│   └── PredictedTrajectoryArray.msg
├── launch/
│   ├── predictive_planning.launch
│   └── full_system.launch
├── config/
│   └── warehouse.yaml
├── logs/                              # 실행 로그 저장
├── run_predictive_planning.sh
├── test_predictive_planning.py
└── README.md
```

## 사용법

### 1. 테스트 실행 (ROS 없이)
```bash
cd /home/pyongjoo/Desktop/newstart/environment
python3 predictive_planning/test_predictive_planning.py
```

### 2. ROS 환경에서 실행
```bash
# 기존 시뮬레이션 실행 후
./predictive_planning/run_predictive_planning.sh

# 또는 launch 파일 사용
roslaunch predictive_planning predictive_planning.launch
```

### 3. Mock 모드 (예측기 없이 테스트)
```bash
./predictive_planning/run_predictive_planning.sh --mock
```

## 핵심 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `sigma_scale` | 0.8 | 개별 샘플 반경 축소 계수 |
| `sigma_min` | 0.3 | 최소 반경 (m) |
| `social_cost_weight` | 2.0 | A* 비용 함수에서 소셜 비용 가중치 |
| `robot_velocity` | 0.8 | 로봇 평균 속도 (m/s) |
| `resolution` | 0.2 | A* 그리드 해상도 (m) |

## ROS 토픽

### Subscribe
- `/pedsim_simulator/simulated_agents` (AgentStates): 보행자 상태
- `/odom` (Odometry): 로봇 위치/속도
- `/move_base_simple/goal` (PoseStamped): 목표 지점

### Publish
- `/predictive_planning/path` (Path): 계획된 경로
- `/predictive_planning/next_waypoint` (PointStamped): 다음 웨이포인트
- `/predictive_planning/markers` (MarkerArray): 시각화 마커

## 데이터 흐름

```
PedSim (보행자 GT)
       ↓
trajectory_prediction (SingularTrajectory)
       ↓ 20개 샘플 예측
PredictionReceiver
       ↓
PredictiveCostCalculator (시간 기반 비용)
       ↓
PredictiveGlobalPlanner (A*)
       ↓
Human-aware Path
```

## 로그 출력

실행 시 `logs/run_XXX/` 폴더에:
- `meta.json`: 메타데이터
- `log.json`: 계획 결과 로그
- `frames/`: 시각화 이미지
- `planning_latest.png`: 최신 시각화

## 주의사항

- 기존 `trajectory_prediction`, `cigp_integration` 모듈을 **수정하지 않음**
- 예측 결과를 **직접 모듈 호출**로 가져옴 (ROS 토픽 의존성 최소화)
- GPU 필요 (SingularTrajectory 모델 로드 시)
