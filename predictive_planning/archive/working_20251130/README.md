# Predictive Planning - Working Archive (2024-11-30)

## 작동 확인된 설정

### 구조
```
PedSim GT → SingularTrajectory (20 samples) → Predictive A* → move_base/DWA
```

### 핵심 파일
- `run_with_predictive_planning.sh` - 실행 스크립트
- `src/predictive_planning_bridge.py` - 메인 노드
- `dwa_local_planner.yaml` - DWA 설정 (도리도리 방지)
- `move_base.launch` - cmd_vel → /cmd_vel_raw 리맵

### Docker 이미지
```
gdae_with_navigation:latest
```

### 실행 방법

**터미널 1**:
```bash
cd /home/pyongjoo/Desktop/newstart/environment/predictive_planning
./run_with_predictive_planning.sh scenario_block_light.xml
```

**터미널 2** (가제보 로드 후 ~30초):
```bash
docker exec gdae_pedsim_predictive_planning bash -c 'source /opt/ros/noetic/setup.bash && rostopic pub -1 /predictive_planning/goal geometry_msgs/PoseStamped "{header: {frame_id: \"odom\"}, pose: {position: {x: 9.0, y: -10.0, z: 0.0}, orientation: {w: 1.0}}}"'
```

### 주요 수정 사항 (CIGP와 동일하게)

1. **move_base.launch**: cmd_vel remap
   - 이전: `/p3dx/cmd_vel` (cmd_vel_inverter 우회)
   - 수정: `/cmd_vel_raw` (cmd_vel_inverter 통과)

2. **dwa_local_planner.yaml**: 도리도리 방지
   - `min_vel_theta: 0` (이전: 0.3)
   - `min_vel_x: 0` (이전: 0.1)

3. **predictive_planning_bridge.py**: 웨이포인트 선택
   - `min_lookahead = 1.0m` (move_base xy_goal_tolerance 0.3m 보다 큼)
   - 경로 재계획 시 웨이포인트 인덱스 리셋

### Topics

**Input**:
- `/pedsim_simulator/simulated_agents` - 보행자 GT
- `/p3dx/odom` - 로봇 오도메트리
- `/predictive_planning/goal` - 목표 지점

**Output**:
- `/move_base_simple/goal` - 웨이포인트 → move_base
- `/predictive_planning/global_path` - 경로 시각화
- `/predictive_planning/markers` - RViz 마커
