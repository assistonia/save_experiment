# PredictedTrajectoryArray.msg
# 모든 에이전트의 예측 궤적 배열

std_msgs/Header header

# 예측된 궤적 목록
predictive_planning/PredictedTrajectory[] trajectories

# 예측 메타데이터
float32 prediction_timestamp    # 예측이 수행된 시뮬레이션 시간
int32 total_agents              # 전체 에이전트 수
int32 predicted_agents          # 예측된 에이전트 수 (히스토리 충분)
