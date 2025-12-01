# PredictedTrajectory.msg
# 단일 에이전트의 예측 궤적 (20개 샘플 x 12 시점)

int32 agent_id

# 현재 위치 및 속도
float32 current_x
float32 current_y
float32 current_vx
float32 current_vy

# 20개 샘플 x 12개 시점의 좌표 (Flattened)
# samples_x[i*12 + t] = 샘플 i의 시점 t에서의 x 좌표
# 총 길이: 20 * 12 = 240
float32[] samples_x
float32[] samples_y

# 메타데이터
float32 time_step      # 시간 간격 (기본 0.4초)
int32 num_samples      # 샘플 수 (기본 20)
int32 pred_horizon     # 예측 시점 수 (기본 12)
