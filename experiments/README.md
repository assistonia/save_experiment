# Experiment: Local vs CIGP vs Predictive Planning

3가지 글로벌 플래너 비교 실험

## 실험 구조

| 조건 | Global Planner | 설명 |
|------|----------------|------|
| **Baseline** | None | Local Planner만 사용 |
| **CIGP** | CCTV-Informed Global Planner | 논문 방법 |
| **Predictive** | Predictive Planning (Time-aware A*) | 시간 인덱스 예측 기반 |

## 비교 조건 예시

```
DWA           - Local only (Baseline)
CIGP-DWA      - CIGP + DWA
PRED-DWA      - Predictive Planning + DWA

DRL_VO        - Local only (Baseline)
CIGP-DRL_VO   - CIGP + DRL-VO
PRED-DRL_VO   - Predictive Planning + DRL-VO
```

## 평가 메트릭 (논문 기준)

### Navigation Quality
- **SR (Success Rate)**: 성공률 (%)
- **Vavg**: 평균 속도 (m/s)
- **ωavg**: 방향 변화 부드러움 (rad/s)

### Social Awareness
- **ITR (Intrusion Time Ratio)**: 개인 공간 침범 비율
- **SD (Social Distance)**: 평균 사회적 거리 (m)

## 사용법

### 1. 3가지 조건 전체 비교 (권장)

```bash
cd /home/pyongjoo/Desktop/newstart/environment/experiments

# DWA 비교 (각 조건 10 에피소드씩)
./run_full_comparison.sh dwa baseline 10

# DRL-VO 비교
./run_full_comparison.sh drl_vo baseline 10

# 혼잡 시나리오
./run_full_comparison.sh dwa congestion 20
```

### 2. 단일 조건 실험

```bash
# Local Only (Baseline)
./run_experiment.sh --planner dwa --global-planner none --episodes 100

# CIGP + Local
./run_experiment.sh --planner dwa --global-planner cigp --episodes 100

# Predictive Planning + Local
./run_experiment.sh --planner dwa --global-planner predictive --episodes 100
```

### 3. CIGP만 비교 (기존 방식)

```bash
./run_comparison.sh dwa baseline 10
```

## 사용 가능한 플래너

### Local Planners
- `dwa`: Dynamic Window Approach
- `drl_vo`: DRL-VO (Deep RL + Vision Obstacle)
- `teb`: Timed Elastic Band
- `orca`: Optimal Reciprocal Collision Avoidance
- `sfm`: Social Force Model

### Global Planners
- `none`: 글로벌 플래너 없음 (Baseline)
- `cigp`: CCTV-Informed Global Planner
- `predictive`: Predictive Planning (Time-aware A*)

## 사용 가능한 시나리오

- `warehouse_pedsim.xml` / `baseline`: 기본 (8명)
- `scenario_block_heavy.xml` / `congestion`: 혼잡 (15명+)
- `scenario_congestion_all.xml` / `circulation`: 매우 혼잡 (20명+)

## 결과 구조

```
experiments/results/full_comparison_dwa_20241130_120000/
├── episodes/              # 개별 에피소드 결과
│   ├── ep_0000_DWA_warehouse_pedsim.json
│   ├── ep_0010_CIGP-DWA_warehouse_pedsim.json
│   ├── ep_0020_PRED-DWA_warehouse_pedsim.json
│   └── ...
├── trajectories/          # 궤적 데이터
├── analysis/              # 분석 결과
│   ├── summary.txt        # 요약 테이블
│   ├── comparisons.json   # 비교 결과
│   └── aggregated_metrics.csv
├── all_episodes.json
└── all_episodes.csv
```

## 결과 예시

```
================================================================================
EXPERIMENT RESULTS SUMMARY
================================================================================

Scenario: Warehouse Pedsim
================================================================================

Method               SR(%)      Vavg       ωavg       ITR        SD(m)
--------------------------------------------------------------------------------
DWA                  68.0       0.67       0.16       0.18       2.38
CIGP-DWA             79.0       0.88       0.14       0.13       2.47
PRED-DWA             82.0       0.85       0.12       0.11       2.65
```

## 파일 구조

```
experiments/
├── config.py                  # 실험 설정 (3가지 글로벌 플래너)
├── metrics_collector.py       # 메트릭 수집기
├── data_logger.py             # 데이터 로거/분석기
├── experiment_runner.py       # 실험 러너 (ROS 노드)
├── run_experiment.sh          # 단일 조건 실험
├── run_comparison.sh          # CIGP 비교 (2조건)
├── run_full_comparison.sh     # 전체 비교 (3조건)
└── README.md
```

## 주의사항

1. **기존 코드 수정 없음**: 실험 코드는 완전히 독립적으로 동작
2. **Docker 필요**: `gdae_with_navigation:yolo` 이미지 필요
3. **GPU 필요**: DRL-VO, Predictive Planning 사용 시 GPU 필요
4. **SingularTrajectory 필요**: Predictive Planning 사용 시 필요

## 참고

- CIGP 논문: Kim et al., "CCTV-Informed Human-Aware Robot Navigation", IEEE RA-L 2024
- 100 에피소드 × 3 조건 = 약 4-5시간 소요 (시나리오당)
