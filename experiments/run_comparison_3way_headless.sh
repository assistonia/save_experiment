#!/bin/bash
# ==============================================================================
# 3-Way Comparison Runner - Headless Mode
#
# Robot (DWA only) vs CIGP vs Predictive Planning 비교
# Headless 모드 (GUI 없이 빠르게)
#
# Usage:
#   ./run_comparison_3way_headless.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results/comparison_3way_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================================="
echo " 3-Way Comparison Test - Headless Mode"
echo "=============================================================="
echo ""
echo " Modules: robot (DWA), cigp, predictive"
echo " Episodes per module: 5"
echo " Results: ${RESULTS_DIR}"
echo ""
echo "=============================================================="
echo ""

# 기존 컨테이너 정리
echo "Cleaning up existing containers..."
docker stop gdae_pedsim_robot gdae_pedsim_cigp gdae_pedsim_predictive_planning 2>/dev/null || true
docker rm gdae_pedsim_robot gdae_pedsim_cigp gdae_pedsim_predictive_planning 2>/dev/null || true
sleep 3

# Robot (DWA only) 테스트
echo "========================================"
echo " [1/3] Testing: robot (DWA local planner)"
echo "========================================"
./run_single_experiment.sh --module robot --episodes 5 --headless 2>&1 | tee "${RESULTS_DIR}/robot_log.txt"

# 결과 복사
ROBOT_RESULT=$(ls -td ${SCRIPT_DIR}/results/exp_robot_* 2>/dev/null | head -1)
if [ -n "$ROBOT_RESULT" ]; then
    cp -r "$ROBOT_RESULT" "${RESULTS_DIR}/robot/"
fi

sleep 5

# CIGP 테스트
echo ""
echo "========================================"
echo " [2/3] Testing: cigp (CIGP global planner)"
echo "========================================"
./run_single_experiment.sh --module cigp --episodes 5 --headless 2>&1 | tee "${RESULTS_DIR}/cigp_log.txt"

# 결과 복사
CIGP_RESULT=$(ls -td ${SCRIPT_DIR}/results/exp_cigp_* 2>/dev/null | head -1)
if [ -n "$CIGP_RESULT" ]; then
    cp -r "$CIGP_RESULT" "${RESULTS_DIR}/cigp/"
fi

sleep 5

# Predictive Planning 테스트
echo ""
echo "========================================"
echo " [3/3] Testing: predictive (Predictive Planning)"
echo "========================================"
./run_single_experiment.sh --module predictive --episodes 5 --headless 2>&1 | tee "${RESULTS_DIR}/predictive_log.txt"

# 결과 복사
PREDICTIVE_RESULT=$(ls -td ${SCRIPT_DIR}/results/exp_predictive_* 2>/dev/null | head -1)
if [ -n "$PREDICTIVE_RESULT" ]; then
    cp -r "$PREDICTIVE_RESULT" "${RESULTS_DIR}/predictive/"
fi

# 결과 요약 생성
echo ""
echo "=============================================================="
echo " 3-WAY COMPARISON RESULTS"
echo "=============================================================="

echo ""
echo "=== Robot (DWA) ==="
cat "${RESULTS_DIR}/robot/summary.json" 2>/dev/null || echo "No results"

echo ""
echo "=== CIGP ==="
cat "${RESULTS_DIR}/cigp/summary.json" 2>/dev/null || echo "No results"

echo ""
echo "=== Predictive ==="
cat "${RESULTS_DIR}/predictive/summary.json" 2>/dev/null || echo "No results"

# Python으로 비교 테이블 생성
export RESULTS_DIR
python3 << 'PYTHON_SCRIPT'
import json
import os

results_dir = os.environ.get('RESULTS_DIR', '.')

def load_summary(path):
    try:
        with open(f"{path}/summary.json") as f:
            return json.load(f)
    except:
        return None

robot = load_summary(f"{results_dir}/robot")
cigp = load_summary(f"{results_dir}/cigp")
pred = load_summary(f"{results_dir}/predictive")

print("\n" + "="*90)
print(" 3-WAY COMPARISON TABLE")
print("="*90)
print(f"{'Metric':<25} {'Robot (DWA)':<20} {'CIGP':<20} {'Predictive':<20}")
print("-"*90)

def fmt(val, fmt_str=".1f"):
    if val is None:
        return "N/A"
    return f"{val:{fmt_str}}"

def get_val(d, key, default=None):
    if d is None:
        return default
    return d.get(key, default)

# Success Rate
print(f"{'Success Rate (%)':<25} {fmt(get_val(robot, 'success_rate')):<20} {fmt(get_val(cigp, 'success_rate')):<20} {fmt(get_val(pred, 'success_rate')):<20}")

# Avg Time
print(f"{'Avg Time (s)':<25} {fmt(get_val(robot, 'avg_time')):<20} {fmt(get_val(cigp, 'avg_time')):<20} {fmt(get_val(pred, 'avg_time')):<20}")

# Avg Velocity
print(f"{'Avg Velocity (m/s)':<25} {fmt(get_val(robot, 'avg_velocity'), '.2f'):<20} {fmt(get_val(cigp, 'avg_velocity'), '.2f'):<20} {fmt(get_val(pred, 'avg_velocity'), '.2f'):<20}")

# Avg Angular Velocity
print(f"{'Avg Angular Vel (rad/s)':<25} {fmt(get_val(robot, 'avg_angular_velocity'), '.2f'):<20} {fmt(get_val(cigp, 'avg_angular_velocity'), '.2f'):<20} {fmt(get_val(pred, 'avg_angular_velocity'), '.2f'):<20}")

# Avg Path Length
print(f"{'Avg Path Length (m)':<25} {fmt(get_val(robot, 'avg_path_length')):<20} {fmt(get_val(cigp, 'avg_path_length')):<20} {fmt(get_val(pred, 'avg_path_length')):<20}")

# Total Collisions
r_col = get_val(robot, 'total_collisions', 'N/A')
c_col = get_val(cigp, 'total_collisions', 'N/A')
p_col = get_val(pred, 'total_collisions', 'N/A')
print(f"{'Total Collisions':<25} {str(r_col):<20} {str(c_col):<20} {str(p_col):<20}")

# Collision Rate
print(f"{'Collision Rate':<25} {fmt(get_val(robot, 'collision_rate'), '.2f'):<20} {fmt(get_val(cigp, 'collision_rate'), '.2f'):<20} {fmt(get_val(pred, 'collision_rate'), '.2f'):<20}")

# ITR
print(f"{'Avg ITR':<25} {fmt(get_val(robot, 'avg_itr'), '.3f'):<20} {fmt(get_val(cigp, 'avg_itr'), '.3f'):<20} {fmt(get_val(pred, 'avg_itr'), '.3f'):<20}")

# Min Human Distance
print(f"{'Min Human Dist (m)':<25} {fmt(get_val(robot, 'min_human_dist'), '.2f'):<20} {fmt(get_val(cigp, 'min_human_dist'), '.2f'):<20} {fmt(get_val(pred, 'min_human_dist'), '.2f'):<20}")

print("="*90)

# 결과 저장
comparison = {
    'robot': robot,
    'cigp': cigp,
    'predictive': pred
}
with open(f"{results_dir}/comparison_summary.json", 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"\nComparison summary saved to: {results_dir}/comparison_summary.json")
PYTHON_SCRIPT

# 시각화
echo ""
echo "Generating trajectory plots..."
python3 visualize_trajectory.py "${RESULTS_DIR}/robot" 2>/dev/null || true
python3 visualize_trajectory.py "${RESULTS_DIR}/cigp" 2>/dev/null || true
python3 visualize_trajectory.py "${RESULTS_DIR}/predictive" 2>/dev/null || true

echo ""
echo "=============================================================="
echo " Results saved to: ${RESULTS_DIR}"
echo "=============================================================="
