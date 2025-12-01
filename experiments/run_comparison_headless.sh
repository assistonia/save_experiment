#!/bin/bash
# ==============================================================================
# Comparison Runner - Headless Mode (빠른 테스트용)
#
# Robot (DWA only) vs CIGP 비교 - Headless 모드 (GUI 없이 빠르게)
# 본격 테스트: 각 모듈당 5개 에피소드
#
# Usage:
#   ./run_comparison_headless.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results/comparison_headless_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================================="
echo " Comparison Test - Headless Mode"
echo "=============================================================="
echo ""
echo " Modules: robot (DWA), cigp"
echo " Episodes per module: 5"
echo " Results: ${RESULTS_DIR}"
echo ""
echo "=============================================================="
echo ""

# 기존 컨테이너 정리
echo "Cleaning up existing containers..."
docker stop gdae_pedsim_robot gdae_pedsim_cigp 2>/dev/null || true
docker rm gdae_pedsim_robot gdae_pedsim_cigp 2>/dev/null || true
sleep 3

# Robot (DWA only) 테스트
echo "========================================"
echo " Testing: robot (DWA local planner)"
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
echo " Testing: cigp (CIGP global planner)"
echo "========================================"
./run_single_experiment.sh --module cigp --episodes 5 --headless 2>&1 | tee "${RESULTS_DIR}/cigp_log.txt"

# 결과 복사
CIGP_RESULT=$(ls -td ${SCRIPT_DIR}/results/exp_cigp_* 2>/dev/null | head -1)
if [ -n "$CIGP_RESULT" ]; then
    cp -r "$CIGP_RESULT" "${RESULTS_DIR}/cigp/"
fi

# 결과 요약 생성
echo ""
echo "=============================================================="
echo " COMPARISON RESULTS - Headless Mode"
echo "=============================================================="

echo ""
echo "=== Robot (DWA) ==="
cat "${RESULTS_DIR}/robot/summary.json" 2>/dev/null || echo "No results"

echo ""
echo "=== CIGP ==="
cat "${RESULTS_DIR}/cigp/summary.json" 2>/dev/null || echo "No results"

# Python으로 비교 테이블 생성
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

print("\n" + "="*70)
print(" COMPARISON TABLE")
print("="*70)
print(f"{'Metric':<25} {'Robot (DWA)':<20} {'CIGP':<20}")
print("-"*70)

if robot and cigp:
    print(f"{'Success Rate':<25} {robot['success_rate']:.1f}%{'':<14} {cigp['success_rate']:.1f}%")
    print(f"{'Avg Time (s)':<25} {robot['avg_time']:.1f}{'':<17} {cigp['avg_time']:.1f}")
    print(f"{'Avg Velocity (m/s)':<25} {robot['avg_velocity']:.2f}{'':<16} {cigp['avg_velocity']:.2f}")
    print(f"{'Total Collisions':<25} {robot['total_collisions']}{'':<18} {cigp['total_collisions']}")
    print(f"{'Collision Rate':<25} {robot['collision_rate']:.2f}{'':<16} {cigp['collision_rate']:.2f}")
else:
    print("Results not available")

print("="*70)
PYTHON_SCRIPT

# 시각화
echo ""
echo "Generating trajectory plots..."
python3 visualize_trajectory.py "${RESULTS_DIR}/robot" 2>/dev/null || true
python3 visualize_trajectory.py "${RESULTS_DIR}/cigp" 2>/dev/null || true

echo ""
echo "=============================================================="
echo " Results saved to: ${RESULTS_DIR}"
echo "=============================================================="
