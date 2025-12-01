#!/bin/bash
# ==============================================================================
# Comparison Runner - GUI Mode (로봇 움직임 확인용)
#
# Robot (DWA only) vs CIGP 비교 - GUI 모드 (화면에서 직접 확인)
# 빠른 테스트: 각 모듈당 3개 에피소드
#
# Usage:
#   ./run_comparison_gui.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/results/comparison_gui_${TIMESTAMP}"

mkdir -p "${RESULTS_DIR}"

echo "=============================================================="
echo " Comparison Test - GUI Mode"
echo "=============================================================="
echo ""
echo " Modules: robot (DWA), cigp"
echo " Episodes per module: 3"
echo " Results: ${RESULTS_DIR}"
echo ""
echo "=============================================================="
echo ""

# Robot (DWA only) 테스트
echo "========================================"
echo " Testing: robot (DWA local planner)"
echo "========================================"
./run_single_experiment.sh --module robot --episodes 3 2>&1 | tee "${RESULTS_DIR}/robot_log.txt"

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
./run_single_experiment.sh --module cigp --episodes 3 2>&1 | tee "${RESULTS_DIR}/cigp_log.txt"

# 결과 복사
CIGP_RESULT=$(ls -td ${SCRIPT_DIR}/results/exp_cigp_* 2>/dev/null | head -1)
if [ -n "$CIGP_RESULT" ]; then
    cp -r "$CIGP_RESULT" "${RESULTS_DIR}/cigp/"
fi

# 결과 요약 생성
echo ""
echo "=============================================================="
echo " COMPARISON RESULTS - GUI Mode"
echo "=============================================================="

echo ""
echo "=== Robot (DWA) ==="
cat "${RESULTS_DIR}/robot/summary.json" 2>/dev/null || echo "No results"

echo ""
echo "=== CIGP ==="
cat "${RESULTS_DIR}/cigp/summary.json" 2>/dev/null || echo "No results"

# 시각화
echo ""
echo "Generating trajectory plots..."
python3 visualize_trajectory.py "${RESULTS_DIR}/robot" 2>/dev/null || true
python3 visualize_trajectory.py "${RESULTS_DIR}/cigp" 2>/dev/null || true

echo ""
echo "=============================================================="
echo " Results saved to: ${RESULTS_DIR}"
echo "=============================================================="
