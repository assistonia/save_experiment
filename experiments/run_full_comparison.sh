#!/bin/bash
# ==============================================================================
# Full Cross-Comparison Experiment Runner
# Global Planners: robot, cigp, predictive
# Local Planners: dwa, teb, drl
# Scenarios: block_dynamic, crossing, congestion_aisle1, block_heavy
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GLOBAL_PLANNERS=("robot" "cigp" "predictive")
LOCAL_PLANNERS=("dwa" "teb" "drl")
SCENARIOS=(
    "scenario_block_dynamic.xml"
    "scenario_crossing.xml"
    "scenario_congestion_aisle1.xml"
    "scenario_block_heavy.xml"
)
EPISODES=5

LOG_DIR="$SCRIPT_DIR/results/full_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " Full Cross-Comparison Experiment"
echo "=============================================================="
echo " Global Planners: ${GLOBAL_PLANNERS[*]}"
echo " Local Planners: ${LOCAL_PLANNERS[*]}"
echo " Scenarios: ${#SCENARIOS[@]} scenarios"
echo " Episodes per experiment: $EPISODES"
echo " Total experiments: $((${#GLOBAL_PLANNERS[@]} * ${#LOCAL_PLANNERS[@]} * ${#SCENARIOS[@]}))"
echo " Log directory: $LOG_DIR"
echo "=============================================================="

# 진행 상황 저장
PROGRESS_FILE="$LOG_DIR/progress.txt"
SUMMARY_FILE="$LOG_DIR/summary.csv"

# CSV 헤더
echo "scenario,global_planner,local_planner,success_rate,avg_time,avg_velocity,path_efficiency,jerk,angular_variance,min_human_dist,total_collisions,avg_itr" > "$SUMMARY_FILE"

total_experiments=$((${#GLOBAL_PLANNERS[@]} * ${#LOCAL_PLANNERS[@]} * ${#SCENARIOS[@]}))
current_exp=0

for scenario in "${SCENARIOS[@]}"; do
    scenario_name="${scenario%.xml}"

    for global in "${GLOBAL_PLANNERS[@]}"; do
        for local in "${LOCAL_PLANNERS[@]}"; do
            current_exp=$((current_exp + 1))

            echo ""
            echo "[$current_exp/$total_experiments] Running: $global + $local on $scenario_name"
            echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting $global + $local on $scenario_name" >> "$PROGRESS_FILE"

            # 실험 실행
            exp_log="$LOG_DIR/${scenario_name}_${global}_${local}.log"

            # 기존 컨테이너 정리
            docker ps -q | xargs -r docker stop 2>/dev/null || true
            sleep 2

            # 실험 실행 (최대 10분)
            timeout 600 ./run_single_experiment.sh \
                --module "$global" \
                --local-planner "$local" \
                --scenario "$scenario" \
                --episodes "$EPISODES" \
                --headless 2>&1 | tee "$exp_log" || {
                    echo "  [WARNING] Experiment timed out or failed"
                    echo "$(date '+%Y-%m-%d %H:%M:%S') - FAILED: $global + $local on $scenario_name" >> "$PROGRESS_FILE"
                    continue
                }

            # 결과 추출
            result_dir=$(grep "Results:" "$exp_log" | tail -1 | awk '{print $2}')

            if [ -f "$result_dir/summary.json" ]; then
                # JSON에서 값 추출
                success_rate=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('success_rate', 0))")
                avg_time=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_time', 0))")
                avg_velocity=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_velocity', 0))")
                path_efficiency=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_path_efficiency', 0))")
                jerk=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_jerk', 0))")
                angular_var=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_angular_variance', 0))")
                min_dist=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('min_human_dist', 0))")
                collisions=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('total_collisions', 0))")
                avg_itr=$(python3 -c "import json; d=json.load(open('$result_dir/summary.json')); print(d.get('avg_itr', 0))")

                # CSV에 추가
                echo "$scenario_name,$global,$local,$success_rate,$avg_time,$avg_velocity,$path_efficiency,$jerk,$angular_var,$min_dist,$collisions,$avg_itr" >> "$SUMMARY_FILE"

                echo "  [SUCCESS] Success Rate: ${success_rate}%, Collisions: $collisions"
                echo "$(date '+%Y-%m-%d %H:%M:%S') - SUCCESS: $global + $local on $scenario_name (SR: ${success_rate}%)" >> "$PROGRESS_FILE"
            else
                echo "  [ERROR] No summary.json found"
                echo "$(date '+%Y-%m-%d %H:%M:%S') - ERROR: $global + $local on $scenario_name - No summary" >> "$PROGRESS_FILE"
            fi

            # 컨테이너 정리
            docker ps -q | xargs -r docker stop 2>/dev/null || true
            sleep 3
        done
    done
done

echo ""
echo "=============================================================="
echo " Experiments Complete!"
echo "=============================================================="
echo " Results: $LOG_DIR"
echo " Summary: $SUMMARY_FILE"
echo ""

# 요약 출력
echo "=== Results Summary ==="
column -t -s',' "$SUMMARY_FILE"

echo ""
echo "Done!"
