#!/bin/bash
# 여러 시나리오를 실행하면서 캘리브레이션 데이터 수집

SCENARIOS=(
    "scenario_work_in_aisle1.xml"
    "scenario_work_in_aisle2.xml"
    "scenario_work_in_aisle3.xml"
    "scenario_work_in_aisle4.xml"
    "scenario_busy_warehouse.xml"
)

DURATION=40  # 각 시나리오당 수집 시간 (초)

echo "=========================================="
echo "Multi-Scenario Calibration"
echo "=========================================="

cd /home/pyongjoo/Desktop/newstart/environment/with_robot

for scenario in "${SCENARIOS[@]}"; do
    echo ""
    echo ">>> Starting: $scenario"
    echo ""

    # 시나리오 실행 (백그라운드)
    ./run_with_cigp.sh "$scenario" &
    SIM_PID=$!

    # 시뮬레이션 시작 대기
    echo "Waiting for simulation to start..."
    sleep 30

    # 캘리브레이션 데이터 수집
    echo "Collecting calibration data for ${DURATION}s..."
    docker exec gdae_pedsim_robot bash -c "source /opt/ros/noetic/setup.bash && source /root/DRL-robot-navigation/catkin_ws/devel_isolated/setup.bash && cd /environment/getimage && timeout $DURATION python3 -c \"
import sys
sys.path.insert(0, '.')
from multi_scenario_calibration import collect_from_scenario
data = collect_from_scenario('$scenario', duration=$DURATION)
import json
with open('/environment/getimage/calib_${scenario%.xml}.json', 'w') as f:
    json.dump({k: [[[float(x) for x in p], [float(x) for x in w]] for p,w in v] for k,v in data.items()}, f)
print('Saved to calib_${scenario%.xml}.json')
\""

    # 시뮬레이션 종료
    echo "Stopping simulation..."
    docker stop gdae_pedsim_robot 2>/dev/null
    kill $SIM_PID 2>/dev/null
    wait $SIM_PID 2>/dev/null

    sleep 5
done

echo ""
echo "=========================================="
echo "All scenarios completed!"
echo "=========================================="
