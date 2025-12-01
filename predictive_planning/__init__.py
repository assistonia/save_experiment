#!/usr/bin/env python3
"""
Predictive Planning Module

SingularTrajectory 예측 결과를 활용한 확률적 경로 계획 모듈.
기존 trajectory_prediction, cigp_integration 모듈과 독립적으로 동작.

주요 구성요소:
- PredictedTrajectoryArray: 20개 샘플 예측 데이터 메시지
- PredictiveCostCalculator: 시간 인식 확률적 비용 계산
- PredictiveGlobalPlanner: A* 기반 예측 경로 계획
- PredictivePlanningBridge: ROS 통합 브릿지 노드
"""

__version__ = '1.0.0'
__author__ = 'predictive_planning'
