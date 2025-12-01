#!/usr/bin/env python3
"""
Trajectory Prediction Integration Module

SingularTrajectory 기반 실시간 경로 예측 모듈.
기존 환경 코드를 수정하지 않고 독립적으로 동작.
"""

from .predictor import TrajectoryPredictor
from .prediction_config import PredictionConfig

__all__ = ['TrajectoryPredictor', 'PredictionConfig']
