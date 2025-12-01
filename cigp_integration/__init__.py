"""
CIGP Integration Module for ROS PedSim Environment

기존 환경 코드를 수정하지 않고 CIGP(CCTV-Informed Global Planner)를
ROS PedSim 시뮬레이터와 통합.

Usage:
    # ROS 노드로 실행
    rosrun cigp_integration cigp_bridge_node.py

    # 또는 직접 실행
    python cigp_bridge_node.py
"""

from .cigp_config import CIGPConfig, get_config, WAREHOUSE_CONFIG

__all__ = ['CIGPConfig', 'get_config', 'WAREHOUSE_CONFIG']
