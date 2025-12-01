"""
CCTV Human Detection Module

CCTV 이미지에서 사람을 검출하고 월드 좌표로 변환.
"""

from .warehouse_config import (
    SHELVES, AISLES, CORRIDORS, CCTV_CONFIGS,
    MAP_X_RANGE, MAP_Y_RANGE,
    is_in_shelf, is_in_valid_region, get_aisle_for_position
)

from .detector import PersonDetector, Detection

from .homography import (
    HomographyTransformer, HomographyManager,
    create_default_calibration
)

from .human_state_extractor import (
    HumanState, HumanStateExtractor, TrajectoryPoint
)

from .visualizer import (
    WarehouseVisualizer, compute_detection_metrics
)

__all__ = [
    # Config
    'SHELVES', 'AISLES', 'CORRIDORS', 'CCTV_CONFIGS',
    'MAP_X_RANGE', 'MAP_Y_RANGE',
    'is_in_shelf', 'is_in_valid_region', 'get_aisle_for_position',

    # Detector
    'PersonDetector', 'Detection',

    # Homography
    'HomographyTransformer', 'HomographyManager',
    'create_default_calibration',

    # Extractor
    'HumanState', 'HumanStateExtractor', 'TrajectoryPoint',

    # Visualizer
    'WarehouseVisualizer', 'compute_detection_metrics',
]
