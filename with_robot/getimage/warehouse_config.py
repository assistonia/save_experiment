"""
Warehouse Configuration

선반, 통로, CCTV 위치/방향 설정.
warehouse.world 파일 기반.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict


# ==================== 맵 범위 ====================
MAP_X_RANGE = (-12.0, 12.0)
MAP_Y_RANGE = (-12.0, 12.0)


# ==================== 선반 정의 ====================
# 각 선반: (x_min, x_max, y_min, y_max)
SHELVES = {
    'shelf_1': (-12, -10, -5, 4),
    'shelf_2': (-7, -5, -5, 4),
    'shelf_3': (-2, 0, -5, 4),
    'shelf_4': (3, 5, -5, 4),
    'shelf_5': (10, 12, -5, 4),
}


# ==================== 통로 정의 ====================
# 세로 통로 (선반 사이)
AISLES = {
    'aisle_1': (-10, -7, -5, 4),   # 선반1-2 사이
    'aisle_2': (-5, -2, -5, 4),    # 선반2-3 사이
    'aisle_3': (0, 3, -5, 4),      # 선반3-4 사이
    'aisle_4': (5, 10, -5, 4),     # 선반4-5 사이 (넓음)
}

# 가로 통로 (상단/하단)
CORRIDORS = {
    'top_corridor': (-12, 12, 4, 12),     # 상단 통로
    'bottom_corridor': (-12, 12, -12, -5), # 하단 통로
}


# ==================== CCTV 설정 ====================
@dataclass
class CCTVConfig:
    """CCTV 카메라 설정"""
    id: int
    position: Tuple[float, float, float]  # (x, y, z)
    yaw: float      # 수평 방향 (라디안)
    pitch: float    # 틸트 (라디안, 아래가 양수)
    fov: float      # 시야각 (라디안)
    image_width: int
    image_height: int
    target_aisle: str  # 담당 통로


# warehouse.world에서 추출한 CCTV 설정
CCTV_CONFIGS = {
    0: CCTVConfig(
        id=0,
        position=(-8.5, 10.0, 3.0),
        yaw=-np.pi/2,       # -90° (아래쪽/남쪽)
        pitch=0.5,          # 28.6° 아래로
        fov=1.047,          # 60°
        image_width=640,
        image_height=480,
        target_aisle='aisle_1'
    ),
    1: CCTVConfig(
        id=1,
        position=(-3.5, -10.0, 3.0),
        yaw=np.pi/2,        # +90° (위쪽/북쪽)
        pitch=0.5,
        fov=1.047,
        image_width=640,
        image_height=480,
        target_aisle='aisle_2'
    ),
    2: CCTVConfig(
        id=2,
        position=(1.5, 10.0, 3.0),
        yaw=-np.pi/2,       # -90° (아래쪽/남쪽)
        pitch=0.5,
        fov=1.047,
        image_width=640,
        image_height=480,
        target_aisle='aisle_3'
    ),
    3: CCTVConfig(
        id=3,
        position=(7.5, -10.0, 3.0),
        yaw=np.pi/2,        # +90° (위쪽/북쪽)
        pitch=0.5,
        fov=1.047,
        image_width=640,
        image_height=480,
        target_aisle='aisle_4'
    ),
}


# ==================== 유틸리티 함수 ====================
def is_in_shelf(x: float, y: float) -> bool:
    """좌표가 선반 영역 내에 있는지 확인"""
    for shelf_name, (x_min, x_max, y_min, y_max) in SHELVES.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return True
    return False


def is_in_valid_region(x: float, y: float) -> bool:
    """좌표가 유효한 영역(통로)에 있는지 확인"""
    # 맵 범위 체크
    if not (MAP_X_RANGE[0] <= x <= MAP_X_RANGE[1]):
        return False
    if not (MAP_Y_RANGE[0] <= y <= MAP_Y_RANGE[1]):
        return False

    # 선반 영역이면 무효
    if is_in_shelf(x, y):
        return False

    return True


def get_aisle_for_position(x: float, y: float) -> str:
    """좌표가 어떤 통로에 있는지 반환"""
    for aisle_name, (x_min, x_max, y_min, y_max) in AISLES.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return aisle_name

    for corridor_name, (x_min, x_max, y_min, y_max) in CORRIDORS.items():
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return corridor_name

    return 'unknown'


def get_cctv_for_aisle(aisle_name: str) -> int:
    """통로를 담당하는 CCTV ID 반환"""
    for cctv_id, config in CCTV_CONFIGS.items():
        if config.target_aisle == aisle_name:
            return cctv_id
    return -1


if __name__ == '__main__':
    # 테스트
    print("=== Warehouse Configuration ===")
    print(f"Map range: X={MAP_X_RANGE}, Y={MAP_Y_RANGE}")
    print(f"\nShelves: {len(SHELVES)}")
    for name, bounds in SHELVES.items():
        print(f"  {name}: {bounds}")

    print(f"\nAisles: {len(AISLES)}")
    for name, bounds in AISLES.items():
        print(f"  {name}: {bounds}")

    print(f"\nCCTVs: {len(CCTV_CONFIGS)}")
    for cctv_id, config in CCTV_CONFIGS.items():
        print(f"  CCTV {cctv_id}: pos={config.position}, yaw={np.degrees(config.yaw):.1f}°, target={config.target_aisle}")

    # 유효 영역 테스트
    test_points = [
        (-9, 0, "Aisle 1 (valid)"),
        (-6, 0, "Shelf 2 (invalid)"),
        (7.5, 0, "Aisle 4 (valid)"),
        (0, 8, "Top corridor (valid)"),
    ]
    print("\n=== Valid Region Test ===")
    for x, y, desc in test_points:
        valid = is_in_valid_region(x, y)
        aisle = get_aisle_for_position(x, y)
        print(f"  ({x}, {y}) {desc}: valid={valid}, aisle={aisle}")
