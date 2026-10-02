"""
三角格子动量路径工具

从 spectral_calculation.py 中剥离出的、与物理模型无关的路径构建函数。
仅依赖 numpy，供 JAX 计算流程复用。
"""

import numpy as np
from typing import Tuple


def build_k_path(high_symmetry_points: list, labels: list, n_points: int) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    构建高对称点路径

    Parameters:
    -----------
    high_symmetry_points : list
        高对称点列表，每个元素是[kx, ky]
    labels : list
        高对称点标签
    n_points : int
        每段路径的点数

    Returns:
    --------
    tuple
        (k_path, k_distances, k_tick_positions) - k路径，距离，标记位置
    """
    n_segments = len(high_symmetry_points) - 1
    k_path = []
    k_distances = []
    k_tick_positions = []

    distance = 0.0

    for i in range(n_segments):
        start_point = np.array(high_symmetry_points[i])
        end_point = np.array(high_symmetry_points[i + 1])

        if i == 0:
            k_tick_positions.append(distance)

        # 生成路径点
        for j in range(n_points):
            if i == n_segments - 1 and j == n_points - 1:
                # 最后一个点
                point = end_point
            else:
                t = j / n_points
                point = (1 - t) * start_point + t * end_point

            k_path.append(point)
            k_distances.append(distance)

            if j < n_points - 1:
                # 计算到下一个点的距离
                next_t = (j + 1) / n_points
                next_point = (1 - next_t) * start_point + next_t * end_point
                distance += np.linalg.norm(next_point - point)

        # 添加段结束点的标记位置
        k_tick_positions.append(distance)

    return np.array(k_path), np.array(k_distances), k_tick_positions


def get_triangular_lattice_path(L1: int = 10) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    获取三角格子的标准高对称点路径

    Returns:
    --------
    tuple
        (k_path, k_distances, k_tick_positions, labels) - 标准三角格子路径
    """
    # 三角格子的高对称点
    high_symmetry_points = [
        [0.5, -0.25],    # X
        [0.5, 0],        # M
        [0, 0],          # Γ
        [2/3, 1/3],      # K'
        [1/2, 1/2],      # M
        [0, 0.25],       # Y
        [1/3, 2/3]       # K
    ]

    labels = ['X', 'M', 'Γ', "K'", 'M', 'Y', 'K']

    return build_k_path(high_symmetry_points, labels, L1*L1)


def convert_to_cartesian_coordinates(k_path: np.ndarray) -> np.ndarray:
    """
    将相对坐标转换为笛卡尔坐标

    Parameters:
    -----------
    k_path : ndarray
        相对坐标中的k路径

    Returns:
    --------
    ndarray
        笛卡尔坐标中的k路径
    """
    # 三角格子的倒格矢
    a1 = np.array([1, 0])
    a2 = np.array([0.5, np.sqrt(3)/2])
    b1 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])
    b2 = np.array([0, 4*np.pi/np.sqrt(3)])

    # 转换矩阵
    b_matrix = np.array([b1, b2])
    a_matrix = np.array([a1, a2])
    t_matrix = b_matrix @ a_matrix.T

    k_cartesian = np.zeros_like(k_path)
    for i, k_point in enumerate(k_path):
        k_cartesian[i] = k_point @ t_matrix

    return k_cartesian
