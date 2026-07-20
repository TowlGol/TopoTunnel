"""
Mapped tunnel path preprocessing script.

Workflow:
1. Read the mapped_path_points.csv data
2. Snap tunnel-path points to grid centers (grid discretization)
3. Apply Laplacian smoothing after grid discretization
4. Optionally apply RDP (Douglas-Peucker) simplification
5. Compute residue positions and summary statistics
6. Optionally cluster paths by outlet distance and path-length difference
7. Save the preprocessed data and prepare a dataset directory for loading

Usage:
    python stage_05_preprocess_dataset.py

Output files:
    - preprocessed_paths.pkl
    - preprocessed_paths.db
    - residue_statistics.csv
    - residue_combination_statistics.csv
    - representative_frame.pdb
    - timing_summary.txt
"""

import os
import pickle
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from path_simplification import rdp_simplify_with_attributes
from pipeline_config import (
    all_points_csv,
    preprocess_pkl,
)

# =====================================================
# 配置参数
# =====================================================
INPUT_CSV = all_points_csv
OUTPUT_PKL = preprocess_pkl
BASE_DIR = Path(__file__).resolve().parent

# 拉普拉斯平滑参数
SMOOTH_ITERATIONS = 50  # 平滑迭代次数
SMOOTH_ALPHA = 0.8      # 平滑强度 (0-1)
SMOOTH_MIN_POINTS = 4   # 最少点数才进行平滑

# RDP (Douglas-Peucker) 简化参数
RDP_ENABLED = False      # 是否启用 RDP 简化
RDP_EPSILON = 0.15      # 最大偏差阈值（埃），越小保真度越高
RDP_MIN_POINTS = 5      # 每条路径最少保留的点数

# 网格离散化参数
GRID_ENABLED = False     # 是否启用网格离散化
GRID_SIZE = 2        # 网格边长（埃），路径点将被吸附到所在网格的中心点

# 聚类参数（差并集算法：终点距离 + 路径长度差）
CLUSTER_ENABLED = False       # 是否启用路径聚类（已暂停）
CLUSTER_ENDPOINT_DIST = 5.0   # 终点距离阈值 x（埃）：两路径终点距离 < x 则视为同类
CLUSTER_LENGTH_DIFF = 5.0    # 路径长度差阈值 y（埃）：两路径长度差 < y 则视为同类

# 多进程参数
NUM_WORKERS = int(os.environ.get("PATHVIS_PREPROCESS_WORKERS", max(1, multiprocessing.cpu_count() - 1)))  # 保留一个核心给系统
CHUNK_SIZE = 200  # 每个任务处理的路径数量（小批次更好利用多核）

# 数据类型映射（内存优化）
DTYPE_MAPPING = {
    'x': 'float32',
    'y': 'float32',
    'z': 'float32',
    'x_origin': 'float32',
    'y_origin': 'float32',
    'z_origin': 'float32',
    'bfactor': 'float32',
    'occupancy': 'float32',
    'length': 'float32',
    'throughput': 'float32',
    'tunnel_cluster': 'int32',
    'qss': 'float32',
    'atom_serial': 'int32',
    'res_seq': 'int16',
    'residue_1': 'int16',
    'residue_2': 'int16',
    'residue_3': 'int16',
    'residue_4': 'int16',
    'atom_1': 'int32',
    'atom_2': 'int32',
    'atom_3': 'int32',
    'atom_4': 'int32',
    'frame_id': 'int32',
    'source_file': 'category',
    'element': 'category',
}
def configure_runtime_paths(input_csv=None, output_pkl=None):
    global INPUT_CSV
    global OUTPUT_PKL
    if input_csv is not None:
        INPUT_CSV = input_csv
    if output_pkl is not None:
        OUTPUT_PKL = output_pkl

# =====================================================
# 网格离散化函数
# =====================================================
def snap_to_grid(x, y, z, grid_size):
    """
    将坐标吸附到所在正方体网格的中心点

    参数:
        x, y, z: 坐标数组 (float32)
        grid_size: 网格边长（埃）

    返回:
        吸附后的 x, y, z 数组
    """
    ix = np.floor(x / grid_size).astype(np.int32)
    iy = np.floor(y / grid_size).astype(np.int32)
    iz = np.floor(z / grid_size).astype(np.int32)
    cx = (ix + 0.5) * grid_size
    cy = (iy + 0.5) * grid_size
    cz = (iz + 0.5) * grid_size
    return cx.astype(np.float32), cy.astype(np.float32), cz.astype(np.float32)


# =====================================================
# 拉普拉斯平滑函数
# =====================================================
def smooth_path_xyz(x, y, z, iterations=SMOOTH_ITERATIONS, alpha=SMOOTH_ALPHA, min_points=SMOOTH_MIN_POINTS):
    """
    对路径进行拉普拉斯平滑，保持端点不变

    参数:
        x, y, z: 坐标数组
        iterations: 迭代次数
        alpha: 平滑强度 (0-1)
        min_points: 最少点数

    返回:
        平滑后的 x, y, z 数组
    """
    n = len(x)
    if n < min_points:
        return x, y, z

    x_s = np.asarray(x, dtype=np.float32).copy()
    y_s = np.asarray(y, dtype=np.float32).copy()
    z_s = np.asarray(z, dtype=np.float32).copy()

    for _ in range(iterations):
        x_new = x_s.copy()
        y_new = y_s.copy()
        z_new = z_s.copy()

        # 计算相邻点的平均值
        x_avg = (x_s[:-2] + x_s[2:]) * 0.5
        y_avg = (y_s[:-2] + y_s[2:]) * 0.5
        z_avg = (z_s[:-2] + z_s[2:]) * 0.5

        # 向平均值方向移动（保持端点不变）
        x_new[1:-1] = x_s[1:-1] + alpha * (x_avg - x_s[1:-1])
        y_new[1:-1] = y_s[1:-1] + alpha * (y_avg - y_s[1:-1])
        z_new[1:-1] = z_s[1:-1] + alpha * (z_avg - z_s[1:-1])

        x_s, y_s, z_s = x_new, y_new, z_new

    return x_s, y_s, z_s


def clean_segment(seg):
    """清理路径段数据"""
    seg = seg.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y", "z", "bfactor"])
    return seg


def process_single_path(seg_data):
    """
    处理单条路径（平滑 + RDP简化）

    参数:
        seg_data: dict 包含路径数据的字典

    返回:
        处理后的路径字典，或 None（如果路径无效）
    """
    # 解包数据
    x = seg_data['x']
    y = seg_data['y']
    z = seg_data['z']
    b = seg_data['b']
    x_origin = seg_data['x_origin']
    y_origin = seg_data['y_origin']
    z_origin = seg_data['z_origin']
    frame_ids = seg_data['frame_ids']
    res_1 = seg_data['res_1']
    res_2 = seg_data['res_2']
    res_3 = seg_data['res_3']
    res_4 = seg_data['res_4']
    atom_1 = seg_data['atom_1']
    atom_2 = seg_data['atom_2']
    atom_3 = seg_data['atom_3']
    atom_4 = seg_data['atom_4']
    occupancy = seg_data['occupancy']
    length_raw = seg_data['length_raw']
    throughput = seg_data['throughput']
    qss_raw = seg_data['qss_raw']
    residues = seg_data['residues']
    residues_int_set = seg_data['residues_int_set']
    tunnel_cluster = seg_data['tunnel_cluster']

    original_points = len(x)
    path_timing = {
        "3c.path.unpack": 0.0,
        "3c.path.selection_copy": 0.0,
        "3c.path.smooth": 0.0,
        "3c.path.rdp": 0.0,
        "3c.path.finalize": 0.0,
        "3c.path.total": 0.0,
    }
    path_total_start = time.perf_counter()

    if original_points < 2:
        return None

    stage_start = time.perf_counter()
    # ========== 步骤1：网格离散化（先吸附到网格，去除噪声） ==========
    if GRID_ENABLED:
        x, y, z = snap_to_grid(x, y, z, GRID_SIZE)
    path_timing["3c.path.unpack"] += time.perf_counter() - stage_start

    # ========== 保存网格离散化后的点数据用于精确选择 ==========
    # 这些数据不经过平滑/RDP，确保框选时能精确检测
    stage_start = time.perf_counter()
    x_selection = np.asarray(x, dtype=np.float32).copy()
    y_selection = np.asarray(y, dtype=np.float32).copy()
    z_selection = np.asarray(z, dtype=np.float32).copy()
    path_timing["3c.path.selection_copy"] += time.perf_counter() - stage_start

    # ========== 步骤2：拉普拉斯平滑（在网格离散化之后） ==========
    stage_start = time.perf_counter()
    x_smooth, y_smooth, z_smooth = smooth_path_xyz(x, y, z)
    path_timing["3c.path.smooth"] += time.perf_counter() - stage_start

    # ========== 步骤3：RDP 简化（在平滑之后）- 仅用于渲染 ==========
    stage_start = time.perf_counter()
    if RDP_ENABLED and len(x_smooth) > RDP_MIN_POINTS:
        # 构建属性字典
        attributes = {
            'b': b,
            'frame_ids': frame_ids,
            'res_1': res_1,
            'res_2': res_2,
            'res_3': res_3,
            'res_4': res_4,
            'atom_1': atom_1,
            'atom_2': atom_2,
            'atom_3': atom_3,
            'atom_4': atom_4,
            'occupancy': occupancy,
            'throughput': throughput,
        }

        if length_raw is not None:
            attributes['length_raw'] = length_raw
        if qss_raw is not None:
            attributes['qss'] = qss_raw

        # 简化主坐标和属性
        x_smooth, y_smooth, z_smooth, simplified_attrs = rdp_simplify_with_attributes(
            x_smooth, y_smooth, z_smooth,
            attributes,
            epsilon=RDP_EPSILON,
            min_points=RDP_MIN_POINTS
        )

        # 更新属性数组
        b = simplified_attrs.get('b', b)
        frame_ids = simplified_attrs.get('frame_ids', frame_ids)
        res_1 = simplified_attrs.get('res_1', res_1)
        res_2 = simplified_attrs.get('res_2', res_2)
        res_3 = simplified_attrs.get('res_3', res_3)
        res_4 = simplified_attrs.get('res_4', res_4)
        atom_1 = simplified_attrs.get('atom_1', atom_1)
        atom_2 = simplified_attrs.get('atom_2', atom_2)
        atom_3 = simplified_attrs.get('atom_3', atom_3)
        atom_4 = simplified_attrs.get('atom_4', atom_4)
        occupancy = simplified_attrs.get('occupancy', occupancy)
        throughput = simplified_attrs.get('throughput', throughput)

        if 'length_raw' in simplified_attrs:
            length_raw = simplified_attrs['length_raw']
            length_relative = (length_raw - length_raw[0]).tolist()
        else:
            length_relative = list(range(len(x_smooth)))

        if 'qss' in simplified_attrs:
            qss = simplified_attrs['qss'].tolist()
        else:
            qss = [0.0] * len(x_smooth)
    else:
        if length_raw is not None:
            length_relative = (length_raw - length_raw[0]).tolist()
        else:
            length_relative = list(range(len(x_smooth)))

        qss = qss_raw.tolist() if qss_raw is not None else [0.0] * len(x_smooth)
    path_timing["3c.path.rdp"] += time.perf_counter() - stage_start

    simplified_points = len(x_smooth)

    # 计算 AABB (轴对齐包围盒) - 使用网格离散化后的点数据确保精度
    stage_start = time.perf_counter()
    bbox = {
        "min": [float(np.min(x_selection)), float(np.min(y_selection)), float(np.min(z_selection))],
        "max": [float(np.max(x_selection)), float(np.max(y_selection)), float(np.max(z_selection))]
    }
    path_timing["3c.path.finalize"] += time.perf_counter() - stage_start
    path_timing["3c.path.total"] = time.perf_counter() - path_total_start

    return {
        "path": {
            "xs": x_smooth.tolist(),
            "ys": y_smooth.tolist(),
            "zs": z_smooth.tolist(),
            "bf": b.tolist() if hasattr(b, 'tolist') else list(b),
            "qss": qss,
            "occupancy": occupancy.tolist() if hasattr(occupancy, 'tolist') else list(occupancy),
            "throughput": throughput.tolist() if hasattr(throughput, 'tolist') else list(throughput),
            "xs_origin": x_origin.tolist() if hasattr(x_origin, 'tolist') else list(x_origin),
            "ys_origin": y_origin.tolist() if hasattr(y_origin, 'tolist') else list(y_origin),
            "zs_origin": z_origin.tolist() if hasattr(z_origin, 'tolist') else list(z_origin),
            "frame_ids": frame_ids.tolist() if hasattr(frame_ids, 'tolist') else list(frame_ids),
            "res_1": res_1.tolist() if hasattr(res_1, 'tolist') else list(res_1),
            "res_2": res_2.tolist() if hasattr(res_2, 'tolist') else list(res_2),
            "res_3": res_3.tolist() if hasattr(res_3, 'tolist') else list(res_3),
            "res_4": res_4.tolist() if hasattr(res_4, 'tolist') else list(res_4),
            "atom_1": atom_1.tolist() if hasattr(atom_1, 'tolist') else list(atom_1),
            "atom_2": atom_2.tolist() if hasattr(atom_2, 'tolist') else list(atom_2),
            "atom_3": atom_3.tolist() if hasattr(atom_3, 'tolist') else list(atom_3),
            "atom_4": atom_4.tolist() if hasattr(atom_4, 'tolist') else list(atom_4),
            "length": length_relative,
            "residues": residues,
            "residues_set": residues_int_set,
            "bbox": bbox,
            # ========== 新增：原始点数据用于精确选择 ==========
            "xs_selection": x_selection.tolist(),
            "ys_selection": y_selection.tolist(),
            "zs_selection": z_selection.tolist(),
            "cluster_id": int(tunnel_cluster),
        },
        "original_points": original_points,
        "simplified_points": simplified_points,
        "timing": path_timing,
    }


def process_path_batch(batch_data):
    """
    处理一批路径（用于多进程）

    参数:
        batch_data: list of seg_data dicts

    返回:
        处理结果列表
    """
    results = []
    for seg_data in batch_data:
        result = process_single_path(seg_data)
        results.append(result)
    return results


def extract_segment_data(seg, columns):
    """
    从 DataFrame 段提取数据为字典（用于传递给子进程）
    """
    x = seg["x"].to_numpy(np.float32)
    y = seg["y"].to_numpy(np.float32)
    z = seg["z"].to_numpy(np.float32)
    b = seg["bfactor"].to_numpy(np.float32)

    x_origin = seg["x_origin"].to_numpy(np.float32) if "x_origin" in columns else x.copy()
    y_origin = seg["y_origin"].to_numpy(np.float32) if "y_origin" in columns else y.copy()
    z_origin = seg["z_origin"].to_numpy(np.float32) if "z_origin" in columns else z.copy()
    frame_ids = seg["frame_id"].to_numpy(np.int32) if "frame_id" in columns else np.zeros(len(x), dtype=np.int32)

    res_1 = seg["residue_1"].to_numpy(np.int16) if "residue_1" in columns else np.zeros(len(x), dtype=np.int16)
    res_2 = seg["residue_2"].to_numpy(np.int16) if "residue_2" in columns else np.zeros(len(x), dtype=np.int16)
    res_3 = seg["residue_3"].to_numpy(np.int16) if "residue_3" in columns else np.zeros(len(x), dtype=np.int16)
    res_4 = seg["residue_4"].to_numpy(np.int16) if "residue_4" in columns else np.zeros(len(x), dtype=np.int16)
    atom_1 = seg["atom_1"].to_numpy(np.int32) if "atom_1" in columns else np.zeros(len(x), dtype=np.int32)
    atom_2 = seg["atom_2"].to_numpy(np.int32) if "atom_2" in columns else np.zeros(len(x), dtype=np.int32)
    atom_3 = seg["atom_3"].to_numpy(np.int32) if "atom_3" in columns else np.zeros(len(x), dtype=np.int32)
    atom_4 = seg["atom_4"].to_numpy(np.int32) if "atom_4" in columns else np.zeros(len(x), dtype=np.int32)

    occupancy = seg["occupancy"].to_numpy(np.float32) if "occupancy" in columns else np.ones(len(x), dtype=np.float32)
    length_raw = seg["length"].to_numpy(np.float32) if "length" in columns else None
    throughput = seg["throughput"].to_numpy(np.float32) if "throughput" in columns else np.zeros(len(x), dtype=np.float32)
    if "tunnel_cluster" in columns:
        tunnel_cluster_values = pd.to_numeric(seg["tunnel_cluster"], errors="coerce").dropna()
        tunnel_cluster = int(tunnel_cluster_values.iloc[0]) if not tunnel_cluster_values.empty else 0
    else:
        tunnel_cluster = 0
    # Compute qss as average of 4 nearest-residue hydrophobicity values (qss1-4)
    if "qss" in columns:
        qss_raw = seg["qss"].to_numpy(np.float32)
    elif all(c in columns for c in ("qss1", "qss2", "qss3", "qss4")):
        qss_raw = (
            seg["qss1"].to_numpy(np.float32)
            + seg["qss2"].to_numpy(np.float32)
            + seg["qss3"].to_numpy(np.float32)
            + seg["qss4"].to_numpy(np.float32)
        ) / 4.0
    else:
        qss_raw = None

    # 残基集合
    residues = list(set().union(
        seg["residue_1"], seg["residue_2"],
        seg["residue_3"], seg["residue_4"]
    ))

    residues_int_set = set()
    for res in residues:
        try:
            residues_int_set.add(int(res))
        except (ValueError, TypeError):
            pass

    return {
        'x': x, 'y': y, 'z': z, 'b': b,
        'x_origin': x_origin, 'y_origin': y_origin, 'z_origin': z_origin,
        'frame_ids': frame_ids,
        'res_1': res_1, 'res_2': res_2, 'res_3': res_3, 'res_4': res_4,
        'atom_1': atom_1, 'atom_2': atom_2, 'atom_3': atom_3, 'atom_4': atom_4,
        'occupancy': occupancy,
        'throughput': throughput,
        'length_raw': length_raw,
        'qss_raw': qss_raw,
        'residues': residues,
        'residues_int_set': residues_int_set,
        'tunnel_cluster': tunnel_cluster,
    }


def prepare_segment_arrays(df, columns):
    x = df["x"].to_numpy(np.float32, copy=False)
    y = df["y"].to_numpy(np.float32, copy=False)
    z = df["z"].to_numpy(np.float32, copy=False)
    b = df["bfactor"].to_numpy(np.float32, copy=False)
    valid_points = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & np.isfinite(b)

    if "x_origin" in columns:
        x_origin = df["x_origin"].to_numpy(np.float32, copy=False)
        y_origin = df["y_origin"].to_numpy(np.float32, copy=False)
        z_origin = df["z_origin"].to_numpy(np.float32, copy=False)
    else:
        x_origin = x
        y_origin = y
        z_origin = z

    if "frame_id" in columns:
        frame_ids = df["frame_id"].to_numpy(np.int32, copy=False)
    else:
        frame_ids = np.zeros(len(df), dtype=np.int32)

    residue_arrays = {
        "res_1": df["residue_1"].to_numpy(np.int16, copy=False) if "residue_1" in columns else np.zeros(len(df), dtype=np.int16),
        "res_2": df["residue_2"].to_numpy(np.int16, copy=False) if "residue_2" in columns else np.zeros(len(df), dtype=np.int16),
        "res_3": df["residue_3"].to_numpy(np.int16, copy=False) if "residue_3" in columns else np.zeros(len(df), dtype=np.int16),
        "res_4": df["residue_4"].to_numpy(np.int16, copy=False) if "residue_4" in columns else np.zeros(len(df), dtype=np.int16),
    }
    atom_arrays = {
        "atom_1": df["atom_1"].to_numpy(np.int32, copy=False) if "atom_1" in columns else np.zeros(len(df), dtype=np.int32),
        "atom_2": df["atom_2"].to_numpy(np.int32, copy=False) if "atom_2" in columns else np.zeros(len(df), dtype=np.int32),
        "atom_3": df["atom_3"].to_numpy(np.int32, copy=False) if "atom_3" in columns else np.zeros(len(df), dtype=np.int32),
        "atom_4": df["atom_4"].to_numpy(np.int32, copy=False) if "atom_4" in columns else np.zeros(len(df), dtype=np.int32),
    }

    occupancy = df["occupancy"].to_numpy(np.float32, copy=False) if "occupancy" in columns else np.ones(len(df), dtype=np.float32)
    length_raw = df["length"].to_numpy(np.float32, copy=False) if "length" in columns else None
    throughput = df["throughput"].to_numpy(np.float32, copy=False) if "throughput" in columns else np.zeros(len(df), dtype=np.float32)
    tunnel_cluster = df["tunnel_cluster"].to_numpy(np.int32, copy=False) if "tunnel_cluster" in columns else np.zeros(len(df), dtype=np.int32)

    if "qss" in columns:
        qss_raw = df["qss"].to_numpy(np.float32, copy=False)
    elif all(c in columns for c in ("qss1", "qss2", "qss3", "qss4")):
        qss_raw = (
            df["qss1"].to_numpy(np.float32, copy=False)
            + df["qss2"].to_numpy(np.float32, copy=False)
            + df["qss3"].to_numpy(np.float32, copy=False)
            + df["qss4"].to_numpy(np.float32, copy=False)
        ) / 4.0
    else:
        qss_raw = None

    return {
        "x": x,
        "y": y,
        "z": z,
        "b": b,
        "x_origin": x_origin,
        "y_origin": y_origin,
        "z_origin": z_origin,
        "frame_ids": frame_ids,
        "occupancy": occupancy,
        "length_raw": length_raw,
        "throughput": throughput,
        "tunnel_cluster": tunnel_cluster,
        "qss_raw": qss_raw,
        "valid_points": valid_points,
        **residue_arrays,
        **atom_arrays,
    }


def extract_segment_data_from_arrays(segment_arrays, start_idx, end_idx):
    valid_mask = segment_arrays["valid_points"][start_idx:end_idx]
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count < 2:
        return None, 0

    segment_slice = slice(start_idx, end_idx)

    def take(name):
        values = segment_arrays[name][segment_slice]
        if valid_count == len(values):
            return np.array(values, copy=True)
        return np.array(values[valid_mask], copy=True)

    x = take("x")
    y = take("y")
    z = take("z")
    b = take("b")
    x_origin = take("x_origin")
    y_origin = take("y_origin")
    z_origin = take("z_origin")
    frame_ids = take("frame_ids")
    res_1 = take("res_1")
    res_2 = take("res_2")
    res_3 = take("res_3")
    res_4 = take("res_4")
    atom_1 = take("atom_1")
    atom_2 = take("atom_2")
    atom_3 = take("atom_3")
    atom_4 = take("atom_4")
    occupancy = take("occupancy")
    throughput = take("throughput")
    length_raw = take("length_raw") if segment_arrays["length_raw"] is not None else None
    qss_raw = take("qss_raw") if segment_arrays["qss_raw"] is not None else None

    residue_concat = np.concatenate((res_1, res_2, res_3, res_4))
    residues = np.unique(residue_concat)
    residues_int_set = set()
    for res in residues:
        try:
            residues_int_set.add(int(res))
        except (ValueError, TypeError):
            pass

    cluster_values = segment_arrays["tunnel_cluster"][segment_slice]
    if valid_count != len(cluster_values):
        cluster_values = cluster_values[valid_mask]
    tunnel_cluster = int(cluster_values[0]) if len(cluster_values) > 0 else 0

    return {
        'x': x, 'y': y, 'z': z, 'b': b,
        'x_origin': x_origin, 'y_origin': y_origin, 'z_origin': z_origin,
        'frame_ids': frame_ids,
        'res_1': res_1, 'res_2': res_2, 'res_3': res_3, 'res_4': res_4,
        'atom_1': atom_1, 'atom_2': atom_2, 'atom_3': atom_3, 'atom_4': atom_4,
        'occupancy': occupancy,
        'throughput': throughput,
        'length_raw': length_raw,
        'qss_raw': qss_raw,
        'residues': residues.tolist(),
        'residues_int_set': residues_int_set,
        'tunnel_cluster': tunnel_cluster,
    }, valid_count


# =====================================================
# 主处理函数
# =====================================================
def preprocess_paths():
    """Preprocess all mapped tunnel paths."""
    start_time = time.time()
    stage_timings = {}
    output_dir = os.path.dirname(OUTPUT_PKL)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def mark_stage(stage_name, stage_start):
        stage_timings[stage_name] = time.perf_counter() - stage_start

    print("=" * 60)
    print("Mapped tunnel path preprocessing")
    print("=" * 60)
    print(f"Input file: {INPUT_CSV}")
    print(f"Output file: {OUTPUT_PKL}")
    print(f"Laplacian smoothing: iterations={SMOOTH_ITERATIONS}, alpha={SMOOTH_ALPHA}")
    grid_status = f"Grid discretization (grid_size={GRID_SIZE} A)" if GRID_ENABLED else "Grid discretization disabled"
    print(f"Grid setting: {grid_status}")
    print("=" * 60)

    print(f"\n[1/6] Reading {INPUT_CSV} ...")
    stage_start = time.perf_counter()

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file does not exist: {INPUT_CSV}")
    if os.path.getsize(INPUT_CSV) == 0:
        raise ValueError(f"Input file is empty: {INPUT_CSV}")

    # stage_04_build_path_points.py writes a standard comma-separated file.
    # 这里不再使用自动分隔符探测，避免 csv.Sniffer 偶发报 “Could not determine delimiter”。
    df_temp = pd.read_csv(INPUT_CSV, nrows=0)
    existing_cols = set(df_temp.columns)
    dtype_to_use = {k: v for k, v in DTYPE_MAPPING.items() if k in existing_cols}

    df = pd.read_csv(INPUT_CSV, dtype=dtype_to_use)
    mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"    Loaded {len(df):,} rows | memory usage {mem_usage:.1f} MB")
    mark_stage("1. Read CSV", stage_start)

    print("\n[2/6] Splitting tunnel paths ...")
    stage_start = time.perf_counter()
    start_idxs = df.index[df["res_seq"] == 1].to_list()
    if not start_idxs or start_idxs[0] != 0:
        start_idxs = [0] + start_idxs
    start_idxs.append(len(df))

    n_paths = len(start_idxs) - 1
    print(f"    Total tunnel paths: {n_paths:,}")
    mark_stage("2. Split tunnel paths", stage_start)

    rdp_status = f"RDP simplification (epsilon={RDP_EPSILON})" if RDP_ENABLED else "RDP simplification disabled"
    print(f"\n[3/6] Processing tunnel paths (grid discretization -> Laplacian smoothing -> {rdp_status}) ...")
    print(f"    Worker processes: {NUM_WORKERS}")

    print("    Extracting tunnel-path segments ...")
    extract_stage_start = time.perf_counter()
    columns = set(df.columns)
    segment_arrays = prepare_segment_arrays(df, columns)
    all_seg_data = []
    cleaned_path_point_counts = []

    for i in tqdm(range(n_paths), desc="    Segment extraction"):
        seg_data, point_count = extract_segment_data_from_arrays(segment_arrays, start_idxs[i], start_idxs[i + 1])
        if seg_data is None:
            continue
        cleaned_path_point_counts.append(point_count)
        all_seg_data.append(seg_data)

    print(f"    Valid tunnel-path segments: {len(all_seg_data):,}")
    mark_stage("3a. Extract segments", extract_stage_start)

    # 分批处理
    batch_stage_start = time.perf_counter()
    batches = []
    for i in range(0, len(all_seg_data), CHUNK_SIZE):
        batches.append(all_seg_data[i:i + CHUNK_SIZE])

    print(f"    Processing batches: {len(batches)}")
    mark_stage("3b. Create batches", batch_stage_start)

    # 多进程处理
    process_stage_start = time.perf_counter()
    PATHS = []
    total_original_points = 0
    total_simplified_points = 0
    original_point_counts = []
    simplified_point_counts = []
    path_stage_totals = {
        "3c.path.unpack": 0.0,
        "3c.path.selection_copy": 0.0,
        "3c.path.smooth": 0.0,
        "3c.path.rdp": 0.0,
        "3c.path.finalize": 0.0,
        "3c.path.total": 0.0,
    }

    if NUM_WORKERS <= 1:
        dispatch_stage_start = time.perf_counter()
        with tqdm(total=len(batches), desc="    Serial processing") as pbar:
            for batch in batches:
                results = process_path_batch(batch)
                for result in results:
                    if result is not None:
                        PATHS.append(result["path"])
                        total_original_points += result["original_points"]
                        total_simplified_points += result["simplified_points"]
                        original_point_counts.append(result["original_points"])
                        simplified_point_counts.append(result["simplified_points"])
                        timing = result.get("timing")
                        if timing:
                            for key, value in timing.items():
                                path_stage_totals[key] = path_stage_totals.get(key, 0.0) + float(value)
                pbar.update(1)
        stage_timings["3c.proc.dispatch_wait"] = time.perf_counter() - dispatch_stage_start
    else:
        # Windows 需要使用 'spawn'，但我们用 map 代替 submit 来减少开销
        executor_start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            stage_timings["3c.proc.pool_startup"] = time.perf_counter() - executor_start
            dispatch_stage_start = time.perf_counter()
            with tqdm(total=len(batches), desc="    Parallel processing") as pbar:
                for results in executor.map(process_path_batch, batches):
                    for result in results:
                        if result is not None:
                            PATHS.append(result["path"])
                            total_original_points += result["original_points"]
                            total_simplified_points += result["simplified_points"]
                            original_point_counts.append(result["original_points"])
                            simplified_point_counts.append(result["simplified_points"])
                            timing = result.get("timing")
                            if timing:
                                for key, value in timing.items():
                                    path_stage_totals[key] = path_stage_totals.get(key, 0.0) + float(value)
                    pbar.update(1)
            stage_timings["3c.proc.dispatch_wait"] = time.perf_counter() - dispatch_stage_start

    print(f"    Valid tunnel paths: {len(PATHS):,}")
    mark_stage("3c. Preprocess tunnel paths", process_stage_start)
    for key, value in path_stage_totals.items():
        stage_timings[key] = value
    path_total = path_stage_totals.get("3c.path.total", 0.0)
    if path_total > 0:
        stage_timings["3c.proc.overhead"] = max(
            0.0,
            stage_timings["3c. Preprocess tunnel paths"] - path_total
        )

    if RDP_ENABLED and total_original_points > 0:
        reduction_pct = (1 - total_simplified_points / total_original_points) * 100
        print(f"    RDP simplification: {total_original_points:,} -> {total_simplified_points:,} points ({reduction_pct:.1f}% reduction)")

    print("\n[4/6] Computing residue statistics ...")
    residue_stage_start = time.perf_counter()

    # 残基计数
    residue_count_stage_start = time.perf_counter()
    INITIAL_COUNTS = {}
    for path in PATHS:
        for res_int in path["residues_set"]:
            INITIAL_COUNTS[res_int] = INITIAL_COUNTS.get(res_int, 0) + 1
    mark_stage("4a. Count residues", residue_count_stage_start)

    print("    Computing residue positions ...")
    residue_position_stage_start = time.perf_counter()
    RESIDUE_POSITIONS = {}
    residue_cols = ["residue_1", "residue_2", "residue_3", "residue_4"]

    x_arr = df["x"].values
    y_arr = df["y"].values
    z_arr = df["z"].values

    for col in residue_cols:
        if col not in df.columns:
            continue
        res_arr = df[col].values
        unique_res = np.unique(res_arr[~np.isnan(res_arr.astype(float))])

        for res_id in unique_res:
            try:
                res_id_int = int(res_id)
                mask = (res_arr == res_id)

                if res_id_int not in RESIDUE_POSITIONS:
                    RESIDUE_POSITIONS[res_id_int] = {
                        "x_sum": 0.0, "y_sum": 0.0, "z_sum": 0.0, "count": 0
                    }

                pos = RESIDUE_POSITIONS[res_id_int]
                pos["x_sum"] += float(np.sum(x_arr[mask]))
                pos["y_sum"] += float(np.sum(y_arr[mask]))
                pos["z_sum"] += float(np.sum(z_arr[mask]))
                pos["count"] += int(np.sum(mask))
            except (ValueError, TypeError):
                pass
    mark_stage("4b. Accumulate residue positions", residue_position_stage_start)

    # 计算平均位置
    residue_average_stage_start = time.perf_counter()
    for res_id, pos in RESIDUE_POSITIONS.items():
        if pos["count"] > 0:
            RESIDUE_POSITIONS[res_id] = {
                "x": pos["x_sum"] / pos["count"],
                "y": pos["y_sum"] / pos["count"],
                "z": pos["z_sum"] / pos["count"],
            }
        else:
            RESIDUE_POSITIONS[res_id] = {"x": 0.0, "y": 0.0, "z": 0.0}
    mark_stage("4c. Average residue positions", residue_average_stage_start)

    # 计算帧范围
    frame_stage_start = time.perf_counter()
    frame_ids_all = []
    for path in PATHS:
        frame_ids = path.get("frame_ids", [])
        if frame_ids and len(frame_ids) > 0:
            frame_id = frame_ids[0]
            if not np.isnan(frame_id):
                frame_ids_all.append(int(frame_id))

    if frame_ids_all:
        FRAME_RANGE = {"min": min(frame_ids_all), "max": max(frame_ids_all)}
    else:
        FRAME_RANGE = {"min": 0, "max": 0}
    mark_stage("4d. Compute frame range", frame_stage_start)
    mark_stage("4. Residue statistics", residue_stage_start)

    print(f"    Residue count: {len(RESIDUE_POSITIONS):,}")
    print(f"    Frame range: {FRAME_RANGE['min']} - {FRAME_RANGE['max']}")

    cluster_stage_start = time.perf_counter()
    CLUSTER_META = None
    if "tunnel_cluster" in columns and len(PATHS) > 0:
        print("\n[5/6] Using Tunnel_cluster values from the input as cluster_id ...")
        cluster_groups = {}
        for path in PATHS:
            cluster_id = int(path.get("cluster_id", 0))
            cluster_groups.setdefault(cluster_id, []).append(path)

        CLUSTER_META = {
            "n_clusters": len(cluster_groups),
            "source": "Tunnel_cluster",
            "endpoint_dist_threshold": None,
            "length_diff_threshold": None,
            "feature_names": ["Tunnel_cluster"],
            "cluster_centers": [],
        }

        for cluster_id in sorted(cluster_groups):
            group = cluster_groups[cluster_id]
            endpoints = [
                [path["xs"][-1], path["ys"][-1], path["zs"][-1]]
                for path in group if path["xs"] and path["ys"] and path["zs"]
            ]
            lengths = [
                path["length"][-1] if path["length"] else 0.0
                for path in group
            ]
            center = {
                "cluster_id": int(cluster_id),
                "count": len(group),
            }
            if endpoints:
                center["exit_x"] = float(np.mean([ep[0] for ep in endpoints]))
                center["exit_y"] = float(np.mean([ep[1] for ep in endpoints]))
                center["exit_z"] = float(np.mean([ep[2] for ep in endpoints]))
            if lengths:
                center["path_length"] = float(np.mean(lengths))
            CLUSTER_META["cluster_centers"].append(center)
    elif CLUSTER_ENABLED and len(PATHS) > 0:
        print(f"\n[5/6] Clustering tunnel paths (outlet distance < {CLUSTER_ENDPOINT_DIST} A, path-length difference < {CLUSTER_LENGTH_DIFF} A) ...")

        # 提取每条路径的终点坐标和总长度
        endpoints = np.array([
            [path["xs"][-1], path["ys"][-1], path["zs"][-1]]
            for path in PATHS
        ], dtype=np.float64)
        lengths = np.array([
            path["length"][-1] if path["length"] else 0.0
            for path in PATHS
        ], dtype=np.float64)

        # 差并集聚类
        # 每个 group 记录：已加入路径的终点列表、长度列表、路径下标列表
        groups = []

        for i in tqdm(range(len(PATHS)), desc="    Path clustering"):
            ep = endpoints[i]
            ln = lengths[i]
            assigned = False

            for group in groups:
                # 与集合中每条路径的终点逐一比较
                for j in range(len(group['indices'])):
                    dist = np.linalg.norm(ep - group['endpoints'][j])
                    ldiff = abs(ln - group['lengths'][j])
                    if dist < CLUSTER_ENDPOINT_DIST and ldiff < CLUSTER_LENGTH_DIFF:
                        group['endpoints'].append(ep)
                        group['lengths'].append(ln)
                        group['indices'].append(i)
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                groups.append({
                    'endpoints': [ep],
                    'lengths': [ln],
                    'indices': [i]
                })

        n_clusters = len(groups)

        # 将 cluster_id 写入每条路径
        for cluster_id, group in enumerate(groups):
            for path_idx in group['indices']:
                PATHS[path_idx]["cluster_id"] = cluster_id

        # 计算每个簇的统计信息（取均值作为代表）
        CLUSTER_META = {
            "n_clusters": n_clusters,
            "endpoint_dist_threshold": CLUSTER_ENDPOINT_DIST,
            "length_diff_threshold": CLUSTER_LENGTH_DIFF,
            "feature_names": ["exit_x", "exit_y", "exit_z", "path_length"],
            "cluster_centers": [
                {
                    "cluster_id": cluster_id,
                    "exit_x": float(np.mean([ep[0] for ep in group['endpoints']])),
                    "exit_y": float(np.mean([ep[1] for ep in group['endpoints']])),
                    "exit_z": float(np.mean([ep[2] for ep in group['endpoints']])),
                    "path_length": float(np.mean(group['lengths'])),
                    "count": len(group['indices']),
                }
                for cluster_id, group in enumerate(groups)
            ],
        }

        print(f"    Total clusters: {n_clusters}")
        for c in CLUSTER_META["cluster_centers"]:
            print(
                f"    Cluster {c['cluster_id']}: {c['count']} paths | "
                f"outlet=({c['exit_x']:.2f}, {c['exit_y']:.2f}, {c['exit_z']:.2f}) | "
                f"path_length={c['path_length']:.1f} A"
            )
    else:
        for path in PATHS:
            path["cluster_id"] = 0
        if CLUSTER_ENABLED:
            print("\n[5/6] No tunnel paths were available; clustering skipped")
    mark_stage("5. Path clustering", cluster_stage_start)

    print(f"\n[6/6] Saving preprocessed data to {OUTPUT_PKL} ...")
    save_stage_start = time.perf_counter()

    preprocessed_data = {
        "PATHS": PATHS,
        "INITIAL_COUNTS": INITIAL_COUNTS,
        "RESIDUE_POSITIONS": RESIDUE_POSITIONS,
        "FRAME_RANGE": FRAME_RANGE,
        "CLUSTER_META": CLUSTER_META,
        "metadata": {
            "source_file": INPUT_CSV,
            "smooth_iterations": SMOOTH_ITERATIONS,
            "smooth_alpha": SMOOTH_ALPHA,
            "smooth_min_points": SMOOTH_MIN_POINTS,
            "rdp_enabled": RDP_ENABLED,
            "rdp_epsilon": RDP_EPSILON if RDP_ENABLED else None,
            "rdp_min_points": RDP_MIN_POINTS if RDP_ENABLED else None,
            "grid_enabled": GRID_ENABLED,
            "grid_size": GRID_SIZE if GRID_ENABLED else None,
            "cluster_enabled": CLUSTER_ENABLED,
            "cluster_k": CLUSTER_META["n_clusters"] if CLUSTER_META else None,
            "cluster_endpoint_dist": CLUSTER_META["endpoint_dist_threshold"] if CLUSTER_META else None,
            "cluster_length_diff": CLUSTER_META["length_diff_threshold"] if CLUSTER_META else None,
            "original_points": total_original_points,
            "simplified_points": total_simplified_points,
            "path_count": len(PATHS),
            "residue_count": len(RESIDUE_POSITIONS),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    output_pkl_path = Path(OUTPUT_PKL).expanduser()
    with output_pkl_path.open("wb") as f:
        pickle.dump(preprocessed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    mark_stage("6a. Save PKL", save_stage_start)

    file_size = output_pkl_path.stat().st_size / 1024 / 1024
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("Preprocessing completed")
    print("=" * 60)
    print(f"Output file: {output_pkl_path}")
    print(f"File size: {file_size:.1f} MB")
    print(f"Tunnel path count: {len(PATHS):,}")
    if cleaned_path_point_counts:
        cleaned_total_points = sum(cleaned_path_point_counts)
        cleaned_min_points = min(cleaned_path_point_counts)
        cleaned_max_points = max(cleaned_path_point_counts)
        cleaned_avg_points = cleaned_total_points / len(cleaned_path_point_counts)
        print(
            f"Cleaned path-point statistics: total={cleaned_total_points:,}, "
            f"per-path min/avg/max={cleaned_min_points}/{cleaned_avg_points:.2f}/{cleaned_max_points}"
        )
    if original_point_counts:
        original_min_points = min(original_point_counts)
        original_max_points = max(original_point_counts)
        original_avg_points = total_original_points / len(original_point_counts)
        print(
            f"Preprocessing input path-point statistics: total={total_original_points:,}, "
            f"per-path min/avg/max={original_min_points}/{original_avg_points:.2f}/{original_max_points}"
        )
    if simplified_point_counts:
        simplified_min_points = min(simplified_point_counts)
        simplified_max_points = max(simplified_point_counts)
        simplified_avg_points = total_simplified_points / len(simplified_point_counts)
        print(
            f"Preprocessed path-point statistics: total={total_simplified_points:,}, "
            f"per-path min/avg/max={simplified_min_points}/{simplified_avg_points:.2f}/{simplified_max_points}"
        )
    print(f"Residue count: {len(RESIDUE_POSITIONS):,}")
    print(f"Elapsed time: {elapsed_time:.1f} s")
    print("Stage timing summary:")
    for stage_name, seconds in sorted(stage_timings.items(), key=lambda item: item[1], reverse=True):
        print(f"  {stage_name:<32} {seconds:8.2f} s")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess PathVisual all_points data")
    parser.add_argument("--input-csv", default=INPUT_CSV, help="input all_points CSV path")
    parser.add_argument("--output-pkl", default=OUTPUT_PKL, help="output preprocessed_paths.pkl path")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime_paths(input_csv=args.input_csv, output_pkl=args.output_pkl)
    preprocess_paths()


if __name__ == "__main__":
    main()
