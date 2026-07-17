"""
pyqtgraph-based profile charts.
Replaces Plotly OverlayChart.vue and StatisticsChart.vue.
"""
import colorsys
import html
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

try:
    import pyqtgraph as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QScrollArea,
    QGraphicsRectItem,
    QFrame,
    QLineEdit,
)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPainter, QLinearGradient, QColor, QFont, QPen, QBrush

from TopoTunnelMiner_UI.core.residue_combination import bottleneck_point_indices


CHART_PRESET_OPTIONS = [
    ("By Radius", "radius"),
    ("By Hydrophobicity", "hydrophobicity"),
    ("By Frame", "frame"),
    ("By Dataset", "dataset"),
]

CHART_AXIS_OPTIONS = [
    ("Radius", "radius"),
    ("Hydrophobicity", "hydrophobicity"),
    ("Frame", "frame"),
    ("Path Length", "path_length"),
]

CHART_OPACITY_OPTIONS = [
    ("55%", 0.72),
    ("70%", 0.86),
    ("85%", 0.96),
    ("100%", 1.00),
    ("120%", 1.14),
    ("140%", 1.28),
]

CHART_DEPTH_OPTIONS = [
    ("Soft", 0.82),
    ("Balanced", 1.00),
    ("Rich", 1.14),
    ("Vivid", 1.28),
]

CHART_SELECTION_OPTIONS = [
    ("Off", "off"),
    ("Path Selection", "path"),
    ("Residue Stats", "residue"),
]

AXIS_META: Dict[str, Dict[str, str]] = {
    "radius": {
        "title": "Radius",
        "axis_label": "Radius (A)",
        "hint_label": "radius",
    },
    "hydrophobicity": {
        "title": "Hydrophobicity",
        "axis_label": "Hydrophobicity",
        "hint_label": "hydrophobicity",
    },
    "frame": {
        "title": "Frame",
        "axis_label": "Frame",
        "hint_label": "frame",
    },
    "path_length": {
        "title": "Path Length",
        "axis_label": "Path Length (A)",
        "hint_label": "path length",
    },
}

_HYDRO_STOPS = [
    (0.00, (33, 102, 172)),
    (0.25, (103, 169, 207)),
    (0.50, (247, 247, 247)),
    (0.75, (239, 138, 98)),
    (1.00, (178, 24, 43)),
]

_COLOR_STOPS = {
    "radius": _HYDRO_STOPS,
    "hydrophobicity": _HYDRO_STOPS,
    "frame": _HYDRO_STOPS,
}

_TARGET_LINE_DEPTH_BOOST = 1.24
_SELECTED_PATH_RGB = (255, 188, 92)
_SELECTED_MARKER_OUTLINE = (255, 247, 235)
_SELECTION_BOX_EDGE = (96, 155, 255, 218)
_SELECTION_BOX_FILL = (96, 155, 255, 34)
_RESIDUE_SUMMARY_RANK_LIMIT = 8


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _color_from_t(stops: List[Tuple[float, Tuple[int, int, int]]], t: float) -> Tuple[int, int, int]:
    """Interpolate a color stop table."""
    t = max(0.0, min(1.0, t))
    lo_pos, lo_rgb = stops[0]
    hi_pos, hi_rgb = stops[-1]
    for i in range(len(stops) - 1):
        if stops[i][0] <= t <= stops[i + 1][0]:
            lo_pos, lo_rgb = stops[i]
            hi_pos, hi_rgb = stops[i + 1]
            break
    span = hi_pos - lo_pos
    factor = (t - lo_pos) / span if span > 0 else 0.0
    return (
        int(lo_rgb[0] + (hi_rgb[0] - lo_rgb[0]) * factor),
        int(lo_rgb[1] + (hi_rgb[1] - lo_rgb[1]) * factor),
        int(lo_rgb[2] + (hi_rgb[2] - lo_rgb[2]) * factor),
    )


def _profile_length(profile: dict) -> int:
    return max(
        len(profile.get("resSeq", [])),
        len(profile.get("radius", [])),
        len(profile.get("hydrophobicity", [])),
        int(profile.get("numPoints", 0)),
    )


def _series_for_profile(profile: dict, axis_key: str) -> np.ndarray:
    """Convert a chart axis key into a numeric per-point series."""
    if axis_key == "path_length":
        values = profile.get("resSeq", [])
        return np.asarray(values, dtype=np.float32)

    if axis_key == "radius":
        values = profile.get("radius", [])
        return np.asarray(values, dtype=np.float32)

    if axis_key == "hydrophobicity":
        values = profile.get("hydrophobicity", [])
        return np.asarray(values, dtype=np.float32)

    if axis_key == "frame":
        n = _profile_length(profile)
        if n <= 0:
            return np.asarray([], dtype=np.float32)
        return np.full(n, float(profile.get("frameId", 0)), dtype=np.float32)

    return np.asarray([], dtype=np.float32)


def _dataset_prefix(profile: dict) -> str:
    return str(profile.get("datasetPrefix") or "Dataset")


def _paired_series(profile: dict, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite paired x/y series for one profile."""
    xs = _series_for_profile(profile, x_key)
    ys = _series_for_profile(profile, y_key)
    count = min(len(xs), len(ys))
    if count == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    xs = np.asarray(xs[:count], dtype=np.float32)
    ys = np.asarray(ys[:count], dtype=np.float32)
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return xs[finite], ys[finite]


def _bottleneck_series_for_profile(profile: dict, x_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite x/y points for the bottleneck location of one profile."""
    xs = _series_for_profile(profile, x_key)
    ys = _series_for_profile(profile, y_key)
    count = min(len(xs), len(ys))
    if count == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    indices = bottleneck_point_indices(profile.get("radius", []), point_count=count)
    if not indices:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    safe_indices = np.asarray(
        sorted(idx for idx in indices if 0 <= idx < count),
        dtype=np.int64,
    )
    if len(safe_indices) == 0:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

    xs = np.asarray(xs[:count], dtype=np.float32)[safe_indices]
    ys = np.asarray(ys[:count], dtype=np.float32)[safe_indices]
    finite = np.isfinite(xs) & np.isfinite(ys)
    if not np.any(finite):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return xs[finite], ys[finite]


def _profile_scalar(profile: dict, axis_key: str) -> float:
    """Scalar summary used for per-path color assignment."""
    values = _series_for_profile(profile, axis_key)
    if len(values) == 0:
        return 0.0
    return float(np.nanmean(values))


def _build_color_scale(profiles: List[dict], color_key: str) -> Dict[str, object]:
    scalars = [_profile_scalar(profile, color_key) for profile in profiles]
    if not scalars:
        scalars = [0.0]

    if color_key == "hydrophobicity":
        extent = max(abs(min(scalars)), abs(max(scalars)), 0.1)
        return {
            "stops": _COLOR_STOPS[color_key],
            "v_min": -extent,
            "v_max": extent,
        }

    v_min = float(min(scalars))
    v_max = float(max(scalars))
    if abs(v_max - v_min) < 1e-6:
        v_min -= 0.5
        v_max += 0.5
    return {
        "stops": _COLOR_STOPS[color_key],
        "v_min": v_min,
        "v_max": v_max,
    }


def _quantized_color(value: float, scale: Dict[str, object], bins: int = 18) -> Tuple[int, int, int]:
    v_min = float(scale["v_min"])
    v_max = float(scale["v_max"])
    if v_max <= v_min:
        return _color_from_t(scale["stops"], 0.5)
    t = (value - v_min) / (v_max - v_min)
    t = max(0.0, min(1.0, t))
    if bins > 1:
        bucket = min(bins - 1, max(0, int(round(t * (bins - 1)))))
        t = bucket / float(bins - 1)
    return _color_from_t(scale["stops"], t)


def _scale_alpha(base_alpha: int, alpha_scale: float) -> int:
    return int(round(_clamp(base_alpha * alpha_scale, 12.0, 220.0)))


def _axis_range_with_padding(min_value: float, max_value: float) -> Tuple[float, float]:
    min_value = float(min_value)
    max_value = float(max_value)
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return 0.0, 1.0
    span = max_value - min_value
    if abs(span) < 1e-9:
        pad = max(0.5, abs(min_value) * 0.08, 0.1)
        return min_value - pad, max_value + pad
    pad = max(span * 0.06, 0.02)
    return min_value - pad, max_value + pad


def _compute_profile_bounds(
    profiles: List[dict],
    x_key: str,
    y_key: str,
) -> Optional[Tuple[float, float, float, float]]:
    x_mins: List[float] = []
    x_maxs: List[float] = []
    y_mins: List[float] = []
    y_maxs: List[float] = []

    for profile in profiles:
        xs, ys = _paired_series(profile, x_key, y_key)
        if len(xs) == 0:
            continue
        x_mins.append(float(np.nanmin(xs)))
        x_maxs.append(float(np.nanmax(xs)))
        y_mins.append(float(np.nanmin(ys)))
        y_maxs.append(float(np.nanmax(ys)))

    if not x_mins or not y_mins:
        return None

    raw_x_min = min(x_mins)
    raw_x_max = max(x_maxs)
    if raw_x_min >= 0.0:
        edge_pad = max(raw_x_max * 0.015, 0.01)
        x_min = -edge_pad
        if abs(raw_x_max) < 1e-9:
            x_max = 1.0
        else:
            x_max = raw_x_max + edge_pad
    else:
        x_min, x_max = _axis_range_with_padding(raw_x_min, raw_x_max)
        x_min = min(x_min, 0.0)
        x_max = max(x_max, 0.0)
    y_min, y_max = _axis_range_with_padding(min(y_mins), max(y_maxs))
    return x_min, x_max, y_min, y_max


def _lock_plot_to_bounds(plot_widget, bounds: Optional[Tuple[float, float, float, float]]):
    if plot_widget is None:
        return
    view_box = plot_widget.getViewBox()
    if view_box is None or bounds is None:
        return
    x_min, x_max, y_min, y_max = bounds
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    try:
        view_box.enableAutoRange(x=False, y=False)
    except Exception:
        pass
    try:
        view_box.setMouseEnabled(x=False, y=False)
    except Exception:
        pass
    try:
        view_box.setLimits(
            xMin=x_min,
            xMax=x_max,
            yMin=y_min,
            yMax=y_max,
            minXRange=x_span,
            maxXRange=x_span,
            minYRange=y_span,
            maxYRange=y_span,
        )
    except Exception:
        pass
    try:
        view_box.setRange(
            xRange=(x_min, x_max),
            yRange=(y_min, y_max),
            padding=0.0,
            disableAutoRange=True,
        )
    except Exception:
        pass


def _dataset_button_stylesheet(color_text: str, checked: bool) -> str:
    color = QColor(color_text)
    if not color.isValid():
        color = QColor("#4169E1")
    red = color.red()
    green = color.green()
    blue = color.blue()
    if checked:
        background = f"rgba({red}, {green}, {blue}, 0.22)"
        border = f"rgba({red}, {green}, {blue}, 0.96)"
        text = "#1f2937"
    else:
        background = f"rgba({red}, {green}, {blue}, 0.08)"
        border = f"rgba({red}, {green}, {blue}, 0.52)"
        text = "#4b5563"
    return (
        "QPushButton {"
        f"background: {background};"
        f"border: 1px solid {border};"
        "border-radius: 8px;"
        f"color: {text};"
        "padding: 4px 10px;"
        "font-size: 11px;"
        "font-weight: 600;"
        "text-align: center;"
        "}"
    )


def _apply_depth_scale(color: Tuple[int, int, int], depth_scale: float) -> Tuple[int, int, int]:
    """Adjust palette richness while preserving hue relationships."""
    depth_scale = _clamp(depth_scale, 0.6, 1.4)
    red, green, blue = [channel / 255.0 for channel in color]
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    if depth_scale >= 1.0:
        saturation = _clamp(saturation * (1.0 + 0.48 * (depth_scale - 1.0)), 0.0, 1.0)
        lightness = _clamp(lightness - 0.12 * (depth_scale - 1.0), 0.0, 1.0)
    else:
        softness = 1.0 - (1.0 - depth_scale) * 0.35
        saturation = _clamp(saturation * softness, 0.0, 1.0)
        lightness = _clamp(lightness + (1.0 - lightness) * 0.18 * (1.0 - depth_scale), 0.0, 1.0)

    out_red, out_green, out_blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (
        int(round(out_red * 255.0)),
        int(round(out_green * 255.0)),
        int(round(out_blue * 255.0)),
    )


def _selected_highlight_color(depth_scale: float) -> Tuple[int, int, int]:
    """Use a warm highlight that stays distinct from the default blue-red palette."""
    return _apply_depth_scale(_SELECTED_PATH_RGB, max(depth_scale, 1.08))


def _profile_overlay_style(path_count: int) -> Tuple[int, float]:
    """Adaptive alpha/width for many overlaid curves."""
    if path_count >= 4000:
        return 34, 0.62
    if path_count >= 2000:
        return 42, 0.72
    if path_count >= 1000:
        return 52, 0.82
    if path_count >= 400:
        return 64, 0.96
    if path_count >= 120:
        return 78, 1.10
    return 104, 1.24


def _statistics_fill_style(path_count: int) -> Tuple[int, int, float, float]:
    if path_count >= 2000:
        return 18, 34, 1.2, 2.0
    if path_count >= 800:
        return 24, 42, 1.2, 2.1
    return 32, 56, 1.3, 2.2


_STAT_OUTER_FILL = (224, 236, 255)
_STAT_INNER_FILL = (176, 202, 245)
_STAT_MEDIAN_LINE = (72, 114, 190)
_STAT_MEAN_LINE = (32, 73, 153)
_STAT_BOTTLENECK_POINT_LIMIT_PER_DATASET = 900


def _aggregate_profile_samples(
    profiles: List[dict],
    x_key: str,
    y_key: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Aggregate profile points into percentile bands over a shared x axis."""
    all_x: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for profile in profiles:
        xs, ys = _paired_series(profile, x_key, y_key)
        if len(xs) == 0:
            continue
        all_x.append(xs.astype(np.float64))
        all_y.append(ys.astype(np.float64))

    if not all_x:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty, empty, empty

    xs = np.concatenate(all_x)
    ys = np.concatenate(all_y)

    x_min = float(np.nanmin(xs))
    x_max = float(np.nanmax(xs))
    if abs(x_max - x_min) < 1e-9:
        buckets = [(x_min, ys)]
    else:
        rounded = np.round(xs, 4)
        unique_count = len(np.unique(rounded))
        use_discrete = unique_count <= 48 or (x_key == "frame" and unique_count <= 160)
        buckets: List[Tuple[float, np.ndarray]] = []
        if use_discrete:
            for value in np.unique(rounded):
                mask = rounded == value
                if np.any(mask):
                    buckets.append((float(np.mean(xs[mask])), ys[mask]))
        else:
            sample_count = len(xs)
            bin_count = min(72, max(24, int(np.sqrt(sample_count / 10.0))))
            edges = np.linspace(x_min, x_max, bin_count + 1)
            bucket_index = np.clip(np.digitize(xs, edges) - 1, 0, bin_count - 1)
            for idx in range(bin_count):
                mask = bucket_index == idx
                if np.any(mask):
                    center = float((edges[idx] + edges[idx + 1]) / 2.0)
                    buckets.append((center, ys[mask]))

    x_vals = np.asarray([bucket[0] for bucket in buckets], dtype=np.float64)
    p10 = np.asarray([np.nanpercentile(bucket[1], 10) for bucket in buckets], dtype=np.float64)
    p25 = np.asarray([np.nanpercentile(bucket[1], 25) for bucket in buckets], dtype=np.float64)
    p50 = np.asarray([np.nanpercentile(bucket[1], 50) for bucket in buckets], dtype=np.float64)
    p75 = np.asarray([np.nanpercentile(bucket[1], 75) for bucket in buckets], dtype=np.float64)
    p90 = np.asarray([np.nanpercentile(bucket[1], 90) for bucket in buckets], dtype=np.float64)
    mean = np.asarray([np.nanmean(bucket[1]) for bucket in buckets], dtype=np.float64)
    return x_vals, p10, p25, p50, p75, p90, mean


def _sample_profiles_for_overlay(profiles: List[dict], limit: int) -> List[dict]:
    """Return a deterministic subset for dense overlay markers."""
    if limit <= 0 or len(profiles) <= limit:
        return list(profiles)
    indices = np.linspace(0, len(profiles) - 1, int(limit), dtype=np.int64)
    return [profiles[int(idx)] for idx in np.unique(indices)]


def _collect_selected_points(
    profiles: List[dict],
    x_key: str,
    y_key: str,
    bounds: Tuple[float, float, float, float],
) -> Tuple[Set[int], Dict[int, List[int]]]:
    """Collect path ids and point indices inside a chart selection rectangle."""
    x_min, x_max, y_min, y_max = bounds
    selected_paths: Set[int] = set()
    selected_points: Dict[int, List[int]] = {}

    for profile in profiles:
        xs = _series_for_profile(profile, x_key)
        ys = _series_for_profile(profile, y_key)
        count = min(len(xs), len(ys))
        if count == 0:
            continue

        xs = np.asarray(xs[:count], dtype=np.float32)
        ys = np.asarray(ys[:count], dtype=np.float32)
        mask = (
            np.isfinite(xs)
            & np.isfinite(ys)
            & (xs >= x_min)
            & (xs <= x_max)
            & (ys >= y_min)
            & (ys <= y_max)
        )
        if not np.any(mask):
            continue

        point_indices = np.flatnonzero(mask).astype(np.int32).tolist()

        if point_indices:
            path_id = int(profile.get("pathIndex", -1))
            if path_id >= 0:
                selected_paths.add(path_id)
                selected_points[path_id] = point_indices

    return selected_paths, selected_points


def _accumulate_residue_bucket(
    bucket: Dict[int, dict],
    residue_id: int,
    path_id: int,
    frame_id: int,
    radius_val: Optional[float],
):
    stats = bucket.setdefault(
        residue_id,
        {
            "count": 0,
            "pathIds": set(),
            "frameIds": set(),
            "radius_sum": 0.0,
            "min_radius": None,
            "max_radius": None,
        },
    )
    stats["count"] += 1
    stats["pathIds"].add(path_id)
    stats["frameIds"].add(frame_id)
    if radius_val is not None and np.isfinite(radius_val):
        stats["radius_sum"] += radius_val
        stats["min_radius"] = (
            radius_val if stats["min_radius"] is None else min(stats["min_radius"], radius_val)
        )
        stats["max_radius"] = (
            radius_val if stats["max_radius"] is None else max(stats["max_radius"], radius_val)
        )


def _finalize_residue_bucket(bucket: Dict[int, dict]) -> List[dict]:
    residues = []
    for residue_id, stats in bucket.items():
        avg_radius = (
            round(stats["radius_sum"] / stats["count"], 4)
            if stats["count"] > 0
            else None
        )
        residues.append(
            {
                "residueId": residue_id,
                "count": stats["count"],
                "pathCount": len(stats["pathIds"]),
                "frameIds": sorted(stats["frameIds"]),
                "avgRadius": avg_radius,
                "minRadius": round(stats["min_radius"], 4)
                if stats["min_radius"] is not None
                else None,
                "maxRadius": round(stats["max_radius"], 4)
                if stats["max_radius"] is not None
                else None,
            }
        )
    residues.sort(key=lambda item: item["count"], reverse=True)
    return residues


def _build_residue_stats(
    profiles: List[dict],
    selected_points: Dict[int, List[int]],
) -> Optional[dict]:
    """Build residue statistics for chart-selected points."""
    if not selected_points:
        return None

    profile_map = {
        int(profile.get("pathIndex", -1)): profile
        for profile in profiles
        if int(profile.get("pathIndex", -1)) >= 0
    }
    return _build_residue_stats_from_map(profile_map, selected_points)


def _all_profile_point_selection(profiles: List[dict]) -> Dict[int, List[int]]:
    """Return a selected-points map that covers every point in each profile."""
    selected_points: Dict[int, List[int]] = {}
    for profile in profiles or []:
        try:
            path_id = int(profile.get("pathIndex", -1))
        except (TypeError, ValueError):
            continue
        if path_id < 0:
            continue

        radii = profile.get("radius", []) or []
        point_count = len(radii)
        if point_count <= 0:
            for key in ("resSeq", "res_1", "res_2", "res_3", "res_4"):
                values = profile.get(key, []) or []
                point_count = max(point_count, len(values))
        if point_count <= 0:
            try:
                point_count = int(profile.get("numPoints", 0) or 0)
            except (TypeError, ValueError):
                point_count = 0
        if point_count > 0:
            selected_points[path_id] = list(range(point_count))
    return selected_points


def _build_residue_stats_from_map(
    profile_map: Dict[int, dict],
    selected_points: Dict[int, List[int]],
) -> Optional[dict]:
    """Build residue statistics with a precomputed path -> profile lookup."""
    if not selected_points:
        return None

    wall_bucket: Dict[int, dict] = {}
    bottleneck_bucket: Dict[int, dict] = {}
    total_points = 0
    valid_paths: Set[int] = set()

    for path_id, point_indices in selected_points.items():
        profile = profile_map.get(path_id)
        if profile is None:
            continue

        valid_paths.add(path_id)
        frame_id = int(profile.get("frameId", 0) or 0)
        radii = profile.get("radius", [])
        bottleneck_indices = bottleneck_point_indices(radii, point_count=len(radii))
        residue_lists = [
            profile.get("res_1", []),
            profile.get("res_2", []),
            profile.get("res_3", []),
            profile.get("res_4", []),
        ]

        for point_idx in point_indices:
            if point_idx < 0:
                continue
            total_points += 1

            point_residues: Set[int] = set()
            for residue_values in residue_lists:
                if point_idx >= len(residue_values):
                    continue
                try:
                    residue_id = int(residue_values[point_idx])
                except (ValueError, TypeError):
                    continue
                if residue_id > 0:
                    point_residues.add(residue_id)

            radius_val: Optional[float] = None
            if point_idx < len(radii):
                try:
                    radius_val = float(radii[point_idx])
                except (ValueError, TypeError):
                    radius_val = None

            for residue_id in point_residues:
                _accumulate_residue_bucket(wall_bucket, residue_id, path_id, frame_id, radius_val)

            is_bottleneck = point_idx in bottleneck_indices
            if is_bottleneck:
                for residue_id in point_residues:
                    _accumulate_residue_bucket(
                        bottleneck_bucket,
                        residue_id,
                        path_id,
                        frame_id,
                        radius_val,
                    )

    return {
        "wallResidues": _finalize_residue_bucket(wall_bucket),
        "bottleneckResidues": _finalize_residue_bucket(bottleneck_bucket),
        "totalPoints": total_points,
        "totalPaths": len(valid_paths),
    }


def _persistent_selection_bounds_for_mode(
    mode: str,
    bounds: Tuple[float, float, float, float],
    selected_paths: Set[int],
    selected_points: Dict[int, List[int]],
    residue_stats: Optional[dict],
) -> Optional[Tuple[float, float, float, float]]:
    """Return persistent selection bounds only for successful finished selections."""
    if mode == "path":
        return bounds if selected_paths else None
    if mode == "residue":
        return bounds if selected_points and residue_stats else None
    return None


def _summarize_residue_stats(stats: Optional[dict]) -> Optional[dict]:
    if not stats:
        return None
    wall = stats.get("wallResidues", [])
    bottleneck = stats.get("bottleneckResidues", [])
    total_points = int(stats.get("totalPoints", 0) or 0)
    total_paths = int(stats.get("totalPaths", 0) or 0)

    weighted_sum = 0.0
    weight = 0
    min_radius = None
    max_radius = None
    for residue in wall:
        count = int(residue.get("count", 0) or 0)
        avg_radius = residue.get("avgRadius")
        if count > 0 and avg_radius is not None and np.isfinite(avg_radius):
            weighted_sum += float(avg_radius) * count
            weight += count
        residue_min = residue.get("minRadius")
        residue_max = residue.get("maxRadius")
        if residue_min is not None:
            min_radius = residue_min if min_radius is None else min(min_radius, residue_min)
        if residue_max is not None:
            max_radius = residue_max if max_radius is None else max(max_radius, residue_max)

    return {
        "totalPoints": total_points,
        "totalPaths": total_paths,
        "wallCount": len(wall),
        "bottleneckCount": len(bottleneck),
        "avgRadius": round(weighted_sum / weight, 3) if weight > 0 else None,
        "minRadius": round(min_radius, 3) if min_radius is not None else None,
        "maxRadius": round(max_radius, 3) if max_radius is not None else None,
        "topWall": wall[:_RESIDUE_SUMMARY_RANK_LIMIT],
        "topBottleneck": bottleneck[:_RESIDUE_SUMMARY_RANK_LIMIT],
    }


def _residue_stats_by_dataset(
    profile_map: Dict[int, dict],
    selected_points: Dict[int, List[int]],
) -> List[Tuple[str, dict]]:
    grouped_selected: Dict[str, Dict[int, List[int]]] = {}
    for path_id, point_indices in selected_points.items():
        profile = profile_map.get(int(path_id))
        if profile is None:
            continue
        grouped_selected.setdefault(_dataset_prefix(profile), {})[int(path_id)] = list(point_indices)
    summaries: List[Tuple[str, dict]] = []
    for prefix, dataset_points in grouped_selected.items():
        stats = _build_residue_stats_from_map(profile_map, dataset_points)
        summary = _summarize_residue_stats(stats)
        if summary is not None:
            summaries.append((prefix, summary))
    return summaries


def _residue_summary_rows(
    stats: Optional[dict],
    *,
    dataset_prefix: Optional[str] = None,
    summary_kind: str = "path",
) -> List[dict]:
    if not stats:
        return []
    if summary_kind == "bottleneck":
        type_key = "bottleneckResidues"
    else:
        summary_kind = "path"
        type_key = "wallResidues"
    rows: List[dict] = []
    for item in stats.get(type_key, []) or []:
        row = {
            "summary_kind": summary_kind,
            "residueId": int(item.get("residueId", 0) or 0),
            "pathCount": int(item.get("pathCount", 0) or 0),
            "avgRadius": item.get("avgRadius"),
            "frameIds": list(item.get("frameIds", []) or []),
            "totalPaths": int(stats.get("totalPaths", 0) or 0),
        }
        if dataset_prefix is not None:
            row["dataset_prefix"] = str(dataset_prefix)
        rows.append(row)
    return rows


def _residue_summary_rows_by_dataset(
    profile_map: Dict[int, dict],
    selected_points: Dict[int, List[int]],
) -> List[dict]:
    grouped_selected: Dict[str, Dict[int, List[int]]] = {}
    for path_id, point_indices in selected_points.items():
        profile = profile_map.get(int(path_id))
        if profile is None:
            continue
        grouped_selected.setdefault(_dataset_prefix(profile), {})[int(path_id)] = list(point_indices)

    rows: List[dict] = []
    for prefix, dataset_points in grouped_selected.items():
        stats = _build_residue_stats_from_map(profile_map, dataset_points)
        rows.extend(
            _residue_summary_rows(
                stats,
                dataset_prefix=prefix,
                summary_kind="path",
            )
        )
        rows.extend(
            _residue_summary_rows(
                stats,
                dataset_prefix=prefix,
                summary_kind="bottleneck",
            )
        )
    return rows


def _residue_overview_html(summary: dict, *, compact: bool = False) -> str:
    if compact:
        return (
            "<div style='color:#606266;'>"
            f"Paths <b>{summary['totalPaths']}</b>  "
            f"Points <b>{summary['totalPoints']}</b>  "
            f"Wall <b>{summary['wallCount']}</b>  "
            f"Bottleneck <b>{summary['bottleneckCount']}</b>"
            "</div>"
        )
    return (
        "<table width='100%' cellspacing='0' cellpadding='4' style='margin:4px 0 8px 0;'>"
        "<tr>"
        f"<td style='background:#f4f8ff; border-radius:4px;'>Paths<br><b>{summary['totalPaths']}</b></td>"
        f"<td style='background:#f4f8ff; border-radius:4px;'>Points<br><b>{summary['totalPoints']}</b></td>"
        f"<td style='background:#f7fdf7; border-radius:4px;'>Wall Residues<br><b>{summary['wallCount']}</b></td>"
        f"<td style='background:#fff7ed; border-radius:4px;'>Bottleneck Residues<br><b>{summary['bottleneckCount']}</b></td>"
        "</tr>"
        "</table>"
        "<div style='color:#606266; margin-bottom:8px;'>"
        f"Wall radius: mean <b>{_format_radius(summary['avgRadius'])}</b>, "
        f"min <b>{_format_radius(summary['minRadius'])}</b>, "
        f"max <b>{_format_radius(summary['maxRadius'])}</b>"
        "</div>"
    )


def _residue_rank_html(title: str, items: List[dict]) -> str:
    rows = []
    for idx, item in enumerate(items[:_RESIDUE_SUMMARY_RANK_LIMIT], start=1):
        rows.append(
            "<tr>"
            f"<td style='color:#909399; width:18px;'>{idx}</td>"
            f"<td><b>#{item['residueId']}</b></td>"
            f"<td align='right'>{int(item.get('pathCount', 0))} paths</td>"
            f"<td align='right'>{_format_radius(item.get('avgRadius'))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='4' style='color:#909399;'>-</td></tr>")
    return (
        "<div style='font-weight:600; color:#303133; margin-bottom:4px;'>"
        f"{html.escape(title)}"
        "</div>"
        "<table width='100%' cellspacing='0' cellpadding='2' style='color:#303133;'>"
        "<tr style='color:#909399;'>"
        "<td></td><td>Residue</td><td align='right'>Paths</td><td align='right'>Mean Radius</td>"
        "</tr>"
        + "".join(rows)
        + "</table>"
    )


def _format_radius(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.2f} Å"


def _frame_preview(frame_ids: List[int]) -> str:
    if not frame_ids:
        return "-"
    preview = ", ".join(str(fid) for fid in frame_ids[:3])
    if len(frame_ids) > 3:
        preview += "..."
    return preview


class _ScalarColorBar(QWidget):
    """Vertical colorbar for scalar color modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(58)
        self._stops = _HYDRO_STOPS
        self._v_min = -1.0
        self._v_max = 1.0
        self._caption = ""

    def set_scale(
        self,
        stops: List[Tuple[float, Tuple[int, int, int]]],
        v_min: float,
        v_max: float,
        caption: str,
    ):
        self._stops = stops
        self._v_min = v_min
        self._v_max = v_max
        self._caption = caption
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont()
        font.setPixelSize(9)
        painter.setFont(font)
        painter.setPen(QColor(80, 80, 80))

        bar_x, bar_w = 4, 14
        title_h = 12 if self._caption else 0
        bar_top = 8 + title_h
        bar_bottom = self.height() - 10
        bar_h = bar_bottom - bar_top
        if bar_h < 20:
            painter.end()
            return

        if self._caption:
            painter.drawText(2, 10, self._caption)

        gradient = QLinearGradient(bar_x, bar_top, bar_x, bar_bottom)
        for pos, rgb in self._stops:
            gradient.setColorAt(1.0 - pos, QColor(*rgb))
        painter.fillRect(bar_x, bar_top, bar_w, bar_h, gradient)
        painter.setPen(QColor(180, 180, 180))
        painter.drawRect(bar_x, bar_top, bar_w, bar_h)

        painter.setPen(QColor(80, 80, 80))
        label_x = bar_x + bar_w + 4
        painter.drawText(label_x, bar_top + 9, f"{self._v_max:.2f}")
        painter.drawText(
            label_x,
            (bar_top + bar_bottom) // 2 + 3,
            f"{(self._v_min + self._v_max) / 2.0:.2f}",
        )
        painter.drawText(label_x, bar_bottom, f"{self._v_min:.2f}")
        painter.end()


if HAS_PG and hasattr(pg, "ViewBox"):
    class _SelectionViewBox(pg.ViewBox):
        """ViewBox with chart-space rectangular selection."""

        def __init__(self):
            super().__init__(enableMenu=False)
            self._selection_enabled = False
            self._drag_origin = None
            self._selection_callback = None
            self._persistent_selection_item = None
            self._persistent_selection_bounds = None

        def set_selection_callback(self, callback):
            self._selection_callback = callback

        def set_selection_enabled(self, enabled: bool):
            self._selection_enabled = enabled
            if not enabled:
                self._drag_origin = None
                try:
                    self.rbScaleBox.hide()
                except Exception:
                    pass
            self.setMouseEnabled(x=False, y=False)
            try:
                self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
            except Exception:
                pass

        def wheelEvent(self, event, axis=None):
            event.accept()
            return

        def _ensure_persistent_selection_item(self):
            if self._persistent_selection_item is not None:
                return self._persistent_selection_item

            item = QGraphicsRectItem(self.childGroup)
            pen = QPen(QColor(*_SELECTION_BOX_EDGE))
            pen.setWidthF(1.5)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(*_SELECTION_BOX_FILL)))
            item.setZValue(1_000_000)
            try:
                item.setAcceptedMouseButtons(Qt.NoButton)
            except Exception:
                pass
            item.setVisible(False)
            self._persistent_selection_item = item
            return item

        def set_persistent_selection(self, bounds: Optional[Tuple[float, float, float, float]]):
            self._persistent_selection_bounds = bounds
            item = self._persistent_selection_item
            if (
                bounds is None
                or len(bounds) != 4
                or abs(bounds[1] - bounds[0]) <= 1e-9
                or abs(bounds[3] - bounds[2]) <= 1e-9
            ):
                if item is not None:
                    item.hide()
                return

            item = self._ensure_persistent_selection_item()
            x_min, x_max, y_min, y_max = bounds
            item.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
            item.show()

        def mouseClickEvent(self, event):
            if self._selection_enabled and event.button() == Qt.LeftButton:
                event.accept()
                return
            super().mouseClickEvent(event)

        def mouseDragEvent(self, event, axis=None):
            if self._selection_enabled and event.button() == Qt.LeftButton:
                event.accept()
                if event.isStart():
                    self._drag_origin = event.buttonDownPos()
                    self.updateScaleBox(self._drag_origin, event.pos())
                    return
                if event.isFinish():
                    origin = self._drag_origin or event.buttonDownPos()
                    self.rbScaleBox.hide()
                    self._drag_origin = None
                    lower = self.mapToView(origin)
                    upper = self.mapToView(event.pos())
                    x_min, x_max = sorted((float(lower.x()), float(upper.x())))
                    y_min, y_max = sorted((float(lower.y()), float(upper.y())))
                    if (
                        self._selection_callback is not None
                        and abs(x_max - x_min) > 1e-9
                        and abs(y_max - y_min) > 1e-9
                    ):
                        self._selection_callback((x_min, x_max, y_min, y_max))
                    return
                if self._drag_origin is not None:
                    self.updateScaleBox(self._drag_origin, event.pos())
                return
            super().mouseDragEvent(event, axis=axis)
else:
    class _SelectionViewBox:
        def __init__(self):
            self._selection_callback = None
            self._persistent_selection_bounds = None

        def set_selection_callback(self, callback):
            self._selection_callback = callback

        def set_selection_enabled(self, enabled: bool):
            return

        def set_persistent_selection(self, bounds):
            self._persistent_selection_bounds = bounds
            return


class ProfileChart(QWidget):
    """Overlay profile chart with shared display configuration."""

    points_selected = Signal(list)
    chart_selection_changed = Signal(list)
    undo_requested = Signal()
    display_config_changed = Signal(str, str, str, bool)
    appearance_changed = Signal(float, float)
    keep_only_requested = Signal(list)
    exclude_requested = Signal(list)
    dataset_filter_changed = Signal(list)
    # Emitted in workspace_mode when residue stats change.
    # Carries (html_text, is_active) so the host can update a side panel.
    residue_stats_updated = Signal(object, bool)

    def __init__(self, parent=None, *, workspace_mode: bool = False):
        super().__init__(parent)
        self._workspace_mode = bool(workspace_mode)
        self._profiles: List[dict] = []
        self._color_key = "hydrophobicity" if self._workspace_mode else "radius"
        self._x_key = "path_length"
        self._y_key = "radius"
        self._advanced_mode = self._workspace_mode
        self._alpha_scale = 1.0
        self._depth_scale = 1.0
        self._selection_mode = "off"
        self._selected_path_ids: Set[int] = set()
        self._selected_points: Dict[int, List[int]] = {}
        self._residue_stats: Optional[dict] = None
        self._profile_lookup: Dict[int, dict] = {}
        self._path_color_map: Dict[int, str] = {}
        self._dataset_filter_checks: Dict[str, QCheckBox] = {}
        self._selected_dataset_prefixes: Set[str] = set()
        self._undo_available = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        title_row = QHBoxLayout()
        self._title = QLabel("Radius Profile")
        self._title.setStyleSheet("font-weight: bold; font-size: 12px;")
        title_row.addWidget(self._title)
        title_row.addStretch()
        self._controls_toggle = QCheckBox("Controls")
        self._controls_toggle.toggled.connect(self._update_advanced_widgets)
        title_row.addWidget(self._controls_toggle)
        layout.addLayout(title_row)

        self._controls_panel = QWidget()
        controls_panel_layout = QVBoxLayout(self._controls_panel)
        controls_panel_layout.setContentsMargins(0, 0, 0, 0)
        controls_panel_layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self._preset_label = QLabel("Color:")
        toolbar.addWidget(self._preset_label)
        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(156)
        for label, key in CHART_PRESET_OPTIONS:
            self._preset_combo.addItem(label, key)
        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        toolbar.addWidget(self._preset_combo)

        self._advanced_toggle = QCheckBox("Advanced")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        toolbar.addWidget(self._advanced_toggle)

        self._selection_label = QLabel("Select:")
        toolbar.addWidget(self._selection_label)
        self._selection_segment = QWidget()
        self._selection_segment.setObjectName("selectionSegment")
        self._selection_segment.setStyleSheet(
            "QWidget#selectionSegment {"
            "background: #ffffff;"
            "border: 1px solid #d9e0ea;"
            "border-radius: 10px;"
            "}"
            "QWidget#selectionSegment QPushButton {"
            "border: none;"
            "background: transparent;"
            "color: #606266;"
            "padding: 4px 12px;"
            "min-width: 78px;"
            "font-size: 11px;"
            "border-radius: 9px;"
            "}"
            "QWidget#selectionSegment QPushButton:checked {"
            "background: #4a97ff;"
            "color: white;"
            "font-weight: 600;"
            "}"
        )
        selection_segment_layout = QHBoxLayout(self._selection_segment)
        selection_segment_layout.setContentsMargins(3, 3, 3, 3)
        selection_segment_layout.setSpacing(0)
        self._path_selection_btn = QPushButton("Path Selection")
        self._path_selection_btn.setCheckable(True)
        self._path_selection_btn.setFixedHeight(28)
        self._path_selection_btn.clicked.connect(
            lambda checked=False: self._on_selection_segment_clicked("path")
        )
        selection_segment_layout.addWidget(self._path_selection_btn)
        self._residue_selection_btn = QPushButton("Residue Stats")
        self._residue_selection_btn.setCheckable(True)
        self._residue_selection_btn.setFixedHeight(28)
        self._residue_selection_btn.clicked.connect(
            lambda checked=False: self._on_selection_segment_clicked("residue")
        )
        selection_segment_layout.addWidget(self._residue_selection_btn)
        toolbar.addWidget(self._selection_segment)

        self._opacity_label = QLabel("Opacity:")
        toolbar.addWidget(self._opacity_label)
        self._opacity_combo = QComboBox()
        self._opacity_combo.setMinimumWidth(76)
        for label, value in CHART_OPACITY_OPTIONS:
            self._opacity_combo.addItem(label, value)
        self._opacity_combo.currentIndexChanged.connect(self._on_appearance_control_changed)
        toolbar.addWidget(self._opacity_combo)

        self._depth_label = QLabel("Depth:")
        toolbar.addWidget(self._depth_label)
        self._depth_combo = QComboBox()
        self._depth_combo.setMinimumWidth(84)
        for label, value in CHART_DEPTH_OPTIONS:
            self._depth_combo.addItem(label, value)
        self._depth_combo.currentIndexChanged.connect(self._on_appearance_control_changed)
        toolbar.addWidget(self._depth_combo)
        toolbar.addStretch()

        controls_panel_layout.addLayout(toolbar)

        self._advanced_row = QWidget()
        advanced_layout = QHBoxLayout(self._advanced_row)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)
        self._x_label = QLabel("X:")
        self._x_combo = QComboBox()
        self._x_combo.setMinimumWidth(136)
        self._y_label = QLabel("Y:")
        self._y_combo = QComboBox()
        self._y_combo.setMinimumWidth(136)
        for label, key in CHART_AXIS_OPTIONS:
            self._x_combo.addItem(label, key)
            self._y_combo.addItem(label, key)
        self._x_combo.currentIndexChanged.connect(self._on_axis_changed)
        self._y_combo.currentIndexChanged.connect(self._on_axis_changed)
        advanced_layout.addWidget(self._x_label)
        advanced_layout.addWidget(self._x_combo)
        advanced_layout.addWidget(self._y_label)
        advanced_layout.addWidget(self._y_combo)
        advanced_layout.addStretch()
        controls_panel_layout.addWidget(self._advanced_row)
        self._dataset_filter_widget = QWidget()
        dataset_filter_layout = QVBoxLayout(self._dataset_filter_widget)
        dataset_filter_layout.setContentsMargins(0, 0, 0, 0)
        dataset_filter_layout.setSpacing(4)
        self._dataset_filter_label = QLabel("Datasets:")
        dataset_filter_layout.addWidget(self._dataset_filter_label)
        self._dataset_filter_scroll = QScrollArea()
        self._dataset_filter_scroll.setWidgetResizable(True)
        self._dataset_filter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._dataset_filter_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._dataset_filter_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._dataset_filter_scroll.setMinimumHeight(42)
        self._dataset_filter_scroll.setMaximumHeight(48)
        self._dataset_filter_checks_container = QWidget()
        self._dataset_filter_checks_layout = QHBoxLayout(self._dataset_filter_checks_container)
        self._dataset_filter_checks_layout.setContentsMargins(0, 0, 0, 0)
        self._dataset_filter_checks_layout.setSpacing(6)
        self._dataset_filter_scroll.setWidget(self._dataset_filter_checks_container)
        dataset_filter_layout.addWidget(self._dataset_filter_scroll)
        self._dataset_filter_widget.setVisible(self._workspace_mode)
        controls_panel_layout.addWidget(self._dataset_filter_widget)
        layout.addWidget(self._controls_panel)

        self._selection_bar = QWidget()
        selection_bar_layout = QHBoxLayout(self._selection_bar)
        selection_bar_layout.setContentsMargins(0, 0, 0, 0)
        selection_bar_layout.setSpacing(6)
        self._selection_info = QLabel("")
        self._selection_info.setStyleSheet(
            "color: #409eff; background: #ecf5ff; border-radius: 4px; padding: 2px 8px;"
        )
        selection_bar_layout.addWidget(self._selection_info)
        self._keep_only_btn = QPushButton("Keep Only")
        self._keep_only_btn.setFixedHeight(22)
        self._keep_only_btn.clicked.connect(self._emit_keep_only)
        selection_bar_layout.addWidget(self._keep_only_btn)
        self._exclude_btn = QPushButton("Exclude")
        self._exclude_btn.setFixedHeight(22)
        self._exclude_btn.clicked.connect(self._emit_exclude)
        selection_bar_layout.addWidget(self._exclude_btn)
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setFixedHeight(22)
        self._undo_btn.setEnabled(False)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        selection_bar_layout.addWidget(self._undo_btn)
        self._clear_selection_btn = QPushButton("Clear")
        self._clear_selection_btn.setFixedHeight(22)
        self._clear_selection_btn.clicked.connect(self._clear_chart_selection)
        selection_bar_layout.addWidget(self._clear_selection_btn)
        selection_bar_layout.addStretch()
        layout.addWidget(self._selection_bar)

        self._residue_summary = QLabel("")
        self._residue_summary.setWordWrap(True)
        self._residue_summary.setStyleSheet(
            "background: #fbfcfe; border: 1px solid #e5ecf6; border-radius: 8px; "
            "padding: 10px 12px; color: #303133; font-size: 11px;"
        )
        self._residue_summary_scroll = None
        if self._workspace_mode:
            # In workspace mode the residue summary is hosted by the parent
            # dialog in a dedicated right-hand column, so we do NOT add it to
            # this widget's own layout.  We still keep the QLabel alive so
            # _update_selection_panels can populate it; visibility is managed
            # externally via the residue_stats_updated signal.
            self._residue_summary.setAlignment(Qt.AlignTop | Qt.AlignLeft)
            self._residue_summary.setVisible(False)
        else:
            self._residue_summary.setVisible(False)

        if not HAS_PG:
            layout.addWidget(QLabel("pyqtgraph not installed"))
            self._plot = None
            self._colorbar = None
            self._plot_container = None
            self._hint = None
            self._set_selection_segment_mode(self._selection_mode)
            self._set_combo_value(self._opacity_combo, self._alpha_scale)
            self._set_combo_value(self._depth_combo, self._depth_scale)
            self._set_combo_data(self._x_combo, self._x_key)
            self._set_combo_data(self._y_combo, self._y_key)
            self._controls_toggle.setChecked(self._workspace_mode)
            self._update_advanced_widgets()
            self._update_selection_panels()
            return

        self._view_box = _SelectionViewBox()
        self._view_box.set_selection_callback(self._on_selection_box_finished)
        self._plot = pg.PlotWidget(viewBox=self._view_box)
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.22)

        self._colorbar = _ScalarColorBar()
        self._colorbar.setVisible(False)

        self._plot_container = QWidget()
        plot_layout = QHBoxLayout(self._plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_layout.addWidget(self._plot, 1)
        plot_layout.addWidget(self._colorbar)
        if self._workspace_mode:
            self._plot.setMinimumHeight(280)
            self._plot_container.setMinimumHeight(294)
        self._plot_container.setVisible(False)
        layout.addWidget(self._plot_container, 1)

        self._hint = QLabel("")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #909399; font-size: 11px; padding: 20px;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        # In workspace_mode the residue summary lives in the host dialog's
        # right-hand panel, not inside this widget.
        if not self._workspace_mode:
            layout.addWidget(self._residue_summary)

        self._set_selection_segment_mode(self._selection_mode)
        self._set_combo_value(self._opacity_combo, self._alpha_scale)
        self._set_combo_value(self._depth_combo, self._depth_scale)
        self._set_combo_data(self._x_combo, self._x_key)
        self._set_combo_data(self._y_combo, self._y_key)
        self._controls_toggle.setChecked(self._workspace_mode)
        self._update_advanced_widgets()
        self._update_selection_panels()
        self._update_text()

    def display_config(self) -> Tuple[str, str, str, bool]:
        return self._color_key, self._x_key, self._y_key, self._advanced_mode

    def appearance(self) -> Tuple[float, float]:
        return self._alpha_scale, self._depth_scale

    def selected_dataset_prefixes(self) -> List[str]:
        return sorted(self._selected_dataset_prefixes)

    def set_display_config(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
        *,
        emit_signal: bool = False,
    ):
        if color_key not in AXIS_META and color_key != "dataset":
            color_key = "radius"
        if x_key not in AXIS_META:
            x_key = "path_length"
        if y_key not in AXIS_META:
            y_key = "radius"
        if not advanced_mode:
            x_key = "path_length"
            if y_key not in AXIS_META:
                y_key = "radius"

        changed = (
            self._color_key != color_key
            or self._x_key != x_key
            or self._y_key != y_key
            or self._advanced_mode != advanced_mode
        )

        self._color_key = color_key
        self._x_key = x_key
        self._y_key = y_key
        self._advanced_mode = advanced_mode

        controls = [
            self._preset_combo,
            self._advanced_toggle,
            self._x_combo,
            self._y_combo,
        ]
        previous = [control.blockSignals(True) for control in controls]
        self._set_combo_data(self._preset_combo, self._color_key)
        self._advanced_toggle.setChecked(self._advanced_mode)
        self._set_combo_data(self._x_combo, self._x_key)
        self._set_combo_data(self._y_combo, self._y_key)
        for control, was_blocked in zip(controls, previous):
            control.blockSignals(was_blocked)

        self._update_advanced_widgets()
        self._update_text()
        self._replot()
        if emit_signal and changed:
            self.display_config_changed.emit(
                self._color_key,
                self._x_key,
                self._y_key,
                self._advanced_mode,
            )

    def set_appearance(
        self,
        alpha_scale: float,
        depth_scale: float,
        *,
        emit_signal: bool = False,
    ):
        alpha_scale = float(_clamp(alpha_scale, 0.5, 1.4))
        depth_scale = float(_clamp(depth_scale, 0.6, 1.4))
        changed = (
            abs(self._alpha_scale - alpha_scale) > 1e-6
            or abs(self._depth_scale - depth_scale) > 1e-6
        )
        self._alpha_scale = alpha_scale
        self._depth_scale = depth_scale

        controls = [self._opacity_combo, self._depth_combo]
        previous = [control.blockSignals(True) for control in controls]
        self._set_combo_value(self._opacity_combo, self._alpha_scale)
        self._set_combo_value(self._depth_combo, self._depth_scale)
        for control, was_blocked in zip(controls, previous):
            control.blockSignals(was_blocked)

        self._replot()
        if emit_signal and changed:
            self.appearance_changed.emit(self._alpha_scale, self._depth_scale)

    def set_profiles(self, profiles: List[dict]):
        self._profiles = profiles
        self._profile_lookup = {
            int(profile.get("pathIndex", -1)): profile
            for profile in self._profiles
            if int(profile.get("pathIndex", -1)) >= 0
        }
        self._sync_dataset_filter_options()
        if not self._workspace_mode:
            self._selected_dataset_prefixes = {
                _dataset_prefix(profile)
                for profile in self._profiles
            }
        self._clear_chart_selection(update_plot=False)
        self.chart_selection_changed.emit([])
        self._replot()

    def all_residue_summary_rows(self) -> List[dict]:
        """Build residue-summary rows for all currently visible chart paths."""
        filtered_profiles = self._filtered_profiles()
        if not filtered_profiles:
            return []
        profile_map = {
            int(profile.get("pathIndex", -1)): profile
            for profile in filtered_profiles
            if int(profile.get("pathIndex", -1)) >= 0
        }
        selected_points = _all_profile_point_selection(filtered_profiles)
        return _residue_summary_rows_by_dataset(profile_map, selected_points)

    def set_path_color_map(self, path_color_map: Dict[int, str]):
        self._path_color_map = {
            int(path_id): str(color)
            for path_id, color in (path_color_map or {}).items()
        }
        self._refresh_dataset_filter_styles()
        self._replot()

    def clear(self):
        self._profiles = []
        self._profile_lookup = {}
        self._clear_chart_selection(update_plot=False)
        self.chart_selection_changed.emit([])
        if self._plot is not None:
            self._plot.clear()
            self._plot_container.setVisible(False)
            self._colorbar.setVisible(False)
        if self._hint is not None:
            self._hint.setVisible(True)
        self._update_text(path_count=0)

    def _set_selection_mode(self, mode: str):
        if mode not in {"off", "path", "residue"}:
            mode = "off"
        self._selection_mode = mode
        self._set_selection_segment_mode(self._selection_mode)
        self._clear_chart_selection(update_plot=False)
        if hasattr(self, "_view_box"):
            self._view_box.set_selection_enabled(mode != "off")
        self._update_advanced_widgets()
        self._replot()

    def _set_combo_data(self, combo: QComboBox, value: str):
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _set_combo_value(self, combo: QComboBox, value: float):
        for idx in range(combo.count()):
            if abs(float(combo.itemData(idx)) - value) < 1e-6:
                combo.setCurrentIndex(idx)
                return
        combo.setCurrentIndex(0)

    def _set_selection_segment_mode(self, mode: str):
        controls = [
            (self._path_selection_btn, mode == "path"),
            (self._residue_selection_btn, mode == "residue"),
        ]
        for button, checked in controls:
            was_blocked = button.blockSignals(True)
            button.setChecked(checked)
            button.blockSignals(was_blocked)

    def _on_preset_changed(self):
        color_key = self._preset_combo.currentData()
        x_key = self._x_key
        y_key = self._y_key
        self.set_display_config(
            color_key,
            x_key,
            y_key,
            self._advanced_mode,
            emit_signal=True,
        )

    def _on_advanced_toggled(self, checked: bool):
        x_key = self._x_combo.currentData() if checked else "path_length"
        y_key = self._y_combo.currentData() if checked else self._y_key
        self.set_display_config(
            self._color_key,
            x_key,
            y_key,
            checked,
            emit_signal=True,
        )

    def _on_axis_changed(self):
        if not self._advanced_mode:
            return
        self.set_display_config(
            self._color_key,
            self._x_combo.currentData(),
            self._y_combo.currentData(),
            True,
            emit_signal=True,
        )

    def _on_appearance_control_changed(self):
        self.set_appearance(
            float(self._opacity_combo.currentData()),
            float(self._depth_combo.currentData()),
            emit_signal=True,
        )

    def _on_selection_segment_clicked(self, mode: str):
        next_mode = "off" if self._selection_mode == mode else mode
        self._set_selection_mode(next_mode)

    def _emit_keep_only(self):
        if self._selected_path_ids:
            self.keep_only_requested.emit(sorted(self._selected_path_ids))

    def _emit_exclude(self):
        if self._selected_path_ids:
            self.exclude_requested.emit(sorted(self._selected_path_ids))

    def _clear_chart_selection(self, *, update_plot: bool = True):
        self._selected_path_ids = set()
        self._selected_points = {}
        self._residue_stats = None
        if hasattr(self, "_view_box"):
            self._view_box.set_persistent_selection(None)
        self._update_selection_panels()
        self.chart_selection_changed.emit([])
        if update_plot:
            self._replot()

    def _sync_dataset_filter_options(self):
        prefixes = []
        seen = set()
        for profile in self._profiles:
            prefix = _dataset_prefix(profile)
            if prefix in seen:
                continue
            seen.add(prefix)
            prefixes.append(prefix)
        if not prefixes:
            self._selected_dataset_prefixes = set()
        elif not self._selected_dataset_prefixes:
            self._selected_dataset_prefixes = set(prefixes)
        else:
            self._selected_dataset_prefixes &= set(prefixes)
            if not self._selected_dataset_prefixes:
                self._selected_dataset_prefixes = set(prefixes)

        for checkbox in self._dataset_filter_checks.values():
            checkbox.deleteLater()
        self._dataset_filter_checks.clear()
        while self._dataset_filter_checks_layout.count():
            item = self._dataset_filter_checks_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        dataset_colors = self._dataset_colors_by_prefix(prefixes)
        for prefix in prefixes:
            checkbox = QPushButton(prefix)
            checkbox.setCheckable(True)
            checkbox.setChecked(prefix in self._selected_dataset_prefixes)
            checkbox.setMinimumHeight(28)
            checkbox.toggled.connect(self._on_dataset_filter_changed)
            self._dataset_filter_checks_layout.addWidget(checkbox, 0)
            self._dataset_filter_checks[prefix] = checkbox
            checkbox.setStyleSheet(
                _dataset_button_stylesheet(
                    dataset_colors.get(prefix, "#4169E1"),
                    checkbox.isChecked(),
                )
            )
        self._dataset_filter_checks_layout.addStretch()

    def _on_dataset_filter_changed(self):
        selected = {
            prefix
            for prefix, checkbox in self._dataset_filter_checks.items()
            if checkbox.isChecked()
        }
        if not selected and self._dataset_filter_checks:
            first_prefix = next(iter(self._dataset_filter_checks))
            self._dataset_filter_checks[first_prefix].blockSignals(True)
            self._dataset_filter_checks[first_prefix].setChecked(True)
            self._dataset_filter_checks[first_prefix].blockSignals(False)
            selected = {first_prefix}
        self._selected_dataset_prefixes = selected
        self._refresh_dataset_filter_styles()
        self.dataset_filter_changed.emit(sorted(self._selected_dataset_prefixes))
        self._clear_chart_selection(update_plot=False)
        self._replot()

    def _dataset_colors_by_prefix(self, prefixes: List[str]) -> Dict[str, str]:
        colors: Dict[str, str] = {}
        for prefix in prefixes:
            for profile in self._profiles:
                if _dataset_prefix(profile) != prefix:
                    continue
                path_id = int(profile.get("pathIndex", -1))
                colors[prefix] = self._path_color_map.get(path_id, "#4169E1")
                break
            colors.setdefault(prefix, "#4169E1")
        return colors

    def _refresh_dataset_filter_styles(self):
        if not self._dataset_filter_checks:
            return
        dataset_colors = self._dataset_colors_by_prefix(list(self._dataset_filter_checks.keys()))
        for prefix, checkbox in self._dataset_filter_checks.items():
            checkbox.setStyleSheet(
                _dataset_button_stylesheet(
                    dataset_colors.get(prefix, "#4169E1"),
                    checkbox.isChecked(),
                )
            )

    def _filtered_profiles(self) -> List[dict]:
        if not self._selected_dataset_prefixes:
            return list(self._profiles)
        return [
            profile for profile in self._profiles
            if _dataset_prefix(profile) in self._selected_dataset_prefixes
        ]

    def _filtered_profile_lookup(self) -> Dict[int, dict]:
        return {
            int(profile.get("pathIndex", -1)): profile
            for profile in self._filtered_profiles()
            if int(profile.get("pathIndex", -1)) >= 0
        }

    def _on_selection_box_finished(self, bounds: Tuple[float, float, float, float]):
        filtered_profiles = self._filtered_profiles()
        if self._selection_mode == "off" or not filtered_profiles:
            return

        selected_paths, selected_points = _collect_selected_points(
            filtered_profiles,
            self._x_key,
            self._y_key,
            bounds,
        )
        self._selected_points = selected_points
        filtered_lookup = self._filtered_profile_lookup()

        if self._selection_mode == "residue":
            # Residue stats mode follows the frontend behavior: only the points
            # inside the chart selection contribute to statistics, without
            # treating whole paths as selected/highlighted.
            self._selected_path_ids = set()
            self._residue_stats = _build_residue_stats_from_map(
                filtered_lookup,
                selected_points,
            )
        else:
            self._selected_path_ids = selected_paths
            self._residue_stats = None

        persistent_bounds = _persistent_selection_bounds_for_mode(
            self._selection_mode,
            bounds,
            self._selected_path_ids,
            self._selected_points,
            self._residue_stats,
        )
        if hasattr(self, "_view_box"):
            self._view_box.set_persistent_selection(persistent_bounds)

        self._update_selection_panels()
        self.chart_selection_changed.emit(sorted(self._selected_path_ids))
        if self._selection_mode == "path":
            self._replot()

    def selected_path_ids(self) -> List[int]:
        return sorted(self._selected_path_ids)

    def filtered_selected_path_ids(
        self,
        *,
        hydrophobicity_min: Optional[float] = None,
        hydrophobicity_max: Optional[float] = None,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[int]:
        return self.filter_path_ids(
            self.selected_path_ids(),
            hydrophobicity_min=hydrophobicity_min,
            hydrophobicity_max=hydrophobicity_max,
            frame_min=frame_min,
            frame_max=frame_max,
        )

    def filter_path_ids(
        self,
        path_ids: List[int],
        *,
        hydrophobicity_min: Optional[float] = None,
        hydrophobicity_max: Optional[float] = None,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[int]:
        result: List[int] = []
        for path_id in path_ids or []:
            profile = self._profile_lookup.get(int(path_id))
            if profile is None:
                continue
            hydro_values = np.asarray(profile.get("hydrophobicity", []), dtype=np.float32)
            avg_hydro = float(np.nanmean(hydro_values)) if hydro_values.size else 0.0
            frame_id = int(profile.get("frameId", 0) or 0)
            if hydrophobicity_min is not None and avg_hydro < float(hydrophobicity_min):
                continue
            if hydrophobicity_max is not None and avg_hydro > float(hydrophobicity_max):
                continue
            if frame_min is not None and frame_id < int(frame_min):
                continue
            if frame_max is not None and frame_id > int(frame_max):
                continue
            result.append(int(path_id))
        return result

    def apply_filtered_path_selection(self, path_ids: List[int]):
        self._selected_path_ids = {
            int(path_id)
            for path_id in (path_ids or [])
            if int(path_id) in self._profile_lookup
        }
        self._update_selection_panels()
        self.chart_selection_changed.emit(sorted(self._selected_path_ids))
        if self._selection_mode == "path":
            self._replot()

    def set_undo_available(self, available: bool):
        self._undo_available = bool(available)
        self._update_selection_panels()

    def _update_selection_panels(self):
        path_mode_active = self._selection_mode == "path" and bool(self._selected_path_ids)
        residue_mode_active = self._selection_mode == "residue"

        self._selection_bar.setVisible(path_mode_active or self._undo_available)
        if path_mode_active:
            self._selection_info.setText(f"{len(self._selected_path_ids)} paths selected")
        elif self._undo_available:
            self._selection_info.setText("Chart history available")
        else:
            self._selection_info.setText("")
        self._undo_btn.setEnabled(self._undo_available)

        if residue_mode_active:
            if self._residue_stats:
                if self._workspace_mode:
                    rows = _residue_summary_rows_by_dataset(
                        self._filtered_profile_lookup(),
                        self._selected_points,
                    )
                else:
                    rows = _residue_summary_rows(self._residue_stats)
                html_text = (
                    f"Selected residue statistics: {len(rows)} rows"
                    if rows else
                    "No residue statistics rows."
                )
                self._residue_summary.setText(html_text)
            else:
                rows = []
                html_text = (
                    "Switch to Residue Stats, then drag a selection box on the chart to summarize wall and bottleneck residues."
                )
                self._residue_summary.setText(html_text)
            if self._workspace_mode:
                self.residue_stats_updated.emit(rows, True)
            else:
                self._residue_summary.setVisible(True)
            return

        if self._workspace_mode:
            self.residue_stats_updated.emit([], False)
        else:
            self._residue_summary.setVisible(False)

    def _update_advanced_widgets(self, *_args):
        controls_visible = self._controls_toggle.isChecked()
        advanced_visible = controls_visible and self._advanced_mode
        self._controls_panel.setVisible(controls_visible)
        self._opacity_label.setVisible(advanced_visible)
        self._opacity_combo.setVisible(advanced_visible)
        self._depth_label.setVisible(advanced_visible)
        self._depth_combo.setVisible(advanced_visible)
        self._advanced_row.setVisible(advanced_visible)

    def _update_text(self, path_count: int | None = None):
        if path_count is None:
            path_count = len(self._filtered_profiles())
        x_meta = AXIS_META[self._x_key]
        y_meta = AXIS_META[self._y_key]
        self._title.setText(f"{y_meta['title']} Profile ({path_count} paths)")
        if self._plot is not None:
            self._plot.setLabel("left", y_meta["axis_label"])
            self._plot.setLabel("bottom", x_meta["axis_label"])
        if self._hint is not None:
            selection_hint = "Selection: off"
            if self._selection_mode == "path":
                selection_hint = "Selection: drag a box to pick paths"
            elif self._selection_mode == "residue":
                selection_hint = "Selection: drag a box to summarize residues"
            self._hint.setText(
                f"Overlay of per-path {y_meta['hint_label']} versus {x_meta['hint_label']}.\n"
                "Each line = one tunnel path.\n\n"
                "Preset controls color mapping only.\n"
                "Enable Advanced to freely choose X/Y axis attributes.\n"
                "Opacity and Depth adjust line visibility and palette richness.\n"
                "Workspace datasets can be filtered independently.\n"
                f"{selection_hint}"
            )

    def _replot(self):
        self._update_text()
        if self._plot is None:
            return

        self._plot.clear()
        filtered_profiles = self._filtered_profiles()
        has_data = bool(filtered_profiles)
        self._plot_container.setVisible(has_data)
        self._hint.setVisible(not has_data)
        self._colorbar.setVisible(False)

        if not has_data:
            return

        base_alpha, base_line_width = _profile_overlay_style(len(filtered_profiles))
        alpha = _scale_alpha(base_alpha, self._alpha_scale)
        line_width = max(0.52, base_line_width * (0.96 + 0.12 * self._depth_scale))
        scale = _build_color_scale(filtered_profiles, self._color_key) if self._color_key != "dataset" else {}
        target_depth_scale = max(self._depth_scale, _TARGET_LINE_DEPTH_BOOST)
        groups: Dict[Tuple[int, int, int], List[Tuple[np.ndarray, np.ndarray]]] = {}
        selected_series: List[Tuple[np.ndarray, np.ndarray]] = []
        highlight_markers = len(filtered_profiles) <= 240

        for profile in filtered_profiles:
            xs, ys = _paired_series(profile, self._x_key, self._y_key)
            if len(xs) == 0:
                continue
            if self._color_key == "dataset":
                path_id = int(profile.get("pathIndex", -1))
                color_text = self._path_color_map.get(path_id, "#4169E1")
                qcolor = QColor(color_text)
                color = (qcolor.red(), qcolor.green(), qcolor.blue())
            else:
                color_value = _profile_scalar(profile, self._color_key)
                color = _apply_depth_scale(_quantized_color(color_value, scale), target_depth_scale)
            groups.setdefault(color, []).append((xs, ys))
            if int(profile.get("pathIndex", -1)) in self._selected_path_ids:
                selected_series.append((xs, ys))

        for color, paths in groups.items():
            segs_x: List[np.ndarray] = []
            segs_y: List[np.ndarray] = []
            for xs, ys in paths:
                segs_x.append(xs.astype(np.float64))
                segs_x.append(np.array([np.nan], dtype=np.float64))
                segs_y.append(ys.astype(np.float64))
                segs_y.append(np.array([np.nan], dtype=np.float64))
            if not segs_x:
                continue
            all_x = np.concatenate(segs_x)
            all_y = np.concatenate(segs_y)
            pen = pg.mkPen(color=pg.mkColor(color[0], color[1], color[2], alpha), width=line_width)
            self._plot.plot(all_x, all_y, pen=pen, connect="finite")

        for xs, ys in selected_series:
            highlight_color = _selected_highlight_color(self._depth_scale)
            selected_pen = pg.mkPen(
                color=pg.mkColor(
                    highlight_color[0],
                    highlight_color[1],
                    highlight_color[2],
                    min(205, alpha + 58),
                ),
                width=line_width + 0.95,
            )
            self._plot.plot(xs.astype(np.float64), ys.astype(np.float64), pen=selected_pen)
            if highlight_markers:
                self._plot.plot(
                    xs.astype(np.float64),
                    ys.astype(np.float64),
                    pen=None,
                    symbol="o",
                    symbolBrush=pg.mkBrush(
                        highlight_color[0],
                        highlight_color[1],
                        highlight_color[2],
                        168,
                    ),
                    symbolPen=pg.mkPen(
                        color=pg.mkColor(
                            _SELECTED_MARKER_OUTLINE[0],
                            _SELECTED_MARKER_OUTLINE[1],
                            _SELECTED_MARKER_OUTLINE[2],
                            180,
                        ),
                        width=0.6,
                    ),
                    symbolSize=4.2,
                )

        if self._color_key != "dataset":
            self._colorbar.set_scale(
                [
                    (position, _apply_depth_scale(rgb, target_depth_scale))
                    for position, rgb in scale["stops"]
                ],
                float(scale["v_min"]),
                float(scale["v_max"]),
                AXIS_META[self._color_key]["title"],
            )
            self._colorbar.setVisible(True)
        _lock_plot_to_bounds(
            self._plot,
            _compute_profile_bounds(filtered_profiles, self._x_key, self._y_key),
        )


class StatisticsChart(QWidget):
    """Aggregated statistics chart (percentile bands + mean)."""

    def __init__(self, parent=None, *, show_bottleneck_points: bool = False):
        super().__init__(parent)
        self._profiles: List[dict] = []
        self._color_key = "radius"
        self._x_key = "path_length"
        self._y_key = "radius"
        self._path_color_map: Dict[int, str] = {}
        self._selected_dataset_prefixes: Set[str] = set()
        self._show_bottleneck_points = bool(show_bottleneck_points)
        self._bottleneck_point_cache: Dict[Tuple[int, str, str], Tuple[np.ndarray, np.ndarray]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)

        self._title = QLabel("Radius Statistics")
        self._title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self._title, 1)

        self._legend_widget = QWidget()
        self._legend_layout = QHBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(0, 0, 0, 0)
        self._legend_layout.setSpacing(8)
        self._legend_widget.setVisible(False)
        header_layout.addWidget(self._legend_widget, 0, Qt.AlignRight)

        layout.addLayout(header_layout)

        if not HAS_PG:
            layout.addWidget(QLabel("pyqtgraph not installed"))
            self._plot = None
            self._hint = None
            return

        self._plot = pg.PlotWidget(viewBox=pg.ViewBox(enableMenu=False))
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.22)
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.setVisible(False)
        layout.addWidget(self._plot)

        self._hint = QLabel("")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setStyleSheet("color: #909399; font-size: 11px; padding: 20px;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self._update_text()

    def set_display_config(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
    ):
        self._color_key = color_key if color_key in AXIS_META or color_key == "dataset" else "radius"
        self._x_key = x_key if x_key in AXIS_META else "path_length"
        self._y_key = y_key if y_key in AXIS_META else "radius"
        if not advanced_mode:
            self._x_key = "path_length"
            if self._y_key not in AXIS_META:
                self._y_key = "radius"
        self._update_text()
        self._replot()

    def set_profiles(self, profiles: List[dict]):
        self._profiles = profiles
        self._bottleneck_point_cache.clear()
        prefixes = {_dataset_prefix(profile) for profile in self._profiles}
        if not self._selected_dataset_prefixes:
            self._selected_dataset_prefixes = set(prefixes)
        else:
            self._selected_dataset_prefixes &= prefixes
            if not self._selected_dataset_prefixes:
                self._selected_dataset_prefixes = set(prefixes)
        self._replot()

    def set_path_color_map(self, path_color_map: Dict[int, str]):
        self._path_color_map = {
            int(path_id): str(color)
            for path_id, color in (path_color_map or {}).items()
        }
        self._replot()

    def set_dataset_filter(self, dataset_prefixes: List[str]):
        self._selected_dataset_prefixes = {str(item) for item in (dataset_prefixes or [])}
        self._replot()

    def _filtered_profiles(self) -> List[dict]:
        if not self._selected_dataset_prefixes:
            return list(self._profiles)
        return [
            profile for profile in self._profiles
            if _dataset_prefix(profile) in self._selected_dataset_prefixes
        ]

    def clear(self):
        self._profiles = []
        self._bottleneck_point_cache.clear()
        if self._plot is not None:
            self._plot.clear()
            self._plot.setVisible(False)
        if self._hint is not None:
            self._hint.setVisible(True)
        self._update_legend([])
        self._update_text(path_count=0)

    def _update_text(self, path_count: int | None = None):
        if path_count is None:
            path_count = len(self._filtered_profiles())
        x_meta = AXIS_META[self._x_key]
        y_meta = AXIS_META[self._y_key]
        self._title.setText(f"{y_meta['title']} Statistics ({path_count} paths)")
        if self._plot is not None:
            self._plot.setLabel("left", y_meta["axis_label"])
            self._plot.setLabel("bottom", x_meta["axis_label"])
        if self._hint is not None:
            self._hint.setText(
                f"Aggregated {y_meta['hint_label']} statistics across paths.\n\n"
                "Displays mean, median, and percentile bands for each dataset.\n"
                f"X axis: {x_meta['title']}\n"
                f"Y axis: {y_meta['title']}"
            )

    def _cached_bottleneck_series_for_profile(
        self,
        profile: dict,
    ) -> Tuple[np.ndarray, np.ndarray]:
        cache_key = (id(profile), self._x_key, self._y_key)
        cached = self._bottleneck_point_cache.get(cache_key)
        if cached is not None:
            return cached
        result = _bottleneck_series_for_profile(profile, self._x_key, self._y_key)
        self._bottleneck_point_cache[cache_key] = result
        if len(self._bottleneck_point_cache) > 12000:
            self._bottleneck_point_cache.clear()
        return result

    def _replot(self):
        self._update_text()
        if self._plot is None:
            return

        self._plot.clear()
        filtered_profiles = self._filtered_profiles()
        if not filtered_profiles:
            self._plot.setVisible(False)
            self._hint.setVisible(True)
            self._update_legend([])
            return

        self._hint.setVisible(False)
        self._plot.setVisible(True)
        grouped: Dict[str, List[dict]] = {}
        for profile in filtered_profiles:
            grouped.setdefault(_dataset_prefix(profile), []).append(profile)
        any_plotted = False
        legend_items: List[Tuple[str, str]] = []
        for prefix, profiles in grouped.items():
            x_vals, p10, p25, p50, p75, p90, mean = _aggregate_profile_samples(
                profiles,
                self._x_key,
                self._y_key,
            )
            if len(x_vals) == 0:
                continue
            any_plotted = True
            path_id = int(profiles[0].get("pathIndex", -1))
            color_text = self._path_color_map.get(path_id, "#4169E1")
            legend_items.append((prefix, color_text))
            qcolor = QColor(color_text)
            fill_10_90_alpha, fill_25_75_alpha, median_width, mean_width = _statistics_fill_style(
                len(profiles)
            )
            outer = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), fill_10_90_alpha)
            inner = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), fill_25_75_alpha)
            median = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), 220)
            mean_color = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), 255)
            fill_10_90 = pg.FillBetweenItem(
                pg.PlotDataItem(x_vals, p10),
                pg.PlotDataItem(x_vals, p90),
                brush=pg.mkBrush(outer),
            )
            self._plot.addItem(fill_10_90)
            fill_25_75 = pg.FillBetweenItem(
                pg.PlotDataItem(x_vals, p25),
                pg.PlotDataItem(x_vals, p75),
                brush=pg.mkBrush(inner),
            )
            self._plot.addItem(fill_25_75)
            median_pen = pg.mkPen(color=median, width=median_width, style=Qt.DashLine)
            self._plot.plot(x_vals, p50, pen=median_pen, name=f"{prefix} median")
            mean_pen = pg.mkPen(color=mean_color, width=mean_width)
            self._plot.plot(x_vals, mean, pen=mean_pen, name=f"{prefix} mean")
            if self._show_bottleneck_points and self._y_key == "radius":
                bottleneck_x: List[np.ndarray] = []
                bottleneck_y: List[np.ndarray] = []
                sampled_profiles = _sample_profiles_for_overlay(
                    profiles,
                    _STAT_BOTTLENECK_POINT_LIMIT_PER_DATASET,
                )
                for profile in sampled_profiles:
                    bx, by = self._cached_bottleneck_series_for_profile(
                        profile
                    )
                    if len(bx) > 0:
                        bottleneck_x.append(bx.astype(np.float64))
                        bottleneck_y.append(by.astype(np.float64))
                if bottleneck_x:
                    point_color = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), 42)
                    point_edge = pg.mkColor(qcolor.red(), qcolor.green(), qcolor.blue(), 62)
                    self._plot.plot(
                        np.concatenate(bottleneck_x),
                        np.concatenate(bottleneck_y),
                        pen=None,
                        symbol="o",
                        symbolSize=2.4,
                        symbolBrush=pg.mkBrush(point_color),
                        symbolPen=pg.mkPen(point_edge, width=0.25),
                    )
        self._update_legend(legend_items)
        if not any_plotted:
            self._plot.setVisible(False)
            self._hint.setVisible(True)
            self._update_legend([])
            return
        _lock_plot_to_bounds(
            self._plot,
            _compute_profile_bounds(filtered_profiles, self._x_key, self._y_key),
        )

    def _update_legend(self, items: List[Tuple[str, str]]):
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        if not items:
            self._legend_widget.setVisible(False)
            return

        for label_text, color_text in items:
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(0, 0, 0, 0)
            item_layout.setSpacing(4)

            swatch = QLabel()
            swatch.setFixedSize(10, 10)
            swatch.setStyleSheet(
                "border-radius: 5px; "
                f"background-color: {color_text}; "
                "border: 1px solid rgba(0,0,0,0.18);"
            )
            item_layout.addWidget(swatch, 0, Qt.AlignVCenter)

            text = QLabel(label_text)
            text.setStyleSheet("font-size: 11px; color: #4b5563;")
            item_layout.addWidget(text, 0, Qt.AlignVCenter)

            self._legend_layout.addWidget(item_widget, 0, Qt.AlignRight)

        self._legend_layout.addStretch(1)
        self._legend_widget.setVisible(True)
