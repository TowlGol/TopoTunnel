"""
Helpers for querying residue-combination statistics CSV outputs.
"""
from __future__ import annotations

import csv
import math
import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple


COMBINATION_PROPERTY_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("Affected Paths", "affected_path_count"),
    ("Affected Points", "affected_point_count"),
    ("Mean Radius", "affected_point_radius_avg"),
    ("Path-weighted Radius", "weighted_radius_score"),
    ("Radius Range", "affected_point_radius_range"),
)

BOTTLENECK_NEIGHBOR_POINT_WINDOW = 0


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def bottleneck_point_indices(
    radii: Sequence[object],
    *,
    point_count: Optional[int] = None,
    neighbor_window: int = BOTTLENECK_NEIGHBOR_POINT_WINDOW,
) -> set[int]:
    """Return the bottleneck point and neighboring profile points used by the charts."""
    if radii is None:
        return set()
    numeric_radii: list[tuple[int, float]] = []
    for idx, value in enumerate(radii):
        try:
            numeric_radii.append((idx, float(value)))
        except (TypeError, ValueError):
            continue
    if not numeric_radii:
        return set()

    min_radius_idx = min(numeric_radii, key=lambda item: item[1])[0]
    max_index = len(radii) - 1
    if point_count is not None and int(point_count) > 0:
        max_index = min(max_index, int(point_count) - 1)
    if max_index < 0:
        return set()

    window = max(0, int(neighbor_window))
    start = max(0, min_radius_idx - window)
    end = min(max_index, min_radius_idx + window)
    return set(range(start, end + 1))


def parse_residue_ids(raw_value) -> Tuple[int, ...]:
    text = str(raw_value or "").strip()
    if not text:
        return ()
    result = []
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        result.append(_safe_int(item))
    return tuple(result)


def parse_residue_names(raw_value) -> Tuple[str, ...]:
    text = str(raw_value or "").strip()
    if not text:
        return ()
    return tuple(item.strip() for item in text.split(";") if item.strip())


def load_residue_combination_statistics_csv(path: str) -> dict:
    abs_path = os.path.abspath(path)
    rows: List[dict] = []
    lookup: Dict[Tuple[int, ...], dict] = {}
    total_paths = 0
    with open(abs_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            residue_ids = parse_residue_ids(raw_row.get("Residue_IDs"))
            if len(residue_ids) < 2:
                continue
            residue_names = parse_residue_names(raw_row.get("Residue_names"))
            row = {
                "combination_size": _safe_int(raw_row.get("Combination_size"), len(residue_ids)),
                "residue_ids": residue_ids,
                "residue_names": residue_names,
                "combination_label": ";".join(str(rid) for rid in residue_ids),
                "affected_point_count": _safe_int(raw_row.get("Affected_point_count")),
                "affected_path_count": _safe_int(raw_row.get("Affected_path_count")),
                "affected_point_radius_avg": _safe_float(raw_row.get("Affected_point_radius_avg")),
                "affected_point_radius_min": _safe_float(raw_row.get("Affected_point_radius_min")),
                "affected_point_radius_max": _safe_float(raw_row.get("Affected_point_radius_max")),
                "affected_point_radius_range": _safe_float(raw_row.get("Affected_point_radius_range")),
            }
            rows.append(row)
            lookup[residue_ids] = row
            total_paths = max(total_paths, int(row["affected_path_count"]))
    return {
        "path": abs_path,
        "rows": rows,
        "lookup": lookup,
        "row_count": len(rows),
        "total_paths": total_paths,
    }


def related_combinations_for_residues(combination_data: dict, residue_ids: Sequence[int]) -> List[dict]:
    cleaned = []
    seen = set()
    for residue_id in residue_ids:
        residue_id = int(residue_id)
        if residue_id <= 0 or residue_id in seen:
            continue
        cleaned.append(residue_id)
        seen.add(residue_id)

    if len(cleaned) < 2:
        return []

    lookup = combination_data.get("lookup", {})
    rows: List[dict] = []
    for size in range(2, min(4, len(cleaned)) + 1):
        for subset in combinations(cleaned, size):
            row = lookup.get(tuple(subset))
            if row is not None:
                payload = dict(row)
                payload["is_exact_match"] = len(subset) == len(cleaned)
                rows.append(payload)
    rows.sort(
        key=lambda row: (
            int(row.get("combination_size", 0)),
            -int(row.get("affected_path_count", 0)),
            tuple(row.get("residue_ids", ())),
        )
    )
    return rows


def all_combination_rows(combination_data: dict) -> List[dict]:
    rows = [dict(row) for row in combination_data.get("rows", [])]
    rows.sort(
        key=lambda row: (
            int(row.get("combination_size", 0)),
            -int(row.get("affected_path_count", 0)),
            tuple(row.get("residue_ids", ())),
        )
    )
    return rows


def _row_lookup(rows: Sequence[dict]) -> Dict[Tuple[int, ...], dict]:
    return {
        tuple(int(item) for item in row.get("residue_ids", ())): dict(row)
        for row in rows or ()
    }


def filter_combination_rows_by_keys(
    rows: Sequence[dict],
    combo_keys: Sequence[Sequence[int]] | set[Tuple[int, ...]],
    *,
    active: bool = True,
    rows_already_scoped: bool = False,
) -> List[dict]:
    if rows_already_scoped or not active:
        return [dict(row) for row in rows or ()]
    normalized_keys = {
        tuple(int(item) for item in combo)
        for combo in (combo_keys or ())
        if len(combo) >= 2
    }
    if not normalized_keys:
        return [dict(row) for row in rows or ()]
    return [
        dict(row)
        for row in rows or ()
        if tuple(int(item) for item in row.get("residue_ids", ())) in normalized_keys
    ]


def compare_combination_row_sets(
    left_rows: Sequence[dict],
    right_rows: Sequence[dict],
    *,
    include_right_only: bool = True,
    left_total_paths: float = 0.0,
    right_total_paths: float = 0.0,
) -> List[dict]:
    left_lookup = _row_lookup(left_rows)
    right_lookup = _row_lookup(right_rows)

    combo_key_set = set(left_lookup.keys())
    if include_right_only:
        combo_key_set |= set(right_lookup.keys())
    combo_keys = sorted(combo_key_set, key=lambda item: (len(item), item))
    result: List[dict] = []
    for combo_key in combo_keys:
        left_row = left_lookup.get(combo_key)
        right_row = right_lookup.get(combo_key)
        template = right_row or left_row or {}
        metrics = {}
        max_abs_delta = 0.0
        for metric in (
            "affected_path_count",
            "affected_point_radius_avg",
        ):
            left_value = float(left_row.get(metric, 0.0)) if left_row else 0.0
            right_value = float(right_row.get(metric, 0.0)) if right_row else 0.0
            delta = right_value - left_value
            metrics[metric] = {
                "left": left_value,
                "right": right_value,
                "delta": delta,
                "abs_delta": abs(delta),
            }
            max_abs_delta = max(max_abs_delta, abs(delta))

        left_path_fraction = (
            (float(left_row.get("affected_path_count", 0.0)) / float(left_total_paths))
            if left_row and float(left_total_paths) > 0
            else 0.0
        )
        right_path_fraction = (
            (float(right_row.get("affected_path_count", 0.0)) / float(right_total_paths))
            if right_row and float(right_total_paths) > 0
            else 0.0
        )
        path_fraction_delta = right_path_fraction - left_path_fraction
        metrics["affected_path_fraction"] = {
            "left": left_path_fraction,
            "right": right_path_fraction,
            "delta": path_fraction_delta,
            "abs_delta": abs(path_fraction_delta),
            "left_total_paths": float(left_total_paths),
            "right_total_paths": float(right_total_paths),
        }
        max_abs_delta = max(max_abs_delta, abs(path_fraction_delta))

        left_weighted_radius = (
            float(left_row.get("affected_point_radius_avg", 0.0)) * float(left_row.get("affected_path_count", 0.0))
            if left_row else 0.0
        )
        right_weighted_radius = (
            float(right_row.get("affected_point_radius_avg", 0.0)) * float(right_row.get("affected_path_count", 0.0))
            if right_row else 0.0
        )
        left_path_count = float(left_row.get("affected_path_count", 0.0)) if left_row else 0.0
        right_path_count = float(right_row.get("affected_path_count", 0.0)) if right_row else 0.0
        weighted_delta = ((right_weighted_radius - left_weighted_radius) / right_path_count) if right_path_count > 0 else 0.0
        metrics["weighted_radius_score"] = {
            "left": left_weighted_radius,
            "right": right_weighted_radius,
            "delta": weighted_delta,
            "abs_delta": abs(weighted_delta),
            "left_path_count": left_path_count,
            "right_path_count": right_path_count,
        }
        max_abs_delta = max(max_abs_delta, abs(weighted_delta))

        result.append(
            {
                "residue_ids": combo_key,
                "residue_names": tuple(template.get("residue_names", ())),
                "combination_size": len(combo_key),
                "combination_label": template.get("combination_label", ";".join(str(item) for item in combo_key)),
                "present_left": left_row is not None,
                "present_right": right_row is not None,
                "is_exact_match": bool(template.get("is_exact_match")),
                "metrics": metrics,
                "changed": max_abs_delta > 1e-12,
                "max_abs_delta": max_abs_delta,
            }
        )
    return result


def compare_combination_statistics(
    left_data: dict,
    right_data: dict,
    residue_ids: Optional[Sequence[int]] = None,
) -> List[dict]:
    if residue_ids:
        left_source_rows = related_combinations_for_residues(left_data, residue_ids)
        right_source_rows = related_combinations_for_residues(right_data, residue_ids)
    else:
        left_source_rows = all_combination_rows(left_data)
        right_source_rows = all_combination_rows(right_data)

    return compare_combination_row_sets(
        left_source_rows,
        right_source_rows,
        left_total_paths=float(left_data.get("total_paths", 0.0) or 0.0),
        right_total_paths=float(right_data.get("total_paths", 0.0) or 0.0),
    )


def profile_residue_set(profile: dict) -> set[int]:
    residue_ids: set[int] = set()
    for key in ("res_1", "res_2", "res_3", "res_4"):
        for value in profile.get(key, []):
            residue_id = int(value or 0)
            if residue_id > 0:
                residue_ids.add(residue_id)
    return residue_ids


def residue_set_from_profiles(profiles: Sequence[dict]) -> set[int]:
    residue_ids: set[int] = set()
    for profile in profiles or ():
        residue_ids.update(profile_residue_set(profile))
    return residue_ids


def filter_profiles_by_residue_subset(
    profiles: Sequence[dict],
    allowed_residue_ids: Sequence[int],
) -> List[dict]:
    allowed = {int(item) for item in allowed_residue_ids if int(item) > 0}
    if not allowed:
        return []
    matched: List[dict] = []
    for profile in profiles or ():
        residue_ids = profile_residue_set(profile)
        if residue_ids and residue_ids.issubset(allowed):
            matched.append(dict(profile))
    return matched


def combination_rows_from_profiles(
    profiles: Sequence[dict],
    *,
    max_size: int = 4,
    bottleneck_only: bool = False,
) -> List[dict]:
    aggregates: Dict[Tuple[int, ...], dict] = {}
    for profile in profiles or ():
        path_id = int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
        residue_columns = (
            profile.get("res_1", []),
            profile.get("res_2", []),
            profile.get("res_3", []),
            profile.get("res_4", []),
        )
        radii = profile.get("radius", [])
        point_count = max((len(column) for column in residue_columns), default=0)
        if not point_count:
            continue
        allowed_indices: Optional[set[int]] = None
        if bottleneck_only and len(radii) > 0:
            allowed_indices = bottleneck_point_indices(radii, point_count=point_count)
        for idx in range(point_count):
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            residue_ids = sorted(
                {
                    int(column[idx])
                    for column in residue_columns
                    if idx < len(column) and int(column[idx]) > 0
                }
            )
            if len(residue_ids) < 2:
                continue
            radius = _safe_float(radii[idx]) if idx < len(radii) else 0.0
            for size in range(2, min(max_size, len(residue_ids)) + 1):
                for combo in combinations(residue_ids, size):
                    entry = aggregates.setdefault(
                        tuple(int(item) for item in combo),
                        {
                            "combination_size": len(combo),
                            "residue_ids": tuple(int(item) for item in combo),
                            "residue_names": (),
                            "combination_label": ";".join(str(item) for item in combo),
                            "affected_point_count": 0,
                            "path_ids": set(),
                            "radius_sum": 0.0,
                            "radius_min": None,
                            "radius_max": None,
                        },
                    )
                    entry["affected_point_count"] += 1
                    entry["path_ids"].add(path_id)
                    entry["radius_sum"] += float(radius)
                    if entry["radius_min"] is None or float(radius) < float(entry["radius_min"]):
                        entry["radius_min"] = float(radius)
                    if entry["radius_max"] is None or float(radius) > float(entry["radius_max"]):
                        entry["radius_max"] = float(radius)

    rows: List[dict] = []
    for combo_key, entry in aggregates.items():
        point_count = int(entry["affected_point_count"])
        radius_min = float(entry["radius_min"] if entry["radius_min"] is not None else 0.0)
        radius_max = float(entry["radius_max"] if entry["radius_max"] is not None else 0.0)
        rows.append(
            {
                "combination_size": int(entry["combination_size"]),
                "residue_ids": combo_key,
                "residue_names": tuple(entry.get("residue_names", ())),
                "combination_label": str(entry["combination_label"]),
                "affected_point_count": point_count,
                "affected_path_count": len(entry["path_ids"]),
                "affected_point_radius_avg": (float(entry["radius_sum"]) / point_count) if point_count else 0.0,
                "affected_point_radius_min": radius_min,
                "affected_point_radius_max": radius_max,
                "affected_point_radius_range": radius_max - radius_min,
            }
        )
    rows.sort(
        key=lambda row: (
            int(row.get("combination_size", 0)),
            -int(row.get("affected_path_count", 0)),
            tuple(row.get("residue_ids", ())),
        )
    )
    return rows


def exit_region_from_path_coords(
    coords_by_path_id: Dict[int, Sequence[Sequence[float]]],
) -> Optional[dict]:
    exit_points: List[Tuple[float, float, float]] = []
    for coords in (coords_by_path_id or {}).values():
        if coords is None or len(coords) == 0:
            continue
        point = coords[-1]
        exit_points.append(
            (_safe_float(point[0]), _safe_float(point[1]), _safe_float(point[2]))
        )
    if not exit_points:
        return None
    count = float(len(exit_points))
    center = (
        sum(point[0] for point in exit_points) / count,
        sum(point[1] for point in exit_points) / count,
        sum(point[2] for point in exit_points) / count,
    )
    radius = 0.0
    for point in exit_points:
        distance = math.sqrt(
            (point[0] - center[0]) ** 2 +
            (point[1] - center[1]) ** 2 +
            (point[2] - center[2]) ** 2
        )
        radius = max(radius, distance)
    return {
        "center": center,
        "radius": radius,
        "count": len(exit_points),
    }


def trimmed_exit_region_from_path_coords(
    coords_by_path_id: Dict[int, Sequence[Sequence[float]]],
    keep_ratio: float = 1.0,
) -> Optional[dict]:
    exit_points: List[Tuple[int, Tuple[float, float, float]]] = []
    for path_id, coords in (coords_by_path_id or {}).items():
        if coords is None or len(coords) == 0:
            continue
        point = coords[-1]
        exit_points.append(
            (
                int(path_id),
                (_safe_float(point[0]), _safe_float(point[1]), _safe_float(point[2])),
            )
        )
    if not exit_points:
        return None
    keep_ratio = min(1.0, max(0.0, float(keep_ratio)))
    if keep_ratio <= 0.0:
        keep_ratio = 1.0
    count = float(len(exit_points))
    center = (
        sum(point[1][0] for point in exit_points) / count,
        sum(point[1][1] for point in exit_points) / count,
        sum(point[1][2] for point in exit_points) / count,
    )
    ranked = []
    for path_id, point in exit_points:
        distance = math.sqrt(
            (point[0] - center[0]) ** 2 +
            (point[1] - center[1]) ** 2 +
            (point[2] - center[2]) ** 2
        )
        ranked.append((distance, path_id, point))
    ranked.sort(key=lambda item: item[0])
    keep_count = max(1, int(math.ceil(len(ranked) * keep_ratio)))
    kept = ranked[:keep_count]
    kept_count = float(len(kept))
    trimmed_center = (
        sum(item[2][0] for item in kept) / kept_count,
        sum(item[2][1] for item in kept) / kept_count,
        sum(item[2][2] for item in kept) / kept_count,
    )
    trimmed_radius = 0.0
    for _, _, point in kept:
        distance = math.sqrt(
            (point[0] - trimmed_center[0]) ** 2 +
            (point[1] - trimmed_center[1]) ** 2 +
            (point[2] - trimmed_center[2]) ** 2
        )
        trimmed_radius = max(trimmed_radius, distance)
    return {
        "center": trimmed_center,
        "radius": trimmed_radius,
        "count": len(kept),
        "original_count": len(exit_points),
        "keep_ratio": keep_ratio,
    }


def filter_path_ids_by_exit_region(
    coords_by_path_id: Dict[int, Sequence[Sequence[float]]],
    region: Optional[dict],
    *,
    tolerance: float = 1e-6,
) -> List[int]:
    if not region:
        return []
    center = region.get("center") or (0.0, 0.0, 0.0)
    radius = max(0.0, float(region.get("radius", 0.0) or 0.0)) + float(tolerance)
    matched: List[int] = []
    for path_id, coords in (coords_by_path_id or {}).items():
        if coords is None or len(coords) == 0:
            continue
        point = coords[-1]
        distance = math.sqrt(
            (_safe_float(point[0]) - _safe_float(center[0])) ** 2 +
            (_safe_float(point[1]) - _safe_float(center[1])) ** 2 +
            (_safe_float(point[2]) - _safe_float(center[2])) ** 2
        )
        if distance <= radius:
            matched.append(int(path_id))
    matched.sort()
    return matched


def scalar_region(values_by_id: Dict[int, float]) -> Optional[dict]:
    values = [float(value) for value in (values_by_id or {}).values()]
    if not values:
        return None
    center = sum(values) / float(len(values))
    radius = max(abs(value - center) for value in values) if values else 0.0
    return {
        "center": center,
        "radius": radius,
        "count": len(values),
    }


def trimmed_length_range(values_by_id: Dict[int, float], keep_ratio: float = 1.0) -> Optional[dict]:
    pairs = sorted((int(item_id), float(value)) for item_id, value in (values_by_id or {}).items())
    if not pairs:
        return None
    keep_ratio = min(1.0, max(0.0, float(keep_ratio)))
    if keep_ratio <= 0.0:
        keep_ratio = 1.0
    values_only = sorted(value for _, value in pairs)
    count = len(values_only)
    trim_each = int(math.floor(count * max(0.0, 1.0 - keep_ratio) / 2.0))
    start = min(trim_each, max(0, count - 1))
    end = max(start + 1, count - trim_each)
    trimmed = values_only[start:end]
    return {
        "min": min(trimmed),
        "max": max(trimmed),
        "count": len(trimmed),
        "original_count": count,
        "keep_ratio": keep_ratio,
    }


def filter_ids_by_scalar_region(
    values_by_id: Dict[int, float],
    region: Optional[dict],
    *,
    tolerance: float = 1e-6,
) -> List[int]:
    if not region:
        return []
    center = float(region.get("center", 0.0) or 0.0)
    radius = max(0.0, float(region.get("radius", 0.0) or 0.0)) + float(tolerance)
    matched = [
        int(item_id)
        for item_id, value in (values_by_id or {}).items()
        if abs(float(value) - center) <= radius
    ]
    matched.sort()
    return matched


def combination_property_value(row: dict, property_key: str, *, compare_mode: bool) -> float:
    property_key = str(property_key or "affected_path_count")
    if compare_mode:
        return float(row.get("metrics", {}).get(property_key, {}).get("delta", 0.0) or 0.0)
    if property_key == "weighted_radius_score":
        return float(row.get("affected_point_radius_avg", 0.0) or 0.0) * float(row.get("affected_path_count", 0.0) or 0.0)
    return float(row.get(property_key, 0.0) or 0.0)


def combination_property_trend(row: dict, property_key: str, *, compare_mode: bool) -> str:
    if not compare_mode:
        return "neutral"
    value = combination_property_value(row, property_key, compare_mode=True)
    if value > 1e-12:
        return "increase"
    if value < -1e-12:
        return "decrease"
    return "neutral"


def combination_centroid(
    residue_ids: Sequence[int],
    residue_positions: Dict[int, Sequence[float]],
) -> Optional[Tuple[float, float, float]]:
    points = []
    for residue_id in residue_ids:
        point = residue_positions.get(int(residue_id))
        if point is None:
            return None
        points.append((_safe_float(point[0]), _safe_float(point[1]), _safe_float(point[2])))
    if not points:
        return None
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def combination_pair_distance_sum(
    residue_ids: Sequence[int],
    residue_positions: Dict[int, Sequence[float]],
    residue_atom_positions: Optional[Dict[int, Sequence[Sequence[float]]]] = None,
) -> Optional[float]:
    points = []
    for residue_id in residue_ids:
        point = residue_positions.get(int(residue_id))
        if point is None:
            return None
        points.append(
            (
                _safe_float(point[0]),
                _safe_float(point[1]),
                _safe_float(point[2]),
            )
        )
    if len(points) < 2:
        return 0.0
    total = 0.0
    for idx in range(len(residue_ids)):
        rid_a = int(residue_ids[idx])
        for jdx in range(idx + 1, len(residue_ids)):
            rid_b = int(residue_ids[jdx])
            min_atom_distance: Optional[float] = None
            if residue_atom_positions:
                atoms_a = residue_atom_positions.get(rid_a) or ()
                atoms_b = residue_atom_positions.get(rid_b) or ()
                if atoms_a and atoms_b:
                    for atom_a in atoms_a:
                        ax = _safe_float(atom_a[0])
                        ay = _safe_float(atom_a[1])
                        az = _safe_float(atom_a[2])
                        for atom_b in atoms_b:
                            distance = math.sqrt(
                                (ax - _safe_float(atom_b[0])) ** 2
                                + (ay - _safe_float(atom_b[1])) ** 2
                                + (az - _safe_float(atom_b[2])) ** 2
                            )
                            if min_atom_distance is None or distance < min_atom_distance:
                                min_atom_distance = distance
            if min_atom_distance is not None:
                total += min_atom_distance
                continue
            total += math.sqrt(
                (points[idx][0] - points[jdx][0]) ** 2
                + (points[idx][1] - points[jdx][1]) ** 2
                + (points[idx][2] - points[jdx][2]) ** 2
            )
    return total


def combination_keys_from_profiles(
    profiles: Sequence[dict],
    *,
    max_size: int = 4,
    bottleneck_only: bool = False,
) -> set[Tuple[int, ...]]:
    result: set[Tuple[int, ...]] = set()
    for profile in profiles or ():
        residue_columns = (
            profile.get("res_1", []),
            profile.get("res_2", []),
            profile.get("res_3", []),
            profile.get("res_4", []),
        )
        radii = profile.get("radius", [])
        point_count = max((len(column) for column in residue_columns), default=0)
        allowed_indices: Optional[set[int]] = None
        if bottleneck_only and len(radii) > 0:
            allowed_indices = bottleneck_point_indices(radii, point_count=point_count)
        for idx in range(point_count):
            if allowed_indices is not None and idx not in allowed_indices:
                continue
            residue_ids = sorted(
                {
                    int(column[idx])
                    for column in residue_columns
                    if idx < len(column) and int(column[idx]) > 0
                }
            )
            for size in range(2, min(max_size, len(residue_ids)) + 1):
                for combo in combinations(residue_ids, size):
                    result.add(tuple(int(item) for item in combo))
    return result
