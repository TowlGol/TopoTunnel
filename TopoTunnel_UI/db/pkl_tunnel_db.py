"""
Direct preprocessed_paths.pkl access layer that mirrors the TunnelDatabase API.
"""
from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, str) and not value.strip():
            return int(default)
        if isinstance(value, float) and math.isnan(value):
            return int(default)
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        if isinstance(value, str) and not value.strip():
            return float(default)
        if isinstance(value, float) and math.isnan(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _as_float_array(values: Any, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    if values is None:
        if fallback is not None:
            return fallback.astype(np.float32, copy=True)
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return np.asarray([float(arr)], dtype=np.float32)
    return arr.reshape(-1).astype(np.float32, copy=False)


def _as_int_array(values: Any, length: int) -> np.ndarray:
    if values is None:
        return np.zeros(length, dtype=np.int32)
    arr = np.asarray(values, dtype=np.int32)
    if arr.ndim == 0:
        return np.full(length, int(arr), dtype=np.int32)
    arr = arr.reshape(-1).astype(np.int32, copy=False)
    if len(arr) == length:
        return arr
    if len(arr) > length:
        return arr[:length]
    padded = np.zeros(length, dtype=np.int32)
    padded[: len(arr)] = arr
    return padded


def _combine_coords(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    count = min(len(xs), len(ys), len(zs))
    if count <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.column_stack((xs[:count], ys[:count], zs[:count])).astype(np.float32, copy=False)


def _resolve_path_cluster_id(path: dict) -> int:
    for key in ("cluster_id", "tunnel_cluster", "Tunnel_id", "Tunnel_cluster"):
        value = path.get(key)
        if value is None:
            continue
        resolved = _safe_int(value, default=0)
        if resolved or value == 0:
            return resolved
    return 0


def _extract_frame_id(path: dict) -> int:
    frame_ids = path.get("frame_ids")
    if frame_ids is not None:
        arr = np.asarray(frame_ids).reshape(-1)
        if arr.size:
            return _safe_int(arr[0], default=0)
    for key in ("frame_id", "frame", "Frame"):
        if key in path:
            return _safe_int(path.get(key), default=0)
    return 0


def _residue_set_from_path(path: dict, res_arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> frozenset[int]:
    raw = path.get("residues_set")
    if raw:
        return frozenset(sorted(_safe_int(value) for value in raw if _safe_int(value) > 0))
    raw = path.get("residues")
    if raw:
        return frozenset(sorted(_safe_int(value) for value in raw if _safe_int(value) > 0))
    residues: set[int] = set()
    for arr in res_arrays:
        if len(arr):
            residues.update(int(value) for value in arr if int(value) > 0)
    return frozenset(sorted(residues))


def _compute_length_values(coords: np.ndarray) -> np.ndarray:
    if len(coords) <= 1:
        return np.zeros(len(coords), dtype=np.float32)
    deltas = np.diff(coords.astype(np.float64), axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths, dtype=np.float64)))
    return cumulative.astype(np.float32, copy=False)


@dataclass(slots=True)
class _PathRecord:
    path_id: int
    frame_id: int
    cluster_id: int
    coords: np.ndarray
    original_coords: np.ndarray
    selection_coords: np.ndarray
    radius: np.ndarray
    hydrophobicity: np.ndarray
    length_vals: np.ndarray
    throughput: np.ndarray
    residues: frozenset[int]
    res_1: np.ndarray
    res_2: np.ndarray
    res_3: np.ndarray
    res_4: np.ndarray
    atom_1: np.ndarray
    atom_2: np.ndarray
    atom_3: np.ndarray
    atom_4: np.ndarray
    meta: dict


class PklTunnelDatabase:
    """Read-only backend that serves preprocessed_paths.pkl directly."""

    def __init__(self, pkl_path: str):
        self.db_path = os.path.abspath(pkl_path)
        self._conn = None
        self._paths_by_id: Dict[int, _PathRecord] = {}
        self._sorted_path_ids: List[int] = []
        self._frame_to_path_ids: Dict[int, List[int]] = {}
        self._cluster_to_path_ids: Dict[int, List[int]] = {}
        self._residue_to_paths: Dict[int, Set[int]] = {}
        self._residue_positions: Dict[int, dict] = {}
        self._metadata: Dict[str, str] = {}
        self._sessions: Dict[str, dict] = {}

    def connect(self):
        if self._conn is not None:
            return
        with open(self.db_path, "rb") as handle:
            data = pickle.load(handle)
        self._load_data(data)
        self._conn = True

    def close(self):
        self._conn = None
        self._paths_by_id.clear()
        self._sorted_path_ids.clear()
        self._frame_to_path_ids.clear()
        self._cluster_to_path_ids.clear()
        self._residue_to_paths.clear()
        self._residue_positions.clear()
        self._metadata.clear()
        self._sessions.clear()

    def _require_loaded(self):
        if self._conn is None:
            self.connect()

    def _load_data(self, data: Any):
        payload = data if isinstance(data, dict) else {}
        raw_paths = payload.get("PATHS", [])
        raw_initial_counts = payload.get("INITIAL_COUNTS", {}) or {}
        raw_residue_positions = payload.get("RESIDUE_POSITIONS", {}) or {}
        raw_frame_range = payload.get("FRAME_RANGE") or {}
        paths_by_id: Dict[int, _PathRecord] = {}
        frame_ids_seen: List[int] = []
        residue_accumulator: Dict[int, list[np.ndarray]] = {}

        for path_index, raw_path in enumerate(raw_paths):
            if not isinstance(raw_path, dict):
                continue
            xs = _as_float_array(raw_path.get("xs"))
            ys = _as_float_array(raw_path.get("ys"))
            zs = _as_float_array(raw_path.get("zs"))
            coords = _combine_coords(xs, ys, zs)
            point_count = len(coords)
            if point_count < 2:
                continue

            xo = _as_float_array(raw_path.get("xs_origin"), fallback=xs)
            yo = _as_float_array(raw_path.get("ys_origin"), fallback=ys)
            zo = _as_float_array(raw_path.get("zs_origin"), fallback=zs)
            original_coords = _combine_coords(xo, yo, zo)
            if len(original_coords) != point_count:
                original_coords = coords.copy()

            xs_selection = _as_float_array(raw_path.get("xs_selection"), fallback=xs)
            ys_selection = _as_float_array(raw_path.get("ys_selection"), fallback=ys)
            zs_selection = _as_float_array(raw_path.get("zs_selection"), fallback=zs)
            selection_coords = _combine_coords(xs_selection, ys_selection, zs_selection)
            if len(selection_coords) == 0:
                selection_coords = coords.copy()

            radius = _as_float_array(raw_path.get("bf"))
            if len(radius) != point_count:
                radius = np.pad(radius[:point_count], (0, max(0, point_count - len(radius))), constant_values=0.0)
            hydrophobicity = _as_float_array(raw_path.get("qss"))
            if len(hydrophobicity) != point_count:
                hydrophobicity = np.pad(
                    hydrophobicity[:point_count],
                    (0, max(0, point_count - len(hydrophobicity))),
                    constant_values=0.0,
                )
            length_vals = _as_float_array(raw_path.get("length"))
            if len(length_vals) != point_count:
                length_vals = _compute_length_values(coords)
            throughput = _as_float_array(raw_path.get("throughput"))
            if len(throughput) != point_count:
                throughput = np.pad(
                    throughput[:point_count],
                    (0, max(0, point_count - len(throughput))),
                    constant_values=0.0,
                )

            res_arrays = (
                _as_int_array(raw_path.get("res_1"), point_count),
                _as_int_array(raw_path.get("res_2"), point_count),
                _as_int_array(raw_path.get("res_3"), point_count),
                _as_int_array(raw_path.get("res_4"), point_count),
            )
            atom_arrays = (
                _as_int_array(raw_path.get("atom_1"), point_count),
                _as_int_array(raw_path.get("atom_2"), point_count),
                _as_int_array(raw_path.get("atom_3"), point_count),
                _as_int_array(raw_path.get("atom_4"), point_count),
            )
            residues = _residue_set_from_path(raw_path, res_arrays)
            frame_id = _extract_frame_id(raw_path)
            cluster_id = _resolve_path_cluster_id(raw_path)
            bbox = raw_path.get("bbox") or {}
            bbox_min = bbox.get("min") if isinstance(bbox, dict) else None
            bbox_max = bbox.get("max") if isinstance(bbox, dict) else None
            if bbox_min is None or bbox_max is None:
                bbox_min = selection_coords.min(axis=0).tolist()
                bbox_max = selection_coords.max(axis=0).tolist()

            meta = {
                "id": int(path_index),
                "frame_id": int(frame_id),
                "cluster_id": int(cluster_id),
                "point_count": int(point_count),
                "path_length": float(length_vals[-1] if len(length_vals) else 0.0),
                "min_radius": float(np.min(radius) if len(radius) else 0.0),
                "avg_radius": float(np.mean(radius) if len(radius) else 0.0),
                "avg_hydrophobicity": float(np.mean(hydrophobicity) if len(hydrophobicity) else 0.0),
                "residues_text": ",".join(str(value) for value in sorted(residues)),
                "residue_count": int(len(residues)),
                "start_x": float(coords[0][0]),
                "start_y": float(coords[0][1]),
                "start_z": float(coords[0][2]),
                "end_x": float(coords[-1][0]),
                "end_y": float(coords[-1][1]),
                "end_z": float(coords[-1][2]),
                "bbox_min_x": float(bbox_min[0]),
                "bbox_min_y": float(bbox_min[1]),
                "bbox_min_z": float(bbox_min[2]),
                "bbox_max_x": float(bbox_max[0]),
                "bbox_max_y": float(bbox_max[1]),
                "bbox_max_z": float(bbox_max[2]),
            }
            record = _PathRecord(
                path_id=int(path_index),
                frame_id=int(frame_id),
                cluster_id=int(cluster_id),
                coords=coords,
                original_coords=original_coords,
                selection_coords=selection_coords,
                radius=radius.astype(np.float32, copy=False),
                hydrophobicity=hydrophobicity.astype(np.float32, copy=False),
                length_vals=length_vals.astype(np.float32, copy=False),
                throughput=throughput.astype(np.float32, copy=False),
                residues=residues,
                res_1=res_arrays[0],
                res_2=res_arrays[1],
                res_3=res_arrays[2],
                res_4=res_arrays[3],
                atom_1=atom_arrays[0],
                atom_2=atom_arrays[1],
                atom_3=atom_arrays[2],
                atom_4=atom_arrays[3],
                meta=meta,
            )
            paths_by_id[record.path_id] = record
            self._frame_to_path_ids.setdefault(record.frame_id, []).append(record.path_id)
            self._cluster_to_path_ids.setdefault(record.cluster_id, []).append(record.path_id)
            for residue_id in residues:
                self._residue_to_paths.setdefault(int(residue_id), set()).add(record.path_id)
            for residue_array in res_arrays:
                positive_mask = residue_array > 0
                if not np.any(positive_mask):
                    continue
                ids = residue_array[positive_mask]
                coords_for_residue = original_coords[positive_mask]
                for residue_id in np.unique(ids):
                    residue_accumulator.setdefault(int(residue_id), []).append(
                        coords_for_residue[ids == residue_id]
                    )
            frame_ids_seen.append(record.frame_id)

        self._paths_by_id = paths_by_id
        self._sorted_path_ids = sorted(paths_by_id)
        for values in self._frame_to_path_ids.values():
            values.sort()
        for values in self._cluster_to_path_ids.values():
            values.sort()

        residue_positions: Dict[int, dict] = {}
        for residue_id, payload in raw_residue_positions.items():
            local_id = _safe_int(residue_id, default=0)
            if local_id <= 0:
                continue
            if isinstance(payload, dict):
                residue_positions[local_id] = {
                    "residue_id": int(local_id),
                    "x": _safe_float(payload.get("x"), 0.0),
                    "y": _safe_float(payload.get("y"), 0.0),
                    "z": _safe_float(payload.get("z"), 0.0),
                    "path_count": int(_safe_int(raw_initial_counts.get(residue_id, raw_initial_counts.get(local_id, 0)), 0)),
                }
        if not residue_positions:
            for residue_id, arrays in residue_accumulator.items():
                if not arrays:
                    continue
                stacked = np.concatenate(arrays, axis=0)
                residue_positions[residue_id] = {
                    "residue_id": int(residue_id),
                    "x": float(np.mean(stacked[:, 0])) if len(stacked) else 0.0,
                    "y": float(np.mean(stacked[:, 1])) if len(stacked) else 0.0,
                    "z": float(np.mean(stacked[:, 2])) if len(stacked) else 0.0,
                    "path_count": int(len(self._residue_to_paths.get(residue_id, ()))),
                }
        else:
            for residue_id, row in residue_positions.items():
                if int(row.get("path_count", 0) or 0) <= 0:
                    row["path_count"] = int(len(self._residue_to_paths.get(residue_id, ())))
        self._residue_positions = residue_positions

        frame_min = _safe_int(raw_frame_range.get("min"), min(frame_ids_seen) if frame_ids_seen else 0)
        frame_max = _safe_int(raw_frame_range.get("max"), max(frame_ids_seen) if frame_ids_seen else 0)
        self._metadata = {
            "frame_min": str(frame_min),
            "frame_max": str(frame_max),
            "path_count": str(len(self._paths_by_id)),
            "residue_count": str(len(self._residue_positions)),
            "source_file": self.db_path,
        }

    def get_meta(self, key: str, default: str = "") -> str:
        self._require_loaded()
        return self._metadata.get(key, default)

    def set_meta(self, key: str, value: str):
        self._require_loaded()
        self._metadata[str(key)] = str(value)

    def path_count(self) -> int:
        self._require_loaded()
        return len(self._paths_by_id)

    def frame_range(self) -> Tuple[int, int]:
        self._require_loaded()
        return (
            _safe_int(self._metadata.get("frame_min"), 0),
            _safe_int(self._metadata.get("frame_max"), 0),
        )

    def get_all_frame_ids(self) -> List[int]:
        self._require_loaded()
        return sorted(self._frame_to_path_ids)

    def get_frame_ids_for_path_ids(self, path_ids: List[int]) -> List[int]:
        self._require_loaded()
        frame_ids = {
            int(record.frame_id)
            for path_id in path_ids
            for record in [self._paths_by_id.get(int(path_id))]
            if record is not None
        }
        return sorted(frame_ids)

    def get_path_ids(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[int]:
        self._require_loaded()
        return self._filtered_path_ids(frame_min=frame_min, frame_max=frame_max)

    def get_path_ids_for_frames(self, frame_ids: List[int]) -> List[int]:
        self._require_loaded()
        if not frame_ids:
            return []
        allowed = {int(frame_id) for frame_id in frame_ids}
        return [
            int(path_id)
            for path_id in self._sorted_path_ids
            if int(self._paths_by_id[path_id].frame_id) in allowed
        ]

    def _filtered_path_ids(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> List[int]:
        self._require_loaded()
        if path_id_filter is None:
            candidate_ids: Set[int] = set(self._sorted_path_ids)
        else:
            candidate_ids = {int(path_id) for path_id in path_id_filter if int(path_id) in self._paths_by_id}

        if residue_filter:
            residue_sets = [self._residue_to_paths.get(int(rid), set()) for rid in residue_filter]
            if not residue_sets or any(not values for values in residue_sets):
                return []
            candidate_ids &= set.intersection(*[set(values) for values in residue_sets])
            if not candidate_ids:
                return []

        if frame_min is None and frame_max is None:
            return sorted(candidate_ids)

        filtered: List[int] = []
        for path_id in sorted(candidate_ids):
            record = self._paths_by_id.get(path_id)
            if record is None:
                continue
            if frame_min is not None and int(record.frame_id) < int(frame_min):
                continue
            if frame_max is not None and int(record.frame_id) > int(frame_max):
                continue
            filtered.append(path_id)
        return filtered

    def get_paths_page(
        self,
        offset: int = 0,
        limit: int = 200,
        order_by: str = "id",
        order_dir: str = "ASC",
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> List[dict]:
        allowed_cols = {
            "id",
            "frame_id",
            "path_length",
            "min_radius",
            "avg_radius",
            "residue_count",
            "cluster_id",
            "avg_hydrophobicity",
        }
        sort_key = order_by if order_by in allowed_cols else "id"
        reverse = str(order_dir).upper() == "DESC"
        path_ids = self._filtered_path_ids(frame_min, frame_max, residue_filter, path_id_filter)
        rows = [dict(self._paths_by_id[path_id].meta) for path_id in path_ids]
        if sort_key == "id":
            rows.sort(key=lambda row: int(row["id"]), reverse=reverse)
        else:
            rows.sort(
                key=lambda row: (row.get(sort_key) is None, row.get(sort_key), int(row["id"])),
                reverse=reverse,
            )
        return rows[int(offset): int(offset) + int(limit)]

    def count_filtered_paths(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> int:
        return len(self._filtered_path_ids(frame_min, frame_max, residue_filter, path_id_filter))

    def get_matched_path_ids(self, residue_ids: Set[int]) -> List[int]:
        self._require_loaded()
        if not residue_ids:
            return []
        residue_sets = [self._residue_to_paths.get(int(rid), set()) for rid in residue_ids]
        if not residue_sets or any(not values for values in residue_sets):
            return []
        return sorted(set.intersection(*[set(values) for values in residue_sets]))

    def get_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        self._require_loaded()
        return {
            int(path_id): self._paths_by_id[int(path_id)].coords
            for path_id in path_ids
            if int(path_id) in self._paths_by_id
        }

    def get_path_length_map(self, path_ids: List[int]) -> Dict[int, float]:
        self._require_loaded()
        return {
            int(path_id): float(self._paths_by_id[int(path_id)].meta.get("path_length", 0.0))
            for path_id in path_ids
            if int(path_id) in self._paths_by_id
        }

    def get_original_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        self._require_loaded()
        return {
            int(path_id): self._paths_by_id[int(path_id)].original_coords
            for path_id in path_ids
            if int(path_id) in self._paths_by_id
        }

    def get_path_exit_points(self, path_ids: List[int]) -> Dict[int, dict]:
        self._require_loaded()
        result: Dict[int, dict] = {}
        for path_id in path_ids:
            record = self._paths_by_id.get(int(path_id))
            if record is None or len(record.coords) == 0:
                continue
            result[int(path_id)] = {
                "cluster_id": int(record.cluster_id),
                "current_exit": record.coords[-1].astype(np.float64, copy=True),
                "original_exit": record.original_coords[-1].astype(np.float64, copy=True),
            }
        return result

    def get_all_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        self._require_loaded()
        return [
            (int(path_id), int(self._paths_by_id[path_id].frame_id), self._paths_by_id[path_id].coords)
            for path_id in self._filtered_path_ids(frame_min=frame_min, frame_max=frame_max)
        ]

    def get_all_original_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        self._require_loaded()
        return [
            (int(path_id), int(self._paths_by_id[path_id].frame_id), self._paths_by_id[path_id].original_coords)
            for path_id in self._filtered_path_ids(frame_min=frame_min, frame_max=frame_max)
        ]

    def get_selection_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        self._require_loaded()
        return {
            int(path_id): self._paths_by_id[int(path_id)].selection_coords
            for path_id in path_ids
            if int(path_id) in self._paths_by_id
        }

    def get_path_point_rows(
        self,
        path_ids: List[int],
        frame_id: Optional[int] = None,
        *,
        original_coords: bool = True,
    ) -> List[dict]:
        self._require_loaded()
        rows: List[dict] = []
        for path_id in sorted(int(value) for value in path_ids):
            record = self._paths_by_id.get(path_id)
            if record is None:
                continue
            if frame_id is not None and int(record.frame_id) != int(frame_id):
                continue
            coords = record.original_coords if original_coords else record.coords
            for seq in range(len(coords)):
                rows.append(
                    {
                        "path_id": int(path_id),
                        "seq": int(seq),
                        "x": float(coords[seq][0]),
                        "y": float(coords[seq][1]),
                        "z": float(coords[seq][2]),
                        "radius": float(record.radius[seq]) if seq < len(record.radius) else 0.0,
                    }
                )
        return rows

    def get_path_profiles(self, path_ids: List[int]) -> List[dict]:
        self._require_loaded()
        profiles: List[dict] = []
        for path_id in path_ids:
            record = self._paths_by_id.get(int(path_id))
            if record is None:
                continue
            profiles.append(
                {
                    "pathIndex": int(record.path_id),
                    "frameId": int(record.frame_id),
                    "cluster_id": int(record.cluster_id),
                    "resSeq": record.length_vals.tolist(),
                    "radius": record.radius.tolist(),
                    "hydrophobicity": record.hydrophobicity.tolist(),
                    "throughput": record.throughput.tolist(),
                    "res_1": record.res_1.tolist(),
                    "res_2": record.res_2.tolist(),
                    "res_3": record.res_3.tolist(),
                    "res_4": record.res_4.tolist(),
                    "atom_1": record.atom_1.tolist(),
                    "atom_2": record.atom_2.tolist(),
                    "atom_3": record.atom_3.tolist(),
                    "atom_4": record.atom_4.tolist(),
                    "numPoints": int(len(record.coords)),
                }
            )
        return profiles

    def get_residue_options(self, selected_residues: Optional[Set[int]] = None) -> List[dict]:
        self._require_loaded()
        if not selected_residues:
            rows = sorted(self._residue_positions.values(), key=lambda row: int(row["residue_id"]))
            return [
                {
                    "label": f"{int(row['residue_id'])} (count={int(row.get('path_count', 0) or 0)})",
                    "value": str(int(row["residue_id"])),
                }
                for row in rows
            ]
        matched_ids = set(self.get_matched_path_ids(selected_residues))
        if not matched_ids:
            return []
        counts: Dict[int, int] = {}
        for path_id in matched_ids:
            record = self._paths_by_id.get(int(path_id))
            if record is None:
                continue
            for residue_id in record.residues:
                counts[int(residue_id)] = counts.get(int(residue_id), 0) + 1
        return [
            {
                "label": f"{residue_id} (count={count})",
                "value": str(residue_id),
            }
            for residue_id, count in sorted(counts.items())
        ]

    def get_residue_positions(self) -> List[dict]:
        self._require_loaded()
        return [dict(row) for _, row in sorted(self._residue_positions.items())]

    def get_cluster_path_groups(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        self._require_loaded()
        if frame_min is None and frame_max is None:
            return {int(cluster_id): list(path_ids) for cluster_id, path_ids in self._cluster_to_path_ids.items()}
        groups: Dict[int, List[int]] = {}
        for path_id in self._filtered_path_ids(frame_min=frame_min, frame_max=frame_max):
            record = self._paths_by_id.get(int(path_id))
            if record is None:
                continue
            groups.setdefault(int(record.cluster_id), []).append(int(path_id))
        return groups

    def get_path_for_export(self, path_id: int) -> Optional[dict]:
        self._require_loaded()
        record = self._paths_by_id.get(int(path_id))
        if record is None:
            return None
        points: List[dict] = []
        for seq in range(len(record.coords)):
            points.append(
                {
                    "path_id": int(record.path_id),
                    "seq": int(seq),
                    "x": float(record.coords[seq][0]),
                    "y": float(record.coords[seq][1]),
                    "z": float(record.coords[seq][2]),
                    "x_origin": float(record.original_coords[seq][0]),
                    "y_origin": float(record.original_coords[seq][1]),
                    "z_origin": float(record.original_coords[seq][2]),
                    "radius": float(record.radius[seq]) if seq < len(record.radius) else 0.0,
                    "hydrophobicity": float(record.hydrophobicity[seq]) if seq < len(record.hydrophobicity) else 0.0,
                    "length_val": float(record.length_vals[seq]) if seq < len(record.length_vals) else 0.0,
                    "throughput": float(record.throughput[seq]) if seq < len(record.throughput) else 0.0,
                    "res_1": int(record.res_1[seq]) if seq < len(record.res_1) else 0,
                    "res_2": int(record.res_2[seq]) if seq < len(record.res_2) else 0,
                    "res_3": int(record.res_3[seq]) if seq < len(record.res_3) else 0,
                    "res_4": int(record.res_4[seq]) if seq < len(record.res_4) else 0,
                    "atom_1": int(record.atom_1[seq]) if seq < len(record.atom_1) else 0,
                    "atom_2": int(record.atom_2[seq]) if seq < len(record.atom_2) else 0,
                    "atom_3": int(record.atom_3[seq]) if seq < len(record.atom_3) else 0,
                    "atom_4": int(record.atom_4[seq]) if seq < len(record.atom_4) else 0,
                }
            )
        return {"meta": dict(record.meta), "points": points}

    def list_sessions(self) -> List[dict]:
        return list(self._sessions.values())

    def create_session(self, session_id: str, name: str, data_json: str = "{}"):
        self._sessions[str(session_id)] = {
            "id": str(session_id),
            "name": str(name),
            "created_at": "",
            "updated_at": "",
            "data_json": str(data_json),
        }

    def load_session(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(str(session_id))
        return dict(session) if session is not None else None

    def save_session(self, session_id: str, name: str, data_json: str):
        self._sessions[str(session_id)] = {
            "id": str(session_id),
            "name": str(name),
            "created_at": "",
            "updated_at": "",
            "data_json": str(data_json),
        }

    def delete_session(self, session_id: str):
        self._sessions.pop(str(session_id), None)
