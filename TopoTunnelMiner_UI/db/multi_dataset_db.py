"""
Multi-dataset aggregation layer that preserves the TunnelDatabase interface.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from TopoTunnelMiner_UI.db.pkl_tunnel_db import PklTunnelDatabase
from TopoTunnelMiner_UI.db.tunnel_db import TunnelDatabase


PATH_ID_BLOCK = 10_000_000
RESIDUE_ID_BLOCK = 1_000_000


@dataclass
class DatasetBinding:
    slot: int
    key: str
    path: str
    folder: str
    name: str
    prefix: str
    db: Union[TunnelDatabase, PklTunnelDatabase]
    residue_statistics_path: str = ""
    residue_combination_statistics_path: str = ""
    md_root_path: str = ""

    @property
    def path_id_base(self) -> int:
        return self.slot * PATH_ID_BLOCK

    @property
    def residue_id_base(self) -> int:
        return self.slot * RESIDUE_ID_BLOCK


class MultiDatasetDatabase:
    """Aggregate multiple datasets behind the original db API."""

    def __init__(self, db_paths: Optional[List[str]] = None):
        self._datasets: List[DatasetBinding] = []
        self._datasets_by_key: Dict[str, DatasetBinding] = {}
        self._path_slot_map: Dict[int, DatasetBinding] = {}
        self._residue_slot_map: Dict[int, DatasetBinding] = {}
        self._next_slot = 1
        self._dataset_counter = 0

        for path in db_paths or []:
            self.add_dataset(path)

    def connect(self):
        for dataset in self._datasets:
            if dataset.db._conn is None:
                dataset.db.connect()

    def close(self):
        for dataset in self._datasets:
            dataset.db.close()

    @property
    def primary_db(self) -> Optional[Union[TunnelDatabase, PklTunnelDatabase]]:
        return self._datasets[0].db if self._datasets else None

    @property
    def primary_dataset_key(self) -> Optional[str]:
        return self._datasets[0].key if self._datasets else None

    def list_datasets(self) -> List[dict]:
        result = []
        for dataset in self._datasets:
            result.append(
                {
                    "key": dataset.key,
                    "name": dataset.name,
                    "path": dataset.path,
                    "folder": dataset.folder,
                    "prefix": dataset.prefix,
                    "path_count": dataset.db.path_count(),
                    "residue_count": int(dataset.db.get_meta("residue_count", "0") or 0),
                    "residue_statistics_path": dataset.residue_statistics_path,
                    "residue_combination_statistics_path": dataset.residue_combination_statistics_path,
                    "md_root_path": dataset.md_root_path,
                }
            )
        return result

    def add_dataset(
        self,
        db_path: str,
        prefix: Optional[str] = None,
        *,
        folder: Optional[str] = None,
        residue_statistics_path: str = "",
        residue_combination_statistics_path: str = "",
        md_root_path: str = "",
    ) -> str:
        abs_path = self._resolve_dataset_path(db_path)
        for dataset in self._datasets:
            if os.path.normcase(dataset.path) == os.path.normcase(abs_path):
                return dataset.key

        if abs_path.lower().endswith(".pkl"):
            db = PklTunnelDatabase(abs_path)
        else:
            db = TunnelDatabase(abs_path)
        db.connect()

        self._dataset_counter += 1
        slot = self._next_slot
        self._next_slot += 1
        key = f"dataset_{self._dataset_counter}"
        name = os.path.basename(abs_path) or abs_path
        unique_prefix = self._make_unique_prefix(prefix or self._default_prefix(name))
        abs_folder = os.path.abspath(folder) if folder else os.path.dirname(abs_path)
        if name.lower() in {"preprocessed_paths.pkl", "tunnel_data.db", "tunnel.db"}:
            folder_name = os.path.basename(abs_folder.rstrip(os.sep))
            if folder_name:
                name = folder_name
        dataset = DatasetBinding(
            slot=slot,
            key=key,
            path=abs_path,
            folder=abs_folder,
            name=name,
            prefix=unique_prefix,
            db=db,
            residue_statistics_path=os.path.abspath(residue_statistics_path) if residue_statistics_path else "",
            residue_combination_statistics_path=(
                os.path.abspath(residue_combination_statistics_path)
                if residue_combination_statistics_path else ""
            ),
            md_root_path=os.path.abspath(md_root_path) if md_root_path else "",
        )
        self._datasets.append(dataset)
        self._datasets_by_key[key] = dataset
        self._path_slot_map[slot] = dataset
        self._residue_slot_map[slot] = dataset
        return key

    @staticmethod
    def _resolve_dataset_path(path: str) -> str:
        abs_path = os.path.abspath(path)
        if not abs_path.lower().endswith(".pkl"):
            return abs_path

        pkl_dir = os.path.dirname(abs_path)
        preferred_db_paths = [
            os.path.join(pkl_dir, "tunnel_data.db"),
            os.path.join(pkl_dir, "tunnel.db"),
            os.path.splitext(abs_path)[0] + ".db",
        ]
        for candidate in preferred_db_paths:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        db_path = os.path.join(pkl_dir, "tunnel_data.db")
        from TopoTunnelMiner_UI.scripts.import_data import import_pkl_to_sqlite
        import_pkl_to_sqlite(abs_path, db_path)
        return os.path.abspath(db_path)

    def remove_dataset(self, key: str) -> bool:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return False
        dataset.db.close()
        self._datasets = [item for item in self._datasets if item.key != key]
        self._datasets_by_key.pop(key, None)
        self._path_slot_map.pop(dataset.slot, None)
        self._residue_slot_map.pop(dataset.slot, None)
        return True

    def update_dataset_prefix(self, key: str, prefix: str) -> bool:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return False
        dataset.prefix = self._make_unique_prefix(prefix, exclude_key=key)
        return True

    def get_dataset(self, key: str) -> Optional[dict]:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return None
        return {
            "key": dataset.key,
            "name": dataset.name,
            "path": dataset.path,
            "folder": dataset.folder,
            "prefix": dataset.prefix,
            "path_count": dataset.db.path_count(),
            "residue_count": int(dataset.db.get_meta("residue_count", "0") or 0),
            "residue_statistics_path": dataset.residue_statistics_path,
            "residue_combination_statistics_path": dataset.residue_combination_statistics_path,
            "md_root_path": dataset.md_root_path,
        }

    def get_dataset_path_ids(
        self,
        key: str,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[int]:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return []
        return [
            self._encode_path_id(dataset, int(local_id))
            for local_id in dataset.db.get_path_ids(frame_min=frame_min, frame_max=frame_max)
        ]

    def get_dataset_cluster_path_groups(
        self,
        key: str,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> Dict[int, Set[int]]:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return {}
        groups: Dict[int, Set[int]] = {}
        for cluster_id, local_ids in dataset.db.get_cluster_path_groups(frame_min, frame_max).items():
            groups[int(cluster_id)] = {
                self._encode_path_id(dataset, int(local_id))
                for local_id in local_ids
            }
        return groups

    def get_dataset_key_for_path_id(self, path_id: int) -> Optional[str]:
        dataset, _ = self._decode_path_id(path_id)
        return dataset.key if dataset is not None else None

    def get_local_path_ids_for_dataset(self, key: str, path_ids: Set[int] | List[int]) -> List[int]:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return []
        local_ids: List[int] = []
        for path_id in path_ids or []:
            bound_dataset, local_id = self._decode_path_id(int(path_id))
            if bound_dataset is not None and bound_dataset.key == key and local_id is not None:
                local_ids.append(int(local_id))
        return local_ids

    def get_local_residue_ids_for_dataset(self, key: str, residue_ids: Set[int] | List[int]) -> List[int]:
        dataset = self._datasets_by_key.get(key)
        if dataset is None:
            return []
        local_ids: List[int] = []
        for residue_id in residue_ids or []:
            bound_dataset, local_id = self._decode_residue_id(int(residue_id))
            if bound_dataset is not None and bound_dataset.key == key and local_id is not None:
                local_ids.append(int(local_id))
        return local_ids

    def format_path_label(self, path_id: int) -> str:
        dataset, local_id = self._decode_path_id(path_id)
        if dataset is None:
            return str(path_id)
        return f"{dataset.prefix}:{local_id}"

    def format_residue_label(self, residue_id: int) -> str:
        dataset, local_id = self._decode_residue_id(residue_id)
        if dataset is None:
            return str(residue_id)
        return f"{dataset.prefix}:{local_id}"

    def get_meta(self, key: str, default: str = "") -> str:
        if key == "residue_count":
            total = sum(int(dataset.db.get_meta("residue_count", "0") or 0) for dataset in self._datasets)
            return str(total)
        if key == "dataset_count":
            return str(len(self._datasets))
        primary = self.primary_db
        return primary.get_meta(key, default) if primary else default

    def path_count(self) -> int:
        return sum(dataset.db.path_count() for dataset in self._datasets)

    def frame_range(self) -> Tuple[int, int]:
        if not self._datasets:
            return 0, 0
        mins: List[int] = []
        maxs: List[int] = []
        for dataset in self._datasets:
            fmin, fmax = dataset.db.frame_range()
            mins.append(int(fmin))
            maxs.append(int(fmax))
        return min(mins), max(maxs)

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
        rows: List[dict] = []
        fetch_limit = max(offset + limit, limit)
        has_residue_filter = bool(residue_filter)
        has_path_filter = path_id_filter is not None

        for dataset in self._datasets:
            local_residue_filter = self._extract_local_residue_filter(dataset, residue_filter)
            local_path_filter = self._extract_local_path_filter(dataset, path_id_filter)
            if has_residue_filter and not local_residue_filter:
                continue
            if has_path_filter and not local_path_filter:
                continue

            local_rows = dataset.db.get_paths_page(
                offset=0,
                limit=fetch_limit,
                order_by=order_by if order_by != "dataset_prefix" else "id",
                order_dir=order_dir,
                frame_min=frame_min,
                frame_max=frame_max,
                residue_filter=local_residue_filter,
                path_id_filter=local_path_filter,
            )
            for row in local_rows:
                rows.append(self._with_dataset_row_fields(dataset, row))

        reverse = order_dir.upper() == "DESC"
        rows.sort(key=lambda row: self._sort_value(row, order_by), reverse=reverse)
        return rows[offset: offset + limit]

    def count_filtered_paths(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> int:
        total = 0
        has_residue_filter = bool(residue_filter)
        has_path_filter = path_id_filter is not None
        for dataset in self._datasets:
            local_residue_filter = self._extract_local_residue_filter(dataset, residue_filter)
            local_path_filter = self._extract_local_path_filter(dataset, path_id_filter)
            if has_residue_filter and not local_residue_filter:
                continue
            if has_path_filter and not local_path_filter:
                continue
            total += dataset.db.count_filtered_paths(
                frame_min=frame_min,
                frame_max=frame_max,
                residue_filter=local_residue_filter,
                path_id_filter=local_path_filter,
            )
        return total

    def get_matched_path_ids(self, residue_ids: Set[int]) -> List[int]:
        matched: List[int] = []
        for dataset in self._datasets:
            local_ids = self._extract_local_residue_filter(dataset, residue_ids)
            if not local_ids:
                continue
            matched.extend(self._encode_path_id(dataset, pid) for pid in dataset.db.get_matched_path_ids(local_ids))
        matched.sort()
        return matched

    def get_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for dataset_key, local_ids in self._group_path_ids(path_ids).items():
            dataset = self._datasets_by_key[dataset_key]
            coords = dataset.db.get_render_coords(local_ids)
            for local_id, arr in coords.items():
                result[self._encode_path_id(dataset, local_id)] = arr
        return result

    def get_original_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for dataset_key, local_ids in self._group_path_ids(path_ids).items():
            dataset = self._datasets_by_key[dataset_key]
            coords = dataset.db.get_original_render_coords(local_ids)
            for local_id, arr in coords.items():
                result[self._encode_path_id(dataset, local_id)] = arr
        return result

    def get_path_exit_points(self, path_ids: List[int]) -> Dict[int, dict]:
        result: Dict[int, dict] = {}
        for dataset_key, local_ids in self._group_path_ids(path_ids).items():
            dataset = self._datasets_by_key[dataset_key]
            endpoints = dataset.db.get_path_exit_points(local_ids)
            for local_id, payload in endpoints.items():
                global_id = self._encode_path_id(dataset, int(local_id))
                item = dict(payload)
                item["dataset_key"] = dataset.key
                item["dataset_prefix"] = dataset.prefix
                item["local_path_id"] = int(local_id)
                result[global_id] = item
        return result

    def get_selection_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for dataset_key, local_ids in self._group_path_ids(path_ids).items():
            dataset = self._datasets_by_key[dataset_key]
            coords = dataset.db.get_selection_coords(local_ids)
            for local_id, arr in coords.items():
                result[self._encode_path_id(dataset, local_id)] = arr
        return result

    def get_all_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        result: List[Tuple[int, int, np.ndarray]] = []
        for dataset in self._datasets:
            for path_id, frame_id, coords in dataset.db.get_all_render_coords(frame_min, frame_max):
                result.append((self._encode_path_id(dataset, path_id), frame_id, coords))
        return result

    def get_all_original_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        result: List[Tuple[int, int, np.ndarray]] = []
        for dataset in self._datasets:
            for path_id, frame_id, coords in dataset.db.get_all_original_render_coords(frame_min, frame_max):
                result.append((self._encode_path_id(dataset, path_id), frame_id, coords))
        return result

    def get_path_profiles(self, path_ids: List[int]) -> List[dict]:
        profiles: List[dict] = []
        for dataset_key, local_ids in self._group_path_ids(path_ids).items():
            dataset = self._datasets_by_key[dataset_key]
            for profile in dataset.db.get_path_profiles(local_ids):
                local_id = int(profile.get("pathIndex", -1))
                profile = dict(profile)
                profile["localPathIndex"] = local_id
                profile["pathIndex"] = self._encode_path_id(dataset, local_id)
                profile["datasetPrefix"] = dataset.prefix
                profile["displayPathIndex"] = f"{dataset.prefix}:{local_id}"
                for residue_key in ("res_1", "res_2", "res_3", "res_4"):
                    profile[residue_key] = [
                        self._encode_residue_id(dataset, int(value))
                        if int(value or 0) > 0 else 0
                        for value in profile.get(residue_key, [])
                    ]
                profiles.append(profile)
        return profiles

    def get_residue_options(
        self,
        selected_residues: Optional[Set[int]] = None,
    ) -> List[dict]:
        options: List[dict] = []
        has_selected = bool(selected_residues)
        for dataset in self._datasets:
            local_selected = self._extract_local_residue_filter(dataset, selected_residues)
            if has_selected and not local_selected:
                continue
            local_options = dataset.db.get_residue_options(local_selected if has_selected else None)
            for option in local_options:
                local_id = int(option["value"])
                global_id = self._encode_residue_id(dataset, local_id)
                options.append(
                    {
                        "label": f"[{dataset.prefix}] {option['label']}",
                        "value": str(global_id),
                    }
                )
        return options

    def get_residue_positions(self) -> List[dict]:
        positions: List[dict] = []
        for dataset in self._datasets:
            for row in dataset.db.get_residue_positions():
                row = dict(row)
                local_id = int(row["residue_id"])
                row["local_residue_id"] = local_id
                row["label"] = f"{dataset.prefix}:{local_id}"
                row["dataset_prefix"] = dataset.prefix
                row["dataset_key"] = dataset.key
                row["residue_id"] = self._encode_residue_id(dataset, local_id)
                positions.append(row)
        return positions

    def get_path_for_export(self, path_id: int) -> Optional[dict]:
        dataset, local_id = self._decode_path_id(path_id)
        if dataset is None:
            return None
        payload = dataset.db.get_path_for_export(local_id)
        if not payload:
            return None
        payload = dict(payload)
        meta = dict(payload.get("meta", {}))
        meta["dataset_prefix"] = dataset.prefix
        meta["local_id"] = local_id
        meta["display_id"] = f"{dataset.prefix}:{local_id}"
        meta["id"] = path_id
        payload["meta"] = meta
        return payload

    def list_sessions(self) -> List[dict]:
        primary = self.primary_db
        return primary.list_sessions() if primary else []

    def create_session(self, session_id: str, name: str, data_json: str = "{}"):
        primary = self.primary_db
        if primary:
            primary.create_session(session_id, name, data_json)

    def load_session(self, session_id: str) -> Optional[dict]:
        primary = self.primary_db
        return primary.load_session(session_id) if primary else None

    def save_session(self, session_id: str, name: str, data_json: str):
        primary = self.primary_db
        if primary:
            primary.save_session(session_id, name, data_json)

    def delete_session(self, session_id: str):
        primary = self.primary_db
        if primary:
            primary.delete_session(session_id)

    def _group_path_ids(self, path_ids: List[int]) -> Dict[str, List[int]]:
        grouped: Dict[str, List[int]] = {}
        for path_id in path_ids:
            dataset, local_id = self._decode_path_id(int(path_id))
            if dataset is None:
                continue
            grouped.setdefault(dataset.key, []).append(local_id)
        return grouped

    def _with_dataset_row_fields(self, dataset: DatasetBinding, row: dict) -> dict:
        payload = dict(row)
        local_id = int(payload["id"])
        payload["local_id"] = local_id
        payload["dataset_prefix"] = dataset.prefix
        payload["dataset_name"] = dataset.name
        payload["display_id"] = f"{dataset.prefix}:{local_id}"
        payload["id"] = self._encode_path_id(dataset, local_id)
        return payload

    def _sort_value(self, row: dict, order_by: str):
        if order_by == "dataset_prefix":
            return (str(row.get("dataset_prefix", "")), int(row.get("local_id", 0)))
        if order_by == "id":
            return (int(row.get("local_id", 0)), str(row.get("dataset_prefix", "")))
        value = row.get(order_by)
        if value is None:
            return (1, 0)
        return (0, value)

    def _extract_local_path_filter(
        self,
        dataset: DatasetBinding,
        path_id_filter: Optional[Set[int]],
    ) -> Optional[Set[int]]:
        if path_id_filter is None:
            return None
        local_ids: Set[int] = set()
        for path_id in path_id_filter:
            bound_dataset, local_id = self._decode_path_id(int(path_id))
            if bound_dataset is not None and bound_dataset.key == dataset.key:
                local_ids.add(local_id)
        return local_ids

    def _extract_local_residue_filter(
        self,
        dataset: DatasetBinding,
        residue_filter: Optional[Set[int]],
    ) -> Optional[Set[int]]:
        if not residue_filter:
            return None
        local_ids: Set[int] = set()
        for residue_id in residue_filter:
            bound_dataset, local_id = self._decode_residue_id(int(residue_id))
            if bound_dataset is not None and bound_dataset.key == dataset.key:
                local_ids.add(local_id)
        return local_ids

    def _encode_path_id(self, dataset: DatasetBinding, local_id: int) -> int:
        return dataset.path_id_base + int(local_id)

    def _decode_path_id(self, path_id: int) -> Tuple[Optional[DatasetBinding], Optional[int]]:
        slot = int(path_id) // PATH_ID_BLOCK
        dataset = self._path_slot_map.get(slot)
        if dataset is None:
            return None, None
        return dataset, int(path_id) - dataset.path_id_base

    def _encode_residue_id(self, dataset: DatasetBinding, local_id: int) -> int:
        return dataset.residue_id_base + int(local_id)

    def _decode_residue_id(self, residue_id: int) -> Tuple[Optional[DatasetBinding], Optional[int]]:
        slot = int(residue_id) // RESIDUE_ID_BLOCK
        dataset = self._residue_slot_map.get(slot)
        if dataset is None:
            return None, None
        return dataset, int(residue_id) - dataset.residue_id_base

    def _default_prefix(self, name: str) -> str:
        stem = os.path.splitext(name)[0]
        cleaned = "".join(ch for ch in stem if ch.isalnum())
        if not cleaned:
            cleaned = "DS"
        return cleaned[:8].upper()

    def _make_unique_prefix(self, prefix: str, exclude_key: Optional[str] = None) -> str:
        cleaned = "".join(ch for ch in str(prefix).strip().upper() if ch.isalnum()) or "DS"
        used = {
            dataset.prefix
            for dataset in self._datasets
            if dataset.key != exclude_key
        }
        candidate = cleaned[:8]
        if candidate not in used:
            return candidate
        suffix = 2
        while True:
            tail = str(suffix)
            candidate = f"{cleaned[:max(1, 8 - len(tail))]}{tail}"
            if candidate not in used:
                return candidate
            suffix += 1
