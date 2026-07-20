"""
QAbstractTableModel backed by SQLite for the paths table.
Supports virtual scrolling (10w+ rows), sorting, filtering.
Only fetches visible chunks from the database.
"""
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, Signal
from PySide6.QtGui import QColor
from typing import Optional, Set, List, Any


# Column definitions
COLUMNS = [
    ("dataset_prefix", "Dataset", 96),
    ("id", "Path ID", 62),
    ("frame_id", "Frame", 70),
    ("path_length", "Length", 80),
    ("min_radius", "Min R", 70),
    ("avg_radius", "Avg R", 70),
    ("avg_hydrophobicity", "Avg Hydro", 80),
    ("residue_count", "# Residues", 80),
    ("cluster_id", "Cluster", 48),
]

COL_KEYS = [c[0] for c in COLUMNS]
COL_HEADERS = [c[1] for c in COLUMNS]
COL_WIDTHS = [c[2] for c in COLUMNS]


class PathTableModel(QAbstractTableModel):
    """
    Virtual-scrolling table model for paths.
    Fetches data in chunks from SQLite, never holds all rows in memory.
    """

    total_count_changed = Signal(int)

    CHUNK_SIZE = 500  # Rows fetched per chunk

    def __init__(self, db, parent=None):
        super().__init__(parent)
        self.db = db

        # Current filter/sort state
        self._sort_col = "id"
        self._sort_dir = "ASC"
        self._frame_min: Optional[int] = None
        self._frame_max: Optional[int] = None
        self._residue_filter: Optional[Set[int]] = None
        self._path_id_filter: Optional[Set[int]] = None

        # Row count
        self._total_rows = 0

        # Chunk cache: {chunk_index: [row_dicts]}
        self._cache = {}
        self._highlighted_paths: Set[int] = set()
        self._selected_paths: Set[int] = set()

        self.refresh()

    def refresh(self):
        """Re-query the total count and clear cache."""
        self.beginResetModel()
        self._cache.clear()
        self._total_rows = self.db.count_filtered_paths(
            frame_min=self._frame_min,
            frame_max=self._frame_max,
            residue_filter=self._residue_filter,
            path_id_filter=self._path_id_filter,
        )
        self.endResetModel()
        self.total_count_changed.emit(self._total_rows)

    def set_filters(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ):
        self._frame_min = frame_min
        self._frame_max = frame_max
        self._residue_filter = residue_filter
        self._path_id_filter = path_id_filter
        self.refresh()

    def set_highlighted_paths(self, path_ids: Set[int]):
        self._highlighted_paths = path_ids
        self._emit_bg_changed()

    def set_selected_paths(self, path_ids: Set[int]):
        self._selected_paths = path_ids
        self._emit_bg_changed()

    def _emit_bg_changed(self):
        # Emit dataChanged for all visible rows
        if self._total_rows > 0:
            self.dataChanged.emit(
                self.index(0, 0),
                self.index(self._total_rows - 1, len(COLUMNS) - 1),
                [Qt.BackgroundRole],
            )

    # ─── QAbstractTableModel interface ──────────────────
    def rowCount(self, parent=QModelIndex()) -> int:
        return self._total_rows

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section: int, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return COL_HEADERS[section]
        return None

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        row_data = self._get_row(row)
        if row_data is None:
            return None

        if role == Qt.DisplayRole:
            key = COL_KEYS[col]
            if key == "id":
                local_id = row_data.get("local_id")
                if local_id is not None:
                    return str(local_id)
                display_id = str(row_data.get("display_id", ""))
                if ":" in display_id:
                    return display_id.split(":", 1)[1]
                return display_id
            if key == "dataset_prefix":
                return row_data.get("dataset_prefix", "")
            val = row_data.get(key)
            if isinstance(val, float):
                return f"{val:.3f}"
            return str(val) if val is not None else ""

        if role == Qt.BackgroundRole:
            path_id = row_data.get("id")
            if path_id is not None and path_id in self._selected_paths:
                return QColor(255, 215, 0, 80)  # Gold for lasso-selected
            if path_id is not None and path_id in self._highlighted_paths:
                return QColor(236, 245, 255)  # Light blue highlight

        if role == Qt.UserRole:
            # Return full row dict
            return row_data

        return None

    def sort(self, column: int, order=Qt.AscendingOrder):
        if 0 <= column < len(COL_KEYS):
            self._sort_col = COL_KEYS[column]
            self._sort_dir = "ASC" if order == Qt.AscendingOrder else "DESC"
            self.refresh()

    # ─── Chunk cache ────────────────────────────────────
    def _get_row(self, row: int) -> Optional[dict]:
        chunk_idx = row // self.CHUNK_SIZE
        if chunk_idx not in self._cache:
            self._fetch_chunk(chunk_idx)
        local_idx = row % self.CHUNK_SIZE
        chunk = self._cache.get(chunk_idx, [])
        if local_idx < len(chunk):
            return chunk[local_idx]
        return None

    def _fetch_chunk(self, chunk_idx: int):
        offset = chunk_idx * self.CHUNK_SIZE
        rows = self.db.get_paths_page(
            offset=offset,
            limit=self.CHUNK_SIZE,
            order_by=self._sort_col,
            order_dir=self._sort_dir,
            frame_min=self._frame_min,
            frame_max=self._frame_max,
            residue_filter=self._residue_filter,
            path_id_filter=self._path_id_filter,
        )
        self._cache[chunk_idx] = rows

        # Evict old chunks (keep max 10 chunks in memory = 5000 rows)
        while len(self._cache) > 10:
            oldest = min(self._cache.keys())
            if oldest != chunk_idx:
                del self._cache[oldest]
            else:
                break

    def get_path_id(self, row: int) -> Optional[int]:
        row_data = self._get_row(row)
        return row_data.get("id") if row_data else None

    def get_path_ids_for_rows(self, rows: List[int]) -> List[int]:
        ids = []
        for row in rows:
            pid = self.get_path_id(row)
            if pid is not None:
                ids.append(pid)
        return ids

    def get_loaded_row_indices_for_path_ids(self, path_ids: Set[int]) -> List[int]:
        """Return row indices for currently cached rows only (O(cached_rows))."""
        if not path_ids:
            return []

        target = set(path_ids)
        result: list[int] = []
        for chunk_idx, rows in self._cache.items():
            base = chunk_idx * self.CHUNK_SIZE
            for i, row in enumerate(rows):
                pid = row.get("id")
                if pid in target:
                    result.append(base + i)
        return result
