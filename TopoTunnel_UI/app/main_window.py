"""
Main application window.
Replaces App.vue - QMainWindow with toolbar, side panel, 3D viewer, charts.
"""
import os
import json
import tempfile
import time
import re
import csv
import pickle
import math
from collections import defaultdict
from typing import Optional, Callable
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QDockWidget, QTabWidget, QToolBar, QLabel, QPushButton,
    QComboBox, QSlider, QStatusBar, QFileDialog, QMessageBox,
    QDialog, QLineEdit, QDialogButtonBox, QApplication,
    QToolButton, QMenu, QWidgetAction, QListView, QCheckBox, QScrollArea, QProgressBar, QFrame,
    QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy, QAbstractItemView, QStyledItemDelegate,
    QStackedWidget, QSpinBox, QDoubleSpinBox,
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject, QEvent
from PySide6.QtGui import QAction, QShortcut, QKeySequence, QFontMetrics, QColor, QBrush, QPainter, QPen

from TopoTunnel_UI.app.widgets.residue_panel import ResiduePanel
from TopoTunnel_UI.app.widgets.path_panel import PathPanel
from TopoTunnel_UI.app.widgets.dataset_panel import DatasetPanel
from TopoTunnel_UI.app.widgets.residue_compare_panel import ResidueComparePanel
from TopoTunnel_UI.app.widgets.residue_combination_panel import ResidueCombinationPanel
from TopoTunnel_UI.app.widgets.protein_viewer_panel import ProteinViewerPanel
from TopoTunnel_UI.app.models.path_model import PathTableModel
from TopoTunnel_UI.db.multi_dataset_db import MultiDatasetDatabase
from TopoTunnel_UI.core.residue_combination import (
    bottleneck_point_indices,
    combination_rows_from_profiles,
    combination_centroid,
    combination_property_trend,
    combination_property_value,
    compare_combination_row_sets,
    exit_region_from_path_coords,
    filter_path_ids_by_exit_region,
    filter_profiles_by_residue_subset,
    profile_residue_set,
    residue_set_from_profiles,
    residue_set_jaccard_similarity,
    trimmed_exit_region_from_path_coords,
    trimmed_length_range,
)
from TopoTunnel_UI.core.state import AppState
from TopoTunnel_UI.core.exporter import export_paths_to_pdb
from TopoTunnel_UI.vis.tunnel_viewer import TunnelViewer3D
from TopoTunnel_UI.vis.pse_parser import parse_pse_file, pse_protein_to_pdb_string
from TopoTunnel_UI.vis.charts import ProfileChart, StatisticsChart


class DataLoader(QObject):
    """Background worker for loading 3D render data."""
    finished = Signal(list, str, int)  # data, coordinate_mode, request_id
    failed = Signal(str, str, int)  # error, coordinate_mode, request_id
    progress = Signal(int)

    def __init__(self, db, frame_min=None, frame_max=None, coordinate_mode="current", request_id=0):
        super().__init__()
        self.db = db
        self.frame_min = frame_min
        self.frame_max = frame_max
        self.coordinate_mode = coordinate_mode if coordinate_mode in {"current", "original"} else "current"
        self.request_id = int(request_id)

    def run(self):
        try:
            if self.coordinate_mode == "original":
                data = self.db.get_all_original_render_coords(self.frame_min, self.frame_max)
            else:
                data = self.db.get_all_render_coords(self.frame_min, self.frame_max)
        except Exception as exc:
            self.failed.emit(str(exc), self.coordinate_mode, self.request_id)
            return
        self.finished.emit(data, self.coordinate_mode, self.request_id)


class ResidueCombinationTaskWorker(QObject):
    """Background worker for residue-combination queries/calculations."""

    finished = Signal(dict)

    def __init__(self, task: str, payload: dict):
        super().__init__()
        self.task = str(task)
        self.payload = dict(payload or {})

    def run(self):
        try:
            if self.task == "query_match":
                result = self._run_query_match()
            elif self.task == "compute_match":
                result = self._run_compute_match()
            else:
                result = {"ok": False, "error": f"unknown_task:{self.task}"}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        result["task"] = self.task
        result["request_id"] = int(self.payload.get("request_id", 0) or 0)
        self.finished.emit(result)

    def _run_query_match(self) -> dict:
        db = self.payload["db"]
        source_key = str(self.payload.get("source_key") or "")
        target_key = str(self.payload.get("target_key") or "")
        source_path_ids = [int(path_id) for path_id in self.payload.get("source_path_ids", [])]
        keep_ratio = float(self.payload.get("keep_ratio", 1.0) or 1.0)
        frame_min = self.payload.get("frame_min")
        frame_max = self.payload.get("frame_max")
        datasets = list(self.payload.get("datasets", []) or [])

        source_binding = getattr(db, "_datasets_by_key", {}).get(source_key)
        target_binding = getattr(db, "_datasets_by_key", {}).get(target_key)
        source_db = source_binding.db if source_binding is not None else None
        target_db = target_binding.db if target_binding is not None else None
        if source_db is None or target_db is None:
            return {"ok": False, "error": "Dataset binding is unavailable, so matched paths cannot be queried"}

        source_local_path_ids = db.get_local_path_ids_for_dataset(source_key, source_path_ids)
        source_profiles = source_db.get_path_profiles(source_local_path_ids)
        source_residue_ids = residue_set_from_profiles(source_profiles)
        if not source_residue_ids:
            return {"ok": False, "error": "No valid residues were extracted from the source paths, so matched paths cannot be queried"}

        source_exit_region = trimmed_exit_region_from_path_coords(
            source_db.get_render_coords(source_local_path_ids),
            keep_ratio,
        )
        if not source_exit_region:
            return {"ok": False, "error": "No valid exit positions were extracted from the source paths, so matched paths cannot be queried"}

        source_length_map = source_db.get_path_length_map(source_local_path_ids)
        source_length_region = trimmed_length_range(source_length_map, keep_ratio)
        if not source_length_region:
            return {"ok": False, "error": "No valid path lengths were extracted from the source paths, so matched paths cannot be queried"}
        source_length_min = float(source_length_region["min"])
        source_length_max = float(source_length_region["max"])

        target_global_path_ids = db.get_dataset_path_ids(target_key, frame_min, frame_max)
        target_local_path_ids = db.get_local_path_ids_for_dataset(target_key, target_global_path_ids)
        target_local_path_ids = filter_path_ids_by_exit_region(
            target_db.get_render_coords(target_local_path_ids),
            source_exit_region,
        )
        target_length_map = target_db.get_path_length_map(target_local_path_ids)
        target_local_path_ids = [
            int(path_id)
            for path_id, path_length in target_length_map.items()
            if float(path_length) >= source_length_min and float(path_length) <= source_length_max
        ]
        target_local_path_ids.sort()
        target_profiles = target_db.get_path_profiles(target_local_path_ids)
        matched_target_profiles = filter_profiles_by_residue_subset(target_profiles, source_residue_ids)
        matched_target_local_ids = [
            int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
            for profile in matched_target_profiles
            if int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1) > 0
        ]
        target_path_id_base = int(getattr(target_binding, "path_id_base", 0) or 0)
        matched_target_global_ids = [
            target_path_id_base + int(local_id)
            for local_id in matched_target_local_ids
        ]
        source_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == source_key), {})
        target_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == target_key), {})
        summary = (
            f"Match query completed | Source [{source_dataset.get('prefix', source_key)}]: {len(source_profiles)} paths | "
            f"Keep ratio {keep_ratio:.2f} | "
            f"Exit radius {float(source_exit_region.get('radius', 0.0)):.3f} | "
            f"Length range [{source_length_min:.3f}, {source_length_max:.3f}] | "
            f"Target [{target_dataset.get('prefix', target_key)}] prefiltered: {len(target_profiles)} paths | "
            f"Residue-matched: {len(matched_target_profiles)} paths"
        )
        return {
            "ok": True,
            "source_profiles_count": len(source_profiles),
            "target_profiles_count": len(target_profiles),
            "matched_target_profiles_count": len(matched_target_profiles),
            "matched_target_global_ids": matched_target_global_ids,
            "target_key": target_key,
            "summary": summary,
        }

    def _run_compute_match(self) -> dict:
        db = self.payload["db"]
        file_a_key = str(self.payload.get("file_a_key") or "")
        file_b_key = str(self.payload.get("file_b_key") or "")
        using_matched_paths = bool(self.payload.get("using_matched_paths"))
        matched_target_key = str(self.payload.get("matched_target_key") or "")
        source_key = str(self.payload.get("source_key") or "")
        matched_target_path_ids = {int(path_id) for path_id in self.payload.get("matched_target_path_ids", [])}
        effective_path_ids = [int(path_id) for path_id in self.payload.get("effective_path_ids", [])]

        profiles_by_dataset_key: dict[str, list[dict]] = {}
        if using_matched_paths:
            target_path_ids = [
                int(path_id)
                for path_id in effective_path_ids
                if db.get_dataset_key_for_path_id(int(path_id)) == matched_target_key
                and int(path_id) in matched_target_path_ids
            ]
            source_path_ids = [
                int(path_id)
                for path_id in effective_path_ids
                if db.get_dataset_key_for_path_id(int(path_id)) == source_key
            ]
            source_binding = getattr(db, "_datasets_by_key", {}).get(source_key)
            target_binding = getattr(db, "_datasets_by_key", {}).get(matched_target_key)
            source_db = source_binding.db if source_binding is not None else None
            target_db = target_binding.db if target_binding is not None else None
            if source_db is None or target_db is None:
                return {"ok": False, "error": "Dataset binding is unavailable, so differences cannot be computed"}
            if not source_path_ids:
                return {"ok": False, "error": "No source-dataset paths are currently selected, so differences cannot be computed"}
            source_local_path_ids = db.get_local_path_ids_for_dataset(source_key, source_path_ids)
            target_local_path_ids = db.get_local_path_ids_for_dataset(matched_target_key, target_path_ids)
            profiles_by_dataset_key = {
                str(source_key): source_db.get_path_profiles(source_local_path_ids),
                str(matched_target_key): target_db.get_path_profiles(target_local_path_ids),
            }
            if file_a_key not in profiles_by_dataset_key or file_b_key not in profiles_by_dataset_key:
                return {"ok": False, "error": "Matched-path source/target datasets do not match the current File A/File B selection"}
            summary = (
                f"{self.payload.get('last_compare_summary', '')} | "
                f"Target paths kept after manual filtering: {len(profiles_by_dataset_key.get(matched_target_key, []))} | "
                f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
            )
        else:
            if not effective_path_ids:
                return {"ok": False, "error": "No paths are currently selected, so differences cannot be computed"}
            for dataset_key in (file_a_key, file_b_key):
                dataset_binding = getattr(db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    return {"ok": False, "error": "Dataset binding is unavailable, so differences cannot be computed"}
                local_path_ids = db.get_local_path_ids_for_dataset(dataset_key, effective_path_ids)
                profiles_by_dataset_key[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
            if not profiles_by_dataset_key.get(file_a_key) and not profiles_by_dataset_key.get(file_b_key):
                return {"ok": False, "error": "The current selection does not contain any File A or File B paths to compare"}
            summary = (
                f"Current selection compare | "
                f"File A: {len(profiles_by_dataset_key.get(file_a_key, []))} paths | "
                f"File B: {len(profiles_by_dataset_key.get(file_b_key, []))} paths | "
                f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
            )

        file_a_profiles = profiles_by_dataset_key.get(file_a_key, [])
        file_b_profiles = profiles_by_dataset_key.get(file_b_key, [])
        compare_rows = compare_combination_row_sets(
            combination_rows_from_profiles(file_a_profiles, bottleneck_only=False),
            combination_rows_from_profiles(file_b_profiles, bottleneck_only=False),
            include_right_only=True,
            left_total_paths=len(file_a_profiles),
            right_total_paths=len(file_b_profiles),
        )
        bottleneck_compare_rows = compare_combination_row_sets(
            combination_rows_from_profiles(file_a_profiles, bottleneck_only=True),
            combination_rows_from_profiles(file_b_profiles, bottleneck_only=True),
            include_right_only=True,
            left_total_paths=len(file_a_profiles),
            right_total_paths=len(file_b_profiles),
        )
        return {
            "ok": True,
            "file_a_key": file_a_key,
            "file_b_key": file_b_key,
            "file_a_profiles_count": len(file_a_profiles),
            "file_b_profiles_count": len(file_b_profiles),
            "profiles_by_dataset_key": profiles_by_dataset_key,
            "compare_rows": compare_rows,
            "bottleneck_compare_rows": bottleneck_compare_rows,
            "summary": summary,
        }


class DockResizeHandle(QFrame):
    """Slim drag handle shown at the right edge of the control panel."""

    def __init__(
        self,
        get_width: Callable[[], int],
        apply_width: Callable[[int], None],
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._get_width = get_width
        self._apply_width = apply_width
        self._dragging = False
        self._start_global_x = 0
        self._start_width = 0
        self.setCursor(Qt.SizeHorCursor)
        self.setFixedWidth(8)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setToolTip("Drag to resize Control Panel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            "QFrame{background:#f8fafc;border-left:1px solid #e5e7eb;}"
            "QFrame:hover{background:#e2e8f0;border-left:1px solid #cbd5e1;}"
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._start_global_x = int(event.globalPosition().x())
            self._start_width = int(self._get_width())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            delta = int(event.globalPosition().x()) - self._start_global_x
            self._apply_width(self._start_width + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging and event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class ObserverTimelineWidget(QWidget):
    """Compact frame timeline for the residue observer."""

    frame_clicked = Signal(str, int)
    zoom_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []
        self._zoom = 1.0
        self._hit_targets: list[dict] = []
        self._timeline_signature = None
        self.setMinimumHeight(72)
        self.setMaximumHeight(88)
        self.setToolTip("Green marks above the axis are File A; green marks below the axis are File B.")

    def set_zoom_percent(self, value: int) -> None:
        self._zoom = max(1.0, min(100.0, int(value) / 100.0))
        self.update()

    def zoom_percent(self) -> int:
        return int(round(float(self._zoom) * 100.0))

    def set_timelines(
        self,
        top_frames,
        top_highlighted_frames,
        top_current_frame: int | None,
        bottom_frames,
        bottom_highlighted_frames,
        bottom_current_frame: int | None,
    ) -> None:
        signature = (
            tuple(sorted({int(frame) for frame in (top_frames or [])})),
            tuple(sorted({int(frame) for frame in (top_highlighted_frames or set())})),
            int(top_current_frame) if top_current_frame is not None else None,
            tuple(sorted({int(frame) for frame in (bottom_frames or [])})),
            tuple(sorted({int(frame) for frame in (bottom_highlighted_frames or set())})),
            int(bottom_current_frame) if bottom_current_frame is not None else None,
        )
        if signature == self._timeline_signature:
            self.update()
            return
        self._timeline_signature = signature
        self._rows = [
            {
                "label": "A",
                "frames": list(signature[0]),
                "highlighted_frames": set(signature[1]),
                "current_frame": signature[2],
                "accent": QColor("#16A34A"),
                "current_accent": QColor("#1D4ED8"),
            },
            {
                "label": "B",
                "frames": list(signature[3]),
                "highlighted_frames": set(signature[4]),
                "current_frame": signature[5],
                "accent": QColor("#16A34A"),
                "current_accent": QColor("#1D4ED8"),
            },
        ]
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self._hit_targets = []

        rect = self.rect().adjusted(8, 6, -8, -6)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#E5E7EB"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 6, 6)

        if not self._rows:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignCenter, "Timeline: load File A / File B frames")
            painter.end()
            return

        left = rect.left() + 36
        right = rect.right() - 8
        width = max(1, right - left)
        all_frames = sorted({
            int(frame)
            for row in self._rows
            for frame in list(row.get("frames") or [])
        })
        if not all_frames:
            painter.setPen(QColor("#6B7280"))
            painter.drawText(rect, Qt.AlignCenter, "Timeline: load File A / File B frames")
            painter.end()
            return

        current_candidates = [
            int(row.get("current_frame"))
            for row in self._rows
            if row.get("current_frame") is not None
        ]
        current_frame = current_candidates[0] if current_candidates else None
        full_min = min(all_frames)
        full_max = max(all_frames)
        frame_min, frame_max = self._visible_frame_range(full_min, full_max, current_frame)
        span = max(1, frame_max - frame_min)
        axis_y = rect.top() + max(30, rect.height() // 2)

        def x_for_frame(frame: int) -> int:
            return int(round(left + ((int(frame) - frame_min) / span) * width))

        painter.setPen(QColor("#374151"))
        painter.drawText(rect.left(), axis_y - 28, 28, 14, Qt.AlignLeft | Qt.AlignVCenter, "A")
        painter.drawText(rect.left(), axis_y + 12, 28, 14, Qt.AlignLeft | Qt.AlignVCenter, "B")

        painter.setPen(QPen(QColor("#9CA3AF"), 2))
        painter.drawLine(left, axis_y, right, axis_y)

        tick_count = min(7, max(2, len(all_frames)))
        if frame_max == frame_min:
            tick_frames = [frame_min]
        else:
            tick_frames = [
                int(round(frame_min + (frame_max - frame_min) * idx / max(1, tick_count - 1)))
                for idx in range(tick_count)
            ]
        painter.setPen(QPen(QColor("#9CA3AF"), 1))
        for frame in tick_frames:
            x = x_for_frame(frame)
            painter.drawLine(x, axis_y - 5, x, axis_y + 5)
            painter.setPen(QColor("#6B7280"))
            painter.drawText(x - 24, axis_y + 18, 48, 12, Qt.AlignCenter, str(frame))
            painter.setPen(QPen(QColor("#9CA3AF"), 1))

        painter.setPen(QColor("#6B7280"))
        painter.drawText(
            left,
            rect.top() + 2,
            width,
            14,
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Frames {full_min}-{full_max} ({len(all_frames)})",
        )

        marker_specs = [
            ("top", self._rows[0] if len(self._rows) > 0 else {}, -1),
            ("bottom", self._rows[1] if len(self._rows) > 1 else {}, 1),
        ]
        for side, row, direction in marker_specs:
            accent = row.get("accent") or QColor("#16A34A")
            painter.setPen(QPen(accent, 2))
            painter.setBrush(QBrush(QColor(accent.red(), accent.green(), accent.blue())))
            for frame in sorted(row.get("highlighted_frames") or set()):
                if frame < frame_min or frame > frame_max:
                    continue
                x = x_for_frame(frame)
                y0 = axis_y + direction * 4
                y1 = axis_y + direction * 18
                dot_y = axis_y + direction * 22
                painter.drawLine(x, y0, x, y1)
                painter.drawEllipse(x - 3, dot_y - 3, 6, 6)
                self._hit_targets.append(
                    {
                        "side": side,
                        "frame": int(frame),
                        "x": int(x),
                        "y": int(dot_y),
                    }
                )

        if current_frame is not None and frame_min <= int(current_frame) <= frame_max:
            x = x_for_frame(int(current_frame))
            painter.setPen(QPen(QColor("#1D4ED8"), 2))
            painter.setBrush(QBrush(QColor("#2563EB")))
            painter.drawLine(x, axis_y - 24, x, axis_y + 24)
            painter.drawEllipse(x - 4, axis_y - 28, 8, 8)

        painter.end()

    def _visible_frame_range(self, frame_min: int, frame_max: int, current_frame) -> tuple[int, int]:
        if self._zoom <= 1.01 or frame_max <= frame_min:
            return int(frame_min), int(frame_max)
        full_span = max(1, int(frame_max) - int(frame_min))
        visible_span = max(1, int(round(full_span / self._zoom)))
        center = int(current_frame) if current_frame is not None else int(round((frame_min + frame_max) / 2))
        start = center - visible_span // 2
        end = start + visible_span
        if start < frame_min:
            start = int(frame_min)
            end = min(int(frame_max), start + visible_span)
        if end > frame_max:
            end = int(frame_max)
            start = max(int(frame_min), end - visible_span)
        return int(start), int(end)

    def mousePressEvent(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        x = float(pos.x())
        y = float(pos.y())
        best = None
        best_dist2 = 999999.0
        for target in self._hit_targets:
            dx = x - float(target.get("x", 0))
            dy = y - float(target.get("y", 0))
            dist2 = dx * dx + dy * dy
            if dist2 < best_dist2:
                best = target
                best_dist2 = dist2
        if best is not None and best_dist2 <= 144.0:
            self.frame_clicked.emit(str(best.get("side") or "top"), int(best.get("frame", 0)))
            return
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        current = self.zoom_percent()
        direction = 1 if delta > 0 else -1
        # Multiplicative zoom keeps wheel control responsive at both low and high zoom.
        factor = 1.25 if direction > 0 else 0.80
        next_value = int(round(current * factor))
        next_value = max(100, min(5000, next_value))
        if next_value != current:
            self.set_zoom_percent(next_value)
            self.zoom_changed.emit(next_value)
        event.accept()


class ChartWorkspaceDialog(QDialog):
    """Detached, larger chart workspace linked to the main window state."""

    closed = Signal()
    path_filter_apply_requested = Signal(dict)
    path_filter_revert_requested = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = getattr(parent, "db", None)
        self.setWindowTitle("Chart Workspace")
        self.setModal(False)
        self.setMinimumSize(1180, 920)
        self.resize(1260, 980)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("Chart Workspace")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        subtitle = QLabel("Expanded, synchronized view for profile and statistics charts")
        subtitle.setStyleSheet("color: #909399; font-size: 11px;")
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text)
        header.addStretch()
        self._close_btn = QPushButton("Close")
        self._close_btn.setFixedHeight(28)
        self._close_btn.clicked.connect(self.close)
        header.addWidget(self._close_btn)
        layout.addLayout(header)

        self._content_splitter = QSplitter(Qt.Horizontal)
        self._content_splitter.setChildrenCollapsible(False)

        chart_column = QWidget()
        chart_layout = QVBoxLayout(chart_column)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        self._splitter = QSplitter(Qt.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._profile_chart = ProfileChart(workspace_mode=True)
        self._statistics_chart = StatisticsChart(show_bottleneck_points=True)
        self._statistics_chart.set_display_config(*self._profile_chart.display_config())
        self._profile_chart.dataset_filter_changed.connect(self._statistics_chart.set_dataset_filter)
        self._profile_chart.residue_stats_updated.connect(self._update_residue_summary_panel)
        self._splitter.addWidget(self._profile_chart)
        self._splitter.addWidget(self._statistics_chart)
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([520, 320])
        chart_layout.addWidget(self._splitter, 1)

        side_panel = QWidget()
        side_panel.setMinimumWidth(320)
        side_panel.setMaximumWidth(460)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(8)

        side_title = QLabel("Residue Summary")
        side_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #1f2937;")
        side_layout.addWidget(side_title)

        side_hint = QLabel(
            "All visible paths are summarized by default. Use `Residue Stats` mode "
            "and drag a selection box to summarize only selected chart points."
        )
        side_hint.setWordWrap(True)
        side_hint.setStyleSheet("color: #909399; font-size: 11px;")
        side_layout.addWidget(side_hint)

        summary_toolbar = QHBoxLayout()
        summary_toolbar.setContentsMargins(0, 0, 0, 0)
        summary_toolbar.setSpacing(6)
        self._residue_summary_view_label = QLabel("View")
        summary_toolbar.addWidget(self._residue_summary_view_label)
        self._residue_summary_mode_combo = QComboBox()
        self._residue_summary_mode_combo.addItem("Path Residues", "path")
        self._residue_summary_mode_combo.addItem("Bottleneck", "bottleneck")
        self._residue_summary_mode_combo.currentIndexChanged.connect(self._refresh_residue_summary_table)
        summary_toolbar.addWidget(self._residue_summary_mode_combo)
        self._residue_summary_heatmap_btn = QPushButton("Compare")
        self._residue_summary_heatmap_btn.setFixedHeight(28)
        self._residue_summary_heatmap_btn.clicked.connect(self._open_residue_summary_heatmap)
        summary_toolbar.addWidget(self._residue_summary_heatmap_btn)
        summary_toolbar.addStretch()
        side_layout.addLayout(summary_toolbar)

        self._residue_summary_scroll = QScrollArea()
        self._residue_summary_scroll.setWidgetResizable(True)
        self._residue_summary_scroll.setFrameShape(QScrollArea.NoFrame)
        self._residue_summary_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._residue_summary_container = QWidget()
        self._residue_summary_layout = QVBoxLayout(self._residue_summary_container)
        self._residue_summary_layout.setContentsMargins(0, 0, 0, 0)
        self._residue_summary_layout.setSpacing(10)
        self._residue_summary_layout.addStretch()
        self._residue_summary_scroll.setWidget(self._residue_summary_container)
        side_layout.addWidget(self._residue_summary_scroll, 1)
        self._residue_summary_rows: list[dict] = []
        self._residue_summary_sort_state: dict[str, tuple[str, bool]] = {}
        self._residue_summary_sections: dict[str, QFrame] = {}
        self._residue_summary_tables: dict[str, QTableWidget] = {}
        self._residue_summary_heatmap_dialog = None

        filter_frame = QFrame()
        filter_frame.setStyleSheet("QFrame{border:1px solid #e5ecf6;border-radius:10px;background:#ffffff;}")
        filter_layout = QVBoxLayout(filter_frame)
        filter_layout.setContentsMargins(12, 12, 12, 12)
        filter_layout.setSpacing(8)

        filter_title = QLabel("Path Filter")
        filter_title.setStyleSheet("font-size: 13px; font-weight: 600; color: #1f2937;")
        filter_layout.addWidget(filter_title)

        self._chart_selection_info = QLabel("No chart-path selection.")
        self._chart_selection_info.setWordWrap(True)
        self._chart_selection_info.setStyleSheet("color: #6b7280; font-size: 11px;")
        filter_layout.addWidget(self._chart_selection_info)

        hydro_row = QHBoxLayout()
        hydro_row.setSpacing(6)
        hydro_row.addWidget(QLabel("Hydro"))
        self._hydro_min_edit = QLineEdit()
        self._hydro_min_edit.setPlaceholderText("min")
        self._hydro_max_edit = QLineEdit()
        self._hydro_max_edit.setPlaceholderText("max")
        self._hydro_min_edit.textChanged.connect(self._update_filtered_selection_preview)
        self._hydro_max_edit.textChanged.connect(self._update_filtered_selection_preview)
        hydro_row.addWidget(self._hydro_min_edit)
        hydro_row.addWidget(self._hydro_max_edit)
        filter_layout.addLayout(hydro_row)

        frame_row = QHBoxLayout()
        frame_row.setSpacing(6)
        frame_row.addWidget(QLabel("Frame"))
        self._frame_min_edit = QLineEdit()
        self._frame_min_edit.setPlaceholderText("min")
        self._frame_max_edit = QLineEdit()
        self._frame_max_edit.setPlaceholderText("max")
        self._frame_min_edit.textChanged.connect(self._update_filtered_selection_preview)
        self._frame_max_edit.textChanged.connect(self._update_filtered_selection_preview)
        frame_row.addWidget(self._frame_min_edit)
        frame_row.addWidget(self._frame_max_edit)
        filter_layout.addLayout(frame_row)

        self._filtered_selection_info = QLabel("Filtered result: 0 paths")
        self._filtered_selection_info.setStyleSheet("color: #6b7280; font-size: 11px;")
        filter_layout.addWidget(self._filtered_selection_info)

        filter_actions = QHBoxLayout()
        filter_actions.setSpacing(6)
        self._apply_filter_btn = QPushButton("Select Filtered")
        self._apply_filter_btn.setFixedHeight(28)
        self._apply_filter_btn.clicked.connect(self._emit_path_filter_request)
        filter_actions.addWidget(self._apply_filter_btn)
        self._revert_filter_btn = QPushButton("Revert")
        self._revert_filter_btn.setFixedHeight(28)
        self._revert_filter_btn.setEnabled(False)
        self._revert_filter_btn.clicked.connect(self._emit_revert_request)
        filter_actions.addWidget(self._revert_filter_btn)
        self._clear_filter_btn = QPushButton("Clear")
        self._clear_filter_btn.setFixedHeight(28)
        self._clear_filter_btn.clicked.connect(self._clear_path_filter_inputs)
        filter_actions.addWidget(self._clear_filter_btn)
        filter_actions.addStretch()
        filter_layout.addLayout(filter_actions)

        side_layout.addWidget(filter_frame)

        self._content_splitter.addWidget(chart_column)
        self._content_splitter.addWidget(side_panel)
        self._content_splitter.setStretchFactor(0, 5)
        self._content_splitter.setStretchFactor(1, 2)
        self._content_splitter.setSizes([920, 340])
        layout.addWidget(self._content_splitter, 1)
        self._update_residue_summary_panel(None, False)
        self._selected_chart_path_ids: list[int] = []
        self._selection_history: list[list[int]] = []

    @property
    def profile_chart(self) -> ProfileChart:
        return self._profile_chart

    @property
    def statistics_chart(self) -> StatisticsChart:
        return self._statistics_chart

    def set_display_config(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
    ):
        self._profile_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
            emit_signal=False,
        )
        self._statistics_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
        )

    def set_profiles(self, profiles):
        self._profile_chart.set_profiles(profiles)
        self._statistics_chart.set_dataset_filter(self._profile_chart.selected_dataset_prefixes())
        self._statistics_chart.set_profiles(profiles)

    def set_path_color_map(self, path_color_map):
        self._profile_chart.set_path_color_map(path_color_map)
        self._statistics_chart.set_path_color_map(path_color_map)

    def set_appearance(self, alpha_scale: float, depth_scale: float):
        self._profile_chart.set_appearance(
            alpha_scale,
            depth_scale,
            emit_signal=False,
        )

    def appearance(self):
        return self._profile_chart.appearance()

    def clear_charts(self):
        self._profile_chart.clear()
        self._statistics_chart.clear()
        self._update_residue_summary_panel(None, False)
        self.set_chart_selected_paths([])

    def show_and_focus(self):
        self.show()
        self.raise_()
        self.activateWindow()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_residue_summary_sections()
        self._refresh_render_surfaces()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            QTimer.singleShot(0, self._refresh_render_surfaces)

    def _refresh_render_surfaces(self):
        for widget in (
            getattr(self, "_profile_chart", None),
            getattr(self, "_statistics_chart", None),
            getattr(self, "_content_splitter", None),
            getattr(self, "_splitter", None),
        ):
            if widget is None:
                continue
            try:
                widget.update()
                widget.repaint()
            except Exception:
                pass

    @staticmethod
    def _blend_with_white(base: tuple[int, int, int], strength: float) -> QColor:
        strength = max(0.0, min(1.0, strength))
        red = int(round(255 + (base[0] - 255) * strength))
        green = int(round(255 + (base[1] - 255) * strength))
        blue = int(round(255 + (base[2] - 255) * strength))
        return QColor(red, green, blue)

    @classmethod
    def _rank_change_color(cls, rank_delta: int, max_abs_delta: int) -> QColor:
        if max_abs_delta <= 0 or rank_delta == 0:
            return QColor("#FFFFFF")
        strength = abs(rank_delta) / max_abs_delta
        strength = 0.28 + strength * 0.68
        if rank_delta > 0:
            return cls._blend_with_white((211, 76, 76), strength)
        return cls._blend_with_white((70, 116, 196), strength)

    @classmethod
    def _new_rank_color(cls) -> QColor:
        return cls._blend_with_white((103, 194, 58), 0.72)

    @classmethod
    def _metric_heatmap_color(cls, value: float, max_value: float) -> QColor:
        if max_value <= 1e-12 or value <= 1e-12:
            return QColor("#FFFFFF")
        strength = value / max_value
        strength = 0.20 + strength * 0.72
        return cls._blend_with_white((64, 158, 255), strength)

    def _sort_residue_summary_rows(self, rows, sort_key: str, descending: bool):
        def sort_value(row):
            if sort_key == "residue_id":
                return int(row.get("localResidueId", row.get("residueId", 0)))
            if sort_key == "path_count":
                return int(row.get("pathCount", 0))
            if sort_key == "avg_radius":
                value = row.get("avgRadius")
                return float(value) if value is not None else float("-inf")
            if sort_key == "change":
                return str(row.get("rank_change_text", "") or "")
            return int(row.get("localResidueId", row.get("residueId", 0)))

        sorted_rows = list(rows)
        sorted_rows.sort(
            key=lambda row: (
                sort_value(row),
                int(row.get("localResidueId", row.get("residueId", 0))),
                int(row.get("pathCount", 0)),
            ),
            reverse=descending,
        )
        return sorted_rows

    def _local_residue_id(self, residue_id: int) -> int:
        try:
            residue_id = int(residue_id)
        except (TypeError, ValueError):
            return 0
        if self.db is not None and hasattr(self.db, "_decode_residue_id"):
            dataset, local_id = self.db._decode_residue_id(residue_id)
            if dataset is not None and local_id is not None:
                return int(local_id)
        return residue_id

    def _ordered_summary_dataset_prefixes(self, grouped_rows: dict[str, list[dict]]) -> list[str]:
        prefixes = set(grouped_rows.keys())
        ordered: list[str] = []
        if self.db is not None and hasattr(self.db, "list_datasets"):
            for dataset in self.db.list_datasets():
                prefix = str(dataset.get("prefix", "") or "")
                if prefix in prefixes and prefix not in ordered:
                    ordered.append(prefix)
        for prefix in sorted(prefixes):
            if prefix not in ordered:
                ordered.append(prefix)
        return ordered

    def _annotate_residue_summary_ranks(self, grouped_rows: dict[str, list[dict]], ordered_prefixes: list[str]):
        rank_maps: dict[str, dict[int, int]] = {}
        max_rank_delta = 0
        for dataset_prefix in ordered_prefixes:
            rank_rows = sorted(
                grouped_rows.get(dataset_prefix, []),
                key=lambda row: (-int(row.get("pathCount", 0)), int(row.get("localResidueId", row.get("residueId", 0)))),
            )
            dataset_rank_map: dict[int, int] = {}
            last_path_count = None
            current_rank = 0
            for index, row in enumerate(rank_rows, start=1):
                path_count = int(row.get("pathCount", 0))
                if last_path_count is None or path_count != last_path_count:
                    current_rank = index
                    last_path_count = path_count
                dataset_rank_map[int(row.get("localResidueId", row.get("residueId", 0)))] = current_rank
            rank_maps[dataset_prefix] = dataset_rank_map

        compare_targets: dict[str, str] = {}
        if len(ordered_prefixes) == 2:
            left, right = ordered_prefixes
            compare_targets[left] = right
            compare_targets[right] = left

        for dataset_prefix in ordered_prefixes:
            target_prefix = compare_targets.get(dataset_prefix, "")
            target_ranks = rank_maps.get(target_prefix, {})
            for row in grouped_rows.get(dataset_prefix, []):
                local_residue_id = int(row.get("localResidueId", row.get("residueId", 0)))
                rank = int(rank_maps.get(dataset_prefix, {}).get(local_residue_id, 0))
                row["rank_change_delta"] = 0
                row["rank_change_text"] = ""
                row["rank_change_kind"] = "none"
                if not target_prefix:
                    continue
                other_rank = target_ranks.get(local_residue_id)
                if other_rank is None:
                    row["rank_change_text"] = "NEW"
                    row["rank_change_kind"] = "new"
                elif rank < other_rank:
                    delta = other_rank - rank
                    row["rank_change_delta"] = delta
                    row["rank_change_text"] = f"UP{delta}"
                    row["rank_change_kind"] = "up"
                    max_rank_delta = max(max_rank_delta, delta)
                elif rank > other_rank:
                    delta = rank - other_rank
                    row["rank_change_delta"] = -delta
                    row["rank_change_text"] = f"DOWN{delta}"
                    row["rank_change_kind"] = "down"
                    max_rank_delta = max(max_rank_delta, delta)
                else:
                    row["rank_change_text"] = "SAME"
                    row["rank_change_kind"] = "same"

        for dataset_prefix in ordered_prefixes:
            for row in grouped_rows.get(dataset_prefix, []):
                change_kind = str(row.get("rank_change_kind", "none") or "none")
                if change_kind == "new":
                    row["rank_change_color"] = self._new_rank_color()
                else:
                    row["rank_change_color"] = self._rank_change_color(
                        int(row.get("rank_change_delta", 0)),
                        max_rank_delta,
                    )

    def _create_residue_summary_table(self, dataset_prefix: str, selected_kind: str | None = None) -> QTableWidget:
        table = QTableWidget(0, 4)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.setWordWrap(False)
        table.setStyleSheet(
            "QTableWidget {"
            "background: #fbfcfe; border: 1px solid #e5ecf6; border-radius: 10px; "
            "color: #303133; font-size: 11px;"
            "}"
        )
        table.setHorizontalHeaderLabels(["Residue", "Paths", "Mean Radius", "Change"])
        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionsClickable(True)
        header.sectionClicked.connect(
            lambda section, prefix=dataset_prefix, kind=selected_kind: self._on_residue_summary_header_clicked(prefix, section, kind)
        )
        return table

    def _populate_residue_summary_table(self, table: QTableWidget, rows):
        table.clearContents()
        table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            rank_change = str(row.get("rank_change_text", "") or "")
            values = (
                str(int(row.get("localResidueId", row.get("residueId", 0)))),
                str(int(row.get("pathCount", 0))),
                f"{float(row.get('avgRadius', 0.0)):.2f} Å" if row.get("avgRadius") is not None else "-",
                rank_change or "-",
            )
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                if col_idx == 3:
                    background = row.get("rank_change_color")
                    if isinstance(background, QColor):
                        item.setBackground(QBrush(background))
                table.setItem(row_idx, col_idx, item)
        table.resizeRowsToContents()
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _resize_residue_summary_sections(self):
        section_count = len(self._residue_summary_sections)
        if section_count <= 0:
            return
        viewport_height = max(0, self._residue_summary_scroll.viewport().height())
        spacing = self._residue_summary_layout.spacing()
        visible_count = min(2, section_count)
        total_spacing = max(0, visible_count - 1) * spacing
        target_height = max(220, (viewport_height - total_spacing) // 2)
        for section in self._residue_summary_sections.values():
            section.setMinimumHeight(target_height)
            section.setMaximumHeight(target_height)

    def _current_residue_summary_grouped_rows(self) -> tuple[list[str], dict[str, list[dict]]]:
        return self._residue_summary_grouped_rows_for_kind(
            str(self._residue_summary_mode_combo.currentData() or "path")
        )

    def _residue_summary_dataset_prefixes(self) -> list[str]:
        prefixes = {
            str(row.get("dataset_prefix") or "Dataset")
            for row in self._residue_summary_rows
        }
        ordered: list[str] = []
        if self.db is not None and hasattr(self.db, "list_datasets"):
            for dataset in self.db.list_datasets():
                prefix = str(dataset.get("prefix", "") or "")
                if prefix in prefixes and prefix not in ordered:
                    ordered.append(prefix)
        for prefix in sorted(prefixes):
            if prefix not in ordered:
                ordered.append(prefix)
        return ordered

    def _residue_summary_grouped_rows_for_kind(self, selected_kind: str) -> tuple[list[str], dict[str, list[dict]]]:
        grouped_rows: dict[str, list[dict]] = {}
        for row in self._residue_summary_rows:
            if str(row.get("summary_kind") or "path") != selected_kind:
                continue
            copied = dict(row)
            copied["localResidueId"] = self._local_residue_id(copied.get("residueId", 0))
            dataset_prefix = str(copied.get("dataset_prefix") or "Dataset")
            grouped_rows.setdefault(dataset_prefix, []).append(copied)
        ordered_prefixes = self._ordered_summary_dataset_prefixes(grouped_rows)
        self._annotate_residue_summary_ranks(grouped_rows, ordered_prefixes)
        return ordered_prefixes, grouped_rows

    def _build_residue_summary_heatmap_payload(self) -> dict:
        ordered_prefixes, grouped_rows = self._current_residue_summary_grouped_rows()
        selected_kind = str(self._residue_summary_mode_combo.currentData() or "path")
        if not ordered_prefixes:
            return {"datasets": [], "rows": [], "kind": selected_kind}

        dataset_maps = {
            prefix: {
                int(row.get("localResidueId", row.get("residueId", 0))): row
                for row in grouped_rows.get(prefix, [])
            }
            for prefix in ordered_prefixes
        }
        compare_prefixes = ordered_prefixes[:2]
        if len(compare_prefixes) < 2:
            compare_prefixes = ordered_prefixes[:1]
        axis_residues = sorted(
            {
                residue_id
                for prefix in compare_prefixes
                for residue_id in dataset_maps.get(prefix, {}).keys()
            }
        )
        if not axis_residues:
            return {"datasets": [], "rows": [], "kind": selected_kind}
        dataset_totals = {
            prefix: max(
                [
                    int(row.get("totalPaths", 0) or 0)
                    for row in grouped_rows.get(prefix, [])
                ]
                or [0]
            )
            for prefix in ordered_prefixes
        }

        rows = []
        base_prefix = compare_prefixes[0]
        compare_prefix = compare_prefixes[-1]
        base_map = dataset_maps.get(base_prefix, {})
        compare_map = dataset_maps.get(compare_prefix, {})
        rank_maps = {}
        for prefix in compare_prefixes:
            ranked_rows = sorted(
                dataset_maps.get(prefix, {}).values(),
                key=lambda row: (-int(row.get("pathCount", 0) or 0), int(row.get("localResidueId", row.get("residueId", 0)))),
            )
            ranks = {}
            last_count = None
            current_rank = 0
            for index, row in enumerate(ranked_rows, start=1):
                path_count = int(row.get("pathCount", 0) or 0)
                if last_count is None or path_count != last_count:
                    current_rank = index
                    last_count = path_count
                ranks[int(row.get("localResidueId", row.get("residueId", 0)))] = current_rank
            rank_maps[prefix] = ranks
        base_total = max(0, int(dataset_totals.get(base_prefix, 0) or 0))
        compare_total = max(0, int(dataset_totals.get(compare_prefix, 0) or 0))
        delta_rows = []
        for residue_id in axis_residues:
            base_row = base_map.get(residue_id)
            compare_row = compare_map.get(residue_id)
            base_count = int(base_row.get("pathCount", 0) or 0) if base_row else 0
            compare_count = int(compare_row.get("pathCount", 0) or 0) if compare_row else 0
            base_percent = (base_count / base_total * 100.0) if base_total > 0 else 0.0
            compare_percent = (compare_count / compare_total * 100.0) if compare_total > 0 else 0.0
            delta_count = compare_count - base_count
            delta_percent = compare_percent - base_percent
            base_rank = rank_maps.get(base_prefix, {}).get(residue_id)
            compare_rank = rank_maps.get(compare_prefix, {}).get(residue_id)
            if base_row is None and compare_row is not None:
                status = "new"
            elif base_rank is not None and compare_rank is not None and compare_rank < base_rank:
                status = "up"
            elif base_rank is not None and compare_rank is not None and compare_rank > base_rank:
                status = "down"
            elif base_row is not None and compare_row is None:
                status = "down"
            else:
                status = "same"
            delta_rows.append(
                {
                    "residue_id": residue_id,
                    "base_count": base_count,
                    "base_percent": base_percent,
                    "compare_count": compare_count,
                    "compare_percent": compare_percent,
                    "delta_count": delta_count,
                    "delta_percent": delta_percent,
                    "status": status,
                    "base_rank": base_rank,
                    "compare_rank": compare_rank,
                }
            )
        delta_rows.sort(
            key=lambda row: (
                0 if row["status"] == "new" else 1,
                -abs(float(row["delta_percent"])),
                -abs(int(row["delta_count"])),
                int(row["residue_id"]),
            )
        )
        for row_residue_id in axis_residues:
            cells = {}
            for col_residue_id in axis_residues:
                if row_residue_id != col_residue_id:
                    cells[col_residue_id] = {"blank": True}
                    continue
                base_row = base_map.get(row_residue_id)
                compare_row = compare_map.get(col_residue_id)
                base_count = int(base_row.get("pathCount", 0) or 0) if base_row else 0
                compare_count = int(compare_row.get("pathCount", 0) or 0) if compare_row else 0
                percent = (compare_count / compare_total * 100.0) if compare_total > 0 else 0.0
                delta = compare_count - base_count
                if base_row is None and compare_row is not None:
                    change_kind = "new"
                elif delta > 0:
                    change_kind = "up"
                else:
                    change_kind = "down"
                cells[col_residue_id] = {
                    "count": compare_count,
                    "percent": percent,
                    "change_kind": change_kind,
                    "change_delta": delta,
                    "base_count": base_count,
                    "compare_count": compare_count,
                    "total_paths": compare_total,
                }
            rows.append({"residue_id": row_residue_id, "cells": cells})
        return {
            "datasets": compare_prefixes,
            "axis_residues": axis_residues,
            "delta_rows": delta_rows,
            "rows": rows,
            "kind": selected_kind,
        }

    def _refresh_residue_summary_table(self):
        while self._residue_summary_layout.count():
            item = self._residue_summary_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._residue_summary_sections = {}
        self._residue_summary_tables = {}
        dataset_prefixes = self._residue_summary_dataset_prefixes()
        single_dataset_mode = len(dataset_prefixes) == 1
        self._residue_summary_view_label.setVisible(not single_dataset_mode)
        self._residue_summary_mode_combo.setVisible(not single_dataset_mode)
        self._residue_summary_heatmap_btn.setVisible(not single_dataset_mode)

        if single_dataset_mode:
            dataset_prefix = dataset_prefixes[0]
            for selected_kind, title_suffix in (("path", "Path Residues"), ("bottleneck", "Bottleneck")):
                _ordered_prefixes, grouped_rows = self._residue_summary_grouped_rows_for_kind(selected_kind)
                rows = grouped_rows.get(dataset_prefix, [])
                if not rows:
                    continue
                self._add_residue_summary_section(
                    dataset_prefix,
                    selected_kind,
                    rows,
                    title=f"{dataset_prefix} - {title_suffix}",
                )
        else:
            selected_kind = str(self._residue_summary_mode_combo.currentData() or "path")
            ordered_prefixes, grouped_rows = self._current_residue_summary_grouped_rows()
            for dataset_prefix in ordered_prefixes:
                self._add_residue_summary_section(
                    dataset_prefix,
                    selected_kind,
                    grouped_rows[dataset_prefix],
                    title=dataset_prefix,
                )

        self._residue_summary_layout.addStretch()
        self._resize_residue_summary_sections()
        self._refresh_residue_summary_heatmap_dialog()

    def _add_residue_summary_section(self, dataset_prefix: str, selected_kind: str, rows: list[dict], *, title: str):
        sort_state_key = f"{selected_kind}:{dataset_prefix}"
        if sort_state_key not in self._residue_summary_sort_state:
            self._residue_summary_sort_state[sort_state_key] = ("path_count", True)
        sort_key, descending = self._residue_summary_sort_state[sort_state_key]
        sorted_rows = self._sort_residue_summary_rows(rows, sort_key, descending)

        section = QFrame()
        section.setStyleSheet("QFrame{border:1px solid #e5ecf6;border-radius:10px;background:#ffffff;}")
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(10, 10, 10, 10)
        section_layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #1f2937;")
        section_layout.addWidget(title_label)

        table = self._create_residue_summary_table(dataset_prefix, selected_kind)
        table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._populate_residue_summary_table(table, sorted_rows)
        section_key = f"{selected_kind}:{dataset_prefix}"
        self._residue_summary_sections[section_key] = section
        self._residue_summary_tables[section_key] = table
        section_layout.addWidget(table)
        self._residue_summary_layout.addWidget(section)

    def _on_residue_summary_header_clicked(self, dataset_prefix: str, section: int, selected_kind: str | None = None):
        sort_keys = {
            0: "residue_id",
            1: "path_count",
            2: "avg_radius",
            3: "change",
        }
        next_key = sort_keys.get(int(section), "residue_id")
        selected_kind = str(selected_kind or self._residue_summary_mode_combo.currentData() or "path")
        sort_state_key = f"{selected_kind}:{dataset_prefix}"
        current_key, current_desc = self._residue_summary_sort_state.get(sort_state_key, ("residue_id", False))
        descending = not current_desc if current_key == next_key else False
        self._residue_summary_sort_state[sort_state_key] = (next_key, descending)
        self._refresh_residue_summary_table()

    def _update_residue_summary_panel(self, rows, is_active: bool):
        if is_active:
            self._residue_summary_rows = list(rows or [])
        else:
            profile_chart = getattr(self, "_profile_chart", None)
            if profile_chart is not None and hasattr(profile_chart, "all_residue_summary_rows"):
                self._residue_summary_rows = profile_chart.all_residue_summary_rows()
            else:
                self._residue_summary_rows = []
        self._refresh_residue_summary_table()

    def _open_residue_summary_heatmap(self):
        if self._residue_summary_heatmap_dialog is None:
            self._residue_summary_heatmap_dialog = _ResidueSummaryHeatmapDialog(self)
            self._residue_summary_heatmap_dialog.destroyed.connect(self._on_residue_summary_heatmap_destroyed)
        self._refresh_residue_summary_heatmap_dialog()
        self._residue_summary_heatmap_dialog.show()
        self._residue_summary_heatmap_dialog.raise_()
        self._residue_summary_heatmap_dialog.activateWindow()

    def _refresh_residue_summary_heatmap_dialog(self):
        if self._residue_summary_heatmap_dialog is None:
            return
        self._residue_summary_heatmap_dialog.refresh(self._build_residue_summary_heatmap_payload())

    def _on_residue_summary_heatmap_destroyed(self, *_args):
        self._residue_summary_heatmap_dialog = None

    def set_chart_selected_paths(self, path_ids):
        self._selected_chart_path_ids = [int(path_id) for path_id in (path_ids or [])]
        if self._selected_chart_path_ids:
            self._chart_selection_info.setText(f"Current box selection: {len(self._selected_chart_path_ids)} paths")
        else:
            self._chart_selection_info.setText("No chart-path selection.")
        self._update_filtered_selection_preview()


    def reset_filter_history(self):
        self._selection_history = []
        self._revert_filter_btn.setEnabled(False)

    def _parse_optional_float(self, text: str) -> Optional[float]:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _parse_optional_int(self, text: str) -> Optional[int]:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            return int(float(text))
        except ValueError:
            return None

    def _current_filter_payload(self) -> dict:
        return {
            "hydrophobicity_min": self._parse_optional_float(self._hydro_min_edit.text()),
            "hydrophobicity_max": self._parse_optional_float(self._hydro_max_edit.text()),
            "frame_min": self._parse_optional_int(self._frame_min_edit.text()),
            "frame_max": self._parse_optional_int(self._frame_max_edit.text()),
        }

    def _update_filtered_selection_preview(self):
        if not self._selected_chart_path_ids:
            self._filtered_selection_info.setText("Filtered result: 0 paths")
            return
        filtered_ids = self._profile_chart.filter_path_ids(
            self._selected_chart_path_ids,
            **self._current_filter_payload(),
        )
        self._filtered_selection_info.setText(f"Filtered result: {len(filtered_ids)} paths")

    def _emit_path_filter_request(self):
        if self._selected_chart_path_ids:
            current = [int(path_id) for path_id in self._selected_chart_path_ids]
            if not self._selection_history or self._selection_history[-1] != current:
                self._selection_history.append(current)
        self._revert_filter_btn.setEnabled(bool(self._selection_history))
        payload = self._current_filter_payload()
        payload["path_ids"] = self._profile_chart.filter_path_ids(
            self._selected_chart_path_ids,
            **payload,
        )
        self.path_filter_apply_requested.emit(payload)
        self._filtered_selection_info.setText(f"Filtered result: {len(payload['path_ids'])} paths")

    def _emit_revert_request(self):
        if not self._selection_history:
            return
        previous = self._selection_history.pop()
        self._revert_filter_btn.setEnabled(bool(self._selection_history))
        self.path_filter_revert_requested.emit(previous)

    def _clear_path_filter_inputs(self):
        for edit in (
            self._hydro_min_edit,
            self._hydro_max_edit,
            self._frame_min_edit,
            self._frame_max_edit,
        ):
            edit.clear()
        self._update_filtered_selection_preview()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, db_path: str | None = None):
        super().__init__()
        self.setWindowTitle("TopoTunnel")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)

        # Core objects
        initial_paths = [db_path] if db_path else []
        self.db = MultiDatasetDatabase(initial_paths)
        self.db.connect()
        self.state = AppState()
        self._protein_path: Optional[str] = None
        self._pse_path: Optional[str] = None
        self._pse_extracted_pdb_path: Optional[str] = None
        self._protein_observer_path: Optional[str] = None
        self._observer_pending_payload: dict = {}
        self._protein_viewer_panel = None
        self._protein_viewer_panel_top = None
        self._protein_viewer_panel_bottom = None
        self._chart_profiles = []
        self._chart_workspace: Optional[ChartWorkspaceDialog] = None
        self._charts_detached = False
        self._attached_chart_sizes = [600, 200]
        self._path_coordinate_mode = "current"
        self._path_coordinate_load_busy = False
        self._path_change_overlay_active = False
        self._path_change_metric_cache_signature: tuple = ()
        self._path_change_metric_cache: dict[int, float] = {}
        self._exit_delta_overlay_active = False
        self._exit_delta_metric_cache_signature: tuple = ()
        self._exit_delta_metric_cache: dict[int, float] = {}
        self._path_delta_excluded_path_ids: set[int] = set()
        self._path_delta_filter_threshold: float = 0.0
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids: set[int] = set()
        self._contained_match_last_compare_summary = ""
        self._suppress_next_path_preview = False
        self._3d_load_request_id = 0
        self._residue_combo_task_request_id = 0
        self._residue_combo_task_thread: QThread | None = None
        self._residue_combo_task_worker: ResidueCombinationTaskWorker | None = None
        self._dataset_path_cache = {}
        self._dataset_residue_pair_csv_path_cache: dict[tuple[str, str], str] = {}
        self._residue_combo_present_frame_cache = {}
        self._primary_residue_position_cache = {}
        self._dataset_residue_position_cache: dict[str, dict[int, tuple[float, float, float]]] = {}
        self._last_3d_overlay_signature = None
        self._last_effective_render_signature = None
        self._last_entry_exit_signature = None
        self._last_observer_overlay_signature = None
        self._top_controls_widget: Optional[QWidget] = None
        self._top_controls_layout: Optional[QGridLayout] = None
        self._top_controls_items: list[QWidget] = []
        self._color_label: Optional[QLabel] = None
        self._bg_label: Optional[QLabel] = None
        self._loading_depth = 0
        self._chart_operation_history: list[list[int]] = []
        self._statistics_last_rows: list[dict] = []
        self._statistics_last_headers: list[str] = []
        # Derived from the Datasets tab once the control-panel tabs are built.
        self._side_dock_min_width = 0
        self._side_dock_width = 0

        # Build UI
        self._build_toolbar()
        self._build_central()
        self._build_side_panel()
        self._relocate_top_controls()
        self._build_bottom_panel()
        self._build_statusbar()

        # Wire signals
        self._connect_signals()

        # Load initial data
        QTimer.singleShot(100, self._init_data)

    def _invalidate_render_signatures(self) -> None:
        self._last_3d_overlay_signature = None
        self._last_effective_render_signature = None
        self._last_entry_exit_signature = None
        self._last_observer_overlay_signature = None
        self._path_change_metric_cache_signature = ()
        self._exit_delta_metric_cache_signature = ()

    # ─── Toolbar ────────────────────────────────────────
    def _configure_toolbar_combo(
        self,
        combo: QComboBox,
        *,
        width: Optional[int] = None,
        min_contents: int = 0,
    ):
        """Display-only tuning for compact toolbar combos and their popups."""
        combo.setView(QListView(combo))
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
        if min_contents > 0:
            combo.setMinimumContentsLength(min_contents)
        if width is not None:
            combo.setFixedWidth(width)
        view = combo.view()
        if view is not None:
            view.setTextElideMode(Qt.ElideNone)
            view.setMinimumWidth(max(combo.width(), combo.sizeHint().width() + 24))

    def _refresh_combo_popup_width(self, combo: QComboBox, extra: int = 40):
        """Ensure popup is wide enough to show full option text."""
        view = combo.view()
        if view is None:
            return
        metrics = QFontMetrics(view.font())
        content_width = 0
        for i in range(combo.count()):
            content_width = max(content_width, metrics.horizontalAdvance(combo.itemText(i)))
        popup_width = max(combo.width(), content_width + extra)
        view.setMinimumWidth(popup_width)

    def _build_toolbar(self):
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar{spacing:3px;padding:2px 4px;}")
        self.addToolBar(tb)
        self._main_toolbar = tb

        _lbl = "font-size:11px;color:#909399;padding:0 1px;"
        _sm = "padding:3px 7px;font-size:11px;min-height:18px;"

        # ── 1  Display ───────────────────────────────────
        lbl = QLabel("Color")
        lbl.setStyleSheet(_lbl); lbl.setToolTip("Background path color (light gray ↔ black)")
        tb.addWidget(lbl)
        self._path_color_slider = QSlider(Qt.Horizontal)
        self._path_color_slider.setRange(0, 100); self._path_color_slider.setValue(35)
        self._path_color_slider.setFixedWidth(55)
        self._path_color_slider.setToolTip("Background path color: left=light gray, right=black")
        self._path_color_slider.valueChanged.connect(self._on_path_color_changed)
        tb.addWidget(self._path_color_slider)

        lbl = QLabel("BG")
        lbl.setStyleSheet(_lbl); lbl.setToolTip("Background opacity")
        tb.addWidget(lbl)
        self._bg_opacity_slider = QSlider(Qt.Horizontal)
        self._bg_opacity_slider.setRange(1, 50); self._bg_opacity_slider.setValue(15)
        self._bg_opacity_slider.setFixedWidth(55); self._bg_opacity_slider.setToolTip("Background opacity")
        self._bg_opacity_slider.valueChanged.connect(self._on_bg_opacity_changed)
        tb.addWidget(self._bg_opacity_slider)

        self._path_change_btn = QPushButton("Path Δ")
        self._path_change_btn.setCheckable(True)
        self._path_change_btn.setStyleSheet(_sm)
        self._path_change_btn.setToolTip("Color paths by original/current coordinate distance; red means larger change")
        self._path_change_btn.toggled.connect(self._toggle_path_change_overlay)
        tb.addWidget(self._path_change_btn)

        self._exit_delta_btn = QPushButton("Exit Δ")
        self._exit_delta_btn.setCheckable(True)
        self._exit_delta_btn.setStyleSheet(_sm)
        self._exit_delta_btn.setToolTip(
            "Color paths by the difference between current and original exit-to-cluster-center distances"
        )
        self._exit_delta_btn.toggled.connect(self._toggle_exit_delta_overlay)
        tb.addWidget(self._exit_delta_btn)

        self._path_delta_filter_slider = QSlider(Qt.Horizontal)
        self._path_delta_filter_slider.setRange(0, 100)
        self._path_delta_filter_slider.setValue(0)
        self._path_delta_filter_slider.setFixedWidth(84)
        self._path_delta_filter_slider.setEnabled(False)
        self._path_delta_filter_slider.setTracking(False)
        self._path_delta_filter_slider.setToolTip("Exclude paths with the active Δ value at or below this threshold")
        self._path_delta_filter_slider.valueChanged.connect(self._on_path_delta_filter_changed)
        tb.addWidget(self._path_delta_filter_slider)

        self._path_delta_filter_label = QLabel("Δ off")
        self._path_delta_filter_label.setStyleSheet(_lbl)
        self._path_delta_filter_label.setMinimumWidth(84)
        self._path_delta_filter_label.setToolTip("Active Δ exclusion threshold")
        tb.addWidget(self._path_delta_filter_label)

        tb.addSeparator()

        # ── 2  Layers ────────────────────────────────────
        self._residue_toggle_btn = QPushButton("Residues")
        self._residue_toggle_btn.setCheckable(True); self._residue_toggle_btn.setStyleSheet(_sm)
        self._residue_toggle_btn.setToolTip("Show/hide residue 3D positions")
        self._residue_toggle_btn.toggled.connect(self._toggle_residue_positions)
        tb.addWidget(self._residue_toggle_btn)

        self._residue_labels_btn = QPushButton("Res Labels")
        self._residue_labels_btn.setCheckable(True); self._residue_labels_btn.setStyleSheet(_sm)
        self._residue_labels_btn.setToolTip("Show/hide labels for currently selected residues")
        self._residue_labels_btn.toggled.connect(self._toggle_residue_labels)
        tb.addWidget(self._residue_labels_btn)

        self._entry_exit_btn = QPushButton("Entry/Exit")
        self._entry_exit_btn.setCheckable(True); self._entry_exit_btn.setStyleSheet(_sm)
        self._entry_exit_btn.setToolTip("Show/hide entrance (green) and exit (red) points")
        self._entry_exit_btn.toggled.connect(self._toggle_entry_exit_points)
        tb.addWidget(self._entry_exit_btn)

        self._entry_exit_scope_btn = QPushButton("EE: All")
        self._entry_exit_scope_btn.setCheckable(True); self._entry_exit_scope_btn.setStyleSheet(_sm)
        self._entry_exit_scope_btn.setToolTip("Toggle entry/exit scope: all paths or current paths")
        self._entry_exit_scope_btn.toggled.connect(self._toggle_entry_exit_scope)
        tb.addWidget(self._entry_exit_scope_btn)

        self._exit_cluster_btn = QPushButton("Exit Cluster")
        self._exit_cluster_btn.setCheckable(True); self._exit_cluster_btn.setStyleSheet(_sm)
        self._exit_cluster_btn.setToolTip("Click an exit sphere to select all visually connected exit spheres at the current zoom")
        self._exit_cluster_btn.toggled.connect(self._toggle_exit_cluster_select)
        tb.addWidget(self._exit_cluster_btn)

        self._focus_btn = QPushButton("Focus")
        self._focus_btn.setCheckable(True); self._focus_btn.setStyleSheet(_sm)
        self._focus_btn.setToolTip("Focus: hide background, show only selected paths")
        self._focus_btn.toggled.connect(self._toggle_focus_mode)
        tb.addWidget(self._focus_btn)

        tb.addSeparator()

        # ── 3  Protein (compact) ─────────────────────────
        self._load_protein_btn = QPushButton("Protein…")
        self._load_protein_btn.setStyleSheet(_sm)
        self._load_protein_btn.setToolTip("Load protein PDB overlay")
        self._load_protein_btn.clicked.connect(self._on_load_protein)
        tb.addWidget(self._load_protein_btn)

        self._load_observer_protein_btn = QPushButton("Residue Observer")
        self._load_observer_protein_btn.setCheckable(True)
        self._load_observer_protein_btn.setChecked(False)
        self._load_observer_protein_btn.setStyleSheet(_sm)
        self._load_observer_protein_btn.setToolTip("Show or hide the right Residue Observer panel")
        self._load_observer_protein_btn.clicked.connect(self._toggle_protein_observer_panel)
        tb.addWidget(self._load_observer_protein_btn)

        self._protein_toggle_btn = QPushButton("Reference PDB")
        self._protein_toggle_btn.setCheckable(True)
        self._protein_toggle_btn.setEnabled(False)
        self._protein_toggle_btn.setStyleSheet(_sm)
        self._protein_toggle_btn.setToolTip("Show/hide the loaded reference PDB overlay")
        self._protein_toggle_btn.toggled.connect(self._toggle_protein_visible)
        tb.addWidget(self._protein_toggle_btn)

        self._protein_style_combo = QComboBox()
        self._protein_style_combo.addItems(["cartoon", "backbone", "tube", "ca_spheres"])
        self._protein_style_combo.setCurrentText("cartoon")
        self._protein_style_combo.setFixedHeight(24)
        self._protein_style_combo.setMaxVisibleItems(10)
        self._configure_toolbar_combo(self._protein_style_combo, width=108, min_contents=10)
        self._refresh_combo_popup_width(self._protein_style_combo)
        self._protein_style_combo.setEnabled(False)
        self._protein_style_combo.setToolTip("Protein style")
        self._protein_style_combo.currentTextChanged.connect(self._on_protein_style_changed)
        tb.addWidget(self._protein_style_combo)

        self._protein_align_combo = QComboBox()
        self._protein_align_combo.addItem("kabsch", "auto_kabsch")
        self._protein_align_combo.addItem("translate", "translate")
        self._protein_align_combo.addItem("none", "none")
        self._protein_align_combo.setCurrentIndex(0)
        self._protein_align_combo.setFixedHeight(24)
        self._protein_align_combo.setMaxVisibleItems(10)
        self._configure_toolbar_combo(self._protein_align_combo, width=104, min_contents=10)
        self._refresh_combo_popup_width(self._protein_align_combo)
        self._protein_align_combo.setToolTip("Alignment mode")
        self._protein_align_combo.currentIndexChanged.connect(self._on_protein_align_mode_changed)
        tb.addWidget(self._protein_align_combo)

        self._protein_opacity_slider = QSlider(Qt.Horizontal)
        self._protein_opacity_slider.setRange(100, 100)
        self._protein_opacity_slider.setValue(100)
        self._protein_opacity_slider.setFixedWidth(40)
        self._protein_opacity_slider.setEnabled(False)
        self._protein_opacity_slider.setToolTip("Protein opacity")
        self._protein_opacity_slider.valueChanged.connect(self._on_protein_opacity_changed)
        tb.addWidget(self._protein_opacity_slider)

        self._clear_protein_btn = QPushButton("Clear")
        self._clear_protein_btn.setProperty("flat", True)
        self._clear_protein_btn.setEnabled(False)
        self._clear_protein_btn.setStyleSheet(_sm)
        self._clear_protein_btn.setToolTip("Remove protein overlay")
        self._clear_protein_btn.clicked.connect(self._on_clear_protein)
        tb.addWidget(self._clear_protein_btn)

        self._binding_sites_btn = QPushButton("Sites")
        self._binding_sites_btn.setCheckable(True)
        self._binding_sites_btn.setChecked(True)
        self._binding_sites_btn.setEnabled(False)
        self._binding_sites_btn.setStyleSheet(_sm)
        self._binding_sites_btn.setToolTip("Show/hide binding site surfaces")
        self._binding_sites_btn.toggled.connect(self._toggle_binding_sites)
        tb.addWidget(self._binding_sites_btn)

        tb.addSeparator()

        # ── 4  Frame (collapsed popup) ───────────────────
        self._frame_menu_btn = QToolButton()
        self._frame_menu_btn.setText("Frame ▾")
        self._frame_menu_btn.setFixedHeight(24)
        self._frame_menu_btn.setToolTip("Set frame range filter")
        self._frame_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._frame_menu_btn.setStyleSheet(
            "QToolButton{padding:3px 7px;font-size:11px;}"
            "QToolButton::menu-indicator{image:none;}"
        )
        _fm = QMenu(self._frame_menu_btn)
        _fw = QWidget()
        _fl = QHBoxLayout(_fw); _fl.setContentsMargins(8, 6, 8, 6); _fl.setSpacing(4)
        self._frame_min_combo = QComboBox()
        self._frame_min_combo.setEditable(True)
        self._frame_min_combo.setFixedHeight(24)
        self._frame_min_combo.setMaxVisibleItems(30)
        self._configure_toolbar_combo(self._frame_min_combo, width=70, min_contents=6)
        _fl.addWidget(self._frame_min_combo)
        _fl.addWidget(QLabel("–"))
        self._frame_max_combo = QComboBox()
        self._frame_max_combo.setEditable(True)
        self._frame_max_combo.setFixedHeight(24)
        self._frame_max_combo.setMaxVisibleItems(30)
        self._configure_toolbar_combo(self._frame_max_combo, width=70, min_contents=6)
        _fl.addWidget(self._frame_max_combo)
        self._apply_frame_btn = QPushButton("Apply")
        self._apply_frame_btn.setFixedHeight(24); self._apply_frame_btn.setStyleSheet(_sm)
        self._apply_frame_btn.clicked.connect(self._apply_frame_range)
        _fl.addWidget(self._apply_frame_btn)
        self._clear_frame_btn = QPushButton("Clear")
        self._clear_frame_btn.setProperty("flat", True)
        self._clear_frame_btn.setFixedHeight(24); self._clear_frame_btn.setStyleSheet(_sm)
        self._clear_frame_btn.clicked.connect(self._clear_frame_range)
        _fl.addWidget(self._clear_frame_btn)
        _fa = QWidgetAction(_fm); _fa.setDefaultWidget(_fw)
        _fm.addAction(_fa)
        self._frame_menu_btn.setMenu(_fm)
        tb.addWidget(self._frame_menu_btn)

        tb.addSeparator()

        # ── 5  Selection ─────────────────────────────────
        self._point_select_btn = QPushButton("Point Select")
        self._point_select_btn.setCheckable(True); self._point_select_btn.setStyleSheet(_sm)
        self._point_select_btn.setToolTip(
            "ON: lasso selects entry/exit points (coarse)\n"
            "OFF: lasso selects paths (fine)")
        self._point_select_btn.toggled.connect(self._toggle_selection_mode)
        tb.addWidget(self._point_select_btn)

        self._lasso_btn = QPushButton("Lasso")
        self._lasso_btn.setCheckable(True); self._lasso_btn.setStyleSheet(_sm)
        self._lasso_btn.setToolTip(
            "Force-lasso mode (L)\nShift+Drag: lasso\nAlt+Shift+Drag: remove\nEsc: clear")
        self._lasso_btn.toggled.connect(self._toggle_lasso)
        tb.addWidget(self._lasso_btn)

        self._clear_sel_btn = QPushButton("Clear Sel")
        self._clear_sel_btn.setProperty("flat", True)
        self._clear_sel_btn.setStyleSheet(_sm)
        self._clear_sel_btn.setToolTip("Clear selection (Esc)")
        self._clear_sel_btn.clicked.connect(self._clear_selection)
        tb.addWidget(self._clear_sel_btn)

    # ─── Central Widget ─────────────────────────────────
    def _build_observer_dataset_panel(self, parent: QWidget, fallback_title: str) -> dict:
        container = QFrame(parent)
        container.setStyleSheet("QFrame{background:#f8fafc;border:1px solid #e5e7eb;border-radius:6px;}")
        container.setMinimumWidth(340)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        title = QLabel(str(fallback_title))
        title.setStyleSheet("font-size:11px;font-weight:600;color:#4b5563;")
        layout.addWidget(title)

        panel = ProteinViewerPanel(container, compact=True)
        panel.setMinimumWidth(300)
        layout.addWidget(panel, 1)
        return {
            "container": container,
            "label": title,
            "panel": panel,
            "fallback_title": str(fallback_title),
            "source_name": str(fallback_title),
        }

    def _protein_observer_slots(self) -> list[dict]:
        slots: list[dict] = []
        for slot in (
            getattr(self, "_protein_viewer_panel_top", None),
            getattr(self, "_protein_viewer_panel_bottom", None),
        ):
            if isinstance(slot, dict):
                slots.append(slot)
        return slots

    def _iter_protein_observer_panels(self) -> list[ProteinViewerPanel]:
        return [
            slot["panel"]
            for slot in self._protein_observer_slots()
            if slot.get("panel") is not None
        ]

    def _set_observer_highlight_style(self, style: str) -> None:
        for panel in self._iter_protein_observer_panels():
            panel.set_highlight_style(style)

    def _set_observer_model_style(self, style: str) -> None:
        for panel in self._iter_protein_observer_panels():
            if hasattr(panel, "set_model_style"):
                panel.set_model_style(style)

    def _set_observer_protein_opacity(self, opacity: float) -> None:
        for panel in self._iter_protein_observer_panels():
            panel.set_protein_opacity(opacity)

    def _set_protein_observer_slot_title(self, slot: dict | None, title: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        label = slot.get("label")
        if label is None:
            return
        text = str(title or slot.get("fallback_title") or "").strip()
        label.setText(text or str(slot.get("fallback_title") or "Observer"))

    def _observer_slot_key(self, slot: dict | None) -> str:
        if slot is getattr(self, "_protein_viewer_panel_top", None):
            return "top"
        if slot is getattr(self, "_protein_viewer_panel_bottom", None):
            return "bottom"
        return ""

    def _observer_frame_value(self) -> int:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return 0
        return int(spin.value())

    def _update_observer_slot_title(self, slot: dict | None, dataset_label: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        panel = slot.get("panel")
        fallback_title = str(slot.get("fallback_title") or "Observer")
        label = str(dataset_label or slot.get("source_name") or fallback_title).strip()
        if panel is not None and hasattr(panel, "has_sequence") and panel.has_sequence():
            current_frame = panel.current_sequence_frame()
            if current_frame is not None:
                self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label} | frame {current_frame}")
                return
            self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label}")
            return
        self._set_protein_observer_slot_title(slot, f"{fallback_title} | {label}")

    def _observer_sequence_slots(self) -> list[dict]:
        result: list[dict] = []
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if panel is not None and hasattr(panel, "has_sequence") and panel.has_sequence():
                result.append(slot)
        return result

    def _observer_slot_combo(self, side: str):
        if str(side) == "top":
            return getattr(self, "_observer_top_sequence_combo", None)
        if str(side) == "bottom":
            return getattr(self, "_observer_bottom_sequence_combo", None)
        return None

    def _observer_dataset_choices(self) -> list[tuple[str, str]]:
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        choices: list[tuple[str, str]] = []
        for dataset in datasets:
            key = str(dataset.get("key") or "").strip()
            if not key:
                continue
            label = self._dataset_display_name(key, str(dataset.get("name") or key))
            choices.append((label, key))
        return choices

    def _preferred_observer_dataset_key(self, side: str) -> str:
        slot = getattr(self, "_protein_viewer_panel_top" if str(side) == "top" else "_protein_viewer_panel_bottom", None)
        if isinstance(slot, dict):
            key = str(slot.get("dataset_key") or "").strip()
            if key:
                return key
        combo = self._observer_slot_combo(side)
        if combo is not None:
            key = str(combo.currentData() or "").strip()
            if key:
                return key
        pending = dict(getattr(self, "_observer_pending_payload", {}) or {})
        side_key = "left_dataset_key" if str(side) == "top" else "right_dataset_key"
        key = str(pending.get(side_key, "") or "").strip()
        if key:
            return key
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is not None:
            if str(side) == "top" and hasattr(panel, "left_dataset_key"):
                key = str(panel.left_dataset_key() or "").strip()
            elif str(side) == "bottom" and hasattr(panel, "right_dataset_key"):
                key = str(panel.right_dataset_key() or "").strip()
            else:
                key = ""
            if key:
                return key
        choices = self._observer_dataset_choices()
        if not choices:
            return ""
        if str(side) == "bottom" and len(choices) >= 2:
            return str(choices[1][1])
        return str(choices[0][1])

    def _observer_sequence_source_candidates(self, dataset_key: str) -> list[tuple[str, str]]:
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or "").strip())
        if dataset is None:
            return []
        md_file_sources: list[str] = []
        md_root_sources: list[str] = []
        md_root_path = str(getattr(dataset, "md_root_path", "") or "").strip()
        if md_root_path:
            normalized_md_root = os.path.abspath(md_root_path)
            if os.path.isfile(normalized_md_root) and normalized_md_root.lower().endswith(".pdb"):
                normalized_md_root = os.path.dirname(normalized_md_root)
            if os.path.isdir(normalized_md_root):
                md_root_sources.append(normalized_md_root)
        dataset_info = self._dataset_info_for_statistics(str(dataset_key or "").strip())
        md_path_file = self._find_md_path_file_near_dataset(dataset_info)
        if md_path_file:
            try:
                with open(md_path_file, "r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        raw_value = str(line).strip().strip("\"'")
                        if not raw_value:
                            continue
                        candidate = (
                            os.path.abspath(raw_value)
                            if os.path.isabs(raw_value)
                            else os.path.abspath(os.path.join(os.path.dirname(md_path_file), raw_value))
                        )
                        if os.path.isfile(candidate) and candidate.lower().endswith(".pdb"):
                            candidate = os.path.dirname(candidate)
                        if candidate and os.path.isdir(candidate):
                            md_file_sources.append(candidate)
            except OSError:
                pass
        roots: list[str] = []
        root_candidates = (
            md_file_sources
            if md_file_sources
            else (md_root_sources if md_root_sources else [getattr(dataset, "folder", "")])
        )
        for candidate in root_candidates:
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and normalized not in roots and os.path.isdir(normalized):
                roots.append(normalized)

        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()
        for root in roots:
            direct_has_pdb = False
            try:
                direct_has_pdb = any(name.lower().endswith(".pdb") for name in os.listdir(root))
            except OSError:
                direct_has_pdb = False
            if direct_has_pdb:
                label = os.path.basename(root) or root
                candidates.append((label, root))
                seen.add(os.path.normcase(root))
                continue
            try:
                child_names = sorted(os.listdir(root))
            except OSError:
                child_names = []
            for child_name in child_names:
                child_path = os.path.abspath(os.path.join(root, child_name))
                if not os.path.isdir(child_path):
                    continue
                key = os.path.normcase(child_path)
                if key in seen:
                    continue
                try:
                    has_pdb = any(name.lower().endswith(".pdb") for name in os.listdir(child_path))
                except OSError:
                    has_pdb = False
                if not has_pdb:
                    continue
                label = f"{os.path.basename(root) or root} / {child_name}"
                candidates.append((label, child_path))
                seen.add(key)
            if candidates:
                continue
            try:
                for walk_root, _dirs, files in os.walk(root):
                    walk_path = os.path.abspath(walk_root)
                    if os.path.normcase(walk_path) == os.path.normcase(root):
                        continue
                    if not any(name.lower().endswith(".pdb") for name in files):
                        continue
                    key = os.path.normcase(walk_path)
                    if key in seen:
                        continue
                    label = os.path.relpath(walk_path, root)
                    candidates.append((f"{os.path.basename(root) or root} / {label}", walk_path))
                    seen.add(key)
                    break
            except OSError:
                pass
        return candidates

    def _observer_resolve_sequence_source(self, dataset_key: str) -> tuple[str, str]:
        dataset_key = str(dataset_key or "").strip()
        candidates = self._observer_sequence_source_candidates(dataset_key)
        if not candidates:
            return "", ""
        if len(candidates) == 1:
            return candidates[0]

        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        tokens: list[str] = []
        for raw_value in (
            getattr(dataset, "name", "") if dataset is not None else "",
            os.path.basename(str(getattr(dataset, "folder", "") or "").strip()) if dataset is not None else "",
            getattr(dataset, "prefix", "") if dataset is not None else "",
        ):
            token = str(raw_value or "").strip().lower()
            if token:
                tokens.append(token)

        ranked: list[tuple[int, str, str]] = []
        for label, path in candidates:
            basename = os.path.basename(str(path or "")).strip().lower()
            label_text = str(label or "").strip().lower()
            score = 0
            for token in tokens:
                if basename == token:
                    score += 8
                elif basename.startswith(token) or basename.endswith(token):
                    score += 5
                elif token in basename:
                    score += 3
                if label_text == token:
                    score += 4
                elif token in label_text:
                    score += 2
            ranked.append((score, label, path))
        ranked.sort(key=lambda item: (-int(item[0]), str(item[1]).lower(), str(item[2]).lower()))
        _score, label, path = ranked[0]
        return str(label), str(path)

    def _set_observer_dataset_selection(self, side: str, dataset_key: str) -> None:
        dataset_key = str(dataset_key or "").strip()
        slot = getattr(self, "_protein_viewer_panel_top" if str(side) == "top" else "_protein_viewer_panel_bottom", None)
        if isinstance(slot, dict):
            slot["dataset_key"] = dataset_key
        combo = self._observer_slot_combo(side)
        if combo is None:
            return
        idx = combo.findData(dataset_key) if dataset_key else 0
        combo.blockSignals(True)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _refresh_observer_sequence_source_combo(self, side: str) -> None:
        combo = self._observer_slot_combo(side)
        if combo is None:
            return
        slot = getattr(self, "_protein_viewer_panel_top" if str(side) == "top" else "_protein_viewer_panel_bottom", None)
        previous_dataset_key = str(slot.get("dataset_key") or "").strip() if isinstance(slot, dict) else ""
        current_key = str(combo.currentData() or "").strip()
        preferred_key = previous_dataset_key or current_key or self._preferred_observer_dataset_key(side)
        choices = self._observer_dataset_choices()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("-", "")
        for label, key in choices:
            combo.addItem(label, key)
        if preferred_key:
            idx = combo.findData(preferred_key)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif len(choices) == 1:
                combo.setCurrentIndex(1)
        elif len(choices) == 1:
            combo.setCurrentIndex(1)
        combo.blockSignals(False)
        if isinstance(slot, dict):
            slot["dataset_key"] = str(combo.currentData() or "").strip()
            slot["sequence_sources"] = self._observer_sequence_source_candidates(slot["dataset_key"])
        selected_key = str(combo.currentData() or "").strip()
        if not selected_key:
            self._on_observer_sequence_source_changed(side)
            return
        should_load = selected_key != previous_dataset_key
        if isinstance(slot, dict):
            panel = slot.get("panel")
            if panel is not None and hasattr(panel, "has_sequence") and not panel.has_sequence():
                should_load = True
        observer_container = getattr(self, "_protein_observer_container", None)
        observer_visible = bool(observer_container is not None and observer_container.isVisible())
        if should_load and observer_visible:
            self._on_observer_sequence_source_changed(side)

    def _on_observer_sequence_source_changed(self, side: str) -> None:
        combo = self._observer_slot_combo(side)
        slot = getattr(self, "_protein_viewer_panel_top" if str(side) == "top" else "_protein_viewer_panel_bottom", None)
        if combo is None or not isinstance(slot, dict):
            return
        dataset_key = str(combo.currentData() or "").strip()
        slot["dataset_key"] = dataset_key
        panel = slot.get("panel")
        if panel is None:
            return
        dataset_label = self._dataset_display_name(dataset_key, str(slot.get("fallback_title") or "Observer")) if dataset_key else str(slot.get("fallback_title") or "Observer")
        if not dataset_key:
            panel.clear_protein()
            slot["source_name"] = dataset_label
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | {dataset_label}",
            )
            self._refresh_observer_frame_controls()
            self._refresh_observer_timeline()
            self._refresh_observer_path_overlays()
            return
        source_label, folder = self._observer_resolve_sequence_source(dataset_key)
        if not folder:
            panel.clear_protein()
            slot["source_name"] = dataset_label
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | {dataset_label} | no MD frames",
            )
            self._protein_observer_debug_label.setText(
                f"{slot.get('fallback_title', 'Observer')} has no readable MD frame folder"
            )
            self._refresh_observer_frame_controls()
            self._refresh_observer_timeline()
            self._refresh_observer_path_overlays()
            return
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        preferred_frame = self._observer_frame_value()
        result = panel.load_pdb_sequence_folder(folder)
        if not bool(result.get("ok")):
            self._protein_observer_debug_label.setText(
                f"{slot.get('fallback_title', 'Observer')} sequence load failed: {result.get('error', 'unknown_error')}"
            )
            self._statusbar.showMessage("Observer frame folder load failed", 4000)
            return
        slot["source_name"] = dataset_label
        slot["sequence_sources"] = self._observer_sequence_source_candidates(dataset_key)
        self._update_observer_slot_title(slot, dataset_label)
        self._refresh_observer_frame_controls(preferred_frame=preferred_frame or int(result.get("current_frame", 0) or 0))
        target_frame = self._observer_frame_value()
        current_frame_raw = result.get("current_frame", result.get("frame", target_frame))
        current_frame = int(target_frame if current_frame_raw is None else current_frame_raw)
        if target_frame != current_frame:
            frame_result = panel.set_sequence_frame(target_frame)
            if bool(frame_result.get("ok")):
                result = {**result, **frame_result}
        panel.set_highlight_residues(
            residue_ids,
            focus_loaded=True,
            refresh_sequence_cache=False,
            defer_render=bool(residue_ids),
        )
        self._protein_observer_debug_label.setText(
            f"{slot.get('fallback_title', 'Observer')} loaded {dataset_label} | "
            f"{source_label or os.path.basename(folder)} | "
            f"{int(result.get('frame_count', 0) or 0)} frames | "
            f"frame {result.get('frame', result.get('current_frame', target_frame))} | "
            f"{folder}"
        )
        self._statusbar.showMessage(
            f"{slot.get('fallback_title', 'Observer')} frame source loaded: {dataset_label}",
            4000,
        )
        self._refresh_observer_path_overlays()

    def _refresh_observer_frame_controls(self, preferred_frame: int | None = None) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        prev_btn = getattr(self, "_observer_prev_btn", None)
        next_btn = getattr(self, "_observer_next_btn", None)
        if spin is None:
            return

        sequence_slots = self._observer_sequence_slots()
        if not sequence_slots:
            spin.blockSignals(True)
            spin.setEnabled(False)
            spin.setRange(0, 0)
            spin.setValue(0)
            spin.blockSignals(False)
            if prev_btn is not None:
                prev_btn.setEnabled(False)
            if next_btn is not None:
                next_btn.setEnabled(False)
            return

        frame_ranges = []
        for slot in sequence_slots:
            panel = slot["panel"]
            frames = panel.available_sequence_frames()
            if not frames:
                continue
            frame_ranges.append((int(frames[0]), int(frames[-1])))
        if not frame_ranges:
            return

        frame_min = min(start for start, _end in frame_ranges)
        frame_max = max(end for _start, end in frame_ranges)

        current_value = int(preferred_frame if preferred_frame is not None else spin.value())
        current_value = max(frame_min, min(frame_max, current_value))
        spin.blockSignals(True)
        spin.setEnabled(True)
        spin.setRange(frame_min, frame_max)
        spin.setValue(current_value)
        spin.blockSignals(False)
        if prev_btn is not None:
            prev_btn.setEnabled(True)
        if next_btn is not None:
            next_btn.setEnabled(True)
        self._refresh_observer_timeline()

    def _observer_dataset_key(self, side: str) -> str:
        return self._preferred_observer_dataset_key(side)

    def _observer_selected_frames_for_dataset(self, dataset_key: str) -> set[int]:
        dataset_key = str(dataset_key or "").strip()
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not dataset_key or not selected_path_ids:
            return set()
        local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, selected_path_ids)
        if not local_ids:
            return set()
        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset is None:
            return set()
        try:
            return set(int(frame) for frame in dataset.db.get_frame_ids_for_path_ids(local_ids))
        except Exception:
            return set()

    def _preferred_observer_frame_from_selection(self) -> int | None:
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not selected_path_ids:
            return None
        frames_by_side: list[set[int]] = []
        for side in ("top", "bottom"):
            dataset_key = self._observer_dataset_key(side)
            if not dataset_key:
                continue
            frames = self._observer_selected_frames_for_dataset(dataset_key)
            if frames:
                frames_by_side.append(frames)
        if not frames_by_side:
            return None
        combined = set().union(*frames_by_side)
        if len(combined) == 1:
            return int(next(iter(combined)))
        return None

    def _sync_observer_frame_to_selection(self) -> None:
        preferred_frame = self._preferred_observer_frame_from_selection()
        if preferred_frame is None:
            return
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        frame_min = int(spin.minimum())
        frame_max = int(spin.maximum())
        target_frame = max(frame_min, min(frame_max, int(preferred_frame)))
        if int(spin.value()) == target_frame:
            return
        spin.setValue(target_frame)

    def _observer_path_points_for_frame(
        self,
        dataset_key: str,
        frame_number: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        dataset_key = str(dataset_key or "").strip()
        selected_path_ids = set(getattr(self.state, "effective_path_ids", set()) or set())
        if not dataset_key or not selected_path_ids:
            return None, None, None
        local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, selected_path_ids)
        if not local_ids:
            return None, None, None
        dataset = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset is None:
            return None, None, None
        try:
            rows = dataset.db.get_path_point_rows(
                local_ids,
                frame_id=int(frame_number),
                original_coords=True,
            )
        except Exception:
            return None, None, None
        if not rows:
            return None, None, None
        path_ids = np.asarray(
            [int(row["path_id"]) for row in rows],
            dtype=np.int64,
        )
        points = np.asarray(
            [(float(row["x"]), float(row["y"]), float(row["z"])) for row in rows],
            dtype=np.float64,
        )
        radii = np.asarray(
            [max(0.1, float(row["radius"] or 0.1)) for row in rows],
            dtype=np.float64,
        )
        return points, radii, path_ids

    def _observer_path_overlay_color(self, dataset_key: str) -> str:
        key = str(dataset_key or "").strip()
        if not key:
            return self.state.get_auto_color(0)
        return str(
            self.state.dataset_colors.get(
                key,
                self.state.get_auto_color(0),
            )
        )

    def _refresh_observer_path_overlays(self, frame_number: int | None = None) -> None:
        frame = int(frame_number if frame_number is not None else self._observer_frame_value())
        top_dataset_key = self._observer_dataset_key("top")
        bottom_dataset_key = self._observer_dataset_key("bottom")
        signature = (
            frame,
            top_dataset_key,
            self._observer_path_overlay_color(top_dataset_key),
            bottom_dataset_key,
            self._observer_path_overlay_color(bottom_dataset_key),
            tuple(sorted(int(path_id) for path_id in self.state.effective_path_ids)),
        )
        if signature == self._last_observer_overlay_signature:
            return
        self._last_observer_overlay_signature = signature
        for side, attr in (("top", "_protein_viewer_panel_top"), ("bottom", "_protein_viewer_panel_bottom")):
            slot = getattr(self, attr, None)
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is None or not hasattr(panel, "set_path_overlay"):
                continue
            dataset_key = self._observer_dataset_key(side)
            points, radii, path_ids = self._observer_path_points_for_frame(
                dataset_key,
                frame,
            )
            if points is None:
                panel.clear_path_overlay()
            else:
                panel.set_path_overlay(
                    points,
                    radii,
                    path_ids,
                    color=self._observer_path_overlay_color(dataset_key),
                )

    def _refresh_observer_timeline(self) -> None:
        timeline = getattr(self, "_observer_timeline", None)
        if timeline is None:
            return
        top_slot = getattr(self, "_protein_viewer_panel_top", None)
        bottom_slot = getattr(self, "_protein_viewer_panel_bottom", None)
        top_panel = top_slot.get("panel") if isinstance(top_slot, dict) else None
        bottom_panel = bottom_slot.get("panel") if isinstance(bottom_slot, dict) else None
        top_frames = []
        bottom_frames = []
        shared_current_frame = self._observer_frame_value()
        top_current_frame = shared_current_frame
        bottom_current_frame = shared_current_frame
        if top_panel is not None and hasattr(top_panel, "available_sequence_frames"):
            top_frames = list(top_panel.available_sequence_frames())
        if bottom_panel is not None and hasattr(bottom_panel, "available_sequence_frames"):
            bottom_frames = list(bottom_panel.available_sequence_frames())
        timeline.set_timelines(
            top_frames,
            self._observer_selected_frames_for_dataset(self._observer_dataset_key("top")),
            top_current_frame,
            bottom_frames,
            self._observer_selected_frames_for_dataset(self._observer_dataset_key("bottom")),
            bottom_current_frame,
        )

    def _on_observer_timeline_zoom_changed(self, value: int) -> None:
        timeline = getattr(self, "_observer_timeline", None)
        if timeline is None:
            return
        timeline.set_zoom_percent(int(value))

    def _sync_observer_timeline_zoom_slider(self, value: int) -> None:
        slider = getattr(self, "_observer_timeline_zoom_slider", None)
        if slider is None:
            return
        value = max(int(slider.minimum()), min(int(slider.maximum()), int(value)))
        if int(slider.value()) == value:
            return
        was_blocked = slider.blockSignals(True)
        slider.setValue(value)
        slider.blockSignals(was_blocked)

    def _on_observer_timeline_frame_clicked(self, side: str, frame: int) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        frame = int(frame)
        if int(spin.value()) != frame:
            spin.setValue(frame)
            self._refresh_observer_timeline()
            return
        self._apply_observer_frame_to_sequence_slots(
            frame,
            self._parse_observer_residue_ids(self._observer_residue_input.text()),
            focus_loaded=True,
        )

    def _apply_observer_frame_to_sequence_slots(
        self,
        frame_number: int,
        residue_ids,
        *,
        focus_loaded: bool,
    ) -> list[str]:
        messages: list[str] = []
        for slot in self._observer_sequence_slots():
            panel = slot.get("panel")
            if panel is None:
                continue
            panel.set_highlight_residues(
                residue_ids,
                focus_loaded=False,
                refresh_sequence_cache=False,
            )
            result = panel.set_sequence_frame(frame_number)
            if not bool(result.get("ok")):
                messages.append(f"{slot.get('fallback_title', 'Observer')}: {result.get('error', 'load_failed')}")
                continue
            panel.set_highlight_residues(residue_ids, focus_loaded=focus_loaded)
            slot["source_name"] = os.path.basename(panel.sequence_folder()) or str(slot.get("fallback_title") or "Observer")
            self._update_observer_slot_title(slot)
            messages.append(
                f"{slot.get('fallback_title', 'Observer')} frame {result.get('frame', frame_number)}"
            )
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays(frame_number)
        return messages

    def _choose_observer_sequence_folder(self, slot_key: str) -> None:
        self._refresh_observer_sequence_source_combo(slot_key)

    def _on_observer_frame_changed(self, value: int) -> None:
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        messages = self._apply_observer_frame_to_sequence_slots(
            int(value),
            residue_ids,
            focus_loaded=False,
        )
        if messages:
            self._protein_observer_debug_label.setText(" | ".join(messages))

    def _step_observer_frame(self, delta: int) -> None:
        spin = getattr(self, "_observer_frame_spin", None)
        if spin is None or not spin.isEnabled():
            return
        spin.setValue(spin.value() + int(delta))

    def _dataset_display_name(self, dataset_key: str, fallback: str) -> str:
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or "").strip())
        if dataset is None:
            return str(fallback)
        return str(dataset.prefix or dataset.name or fallback)

    def _dataset_representative_pdb_path(self, dataset_key: str) -> str:
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key or "").strip())
        search_roots: list[str] = []
        for candidate in (
            getattr(dataset, "folder", ""),
            os.path.dirname(str(getattr(dataset, "path", "") or "").strip()),
        ):
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and normalized not in search_roots and os.path.isdir(normalized):
                search_roots.append(normalized)

        for root in search_roots:
            direct = os.path.join(root, "representative_frame.pdb")
            if os.path.exists(direct):
                return direct

            for walk_root, _dirs, files in os.walk(root):
                if "representative_frame.pdb" in files:
                    return os.path.abspath(os.path.join(walk_root, "representative_frame.pdb"))

        for root in list(search_roots):
            current = root
            for _ in range(4):
                parent = os.path.dirname(current)
                if not parent or parent == current:
                    break
                direct = os.path.join(parent, "representative_frame.pdb")
                if os.path.exists(direct):
                    return direct
                current = parent

        dataset_name = str(getattr(dataset, "name", "") or "")
        dataset_folder = str(getattr(dataset, "folder", "") or "")
        fallback_tokens = [
            token for token in (
                dataset_name.split("_")[0] if dataset_name else "",
                os.path.basename(dataset_folder).split("_")[0] if dataset_folder else "",
            )
            if token
        ]
        database_root = os.path.abspath(os.path.join(os.getcwd(), "database"))
        for token in fallback_tokens:
            candidate = os.path.join(database_root, token, "representative_frame.pdb")
            if os.path.exists(candidate):
                return candidate
        return ""

    def _clear_protein_observer_slot(self, slot: dict | None, title: str | None = None) -> None:
        if not isinstance(slot, dict):
            return
        panel = slot.get("panel")
        if panel is not None:
            panel.clear_protein()
        self._set_protein_observer_slot_title(slot, title)

    def _load_combination_into_observer_panel(
        self,
        slot: dict | None,
        dataset_key: str,
        residue_ids,
        *,
        present: bool,
    ) -> None:
        if not isinstance(slot, dict):
            return
        requested_dataset_key = str(dataset_key or "").strip()
        current_dataset_key = str(slot.get("dataset_key") or "").strip()
        slot["dataset_key"] = requested_dataset_key
        default_title = str(slot.get("fallback_title") or "Observer")
        dataset_label = self._dataset_display_name(requested_dataset_key, default_title) if requested_dataset_key else default_title
        panel = slot.get("panel")
        if (
            panel is not None
            and hasattr(panel, "has_sequence")
            and panel.has_sequence()
            and current_dataset_key == requested_dataset_key
        ):
            result = panel.set_sequence_frame(self._observer_frame_value())
            if not bool(result.get("ok")):
                self._clear_protein_observer_slot(slot, f"{default_title} | {dataset_label}")
                return
            panel.set_highlight_residues(residue_ids, focus_loaded=True)
            slot["source_name"] = os.path.basename(panel.sequence_folder()) or dataset_label
            self._update_observer_slot_title(slot, slot["source_name"])
            return

        title = f"{default_title} | {dataset_label}"
        if not requested_dataset_key:
            self._clear_protein_observer_slot(slot, title)
            return
        if panel is None:
            return

        source_label, folder = self._observer_resolve_sequence_source(requested_dataset_key)
        if folder:
            result = panel.load_pdb_sequence_folder(folder)
            if bool(result.get("ok")):
                self._refresh_observer_frame_controls(
                    preferred_frame=self._observer_frame_value()
                    if self._observer_frame_value() is not None
                    else int(result.get("current_frame", 0) or 0)
                )
                panel.set_sequence_frame(self._observer_frame_value())
                panel.set_highlight_residues(residue_ids if present else (), focus_loaded=True)
                suffix = "" if present else " | no pair"
                slot["source_name"] = source_label or os.path.basename(folder) or dataset_label
                self._set_protein_observer_slot_title(
                    slot,
                    f"{title}{suffix} | {slot['source_name']} | {int(result.get('frame_count', 0) or 0)} frames",
                )
                self._refresh_observer_timeline()
                self._refresh_observer_path_overlays()
                return

        pdb_path = self._dataset_representative_pdb_path(requested_dataset_key)
        if not pdb_path:
            self._clear_protein_observer_slot(slot, title)
            return

        panel.load_pdb_file(pdb_path, first_model_only=True)
        panel.set_highlight_residues(residue_ids if present else (), focus_loaded=True)
        suffix = "" if present else " | no pair"
        slot["source_name"] = os.path.basename(os.path.dirname(pdb_path)) or dataset_label
        self._set_protein_observer_slot_title(
            slot,
            f"{title}{suffix} | {slot['source_name']}",
        )

    def _update_combination_observer_views(self, payload: dict | None) -> None:
        payload = payload or {}
        left_residue_ids = tuple(
            int(rid) for rid in payload.get("left_residue_ids", payload.get("residue_ids", ())) if int(rid) > 0
        )
        right_residue_ids = tuple(
            int(rid) for rid in payload.get("right_residue_ids", payload.get("residue_ids", ())) if int(rid) > 0
        )
        if len(left_residue_ids) < 1 and len(right_residue_ids) < 1:
            return
        left_dataset_key = str(payload.get("left_dataset_key", "") or "")
        right_dataset_key = str(payload.get("right_dataset_key", "") or "")
        self._set_observer_dataset_selection("top", left_dataset_key)
        self._set_observer_dataset_selection("bottom", right_dataset_key)
        self._refresh_observer_sequence_source_combo("top")
        self._refresh_observer_sequence_source_combo("bottom")
        self._load_combination_into_observer_panel(
            getattr(self, "_protein_viewer_panel_top", None),
            left_dataset_key,
            left_residue_ids,
            present=bool(payload.get("present_left")) and bool(left_residue_ids),
        )
        self._load_combination_into_observer_panel(
            getattr(self, "_protein_viewer_panel_bottom", None),
            right_dataset_key,
            right_residue_ids,
            present=bool(payload.get("present_right")) and bool(right_residue_ids),
        )

    @staticmethod
    def _parse_observer_residue_ids(text: str) -> tuple[int, ...]:
        residue_ids: list[int] = []
        for token in re.findall(r"\d+", str(text or "")):
            try:
                residue_id = int(token)
            except ValueError:
                continue
            if residue_id > 0 and residue_id not in residue_ids:
                residue_ids.append(residue_id)
        return tuple(residue_ids)

    def _apply_observer_debug_selection(self) -> None:
        residue_ids = self._parse_observer_residue_ids(self._observer_residue_input.text())
        if len(residue_ids) < 1:
            self._protein_observer_debug_label.setText("Enter at least one residue ID, for example: 72")
            self._statusbar.showMessage("Observer debug input needs at least one residue ID", 3000)
            return

        if self._observer_sequence_slots():
            frame_number = self._observer_frame_value()
            messages = self._apply_observer_frame_to_sequence_slots(
                frame_number,
                residue_ids,
                focus_loaded=True,
            )
            if messages:
                self._protein_observer_debug_label.setText(" | ".join(messages))
                self._statusbar.showMessage(
                    f"Observer updated at frame {frame_number} for residues {', '.join(str(rid) for rid in residue_ids)}",
                    4000,
                )
                return

        loaded_messages: list[str] = []
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if (
                panel is None
                or not hasattr(panel, "has_loaded_protein")
                or not panel.has_loaded_protein()
            ):
                continue
            panel.set_highlight_residues(residue_ids, focus_loaded=True)
            loaded_messages.append(str(slot.get("fallback_title") or "Observer"))
        if loaded_messages:
            self._protein_observer_debug_label.setText(
                f"Observer residues: {', '.join(str(rid) for rid in residue_ids)} | "
                f"applied to {', '.join(loaded_messages)}"
            )
            self._statusbar.showMessage(
                f"Observer updated for residues {', '.join(str(rid) for rid in residue_ids)}",
                4000,
            )
            return

        pending = dict(self._observer_pending_payload or {})
        panel = getattr(self, "_residue_combination_panel", None)
        left_dataset_key = str(
            pending.get("left_dataset_key")
            or (panel.left_dataset_key() if panel is not None else "")
            or ""
        ).strip()
        right_dataset_key = str(
            pending.get("right_dataset_key")
            or (panel.right_dataset_key() if panel is not None else "")
            or ""
        ).strip()
        pending_residue_ids = tuple(int(rid) for rid in pending.get("residue_ids", ()) if int(rid) > 0)
        use_pending_presence = pending_residue_ids == residue_ids
        payload = {
            "residue_ids": residue_ids,
            "left_dataset_key": left_dataset_key,
            "right_dataset_key": right_dataset_key,
            "present_left": bool(pending.get("present_left")) if use_pending_presence else bool(left_dataset_key),
            "present_right": bool(pending.get("present_right")) if use_pending_presence else bool(right_dataset_key),
        }
        left_pdb = self._dataset_representative_pdb_path(left_dataset_key) if left_dataset_key else ""
        right_pdb = self._dataset_representative_pdb_path(right_dataset_key) if right_dataset_key else ""
        if not left_pdb and not right_pdb:
            self._protein_observer_debug_label.setText(
                "Observer PDB not found for either dataset. Check representative_frame.pdb lookup."
            )
            self._statusbar.showMessage("Observer PDB not found for the selected datasets", 4000)
            return
        self._update_combination_observer_views(payload)
        self._protein_observer_debug_label.setText(
            f"Observer residues: {', '.join(str(rid) for rid in residue_ids)} | "
            f"A={os.path.basename(left_pdb) if left_pdb else 'missing'} | "
            f"B={os.path.basename(right_pdb) if right_pdb else 'missing'}"
        )
        self._statusbar.showMessage(
            f"Observer updated for residues {', '.join(str(rid) for rid in residue_ids)}",
            4000,
        )

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._central_layout = layout

        # Main splitter: 3D viewer (top) + charts (bottom)
        self._main_splitter = QSplitter(Qt.Vertical)

        # Main visual area: tunnel/path VTK view + right-side protein observer.
        self._main_view_stack = QStackedWidget()
        self._viewer = TunnelViewer3D()

        self._main_visual_splitter = QSplitter(Qt.Horizontal)
        self._main_visual_splitter.setChildrenCollapsible(False)
        self._main_visual_splitter.addWidget(self._viewer)

        self._protein_observer_container = QFrame()
        self._protein_observer_container.setObjectName("proteinObserverContainer")
        self._protein_observer_container.setStyleSheet(
            "#proteinObserverContainer{background:#ffffff;border-left:1px solid #e5e7eb;}"
        )
        observer_layout = QVBoxLayout(self._protein_observer_container)
        observer_layout.setContentsMargins(8, 8, 8, 8)
        observer_layout.setSpacing(6)

        self._protein_observer_title = QLabel("Residue Observer")
        self._protein_observer_title.setStyleSheet("font-size:12px;font-weight:600;color:#1f2937;")
        observer_layout.addWidget(self._protein_observer_title)

        observer_debug_row = QHBoxLayout()
        observer_debug_row.setSpacing(6)
        self._observer_residue_input = QLineEdit()
        self._observer_residue_input.setPlaceholderText("Residues, e.g. 72,149")
        self._observer_residue_input.returnPressed.connect(self._apply_observer_debug_selection)
        observer_debug_row.addWidget(self._observer_residue_input, 1)
        self._observer_apply_btn = QPushButton("Apply")
        self._observer_apply_btn.setFixedHeight(24)
        self._observer_apply_btn.clicked.connect(self._apply_observer_debug_selection)
        observer_debug_row.addWidget(self._observer_apply_btn)
        observer_layout.addLayout(observer_debug_row)

        observer_sequence_row = QHBoxLayout()
        observer_sequence_row.setSpacing(6)
        observer_sequence_row.addWidget(QLabel("MD Dataset"))
        self._observer_top_sequence_combo = QComboBox()
        self._observer_top_sequence_combo.setMinimumWidth(148)
        self._observer_top_sequence_combo.currentIndexChanged.connect(
            lambda _idx: self._on_observer_sequence_source_changed("top")
        )
        observer_sequence_row.addWidget(self._observer_top_sequence_combo, 1)
        self._observer_bottom_sequence_combo = QComboBox()
        self._observer_bottom_sequence_combo.setMinimumWidth(148)
        self._observer_bottom_sequence_combo.currentIndexChanged.connect(
            lambda _idx: self._on_observer_sequence_source_changed("bottom")
        )
        observer_sequence_row.addWidget(self._observer_bottom_sequence_combo, 1)
        observer_layout.addLayout(observer_sequence_row)

        observer_model_row = QHBoxLayout()
        observer_model_row.setSpacing(6)
        observer_model_row.addWidget(QLabel("Model"))
        self._observer_model_style_combo = QComboBox()
        self._observer_model_style_combo.addItem("Backbone", "backbone")
        self._observer_model_style_combo.addItem("Cartoon", "cartoon")
        self._observer_model_style_combo.addItem("Tube", "tube")
        self._observer_model_style_combo.addItem("CA Spheres", "ca_spheres")
        self._observer_model_style_combo.currentIndexChanged.connect(
            lambda _idx: self._set_observer_model_style(
                str(self._observer_model_style_combo.currentData() or "backbone")
            )
        )
        observer_model_row.addWidget(self._observer_model_style_combo, 1)
        observer_model_row.addWidget(QLabel("Opacity"))
        self._observer_protein_opacity_slider = QSlider(Qt.Horizontal)
        self._observer_protein_opacity_slider.setRange(0, 100)
        self._observer_protein_opacity_slider.setValue(100)
        self._observer_protein_opacity_slider.setMinimumWidth(120)
        self._observer_protein_opacity_slider.setToolTip("Protein opacity in observer panels")
        self._observer_protein_opacity_slider.valueChanged.connect(
            lambda value: self._set_observer_protein_opacity(int(value) / 100.0)
        )
        observer_model_row.addWidget(self._observer_protein_opacity_slider, 1)
        observer_layout.addLayout(observer_model_row)

        observer_bottom_row = QHBoxLayout()
        observer_bottom_row.setSpacing(6)
        self._observer_prev_btn = QPushButton("Prev")
        self._observer_prev_btn.setFixedHeight(24)
        self._observer_prev_btn.setEnabled(False)
        self._observer_prev_btn.clicked.connect(lambda: self._step_observer_frame(-1))
        observer_bottom_row.addWidget(self._observer_prev_btn)
        observer_bottom_row.addWidget(QLabel("Frame"))
        self._observer_frame_spin = QSpinBox()
        self._observer_frame_spin.setEnabled(False)
        self._observer_frame_spin.setRange(0, 0)
        self._observer_frame_spin.setFixedHeight(24)
        self._observer_frame_spin.setMinimumWidth(86)
        self._observer_frame_spin.valueChanged.connect(self._on_observer_frame_changed)
        observer_bottom_row.addWidget(self._observer_frame_spin)
        self._observer_next_btn = QPushButton("Next")
        self._observer_next_btn.setFixedHeight(24)
        self._observer_next_btn.setEnabled(False)
        self._observer_next_btn.clicked.connect(lambda: self._step_observer_frame(1))
        observer_bottom_row.addWidget(self._observer_next_btn)
        observer_bottom_row.addWidget(QLabel("Highlight"))
        self._observer_highlight_style_combo = QComboBox()
        self._observer_highlight_style_combo.addItem("Ball-and-Stick", "ball_and_stick")
        self._observer_highlight_style_combo.addItem("Sticks", "sticks")
        self._observer_highlight_style_combo.addItem("Spheres", "spheres")
        self._observer_highlight_style_combo.currentIndexChanged.connect(
            lambda _idx: self._set_observer_highlight_style(
                str(self._observer_highlight_style_combo.currentData() or "ball_and_stick")
            )
        )
        observer_bottom_row.addWidget(self._observer_highlight_style_combo, 1)
        observer_layout.addLayout(observer_bottom_row)

        self._protein_observer_debug_label = QLabel("Click a residue pair to fill the box, then press Apply.")
        self._protein_observer_debug_label.setWordWrap(True)
        self._protein_observer_debug_label.setStyleSheet("font-size:11px;color:#6b7280;")
        observer_layout.addWidget(self._protein_observer_debug_label)

        observer_splitter = QSplitter(Qt.Vertical, self._protein_observer_container)
        observer_splitter.setChildrenCollapsible(False)

        self._protein_viewer_panel_top = self._build_observer_dataset_panel(
            self._protein_observer_container,
            "File A",
        )
        self._protein_viewer_panel_bottom = self._build_observer_dataset_panel(
            self._protein_observer_container,
            "File B",
        )
        self._observer_timeline = ObserverTimelineWidget(self._protein_observer_container)
        self._observer_timeline.frame_clicked.connect(self._on_observer_timeline_frame_clicked)
        self._observer_timeline.zoom_changed.connect(self._sync_observer_timeline_zoom_slider)
        observer_splitter.addWidget(self._protein_viewer_panel_top["container"])
        observer_splitter.addWidget(self._observer_timeline)
        observer_splitter.addWidget(self._protein_viewer_panel_bottom["container"])
        observer_splitter.setStretchFactor(0, 1)
        observer_splitter.setStretchFactor(1, 0)
        observer_splitter.setStretchFactor(2, 1)
        observer_splitter.setSizes([280, 60, 280])
        observer_splitter.setMinimumWidth(360)
        observer_layout.addWidget(observer_splitter, 1)

        # Compatibility alias for older code paths that expect a single observer panel.
        self._protein_viewer_panel = self._protein_viewer_panel_top["panel"]
        self._set_protein_observer_slot_title(self._protein_viewer_panel_top, "File A")
        self._set_protein_observer_slot_title(self._protein_viewer_panel_bottom, "File B")
        self._set_observer_model_style("backbone")
        self._set_observer_highlight_style("ball_and_stick")
        self._set_observer_protein_opacity(1.0)

        self._main_visual_splitter.addWidget(self._protein_observer_container)
        self._main_visual_splitter.setStretchFactor(0, 1)
        self._main_visual_splitter.setStretchFactor(1, 0)
        self._main_visual_splitter.setSizes([900, 320])
        self._protein_observer_container.setMinimumWidth(420)
        self._protein_observer_container.setVisible(False)

        self._main_view_stack.addWidget(self._main_visual_splitter)
        self._main_view_stack.setCurrentWidget(self._main_visual_splitter)
        self._main_splitter.addWidget(self._main_view_stack)

        # Bottom charts
        charts_widget = QWidget()
        charts_outer = QVBoxLayout(charts_widget)
        charts_outer.setContentsMargins(4, 2, 4, 4)
        charts_outer.setSpacing(2)

        # Sync button bar
        btn_bar = QHBoxLayout()
        self._chart_status_label = QLabel("")
        btn_bar.addWidget(self._chart_status_label)
        btn_bar.addStretch()

        self._sync_charts_btn = QPushButton("Sync Charts")
        self._sync_charts_btn.setFixedHeight(24)
        self._sync_charts_btn.setFixedWidth(90)
        self._sync_charts_btn.clicked.connect(self._sync_charts)
        btn_bar.addWidget(self._sync_charts_btn)
        self._open_charts_btn = QPushButton("Chart Window")
        self._open_charts_btn.setFixedHeight(24)
        self._open_charts_btn.setFixedWidth(104)
        self._open_charts_btn.setToolTip("Open a larger synchronized chart workspace")
        self._open_charts_btn.clicked.connect(self._open_chart_workspace)
        btn_bar.addWidget(self._open_charts_btn)
        charts_outer.addLayout(btn_bar)

        # Charts row
        self._detached_hint = QLabel(
            "Charts are currently displayed in Chart Workspace. "
            "Use the button above to reopen the detached window."
        )
        self._detached_hint.setStyleSheet(
            "color: #909399; font-size: 11px; padding: 10px 4px 6px 4px;"
        )
        self._detached_hint.setWordWrap(True)
        self._detached_hint.setVisible(False)
        charts_outer.addWidget(self._detached_hint)

        self._bottom_charts_container = QWidget()
        charts_layout = QHBoxLayout()
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(6)
        self._bottom_charts_container.setLayout(charts_layout)

        self._profile_chart = ProfileChart()
        self._statistics_chart = StatisticsChart()
        self._statistics_chart.set_display_config(*self._profile_chart.display_config())
        charts_layout.addWidget(self._profile_chart, 1)
        charts_layout.addWidget(self._statistics_chart, 1)
        charts_outer.addWidget(self._bottom_charts_container, 1)

        self._main_splitter.addWidget(charts_widget)
        self._main_splitter.setStretchFactor(0, 3)
        self._main_splitter.setStretchFactor(1, 1)
        self._main_splitter.setSizes([600, 200])

        layout.addWidget(self._main_splitter)

    def _relocate_top_controls(self):
        self._main_toolbar.setVisible(False)

        self._protein_opacity_slider.setVisible(False)
        self._clear_protein_btn.setVisible(False)
        self._binding_sites_btn.setVisible(False)
        self._protein_align_combo.setVisible(False)
        self._lasso_btn.setVisible(False)

        top_controls = QWidget()
        top_controls.setObjectName("mainViewControls")
        top_controls.setStyleSheet(
            "#mainViewControls{background:#ffffff;border-bottom:1px solid #e5e7eb;}"
        )
        self._top_controls_widget = top_controls

        controls_layout = QGridLayout(top_controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setHorizontalSpacing(8)
        controls_layout.setVerticalSpacing(6)
        self._top_controls_layout = controls_layout

        self._color_label = QLabel("Color")
        self._color_label.setStyleSheet("font-size:11px;color:#909399;")
        self._bg_label = QLabel("BG")
        self._bg_label.setStyleSheet("font-size:11px;color:#909399;")
        self._top_controls_items = [
            self._color_label,
            self._path_color_slider,
            self._bg_label,
            self._bg_opacity_slider,
            self._point_select_btn,
            self._clear_sel_btn,
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._protein_toggle_btn,
            self._load_observer_protein_btn,
        ]

        self._central_layout.insertWidget(0, top_controls)
        top_controls.setVisible(True)
        self._refresh_top_controls_layout()

        self._load_protein_btn.setText("Load Protein")
        self._load_observer_protein_btn.setText("Residue Observer")

        button_row = self._dataset_panel._button_row
        for widget in (
            self._dataset_panel._add_btn,
            self._dataset_panel._prefix_btn,
            self._dataset_panel._color_btn,
            self._dataset_panel._reference_btn,
            self._dataset_panel._visibility_btn,
            self._dataset_panel._cluster_mode_btn,
            self._dataset_panel._cluster_labels_btn,
            self._dataset_panel._remove_btn,
        ):
            button_row.removeWidget(widget)

        metrics = QFontMetrics(self.font())
        shared_width = max(
            128,
            metrics.horizontalAdvance(self._dataset_panel._add_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._prefix_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._color_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._reference_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._visibility_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._cluster_mode_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._cluster_labels_btn.text()),
            metrics.horizontalAdvance(self._dataset_panel._remove_btn.text()),
        ) + 24
        for button in (
            self._dataset_panel._add_btn,
            self._dataset_panel._prefix_btn,
            self._dataset_panel._color_btn,
            self._dataset_panel._reference_btn,
            self._dataset_panel._visibility_btn,
            self._dataset_panel._cluster_mode_btn,
            self._dataset_panel._cluster_labels_btn,
            self._dataset_panel._remove_btn,
        ):
            button.setMinimumWidth(shared_width)
            button.setFixedHeight(self._dataset_panel._add_btn.sizeHint().height())
        button_row.addWidget(self._dataset_panel._add_btn, 0, 0)
        button_row.addWidget(self._dataset_panel._prefix_btn, 0, 1)
        button_row.addWidget(self._dataset_panel._color_btn, 0, 2)
        button_row.addWidget(self._dataset_panel._reference_btn, 0, 3)
        button_row.addWidget(self._dataset_panel._visibility_btn, 1, 0)
        button_row.addWidget(self._dataset_panel._cluster_mode_btn, 1, 1)
        button_row.addWidget(self._dataset_panel._cluster_labels_btn, 1, 2)
        button_row.addWidget(self._dataset_panel._remove_btn, 1, 3)

        for button in (
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._point_select_btn,
            self._clear_sel_btn,
            self._protein_toggle_btn,
        ):
            button.setMinimumWidth(max(button.minimumWidth(), 78 if button is not self._protein_toggle_btn else 118))
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        for widget in (
            self._path_color_slider,
            self._bg_opacity_slider,
            self._entry_exit_btn,
            self._entry_exit_scope_btn,
            self._focus_btn,
            self._protein_toggle_btn,
            self._load_observer_protein_btn,
            self._point_select_btn,
            self._clear_sel_btn,
        ):
            widget.setVisible(True)
        for widget in (
            self._path_change_btn,
            self._exit_delta_btn,
            self._path_delta_filter_slider,
            self._path_delta_filter_label,
            self._residue_toggle_btn,
            self._residue_labels_btn,
            self._exit_cluster_btn,
            self._load_protein_btn,
        ):
            widget.setVisible(False)
        self._protein_style_combo.setVisible(False)
        self._frame_menu_btn.setVisible(False)

        self._residue_toggle_btn.blockSignals(True)
        self._residue_toggle_btn.setChecked(False)
        self._residue_toggle_btn.blockSignals(False)
        self._residue_labels_btn.blockSignals(True)
        self._residue_labels_btn.setChecked(False)
        self._residue_labels_btn.blockSignals(False)
        self.state.show_all_residues = False
        self._viewer.clear_residues()
        self._viewer.clear_residue_labels()

    def _refresh_top_controls_layout(self):
        if self._top_controls_layout is None or self._top_controls_widget is None:
            return
        layout = self._top_controls_layout
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(self._top_controls_widget)

        available_width = max(600, self._top_controls_widget.width())
        columns = len(self._top_controls_items) if available_width >= 1220 else 7
        for index, widget in enumerate(self._top_controls_items):
            row = index // columns
            column = index % columns
            layout.addWidget(widget, row, column)
        for column in range(columns):
            layout.setColumnStretch(column, 0)
        layout.setColumnStretch(columns, 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_top_controls_layout()

    # ─── Side Panel (Dock) ──────────────────────────────
    def _build_side_panel(self):
        dock = QDockWidget("Control Panel", self)
        dock.setObjectName("ControlPanelDock")
        dock.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dock.setAutoFillBackground(True)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        tabs = QTabWidget()
        tabs.setObjectName("ControlPanelTabs")
        tabs.setElideMode(Qt.TextElideMode.ElideNone)
        tabs.tabBar().setExpanding(False)
        tabs.tabBar().setUsesScrollButtons(True)
        tabs.setDocumentMode(True)
        tabs.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        tabs.setAutoFillBackground(True)

        # Hidden helper panels kept for existing selection / table logic.
        self._residue_panel = ResiduePanel(self.state, self.db)

        self._path_panel = PathPanel(self.state, self.db)

        # Visible tabs
        self._dataset_panel = DatasetPanel(self.db, self.state)
        tabs.addTab(self._dataset_panel, "Datasets")

        tabs.addTab(self._path_panel, "Paths")

        self._residue_compare_panel = ResidueComparePanel()
        tabs.addTab(self._residue_compare_panel, "Residue Compare")

        self._residue_combination_panel = ResidueCombinationPanel()
        tabs.addTab(self._residue_combination_panel, "Residue Combination Compare")
        self._statistics_panel = None

        dataset_button_row_width = self._dataset_panel.primary_button_row_width()
        self._side_dock_width = max(1, int(dataset_button_row_width) + 20)
        self._side_dock_min_width = self._side_dock_width
        dock.setMinimumWidth(self._side_dock_min_width)
        dock.resize(self._side_dock_width, dock.height())

        dock_container = QWidget()
        dock_container.setObjectName("ControlPanelContainer")
        dock_container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dock_container.setAutoFillBackground(True)
        dock_container.setMinimumWidth(int(self._side_dock_min_width))
        dock_layout = QHBoxLayout(dock_container)
        dock_layout.setContentsMargins(0, 0, 0, 0)
        dock_layout.setSpacing(0)
        dock_layout.addWidget(tabs, 1)
        self._side_dock_resize_handle = DockResizeHandle(
            lambda: int(self._side_dock.width()) if getattr(self, "_side_dock", None) is not None else int(self._side_dock_width),
            self._apply_side_dock_width,
            dock_container,
        )
        dock_layout.addWidget(self._side_dock_resize_handle, 0)

        dock.setWidget(dock_container)
        self._side_dock = dock
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._apply_side_dock_width(self._side_dock_width)
        QTimer.singleShot(0, self._apply_initial_side_dock_width)

    def _apply_initial_side_dock_width(self):
        dataset_button_row_width = self._dataset_panel.primary_button_row_width()
        self._side_dock_width = max(1, int(dataset_button_row_width) + 20)
        self._side_dock_min_width = self._side_dock_width
        self._apply_side_dock_width(self._side_dock_width)

    def _apply_side_dock_width(self, width: int):
        dock = getattr(self, "_side_dock", None)
        min_width = int(getattr(self, "_side_dock_min_width", 0))
        width_value = max(min_width, int(width))
        if self.width() > 0:
            width_value = min(width_value, max(min_width, int(self.width()) - 220))
        self._side_dock_width = width_value
        if dock is not None:
            dock.setMinimumWidth(min_width)
            dock.resize(width_value, dock.height())
            try:
                self.resizeDocks([dock], [width_value], Qt.Horizontal)
            except Exception:
                pass
            widget = dock.widget()
            if widget is not None:
                widget.setMinimumWidth(min_width)
                widget.updateGeometry()
                widget.update()
            dock.updateGeometry()
            dock.update()

    def _build_statistics_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        self._statistics_selection_label = QLabel("Selected: 0 paths")
        self._statistics_selection_label.setStyleSheet("font-weight:600;color:#1f2937;")
        header_row.addWidget(self._statistics_selection_label)
        header_row.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.clicked.connect(self._refresh_statistics_selection_label)
        header_row.addWidget(refresh_btn)
        layout.addLayout(header_row)

        button_row = QHBoxLayout()
        button_row.setSpacing(6)
        self._stats_entrance_centroid_btn = QPushButton("Entrance Centroid")
        self._stats_entrance_centroid_btn.setFixedHeight(28)
        self._stats_entrance_centroid_btn.clicked.connect(self._compute_statistics_entrance_centroid)
        button_row.addWidget(self._stats_entrance_centroid_btn)
        self._stats_centerline_distance_btn = QPushButton("Centerline Distance")
        self._stats_centerline_distance_btn.setFixedHeight(28)
        self._stats_centerline_distance_btn.clicked.connect(self._compute_statistics_centerline_distance)
        button_row.addWidget(self._stats_centerline_distance_btn)
        self._stats_export_cluster_btn = QPushButton("Cluster Count Results")
        self._stats_export_cluster_btn.setFixedHeight(28)
        self._stats_export_cluster_btn.clicked.connect(self._compute_dataset_cluster_results_export)
        button_row.addWidget(self._stats_export_cluster_btn)
        self._stats_cluster_compactness_btn = QPushButton("Cluster Compactness")
        self._stats_cluster_compactness_btn.setFixedHeight(28)
        self._stats_cluster_compactness_btn.clicked.connect(self._compute_dataset_cluster_compactness)
        button_row.addWidget(self._stats_cluster_compactness_btn)
        self._stats_cluster_residue_similarity_btn = QPushButton("Cluster Residue Similarity")
        self._stats_cluster_residue_similarity_btn.setFixedHeight(28)
        self._stats_cluster_residue_similarity_btn.clicked.connect(self._compute_dataset_cluster_residue_similarity)
        button_row.addWidget(self._stats_cluster_residue_similarity_btn)
        layout.addLayout(button_row)

        export_row = QHBoxLayout()
        self._statistics_summary_label = QLabel("No statistics calculated.")
        self._statistics_summary_label.setWordWrap(True)
        self._statistics_summary_label.setStyleSheet("font-size:11px;color:#6b7280;")
        export_row.addWidget(self._statistics_summary_label, 1)
        self._stats_export_btn = QPushButton("Export CSV")
        self._stats_export_btn.setFixedHeight(26)
        self._stats_export_btn.setEnabled(False)
        self._stats_export_btn.clicked.connect(self._export_statistics_csv)
        export_row.addWidget(self._stats_export_btn)
        layout.addLayout(export_row)

        self._statistics_table = QTableWidget(0, 0)
        self._statistics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._statistics_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._statistics_table.setAlternatingRowColors(True)
        self._statistics_table.verticalHeader().setVisible(False)
        self._statistics_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._statistics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._statistics_table, 1)

        return panel

    # ─── Bottom Panel ───────────────────────────────────
    def _build_bottom_panel(self):
        pass  # Charts already in main splitter

    # ─── Status Bar ─────────────────────────────────────
    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._status_paths = QLabel("Paths: 0")
        self._status_residues = QLabel("Residues: 0")
        self._status_memory = QLabel("")
        self._loading_bar = QProgressBar()
        self._loading_bar.setRange(0, 0)
        self._loading_bar.setTextVisible(False)
        self._loading_bar.setFixedWidth(120)
        self._loading_bar.setFixedHeight(12)
        self._loading_bar.setVisible(False)
        self._loading_bar.setStyleSheet(
            "QProgressBar{border:1px solid #d1d5db;border-radius:6px;background:#f3f4f6;}"
            "QProgressBar::chunk{border-radius:6px;background:#22c55e;}"
        )
        self._statusbar.addWidget(self._status_paths)
        self._statusbar.addWidget(QLabel("|"))
        self._statusbar.addWidget(self._status_residues)
        self._statusbar.addWidget(QLabel("|"))
        self._status_selected = QLabel("Selected: 0 paths")
        self._statusbar.addWidget(self._status_selected)
        self._statusbar.addPermanentWidget(self._loading_bar)
        self._statusbar.addPermanentWidget(self._status_memory)

    def _begin_loading(self, message: str):
        self._loading_depth += 1
        self._loading_bar.setVisible(True)
        self._statusbar.showMessage(message)
        QApplication.processEvents()

    def _end_loading(self, message: str | None = None, timeout_ms: int = 3000):
        self._loading_depth = max(0, self._loading_depth - 1)
        if self._loading_depth == 0:
            self._loading_bar.setVisible(False)
            if message:
                self._statusbar.showMessage(message, timeout_ms)

    # ─── Signal wiring ──────────────────────────────────
    def _connect_signals(self):
        # Residue panel → update matched paths
        self._residue_panel.selection_changed.connect(self._on_residue_changed)

        # Path panel → show profiles, export
        self._path_panel.paths_selected.connect(self._on_paths_selected)
        self._path_panel.path_selected.connect(self._on_path_clicked)
        self._path_panel.export_requested.connect(self._on_export)
        self._path_panel.profile_requested.connect(self._on_load_selected_profiles)
        self._dataset_panel.datasets_changed.connect(self._on_datasets_changed)
        self._dataset_panel.dataset_color_changed.connect(self._on_dataset_color_changed)
        self._dataset_panel.dataset_visibility_changed.connect(self._on_dataset_visibility_changed)
        self._dataset_panel.dataset_visibility_batch_changed.connect(self._on_dataset_visibility_batch_changed)
        self._dataset_panel.dataset_selection_changed.connect(self._on_dataset_selection_changed)
        self._dataset_panel.dataset_cluster_mode_changed.connect(self._on_dataset_cluster_mode_changed)
        self._dataset_panel.dataset_cluster_labels_changed.connect(self._on_dataset_cluster_labels_changed)
        self._dataset_panel.cluster_paths_selected.connect(self._on_dataset_cluster_paths_selected)
        self._dataset_panel.reference_protein_requested.connect(self._on_load_protein)

        # Shared chart config
        self._profile_chart.display_config_changed.connect(
            self._on_chart_display_config_changed
        )
        self._profile_chart.appearance_changed.connect(
            self._on_chart_appearance_changed
        )
        self._profile_chart.keep_only_requested.connect(
            self._on_chart_keep_only_requested
        )
        self._profile_chart.exclude_requested.connect(
            self._on_chart_exclude_requested
        )
        self._profile_chart.chart_selection_changed.connect(
            self._on_chart_selection_changed
        )
        self._profile_chart.undo_requested.connect(
            self._on_chart_undo_requested
        )

        # Lasso selection
        self._viewer.lasso_selected.connect(self._on_lasso_selected)
        self._viewer.exit_cluster_selected.connect(self._on_exit_cluster_selected)
        self._viewer.path_clicked.connect(self._on_viewer_path_clicked)
        self._viewer.path_coordinate_mode_changed.connect(
            self._on_viewer_coordinate_mode_changed
        )

        # Residue 3D click
        self._viewer.residue_clicked.connect(self._on_residue_3d_clicked)
        self._viewer.combination_clicked.connect(self._on_combination_marker_clicked)
        self._residue_compare_panel.residue_selection_requested.connect(
            self._on_residue_compare_selection_requested
        )
        self._residue_compare_panel.residue_clear_requested.connect(
            self._on_residue_compare_clear_requested
        )
        self._residue_combination_panel.visualization_changed.connect(
            self._refresh_residue_combination_markers
        )
        self._residue_combination_panel.manual_update_requested.connect(
            self._manual_update_residue_combination_compare
        )
        self._residue_combination_panel.contained_match_query_requested.connect(
            self._query_residue_combination_contained_match
        )
        self._residue_combination_panel.contained_match_compute_requested.connect(
            self._compute_residue_combination_contained_match
        )
        self._residue_combination_panel.contained_match_clear_requested.connect(
            self._clear_residue_combination_contained_match
        )
        self._residue_combination_panel.combination_selection_requested.connect(
            self._on_combination_selection_requested
        )
        self._residue_combination_panel.left_residue_set_requested.connect(
            self._on_left_residue_set_requested
        )

        # State signals — unified effective set drives 3D + table + status
        self.state.frame_range_changed.connect(self._reload_3d)
        self.state.effective_paths_changed.connect(self._on_effective_paths_changed)
        self.state.residue_selection_changed.connect(self._refresh_residue_3d)
        self.state.residue_selection_changed.connect(self._sync_residue_compare_selection_state)
        self.state.category_visibility_changed.connect(self._on_category_visibility_changed)
        self.state.dataset_display_changed.connect(self._on_dataset_display_changed)

        # Selection mode + entry/exit display
        self.state.selection_mode_changed.connect(
            lambda mode: self._viewer.set_selection_mode(mode)
        )
        self.state.entry_exit_display_changed.connect(self._on_entry_exit_display_changed)
        self.state.entry_exit_scope_changed.connect(self._on_entry_exit_scope_changed)

        # Chart debounce timer
        self._chart_timer = QTimer()
        self._chart_timer.setSingleShot(True)
        self._chart_timer.setInterval(150)
        self._chart_timer.timeout.connect(self._do_chart_update)

        # Keyboard shortcuts
        QShortcut(QKeySequence("L"), self).activated.connect(
            lambda: self._lasso_btn.toggle()
        )
        QShortcut(QKeySequence("F"), self).activated.connect(
            lambda: self._focus_btn.toggle()
        )
        QShortcut(QKeySequence("Escape"), self).activated.connect(
            self._clear_selection
        )

    def _on_chart_display_config_changed(
        self,
        color_key: str,
        x_key: str,
        y_key: str,
        advanced_mode: bool,
    ):
        if not self._charts_detached:
            self._profile_chart.set_display_config(
                color_key,
                x_key,
                y_key,
                advanced_mode,
                emit_signal=False,
            )
            self._statistics_chart.set_display_config(
                color_key,
                x_key,
                y_key,
                advanced_mode,
            )
        if self._chart_workspace is not None:
            self._chart_workspace.set_display_config(
                color_key,
                x_key,
                y_key,
                advanced_mode,
            )

    def _on_chart_appearance_changed(
        self,
        alpha_scale: float,
        depth_scale: float,
    ):
        if not self._charts_detached:
            self._profile_chart.set_appearance(
                alpha_scale,
                depth_scale,
                emit_signal=False,
            )
        if self._chart_workspace is not None:
            self._chart_workspace.set_appearance(
                alpha_scale,
                depth_scale,
            )

    def _on_chart_keep_only_requested(self, path_ids):
        selected = set(int(pid) for pid in path_ids)
        if not selected:
            return
        self._push_chart_operation_history()
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) in selected
        ]
        self._set_chart_profiles(profiles)
        self._chart_status_label.setText(f"Chart kept {len(selected)} paths")
        self._statusbar.showMessage(f"Chart kept {len(selected)} paths", 3000)

    def _on_chart_exclude_requested(self, path_ids):
        selected = set(int(pid) for pid in path_ids)
        if not selected:
            return
        self._push_chart_operation_history()
        self.state.apply_lasso(selected, "remove")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) not in selected
        ]
        if profiles:
            self._set_chart_profiles(profiles)
        else:
            self._clear_chart_views()
        self._chart_status_label.setText(f"Chart excluded {len(selected)} paths")
        self._statusbar.showMessage(f"Chart excluded {len(selected)} paths", 3000)

    def _on_chart_selection_changed(self, path_ids):
        if self._chart_workspace is not None:
            self._chart_workspace.reset_filter_history()
            self._chart_workspace.set_chart_selected_paths(path_ids)

    def _push_chart_operation_history(self):
        current = sorted(
            int(profile.get("pathIndex", -1))
            for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) > 0
        )
        if not current:
            return
        if self._chart_operation_history and self._chart_operation_history[-1] == current:
            return
        self._chart_operation_history.append(current)
        if len(self._chart_operation_history) > 5:
            self._chart_operation_history = self._chart_operation_history[-5:]
        self._sync_chart_undo_state()

    def _sync_chart_undo_state(self):
        available = bool(self._chart_operation_history)
        self._profile_chart.set_undo_available(available)
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.set_undo_available(available)

    def _restore_chart_path_ids(self, restored_ids: list[int]):
        selected = set(int(pid) for pid in restored_ids if int(pid) > 0)
        if not selected:
            return
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        profiles = [
            profile for profile in self._chart_profiles
            if int(profile.get("pathIndex", -1)) in selected
        ]
        if not profiles:
            profiles = self.db.get_path_profiles(sorted(selected))
        self._set_chart_profiles(profiles)
        if not self._charts_detached:
            self._profile_chart.apply_filtered_path_selection(sorted(selected))
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(selected))
            self._chart_workspace.set_chart_selected_paths(sorted(selected))

    def _on_chart_undo_requested(self):
        if not self._chart_operation_history:
            self._statusbar.showMessage("No chart operation to undo", 3000)
            return
        restored = self._chart_operation_history.pop()
        self._restore_chart_path_ids(restored)
        self._sync_chart_undo_state()
        self._chart_status_label.setText(f"Chart undo restored {len(restored)} paths")
        self._statusbar.showMessage(f"Chart undo restored {len(restored)} paths", 3000)

    def _on_chart_workspace_filter_apply_requested(self, payload: dict):
        selected = set(int(pid) for pid in payload.get("path_ids", []) if int(pid) > 0)
        if not selected:
            self._statusbar.showMessage("No paths matched the current chart filter", 3000)
            return
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(selected))
            self._chart_workspace.set_chart_selected_paths(sorted(selected))
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        self._chart_status_label.setText(f"Chart filter selected {len(selected)} paths")
        self._statusbar.showMessage(f"Chart filter selected {len(selected)} paths", 3000)

    def _on_chart_workspace_filter_revert_requested(self, path_ids):
        restored = set(int(pid) for pid in (path_ids or []) if int(pid) > 0)
        if not restored:
            self._statusbar.showMessage("No previous chart selection to restore", 3000)
            return
        if self._chart_workspace is not None:
            self._chart_workspace.profile_chart.apply_filtered_path_selection(sorted(restored))
            self._chart_workspace.set_chart_selected_paths(sorted(restored))
        self.state.apply_lasso(restored, "replace")
        self._path_panel.refresh()
        self._chart_status_label.setText(f"Chart selection restored to {len(restored)} paths")
        self._statusbar.showMessage(f"Chart selection restored to {len(restored)} paths", 3000)

    def _set_chart_profiles(self, profiles):
        self._chart_profiles = list(profiles)
        path_color_map = self._chart_path_color_map(self._chart_profiles)
        if not self._charts_detached:
            self._profile_chart.set_profiles(self._chart_profiles)
            self._profile_chart.set_path_color_map(path_color_map)
            self._statistics_chart.set_dataset_filter(self._profile_chart.selected_dataset_prefixes())
            self._statistics_chart.set_profiles(self._chart_profiles)
            self._statistics_chart.set_path_color_map(path_color_map)
        if self._chart_workspace is not None:
            self._chart_workspace.set_profiles(self._chart_profiles)
            self._chart_workspace.set_path_color_map(path_color_map)
            self._chart_workspace.set_chart_selected_paths(
                self._profile_chart.selected_path_ids() if not self._charts_detached else self._chart_workspace.profile_chart.selected_path_ids()
            )
        self._sync_chart_undo_state()

    def _clear_chart_views(self):
        self._chart_profiles = []
        if not self._charts_detached:
            self._profile_chart.clear()
            self._statistics_chart.clear()
        if self._chart_workspace is not None:
            self._chart_workspace.clear_charts()
        self._sync_chart_undo_state()

    def _active_chart_display_config(self):
        if self._charts_detached and self._chart_workspace is not None:
            return self._chart_workspace.profile_chart.display_config()
        return self._profile_chart.display_config()

    def _active_chart_appearance(self):
        if self._charts_detached and self._chart_workspace is not None:
            return self._chart_workspace.appearance()
        return self._profile_chart.appearance()

    def _set_main_charts_detached(self, detached: bool):
        if self._charts_detached == detached:
            return
        if detached:
            self._charts_detached = True
            self._bottom_charts_container.setVisible(False)
            self._detached_hint.setVisible(True)
            current_sizes = self._main_splitter.sizes()
            if current_sizes and sum(current_sizes) > 0:
                self._attached_chart_sizes = current_sizes
            self._profile_chart.clear()
            self._statistics_chart.clear()
            self._main_splitter.setSizes([760, 56])
            return

        if self._chart_workspace is not None:
            color_key, x_key, y_key, advanced_mode = (
                self._chart_workspace.profile_chart.display_config()
            )
            alpha_scale, depth_scale = self._chart_workspace.appearance()
            selected_path_ids = self._chart_workspace.profile_chart.selected_path_ids()
        else:
            color_key, x_key, y_key, advanced_mode = self._profile_chart.display_config()
            alpha_scale, depth_scale = self._profile_chart.appearance()
            selected_path_ids = self._profile_chart.selected_path_ids()

        self._charts_detached = False
        self._bottom_charts_container.setVisible(True)
        self._detached_hint.setVisible(False)
        self._profile_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
            emit_signal=False,
        )
        self._statistics_chart.set_display_config(
            color_key,
            x_key,
            y_key,
            advanced_mode,
        )
        self._profile_chart.set_appearance(
            alpha_scale,
            depth_scale,
            emit_signal=False,
        )
        if self._chart_profiles:
            self._profile_chart.set_profiles(self._chart_profiles)
            path_color_map = self._chart_path_color_map(self._chart_profiles)
            self._profile_chart.set_path_color_map(path_color_map)
            self._statistics_chart.set_dataset_filter(self._profile_chart.selected_dataset_prefixes())
            self._statistics_chart.set_profiles(self._chart_profiles)
            self._statistics_chart.set_path_color_map(path_color_map)
            if selected_path_ids:
                self._profile_chart.apply_filtered_path_selection(selected_path_ids)
        else:
            self._profile_chart.clear()
            self._statistics_chart.clear()
        self._profile_chart.set_undo_available(bool(self._chart_operation_history))
        self._main_splitter.setSizes(self._attached_chart_sizes or [600, 200])

    def _open_chart_workspace(self):
        is_new_workspace = False
        if self._chart_workspace is None:
            is_new_workspace = True
            self._chart_workspace = ChartWorkspaceDialog(self)
            self._chart_workspace.profile_chart.display_config_changed.connect(
                self._on_chart_display_config_changed
            )
            self._chart_workspace.profile_chart.appearance_changed.connect(
                self._on_chart_appearance_changed
            )
            self._chart_workspace.profile_chart.keep_only_requested.connect(
                self._on_chart_keep_only_requested
            )
            self._chart_workspace.profile_chart.exclude_requested.connect(
                self._on_chart_exclude_requested
            )
            self._chart_workspace.profile_chart.chart_selection_changed.connect(
                self._chart_workspace.set_chart_selected_paths
            )
            self._chart_workspace.profile_chart.undo_requested.connect(
                self._on_chart_undo_requested
            )
            self._chart_workspace.path_filter_apply_requested.connect(
                self._on_chart_workspace_filter_apply_requested
            )
            self._chart_workspace.path_filter_revert_requested.connect(
                self._on_chart_workspace_filter_revert_requested
            )
            self._chart_workspace.closed.connect(
                lambda: self._set_main_charts_detached(False)
            )
        self._chart_workspace.set_display_config(*self._active_chart_display_config())
        self._chart_workspace.set_appearance(*self._active_chart_appearance())
        if self._chart_profiles:
            path_color_map = self._chart_path_color_map(self._chart_profiles)
            self._chart_workspace.set_profiles(self._chart_profiles)
            self._chart_workspace.set_path_color_map(path_color_map)
            self._chart_workspace.profile_chart.set_undo_available(bool(self._chart_operation_history))
            selected_path_ids = self._profile_chart.selected_path_ids()
            if selected_path_ids:
                self._chart_workspace.profile_chart.apply_filtered_path_selection(selected_path_ids)
            self._chart_workspace.set_chart_selected_paths(
                self._chart_workspace.profile_chart.selected_path_ids()
            )
        else:
            self._chart_workspace.clear_charts()
        self._set_main_charts_detached(True)
        self._chart_workspace.show_and_focus()

    # ─── Data Loading ───────────────────────────────────
    def _init_data(self):
        """Load initial data after window is shown."""
        self._refresh_dataset_views()
        self._statusbar.showMessage("Ready", 3000)

    def _refresh_dataset_views(self):
        self._invalidate_render_signatures()
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        n_paths = self.db.path_count()
        fmin, fmax = self.db.frame_range()

        self._status_paths.setText(f"Paths: {n_paths:,}")
        n_res = int(self.db.get_meta("residue_count", "0"))
        self._status_residues.setText(f"Residues: {n_res:,}")

        # Frame range combos
        self._frame_min_combo.clear()
        self._frame_max_combo.clear()
        self._frame_min_combo.addItem(str(fmin))
        self._frame_max_combo.addItem(str(fmax))
        self._refresh_combo_popup_width(self._frame_min_combo)
        self._refresh_combo_popup_width(self._frame_max_combo)

        self._residue_panel.set_db(self.db)
        self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
        # Init residue panel
        self._residue_panel.load_options()

        self._path_panel.set_db(self.db)
        # Init path table
        self._path_panel.init_model()

        self.state.ensure_dataset_defaults([
            dataset["key"] for dataset in self.db.list_datasets()
        ])

        self._dataset_panel.refresh()
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        self._residue_compare_panel.set_dataset_choices(datasets)
        self._residue_combination_panel.set_db(self._primary_analysis_db())
        self._residue_combination_panel.set_dataset_choices(datasets)
        self._residue_combination_panel.set_residue_position_maps(
            self._dataset_residue_position_maps()
        )
        self._residue_combination_panel.set_residue_atom_position_maps(
            self._dataset_residue_atom_position_maps()
        )
        self._sync_residue_compare_selection_state()
        self._prime_primary_residue_positions()
        profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
        self._refresh_residue_combination_current_bottlenecks(
            profiles_by_dataset_key=profiles_by_dataset_key,
        )
        self._refresh_residue_combination_contained_match_status()
        self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
        self._sync_residue_combination_context()

        # Load 3D background paths in background thread
        self._reload_3d()

    def _reload_3d(self):
        """Load and render 3D background paths."""
        if self._path_coordinate_load_busy:
            return
        self._path_coordinate_load_busy = True
        if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
            self._viewer.set_path_coordinate_mode_busy(True)
        mode_label = "original paths" if self._path_coordinate_mode == "original" else "current paths"
        self._begin_loading(f"Loading 3D data ({mode_label})...")

        # Use QThread for non-blocking load
        self._3d_load_request_id += 1
        self._loader_thread = QThread()
        self._loader = DataLoader(
            self.db,
            self.state.frame_min,
            self.state.frame_max,
            self._path_coordinate_mode,
            self._3d_load_request_id,
        )
        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_3d_loaded)
        self._loader.failed.connect(self._on_3d_load_failed)
        self._loader.finished.connect(self._loader_thread.quit)
        self._loader.failed.connect(self._loader_thread.quit)
        self._loader.finished.connect(self._loader.deleteLater)
        self._loader.failed.connect(self._loader.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.start()

    def _on_3d_loaded(self, data, coordinate_mode: str, request_id: int):
        """Callback when 3D data is loaded."""
        if request_id != self._3d_load_request_id:
            return
        try:
            if coordinate_mode != self._path_coordinate_mode:
                return

            opacity = self._bg_opacity_slider.value() / 100.0
            self._viewer.render_background_paths(data, opacity=opacity)
            self._invalidate_render_signatures()
            self._viewer.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)
            self._clear_path_delta_exclusion(reset_slider=False, refresh=False)
            self._refresh_3d_overlays()
            if self._path_change_overlay_active:
                self._apply_path_change_overlay()
            elif self._exit_delta_overlay_active:
                self._apply_exit_delta_overlay()
            self._refresh_residue_combination_markers()
            mode_label = "original paths" if self._path_coordinate_mode == "original" else "current paths"
            self._end_loading(f"Loaded {len(data):,} 3D paths ({mode_label})", 3000)
        finally:
            self._path_coordinate_load_busy = False
            if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
                self._viewer.set_path_coordinate_mode_busy(False)

    def _on_3d_load_failed(self, error: str, coordinate_mode: str, request_id: int):
        if request_id != self._3d_load_request_id:
            return
        try:
            mode_label = "original paths" if coordinate_mode == "original" else "current paths"
            self._statusbar.showMessage(f"Failed to load 3D data ({mode_label}): {error}", 5000)
            self._end_loading()
        finally:
            self._path_coordinate_load_busy = False
            if hasattr(self._viewer, "set_path_coordinate_mode_busy"):
                self._viewer.set_path_coordinate_mode_busy(False)

    # ─── Event handlers ─────────────────────────────────
    def _on_residue_changed(self):
        """Residue selection changed → update effective paths + table (charts debounced)."""
        t_total = time.perf_counter()

        if not self.state.selected_residues:
            self.state.set_residue_filtered_paths([])
            self._path_panel.refresh()
            self._viewer.clear_effective()
            self._clear_chart_views()
            self._residue_panel.load_options()
            self._sync_residue_combination_context()
            self._refresh_residue_combination_markers()
            t_elapsed = (time.perf_counter() - t_total) * 1000
            print(f"[PERF][ResidueClick] total={t_elapsed:.1f}ms (clear)")
            return

        t0 = time.perf_counter()
        matched_ids = self.db.get_matched_path_ids(self.state.selected_residues)
        t_query = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] query_paths={t_query:.1f}ms")

        t0 = time.perf_counter()
        self.state.set_residue_filtered_paths(matched_ids)
        t_recompute = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] recompute_effective={t_recompute:.1f}ms")

        t0 = time.perf_counter()
        self._path_panel.refresh()
        t_panel = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] update_path_panel={t_panel:.1f}ms")

        opts = self.db.get_residue_options(self.state.selected_residues)
        self._residue_panel.load_options(opts)
        self._sync_residue_combination_context()
        self._refresh_residue_combination_markers()

        self._schedule_chart_update()

        t_elapsed = (time.perf_counter() - t_total) * 1000
        print(f"[PERF][ResidueClick] total={t_elapsed:.1f}ms")
        self._statusbar.showMessage(f"Matched {len(matched_ids)} paths", 3000)

    # ─── Chart debounce ──────────────────────────────────
    def _schedule_chart_update(self):
        """Restart chart debounce timer (150ms)."""
        self._chart_timer.start()

    def _do_chart_update(self):
        """Update chart status label; clear charts when no selection."""
        effective = self.state.effective_path_ids
        if not effective:
            self._clear_chart_views()
            self._chart_status_label.setText("")
            return
        self._chart_status_label.setText(
            f"{len(effective)} paths selected — click Sync Charts to update"
        )

    def _sync_charts(self):
        """Manually triggered: compute charts for visible effective paths only."""
        effective = self._visible_effective_path_ids()
        if not effective:
            self._clear_chart_views()
            self._chart_status_label.setText("")
            return

        ids = list(effective)
        self._chart_status_label.setText(f"Computing {len(ids)} paths...")
        QApplication.processEvents()

        t0 = time.perf_counter()
        profiles = self.db.get_path_profiles(ids)
        t_fetch = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._set_chart_profiles(profiles)
        t_chart_total = (time.perf_counter() - t0) * 1000

        self._chart_status_label.setText(
            f"{len(profiles)} paths — "
            f"fetch {t_fetch:.0f}ms, charts {t_chart_total:.0f}ms"
        )
        print(
            f"[PERF][SyncCharts] paths={len(profiles)}, "
            f"fetch={t_fetch:.1f}ms, charts={t_chart_total:.1f}ms"
        )

    # ─── Lasso selection ─────────────────────────────────
    def _toggle_lasso(self, checked: bool):
        self._viewer.set_lasso_mode(checked)
        if checked:
            self._statusbar.showMessage(
                "Force lasso mode ON — plain left-drag=lasso. Shift+left-drag always works.",
                5000,
            )
        else:
            self._statusbar.showMessage("Force lasso mode OFF — Shift+left-drag still works", 3000)

    def _on_lasso_selected(self, path_ids, mode):
        incoming = len(path_ids) if path_ids is not None else 0
        print(f"[LASSO][L5][MainWindow] received lasso_selected mode={mode}, incoming={incoming}")
        if not path_ids:
            path_ids = set()
        self.state.apply_lasso(path_ids, mode)
        self._schedule_chart_update()

    def _on_effective_paths_changed(self):
        """Unified handler: effective set changed → update 3D + status."""
        n = len(self.state.effective_path_ids)
        self._status_selected.setText(f"Effective: {n} paths")
        self._refresh_statistics_selection_label()
        t0 = time.perf_counter()
        self._render_effective_paths_if_needed()
        t_3d = (time.perf_counter() - t0) * 1000
        print(f"[PERF][ResidueClick] update_3d={t_3d:.1f}ms")
        self._refresh_residue_combination_contained_match_status()
        # Refresh entry/exit point highlight if visible
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
        self._sync_observer_frame_to_selection()
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays()

    def _manual_update_residue_combination_compare(self, mode: str = "dataset"):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        prefer_explicit_selection = str(mode or "").strip().lower() == "path"
        scope_label = "path" if prefer_explicit_selection else "dataset"
        self._begin_loading(f"Updating residue-combination compare context by {scope_label}...")
        finished = False
        total_started = time.perf_counter()
        panel.begin_view_update_batch()
        try:
            phase_started = time.perf_counter()
            profiles_by_dataset_key = self._refresh_residue_combination_current_paths(
                prefer_explicit_selection=prefer_explicit_selection,
            )
            current_paths_ms = (time.perf_counter() - phase_started) * 1000.0

            phase_started = time.perf_counter()
            if panel.current_bottleneck_filter_active():
                self._refresh_residue_combination_current_bottlenecks(
                    prefer_explicit_selection=prefer_explicit_selection,
                    profiles_by_dataset_key=profiles_by_dataset_key,
                )
            else:
                # Bottleneck rows are only needed when that filter is enabled.
                # Clear stale context so the first toggle requests a fresh load.
                panel.set_current_bottleneck_combination_keys(set())
                panel.set_current_bottleneck_combination_rows({}, {})
            bottlenecks_ms = (time.perf_counter() - phase_started) * 1000.0
            self._refresh_residue_combination_contained_match_status()

            phase_started = time.perf_counter()
            self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
            distances_ms = (time.perf_counter() - phase_started) * 1000.0
            self._sync_residue_combination_context()
            # All setters above may request a refresh. Defer the actual table
            # rebuild until the complete path context is ready.
            panel._refresh_view()
            finished = True
        finally:
            render_started = time.perf_counter()
            panel.end_view_update_batch()
            render_ms = (time.perf_counter() - render_started) * 1000.0
            if finished:
                total_ms = (time.perf_counter() - total_started) * 1000.0
                print(
                    f"[PERF][Load{scope_label.title()}] paths={current_paths_ms:.1f}ms "
                    f"bottlenecks={bottlenecks_ms:.1f}ms distances={distances_ms:.1f}ms "
                    f"render={render_ms:.1f}ms total={total_ms:.1f}ms"
                )
                self._end_loading(f"Residue Combination Compare updated by {scope_label}", 3000)
            else:
                self._end_loading()

    def _clear_selection(self):
        """Esc: clear lasso edits, restore to pure residue filter result."""
        self.state.clear_lasso_edits()
        self._schedule_chart_update()

    def _on_category_visibility_changed(self):
        """Category visibility toggled → render visible categories in 3D."""
        t_total = time.perf_counter()

        # Build visible overlay payloads from cache (no DB query)
        t0 = time.perf_counter()
        visible_categories = self._visible_category_payload()
        visible_datasets = self._visible_dataset_payload()
        t_fetch = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self._last_3d_overlay_signature = None
        self._refresh_3d_overlays()
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
        elif self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
        t_render = (time.perf_counter() - t0) * 1000
        t_bg = 0.0
        t_ee = 0.0

        total_paths = sum(len(ids) for ids, _ in visible_categories.values())
        t_elapsed = (time.perf_counter() - t_total) * 1000

        print(f"[PERF][CategoryClick] fetch_category_ids={t_fetch:.1f}ms")
        print(f"[PERF][CategoryClick] update_3d={t_render:.1f}ms")
        print(f"[PERF][CategoryClick] update_bg_exclusion={t_bg:.1f}ms")
        print(f"[PERF][CategoryClick] update_entry_exit={t_ee:.1f}ms")
        print(f"[PERF][CategoryClick] total={t_elapsed:.1f}ms")

        self._statusbar.showMessage(
            f"Showing {len(visible_categories)} categories ({total_paths} paths)", 3000
        )

    def _on_dataset_display_changed(self):
        self._refresh_3d_overlays()
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
        elif self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
        self._refresh_observer_path_overlays()
        self._dataset_panel.refresh()
        self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
        self._residue_panel.load_options()
        if self._chart_profiles:
            path_color_map = self._chart_path_color_map(self._chart_profiles)
            self._profile_chart.set_path_color_map(path_color_map)
            self._statistics_chart.set_path_color_map(path_color_map)
            if self._chart_workspace is not None:
                self._chart_workspace.set_path_color_map(path_color_map)

    def _on_dataset_color_changed(self, key: str, color: str):
        self.state.set_dataset_color(key, color)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        self._statusbar.showMessage(f"Dataset {label} tunnel color: {color}", 3000)

    def _on_dataset_selection_changed(self, dataset_keys):
        self._residue_panel.set_dataset_filter(dataset_keys)
        self._residue_panel.load_options()

    def _on_dataset_cluster_mode_changed(self, key: str, enabled: bool):
        self.state.set_dataset_cluster_display(key if enabled else None)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        mode_text = "cluster colors" if enabled else "dataset color"
        self._statusbar.showMessage(f"Dataset {label} render mode: {mode_text}", 3000)

    def _on_dataset_cluster_labels_changed(self, key: str, enabled: bool):
        self.state.set_dataset_cluster_labels(key if enabled else None)
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        mode_text = "cluster labels" if enabled else "labels hidden"
        self._statusbar.showMessage(f"Dataset {label}: {mode_text}", 3000)

    def _on_dataset_cluster_paths_selected(self, path_ids):
        selected = {int(pid) for pid in (path_ids or [])}
        self.state.apply_lasso(selected, "replace")
        self._schedule_chart_update()
        if selected:
            self._statusbar.showMessage(f"Selected {len(selected)} paths by cluster", 3000)
        else:
            self._statusbar.showMessage("Cleared cluster path selection", 3000)

    def _on_dataset_visibility_changed(self, key: str, visible: bool):
        self.state.set_dataset_visible(key, visible)
        self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
        self._residue_panel.load_options()
        dataset = self.db.get_dataset(key) if hasattr(self.db, "get_dataset") else None
        label = dataset["prefix"] if dataset else key
        action = "shown" if visible else "hidden"
        self._statusbar.showMessage(f"Dataset {label} tunnels {action}", 3000)

    def _on_dataset_visibility_batch_changed(self, keys, visible: bool):
        dataset_labels: list[str] = []
        normalized_keys: list[str] = []
        for key in keys or []:
            normalized = str(key or "").strip()
            if not normalized:
                continue
            normalized_keys.append(normalized)
            dataset = self.db.get_dataset(normalized) if hasattr(self.db, "get_dataset") else None
            dataset_labels.append(dataset["prefix"] if dataset else normalized)
        if normalized_keys:
            self.state.set_dataset_visibility_batch(normalized_keys, visible)
        self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
        self._residue_panel.load_options()
        if not dataset_labels:
            return
        action = "shown" if visible else "hidden"
        self._statusbar.showMessage(
            f"{len(dataset_labels)} datasets tunnels {action}: {', '.join(dataset_labels[:4])}"
            + (" ..." if len(dataset_labels) > 4 else ""),
            3000,
        )

    def _on_paths_selected(self, path_ids):
        """Multiple paths selected → update 3D only; charts via Sync button."""
        if self._suppress_next_path_preview:
            self._suppress_next_path_preview = False
            return
        if path_ids and len(path_ids) == 1:
            return
        if not path_ids:
            self._last_preview_path_ids = []
            self._last_highlighted_preview_ids = set()
            self._last_preview_color_map = {}
            self._contained_match_preview_active = False
            self._viewer.clear_matched()
            self._refresh_statistics_selection_label()
            return

        # Render in 3D
        preview_ids = [int(pid) for pid in path_ids[:200]]
        self._last_preview_path_ids = preview_ids
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        coords = self._get_display_path_coords(self._filter_hidden_dataset_paths(preview_ids))
        self._viewer.render_matched_paths(coords)

        self._chart_status_label.setText(
            f"{len(path_ids)} paths selected — click Sync Charts to update"
        )
        self._refresh_statistics_selection_label()

    def _on_path_clicked(self, path_id):
        """Single path clicked → select it in the main 3D view."""
        if self._suppress_next_path_preview:
            self._suppress_next_path_preview = False
            return
        path_id = int(path_id)
        self.state.apply_lasso({path_id}, "replace")
        self._schedule_chart_update()
        self._last_preview_path_ids = [path_id]
        self._last_highlighted_preview_ids = {path_id}
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._viewer.clear_matched()
        self._chart_status_label.setText("1 path selected")
        self._refresh_statistics_selection_label()

    def _on_viewer_path_clicked(self, path_id: int):
        """In Focus mode, clicking a visible path removes it from the current selection."""
        if self.state.focus_mode and path_id in self.state.effective_path_ids:
            self.state.apply_lasso({path_id}, "remove")
            self._schedule_chart_update()
            self._statusbar.showMessage(
                f"Removed path {self._format_path_label(path_id)} from selection "
                f"({len(self.state.effective_path_ids)} remaining)",
                3000,
            )
            return
        self._on_path_clicked(path_id)

    def _on_load_selected_profiles(self, path_ids):
        """Load Selected clicked → update profile/statistics charts only."""
        if not path_ids:
            return
        profiles = self.db.get_path_profiles(path_ids)
        self._set_chart_profiles(profiles)
        self._statusbar.showMessage(
            f"Loaded profiles for {len(profiles)} selected paths", 3000
        )

    def _on_path_color_changed(self, value):
        """Map slider 0-100 to light gray (#C8C8C8) → black (#000000) for background paths."""
        gray = int(200 * (1 - value / 100.0))
        color = f"#{gray:02X}{gray:02X}{gray:02X}"
        self._viewer.set_bg_color(color)

    def _on_bg_opacity_changed(self, value):
        opacity = value / 100.0
        self.state.bg_opacity = opacity
        self._viewer.set_bg_opacity(opacity)

    def _statistics_context_path_ids(self) -> list[int]:
        selected = self._path_panel_selected_path_ids()
        if selected:
            return sorted(selected)
        effective = {
            int(path_id)
            for path_id in getattr(self.state, "effective_path_ids", set())
            if int(path_id) > 0
        }
        if effective:
            return sorted(effective)
        visible = self.state.get_visible_category_path_ids()
        return sorted(int(path_id) for path_id in visible if int(path_id) > 0)

    def _refresh_statistics_selection_label(self):
        label = getattr(self, "_statistics_selection_label", None)
        if label is None:
            return
        count = len(self._statistics_context_path_ids())
        label.setText(f"Selected: {count:,} paths")

    @staticmethod
    def _format_stat_float(value: float) -> str:
        if not np.isfinite(float(value)):
            return ""
        return f"{float(value):.4f}"

    def _set_statistics_table(self, headers: list[str], rows: list[dict], summary: str):
        self._statistics_last_headers = list(headers)
        self._statistics_last_rows = [dict(row) for row in rows]
        table = getattr(self, "_statistics_table", None)
        if table is None:
            return
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(headers)
        for row_idx, row in enumerate(rows):
            for col_idx, header in enumerate(headers):
                value = row.get(header, "")
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(row_idx, col_idx, item)
        table.resizeColumnsToContents()
        if headers:
            table.horizontalHeader().setStretchLastSection(True)
        self._statistics_summary_label.setText(summary)
        self._stats_export_btn.setEnabled(bool(rows))
        self._refresh_statistics_selection_label()

    def _current_and_original_coords_for_statistics(self, path_ids: list[int]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        path_ids = [int(path_id) for path_id in path_ids if int(path_id) > 0]
        if not path_ids:
            return {}, {}
        loaded = dict(getattr(self._viewer, "_bg_data_dict", {}) or {})
        if self._path_coordinate_mode == "current":
            current = {
                int(path_id): coords
                for path_id, coords in loaded.items()
                if int(path_id) in path_ids
            }
            missing_current = set(path_ids) - set(current)
            if missing_current:
                current.update(self.db.get_render_coords(list(missing_current)))
            original = self.db.get_original_render_coords(path_ids)
        else:
            original = {
                int(path_id): coords
                for path_id, coords in loaded.items()
                if int(path_id) in path_ids
            }
            missing_original = set(path_ids) - set(original)
            if missing_original:
                original.update(self.db.get_original_render_coords(list(missing_original)))
            current = self.db.get_render_coords(path_ids)
        return current, original

    def _compute_statistics_entrance_centroid(self):
        path_ids = self._statistics_context_path_ids()
        if not path_ids:
            QMessageBox.information(self, "Statistics", "No selected paths are available.")
            return
        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_points = [
            np.asarray(coords, dtype=np.float64)[0, :3]
            for coords in current.values()
            if np.asarray(coords).ndim == 2 and len(coords) > 0
        ]
        original_points = [
            np.asarray(coords, dtype=np.float64)[0, :3]
            for coords in original.values()
            if np.asarray(coords).ndim == 2 and len(coords) > 0
        ]
        if not current_points and not original_points:
            QMessageBox.information(self, "Statistics", "No entrance coordinates were available for the selected paths.")
            return

        rows: list[dict] = []
        if current_points:
            centroid = np.mean(np.vstack(current_points), axis=0)
            rows.append({
                "Metric": "Mapped/current entrance centroid",
                "Path Count": len(current_points),
                "X": self._format_stat_float(float(centroid[0])),
                "Y": self._format_stat_float(float(centroid[1])),
                "Z": self._format_stat_float(float(centroid[2])),
            })
        if original_points:
            centroid = np.mean(np.vstack(original_points), axis=0)
            rows.append({
                "Metric": "Original entrance centroid",
                "Path Count": len(original_points),
                "X": self._format_stat_float(float(centroid[0])),
                "Y": self._format_stat_float(float(centroid[1])),
                "Z": self._format_stat_float(float(centroid[2])),
            })
        self._set_statistics_table(
            ["Metric", "Path Count", "X", "Y", "Z"],
            rows,
            f"Entrance centroids calculated from {len(path_ids):,} selected paths.",
        )
        self._statusbar.showMessage("Entrance centroid statistics calculated", 3000)

    @staticmethod
    def _mean_centerline_and_distances(coords_by_id: dict[int, np.ndarray], sample_count: int = 64) -> tuple[np.ndarray | None, dict[int, float]]:
        samples: dict[int, np.ndarray] = {}
        for path_id, coords in coords_by_id.items():
            sample = MainWindow._resample_path_coords_interpolated(coords, sample_count=sample_count)
            if sample.ndim == 2 and sample.shape[0] == int(sample_count):
                samples[int(path_id)] = sample.astype(np.float64)
        if not samples:
            return None, {}
        stack = np.stack(list(samples.values()), axis=0)
        centerline = np.mean(stack, axis=0)
        distances = {
            int(path_id): float(np.sum(np.linalg.norm(sample - centerline, axis=1)))
            for path_id, sample in samples.items()
        }
        return centerline, distances

    def _compute_statistics_centerline_distance(self):
        path_ids = self._statistics_context_path_ids()
        if not path_ids:
            QMessageBox.information(self, "Statistics", "No selected paths are available.")
            return
        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_centerline, current_distances = self._mean_centerline_and_distances(current)
        original_centerline, original_distances = self._mean_centerline_and_distances(original)
        if not current_distances and not original_distances:
            QMessageBox.information(self, "Statistics", "No path coordinates were available for centerline statistics.")
            return

        all_ids = sorted(set(current_distances) | set(original_distances))
        rows: list[dict] = []
        for path_id in all_ids:
            current_value = current_distances.get(int(path_id))
            original_value = original_distances.get(int(path_id))
            delta_value = None
            if current_value is not None and original_value is not None:
                delta_value = current_value - original_value
            rows.append({
                "Path": self._format_path_label(int(path_id)),
                "Current Sum Dist": self._format_stat_float(current_value) if current_value is not None else "",
                "Original Sum Dist": self._format_stat_float(original_value) if original_value is not None else "",
                "Delta Current-Original": self._format_stat_float(delta_value) if delta_value is not None else "",
            })

        current_total = float(np.sum(list(current_distances.values()))) if current_distances else float("nan")
        original_total = float(np.sum(list(original_distances.values()))) if original_distances else float("nan")
        summary_parts = []
        if current_centerline is not None:
            summary_parts.append(
                f"mapped/current total summed distance {self._format_stat_float(current_total)}"
            )
        if original_centerline is not None:
            summary_parts.append(
                f"original total summed distance {self._format_stat_float(original_total)}"
            )
        self._set_statistics_table(
            ["Path", "Current Sum Dist", "Original Sum Dist", "Delta Current-Original"],
            rows,
            f"Centerline distances for {len(all_ids):,} paths; " + "; ".join(summary_parts) + ".",
        )
        self._statusbar.showMessage("Centerline distance statistics calculated", 3000)

    def _current_statistics_dataset_key(self) -> str:
        panel = getattr(self, "_dataset_panel", None)
        if panel is not None and hasattr(panel, "current_dataset_key"):
            key = panel.current_dataset_key()
            if key:
                return str(key)
        keys = []
        if panel is not None and hasattr(panel, "selected_dataset_keys"):
            keys = list(panel.selected_dataset_keys() or [])
        if len(keys) == 1:
            return str(keys[0])
        primary_key = getattr(self.db, "primary_dataset_key", None)
        if callable(primary_key):
            value = primary_key()
            return str(value or "")
        return str(primary_key or "")

    def _select_dataset_clusters_by_id_range(self) -> tuple[str, dict[int, set[int]], list[int], int, int] | None:
        dataset_key = self._current_statistics_dataset_key()
        if not dataset_key:
            QMessageBox.information(self, "Statistics", "No current dataset is available.")
            return None
        if not hasattr(self.db, "get_dataset_cluster_path_groups"):
            QMessageBox.information(self, "Statistics", "Cluster statistics are unavailable for the current database.")
            return None

        cluster_groups = self.db.get_dataset_cluster_path_groups(
            dataset_key,
            self.state.frame_min,
            self.state.frame_max,
        )
        available_cluster_ids = sorted(
            int(cluster_id)
            for cluster_id, ids in cluster_groups.items()
            if len(ids or set()) > 0
        )
        if not available_cluster_ids:
            QMessageBox.information(self, "Statistics", "No clusters were found in the current dataset.")
            return None

        dialog = QDialog(self)
        dialog.setWindowTitle("Cluster ID Range")
        dialog_layout = QVBoxLayout(dialog)
        form_layout = QGridLayout()
        start_spin = QSpinBox(dialog)
        end_spin = QSpinBox(dialog)
        for spin in (start_spin, end_spin):
            spin.setRange(0, 1_000_000_000)
            spin.setSingleStep(1)
        default_start = min(available_cluster_ids)
        default_end = min(max(available_cluster_ids), max(default_start, 10))
        start_spin.setValue(int(default_start))
        end_spin.setValue(int(default_end))
        form_layout.addWidget(QLabel("Cluster ID From"), 0, 0)
        form_layout.addWidget(start_spin, 0, 1)
        form_layout.addWidget(QLabel("Cluster ID To"), 1, 0)
        form_layout.addWidget(end_spin, 1, 1)
        dialog_layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        dialog_layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return None
        cluster_id_start = int(start_spin.value())
        cluster_id_end = int(end_spin.value())
        if cluster_id_start > cluster_id_end:
            QMessageBox.information(self, "Statistics", "Cluster ID From must be less than or equal to Cluster ID To.")
            return None
        target_clusters = [
            int(cluster_id)
            for cluster_id, ids in sorted(cluster_groups.items(), key=lambda item: int(item[0]))
            if int(cluster_id) >= int(cluster_id_start)
            and int(cluster_id) <= int(cluster_id_end)
            and len(ids or set()) > 0
        ]
        if not target_clusters:
            QMessageBox.information(
                self,
                "Statistics",
                f"No clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] were found in the current dataset.",
            )
            return None
        return dataset_key, cluster_groups, target_clusters, int(cluster_id_start), int(cluster_id_end)

    @staticmethod
    def _pairwise_path_compactness(
        coords_by_id: dict[int, np.ndarray],
        sample_count: int = 64,
    ) -> dict:
        samples: list[np.ndarray] = []
        for path_id, coords in sorted(coords_by_id.items()):
            sample = MainWindow._resample_path_coords_interpolated(coords, sample_count=sample_count)
            if sample.ndim == 2 and sample.shape[0] == int(sample_count):
                samples.append(sample.astype(np.float64))
        path_count = len(samples)
        if path_count < 2:
            return {"path_count": path_count, "pair_count": 0, "total_pair_count": 0}

        stack = np.stack(samples, axis=0)
        total_pairs = path_count * (path_count - 1) // 2
        distances: list[np.ndarray] = []
        for idx in range(path_count - 1):
            pair_distances = np.sum(
                np.linalg.norm(stack[idx + 1:] - stack[idx], axis=2),
                axis=1,
            )
            if pair_distances.size:
                distances.append(pair_distances.astype(np.float64, copy=False))
        if not distances:
            return {"path_count": path_count, "pair_count": 0, "total_pair_count": int(total_pairs)}
        values = np.concatenate(distances)
        return {
            "path_count": path_count,
            "pair_count": int(values.size),
            "total_pair_count": int(total_pairs),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "std": float(np.std(values)),
        }

    @staticmethod
    def _jaccard_filtered_pairwise_path_compactness(
        current_coords_by_id: dict[int, np.ndarray],
        original_coords_by_id: dict[int, np.ndarray],
        residue_sets_by_id: dict[int, set[int]],
        jaccard_threshold: float,
        atom_sets_by_id: dict[int, set[object]] | None = None,
        atom_jaccard_threshold: float | None = None,
        sample_count: int = 64,
        precomputed_candidates: dict | None = None,
    ) -> dict:
        candidate_info = precomputed_candidates or MainWindow._jaccard_filtered_pair_candidates(
            residue_sets_by_id,
            float(jaccard_threshold),
            atom_sets_by_id,
            atom_jaccard_threshold,
        )
        residue_sets: dict[int, set[int]] = {
            int(path_id): set(values)
            for path_id, values in dict(candidate_info.get("_residue_sets", {})).items()
            if values
        }
        path_count = int(candidate_info.get("path_count", len(residue_sets)) or 0)
        total_pairs = int(candidate_info.get("total_pair_count", path_count * (path_count - 1) // 2) or 0)
        threshold = float(jaccard_threshold)
        current_values: list[float] = []
        original_values: list[float] = []
        delta_values: list[float] = []
        abs_delta_values: list[float] = []
        jaccard_values: list[float] = []
        atom_jaccard_values: list[float] = []
        atom_sets = atom_sets_by_id or {}
        atom_threshold = float(atom_jaccard_threshold) if atom_jaccard_threshold is not None else None
        use_atom_filter = atom_threshold is not None
        candidate_pairs = set(candidate_info.get("_candidate_pairs", set()) or set())
        valid_path_ids = {int(path_id) for pair in candidate_pairs for path_id in pair}
        current_samples: dict[int, np.ndarray] = {}
        original_samples: dict[int, np.ndarray] = {}
        for path_id in sorted(valid_path_ids):
            if int(path_id) not in current_coords_by_id or int(path_id) not in original_coords_by_id:
                continue
            current_sample = MainWindow._resample_path_coords_interpolated(
                current_coords_by_id[int(path_id)],
                sample_count=sample_count,
            )
            original_sample = MainWindow._resample_path_coords_interpolated(
                original_coords_by_id[int(path_id)],
                sample_count=sample_count,
            )
            if (
                current_sample.ndim == 2
                and original_sample.ndim == 2
                and current_sample.shape[0] == int(sample_count)
                and original_sample.shape[0] == int(sample_count)
            ):
                current_samples[int(path_id)] = current_sample.astype(np.float64)
                original_samples[int(path_id)] = original_sample.astype(np.float64)
        increased_pair_count = 0
        decreased_pair_count = 0
        near_zero_pair_count = 0
        near_zero_tolerance = 1e-6

        for left_id, right_id in sorted(candidate_pairs):
            if int(left_id) not in current_samples or int(right_id) not in current_samples:
                continue
            left_set = residue_sets[int(left_id)]
            right_set = residue_sets[int(right_id)]
            union = left_set | right_set
            if not union:
                continue
            jaccard = float(len(left_set & right_set)) / float(len(union))
            if jaccard <= threshold:
                continue

            atom_jaccard: float | None = None
            left_atom_set = atom_sets.get(int(left_id), set())
            right_atom_set = atom_sets.get(int(right_id), set())
            atom_union = left_atom_set | right_atom_set
            if atom_union:
                atom_jaccard = float(len(left_atom_set & right_atom_set)) / float(len(atom_union))
            elif use_atom_filter:
                continue
            if use_atom_filter and (atom_jaccard is None or atom_jaccard <= float(atom_threshold)):
                continue

            current_diff = current_samples[int(right_id)] - current_samples[int(left_id)]
            original_diff = original_samples[int(right_id)] - original_samples[int(left_id)]
            current_distance = float(np.sum(np.linalg.norm(current_diff, axis=1)))
            original_distance = float(np.sum(np.linalg.norm(original_diff, axis=1)))
            distance_delta = current_distance - original_distance
            current_values.append(current_distance)
            original_values.append(original_distance)
            delta_values.append(distance_delta)
            abs_delta_values.append(abs(distance_delta))
            if abs(distance_delta) <= near_zero_tolerance:
                near_zero_pair_count += 1
            elif distance_delta > 0.0:
                increased_pair_count += 1
            else:
                decreased_pair_count += 1
            jaccard_values.append(float(jaccard))
            if atom_jaccard is not None:
                atom_jaccard_values.append(float(atom_jaccard))

        result = {
            "path_count": path_count,
            "pair_count": len(current_values),
            "total_pair_count": int(total_pairs),
            "skipped_pair_count": int(candidate_info.get("skipped_pair_count", 0) or 0),
            "residue_skipped_pair_count": int(candidate_info.get("residue_skipped_pair_count", 0) or 0),
            "atom_skipped_pair_count": int(candidate_info.get("atom_skipped_pair_count", 0) or 0),
            "missing_atom_pair_count": int(candidate_info.get("missing_atom_pair_count", 0) or 0),
            "residue_shared_pair_count": int(candidate_info.get("residue_shared_pair_count", 0) or 0),
            "residue_candidate_pair_count": int(candidate_info.get("residue_candidate_pair_count", 0) or 0),
            "atom_shared_pair_count": int(candidate_info.get("atom_shared_pair_count", 0) or 0),
            "atom_candidate_pair_count": int(candidate_info.get("atom_candidate_pair_count", 0) or 0),
            "atom_pair_count": len(atom_jaccard_values),
            "increased_pair_count": int(increased_pair_count),
            "decreased_pair_count": int(decreased_pair_count),
            "near_zero_pair_count": int(near_zero_pair_count),
        }
        for prefix, values in (
            ("current", current_values),
            ("original", original_values),
            ("distance_delta", delta_values),
            ("abs_distance_delta", abs_delta_values),
            ("jaccard", jaccard_values),
            ("atom_jaccard", atom_jaccard_values),
        ):
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            result[f"{prefix}_mean"] = float(np.mean(arr))
            result[f"{prefix}_median"] = float(np.median(arr))
            result[f"{prefix}_min"] = float(np.min(arr))
            result[f"{prefix}_max"] = float(np.max(arr))
            result[f"{prefix}_std"] = float(np.std(arr))
        return result

    @staticmethod
    def _jaccard_filtered_pair_candidates(
        residue_sets_by_id: dict[int, set[int]],
        jaccard_threshold: float,
        atom_sets_by_id: dict[int, set[object]] | None = None,
        atom_jaccard_threshold: float | None = None,
    ) -> dict:
        residue_sets: dict[int, set[int]] = {}
        for path_id, raw_set in sorted((residue_sets_by_id or {}).items()):
            residue_set = {int(residue_id) for residue_id in raw_set if int(residue_id) > 0}
            if residue_set:
                residue_sets[int(path_id)] = residue_set

        path_ids = sorted(residue_sets)
        path_count = len(path_ids)
        total_pairs = path_count * (path_count - 1) // 2
        if path_count < 2:
            return {
                "_residue_sets": residue_sets,
                "_candidate_pairs": set(),
                "path_count": path_count,
                "total_pair_count": int(total_pairs),
                "skipped_pair_count": 0,
                "residue_skipped_pair_count": 0,
                "atom_skipped_pair_count": 0,
                "missing_atom_pair_count": 0,
                "residue_shared_pair_count": 0,
                "residue_candidate_pair_count": 0,
                "atom_shared_pair_count": 0,
                "atom_candidate_pair_count": 0,
            }

        residue_candidate_pairs, residue_prefix_pair_count = MainWindow._jaccard_candidate_pairs_by_inverted_index(
            residue_sets,
            float(jaccard_threshold),
        )
        atom_candidate_pairs: set[tuple[int, int]] | None = None
        atom_prefix_pair_count = 0
        atom_sets = atom_sets_by_id or {}
        if atom_jaccard_threshold is not None:
            atom_candidate_pairs, atom_prefix_pair_count = MainWindow._jaccard_candidate_pairs_by_inverted_index(
                {
                    int(path_id): set(atom_sets.get(int(path_id), set()))
                    for path_id in path_ids
                    if atom_sets.get(int(path_id))
                },
                float(atom_jaccard_threshold),
            )

        candidate_pairs = set(residue_candidate_pairs)
        atom_skipped_pair_count = 0
        missing_atom_pair_count = 0
        if atom_candidate_pairs is not None:
            candidate_pairs &= atom_candidate_pairs
            for left_id, right_id in residue_candidate_pairs - atom_candidate_pairs:
                left_atom_set = atom_sets.get(int(left_id), set())
                right_atom_set = atom_sets.get(int(right_id), set())
                if not (left_atom_set | right_atom_set):
                    missing_atom_pair_count += 1
                else:
                    atom_skipped_pair_count += 1

        return {
            "_residue_sets": residue_sets,
            "_candidate_pairs": candidate_pairs,
            "path_count": path_count,
            "total_pair_count": int(total_pairs),
            "skipped_pair_count": int(total_pairs) - len(candidate_pairs),
            "residue_skipped_pair_count": int(total_pairs) - len(residue_candidate_pairs),
            "atom_skipped_pair_count": int(atom_skipped_pair_count),
            "missing_atom_pair_count": int(missing_atom_pair_count),
            "residue_shared_pair_count": int(residue_prefix_pair_count),
            "residue_candidate_pair_count": len(residue_candidate_pairs),
            "atom_shared_pair_count": int(atom_prefix_pair_count),
            "atom_candidate_pair_count": len(atom_candidate_pairs) if atom_candidate_pairs is not None else 0,
        }

    @staticmethod
    def _pairwise_residue_similarity_stats(
        residue_sets_by_id: dict[int, set[int]],
    ) -> dict:
        residue_sets = [
            set(residue_sets_by_id[path_id])
            for path_id in sorted(residue_sets_by_id)
            if residue_sets_by_id.get(path_id)
        ]
        path_count = len(residue_sets)
        if path_count < 2:
            unique_residues = set().union(*residue_sets) if residue_sets else set()
            return {
                "path_count": path_count,
                "pair_count": 0,
                "total_pair_count": 0,
                "unique_residue_count": len(unique_residues),
                "core_residue_count": len(unique_residues) if path_count == 1 else 0,
            }

        total_pairs = path_count * (path_count - 1) // 2
        pairs = [
            (left, right)
            for left in range(path_count - 1)
            for right in range(left + 1, path_count)
        ]

        values: list[float] = []
        for left, right in pairs:
            union = residue_sets[int(left)] | residue_sets[int(right)]
            if not union:
                continue
            intersection = residue_sets[int(left)] & residue_sets[int(right)]
            values.append(float(len(intersection)) / float(len(union)))
        unique_residues: set[int] = set().union(*residue_sets)
        core_threshold = max(1, int(np.ceil(path_count * 0.8)))
        residue_counts: dict[int, int] = {}
        for residue_set in residue_sets:
            for residue_id in residue_set:
                residue_counts[int(residue_id)] = residue_counts.get(int(residue_id), 0) + 1
        core_residue_count = sum(1 for count in residue_counts.values() if int(count) >= core_threshold)
        if not values:
            return {
                "path_count": path_count,
                "pair_count": 0,
                "total_pair_count": int(total_pairs),
                "unique_residue_count": len(unique_residues),
                "core_residue_count": int(core_residue_count),
            }
        arr = np.asarray(values, dtype=np.float64)
        return {
            "path_count": path_count,
            "pair_count": int(arr.size),
            "total_pair_count": int(total_pairs),
            "unique_residue_count": len(unique_residues),
            "core_residue_count": int(core_residue_count),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    def _dataset_residue_atom_key_map_for_statistics(self, dataset_key: str) -> dict[int, set[tuple[int, int]]]:
        protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
        aligned_atoms = list(protein_cache.get("atoms_aligned", []) or [])
        if not aligned_atoms:
            return {}

        residue_id_base = 0
        datasets_by_key = getattr(self.db, "_datasets_by_key", {})
        dataset = datasets_by_key.get(str(dataset_key)) if isinstance(datasets_by_key, dict) else None
        if dataset is not None:
            residue_id_base = int(getattr(dataset, "residue_id_base", 0) or 0)

        residue_atom_keys: dict[int, set[tuple[int, int]]] = {}
        for atom_idx, atom in enumerate(aligned_atoms):
            try:
                local_residue_id = int(atom.get("res_seq", 0) or 0)
            except Exception:
                continue
            if local_residue_id <= 0:
                continue
            residue_id = int(residue_id_base) + int(local_residue_id)
            residue_atom_keys.setdefault(residue_id, set()).add((residue_id, int(atom_idx)))
        return residue_atom_keys

    def _dataset_info_for_statistics(self, dataset_key: str) -> dict:
        if hasattr(self.db, "get_dataset"):
            dataset_info = self.db.get_dataset(str(dataset_key))
            if dataset_info:
                return dict(dataset_info)
        return {}

    @staticmethod
    def _find_named_file_near_dataset(dataset_info: dict, filename: str) -> str:
        target = str(filename or "").lower()
        candidates: list[tuple[int, str]] = []
        search_roots = [
            str(dataset_info.get("folder") or ""),
            os.path.dirname(str(dataset_info.get("path") or "")),
            os.path.dirname(str(dataset_info.get("residue_statistics_path") or "")),
        ]
        expanded_roots: list[tuple[int, str]] = []
        for root in search_roots:
            if not root:
                continue
            expanded_roots.append((0, root))
            parent = os.path.dirname(root)
            if parent and parent != root:
                expanded_roots.append((2, parent))
                try:
                    for name in os.listdir(parent):
                        child = os.path.join(parent, name)
                        if os.path.isdir(child):
                            expanded_roots.append((1, child))
                except OSError:
                    pass
        seen_roots: set[str] = set()
        for priority, root in expanded_roots:
            if not root or not os.path.isdir(root):
                continue
            normalized_root = os.path.normcase(os.path.abspath(root))
            if normalized_root in seen_roots:
                continue
            seen_roots.add(normalized_root)
            for current_root, _dirs, files in os.walk(root):
                for name in files:
                    if name.lower() == target:
                        candidates.append((int(priority), os.path.join(current_root, name)))
        if not candidates:
            return ""
        unique: dict[str, int] = {}
        for priority, path in candidates:
            abs_path = os.path.abspath(path)
            unique[abs_path] = min(priority, unique.get(abs_path, priority))
        ranked = sorted(unique.items(), key=lambda item: (item[1], len(item[0]), item[0].lower()))
        return ranked[0][0]

    @staticmethod
    def _find_md_path_file_near_dataset(dataset_info: dict) -> str:
        candidate_roots = [
            str(dataset_info.get("folder") or ""),
            os.path.dirname(str(dataset_info.get("path") or "")),
            os.path.dirname(str(dataset_info.get("residue_statistics_path") or "")),
        ]
        checked: set[str] = set()
        for root in candidate_roots:
            normalized = os.path.abspath(str(root or "").strip()) if str(root or "").strip() else ""
            if not normalized or normalized in checked:
                continue
            checked.add(normalized)
            direct = os.path.join(normalized, "MD_path.txt")
            if os.path.isfile(direct):
                return direct
            parent = os.path.dirname(normalized)
            if parent and parent != normalized:
                parent_direct = os.path.join(parent, "MD_path.txt")
                if os.path.isfile(parent_direct):
                    return parent_direct
        return ""

    @staticmethod
    def _tunnel_id_from_source_file(source_file: str) -> int | None:
        match = re.search(r"tunnel[_-](\d+)", str(source_file or ""), flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _safe_csv_int(value) -> int | None:
        try:
            return int(float(str(value or "").strip()))
        except (TypeError, ValueError):
            return None

    def _all_points_path_atom_keys(self, all_points_path: str) -> dict[int, tuple[int, int]]:
        abs_path = os.path.abspath(all_points_path)
        stat = os.stat(abs_path)
        cache_key = ("all_points_path_keys", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        if cache_key in cache:
            return dict(cache[cache_key])

        result: dict[int, tuple[int, int]] = {}
        path_index = -1
        with open(abs_path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            required = {"res_seq", "source_file", "frame_id"}
            if not required.issubset(set(reader.fieldnames)):
                return {}
            current_key: tuple[int, int] | None = None
            for row in reader:
                res_seq = self._safe_csv_int(row.get("res_seq"))
                if res_seq == 1 or path_index < 0:
                    if current_key is not None and path_index >= 0:
                        result[int(path_index)] = current_key
                    path_index += 1
                    frame_id = self._safe_csv_int(row.get("frame_id"))
                    tunnel_id = self._tunnel_id_from_source_file(str(row.get("source_file") or ""))
                    current_key = (
                        int(frame_id) if frame_id is not None else -1,
                        int(tunnel_id) if tunnel_id is not None else -1,
                    )
            if current_key is not None and path_index >= 0:
                result[int(path_index)] = current_key

        cache[cache_key] = dict(result)
        self._statistics_atom_csv_cache = cache
        return result

    def _atom_network_key_atom_sets(self, atom_network_path: str) -> dict[tuple[int, int], set[str]]:
        abs_path = os.path.abspath(atom_network_path)
        stat = os.stat(abs_path)
        cache_key = ("atom_network_key_sets", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        if cache_key in cache:
            return {key: set(value) for key, value in cache[cache_key].items()}

        result: dict[tuple[int, int], set[str]] = defaultdict(set)
        atom_columns = ("T_F_A_1_ID", "T_F_A_2_ID", "T_F_A_3_ID", "T_F_A_4_ID")
        with open(abs_path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return {}
            required = {"pdb_id", "Tunnel_ID", *atom_columns}
            if not required.issubset(set(reader.fieldnames)):
                return {}
            for row in reader:
                frame_id = self._safe_csv_int(row.get("pdb_id"))
                tunnel_id = self._safe_csv_int(row.get("Tunnel_ID"))
                if frame_id is None or tunnel_id is None or tunnel_id <= 0:
                    continue
                atom_set = result[(int(frame_id), int(tunnel_id))]
                for column in atom_columns:
                    atom_id = str(row.get(column) or "").strip()
                    if atom_id and atom_id != "0" and "_" in atom_id:
                        atom_set.add(atom_id)

        materialized = {key: set(value) for key, value in result.items() if value}
        cache[cache_key] = {key: set(value) for key, value in materialized.items()}
        self._statistics_atom_csv_cache = cache
        return materialized

    @staticmethod
    def _atom_set_from_path_payload(path_payload: dict) -> set[str]:
        atom_set: set[str] = set()
        for key in ("atom_1", "atom_2", "atom_3", "atom_4"):
            for value in path_payload.get(key, []) or []:
                atom_id = str(value or "").strip()
                if atom_id and atom_id != "0":
                    atom_set.add(atom_id)
        return atom_set

    @staticmethod
    def _jaccard_candidate_pairs_by_inverted_index(
        sets_by_id: dict[int, set],
        threshold: float,
    ) -> tuple[set[tuple[int, int]], int]:
        items = {
            int(path_id): set(values)
            for path_id, values in (sets_by_id or {}).items()
            if values
        }
        if len(items) < 2:
            return set(), 0
        threshold = float(threshold)
        if threshold >= 1.0:
            return set(), 0

        value_counts: dict[object, int] = defaultdict(int)
        for values in items.values():
            for value in values:
                value_counts[value] += 1

        ordered_items: dict[int, list[object]] = {}
        for path_id, values in items.items():
            ordered_items[int(path_id)] = sorted(
                values,
                key=lambda value: (int(value_counts[value]), str(type(value)), str(value)),
            )

        # Prefix filtering keeps the Jaccard result exact while reducing the pair
        # candidates inserted into the inverted index.
        inverted: dict[object, list[int]] = defaultdict(list)
        for path_id, values in ordered_items.items():
            if threshold <= 0.0:
                prefix_length = len(values)
            else:
                prefix_length = len(values) - int(math.ceil(threshold * len(values))) + 1
            prefix_length = max(0, min(len(values), int(prefix_length)))
            for value in values[:prefix_length]:
                inverted[value].append(int(path_id))

        potential_pairs: set[tuple[int, int]] = set()
        for ids in inverted.values():
            if len(ids) < 2:
                continue
            ids = sorted(ids)
            for left_idx, left_id in enumerate(ids[:-1]):
                for right_id in ids[left_idx + 1:]:
                    left_len = len(items[int(left_id)])
                    right_len = len(items[int(right_id)])
                    if threshold > 0.0 and float(min(left_len, right_len)) <= threshold * float(max(left_len, right_len)):
                        continue
                    pair = (int(left_id), int(right_id))
                    potential_pairs.add(pair)

        candidates: set[tuple[int, int]] = set()
        for pair in potential_pairs:
            left_id, right_id = pair
            intersection_count = len(items[left_id] & items[right_id])
            union_count = len(items[left_id]) + len(items[right_id]) - int(intersection_count)
            if union_count > 0 and (float(intersection_count) / float(union_count)) > threshold:
                candidates.add(pair)
        return candidates, len(potential_pairs)

    def _dataset_atom_sets_from_preprocessed_pkl(
        self,
        dataset_key: str,
        path_ids: list[int],
    ) -> tuple[dict[int, set[str]], str]:
        dataset_info = self._dataset_info_for_statistics(dataset_key)
        if not dataset_info:
            return {}, ""
        pkl_path = self._find_named_file_near_dataset(dataset_info, "preprocessed_paths.pkl")
        if not pkl_path:
            return {}, ""

        abs_path = os.path.abspath(pkl_path)
        stat = os.stat(abs_path)
        cache_key = ("preprocessed_pkl_atom_sets", abs_path, int(stat.st_mtime_ns), int(stat.st_size))
        cache = getattr(self, "_statistics_atom_csv_cache", {})
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        path_base = int(getattr(dataset_binding, "path_id_base", 0) or 0) if dataset_binding is not None else 0
        local_ids = sorted({int(path_id) - path_base for path_id in path_ids if int(path_id) - path_base >= 0})
        if cache_key in cache:
            cached_sets = cache[cache_key]
            return {
                int(local_id) + path_base: set(cached_sets[int(local_id)])
                for local_id in local_ids
                if int(local_id) in cached_sets
            }, abs_path

        with open(abs_path, "rb") as handle:
            data = pickle.load(handle)
        paths = list((data or {}).get("PATHS", []) if isinstance(data, dict) else [])
        cached_sets: dict[int, frozenset[str]] = {}
        for local_id, path_payload in enumerate(paths):
            if not isinstance(path_payload, dict):
                continue
            atom_set = self._atom_set_from_path_payload(path_payload)
            if atom_set:
                cached_sets[int(local_id)] = frozenset(atom_set)
        cache[cache_key] = cached_sets
        self._statistics_atom_csv_cache = cache

        result: dict[int, set[str]] = {}
        for local_id in local_ids:
            atom_set = cached_sets.get(int(local_id))
            if atom_set:
                result[int(local_id) + path_base] = set(atom_set)
        return result, abs_path

    def _dataset_atom_sets_from_network_csv(
        self,
        dataset_key: str,
        path_ids: list[int],
    ) -> tuple[dict[int, set[str]], str]:
        dataset_info = self._dataset_info_for_statistics(dataset_key)
        if not dataset_info:
            return {}, ""
        atom_network_path = self._find_named_file_near_dataset(dataset_info, "atom_network.csv")
        if not atom_network_path:
            atom_network_path = self._find_named_file_near_dataset(dataset_info, "1.atom_network.csv")
        all_points_path = self._find_named_file_near_dataset(dataset_info, "mapped_path_points.csv")
        if not all_points_path:
            all_points_path = self._find_named_file_near_dataset(dataset_info, "4.all_points.csv")
        if not atom_network_path or not all_points_path:
            return {}, atom_network_path or ""

        path_keys = self._all_points_path_atom_keys(all_points_path)
        key_atom_sets = self._atom_network_key_atom_sets(atom_network_path)
        if not path_keys or not key_atom_sets:
            return {}, atom_network_path

        result: dict[int, set[str]] = {}
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        path_base = int(getattr(dataset_binding, "path_id_base", 0) or 0) if dataset_binding is not None else 0
        for path_id in path_ids:
            local_path_id = int(path_id) - path_base
            key = path_keys.get(int(local_path_id))
            if key is None:
                continue
            atom_set = key_atom_sets.get(key)
            if atom_set:
                result[int(path_id)] = set(atom_set)
        return result, atom_network_path

    def _compute_dataset_cluster_results_export(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection
        path_ids: list[int] = []
        cluster_rows: list[tuple[int, int]] = []
        for cluster_id in target_clusters:
            ids = sorted(int(path_id) for path_id in cluster_groups.get(int(cluster_id), set()) if int(path_id) > 0)
            for path_id in ids:
                path_ids.append(int(path_id))
                cluster_rows.append((int(cluster_id), int(path_id)))
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        current, original = self._current_and_original_coords_for_statistics(path_ids)
        current_distances: dict[int, float] = {}
        original_distances: dict[int, float] = {}
        path_id_set = set(path_ids)
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) in path_id_set
            }
            if not cluster_path_ids:
                continue
            _current_centerline, cluster_current_distances = self._mean_centerline_and_distances({
                int(path_id): coords
                for path_id, coords in current.items()
                if int(path_id) in cluster_path_ids
            })
            _original_centerline, cluster_original_distances = self._mean_centerline_and_distances({
                int(path_id): coords
                for path_id, coords in original.items()
                if int(path_id) in cluster_path_ids
            })
            current_distances.update(cluster_current_distances)
            original_distances.update(cluster_original_distances)

        headers = ["Cluster", "Path", "Current Sum Dist", "Original Sum Dist", "Delta Current-Original"]
        rows: list[dict] = []
        for cluster_id, path_id in cluster_rows:
            current_value = current_distances.get(int(path_id))
            original_value = original_distances.get(int(path_id))
            delta_value = None
            if current_value is not None and original_value is not None:
                delta_value = current_value - original_value
            rows.append({
                "Cluster": str(cluster_id),
                "Path": self._format_path_label(int(path_id)),
                "Current Sum Dist": self._format_stat_float(current_value) if current_value is not None else "",
                "Original Sum Dist": self._format_stat_float(original_value) if original_value is not None else "",
                "Delta Current-Original": self._format_stat_float(delta_value) if delta_value is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        present_clusters = sorted({int(cluster_id) for cluster_id, _path_id in cluster_rows})
        current_total = float(np.sum(list(current_distances.values()))) if current_distances else float("nan")
        original_total = float(np.sum(list(original_distances.values()))) if original_distances else float("nan")
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster centerline distances for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"{len(rows):,} paths across clusters {', '.join(str(cid) for cid in present_clusters)}; "
                f"mapped/current total summed distance {self._format_stat_float(current_total)}; "
                f"original total summed distance {self._format_stat_float(original_total)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster results for export: {dataset_label} ({len(rows)} paths, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _compute_dataset_cluster_compactness(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection

        path_ids: list[int] = []
        for cluster_id in target_clusters:
            path_ids.extend(
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            )
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        threshold_dialog = QDialog(self)
        threshold_dialog.setWindowTitle("Jaccard Thresholds")
        threshold_layout = QVBoxLayout(threshold_dialog)
        threshold_form = QGridLayout()
        residue_threshold_spin = QDoubleSpinBox(threshold_dialog)
        atom_threshold_spin = QDoubleSpinBox(threshold_dialog)
        for spin in (residue_threshold_spin, atom_threshold_spin):
            spin.setRange(0.0, 1.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
        residue_threshold_spin.setValue(0.500)
        atom_threshold_spin.setValue(0.500)
        threshold_form.addWidget(QLabel("Residue Jaccard >"), 0, 0)
        threshold_form.addWidget(residue_threshold_spin, 0, 1)
        threshold_form.addWidget(QLabel("Atom Jaccard >"), 1, 0)
        threshold_form.addWidget(atom_threshold_spin, 1, 1)
        threshold_layout.addLayout(threshold_form)
        threshold_buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, threshold_dialog)
        threshold_buttons.accepted.connect(threshold_dialog.accept)
        threshold_buttons.rejected.connect(threshold_dialog.reject)
        threshold_layout.addWidget(threshold_buttons)
        if threshold_dialog.exec() != QDialog.Accepted:
            return
        jaccard_threshold = float(residue_threshold_spin.value())
        atom_jaccard_threshold = float(atom_threshold_spin.value())

        profiles = self.db.get_path_profiles(path_ids)
        residue_sets_by_id: dict[int, set[int]] = {}
        for profile in profiles:
            path_id = int(profile.get("pathIndex", 0) or 0)
            if path_id <= 0:
                continue
            residue_set = profile_residue_set(profile)
            if residue_set:
                residue_sets_by_id[int(path_id)] = set(residue_set)
        if not residue_sets_by_id:
            QMessageBox.information(self, "Statistics", "No residue composition data were available for the selected clusters.")
            return

        self._statusbar.showMessage("Loading path atom sets from preprocessed data...", 4000)
        QApplication.processEvents()
        atom_sets_by_id, atom_source_path = self._dataset_atom_sets_from_preprocessed_pkl(dataset_key, path_ids)
        if not atom_sets_by_id:
            atom_sets_by_id, atom_source_path = self._dataset_atom_sets_from_network_csv(dataset_key, path_ids)
        residue_atom_key_map: dict[int, set[tuple[int, int]]] = {}
        if not atom_sets_by_id:
            residue_atom_key_map = self._dataset_residue_atom_key_map_for_statistics(dataset_key)
            if residue_atom_key_map:
                for path_id, residue_set in residue_sets_by_id.items():
                    atom_set: set[tuple[int, int]] = set()
                    for residue_id in residue_set:
                        atom_set.update(residue_atom_key_map.get(int(residue_id), set()))
                    if atom_set:
                        atom_sets_by_id[int(path_id)] = atom_set
        if not atom_sets_by_id:
            protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
            aligned_atom_count = len(list(protein_cache.get("atoms_aligned", []) or []))
            selected_residue_ids = set().union(*residue_sets_by_id.values()) if residue_sets_by_id else set()
            atom_residue_ids = set(residue_atom_key_map)
            matched_residue_ids = selected_residue_ids & atom_residue_ids
            sample_path_residues = ", ".join(str(value) for value in sorted(selected_residue_ids)[:8])
            sample_atom_residues = ", ".join(str(value) for value in sorted(atom_residue_ids)[:8])
            dataset_info = self._dataset_info_for_statistics(dataset_key)
            dataset_folder = str(dataset_info.get("folder") or "")
            QMessageBox.information(
                self,
                "Statistics",
                (
                    "No atom composition data were available for the selected paths.\n\n"
                    f"Dataset folder: {dataset_folder or 'unknown'}\n"
                    f"Atom source: {atom_source_path or 'not found'}\n"
                    f"Loaded protein atoms: {aligned_atom_count:,}\n"
                    f"Path residue IDs: {len(selected_residue_ids):,}"
                    + (f" (sample: {sample_path_residues})" if sample_path_residues else "")
                    + "\n"
                    f"Protein atom residue IDs: {len(atom_residue_ids):,}"
                    + (f" (sample: {sample_atom_residues})" if sample_atom_residues else "")
                    + "\n"
                    f"Matched residue IDs: {len(matched_residue_ids):,}\n\n"
                    "Load a protein model first, or check that the protein residue numbering matches the path residue IDs."
                ),
            )
            return

        headers = [
            "Cluster",
            "Path Count",
            "Residue Path Count",
            "Candidate Path Count",
            "Can Test",
            "Pair Count",
            "Residue Candidate Pairs",
            "Atom Candidate Pairs",
            "Residue Shared Pairs",
            "Atom Shared Pairs",
            "Skipped Pair Count",
            "Residue Skipped Pairs",
            "Atom Skipped Pairs",
            "Missing Atom Pairs",
            "Total Pair Count",
            "Mean Residue Jaccard",
            "Min Residue Jaccard",
            "Max Residue Jaccard",
            "Atom Path Count",
            "Atom Pair Count",
            "Mean Atom Jaccard",
            "Median Atom Jaccard",
            "Min Atom Jaccard",
            "Max Atom Jaccard",
            "Std Atom Jaccard",
            "Topo-Spatial Gap Mean",
            "Topo-Spatial Gap Median",
            "Abs Gap Mean",
            "Abs Gap Median",
            "Gap Min",
            "Gap Max",
            "Gap Std",
            "Topo Greater Pairs",
            "Topo Smaller Pairs",
            "Gap Stable Pairs",
            "Topo Pair Dist Mean",
            "Spatial Pair Dist Mean",
            "Topo Pair Dist Median",
            "Spatial Pair Dist Median",
            "Topo Pair Dist Min",
            "Spatial Pair Dist Min",
            "Topo Pair Dist Max",
            "Spatial Pair Dist Max",
            "Topo Pair Dist Std",
            "Spatial Pair Dist Std",
        ]
        rows: list[dict] = []
        current_weighted_total = 0.0
        current_weighted_count = 0
        original_weighted_total = 0.0
        original_weighted_count = 0
        delta_weighted_total = 0.0
        delta_weighted_count = 0
        abs_delta_weighted_total = 0.0
        abs_delta_weighted_count = 0
        atom_jaccard_weighted_total = 0.0
        atom_jaccard_weighted_count = 0
        testable_clusters: list[int] = []
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            }
            cluster_residue_sets = {
                int(path_id): residue_set
                for path_id, residue_set in residue_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            }
            residue_path_count = len(cluster_residue_sets)
            cluster_atom_sets = {
                int(path_id): atom_set
                for path_id, atom_set in atom_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            }
            candidate_info = self._jaccard_filtered_pair_candidates(
                cluster_residue_sets,
                float(jaccard_threshold),
                cluster_atom_sets,
                float(atom_jaccard_threshold),
            )
            candidate_path_ids = sorted({
                int(path_id)
                for pair in set(candidate_info.get("_candidate_pairs", set()) or set())
                for path_id in pair
            })
            if candidate_path_ids:
                current, original = self._current_and_original_coords_for_statistics(candidate_path_ids)
            else:
                current, original = {}, {}
            stats = self._jaccard_filtered_pairwise_path_compactness(
                current,
                original,
                cluster_residue_sets,
                float(jaccard_threshold),
                cluster_atom_sets,
                float(atom_jaccard_threshold),
                precomputed_candidates=candidate_info,
            )
            current_mean = stats.get("current_mean")
            original_mean = stats.get("original_mean")
            current_median = stats.get("current_median")
            original_median = stats.get("original_median")
            delta_pair_mean = stats.get("distance_delta_mean")
            abs_delta_pair_mean = stats.get("abs_distance_delta_mean")

            pair_count = int(stats.get("pair_count", 0) or 0)
            if pair_count > 0:
                testable_clusters.append(int(cluster_id))
            atom_pair_count = int(stats.get("atom_pair_count", 0) or 0)
            atom_jaccard_mean = stats.get("atom_jaccard_mean")
            if atom_jaccard_mean is not None and atom_pair_count > 0:
                atom_jaccard_weighted_total += float(atom_jaccard_mean) * atom_pair_count
                atom_jaccard_weighted_count += atom_pair_count
            if current_mean is not None and pair_count > 0:
                current_weighted_total += float(current_mean) * pair_count
                current_weighted_count += pair_count
            if original_mean is not None and pair_count > 0:
                original_weighted_total += float(original_mean) * pair_count
                original_weighted_count += pair_count
            if delta_pair_mean is not None and pair_count > 0:
                delta_weighted_total += float(delta_pair_mean) * pair_count
                delta_weighted_count += pair_count
            if abs_delta_pair_mean is not None and pair_count > 0:
                abs_delta_weighted_total += float(abs_delta_pair_mean) * pair_count
                abs_delta_weighted_count += pair_count

            rows.append({
                "Cluster": str(cluster_id),
                "Path Count": str(len(cluster_path_ids)),
                "Residue Path Count": str(residue_path_count),
                "Candidate Path Count": str(int(stats.get("path_count", 0) or 0)),
                "Can Test": "Yes" if pair_count > 0 else "No",
                "Pair Count": str(pair_count),
                "Residue Candidate Pairs": str(int(stats.get("residue_candidate_pair_count", 0) or 0)),
                "Atom Candidate Pairs": str(int(stats.get("atom_candidate_pair_count", 0) or 0)),
                "Residue Shared Pairs": str(int(stats.get("residue_shared_pair_count", 0) or 0)),
                "Atom Shared Pairs": str(int(stats.get("atom_shared_pair_count", 0) or 0)),
                "Skipped Pair Count": str(int(stats.get("skipped_pair_count", 0) or 0)),
                "Residue Skipped Pairs": str(int(stats.get("residue_skipped_pair_count", 0) or 0)),
                "Atom Skipped Pairs": str(int(stats.get("atom_skipped_pair_count", 0) or 0)),
                "Missing Atom Pairs": str(int(stats.get("missing_atom_pair_count", 0) or 0)),
                "Total Pair Count": str(int(stats.get("total_pair_count", 0) or 0)),
                "Mean Residue Jaccard": self._format_stat_float(stats.get("jaccard_mean")) if stats.get("jaccard_mean") is not None else "",
                "Min Residue Jaccard": self._format_stat_float(stats.get("jaccard_min")) if stats.get("jaccard_min") is not None else "",
                "Max Residue Jaccard": self._format_stat_float(stats.get("jaccard_max")) if stats.get("jaccard_max") is not None else "",
                "Atom Path Count": str(sum(1 for path_id in cluster_path_ids if int(path_id) in atom_sets_by_id)),
                "Atom Pair Count": str(atom_pair_count),
                "Mean Atom Jaccard": self._format_stat_float(atom_jaccard_mean) if atom_jaccard_mean is not None else "",
                "Median Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_median")) if stats.get("atom_jaccard_median") is not None else "",
                "Min Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_min")) if stats.get("atom_jaccard_min") is not None else "",
                "Max Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_max")) if stats.get("atom_jaccard_max") is not None else "",
                "Std Atom Jaccard": self._format_stat_float(stats.get("atom_jaccard_std")) if stats.get("atom_jaccard_std") is not None else "",
                "Topo-Spatial Gap Mean": self._format_stat_float(delta_pair_mean) if delta_pair_mean is not None else "",
                "Topo-Spatial Gap Median": self._format_stat_float(stats.get("distance_delta_median")) if stats.get("distance_delta_median") is not None else "",
                "Abs Gap Mean": self._format_stat_float(abs_delta_pair_mean) if abs_delta_pair_mean is not None else "",
                "Abs Gap Median": self._format_stat_float(stats.get("abs_distance_delta_median")) if stats.get("abs_distance_delta_median") is not None else "",
                "Gap Min": self._format_stat_float(stats.get("distance_delta_min")) if stats.get("distance_delta_min") is not None else "",
                "Gap Max": self._format_stat_float(stats.get("distance_delta_max")) if stats.get("distance_delta_max") is not None else "",
                "Gap Std": self._format_stat_float(stats.get("distance_delta_std")) if stats.get("distance_delta_std") is not None else "",
                "Topo Greater Pairs": str(int(stats.get("increased_pair_count", 0) or 0)),
                "Topo Smaller Pairs": str(int(stats.get("decreased_pair_count", 0) or 0)),
                "Gap Stable Pairs": str(int(stats.get("near_zero_pair_count", 0) or 0)),
                "Topo Pair Dist Mean": self._format_stat_float(current_mean) if current_mean is not None else "",
                "Spatial Pair Dist Mean": self._format_stat_float(original_mean) if original_mean is not None else "",
                "Topo Pair Dist Median": self._format_stat_float(current_median) if current_median is not None else "",
                "Spatial Pair Dist Median": self._format_stat_float(original_median) if original_median is not None else "",
                "Topo Pair Dist Min": self._format_stat_float(stats.get("current_min")) if stats.get("current_min") is not None else "",
                "Spatial Pair Dist Min": self._format_stat_float(stats.get("original_min")) if stats.get("original_min") is not None else "",
                "Topo Pair Dist Max": self._format_stat_float(stats.get("current_max")) if stats.get("current_max") is not None else "",
                "Spatial Pair Dist Max": self._format_stat_float(stats.get("original_max")) if stats.get("original_max") is not None else "",
                "Topo Pair Dist Std": self._format_stat_float(stats.get("current_std")) if stats.get("current_std") is not None else "",
                "Spatial Pair Dist Std": self._format_stat_float(stats.get("original_std")) if stats.get("original_std") is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        current_mean = (current_weighted_total / current_weighted_count) if current_weighted_count > 0 else float("nan")
        original_mean = (original_weighted_total / original_weighted_count) if original_weighted_count > 0 else float("nan")
        delta_mean = (delta_weighted_total / delta_weighted_count) if delta_weighted_count > 0 else float("nan")
        abs_delta_mean = (
            abs_delta_weighted_total / abs_delta_weighted_count
            if abs_delta_weighted_count > 0
            else float("nan")
        )
        atom_jaccard_mean = (
            atom_jaccard_weighted_total / atom_jaccard_weighted_count
            if atom_jaccard_weighted_count > 0
            else float("nan")
        )
        if testable_clusters:
            visible_testable_clusters = ", ".join(str(cluster_id) for cluster_id in testable_clusters[:50])
            if len(testable_clusters) > 50:
                visible_testable_clusters += f", ... (+{len(testable_clusters) - 50} more)"
        else:
            visible_testable_clusters = "none"
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster compactness for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"residue Jaccard > {float(jaccard_threshold):.3f}; "
                f"atom Jaccard > {float(atom_jaccard_threshold):.3f}; "
                f"{len(rows):,} clusters; "
                f"{len(testable_clusters):,} can be tested ({visible_testable_clusters}); "
                f"weighted atom Jaccard {self._format_stat_float(atom_jaccard_mean)}; "
                f"weighted topo pair distance {self._format_stat_float(current_mean)}; "
                f"weighted spatial pair distance {self._format_stat_float(original_mean)}; "
                f"weighted topo-spatial gap {self._format_stat_float(delta_mean)}; "
                f"weighted abs gap {self._format_stat_float(abs_delta_mean)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster compactness: {dataset_label} ({len(rows)} clusters, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _compute_dataset_cluster_residue_similarity(self):
        selection = self._select_dataset_clusters_by_id_range()
        if selection is None:
            return
        dataset_key, cluster_groups, target_clusters, cluster_id_start, cluster_id_end = selection

        path_ids: list[int] = []
        for cluster_id in target_clusters:
            path_ids.extend(
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            )
        if not path_ids:
            QMessageBox.information(
                self,
                "Statistics",
                f"No paths were found for clusters with ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}] in the current dataset.",
            )
            return

        profiles = self.db.get_path_profiles(path_ids)
        residue_sets_by_id: dict[int, set[int]] = {}
        for profile in profiles:
            path_id = int(profile.get("pathIndex", 0) or 0)
            if path_id <= 0:
                continue
            residue_set = profile_residue_set(profile)
            if residue_set:
                residue_sets_by_id[int(path_id)] = set(residue_set)
        if not residue_sets_by_id:
            QMessageBox.information(self, "Statistics", "No residue composition data were available for the selected clusters.")
            return

        headers = [
            "Cluster",
            "Path Count",
            "Residue Path Count",
            "Pair Count",
            "Unique Residue Count",
            "Core Residue Count",
            "Mean Similarity",
            "Median Similarity",
            "Min Similarity",
            "Max Similarity",
            "Std Similarity",
        ]
        rows: list[dict] = []
        weighted_total = 0.0
        weighted_count = 0
        for cluster_id in target_clusters:
            cluster_path_ids = {
                int(path_id)
                for path_id in cluster_groups.get(int(cluster_id), set())
                if int(path_id) > 0
            }
            stats = self._pairwise_residue_similarity_stats({
                int(path_id): residue_set
                for path_id, residue_set in residue_sets_by_id.items()
                if int(path_id) in cluster_path_ids
            })
            pair_count = int(stats.get("pair_count", 0) or 0)
            mean_similarity = stats.get("mean")
            if mean_similarity is not None and pair_count > 0:
                weighted_total += float(mean_similarity) * pair_count
                weighted_count += pair_count
            rows.append({
                "Cluster": str(cluster_id),
                "Path Count": str(len(cluster_path_ids)),
                "Residue Path Count": str(int(stats.get("path_count", 0) or 0)),
                "Pair Count": str(pair_count),
                "Unique Residue Count": str(int(stats.get("unique_residue_count", 0) or 0)),
                "Core Residue Count": str(int(stats.get("core_residue_count", 0) or 0)),
                "Mean Similarity": self._format_stat_float(mean_similarity) if mean_similarity is not None else "",
                "Median Similarity": self._format_stat_float(stats.get("median")) if stats.get("median") is not None else "",
                "Min Similarity": self._format_stat_float(stats.get("min")) if stats.get("min") is not None else "",
                "Max Similarity": self._format_stat_float(stats.get("max")) if stats.get("max") is not None else "",
                "Std Similarity": self._format_stat_float(stats.get("std")) if stats.get("std") is not None else "",
            })

        dataset_info = getattr(self.db, "get_dataset", lambda _key: None)(dataset_key) if hasattr(self.db, "get_dataset") else None
        dataset_label = str((dataset_info or {}).get("prefix") or dataset_key)
        weighted_mean = (weighted_total / weighted_count) if weighted_count > 0 else float("nan")
        self._set_statistics_table(
            headers,
            rows,
            (
                f"Cluster residue similarity for dataset {dataset_label}; "
                f"cluster ID in [{int(cluster_id_start):,}, {int(cluster_id_end):,}]; "
                f"{len(rows):,} clusters; "
                f"weighted mean similarity {self._format_stat_float(weighted_mean)}."
            ),
        )
        self._statusbar.showMessage(
            f"Prepared cluster residue similarity: {dataset_label} ({len(rows)} clusters, cluster ID {int(cluster_id_start):,}-{int(cluster_id_end):,})",
            4000,
        )

    def _export_statistics_csv(self):
        headers = list(getattr(self, "_statistics_last_headers", []) or [])
        rows = list(getattr(self, "_statistics_last_rows", []) or [])
        if not headers or not rows:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Statistics CSV",
            "path_statistics.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=headers)
                writer.writeheader()
                for row in rows:
                    writer.writerow({header: row.get(header, "") for header in headers})
        except OSError as exc:
            QMessageBox.warning(self, "Export Statistics CSV", f"Failed to export CSV:\n{exc}")
            return
        self._statusbar.showMessage(f"Statistics exported: {path}", 4000)

    @staticmethod
    def _resample_path_coords(coords: np.ndarray, sample_count: int = 64) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[:, :3]
        if arr.shape[0] <= sample_count:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, int(sample_count), dtype=np.int64)
        return arr[idx]

    @staticmethod
    def _resample_path_coords_interpolated(coords: np.ndarray, sample_count: int = 64) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)
        arr = arr[:, :3]
        if int(sample_count) <= 1:
            return arr[:1].copy()

        segment_lengths = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 1e-8:
            return np.repeat(arr[:1], int(sample_count), axis=0)

        targets = np.linspace(0.0, total_length, int(sample_count), dtype=np.float64)
        result = np.empty((int(sample_count), 3), dtype=np.float64)
        for axis in range(3):
            result[:, axis] = np.interp(targets, cumulative, arr[:, axis])
        return result

    def _path_change_signature(self, path_ids: set[int]) -> tuple:
        return (
            tuple(sorted(int(path_id) for path_id in path_ids)),
            self.state.frame_min,
            self.state.frame_max,
            self._path_coordinate_mode,
        )

    def _path_change_distance_metric(self, path_ids: set[int]) -> dict[int, float]:
        path_ids = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        signature = self._path_change_signature(path_ids)
        if signature == self._path_change_metric_cache_signature:
            return dict(self._path_change_metric_cache)
        if not path_ids:
            return {}

        loaded_coords = dict(getattr(self._viewer, "_bg_data_dict", {}) or {})
        if self._path_coordinate_mode == "current":
            current_by_id = {
                int(path_id): coords
                for path_id, coords in loaded_coords.items()
                if int(path_id) in path_ids
            }
            missing_current = path_ids - set(current_by_id)
            if missing_current:
                current_by_id.update(self.db.get_render_coords(list(missing_current)))
            original_by_id = self.db.get_original_render_coords(list(path_ids))
        else:
            original_by_id = {
                int(path_id): coords
                for path_id, coords in loaded_coords.items()
                if int(path_id) in path_ids
            }
            missing_original = path_ids - set(original_by_id)
            if missing_original:
                original_by_id.update(self.db.get_original_render_coords(list(missing_original)))
            current_by_id = self.db.get_render_coords(list(path_ids))
        metrics: dict[int, float] = {}
        for path_id in sorted(path_ids):
            current = current_by_id.get(int(path_id))
            original = original_by_id.get(int(path_id))
            if current is None or original is None:
                continue
            current_sample = self._resample_path_coords(current)
            original_sample = self._resample_path_coords(original, sample_count=len(current_sample) or 64)
            count = min(len(current_sample), len(original_sample))
            if count <= 0:
                continue
            diff = current_sample[:count] - original_sample[:count]
            distances = np.sqrt(np.sum(diff * diff, axis=1))
            if distances.size:
                metrics[int(path_id)] = float(np.mean(distances))

        self._path_change_metric_cache_signature = signature
        self._path_change_metric_cache = dict(metrics)
        return metrics

    def _exit_delta_signature(self, path_ids: set[int]) -> tuple:
        return (
            tuple(sorted(int(path_id) for path_id in path_ids)),
            self.state.frame_min,
            self.state.frame_max,
        )

    def _exit_cluster_distance_delta_metric(self, path_ids: set[int]) -> dict[int, float]:
        path_ids = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        signature = self._exit_delta_signature(path_ids)
        if signature == self._exit_delta_metric_cache_signature:
            return dict(self._exit_delta_metric_cache)
        if not path_ids or not hasattr(self.db, "get_path_exit_points"):
            return {}

        endpoints = self.db.get_path_exit_points(list(path_ids))
        groups: dict[tuple[str, int], list[int]] = {}
        for path_id, payload in endpoints.items():
            dataset_key = str(
                payload.get("dataset_key")
                or getattr(self.db, "get_dataset_key_for_path_id", lambda _pid: "")(int(path_id))
                or ""
            )
            cluster_id = int(payload.get("cluster_id", 0) or 0)
            groups.setdefault((dataset_key, cluster_id), []).append(int(path_id))

        metrics: dict[int, float] = {}
        for _group_key, group_path_ids in groups.items():
            valid_path_ids = [
                int(path_id)
                for path_id in group_path_ids
                if int(path_id) in endpoints
            ]
            if not valid_path_ids:
                continue
            current_points = np.vstack([
                np.asarray(endpoints[path_id]["current_exit"], dtype=np.float64)
                for path_id in valid_path_ids
            ])
            original_points = np.vstack([
                np.asarray(endpoints[path_id]["original_exit"], dtype=np.float64)
                for path_id in valid_path_ids
            ])
            current_center = np.mean(current_points, axis=0)
            original_center = np.mean(original_points, axis=0)
            for path_id in valid_path_ids:
                current_exit = np.asarray(endpoints[path_id]["current_exit"], dtype=np.float64)
                original_exit = np.asarray(endpoints[path_id]["original_exit"], dtype=np.float64)
                current_distance = float(np.linalg.norm(current_exit - current_center))
                original_distance = float(np.linalg.norm(original_exit - original_center))
                metrics[int(path_id)] = abs(current_distance - original_distance)

        self._exit_delta_metric_cache_signature = signature
        self._exit_delta_metric_cache = dict(metrics)
        return metrics

    @staticmethod
    def _format_path_delta_threshold(value: float) -> str:
        value = float(value)
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        return f"{value:.2f}"

    def _base_display_excluded_path_ids(self) -> set[int]:
        return (
            self.state.get_excluded_path_ids()
            | self._hidden_dataset_path_ids()
        )

    def _sync_path_delta_filter_controls(self, metrics: dict[int, float] | None = None):
        slider = getattr(self, "_path_delta_filter_slider", None)
        label = getattr(self, "_path_delta_filter_label", None)
        if slider is None or label is None:
            return
        metrics = metrics or {}
        active_label = "Exit Δ" if self._exit_delta_overlay_active else "Path Δ"
        overlay_active = bool(self._path_change_overlay_active or self._exit_delta_overlay_active)
        enabled = bool(overlay_active and metrics)
        slider.setEnabled(enabled)
        if not enabled or slider.value() <= 0:
            label.setText(f"{active_label} off" if overlay_active else "Δ off")
            label.setToolTip(f"{active_label} exclusion threshold" if overlay_active else "Active Δ exclusion threshold")
            return
        threshold = float(getattr(self, "_path_delta_filter_threshold", 0.0) or 0.0)
        label.setText(f"{active_label}≤{self._format_path_delta_threshold(threshold)}")
        label.setToolTip(
            f"Excluding paths with {active_label} <= {self._format_path_delta_threshold(threshold)}"
        )

    def _clear_path_delta_exclusion(self, *, reset_slider: bool = False, refresh: bool = True):
        changed = bool(getattr(self, "_path_delta_excluded_path_ids", set()))
        self._path_delta_excluded_path_ids.clear()
        self._path_delta_filter_threshold = 0.0
        slider = getattr(self, "_path_delta_filter_slider", None)
        if reset_slider and slider is not None and slider.value() != 0:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
        self._sync_path_delta_filter_controls({})
        if changed and refresh:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()

    def _apply_path_delta_exclusion(self, metrics: dict[int, float]) -> bool:
        slider = getattr(self, "_path_delta_filter_slider", None)
        overlay_active = bool(self._path_change_overlay_active or self._exit_delta_overlay_active)
        if slider is None or not overlay_active or not metrics or slider.value() <= 0:
            previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
            self._path_delta_excluded_path_ids.clear()
            self._path_delta_filter_threshold = 0.0
            self._sync_path_delta_filter_controls(metrics if overlay_active else {})
            return bool(previous)

        finite_values = [
            float(value)
            for value in metrics.values()
            if np.isfinite(float(value))
        ]
        if not finite_values:
            previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
            self._path_delta_excluded_path_ids.clear()
            self._path_delta_filter_threshold = 0.0
            self._sync_path_delta_filter_controls({})
            return bool(previous)

        max_delta = max(finite_values)
        threshold = max_delta * (float(slider.value()) / 100.0)
        self._path_delta_filter_threshold = threshold
        excluded = {
            int(path_id)
            for path_id, value in metrics.items()
            if np.isfinite(float(value)) and float(value) <= threshold
        }
        previous = set(getattr(self, "_path_delta_excluded_path_ids", set()))
        self._path_delta_excluded_path_ids = excluded
        self._sync_path_delta_filter_controls(metrics)
        return previous != excluded

    def _on_path_delta_filter_changed(self, _value: int):
        if self._path_change_overlay_active:
            self._apply_path_change_overlay()
            return
        if self._exit_delta_overlay_active:
            self._apply_exit_delta_overlay()
            return
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=False, refresh=True)
            return

    def _apply_path_change_overlay(self):
        visible_ids = set(getattr(self._viewer, "_bg_data_dict", {}).keys())
        visible_ids -= self._base_display_excluded_path_ids()
        metrics = self._path_change_distance_metric(visible_ids)
        delta_exclusion_changed = self._apply_path_delta_exclusion(metrics)
        self._viewer.set_path_change_metric_map(
            metrics,
            self._display_excluded_path_ids(),
            cmap="Reds",
        )
        if delta_exclusion_changed:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
        self._viewer.set_bg_visible(True)
        self._viewer.set_dataset_visible(False)
        if metrics:
            max_distance = max(metrics.values())
            hidden_count = len(getattr(self, "_path_delta_excluded_path_ids", set()))
            hidden_suffix = f", Δ-filter hidden {hidden_count}" if hidden_count else ""
            self._statusbar.showMessage(
                f"Path change distance overlay: {len(metrics)} paths, max mean distance {max_distance:.3f}{hidden_suffix}",
                4000,
            )
        else:
            self._statusbar.showMessage("No matching original/current path coordinates were available", 4000)

    def _toggle_path_change_overlay(self, checked: bool):
        self._path_change_overlay_active = bool(checked)
        if self._path_change_overlay_active:
            exit_btn = getattr(self, "_exit_delta_btn", None)
            if exit_btn is not None and exit_btn.isChecked():
                exit_btn.blockSignals(True)
                exit_btn.setChecked(False)
                exit_btn.blockSignals(False)
            self._exit_delta_overlay_active = False
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
            if not self._exit_delta_overlay_active:
                self._viewer.set_path_change_metric_map(None, self._display_excluded_path_ids())
                self._last_3d_overlay_signature = None
                self._refresh_3d_overlays()
                self._statusbar.showMessage("Path change distance overlay disabled", 2500)
            return
        self._apply_path_change_overlay()

    def _apply_exit_delta_overlay(self):
        visible_ids = set(getattr(self._viewer, "_bg_data_dict", {}).keys())
        visible_ids -= self._base_display_excluded_path_ids()
        metrics = self._exit_cluster_distance_delta_metric(visible_ids)
        delta_exclusion_changed = self._apply_path_delta_exclusion(metrics)
        self._viewer.set_path_change_metric_map(
            metrics,
            self._display_excluded_path_ids(),
            cmap="bwr",
            clim_percentile=95.0,
        )
        if delta_exclusion_changed:
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
        self._viewer.set_bg_visible(True)
        self._viewer.set_dataset_visible(False)
        if metrics:
            max_delta = max(metrics.values())
            hidden_count = len(getattr(self, "_path_delta_excluded_path_ids", set()))
            hidden_suffix = f", Exit Δ-filter hidden {hidden_count}" if hidden_count else ""
            self._statusbar.showMessage(
                f"Exit Δ overlay: {len(metrics)} paths, max cluster-distance difference {max_delta:.3f}{hidden_suffix}",
                4000,
            )
        else:
            self._statusbar.showMessage("No clustered current/original exit coordinates were available", 4000)

    def _toggle_exit_delta_overlay(self, checked: bool):
        self._exit_delta_overlay_active = bool(checked)
        if self._exit_delta_overlay_active:
            path_btn = getattr(self, "_path_change_btn", None)
            if path_btn is not None and path_btn.isChecked():
                path_btn.blockSignals(True)
                path_btn.setChecked(False)
                path_btn.blockSignals(False)
            self._path_change_overlay_active = False
            self._apply_exit_delta_overlay()
            return
        if not self._path_change_overlay_active:
            self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
            self._viewer.set_path_change_metric_map(None, self._display_excluded_path_ids())
            self._last_3d_overlay_signature = None
            self._refresh_3d_overlays()
            self._statusbar.showMessage("Exit Δ overlay disabled", 2500)

    def _visible_category_payload(self):
        visible = {}
        for name in self.state.visible_categories:
            if name in self.state.categories:
                path_ids = self._filter_display_path_set(self.state.categories[name])
                color = self.state.category_colors.get(name, "#00CC96")
                if path_ids:
                    visible[name] = (path_ids, color)
        return visible

    def _visible_dataset_payload(self):
        visible = {}
        excluded = self._display_excluded_path_ids()
        for dataset in self.db.list_datasets():
            key = dataset["key"]
            if key not in self.state.visible_datasets:
                continue
            if self.state.dataset_cluster_display_key == key and hasattr(self.db, "get_dataset_cluster_path_groups"):
                cluster_groups = self.db.get_dataset_cluster_path_groups(
                    key,
                    frame_min=self.state.frame_min,
                    frame_max=self.state.frame_max,
                )
                for cluster_id, cluster_path_ids in cluster_groups.items():
                    path_ids = set(cluster_path_ids) - excluded
                    if path_ids:
                        visible[f"dataset:{key}:cluster:{cluster_id}"] = (
                            path_ids,
                            self.state.get_auto_color(int(cluster_id)),
                        )
                continue
            path_ids = self._dataset_path_ids(key) - excluded
            if path_ids:
                visible[f"dataset:{key}"] = (
                    path_ids,
                    self.state.dataset_colors.get(key, self.state.get_auto_color(0)),
                )
        return visible

    def _visible_dataset_path_ids(self) -> set[int]:
        visible_ids = set()
        for path_ids, _color in self._visible_dataset_payload().values():
            visible_ids.update(path_ids)
        return visible_ids

    def _visible_dataset_path_color_map(self) -> dict[int, str]:
        color_map: dict[int, str] = {}
        for path_ids, color in self._visible_dataset_payload().values():
            for path_id in path_ids:
                color_map[int(path_id)] = str(color)
        return color_map

    def _chart_path_color_map(self, profiles) -> dict[int, str]:
        color_map: dict[int, str] = {}
        for profile in profiles or []:
            path_id = int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
            if path_id <= 0:
                continue
            dataset_key = getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(path_id)
            if not dataset_key:
                color_map[path_id] = "#4169E1"
                continue
            if self.state.dataset_cluster_display_key == dataset_key:
                cluster_id = int(profile.get("cluster_id", 0) or 0)
                color_map[path_id] = self.state.get_auto_color(cluster_id)
            else:
                color_map[path_id] = self.state.dataset_colors.get(
                    dataset_key,
                    self.state.get_auto_color(0),
                )
        return color_map

    def _visible_cluster_label_payload(self) -> list[dict]:
        dataset_key = str(self.state.dataset_cluster_label_key or "")
        if not dataset_key or not hasattr(self.db, "get_dataset_cluster_path_groups"):
            return []
        excluded = self._display_excluded_path_ids()
        cluster_groups = self.db.get_dataset_cluster_path_groups(
            dataset_key,
            frame_min=self.state.frame_min,
            frame_max=self.state.frame_max,
        )
        labels: list[dict] = []
        for cluster_id, cluster_path_ids in cluster_groups.items():
            exit_points = []
            for path_id in cluster_path_ids:
                if int(path_id) in excluded:
                    continue
                coords = self._viewer._bg_data_dict.get(int(path_id))
                if coords is None or len(coords) == 0:
                    continue
                exit_points.append(np.asarray(coords[-1], dtype=np.float64))
            if not exit_points:
                continue
            mean_point = np.mean(np.vstack(exit_points), axis=0)
            labels.append(
                {
                    "text": f"C{int(cluster_id)}",
                    "position": tuple(float(value) for value in mean_point[:3]),
                    "color": self.state.get_auto_color(int(cluster_id)),
                }
            )
        return labels

    def _dataset_path_ids(self, key: str) -> set[int]:
        cache_key = (key, self.state.frame_min, self.state.frame_max)
        if cache_key not in self._dataset_path_cache:
            self._dataset_path_cache[cache_key] = set(self.db.get_dataset_path_ids(
                key,
                frame_min=self.state.frame_min,
                frame_max=self.state.frame_max,
            ))
        return set(self._dataset_path_cache[cache_key])

    def _hidden_dataset_path_ids(self) -> set[int]:
        hidden = set()
        for dataset in self.db.list_datasets():
            key = dataset["key"]
            if key not in self.state.visible_datasets:
                hidden.update(self._dataset_path_ids(key))
        return hidden

    def _display_excluded_path_ids(self) -> set[int]:
        return self._base_display_excluded_path_ids() | set(getattr(self, "_path_delta_excluded_path_ids", set()))

    def _filter_hidden_dataset_path_set(self, path_ids) -> set[int]:
        return set(path_ids) - self._hidden_dataset_path_ids()

    def _filter_hidden_dataset_paths(self, path_ids):
        hidden = self._hidden_dataset_path_ids()
        return [int(pid) for pid in path_ids if int(pid) not in hidden]

    def _filter_display_path_set(self, path_ids) -> set[int]:
        return set(path_ids) - self._display_excluded_path_ids()

    def _filter_display_paths(self, path_ids):
        excluded = self._display_excluded_path_ids()
        return [int(pid) for pid in path_ids if int(pid) not in excluded]

    def _visible_effective_path_ids(self) -> set[int]:
        return self._filter_display_path_set(self.state.effective_path_ids)

    def _path_panel_selected_path_ids(self) -> set[int]:
        panel = getattr(self, "_path_panel", None)
        if panel is None or not hasattr(panel, "selected_path_ids"):
            return set()
        try:
            return {
                int(path_id)
                for path_id in panel.selected_path_ids()
                if int(path_id) > 0
            }
        except Exception:
            return set()

    def _residue_combination_context_path_ids(
        self,
        *,
        prefer_explicit_selection: bool = False,
    ) -> set[int]:
        selected = self._path_panel_selected_path_ids()
        effective = {
            int(path_id)
            for path_id in getattr(self.state, "effective_path_ids", set())
            if int(path_id) > 0
        }
        preview = {
            int(path_id)
            for path_id in (getattr(self, "_last_preview_path_ids", []) or [])
            if int(path_id) > 0
        }
        # Reload/manual context refresh must not be narrowed by a table-row click.
        if prefer_explicit_selection and selected:
            return selected
        if effective:
            return effective
        if prefer_explicit_selection and preview:
            return preview
        return set()

    def _get_display_path_coords(self, path_ids):
        if self._path_coordinate_mode == "original":
            return self.db.get_original_render_coords(path_ids)
        return self.db.get_render_coords(path_ids)

    def _payload_signature(self, payload: dict) -> tuple:
        items = []
        for key, value in (payload or {}).items():
            path_ids, color = value
            items.append((str(key), tuple(sorted(int(path_id) for path_id in (path_ids or ()))), str(color)))
        return tuple(sorted(items))

    def _cluster_label_signature(self, labels: list[dict]) -> tuple:
        result = []
        for entry in labels or []:
            position = entry.get("position", ())
            try:
                pos = tuple(round(float(value), 4) for value in position)
            except Exception:
                pos = ()
            result.append((str(entry.get("text", "") or ""), pos, str(entry.get("color", "") or "")))
        return tuple(result)

    def _entry_exit_signature_values(self) -> tuple:
        display_ids = self._get_entry_exit_display_ids()
        return (
            tuple(sorted(int(path_id) for path_id in self.state.effective_path_ids)),
            None if display_ids is None else tuple(sorted(int(path_id) for path_id in display_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in self._visible_dataset_path_color_map().items())),
        )

    def _render_effective_paths_if_needed(self, path_ids=None, color_map=None) -> None:
        visible_ids = set(path_ids) if path_ids is not None else self._visible_effective_path_ids()
        visible_color_map = dict(color_map) if color_map is not None else self._visible_dataset_path_color_map()
        signature = (
            tuple(sorted(int(path_id) for path_id in visible_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in visible_color_map.items())),
        )
        if signature == self._last_effective_render_signature:
            return
        self._last_effective_render_signature = signature
        self._viewer.render_effective_paths(visible_ids, color_map=visible_color_map)

    def _refresh_3d_overlays(self):
        excluded = self._display_excluded_path_ids()
        dataset_payload = self._visible_dataset_payload()
        cluster_payload = self._visible_cluster_label_payload()
        category_payload = self._visible_category_payload()
        effective_ids = self._visible_effective_path_ids()
        effective_color_map = self._visible_dataset_path_color_map()
        visible_preview_ids = (
            tuple(self._filter_display_paths(self._last_preview_path_ids))
            if self._last_preview_path_ids else ()
        )
        highlighted_preview_ids = tuple(sorted(self._filter_display_path_set(self._last_highlighted_preview_ids)))
        signature = (
            tuple(sorted(int(path_id) for path_id in excluded)),
            self._payload_signature(dataset_payload),
            self._cluster_label_signature(cluster_payload),
            self._payload_signature(category_payload),
            tuple(sorted(int(path_id) for path_id in effective_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in effective_color_map.items())),
            visible_preview_ids,
            highlighted_preview_ids,
            tuple(sorted((int(path_id), str(color)) for path_id, color in (self._last_preview_color_map or {}).items())),
            bool(self.state.focus_mode),
            bool(self.state.show_entry_exit_points),
            self._entry_exit_signature_values(),
        )
        if signature == self._last_3d_overlay_signature:
            return
        self._last_3d_overlay_signature = signature
        if hasattr(self._viewer, "begin_render_batch"):
            self._viewer.begin_render_batch()
        try:
            self._viewer.set_bg_excluded_paths(set(excluded))
            self._viewer.render_dataset_paths(dataset_payload)
            self._viewer.render_cluster_labels(cluster_payload)
            self._viewer.render_category_paths(category_payload)
            self._last_effective_render_signature = None
            self._render_effective_paths_if_needed(effective_ids, effective_color_map)

            if visible_preview_ids:
                coords = self._get_display_path_coords(list(visible_preview_ids))
                self._viewer.render_matched_paths(
                    coords,
                    color_map=self._last_preview_color_map or None,
                    highlighted=set(highlighted_preview_ids),
                )
            else:
                self._viewer.clear_matched()

            if self.state.focus_mode:
                self._viewer.set_bg_visible(False)
                self._viewer.set_dataset_visible(False)
            elif self._path_change_overlay_active or self._exit_delta_overlay_active:
                self._viewer.set_bg_visible(True)
                self._viewer.set_dataset_visible(False)
            else:
                self._viewer.set_bg_visible(True)
                self._viewer.set_dataset_visible(True)
            if self.state.show_entry_exit_points:
                self._last_entry_exit_signature = None
                self._refresh_entry_exit_points()
        finally:
            if hasattr(self._viewer, "end_render_batch"):
                self._viewer.end_render_batch()

    def _on_viewer_coordinate_mode_changed(self, mode: str):
        mode = "original" if mode == "original" else "current"
        if self._path_coordinate_load_busy:
            self._viewer.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)
            return
        if self._path_coordinate_mode == mode:
            return
        self._path_coordinate_mode = mode
        self._invalidate_render_signatures()
        mode_label = "original paths" if mode == "original" else "current paths"
        self._statusbar.showMessage(f"3D coordinate mode switched to {mode_label}", 2500)
        self._reload_3d()

    # ─── Entry/Exit point display + selection mode ────────
    def _toggle_focus_mode(self, checked: bool):
        self.state.focus_mode = checked
        self._viewer.set_focus_mode(checked)
        self._viewer.set_bg_visible(not checked)
        self._viewer.set_dataset_visible(not checked)
        if checked:
            self._statusbar.showMessage("Focus mode: showing only selected paths", 3000)
        else:
            self._statusbar.showMessage("Focus mode off: showing all paths", 3000)
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()

    def _toggle_entry_exit_points(self, checked: bool):
        self.state.set_show_entry_exit_points(checked)

    def _toggle_entry_exit_scope(self, checked: bool):
        self.state.set_entry_exit_current_only(checked)

    def _toggle_exit_cluster_select(self, checked: bool):
        self._viewer.set_exit_cluster_select_mode(bool(checked))
        if checked:
            if not self._entry_exit_btn.isChecked():
                self._entry_exit_btn.setChecked(True)
            self._statusbar.showMessage(
                "Exit Cluster ON - click an exit point to select all connected exit spheres at current zoom",
                5000,
            )
        else:
            self._statusbar.showMessage("Exit Cluster OFF", 2500)

    def _get_entry_exit_display_ids(self):
        if not self.state.entry_exit_current_only:
            return None
        if self.state.effective_path_ids:
            return set(self.state.effective_path_ids)
        visible = self.state.get_visible_category_path_ids()
        return set(visible)

    def _refresh_entry_exit_scope_btn(self):
        current_only = self.state.entry_exit_current_only
        self._entry_exit_scope_btn.blockSignals(True)
        self._entry_exit_scope_btn.setChecked(current_only)
        self._entry_exit_scope_btn.setText("EE: Current" if current_only else "EE: All")
        self._entry_exit_scope_btn.setToolTip(
            "Showing entry/exit points for current paths only"
            if current_only else
            "Showing entry/exit points for all paths"
        )
        self._entry_exit_scope_btn.blockSignals(False)

    def _refresh_entry_exit_points(self):
        display_ids = self._get_entry_exit_display_ids()
        signature = self._entry_exit_signature_values()
        if signature == self._last_entry_exit_signature:
            return
        self._last_entry_exit_signature = signature
        self._viewer.render_entry_exit_points(
            show=True,
            effective_ids=self.state.effective_path_ids,
            display_ids=display_ids,
            path_color_map=self._visible_dataset_path_color_map(),
        )

    def _on_entry_exit_display_changed(self, show: bool):
        if show:
            self._refresh_entry_exit_points()
            counts = self._viewer._rendered_entry_exit_counts
            scope = "current" if self.state.entry_exit_current_only else "all"
            self._statusbar.showMessage(
                f"Entry/exit points ({scope}): {counts['entries']} entries + "
                f"{counts['exits']} exits",
                3000,
            )
        else:
            self._viewer.clear_entry_exit_points()
            self._statusbar.showMessage("Entry/exit points hidden", 3000)

    def _on_entry_exit_scope_changed(self, current_only: bool):
        self._refresh_entry_exit_scope_btn()
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
            counts = self._viewer._rendered_entry_exit_counts
            scope = "current" if current_only else "all"
            self._statusbar.showMessage(
                f"Entry/exit scope: {scope} "
                f"({counts['entries']} entries + {counts['exits']} exits)",
                3000,
            )

    def _toggle_selection_mode(self, checked: bool):
        mode = "entrance_exit" if checked else "path"
        self.state.set_selection_mode(mode)
        if checked:
            # Auto-show entry/exit points when switching to point select mode
            if not self._entry_exit_btn.isChecked():
                self._entry_exit_btn.setChecked(True)
            self._statusbar.showMessage(
                "Point Select ON — lasso selects entry/exit points (coarse filter)", 5000
            )
        else:
            self._statusbar.showMessage(
                "Path Select — lasso selects paths (fine filter)", 3000
            )

    def _on_exit_cluster_selected(self, path_ids):
        selected = {int(path_id) for path_id in (path_ids or set()) if int(path_id) > 0}
        if not selected:
            self._statusbar.showMessage("No connected exit points found at this zoom", 3000)
            return
        self.state.apply_lasso(selected, "replace")
        self._path_panel.refresh()
        self._path_panel.set_highlighted_paths(sorted(selected))
        self._last_highlighted_preview_ids = set(selected)
        self._status_selected.setText(f"Effective: {len(self.state.effective_path_ids):,} paths")
        if self.state.show_entry_exit_points:
            self._refresh_entry_exit_points()
        self._statusbar.showMessage(
            f"Selected {len(selected)} paths from connected exit points",
            4000,
        )

    def _toggle_residue_positions(self, checked: bool):
        self.state.show_all_residues = checked
        if checked:
            positions = self._residue_positions_with_dataset_colors()
            if positions:
                self._viewer.set_residue_data(positions)
                self._viewer.render_residues(self.state.selected_residues)
                self._refresh_residue_labels()
                self._refresh_residue_combination_markers()
                self._statusbar.showMessage(
                    f"Showing {len(positions)} residues", 3000
                )
        else:
            self._viewer.clear_residues()
            self._statusbar.showMessage("Residue positions hidden", 3000)

    def _refresh_residue_3d(self):
        """Re-render residue spheres when selection changes (if toggle is on)."""
        if self.state.show_all_residues:
            self._viewer.render_residues(self.state.selected_residues)
            self._refresh_residue_labels()
        self._refresh_residue_combination_markers()

    def _toggle_residue_labels(self, checked: bool):
        if not checked:
            self._viewer.clear_residue_labels()
            self._statusbar.showMessage("Residue labels hidden", 3000)
            return
        self._refresh_residue_labels()
        self._statusbar.showMessage("Residue labels shown for selected residues", 3000)

    def _refresh_residue_labels(self):
        if not getattr(self, "_residue_labels_btn", None):
            return
        if not self._residue_labels_btn.isChecked() or not self.state.show_all_residues:
            self._viewer.clear_residue_labels()
            return
        self._viewer.render_residue_labels(self.state.selected_residues)

    def _on_residue_3d_clicked(self, residue_id: int):
        """Residue sphere clicked in 3D → toggle selection in panel."""
        self.state.toggle_residue(residue_id)
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _on_residue_compare_selection_requested(self, dataset_key: str, local_residue_id: int):
        if not dataset_key or int(local_residue_id) <= 0:
            return
        dataset = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        if dataset is None:
            return
        global_residue_id = dataset.residue_id_base + int(local_residue_id)
        self.state.toggle_residue(global_residue_id)
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _on_residue_compare_clear_requested(self):
        self.state.clear_residues()
        self._residue_panel.load_options()
        self._on_residue_changed()

    def _sync_residue_compare_selection_state(self):
        selected_keys: set[tuple[str, int]] = set()
        for residue_id in self.state.selected_residues:
            dataset, local_id = getattr(self.db, "_decode_residue_id", lambda _rid: (None, None))(int(residue_id))
            if dataset is None or local_id is None:
                continue
            selected_keys.add((str(dataset.key), int(local_id)))
        if getattr(self, "_residue_compare_panel", None):
            self._residue_compare_panel.set_selected_residues(selected_keys)

    def _prime_primary_residue_positions(self):
        positions = []
        primary_db = self._primary_analysis_db()
        if primary_db is not None:
            positions = primary_db.get_residue_positions()
        self._primary_residue_position_cache = {
            int(row["residue_id"]): np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            for row in positions
            if row.get("residue_id") is not None
        }

    def _residue_positions_with_dataset_colors(self) -> list[dict]:
        positions = self.db.get_residue_positions()
        for row in positions:
            dataset_key = str(row.get("dataset_key", "") or "")
            row["dataset_color"] = self.state.dataset_colors.get(dataset_key, "#4169E1")
        return positions

    def _dataset_residue_position_maps(self) -> dict[str, dict[int, tuple[float, float, float]]]:
        result: dict[str, dict[int, tuple[float, float, float]]] = {}
        for dataset_info in self.db.list_datasets() if hasattr(self.db, "list_datasets") else []:
            dataset_key = str(dataset_info.get("key", "") or "")
            binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
            dataset_db = binding.db if binding is not None else None
            if not dataset_key or dataset_db is None:
                continue
            result[dataset_key] = {
                int(row["residue_id"]): (float(row["x"]), float(row["y"]), float(row["z"]))
                for row in dataset_db.get_residue_positions()
                if row.get("residue_id") is not None
            }
        self._dataset_residue_position_cache = result
        return result

    def _dataset_residue_atom_position_maps(self) -> dict[str, dict[int, list[tuple[float, float, float]]]]:
        result: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
        protein_cache = getattr(self._viewer, "_protein_model_cache", None) or {}
        aligned_atoms = list(protein_cache.get("atoms_aligned", []) or [])
        if not aligned_atoms:
            return result

        residue_atoms: dict[int, list[tuple[float, float, float]]] = {}
        for atom in aligned_atoms:
            try:
                residue_id = int(atom.get("res_seq", 0) or 0)
                position = np.asarray(atom.get("position"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            if residue_id <= 0:
                continue
            residue_atoms.setdefault(residue_id, []).append(
                (float(position[0]), float(position[1]), float(position[2]))
            )

        if not residue_atoms:
            return result

        for dataset_info in self.db.list_datasets() if hasattr(self.db, "list_datasets") else []:
            dataset_key = str(dataset_info.get("key", "") or "")
            if dataset_key:
                result[dataset_key] = {
                    int(residue_id): list(atom_positions)
                    for residue_id, atom_positions in residue_atoms.items()
                }
        return result

    def _encode_residue_ids_for_all_datasets(self, residue_ids) -> list[int]:
        local_ids = [int(rid) for rid in residue_ids if int(rid) > 0]
        if not local_ids or not hasattr(self.db, "list_datasets") or not hasattr(self.db, "_datasets_by_key"):
            return local_ids

        encoded_ids: list[int] = []
        for dataset_info in self.db.list_datasets():
            dataset_key = str(dataset_info.get("key") or "")
            dataset = self.db._datasets_by_key.get(dataset_key)
            if dataset is None:
                continue
            encoded_ids.extend(dataset.residue_id_base + local_id for local_id in local_ids)
        return encoded_ids or local_ids

    def _dataset_combination_matched_path_ids(self, dataset_key: str, residue_ids) -> list[int]:
        residue_tuple = tuple(sorted(int(rid) for rid in residue_ids if int(rid) > 0))
        if len(residue_tuple) < 2:
            return []
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(str(dataset_key))
        dataset_db = dataset_binding.db if dataset_binding is not None else None
        if dataset_binding is None or dataset_db is None:
            return []

        local_path_ids = dataset_db.get_path_ids()
        profiles = dataset_db.get_path_profiles(local_path_ids)
        matched_local_ids: list[int] = []
        for profile in profiles:
            residue_columns = (
                profile.get("res_1", []),
                profile.get("res_2", []),
                profile.get("res_3", []),
                profile.get("res_4", []),
            )
            point_count = max((len(column) for column in residue_columns), default=0)
            has_combination = False
            for idx in range(point_count):
                point_residue_ids = tuple(sorted(
                    {
                        int(column[idx])
                        for column in residue_columns
                        if idx < len(column) and int(column[idx]) > 0
                    }
                ))
                if point_residue_ids == residue_tuple:
                    has_combination = True
                    break
            if has_combination:
                local_path_id = int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
                if local_path_id > 0:
                    matched_local_ids.append(local_path_id)
        return [
            dataset_binding.path_id_base + int(local_path_id)
            for local_path_id in matched_local_ids
        ]

    def _build_residue_combination_markers(self) -> list[dict]:
        if not self._primary_residue_position_cache:
            self._prime_primary_residue_positions()
        if not self._dataset_residue_position_cache:
            self._dataset_residue_position_maps()

        property_key = self._residue_combination_panel.selected_property_key()
        property_label = self._residue_combination_panel.selected_property_label()
        compare_mode = self._residue_combination_panel.is_compare_mode()
        active_keys = self._residue_combination_panel.active_dataset_keys()
        marker_position_maps = [
            self._dataset_residue_position_cache.get(key, {})
            for key in reversed(active_keys)
            if self._dataset_residue_position_cache.get(key)
        ]
        marker_position_maps.append(self._primary_residue_position_cache)
        markers = []
        for row in self._residue_combination_panel.visible_rows():
            residue_ids = tuple(int(rid) for rid in row.get("residue_ids", ()) if int(rid) > 0)
            centroid = None
            for position_map in marker_position_maps:
                centroid = combination_centroid(residue_ids, position_map)
                if centroid is not None:
                    break
            if centroid is None:
                continue
            trend = combination_property_trend(row, property_key, compare_mode=compare_mode)
            value = combination_property_value(row, property_key, compare_mode=compare_mode)
            markers.append(
                {
                    "marker_key": "combo_" + "_".join(str(rid) for rid in residue_ids),
                    "residue_ids": residue_ids,
                    "residue_names": tuple(row.get("residue_names", ())),
                    "combination_label": row.get("combination_label", ""),
                    "affected_path_count": int(row.get("affected_path_count", row.get("metrics", {}).get("affected_path_count", {}).get("right", 0)) or 0),
                    "property_key": property_key,
                    "property_label": property_label,
                    "property_value": value,
                    "property_trend": trend,
                    "property_trend_label": {
                        "increase": "Increase",
                        "decrease": "Decrease",
                        "neutral": "Neutral" if compare_mode else "",
                    }.get(trend, ""),
                    "centroid": centroid,
                }
            )
        return markers

    def _refresh_residue_combination_current_paths(
        self,
        *,
        prefer_explicit_selection: bool = False,
    ):
        if not getattr(self, "_residue_combination_panel", None):
            return {}
        combo_keys: set[tuple[int, ...]] = set()
        rows_by_dataset: dict[str, list[dict]] = {}
        total_paths_by_dataset: dict[str, int] = {}
        profiles_by_dataset: dict[str, list[dict]] = {}
        panel = self._residue_combination_panel
        active_dataset_keys = set(panel.active_dataset_keys())
        context_path_ids = self._residue_combination_context_path_ids(
            prefer_explicit_selection=prefer_explicit_selection,
        )
        if context_path_ids and hasattr(self.db, "list_datasets") and hasattr(self.db, "get_local_path_ids_for_dataset"):
            for dataset_info in self.db.list_datasets():
                dataset_key = str(dataset_info.get("key") or "")
                if not dataset_key:
                    continue
                if active_dataset_keys and dataset_key not in active_dataset_keys:
                    continue
                local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                if not local_path_ids:
                    continue
                dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    continue
                profiles = dataset_db.get_path_profiles(local_path_ids)
                profiles_by_dataset[dataset_key] = list(profiles)
                rows = combination_rows_from_profiles(profiles)
                rows_by_dataset[dataset_key] = rows
                combo_keys.update(tuple(row.get("residue_ids", ())) for row in rows)
                total_paths_by_dataset[dataset_key] = len(set(int(path_id) for path_id in local_path_ids))
        elif context_path_ids:
            profiles = self.db.get_path_profiles(list(context_path_ids))
            primary_key = self.db.primary_dataset_key() if hasattr(self.db, "primary_dataset_key") else ""
            if primary_key:
                profiles_by_dataset[str(primary_key)] = list(profiles)
                rows = combination_rows_from_profiles(profiles)
                rows_by_dataset[str(primary_key)] = rows
                combo_keys = {tuple(row.get("residue_ids", ())) for row in rows}
                total_paths_by_dataset[str(primary_key)] = len(set(int(path_id) for path_id in context_path_ids))
        self._residue_combination_panel.set_current_path_combination_keys(combo_keys)
        self._residue_combination_panel.set_current_path_combination_rows(
            rows_by_dataset,
            total_paths_by_dataset,
        )
        return profiles_by_dataset

    def _refresh_residue_combination_current_bottlenecks(
        self,
        *,
        prefer_explicit_selection: bool = False,
        profiles_by_dataset_key: dict[str, list[dict]] | None = None,
    ):
        if not getattr(self, "_residue_combination_panel", None):
            return
        combo_keys: set[tuple[int, ...]] = set()
        rows_by_dataset: dict[str, list[dict]] = {}
        total_paths_by_dataset: dict[str, int] = {}
        cached_profiles_by_dataset = {
            str(dataset_key): list(profiles or [])
            for dataset_key, profiles in dict(profiles_by_dataset_key or {}).items()
        }
        panel = self._residue_combination_panel
        active_dataset_keys = set(panel.active_dataset_keys())
        context_path_ids = self._residue_combination_context_path_ids(
            prefer_explicit_selection=prefer_explicit_selection,
        )
        if context_path_ids and hasattr(self.db, "list_datasets") and hasattr(self.db, "get_local_path_ids_for_dataset"):
            for dataset_info in self.db.list_datasets():
                dataset_key = str(dataset_info.get("key") or "")
                if not dataset_key:
                    continue
                if active_dataset_keys and dataset_key not in active_dataset_keys:
                    continue
                local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                if not local_path_ids:
                    continue
                dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                dataset_db = dataset_binding.db if dataset_binding is not None else None
                if dataset_db is None:
                    continue
                cached_profiles = cached_profiles_by_dataset.get(dataset_key)
                profiles = (
                    list(cached_profiles)
                    if cached_profiles is not None
                    else dataset_db.get_path_profiles(local_path_ids)
                )
                rows = combination_rows_from_profiles(
                    profiles,
                    bottleneck_only=True,
                )
                rows_by_dataset[dataset_key] = rows
                combo_keys.update(tuple(row.get("residue_ids", ())) for row in rows)
                total_paths_by_dataset[dataset_key] = len(set(int(path_id) for path_id in local_path_ids))
        elif context_path_ids:
            primary_key = self.db.primary_dataset_key() if hasattr(self.db, "primary_dataset_key") else ""
            profiles = (
                list(cached_profiles_by_dataset.get(primary_key, []))
                if primary_key and primary_key in cached_profiles_by_dataset
                else self.db.get_path_profiles(list(context_path_ids))
            )
            if primary_key:
                rows = combination_rows_from_profiles(
                    profiles,
                    bottleneck_only=True,
                )
                rows_by_dataset[str(primary_key)] = rows
                combo_keys = {tuple(row.get("residue_ids", ())) for row in rows}
                total_paths_by_dataset[str(primary_key)] = len(set(int(path_id) for path_id in context_path_ids))
        self._residue_combination_panel.set_current_bottleneck_combination_keys(combo_keys)
        self._residue_combination_panel.set_current_bottleneck_combination_rows(
            rows_by_dataset,
            total_paths_by_dataset,
        )

    def _combination_present_frames_from_profiles(
        self,
        profiles: list[dict],
        *,
        max_size: int = 4,
        bottleneck_only: bool = False,
    ) -> dict[tuple[int, ...], set[int]]:
        from itertools import combinations

        result: dict[tuple[int, ...], set[int]] = {}
        for profile in profiles or ():
            try:
                frame_id = int(profile.get("frameId", profile.get("frame_id", 0)) or 0)
            except (TypeError, ValueError):
                continue
            if frame_id <= 0:
                continue
            residue_columns = (
                profile.get("res_1", []),
                profile.get("res_2", []),
                profile.get("res_3", []),
                profile.get("res_4", []),
            )
            radii = profile.get("radius", [])
            point_count = max((len(column) for column in residue_columns), default=0)
            allowed_indices: set[int] | None = None
            if bottleneck_only and len(radii) > 0:
                allowed_indices = bottleneck_point_indices(radii, point_count=point_count)
            for idx in range(point_count):
                if allowed_indices is not None and idx not in allowed_indices:
                    continue
                residue_ids = sorted(
                    {
                        int(column[idx])
                        for column in residue_columns
                        if idx < len(column) and int(column[idx] or 0) > 0
                    }
                )
                for size in range(2, min(int(max_size), len(residue_ids)) + 1):
                    for combo in combinations(residue_ids, size):
                        result.setdefault(tuple(int(item) for item in combo), set()).add(frame_id)
        return result

    def _profiles_for_dataset_path_scope(self, dataset_key: str, frame_ids: set[int] | None = None) -> list[dict]:
        dataset_key = str(dataset_key or "").strip()
        if not dataset_key:
            return []
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        dataset_db = dataset_binding.db if dataset_binding is not None else None
        if dataset_db is None:
            return []
        normalized_frames = sorted({int(frame) for frame in (frame_ids or set()) if int(frame) > 0})
        global_ids = []
        try:
            if hasattr(self.db, "get_dataset_path_ids"):
                if normalized_frames:
                    global_ids = self.db.get_dataset_path_ids(dataset_key, normalized_frames[0], normalized_frames[-1])
                else:
                    global_ids = self.db.get_dataset_path_ids(dataset_key, self.state.frame_min, self.state.frame_max)
        except Exception:
            global_ids = []
        try:
            local_ids = self.db.get_local_path_ids_for_dataset(dataset_key, global_ids) if global_ids else []
        except Exception:
            local_ids = []
        if not local_ids:
            try:
                if normalized_frames:
                    local_ids = dataset_db.get_path_ids_for_frames(normalized_frames)
                else:
                    local_ids = dataset_db.get_path_ids()
            except Exception:
                local_ids = []
        if not local_ids:
            return []
        try:
            return dataset_db.get_path_profiles(local_ids)
        except Exception:
            return []

    def _dataset_residue_pair_distance_csv_path(self, dataset_key: str) -> str:
        dataset_key = str(dataset_key or "").strip()
        cache_key = ("distance", dataset_key)
        cached = self._dataset_residue_pair_csv_path_cache.get(cache_key)
        if cached is not None:
            return cached
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset_binding is None:
            self._dataset_residue_pair_csv_path_cache[cache_key] = ""
            return ""
        target_names = (
            "residue_pair_distances.npy",
            "residue_pair_distance.npy",
            "residue_pari_distance.npy",
            "residue_pari_distances.npy",
            "residue_pair_distances.csv",
            "residue_pair_distance.csv",
            "residue_pari_distance.csv",
            "residue_pari_distances.csv",
        )
        result = self._find_dataset_named_file(dataset_binding, target_names)
        if result:
            self._dataset_residue_pair_csv_path_cache[cache_key] = result
            return result
        self._dataset_residue_pair_csv_path_cache[cache_key] = ""
        return ""

    def _find_dataset_named_file(self, dataset_binding, target_names: tuple[str, ...]) -> str:
        search_roots: list[str] = []
        for candidate in (
            os.path.dirname(str(getattr(dataset_binding, "residue_combination_statistics_path", "") or "")),
            os.path.dirname(str(getattr(dataset_binding, "residue_statistics_path", "") or "")),
            os.path.dirname(str(getattr(dataset_binding, "path", "") or "")),
            getattr(dataset_binding, "folder", ""),
        ):
            normalized = os.path.abspath(str(candidate or "").strip()) if str(candidate or "").strip() else ""
            if normalized and os.path.isdir(normalized) and normalized not in search_roots:
                search_roots.append(normalized)
        normalized_targets = {str(name).lower(): str(name) for name in target_names}
        for root in search_roots:
            try:
                lower_map = {name.lower(): name for name in os.listdir(root)}
            except OSError:
                continue
            for lower_name in normalized_targets:
                actual = lower_map.get(lower_name)
                if actual:
                    return os.path.abspath(os.path.join(root, actual))
        return ""

    def _dataset_residue_pair_presence_stats_csv_path(self, dataset_key: str) -> str:
        dataset_key = str(dataset_key or "").strip()
        cache_key = ("presence_stats", dataset_key)
        cached = self._dataset_residue_pair_csv_path_cache.get(cache_key)
        if cached is not None:
            return cached
        dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
        if dataset_binding is None:
            self._dataset_residue_pair_csv_path_cache[cache_key] = ""
            return ""
        target_names = (
            "residue_pair_path_presence_stats.csv",
            "residue_pari_path_presence_stats.csv",
        )
        result = self._find_dataset_named_file(dataset_binding, target_names)
        if result:
            self._dataset_residue_pair_csv_path_cache[cache_key] = result
            return result
        self._dataset_residue_pair_csv_path_cache[cache_key] = ""
        return ""

    def _residue_combination_presence_stats_ready(self) -> bool:
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is None:
            return False
        try:
            file_a_key = panel.left_dataset_key() if hasattr(panel, "left_dataset_key") else ""
            file_b_key = panel.right_dataset_key() if hasattr(panel, "right_dataset_key") else ""
        except Exception:
            return False
        return bool(
            self._dataset_residue_pair_presence_stats_csv_path(file_a_key)
            and self._dataset_residue_pair_presence_stats_csv_path(file_b_key)
        )

    def _sync_residue_combination_observer_frame_distances(
        self,
        profiles_by_dataset_key: dict[str, list[dict]] | None = None,
    ) -> None:
        panel = getattr(self, "_residue_combination_panel", None)
        if panel is None or not hasattr(panel, "set_observer_frame_distance_inputs"):
            return
        top_slot = getattr(self, "_protein_viewer_panel_top", None)
        bottom_slot = getattr(self, "_protein_viewer_panel_bottom", None)
        top_panel = top_slot.get("panel") if isinstance(top_slot, dict) else None
        bottom_panel = bottom_slot.get("panel") if isinstance(bottom_slot, dict) else None
        file_a_key = panel.left_dataset_key() if hasattr(panel, "left_dataset_key") else ""
        file_b_key = panel.right_dataset_key() if hasattr(panel, "right_dataset_key") else ""

        def frame_paths_for(observer_panel) -> dict[int, str]:
            if observer_panel is None or not hasattr(observer_panel, "sequence_frame_paths"):
                return {}
            try:
                return observer_panel.sequence_frame_paths()
            except Exception:
                return {}

        left_frame_paths = frame_paths_for(top_panel)
        right_frame_paths = frame_paths_for(bottom_panel)
        left_presence_stats_csv_path = self._dataset_residue_pair_presence_stats_csv_path(file_a_key)
        right_presence_stats_csv_path = self._dataset_residue_pair_presence_stats_csv_path(file_b_key)
        left_distance_csv_path = self._dataset_residue_pair_distance_csv_path(file_a_key)
        right_distance_csv_path = self._dataset_residue_pair_distance_csv_path(file_b_key)
        scope_mode = (
            str(panel.loaded_scope_mode() or "dataset")
            if hasattr(panel, "loaded_scope_mode")
            else "dataset"
        )
        prefer_context_stats = bool(profiles_by_dataset_key) or scope_mode == "path" or bool(
            getattr(panel, "current_path_filter_active", lambda: False)()
        ) or bool(
            getattr(panel, "current_bottleneck_filter_active", lambda: False)()
        )
        context_profiles_by_dataset: dict[str, list[dict]] = {
            str(dataset_key): list(profiles or [])
            for dataset_key, profiles in dict(profiles_by_dataset_key or {}).items()
        }
        if prefer_context_stats and not context_profiles_by_dataset:
            context_path_ids = self._residue_combination_context_path_ids(
                prefer_explicit_selection=scope_mode == "path",
            )
            if context_path_ids and hasattr(self.db, "get_local_path_ids_for_dataset"):
                for dataset_key in {str(file_a_key or "").strip(), str(file_b_key or "").strip()}:
                    if not dataset_key:
                        continue
                    dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                    dataset_db = dataset_binding.db if dataset_binding is not None else None
                    if dataset_db is None:
                        continue
                    local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                    if not local_path_ids:
                        context_profiles_by_dataset[dataset_key] = []
                        continue
                    context_profiles_by_dataset[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
        frame_distances_ready = bool(
            (left_presence_stats_csv_path and right_presence_stats_csv_path)
            or (left_distance_csv_path and right_distance_csv_path)
            or (left_frame_paths and right_frame_paths)
        )
        presence_stats_ready = bool(left_presence_stats_csv_path and right_presence_stats_csv_path) and not prefer_context_stats

        def present_frames_for(dataset_key: str) -> dict[tuple[int, ...], set[int]]:
            if not frame_distances_ready:
                return {}
            if presence_stats_ready:
                return {}
            dataset_key = str(dataset_key or "").strip()
            if not dataset_key:
                return {}
            profiles = context_profiles_by_dataset.get(dataset_key) if prefer_context_stats else None
            if profiles is None:
                frames = set(left_frame_paths.keys()) if dataset_key == file_a_key else set(right_frame_paths.keys())
                bottleneck_only = bool(
                    getattr(panel, "current_bottleneck_filter_active", lambda: False)()
                )
                cache_key = (dataset_key, tuple(sorted(int(frame) for frame in frames)), bottleneck_only)
                cached = self._residue_combo_present_frame_cache.get(cache_key)
                if cached is not None:
                    return {
                        tuple(combo): set(frame_set)
                        for combo, frame_set in cached.items()
                    }
                profiles = self._profiles_for_dataset_path_scope(dataset_key, frames)
                present_frames = self._combination_present_frames_from_profiles(
                    profiles,
                    bottleneck_only=bottleneck_only,
                )
                self._residue_combo_present_frame_cache[cache_key] = {
                    tuple(combo): set(frame_set)
                    for combo, frame_set in present_frames.items()
                }
                if len(self._residue_combo_present_frame_cache) > 64:
                    self._residue_combo_present_frame_cache.clear()
                return present_frames
            bottleneck_only = bool(
                getattr(panel, "current_bottleneck_filter_active", lambda: False)()
            )
            profile_path_ids = tuple(sorted({
                int(profile.get("pathIndex", profile.get("localPathIndex", 0)) or 0)
                for profile in profiles
                if int(profile.get("pathIndex", profile.get("localPathIndex", 0)) or 0) > 0
            }))
            profile_frame_ids = tuple(sorted({
                int(profile.get("frameId", profile.get("frame_id", 0)) or 0)
                for profile in profiles
                if int(profile.get("frameId", profile.get("frame_id", 0)) or 0) > 0
            }))
            cache_key = ("profiles", dataset_key, profile_path_ids, profile_frame_ids, bottleneck_only)
            cached = self._residue_combo_present_frame_cache.get(cache_key)
            if cached is not None:
                return {
                    tuple(combo): set(frame_set)
                    for combo, frame_set in cached.items()
                }
            present_frames = self._combination_present_frames_from_profiles(
                profiles,
                bottleneck_only=bottleneck_only,
            )
            self._residue_combo_present_frame_cache[cache_key] = {
                tuple(combo): set(frame_set)
                for combo, frame_set in present_frames.items()
            }
            if len(self._residue_combo_present_frame_cache) > 64:
                self._residue_combo_present_frame_cache.clear()
            return present_frames

        panel.set_observer_frame_distance_inputs(
            left={
                "dataset_key": file_a_key,
                "frame_paths": left_frame_paths,
                "combo_present_frames": present_frames_for(file_a_key),
                "precomputed_distance_csv_path": left_distance_csv_path,
                "precomputed_presence_stats_csv_path": left_presence_stats_csv_path,
                "prefer_context_stats": prefer_context_stats,
            },
            right={
                "dataset_key": file_b_key,
                "frame_paths": right_frame_paths,
                "combo_present_frames": present_frames_for(file_b_key),
                "precomputed_distance_csv_path": right_distance_csv_path,
                "precomputed_presence_stats_csv_path": right_presence_stats_csv_path,
                "prefer_context_stats": prefer_context_stats,
            },
        )

    def _refresh_residue_combination_contained_match_status(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        panel.set_contained_match_status(
            locked=bool(self._contained_match_current_target_key),
            locked_count=len(self._contained_match_current_target_path_ids),
            candidate_count=len(self._contained_match_current_target_path_ids),
        )

    def _clear_residue_combination_contained_match_preview(self):
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids.clear()
        self._refresh_residue_combination_contained_match_status()

    def _clear_residue_combination_contained_match(self):
        if self._contained_match_current_target_path_ids:
            self.state.apply_lasso(set(self._contained_match_current_target_path_ids), "remove")
            self._path_panel.refresh()
        self._contained_match_last_compare_summary = ""
        if getattr(self, "_residue_combination_panel", None):
            self._residue_combination_panel.clear_contained_match_result()
        self._path_panel.set_highlighted_paths([])
        self._clear_residue_combination_contained_match_preview()
        self._viewer.set_selected_combination(None)
        profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
        self._refresh_residue_combination_current_bottlenecks(
            profiles_by_dataset_key=profiles_by_dataset_key,
        )
        self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
        self._refresh_residue_combination_markers()
        self._statusbar.showMessage("Cleared matched paths and residue-pair difference results", 3000)

    def _query_residue_combination_contained_match(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        if not panel.is_compare_mode():
            self._statusbar.showMessage("Load File A and File B before querying matched paths", 3000)
            return
        source_key = panel.contained_match_source_dataset_key()
        if not source_key:
            self._statusbar.showMessage("Select a source dataset first", 3000)
            return
        context_path_ids = self._residue_combination_context_path_ids(
            prefer_explicit_selection=True,
        )
        source_path_ids = [
            int(path_id)
            for path_id in context_path_ids
            if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == source_key
        ]
        single_path_mode = panel.contained_match_single_path_mode()
        if single_path_mode and len(source_path_ids) != 1:
            QMessageBox.warning(
                self,
                "Single-path search",
                "Select exactly one path in the source dataset before running a single-path search.",
            )
            return
        self._begin_loading("Querying matched paths...")
        finished = False
        try:
            if not source_key:
                self._statusbar.showMessage("Select a source dataset first", 3000)
                return

            active_dataset_keys = panel.active_dataset_keys()
            target_candidates = [
                str(dataset_key)
                for dataset_key in active_dataset_keys
                if str(dataset_key) and str(dataset_key) != source_key
            ]
            if not target_candidates:
                self._statusbar.showMessage("Select a target dataset in File B before querying matched paths", 3000)
                return
            target_key = str(target_candidates[0])

            if not source_path_ids:
                self._statusbar.showMessage("No paths are currently selected in the source dataset, so matched paths cannot be queried", 3000)
                return

            source_local_path_ids = (
                self.db.get_local_path_ids_for_dataset(source_key, source_path_ids)
                if hasattr(self.db, "get_local_path_ids_for_dataset") else []
            )
            source_binding = getattr(self.db, "_datasets_by_key", {}).get(source_key)
            target_binding = getattr(self.db, "_datasets_by_key", {}).get(target_key)
            source_db = source_binding.db if source_binding is not None else None
            target_db = target_binding.db if target_binding is not None else None
            if source_db is None or target_db is None:
                self._statusbar.showMessage("Dataset binding is unavailable, so matched paths cannot be queried", 3000)
                return

            source_profiles = source_db.get_path_profiles(source_local_path_ids)
            source_residue_ids = residue_set_from_profiles(source_profiles)
            if not source_residue_ids:
                self._statusbar.showMessage("No valid residues were extracted from the source paths, so matched paths cannot be queried", 3000)
                return

            target_local_path_ids = (
                self.db.get_local_path_ids_for_dataset(
                    target_key,
                    self.db.get_dataset_path_ids(target_key, self.state.frame_min, self.state.frame_max),
                )
                if hasattr(self.db, "get_dataset_path_ids") and hasattr(self.db, "get_local_path_ids_for_dataset")
                else []
            )
            if single_path_mode:
                params = panel.contained_match_single_path_parameters()
                source_local_path_id = int(source_local_path_ids[0])
                source_coords = source_db.get_render_coords([source_local_path_id]).get(source_local_path_id)
                source_lengths = source_db.get_path_length_map([source_local_path_id])
                source_length = source_lengths.get(source_local_path_id)
                if source_coords is None or len(source_coords) == 0:
                    self._statusbar.showMessage("The selected source path has no valid coordinates", 3000)
                    return
                if source_length is None:
                    self._statusbar.showMessage("The selected source path has no valid length", 3000)
                    return

                source_exit = np.asarray(source_coords[-1], dtype=float)
                target_length_map = target_db.get_path_length_map(target_local_path_ids)
                target_local_path_ids = [
                    int(path_id)
                    for path_id, path_length in target_length_map.items()
                    if abs(float(path_length) - float(source_length)) <= params["length_radius"]
                ]
                target_local_path_ids.sort()
                target_coords = target_db.get_render_coords(target_local_path_ids)
                target_local_path_ids = [
                    path_id
                    for path_id in target_local_path_ids
                    if path_id in target_coords
                    and len(target_coords[path_id]) > 0
                    and float(np.linalg.norm(np.asarray(target_coords[path_id][-1], dtype=float) - source_exit))
                    <= params["exit_radius"]
                ]
                target_profiles = target_db.get_path_profiles(target_local_path_ids)
                matched_target_profiles = []
                for profile in target_profiles:
                    target_residue_ids = profile_residue_set(profile)
                    if not target_residue_ids:
                        continue
                    jaccard = residue_set_jaccard_similarity(source_residue_ids, target_residue_ids)
                    if jaccard > params["jaccard_threshold"]:
                        matched_profile = dict(profile)
                        matched_profile["jaccard_similarity"] = jaccard
                        matched_target_profiles.append(matched_profile)
            else:
                keep_ratio = panel.contained_match_keep_ratio()
                source_exit_region = trimmed_exit_region_from_path_coords(
                    source_db.get_render_coords(source_local_path_ids),
                    keep_ratio,
                )
                if not source_exit_region:
                    self._statusbar.showMessage("No valid exit positions were extracted from the source paths, so matched paths cannot be queried", 3000)
                    return
                source_length_map = source_db.get_path_length_map(source_local_path_ids)
                source_length_region = trimmed_length_range(source_length_map, keep_ratio)
                if not source_length_region:
                    self._statusbar.showMessage("No valid path lengths were extracted from the source paths, so matched paths cannot be queried", 3000)
                    return
                source_length_min = float(source_length_region["min"])
                source_length_max = float(source_length_region["max"])
                target_local_path_ids = filter_path_ids_by_exit_region(
                    target_db.get_render_coords(target_local_path_ids),
                    source_exit_region,
                )
                target_length_map = target_db.get_path_length_map(target_local_path_ids)
                target_local_path_ids = [
                    int(path_id)
                    for path_id, path_length in target_length_map.items()
                    if float(path_length) >= source_length_min and float(path_length) <= source_length_max
                ]
                target_local_path_ids.sort()
                target_profiles = target_db.get_path_profiles(target_local_path_ids)
                matched_target_profiles = filter_profiles_by_residue_subset(target_profiles, source_residue_ids)
            matched_target_local_ids = [
                int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1)
                for profile in matched_target_profiles
                if int(profile.get("pathIndex", profile.get("localPathIndex", -1)) or -1) > 0
            ]
            target_path_id_base = int(getattr(target_binding, "path_id_base", 0) or 0)
            matched_target_global_ids = [
                target_path_id_base + int(local_id)
                for local_id in matched_target_local_ids
            ]
            source_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == source_key), {})
            target_dataset = next((dataset for dataset in datasets if str(dataset.get("key") or "") == target_key), {})
            previous_target_path_ids = set(self._contained_match_current_target_path_ids)
            self._contained_match_current_target_key = target_key
            self._contained_match_current_target_path_ids = set(matched_target_global_ids)
            if single_path_mode:
                self._contained_match_last_compare_summary = (
                    f"Single-path query completed | Source [{source_dataset.get('prefix', source_key)}] path {source_local_path_id} | "
                    f"Jaccard > {params['jaccard_threshold']:.2f} | Exit radius {params['exit_radius']:.2f} Å | "
                    f"Length radius {params['length_radius']:.2f} Å | "
                    f"Target [{target_dataset.get('prefix', target_key)}] passed exit/length: {len(target_profiles)} paths | "
                    f"Residue-matched: {len(matched_target_profiles)} paths"
                )
            else:
                self._contained_match_last_compare_summary = (
                    f"Match query completed | Source [{source_dataset.get('prefix', source_key)}]: {len(source_profiles)} paths | "
                    f"Keep ratio {keep_ratio:.2f} | "
                    f"Exit radius {float(source_exit_region.get('radius', 0.0)):.3f} | "
                    f"Length range [{source_length_min:.3f}, {source_length_max:.3f}] | "
                    f"Target [{target_dataset.get('prefix', target_key)}] prefiltered: {len(target_profiles)} paths | "
                    f"Residue-matched: {len(matched_target_profiles)} paths"
                )
            self._suppress_next_path_preview = True
            next_selection = set(source_path_ids) | set(matched_target_global_ids)
            if not next_selection:
                next_selection = set(context_path_ids) - previous_target_path_ids
            self.state.apply_lasso(next_selection, "replace")
            self._path_panel.refresh()
            self._path_panel.set_highlighted_paths(matched_target_global_ids)
            self._refresh_residue_combination_contained_match_status()
            if matched_target_global_ids:
                self._end_loading(
                    f"Found {len(matched_target_global_ids)} target matched paths; source-path selection was kept and this round of matched paths was added to the current selection",
                    4000,
                )
            else:
                self._end_loading(
                    "No target matched paths were found; residue-pair differences can still be computed and the target side will be treated as 0",
                    4000,
                )
            finished = True
        finally:
            if not finished:
                self._end_loading()

    def _compute_residue_combination_contained_match(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        panel = self._residue_combination_panel
        if not panel.is_compare_mode():
            self._statusbar.showMessage("Load File A and File B before computing differences", 3000)
            return
        self._begin_loading("Computing residue-pair differences...")
        finished = False
        try:
            file_a_key = panel.left_dataset_key()
            file_b_key = panel.right_dataset_key()
            if not file_a_key or not file_b_key:
                self._statusbar.showMessage("File A/File B dataset mapping is unavailable, so differences cannot be computed", 3000)
                return

            using_matched_paths = bool(self._contained_match_current_target_key)
            profiles_by_dataset_key: dict[str, list[dict]] = {}
            summary: str

            if using_matched_paths:
                source_key = panel.contained_match_source_dataset_key()
                if not source_key:
                    self._statusbar.showMessage("Select a source dataset first", 3000)
                    return

                context_path_ids = self._residue_combination_context_path_ids()
                target_path_ids = [
                    int(path_id)
                    for path_id in context_path_ids
                    if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == self._contained_match_current_target_key
                    and int(path_id) in self._contained_match_current_target_path_ids
                ]
                source_path_ids = [
                    int(path_id)
                    for path_id in context_path_ids
                    if getattr(self.db, "get_dataset_key_for_path_id", lambda _path_id: None)(int(path_id)) == source_key
                ]
                source_binding = getattr(self.db, "_datasets_by_key", {}).get(source_key)
                target_binding = getattr(self.db, "_datasets_by_key", {}).get(self._contained_match_current_target_key)
                source_db = source_binding.db if source_binding is not None else None
                target_db = target_binding.db if target_binding is not None else None
                if source_db is None or target_db is None:
                    self._statusbar.showMessage("Dataset binding is unavailable, so differences cannot be computed", 3000)
                    return
                if not source_path_ids:
                    self._statusbar.showMessage("No source-dataset paths are currently selected, so differences cannot be computed", 3000)
                    return

                source_local_path_ids = self.db.get_local_path_ids_for_dataset(
                    source_key,
                    source_path_ids,
                )
                target_local_path_ids = self.db.get_local_path_ids_for_dataset(
                    self._contained_match_current_target_key,
                    target_path_ids,
                )
                profiles_by_dataset_key = {
                    str(source_key): source_db.get_path_profiles(source_local_path_ids),
                    str(self._contained_match_current_target_key): target_db.get_path_profiles(target_local_path_ids),
                }
                if file_a_key not in profiles_by_dataset_key or file_b_key not in profiles_by_dataset_key:
                    self._statusbar.showMessage("Matched-path source/target datasets do not match the current File A/File B selection", 3000)
                    return
                summary = (
                    f"{self._contained_match_last_compare_summary} | "
                    f"Target paths kept after manual filtering: {len(profiles_by_dataset_key.get(self._contained_match_current_target_key, []))} | "
                    f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
                )
            else:
                context_path_ids = list(self._residue_combination_context_path_ids())
                if not context_path_ids:
                    self._statusbar.showMessage("No paths are currently selected, so differences cannot be computed", 3000)
                    return
                for dataset_key in (file_a_key, file_b_key):
                    dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
                    dataset_db = dataset_binding.db if dataset_binding is not None else None
                    if dataset_db is None:
                        self._statusbar.showMessage("Dataset binding is unavailable, so differences cannot be computed", 3000)
                        return
                    local_path_ids = self.db.get_local_path_ids_for_dataset(dataset_key, context_path_ids)
                    profiles_by_dataset_key[dataset_key] = dataset_db.get_path_profiles(local_path_ids)
                if not profiles_by_dataset_key.get(file_a_key) and not profiles_by_dataset_key.get(file_b_key):
                    self._statusbar.showMessage("The current selection does not contain any File A or File B paths to compare", 3000)
                    return
                summary = (
                    f"Current selection compare | "
                    f"File A: {len(profiles_by_dataset_key.get(file_a_key, []))} paths | "
                    f"File B: {len(profiles_by_dataset_key.get(file_b_key, []))} paths | "
                    f"Displayed deltas use File B - File A ({file_b_key} - {file_a_key})"
                )

            file_a_profiles = profiles_by_dataset_key.get(file_a_key, [])
            file_b_profiles = profiles_by_dataset_key.get(file_b_key, [])
            compare_rows = compare_combination_row_sets(
                combination_rows_from_profiles(
                    file_a_profiles,
                    bottleneck_only=False,
                ),
                combination_rows_from_profiles(
                    file_b_profiles,
                    bottleneck_only=False,
                ),
                include_right_only=True,
                left_total_paths=len(file_a_profiles),
                right_total_paths=len(file_b_profiles),
            )
            bottleneck_compare_rows = compare_combination_row_sets(
                combination_rows_from_profiles(
                    file_a_profiles,
                    bottleneck_only=True,
                ),
                combination_rows_from_profiles(
                    file_b_profiles,
                    bottleneck_only=True,
                ),
                include_right_only=True,
                left_total_paths=len(file_a_profiles),
                right_total_paths=len(file_b_profiles),
            )
            panel.set_contained_match_result(
                compare_rows,
                summary,
                active=True,
                bottleneck_rows=bottleneck_compare_rows,
            )
            self._sync_residue_combination_observer_frame_distances(profiles_by_dataset_key)
            self._refresh_residue_combination_markers()
            self._end_loading(
                f"Computed residue-pair differences as File B - File A using {len(file_a_profiles)} File A paths and {len(file_b_profiles)} File B paths",
                4000,
            )
            finished = True
        finally:
            if not finished:
                self._end_loading()

    def _refresh_residue_combination_markers(self):
        if not getattr(self, "_residue_combination_panel", None):
            return
        if not self._residue_combination_panel.combination_markers_enabled():
            self._viewer.clear_combination_markers()
            return
        markers = self._build_residue_combination_markers()
        self._viewer.render_combination_markers(markers)

    def _on_combination_marker_clicked(self, payload):
        residue_ids = tuple(int(rid) for rid in (payload or {}).get("residue_ids", ()) if int(rid) > 0)
        if len(residue_ids) < 2:
            return
        matched_ids = self.db.get_matched_path_ids(set(self._encode_residue_ids_for_all_datasets(residue_ids)))
        self.state.apply_lasso(set(matched_ids), "replace")
        self._path_panel.set_highlighted_paths(matched_ids)
        self._last_highlighted_preview_ids = set(matched_ids)
        self._statusbar.showMessage(
            f"{payload.get('combination_label', 'Residue Combination')} linked to {len(matched_ids)} paths",
            4000,
        )

    def _on_combination_selection_requested(self, payload):
        payload = dict(payload or {})
        self._observer_pending_payload = payload
        residue_ids = tuple(int(rid) for rid in payload.get("residue_ids", ()) if int(rid) > 0)
        if len(residue_ids) < 2:
            return
        if (
            getattr(self, "_protein_observer_container", None) is not None
            and not self._protein_observer_container.isVisible()
        ):
            self._toggle_protein_observer_panel(True)
        self._observer_residue_input.setText(",".join(str(rid) for rid in residue_ids))
        self._update_combination_observer_views(payload)
        self._protein_observer_debug_label.setText(
            f"Observer residues: {', '.join(str(rid) for rid in residue_ids)} | "
            f"{payload.get('combination_label', 'Residue Combination')}"
        )
        self._statusbar.showMessage(
            f"Residue Observer updated for residues {', '.join(str(rid) for rid in residue_ids)}",
            4000,
        )

    def _on_left_residue_set_requested(self, payload):
        payload = dict(payload or {})
        left_residue_ids = tuple(int(rid) for rid in payload.get("left_residue_ids", ()) if int(rid) > 0)
        right_residue_ids = tuple(int(rid) for rid in payload.get("right_residue_ids", ()) if int(rid) > 0)
        if len(left_residue_ids) < 2 and len(right_residue_ids) < 2:
            self._statusbar.showMessage("No visible File A / File B residue sets to show", 3000)
            return

        normalized_payload = dict(payload)
        normalized_payload["present_left"] = bool(
            str(payload.get("left_dataset_key", "") or "").strip() and len(left_residue_ids) >= 2
        )
        normalized_payload["present_right"] = bool(
            str(payload.get("right_dataset_key", "") or "").strip() and len(right_residue_ids) >= 2
        )
        normalized_payload["right_dataset_key"] = str(payload.get("right_dataset_key", "") or "")
        self._observer_pending_payload = normalized_payload

        if (
            getattr(self, "_protein_observer_container", None) is not None
            and not self._protein_observer_container.isVisible()
        ):
            self._toggle_protein_observer_panel(True)

        merged_residue_ids = tuple(
            sorted(set(int(rid) for rid in (*left_residue_ids, *right_residue_ids) if int(rid) > 0))
        )
        self._observer_residue_input.setText(",".join(str(rid) for rid in merged_residue_ids))
        self._update_combination_observer_views(normalized_payload)
        self._protein_observer_debug_label.setText(
            f"Observer residues: "
            f"A={len(left_residue_ids)} | B={len(right_residue_ids)} | "
            f"{normalized_payload.get('combination_label', 'Visible residues')}"
        )
        self._statusbar.showMessage(
            f"Residue Observer updated with A={len(left_residue_ids)} and B={len(right_residue_ids)} residues",
            4000,
        )

    # ─── Protein overlay ─────────────────────────────────
    def _set_protein_controls_enabled(self, enabled: bool):
        self._protein_toggle_btn.setEnabled(enabled)
        self._protein_style_combo.setEnabled(enabled)
        self._protein_opacity_slider.setEnabled(enabled)
        self._clear_protein_btn.setEnabled(enabled)

    def _build_residue_anchor_map(self) -> dict:
        anchors = {}
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        primary_prefix = datasets[0]["prefix"] if datasets else None
        for row in self.db.get_residue_positions():
            try:
                if primary_prefix is not None and row.get("dataset_prefix") != primary_prefix:
                    continue
                rid = int(row.get("local_residue_id", row["residue_id"]))
                anchors[rid] = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)
            except Exception:
                continue
        return anchors

    def _cleanup_pse_extracted_pdb(self) -> None:
        path = self._pse_extracted_pdb_path
        self._pse_extracted_pdb_path = None
        if not path:
            return
        try:
            os.unlink(path)
        except OSError:
            pass

    def _extract_pse_protein_to_pdb_file(self, molecule) -> str:
        self._cleanup_pse_extracted_pdb()
        pdb_text = pse_protein_to_pdb_string(molecule)
        fd, path = tempfile.mkstemp(suffix=".pdb", prefix="new_path_pse_")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(pdb_text)
        self._pse_extracted_pdb_path = path
        return path

    def _can_reuse_loaded_pdb_for_pse(self) -> bool:
        if not self._protein_path or self._viewer._protein_model_cache is None:
            return False
        if self._protein_path == self._pse_extracted_pdb_path:
            return False
        if not self._protein_path.lower().endswith(".pdb"):
            return False
        return os.path.exists(self._protein_path)

    def _sync_protein_controls_after_load(self, source_key: str, keep_toggle_state: bool) -> None:
        self._protein_path = source_key
        self._set_protein_controls_enabled(True)
        visible = self._protein_toggle_btn.isChecked() if keep_toggle_state else True
        self._protein_toggle_btn.blockSignals(True)
        self._protein_toggle_btn.setChecked(visible)
        self._protein_toggle_btn.blockSignals(False)
        self._viewer.set_protein_opacity(self._protein_opacity_slider.value() / 100.0)
        self._viewer.set_protein_visible(visible)

    def _reload_current_protein(self, keep_toggle_state: bool = True) -> None:
        if self._pse_path:
            self._load_pse_file(self._pse_path, keep_toggle_state=keep_toggle_state)
        elif self._protein_path:
            self._load_protein_file(self._protein_path, keep_toggle_state=keep_toggle_state)

    def _load_protein_file(self, pdb_path: str, keep_toggle_state: bool = False) -> bool:
        style = self._protein_style_combo.currentText().strip() or "cartoon"
        align_mode = self._protein_align_combo.currentData() or self._protein_align_combo.currentText()
        anchors = self._build_residue_anchor_map()
        result = self._viewer.load_protein_from_file(
            pdb_path=pdb_path,
            style=style,
            align_mode=str(align_mode),
            residue_anchor_positions=anchors,
        )
        if not result.get("ok"):
            msg = str(result.get("error", "unknown_error"))
            self._statusbar.showMessage(f"Failed to load protein: {msg}", 5000)
            QMessageBox.warning(self, "Load Protein Failed", f"Failed to load protein model:\n{msg}")
            return False

        if pdb_path != self._pse_extracted_pdb_path:
            self._cleanup_pse_extracted_pdb()
            self._pse_path = None
        self._sync_protein_controls_after_load(pdb_path, keep_toggle_state)
        self._residue_combination_panel.set_residue_atom_position_maps(
            self._dataset_residue_atom_position_maps()
        )
        rmsd = result.get("rmsd")
        rmsd_text = f"{float(rmsd):.3f}" if isinstance(rmsd, (int, float)) else "n/a"
        self._statusbar.showMessage(
            (
                f"Protein loaded: atoms={result.get('atom_count', 0)}, "
                f"match={result.get('matched_count', 0)}, "
                f"align={result.get('alignment_mode', 'n/a')}, rmsd={rmsd_text}"
            ),
            6000,
        )
        return True

    def _on_load_protein(self):
        pdb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Protein Structure",
            "",
            "Supported Files (*.pdb *.pse);;PDB Files (*.pdb);;PyMOL Session (*.pse);;All Files (*)",
        )
        if not pdb_path:
            return
        if pdb_path.lower().endswith(".pse"):
            self._load_pse_file(pdb_path)
        else:
            self._load_protein_file(pdb_path, keep_toggle_state=False)

    def _on_load_observer_protein(self):
        pdb_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Observer Protein PDB",
            "",
            "PDB Files (*.pdb);;All Files (*)",
        )
        if not pdb_path:
            return
        self._protein_observer_path = pdb_path
        for slot in self._protein_observer_slots():
            panel = slot.get("panel")
            if panel is not None:
                panel.load_pdb_file(pdb_path, first_model_only=True)
                panel.set_highlight_residues(())
            self._set_protein_observer_slot_title(
                slot,
                f"{slot.get('fallback_title', 'Observer')} | Manual PDB",
            )
        self._statusbar.showMessage(
            f"Residue observer loaded first frame: {os.path.basename(pdb_path)}",
            4000,
        )

    def _toggle_protein_observer_panel(self, checked: bool | None = None):
        container = getattr(self, "_protein_observer_container", None)
        btn = getattr(self, "_load_observer_protein_btn", None)
        if container is None:
            return
        if checked is None:
            checked = not bool(container.isVisible())
        container.setVisible(bool(checked))
        if bool(checked):
            self._refresh_observer_sequence_source_combo("top")
            self._refresh_observer_sequence_source_combo("bottom")
            splitter = getattr(self, "_main_visual_splitter", None)
            if splitter is not None:
                try:
                    total_width = max(680, int(splitter.width()))
                    observer_width = min(460, max(420, total_width // 4))
                    splitter.setSizes([max(320, total_width - observer_width), observer_width])
                except Exception:
                    pass
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.setText("Residue Observer")
            btn.blockSignals(False)
        self._statusbar.showMessage(
            "Residue Observer shown" if bool(checked) else "Residue Observer hidden",
            2500,
        )

    def _toggle_protein_visible(self, checked: bool):
        self._viewer.set_protein_visible(checked)
        if checked:
            self._statusbar.showMessage("Protein overlay enabled", 3000)
        else:
            self._statusbar.showMessage("Protein overlay hidden", 3000)

    def _on_protein_style_changed(self, style: str):
        self._viewer.set_protein_style(style)
        if self._protein_path:
            self._statusbar.showMessage(f"Protein style: {style}", 2000)

    def _on_protein_opacity_changed(self, value: int):
        self._viewer.set_protein_opacity(value / 100.0)

    def _on_protein_align_mode_changed(self, index: int):
        _ = index
        if not self._protein_path and not self._pse_path:
            return
        self._reload_current_protein(keep_toggle_state=True)

    def _on_clear_protein(self):
        self._viewer.clear_protein()
        self._viewer.clear_binding_sites()
        self._cleanup_pse_extracted_pdb()
        self._protein_path = None
        self._pse_path = None
        self._protein_toggle_btn.blockSignals(True)
        self._protein_toggle_btn.setChecked(False)
        self._protein_toggle_btn.blockSignals(False)
        self._set_protein_controls_enabled(False)
        self._binding_sites_btn.blockSignals(True)
        self._binding_sites_btn.setChecked(True)
        self._binding_sites_btn.setEnabled(False)
        self._binding_sites_btn.blockSignals(False)
        self._residue_combination_panel.set_residue_atom_position_maps({})
        self._statusbar.showMessage("Protein cleared", 3000)

    def _load_pse_file(self, pse_path: str, keep_toggle_state: bool = False):
        """Load a PyMOL .pse session: protein as cartoon + binding sites as surfaces."""
        try:
            molecules = parse_pse_file(pse_path)
        except Exception as exc:
            QMessageBox.warning(self, "Load PSE Failed", f"Failed to parse PSE file:\n{exc}")
            return

        if not molecules:
            QMessageBox.warning(self, "Load PSE Failed", "No molecular objects found in PSE file.")
            return

        # Separate protein(s) and binding site objects
        proteins = [m for m in molecules if m.is_protein]
        binding_sites = [m for m in molecules if not m.is_protein]

        # Prefer an already loaded real PDB, because its ribbon appearance is usually better.
        protein_loaded = False
        reused_existing_pdb = False
        protein_error: Optional[str] = None
        if self._can_reuse_loaded_pdb_for_pse():
            protein_loaded = True
            reused_existing_pdb = True
            self._sync_protein_controls_after_load(self._protein_path, keep_toggle_state)
        elif proteins:
            try:
                extracted_pdb_path = self._extract_pse_protein_to_pdb_file(proteins[0])
            except Exception as exc:
                extracted_pdb_path = ""
                protein_error = str(exc)

            if extracted_pdb_path:
                protein_loaded = self._load_protein_file(
                    extracted_pdb_path,
                    keep_toggle_state=keep_toggle_state,
                )
                if not protein_loaded:
                    protein_error = "failed_to_load_extracted_pdb"

        if not protein_loaded and not binding_sites:
            message = "No protein or binding sites found."
            if protein_error:
                message = f"Failed to load protein model:\n{protein_error}"
            QMessageBox.warning(self, "Load PSE", message)
            return

        if protein_loaded:
            self._pse_path = pse_path

        # Load binding sites
        if binding_sites:
            # Get alignment transform from protein loading
            align_R = None
            align_t = None
            align_scale = 1.0
            diag = self._viewer._protein_last_alignment_diag
            if diag and self._viewer._protein_model_cache:
                cache = self._viewer._protein_model_cache
                backbone = cache.get("backbone_raw", {})
                from TopoTunnel_UI.vis.protein_model import estimate_alignment
                anchors = self._build_residue_anchor_map()
                align_info = estimate_alignment(
                    backbone.get("ca_by_resseq", {}),
                    anchors,
                    mode=diag.get("mode", "auto_kabsch"),
                )
                align_R = np.asarray(align_info["R"], dtype=np.float64)
                align_t = np.asarray(align_info["t"], dtype=np.float64)
                align_scale = float(align_info.get("scale", 1.0))

            mol_data = [
                {"name": m.name, "coords": m.coords, "color": m.color}
                for m in binding_sites
            ]
            result = self._viewer.load_binding_sites(
                mol_data,
                align_R=align_R,
                align_t=align_t,
                align_scale=align_scale,
            )

            site_count = result.get("count", 0)
            self._binding_sites_btn.setEnabled(site_count > 0)
            self._binding_sites_btn.blockSignals(True)
            self._binding_sites_btn.setChecked(True)
            self._binding_sites_btn.blockSignals(False)

            self._statusbar.showMessage(
                (
                    f"PSE loaded: {len(proteins)} protein(s), "
                    f"{site_count} binding site surface(s)"
                    + ("; reused current PDB protein" if reused_existing_pdb else "")
                ),
                6000,
            )
        elif protein_loaded:
            self._statusbar.showMessage(
                "PSE loaded: protein only (no binding sites found)"
                + ("; reused current PDB protein" if reused_existing_pdb else ""),
                6000,
            )
        elif protein_error:
            self._statusbar.showMessage(
                f"PSE protein load failed, binding sites loaded: {protein_error}",
                6000,
            )

    def _toggle_binding_sites(self, checked: bool):
        self._viewer.set_binding_sites_visible(checked)
        if checked:
            self._statusbar.showMessage("Binding site surfaces shown", 3000)
        else:
            self._statusbar.showMessage("Binding site surfaces hidden", 3000)

    def _on_export(self, cat_name, path_ids):
        """Export category paths to PDB files."""
        split_frame_paths = self._prompt_export_options()
        if split_frame_paths is None:
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory"
        )
        if not dir_path:
            return

        exported = export_paths_to_pdb(
            self.db,
            path_ids,
            cat_name,
            dir_path,
            split_frame_paths=split_frame_paths,
        )
        mode_text = "split by path" if split_frame_paths else "merged by frame"
        metadata_text = (
            "\nAn additional TXT metadata summary file was generated for split exports."
            if split_frame_paths
            else ""
        )
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {len(exported)} PDB files to:\n{dir_path}\n"
            f"Mode: {mode_text}{metadata_text}"
        )

    def _prompt_export_options(self) -> Optional[bool]:
        dialog = QDialog(self)
        dialog.setWindowTitle("PDB Export Options")
        dialog.setModal(True)

        layout = QVBoxLayout(dialog)
        description = QLabel(
            "Choose whether paths from the same frame are exported into one PDB "
            "file or split into multiple files."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        split_checkbox = QCheckBox("Split same-frame paths into separate files")
        split_checkbox.setChecked(False)
        layout.addWidget(split_checkbox)

        hint = QLabel(
            "Off: Category_234606_33_centroids.pdb\n"
            "On: Category_234606_33_centroids_1.pdb, "
            "Category_234606_33_centroids_2.pdb, ..."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal,
            dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return None
        return split_checkbox.isChecked()

    # ─── Frame range ────────────────────────────────────
    def _apply_frame_range(self):
        try:
            fmin = int(self._frame_min_combo.currentText())
            fmax = int(self._frame_max_combo.currentText())
            self.state.set_frame_range(fmin, fmax)
            self._path_panel.refresh()
        except ValueError:
            self._statusbar.showMessage("Invalid frame range", 3000)

    def _clear_frame_range(self):
        self.state.set_frame_range(None, None)
        self._path_panel.refresh()
        self._reload_3d()

    def _format_path_label(self, path_id: int) -> str:
        if hasattr(self.db, "format_path_label"):
            return self.db.format_path_label(path_id)
        return str(path_id)

    def _primary_analysis_db(self):
        if hasattr(self.db, "primary_db"):
            return self.db.primary_db
        return self.db

    def _sync_residue_combination_context(self):
        primary_key = getattr(self.db, "primary_dataset_key", None)
        local_residue_ids = []
        if primary_key and hasattr(self.db, "get_local_residue_ids_for_dataset"):
            local_residue_ids = self.db.get_local_residue_ids_for_dataset(
                primary_key,
                self.state.selected_residues,
            )
        else:
            local_residue_ids = [int(rid) for rid in self.state.selected_residues]
        self._residue_combination_panel.set_current_selected_residues(local_residue_ids)

    def _reset_state_for_dataset_change(self):
        self.state.selected_residues.clear()
        self.state.residue_filtered_path_ids.clear()
        self.state.lasso_added_path_ids.clear()
        self.state.lasso_removed_path_ids.clear()
        self.state.effective_path_ids.clear()
        self.state.selected_path_ids.clear()
        self.state.matched_path_ids.clear()
        self.state.categories.clear()
        self.state.category_colors.clear()
        self.state.visible_categories.clear()
        self.state.excluded_categories.clear()
        self.state.active_category = None
        self.state.frame_min = None
        self.state.frame_max = None
        self.state.current_session_id = None
        self.state.current_session_name = None
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        self._primary_residue_position_cache.clear()
        self._dataset_residue_position_cache.clear()
        self._last_preview_path_ids = []
        self._last_highlighted_preview_ids = set()
        self._last_preview_color_map = {}
        self._contained_match_preview_active = False
        self._contained_match_current_target_key = ""
        self._contained_match_current_target_path_ids.clear()
        self._contained_match_last_compare_summary = ""
        self._clear_path_delta_exclusion(reset_slider=True, refresh=False)
        self._status_selected.setText("Selected: 0 paths")
        self._viewer.clear_effective()
        self._viewer.clear_categories()
        self._viewer.clear_matched()
        self._clear_chart_views()
        for slot_attr in ("_protein_viewer_panel_top", "_protein_viewer_panel_bottom"):
            slot = getattr(self, slot_attr, None)
            panel = slot.get("panel") if isinstance(slot, dict) else None
            if panel is not None and hasattr(panel, "clear_path_overlay"):
                panel.clear_path_overlay()
        self.state.residue_selection_changed.emit()
        self.state.category_changed.emit()
        self.state.category_visibility_changed.emit()
        self.state.effective_paths_changed.emit()

    def _on_datasets_changed(self, change_info=None):
        info = change_info if isinstance(change_info, dict) else {}
        action = str(info.get("action") or "")
        reset_state = bool(info.get("reset_state", True))
        refresh_3d = bool(info.get("refresh_3d", True))
        removed_keys = [
            str(key)
            for key in (info.get("removed_keys") or [])
            if str(key)
        ]
        removed_key = str(info.get("removed_key") or "")
        if removed_key and removed_key not in removed_keys:
            removed_keys.append(removed_key)
        removed_path_ids = {
            int(path_id)
            for path_id in (info.get("removed_path_ids") or [])
        }

        for key in removed_keys:
            self.state.visible_datasets.discard(key)
            if self.state.dataset_cluster_display_key == key:
                self.state.dataset_cluster_display_key = None
            if self.state.dataset_cluster_label_key == key:
                self.state.dataset_cluster_label_key = None
            self.state.dataset_colors.pop(key, None)

        if reset_state:
            self._reset_state_for_dataset_change()
        elif removed_path_ids:
            self._prune_removed_dataset_state(removed_path_ids)
        if not refresh_3d:
            self._refresh_dataset_views_without_3d()
        else:
            self._refresh_dataset_views()
        if self.state.show_all_residues:
            positions = self._residue_positions_with_dataset_colors()
            self._viewer.set_residue_data(positions)
            self._viewer.render_residues(set())
            self._refresh_residue_labels()
        self._prime_primary_residue_positions()
        profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
        self._refresh_residue_combination_current_bottlenecks(
            profiles_by_dataset_key=profiles_by_dataset_key,
        )
        self._refresh_residue_combination_markers()
        self._refresh_observer_timeline()
        self._refresh_observer_path_overlays()
        self._refresh_observer_sequence_source_combo("top")
        self._refresh_observer_sequence_source_combo("bottom")
        if self._protein_path or self._pse_path:
            self._reload_current_protein(keep_toggle_state=True)
        dataset_count = len(self.db.list_datasets())
        if action == "remove" and not reset_state:
            self._statusbar.showMessage(
                f"Removed hidden dataset; kept current view ({dataset_count} dataset(s) loaded)",
                4000,
            )
        elif reset_state:
            self._statusbar.showMessage(
                f"Loaded {dataset_count} dataset(s); analysis state was reset",
                4000,
            )
        else:
            self._statusbar.showMessage(
                f"Loaded {dataset_count} dataset(s)",
                3000,
            )

    def _prune_removed_dataset_state(self, removed_path_ids: set[int]):
        if not removed_path_ids:
            return

        self.state.residue_filtered_path_ids.difference_update(removed_path_ids)
        self.state.lasso_added_path_ids.difference_update(removed_path_ids)
        self.state.lasso_removed_path_ids.difference_update(removed_path_ids)
        self.state.effective_path_ids.difference_update(removed_path_ids)
        self.state.selected_path_ids.difference_update(removed_path_ids)
        self.state.box_selected_paths.difference_update(removed_path_ids)
        self.state.global_excluded_paths.difference_update(removed_path_ids)
        self.state.matched_path_ids = [
            int(path_id)
            for path_id in self.state.matched_path_ids
            if int(path_id) not in removed_path_ids
        ]

        for name, path_ids in list(self.state.categories.items()):
            filtered = [
                int(path_id)
                for path_id in path_ids
                if int(path_id) not in removed_path_ids
            ]
            self.state.categories[name] = filtered

        self._last_preview_path_ids = [
            int(path_id)
            for path_id in self._last_preview_path_ids
            if int(path_id) not in removed_path_ids
        ]
        self._last_highlighted_preview_ids.difference_update(removed_path_ids)
        for path_id in list(self._last_preview_color_map.keys()):
            if int(path_id) in removed_path_ids:
                self._last_preview_color_map.pop(path_id, None)
        self._contained_match_current_target_path_ids.difference_update(removed_path_ids)
        self._path_delta_excluded_path_ids.difference_update(removed_path_ids)
        self._chart_operation_history = [
            [int(path_id) for path_id in path_ids if int(path_id) not in removed_path_ids]
            for path_ids in self._chart_operation_history
        ]

        self.state.recompute_effective()
        self.state.effective_paths_changed.emit()
        self.state.category_changed.emit()

    def _refresh_dataset_views_without_3d(self):
        self._invalidate_render_signatures()
        self._dataset_path_cache.clear()
        self._dataset_residue_pair_csv_path_cache.clear()
        self._residue_combo_present_frame_cache.clear()
        n_paths = self.db.path_count()
        fmin, fmax = self.db.frame_range()

        self._status_paths.setText(f"Paths: {n_paths:,}")
        n_res = int(self.db.get_meta("residue_count", "0"))
        self._status_residues.setText(f"Residues: {n_res:,}")

        self._frame_min_combo.clear()
        self._frame_max_combo.clear()
        self._frame_min_combo.addItem(str(fmin))
        self._frame_max_combo.addItem(str(fmax))
        self._refresh_combo_popup_width(self._frame_min_combo)
        self._refresh_combo_popup_width(self._frame_max_combo)

        self._residue_panel.set_db(self.db)
        self._residue_panel.set_dataset_filter(self._dataset_panel.selected_dataset_keys())
        self._residue_panel.load_options()

        self._path_panel.set_db(self.db)
        self._path_panel.init_model()

        self.state.ensure_dataset_defaults([
            dataset["key"] for dataset in self.db.list_datasets()
        ])

        self._dataset_panel.refresh()
        datasets = self.db.list_datasets() if hasattr(self.db, "list_datasets") else []
        self._residue_compare_panel.set_dataset_choices(datasets)
        self._residue_combination_panel.set_db(self._primary_analysis_db())
        self._residue_combination_panel.set_dataset_choices(datasets)
        self._residue_combination_panel.set_residue_position_maps(
            self._dataset_residue_position_maps()
        )
        self._residue_combination_panel.set_residue_atom_position_maps(
            self._dataset_residue_atom_position_maps()
        )
        self._sync_residue_compare_selection_state()
        self._prime_primary_residue_positions()
        profiles_by_dataset_key = self._refresh_residue_combination_current_paths()
        self._refresh_residue_combination_current_bottlenecks(
            profiles_by_dataset_key=profiles_by_dataset_key,
        )
        self._refresh_residue_combination_contained_match_status()
        self._sync_residue_combination_context()

    # ─── Cleanup ────────────────────────────────────────
    def closeEvent(self, event):
        for panel in self._iter_protein_observer_panels():
            try:
                panel.shutdown()
            except Exception:
                pass
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "shutdown"):
            try:
                viewer.shutdown()
            except Exception:
                pass
        self.db.close()
        super().closeEvent(event)


class _BackgroundRoleDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        background = index.data(Qt.BackgroundRole)
        if isinstance(background, QBrush):
            option.backgroundBrush = background


class _ResidueSummaryHeatmapDialog(QDialog):
    def __init__(self, owner: ChartWorkspaceDialog):
        super().__init__(owner)
        self._owner = owner
        self.setWindowTitle("Residue Summary Compare")
        self.setModal(False)
        self.setMinimumSize(960, 680)
        self.resize(1120, 780)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Residue Summary Compare")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        header.addWidget(title)
        header.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(28)
        close_btn.clicked.connect(self.close)
        header.addWidget(close_btn)
        layout.addLayout(header)

        self._summary = QLabel("No heatmap data.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #606266; font-size: 11px;")
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionsClickable(True)
        self._table.horizontalHeader().sectionClicked.connect(self._on_delta_header_clicked)
        self._table.setSortingEnabled(False)
        self._table.setItemDelegate(_BackgroundRoleDelegate(self._table))
        self._table.setObjectName("ResidueSummaryCompareTable")
        self._table.setAutoFillBackground(True)
        self._table.setStyleSheet(
            "QTableWidget#ResidueSummaryCompareTable { background: #ffffff; }"
            "QTableWidget#ResidueSummaryCompareTable::item { padding: 4px 8px; }"
            "QTableWidget#ResidueSummaryCompareTable::item:hover { background: rgba(0, 0, 0, 0); }"
        )
        layout.addWidget(self._table, 1)

        self._legend = QLabel("")
        self._legend.setWordWrap(True)
        self._legend.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(self._legend)

        self._payload = {"datasets": [], "rows": [], "kind": "path"}
        self._sort_column = 6
        self._sort_descending = True

    def refresh(self, payload: dict):
        self._payload = dict(payload or {"datasets": [], "rows": [], "kind": "path"})
        self._rebuild_table()

    def _on_delta_header_clicked(self, section: int):
        if section == self._sort_column:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_column = section
            self._sort_descending = section not in {0}
        self._rebuild_table()

    def _rebuild_table(self):
        datasets = list(self._payload.get("datasets", []) or [])
        delta_rows = list(self._payload.get("delta_rows", []) or [])
        kind = str(self._payload.get("kind", "path") or "path")
        kind_label = "Path Residues" if kind == "path" else "Bottleneck"

        if len(datasets) < 2 or not delta_rows:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._summary.setText(f"Need two datasets with {kind_label} data for comparison.")
            self._legend.setText("")
            return

        self._rebuild_delta_table(datasets, delta_rows, kind_label)

    def _rebuild_delta_table(self, datasets: list, delta_rows: list, kind_label: str):
        base_dataset = str(datasets[0])
        compare_dataset = str(datasets[1])
        headers = [
            "Residue",
            f"{base_dataset} Count",
            f"{base_dataset} %",
            f"{compare_dataset} Count",
            f"{compare_dataset} %",
            "Delta Count",
            "Delta %",
            "Status",
        ]
        self._table.setSortingEnabled(False)
        self._table.clear()
        self._table.setColumnCount(len(headers))
        self._table.setRowCount(len(delta_rows))
        self._table.setHorizontalHeaderLabels(headers)

        delta_rows = self._sorted_delta_rows(delta_rows)
        max_abs_delta = max(abs(int(row.get("delta_count", 0) or 0)) for row in delta_rows)
        for row_idx, row in enumerate(delta_rows):
            status = str(row.get("status", "same") or "same")
            delta_count = int(row.get("delta_count", 0) or 0)
            delta_percent = float(row.get("delta_percent", 0.0) or 0.0)
            background = self._delta_status_color(status, delta_count, max_abs_delta)
            status_text = self._delta_status_text(row)
            values = [
                str(int(row.get("residue_id", 0) or 0)),
                str(int(row.get("base_count", 0) or 0)),
                f"{float(row.get('base_percent', 0.0) or 0.0):.1f}%",
                str(int(row.get("compare_count", 0) or 0)),
                f"{float(row.get('compare_percent', 0.0) or 0.0):.1f}%",
                f"{delta_count:+d}",
                f"{delta_percent:+.1f}%",
                status_text,
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                item.setData(Qt.BackgroundRole, QBrush(background))
                item.setData(Qt.ForegroundRole, QColor("#1f2937"))
                item.setBackground(QBrush(background))
                if col_idx == 7:
                    item.setForeground(QBrush(self._delta_status_text_color(status)))
                self._table.setItem(row_idx, col_idx, item)

        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._summary.setText(
            f"{kind_label} delta table comparing {compare_dataset} against {base_dataset}. "
            f"Percent = residue path count / current selected path count."
        )
        self._legend.setText(
            "Legend: red = comparison increased, blue = comparison did not increase/decreased, "
            "green = new Bottleneck residue."
        )

    def _sorted_delta_rows(self, delta_rows: list) -> list:
        status_rank = {"new": 3, "up": 2, "same": 1, "down": 0}

        def key(row):
            status = str(row.get("status", "same") or "same")
            keys = [
                int(row.get("residue_id", 0) or 0),
                int(row.get("base_count", 0) or 0),
                float(row.get("base_percent", 0.0) or 0.0),
                int(row.get("compare_count", 0) or 0),
                float(row.get("compare_percent", 0.0) or 0.0),
                int(row.get("delta_count", 0) or 0),
                float(row.get("delta_percent", 0.0) or 0.0),
                status_rank.get(status, 1),
            ]
            primary = keys[self._sort_column] if 0 <= self._sort_column < len(keys) else keys[6]
            return (primary, int(row.get("residue_id", 0) or 0))

        return sorted(delta_rows, key=key, reverse=self._sort_descending)

    def _delta_status_text(self, row: dict) -> str:
        status = str(row.get("status", "same") or "same")
        base_rank = row.get("base_rank")
        compare_rank = row.get("compare_rank")
        if status == "new":
            return f"NEW {compare_rank}" if compare_rank is not None else "NEW"
        if status == "up":
            return f"{base_rank} UP {compare_rank}" if base_rank is not None and compare_rank is not None else "UP"
        if status == "down":
            return f"{base_rank} DOWN {compare_rank}" if base_rank is not None and compare_rank is not None else "DOWN"
        return f"{base_rank} SAME {compare_rank}" if base_rank is not None and compare_rank is not None else "SAME"

    def _delta_status_text_color(self, status: str) -> QColor:
        if status == "new":
            return QColor("#1B5E20")
        if status == "up":
            return QColor("#7F1D1D")
        return QColor("#1E3A8A")

    def _delta_status_color(self, status: str, delta_count: int, max_abs_delta: int) -> QColor:
        if status == "new":
            return QColor("#B7E4A8")
        if status == "up":
            strength = min(1.0, abs(delta_count) / max(1, max_abs_delta))
            return self._owner._blend_with_white((220, 38, 38), 0.45 + 0.35 * strength)
        strength = 0.55 if delta_count == 0 else min(0.90, 0.45 + abs(delta_count) / max(1, max_abs_delta) * 0.35)
        return self._owner._blend_with_white((37, 99, 235), strength)
