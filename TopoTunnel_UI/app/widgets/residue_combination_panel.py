"""
Panel for inspecting relationships among up to four specific residues.
"""
from __future__ import annotations

import csv
import copy
from collections import OrderedDict
import os

import numpy as np

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QBrush, QPalette
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QRadioButton,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QFrame,
    QSizePolicy,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
)

from TopoTunnel_UI.core.residue_combination import (
    COMBINATION_PROPERTY_OPTIONS,
    all_combination_rows,
    combination_property_trend,
    combination_property_value,
    compare_combination_row_sets,
    compare_combination_statistics,
    filter_combination_rows_by_keys,
    load_residue_combination_statistics_csv,
)
from TopoTunnel_UI.vis.protein_model import parse_pdb_text


SORT_OPTIONS = [
    ("By Path Change / Path Count", "path"),
    ("By Radius Change / Radius Range", "radius"),
    ("By Combination Size", "size"),
    ("By Combination Label", "label"),
]

MAX_BULK_DISTANCE_ANNOTATION_ROWS = 800
DEFERRED_DISTANCE_ANNOTATION_CHUNK_SIZE = 64


def _format_float(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_delta(value: float) -> str:
    if abs(value) < 1e-12:
        return "0"
    if abs(value) >= 1000:
        return f"{value:+.0f}"
    if abs(value) >= 100:
        return f"{value:+.1f}"
    if abs(value) >= 1:
        return f"{value:+.2f}"
    return f"{value:+.3f}"


def _first_present(record: dict, names: tuple[str, ...], default=""):
    lower_map = {str(key).strip().lower(): value for key, value in record.items()}
    for name in names:
        key = str(name).strip().lower()
        if key in lower_map:
            return lower_map[key]
    return default


def _safe_float_value(value, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int_value(value, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _blend_with_white(base: tuple[int, int, int], strength: float) -> QColor:
    strength = max(0.0, min(1.0, strength))
    red = int(round(255 + (base[0] - 255) * strength))
    green = int(round(255 + (base[1] - 255) * strength))
    blue = int(round(255 + (base[2] - 255) * strength))
    return QColor(red, green, blue)


def _delta_color(delta: float, max_abs_delta: float) -> QColor:
    if max_abs_delta <= 1e-12 or abs(delta) <= 1e-12:
        return QColor("#FFFFFF")
    strength = abs(delta) / max_abs_delta
    strength = 0.28 + strength * 0.68
    if delta > 0:
        return _blend_with_white((211, 76, 76), strength)
    return _blend_with_white((70, 116, 196), strength)


class _HeatmapItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)

        background = index.data(Qt.BackgroundRole)
        brush = None
        if isinstance(background, QBrush):
            brush = background
        elif isinstance(background, QColor):
            brush = QBrush(background)

        painter.save()
        if brush is not None:
            painter.fillRect(opt.rect, brush)

        opt.backgroundBrush = QBrush(Qt.NoBrush)
        if opt.state & QStyle.State_Selected:
            opt.palette.setBrush(QPalette.Highlight, QBrush(QColor(64, 158, 255, 44)))
            opt.palette.setBrush(QPalette.HighlightedText, opt.palette.brush(QPalette.Text))

        style = opt.widget.style() if opt.widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        else:
            super().paint(painter, opt, index)

        if opt.state & QStyle.State_Selected:
            painter.setPen(QColor("#409EFF"))
            painter.drawRect(opt.rect.adjusted(0, 0, -1, -1))
        painter.restore()


class _SortableTableWidgetItem(QTableWidgetItem):
    def __init__(self, text: str = "", sort_value=None):
        super().__init__(text)
        self._sort_value = sort_value if sort_value is not None else text

    def __lt__(self, other):
        if isinstance(other, _SortableTableWidgetItem):
            return self._sort_value < other._sort_value
        return super().__lt__(other)


class ResidueCombinationPanel(QWidget):
    visualization_changed = Signal()
    manual_update_requested = Signal(str)
    contained_match_query_requested = Signal()
    contained_match_compute_requested = Signal()
    contained_match_clear_requested = Signal()
    combination_selection_requested = Signal(object)
    left_residue_set_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._db = None
        self._left_data = None
        self._right_data = None
        self._left_csv_path = ""
        self._right_csv_path = ""
        self._pending_left_csv_path = ""
        self._pending_right_csv_path = ""
        self._dataset_choices_cache_signature: tuple = ()
        self._current_compare_cache_signature: tuple = ()
        self._csv_selection_dirty = False
        self._view_update_batch_depth = 0
        self._view_refresh_pending = False
        self._distance_annotation_request_id = 0
        self._deferred_distance_sorting_enabled: bool | None = None
        self._combination_data_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._precomputed_compare_csv_cache: OrderedDict[tuple, tuple[list[dict], str]] = OrderedDict()
        self._dynamic_compare_rows_cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._current_selected_residue_ids: list[int] = []
        self._selection_filter_active = False
        self._filter_groups: list[dict] = []
        self._loaded_scope_mode = "dataset"
        self._current_path_combination_keys: set[tuple[int, ...]] = set()
        self._current_path_rows_by_dataset: dict[str, list[dict]] = {}
        self._current_path_total_paths_by_dataset: dict[str, int] = {}
        self._current_path_filter_active = False
        self._current_bottleneck_combination_keys: set[tuple[int, ...]] = set()
        self._current_bottleneck_rows_by_dataset: dict[str, list[dict]] = {}
        self._current_bottleneck_total_paths_by_dataset: dict[str, int] = {}
        self._current_bottleneck_filter_active = False
        self._dataset_choices: list[dict] = []
        self._left_dataset_key = ""
        self._right_dataset_key = ""
        self._residue_position_maps: dict[str, dict[int, tuple[float, float, float]]] = {}
        self._residue_atom_position_maps: dict[str, dict[int, list[tuple[float, float, float]]]] = {}
        self._contained_match_rows: list[dict] = []
        self._contained_match_bottleneck_rows: list[dict] = []
        self._contained_match_summary = ""
        self._contained_match_locked = False
        self._contained_match_locked_count = 0
        self._contained_match_candidate_count = 0
        self._contained_match_result_active = False
        self._contained_match_busy = False
        self._selected_compare_marker_key = ""
        self._observer_frame_distance_inputs = {
            "left": {
                "dataset_key": "",
                "frame_paths": {},
                "combo_present_frames": {},
                "precomputed_distance_csv_path": "",
                "precomputed_distance_file_key": None,
                "precomputed_presence_stats_csv_path": "",
                "prefer_context_stats": False,
            },
            "right": {
                "dataset_key": "",
                "frame_paths": {},
                "combo_present_frames": {},
                "precomputed_distance_csv_path": "",
                "precomputed_distance_file_key": None,
                "precomputed_presence_stats_csv_path": "",
                "prefer_context_stats": False,
            },
        }
        self._pdb_residue_atom_cache: OrderedDict[tuple[str, int, int], dict[int, np.ndarray]] = OrderedDict()
        self._frame_residue_pair_distance_cache: OrderedDict[tuple, float | None] = OrderedDict()
        self._frame_combo_distance_cache: OrderedDict[tuple, float | None] = OrderedDict()
        self._precomputed_presence_stats_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._precomputed_combo_presence_stats_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._precomputed_distance_file_index_cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._precomputed_pair_distance_cache: OrderedDict[tuple, tuple[np.ndarray, np.ndarray] | None] = OrderedDict()
        self._precomputed_combo_distance_cache: OrderedDict[tuple, dict] = OrderedDict()
        # NPY rows are cumulative distances. Cache derived statistics separately
        # so repeated table refreshes do not rescan the same row.
        self._precomputed_npy_frame_stats_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._observer_frame_distance_values_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._observer_distance_stats_cache: OrderedDict[tuple, dict] = OrderedDict()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Residue Combination")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        controls_frame = QFrame()
        controls_frame.setFrameShape(QFrame.StyledPanel)
        controls_frame.setStyleSheet("QFrame{border:1px solid #e5e7eb;border-radius:6px;background:#fbfcfd;}")
        controls_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        controls_layout = QVBoxLayout(controls_frame)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        layout.addWidget(controls_frame)

        data_block, data_layout = self._create_flow_block("Data Input")
        controls_layout.addWidget(data_block)

        hint = QLabel(
            "Choose one or two loaded datasets that include `residue_combination_statistics.csv`. "
            "After selecting 2-4 residues, inspect combination statistics or compare File A vs File B."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #909399; font-size: 11px;")
        data_layout.addWidget(hint)

        self._left_csv_combo = self._build_dataset_row(data_layout, "File A")
        self._right_csv_combo = self._build_dataset_row(data_layout, "File B")

        chooser = QHBoxLayout()
        chooser.setSpacing(6)
        self._residue_combos = []
        for idx in range(4):
            chooser.addWidget(QLabel(f"Residue {idx + 1}"))
            combo = QComboBox()
            combo.setMinimumWidth(120)
            combo.currentIndexChanged.connect(self._refresh_view)
            self._residue_combos.append(combo)
            chooser.addWidget(combo)
        data_layout.addLayout(chooser)

        filter_block, filter_layout = self._create_flow_block("Display and Filter")
        controls_layout.addWidget(filter_block)
        self._sort_label = QLabel("Sort")
        self._sort_label.setVisible(False)
        self._sort_combo = QComboBox()
        for label, key in SORT_OPTIONS:
            self._sort_combo.addItem(label, key)
        self._sort_combo.currentIndexChanged.connect(self._refresh_view)
        self._sort_combo.setVisible(False)
        display_row = QGridLayout()
        display_row.setHorizontalSpacing(6)
        display_row.setVerticalSpacing(6)
        display_row.addWidget(QLabel("Size"), 0, 0)
        self._size_filter_combo = QComboBox()
        self._size_filter_combo.addItem("All", 0)
        self._size_filter_combo.addItem("2 residues", 2)
        self._size_filter_combo.addItem("3 residues", 3)
        self._size_filter_combo.addItem("4 residues", 4)
        self._size_filter_combo.currentIndexChanged.connect(self._refresh_view)
        self._size_filter_combo.setMaximumWidth(110)
        display_row.addWidget(self._size_filter_combo, 0, 1)

        display_row.addWidget(QLabel("Color"), 0, 2)
        self._color_property_combo = QComboBox()
        for label, key in COMBINATION_PROPERTY_OPTIONS:
            self._color_property_combo.addItem(label, key)
        self._color_property_combo.currentIndexChanged.connect(self._on_visualization_controls_changed)
        self._color_property_combo.setMaximumWidth(190)
        display_row.addWidget(self._color_property_combo, 0, 3)

        display_row.addWidget(QLabel("Groups"), 0, 4)
        self._add_filter_group_btn = QPushButton("+ Group")
        self._add_filter_group_btn.clicked.connect(self._add_filter_group)
        self._add_filter_group_btn.setToolTip("Add a new filter group")
        self._add_filter_group_btn.setMaximumWidth(90)
        display_row.addWidget(self._add_filter_group_btn, 0, 5)

        self._show_markers_btn = QPushButton("Show")
        self._show_markers_btn.setCheckable(True)
        self._show_markers_btn.toggled.connect(self._on_visualization_controls_changed)
        self._show_markers_btn.setToolTip("Show residue-combination markers in the main 3D view")
        self._show_markers_btn.setMaximumWidth(76)
        display_row.addWidget(self._show_markers_btn, 0, 6)
        display_row.setColumnStretch(7, 1)
        filter_layout.addLayout(display_row)

        self._filter_groups_layout = QVBoxLayout()
        self._filter_groups_layout.setSpacing(6)
        filter_layout.addLayout(self._filter_groups_layout)

        match_block, match_layout = self._create_flow_block("Matched Paths")
        controls_layout.addWidget(match_block)
        match_row = QHBoxLayout()
        match_row.setSpacing(6)
        match_row.addWidget(QLabel("Source"))
        self._contained_match_dataset_combo = QComboBox()
        self._contained_match_dataset_combo.setMinimumWidth(120)
        self._contained_match_dataset_combo.setMaximumWidth(190)
        self._contained_match_dataset_combo.setToolTip("Select the source dataset used to query matched paths")
        match_row.addWidget(self._contained_match_dataset_combo)

        match_row.addWidget(QLabel("Keep"))
        self._contained_match_keep_ratio_edit = QLineEdit("1.0")
        self._contained_match_keep_ratio_edit.setPlaceholderText("0.8")
        self._contained_match_keep_ratio_edit.setFixedWidth(76)
        self._contained_match_keep_ratio_edit.setToolTip(
            "Keep ratio for source paths before computing exit/length bounds.\n"
            "Example: 0.8 removes the shortest 10%, longest 10%, and farthest 20% exits."
        )
        match_row.addWidget(self._contained_match_keep_ratio_edit)

        self._contained_match_single_path_toggle = QRadioButton("Single-path search")
        self._contained_match_single_path_toggle.setAutoExclusive(False)
        self._contained_match_single_path_toggle.setToolTip(
            "Match target paths against exactly one selected source path using residue Jaccard similarity, exit distance, and path length"
        )
        self._contained_match_single_path_toggle.toggled.connect(self._sync_contained_match_state)
        match_row.addWidget(self._contained_match_single_path_toggle)

        self._contained_match_query_btn = QPushButton("Query")
        self._contained_match_query_btn.setEnabled(False)
        self._contained_match_query_btn.clicked.connect(self.contained_match_query_requested.emit)
        self._contained_match_query_btn.setToolTip("Query matched target paths using source-path residue, exit, and length constraints")
        self._contained_match_query_btn.setMaximumWidth(110)
        match_row.addWidget(self._contained_match_query_btn)

        self._contained_match_compute_btn = QPushButton("Compare")
        self._contained_match_compute_btn.setEnabled(False)
        self._contained_match_compute_btn.clicked.connect(self.contained_match_compute_requested.emit)
        self._contained_match_compute_btn.setToolTip(
            "Compute residue-pair differences using the current matched-path result, "
            "or compare the currently selected File A / File B paths directly"
        )
        self._contained_match_compute_btn.setMaximumWidth(110)
        match_row.addWidget(self._contained_match_compute_btn)

        self._contained_match_clear_btn = QPushButton("Reset")
        self._contained_match_clear_btn.setEnabled(False)
        self._contained_match_clear_btn.clicked.connect(self.contained_match_clear_requested.emit)
        self._contained_match_clear_btn.setToolTip("Clear the current matched-path query result and comparison result")
        self._contained_match_clear_btn.setMaximumWidth(110)
        match_row.addWidget(self._contained_match_clear_btn)
        match_row.addStretch()
        match_layout.addLayout(match_row)

        self._contained_match_single_path_row = QWidget()
        single_path_row = QHBoxLayout(self._contained_match_single_path_row)
        single_path_row.setContentsMargins(0, 0, 0, 0)
        single_path_row.setSpacing(6)
        single_path_row.addWidget(QLabel("Jaccard >"))
        self._contained_match_jaccard_spin = QDoubleSpinBox()
        self._contained_match_jaccard_spin.setRange(0.0, 1.0)
        self._contained_match_jaccard_spin.setDecimals(2)
        self._contained_match_jaccard_spin.setSingleStep(0.05)
        self._contained_match_jaccard_spin.setValue(0.5)
        self._contained_match_jaccard_spin.setFixedWidth(76)
        self._contained_match_jaccard_spin.setToolTip("Minimum residue-set Jaccard similarity; target paths must be strictly greater")
        single_path_row.addWidget(self._contained_match_jaccard_spin)
        single_path_row.addWidget(QLabel("Exit radius (Å)"))
        self._contained_match_exit_radius_spin = QDoubleSpinBox()
        self._contained_match_exit_radius_spin.setRange(0.0, 10000.0)
        self._contained_match_exit_radius_spin.setDecimals(2)
        self._contained_match_exit_radius_spin.setSingleStep(1.0)
        self._contained_match_exit_radius_spin.setValue(5.0)
        self._contained_match_exit_radius_spin.setFixedWidth(86)
        single_path_row.addWidget(self._contained_match_exit_radius_spin)
        single_path_row.addWidget(QLabel("Length radius (Å)"))
        self._contained_match_length_radius_spin = QDoubleSpinBox()
        self._contained_match_length_radius_spin.setRange(0.0, 100000.0)
        self._contained_match_length_radius_spin.setDecimals(2)
        self._contained_match_length_radius_spin.setSingleStep(1.0)
        self._contained_match_length_radius_spin.setValue(5.0)
        self._contained_match_length_radius_spin.setFixedWidth(86)
        single_path_row.addWidget(self._contained_match_length_radius_spin)
        single_path_row.addStretch()
        self._contained_match_single_path_row.setVisible(False)
        match_layout.addWidget(self._contained_match_single_path_row)

        tools_block, tools_layout = self._create_flow_block("Selection Tools")
        controls_layout.addWidget(tools_block)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._use_selected_btn = QPushButton("Use Sel")
        self._use_selected_btn.clicked.connect(self._apply_current_selected_residues)
        self._use_selected_btn.setToolTip("Use the current residue selection from the main view")
        self._use_selected_btn.setMaximumWidth(100)
        self._use_selected_btn.setVisible(False)
        actions.addWidget(self._use_selected_btn)

        self._filter_btn = QPushButton("Sel Only")
        self._filter_btn.setCheckable(True)
        self._filter_btn.setEnabled(False)
        self._filter_btn.toggled.connect(self._on_filter_toggled)
        self._filter_btn.setToolTip("Show only combinations that contain the currently selected residues")
        self._filter_btn.setMaximumWidth(100)
        actions.addWidget(self._filter_btn)

        self._current_path_filter_btn = QPushButton("Path Only")
        self._current_path_filter_btn.setCheckable(True)
        self._current_path_filter_btn.setEnabled(False)
        self._current_path_filter_btn.toggled.connect(self._on_current_path_filter_toggled)
        self._current_path_filter_btn.setToolTip("Show only combinations observed in the current selected paths")
        self._current_path_filter_btn.setMaximumWidth(100)
        self._current_path_filter_btn.setVisible(False)
        actions.addWidget(self._current_path_filter_btn)

        self._current_bottleneck_filter_btn = QPushButton("Bottleneck")
        self._current_bottleneck_filter_btn.setCheckable(True)
        self._current_bottleneck_filter_btn.setEnabled(False)
        self._current_bottleneck_filter_btn.toggled.connect(self._on_current_bottleneck_filter_toggled)
        self._current_bottleneck_filter_btn.setToolTip("Show only combinations observed at the bottleneck of selected paths")
        self._current_bottleneck_filter_btn.setMaximumWidth(110)
        actions.addWidget(self._current_bottleneck_filter_btn)

        self._show_left_residues_btn = QPushButton("Show AB Res")
        self._show_left_residues_btn.setEnabled(False)
        self._show_left_residues_btn.clicked.connect(self._emit_left_residue_set_request)
        self._show_left_residues_btn.setToolTip("Show all residues currently involved in the visible File A and File B combination rows")
        self._show_left_residues_btn.setMaximumWidth(126)
        actions.addWidget(self._show_left_residues_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_residue_slots)
        self._clear_btn.setToolTip("Clear the residue slots")
        self._clear_btn.setMaximumWidth(90)
        actions.addWidget(self._clear_btn)

        self._load_dataset_btn = QPushButton("Load DS")
        self._load_dataset_btn.clicked.connect(self._on_load_dataset_clicked)
        self._load_dataset_btn.setToolTip("Load the selected residue-combination CSV files by dataset scope")
        self._load_dataset_btn.setMaximumWidth(94)
        actions.addWidget(self._load_dataset_btn)

        self._load_path_btn = QPushButton("Load Path")
        self._load_path_btn.clicked.connect(self._on_load_path_clicked)
        self._load_path_btn.setToolTip("Load the selected residue-combination CSV files using the current path selection as context")
        self._load_path_btn.setMaximumWidth(104)
        actions.addWidget(self._load_path_btn)

        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self._refresh_view)
        self._refresh_btn.setToolTip("Refresh the current table and filters")
        self._refresh_btn.setMaximumWidth(90)
        actions.addWidget(self._refresh_btn)
        actions.addStretch()
        tools_layout.addLayout(actions)

        self._fit_button_texts(
            self._add_filter_group_btn,
            self._show_markers_btn,
            self._contained_match_query_btn,
            self._contained_match_compute_btn,
            self._contained_match_clear_btn,
            self._use_selected_btn,
            self._filter_btn,
            self._current_path_filter_btn,
            self._current_bottleneck_filter_btn,
            self._clear_btn,
            self._load_dataset_btn,
            self._load_path_btn,
            self._refresh_btn,
        )

        self._summary = QLabel("Please select 2-4 residues.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #606266; font-size: 11px;")
        controls_layout.addWidget(self._summary)

        results_frame = QFrame()
        results_frame.setFrameShape(QFrame.StyledPanel)
        results_frame.setStyleSheet("QFrame{border:1px solid #e5e7eb;border-radius:6px;background:#ffffff;}")
        results_layout = QVBoxLayout(results_frame)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(8)
        layout.addWidget(results_frame, 1)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(False)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setItemDelegate(_HeatmapItemDelegate(self._table))
        self._table.setStyleSheet("QTableView::item:hover { background-color: transparent; }")
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._update_detail)
        self._table.cellClicked.connect(self._handle_cell_clicked)
        results_layout.addWidget(self._table, 1)

        table_actions = QHBoxLayout()
        table_actions.setContentsMargins(0, 0, 0, 0)
        table_actions.addStretch()
        self._save_table_btn = QPushButton("Save Table")
        self._save_table_btn.setToolTip("Save the current visible combination table as a CSV file")
        self._save_table_btn.setFixedHeight(24)
        self._save_table_btn.setMaximumWidth(96)
        self._save_table_btn.setEnabled(False)
        self._save_table_btn.clicked.connect(self._save_current_table)
        table_actions.addWidget(self._save_table_btn)
        results_layout.addLayout(table_actions)

        self._detail = QLabel("Select a combination to inspect its detailed statistics.")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #606266; font-size: 11px;")
        results_layout.addWidget(self._detail)
        self._visible_rows: list[dict] = []
        self._add_filter_group()

    def _create_flow_block(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame{border:1px solid #dcdfe6;border-radius:6px;background:#ffffff;}")
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: 600; color: #303133;")
        layout.addWidget(title_label)
        return frame, layout

    def _build_dataset_row(self, parent_layout: QVBoxLayout, label_text: str) -> QComboBox:
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel(label_text))

        combo = QComboBox()
        combo.currentIndexChanged.connect(self._on_csv_selection_changed)
        row.addWidget(combo, 1)

        parent_layout.addLayout(row)
        return combo

    def set_db(self, db):
        self._db = db
        self._load_residue_options()
        self._auto_pick_default_csv()

    @staticmethod
    def _normalized_path(path: str) -> str:
        text = str(path or "").strip()
        return os.path.abspath(text) if text else ""

    def _dataset_choices_signature_for(self, datasets: list[dict]) -> tuple:
        signature = []
        for dataset in datasets or []:
            combo_path = self._normalized_path(dataset.get("residue_combination_statistics_path", ""))
            signature.append(
                (
                    str(dataset.get("key", "") or "").strip(),
                    str(dataset.get("name", "") or "").strip(),
                    str(dataset.get("prefix", "") or "").strip(),
                    self._normalized_path(dataset.get("path", "")),
                    self._normalized_path(dataset.get("folder", "")),
                    self._normalized_path(dataset.get("residue_statistics_path", "")),
                    combo_path,
                    self._file_cache_key(combo_path),
                )
            )
        return tuple(signature)

    def set_dataset_choices(self, datasets: list[dict]):
        normalized = [dict(item) for item in (datasets or [])]
        signature = self._dataset_choices_signature_for(normalized)
        dataset_choices_changed = signature != self._dataset_choices_cache_signature
        if dataset_choices_changed:
            self._left_data = None
            self._right_data = None
            self._left_csv_path = ""
            self._right_csv_path = ""
            self._left_dataset_key = ""
            self._right_dataset_key = ""
            self._current_compare_cache_signature = ()
            self._clear_compare_dataset_state()
            self._clear_all_compare_caches(include_combination_data=True, include_frame_caches=True)
        self._dataset_choices = normalized
        self._dataset_choices_cache_signature = signature
        self._sync_csv_dataset_combos()
        self._refresh_contained_match_dataset_combo()
        self._sync_contained_match_state()
        self._refresh_view()

    def _refresh_contained_match_dataset_combo(self):
        current_key = self.contained_match_source_dataset_key()
        allowed_keys = {
            str(key).strip()
            for key in (self._left_dataset_key, self._right_dataset_key)
            if str(key).strip()
        }
        self._contained_match_dataset_combo.blockSignals(True)
        self._contained_match_dataset_combo.clear()
        for dataset in self._dataset_choices:
            dataset_key = str(dataset.get("key") or "").strip()
            if allowed_keys and dataset_key not in allowed_keys:
                continue
            label = f"[{dataset.get('prefix', dataset.get('key', ''))}] {dataset.get('name', dataset.get('key', ''))}"
            self._contained_match_dataset_combo.addItem(label, dataset_key)
        if self._contained_match_dataset_combo.count():
            idx = self._contained_match_dataset_combo.findData(current_key)
            self._contained_match_dataset_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._contained_match_dataset_combo.blockSignals(False)

    def set_residue_position_maps(self, maps: dict[str, dict[int, tuple[float, float, float]]]):
        self._residue_position_maps = {
            str(dataset_key): {
                int(residue_id): tuple(float(v) for v in coords)
                for residue_id, coords in (position_map or {}).items()
            }
            for dataset_key, position_map in (maps or {}).items()
        }
        self._refresh_view()

    def set_residue_atom_position_maps(self, maps: dict[str, dict[int, list[tuple[float, float, float]]]]):
        self._residue_atom_position_maps = {
            str(dataset_key): {
                int(residue_id): [
                    tuple(float(v) for v in coords)
                    for coords in (atom_positions or [])
                    if coords is not None and len(coords) >= 3
                ]
                for residue_id, atom_positions in (position_map or {}).items()
            }
            for dataset_key, position_map in (maps or {}).items()
        }
        self._refresh_view()

    def set_observer_frame_distance_inputs(
        self,
        *,
        left: dict | None = None,
        right: dict | None = None,
    ) -> None:
        changed = False
        for side, payload in (("left", left), ("right", right)):
            if payload is None:
                continue
            frame_paths = {}
            for frame, path in dict(payload.get("frame_paths", {}) or {}).items():
                try:
                    frame_id = int(frame)
                except (TypeError, ValueError):
                    continue
                normalized_path = str(path or "").strip()
                if normalized_path:
                    frame_paths[frame_id] = normalized_path
            combo_present_frames = {}
            for combo, frames in dict(payload.get("combo_present_frames", {}) or {}).items():
                try:
                    combo_key = tuple(sorted(int(item) for item in combo if int(item) > 0))
                except (TypeError, ValueError):
                    continue
                if len(combo_key) < 2:
                    continue
                normalized_frames = set()
                for frame in frames or ():
                    try:
                        normalized_frames.add(int(frame))
                    except (TypeError, ValueError):
                        continue
                combo_present_frames[combo_key] = normalized_frames
            distance_path = str(payload.get("precomputed_distance_csv_path", "") or "").strip()
            normalized_payload = {
                "dataset_key": str(payload.get("dataset_key", "") or "").strip(),
                "frame_paths": frame_paths,
                "combo_present_frames": combo_present_frames,
                "precomputed_distance_csv_path": distance_path,
                "precomputed_distance_file_key": self._precomputed_csv_cache_key(distance_path),
                "precomputed_presence_stats_csv_path": str(payload.get("precomputed_presence_stats_csv_path", "") or "").strip(),
                "prefer_context_stats": bool(payload.get("prefer_context_stats")),
            }
            if normalized_payload != self._observer_frame_distance_inputs.get(side, {}):
                self._observer_frame_distance_inputs[side] = normalized_payload
                changed = True
        if not changed:
            return
        self._observer_distance_stats_cache.clear()
        self._observer_frame_distance_values_cache.clear()
        self._precomputed_combo_presence_stats_cache.clear()
        self._precomputed_combo_distance_cache.clear()
        self._precomputed_npy_frame_stats_cache.clear()
        self._refresh_view()

    def clear_observer_frame_distance_inputs(self) -> None:
        for side in ("left", "right"):
            self._observer_frame_distance_inputs[side] = {
                "dataset_key": "",
                "frame_paths": {},
                "combo_present_frames": {},
                "precomputed_distance_csv_path": "",
                "precomputed_distance_file_key": None,
                "precomputed_presence_stats_csv_path": "",
                "prefer_context_stats": False,
            }
        self._observer_distance_stats_cache.clear()
        self._observer_frame_distance_values_cache.clear()
        self._precomputed_combo_presence_stats_cache.clear()
        self._precomputed_combo_distance_cache.clear()
        self._precomputed_npy_frame_stats_cache.clear()

    def _observer_frame_distances_ready(self) -> bool:
        left_inputs = self._observer_frame_distance_inputs.get("left", {})
        right_inputs = self._observer_frame_distance_inputs.get("right", {})
        if (
            str(left_inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
            and str(right_inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
        ):
            return True
        if (
            str(left_inputs.get("precomputed_distance_csv_path", "") or "").strip()
            and str(right_inputs.get("precomputed_distance_csv_path", "") or "").strip()
        ):
            return True
        return bool(
            left_inputs.get("frame_paths")
            and right_inputs.get("frame_paths")
        )

    def _observer_presence_stats_ready(self) -> bool:
        left_inputs = self._observer_frame_distance_inputs.get("left", {})
        right_inputs = self._observer_frame_distance_inputs.get("right", {})
        return bool(
            str(left_inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
            and str(right_inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
        )

    def _empty_frame_distance_metric(self) -> dict:
        metric = {}
        for side in ("left", "right"):
            metric[f"{side}_present"] = None
            metric[f"{side}_absent"] = None
            metric[f"{side}_present_mean"] = None
            metric[f"{side}_absent_mean"] = None
            metric[f"{side}_present_n"] = 0
            metric[f"{side}_absent_n"] = 0
            metric[f"{side}_status"] = "load_pdb_frames"
        return metric

    def _empty_frame_distance_stats(self, status: str = "precomputed_distance_required") -> dict:
        return {
            "present": None,
            "absent": None,
            "present_sum": None,
            "absent_sum": None,
            "present_mean": None,
            "absent_mean": None,
            "present_n": 0,
            "absent_n": 0,
            "missing_frames": 0,
            "status": status,
        }

    def _precomputed_csv_cache_key(self, csv_path: str) -> tuple[str, int, int] | None:
        raw_path = str(csv_path or "").strip()
        if not raw_path:
            return None
        normalized_path = os.path.abspath(raw_path)
        if not os.path.isfile(normalized_path):
            return None
        try:
            stat = os.stat(normalized_path)
            return (normalized_path, int(getattr(stat, "st_mtime_ns", int(float(stat.st_mtime) * 1_000_000_000))), int(stat.st_size))
        except OSError:
            return None

    def _safe_csv_float(self, value) -> float | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def _safe_csv_bool(self, value) -> bool | None:
        text = str(value or "").strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    def _precomputed_presence_stats(self, csv_path: str) -> dict | None:
        cache_key = self._precomputed_csv_cache_key(csv_path)
        if cache_key is None:
            return None
        cached = self._precomputed_presence_stats_cache.get(cache_key)
        if cached is not None:
            self._precomputed_presence_stats_cache.move_to_end(cache_key)
            return cached

        pair_stats: dict[tuple[int, int], dict] = {}
        try:
            with open(cache_key[0], "r", encoding="utf-8-sig", newline="", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for record in reader:
                    try:
                        rid_a = int(record.get("Residue_ID_1", 0) or 0)
                        rid_b = int(record.get("Residue_ID_2", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if rid_a <= 0 or rid_b <= 0:
                        continue
                    pair_stats[(min(rid_a, rid_b), max(rid_a, rid_b))] = {
                        "present_n": int(float(record.get("Path_present_frame_count", 0) or 0)),
                        "absent_n": int(float(record.get("Path_absent_frame_count", 0) or 0)),
                        "present_mean": self._safe_csv_float(record.get("Path_present_distance_mean")),
                        "absent_mean": self._safe_csv_float(record.get("Path_absent_distance_mean")),
                        "present_median": self._safe_csv_float(record.get("Path_present_distance_median")),
                        "absent_median": self._safe_csv_float(record.get("Path_absent_distance_median")),
                        "mean_difference": self._safe_csv_float(record.get("Mean_difference_present_minus_absent")),
                        "cohen_d": self._safe_csv_float(record.get("Cohen_d")),
                    }
        except OSError:
            return None

        result = {"path": cache_key[0], "pair_stats": pair_stats}
        self._precomputed_presence_stats_cache[cache_key] = result
        self._precomputed_presence_stats_cache.move_to_end(cache_key)
        while len(self._precomputed_presence_stats_cache) > 12:
            self._precomputed_presence_stats_cache.popitem(last=False)
        return result

    def _precomputed_combo_presence_stats(self, side: str, residue_ids: tuple[int, ...]) -> dict | None:
        inputs = self._observer_frame_distance_inputs.get(side, {})
        csv_path = str(inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
        file_key = self._precomputed_csv_cache_key(csv_path)
        if file_key is None:
            return None
        dataset_key = str(inputs.get("dataset_key", "") or "").strip()
        cache_key = (side, dataset_key, file_key, residue_ids)
        cached = self._precomputed_combo_presence_stats_cache.get(cache_key)
        if cached is not None:
            self._precomputed_combo_presence_stats_cache.move_to_end(cache_key)
            return dict(cached)
        stats_data = self._precomputed_presence_stats(csv_path)
        if not stats_data:
            return None
        pair_stats_map = stats_data.get("pair_stats", {}) or {}
        pair_stats = []
        missing_pairs = 0
        for idx, rid_a in enumerate(residue_ids[:-1]):
            for rid_b in residue_ids[idx + 1:]:
                item = pair_stats_map.get((min(int(rid_a), int(rid_b)), max(int(rid_a), int(rid_b))))
                if item is None:
                    missing_pairs += 1
                    continue
                pair_stats.append(item)
        if not pair_stats or missing_pairs > 0:
            return None

        def mean_of(key: str) -> float | None:
            values = [float(item[key]) for item in pair_stats if item.get(key) is not None]
            return float(np.mean(values)) if values else None

        present_mean = mean_of("present_mean")
        absent_mean = mean_of("absent_mean")
        result = {
            "present": present_mean,
            "absent": absent_mean,
            "present_mean": present_mean,
            "absent_mean": absent_mean,
            "present_n": int(min((item.get("present_n", 0) for item in pair_stats), default=0)),
            "absent_n": int(min((item.get("absent_n", 0) for item in pair_stats), default=0)),
            "cohen_d": mean_of("cohen_d"),
            "mean_difference": mean_of("mean_difference"),
            "missing_frames": 0,
            "status": "presence_stats_csv" if len(residue_ids) == 2 else f"presence_stats_csv_pair_mean:{len(pair_stats)}",
        }
        self._precomputed_combo_presence_stats_cache[cache_key] = dict(result)
        self._precomputed_combo_presence_stats_cache.move_to_end(cache_key)
        while len(self._precomputed_combo_presence_stats_cache) > 8192:
            self._precomputed_combo_presence_stats_cache.popitem(last=False)
        return result

    def _precomputed_distance_file_index(self, csv_path: str) -> dict | None:
        cache_key = self._precomputed_csv_cache_key(csv_path)
        if cache_key is None:
            return None
        cached = self._precomputed_distance_file_index_cache.get(cache_key)
        if cached is not None:
            self._precomputed_distance_file_index_cache.move_to_end(cache_key)
            return cached

        normalized_path = cache_key[0]
        ext = os.path.splitext(normalized_path)[1].lower()
        if ext == ".npy":
            indexed = self._precomputed_npy_distance_file_index(cache_key)
        else:
            indexed = self._precomputed_csv_distance_file_index(cache_key)
        if indexed is None:
            return None
        self._precomputed_distance_file_index_cache[cache_key] = indexed
        self._precomputed_distance_file_index_cache.move_to_end(cache_key)
        while len(self._precomputed_distance_file_index_cache) > 12:
            self._precomputed_distance_file_index_cache.popitem(last=False)
        return indexed

    def _precomputed_csv_distance_file_index(self, cache_key: tuple[str, int, int]) -> dict | None:
        pair_offsets: dict[tuple[int, int], int] = {}
        frame_ids: list[int] = []
        try:
            with open(cache_key[0], "rb") as handle:
                header = handle.readline().decode("utf-8-sig", errors="ignore").strip()
                columns = [item.strip() for item in header.split(",")]
                for column in columns[3:]:
                    if column.lower().startswith("frame_"):
                        try:
                            frame_ids.append(int(column.split("_", 1)[1]))
                        except (IndexError, ValueError):
                            frame_ids.append(len(frame_ids) + 1)
                    else:
                        frame_ids.append(len(frame_ids) + 1)
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    parts = line.split(b",", 3)
                    if len(parts) < 4:
                        continue
                    try:
                        rid_a = int(parts[1])
                        rid_b = int(parts[2])
                    except ValueError:
                        continue
                    if rid_a <= 0 or rid_b <= 0:
                        continue
                    pair_offsets[(min(rid_a, rid_b), max(rid_a, rid_b))] = offset
        except OSError:
            return None

        return {
            "kind": "csv",
            "path": cache_key[0],
            "frame_ids": np.asarray(frame_ids, dtype=np.int64),
            "pair_offsets": pair_offsets,
        }

    def _precomputed_npy_distance_file_index(self, cache_key: tuple[str, int, int]) -> dict | None:
        npy_path = cache_key[0]
        combo_csv_path = os.path.join(os.path.dirname(npy_path), "residue_combination_statistics.csv")
        if not os.path.exists(combo_csv_path):
            return None
        combo_rows: dict[tuple[int, ...], int] = {}
        try:
            values = np.load(npy_path, mmap_mode="r")
        except Exception:
            return None
        if getattr(values, "ndim", 0) != 2:
            return None
        try:
            with open(combo_csv_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
                reader = csv.DictReader(handle)
                row_index = 0
                for record in reader:
                    if row_index >= int(values.shape[0]):
                        break
                    try:
                        residue_ids = tuple(
                            sorted(
                                int(item)
                                for item in str(record.get("Residue_IDs", "") or "").replace(",", ";").split(";")
                                if int(item) > 0
                            )
                        )
                    except (TypeError, ValueError):
                        row_index += 1
                        continue
                    if len(residue_ids) >= 2:
                        combo_rows[residue_ids] = row_index
                    row_index += 1
        except OSError:
            return None
        frame_count = int(values.shape[1])
        return {
            "kind": "npy",
            "path": npy_path,
            "csv_path": combo_csv_path,
            "frame_ids": np.arange(1, frame_count + 1, dtype=np.int64),
            "combo_rows": combo_rows,
            "values": values,
        }

    @staticmethod
    def _npy_distance_row_to_frame_values(row: np.ndarray) -> np.ndarray:
        values = np.asarray(row, dtype=np.float32)
        if values.size <= 1:
            return values
        finite = values[np.isfinite(values)]
        if finite.size <= 1:
            return values
        # Some generated .npy files store prefix sums; convert them back to per-frame distances.
        sample = finite[: min(int(finite.size), 256)]
        is_monotonic = bool(np.all(np.diff(sample) >= -1e-5))
        if is_monotonic and float(sample[-1]) > max(100.0, float(np.nanmedian(sample)) * 1.5):
            restored = np.diff(values, prepend=0.0).astype(np.float32, copy=False)
            return restored
        return values

    @staticmethod
    def _prefix_mean_for_frames(prefix_values: np.ndarray, frame_ids: np.ndarray, frames: set[int]) -> tuple[float | None, int]:
        if not frames:
            return None, 0
        prefix = np.asarray(prefix_values, dtype=np.float64)
        ids = np.asarray(frame_ids, dtype=np.int64)
        count = min(int(prefix.size), int(ids.size))
        if count <= 0:
            return None, 0
        prefix = prefix[:count]
        ids = ids[:count]
        requested = np.asarray(tuple(frames), dtype=np.int64)
        indices = np.searchsorted(ids, requested, side="left")
        in_range = indices < count
        if not np.any(in_range):
            return None, 0
        indices = indices[in_range]
        requested = requested[in_range]
        exact_match = ids[indices] == requested
        if not np.any(exact_match):
            return None, 0
        frame_values = np.diff(prefix, prepend=0.0)
        selected = frame_values[indices[exact_match]]
        return float(np.mean(selected)), int(selected.size)

    def _precomputed_npy_frame_distance_stats(
        self,
        side: str,
        residue_ids: tuple[int, ...],
        present_frames: set[int],
    ) -> dict | None:
        inputs = self._observer_frame_distance_inputs.get(side, {})
        path = str(inputs.get("precomputed_distance_csv_path", "") or "").strip()
        file_key = inputs.get("precomputed_distance_file_key")
        if file_key is None:
            file_key = self._precomputed_csv_cache_key(path)
        if file_key is None:
            return None
        normalized_residue_ids = tuple(int(item) for item in residue_ids)
        normalized_present_frames = tuple(sorted(int(frame) for frame in present_frames))
        cache_key = (
            side,
            str(inputs.get("dataset_key", "") or "").strip(),
            file_key,
            normalized_residue_ids,
            normalized_present_frames,
        )
        cached = self._precomputed_npy_frame_stats_cache.get(cache_key)
        if cached is not None:
            self._precomputed_npy_frame_stats_cache.move_to_end(cache_key)
            return dict(cached)

        indexed = self._precomputed_distance_file_index_cache.get(file_key)
        if indexed is None:
            indexed = self._precomputed_distance_file_index(path)
        if not indexed or indexed.get("kind") != "npy":
            return None
        combo_rows = indexed.get("combo_rows", {}) or {}
        row_index = combo_rows.get(normalized_residue_ids)
        values = indexed.get("values")
        frame_ids = np.asarray(indexed.get("frame_ids", []), dtype=np.int64)
        if row_index is None or values is None or frame_ids.size <= 0:
            return None
        try:
            # Keep the memory-mapped array lazy until this exact row is needed.
            prefix_row = np.asarray(
                values[int(row_index), :int(frame_ids.size)],
                dtype=np.float64,
            )
        except Exception:
            return None

        # The NPY contains cumulative sums. Derive all per-frame values once,
        # then split them with a vectorized mask instead of Python frame loops.
        frame_values = np.diff(prefix_row, prepend=0.0)
        if normalized_present_frames:
            present_mask = np.isin(
                frame_ids,
                np.asarray(normalized_present_frames, dtype=np.int64),
                assume_unique=True,
            )
        else:
            present_mask = np.zeros(frame_ids.shape, dtype=bool)
        present_values = frame_values[present_mask]
        absent_values = frame_values[~present_mask]
        present_mean = float(np.mean(present_values)) if present_values.size else None
        absent_mean = float(np.mean(absent_values)) if absent_values.size else None
        result = {
            "present": present_mean,
            "absent": absent_mean,
            "present_mean": present_mean,
            "absent_mean": absent_mean,
            "present_n": int(present_values.size),
            "absent_n": int(absent_values.size),
            "missing_frames": 0,
            "status": "precomputed_npy_prefix_mean",
        }
        self._precomputed_npy_frame_stats_cache[cache_key] = dict(result)
        self._precomputed_npy_frame_stats_cache.move_to_end(cache_key)
        # A typical dataset has about 15k combinations per side. Retaining one
        # complete two-dataset pass prevents LRU thrashing on the next refresh.
        while len(self._precomputed_npy_frame_stats_cache) > 32768:
            self._precomputed_npy_frame_stats_cache.popitem(last=False)
        return result

    def _precomputed_pair_distances(
        self,
        csv_path: str,
        rid_a: int,
        rid_b: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        file_key = self._precomputed_csv_cache_key(csv_path)
        if file_key is None:
            return None
        pair_key = (file_key, min(int(rid_a), int(rid_b)), max(int(rid_a), int(rid_b)))
        missing = object()
        cached = self._precomputed_pair_distance_cache.get(pair_key, missing)
        if cached is not missing:
            self._precomputed_pair_distance_cache.move_to_end(pair_key)
            return cached

        indexed = self._precomputed_distance_file_index(csv_path)
        if not indexed:
            self._precomputed_pair_distance_cache[pair_key] = None
            return None
        if indexed.get("kind") == "npy":
            row_index = indexed.get("pair_rows", {}).get((pair_key[1], pair_key[2]))
            values = indexed.get("values")
            frame_ids = np.asarray(indexed.get("frame_ids", []), dtype=np.int64)
            if row_index is None or values is None or frame_ids.size <= 0:
                self._precomputed_pair_distance_cache[pair_key] = None
                return None
            try:
                row = np.asarray(values[int(row_index), :int(frame_ids.size)], dtype=np.float32)
            except Exception:
                self._precomputed_pair_distance_cache[pair_key] = None
                return None
            if row.size <= 0:
                self._precomputed_pair_distance_cache[pair_key] = None
                return None
            result = (frame_ids, row)
            self._precomputed_pair_distance_cache[pair_key] = result
            self._precomputed_pair_distance_cache.move_to_end(pair_key)
            while len(self._precomputed_pair_distance_cache) > 16384:
                self._precomputed_pair_distance_cache.popitem(last=False)
            return result

        offset = indexed["pair_offsets"].get((pair_key[1], pair_key[2]))
        if offset is None:
            self._precomputed_pair_distance_cache[pair_key] = None
            return None
        try:
            with open(indexed["path"], "rb") as handle:
                handle.seek(int(offset))
                line = handle.readline()
        except OSError:
            self._precomputed_pair_distance_cache[pair_key] = None
            return None
        parts = line.split(b",", 3)
        if len(parts) < 4:
            self._precomputed_pair_distance_cache[pair_key] = None
            return None
        values = np.fromstring(parts[3].decode("ascii", errors="ignore"), sep=",", dtype=np.float64)
        frame_ids = np.asarray(indexed["frame_ids"], dtype=np.int64)
        if values.size <= 0 or frame_ids.size <= 0:
            self._precomputed_pair_distance_cache[pair_key] = None
            return None
        count = min(int(values.size), int(frame_ids.size))
        result = (frame_ids[:count].copy(), values[:count].copy())
        self._precomputed_pair_distance_cache[pair_key] = result
        self._precomputed_pair_distance_cache.move_to_end(pair_key)
        while len(self._precomputed_pair_distance_cache) > 16384:
            self._precomputed_pair_distance_cache.popitem(last=False)
        return result

    def _precomputed_combo_distance_values(self, side: str, residue_ids: tuple[int, ...]) -> dict | None:
        inputs = self._observer_frame_distance_inputs.get(side, {})
        csv_path = str(inputs.get("precomputed_distance_csv_path", "") or "").strip()
        file_key = inputs.get("precomputed_distance_file_key")
        if file_key is None:
            file_key = self._precomputed_csv_cache_key(csv_path)
        if file_key is None:
            return None
        dataset_key = str(inputs.get("dataset_key", "") or "").strip()
        cache_key = (side, dataset_key, file_key, residue_ids)
        cached = self._precomputed_combo_distance_cache.get(cache_key)
        if cached is not None:
            self._precomputed_combo_distance_cache.move_to_end(cache_key)
            return dict(cached)

        indexed = self._precomputed_distance_file_index_cache.get(file_key)
        if indexed is None:
            indexed = self._precomputed_distance_file_index(csv_path)
        if indexed and indexed.get("kind") == "npy":
            combo_rows = indexed.get("combo_rows", {}) or {}
            row_index = combo_rows.get(tuple(int(item) for item in residue_ids))
            values = indexed.get("values")
            frame_ids = np.asarray(indexed.get("frame_ids", []), dtype=np.int64)
            if row_index is not None and values is not None and frame_ids.size > 0:
                try:
                    row = np.asarray(values[int(row_index), :int(frame_ids.size)], dtype=np.float32)
                except Exception:
                    row = np.asarray([], dtype=np.float32)
                if row.size > 0:
                    result = {
                        "frame_ids": frame_ids[: int(row.size)],
                        "distance_values": row,
                        "missing_frames": 0,
                        "status": "precomputed_npy_prefix",
                    }
                    self._precomputed_combo_distance_cache[cache_key] = dict(result)
                    self._precomputed_combo_distance_cache.move_to_end(cache_key)
                    while len(self._precomputed_combo_distance_cache) > 8192:
                        self._precomputed_combo_distance_cache.popitem(last=False)
                    return result

        frame_ids: np.ndarray | None = None
        total_values: np.ndarray | None = None
        missing_pairs = 0
        for idx, rid_a in enumerate(residue_ids[:-1]):
            for rid_b in residue_ids[idx + 1:]:
                pair_values = self._precomputed_pair_distances(csv_path, int(rid_a), int(rid_b))
                if pair_values is None:
                    missing_pairs += 1
                    continue
                pair_frames, pair_distances = pair_values
                if frame_ids is None:
                    frame_ids = pair_frames
                    total_values = pair_distances.astype(np.float64, copy=True)
                    continue
                if np.array_equal(frame_ids, pair_frames):
                    total_values = total_values + pair_distances
                    continue
                common_frames, left_idx, right_idx = np.intersect1d(
                    frame_ids,
                    pair_frames,
                    assume_unique=False,
                    return_indices=True,
                )
                if common_frames.size <= 0:
                    missing_pairs += 1
                    continue
                frame_ids = common_frames
                total_values = total_values[left_idx] + pair_distances[right_idx]

        if frame_ids is None or total_values is None or frame_ids.size <= 0 or missing_pairs > 0:
            result = {
                "frame_ids": np.asarray([], dtype=np.int64),
                "distance_values": np.asarray([], dtype=np.float64),
                "missing_frames": 0,
                "status": f"precomputed_missing_pairs:{missing_pairs}",
            }
        else:
            result = {
                "frame_ids": np.asarray(frame_ids, dtype=np.int64),
                "distance_values": np.asarray(total_values, dtype=np.float64),
                "missing_frames": 0,
                "status": "precomputed_csv",
            }
        self._precomputed_combo_distance_cache[cache_key] = dict(result)
        self._precomputed_combo_distance_cache.move_to_end(cache_key)
        while len(self._precomputed_combo_distance_cache) > 8192:
            self._precomputed_combo_distance_cache.popitem(last=False)
        return result

    def _pdb_cache_key(self, pdb_path: str) -> tuple[str, int, int] | None:
        normalized_path = os.path.abspath(str(pdb_path or "").strip())
        if not normalized_path or not os.path.exists(normalized_path):
            return None
        try:
            stat = os.stat(normalized_path)
            return (normalized_path, int(getattr(stat, "st_mtime_ns", int(float(stat.st_mtime) * 1_000_000_000))), int(stat.st_size))
        except OSError:
            return None

    def _pdb_residue_atom_positions(self, pdb_path: str) -> dict[int, np.ndarray]:
        cache_key = self._pdb_cache_key(pdb_path)
        if cache_key is None:
            return {}
        cached = self._pdb_residue_atom_cache.get(cache_key)
        if cached is not None:
            self._pdb_residue_atom_cache.move_to_end(cache_key)
            return cached
        try:
            with open(cache_key[0], "r", encoding="utf-8", errors="ignore") as handle:
                atoms = parse_pdb_text(handle.read())
        except OSError:
            return {}
        residue_atoms: dict[int, list[np.ndarray]] = {}
        for atom in atoms:
            try:
                residue_id = int(atom.get("res_seq", 0) or 0)
                position = np.asarray(atom.get("position"), dtype=np.float64)
            except Exception:
                continue
            if residue_id <= 0 or position.ndim != 1 or position.shape[0] < 3:
                continue
            residue_atoms.setdefault(residue_id, []).append(position[:3])
        parsed: dict[int, np.ndarray] = {
            residue_id: np.vstack(points).astype(np.float64, copy=False)
            for residue_id, points in residue_atoms.items()
            if points
        }
        self._pdb_residue_atom_cache[cache_key] = parsed
        self._pdb_residue_atom_cache.move_to_end(cache_key)
        while len(self._pdb_residue_atom_cache) > 160:
            self._pdb_residue_atom_cache.popitem(last=False)
        return parsed

    def _residue_pair_distance_for_frame(
        self,
        residue_atoms: dict[int, np.ndarray],
        pdb_key: tuple[str, int, int],
        rid_a: int,
        rid_b: int,
    ) -> float | None:
        pair_key = (pdb_key, min(int(rid_a), int(rid_b)), max(int(rid_a), int(rid_b)))
        missing = object()
        cached = self._frame_residue_pair_distance_cache.get(pair_key, missing)
        if cached is not missing:
            self._frame_residue_pair_distance_cache.move_to_end(pair_key)
            return cached
        arr_a = residue_atoms.get(int(rid_a))
        arr_b = residue_atoms.get(int(rid_b))
        if arr_a is None or arr_b is None or arr_a.size <= 0 or arr_b.size <= 0:
            self._frame_residue_pair_distance_cache[pair_key] = None
            return None
        diff = arr_a[:, None, :] - arr_b[None, :, :]
        distances = np.sqrt(np.sum(diff * diff, axis=2))
        if distances.size <= 0:
            self._frame_residue_pair_distance_cache[pair_key] = None
            return None
        distance = float(np.min(distances))
        self._frame_residue_pair_distance_cache[pair_key] = distance
        self._frame_residue_pair_distance_cache.move_to_end(pair_key)
        while len(self._frame_residue_pair_distance_cache) > 262144:
            self._frame_residue_pair_distance_cache.popitem(last=False)
        return distance

    def _residue_combo_distance_for_frame(self, residue_ids: tuple[int, ...], pdb_path: str) -> float | None:
        pdb_key = self._pdb_cache_key(pdb_path)
        if pdb_key is None:
            return None
        distance_cache_key = (pdb_key, residue_ids)
        missing = object()
        cached = self._frame_combo_distance_cache.get(distance_cache_key, missing)
        if cached is not missing:
            self._frame_combo_distance_cache.move_to_end(distance_cache_key)
            return cached
        residue_atoms = self._pdb_residue_atom_positions(pdb_path)
        if not residue_atoms:
            self._frame_combo_distance_cache[distance_cache_key] = None
            return None
        total = 0.0
        for idx, rid_a in enumerate(residue_ids[:-1]):
            for rid_b in residue_ids[idx + 1:]:
                distance = self._residue_pair_distance_for_frame(residue_atoms, pdb_key, int(rid_a), int(rid_b))
                if distance is None:
                    self._frame_combo_distance_cache[distance_cache_key] = None
                    return None
                total += float(distance)
        self._frame_combo_distance_cache[distance_cache_key] = total
        self._frame_combo_distance_cache.move_to_end(distance_cache_key)
        while len(self._frame_combo_distance_cache) > 131072:
            self._frame_combo_distance_cache.popitem(last=False)
        return total

    def _observer_input_cache_signature(self, side: str) -> tuple:
        inputs = self._observer_frame_distance_inputs.get(side, {})
        frame_paths = inputs.get("frame_paths", {}) or {}
        frame_signature = tuple(
            sorted(
                (int(frame), self._normalized_path(path))
                for frame, path in frame_paths.items()
                if str(path or "").strip()
            )
        )
        distance_path = str(inputs.get("precomputed_distance_csv_path", "") or "").strip()
        presence_path = str(inputs.get("precomputed_presence_stats_csv_path", "") or "").strip()
        return (
            str(inputs.get("dataset_key", "") or "").strip(),
            self._precomputed_csv_cache_key(distance_path) or self._normalized_path(distance_path),
            self._precomputed_csv_cache_key(presence_path) or self._normalized_path(presence_path),
            frame_signature,
        )

    def _frame_distance_values_for_side(self, side: str, residue_ids: tuple[int, ...]) -> dict:
        precomputed = self._precomputed_combo_distance_values(side, residue_ids)
        if precomputed is not None and str(precomputed.get("status", "")).startswith("precomputed_"):
            return precomputed

        inputs = self._observer_frame_distance_inputs.get(side, {})
        frame_paths = inputs.get("frame_paths", {}) or {}
        frame_signature = tuple(sorted((int(frame), str(path)) for frame, path in frame_paths.items()))
        cache_key = (side, self._observer_input_cache_signature(side), residue_ids)
        cached = self._observer_frame_distance_values_cache.get(cache_key)
        if cached is not None:
            self._observer_frame_distance_values_cache.move_to_end(cache_key)
            return dict(cached)

        values_by_frame: dict[int, float] = {}
        missing_frames = 0
        for frame, path in frame_signature:
            distance = self._residue_combo_distance_for_frame(residue_ids, path)
            if distance is None:
                missing_frames += 1
                continue
            values_by_frame[int(frame)] = float(distance)
        result = {
            "values_by_frame": values_by_frame,
            "frame_ids": np.asarray(sorted(values_by_frame.keys()), dtype=np.int64),
            "distance_values": np.asarray(
                [values_by_frame[frame] for frame in sorted(values_by_frame.keys())],
                dtype=np.float64,
            ),
            "missing_frames": missing_frames,
            "status": "ok" if frame_paths else "load_pdb_frames",
        }
        self._observer_frame_distance_values_cache[cache_key] = dict(result)
        self._observer_frame_distance_values_cache.move_to_end(cache_key)
        while len(self._observer_frame_distance_values_cache) > 8192:
            self._observer_frame_distance_values_cache.popitem(last=False)
        return result

    def _frame_distance_stats_for_side(self, side: str, residue_ids: tuple[int, ...]) -> dict:
        inputs = self._observer_frame_distance_inputs.get(side, {})
        combo_present_frames = inputs.get("combo_present_frames", {}) or {}
        present_frames = set(int(frame) for frame in combo_present_frames.get(residue_ids, set()) or set())
        prefer_context_stats = bool(inputs.get("prefer_context_stats"))
        allow_precomputed_presence = (
            not prefer_context_stats
            and
            self._loaded_scope_mode == "dataset"
            and not self._current_path_filter_active
            and not self._current_bottleneck_filter_active
        )
        if allow_precomputed_presence:
            precomputed_stats = self._precomputed_combo_presence_stats(side, residue_ids)
            if precomputed_stats is not None:
                return precomputed_stats
        npy_stats = self._precomputed_npy_frame_distance_stats(side, residue_ids, present_frames)
        if npy_stats is not None:
            return npy_stats

        distance_values = self._precomputed_combo_distance_values(side, residue_ids)
        if distance_values is None:
            return self._empty_frame_distance_stats()
        frame_ids = np.asarray(distance_values.get("frame_ids", np.asarray([], dtype=np.int64)), dtype=np.int64)
        values = np.asarray(distance_values.get("distance_values", np.asarray([], dtype=np.float64)), dtype=np.float64)
        cache_key = (
            side,
            self._observer_input_cache_signature(side),
            residue_ids,
            tuple(int(frame) for frame in frame_ids.tolist()),
            tuple(sorted(present_frames)),
        )
        cached = self._observer_distance_stats_cache.get(cache_key)
        if cached is not None:
            self._observer_distance_stats_cache.move_to_end(cache_key)
            return dict(cached)
        if frame_ids.size and present_frames:
            present_mask = np.isin(frame_ids, np.asarray(sorted(present_frames), dtype=np.int64))
        else:
            present_mask = np.zeros(frame_ids.shape, dtype=bool)
        present_values = values[present_mask]
        absent_values = values[~present_mask]
        present_sum = float(np.sum(present_values)) if present_values.size else None
        absent_sum = float(np.sum(absent_values)) if absent_values.size else None
        present_mean = float(np.mean(present_values)) if present_values.size else None
        absent_mean = float(np.mean(absent_values)) if absent_values.size else None
        stats = {
            "present": present_mean,
            "absent": absent_mean,
            "present_sum": present_sum,
            "absent_sum": absent_sum,
            "present_mean": present_mean,
            "absent_mean": absent_mean,
            "present_n": len(present_values),
            "absent_n": len(absent_values),
            "missing_frames": int(distance_values.get("missing_frames", 0) or 0),
            "status": str(distance_values.get("status", "ok") or "ok"),
        }
        self._observer_distance_stats_cache[cache_key] = dict(stats)
        self._observer_distance_stats_cache.move_to_end(cache_key)
        while len(self._observer_distance_stats_cache) > 4096:
            self._observer_distance_stats_cache.popitem(last=False)
        return stats

    def _annotate_compare_distance_sums(self, rows: list[dict]) -> None:
        frame_distances_ready = self._observer_frame_distances_ready()
        for row in rows:
            residue_ids = tuple(sorted(int(rid) for rid in row.get("residue_ids", ()) if int(rid) > 0))
            metrics = row.setdefault("metrics", {})
            distance_metric = self._empty_frame_distance_metric()
            if len(residue_ids) < 2 or not frame_distances_ready:
                metrics["residue_pair_distance_sum"] = distance_metric
                continue
            for side in ("left", "right"):
                stats = self._frame_distance_stats_for_side(side, residue_ids)
                for key, value in stats.items():
                    distance_metric[f"{side}_{key}"] = value
            metrics["residue_pair_distance_sum"] = distance_metric

    def _set_empty_compare_distance_sums(self, rows: list[dict]) -> None:
        for row in rows:
            row.setdefault("metrics", {})["residue_pair_distance_sum"] = self._empty_frame_distance_metric()

    def _bulk_presence_stats_available(self) -> bool:
        if not self._observer_presence_stats_ready():
            return False
        return not any(
            bool(self._observer_frame_distance_inputs.get(side, {}).get("prefer_context_stats"))
            for side in ("left", "right")
        )

    def _cancel_deferred_distance_annotation(self) -> None:
        self._distance_annotation_request_id += 1
        if self._deferred_distance_sorting_enabled is not None:
            self._table.setSortingEnabled(self._deferred_distance_sorting_enabled)
            self._deferred_distance_sorting_enabled = None

    def _schedule_deferred_distance_annotation(self) -> None:
        row_items = [
            self._table.item(row_idx, 0)
            for row_idx in range(self._table.rowCount())
            if self._table.item(row_idx, 0) is not None
        ]
        if not row_items:
            return
        request_id = self._distance_annotation_request_id
        self._deferred_distance_sorting_enabled = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        QTimer.singleShot(
            0,
            lambda: self._run_deferred_distance_annotation(request_id, row_items, 0),
        )

    def _run_deferred_distance_annotation(
        self,
        request_id: int,
        row_items: list[QTableWidgetItem],
        start: int,
    ) -> None:
        if request_id != self._distance_annotation_request_id:
            return
        end = min(len(row_items), start + DEFERRED_DISTANCE_ANNOTATION_CHUNK_SIZE)
        for item in row_items[start:end]:
            row_idx = item.row()
            if row_idx < 0:
                continue
            payload = item.data(Qt.UserRole) or {}
            row = payload.get("row") if isinstance(payload, dict) else None
            if not isinstance(row, dict):
                continue
            self._annotate_compare_distance_sums([row])
            self._set_compare_distance_cells(row_idx, row)
        if end < len(row_items):
            QTimer.singleShot(
                0,
                lambda: self._run_deferred_distance_annotation(request_id, row_items, end),
            )
            return
        if self._deferred_distance_sorting_enabled is not None:
            self._table.setSortingEnabled(self._deferred_distance_sorting_enabled)
            self._deferred_distance_sorting_enabled = None
        self._update_detail()

    def _sync_csv_dataset_combos(self):
        current_left = str(
            self._left_csv_combo.currentData()
            or self._pending_left_csv_path
            or self._left_csv_path
            or ""
        ).strip()
        current_right = str(
            self._right_csv_combo.currentData()
            or self._pending_right_csv_path
            or self._right_csv_path
            or ""
        ).strip()
        options = [
            item for item in self._dataset_choices
            if item.get("residue_combination_statistics_path")
        ]

        for combo, current_value, empty_label in (
            (self._left_csv_combo, current_left, "Choose File A dataset"),
            (self._right_csv_combo, current_right, "Choose File B dataset"),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(empty_label, "")
            for dataset in options:
                label = (
                    f"[{dataset.get('prefix', dataset.get('key', ''))}] "
                    f"{dataset.get('name', dataset.get('key', ''))}"
                )
                combo.addItem(label, dataset.get("residue_combination_statistics_path", ""))
            idx = combo.findData(current_value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.count() > 1:
                combo.setCurrentIndex(1 if combo is self._left_csv_combo else 0)
            combo.blockSignals(False)
        self._pending_left_csv_path = str(self._left_csv_combo.currentData() or "").strip()
        self._pending_right_csv_path = str(self._right_csv_combo.currentData() or "").strip()
        self._csv_selection_dirty = self._is_csv_selection_dirty()

    def _dataset_key_for_path(self, path: str) -> str:
        normalized = os.path.abspath(str(path or "")).strip()
        for dataset in self._dataset_choices:
            candidate = os.path.abspath(str(dataset.get("residue_combination_statistics_path", "") or "")).strip()
            if candidate and candidate == normalized:
                return str(dataset.get("key", "") or "")
        return ""

    def _csv_selection_paths(self) -> tuple[str, str]:
        return (
            str(self._left_csv_combo.currentData() or "").strip(),
            str(self._right_csv_combo.currentData() or "").strip(),
        )

    def _is_csv_selection_dirty(self) -> bool:
        pending_left = self._pending_left_csv_path
        pending_right = self._pending_right_csv_path
        if pending_left != self._left_csv_path or pending_right != self._right_csv_path:
            return bool(pending_left or pending_right or self._left_csv_path or self._right_csv_path)
        if (pending_left and self._left_data is None) or (pending_right and self._right_data is None):
            return True
        return False

    def _on_csv_selection_changed(self):
        self._pending_left_csv_path, self._pending_right_csv_path = self._csv_selection_paths()
        if not self._is_csv_selection_dirty():
            self._csv_selection_dirty = False
            self._refresh_view()
            return
        self._csv_selection_dirty = True
        self._left_data = None
        self._right_data = None
        self._left_csv_path = ""
        self._right_csv_path = ""
        self._left_dataset_key = self._dataset_key_for_path(self._pending_left_csv_path)
        self._right_dataset_key = self._dataset_key_for_path(self._pending_right_csv_path)
        self._current_compare_cache_signature = ()
        self._clear_compare_dataset_state()
        self._clear_compare_result_caches()
        self._refresh_contained_match_dataset_combo()
        self._sync_contained_match_state()
        self._sync_current_context_filter_buttons()
        self._refresh_view()

    def _current_compare_dataset_signature(self) -> tuple:
        return (
            self._left_dataset_key,
            self._right_dataset_key,
            self._normalized_path(self._left_csv_path),
            self._normalized_path(self._right_csv_path),
            self._file_cache_key(self._left_csv_path),
            self._file_cache_key(self._right_csv_path),
        )

    def _clear_all_compare_caches(
        self,
        *,
        include_combination_data: bool = False,
        include_frame_caches: bool = False,
    ) -> None:
        if include_combination_data:
            self._combination_data_cache.clear()
        self._precomputed_compare_csv_cache.clear()
        self._dynamic_compare_rows_cache.clear()
        self._precomputed_presence_stats_cache.clear()
        self._precomputed_combo_presence_stats_cache.clear()
        self._precomputed_distance_file_index_cache.clear()
        self._precomputed_pair_distance_cache.clear()
        self._precomputed_combo_distance_cache.clear()
        self._precomputed_npy_frame_stats_cache.clear()
        self._observer_frame_distance_values_cache.clear()
        self._observer_distance_stats_cache.clear()
        if include_frame_caches:
            self._pdb_residue_atom_cache.clear()
            self._frame_residue_pair_distance_cache.clear()
            self._frame_combo_distance_cache.clear()

    def _clear_compare_result_caches(self) -> None:
        self._precomputed_compare_csv_cache.clear()
        self._dynamic_compare_rows_cache.clear()

    def set_contained_match_result(
        self,
        rows: list[dict],
        summary: str,
        *,
        active: bool = True,
        bottleneck_rows: list[dict] | None = None,
    ):
        self._contained_match_rows = [dict(row) for row in (rows or [])]
        self._contained_match_bottleneck_rows = [dict(row) for row in (bottleneck_rows or [])]
        self._contained_match_summary = str(summary or "")
        self._contained_match_result_active = bool(active)
        self._refresh_view()

    def clear_contained_match_result(self):
        self._contained_match_rows = []
        self._contained_match_bottleneck_rows = []
        self._contained_match_summary = ""
        self._contained_match_result_active = False
        self._refresh_view()

    def _clear_compare_dataset_state(self):
        self._selected_compare_marker_key = ""
        self._contained_match_rows = []
        self._contained_match_bottleneck_rows = []
        self._contained_match_summary = ""
        self._contained_match_result_active = False
        self._visible_rows = []
        self._dynamic_compare_rows_cache.clear()
        self.clear_observer_frame_distance_inputs()

    def set_contained_match_status(
        self,
        *,
        locked: bool,
        locked_count: int = 0,
        candidate_count: int = 0,
    ):
        self._contained_match_locked = bool(locked)
        self._contained_match_locked_count = max(0, int(locked_count))
        self._contained_match_candidate_count = max(0, int(candidate_count))
        self._sync_contained_match_state()

    def set_current_selected_residues(self, residue_ids: list[int]):
        self._current_selected_residue_ids = [int(rid) for rid in residue_ids if int(rid) > 0]
        self._use_selected_btn.setEnabled(2 <= len(self._current_selected_residue_ids) <= 4)

    def _set_loaded_scope_mode(self, mode: str) -> bool:
        normalized = "path" if str(mode or "").strip().lower() == "path" else "dataset"
        changed = self._loaded_scope_mode != normalized
        self._loaded_scope_mode = normalized
        return changed

    def _dataset_total_path_count(self, dataset_key: str) -> int:
        normalized_key = str(dataset_key or "").strip()
        if not normalized_key:
            return 0
        for dataset in self._dataset_choices:
            if str(dataset.get("key", "") or "").strip() != normalized_key:
                continue
            try:
                return max(0, int(dataset.get("path_count", 0) or 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def _source_total_path_count(self, dataset_key: str, fallback_data: dict | None = None) -> int:
        total_paths = self._dataset_total_path_count(dataset_key)
        if total_paths > 0:
            return total_paths
        if fallback_data is None:
            return 0
        try:
            return max(0, int(fallback_data.get("total_paths", 0) or 0))
        except (TypeError, ValueError, AttributeError):
            return 0

    def _normalized_combination_data(self, data: dict | None, dataset_key: str) -> dict | None:
        if not data:
            return None
        normalized = dict(data)
        total_paths = self._source_total_path_count(dataset_key, normalized)
        if total_paths > 0:
            normalized["total_paths"] = int(total_paths)
        return normalized

    def _scope_total_paths_for_dataset(self, dataset_key: str, *, bottleneck: bool = False) -> int:
        normalized_key = str(dataset_key or "").strip()
        if not normalized_key:
            return 0
        if bottleneck:
            return max(0, int(self._current_bottleneck_total_paths_by_dataset.get(normalized_key, 0) or 0))
        if self._loaded_scope_mode == "path":
            return max(0, int(self._current_path_total_paths_by_dataset.get(normalized_key, 0) or 0))
        fallback_data = None
        if normalized_key == str(self._left_dataset_key or "").strip():
            fallback_data = self._left_data
        elif normalized_key == str(self._right_dataset_key or "").strip():
            fallback_data = self._right_data
        return self._source_total_path_count(normalized_key, fallback_data)

    def set_current_path_combination_keys(self, combo_keys):
        previous_enabled = self._current_path_filter_btn.isEnabled()
        self._current_path_combination_keys = {
            tuple(int(item) for item in combo)
            for combo in (combo_keys or set())
            if len(combo) >= 2
        }
        enabled = bool(self._left_data or self._right_data)
        self._current_path_filter_btn.setEnabled(enabled)
        should_refresh = False
        if not enabled and self._current_path_filter_btn.isChecked():
            self._current_path_filter_btn.blockSignals(True)
            self._current_path_filter_btn.setChecked(False)
            self._current_path_filter_btn.blockSignals(False)
            self._current_path_filter_active = False
            should_refresh = True
        elif self._current_path_filter_active:
            should_refresh = True
        elif previous_enabled != enabled:
            self._current_path_filter_btn.update()
        if should_refresh:
            self._refresh_view()

    def set_current_path_combination_rows(
        self,
        rows_by_dataset: dict[str, list[dict]],
        total_paths_by_dataset: dict[str, int] | None = None,
    ):
        self._current_path_rows_by_dataset = {
            str(dataset_key): [dict(row) for row in (rows or [])]
            for dataset_key, rows in (rows_by_dataset or {}).items()
        }
        self._current_path_total_paths_by_dataset = {
            str(dataset_key): max(0, int(total_paths))
            for dataset_key, total_paths in (total_paths_by_dataset or {}).items()
        }
        if self._loaded_scope_mode == "path" or self._current_path_filter_active:
            self._refresh_view()

    def set_current_bottleneck_combination_keys(self, combo_keys):
        previous_enabled = self._current_bottleneck_filter_btn.isEnabled()
        self._current_bottleneck_combination_keys = {
            tuple(int(item) for item in combo)
            for combo in (combo_keys or set())
            if len(combo) >= 2
        }
        enabled = bool(self._left_data or self._right_data)
        self._current_bottleneck_filter_btn.setEnabled(enabled)
        should_refresh = False
        if not enabled and self._current_bottleneck_filter_btn.isChecked():
            self._current_bottleneck_filter_btn.blockSignals(True)
            self._current_bottleneck_filter_btn.setChecked(False)
            self._current_bottleneck_filter_btn.blockSignals(False)
            self._current_bottleneck_filter_active = False
            should_refresh = True
        elif self._current_bottleneck_filter_active:
            should_refresh = True
        elif previous_enabled != enabled:
            self._current_bottleneck_filter_btn.update()
        if should_refresh:
            self._refresh_view()

    def set_current_bottleneck_combination_rows(
        self,
        rows_by_dataset: dict[str, list[dict]],
        total_paths_by_dataset: dict[str, int] | None = None,
    ):
        self._current_bottleneck_rows_by_dataset = {
            str(dataset_key): [dict(row) for row in (rows or [])]
            for dataset_key, rows in (rows_by_dataset or {}).items()
        }
        self._current_bottleneck_total_paths_by_dataset = {
            str(dataset_key): max(0, int(total_paths))
            for dataset_key, total_paths in (total_paths_by_dataset or {}).items()
        }
        if self._current_bottleneck_filter_active:
            self._refresh_view()

    def _auto_pick_default_csv(self):
        if str(self._left_csv_combo.currentData() or "").strip():
            return
        options = [
            item for item in self._dataset_choices
            if item.get("residue_combination_statistics_path")
        ]
        if not options:
            return
        self._sync_csv_dataset_combos()

    def _load_residue_options(self):
        options = self._db.get_residue_options() if self._db is not None else []
        current_values = [combo.currentData() for combo in self._residue_combos]
        for combo in self._residue_combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("-", 0)
            for option in options:
                combo.addItem(option["label"], int(option["value"]))
            combo.blockSignals(False)
        for combo, value in zip(self._residue_combos, current_values):
            idx = combo.findData(value)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

    def begin_view_update_batch(self) -> None:
        self._view_update_batch_depth += 1

    def end_view_update_batch(self) -> None:
        if self._view_update_batch_depth <= 0:
            return
        self._view_update_batch_depth -= 1
        self._flush_pending_view_refresh()

    def _flush_pending_view_refresh(self) -> None:
        if self._view_update_batch_depth == 0 and self._view_refresh_pending:
            self._view_refresh_pending = False
            self._refresh_view()

    def _reload_csvs(self, *, refresh: bool = True):
        previous_signature = self._current_compare_cache_signature
        previous_left_dataset_key = self._left_dataset_key
        previous_right_dataset_key = self._right_dataset_key
        was_dirty = bool(self._csv_selection_dirty)
        had_contained_state = (
            self._contained_match_locked
            or self._contained_match_candidate_count > 0
            or self._contained_match_result_active
        )
        self._pending_left_csv_path, self._pending_right_csv_path = self._csv_selection_paths()
        left_path = self._pending_left_csv_path
        right_path = self._pending_right_csv_path
        self._left_csv_path = left_path
        self._right_csv_path = right_path
        self._left_dataset_key = self._dataset_key_for_path(left_path)
        self._right_dataset_key = self._dataset_key_for_path(right_path)
        new_signature = self._current_compare_dataset_signature()
        dataset_mapping_changed = (
            previous_left_dataset_key != self._left_dataset_key
            or previous_right_dataset_key != self._right_dataset_key
        )
        compare_inputs_changed = previous_signature != new_signature or dataset_mapping_changed
        if dataset_mapping_changed:
            self._clear_compare_dataset_state()
        if compare_inputs_changed:
            self._clear_compare_result_caches()

        previous_left_path_key = previous_signature[2] if len(previous_signature) > 4 else ""
        previous_right_path_key = previous_signature[3] if len(previous_signature) > 4 else ""
        previous_left_file_key = previous_signature[4] if len(previous_signature) > 4 else None
        previous_right_file_key = previous_signature[5] if len(previous_signature) > 5 else None
        current_left_path_key = new_signature[2] if len(new_signature) > 4 else ""
        current_right_path_key = new_signature[3] if len(new_signature) > 4 else ""
        current_left_file_key = new_signature[4] if len(new_signature) > 4 else None
        current_right_file_key = new_signature[5] if len(new_signature) > 5 else None
        left_needs_load = bool(left_path) and (
            self._left_data is None
            or previous_left_path_key != current_left_path_key
            or previous_left_file_key != current_left_file_key
        )
        right_needs_load = bool(right_path) and (
            self._right_data is None
            or previous_right_path_key != current_right_path_key
            or previous_right_file_key != current_right_file_key
        )
        view_needs_refresh = bool(compare_inputs_changed or was_dirty)

        try:
            if left_path:
                if left_needs_load:
                    self._left_data = self._normalized_combination_data(
                        self._load_combination_statistics_cached(left_path),
                        self._left_dataset_key,
                    )
                    view_needs_refresh = True
            else:
                if self._left_data is not None:
                    view_needs_refresh = True
                self._left_data = None
            if right_path:
                if right_needs_load:
                    self._right_data = self._normalized_combination_data(
                        self._load_combination_statistics_cached(right_path),
                        self._right_dataset_key,
                    )
                    view_needs_refresh = True
            else:
                if self._right_data is not None:
                    view_needs_refresh = True
                self._right_data = None
            self._csv_selection_dirty = False
            self._current_compare_cache_signature = self._current_compare_dataset_signature()
        except Exception as exc:
            self._left_data = None
            self._right_data = None
            self._current_compare_cache_signature = ()
            self._csv_selection_dirty = bool(left_path or right_path)
            QMessageBox.warning(self, "Residue Combination", f"Failed to load combination-statistics CSV:\n{exc}")
            view_needs_refresh = True
        if dataset_mapping_changed and had_contained_state:
            self.contained_match_clear_requested.emit()
        self._refresh_contained_match_dataset_combo()
        self._sync_contained_match_state()
        self._sync_current_context_filter_buttons()
        if view_needs_refresh:
            if refresh:
                self._refresh_view()
            else:
                self._view_refresh_pending = True

    def _set_current_path_scope_active(self, active: bool) -> bool:
        active = bool(active)
        changed = self._current_path_filter_active != active
        self._current_path_filter_active = active
        if self._current_path_filter_btn.isChecked() != active:
            self._current_path_filter_btn.blockSignals(True)
            self._current_path_filter_btn.setChecked(active)
            self._current_path_filter_btn.blockSignals(False)
            changed = True
        return changed

    def _on_load_dataset_clicked(self):
        scope_changed = self._set_loaded_scope_mode("dataset")
        self._reload_csvs(refresh=False)
        scope_changed = self._set_current_path_scope_active(False) or scope_changed
        self._view_refresh_pending = self._view_refresh_pending or scope_changed
        self.manual_update_requested.emit("dataset")
        self._flush_pending_view_refresh()

    def _on_load_path_clicked(self):
        scope_changed = self._set_loaded_scope_mode("path")
        self._reload_csvs(refresh=False)
        scope_changed = self._set_current_path_scope_active(True) or scope_changed
        self._view_refresh_pending = self._view_refresh_pending or scope_changed
        self.manual_update_requested.emit("path")
        self._flush_pending_view_refresh()

    def _sync_current_context_filter_buttons(self):
        enabled = bool(self._left_data or self._right_data)
        for button, attr_name in (
            (self._current_path_filter_btn, "_current_path_filter_active"),
            (self._current_bottleneck_filter_btn, "_current_bottleneck_filter_active"),
        ):
            button.setEnabled(enabled)
            if not enabled and button.isChecked():
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
                setattr(self, attr_name, False)

    def _file_cache_key(self, path: str) -> tuple[str, int, int] | None:
        normalized_path = os.path.abspath(str(path or "").strip())
        if not normalized_path or not os.path.exists(normalized_path):
            return None
        try:
            stat = os.stat(normalized_path)
            return (normalized_path, int(getattr(stat, "st_mtime_ns", int(float(stat.st_mtime) * 1_000_000_000))), int(stat.st_size))
        except OSError:
            return None

    def _load_combination_statistics_cached(self, path: str) -> dict:
        cache_key = self._file_cache_key(path)
        if cache_key is None:
            return load_residue_combination_statistics_csv(path)
        cached = self._combination_data_cache.get(cache_key)
        if cached is not None:
            self._combination_data_cache.move_to_end(cache_key)
            return cached
        data = load_residue_combination_statistics_csv(cache_key[0])
        self._combination_data_cache[cache_key] = data
        self._combination_data_cache.move_to_end(cache_key)
        while len(self._combination_data_cache) > 16:
            self._combination_data_cache.popitem(last=False)
        return data

    def _precomputed_compare_search_roots(self) -> list[str]:
        roots: list[str] = []
        for path in (self._left_csv_path, self._right_csv_path):
            base = os.path.dirname(os.path.abspath(str(path or "").strip())) if str(path or "").strip() else ""
            if base and os.path.isdir(base) and base not in roots:
                roots.append(base)
        return roots

    def _precomputed_compare_csv_path(self) -> str:
        target_names = (
            "residue_combination_compare.csv",
            "residue_combination_comparison.csv",
            "residue_combination_compare_statistics.csv",
            "residue_combination_delta.csv",
            "residue_combination_diff.csv",
            "residue_combination_difference.csv",
            "combination_compare.csv",
            "combination_comparison.csv",
        )
        for root in self._precomputed_compare_search_roots():
            lower_map = {}
            try:
                lower_map = {name.lower(): name for name in os.listdir(root)}
            except OSError:
                continue
            for name in target_names:
                actual = lower_map.get(name.lower())
                if actual:
                    return os.path.abspath(os.path.join(root, actual))
        return ""

    def _parse_precomputed_compare_row(self, record: dict) -> dict | None:
        residue_ids = _first_present(
            record,
            (
                "Residue_IDs",
                "residue_ids",
                "combination",
                "Combination",
                "combination_label",
                "Combination_label",
            ),
        )
        residue_ids = tuple(
            int(item)
            for item in str(residue_ids or "").replace(",", ";").split(";")
            if str(item).strip() and int(float(str(item).strip())) > 0
        )
        if len(residue_ids) < 2:
            return None
        residue_ids = tuple(sorted(residue_ids))
        residue_names_raw = _first_present(record, ("Residue_names", "residue_names", "names"), "")
        residue_names = tuple(item.strip() for item in str(residue_names_raw or "").split(";") if item.strip())

        left_paths = _safe_float_value(
            _first_present(
                record,
                (
                    "A_affected_path_count",
                    "File_A_affected_path_count",
                    "left_affected_path_count",
                    "base_count",
                    "A Count",
                    "A_count",
                    "left_count",
                ),
            )
        )
        right_paths = _safe_float_value(
            _first_present(
                record,
                (
                    "B_affected_path_count",
                    "File_B_affected_path_count",
                    "right_affected_path_count",
                    "compare_count",
                    "B Count",
                    "B_count",
                    "right_count",
                ),
            )
        )
        path_delta = _safe_float_value(
            _first_present(
                record,
                (
                    "delta_affected_path_count",
                    "affected_path_count_delta",
                    "Delta Paths",
                    "path_count_delta",
                    "delta_count",
                ),
                right_paths - left_paths,
            ),
            right_paths - left_paths,
        )
        left_fraction = _safe_float_value(
            _first_present(record, ("A_affected_path_fraction", "left_path_fraction", "base_percent", "A %", "A_percent"), 0.0)
        )
        right_fraction = _safe_float_value(
            _first_present(record, ("B_affected_path_fraction", "right_path_fraction", "compare_percent", "B %", "B_percent"), 0.0)
        )
        left_total_paths = self._scope_total_paths_for_dataset(self._left_dataset_key)
        right_total_paths = self._scope_total_paths_for_dataset(self._right_dataset_key)
        if abs(left_fraction) > 1.0:
            left_fraction /= 100.0
        if abs(right_fraction) > 1.0:
            right_fraction /= 100.0
        if left_total_paths > 0:
            left_fraction = float(left_paths) / float(left_total_paths)
        if right_total_paths > 0:
            right_fraction = float(right_paths) / float(right_total_paths)
        fraction_delta = right_fraction - left_fraction

        left_radius = _safe_float_value(
            _first_present(
                record,
                ("A_affected_point_radius_avg", "File_A_mean_radius", "left_mean_radius", "A Mean Radius", "A_radius"),
            )
        )
        right_radius = _safe_float_value(
            _first_present(
                record,
                ("B_affected_point_radius_avg", "File_B_mean_radius", "right_mean_radius", "B Mean Radius", "B_radius"),
            )
        )
        radius_delta = _safe_float_value(
            _first_present(
                record,
                ("delta_affected_point_radius_avg", "mean_radius_delta", "Delta Mean Radius", "radius_delta"),
                right_radius - left_radius,
            ),
            right_radius - left_radius,
        )

        present_left = _safe_int_value(_first_present(record, ("present_left", "A_present", "left_present"), 1 if left_paths else 0)) > 0
        present_right = _safe_int_value(_first_present(record, ("present_right", "B_present", "right_present"), 1 if right_paths else 0)) > 0
        if not present_left and left_paths:
            present_left = True
        if not present_right and right_paths:
            present_right = True

        distance_metric = self._empty_frame_distance_metric()
        distance_metric_keys: set[str] = set()
        for side, prefix_names in (
            ("left", ("A", "File_A", "left")),
            ("right", ("B", "File_B", "right")),
        ):
            for state in ("present", "absent"):
                value = _first_present(
                    record,
                    tuple(
                        f"{prefix}_{state}_mean"
                        for prefix in prefix_names
                    ) + tuple(
                        f"{prefix.title()} {state.title()} Mean"
                        for prefix in prefix_names
                    ),
                    None,
                )
                if value is not None and str(value).strip():
                    numeric = _safe_float_value(value)
                    distance_metric[f"{side}_{state}"] = numeric
                    distance_metric[f"{side}_{state}_mean"] = numeric
                    distance_metric_keys.add(f"{side}_{state}")
                n_value = _first_present(
                    record,
                    tuple(f"{prefix}_{state}_n" for prefix in prefix_names),
                    "",
                )
                if str(n_value).strip():
                    distance_metric[f"{side}_{state}_n"] = _safe_int_value(n_value)
            distance_metric[f"{side}_status"] = "precomputed_compare_csv"

        metrics = {
            "affected_path_count": {
                "left": left_paths,
                "right": right_paths,
                "delta": path_delta,
                "abs_delta": abs(path_delta),
            },
            "affected_path_fraction": {
                "left": left_fraction,
                "right": right_fraction,
                "delta": fraction_delta,
                "abs_delta": abs(fraction_delta),
                "left_total_paths": float(left_total_paths),
                "right_total_paths": float(right_total_paths),
            },
            "affected_point_radius_avg": {
                "left": left_radius,
                "right": right_radius,
                "delta": radius_delta,
                "abs_delta": abs(radius_delta),
            },
            "weighted_radius_score": {
                "left": left_radius * left_paths,
                "right": right_radius * right_paths,
                "delta": radius_delta,
                "abs_delta": abs(radius_delta),
                "left_path_count": left_paths,
                "right_path_count": right_paths,
            },
            "residue_pair_distance_sum": distance_metric,
        }
        return {
            "residue_ids": residue_ids,
            "residue_names": residue_names,
            "combination_size": _safe_int_value(
                _first_present(record, ("Combination_size", "combination_size", "size"), len(residue_ids)),
                len(residue_ids),
            ),
            "combination_label": ";".join(str(item) for item in residue_ids),
            "present_left": present_left,
            "present_right": present_right,
            "is_exact_match": False,
            "metrics": metrics,
            "changed": max(abs(path_delta), abs(fraction_delta), abs(radius_delta)) > 1e-12,
            "max_abs_delta": max(abs(path_delta), abs(fraction_delta), abs(radius_delta)),
            "_precomputed_distance_metric": all(
                key in distance_metric_keys
                for key in ("left_present", "left_absent", "right_present", "right_absent")
            ),
        }

    def _load_precomputed_compare_rows(self) -> tuple[list[dict], str] | None:
        csv_path = self._precomputed_compare_csv_path()
        file_key = self._file_cache_key(csv_path)
        if file_key is None:
            return None
        cache_key = (
            "precomputed_compare",
            self._current_compare_dataset_signature(),
            file_key,
        )
        cached = self._precomputed_compare_csv_cache.get(cache_key)
        if cached is not None:
            self._precomputed_compare_csv_cache.move_to_end(cache_key)
            return (copy.deepcopy(cached[0]), cached[1])
        rows: list[dict] = []
        try:
            with open(file_key[0], "r", encoding="utf-8-sig", newline="", errors="ignore") as handle:
                reader = csv.DictReader(handle)
                for record in reader:
                    try:
                        row = self._parse_precomputed_compare_row(record)
                    except Exception:
                        row = None
                    if row is not None:
                        rows.append(row)
        except OSError:
            return None
        if not rows:
            return None
        result = (rows, file_key[0])
        self._precomputed_compare_csv_cache[cache_key] = (copy.deepcopy(rows), file_key[0])
        self._precomputed_compare_csv_cache.move_to_end(cache_key)
        while len(self._precomputed_compare_csv_cache) > 8:
            self._precomputed_compare_csv_cache.popitem(last=False)
        return result

    def _dynamic_compare_rows_cached(self) -> list[dict]:
        left_key = self._file_cache_key(self._left_csv_path)
        right_key = self._file_cache_key(self._right_csv_path)
        cache_key = (
            "dynamic_compare",
            self._current_compare_dataset_signature(),
            left_key,
            right_key,
        )
        cached = self._dynamic_compare_rows_cache.get(cache_key)
        if cached is not None:
            self._dynamic_compare_rows_cache.move_to_end(cache_key)
            return copy.deepcopy(cached)
        rows = [dict(row) for row in compare_combination_statistics(self._left_data, self._right_data)]
        self._dynamic_compare_rows_cache[cache_key] = copy.deepcopy(rows)
        self._dynamic_compare_rows_cache.move_to_end(cache_key)
        while len(self._dynamic_compare_rows_cache) > 8:
            self._dynamic_compare_rows_cache.popitem(last=False)
        return rows

    def _selected_residue_ids(self) -> list[int]:
        residue_ids = []
        seen = set()
        for combo in self._residue_combos:
            residue_id = int(combo.currentData() or 0)
            if residue_id <= 0 or residue_id in seen:
                continue
            residue_ids.append(residue_id)
            seen.add(residue_id)
        return residue_ids

    def _sync_filter_button_state(self, selected: list[int]):
        compare_mode = bool(self._left_data and self._right_data)
        enabled = (1 <= len(selected) <= 4) if compare_mode else (2 <= len(selected) <= 4)
        self._filter_btn.setEnabled(enabled)
        if not enabled and self._filter_btn.isChecked():
            self._filter_btn.blockSignals(True)
            self._filter_btn.setChecked(False)
            self._filter_btn.blockSignals(False)
            self._selection_filter_active = False

    def _on_filter_toggled(self, checked: bool):
        self._selection_filter_active = bool(checked)
        self._refresh_view()

    def _on_current_path_filter_toggled(self, checked: bool):
        self._current_path_filter_active = bool(checked)
        if checked and not self._current_path_combination_keys:
            self.manual_update_requested.emit("path")
            return
        self._refresh_view()

    def _on_current_bottleneck_filter_toggled(self, checked: bool):
        self._current_bottleneck_filter_active = bool(checked)
        if checked and not self._current_bottleneck_combination_keys:
            self.manual_update_requested.emit("path")
            return
        self._refresh_view()

    def _on_visualization_controls_changed(self):
        self.visualization_changed.emit()

    def _sync_contained_match_state(self):
        compare_mode = bool(self._left_data and self._right_data)
        enabled = compare_mode and self._contained_match_dataset_combo.count() >= 2
        controls_enabled = enabled and not self._contained_match_busy
        self._contained_match_dataset_combo.setEnabled(controls_enabled)
        single_path_mode = self._contained_match_single_path_toggle.isChecked()
        self._contained_match_keep_ratio_edit.setEnabled(controls_enabled and not single_path_mode)
        self._contained_match_single_path_toggle.setEnabled(controls_enabled)
        self._contained_match_single_path_row.setVisible(single_path_mode)
        for spin in (
            self._contained_match_jaccard_spin,
            self._contained_match_exit_radius_spin,
            self._contained_match_length_radius_spin,
        ):
            spin.setEnabled(controls_enabled and single_path_mode)
        self._contained_match_query_btn.setEnabled(controls_enabled)
        self._contained_match_query_btn.setText(
            "Working..."
            if self._contained_match_busy else
            f"Query ({self._contained_match_candidate_count})"
            if self._contained_match_candidate_count > 0 else
            "Query"
        )
        self._contained_match_compute_btn.setEnabled(controls_enabled)
        self._contained_match_clear_btn.setEnabled(
            controls_enabled and (
                self._contained_match_locked or
                self._contained_match_candidate_count > 0 or
                self._contained_match_result_active
            )
        )
        self._contained_match_compute_btn.setText(
            "Working..."
            if self._contained_match_busy else
            f"Compare ({self._contained_match_locked_count})"
            if self._contained_match_locked else
            "Compare"
        )
        self._fit_button_texts(
            self._contained_match_query_btn,
            self._contained_match_compute_btn,
            self._contained_match_clear_btn,
        )

    def set_contained_match_busy(self, busy: bool):
        self._contained_match_busy = bool(busy)
        self._sync_contained_match_state()

    def _fit_button_texts(self, *buttons: QPushButton):
        for button in buttons:
            width = button.fontMetrics().horizontalAdvance(button.text()) + 28
            button.setMinimumWidth(max(button.minimumWidth(), width))

    def _create_filter_group_widgets(self) -> dict:
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame{border:1px solid #dcdfe6;border-radius:4px;}")
        row = QHBoxLayout(frame)
        row.setContentsMargins(6, 6, 6, 6)
        row.setSpacing(6)

        row.addWidget(QLabel("Property"))
        property_combo = QComboBox()
        for label, key in COMBINATION_PROPERTY_OPTIONS:
            property_combo.addItem(label, key)
        property_combo.currentIndexChanged.connect(self._on_filter_group_changed)
        row.addWidget(property_combo)

        row.addWidget(QLabel("Filter"))
        trend_combo = QComboBox()
        trend_combo.addItem("All", "all")
        trend_combo.addItem("Increase", "increase")
        trend_combo.addItem("Decrease", "decrease")
        trend_combo.addItem("Neutral", "neutral")
        trend_combo.currentIndexChanged.connect(self._refresh_view)
        row.addWidget(trend_combo)

        row.addWidget(QLabel("Threshold"))
        threshold_edit = QLineEdit()
        threshold_edit.setPlaceholderText("|value| >")
        threshold_edit.setMaximumWidth(90)
        threshold_edit.editingFinished.connect(self._refresh_view)
        row.addWidget(threshold_edit)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(lambda: self._remove_filter_group(group))
        row.addWidget(remove_btn)
        row.addStretch()

        group = {
            "frame": frame,
            "property_combo": property_combo,
            "trend_combo": trend_combo,
            "threshold_edit": threshold_edit,
            "remove_btn": remove_btn,
        }
        return group

    def _add_filter_group(self):
        group = self._create_filter_group_widgets()
        self._filter_groups.append(group)
        self._filter_groups_layout.addWidget(group["frame"])
        self._sync_filter_group_state()
        self._refresh_view()

    def _remove_filter_group(self, group: dict):
        if group not in self._filter_groups:
            return
        if len(self._filter_groups) <= 1:
            return
        self._filter_groups.remove(group)
        frame = group["frame"]
        self._filter_groups_layout.removeWidget(frame)
        frame.deleteLater()
        self._sync_filter_group_state()
        self._refresh_view()

    def _sync_filter_group_state(self):
        compare_mode = bool(self._left_data and self._right_data)
        for group in self._filter_groups:
            trend_combo = group["trend_combo"]
            trend_combo.setEnabled(compare_mode)
            if not compare_mode:
                idx = trend_combo.findData("all")
                if idx >= 0 and trend_combo.currentIndex() != idx:
                    trend_combo.blockSignals(True)
                    trend_combo.setCurrentIndex(idx)
                    trend_combo.blockSignals(False)
            group["remove_btn"].setEnabled(len(self._filter_groups) > 1)

    def _on_filter_group_changed(self):
        self._refresh_view()
        self.visualization_changed.emit()

    def _apply_current_selected_residues(self):
        if not (2 <= len(self._current_selected_residue_ids) <= 4):
            self._summary.setText("The current residue selection must contain 2-4 residues before it can be applied here.")
            return
        values = self._current_selected_residue_ids[:4]
        for idx, combo in enumerate(self._residue_combos):
            target = values[idx] if idx < len(values) else 0
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData(target) if combo.findData(target) >= 0 else 0)
            combo.blockSignals(False)
        self._refresh_view()

    def _clear_residue_slots(self):
        for combo in self._residue_combos:
            combo.blockSignals(True)
            combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._filter_btn.blockSignals(True)
        self._filter_btn.setChecked(False)
        self._filter_btn.blockSignals(False)
        self._selection_filter_active = False
        self._refresh_view()

    def _refresh_view(self):
        if self._view_update_batch_depth > 0:
            self._view_refresh_pending = True
            return
        self._view_refresh_pending = False
        self._cancel_deferred_distance_annotation()
        selected = self._selected_residue_ids()
        self._sync_filter_button_state(selected)
        self._sync_filter_group_state()
        self._sync_contained_match_state()
        if not self._left_data and not self._right_data:
            self._visible_rows = []
            self._sync_left_residue_set_button()
            if self._csv_selection_dirty:
                self._summary.setText(
                    "Dataset selection changed. Click Load DS or Load Path to load the selected residue-combination CSV files."
                )
            else:
                self._summary.setText("Please choose at least one dataset with `residue_combination_statistics.csv` first.")
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._sync_save_table_button()
            self._detail.setText("Select a combination to inspect its detailed statistics.")
            self.visualization_changed.emit()
            return
        if self._left_data and self._right_data:
            if self.contained_match_result_active():
                rows_override = (
                    self._contained_match_bottleneck_rows
                    if self._current_bottleneck_filter_active
                    else self._contained_match_rows
                )
                self._populate_compare_rows(
                    selected,
                    rows_override=rows_override,
                    summary_prefix=self._contained_match_summary,
                    rows_are_bottleneck_scoped=(
                        self._current_bottleneck_filter_active
                        and rows_override is self._contained_match_bottleneck_rows
                    ),
                )
            else:
                self._populate_compare_rows(selected)
        else:
            self._populate_single_rows(selected)
        self.visualization_changed.emit()

    def _apply_size_filter(self, rows: list[dict]) -> list[dict]:
        target_size = int(self._size_filter_combo.currentData() or 0)
        if target_size <= 0:
            return rows
        return [
            row for row in rows
            if int(row.get("combination_size", len(row.get("residue_ids", ())))) == target_size
        ]

    def _group_threshold_value(self, group: dict) -> float | None:
        text = group["threshold_edit"].text().strip()
        if not text:
            return None
        try:
            return abs(float(text))
        except ValueError:
            group["threshold_edit"].setText("")
            return None

    def _apply_filter_groups(self, rows: list[dict], *, compare_mode: bool) -> list[dict]:
        filtered = rows
        for group in self._filter_groups:
            property_key = str(group["property_combo"].currentData() or "affected_path_count")
            trend_filter = str(group["trend_combo"].currentData() or "all")
            threshold = self._group_threshold_value(group)
            next_rows = []
            for row in filtered:
                if compare_mode and trend_filter != "all":
                    if combination_property_trend(row, property_key, compare_mode=True) != trend_filter:
                        continue
                value = combination_property_value(row, property_key, compare_mode=compare_mode)
                if threshold is not None and abs(value) <= threshold:
                    continue
                next_rows.append(row)
            filtered = next_rows
        return filtered

    def _apply_current_path_filter(self, rows: list[dict]) -> list[dict]:
        if not self._current_path_filter_active or not self._current_path_combination_keys:
            return rows
        return [
            row for row in rows
            if tuple(int(item) for item in row.get("residue_ids", ())) in self._current_path_combination_keys
        ]

    def _apply_current_bottleneck_filter(
        self,
        rows: list[dict],
        *,
        rows_are_bottleneck_scoped: bool = False,
    ) -> list[dict]:
        return filter_combination_rows_by_keys(
            rows,
            self._current_bottleneck_combination_keys,
            active=self._current_bottleneck_filter_active,
            rows_already_scoped=rows_are_bottleneck_scoped,
        )

    def _populate_single_rows(self, selected: list[int]):
        source_data = self._left_data or self._right_data
        source_key = self._left_dataset_key if self._left_data else self._right_dataset_key
        if self._current_bottleneck_filter_active:
            rows = [dict(row) for row in self._current_bottleneck_rows_by_dataset.get(source_key, [])]
        elif self._loaded_scope_mode == "path":
            rows = [dict(row) for row in self._current_path_rows_by_dataset.get(source_key, [])]
        else:
            rows = [dict(row) for row in all_combination_rows(source_data)]
        self._annotate_selection_matches(rows, selected)
        if self._selection_filter_active:
            rows = [row for row in rows if row.get("selection_match")]
        rows = self._apply_current_path_filter(rows)
        rows = self._apply_current_bottleneck_filter(rows)
        rows = self._apply_size_filter(rows)
        rows = self._apply_filter_groups(rows, compare_mode=False)
        rows = self._sort_single_rows(rows)
        self._visible_rows = [dict(row) for row in rows]
        self._sync_left_residue_set_button()
        related_rows = [row for row in rows if row.get("selection_match")]
        source_name = os.path.basename(source_data.get("path", "") or "") or "Current File"
        if self._loaded_scope_mode == "path":
            source_name = f"{source_name} | Current Selected Paths"
        if len(selected) >= 1:
            if self._selection_filter_active:
                self._summary.setText(
                    f"{source_name} | Showing only the {len(rows)} combinations that contain the current {len(selected)} selected residue(s)."
                )
            else:
                self._summary.setText(
                    f"{source_name} | Showing all {len(rows)} combinations; "
                    f"{len(related_rows)} of them contain the current {len(selected)} selected residue(s)."
                )
        else:
            self._summary.setText(f"{source_name} | Showing all {len(rows)} combination records.")

        self._table.setUpdatesEnabled(False)
        try:
            self._table.clear()
            self._table.setColumnCount(6)
            self._table.setHorizontalHeaderLabels(
                ["Combination", "Size", "Affected Paths", "Mean Radius", "Radius Range", "Match"]
            )
            self._table.setRowCount(len(rows))
            for row_idx, row in enumerate(rows):
                combo_item = QTableWidgetItem(row["combination_label"])
                combo_item.setData(Qt.UserRole, {"mode": "single", "row": row})
                if row.get("selection_match") == "Exact Match":
                    combo_item.setBackground(Qt.yellow)
                elif row.get("selection_match"):
                    combo_item.setBackground(Qt.lightGray)
                self._table.setItem(row_idx, 0, combo_item)
                self._table.setItem(row_idx, 1, QTableWidgetItem(str(row["combination_size"])))
                self._table.setItem(row_idx, 2, QTableWidgetItem(str(row["affected_path_count"])))
                self._table.setItem(row_idx, 3, QTableWidgetItem(_format_float(row["affected_point_radius_avg"])))
                self._table.setItem(row_idx, 4, QTableWidgetItem(_format_float(row["affected_point_radius_range"])))
                self._table.setItem(
                    row_idx,
                    5,
                    QTableWidgetItem("Combination Statistics"),
                )
            self._finish_table(rows)
        finally:
            self._table.setUpdatesEnabled(True)

    def _set_compare_distance_cells(self, row_idx: int, row: dict) -> None:
        distance_metric = row.get("metrics", {}).get("residue_pair_distance_sum", {})
        for col_idx, key, label in (
            (6, "left_present", "File A present"),
            (7, "left_absent", "File A absent"),
            (8, "right_present", "File B present"),
            (9, "right_absent", "File B absent"),
        ):
            value = distance_metric.get(key)
            numeric_value = float(value) if value is not None else 0.0
            item = _SortableTableWidgetItem(
                _format_float(numeric_value) if value is not None else "-",
                sort_value=numeric_value,
            )
            item.setTextAlignment(Qt.AlignCenter)
            side = "right" if key.startswith("right") else "left"
            state = "present" if key.endswith("present") else "absent"
            sample_count = int(distance_metric.get(f"{side}_{state}_n", 0) or 0)
            missing_frames = int(distance_metric.get(f"{side}_missing_frames", 0) or 0)
            status = str(distance_metric.get(f"{side}_status", "") or "")
            item.setToolTip(
                f"{label} distance mean: "
                f"{_format_float(numeric_value) if value is not None else 'n/a'}\n"
                f"Frame samples: {sample_count}\n"
                f"Skipped frames: {missing_frames}\n"
                f"Status: {status or 'ok'}"
            )
            self._table.setItem(row_idx, col_idx, item)

    def _populate_compare_rows(
        self,
        selected: list[int],
        rows_override: list[dict] | None = None,
        summary_prefix: str | None = None,
        *,
        rows_are_bottleneck_scoped: bool = False,
    ):
        used_precomputed_compare = False
        precomputed_source = ""
        if rows_override is not None:
            rows = [dict(row) for row in rows_override]
        elif self._current_bottleneck_filter_active:
            left_rows = self._current_bottleneck_rows_by_dataset.get(self._left_dataset_key, [])
            right_rows = self._current_bottleneck_rows_by_dataset.get(self._right_dataset_key, [])
            rows = compare_combination_row_sets(
                left_rows,
                right_rows,
                left_total_paths=self._scope_total_paths_for_dataset(self._left_dataset_key, bottleneck=True),
                right_total_paths=self._scope_total_paths_for_dataset(self._right_dataset_key, bottleneck=True),
            )
        elif self._loaded_scope_mode == "path":
            left_rows = self._current_path_rows_by_dataset.get(self._left_dataset_key, [])
            right_rows = self._current_path_rows_by_dataset.get(self._right_dataset_key, [])
            rows = compare_combination_row_sets(
                left_rows,
                right_rows,
                left_total_paths=self._scope_total_paths_for_dataset(self._left_dataset_key),
                right_total_paths=self._scope_total_paths_for_dataset(self._right_dataset_key),
            )
        else:
            precomputed = self._load_precomputed_compare_rows()
            if precomputed is not None:
                precomputed_rows, precomputed_source = precomputed
                rows = [dict(row) for row in precomputed_rows]
                used_precomputed_compare = True
            else:
                rows = self._dynamic_compare_rows_cached()
        self._annotate_selection_matches(rows, selected)
        if self._selection_filter_active:
            rows = [row for row in rows if row.get("selection_match")]
        rows = self._apply_current_path_filter(rows)
        rows = self._apply_current_bottleneck_filter(
            rows,
            rows_are_bottleneck_scoped=rows_are_bottleneck_scoped,
        )
        rows = self._apply_size_filter(rows)
        rows = self._apply_filter_groups(rows, compare_mode=True)
        rows = self._sort_compare_rows(rows)
        defer_distance_annotation = False
        should_annotate_distances = not used_precomputed_compare or not all(
            row.get("_precomputed_distance_metric") for row in rows
        )
        if (
            should_annotate_distances
            and len(rows) > MAX_BULK_DISTANCE_ANNOTATION_ROWS
            and not self._bulk_presence_stats_available()
        ):
            self._set_empty_compare_distance_sums(rows)
            defer_distance_annotation = self._observer_frame_distances_ready()
        elif should_annotate_distances:
            self._annotate_compare_distance_sums(rows)
        self._visible_rows = [dict(row) for row in rows]
        self._sync_left_residue_set_button()
        changed_rows = [row for row in rows if row.get("changed")]
        related_rows = [row for row in rows if row.get("selection_match")]
        if summary_prefix:
            summary_head = summary_prefix.strip()
        elif self._loaded_scope_mode == "path":
            summary_head = "Current selected paths"
        elif used_precomputed_compare:
            summary_head = f"Precomputed comparison | {os.path.basename(precomputed_source)}"
        else:
            summary_head = "File A / File B"
        if len(selected) >= 1:
            if self._selection_filter_active:
                self._summary.setText(
                    f"{summary_head} showing only the {len(rows)} combinations containing the current {len(selected)} selected residue(s), "
                    f"with differences in {len(changed_rows)} of them."
                )
            else:
                self._summary.setText(
                    f"{summary_head} showing {len(rows)} combinations total, "
                    f"with differences in {len(changed_rows)} and {len(related_rows)} containing the current {len(selected)} selected residue(s)."
                )
        else:
            self._summary.setText(
                f"{summary_head} showing {len(rows)} combinations total, with differences in {len(changed_rows)} of them."
            )

        self._table.setUpdatesEnabled(False)
        self._table.clear()
        was_sorting = self._table.isSortingEnabled()
        self._table.setSortingEnabled(False)
        try:
            self._table.setColumnCount(10)
            self._table.setHorizontalHeaderLabels(
                [
                    "Combination",
                    "Status",
                    "Δ Paths",
                    "Δ Path %",
                    "A Mean Radius",
                    "B Mean Radius",
                    "A Present Mean",
                    "A Absent Mean",
                    "B Present Mean",
                    "B Absent Mean",
                ]
            )
            self._table.setRowCount(len(rows))
            metric_deltas = {
                "paths": max((abs(float(row["metrics"]["affected_path_count"]["delta"])) for row in rows), default=0.0),
                "path_fraction": max((abs(float(row["metrics"]["affected_path_fraction"]["delta"])) for row in rows), default=0.0),
            }

            for row_idx, row in enumerate(rows):
                combo_sort_key = tuple(int(rid) for rid in row.get("residue_ids", ()))
                combo_item = _SortableTableWidgetItem(row["combination_label"], sort_value=(len(combo_sort_key), combo_sort_key))
                combo_item.setData(Qt.UserRole, {"mode": "compare", "row": row})
                combo_item.setToolTip(row["combination_label"])
                if row.get("selection_match") == "Exact Match":
                    combo_item.setBackground(Qt.yellow)
                elif row.get("selection_match"):
                    combo_item.setBackground(Qt.lightGray)
                self._table.setItem(row_idx, 0, combo_item)

                if row["present_left"] and row["present_right"]:
                    status_text = "Both"
                    status_sort = 2
                elif row["present_left"]:
                    status_text = "Only A"
                    status_sort = 1
                else:
                    status_text = "Only B"
                    status_sort = 0
                status_item = _SortableTableWidgetItem(status_text, sort_value=status_sort)
                status_item.setTextAlignment(Qt.AlignCenter)
                status_item.setToolTip(
                    "Present in both File A and File B."
                    if status_text == "Both" else
                    ("Present only in File A." if status_text == "Only A" else "Present only in File B.")
                )
                self._table.setItem(row_idx, 1, status_item)

                path_metric = row["metrics"]["affected_path_count"]
                path_fraction_metric = row["metrics"]["affected_path_fraction"]
                avg_metric = row["metrics"]["affected_point_radius_avg"]
                for col_idx, value, max_abs, tip in (
                    (
                        2,
                        float(path_metric["delta"]),
                        metric_deltas["paths"],
                        f"File A: {_format_float(path_metric['left'])}\nFile B: {_format_float(path_metric['right'])}\nDelta (B-A): {_format_delta(float(path_metric['delta']))}",
                    ),
                    (
                        3,
                        float(path_fraction_metric["delta"]) * 100.0,
                        metric_deltas["path_fraction"] * 100.0,
                        (
                            f"File A: {float(path_fraction_metric['left']) * 100.0:.2f}% "
                            f"({int(float(path_fraction_metric.get('left_total_paths', 0.0) or 0.0))} total paths)\n"
                            f"File B: {float(path_fraction_metric['right']) * 100.0:.2f}% "
                            f"({int(float(path_fraction_metric.get('right_total_paths', 0.0) or 0.0))} total paths)\n"
                            f"Delta (B-A): {(float(path_fraction_metric['delta']) * 100.0):+.2f}%"
                        ),
                    ),
                ):
                    item = _SortableTableWidgetItem(_format_delta(value), sort_value=value)
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setBackground(QBrush(_delta_color(value, max_abs)))
                    if max_abs > 0 and abs(value) >= (max_abs * 0.72):
                        item.setForeground(QBrush(QColor("#FFFFFF")))
                    item.setToolTip(tip)
                    self._table.setItem(row_idx, col_idx, item)

                avg_a_item = _SortableTableWidgetItem(_format_float(float(avg_metric["left"])), sort_value=float(avg_metric["left"]))
                avg_a_item.setTextAlignment(Qt.AlignCenter)
                avg_a_item.setToolTip(f"File A mean radius: {_format_float(avg_metric['left'])}")
                self._table.setItem(row_idx, 4, avg_a_item)

                avg_b_item = _SortableTableWidgetItem(_format_float(float(avg_metric["right"])), sort_value=float(avg_metric["right"]))
                avg_b_item.setTextAlignment(Qt.AlignCenter)
                avg_b_item.setToolTip(f"File B mean radius: {_format_float(avg_metric['right'])}")
                self._table.setItem(row_idx, 5, avg_b_item)

                self._set_compare_distance_cells(row_idx, row)
            self._refresh_compare_presence_widgets()
            self._table.setSortingEnabled(was_sorting)
            self._finish_table(rows)
        finally:
            self._table.setSortingEnabled(was_sorting)
            self._table.setUpdatesEnabled(True)
        if defer_distance_annotation:
            self._schedule_deferred_distance_annotation()

    def _finish_table(self, rows: list[dict]):
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Fixed)
        if self._table.columnCount() > 0:
            header.setSectionResizeMode(0, QHeaderView.Fixed)
            combo_width = int(186 * 3 / 5)
            self._table.setColumnWidth(0, combo_width)
        for col_idx in range(1, self._table.columnCount()):
            header.setSectionResizeMode(col_idx, QHeaderView.Fixed)
            header_item = self._table.horizontalHeaderItem(col_idx)
            label_text = header_item.text() if header_item is not None else ""
            label_width = self._table.fontMetrics().horizontalAdvance(label_text)
            if col_idx == 1:
                extra_width = 68
            elif col_idx == 7:
                extra_width = 52
            elif col_idx == 2:
                extra_width = 33
            else:
                extra_width = 23
            self._table.setColumnWidth(col_idx, max(43, label_width + extra_width))
        if rows:
            self._table.selectRow(0)
        else:
            self._detail.setText("No valid combinations were found for the selected residues in the current file or across the two files.")
        self._sync_save_table_button()

    def _sync_save_table_button(self):
        if not hasattr(self, "_save_table_btn"):
            return
        self._save_table_btn.setEnabled(
            self._table.rowCount() > 0 and self._table.columnCount() > 0
        )

    def _table_header_texts(self) -> list[str]:
        headers: list[str] = []
        for col_idx in range(self._table.columnCount()):
            header_item = self._table.horizontalHeaderItem(col_idx)
            headers.append(header_item.text() if header_item is not None else f"Column {col_idx + 1}")
        return headers

    def _table_cell_text(self, row_idx: int, col_idx: int) -> str:
        item = self._table.item(row_idx, col_idx)
        if item is not None:
            return item.text()
        widget = self._table.cellWidget(row_idx, col_idx)
        if widget is None:
            return ""
        label = widget.findChild(QLabel)
        return label.text() if label is not None else ""

    def _save_current_table(self):
        if self._table.rowCount() <= 0 or self._table.columnCount() <= 0:
            QMessageBox.information(self, "Save Table", "There is no table data to save.")
            self._sync_save_table_button()
            return

        base_dir = (
            os.path.dirname(self._left_csv_path)
            or os.path.dirname(self._right_csv_path)
            or os.getcwd()
        )
        default_name = (
            "residue_combination_compare_table.csv"
            if self.is_compare_mode()
            else "residue_combination_table.csv"
        )
        default_path = os.path.join(base_dir, default_name)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Residue Combination Table",
            default_path,
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".csv"

        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(self._table_header_texts())
                for row_idx in range(self._table.rowCount()):
                    writer.writerow(
                        [
                            self._table_cell_text(row_idx, col_idx)
                            for col_idx in range(self._table.columnCount())
                        ]
                    )
        except OSError as exc:
            QMessageBox.warning(self, "Save Table", f"Failed to save table:\n{exc}")
            return

        self._summary.setText(f"Saved current table to {path}")

    def _presence_indicator_widget(self, label: str, active: bool = False) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        indicator = QLabel(label)
        indicator.setAlignment(Qt.AlignCenter)
        indicator.setStyleSheet(
            "background: #DCFCE7; border: 1px solid #22C55E; color: #166534; border-radius: 4px; "
            "font-size: 11px; font-weight: 700; padding: 2px 8px;"
            if active else
            "background: #FFFFFF; border: 1px solid #D1D5DB; color: #374151; border-radius: 4px; "
            "font-size: 11px; font-weight: 600; padding: 2px 8px;"
        )
        layout.addWidget(indicator)
        return widget

    def _refresh_compare_presence_widgets(self):
        if self._table.columnCount() < 2:
            return
        for row_idx in range(self._table.rowCount()):
            row_item = self._table.item(row_idx, 0)
            payload = row_item.data(Qt.UserRole) if row_item is not None else None
            row_data = payload.get("row", {}) if isinstance(payload, dict) else {}
            residue_ids = tuple(int(rid) for rid in row_data.get("residue_ids", ()) if int(rid) > 0)
            marker_key = "combo_" + "_".join(str(rid) for rid in residue_ids) if len(residue_ids) >= 2 else ""
            if row_data.get("present_left") and row_data.get("present_right"):
                status_text = "Both"
            elif row_data.get("present_left"):
                status_text = "Only A"
            else:
                status_text = "Only B"
            active = marker_key == self._selected_compare_marker_key
            self._table.removeCellWidget(row_idx, 1)
            status_item = self._table.item(row_idx, 1)
            if status_item is None:
                status_item = _SortableTableWidgetItem(status_text)
                status_item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(row_idx, 1, status_item)
            status_item.setText(status_text)
            status_item.setBackground(QBrush(QColor("#DCFCE7" if active else "#FFFFFF")))
            status_item.setForeground(QBrush(QColor("#166534" if active else "#374151")))

    def _handle_cell_clicked(self, row: int, col: int):
        row_item = self._table.item(row, 0)
        payload = row_item.data(Qt.UserRole) if row_item is not None else None
        row_data = payload.get("row", {}) if isinstance(payload, dict) else {}
        if payload and payload.get("mode") == "compare" and col == 1:
            present = bool(row_data.get("present_left")) or bool(row_data.get("present_right"))
            residue_ids = tuple(int(rid) for rid in row_data.get("residue_ids", ()) if int(rid) > 0)
            if present and len(residue_ids) >= 2:
                self._selected_compare_marker_key = "combo_" + "_".join(str(rid) for rid in residue_ids)
                self._refresh_compare_presence_widgets()
                self.combination_selection_requested.emit(
                    {
                        "marker_key": self._selected_compare_marker_key,
                        "residue_ids": residue_ids,
                        "residue_names": tuple(row_data.get("residue_names", ())),
                        "combination_label": row_data.get("combination_label", ""),
                        "present_left": bool(row_data.get("present_left")),
                        "present_right": bool(row_data.get("present_right")),
                        "left_dataset_key": self._left_dataset_key,
                        "right_dataset_key": self._right_dataset_key,
                    }
                )
        self._update_detail()

    def _annotate_selection_matches(self, rows: list[dict], selected: list[int]):
        selected_tuple = tuple(int(item) for item in selected if int(item) > 0)
        selected_set = set(selected_tuple)
        for row in rows:
            row["selection_match"] = ""
            row_ids = tuple(int(item) for item in row.get("residue_ids", ()))
            if len(selected_tuple) < 1:
                continue
            if row_ids == selected_tuple:
                row["selection_match"] = "Exact Match"
            elif selected_set.issubset(set(row_ids)):
                row["selection_match"] = "Contains Selected Residues"

    def _sort_single_rows(self, rows: list[dict]) -> list[dict]:
        mode = str(self._sort_combo.currentData() or "path")
        if mode == "radius":
            return sorted(rows, key=lambda row: (-float(row.get("affected_point_radius_range", 0.0)), tuple(row.get("residue_ids", ()))))
        if mode == "size":
            return sorted(rows, key=lambda row: (int(row.get("combination_size", 0)), tuple(row.get("residue_ids", ()))))
        if mode == "label":
            return sorted(rows, key=lambda row: (str(row.get("combination_label", "")), tuple(row.get("residue_ids", ()))))
        return sorted(rows, key=lambda row: (-int(row.get("affected_path_count", 0)), tuple(row.get("residue_ids", ()))))

    def _sort_compare_rows(self, rows: list[dict]) -> list[dict]:
        mode = str(self._sort_combo.currentData() or "path")
        if mode == "radius":
            return sorted(
                rows,
                key=lambda row: (
                    -float(row["metrics"]["affected_point_radius_avg"]["abs_delta"]),
                    tuple(row.get("residue_ids", ())),
                ),
            )
        if mode == "size":
            return sorted(rows, key=lambda row: (int(row.get("combination_size", 0)), tuple(row.get("residue_ids", ()))))
        if mode == "label":
            return sorted(rows, key=lambda row: (str(row.get("combination_label", "")), tuple(row.get("residue_ids", ()))))
        return sorted(
            rows,
            key=lambda row: (
                -float(row["metrics"]["affected_path_count"]["abs_delta"]),
                -float(row["metrics"]["affected_path_fraction"]["abs_delta"]),
                tuple(row.get("residue_ids", ())),
            ),
        )

    def visible_rows(self) -> list[dict]:
        return [dict(row) for row in self._visible_rows]

    def _visible_residue_ids_for_side(self, side: str) -> tuple[int, ...]:
        residue_ids: set[int] = set()
        compare_mode = self.is_compare_mode()
        side = "right" if str(side).strip().lower() in {"right", "b", "bottom"} else "left"
        for row in self._visible_rows:
            if compare_mode and not bool(row.get("present_right" if side == "right" else "present_left")):
                continue
            for residue_id in row.get("residue_ids", ()):
                try:
                    normalized = int(residue_id)
                except (TypeError, ValueError):
                    continue
                if normalized > 0:
                    residue_ids.add(normalized)
        return tuple(sorted(residue_ids))

    def _sync_left_residue_set_button(self) -> None:
        button = getattr(self, "_show_left_residues_btn", None)
        if button is None:
            return
        left_residue_ids = self._visible_residue_ids_for_side("left")
        right_residue_ids = self._visible_residue_ids_for_side("right")
        left_dataset_key = self.left_dataset_key()
        right_dataset_key = self.right_dataset_key()
        button.setEnabled(
            bool(
                (left_dataset_key and len(left_residue_ids) >= 2)
                or (right_dataset_key and len(right_residue_ids) >= 2)
            )
        )

    def _emit_left_residue_set_request(self) -> None:
        left_residue_ids = self._visible_residue_ids_for_side("left")
        right_residue_ids = self._visible_residue_ids_for_side("right")
        left_dataset_key = self.left_dataset_key()
        right_dataset_key = self.right_dataset_key()
        if (
            (not left_dataset_key or len(left_residue_ids) < 2)
            and (not right_dataset_key or len(right_residue_ids) < 2)
        ):
            self._detail.setText("No visible File A / File B residue sets are available to show.")
            return
        self.left_residue_set_requested.emit(
            {
                "mode": "visible_residue_sets",
                "left_residue_ids": left_residue_ids,
                "right_residue_ids": right_residue_ids,
                "residue_ids": tuple(sorted(set(left_residue_ids).union(right_residue_ids))),
                "combination_label": (
                    f"Visible residues | A={len(left_residue_ids)} | B={len(right_residue_ids)}"
                ),
                "present_left": bool(left_dataset_key and len(left_residue_ids) >= 2),
                "present_right": bool(right_dataset_key and len(right_residue_ids) >= 2),
                "left_dataset_key": left_dataset_key,
                "right_dataset_key": right_dataset_key,
            }
        )

    def selected_property_key(self) -> str:
        return str(self._color_property_combo.currentData() or "affected_path_count")

    def selected_property_label(self) -> str:
        return self._color_property_combo.currentText().strip() or "Affected Paths"

    def combination_markers_enabled(self) -> bool:
        return self._show_markers_btn.isChecked()

    def is_compare_mode(self) -> bool:
        return bool(self._left_data and self._right_data)

    def current_path_filter_active(self) -> bool:
        return bool(self._current_path_filter_active)

    def current_bottleneck_filter_active(self) -> bool:
        return bool(self._current_bottleneck_filter_active)

    def loaded_scope_mode(self) -> str:
        return str(self._loaded_scope_mode or "dataset")

    def contained_match_result_active(self) -> bool:
        return bool(self._contained_match_result_active)

    def contained_match_source_dataset_key(self) -> str:
        return str(self._contained_match_dataset_combo.currentData() or "")

    def active_dataset_keys(self) -> list[str]:
        keys: list[str] = []
        for key in (self._left_dataset_key, self._right_dataset_key):
            normalized = str(key or "").strip()
            if normalized and normalized not in keys:
                keys.append(normalized)
        return keys

    def left_dataset_key(self) -> str:
        return str(self._left_dataset_key or "").strip()

    def right_dataset_key(self) -> str:
        return str(self._right_dataset_key or "").strip()

    def contained_match_keep_ratio(self) -> float:
        text = self._contained_match_keep_ratio_edit.text().strip()
        try:
            value = float(text)
        except ValueError:
            value = 1.0
        value = min(1.0, max(0.05, value))
        normalized = f"{value:.2f}".rstrip("0").rstrip(".")
        self._contained_match_keep_ratio_edit.setText(normalized or "1")
        return value

    def contained_match_single_path_mode(self) -> bool:
        return self._contained_match_single_path_toggle.isChecked()

    def contained_match_single_path_parameters(self) -> dict:
        return {
            "jaccard_threshold": float(self._contained_match_jaccard_spin.value()),
            "exit_radius": float(self._contained_match_exit_radius_spin.value()),
            "length_radius": float(self._contained_match_length_radius_spin.value()),
        }

    def selected_size_filter(self) -> int:
        return int(self._size_filter_combo.currentData() or 0)

    def _distance_detail_for_side(self, distance_metric: dict, side: str, label: str) -> str:
        present_mean = distance_metric.get(f"{side}_present_mean")
        absent_mean = distance_metric.get(f"{side}_absent_mean")
        present_n = int(distance_metric.get(f"{side}_present_n", 0) or 0)
        absent_n = int(distance_metric.get(f"{side}_absent_n", 0) or 0)
        return (
            f"{label} present mean={_format_float(float(present_mean)) if present_mean is not None else '-'} "
            f"(n={present_n}), "
            f"absent mean={_format_float(float(absent_mean)) if absent_mean is not None else '-'} "
            f"(n={absent_n})"
        )

    def _update_detail(self):
        selected_items = self._table.selectedItems()
        if not selected_items:
            self._detail.setText("Select a combination to inspect its detailed statistics.")
            return
        row_item = self._table.item(selected_items[0].row(), 0)
        if row_item is None:
            self._detail.setText("Select a combination to inspect its detailed statistics.")
            return
        payload = row_item.data(Qt.UserRole) or {}
        mode = payload.get("mode", "single")
        row = payload.get("row", {})
        if mode == "compare":
            path_metric = row["metrics"]["affected_path_count"]
            path_fraction_metric = row["metrics"]["affected_path_fraction"]
            avg_metric = row["metrics"]["affected_point_radius_avg"]
            distance_metric = row.get("metrics", {}).get("residue_pair_distance_sum", {})
            self._detail.setText(
                f"{row.get('combination_label', 'Combination')} | "
                f"Paths A={_format_float(path_metric['left'])}, B={_format_float(path_metric['right'])}, Δ={path_metric['delta']:+.0f} | "
                f"Path % A={float(path_fraction_metric['left']) * 100.0:.2f}%, "
                f"B={float(path_fraction_metric['right']) * 100.0:.2f}%, "
                f"Δ={(float(path_fraction_metric['delta']) * 100.0):+.2f}% | "
                f"Mean Radius A={avg_metric['left']:.3f}, B={avg_metric['right']:.3f} | "
                f"{self._distance_detail_for_side(distance_metric, 'left', 'A')} | "
                f"{self._distance_detail_for_side(distance_metric, 'right', 'B')}"
            )
            return

        self._detail.setText(
            f"{row.get('combination_label', 'Combination')} | "
            f"Affected Paths {row.get('affected_path_count', 0)} | "
            f"Mean Radius {_format_float(float(row.get('affected_point_radius_avg', 0.0)))} | "
            f"Min {_format_float(float(row.get('affected_point_radius_min', 0.0)))} | "
            f"Max {_format_float(float(row.get('affected_point_radius_max', 0.0)))} | "
            f"Range {_format_float(float(row.get('affected_point_radius_range', 0.0)))}"
        )
