"""
Residue statistics CSV comparison panel with heatmap-style delta display.
"""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QColor, QBrush, QPalette
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QMessageBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QDialog,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTextBrowser,
)

from TopoTunnel_UI.core.residue_compare import (
    compare_residue_statistics,
    presence_status_label,
    presence_status_tooltip,
    load_residue_statistics_csv,
    metric_label,
    metric_tooltip,
    order_comparison_rows,
)


def _format_number(value: float) -> str:
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


def _format_ratio(ratio) -> str:
    if ratio is None:
        return "-"
    return f"{ratio * 100:+.1f}%"


def _format_cooperative_change_entry(item: dict) -> str:
    partner_name = item.get("right_name") or item.get("left_name") or "UNK"
    return (
        f"#{int(item.get('residue_id', 0))} {partner_name} "
            f"{int(item.get('delta', 0)):+d} paths "
        f"(A={int(item.get('left_count', 0))}, B={int(item.get('right_count', 0))})"
    )


def _cooperative_change_lines(row: dict) -> list[str]:
    residue_label = row.get("residue_label", "Residue")
    changes = list(row.get("cooperative_partner_changes", []) or [])
    if not changes:
        return [f"{residue_label}: no change in cooperative-residue co-occurrence count."]

    increased = [item for item in changes if int(item.get("delta", 0)) > 0]
    decreased = [item for item in changes if int(item.get("delta", 0)) < 0]
    lines = [
        f"Cooperative-residue path-count changes for {residue_label} (B-A)",
        f"Increased: {len(increased)}, decreased: {len(decreased)}.",
    ]
    if increased:
        lines.append("Increased:")
        lines.extend(_format_cooperative_change_entry(item) for item in increased)
    if decreased:
        lines.append("Decreased:")
        lines.extend(_format_cooperative_change_entry(item) for item in decreased)
    return lines


def _cooperative_change_html(row: dict) -> str:
    residue_label = row.get("residue_label", "Residue")
    changes = list(row.get("cooperative_partner_changes", []) or [])
    if not changes:
        return (
            "<div style='color:#606266;'>"
            f"<b>{residue_label}</b><br>No increased or decreased cooperative residue details."
            "</div>"
        )

    increased = [item for item in changes if int(item.get("delta", 0)) > 0]
    decreased = [item for item in changes if int(item.get("delta", 0)) < 0]

    def _section_html(title: str, accent: str, items: list[dict], empty_text: str) -> str:
        rows = []
        if items:
            for item in items:
                partner_name = item.get("right_name") or item.get("left_name") or "UNK"
                rows.append(
                    "<div style='margin:4px 0; padding:6px 8px; border:1px solid #E5E7EB; border-radius:6px; background:#FFFFFF;'>"
                    f"<b>#{int(item.get('residue_id', 0))} {partner_name}</b><br>"
                    f"Δ {int(item.get('delta', 0)):+d} paths"
                    f" | A={int(item.get('left_count', 0))}"
                    f" | B={int(item.get('right_count', 0))}"
                    "</div>"
                )
        else:
            rows.append(
                "<div style='margin:4px 0; padding:6px 8px; border:1px dashed #D1D5DB; border-radius:6px; color:#909399;'>"
                f"{empty_text}"
                "</div>"
            )
        return (
            "<td valign='top' width='50%' style='padding:0 6px;'>"
            f"<div style='font-weight:700; color:{accent}; margin-bottom:6px;'>{title}</div>"
            + "".join(rows)
            + "</td>"
        )

    return (
        "<div style='font-size:11px; color:#303133;'>"
        f"<div style='margin-bottom:8px;'><b>Residues' cell increased/decrease details</b><br>"
        f"{residue_label}</div>"
        "<table width='100%' cellspacing='0' cellpadding='0'><tr>"
        + _section_html("Increased", "#D14343", increased, "No increased partners")
        + _section_html("Decreased", "#2F6FDD", decreased, "No decreased partners")
        + "</tr></table></div>"
    )


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
    """Paint item-role heatmap backgrounds without losing them on selection."""

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


class ResidueHeatmapDialog(QDialog):
    def __init__(self, panel: "ResidueComparePanel", parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setWindowTitle("Residue Heatmap")
        self.setModal(False)
        self.setMinimumSize(1120, 720)
        self.resize(1280, 860)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("Residue Heatmap")
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

        self._table = panel._create_heatmap_table()
        self._table.itemSelectionChanged.connect(self._update_detail)
        self._table.cellClicked.connect(self._handle_cell_clicked)
        layout.addWidget(self._table, 1)

        legend = QLabel("Legend: blue = File B lower than File A, red = File B higher than File A, white = no change")
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(legend)

        self._detail = QLabel("Select a residue row to inspect its main changes.")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #606266; font-size: 11px;")
        layout.addWidget(self._detail)

        self._cooperative_detail = panel._create_cooperative_detail_box()
        layout.addWidget(self._cooperative_detail)

    def refresh(self, comparison: dict, rows: list[dict], metrics: list[str]):
        if not comparison:
            self._summary.setText("No heatmap data.")
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._detail.setText("Select a residue row to inspect its main changes.")
            self._cooperative_detail.setHtml("")
            return

        left_name = os.path.basename(comparison.get("left_path", "") or "") or "File A"
        right_name = os.path.basename(comparison.get("right_path", "") or "") or "File B"
        self.setWindowTitle(f"Residue Heatmap - {left_name} vs {right_name}")
        self._panel._populate_heatmap_table(
            self._table,
            self._summary,
            self._detail,
            self._cooperative_detail,
            comparison,
            rows,
            metrics,
        )

    def _update_detail(self):
        self._panel._update_detail_for(self._table, self._detail, self._cooperative_detail)

    def _handle_cell_clicked(self, row: int, col: int):
        self._panel._on_table_cell_clicked(self._table, row, col, self._detail, self._cooperative_detail)


class ResidueComparePanel(QWidget):
    residue_selection_requested = Signal(str, int)
    residue_clear_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._left_data = None
        self._right_data = None
        self._comparison = None
        self._heatmap_dialog = None
        self._dataset_choices: list[dict] = []
        self._left_dataset_key = ""
        self._right_dataset_key = ""
        self._selected_residue_keys: set[tuple[str, int]] = set()
        self._csv_cache: dict[str, dict] = {}
        self._comparison_cache: dict[tuple[str, str], dict] = {}
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(0)
        self._reload_timer.timeout.connect(self._reload_files_immediately)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Residue Compare")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        hint = QLabel(
            "Choose two loaded datasets that include `residue_statistics.csv` to compare residue-property changes. "
            "Red means File B is higher; blue means File B is lower."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(hint)

        self._left_combo = self._build_dataset_row(layout, "File A")
        self._right_combo = self._build_dataset_row(layout, "File B")

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self.residue_clear_requested.emit)
        controls.addWidget(self._reload_btn)

        self._popup_btn = QPushButton("Open Heatmap")
        self._popup_btn.setEnabled(False)
        self._popup_btn.clicked.connect(self._open_heatmap_dialog)
        controls.addWidget(self._popup_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self._summary = QLabel("Load two residue-statistics CSV files to show the heatmap summary.")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("color: #606266; font-size: 11px;")
        layout.addWidget(self._summary)

        self._table = self._create_heatmap_table()
        self._table.itemSelectionChanged.connect(self._update_detail)
        self._table.cellClicked.connect(self._handle_cell_clicked)
        layout.addWidget(self._table, 1)

        legend = QLabel("Legend: blue = File B lower than File A, red = File B higher than File A, white = no change")
        legend.setWordWrap(True)
        legend.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(legend)

        self._detail = QLabel("Select a residue row to inspect its main changes.")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("color: #606266; font-size: 11px;")
        layout.addWidget(self._detail)

        self._cooperative_detail = self._create_cooperative_detail_box()
        layout.addWidget(self._cooperative_detail)

    def _create_heatmap_table(self) -> QTableWidget:
        table = QTableWidget(0, 0)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.setSelectionMode(QAbstractItemView.SingleSelection)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setItemDelegate(_HeatmapItemDelegate(table))
        table.setStyleSheet("QTableView::item:hover { background-color: transparent; }")
        table.setSortingEnabled(True)
        return table

    def _create_cooperative_detail_box(self) -> QTextBrowser:
        box = QTextBrowser()
        box.setReadOnly(True)
        box.setOpenExternalLinks(False)
        box.setStyleSheet(
            "QTextBrowser {"
            "border: 1px solid #E5E7EB;"
            "border-radius: 8px;"
            "background: #F9FAFB;"
            "padding: 6px;"
            "font-size: 11px;"
            "}"
        )
        box.setMinimumHeight(150)
        box.setMaximumHeight(220)
        box.setHtml(
            "<div style='color:#909399;'>"
            "Select a row to inspect increased and decreased cooperative residue details."
            "</div>"
        )
        return box

    def _build_dataset_row(self, parent_layout: QVBoxLayout, label_text: str) -> QComboBox:
        row = QHBoxLayout()
        row.setSpacing(6)
        label = QLabel(label_text)
        label.setFixedWidth(40)
        row.addWidget(label)

        combo = QComboBox()
        combo.currentIndexChanged.connect(self._reload_files)
        row.addWidget(combo, 1)

        parent_layout.addLayout(row)
        return combo

    def set_dataset_choices(self, datasets: list[dict]):
        self._dataset_choices = [dict(item) for item in (datasets or [])]
        self._sync_dataset_combos()

    def _sync_dataset_combos(self):
        current_left = self._left_combo.currentData()
        current_right = self._right_combo.currentData()
        options = [
            item for item in self._dataset_choices
            if item.get("residue_statistics_path")
        ]

        for combo, current_value, empty_label in (
            (self._left_combo, current_left, "Choose File A dataset"),
            (self._right_combo, current_right, "Choose File B dataset"),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(empty_label, "")
            for dataset in options:
                label = (
                    f"[{dataset.get('prefix', dataset.get('key', ''))}] "
                    f"{dataset.get('name', dataset.get('key', ''))}"
                )
                combo.addItem(label, dataset.get("residue_statistics_path", ""))
            idx = combo.findData(current_value)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            elif combo.count() > 1:
                combo.setCurrentIndex(1 if combo is self._left_combo else 0)
            combo.blockSignals(False)
        self._schedule_reload()

    def _schedule_reload(self):
        self._reload_timer.start()

    def _reload_files(self):
        self._schedule_reload()

    def _reload_files_immediately(self):
        left_path = str(self._left_combo.currentData() or "").strip()
        right_path = str(self._right_combo.currentData() or "").strip()
        self._left_dataset_key = str(self._left_combo.currentText() and self._dataset_key_for_path(left_path) or "")
        self._right_dataset_key = str(self._right_combo.currentText() and self._dataset_key_for_path(right_path) or "")

        if not left_path or not right_path:
            self._comparison = None
            self._summary.setText("Choose both File A and File B datasets to build the residue heatmap.")
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._detail.setText("Select a residue row to inspect its main changes.")
            self._cooperative_detail.setHtml("")
            self._popup_btn.setEnabled(False)
            self._refresh_heatmap_dialog()
            return

        try:
            self._left_data = self._load_residue_statistics_cached(left_path)
            self._right_data = self._load_residue_statistics_cached(right_path)
            cache_key = (self._left_data.get("path", left_path), self._right_data.get("path", right_path))
            comparison = self._comparison_cache.get(cache_key)
            if comparison is None:
                comparison = compare_residue_statistics(self._left_data, self._right_data)
                self._comparison_cache[cache_key] = comparison
            self._comparison = comparison
        except Exception as exc:
            self._comparison = None
            QMessageBox.warning(self, "Residue Compare", f"Failed to load residue-statistics CSV:\n{exc}")
            self._summary.setText("Failed to load one or both CSV files.")
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._detail.setText("Select a residue row to inspect its main changes.")
            self._cooperative_detail.setHtml("")
            self._popup_btn.setEnabled(False)
            self._refresh_heatmap_dialog()
            return

        self._popup_btn.setEnabled(True)
        self._refresh_table()

    def _load_residue_statistics_cached(self, path: str) -> dict:
        abs_path = os.path.abspath(str(path or "")).strip()
        if not abs_path:
            raise ValueError("Empty CSV path")
        cached = self._csv_cache.get(abs_path)
        if cached is None:
            cached = load_residue_statistics_csv(abs_path)
            self._csv_cache[abs_path] = cached
        return cached

    def _dataset_key_for_path(self, path: str) -> str:
        normalized = os.path.abspath(str(path or "")).strip()
        for dataset in self._dataset_choices:
            candidate = os.path.abspath(str(dataset.get("residue_statistics_path", "") or "")).strip()
            if candidate and candidate == normalized:
                return str(dataset.get("key", "") or "")
        return ""

    def _refresh_table(self):
        if not self._comparison:
            return

        metrics = self._visible_metrics(self._comparison.get("metrics", []))
        rows = order_comparison_rows(self._comparison, sort_mode="residue_id", changed_only=False)
        self._populate_heatmap_table(
            self._table,
            self._summary,
            self._detail,
            self._cooperative_detail,
            self._comparison,
            rows,
            metrics,
        )
        self._refresh_heatmap_dialog(rows=rows, metrics=metrics)

    def _visible_metrics(self, metrics) -> list[str]:
        return [
            str(metric)
            for metric in (metrics or [])
            if str(metric) != "Affected_point_count"
        ]

    def set_selected_residues(self, selected_residue_keys: set[tuple[str, int]]):
        self._selected_residue_keys = {
            (str(dataset_key), int(residue_id))
            for dataset_key, residue_id in (selected_residue_keys or set())
            if str(dataset_key) and int(residue_id) > 0
        }
        if self._comparison:
            self._refresh_table()

    def _presence_indicator_widget(self, present: bool, active: bool = False) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch()
        indicator = QLabel("✓" if (present and active) else "")
        indicator.setAlignment(Qt.AlignCenter)
        indicator.setFixedSize(16, 16)
        if present:
            indicator.setStyleSheet(
                "background: #FFFFFF; "
                f"border: 1px solid {'#409EFF' if active else '#C9CDD4'}; "
                f"color: {'#409EFF' if active else '#FFFFFF'}; "
                "border-radius: 3px; font-size: 12px; font-weight: 700;"
            )
        else:
            indicator.setStyleSheet(
                "background: #E5E7EB; "
                "border: 1px solid #D1D5DB; "
                "color: #E5E7EB; "
                "border-radius: 3px; font-size: 12px; font-weight: 700;"
            )
        layout.addWidget(indicator)
        layout.addStretch()
        return widget

    def _apply_compact_column_widths(self, table: QTableWidget):
        header = table.horizontalHeader()
        metrics = list(getattr(table, "_comparison_metrics", []) or [])
        table.setColumnWidth(0, max(220, int(table.viewport().width() * 0.28)))
        header.setSectionResizeMode(0, QHeaderView.Stretch)

        for col_idx in (1, 2):
            header.setSectionResizeMode(col_idx, QHeaderView.Fixed)
            label = table.horizontalHeaderItem(col_idx)
            label_width = table.fontMetrics().horizontalAdvance(label.text() if label else "")
            table.setColumnWidth(col_idx, max(34, label_width + 18))

        for offset, metric in enumerate(metrics, start=3):
            header.setSectionResizeMode(offset, QHeaderView.Fixed)
            header_item = table.horizontalHeaderItem(offset)
            header_text = header_item.text() if header_item is not None else metric_label(metric)
            label_width = table.fontMetrics().horizontalAdvance(header_text)
            table.setColumnWidth(offset, max(52, label_width + 20))

    def _populate_heatmap_table(
        self,
        table: QTableWidget,
        summary_label: QLabel,
        detail_label: QLabel,
        cooperative_detail_box: QTextBrowser,
        comparison: dict,
        rows: list[dict],
        metrics: list[str],
    ):
        selected_residue_id = None
        current_row = table.currentRow()
        if current_row >= 0:
            current_item = table.item(current_row, 0)
            if current_item is not None:
                current_data = current_item.data(Qt.UserRole) or {}
                selected_residue_id = current_data.get("residue_id")

        table.blockSignals(True)
        table._comparison_metrics = list(metrics)
        was_sorting = table.isSortingEnabled()
        table.setSortingEnabled(False)
        table.clear()
        table.setColumnCount(3 + len(metrics))
        table.setHorizontalHeaderLabels(
            ["Residue", "A", "B"] + [metric_label(metric) for metric in metrics]
        )
        table.setRowCount(len(rows))

        status_counts = {"both": 0, "left_only": 0, "right_only": 0}
        for row in rows:
            status_counts[row.get("presence_status", "both")] = status_counts.get(row.get("presence_status", "both"), 0) + 1

        summary_label.setText(
            f"Compared {comparison['row_count']} residues, with {comparison['changed_row_count']} changed; "
            f"currently showing {len(rows)} rows across {len(metrics)} numeric properties. "
            f"Present in both: {status_counts.get('both', 0)}, File A only: {status_counts.get('left_only', 0)}, File B only: {status_counts.get('right_only', 0)}. "
            f"White box means present, gray means absent, and a check mark means currently selected."
        )

        residue_header = table.horizontalHeaderItem(0)
        if residue_header is not None:
            residue_header.setToolTip("Residue ID and name. If the names differ, the label is shown as A->B.")
        left_header = table.horizontalHeaderItem(1)
        if left_header is not None:
            left_header.setToolTip("Equivalent to selecting this residue ID from File A.")
        right_header = table.horizontalHeaderItem(2)
        if right_header is not None:
            right_header.setToolTip("Equivalent to selecting this residue ID from File B.")

        metric_ranges = comparison.get("metric_ranges", {})
        for col_idx, metric in enumerate(metrics, start=3):
            header_item = table.horizontalHeaderItem(col_idx)
            if header_item is not None:
                header_item.setToolTip(metric_tooltip(metric))

        for row_idx, row in enumerate(rows):
            residue_item = _SortableTableWidgetItem(
                row["residue_label"],
                sort_value=(int(row["residue_id"]), row["residue_label"]),
            )
            residue_item.setToolTip(
                f"Residue #{row['residue_id']}\n"
                f"File A: {row['left_name'] or '-'}\n"
                f"File B: {row['right_name'] or '-'}"
            )
            residue_item.setData(Qt.UserRole, row)
            table.setItem(row_idx, 0, residue_item)

            left_item = _SortableTableWidgetItem("", sort_value=1 if row.get("present_left") else 0)
            left_item.setToolTip(
                f"File A residue ID: {row['residue_id']}" if row.get("present_left") else "This residue is not present in File A."
            )
            table.setItem(row_idx, 1, left_item)
            left_selected = (self._left_dataset_key, int(row.get("residue_id", 0))) in self._selected_residue_keys
            table.setCellWidget(
                row_idx,
                1,
                self._presence_indicator_widget(bool(row.get("present_left")), active=left_selected),
            )

            right_item = _SortableTableWidgetItem("", sort_value=1 if row.get("present_right") else 0)
            right_item.setToolTip(
                f"File B residue ID: {row['residue_id']}" if row.get("present_right") else "This residue is not present in File B."
            )
            table.setItem(row_idx, 2, right_item)
            right_selected = (self._right_dataset_key, int(row.get("residue_id", 0))) in self._selected_residue_keys
            table.setCellWidget(
                row_idx,
                2,
                self._presence_indicator_widget(bool(row.get("present_right")), active=right_selected),
            )

            for col_idx, metric in enumerate(metrics, start=3):
                cell = row["cells"][metric]
                max_abs_delta = metric_ranges.get(metric, 0.0)
                item = _SortableTableWidgetItem(
                    _format_delta(cell["delta"]),
                    sort_value=float(cell["delta"]),
                )
                item.setTextAlignment(Qt.AlignCenter)
                item.setBackground(QBrush(_delta_color(cell["delta"], max_abs_delta)))
                if max_abs_delta > 0 and abs(cell["delta"]) >= (max_abs_delta * 0.72):
                    item.setForeground(QBrush(QColor("#FFFFFF")))
                metric_tip = metric_tooltip(metric)
                if metric == "Cooperative_residue_count":
                    metric_tip = f"{metric_tip}\nClick the cell to inspect increased/decreased cooperative partners."
                item.setToolTip(
                    f"{metric_label(metric)}\n"
                    f"File A: {_format_number(cell['left'])}\n"
                    f"File B: {_format_number(cell['right'])}\n"
                    f"Delta (B-A): {_format_delta(cell['delta'])}\n"
                    f"Relative: {_format_ratio(cell['ratio'])}\n"
                    f"{metric_tip}"
                )
                table.setItem(row_idx, col_idx, item)

        self._apply_compact_column_widths(table)
        table.blockSignals(False)
        table.setSortingEnabled(was_sorting)

        target_row = 0 if rows else -1
        if selected_residue_id is not None:
            for row_idx, row in enumerate(rows):
                if int(row.get("residue_id", -1)) == int(selected_residue_id):
                    target_row = row_idx
                    break

        if target_row >= 0:
            table.selectRow(target_row)
        else:
            cooperative_detail_box.setHtml("")
        self._update_detail_for(table, detail_label, cooperative_detail_box)

    def _update_detail(self):
        self._update_detail_for(self._table, self._detail, self._cooperative_detail)

    def _update_detail_for(self, table: QTableWidget, detail_label: QLabel, cooperative_detail_box: QTextBrowser):
        selected_items = table.selectedItems()
        if not selected_items:
            detail_label.setText("Select a residue row to inspect its main changes.")
            cooperative_detail_box.setHtml(
                "<div style='color:#909399;'>Select a row to inspect increased and decreased cooperative residue details.</div>"
            )
            return

        row_item = table.item(selected_items[0].row(), 0)
        if row_item is None:
            detail_label.setText("Select a residue row to inspect its main changes.")
            cooperative_detail_box.setHtml(
                "<div style='color:#909399;'>Select a row to inspect increased and decreased cooperative residue details.</div>"
            )
            return

        row = row_item.data(Qt.UserRole) or {}
        changed_metrics = sorted(
            (
                (metric, cell)
                for metric, cell in row.get("cells", {}).items()
            ),
            key=lambda entry: (-float(entry[1]["abs_delta"]), entry[0]),
        )
        preview = changed_metrics[:3]
        if not preview or float(preview[0][1]["abs_delta"]) <= 1e-12:
            detail_label.setText(f"{row.get('residue_label', 'Residue')}: no numeric change.")
            cooperative_detail_box.setHtml(_cooperative_change_html(row))
            return

        parts = [
            (
                f"{metric_label(metric)} {_format_delta(cell['delta'])} "
                f"(A={_format_number(cell['left'])}, B={_format_number(cell['right'])})"
            )
            for metric, cell in preview
        ]
        detail_label.setText(
            f"{row.get('residue_label', 'Residue')} | "
            + " | ".join(parts)
        )
        cooperative_detail_box.setHtml(_cooperative_change_html(row))

    def _handle_cell_clicked(self, row: int, col: int):
        self._on_table_cell_clicked(self._table, row, col, self._detail, self._cooperative_detail)

    def _on_table_cell_clicked(
        self,
        table: QTableWidget,
        row: int,
        col: int,
        detail_label: QLabel,
        cooperative_detail_box: QTextBrowser,
    ):
        row_item = table.item(row, 0)
        row_data = row_item.data(Qt.UserRole) if row_item is not None else None
        if row_data and col in (1, 2):
            dataset_key = self._left_dataset_key if col == 1 else self._right_dataset_key
            if dataset_key:
                present = bool(row_data.get("present_left")) if col == 1 else bool(row_data.get("present_right"))
                if present:
                    self.residue_selection_requested.emit(dataset_key, int(row_data.get("residue_id", 0)))

        metrics = list(getattr(table, "_comparison_metrics", []) or [])
        metric = None
        if col >= 3 and (col - 3) < len(metrics):
            metric = metrics[col - 3]
        table._detail_metric = metric
        self._update_detail_for(table, detail_label, cooperative_detail_box)

    def _open_heatmap_dialog(self):
        if not self._comparison:
            QMessageBox.information(self, "Residue Heatmap", "Load both File A and File B first.")
            return

        if self._heatmap_dialog is None:
            self._heatmap_dialog = ResidueHeatmapDialog(self, self.window())
            self._heatmap_dialog.destroyed.connect(self._on_heatmap_dialog_destroyed)

        self._refresh_heatmap_dialog()
        self._heatmap_dialog.show()
        self._heatmap_dialog.raise_()
        self._heatmap_dialog.activateWindow()

    def _refresh_heatmap_dialog(self, rows: list[dict] | None = None, metrics: list[str] | None = None):
        if self._heatmap_dialog is None:
            return
        if not self._comparison:
            self._heatmap_dialog.refresh({}, [], [])
            return
        if rows is None:
            rows = order_comparison_rows(self._comparison, sort_mode="residue_id", changed_only=False)
        if metrics is None:
            metrics = self._visible_metrics(self._comparison.get("metrics", []))
        self._heatmap_dialog.refresh(self._comparison, rows, metrics)

    def _on_heatmap_dialog_destroyed(self, *_args):
        self._heatmap_dialog = None
