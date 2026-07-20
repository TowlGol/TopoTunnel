"""
Residue selection panel - replaces ResiduesTab.vue.
Search box + checkable list + selected tags.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget,
    QListWidgetItem, QLabel, QPushButton, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt, Signal


class ResiduePanel(QWidget):
    """Residue selection panel with search and tag display."""

    selection_changed = Signal()  # Emitted when residues are toggled

    def __init__(self, state, db, parent=None):
        super().__init__(parent)
        self.state = state
        self.db = db
        self._dataset_keys_filter: list[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ─── Selected Tags ──────────────────────────────
        selected_header = QHBoxLayout()
        selected_header.addWidget(QLabel("Selected Residues"))
        self._clear_btn = QPushButton("Clear All")
        self._clear_btn.setProperty("flat", True)
        self._clear_btn.setFixedHeight(24)
        self._clear_btn.clicked.connect(self._clear_all)
        selected_header.addStretch()
        selected_header.addWidget(self._clear_btn)
        layout.addLayout(selected_header)

        self._tags_widget = QWidget()
        self._tags_layout = QHBoxLayout(self._tags_widget)
        self._tags_layout.setContentsMargins(0, 0, 0, 0)
        self._tags_layout.setSpacing(4)
        self._tags_layout.addStretch()

        tags_scroll = QScrollArea()
        tags_scroll.setWidget(self._tags_widget)
        tags_scroll.setWidgetResizable(True)
        tags_scroll.setFixedHeight(40)
        tags_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        tags_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        tags_scroll.setFrameShape(QFrame.NoFrame)
        layout.addWidget(tags_scroll)

        # ─── Search ─────────────────────────────────────
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search residues...")
        self._search.textChanged.connect(self._filter_list)
        layout.addWidget(self._search)

        # ─── Available Residues List ────────────────────
        layout.addWidget(QLabel("Available Residues"))
        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        # ─── Count label ────────────────────────────────
        self._count_label = QLabel("0 residues available")
        self._count_label.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(self._count_label)

    def set_db(self, db):
        self.db = db

    def set_dataset_filter(self, dataset_keys):
        self._dataset_keys_filter = [
            str(key).strip()
            for key in (dataset_keys or [])
            if str(key).strip()
        ]

    def load_options(self, options=None):
        """Load residue options from database or provided list."""
        if options is None:
            options = self._filtered_residue_options(
                self.state.selected_residues if self.state.selected_residues else None
            )

        self._list.blockSignals(True)
        self._list.clear()
        for opt in options:
            item = QListWidgetItem(opt["label"])
            item.setData(Qt.UserRole, int(opt["value"]))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            rid = int(opt["value"])
            item.setCheckState(
                Qt.CheckState.Checked if rid in self.state.selected_residues else Qt.CheckState.Unchecked
            )
            self._list.addItem(item)
        self._list.blockSignals(False)

        self._count_label.setText(f"{len(options)} residues available")
        self._update_tags()

    def _filtered_residue_options(self, selected_residues=None):
        if not self._dataset_keys_filter or not hasattr(self.db, "get_dataset"):
            return self.db.get_residue_options(selected_residues)

        filtered_options = []
        selected_set = set(int(rid) for rid in (selected_residues or []))
        has_selected = bool(selected_set)
        for dataset_key in self._dataset_keys_filter:
            dataset = self.db.get_dataset(dataset_key)
            dataset_binding = getattr(self.db, "_datasets_by_key", {}).get(dataset_key)
            dataset_db = dataset_binding.db if dataset_binding is not None else None
            if dataset is None or dataset_db is None:
                continue
            local_selected = (
                self.db.get_local_residue_ids_for_dataset(dataset_key, selected_set)
                if has_selected and hasattr(self.db, "get_local_residue_ids_for_dataset")
                else []
            )
            if has_selected and not local_selected:
                continue
            local_options = dataset_db.get_residue_options(set(local_selected) if has_selected else None)
            for option in local_options:
                local_id = int(option["value"])
                global_id = dataset_binding.residue_id_base + local_id
                filtered_options.append(
                    {
                        "label": f"[{dataset['prefix']}] {option['label']}",
                        "value": str(global_id),
                    }
                )
        return filtered_options

    def _on_item_changed(self, item: QListWidgetItem):
        rid = item.data(Qt.UserRole)
        if item.checkState() == Qt.CheckState.Checked:
            self.state.selected_residues.add(rid)
        else:
            self.state.selected_residues.discard(rid)
        self._update_tags()
        self.selection_changed.emit()

    def _filter_list(self, text: str):
        text = text.strip().lower()
        for i in range(self._list.count()):
            item = self._list.item(i)
            item.setHidden(text not in item.text().lower())

    def _clear_all(self):
        self.state.clear_residues()
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Unchecked)
        self._list.blockSignals(False)
        self._update_tags()
        self.selection_changed.emit()

    def _update_tags(self):
        # Clear existing tags
        while self._tags_layout.count() > 1:
            child = self._tags_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Add tags for selected residues
        for rid in sorted(self.state.selected_residues):
            label = self.db.format_residue_label(rid) if hasattr(self.db, "format_residue_label") else str(rid)
            tag = QPushButton(f"{label} x")
            tag.setFixedHeight(22)
            tag.setStyleSheet(
                "background-color: #ecf5ff; color: #409eff; "
                "border: 1px solid #d9ecff; border-radius: 3px; "
                "padding: 0 6px; font-size: 11px;"
            )
            tag.clicked.connect(lambda checked, r=rid: self._remove_tag(r))
            self._tags_layout.insertWidget(self._tags_layout.count() - 1, tag)

    def _remove_tag(self, rid: int):
        self.state.selected_residues.discard(rid)
        # Update list checkbox
        self._list.blockSignals(True)
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.UserRole) == rid:
                item.setCheckState(Qt.CheckState.Unchecked)
                break
        self._list.blockSignals(False)
        self._update_tags()
        self.selection_changed.emit()
