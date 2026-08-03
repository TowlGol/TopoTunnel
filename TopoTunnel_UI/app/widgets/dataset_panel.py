"""
Dataset management panel for loading and distinguishing multiple datasets.
"""
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QPushButton, QFileDialog, QMessageBox, QInputDialog, QColorDialog,
    QListView, QAbstractItemView, QFrame, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal


class DatasetPanel(QWidget):
    datasets_changed = Signal(dict)
    dataset_color_changed = Signal(str, str)
    dataset_visibility_changed = Signal(str, bool)
    dataset_visibility_batch_changed = Signal(object, bool)
    dataset_selection_changed = Signal(object)
    dataset_cluster_mode_changed = Signal(str, bool)
    dataset_cluster_labels_changed = Signal(str, bool)
    cluster_paths_selected = Signal(object)
    reference_protein_requested = Signal()

    def __init__(self, db, state=None, parent=None):
        super().__init__(parent)
        self.db = db
        self.state = state
        self._cluster_item_path_ids: dict[tuple[str, int], set[int]] = {}
        self._suppress_cluster_item_changed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QLabel("Loaded Datasets")
        header.setStyleSheet("font-weight: bold;")
        layout.addWidget(header)

        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(self._summary_label)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(1)
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._tree.setIndentation(14)
        self._tree.setRootIsDecorated(True)
        self._tree.setStyleSheet(
            "QTreeWidget{border:1px solid #E5E7EB; border-radius:10px; background:#FAFBFC;}"
            "QTreeWidget::item{padding:4px 0;}"
            "QTreeWidget::item:selected{background:#EAF2FF;}"
        )
        self._tree.itemSelectionChanged.connect(self._update_controls)
        self._tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._tree, 1)

        button_row = QGridLayout()
        self._button_row = button_row
        button_row.setHorizontalSpacing(6)
        button_row.setVerticalSpacing(6)
        self._add_btn = QPushButton("Add Dataset")
        self._add_btn.clicked.connect(self._on_add)
        button_row.addWidget(self._add_btn, 0, 0)

        self._prefix_btn = QPushButton("Edit Prefix")
        self._prefix_btn.clicked.connect(self._on_edit_prefix)
        button_row.addWidget(self._prefix_btn, 0, 1)

        self._color_btn = QPushButton("Pick Color")
        self._color_btn.clicked.connect(self._on_pick_color)
        button_row.addWidget(self._color_btn, 0, 2)

        self._reference_btn = QPushButton("Reference PDB")
        self._reference_btn.clicked.connect(self.reference_protein_requested.emit)
        button_row.addWidget(self._reference_btn, 0, 3)

        self._visibility_btn = QPushButton("Show Tunnels")
        self._visibility_btn.clicked.connect(self._on_toggle_visibility)
        button_row.addWidget(self._visibility_btn, 1, 0)

        self._cluster_mode_btn = QPushButton("Cluster Colors")
        self._cluster_mode_btn.setCheckable(True)
        self._cluster_mode_btn.clicked.connect(self._on_toggle_cluster_mode)
        button_row.addWidget(self._cluster_mode_btn, 1, 1)

        self._cluster_labels_btn = QPushButton("Cluster Labels")
        self._cluster_labels_btn.setCheckable(True)
        self._cluster_labels_btn.clicked.connect(self._on_toggle_cluster_labels)
        button_row.addWidget(self._cluster_labels_btn, 1, 2)

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setProperty("danger", True)
        self._remove_btn.clicked.connect(self._on_remove)
        button_row.addWidget(self._remove_btn, 1, 3)
        button_row.setColumnStretch(0, 1)
        button_row.setColumnStretch(1, 1)
        button_row.setColumnStretch(2, 1)
        button_row.setColumnStretch(3, 1)
        layout.addLayout(button_row)

        self._fit_button_texts(
            self._add_btn,
            self._prefix_btn,
            self._color_btn,
            self._reference_btn,
            self._visibility_btn,
            self._cluster_mode_btn,
            self._cluster_labels_btn,
            self._remove_btn,
        )

        hint = QLabel(
            "Color bar matches the main view. "
            "Status tags show visibility and render mode directly."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #909399; font-size: 11px;")
        layout.addWidget(hint)

        self.refresh()

    def primary_button_row_width(self) -> int:
        """Return the full width required by the four top-row buttons."""
        self.ensurePolished()
        margins = self.layout().contentsMargins()
        return (
            self._button_row.minimumSize().width()
            + margins.left()
            + margins.right()
        )

    def refresh(self):
        current_key = self._current_key()
        selected_keys = self._selected_dataset_keys()
        datasets = self.db.list_datasets()
        checked_clusters = self._checked_cluster_keys()
        self._suppress_cluster_item_changed = True
        self._tree.clear()
        self._cluster_item_path_ids.clear()
        total_paths = 0
        total_residues = 0
        for dataset in datasets:
            total_paths += int(dataset["path_count"])
            total_residues += int(dataset["residue_count"])
            item = QTreeWidgetItem([""])
            item.setData(0, Qt.UserRole, {"type": "dataset", "key": dataset["key"]})
            self._tree.addTopLevelItem(item)
            item.setFirstColumnSpanned(True)
            widget = self._create_dataset_item_widget(dataset)
            self._tree.setItemWidget(item, 0, widget)
            item.setSizeHint(0, widget.sizeHint())

            if hasattr(self.db, "get_dataset_cluster_path_groups"):
                cluster_groups = self.db.get_dataset_cluster_path_groups(dataset["key"])
                for cluster_id, path_ids in sorted(cluster_groups.items(), key=lambda kv: (kv[0], len(kv[1]))):
                    child = QTreeWidgetItem(
                        [f"Cluster {cluster_id} ({len(path_ids)} paths)"]
                    )
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
                    child.setCheckState(0, Qt.Unchecked)
                    child.setData(
                        0,
                        Qt.UserRole,
                        {"type": "cluster", "key": dataset["key"], "cluster_id": int(cluster_id)},
                    )
                    if (dataset["key"], int(cluster_id)) in checked_clusters:
                        child.setCheckState(0, Qt.Checked)
                    item.addChild(child)
                    self._cluster_item_path_ids[(dataset["key"], int(cluster_id))] = set(path_ids)
            item.setExpanded(False)
        self._summary_label.setText(
            f"{len(datasets)} dataset(s) loaded | {total_paths:,} paths | {total_residues:,} residues"
        )
        if current_key:
            for i in range(self._tree.topLevelItemCount()):
                item = self._tree.topLevelItem(i)
                payload = item.data(0, Qt.UserRole) or {}
                if payload.get("key") == current_key:
                    self._tree.setCurrentItem(item)
                    break
        if selected_keys:
            for i in range(self._tree.topLevelItemCount()):
                item = self._tree.topLevelItem(i)
                payload = item.data(0, Qt.UserRole) or {}
                if payload.get("key") in selected_keys:
                    item.setSelected(True)
        if self._tree.currentItem() is None and self._tree.topLevelItemCount():
            self._tree.setCurrentItem(self._tree.topLevelItem(0))
        self._suppress_cluster_item_changed = False
        self._update_controls()
        if checked_clusters:
            self._emit_cluster_selection()

    def _current_key(self):
        item = self._tree.currentItem()
        if item is None:
            return None
        payload = item.data(0, Qt.UserRole) or {}
        if payload.get("type") == "cluster":
            return payload.get("key")
        return payload.get("key")

    def _selected_dataset_keys(self) -> list[str]:
        keys: list[str] = []
        seen: set[str] = set()
        for item in self._tree.selectedItems():
            payload = item.data(0, Qt.UserRole) or {}
            key = str(payload.get("key") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    def selected_dataset_keys(self) -> list[str]:
        return self._selected_dataset_keys()

    def current_dataset_key(self) -> str | None:
        key = self._current_key()
        return str(key) if key else None

    def _dataset_color(self, key: str) -> str:
        if self.state is not None:
            return self.state.dataset_colors.get(key, "#4169E1")
        return "#4169E1"

    def _badge_style(self, background: str, foreground: str = "#FFFFFF") -> str:
        return (
            f"background:{background}; color:{foreground}; border-radius:9px; "
            "padding:2px 8px; font-size:10px; font-weight:600;"
        )

    def _badge_label(self, text: str, background: str, foreground: str = "#FFFFFF") -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(self._badge_style(background, foreground))
        return label

    def _create_dataset_item_widget(self, dataset: dict) -> QWidget:
        key = str(dataset["key"])
        color = self._dataset_color(key)
        visible = self._is_dataset_visible(key)
        cluster_mode = self._is_cluster_mode(key)
        cluster_labels = self._is_cluster_labels_mode(key)

        frame = QFrame()
        frame.setObjectName("datasetCard")
        frame.setStyleSheet(
            "QFrame#datasetCard{"
            "border:1px solid #E5E7EB; border-radius:10px; background:#FFFFFF;"
            "}"
        )
        if not visible:
            frame.setStyleSheet(
                "QFrame#datasetCard{"
                "border:1px solid #E5E7EB; border-radius:10px; background:#F8FAFC;"
                "}"
            )
        root = QHBoxLayout(frame)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        color_bar = QFrame()
        color_bar.setFixedWidth(12)
        color_bar.setStyleSheet(
            f"background:{color}; border:1px solid rgba(15,23,42,0.12); border-radius:6px;"
        )
        root.addWidget(color_bar)

        content = QVBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(4)
        root.addLayout(content, 1)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(6)
        content.addLayout(title_row)

        prefix_label = QLabel(f"[{dataset['prefix']}]")
        prefix_label.setStyleSheet(
            f"color:{color}; font-size:11px; font-weight:700; "
            "background:#F4F7FB; border-radius:6px; padding:1px 6px;"
        )
        title_row.addWidget(prefix_label, 0, Qt.AlignVCenter)

        name_label = QLabel(str(dataset["name"]))
        name_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        name_label.setStyleSheet(
            "color:#111827; font-size:12px; font-weight:600;"
            if visible else
            "color:#9CA3AF; font-size:12px; font-weight:600;"
        )
        name_label.setToolTip(str(dataset.get("path") or dataset["name"]))
        title_row.addWidget(name_label, 1)

        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(4)
        badge_row.addWidget(
            self._badge_label(
                "Visible" if visible else "Hidden",
                "#16A34A" if visible else "#9CA3AF",
            )
        )
        badge_row.addWidget(
            self._badge_label(
                "Cluster" if cluster_mode else "Dataset",
                "#2563EB" if cluster_mode else "#64748B",
            )
        )
        if cluster_labels:
            badge_row.addWidget(self._badge_label("Labels", "#7C3AED"))
        title_row.addLayout(badge_row)

        meta_label = QLabel(
            f"{int(dataset['path_count']):,} paths   {int(dataset['residue_count']):,} residues"
        )
        meta_label.setStyleSheet(
            "color:#6B7280; font-size:11px;"
            if visible else
            "color:#9CA3AF; font-size:11px;"
        )
        content.addWidget(meta_label)

        folder_label = QLabel(os.path.basename(str(dataset.get("folder") or "")) or str(dataset.get("folder") or ""))
        folder_label.setStyleSheet(
            "color:#9CA3AF; font-size:10px;"
            if visible else
            "color:#C0C4CC; font-size:10px;"
        )
        folder_label.setToolTip(str(dataset.get("folder") or ""))
        content.addWidget(folder_label)

        return frame

    def _is_dataset_visible(self, key: str) -> bool:
        return bool(self.state is not None and key in self.state.visible_datasets)

    def _is_cluster_mode(self, key: str) -> bool:
        return bool(self.state is not None and self.state.dataset_cluster_display_key == key)

    def _is_cluster_labels_mode(self, key: str) -> bool:
        return bool(self.state is not None and self.state.dataset_cluster_label_key == key)

    def _update_controls(self):
        selected_keys = self._selected_dataset_keys()
        has_selection = bool(selected_keys)
        single_selection = len(selected_keys) == 1
        current_key = selected_keys[0] if single_selection else None
        self.dataset_selection_changed.emit(selected_keys)
        self._prefix_btn.setEnabled(single_selection)
        self._color_btn.setEnabled(single_selection)
        self._remove_btn.setEnabled(has_selection)
        self._visibility_btn.setEnabled(has_selection)
        self._cluster_mode_btn.setEnabled(single_selection)
        self._cluster_labels_btn.setEnabled(single_selection)
        if has_selection and all(self._is_dataset_visible(key) for key in selected_keys):
            self._visibility_btn.setText("Hide Tunnels")
        else:
            self._visibility_btn.setText("Show Tunnels")
        self._cluster_mode_btn.blockSignals(True)
        self._cluster_mode_btn.setChecked(single_selection and self._is_cluster_mode(current_key))
        self._cluster_mode_btn.blockSignals(False)
        self._cluster_mode_btn.setText("Dataset Colors" if single_selection and self._is_cluster_mode(current_key) else "Cluster Colors")
        self._cluster_labels_btn.blockSignals(True)
        self._cluster_labels_btn.setChecked(single_selection and self._is_cluster_labels_mode(current_key))
        self._cluster_labels_btn.blockSignals(False)
        self._cluster_labels_btn.setText("Hide Labels" if single_selection and self._is_cluster_labels_mode(current_key) else "Cluster Labels")
        self._fit_button_texts(self._visibility_btn, self._cluster_mode_btn, self._cluster_labels_btn)

    def _fit_button_texts(self, *buttons: QPushButton):
        for button in buttons:
            width = button.fontMetrics().horizontalAdvance(button.text()) + 28
            button.setMinimumWidth(max(button.minimumWidth(), width))

    def _emit_cluster_selection(self):
        selected_paths: set[int] = set()
        for i in range(self._tree.topLevelItemCount()):
            dataset_item = self._tree.topLevelItem(i)
            for idx in range(dataset_item.childCount()):
                child = dataset_item.child(idx)
                if child.checkState(0) != Qt.Checked:
                    continue
                payload = child.data(0, Qt.UserRole) or {}
                key = payload.get("key")
                cluster_id = payload.get("cluster_id")
                if key is None or cluster_id is None:
                    continue
                selected_paths.update(self._cluster_item_path_ids.get((str(key), int(cluster_id)), set()))
        self.cluster_paths_selected.emit(selected_paths)

    def _checked_cluster_keys(self) -> set[tuple[str, int]]:
        checked: set[tuple[str, int]] = set()
        for i in range(self._tree.topLevelItemCount()):
            dataset_item = self._tree.topLevelItem(i)
            for idx in range(dataset_item.childCount()):
                child = dataset_item.child(idx)
                if child.checkState(0) != Qt.Checked:
                    continue
                payload = child.data(0, Qt.UserRole) or {}
                key = payload.get("key")
                cluster_id = payload.get("cluster_id")
                if key is None or cluster_id is None:
                    continue
                checked.add((str(key), int(cluster_id)))
        return checked

    def _on_item_changed(self, item, column):
        if self._suppress_cluster_item_changed or column != 0:
            return
        payload = item.data(0, Qt.UserRole) or {}
        if payload.get("type") != "cluster":
            return
        self._emit_cluster_selection()

    def _on_add(self):
        folder_paths = self._select_dataset_folders()
        changed = False
        failures: list[tuple[str, Exception]] = []
        for folder_path in folder_paths:
            try:
                dataset_info = self._prepare_dataset_folder(folder_path)
                self.db.add_dataset(
                    dataset_info["db_path"],
                    folder=dataset_info["folder"],
                    residue_statistics_path=dataset_info["residue_statistics_path"],
                    residue_combination_statistics_path=dataset_info["residue_combination_statistics_path"],
                    md_root_path=dataset_info["md_root_path"],
                )
                changed = True
            except Exception as exc:
                failures.append((folder_path, exc))
        if failures:
            detail_lines = [
                f"{folder_path}\n{exc}"
                for folder_path, exc in failures
            ]
            QMessageBox.warning(
                self,
                "Dataset Import",
                "Failed to add some dataset folders:\n\n" + "\n\n".join(detail_lines),
            )
        if changed:
            self.refresh()
            self.datasets_changed.emit({
                "action": "add",
                "refresh_3d": True,
                "reset_state": True,
            })

    def _select_dataset_folders(self) -> list[str]:
        dialog = QFileDialog(self, "Select Dataset Folder(s)")
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        view = dialog.findChild(QListView, "listView")
        if view is not None:
            view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tree = dialog.findChild(QTreeWidget)
        if tree is not None:
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dialog.exec() != QFileDialog.Accepted:
            return []
        selected = [
            os.path.abspath(path)
            for path in dialog.selectedFiles()
            if path and os.path.isdir(path)
        ]
        unique: list[str] = []
        seen: set[str] = set()
        for path in selected:
            normalized = os.path.normcase(path)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(path)
        return unique

    def _prepare_dataset_folder(self, folder_path: str) -> dict:
        folder = os.path.abspath(folder_path)
        if not os.path.isdir(folder):
            raise ValueError("Selected path is not a folder.")

        db_path = self._find_first_file(folder, ".db")
        pkl_path = self._find_preferred_pkl_file(folder)
        db_path = db_path or pkl_path
        if not db_path:
            raise ValueError("No preprocessed_paths.pkl, .pkl file, or .db file was found in the selected folder.")

        residue_statistics_path = self._find_named_file(folder, "residue_statistics.csv")
        if not residue_statistics_path:
            residue_statistics_path = self._find_named_file(folder, "residue_static.csv")
        residue_combination_statistics_path = self._find_named_file(folder, "residue_combination_statistics.csv")
        if not residue_combination_statistics_path:
            residue_combination_statistics_path = self._find_named_file(folder, "residue_combination_statitics.csv")
        md_root_path = self._read_md_root_path(folder)

        return {
            "folder": folder,
            "db_path": db_path,
            "residue_statistics_path": residue_statistics_path or "",
            "residue_combination_statistics_path": residue_combination_statistics_path or "",
            "md_root_path": md_root_path or "",
        }

    def _find_first_file(self, folder: str, extension: str) -> str:
        extension = extension.lower()
        candidates: list[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(extension):
                    candidates.append(os.path.join(root, name))
        if not candidates:
            return ""
        candidates.sort(key=lambda path: (os.path.dirname(path) != folder, path.lower()))
        return os.path.abspath(candidates[0])

    def _find_named_file(self, folder: str, filename: str) -> str:
        target = filename.lower()
        candidates: list[str] = []
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if name.lower() == target:
                    candidates.append(os.path.join(root, name))
        if not candidates:
            return ""
        candidates.sort(key=lambda path: (os.path.dirname(path) != folder, path.lower()))
        return os.path.abspath(candidates[0])

    def _find_preferred_pkl_file(self, folder: str) -> str:
        preferred = self._find_named_file(folder, "preprocessed_paths.pkl")
        if preferred:
            return preferred
        return self._find_first_file(folder, ".pkl")

    def _read_md_root_path(self, folder: str) -> str:
        md_path_file = self._find_named_file(folder, "MD_path.txt")
        if not md_path_file:
            return ""
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
                    if candidate and os.path.isfile(candidate) and candidate.lower().endswith(".pdb"):
                        candidate = os.path.dirname(candidate)
                    if candidate and os.path.isdir(candidate):
                        return candidate
        except OSError:
            return ""
        return ""

    def _on_edit_prefix(self):
        key = self._current_key()
        if not key:
            return
        dataset = next((row for row in self.db.list_datasets() if row["key"] == key), None)
        if dataset is None:
            return
        prefix, ok = QInputDialog.getText(
            self,
            "Edit Prefix",
            "Dataset prefix:",
            text=dataset["prefix"],
        )
        if not ok:
            return
        prefix = prefix.strip()
        if not prefix:
            return
        self.db.update_dataset_prefix(key, prefix)
        self.refresh()
        self.datasets_changed.emit({
            "action": "prefix",
            "refresh_3d": False,
            "reset_state": False,
        })

    def _on_pick_color(self):
        key = self._current_key()
        if not key:
            return
        current = self._dataset_color(key)
        color = QColorDialog.getColor(parent=self)
        if not color.isValid():
            return
        hex_color = color.name().upper()
        if hex_color == current:
            return
        self.dataset_color_changed.emit(key, hex_color)
        self.refresh()

    def _on_toggle_visibility(self):
        keys = self._selected_dataset_keys()
        if not keys:
            return
        visible = not all(self._is_dataset_visible(key) for key in keys)
        if len(keys) == 1:
            self.dataset_visibility_changed.emit(keys[0], visible)
        else:
            self.dataset_visibility_batch_changed.emit(keys, visible)
        self.refresh()

    def _on_toggle_cluster_mode(self):
        key = self._current_key()
        if not key:
            return
        enabled = not self._is_cluster_mode(key)
        self.dataset_cluster_mode_changed.emit(key, enabled)
        self.refresh()

    def _on_toggle_cluster_labels(self):
        key = self._current_key()
        if not key:
            return
        enabled = not self._is_cluster_labels_mode(key)
        self.dataset_cluster_labels_changed.emit(key, enabled)
        self.refresh()

    def _on_remove(self):
        keys = self._selected_dataset_keys()
        datasets = self.db.list_datasets()
        if not keys:
            return
        selected_datasets = [row for row in datasets if row["key"] in keys]
        if not selected_datasets:
            return
        count = len(selected_datasets)
        if count == 1:
            dataset = selected_datasets[0]
            prompt = f"Remove dataset [{dataset['prefix']}] {dataset['name']}?"
        else:
            prompt = f"Remove {count} selected datasets?"
        reply = QMessageBox.question(
            self,
            "Remove Dataset",
            prompt,
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        removed_keys: list[str] = []
        removed_path_ids: list[int] = []
        any_visible = False
        for dataset in selected_datasets:
            key = str(dataset["key"])
            any_visible = any_visible or self._is_dataset_visible(key)
            if hasattr(self.db, "get_dataset_path_ids"):
                removed_path_ids.extend(int(path_id) for path_id in self.db.get_dataset_path_ids(key))
            if self.db.remove_dataset(key):
                removed_keys.append(key)
        if not removed_keys:
            return
        self.refresh()
        self.datasets_changed.emit({
            "action": "remove",
            "refresh_3d": bool(any_visible),
            "reset_state": bool(any_visible),
            "removed_key": removed_keys[0] if len(removed_keys) == 1 else "",
            "removed_keys": removed_keys,
            "removed_path_ids": removed_path_ids,
            "was_visible": bool(any_visible),
        })
