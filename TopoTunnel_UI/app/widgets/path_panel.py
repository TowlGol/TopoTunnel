"""
Path list panel - replaces PathsTab.vue.
Category management + path table + batch operations.
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QHeaderView,
    QLabel, QPushButton, QLineEdit, QGroupBox, QListWidget,
    QListWidgetItem, QAbstractItemView, QMessageBox, QColorDialog,
    QFileDialog, QTabBar, QMenu, QSizePolicy, QApplication,
)
from PySide6.QtCore import Qt, Signal, QItemSelectionModel, QTimer

from TopoTunnel_UI.app.models.path_model import PathTableModel, COL_KEYS, COL_WIDTHS


class PathPanel(QWidget):
    """Path list with categories, filtering, batch operations."""

    path_selected = Signal(int)           # Single path clicked
    paths_selected = Signal(list)         # Multiple paths selected
    export_requested = Signal(str, list)  # (category_name, path_ids)
    profile_requested = Signal(list)      # path_ids for profile charts

    def __init__(self, state, db, parent=None):
        super().__init__(parent)
        self.state = state
        self.db = db
        self._manual_selected_path_ids = set()
        self._table_filter_signature = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        left_widget = QWidget(self)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        right_widget = QWidget(self)
        right_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        right_widget.setMinimumWidth(244)
        right_widget.setMaximumWidth(320)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        # ─── Category Management ────────────────────────
        cat_group = QGroupBox("Path Clusters")
        self._cat_group = cat_group
        cat_layout = QVBoxLayout(cat_group)
        cat_layout.setContentsMargins(6, 6, 6, 6)
        cat_layout.setSpacing(4)

        # Category creation
        cat_create = QHBoxLayout()
        cat_create.setSpacing(4)
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("Cluster name...")
        self._cat_input.returnPressed.connect(self._create_category)
        cat_create.addWidget(self._cat_input)
        self._cat_create_btn = QPushButton("Create")
        self._cat_create_btn.clicked.connect(self._create_category)
        cat_create.addWidget(self._cat_create_btn)
        cat_layout.addLayout(cat_create)

        self._cat_from_current_btn = QPushButton("Create From Current")
        self._cat_from_current_btn.clicked.connect(self._create_category)
        cat_layout.addWidget(self._cat_from_current_btn)

        # Category list
        self._cat_list = QListWidget()
        self._cat_list.setMinimumHeight(180)
        self._cat_list.itemClicked.connect(self._on_category_clicked)
        self._cat_list.itemDoubleClicked.connect(self._on_category_toggle_clicked)
        self._cat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._cat_list.customContextMenuRequested.connect(self._on_category_context_menu)
        cat_layout.addWidget(self._cat_list)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(4)
        self._cat_select_btn = QPushButton("Select")
        self._cat_select_btn.clicked.connect(self._select_category_paths)
        quick_row.addWidget(self._cat_select_btn)
        self._cat_show_btn = QPushButton("Show")
        self._cat_show_btn.clicked.connect(self._toggle_active_category_visibility)
        quick_row.addWidget(self._cat_show_btn)
        cat_layout.addLayout(quick_row)

        self._cat_select_show_btn = QPushButton("Select && Show")
        self._cat_select_show_btn.clicked.connect(self._select_and_show_category_paths)
        cat_layout.addWidget(self._cat_select_show_btn)

        self._cat_color_btn = QPushButton("Set Color")
        self._cat_color_btn.clicked.connect(self._choose_category_color)
        cat_layout.addWidget(self._cat_color_btn)

        # Batch category actions
        batch_row = QHBoxLayout()
        batch_row.setSpacing(2)
        for label, slot in [
            ("Show All", lambda: self.state.show_all_categories()),
            ("Hide All", lambda: self.state.hide_all_categories()),
            ("Excl All", lambda: self.state.exclude_all_categories()),
            ("Incl All", lambda: self.state.clear_all_exclusions()),
        ]:
            btn = QPushButton(label)
            btn.setProperty("flat", True)
            btn.setMaximumHeight(22)
            btn.setStyleSheet("font-size: 10px; padding: 1px 4px;")
            btn.clicked.connect(slot)
            batch_row.addWidget(btn)
        batch_row.addStretch()
        cat_layout.addLayout(batch_row)

        # Category actions
        cat_actions = QHBoxLayout()
        cat_actions.setSpacing(4)
        self._cat_export_btn = QPushButton("Export")
        self._cat_export_btn.setProperty("flat", True)
        self._cat_export_btn.clicked.connect(self._export_category)
        cat_actions.addWidget(self._cat_export_btn)
        self._cat_delete_btn = QPushButton("Delete")
        self._cat_delete_btn.setProperty("danger", True)
        self._cat_delete_btn.clicked.connect(self._delete_category)
        cat_actions.addWidget(self._cat_delete_btn)
        cat_actions.addStretch()
        cat_layout.addLayout(cat_actions)

        self._cat_hint = QLabel("Single click selects a cluster. Double click toggles 3D display.")
        self._cat_hint.setWordWrap(True)
        self._cat_hint.setStyleSheet("color:#909399;font-size:11px;")
        cat_layout.addWidget(self._cat_hint)

        right_layout.addWidget(cat_group, 1)

        # ─── Sub-tab bar: All / Selected ─────────────────
        tab_row = QHBoxLayout()
        tab_row.setSpacing(6)

        self._sub_tabs = QTabBar()
        self._sub_tabs.addTab("All Paths")
        self._sub_tabs.addTab("Selected (0)")
        self._sub_tabs.setCurrentIndex(0)
        self._sub_tabs.setTabToolTip(0, "Show all paths matching the current frame and residue filters")
        self._sub_tabs.setTabToolTip(1, "Show the current selection immediately")
        self._sub_tabs.setExpanding(False)
        self._sub_tabs.setStyleSheet("QTabBar::tab { min-width: 100px; }")
        self._sub_tabs.currentChanged.connect(self._on_sub_tab_changed)
        tab_row.addWidget(self._sub_tabs)

        self._load_profiles_btn = QPushButton("Load Profiles")
        self._load_profiles_btn.setToolTip("Send the current selected paths to the profile chart")
        self._load_profiles_btn.setEnabled(False)
        self._load_profiles_btn.setVisible(False)
        self._load_profiles_btn.clicked.connect(self._on_load_profiles)
        tab_row.addWidget(self._load_profiles_btn)
        tab_row.addStretch()

        left_layout.addLayout(tab_row)

        # ─── Path Table ─────────────────────────────────
        path_header = QHBoxLayout()
        self._path_count_label = QLabel("Paths: 0")
        self._path_count_label.setStyleSheet("font-weight: bold;")
        path_header.addWidget(self._path_count_label)
        path_header.addStretch()

        # Batch add to category
        self._batch_add_btn = QPushButton("Add Selected to Category")
        self._batch_add_btn.setProperty("flat", True)
        self._batch_add_btn.clicked.connect(self._batch_add_to_category)
        self._batch_add_btn.setEnabled(False)
        path_header.addWidget(self._batch_add_btn)
        left_layout.addLayout(path_header)

        # Table view
        self._table = QTableView()
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._table.setSortingEnabled(True)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)

        # Model will be set after db is ready
        self._model: PathTableModel = None
        left_layout.addWidget(self._table, 1)  # stretch=1: fill remaining space

        # ─── Status ─────────────────────────────────────
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #909399; font-size: 11px;")
        left_layout.addWidget(self._status_label)

        # Connect state signals
        self.state.category_changed.connect(self._refresh_categories)
        self.state.category_visibility_changed.connect(self._update_selected_view_state)
        self.state.effective_paths_changed.connect(self._on_effective_changed)

        layout.addWidget(left_widget, 1)
        layout.addWidget(right_widget, 0)
        self._update_category_action_state()

    def set_db(self, db):
        self.db = db

    def init_model(self):
        """Initialize the table model after db is available."""
        preserved_tab = self._sub_tabs.currentIndex()
        preserved_selection_ids = set(self._manual_selected_path_ids or self.state.effective_path_ids)
        self._model = PathTableModel(self.db)
        self._table_filter_signature = None
        self._table.setModel(self._model)

        # Set column widths
        header = self._table.horizontalHeader()
        for i, w in enumerate(COL_WIDTHS):
            header.resizeSection(i, w)
        header.setSectionResizeMode(0, QHeaderView.Fixed)

        # Connect selection
        sel_model = self._table.selectionModel()
        sel_model.selectionChanged.connect(self._on_table_selection)
        self._table.clicked.connect(self._on_table_clicked)

        self._model.total_count_changed.connect(
            lambda count: self._path_count_label.setText(f"Paths: {count:,}")
        )
        self._model.set_selected_paths(set(self.state.effective_path_ids))
        self._on_sub_tab_changed(preserved_tab)
        if preserved_tab == 0 and preserved_selection_ids:
            QTimer.singleShot(
                0,
                lambda ids=preserved_selection_ids: self._sync_table_selection_from_path_ids(ids),
            )
        self._update_selected_view_state()

    def prune_removed_path_ids(self, path_ids):
        removed = {int(path_id) for path_id in (path_ids or ())}
        if not removed:
            return
        self._manual_selected_path_ids.difference_update(removed)

    def refresh(self):
        """Reload data while respecting the active All Paths / Selected view."""
        self._apply_table_filters(force=True)
        self._update_selected_view_state(refresh_table=False)

    def _apply_table_filters(self, *, force=False):
        if self._model is None:
            return
        selected_ids = self._get_selected_view_path_ids() if self._sub_tabs.currentIndex() == 1 else None
        residues = set(self.state.selected_residues) or None
        signature = (
            self.state.frame_min,
            self.state.frame_max,
            frozenset(residues or ()),
            None if selected_ids is None else frozenset(selected_ids),
        )
        if not force and signature == self._table_filter_signature:
            return

        # A model reset clears Qt's row selection; it must not clear the user's
        # path selection or emit a new 3D preview merely because tabs changed.
        selection_model = self._table.selectionModel()
        was_blocked = selection_model.blockSignals(True)
        try:
            self._model.set_filters(
                frame_min=self.state.frame_min,
                frame_max=self.state.frame_max,
                residue_filter=residues,
                path_id_filter=selected_ids,
            )
        finally:
            selection_model.blockSignals(was_blocked)
        self._table_filter_signature = signature
        self._sync_table_selection_from_path_ids(
            self._manual_selected_path_ids or self.state.effective_path_ids
        )

    def set_highlighted_paths(self, path_ids):
        if self._model:
            self._model.set_highlighted_paths(set(path_ids))

    def selected_path_ids(self) -> list[int]:
        return sorted(int(path_id) for path_id in self._manual_selected_path_ids if int(path_id) > 0)

    # ─── Category management ────────────────────────────
    def _create_category(self):
        name = self._cat_input.text().strip()
        if not name:
            import datetime
            name = f"Category_{datetime.datetime.now().strftime('%H%M%S')}"

        # Use table selection → effective paths → matched paths
        sel_rows = self._table.selectionModel().selectedRows()
        if sel_rows and self._model:
            path_ids = self._model.get_path_ids_for_rows([r.row() for r in sel_rows])
        elif self.state.effective_path_ids:
            path_ids = list(self.state.effective_path_ids)
        elif self.state.matched_path_ids:
            path_ids = self.state.matched_path_ids
        else:
            path_ids = []

        if path_ids:
            self.state.add_category(name, path_ids)
            self._cat_input.clear()
            self._status_label.setText(f"Created '{name}' with {len(path_ids)} paths")

    def _refresh_categories(self):
        self._cat_list.clear()
        from PySide6.QtGui import QPixmap, QIcon, QPainter, QColor as QC, QBrush, QFont
        for name, ids in self.state.categories.items():
            color = self.state.category_colors.get(name, "#4169E1")
            is_visible = name in self.state.visible_categories
            is_excluded = name in self.state.excluded_categories

            # Build label with state indicator
            label = f"  {name} ({len(ids)})"
            if is_visible and is_excluded:
                label += "  [V|X]"
            elif is_visible:
                label += "  [V]"
            elif is_excluded:
                label += "  [X]"

            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, name)

            # Color indicator icon
            pm = QPixmap(12, 12)
            pm.fill(Qt.transparent)
            p = QPainter(pm)
            p.setBrush(QC(color))
            p.setPen(Qt.NoPen)
            p.drawEllipse(0, 0, 12, 12)
            p.end()
            item.setIcon(QIcon(pm))

            # Visual styling based on state
            if is_visible:
                bg = QC(color)
                bg.setAlpha(30)
                item.setBackground(QBrush(bg))
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            if is_excluded:
                item.setForeground(QBrush(QC("#999999")))
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)

            if self.state.active_category == name:
                item.setSelected(True)

            self._cat_list.addItem(item)
        self._update_category_action_state()
        self._update_selected_view_state()

    def _on_category_clicked(self, item):
        name = item.data(Qt.UserRole)
        self.state.set_active_category(name)
        self._update_category_action_state()

    def _on_category_toggle_clicked(self, item):
        name = item.data(Qt.UserRole)
        self.state.set_active_category(name)
        self.state.toggle_category_visible(name)
        self._update_category_action_state()

    def _export_category(self):
        if not self.state.active_category:
            return
        cat_name = self.state.active_category
        path_ids = self.state.categories.get(cat_name, [])
        if path_ids:
            self.export_requested.emit(cat_name, path_ids)

    def _delete_category(self):
        if not self.state.active_category:
            return
        reply = QMessageBox.question(
            self, "Delete Category",
            f"Delete category '{self.state.active_category}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.state.delete_category(self.state.active_category)

    def _on_category_context_menu(self, pos):
        """Right-click context menu for category items."""
        item = self._cat_list.itemAt(pos)
        if not item:
            return
        name = item.data(Qt.UserRole)
        self.state.active_category = name

        is_visible = name in self.state.visible_categories
        is_excluded = name in self.state.excluded_categories

        menu = QMenu(self)
        if is_visible:
            menu.addAction("Hide", lambda: self.state.toggle_category_visible(name))
        else:
            menu.addAction("Show", lambda: self.state.toggle_category_visible(name))

        menu.addSeparator()
        if is_excluded:
            menu.addAction("Remove Exclusion", lambda: self.state.toggle_category_excluded(name))
        else:
            menu.addAction("Exclude from Candidates", lambda: self.state.toggle_category_excluded(name))

        menu.addSeparator()
        menu.addAction("Export", lambda: self._export_specific_category(name))

        del_action = menu.addAction("Delete")
        del_action.triggered.connect(lambda: self._delete_specific_category(name))

        menu.exec_(self._cat_list.mapToGlobal(pos))

    def _choose_category_color(self):
        name = self.state.active_category
        if not name or name not in self.state.categories:
            return
        from PySide6.QtGui import QColor
        initial = QColor(self.state.category_colors.get(name, "#4169E1"))
        color = QColorDialog.getColor(initial, self, "Select Cluster Color")
        if not color.isValid():
            return
        self.state.category_colors[name] = color.name().upper()
        self.state.category_changed.emit()
        self.state.category_visibility_changed.emit()

    def _select_category_paths(self):
        name = self.state.active_category
        if not name or name not in self.state.categories:
            return
        path_ids = [
            int(path_id)
            for path_id in self.state.categories.get(name, [])
            if int(path_id) > 0
        ]
        if not path_ids:
            return
        self.state.apply_lasso(set(path_ids), "replace")
        self._status_label.setText(f"Selected cluster '{name}' ({len(path_ids)} paths)")

    def _toggle_active_category_visibility(self):
        name = self.state.active_category
        if not name or name not in self.state.categories:
            return
        self.state.toggle_category_visible(name)
        self._update_category_action_state()

    def _select_and_show_category_paths(self):
        name = self.state.active_category
        if not name or name not in self.state.categories:
            return
        path_ids = [
            int(path_id)
            for path_id in self.state.categories.get(name, [])
            if int(path_id) > 0
        ]
        if not path_ids:
            return
        if name not in self.state.visible_categories:
            self.state.toggle_category_visible(name)
        self.state.apply_lasso(set(path_ids), "replace")
        self._status_label.setText(f"Selected and displayed cluster '{name}' ({len(path_ids)} paths)")

    def _update_category_action_state(self):
        name = self.state.active_category
        has_active = bool(name and name in self.state.categories)
        for btn in (
            getattr(self, "_cat_export_btn", None),
            getattr(self, "_cat_delete_btn", None),
            getattr(self, "_cat_select_btn", None),
            getattr(self, "_cat_select_show_btn", None),
            getattr(self, "_cat_color_btn", None),
            getattr(self, "_cat_show_btn", None),
        ):
            if btn is not None:
                btn.setEnabled(has_active)
        if getattr(self, "_cat_show_btn", None) is not None:
            if has_active and name in self.state.visible_categories:
                self._cat_show_btn.setText("Hide")
            else:
                self._cat_show_btn.setText("Show")

    def _export_specific_category(self, name: str):
        path_ids = self.state.categories.get(name, [])
        if path_ids:
            self.export_requested.emit(name, path_ids)

    def _delete_specific_category(self, name: str):
        reply = QMessageBox.question(
            self, "Delete Category",
            f"Delete category '{name}'?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.state.delete_category(name)

    def _batch_add_to_category(self):
        """Add selected table rows or lasso-selected paths to active category."""
        if not self.state.active_category or not self._model:
            return
        sel_rows = self._table.selectionModel().selectedRows()
        if sel_rows:
            path_ids = self._model.get_path_ids_for_rows([r.row() for r in sel_rows])
        elif self.state.effective_path_ids:
            path_ids = list(self.state.effective_path_ids)
        else:
            path_ids = []
        if path_ids:
            name = self.state.active_category
            existing = self.state.categories.get(name, [])
            merged = list(set(existing + path_ids))
            self.state.categories[name] = merged
            self.state.category_changed.emit()

    # ─── Table selection ────────────────────────────────
    def _on_table_selection(self, selected, deselected):
        sel_rows = self._table.selectionModel().selectedRows()
        has_selection = len(sel_rows) > 0
        self._batch_add_btn.setEnabled(has_selection)

        if has_selection and self._model:
            path_ids = self._model.get_path_ids_for_rows([r.row() for r in sel_rows])
            self._manual_selected_path_ids = {
                int(path_id)
                for path_id in path_ids
                if int(path_id) > 0
            }
            self.paths_selected.emit(path_ids)
            if len(path_ids) == 1:
                self.path_selected.emit(path_ids[0])
        else:
            self._manual_selected_path_ids.clear()
            self.paths_selected.emit([])
        self._update_selected_view_state(refresh_table=False)

    def _on_table_clicked(self, index):
        if not index.isValid() or not self._model:
            return
        try:
            path_id_col = COL_KEYS.index("id")
        except ValueError:
            path_id_col = 1
        if int(index.column()) != int(path_id_col):
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier | Qt.AltModifier | Qt.MetaModifier):
            return
        path_id = self._model.get_path_id(int(index.row()))
        if path_id is None:
            return
        self.path_selected.emit(int(path_id))

    def _on_effective_changed(self):
        """Unified handler: effective_path_ids changed → sync table + labels."""
        n = len(self.state.effective_path_ids)
        print(f"[PATH_PANEL] synced rows={n}")
        self._manual_selected_path_ids.clear()

        if self._model:
            self._model.set_selected_paths(set(self.state.effective_path_ids))
            if self._sub_tabs.currentIndex() == 0:
                self._sync_table_selection_from_state()
        self._update_selected_view_state()

    # ─── Sub-tab switching ───────────────────────────────
    def _on_sub_tab_changed(self, index):
        """Switch table scope immediately without changing the path selection."""
        self._load_profiles_btn.setVisible(index == 1)
        self.refresh()

    def _on_load_profiles(self):
        """Load charts on request; the Selected table is already up to date."""
        selected = self._get_selected_view_path_ids()
        if not selected or not self._model:
            return
        self.profile_requested.emit(sorted(selected))

    def _get_selected_view_path_ids(self) -> set[int]:
        """Use table selection, effective paths, or the existing category fallback."""
        if self._manual_selected_path_ids:
            return set(self._manual_selected_path_ids)
        if self.state.effective_path_ids:
            return set(self.state.effective_path_ids)
        return self.state.get_visible_category_path_ids()

    def _update_selected_view_state(self, *, refresh_table=True):
        selected = self._get_selected_view_path_ids()
        n = len(selected)
        self._sub_tabs.setTabText(1, f"Selected ({n})")
        self._load_profiles_btn.setEnabled(n > 0 and self._sub_tabs.currentIndex() == 1)

        if self._manual_selected_path_ids:
            self._status_label.setText(f"Table selection: {n} paths")
        elif self.state.effective_path_ids:
            self._status_label.setText(f"Effective: {n} paths" if n else "")
        elif selected:
            self._status_label.setText(f"Visible categories: {n} paths")
        else:
            self._status_label.setText("")
        if refresh_table and self._sub_tabs.currentIndex() == 1:
            self._apply_table_filters()

    def _sync_table_selection_from_state(self):
        """Sync effective_path_ids to current cached table rows."""
        self._sync_table_selection_from_path_ids(set(self.state.effective_path_ids))

    def _sync_table_selection_from_path_ids(self, path_ids):
        """Select requested path IDs among rows currently cached by the virtual table."""
        if not self._model or not self._table.selectionModel():
            return

        row_indices = self._model.get_loaded_row_indices_for_path_ids(
            set(path_ids or ())
        )
        sel_model = self._table.selectionModel()
        sel_model.blockSignals(True)
        sel_model.clearSelection()
        flags = QItemSelectionModel.Select | QItemSelectionModel.Rows
        for row in row_indices:
            idx = self._model.index(row, 0)
            sel_model.select(idx, flags)
        sel_model.blockSignals(False)
        self._batch_add_btn.setEnabled(len(row_indices) > 0)
