"""
VTK-based 3D tunnel path viewer.
Replaces deck.gl DeckGL3D.vue.
"""
import time
import numpy as np
from typing import Optional, List, Dict, Set, Tuple, Sequence

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk, numpy_to_vtkIdTypeArray
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    vtk = None
    vtk_to_numpy = None
    numpy_to_vtk = None
    numpy_to_vtkIdTypeArray = None

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QToolTip
from PySide6.QtCore import Signal, Qt, QEvent, QPoint

from TopoTunnel_UI.vis.lasso import (
    LassoOverlay,
    project_points_to_screen,
    points_in_polygon,
)
from TopoTunnel_UI.vis.residue_display import (
    combination_render_style,
    format_combination_hover_text,
    residue_render_style,
    format_residue_hover_text,
)
from TopoTunnel_UI.vis.protein_model import (
    parse_pdb_atoms,
    build_backbone_data,
    estimate_alignment,
    apply_transform,
    is_hbond_acceptor_atom,
    is_hbond_donor_atom,
    residue_key,
)
from TopoTunnel_UI.vis.cartoon_ribbon import generate_cartoon_ribbon
from TopoTunnel_UI.vis.binding_surface import generate_binding_site_surface


class TunnelInteractorStyle(vtk.vtkInteractorStyleTrackballCamera if HAS_VTK else object):
    """Custom VTK interaction mapping:
    - Left drag: rotate (default)
    - Right drag: pan (remap right -> middle)
    - Wheel: zoom (default)
    """

    def __init__(self, log_cb=None, is_lasso_drawing_cb=None):
        super().__init__()
        self._log_cb = log_cb
        self._is_lasso_drawing_cb = is_lasso_drawing_cb
        self._right_panning = False
        self._pan_move_count = 0

    def _log(self, msg: str):
        if self._log_cb:
            self._log_cb("RMB", msg)

    def OnRightButtonDown(self):
        lasso_drawing = bool(self._is_lasso_drawing_cb and self._is_lasso_drawing_cb())
        self._log(f"OnRightButtonDown triggered, lasso_drawing={lasso_drawing}")
        if lasso_drawing:
            self._log("ignore right-pan start because lasso is drawing")
            return
        self._right_panning = True
        self._pan_move_count = 0
        self.StartPan()
        self._log("StartPan() called")

    def OnMouseMove(self):
        if self._right_panning:
            self._pan_move_count += 1
            self.Pan()
            if self._pan_move_count <= 8 or self._pan_move_count % 30 == 0:
                self._log(f"Pan() called, move_count={self._pan_move_count}")
            return
        super().OnMouseMove()

    def OnRightButtonUp(self):
        self._log(f"OnRightButtonUp triggered, right_panning={self._right_panning}")
        if self._right_panning:
            self.EndPan()
            self._right_panning = False
            self._log("EndPan() called")
            return
        super().OnRightButtonUp()


class TunnelViewer3D(QWidget):
    _HIGHLIGHT_ELEMENT_COLORS = {
        "C": "#39E75F",
        "N": "#4F7DFF",
        "O": "#E24A4A",
        "S": "#F2994A",
        "P": "#F2994A",
        "H": "#F5F5F5",
        "FE": "#C97A2B",
        "MG": "#63C7C7",
        "ZN": "#8D99AE",
        "CA": "#7ED957",
        "CL": "#67C56B",
        "BR": "#A66B3D",
    }
    _HIGHLIGHT_ELEMENT_RADII = {
        # Standard van der Waals radii (Angstrom-like display units).
        # `spheres` style uses these directly so atom size follows element type.
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "FE": 1.94,
        "MG": 1.73,
        "ZN": 1.39,
        "CA": 2.31,
        "CL": 1.75,
        "BR": 1.85,
    }
    _HIGHLIGHT_COVALENT_RADII = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "S": 1.05,
        "P": 1.07,
        "FE": 1.24,
        "MG": 1.30,
        "ZN": 1.22,
        "CA": 1.76,
        "CL": 1.02,
        "BR": 1.20,
    }
    _HIGHLIGHT_DEFAULT_COLOR = "#B8BCC6"
    _HIGHLIGHT_DEFAULT_RADIUS = 1.70
    _HIGHLIGHT_BOND_RADIUS = 0.16
    _HIGHLIGHT_STICK_BOND_RADIUS = 0.20
    _HIGHLIGHT_HBOND_RADIUS = 0.06
    _HIGHLIGHT_BOND_MAX_SCALE = 1.18
    _HIGHLIGHT_SPHERE_THETA_RES = 26
    _HIGHLIGHT_SPHERE_PHI_RES = 26
    _HIGHLIGHT_SPHERE_THETA_RES_LOW = 16
    _HIGHLIGHT_SPHERE_PHI_RES_LOW = 16
    _HIGHLIGHT_BOND_SIDES = 18
    _HIGHLIGHT_BOND_SIDES_LOW = 10
    _HIGHLIGHT_STYLE_RADIUS_SCALE = {
        "ball_and_stick": 0.32,
        "sticks": 0.0,
        "spheres": 1.00,
    }
    """3D visualization of tunnel paths using VTK/PyVista."""

    path_clicked = Signal(int)       # path_id
    paths_selected = Signal(list)    # list of path_ids (from box selection)
    residue_clicked = Signal(int)    # residue_id
    combination_clicked = Signal(object)  # marker payload
    lasso_selected = Signal(object, str)  # (set of path_ids, mode)
    exit_cluster_selected = Signal(object)  # set of path_ids from connected exit spheres
    path_coordinate_mode_changed = Signal(str)

    # Actor name prefixes for layer management
    _BG_PREFIX = "bg_path_"
    _MATCH_PREFIX = "match_path_"
    _MARKER_PREFIX = "marker_"
    _PROTEIN_PREFIX = "protein_"
    _CAT_PREFIX = "category_"
    _DATASET_PREFIX = "dataset_"
    _ENTRYEXIT_PREFIX = "entryexit_"
    _CLUSTER_LABEL_PREFIX = "clusterlabel_"
    _BINDING_PREFIX = "binding_site_"
    _PROTEIN_CARTOON_COLOR = "#39E75F"

    # Lasso interaction constants
    _LASSO_MIN_DIST_SQ = 16  # 4px squared
    _LASSO_MAX_SAMPLES_PER_PATH = 256
    _LASSO_MIDPOINT_ONLY_THRESHOLD = 20000
    _ORIENTATION_VIEWPORT = (0.02, 0.02, 0.18, 0.18)
    _ORIENTATION_BG_RGB = (0.92, 0.92, 0.92)
    _ORIENTATION_OUTLINE_RGB = (0.72, 0.70, 0.64)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path_coordinate_mode = "current"
        self._path_mode_btn: Optional[QPushButton] = None
        self._path_coordinate_mode_busy = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not HAS_VTK:
            label = QLabel("VTK/PyVista not installed.\npip install pyvista pyvistaqt vtk")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
            self._plotter = None
            return

        self._plotter = QtInteractor(self)
        self._plotter.set_background("white")
        layout.addWidget(self._plotter)
        self._path_mode_btn = QPushButton(self)
        self._path_mode_btn.setCursor(Qt.PointingHandCursor)
        self._path_mode_btn.setCheckable(True)
        self._path_mode_btn.clicked.connect(self._toggle_path_coordinate_mode)
        self._path_mode_btn.setStyleSheet(
            "QPushButton {"
            "background: rgba(255, 255, 255, 230);"
            "border: 1px solid rgba(64, 77, 104, 72);"
            "border-radius: 8px;"
            "padding: 4px 10px;"
            "font-size: 11px;"
            "font-weight: 600;"
            "color: #344054;"
            "}"
            "QPushButton:hover { background: rgba(255, 255, 255, 245); }"
            "QPushButton:checked {"
            "background: rgba(237, 247, 255, 238);"
            "border-color: rgba(59, 130, 246, 120);"
            "color: #1d4ed8;"
            "}"
        )
        self.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)
        self._path_mode_btn.raise_()

        # Use custom interaction mapping: right-drag pans instead of zooming.
        self._custom_style = TunnelInteractorStyle(
            log_cb=self._log_lasso,
            is_lasso_drawing_cb=lambda: self._lasso_drawing,
        )
        # QtInteractor proxies VTK methods via __getattr__, so SetInteractorStyle
        # is applied to the underlying vtkGenericRenderWindowInteractor.
        self._plotter.SetInteractorStyle(self._custom_style)
        self._plotter.setContextMenuPolicy(Qt.NoContextMenu)
        self._orientation_axes_actor = None
        self._orientation_widget = None
        self._setup_orientation_widget()

        # Track actors for efficient updates
        self._bg_actors = []
        self._match_actors = []
        self._marker_actors = []
        self._protein_actors = []
        self._protein_highlight_actors = []
        self._protein_highlight_actor_map: Dict[str, object] = {}
        self._protein_highlight_pipeline_map: Dict[str, dict] = {}

        # Protein model state
        self._protein_visible = True
        self._protein_style = "cartoon"
        self._protein_opacity = 1.0
        self._protein_highlight_style = "ball_and_stick"
        self._protein_highlight_quality = "high"
        self._protein_highlight_render_enabled = True
        self._protein_highlight_mesh_payload: Optional[dict] = None
        self._protein_source_path: Optional[str] = None
        self._protein_last_alignment_diag: Optional[dict] = None
        # Cache raw parse + aligned geometry for style switching without re-parsing.
        self._protein_model_cache: Optional[dict] = None
        self._protein_highlight_residue_ids: Tuple[int, ...] = ()

        # Binding site state
        self._binding_site_actors: list = []
        self._binding_site_visible: bool = True
        self._binding_site_cache: Optional[List[dict]] = None  # [{mol, surface, color}]

        # Residue position data: {residue_id: (x, y, z)}
        self._residue_coords: Dict[int, np.ndarray] = {}
        self._residue_info: Dict[int, dict] = {}
        self._residue_visible = False
        self._hovered_residue_id: Optional[int] = None
        self._combination_marker_coords: Dict[str, np.ndarray] = {}
        self._combination_marker_info: Dict[str, dict] = {}
        self._combination_visible = False
        self._hovered_combination_key: Optional[str] = None
        self._selected_combination_key: Optional[str] = None

        # Data caches
        self._bg_mesh = None
        self._bg_cell_path_ids: Optional[np.ndarray] = None  # cell_id -> path_id
        self._view_initialized = False

        # Cached background data for highlight mesh building
        self._bg_data_dict: Dict[int, np.ndarray] = {}
        self._bg_midpoint_dict: Dict[int, np.ndarray] = {}
        self._last_lasso_elapsed_ms: Optional[float] = None
        self._bg_opacity: float = 0.15
        self._bg_color: str = "#505050"
        self._bg_change_metric_map: Optional[Dict[int, float]] = None
        self._bg_change_metric_cmap: str = "Reds"
        self._bg_change_metric_clim_percentile: Optional[float] = None
        self._excluded_path_ids: Set[int] = set()
        self._bg_exclusion_signature: Tuple[int, ...] | None = None
        self._focus_mode: bool = False
        self._effective_path_ids: Set[int] = set()
        self._effective_render_signature: tuple | None = None
        self._category_render_signature: tuple | None = None
        self._dataset_render_signature: tuple | None = None
        self._entry_exit_render_signature: tuple | None = None
        self._render_suspend_depth = 0
        self._render_pending = False

        # Entry/exit point caches (derived from bg_data_dict)
        self._entry_points: Dict[int, np.ndarray] = {}  # path_id -> start xyz
        self._exit_points: Dict[int, np.ndarray] = {}   # path_id -> end xyz
        self._entry_exit_visible = False
        self._rendered_entry_exit_counts = {"entries": 0, "exits": 0}
        self._entry_exit_display_ids: Optional[Set[int]] = None
        self._entry_exit_effective_ids: Set[int] = set()
        self._exit_cluster_select_mode = False

        # Selection mode: 'path' or 'entrance_exit'
        self._selection_mode: str = "path"

        # Lasso overlay + mode
        self._lasso_overlay = LassoOverlay(self._plotter)
        self._lasso_overlay.resize(self._plotter.size())
        self._lasso_overlay.raise_()

        self._lasso_mode = False
        self._lasso_drawing = False
        self._lasso_points: list[tuple[float, float]] = []
        self._lasso_modifier: Optional[str] = None
        self._lasso_button: Optional[Qt.MouseButton] = None  # which button started lasso
        self._right_drag_active = False
        self._right_move_count = 0
        self._left_press_pos: Optional[Tuple[int, int]] = None  # for residue click detection

        # Install event filter on the plotter (and any child that receives mouse)
        self.setMouseTracking(True)
        self._plotter.setMouseTracking(True)
        self._plotter.installEventFilter(self)
        for child in self._plotter.findChildren(QWidget):
            child.installEventFilter(self)
            child.setContextMenuPolicy(Qt.NoContextMenu)
            child.setMouseTracking(True)

        self._closing = False
        self._update_overlay_controls_geometry()

    def begin_render_batch(self) -> None:
        self._render_suspend_depth += 1

    def end_render_batch(self, *, render: bool = True) -> None:
        if self._render_suspend_depth > 0:
            self._render_suspend_depth -= 1
        if self._render_suspend_depth == 0 and render and self._render_pending:
            self._render_pending = False
            if self._plotter:
                self._plotter.render()

    def _request_render(self) -> None:
        if not self._plotter:
            return
        if self._render_suspend_depth > 0:
            self._render_pending = True
            return
        self._plotter.render()

    def _path_payload_signature(self, payload: Dict[str, tuple]) -> tuple:
        result = []
        for key, value in (payload or {}).items():
            path_ids, color = value
            result.append((str(key), tuple(sorted(int(path_id) for path_id in (path_ids or ()))), str(color)))
        return tuple(sorted(result))

    def _setup_orientation_widget(self):
        if not self._plotter or vtk is None:
            return
        try:
            self._orientation_axes_actor = self._plotter.add_axes(
                interactive=False,
                line_width=5,
                labels_off=True,
                x_color="#F26B6B",
                y_color="#67BE76",
                z_color="#5F63E6",
                cone_radius=0.55,
                shaft_length=0.72,
                tip_length=0.28,
                ambient=0.85,
                viewport=self._ORIENTATION_VIEWPORT,
            )
            self._orientation_widget = getattr(
                self._plotter.renderer,
                "axes_widget",
                None,
            )
            if self._orientation_widget is None:
                return
            self._orientation_widget.InteractiveOff()
            self._orientation_widget.SetOutlineColor(*self._ORIENTATION_OUTLINE_RGB)
            marker_renderer = self._orientation_widget.GetRenderer()
            if marker_renderer is not None:
                marker_renderer.GradientBackgroundOff()
                marker_renderer.SetBackground(*self._ORIENTATION_BG_RGB)
        except Exception as exc:
            self._orientation_axes_actor = None
            self._orientation_widget = None
            print(f"[VIEW] orientation widget setup skipped: {exc}")

    def _log_lasso(self, stage: str, message: str):
        print(f"[LASSO][{stage}] {message}")

    # ─── Lasso mode ──────────────────────────────────────

    @property
    def lasso_mode(self) -> bool:
        return self._lasso_mode

    def set_lasso_mode(self, enabled: bool):
        self._lasso_mode = enabled
        if enabled:
            self._plotter.setCursor(Qt.CrossCursor)
        else:
            self._plotter.setCursor(Qt.ArrowCursor)
            self.clear_lasso_overlay("mode_off")

    def set_selection_mode(self, mode: str):
        self._selection_mode = mode if mode in ("path", "entrance_exit") else "path"

    def set_exit_cluster_select_mode(self, enabled: bool):
        self._exit_cluster_select_mode = bool(enabled)
        if enabled:
            self._plotter.setCursor(Qt.CrossCursor)
        elif not self._lasso_mode:
            self._plotter.setCursor(Qt.ArrowCursor)

    def set_focus_mode(self, enabled: bool):
        self._focus_mode = bool(enabled)

    def clear_lasso_overlay(self, reason: str = ""):
        """Clear overlay in all lasso exit paths.

        This only updates the Qt overlay widget and does not touch VTK camera.
        """
        if reason:
            self._log_lasso("LASSO_CLEAR", f"reason={reason}")
        self._lasso_drawing = False
        self._lasso_button = None
        self._lasso_points.clear()
        self._lasso_overlay.clear()

    def _toggle_path_coordinate_mode(self, _checked: bool = False):
        if self._path_coordinate_mode_busy:
            return
        mode = "original" if self._path_coordinate_mode == "current" else "current"
        self.set_path_coordinate_mode(mode, emit_signal=True)

    def set_path_coordinate_mode(self, mode: str, *, emit_signal: bool = False):
        mode = "original" if mode == "original" else "current"
        changed = self._path_coordinate_mode != mode
        self._path_coordinate_mode = mode
        if self._path_mode_btn is not None:
            was_blocked = self._path_mode_btn.blockSignals(True)
            self._path_mode_btn.setChecked(mode == "original")
            if mode == "original":
                self._path_mode_btn.setText("Original Paths")
                self._path_mode_btn.setToolTip("Switch 3D rendering to the original export coordinates")
            else:
                self._path_mode_btn.setText("Current Paths")
                self._path_mode_btn.setToolTip("Switch 3D rendering to the current processed coordinates")
            self._path_mode_btn.adjustSize()
            self._path_mode_btn.blockSignals(was_blocked)
            self._update_overlay_controls_geometry()
        if emit_signal and changed:
            self.path_coordinate_mode_changed.emit(self._path_coordinate_mode)

    def set_path_coordinate_mode_busy(self, busy: bool):
        self._path_coordinate_mode_busy = bool(busy)
        if self._path_mode_btn is None:
            return
        self._path_mode_btn.setEnabled(not self._path_coordinate_mode_busy)
        if self._path_coordinate_mode_busy:
            self._path_mode_btn.setText("Loading...")
            self._path_mode_btn.setToolTip("Path coordinate mode is loading")
            self._path_mode_btn.adjustSize()
            self._update_overlay_controls_geometry()
        else:
            self.set_path_coordinate_mode(self._path_coordinate_mode, emit_signal=False)

    def _update_overlay_controls_geometry(self):
        if self._path_mode_btn is None:
            return
        width = max(118, self._path_mode_btn.sizeHint().width())
        self._path_mode_btn.resize(width, 30)
        margin = 12
        self._path_mode_btn.move(
            max(margin, self.width() - width - margin),
            margin,
        )
        self._path_mode_btn.raise_()

    def _event_pos_in_plotter(self, obj, event) -> QPoint:
        """Map a mouse event to plotter-local coordinates.

        Using global cursor coordinates is more stable than chaining local
        widget mappings when the VTK interactor is backed by nested/native
        child widgets or when high-DPI scaling is active.
        """
        if hasattr(event, "globalPosition"):
            return self._plotter.mapFromGlobal(event.globalPosition().toPoint())
        return obj.mapTo(self._plotter, event.pos())

    # ─── Event filter (lasso intercept) ──────────────────

    def eventFilter(self, obj, event):
        if not self._plotter:
            return False

        etype = event.type()

        if etype in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.FocusOut,
            QEvent.Type.WindowDeactivate,
            QEvent.Type.Hide,
            QEvent.Type.Leave,
        ):
            self._hide_residue_hover_tooltip()
            self._hide_combination_hover_tooltip()

        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.RightButton:
            mods = event.modifiers()
            if mods & Qt.ShiftModifier:
                # Shift+RMB → lasso remove selection
                pos = self._event_pos_in_plotter(obj, event)
                self._lasso_drawing = True
                self._lasso_button = Qt.RightButton
                self._lasso_points = [(pos.x(), pos.y())]
                self._lasso_modifier = "remove"
                self._lasso_overlay.set_points(self._lasso_points)
                self._log_lasso("RMB", "Shift+RMB lasso-remove started")
                return True
            self._right_drag_active = True
            self._right_move_count = 0
            self._log_lasso(
                "RMB",
                f"RightButtonPress detected, lasso_drawing={self._lasso_drawing}",
            )
            return False

        if etype == QEvent.Type.MouseMove and self._right_drag_active and not self._lasso_drawing and (event.buttons() & Qt.RightButton):
            self._right_move_count += 1
            if self._right_move_count <= 8 or self._right_move_count % 30 == 0:
                self._log_lasso(
                    "RMB",
                    f"MouseMove with RMB, count={self._right_move_count}, lasso_drawing={self._lasso_drawing}",
                )
            return False

        if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.RightButton:
            if self._lasso_drawing and self._lasso_button == Qt.RightButton:
                # Finish Shift+RMB lasso-remove
                self._log_lasso("RMB", f"Shift+RMB lasso release, points={len(self._lasso_points)}")
                if len(self._lasso_points) >= 3:
                    poly = np.asarray(self._lasso_points, dtype=np.float64)
                    pmin = poly.min(axis=0)
                    pmax = poly.max(axis=0)
                    bbox_area = max(0.0, float(pmax[0] - pmin[0])) * max(0.0, float(pmax[1] - pmin[1]))
                    self._log_lasso("RMB", f"lasso-remove bbox_area={bbox_area:.1f}")
                    self._lasso_overlay.set_points(self._lasso_points, closed=True)
                    self._execute_lasso_selection()
                    self.clear_lasso_overlay("rmb_release_done")
                else:
                    self.clear_lasso_overlay("rmb_release_too_few_points")
                self._lasso_button = None
                return True
            self._log_lasso(
                "RMB",
                f"RightButtonRelease detected, move_count={self._right_move_count}, lasso_drawing={self._lasso_drawing}",
            )
            self._right_drag_active = False
            self._right_move_count = 0
            return False

        if etype in (QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate, QEvent.Type.Hide):
            if self._lasso_drawing or self._lasso_points:
                self.clear_lasso_overlay("focus_or_window_deactivate")
            return False

        if etype == QEvent.Type.KeyPress and event.key() == Qt.Key_Escape:
            self.clear_lasso_overlay("esc")
            return False  # propagate to MainWindow QShortcut for state clearing

        if etype == QEvent.Type.MouseButtonPress and event.button() == Qt.LeftButton:
            mods = event.modifiers()
            # Shift+LMB should always trigger lasso (even if lasso toggle is off).
            # If lasso toggle is on, plain LMB also triggers lasso (force mode).
            should_start_lasso = self._lasso_mode or bool(mods & Qt.ShiftModifier)
            if not should_start_lasso:
                # Record press position for residue pick on release
                pos = self._event_pos_in_plotter(obj, event)
                self._left_press_pos = (pos.x(), pos.y())
                return False
            pos = self._event_pos_in_plotter(obj, event)
            self._lasso_drawing = True
            self._lasso_button = Qt.LeftButton
            self._lasso_points = [(pos.x(), pos.y())]
            if mods & Qt.AltModifier:
                self._lasso_modifier = "remove"
            elif mods & Qt.ShiftModifier:
                self._lasso_modifier = "add"
            else:
                self._lasso_modifier = "replace"
            self._lasso_overlay.set_points(self._lasso_points)
            return True

        if etype == QEvent.Type.MouseMove and self._lasso_drawing:
            pos = self._event_pos_in_plotter(obj, event)
            px, py = pos.x(), pos.y()
            lx, ly = self._lasso_points[-1]
            if (px - lx) ** 2 + (py - ly) ** 2 >= self._LASSO_MIN_DIST_SQ:
                self._lasso_points.append((px, py))
                self._lasso_overlay.set_points(self._lasso_points)
            return True

        if (
            etype == QEvent.Type.MouseMove
            and not self._lasso_drawing
            and not event.buttons()
        ):
            pos = self._event_pos_in_plotter(obj, event)
            self._update_marker_hover_tooltip(
                obj,
                event.pos(),
                pos.x(),
                pos.y(),
            )
            return False

        # Plain LMB release (no lasso) — check residue pick
        if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton and not self._lasso_drawing and self._left_press_pos is not None:
            pos = self._event_pos_in_plotter(obj, event)
            dx = pos.x() - self._left_press_pos[0]
            dy = pos.y() - self._left_press_pos[1]
            self._left_press_pos = None
            if dx * dx + dy * dy < 25:  # click, not drag (5px threshold)
                if self._exit_cluster_select_mode:
                    selected_exit_ids = self.pick_connected_exit_cluster_at(pos.x(), pos.y())
                    if selected_exit_ids:
                        self.exit_cluster_selected.emit(selected_exit_ids)
                    else:
                        self.exit_cluster_selected.emit(set())
                    return False
                combo_info = self.pick_combination_at(pos.x(), pos.y())
                if combo_info is not None:
                    self.set_selected_combination(str(combo_info.get("marker_key", "")) or None)
                    self.combination_clicked.emit(combo_info)
                    return False
                rid = self.pick_residue_at(pos.x(), pos.y())
                if rid is not None:
                    self.residue_clicked.emit(rid)
                    return False
                pid = self.pick_path_at(pos.x(), pos.y())
                if pid is not None:
                    self.path_clicked.emit(pid)
                    return False
            return False

        if etype == QEvent.Type.MouseButtonRelease and event.button() == Qt.LeftButton and self._lasso_drawing and self._lasso_button == Qt.LeftButton:
            if len(self._lasso_points) >= 3:
                poly = np.asarray(self._lasso_points, dtype=np.float64)
                pmin = poly.min(axis=0)
                pmax = poly.max(axis=0)
                bbox_w = max(0.0, float(pmax[0] - pmin[0]))
                bbox_h = max(0.0, float(pmax[1] - pmin[1]))
                bbox_area = bbox_w * bbox_h
                close_dist = float(np.linalg.norm(poly[0] - poly[-1]))
                self._log_lasso(
                    "L1",
                    f"release points={len(self._lasso_points)}, "
                    f"bbox=({pmin[0]:.1f},{pmin[1]:.1f})-({pmax[0]:.1f},{pmax[1]:.1f}), "
                    f"bbox_area={bbox_area:.1f}, close_dist={close_dist:.2f}",
                )
                if len(self._lasso_points) < 10 or bbox_area < 100.0:
                    self._log_lasso("L1", "WARNING: polygon points too few or bbox too small")
                self._lasso_overlay.set_points(self._lasso_points, closed=True)
                self._execute_lasso_selection()
                self.clear_lasso_overlay("release_done")
            else:
                self._log_lasso("L1", f"release ignored: points={len(self._lasso_points)} < 3")
                self.clear_lasso_overlay("release_too_few_points")
            self._lasso_button = None
            return True

        return False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_lasso_overlay") and self._lasso_overlay is not None:
            self._lasso_overlay.resize(self._plotter.size())
        self._update_overlay_controls_geometry()

    def _capture_camera_state(self) -> Optional[dict]:
        if not self._plotter:
            return None
        renderer = self._plotter.renderer
        cam = renderer.GetActiveCamera() if renderer else None
        if cam is None:
            return None
        return {
            "position": cam.GetPosition(),
            "focal_point": cam.GetFocalPoint(),
            "view_up": cam.GetViewUp(),
            "clipping_range": cam.GetClippingRange(),
            "parallel_projection": bool(cam.GetParallelProjection()),
            "parallel_scale": cam.GetParallelScale(),
            "view_angle": cam.GetViewAngle(),
        }

    def _restore_camera_state(self, state: Optional[dict]):
        if not state or not self._plotter:
            return
        renderer = self._plotter.renderer
        cam = renderer.GetActiveCamera() if renderer else None
        if cam is None:
            return
        cam.SetPosition(*state["position"])
        cam.SetFocalPoint(*state["focal_point"])
        cam.SetViewUp(*state["view_up"])
        cam.SetClippingRange(*state["clipping_range"])
        if state["parallel_projection"]:
            cam.ParallelProjectionOn()
            cam.SetParallelScale(state["parallel_scale"])
        else:
            cam.ParallelProjectionOff()
            cam.SetViewAngle(state["view_angle"])

    # ─── Lasso selection core ────────────────────────────

    def _execute_lasso_selection(self):
        if self._selection_mode == "entrance_exit":
            self._execute_entry_exit_lasso()
            return

        if self._bg_mesh is None or not self._bg_data_dict:
            self._log_lasso("L2", "skip selection: background mesh/data missing")
            return

        t0 = time.perf_counter()
        polygon = np.array(self._lasso_points, dtype=np.float64)
        if polygon.shape[0] < 3:
            self._log_lasso("L1", f"skip selection: polygon point count={polygon.shape[0]}")
            return

        # Polygon screen bbox (Qt coordinates)
        poly_min = polygon.min(axis=0)
        poly_max = polygon.max(axis=0)

        x0 = int(max(0, np.floor(poly_min[0])))
        y0 = int(max(0, np.floor(poly_min[1])))
        x1 = int(min(self._plotter.width() - 1, np.ceil(poly_max[0])))
        y1 = int(min(self._plotter.height() - 1, np.ceil(poly_max[1])))

        path_arr_exists = bool(
            self._bg_mesh is not None and
            "path_id" in self._bg_mesh.cell_data
        )
        cell_count = int(self._bg_mesh.n_cells) if self._bg_mesh is not None else 0
        path_arr_len = int(len(self._bg_mesh.cell_data["path_id"])) if path_arr_exists else 0
        self._log_lasso(
            "L3",
            f"mesh_cells={cell_count}, has_cell_path_id={path_arr_exists}, "
            f"path_id_len={path_arr_len}, len_match={path_arr_len == cell_count}",
        )

        # In Focus mode the background actor is hidden, so VTK area picking on the
        # background mesh can return an empty set even though effective paths are visible.
        # In that state, refine directly against the currently visible effective paths.
        if self._focus_mode and self._effective_path_ids:
            candidate_ids = set(self._effective_path_ids)
            pick_diag = {
                "method": "focus_effective",
                "selection_is_none": False,
                "selection_nodes": 1,
                "candidate_cell_count": len(candidate_ids),
                "candidate_path_count": len(candidate_ids),
                "cell_path_samples": [],
            }
        else:
            # Stage 1: VTK candidate picking to shrink candidate set.
            candidate_ids, pick_diag = self._pick_candidate_paths_from_rect(x0, y0, x1, y1)
        self._log_lasso(
            "L2",
            f"method={pick_diag.get('method')}, selection_is_none={pick_diag.get('selection_is_none')}, "
            f"selection_nodes={pick_diag.get('selection_nodes')}, "
            f"candidate_cells={pick_diag.get('candidate_cell_count')}, "
            f"candidate_paths={pick_diag.get('candidate_path_count')}",
        )
        sample_pairs = pick_diag.get("cell_path_samples", [])
        if sample_pairs:
            self._log_lasso("L3", f"cell->path sample={sample_pairs}")

        if not candidate_ids:
            self._log_lasso("L4", "candidate set is empty after picking")
            self.lasso_selected.emit(set(), self._lasso_modifier or "replace")
            return

        # Stage 2: precise polygon-in-screen test on candidates only.
        selected_ids, refine_diag = self._polygon_refine_candidates(
            candidate_ids, polygon, poly_min, poly_max
        )
        self._log_lasso(
            "L4",
            f"refine_mode={refine_diag.get('mode')}, projected_range={refine_diag.get('projected_range')}, "
            f"visible_ratio={refine_diag.get('visible_ratio')}, hits={refine_diag.get('hits')}",
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._last_lasso_elapsed_ms = elapsed_ms
        self._log_lasso(
            "DONE",
            f"selected={len(selected_ids)}, elapsed_ms={elapsed_ms:.1f}, "
            f"candidates={len(candidate_ids)}, total_paths={len(self._bg_data_dict)}",
        )

        self.lasso_selected.emit(selected_ids, self._lasso_modifier or "replace")

    def _pick_candidate_paths_from_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
    ) -> Tuple[Set[int], dict]:
        """Use VTK selector to get candidate cells/path_ids.

        Coordinate convention:
        - Qt mouse coordinates: origin at top-left.
        - VTK display coordinates: origin at bottom-left.
        We convert Qt rect -> VTK display rect before calling selector.
        """
        diag = {
            "method": "none",
            "selection_is_none": True,
            "selection_nodes": 0,
            "candidate_cell_count": 0,
            "candidate_path_count": 0,
            "cell_path_samples": [],
        }

        if not HAS_VTK or vtk is None or self._bg_mesh is None:
            return set(), diag
        if x1 <= x0 or y1 <= y0:
            return set(), diag

        # Force latest camera/viewport state before selection.
        self._plotter.render()

        rw = getattr(self._plotter, "ren_win", None)
        if rw is not None:
            disp_w, disp_h = rw.GetSize()
        else:
            disp_w, disp_h = self._plotter.width(), self._plotter.height()
        disp_w = max(1, int(disp_w))
        disp_h = max(1, int(disp_h))
        qt_w = max(1, int(self._plotter.width()))
        qt_h = max(1, int(self._plotter.height()))
        sx = disp_w / qt_w
        sy = disp_h / qt_h

        dx0 = int(np.floor(min(x0, x1) * sx))
        dx1 = int(np.ceil(max(x0, x1) * sx))
        dy0_qt = int(np.floor(min(y0, y1) * sy))
        dy1_qt = int(np.ceil(max(y0, y1) * sy))

        # Qt(top-left) -> VTK(bottom-left) Y flip.
        dy0 = max(0, disp_h - 1 - dy1_qt)
        dy1 = min(disp_h - 1, disp_h - 1 - dy0_qt)
        dx0 = max(0, min(disp_w - 1, dx0))
        dx1 = max(0, min(disp_w - 1, dx1))
        if dx0 > dx1:
            dx0, dx1 = dx1, dx0
        if dy0 > dy1:
            dy0, dy1 = dy1, dy0

        candidate_ids: Set[int] = set()
        picker = vtk.vtkAreaPicker()
        ok = picker.AreaPick(dx0, dy0, dx1, dy1, self._plotter.renderer)
        if ok == 0:
            diag["method"] = "frustum"
            diag["selection_is_none"] = True
            return set(), diag

        frustum = picker.GetFrustum()
        if frustum is None:
            diag["method"] = "frustum"
            diag["selection_is_none"] = True
            return set(), diag

        extractor = vtk.vtkExtractSelectedFrustum()
        extractor.SetFrustum(frustum)
        extractor.SetInputData(self._bg_mesh)
        extractor.PreserveTopologyOff()
        extractor.Update()

        out = extractor.GetOutput()
        if out is None or out.GetNumberOfCells() == 0:
            diag["method"] = "frustum"
            diag["selection_is_none"] = False
            diag["selection_nodes"] = 0
            return set(), diag

        path_arr = out.GetCellData().GetArray("path_id")
        if path_arr is None or path_arr.GetNumberOfTuples() == 0:
            diag["method"] = "frustum"
            diag["selection_is_none"] = False
            return set(), diag

        path_ids = np.asarray(vtk_to_numpy(path_arr), dtype=np.int64).ravel()
        candidate_ids = set(path_ids.tolist())
        diag["method"] = "frustum"
        diag["selection_is_none"] = False
        diag["selection_nodes"] = 1
        diag["candidate_cell_count"] = int(out.GetNumberOfCells())
        diag["candidate_path_count"] = int(len(candidate_ids))
        diag["cell_path_samples"] = [
            (int(i), int(path_ids[i])) for i in range(min(10, len(path_ids)))
        ]
        return candidate_ids, diag

    def _sample_path_for_lasso(self, coords: np.ndarray) -> np.ndarray:
        """Downsample long polylines for screen-space polygon testing."""
        n = len(coords)
        if n <= self._LASSO_MAX_SAMPLES_PER_PATH:
            return coords
        step = max(1, n // self._LASSO_MAX_SAMPLES_PER_PATH)
        sampled = coords[::step]
        if (n - 1) % step != 0:
            sampled = np.vstack((sampled, coords[-1:]))
        return sampled

    def _polygon_refine_candidates(
        self,
        candidate_ids: Set[int],
        polygon: np.ndarray,
        poly_min: np.ndarray,
        poly_max: np.ndarray,
    ) -> Tuple[Set[int], dict]:
        """Project candidate paths to screen and keep paths inside lasso polygon."""
        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())
        diag = {
            "mode": "full",
            "projected_range": "n/a",
            "visible_ratio": 0.0,
            "hits": 0,
        }

        if len(candidate_ids) >= self._LASSO_MIDPOINT_ONLY_THRESHOLD:
            diag["mode"] = "midpoint"
            mids = []
            mid_ids = []
            for pid in candidate_ids:
                mid = self._bg_midpoint_dict.get(int(pid))
                if mid is None:
                    continue
                mids.append(mid)
                mid_ids.append(int(pid))
            if not mids:
                return set(), diag

            mid_world = np.asarray(mids, dtype=np.float64)
            mid_screen = project_points_to_screen(mid_world, renderer, w, h)
            diag["projected_range"] = (
                f"x[{float(np.min(mid_screen[:, 0])):.1f},{float(np.max(mid_screen[:, 0])):.1f}] "
                f"y[{float(np.min(mid_screen[:, 1])):.1f},{float(np.max(mid_screen[:, 1])):.1f}]"
            )
            in_view = (
                (mid_screen[:, 0] >= 0.0) & (mid_screen[:, 0] <= float(w)) &
                (mid_screen[:, 1] >= 0.0) & (mid_screen[:, 1] <= float(h))
            )
            diag["visible_ratio"] = float(np.mean(in_view)) if len(in_view) > 0 else 0.0
            in_bbox = (
                (mid_screen[:, 0] >= poly_min[0]) & (mid_screen[:, 0] <= poly_max[0]) &
                (mid_screen[:, 1] >= poly_min[1]) & (mid_screen[:, 1] <= poly_max[1])
            )
            if not np.any(in_bbox):
                return set(), diag

            inside = points_in_polygon(mid_screen[in_bbox], polygon)
            selected_idx = np.where(in_bbox)[0][inside]
            selected = {mid_ids[i] for i in selected_idx}
            diag["hits"] = int(len(selected))
            return selected, diag

        candidate_list: list[int] = []
        samples: list[np.ndarray] = []
        counts: list[int] = []

        for pid in candidate_ids:
            coords = self._bg_data_dict.get(int(pid))
            if coords is None or len(coords) < 2:
                continue
            sampled = self._sample_path_for_lasso(coords)
            candidate_list.append(int(pid))
            samples.append(sampled.astype(np.float64, copy=False))
            counts.append(len(sampled))

        if not samples:
            return set(), diag

        all_world = np.vstack(samples)
        all_screen = project_points_to_screen(all_world, renderer, w, h)
        diag["projected_range"] = (
            f"x[{float(np.min(all_screen[:, 0])):.1f},{float(np.max(all_screen[:, 0])):.1f}] "
            f"y[{float(np.min(all_screen[:, 1])):.1f},{float(np.max(all_screen[:, 1])):.1f}]"
        )
        in_view = (
            (all_screen[:, 0] >= 0.0) & (all_screen[:, 0] <= float(w)) &
            (all_screen[:, 1] >= 0.0) & (all_screen[:, 1] <= float(h))
        )
        diag["visible_ratio"] = float(np.mean(in_view)) if len(in_view) > 0 else 0.0

        selected: Set[int] = set()
        offset = 0
        for pid, cnt in zip(candidate_list, counts):
            pts = all_screen[offset:offset + cnt]
            offset += cnt

            in_bbox = (
                (pts[:, 0] >= poly_min[0]) & (pts[:, 0] <= poly_max[0]) &
                (pts[:, 1] >= poly_min[1]) & (pts[:, 1] <= poly_max[1])
            )
            if not np.any(in_bbox):
                continue

            if np.any(points_in_polygon(pts[in_bbox], polygon)):
                selected.add(pid)

        diag["hits"] = int(len(selected))
        return selected, diag

    # ─── Effective paths rendering ────────────────────────

    def render_effective_paths(
        self,
        path_ids: Set[int],
        color: str = "#00CC96",
        color_map: Optional[Dict[int, str]] = None,
    ):
        """Render effective paths, preserving per-path colors when provided."""
        signature = (
            tuple(sorted(int(path_id) for path_id in (path_ids or set()))),
            tuple(sorted((int(path_id), str(path_color)) for path_id, path_color in (color_map or {}).items())),
            str(color),
        )
        if signature == self._effective_render_signature:
            return
        self._effective_render_signature = signature
        cam_state = self._capture_camera_state()
        self._effective_path_ids = set(path_ids)
        self._remove_effective(render=False, clear_signature=False)
        self._remove_selection_highlight(render=False)
        self._remove_matched(render=False)

        if not self._plotter or not path_ids:
            self._restore_camera_state(cam_state)
            if self._plotter:
                self._request_render()
            return

        grouped_points: Dict[str, list] = {}
        grouped_lines: Dict[str, list] = {}
        grouped_offsets: Dict[str, int] = {}
        rendered_count = 0
        for pid in path_ids:
            coords = self._bg_data_dict.get(pid)
            if coords is None or len(coords) < 2:
                continue
            path_color = str((color_map or {}).get(int(pid), color))
            offset = grouped_offsets.get(path_color, 0)
            n = len(coords)
            grouped_points.setdefault(path_color, []).append(coords)
            line = np.empty(n + 1, dtype=np.int64)
            line[0] = n
            line[1:] = np.arange(offset, offset + n)
            grouped_lines.setdefault(path_color, []).append(line)
            grouped_offsets[path_color] = offset + n
            rendered_count += 1

        if not grouped_points:
            self._restore_camera_state(cam_state)
            self._request_render()
            return

        for idx, (path_color, point_groups) in enumerate(grouped_points.items()):
            mesh = pv.PolyData()
            mesh.points = np.vstack(point_groups).astype(np.float64)
            mesh.lines = np.concatenate(grouped_lines.get(path_color, []))
            self._plotter.add_mesh(
                mesh,
                color=path_color,
                opacity=1.0,
                line_width=2.8,
                render_lines_as_tubes=True,
                name=f"effective_paths_{idx}",
                pickable=False,
                reset_camera=False,
                render=False,
            )
        self._restore_camera_state(cam_state)
        self._request_render()
        print(f"[VIEW] render effective paths={rendered_count}")

    def clear_effective(self):
        """Remove effective, matched, and lasso-highlight layers."""
        self._effective_path_ids.clear()
        self._remove_effective()
        self._remove_matched()
        self._remove_selection_highlight()

    def _remove_effective(self, render: bool = True, *, clear_signature: bool = True):
        if not self._plotter:
            return
        if clear_signature:
            self._effective_render_signature = None
        to_remove = [
            name for name in list(self._plotter.actors.keys())
            if isinstance(name, str) and name.startswith("effective_paths")
        ]
        for name in to_remove:
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._request_render()

    # ─── Multi-category rendering ─────────────────────────

    def render_category_paths(self, visible_categories: Dict[str, tuple]):
        """Render multiple categories as separate merged meshes with distinct colors.

        Args:
            visible_categories: {name: (path_ids_set, color_hex)}
        """
        signature = self._path_payload_signature(visible_categories)
        if signature == self._category_render_signature:
            return
        self._category_render_signature = signature
        t0 = time.perf_counter()
        cam_state = self._capture_camera_state()
        self._remove_categories(render=False, clear_signature=False)

        if not self._plotter or not visible_categories:
            self._restore_camera_state(cam_state)
            if self._plotter:
                self._request_render()
            return

        total_rendered = 0
        for cat_name, (path_ids, color) in visible_categories.items():
            all_points = []
            all_lines = []
            offset = 0
            for pid in path_ids:
                coords = self._bg_data_dict.get(pid)
                if coords is None or len(coords) < 2:
                    continue
                n = len(coords)
                all_points.append(coords)
                line = np.empty(n + 1, dtype=np.int64)
                line[0] = n
                line[1:] = np.arange(offset, offset + n)
                all_lines.append(line)
                offset += n

            if not all_points:
                continue

            mesh = pv.PolyData()
            mesh.points = np.vstack(all_points).astype(np.float64)
            mesh.lines = np.concatenate(all_lines)

            actor_name = f"{self._CAT_PREFIX}{cat_name}"
            self._plotter.add_mesh(
                mesh,
                color=color,
                opacity=0.85,
                line_width=2.5,
                render_lines_as_tubes=True,
                name=actor_name,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            total_rendered += len(all_points)

        self._restore_camera_state(cam_state)
        self._request_render()
        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"[PERF][CategoryRender] categories={len(visible_categories)}, "
            f"total_paths={total_rendered}, elapsed={elapsed:.1f}ms"
        )

    def _remove_categories(self, render: bool = True, *, clear_signature: bool = True):
        if not self._plotter:
            return
        if clear_signature:
            self._category_render_signature = None
        to_remove = [
            name for name in list(self._plotter.actors.keys())
            if isinstance(name, str) and name.startswith(self._CAT_PREFIX)
        ]
        for name in to_remove:
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._request_render()

    def clear_categories(self):
        """Remove all category layers from 3D view."""
        self._remove_categories()

    def render_dataset_paths(self, visible_datasets: Dict[str, tuple]):
        """Render dataset-colored paths with background-like styling."""
        signature = self._path_payload_signature(visible_datasets)
        if signature == self._dataset_render_signature:
            return
        self._dataset_render_signature = signature
        t0 = time.perf_counter()
        cam_state = self._capture_camera_state()
        self._remove_datasets(render=False, clear_signature=False)

        if not self._plotter or not visible_datasets:
            self._restore_camera_state(cam_state)
            if self._plotter:
                self._request_render()
            return

        total_rendered = 0
        opacity = min(0.35, max(0.08, float(self._bg_opacity)))
        for dataset_name, (path_ids, color) in visible_datasets.items():
            all_points = []
            all_lines = []
            offset = 0
            for pid in path_ids:
                coords = self._bg_data_dict.get(pid)
                if coords is None or len(coords) < 2:
                    continue
                n = len(coords)
                all_points.append(coords)
                line = np.empty(n + 1, dtype=np.int64)
                line[0] = n
                line[1:] = np.arange(offset, offset + n)
                all_lines.append(line)
                offset += n

            if not all_points:
                continue

            mesh = pv.PolyData()
            mesh.points = np.vstack(all_points).astype(np.float64)
            mesh.lines = np.concatenate(all_lines)

            actor_name = f"{self._DATASET_PREFIX}{dataset_name}"
            self._plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                line_width=1.0,
                render_lines_as_tubes=False,
                name=actor_name,
                pickable=False,
                reset_camera=False,
                render=False,
            )
            total_rendered += len(all_points)

        self._restore_camera_state(cam_state)
        self._request_render()
        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"[PERF][DatasetRender] datasets={len(visible_datasets)}, "
            f"total_paths={total_rendered}, elapsed={elapsed:.1f}ms"
        )

    def _remove_datasets(self, render: bool = True, *, clear_signature: bool = True):
        if not self._plotter:
            return
        if clear_signature:
            self._dataset_render_signature = None
        to_remove = [
            name for name in list(self._plotter.actors.keys())
            if isinstance(name, str) and name.startswith(self._DATASET_PREFIX)
        ]
        for name in to_remove:
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._request_render()

    def render_cluster_labels(self, labels: List[dict]):
        if not self._plotter:
            return
        self.clear_cluster_labels(render=False)
        if not labels:
            self._request_render()
            return
        cam_state = self._capture_camera_state()
        for idx, entry in enumerate(labels):
            point = np.asarray(entry.get("position", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(1, 3)
            text = str(entry.get("text", "") or "")
            if not text:
                continue
            pdata = pv.PolyData(point)
            self._plotter.add_point_labels(
                pdata,
                [text],
                name=f"{self._CLUSTER_LABEL_PREFIX}{idx}",
                point_size=1,
                font_size=12,
                text_color=str(entry.get("color", "#FFFFFF")),
                shape_opacity=0.18,
                shape_color="#111111",
                margin=2,
                show_points=False,
                always_visible=True,
                render=False,
            )
        self._restore_camera_state(cam_state)
        self._request_render()

    def clear_cluster_labels(self, render: bool = True):
        if not self._plotter:
            return
        to_remove = [
            name for name in list(self._plotter.actors.keys())
            if isinstance(name, str) and name.startswith(self._CLUSTER_LABEL_PREFIX)
        ]
        for name in to_remove:
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._request_render()

    # ─── Entry / Exit point rendering ─────────────────────

    def render_entry_exit_points(
        self,
        show: bool = True,
        effective_ids: Optional[Set[int]] = None,
        display_ids: Optional[Set[int]] = None,
        path_color_map: Optional[Dict[int, str]] = None,
    ):
        """Render start/end points, optionally colored by current path rendering.

        Points belonging to effective_ids are drawn larger.
        """
        signature = (
            bool(show),
            tuple(sorted(int(path_id) for path_id in (effective_ids or set()))),
            None if display_ids is None else tuple(sorted(int(path_id) for path_id in display_ids)),
            tuple(sorted((int(path_id), str(color)) for path_id, color in (path_color_map or {}).items())),
            self._bg_exclusion_signature,
        )
        if signature == self._entry_exit_render_signature:
            return
        self._entry_exit_render_signature = signature
        cam = self._capture_camera_state()
        self._remove_entry_exit(render=False, clear_signature=False)
        self._entry_exit_visible = show
        self._rendered_entry_exit_counts = {"entries": 0, "exits": 0}

        if not show or not self._plotter or not self._entry_points:
            self._restore_camera_state(cam)
            if self._plotter:
                self._request_render()
            return

        eff = effective_ids or set()
        visible_ids = None if display_ids is None else set(display_ids)
        self._entry_exit_effective_ids = {int(pid) for pid in eff}
        self._entry_exit_display_ids = None if visible_ids is None else {int(pid) for pid in visible_ids}

        def _render_grouped_points(points_by_color, *, point_size: float, opacity: float, suffix: str):
            actor_index = 0
            total_points = 0
            for color, items in points_by_color.items():
                if not items:
                    continue
                coords = np.array([item[1] for item in items], dtype=np.float64)
                ids = np.array([item[0] for item in items], dtype=np.int32)
                cloud = pv.PolyData(coords)
                cloud["path_id"] = ids
                self._plotter.add_mesh(
                    cloud,
                    color=color,
                    opacity=opacity,
                    point_size=point_size,
                    render_points_as_spheres=True,
                    name=f"{self._ENTRYEXIT_PREFIX}{suffix}_{actor_index}",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                total_points += len(items)
                actor_index += 1
            return total_points

        def _collect_points(source_points, default_color: str):
            bg: Dict[str, list] = {}
            eff_map: Dict[str, list] = {}
            for pid, pt in source_points.items():
                if pid in self._excluded_path_ids:
                    continue
                if visible_ids is not None and pid not in visible_ids:
                    continue
                color = (path_color_map or {}).get(pid, default_color)
                target = eff_map if pid in eff else bg
                target.setdefault(color, []).append((pid, pt))
            return bg, eff_map

        start_bg, start_eff = _collect_points(self._entry_points, "#2ECC71")
        end_bg, end_eff = _collect_points(self._exit_points, "#E74C3C")
        start_bg_count = _render_grouped_points(start_bg, point_size=7.0, opacity=1.0, suffix="entry_bg")
        start_eff_count = _render_grouped_points(start_eff, point_size=12.0, opacity=1.0, suffix="entry_eff")
        end_bg_count = _render_grouped_points(end_bg, point_size=7.0, opacity=1.0, suffix="exit_bg")
        end_eff_count = _render_grouped_points(end_eff, point_size=12.0, opacity=1.0, suffix="exit_eff")

        self._rendered_entry_exit_counts = {
            "entries": start_bg_count + start_eff_count,
            "exits": end_bg_count + end_eff_count,
        }
        self._restore_camera_state(cam)
        self._request_render()
        total = (
            self._rendered_entry_exit_counts["entries"] +
            self._rendered_entry_exit_counts["exits"]
        )
        print(
            f"[VIEW] render entry/exit points={total}, "
            f"effective_highlighted={len(eff)}"
        )

    def _remove_entry_exit(self, render: bool = True, *, clear_signature: bool = True):
        if not self._plotter:
            return
        if clear_signature:
            self._entry_exit_render_signature = None
        for name in list(self._plotter.actors.keys()):
            if not (isinstance(name, str) and name.startswith(self._ENTRYEXIT_PREFIX)):
                continue
            try:
                self._plotter.remove_actor(name, render=render)
            except Exception:
                pass

    def clear_entry_exit_points(self):
        self._entry_exit_visible = False
        self._rendered_entry_exit_counts = {"entries": 0, "exits": 0}
        self._entry_exit_display_ids = None
        self._entry_exit_effective_ids.clear()
        self._remove_entry_exit()

    # ─── Entry/Exit lasso selection ───────────────────────

    def _execute_entry_exit_lasso(self):
        """Lasso in entrance_exit mode: select entry/exit points → path_ids."""
        if not self._entry_points and not self._exit_points:
            self._log_lasso("EESEL", "skip: no entry/exit data")
            return

        t0 = time.perf_counter()
        polygon = np.array(self._lasso_points, dtype=np.float64)
        if polygon.shape[0] < 3:
            return

        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())

        # Collect all entry + exit points
        world_pts = []
        path_ids_list = []
        for pid, pt in self._entry_points.items():
            world_pts.append(pt)
            path_ids_list.append(pid)
        for pid, pt in self._exit_points.items():
            world_pts.append(pt)
            path_ids_list.append(pid)

        if not world_pts:
            return

        world = np.array(world_pts, dtype=np.float64)
        screen = project_points_to_screen(world, renderer, w, h)

        poly_min = polygon.min(axis=0)
        poly_max = polygon.max(axis=0)

        in_bbox = (
            (screen[:, 0] >= poly_min[0]) & (screen[:, 0] <= poly_max[0]) &
            (screen[:, 1] >= poly_min[1]) & (screen[:, 1] <= poly_max[1])
        )

        if not np.any(in_bbox):
            self._log_lasso("EESEL", "no points in bbox")
            self.lasso_selected.emit(set(), self._lasso_modifier or "replace")
            return

        inside = points_in_polygon(screen[in_bbox], polygon)
        sel_indices = np.where(in_bbox)[0][inside]
        selected_path_ids = {path_ids_list[i] for i in sel_indices}

        elapsed = (time.perf_counter() - t0) * 1000
        self._log_lasso(
            "EESEL",
            f"selected {len(selected_path_ids)} paths from {len(world_pts)} points "
            f"in {elapsed:.1f}ms"
        )
        self.lasso_selected.emit(selected_path_ids, self._lasso_modifier or "replace")

    # ─── Selection highlight (legacy) ─────────────────────

    def highlight_selected_paths(self, path_ids: Set[int]):
        """Render selected paths as a gold highlight layer."""
        cam_state = self._capture_camera_state()
        self._remove_selection_highlight(render=False)
        if not self._plotter or not path_ids:
            self._log_lasso("L5", f"highlight skip: path_ids={len(path_ids) if path_ids else 0}")
            self._restore_camera_state(cam_state)
            if self._plotter:
                self._plotter.render()
            return

        all_points = []
        all_lines = []
        offset = 0
        for pid in path_ids:
            coords = self._bg_data_dict.get(pid)
            if coords is None or len(coords) < 2:
                continue
            n = len(coords)
            all_points.append(coords)
            line = np.empty(n + 1, dtype=np.int64)
            line[0] = n
            line[1:] = np.arange(offset, offset + n)
            all_lines.append(line)
            offset += n

        if not all_points:
            self._log_lasso("L5", "highlight skip: no valid coords for selected paths")
            self._restore_camera_state(cam_state)
            self._plotter.render()
            return

        mesh = pv.PolyData()
        mesh.points = np.vstack(all_points).astype(np.float64)
        mesh.lines = np.concatenate(all_lines)

        self._plotter.add_mesh(
            mesh,
            color="#FFD700",
            opacity=0.9,
            line_width=3,
            render_lines_as_tubes=True,
            name="lasso_selection",
            pickable=False,
            reset_camera=False,
            render=False,
        )
        self._restore_camera_state(cam_state)
        self._plotter.render()
        self._log_lasso("L5", f"highlight actor updated with {len(all_points)} paths")

    def _remove_selection_highlight(self, render: bool = True):
        if self._plotter:
            try:
                self._plotter.remove_actor("lasso_selection", render=render)
            except Exception:
                pass

    # ─── Residue 3D rendering ─────────────────────────────

    def set_residue_data(self, positions: List[dict]):
        """Cache residue positions: [{'residue_id': int, 'x','y','z': float}, ...]."""
        self._residue_coords = {
            p["residue_id"]: np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
            for p in positions
        }
        self._residue_info = {
            int(p["residue_id"]): {
                "residue_id": int(p["residue_id"]),
                "label": str(p.get("label", "") or f"#{int(p['residue_id'])}"),
                "x": float(p["x"]),
                "y": float(p["y"]),
                "z": float(p["z"]),
                "path_count": int(p.get("path_count", 0) or 0),
                "dataset_color": str(p.get("dataset_color", "") or ""),
                "dataset_key": str(p.get("dataset_key", "") or ""),
            }
            for p in positions
        }

    def _hide_residue_hover_tooltip(self):
        if self._hovered_residue_id is None:
            return
        self._hovered_residue_id = None
        QToolTip.hideText()

    def _hide_combination_hover_tooltip(self):
        if self._hovered_combination_key is None:
            return
        self._hovered_combination_key = None
        QToolTip.hideText()
        self._update_combination_focus_actor()

    def _update_marker_hover_tooltip(
        self,
        obj: QWidget,
        local_pos,
        x: int,
        y: int,
    ):
        combination_info = self.pick_combination_at(x, y, max_dist_px=16.0)
        if combination_info is not None:
            combo_key = str(combination_info.get("marker_key", ""))
            if combo_key == self._hovered_combination_key:
                return
            self._hovered_residue_id = None
            text = format_combination_hover_text(combination_info)
            global_pos = obj.mapToGlobal(local_pos) + QPoint(14, 18)
            QToolTip.showText(global_pos, text, self._plotter)
            self._hovered_combination_key = combo_key
            self._update_combination_focus_actor()
            return

        self._hide_combination_hover_tooltip()
        residue_id = self.pick_residue_at(x, y, max_dist_px=16.0)
        if residue_id is None:
            self._hide_residue_hover_tooltip()
            return

        if residue_id == self._hovered_residue_id:
            return

        text = format_residue_hover_text(
            residue_id,
            self._residue_info.get(int(residue_id)),
        )
        global_pos = obj.mapToGlobal(local_pos) + QPoint(14, 18)
        QToolTip.showText(global_pos, text, self._plotter)
        self._hovered_residue_id = int(residue_id)

    def render_residues(self, selected_ids: Set[int]):
        """Render residues using dataset colors, with stronger emphasis for selected ones."""
        if not self._plotter or not self._residue_coords:
            return

        self._residue_visible = True
        cam = self._capture_camera_state()

        # Remove old actors
        for name in ("res_unselected", "res_selected", "res_labels"):
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass

        unselected_style = residue_render_style(False)
        selected_style = residue_render_style(True)
        buckets: Dict[tuple[str, bool], dict] = {}
        for rid, coord in self._residue_coords.items():
            info = self._residue_info.get(int(rid), {})
            dataset_color = str(info.get("dataset_color", "") or "#FF6B6B")
            is_selected = rid in selected_ids
            bucket = buckets.setdefault(
                (dataset_color, is_selected),
                {"points": [], "ids": []},
            )
            bucket["points"].append(coord)
            bucket["ids"].append(rid)

        actor_index = 0
        for (dataset_color, is_selected), bucket in buckets.items():
            points = bucket["points"]
            residue_ids = bucket["ids"]
            if not points:
                continue
            style = selected_style if is_selected else unselected_style
            cloud = pv.PolyData(np.array(points, dtype=np.float64))
            cloud["residue_id"] = np.array(residue_ids, dtype=np.int32)
            point_size = style.point_size * (style.selected_scale if is_selected else 1.0)
            self._plotter.add_mesh(
                cloud,
                color=dataset_color,
                opacity=style.opacity,
                point_size=point_size,
                render_points_as_spheres=True,
                name=f"residue_actor_{actor_index}",
                pickable=True,
                reset_camera=False,
                render=False,
            )
            actor_index += 1

        self._restore_camera_state(cam)
        self._plotter.render()

    def render_residue_labels(self, selected_ids: Set[int]):
        if not self._plotter:
            return
        self.clear_residue_labels(render=False)
        target_ids = set(int(rid) for rid in (selected_ids or set()) if int(rid) in self._residue_coords)
        if not target_ids:
            self._plotter.render()
            return

        cam = self._capture_camera_state()
        # Rendering one label actor per residue is too expensive. Batch them into
        # a single label actor and cap the label count to keep interaction usable.
        sorted_ids = sorted(target_ids)[:80]
        points = []
        labels = []
        for residue_id in sorted_ids:
            coord = self._residue_coords.get(int(residue_id))
            info = self._residue_info.get(int(residue_id), {})
            if coord is None:
                continue
            label_text = str(info.get("label", "") or "")
            residue_number = label_text.split(":")[-1].strip() if ":" in label_text else label_text.strip()
            text = residue_number or str(int(residue_id))
            lifted = np.asarray(coord, dtype=np.float64).copy()
            lifted[2] += 0.65
            points.append(lifted)
            labels.append(text)

        if points and labels:
            pdata = pv.PolyData(np.asarray(points, dtype=np.float64))
            self._plotter.add_point_labels(
                pdata,
                labels,
                name="res_labels_batch",
                point_size=1,
                font_size=12,
                text_color="#FFFFFF",
                shape_opacity=0.22,
                shape_color="#111111",
                margin=2,
                show_points=False,
                always_visible=True,
                render=False,
            )

        self._restore_camera_state(cam)
        self._plotter.render()

    def clear_residue_labels(self, *, render: bool = True):
        if not self._plotter:
            return
        actor_names = list(getattr(self._plotter, "actors", {}).keys())
        for name in actor_names:
            if not str(name).startswith("res_labels"):
                continue
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._plotter.render()

    def clear_residues(self):
        """Remove all residue actors."""
        self._residue_visible = False
        self._hide_residue_hover_tooltip()
        if not self._plotter:
            return
        self.clear_residue_labels(render=False)
        actor_names = list(getattr(self._plotter, "actors", {}).keys())
        for name in actor_names:
            if not str(name).startswith("residue_actor_") and name != "res_labels":
                continue
            try:
                self._plotter.remove_actor(name)
            except Exception:
                pass

    def render_combination_markers(self, markers: List[dict]):
        if not self._plotter:
            return

        self.clear_combination_markers(render=False)
        if not markers:
            self._plotter.render()
            return

        self._combination_visible = True
        if self._selected_combination_key and self._selected_combination_key not in {
            str(marker.get("marker_key", "")) for marker in markers
        }:
            self._selected_combination_key = None
        cam = self._capture_camera_state()

        buckets: Dict[str, dict] = {
            "increase": {"points": [], "keys": []},
            "decrease": {"points": [], "keys": []},
            "neutral": {"points": [], "keys": []},
        }
        self._combination_marker_coords = {}
        self._combination_marker_info = {}

        for index, marker in enumerate(markers):
            centroid = marker.get("centroid")
            if centroid is None or len(centroid) != 3:
                continue
            marker_key = str(marker.get("marker_key") or f"combo_{index}")
            point = np.array(centroid, dtype=np.float64)
            trend = str(marker.get("property_trend") or "neutral")
            bucket = buckets.get(trend, buckets["neutral"])
            bucket["points"].append(point)
            bucket["keys"].append(marker_key)
            info = dict(marker)
            info["marker_key"] = marker_key
            self._combination_marker_coords[marker_key] = point
            self._combination_marker_info[marker_key] = info

        for trend, actor_name in (
            ("neutral", "combo_neutral"),
            ("increase", "combo_increase"),
            ("decrease", "combo_decrease"),
        ):
            points = buckets[trend]["points"]
            if not points:
                continue
            style = combination_render_style(trend)
            cloud = pv.PolyData(np.array(points, dtype=np.float64))
            cloud["marker_index"] = np.arange(len(points), dtype=np.int32)
            halo_actor = self._plotter.add_mesh(
                cloud,
                color=style.color,
                opacity=0.34,
                point_size=style.point_size + 10.0,
                render_points_as_spheres=True,
                name=f"{actor_name}_halo",
                pickable=True,
                reset_camera=False,
                render=False,
            )
            self._emphasize_marker_actor(halo_actor)
            core_actor = self._plotter.add_mesh(
                cloud,
                color=style.color,
                opacity=style.opacity,
                point_size=style.point_size,
                render_points_as_spheres=True,
                name=actor_name,
                pickable=True,
                reset_camera=False,
                render=False,
            )
            self._emphasize_marker_actor(core_actor)

        self._update_combination_focus_actor(render=False)
        self._restore_camera_state(cam)
        self._plotter.render()

    def clear_combination_markers(self, render: bool = True):
        self._combination_visible = False
        self._combination_marker_coords.clear()
        self._combination_marker_info.clear()
        self._selected_combination_key = None
        self._hide_combination_hover_tooltip()
        if not self._plotter:
            return
        for name in (
            "combo_neutral",
            "combo_increase",
            "combo_decrease",
            "combo_neutral_halo",
            "combo_increase_halo",
            "combo_decrease_halo",
            "combo_focus_halo",
            "combo_focus_core",
        ):
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            self._plotter.render()

    def set_selected_combination(self, marker_key: Optional[str]):
        marker_key = str(marker_key or "").strip() or None
        self._selected_combination_key = marker_key
        self._update_combination_focus_actor()

    def _update_combination_focus_actor(self, render: bool = True):
        if not self._plotter:
            return
        for name in ("combo_focus_halo", "combo_focus_core"):
            try:
                self._plotter.remove_actor(name, render=False)
            except Exception:
                pass

        marker_key = self._hovered_combination_key or self._selected_combination_key
        if not marker_key or marker_key not in self._combination_marker_coords:
            if render:
                self._plotter.render()
            return

        info = self._combination_marker_info.get(marker_key, {})
        trend = str(info.get("property_trend") or "neutral")
        style = combination_render_style(trend)
        point = self._combination_marker_coords[marker_key]
        cloud = pv.PolyData(np.array([point], dtype=np.float64))

        halo_actor = self._plotter.add_mesh(
            cloud,
            color=style.color,
            opacity=0.30,
            point_size=style.point_size + 18.0,
            render_points_as_spheres=True,
            name="combo_focus_halo",
            pickable=False,
            reset_camera=False,
            render=False,
        )
        self._emphasize_marker_actor(halo_actor)

        core_actor = self._plotter.add_mesh(
            cloud,
            color=style.color,
            opacity=1.0,
            point_size=style.point_size + 7.0,
            render_points_as_spheres=True,
            name="combo_focus_core",
            pickable=False,
            reset_camera=False,
            render=False,
        )
        self._emphasize_marker_actor(core_actor)
        if render:
            self._plotter.render()

    def _emphasize_marker_actor(self, actor):
        if actor is None:
            return
        try:
            actor.ForceOpaqueOn()
        except Exception:
            pass
        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetAmbient(0.85)
                prop.SetDiffuse(0.20)
                prop.SetSpecular(0.18)
                prop.SetSpecularPower(16.0)
        except Exception:
            pass
        try:
            mapper = actor.GetMapper()
            if mapper is not None:
                mapper.SetResolveCoincidentTopologyToPolygonOffset()
                mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-18.0)
        except Exception:
            pass

    def pick_combination_at(self, x: int, y: int, max_dist_px: float = 20.0) -> Optional[dict]:
        if not self._plotter or not self._combination_marker_coords or not self._combination_visible:
            return None

        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())
        keys = list(self._combination_marker_coords.keys())
        world = np.array([self._combination_marker_coords[key] for key in keys], dtype=np.float64)
        screen = project_points_to_screen(world, renderer, w, h)
        dists = (screen[:, 0] - x) ** 2 + (screen[:, 1] - y) ** 2
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < float(max_dist_px) ** 2:
            return dict(self._combination_marker_info.get(keys[min_idx], {}))
        return None

    def pick_residue_at(self, x: int, y: int, max_dist_px: float = 20.0) -> Optional[int]:
        """Pick the closest residue sphere at screen position (x, y).
        Returns residue_id or None."""
        if not self._plotter or not self._residue_coords or not self._residue_visible:
            return None

        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())

        # Project all residues to screen and find nearest to click
        rids = list(self._residue_coords.keys())
        world = np.array([self._residue_coords[r] for r in rids], dtype=np.float64)
        screen = project_points_to_screen(world, renderer, w, h)

        dists = (screen[:, 0] - x) ** 2 + (screen[:, 1] - y) ** 2
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < float(max_dist_px) ** 2:
            return rids[min_idx]
        return None

    def pick_connected_exit_cluster_at(self, x: int, y: int) -> Set[int]:
        """Pick an exit sphere and return all exit spheres connected at current zoom."""
        if not self._plotter or not self._exit_points or not self._entry_exit_visible:
            return set()

        visible_ids = self._entry_exit_display_ids
        path_ids = []
        points = []
        radii = []
        for pid, point in self._exit_points.items():
            pid = int(pid)
            if pid in self._excluded_path_ids:
                continue
            if visible_ids is not None and pid not in visible_ids:
                continue
            path_ids.append(pid)
            points.append(point)
            radii.append(6.0 if pid in self._entry_exit_effective_ids else 3.5)

        if not points:
            return set()

        world = np.array(points, dtype=np.float64)
        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())
        screen = project_points_to_screen(world, renderer, w, h)
        radii_arr = np.asarray(radii, dtype=np.float64)
        click = np.array([float(x), float(y)], dtype=np.float64)
        click_dists = np.linalg.norm(screen - click, axis=1)
        hit = click_dists <= np.maximum(radii_arr, 5.0)
        if not np.any(hit):
            return set()

        seed = int(np.where(hit)[0][np.argmin(click_dists[hit])])
        world_radius = self._exit_cluster_world_radius(world[seed], radii_arr[seed] * 2.0, renderer, w, h)
        if not np.isfinite(world_radius) or world_radius <= 0:
            return {path_ids[seed]}
        # Use one conservative threshold for the connected component. This tracks
        # the current camera zoom while keeping neighbor lookup cheap.
        threshold = max(world_radius, 1e-6)
        threshold_sq = threshold * threshold
        cell_size = threshold
        cells: Dict[Tuple[int, int, int], list[int]] = {}
        keys = np.floor(world / cell_size).astype(np.int64)
        for idx, key in enumerate(keys):
            cells.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(idx)

        visited = np.zeros(len(path_ids), dtype=bool)
        queue = [seed]
        visited[seed] = True
        while queue:
            idx = queue.pop()
            key = keys[idx]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for next_idx in cells.get((int(key[0] + dx), int(key[1] + dy), int(key[2] + dz)), ()):
                            if visited[next_idx]:
                                continue
                            delta = world[next_idx] - world[idx]
                            if float(np.dot(delta, delta)) <= threshold_sq:
                                visited[next_idx] = True
                                queue.append(next_idx)

        return {path_ids[i] for i in np.where(visited)[0]}

    def _exit_cluster_world_radius(self, world_point: np.ndarray, diameter_px: float, renderer, width: int, height: int) -> float:
        camera = renderer.GetActiveCamera() if renderer is not None else None
        if camera is None:
            return 0.0
        position = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        view_dir = focal - position
        norm = float(np.linalg.norm(view_dir))
        if norm <= 1e-12:
            return 0.0
        view_dir /= norm
        depth = abs(float(np.dot(np.asarray(world_point, dtype=np.float64) - position, view_dir)))
        if camera.GetParallelProjection():
            world_height = float(camera.GetParallelScale()) * 2.0
            return (float(diameter_px) / max(1.0, float(height))) * world_height
        view_angle = np.deg2rad(float(camera.GetViewAngle()))
        world_height = 2.0 * depth * np.tan(view_angle / 2.0)
        return (float(diameter_px) / max(1.0, float(height))) * world_height

    def _screen_polyline_min_dist_sq(self, screen: np.ndarray, x: int, y: int) -> float:
        click = np.array([float(x), float(y)], dtype=np.float64)
        if len(screen) == 0:
            return float("inf")
        if len(screen) == 1:
            delta = screen[0] - click
            return float(np.dot(delta, delta))

        a = screen[:-1]
        b = screen[1:]
        ab = b - a
        denom = np.sum(ab * ab, axis=1)
        ap = click - a
        t = np.zeros(len(a), dtype=np.float64)
        valid = denom > 1e-12
        t[valid] = np.sum(ap[valid] * ab[valid], axis=1) / denom[valid]
        t = np.clip(t, 0.0, 1.0)
        closest = a + ab * t[:, None]
        delta = closest - click
        dists = np.sum(delta * delta, axis=1)
        return float(np.min(dists))

    def pick_path_at(self, x: int, y: int) -> Optional[int]:
        """Pick a currently focused path near screen position (x, y)."""
        if not self._plotter or not self._focus_mode or not self._effective_path_ids:
            return None

        renderer = self._plotter.renderer
        w = max(1, self._plotter.width())
        h = max(1, self._plotter.height())
        max_dist_sq = 12.0 * 12.0
        best_pid: Optional[int] = None
        best_dist_sq = float("inf")

        for pid in self._effective_path_ids:
            coords = self._bg_data_dict.get(int(pid))
            if coords is None or len(coords) < 2:
                continue

            sampled = self._sample_path_for_lasso(coords).astype(np.float64, copy=False)
            screen = project_points_to_screen(sampled, renderer, w, h)
            pad = 12.0
            if (
                x < float(np.min(screen[:, 0])) - pad or
                x > float(np.max(screen[:, 0])) + pad or
                y < float(np.min(screen[:, 1])) - pad or
                y > float(np.max(screen[:, 1])) + pad
            ):
                continue

            dist_sq = self._screen_polyline_min_dist_sq(screen, x, y)
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_pid = int(pid)

        if best_pid is not None and best_dist_sq <= max_dist_sq:
            return best_pid
        return None

    # ─── Protein model rendering ─────────────────────────

    def _load_protein_model(
        self,
        source_key: str,
        raw_atoms: List[dict],
        backbone: dict,
        ribbon_raw,
        style: str,
        align_mode: str,
        residue_anchor_positions: Optional[Dict[int, np.ndarray]] = None,
        *,
        render: bool = True,
    ) -> dict:
        if not raw_atoms:
            return {"ok": False, "error": "empty_or_invalid_pdb"}

        anchor_map: Dict[int, np.ndarray] = {}
        if residue_anchor_positions:
            for rid, pos in residue_anchor_positions.items():
                try:
                    rid_int = int(rid)
                    arr = np.asarray(pos, dtype=np.float64).reshape(3)
                    anchor_map[rid_int] = arr
                except Exception:
                    continue

        align_diag = estimate_alignment(
            backbone.get("ca_by_resseq", {}),
            anchor_map,
            mode=align_mode,
        )
        R = np.asarray(align_diag["R"], dtype=np.float64)
        t = np.asarray(align_diag["t"], dtype=np.float64)
        scale = float(align_diag.get("scale", 1.0))

        atom_points = np.vstack([a["position"] for a in raw_atoms]).astype(np.float64)
        atom_points_t = apply_transform(atom_points, R, t, scale=scale)

        aligned_atoms = []
        for atom, pos in zip(raw_atoms, atom_points_t):
            aligned_atoms.append(
                {
                    **atom,
                    "position": np.asarray(pos, dtype=np.float64),
                }
            )

        ca_atoms = backbone.get("ca_atoms", [])
        if ca_atoms:
            ca_points = np.vstack([a["position"] for a in ca_atoms]).astype(np.float64)
            ca_points_t = apply_transform(ca_points, R, t, scale=scale)
        else:
            ca_points_t = np.empty((0, 3), dtype=np.float64)

        segments_t = []
        segments = backbone.get("segments", [])
        if segments:
            src = np.vstack([s["source"] for s in segments]).astype(np.float64)
            dst = np.vstack([s["target"] for s in segments]).astype(np.float64)
            src_t = apply_transform(src, R, t, scale=scale)
            dst_t = apply_transform(dst, R, t, scale=scale)
            for seg, s0, s1 in zip(segments, src_t, dst_t):
                segments_t.append(
                    {
                        **seg,
                        "source": np.asarray(s0, dtype=np.float64),
                        "target": np.asarray(s1, dtype=np.float64),
                    }
                )

        ribbon_t = None
        if ribbon_raw is not None and getattr(ribbon_raw, "n_points", 0) > 0:
            try:
                ribbon_t = ribbon_raw.copy(deep=True)
                rpts = np.asarray(ribbon_t.points, dtype=np.float64)
                ribbon_t.points = apply_transform(rpts, R, t, scale=scale)
            except Exception:
                ribbon_t = None

        self._protein_model_cache = {
            "source_path": source_key,
            "raw_atoms": raw_atoms,
            "backbone_raw": backbone,
            "ribbon_raw": ribbon_raw,
            "ribbon_aligned": ribbon_t,
            "atoms_aligned": aligned_atoms,
            "ca_points_aligned": ca_points_t,
            "segments_aligned": segments_t,
        }
        self._protein_source_path = source_key
        self._protein_style = style
        self._protein_last_alignment_diag = {
            "mode": align_diag.get("mode"),
            "matched_count": int(align_diag.get("matched_count", 0)),
            "rmsd": align_diag.get("rmsd"),
            "scale": float(align_diag.get("scale", 1.0)),
            "reason": align_diag.get("reason", ""),
        }

        cam_state = self._capture_camera_state()
        self._remove_protein(render=False, keep_highlight_cache=True)
        if self._protein_visible:
            self._render_protein_model(render=False)
        self._restore_camera_state(cam_state)
        if render:
            self._plotter.render()

        return {
            "ok": True,
            "path": source_key,
            "style": self._protein_style,
            "atom_count": len(raw_atoms),
            "ca_count": len(ca_atoms),
            "segment_count": len(segments_t),
            "alignment_mode": align_diag.get("mode"),
            "matched_count": int(align_diag.get("matched_count", 0)),
            "rmsd": align_diag.get("rmsd"),
            "scale": float(align_diag.get("scale", 1.0)),
            "reason": align_diag.get("reason", ""),
        }

    def load_protein_from_file(
        self,
        pdb_path: str,
        style: str = "cartoon",
        align_mode: str = "auto_kabsch",
        residue_anchor_positions: Optional[Dict[int, np.ndarray]] = None,
        *,
        render: bool = True,
    ) -> dict:
        """Load protein from PDB and render it in the same renderer as paths."""
        if not self._plotter:
            return {"ok": False, "error": "plotter_unavailable"}

        valid_styles = {"cartoon", "backbone", "tube", "ca_spheres"}
        style = style if style in valid_styles else "cartoon"

        reuse_raw_cache = (
            self._protein_model_cache is not None
            and self._protein_source_path == pdb_path
            and "raw_atoms" in self._protein_model_cache
            and "backbone_raw" in self._protein_model_cache
        )

        if reuse_raw_cache:
            raw_atoms = self._protein_model_cache["raw_atoms"]
            backbone = self._protein_model_cache["backbone_raw"]
            ribbon_raw = self._protein_model_cache.get("ribbon_raw")
        else:
            raw_atoms = parse_pdb_atoms(pdb_path)
            backbone = build_backbone_data(raw_atoms)
            ribbon_raw = None
            try:
                ribbon_raw = generate_cartoon_ribbon(pdb_path, atoms=raw_atoms)
            except Exception:
                ribbon_raw = None

        return self._load_protein_model(
            source_key=pdb_path,
            raw_atoms=raw_atoms,
            backbone=backbone,
            ribbon_raw=ribbon_raw,
            style=style,
            align_mode=align_mode,
            residue_anchor_positions=residue_anchor_positions,
            render=render,
        )

    def load_protein_from_atoms(
        self,
        atoms: Sequence[dict],
        source_key: str,
        style: str = "cartoon",
        align_mode: str = "auto_kabsch",
        residue_anchor_positions: Optional[Dict[int, np.ndarray]] = None,
        ss_records: Optional[Dict[Tuple[str, int], str]] = None,
        *,
        render: bool = True,
    ) -> dict:
        """Load protein from pre-parsed atoms using the same render pipeline as PDB."""
        if not self._plotter:
            return {"ok": False, "error": "plotter_unavailable"}

        valid_styles = {"cartoon", "backbone", "tube", "ca_spheres"}
        style = style if style in valid_styles else "cartoon"
        source_key = str(source_key or "<in_memory_protein>")

        reuse_raw_cache = (
            self._protein_model_cache is not None
            and self._protein_source_path == source_key
            and "raw_atoms" in self._protein_model_cache
            and "backbone_raw" in self._protein_model_cache
        )

        if reuse_raw_cache:
            raw_atoms = self._protein_model_cache["raw_atoms"]
            backbone = self._protein_model_cache["backbone_raw"]
            ribbon_raw = self._protein_model_cache.get("ribbon_raw")
        else:
            raw_atoms = []
            for atom in atoms:
                try:
                    position = np.asarray(atom["position"], dtype=np.float64).reshape(3)
                except Exception:
                    continue
                raw_atoms.append(
                    {
                        **atom,
                        "position": position,
                    }
                )

            backbone = build_backbone_data(raw_atoms)
            ribbon_raw = None
            try:
                ribbon_raw = generate_cartoon_ribbon(
                    source_key,
                    atoms=raw_atoms,
                    ss_records=ss_records,
                )
            except Exception:
                ribbon_raw = None

        return self._load_protein_model(
            source_key=source_key,
            raw_atoms=raw_atoms,
            backbone=backbone,
            ribbon_raw=ribbon_raw,
            style=style,
            align_mode=align_mode,
            residue_anchor_positions=residue_anchor_positions,
            render=render,
        )

    def load_protein_from_prepared(
        self,
        source_key: str,
        raw_atoms: Sequence[dict],
        backbone: Optional[dict] = None,
        ribbon_raw=None,
        style: str = "cartoon",
        align_mode: str = "auto_kabsch",
        residue_anchor_positions: Optional[Dict[int, np.ndarray]] = None,
        *,
        render: bool = True,
    ) -> dict:
        """Load a protein from already prepared render data."""
        if not self._plotter:
            return {"ok": False, "error": "plotter_unavailable"}

        valid_styles = {"cartoon", "backbone", "tube", "ca_spheres"}
        style = style if style in valid_styles else "cartoon"
        source_key = str(source_key or "<prepared_protein>")

        prepared_atoms = list(raw_atoms or [])
        prepared_backbone = backbone if isinstance(backbone, dict) else build_backbone_data(prepared_atoms)

        return self._load_protein_model(
            source_key=source_key,
            raw_atoms=prepared_atoms,
            backbone=prepared_backbone,
            ribbon_raw=ribbon_raw,
            style=style,
            align_mode=align_mode,
            residue_anchor_positions=residue_anchor_positions,
            render=render,
        )

    def set_protein_visible(self, visible: bool):
        self._protein_visible = bool(visible)
        if not self._plotter or self._protein_model_cache is None:
            return
        cam_state = self._capture_camera_state()
        self._remove_protein(render=False)
        if self._protein_visible:
            self._render_protein_model(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_style(self, style: str):
        valid_styles = {"cartoon", "backbone", "tube", "ca_spheres"}
        if style not in valid_styles:
            return
        self._protein_style = style
        if not self._plotter or self._protein_model_cache is None or not self._protein_visible:
            return
        cam_state = self._capture_camera_state()
        self._remove_protein(render=False, keep_highlight_cache=True)
        self._render_protein_model(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_opacity(self, opacity: float, *, render: bool = True):
        self._protein_opacity = float(np.clip(float(opacity), 0.0, 1.0))
        if not self._plotter:
            return
        hide_protein = self._protein_opacity <= 0.05
        if (
            not self._protein_actors
            and not hide_protein
            and self._protein_visible
            and self._protein_model_cache is not None
        ):
            cam_state = self._capture_camera_state()
            self._render_protein_model(render=False)
            self._restore_camera_state(cam_state)
            if render:
                self._plotter.render()
            return
        for actor in self._protein_actors:
            try:
                prop = actor.GetProperty()
                if prop is not None:
                    prop.SetOpacity(max(0.05, self._protein_opacity))
                actor.SetVisibility(0 if hide_protein else 1)
                self._tune_protein_actor_priority(actor, opacity=max(0.05, self._protein_opacity))
            except Exception:
                pass
        if render:
            self._plotter.render()

    def clear_protein(self):
        self._protein_model_cache = None
        self._protein_source_path = None
        self._protein_last_alignment_diag = None
        self._protein_highlight_residue_ids = ()
        self._protein_highlight_mesh_payload = None
        if not self._plotter:
            self._protein_actors.clear()
            self._protein_highlight_actors.clear()
            return
        cam_state = self._capture_camera_state()
        self._remove_protein(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_highlight_residues(
        self,
        residue_ids: Sequence[int],
        *,
        refresh: bool = True,
        render: bool = True,
    ):
        normalized = tuple(sorted({int(rid) for rid in (residue_ids or []) if int(rid) > 0}))
        self._protein_highlight_residue_ids = normalized
        if refresh:
            self._protein_highlight_mesh_payload = None
        if not render:
            return
        if (
            not self._plotter
            or self._protein_model_cache is None
            or not self._protein_visible
        ):
            return
        cam_state = self._capture_camera_state()
        self._remove_protein_highlight(render=False, keep_cache=True)
        self._render_protein_highlight(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_highlight_style(self, style: str):
        normalized = str(style or "ball_and_stick").strip().lower().replace("-", "_")
        valid_styles = {"ball_and_stick", "sticks", "spheres"}
        if normalized not in valid_styles:
            normalized = "ball_and_stick"
        self._protein_highlight_style = normalized
        self._protein_highlight_mesh_payload = None
        if not self._plotter or self._protein_model_cache is None or not self._protein_visible:
            return
        cam_state = self._capture_camera_state()
        self._remove_protein_highlight(render=False, keep_cache=True)
        self._render_protein_highlight(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_highlight_quality(self, quality: str):
        normalized = str(quality or "high").strip().lower()
        if normalized not in {"high", "low"}:
            normalized = "high"
        self._protein_highlight_quality = normalized
        self._protein_highlight_mesh_payload = None
        if not self._plotter or self._protein_model_cache is None or not self._protein_visible:
            return
        cam_state = self._capture_camera_state()
        self._remove_protein_highlight(render=False, keep_cache=True)
        self._render_protein_highlight(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_highlight_render_enabled(self, enabled: bool, *, refresh: bool = True):
        self._protein_highlight_render_enabled = bool(enabled)
        if not self._plotter:
            return
        if not self._protein_highlight_render_enabled:
            self._remove_protein_highlight(render=refresh, keep_cache=True)
            return
        if not refresh or self._protein_model_cache is None or not self._protein_visible:
            return
        cam_state = self._capture_camera_state()
        self._remove_protein_highlight(render=False, keep_cache=True)
        self._render_protein_highlight(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def set_protein_highlight_mesh_payload(self, payload: Optional[dict]) -> None:
        self._protein_highlight_mesh_payload = payload if isinstance(payload, dict) else None

    def focus_protein_residues(self, residue_ids: Sequence[int], *, render: bool = True) -> bool:
        normalized = tuple(sorted({int(rid) for rid in (residue_ids or []) if int(rid) > 0}))
        if not normalized or not self._plotter or self._protein_model_cache is None:
            return False

        atoms = [
            atom for atom in (self._protein_model_cache.get("atoms_aligned", []) or [])
            if int(atom.get("res_seq", 0) or 0) in normalized
        ]
        if not atoms:
            return False

        renderer = self._plotter.renderer
        camera = renderer.GetActiveCamera() if renderer is not None else None
        if camera is None:
            return False

        atom_positions = np.vstack([
            np.asarray(atom["position"], dtype=np.float64).reshape(3)
            for atom in atoms
        ]).astype(np.float64)
        center = atom_positions.mean(axis=0)
        highlight_style = str(self._protein_highlight_style or "ball_and_stick")
        atom_radius_scale = float(self._HIGHLIGHT_STYLE_RADIUS_SCALE.get(highlight_style, 0.32))
        atom_radii = np.asarray(
            [
                self._highlight_radius_for_element(
                    str(atom.get("element") or "").strip().upper() or "C"
                ) * atom_radius_scale
                for atom in atoms
            ],
            dtype=np.float64,
        )
        center_offsets = np.linalg.norm(atom_positions - center, axis=1)
        extent = float(np.max(center_offsets + atom_radii)) if len(atom_positions) > 0 else 0.0
        extent = max(extent, float(np.max(atom_radii)) if len(atom_radii) > 0 else 0.0, 1.25)

        position = np.asarray(camera.GetPosition(), dtype=np.float64)
        focal_point = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
        direction = position - focal_point
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            norm = 1.0
        direction /= norm

        if camera.GetParallelProjection():
            distance = max(extent * 2.6, 6.0)
            camera.SetParallelScale(extent * 1.18)
        else:
            view_angle = max(10.0, float(camera.GetViewAngle()))
            padding = 1.12 if highlight_style == "spheres" else 1.32
            distance = max(extent / np.tan(np.deg2rad(view_angle) * 0.5) * padding, 6.0)

        new_position = center + direction * distance
        camera.SetFocalPoint(*center)
        camera.SetPosition(*new_position)
        try:
            renderer.ResetCameraClippingRange()
        except Exception:
            pass
        if render:
            self._plotter.render()
        return True

    # ─── Binding site surface rendering ───────────────────

    def load_binding_sites(
        self,
        molecules: List[dict],
        align_R: Optional[np.ndarray] = None,
        align_t: Optional[np.ndarray] = None,
        align_scale: float = 1.0,
    ) -> dict:
        """Load binding site molecules and generate Gaussian surfaces.

        Parameters
        ----------
        molecules : list of dict
            Each dict has keys: name, coords (N,3), color (hex str).
        align_R, align_t, align_scale :
            Optional alignment transform to match protein overlay.
        """
        if not self._plotter:
            return {"ok": False, "error": "plotter_unavailable"}

        self._remove_binding_sites(render=False)
        cache = []
        protein_atoms = []
        if self._protein_model_cache:
            protein_atoms = list(self._protein_model_cache.get("atoms_aligned", []) or [])

        for mol_info in molecules:
            coords = np.asarray(mol_info["coords"], dtype=np.float64)
            if len(coords) < 3:
                continue

            # Apply alignment transform if available
            if align_R is not None and align_t is not None:
                coords = apply_transform(coords, align_R, align_t, scale=align_scale)

            surface = generate_binding_site_surface(
                coords,
                protein_atoms=protein_atoms,
            )
            if surface is None:
                continue

            cache.append({
                "name": mol_info["name"],
                "surface": surface,
                "color": mol_info.get("color", "#CCCCCC"),
                "coords_aligned": coords,
            })

        self._binding_site_cache = cache
        if self._binding_site_visible:
            self._render_binding_sites(render=True)

        return {"ok": True, "count": len(cache)}

    def _render_binding_sites(self, render: bool = True):
        if not self._plotter or not self._binding_site_cache:
            return

        for i, entry in enumerate(self._binding_site_cache):
            surface = entry["surface"]
            color = entry["color"]
            wire_actor = self._plotter.add_mesh(
                surface,
                color=color,
                style="wireframe",
                opacity=1.0,
                line_width=1.15,
                name=f"{self._BINDING_PREFIX}{i}_mesh",
                pickable=False,
                reset_camera=False,
                render=False,
            )
            if wire_actor is not None:
                try:
                    prop = wire_actor.GetProperty()
                    if prop:
                        prop.SetInterpolationToFlat()
                        prop.SetAmbient(1.0)
                        prop.SetDiffuse(0.0)
                        prop.SetSpecular(0.0)
                        prop.LightingOff()
                        try:
                            prop.SetRenderLinesAsTubes(True)
                        except Exception:
                            pass
                    mapper = wire_actor.GetMapper()
                    if mapper is not None:
                        mapper.SetResolveCoincidentTopologyToPolygonOffset()
                        mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-8.0, -8.0)
                        mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-8.0, -8.0)
                except Exception:
                    pass
                self._binding_site_actors.append(wire_actor)

        if render:
            self._plotter.render()

    def _remove_binding_sites(self, render: bool = True):
        if self._plotter:
            for actor in self._binding_site_actors:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self._binding_site_actors.clear()
        if self._plotter and render:
            self._plotter.render()

    def set_binding_sites_visible(self, visible: bool):
        self._binding_site_visible = bool(visible)
        if not self._plotter:
            return
        cam_state = self._capture_camera_state()
        self._remove_binding_sites(render=False)
        if self._binding_site_visible and self._binding_site_cache:
            self._render_binding_sites(render=False)
        self._restore_camera_state(cam_state)
        self._plotter.render()

    def clear_binding_sites(self):
        self._binding_site_cache = None
        self._remove_binding_sites(render=True)

    def _render_protein_model(self, render: bool = True):
        if not self._plotter or self._protein_model_cache is None or not self._protein_visible:
            return

        style = self._protein_style
        opacity = float(np.clip(self._protein_opacity, 0.0, 1.0))
        hide_protein = opacity <= 0.05
        actor_opacity = max(0.05, opacity)
        cache = self._protein_model_cache

        # Cartoon ribbon style (closest to classic protein ribbon view).
        if style == "cartoon":
            ribbon = cache.get("ribbon_aligned")
            if ribbon is not None and getattr(ribbon, "n_points", 0) > 0:
                actor = self._plotter.add_mesh(
                    ribbon,
                    color=self._PROTEIN_CARTOON_COLOR,
                    opacity=actor_opacity,
                    smooth_shading=True,
                    name=f"{self._PROTEIN_PREFIX}cartoon",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                self._apply_pymol_cartoon_material(actor)
                actor.SetVisibility(0 if hide_protein else 1)
                self._tune_protein_actor_priority(actor, opacity=actor_opacity)
                self._protein_actors.append(actor)
            # If ribbon generation failed, gracefully fallback to backbone render.
            else:
                style = "backbone"

        # Backbone/tube line segments
        if style in {"backbone", "tube"}:
            segments = cache.get("segments_aligned", [])
            if segments:
                src = np.vstack([seg["source"] for seg in segments]).astype(np.float64)
                dst = np.vstack([seg["target"] for seg in segments]).astype(np.float64)
                points = np.vstack((src, dst))
                n_seg = len(segments)
                lines = np.empty(n_seg * 3, dtype=np.int64)
                lines[0::3] = 2
                lines[1::3] = np.arange(0, 2 * n_seg, 2, dtype=np.int64)
                lines[2::3] = np.arange(1, 2 * n_seg, 2, dtype=np.int64)
                mesh = pv.PolyData()
                mesh.points = points
                mesh.lines = lines
                actor = self._plotter.add_mesh(
                    mesh,
                    color="#B8B8C4" if style == "tube" else "#A8A8A8",
                    opacity=actor_opacity,
                    line_width=5.0 if style == "tube" else 2.0,
                    render_lines_as_tubes=(style == "tube"),
                    name=f"{self._PROTEIN_PREFIX}backbone",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                actor.SetVisibility(0 if hide_protein else 1)
                self._tune_protein_actor_priority(actor, opacity=actor_opacity)
                self._protein_actors.append(actor)

        # CA marker spheres for backbone and ca_spheres style.
        if style in {"backbone", "ca_spheres"}:
            ca_points = np.asarray(cache.get("ca_points_aligned", np.empty((0, 3))), dtype=np.float64)
            if len(ca_points) > 0:
                cloud = pv.PolyData(ca_points)
                actor = self._plotter.add_mesh(
                    cloud,
                    color="#BFC5D8" if style == "ca_spheres" else "#B0B0B0",
                    opacity=actor_opacity,
                    point_size=12.0 if style == "ca_spheres" else 7.0,
                    render_points_as_spheres=True,
                    name=f"{self._PROTEIN_PREFIX}ca",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                actor.SetVisibility(0 if hide_protein else 1)
                self._tune_protein_actor_priority(actor, opacity=actor_opacity)
                self._protein_actors.append(actor)

        if self._protein_highlight_render_enabled:
            self._render_protein_highlight(render=False)
        else:
            self._remove_protein_highlight(render=False)

        if render:
            self._plotter.render()

    @classmethod
    def _highlight_color_for_element(cls, element: str) -> str:
        key = str(element or "").strip().upper()
        return cls._HIGHLIGHT_ELEMENT_COLORS.get(key, cls._HIGHLIGHT_DEFAULT_COLOR)

    @classmethod
    def _highlight_radius_for_element(cls, element: str) -> float:
        key = str(element or "").strip().upper()
        return float(cls._HIGHLIGHT_ELEMENT_RADII.get(key, cls._HIGHLIGHT_DEFAULT_RADIUS))

    @classmethod
    def _highlight_covalent_radius_for_element(cls, element: str) -> float:
        key = str(element or "").strip().upper()
        return float(cls._HIGHLIGHT_COVALENT_RADII.get(key, 0.77))

    def _build_highlight_atom_mesh(self, positions: np.ndarray, radius: float):
        cloud = pv.PolyData(np.asarray(positions, dtype=np.float64))
        cloud["radius"] = np.full(len(cloud.points), float(radius), dtype=np.float64)
        low_quality = self._protein_highlight_quality == "low"
        sphere = pv.Sphere(
            radius=1.0,
            theta_resolution=(
                self._HIGHLIGHT_SPHERE_THETA_RES_LOW
                if low_quality
                else self._HIGHLIGHT_SPHERE_THETA_RES
            ),
            phi_resolution=(
                self._HIGHLIGHT_SPHERE_PHI_RES_LOW
                if low_quality
                else self._HIGHLIGHT_SPHERE_PHI_RES
            ),
        )
        try:
            return cloud.glyph(scale="radius", geom=sphere, orient=False, factor=1.0)
        except Exception:
            return cloud

    def _build_highlight_bond_mesh(self, start: np.ndarray, end: np.ndarray, radius: float | None = None):
        line_mesh = pv.PolyData()
        line_mesh.points = np.vstack((start, end)).astype(np.float64)
        line_mesh.lines = np.array([2, 0, 1], dtype=np.int64)
        try:
            low_quality = self._protein_highlight_quality == "low"
            return line_mesh.tube(
                radius=float(radius if radius is not None else self._HIGHLIGHT_BOND_RADIUS),
                n_sides=self._HIGHLIGHT_BOND_SIDES_LOW if low_quality else self._HIGHLIGHT_BOND_SIDES,
                capping=True,
            )
        except Exception:
            return line_mesh

    @staticmethod
    def _compute_highlight_bond_anchors(
        start: np.ndarray,
        end: np.ndarray,
        start_radius: float,
        end_radius: float,
        bond_radius: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return None
        direction /= length

        start_trim_request = max(0.0, float(start_radius) * 0.96)
        end_trim_request = max(0.0, float(end_radius) * 0.96)
        trim_total = start_trim_request + end_trim_request
        visible_target = max(float(bond_radius) * 1.35, min(0.24, length * 0.22))
        available_trim = max(0.0, length - visible_target)
        if trim_total > 1e-8:
            trim_scale = min(1.0, available_trim / trim_total)
        else:
            trim_scale = 0.0

        start_trim = start_trim_request * trim_scale
        end_trim = end_trim_request * trim_scale
        start_anchor = np.asarray(start, dtype=np.float64) + direction * start_trim
        end_anchor = np.asarray(end, dtype=np.float64) - direction * end_trim
        if float(np.linalg.norm(end_anchor - start_anchor)) <= 1e-4:
            return None
        return start_anchor, end_anchor

    @staticmethod
    def _compute_highlight_hbond_anchors(
        start: np.ndarray,
        end: np.ndarray,
        start_radius: float,
        end_radius: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return None
        direction /= length
        start_trim = min(max(0.0, float(start_radius) * 0.42), length * 0.18)
        end_trim = min(max(0.0, float(end_radius) * 0.42), length * 0.18)
        start_anchor = np.asarray(start, dtype=np.float64) + direction * start_trim
        end_anchor = np.asarray(end, dtype=np.float64) - direction * end_trim
        if float(np.linalg.norm(end_anchor - start_anchor)) <= 1e-4:
            return None
        return start_anchor, end_anchor

    @staticmethod
    def _build_dashed_segments(
        start: np.ndarray,
        end: np.ndarray,
        *,
        dash_length: float = 0.24,
        gap_length: float = 0.16,
    ) -> list[np.ndarray]:
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            return []
        direction /= length
        segments: list[np.ndarray] = []
        offset = 0.0
        while offset < length:
            seg_end = min(length, offset + float(dash_length))
            if seg_end - offset > 1e-4:
                seg_start_point = np.asarray(start, dtype=np.float64) + direction * offset
                seg_end_point = np.asarray(start, dtype=np.float64) + direction * seg_end
                segments.append(np.vstack((seg_start_point, seg_end_point)).astype(np.float64))
            offset += float(dash_length) + float(gap_length)
        return segments

    def _build_protein_hbond_specs(
        self,
        atoms: Sequence[dict],
        atom_positions: np.ndarray,
        atom_elements: list[str],
        visible_atom_radii: np.ndarray,
        *,
        render_atoms: bool,
    ) -> list[dict]:
        hydrogen_indexes = [idx for idx, element in enumerate(atom_elements) if element == "H"]
        donor_h_map: dict[int, list[int]] = {}
        for h_idx in hydrogen_indexes:
            best_donor_idx = None
            best_distance = float("inf")
            h_pos = atom_positions[h_idx]
            h_residue = residue_key(atoms[h_idx])
            for donor_idx, donor_atom in enumerate(atoms):
                if donor_idx == h_idx or not is_hbond_donor_atom(donor_atom):
                    continue
                if residue_key(donor_atom) != h_residue:
                    continue
                distance = float(np.linalg.norm(atom_positions[donor_idx] - h_pos))
                if distance <= 1.30 and distance < best_distance:
                    best_distance = distance
                    best_donor_idx = donor_idx
            if best_donor_idx is not None:
                donor_h_map.setdefault(best_donor_idx, []).append(h_idx)

        hbond_segments: list[np.ndarray] = []
        seen_pairs: set[tuple[int, int, int]] = set()
        for donor_idx, donor_atom in enumerate(atoms):
            if not is_hbond_donor_atom(donor_atom):
                continue
            donor_pos = atom_positions[donor_idx]
            donor_residue = residue_key(donor_atom)
            donor_hydrogens = donor_h_map.get(donor_idx, [])
            for acceptor_idx, acceptor_atom in enumerate(atoms):
                if acceptor_idx == donor_idx:
                    continue
                if not is_hbond_acceptor_atom(acceptor_atom):
                    continue
                if residue_key(acceptor_atom) == donor_residue:
                    continue
                acceptor_pos = atom_positions[acceptor_idx]
                d_a_distance = float(np.linalg.norm(acceptor_pos - donor_pos))
                if d_a_distance > 3.6:
                    continue

                best_segment: tuple[np.ndarray, np.ndarray] | None = None
                best_score = float("inf")
                best_h_idx = -1
                for hydrogen_idx in donor_hydrogens:
                    hydrogen_pos = atom_positions[hydrogen_idx]
                    h_a_distance = float(np.linalg.norm(acceptor_pos - hydrogen_pos))
                    if h_a_distance > 2.7:
                        continue
                    dh_direction = hydrogen_pos - donor_pos
                    dh_norm = float(np.linalg.norm(dh_direction))
                    if dh_norm <= 1e-8:
                        continue
                    dh_direction /= dh_norm
                    ha_direction = acceptor_pos - hydrogen_pos
                    ha_norm = float(np.linalg.norm(ha_direction))
                    if ha_norm <= 1e-8:
                        continue
                    ha_direction /= ha_norm
                    angle = float(np.degrees(np.arccos(np.clip(np.dot(dh_direction, ha_direction), -1.0, 1.0))))
                    if angle < 105.0:
                        continue
                    score = h_a_distance + (180.0 - angle) * 0.01
                    if score < best_score:
                        best_score = score
                        best_h_idx = hydrogen_idx
                        if render_atoms:
                            anchors = self._compute_highlight_hbond_anchors(
                                hydrogen_pos,
                                acceptor_pos,
                                float(visible_atom_radii[hydrogen_idx]),
                                float(visible_atom_radii[acceptor_idx]),
                            )
                            if anchors is None:
                                continue
                            best_segment = anchors
                        else:
                            best_segment = (hydrogen_pos, acceptor_pos)

                if best_segment is None:
                    if donor_hydrogens and d_a_distance > 3.3:
                        continue
                    if render_atoms:
                        anchors = self._compute_highlight_hbond_anchors(
                            donor_pos,
                            acceptor_pos,
                            float(visible_atom_radii[donor_idx]),
                            float(visible_atom_radii[acceptor_idx]),
                        )
                        if anchors is None:
                            continue
                        best_segment = anchors
                    else:
                        best_segment = (donor_pos, acceptor_pos)

                pair_key = (donor_idx, best_h_idx, acceptor_idx)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                segment_start, segment_end = best_segment
                hbond_segments.extend(
                    self._build_dashed_segments(
                        segment_start,
                        segment_end,
                        dash_length=0.28,
                        gap_length=0.18,
                    )
                )

        if not hbond_segments:
            return []
        return [
            {
                "color": "#58B5FF",
                "radius": float(self._HIGHLIGHT_HBOND_RADIUS),
                "segments": np.ascontiguousarray(np.stack(hbond_segments).astype(np.float64)),
            }
        ]

    @staticmethod
    def _build_vtk_points(points: np.ndarray):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(
            numpy_to_vtk(
                np.ascontiguousarray(np.asarray(points, dtype=np.float64)),
                deep=True,
            )
        )
        return vtk_points

    @staticmethod
    def _build_vtk_lines(segments: np.ndarray):
        segment_array = np.ascontiguousarray(np.asarray(segments, dtype=np.float64))
        line_count = int(len(segment_array))
        points = segment_array.reshape(-1, 3)
        cell_data = np.empty(line_count * 3, dtype=np.int64)
        cell_data[0::3] = 2
        cell_data[1::3] = np.arange(0, line_count * 2, 2, dtype=np.int64)
        cell_data[2::3] = np.arange(1, line_count * 2, 2, dtype=np.int64)
        vtk_points = TunnelViewer3D._build_vtk_points(points)
        vtk_lines = vtk.vtkCellArray()
        vtk_lines.SetCells(
            line_count,
            numpy_to_vtkIdTypeArray(
                np.ascontiguousarray(cell_data),
                deep=True,
            ),
        )
        return vtk_points, vtk_lines

    def _apply_protein_highlight_actor_material(
        self,
        actor,
        *,
        color: str,
        specular: float,
        specular_power: float,
        ambient: float,
        diffuse: float,
        force_opaque: bool = False,
    ) -> None:
        try:
            prop = actor.GetProperty()
            if prop is None:
                return
            prop.SetColor(*pv.Color(color).float_rgb)
            prop.SetSpecular(float(specular))
            prop.SetSpecularPower(float(specular_power))
            prop.SetAmbient(float(ambient))
            prop.SetDiffuse(float(diffuse))
            prop.SetOpacity(1.0)
            if force_opaque:
                prop.SetInterpolationToPhong()
        except Exception:
            pass
        self._tune_protein_actor_priority(actor, opacity=1.0, force_opaque=force_opaque)

    def _create_protein_highlight_atom_pipeline(
        self,
        key: str,
        *,
        color: str,
        specular: float,
        specular_power: float,
        ambient: float,
        diffuse: float,
        force_opaque: bool = False,
    ) -> dict | None:
        if not self._plotter or vtk is None:
            return None
        polydata = vtk.vtkPolyData()
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(polydata)
        glyph.SetSourceConnection(sphere.GetOutputPort())
        glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleFactor(1.0)
        glyph.OrientOff()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        self._plotter.renderer.AddActor(actor)
        self._protein_highlight_actor_map[key] = actor
        self._protein_highlight_actors.append(actor)
        pipeline = {
            "kind": "atoms",
            "actor": actor,
            "polydata": polydata,
            "sphere": sphere,
            "glyph": glyph,
            "mapper": mapper,
        }
        self._protein_highlight_pipeline_map[key] = pipeline
        self._apply_protein_highlight_actor_material(
            actor,
            color=color,
            specular=specular,
            specular_power=specular_power,
            ambient=ambient,
            diffuse=diffuse,
            force_opaque=force_opaque,
        )
        return pipeline

    def _create_protein_highlight_bond_pipeline(
        self,
        key: str,
        *,
        color: str,
        specular: float,
        specular_power: float,
        ambient: float,
        diffuse: float,
        force_opaque: bool = False,
    ) -> dict | None:
        if not self._plotter or vtk is None:
            return None
        polydata = vtk.vtkPolyData()
        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.CappingOn()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.PickableOff()
        self._plotter.renderer.AddActor(actor)
        self._protein_highlight_actor_map[key] = actor
        self._protein_highlight_actors.append(actor)
        pipeline = {
            "kind": "bonds",
            "actor": actor,
            "polydata": polydata,
            "tube": tube,
            "mapper": mapper,
        }
        self._protein_highlight_pipeline_map[key] = pipeline
        self._apply_protein_highlight_actor_material(
            actor,
            color=color,
            specular=specular,
            specular_power=specular_power,
            ambient=ambient,
            diffuse=diffuse,
            force_opaque=force_opaque,
        )
        return pipeline

    def _upsert_protein_highlight_atom_actor(
        self,
        key: str,
        positions: np.ndarray,
        radius: float,
        *,
        color: str,
        specular: float,
        specular_power: float,
        ambient: float,
        diffuse: float,
        force_opaque: bool = False,
    ):
        pipeline = self._protein_highlight_pipeline_map.get(key)
        if pipeline is None or pipeline.get("kind") != "atoms":
            pipeline = self._create_protein_highlight_atom_pipeline(
                key,
                color=color,
                specular=specular,
                specular_power=specular_power,
                ambient=ambient,
                diffuse=diffuse,
                force_opaque=force_opaque,
            )
            if pipeline is None:
                return None
        actor = pipeline.get("actor")
        polydata = pipeline.get("polydata")
        sphere = pipeline.get("sphere")
        glyph = pipeline.get("glyph")
        if actor is None or polydata is None or sphere is None or glyph is None:
            return None
        theta_res = (
            self._HIGHLIGHT_SPHERE_THETA_RES_LOW
            if self._protein_highlight_quality == "low"
            else self._HIGHLIGHT_SPHERE_THETA_RES
        )
        phi_res = (
            self._HIGHLIGHT_SPHERE_PHI_RES_LOW
            if self._protein_highlight_quality == "low"
            else self._HIGHLIGHT_SPHERE_PHI_RES
        )
        sphere.SetThetaResolution(int(theta_res))
        sphere.SetPhiResolution(int(phi_res))
        positions_array = np.ascontiguousarray(np.asarray(positions, dtype=np.float64))
        polydata.SetPoints(self._build_vtk_points(positions_array))
        radius_array = numpy_to_vtk(
            np.full(len(positions_array), float(radius), dtype=np.float64),
            deep=True,
        )
        radius_array.SetName("radius")
        polydata.GetPointData().SetScalars(radius_array)
        polydata.Modified()
        sphere.Modified()
        glyph.Update()
        actor.VisibilityOn()
        self._apply_protein_highlight_actor_material(
            actor,
            color=color,
            specular=specular,
            specular_power=specular_power,
            ambient=ambient,
            diffuse=diffuse,
            force_opaque=force_opaque,
        )
        return actor

    def _upsert_protein_highlight_bond_actor(
        self,
        key: str,
        segments: np.ndarray,
        radius: float,
        *,
        color: str,
        specular: float,
        specular_power: float,
        ambient: float,
        diffuse: float,
        force_opaque: bool = False,
    ):
        pipeline = self._protein_highlight_pipeline_map.get(key)
        if pipeline is None or pipeline.get("kind") != "bonds":
            pipeline = self._create_protein_highlight_bond_pipeline(
                key,
                color=color,
                specular=specular,
                specular_power=specular_power,
                ambient=ambient,
                diffuse=diffuse,
                force_opaque=force_opaque,
            )
            if pipeline is None:
                return None
        actor = pipeline.get("actor")
        polydata = pipeline.get("polydata")
        tube = pipeline.get("tube")
        if actor is None or polydata is None or tube is None:
            return None
        segment_array = np.ascontiguousarray(np.asarray(segments, dtype=np.float64))
        vtk_points, vtk_lines = self._build_vtk_lines(segment_array)
        polydata.SetPoints(vtk_points)
        polydata.SetLines(vtk_lines)
        polydata.Modified()
        tube.SetRadius(float(radius))
        tube.SetNumberOfSides(
            self._HIGHLIGHT_BOND_SIDES_LOW
            if self._protein_highlight_quality == "low"
            else self._HIGHLIGHT_BOND_SIDES
        )
        tube.Modified()
        tube.Update()
        actor.VisibilityOn()
        self._apply_protein_highlight_actor_material(
            actor,
            color=color,
            specular=specular,
            specular_power=specular_power,
            ambient=ambient,
            diffuse=diffuse,
            force_opaque=force_opaque,
        )
        return actor

    def _render_protein_highlight_payload(self, payload: dict, used_keys: set[str]) -> None:
        atom_specs = list(payload.get("atom_specs") or [])
        bond_specs = list(payload.get("bond_specs") or [])
        hbond_specs = list(payload.get("hbond_specs") or [])
        for item in atom_specs:
            positions = item.get("positions")
            if positions is None:
                continue
            color = str(item.get("color") or self._HIGHLIGHT_DEFAULT_COLOR)
            radius = float(item.get("radius") or self._HIGHLIGHT_DEFAULT_RADIUS)
            color_name = str(color).replace("#", "")
            actor_key = f"{self._PROTEIN_PREFIX}highlight_atoms_{color_name}"
            actor = self._upsert_protein_highlight_atom_actor(
                actor_key,
                np.asarray(positions, dtype=np.float64),
                radius,
                color=color,
                specular=0.18,
                specular_power=24.0,
                ambient=0.15,
                diffuse=0.85,
                force_opaque=True,
            )
            if actor is not None:
                used_keys.add(actor_key)
        for item in bond_specs:
            segments = item.get("segments")
            if segments is None:
                continue
            color = str(item.get("color") or self._HIGHLIGHT_DEFAULT_COLOR)
            radius = float(item.get("radius") or self._HIGHLIGHT_BOND_RADIUS)
            color_name = str(color).replace("#", "")
            actor_key = f"{self._PROTEIN_PREFIX}highlight_bonds_{color_name}"
            actor = self._upsert_protein_highlight_bond_actor(
                actor_key,
                np.asarray(segments, dtype=np.float64),
                radius,
                color=color,
                specular=0.12,
                specular_power=18.0,
                ambient=0.12,
                diffuse=0.88,
                force_opaque=True,
            )
            if actor is not None:
                used_keys.add(actor_key)
        for item in hbond_specs:
            segments = item.get("segments")
            if segments is None:
                continue
            color = str(item.get("color") or "#58B5FF")
            radius = float(item.get("radius") or self._HIGHLIGHT_HBOND_RADIUS)
            color_name = str(color).replace("#", "")
            actor_key = f"{self._PROTEIN_PREFIX}highlight_hbonds_{color_name}"
            actor = self._upsert_protein_highlight_bond_actor(
                actor_key,
                np.asarray(segments, dtype=np.float64),
                radius,
                color=color,
                specular=0.06,
                specular_power=10.0,
                ambient=0.22,
                diffuse=0.78,
                force_opaque=True,
            )
            if actor is not None:
                used_keys.add(actor_key)

    def _build_protein_highlight_payload_from_atoms(self, atoms: Sequence[dict]) -> dict | None:
        if not atoms:
            return None
        bond_pairs: list[tuple[int, int]] = []
        atom_positions = np.vstack([np.asarray(atom["position"], dtype=np.float64) for atom in atoms]).astype(np.float64)
        atom_elements = [
            str(atom.get("element") or "").strip().upper() or "C"
            for atom in atoms
        ]
        atom_radii = np.asarray(
            [self._highlight_radius_for_element(element) for element in atom_elements],
            dtype=np.float64,
        )
        bond_radii = np.asarray(
            [self._highlight_covalent_radius_for_element(element) for element in atom_elements],
            dtype=np.float64,
        )
        highlight_style = str(self._protein_highlight_style or "ball_and_stick")
        render_atoms = highlight_style in {"ball_and_stick", "spheres"}
        render_bonds = highlight_style in {"ball_and_stick", "sticks"}
        atom_radius_scale = float(self._HIGHLIGHT_STYLE_RADIUS_SCALE.get(highlight_style, 0.72))
        visible_atom_radii = atom_radii * atom_radius_scale if render_atoms else atom_radii

        for idx, atom_a in enumerate(atoms):
            pos_a = atom_positions[idx]
            for jdx in range(idx + 1, len(atoms)):
                atom_b = atoms[jdx]
                if int(atom_a.get("res_seq", 0) or 0) != int(atom_b.get("res_seq", 0) or 0):
                    continue
                pos_b = atom_positions[jdx]
                distance = float(np.linalg.norm(pos_a - pos_b))
                max_distance = min(
                    2.2,
                    (bond_radii[idx] + bond_radii[jdx]) * self._HIGHLIGHT_BOND_MAX_SCALE,
                )
                if distance <= max_distance:
                    bond_pairs.append((idx, jdx))

        atom_specs: list[dict] = []
        bond_specs: list[dict] = []
        hbond_specs: list[dict] = []

        if render_atoms:
            atoms_by_element: dict[str, list[np.ndarray]] = {}
            for element, position in zip(atom_elements, atom_positions):
                atoms_by_element.setdefault(element, []).append(position)
            for element, positions in atoms_by_element.items():
                atom_specs.append(
                    {
                        "color": self._highlight_color_for_element(element),
                        "radius": float(self._highlight_radius_for_element(element) * atom_radius_scale),
                        "positions": np.ascontiguousarray(np.vstack(positions).astype(np.float64)),
                    }
                )

        if render_bonds and bond_pairs:
            bond_specs_by_color: dict[str, list[np.ndarray]] = {}
            bond_radius = (
                self._HIGHLIGHT_BOND_RADIUS
                if highlight_style == "ball_and_stick"
                else self._HIGHLIGHT_STICK_BOND_RADIUS
            )
            for start_idx, end_idx in bond_pairs:
                start = atom_positions[start_idx]
                end = atom_positions[end_idx]
                if render_atoms:
                    anchors = self._compute_highlight_bond_anchors(
                        start,
                        end,
                        float(visible_atom_radii[start_idx]),
                        float(visible_atom_radii[end_idx]),
                        float(bond_radius),
                    )
                    if anchors is None:
                        continue
                    start_anchor, end_anchor = anchors
                else:
                    start_anchor = start
                    end_anchor = end
                anchor_distance = float(np.linalg.norm(end_anchor - start_anchor))
                if anchor_distance <= 1e-4:
                    continue
                midpoint = (start_anchor + end_anchor) * 0.5
                start_color = self._highlight_color_for_element(atom_elements[start_idx])
                end_color = self._highlight_color_for_element(atom_elements[end_idx])
                bond_specs_by_color.setdefault(start_color, []).append(
                    np.vstack((start_anchor, midpoint)).astype(np.float64)
                )
                bond_specs_by_color.setdefault(end_color, []).append(
                    np.vstack((midpoint, end_anchor)).astype(np.float64)
                )
            for color, segments in bond_specs_by_color.items():
                if not segments:
                    continue
                bond_specs.append(
                    {
                        "color": color,
                        "radius": float(bond_radius),
                        "segments": np.ascontiguousarray(np.stack(segments).astype(np.float64)),
                    }
                )

        if len({int(atom.get("res_seq", 0) or 0) for atom in atoms}) >= 2:
            hbond_specs = self._build_protein_hbond_specs(
                atoms,
                atom_positions,
                atom_elements,
                visible_atom_radii,
                render_atoms=render_atoms,
            )

        return {
            "atom_specs": atom_specs,
            "bond_specs": bond_specs,
            "hbond_specs": hbond_specs,
        }

    def _hide_unused_protein_highlight_actors(self, used_keys: set[str]) -> None:
        for key, actor in self._protein_highlight_actor_map.items():
            try:
                actor.SetVisibility(1 if key in used_keys else 0)
            except Exception:
                pass

    def _render_protein_highlight(self, render: bool = True):
        if (
            not self._protein_highlight_render_enabled
            or not self._plotter
            or self._protein_model_cache is None
        ):
            self._remove_protein_highlight(render=False, keep_cache=True)
            return
        residue_ids = self._protein_highlight_residue_ids
        if not residue_ids:
            self._remove_protein_highlight(render=False, keep_cache=True)
            return
        used_keys: set[str] = set()
        cached_payload = self._protein_highlight_mesh_payload
        if cached_payload and (cached_payload.get("atom_specs") or cached_payload.get("bond_specs")):
            self._render_protein_highlight_payload(cached_payload, used_keys)
            self._hide_unused_protein_highlight_actors(used_keys)
            if render:
                self._plotter.render()
            return

        atoms = [
            atom for atom in (self._protein_model_cache.get("atoms_aligned", []) or [])
            if int(atom.get("res_seq", 0) or 0) in residue_ids
        ]
        if not atoms:
            self._remove_protein_highlight(render=False, keep_cache=True)
            return
        dynamic_payload = self._build_protein_highlight_payload_from_atoms(atoms)
        if dynamic_payload is not None:
            self._render_protein_highlight_payload(dynamic_payload, used_keys)

        self._hide_unused_protein_highlight_actors(used_keys)

        if render:
            self._plotter.render()

    def _remove_protein(self, render: bool = True, keep_highlight_cache: bool = False):
        if self._plotter:
            for actor in self._protein_actors:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
            # Name-based cleanup for stale actors.
            for name in ("protein_backbone", "protein_ca"):
                try:
                    self._plotter.remove_actor(name, render=False)
                except Exception:
                    pass
            try:
                self._plotter.remove_actor("protein_cartoon", render=False)
            except Exception:
                pass
        self._protein_actors.clear()
        self._remove_protein_highlight(render=False, keep_cache=keep_highlight_cache)
        if self._plotter and render:
            self._plotter.render()

    def _remove_protein_highlight(self, render: bool = True, keep_cache: bool = False):
        if keep_cache:
            for actor in self._protein_highlight_actor_map.values():
                try:
                    actor.VisibilityOff()
                except Exception:
                    pass
            if self._plotter and render:
                self._plotter.render()
            return
        if self._plotter:
            for actor in self._protein_highlight_actors:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
            for name in (
                f"{self._PROTEIN_PREFIX}highlight_atoms",
                f"{self._PROTEIN_PREFIX}highlight_bonds",
            ):
                try:
                    self._plotter.remove_actor(name, render=False)
                except Exception:
                    pass
        self._protein_highlight_actors.clear()
        self._protein_highlight_actor_map.clear()
        self._protein_highlight_pipeline_map.clear()
        if self._plotter and render:
            self._plotter.render()

    def _tune_protein_actor_priority(self, actor, opacity: float | None = None, force_opaque: bool = False):
        """Keep protein visually readable while respecting current opacity."""
        if actor is None:
            return
        target_opacity = float(np.clip(
            self._protein_opacity if opacity is None else opacity,
            0.05,
            1.0,
        ))
        try:
            if force_opaque or target_opacity >= 0.999:
                actor.ForceOpaqueOn()
            else:
                actor.ForceOpaqueOff()
        except Exception:
            pass
        try:
            prop = actor.GetProperty()
            if prop is not None:
                prop.SetOpacity(target_opacity)
                prop.SetLighting(True)
                if force_opaque:
                    prop.SetOpacity(1.0)
        except Exception:
            pass
        try:
            mapper = actor.GetMapper()
            if mapper is not None:
                if force_opaque:
                    mapper.SetResolveCoincidentTopologyToOff()
                else:
                    # Bring protein slightly to foreground in Z-buffer resolve path.
                    mapper.SetResolveCoincidentTopologyToPolygonOffset()
                    mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-12.0, -12.0)
                    mapper.SetRelativeCoincidentTopologyLineOffsetParameters(-12.0, -12.0)
                    mapper.SetRelativeCoincidentTopologyPointOffsetParameter(-12.0)
        except Exception:
            pass

    def _apply_pymol_cartoon_material(self, actor):
        """Use a simple Phong material close to PyMOL default cartoon look."""
        if actor is None:
            return
        try:
            prop = actor.GetProperty()
            if prop is None:
                return
            prop.SetInterpolationToPhong()
            prop.SetColor(0.22, 0.90, 0.35)
            prop.SetAmbient(0.25)
            prop.SetDiffuse(0.75)
            prop.SetSpecular(0.42)
            prop.SetSpecularPower(32.0)
            prop.SetSpecularColor(1.0, 1.0, 1.0)
        except Exception:
            pass

    # ─── Live property updates ────────────────────────────

    def set_bg_opacity(self, opacity: float):
        """Update background paths opacity without re-rendering mesh."""
        if not self._plotter or not self._bg_actors:
            return
        for actor in self._bg_actors:
            if actor:
                actor.GetProperty().SetOpacity(opacity)
        self._plotter.render()

    def set_bg_line_width(self, width: float):
        """Update background paths line width without re-rendering mesh."""
        if not self._plotter or not self._bg_actors:
            return
        for actor in self._bg_actors:
            if actor:
                actor.GetProperty().SetLineWidth(width)
        self._request_render()

    def set_bg_color(self, hex_color: str):
        """Update background paths color without re-building mesh."""
        if not self._plotter or not self._bg_actors:
            return
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        for actor in self._bg_actors:
            if actor:
                actor.GetProperty().SetColor(r, g, b)
        self._request_render()

    def set_bg_visible(self, visible: bool):
        """Show/hide all background path actors (for Focus mode)."""
        if not self._plotter or not self._bg_actors:
            return
        for actor in self._bg_actors:
            if actor:
                actor.SetVisibility(visible)
        self._request_render()

    def set_dataset_visible(self, visible: bool):
        """Show/hide dataset-colored overlay actors."""
        if not self._plotter:
            return
        changed = False
        for name, actor in list(self._plotter.actors.items()):
            if isinstance(name, str) and name.startswith(self._DATASET_PREFIX) and actor:
                actor.SetVisibility(visible)
                changed = True
        if changed:
            self._request_render()

    # ─── Rendering ───────────────────────────────────────

    def _add_background_mesh_actor(
        self,
        mesh,
        *,
        name: str = "background_paths",
        render: bool = False,
    ):
        if self._bg_change_metric_map:
            values = np.asarray(
                [
                    float(self._bg_change_metric_map.get(int(path_id), 0.0))
                    for path_id in self._bg_cell_path_ids
                ],
                dtype=np.float32,
            )
            mesh.cell_data["path_change_distance"] = values
            finite_values = values[np.isfinite(values)]
            if finite_values.size and self._bg_change_metric_clim_percentile:
                high_value = float(np.percentile(
                    finite_values,
                    float(self._bg_change_metric_clim_percentile),
                ))
                max_value = float(np.nanmax(finite_values))
                high_value = min(max(high_value, 1e-6), max(max_value, 1e-6))
            else:
                high_value = float(np.nanmax(values)) if values.size else 0.0
            clim = (0.0, max(high_value, 1e-6))
            return self._plotter.add_mesh(
                mesh,
                scalars="path_change_distance",
                cmap=self._bg_change_metric_cmap,
                clim=clim,
                opacity=max(float(self._bg_opacity), 0.28),
                line_width=1.2,
                render_lines_as_tubes=False,
                name=name,
                pickable=True,
                reset_camera=False,
                render=render,
                show_scalar_bar=False,
            )
        return self._plotter.add_mesh(
            mesh,
            color=self._bg_color,
            opacity=self._bg_opacity,
            line_width=1,
            render_lines_as_tubes=False,
            name=name,
            pickable=True,
            reset_camera=False,
            render=render,
        )

    def set_path_change_metric_map(
        self,
        metric_map: Optional[Dict[int, float]],
        excluded_ids: Optional[Set[int]] = None,
        *,
        cmap: str = "Reds",
        clim_percentile: Optional[float] = None,
    ):
        self._bg_change_metric_cmap = str(cmap or "Reds")
        if clim_percentile is None:
            self._bg_change_metric_clim_percentile = None
        else:
            self._bg_change_metric_clim_percentile = float(np.clip(float(clim_percentile), 1.0, 100.0))
        normalized = None
        if metric_map:
            normalized = {
                int(path_id): float(value)
                for path_id, value in metric_map.items()
                if int(path_id) > 0 and np.isfinite(float(value))
            }
        self._bg_change_metric_map = normalized if normalized else None
        self._bg_exclusion_signature = None
        self.set_bg_excluded_paths(self._excluded_path_ids if excluded_ids is None else excluded_ids)

    @property
    def plotter(self):
        return self._plotter

    def render_background_paths(
        self,
        paths: List[Tuple[int, int, np.ndarray]],
        opacity: float = 0.15,
        color: str = "#505050",
    ):
        """Render all background paths as a single merged PolyData for performance."""
        if not self._plotter:
            return

        self._bg_opacity = opacity
        self._bg_color = color

        # Remove old background
        self._remove_bg(render=False)

        if not paths:
            self._request_render()
            return

        # Build merged polydata + cache data for selection/highlight
        all_points = []
        all_lines = []
        point_offset = 0
        cell_path_ids = []
        bg_dict = {}
        midpoint_dict = {}

        for path_id, frame_id, coords in paths:
            n = len(coords)
            if n < 2:
                continue
            all_points.append(coords)

            # Line connectivity: [n_pts, idx0, idx1, ..., idx_{n-1}]
            line = np.empty(n + 1, dtype=np.int64)
            line[0] = n
            line[1:] = np.arange(point_offset, point_offset + n)
            all_lines.append(line)
            cell_path_ids.append(path_id)
            point_offset += n

            bg_dict[path_id] = coords
            midpoint_dict[path_id] = coords[n // 2]

        if not all_points:
            return

        self._bg_data_dict = bg_dict
        self._bg_midpoint_dict = midpoint_dict

        # Build entry/exit point caches from path endpoints
        entry_pts = {}
        exit_pts = {}
        for pid, coords in bg_dict.items():
            if len(coords) >= 2:
                entry_pts[pid] = coords[0].astype(np.float64)
                exit_pts[pid] = coords[-1].astype(np.float64)
        self._entry_points = entry_pts
        self._exit_points = exit_pts

        points = np.vstack(all_points)
        lines = np.concatenate(all_lines)

        mesh = pv.PolyData()
        mesh.points = points
        mesh.lines = lines
        # Keep exact path traceability: selected cell -> CellData["path_id"] -> path_id.
        mesh.cell_data["path_id"] = np.asarray(cell_path_ids, dtype=np.int32)
        self._bg_cell_path_ids = np.asarray(cell_path_ids, dtype=np.int64)

        actor = self._add_background_mesh_actor(mesh, render=False)
        self._bg_actors.append(actor)
        self._bg_mesh = mesh
        self._excluded_path_ids = set()
        self._bg_exclusion_signature = ()
        self._log_lasso(
            "L3",
            f"build merged polydata: cells={mesh.n_cells}, "
            f"path_id_len={len(mesh.cell_data['path_id'])}, pickable={actor.GetPickable() if actor else 'unknown'}",
        )

        if not self._view_initialized:
            self._plotter.reset_camera()
            self._view_initialized = True

    def set_bg_excluded_paths(self, excluded_ids: Set[int]):
        """Rebuild background mesh excluding specified paths (they become invisible + unpickable)."""
        excluded_set = {int(path_id) for path_id in (excluded_ids or set())}
        signature = tuple(sorted(excluded_set))
        if signature == self._bg_exclusion_signature and self._bg_actors:
            self._excluded_path_ids = excluded_set
            return
        self._excluded_path_ids = excluded_set
        if not self._plotter or not self._bg_data_dict:
            return

        t0 = time.perf_counter()
        cam_state = self._capture_camera_state()

        # Remove old background actor only (keep caches)
        try:
            self._plotter.remove_actor("background_paths", render=False)
        except Exception:
            pass
        self._bg_actors.clear()

        all_points = []
        all_lines = []
        point_offset = 0
        cell_path_ids = []

        for path_id, coords in self._bg_data_dict.items():
            if path_id in excluded_set:
                continue
            n = len(coords)
            if n < 2:
                continue
            all_points.append(coords)
            line = np.empty(n + 1, dtype=np.int64)
            line[0] = n
            line[1:] = np.arange(point_offset, point_offset + n)
            all_lines.append(line)
            cell_path_ids.append(path_id)
            point_offset += n

        if not all_points:
            self._bg_mesh = None
            self._bg_cell_path_ids = None
            self._restore_camera_state(cam_state)
            self._request_render()
            return

        mesh = pv.PolyData()
        mesh.points = np.vstack(all_points)
        mesh.lines = np.concatenate(all_lines)
        mesh.cell_data["path_id"] = np.asarray(cell_path_ids, dtype=np.int32)
        self._bg_cell_path_ids = np.asarray(cell_path_ids, dtype=np.int64)
        self._bg_mesh = mesh

        actor = self._add_background_mesh_actor(mesh, render=False)
        self._bg_actors = [actor]
        self._bg_exclusion_signature = signature

        self._restore_camera_state(cam_state)
        self._request_render()

        elapsed = (time.perf_counter() - t0) * 1000
        total = len(self._bg_data_dict)
        shown = len(cell_path_ids)
        print(f"[PERF][BgExclude] excluded={total - shown}, shown={shown}, elapsed={elapsed:.1f}ms")

    def render_matched_paths(
        self,
        path_coords: Dict[int, np.ndarray],
        color_map: Optional[Dict[int, str]] = None,
        opacity: float = 0.8,
        line_width: float = 2.0,
        highlighted: Optional[Set[int]] = None,
    ):
        """Render matched/selected paths with per-path colors."""
        if not self._plotter:
            return

        cam_state = self._capture_camera_state()
        self._remove_matched(render=False)

        default_color = "#00CC96"
        highlighted = highlighted or set()
        grouped_points: Dict[tuple[str, bool], list] = {}
        grouped_lines: Dict[tuple[str, bool], list] = {}
        grouped_offsets: Dict[tuple[str, bool], int] = {}

        for pid, coords in path_coords.items():
            if len(coords) < 2:
                continue
            color = str((color_map or {}).get(pid, default_color))
            is_highlighted = pid in highlighted
            group_key = (color, is_highlighted)
            offset = grouped_offsets.get(group_key, 0)
            n = len(coords)
            grouped_points.setdefault(group_key, []).append(coords)
            line = np.empty(n + 1, dtype=np.int64)
            line[0] = n
            line[1:] = np.arange(offset, offset + n)
            grouped_lines.setdefault(group_key, []).append(line)
            grouped_offsets[group_key] = offset + n

        for actor_index, ((color, is_highlighted), point_groups) in enumerate(grouped_points.items()):
            if not point_groups:
                continue
            mesh = pv.PolyData()
            mesh.points = np.vstack(point_groups).astype(np.float64)
            mesh.lines = np.concatenate(grouped_lines.get((color, is_highlighted), []))
            width = line_width * 2 if is_highlighted else line_width
            actor = self._plotter.add_mesh(
                mesh,
                color=color,
                opacity=opacity,
                line_width=width,
                render_lines_as_tubes=True,
                name=f"{self._MATCH_PREFIX}{actor_index}",
                pickable=True,
                reset_camera=False,
                render=False,
            )
            self._match_actors.append(actor)
        self._restore_camera_state(cam_state)
        self._request_render()

    def render_markers(
        self,
        positions: np.ndarray,
        colors=None,
        point_size: float = 8.0,
        name: str = "markers",
    ):
        """Render point markers (start/end points, residue positions)."""
        if not self._plotter or len(positions) == 0:
            return

        cloud = pv.PolyData(positions.astype(np.float64))
        actor = self._plotter.add_mesh(
            cloud,
            color=colors if isinstance(colors, str) else "red",
            point_size=point_size,
            render_points_as_spheres=True,
            name=name,
            pickable=True,
            reset_camera=False,
        )
        self._marker_actors.append(actor)

    def clear_matched(self):
        self._remove_matched()

    def clear_all(self, *, render: bool = True):
        self._remove_bg(render=False)
        self._remove_datasets(render=False)
        self._remove_matched(render=False)
        self._remove_markers(render=False)
        self.clear_combination_markers(render=False)
        self._remove_selection_highlight(render=False)
        self._remove_protein(render=False)
        self._remove_categories(render=False)
        if self._plotter and render:
            self._plotter.render()

    def reset_camera(self):
        if self._plotter:
            self._plotter.reset_camera()

    def screenshot(self, path: str):
        if self._plotter:
            self._plotter.screenshot(path)

    def shutdown(self):
        if self._closing:
            return
        self._closing = True
        plotter = getattr(self, "_plotter", None)
        if plotter is None:
            return
        try:
            self.clear_all(render=False)
        except Exception:
            pass
        try:
            self.clear_lasso_overlay("shutdown")
        except Exception:
            pass
        try:
            self._lasso_overlay.hide()
        except Exception:
            pass
        for child in list(plotter.findChildren(QWidget)):
            try:
                child.removeEventFilter(self)
            except Exception:
                pass
        try:
            plotter.removeEventFilter(self)
        except Exception:
            pass
        try:
            plotter.setUpdatesEnabled(False)
        except Exception:
            pass
        interactor = getattr(plotter, "interactor", None)
        if interactor is None:
            interactor = getattr(plotter, "iren", None)
        if interactor is not None:
            try:
                interactor.Disable()
            except Exception:
                pass
            try:
                interactor.TerminateApp()
            except Exception:
                pass
        render_window = None
        try:
            render_window = plotter.GetRenderWindow()
        except Exception:
            render_window = getattr(plotter, "ren_win", None)
        if render_window is not None:
            try:
                render_window.Finalize()
            except Exception:
                pass
        try:
            plotter.close()
        except Exception:
            pass
        self._plotter = None

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    # ─── Internal cleanup ───────────────────────────────
    def _remove_bg(self, *, render: bool = True):
        if self._plotter:
            try:
                self._plotter.remove_actor("background_paths", render=False)
            except Exception:
                pass
        self._bg_actors.clear()
        self._bg_mesh = None
        self._bg_cell_path_ids = None
        self._bg_data_dict.clear()
        self._bg_midpoint_dict.clear()
        self._entry_points.clear()
        self._exit_points.clear()
        self._bg_exclusion_signature = None
        self._effective_render_signature = None
        self._category_render_signature = None
        self._dataset_render_signature = None
        self._entry_exit_render_signature = None
        if render:
            self._request_render()

    def _remove_matched(self, render: bool = True):
        if self._plotter:
            for actor in self._match_actors:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self._match_actors.clear()
        if render:
            self._request_render()

    def _remove_markers(self, render: bool = True):
        if self._plotter:
            for actor in self._marker_actors:
                try:
                    self._plotter.remove_actor(actor, render=False)
                except Exception:
                    pass
        self._marker_actors.clear()
        if render:
            self._request_render()

    def enable_rubber_band_selection(self, callback):
        """Enable rectangular area selection (box pick)."""
        if self._plotter:
            # VTK rubber band picking for area selection
            self._plotter.enable_rubber_band_picking(
                callback=callback, show=True
            )
