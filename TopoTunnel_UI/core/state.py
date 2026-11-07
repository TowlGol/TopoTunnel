"""
Application-wide state manager.
Replaces the Pinia tunnelStore.js with Python + Qt Signals.
"""
import time
import json
from typing import Optional, Set, List, Dict, Any
from PySide6.QtCore import QObject, Signal


class AppState(QObject):
    """Central application state, emits signals on changes."""

    # Signals for UI updates
    residue_selection_changed = Signal()
    matched_paths_changed = Signal()
    path_filter_changed = Signal()
    category_changed = Signal()
    category_visibility_changed = Signal()  # visible/excluded state changed
    session_changed = Signal()
    view_settings_changed = Signal()
    frame_range_changed = Signal()
    selected_paths_changed = Signal()
    effective_paths_changed = Signal()
    selection_mode_changed = Signal(str)    # 'path' | 'entrance_exit'
    entry_exit_display_changed = Signal(bool)
    entry_exit_scope_changed = Signal(bool)
    dataset_display_changed = Signal()

    def __init__(self):
        super().__init__()

        # Session
        self.current_session_id: Optional[str] = None
        self.current_session_name: Optional[str] = None

        # Residue selection
        self.selected_residues: Set[int] = set()
        self.residue_options: List[dict] = []

        # Matched paths (result of residue filtering) — legacy alias
        self.matched_path_ids: List[int] = []

        # ── Unified path state model ──
        self.residue_filtered_path_ids: Set[int] = set()
        self.lasso_added_path_ids: Set[int] = set()
        self.lasso_removed_path_ids: Set[int] = set()
        self.effective_path_ids: Set[int] = set()

        # Categories: name -> list of path indices
        self.categories: Dict[str, List[int]] = {}
        self.category_colors: Dict[str, str] = {}
        self.active_category: Optional[str] = None
        self.visible_categories: Set[str] = set()    # categories rendered in 3D
        self.excluded_categories: Set[str] = set()   # categories excluded from candidates

        # Excluded paths
        self.global_excluded_paths: Set[int] = set()
        self.dataset_colors: Dict[str, str] = {}
        self.dataset_start_colors: Dict[str, str] = {}
        self.dataset_end_colors: Dict[str, str] = {}
        self.visible_datasets: Set[str] = set()
        self.dataset_cluster_display_key: Optional[str] = None
        self.dataset_cluster_label_key: Optional[str] = None

        # Legacy box/lasso selection (kept for backward compat)
        self.box_selected_paths: Set[int] = set()
        self.selected_path_ids: Set[int] = set()

        # View settings
        self.show_start_markers = False
        self.show_end_markers = False
        self.show_all_residues = False
        self.show_entry_exit_points = False
        self.entry_exit_current_only = False
        self.focus_mode = False
        self.selection_mode: str = "path"   # 'path' | 'entrance_exit'
        self.path_width = 2.0
        self.match_opacity = 0.8
        self.bg_opacity = 0.08

        # Frame range
        self.frame_min: Optional[int] = None
        self.frame_max: Optional[int] = None

        # Color palette
        self.color_palette = [
            "#4169E1", "#FF7A59", "#8B5CF6", "#F59E0B",
            "#E11D48", "#06B6D4", "#A855F7", "#F97316",
        ]

    # ── Unified state methods ────────────────────────────

    def recompute_effective(self) -> float:
        """Recompute effective_path_ids. Returns elapsed ms."""
        t0 = time.perf_counter()
        excluded = self.get_excluded_path_ids()
        self.effective_path_ids = (
            (self.residue_filtered_path_ids | self.lasso_added_path_ids)
            - self.lasso_removed_path_ids
            - excluded
        )
        self.selected_path_ids = set(self.effective_path_ids)
        elapsed = (time.perf_counter() - t0) * 1000
        print(
            f"[STATE] residue_filtered={len(self.residue_filtered_path_ids)}, "
            f"lasso_added={len(self.lasso_added_path_ids)}, "
            f"lasso_removed={len(self.lasso_removed_path_ids)}, "
            f"excluded={len(excluded)}, "
            f"effective={len(self.effective_path_ids)}"
        )
        return elapsed

    def set_residue_filtered_paths(self, ids: List[int], emit: bool = True):
        """Update residue-filtered base set and recompute effective.

        Clears lasso edits so they don't carry over across residue changes.
        """
        self.residue_filtered_path_ids = set(ids)
        self.matched_path_ids = ids
        self.lasso_added_path_ids.clear()
        self.lasso_removed_path_ids.clear()
        self.recompute_effective()
        if emit:
            self.effective_paths_changed.emit()

    def apply_lasso(self, path_ids, mode: str = "replace"):
        """Apply lasso selection to the unified state.

        mode: 'replace' — set effective to exactly these paths
              'add'     — add these paths to effective
              'remove'  — remove these paths from effective
        """
        ids = set(path_ids) if not isinstance(path_ids, set) else path_ids
        print(
            f"[LASSO][L5][State] apply_lasso mode={mode}, incoming={len(ids)}, "
            f"effective_before={len(self.effective_path_ids)}"
        )
        if mode == "replace":
            self.lasso_added_path_ids = ids - self.residue_filtered_path_ids
            self.lasso_removed_path_ids = self.residue_filtered_path_ids - ids
        elif mode == "add":
            self.lasso_added_path_ids |= ids
            self.lasso_removed_path_ids -= ids
        elif mode == "remove":
            currently_effective = (
                (self.residue_filtered_path_ids | self.lasso_added_path_ids)
                - self.lasso_removed_path_ids
            )
            to_remove = ids & currently_effective
            self.lasso_removed_path_ids |= (to_remove & self.residue_filtered_path_ids)
            self.lasso_added_path_ids -= to_remove

        self.recompute_effective()
        self.effective_paths_changed.emit()

    def clear_lasso_edits(self):
        """Clear lasso modifications, restoring to pure residue filter result."""
        self.lasso_added_path_ids.clear()
        self.lasso_removed_path_ids.clear()
        self.recompute_effective()
        self.effective_paths_changed.emit()

    # ── Residue helpers ──────────────────────────────────

    def toggle_residue(self, residue_id: int):
        if residue_id in self.selected_residues:
            self.selected_residues.discard(residue_id)
        else:
            self.selected_residues.add(residue_id)
        self.residue_selection_changed.emit()

    def clear_residues(self):
        self.selected_residues.clear()
        self.matched_path_ids.clear()
        self.residue_filtered_path_ids.clear()
        self.lasso_added_path_ids.clear()
        self.lasso_removed_path_ids.clear()
        self.effective_path_ids.clear()
        self.selected_path_ids.clear()
        self.residue_selection_changed.emit()
        self.matched_paths_changed.emit()
        self.effective_paths_changed.emit()

    def set_matched_paths(self, ids: List[int]):
        self.matched_path_ids = ids
        self.matched_paths_changed.emit()

    def add_category(self, name: str, path_ids: List[int]):
        self.categories[name] = list(path_ids)
        idx = len(self.categories) - 1
        self.category_colors[name] = self.color_palette[idx % len(self.color_palette)]
        self.category_changed.emit()

    def delete_category(self, name: str):
        was_excluded = name in self.excluded_categories
        self.categories.pop(name, None)
        self.category_colors.pop(name, None)
        self.visible_categories.discard(name)
        self.excluded_categories.discard(name)
        if self.active_category == name:
            self.active_category = None
        if was_excluded:
            self.recompute_effective()
            self.effective_paths_changed.emit()
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def set_active_category(self, name: Optional[str]):
        self.active_category = name
        self.category_changed.emit()

    def toggle_category_visible(self, name: str):
        """Toggle 3D visibility of a category."""
        if name not in self.categories:
            return
        if name in self.visible_categories:
            self.visible_categories.discard(name)
        else:
            self.visible_categories.add(name)
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def toggle_category_excluded(self, name: str):
        """Toggle exclusion of a category from candidate paths."""
        if name not in self.categories:
            return
        if name in self.excluded_categories:
            self.excluded_categories.discard(name)
        else:
            self.excluded_categories.add(name)
        self.recompute_effective()
        self.effective_paths_changed.emit()
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def get_excluded_path_ids(self) -> Set[int]:
        """Return union of all excluded category path IDs."""
        result: Set[int] = set()
        for name in self.excluded_categories:
            ids = self.categories.get(name, [])
            result.update(ids)
        return result

    def get_visible_category_path_ids(self) -> Set[int]:
        """Return union of all currently visible category path IDs."""
        result: Set[int] = set()
        for name in self.visible_categories:
            ids = self.categories.get(name, [])
            result.update(ids)
        return result

    # ─── Batch category operations ────────────────────────

    def show_all_categories(self):
        self.visible_categories = set(self.categories.keys())
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def hide_all_categories(self):
        self.visible_categories.clear()
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def exclude_all_categories(self):
        self.excluded_categories = set(self.categories.keys())
        self.recompute_effective()
        self.effective_paths_changed.emit()
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def clear_all_exclusions(self):
        if not self.excluded_categories:
            return
        self.excluded_categories.clear()
        self.recompute_effective()
        self.effective_paths_changed.emit()
        self.category_changed.emit()
        self.category_visibility_changed.emit()

    def exclude_paths(self, path_ids: List[int]):
        self.global_excluded_paths.update(path_ids)
        self.path_filter_changed.emit()

    def include_paths(self, path_ids: List[int]):
        self.global_excluded_paths.difference_update(path_ids)
        self.path_filter_changed.emit()

    def set_box_selection(self, path_ids: Set[int]):
        self.box_selected_paths = path_ids
        self.matched_paths_changed.emit()

    def clear_box_selection(self):
        self.box_selected_paths.clear()
        self.matched_paths_changed.emit()

    def set_selected_paths(self, path_ids, mode: str = "replace"):
        """Legacy: Update lasso/selection path IDs."""
        ids = set(path_ids) if not isinstance(path_ids, set) else path_ids
        self.apply_lasso(ids, mode)

    def clear_selected_paths(self):
        self.clear_lasso_edits()

    def set_selection_mode(self, mode: str):
        if mode not in ("path", "entrance_exit"):
            mode = "path"
        if mode != self.selection_mode:
            self.selection_mode = mode
            self.selection_mode_changed.emit(mode)

    def set_show_entry_exit_points(self, show: bool):
        self.show_entry_exit_points = show
        self.entry_exit_display_changed.emit(show)

    def set_entry_exit_current_only(self, current_only: bool):
        current_only = bool(current_only)
        if current_only != self.entry_exit_current_only:
            self.entry_exit_current_only = current_only
            self.entry_exit_scope_changed.emit(current_only)

    def set_frame_range(self, fmin: Optional[int], fmax: Optional[int]):
        self.frame_min = fmin
        self.frame_max = fmax
        self.frame_range_changed.emit()

    def get_auto_color(self, idx: int) -> str:
        return self.color_palette[idx % len(self.color_palette)]

    def ensure_dataset_defaults(self, dataset_keys: List[str]):
        """Ensure new datasets get default color/visibility without resetting existing ones."""
        keys = [str(key) for key in dataset_keys]
        known_keys = set(self.dataset_colors.keys()) | self.visible_datasets
        changed = False

        for idx, key in enumerate(keys):
            if key in known_keys:
                # Already has state — do NOT overwrite color or visibility
                continue
            # Brand-new dataset: assign default color and make visible
            self.dataset_colors[key] = self.get_auto_color(idx)
            self.dataset_start_colors.setdefault(key, self.dataset_colors[key])
            self.dataset_end_colors.setdefault(key, self.dataset_colors[key])
            self.visible_datasets.add(key)
            changed = True

        for idx, key in enumerate(keys):
            default_color = self.dataset_colors.get(key, self.get_auto_color(idx))
            self.dataset_start_colors.setdefault(key, default_color)
            self.dataset_end_colors.setdefault(key, default_color)

        # Remove stale keys that no longer exist in the database
        stale_keys = set(self.dataset_colors.keys()) - set(keys)
        if stale_keys:
            for key in stale_keys:
                self.dataset_colors.pop(key, None)
                self.dataset_start_colors.pop(key, None)
                self.dataset_end_colors.pop(key, None)
                self.visible_datasets.discard(key)
            changed = True

        for color_map in (self.dataset_start_colors, self.dataset_end_colors):
            for key in set(color_map) - set(keys):
                color_map.pop(key, None)
                changed = True

        stale_visible = {key for key in self.visible_datasets if key not in keys}
        if stale_visible:
            self.visible_datasets.difference_update(stale_visible)
            changed = True

        if self.dataset_cluster_display_key and self.dataset_cluster_display_key not in keys:
            self.dataset_cluster_display_key = None
            changed = True
        if self.dataset_cluster_label_key and self.dataset_cluster_label_key not in keys:
            self.dataset_cluster_label_key = None
            changed = True

        if changed:
            self.dataset_display_changed.emit()

    def set_dataset_color(self, key: str, color: str, targets=None):
        color = str(color or "").strip()
        if not key or not color:
            return
        target_set = {str(target).strip().lower() for target in (targets or ("path",))}
        changed = False
        if "path" in target_set and self.dataset_colors.get(key) != color:
            self.dataset_colors[key] = color
            changed = True
        if "start_point" in target_set and self.dataset_start_colors.get(key) != color:
            self.dataset_start_colors[key] = color
            changed = True
        if "endpoint" in target_set and self.dataset_end_colors.get(key) != color:
            self.dataset_end_colors[key] = color
            changed = True
        if changed:
            self.dataset_display_changed.emit()

    def set_dataset_visible(self, key: str, visible: bool):
        if not key:
            return
        changed = False
        if visible:
            if key not in self.visible_datasets:
                self.visible_datasets.add(key)
                changed = True
        else:
            if key in self.visible_datasets:
                self.visible_datasets.discard(key)
                changed = True
        if changed:
            self.dataset_display_changed.emit()

    def set_dataset_visibility_batch(self, keys, visible: bool):
        normalized_keys = [
            str(key).strip()
            for key in (keys or [])
            if str(key).strip()
        ]
        if not normalized_keys:
            return
        changed = False
        if visible:
            for key in normalized_keys:
                if key not in self.visible_datasets:
                    self.visible_datasets.add(key)
                    changed = True
        else:
            for key in normalized_keys:
                if key in self.visible_datasets:
                    self.visible_datasets.discard(key)
                    changed = True
        if changed:
            self.dataset_display_changed.emit()

    def toggle_dataset_visible(self, key: str):
        self.set_dataset_visible(key, key not in self.visible_datasets)

    def set_dataset_cluster_display(self, key: Optional[str]):
        normalized = str(key).strip() if key else None
        if self.dataset_cluster_display_key == normalized:
            return
        self.dataset_cluster_display_key = normalized
        self.dataset_display_changed.emit()

    def set_dataset_cluster_labels(self, key: Optional[str]):
        normalized = str(key).strip() if key else None
        if self.dataset_cluster_label_key == normalized:
            return
        self.dataset_cluster_label_key = normalized
        self.dataset_display_changed.emit()

    def serialize(self) -> str:
        """Serialize state to JSON for session save."""
        return json.dumps({
            "selectedResidues": list(self.selected_residues),
            "categories": self.categories,
            "categoryColors": self.category_colors,
            "visibleCategories": list(self.visible_categories),
            "excludedCategories": list(self.excluded_categories),
            "globalExcludedPaths": list(self.global_excluded_paths),
            "datasetColors": self.dataset_colors,
            "datasetStartColors": self.dataset_start_colors,
            "datasetEndColors": self.dataset_end_colors,
            "visibleDatasets": list(self.visible_datasets),
            "datasetClusterDisplayKey": self.dataset_cluster_display_key,
            "datasetClusterLabelKey": self.dataset_cluster_label_key,
            "showStart": self.show_start_markers,
            "showEnd": self.show_end_markers,
            "showAllResidues": self.show_all_residues,
        })

    def deserialize(self, data_json: str):
        """Restore state from JSON."""
        try:
            d = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            return
        self.selected_residues = set(d.get("selectedResidues", []))
        self.categories = d.get("categories", {})
        self.category_colors = d.get("categoryColors", {})
        self.visible_categories = set(d.get("visibleCategories", []))
        self.excluded_categories = set(d.get("excludedCategories", []))
        self.global_excluded_paths = set(d.get("globalExcludedPaths", []))
        self.dataset_colors = d.get("datasetColors", {})
        self.dataset_start_colors = d.get("datasetStartColors", {})
        self.dataset_end_colors = d.get("datasetEndColors", {})
        self.visible_datasets = set(d.get("visibleDatasets", []))
        self.dataset_cluster_display_key = d.get("datasetClusterDisplayKey")
        self.dataset_cluster_label_key = d.get("datasetClusterLabelKey")
        self.show_start_markers = d.get("showStart", False)
        self.show_end_markers = d.get("showEnd", False)
        self.show_all_residues = d.get("showAllResidues", False)
        self.residue_selection_changed.emit()
        self.category_changed.emit()
        self.category_visibility_changed.emit()
        self.dataset_display_changed.emit()
        self.view_settings_changed.emit()
