"""
Local protein observer panel.
Uses the project's native VTK/PyVista protein renderer instead of 3Dmol.js.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import os
import re
import tempfile
import threading
from pathlib import Path
import numpy as np

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

try:
    import pyvista as pv
except ImportError:
    pv = None  # type: ignore[assignment]

from TopoTunnel_UI.vis.cartoon_ribbon import generate_cartoon_ribbon
from TopoTunnel_UI.vis.protein_model import (
    build_backbone_data,
    is_hbond_acceptor_atom,
    is_hbond_donor_atom,
    parse_pdb_text,
    residue_key,
)
from TopoTunnel_UI.vis.tunnel_viewer import TunnelViewer3D


def _extract_first_model_text(pdb_text: str) -> str:
    text = str(pdb_text or "")
    lines = text.splitlines()
    if not any(line[:6].strip().upper() == "MODEL" for line in lines):
        return text

    preamble: list[str] = []
    model_lines: list[str] = []
    seen_model = False
    for line in lines:
        record = line[:6].strip().upper()
        if record == "MODEL":
            if seen_model:
                break
            seen_model = True
            continue
        if record == "ENDMDL" and seen_model:
            break
        if seen_model:
            model_lines.append(line)
        else:
            preamble.append(line)

    result = "\n".join([*preamble, *model_lines])
    if result and (text.endswith("\n") or text.endswith("\r\n")):
        result += "\n"
    return result or text


def _prepare_sequence_payload(path: str, label: str, frame: int, *, build_ribbon: bool = True) -> dict:
    text = _extract_first_model_text(
        Path(path).read_text(encoding="utf-8", errors="replace")
    )
    raw_atoms = parse_pdb_text(text)
    backbone = build_backbone_data(raw_atoms)
    ribbon_raw = None
    if build_ribbon:
        try:
            ribbon_raw = generate_cartoon_ribbon(str(path), atoms=raw_atoms)
        except Exception:
            ribbon_raw = None
    return {
        "source_key": str(path),
        "raw_atoms": raw_atoms,
        "backbone": backbone,
        "ribbon_raw": ribbon_raw,
        "label": str(label),
        "frame": int(frame),
        "highlight_cache_key": None,
        "highlight_mesh_payload": None,
    }


def _build_path_overlay_mesh(points: np.ndarray, radii: np.ndarray | None = None):
    if pv is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return None
    cloud = pv.PolyData(pts)
    if radii is not None:
        try:
            arr = np.asarray(radii, dtype=np.float64).reshape(-1)
            if len(arr) == len(pts):
                cloud["radius"] = np.maximum(arr, 0.1)
        except Exception:
            pass
    sphere = pv.Sphere(radius=1.0, theta_resolution=18, phi_resolution=18)
    try:
        if radii is not None and "radius" in cloud.array_names:
            return cloud.glyph(scale="radius", geom=sphere, orient=False, factor=1.0)
        return cloud.glyph(geom=sphere, orient=False, factor=1.0)
    except Exception:
        return cloud


def _build_path_overlay_centerline(points: np.ndarray, path_ids: np.ndarray | None = None):
    if pv is None:
        return None
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        return None
    if path_ids is None:
        ids = np.zeros(len(pts), dtype=np.int64)
    else:
        try:
            ids = np.asarray(path_ids, dtype=np.int64).reshape(-1)
        except Exception:
            ids = np.zeros(len(pts), dtype=np.int64)
        if len(ids) != len(pts):
            ids = np.zeros(len(pts), dtype=np.int64)

    line_cells: list[int] = []
    for idx in range(len(pts) - 1):
        if int(ids[idx]) != int(ids[idx + 1]):
            continue
        line_cells.extend((2, idx, idx + 1))
    if not line_cells:
        return None

    mesh = pv.PolyData()
    mesh.points = pts
    mesh.lines = np.asarray(line_cells, dtype=np.int64)
    return mesh


def _normalize_overlay_color(color: str | None, fallback: str = "#FFB000") -> str:
    text = str(color or "").strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", text):
        if not text.startswith("#"):
            text = f"#{text}"
        return text.upper()
    return str(fallback or "#FFB000").strip().upper()


def _shade_overlay_color(color: str | None, factor: float, fallback: str = "#9A6700") -> str:
    normalized = _normalize_overlay_color(color, fallback=fallback)
    try:
        red = int(normalized[1:3], 16)
        green = int(normalized[3:5], 16)
        blue = int(normalized[5:7], 16)
    except Exception:
        return _normalize_overlay_color(fallback, fallback="#9A6700")
    scale = max(0.0, min(1.0, float(factor)))
    return "#{:02X}{:02X}{:02X}".format(
        int(round(red * scale)),
        int(round(green * scale)),
        int(round(blue * scale)),
    )


def _normalize_highlight_cache_key(residue_ids, style: str, quality: str) -> tuple | None:
    normalized_ids = tuple(sorted({int(rid) for rid in (residue_ids or ()) if int(rid) > 0}))
    if not normalized_ids:
        return None
    normalized_style = str(style or "ball_and_stick").strip().lower().replace("-", "_")
    if normalized_style not in {"ball_and_stick", "sticks", "spheres"}:
        normalized_style = "ball_and_stick"
    normalized_quality = str(quality or "high").strip().lower()
    if normalized_quality not in {"high", "low"}:
        normalized_quality = "high"
    return (normalized_ids, normalized_style, normalized_quality)


def _highlight_color_for_element(element: str) -> str:
    key = str(element or "").strip().upper()
    return TunnelViewer3D._HIGHLIGHT_ELEMENT_COLORS.get(key, TunnelViewer3D._HIGHLIGHT_DEFAULT_COLOR)


def _highlight_radius_for_element(element: str) -> float:
    key = str(element or "").strip().upper()
    return float(TunnelViewer3D._HIGHLIGHT_ELEMENT_RADII.get(key, TunnelViewer3D._HIGHLIGHT_DEFAULT_RADIUS))


def _highlight_covalent_radius_for_element(element: str) -> float:
    key = str(element or "").strip().upper()
    return float(TunnelViewer3D._HIGHLIGHT_COVALENT_RADII.get(key, 0.77))


def _build_highlight_atom_mesh(positions: np.ndarray, radius: float, quality: str):
    if pv is None:
        return None
    cloud = pv.PolyData(np.asarray(positions, dtype=np.float64))
    cloud["radius"] = np.full(len(cloud.points), float(radius), dtype=np.float64)
    low_quality = str(quality) == "low"
    sphere = pv.Sphere(
        radius=1.0,
        theta_resolution=(
            TunnelViewer3D._HIGHLIGHT_SPHERE_THETA_RES_LOW
            if low_quality
            else TunnelViewer3D._HIGHLIGHT_SPHERE_THETA_RES
        ),
        phi_resolution=(
            TunnelViewer3D._HIGHLIGHT_SPHERE_PHI_RES_LOW
            if low_quality
            else TunnelViewer3D._HIGHLIGHT_SPHERE_PHI_RES
        ),
    )
    try:
        return cloud.glyph(scale="radius", geom=sphere, orient=False, factor=1.0)
    except Exception:
        return cloud


def _build_highlight_bond_mesh(start: np.ndarray, end: np.ndarray, radius: float, quality: str):
    if pv is None:
        return None
    line_mesh = pv.PolyData()
    line_mesh.points = np.vstack((start, end)).astype(np.float64)
    line_mesh.lines = np.array([2, 0, 1], dtype=np.int64)
    try:
        return line_mesh.tube(
            radius=float(radius),
            n_sides=(
                TunnelViewer3D._HIGHLIGHT_BOND_SIDES_LOW
                if str(quality) == "low"
                else TunnelViewer3D._HIGHLIGHT_BOND_SIDES
            ),
            capping=True,
        )
    except Exception:
        return line_mesh


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


def _prepare_hbond_specs(
    atoms: list[dict],
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
                        anchors = _compute_highlight_hbond_anchors(
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
                    anchors = _compute_highlight_hbond_anchors(
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
                _build_dashed_segments(
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
            "radius": 0.06,
            "segments": np.ascontiguousarray(np.stack(hbond_segments).astype(np.float64)),
        }
    ]


def _prepare_highlight_mesh_payload(raw_atoms: list[dict], cache_key: tuple | None) -> dict | None:
    if pv is None or not cache_key:
        return None
    residue_ids, highlight_style, quality = cache_key
    atoms = [
        atom for atom in (raw_atoms or [])
        if int(atom.get("res_seq", 0) or 0) in residue_ids
    ]
    if not atoms:
        return None

    atom_positions = np.vstack([np.asarray(atom["position"], dtype=np.float64) for atom in atoms]).astype(np.float64)
    atom_elements = [
        str(atom.get("element") or "").strip().upper() or "C"
        for atom in atoms
    ]
    atom_radii = np.asarray(
        [_highlight_radius_for_element(element) for element in atom_elements],
        dtype=np.float64,
    )
    bond_radii = np.asarray(
        [_highlight_covalent_radius_for_element(element) for element in atom_elements],
        dtype=np.float64,
    )
    render_atoms = highlight_style in {"ball_and_stick", "spheres"}
    render_bonds = highlight_style in {"ball_and_stick", "sticks"}
    atom_radius_scale = float(TunnelViewer3D._HIGHLIGHT_STYLE_RADIUS_SCALE.get(highlight_style, 0.72))
    visible_atom_radii = atom_radii * atom_radius_scale if render_atoms else atom_radii

    atom_specs: list[dict] = []
    bond_specs: list[dict] = []
    hbond_specs: list[dict] = []

    if render_atoms:
        atoms_by_element: dict[str, list[np.ndarray]] = {}
        for element, position in zip(atom_elements, atom_positions):
            atoms_by_element.setdefault(element, []).append(position)
        for element, positions in atoms_by_element.items():
            radius = _highlight_radius_for_element(element) * atom_radius_scale
            atom_specs.append(
                {
                    "color": _highlight_color_for_element(element),
                    "radius": float(radius),
                    "positions": np.ascontiguousarray(np.vstack(positions).astype(np.float64)),
                }
            )

    if render_bonds:
        bond_pairs: list[tuple[int, int]] = []
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
                    (bond_radii[idx] + bond_radii[jdx]) * TunnelViewer3D._HIGHLIGHT_BOND_MAX_SCALE,
                )
                if distance <= max_distance:
                    bond_pairs.append((idx, jdx))

        bond_specs_by_color: dict[str, list[np.ndarray]] = {}
        bond_radius = (
            TunnelViewer3D._HIGHLIGHT_BOND_RADIUS
            if highlight_style == "ball_and_stick"
            else TunnelViewer3D._HIGHLIGHT_STICK_BOND_RADIUS
        )
        for start_idx, end_idx in bond_pairs:
            start = atom_positions[start_idx]
            end = atom_positions[end_idx]
            if render_atoms:
                anchors = _compute_highlight_bond_anchors(
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
            start_color = _highlight_color_for_element(atom_elements[start_idx])
            end_color = _highlight_color_for_element(atom_elements[end_idx])
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
        hbond_specs = _prepare_hbond_specs(
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


class _SequencePrepareWorker(QObject):
    prepared = Signal(int, int, object, str)

    def __init__(self):
        super().__init__()
        self._generation_lock = threading.Lock()
        self._current_generation = 0

    def set_current_generation(self, generation: int) -> None:
        with self._generation_lock:
            self._current_generation = int(generation)

    def _is_current_generation(self, generation: int) -> bool:
        with self._generation_lock:
            return int(generation) == int(self._current_generation)

    @Slot(int, int, str, str, int, object, bool)
    def prepare(
        self,
        generation: int,
        index: int,
        path: str,
        label: str,
        frame: int,
        highlight_cache_key: object,
        prepare_highlight: bool,
    ) -> None:
        payload = None
        error = ""
        try:
            if not self._is_current_generation(generation):
                self.prepared.emit(int(generation), int(index), None, "stale_generation")
                return
            payload = _prepare_sequence_payload(path, label, frame, build_ribbon=False)
            if not self._is_current_generation(generation):
                self.prepared.emit(int(generation), int(index), None, "stale_generation")
                return
            cache_key = highlight_cache_key if isinstance(highlight_cache_key, tuple) else None
            payload["highlight_cache_key"] = cache_key
            if prepare_highlight:
                try:
                    payload["highlight_mesh_payload"] = _prepare_highlight_mesh_payload(
                        payload.get("raw_atoms", []),
                        cache_key,
                    )
                except Exception as highlight_exc:
                    payload["highlight_mesh_payload"] = None
                    payload["highlight_error"] = str(highlight_exc)
        except Exception as exc:
            error = str(exc)
        self.prepared.emit(int(generation), int(index), payload, error)


class ProteinViewerPanel(QWidget):
    sequence_prepare_completed = Signal(int, int, object, str)
    _SEQUENCE_CACHE_RADIUS = 6
    _SEQUENCE_CACHE_LIMIT = 32
    _SEQUENCE_EAGER_RADIUS = 1
    _SEQUENCE_PREFETCH_DELAY_MS = 120
    _SEQUENCE_WORKER_COUNT = 3
    _SEQUENCE_MAX_URGENT_ACTIVE = 2
    _SEQUENCE_MAX_PREFETCH_ACTIVE = 2

    def __init__(self, parent=None, *, compact: bool = False):
        super().__init__(parent)
        self._compact = bool(compact)
        self._pdb_text = ""
        self._pdb_label = ""
        self._loaded = False
        self._temp_pdb_path: str | None = None
        self._highlight_residue_ids: tuple[int, ...] = ()
        self._highlight_style = "ball_and_stick"
        self._protein_opacity = 1.0
        self._path_overlay_payload: dict | None = None
        self._path_overlay_actors: list = []
        self._residue_label_actor = None
        self._deferred_highlight_focus = False
        self._deferred_highlight_preserve_cached = False
        self._sequence_folder = ""
        self._sequence_entries: list[dict] = []
        self._sequence_current_index = -1
        self._sequence_requested_index = -1
        self._sequence_cache: OrderedDict[int, dict] = OrderedDict()
        self._sequence_urgent_queue: deque[int] = deque()
        self._sequence_prefetch_queue: deque[int] = deque()
        self._sequence_pending_indexes: set[int] = set()
        self._sequence_force_prepare_indexes: set[int] = set()
        self._sequence_generation_lock = threading.Lock()
        self._sequence_generation = 0
        self._sequence_prefetch_center_index = -1
        self._sequence_prefetch_force_highlight_refresh = False
        self._sequence_active_tokens: set[tuple[int, int]] = set()
        self._sequence_task_lock = threading.Lock()
        self._sequence_task_channels: dict[tuple[int, int], str] = {}
        self._sequence_active_by_channel: dict[str, int] = {"urgent": 0, "prefetch": 0}

        layout = QVBoxLayout(self)
        if self._compact:
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(4)
        else:
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)

        self._style_combo = QComboBox()
        self._style_combo.addItem("Cartoon", "cartoon")
        self._style_combo.addItem("Backbone", "backbone")
        self._style_combo.addItem("Tube", "tube")
        self._style_combo.addItem("CA Spheres", "ca_spheres")
        if self._compact:
            self._style_combo.setCurrentIndex(1)
        self._style_combo.currentIndexChanged.connect(self._apply_style)

        self._reset_btn = QPushButton("Reset View")
        self._reset_btn.clicked.connect(self._reset_view)

        if self._compact:
            self._style_combo.setVisible(False)
            self._reset_btn.setVisible(False)
        else:
            layout.addWidget(self._style_combo)
            layout.addWidget(self._reset_btn)

        self._status_label = QLabel("No protein loaded.")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color:#606266;font-size:11px;")
        if not self._compact:
            layout.addWidget(self._status_label)

        self._viewer = TunnelViewer3D(self)
        self._viewer.set_selection_mode("path")
        self._viewer.set_lasso_mode(False)
        self._viewer.set_protein_highlight_style(self._highlight_style)
        self._viewer.set_protein_opacity(self._protein_opacity)
        if self._compact:
            self._viewer.set_protein_highlight_quality("low")
        if getattr(self._viewer, "_path_mode_btn", None) is not None:
            self._viewer._path_mode_btn.hide()
        layout.addWidget(self._viewer, 1)

        self._deferred_highlight_timer = QTimer(self)
        self._deferred_highlight_timer.setSingleShot(True)
        self._deferred_highlight_timer.timeout.connect(self._apply_deferred_highlight)

        self._sequence_prefetch_timer = QTimer(self)
        self._sequence_prefetch_timer.setSingleShot(True)
        self._sequence_prefetch_timer.timeout.connect(self._run_deferred_sequence_prefetch)

        self.sequence_prepare_completed.connect(self._on_sequence_entry_prepared)
        self._sequence_urgent_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ProteinSeqUrgent",
        )
        self._sequence_prefetch_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="ProteinSeqPrefetch",
        )

    def _log(self, message: str):
        print(f"[ProteinViewer] {message}", flush=True)

    def _advance_sequence_generation(self) -> int:
        with self._sequence_generation_lock:
            self._sequence_generation += 1
            return int(self._sequence_generation)

    def _current_sequence_generation(self) -> int:
        with self._sequence_generation_lock:
            return int(self._sequence_generation)

    def _is_current_sequence_generation(self, generation: int) -> bool:
        with self._sequence_generation_lock:
            return int(generation) == int(self._sequence_generation)

    def _prepare_sequence_entry_task(
        self,
        generation: int,
        index: int,
        path: str,
        label: str,
        frame: int,
        highlight_cache_key: object,
        prepare_highlight: bool,
    ) -> tuple[int, int, object, str]:
        payload = None
        error = ""
        try:
            if not self._is_current_sequence_generation(generation):
                return int(generation), int(index), None, "stale_generation"
            build_ribbon = str(self._style_combo.currentData() or "cartoon") == "cartoon"
            payload = _prepare_sequence_payload(path, label, frame, build_ribbon=build_ribbon)
            if not self._is_current_sequence_generation(generation):
                return int(generation), int(index), None, "stale_generation"
            cache_key = highlight_cache_key if isinstance(highlight_cache_key, tuple) else None
            payload["highlight_cache_key"] = cache_key
            if prepare_highlight and cache_key is not None:
                try:
                    payload["highlight_mesh_payload"] = _prepare_highlight_mesh_payload(
                        payload.get("raw_atoms", []),
                        cache_key,
                    )
                except Exception as highlight_exc:
                    payload["highlight_mesh_payload"] = None
                    payload["highlight_error"] = str(highlight_exc)
        except Exception as exc:
            error = str(exc)
        return int(generation), int(index), payload, error

    def _submit_sequence_prepare(
        self,
        index: int,
        *,
        urgent: bool,
        force: bool = False,
    ) -> None:
        if not self._sequence_entries:
            return
        index = max(0, min(len(self._sequence_entries) - 1, int(index)))
        entry = self._sequence_entries[index]
        generation = self._current_sequence_generation()
        token = (generation, index)
        channel = "urgent" if urgent else "prefetch"
        with self._sequence_task_lock:
            if token in self._sequence_active_tokens:
                return False
            self._sequence_active_tokens.add(token)
            self._sequence_task_channels[token] = channel
            self._sequence_active_by_channel[channel] = self._sequence_active_by_channel.get(channel, 0) + 1
        highlight_cache_key = self._current_highlight_cache_key()
        prepare_highlight = bool(
            highlight_cache_key
            and index == int(self._sequence_requested_index)
        )
        executor = self._sequence_urgent_executor if urgent else self._sequence_prefetch_executor
        future = executor.submit(
            self._prepare_sequence_entry_task,
            generation,
            index,
            str(entry["path"]),
            str(entry["label"]),
            int(entry["frame"]),
            highlight_cache_key,
            prepare_highlight if not force or urgent else True,
        )
        future.add_done_callback(
            lambda fut, gen=generation, idx=index: self._on_sequence_future_done(fut, gen, idx)
        )
        return True

    def _on_sequence_future_done(self, future, generation: int, index: int) -> None:
        try:
            gen, idx, payload, error = future.result()
        except Exception as exc:
            gen, idx, payload, error = int(generation), int(index), None, str(exc)
        with self._sequence_task_lock:
            channel = self._sequence_task_channels.pop((int(gen), int(idx)), None)
            self._sequence_active_tokens.discard((int(gen), int(idx)))
            if channel:
                self._sequence_active_by_channel[channel] = max(
                    0,
                    self._sequence_active_by_channel.get(channel, 0) - 1,
                )
        self.sequence_prepare_completed.emit(int(gen), int(idx), payload, error)

    def clear_protein(self):
        self._advance_sequence_generation()
        self._pdb_text = ""
        self._pdb_label = ""
        self._loaded = False
        self._highlight_residue_ids = ()
        self._sequence_requested_index = -1
        self._deferred_highlight_timer.stop()
        self._cancel_sequence_prefetch()
        self._cleanup_temp_file()
        self._status_label.setText("No protein loaded.")
        self.clear_highlight_labels(render=False)
        self._viewer.clear_protein()

    def clear_path_overlay(self):
        self._path_overlay_payload = None
        if not self._viewer:
            return
        for actor in self._path_overlay_actors:
            try:
                self._viewer._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._path_overlay_actors.clear()
        try:
            self._viewer._plotter.render()
        except Exception:
            pass

    def set_path_overlay(
        self,
        points: np.ndarray | None,
        radii: np.ndarray | None = None,
        path_ids: np.ndarray | None = None,
        color: str | None = None,
    ):
        self._path_overlay_payload = {
            "points": points,
            "radii": radii,
            "path_ids": path_ids,
            "color": color,
        }
        if not self._viewer or not self._viewer._plotter:
            return
        self._refresh_path_overlay()

    def _refresh_path_overlay(self):
        if not self._viewer or not self._viewer._plotter:
            return
        for actor in self._path_overlay_actors:
            try:
                self._viewer._plotter.remove_actor(actor, render=False)
            except Exception:
                pass
        self._path_overlay_actors.clear()
        payload = self._path_overlay_payload or {}
        points = payload.get("points")
        if points is None:
            self._viewer._plotter.render()
            return
        sphere_mesh = _build_path_overlay_mesh(points, payload.get("radii"))
        line_mesh = _build_path_overlay_centerline(points, payload.get("path_ids"))
        if sphere_mesh is None and line_mesh is None:
            self._viewer._plotter.render()
            return
        sphere_color = _normalize_overlay_color(payload.get("color"), fallback="#FFB000")
        line_color = _shade_overlay_color(sphere_color, 0.62, fallback="#9A6700")
        try:
            if line_mesh is not None:
                actor = self._viewer._plotter.add_mesh(
                    line_mesh,
                    color=line_color,
                    opacity=0.9,
                    line_width=1.6,
                    render_lines_as_tubes=False,
                    name="observer_path_overlay_line",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                self._path_overlay_actors.append(actor)
            if sphere_mesh is not None:
                actor = self._viewer._plotter.add_mesh(
                    sphere_mesh,
                    color=sphere_color,
                    opacity=0.95,
                    name="observer_path_overlay",
                    pickable=False,
                    reset_camera=False,
                    render=False,
                )
                self._path_overlay_actors.append(actor)
        except Exception:
            pass
        self._viewer._plotter.render()

    def clear_highlight_labels(self, *, render: bool = True):
        self._residue_label_actor = None
        plotter = getattr(self._viewer, "_plotter", None) if self._viewer else None
        if plotter is None:
            return
        for name in ("observer_residue_labels",):
            try:
                plotter.remove_actor(name, render=False)
            except Exception:
                pass
        if render:
            try:
                plotter.render()
            except Exception:
                pass

    def _refresh_highlight_labels(self, *, render: bool = True):
        plotter = getattr(self._viewer, "_plotter", None) if self._viewer else None
        if plotter is None:
            return
        self.clear_highlight_labels(render=False)
        if not self._highlight_residue_ids:
            if render:
                plotter.render()
            return
        model_cache = getattr(self._viewer, "_protein_model_cache", None)
        atoms = list((model_cache or {}).get("atoms_aligned", []) or [])
        if not atoms:
            if render:
                plotter.render()
            return
        residue_points: dict[int, list[np.ndarray]] = {}
        for atom in atoms:
            try:
                rid = int(atom.get("res_seq", 0) or 0)
            except Exception:
                continue
            if rid not in self._highlight_residue_ids:
                continue
            residue_points.setdefault(rid, []).append(np.asarray(atom["position"], dtype=np.float64))
        points = []
        labels = []
        for rid in self._highlight_residue_ids:
            atom_points = residue_points.get(int(rid), [])
            if not atom_points:
                continue
            center = np.vstack(atom_points).mean(axis=0)
            points.append(center)
            labels.append(str(int(rid)))
        if not points:
            if render:
                plotter.render()
            return
        try:
            self._residue_label_actor = plotter.add_point_labels(
                np.asarray(points, dtype=np.float64),
                labels,
                name="observer_residue_labels",
                font_size=14,
                text_color="#111827",
                shape_color="#FFFFFF",
                shape_opacity=0.72,
                point_size=0,
                always_visible=True,
                render=False,
            )
        except Exception:
            self._residue_label_actor = None
        if render:
            try:
                plotter.render()
            except Exception:
                pass

    @staticmethod
    def _extract_first_model(pdb_text: str) -> str:
        return _extract_first_model_text(pdb_text)

    def load_pdb_file(self, path: str, *, first_model_only: bool = False):
        pdb_path = Path(path)
        self._advance_sequence_generation()
        self._cancel_sequence_prefetch()
        self._sequence_folder = ""
        self._sequence_entries = []
        self._sequence_current_index = -1
        self._sequence_requested_index = -1
        self._sequence_cache.clear()
        return self.load_pdb_text(
            pdb_path.read_text(encoding="utf-8", errors="replace"),
            pdb_path.name,
            first_model_only=first_model_only,
        )

    def load_pdb_text(
        self,
        pdb_text: str,
        label: str = "protein.pdb",
        *,
        first_model_only: bool = False,
        preserve_view: bool = False,
        focus_highlight: bool = True,
    ):
        text = self._extract_first_model(pdb_text) if first_model_only else str(pdb_text or "")
        self._pdb_text = text
        self._pdb_label = str(label or "protein.pdb")
        self._loaded = bool(self._pdb_text.strip())
        self._log(
            f"load_pdb_text label={self._pdb_label} loaded={self._loaded} "
            f"chars={len(self._pdb_text)} first_model_only={first_model_only}"
        )
        if not self._loaded:
            self.clear_protein()
            return {"ok": False, "error": "empty_pdb"}

        temp_path = self._write_temp_pdb(self._pdb_text)
        style = str(self._style_combo.currentData() or "cartoon")
        result = self._viewer.load_protein_from_file(
            temp_path,
            style=style,
            align_mode="none",
            residue_anchor_positions=None,
        )
        if not bool(result.get("ok")):
            self._loaded = False
            error = str(result.get("error") or "unknown_error")
            self._status_label.setText(f"Failed to load protein: {error}")
            self._log(f"load failed error={error}")
            return result

        hint = "Loaded first frame." if first_model_only else "Loaded structure."
        self._status_label.setText(f"{hint} {self._pdb_label}")
        self._viewer.set_protein_opacity(self._protein_opacity)
        self._refresh_highlight_labels()
        if not preserve_view:
            if focus_highlight and self._highlight_residue_ids:
                focused = self._viewer.focus_protein_residues(self._highlight_residue_ids)
                if not focused:
                    self._viewer.reset_camera()
            else:
                self._viewer.reset_camera()
        return {
            "ok": True,
            "label": self._pdb_label,
        }

    def _apply_style(self):
        style = str(self._style_combo.currentData() or "cartoon")
        if self._sequence_entries and 0 <= self._sequence_current_index < len(self._sequence_entries):
            payload = self._prepare_sequence_entry(self._sequence_current_index)
            needs_ribbon = self._payload_needs_ribbon_refresh(payload, style)
            if payload is not None and not needs_ribbon:
                self._viewer.load_protein_from_prepared(
                    source_key=str(payload.get("source_key") or self._pdb_label or "<sequence_frame>"),
                    raw_atoms=payload.get("raw_atoms", []),
                    backbone=payload.get("backbone"),
                    ribbon_raw=payload.get("ribbon_raw"),
                    style=style,
                    align_mode="none",
                    residue_anchor_positions=None,
                )
                self._viewer.set_protein_opacity(self._protein_opacity)
                self._schedule_deferred_sequence_prefetch(self._sequence_current_index)
                return
            self._enqueue_sequence_prepare(self._sequence_current_index, urgent=True, force=needs_ribbon)
            return
        self._viewer.set_protein_style(style)

    def set_model_style(self, style: str):
        normalized = str(style or "backbone").strip().lower().replace("-", "_")
        if normalized not in {"cartoon", "backbone", "tube", "ca_spheres"}:
            normalized = "backbone"
        idx = self._style_combo.findData(normalized)
        if idx < 0:
            idx = self._style_combo.findData("backbone")
        if idx >= 0 and self._style_combo.currentIndex() != idx:
            self._style_combo.setCurrentIndex(idx)
        else:
            self._apply_style()

    def _reset_view(self):
        self._viewer.reset_camera()
        self.force_refresh()

    def force_refresh(self):
        if not self.isVisible():
            return
        self._viewer.update()
        self._viewer.repaint()
        plotter = getattr(self._viewer, "_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass

    def set_highlight_residues(
        self,
        residue_ids,
        *,
        focus_loaded: bool = True,
        refresh_sequence_cache: bool = True,
        defer_render: bool = False,
    ):
        normalized = tuple(
            sorted({int(rid) for rid in (residue_ids or []) if int(rid) > 0})
        )
        changed = normalized != self._highlight_residue_ids
        self._highlight_residue_ids = normalized
        if defer_render:
            return
        if self._sequence_entries and changed and refresh_sequence_cache and not self._compact:
            force_highlight_refresh = bool(self._current_highlight_cache_key())
            target_index = (
                self._sequence_current_index
                if self._sequence_current_index >= 0
                else self._sequence_requested_index
            )
            if 0 <= target_index < len(self._sequence_entries):
                self._enqueue_sequence_prepare(
                    target_index,
                    urgent=True,
                    force=force_highlight_refresh,
                )
            if self._sequence_current_index >= 0:
                self._prepare_sequence_neighbors(
                    self._sequence_current_index,
                    force_highlight_refresh=force_highlight_refresh,
                )
                self._schedule_sequence_prefetch(
                    self._sequence_current_index,
                    force_highlight_refresh=force_highlight_refresh,
                )
        if changed or (self._compact and self._loaded):
            self._schedule_highlight_refresh(focus=focus_loaded)
        elif self._loaded and focus_loaded and self._highlight_residue_ids:
            self._viewer.focus_protein_residues(self._highlight_residue_ids)
        if self._loaded:
            self._refresh_highlight_labels()

    def set_highlight_style(self, style: str):
        normalized = str(style or "ball_and_stick").strip().lower().replace("-", "_")
        if normalized not in {"ball_and_stick", "sticks", "spheres"}:
            normalized = "ball_and_stick"
        self._highlight_style = normalized
        self._viewer.set_protein_highlight_style(normalized)
        if self._sequence_entries and self._sequence_current_index >= 0 and not self._compact:
            force_highlight_refresh = bool(self._current_highlight_cache_key())
            self._prepare_sequence_neighbors(
                self._sequence_current_index,
                force_highlight_refresh=force_highlight_refresh,
            )
            self._schedule_sequence_prefetch(
                self._sequence_current_index,
                force_highlight_refresh=force_highlight_refresh,
            )
        if self._loaded and self._highlight_residue_ids and self._compact:
            self._schedule_highlight_refresh(focus=False)

    def set_protein_opacity(self, opacity: float):
        try:
            value = float(opacity)
        except (TypeError, ValueError):
            value = 1.0
        self._protein_opacity = max(0.0, min(1.0, value))
        self._viewer.set_protein_opacity(self._protein_opacity)

    def _current_highlight_cache_key(self) -> tuple | None:
        quality = "low" if self._compact else "high"
        return _normalize_highlight_cache_key(
            self._highlight_residue_ids,
            self._highlight_style,
            quality,
        )

    @staticmethod
    def _frame_number_from_name(path: str, fallback_index: int) -> int:
        matches = re.findall(r"\d+", os.path.basename(str(path or "")))
        if matches:
            try:
                return int(matches[-1])
            except ValueError:
                pass
        return int(fallback_index)

    def has_sequence(self) -> bool:
        return bool(self._sequence_entries)

    def has_loaded_protein(self) -> bool:
        return bool(self._loaded)

    def sequence_folder(self) -> str:
        return str(self._sequence_folder or "")

    def current_sequence_frame(self) -> int | None:
        if 0 <= self._sequence_current_index < len(self._sequence_entries):
            return int(self._sequence_entries[self._sequence_current_index]["frame"])
        return None

    def available_sequence_frames(self) -> tuple[int, ...]:
        return tuple(int(entry["frame"]) for entry in self._sequence_entries)

    def sequence_frame_paths(self) -> dict[int, str]:
        return {
            int(entry["frame"]): str(entry["path"])
            for entry in self._sequence_entries
            if str(entry.get("path") or "").strip()
        }

    def load_pdb_sequence_folder(self, folder: str) -> dict:
        self._advance_sequence_generation()
        self._cancel_sequence_prefetch()
        self._sequence_folder = ""
        self._sequence_entries = []
        self._sequence_current_index = -1
        self._sequence_requested_index = -1
        self._sequence_cache.clear()

        root = os.path.abspath(str(folder or "").strip())
        if not root or not os.path.isdir(root):
            self._status_label.setText("No protein loaded.")
            return {"ok": False, "error": "invalid_folder"}

        pdb_paths: list[str] = []
        try:
            for current_root, _dirs, files in os.walk(root):
                for name in files:
                    if name.lower().endswith(".pdb"):
                        pdb_paths.append(os.path.join(current_root, name))
        except OSError:
            pdb_paths = []
        if not pdb_paths:
            self._status_label.setText("No protein loaded.")
            return {"ok": False, "error": "no_pdb_files"}

        entries: list[dict] = []
        for idx, path in enumerate(sorted(pdb_paths), start=1):
            entries.append(
                {
                    "frame": self._frame_number_from_name(path, idx),
                    "path": os.path.abspath(path),
                    "label": os.path.basename(path),
                }
            )
        entries.sort(key=lambda entry: (int(entry["frame"]), str(entry["label"]).lower()))

        self._sequence_folder = root
        self._sequence_entries = entries
        first_frame = int(entries[0]["frame"])
        result = self.set_sequence_frame(first_frame)
        if not bool(result.get("ok")):
            self._sequence_folder = ""
            self._sequence_entries = []
            self._sequence_current_index = -1
            self._sequence_requested_index = -1
            self._sequence_cache.clear()
            return result
        return {
            "ok": True,
            "frame_count": len(entries),
            "frame_min": int(entries[0]["frame"]),
            "frame_max": int(entries[-1]["frame"]),
            "current_frame": int(result.get("frame", first_frame)),
            "folder": root,
        }

    def set_sequence_frame(self, frame_number: int) -> dict:
        if not self._sequence_entries:
            return {"ok": False, "error": "sequence_unavailable"}
        target = int(frame_number)
        best_index = min(
            range(len(self._sequence_entries)),
            key=lambda idx: (
                abs(int(self._sequence_entries[idx]["frame"]) - target),
                idx,
            ),
        )
        return self._load_sequence_index(best_index)

    def step_sequence_frame(self, delta: int) -> dict:
        if not self._sequence_entries:
            return {"ok": False, "error": "sequence_unavailable"}
        if self._sequence_current_index < 0:
            return self._load_sequence_index(0)
        next_index = max(0, min(len(self._sequence_entries) - 1, self._sequence_current_index + int(delta)))
        return self._load_sequence_index(next_index)

    def _prepare_sequence_entry(self, index: int) -> dict | None:
        cached = self._sequence_cache.get(index)
        if cached is not None:
            self._sequence_cache.move_to_end(index)
            return cached
        return None

    def _payload_needs_ribbon_refresh(self, payload: dict | None, style: str | None = None) -> bool:
        if not isinstance(payload, dict):
            return False
        normalized_style = str(style or self._style_combo.currentData() or "cartoon").strip().lower().replace("-", "_")
        if normalized_style != "cartoon":
            return False
        ribbon_raw = payload.get("ribbon_raw")
        return ribbon_raw is None or getattr(ribbon_raw, "n_points", 0) <= 0

    def _payload_needs_highlight_refresh(
        self,
        payload: dict | None,
        cache_key: tuple | None = None,
    ) -> bool:
        if not isinstance(payload, dict):
            return False
        current_key = cache_key if isinstance(cache_key, tuple) else self._current_highlight_cache_key()
        if current_key is None:
            return False
        if payload.get("highlight_cache_key") != current_key:
            return True
        return not isinstance(payload.get("highlight_mesh_payload"), dict)

    def _trim_sequence_cache(self, center_index: int) -> None:
        if not self._sequence_cache:
            return
        keep_min = max(0, center_index - self._SEQUENCE_CACHE_RADIUS)
        keep_max = min(len(self._sequence_entries) - 1, center_index + self._SEQUENCE_CACHE_RADIUS)
        for cache_index in list(self._sequence_cache.keys()):
            if cache_index < keep_min or cache_index > keep_max:
                self._sequence_cache.pop(cache_index, None)
        while len(self._sequence_cache) > self._SEQUENCE_CACHE_LIMIT:
            self._sequence_cache.popitem(last=False)

    def _cancel_sequence_prefetch(self) -> None:
        self._sequence_prefetch_timer.stop()
        self._sequence_prefetch_center_index = -1
        self._sequence_prefetch_force_highlight_refresh = False
        self._sequence_urgent_queue.clear()
        self._sequence_prefetch_queue.clear()
        self._sequence_pending_indexes.clear()
        self._sequence_force_prepare_indexes.clear()

    def _clear_queued_sequence_work(self) -> None:
        self._sequence_prefetch_timer.stop()
        self._sequence_prefetch_center_index = -1
        self._sequence_prefetch_force_highlight_refresh = False
        self._sequence_urgent_queue.clear()
        self._sequence_prefetch_queue.clear()
        self._sequence_pending_indexes.clear()
        self._sequence_force_prepare_indexes.clear()

    def _active_sequence_count(self, channel: str) -> int:
        with self._sequence_task_lock:
            return int(self._sequence_active_by_channel.get(str(channel), 0))

    def _schedule_highlight_refresh(self, *, focus: bool, preserve_cached: bool = False) -> None:
        self._deferred_highlight_focus = bool(focus and self._highlight_residue_ids)
        self._deferred_highlight_preserve_cached = bool(preserve_cached)
        if not self._compact or not self._loaded:
            if not preserve_cached:
                self._viewer.set_protein_highlight_mesh_payload(None)
            self._viewer.set_protein_highlight_render_enabled(True, refresh=False)
            self._viewer.set_protein_highlight_residues(
                self._highlight_residue_ids,
                refresh=not preserve_cached,
            )
            if self._deferred_highlight_focus:
                self._viewer.focus_protein_residues(self._highlight_residue_ids)
            self._deferred_highlight_preserve_cached = False
            return
        if not preserve_cached:
            self._viewer.set_protein_highlight_mesh_payload(None)
        self._viewer.set_protein_highlight_render_enabled(False, refresh=False)
        self._viewer.set_protein_highlight_residues(self._highlight_residue_ids, refresh=False)
        self._viewer.set_protein_highlight_render_enabled(True, refresh=False)
        self._viewer.set_protein_highlight_residues(
            self._highlight_residue_ids,
            refresh=not preserve_cached,
        )
        if self._deferred_highlight_focus and self._highlight_residue_ids:
            self._viewer.focus_protein_residues(self._highlight_residue_ids)
        self._deferred_highlight_focus = False
        self._deferred_highlight_preserve_cached = False

    def _apply_deferred_highlight(self) -> None:
        preserve_cached = bool(self._deferred_highlight_preserve_cached)
        self._viewer.set_protein_highlight_render_enabled(True, refresh=False)
        self._viewer.set_protein_highlight_residues(
            self._highlight_residue_ids,
            refresh=not preserve_cached,
        )
        self._refresh_highlight_labels(render=False)
        if self._deferred_highlight_focus and self._highlight_residue_ids:
            self._viewer.focus_protein_residues(self._highlight_residue_ids)
        self._deferred_highlight_focus = False
        self._deferred_highlight_preserve_cached = False

    def _remove_pending_sequence_index(self, index: int) -> None:
        try:
            self._sequence_urgent_queue.remove(index)
        except ValueError:
            pass
        try:
            self._sequence_prefetch_queue.remove(index)
        except ValueError:
            pass
        self._sequence_pending_indexes.discard(index)
        self._sequence_force_prepare_indexes.discard(index)

    def _enqueue_sequence_prepare(
        self,
        index: int,
        *,
        urgent: bool,
        force: bool = False,
        dispatch: bool = True,
    ) -> None:
        if not self._sequence_entries:
            return
        index = max(0, min(len(self._sequence_entries) - 1, int(index)))
        cached = self._sequence_cache.get(index)
        if cached is not None:
            self._sequence_cache.move_to_end(index)
            needs_refresh = self._payload_needs_highlight_refresh(cached) or self._payload_needs_ribbon_refresh(cached)
            if not force or not needs_refresh:
                return
        token = (self._current_sequence_generation(), index)
        with self._sequence_task_lock:
            if token in self._sequence_active_tokens:
                return
        self._remove_pending_sequence_index(index)
        if urgent:
            self._sequence_urgent_queue.appendleft(index)
        else:
            self._sequence_prefetch_queue.append(index)
        self._sequence_pending_indexes.add(index)
        if force:
            self._sequence_force_prepare_indexes.add(index)
        if dispatch:
            self._dispatch_next_sequence_prepare()

    def _dispatch_next_sequence_prepare(self) -> None:
        if not self._sequence_entries:
            return
        while self._sequence_urgent_queue and self._active_sequence_count("urgent") < self._SEQUENCE_MAX_URGENT_ACTIVE:
            index = self._sequence_urgent_queue.popleft()
            self._sequence_pending_indexes.discard(index)
            force_refresh = index in self._sequence_force_prepare_indexes
            if index < 0 or index >= len(self._sequence_entries):
                self._sequence_force_prepare_indexes.discard(index)
                continue
            cached = self._sequence_cache.get(index)
            if cached is not None:
                self._sequence_cache.move_to_end(index)
                needs_refresh = self._payload_needs_highlight_refresh(cached) or self._payload_needs_ribbon_refresh(cached)
                if not force_refresh or not needs_refresh:
                    self._sequence_force_prepare_indexes.discard(index)
                    continue
            self._sequence_force_prepare_indexes.discard(index)
            self._submit_sequence_prepare(index, urgent=True, force=force_refresh)
        while (
            not self._sequence_urgent_queue
            and self._sequence_prefetch_queue
            and self._active_sequence_count("prefetch") < self._SEQUENCE_MAX_PREFETCH_ACTIVE
        ):
            index = self._sequence_prefetch_queue.popleft()
            self._sequence_pending_indexes.discard(index)
            force_refresh = index in self._sequence_force_prepare_indexes
            if index < 0 or index >= len(self._sequence_entries):
                self._sequence_force_prepare_indexes.discard(index)
                continue
            cached = self._sequence_cache.get(index)
            if cached is not None:
                self._sequence_cache.move_to_end(index)
                needs_refresh = self._payload_needs_highlight_refresh(cached) or self._payload_needs_ribbon_refresh(cached)
                if not force_refresh or not needs_refresh:
                    self._sequence_force_prepare_indexes.discard(index)
                    continue
            self._sequence_force_prepare_indexes.discard(index)
            self._submit_sequence_prepare(index, urgent=False, force=force_refresh)

    def _prepare_sequence_neighbors(
        self,
        center_index: int,
        *,
        force_highlight_refresh: bool = False,
        dispatch: bool = True,
    ) -> None:
        if not self._sequence_entries:
            return
        force_refresh = bool(force_highlight_refresh and self._current_highlight_cache_key())
        for offset in range(1, self._SEQUENCE_EAGER_RADIUS + 1):
            for index in (center_index - offset, center_index + offset):
                if index < 0 or index >= len(self._sequence_entries):
                    continue
                self._enqueue_sequence_prepare(
                    index,
                    urgent=True,
                    force=force_refresh,
                    dispatch=dispatch,
                )
        self._trim_sequence_cache(center_index)

    def _schedule_sequence_prefetch(
        self,
        center_index: int,
        *,
        force_highlight_refresh: bool = False,
        dispatch: bool = True,
    ) -> None:
        if not self._sequence_entries:
            return
        current_highlight_cache_key = (
            self._current_highlight_cache_key() if force_highlight_refresh else None
        )
        current_style = str(self._style_combo.currentData() or "cartoon")
        self._sequence_prefetch_queue.clear()
        self._sequence_pending_indexes = {
            idx for idx in self._sequence_pending_indexes
            if idx in self._sequence_urgent_queue
        }
        start = max(0, center_index - self._SEQUENCE_CACHE_RADIUS)
        end = min(len(self._sequence_entries), center_index + self._SEQUENCE_CACHE_RADIUS + 1)
        pending = []
        for index in range(start, end):
            if index == center_index:
                continue
            if index in self._sequence_cache:
                cached = self._sequence_cache.get(index)
                if cached is not None:
                    self._sequence_cache.move_to_end(index)
                    needs_highlight = self._payload_needs_highlight_refresh(cached, current_highlight_cache_key)
                    needs_ribbon = self._payload_needs_ribbon_refresh(cached, current_style)
                    if not needs_highlight and not needs_ribbon:
                        continue
            pending.append(index)
        pending.sort(key=lambda idx: (abs(idx - center_index), idx))
        for index in pending:
            if index in self._sequence_pending_indexes:
                continue
            self._sequence_prefetch_queue.append(index)
            self._sequence_pending_indexes.add(index)
        if dispatch:
            self._dispatch_next_sequence_prepare()

    def _schedule_deferred_sequence_prefetch(
        self,
        center_index: int,
        *,
        force_highlight_refresh: bool = False,
    ) -> None:
        if not self._sequence_entries:
            return
        self._sequence_prefetch_center_index = int(center_index)
        self._sequence_prefetch_force_highlight_refresh = bool(force_highlight_refresh)
        self._sequence_prefetch_timer.start(int(self._SEQUENCE_PREFETCH_DELAY_MS))

    def _run_deferred_sequence_prefetch(self) -> None:
        if not self._sequence_entries:
            return
        center_index = int(self._sequence_prefetch_center_index)
        if center_index < 0 or center_index >= len(self._sequence_entries):
            return
        if center_index != int(self._sequence_current_index):
            return
        force_refresh = bool(self._sequence_prefetch_force_highlight_refresh)
        self._prepare_sequence_neighbors(
            center_index,
            force_highlight_refresh=force_refresh,
            dispatch=False,
        )
        self._schedule_sequence_prefetch(
            center_index,
            force_highlight_refresh=force_refresh,
        )

    def _load_sequence_index(self, index: int) -> dict:
        if not self._sequence_entries:
            return {"ok": False, "error": "sequence_unavailable"}
        index = max(0, min(len(self._sequence_entries) - 1, int(index)))
        entry = self._sequence_entries[index]
        self._sequence_requested_index = index
        self._clear_queued_sequence_work()
        cached = self._sequence_cache.get(index)
        if cached is not None:
            self._sequence_cache.move_to_end(index)
            return self._render_sequence_payload(index, cached)
        self._status_label.setText(f"Loading frame {entry['frame']}... {entry['label']}")
        self._submit_sequence_prepare(index, urgent=True, force=True)
        self._dispatch_next_sequence_prepare()
        return {
            "ok": True,
            "pending": True,
            "frame": int(entry["frame"]),
            "index": index,
            "label": str(entry["label"]),
            "path": str(entry["path"]),
        }

    def _render_sequence_payload(self, index: int, payload: dict) -> dict:
        entry = self._sequence_entries[index]
        preserve_view = self._loaded and self._sequence_current_index >= 0
        current_highlight_cache_key = self._current_highlight_cache_key()
        cached_highlight_payload = None
        if payload.get("highlight_cache_key") == current_highlight_cache_key:
            cached_highlight_payload = payload.get("highlight_mesh_payload")
        self._viewer.set_protein_highlight_mesh_payload(cached_highlight_payload)
        if self._highlight_residue_ids:
            self._deferred_highlight_timer.stop()
            self._viewer.set_protein_highlight_render_enabled(True, refresh=False)
            self._viewer.set_protein_highlight_residues(
                self._highlight_residue_ids,
                refresh=not bool(cached_highlight_payload),
                render=False,
            )
        result = self._viewer.load_protein_from_prepared(
            source_key=str(payload.get("source_key") or entry["path"]),
            raw_atoms=payload.get("raw_atoms", []),
            backbone=payload.get("backbone"),
            ribbon_raw=payload.get("ribbon_raw"),
            style=str(self._style_combo.currentData() or "cartoon"),
            align_mode="none",
            residue_anchor_positions=None,
            render=False,
        )
        if not bool(result.get("ok")):
            self._loaded = False
            error = str(result.get("error") or "unknown_error")
            self._status_label.setText(f"Failed to load protein: {error}")
            return result
        self._viewer.set_protein_opacity(self._protein_opacity, render=False)
        self._pdb_text = ""
        self._pdb_label = f"{entry['label']} | frame {entry['frame']}"
        self._loaded = True
        self._status_label.setText(f"Loaded frame {entry['frame']}. {entry['label']}")
        self._refresh_highlight_labels(render=False)
        if not preserve_view:
            if self._highlight_residue_ids:
                focused = self._viewer.focus_protein_residues(
                    self._highlight_residue_ids,
                    render=False,
                )
                if not focused:
                    self._viewer.reset_camera()
            else:
                self._viewer.reset_camera()
        self._sequence_current_index = index
        plotter = getattr(self._viewer, "_plotter", None)
        if plotter is not None:
            try:
                plotter.render()
            except Exception:
                pass
        self._schedule_deferred_sequence_prefetch(index)
        return {
            "ok": True,
            "frame": int(entry["frame"]),
            "index": index,
            "label": str(entry["label"]),
            "path": str(entry["path"]),
        }

    @Slot(int, int, object, str)
    def _on_sequence_entry_prepared(
        self,
        generation: int,
        index: int,
        payload: object,
        error: str,
    ) -> None:
        if int(generation) != int(self._sequence_generation):
            self._dispatch_next_sequence_prepare()
            return
        if not error and isinstance(payload, dict):
            self._sequence_cache[int(index)] = payload
            self._sequence_cache.move_to_end(int(index))
            self._trim_sequence_cache(int(index))
            if int(index) == int(self._sequence_requested_index):
                self._render_sequence_payload(int(index), payload)
        elif int(index) == int(self._sequence_requested_index):
            self._status_label.setText(f"Failed to load frame: {error or 'unknown_error'}")
        self._dispatch_next_sequence_prepare()

    def _write_temp_pdb(self, pdb_text: str) -> str:
        self._cleanup_temp_file()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".pdb",
            prefix="observer_",
            encoding="utf-8",
            delete=False,
        )
        try:
            handle.write(pdb_text)
            self._temp_pdb_path = handle.name
        finally:
            handle.close()
        return str(self._temp_pdb_path)

    def _cleanup_temp_file(self):
        path = str(self._temp_pdb_path or "").strip()
        self._temp_pdb_path = None
        if not path:
            return
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass

    def shutdown(self):
        self._advance_sequence_generation()
        self._deferred_highlight_timer.stop()
        self._cancel_sequence_prefetch()
        self._cleanup_temp_file()
        viewer = getattr(self, "_viewer", None)
        if viewer is not None and hasattr(viewer, "shutdown"):
            try:
                viewer.shutdown()
            except Exception:
                pass
        for executor_name in ("_sequence_urgent_executor", "_sequence_prefetch_executor"):
            executor = getattr(self, executor_name, None)
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)
