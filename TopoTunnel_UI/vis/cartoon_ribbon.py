"""
Protein cartoon ribbon mesh generator.

Builds a PyMOL-style cartoon ribbon mesh from PDB data using:
  backbone spline → oriented Frenet frames → cross-section sweep → triangle mesh.

Secondary structure is assigned from PDB HELIX/SHEET records when available,
otherwise computed from backbone phi/psi dihedral angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

try:
    import pyvista as pv
except ImportError:
    pv = None  # type: ignore[assignment]

# ─── Constants ────────────────────────────────────────────────────────

N_PROFILE_VERTS = 24
SAMPLES_PER_RESIDUE = 8

# Cross-section dimensions (Angstroms)
HELIX_HALF_WIDTH = 1.25
HELIX_HALF_THICK = 0.42
SHEET_HALF_WIDTH = 1.35
SHEET_HALF_THICK = 0.34
COIL_RADIUS = 0.4
SHEET_ARROW_WIDTH = 1.95
ARROW_TAPER_RESIDUES = 1.5

# Dihedral angle thresholds for SS assignment (degrees)
HELIX_PHI_CENTER, HELIX_PHI_TOL = -57.0, 30.0
HELIX_PSI_CENTER, HELIX_PSI_TOL = -47.0, 30.0
SHEET_PHI_CENTER, SHEET_PHI_TOL = -120.0, 40.0
SHEET_PSI_CENTER, SHEET_PSI_TOL = 130.0, 40.0

MIN_HELIX_LEN = 4
MIN_SHEET_LEN = 2
MAX_CA_CA_DIST = 5.0
TRANSITION_SPAN = 1.0


# ─── Data Structures ─────────────────────────────────────────────────

@dataclass
class ResidueGuide:
    """Backbone guide atoms for one residue."""
    chain_id: str
    res_seq: int
    ca: np.ndarray            # (3,)
    o: Optional[np.ndarray] = None   # (3,)
    n_atom: Optional[np.ndarray] = None   # (3,) amide nitrogen
    c_atom: Optional[np.ndarray] = None   # (3,) carbonyl carbon


# ─── Utility ──────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector(s).  Works for 1-D or 2-D (row-wise)."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 1e-12 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


# ─── 1. Guide Atom Extraction ────────────────────────────────────────

def extract_guide_atoms(
    atoms: Sequence[dict],
) -> Dict[str, List[ResidueGuide]]:
    """Group atoms by chain/residue and extract CA, O, N, C positions."""
    buckets: Dict[Tuple[str, int], Dict[str, np.ndarray]] = {}
    for a in atoms:
        key = (a.get("chain_id", ""), int(a.get("res_seq", 0)))
        name = a.get("atom_name", "")
        if name in ("CA", "O", "N", "C"):
            buckets.setdefault(key, {})[name] = np.asarray(
                a["position"], dtype=np.float64
            )

    chains: Dict[str, List[ResidueGuide]] = {}
    for (cid, rseq), grp in sorted(buckets.items()):
        ca = grp.get("CA")
        if ca is None:
            continue
        rg = ResidueGuide(
            chain_id=cid,
            res_seq=rseq,
            ca=ca,
            o=grp.get("O"),
            n_atom=grp.get("N"),
            c_atom=grp.get("C"),
        )
        chains.setdefault(cid, []).append(rg)

    for lst in chains.values():
        lst.sort(key=lambda r: r.res_seq)
    return chains


# ─── 2. Chain Segment Splitting ───────────────────────────────────────

def split_chain_segments(
    residues: List[ResidueGuide],
    max_gap: float = MAX_CA_CA_DIST,
) -> List[List[ResidueGuide]]:
    """Split chain at large CA-CA gaps or residue number jumps."""
    if not residues:
        return []
    segments: List[List[ResidueGuide]] = [[residues[0]]]
    for i in range(1, len(residues)):
        dist = float(np.linalg.norm(residues[i].ca - residues[i - 1].ca))
        gap = abs(residues[i].res_seq - residues[i - 1].res_seq)
        if dist > max_gap or gap > 5:
            segments.append([])
        segments[-1].append(residues[i])
    return segments


# ─── 3. Parse HELIX / SHEET Records ──────────────────────────────────

def parse_ss_records(pdb_path: str) -> Dict[Tuple[str, int], str]:
    """Parse PDB HELIX/SHEET records.  Returns (chain, resseq) → 'H'/'E'."""
    ss_map: Dict[Tuple[str, int], str] = {}
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                rec = line[:6].strip()
                if rec == "HELIX":
                    try:
                        chain = line[19]
                        start = int(line[21:25])
                        end = int(line[33:37])
                        for r in range(start, end + 1):
                            ss_map[(chain, r)] = "H"
                    except (IndexError, ValueError):
                        continue
                elif rec == "SHEET":
                    try:
                        chain = line[21]
                        start = int(line[22:26])
                        end = int(line[33:37])
                        for r in range(start, end + 1):
                            ss_map[(chain, r)] = "E"
                    except (IndexError, ValueError):
                        continue
    except OSError:
        pass
    return ss_map


# ─── 4. Dihedral Angle Computation ───────────────────────────────────

def compute_dihedral(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray,
) -> float:
    """Compute dihedral angle in degrees for four consecutive points."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_len = np.linalg.norm(n1)
    n2_len = np.linalg.norm(n2)
    if n1_len < 1e-12 or n2_len < 1e-12:
        return 0.0
    n1 /= n1_len
    n2 /= n2_len
    b2_hat = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2_hat)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return math.degrees(math.atan2(y, x))


# ─── 5. SS Assignment from Geometry ──────────────────────────────────

def _in_helix_region(phi: float, psi: float) -> bool:
    """Check if phi/psi falls in helix region (standard or mirror)."""
    return (
        (abs(phi - HELIX_PHI_CENTER) <= HELIX_PHI_TOL and
         abs(psi - HELIX_PSI_CENTER) <= HELIX_PSI_TOL) or
        (abs(phi + HELIX_PHI_CENTER) <= HELIX_PHI_TOL and
         abs(psi + HELIX_PSI_CENTER) <= HELIX_PSI_TOL)
    )


def _in_sheet_region(phi: float, psi: float) -> bool:
    """Check if phi/psi falls in sheet region (standard or mirror)."""
    return (
        (abs(phi - SHEET_PHI_CENTER) <= SHEET_PHI_TOL and
         abs(psi - SHEET_PSI_CENTER) <= SHEET_PSI_TOL) or
        (abs(phi + SHEET_PHI_CENTER) <= SHEET_PHI_TOL and
         abs(psi + SHEET_PSI_CENTER) <= SHEET_PSI_TOL)
    )


def assign_ss_from_geometry(residues: List[ResidueGuide]) -> List[str]:
    """Assign secondary structure from backbone phi/psi dihedrals.

    Checks both standard and mirror Ramachandran regions to handle
    PDB files from MD simulations with varying chirality conventions.
    """
    n = len(residues)
    raw: List[str] = ["C"] * n

    for i in range(1, n - 1):
        prev, cur, nxt = residues[i - 1], residues[i], residues[i + 1]
        # Need C of previous residue, N/CA/C of current, N of next
        if prev.c_atom is None or cur.n_atom is None or cur.ca is None:
            continue
        if cur.c_atom is None or nxt.n_atom is None:
            continue

        phi = compute_dihedral(prev.c_atom, cur.n_atom, cur.ca, cur.c_atom)
        psi = compute_dihedral(cur.n_atom, cur.ca, cur.c_atom, nxt.n_atom)

        if _in_helix_region(phi, psi):
            raw[i] = "H"
        elif _in_sheet_region(phi, psi):
            raw[i] = "E"

    # Post-filter: remove short runs
    result = list(raw)
    _filter_short_runs(result, "H", MIN_HELIX_LEN)
    _filter_short_runs(result, "E", MIN_SHEET_LEN)
    return result


def _filter_short_runs(ss: List[str], target: str, min_len: int) -> None:
    """Replace runs of *target* shorter than *min_len* with 'C' in-place."""
    i = 0
    while i < len(ss):
        if ss[i] == target:
            j = i
            while j < len(ss) and ss[j] == target:
                j += 1
            if j - i < min_len:
                for k in range(i, j):
                    ss[k] = "C"
            i = j
        else:
            i += 1


# ─── 6. Spline Construction ──────────────────────────────────────────

def build_spline(
    ca_positions: np.ndarray,
    samples_per_res: int = SAMPLES_PER_RESIDUE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit cubic spline through CA positions and sample densely.

    Returns (positions, derivatives, t_values) each with M rows.
    """
    n = len(ca_positions)
    t_knots = np.arange(n, dtype=np.float64)
    n_samples = max(samples_per_res * (n - 1) + 1, 2)
    t_eval = np.linspace(0, n - 1, n_samples)

    if n >= 4:
        cs = CubicSpline(t_knots, ca_positions, bc_type="natural")
        positions = cs(t_eval)
        derivatives = cs(t_eval, 1)
    elif n == 3:
        # Quadratic interpolation per component
        positions = np.column_stack([
            interp1d(t_knots, ca_positions[:, d], kind="quadratic")(t_eval)
            for d in range(3)
        ])
        # Numerical derivative
        dt = t_eval[1] - t_eval[0] if len(t_eval) > 1 else 1.0
        derivatives = np.gradient(positions, dt, axis=0)
    else:
        # Linear for 2 points
        positions = np.column_stack([
            np.interp(t_eval, t_knots, ca_positions[:, d])
            for d in range(3)
        ])
        derivatives = np.gradient(positions, t_eval[1] - t_eval[0] if len(t_eval) > 1 else 1.0, axis=0)

    return positions, derivatives, t_eval


# ─── 7. O-direction Interpolation ────────────────────────────────────

def interpolate_o_directions(
    residues: List[ResidueGuide],
    t_values: np.ndarray,
) -> np.ndarray:
    """Interpolate normalized O-atom directions to spline sample points."""
    n = len(residues)
    t_knots = np.arange(n, dtype=np.float64)
    o_dirs = np.zeros((n, 3), dtype=np.float64)

    for i, r in enumerate(residues):
        if r.o is not None:
            d = r.o - r.ca
            norm = np.linalg.norm(d)
            o_dirs[i] = d / norm if norm > 1e-12 else np.array([0, 1, 0])
        else:
            o_dirs[i] = np.array([0, 1, 0])

    # Synthesize O directions for residues that had none
    # (already handled above with [0,1,0] fallback, but try neighbor-based)
    for i in range(n):
        if residues[i].o is None:
            # Use cross product of adjacent CA vectors as approximate normal
            if 0 < i < n - 1:
                v1 = residues[i].ca - residues[i - 1].ca
                v2 = residues[i + 1].ca - residues[i].ca
                cr = np.cross(v1, v2)
                norm = np.linalg.norm(cr)
                if norm > 1e-12:
                    o_dirs[i] = cr / norm

    # Ensure consistent handedness: flip directions that reverse relative to neighbors
    for i in range(1, n):
        if np.dot(o_dirs[i], o_dirs[i - 1]) < 0:
            o_dirs[i] = -o_dirs[i]

    # Interpolate
    interp_dirs = np.column_stack([
        interp1d(t_knots, o_dirs[:, d], kind="linear", fill_value="extrapolate")(
            t_values
        )
        for d in range(3)
    ])
    return _normalize(interp_dirs)


# ─── 8. Frenet Frame Computation ─────────────────────────────────────

def compute_frames(
    positions: np.ndarray,
    derivatives: np.ndarray,
    o_directions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute oriented (T, N, B) frames via Gram-Schmidt on O-direction."""
    m = len(positions)
    T = _normalize(derivatives)
    N = np.zeros_like(T)
    B = np.zeros_like(T)

    for i in range(m):
        o_dir = o_directions[i]
        t = T[i]

        # Gram-Schmidt: remove tangential component from O-direction
        o_proj = o_dir - np.dot(o_dir, t) * t
        o_proj_len = np.linalg.norm(o_proj)

        if o_proj_len > 1e-6:
            N[i] = o_proj / o_proj_len
        else:
            # Degenerate: O-direction parallel to tangent
            if i > 0:
                N[i] = N[i - 1]
            else:
                # Pick arbitrary perpendicular
                perp = np.array([1, 0, 0]) if abs(t[0]) < 0.9 else np.array([0, 1, 0])
                o_proj = perp - np.dot(perp, t) * t
                N[i] = o_proj / np.linalg.norm(o_proj)

        B[i] = np.cross(t, N[i])
        b_len = np.linalg.norm(B[i])
        if b_len > 1e-12:
            B[i] /= b_len

    # Smooth N vectors with a small moving average to reduce jitter
    if m > 3:
        kernel = np.array([0.25, 0.5, 0.25])
        for d in range(3):
            N[1:-1, d] = np.convolve(N[:, d], kernel, mode="valid")
        # Re-normalize and recompute B
        for i in range(m):
            t = T[i]
            N[i] = N[i] - np.dot(N[i], t) * t
            n_len = np.linalg.norm(N[i])
            if n_len > 1e-12:
                N[i] /= n_len
            B[i] = np.cross(t, N[i])
            b_len = np.linalg.norm(B[i])
            if b_len > 1e-12:
                B[i] /= b_len

    return T, N, B


# ─── 9. Cross-Section Profiles ───────────────────────────────────────

def make_cross_section_profiles(
    n_verts: int = N_PROFILE_VERTS,
) -> Dict[str, np.ndarray]:
    """Pre-compute 2-D cross-section profiles for H/E/C.

    Returns dict mapping type → (n_verts, 2) array of (normal, binormal) offsets.
    """
    theta = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Helix: rounded rectangle for a fuller PyMOL-like ribbon body.
    helix_eps = 0.72
    helix = np.column_stack([
        HELIX_HALF_WIDTH * np.sign(cos_t) * np.abs(cos_t) ** helix_eps,
        HELIX_HALF_THICK * np.sign(sin_t) * np.abs(sin_t) ** helix_eps,
    ])

    # Sheet: slightly thicker superellipse so beta strands do not look collapsed.
    eps = 0.58  # lower = sharper corners
    sheet = np.column_stack([
        SHEET_HALF_WIDTH * np.sign(cos_t) * np.abs(cos_t) ** eps,
        SHEET_HALF_THICK * np.sign(sin_t) * np.abs(sin_t) ** eps,
    ])

    # Coil: keep loops visibly connected to widened helix/sheet sections.
    coil = np.column_stack([COIL_RADIUS * cos_t, COIL_RADIUS * sin_t])

    return {"H": helix, "E": sheet, "C": coil}


# ─── 10. Per-sample Cross-section with Transitions & Arrows ──────────

def _compute_per_sample_cross_sections(
    t_values: np.ndarray,
    ss_per_residue: List[str],
    profiles: Dict[str, np.ndarray],
    n_residues: int,
) -> List[np.ndarray]:
    """Compute cross-section at each spline sample, handling transitions and arrows."""
    m = len(t_values)
    n_verts = profiles["C"].shape[0]
    sections: List[np.ndarray] = []

    # Precompute sheet strand ends (C-terminal end of each 'E' run)
    sheet_ends: List[int] = []  # residue indices where sheet strands end
    for i in range(n_residues):
        if ss_per_residue[i] == "E":
            if i == n_residues - 1 or ss_per_residue[i + 1] != "E":
                sheet_ends.append(i)

    for si in range(m):
        t = t_values[si]
        res_idx = int(round(t))
        res_idx = max(0, min(res_idx, n_residues - 1))
        ss_type = ss_per_residue[res_idx]

        profile = profiles[ss_type].copy()

        # Check for transition morphing
        frac = t - int(t)
        left_res = max(0, int(t))
        right_res = min(n_residues - 1, left_res + 1)
        if left_res != right_res:
            ss_left = ss_per_residue[left_res]
            ss_right = ss_per_residue[right_res]
            if ss_left != ss_right:
                # Morph between profiles
                alpha = frac
                profile = (1.0 - alpha) * profiles[ss_left] + alpha * profiles[ss_right]

        # Sheet arrow logic
        if ss_type == "E":
            for end_idx in sheet_ends:
                dist_to_end = end_idx - t  # positive means before end
                if -0.5 <= dist_to_end <= ARROW_TAPER_RESIDUES:
                    if dist_to_end > 0:
                        # Widen toward arrow base
                        widen = 1.0 + (SHEET_ARROW_WIDTH / SHEET_HALF_WIDTH - 1.0) * (
                            1.0 - dist_to_end / ARROW_TAPER_RESIDUES
                        )
                        profile[:, 0] *= widen
                    else:
                        # Taper to point after end
                        taper = max(0.0, 1.0 + dist_to_end * 2.0)
                        profile[:, 0] *= taper
                    break

        sections.append(profile)
    return sections


# ─── 11. Mesh Assembly ───────────────────────────────────────────────

def build_tube_mesh(
    positions: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    cross_sections: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep cross-sections along spline to build triangle mesh.

    Returns (vertices, faces, normals).
    """
    m = len(positions)
    v = cross_sections[0].shape[0]

    # Ring vertices: v[i*V+j] = pos[i] + cs[i][j,0]*N[i] + cs[i][j,1]*B[i]
    total_verts = m * v + 2  # +2 for end caps
    vertices = np.empty((total_verts, 3), dtype=np.float64)
    normals = np.empty((total_verts, 3), dtype=np.float64)

    for i in range(m):
        cs = cross_sections[i]  # (V, 2)
        base = i * v
        for j in range(v):
            vertices[base + j] = (
                positions[i] + cs[j, 0] * N[i] + cs[j, 1] * B[i]
            )
            # Analytical normal
            n_dir = cs[j, 0] * N[i] + cs[j, 1] * B[i]
            n_len = np.linalg.norm(n_dir)
            if n_len > 1e-12:
                normals[base + j] = n_dir / n_len
            else:
                normals[base + j] = T[i]  # collapsed vertex (arrow tip)

    # Cap centers
    vertices[m * v] = positions[0]
    vertices[m * v + 1] = positions[-1]
    normals[m * v] = -T[0]
    normals[m * v + 1] = T[-1]

    # Side faces: 2 triangles per quad
    n_side_tris = (m - 1) * v * 2
    n_cap_tris = v * 2
    faces = np.empty((n_side_tris + n_cap_tris, 3), dtype=np.int64)

    fi = 0
    for i in range(m - 1):
        for j in range(v):
            j_next = (j + 1) % v
            a = i * v + j
            b_idx = i * v + j_next
            c = (i + 1) * v + j
            d = (i + 1) * v + j_next
            faces[fi] = [a, c, b_idx]
            faces[fi + 1] = [c, d, b_idx]
            fi += 2

    # Start cap
    cap_start = m * v
    for j in range(v):
        j_next = (j + 1) % v
        faces[fi] = [cap_start, j_next, j]
        fi += 1

    # End cap
    cap_end = m * v + 1
    last_ring = (m - 1) * v
    for j in range(v):
        j_next = (j + 1) % v
        faces[fi] = [cap_end, last_ring + j, last_ring + j_next]
        fi += 1

    return vertices, faces, normals


# ─── 12. Top-level Entry Point ────────────────────────────────────────

def generate_cartoon_ribbon(
    pdb_path: str,
    atoms: Optional[Sequence[dict]] = None,
    ss_records: Optional[Dict[Tuple[str, int], str]] = None,
    samples_per_res: int = SAMPLES_PER_RESIDUE,
    n_profile_verts: int = N_PROFILE_VERTS,
) -> Optional["pv.PolyData"]:
    """Generate a cartoon ribbon mesh from PDB data.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file (used for SS record parsing).
    atoms : list of dict, optional
        Pre-parsed atoms from ``parse_pdb_atoms()``.
        If *None*, the PDB file is parsed via the same function.
    ss_records : dict, optional
        Explicit secondary-structure mapping ``(chain_id, res_seq) -> 'H'/'E'``.
        When provided, missing residues still fall back to geometry-based assignment.
    samples_per_res : int
        Spline sampling density (points per residue).
    n_profile_verts : int
        Vertices per cross-section ring.

    Returns
    -------
    pv.PolyData with triangle faces and point normals, or *None* on failure.
    """
    if pv is None:
        return None

    # Parse atoms if not provided
    if atoms is None:
        from TopoTunnel_UI.vis.protein_model import parse_pdb_atoms
        atoms = parse_pdb_atoms(pdb_path)

    if not atoms:
        return None

    # Extract guide atoms per chain
    chains = extract_guide_atoms(atoms)
    if not chains:
        return None

    # Parse SS records from PDB unless the caller already supplied them.
    if ss_records is None:
        ss_records = parse_ss_records(pdb_path)

    # Pre-compute cross-section profiles
    profiles = make_cross_section_profiles(n_profile_verts)

    # Accumulate mesh parts from all chain segments
    all_verts: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    vert_offset = 0

    for chain_id, residues in chains.items():
        segments = split_chain_segments(residues)
        for segment in segments:
            if len(segment) < 2:
                continue

            # SS assignment
            if ss_records:
                ss = []
                missing_indices: List[int] = []
                for idx, residue in enumerate(segment):
                    value = ss_records.get((residue.chain_id, residue.res_seq), "")
                    if value in {"H", "E"}:
                        ss.append(value)
                    else:
                        ss.append("")
                        missing_indices.append(idx)

                if missing_indices:
                    geom_ss = assign_ss_from_geometry(segment)
                    for idx in missing_indices:
                        ss[idx] = geom_ss[idx]

                ss = [value if value in {"H", "E"} else "C" for value in ss]
            else:
                ss = assign_ss_from_geometry(segment)

            # Build spline
            ca_pos = np.array([r.ca for r in segment], dtype=np.float64)
            positions, derivatives, t_values = build_spline(
                ca_pos, samples_per_res
            )

            # Interpolate O-directions and compute frames
            o_dirs = interpolate_o_directions(segment, t_values)
            T, N_frame, B = compute_frames(positions, derivatives, o_dirs)

            # Compute per-sample cross-sections
            cross_sections = _compute_per_sample_cross_sections(
                t_values, ss, profiles, len(segment),
            )

            # Build tube mesh
            verts, faces, norms = build_tube_mesh(
                positions, T, N_frame, B, cross_sections,
            )

            # Offset face indices and accumulate
            all_verts.append(verts)
            all_faces.append(faces + vert_offset)
            all_normals.append(norms)
            vert_offset += len(verts)

    if not all_verts:
        return None

    # Merge all segments
    merged_verts = np.vstack(all_verts)
    merged_faces = np.vstack(all_faces)
    merged_normals = np.vstack(all_normals)

    # Build PyVista PolyData
    # PyVista face format: [n_verts, v0, v1, v2, ...]
    n_faces = len(merged_faces)
    pv_faces = np.empty(n_faces * 4, dtype=np.int64)
    pv_faces[0::4] = 3
    pv_faces[1::4] = merged_faces[:, 0]
    pv_faces[2::4] = merged_faces[:, 1]
    pv_faces[3::4] = merged_faces[:, 2]

    mesh = pv.PolyData(merged_verts, pv_faces)
    mesh.point_data["Normals"] = merged_normals
    return mesh
