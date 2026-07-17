"""
Protein model parsing and alignment helpers for VTK rendering.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Optional, Sequence

import numpy as np

_HBOND_SIDECHAIN_DONORS = {
    "ARG": {"NE", "NH1", "NH2"},
    "ASN": {"ND2"},
    "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"},
    "HID": {"ND1", "NE2"},
    "HIE": {"ND1", "NE2"},
    "HIP": {"ND1", "NE2"},
    "LYS": {"NZ"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TRP": {"NE1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
}

_HBOND_SIDECHAIN_ACCEPTORS = {
    "ASN": {"OD1"},
    "GLN": {"OE1"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
    "HID": {"ND1", "NE2"},
    "HIE": {"ND1", "NE2"},
    "SER": {"OG"},
    "THR": {"OG1"},
    "TYR": {"OH"},
    "CYS": {"SG"},
    "MET": {"SD"},
}


def _normalize_element_symbol(raw: str) -> str:
    token = str(raw or "").strip().upper()
    if not token:
        return ""
    if len(token) >= 2 and token[:2].isalpha():
        two = token[:2]
        if two in {"CL", "BR", "NA", "MG", "FE", "ZN", "CA"}:
            return two
    if token[0].isalpha():
        return token[0]
    return ""


def normalize_atom_name(atom_name: str) -> str:
    return str(atom_name or "").strip().upper()


def residue_key(atom: dict) -> tuple[str, int, str]:
    return (
        str(atom.get("chain_id", "") or "").strip(),
        int(atom.get("res_seq", 0) or 0),
        str(atom.get("res_name", "") or "").strip().upper(),
    )


def is_hbond_donor_atom(atom: dict) -> bool:
    atom_name = normalize_atom_name(atom.get("atom_name", ""))
    res_name = str(atom.get("res_name", "") or "").strip().upper()
    element = str(atom.get("element", "") or "").strip().upper()
    if atom_name == "N":
        return res_name != "PRO"
    if atom_name in _HBOND_SIDECHAIN_DONORS.get(res_name, set()):
        return True
    return res_name.startswith("HET") and element in {"N", "O", "S"}


def is_hbond_acceptor_atom(atom: dict) -> bool:
    atom_name = normalize_atom_name(atom.get("atom_name", ""))
    res_name = str(atom.get("res_name", "") or "").strip().upper()
    element = str(atom.get("element", "") or "").strip().upper()
    if atom_name in {"O", "OXT"}:
        return True
    if atom_name in _HBOND_SIDECHAIN_ACCEPTORS.get(res_name, set()):
        return True
    return res_name.startswith("HET") and element in {"N", "O", "S"}


def _guess_element(atom_name: str) -> str:
    stripped = atom_name.strip()
    if not stripped:
        return "C"
    stripped = stripped.lstrip("0123456789")
    if not stripped:
        return "C"
    # Handle two-letter elements from atom name prefix when available.
    if len(stripped) >= 2 and stripped[:2].isalpha():
        two = stripped[:2].upper()
        if two in {"CL", "BR", "NA", "MG", "FE", "ZN", "CA"}:
            return two
    return stripped[0].upper()


def _extract_element_from_line(line: str, atom_name: str) -> str:
    stripped = str(line or "").rstrip()
    if stripped:
        match = re.search(r"([A-Za-z]{1,2})\s*$", stripped)
        if match:
            element = _normalize_element_symbol(match.group(1))
            if element:
                return element
        for token in reversed(stripped.split()):
            if token.isalpha():
                element = _normalize_element_symbol(token)
                if element:
                    return element

    if len(line) >= 78:
        element = _normalize_element_symbol(line[76:78])
        if element:
            return element

    return _guess_element(atom_name)


def _parse_atom_line(line: str) -> Optional[dict]:
    record = line[:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None

    atom_name = line[12:16].strip()
    res_name = line[17:20].strip()
    chain_id = line[21:22].strip()
    try:
        res_seq = int(line[22:26].strip())
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None

    element = _extract_element_from_line(line, atom_name)

    return {
        "position": np.array([x, y, z], dtype=np.float64),
        "element": element,
        "res_seq": res_seq,
        "res_name": res_name,
        "atom_name": atom_name,
        "chain_id": chain_id,
    }


def parse_pdb_atoms(pdb_path: str) -> list[dict]:
    """Parse ATOM/HETATM records from a PDB file."""
    path = Path(pdb_path)
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    atoms: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            atom = _parse_atom_line(line)
            if atom is not None:
                atoms.append(atom)
    return atoms


def parse_pdb_text(pdb_text: str) -> list[dict]:
    """Parse ATOM/HETATM records directly from PDB text."""
    atoms: list[dict] = []
    for line in str(pdb_text or "").splitlines():
        atom = _parse_atom_line(line)
        if atom is not None:
            atoms.append(atom)
    return atoms


def build_backbone_data(atoms: Sequence[dict], max_gap: int = 5) -> dict:
    """Build CA backbone segments and residue-indexed CA anchor map."""
    ca_atoms = [a for a in atoms if a.get("atom_name") == "CA"]
    ca_atoms.sort(key=lambda a: (str(a.get("chain_id", "")), int(a.get("res_seq", 0))))

    segments: list[dict] = []
    for i in range(len(ca_atoms) - 1):
        cur = ca_atoms[i]
        nxt = ca_atoms[i + 1]
        if cur["chain_id"] != nxt["chain_id"]:
            continue
        if abs(int(nxt["res_seq"]) - int(cur["res_seq"])) > max_gap:
            continue
        segments.append(
            {
                "source": cur["position"],
                "target": nxt["position"],
                "chain_id": cur["chain_id"],
                "res_seq_from": int(cur["res_seq"]),
                "res_seq_to": int(nxt["res_seq"]),
            }
        )

    # If multiple chains share residue IDs, average them for robust coarse alignment.
    buckets: dict[int, list[np.ndarray]] = {}
    for atom in ca_atoms:
        rid = int(atom["res_seq"])
        buckets.setdefault(rid, []).append(np.asarray(atom["position"], dtype=np.float64))

    ca_by_resseq = {
        rid: np.mean(np.vstack(points), axis=0).astype(np.float64)
        for rid, points in buckets.items()
    }

    return {
        "ca_atoms": ca_atoms,
        "segments": segments,
        "ca_by_resseq": ca_by_resseq,
    }


def apply_transform(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Apply transform p' = (scale * p) * R^T + t."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    rot = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trans = np.asarray(t, dtype=np.float64).reshape(3)
    return (pts * float(scale)) @ rot.T + trans


def kabsch_rigid_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute rigid transform (R, t) minimizing squared error from src to dst."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"src/dst shape mismatch: {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"Expected (N,3) arrays, got {src.shape}")
    if src.shape[0] < 3:
        raise ValueError("Kabsch requires at least 3 matched points")

    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    src_c = src - src_center
    dst_c = dst - dst_center

    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = dst_center - (R @ src_center)
    fitted = apply_transform(src, R, t)
    rmsd = float(np.sqrt(np.mean(np.sum((fitted - dst) ** 2, axis=1))))
    return R.astype(np.float64), t.astype(np.float64), rmsd


def _estimate_uniform_scale(src_pts: np.ndarray, dst_pts: np.ndarray) -> float:
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        return 1.0

    src_c = src - src.mean(axis=0)
    dst_c = dst - dst.mean(axis=0)
    src_norm = np.linalg.norm(src_c, axis=1)
    dst_norm = np.linalg.norm(dst_c, axis=1)
    valid = src_norm > 1e-8
    if not np.any(valid):
        return 1.0

    ratios = dst_norm[valid] / src_norm[valid]
    scale = float(np.median(ratios))
    if not np.isfinite(scale) or scale <= 0:
        return 1.0
    return scale


def estimate_alignment(
    ca_by_resseq: Dict[int, np.ndarray],
    residue_anchor_positions: Dict[int, np.ndarray],
    mode: str = "auto_kabsch",
    min_matches: int = 20,
    allow_scale: bool = True,
) -> dict:
    """Estimate transform from PDB CA coordinates to tunnel residue anchor coordinates."""
    mode = str(mode or "auto_kabsch").lower()
    valid_modes = {"auto_kabsch", "translate", "none"}
    if mode not in valid_modes:
        mode = "auto_kabsch"

    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    scale = 1.0
    rmsd: Optional[float] = None

    src_map = {int(k): np.asarray(v, dtype=np.float64) for k, v in (ca_by_resseq or {}).items()}
    dst_map = {int(k): np.asarray(v, dtype=np.float64) for k, v in (residue_anchor_positions or {}).items()}
    matched_ids = sorted(set(src_map.keys()) & set(dst_map.keys()))

    if mode == "none":
        return {
            "mode": "none",
            "requested_mode": mode,
            "matched_count": len(matched_ids),
            "matched_residue_ids": matched_ids,
            "rmsd": None,
            "scale": 1.0,
            "R": R,
            "t": t,
            "reason": "alignment_disabled",
        }

    def _fallback_translate(reason: str) -> dict:
        nonlocal R, t, scale
        if matched_ids:
            src_pts = np.vstack([src_map[rid] for rid in matched_ids])
            dst_pts = np.vstack([dst_map[rid] for rid in matched_ids])
            if allow_scale:
                scale = _estimate_uniform_scale(src_pts, dst_pts)
            src_center = np.mean(src_pts, axis=0)
            dst_center = np.mean(dst_pts, axis=0)
        elif src_map and dst_map:
            src_pts = np.vstack(list(src_map.values()))
            dst_pts = np.vstack(list(dst_map.values()))
            src_center = np.mean(src_pts, axis=0)
            dst_center = np.mean(dst_pts, axis=0)
            if allow_scale:
                src_span = np.ptp(src_pts, axis=0)
                dst_span = np.ptp(dst_pts, axis=0)
                valid = src_span > 1e-8
                if np.any(valid):
                    raw = float(np.mean(dst_span[valid] / src_span[valid]))
                    if np.isfinite(raw) and raw > 0:
                        scale = raw
        else:
            src_center = np.zeros(3, dtype=np.float64)
            dst_center = np.zeros(3, dtype=np.float64)
        t = dst_center - src_center * scale
        return {
            "mode": "translate",
            "requested_mode": mode,
            "matched_count": len(matched_ids),
            "matched_residue_ids": matched_ids,
            "rmsd": None,
            "scale": float(scale),
            "R": R,
            "t": t,
            "reason": reason,
        }

    if mode == "translate":
        return _fallback_translate("translate_mode_requested")

    # auto_kabsch
    if len(matched_ids) < min_matches:
        return _fallback_translate(f"insufficient_matches_{len(matched_ids)}")

    src = np.vstack([src_map[rid] for rid in matched_ids])
    dst = np.vstack([dst_map[rid] for rid in matched_ids])

    try:
        R, t, rmsd = kabsch_rigid_transform(src, dst)
        return {
            "mode": "kabsch",
            "requested_mode": mode,
            "matched_count": len(matched_ids),
            "matched_residue_ids": matched_ids,
            "rmsd": float(rmsd),
            "scale": 1.0,
            "R": R,
            "t": t,
            "reason": "kabsch_success",
        }
    except Exception as exc:
        return _fallback_translate(f"kabsch_failed:{type(exc).__name__}")
