"""
Parse PyMOL .pse session files and extract molecular objects.

.pse files are Python pickle archives containing all scene objects,
coordinates, bonds, colors, and display settings from a PyMOL session.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── PyMOL color index → hex mapping (standard palette) ──────────────

_PYMOL_COLOR_TABLE: Dict[int, str] = {
    0: "#33FF33",   # green (carbon)
    1: "#00FFFF",   # cyan
    2: "#FF00FF",   # lightmagenta
    4: "#00FF00",   # green
    5: "#FFFF00",   # yellow
    6: "#00FF00",   # green
    7: "#FFA500",   # orange
    8: "#EE82EE",   # violet
    9: "#00FFFF",   # cyan
    10: "#FA8072",  # salmon
    11: "#00FF00",  # lime
    12: "#6A5ACD",  # slate
    13: "#008080",  # deepteal
    14: "#FF69B4",  # hotpink
    15: "#006400",  # darkgreen
    16: "#F5DEB3",  # wheat
    17: "#9400D3",  # deeppurple
    18: "#0000CD",  # marine
    19: "#808080",  # gray
    26: "#33FF33",  # carbon (element)
    27: "#3333FF",  # nitrogen
    28: "#FF4C4C",  # oxygen
    29: "#E5C53F",  # sulfur
    30: "#E6E6E6",  # hydrogen
    31: "#E6E6E6",  # hydrogen
}

# Default colors for binding sites when PyMOL color is ambiguous
BINDING_SITE_COLORS = ["#FFCC00", "#AA66FF", "#33CC66", "#00CCCC"]


@dataclass
class PseMolecule:
    """A single molecular object extracted from a .pse session."""
    name: str
    chain: str
    atoms: List[dict]       # [{name, elem, resn, resi, resv, chain, position, color_idx, ss, hetatm}]
    coords: np.ndarray      # (N, 3) float64
    bonds: List[Tuple[int, int]]
    color: str              # hex color string
    is_protein: bool        # True if standard amino acids dominate


# Standard amino acid residue names (3-letter codes)
_STANDARD_AA = frozenset({
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "HIE", "HID", "HIP", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
})


class _SafeUnpickler(pickle.Unpickler):
    """Unpickler that stubs out chempy/pymol module references."""

    def find_class(self, module: str, name: str):
        if module in ("builtins", "__builtin__", "copy_reg", "copyreg"):
            return super().find_class(module, name)
        # Stub any chempy or pymol class
        if module.startswith(("chempy", "pymol")):
            return _make_dummy(module, name)
        return super().find_class(module, name)


def _make_dummy(module: str, name: str):
    """Create a dummy class that silently accepts __setstate__."""
    class Dummy:
        def __init__(self, *args, **kwargs):
            pass
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self._state = state
    Dummy.__name__ = name
    Dummy.__qualname__ = f"{module}.{name}"
    return Dummy


def _pymol_color_to_hex(color_idx: int) -> str:
    return _PYMOL_COLOR_TABLE.get(color_idx, "#CCCCCC")


def _guess_element(atom_name: str) -> str:
    stripped = str(atom_name or "").strip().lstrip("0123456789")
    if not stripped:
        return "C"
    if len(stripped) >= 2 and stripped[:2].isalpha():
        two = stripped[:2].upper()
        if two in {"CL", "BR", "NA", "MG", "FE", "ZN", "CA"}:
            return two
    return stripped[0].upper()


def _parse_resv(value) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_pse_resseq(atom: dict) -> Optional[int]:
    resv = _parse_resv(atom.get("resv"))
    if resv is not None:
        return resv

    resi = str(atom.get("resi") or "").strip()
    if not resi:
        return None
    match = re.match(r"[-+]?\d+", resi)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _normalize_ss_label(value: str) -> str:
    ss = str(value or "").strip().upper()
    if ss == "H":
        return "H"
    if ss in {"S", "E"}:
        return "E"
    return ""


def _extract_molecule(names_entry: list) -> Optional[PseMolecule]:
    """Extract a single molecular object from a PSE names entry."""
    if not names_entry or not isinstance(names_entry, (list, tuple)):
        return None
    if len(names_entry) < 6:
        return None

    obj_name = names_entry[0]
    obj_data = names_entry[5]
    if not isinstance(obj_data, (list, tuple)) or len(obj_data) < 8:
        return None

    # obj_data layout for ObjectMolecule:
    # [0] = descriptor (14 items)
    # [2] = n_atoms
    # [3] = n_bonds
    # [4] = coordset list
    # [6] = bond/connectivity data
    # [7] = atom records list
    atom_records = obj_data[7]
    if not isinstance(atom_records, (list, tuple)) or len(atom_records) == 0:
        return None

    # Extract coordinates from first coordinate set
    coord_sets = obj_data[4]
    if not isinstance(coord_sets, (list, tuple)) or len(coord_sets) == 0:
        return None
    cs = coord_sets[0]
    if not isinstance(cs, (list, tuple)) or len(cs) < 3:
        return None
    coords_flat = cs[2]
    if not isinstance(coords_flat, (list, tuple)) or len(coords_flat) < 3:
        return None

    n_atoms = len(atom_records)
    coords = np.array(coords_flat, dtype=np.float64).reshape(-1, 3)

    # Atom records format (49 fields):
    # [0]=resv, [1]=chain, [2]=alt, [3]=resi, [4]=segi, [5]=resn,
    # [6]=name, [7]=elem, [8]=textType, [9]=label, [10]=ssType,
    # [11]=hetatm, [12]=bonded, [13]=id, [14]=b, [15]=q,
    # [16]=vdw, [17]=partialCharge, [18]=formalCharge, [19]=geom,
    # [20]=visRep, [21]=color, ...
    atoms = []
    chains = set()
    resn_counter: Dict[str, int] = {}
    color_indices = set()

    for i, rec in enumerate(atom_records):
        if not isinstance(rec, (list, tuple)) or len(rec) < 22:
            continue
        chain = str(rec[1]) if rec[1] else ""
        resv = _parse_resv(rec[0]) if len(rec) > 0 else None
        resn = str(rec[5]) if rec[5] else ""
        resi = str(rec[3]) if rec[3] else ""
        aname = str(rec[6]) if rec[6] else ""
        elem = str(rec[7]) if rec[7] else ""
        ss = str(rec[10]) if len(rec) > 10 and rec[10] else ""
        hetatm = bool(rec[11]) if len(rec) > 11 else False
        color_idx = int(rec[21]) if len(rec) > 21 else 0

        chains.add(chain)
        resn_counter[resn] = resn_counter.get(resn, 0) + 1
        color_indices.add(color_idx)

        coord = coords[i] if i < len(coords) else np.zeros(3)
        atoms.append({
            "name": aname,
            "elem": elem,
            "resn": resn,
            "resi": resi,
            "resv": resv,
            "chain": chain,
            "position": coord,
            "color_idx": color_idx,
            "ss": ss,
            "hetatm": hetatm,
        })

    if not atoms:
        return None

    # Determine dominant chain
    chain = max(chains, key=lambda c: sum(
        1 for a in atoms if a["chain"] == c
    )) if chains else ""

    # Determine if this is a protein (majority standard AA residues)
    aa_atom_count = sum(v for k, v in resn_counter.items() if k in _STANDARD_AA)
    is_protein = aa_atom_count > len(atoms) * 0.5

    # Determine representative color
    if color_indices:
        # Most common color index
        dominant_color_idx = max(color_indices, key=lambda c: sum(
            1 for a in atoms if a["color_idx"] == c
        ))
        color = _pymol_color_to_hex(dominant_color_idx)
    else:
        color = "#CCCCCC"

    # Extract bonds from obj_data[6]
    bonds: List[Tuple[int, int]] = []
    bond_data = obj_data[6]
    if isinstance(bond_data, (list, tuple)):
        for b in bond_data:
            if isinstance(b, (list, tuple)) and len(b) >= 3:
                try:
                    bonds.append((int(b[0]), int(b[1])))
                except (ValueError, TypeError):
                    pass

    return PseMolecule(
        name=obj_name,
        chain=chain,
        atoms=atoms,
        coords=coords[:len(atoms)],
        bonds=bonds,
        color=color,
        is_protein=is_protein,
    )


def parse_pse_file(pse_path: str) -> List[PseMolecule]:
    """Parse a PyMOL .pse session file and return all molecular objects.

    Returns a list of PseMolecule, with protein objects first,
    then ligand/binding site objects.
    """
    path = Path(pse_path)
    if not path.exists():
        raise FileNotFoundError(f"PSE file not found: {pse_path}")

    with open(path, "rb") as f:
        data = _SafeUnpickler(f, encoding="latin1").load()

    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"Invalid PSE file format: {pse_path}")

    molecules: List[PseMolecule] = []
    names_list = data["names"]

    for entry in names_list:
        if entry is None:
            continue
        mol = _extract_molecule(entry)
        if mol is not None:
            molecules.append(mol)

    # Assign default binding site colors for non-protein molecules
    bs_idx = 0
    for mol in molecules:
        if not mol.is_protein:
            mol.color = BINDING_SITE_COLORS[bs_idx % len(BINDING_SITE_COLORS)]
            bs_idx += 1

    # Sort: proteins first, then binding sites
    molecules.sort(key=lambda m: (0 if m.is_protein else 1, m.name))
    return molecules


def pse_protein_to_model_data(
    mol: PseMolecule,
) -> Tuple[List[dict], Dict[Tuple[str, int], str]]:
    """Convert a PSE protein object to the internal protein render format."""
    atoms: List[dict] = []
    ss_records: Dict[Tuple[str, int], str] = {}

    for atom in mol.atoms:
        res_seq = _parse_pse_resseq(atom)
        if res_seq is None:
            continue
        atom_name = str(atom.get("name") or "").strip()
        if not atom_name:
            continue
        try:
            position = np.asarray(atom["position"], dtype=np.float64).reshape(3)
        except Exception:
            continue

        chain_id = str(atom.get("chain") or "")[:1]
        res_name = str(atom.get("resn") or "UNK").strip().upper()[:3] or "UNK"
        element = str(atom.get("elem") or "").strip().upper()[:2]
        if not element:
            element = _guess_element(atom_name)

        atoms.append(
            {
                "position": position,
                "element": element,
                "res_seq": int(res_seq),
                "res_name": res_name,
                "atom_name": atom_name[:4],
                "chain_id": chain_id,
                "is_hetatm": bool(atom.get("hetatm", False)),
            }
        )

        ss = _normalize_ss_label(atom.get("ss", ""))
        if ss:
            ss_records.setdefault((chain_id, int(res_seq)), ss)

    return atoms, ss_records


def _build_pdb_helix_line(
    serial: int,
    start_resn: str,
    chain: str,
    start_seq: int,
    end_resn: str,
    end_seq: int,
) -> str:
    line = [" "] * 80
    line[0:6] = list("HELIX ")
    line[7:10] = list(f"{serial:>3d}")
    line[11:14] = list(f"{serial:>3d}")
    line[15:18] = list(f"{start_resn[:3]:>3s}")
    line[19] = chain[:1] if chain else " "
    line[21:25] = list(f"{start_seq:>4d}")
    line[27:30] = list(f"{end_resn[:3]:>3s}")
    line[31] = chain[:1] if chain else " "
    line[33:37] = list(f"{end_seq:>4d}")
    line[38:40] = list(f"{1:>2d}")
    line[71:76] = list(f"{end_seq - start_seq + 1:>5d}")
    return "".join(line).rstrip()


def _build_pdb_sheet_line(
    serial: int,
    start_resn: str,
    chain: str,
    start_seq: int,
    end_resn: str,
    end_seq: int,
) -> str:
    line = [" "] * 80
    line[0:6] = list("SHEET ")
    line[7:10] = list(f"{serial:>3d}")
    line[11:14] = list(f"{f'S{serial:02d}'[-3:]:>3s}")
    line[14:16] = list(f"{1:>2d}")
    line[17:20] = list(f"{start_resn[:3]:>3s}")
    line[21] = chain[:1] if chain else " "
    line[22:26] = list(f"{start_seq:>4d}")
    line[28:31] = list(f"{end_resn[:3]:>3s}")
    line[32] = chain[:1] if chain else " "
    line[33:37] = list(f"{end_seq:>4d}")
    line[38:40] = list(f"{0:>2d}")
    return "".join(line).rstrip()


def _build_pdb_ss_records(
    atoms: List[dict],
    ss_records: Dict[Tuple[str, int], str],
) -> List[str]:
    residue_meta: Dict[Tuple[str, int], dict] = {}
    for atom in atoms:
        key = (str(atom.get("chain_id", "")), int(atom.get("res_seq", 0)))
        residue_meta.setdefault(
            key,
            {
                "chain_id": key[0],
                "res_seq": key[1],
                "res_name": str(atom.get("res_name", "UNK"))[:3] or "UNK",
                "ss": ss_records.get(key, ""),
            },
        )

    residues = [
        residue_meta[key]
        for key in sorted(residue_meta.keys(), key=lambda item: (item[0], item[1]))
    ]

    records: List[str] = []
    helix_id = 0
    sheet_id = 0
    i = 0
    while i < len(residues):
        ss = residues[i]["ss"]
        if ss not in {"H", "E"}:
            i += 1
            continue

        start = residues[i]
        end = start
        j = i + 1
        while j < len(residues):
            cur = residues[j]
            if cur["ss"] != ss:
                break
            if cur["chain_id"] != end["chain_id"]:
                break
            if int(cur["res_seq"]) != int(end["res_seq"]) + 1:
                break
            end = cur
            j += 1

        if ss == "H":
            helix_id += 1
            records.append(
                _build_pdb_helix_line(
                    helix_id,
                    str(start["res_name"]),
                    str(start["chain_id"]),
                    int(start["res_seq"]),
                    str(end["res_name"]),
                    int(end["res_seq"]),
                )
            )
        else:
            sheet_id += 1
            records.append(
                _build_pdb_sheet_line(
                    sheet_id,
                    str(start["res_name"]),
                    str(start["chain_id"]),
                    int(start["res_seq"]),
                    str(end["res_name"]),
                    int(end["res_seq"]),
                )
            )
        i = j

    return records


def pse_protein_to_pdb_string(mol: PseMolecule) -> str:
    """Convert a PseMolecule (protein) to a PDB-format string.

    This enables reuse of the existing PDB loading pipeline for
    protein structure extracted from a .pse session.
    """
    atoms, ss_records = pse_protein_to_model_data(mol)
    lines = []
    lines.extend(_build_pdb_ss_records(atoms, ss_records))

    for serial, atom in enumerate(atoms, start=1):
        aname = str(atom["atom_name"])[:4].ljust(4)
        resn = str(atom["res_name"])[:3].ljust(3)
        chain = str(atom["chain_id"])[:1] if atom["chain_id"] else " "
        resi = f"{int(atom['res_seq']):4d}"
        pos = atom["position"]
        elem = str(atom["element"])[:2].rjust(2) if atom["element"] else "  "
        b_factor = 0.0
        occupancy = 1.0

        record = "HETATM" if atom.get("is_hetatm", False) else "ATOM  "
        line = (
            f"{record}{serial:5d} {aname:4s} {resn:3s} {chain:1s}"
            f"{resi:4s}    "
            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}"
            f"          {elem:2s}  "
        )
        lines.append(line)

    lines.append("END")
    return "\n".join(lines) + "\n"
