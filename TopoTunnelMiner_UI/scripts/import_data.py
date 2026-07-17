#!/usr/bin/env python3
"""
Import preprocessed path data (pkl) into SQLite database.

Usage:
    python import_data.py <pkl_path> [db_path]

Example:
    python import_data.py ../new-path/4.5.preprocessed_paths.pkl tunnel_data.db
"""
import sys
import os
import time
import pickle
import sqlite3
import struct
import numpy as np
from tqdm import tqdm

# Add parent dir to path for db module access
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.tunnel_db import TunnelDatabase, SCHEMA_SQL

# Kyte-Doolittle hydrophobicity scale (same as residue_hydrophobicity.py)
_KD_SCALE = {
    "ARG": -4.5, "LYS": -3.9, "ASN": -3.5, "ASP": -3.5,
    "GLN": -3.5, "GLU": -3.5, "HIS": -3.2, "PRO": -1.6,
    "TYR": -1.3, "TRP": -0.9, "SER": -0.8, "THR": -0.7,
    "GLY": -0.4, "ALA": 1.8, "MET": 1.9, "CYS": 2.5,
    "PHE": 2.8, "LEU": 3.8, "VAL": 4.2, "ILE": 2.5,
    # AMBER histidine variants
    "HID": -3.2, "HIE": -3.2, "HIP": -3.2,
}


def _resolve_path_cluster_id(path: dict) -> int:
    for key in ("tunnel_cluster", "Tunnel_id", "cluster_id"):
        value = path.get(key)
        if value is None:
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return 0


def _build_resseq_hydro_map(pdb_path: str) -> dict:
    """Build resseq → hydrophobicity mapping from a PDB file."""
    res_map: dict[int, float] = {}
    try:
        with open(pdb_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("ATOM"):
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                res_name = line[17:20].strip()
                try:
                    res_seq = int(line[22:26].strip())
                except ValueError:
                    continue
                res_map[res_seq] = _KD_SCALE.get(res_name, 0.0)
    except OSError:
        pass
    return res_map


def import_pkl_to_sqlite(pkl_path: str, db_path: str, pdb_path: str = ""):
    start = time.time()

    # Load pkl
    print(f"[1/6] Loading {pkl_path}...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    PATHS = data["PATHS"]
    INITIAL_COUNTS = data["INITIAL_COUNTS"]
    RESIDUE_POSITIONS = data["RESIDUE_POSITIONS"]
    FRAME_RANGE = data["FRAME_RANGE"]

    # Build residue hydrophobicity lookup from PDB (for recomputing qss when missing)
    resseq_hydro: dict = {}
    if pdb_path:
        resseq_hydro = _build_resseq_hydro_map(pdb_path)
        if resseq_hydro:
            print(f"       Built hydrophobicity map from PDB: {len(resseq_hydro)} residues")

    print(f"       Loaded {len(PATHS):,} paths, "
          f"{len(RESIDUE_POSITIONS):,} residues, "
          f"frames {FRAME_RANGE['min']}-{FRAME_RANGE['max']}")

    # Remove existing db
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"       Removed existing {db_path}")

    # Create database
    print(f"[2/6] Creating SQLite database: {db_path}")
    db = TunnelDatabase(db_path)
    db.connect()
    db.init_schema()
    conn = db.conn

    # Disable auto-commit for bulk insert performance
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-256000")  # 256MB

    # Import paths
    print(f"[3/6] Importing {len(PATHS):,} paths...")
    for idx, path in enumerate(tqdm(PATHS, desc="       Paths")):
        xs = path.get("xs", [])
        ys = path.get("ys", [])
        zs = path.get("zs", [])
        bf = path.get("bf", [])
        qss = path.get("qss", [])
        # If qss is missing or all-zero, try qss1-4 or recompute from PDB residue names
        if not qss or all(v == 0 for v in qss):
            qss1 = path.get("qss1", [])
            qss2 = path.get("qss2", [])
            qss3 = path.get("qss3", [])
            qss4 = path.get("qss4", [])
            if qss1 and qss2 and qss3 and qss4 and len(qss1) == len(qss2) == len(qss3) == len(qss4):
                qss = [
                    (float(qss1[i]) + float(qss2[i]) + float(qss3[i]) + float(qss4[i])) / 4.0
                    for i in range(len(qss1))
                ]
            elif resseq_hydro:
                # Recompute from 4 nearest-residue IDs via PDB residue name lookup
                r1 = path.get("res_1", [])
                r2 = path.get("res_2", [])
                r3 = path.get("res_3", [])
                r4 = path.get("res_4", [])
                n = len(xs)
                qss = []
                for i in range(n):
                    vals = []
                    for rl in (r1, r2, r3, r4):
                        if i < len(rl):
                            rid = int(rl[i])
                            if rid in resseq_hydro:
                                vals.append(resseq_hydro[rid])
                    qss.append(sum(vals) / len(vals) if vals else 0.0)
        frame_ids = path.get("frame_ids", [])
        frame_id = int(frame_ids[0]) if frame_ids and len(frame_ids) > 0 and not np.isnan(frame_ids[0]) else 0
        residues_set = path.get("residues_set", set())
        if not isinstance(residues_set, set):
            residues_set = set(residues_set)
        n = len(xs)

        if n < 2:
            continue

        # Compute stats
        bf_arr = np.array(bf, dtype=np.float32) if bf else np.zeros(n, dtype=np.float32)
        qss_arr = np.array(qss, dtype=np.float32) if qss else np.zeros(n, dtype=np.float32)
        min_r = float(np.min(bf_arr)) if len(bf_arr) > 0 else 0
        avg_r = float(np.mean(bf_arr)) if len(bf_arr) > 0 else 0
        avg_h = float(np.mean(qss_arr)) if len(qss_arr) > 0 else 0
        length_vals = path.get("length", [])
        path_length = float(length_vals[-1]) if length_vals else 0

        # Bounding box
        bbox = path.get("bbox")
        if bbox:
            bmin = bbox.get("min", [0, 0, 0])
            bmax = bbox.get("max", [0, 0, 0])
        else:
            bmin = [float(min(xs)), float(min(ys)), float(min(zs))]
            bmax = [float(max(xs)), float(max(ys)), float(max(zs))]

        # Insert path metadata
        conn.execute(
            "INSERT INTO paths VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                idx, frame_id, _resolve_path_cluster_id(path), n, path_length,
                min_r, avg_r, avg_h,
                ",".join(str(r) for r in sorted(residues_set)),
                len(residues_set),
                float(xs[0]), float(ys[0]), float(zs[0]),
                float(xs[-1]), float(ys[-1]), float(zs[-1]),
                bmin[0], bmin[1], bmin[2],
                bmax[0], bmax[1], bmax[2],
            ),
        )

        # Pack render coords as float32 BLOB
        render_arr = np.column_stack([xs, ys, zs]).astype(np.float32)
        render_blob = render_arr.tobytes()

        # Selection coords (original points before RDP)
        sel_xs = path.get("xs_selection", xs)
        sel_ys = path.get("ys_selection", ys)
        sel_zs = path.get("zs_selection", zs)
        sel_arr = np.column_stack([sel_xs, sel_ys, sel_zs]).astype(np.float32)
        sel_blob = sel_arr.tobytes()

        conn.execute(
            "INSERT INTO path_render_data VALUES (?,?,?)",
            (idx, render_blob, sel_blob),
        )

        # Residue inverted index
        for r in residues_set:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO residue_paths VALUES (?,?)",
                    (int(r), idx),
                )
            except (ValueError, TypeError):
                pass

        # Path points for profiles
        xo = path.get("xs_origin", xs)
        yo = path.get("ys_origin", ys)
        zo = path.get("zs_origin", zs)
        res_1 = path.get("res_1", [0] * n)
        res_2 = path.get("res_2", [0] * n)
        res_3 = path.get("res_3", [0] * n)
        res_4 = path.get("res_4", [0] * n)
        atom_1 = path.get("atom_1", [0] * n)
        atom_2 = path.get("atom_2", [0] * n)
        atom_3 = path.get("atom_3", [0] * n)
        atom_4 = path.get("atom_4", [0] * n)
        length_list = path.get("length", list(range(n)))
        throughput = path.get("throughput", [0.0] * n)
        occ = path.get("occupancy", [1.0] * n)

        point_data = []
        for i in range(n):
            point_data.append((
                idx, i,
                float(xs[i]), float(ys[i]), float(zs[i]),
                float(xo[i]) if i < len(xo) else float(xs[i]),
                float(yo[i]) if i < len(yo) else float(ys[i]),
                float(zo[i]) if i < len(zo) else float(zs[i]),
                float(bf[i]) if i < len(bf) else 0,
                float(qss[i]) if i < len(qss) else 0,
                float(length_list[i]) if i < len(length_list) else 0,
                float(throughput[i]) if i < len(throughput) else 0,
                int(res_1[i]) if i < len(res_1) else 0,
                int(res_2[i]) if i < len(res_2) else 0,
                int(res_3[i]) if i < len(res_3) else 0,
                int(res_4[i]) if i < len(res_4) else 0,
                int(atom_1[i]) if i < len(atom_1) else 0,
                int(atom_2[i]) if i < len(atom_2) else 0,
                int(atom_3[i]) if i < len(atom_3) else 0,
                int(atom_4[i]) if i < len(atom_4) else 0,
            ))
        conn.executemany(
            "INSERT INTO path_points VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            point_data,
        )

        # Commit periodically
        if idx % 500 == 0:
            conn.commit()

    conn.commit()

    # Import residue positions
    print(f"[4/6] Importing {len(RESIDUE_POSITIONS):,} residue positions...")
    for res_id, pos in tqdm(RESIDUE_POSITIONS.items(), desc="       Residues"):
        count = INITIAL_COUNTS.get(res_id, 0)
        conn.execute(
            "INSERT OR REPLACE INTO residue_positions VALUES (?,?,?,?,?)",
            (int(res_id), float(pos["x"]), float(pos["y"]), float(pos["z"]), count),
        )
    conn.commit()

    # Metadata
    print("[5/6] Storing metadata...")
    db.set_meta("frame_min", str(FRAME_RANGE["min"]))
    db.set_meta("frame_max", str(FRAME_RANGE["max"]))
    db.set_meta("path_count", str(len(PATHS)))
    db.set_meta("residue_count", str(len(RESIDUE_POSITIONS)))
    db.set_meta("source_file", pkl_path)
    db.set_meta("import_time", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Re-enable WAL and create indexes
    print("[6/6] Creating indexes and optimizing...")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("ANALYZE")
    conn.commit()

    db_size = os.path.getsize(db_path) / 1024 / 1024
    elapsed = time.time() - start
    print(f"\nDone! {db_path} ({db_size:.1f} MB) created in {elapsed:.1f}s")
    print(f"  Paths: {len(PATHS):,}")
    print(f"  Residues: {len(RESIDUE_POSITIONS):,}")
    print(f"  Frames: {FRAME_RANGE['min']}-{FRAME_RANGE['max']}")

    db.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python import_data.py <pkl_path> [db_path] [pdb_path]")
        print("Example: python import_data.py ../new-path/4.5.preprocessed_paths.pkl tunnel_data.db ../out/10K/aligned_1.pdb")
        sys.exit(1)

    pkl = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else "tunnel_data.db"
    pdb = sys.argv[3] if len(sys.argv) > 3 else ""

    if not os.path.exists(pkl):
        print(f"Error: {pkl} not found")
        sys.exit(1)

    import_pkl_to_sqlite(pkl, db, pdb_path=pdb)
