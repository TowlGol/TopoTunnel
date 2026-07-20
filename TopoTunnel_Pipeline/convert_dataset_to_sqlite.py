import argparse
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE metadata (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """,
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        data_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE paths (
        id INTEGER PRIMARY KEY,
        frame_id INTEGER NOT NULL,
        cluster_id INTEGER DEFAULT 0,
        point_count INTEGER NOT NULL,
        path_length REAL DEFAULT 0,
        min_radius REAL DEFAULT 0,
        avg_radius REAL DEFAULT 0,
        avg_hydrophobicity REAL DEFAULT 0,
        residues_text TEXT DEFAULT '',
        residue_count INTEGER DEFAULT 0,
        start_x REAL,
        start_y REAL,
        start_z REAL,
        end_x REAL,
        end_y REAL,
        end_z REAL,
        bbox_min_x REAL,
        bbox_min_y REAL,
        bbox_min_z REAL,
        bbox_max_x REAL,
        bbox_max_y REAL,
        bbox_max_z REAL
    )
    """,
    """
    CREATE TABLE path_points (
        path_id INTEGER NOT NULL,
        seq INTEGER NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        z REAL NOT NULL,
        x_origin REAL,
        y_origin REAL,
        z_origin REAL,
        radius REAL DEFAULT 0,
        hydrophobicity REAL DEFAULT 0,
        length_val REAL DEFAULT 0,
        throughput REAL DEFAULT 0,
        res_1 INTEGER DEFAULT 0,
        res_2 INTEGER DEFAULT 0,
        res_3 INTEGER DEFAULT 0,
        res_4 INTEGER DEFAULT 0,
        atom_1 INTEGER DEFAULT 0,
        atom_2 INTEGER DEFAULT 0,
        atom_3 INTEGER DEFAULT 0,
        atom_4 INTEGER DEFAULT 0,
        PRIMARY KEY (path_id, seq)
    )
    """,
    """
    CREATE TABLE path_render_data (
        path_id INTEGER PRIMARY KEY,
        render_coords BLOB NOT NULL,
        selection_coords BLOB
    )
    """,
    """
    CREATE TABLE residue_paths (
        residue_id INTEGER NOT NULL,
        path_id INTEGER NOT NULL,
        PRIMARY KEY (residue_id, path_id)
    )
    """,
    """
    CREATE TABLE residue_positions (
        residue_id INTEGER PRIMARY KEY,
        x REAL NOT NULL,
        y REAL NOT NULL,
        z REAL NOT NULL,
        path_count INTEGER DEFAULT 0
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX idx_paths_cluster ON paths(cluster_id)",
    "CREATE INDEX idx_paths_frame ON paths(frame_id)",
    "CREATE INDEX idx_pp_path ON path_points(path_id)",
    "CREATE INDEX idx_rp_path ON residue_paths(path_id)",
    "CREATE INDEX idx_rp_residue ON residue_paths(residue_id)",
]


def _float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _nullable_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sorted_residues(values):
    residues = set()
    for value in values or []:
        residue_id = _int(value, 0)
        if residue_id > 0:
            residues.add(residue_id)
    return sorted(residues)


def _coords_blob(xs, ys, zs):
    if not xs:
        return None
    coords = np.empty((len(xs), 3), dtype=np.float32)
    coords[:, 0] = np.asarray(xs, dtype=np.float32)
    coords[:, 1] = np.asarray(ys, dtype=np.float32)
    coords[:, 2] = np.asarray(zs, dtype=np.float32)
    return coords.tobytes(order="C")


def _chunked_executemany(cursor, sql, rows, chunk_size):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= chunk_size:
            cursor.executemany(sql, batch)
            batch.clear()
    if batch:
        cursor.executemany(sql, batch)


def convert_pkl_to_sqlite(pkl_path, db_path=None, *, overwrite=True, progress_every=5000):
    pkl_path = Path(pkl_path)
    if db_path is None:
        db_path = pkl_path.with_suffix(".db")
    db_path = Path(db_path)

    if not pkl_path.is_file():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    if db_path.exists():
        if overwrite:
            db_path.unlink()
        else:
            raise FileExistsError(f"DB already exists: {db_path}")

    with pkl_path.open("rb") as handle:
        data = pickle.load(handle)

    paths = data.get("PATHS", [])
    residue_positions = data.get("RESIDUE_POSITIONS", {})
    initial_counts = data.get("INITIAL_COUNTS", {})
    frame_range = data.get("FRAME_RANGE", {}) or {}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-200000")
        conn.execute("PRAGMA page_size=32768")

        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)

        import_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata_rows = [
            ("frame_min", str(_int(frame_range.get("min"), 0))),
            ("frame_max", str(_int(frame_range.get("max"), 0))),
            ("import_time", import_time),
            ("path_count", str(len(paths))),
            ("residue_count", str(len(residue_positions))),
            ("source_file", str(pkl_path)),
        ]
        conn.executemany("INSERT INTO metadata(key, value) VALUES (?, ?)", metadata_rows)

        conn.commit()
        cursor = conn.cursor()

        path_sql = """
            INSERT INTO paths (
                id, frame_id, cluster_id, point_count, path_length, min_radius, avg_radius,
                avg_hydrophobicity, residues_text, residue_count, start_x, start_y, start_z,
                end_x, end_y, end_z, bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        point_sql = """
            INSERT INTO path_points (
                path_id, seq, x, y, z, x_origin, y_origin, z_origin, radius, hydrophobicity,
                length_val, throughput, res_1, res_2, res_3, res_4, atom_1, atom_2, atom_3, atom_4
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        render_sql = "INSERT INTO path_render_data(path_id, render_coords, selection_coords) VALUES (?, ?, ?)"
        residue_path_sql = "INSERT INTO residue_paths(residue_id, path_id) VALUES (?, ?)"
        residue_position_sql = "INSERT INTO residue_positions(residue_id, x, y, z, path_count) VALUES (?, ?, ?, ?, ?)"

        def iter_path_rows():
            for path_id, path in enumerate(paths):
                xs = path.get("xs", []) or []
                ys = path.get("ys", []) or []
                zs = path.get("zs", []) or []
                bf = path.get("bf", []) or []
                qss = path.get("qss", []) or []
                residues = _sorted_residues(path.get("residues", []))
                bbox = path.get("bbox", {}) or {}
                bbox_min = bbox.get("min", [None, None, None])
                bbox_max = bbox.get("max", [None, None, None])
                point_count = len(xs)
                yield (
                    path_id,
                    _int((path.get("frame_ids") or [0])[0], 0),
                    _int(path.get("cluster_id", 0), 0),
                    point_count,
                    _float((path.get("length") or [0])[-1] if point_count else 0.0, 0.0),
                    _float(min(bf) if bf else 0.0, 0.0),
                    _float(sum(bf) / len(bf) if bf else 0.0, 0.0),
                    _float(sum(qss) / len(qss) if qss else 0.0, 0.0),
                    ",".join(str(residue_id) for residue_id in residues),
                    len(residues),
                    _nullable_float(xs[0] if point_count else None),
                    _nullable_float(ys[0] if point_count else None),
                    _nullable_float(zs[0] if point_count else None),
                    _nullable_float(xs[-1] if point_count else None),
                    _nullable_float(ys[-1] if point_count else None),
                    _nullable_float(zs[-1] if point_count else None),
                    _nullable_float(bbox_min[0] if len(bbox_min) > 0 else None),
                    _nullable_float(bbox_min[1] if len(bbox_min) > 1 else None),
                    _nullable_float(bbox_min[2] if len(bbox_min) > 2 else None),
                    _nullable_float(bbox_max[0] if len(bbox_max) > 0 else None),
                    _nullable_float(bbox_max[1] if len(bbox_max) > 1 else None),
                    _nullable_float(bbox_max[2] if len(bbox_max) > 2 else None),
                )

        def iter_point_rows():
            for path_id, path in enumerate(paths):
                xs = path.get("xs", []) or []
                ys = path.get("ys", []) or []
                zs = path.get("zs", []) or []
                xs_origin = path.get("xs_origin", []) or xs
                ys_origin = path.get("ys_origin", []) or ys
                zs_origin = path.get("zs_origin", []) or zs
                bf = path.get("bf", []) or []
                qss = path.get("qss", []) or []
                length_vals = path.get("length", []) or []
                throughput = path.get("throughput", []) or []
                res_1 = path.get("res_1", []) or []
                res_2 = path.get("res_2", []) or []
                res_3 = path.get("res_3", []) or []
                res_4 = path.get("res_4", []) or []
                atom_1 = path.get("atom_1", []) or []
                atom_2 = path.get("atom_2", []) or []
                atom_3 = path.get("atom_3", []) or []
                atom_4 = path.get("atom_4", []) or []

                for seq in range(len(xs)):
                    yield (
                        path_id,
                        seq,
                        _float(xs[seq], 0.0),
                        _float(ys[seq], 0.0),
                        _float(zs[seq], 0.0),
                        _float(xs_origin[seq] if seq < len(xs_origin) else xs[seq], 0.0),
                        _float(ys_origin[seq] if seq < len(ys_origin) else ys[seq], 0.0),
                        _float(zs_origin[seq] if seq < len(zs_origin) else zs[seq], 0.0),
                        _float(bf[seq] if seq < len(bf) else 0.0, 0.0),
                        _float(qss[seq] if seq < len(qss) else 0.0, 0.0),
                        _float(length_vals[seq] if seq < len(length_vals) else seq, 0.0),
                        _float(throughput[seq] if seq < len(throughput) else 0.0, 0.0),
                        _int(res_1[seq] if seq < len(res_1) else 0, 0),
                        _int(res_2[seq] if seq < len(res_2) else 0, 0),
                        _int(res_3[seq] if seq < len(res_3) else 0, 0),
                        _int(res_4[seq] if seq < len(res_4) else 0, 0),
                        _int(atom_1[seq] if seq < len(atom_1) else 0, 0),
                        _int(atom_2[seq] if seq < len(atom_2) else 0, 0),
                        _int(atom_3[seq] if seq < len(atom_3) else 0, 0),
                        _int(atom_4[seq] if seq < len(atom_4) else 0, 0),
                    )

        def iter_render_rows():
            for path_id, path in enumerate(paths):
                xs = path.get("xs", []) or []
                ys = path.get("ys", []) or []
                zs = path.get("zs", []) or []
                xs_sel = path.get("xs_selection", []) or xs
                ys_sel = path.get("ys_selection", []) or ys
                zs_sel = path.get("zs_selection", []) or zs
                yield (
                    path_id,
                    _coords_blob(xs, ys, zs),
                    _coords_blob(xs_sel, ys_sel, zs_sel),
                )

        def iter_residue_path_rows():
            for path_id, path in enumerate(paths):
                for residue_id in _sorted_residues(path.get("residues", [])):
                    yield (residue_id, path_id)

        def iter_residue_position_rows():
            for residue_id in sorted(residue_positions):
                pos = residue_positions[residue_id] or {}
                yield (
                    _int(residue_id, 0),
                    _float(pos.get("x"), 0.0),
                    _float(pos.get("y"), 0.0),
                    _float(pos.get("z"), 0.0),
                    _int(initial_counts.get(residue_id, 0), 0),
                )

        _chunked_executemany(cursor, path_sql, iter_path_rows(), chunk_size=2000)
        _chunked_executemany(cursor, point_sql, iter_point_rows(), chunk_size=20000)
        _chunked_executemany(cursor, render_sql, iter_render_rows(), chunk_size=2000)
        _chunked_executemany(cursor, residue_path_sql, iter_residue_path_rows(), chunk_size=20000)
        _chunked_executemany(cursor, residue_position_sql, iter_residue_position_rows(), chunk_size=2000)

        for statement in INDEX_STATEMENTS:
            conn.execute(statement)
        conn.execute("ANALYZE")
        conn.commit()
    except Exception:
        conn.close()
        if db_path.exists():
            db_path.unlink()
        raise
    else:
        conn.close()

    return db_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert preprocessed_paths.pkl to preprocessed_paths.db")
    parser.add_argument("--input-pkl", required=True, help="input preprocessed_paths.pkl")
    parser.add_argument("--output-db", default="", help="output sqlite db path")
    parser.add_argument("--delete-pkl", action="store_true", help="delete input pkl after successful conversion")
    return parser.parse_args()


def main():
    args = parse_args()
    output_db = args.output_db.strip() or None
    db_path = convert_pkl_to_sqlite(args.input_pkl, output_db)
    if args.delete_pkl:
        os.remove(args.input_pkl)
    print(f"[OK] Wrote {db_path}")


if __name__ == "__main__":
    main()
