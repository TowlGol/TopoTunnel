"""
SQLite schema and database access layer for tunnel data.

Tables:
- paths: Path metadata (one row per path)
- path_render_data: Pre-packed binary coords for rendering (BLOB)
- path_points: Per-point data for profiles/export
- residue_paths: Inverted index (residue_id -> path_id)
- residue_positions: Average 3D positions of residues
- sessions: Session state storage
- metadata: Key-value store for config
"""
import sqlite3
import os
import struct
import numpy as np
from typing import Optional, List, Set, Dict, Any, Tuple


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paths (
    id              INTEGER PRIMARY KEY,
    frame_id        INTEGER NOT NULL,
    cluster_id      INTEGER DEFAULT 0,
    point_count     INTEGER NOT NULL,
    path_length     REAL DEFAULT 0,
    min_radius      REAL DEFAULT 0,
    avg_radius      REAL DEFAULT 0,
    avg_hydrophobicity REAL DEFAULT 0,
    residues_text   TEXT DEFAULT '',
    residue_count   INTEGER DEFAULT 0,
    start_x REAL, start_y REAL, start_z REAL,
    end_x   REAL, end_y   REAL, end_z   REAL,
    bbox_min_x REAL, bbox_min_y REAL, bbox_min_z REAL,
    bbox_max_x REAL, bbox_max_y REAL, bbox_max_z REAL
);

CREATE TABLE IF NOT EXISTS path_render_data (
    path_id         INTEGER PRIMARY KEY REFERENCES paths(id),
    render_coords   BLOB NOT NULL,
    selection_coords BLOB
);

CREATE TABLE IF NOT EXISTS path_points (
    path_id     INTEGER NOT NULL REFERENCES paths(id),
    seq         INTEGER NOT NULL,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    z           REAL NOT NULL,
    x_origin    REAL,
    y_origin    REAL,
    z_origin    REAL,
    radius      REAL DEFAULT 0,
    hydrophobicity REAL DEFAULT 0,
    length_val  REAL DEFAULT 0,
    throughput  REAL DEFAULT 0,
    res_1 INTEGER DEFAULT 0,
    res_2 INTEGER DEFAULT 0,
    res_3 INTEGER DEFAULT 0,
    res_4 INTEGER DEFAULT 0,
    atom_1 INTEGER DEFAULT 0,
    atom_2 INTEGER DEFAULT 0,
    atom_3 INTEGER DEFAULT 0,
    atom_4 INTEGER DEFAULT 0,
    PRIMARY KEY (path_id, seq)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS residue_paths (
    residue_id  INTEGER NOT NULL,
    path_id     INTEGER NOT NULL REFERENCES paths(id),
    PRIMARY KEY (residue_id, path_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS residue_positions (
    residue_id  INTEGER PRIMARY KEY,
    x           REAL NOT NULL,
    y           REAL NOT NULL,
    z           REAL NOT NULL,
    path_count  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    data_json   TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT
);

CREATE INDEX IF NOT EXISTS idx_paths_frame ON paths(frame_id);
CREATE INDEX IF NOT EXISTS idx_paths_cluster ON paths(cluster_id);
CREATE INDEX IF NOT EXISTS idx_rp_residue ON residue_paths(residue_id);
CREATE INDEX IF NOT EXISTS idx_rp_path ON residue_paths(path_id);
CREATE INDEX IF NOT EXISTS idx_pp_path ON path_points(path_id);
"""


class TunnelDatabase:
    """Thread-safe SQLite database access for tunnel data."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._has_throughput_column = True

    def connect(self):
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema_compatibility()

    def init_schema(self):
        self._conn.executescript(SCHEMA_SQL)
        self._ensure_schema_compatibility()
        self._conn.commit()

    def _ensure_schema_compatibility(self):
        """Apply additive schema migrations needed by newer imports."""
        if self._conn is None:
            return
        has_path_points = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='path_points'"
        ).fetchone()
        if not has_path_points:
            return
        cols = self._conn.execute("PRAGMA table_info(path_points)").fetchall()
        names = {row["name"] for row in cols}
        self._has_throughput_column = "throughput" in names
        if not self._has_throughput_column:
            try:
                self._conn.execute(
                    "ALTER TABLE path_points ADD COLUMN throughput REAL DEFAULT 0"
                )
                self._conn.commit()
                self._has_throughput_column = True
            except sqlite3.OperationalError:
                self._has_throughput_column = False
        for column_name in ("atom_1", "atom_2", "atom_3", "atom_4"):
            if column_name in names:
                continue
            try:
                self._conn.execute(
                    f"ALTER TABLE path_points ADD COLUMN {column_name} INTEGER DEFAULT 0"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        return self._conn

    # ─── Metadata ───────────────────────────────────────
    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
            (key, value),
        )
        self.conn.commit()

    # ─── Path queries ───────────────────────────────────
    def path_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM paths").fetchone()[0]

    def frame_range(self) -> Tuple[int, int]:
        fmin = int(self.get_meta("frame_min", "0"))
        fmax = int(self.get_meta("frame_max", "0"))
        return fmin, fmax

    def get_all_frame_ids(self) -> List[int]:
        rows = self.conn.execute(
            "SELECT DISTINCT frame_id FROM paths ORDER BY frame_id"
        ).fetchall()
        return [int(row["frame_id"]) for row in rows]

    def get_frame_ids_for_path_ids(self, path_ids: List[int]) -> List[int]:
        if not path_ids:
            return []
        placeholders = ",".join("?" * len(path_ids))
        rows = self.conn.execute(
            f"SELECT DISTINCT frame_id FROM paths WHERE id IN ({placeholders}) ORDER BY frame_id",
            [int(pid) for pid in path_ids],
        ).fetchall()
        return [int(row["frame_id"]) for row in rows]

    def get_path_ids(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[int]:
        conditions = ["1=1"]
        params: list = []
        if frame_min is not None:
            conditions.append("frame_id >= ?")
            params.append(int(frame_min))
        if frame_max is not None:
            conditions.append("frame_id <= ?")
            params.append(int(frame_max))
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT id FROM paths WHERE {where} ORDER BY id",
            params,
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def get_path_ids_for_frames(self, frame_ids: List[int]) -> List[int]:
        if not frame_ids:
            return []
        placeholders = ",".join("?" * len(frame_ids))
        rows = self.conn.execute(
            f"SELECT id FROM paths WHERE frame_id IN ({placeholders}) ORDER BY id",
            [int(frame_id) for frame_id in frame_ids],
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def _populate_sel_ids(self, path_ids: Set[int]):
        """Populate temp table with selected path IDs for efficient filtering."""
        c = self.conn
        c.execute("CREATE TEMP TABLE IF NOT EXISTS _sel_ids(id INTEGER PRIMARY KEY)")
        c.execute("DELETE FROM _sel_ids")
        c.executemany("INSERT INTO _sel_ids(id) VALUES(?)", [(pid,) for pid in path_ids])

    def get_paths_page(
        self,
        offset: int = 0,
        limit: int = 200,
        order_by: str = "id",
        order_dir: str = "ASC",
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> List[dict]:
        """Paginated path query with optional filters."""
        conditions = ["1=1"]
        params: list = []

        if path_id_filter is not None:
            self._populate_sel_ids(path_id_filter)
            conditions.append("id IN (SELECT id FROM _sel_ids)")

        if frame_min is not None:
            conditions.append("frame_id >= ?")
            params.append(frame_min)
        if frame_max is not None:
            conditions.append("frame_id <= ?")
            params.append(frame_max)

        if residue_filter:
            # Intersection: path must contain ALL selected residues
            for rid in residue_filter:
                conditions.append(
                    f"id IN (SELECT path_id FROM residue_paths WHERE residue_id=?)"
                )
                params.append(rid)

        # Validate order_by to prevent SQL injection
        allowed_cols = {
            "id", "frame_id", "path_length", "min_radius",
            "avg_radius", "residue_count", "cluster_id",
            "avg_hydrophobicity",
        }
        if order_by not in allowed_cols:
            order_by = "id"
        if order_dir.upper() not in ("ASC", "DESC"):
            order_dir = "ASC"

        where = " AND ".join(conditions)
        sql = f"""
            SELECT * FROM paths
            WHERE {where}
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        rows = self.conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count_filtered_paths(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
        residue_filter: Optional[Set[int]] = None,
        path_id_filter: Optional[Set[int]] = None,
    ) -> int:
        conditions = ["1=1"]
        params: list = []
        if path_id_filter is not None:
            self._populate_sel_ids(path_id_filter)
            conditions.append("id IN (SELECT id FROM _sel_ids)")
        if frame_min is not None:
            conditions.append("frame_id >= ?")
            params.append(frame_min)
        if frame_max is not None:
            conditions.append("frame_id <= ?")
            params.append(frame_max)
        if residue_filter:
            for rid in residue_filter:
                conditions.append(
                    "id IN (SELECT path_id FROM residue_paths WHERE residue_id=?)"
                )
                params.append(rid)
        where = " AND ".join(conditions)
        return self.conn.execute(
            f"SELECT COUNT(*) FROM paths WHERE {where}", params
        ).fetchone()[0]

    def get_matched_path_ids(self, residue_ids: Set[int]) -> List[int]:
        """Get path IDs that contain ALL given residues (intersection)."""
        if not residue_ids:
            return []
        conditions = []
        params = []
        for rid in residue_ids:
            conditions.append(
                "id IN (SELECT path_id FROM residue_paths WHERE residue_id=?)"
            )
            params.append(rid)
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT id FROM paths WHERE {where} ORDER BY id", params
        ).fetchall()
        return [r[0] for r in rows]

    # ─── Render data ────────────────────────────────────
    def get_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get packed render coordinates as numpy arrays."""
        if not path_ids:
            return {}
        result = {}
        chunk_size = 900
        for start in range(0, len(path_ids), chunk_size):
            chunk = path_ids[start:start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT path_id, render_coords FROM path_render_data "
                f"WHERE path_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                blob = row["render_coords"]
                arr = np.frombuffer(blob, dtype=np.float32).reshape(-1, 3)
                result[row["path_id"]] = arr
        return result

    def get_path_length_map(self, path_ids: List[int]) -> Dict[int, float]:
        """Get path_length values keyed by path id."""
        if not path_ids:
            return {}
        result: Dict[int, float] = {}
        chunk_size = 900
        for start in range(0, len(path_ids), chunk_size):
            chunk = path_ids[start:start + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT id, path_length FROM paths WHERE id IN ({placeholders})",
                chunk,
            ).fetchall()
            for row in rows:
                result[int(row["id"])] = float(row["path_length"] or 0.0)
        return result

    @staticmethod
    def _group_original_coord_rows(rows) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        current_path_id: Optional[int] = None
        current_coords: List[Tuple[float, float, float]] = []

        for row in rows:
            path_id = int(row["path_id"])
            if current_path_id is None:
                current_path_id = path_id
            if path_id != current_path_id:
                result[current_path_id] = np.asarray(current_coords, dtype=np.float32)
                current_path_id = path_id
                current_coords = []
            current_coords.append((row["x"], row["y"], row["z"]))

        if current_path_id is not None:
            result[current_path_id] = np.asarray(current_coords, dtype=np.float32)
        return result

    @staticmethod
    def _group_original_coord_rows_with_frame(rows) -> List[Tuple[int, int, np.ndarray]]:
        result: List[Tuple[int, int, np.ndarray]] = []
        current_path_id: Optional[int] = None
        current_frame_id: Optional[int] = None
        current_coords: List[Tuple[float, float, float]] = []

        for row in rows:
            path_id = int(row["path_id"])
            frame_id = int(row["frame_id"])
            if current_path_id is None:
                current_path_id = path_id
                current_frame_id = frame_id
            if path_id != current_path_id:
                result.append(
                    (
                        current_path_id,
                        int(current_frame_id or 0),
                        np.asarray(current_coords, dtype=np.float32),
                    )
                )
                current_path_id = path_id
                current_frame_id = frame_id
                current_coords = []
            current_coords.append((row["x"], row["y"], row["z"]))

        if current_path_id is not None:
            result.append(
                (
                    current_path_id,
                    int(current_frame_id or 0),
                    np.asarray(current_coords, dtype=np.float32),
                )
            )
        return result

    def get_original_render_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get original per-point coordinates used by PDB export."""
        if not path_ids:
            return {}
        result: Dict[int, np.ndarray] = {}
        chunk_size = 900
        for start in range(0, len(path_ids), chunk_size):
            chunk = [int(path_id) for path_id in path_ids[start:start + chunk_size]]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"SELECT path_id, "
                f"COALESCE(x_origin, x) AS x, "
                f"COALESCE(y_origin, y) AS y, "
                f"COALESCE(z_origin, z) AS z "
                f"FROM path_points "
                f"WHERE path_id IN ({placeholders}) "
                f"ORDER BY path_id, seq",
                chunk,
            ).fetchall()
            result.update(self._group_original_coord_rows(rows))
        return result

    def get_path_exit_points(self, path_ids: List[int]) -> Dict[int, dict]:
        """Get current/original exit endpoints and cluster IDs for paths."""
        if not path_ids:
            return {}
        result: Dict[int, dict] = {}
        chunk_size = 900
        for start in range(0, len(path_ids), chunk_size):
            chunk = [int(path_id) for path_id in path_ids[start:start + chunk_size]]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            rows = self.conn.execute(
                f"""
                SELECT
                    p.id,
                    p.cluster_id,
                    COALESCE(p.end_x, pp.x) AS current_end_x,
                    COALESCE(p.end_y, pp.y) AS current_end_y,
                    COALESCE(p.end_z, pp.z) AS current_end_z,
                    COALESCE(pp.x_origin, pp.x, p.end_x) AS original_end_x,
                    COALESCE(pp.y_origin, pp.y, p.end_y) AS original_end_y,
                    COALESCE(pp.z_origin, pp.z, p.end_z) AS original_end_z
                FROM paths p
                LEFT JOIN path_points pp
                    ON pp.path_id = p.id
                    AND pp.seq = (
                        SELECT MAX(seq)
                        FROM path_points
                        WHERE path_id = p.id
                    )
                WHERE p.id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            for row in rows:
                path_id = int(row["id"])
                current_exit = np.asarray(
                    [
                        float(row["current_end_x"] or 0.0),
                        float(row["current_end_y"] or 0.0),
                        float(row["current_end_z"] or 0.0),
                    ],
                    dtype=np.float64,
                )
                original_exit = np.asarray(
                    [
                        float(row["original_end_x"] or 0.0),
                        float(row["original_end_y"] or 0.0),
                        float(row["original_end_z"] or 0.0),
                    ],
                    dtype=np.float64,
                )
                result[path_id] = {
                    "cluster_id": int(row["cluster_id"] or 0),
                    "current_exit": current_exit,
                    "original_exit": original_exit,
                }
        return result

    def get_all_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        """Get all render coords with frame filtering. Returns (path_id, frame_id, coords)."""
        conditions = ["1=1"]
        params = []
        if frame_min is not None:
            conditions.append("p.frame_id >= ?")
            params.append(frame_min)
        if frame_max is not None:
            conditions.append("p.frame_id <= ?")
            params.append(frame_max)
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT p.id, p.frame_id, r.render_coords "
            f"FROM paths p JOIN path_render_data r ON p.id=r.path_id "
            f"WHERE {where}", params
        ).fetchall()
        result = []
        for row in rows:
            arr = np.frombuffer(row["render_coords"], dtype=np.float32).reshape(-1, 3)
            result.append((row["id"], row["frame_id"], arr))
        return result

    def get_all_original_render_coords(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> List[Tuple[int, int, np.ndarray]]:
        """Get original per-point coordinates with frame filtering."""
        conditions = ["1=1"]
        params = []
        if frame_min is not None:
            conditions.append("p.frame_id >= ?")
            params.append(frame_min)
        if frame_max is not None:
            conditions.append("p.frame_id <= ?")
            params.append(frame_max)
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT p.id AS path_id, p.frame_id AS frame_id, "
            f"COALESCE(pp.x_origin, pp.x) AS x, "
            f"COALESCE(pp.y_origin, pp.y) AS y, "
            f"COALESCE(pp.z_origin, pp.z) AS z "
            f"FROM paths p "
            f"JOIN path_points pp ON p.id = pp.path_id "
            f"WHERE {where} "
            f"ORDER BY p.id, pp.seq",
            params,
        ).fetchall()
        return self._group_original_coord_rows_with_frame(rows)

    def get_selection_coords(self, path_ids: List[int]) -> Dict[int, np.ndarray]:
        """Get selection coordinates (full resolution) for lasso selection."""
        if not path_ids:
            return {}
        placeholders = ",".join("?" * len(path_ids))
        rows = self.conn.execute(
            f"SELECT path_id, COALESCE(selection_coords, render_coords) as coords "
            f"FROM path_render_data WHERE path_id IN ({placeholders})",
            path_ids,
        ).fetchall()
        result = {}
        for row in rows:
            arr = np.frombuffer(row["coords"], dtype=np.float32).reshape(-1, 3)
            result[row["path_id"]] = arr
        return result

    def get_path_point_rows(
        self,
        path_ids: List[int],
        frame_id: Optional[int] = None,
        *,
        original_coords: bool = True,
    ) -> List[dict]:
        if not path_ids:
            return []
        placeholders = ",".join("?" * len(path_ids))
        coord_x = "COALESCE(pp.x_origin, pp.x)" if original_coords else "pp.x"
        coord_y = "COALESCE(pp.y_origin, pp.y)" if original_coords else "pp.y"
        coord_z = "COALESCE(pp.z_origin, pp.z)" if original_coords else "pp.z"
        conditions = [f"pp.path_id IN ({placeholders})"]
        params: list = [int(path_id) for path_id in path_ids]
        if frame_id is not None:
            conditions.append("p.frame_id = ?")
            params.append(int(frame_id))
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"""
            SELECT pp.path_id,
                   pp.seq,
                   {coord_x} AS x,
                   {coord_y} AS y,
                   {coord_z} AS z,
                   pp.radius
            FROM path_points pp
            JOIN paths p ON p.id = pp.path_id
            WHERE {where}
            ORDER BY pp.path_id, pp.seq
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]

    # ─── Path profiles ──────────────────────────────────
    def get_path_profiles(self, path_ids: List[int]) -> List[dict]:
        """Get per-point data for profile charts."""
        if not path_ids:
            return []
        throughput_expr = "throughput" if self._has_throughput_column else "0 AS throughput"
        profiles = []
        for pid in path_ids:
            rows = self.conn.execute(
                f"SELECT seq, radius, hydrophobicity, length_val, {throughput_expr}, "
                "res_1, res_2, res_3, res_4, atom_1, atom_2, atom_3, atom_4 "
                "FROM path_points WHERE path_id=? ORDER BY seq",
                (pid,),
            ).fetchall()
            if not rows:
                continue
            meta = self.conn.execute(
                "SELECT frame_id, cluster_id FROM paths WHERE id=?", (pid,)
            ).fetchone()
            profiles.append({
                "pathIndex": pid,
                "frameId": meta["frame_id"] if meta else 0,
                "cluster_id": meta["cluster_id"] if meta else 0,
                "resSeq": [r["length_val"] for r in rows],
                "radius": [r["radius"] for r in rows],
                "hydrophobicity": [r["hydrophobicity"] for r in rows],
                "throughput": [r["throughput"] for r in rows],
                "res_1": [r["res_1"] for r in rows],
                "res_2": [r["res_2"] for r in rows],
                "res_3": [r["res_3"] for r in rows],
                "res_4": [r["res_4"] for r in rows],
                "atom_1": [r["atom_1"] for r in rows],
                "atom_2": [r["atom_2"] for r in rows],
                "atom_3": [r["atom_3"] for r in rows],
                "atom_4": [r["atom_4"] for r in rows],
                "numPoints": len(rows),
            })
        return profiles

    # ─── Residues ───────────────────────────────────────
    def get_residue_options(
        self, selected_residues: Optional[Set[int]] = None
    ) -> List[dict]:
        """Get available residues with counts."""
        if not selected_residues:
            rows = self.conn.execute(
                "SELECT residue_id, path_count FROM residue_positions ORDER BY residue_id"
            ).fetchall()
            return [
                {"label": f"{r['residue_id']} (count={r['path_count']})",
                 "value": str(r["residue_id"])}
                for r in rows
            ]
        # Counts based on matched paths
        matched = self.get_matched_path_ids(selected_residues)
        if not matched:
            return []
        placeholders = ",".join("?" * len(matched))
        rows = self.conn.execute(
            f"SELECT residue_id, COUNT(*) as cnt FROM residue_paths "
            f"WHERE path_id IN ({placeholders}) "
            f"GROUP BY residue_id ORDER BY residue_id",
            matched,
        ).fetchall()
        return [
            {"label": f"{r['residue_id']} (count={r['cnt']})",
             "value": str(r["residue_id"])}
            for r in rows
        ]

    def get_residue_positions(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT residue_id, x, y, z, path_count "
            "FROM residue_positions ORDER BY residue_id"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_cluster_path_groups(
        self,
        frame_min: Optional[int] = None,
        frame_max: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        conditions = ["1=1"]
        params: list = []
        if frame_min is not None:
            conditions.append("frame_id >= ?")
            params.append(frame_min)
        if frame_max is not None:
            conditions.append("frame_id <= ?")
            params.append(frame_max)
        where = " AND ".join(conditions)
        rows = self.conn.execute(
            f"SELECT id, cluster_id FROM paths WHERE {where} ORDER BY cluster_id, id",
            params,
        ).fetchall()
        groups: Dict[int, List[int]] = {}
        for row in rows:
            cluster_id = int(row["cluster_id"] or 0)
            groups.setdefault(cluster_id, []).append(int(row["id"]))
        return groups

    # ─── Export ─────────────────────────────────────────
    def get_path_for_export(self, path_id: int) -> Optional[dict]:
        """Get full path data for PDB export."""
        meta = self.conn.execute(
            "SELECT * FROM paths WHERE id=?", (path_id,)
        ).fetchone()
        if not meta:
            return None
        points = self.conn.execute(
            "SELECT * FROM path_points WHERE path_id=? ORDER BY seq",
            (path_id,),
        ).fetchall()
        return {"meta": dict(meta), "points": [dict(p) for p in points]}

    # ─── Sessions ───────────────────────────────────────
    def list_sessions(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def create_session(self, session_id: str, name: str, data_json: str = "{}"):
        import datetime
        now = datetime.datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO sessions(id, name, created_at, updated_at, data_json) "
            "VALUES(?,?,?,?,?)",
            (session_id, name, now, now, data_json),
        )
        self.conn.commit()

    def load_session(self, session_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
        return dict(row) if row else None

    def save_session(self, session_id: str, name: str, data_json: str):
        import datetime
        now = datetime.datetime.now().isoformat()
        self.conn.execute(
            "UPDATE sessions SET name=?, updated_at=?, data_json=? WHERE id=?",
            (name, now, data_json, session_id),
        )
        self.conn.commit()

    def delete_session(self, session_id: str):
        self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.conn.commit()
