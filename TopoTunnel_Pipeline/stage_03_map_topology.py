# -*-coding:utf-8-*-

import argparse
import csv
import glob
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from collections import defaultdict
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats as scipy_stats

from pipeline_config import (
    PDB_path,
    all_points_csv,
    dataDir,
    excluded_residues,
    mutations,
    four_atoms_csv,
    residue_stats_csv,
    residue_combination_stats_csv,
    residue_distance_relation_csv,
    residue_pair_distance_csv,
    residue_pair_path_presence_stats_csv,
    tunnel_profile_csv,
)
from atomic_properties import atomradius
from mutation_support import (
    build_mutation_lookup,
    get_mutated_resname,
    validate_mutation_observations,
)
from residue_properties import residuehydrophobicity


out_path = four_atoms_csv
residue_stats_path = residue_stats_csv
residue_combination_stats_path = residue_combination_stats_csv
residue_distance_relation_path = residue_distance_relation_csv
residue_pair_distance_path = residue_pair_distance_csv
residue_pair_path_presence_stats_path = residue_pair_path_presence_stats_csv
representative_pdb_path = PDB_path
all_points_output_path = all_points_csv
fixed_dir_path = "data/fixed/"

IGNORE_H = False
TOP_ATOMS = 4
SEARCH_K = int(os.environ.get("PATHVIS_SEARCH_K", "48"))
PRECISION = 10
PROGRESS_INTERVAL = int(os.environ.get("PATHVIS_PROGRESS_INTERVAL", "100"))
CPU_COUNT = os.cpu_count() or 1
MAX_WORKERS = int(os.environ.get("PATHVIS_MAX_WORKERS", str(max(1, min(16, CPU_COUNT - 1)))))
KD_QUERY_WORKERS = int(os.environ.get("PATHVIS_KD_QUERY_WORKERS", "1" if MAX_WORKERS > 1 else "-1"))
PDB_BATCH_MAX_GROUPS = int(os.environ.get("PATHVIS_PDB_BATCH_MAX_GROUPS", "0"))
PDB_CONTEXT_CACHE_SIZE = int(os.environ.get("PATHVIS_PDB_CONTEXT_CACHE_SIZE", "64"))
WRITE_RESIDUE_DISTANCE_RELATIONS = os.environ.get("PATHVIS_WRITE_RESIDUE_DISTANCE_RELATIONS", "1").strip().lower() not in {"0", "false", "no", "off"}
WRITE_RESIDUE_DISTANCE_RELATION_CSV = os.environ.get("PATHVIS_WRITE_RESIDUE_DISTANCE_RELATION_CSV", "1").strip().lower() not in {"0", "false", "no", "off"}
WRITE_ATOM_NETWORK_CSV = os.environ.get("PATHVIS_WRITE_ATOM_NETWORK_CSV", "1").strip().lower() not in {"0", "false", "no", "off"}
RESIDUE_DISTANCE_OUTPUT_MODE = os.environ.get("PATHVIS_RESIDUE_DISTANCE_OUTPUT_MODE", "compact").strip().lower()
RESIDUE_DISTANCE_PRECISION = int(os.environ.get("PATHVIS_RESIDUE_DISTANCE_PRECISION", "6"))
RESIDUE_DISTANCE_MAX_FRAMES = int(os.environ.get("PATHVIS_RESIDUE_DISTANCE_MAX_FRAMES", "0"))
RESIDUE_DISTANCE_FRAME_WORKERS = int(os.environ.get("PATHVIS_RESIDUE_DISTANCE_FRAME_WORKERS", str(max(1, min(8, CPU_COUNT - 1)))))
RESIDUE_DISTANCE_FRAME_CHUNKSIZE = int(os.environ.get("PATHVIS_RESIDUE_DISTANCE_FRAME_CHUNKSIZE", "8"))
CUMULATIVE_DISTANCE_CSV_MODE = os.environ.get("PATHVIS_CUMULATIVE_DISTANCE_CSV_MODE", "auto").strip().lower()
CUMULATIVE_DISTANCE_CSV_MAX_CELLS = int(os.environ.get("PATHVIS_CUMULATIVE_DISTANCE_CSV_MAX_CELLS", "5000000"))
CUMULATIVE_DISTANCE_COMBO_CHUNK_SIZE = int(os.environ.get("PATHVIS_CUMULATIVE_DISTANCE_COMBO_CHUNK_SIZE", "512"))
EXCLUDED_RESIDUES = frozenset(excluded_residues or [])
MUTATION_LOOKUP = build_mutation_lookup(mutations)
HYDROGEN_ELEMENTS = frozenset({"H", "D", "T"})
RADIUS_CACHE = {}
HYDROPHOBICITY_CACHE = {}
_DISTANCE_WORKER_RESIDUE_INDEX_BY_ID = None
_DISTANCE_WORKER_PAIR_RESIDUE_INDICES = None
_DISTANCE_WORKER_IGNORE_H = False
_REPRESENTATIVE_ATOM_RECORDS = None


def configure_runtime_mutations(raw_mutations):
    global MUTATION_LOOKUP
    global _REPRESENTATIVE_ATOM_RECORDS
    MUTATION_LOOKUP = build_mutation_lookup(raw_mutations)
    _REPRESENTATIVE_ATOM_RECORDS = None
    load_structure_context.cache_clear()


def init_presence_accumulator():
    return {}


def update_path_pair_presence(presence_accumulator, pdb_id, path_residues):
    frame_id = normalize_frame_id(pdb_id)
    if not frame_id:
        return
    residue_ids = sorted(int(residue_id) for residue_id in path_residues if int(residue_id) > 0)
    if len(residue_ids) < 2:
        return
    frame_pairs = presence_accumulator.setdefault(frame_id, set())
    frame_pairs.update(combinations(residue_ids, 2))


def configure_runtime_paths(
        tunnel_profile_path=None,
        pdb_root=None,
        atom_network_output_path=None,
        residue_stats_output_path=None,
        combination_stats_output_path=None,
        residue_distance_output_path=None,
        residue_pair_output_path=None,
        path_presence_stats_output_path=None,
        representative_pdb_input_path=None,
        all_points_csv_output_path=None,
        mutations_input=None):
    global dataDir
    global tunnel_profile_csv
    global out_path
    global residue_stats_path
    global residue_combination_stats_path
    global residue_distance_relation_path
    global residue_pair_distance_path
    global residue_pair_path_presence_stats_path
    global representative_pdb_path
    global all_points_output_path
    global _REPRESENTATIVE_ATOM_RECORDS

    if tunnel_profile_path is not None:
        tunnel_profile_csv = tunnel_profile_path
    if pdb_root is not None:
        dataDir = pdb_root
    if atom_network_output_path is not None:
        out_path = atom_network_output_path
    if residue_stats_output_path is not None:
        residue_stats_path = residue_stats_output_path
    if combination_stats_output_path is not None:
        residue_combination_stats_path = combination_stats_output_path
    if residue_distance_output_path is not None:
        residue_distance_relation_path = residue_distance_output_path
    if residue_pair_output_path is not None:
        residue_pair_distance_path = residue_pair_output_path
    if path_presence_stats_output_path is not None:
        residue_pair_path_presence_stats_path = path_presence_stats_output_path
    if representative_pdb_input_path is not None:
        representative_pdb_path = representative_pdb_input_path
    if all_points_csv_output_path is not None:
        all_points_output_path = all_points_csv_output_path
    if mutations_input is not None:
        configure_runtime_mutations(mutations_input)

    _REPRESENTATIVE_ATOM_RECORDS = None
    load_structure_context.cache_clear()

OUTPUT_HEADER = [
    "T_F_A_1_ID", "T_F_A_1_element", "T_F_A_1_radius", "T_F_A_1_serial",
    "T_F_A_2_ID", "T_F_A_2_element", "T_F_A_2_radius", "T_F_A_2_serial",
    "T_F_A_3_ID", "T_F_A_3_element", "T_F_A_3_radius", "T_F_A_3_serial",
    "T_F_A_4_ID", "T_F_A_4_element", "T_F_A_4_radius", "T_F_A_4_serial",
    "T_F_M_1_ID", "T_F_M_1_resname", "T_F_M_1_hydrophobicity",
    "T_F_M_2_ID", "T_F_M_2_resname", "T_F_M_2_hydrophobicity",
    "T_F_M_3_ID", "T_F_M_3_resname", "T_F_M_3_hydrophobicity",
    "T_F_M_4_ID", "T_F_M_4_resname", "T_F_M_4_hydrophobicity",
    "T_F_T_ID", "T_F_T_x", "T_F_T_y", "T_F_T_z", "T_F_T_radius", "T_F_T_length", "T_F_T_freedom",
    "T_F_Topo_ID",
    "T_F_G_x", "T_F_G_y",
    "Tunnel_ID", "Tunnel_length", "pdb_id", "Throughput", "Tunnel_cluster",
]

ALL_POINTS_OUTPUT_COLUMNS = [
    "source_file", "atom_serial", "res_seq", "x", "y", "z", "bfactor", "occupancy", "element",
    "residue_1", "residue_2", "residue_3", "residue_4",
    "x_origin", "y_origin", "z_origin", "frame_id", "length", "throughput", "tunnel_cluster",
    "qss1", "qss2", "qss3", "qss4",
    "atom_1", "atom_2", "atom_3", "atom_4",
]

RESIDUE_STATS_HEADER = [
    "Residue_ID",
    "Residue_name",
    "Affected_point_count",
    "Affected_path_count",
    "Affected_point_radius_avg",
    "Affected_point_radius_min",
    "Affected_point_radius_max",
    "Cooperative_residue_count",
    "Cooperative_residue_ids",
    "Cooperative_residue_details",
    "Cooperative_residue_type_counts",
    "Cooperative_residue_type_path_counts",
]

RESIDUE_COMBINATION_STATS_HEADER = [
    "Combination_size",
    "Residue_IDs",
    "Residue_names",
    "Affected_point_count",
    "Affected_path_count",
    "Affected_point_radius_avg",
    "Affected_point_radius_min",
    "Affected_point_radius_max",
    "Affected_point_radius_range",
]


def safe_float(value, default=0.0):
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return default


def get_element_from_name(atom_name):
    atom_name = atom_name.strip().upper()
    if atom_name.startswith("FE"):
        return "FE"
    if atom_name and atom_name[0].isalpha():
        return atom_name[0]
    return "X"


def get_cached_radius(element):
    cached = RADIUS_CACHE.get(element)
    if cached is None:
        cached = safe_float(atomradius(element), 1.5)
        RADIUS_CACHE[element] = cached
    return cached


def get_cached_hydrophobicity(resname):
    cached = HYDROPHOBICITY_CACHE.get(resname)
    if cached is None:
        cached = safe_float(residuehydrophobicity(resname))
        HYDROPHOBICITY_CACHE[resname] = cached
    return cached


def build_atom_locator(residue_id, resname, atom_name):
    return f"{int(residue_id)}_{str(resname).strip()}_{str(atom_name).strip()}"


@lru_cache(maxsize=None)
def extract_pdb_digits(pdb_name):
    return "".join(re.findall(r"\d", pdb_name)) or "0"


def extract_atoms_from_pdb(pdb_path, ignore_h=False):
    coords = []
    serials = []
    atom_locators = []
    elements = []
    res_ids = []
    resnames = []
    atom_names = []
    hydrophobicities = []
    vdw_radii = []

    excluded = EXCLUDED_RESIDUES
    mutation_lookup = MUTATION_LOOKUP
    hydrogen_elements = HYDROGEN_ELEMENTS
    observed_resnames = {}
    append_coord = coords.append
    append_serial = serials.append
    append_atom_locator = atom_locators.append
    append_element = elements.append
    append_res_id = res_ids.append
    append_resname = resnames.append
    append_atom_name = atom_names.append
    append_hydrophobicity = hydrophobicities.append
    append_radius = vdw_radii.append

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                residue_id = int(line[22:26])
                serial = int(line[6:11])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            if residue_id in excluded:
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                element = get_element_from_name(atom_name)
            if ignore_h and element in hydrogen_elements:
                continue

            resname = line[17:20].strip()
            observed_resnames.setdefault(residue_id, set()).add(resname)
            if mutation_lookup:
                resname = get_mutated_resname(residue_id, resname, mutation_lookup)

            append_coord((x, y, z))
            append_serial(serial)
            append_atom_locator(build_atom_locator(residue_id, resname, atom_name))
            append_element(element)
            append_res_id(residue_id)
            append_resname(resname)
            append_atom_name(atom_name)
            append_hydrophobicity(get_cached_hydrophobicity(resname))
            append_radius(get_cached_radius(element))

    if not coords:
        raise ValueError(f"No atoms available after filtering: {pdb_path}")
    if mutation_lookup:
        validate_mutation_observations(observed_resnames, pdb_path, mutation_lookup)

    coords_array = np.asarray(coords, dtype=np.float64)
    atom_locators_array = np.asarray(atom_locators, dtype=object)
    representative_coords = np.full((len(atom_locators_array), 3), np.nan, dtype=np.float64)
    representative_valid = np.zeros((len(atom_locators_array),), dtype=bool)
    if str(all_points_output_path).strip():
        representative_atom_records = get_representative_atom_records()
        representative_records = [representative_atom_records.get(locator) for locator in atom_locators_array]
        representative_valid = np.fromiter(
            (record is not None for record in representative_records),
            dtype=bool,
            count=len(representative_records),
        )
        if representative_valid.any():
            representative_coords[representative_valid] = np.asarray(
                [record[1:] for record in representative_records if record is not None],
                dtype=np.float64,
            )
    return {
        "coords": coords_array,
        "serials": np.asarray(serials, dtype=np.int32),
        "atom_locators": atom_locators_array,
        "elements": np.asarray(elements, dtype=object),
        "res_ids": np.asarray(res_ids, dtype=np.int32),
        "resnames": np.asarray(resnames, dtype=object),
        "atom_names": np.asarray(atom_names, dtype=object),
        "hydrophobicities": np.asarray(hydrophobicities, dtype=np.float64),
        "vdw_radii": np.asarray(vdw_radii, dtype=np.float64),
        "representative_coords": representative_coords,
        "representative_valid": representative_valid,
        "kdtree": cKDTree(coords_array),
    }


@lru_cache(maxsize=PDB_CONTEXT_CACHE_SIZE if PDB_CONTEXT_CACHE_SIZE > 0 else None)
def load_structure_context(pdb_root, pdb_name, ignore_h):
    pdb_path = os.path.join(pdb_root, pdb_name)
    return extract_atoms_from_pdb(pdb_path, ignore_h=ignore_h)


def iter_tunnel_groups(profile_path):
    with open(profile_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)

        rows_group = []
        group_index = 0
        for row in reader:
            rows_group.append(row)
            if len(rows_group) == 7:
                yield rows_group, group_index
                rows_group = []
                group_index += 1

        if rows_group:
            print(f"[Warn] Ignored incomplete tunnel group with {len(rows_group)} rows at the end of file.")


def parse_group_points(rows_group):
    x_row, y_row, z_row, _, length_row, radius_row, _ = rows_group
    max_cols = min(len(x_row), len(y_row), len(z_row), len(length_row), len(radius_row))
    end_idx = 13
    while end_idx < max_cols and str(x_row[end_idx]).strip():
        end_idx += 1

    if end_idx == 13:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )

    point_count = end_idx - 13
    xs = np.fromiter((safe_float(x_row[i]) for i in range(13, end_idx)), dtype=np.float64, count=point_count)
    ys = np.fromiter((safe_float(y_row[i]) for i in range(13, end_idx)), dtype=np.float64, count=point_count)
    zs = np.fromiter((safe_float(z_row[i]) for i in range(13, end_idx)), dtype=np.float64, count=point_count)
    point_lengths = np.fromiter((safe_float(length_row[i]) for i in range(13, end_idx)), dtype=np.float64, count=point_count)
    point_radii = np.fromiter((safe_float(radius_row[i]) for i in range(13, end_idx)), dtype=np.float64, count=point_count)

    return (
        np.column_stack((xs, ys, zs)),
        point_lengths,
        point_radii,
    )


def query_nearest_atoms(points_xyz, context):
    search_k = min(max(TOP_ATOMS, SEARCH_K), len(context["serials"]))
    try:
        distances, indices = context["kdtree"].query(points_xyz, k=search_k, workers=KD_QUERY_WORKERS)
    except TypeError:
        distances, indices = context["kdtree"].query(points_xyz, k=search_k)

    if search_k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)


def load_representative_atom_records(pdb_path):
    atom_records = {}
    observed_resnames = {}
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Representative PDB not found: {pdb_path}")

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                serial = int(line[6:11])
                residue_id = int(line[22:26])
                resname = line[17:20].strip()
                observed_resnames.setdefault(residue_id, set()).add(resname)
                if MUTATION_LOOKUP:
                    resname = get_mutated_resname(residue_id, resname, MUTATION_LOOKUP)
                atom_name = line[12:16].strip()
                atom_records[build_atom_locator(residue_id, resname, atom_name)] = (
                    serial,
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (ValueError, IndexError):
                continue

    if MUTATION_LOOKUP:
        validate_mutation_observations(observed_resnames, pdb_path, MUTATION_LOOKUP)
    return atom_records


def get_representative_atom_records():
    global _REPRESENTATIVE_ATOM_RECORDS
    if _REPRESENTATIVE_ATOM_RECORDS is None:
        _REPRESENTATIVE_ATOM_RECORDS = load_representative_atom_records(representative_pdb_path)
    return _REPRESENTATIVE_ATOM_RECORDS


def compute_point_centroid(selected_indices, context):
    selected_indices = np.asarray(selected_indices[:TOP_ATOMS], dtype=np.int64)
    representative_valid = context["representative_valid"][selected_indices]
    if representative_valid.size == TOP_ATOMS and representative_valid.all():
        return np.mean(context["representative_coords"][selected_indices], axis=0)
    return np.mean(context["coords"][selected_indices], axis=0)


def build_all_points_row(
        source_file,
        atom_serial,
        res_seq,
        centroid_xyz,
        point_radius,
        occupancy_value,
        residue_ids,
        point_xyz,
        pdb_id,
        point_length,
        throughput,
        tunnel_cluster,
        qss_values,
        atom_serials):
    return [
        source_file,
        int(atom_serial),
        int(res_seq),
        float(centroid_xyz[0]),
        float(centroid_xyz[1]),
        float(centroid_xyz[2]),
        float(point_radius),
        float(occupancy_value),
        "C",
        int(residue_ids[0]),
        int(residue_ids[1]),
        int(residue_ids[2]),
        int(residue_ids[3]),
        float(point_xyz[0]),
        float(point_xyz[1]),
        float(point_xyz[2]),
        int(safe_float(pdb_id, 0.0)),
        float(point_length),
        float(throughput),
        int(tunnel_cluster),
        float(qss_values[0]),
        float(qss_values[1]),
        float(qss_values[2]),
        float(qss_values[3]),
        int(atom_serials[0]),
        int(atom_serials[1]),
        int(atom_serials[2]),
        int(atom_serials[3]),
    ]


def build_point_row(selected_indices, tunnel_point_id, point_xyz, point_radius, point_length,
                    tunnel_id, tunnel_length, pdb_id, throughput, tunnel_cluster, context):
    atom_locators = context["atom_locators"]
    elements = context["elements"]
    serials = context["serials"]
    vdw_radii = context["vdw_radii"]
    res_ids = context["res_ids"]
    resnames = context["resnames"]
    hydrophobicities = context["hydrophobicities"]
    row = []

    padded_indices = list(selected_indices[:TOP_ATOMS])
    while len(padded_indices) < TOP_ATOMS:
        padded_indices.append(None)

    for atom_idx in padded_indices:
        if atom_idx is None:
            row.extend(["0", "0", 0.0, 0])
        else:
            row.extend([
                str(atom_locators[atom_idx]),
                str(elements[atom_idx]),
                float(vdw_radii[atom_idx]),
                int(serials[atom_idx]),
            ])

    for atom_idx in padded_indices:
        if atom_idx is None:
            row.extend([0, "0", 0.0])
        else:
            row.extend([
                int(res_ids[atom_idx]),
                str(resnames[atom_idx]),
                float(hydrophobicities[atom_idx]),
            ])

    row.extend([
        float(tunnel_point_id),
        float(point_xyz[0]),
        float(point_xyz[1]),
        float(point_xyz[2]),
        float(point_radius),
        float(point_length),
        0.0,
        0,
        0.0,
        0.0,
        tunnel_id,
        float(tunnel_length),
        pdb_id,
        float(throughput),
        int(tunnel_cluster),
    ])
    return row


def build_zero_row():
    row = [0] * len(OUTPUT_HEADER)
    for idx in (0, 4, 8, 12):
        row[idx] = "0"
    return row


def build_terminal_row(tunnel_id, tunnel_length, pdb_id, throughput, tunnel_cluster):
    row = []
    for _ in range(TOP_ATOMS):
        row.extend(["1", "0", 0.0, 1])
    for _ in range(TOP_ATOMS):
        row.extend([1, "0", 0.0])

    row.extend([
        1.0,
        0.0, 0.0, 0.0,
        0.0, 0.0,
        1.0,
        0,
        1.0, 0.0,
        tunnel_id,
        float(tunnel_length),
        pdb_id,
        float(throughput),
        int(tunnel_cluster),
    ])
    return row


def init_residue_stat(resname=""):
    return {
        "resname": str(resname or "").strip(),
        "point_count": 0,
        "radius_sum": 0.0,
        "radius_min": None,
        "radius_max": None,
        "path_count": 0,
        "partners": {},
    }


def init_combination_stat(residue_items):
    return {
        "residue_names": tuple(str(resname or "").strip() for _, resname in residue_items),
        "point_count": 0,
        "radius_sum": 0.0,
        "radius_min": None,
        "radius_max": None,
        "path_count": 0,
    }


def ensure_residue_stat(stats_dict, residue_id, resname=""):
    residue_id = int(residue_id)
    stat = stats_dict.get(residue_id)
    clean_name = str(resname or "").strip()
    if stat is None:
        stat = init_residue_stat(clean_name)
        stats_dict[residue_id] = stat
    elif clean_name and not stat["resname"]:
        stat["resname"] = clean_name
    return stat


def ensure_combination_stat(stats_dict, residue_items):
    combo_key = tuple(int(residue_id) for residue_id, _ in residue_items)
    stat = stats_dict.get(combo_key)
    if stat is None:
        stat = init_combination_stat(residue_items)
        stats_dict[combo_key] = stat
    return combo_key, stat


def iter_residue_combinations(residue_items):
    item_count = len(residue_items)
    if item_count < 2:
        return

    for idx0 in range(item_count - 1):
        item0 = residue_items[idx0]
        for idx1 in range(idx0 + 1, item_count):
            item1 = residue_items[idx1]
            yield (item0, item1)

            for idx2 in range(idx1 + 1, item_count):
                item2 = residue_items[idx2]
                yield (item0, item1, item2)

                for idx3 in range(idx2 + 1, item_count):
                    yield (item0, item1, item2, residue_items[idx3])


def update_point_residue_stats(stats_dict, point_residues, point_radius):
    radius_value = safe_float(point_radius)
    for residue_id, resname in point_residues.items():
        if residue_id <= 0:
            continue
        stat = ensure_residue_stat(stats_dict, residue_id, resname)
        stat["point_count"] += 1
        stat["radius_sum"] += radius_value
        stat["radius_min"] = radius_value if stat["radius_min"] is None else min(stat["radius_min"], radius_value)
        stat["radius_max"] = radius_value if stat["radius_max"] is None else max(stat["radius_max"], radius_value)


def update_point_combination_stats(stats_dict, point_residues, point_radius, path_combination_keys):
    residue_items = tuple(
        (int(residue_id), str(resname or "").strip())
        for residue_id, resname in sorted(point_residues.items())
        if int(residue_id) > 0
    )
    if len(residue_items) < 2:
        return

    radius_value = safe_float(point_radius)
    for combination in iter_residue_combinations(residue_items):
        combo_key, stat = ensure_combination_stat(stats_dict, combination)
        stat["point_count"] += 1
        stat["radius_sum"] += radius_value
        stat["radius_min"] = radius_value if stat["radius_min"] is None else min(stat["radius_min"], radius_value)
        stat["radius_max"] = radius_value if stat["radius_max"] is None else max(stat["radius_max"], radius_value)
        path_combination_keys.add(combo_key)


def update_path_residue_stats(stats_dict, path_residues):
    residue_items = [(int(residue_id), str(resname or "").strip()) for residue_id, resname in path_residues.items() if int(residue_id) > 0]

    for residue_id, resname in residue_items:
        stat = ensure_residue_stat(stats_dict, residue_id, resname)
        stat["path_count"] += 1

    for residue_id, _ in residue_items:
        stat = stats_dict[residue_id]
        partners = stat["partners"]
        for partner_id, partner_resname in residue_items:
            if partner_id == residue_id:
                continue
            partner_entry = partners.get(partner_id)
            if partner_entry is None:
                partners[partner_id] = {
                    "resname": partner_resname,
                    "shared_path_count": 1,
                }
            else:
                if partner_resname and not partner_entry["resname"]:
                    partner_entry["resname"] = partner_resname
                partner_entry["shared_path_count"] += 1


def update_path_combination_stats(stats_dict, path_combination_keys):
    for combo_key in path_combination_keys:
        stat = stats_dict.get(combo_key)
        if stat is not None:
            stat["path_count"] += 1


def merge_partner_stats(target_partners, source_partners):
    for partner_id, partner_info in source_partners.items():
        partner_id = int(partner_id)
        target_entry = target_partners.get(partner_id)
        source_resname = str(partner_info.get("resname", "")).strip()
        source_count = int(partner_info.get("shared_path_count", 0))
        if target_entry is None:
            target_partners[partner_id] = {
                "resname": source_resname,
                "shared_path_count": source_count,
            }
        else:
            if source_resname and not target_entry["resname"]:
                target_entry["resname"] = source_resname
            target_entry["shared_path_count"] += source_count


def merge_residue_stats(target_stats, source_stats):
    for residue_id, source in source_stats.items():
        residue_id = int(residue_id)
        target = ensure_residue_stat(target_stats, residue_id, source.get("resname", ""))
        target["point_count"] += int(source.get("point_count", 0))
        target["radius_sum"] += float(source.get("radius_sum", 0.0))

        source_min = source.get("radius_min")
        if source_min is not None:
            target["radius_min"] = source_min if target["radius_min"] is None else min(target["radius_min"], source_min)

        source_max = source.get("radius_max")
        if source_max is not None:
            target["radius_max"] = source_max if target["radius_max"] is None else max(target["radius_max"], source_max)

        target["path_count"] += int(source.get("path_count", 0))
        merge_partner_stats(target["partners"], source.get("partners", {}))


def merge_combination_stats(target_stats, source_stats):
    for combo_key, source in source_stats.items():
        target = target_stats.get(combo_key)
        if target is None:
            target_stats[combo_key] = {
                "residue_names": tuple(source.get("residue_names", ())),
                "point_count": int(source.get("point_count", 0)),
                "radius_sum": float(source.get("radius_sum", 0.0)),
                "radius_min": source.get("radius_min"),
                "radius_max": source.get("radius_max"),
                "path_count": int(source.get("path_count", 0)),
            }
            continue

        if not target["residue_names"]:
            target["residue_names"] = tuple(source.get("residue_names", ()))
        target["point_count"] += int(source.get("point_count", 0))
        target["radius_sum"] += float(source.get("radius_sum", 0.0))

        source_min = source.get("radius_min")
        if source_min is not None:
            target["radius_min"] = source_min if target["radius_min"] is None else min(target["radius_min"], source_min)

        source_max = source.get("radius_max")
        if source_max is not None:
            target["radius_max"] = source_max if target["radius_max"] is None else max(target["radius_max"], source_max)

        target["path_count"] += int(source.get("path_count", 0))


def format_count_mapping(count_mapping):
    if not count_mapping:
        return ""
    items = sorted(count_mapping.items(), key=lambda item: (-item[1], str(item[0])))
    return ";".join(f"{key}:{value}" for key, value in items)


def write_residue_statistics(output_path, residue_stats):
    ensure_output_parent(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(RESIDUE_STATS_HEADER)

        for residue_id in sorted(residue_stats):
            stat = residue_stats[residue_id]
            point_count = int(stat["point_count"])
            avg_radius = (stat["radius_sum"] / point_count) if point_count else 0.0
            partners = stat["partners"]
            sorted_partners = sorted(
                partners.items(),
                key=lambda item: (-int(item[1].get("shared_path_count", 0)), int(item[0])),
            )

            partner_ids = ";".join(str(partner_id) for partner_id, _ in sorted_partners)
            partner_details = ";".join(
                f"{partner_id}:{partner_info.get('resname', '')}:{int(partner_info.get('shared_path_count', 0))}"
                for partner_id, partner_info in sorted_partners
            )

            partner_type_unique_counts = defaultdict(int)
            partner_type_path_counts = defaultdict(int)
            for _, partner_info in sorted_partners:
                partner_resname = str(partner_info.get("resname", "")).strip() or "UNK"
                partner_type_unique_counts[partner_resname] += 1
                partner_type_path_counts[partner_resname] += int(partner_info.get("shared_path_count", 0))

            writer.writerow([
                residue_id,
                stat["resname"],
                point_count,
                int(stat["path_count"]),
                avg_radius,
                0.0 if stat["radius_min"] is None else stat["radius_min"],
                0.0 if stat["radius_max"] is None else stat["radius_max"],
                len(partners),
                partner_ids,
                partner_details,
                format_count_mapping(partner_type_unique_counts),
                format_count_mapping(partner_type_path_counts),
            ])


def write_residue_combination_statistics(output_path, combination_stats):
    ensure_output_parent(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(RESIDUE_COMBINATION_STATS_HEADER)

        for combo_key in sorted(combination_stats, key=lambda item: (len(item), item)):
            stat = combination_stats[combo_key]
            point_count = int(stat["point_count"])
            avg_radius = (stat["radius_sum"] / point_count) if point_count else 0.0
            min_radius = 0.0 if stat["radius_min"] is None else stat["radius_min"]
            max_radius = 0.0 if stat["radius_max"] is None else stat["radius_max"]

            writer.writerow([
                len(combo_key),
                ";".join(str(residue_id) for residue_id in combo_key),
                ";".join(stat["residue_names"]),
                point_count,
                int(stat["path_count"]),
                avg_radius,
                min_radius,
                max_radius,
                max_radius - min_radius,
            ])


def natural_sort_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def parse_residue_ids(value):
    residue_ids = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part:
            continue
        residue_ids.append(int(part))
    return tuple(residue_ids)


def parse_residue_names(value):
    return tuple(part.strip() for part in str(value or "").split(";") if part.strip())


def read_distance_relation_combinations(combination_stats_path):
    combinations_to_process = []
    unique_residue_ids = set()

    with open(combination_stats_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            residue_ids = parse_residue_ids(row.get("Residue_IDs", ""))
            if len(residue_ids) not in (2, 3, 4):
                continue

            residue_names = parse_residue_names(row.get("Residue_names", ""))
            if len(residue_names) < len(residue_ids):
                residue_names = residue_names + ("",) * (len(residue_ids) - len(residue_names))
            elif len(residue_names) > len(residue_ids):
                residue_names = residue_names[:len(residue_ids)]

            pair_ids = tuple(combinations(residue_ids, 2))
            combinations_to_process.append({
                "size": len(residue_ids),
                "residue_ids": residue_ids,
                "residue_names": residue_names,
                "pair_ids": pair_ids,
            })
            unique_residue_ids.update(residue_ids)

    unique_residue_ids = tuple(sorted(unique_residue_ids))
    return combinations_to_process, unique_residue_ids


def iter_pdb_frame_paths(pdb_root):
    frame_paths = glob.glob(os.path.join(pdb_root, "*.pdb"))
    return sorted(frame_paths, key=natural_sort_key)


def extract_target_residue_centroids(pdb_path, target_index_by_residue_id, ignore_h=False):
    residue_count = len(target_index_by_residue_id)
    sums = np.zeros((residue_count, 3), dtype=np.float64)
    counts = np.zeros((residue_count,), dtype=np.int32)
    hydrogen_elements = HYDROGEN_ELEMENTS
    mutation_lookup = MUTATION_LOOKUP

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            try:
                residue_id = int(line[22:26])
            except ValueError:
                continue

            residue_index = target_index_by_residue_id.get(residue_id)
            if residue_index is None:
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not element:
                element = get_element_from_name(atom_name)
            if ignore_h and element in hydrogen_elements:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            if mutation_lookup:
                resname = line[17:20].strip()
                get_mutated_resname(residue_id, resname, mutation_lookup)

            sums[residue_index] += (x, y, z)
            counts[residue_index] += 1

    valid = counts > 0
    centroids = np.full((residue_count, 3), np.nan, dtype=np.float64)
    centroids[valid] = sums[valid] / counts[valid, np.newaxis]
    return centroids, valid


def build_residue_distance_matrix(centroids, valid_mask):
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distance_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    invalid = ~valid_mask
    if np.any(invalid):
        distance_matrix[invalid, :] = np.nan
        distance_matrix[:, invalid] = np.nan
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def format_distance_cell(distance_value, cumulative_value):
    precision = RESIDUE_DISTANCE_PRECISION
    return f"{distance_value:.{precision}f};{cumulative_value:.{precision}f}"


def ensure_output_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def build_pair_index(combinations_to_process, residue_index_by_id):
    pair_to_index = {}
    combo_pair_indices = []

    for combo in combinations_to_process:
        pair_indices = []
        for pair in combo["pair_ids"]:
            pair_key = tuple(sorted((int(pair[0]), int(pair[1]))))
            pair_index = pair_to_index.get(pair_key)
            if pair_index is None:
                pair_index = len(pair_to_index)
                pair_to_index[pair_key] = pair_index
            pair_indices.append(pair_index)
        combo_pair_indices.append(pair_indices)

    pair_items = sorted(pair_to_index.items(), key=lambda item: item[1])
    pair_residue_indices = np.asarray(
        [(residue_index_by_id[pair[0]], residue_index_by_id[pair[1]]) for pair, _ in pair_items],
        dtype=np.int32,
    )
    return pair_items, pair_residue_indices, combo_pair_indices


def calculate_pair_distances(centroids, valid_mask, pair_residue_indices):
    coords_1 = centroids[pair_residue_indices[:, 0]]
    coords_2 = centroids[pair_residue_indices[:, 1]]
    deltas = coords_1 - coords_2
    pair_values = np.sqrt(np.sum(deltas * deltas, axis=1))
    valid_pairs = valid_mask[pair_residue_indices[:, 0]] & valid_mask[pair_residue_indices[:, 1]]
    pair_values[~valid_pairs] = np.nan
    return pair_values


def init_distance_worker(residue_index_by_id, pair_residue_indices, ignore_h):
    global _DISTANCE_WORKER_RESIDUE_INDEX_BY_ID
    global _DISTANCE_WORKER_PAIR_RESIDUE_INDICES
    global _DISTANCE_WORKER_IGNORE_H
    _DISTANCE_WORKER_RESIDUE_INDEX_BY_ID = residue_index_by_id
    _DISTANCE_WORKER_PAIR_RESIDUE_INDICES = pair_residue_indices
    _DISTANCE_WORKER_IGNORE_H = ignore_h


def calculate_pair_distances_for_frame(task):
    frame_idx, frame_path = task
    centroids, valid_mask = extract_target_residue_centroids(
        frame_path,
        _DISTANCE_WORKER_RESIDUE_INDEX_BY_ID,
        ignore_h=_DISTANCE_WORKER_IGNORE_H,
    )
    pair_values = calculate_pair_distances(centroids, valid_mask, _DISTANCE_WORKER_PAIR_RESIDUE_INDICES)
    return frame_idx, pair_values.astype(np.float32)


def iter_pair_distance_results(frame_paths, residue_index_by_id, pair_residue_indices):
    tasks = list(enumerate(frame_paths))
    worker_count = min(RESIDUE_DISTANCE_FRAME_WORKERS, len(tasks))
    if worker_count <= 1:
        init_distance_worker(residue_index_by_id, pair_residue_indices, IGNORE_H)
        for task in tasks:
            yield calculate_pair_distances_for_frame(task)
        return

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=init_distance_worker,
        initargs=(residue_index_by_id, pair_residue_indices, IGNORE_H),
    ) as executor:
        yield from executor.map(
            calculate_pair_distances_for_frame,
            tasks,
            chunksize=max(1, RESIDUE_DISTANCE_FRAME_CHUNKSIZE),
        )


def normalize_frame_id(value):
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        digits = "".join(re.findall(r"\d", text))
        if digits:
            return str(int(digits))
    return text


def build_frame_id_to_index(frame_paths):
    frame_id_to_index = {}
    for frame_idx, frame_path in enumerate(frame_paths):
        frame_id = normalize_frame_id(extract_pdb_digits(os.path.basename(frame_path)))
        if frame_id:
            frame_id_to_index[frame_id] = frame_idx
    return frame_id_to_index


def read_pair_path_presence(atom_network_path, pair_to_index, frame_id_to_index, pair_count, frame_count):
    presence = np.zeros((pair_count, frame_count), dtype=bool)
    if not atom_network_path or not os.path.exists(atom_network_path):
        print(f"[Warn] Atom network CSV not found, skipped path-presence stats: {atom_network_path}")
        return presence, 0, 0

    rows_processed = 0
    marks = 0
    residue_columns = [f"T_F_M_{idx}_ID" for idx in range(1, TOP_ATOMS + 1)]

    with open(atom_network_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_processed += 1
            if safe_float(row.get("T_F_freedom", 0.0)) != 0.0:
                continue

            frame_idx = frame_id_to_index.get(normalize_frame_id(row.get("pdb_id", "")))
            if frame_idx is None:
                continue

            residue_ids = set()
            for column in residue_columns:
                try:
                    residue_id = int(float(str(row.get(column, "")).strip()))
                except ValueError:
                    continue
                if residue_id > 0:
                    residue_ids.add(residue_id)

            if len(residue_ids) < 2:
                continue

            for residue_1, residue_2 in combinations(sorted(residue_ids), 2):
                pair_idx = pair_to_index.get((residue_1, residue_2))
                if pair_idx is None:
                    continue
                if not presence[pair_idx, frame_idx]:
                    presence[pair_idx, frame_idx] = True
                    marks += 1

    print(f"Read {rows_processed} atom-network rows, marked {marks} pair/frame path-presence events")
    return presence, rows_processed, marks


def build_presence_matrix_from_accumulator(presence_accumulator, pair_to_index, frame_id_to_index, pair_count, frame_count):
    presence = np.zeros((pair_count, frame_count), dtype=bool)
    marks = 0
    for frame_id, frame_pairs in presence_accumulator.items():
        frame_idx = frame_id_to_index.get(normalize_frame_id(frame_id))
        if frame_idx is None:
            continue
        for pair in frame_pairs:
            pair_idx = pair_to_index.get(pair)
            if pair_idx is None or presence[pair_idx, frame_idx]:
                continue
            presence[pair_idx, frame_idx] = True
            marks += 1
    print(f"Built {marks} pair/frame path-presence events from 03 in-memory accumulator")
    return presence, marks


def safe_nan_stat(values, fn, default=np.nan):
    if values.size == 0:
        return default
    return float(fn(values))


def cohen_d(group_1, group_2):
    n1 = len(group_1)
    n2 = len(group_2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = float(np.var(group_1, ddof=1))
    var2 = float(np.var(group_2, ddof=1))
    pooled = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    if pooled <= 0.0:
        return np.nan
    return float((np.mean(group_1) - np.mean(group_2)) / np.sqrt(pooled))


def benjamini_hochberg_qvalues(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values, np.nan)
    valid_mask = np.isfinite(p_values)
    valid_p = p_values[valid_mask]
    n = len(valid_p)
    if n == 0:
        return q_values

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    sorted_q = sorted_p * n / (np.arange(n, dtype=np.float64) + 1.0)
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)

    valid_indices = np.where(valid_mask)[0]
    q_values[valid_indices[order]] = sorted_q
    return q_values


def format_optional_float(value):
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{RESIDUE_DISTANCE_PRECISION}g}"


def write_pair_path_presence_statistics(
        output_path,
        atom_network_path,
        pair_items,
        pair_distance_values,
        frame_paths,
        presence_accumulator=None):
    ensure_output_parent(output_path)
    pair_to_index = {pair: index for pair, index in pair_items}
    frame_id_to_index = build_frame_id_to_index(frame_paths)
    pair_count, frame_count = pair_distance_values.shape
    if presence_accumulator is not None:
        presence, marks = build_presence_matrix_from_accumulator(
            presence_accumulator,
            pair_to_index,
            frame_id_to_index,
            pair_count,
            frame_count,
        )
    else:
        presence, _rows_processed, marks = read_pair_path_presence(
            atom_network_path,
            pair_to_index,
            frame_id_to_index,
            pair_count,
            frame_count,
        )

    rows = []
    welch_p_values = []
    mannwhitney_p_values = []

    for pair_idx, (pair, _index) in enumerate(pair_items):
        distances = np.asarray(pair_distance_values[pair_idx, :], dtype=np.float64)
        finite_mask = np.isfinite(distances)
        present_mask = presence[pair_idx, :] & finite_mask
        absent_mask = (~presence[pair_idx, :]) & finite_mask
        present = distances[present_mask]
        absent = distances[absent_mask]

        welch_stat = np.nan
        welch_p = np.nan
        mannwhitney_stat = np.nan
        mannwhitney_p = np.nan
        note_parts = []

        if len(present) >= 2 and len(absent) >= 2:
            try:
                welch = scipy_stats.ttest_ind(present, absent, equal_var=False, nan_policy="omit")
                welch_stat = float(welch.statistic)
                welch_p = float(welch.pvalue)
            except Exception as exc:
                note_parts.append(f"welch_error:{exc}")
        else:
            note_parts.append("welch_requires_at_least_2_values_per_group")

        if len(present) >= 1 and len(absent) >= 1:
            try:
                mannwhitney = scipy_stats.mannwhitneyu(present, absent, alternative="two-sided")
                mannwhitney_stat = float(mannwhitney.statistic)
                mannwhitney_p = float(mannwhitney.pvalue)
            except Exception as exc:
                note_parts.append(f"mannwhitney_error:{exc}")
        else:
            note_parts.append("mannwhitney_requires_values_in_both_groups")

        present_mean = safe_nan_stat(present, np.mean)
        absent_mean = safe_nan_stat(absent, np.mean)
        mean_diff = present_mean - absent_mean if np.isfinite(present_mean) and np.isfinite(absent_mean) else np.nan

        rows.append({
            "pair": pair,
            "present_count": len(present),
            "absent_count": len(absent),
            "present_mean": present_mean,
            "absent_mean": absent_mean,
            "mean_diff": mean_diff,
            "present_median": safe_nan_stat(present, np.median),
            "absent_median": safe_nan_stat(absent, np.median),
            "present_std": safe_nan_stat(present, lambda values: np.std(values, ddof=1)) if len(present) >= 2 else np.nan,
            "absent_std": safe_nan_stat(absent, lambda values: np.std(values, ddof=1)) if len(absent) >= 2 else np.nan,
            "welch_stat": welch_stat,
            "welch_p": welch_p,
            "mannwhitney_stat": mannwhitney_stat,
            "mannwhitney_p": mannwhitney_p,
            "cohen_d": cohen_d(present, absent),
            "notes": ";".join(note_parts),
        })
        welch_p_values.append(welch_p)
        mannwhitney_p_values.append(mannwhitney_p)

    welch_q_values = benjamini_hochberg_qvalues(welch_p_values)
    mannwhitney_q_values = benjamini_hochberg_qvalues(mannwhitney_p_values)

    header = [
        "Pair_ID",
        "Residue_ID_1",
        "Residue_ID_2",
        "Path_present_frame_count",
        "Path_absent_frame_count",
        "Path_present_distance_mean",
        "Path_absent_distance_mean",
        "Mean_difference_present_minus_absent",
        "Path_present_distance_median",
        "Path_absent_distance_median",
        "Path_present_distance_std",
        "Path_absent_distance_std",
        "Welch_t_statistic",
        "Welch_p_value",
        "Welch_q_value_bh_fdr",
        "MannWhitney_U_statistic",
        "MannWhitney_p_value",
        "MannWhitney_q_value_bh_fdr",
        "Cohen_d",
        "Significant_Welch_q_lt_0.05",
        "Significant_MannWhitney_q_lt_0.05",
        "Notes",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for row_idx, row in enumerate(rows):
            pair = row["pair"]
            welch_q = welch_q_values[row_idx]
            mannwhitney_q = mannwhitney_q_values[row_idx]
            writer.writerow([
                f"{pair[0]}-{pair[1]}",
                pair[0],
                pair[1],
                row["present_count"],
                row["absent_count"],
                format_optional_float(row["present_mean"]),
                format_optional_float(row["absent_mean"]),
                format_optional_float(row["mean_diff"]),
                format_optional_float(row["present_median"]),
                format_optional_float(row["absent_median"]),
                format_optional_float(row["present_std"]),
                format_optional_float(row["absent_std"]),
                format_optional_float(row["welch_stat"]),
                format_optional_float(row["welch_p"]),
                format_optional_float(welch_q),
                format_optional_float(row["mannwhitney_stat"]),
                format_optional_float(row["mannwhitney_p"]),
                format_optional_float(mannwhitney_q),
                format_optional_float(row["cohen_d"]),
                bool(np.isfinite(welch_q) and welch_q < 0.05),
                bool(np.isfinite(mannwhitney_q) and mannwhitney_q < 0.05),
                row["notes"],
            ])

    print(f"[OK] Wrote {output_path}")
    print(f"Pair/frame path-presence events: {marks}")
    return len(rows), marks


def write_combination_cumulative_distance_outputs(
        output_path,
        combinations_to_process,
        combo_pair_indices,
        pair_distance_values,
        frame_count):
    ensure_output_parent(output_path)
    npy_path = os.path.splitext(output_path)[0] + ".npy"
    cell_count = len(combinations_to_process) * frame_count
    csv_mode = CUMULATIVE_DISTANCE_CSV_MODE
    write_csv = csv_mode in {"1", "true", "yes", "on", "always", "force"}
    if csv_mode in {"0", "false", "no", "off", "never", "skip"}:
        write_csv = False
    elif csv_mode == "auto":
        write_csv = cell_count <= CUMULATIVE_DISTANCE_CSV_MAX_CELLS
    elif csv_mode not in {"1", "true", "yes", "on", "always", "force"}:
        raise ValueError(
            "Unsupported PATHVIS_CUMULATIVE_DISTANCE_CSV_MODE="
            f"{CUMULATIVE_DISTANCE_CSV_MODE!r}. Use 'auto', 'always', or 'never'."
        )

    cumulative_matrix = np.lib.format.open_memmap(
        npy_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(combinations_to_process), frame_count),
    )

    header = [
        "Combination_size",
        "Residue_IDs",
        "Residue_names",
        "Pair_IDs",
        "Pair_count",
        "Value_definition",
    ]
    header.extend(f"Frame_{frame_idx + 1}" for frame_idx in range(frame_count))

    csv_handle = None
    writer = None
    if write_csv:
        csv_handle = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_handle)
        writer.writerow(header)
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(
            f"[SKIP] Large cumulative CSV skipped: {output_path} "
            f"({cell_count} frame cells > {CUMULATIVE_DISTANCE_CSV_MAX_CELLS}). "
            f"Numeric matrix will be written to {npy_path}"
        )

    try:
        combo_count = len(combinations_to_process)
        max_pair_count = max((len(indices) for indices in combo_pair_indices), default=0)
        chunk_size = max(1, CUMULATIVE_DISTANCE_COMBO_CHUNK_SIZE)

        for chunk_start in range(0, combo_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, combo_count)
            chunk_len = chunk_end - chunk_start
            combo_distances = np.zeros((chunk_len, frame_count), dtype=np.float32)
            invalid_mask = np.zeros((chunk_len, frame_count), dtype=bool)

            for pair_slot in range(max_pair_count):
                slot_indices = []
                slot_rows = []
                for local_row, pair_indices in enumerate(combo_pair_indices[chunk_start:chunk_end]):
                    if pair_slot < len(pair_indices):
                        slot_rows.append(local_row)
                        slot_indices.append(pair_indices[pair_slot])

                if not slot_indices:
                    continue

                slot_rows = np.asarray(slot_rows, dtype=np.int32)
                slot_values = np.asarray(pair_distance_values[slot_indices, :], dtype=np.float32)
                combo_distances[slot_rows, :] += np.nan_to_num(slot_values, nan=0.0)
                invalid_mask[slot_rows, :] |= np.isnan(slot_values)

            combo_distances[invalid_mask] = np.nan
            valid_distances = np.where(np.isnan(combo_distances), 0.0, combo_distances)
            cumulative_block = np.cumsum(valid_distances, axis=1, dtype=np.float32)
            cumulative_block[np.isnan(combo_distances)] = np.nan
            cumulative_matrix[chunk_start:chunk_end, :] = cumulative_block

            if write_csv:
                for local_row, combo in enumerate(combinations_to_process[chunk_start:chunk_end]):
                    cumulative_distances = cumulative_block[local_row, :]
                    frame_cells = [
                        "" if np.isnan(cumulative_value) else f"{float(cumulative_value):.{RESIDUE_DISTANCE_PRECISION}f}"
                        for cumulative_value in cumulative_distances
                    ]
                    writer.writerow([
                        combo["size"],
                        ";".join(str(residue_id) for residue_id in combo["residue_ids"]),
                        ";".join(combo["residue_names"]),
                        "|".join(f"{pair[0]}-{pair[1]}" for pair in combo["pair_ids"]),
                        len(combo["pair_ids"]),
                        "cumulative_sum_of_pair_distances_from_frame_1_to_frame_i",
                        *frame_cells,
                    ])

            written_rows = chunk_end
            if written_rows % PROGRESS_INTERVAL == 0 or written_rows == combo_count:
                target = "CSV + NPY" if write_csv else "NPY"
                print(f"Wrote {written_rows}/{combo_count} cumulative combination distance rows to {target}")
    finally:
        cumulative_matrix.flush()
        del cumulative_matrix
        if csv_handle is not None:
            csv_handle.close()

    print(f"[OK] Wrote {npy_path}")
    if write_csv:
        print(f"[OK] Wrote {output_path}")
    return npy_path, write_csv


def write_residue_distance_relations_compact(
        relation_output_path,
        pair_output_path,
        combination_stats_path,
        pdb_root,
        atom_network_path=None,
        path_presence_stats_output_path=None,
        presence_accumulator=None):
    combinations_to_process, unique_residue_ids = read_distance_relation_combinations(combination_stats_path)
    frame_paths = iter_pdb_frame_paths(pdb_root)
    if RESIDUE_DISTANCE_MAX_FRAMES > 0:
        frame_paths = frame_paths[:RESIDUE_DISTANCE_MAX_FRAMES]

    ensure_output_parent(pair_output_path)
    if WRITE_RESIDUE_DISTANCE_RELATION_CSV:
        ensure_output_parent(relation_output_path)

    relation_header = [
        "Combination_size",
        "Residue_IDs",
        "Residue_names",
        "Pair_IDs",
        "Pair_count",
        "Value_definition",
        "Cumulative_distance_npy",
        "Cumulative_distance_csv",
    ]

    if not combinations_to_process:
        if WRITE_RESIDUE_DISTANCE_RELATION_CSV:
            with open(relation_output_path, "w", newline="", encoding="utf-8") as fout:
                csv.writer(fout).writerow(relation_header)
        empty_npy_path = os.path.splitext(pair_output_path)[0] + ".npy"
        np.save(empty_npy_path, np.empty((0, 0), dtype=np.float32))
        print(f"[Warn] No residue combinations available for distance relations: {combination_stats_path}")
        return 0, 0, 0

    if not frame_paths:
        raise FileNotFoundError(f"No PDB frame files found in {pdb_root}")

    residue_index_by_id = {residue_id: idx for idx, residue_id in enumerate(unique_residue_ids)}
    pair_items, pair_residue_indices, combo_pair_indices = build_pair_index(
        combinations_to_process,
        residue_index_by_id,
    )
    pair_count = len(pair_items)
    frame_count = len(frame_paths)

    tmp_path = f"{pair_output_path}.tmp_pair_distance_values.dat"
    pair_distance_values = np.memmap(
        tmp_path,
        dtype=np.float32,
        mode="w+",
        shape=(pair_count, frame_count),
    )

    try:
        for frame_idx, pair_values in iter_pair_distance_results(frame_paths, residue_index_by_id, pair_residue_indices):
            pair_distance_values[:, frame_idx] = pair_values

            processed_frames = frame_idx + 1
            if processed_frames % PROGRESS_INTERVAL == 0 or processed_frames == frame_count:
                print(
                    f"Processed {processed_frames}/{frame_count} residue-distance frames, "
                    f"{pair_count} unique pairs"
                )

        pair_distance_values.flush()

        cumulative_npy_path, cumulative_csv_written = write_combination_cumulative_distance_outputs(
            output_path=pair_output_path,
            combinations_to_process=combinations_to_process,
            combo_pair_indices=combo_pair_indices,
            pair_distance_values=pair_distance_values,
            frame_count=frame_count,
        )

        if WRITE_RESIDUE_DISTANCE_RELATION_CSV:
            with open(relation_output_path, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                writer.writerow(relation_header)
                cumulative_npy_name = os.path.basename(cumulative_npy_path)
                cumulative_csv_name = os.path.basename(pair_output_path) if cumulative_csv_written else ""
                for combo in combinations_to_process:
                    writer.writerow([
                        combo["size"],
                        ";".join(str(residue_id) for residue_id in combo["residue_ids"]),
                        ";".join(combo["residue_names"]),
                        "|".join(f"{pair[0]}-{pair[1]}" for pair in combo["pair_ids"]),
                        len(combo["pair_ids"]),
                        "cumulative_sum_of_pair_distances_from_frame_1_to_frame_i",
                        cumulative_npy_name,
                        cumulative_csv_name,
                    ])

        if path_presence_stats_output_path and atom_network_path and os.path.exists(atom_network_path):
            write_pair_path_presence_statistics(
                output_path=path_presence_stats_output_path,
                atom_network_path=atom_network_path,
                pair_items=pair_items,
                pair_distance_values=pair_distance_values,
                frame_paths=frame_paths,
                presence_accumulator=presence_accumulator,
            )
        elif path_presence_stats_output_path and presence_accumulator is not None:
            write_pair_path_presence_statistics(
                output_path=path_presence_stats_output_path,
                atom_network_path=atom_network_path,
                pair_items=pair_items,
                pair_distance_values=pair_distance_values,
                frame_paths=frame_paths,
                presence_accumulator=presence_accumulator,
            )
        elif path_presence_stats_output_path:
            print(f"[Warn] Atom network CSV was not provided or not found, skipped path-presence stats: {atom_network_path}")

        return len(combinations_to_process), frame_count, pair_count
    finally:
        del pair_distance_values
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_residue_distance_relations(output_path, combination_stats_path, pdb_root):
    combinations_to_process, unique_residue_ids = read_distance_relation_combinations(combination_stats_path)
    frame_paths = iter_pdb_frame_paths(pdb_root)
    if RESIDUE_DISTANCE_MAX_FRAMES > 0:
        frame_paths = frame_paths[:RESIDUE_DISTANCE_MAX_FRAMES]

    ensure_output_parent(output_path)
    if not combinations_to_process:
        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            csv.writer(fout).writerow([
                "Combination_size", "Residue_IDs", "Residue_names", "Pair_IDs", "Pair_count",
            ])
        print(f"[Warn] No residue combinations available for distance relations: {combination_stats_path}")
        return 0, 0

    if not frame_paths:
        raise FileNotFoundError(f"No PDB frame files found in {pdb_root}")

    residue_index_by_id = {residue_id: idx for idx, residue_id in enumerate(unique_residue_ids)}
    pair_items, pair_residue_indices, combo_pair_indices = build_pair_index(
        combinations_to_process,
        residue_index_by_id,
    )
    dummy_pair_index = len(pair_items)
    combo_pair_index_matrix = np.full((len(combinations_to_process), 6), dummy_pair_index, dtype=np.int32)
    for combo_idx, pair_indices in enumerate(combo_pair_indices):
        combo_pair_index_matrix[combo_idx, :len(pair_indices)] = pair_indices

    tmp_path = f"{output_path}.tmp_distance_values.dat"
    distance_values = np.memmap(
        tmp_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(combinations_to_process), len(frame_paths)),
    )

    try:
        for frame_idx, frame_path in enumerate(frame_paths):
            centroids, valid_mask = extract_target_residue_centroids(frame_path, residue_index_by_id, ignore_h=IGNORE_H)
            distance_matrix = build_residue_distance_matrix(centroids, valid_mask)
            pair_values = distance_matrix[pair_residue_indices[:, 0], pair_residue_indices[:, 1]]
            pair_values_with_dummy = np.empty((len(pair_values) + 1,), dtype=np.float64)
            pair_values_with_dummy[:-1] = pair_values
            pair_values_with_dummy[-1] = 0.0
            combo_values = pair_values_with_dummy[combo_pair_index_matrix].sum(axis=1)
            distance_values[:, frame_idx] = combo_values.astype(np.float32)

            processed_frames = frame_idx + 1
            if processed_frames % PROGRESS_INTERVAL == 0 or processed_frames == len(frame_paths):
                print(
                    f"Processed {processed_frames}/{len(frame_paths)} residue-distance frames, "
                    f"{len(combinations_to_process)} combinations"
                )

        distance_values.flush()

        header = [
            "Combination_size",
            "Residue_IDs",
            "Residue_names",
            "Pair_IDs",
            "Pair_count",
        ]
        header.extend(f"Frame_{frame_idx + 1}" for frame_idx in range(len(frame_paths)))

        with open(output_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)

            for combo_idx, combo in enumerate(combinations_to_process):
                frame_distances = np.asarray(distance_values[combo_idx, :], dtype=np.float64)
                valid_distances = np.where(np.isnan(frame_distances), 0.0, frame_distances)
                cumulative_distances = np.cumsum(valid_distances)
                frame_cells = [
                    "" if np.isnan(distance_value) else format_distance_cell(distance_value, cumulative_value)
                    for distance_value, cumulative_value in zip(frame_distances, cumulative_distances)
                ]

                writer.writerow([
                    combo["size"],
                    ";".join(str(residue_id) for residue_id in combo["residue_ids"]),
                    ";".join(combo["residue_names"]),
                    "|".join(f"{pair[0]}-{pair[1]}" for pair in combo["pair_ids"]),
                    len(combo["pair_ids"]),
                    *frame_cells,
                ])

                written_rows = combo_idx + 1
                if written_rows % PROGRESS_INTERVAL == 0 or written_rows == len(combinations_to_process):
                    print(f"Wrote {written_rows}/{len(combinations_to_process)} residue-distance rows")

        return len(combinations_to_process), len(frame_paths)
    finally:
        del distance_values
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def write_residue_distance_outputs(
        relation_output_path,
        pair_output_path,
        combination_stats_path,
        pdb_root,
        atom_network_path=None,
        path_presence_stats_output_path=None,
        presence_accumulator=None):
    mode = RESIDUE_DISTANCE_OUTPUT_MODE
    if mode in {"wide", "legacy"}:
        relation_count, frame_count = write_residue_distance_relations(
            relation_output_path,
            combination_stats_path,
            pdb_root,
        )
        return relation_count, frame_count, 0

    if mode not in {"compact", "pair", "pairs"}:
        raise ValueError(
            "Unsupported PATHVIS_RESIDUE_DISTANCE_OUTPUT_MODE="
            f"{RESIDUE_DISTANCE_OUTPUT_MODE!r}. Use 'compact' or 'wide'."
        )

    return write_residue_distance_relations_compact(
        relation_output_path=relation_output_path,
        pair_output_path=pair_output_path,
        combination_stats_path=combination_stats_path,
        pdb_root=pdb_root,
        atom_network_path=atom_network_path,
        path_presence_stats_output_path=path_presence_stats_output_path,
        presence_accumulator=presence_accumulator,
    )


def select_top_atom_indices(distances, indices, point_radii, context):
    serials = context["serials"]
    vdw_radii = context["vdw_radii"]
    adjusted_distances = np.round(
        distances - vdw_radii[indices] - point_radii[:, np.newaxis],
        PRECISION,
    )
    candidate_serials = serials[indices]

    selected_by_point = []
    top_atoms = TOP_ATOMS
    for point_idx in range(indices.shape[0]):
        candidate_indices = indices[point_idx]
        order = np.lexsort((candidate_serials[point_idx], adjusted_distances[point_idx]))
        selected = candidate_indices[order[:top_atoms]]
        selected = selected[np.argsort(serials[selected], kind="stable")]
        selected_by_point.append(selected)
    return selected_by_point


def process_tunnel_group_with_context(task, context):
    rows_group, group_index = task
    meta_row = rows_group[0]
    pdb_name = meta_row[0].strip()
    tunnel_cluster = int(safe_float(meta_row[1], 0.0))
    tunnel_id = meta_row[2].strip()
    throughput = safe_float(meta_row[3])
    tunnel_length = safe_float(meta_row[10])
    pdb_id = extract_pdb_digits(pdb_name)
    tunnel_point_id = float(group_index)

    points_xyz, point_lengths, point_radii = parse_group_points(rows_group)
    write_atom_network = WRITE_ATOM_NETWORK_CSV
    rows_to_write = [build_zero_row()] if write_atom_network else []
    local_residue_stats = {}
    local_combination_stats = {}
    local_presence_accumulator = init_presence_accumulator()
    all_points_rows = []
    occupancy_value = group_index + 1

    if points_xyz.size == 0:
        if write_atom_network:
            rows_to_write.append(build_terminal_row(tunnel_id, tunnel_length, pdb_id, throughput, tunnel_cluster))
        return rows_to_write, all_points_rows, 0, local_residue_stats, local_combination_stats, local_presence_accumulator

    distances, indices = query_nearest_atoms(points_xyz, context)
    res_ids = context["res_ids"]
    resnames = context["resnames"]
    hydrophobicities = context["hydrophobicities"]
    serials = context["serials"]
    selected_by_point = select_top_atom_indices(distances, indices, point_radii, context)
    path_residues = {}
    path_combination_keys = set()

    for point_idx, selected in enumerate(selected_by_point):
        point_residues = {}
        for atom_idx in selected:
            residue_id = int(res_ids[atom_idx])
            if residue_id <= 0:
                continue
            point_residues[residue_id] = str(resnames[atom_idx])

        update_point_residue_stats(local_residue_stats, point_residues, point_radii[point_idx])
        update_point_combination_stats(local_combination_stats, point_residues, point_radii[point_idx], path_combination_keys)
        path_residues.update(point_residues)

        padded_selected = list(selected[:TOP_ATOMS])
        while len(padded_selected) < TOP_ATOMS:
            padded_selected.append(padded_selected[-1] if padded_selected else 0)
        residue_ids = [0, 0, 0, 0]
        qss_values = [0.0, 0.0, 0.0, 0.0]
        atom_serials = [0, 0, 0, 0]
        for slot_idx, atom_idx in enumerate(padded_selected[:TOP_ATOMS]):
            residue_ids[slot_idx] = int(res_ids[atom_idx])
            qss_values[slot_idx] = float(hydrophobicities[atom_idx])
            atom_serials[slot_idx] = int(serials[atom_idx])

        centroid_xyz = compute_point_centroid(selected, context)
        all_points_rows.append(build_all_points_row(
            source_file=f"tunnel_{int(safe_float(tunnel_id, 0.0))}_frame_{int(safe_float(pdb_id, 0.0))}.pdb",
            atom_serial=point_idx + 1,
            res_seq=point_idx + 1,
            centroid_xyz=centroid_xyz,
            point_radius=point_radii[point_idx],
            occupancy_value=occupancy_value,
            residue_ids=residue_ids,
            point_xyz=points_xyz[point_idx],
            pdb_id=pdb_id,
            point_length=point_lengths[point_idx],
            throughput=throughput,
            tunnel_cluster=tunnel_cluster,
            qss_values=qss_values,
            atom_serials=atom_serials,
        ))

        if write_atom_network:
            rows_to_write.append(build_point_row(
                selected_indices=selected,
                tunnel_point_id=tunnel_point_id,
                point_xyz=points_xyz[point_idx],
                point_radius=point_radii[point_idx],
                point_length=point_lengths[point_idx],
                tunnel_id=tunnel_id,
                tunnel_length=tunnel_length,
                pdb_id=pdb_id,
                throughput=throughput,
                tunnel_cluster=tunnel_cluster,
                context=context,
            ))

    update_path_residue_stats(local_residue_stats, path_residues)
    update_path_combination_stats(local_combination_stats, path_combination_keys)
    update_path_pair_presence(local_presence_accumulator, pdb_id, path_residues)
    if write_atom_network:
        rows_to_write.append(build_terminal_row(tunnel_id, tunnel_length, pdb_id, throughput, tunnel_cluster))
    return rows_to_write, all_points_rows, len(points_xyz), local_residue_stats, local_combination_stats, local_presence_accumulator


def process_tunnel_group(task):
    rows_group, _ = task
    pdb_name = rows_group[0][0].strip()
    context = load_structure_context(dataDir, pdb_name, IGNORE_H)
    return process_tunnel_group_with_context(task, context)


def preload_structure_contexts(groups):
    seen_pdb_names = set()
    for rows_group, _ in groups:
        pdb_name = rows_group[0][0].strip()
        if pdb_name in seen_pdb_names:
            continue
        load_structure_context(dataDir, pdb_name, IGNORE_H)
        seen_pdb_names.add(pdb_name)


def build_pdb_group_batches(groups):
    groups_by_pdb = defaultdict(list)
    first_index_by_pdb = {}
    for task in groups:
        rows_group, group_index = task
        pdb_name = rows_group[0][0].strip()
        groups_by_pdb[pdb_name].append(task)
        first_index_by_pdb.setdefault(pdb_name, group_index)

    batches = []
    max_groups = PDB_BATCH_MAX_GROUPS
    for pdb_name in sorted(groups_by_pdb, key=lambda name: first_index_by_pdb[name]):
        pdb_groups = groups_by_pdb[pdb_name]
        if max_groups > 0:
            for start_idx in range(0, len(pdb_groups), max_groups):
                batches.append((pdb_name, pdb_groups[start_idx:start_idx + max_groups]))
        else:
            batches.append((pdb_name, pdb_groups))
    return batches


def process_pdb_group_batch(batch):
    pdb_name, group_tasks = batch
    context = load_structure_context(dataDir, pdb_name, IGNORE_H)
    results = []
    for task in group_tasks:
        _, group_index = task
        results.append((group_index, process_tunnel_group_with_context(task, context)))
    return pdb_name, results


def main():
    start_time = time.time()
    input_profile_path = tunnel_profile_csv
    groups = list(iter_tunnel_groups(input_profile_path))

    groups_processed = 0
    points_processed = 0
    residue_stats = {}
    combination_stats = {}
    pair_presence_accumulator = init_presence_accumulator()
    batches = build_pdb_group_batches(groups)
    pending_results = {}
    next_group_to_write = 0
    all_points_output_enabled = bool(str(all_points_output_path).strip())
    all_points_handle = None
    all_points_writer = None

    if all_points_output_enabled:
        ensure_output_parent(all_points_output_path)
        all_points_handle = open(all_points_output_path, "w", newline="", encoding="utf-8")
        all_points_writer = csv.writer(all_points_handle)
        all_points_writer.writerow(ALL_POINTS_OUTPUT_COLUMNS)

    atom_network_handle = None
    try:
        if WRITE_ATOM_NETWORK_CSV:
            ensure_output_parent(out_path)
            atom_network_handle = open(out_path, "w", newline="", encoding="utf-8")
            writer = csv.writer(atom_network_handle)
            writer.writerow(OUTPUT_HEADER)
        else:
            writer = None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_pdb_group_batch, batch) for batch in batches]
            for future in as_completed(futures):
                _, batch_results = future.result()
                for group_index, result in batch_results:
                    pending_results[group_index] = result

                while next_group_to_write in pending_results:
                    rows_to_write, all_points_rows, point_count, local_residue_stats, local_combination_stats, local_presence_accumulator = pending_results.pop(next_group_to_write)
                    if writer is not None:
                        writer.writerows(rows_to_write)
                    if all_points_writer is not None and all_points_rows:
                        all_points_writer.writerows(all_points_rows)
                    merge_residue_stats(residue_stats, local_residue_stats)
                    merge_combination_stats(combination_stats, local_combination_stats)
                    for frame_id, frame_pairs in local_presence_accumulator.items():
                        pair_presence_accumulator.setdefault(frame_id, set()).update(frame_pairs)

                    groups_processed += 1
                    points_processed += point_count
                    next_group_to_write += 1

                    if groups_processed % PROGRESS_INTERVAL == 0:
                        print(
                            f"Processed {groups_processed} tunnels, {points_processed} points, "
                            f"cached PDBs {load_structure_context.cache_info().currsize}"
                        )
    finally:
        if atom_network_handle is not None:
            atom_network_handle.close()
        if all_points_handle is not None:
            all_points_handle.close()

    if groups_processed != len(groups):
        raise RuntimeError(f"Processed {groups_processed}/{len(groups)} tunnel groups")

    write_residue_statistics(residue_stats_path, residue_stats)
    write_residue_combination_statistics(residue_combination_stats_path, combination_stats)
    distance_relation_count = 0
    distance_relation_frame_count = 0
    distance_pair_count = 0
    if WRITE_RESIDUE_DISTANCE_RELATIONS:
        distance_relation_count, distance_relation_frame_count, distance_pair_count = write_residue_distance_outputs(
            residue_distance_relation_path,
            residue_pair_distance_path,
            residue_combination_stats_path,
            dataDir,
            atom_network_path=out_path,
            path_presence_stats_output_path=residue_pair_path_presence_stats_path,
            presence_accumulator=pair_presence_accumulator,
        )

    elapsed_time = time.time() - start_time
    if WRITE_ATOM_NETWORK_CSV:
        print(f"[OK] Wrote {out_path}")
    if all_points_output_enabled:
        print(f"[OK] Wrote {all_points_output_path}")
    print(f"[OK] Wrote {residue_stats_path}")
    print(f"[OK] Wrote {residue_combination_stats_path}")
    if WRITE_RESIDUE_DISTANCE_RELATIONS:
        print(f"[OK] Wrote {residue_distance_relation_path}")
        if RESIDUE_DISTANCE_OUTPUT_MODE not in {"wide", "legacy"}:
            cumulative_npy_path = os.path.splitext(residue_pair_distance_path)[0] + ".npy"
            if os.path.exists(cumulative_npy_path):
                print(f"[OK] Wrote {cumulative_npy_path}")
            if os.path.exists(residue_pair_distance_path):
                print(f"[OK] Wrote {residue_pair_distance_path}")
            if os.path.exists(residue_pair_path_presence_stats_path):
                print(f"[OK] Wrote {residue_pair_path_presence_stats_path}")
    print(f"Tunnels: {groups_processed}, points: {points_processed}")
    print(f"Residues with statistics: {len(residue_stats)}")
    print(f"Residue combinations with statistics: {len(combination_stats)}")
    if WRITE_RESIDUE_DISTANCE_RELATIONS:
        print(f"Residue distance relation combinations: {distance_relation_count}")
        print(f"Residue distance relation frames: {distance_relation_frame_count}")
        if RESIDUE_DISTANCE_OUTPUT_MODE not in {"wide", "legacy"}:
            print(f"Residue distance unique pairs: {distance_pair_count}")
    print(f"PDB cache: {load_structure_context.cache_info()}")
    print(f"Thread workers: {MAX_WORKERS}, KDTree workers/query: {KD_QUERY_WORKERS}")
    print(f"Elapsed: {elapsed_time:.2f} seconds")


def parse_cli_args():
    parser = argparse.ArgumentParser(description="PathVisual four-atoms and residue statistics")
    parser.add_argument(
        "--tunnel-profile-csv",
        default=tunnel_profile_csv,
        help="input tunnel_profiles.csv path for the normal 03 pipeline",
    )
    parser.add_argument(
        "--pdb-root",
        default=dataDir,
        help="PDB frame directory",
    )
    parser.add_argument(
        "--atom-network-output-csv",
        default=out_path,
        help="output atom_network.csv path for the topology-mapping stage",
    )
    parser.add_argument(
        "--representative-pdb",
        default=representative_pdb_path,
        help="representative PDB path used to build all_points.csv directly in the normal 03 pipeline",
    )
    parser.add_argument(
        "--all-points-output-csv",
        default=all_points_output_path,
        help="output all_points.csv path written directly by the normal 03 pipeline",
    )
    parser.add_argument(
        "--mutations",
        default=None,
        help="comma-separated mutation specs like L176W,F142L; overrides pipeline_config mutations for this run",
    )
    parser.add_argument(
        "--residue-stats-output-csv",
        default=residue_stats_path,
        help="output residue_statistics.csv path for the normal 03 pipeline",
    )
    parser.add_argument(
        "--combination-stats-csv",
        "--combination-stats-output-csv",
        dest="combination_stats_csv",
        default=residue_combination_stats_path,
        help="residue_combination_statistics.csv path",
    )
    parser.add_argument(
        "--residue-distance-only",
        action="store_true",
        help="only write residue distance relation CSV from existing residue combination statistics",
    )
    parser.add_argument(
        "--residue-distance-output-csv",
        default=residue_distance_relation_path,
        help="output residue_distance_relations.csv path",
    )
    parser.add_argument(
        "--residue-pair-output-csv",
        default=residue_pair_distance_path,
        help="output cumulative combination distance CSV path",
    )
    parser.add_argument(
        "--path-presence-stats-output-csv",
        default=residue_pair_path_presence_stats_path,
        help="output residue_pair_path_presence_stats.csv path",
    )
    parser.add_argument(
        "--atom-network-csv",
        default=out_path,
        help="input atom_network.csv path used to classify path-present/path-absent frames",
    )
    parser.add_argument(
        "--skip-residue-distance",
        action="store_true",
        help="skip residue distance relation outputs in the normal 03 pipeline",
    )
    return parser.parse_args()


def run_residue_distance_only(
        combination_stats_path,
        output_path,
        pair_output_path,
        path_presence_stats_output_path,
        atom_network_path,
        pdb_root):
    start_time = time.time()
    if not os.path.exists(combination_stats_path):
        raise FileNotFoundError(f"Residue combination statistics CSV not found: {combination_stats_path}")
    if not os.path.isdir(pdb_root):
        raise NotADirectoryError(f"PDB frame directory not found: {pdb_root}")

    relation_count, frame_count, pair_count = write_residue_distance_outputs(
        output_path,
        pair_output_path,
        combination_stats_path,
        pdb_root,
        atom_network_path=atom_network_path,
        path_presence_stats_output_path=path_presence_stats_output_path,
    )
    elapsed_time = time.time() - start_time
    print(f"[OK] Wrote {output_path}")
    if RESIDUE_DISTANCE_OUTPUT_MODE not in {"wide", "legacy"}:
        cumulative_npy_path = os.path.splitext(pair_output_path)[0] + ".npy"
        if os.path.exists(cumulative_npy_path):
            print(f"[OK] Wrote {cumulative_npy_path}")
        if os.path.exists(pair_output_path):
            print(f"[OK] Wrote {pair_output_path}")
        if os.path.exists(path_presence_stats_output_path):
            print(f"[OK] Wrote {path_presence_stats_output_path}")
    print(f"Residue distance relation combinations: {relation_count}")
    print(f"Residue distance relation frames: {frame_count}")
    if RESIDUE_DISTANCE_OUTPUT_MODE not in {"wide", "legacy"}:
        print(f"Residue distance unique pairs: {pair_count}")
    print(f"Elapsed: {elapsed_time:.2f} seconds")


def cli_main():
    global WRITE_RESIDUE_DISTANCE_RELATIONS
    args = parse_cli_args()
    if args.residue_distance_only:
        run_residue_distance_only(
            combination_stats_path=args.combination_stats_csv,
            output_path=args.residue_distance_output_csv,
            pair_output_path=args.residue_pair_output_csv,
            path_presence_stats_output_path=args.path_presence_stats_output_csv,
            atom_network_path=args.atom_network_csv,
            pdb_root=args.pdb_root,
        )
        return
    configure_runtime_paths(
        tunnel_profile_path=args.tunnel_profile_csv,
        pdb_root=args.pdb_root,
        atom_network_output_path=args.atom_network_output_csv,
        representative_pdb_input_path=args.representative_pdb,
        all_points_csv_output_path=args.all_points_output_csv,
        mutations_input=args.mutations,
        residue_stats_output_path=args.residue_stats_output_csv,
        combination_stats_output_path=args.combination_stats_csv,
        residue_distance_output_path=args.residue_distance_output_csv,
        residue_pair_output_path=args.residue_pair_output_csv,
        path_presence_stats_output_path=args.path_presence_stats_output_csv,
    )
    if args.skip_residue_distance:
        WRITE_RESIDUE_DISTANCE_RELATIONS = False
    main()


if __name__ == "__main__":
    cli_main()
