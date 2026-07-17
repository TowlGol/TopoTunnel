#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast topology-to-path-point bridge.

This script converts stage_03_map_topology.py output (atom_network.csv) directly
to the mapped path-point CSV consumed by stage_05_preprocess_dataset.py. It
preserves the behavioral output of the former centroid/merge/cluster path:

    atom_network.csv -> centroid PDBs -> merged PDBs -> all_points.csv

The PDB files are not written. Centroid coordinates and numeric fields are
rounded to the same precision used by the intermediate PDB representation.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_config import PDB_path, all_points_csv, four_atoms_csv, mutations
from mutation_support import (
    build_mutation_lookup,
    get_mutated_resname,
    validate_mutation_observations,
)


MAX_SERIAL = 99999
ID_COLS = ["T_F_A_1_ID", "T_F_A_2_ID", "T_F_A_3_ID", "T_F_A_4_ID"]
ATOM_SERIAL_COLS = ["T_F_A_1_serial", "T_F_A_2_serial", "T_F_A_3_serial", "T_F_A_4_serial"]
RESIDUE_ID_COLS = ["T_F_M_1_ID", "T_F_M_2_ID", "T_F_M_3_ID", "T_F_M_4_ID"]
RESIDUE_QSS_COLS = [
    "T_F_M_1_hydrophobicity",
    "T_F_M_2_hydrophobicity",
    "T_F_M_3_hydrophobicity",
    "T_F_M_4_hydrophobicity",
]
SENTINEL_NUMERIC_COLS = [
    "T_F_T_ID", "T_F_T_x", "T_F_T_y", "T_F_T_z", "T_F_T_radius",
    "T_F_T_length", "T_F_T_freedom", "T_F_Topo_ID", "T_F_G_x", "T_F_G_y",
]
OUTPUT_COLUMNS = [
    "source_file", "atom_serial", "res_seq", "x", "y", "z", "bfactor", "occupancy", "element",
    "residue_1", "residue_2", "residue_3", "residue_4",
    "x_origin", "y_origin", "z_origin", "frame_id", "length", "throughput", "tunnel_cluster",
    "qss1", "qss2", "qss3", "qss4",
    "atom_1", "atom_2", "atom_3", "atom_4",
]
MUTATION_LOOKUP = build_mutation_lookup(mutations)


def configure_runtime_mutations(raw_mutations):
    global MUTATION_LOOKUP
    MUTATION_LOOKUP = build_mutation_lookup(raw_mutations)


def build_atom_locator(residue_id, resname, atom_name):
    return f"{int(residue_id)}_{str(resname).strip()}_{str(atom_name).strip()}"


def load_atom_records(pdb_path):
    atom_records = {}
    observed_resnames = {}
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB not found: {pdb_path}")

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


def mark_placeholder_rows(df):
    normalized_ids = df[ID_COLS].fillna("").astype(str).apply(lambda col: col.str.strip())
    leading_zero_mask = normalized_ids.eq("0").all(axis=1)
    legacy_terminal_mask = (
        normalized_ids.eq("1").all(axis=1)
        & (df[RESIDUE_ID_COLS].fillna(-1) == 1).all(axis=1)
        & (df[["T_F_T_x", "T_F_T_y", "T_F_T_z", "T_F_T_radius", "T_F_T_length", "T_F_Topo_ID", "T_F_G_y"]] == 0).all(axis=1)
        & (df["T_F_T_ID"] == 1)
        & (df["T_F_T_freedom"] == 1)
        & (df["T_F_G_x"] == 1)
    )
    return leading_zero_mask | legacy_terminal_mask


def load_filtered_dataframe(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    for col in ID_COLS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    for col in RESIDUE_ID_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ATOM_SERIAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
    for col in SENTINEL_NUMERIC_COLS:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0.0)

    df = df.loc[~mark_placeholder_rows(df)].copy()
    df = df.loc[~df[ID_COLS].eq("").any(axis=1)].copy()
    if df.empty:
        return df

    df[RESIDUE_ID_COLS] = df[RESIDUE_ID_COLS].fillna(0).astype(np.int32)
    return df


def attach_centroids(df, atom_records):
    coords_arrays = []
    serial_arrays = []
    for col in ID_COLS:
        records = df[col].map(atom_records)
        coord_array = np.full((len(df), 3), np.nan, dtype=np.float64)
        serial_array = np.zeros((len(df),), dtype=np.int32)
        valid_idx = records.notna().to_numpy()
        if valid_idx.any():
            record_values = records.loc[valid_idx].tolist()
            serial_array[valid_idx] = np.asarray([item[0] for item in record_values], dtype=np.int32)
            coord_array[valid_idx] = np.asarray([item[1:] for item in record_values], dtype=np.float64)
        coords_arrays.append(coord_array)
        serial_arrays.append(serial_array)

    valid_mask = np.ones(len(df), dtype=bool)
    for coord_array in coords_arrays:
        valid_mask &= ~np.isnan(coord_array).any(axis=1)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return df

    valid_positions = valid_mask.nonzero()[0]
    coords_sum = coords_arrays[0][valid_positions].copy()
    for coord_array in coords_arrays[1:]:
        coords_sum += coord_array[valid_positions]

    centroids = coords_sum / len(ID_COLS)
    df["x"] = pdb_formatted_float_array(centroids[:, 0], 3)
    df["y"] = pdb_formatted_float_array(centroids[:, 1], 3)
    df["z"] = pdb_formatted_float_array(centroids[:, 2], 3)
    for idx, serial_array in enumerate(serial_arrays, start=1):
        serial_col = ATOM_SERIAL_COLS[idx - 1]
        if serial_col in df.columns:
            df[f"atom_{idx}"] = pd.to_numeric(df[serial_col], errors="coerce").fillna(0).astype(np.int32)
        else:
            df[f"atom_{idx}"] = serial_array[valid_positions]
    return df.reset_index(drop=True)


def normalize_digit_text(value):
    text = str(value).strip()
    try:
        return str(int(float(text)))
    except (TypeError, ValueError):
        pass
    digits = "".join(re.findall(r"\d+", text))
    return str(int(digits)) if digits else ""


def pdb_formatted_float_array(values, precision):
    fmt = f"{{:.{precision}f}}"
    return np.asarray([float(fmt.format(float(value))) for value in values], dtype=np.float64)


def prepare_all_points_dataframe(atom_network_csv, pdb_path):
    atom_records = load_atom_records(pdb_path)
    df = load_filtered_dataframe(atom_network_csv)
    if df.empty:
        return df
    df = attach_centroids(df, atom_records)
    if df.empty:
        return df

    df["_tunnel_id"] = df["Tunnel_ID"].map(normalize_digit_text)
    df["_frame_id"] = df["pdb_id"].map(normalize_digit_text)
    df = df.loc[(df["_tunnel_id"] != "") & (df["_frame_id"] != "")].copy()
    if df.empty:
        return df

    df["_tunnel_sort"] = df["_tunnel_id"].astype(np.int64)
    df["_frame_sort"] = df["_frame_id"].astype(np.int64)
    df["bfactor"] = pdb_formatted_float_array(pd.to_numeric(df["T_F_T_radius"], errors="coerce").fillna(0.0), 2)
    df["x_origin"] = pdb_formatted_float_array(pd.to_numeric(df["T_F_T_x"], errors="coerce").fillna(0.0), 3)
    df["y_origin"] = pdb_formatted_float_array(pd.to_numeric(df["T_F_T_y"], errors="coerce").fillna(0.0), 3)
    df["z_origin"] = pdb_formatted_float_array(pd.to_numeric(df["T_F_T_z"], errors="coerce").fillna(0.0), 3)
    df["length"] = pdb_formatted_float_array(pd.to_numeric(df["T_F_T_length"], errors="coerce").fillna(0.0), 3)
    df["throughput"] = pdb_formatted_float_array(pd.to_numeric(df.get("Throughput"), errors="coerce").fillna(0.0), 3)
    df["tunnel_cluster"] = pd.to_numeric(df.get("Tunnel_cluster"), errors="coerce").fillna(0).astype(np.int32)
    for idx, col in enumerate(RESIDUE_ID_COLS, start=1):
        df[f"residue_{idx}"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(np.int32)
    for idx, col in enumerate(RESIDUE_QSS_COLS, start=1):
        df[f"qss{idx}"] = pdb_formatted_float_array(pd.to_numeric(df[col], errors="coerce").fillna(0.0), 3)
    return df.reset_index(drop=True)


def build_synthetic_merged_files(df):
    groups_by_tunnel = defaultdict(list)
    for (tunnel_id, frame_id), group in df.groupby(["_tunnel_sort", "_frame_sort"], sort=False):
        groups_by_tunnel[int(tunnel_id)].append((int(frame_id), group.index.to_numpy(dtype=np.int64)))

    synthetic_files = []
    for tunnel_id in sorted(groups_by_tunnel):
        part_idx = 1
        current_offset = 0
        frames = []
        for frame_id, indices in sorted(groups_by_tunnel[tunnel_id], key=lambda item: item[0]):
            point_count = len(indices)
            if point_count > MAX_SERIAL:
                print(f"[WARN] Skipped tunnel {tunnel_id} frame {frame_id}: {point_count} points exceed PDB serial limit")
                continue
            if frames and current_offset + point_count > MAX_SERIAL:
                synthetic_files.append({
                    "source_file": f"tunnel_{tunnel_id}_merged_part{part_idx}.pdb",
                    "frames": frames,
                })
                part_idx += 1
                current_offset = 0
                frames = []
            frames.append({
                "frame_id": frame_id,
                "indices": indices,
                "atom_offset": current_offset,
            })
            current_offset += point_count
        if frames:
            synthetic_files.append({
                "source_file": f"tunnel_{tunnel_id}_merged_part{part_idx}.pdb",
                "frames": frames,
            })

    synthetic_files.sort(key=lambda item: item["source_file"])
    return synthetic_files


def write_all_points_csv(df, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic_files = build_synthetic_merged_files(df)

    arrays = {col: df[col].to_numpy() for col in [
        "x", "y", "z", "bfactor", "x_origin", "y_origin", "z_origin",
        "length", "throughput", "tunnel_cluster", "qss1", "qss2", "qss3", "qss4",
        "atom_1", "atom_2", "atom_3", "atom_4",
        "residue_1", "residue_2", "residue_3", "residue_4",
    ]}

    row_count = 0
    path_counter = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTPUT_COLUMNS)
        for file_info in tqdm(synthetic_files, desc="Writing fast all_points", unit="file"):
            source_file = file_info["source_file"]
            for frame_info in file_info["frames"]:
                indices = frame_info["indices"]
                atom_offset = int(frame_info["atom_offset"])
                frame_id = int(frame_info["frame_id"])
                path_counter += 1
                for point_idx, row_idx in enumerate(indices, start=1):
                    writer.writerow([
                        source_file,
                        atom_offset + point_idx,
                        point_idx,
                        arrays["x"][row_idx],
                        arrays["y"][row_idx],
                        arrays["z"][row_idx],
                        arrays["bfactor"][row_idx],
                        path_counter,
                        "C",
                        int(arrays["residue_1"][row_idx]),
                        int(arrays["residue_2"][row_idx]),
                        int(arrays["residue_3"][row_idx]),
                        int(arrays["residue_4"][row_idx]),
                        arrays["x_origin"][row_idx],
                        arrays["y_origin"][row_idx],
                        arrays["z_origin"][row_idx],
                        frame_id,
                        arrays["length"][row_idx],
                        arrays["throughput"][row_idx],
                        int(arrays["tunnel_cluster"][row_idx]),
                        arrays["qss1"][row_idx],
                        arrays["qss2"][row_idx],
                        arrays["qss3"][row_idx],
                        arrays["qss4"][row_idx],
                        int(arrays["atom_1"][row_idx]),
                        int(arrays["atom_2"][row_idx]),
                        int(arrays["atom_3"][row_idx]),
                        int(arrays["atom_4"][row_idx]),
                    ])
                    row_count += 1
    return row_count, path_counter, len(synthetic_files)


def convert_atom_network_to_all_points(atom_network_csv, pdb_path, output_csv):
    start = time.time()
    print("=" * 60)
    print("FAST 03 -> all_points")
    print("=" * 60)
    print(f"atom_network_csv: {atom_network_csv}")
    print(f"representative PDB: {pdb_path}")
    print(f"output CSV: {output_csv}")

    df = prepare_all_points_dataframe(atom_network_csv, pdb_path)
    if df.empty:
        raise ValueError("No valid tunnel points after filtering atom_network CSV")
    row_count, path_count, file_count = write_all_points_csv(df, output_csv)

    elapsed = time.time() - start
    print(f"[OK] Wrote {output_csv}")
    print(f"Rows: {row_count:,}, paths: {path_count:,}, synthetic merged files: {file_count:,}")
    print(f"Elapsed: {elapsed:.2f}s")
    return row_count


def parse_args():
    parser = argparse.ArgumentParser(description="Fast convert 03 atom network CSV to 07 all_points CSV")
    parser.add_argument("--atom-network-csv", default=four_atoms_csv)
    parser.add_argument("--pdb", default=PDB_path)
    parser.add_argument("--out-csv", default=all_points_csv)
    parser.add_argument(
        "--mutations",
        default=None,
        help="comma-separated mutation specs like L176W,F142L; overrides pipeline_config mutations for this run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_runtime_mutations(args.mutations)
    convert_atom_network_to_all_points(
        atom_network_csv=args.atom_network_csv,
        pdb_path=args.pdb,
        output_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
