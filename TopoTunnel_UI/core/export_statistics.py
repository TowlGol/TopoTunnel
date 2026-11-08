"""Summarize stored path data and existing distance arrays for UI exports.

Pure statistics helpers adapted from the stage-03 CSV format. No pipeline
configuration, topology mapping, structure processing or stage execution.
"""
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats

RESIDUE_DISTANCE_PRECISION = 6

def ensure_output_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

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

def write_pair_path_presence_statistics(output_path, pair_items, pair_distance_values, frame_ids, presence_accumulator):
    ensure_output_parent(output_path)
    presence = np.asarray([
        [pair in presence_accumulator.get(str(frame), set()) for frame in frame_ids]
        for pair, _ in pair_items
    ], dtype=bool).reshape(pair_distance_values.shape)
    marks = int(np.count_nonzero(presence))

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
