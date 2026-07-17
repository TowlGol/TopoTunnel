"""
Helpers for comparing residue statistics CSV outputs.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional


IDENTITY_COLUMNS = {"Residue_ID", "Residue_name"}

METRIC_LABELS = {
    "Affected_point_count": "Affected Points",
    "Affected_path_count": "Affected Paths",
    "Affected_point_radius_avg": "Mean Radius",
    "Affected_point_radius_min": "Min Radius",
    "Affected_point_radius_max": "Max Radius",
    "Cooperative_residue_count": "Cooperative Residues",
}

METRIC_TOOLTIPS = {
    "Affected_point_count": "Difference in affected point count",
    "Affected_path_count": "Difference in affected path count",
    "Affected_point_radius_avg": "Difference in mean affected-point radius",
    "Affected_point_radius_min": "Difference in minimum affected-point radius",
    "Affected_point_radius_max": "Difference in maximum affected-point radius",
    "Cooperative_residue_count": "Difference in cooperative residue count",
}

PRESENCE_STATUS_LABELS = {
    "both": "Present in Both",
    "left_only": "File A Only",
    "right_only": "File B Only",
}


def _safe_float(value) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_residue_id(raw_value) -> int:
    value = _safe_float(raw_value)
    if value is None:
        raise ValueError("Residue_ID is missing or invalid")
    return int(value)


def _looks_numeric_column(rows: List[dict], column: str) -> bool:
    seen_numeric = False
    for row in rows:
        raw_value = str(row.get(column, "") or "").strip()
        if not raw_value:
            continue
        if ";" in raw_value or ":" in raw_value:
            return False
        if _safe_float(raw_value) is None:
            return False
        seen_numeric = True
    return seen_numeric


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def metric_tooltip(metric: str) -> str:
    return METRIC_TOOLTIPS.get(metric, metric_label(metric))


def classify_presence(present_left: bool, present_right: bool) -> str:
    if present_left and present_right:
        return "both"
    if present_left:
        return "left_only"
    return "right_only"


def presence_status_label(status: str) -> str:
    return PRESENCE_STATUS_LABELS.get(status, status)


def presence_status_tooltip(status: str, left_label: str = "File A", right_label: str = "File B") -> str:
    if status == "both":
        return f"This residue is present in both {left_label} and {right_label}, so their values can be compared directly."
    if status == "left_only":
        return f"This residue is present only in {left_label}; missing values in {right_label} are treated as 0 in delta calculations."
    if status == "right_only":
        return f"This residue is present only in {right_label}; missing values in {left_label} are treated as 0 in delta calculations."
    return status


def parse_cooperative_residue_details(raw_value) -> Dict[int, dict]:
    text = str(raw_value or "").strip()
    if not text:
        return {}

    partners: Dict[int, dict] = {}
    for chunk in text.split(";"):
        item = chunk.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 3:
            continue
        partner_id = _parse_residue_id(parts[0])
        partner_name = str(parts[1] or "").strip()
        shared_count = int(_safe_float(parts[2]) or 0)
        partners[partner_id] = {
            "residue_id": partner_id,
            "residue_name": partner_name,
            "shared_count": shared_count,
        }
    return partners


def compare_cooperative_partner_details(left_details: Dict[int, dict], right_details: Dict[int, dict]) -> List[dict]:
    partner_ids = sorted(set(left_details.keys()) | set(right_details.keys()))
    rows: List[dict] = []
    for partner_id in partner_ids:
        left_row = left_details.get(partner_id)
        right_row = right_details.get(partner_id)
        left_count = int(left_row.get("shared_count", 0)) if left_row else 0
        right_count = int(right_row.get("shared_count", 0)) if right_row else 0
        delta = right_count - left_count
        if delta == 0:
            continue
        rows.append(
            {
                "residue_id": partner_id,
                "left_name": str(left_row.get("residue_name", "")) if left_row else "",
                "right_name": str(right_row.get("residue_name", "")) if right_row else "",
                "left_count": left_count,
                "right_count": right_count,
                "delta": delta,
                "abs_delta": abs(delta),
            }
        )
    rows.sort(key=lambda item: (-int(item["abs_delta"]), int(item["residue_id"])))
    return rows


def load_residue_statistics_csv(path: str) -> dict:
    abs_path = os.path.abspath(path)
    with open(abs_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV is missing a header row")
    if "Residue_ID" not in fieldnames:
        raise ValueError("CSV is missing required column: Residue_ID")

    numeric_columns = [
        column
        for column in fieldnames
        if column not in IDENTITY_COLUMNS and _looks_numeric_column(rows, column)
    ]

    residues: Dict[int, dict] = {}
    for row in rows:
        residue_id = _parse_residue_id(row.get("Residue_ID"))
        residue_name = str(row.get("Residue_name", "") or "").strip()
        metrics = {
            metric: float(_safe_float(row.get(metric)) or 0.0)
            for metric in numeric_columns
        }
        cooperative_details = parse_cooperative_residue_details(row.get("Cooperative_residue_details"))
        residues[residue_id] = {
            "residue_id": residue_id,
            "residue_name": residue_name,
            "metrics": metrics,
            "cooperative_details": cooperative_details,
        }

    return {
        "path": abs_path,
        "fieldnames": fieldnames,
        "metrics": numeric_columns,
        "residues": residues,
        "row_count": len(residues),
    }


def _merge_metric_names(left_metrics: List[str], right_metrics: List[str]) -> List[str]:
    ordered: List[str] = []
    for metric in list(left_metrics) + list(right_metrics):
        if metric not in ordered:
            ordered.append(metric)
    return ordered


def _build_residue_label(residue_id: int, left_name: str, right_name: str) -> str:
    if left_name and right_name and left_name != right_name:
        return f"{residue_id} {left_name}->{right_name}"
    residue_name = right_name or left_name
    return f"{residue_id} {residue_name}".strip()


def compare_residue_statistics(left: dict, right: dict) -> dict:
    metrics = _merge_metric_names(left.get("metrics", []), right.get("metrics", []))
    residue_ids = sorted(set(left.get("residues", {}).keys()) | set(right.get("residues", {}).keys()))

    metric_ranges = {metric: 0.0 for metric in metrics}
    rows = []

    for residue_id in residue_ids:
        left_row = left.get("residues", {}).get(residue_id)
        right_row = right.get("residues", {}).get(residue_id)
        left_name = left_row["residue_name"] if left_row else ""
        right_name = right_row["residue_name"] if right_row else ""
        present_left = left_row is not None
        present_right = right_row is not None
        left_cooperative_details = left_row.get("cooperative_details", {}) if left_row else {}
        right_cooperative_details = right_row.get("cooperative_details", {}) if right_row else {}
        cooperative_partner_changes = compare_cooperative_partner_details(
            left_cooperative_details,
            right_cooperative_details,
        )

        cells = {}
        max_abs_delta = 0.0
        changed = False

        for metric in metrics:
            left_value = left_row["metrics"].get(metric, 0.0) if left_row else 0.0
            right_value = right_row["metrics"].get(metric, 0.0) if right_row else 0.0
            delta = float(right_value - left_value)
            abs_delta = abs(delta)
            baseline = abs(left_value)
            ratio = None if baseline <= 1e-12 else (delta / baseline)
            if abs_delta > 1e-12:
                changed = True
            max_abs_delta = max(max_abs_delta, abs_delta)
            metric_ranges[metric] = max(metric_ranges[metric], abs_delta)
            cells[metric] = {
                "left": float(left_value),
                "right": float(right_value),
                "delta": delta,
                "abs_delta": abs_delta,
                "ratio": ratio,
            }

        if cooperative_partner_changes:
            changed = True

        rows.append(
            {
                "residue_id": residue_id,
                "residue_label": _build_residue_label(residue_id, left_name, right_name),
                "left_name": left_name,
                "right_name": right_name,
                "present_left": present_left,
                "present_right": present_right,
                "presence_status": classify_presence(present_left, present_right),
                "cooperative_partner_changes": cooperative_partner_changes,
                "changed": changed,
                "max_abs_delta": max_abs_delta,
                "cells": cells,
            }
        )

    changed_rows = sum(1 for row in rows if row["changed"])
    return {
        "left_path": left.get("path", ""),
        "right_path": right.get("path", ""),
        "metrics": metrics,
        "metric_ranges": metric_ranges,
        "rows": rows,
        "row_count": len(rows),
        "changed_row_count": changed_rows,
    }


def order_comparison_rows(comparison: dict, sort_mode: str = "max_abs_delta", changed_only: bool = False) -> List[dict]:
    rows = list(comparison.get("rows", []))
    if changed_only:
        rows = [row for row in rows if row.get("changed")]

    if sort_mode == "residue_id":
        return sorted(rows, key=lambda row: (int(row["residue_id"]), row["residue_label"]))
    if sort_mode == "residue_name":
        return sorted(rows, key=lambda row: (row["residue_label"], int(row["residue_id"])))
    return sorted(
        rows,
        key=lambda row: (-float(row["max_abs_delta"]), int(row["residue_id"])),
    )
