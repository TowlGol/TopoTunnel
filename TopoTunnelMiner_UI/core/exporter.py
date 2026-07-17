"""
Core business logic: path export to PDB format.
Reuses the same output format as 5.draw_Path_backend.py generate_pdb_content().
"""
import os
from collections import OrderedDict
from typing import List, Optional, Tuple


_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
_PATH_SEPARATOR_BLANK_LINES = 3


def _chain_id_for_index(index: int) -> str:
    if index < len(_CHAIN_IDS):
        return _CHAIN_IDS[index]
    return _CHAIN_IDS[index % len(_CHAIN_IDS)]


def _format_atom_record(
    serial: int,
    chain_id: str,
    res_seq: int,
    x: float,
    y: float,
    z: float,
    occ: float,
    b_factor: float,
    element: str = "C",
) -> str:
    return (
        f"ATOM  {serial:5d} CG   CEN {chain_id}{res_seq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occ:6.2f}{b_factor:6.2f}           {element:>2s}"
    )


def _build_export_filename(
    category_name: str,
    frame_id: int,
    split_index: Optional[int] = None,
) -> str:
    stem = f"{category_name}_{frame_id}_centroids"
    if split_index is not None:
        return f"{stem}_{split_index}.pdb"
    return f"{stem}.pdb"


def _build_metadata_summary_filename(category_name: str) -> str:
    return f"{category_name}_centroids.txt"


def _path_throughput(points: list) -> float:
    for pt in points:
        if "throughput" in pt and pt["throughput"] is not None:
            return float(pt["throughput"])
    return 0.0


def _write_split_metadata_summary(filepath: str, entries: list[dict]):
    sorted_entries = sorted(
        entries,
        key=lambda entry: float(entry["throughput"]),
        reverse=True,
    )
    lines = ["id,throughput,frame_id,path_id"]
    for idx, entry in enumerate(sorted_entries, start=1):
        lines.append(
            f"{idx},{entry['throughput']:g},{entry['frame_id']},{entry['path_id']}"
        )
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_ranked_exports(
    export_dir: str,
    folder_name: str,
    entries: list[dict],
    reverse: bool,
    limit: int = 100,
):
    target_dir = os.path.join(export_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    ranked = sorted(
        entries,
        key=lambda entry: (float(entry["throughput"]), int(entry["frame_id"]), int(entry["path_id"])),
        reverse=reverse,
    )[:limit]
    for index, entry in enumerate(ranked, start=1):
        filepath = os.path.join(target_dir, f"{index}.pdb")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(entry["content"])


def _build_path_records(
    points: list,
    chain_id: str,
    atom_serial_start: int,
) -> Tuple[list[str], int]:
    lines: list[str] = []

    for i, pt in enumerate(points):
        serial = atom_serial_start + i
        res_seq = i + 1
        x_orig = pt.get("x_origin", pt["x"])
        y_orig = pt.get("y_origin", pt["y"])
        z_orig = pt.get("z_origin", pt["z"])
        b_factor = pt.get("radius", 1.0)
        occ = 1.0
        lines.append(
            _format_atom_record(
                serial=serial,
                chain_id=chain_id,
                res_seq=res_seq,
                x=x_orig,
                y=y_orig,
                z=z_orig,
                occ=occ,
                b_factor=b_factor,
            )
        )
        if i < len(points) - 1:
            lines.append(f"CONECT{serial:5d}{serial + 1:5d}")

    return lines, atom_serial_start + len(points)


def generate_pdb_content(points: list, cat_name: str, frame_id: int) -> str:
    """Generate PDB format content from path points."""
    path_lines, _ = _build_path_records(points, "Z", 1)
    return ("\n".join(path_lines) + "\n") if path_lines else ""


def generate_grouped_pdb_content(
    frame_paths: List[Tuple[int, list]],
    cat_name: str,
    frame_id: int,
    separator_blank_lines: int = _PATH_SEPARATOR_BLANK_LINES,
) -> str:
    """Generate one PDB per frame with explicit path boundaries."""
    if len(frame_paths) == 1:
        return generate_pdb_content(frame_paths[0][1], cat_name, frame_id)

    lines: list[str] = []

    atom_serial = 1
    for path_index, (_, points) in enumerate(frame_paths, start=1):
        if path_index > 1 and separator_blank_lines > 0:
            lines.extend([""] * separator_blank_lines)
        chain_id = _chain_id_for_index(path_index - 1)
        path_lines, atom_serial = _build_path_records(points, chain_id, atom_serial)
        lines.extend(path_lines)

    return ("\n".join(lines) + "\n") if lines else ""


def export_paths_to_pdb(
    db,
    path_ids: List[int],
    category_name: str,
    export_dir: str,
    split_frame_paths: bool = False,
) -> List[str]:
    """Export one PDB per frame, grouping same-frame paths with clear separators."""
    os.makedirs(export_dir, exist_ok=True)
    grouped: OrderedDict[int, List[Tuple[int, list]]] = OrderedDict()

    for pid in path_ids:
        data = db.get_path_for_export(pid)
        if not data:
            continue
        meta = data["meta"]
        points = data["points"]
        frame_id = meta["frame_id"]
        grouped.setdefault(frame_id, []).append((pid, points))

    exported = []
    metadata_entries: list[dict] = []
    for frame_id, frame_paths in grouped.items():
        if split_frame_paths:
            for split_index, (path_id, points) in enumerate(frame_paths, start=1):
                filename = _build_export_filename(category_name, frame_id, split_index)
                filepath = os.path.join(export_dir, filename)
                content = generate_pdb_content(points, category_name, frame_id)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                metadata_entries.append(
                    {
                        "path_id": path_id,
                        "throughput": _path_throughput(points),
                        "frame_id": frame_id,
                        "content": content,
                    }
                )
                exported.append(filename)
            continue

        filename = _build_export_filename(category_name, frame_id)
        filepath = os.path.join(export_dir, filename)
        content = generate_grouped_pdb_content(frame_paths, category_name, frame_id)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        exported.append(filename)

    if split_frame_paths and metadata_entries:
        metadata_path = os.path.join(
            export_dir,
            _build_metadata_summary_filename(category_name),
        )
        _write_split_metadata_summary(metadata_path, metadata_entries)
        _write_ranked_exports(export_dir, "top100", metadata_entries, reverse=True, limit=100)
        _write_ranked_exports(export_dir, "end100", metadata_entries, reverse=False, limit=100)

    return exported
