#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 这个是我们研究出来的方法，主要分为三部分进行走。首先将pdb分为n个部分，分别进行运行。获得路径隧道等信息。然后在隧道基础上对于每一个单独的隧道进行计算并生成单独的文件。最后根据Tunnel的情况合并profile.最终获得相关的内容。他有两个好处：首先它的运行速度会变的更快，其次他它可以获取到更多的路径节点。

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

_PRINT_LOCK = threading.Lock()



def extract_cluster_serial(name: str):
    """
    从文件名中提取排序序号，例如:
      tun_cl_001_1.pdb -> (1, 1)
      tun_cl_012_3.pdb -> (12, 3)
    """
    m = re.match(r"tun_cl_(\d+)_(\d+)\.pdb$", name, flags=re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (10**9, name)


def collect_cluster_timeless_pdbs(stage2_out_root: Path):
    """
    收集所有:
      stage2_single_outputs/*_out/data/clusters_timeless/*.pdb

    返回:
      {
        "tun_cl_001_1.pdb": [Path(...), Path(...), ...],
        ...
      }
    """
    grouped = defaultdict(list)

    for out_dir in sorted(stage2_out_root.glob("*_out")):
        cluster_dir = out_dir / "data" / "clusters_timeless"
        if not cluster_dir.exists():
            continue

        for pdb_file in sorted(cluster_dir.glob("*.pdb"), key=lambda p: extract_cluster_serial(p.name)):
            grouped[pdb_file.name].append(pdb_file)

    return grouped


def _stage4_manifest_path(output_file: Path) -> Path:
    return output_file.with_name(f".{output_file.name}.stage4.json")


def _stage4_manifest(input_files: list[Path], include_conect: bool):
    inputs = []
    for pdb_path in input_files:
        stat = pdb_path.stat()
        inputs.append({
            "path": str(pdb_path),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return {
        "version": 1,
        "include_conect": bool(include_conect),
        "inputs": inputs,
    }


def _is_output_current(output_file: Path, input_files: list[Path], include_conect: bool) -> bool:
    if not output_file.exists():
        return False
    try:
        out_stat = output_file.stat()
        if out_stat.st_size <= 0:
            return False
        manifest_file = _stage4_manifest_path(output_file)
        if not manifest_file.exists():
            return False
        old_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        return old_manifest == _stage4_manifest(input_files, include_conect)
    except (OSError, json.JSONDecodeError):
        return False


def merge_same_named_cluster_pdbs(
    input_files: list[Path],
    output_file: Path,
    include_conect: bool = True,
    incremental: bool = True,
):
    """
    合并多个同名 cluster pdb：
    - 不写 MODEL / ENDMDL
    - 直接顺次写 ATOM / HETATM / CONECT
    - 重新编号 ATOM serial
    - 同一个来源文件内部的 CONECT 保留并映射到新的 serial
    - 不在两个来源文件之间额外写 CONECT
    """
    if incremental and _is_output_current(output_file, input_files, include_conect):
        return {"status": "skipped", "atoms": 0, "inputs": len(input_files)}

    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = output_file.with_name(f".{output_file.name}.{threading.get_ident()}.tmp")
    next_serial = 1
    atom_count = 0

    try:
        with tmp_file.open("w", encoding="utf-8", newline="\n") as fout:
            for pdb_path in input_files:
                serial_map = {} if include_conect else None
                conect_lines = [] if include_conect else None

                with pdb_path.open("r", encoding="utf-8", errors="replace") as fin:
                    for raw_line in fin:
                        line = raw_line.rstrip("\r\n")
                        rec = line[:6]

                        if rec == "ATOM  " or rec == "HETATM":
                            if include_conect:
                                try:
                                    old_serial = int(line[6:11].strip())
                                except ValueError:
                                    continue

                            new_serial = next_serial
                            next_serial += 1
                            atom_count += 1
                            if include_conect:
                                serial_map[old_serial] = new_serial

                            fout.write(f"{line[:6]}{new_serial:5d}{line[11:]}\n")

                        elif include_conect and rec == "CONECT":
                            conect_lines.append(line)

                if include_conect:
                    for line in conect_lines:
                        parts = line.split()
                        if len(parts) < 3:
                            continue

                        try:
                            old_src = int(parts[1])
                        except ValueError:
                            continue

                        if old_src not in serial_map:
                            continue

                        new_src = serial_map[old_src]
                        new_targets = []

                        for x in parts[2:]:
                            try:
                                old_tgt = int(x)
                            except ValueError:
                                continue
                            if old_tgt in serial_map:
                                new_targets.append(serial_map[old_tgt])

                        if new_targets:
                            fout.write("CONECT" + f"{new_src:5d}" + "".join(f"{t:5d}" for t in new_targets) + "\n")

            fout.write("END\n")

        tmp_file.replace(output_file)
        _stage4_manifest_path(output_file).write_text(
            json.dumps(_stage4_manifest(input_files, include_conect), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    return {"status": "merged", "atoms": atom_count, "inputs": len(input_files)}


def merge_clusters_timeless_from_stage2(
    stage2_out_root: Path,
    merged_root: Path,
    workers: int = 1,
    incremental: bool = True,
    include_conect: bool = True,
):
    """
    把所有 stage2 输出目录中的 clusters_timeless 同名 pdb 合并到 merged_root
    """
    grouped = collect_cluster_timeless_pdbs(stage2_out_root)

    if not grouped:
        safe_print("[WARN] No clusters_timeless pdb files found.")
        return False

    merged_root.mkdir(parents=True, exist_ok=True)

    names = sorted(grouped.keys(), key=extract_cluster_serial)
    workers = max(1, min(int(workers or 1), len(names)))
    safe_print(
        f"[INFO] Found {len(names)} cluster pdb names to merge. "
        f"workers={workers}, incremental={'yes' if incremental else 'no'}, "
        f"conect={'yes' if include_conect else 'no'}"
    )

    done = 0
    merged = 0
    skipped = 0
    total_atoms = 0
    log_every = max(1, min(100, len(names) // 20 or 1))

    def merge_one(name: str):
        files = grouped[name]
        out_file = merged_root / name
        result = merge_same_named_cluster_pdbs(
            files,
            out_file,
            include_conect=include_conect,
            incremental=incremental,
        )
        return name, result

    if workers == 1:
        iterator = (merge_one(name) for name in names)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(merge_one, name) for name in names]
        iterator = (future.result() for future in as_completed(futures))

    try:
        for name, result in iterator:
            done += 1
            if result["status"] == "skipped":
                skipped += 1
            else:
                merged += 1
                total_atoms += result["atoms"]

            if done == 1 or done == len(names) or done % log_every == 0:
                safe_print(
                    f"[STAGE4] {done}/{len(names)} done, "
                    f"merged={merged}, skipped={skipped}, last={name}"
                )
    finally:
        if workers != 1:
            executor.shutdown(wait=True, cancel_futures=False)

    safe_print(
        f"[DONE] clusters_timeless merged into: {merged_root} "
        f"(merged={merged}, skipped={skipped}, atoms_written={total_atoms})"
    )
    return True


def safe_print(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def extract_frame_number(pdb_file: Path):
    m = re.search(r"(\d+)(?=\.pdb$)", pdb_file.name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def split_into_n_parts(sorted_seq_indices, n_parts: int):
    total = len(sorted_seq_indices)
    if n_parts < 1:
        raise ValueError("n_parts must be >= 1")
    if n_parts > total:
        n_parts = total

    base = total // n_parts
    rem = total % n_parts

    parts = []
    start = 0
    for i in range(n_parts):
        size = base + (1 if i < rem else 0)
        end = start + size
        sub = sorted_seq_indices[start:end]
        parts.append({
            "part_id": i + 1,
            "part_name": f"part_{i+1:04d}",
            "seq_indices": sub,
            "first_frame": min(sub),
            "last_frame": max(sub),
        })
        start = end
    return parts


def rewrite_config(
    original_conf: Path,
    output_conf: Path,
    first_frame: int,
    last_frame: int,
    stop_after_value: str,
    load_tunnels_value: str | None = None,
    load_cluster_tree_value: str | None = None,
    swap_value: str | None = None,
):
    with original_conf.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    found_first = False
    found_last = False
    found_stop = False
    found_load_tunnels = False
    found_load_cluster_tree = False
    found_swap = False

    out_lines = []

    for line in lines:
        if re.match(r"^\s*first_frame\b", line):
            out_lines.append(f"first_frame {first_frame}\n")
            found_first = True
        elif re.match(r"^\s*last_frame\b", line):
            out_lines.append(f"last_frame {last_frame}\n")
            found_last = True
        elif re.match(r"^\s*stop_after\b", line):
            out_lines.append(f"stop_after {stop_after_value}\n")
            found_stop = True
        elif load_tunnels_value is not None and re.match(r"^\s*load_tunnels\b", line):
            out_lines.append(f"load_tunnels {load_tunnels_value}\n")
            found_load_tunnels = True
        elif load_cluster_tree_value is not None and re.match(r"^\s*load_cluster_tree\b", line):
            out_lines.append(f"load_cluster_tree {load_cluster_tree_value}\n")
            found_load_cluster_tree = True
        elif swap_value is not None and re.match(r"^\s*swap\b", line):
            out_lines.append(f"swap {swap_value}\n")
            found_swap = True
        else:
            out_lines.append(line)

    if not found_first:
        out_lines.append(f"\nfirst_frame {first_frame}\n")
    if not found_last:
        out_lines.append(f"last_frame {last_frame}\n")
    if not found_stop:
        out_lines.append(f"stop_after {stop_after_value}\n")
    if load_tunnels_value is not None and not found_load_tunnels:
        out_lines.append(f"load_tunnels {load_tunnels_value}\n")
    if load_cluster_tree_value is not None and not found_load_cluster_tree:
        out_lines.append(f"load_cluster_tree {load_cluster_tree_value}\n")
    if swap_value is not None and not found_swap:
        out_lines.append(f"swap {swap_value}\n")

    output_conf.parent.mkdir(parents=True, exist_ok=True)
    with output_conf.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


def run_one_job(
    job_name: str,
    input_dir: Path,
    conf_file: Path,
    out_dir: Path,
    java: str,
    xmx: str,
    lib_dir: Path,
    caver_jar: Path,
    home_dir: Path,
    dry_run: bool,
    overwrite: bool,
    java_active_processor_count: int | None,
    java_parallel_gc_threads: int | None,
    java_conc_gc_threads: int | None,
    complete_markers: tuple[str, ...] | None = None,
):
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        if complete_markers is None:
            complete = True
        else:
            complete = any((out_dir / marker).exists() for marker in complete_markers)

        if complete:
            safe_print(f"[SKIP] {job_name} (complete output exists: {out_dir})")
            return {
                "job_name": job_name,
                "returncode": 0,
                "elapsed_sec": 0.0,
                "status": "skipped",
                "out_dir": str(out_dir),
            }

        safe_print(f"[RERUN] {job_name} (incomplete output exists: {out_dir})")

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [java, f"-Xmx{xmx}"]

    if java_active_processor_count is not None:
        cmd.append(f"-XX:ActiveProcessorCount={java_active_processor_count}")
    if java_parallel_gc_threads is not None:
        cmd.append(f"-XX:ParallelGCThreads={java_parallel_gc_threads}")
    if java_conc_gc_threads is not None:
        cmd.append(f"-XX:ConcGCThreads={java_conc_gc_threads}")

    cmd.extend([
        "-cp",
        str(lib_dir),
        "-jar",
        str(caver_jar),
        "-home",
        str(home_dir),
        "-pdb",
        str(input_dir),
        "-conf",
        str(conf_file),
        "-out",
        str(out_dir),
    ])

    log_path = out_dir / "run.log"

    safe_print(f"[RUN] {job_name}")
    safe_print("      " + " ".join(cmd))

    if dry_run:
        return {
            "job_name": job_name,
            "returncode": 0,
            "elapsed_sec": 0.0,
            "status": "dry-run",
            "out_dir": str(out_dir),
        }

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    status = "ok" if p.returncode == 0 else "error"
    if p.returncode == 0:
        safe_print(f"[OK ] {job_name} ({dt:.1f}s) -> {out_dir}")
    else:
        safe_print(f"[ERR] {job_name} returncode={p.returncode} (see {log_path})")

    return {
        "job_name": job_name,
        "returncode": p.returncode,
        "elapsed_sec": dt,
        "status": status,
        "out_dir": str(out_dir),
    }


def run_jobs_parallel(
    jobs,
    input_dir,
    java,
    xmx,
    lib_dir,
    caver_jar,
    home_dir,
    dry_run,
    overwrite,
    jobs_n,
    java_active_processor_count,
    java_parallel_gc_threads,
    java_conc_gc_threads,
    complete_markers=None,
):
    rc_any = 0
    done = 0
    total = len(jobs)
    results = []

    with ThreadPoolExecutor(max_workers=jobs_n) as ex:
        futures = {
            ex.submit(
                run_one_job,
                rec["job_name"],
                input_dir,
                rec["conf_file"],
                rec["out_dir"],
                java,
                xmx,
                lib_dir,
                caver_jar,
                home_dir,
                dry_run,
                overwrite,
                java_active_processor_count,
                java_parallel_gc_threads,
                java_conc_gc_threads,
                complete_markers,
            ): rec
            for rec in jobs
        }

        for fut in as_completed(futures):
            rec = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "job_name": rec["job_name"],
                    "returncode": 1,
                    "elapsed_sec": 0.0,
                    "status": f"exception: {e!r}",
                    "out_dir": str(rec["out_dir"]),
                }
                safe_print(f"[EXC] {rec['job_name']}: {e!r}")

            done += 1
            safe_print(f"[PROG] {done}/{total} finished")

            if result["returncode"] != 0:
                rc_any = rc_any or result["returncode"]

            merged = dict(rec)
            merged.update(result)
            results.append(merged)

    return rc_any, results


def build_stage1_file_index(stage1_out_root: Path):
    index = {}
    for path in stage1_out_root.rglob("*"):
        if not path.is_file():
            continue
        subdir = path.parent.name
        key = (subdir, path.name)
        index.setdefault(key, []).append(path)
    return index


def summarize_failed_jobs(results):
    failed = [rec for rec in results if int(rec.get("returncode", 0)) != 0]
    if not failed:
        return ""
    pieces = []
    for rec in failed:
        job_name = rec.get("job_name", "unknown")
        log_path = Path(rec.get("out_dir", "")) / "run.log"
        pieces.append(f"{job_name} (log: {log_path})")
    return ", ".join(pieces)


def find_stage1_file_across_parts(stage1_out_root: Path, subdir: str, filename: str, file_index=None) -> Path:
    if file_index is None:
        file_index = build_stage1_file_index(stage1_out_root)
    candidates = sorted(file_index.get((subdir, filename), []))
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {filename} under any stage1 part output in {stage1_out_root}"
        )
    if len(candidates) > 1:
        safe_print(f"[WARN] Found multiple matches for {filename}, using first: {candidates[0]}")
    return candidates[0]


def prepare_stage2_minimal_output(
    stage1_out_root: Path,
    stage2_out: Path,
    pdb_file: Path,
    overwrite: bool,
    file_index=None,
):
    if stage2_out.exists():
        if overwrite:
            shutil.rmtree(stage2_out)
        else:
            data_dir = stage2_out / "data"
            tunnel_edges_dir = data_dir / "tunnel_edges"
            tunnels_dir = data_dir / "tunnels"
            pdb_name = pdb_file.name
            expected = [
                data_dir / pdb_name,
                tunnel_edges_dir / f"edges_{pdb_name}.dat",
                tunnels_dir / f"tunnels_{pdb_name}.obj",
            ]
            if all(path.exists() for path in expected):
                return

    data_dir = stage2_out / "data"
    tunnel_edges_dir = data_dir / "tunnel_edges"
    tunnels_dir = data_dir / "tunnels"

    tunnel_edges_dir.mkdir(parents=True, exist_ok=True)
    tunnels_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pdb_file, data_dir / pdb_file.name)

    pdb_name = pdb_file.name
    edges_name = f"edges_{pdb_name}.dat"
    tunnels_name = f"tunnels_{pdb_name}.obj"

    src_edges = find_stage1_file_across_parts(stage1_out_root, "tunnel_edges", edges_name, file_index=file_index)
    src_tunnels = find_stage1_file_across_parts(stage1_out_root, "tunnels", tunnels_name, file_index=file_index)

    shutil.copy2(src_edges, tunnel_edges_dir / src_edges.name)
    shutil.copy2(src_tunnels, tunnels_dir / src_tunnels.name)


def merge_tunnel_profiles_from_root(
    source_root: Path,
    out_csv: Path,
    keep_header_from_first: bool = True,
    source_label: str = "stage3",
    expected_job_names: list[str] | None = None,
):
    """
    递归收集指定根目录下所有 *_out/analysis/tunnel_profiles.csv，然后合并。
    """
    csv_paths = sorted(source_root.glob("*_out/analysis/tunnel_profiles.csv"))

    safe_print(f"[INFO] {source_label} found {len(csv_paths)} tunnel_profiles.csv files under: {source_root}")

    if expected_job_names:
        missing = []
        for job_name in expected_job_names:
            expected_path = source_root / f"{job_name}_out" / "analysis" / "tunnel_profiles.csv"
            if not expected_path.exists():
                missing.append(str(expected_path))
        if missing:
            safe_print(f"[ERR] Missing {len(missing)} tunnel_profiles.csv files required for {source_label}.")
            for path_text in missing[:20]:
                safe_print(f"[MISS] {path_text}")
            if len(missing) > 20:
                safe_print(f"[MISS] ... and {len(missing) - 20} more")
            raise FileNotFoundError(
                f"{source_label} requires complete part outputs, but {len(missing)} tunnel_profiles.csv files are missing."
            )

    if not csv_paths:
        safe_print(f"[WARN] No tunnel_profiles.csv found for {source_label} merge.")
        return False

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged_count = 0
    skipped_empty = 0
    written_any = False

    with out_csv.open("w", encoding="utf-8", newline="") as fout:
        for p in csv_paths:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines(True)
            if not lines:
                skipped_empty += 1
                safe_print(f"[WARN] Empty CSV skipped: {p}")
                continue

            if keep_header_from_first:
                if not written_any:
                    fout.write(lines[0])
                    if len(lines) > 1:
                        fout.writelines(lines[1:])
                    written_any = True
                else:
                    if len(lines) > 1:
                        fout.writelines(lines[1:])
            else:
                if len(lines) > 1:
                    fout.writelines(lines[1:])

            merged_count += 1
            safe_print(f"[CSV] merged: {p}")

    safe_print(f"[INFO] {source_label} merged CSV files: {merged_count}")
    if skipped_empty:
        safe_print(f"[INFO] {source_label} skipped empty CSV files: {skipped_empty}")
    safe_print(f"[CSV] output -> {out_csv}")
    return True


def write_manifest(parts, single_jobs, out_root: Path):
    manifest1 = out_root / "stage1_parts_manifest.txt"
    with manifest1.open("w", encoding="utf-8") as f:
        f.write("part_id\tpart_name\tfirst_frame\tlast_frame\tn_frames\tseq_indices\n")
        for rec in parts:
            seq_str = ",".join(str(x) for x in rec["seq_indices"])
            f.write(
                f"{rec['part_id']}\t{rec['part_name']}\t"
                f"{rec['first_frame']}\t{rec['last_frame']}\t"
                f"{len(rec['seq_indices'])}\t{seq_str}\n"
            )

    if single_jobs is None:
        return

    manifest2 = out_root / "stage2_single_manifest.txt"
    with manifest2.open("w", encoding="utf-8") as f:
        f.write("job_name\tseq_index\tframe_num\tpdb_name\n")
        for rec in single_jobs:
            f.write(f"{rec['job_name']}\t{rec['seq_index']}\t{rec['frame_num']}\t{rec['pdb_name']}\n")


def write_timing_summary(file_path: Path, title: str, results, total_elapsed: float):
    with file_path.open("w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"total_elapsed_sec\t{total_elapsed:.3f}\n")
        f.write(f"total_elapsed_min\t{total_elapsed/60:.3f}\n")
        f.write("\n")
        f.write("job_name\tstatus\telapsed_sec\telapsed_min\treturncode\tout_dir\n")
        for rec in sorted(results, key=lambda x: x["job_name"]):
            f.write(
                f"{rec['job_name']}\t{rec['status']}\t{rec['elapsed_sec']:.3f}\t"
                f"{rec['elapsed_sec']/60:.3f}\t{rec['returncode']}\t{rec['out_dir']}\n"
            )


def write_workflow_summary(
    file_path: Path,
    stage1_elapsed: float,
    stage2_elapsed: float,
    stage3_elapsed: float,
    stage4_elapsed: float,
    workflow_elapsed: float,
):
    with file_path.open("w", encoding="utf-8") as f:
        f.write("workflow_timing_summary\n")
        f.write(f"stage1_elapsed_sec\t{stage1_elapsed:.3f}\n")
        f.write(f"stage1_elapsed_min\t{stage1_elapsed/60:.3f}\n")
        f.write(f"stage2_elapsed_sec\t{stage2_elapsed:.3f}\n")
        f.write(f"stage2_elapsed_min\t{stage2_elapsed/60:.3f}\n")
        f.write(f"stage3_elapsed_sec\t{stage3_elapsed:.3f}\n")
        f.write(f"stage3_elapsed_min\t{stage3_elapsed/60:.3f}\n")
        f.write(f"stage4_elapsed_sec\t{stage4_elapsed:.3f}\n")
        f.write(f"stage4_elapsed_min\t{stage4_elapsed/60:.3f}\n")
        f.write(f"workflow_elapsed_sec\t{workflow_elapsed:.3f}\n")
        f.write(f"workflow_elapsed_min\t{workflow_elapsed/60:.3f}\n")


def main():
    workflow_t0 = time.time()

    ap = argparse.ArgumentParser(
        description="Segmented CAVER workflow for tunnel computation, optional continuation, and tunnel profile aggregation."
    )

    ap.add_argument("--input-dir", required=True, help="Input directory containing *.pdb files")
    ap.add_argument("--conf", required=True, help="Path to the source CAVER configuration file")
    ap.add_argument("--n-parts", required=True, type=int, help="Number of segmented jobs for stage 1")
    ap.add_argument("--out-root", required=True, help="Root output directory")

    ap.add_argument("--run-stage1", choices=["yes", "no"], default="yes")
    ap.add_argument("--run-stage2", choices=["yes", "no"], default="yes")
    ap.add_argument("--run-stage3", choices=["yes", "no"], default="yes")
    ap.add_argument(
        "--stage1-stop-after",
        default="tunnels",
        help="Value written to stop_after for stage 1, for example tunnels or never",
    )
    ap.add_argument(
        "--stage3-source",
        choices=["stage1", "stage2"],
        default="stage2",
        help="Output level used when stage 3 aggregates tunnel_profiles.csv",
    )
    ap.add_argument("--java", default="java", help="Java executable")
    ap.add_argument("--xmx", default="512m", help="Java heap size for -Xmx, for example 512m or 1G")
    ap.add_argument("--lib-dir", default="./lib", help="CAVER lib directory")
    ap.add_argument("--caver-jar", default="./caver.jar", help="Path to caver.jar")
    ap.add_argument("--home-dir", default="./", help="Path passed to CAVER -home")

    ap.add_argument("--jobs", type=int, default=4, help="Number of parallel jobs (default: 4)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them")

    ap.add_argument("--stage2-load-tunnels", choices=["yes", "no"], default="yes")
    ap.add_argument("--stage2-load-cluster-tree", choices=["yes", "no"], default="no")
    ap.add_argument("--stage2-swap", choices=["yes", "no", "maybe"], default="yes")

    ap.add_argument("--java-active-processor-count", type=int, default=1)
    ap.add_argument("--java-parallel-gc-threads", type=int, default=1)
    ap.add_argument("--java-conc-gc-threads", type=int, default=1)

    ap.add_argument("--stage3-out-csv", default="merged_tunnel_profiles.csv",
                    help="Output CSV filename produced by stage 3")
    ap.add_argument("--stage3-keep-header", choices=["yes", "no"], default="yes",
                    help="Keep the header row from the first CSV during stage 3 aggregation")
    
    ap.add_argument("--run-stage4", choices=["yes", "no"], default="no")
    ap.add_argument(
        "--stage4-out-dir",
        default="merged_clusters_timeless",
        help="Output directory name for stage 4 merged clusters_timeless files"
    )
    ap.add_argument(
        "--stage4-workers",
        type=int,
        default=0,
        help="Parallel workers for stage 4 PDB merge; 0 reuses --jobs"
    )
    ap.add_argument(
        "--stage4-incremental",
        choices=["yes", "no"],
        default="yes",
        help="Skip stage 4 outputs that are already newer than all inputs"
    )
    ap.add_argument(
        "--stage4-include-conect",
        choices=["yes", "no"],
        default="yes",
        help="Keep and remap CONECT records in stage 4; no is faster and produces smaller files"
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    conf_file = Path(args.conf)
    out_root = Path(args.out_root)

    if not input_dir.is_dir():
        print(f"Input dir not found: {input_dir}", file=sys.stderr)
        return 2
    if not conf_file.exists():
        print(f"Config file not found: {conf_file}", file=sys.stderr)
        return 2

    pdb_files = sorted(
        input_dir.glob("*.pdb"),
        key=lambda p: extract_frame_number(p) if extract_frame_number(p) is not None else float("inf")
    )
    if not pdb_files:
        print(f"No .pdb found in {input_dir}", file=sys.stderr)
        return 2

    pdb_records = []
    for seq_index, p in enumerate(pdb_files, start=1):
        frame_num = extract_frame_number(p)
        if frame_num is None:
            continue
        pdb_records.append({
            "seq_index": seq_index,
            "frame_num": frame_num,
            "pdb_path": p,
            "pdb_name": p.name,
            "pdb_stem": p.stem,
        })

    if not pdb_records:
        print("Unable to extract frame numbers from filenames. Expected names similar to aligned_123.pdb.", file=sys.stderr)
        return 2

    out_root.mkdir(parents=True, exist_ok=True)

    stage1_conf_root = out_root / "stage1_part_configs"
    stage1_out_root = out_root / "stage1_part_outputs"
    stage1_conf_root.mkdir(parents=True, exist_ok=True)
    stage1_out_root.mkdir(parents=True, exist_ok=True)

    seq_indices = [x["seq_index"] for x in pdb_records]
    parts = split_into_n_parts(seq_indices, args.n_parts)

    stage1_jobs = []
    seq_to_part = {}

    for rec in parts:
        conf_out = stage1_conf_root / f"{rec['part_name']}_config.txt"
        rewrite_config(
            original_conf=conf_file,
            output_conf=conf_out,
            first_frame=rec["first_frame"],
            last_frame=rec["last_frame"],
            stop_after_value=args.stage1_stop_after,
        )
        rec["conf_file"] = conf_out
        rec["out_dir"] = stage1_out_root / f"{rec['part_name']}_out"

        for seq_idx in rec["seq_indices"]:
            seq_to_part[seq_idx] = rec

        stage1_jobs.append({
            "job_name": rec["part_name"],
            "conf_file": conf_out,
            "out_dir": rec["out_dir"],
            "first_frame": rec["first_frame"],
            "last_frame": rec["last_frame"],
        })

    stage2_conf_root = out_root / "stage2_single_configs"
    stage2_out_root = out_root / "stage2_single_outputs"
    needs_stage2_manifest = (
        args.run_stage2 == "yes"
        or (args.run_stage3 == "yes" and args.stage3_source == "stage2")
        or args.run_stage4 == "yes"
    )
    if args.run_stage2 == "yes":
        stage2_conf_root.mkdir(parents=True, exist_ok=True)
        stage2_out_root.mkdir(parents=True, exist_ok=True)

    single_jobs = []
    for rec in pdb_records:
        seq_index = rec["seq_index"]
        frame_num = rec["frame_num"]
        pdb_path = rec["pdb_path"]

        job_name = f"{pdb_path.stem}_frame_{frame_num}"
        out_dir = stage2_out_root / f"{job_name}_out"
        job_rec = {
            "job_name": job_name,
            "seq_index": seq_index,
            "frame_num": frame_num,
            "pdb_name": rec["pdb_name"],
            "pdb_path": pdb_path,
            "parent_part_name": seq_to_part[seq_index]["part_name"],
            "out_dir": out_dir,
        }

        if args.run_stage2 == "yes":
            conf_out = stage2_conf_root / f"{job_name}_config.txt"
            rewrite_config(
                original_conf=conf_file,
                output_conf=conf_out,
                first_frame=seq_index,
                last_frame=seq_index,
                stop_after_value="never",
                load_tunnels_value=args.stage2_load_tunnels,
                load_cluster_tree_value=args.stage2_load_cluster_tree,
                swap_value=args.stage2_swap,
            )
            job_rec["conf_file"] = conf_out

        single_jobs.append(job_rec)

    write_manifest(parts, single_jobs if needs_stage2_manifest else None, out_root)

    safe_print(f"[INFO] Found {len(pdb_files)} pdb files")
    safe_print(f"[INFO] Valid pdb records: {len(pdb_records)}")
    safe_print(f"[INFO] Stage1 parts: {len(parts)}")
    safe_print(f"[INFO] Stage2 single-frame jobs: {len(single_jobs)}")

    lib_dir = Path(args.lib_dir)
    caver_jar = Path(args.caver_jar)
    home_dir = Path(args.home_dir)

    for p in [lib_dir, caver_jar]:
        if not p.exists():
            print(f"Not found: {p}", file=sys.stderr)
            return 2

    rc_any = 0
    stage1_elapsed = 0.0
    stage2_elapsed = 0.0
    stage3_elapsed = 0.0
    stage1_results = []
    stage2_results = []

    if args.run_stage1 == "yes":
        safe_print(f"[INFO] Running Stage1 (stop_after={args.stage1_stop_after})...")
        stage1_t0 = time.time()
        rc1, stage1_results = run_jobs_parallel(
            jobs=stage1_jobs,
            input_dir=input_dir,
            java=args.java,
            xmx=args.xmx,
            lib_dir=lib_dir,
            caver_jar=caver_jar,
            home_dir=home_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            jobs_n=args.jobs,
            java_active_processor_count=args.java_active_processor_count,
            java_parallel_gc_threads=args.java_parallel_gc_threads,
            java_conc_gc_threads=args.java_conc_gc_threads,
        )
        stage1_elapsed = time.time() - stage1_t0
        rc_any = rc_any or rc1

        write_timing_summary(
            out_root / "stage1_timing_summary.txt",
            "stage1_timing_summary",
            stage1_results,
            stage1_elapsed,
        )
        if rc1:
            failed_summary = summarize_failed_jobs(stage1_results)
            raise RuntimeError(
                "Stage1 failed for one or more parts. "
                f"See stage1_timing_summary.txt under {out_root}. "
                f"Failed jobs: {failed_summary}"
            )
    else:
        safe_print("[INFO] Stage1 skipped.")

    if args.run_stage2 == "yes":
        safe_print("[INFO] Preparing Stage2 minimal output directories...")
        stage1_file_index = build_stage1_file_index(stage1_out_root)
        for rec in single_jobs:
            prepare_stage2_minimal_output(
                stage1_out_root=stage1_out_root,
                stage2_out=rec["out_dir"],
                pdb_file=rec["pdb_path"],
                overwrite=args.overwrite,
                file_index=stage1_file_index,
            )

        safe_print("[INFO] Running Stage2 (single-frame, continue from stage1 minimal files)...")
        stage2_t0 = time.time()
        rc2, stage2_results = run_jobs_parallel(
            jobs=single_jobs,
            input_dir=input_dir,
            java=args.java,
            xmx=args.xmx,
            lib_dir=lib_dir,
            caver_jar=caver_jar,
            home_dir=home_dir,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            jobs_n=args.jobs,
            java_active_processor_count=args.java_active_processor_count,
            java_parallel_gc_threads=args.java_parallel_gc_threads,
            java_conc_gc_threads=args.java_conc_gc_threads,
            complete_markers=("analysis/tunnel_profiles.csv",),
        )
        stage2_elapsed = time.time() - stage2_t0
        rc_any = rc_any or rc2

        write_timing_summary(
            out_root / "stage2_timing_summary.txt",
            "stage2_timing_summary",
            stage2_results,
            stage2_elapsed,
        )
    else:
        safe_print("[INFO] Stage2 skipped.")

    if args.run_stage3 == "yes":
        safe_print("[INFO] Running Stage3 (merge tunnel_profiles.csv)...")
        stage3_t0 = time.time()
        keep_header = args.stage3_keep_header == "yes"
        source_root = stage1_out_root if args.stage3_source == "stage1" else stage2_out_root
        merge_tunnel_profiles_from_root(
            source_root=source_root,
            out_csv=out_root / args.stage3_out_csv,
            keep_header_from_first=keep_header,
            source_label=f"Stage3({args.stage3_source})",
            expected_job_names=[rec["part_name"] for rec in parts] if args.stage3_source == "stage1" else [rec["job_name"] for rec in single_jobs],
        )
        stage3_elapsed = time.time() - stage3_t0
    else:
        safe_print("[INFO] Stage3 skipped.")

    
    stage4_elapsed = 0.0
    if args.run_stage4 == "yes":
        safe_print("[INFO] Running Stage4 (merge clusters_timeless pdbs)...")
        stage4_t0 = time.time()
        merge_clusters_timeless_from_stage2(
            stage2_out_root=stage2_out_root,
            merged_root=out_root / args.stage4_out_dir,
            workers=args.stage4_workers if args.stage4_workers > 0 else args.jobs,
            incremental=args.stage4_incremental == "yes",
            include_conect=args.stage4_include_conect == "yes",
        )
        stage4_elapsed = time.time() - stage4_t0
    else:
        safe_print("[INFO] Stage4 skipped.")

    workflow_elapsed = time.time() - workflow_t0
    write_workflow_summary(
        out_root / "workflow_timing_summary.txt",
        stage1_elapsed,
        stage2_elapsed,
        stage3_elapsed,
        stage4_elapsed,
        workflow_elapsed,
    )

    safe_print(f"[TIME] Stage1 total: {stage1_elapsed:.1f}s ({stage1_elapsed/60:.2f} min)")
    safe_print(f"[TIME] Stage2 total: {stage2_elapsed:.1f}s ({stage2_elapsed/60:.2f} min)")
    safe_print(f"[TIME] Stage3 total: {stage3_elapsed:.1f}s ({stage3_elapsed/60:.2f} min)")
    safe_print(f"[TIME] Stage4 total: {stage4_elapsed:.1f}s ({stage4_elapsed/60:.2f} min)")
    safe_print(f"[TIME] Workflow total: {workflow_elapsed:.1f}s ({workflow_elapsed/60:.2f} min)")
    safe_print("[DONE] Workflow finished.")
    return rc_any


if __name__ == "__main__":
    raise SystemExit(main())
