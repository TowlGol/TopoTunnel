from pathlib import Path
import shutil
import time
import re
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pipeline_config import (
    rmsd_root_dir,
    rmsd_target_folders,
    rmsd_output_dir,
    rmsd_workers,
    rmsd_chunk_size,
    rmsd_use_memory_cache,
)


# =========================
# 1. 总目录
# =========================
ROOT_DIR = Path(rmsd_root_dir)

# =========================
# 2. 要处理的 PDB 文件夹
# =========================
TARGET_FOLDERS = [
    *rmsd_target_folders,
]

# =========================
# 3. PDB 文件匹配
# =========================
FILE_PATTERN = "*.pdb"

# =========================
# 4. 输出目录
# =========================
OUTPUT_DIR = Path(rmsd_output_dir)

# =========================
# 5. 只读取这些主链原子
# =========================
BACKBONE_NAMES = {"N", "CA", "C"}

# =========================
# 6. 多进程设置
# =========================
# 建议先试 8、12、16
# 不一定越大越快，因为读取 PDB 会受到磁盘 I/O 限制
N_WORKERS = rmsd_workers

# 每个进程一次处理多少个 PDB
CHUNK_SIZE = rmsd_chunk_size

# 是否缓存全部坐标到内存
# True 更快
# False 省内存，但会读两遍 PDB
USE_MEMORY_CACHE = rmsd_use_memory_cache

# 坐标缓存精度
CACHE_DTYPE = np.float32


def natural_key(path: Path):
    """
    让 aligned_2.pdb 排在 aligned_10.pdb 前面。
    """
    return [
        int(s) if s.isdigit() else s.lower()
        for s in re.split(r"(\d+)", path.name)
    ]


def extract_frame_id(path: Path):
    """
    从 aligned_1234.pdb 中提取 1234。
    如果提取失败，返回 None。
    """
    m = re.search(r"aligned_(\d+)", path.stem)
    if m:
        return int(m.group(1))
    return None


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    else:
        return f"{seconds / 3600:.2f} h"


def make_chunks(items, chunk_size):
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def read_n_ca_c_coords_fast(pdb_file: Path) -> np.ndarray:
    """
    快速读取 PDB 中 ATOM 记录的 N / CA / C 坐标。

    使用 PDB 固定列：
    atom name: 12:16
    x: 30:38
    y: 38:46
    z: 46:54

    返回:
        shape = (n_atoms, 3)
    """
    coords = []

    with pdb_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 比 startswith("ATOM") 稍微快一点
            if line[:4] != "ATOM":
                continue

            atom_name = line[12:16].strip()

            if atom_name not in BACKBONE_NAMES:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            coords.append((x, y, z))

    if not coords:
        raise ValueError("No N/CA/C coordinates were read from the PDB file")

    return np.asarray(coords, dtype=CACHE_DTYPE)


def read_chunk_worker(pdb_files):
    """
    子进程 worker：
    读取一批 PDB 的 N/CA/C 坐标。
    """
    records = []
    invalid_records = []

    expected_shape = None

    for pdb_file_str in pdb_files:
        pdb_file = Path(pdb_file_str)

        try:
            coords = read_n_ca_c_coords_fast(pdb_file)
        except Exception as e:
            invalid_records.append((pdb_file.name, str(e)))
            continue

        if expected_shape is None:
            expected_shape = coords.shape

        records.append((str(pdb_file), coords))

    return records, invalid_records


def read_all_coords_parallel(pdb_files):
    """
    多进程读取所有 PDB 坐标。
    """
    start_time = time.perf_counter()

    pdb_file_strs = [str(p) for p in pdb_files]
    chunks = list(make_chunks(pdb_file_strs, CHUNK_SIZE))

    all_records = []
    all_invalid = []

    finished_chunks = 0

    print(f"  Total PDB files: {len(pdb_files)}")
    print(f"  Chunk count: {len(chunks)}")
    print(f"  Worker processes: {N_WORKERS}")
    print(f"  PDB files per chunk: {CHUNK_SIZE}")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [
            executor.submit(read_chunk_worker, chunk)
            for chunk in chunks
        ]

        for fut in as_completed(futures):
            records, invalid_records = fut.result()

            all_records.extend(records)
            all_invalid.extend(invalid_records)

            finished_chunks += 1

            if (
                finished_chunks == 1
                or finished_chunks % 5 == 0
                or finished_chunks == len(chunks)
            ):
                elapsed = time.perf_counter() - start_time
                print(
                    f"  [Parallel read] completed chunk {finished_chunks}/{len(chunks)} "
                    f"| valid PDB files loaded {len(all_records)} "
                    f"| invalid {len(all_invalid)} "
                    f"| elapsed {format_seconds(elapsed)}"
                )

    # 因为 as_completed 会打乱顺序，所以这里按文件名重新排序
    all_records.sort(key=lambda item: natural_key(Path(item[0])))

    return all_records, all_invalid


def validate_and_stack_records(records):
    """
    检查所有 PDB 的 N/CA/C 原子数是否一致，并堆叠为数组。

    返回：
        valid_files: list[Path]
        coords_all: shape = (n_frames, n_atoms, 3)
        invalid_records
    """
    valid_files = []
    valid_coords = []
    invalid_records = []

    expected_shape = None

    for pdb_file_str, coords in records:
        pdb_file = Path(pdb_file_str)

        if expected_shape is None:
            expected_shape = coords.shape

        if coords.shape != expected_shape:
            invalid_records.append(
                (
                    pdb_file.name,
                    f"Inconsistent N/CA/C atom count: {coords.shape} vs expected {expected_shape}"
                )
            )
            continue

        valid_files.append(pdb_file)
        valid_coords.append(coords)

    if not valid_coords:
        return [], None, invalid_records

    coords_all = np.stack(valid_coords, axis=0)

    return valid_files, coords_all, invalid_records


def compute_mean_and_rmsd_vectorized(coords_all):
    """
    NumPy 向量化计算：
    1. mean_coords
    2. 每一帧到 mean_coords 的 RMSD
    """
    mean_coords = coords_all.mean(axis=0, dtype=np.float64)

    diff = coords_all.astype(np.float64, copy=False) - mean_coords

    rmsd_values = np.sqrt(
        np.mean(
            np.sum(diff * diff, axis=2),
            axis=1
        )
    )

    return mean_coords, rmsd_values


def compute_mean_from_files_two_pass(valid_files):
    """
    省内存模式第一遍：
    从 PDB 文件中计算平均坐标。
    """
    coord_sum = None
    n_valid = 0

    start = time.perf_counter()

    for i, pdb_file in enumerate(valid_files, start=1):
        coords = read_n_ca_c_coords_fast(pdb_file).astype(np.float64)

        if coord_sum is None:
            coord_sum = np.zeros_like(coords, dtype=np.float64)

        coord_sum += coords
        n_valid += 1

        if i == 1 or i % 1000 == 0 or i == len(valid_files):
            elapsed = time.perf_counter() - start
            print(
                f"  [Pass 1: mean coordinates] {i}/{len(valid_files)} "
                f"| elapsed {format_seconds(elapsed)}"
            )

    return coord_sum / n_valid


def compute_rmsd_from_files_two_pass(valid_files, mean_coords):
    """
    省内存模式第二遍：
    从 PDB 文件计算每一帧 RMSD。
    """
    rmsd_values = np.zeros(len(valid_files), dtype=np.float64)

    best_idx = None
    best_rmsd = float("inf")

    start = time.perf_counter()

    for i, pdb_file in enumerate(valid_files):
        coords = read_n_ca_c_coords_fast(pdb_file).astype(np.float64)

        diff = coords - mean_coords
        val = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

        rmsd_values[i] = val

        if val < best_rmsd:
            best_rmsd = val
            best_idx = i

        count = i + 1

        if count == 1 or count % 1000 == 0 or count == len(valid_files):
            elapsed = time.perf_counter() - start
            print(
                f"  [Pass 2: RMSD] {count}/{len(valid_files)} "
                f"| current best {valid_files[best_idx].name}, RMSD {best_rmsd:.6f} "
                f"| elapsed {format_seconds(elapsed)}"
            )

    return rmsd_values


def save_invalid_records(invalid_records, out_file: Path):
    with out_file.open("w", encoding="utf-8") as f:
        f.write("File\tReason\n")
        for name, reason in invalid_records:
            f.write(f"{name}\t{reason}\n")


def save_rmsd_rank(valid_files, rmsd_values, out_file: Path):
    order = np.argsort(rmsd_values)

    with out_file.open("w", encoding="utf-8") as f:
        f.write("Rank\tFile\tFrameID\tRMSD_to_Mean_N_CA_C\n")

        for rank, idx in enumerate(order, start=1):
            pdb_file = valid_files[idx]
            frame_id = extract_frame_id(pdb_file)

            if frame_id is None:
                frame_id_text = str(idx)
            else:
                frame_id_text = str(frame_id)

            f.write(
                f"{rank}\t"
                f"{pdb_file.name}\t"
                f"{frame_id_text}\t"
                f"{rmsd_values[idx]:.6f}\n"
            )


def analyze_folder(folder: Path):
    folder_start = time.perf_counter()

    print("\n" + "=" * 100)
    print(f"[Analyze] {folder}")

    out_subdir = OUTPUT_DIR / folder.name
    out_subdir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(folder.glob(FILE_PATTERN), key=natural_key)

    print(f"  Detected PDB files: {len(pdb_files)}")

    if not pdb_files:
        print("  No PDB files were found")
        return None

    if USE_MEMORY_CACHE:
        print("\n  Execution mode: parallel read + memory cache + NumPy vectorized RMSD")

        records, invalid_records_read = read_all_coords_parallel(pdb_files)

        valid_files, coords_all, invalid_records_shape = validate_and_stack_records(records)

        invalid_records = invalid_records_read + invalid_records_shape

        if coords_all is None:
            print("  No valid PDB files")
            save_invalid_records(invalid_records, out_subdir / "invalid_files.txt")
            return None

        print(f"\n  Valid PDB files: {len(valid_files)}")
        print(f"  Backbone N/CA/C atoms per frame: {coords_all.shape[1]}")
        print(f"  Coordinate cache shape: {coords_all.shape}")
        print(f"  Coordinate cache memory: {coords_all.nbytes / 1024 / 1024 / 1024:.3f} GB")

        calc_start = time.perf_counter()
        mean_coords, rmsd_values = compute_mean_and_rmsd_vectorized(coords_all)
        calc_elapsed = time.perf_counter() - calc_start

        print(f"  NumPy vectorized RMSD time: {format_seconds(calc_elapsed)}")

    else:
        print("\n  Execution mode: memory-saving two-pass PDB read")

        # 这里先单线程验证。
        # 如果你不想省内存，强烈建议 USE_MEMORY_CACHE=True。
        valid_files = []
        invalid_records = []
        expected_shape = None

        for pdb_file in pdb_files:
            try:
                coords = read_n_ca_c_coords_fast(pdb_file)
            except Exception as e:
                invalid_records.append((pdb_file.name, str(e)))
                continue

            if expected_shape is None:
                expected_shape = coords.shape

            if coords.shape != expected_shape:
                invalid_records.append(
                    (
                        pdb_file.name,
                        f"Inconsistent N/CA/C atom count: {coords.shape} vs expected {expected_shape}"
                    )
                )
                continue

            valid_files.append(pdb_file)

        if not valid_files:
            print("  No valid PDB files")
            save_invalid_records(invalid_records, out_subdir / "invalid_files.txt")
            return None

        print(f"  Valid PDB files: {len(valid_files)}")

        mean_coords = compute_mean_from_files_two_pass(valid_files)
        rmsd_values = compute_rmsd_from_files_two_pass(valid_files, mean_coords)

    save_invalid_records(invalid_records, out_subdir / "invalid_files.txt")

    best_idx = int(np.argmin(rmsd_values))
    best_file = valid_files[best_idx]
    best_rmsd = float(rmsd_values[best_idx])

    save_rmsd_rank(
        valid_files,
        rmsd_values,
        out_subdir / "rmsd_to_mean_ranked.txt"
    )

    # 保存平均构象坐标
    np.save(
        out_subdir / "mean_coords_N_CA_C.npy",
        mean_coords.astype(np.float32)
    )

    # 复制最佳 PDB
    shutil.copy2(best_file, out_subdir / best_file.name)

    elapsed = time.perf_counter() - folder_start

    with (out_subdir / "best_result.txt").open("w", encoding="utf-8") as f:
        f.write(f"Folder: {folder.name}\n")
        f.write(f"InputFolder: {folder}\n")
        f.write(f"BestFile: {best_file.name}\n")
        f.write(f"BestFrameID: {extract_frame_id(best_file)}\n")
        f.write(f"BestIndexInValidList: {best_idx}\n")
        f.write(f"BestRMSD_to_Mean_N_CA_C: {best_rmsd:.6f}\n")
        f.write(f"NumPDBFiles: {len(pdb_files)}\n")
        f.write(f"NumValidFiles: {len(valid_files)}\n")
        f.write(f"NumInvalidFiles: {len(invalid_records)}\n")
        f.write("AtomSelection: protein and name N CA C\n")
        f.write("AlreadyAligned: Yes, assumed aligned to first frame before this script\n")
        f.write(f"UseMemoryCache: {USE_MEMORY_CACHE}\n")
        f.write(f"NWorkers: {N_WORKERS}\n")
        f.write(f"ChunkSize: {CHUNK_SIZE}\n")
        f.write(f"ElapsedSeconds: {elapsed:.2f}\n")
        f.write(f"ElapsedReadable: {format_seconds(elapsed)}\n")

    print("\n  Folder completed")
    print(f"  Closest file to the mean conformation: {best_file.name}")
    print(f"  Frame ID: {extract_frame_id(best_file)}")
    print(f"  RMSD to mean: {best_rmsd:.6f}")
    print(f"  Folder elapsed time: {format_seconds(elapsed)}")

    return {
        "folder": folder.name,
        "best_file": best_file.name,
        "best_frame_id": extract_frame_id(best_file),
        "best_rmsd": best_rmsd,
        "n_pdb": len(pdb_files),
        "n_valid": len(valid_files),
        "n_invalid": len(invalid_records),
        "elapsed": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Select the reference conformation closest to the mean N/CA/C structure")
    parser.add_argument("--root-dir", default=str(ROOT_DIR), help="Root directory containing PDB files")
    parser.add_argument(
        "--target-folders",
        default=",".join(TARGET_FOLDERS),
        help="Comma-separated subdirectories to analyze",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="PDB files processed per chunk")
    parser.add_argument(
        "--use-memory-cache",
        choices=("yes", "no"),
        default="yes" if USE_MEMORY_CACHE else "no",
        help="Whether all coordinates should be cached in memory",
    )
    return parser.parse_args()


def main():
    global ROOT_DIR, TARGET_FOLDERS, OUTPUT_DIR, N_WORKERS, CHUNK_SIZE, USE_MEMORY_CACHE
    args = parse_args()
    ROOT_DIR = Path(args.root_dir)
    TARGET_FOLDERS = [item.strip() for item in args.target_folders.split(",") if item.strip()]
    OUTPUT_DIR = Path(args.output_dir)
    N_WORKERS = int(args.workers)
    CHUNK_SIZE = int(args.chunk_size)
    USE_MEMORY_CACHE = args.use_memory_cache == "yes"

    total_start = time.perf_counter()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Reference conformation selection from aligned PDB frames")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"N_WORKERS: {N_WORKERS}")
    print(f"USE_MEMORY_CACHE: {USE_MEMORY_CACHE}")

    summary_lines = []
    summary_lines.append(
        "Folder\tBestFile\tBestFrameID\tBestRMSD_to_Mean_N_CA_C\t"
        "NumPDBFiles\tNumValidFiles\tNumInvalidFiles\tElapsedSeconds\n"
    )

    for folder_name in TARGET_FOLDERS:
        folder = ROOT_DIR / folder_name

        if not folder.exists():
            print(f"[Skip] Folder does not exist: {folder}")
            continue

        if not folder.is_dir():
            print(f"[Skip] Not a directory: {folder}")
            continue

        result = analyze_folder(folder)

        if result is None:
            continue

        summary_lines.append(
            f"{result['folder']}\t"
            f"{result['best_file']}\t"
            f"{result['best_frame_id']}\t"
            f"{result['best_rmsd']:.6f}\t"
            f"{result['n_pdb']}\t"
            f"{result['n_valid']}\t"
            f"{result['n_invalid']}\t"
            f"{result['elapsed']:.2f}\n"
        )

    summary_file = OUTPUT_DIR / "mean_rmsd_summary.txt"

    with summary_file.open("w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 100)
    print("Completed")
    print(f"Summary file: {summary_file}")
    print(f"Detailed output directory: {OUTPUT_DIR}")
    print(f"Total elapsed time: {format_seconds(total_elapsed)}")


if __name__ == "__main__":
    main()
