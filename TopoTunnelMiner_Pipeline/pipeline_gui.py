import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import os
import tempfile
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext

from pipeline_config import (
    PDB_path,
    caver_conf,
    caver_home_dir,
    caver_jobs,
    caver_n_parts,
    caver_overwrite,
    caver_run_stage1,
    caver_run_stage2,
    caver_run_stage3,
    caver_run_stage4,
    caver_script,
    caver_stage2_load_tunnels,
    caver_stage3_out_csv,
    caver_stage4_out_dir,
    caver_xmx,
    dataDir,
    excluded_residues,
    mutations,
    output_root_dir,
    project_name,
    rmsd_chunk_size,
    rmsd_use_memory_cache,
    rmsd_workers,
    tunnel_profile_csv,
)
from convert_dataset_to_sqlite import convert_pkl_to_sqlite


BASE_DIR = Path(__file__).resolve().parent
CONFIG_FILE = BASE_DIR / "pipeline_config.py"

STEP_SCRIPTS = {
    "1": "stage_01_run_caver.py",
    "1full": "stage_01_run_caver.py",
    "1merge": "stage_01_run_caver.py",
    "min": "stage_05_preprocess_dataset.py",
    "2": "stage_02_select_reference.py",
    "3": "stage_03_map_topology.py",
    "3d": "stage_03_map_topology.py",
    "fast": "stage_04_build_path_points.py",
    "7": "stage_05_preprocess_dataset.py",
}

STEP_LABELS = {
    "1": "Segmented CAVER Tunnel Computation and Export",
    "1full": "Segmented CAVER Tunnel Computation and Tunnel-Profile Aggregation",
    "1merge": "Tunnel Profile Aggregation",
    "min": "Reference-Conformation Mapping and Dataset Preparation",
    "2": "Reference-Conformation Selection",
    "3": "Tunnel-Path Topological Representation and Residue Statistics",
    "3d": "Residue-Pair Distance Analysis",
    "fast": "Direct Path-Point Export from Topological Mapping",
    "7": "Tunnel-Path Preprocessing and Dataset Preparation",
}

MINIMAL_FIELDS = [
    ("dataDir", "Aligned MD Frame Directory", "dir", dataDir),
    ("output_root_dir", "Output Root Directory", "dir", output_root_dir),
    ("project_name", "Dataset Name", "str", project_name),
    ("tunnel_profile_csv", "Tunnel-Profile CSV File", "file", tunnel_profile_csv),
    ("PDB_path", "Reference-Conformation PDB File", "file", PDB_path),
    ("pipeline_cpu_workers", "CPU Workers for Mapping and Preprocessing", "int", rmsd_workers),
    ("excluded_residues", "Excluded Residues", "list_int", excluded_residues),
    ("mutations", "Mutation Sites", "list_str", mutations),
]

CAVER_EXTRA_FIELDS = [
    ("caver_n_parts", "Number of Segments", "int", caver_n_parts),
    ("caver_jobs", "Parallel CAVER Jobs", "int", caver_jobs),
    ("caver_xmx", "Java Heap Size (-Xmx)", "str", caver_xmx),
    ("caver_conf", "CAVER Configuration File", "file", caver_conf),
    ("caver_home_dir", "CAVER Home Directory", "dir", caver_home_dir),
    ("caver_run_stage1", "stage1", "bool", caver_run_stage1 == "yes"),
    ("caver_run_stage2", "stage2", "bool", caver_run_stage2 == "yes"),
    ("caver_run_stage3", "stage3", "bool", caver_run_stage3 == "yes"),
    ("caver_run_stage4", "stage4", "bool", caver_run_stage4 == "yes"),
]


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"
def format_minutes(seconds: float) -> str:
    return f"{seconds/60:.2f}s"


def derive_dataset_name(pdb_frames_dir: str) -> str:
    folder_name = Path(pdb_frames_dir).name.strip()
    if folder_name.endswith("_pdb_frames"):
        folder_name = folder_name[:-11]
    return folder_name or "dataset"


def natural_sort_key(path_value: str) -> list:
    name = Path(path_value).name
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def extract_frame_id_from_path(path_value: Path):
    stem = path_value.stem
    match = re.search(r"aligned_(\d+)", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def resolve_caver_jar(caver_home_dir: str) -> str:
    root = Path(caver_home_dir).expanduser()
    if not caver_home_dir.strip():
        return ""
    if not root.exists():
        raise FileNotFoundError(f"CAVER home directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"CAVER home path is not a directory: {root}")

    direct = root / "caver.jar"
    if direct.is_file():
        return str(direct)

    candidates = []
    for jar_path in root.rglob("*.jar"):
        name = jar_path.name.lower()
        if "source" in name or "javadoc" in name:
            continue
        score = 0
        if name == "caver.jar":
            score += 100
        if "caver" in name:
            score += 20
        if (jar_path.parent / "lib").is_dir():
            score += 10
        if (jar_path.parent.parent / "lib").is_dir():
            score += 5
        candidates.append((score, len(jar_path.parts), jar_path))

    if not candidates:
        raise FileNotFoundError(f"No JAR file was found under the CAVER home directory: {root}")

    candidates.sort(key=lambda item: (-item[0], item[1], str(item[2]).lower()))
    return str(candidates[0][2])


def to_abs_path(path_value: str, base_dir: Path = BASE_DIR) -> str:
    path_obj = Path(path_value).expanduser()
    if path_obj.is_absolute():
        return str(path_obj)
    return str((base_dir / path_obj).resolve())


def path_to_env_literal(value) -> str:
    return format_env_literal("str", value)


def upsert_config_property(lines, key: str, value: str):
    pattern = re.compile(rf"^\s*{re.escape(key)}\b", re.IGNORECASE)
    replacement = f"{key} {value}\n"
    found = False
    updated = []
    for line in lines:
        if pattern.match(line):
            if not found:
                updated.append(replacement)
                found = True
            continue
        updated.append(line)
    if not found:
        if updated and updated[-1] and not updated[-1].endswith("\n"):
            updated[-1] = updated[-1] + "\n"
        updated.append(replacement)
    return updated


def prepare_caver_runtime_config(config: dict) -> str:
    source_conf = Path(to_abs_path(config["caver_conf"]))
    if not source_conf.exists():
        raise FileNotFoundError(f"CAVER configuration file does not exist: {source_conf}")

    out_root = Path(to_abs_path(config["caver_out_root"]))
    out_root.mkdir(parents=True, exist_ok=True)
    runtime_conf = out_root / "runtime_caver_config.txt"

    lines = source_conf.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
    lines = upsert_config_property(lines, "load_cluster_tree", "no")
    lines = upsert_config_property(lines, "stop_after", config.get("caver_stage1_stop_after", "tunnels"))
    runtime_conf.write_text("".join(lines), encoding="utf-8")
    return str(runtime_conf)


def derive_config(minimal_config: dict) -> dict:
    data_dir = minimal_config["dataDir"].strip()
    dataset_name = minimal_config["project_name"].strip() or derive_dataset_name(data_dir)
    output_root = minimal_config["output_root_dir"].strip()
    run_root = Path(output_root) / dataset_name

    caver_run_dir = run_root / "01_caver"
    reference_run_dir = run_root / "02_reference"
    topology_run_dir = run_root / "03_topology"
    path_points_run_dir = run_root / "04_path_points"
    preprocess_run_dir = run_root
    frame_index_txt = run_root / "frame_index.txt"

    return {
        "name": dataset_name,
        "dataDir": data_dir,
        "output_root_dir": output_root,
        "project_name": dataset_name,
        "excluded_residues": minimal_config["excluded_residues"],
        "mutations": minimal_config["mutations"],
        "input_reference_pdb": minimal_config["PDB_path"].strip(),
        "run_root": str(run_root),
        "tunnel_profile_csv": minimal_config["tunnel_profile_csv"].strip() or str(caver_run_dir / "all_tunnel_profiles.csv"),
        "PDB_path": minimal_config["PDB_path"].strip() or str(run_root / "representative_frame.pdb"),
        "four_atoms_csv": str(topology_run_dir / "atom_network.csv"),
        "residue_stats_csv": str(preprocess_run_dir / "residue_statistics.csv"),
        "residue_combination_stats_csv": str(preprocess_run_dir / "residue_combination_statistics.csv"),
        "residue_distance_relation_csv": str(preprocess_run_dir / "residue_distance_relations.csv"),
        "residue_pair_distance_csv": str(preprocess_run_dir / "residue_pair_distances.csv"),
        "residue_pair_path_presence_stats_csv": str(preprocess_run_dir / "residue_pair_path_presence_stats.csv"),
        "frame_index_txt": str(frame_index_txt),
        "all_points_csv": str(path_points_run_dir / "mapped_path_points.csv"),
        "preprocess_pkl": str(preprocess_run_dir / "preprocessed_paths.pkl"),
        "timing_summary_file": str(preprocess_run_dir / "timing_summary.txt"),
        "caver_script": caver_script,
        "caver_input_dir": data_dir,
        "caver_conf": minimal_config["caver_conf"],
        "caver_home_dir": minimal_config["caver_home_dir"],
        "caver_jar_path": resolve_caver_jar(minimal_config["caver_home_dir"]),
        "caver_runtime_conf": "",
        "caver_n_parts": minimal_config["caver_n_parts"],
        "caver_out_root": str(caver_run_dir),
        "caver_run_stage1": "yes" if minimal_config["caver_run_stage1"] else "no",
        "caver_run_stage2": "yes" if minimal_config["caver_run_stage2"] else "no",
        "caver_run_stage3": "yes" if minimal_config["caver_run_stage3"] else "no",
        "caver_run_stage4": "yes" if minimal_config["caver_run_stage4"] else "no",
        "caver_jobs": minimal_config["caver_jobs"],
        "caver_xmx": minimal_config["caver_xmx"],
        "caver_stage2_load_tunnels": caver_stage2_load_tunnels,
        "caver_stage3_out_csv": caver_stage3_out_csv,
        "caver_stage4_out_dir": caver_stage4_out_dir,
        "caver_overwrite": caver_overwrite,
        "rmsd_root_dir": data_dir,
        "rmsd_target_folders": ["."],
        "rmsd_output_dir": str(reference_run_dir),
        "rmsd_workers": int(minimal_config["pipeline_cpu_workers"]),
        "pipeline_cpu_workers": int(minimal_config["pipeline_cpu_workers"]),
        "rmsd_chunk_size": rmsd_chunk_size,
        "rmsd_use_memory_cache": rmsd_use_memory_cache,
    }


def build_fast_temp_config(config: dict) -> tuple[dict, Path]:
    run_root = Path(config["run_root"])
    run_root.mkdir(parents=True, exist_ok=True)
    fast_temp_root = Path(tempfile.mkdtemp(prefix=".fast_pipeline_", dir=str(run_root)))
    fast_reference_root = fast_temp_root / "02_reference"
    fast_topology_root = fast_temp_root / "03_topology"
    fast_path_points_root = fast_temp_root / "04_path_points"
    fast_preprocess_pkl = fast_temp_root / "preprocessed_paths.pkl"

    fast_config = dict(config)
    fast_config.update({
        "fast_temp_root": str(fast_temp_root),
        "input_reference_pdb": config.get("input_reference_pdb", config["PDB_path"]),
        "rmsd_output_dir": str(fast_reference_root),
        "PDB_path": str(fast_reference_root / "representative_frame.pdb"),
        "four_atoms_csv": str(fast_topology_root / "atom_network.csv"),
        "residue_distance_relation_csv": str(fast_temp_root / "residue_distance_relations.csv"),
        "residue_pair_distance_csv": str(fast_temp_root / "residue_pair_distances.csv"),
        "all_points_csv": str(fast_path_points_root / "mapped_path_points.csv"),
        "preprocess_pkl": str(fast_preprocess_pkl),
        "timing_summary_file": str(fast_temp_root / "timing_summary.txt"),
    })
    return fast_config, fast_temp_root


def parse_best_result(best_result_path: Path):
    best_file = None
    best_frame_id = None
    with best_result_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("BestFile:"):
                best_file = line.split(":", 1)[1].strip()
            elif line.startswith("BestFrameID:"):
                best_frame_id = line.split(":", 1)[1].strip()
    return best_file, best_frame_id


def format_env_literal(field_type: str, value):
    if field_type == "int":
        return str(int(value))
    if field_type == "list_int":
        return "[" + ", ".join(str(int(item)) for item in value) + "]"
    if field_type == "list_str":
        return "[" + ", ".join(repr(str(item)) for item in value) + "]"
    if field_type == "str":
        return repr(str(value).replace("\\", "/"))
    return repr(str(value))


class RunnerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("TopoTunnelMiner Analysis Runner")
        self.default_window_width = 1120
        self.default_window_height = 860
        self.root.geometry(f"{self.default_window_width}x{self.default_window_height}")

        self.log_queue = queue.Queue()
        self.running = False
        self.step_vars = {}
        self.checkbuttons = []
        self.run_thread = None
        self.config_vars = {}
        self.caver_extra_vars = {}
        self.auto_input_keys = {"tunnel_profile_csv", "PDB_path"}
        self.manual_input_override_keys = set()
        self._syncing_input_defaults = False

        self._build_ui()
        self.refresh_input_defaults()
        self.root.after(0, self.update_window_height_limit)
        self._poll_log_queue()

    def _build_ui(self):
        self.scrollable_frame = tk.Frame(self.root)
        self.scrollable_frame.pack(fill="both", expand=True)

        title = tk.Label(
            self.scrollable_frame,
            text="TopoTunnelMiner: Tunnel Computation and Topology-Based Mapping onto the Reference Conformation",
            font=("Microsoft YaHei UI", 15, "bold"),
            anchor="w",
        )
        title.pack(fill="x", padx=12, pady=(12, 6))

        config_outer = tk.LabelFrame(self.scrollable_frame, text="Analysis Inputs", padx=8, pady=8)
        config_outer.pack(fill="x", padx=12, pady=6)

        btn_row = tk.Frame(config_outer)
        btn_row.pack(fill="x", pady=(0, 8))
        tk.Button(btn_row, text="Save Input Parameters", command=self.save_config).pack(side="left")

        form = tk.Frame(config_outer)
        form.pack(fill="x")

        for row, (key, label, field_type, default_value) in enumerate(MINIMAL_FIELDS):
            tk.Label(form, text=label, width=28, anchor="w").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            if field_type == "list_int":
                initial = ",".join(str(x) for x in default_value)
            elif field_type == "list_str":
                initial = ",".join(str(x) for x in default_value)
            else:
                initial = str(default_value)
            var = tk.StringVar(value=initial)
            if key in {"dataDir", "output_root_dir", "project_name"}:
                var.trace_add("write", lambda *_args: self.refresh_input_defaults())
            entry = tk.Entry(form, textvariable=var, width=96)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            if key in self.auto_input_keys:
                entry.bind("<KeyRelease>", lambda _e, k=key: self.mark_input_override(k))
            self.config_vars[key] = (var, field_type)
            if field_type in {"dir", "file"}:
                button = tk.Button(
                    form,
                    text="Browse",
                    width=8,
                    command=lambda k=key: self.browse_path(k),
                )
                button.grid(row=row, column=2, sticky="w", padx=(8, 0), pady=3)
        form.grid_columnconfigure(1, weight=1)

        steps_frame = tk.LabelFrame(self.scrollable_frame, text="Analysis Configuration", padx=10, pady=8)
        steps_frame.pack(fill="x", padx=12, pady=6)

        top_btn_frame = tk.Frame(steps_frame)
        top_btn_frame.pack(fill="x", pady=(0, 8))
        tk.Button(top_btn_frame, text="Select All", command=self.select_all, width=10).pack(side="left", padx=(0, 8))
        tk.Button(top_btn_frame, text="Clear", command=self.clear_all, width=10).pack(side="left")

        checkbox_frame = tk.Frame(steps_frame)
        checkbox_frame.pack(fill="x")
        checkbox_frame.grid_columnconfigure(0, weight=1)

        main_options_frame = tk.LabelFrame(checkbox_frame, text="Analysis Modes", padx=8, pady=6)
        main_options_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=4)

        self.step_vars["1full"] = tk.BooleanVar(value=False)
        caver_full_cb = tk.Checkbutton(
            main_options_frame,
            text=STEP_LABELS["1full"],
            variable=self.step_vars["1full"],
            command=lambda: self.on_caver_mode_toggle("1full"),
            anchor="w",
            justify="left",
            wraplength=860,
            font=("Microsoft YaHei UI", 10),
        )
        caver_full_cb.pack(fill="x", anchor="w", pady=(4, 0))
        self.checkbuttons.append(caver_full_cb)

        self.step_vars["min"] = tk.BooleanVar(value=False)
        minimal_pipeline_cb = tk.Checkbutton(
            main_options_frame,
            text=STEP_LABELS["min"],
            variable=self.step_vars["min"],
            anchor="w",
            justify="left",
            wraplength=860,
            font=("Microsoft YaHei UI", 10),
        )
        minimal_pipeline_cb.pack(fill="x", anchor="w", pady=(4, 0))
        self.checkbuttons.append(minimal_pipeline_cb)

        self.caver_extra_frame = tk.Frame(main_options_frame)
        caver_params_frame = tk.LabelFrame(self.caver_extra_frame, text="CAVER Computation Parameters", padx=8, pady=6)
        caver_params_frame.pack(fill="x", padx=(12, 0), pady=(6, 4))

        param_row = 0
        for key, label, field_type, default_value in CAVER_EXTRA_FIELDS:
            if field_type == "bool":
                var = tk.BooleanVar(value=bool(default_value))
            else:
                tk.Label(caver_params_frame, text=label, width=24, anchor="w").grid(
                    row=param_row, column=0, sticky="w", padx=(0, 8), pady=3
                )
                var = tk.StringVar(value=str(default_value))
                entry = tk.Entry(caver_params_frame, textvariable=var, width=34)
                entry.grid(row=param_row, column=1, sticky="ew", pady=3)
                if field_type in {"dir", "file"}:
                    button = tk.Button(
                        caver_params_frame,
                        text="Browse",
                        width=6,
                        command=lambda k=key: self.browse_path(k),
                    )
                    button.grid(row=param_row, column=2, sticky="w", padx=(8, 0), pady=3)
                param_row += 1
            self.caver_extra_vars[key] = (var, field_type)
        caver_params_frame.grid_columnconfigure(1, weight=1)

        action_frame = tk.Frame(self.scrollable_frame)
        action_frame.pack(fill="x", padx=12, pady=6)
        self.run_button = tk.Button(
            action_frame,
            text="Save Parameters and Run Selected Analysis",
            command=self.start_run,
            height=2,
        )
        self.run_button.pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(action_frame, textvariable=self.status_var, anchor="w").pack(side="left", padx=12)

        log_frame = tk.LabelFrame(self.scrollable_frame, text="Analysis Log", padx=8, pady=8)
        log_frame.pack(fill="both", expand=True, padx=12, pady=(6, 12))
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            height=22,
        )
        self.log_text.pack(fill="both", expand=True)
        self.toggle_caver_extra_fields()

    def update_window_height_limit(self):
        self.root.update_idletasks()
        content_height = self.scrollable_frame.winfo_reqheight()
        screen_width = self.root.winfo_screenwidth()
        window_width = self.root.winfo_width() if self.root.winfo_width() > 1 else self.default_window_width
        window_height = self.root.winfo_height() if self.root.winfo_height() > 1 else self.default_window_height
        chrome_height = max(48, self.root.winfo_height() - self.scrollable_frame.winfo_height())
        max_height = min(content_height + chrome_height + 8, self.root.winfo_screenheight() - 72)
        min_width = min(960, screen_width)
        min_height = min(720, max_height)
        current_width = max(window_width, min_width)
        current_height = min(max(window_height, min_height), max_height)
        self.root.minsize(min_width, min_height)
        self.root.maxsize(screen_width, max_height)
        self.root.geometry(f"{current_width}x{current_height}")

    def get_minimal_config(self):
        config = {}
        for key, (var, field_type) in self.config_vars.items():
            raw_value = var.get().strip()
            if field_type == "list_int":
                config[key] = [int(item.strip()) for item in raw_value.split(",") if item.strip()]
            elif field_type == "list_str":
                config[key] = [item.strip() for item in raw_value.split(",") if item.strip()]
            else:
                config[key] = raw_value
        for key, (var, field_type) in self.caver_extra_vars.items():
            if field_type == "bool":
                config[key] = bool(var.get())
            else:
                raw_value = var.get().strip()
                if field_type == "int":
                    config[key] = int(raw_value)
                else:
                    config[key] = raw_value
        return config

    def browse_path(self, key: str):
        if key in self.config_vars:
            var, field_type = self.config_vars[key]
        else:
            var, field_type = self.caver_extra_vars[key]

        current_value = var.get().strip()
        initial_dir = None
        if current_value:
            current_path = Path(current_value)
            initial_dir = str(current_path.parent if current_path.is_file() else current_path)
        else:
            initial_dir = str(BASE_DIR)

        if field_type == "dir":
            selected = filedialog.askdirectory(initialdir=initial_dir, title="Select Directory")
        else:
            selected = filedialog.askopenfilename(initialdir=initial_dir, title="Select File")

        if selected:
            if key in self.auto_input_keys:
                self.mark_input_override(key)
            var.set(selected)

    def update_preview(self, *_args):
        return

    def clear_preview_overrides(self):
        return

    def refresh_all_derived_paths(self):
        return

    def get_runtime_config(self):
        return derive_config(self.get_minimal_config())

    def set_input_value(self, key: str, value: str):
        target = None
        if key in self.config_vars:
            target = self.config_vars[key]
        elif key in self.caver_extra_vars:
            target = self.caver_extra_vars[key]
        if target is None:
            return
        var, _field_type = target
        self._syncing_input_defaults = True
        var.set(str(value))
        self._syncing_input_defaults = False

    def mark_input_override(self, key: str):
        if self._syncing_input_defaults:
            return
        self.manual_input_override_keys.add(key)

    def refresh_input_defaults(self):
        try:
            minimal = self.get_minimal_config()
        except Exception:
            return
        default_seed = dict(minimal)
        for key in self.auto_input_keys:
            default_seed[key] = ""
        derived = derive_config(default_seed)
        for key in self.auto_input_keys:
            current_var, _field_type = self.config_vars[key]
            current_value = current_var.get().strip()
            if key not in self.manual_input_override_keys or not current_value:
                self.set_input_value(key, str(derived.get(key, "")))

    def update_config_assignments(self, replacements: dict):
        text = CONFIG_FILE.read_text(encoding="utf-8")
        for key, value in replacements.items():
            pattern = re.compile(rf"^{re.escape(key)}\s*=\s*.*$", re.MULTILINE)
            if pattern.search(text):
                text = pattern.sub(lambda _m, k=key, v=value: f"{k} = {v}", text, count=1)
            else:
                if text and not text.endswith("\n"):
                    text += "\n"
                text += f"{key} = {value}\n"
        CONFIG_FILE.write_text(text, encoding="utf-8")

    def save_config(self):
        try:
            minimal = self.get_minimal_config()
            derived = self.get_runtime_config()
        except Exception as exc:
            messagebox.showerror("Parameter Error", str(exc))
            return

        replacements = {
            "dataDir": format_env_literal("str", minimal["dataDir"]),
            "output_root_dir": format_env_literal("str", derived["output_root_dir"]),
            "project_name": format_env_literal("str", derived["project_name"]),
            "rmsd_workers": format_env_literal("int", derived["rmsd_workers"]),
            "pipeline_cpu_workers": format_env_literal("int", derived["pipeline_cpu_workers"]),
            "excluded_residues": format_env_literal("list_int", minimal["excluded_residues"]),
            "mutations": format_env_literal("list_str", minimal["mutations"]),
            "caver_n_parts": format_env_literal("int", minimal["caver_n_parts"]),
            "caver_jobs": format_env_literal("int", minimal["caver_jobs"]),
            "caver_xmx": format_env_literal("str", minimal["caver_xmx"]),
            "caver_conf": format_env_literal("str", minimal["caver_conf"]),
            "caver_home_dir": format_env_literal("str", minimal["caver_home_dir"]),
            "caver_jar_path": format_env_literal("str", derived["caver_jar_path"]),
            "caver_run_stage1": format_env_literal("str", derived["caver_run_stage1"]),
            "caver_run_stage2": format_env_literal("str", derived["caver_run_stage2"]),
            "caver_run_stage3": format_env_literal("str", derived["caver_run_stage3"]),
            "caver_run_stage4": format_env_literal("str", derived["caver_run_stage4"]),
            "name": format_env_literal("str", derived["name"]),
            "project_root": "Path(output_root_dir)",
            "run_root": "project_root / name",
            "caver_run_dir": "run_root / \"01_caver\"",
            "tunnel_profile_csv": format_env_literal("str", derived["tunnel_profile_csv"]),
            "rmsd_run_dir": "run_root / \"02_reference\"",
            "PDB_path": format_env_literal("str", derived["PDB_path"]),
            "four_atoms_run_dir": "run_root / \"03_topology\"",
            "four_atoms_csv": path_to_env_literal(derived["four_atoms_csv"]),
            "path_points_run_dir": "run_root / \"04_path_points\"",
            "all_points_csv": path_to_env_literal(derived["all_points_csv"]),
            "preprocess_run_dir": "run_root",
            "residue_stats_csv": format_env_literal("str", derived["residue_stats_csv"]),
            "residue_combination_stats_csv": format_env_literal("str", derived["residue_combination_stats_csv"]),
            "residue_distance_relation_csv": format_env_literal("str", derived["residue_distance_relation_csv"]),
            "residue_pair_distance_csv": format_env_literal("str", derived["residue_pair_distance_csv"]),
            "residue_pair_path_presence_stats_csv": format_env_literal("str", derived["residue_pair_path_presence_stats_csv"]),
            "frame_index_txt": format_env_literal("str", derived["frame_index_txt"]),
            "caver_input_dir": format_env_literal("str", derived["caver_input_dir"]),
            "caver_out_root": format_env_literal("str", derived["caver_out_root"]),
            "rmsd_root_dir": format_env_literal("str", derived["rmsd_root_dir"]),
            "rmsd_output_dir": format_env_literal("str", derived["rmsd_output_dir"]),
            "preprocess_pkl": format_env_literal("str", derived["preprocess_pkl"]),
            "timing_summary_file": format_env_literal("str", derived["timing_summary_file"]),
        }
        self.update_config_assignments(replacements)
        messagebox.showinfo("Saved", "Input parameters and runtime configuration were written to pipeline_config.py.")

    def select_all(self):
        for step_id, var in self.step_vars.items():
            var.set(step_id in {"1full", "min"})
        self.toggle_caver_extra_fields()

    def clear_all(self):
        for var in self.step_vars.values():
            var.set(False)
        self.toggle_caver_extra_fields()

    def toggle_pipeline_children(self):
        return

    def sync_pipeline_parent(self):
        return

    def on_caver_mode_toggle(self, selected_step_id: str):
        self.toggle_caver_extra_fields()

    def toggle_caver_extra_fields(self):
        if self.step_vars["1full"].get():
            self.caver_extra_frame.pack(fill="x", pady=(6, 0))
        else:
            self.caver_extra_frame.pack_forget()
        self.root.after_idle(self.update_window_height_limit)

    def enqueue_log(self, message: str):
        self.log_queue.put(message)

    def _poll_log_queue(self):
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        self.root.after(100, self._poll_log_queue)

    def set_running(self, running: bool):
        self.running = running
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        for cb in self.checkbuttons:
            cb.configure(state=state)

    def selected_steps(self):
        selected = []
        if self.step_vars.get("1full") is not None and self.step_vars["1full"].get():
            selected.extend(["1full", "1merge"])
        if self.step_vars.get("min") is not None and self.step_vars["min"].get():
            selected.extend(["2", "3", "fast", "7"])
        return selected

    def resolve_caver_step_settings(self, step_id: str, config: dict):
        settings = {
            "stage1_stop_after": "tunnels",
            "run_stage1": config["caver_run_stage1"],
            "run_stage2": config["caver_run_stage2"],
            "run_stage3": config["caver_run_stage3"],
            "run_stage4": config["caver_run_stage4"],
        }
        if step_id == "1full":
            settings.update({
                "stage1_stop_after": "never",
                "run_stage1": "yes",
                "run_stage2": "no",
                "run_stage3": "no",
                "run_stage4": "no",
            })
        return settings

    def build_command(self, step_id: str, config: dict):
        if step_id in {"1", "1full"}:
            settings = self.resolve_caver_step_settings(step_id, config)
            script_path = BASE_DIR / config["caver_script"]
            command = [
                sys.executable,
                str(script_path),
                "--input-dir", to_abs_path(config["caver_input_dir"]),
                "--conf", to_abs_path(config["caver_runtime_conf"] or config["caver_conf"]),
                "--n-parts", str(config["caver_n_parts"]),
                "--out-root", to_abs_path(config["caver_out_root"]),
                "--run-stage1", settings["run_stage1"],
                "--run-stage2", settings["run_stage2"],
                "--run-stage3", settings["run_stage3"],
                "--run-stage4", settings["run_stage4"],
                "--stage1-stop-after", settings["stage1_stop_after"],
                "--jobs", str(config["caver_jobs"]),
                "--xmx", str(config["caver_xmx"]),
                "--home-dir", to_abs_path(config["caver_home_dir"]),
                "--stage2-load-tunnels", config["caver_stage2_load_tunnels"],
                "--stage3-out-csv", config["caver_stage3_out_csv"],
                "--stage4-out-dir", config["caver_stage4_out_dir"],
            ]
            if str(config.get("caver_jar_path", "")).strip():
                command.extend(["--caver-jar", to_abs_path(str(config["caver_jar_path"]).strip())])
            if config["caver_overwrite"]:
                command.append("--overwrite")
            return script_path, command

        if step_id == "1merge":
            script_path = BASE_DIR / config["caver_script"]
            command = [
                sys.executable,
                str(script_path),
                "--input-dir", to_abs_path(config["caver_input_dir"]),
                "--conf", to_abs_path(config["caver_conf"]),
                "--n-parts", str(config["caver_n_parts"]),
                "--out-root", to_abs_path(config["caver_out_root"]),
                "--run-stage1", "no",
                "--run-stage2", "no",
                "--run-stage3", "yes",
                "--run-stage4", "no",
                "--stage3-source", "stage1",
                "--stage3-out-csv", config["caver_stage3_out_csv"],
                "--jobs", str(config["caver_jobs"]),
                "--xmx", str(config["caver_xmx"]),
                "--home-dir", to_abs_path(config["caver_home_dir"]),
            ]
            if str(config.get("caver_jar_path", "")).strip():
                command.extend(["--caver-jar", to_abs_path(str(config["caver_jar_path"]).strip())])
            if config["caver_overwrite"]:
                command.append("--overwrite")
            return script_path, command

        if step_id == "2":
            script_path = BASE_DIR / STEP_SCRIPTS[step_id]
            command = [
                sys.executable,
                str(script_path),
                "--root-dir", config["rmsd_root_dir"],
                "--target-folders", ",".join(config["rmsd_target_folders"]),
                "--output-dir", config["rmsd_output_dir"],
                "--workers", str(config["rmsd_workers"]),
                "--chunk-size", str(config["rmsd_chunk_size"]),
                "--use-memory-cache", "yes" if config["rmsd_use_memory_cache"] else "no",
            ]
            return script_path, command

        if step_id == "3d":
            script_path = BASE_DIR / STEP_SCRIPTS[step_id]
            command = [
                sys.executable,
                str(script_path),
                "--residue-distance-only",
                "--combination-stats-csv", to_abs_path(config["residue_combination_stats_csv"]),
                "--residue-distance-output-csv", to_abs_path(config["residue_distance_relation_csv"]),
                "--residue-pair-output-csv", to_abs_path(config["residue_pair_distance_csv"]),
                "--path-presence-stats-output-csv", to_abs_path(config["residue_pair_path_presence_stats_csv"]),
                "--atom-network-csv", to_abs_path(config["four_atoms_csv"]),
                "--pdb-root", to_abs_path(config["dataDir"]),
            ]
            return script_path, command

        if step_id == "fast":
            script_path = BASE_DIR / STEP_SCRIPTS[step_id]
            command = [
                sys.executable,
                str(script_path),
                "--atom-network-csv", to_abs_path(config["four_atoms_csv"]),
                "--pdb", to_abs_path(config["PDB_path"]),
                "--out-csv", to_abs_path(config["all_points_csv"]),
            ]
            return script_path, command

        if step_id == "7":
            script_path = BASE_DIR / STEP_SCRIPTS[step_id]
            command = [
                sys.executable,
                str(script_path),
                "--input-csv", to_abs_path(config["all_points_csv"]),
                "--output-pkl", to_abs_path(config["preprocess_pkl"]),
            ]
            return script_path, command

        script_path = BASE_DIR / STEP_SCRIPTS[step_id]
        command = [sys.executable, str(script_path)]
        if step_id == "3":
            direct_all_points_output = to_abs_path(config["all_points_csv"]) if config.get("direct_03_to_07") else ""
            command.extend([
                "--tunnel-profile-csv", to_abs_path(config["tunnel_profile_csv"]),
                "--pdb-root", to_abs_path(config["dataDir"]),
                "--atom-network-output-csv", to_abs_path(config["four_atoms_csv"]),
                "--representative-pdb", to_abs_path(config["PDB_path"]),
                "--all-points-output-csv", direct_all_points_output,
                "--residue-stats-output-csv", to_abs_path(config["residue_stats_csv"]),
                "--combination-stats-output-csv", to_abs_path(config["residue_combination_stats_csv"]),
                "--residue-distance-output-csv", to_abs_path(config["residue_distance_relation_csv"]),
                "--residue-pair-output-csv", to_abs_path(config["residue_pair_distance_csv"]),
                "--path-presence-stats-output-csv", to_abs_path(config["residue_pair_path_presence_stats_csv"]),
            ])
        if step_id == "3" and config.get("skip_residue_distance"):
            command.append("--skip-residue-distance")
        return script_path, command

    def sync_tunnel_profile_from_caver(self, config):
        out_root = Path(config["caver_out_root"])
        if not out_root.exists():
            self.enqueue_log(f"[WARN] CAVER output directory does not exist; tunnel_profile_csv was not updated: {out_root}")
            return
        candidates = list(out_root.rglob(config["caver_stage3_out_csv"]))
        if not candidates:
            self.enqueue_log(f"[WARN] CAVER output CSV was not found: {config['caver_stage3_out_csv']}")
            return
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = candidates[0].resolve()
        config["tunnel_profile_csv"] = str(chosen)
        if not config.get("fast_temp_root"):
            self.set_input_value("tunnel_profile_csv", str(chosen))
        self.enqueue_log(f"[SYNC] tunnel_profile_csv -> {chosen}")

    def sync_pdb_path_from_rmsd(self, config):
        output_root = Path(config["rmsd_output_dir"])
        best_result = output_root / Path(config["rmsd_root_dir"]).name / "best_result.txt"
        if not best_result.exists():
            self.enqueue_log(f"[WARN] best_result.txt was not found: {best_result}")
            return
        best_file, best_frame_id = parse_best_result(best_result)
        if not best_file:
            self.enqueue_log(f"[WARN] BestFile was not found in best_result.txt: {best_result}")
            return
        best_pdb_path = (best_result.parent / best_file).resolve()
        representative_path = Path(config["PDB_path"])
        representative_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_pdb_path, representative_path)
        config["PDB_path"] = str(representative_path.resolve())
        if not config.get("fast_temp_root"):
            self.set_input_value("PDB_path", str(representative_path.resolve()))
        self.enqueue_log(f"[SYNC] PDB_path -> {representative_path.resolve()} (frame_id={best_frame_id})")

    def import_existing_representative_frame(self, config: dict, source_path: Path, target_path: Path) -> float:
        start = time.perf_counter()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if source_path != target_path:
            shutil.copy2(source_path, target_path)
            self.enqueue_log(f"[SYNC] Imported the reference conformation into the RMSD output directory: {target_path}")
            self.enqueue_log(f"[CFG] source: {source_path}")
        else:
            self.enqueue_log(f"[CFG] Reusing the existing reference conformation in the RMSD output directory: {target_path}")

        resolved_target = str(target_path.resolve())
        config["PDB_path"] = resolved_target
        self.set_input_value("PDB_path", resolved_target)
        self.update_env_assignments({"PDB_path": format_env_literal("str", resolved_target)})
        self.enqueue_log(f"[SYNC] PDB_path -> {resolved_target}")
        return time.perf_counter() - start

    def write_timing_log(self, timings, total_elapsed, config):
        log_path = Path(config["timing_summary_file"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        timestamped_log_path = log_path.with_name(f"{log_path.stem}_{timestamp}{log_path.suffix}")

        content_lines = ["========== Timing Summary ==========\n"]
        for step_id, step_label, elapsed in timings:
            content_lines.append(f"{step_label:<30} {format_seconds(elapsed)}\n")
        content_lines.append("------------------------------------\n")
        content_lines.append(f"{'Total':<35} {format_seconds(total_elapsed)}\n")
        content = "".join(content_lines)

        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(content)
        with open(timestamped_log_path, "w", encoding="utf-8") as handle:
            handle.write(content)

        self.enqueue_log(f"[LOG] Canonical timing log: {log_path}")
        self.enqueue_log(f"[LOG] Timestamped timing log: {timestamped_log_path}")

    def emit_timing_summary(self, timings, total_elapsed, config):
        self.enqueue_log("")
        self.enqueue_log("========== Timing Summary ==========")
        for _step_id, step_label, elapsed in timings:
            self.enqueue_log(f"{step_label:<30} {format_seconds(elapsed)}")
        self.enqueue_log("------------------------------------")
        self.enqueue_log(f"{'Total':<35} {format_seconds(total_elapsed)}")
        self.write_timing_log(timings, total_elapsed, config)

    def build_step_env(self, step_id: str, config: dict):
        env = dict(os.environ)
        cpu_workers = max(1, int(config.get("pipeline_cpu_workers", config.get("rmsd_workers", 1))))
        if step_id in {"2", "3", "fast", "7"}:
            env["PATHVIS_MAX_WORKERS"] = str(cpu_workers)
            env["PATHVIS_PREPROCESS_WORKERS"] = str(cpu_workers)
            env["PATHVIS_RESIDUE_DISTANCE_FRAME_WORKERS"] = str(cpu_workers)
            env["PATHVIS_KD_QUERY_WORKERS"] = "1" if cpu_workers > 1 else "-1"
            env["PATHVIS_CUMULATIVE_DISTANCE_CSV_MODE"] = "never"
        if config.get("fast_temp_root"):
            env["PATHVIS_WRITE_RESIDUE_DISTANCE_RELATION_CSV"] = "0"
        if step_id == "7" and config.get("pathvis_prepare_dataset_artifacts") is not None:
            env["PATHVIS_PREPARE_DATASET_ARTIFACTS"] = str(config["pathvis_prepare_dataset_artifacts"])
        return env

    def cleanup_fast_pipeline_intermediates(self, config: dict):
        targets = [
            Path(config["rmsd_output_dir"]),
            Path(config["four_atoms_csv"]).parent,
            Path(config["all_points_csv"]).parent,
            Path(config["residue_distance_relation_csv"]),
            Path(config["residue_pair_distance_csv"]),
            Path(config["tunnel_profile_csv"]),
            Path(config["caver_out_root"]) / config["caver_stage3_out_csv"],
        ]
        removed = []
        for target in targets:
            if not target.exists():
                continue
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                removed.append(str(target))
            except OSError as exc:
                self.enqueue_log(f"[WARN] Failed to remove intermediate artifact: {target} ({exc})")

        npy_path = Path(config["residue_pair_distance_csv"]).with_suffix(".npy")
        if not npy_path.exists():
            self.enqueue_log(f"[WARN] Final NPY file was not found: {npy_path}")
        if removed:
            self.enqueue_log("[CLEAN] Removed intermediate artifacts:")
            for path_text in removed:
                self.enqueue_log(f"  - {path_text}")

    def cleanup_fast_root_intermediates(self, config: dict):
        targets = [
            Path(config["rmsd_output_dir"]),
            Path(config["four_atoms_csv"]).parent,
            Path(config["all_points_csv"]).parent,
            Path(config["residue_distance_relation_csv"]),
            Path(config["residue_pair_distance_csv"]),
            Path(config["preprocess_pkl"]),
            Path(config["timing_summary_file"]),
        ]
        removed = []
        for target in targets:
            if not target.exists():
                continue
            try:
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                removed.append(str(target))
            except OSError as exc:
                self.enqueue_log(f"[WARN] Failed to remove legacy intermediate artifact from the dataset root: {target} ({exc})")
        if removed:
            self.enqueue_log("[CLEAN] Removed legacy intermediate artifacts from the dataset root:")
            for path_text in removed:
                self.enqueue_log(f"  - {path_text}")

    def finalize_fast_pipeline_outputs(self, config: dict, final_config: dict | None = None):
        temp_pair_npy = Path(config["residue_pair_distance_csv"]).with_suffix(".npy")
        temp_pkl_path = Path(config["preprocess_pkl"])
        temp_pdb_path = Path(config["PDB_path"])
        final_root = Path(final_config["run_root"]) if final_config is not None else Path(config["run_root"])
        final_pair_npy = final_root / "residue_pair_distances.npy"
        final_db_path = final_root / "preprocessed_paths.db"
        final_pkl_path = final_root / "preprocessed_paths.pkl"
        final_pdb_path = final_root / "representative_frame.pdb"
        if not temp_pair_npy.is_file():
            raise FileNotFoundError(f"Fast pipeline missing final NPY: {temp_pair_npy}")
        if not temp_pkl_path.is_file():
            raise FileNotFoundError(f"Fast pipeline missing final PKL: {temp_pkl_path}")
        final_pair_npy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_pair_npy, final_pair_npy)
        self.enqueue_log(f"[SYNC] residue_pair_distances.npy -> {final_pair_npy.resolve()}")
        if not temp_pdb_path.is_file():
            raise FileNotFoundError(f"Fast pipeline missing representative PDB: {temp_pdb_path}")
        final_pdb_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temp_pdb_path, final_pdb_path)
        convert_pkl_to_sqlite(temp_pkl_path, final_db_path)
        self.enqueue_log(f"[SYNC] preprocessed_paths.db -> {final_db_path.resolve()}")
        if final_pkl_path.exists():
            try:
                final_pkl_path.unlink()
                self.enqueue_log(f"[CLEAN] Removed legacy PKL file: {final_pkl_path.resolve()}")
            except OSError as exc:
                self.enqueue_log(f"[WARN] Failed to remove legacy PKL file: {final_pkl_path} ({exc})")
        config["PDB_path"] = str(final_pdb_path.resolve())
        if final_config is not None:
            final_config["PDB_path"] = str(final_pdb_path.resolve())
        self.set_input_value("PDB_path", str(final_pdb_path.resolve()))
        self.enqueue_log(f"[SYNC] Reference conformation PDB -> {final_pdb_path.resolve()}")

    def cleanup_fast_temp_dir(self, temp_root: Path):
        if not temp_root.exists():
            return
        try:
            shutil.rmtree(temp_root)
            self.enqueue_log(f"[CLEAN] Removed FAST temporary directory: {temp_root}")
        except OSError as exc:
            self.enqueue_log(f"[WARN] Failed to remove FAST temporary directory: {temp_root} ({exc})")

    def start_run(self):
        if self.running:
            return
        steps = self.selected_steps()
        if not steps:
            messagebox.showwarning("No Analysis Selected", "Select at least one analysis mode.")
            return
        try:
            self.save_config()
            config = self.get_runtime_config()
        except Exception as exc:
            messagebox.showerror("Parameter Error", str(exc))
            return

        self.set_running(True)
        self.status_var.set("Running...")
        self.enqueue_log("=" * 70)
        self.enqueue_log("Starting selected analysis:")
        for step_id in steps:
            self.enqueue_log(f"  {STEP_LABELS[step_id]}")
        self.enqueue_log("=" * 70)
        self.run_thread = threading.Thread(target=self.run_pipeline, args=(steps, config), daemon=True)
        self.run_thread.start()

    def run_pipeline(self, steps, config):
        timings = []
        total_start = time.perf_counter()
        fast_temp_root = None
        success = False
        cleanup_root_config = None
        try:
            minimal_mode = self.step_vars.get("min") is not None and self.step_vars["min"].get()
            if minimal_mode:
                config["pathvis_prepare_dataset_artifacts"] = "0"
                config["skip_residue_distance"] = False
                config["direct_03_to_07"] = False
                cleanup_root_config = dict(config)
                config, fast_temp_root = build_fast_temp_config(config)
                self.enqueue_log(f"[CFG] FAST temporary directory: {fast_temp_root}")
            for step_id in steps:
                elapsed = self.run_step(step_id, config)
                timings.append((step_id, STEP_LABELS[step_id], elapsed))
            if minimal_mode:
                if cleanup_root_config is not None:
                    self.cleanup_fast_root_intermediates(cleanup_root_config)
                self.finalize_fast_pipeline_outputs(config, cleanup_root_config)
            total_elapsed = time.perf_counter() - total_start
            self.emit_timing_summary(timings, total_elapsed, config)
            self.status_var.set("Completed")
            self.enqueue_log("[DONE] Selected steps finished successfully.")
            success = True
        except Exception as exc:
            self.status_var.set("Failed")
            total_elapsed = time.perf_counter() - total_start
            if timings:
                self.emit_timing_summary(timings, total_elapsed, config)
            self.enqueue_log(f"[FAILED] {exc}")
            self.root.after(0, lambda: messagebox.showerror("Analysis Failed", str(exc)))
        finally:
            if fast_temp_root is not None and success:
                self.cleanup_fast_temp_dir(Path(fast_temp_root))
            self.root.after(0, lambda: self.set_running(False))

    def run_step(self, step_id: str, config: dict) -> float:
        if step_id == "2":
            mutations = config.get("mutations") or []
            representative_path = Path(config["PDB_path"])
            selected_reference_path = Path(config.get("input_reference_pdb") or config["PDB_path"])
            expected_path = (Path(config["rmsd_output_dir"]) / "representative_frame.pdb").resolve()
            if not mutations and selected_reference_path.is_file():
                resolved = selected_reference_path.resolve()
                self.enqueue_log("")
                if resolved != expected_path:
                    self.enqueue_log(f"[IMPORT] {STEP_LABELS[step_id]}")
                    return self.import_existing_representative_frame(config, resolved, expected_path)
                self.set_input_value("PDB_path", str(resolved))
                self.enqueue_log(f"[SKIP] {STEP_LABELS[step_id]}")
                self.enqueue_log(f"[CFG] Reusing the existing reference conformation in the RMSD output directory: {resolved}")
                return 0.0

        caver_settings = None
        if step_id in {"1", "1full"}:
            caver_settings = self.resolve_caver_step_settings(step_id, config)
            config["caver_stage1_stop_after"] = caver_settings["stage1_stop_after"]
            config["caver_runtime_conf"] = prepare_caver_runtime_config(config)
        script_path, command = self.build_command(step_id, config)
        if not script_path.exists():
            raise FileNotFoundError(f"{script_path.name} not found")
        self.enqueue_log("")
        self.enqueue_log(f"[RUN] {STEP_LABELS[step_id]}")
        if step_id in {"1", "1full"}:
            self.enqueue_log(f"[CFG] conf: {config['caver_conf']}")
            self.enqueue_log(f"[CFG] runtime_conf: {config['caver_runtime_conf']}")
            self.enqueue_log(f"[CFG] caver_home_dir: {config['caver_home_dir']}")
            self.enqueue_log(f"[CFG] resolved_caver_jar: {config['caver_jar_path']}")
            self.enqueue_log(f"[CFG] npart/jobs/xmx: {config['caver_n_parts']}/{config['caver_jobs']}/{config['caver_xmx']}")
            self.enqueue_log(f"[CFG] stage1_stop_after: {caver_settings['stage1_stop_after']}")
            self.enqueue_log(
                f"[CFG] stages: {caver_settings['run_stage1']}/{caver_settings['run_stage2']}/"
                f"{caver_settings['run_stage3']}/{caver_settings['run_stage4']}"
            )
        elif step_id == "1merge":
            self.enqueue_log(f"[CFG] conf: {config['caver_conf']}")
            self.enqueue_log(f"[CFG] caver_home_dir: {config['caver_home_dir']}")
            self.enqueue_log(f"[CFG] resolved_caver_jar: {config['caver_jar_path']}")
            self.enqueue_log(f"[CFG] npart/jobs/xmx: {config['caver_n_parts']}/{config['caver_jobs']}/{config['caver_xmx']}")
            self.enqueue_log(f"[CFG] stage3_source: stage1")
            self.enqueue_log(f"[CFG] stage3_out_csv: {config['caver_stage3_out_csv']}")
        self.enqueue_log("Command:")
        self.enqueue_log("  " + " ".join(command))
        start = time.perf_counter()
        process_cwd = BASE_DIR
        if step_id in {"1", "1full", "1merge"} and str(config.get("caver_home_dir", "")).strip():
            process_cwd = Path(to_abs_path(config["caver_home_dir"]))
            self.enqueue_log(f"[CFG] working_dir: {process_cwd}")
        process = subprocess.Popen(
            command,
            cwd=process_cwd,
            env=self.build_step_env(step_id, config),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
        assert process.stdout is not None
        for line in process.stdout:
            self.enqueue_log(line.rstrip())
        return_code = process.wait()
        elapsed = time.perf_counter() - start
        if return_code != 0:
            raise RuntimeError(f"{script_path.name} exited with code {return_code} after {format_seconds(elapsed)}")
        if step_id in {"1", "1full"}:
            if caver_settings["run_stage3"] == "yes":
                self.sync_tunnel_profile_from_caver(config)
        elif step_id == "1merge":
            self.sync_tunnel_profile_from_caver(config)
        elif step_id == "2":
            self.sync_pdb_path_from_rmsd(config)
        self.enqueue_log(f"[OK] {STEP_LABELS[step_id]} | Time: {format_seconds(elapsed)}")
        return elapsed


def main():
    root = tk.Tk()
    RunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
