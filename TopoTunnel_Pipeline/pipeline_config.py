from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Values shown when the GUI starts. The GUI updates this file when parameters
# are saved, while all generated analysis paths are derived from these inputs.
dataDir = ""
output_root_dir = str(BASE_DIR / "output")
project_name = "dataset"
pipeline_cpu_workers = 4
excluded_residues = []
mutations = []

name = project_name
project_root = Path(output_root_dir)
run_root = project_root / name

caver_run_dir = run_root / "01_caver"
tunnel_profile_csv = str(caver_run_dir / "all_tunnel_profiles.csv")

rmsd_run_dir = run_root / "02_reference"
PDB_path = str(run_root / "representative_frame.pdb")

four_atoms_run_dir = run_root / "03_topology"
four_atoms_csv = str(four_atoms_run_dir / "atom_network.csv")

path_points_run_dir = run_root / "04_path_points"
all_points_csv = str(path_points_run_dir / "mapped_path_points.csv")

preprocess_run_dir = run_root
preprocess_pkl = str(run_root / "preprocessed_paths.pkl")
residue_stats_csv = str(run_root / "residue_statistics.csv")
residue_combination_stats_csv = str(run_root / "residue_combination_statistics.csv")
residue_distance_relation_csv = str(run_root / "residue_distance_relations.csv")
residue_pair_distance_csv = str(run_root / "residue_pair_distances.csv")
residue_pair_path_presence_stats_csv = str(run_root / "residue_pair_path_presence_stats.csv")
frame_index_txt = str(run_root / "frame_index.txt")
timing_summary_file = str(run_root / "timing_summary.txt")

# CAVER settings. Protein-specific paths must be selected in the GUI before
# running the segmented CAVER computation.
caver_script = "stage_01_run_caver.py"
caver_input_dir = dataDir
caver_conf = ""
caver_home_dir = ""
caver_jar_path = ""
caver_n_parts = 8
caver_out_root = str(caver_run_dir)
caver_run_stage1 = "yes"
caver_run_stage2 = "no"
caver_run_stage3 = "no"
caver_run_stage4 = "no"
caver_jobs = 4
caver_xmx = "2G"
caver_stage2_load_tunnels = "yes"
caver_stage3_out_csv = "all_tunnel_profiles.csv"
caver_stage4_out_dir = "merged_clusters_timeless"
caver_overwrite = True

# Representative-conformation search settings.
rmsd_root_dir = dataDir
rmsd_target_folders = ["."]
rmsd_output_dir = str(rmsd_run_dir)
rmsd_workers = pipeline_cpu_workers
rmsd_chunk_size = 200
rmsd_use_memory_cache = True
