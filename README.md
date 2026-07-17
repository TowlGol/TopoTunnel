# TopoTunnelMiner

TopoTunnelMiner is a topology-guided visual-analytics workflow for mapping and comparing protein tunnels across molecular-dynamics conformations and datasets. This repository separates data preparation from interactive analysis:

- `TopoTunnelMiner_Pipeline` converts aligned PDB frames and CAVER tunnel profiles into a topology-mapped dataset.
- `TopoTunnelMiner_UI` loads one or more prepared datasets for path, residue, residue-pair, and cross-dataset analysis.
- `CYP3A4` is a complete case study showing the input, CAVER output, and final UI dataset expected by the workflow.

## Repository Structure

```text
Github_Version/
|-- README.md
|-- .gitignore
|-- .gitattributes
|-- CYP3A4/
|   |-- README.md
|   |-- inputs/
|   |   |-- caver_config.txt
|   |   `-- pdb_frames/
|   |-- caver_output/
|   `-- ui_dataset/
|-- TopoTunnelMiner_Pipeline/
|   |-- README.md
|   |-- pipeline_gui.py
|   |-- pipeline_config.py
|   |-- stage_01_run_caver.py
|   |-- stage_02_select_reference.py
|   |-- stage_03_map_topology.py
|   |-- stage_04_build_path_points.py
|   |-- stage_05_preprocess_dataset.py
|   `-- supporting Python modules
`-- TopoTunnelMiner_UI/
    |-- README.md
    |-- __main__.py
    |-- app/
    |-- core/
    |-- db/
    |-- vis/
    `-- resources/
```

### Component Responsibilities

| Component | Role |
|---|---|
| `CYP3A4` | Reproducible example containing 500 aligned conformations, CAVER results, and a ready-to-load TopoTunnelMiner dataset. |
| `TopoTunnelMiner_Pipeline` | Reference-conformation selection, four-atom topology mapping, mapped point construction, residue statistics, path preprocessing, and SQLite generation. It can optionally run segmented CAVER calculations. |
| `TopoTunnelMiner_UI` | Desktop application for tunnel display, path selection, constraint-based matching, synchronized profiles, residue comparison, and residue-pair/motif analysis. |

See each component's own `README.md` for file-level and control-level documentation.

## Installation

Python 3.10 or 3.11 is recommended. CAVER and Java are required only when tunnel profiles must be calculated from PDB frames.

```powershell
cd <path-to-repository>\Github_Version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r TopoTunnelMiner_Pipeline\requirements.txt
python -m pip install -r TopoTunnelMiner_UI\requirements.txt
```

The pipeline and UI may also be installed in separate environments when Qt/VTK dependencies need to be isolated from data-processing dependencies.

## End-to-End CYP3A4 Workflow

### Route A: Use the Existing CAVER Results

This is the shortest reproducible route because the case already includes `caver_output/analysis/tunnel_profiles.csv`.

1. Start the processing interface:

   ```powershell
   cd TopoTunnelMiner_Pipeline
   python pipeline_gui.py
   ```

2. Enter the following values:

   | Field | CYP3A4 value |
   |---|---|
   | Aligned MD Frame Directory | `..\CYP3A4\inputs\pdb_frames` |
   | Output Root Directory | `..\CYP3A4` |
   | Dataset Name | `ui_dataset_generated` |
   | Tunnel-Profile CSV File | `..\CYP3A4\caver_output\analysis\tunnel_profiles.csv` |
   | Reference-Conformation PDB File | `..\CYP3A4\ui_dataset\representative_frame.pdb`, or empty to select it again |
   | CPU Workers | `4` or `8` |
   | Excluded Residues | empty |
   | Mutation Sites | empty |

3. Select only **Reference-Conformation Mapping and Dataset Preparation**.
4. Click **Save Parameters and Run Selected Analysis**.
5. Confirm that `CYP3A4/ui_dataset_generated` contains `preprocessed_paths.db` and the five associated structure/statistics files.

### Route B: Recompute CAVER First

1. Install CAVER 3.x and verify `java -version`.
2. In the pipeline interface, also select **Segmented CAVER Tunnel Computation and Tunnel-Profile Aggregation**.
3. Use `CYP3A4/inputs/caver_config.txt` as the CAVER configuration.
4. A practical starting point for 500 frames is 8 segments, 4 parallel CAVER jobs, and a `2G` Java heap per job.
5. Select the CAVER installation directory containing `caver.jar` and `lib`.
6. Run both selected stages. The aggregated profile is passed automatically to topology mapping.

The complete meaning and accepted format of every parameter are documented in `TopoTunnelMiner_Pipeline/README.md`.

## Load the Result in the UI

Run the application from the repository root, which contains the `TopoTunnelMiner_UI` package:

```powershell
cd <path-to-repository>\Github_Version
python -m TopoTunnelMiner_UI
```

Then:

1. Open the `Datasets` tab.
2. Click `Add Dataset`.
3. Select `CYP3A4/ui_dataset`, or the newly generated `CYP3A4/ui_dataset_generated` directory.
4. Select the dataset and click `Show Tunnels`.
5. Use the `Paths`, `Residue Compare`, and `Residue Combination Compare` tabs for analysis.

The UI README documents every button, mouse interaction, selection mode, matching control, chart, and export option.

## UI Dataset Contract

A complete generated dataset contains:

| File | Purpose |
|---|---|
| `preprocessed_paths.db` | Primary SQLite database containing paths, geometry, residues, and mapped topology. |
| `representative_frame.pdb` | Shared reference conformation used for mapping and protein display. |
| `residue_statistics.csv` | Per-residue path, point, radius, and cooperative-residue summaries. |
| `residue_combination_statistics.csv` | Residue-pair and higher-order bottleneck motif summaries. |
| `residue_pair_distances.npy` | Precomputed residue-pair/combination distance data. |
| `residue_pair_path_presence_stats.csv` | Pair-distance statistics conditioned on path presence. |

Only the database is required for basic tunnel display. The other files enable protein rendering and residue-level analyses.

## Large Files and Data Availability

The CYP3A4 case includes files above GitHub's normal 100 MB object limit, including `preprocessed_paths.db`. Do not commit these files as ordinary Git blobs. This repository includes `.gitattributes` rules for Git LFS:

```powershell
git lfs install
git lfs track "*.db" "*.npy" "*.pdb" "*.obj" "*.dat"
```

For a source-code-only release, omit the large case data and provide the processed experimental archives through the project Zenodo record (`https://zenodo.org/records/21287445`). Keep the directory layout documented here so downloaded data can be placed back without changing commands.

## Notes

- PDB frame names in `tunnel_profiles.csv` must exactly match files in the aligned-frame directory.
- Compatible atom names, atom identities, residue numbers, and chain topology are required across conformations.
- Generated data should remain outside the UI package source. Point `Add Dataset` to the output folder instead.
- Existing datasets that use the former intermediate names `1.atom_network.csv` and `4.all_points.csv` remain readable; new runs use `atom_network.csv` and `mapped_path_points.csv`.
