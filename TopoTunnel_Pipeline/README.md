# TopoTunnel Processing Pipeline

`TopoTunnel_Pipeline` converts molecular-dynamics structures and CAVER tunnel profiles into a dataset that can be loaded directly by `TopoTunnel_UI`.

The pipeline supports two entry points:

1. Run segmented CAVER calculations and aggregate their tunnel profiles.
2. Start from an existing CAVER `tunnel_profiles.csv` file and prepare the topology-mapped UI dataset.

The optional CYP3A4 reproduction data are distributed separately through Zenodo (`https://zenodo.org/records/24049648`). After downloading the deposited CAVER output, you can skip the CAVER stage and run only **Reference-Conformation Mapping and Dataset Preparation**.

## 1. Processing Workflow

```text
MD PDB frames
   |
   +-- optional stage 01: segmented CAVER calculation
   |                         |
   |                         `-- all_tunnel_profiles.csv
   |
existing tunnel_profiles.csv-+
                             |
stage 02: select/reuse reference conformation
                             |
stage 03: map tunnel points to four nearby atoms and residues
                             |
stage 04: construct mapped path-point records
                             |
stage 05: preprocess paths and build summary artifacts
                             |
SQLite conversion            |
                             v
                  UI-loadable dataset folder
```

## 2. File Structure

| File | Purpose |
|---|---|
| `pipeline_gui.py` | Tkinter GUI and workflow orchestrator. This is the normal entry point. |
| `pipeline_config.py` | Saved GUI defaults and derived runtime paths. The GUI updates this file when parameters are saved. |
| `stage_01_run_caver.py` | Splits PDB frames into CAVER jobs, runs them in parallel, and aggregates `tunnel_profiles.csv`. |
| `stage_02_select_reference.py` | Selects the frame closest to the mean N/CA/C backbone conformation when no reference PDB is reused. |
| `stage_03_map_topology.py` | Assigns four surface-gap-ranked atoms to each tunnel point, maps points to the reference conformation, and computes residue-level statistics. |
| `stage_04_build_path_points.py` | Converts `atom_network.csv` directly into `mapped_path_points.csv`. |
| `stage_05_preprocess_dataset.py` | Smooths/preprocesses paths and writes the intermediate `preprocessed_paths.pkl`. |
| `convert_dataset_to_sqlite.py` | Converts the intermediate PKL into final `preprocessed_paths.db`. |
| `mutation_support.py` | Parses mutation specifications and maps mutation-aware residue names. |
| `atomic_properties.py` | Van der Waals radii used by the surface-gap calculation. |
| `residue_properties.py` | Residue hydrophobicity values written to path profiles. |
| `path_simplification.py` | Optional RDP path-simplification helpers. |
| `requirements.txt` | Legacy Pipeline-only dependency list; the repository-root `pyproject.toml` is the source for the unified `uv` environment. |
| `run_gui.bat` | Windows launcher for `pipeline_gui.py`. |
| `.gitignore` | Excludes caches, generated output, and temporary processing directories. |

## 3. Installation

Requirements:

- Target operating systems: Windows, macOS, and Linux. The Pipeline uses Python/Tkinter and invokes the Java-based CAVER command-line tool directly.
- Python 3.10 or 3.11 with Tk support
- Java and CAVER 3.x only when running stage 01

Install Astral's `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). For example:

```powershell
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the repository root, run `uv sync` to create `.venv` and install all Pipeline and UI dependencies specified by the root `pyproject.toml` and `uv.lock`:

```text
uv sync
```

Start the Pipeline GUI from the repository root:

```bash
uv run python TopoTunnel_Pipeline/pipeline_gui.py
```

The Pipeline GUI uses Tkinter. Confirm that the Python interpreter selected by `uv` has Tk support before running the GUI.

On Windows, `run_gui.bat` is also available as a convenience launcher.

## 4. Required Inputs

### 4.1 Aligned MD PDB Frames

PDB frames must be stored directly in one directory. The pipeline does not recursively discover frame files for the main mapping input.

Recommended naming:

```text
aligned_1.pdb
aligned_2.pdb
...
aligned_500.pdb
```

All frames must use compatible atom names, residue numbering, and topology. The PDB names recorded in the CAVER tunnel-profile file must match files in this directory exactly.

The directory may contain only one PDB file for a single-structure analysis. Use a frame-like name such as `aligned_1.pdb` when running the optional CAVER stage. The single-structure workflow still requires a matching CAVER tunnel-profile CSV; a PDB alone is not sufficient to create tunnel paths.

### 4.2 CAVER Tunnel Profiles

Use either:

- `tunnel_profiles.csv` from one completed CAVER calculation; or
- `all_tunnel_profiles.csv` generated by stage 01 from segmented jobs.

The parser expects the native CAVER profile layout: one header row followed by seven rows per tunnel, with sampled path-point fields beginning at column 14.

### 4.3 Reference PDB

The reference PDB must use residue numbering and atom names compatible with the MD frames.

- If a valid reference PDB is supplied and `Mutation Sites` is empty, the pipeline reuses it.
- If no valid reference is supplied, stage 02 selects the MD frame nearest to the mean N/CA/C backbone conformation.
- Mutation specifications alter identity handling but do not build missing side-chain atoms.

## 5. Analysis Input Parameters

### `Aligned MD Frame Directory`

- Select the directory containing the frame-level PDB files.
- The GUI expects a directory even for a single-structure analysis; place the one PDB file in its own directory instead of selecting the file directly.
- External CYP3A4 example: the directory containing the 500 aligned PDB frames downloaded from Zenodo, such as `..\CYP3A4\inputs\pdb_frames` in a local arrangement.

### `Output Root Directory`

- Parent directory for generated dataset folders.
- Result path: `<Output Root>/<Dataset Name>/`.
- For the external CYP3A4 case, use an independent writable output directory or a local ignored data directory such as `..\CYP3A4`.
- Do not point it at `TopoTunnel_UI/database`; keeping generated data outside the application source is safer.

### `Dataset Name`

- Name of the output subdirectory.
- Use letters, numbers, underscores, or hyphens.
- CYP3A4 example: `ui_dataset_generated`.

### `Tunnel-Profile CSV File`

- Existing CAVER profile file used by the mapping stage.
- External CYP3A4 example: the extracted CAVER `analysis\tunnel_profiles.csv`, optionally arranged as `..\CYP3A4\caver_output\analysis\tunnel_profiles.csv`.
- When stage 01 is selected, the GUI updates this input to the newly aggregated `01_caver/all_tunnel_profiles.csv`.

### `Reference-Conformation PDB File`

- Optional explicit reference conformation.
- External CYP3A4 example: the deposited `representative_frame.pdb` for exact reference reuse, optionally arranged as `..\CYP3A4\ui_dataset\representative_frame.pdb`.
- Leave empty to select a reference from the 500 PDB frames.

### `CPU Workers for Mapping and Preprocessing`

- Positive integer controlling reference selection, topology mapping, residue-distance calculations, and preprocessing.
- Recommended starting values:
  - 4 workers for 8 GB RAM or a laptop;
  - 8 workers for 16 GB RAM;
  - physical core count minus one for larger workstations.
- More workers can increase memory and disk contention. Start with 4 or 8 rather than the maximum logical-core count.

### `Excluded Residues`

- Comma-separated residue sequence numbers excluded from the four-atom neighborhood search.
- Example: `1,2,500`.
- Leave empty for the deposited CYP3A4 reproduction. The atom exclusions in the CAVER configuration control CAVER tunnel detection and are separate from this topology-mapping option.
- Chain IDs and ranges such as `1-10` are not supported.

### `Mutation Sites`

- Comma-separated substitutions such as `L176W,F142L` or `LEU176TRP,PHE142LEU`.
- Leave empty for CYP3A4.
- This option validates residue identities and supports mutation-aware mapping. It is not a side-chain modeling tool.

### Single-structure limitations

When the input directory contains one PDB and the profile CSV refers to it, stages 03-05 can map and preprocess that conformation. A reference PDB may be supplied explicitly, or the single PDB can be selected as the representative structure. This mode does not provide a meaningful ensemble-level comparison, and the pipeline does not accept standalone force-field topology files such as PSF, PRMTOP, TOP, or TPR in place of PDB coordinates.

## 6. CAVER Parameters

These fields are needed only when **Segmented CAVER Tunnel Computation and Tunnel-Profile Aggregation** is selected.

### `Number of Segments`

- Number of contiguous, nonoverlapping frame ranges.
- Recommended: one to four times the number of parallel CAVER jobs.
- CYP3A4 starting value: `8` segments for 500 frames.

### `Parallel CAVER Jobs`

- Number of simultaneous Java/CAVER processes.
- CYP3A4 starting value: `4`.
- Ensure that `jobs x Java heap` plus Python and operating-system memory fits in RAM.

### `Java Heap Size (-Xmx)`

- Heap limit per CAVER job, not the total for all jobs.
- CYP3A4 starting value: `2G`.
- With four jobs at `2G`, reserve more than 8 GB because Java, Python, and the operating system need additional memory.

### `CAVER Configuration File`

- Protein-specific CAVER settings.
- External CYP3A4 example: select the downloaded configuration file; one optional local path is `..\CYP3A4\inputs\caver_config.txt`.
- The segmented runner overrides frame-range and stage-control fields in generated runtime copies; it does not modify the source file.

### `CAVER Home Directory`

- Directory containing `caver.jar` and `lib`, or a parent directory from which the JAR can be discovered.
- Verify `java -version` before running.

### CYP3A4 Key CAVER Settings

The CYP3A4 configuration distributed with the Zenodo data uses:

| Parameter | Value |
|---|---|
| Starting region | Heme group `O:500` |
| Excluded atoms | Phe304, Ala305, Thr309, Cys442, and heme atoms, represented by explicit atom serials |
| `probe_radius` | 1.0 A |
| `shell_radius` | 5.0 A |
| `shell_depth` | 4.0 A |
| `clustering` | `average_link` |
| `clustering_threshold` | 3.5 A |
| `max_output_clusters` | 50 |
| `frame_clustering_threshold` | 1.0 A |

The configuration file is authoritative for the complete parameter set.

## 7. Analysis Modes and Buttons

### Analysis Modes

- **Segmented CAVER Tunnel Computation and Tunnel-Profile Aggregation**: runs stage 01 and combines segmented profiles.
- **Reference-Conformation Mapping and Dataset Preparation**: runs stages 02-05 and converts the result to SQLite.
- Selecting both runs CAVER first and immediately feeds the aggregated profile into mapping.

### Buttons

- `Save Input Parameters`: writes current values to `pipeline_config.py`.
- `Select All`: selects both modes.
- `Clear`: clears mode selection.
- `Browse`: selects a file or directory for the corresponding field.
- `Save Parameters and Run Selected Analysis`: validates inputs, saves them, and executes selected stages.

Commands, stage timings, and errors are displayed in `Analysis Log`. Diagnose failures from the first concrete error above `[FAILED]`, not only from the final exit code.

## 8. CYP3A4 Quick Reproduction

The CYP3A4 case is not included in this GitHub repository. Download the relevant structures, CAVER output, and processed data from `https://zenodo.org/records/24049648`, then select their extracted locations in the GUI. If you arrange the downloaded data in a local ignored `CYP3A4` directory beside the code, the shortest path is:

1. Run `python pipeline_gui.py`.
2. Set `Aligned MD Frame Directory` to the extracted directory containing the 500 aligned PDB frames, for example `CYP3A4/inputs/pdb_frames`.
3. Set `Output Root Directory` to `CYP3A4` or another writable directory.
4. Set `Dataset Name` to `ui_dataset_generated`.
5. Set `Tunnel-Profile CSV File` to the extracted CAVER `analysis/tunnel_profiles.csv`, for example `CYP3A4/caver_output/analysis/tunnel_profiles.csv`.
6. Set `Reference-Conformation PDB File` to the deposited `representative_frame.pdb` for exact reuse, or leave it empty to rerun reference selection.
7. Set workers to 4 or 8.
8. Leave `Excluded Residues` and `Mutation Sites` empty.
9. Select only **Reference-Conformation Mapping and Dataset Preparation**.
10. Click `Save Parameters and Run Selected Analysis`.

After completion, load the generated dataset directory with `TopoTunnel_UI -> Datasets -> Add Dataset`.

## 9. Output Structure

```text
<Output Root>/<Dataset Name>/
|-- preprocessed_paths.db
|-- representative_frame.pdb
|-- residue_statistics.csv
|-- residue_combination_statistics.csv
|-- residue_pair_distances.npy
|-- residue_pair_path_presence_stats.csv
|-- 01_caver/                 # only when CAVER is run here
|-- 02_reference/             # temporary/intermediate
|-- 03_topology/              # temporary/intermediate
`-- 04_path_points/           # temporary/intermediate
```

The GUI's fast workflow removes temporary intermediates after successful conversion. The six files at the dataset root are the important UI inputs.

### Final Files

| File | Purpose |
|---|---|
| `preprocessed_paths.db` | Main SQLite path database loaded by the UI. |
| `representative_frame.pdb` | Shared reference conformation used for mapped coordinates. |
| `residue_statistics.csv` | Single-residue path, point, radius, and cooperative-residue summaries. |
| `residue_combination_statistics.csv` | Two-, three-, and four-residue combination statistics. |
| `residue_pair_distances.npy` | Per-frame cumulative pair/combination-distance matrix used for fast comparison. |
| `residue_pair_path_presence_stats.csv` | Path-present/path-absent residue-pair distance statistics. |

## 10. Troubleshooting

### PDB file not found

The PDB name in `tunnel_profiles.csv` must match a file in the aligned-frame directory. Check spelling, extension, and frame numbering.

### CAVER JAR or `lib` not found

Select the CAVER installation directory, not a CAVER output directory. Verify Java is on `PATH`.

### High memory use

Reduce mapping workers. For CAVER, reduce parallel jobs or Java heap. Increasing concurrency does not guarantee faster processing on memory-limited systems or mechanical disks.

### Old values return at startup

The GUI persists values in `pipeline_config.py`. Close the GUI and edit the input values near the top of that file, or save new values from the GUI.
