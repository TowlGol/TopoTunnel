# TopoTunnel

TopoTunnel is a topology-guided processing and visual-analytics framework for mapping and comparing protein tunnels across molecular-dynamics conformations and datasets. It represents tunnel points through their local protein-atom context, projects paths into a shared reference conformation, and supports path-level, residue-level, and residue-network analysis across simulation conditions.

This GitHub repository contains the source code and documentation only. Experimental datasets are distributed separately through Zenodo:

**Data record:** https://zenodo.org/records/21287445

## Main Capabilities

- Convert aligned MD structures and CAVER tunnel profiles into a topology-mapped SQLite dataset.
- Select or reuse a shared reference conformation.
- Assign four surface-gap-ranked protein atoms to each tunnel point.
- Compare tunnel ensembles across conformations and independent datasets.
- Retrieve corresponding paths using residue composition, exit position, and path length constraints.
- Inspect synchronized tunnel profiles, bottleneck residues, residue pairs, and higher-order residue motifs.
- Export selected paths for downstream structural analysis and visualization.

## Repository Structure

```text
TopoTunnel/
|-- README.md
|-- .gitignore
|-- .gitattributes
|-- TopoTunnel_Pipeline/
|   |-- README.md
|   |-- pipeline_gui.py
|   |-- pipeline_config.py
|   |-- stage_01_run_caver.py
|   |-- stage_02_select_reference.py
|   |-- stage_03_map_topology.py
|   |-- stage_04_build_path_points.py
|   |-- stage_05_preprocess_dataset.py
|   `-- supporting Python modules
`-- TopoTunnel_UI/
    |-- README.md
    |-- __main__.py
    |-- app/
    |-- core/
    |-- db/
    |-- vis/
    `-- resources/
```

| Component | Purpose |
|---|---|
| `TopoTunnel_Pipeline` | Converts aligned PDB frames and CAVER tunnel profiles into a UI-loadable topology-mapped dataset. It can optionally run segmented CAVER calculations. |
| `TopoTunnel_UI` | Desktop application for tunnel visualization, path selection, constraint-based matching, synchronized profiles, and residue-network comparison. |

See the component READMEs for complete parameter and interface documentation:

- [Processing pipeline](TopoTunnel_Pipeline/README.md)
- [Desktop UI](TopoTunnel_UI/README.md)

## Tutorial Video

<p align="center">
  <a href="https://youtu.be/kNIT94s-Q9E">
    <img src="https://img.youtube.com/vi/kNIT94s-Q9E/maxresdefault.jpg" alt="Tutorial Video for TopoTunnel" width="800">
  </a>
</p>

**[Watch Tutorial Video for TopoTunnel on YouTube](https://youtu.be/kNIT94s-Q9E)**

This video provides a complete walkthrough of TopoTunnel, including the source code, data organization, and use of both the processing pipeline interface and the desktop visualization interface.

## Requirements

- Windows 10 or 11
- Python 3.10 or 3.11
- Java and CAVER 3.x only when tunnel profiles must be computed from PDB frames
- An OpenGL-capable graphics driver for the VTK/PyVista interface
- At least 8 GB RAM; 16 GB or more is recommended for large datasets

## Installation

Clone the code repository and create an isolated Python environment:

```powershell
git clone https://github.com/TowlGol/TopoTunnel.git
cd TopoTunnel

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r TopoTunnel_Pipeline\requirements.txt
python -m pip install -r TopoTunnel_UI\requirements.txt
```

The pipeline and desktop UI can also be installed in separate environments when Qt/VTK dependencies need to be isolated from data-processing dependencies.

## Processing Your Own Dataset

The minimum inputs are:

1. A directory containing aligned frame-level PDB structures.
2. A CAVER `tunnel_profiles.csv` file generated from those structures.
3. Optionally, an explicit reference-conformation PDB.

Start the processing interface:

```powershell
cd TopoTunnel_Pipeline
python pipeline_gui.py
```

Choose **Reference-Conformation Mapping and Dataset Preparation**, provide the aligned-frame directory and tunnel-profile CSV, select an output directory, and run the analysis. If CAVER output is not available, also select **Segmented CAVER Tunnel Computation and Tunnel-Profile Aggregation** and provide the CAVER installation and configuration paths.

The pipeline README explains every input field, parameter, processing stage, and troubleshooting step.

## Output Dataset

A complete generated dataset contains:

```text
my_dataset/
|-- preprocessed_paths.db
|-- representative_frame.pdb
|-- residue_statistics.csv
|-- residue_combination_statistics.csv
|-- residue_pair_distances.npy
`-- residue_pair_path_presence_stats.csv
```

| File | Purpose |
|---|---|
| `preprocessed_paths.db` | Main SQLite database containing path geometry, frame and cluster provenance, lining residues, and mapped topology. |
| `representative_frame.pdb` | Shared reference conformation used for mapping and protein display. |
| `residue_statistics.csv` | Per-residue path, point, radius, and cooperative-residue summaries. |
| `residue_combination_statistics.csv` | Residue-pair and higher-order bottleneck-motif summaries. |
| `residue_pair_distances.npy` | Precomputed residue-pair and residue-combination distance data. |
| `residue_pair_path_presence_stats.csv` | Pair-distance statistics conditioned on path presence. |

Only the database is required for basic tunnel display. Keeping all six files together enables the complete structural and residue-network workflow.

## Starting the Desktop UI

Run the application from the repository root:

```powershell
python -m TopoTunnel_UI
```

Then:

1. Open the `Datasets` tab.
2. Click `Add Dataset`.
3. Select a generated dataset directory containing `preprocessed_paths.db`.
4. Select the dataset and click `Show Tunnels`.
5. Use the `Paths`, `Residue Compare`, and `Residue Combination Compare` tabs for interactive analysis.

The [UI documentation](TopoTunnel_UI/README.md) describes every button, mouse interaction, matching control, chart, and export operation.

## Data Availability

The Zenodo record contains the processed experimental data used in the TopoTunnel study, including:

- LinB WT, Open, and Closed cMD/GaMD analysis outputs;
- selected LinB structural inputs;
- CAVER configuration files;
- residue-exclusion controls; and
- energy-based matching-validation data.

Experimental data should be cited using the metadata and references provided on the Zenodo record. The original simulation studies remain the provenance sources for their respective molecular-dynamics datasets.

## License

Copyright (c) 2026 Yapeng Liu. All rights reserved. This project is provided for local, non-commercial research evaluation only. Republishing, reproducing, mirroring, redistributing, sublicensing, selling, or incorporating any part of the code or documentation into another distributed work is prohibited without prior written permission. See [LICENSE](LICENSE) for the complete terms.

## Important Notes

- PDB names recorded in `tunnel_profiles.csv` must match files in the aligned-frame directory.
- Atom names, atom identities, residue numbering, and chain topology must remain compatible across conformations.
- Generated datasets should remain outside the application source directories.
- Existing datasets using the former intermediate names `1.atom_network.csv` and `4.all_points.csv` remain readable; new runs use `atom_network.csv` and `mapped_path_points.csv`.
