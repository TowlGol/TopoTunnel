# TopoTunnel Desktop UI

This directory contains the TopoTunnel desktop frontend, including the application entry point, database access layer, three-dimensional visualization, charts, data-import helpers, and theme resources. Experimental databases, PDB trajectories, and analysis CSV files are not included in the source package. Generate them with `TopoTunnel_Pipeline` or download them separately from the project Zenodo record.

This guide covers installation, startup, dataset preparation, loading, interactive analysis, path matching, chart operations, and export behavior.

## 1. Package Contents

```text
TopoTunnel_UI/
|-- README.md
|-- requirements.txt
|-- __main__.py
|-- app/                 # Main window, widgets, and table models
|-- core/                # State, comparison, filtering, and export logic
|-- db/                  # SQLite and PKL data-access layers
|-- vis/                 # VTK/PyVista views and pyqtgraph charts
|-- scripts/import_data.py
|-- resources/theme.qss
`-- database/            # Empty runtime directory for optional imports
```

Experimental datasets and the CYP3A4 reproduction case are not included in the GitHub source repository. They can be downloaded from `https://zenodo.org/records/21287445` and loaded from any writable directory.

### 1.1 Source File Index

| File | Purpose |
|---|---|
| `__main__.py` | Command-line entry point; configures Qt, loads the theme, resolves an optional database path, and creates the main window. |
| `app/main_window.py` | Main-window coordinator connecting datasets, path tables, the 3D viewer, charts, matching, and residue analysis. |
| `app/models/path_model.py` | Qt model for path-table values, sorting, and display. |
| `app/widgets/dataset_panel.py` | Dataset tree, dataset management, prefixes, colors, cluster visibility, and reference-structure controls. |
| `app/widgets/path_panel.py` | Path table, path categories, selection, visibility, exclusion, and export controls. |
| `app/widgets/protein_viewer_panel.py` | Dual-structure Residue Observer, frame navigation, and residue highlighting. |
| `app/widgets/residue_panel.py` | Single-residue filters and residue-property controls. |
| `app/widgets/residue_compare_panel.py` | Cross-dataset residue-statistics table and heatmap entry point. |
| `app/widgets/residue_combination_panel.py` | Residue-pair and motif comparison, filtering, constraint-based matching, and result linking. |
| `core/state.py` | Shared model for datasets, path selections, filters, and visibility state. |
| `core/exporter.py` | PDB export for selected paths and path categories. |
| `core/residue_compare.py` | Loading, alignment, and difference calculation for single-residue statistics. |
| `core/residue_combination.py` | Calculation and filtering of residue-pair and higher-order combination statistics. |
| `db/tunnel_db.py` | Query interface for one SQLite tunnel dataset. |
| `db/pkl_tunnel_db.py` | Compatibility reader for legacy PKL datasets. |
| `db/multi_dataset_db.py` | Unified indexing and query layer for multiple datasets. |
| `scripts/import_data.py` | Converts a preprocessed PKL file to a UI-readable SQLite database. |
| `vis/tunnel_viewer.py` | Main PyVista/VTK tunnel view, path picking, and layer rendering. |
| `vis/lasso.py` | Screen-space lasso drawing and path/point selection. |
| `vis/charts.py` | Synchronized path profiles and statistics charts. |
| `vis/protein_model.py` | PDB parsing, alignment, and protein-model geometry. |
| `vis/cartoon_ribbon.py` | Cartoon and ribbon protein geometry. |
| `vis/residue_display.py` | Residue positions, labels, highlights, and combination markers. |
| `vis/binding_surface.py` | Construction and display of binding-site surfaces. |
| `vis/pse_parser.py` | Helper for reading protein and non-protein objects from PyMOL PSE sessions. |
| `resources/theme.qss` | Qt theme, widget dimensions, and color styles. |
| `requirements.txt` | Python dependencies required by the UI. |
| `__init__.py` files | Python package markers and package-level exports. |

## 2. System Requirements

- Windows 10 or 11 is the primary tested platform, 64-bit recommended. The GUI has also been launched on macOS; complete cross-platform workflow and Linux support have not yet been systematically validated.
- Python 3.10 or 3.11. Standard CPython in an isolated virtual environment is recommended.
- A graphics driver supporting OpenGL 3.2 or newer for the VTK 3D view.
- At least 8 GB RAM; 16 GB or more is recommended for large multi-dataset sessions.
- A display resolution of at least 1920 x 1080 is recommended. The minimum practical window size is 1200 x 800.

Avoid installing the application into an Anaconda base environment that already contains Qt packages. Conflicting Qt DLLs can cause `DLL load failed while importing QtCore` or `QtWidgets` errors.

## 3. Installation

### 3.1 Open a terminal

Open PowerShell on Windows or a shell on macOS/Linux, then change to the repository root containing `TopoTunnel_UI`:

```text
cd <path-to-repository>/TopoTunnel
```

If the repository is located elsewhere, replace the path with the actual location.

### 3.2 Install uv and create a virtual environment

Install Astral's `uv` using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/). For example:

```powershell
# Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment:

```text
uv python install 3.11
uv venv --python 3.11 .venv
```

Activate it with the command for your shell:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

If PowerShell blocks activation, leave the environment inactive and use the environment's Python executable explicitly.

### 3.3 Install Dependencies

```bash
uv pip install -r TopoTunnel_UI/requirements.txt
```

| Dependency | Purpose |
|---|---|
| `PySide6` | Desktop interface and Qt event system. |
| `numpy`, `scipy` | Numerical calculations, interpolation, and geometry processing. |
| `pyvista`, `pyvistaqt`, `vtk` | 3D tunnel, protein, and residue visualization. |
| `pyqtgraph` | Path profiles and statistics charts. |
| `tqdm` | Progress reporting during first-time PKL-to-SQLite conversion. |

### 3.4 Verify the Installation

```bash
python -c "from PySide6 import QtCore; import numpy, scipy, pyvista, pyvistaqt, vtk, pyqtgraph; print('Environment OK:', QtCore.__version__)"
```

## 4. Start the Application

Run from the repository root:

```powershell
python -m TopoTunnel_UI
```

An empty 3D view is expected when no dataset is loaded. Use `Datasets -> Add Dataset` to load an external dataset.

## 5. Dataset Requirements

### 5.1 Minimum Requirement

Each dataset must be in its own directory and contain either:

- a TopoTunnel SQLite database such as `preprocessed_paths.db` or `tunnel_data.db`; or
- a preprocessed PKL file such as `preprocessed_paths.pkl`.

Keep one primary database in each dataset directory so recursive discovery does not select an unintended database.

When only a PKL file is present, the application creates `tunnel_data.db` in the dataset directory. The directory must therefore be writable and have enough free space for the converted database.

### 5.2 Recommended Dataset Layout

```text
my_dataset/
|-- preprocessed_paths.db                 # Required for direct loading
|-- representative_frame.pdb              # Optional reference protein
|-- residue_statistics.csv                # Optional residue comparison data
|-- residue_combination_statistics.csv    # Optional motif comparison data
|-- residue_pair_distances.npy            # Optional precomputed distances
|-- residue_pair_path_presence_stats.csv  # Optional path-presence statistics
`-- MD_path.txt                            # Optional trajectory location
```

The first non-empty line of `MD_path.txt` should contain an absolute path or a path relative to `MD_path.txt` pointing to the frame-level PDB directory.

### 5.3 Optional Files and Enabled Features

| File | Enables |
|---|---|
| `.db` or `.pkl` | Path table, 3D tunnels, profiles, frame filtering, path selection, and matching. |
| `residue_statistics.csv` | Cross-dataset residue-property heatmaps and cooperative-residue changes. |
| `residue_combination_statistics.csv` | Two- to four-residue combinations, residue-pair/motif comparison, and 3D combination markers. |
| `representative_frame.pdb` | Reference-protein display and static Residue Observer structure. |
| `MD_path.txt` and frame PDBs | Frame-by-frame Residue Observer browsing and local conformational comparison. |
| `residue_pair_distances.npy` or `.csv` | Fast access to precomputed residue-pair distances. |
| `residue_pair_path_presence_stats.csv` | Residue-pair statistics conditioned on path presence or absence. |

Missing optional files do not prevent basic database loading, but the corresponding comparison controls may remain disabled.

## 6. First Use

1. Start the application and open the `Datasets` tab.
2. Click `Add Dataset` and select one or more dataset directories.
3. Wait for loading to finish. The status area reports path count, residue count, and loaded dataset count.
4. Select a dataset and click `Show Tunnels`.
5. Open the `Paths` tab and use Ctrl/Shift to select multiple rows.
6. Click `Sync Charts` to load profiles for the current valid paths.
7. Add a second dataset before using residue comparison or cross-dataset matching.

## 7. Main Window Layout

- **Top control area:** path color, difference overlays, entry/exit markers, protein display, frame range, and selection mode.
- **Left Control Panel:** `Datasets`, `Paths`, `Residue Compare`, and `Residue Combination Compare` tabs.
- **Central 3D view:** tunnels, selected paths, residues, entry/exit points, protein, and combination markers.
- **Right Residue Observer:** paired local structures and frame-by-frame comparison.
- **Bottom chart area:** path profiles, statistics, synchronization, and detached chart workspace.
- **Status bar:** path count, residue count, selected-path count, loading state, and operation messages.

The right edge of the Control Panel and the separators between the 3D view, charts, and Residue Observer can be dragged to resize the layout.

## 8. 3D View Interaction

### 8.1 Camera and Object Interaction

| Action | Effect |
|---|---|
| Left-drag | Rotate the 3D view; becomes lasso selection when lasso mode or Shift is active. |
| Right-drag | Pan the view. |
| Mouse wheel | Zoom in or out. |
| Left-click on a path | Select the path and synchronize the path table and charts. |
| Left-click on a residue marker | Select the residue. |
| Left-click on a combination marker | Select the combination and synchronize its details and Observer. |
| Hover over a residue or combination marker | Show marker information. |

### 8.2 Lasso Selection

| Action | Selection behavior |
|---|---|
| `Shift + left-drag` | Add paths inside the lasso to the current selection. |
| `Alt + Shift + left-drag` | Remove paths inside the lasso from the current selection. |
| `Shift + right-drag` | Remove paths inside the lasso from the current selection. |
| Left-drag with `Lasso` enabled | Replace the current selection with the lasso result. |
| `Esc` | Clear the selection and lasso outline. |

When `Point Select` is disabled, lasso selection uses the path centerline. When it is enabled, selection targets entry/exit points for coarser selection.

### 8.3 View Button

| Button | Purpose |
|---|---|
| `Current Paths` / `Original Paths` | Switch between processed/mapped coordinates and original exported coordinates. The button shows `Loading...` during loading. Full comparison requires original coordinates in the database. |

## 9. Top Control Area

### 9.1 Path Display and Difference Overlays

| Control | Purpose |
|---|---|
| `Color` slider | Adjust background path color from light gray toward black. |
| `BG` slider | Adjust the opacity of unselected background paths. |
| `Path Delta` | Color paths by pointwise displacement between original and current coordinates; red indicates larger change. |
| `Exit Delta` | Color paths by the change in exit-to-cluster-center distance between original and current coordinates. |
| Delta threshold slider | Hide paths whose enabled Delta value is at or below the threshold. The adjacent label shows the current value. |

`Path Delta` describes displacement along the complete path, whereas `Exit Delta` focuses on exit-region rearrangement. Both require current and original coordinates in the database.

### 9.2 Layers and Selection

| Button | Purpose |
|---|---|
| `Residues` | Show or hide residue positions in the 3D view. |
| `Res Labels` | Show or hide labels for selected residues. |
| `Entry/Exit` | Show or hide entry and exit points; entry is green and exit is red. |
| `EE: All` / `EE: Current` | Show entry/exit points for all paths or only current valid paths. |
| `Exit Cluster` | After activation, click an exit sphere to select visually connected exits in the current camera view. |
| `Focus` | Hide background paths and show only selected/valid paths. Shortcut: `F`. |
| `Point Select` | Switch lasso targets between paths and entry/exit points. |
| `Lasso` | Force ordinary left-drag to perform lasso selection. Shortcut: `L`. |
| `Clear Sel` | Clear the current path selection. Shortcut: `Esc`. |

### 9.3 Reference Protein and Binding Sites

| Control | Purpose |
|---|---|
| `Protein...` | Load a reference `.pdb` or PyMOL `.pse` session. Non-protein PSE objects can be displayed as binding-site surfaces. |
| `Residue Observer` | Show or hide the right-side paired structure observer. |
| `Reference PDB` | Show or hide the loaded reference protein; unavailable until a structure is loaded. |
| Style menu | Select `cartoon`, `backbone`, `tube`, or `ca_spheres`. |
| Alignment menu | `kabsch` performs rigid alignment using available anchors; `translate` applies translation only; `none` preserves input coordinates. |
| `Opacity` slider | Control reference-protein opacity. The current main toolbar keeps this value at 100%. |
| `Clear` | Remove the reference protein, PSE binding sites, and related caches. |
| `Sites` | Show or hide binding-site surfaces loaded from a PSE session. |

### 9.4 Frame Range

Click `Frame` to open the range menu:

| Button | Purpose |
|---|---|
| `Apply` | Use the entered minimum and maximum frame values to filter paths, the 3D view, and subsequent analyses. |
| `Clear` | Remove the frame range and restore all frames. |

## 10. Datasets Tab

The dataset tree supports Ctrl/Shift multi-selection. Expanding a dataset shows CAVER Cluster entries. Selecting a cluster adds its paths to the current selection. CAVER clusters are input spatial clusters; manual path categories in the `Paths` tab are user-defined groups and should not be confused with them.

| Button | Purpose |
|---|---|
| `Add Dataset` | Select one or more dataset directories. The application recursively searches for the main `.db`/`.pkl` and optional statistics files. |
| `Edit Prefix` | Change the short dataset prefix used in tables, charts, and comparisons. Prefixes remain unique within the session. |
| `Pick Color` | Set the dataset base color in the 3D view and charts. |
| `Reference PDB` | Open a PDB/PSE chooser and load a structure into the central 3D view. |
| `Show Tunnels` / `Hide Tunnels` | Show or hide tunnels from the selected datasets. |
| `Cluster Colors` / `Uniform Color` | Switch between per-cluster colors and one color per dataset. |
| `Cluster Labels` / `Hide Labels` | Show or hide 3D cluster labels. The behavior is clearest when one dataset is selected. |
| `Remove` | Remove selected datasets from the current session without deleting files. |

Cluster checkboxes select paths; they do not delete or recompute clusters.

## 11. Paths Tab

The path table includes `Dataset`, `Path ID`, `Frame`, `Length`, `Min R`, `Avg R`, `Avg Hydro`, `# Residues`, and `Cluster`. Click a column heading to sort. Use Ctrl/Shift for multi-row selection.

### 11.1 Path Category Controls

| Button | Purpose |
|---|---|
| `Create` | Create a category from table-selected paths. If no rows are selected, use current valid paths or the most recent matching paths. An empty name receives a timestamp-based name. |
| `Create From Current` | Create a category from the current selection context. |
| `Select` | Set paths in the current category as the current selection. |
| `Show` / `Hide` | Show or hide the category highlight in 3D. |
| `Select & Show` | Select and show the category together. |
| `Set Color` | Set the category display color. |
| `Show All` / `Hide All` | Show or hide all manual categories. |
| `Excl All` / `Incl All` | Mark all categories as excluded candidates or clear their exclusion state. |
| `Export` | Export category paths as PDB files, either merged by frame or split into separate files. |
| `Delete` | Delete the manual category without deleting original paths. |
| `Add Selected to Category` | Append selected table rows to the current category and remove duplicates. |

Category list actions:

- Single-click a category to make it current.
- Double-click a category to toggle its 3D visibility.
- Right-click a category to show/hide, include/exclude, export, or delete it.
- `[V]` means visible, `[X]` means excluded, and `[V|X]` means both states apply.

### 11.2 Path Table Views

| Control | Purpose |
|---|---|
| `All Paths` | Show all paths satisfying the current frame and residue filters. |
| `Selected (N)` | Show the current valid paths or visible category paths. |
| `Load Selected` | Load the current selection in the `Selected` view and send it to the profile chart. |

## 12. Residue Compare Tab

This tab requires both datasets to provide `residue_statistics.csv`. Positive File B minus File A differences are shown in red; negative differences are blue; values near zero are white.

| Control | Purpose |
|---|---|
| `File A`, `File B` | Choose the two loaded datasets to compare. |
| `Reload` | Clear linked residue selection and refresh views driven by the current selection. Changing either file reloads its CSV automatically. |
| `Open Heatmap` | Open a larger residue-difference heatmap in a separate window. |
| Heatmap `Close` | Close the separate heatmap window. |

The table compares affected points, affected paths, mean/min/max radius, and cooperative-residue values. Clicking a row shows details below. Clicking a File A or File B cell sends the residue to the 3D selection. Cooperative-residue columns show increased and decreased partners.

## 13. Residue Combination Compare Tab

This tab analyzes combinations of two to four residues, residue pairs, and bottleneck motifs. At least one dataset must provide `residue_combination_statistics.csv`; cross-dataset comparison requires both File A and File B.

### 13.1 Data, Display, and Filter Controls

| Control | Purpose |
|---|---|
| `File A`, `File B` | Select one dataset for single-dataset analysis or two datasets for a difference comparison. |
| `Residue 1-4` | Specify two to four residues of interest; unused slots remain `-`. |
| `Size` | Show only two-, three-, or four-residue combinations, or show all sizes. |
| `Color` | Color markers by affected paths, affected points, mean radius, path-weighted radius, or radius range. Comparison mode uses File B minus File A. |
| `+ Group` | Add a filter group with an attribute, Increase/Decrease/Neutral trend, and absolute threshold. |
| Filter-group `Remove` | Remove a filter group. |
| `Show` | Show or hide centroids of visible combinations in the 3D view. |
| `Sel Only` | Show only combinations containing the current residue selection. |
| `Bottleneck` | Show only combinations occurring at bottleneck points of the selected paths. |
| `Show AB Res` | Show all residues involved in visible File A/File B combinations. |
| `Clear` | Clear the four residue slots. |
| `Load DS` | Load combinations over the full selected-dataset range. |
| `Load Path` | Load or recalculate combinations in the context of current path selections. |
| `Refresh` | Refresh the table, filters, and 3D display. |
| `Save Table` | Save the currently visible combination table as CSV. |

Clicking a combination row updates its details. Clicking presence or File A/File B cells links the selection to the 3D view and Residue Observer. When `Show` is enabled, clicking a 3D combination marker selects the corresponding table record.

### 13.2 Matched Paths Controls

| Control | Purpose |
|---|---|
| `Source` | Select the source dataset from File A or File B; the other dataset becomes the target. |
| `Keep` | Set the retention ratio `K`, with values between 0 and 1. |
| `Query` | Retrieve target paths using the source residue set, exit region, and path-length constraints. |
| `Compare` | Compare residue-pair and combination statistics for the current matched paths. Without a locked match, compare the current File A/File B selections directly. |
| `Reset` | Clear the current match and comparison result and restore ordinary selection context. |

Before pressing `Query`:

1. Select two datasets in File A and File B.
2. Choose the source dataset.
3. Select source paths in the path table, CAVER Cluster checkboxes, 3D lasso, or a manual category.
4. Set `Keep` and click `Query`.

At `Keep=0.8`, the exit constraint retains the 80% of source paths closest to the initial exit centroid and recomputes the exit region. The length constraint uses the central 80% of the source length distribution. The residue constraint still uses the union of residues from all source paths. A target path is retained only if its exit lies in the reference exit region, its length lies in the retained interval, and its non-empty residue set is a subset of the source residue union. The source paths remain selected and matched target paths are highlighted in the path table.

## 14. Residue Observer

Click `Residue Observer` in the top controls to show the right-side panel. It contains two protein views and a shared timeline.

| Control | Purpose |
|---|---|
| Residue input | Enter comma-separated residue numbers, for example `72,149`. |
| `Apply` | Highlight the entered residues in both Observer views. |
| Two `MD Dataset` menus | Select frame-level PDB sources for the upper and lower views. |
| `Model` | Select Backbone, Cartoon, Tube, or CA Spheres. |
| `Opacity` | Adjust protein opacity. |
| `Prev` | Move to the previous frame. |
| `Frame` spin box | Jump directly to a frame. |
| `Next` | Move to the next frame. |
| `Highlight` | Select Ball-and-Stick, Sticks, or Spheres residue highlighting. |

Click the timeline to jump to a frame. If `MD_path.txt` is present, the application uses the specified frame directory first. Without frame-level PDBs, the Observer can still show the static `representative_frame.pdb`, but `Prev` and `Next` remain disabled.

## 15. Bottom Path Charts

Charts do not recompute immediately after every path-selection change. Use `Sync Charts` after changing the selection.

| Button | Purpose |
|---|---|
| `Sync Charts` | Read profiles for current valid paths and update the Profile and Statistics charts. |
| `Chart Window` | Move charts to a separate synchronized workspace. |

### 15.1 Profile Chart Controls

| Control | Purpose |
|---|---|
| `Controls` | Expand or collapse chart controls. |
| `Color` | Color curves by Radius, Hydrophobicity, Frame, or Dataset. |
| `Advanced` | Show custom X/Y-axis controls. Available variables include Radius, Hydrophobicity, Frame, and Path Length. |
| `Path Selection` | Enable rectangular selection of paths crossing the selected region. Click again to disable. |
| `Residue Stats` | Enable point-level rectangular selection and summarize associated residues. |
| `Opacity` | Adjust profile-line opacity. |
| `Depth` | Adjust color depth and visual hierarchy. |
| Dataset checkboxes | Show or hide curves by dataset in the Chart Workspace and synchronize the Statistics chart. |
| `Keep Only` | Keep only paths selected in the chart and replace the main-window selection. |
| `Exclude` | Remove chart-selected paths from the chart and main-window selection. |
| `Undo` | Undo the latest `Keep Only` or `Exclude`; up to five chart operations are retained. |
| `Clear` | Clear the rectangular chart selection without deleting data. |

The chart view disables wheel zoom. Rectangular dragging is active only in `Path Selection` or `Residue Stats` mode.

### 15.2 Statistics Chart

The Statistics chart displays the mean and quantile range of paths currently shown in the Profile chart. It inherits color, axis, and dataset filters and does not provide independent path-selection controls.

## 16. Chart Workspace

| Control | Purpose |
|---|---|
| `Close` | Close the detached chart window and restore charts to the bottom of the main window. |
| Residue Summary `Path Residues` | Summarize all residues in selected chart paths. |
| Residue Summary `Bottleneck` | Summarize residues at the minimum-radius point of each path. |
| Residue Summary `Compare` | Open a residue-summary comparison heatmap organized by dataset. |
| Path Filter `Select Filtered` | Filter chart paths by Hydrophobicity and Frame ranges and synchronize the main selection. |
| `Revert` | Restore the path selection from before the filter was applied. |
| `Clear` | Clear Hydrophobicity and Frame filter values. |

## 17. Keyboard Shortcuts

| Shortcut | Purpose |
|---|---|
| `L` | Toggle forced lasso mode. |
| `F` | Toggle Focus mode. |
| `Esc` | Clear the current selection and unfinished lasso outline. |
| `Shift + left-drag` | Add lasso selection. |
| `Alt + Shift + left-drag` | Remove lasso selection. |
| `Shift + right-drag` | Remove lasso selection. |

## 18. Recommended Workflows

### 18.1 Single-Dataset Path Inspection

1. Load a dataset with `Add Dataset`.
2. Click `Show Tunnels`.
3. Narrow the scope with the frame range or CAVER Cluster checkboxes.
4. Select paths with the 3D lasso.
5. Create a manual category in `Paths` and set its color.
6. Click `Sync Charts` to inspect radius and hydrophobicity profiles.
7. Export selected paths as PDB files.

### 18.2 Cross-Dataset Constraint Matching

1. Load two datasets and assign distinct prefixes and colors.
2. Select source paths in the source dataset.
3. Open `Residue Combination Compare` and select File A and File B.
4. Set `Source` and `Keep`, then click `Query`.
5. Inspect source count, exit region, length interval, and target-hit count in the status area.
6. Click `Compare` to calculate residue-pair and combination differences.
7. Use `Show`, `Show AB Res`, and `Residue Observer` to inspect local structures.

### 18.3 Residue-Network Analysis

1. Ensure both datasets provide residue and combination statistics CSV files.
2. Use `Residue Compare` to identify strongly changing residues or partner relationships.
3. Click File A or File B cells to link residues to the 3D view.
4. Select two to four candidate residues in `Residue Combination Compare`.
5. Use `Bottleneck` and Increase/Decrease filter groups to focus on relevant motifs.
6. Inspect residue distances and side-chain conformations frame by frame in the Observer.
7. Save the visible combination table with `Save Table`.

## 19. Exported Files

- Path-category export produces PDB files; split-by-frame mode also produces text metadata summaries.
- `Save Table` exports the currently visible residue-combination table as CSV.
- First-time PKL loading creates `tunnel_data.db` in the dataset directory.

`Remove`, `Delete`, and filtering operations modify only the current UI session. They do not delete original dataset files.

## 20. Troubleshooting

### 20.1 `DLL load failed while importing QtCore/QtWidgets`

This usually indicates conflicting Qt DLLs. Exit the Anaconda base environment, create a clean Python 3.10/3.11 virtual environment, and install `requirements.txt` there. Do not mix conda Qt packages with pip-installed PySide6.

### 20.2 Black 3D View or OpenGL Errors

- Update the graphics driver.
- Confirm OpenGL 3.2 or newer.
- Run in a local desktop session when possible; Windows Remote Desktop may not provide a suitable OpenGL context.
- Confirm that `pyvista`, `pyvistaqt`, and `vtk` come from the same virtual environment.

### 20.3 `pyqtgraph not installed`

```powershell
python -m pip install pyqtgraph
```

Restart the application afterward.

### 20.4 `Add Dataset` Cannot Find a Database

Select a dataset directory rather than an individual file. Confirm that it contains a `.db`, `preprocessed_paths.pkl`, or another `.pkl` file. Check that file extensions are not hidden or accidentally changed.

### 20.5 PKL Loading Is Slow

The application converts PKL to SQLite and creates indexes on the first load. Wait for the operation to finish and do not add the same directory repeatedly. Later loads use the generated `tunnel_data.db`.

### 20.6 Residue Comparison Has No Data

Confirm that the dataset directory contains correctly named `residue_statistics.csv` and `residue_combination_statistics.csv` files, then remove and re-add the dataset.

### 20.7 Residue Observer Has No Frame Data

Check that `MD_path.txt` points to a valid directory containing PDB files. With only `representative_frame.pdb`, the Observer can display a static structure but cannot provide frame navigation.

### 20.8 Controls Are Compressed or Truncated

- Maximize the window.
- Drag the right edge of the Control Panel to adjust its width.
- Drag the separators between the 3D view, charts, and Observer.
- Use the tab-bar scroll buttons when tab labels do not fit.
