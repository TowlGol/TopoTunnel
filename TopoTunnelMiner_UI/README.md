# TopoTunnelMiner Desktop UI

本目录包含 TopoTunnelMiner 桌面前端，包括启动入口、数据库访问层、三维可视化、图表、数据导入辅助代码和主题文件。应用源码不内置实验数据库、PDB轨迹或统计结果，数据应由处理流水线生成或单独下载。

本文档按照“安装 -> 启动 -> 准备数据 -> 加载数据 -> 交互分析 -> 导出结果”的顺序说明当前版本的使用方法，并逐一解释主界面中可见的按钮和交互方式。

## 1. 目录内容

```text
Github_Version/
|-- README.md
|-- TopoTunnelMiner_Pipeline/
`-- TopoTunnelMiner_UI/
    |-- README.md
    |-- requirements.txt
    |-- __main__.py
    |-- app/                 # 主窗口、控件和表格模型
    |-- core/                # 筛选、比较、状态和导出逻辑
    |-- db/                  # SQLite/PKL数据访问层
    |-- vis/                 # VTK/PyVista三维视图和pyqtgraph图表
    |-- scripts/import_data.py
    |-- resources/theme.qss
    `-- database/            # 空目录，可放用户自己的数据
```

实验数据和CYP3A4复现案例不包含在GitHub代码仓库中，可从项目Zenodo记录（`https://zenodo.org/records/21287445`）单独下载。下载后的数据集可以放在任意可写目录，并通过`Add Dataset`加载。

### 1.1 源码文件索引

| 文件 | 作用 |
|---|---|
| `__main__.py` | 命令行启动入口，配置Qt运行环境、加载主题、解析可选数据库参数并创建主窗口。 |
| `app/main_window.py` | 主窗口和应用级交互协调器，连接数据集、路径表、三维视图、图表、匹配和残基分析。 |
| `app/models/path_model.py` | Qt路径表数据模型，负责列值、排序和表格显示。 |
| `app/widgets/dataset_panel.py` | 数据集树、添加/移除数据集、前缀、颜色、簇显示和参考结构控件。 |
| `app/widgets/path_panel.py` | 路径表、路径类别、选择、显示、排除和导出控件。 |
| `app/widgets/protein_viewer_panel.py` | Residue Observer的双结构视图、帧导航和残基高亮。 |
| `app/widgets/residue_panel.py` | 单残基筛选与残基属性控件。 |
| `app/widgets/residue_compare_panel.py` | 两数据集残基统计比较表和热图入口。 |
| `app/widgets/residue_combination_panel.py` | 残基对/组合比较、筛选、constraint-based matching及结果联动。 |
| `core/state.py` | 当前数据集、路径选择、过滤和可见状态的共享状态模型。 |
| `core/exporter.py` | 选中路径和路径类别的PDB导出逻辑。 |
| `core/residue_compare.py` | 单残基统计读取、对齐和跨数据集差值计算。 |
| `core/residue_combination.py` | 残基对与高阶组合统计、筛选和比较计算。 |
| `db/tunnel_db.py` | 单个SQLite数据集的查询接口。 |
| `db/pkl_tunnel_db.py` | 旧版PKL数据集的兼容读取接口。 |
| `db/multi_dataset_db.py` | 多数据集统一编号、查询和合并访问层。 |
| `scripts/import_data.py` | 将预处理PKL转换为UI可读取的SQLite数据库。 |
| `vis/tunnel_viewer.py` | PyVista/VTK主三维隧道视图、路径拾取和图层渲染。 |
| `vis/lasso.py` | 屏幕空间套索绘制和路径/点选择。 |
| `vis/charts.py` | 同步路径剖面图和统计图表。 |
| `vis/protein_model.py` | PDB蛋白结构解析、对齐和模型几何构建。 |
| `vis/cartoon_ribbon.py` | Cartoon/ribbon蛋白显示几何。 |
| `vis/residue_display.py` | 残基位置、标签、高亮和组合标记渲染。 |
| `vis/binding_surface.py` | 结合位点表面的构建与显示。 |
| `vis/pse_parser.py` | PyMOL PSE会话中蛋白和非蛋白对象的读取辅助。 |
| `resources/theme.qss` | Qt界面主题、控件尺寸和颜色样式。 |
| `requirements.txt` | UI运行所需Python依赖。 |
| `__init__.py`及子目录中的`__init__.py` | Python包标记和必要的包级导出。 |

## 2. 系统要求

- Windows 10/11，推荐64位系统。
- Python 3.10或3.11。建议使用标准CPython创建独立虚拟环境。
- 显卡和驱动应支持OpenGL 3.2或更高版本，以正常显示VTK三维视图。
- 推荐至少8 GB内存；加载大型多数据集时建议16 GB或更多。
- 建议显示分辨率不低于1920 x 1080。程序窗口最低尺寸为1200 x 800。

不要将程序安装到已有大量Qt依赖的Anaconda基础环境中。不同来源的Qt DLL可能发生冲突，表现为`DLL load failed while importing QtCore/QtWidgets`。独立虚拟环境更容易复现。

## 3. 安装步骤

### 3.1 打开PowerShell并进入项目目录

```powershell
cd <path-to-repository>\Github_Version
```

如果仓库被移动到了其他位置，将上面的路径替换为实际路径。

### 3.2 创建虚拟环境

```powershell
py -3.11 -m venv .venv
```

如果`py`命令不可用，可使用Python可执行文件的完整路径：

```powershell
C:\Path\To\Python311\python.exe -m venv .venv
```

### 3.3 激活虚拟环境

```powershell
.\.venv\Scripts\Activate.ps1
```

激活后，命令行前通常会出现`(.venv)`。如果PowerShell阻止执行激活脚本，可以不激活，后续命令改用`.\.venv\Scripts\python.exe`。

### 3.4 更新pip并安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r TopoTunnelMiner_UI\requirements.txt
```

主要依赖包括：

| 依赖 | 用途 |
|---|---|
| `PySide6` | 桌面界面和Qt事件系统 |
| `numpy`、`scipy` | 数值计算、插值和几何处理 |
| `pyvista`、`pyvistaqt`、`vtk` | 隧道、蛋白质和残基的三维显示 |
| `pyqtgraph` | 路径剖面和统计图表 |
| `tqdm` | PKL首次转换为SQLite时显示进度 |

### 3.5 验证安装

```powershell
python -c "from PySide6 import QtCore; import numpy, scipy, pyvista, pyvistaqt, vtk, pyqtgraph; print('Environment OK:', QtCore.__version__)"
```

如果命令输出`Environment OK`和Qt版本号，运行环境已准备好。

## 4. 启动程序

必须从包含`TopoTunnelMiner_UI`目录的仓库根目录运行：

```powershell
python -m TopoTunnelMiner_UI
```

启动后出现空白三维视图是正常的，因为应用源码不内置数据。请在左侧`Datasets`标签页使用`Add Dataset`加载外部数据。

## 5. 数据目录要求

### 5.1 最低要求

每个数据集应放在独立文件夹中，并至少包含以下一种文件：

- 一个TopoTunnelMiner SQLite数据库，例如`preprocessed_paths.db`、`tunnel_data.db`或其他`.db`文件；或
- `preprocessed_paths.pkl`，也可以是其他包含预处理路径数据的`.pkl`文件。

推荐每个文件夹只保留一个主数据库，避免程序递归扫描时选中非目标`.db`文件。

当文件夹中只有PKL而没有数据库时，程序会在该数据目录中生成`tunnel_data.db`。因此数据目录必须可写，并需要预留转换后的数据库空间。

### 5.2 推荐目录结构

```text
my_dataset/
|-- preprocessed_paths.db                 # 必需，或使用preprocessed_paths.pkl
|-- representative_frame.pdb              # 可选，参考蛋白和Residue Observer
|-- residue_statistics.csv                 # 可选，Residue Compare
|-- residue_combination_statistics.csv     # 可选，Residue Combination Compare
|-- residue_pair_distances.npy             # 可选，残基对距离统计
|-- residue_pair_path_presence_stats.csv   # 可选，路径出现/未出现条件统计
`-- MD_path.txt                            # 可选，指向逐帧PDB目录
```

`MD_path.txt`的第一条有效非空行应是逐帧PDB目录的绝对路径或相对于`MD_path.txt`的路径，例如：

```text
D:\Dataset\LinB\wt_cmd_run1_pdb_frames
```

### 5.3 可选文件与功能关系

| 文件 | 启用的功能 |
|---|---|
| `.db`或`.pkl` | 路径表、三维隧道、剖面图、帧筛选、路径选择和匹配 |
| `residue_statistics.csv` | 两数据集残基属性热图和协同残基变化 |
| `residue_combination_statistics.csv` | 2-4残基组合、残基对/基序比较和三维组合标记 |
| `representative_frame.pdb` | 参考蛋白显示以及Residue Observer静态结构 |
| `MD_path.txt`和逐帧PDB | Residue Observer逐帧浏览和局部构象对比 |
| `residue_pair_distances.npy/.csv` | 预计算残基对距离的快速读取 |
| `residue_pair_path_presence_stats.csv` | 路径出现与未出现条件下的残基对统计 |

缺少可选文件不会阻止程序加载主数据库，但对应比较按钮可能保持禁用，或界面显示找不到配套文件。

## 6. 首次使用流程

1. 启动程序并打开左侧`Datasets`标签页。
2. 点击`Add Dataset`，选择一个或多个数据集文件夹。
3. 等待状态栏中的加载进度结束。程序会显示路径总数、残基数和已加载数据集数量。
4. 在数据集树中选择一个数据集，使用`Show Tunnels`显示其路径。
5. 在中央三维视图中旋转、平移和缩放，或者点击路径进行选择。
6. 打开`Paths`标签页查看路径属性，使用Ctrl/Shift选择多个路径。
7. 点击中央图表上方的`Sync Charts`，加载当前有效路径的剖面与统计图。
8. 需要跨数据集比较时，再添加第二个数据集并进入`Residue Compare`或`Residue Combination Compare`。

## 7. 主界面布局

- **顶部控制区**：路径颜色、差异叠加、入口/出口、蛋白质、帧范围和选择模式。
- **左侧Control Panel**：`Datasets`、`Paths`、`Residue Compare`和`Residue Combination Compare`四个标签页。
- **中央三维视图**：隧道、选中路径、残基、入口/出口、蛋白质和组合标记。
- **右侧Residue Observer**：按需显示两个数据集的局部蛋白质结构和逐帧变化。
- **底部图表区**：路径剖面图、统计图、同步按钮和独立图表窗口。
- **状态栏**：显示路径数、残基数、当前选中路径数、加载状态和操作提示。

左侧Control Panel右边缘可以拖动以调整宽度。中央三维视图、底部图表和Residue Observer之间的分隔条也可以拖动。

## 8. 三维视图交互

### 8.1 摄像机操作

| 操作 | 效果 |
|---|---|
| 左键拖动 | 旋转三维视图；启用Lasso或按住Shift时改为套索 |
| 右键拖动 | 平移视图 |
| 鼠标滚轮 | 放大或缩小 |
| 左键单击路径 | 选中路径并联动路径表和图表 |
| 左键单击残基标记 | 选择对应残基 |
| 左键单击残基组合标记 | 选中组合并联动组合详情/Observer |
| 悬停残基或组合标记 | 显示标记信息提示 |

### 8.2 套索选择

| 操作 | 选择模式 |
|---|---|
| `Shift + 左键拖动` | 将套索范围内路径加入当前选择 |
| `Alt + Shift + 左键拖动` | 从当前选择中移除套索范围内路径 |
| `Shift + 右键拖动` | 从当前选择中移除套索范围内路径 |
| 开启`Lasso`后直接左键拖动 | 用套索结果替换当前选择 |
| `Esc` | 清除当前选择和套索轮廓 |

`Point Select`关闭时，套索按路径中心线进行较精细选择；开启时，套索针对入口/出口点进行较粗粒度选择。

### 8.3 视图内按钮

| 按钮 | 作用 |
|---|---|
| `Current Paths` / `Original Paths` | 在当前处理/映射坐标和原始导出坐标之间切换。加载期间显示`Loading...`。如果数据库没有原始坐标，该模式不可提供完整对比。 |

## 9. 顶部控制区

### 9.1 路径显示与差异

| 控件 | 作用 |
|---|---|
| `Color`滑块 | 调整背景路径颜色，由浅灰逐渐变为黑色。 |
| `BG`滑块 | 调整未选中背景路径的透明度。 |
| `Path Δ` | 按同一路径原始坐标与当前坐标的点位距离着色，红色表示变化较大。 |
| `Exit Δ` | 按出口到所属簇中心的距离在原始/当前坐标间的变化着色。 |
| Delta阈值滑块 | 在启用某个Delta叠加后，排除Delta值小于或等于当前阈值的路径。旁边标签显示实际阈值。 |

Delta叠加需要数据库同时保存当前坐标与原始坐标。`Path Delta`和`Exit Delta`用于不同问题：前者观察整条路径位移，后者侧重出口区域重排。

### 9.2 图层和选择

| 按钮 | 作用 |
|---|---|
| `Residues` | 显示或隐藏残基三维位置。 |
| `Res Labels` | 显示或隐藏当前选中残基的文字标签。 |
| `Entry/Exit` | 显示或隐藏入口点和出口点；入口为绿色，出口为红色。 |
| `EE: All` / `EE: Current` | 切换入口/出口点的范围：全部路径或当前有效路径。 |
| `Exit Cluster` | 开启后单击一个出口球，选择当前视角和缩放下视觉连通的一组出口。 |
| `Focus` | 隐藏背景路径，只显示当前选中/有效路径。快捷键为`F`。 |
| `Point Select` | 在“选择路径”和“选择入口/出口点”两种套索对象之间切换。 |
| `Lasso` | 开启强制套索模式，普通左键拖动也执行套索。快捷键为`L`。 |
| `Clear Sel` | 清除当前路径选择。快捷键为`Esc`。 |

### 9.3 参考蛋白和结合位点

| 控件 | 作用 |
|---|---|
| `Protein…` | 加载`.pdb`参考蛋白或`.pse` PyMOL会话。PSE中的非蛋白对象可作为结合位点表面显示。 |
| `Residue Observer` | 显示或隐藏右侧双数据集局部结构观察器。 |
| `Reference PDB` | 显示或隐藏已加载的参考蛋白。加载前不可用。 |
| 样式下拉框 | 在`cartoon`、`backbone`、`tube`和`ca_spheres`之间切换。 |
| 对齐下拉框 | `kabsch`按可用锚点刚体对齐；`translate`仅平移；`none`保留输入坐标。 |
| Opacity滑块 | 控制参考蛋白透明度。当前版本的主工具栏将该值固定为100%。 |
| `Clear` | 移除参考蛋白、PSE结合位点和相关缓存。 |
| `Sites` | 显示或隐藏从PSE载入的结合位点表面；没有位点时不可用。 |

### 9.4 帧范围

点击`Frame`打开范围菜单：

| 按钮 | 作用 |
|---|---|
| `Apply` | 使用输入的最小帧和最大帧筛选路径、三维视图及后续分析。 |
| `Clear` | 清除帧范围并恢复全部帧。 |

## 10. Datasets标签页

数据集树支持Ctrl/Shift多选。展开数据集可以看到CAVER Cluster子项，勾选一个或多个Cluster会将其路径加入当前选择。这里的Cluster是输入数据库中的CAVER空间聚类；`Paths`标签页中的Path Clusters则是用户手动建立的路径类别，两者不要混淆。

| 按钮 | 作用 |
|---|---|
| `Add Dataset` | 选择一个或多个数据集文件夹。程序递归查找主`.db`或`.pkl`及可选统计文件。 |
| `Edit Prefix` | 修改数据集在路径表、图表和比较视图中的短前缀。前缀在当前会话内保持唯一。 |
| `Pick Color` | 设置该数据集在三维视图和图表中的基础颜色。 |
| `Reference PDB` | 打开PDB/PSE选择器，将结构加载到中央三维视图。 |
| `Show Tunnels` / `Hide Tunnels` | 显示或隐藏选中的一个或多个数据集隧道。 |
| `Cluster Colors` / `Uniform Color` | 在按CAVER Cluster分别着色和按数据集统一着色之间切换。 |
| `Cluster Labels` / `Hide Labels` | 显示或隐藏三维Cluster标签。仅单选一个数据集时操作明确。 |
| `Remove` | 从当前会话移除选中的数据集，不删除磁盘文件。 |

树中的Cluster复选框用于路径选择，不会删除或重新计算聚类。选择不同数据集还会更新残基和比较控件的数据范围。

## 11. Paths标签页

路径表列包括`Dataset`、`Path ID`、`Frame`、`Length`、`Min R`、`Avg R`、`Avg Hydro`、`# Residues`和`Cluster`。单击列标题可排序；使用Ctrl/Shift可选择多行。

### 11.1 路径类别按钮

| 按钮 | 作用 |
|---|---|
| `Create` | 使用表格选中路径创建类别；若表格无选择，则使用当前有效路径或最近匹配路径。名称为空时自动生成时间戳名称。 |
| `Create From Current` | 从当前选择上下文创建类别，采用与`Create`相同的优先级。 |
| `Select` | 将当前类别中的路径设为当前选择。 |
| `Show` / `Hide` | 显示或隐藏当前类别的三维高亮。 |
| `Select & Show` | 同时选择并显示当前类别。 |
| `Set Color` | 为当前类别设置三维显示颜色。 |
| `Show All` | 显示所有手动类别。 |
| `Hide All` | 隐藏所有手动类别。 |
| `Excl All` | 将所有类别标记为候选排除集合。 |
| `Incl All` | 清除所有类别排除状态。 |
| `Export` | 将当前类别路径导出为PDB。可选择按帧合并或将同帧路径拆分为独立文件。 |
| `Delete` | 删除当前手动类别，不删除原始路径。 |
| `Add Selected to Category` | 将路径表中选中的行追加到当前类别并去重。 |

类别列表操作：

- 单击类别：设为当前类别。
- 双击类别：切换三维显示。
- 右键类别：可显示/隐藏、加入/移除候选排除、导出或删除。
- `[V]`表示可见，`[X]`表示被排除，`[V|X]`表示两种状态同时存在。

### 11.2 路径表视图按钮

| 控件 | 作用 |
|---|---|
| `All Paths` | 显示满足当前帧/残基筛选的全部路径。 |
| `Selected (N)` | 切换到当前有效路径或可见类别路径的专用视图。 |
| `Load Selected` | 在`Selected`页中真正加载当前选择，并将其发送到剖面图。 |

## 12. Residue Compare标签页

该标签页要求两个数据集都提供`residue_statistics.csv`。File B减File A为正时显示红色，为负时显示蓝色，接近零时显示白色。

| 控件/按钮 | 作用 |
|---|---|
| `File A`、`File B` | 选择要比较的两个已加载数据集。 |
| `Reload` | 清除当前联动残基选择并刷新由选择状态驱动的视图；更换File A或File B时会自动重新读取相应CSV。 |
| `Open Heatmap` | 在独立窗口中打开更大的残基变化热图。 |
| 热图窗口`Close` | 关闭独立热图窗口。 |

表格可比较Affected Points、Affected Paths、Mean/Min/Max Radius和Cooperative Residues等数值。单击一行会在下方显示该残基的主要变化；单击File A或File B对应单元格可将该数据集中的残基发送到三维选择；协同残基列会显示增加和减少的合作伙伴详情。

## 13. Residue Combination Compare标签页

该标签页用于分析2-4个残基形成的组合、残基对和瓶颈基序。至少一个数据集需要`residue_combination_statistics.csv`；跨数据集比较时应同时选择File A和File B。

### 13.1 数据、显示和筛选按钮

| 控件/按钮 | 作用 |
|---|---|
| `File A`、`File B` | 选择一个数据集进行单集分析，或选择两个数据集进行差异比较。 |
| `Residue 1-4` | 指定需要重点检查的2-4个残基；未使用的槽位保持`-`。 |
| `Size` | 只显示2、3或4残基组合，或显示全部。 |
| `Color` | 按Affected Paths、Affected Points、Mean Radius、Path-weighted Radius或Radius Range为组合标记着色。比较模式下使用File B减File A。 |
| `+ Group` | 新增一个筛选组。每组可选择属性、Increase/Decrease/Neutral趋势和绝对阈值。 |
| 筛选组`Remove` | 删除该筛选组。 |
| `Show` | 在中央三维视图显示或隐藏当前可见组合的质心标记。 |
| `Sel Only` | 只显示包含当前残基选择的组合。 |
| `Bottleneck` | 只显示当前所选路径瓶颈点处出现的组合。 |
| `Show AB Res` | 在三维视图中显示当前可见File A/File B组合涉及的全部残基。 |
| `Clear` | 清空四个残基槽位。 |
| `Load DS` | 按完整数据集范围加载所选组合CSV。 |
| `Load Path` | 以当前路径选择为上下文重新计算/加载组合。 |
| `Refresh` | 刷新当前表格、筛选和显示。 |
| `Save Table` | 将当前可见组合表保存为CSV。 |

单击组合表行会更新详细统计；单击存在状态或File A/File B相关单元格会联动三维残基与Residue Observer。开启`Show`后，单击中央三维组合标记也会返回并选中对应表格记录。

### 13.2 Matched Paths按钮

| 控件/按钮 | 作用 |
|---|---|
| `Source` | 在File A/File B中指定源数据集；另一个数据集作为目标。 |
| `Keep` | 设置`K`保留比例，取值应在0到1之间。 |
| `Query` | 使用源路径的残基集合、出口区域和路径长度约束检索目标路径。 |
| `Compare` | 对当前匹配结果计算残基对/组合差异；没有匹配锁定时，直接比较当前File A/File B路径。 |
| `Reset` | 清除本轮匹配路径和比较结果，恢复普通选择上下文。 |

执行`Query`前必须：

1. 在File A和File B中选择两个数据集；
2. 在`Source`中指定源数据集；
3. 在路径表、CAVER Cluster复选框、三维套索或手动类别中选中源路径；
4. 设置`Keep`后点击`Query`。

`Keep=0.8`时，出口约束保留距离初始出口中心最近的80%源路径并重新计算出口球；长度约束去除最短10%和最长10%，使用中间80%的长度范围；残基约束仍使用全部源路径的残基并集。目标路径只有在出口位于参考出口球内、长度位于保留区间内且其非空残基集合是源残基并集的子集时才会保留。查询完成后，源路径保持选中，匹配到的目标路径被加入选择并在路径表中高亮。

## 14. Residue Observer

点击顶部`Residue Observer`显示右侧面板。它包含File A和File B两个蛋白质视图以及中间时间轴。

| 控件/按钮 | 作用 |
|---|---|
| 残基输入框 | 输入逗号分隔的残基编号，例如`72,149`。 |
| `Apply` | 在两个Observer视图中高亮输入残基。 |
| 两个`MD Dataset`下拉框 | 分别为上、下视图选择逐帧PDB来源。 |
| `Model` | 选择Backbone、Cartoon、Tube或CA Spheres。 |
| `Opacity` | 调整Observer蛋白质透明度。 |
| `Prev` | 移动到上一帧。 |
| `Frame`数值框 | 直接跳转到指定帧。 |
| `Next` | 移动到下一帧。 |
| `Highlight` | 选择Ball-and-Stick、Sticks或Spheres残基高亮样式。 |

单击时间轴可跳转帧。若数据集包含`MD_path.txt`，程序优先从指定逐帧PDB目录建立序列；否则尝试在数据集目录附近寻找PDB。没有逐帧PDB时，Observer仍可使用`representative_frame.pdb`显示静态结构，但Prev/Next会保持禁用。

## 15. 底部路径图表

路径选择发生变化后，图表不会立即重算。状态文字会提示当前选择数量，需要手动同步。

| 按钮 | 作用 |
|---|---|
| `Sync Charts` | 读取当前可见有效路径的剖面，更新Profile和Statistics图。 |
| `Chart Window` | 将图表移到独立的大型同步工作区；主窗口底部显示分离提示。 |

### 15.1 Profile图表控件

| 控件/按钮 | 作用 |
|---|---|
| `Controls` | 展开或收起图表控制区。 |
| `Color` | 按Radius、Hydrophobicity、Frame或Dataset着色。 |
| `Advanced` | 显示自定义X/Y轴选项。可选Radius、Hydrophobicity、Frame和Path Length。 |
| `Path Selection` | 开启矩形框选；拖动矩形后选中经过范围的路径。再次点击可关闭。 |
| `Residue Stats` | 开启点级矩形框选，并统计选中剖面点关联的残基。 |
| `Opacity` | 调整剖面线透明度。 |
| `Depth` | 调整颜色深浅/视觉层次。 |
| 数据集复选框 | 在Chart Workspace中按数据集显示或隐藏曲线，并同步Statistics图。 |
| `Keep Only` | 仅保留图表框选得到的路径，同时替换主界面当前选择。 |
| `Exclude` | 从图表和主界面当前选择中移除框选路径。 |
| `Undo` | 撤销最近一次Keep Only/Exclude；最多保留最近5步图表操作。 |
| `Clear` | 清除图表矩形选择，不等同于删除数据。 |

图表视图禁用了滚轮缩放，矩形拖动仅在`Path Selection`或`Residue Stats`模式下生效。

### 15.2 Statistics图

Statistics图根据当前Profile图中的路径显示均值和分位范围，并继承颜色、X/Y轴及数据集筛选。该图主要用于比较路径集合总体趋势，不提供独立的路径选择按钮。

## 16. Chart Workspace按钮

| 控件/按钮 | 作用 |
|---|---|
| `Close` | 关闭独立图表窗口，并把图表恢复到主窗口底部。 |
| Residue Summary的`Path Residues` | 按所选图表路径统计全部路径残基。 |
| Residue Summary的`Bottleneck` | 仅统计各路径最小半径点的残基。 |
| Residue Summary的`Compare` | 打开按数据集组织的残基摘要比较热图。 |
| Path Filter的`Select Filtered` | 按Hydrophobicity和Frame输入范围筛选当前图表路径，并同步主界面选择。 |
| `Revert` | 恢复应用筛选前的路径选择。 |
| `Clear` | 清空Hydrophobicity和Frame筛选输入。 |

## 17. 快捷键汇总

| 快捷键 | 作用 |
|---|---|
| `L` | 开启/关闭强制Lasso模式 |
| `F` | 开启/关闭Focus模式 |
| `Esc` | 清除当前选择和未完成的套索轮廓 |
| `Shift + 左拖` | 套索追加选择 |
| `Alt + Shift + 左拖` | 套索移除选择 |
| `Shift + 右拖` | 套索移除选择 |

## 18. 推荐分析工作流

### 18.1 单数据集路径检查

1. `Add Dataset`加载数据。
2. `Show Tunnels`显示路径。
3. 使用Frame范围或CAVER Cluster复选框缩小范围。
4. 在三维视图套索选择路径。
5. 在`Paths`中创建手动类别并设置颜色。
6. 点击`Sync Charts`查看半径和疏水性剖面。
7. 使用`Export`导出目标路径PDB。

### 18.2 跨数据集约束匹配

1. 加载两个数据集并设置不同Prefix/Color。
2. 在源数据集中选择目标路径集合。
3. 打开`Residue Combination Compare`并选择File A/File B。
4. 设置`Source`和`Keep`，点击`Query`。
5. 检查状态栏中的源路径数、出口半径、长度范围和目标命中数。
6. 点击`Compare`计算匹配集合的残基对/组合变化。
7. 使用`Show`、`Show AB Res`和`Residue Observer`检查局部结构。

### 18.3 残基网络变化分析

1. 确保两个数据集都有残基统计和组合统计CSV。
2. 在`Residue Compare`识别变化明显或合作伙伴发生变化的残基。
3. 点击对应File A/File B单元格将残基联动到三维视图。
4. 在`Residue Combination Compare`选择2-4个候选残基。
5. 使用`Bottleneck`限定瓶颈组合，并用筛选组查看Increase/Decrease。
6. 在Observer中逐帧检查残基距离和侧链构象。
7. 使用`Save Table`保存当前组合结果。

## 19. 输出文件

- 路径类别导出：PDB文件；拆分模式下额外生成TXT元数据摘要。
- 残基组合结果：`Save Table`保存当前可见CSV。
- PKL首次加载：在数据目录生成`tunnel_data.db`。

程序中的`Remove`、`Delete`和筛选操作只改变当前会话状态，不会删除原始数据集文件。

## 20. 常见问题

### 20.1 `DLL load failed while importing QtCore/QtWidgets`

通常是Qt DLL冲突。退出Anaconda base环境，使用标准Python 3.10/3.11重新创建`.venv`并安装`requirements.txt`。不要混用conda安装的Qt和pip安装的PySide6。

### 20.2 三维区域黑屏或出现OpenGL错误

- 更新显卡驱动。
- 确认OpenGL版本不低于3.2。
- 尽量在本机桌面会话运行；Windows远程桌面可能无法提供合适的OpenGL上下文。
- 确认`pyvista`、`pyvistaqt`和`vtk`来自同一虚拟环境。

### 20.3 图表显示`pyqtgraph not installed`

```powershell
python -m pip install pyqtgraph
```

然后重新启动程序。

### 20.4 Add Dataset提示没有找到数据库

确认所选的是“数据集文件夹”而不是单个文件，并且目录内包含`.db`、`preprocessed_paths.pkl`或其他`.pkl`。检查文件扩展名没有被系统隐藏后误改。

### 20.5 PKL首次加载很慢

程序会将PKL转换为SQLite并创建索引，这是一次性过程。等待状态结束，不要同时重复添加同一目录。后续会优先使用生成的`tunnel_data.db`。

### 20.6 Residue Compare或Combination Compare没有数据

确认数据集目录包含正确命名的`residue_statistics.csv`和`residue_combination_statistics.csv`，并重新添加数据集。程序在添加时记录这些文件的位置。

### 20.7 Residue Observer没有逐帧内容

检查`MD_path.txt`中的路径是否在当前机器上存在，以及目标目录是否直接或在子目录中包含PDB文件。仅有`representative_frame.pdb`时只能显示静态结构。

### 20.8 界面控件被压缩或看不全

- 最大化窗口。
- 拖动Control Panel右侧窄边调整宽度。
- 拖动中央/图表或三维/Observer分隔条。
- 标签页标题过长时使用标签栏两侧滚动按钮。
