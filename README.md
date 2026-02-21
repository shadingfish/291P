# Communication and Memory Overhead Tool

**A tool to calculate the communication and memory overhead of collaborative operations under flexible system configuration.** It supports flexible topology and GPU count (switch, tree, mesh, torus, hierarchical), and collaborative operations including AllReduce, AllGather, ReduceScatter, Broadcast, All2All (and P2P for Pipeline Parallelism). After developing the tool, you can use it to analyze an existing configuration (e.g. DP/TP/PP/CP with given GPU count and topology).

当前版本支持 **Ring、Tree、Hierarchical** 三种拓扑；**Switch、Mesh、Torus** 为计划中。集体操作包括 AllReduce、AllGather、ReduceScatter、Broadcast、All2All；**DP、TP、CP** 已在分析中接入（CP 使用 All2All）；**PP** 的 P2P 为计划中。基础版本 **不包含 ZeRO**（每 GPU 内存 = 完整模型 + 梯度 + 优化器状态）。提供 **每维度** 的通信报告，以及可选的 **每 step 总通信时间上界**（当提供 num_layers 且 DP/TP/CP 共存时）。所有公式均引用课程讲义；详见 [comm_overhead/REFERENCES.md](comm_overhead/REFERENCES.md)。

---

# Part A — English

## 1. Overview

This tool computes:

- **Communication overhead**: Latency (seconds), communication volume (bytes), and number of steps for each collective operation (AllReduce, AllGather, ReduceScatter, Broadcast, All2All) under a given **topology** (how GPUs are connected) and **tensor size M** (bytes).
- **Memory overhead**: Per-GPU memory (bytes) for training with a given model size, under **ZeRO-0** (no optimizer state partitioning).

Inputs include: number of GPUs **N**, topology kind (ring, tree, hierarchical; later switch, mesh, torus), link bandwidth **B** and latency **α**, and for hierarchical topology intra-node vs inter-node **B1/α1** and **B2/α2**. For full **analysis**, you also provide parallelism degrees (DP, TP, PP, CP), model parameter count, and optionally batch size, sequence length, and hidden size for TP/PP/CP communication sizing.

Outputs include: per-collective latency and volume (one report per dimension: DP, TP, CP when enabled), per-GPU memory, and a short summary. When `num_layers` is set and DP/TP/CP coexist, a **total communication time per step (upper bound, no overlap)** is also reported. When no parallelism (dp=tp=cp=1), only memory overhead is reported; communication overhead is zero.

## 2. Project structure

```
291P/
├── README.md                       # This file: overview, usage, extension, remaining tasks
├── example_usage.py                # Example runs: collective, memory, full analysis
├── run_cli.py                      # CLI: collective | analysis | list-cmds | run-all
├── requirements.txt               # Python dependencies
├── comm_overhead/                  # Main package
│   ├── __init__.py                # Public API (TopologyKind, build_topology, collective_*, analyze_config, Config, etc.)
│   ├── REFERENCES.md               # Formula sources (L4, L5, L6, L7 slides)
│   ├── constants.py               # Default B, alpha, dtype sizes (e.g. NVLink, InfiniBand)
│   ├── topology.py                # Topology: Ring, Tree, Hierarchical (Switch/Mesh/Torus TBD)
│   ├── collective.py              # Collective latency/volume per op and topology
│   ├── memory.py                  # Per-GPU memory (ZeRO-0 only)
│   └── analyze.py                 # Orchestrates topology + collective + memory for a config
├── scripts/
│   └── run_all_commands.sh        # Runs suggested CLI commands for comparison
└── docs/
    ├── TASKS.md                   # Remaining tasks and assignment (detailed, EN+ZH)
    └── 分布式LLM训练与通信-背景知识.md
```

- **topology.py**: Input = kind, N, B/α (or B1/α1, B2/α2, gpus_per_node). Output = `TopologyDesc`. New topologies (e.g. MESH, TORUS, SWITCH) are added here.
- **collective.py**: Input = collective kind, M (bytes), `TopologyDesc`. Output = `CollectiveResult` (latency_s, volume_bytes, steps). New collectives or topology-specific formulas are added here.
- **memory.py**: Input = num_parameters, dtype. Output = per-GPU bytes (and optional breakdown). ZeRO-1/2/3 can be added later.
- **analyze.py**: Input = `Config` (GPUs, topology, DP/TP/CP/PP degrees, model size, optional batch/seq/hidden/num_layers). Output = `AnalysisResult` (per-GPU memory, collective reports per dimension, optional total_comm_per_step). DP, TP, and CP are implemented; PP (P2P) is planned.

## 3. Inputs and outputs by module

| Module       | Inputs | Outputs |
|-------------|--------|---------|
| **topology**  | kind, N, B, α (or B1/α1, B2/α2, gpus_per_node) | `TopologyDesc` |
| **collective**| `CollectiveKind`, M (bytes), `TopologyDesc` | `CollectiveResult`: latency_s, volume_bytes, steps; optional intra_latency_s, inter_latency_s |
| **memory**    | num_parameters, optional weight/grad/optimizer bytes per param | per_gpu_memory_bytes; optional `MemoryBreakdown` |
| **analyze**   | `Config` (num_gpus, topology_kind, dp/tp/cp/pp degree, num_parameters, optional batch/seq/hidden/num_layers) | `AnalysisResult`: per_gpu_memory_bytes, collective_reports (DP/TP/CP), dp_allreduce_latency_s, summary, gpus_per_node, num_nodes, optional total_comm_time_per_step_upper_bound_s |

## 4. Topology × Collective support (current vs planned)

| Collective   | RING | TREE | HIERARCHICAL | SWITCH | MESH | TORUS |
|-------------|------|------|--------------|--------|------|-------|
| ALL_REDUCE  | yes  | yes* | yes          | TBD    | TBD  | TBD   |
| ALL_GATHER  | yes  | ring-style | yes     | TBD    | TBD  | TBD   |
| REDUCE_SCATTER | yes | ring-style | yes     | TBD    | TBD  | TBD   |
| BROADCAST   | yes  | ring-style | yes     | TBD    | TBD  | TBD   |
| ALL_TO_ALL  | yes  | ring-style | yes*** | TBD    | TBD  | TBD   |

- \* TREE: Only AllReduce has a dedicated tree algorithm; other collectives use ring-style formula (B, α).
- \*\*\* HIERARCHICAL All2All: simplified (B2/α2 dominant).  
- TBD = to be implemented (see [docs/TASKS.md](docs/TASKS.md)).

## 5. How to run

### 5.1 Example script (from project root 291P)

```bash
PYTHONPATH=. python example_usage.py
```

This runs: (1) one AllReduce latency (Ring, N=8, M=140GB), (2) per-GPU memory for 70B parameters (no ZeRO), (3) full analysis for a small DP=8 config, and (4) hierarchical examples with/without gpus_per_node.

### 5.2 Command-line interface

Use `run_cli.py` with one of four modes: **collective**, **analysis**, **list-cmds**, **run-all**.

```bash
# List all suggested commands (copy-paste to run each)
PYTHONPATH=. python run_cli.py list-cmds

# Single collective: topology, N, optional gpus_per_node (for hierarchical)
PYTHONPATH=. python run_cli.py collective --topology ring --N 8 --M 140e9
PYTHONPATH=. python run_cli.py collective --topology tree --N 8 --M 140e9
PYTHONPATH=. python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9

# Full analysis (DP=8, memory)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9

# Memory only (no P: no collective reports, only per-GPU memory)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 1 --dp 1 --tp 1 --cp 1 --params 70e9

# With TP=2 (requires --seq-length and --hidden-size)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096

# With CP=4 (Context Parallelism, All2All)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --cp 4 --params 70e9 --seq-length 1024 --hidden-size 4096

# DP+TP+CP with --num-layers (reports total comm/step upper bound)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32

# Run all predefined combinations and print a CSV table
PYTHONPATH=. python run_cli.py run-all

# Verbose (show formula and calculation)
PYTHONPATH=. python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9 --verbose
PYTHONPATH=. python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9 --verbose
```

**Common arguments:**

- `--topology`: ring | tree | hierarchical (later: switch | mesh | torus)
- `--N`: number of GPUs for collective mode
- `--M`: tensor size in bytes (e.g. 140e9)
- `--gpus-per-node`: for hierarchical only (e.g. 4)
- `--num-gpus`, `--dp`, `--tp`, `--cp`, `--params`: for analysis mode
- `--seq-length`, `--hidden-size`, `--batch-size`: optional, for TP/CP (and PP when implemented) sizing
- `--num-layers`: optional; when set with DP+TP+CP, reports total communication time per step (upper bound, no overlap)
- `--tree-algo`: use tree algorithm for AllReduce (flat topologies only)
- `--verbose`: print formula and intermediate values

### 5.3 Batch script

To run each suggested command one by one (from 291P root):

```bash
bash scripts/run_all_commands.sh
```

The script runs: single AllReduce (topology × N), full analysis (DP=8), memory-only (no P), TP=2, CP=4, DP+TP+CP with num-layers, and one verbose example. Or use `run_cli.py run-all` for a single CSV-style table.

## 6. How to extend (for team members)

1. **New topology (e.g. Switch, Mesh, Torus)**  
   Edit `topology.py`: add to `TopologyKind`, extend `TopologyDesc` if needed (e.g. n_x, n_y for mesh), implement in `build_topology`. In `collective.py`, add a branch for the new topology in `collective_latency_and_volume()` and implement (or call) the corresponding latency/volume formulas. Cite sources in `REFERENCES.md`.

2. **New collective or algorithm**  
   Edit `collective.py`: add to `CollectiveKind` if it’s a new op; add a `_..._flat` or topology-specific helper with formula and reference; call it from `collective_latency_and_volume`.

3. **ZeRO (memory)**  
   Edit `memory.py`: add a `zero_stage` (0/1/2/3) parameter and formulas from L5 slides 21–26; return per-GPU memory and, if needed, communication equivalent. Optionally extend `analyze.py` to pass ZeRO stage and report ZeRO-related collectives.

4. **PP in analyze**  
   **CP** is already implemented (All2All over cp_degree GPUs when cp_degree>1 and seq_length/hidden_size are set). **PP**: Add a P2P latency model (e.g. in collective or `p2p.py`). In `analyze.py`, when `pp_degree > 1`, compute activation size (e.g. 2·batch·seq·hidden·dtype) and call the P2P model; append a report. See [docs/TASKS.md](docs/TASKS.md) Task 4.

5. **References and formula documentation**  
   When adding formulas, **write the formula clearly in code comments** (e.g. next to the calculation or in the helper’s docstring) and **cite the source** (e.g. "L4 slide 34", "Lecture 5 slide 4", or paper + equation). Add or update the corresponding entry in `REFERENCES.md`.

## 7. Remaining tasks (summary)

The following tasks are to be implemented and assigned; **detailed steps, file-level changes, and cmdline examples** are in **[docs/TASKS.md](docs/TASKS.md)**.

| # | Task | What to do | Main files |
|---|------|------------|------------|
| **1** | SWITCH topology | Implement SWITCH in topology.py and collective.py (all five collectives); add to CLI and run_all/list_cmds; ensure analysis uses it for DP/TP/CP when `--topology switch`. | topology.py, collective.py, REFERENCES.md, run_cli.py, optional scripts/run_all_commands.sh |
| **2** | MESH topology | Implement MESH (2D grid n_x×n_y=N) in topology.py and collective.py; optional --mesh-nx/--mesh-ny in CLI; add to list_cmds and run_all. | topology.py, collective.py, REFERENCES.md, analyze.py (if mesh dims in Config), run_cli.py, optional scripts |
| **3** | TORUS topology | Implement TORUS (2D mesh with wrap-around) in topology.py and collective.py; add to CLI and list_cmds/run_all. | topology.py, collective.py, REFERENCES.md, run_cli.py, optional scripts |
| **4** | PP (Pipeline Parallelism) | Add P2P latency model; in analyze.py when pp_degree>1 compute activation size and P2P latency, add report; add --pp to CLI and document usage scenarios. For all new formulas: document in comments + REFERENCES.md. | collective.py or p2p.py, analyze.py, REFERENCES.md, run_cli.py |
| **5** | Integration + presentation | Merge branches, run run-all and list-cmds; everyone participates in PPT; one person presents. | No fixed code list; optional README/docs polish |

**Note:** Once a new topology is implemented, **DP/TP/CP** are automatically supported in analysis (same `analyze_config` and `run_cli.py analysis`); no separate “DP/TP/CP implementation” per person is required. **PP** is a separate feature (P2P, not a topology) and is assigned as a single task (Task 4). **Formula documentation:** For every new formula (Tasks 1–4), write the formula explicitly in code comments and cite the source (e.g. PPT slide or paper); see [docs/TASKS.md](docs/TASKS.md).

## 8. References

- **L4**: Lecture 4 LLM Training - Foundations (Ring/Tree AllReduce, AllGather, ReduceScatter, hierarchical).
- **L5**: Lecture 5 LLM Training - DP (ZeRO memory, hierarchical ring formulas).
- **L6**: Lecture 6 LLM Training - PP+TP (TP communication, PP P2P).
- **L7**: Lecture 7 LLM Training - CP (All2All, Ulysses).

See [comm_overhead/REFERENCES.md](comm_overhead/REFERENCES.md) for formula-to-slide mapping.

---

# Part B — 中文

## 1. 概述

本工具用于在**灵活的系统配置**下计算**协作操作的通信与内存开销**，支持灵活的**拓扑与 GPU 数量**（当前已实现 ring、tree、hierarchical；计划中 switch、mesh、torus），以及 **AllReduce、AllGather、ReduceScatter、Broadcast、All2All**。**DP、TP、CP** 已在分析中接入（CP 使用 All2All）；**PP** 的 P2P 为计划中。可用于分析给定配置（DP/TP/CP/PP + GPU 数与拓扑）的通信与内存表现。

- **通信开销**：在给定**拓扑**与**张量大小 M** 下，各集体操作的延迟、通信量、步数；按维度分别报告（DP/TP/CP）。当提供 **num_layers** 且 DP/TP/CP 共存时，额外报告**每 step 总通信时间上界**（无重叠假设）。当无任何 P（dp=tp=cp=1）时，通信开销为 0，仅报告内存。
- **内存开销**：在 **ZeRO-0** 下，给定模型规模时的每 GPU 内存（字节）；与是否开启 P 无关。

输入包括：GPU 数 **N**、拓扑类型（ring / tree / hierarchical；后续 switch / mesh / torus）、链路带宽 **B** 与延迟 **α**；分层拓扑下还有节点内/节点间 **B1/α1**、**B2/α2**。进行完整**分析**时还可提供并行度（DP/TP/PP/CP）、模型参数量，以及可选的 batch/seq/hidden 用于 TP/PP/CP 的通信规模估算。

输出包括：各集体操作的延迟与通信量、每 GPU 内存、简要摘要，便于比较不同拓扑或配置。

## 2. 项目结构

```
291P/
├── README.md                       # 本文件：概述、用法、扩展、剩余任务
├── example_usage.py                # 示例：集体、内存、完整分析
├── run_cli.py                      # 命令行：collective | analysis | list-cmds | run-all
├── requirements.txt               # 依赖
├── comm_overhead/                  # 主包
│   ├── __init__.py                # 对外 API
│   ├── REFERENCES.md              # 公式来源（L4–L7）
│   ├── constants.py               # 默认 B、α、dtype 等
│   ├── topology.py                # 拓扑：Ring、Tree、Hierarchical（Switch/Mesh/Torus 待实现）
│   ├── collective.py             # 各集体操作的延迟/体积
│   ├── memory.py                  # 每 GPU 内存（仅 ZeRO-0）
│   └── analyze.py                 # 根据配置编排拓扑+集体+内存
├── scripts/
│   └── run_all_commands.sh        # 批量运行建议命令
└── docs/
    ├── TASKS.md                   # 剩余任务与分工（详细，中英）
    └── 分布式LLM训练与通信-背景知识.md
```

各模块职责与扩展点见上文 Part A 第 2 节。

## 3. 模块输入输出（简要）

| 模块       | 输入 | 输出 |
|------------|------|------|
| **topology**  | kind, N, B, α（或 B1/α1, B2/α2, gpus_per_node） | `TopologyDesc` |
| **collective**| 集体类型、M（字节）、`TopologyDesc` | `CollectiveResult`：latency_s, volume_bytes, steps 等 |
| **memory**    | 参数量等 | 每 GPU 内存及可选 breakdown |
| **analyze**   | `Config`（GPU 数、拓扑、DP/TP/CP/PP、参数量，可选 batch/seq/hidden/num_layers） | `AnalysisResult`（内存、DP/TP/CP 集体报告、摘要，可选 total_comm_per_step 上界） |

## 4. 拓扑 × 集体支持（当前 vs 计划）

与 Part A 第 4 节表格一致：RING/TREE/HIERARCHICAL 已实现；SWITCH/MESH/TORUS 为 TBD；TREE 仅 AllReduce 有专用树算法，其余为 ring-style。

## 5. 如何运行

### 5.1 示例脚本（在项目根目录 291P 下）

```bash
PYTHONPATH=. python example_usage.py
```

### 5.2 命令行

```bash
# 列出建议命令
PYTHONPATH=. python run_cli.py list-cmds

# 单次集体：拓扑、N、可选 gpus_per_node
PYTHONPATH=. python run_cli.py collective --topology ring --N 8 --M 140e9
PYTHONPATH=. python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9

# 完整分析（DP=8，内存）
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9

# 仅内存（无 P：无集体报告，仅每 GPU 内存）
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 1 --dp 1 --tp 1 --cp 1 --params 70e9

# 带 TP=2、CP=4
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --cp 4 --params 70e9 --seq-length 1024 --hidden-size 4096

# DP+TP+CP 并给 num-layers（输出每 step 总通信时间上界）
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32

# 运行全部预定义组合并输出 CSV 表
PYTHONPATH=. python run_cli.py run-all

# 详细输出（公式与中间结果）
PYTHONPATH=. python run_cli.py collective --topology hierarchical --N 8 --gpus-per-node 4 --M 140e9 --verbose
```

常用参数：`--dp` / `--tp` / `--cp`、`--num-layers`（与 DP+TP+CP 一起用时输出总通信/step）、`--seq-length` / `--hidden-size`（TP/CP 必需）。详见 Part A 第 5.2 节。

### 5.3 批量脚本

```bash
bash scripts/run_all_commands.sh
```

脚本会依次运行：单次 AllReduce、DP=8 分析、仅内存（无 P）、TP=2、CP=4、DP+TP+CP+num-layers、以及一条 verbose 示例。或使用 `run_cli.py run-all` 一次性输出 CSV 对比表。

## 6. 如何扩展

与 Part A 第 6 节对应：新拓扑在 topology.py + collective.py；新集体/算法在 collective.py；ZeRO 在 memory.py。**CP** 已在 analyze 中实现；**PP** 在 analyze 中接入 P2P 即可，见 [docs/TASKS.md](docs/TASKS.md) 任务 4。**所有新增公式均须在注释中写清并注明出处（PPT 或论文），并在 REFERENCES.md 中更新。**

## 7. 剩余任务（摘要）

以下任务需实现并分工；**具体步骤、需修改文件、命令行示例**见 **[docs/TASKS.md](docs/TASKS.md)**。

| # | 任务 | 内容 | 主要文件 |
|---|------|------|----------|
| **1** | SWITCH 拓扑 | 在 topology.py 与 collective.py 中实现 SWITCH 及五种集体；接入 CLI、list_cmds、run_all；analysis 下 `--topology switch` 即支持 DP/TP/CP。 | topology.py, collective.py, REFERENCES.md, run_cli.py, 可选 scripts |
| **2** | MESH 拓扑 | 实现 2D 网格（n_x×n_y=N）及集体公式；可选 --mesh-nx/--mesh-ny；接入 CLI 与 run_all。 | topology.py, collective.py, REFERENCES.md, 可选 analyze.py, run_cli.py |
| **3** | TORUS 拓扑 | 实现带环绕的 2D 网格及集体公式；接入 CLI 与 run_all。 | topology.py, collective.py, REFERENCES.md, run_cli.py |
| **4** | PP（流水线并行） | P2P 延迟模型；analyze 中 pp_degree>1 时计算激活大小与 P2P 延迟并写报告；CLI 增加 --pp；文档化使用场景。所有新公式须在注释中写清并注明出处。 | collective.py 或 p2p.py, analyze.py, REFERENCES.md, run_cli.py |
| **5** | 整合与演讲 | 合并分支、跑 run-all/list-cmds；全员参与 PPT；一人主讲。 | 无固定代码列表；可选 README/docs 润色 |

**说明**：新拓扑实现后，**DP/TP/CP** 在 analysis 中会**自动**使用该拓扑，无需每人单独做「DP/TP/CP 实现」。**PP** 为独立功能（P2P），单独作为任务 4 由一人完成。**公式与出处**：任务 1–4 中所有新增公式都必须在代码注释中写清公式并注明出处（如 PPT 页码或论文），详见 [docs/TASKS.md](docs/TASKS.md)。

## 8. 参考文献

与 Part A 第 8 节一致；详见 [comm_overhead/REFERENCES.md](comm_overhead/REFERENCES.md)。
