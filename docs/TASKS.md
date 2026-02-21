# Remaining Tasks & Task Assignment / 剩余任务与分工

This document lists the remaining implementation tasks for the Communication and Memory Overhead Tool, with **what to implement**, **which files to modify**, and **cmdline examples**, so tasks can be clearly assigned to team members.

本文档列出通信与内存开销工具的剩余实现任务，包括**要实现什么**、**需要修改哪些文件**、以及**命令行示例**，便于将任务清晰分发给组员。

---

## Task assignment summary / 任务分工总览

| Task | Owner | Scope |
|------|--------|--------|
| **Task 1** | Person A | SWITCH topology + collective formulas + CLI/analysis support + cmdline examples |
| **Task 2** | Person B | MESH topology + collective formulas + CLI/analysis support + cmdline examples |
| **Task 3** | Person C | TORUS topology + collective formulas + CLI/analysis support + cmdline examples |
| **Task 4** | Person D | PP (Pipeline Parallelism): P2P model + analyze integration + usage scenarios + cmdline examples |
| **Task 5** | You + All | Integration, testing, PPT (everyone), presentation (you) |

**Note on DP/TP/CP:** Once a new topology (Switch/Mesh/Torus) is implemented in `topology.py` and `collective.py`, the existing `analyze.py` and `run_cli.py` will **automatically** use it for DP (gradient AllReduce), TP (per-layer AllReduce), and CP (All2All) when the user selects `--topology switch/mesh/torus`. No separate "DP/TP/CP implementation" per person is needed—only implement the topology and collectives, then add the topology to the CLI and run_all/list_cmds.

**关于 DP/TP/CP：** 新拓扑在 `topology.py` 与 `collective.py` 实现后，现有 `analyze.py` 与 `run_cli.py` 会在用户选择 `--topology switch/mesh/torus` 时**自动**将该拓扑用于 DP（梯度 AllReduce）、TP（每层 AllReduce）、CP（All2All）。无需每人再单独做「DP/TP/CP 实现」，只需实现拓扑与集体通信并接入 CLI 与 list-cmds/run-all。

**Formula documentation (公式与出处)：** For each task that adds new formulas (Tasks 1–4), you **must** write the formulas clearly in code comments (e.g. in the helper functions or next to the calculation), and **cite the source** (e.g. "L4 slide 34", "Lecture 5 slide 4", or a paper title + equation number). Add or update the corresponding entry in `comm_overhead/REFERENCES.md`. 所有涉及新公式的任务（任务 1–4）中，**必须在代码注释里写清使用的公式**（可在辅助函数内或计算处），并**注明出处**（如「L4 第 34 页」「Lecture 5 slide 4」或论文名+公式编号）；同时在 `comm_overhead/REFERENCES.md` 中补充或更新对应条目。

---

# Part A: English — Detailed task descriptions

---

## Task 1: SWITCH topology

### Goal
Implement the **SWITCH** topology (idealized star / full bisection bandwidth) and latency/volume formulas for all supported collectives (AllReduce, AllGather, ReduceScatter, Broadcast, All2All). Ensure the topology can be used in `analysis` mode for DP/TP/CP (by adding it to CLI). Provide cmdline examples.

### Implementation steps
1. **Topology**
   - In `comm_overhead/topology.py`:
     - Add `SWITCH = "switch"` to `TopologyKind` enum.
     - `TopologyDesc` already has `B` and `alpha`; SWITCH can use these (one link bandwidth/latency from each GPU to the switch). No new fields required unless you want to model switch-specific params.
     - In `build_topology()`, add a branch for `kind == TopologyKind.SWITCH`: return a `TopologyDesc` with `kind=SWITCH`, `N`, `B`, `alpha` (use defaults from constants if not provided).
2. **Collectives**
   - In `comm_overhead/collective.py`:
     - In `collective_latency_and_volume()`, after the existing `if topology.kind == TopologyKind.HIERARCHICAL` block, add handling for `TopologyKind.SWITCH`.
     - For SWITCH, typical model: AllReduce = 2 steps (reduce to one root, then broadcast); AllGather/ReduceScatter/Broadcast can be 1–2 steps; All2All can be one round. Implement helper functions e.g. `_switch_allreduce_flat(M, N, B, alpha)` and similar for other collectives, or document that you use a specific formula (e.g. 2*(M/B + alpha) for AllReduce).
     - **In each helper (or at the calculation site), add a comment that states the formula explicitly and the source**, e.g. `# AllReduce on switch: latency = 2*(M/B + alpha); steps = 2. Source: L4 slide XX / paper YYY Eq. Z.`
     - Update the SUPPORTED COMBINATIONS table in the module docstring: set SWITCH column to "yes" for each collective you implement.
3. **References**
   - In `comm_overhead/REFERENCES.md`, add a line for SWITCH (e.g. "Idealized switch / full bisection bandwidth model" and any lecture or paper reference).
4. **CLI**
   - In `run_cli.py`:
     - In `_topology()`, add `"switch": TopologyKind.SWITCH` (and ensure `TopologyKind.SWITCH` is imported or available).
     - In the `--topology` argument, add `"switch"` to the choices list.
     - In `list_cmds()`, add at least one collective and one analysis command using `--topology switch`.
     - In `run_all()`, add switch to the list of topologies (e.g. include switch in the same loops as ring/tree/hierarchical, or add a separate loop for switch).
5. **Script (optional)**
   - In `scripts/run_all_commands.sh`, add 1–2 commands for switch (collective and analysis).

### Files to modify
| File | What to change |
|------|----------------|
| `comm_overhead/topology.py` | Add `TopologyKind.SWITCH`, handle SWITCH in `build_topology()`. |
| `comm_overhead/collective.py` | Add branch for `TopologyKind.SWITCH` in `collective_latency_and_volume()`; add `_switch_*` helpers; update docstring table. |
| `comm_overhead/REFERENCES.md` | Add SWITCH formula/source. |
| `run_cli.py` | `_topology()` mapping; `--topology` choices; `list_cmds()`; `run_all()`. |
| `scripts/run_all_commands.sh` | (Optional) Add switch commands. |

### Cmdline examples (to add to list_cmds and docs)
```bash
# Single AllReduce on SWITCH, N=8, M=140GB
PYTHONPATH=. python run_cli.py collective --topology switch --N 8 --M 140e9

# Full analysis: 8 GPUs, DP=8, 70B params, SWITCH topology
PYTHONPATH=. python run_cli.py analysis --topology switch --num-gpus 8 --dp 8 --params 70e9

# With TP=2 (TP AllReduce will use SWITCH once topology is implemented)
PYTHONPATH=. python run_cli.py analysis --topology switch --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

### Checklist
- [ ] `TopologyKind.SWITCH` added and `build_topology()` returns correct desc for SWITCH.
- [ ] All five collectives (AllReduce, AllGather, ReduceScatter, Broadcast, All2All) have SWITCH formulas in collective.py.
- [ ] **Each formula is written clearly in comments with source (e.g. PPT slide or paper).** REFERENCES.md has SWITCH entries.
- [ ] `run_cli.py collective --topology switch --N 8 --M 140e9` runs without error.
- [ ] `run_cli.py analysis --topology switch --num-gpus 8 --dp 8 --params 70e9` runs and reports DP AllReduce latency.
- [ ] REFERENCES.md updated; list_cmds and run_all include switch.

---

## Task 2: MESH topology

### Goal
Implement the **MESH** topology (2D grid: n_x × n_y = N) and latency/volume formulas for all collectives. Support optional mesh dimensions (e.g. `--mesh-nx`, `--mesh-ny`) if needed. Ensure analysis mode supports `--topology mesh` for DP/TP/CP. Provide cmdline examples.

### Implementation steps
1. **Topology**
   - In `comm_overhead/topology.py`:
     - Add `MESH = "mesh"` to `TopologyKind`.
     - Extend `TopologyDesc` with optional fields for mesh, e.g. `n_x: Optional[int] = None`, `n_y: Optional[int] = None`. If not set, collective layer can derive from N (e.g. sqrt(N) for square mesh) or require the caller to set them.
     - In `build_topology()`, add branch for `TopologyKind.MESH`: validate n_x*n_y == N if both provided; otherwise compute default (e.g. n_x, n_y from N); return TopologyDesc with kind=MESH, N, B, alpha, and n_x, n_y.
2. **Collectives**
   - In `comm_overhead/collective.py`:
     - Add handling for `TopologyKind.MESH` in `collective_latency_and_volume()`. Typical approach: 2D mesh collectives in two phases (e.g. reduce along rows then along columns, or ring along dimension 0 then dimension 1). Implement helpers e.g. `_mesh_allreduce_flat(M, n_x, n_y, B, alpha)` with steps and latency from literature/slides. Do the same for AllGather, ReduceScatter, Broadcast, All2All.
     - **In each helper, document the formula and source in comments**, e.g. `# Mesh AllReduce: latency = ...; steps = ...; Source: L4 slide XX / paper YYY.`
     - Update SUPPORTED COMBINATIONS: MESH column "yes" for each implemented collective.
3. **Config and CLI**
   - If mesh dimensions must be passed from CLI to analyze: in `comm_overhead/analyze.py`, add optional `mesh_nx`, `mesh_ny` to `Config`, and pass them into `build_topology()` when kind is MESH. In `run_cli.py`, add `--mesh-nx`, `--mesh-ny` and pass to Config.
   - In `run_cli.py`: add `"mesh"` to `_topology()` and `--topology` choices; add mesh to `list_cmds()` and `run_all()`.
4. **References**
   - In `comm_overhead/REFERENCES.md`, add MESH formula source (e.g. 2D mesh AllReduce steps/latency).
5. **Script**
   - Optionally add mesh commands to `scripts/run_all_commands.sh`.

### Files to modify
| File | What to change |
|------|----------------|
| `comm_overhead/topology.py` | Add `TopologyKind.MESH`; add `n_x`, `n_y` to TopologyDesc if needed; implement `build_topology()` for MESH. |
| `comm_overhead/collective.py` | Add MESH branch and `_mesh_*` helpers; update docstring. |
| `comm_overhead/REFERENCES.md` | Add MESH source. |
| `comm_overhead/analyze.py` | Optional: add `mesh_nx`, `mesh_ny` to Config and use in build_topology when topology_kind is MESH. |
| `run_cli.py` | `_topology()`; `--topology`; optional `--mesh-nx`, `--mesh-ny`; `list_cmds()`; `run_all()`. |
| `scripts/run_all_commands.sh` | Optional: mesh commands. |

### Cmdline examples
```bash
# AllReduce on MESH, 8 GPUs (e.g. 2x4 or 4x2)
PYTHONPATH=. python run_cli.py collective --topology mesh --N 8 --M 140e9

# If you add --mesh-nx/--mesh-ny:
PYTHONPATH=. python run_cli.py collective --topology mesh --N 8 --mesh-nx 4 --mesh-ny 2 --M 140e9

# Analysis with mesh
PYTHONPATH=. python run_cli.py analysis --topology mesh --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology mesh --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

### Checklist
- [ ] MESH topology built with correct n_x, n_y (explicit or derived).
- [ ] All five collectives implemented for MESH.
- [ ] **Formulas documented in code comments with source (PPT or paper);** REFERENCES.md updated.
- [ ] collective and analysis modes work with `--topology mesh`.
- [ ] REFERENCES.md and CLI updated.

---

## Task 3: TORUS topology

### Goal
Implement the **TORUS** topology (2D mesh with wrap-around links) and latency/volume formulas for all collectives. Similar to MESH but with different step counts or contention; support `--topology torus` and DP/TP/CP via existing analysis. Provide cmdline examples.

### Implementation steps
1. **Topology**
   - In `comm_overhead/topology.py`: Add `TORUS = "torus"` to `TopologyKind`; add optional `n_x`, `n_y` (or reuse same as MESH) in TopologyDesc; in `build_topology()` add branch for TORUS (same parameter pattern as MESH).
2. **Collectives**
   - In `comm_overhead/collective.py`: Add branch for `TopologyKind.TORUS`; implement `_torus_*` helpers (torus often has half the effective diameter for ring-like operations due to wrap-around). **Document each formula and source in comments** (e.g. L4/paper). Update SUPPORTED COMBINATIONS.
3. **References and CLI**
   - Update REFERENCES.md; add `"torus"` to run_cli.py (`_topology()`, choices, list_cmds, run_all). If Config already has mesh_nx/mesh_ny, reuse for torus or add torus_nx/torus_ny as needed.

### Files to modify
| File | What to change |
|------|----------------|
| `comm_overhead/topology.py` | Add `TopologyKind.TORUS`; TopologyDesc (reuse or add torus fields); `build_topology()` for TORUS. |
| `comm_overhead/collective.py` | TORUS branch and `_torus_*` helpers; docstring. |
| `comm_overhead/REFERENCES.md` | Torus formula source. |
| `comm_overhead/analyze.py` | If needed, pass torus dimensions when topology_kind is TORUS (same as mesh). |
| `run_cli.py` | torus in _topology, choices, list_cmds, run_all. |
| `scripts/run_all_commands.sh` | Optional: torus commands. |

### Cmdline examples
```bash
PYTHONPATH=. python run_cli.py collective --topology torus --N 8 --M 140e9
PYTHONPATH=. python run_cli.py analysis --topology torus --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology torus --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

### Checklist
- [ ] TORUS topology and all collectives implemented; **formulas in comments with source;** CLI supports torus; REFERENCES updated.

---

## Task 4: Pipeline Parallelism (PP)

### Goal
Implement **PP (Pipeline Parallelism)** communication overhead in the tool: a P2P latency model for activation transfer between stages, integration into `analyze_config()` when `pp_degree > 1`, usage scenarios documented, and cmdline examples.

### Implementation steps
1. **P2P latency model**
   - Option A: Add a function in `comm_overhead/collective.py`, e.g. `p2p_latency(M_bytes, B, alpha)` returning latency = M/B + alpha (one send of size M over a link with bandwidth B and latency alpha). Option B: Create `comm_overhead/p2p.py` with this function and have analyze call it.
   - **In the P2P function (or next to the call), add a comment with the formula and source**, e.g. `# P2P latency = M/B + alpha (single link). Source: L6 PP slide XX / paper YYY.`
   - If PP stage boundaries can be intra-node vs inter-node: use topology (e.g. B1/alpha1 vs B2/alpha2 and gpus_per_node) to choose B and alpha per boundary. For simplicity, first assume one B/alpha (e.g. inter-node B2/alpha2 for multi-node PP).
2. **Activation size M**
   - PP transfers activations between stages. Approximate: M_pp = 2 * batch_size * seq_length * hidden_size * sizeof(dtype) (forward + backward). Use constants.BYTES_FP16 or similar.
3. **analyze.py**
   - When `config.pp_degree > 1` and activation size can be computed (e.g. batch_size, seq_length, hidden_size set or defaults):
     - Compute M_pp as above.
     - Call the P2P latency function (with topology B/alpha or B2/alpha2).
     - Append a `CollectiveReport` (e.g. name "PP activation P2P (per stage boundary)") with latency_s = p2p_latency, volume_bytes = M_pp, steps = 1 (or number of boundaries). Optionally add a field in AnalysisResult for pp_p2p_latency_s and include in summary.
4. **REFERENCES.md**
   - Add reference to L6 (PP P2P communication).
5. **run_cli.py**
   - Add `--pp` argument (e.g. `parser.add_argument("--pp", type=int, default=1)`) and pass to `Config(pp_degree=args.pp)`. Add PP examples to `list_cmds()` (e.g. analysis with --pp 2 or --pp 4 and batch/seq/hidden).
6. **Usage scenarios (document in README or here)**
   - When to use PP: when the model is too large to fit on one device and you split layers across stages; each stage boundary incurs P2P transfer of activations (forward) and gradients (backward). This tool reports the per-boundary P2P latency so users can compare different topologies or PP degrees.

### Files to modify
| File | What to change |
|------|----------------|
| `comm_overhead/collective.py` or `comm_overhead/p2p.py` | Add P2P latency function (M, B, alpha) -> latency. |
| `comm_overhead/analyze.py` | When pp_degree > 1: compute M_pp, call P2P, add report and optionally summary. |
| `comm_overhead/REFERENCES.md` | L6 PP P2P. |
| `run_cli.py` | Add `--pp`; pass to Config; add PP examples in list_cmds. |
| `README.md` or this file | Short "PP usage scenarios" paragraph. |

### Cmdline examples
```bash
# Analysis with PP=4, 8 GPUs, 70B params, with batch/seq/hidden for activation size
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --pp 4 --params 70e9 --batch-size 2 --seq-length 1024 --hidden-size 4096

# PP + DP (e.g. pp=2, dp=4 on 8 GPUs)
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 4 --pp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

### Checklist
- [ ] P2P latency function implemented and called from analyze; **formula and source (e.g. L6) in comments.**
- [ ] analyze_config with pp_degree > 1 produces a PP P2P report.
- [ ] CLI has --pp and example commands run.
- [ ] Usage scenario described in README or TASKS.md.

---

## Task 5: Integration and presentation

### Goal
Integrate all branches (Switch, Mesh, Torus, PP), run full tests, prepare PPT with **everyone participating**, and deliver the presentation (your responsibility).

### Implementation steps
1. **Integration**
   - Merge task branches; resolve conflicts in topology.py, collective.py, run_cli.py, analyze.py. Ensure run_cli.py supports all topologies (ring, tree, hierarchical, switch, mesh, torus) and --pp.
2. **Testing**
   - Run `PYTHONPATH=. python run_cli.py run-all` and confirm no errors.
   - Run `PYTHONPATH=. python run_cli.py list-cmds` and spot-check several commands (collective for each topology, analysis with dp/tp/pp).
   - Optionally run `bash scripts/run_all_commands.sh`.
3. **PPT**
   - Outline: project goal (tool for communication and memory overhead under flexible topology and GPU count); supported topologies and collectives; how DP/TP/CP/PP are reflected; demo (example_usage + CLI examples); extension points and references. **Everyone contributes to slide content.**
4. **Presentation**
   - One person (you) presents; others can support (e.g. live demo, Q&A).

### Files to modify
- No mandatory code file list; possibly final README proofread, or a short integration note in docs.

### Checklist
- [ ] All topologies and PP work in one codebase; run-all and list-cmds complete.
- [ ] PPT completed with team input.
- [ ] Presentation delivered.

---

# Part B: 中文 — 任务说明（与 Part A 对应）

---

## 任务 1：SWITCH 拓扑

### 目标
实现 **SWITCH** 拓扑（理想化星型/全二分带宽）及五种集体操作（AllReduce、AllGather、ReduceScatter、Broadcast、All2All）的延迟与体积公式；在 analysis 模式下支持 `--topology switch`（DP/TP/CP 自动使用该拓扑）；提供命令行示例。

### 实现步骤
1. **拓扑**：在 `comm_overhead/topology.py` 中为 `TopologyKind` 增加 `SWITCH`，在 `build_topology()` 中为 SWITCH 构造并返回 `TopologyDesc`（可复用 B、alpha）。
2. **集体通信**：在 `comm_overhead/collective.py` 的 `collective_latency_and_volume()` 中增加对 `TopologyKind.SWITCH` 的分支，实现 `_switch_*` 辅助函数（如 AllReduce 两步：先 reduce 到根再 broadcast），并更新模块文档中的支持组合表。**在每条公式处用注释写清公式本身和出处**（如「L4 第 34 页」或论文名+公式编号）。
3. **参考文献**：在 `comm_overhead/REFERENCES.md` 中补充 SWITCH 公式来源。
4. **CLI**：在 `run_cli.py` 的 `_topology()` 和 `--topology` 中增加 `switch`；在 `list_cmds()` 和 `run_all()` 中增加 switch 的 collective 与 analysis 示例；可选在 `scripts/run_all_commands.sh` 中增加命令。

### 需修改文件
- `comm_overhead/topology.py`：新增 `TopologyKind.SWITCH`，在 `build_topology()` 中处理 SWITCH。
- `comm_overhead/collective.py`：SWITCH 分支及 `_switch_*` 辅助函数（**注释中写清公式与出处**），更新文档表。
- `comm_overhead/REFERENCES.md`：SWITCH 公式来源。
- `run_cli.py`：`_topology()`、`--topology` 选项、`list_cmds()`、`run_all()`。
- `scripts/run_all_commands.sh`（可选）。

### 命令行示例
```bash
PYTHONPATH=. python run_cli.py collective --topology switch --N 8 --M 140e9
PYTHONPATH=. python run_cli.py analysis --topology switch --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology switch --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

---

## 任务 2：MESH 拓扑

### 目标
实现 **MESH**（2D 网格，n_x × n_y = N）及五种集体操作的公式；支持 `--topology mesh`，必要时支持 `--mesh-nx`/`--mesh-ny`；DP/TP/CP 通过现有 analysis 自动支持；提供命令行示例。

### 实现步骤
1. **拓扑**：在 `topology.py` 中增加 `TopologyKind.MESH`，在 `TopologyDesc` 中增加可选 `n_x`、`n_y`，在 `build_topology()` 中实现 MESH 的构造与校验。
2. **集体通信**：在 `collective.py` 中为 MESH 实现各 collective 的延迟/体积（如二维分阶段 Ring 或递归减半），更新支持组合表。**在辅助函数或计算处用注释写清公式和出处**（如 PPT 页码或论文）。
3. **Config/CLI**：若需从命令行传入网格维度，在 `analyze.py` 的 Config 中增加 `mesh_nx`/`mesh_ny` 并在构建拓扑时传入；在 `run_cli.py` 中增加 `--mesh-nx`/`--mesh-ny`、mesh 的 list_cmds 与 run_all。
4. **参考文献**：在 REFERENCES.md 中补充 MESH 公式来源。

### 需修改文件
- `comm_overhead/topology.py`：MESH 枚举、TopologyDesc 扩展、build_topology 中 MESH 分支。
- `comm_overhead/collective.py`：MESH 分支及 `_mesh_*` 辅助函数（**注释中写清公式与出处**）。
- `comm_overhead/REFERENCES.md`：MESH 来源。
- `comm_overhead/analyze.py`（可选）：Config 中 mesh_nx/mesh_ny。
- `run_cli.py`：mesh 映射、可选 mesh-nx/mesh-ny、list_cmds、run_all。
- `scripts/run_all_commands.sh`（可选）。

### 命令行示例
```bash
PYTHONPATH=. python run_cli.py collective --topology mesh --N 8 --M 140e9
PYTHONPATH=. python run_cli.py analysis --topology mesh --num-gpus 8 --dp 8 --params 70e9
PYTHONPATH=. python run_cli.py analysis --topology mesh --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

---

## 任务 3：TORUS 拓扑

### 目标
实现 **TORUS**（带环绕的 2D 网格）及五种集体操作的公式；支持 `--topology torus`；DP/TP/CP 自动支持；提供命令行示例。

### 实现步骤
与 MESH 类似：在 `topology.py` 中增加 TORUS 及可选维度；在 `collective.py` 中实现 TORUS 专用公式（步数/延迟与 MESH 不同），**在注释中写清每条公式及出处**；更新 REFERENCES、run_cli 的 list_cmds 与 run_all。

### 需修改文件
- `comm_overhead/topology.py`：TORUS 枚举与 build_topology。
- `comm_overhead/collective.py`：TORUS 分支及 `_torus_*` 辅助函数（**注释中写清公式与出处**）。
- `comm_overhead/REFERENCES.md`：Torus 公式来源。
- `comm_overhead/analyze.py`（若需传递 torus 维度）。
- `run_cli.py`：torus 映射与示例。
- `scripts/run_all_commands.sh`（可选）。

### 命令行示例
```bash
PYTHONPATH=. python run_cli.py collective --topology torus --N 8 --M 140e9
PYTHONPATH=. python run_cli.py analysis --topology torus --num-gpus 8 --dp 8 --params 70e9
```

---

## 任务 4：流水线并行（PP）

### 目标
实现 **PP** 的通信开销：P2P 延迟模型（stage 间激活传输）、在 `analyze_config()` 中当 `pp_degree > 1` 时计算并报告、补充使用场景说明与命令行示例。

### 实现步骤
1. **P2P 模型**：在 `collective.py` 或新建 `p2p.py` 中实现 `p2p_latency(M, B, alpha)`；若区分 intra/inter，可结合 topology 的 B1/alpha1、B2/alpha2。**在函数内或调用处用注释写清公式和出处**（如 L6 PP 对应幻灯片或论文）。
2. **激活体积**：M_pp = 2 * batch * seq * hidden * sizeof(dtype)，在 analyze 中根据 Config 计算。
3. **analyze.py**：当 `pp_degree > 1` 时调用 P2P，将结果加入 collective_reports 与可选 summary。
4. **REFERENCES.md**：引用 L6 PP P2P。
5. **run_cli.py**：增加 `--pp`，在 list_cmds 中增加 PP 示例。
6. **使用场景**：在 README 或本文档中写一段「何时用 PP、与本工具的关系」。

### 需修改文件
- `comm_overhead/collective.py` 或 `comm_overhead/p2p.py`：P2P 延迟函数。
- `comm_overhead/analyze.py`：pp_degree > 1 时计算 M_pp、调用 P2P、写报告。
- `comm_overhead/REFERENCES.md`：L6。
- `run_cli.py`：`--pp` 与示例。
- README 或 TASKS.md：PP 使用场景说明。

### 命令行示例
```bash
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --pp 4 --params 70e9 --batch-size 2 --seq-length 1024 --hidden-size 4096
PYTHONPATH=. python run_cli.py analysis --topology ring --num-gpus 8 --dp 4 --pp 2 --params 70e9 --seq-length 1024 --hidden-size 4096
```

---

## 任务 5：整合与演讲

### 目标
合并各人代码、整体测试、**全员参与制作 PPT**、由你负责演讲。

### 实现步骤
1. 合并各分支，解决 topology/collective/run_cli/analyze 冲突。
2. 运行 `run_cli.py run-all` 与 list-cmds 示例做回归测试；可选运行 `scripts/run_all_commands.sh`。
3. PPT 大纲：项目目标、支持拓扑与集体操作、DP/TP/CP/PP 在工具中的体现、示例输出、扩展与参考文献；**PPT 内容由全员参与**。
4. 由你进行课堂演讲；其他人可协助演示或答疑。

### 需修改文件
- 无强制代码列表；可选对 README 做最终校对或增加整合说明。

---

以上为剩余任务与分工的完整说明；详细步骤与文件列表已按任务列出，便于直接分发给对应负责人。
