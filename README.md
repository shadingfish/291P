# Communication and Memory Overhead Tool

**A tool to calculate the communication and memory overhead of collaborative operations under flexible system configuration.** It supports flexible topology and GPU count (switch, tree, mesh, torus, hierarchical), and collaborative operations including AllReduce, AllGather, ReduceScatter, Broadcast, All2All (and P2P for Pipeline Parallelism). After developing the tool, you can use it to analyze an existing configuration (e.g. DP/TP/PP/CP with given GPU count and topology).

Current implementation models communication for **Ring**, **Tree**, and **Hierarchical** topologies. **Switch/Mesh/Torus** and **PP (P2P)** are planned. Memory is **ZeRO-0 only** (full model + grads + optimizer states per GPU). The tool reports per-collective latency/volume/steps, and can also report a total per-step comm-time **upper bound** when `--num-layers` is set.

---

## 1. Overview

This tool computes:

- **Communication overhead**: Latency (seconds), communication volume (bytes), and number of steps for each collective operation (AllReduce, AllGather, ReduceScatter, Broadcast, All2All) under a given **topology** (how GPUs are connected) and **tensor size M** (bytes).
- **Memory overhead**: Per-GPU memory (bytes) for training with a given model size, under **ZeRO-0** (no optimizer state partitioning).

Inputs include: number of GPUs **N**, topology kind (ring, tree, hierarchical; later switch, mesh, torus), link bandwidth **B** and latency **α**, and for hierarchical topology intra-node vs inter-node **B1/α1** and **B2/α2**. For full **analysis**, you also provide parallelism degrees (DP, TP, PP, CP), model parameter count, and optionally batch size, sequence length, and hidden size for TP/PP/CP communication sizing.

Outputs include: per-collective latency and volume (one report per dimension: DP, TP, CP when enabled), per-GPU memory, and a short summary. When `num_layers` is set and DP/TP/CP coexist, a **total communication time per step (upper bound, no overlap)** is also reported. When no parallelism (dp=tp=cp=1), only memory overhead is reported; communication overhead is zero.

## 2. Supported topologies (current)
This tool currently models communication for **Ring**, **Tree**, and **Hierarchical** topologies and reports **ZeRO-0 only** per-GPU memory. **Switch/Mesh/Torus** and **PP (P2P)** are planned. In `analysis` mode you can enable DP/TP/CP (CP maps to All2All).

## 5. How to run

### 5.1 Command-line interface

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

```bash
bash scripts/run_all_commands.sh
```

The script runs: single AllReduce (topology × N), full analysis (DP=8), memory-only (no P), TP=2, CP=4, DP+TP+CP with num-layers, and one verbose example. Or use `run_cli.py run-all` for a single CSV-style table.


### 5.2 Gradio Web UI (`gradio_app.py`)

Install and start:
```bash
pip install gradio
python gradio_app.py
```

Then open the printed URL (default: `http://127.0.0.1:7860`). The UI has four tabs: **Single Collective**, **Full Analysis**, **Memory Calculator**, and **Run All (Comparison)**.


## Notes
These are analytical estimates (alpha-beta style) for comparison and scaling, not cycle-accurate vendor runtimes. Formula details are documented in code comments.
