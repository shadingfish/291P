"""
Command-line interface for communication and memory overhead tool.

Usage (from project root 291P):
  PYTHONPATH=. python run_cli.py <mode> [options]

Modes:
  collective   Single AllReduce: vary topology, N, gpus_per_node. Output: latency, steps, intra/inter.
  analysis     Full config (DP/TP, memory). Output: summary, per-collective, optional verbose.
  list-cmds    Print all suggested commands for horizontal comparison (copy-paste to run each).
  run-all      Run all predefined combinations and print a comparison table.
"""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from comm_overhead import (
    TopologyKind,
    build_topology,
    CollectiveKind,
    collective_latency_and_volume,
    per_gpu_memory_bytes,
    analyze_config,
)
from comm_overhead.analyze import Config


def _topology(s: str) -> TopologyKind:
    m = {
        "ring": TopologyKind.RING,
        "tree": TopologyKind.TREE,
        "hierarchical": TopologyKind.HIERARCHICAL,
        "switch": TopologyKind.SWITCH,
        "mesh": TopologyKind.MESH,
    }
    return m[s.lower()]


def cmd_collective(args) -> None:
    """Run one AllReduce with given topology and N."""
    topo = build_topology(
        kind=_topology(args.topology),
        N=args.N,
        gpus_per_node=args.gpus_per_node,
        n_x=args.mesh_nx,
        n_y=args.mesh_ny,
    )
    M = float(args.M)
    res = collective_latency_and_volume(
        CollectiveKind.ALL_REDUCE, M, topo,
        algorithm="tree" if args.tree_algo else None,
        verbose=args.verbose,
    )
    tag = f"topology={args.topology}, N={args.N}"
    if args.gpus_per_node:
        tag += f", gpus_per_node={args.gpus_per_node}"
    print(f"[Collective] AllReduce M={M:.2e} B, {tag}")
    print(f"  latency_ms={res.latency_s*1000:.2f}, steps={res.steps}, volume_GB={res.volume_bytes/1e9:.2f}")
    if res.intra_latency_s is not None:
        print(f"  intra_ms={res.intra_latency_s*1000:.2f}, inter_ms={res.inter_latency_s*1000:.2f}")


def cmd_analysis(args) -> None:
    """Run full analyze_config with Config from CLI."""
    config = Config(
        num_gpus=args.num_gpus,
        topology_kind=_topology(args.topology),
        gpus_per_node=args.gpus_per_node,
        dp_degree=args.dp,
        tp_degree=args.tp,
        cp_degree=args.cp,
        num_parameters=float(args.params),
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        mesh_nx=args.mesh_nx,
        mesh_ny=args.mesh_ny,
    )
    result = analyze_config(
        config,
        include_memory_breakdown=args.memory_breakdown,
        verbose=args.verbose,
    )
    print(f"[Analysis] {result.summary}")
    if result.gpus_per_node is not None:
        print(f"  gpus_per_node={result.gpus_per_node}, num_nodes={result.num_nodes}")
    if result.total_comm_time_per_step_upper_bound_s is not None:
        print(f"  total_comm_per_step_ms={result.total_comm_time_per_step_upper_bound_s*1000:.2f} (upper bound, no overlap)")
    print(f"  per_gpu_memory_GB={result.per_gpu_memory_bytes/1e9:.2f}")
    for r in result.collective_reports:
        line = f"  {r.name}: latency_ms={r.latency_s*1000:.2f}, steps={r.steps}, volume_GB={r.volume_bytes/1e9:.2f}"
        if r.intra_latency_s is not None:
            line += f", intra_ms={r.intra_latency_s*1000:.2f}, inter_ms={r.inter_latency_s*1000:.2f}"
        print(line)


def list_cmds() -> None:
    """Print all suggested commands for horizontal comparison."""
    base = "PYTHONPATH=. python run_cli.py"
    M = "140e9"
    print("# Usage: list-cmds prints commands; run-all runs them as a table.")
    print(f"#   {base} list-cmds   # copy-paste any line below to run")
    print(f"#   {base} run-all     # run all and print CSV table\n")
    print("# ---- 1. Single AllReduce (M=140GB), vary topology and N ----\n")
    print(f"{base} collective --topology ring --N 8 --M {M}")
    print(f"{base} collective --topology tree --N 8 --M {M}")
    print(f"{base} collective --topology tree --N 8 --M {M} --tree-algo")
    print(f"{base} collective --topology hierarchical --N 8 --M {M}")
    print(f"{base} collective --topology hierarchical --N 8 --gpus-per-node 4 --M {M}")
    print(f"{base} collective --topology ring --N 16 --M {M}")
    print(f"{base} collective --topology tree --N 16 --M {M}")
    print(f"{base} collective --topology hierarchical --N 16 --M {M}")
    print(f"{base} collective --topology hierarchical --N 16 --gpus-per-node 8 --M {M}")
    print(f"{base} collective --topology hierarchical --N 16 --gpus-per-node 4 --M {M}")
    print(f"{base} collective --topology switch --N 8 --M {M}")
    print(f"{base} analysis --topology switch --num-gpus 8 --dp 8 --params 70e9")
    print(f"{base} collective --topology mesh --N 8 --M {M}")
    print(f"{base} collective --topology mesh --N 8 --mesh-nx 4 --mesh-ny 2 --M {M}")
    print(f"{base} analysis --topology mesh --num-gpus 8 --dp 8 --params 70e9")
    print()
    print("# ---- 2. Full analysis (DP=8, 70B). Compare DP AllReduce latency and memory ----\n")
    print(f"{base} analysis --topology ring --num-gpus 8 --dp 8 --params 70e9")
    print(f"{base} analysis --topology tree --num-gpus 8 --dp 8 --params 70e9")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 8 --params 70e9")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9")
    print()
    print("# ---- 3. Memory only (no DP/TP/CP): per-GPU memory for 70B ----\n")
    print(f"{base} analysis --topology ring --num-gpus 1 --dp 1 --tp 1 --cp 1 --params 70e9")
    print()
    print("# ---- 4. With TP=2 (per-layer AllReduce). Needs seq-length, hidden-size ----\n")
    print(f"{base} analysis --topology ring --num-gpus 8 --dp 4 --tp 2 --params 70e9 --seq-length 1024 --hidden-size 4096")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 4 --tp 2 --gpus-per-node 4 --params 70e9 --seq-length 1024 --hidden-size 4096")
    print()
    print("# ---- 5. With CP=4 (Context Parallelism, All2All). Needs seq-length, hidden-size ----\n")
    print(f"{base} analysis --topology ring --num-gpus 8 --dp 2 --cp 4 --params 70e9 --seq-length 1024 --hidden-size 4096")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 2 --cp 4 --gpus-per-node 4 --params 70e9 --seq-length 1024 --hidden-size 4096")
    print()
    print("# ---- 6. DP+TP+CP with num-layers (combined total comm/step) ----\n")
    print(f"{base} analysis --topology ring --num-gpus 8 --dp 2 --tp 2 --cp 2 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 2 --tp 2 --cp 2 --gpus-per-node 4 --params 70e9 --seq-length 1024 --hidden-size 4096 --num-layers 32")
    print()
    print("# ---- 7. Verbose (show formula and calculation) ----\n")
    print(f"{base} collective --topology ring --N 8 --M {M} --verbose")
    print(f"{base} collective --topology hierarchical --N 8 --gpus-per-node 4 --M {M} --verbose")
    print(f"{base} analysis --topology hierarchical --num-gpus 8 --dp 8 --gpus-per-node 4 --params 70e9 --verbose")


def run_all() -> None:
    """Run all predefined combinations and print a comparison table."""
    M = 140e9
    rows = []

    # Collective: topology x N x gpus_per_node
    for topo_name, topo_kind in [
        ("ring", TopologyKind.RING),
        ("tree", TopologyKind.TREE),
        ("hierarchical", TopologyKind.HIERARCHICAL),
        ("switch", TopologyKind.SWITCH),
        ("mesh", TopologyKind.MESH),
    ]:
        for N in [8, 16]:
            if topo_kind != TopologyKind.HIERARCHICAL:
                gpus_per_node_list = [None]
            else:
                gpus_per_node_list = [None, 4] if N == 8 else [None, 4, 8]
            for gpn in gpus_per_node_list:
                topo = build_topology(kind=topo_kind, N=N, gpus_per_node=gpn)
                res = collective_latency_and_volume(CollectiveKind.ALL_REDUCE, M, topo)
                tag = f"collective,{topo_name},N={N}"
                if gpn is not None:
                    tag += f",gpn={gpn}"
                rows.append((tag, res.latency_s * 1000, res.steps, res.intra_latency_s * 1000 if res.intra_latency_s else None, res.inter_latency_s * 1000 if res.inter_latency_s else None))

    # Analysis: ring vs tree vs hierarchical (8 GPUs, DP=8)
    for topo_name, topo_kind in [
        ("ring", TopologyKind.RING),
        ("tree", TopologyKind.TREE),
        ("hierarchical", TopologyKind.HIERARCHICAL),
        ("switch", TopologyKind.SWITCH),
        ("mesh", TopologyKind.MESH),
    ]:
        gpn = 4 if topo_kind == TopologyKind.HIERARCHICAL else None
        config = Config(num_gpus=8, topology_kind=topo_kind, dp_degree=8, num_parameters=70e9, gpus_per_node=gpn)
        result = analyze_config(config, include_memory_breakdown=False)
        dp_lat = result.dp_allreduce_latency_s * 1000 if result.dp_allreduce_latency_s else None
        tag = f"analysis,{topo_name},num_gpus=8,dp=8"
        if gpn:
            tag += f",gpn={gpn}"
        rows.append((tag, dp_lat, "", None, None))

    # Print table
    print("tag,latency_ms,steps,intra_ms,inter_ms")
    for (tag, lat_ms, steps, intra_ms, inter_ms) in rows:
        intra_s = f"{intra_ms:.2f}" if intra_ms is not None else ""
        inter_s = f"{inter_ms:.2f}" if inter_ms is not None else ""
        steps_s = str(steps) if steps is not None else ""
        lat_s = f"{lat_ms:.2f}" if lat_ms is not None else ""
        print(f"{tag},{lat_s},{steps_s},{intra_s},{inter_s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Communication and memory overhead CLI.")
    parser.add_argument("mode", choices=["collective", "analysis", "list-cmds", "run-all"], help="Mode: collective | analysis | list-cmds | run-all")
    parser.add_argument("--topology", default="ring", choices=["ring", "tree", "hierarchical", "switch", "mesh"], help="Topology (default: ring)")
    parser.add_argument("--N", type=int, default=8, help="Number of GPUs for collective (default: 8)")
    parser.add_argument("--M", default="140e9", help="Tensor size in bytes for collective (default: 140e9)")
    parser.add_argument("--gpus-per-node", type=int, default=None, help="GPUs per node (hierarchical only)")
    parser.add_argument("--mesh-nx", type=int, default=None, help="Mesh dim n_x (mesh only; require nx*ny==N)")
    parser.add_argument("--mesh-ny", type=int, default=None, help="Mesh dim n_y (mesh only; require nx*ny==N)")
    parser.add_argument("--tree-algo", action="store_true", help="Use tree algorithm for AllReduce (flat topology)")
    parser.add_argument("--verbose", action="store_true", help="Print formula and calculation")
    # analysis
    parser.add_argument("--num-gpus", type=int, default=8, help="Total GPUs for analysis (default: 8)")
    parser.add_argument("--dp", type=int, default=1, help="DP degree (default: 1)")
    parser.add_argument("--tp", type=int, default=1, help="TP degree (default: 1)")
    parser.add_argument("--cp", type=int, default=1, help="CP degree (default: 1)")
    parser.add_argument("--params", default="70e9", help="Model parameter count (default: 70e9)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers (for combined comm/step when DP+TP+CP)")
    parser.add_argument("--no-memory-breakdown", action="store_true", help="Skip memory breakdown in analysis")
    args = parser.parse_args()
    args.memory_breakdown = not getattr(args, "no_memory_breakdown", False)

    if args.mode == "list-cmds":
        list_cmds()
        return
    if args.mode == "run-all":
        run_all()
        return
    if args.mode == "collective":
        cmd_collective(args)
        return
    if args.mode == "analysis":
        cmd_analysis(args)
        return


if __name__ == "__main__":
    main()
