"""
Example: run the communication and memory overhead tool on a sample config.

Usage (from project root 291P):
  PYTHONPATH=. python example_usage.py

Or from 291P:  python example_usage.py  (if run from 291P, path is set below).
"""

import sys
from pathlib import Path

# Allow importing comm_overhead when run from project root
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
    AnalysisResult,
)
from comm_overhead.analyze import Config


def run_standalone_collective():
    """Example: compute one AllReduce latency (flat Ring)."""
    # Inputs: Ring, 8 GPUs, default NVLink B/alpha
    topo = build_topology(TopologyKind.RING, N=8)
    # Tensor size: 70B params * 2 bytes (FP16 gradient) = 140e9 bytes
    M = 70e9 * 2
    res = collective_latency_and_volume(CollectiveKind.ALL_REDUCE, M, topo)
    print("AllReduce (Ring, N=8, M=140GB):")
    print(f"  Latency: {res.latency_s:.3f} s ({res.latency_s*1000:.1f} ms)")
    print(f"  Volume per GPU: {res.volume_bytes/1e9:.2f} GB")
    print(f"  Steps: {res.steps}")


def run_memory_only():
    """Example: per-GPU memory for 70B model, no ZeRO."""
    total, breakdown = per_gpu_memory_bytes(70e9, return_breakdown=True)
    print("Per-GPU memory (70B params, no ZeRO):")
    print(f"  Total: {total/1e9:.2f} GB")
    print(f"  Weights: {breakdown.weights_bytes/1e9:.2f} GB")
    print(f"  Gradients: {breakdown.gradients_bytes/1e9:.2f} GB")
    print(f"  Optimizer: {breakdown.optimizer_bytes/1e9:.2f} GB")


def run_full_analysis():
    """Example: full config analysis (DP=8, 70B model, Ring)."""
    config = Config(
        num_gpus=8,
        topology_kind=TopologyKind.RING,
        # Parallelism degrees: where they are used (see analyze.py)
        dp_degree=8,   # USED: DP gradient AllReduce over dp_degree GPUs (N=8, M=params*2).
        pp_degree=1,   # Reserved: not used in basic version (PP P2P not computed yet).
        tp_degree=1,   # USED when >1 and hidden_size/seq_length set: TP per-layer AllReduce.
        cp_degree=1,   # Reserved: not used in basic version (CP All2All not computed yet).
        num_parameters=70e9,
        batch_size=4,
        seq_length=1024,
        hidden_size=4096,
    )
    result = analyze_config(config, include_memory_breakdown=True)
    print("Analysis result:")
    print(f"  {result.summary}")
    if result.gpus_per_node is not None:
        print(f"  GPUs per node: {result.gpus_per_node}, Nodes: {result.num_nodes}")
    print(f"  Memory breakdown: weights {result.memory_breakdown.weights_bytes/1e9:.1f} GB, "
          f"grads {result.memory_breakdown.gradients_bytes/1e9:.1f} GB, "
          f"optimizer {result.memory_breakdown.optimizer_bytes/1e9:.1f} GB")
    for r in result.collective_reports:
        line = f"  {r.name}: latency {r.latency_s*1000:.2f} ms, steps {r.steps}"
        if r.intra_latency_s is not None:
            line += f", intra {r.intra_latency_s*1000:.2f} ms"
        if r.inter_latency_s is not None:
            line += f", inter {r.inter_latency_s*1000:.2f} ms"
        print(line)


def run_full_analysis_hierarchical_with_verbose():
    """Example: hierarchical config (2 nodes x 4 GPUs) with verbose calculation."""
    config = Config(
        num_gpus=8,
        topology_kind=TopologyKind.HIERARCHICAL,
        gpus_per_node=4,
        dp_degree=8,
        num_parameters=70e9,
    )
    print("Analysis (Hierarchical, 8 GPUs, 4 per node) with formula breakdown:")
    result = analyze_config(config, include_memory_breakdown=False, verbose=True)
    print(f"  Summary: {result.summary}")
    print(f"  GPUs per node: {result.gpus_per_node}, Nodes: {result.num_nodes}")
    for r in result.collective_reports:
        print(f"  {r.name}: total {r.latency_s*1000:.2f} ms, intra {r.intra_latency_s*1000:.2f} ms, inter {r.inter_latency_s*1000:.2f} ms")


def run_hierarchical_with_gpus_per_node():
    """Example: hierarchical AllReduce with gpus_per_node (Case 2, L5 slide). 8 GPUs, 2 nodes x 4."""
    M = 140e9  # 70B * 2 bytes
    # HIERARCHICAL with gpus_per_node=None: still uses B1/B2; intra_steps fallback to N-1=7 (not 0).
    # So this is NOT the same as standalone RING (which uses a single B for all 2*(N-1) steps).
    topo_flat = build_topology(TopologyKind.HIERARCHICAL, N=8, gpus_per_node=None)
    res_flat = collective_latency_and_volume(CollectiveKind.ALL_REDUCE, M, topo_flat)
    # With gpus_per_node=4: intra steps = 3 per phase (matches slide Case 2).
    topo_case2 = build_topology(TopologyKind.HIERARCHICAL, N=8, gpus_per_node=4)
    res_case2 = collective_latency_and_volume(CollectiveKind.ALL_REDUCE, M, topo_case2)
    print("Hierarchical AllReduce (N=8, M=140GB):")
    print(f"  HIERARCHICAL, gpus_per_node=None (intra steps=7, still uses B1/B2): latency {res_flat.latency_s*1000:.1f} ms, steps {res_flat.steps}")
    print(f"  HIERARCHICAL, gpus_per_node=4 (intra steps=3): latency {res_case2.latency_s*1000:.1f} ms, steps {res_case2.steps}")
    print("  (For flat single-link Ring, use TopologyKind.RING instead of HIERARCHICAL.)")


if __name__ == "__main__":
    print("=== Standalone collective ===\n")
    run_standalone_collective()
    print("\n=== Memory only ===\n")
    run_memory_only()
    print("\n=== Full analysis ===\n")
    run_full_analysis()
    print("\n=== Hierarchical with gpus_per_node ===\n")
    run_hierarchical_with_gpus_per_node()
    print("\n=== Full analysis (hierarchical) with verbose calculation ===\n")
    run_full_analysis_hierarchical_with_verbose()
