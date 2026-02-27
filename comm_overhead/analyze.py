"""
Analyze module: orchestrate topology, collective, and memory for a given config.

Inputs:  GPU count, topology params, parallelism degrees (DP/PP/TP/CP), model size,
         optional batch/seq for TP/PP/CP communication sizing.
Outputs: AnalysisResult with per-collective latencies, per-GPU memory, and summary.

This module does not add new formulas; it calls topology, collective, and memory
with parameters derived from the config. References are in those modules.

Used by:
  - example_usage.py: analyze_config(), Config, AnalysisResult for full config analysis.
  - comm_overhead/__init__.py: re-exports analyze_config, AnalysisResult, Config.
  - Any script that does: from comm_overhead import analyze_config, Config; analyze_config(Config(...))
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from comm_overhead.topology import TopologyKind, build_topology
from comm_overhead.collective import (
    CollectiveKind,
    collective_latency_and_volume,
    CollectiveResult,
)
from comm_overhead.memory import per_gpu_memory_bytes, MemoryBreakdown
from comm_overhead.constants import BYTES_FP16


@dataclass
class Config:
    """
    Input configuration for analyze_config().

    Attributes:
        num_gpus: Total number of GPUs.
        topology_kind: RING, TREE, or HIERARCHICAL.
        B: (Flat) Bandwidth bytes/s. Optional.
        alpha: (Flat) Latency seconds. Optional.
        B1, alpha1, B2, alpha2, gpus_per_node: (Hierarchical) Optional.
        dp_degree: Data parallelism degree. Default 1.
            USED: When > 1, we compute one DP gradient AllReduce over dp_degree GPUs
            (tensor size M = num_parameters * 2 bytes).
        pp_degree: Pipeline parallelism degree. Default 1.
            NOT USED in basic version (PP P2P communication not computed).
        tp_degree: Tensor parallelism degree. Default 1.
            USED: When > 1 and hidden_size/seq_length are set, we compute one TP
            per-layer AllReduce over tp_degree GPUs (tensor size from batch*seq*hidden).
        cp_degree: Context parallelism degree. Default 1.
            USED: When > 1 and seq_length/hidden_size are set, we compute one CP
            All2All over cp_degree GPUs (tensor size M from batch*seq*hidden, QKV-style).
        num_parameters: Model parameter count (e.g. 70e9).
        batch_size: Global batch size (optional, for TP/PP activation sizing).
        seq_length: Sequence length (optional, for TP/CP sizing).
        hidden_size: Hidden dimension (optional, for TP per-layer tensor size).
        num_layers: Number of transformer layers (optional). When set, used to compute
            a combined "total communication time per step" (DP once + num_layers * (TP + CP)).
    """
    num_gpus: int
    topology_kind: TopologyKind = TopologyKind.RING
    B: Optional[float] = None
    alpha: Optional[float] = None
    B1: Optional[float] = None
    alpha1: Optional[float] = None
    B2: Optional[float] = None
    alpha2: Optional[float] = None
    gpus_per_node: Optional[int] = None
    dp_degree: int = 1
    pp_degree: int = 1
    tp_degree: int = 1
    cp_degree: int = 1
    num_parameters: float = 0.0
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    # Generic 2D grid dims (for mesh / torus): nx * ny == degree.
    nx: Optional[int] = None
    ny: Optional[int] = None

@dataclass
class CollectiveReport:
    """One collective's result for reporting."""
    name: str
    latency_s: float
    volume_bytes: float
    steps: int
    tensor_size_M_bytes: float
    intra_latency_s: Optional[float] = None
    """(Hierarchical only) Intra-node phase time, seconds."""
    inter_latency_s: Optional[float] = None
    """(Hierarchical only) Inter-node phase time, seconds."""


@dataclass
class AnalysisResult:
    """Output of analyze_config(): all communication and memory results."""
    per_gpu_memory_bytes: float
    memory_breakdown: Optional[MemoryBreakdown] = None
    collective_reports: List[CollectiveReport] = field(default_factory=list)
    dp_allreduce_latency_s: Optional[float] = None
    summary: str = ""
    gpus_per_node: Optional[int] = None
    """When topology is HIERARCHICAL and set: GPUs per node."""
    num_nodes: Optional[int] = None
    """When gpus_per_node is set: num_gpus // gpus_per_node."""
    total_comm_time_per_step_upper_bound_s: Optional[float] = None
    """When DP+TP+CP coexist and num_layers is set: DP once + num_layers*(TP + CP). No overlap."""


def analyze_config(
    config: Config,
    include_memory_breakdown: bool = True,
    verbose: bool = False,
) -> AnalysisResult:
    """
    Analyze communication and memory for the given config (basic version, no ZeRO).

    Inputs:
        config: Config with num_gpus, topology, parallelism degrees, num_parameters,
                and optional batch_size/seq_length/hidden_size for TP/PP/CP.
        include_memory_breakdown: If True, include weight/grad/optimizer breakdown.

    Outputs:
        AnalysisResult with:
          - per_gpu_memory_bytes: From memory module (ZeRO-0).
          - memory_breakdown: Optional weights/grads/optimizer bytes.
          - collective_reports: List of CollectiveReport for each collective computed.
          - dp_allreduce_latency_s: Latency of one DP gradient AllReduce (if DP > 1).
          - summary: Short human-readable summary.

    Logic:
        - Build topology from config (N = num_gpus, B/alpha or B1/alpha1/B2/alpha2).
        - DP: One AllReduce per step for gradients; tensor size M = num_parameters * 2 (FP16).
        - TP: Per-layer AllReduce when tp_degree > 1.
        - CP: Per-layer All2All when cp_degree > 1 (sequence-dimension exchange, L7).
        - PP: P2P when pp_degree > 1 (not yet implemented).
    """
    reports: List[CollectiveReport] = []

    # Per-GPU memory (no ZeRO). Source: L5 slide 18.
    mem_result = per_gpu_memory_bytes(
        num_parameters=config.num_parameters,
        return_breakdown=include_memory_breakdown,
    )
    if include_memory_breakdown:
        total_mem, breakdown = mem_result
    else:
        total_mem = mem_result
        breakdown = None

    # DP, TP, CP can coexist (e.g. dp=4, tp=2, cp=2 on 16 GPUs). We call build_topology and
    # collective_latency_and_volume once per dimension because each has different N (degree),
    # collective kind (AllReduce vs All2All), and tensor size M.
    # -------------------------------------------------------------------------
    # DP gradient AllReduce: over dp_degree GPUs; M = parameters * 2 bytes (FP16 gradient).
    dp_allreduce_latency_s: Optional[float] = None
    if config.dp_degree > 1:
        M_dp = config.num_parameters * BYTES_FP16
        dp_topology = build_topology(
            kind=config.topology_kind,
            N=config.dp_degree,
            B=config.B,
            alpha=config.alpha,
            B1=config.B1,
            alpha1=config.alpha1,
            B2=config.B2,
            alpha2=config.alpha2,
            gpus_per_node=config.gpus_per_node,
            n_x=config.nx,
            n_y=config.ny,
        )
        res = collective_latency_and_volume(
            CollectiveKind.ALL_REDUCE,
            M_dp,
            dp_topology,
            verbose=verbose,
        )
        dp_allreduce_latency_s = res.latency_s
        reports.append(CollectiveReport(
            name="DP gradient AllReduce",
            latency_s=res.latency_s,
            volume_bytes=res.volume_bytes,
            steps=res.steps,
            tensor_size_M_bytes=M_dp,
            intra_latency_s=res.intra_latency_s,
            inter_latency_s=res.inter_latency_s,
        ))

    # TP per-layer AllReduce (when tp_degree > 1).
    tp_latency_s: Optional[float] = None
    if config.tp_degree > 1 and config.hidden_size is not None and config.seq_length is not None:
        # Per transformer layer: ~2 AllReduce on [batch, seq, hidden] per layer; simplify to one.
        # M_tp ~ 2 * batch * seq * hidden * 2 bytes (one allreduce). Use micro-batch 1.
        batch = config.batch_size or 1
        M_tp = 2 * batch * config.seq_length * config.hidden_size * BYTES_FP16
        tp_topology = build_topology(
            kind=config.topology_kind,
            N=config.tp_degree,
            B=config.B,
            alpha=config.alpha,
            B1=config.B1,
            alpha1=config.alpha1,
            B2=config.B2,
            alpha2=config.alpha2,
            gpus_per_node=config.gpus_per_node,
            n_x=config.nx,
            n_y=config.ny,
        )
        res = collective_latency_and_volume(
            CollectiveKind.ALL_REDUCE,
            M_tp,
            tp_topology,
            verbose=verbose,
        )
        tp_latency_s = res.latency_s
        reports.append(CollectiveReport(
            name="TP per-layer AllReduce (one layer)",
            latency_s=res.latency_s,
            volume_bytes=res.volume_bytes,
            steps=res.steps,
            tensor_size_M_bytes=M_tp,
            intra_latency_s=res.intra_latency_s,
            inter_latency_s=res.inter_latency_s,
        ))

    # CP sequence All2All: over cp_degree GPUs; M = QKV-style tensor (4 * batch * seq * hidden).
    # Used for context parallelism (e.g. Ulysses) along sequence dimension. Source: L7.
    cp_latency_s: Optional[float] = None
    if config.cp_degree > 1 and config.seq_length is not None and config.hidden_size is not None:
        batch = config.batch_size or 1
        # Total tensor size for All2All: Q,K,V (and often one more), 4 * batch * seq * hidden * 2 bytes.
        M_cp = 4 * batch * config.seq_length * config.hidden_size * BYTES_FP16
        cp_topology = build_topology(
            kind=config.topology_kind,
            N=config.cp_degree,
            B=config.B,
            alpha=config.alpha,
            B1=config.B1,
            alpha1=config.alpha1,
            B2=config.B2,
            alpha2=config.alpha2,
            gpus_per_node=config.gpus_per_node,
            n_x=config.nx,
            n_y=config.ny,
        )
        res = collective_latency_and_volume(
            CollectiveKind.ALL_TO_ALL,
            M_cp,
            cp_topology,
            verbose=verbose,
        )
        cp_latency_s = res.latency_s
        reports.append(CollectiveReport(
            name="CP sequence All2All (one layer)",
            latency_s=res.latency_s,
            volume_bytes=res.volume_bytes,
            steps=res.steps,
            tensor_size_M_bytes=M_cp,
            intra_latency_s=res.intra_latency_s,
            inter_latency_s=res.inter_latency_s,
        ))

    # Topology layout (for hierarchical)
    gpus_per_node: Optional[int] = config.gpus_per_node if config.topology_kind == TopologyKind.HIERARCHICAL else None
    num_nodes: Optional[int] = None
    if gpus_per_node and gpus_per_node > 0 and config.num_gpus % gpus_per_node == 0:
        num_nodes = config.num_gpus // gpus_per_node

    # Combined "total communication time per step" when DP+TP+CP coexist (no overlap).
    # DP once per step; TP and CP each num_layers times per step.
    total_comm_per_step_s: Optional[float] = None
    if config.num_layers is not None and config.num_layers > 0:
        dp_s = dp_allreduce_latency_s if dp_allreduce_latency_s is not None else 0.0
        tp_s = tp_latency_s if tp_latency_s is not None else 0.0
        cp_s = cp_latency_s if cp_latency_s is not None else 0.0
        total_comm_per_step_s = dp_s + config.num_layers * (tp_s + cp_s)

    # Build summary string
    summary_parts = [
        f"GPUs: {config.num_gpus}, Topology: {config.topology_kind.value}",
        f"Per-GPU memory (no ZeRO): {total_mem / (1024**3):.2f} GB",
    ]
    if gpus_per_node is not None and gpus_per_node > 0:
        summary_parts.append(f"GPUs per node: {gpus_per_node}")
    if num_nodes is not None:
        summary_parts.append(f"Nodes: {num_nodes}")
    if dp_allreduce_latency_s is not None:
        summary_parts.append(f"DP AllReduce latency: {dp_allreduce_latency_s*1000:.2f} ms")
    if total_comm_per_step_s is not None:
        summary_parts.append(f"Total comm/step (upper, no overlap): {total_comm_per_step_s*1000:.2f} ms")
    summary = "; ".join(summary_parts)

    return AnalysisResult(
        per_gpu_memory_bytes=total_mem,
        memory_breakdown=breakdown,
        collective_reports=reports,
        dp_allreduce_latency_s=dp_allreduce_latency_s,
        summary=summary,
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
        total_comm_time_per_step_upper_bound_s=total_comm_per_step_s,
    )
