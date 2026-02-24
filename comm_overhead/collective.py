"""
Collective communication module: latency and volume for each collective op.

Inputs:  collective kind, tensor size M (bytes), topology description.
Outputs: latency (seconds), total communication volume (bytes), number of steps.

All formulas are from lecture slides; each formula is cited in a comment.
See REFERENCES.md.

-----------------------------------------------------------------------------
SUPPORTED COMBINATIONS (collective_kind x topology_kind)
-----------------------------------------------------------------------------
  Implemented:
                            RING    TREE    HIERARCHICAL(RING ONLY)   SWITCH   MESH   TORUS
  ALL_REDUCE                 yes    yes*    yes            yes      TBD    TBD
  ALL_GATHER                 yes    no**    yes            yes      TBD    TBD
  REDUCE_SCATTER             yes    no**    yes            yes      TBD    TBD
  BROADCAST                  yes    no**    yes            yes      TBD    TBD
  ALL_TO_ALL                 yes    no**    yes***         yes      TBD    TBD

  * TREE: AllReduce uses tree algorithm. Other collectives use ring-style formula.
  ** TREE + non-AllReduce: ring-style formula (B, alpha), no dedicated tree algo.
  *** HIERARCHICAL All2All: simplified (B2/alpha2 dominant).

  Planned (Switch / Mesh / Torus) – not yet implemented:
  - SWITCH: idealized star or full bisection; often 1 or few steps for many
    collectives (e.g. single-round broadcast, reduce to root then broadcast).
    When implemented: ALL_REDUCE, ALL_GATHER, REDUCE_SCATTER, BROADCAST, ALL_TO_ALL
    can each have a switch-specific formula (e.g. 2 steps for AllReduce).
  - MESH: 2D grid (n_x * n_y = N). Typically dimension-ordered ring or
    recursive halving. When implemented: all collectives along rows/columns
    or multi-phase; document per-collective steps and which links (B) are used.
  - TORUS: mesh with wrap-around; same collective algorithms as mesh with
    different step counts or contention model. When implemented: same as MESH
    with torus-specific step/latency formulas.

-----------------------------------------------------------------------------
HIERARCHICAL: per-node topology
-----------------------------------------------------------------------------
  Current: Intra-node is always RING (steps = gpus_per_node - 1 or N - 1).
  Keep "intra-node = Ring only" for simplicity and alignment
  with L4/L5 slides.
  Can Optionally allow a future parameter intra_node_topology
  (e.g. RING | TREE) so that within each node we use that topology's formula
  for the same collective; then HIERARCHICAL would support e.g. "Ring within
  node, Ring across nodes" (current) or "Tree within node, Ring across nodes".
  Not required for basic version.

-----------------------------------------------------------------------------
CONSTRAINTS (topology <-> collective)
-----------------------------------------------------------------------------
  1. algorithm: Only applies to ALL_REDUCE and only when topology is flat
     (RING or TREE). Ignored for HIERARCHICAL (always hierarchical ring).
  2. HIERARCHICAL: Requires B1, alpha1, B2, alpha2. If gpus_per_node is set (e.g. 4),
     intra-node steps use (gpus_per_node - 1) per phase (Case 2, L5 slide); otherwise
     intra steps = (N - 1) (simplified model without node count).
  3. TREE topology: Only ALL_REDUCE has a distinct tree formula. All other
     collectives with TREE topology are computed with ring-style flat formula
     (same as RING), using B and alpha from TopologyDesc.
  4. ALL_TO_ALL: For HIERARCHICAL we use B2/alpha2 only (inter-node phase
     assumed to dominate). Flat uses B and alpha.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from comm_overhead.topology import TopologyDesc, TopologyKind


# -----------------------------------------------------------------------------
# Input: collective operation type
# -----------------------------------------------------------------------------

class CollectiveKind(str, Enum):
    """
    Collective operation types. Extensible (e.g. add REDUCE, SCAN later).

    - ALL_REDUCE:  Every GPU has a buffer of size M; after the op, every GPU has
                   the same reduced value (e.g. sum). Used for gradient sync in DP.
    - ALL_GATHER:  Each GPU has one chunk of size M/N; after the op, every GPU has
                   all chunks concatenated (full size M). Used in ZeRO to gather weights.
    - REDUCE_SCATTER: Reduce (e.g. sum) across GPUs, then scatter so each GPU has
                      one chunk of the result. Used in ZeRO gradient phase.
    - BROADCAST:     One GPU (root) has buffer M; after the op, every GPU has a copy.
    - ALL_TO_ALL:   Each GPU sends a distinct chunk to every other GPU. Used in CP
                    (e.g. Ulysses) for sequence-dimension exchange.
    """
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    BROADCAST = "broadcast"
    ALL_TO_ALL = "all_to_all"


# -----------------------------------------------------------------------------
# Output: result of one collective
# -----------------------------------------------------------------------------

@dataclass
class CollectiveResult:
    """Output of collective_latency_and_volume."""
    latency_s: float
    """Total wall-clock time for the collective in seconds."""
    volume_bytes: float
    """Bytes sent (or sent+received) per GPU for this collective."""
    steps: int
    """Number of communication steps (e.g. ring has 2*(N-1) for AllReduce)."""
    intra_latency_s: Optional[float] = None
    """(Hierarchical only) Time spent in intra-node phase(s), seconds."""
    inter_latency_s: Optional[float] = None
    """(Hierarchical only) Time spent in inter-node phase(s), seconds."""


def _ring_allreduce_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    Ring AllReduce = ReduceScatter + AllGather.

    Parameters (inputs):
        M: Total tensor size in bytes. Each GPU holds a buffer of this size before
           and after the op (e.g. gradient buffer = num_parameters * 2 for FP16).
        N: Number of GPUs in the ring.
        B: Link bandwidth in bytes per second (e.g. NVLink ~900e9). Each step
           transfers M/N bytes over one link.
        alpha: Per-step latency in seconds (e.g. few microseconds). Added once per step.

    Formula: 2*(N-1) steps; each step time = M/(N*B) + alpha.
    Source: L4 slide 34.
    """
    steps = 2 * (N - 1)
    time_per_step = M / (N * B) + alpha
    latency = steps * time_per_step
    # Each GPU sends 2*(N-1) chunks of size M/N; total volume per GPU = 2*(N-1)*(M/N)
    volume_per_gpu = 2 * (N - 1) * (M / N)
    return latency, volume_per_gpu, steps


def _tree_allreduce_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    Tree AllReduce: reduce up the tree, then broadcast down.

    Parameters (inputs):
        M: Tensor size in bytes. In tree algorithm, each step can move up to M bytes.
        N: Number of GPUs (tree leaves).
        B: Link bandwidth in bytes per second.
        alpha: Per-step latency in seconds.

    Formula: 2*ceil(log2(N)) steps; each step time = M/B + alpha.
    Source: L4 slide 36.
    """
    steps = 2 * int(math.ceil(math.log2(N))) if N > 1 else 0
    if steps == 0:
        return 0.0, 0.0, 0
    time_per_step = M / B + alpha
    latency = steps * time_per_step
    volume_per_gpu = steps * M  # upper bound; tree exact volume is op-dependent
    return latency, volume_per_gpu, steps


def _switch_flat_model(M: float, N: int, B: float, alpha: float, kind: str) -> Tuple[float, float, int]:
    """
    Switch Topology: Ideal Full-Bisection Non-blocking Model.

    Assumes a centralized switch where every GPU has a dedicated link of bandwidth B.
    Communication is modeled based on logical phases rather than hop-by-hop forwarding.

    Parameters (inputs):
        M: Total tensor size in bytes.
        N: Number of GPUs connected to the switch.
        B: Link bandwidth per GPU in bytes per second.
        alpha: Per-step (round-trip/handshake) latency in seconds.
        kind: The collective operation type (string value of CollectiveKind).

    Formulas:
        - AllReduce: 2 logical steps (Reduce-to-Root + Broadcast).
          latency = 2 * (M/B + alpha).
        - All2All: N-1 steps to handle distinct chunks for each peer.
          latency = M/B + (N-1) * alpha.
        - Others (AG/RS/Broadcast): 1 logical step of parallel transfer.
          latency = M/B + alpha.

    Source: Idealized Full-Bisection Model (Task 1 Specification).
    """
    if N <= 1:
        return 0.0, 0.0, 0

    if kind == "all_reduce":
        # Two logical phases: aggregate to switch/root, then distribute.
        # Formula: latency = 2 * (M/B + alpha)
        # Source: NCCL Star-Algorithm Model; aligns with L4 Slide 43 (Low Latency Strength).
        steps = 2
        latency = 2 * (M / B + alpha)
        # Volume: 2 * (N-1)/N * M to align with Ring cargo definitions.
        volume = 2 * ((N - 1) / N) * M
    elif kind == "all_to_all":
        # Each GPU logically serializes connection overhead for N-1 peers.
        # Formula: latency = M/B + (N-1) * alpha
        # Source: DeepSpeed-Ulysses: System Optimizations for Enabling Training of Extremely Long Sequence Transformers" (2023
        steps = N - 1
        latency = (M / B) + (N - 1) * alpha
        volume = 2 * ((N - 1) / N) * M
    else:
        # AllGather, ReduceScatter, and Broadcast take 1 parallel round.
        # Thakur et al. "Optimization of Collective Communication Operations in MPICH." (2005)
        steps = 1
        latency = (M / B) + alpha
        volume = ((N - 1) / N) * M

    return latency, volume, steps

def _ring_allgather_or_reducescatter_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    Ring AllGather or ReduceScatter: (N-1) steps, each step moves M/N bytes.

    Parameters (inputs):
        M: Total tensor size in bytes (full buffer size for AllGather result, or
           total reduced size for ReduceScatter input).
        N: Number of GPUs.
        B: Link bandwidth in bytes per second.
        alpha: Per-step latency in seconds.

    Formula: (N-1) * (M/(N*B) + alpha).
    Source: L4 slide 34.
    """
    steps = N - 1
    time_per_step = M / (N * B) + alpha
    latency = steps * time_per_step
    volume_per_gpu = (N - 1) * (M / N)
    return latency, volume_per_gpu, steps

def _hierarchical_ring_allreduce(
    M: float,
    N: int,
    B1: float,
    alpha1: float,
    B2: float,
    alpha2: float,
    gpus_per_node: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Hierarchical Ring AllReduce (two-level: intra-node then inter-node).

    Parameters (inputs):
        M: Total tensor size in bytes (same meaning as flat Ring).
        N: Total number of GPUs across all nodes.
        B1, alpha1: Intra-node bandwidth and latency.
        B2, alpha2: Inter-node bandwidth and latency.
        gpus_per_node: GPUs per node (e.g. 4 for 2 nodes x 4 GPUs). If None or 0,
                      fall back to simplified model: intra steps = N-1 (no node count).

    Formula (Case 2, L5 slide): When gpus_per_node is set, intra steps use per-node count:
      intra = (gpus_per_node - 1) * (M/(N*B1) + alpha1)   [within each node]
      inter = M/(N*B2) + alpha2   [one pairwise step across nodes]
      latency = 2 * (intra + inter)   [ReduceScatter phase + AllGather phase]
    Source: L4 slide 38, L5 slide 4 (Recap on Ring-based All-Reduce on Hierarchical Topology).
    """
    if gpus_per_node and gpus_per_node > 0:
        intra_steps = gpus_per_node - 1
    else:
        intra_steps = N - 1
    intra = intra_steps * (M / (N * B1) + alpha1)
    inter = M / (N * B2) + alpha2
    latency = 2 * (intra + inter)
    steps = 2 * intra_steps + 2
    volume_per_gpu = 2 * intra_steps * (M / N) + 2 * (M / N)
    return latency, volume_per_gpu, steps, 2 * intra, 2 * inter


def _hierarchical_ring_allgather_reducescatter(
    M: float,
    N: int,
    B1: float,
    alpha1: float,
    B2: float,
    alpha2: float,
    gpus_per_node: Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    Hierarchical AllGather or ReduceScatter: intra-node phase + inter-node phase.

    Parameters (inputs): Same as _hierarchical_ring_allreduce; gpus_per_node optional.
    When set: intra steps = (gpus_per_node - 1). Otherwise intra steps = (N - 1).
    Formula: intra + inter. Source: L5 slide 4.
    """
    if gpus_per_node and gpus_per_node > 0:
        intra_steps = gpus_per_node - 1
    else:
        intra_steps = N - 1
    intra = intra_steps * (M / (N * B1) + alpha1)
    inter = M / (N * B2) + alpha2
    latency = intra + inter
    steps = intra_steps + 1
    volume_per_gpu = intra_steps * (M / N) + (M / N)
    return latency, volume_per_gpu, steps, intra, inter


def _print_ring_allreduce(M: float, N: int, B: float, alpha: float, lat: float, st: int) -> None:
    """Print Ring AllReduce formula and calculation (L4 slide 34)."""
    steps_rs, steps_ag = N - 1, N - 1
    time_per = M / (N * B) + alpha
    print("  [Formula] Ring AllReduce = ReduceScatter + AllGather (L4 slide 34)")
    print(f"  [Input]   N={N}, M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")
    print(f"  [Step]   steps = 2*(N-1) = 2*{N-1} = {st}; time_per_step = M/(N*B)+alpha = {M/(N*B):.4e} + {alpha:.2e} = {time_per:.4e} s")
    print(f"  [Result] latency = steps * time_per_step = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_tree_allreduce(M: float, N: int, B: float, alpha: float, lat: float, st: int) -> None:
    """Print Tree AllReduce formula and calculation (L4 slide 36)."""
    from math import ceil, log2
    steps = 2 * int(ceil(log2(N))) if N > 1 else 0
    time_per = M / B + alpha if steps else 0
    print("  [Formula] Tree AllReduce: 2*ceil(log2(N)) steps, each M/B+alpha (L4 slide 36)")
    print(f"  [Input]   N={N}, M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")
    print(f"  [Step]   steps = 2*ceil(log2(N)) = {st}; time_per_step = M/B+alpha = {M/B:.4e} + {alpha:.2e} = {time_per:.4e} s")
    print(f"  [Result] latency = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_hierarchical_allreduce(
    M: float, N: int, B1: float, alpha1: float, B2: float, alpha2: float,
    gpus_per_node: Optional[int], intra: float, inter: float, lat: float,
) -> None:
    """Print Hierarchical Ring AllReduce formula and calculation (L5 slide, Case 2)."""
    intra_steps = (gpus_per_node - 1) if gpus_per_node and gpus_per_node > 0 else (N - 1)
    time_intra_one = M / (N * B1) + alpha1
    time_inter_one = M / (N * B2) + alpha2
    intra_one_phase = intra_steps * time_intra_one
    print("  [Formula] Hierarchical Ring AllReduce (Case 2, L5 slide): 2*(intra + inter)")
    print(f"  [Input]   N={N}, M={M:.2e} B, B1={B1:.2e} B/s, alpha1={alpha1:.2e} s, B2={B2:.2e} B/s, alpha2={alpha2:.2e} s, gpus_per_node={gpus_per_node}")
    print(f"  [Intra]   intra_steps = {intra_steps}; time_intra_per_step = M/(N*B1)+alpha1 = {time_intra_one:.4e} s; intra_one_phase = {intra_one_phase:.4e} s")
    print(f"  [Inter]   time_inter = M/(N*B2)+alpha2 = {time_inter_one:.4e} s")
    print(f"  [Phase]   ReduceScatter: intra + inter = {intra_one_phase:.4e} + {time_inter_one:.4e} = {intra_one_phase+time_inter_one:.4e} s; AllGather same => total intra = {intra:.4e} s, inter = {inter:.4e} s")
    print(f"  [Result] latency = 2*(intra+inter) = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_hierarchical_ag_rs(
    M: float, N: int, B1: float, alpha1: float, B2: float, alpha2: float,
    gpus_per_node: Optional[int], intra: float, inter: float, lat: float,
) -> None:
    """Print Hierarchical AllGather/ReduceScatter formula (L5 slide 4)."""
    intra_steps = (gpus_per_node - 1) if gpus_per_node and gpus_per_node > 0 else (N - 1)
    print("  [Formula] Hierarchical AllGather/ReduceScatter: intra + inter (L5 slide 4)")
    print(f"  [Input]   N={N}, gpus_per_node={gpus_per_node} => intra_steps={intra_steps}")
    print(f"  [Result] intra_latency = {intra:.4e} s, inter_latency = {inter:.4e} s, total = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_switch_info(kind: str, M: float, N: int, B: float, alpha: float, lat: float, st: int) -> None:
    """Print Switch Ideal model formula and calculation."""
    print(f"  [Formula] Switch {kind.capitalize()} (Ideal Full-Bisection Model)")
    print(f"  [Input]   N={N}, M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")

    if kind == "all_reduce":
        print(f"  [Step]    steps = 2 (Reduce + Broadcast); latency = 2 * (M/B + alpha)")
    elif kind == "all_to_all":
        print(f"  [Step]    steps = N-1 ({N - 1}); latency = M/B + (N-1)*alpha")
    else:
        print(f"  [Step]    steps = 1; latency = M/B + alpha")

    print(f"  [Result]  latency = {lat:.4e} s ({lat * 1000:.2f} ms)")

def collective_latency_and_volume(
    kind: CollectiveKind,
    M: float,
    topology: TopologyDesc,
    algorithm: Optional[str] = None,
    verbose: bool = False,
) -> CollectiveResult:
    """
    Compute latency (s), communication volume (bytes), and step count for one collective.

    ----------
    Input parameters (detailed)
    ----------

    kind : CollectiveKind
        Which collective to compute.
        - ALL_REDUCE:  Gradient sync in DP; each GPU has buffer M, all get the same sum.
        - ALL_GATHER:  Gather sharded data (e.g. ZeRO weights); input M/N per GPU, output M.
        - REDUCE_SCATTER:  Reduce then scatter (e.g. ZeRO gradient phase).
        - BROADCAST:  One root sends buffer M to all others.
        - ALL_TO_ALL:  Each GPU sends/receives chunks; used in CP (Ulysses). M = total
                       tensor size involved (e.g. 4*S*h for attention buffers).

    M : float
        Tensor size in bytes. Interpretation by op:
        - AllReduce / Broadcast:  size of the buffer on each GPU (e.g. gradient =
          num_parameters * 2 for FP16). Same M before and after.
        - AllGather / ReduceScatter:  total size of the full tensor (so each chunk is M/N).
        - All2All:  total size of the tensor being exchanged; each GPU sends (N-1)*(M/N) bytes.
        Example: 70B params, FP16 gradient -> M = 70e9 * 2 = 140e9 bytes.

    topology : TopologyDesc
        From build_topology(). Supplies:
        - N:  number of GPUs participating in this collective.
        - For RING/TREE:  B (bytes/s) and alpha (seconds) for each link.
        - For HIERARCHICAL:  B1/alpha1 (intra-node, e.g. NVLink), B2/alpha2 (inter-node,
          e.g. InfiniBand). Used to compute per-step time = data/B + alpha.

    algorithm : Optional[str]
        Only used for ALL_REDUCE on flat topology. "ring" (default) or "tree".
        Ring: more steps, less data per step (good for large M). Tree: fewer steps,
        more data per step (good for small M or latency-sensitive).

    ----------
    Outputs (CollectiveResult)
    ----------

    latency_s : float
        Total wall-clock time in seconds for the collective.
    volume_bytes : float
        Bytes sent (and optionally received) per GPU for this collective.
    steps : int
        Number of communication steps (e.g. ring AllReduce has 2*(N-1)).

    ----------
    Variable reference
    ----------

    N  = number of GPUs (from topology.N).
    B  = link bandwidth in bytes per second (flat topology).
    alpha = per-step latency in seconds (flat topology).
    B1, alpha1 = intra-node bandwidth and latency (hierarchical).
    B2, alpha2 = inter-node bandwidth and latency (hierarchical).

    References:
        L4 slide 34 (Ring AllReduce, AllGather, ReduceScatter).
        L4 slide 36 (Tree AllReduce).
        L4 slide 38, L5 slide 4 (hierarchical).
        L7 (All2All volume; we use a simple model here).
    """
    N = topology.N
    if N <= 0:
        return CollectiveResult(latency_s=0.0, volume_bytes=0.0, steps=0)
    if topology.kind == TopologyKind.HIERARCHICAL:
        B1, alpha1 = topology.B1, topology.alpha1
        B2, alpha2 = topology.B2, topology.alpha2
        gpus_per_node = topology.gpus_per_node
    else:
        B, alpha = topology.B, topology.alpha
        gpus_per_node = None

    if kind == CollectiveKind.ALL_REDUCE:
        if topology.kind == TopologyKind.HIERARCHICAL:
            lat, vol, st, intra, inter = _hierarchical_ring_allreduce(
                M, N, B1, alpha1, B2, alpha2, gpus_per_node
            )
            if verbose:
                _print_hierarchical_allreduce(M, N, B1, alpha1, B2, alpha2, gpus_per_node, intra, inter, lat)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st, intra_latency_s=intra, inter_latency_s=inter)
        elif topology.kind == TopologyKind.TREE or algorithm == "tree":
            lat, vol, st = _tree_allreduce_flat(M, N, B, alpha)
            if verbose:
                _print_tree_allreduce(M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        elif topology.kind == TopologyKind.SWITCH:
            lat, vol, st = _switch_flat_model(M, N, B, alpha, kind.value)
            if verbose:
                _print_switch_info(kind.value, M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        else:
            lat, vol, st = _ring_allreduce_flat(M, N, B, alpha)
            if verbose:
                _print_ring_allreduce(M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)

    if kind == CollectiveKind.ALL_GATHER or kind == CollectiveKind.REDUCE_SCATTER:
        if topology.kind == TopologyKind.HIERARCHICAL:
            lat, vol, st, intra, inter = _hierarchical_ring_allgather_reducescatter(
                M, N, B1, alpha1, B2, alpha2, gpus_per_node
            )
            if verbose:
                _print_hierarchical_ag_rs(M, N, B1, alpha1, B2, alpha2, gpus_per_node, intra, inter, lat)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st, intra_latency_s=intra, inter_latency_s=inter)
        elif topology.kind == TopologyKind.SWITCH:
            lat, vol, st = _switch_flat_model(M, N, B, alpha, kind.value)
            if verbose:
                _print_switch_info(kind.value, M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        else:
            lat, vol, st = _ring_allgather_or_reducescatter_flat(M, N, B, alpha)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)

    if kind == CollectiveKind.BROADCAST:
        if topology.kind == TopologyKind.HIERARCHICAL:
            lat, vol, st, intra, inter = _hierarchical_ring_allgather_reducescatter(
                M, N, B1, alpha1, B2, alpha2, gpus_per_node
            )
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st, intra_latency_s=intra, inter_latency_s=inter)
        elif topology.kind == TopologyKind.SWITCH:
            lat, vol, st = _switch_flat_model(M, N, B, alpha, kind.value)
            if verbose:
                _print_switch_info(kind.value, M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        else:
            lat, vol, st = _ring_allgather_or_reducescatter_flat(M, N, B, alpha)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)

    if kind == CollectiveKind.ALL_TO_ALL:
        if topology.kind == TopologyKind.SWITCH:
            lat, vol, st = _switch_flat_model(M, N, B, alpha, kind.value)
            if verbose:
                _print_switch_info(kind.value, M, N, B, alpha, lat, st)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        # All2All: each GPU sends (N-1) chunks of size M/N, receives (N-1) chunks of size M/N.
        # Input M = total tensor size (e.g. 4*S*h for Ulysses QKV). Volume per GPU = 2*(N-1)*M/N.
        # Latency: simplified as volume_per_gpu / B + alpha*(N-1). Source: L7.
        volume_per_gpu = 2 * (N - 1) * (M / N)
        if topology.kind == TopologyKind.HIERARCHICAL:
            # Rough: assume one inter-node exchange dominates
            latency = volume_per_gpu / B2 + alpha2 * (N - 1)
        else:
            latency = volume_per_gpu / B + alpha * (N - 1)
        steps = N - 1
        return CollectiveResult(latency_s=latency, volume_bytes=volume_per_gpu, steps=steps)

    return CollectiveResult(latency_s=0.0, volume_bytes=0.0, steps=0)