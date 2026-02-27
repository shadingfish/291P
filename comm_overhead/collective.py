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
  ALL_REDUCE                 yes    yes*    yes            yes      yes    yes
  ALL_GATHER                 yes    no**    yes            yes      yes    yes
  REDUCE_SCATTER             yes    no**    yes            yes      yes    yes
  BROADCAST                  yes    no**    yes            yes      yes    yes
  ALL_TO_ALL                 yes    no**    yes***         yes      yes    yes

  * TREE: AllReduce uses tree algorithm. Other collectives use ring-style formula.
  ** TREE + non-AllReduce: ring-style formula (B, alpha), no dedicated tree algo.
  *** HIERARCHICAL All2All: simplified (B2/alpha2 dominant).
  MESH: 2D grid (n_x * n_y = N); dimension-ordered ring (row then column). See
    _mesh_* helpers; source Rabenseifner ICCS 2004, Chan et al. HiPC 2008 (All2All).

  Planned:
  - TORUS: mesh with wrap-around; same collective algorithms as mesh with
    different step counts or contention model.

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


def _require_mesh_dims(topology: TopologyDesc) -> tuple[int, int]:
    """Return (n_x, n_y) for MESH; fallback to near-square if n_x/n_y missing."""
    if topology.n_x is not None and topology.n_y is not None and topology.n_x > 0 and topology.n_y > 0 and topology.n_x * topology.n_y == topology.N:
        return int(topology.n_x), int(topology.n_y)
    from comm_overhead.topology import _default_mesh_dims
    return _default_mesh_dims(topology.N)


def _mesh_allreduce(M: float, n_x: int, n_y: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D Mesh AllReduce via dimension-ordered Ring: RS(row) + AR(col on chunk) + AG(row).

    Latency:
      t = 2*(n_y-1) * (M/(n_y*B) + alpha) + 2*(n_x-1) * (M/(N*B) + alpha)

    Volume per GPU:
      v = 2*(n_y-1) * (M/n_y) + 2*(n_x-1) * (M/N)

    Source:
      R. Rabenseifner, "Optimization of Collective Reduction Operations",
      ICCS 2004.
      (AllReduce = ReduceScatter + AllGather; 2D mesh obtained via dimension-wise composition.)
    """
    N = n_x * n_y
    if N <= 1:
        return 0.0, 0.0, 0

    t_row = 2 * (n_y - 1) * (M / (n_y * B) + alpha)
    t_col = 2 * (n_x - 1) * (M / (N * B) + alpha)
    latency = t_row + t_col

    volume = 2 * (n_y - 1) * (M / n_y) + 2 * (n_x - 1) * (M / N)
    steps = 2 * (n_y - 1) + 2 * (n_x - 1)
    return latency, volume, steps


def _mesh_allgather(M: float, n_x: int, n_y: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D Mesh AllGather via dimension order: AG(col on M/n_y) + AG(row on M).

    Latency:
      t = (n_x-1) * (M/(N*B) + alpha) + (n_y-1) * (M/(n_y*B) + alpha)

    Volume per GPU:
      v = (n_x-1) * (M/N) + (n_y-1) * (M/n_y)

    Source:
      R. Rabenseifner, "Optimization of Collective Reduction Operations",
      ICCS 2004.
      (Ring AllGather used as building block; extended via dimension-wise composition.)
    """
    N = n_x * n_y
    if N <= 1:
        return 0.0, 0.0, 0
    
    t_col = (n_x - 1) * (M / (N * B) + alpha)
    t_row = (n_y - 1) * (M / (n_y * B) + alpha)
    latency = t_col + t_row

    volume = (n_x - 1) * (M / N) + (n_y - 1) * (M / n_y)
    steps = (n_x - 1) + (n_y - 1)
    return latency, volume, steps


def _mesh_reducescatter(M: float, n_x: int, n_y: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D Mesh ReduceScatter via dimension order: RS(row on M) + RS(col on M/n_y).

    Latency:
      t = (n_y-1) * (M/(n_y*B) + alpha) + (n_x-1) * (M/(N*B) + alpha)

    Volume per GPU:
      v = (n_y-1) * (M/n_y) + (n_x-1) * (M/N)

    Source:
      R. Rabenseifner, "Optimization of Collective Reduction Operations",
      ICCS 2004.
      (Ring ReduceScatter building block composed across mesh dimensions.)
    """
    N = n_x * n_y
    if N <= 1:
        return 0.0, 0.0, 0
    
    t_row = (n_y - 1) * (M / (n_y * B) + alpha)
    t_col = (n_x - 1) * (M / (N * B) + alpha)
    latency = t_row + t_col

    volume = (n_y - 1) * (M / n_y) + (n_x - 1) * (M / N)
    steps = (n_y - 1) + (n_x - 1)
    return latency, volume, steps


def _mesh_broadcast(M: float, n_x: int, n_y: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D Mesh Broadcast in two phases: along one dimension, then the other.

    Latency:
      t = (n_y-1) * (M/(n_y*B) + alpha) + (n_x-1) * (M/(n_x*B) + alpha)

    Volume per GPU:
      Use an average/upper-bound proxy matching the two ring phases:
      v = (n_y-1) * (M/n_y) + (n_x-1) * (M/n_x)

    Source:
      R. Rabenseifner, "Optimization of Collective Reduction Operations",
      ICCS 2004.
      (Broadcast modeled using the same α-β cost framework and dimension-wise decomposition.)
    """
    N = n_x * n_y
    if N <= 1:
        return 0.0, 0.0, 0
    
    t_row = (n_y - 1) * (M / (n_y * B) + alpha)
    t_col = (n_x - 1) * (M / (n_x * B) + alpha)
    latency = t_row + t_col

    volume = (n_y - 1) * (M / n_y) + (n_x - 1) * (M / n_x)
    steps = (n_y - 1) + (n_x - 1)
    return latency, volume, steps


def _mesh_alltoall(M: float, n_x: int, n_y: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D Mesh All2All via two local All2All phases: rows then columns.

    Latency:
      t = [2*(n_y-1)*M/n_y]/B + alpha*(n_y-1)
        + [2*(n_x-1)*M/n_x]/B + alpha*(n_x-1)

    Volume per GPU:
      v = 2*(n_y-1)*M/n_y + 2*(n_x-1)*M/n_x

    Source:
      A. Chan, P. Balaji, W. Gropp, R. Thakur, "Communication Analysis of Parallel 3D FFT for Flat Cartesian Meshes",
      HiPC 2008.
      (Row/column All-to-All, a.k.a. pencil transpose.)
    """
    N = n_x * n_y
    if N <= 1:
        return 0.0, 0.0, 0
    
    v_row = 2 * (n_y - 1) * (M / n_y)
    v_col = 2 * (n_x - 1) * (M / n_x)
    latency = (v_row / B + alpha * (n_y - 1)) + (v_col / B + alpha * (n_x - 1))
    
    volume = v_row + v_col
    steps = (n_y - 1) + (n_x - 1)
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

def _torus_dims(topology: TopologyDesc) -> Tuple[int, int]:
    """
    Get TORUS grid dims (n_x, n_y).

    TopologyDesc was constructed by build_topology(), which sets n_x/n_y.
    """
    if topology.n_x is None or topology.n_y is None:
        raise ValueError("TORUS requires n_x and n_y to be set (expected from build_topology()).")
    return topology.n_x, topology.n_y

def _torus_allreduce_flat(M: float, N: int, nx: int, ny: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    TORUS AllReduce (2D) modeled as dimension-ordered bucket AllReduce:
    ReduceScatter + AllGather across X (rows), then across Y (columns).

    Model:
      - Use 1D Ring step-time template (α–β): time_per_step = payload/(k*B) + alpha
        where k = the group size along that dimension.
        Source: L4 slide 34.
      - Dimension-ordered view:
        Source: [Collective algorithms for multiported torus networks; Sack & Gropp (2015)].

    Convention:
      N = nx * ny. M is the full tensor size (bytes).

    Bucket phase sizing:
      - X phase (row group size nx): total tensor within a row = M/ny.
      - Y phase (column group size ny): after X, each rank holds a row-block (size M/ny);
        the column AllReduce then operates on total tensor size M.

    Formula (AllReduce = 2 phases, each uses Ring AllReduce steps):
      latency =
        2*(nx-1) * ( (M/ny)/(nx*B) + alpha )
      + 2*(ny-1) * ( M/(ny*B) + alpha )

      steps = 2*(nx-1) + 2*(ny-1)

    Volume:
      volume_per_gpu = 2*(N-1)*(M/N)
    """
    if N <= 1:
        return 0.0, 0.0, 0
    lat_x = 2 * (nx - 1) * ((M / ny) / (nx * B) + alpha)
    lat_y = 2 * (ny - 1) * (M / (ny * B) + alpha)
    latency = lat_x + lat_y
    steps = 2 * (nx - 1) + 2 * (ny - 1)
    volume_per_gpu = 2 * (N - 1) * (M / N)
    return latency, volume_per_gpu, steps

def _torus_ringstyle_flat(M: float, N: int, nx: int, ny: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    TORUS AllGather / ReduceScatter / Broadcast modeled as 2D dimension-ordered
    ring-style collective: X phase then Y phase.

    We reuse ring-style (N-1) step model:
      Ring-style latency(k, payload) = (k-1) * ( payload/(k*B) + alpha )
    Source basis: L4 slide 34 (Ring AllGather / ReduceScatter).

    Payload approximation:
      X phase payload ~= M/ny, group size nx
      Y phase payload ~= M/nx, group size ny

    Total:
      latency = (nx-1)*((M/ny)/(nx*B)+alpha) + (ny-1)*((M/nx)/(ny*B)+alpha)
      steps   = (nx-1) + (ny-1)

    Volume:
      volume_per_gpu = (N-1)*(M/N)
    """
    if N <= 1:
        return 0.0, 0.0, 0

    lat_x = (nx - 1) * ((M / ny) / (nx * B) + alpha)
    lat_y = (ny - 1) * ((M / nx) / (ny * B) + alpha)
    latency = lat_x + lat_y

    steps = (nx - 1) + (ny - 1)

    volume_per_gpu = (N - 1) * (M / N)
    return latency, volume_per_gpu, steps


def _torus_all2all_flat(M: float, N: int, nx: int, ny: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """
    TORUS All2All modeled as a 2-phase (X then Y) approximation.

    Convention:
      N = nx * ny. M is the full tensor size (bytes).

    Timing approximation (dimension-ordered, bucket-like phases):
      T \approx (nx-1) * ( (M/ny)/(nx*B) + alpha ) + (ny-1) * ( M/(ny*B) + alpha )

    References:
      - All2All volume/context: L7.
      - Per-step α–β timing template reused from 1D ring-style: L4 slide 34.
      - Dimension-ordered multi-dim communication is a common closed-form approximation.

    Volume per GPU:
      volume_per_gpu = 2*(N-1)*(M/N)
    """
    if N <= 1:
        return 0.0, 0.0, 0

    lat_x = (nx - 1) * ((M / ny) / (nx * B) + alpha)
    lat_y = (ny - 1) * (M / (ny * B) + alpha)

    latency = lat_x + lat_y
    steps = (nx - 1) + (ny - 1)

    volume_per_gpu = 2 * (N - 1) * (M / N)
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


def _print_torus_allreduce(
    M: float, N: int, nx: int, ny: int, B: float, alpha: float, lat: float, st: int
) -> None:
    """Print TORUS AllReduce formula and calculation (dimension-ordered; L4 slide 34; Sack & Gropp 2015)."""
    payload_x = M / ny
    t_step_x = payload_x / (nx * B) + alpha
    steps_x = 2 * (nx - 1)
    lat_x = steps_x * t_step_x

    payload_y = M
    t_step_y = payload_y / (ny * B) + alpha
    steps_y = 2 * (ny - 1)
    lat_y = steps_y * t_step_y

    print("  [Formula] TORUS AllReduce (bucket, X then Y) (L4 slide 34; Sack & Gropp 2015)")
    print(f"  [Input]   N={N} (nx={nx}, ny={ny}), M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")
    print(f"  [X]       steps_x = 2*(nx-1) = {steps_x}; total_x = M/ny = {payload_x:.2e} B")
    print(f"            time_x_per_step = total_x/(nx*B)+alpha = {payload_x/(nx*B):.4e} + {alpha:.2e} = {t_step_x:.4e} s")
    print(f"            lat_x = {lat_x:.4e} s ({lat_x*1000:.2f} ms)")
    print(f"  [Y]       steps_y = 2*(ny-1) = {steps_y}; total_y = M = {payload_y:.2e} B")
    print(f"            time_y_per_step = total_y/(ny*B)+alpha = {payload_y/(ny*B):.4e} + {alpha:.2e} = {t_step_y:.4e} s")
    print(f"            lat_y = {lat_y:.4e} s ({lat_y*1000:.2f} ms)")
    print(f"  [Total]   steps = {steps_x}+{steps_y} = {st}; latency = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_torus_ringstyle(
    kind: str, M: float, N: int, nx: int, ny: int, B: float, alpha: float, lat: float, st: int
) -> None:
    """Print TORUS ring-style (AG/RS/Broadcast) formula and calculation (bucket; L4 slide 34; Sack & Gropp 2015)."""
    payload_x = M / ny
    t_step_x = payload_x / (nx * B) + alpha
    steps_x = nx - 1
    lat_x = steps_x * t_step_x

    payload_y = M
    t_step_y = payload_y / (ny * B) + alpha
    steps_y = ny - 1
    lat_y = steps_y * t_step_y

    print(f"  [Formula] TORUS {kind} (bucket, X then Y) (L4 slide 34; Sack & Gropp 2015)")
    print(f"  [Input]   N={N} (nx={nx}, ny={ny}), M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")
    print(f"  [X]       steps_x = nx-1 = {steps_x}; total_x = M/ny = {payload_x:.2e} B")
    print(f"            time_x_per_step = total_x/(nx*B)+alpha = {payload_x/(nx*B):.4e} + {alpha:.2e} = {t_step_x:.4e} s")
    print(f"            lat_x = {lat_x:.4e} s ({lat_x*1000:.2f} ms)")
    print(f"  [Y]       steps_y = ny-1 = {steps_y}; total_y = M = {payload_y:.2e} B")
    print(f"            time_y_per_step = total_y/(ny*B)+alpha = {payload_y/(ny*B):.4e} + {alpha:.2e} = {t_step_y:.4e} s")
    print(f"            lat_y = {lat_y:.4e} s ({lat_y*1000:.2f} ms)")
    print(f"  [Total]   steps = {steps_x}+{steps_y} = {st}; latency = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_torus_all2all(
    M: float, N: int, nx: int, ny: int, B: float, alpha: float, lat: float, st: int
) -> None:
    """Print TORUS All2All two-phase approximation (L7 volume; timing template L4 slide 34)."""
    payload_x = M / ny
    t_step_x = payload_x / (nx * B) + alpha
    steps_x = nx - 1
    lat_x = steps_x * t_step_x

    payload_y = M
    t_step_y = payload_y / (ny * B) + alpha
    steps_y = ny - 1
    lat_y = steps_y * t_step_y
    volume_per_gpu = 2 * (N - 1) * (M / N)

    print("  [Formula] TORUS All2All ≈ X-phase + Y-phase (L7 volume; timing template L4 slide 34)")
    print(f"  [Input]   N={N} (nx={nx}, ny={ny}), M={M:.2e} B, B={B:.2e} B/s, alpha={alpha:.2e} s")
    print(f"  [X]       steps_x = nx-1 = {steps_x}; total_x = M/ny = {payload_x:.2e} B")
    print(f"            time_x_per_step = total_x/(nx*B)+alpha = {payload_x/(nx*B):.4e} + {alpha:.2e} = {t_step_x:.4e} s")
    print(f"            lat_x = {lat_x:.4e} s ({lat_x*1000:.2f} ms)")
    print(f"  [Y]       steps_y = ny-1 = {steps_y}; total_y = M = {payload_y:.2e} B")
    print(f"            time_y_per_step = total_y/(ny*B)+alpha = {payload_y/(ny*B):.4e} + {alpha:.2e} = {t_step_y:.4e} s")
    print(f"            lat_y = {lat_y:.4e} s ({lat_y*1000:.2f} ms)")
    print(f"  [Total]   steps = {steps_x}+{steps_y} = {st}; latency = {lat:.4e} s ({lat*1000:.2f} ms)")
    print(f"  [Volume]  volume_per_gpu = 2*(N-1)*(M/N) = {volume_per_gpu:.2e} B ({volume_per_gpu/1e9:.2f} GB)")

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
        elif topology.kind == TopologyKind.TORUS:
            nx, ny = _torus_dims(topology)
            lat, vol, st = _torus_allreduce_flat(M, N, nx, ny, B, alpha)
            if verbose:
                _print_torus_allreduce(M, N, nx, ny, B, alpha, lat, st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            lat, vol, st = _mesh_allreduce(M, n_x, n_y, B, alpha)
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
        elif topology.kind == TopologyKind.TORUS:
            nx, ny = _torus_dims(topology)
            lat, vol, st = _torus_ringstyle_flat(M, N, nx, ny, B, alpha)
            if verbose:
                _print_torus_ringstyle(kind.value, M, N, nx, ny, B, alpha, lat, st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            if kind == CollectiveKind.ALL_GATHER:
                lat, vol, st = _mesh_allgather(M, n_x, n_y, B, alpha)
            else:
                lat, vol, st = _mesh_reducescatter(M, n_x, n_y, B, alpha)
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
        elif topology.kind == TopologyKind.TORUS:
            nx, ny = _torus_dims(topology)
            lat, vol, st = _torus_ringstyle_flat(M, N, nx, ny, B, alpha)
            if verbose:
                _print_torus_ringstyle(kind.value, M, N, nx, ny, B, alpha, lat, st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            lat, vol, st = _mesh_broadcast(M, n_x, n_y, B, alpha)
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
        
        if topology.kind == TopologyKind.TORUS:
            nx, ny = _torus_dims(topology)
            lat, vol, st = _torus_all2all_flat(M, N, nx, ny, B, alpha)
            if verbose:
                _print_torus_all2all(M, N, nx, ny, B, alpha, lat, st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            lat, vol, st = _mesh_alltoall(M, n_x, n_y, B, alpha)
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