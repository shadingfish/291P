"""
Collective communication model.

We compute the modeled latency (s), logical volume per GPU (bytes), and step count
for a single collective op under a given topology.

"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from comm_overhead.topology import TopologyDesc, TopologyKind


# Input: collective operation type

class CollectiveKind(str, Enum):
    """Collective op kinds supported by this tool."""
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
    """Total wall-clock time for the collective (seconds)."""
    volume_bytes: float
    """Logical bytes per GPU for this collective."""
    steps: int
    """Number of communication steps (algorithmic, not network hops)."""
    intra_latency_s: Optional[float] = None
    """(Hierarchical only) Intra-node phase time (seconds)."""
    inter_latency_s: Optional[float] = None
    """(Hierarchical only) Inter-node phase time (seconds)."""


def _ring_allreduce_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """Ring AllReduce (RS+AG): steps=2*(N-1), latency=steps*(M/(N*B)+alpha);"""
    steps = 2 * (N - 1)
    time_per_step = M / (N * B) + alpha
    latency = steps * time_per_step
    # Each GPU sends 2*(N-1) chunks of size M/N; total volume per GPU = 2*(N-1)*(M/N)
    volume_per_gpu = 2 * (N - 1) * (M / N)
    return latency, volume_per_gpu, steps


def _tree_allreduce_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """Tree AllReduce: steps=2*ceil(log2 N), latency=steps*(M/B+alpha); """
    steps = 2 * int(math.ceil(math.log2(N))) if N > 1 else 0
    if steps == 0:
        return 0.0, 0.0, 0
    time_per_step = M / B + alpha
    latency = steps * time_per_step
    # Approximate per-GPU volume: up to one M in reduce phase + one M in broadcast.
    volume_per_gpu = 2 * M
    return latency, volume_per_gpu, steps


def _tree_broadcast_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """Tree Broadcast: steps=ceil(log2 N), latency=steps*(M/B+alpha);"""
    steps = int(math.ceil(math.log2(N))) if N > 1 else 0
    if steps == 0:
        return 0.0, 0.0, 0
    time_per_step = M / B + alpha
    latency = steps * time_per_step
    volume_per_gpu = M  # each receiving GPU gets M; root sends M
    return latency, volume_per_gpu, steps


def _switch_flat_model(M: float, N: int, B: float, alpha: float, kind: str) -> Tuple[float, float, int]:
    """Ideal switch model: phase-based latency for AllReduce/AG/RS/Bcast/All2All"""
    if N <= 1:
        return 0.0, 0.0, 0

    if kind == "all_reduce":
        # Switch AllReduce: latency = 2*(M/B + alpha). L4 slide 43.
        steps = 2
        latency = 2 * (M / B + alpha)
        volume = 2 * ((N - 1) / N) * M
    elif kind == "all_to_all":
        # Switch All2All: latency = M/B + (N-1)*alpha. DeepSpeed-Ulysses (L7).
        steps = N - 1
        latency = (M / B) + (N - 1) * alpha
        volume = 2 * ((N - 1) / N) * M
    else:
        # Switch AG/RS/Bcast: one parallel round. Thakur et al. (2005).
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
    """2D mesh AllReduce via dimension-ordered ring """
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
    """2D mesh AllGather via dimension order """
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
    """2D mesh ReduceScatter via dimension order"""
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
    """2D mesh Broadcast as two-phase dimension transfer"""
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
    """2D mesh All2All via row/column phases """
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
    """Ring AG/RS: steps=N-1, latency=steps*(M/(N*B)+alpha);"""
    steps = N - 1
    time_per_step = M / (N * B) + alpha
    latency = steps * time_per_step
    volume_per_gpu = (N - 1) * (M / N)
    return latency, volume_per_gpu, steps


def _ring_broadcast_flat(M: float, N: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """Ring-style Broadcast (simplified): steps=N-1, latency=steps*(M/B+alpha)."""
    if N <= 1:
        return 0.0, 0.0, 0
    steps = N - 1
    time_per_step = M / B + alpha
    latency = steps * time_per_step
    volume_per_gpu = steps * M
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
    """2D torus AllReduce via dimension-ordered phases"""
    if N <= 1:
        return 0.0, 0.0, 0
    lat_x = 2 * (nx - 1) * ((M / ny) / (nx * B) + alpha)
    lat_y = 2 * (ny - 1) * (M / (ny * B) + alpha)
    latency = lat_x + lat_y
    steps = 2 * (nx - 1) + 2 * (ny - 1)
    volume_per_gpu = 2 * (N - 1) * (M / N)
    return latency, volume_per_gpu, steps

def _torus_ringstyle_flat(M: float, N: int, nx: int, ny: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D torus ring-style AG/RS/Bcast as X then Y phases"""
    if N <= 1:
        return 0.0, 0.0, 0

    lat_x = (nx - 1) * ((M / ny) / (nx * B) + alpha)
    lat_y = (ny - 1) * ((M / nx) / (ny * B) + alpha)
    latency = lat_x + lat_y

    steps = (nx - 1) + (ny - 1)

    volume_per_gpu = (N - 1) * (M / N)
    return latency, volume_per_gpu, steps


def _torus_all2all_flat(M: float, N: int, nx: int, ny: int, B: float, alpha: float) -> Tuple[float, float, int]:
    """2D torus All2All as X then Y approximation"""
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
    """Hierarchical Ring AllReduce: intra + inter phases"""
    if gpus_per_node and gpus_per_node > 0:
        intra_steps = gpus_per_node - 1
        num_nodes = N // gpus_per_node
        inter_steps_per_phase = max(0, num_nodes - 1)
    else:
        intra_steps = N - 1
        inter_steps_per_phase = 1  # Fallback when gpus_per_node unset

    intra = intra_steps * (M / (N * B1) + alpha1)
    time_inter_per_step = M / (N * B2) + alpha2
    inter = inter_steps_per_phase * time_inter_per_step

    latency = 2 * (intra + inter)
    steps = 2 * intra_steps + 2 * inter_steps_per_phase
    volume_per_gpu = 2 * intra_steps * (M / N) + 2 * inter_steps_per_phase * (M / N)
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
    """Hierarchical AG/RS: intra + inter phases (same node logic as hierarchical AllReduce)."""
    if gpus_per_node and gpus_per_node > 0:
        intra_steps = gpus_per_node - 1
        num_nodes = N // gpus_per_node
        inter_steps = max(0, num_nodes - 1)
    else:
        intra_steps = N - 1
        inter_steps = 1

    intra = intra_steps * (M / (N * B1) + alpha1)
    time_inter_per_step = M / (N * B2) + alpha2
    inter = inter_steps * time_inter_per_step
    latency = intra + inter
    steps = intra_steps + inter_steps
    volume_per_gpu = intra_steps * (M / N) + inter_steps * (M / N)
    return latency, volume_per_gpu, steps, intra, inter


def _hierarchical_broadcast(
    M: float,
    N: int,
    B1: float,
    alpha1: float,
    B2: float,
    alpha2: float,
    gpus_per_node: Optional[int] = None,
) -> Tuple[float, float, int, float, float]:
    """Hierarchical Broadcast: intra (g-1) + inter (k-1) phases; volume per GPU = M."""
    if gpus_per_node and gpus_per_node > 0:
        g = gpus_per_node
        intra_steps = g - 1
        num_nodes = N // g
        inter_steps = max(0, num_nodes - 1)
    else:
        # Fallback: treat as flat with interconnect B2/alpha2 dominant.
        intra_steps = 0
        inter_steps = N - 1

    intra = intra_steps * (M / B1 + alpha1) if intra_steps > 0 else 0.0
    inter = inter_steps * (M / B2 + alpha2) if inter_steps > 0 else 0.0
    latency = intra + inter
    steps = intra_steps + inter_steps
    volume_per_gpu = M
    return latency, volume_per_gpu, steps, intra, inter

def _hierarchical_all2all(
    M: float,
    N: int,
    B1: float,
    alpha1: float,
    B2: float,
    alpha2: float,
    gpus_per_node: Optional[int] = None,
) -> Tuple[float, float, int, float, float]:
    """Hierarchical All2All: intra + inter phases; volume per GPU = (N-1)*M/N."""
    volume_per_gpu = (N - 1) * (M / N)
    if gpus_per_node and gpus_per_node > 0:
        g = gpus_per_node
        num_nodes = N // g
        inter_steps = max(0, num_nodes - 1)
        intra = (g - 1) * (M / (N * B1) + alpha1)
        inter = inter_steps * (M / (N * B2) + alpha2)
        latency = intra + inter
        steps = (g - 1) + inter_steps
        return latency, volume_per_gpu, steps, intra, inter
    else:
        # Fallback: B2/alpha2 dominant (no node count)
        latency = volume_per_gpu / B2 + alpha2 * (N - 1)
        steps = N - 1
        return latency, volume_per_gpu, steps, 0.0, latency


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
    """Print Hierarchical Ring AllReduce formula (L5 Case 2 or ring-over-nodes)."""
    intra_steps = (gpus_per_node - 1) if gpus_per_node and gpus_per_node > 0 else (N - 1)
    time_intra_one = M / (N * B1) + alpha1
    time_inter_one = M / (N * B2) + alpha2
    intra_one_phase = intra_steps * time_intra_one
    if gpus_per_node and gpus_per_node > 0:
        num_nodes = N // gpus_per_node
        inter_steps = max(0, num_nodes - 1)
        print(f"  [Formula] Hierarchical Ring AllReduce: 2*(intra + inter), {inter_steps} inter step(s)/phase (ring over {num_nodes} nodes)")
    else:
        print("  [Formula] Hierarchical Ring AllReduce (fallback): 2*(intra + inter)")
    print(f"  [Input]   N={N}, M={M:.2e} B, B1={B1:.2e} B/s, alpha1={alpha1:.2e} s, B2={B2:.2e} B/s, alpha2={alpha2:.2e} s, gpus_per_node={gpus_per_node}")
    print(f"  [Intra]   intra_steps = {intra_steps}; time_intra_per_step = M/(N*B1)+alpha1 = {time_intra_one:.4e} s; intra_one_phase = {intra_one_phase:.4e} s")
    print(f"  [Inter]   time_inter_per_step = M/(N*B2)+alpha2 = {time_inter_one:.4e} s; total inter = {inter:.4e} s")
    print(f"  [Result] latency = 2*(intra+inter) = {lat:.4e} s ({lat*1000:.2f} ms)")


def _print_hierarchical_ag_rs(
    M: float, N: int, B1: float, alpha1: float, B2: float, alpha2: float,
    gpus_per_node: Optional[int], intra: float, inter: float, lat: float,
) -> None:
    """Print Hierarchical AllGather/ReduceScatter formula (L5 slide 4 or ring-over-nodes)."""
    intra_steps = (gpus_per_node - 1) if gpus_per_node and gpus_per_node > 0 else (N - 1)
    if gpus_per_node and gpus_per_node > 0:
        num_nodes = N // gpus_per_node
        print(f"  [Formula] Hierarchical AllGather/ReduceScatter: intra + inter (ring over {num_nodes} nodes)")
    else:
        print("  [Formula] Hierarchical AllGather/ReduceScatter: intra + inter")
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
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
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
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
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
            lat, vol, st, intra, inter = _hierarchical_broadcast(
                M, N, B1, alpha1, B2, alpha2, gpus_per_node
            )
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st, intra_latency_s=intra, inter_latency_s=inter)
        elif topology.kind == TopologyKind.TREE:
            lat, vol, st = _tree_broadcast_flat(M, N, B, alpha)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
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
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            lat, vol, st = _mesh_broadcast(M, n_x, n_y, B, alpha)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        else:
            lat, vol, st = _ring_broadcast_flat(M, N, B, alpha)
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
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        elif topology.kind == TopologyKind.MESH:
            n_x, n_y = _require_mesh_dims(topology)
            lat, vol, st = _mesh_alltoall(M, n_x, n_y, B, alpha)
            return CollectiveResult(latency_s=lat, volume_bytes=vol, steps=st)
        elif topology.kind == TopologyKind.HIERARCHICAL:
            lat, vol, st, intra, inter = _hierarchical_all2all(
                M, N, B1, alpha1, B2, alpha2, gpus_per_node
            )
            return CollectiveResult(
                latency_s=lat,
                volume_bytes=vol,
                steps=st,
                intra_latency_s=intra,
                inter_latency_s=inter,
            )
        # Flat All2All (ring-style model):
        # Each GPU has total payload M, logically split into N chunks of size M/N.
        # It sends (N-1) distinct chunks to peers, so total logical send volume per GPU
        # is (N-1) * (M/N). We ignore extra forwarding hops on intermediate GPUs.
        # Latency uses the same α–β template as ring AG/RS: (N-1) * (M/(N*B) + α),
        # which can be written as volume_per_gpu/B + alpha*(N-1).
        volume_per_gpu = (N - 1) * (M / N)
        latency = volume_per_gpu / B + alpha * (N - 1)
        steps = N - 1
        return CollectiveResult(latency_s=latency, volume_bytes=volume_per_gpu, steps=steps)

    return CollectiveResult(latency_s=0.0, volume_bytes=0.0, steps=0)