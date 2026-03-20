"""Topology parameter container for the communication cost model."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from math import isqrt

from comm_overhead.constants import (
    BANDWIDTH_NVLINK_BPS,
    BANDWIDTH_INFINIBAND_BPS,
    LATENCY_NVLINK_S,
    LATENCY_INFINIBAND_S,
)


class TopologyKind(str, Enum):
    """Supported topology types. Extensible (e.g. add MESH, TORUS later)."""
    RING = "ring"
    TREE = "tree"
    HIERARCHICAL = "hierarchical"
    SWITCH = "switch"
    TORUS = "torus"
    MESH = "mesh"


@dataclass
class TopologyDesc:
    """Holds the network params that `collective.py` uses (depends on topology kind)."""
    kind: TopologyKind
    N: int
    B: float = 0.0
    alpha: float = 0.0
    B1: float = 0.0
    alpha1: float = 0.0
    B2: float = 0.0
    alpha2: float = 0.0
    gpus_per_node: Optional[int] = None
    n_x: Optional[int] = None
    n_y: Optional[int] = None


def _default_mesh_dims(N: int) -> tuple[int, int]:
    """Pick a near-square (n_x, n_y) such that n_x * n_y == N."""
    if N <= 0:
        return 0, 0
    r = int(N ** 0.5)
    for nx in range(r, 0, -1):
        if N % nx == 0:
            return nx, N // nx
    return 1, N

def build_topology(
    kind: TopologyKind,
    N: int,
    B: Optional[float] = None,
    alpha: Optional[float] = None,
    B1: Optional[float] = None,
    alpha1: Optional[float] = None,
    B2: Optional[float] = None,
    alpha2: Optional[float] = None,
    gpus_per_node: Optional[int] = None,
    n_x: Optional[int] = None,
    n_y: Optional[int] = None,
) -> TopologyDesc:
    """Construct a `TopologyDesc` (flat vs hierarchical vs mesh/torus params)."""
    if kind == TopologyKind.HIERARCHICAL:
        desc = TopologyDesc(
            kind=kind,
            N=N,
            B1=B1 if B1 is not None else BANDWIDTH_NVLINK_BPS,
            alpha1=alpha1 if alpha1 is not None else LATENCY_NVLINK_S,
            B2=B2 if B2 is not None else BANDWIDTH_INFINIBAND_BPS,
            alpha2=alpha2 if alpha2 is not None else LATENCY_INFINIBAND_S,
            gpus_per_node=gpus_per_node,
        )
    elif kind == TopologyKind.RING:
        desc = TopologyDesc(
            kind=kind,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
        )
    elif kind == TopologyKind.SWITCH:
        return TopologyDesc(
            kind=TopologyKind.SWITCH,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
        )
    elif kind == TopologyKind.TORUS:
        if n_x is not None and n_y is not None:
            if n_x * n_y != N:
                raise ValueError(f"TORUS requires n_x*n_y == N, got {n_x}*{n_y} != {N}")
        else:
            r = int(isqrt(N))
            nx = 1
            for d in range(r, 0, -1):
                if N % d == 0:
                    nx = d
                    break
            n_x = nx
            n_y = N // nx

        desc = TopologyDesc(
            kind=TopologyKind.TORUS,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
            n_x=n_x,
            n_y=n_y,
        )
    elif kind == TopologyKind.MESH:
        if n_x is None or n_y is None or n_x <= 0 or n_y <= 0 or n_x * n_y != N:
            n_x, n_y = _default_mesh_dims(N)
        return TopologyDesc(
            kind=TopologyKind.MESH,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
            n_x=n_x,
            n_y=n_y,
        )
    else:
        desc = TopologyDesc(
            kind=kind,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
        )
    return desc
