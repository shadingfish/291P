"""
Topology module: describes how GPUs are connected for communication.

Inputs:  topology kind, number of GPUs N, bandwidth B (bytes/s), latency alpha (s).
         For hierarchical: intra-node B1/alpha1, inter-node B2/alpha2.
Outputs: TopologyDesc (passed to collective module to compute latency).

Formula sources: Lecture 4 (Foundations), Lecture 5 (DP, hierarchical).
See REFERENCES.md for full citation list.

-----------------------------------------------------------------------------
CONSTRAINTS WHEN USED WITH collective.py
-----------------------------------------------------------------------------
  - RING:  Used for all collectives. Supply N, B, alpha.
  - TREE:  Only AllReduce uses a dedicated tree formula; other collectives
           are computed as flat (ring-style) using B, alpha.
  - HIERARCHICAL:  Must supply B1, alpha1, B2, alpha2. gpus_per_node is optional;
           when set (e.g. 4 for 2 nodes x 4 GPUs), collective formulas use
           (gpus_per_node - 1) intra steps per phase (Case 2, L5 slide); when None,
           intra steps = (N - 1) (simplified model).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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


@dataclass
class TopologyDesc:
    """
    Description of the network topology for use by the collective module.

    Attributes:
        kind: One of RING, TREE, HIERARCHICAL.
        N: Number of GPUs participating in the collective.
        B: (Flat topologies) Link bandwidth in bytes per second.
        alpha: (Flat topologies) Per-step latency in seconds.
        B1: (Hierarchical) Intra-node bandwidth, bytes/s.
        alpha1: (Hierarchical) Intra-node per-step latency, seconds.
        B2: (Hierarchical) Inter-node bandwidth, bytes/s.
        alpha2: (Hierarchical) Inter-node per-step latency, seconds.
        gpus_per_node: (Hierarchical) GPUs per node; if None, assumed single node.
    """
    kind: TopologyKind
    N: int
    B: float = 0.0
    alpha: float = 0.0
    B1: float = 0.0
    alpha1: float = 0.0
    B2: float = 0.0
    alpha2: float = 0.0
    gpus_per_node: Optional[int] = None


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
) -> TopologyDesc:
    """
    Build a topology description from the given parameters.

    Inputs:
        kind: Topology type (RING, TREE, or HIERARCHICAL).
        N: Number of GPUs.
        B: (Flat) Bandwidth in bytes/s. Default: NVLink from constants.
        alpha: (Flat) Latency in seconds. Default: LATENCY_NVLINK_S.
        B1, alpha1: (Hierarchical) Intra-node bandwidth and latency. Default: NVLink.
        B2, alpha2: (Hierarchical) Inter-node bandwidth and latency. Default: InfiniBand.
        gpus_per_node: (Hierarchical) GPUs per node (e.g. 4 for 2 nodes x 4 GPUs).
                       When set, collective uses (gpus_per_node - 1) intra steps;
                       when None, uses (N - 1) for backward compatibility.

    Outputs:
        TopologyDesc to pass to collective_latency_and_volume().

    References:
        L4 slide 16 (NVLink ~900 GB/s, IB ~50 GB/s).
        L4 slide 38, L5 slide 4 (hierarchical two-level formulas).
    """
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
    else:
        desc = TopologyDesc(
            kind=kind,
            N=N,
            B=B if B is not None else BANDWIDTH_NVLINK_BPS,
            alpha=alpha if alpha is not None else LATENCY_NVLINK_S,
        )
    return desc
