"""
Communication and memory overhead tool for collaborative operations.

Modules (each can be developed/extended by different owners):
  - topology: Ring, Tree, Hierarchical topology descriptions
  - collective: AllReduce, AllGather, ReduceScatter, Broadcast, All2All latency/volume
  - memory: Per-GPU memory (basic: no ZeRO)
  - analyze: Orchestrates topology + collective + memory for a given config

See README.md for project structure and extension guide.
"""

from comm_overhead.topology import (
    TopologyKind,
    TopologyDesc,
    build_topology,
)
from comm_overhead.collective import (
    CollectiveKind,
    collective_latency_and_volume,
)
from comm_overhead.memory import per_gpu_memory_bytes
from comm_overhead.analyze import analyze_config, AnalysisResult, Config

__all__ = [
    "TopologyKind",
    "TopologyDesc",
    "build_topology",
    "CollectiveKind",
    "collective_latency_and_volume",
    "per_gpu_memory_bytes",
    "analyze_config",
    "AnalysisResult",
    "Config",
]
