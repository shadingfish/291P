"""
Default constants for bandwidth, latency, and data types.

Used across topology and collective modules when the user does not supply values.
All formula sources are cited in REFERENCES.md and in the topology/collective modules.
"""

# -----------------------------------------------------------------------------
# Bandwidth (bytes per second) — source: Lecture 4, slide 16
# -----------------------------------------------------------------------------
# NVLink within a node: bidirectional ~900 GB/s per GPU
BANDWIDTH_NVLINK_GBPS = 900.0
BANDWIDTH_NVLINK_BPS = BANDWIDTH_NVLINK_GBPS * (1024**3)

# InfiniBand (or similar) across nodes: ~50 GB/s per GPU
BANDWIDTH_INFINIBAND_GBPS = 50.0
BANDWIDTH_INFINIBAND_BPS = BANDWIDTH_INFINIBAND_GBPS * (1024**3)

# -----------------------------------------------------------------------------
# Latency (seconds) — typical orders of magnitude; not from a specific slide
# -----------------------------------------------------------------------------
# Intra-node latency (NVLink): low, e.g. microseconds
LATENCY_NVLINK_S = 1e-6

# Inter-node latency (InfiniBand): higher
LATENCY_INFINIBAND_S = 5e-6

# -----------------------------------------------------------------------------
# Data type sizes (bytes per element) — standard
# -----------------------------------------------------------------------------
BYTES_FP16 = 2
BYTES_FP32 = 4
BYTES_BF16 = 2
