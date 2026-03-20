"""Per-GPU memory accounting for training (baseline, no ZeRO partitioning)."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from comm_overhead.constants import BYTES_FP16, BYTES_FP32


@dataclass
class MemoryBreakdown:
    """Per-component memory in bytes (for reporting)."""
    weights_bytes: float
    gradients_bytes: float
    optimizer_bytes: float


def per_gpu_memory_bytes(
    num_parameters: float,
    weight_grad_bytes_per_param: int = BYTES_FP16,
    optimizer_bytes_per_param: int = 12,
    return_breakdown: bool = False,
) -> Union[float, Tuple[float, MemoryBreakdown]]:
    """
    ZeRO-0: each rank keeps full weights, grads, and AdamW states. Mem_gpu ≈ 16Ψ bytes.
    """
    Psi = num_parameters
    weights_bytes = Psi * weight_grad_bytes_per_param
    gradients_bytes = Psi * weight_grad_bytes_per_param
    optimizer_bytes = Psi * optimizer_bytes_per_param
    total = weights_bytes + gradients_bytes + optimizer_bytes

    if return_breakdown:
        return total, MemoryBreakdown(
            weights_bytes=weights_bytes,
            gradients_bytes=gradients_bytes,
            optimizer_bytes=optimizer_bytes,
        )
    return total
