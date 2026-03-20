"""
Memory module: per-GPU memory for training (basic version, no ZeRO).

Inputs:  parameter count Psi, optional dtype (FP16/FP32 for weight/grad and optimizer).
Outputs: per-GPU memory in bytes; optional breakdown (weights, gradients, optimizer).

Formula source: Lecture 5, slide 18 (AdamW optimizer states).
See REFERENCES.md.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

from comm_overhead.constants import BYTES_FP16, BYTES_FP32


@dataclass
class MemoryBreakdown:
    weights_bytes: float
    gradients_bytes: float
    optimizer_bytes: float


def per_gpu_memory_bytes(
    num_parameters: float,
    weight_grad_bytes_per_param: int = BYTES_FP16,
    optimizer_bytes_per_param: int = 12,
    return_breakdown: bool = False,
) -> Union[float, Tuple[float, MemoryBreakdown]]:
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
