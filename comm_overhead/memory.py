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
    Per-GPU memory for training without ZeRO (each GPU holds full model + grad + optimizer).

    Inputs:
        num_parameters: Psi, total number of parameters (e.g. 70e9 for 70B).
        weight_grad_bytes_per_param: Bytes per parameter for weights and gradients.
            Default 2 (FP16). Source: L5 slide 18.
        optimizer_bytes_per_param: Total optimizer state bytes per parameter. For AdamW:
            first moment (4) + second moment (4) + master weights (4) = 12. Default 12.
            Source: L5 slide 18.
        return_breakdown: If True, also return MemoryBreakdown (weights, gradients, optimizer).

    Outputs:
        Total per-GPU memory in bytes. If return_breakdown is True, returns (total, breakdown).

    Variable meanings:
        Psi (num_parameters): Model parameter count.
        Weights: Psi * 2 bytes (FP16).
        Gradients: Psi * 2 bytes (FP16).
        Optimizer states: Psi * 12 bytes (three FP32 copies for AdamW).

    Formula (source L5 slide 18):
        Model Weight: 2 * Psi bytes
        Gradients: 2 * Psi bytes
        Optimizer (AdamW): 12 * Psi bytes  (m_t, v_t, master weight each 4*Psi)
        Total = 16 * Psi bytes (for default FP16 + AdamW)

    References:
        L5 slide 18 (memory cost in training, AdamW state equations).
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
