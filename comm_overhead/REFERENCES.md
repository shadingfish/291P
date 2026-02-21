# References and Formula Sources

All formulas and conventions in this project are tied to the following sources. When adding new code, cite the source in docstrings (e.g. `[L4-p34]`).

## Lecture slides (primary)

| Id   | File | Description |
|------|------|-------------|
| L4   | Lecture 4 LLM Training - Foundations | Ring/Tree AllReduce, AllGather, ReduceScatter, hierarchical topology |
| L5   | Lecture 5 LLM Training - DP | ZeRO memory, hierarchical Ring formulas, NCCL |
| L6   | Lecture 6 LLM Training - PP+TP | PP P2P, TP AllReduce per layer, 2D TP |
| L7   | Lecture 7 LLM Training - CP | CP All2All, Ulysses communication volume |

## Formula → source mapping

- **Ring AllReduce latency**: `2(N-1) * (M/(N*B) + alpha)` — [L4, slide 34].
- **Tree AllReduce latency**: `2*ceil(log2(N)) * (M/B + alpha)` — [L4, slide 36].
- **Ring AllGather / ReduceScatter**: `(N-1) * (M/(N*B) + alpha)` — [L4, slide 34].
- **Hierarchical Ring (2-level)**: Intra ReduceScatter + Inter reduction + Intra AllGather — [L4, slide 38]; formulas with B1/α1, B2/α2 — [L5, slide 4].
- **Per-GPU memory (no ZeRO)**: 2·Ψ (weights) + 2·Ψ (gradients) + 12·Ψ (optimizer states) — [L5, slide 18].
- **ZeRO-1/2/3** (for future extension): [L5, slides 21–26].

## Bandwidth / latency defaults

- NVLink intra-node: ~900 GB/s, low latency — [L4, slide 16].
- InfiniBand inter-node: ~50 GB/s per GPU — [L4, slide 16].
