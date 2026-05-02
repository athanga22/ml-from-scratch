# 01 — Foundations

Tensors, autograd, manual backprop, and the optimizers that train everything else.

## Files

| File | What it implements | Status |
|---|---|---|
| `01_tensor_ops.py` | Broadcasting, reshape, einsum patterns | ⬜ |
| `02_manual_backprop.py` | Backward pass for a 2-layer MLP, no autograd | ⬜ |
| `03_autograd_basics.py` | Computation graph, `requires_grad`, `.backward()` | ⬜ |
| `04_optimizers.py` | SGD, Momentum, Adam from scratch | ⬜ |
| `05_lr_schedulers.py` | Cosine, warmup, step decay | ⬜ |

## What I learned

*Fill this in as you go. One bullet per file. These bullets become your interview answers.*

- Manual backprop teaches you that `.backward()` is just chain rule applied to a graph — the "magic" is graph construction, not differentiation
- Adam = momentum (1st moment) + per-parameter scaling (2nd moment), with bias correction for the first few steps

## Gotchas

- *Add the non-obvious bug or insight from each file here. This is the gold for interviews.*

## References

- Deep-ML: Linear Algebra, PyTorch Basics
- *Add papers/posts that were actually useful*
