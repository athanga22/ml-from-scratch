# AI Foundations

PyTorch implementations of core deep learning and LLM components, built from first principles. Each module is paired with a short writeup explaining the math, the engineering tradeoffs, and the gotchas.

> Why this repo exists: I work on agentic RAG systems and LLM evaluation pipelines in production. This repo is where I keep the foundations sharp — implementing the building blocks (attention, backprop, LoRA, KV cache) from scratch so I can reason about them under interview pressure and in production debugging.

## Structure

| Module | What's inside |
|---|---|
| [01_foundations](./01_foundations) | Tensors, autograd, manual backprop, gradient descent variants |
| [02_deep_learning](./02_deep_learning) | MLPs, CNNs, RNNs, normalization layers, optimizers |
| [03_attention_transformers](./03_attention_transformers) | Scaled dot-product attention, multi-head attention, positional encodings, full transformer block, GPT-style decoder |
| [04_fine_tuning](./04_fine_tuning) | LoRA, QLoRA, parameter-efficient fine-tuning, training loops |
| [05_inference](./05_inference) | KV cache, batching, quantization basics, sampling strategies |

## How to read this repo

Each implementation follows the same pattern:

1. **`*.py`** — clean implementation with type hints and docstrings
2. **`README.md`** — short writeup: what it does, why it works, what breaks
3. **`test_*.py`** — sanity tests against PyTorch's reference implementation

Start with `01_foundations` and work forward. Files are numbered.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

## Status

🟢 = done · 🟡 = in progress · ⬜ = not started

- 🟡 01 — Foundations
- ⬜ 02 — Deep Learning
- ⬜ 03 — Attention & Transformers
- ⬜ 04 — Fine-Tuning
- ⬜ 05 — Inference

## About

I'm Ashish Thanga, an AI Engineer focused on LLM systems. [LinkedIn](https://www.linkedin.com/in/) · [Portfolio]()
