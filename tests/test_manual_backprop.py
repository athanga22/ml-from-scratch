"""
Verify manual backprop matches torch.autograd to machine precision.

This is the test pattern for the whole repo: every from-scratch
implementation gets compared against PyTorch's official version.
If they don't match, the from-scratch one is wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foundations.manual_backprop import ManualMLP, mse_loss  # type: ignore


def test_manual_backprop_matches_autograd():
    torch.manual_seed(42)
    x = torch.randn(8, 4)
    y = torch.randn(8, 1)

    # Manual model
    manual = ManualMLP(in_dim=4, hidden_dim=16, out_dim=1, seed=0)
    pred = manual.forward(x)
    grads = manual.backward(y)

    # PyTorch reference using the same weights
    W1 = manual.W1.clone().requires_grad_(True)
    b1 = manual.b1.clone().requires_grad_(True)
    W2 = manual.W2.clone().requires_grad_(True)
    b2 = manual.b2.clone().requires_grad_(True)

    z1 = x @ W1 + b1
    a1 = F.relu(z1)
    z2 = a1 @ W2 + b2
    loss = F.mse_loss(z2, y)
    loss.backward()

    # Compare every gradient
    assert torch.allclose(grads["W1"], W1.grad, atol=1e-6), "W1 gradient mismatch"
    assert torch.allclose(grads["b1"], b1.grad, atol=1e-6), "b1 gradient mismatch"
    assert torch.allclose(grads["W2"], W2.grad, atol=1e-6), "W2 gradient mismatch"
    assert torch.allclose(grads["b2"], b2.grad, atol=1e-6), "b2 gradient mismatch"


if __name__ == "__main__":
    test_manual_backprop_matches_autograd()
    print("✓ manual backprop matches torch.autograd")
