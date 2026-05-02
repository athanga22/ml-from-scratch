"""
Manual backpropagation for a 2-layer MLP.

The point of this file: prove I understand that autograd is just
chain rule + a computation graph. No torch.autograd here —
we compute every gradient by hand.

Architecture:
    x -> Linear(in, hidden) -> ReLU -> Linear(hidden, out) -> MSE loss

Forward pass:
    z1 = x @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    loss = mean((z2 - y) ** 2)

Backward pass (chain rule):
    dL/dz2 = 2 * (z2 - y) / N
    dL/dW2 = a1.T @ dL/dz2
    dL/db2 = sum(dL/dz2, dim=0)
    dL/da1 = dL/dz2 @ W2.T
    dL/dz1 = dL/da1 * (z1 > 0)        # ReLU derivative
    dL/dW1 = x.T @ dL/dz1
    dL/db1 = sum(dL/dz1, dim=0)
"""

import torch


class ManualMLP:
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        # Kaiming-ish init for ReLU
        self.W1 = torch.randn(in_dim, hidden_dim, generator=g) * (2.0 / in_dim) ** 0.5
        self.b1 = torch.zeros(hidden_dim)
        self.W2 = torch.randn(hidden_dim, out_dim, generator=g) * (2.0 / hidden_dim) ** 0.5
        self.b2 = torch.zeros(out_dim)

        # Cached activations for backward pass
        self._cache: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = x @ self.W1 + self.b1
        a1 = torch.clamp(z1, min=0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        self._cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
        return z2

    def backward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        x, z1, a1, z2 = (self._cache[k] for k in ["x", "z1", "a1", "z2"])
        N = y.shape[0]

        dz2 = 2 * (z2 - y) / N
        dW2 = a1.T @ dz2
        db2 = dz2.sum(dim=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0).float()  # ReLU derivative
        dW1 = x.T @ dz1
        db1 = dz1.sum(dim=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads: dict[str, torch.Tensor], lr: float = 1e-2) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean()


if __name__ == "__main__":
    # Sanity check: train on a tiny regression task, verify loss decreases
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = (x @ torch.tensor([[1.0], [-2.0], [3.0], [0.5]])) + 0.1 * torch.randn(64, 1)

    model = ManualMLP(in_dim=4, hidden_dim=16, out_dim=1)
    for step in range(200):
        pred = model.forward(x)
        loss = mse_loss(pred, y)
        grads = model.backward(y)
        model.step(grads, lr=0.05)
        if step % 50 == 0:
            print(f"step {step:3d}  loss {loss.item():.4f}")
