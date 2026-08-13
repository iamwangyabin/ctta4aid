"""Vendored LAME algorithm core with framework-independent imports.

Source commit: d2e5f63090bc1c8129bf7cbd781029a5955e1a67
Source file: src/adaptation/lame.py
License: CC BY-NC-SA 4.0, see THIRD_PARTY_NOTICES.md

The affinity definitions, Laplacian fixed-point update, convergence condition,
and entropy energy below are the authors' released implementation. Detectron2
runner/model I/O is deliberately left to the framework wrapper.

Modifications (2026-07-16): removed Detectron2 registry/runner imports and
added an explicit affinity factory; the numerical method is unchanged.
"""

import logging

import torch


logger = logging.getLogger(__name__)


class AffinityMatrix:
    def __init__(self, **kwargs):
        pass

    def __call__(self, X, **kwargs):
        raise NotImplementedError

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())


class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)
        n_neighbors = min(self.knn + 1, N)
        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]
        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)
        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs["knn"]

    def __call__(self, X):
        N = X.size(0)
        dist = torch.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=-1, p=2)
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]
        sigma = kth_dist.mean()
        return torch.exp(-dist**2 / (2 * sigma**2))


class linear_affinity(AffinityMatrix):
    def __call__(self, X: torch.Tensor):
        return torch.matmul(X, X.t())


def build_affinity(name: str, *, sigma: float = 1.0, knn: int = 5):
    classes = {
        "knn": kNN_affinity,
        "rbf": rbf_affinity,
        "linear": linear_affinity,
    }
    normalized = name.lower()
    if normalized not in classes:
        raise ValueError(f"Unknown LAME affinity: {name}")
    return classes[normalized](sigma=sigma, knn=knn)


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    E_list = []
    oldE = float("inf")
    Y = (-unary).softmax(-1)
    for i in range(max_steps):
        pairwise = bound_lambda * kernel.matmul(Y)
        exponent = -unary + pairwise
        Y = exponent.softmax(-1)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)
        if i > 1 and abs(E - oldE) <= 1e-8 * abs(oldE):
            logger.info("Converged in %d iterations", i)
            break
        oldE = E
    return Y


def entropy_energy(Y, unary, pairwise, bound_lambda):
    return (
        unary * Y
        - bound_lambda * pairwise * Y
        + Y * torch.log(Y.clip(1e-20))
    ).sum()
