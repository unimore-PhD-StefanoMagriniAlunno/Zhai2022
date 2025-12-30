import pytest

__all__ = ["pytest"]

import zhai2022.euler_maruyama as em


def test_em_trajectory():
    import numpy as np
    from zhai2022.euler_maruyama import trajectory

    # model settings
    n_dim = 2

    def drift(X, t):
        return X

    def diffusion(X, t):
        return np.eye(n_dim)

    initial_state = np.zeros(2)

    # scheme settings
    n_agents = 5000
    n_steps = 1000
    dt = 0.01
    X0 = np.tile(initial_state, (n_agents, 1))

    X = trajectory(X0, n_steps, dt, drift, diffusion)

    assert X.shape == (n_steps + 1, n_agents, n_dim)
    assert X.dtype == X0.dtype


def test_em_cuda_trajectory():
    import torch
    from zhai2022.euler_maruyama.cuda import trajectory

    # model settings
    n_dim = 2

    def drift(X, t):
        return X

    def diffusion(X, t):
        return torch.eye(n_dim)

    initial_state = torch.zeros(2)

    # scheme settings
    n_agents = 5000
    n_steps = 1000
    dt = 0.01
    X0 = torch.tile(initial_state, (n_agents, 1))

    X = trajectory(X0, n_steps, dt, drift, diffusion)

    assert X.shape == (n_steps + 1, n_agents, n_dim)
    assert X.dtype == X0.dtype
    assert X.device == X0.device
