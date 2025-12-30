import pytest

__all__ = ["pytest"]

from zhai2022.sde import Model


def test_sde():
    import numpy as np

    # Define drift, diffusion, and initial state functions
    def drift(X, t):
        return -X

    def diffusion(X, t):
        return np.tile(np.eye(X.shape[1]), (X.shape[0], 1, 1))

    def initial_state(n_samples):
        return np.random.randn(n_samples, 2)

    # Create a Model instance
    model = Model(drift, diffusion, initial_state, initial_time=0.0, n_dim=2)

    assert model.n_dim == 2
    assert model.initial_time == 0.0
    assert model.sample_initial_state(5).shape == (5, 2)


from zhai2022.sde.euler_maruyama import EulerMaruyama


def test_euler_maruyama():
    import numpy as np

    def drift(X, t):
        return -X

    def diffusion(X, t):
        return np.tile(np.eye(X.shape[1]), (X.shape[0], 1, 1))

    def initial_state(n_samples):
        return np.random.randn(n_samples, 2)

    model = Model(drift, diffusion, initial_state, initial_time=0.0, n_dim=2)

    dt_schedule = np.ones(100) / np.arange(1, 101) * 0.1
    euler_maruyama = EulerMaruyama(model, dt_schedule)
    assert euler_maruyama.n_steps == 100

    X = euler_maruyama.get_trajectory(n_samples=200)
    assert X.shape == (1 + len(dt_schedule), 200, model.n_dim)

    X = euler_maruyama.get_end(n_samples=150)
    assert X.shape == (150, model.n_dim)
