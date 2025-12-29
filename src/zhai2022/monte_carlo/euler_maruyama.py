import numpy as np
from numpy.typing import NDArray
from typing import Callable


def euler_maruyama(
    X0: NDArray[np.float64],
    steps: int,
    dt: float,
    drift: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
    diffusion: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Simulate paths of a stochastic differential equation (SDE) using the Euler-Maruyama method.

    Parameters
    ----------
    X0 : NDArray[np.float64]
        Initial condition with shape (n_samples, n_dim).
    steps : int
        Number of time steps to simulate.
    dt : float
        Time step size.
    drift : Callable[[NDArray[np.float64], float], NDArray[np.float64]]
        Function to compute the drift term. It takes the current state and time as input and returns the drift.
    diffusion : Callable[[NDArray[np.float64], float], NDArray[np.float64]]
        Function to compute the diffusion term. It takes the current state and time as input and returns the diffusion.

    Raises
    ------
        ValueError: X0 must be a 2D array with shape (n_samples, n_dim)
        ValueError: steps must be a non-negative integer
        ValueError: dt must be a positive float

    Returns
    -------
        NDArray[np.float64]
            Simulated paths with shape (steps + 1, n_samples, n_dim).
    """

    # check input validity
    if X0.ndim != 2:
        raise ValueError("X0 must be a 2D array with shape (n_samples, n_dim)")
    n_samples, n_dim = X0.shape
    if steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if dt <= 0:
        raise ValueError("dt must be a positive float")

    X = np.zeros((steps + 1, n_samples, n_dim), dtype=X0.dtype)
    X[0] = X0
    sqrt_dt: float = np.sqrt(dt)
    for k in range(1, steps + 1):
        dW = np.asarray(np.random.normal(0.0, sqrt_dt, size=(n_samples, n_dim))).astype(
            X0.dtype
        )
        X[k] = X[k - 1] + drift(X[k - 1], k * dt) + diffusion(X[k - 1], k * dt) * dW

    return X
