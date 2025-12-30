import torch
from math import sqrt
from typing import Callable


def trajectory(
    X0: torch.Tensor,
    n_steps: int,
    dt: float,
    drift: Callable[[torch.Tensor, float], torch.Tensor],
    diffusion: Callable[[torch.Tensor, float], torch.Tensor],
) -> torch.Tensor:
    """Simulate paths of a stochastic differential equation (SDE) using the Euler-Maruyama method.
    This function computes the trajectories of an SDE given the initial conditions, number of steps, time step size,
    drift function, and diffusion function.

    SDE:
        :math:`dX_t = dt\\text{drift}(X_t, t) + \\text{diffusion}(X_t, t) dW_t \\quad X_t\\in\\mathbb{R}^{d}`

        where :math:`W_t` is a standard Wiener process in :math:`\\mathbb{R}^{d}`, :math:`\\text{drift}\\in\\mathbb{R}^{d}`, :math:`\\text{diffusion}\\in\\mathbb{R}^{d\\times d}`.

    Discretization (Euler-Maruyama):
        :math:`X_{k+1} = X_k + dt\\text{drift}(X_k, t_k) + \\text{diffusion}(X_k, t_k) dW_k`

        where :math:`dW_k \\sim \\mathcal{N}(0, dt)`

    Parameters
    ----------
        X0 : torch.tensor
            Initial condition with shape (n_samples, n_dim).
        n_steps : int
            Number of time steps to simulate.
        dt : float
            Time step size.
        drift : Callable[[torch.tensor, float], torch.tensor]
            Function to compute the drift term. It takes the current state (with shape (`n_samples`, `n_dim`)) and time as input and returns the drift (with shape (`n_samples`,`n_dim`)).
        diffusion : Callable[[torch.tensor, float], torch.tensor]
            Function to compute the diffusion term. It takes the current state (with shape (`n_samples`, `n_dim`)) and time as input and returns the diffusion (with shape (`n_samples`,`n_dim`, `n_dim`)).

    Raises
    ------
        ValueError: If `n_steps` is negative.
        ValueError: If `dt` is non-positive.
        ValueError: If `X0` has incorrect shape.
        TypeError:  If `X0` has incorrect dtype.

    Returns
    -------
        torch.tensor
            Simulated paths of the stochastic differential equation with shape (`n_steps` + 1, `n_samples`, `n_dim`) and the same dtype as `X0`.

    Examples
    --------
    Example usage of this function:

    >>> X0 = torch.zeros(2)
    >>> n_steps = 1000
    >>> dt = 0.01

    >>> def drift(X, t):
    >>>     return X  # simple drift example

    >>> def diffusion(X, t):
    >>>     return torch.eye(X.shape[1])  # simple diffusion example

    >>> X = trajectory(X0, n_steps, dt, drift, diffusion)
    """

    # check input validity
    if X0.dtype not in [torch.float32, torch.float64]:
        raise ValueError("X0 must have dtype float32 or float64")
    if X0.ndim != 2:
        raise ValueError("X0 must be a 2D tensor with shape (n_samples, n_dim)")
    n_samples, n_dim = X0.shape
    if n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if dt <= 0:
        raise ValueError("dt must be a positive float")

    X = torch.zeros((n_steps + 1, n_samples, n_dim), dtype=X0.dtype, device=X0.device)
    X[0] = X0
    sqrt_dt: float = sqrt(dt)
    for k in range(1, n_steps + 1):
        dW = torch.randn((n_samples, n_dim), dtype=X0.dtype, device=X0.device) * sqrt_dt

        drift_term = drift(X[k - 1], k * dt)
        drift_term = torch.asarray(drift_term, dtype=X0.dtype, device=X0.device)
        try:
            drift_term = torch.broadcast_to(drift_term, (n_samples, n_dim))
        except ValueError:
            raise ValueError(
                f"drift function returned array with shape {drift_term.shape}, expected {(n_samples, n_dim)} or a shape broadcastable to it."
            )

        diffusion_term = diffusion(X[k - 1], k * dt)
        diffusion_term = torch.asarray(diffusion_term, dtype=X0.dtype, device=X0.device)
        try:
            diffusion_term = torch.broadcast_to(
                diffusion_term, (n_samples, n_dim, n_dim)
            )
        except ValueError:
            raise ValueError(
                f"diffusion function returned array with shape {diffusion_term.shape}, expected {(n_samples, n_dim, n_dim)} or a shape broadcastable to it."
            )

        X[k] = (
            X[k - 1]
            + dt * drift_term
            + torch.asarray(
                torch.einsum("sij, sj -> si", diffusion_term, dW),
                dtype=X0.dtype,
                device=X0.device,
            )
        )

    return X
