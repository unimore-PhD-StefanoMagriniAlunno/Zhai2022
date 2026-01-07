Euler-Maruyama Method Documentation
===================================

This module implements the Euler-Maruyama method for simulating stochastic differential equations (SDEs):

:math:`dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t`

where:

- :math:`X_t` is the state variable at time :math:`t`, it is a stochastic process in :math:`\mathbb{R}^n`
- :math:`\mu` is the drift coefficient, a function :math:`\mu: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`\sigma` is the diffusion coefficient, a function :math:`\sigma: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`W_t` is a Wiener process (or Brownian motion) in :math:`\mathbb{R}^n`.

The Euler-Maruyama method is a numerical technique used to approximate solutions to stochastic differential equations.
It is an extension of the Euler method for ordinary differential equations, incorporating stochastic components.

The method discretizes the time interval :math:`[0, T]` into :math:`N` steps of size :math:`\Delta t = T/N`.
At each time step, the state variable is updated according to the formula:

:math:`X_{t+\Delta t} = X_t + \mu(X_t, t) \Delta t + \sigma(X_t, t) \Delta W_t`

where :math:`\Delta W_t` is a normally distributed random variable with mean :math:`0` and variance :math:`\Delta t`, representing the increment of the Wiener process.

The method also requires the initial condition :math:`X_0`, which is typically sampled from a specified distribution.

Error Analysis
--------------

The strong error of the Euler-Maruyama method is defined as the expected value of the absolute difference between the true solution :math:`X_T` and the numerical approximation :math:`X_T^{\Delta t}` at time :math:`T`:

:math:`E\left[\left\|X_T(W_t) - X_T^{\Delta t}(W_t)\right\|^2\right]`

where:

- :math:`X_T(W_t)` is the exact solution of the SDE at time :math:`T` given the Wiener process :math:`W_t`,
- :math:`X_T^{\Delta t}(W_t)` is the approximation obtained using the Euler-Maruyama method with time step :math:`\Delta t` via Wiener process :math:`W_t`.

For this definition of error, it can be shown that the Euler-Maruyama method has a strong convergence order of :math:`0.5`, meaning that the error decreases proportionally to :math:`\sqrt{\Delta t}` as the time step :math:`\Delta t` approaches zero.

The weak error is defined in terms of the expected value of a test function :math:`\phi` applied to the solution:

:math:`\left|E\left[\phi(X_T(W_t))\right] - E\left[\phi(X_T^{\Delta t}(W_t))\right]\right|`

For the weak error, the Euler-Maruyama method has a convergence order of :math:`1`, meaning that the error decreases proportionally to :math:`\Delta t` as the time step :math:`\Delta t` approaches zero.
A consequence of this is that the Euler-Maruyama method is particularly effective for approximating expectations of functionals of the solution, which is often the primary interest in applications involving SDEs.

Implementations
~~~~~~~~~~~~~~~~~

.. automodule:: zhai2022.sde.euler_maruyama
    :show-inheritance:
    :inherited-members:
    :members:
    :private-members:
    :special-members: __init__
    :no-index:
