Monte Carlo Documentation
=========================

This module implements the Euler-Maruyama method for simulating stochastic differential equations (SDEs):

:math:`dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t`

where:

- :math:`X_t` is the state variable at time :math:`t`, it is a stochastic process in :math:`\mathbb{R}^n`
- :math:`\mu` is the drift coefficient, a function :math:`\mu: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`\sigma` is the diffusion coefficient, a function :math:`\sigma: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`W_t` is a Wiener process (or Brownian motion) in :math:`\mathbb{R}^n`.

In :cite:p:`SDE_existence_uniqueness` the author proved that these conditions ensure the existence and uniqueness of solutions:

- :math:`\mu` and :math:`\sigma` should be measurable functions.
- They should satisfy the Lipschitz condition: there exists a constant :math:`L > 0` such that for all :math:`x, y \in \mathbb{R}^n` and :math:`t \in \mathbb{R}^+`, :math:`\|\mu(x, t) - \mu(y, t)\| + \|\sigma(x, t) - \sigma(y, t)\| \leq L \|x - y\|`.
- They should also satisfy the linear growth condition: there exists a constant :math:`K > 0` such that for all :math:`x \in \mathbb{R}^n` and :math:`t \in \mathbb{R}^+`, :math:`\|\mu(x, t)\|^2 + \|\sigma(x, t)\|^2 \leq K(1 + \|x\|^2)`.

In summary, these conditions ensure that the SDE has a unique solution, which is essential for the stability and reliability of numerical methods like the Euler-Maruyama method.

A Monte Carlo method is employed to simulate multiple trajectories of the SDE, allowing for statistical analysis of the system's behavior over time.
In particular, given an initial condition :math:`X_0` (a random variable), we generate :math:`M` independent initial states :math:`\left\{Y_0^{(i)}\right\}_{i=1}^M` according to the distribution of :math:`X_0`.
For each initial state, we simulate the trajectories :math:`\left\{Y_t^{(i)}\right\}_{t\in(0,T)}` using some scheme.
The idea is to approximate the distribution of :math:`X_t` by the empirical distribution of the samples :math:`\left\{Y_t^{(i)}\right\}_{i=1}^M` at each time point :math:`t`.

Euler-Maruyama Method
-----------------------

The Euler-Maruyama method is a numerical technique used to approximate solutions to stochastic differential equations.
It is an extension of the Euler method for ordinary differential equations, incorporating stochastic components.

The method discretizes the time interval :math:`[0, T]` into :math:`N` steps of size :math:`\Delta t = T/N`.
At each time step, the state variable is updated according to the formula:

:math:`X_{t+\Delta t} = X_t + \mu(X_t, t) \Delta t + \sigma(X_t, t) \Delta W_t`

where :math:`\Delta W_t` is a normally distributed random variable with mean :math:`0` and variance :math:`\Delta t`, representing the increment of the Wiener process.

The method also requires the initial condition :math:`X_0`, which is typically sampled from a specified distribution.

Error Analysis
~~~~~~~~~~~~~~

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

.. autofunction:: zhai2022.euler_maruyama.trajectory
    :no-index:

.. autofunction:: zhai2022.euler_maruyama.cuda.trajectory
    :no-index:
