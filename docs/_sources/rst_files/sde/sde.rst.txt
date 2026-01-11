SDE Model Documentation
=======================

This module implements a class that represents the SDE:

:math:`dX_t = \mu(X_t, t) dt + \sigma(X_t, t) dW_t`

where:

- :math:`X_t` is the state variable at time :math:`t`, it is a stochastic process in :math:`\mathbb{R}^n`
- :math:`\mu` is the drift coefficient, a function :math:`\mu: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`\sigma` is the diffusion coefficient, a function :math:`\sigma: \mathbb{R}^n \times \mathbb{R}^+ \to \mathbb{R}^n`
- :math:`W_t` is a Wiener process (or Brownian motion) in :math:`\mathbb{R}^n`.

In :cite:p:`SDE_existence_uniqueness`, it is proven that the following conditions ensure the existence and uniqueness of solutions:

- :math:`\mu` and :math:`\sigma` should be measurable functions.
- They must satisfy the Lipschitz condition: there exists a constant :math:`L > 0` such that for all :math:`x, y \in \mathbb{R}^n` and :math:`t \in \mathbb{R}^+`, :math:`\|\mu(x, t) - \mu(y, t)\| + \|\sigma(x, t) - \sigma(y, t)\| \leq L \|x - y\|`.
- Additionally, they must satisfy the linear growth condition: there exists a constant :math:`K > 0` such that for all :math:`x \in \mathbb{R}^n` and :math:`t \in \mathbb{R}^+`, :math:`\|\mu(x, t)\|^2 + \|\sigma(x, t)\|^2 \leq K(1 + \|x\|^2)`.

In summary, these conditions ensure that the SDE has a unique solution, which is essential for the stability and reliability of numerical methods like the Euler-Maruyama method.

Implementations
~~~~~~~~~~~~~~~

.. automodule:: zhai2022.sde
    :show-inheritance:
    :inherited-members:
    :members:
    :private-members:
    :special-members: __init__
    :no-index:

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    euler_maruyama/euler_maruyama
    torch/torch