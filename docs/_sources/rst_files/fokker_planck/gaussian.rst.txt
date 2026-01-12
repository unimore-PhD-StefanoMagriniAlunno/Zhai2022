Gaussian Steady State for Fokker-Planck Equations
=================================================

This document illustrates the Gaussian steady-state solution for a specific class of Fokker-Planck equations characterized by linear drift and constant diffusion coefficients.

In particular, we consider the Fokker-Planck equation from a potential function :math:`U(x)` with a rotational component defined by a skew-symmetric matrix :math:`J`. The drift term is given by:

:math:`b(x) = -\nabla U(x) + J x`
and the diffusion term is constant:
:math:`a(x) = \sigma^2 I`
where :math:`I` is the identity matrix and :math:`\sigma > 0` is a constant.

The steady-state solution :math:`u_{ss}(x)` of the Fokker-Planck equation in this case is a Gaussian distribution given by:

:math:`u_{ss}(x) \propto \exp\left(-2\frac{U(x)}{\sigma^2}\right)`

The user can provide the potential function :math:`U(x)` along with its first and second derivatives to instantiate the `Gaussian` class. In this way the computation of :math:`u_t` and other relevant quantities can be performed efficiently.

In particular, this class uses these identities to compute the time evolution of the probability density function :math:`u(x, t)` under the Fokker-Planck dynamics.

:math:`\frac{\partial u}{\partial t} = -(\nabla \cdot b) u - b \cdot \nabla u + \frac{1}{2} a \cdot \nabla^2 u`

:math:`-(\nabla \cdot b) = \Delta U(x) = 4 V''\left(\left\|x\right\|^2\right)\left\|x\right\|^2 + 2dV'\left(\left\|x\right\|^2\right)`

Implementations
~~~~~~~~~~~~~~~
.. automodule:: zhai2022.fokker_planck.gaussian
    :show-inheritance:
    :inherited-members:
    :members:
    :private-members:
    :special-members: __init__
    :no-index: