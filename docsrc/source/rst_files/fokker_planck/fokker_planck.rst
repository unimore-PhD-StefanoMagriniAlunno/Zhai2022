Fokker-Planck Equation Model
============================

This module provides a class to model the Fokker-Planck equation, which describes the time evolution of the probability density function of the velocity of a particle under the influence of forces such as drag and random forces.

Formally, the Fokker-Planck equation is given by:

:math:`\frac{\partial u}{\partial t} = -\nabla \cdot (b(x, t) u) + \frac{1}{2} \nabla^2 : (a(x, t) u)`

where :math:`u(x, t)` is the probability density function, :math:`b(x, t)` is the drift coefficient, and :math:`a(x, t)` is the diffusion coefficient.

This module uses the `Differential` class from the `zhai2022.differential` module to compute the necessary derivatives.

Implementations
~~~~~~~~~~~~~~~
.. automodule:: zhai2022.fokker_planck.model
    :show-inheritance:
    :inherited-members:
    :members:
    :private-members:
    :special-members: __init__
    :no-index:

.. toctree::
    :maxdepth: 2
    :caption: Contents:
    
    gaussian

