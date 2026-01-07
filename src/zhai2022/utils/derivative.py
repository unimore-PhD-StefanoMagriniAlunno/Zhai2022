import torch
from torch.autograd.functional import jacobian


class Derivative:
    @staticmethod
    def numerical_derivative(func, x, h=1e-5):
        """
        Calculate the numerical derivative of a function at a given point using central difference.

        Parameters:
        func : callable
            The function for which to compute the derivative.
        x : float
            The point at which to compute the derivative.
        h : float, optional
            The step size for the finite difference approximation (default is 1e-5).

        Returns:
        float
            The numerical derivative of the function at point x.
        """
        return (func(x + h) - func(x - h)) / (2 * h)
