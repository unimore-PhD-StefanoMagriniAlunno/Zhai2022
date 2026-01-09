import torch
from torch.autograd.functional import jacobian
from typing import Callable


def nabla_at(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Compute the Jacobian matrix of a vector-valued function at a given point.

    In particular, :math:`J(f)_{b_1\\ldotsb_n,a_1\\ldots a_m} = \\frac{df_{b_1\\ldots b_n}}{dx_{a_1\\ldots a_m}}`

    Parameters
    ----------
    f : Callable[[torch.Tensor], torch.Tensor]
        A function that takes a tensor input with shape (N,a1,...) and returns a tensor output with shape (N,b1,...).
        In particular, f should be differentiable and support autograd.
    x : torch.Tensor
        A tensor input with shape (a1,...) at which to compute the Jacobian.
    create_graph : bool, optional
        If True, the computation graph of the Jacobian will be constructed, allowing for higher-order derivatives. Default is False.

    Raises
    ------
    RuntimeError
        If the Jacobian computation fails, possibly due to non-differentiable function f.

    Returns
    -------
    torch.Tensor
        A tensor representing the Jacobian matrices with shape (b1,...,a1,...),
        where each slice along the first dimension corresponds to the Jacobian of f at the respective input in x.
        In particular, if J is the returned tensor, then J[b1,...,bn,a1,...,am] = df_b1...bn / dx_a1...am.

    Examples
    --------
    >>> import torch
    >>> from zhai2022.differential import nabla_at
    >>> # Define a vector-valued function
    >>> def f(x):
    ...     return torch.stack([x**2, x**3], dim=-1)

    `f` takes a tensor with shape (N,) and returns a tensor with shape (N,2) where N is the batch size.

    >>> # Point at which to compute the Jacobian
    >>> x = torch.tensor(1.0)  # shape ()
    >>> # Compute the Jacobian at x
    >>> J = nabla_at(f, x)  # shape (2,)

    `J[i]` is the derivative of the i-th output of f with respect to x.

    >>> J.shape
    torch.Size([2])
    >>> J
    tensor([2., 3.])
    """
    # activate autograd
    x = x.requires_grad_(True)
    try:
        J: torch.Tensor = jacobian(
            f,
            x[None, ...],
            create_graph=create_graph,
        )  # has shape (1,b1,...,1,a1,...)
    except RuntimeError as e:
        raise RuntimeError(
            "Error computing Jacobian. Ensure that the function f is differentiable and supports autograd."
        ) from e
    else:
        K = len(x.shape) + 1
        J = J.squeeze(0).squeeze(-K)  # has shape (b1,...,a1,...)
        return J


def div_at(
    f: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    create_graph: bool = False,
) -> torch.Tensor:
    """
    Compute the divergence of a function at a given point.

    In particular, :math:`\\text{Div}(f)_{b_1\\ldotsb_n} = \\sum_{a_1\\ldots a_m} \\frac{df_{b_1...b_n}}{dx_{a_1\\ldots a_m}}`

    Parameters
    ----------
    f : Callable[[torch.Tensor], torch.Tensor]
        A function that takes a tensor input with shape (N,a1,...) and returns a tensor output with shape (N,b1,...,a1,...).
        In particular, f should be differentiable and support autograd.
    x : torch.Tensor
        A tensor input with shape (a1,...) at which to compute the divergence.
    create_graph : bool, optional
        If True, the computation graph of the divergence will be constructed, allowing for higher-order derivatives. Default is False.

    Raises
    ------
    ValueError
        If the output of f does not have the same leading dimensions as the input x.

    Returns
    -------
    torch.Tensor
        A tensor representing the divergence with shape (b1,...),
        which is the sum of the diagonal elements of the Jacobian matrix of f at x.
        In particular, if D is the returned tensor, then D[b1,...,bn] = sum_i df_b1...bn / dx_ai.

    Examples
    --------
    >>> import torch
    >>> from zhai2022.differential import div_at
    >>> # Define a vector-valued function
    >>> def f(x):
    ...     return torch.stack([
                torch.stack([x[:,0]**2+x[:,1]**2, x[:,0]+x[:,1]], dim=-1),
                torch.stack([x[:,0]*x[:,1], x[:,0]-x[:,1]], dim=-1),
                torch.stack([x[:,0]**3, x[:,1]**3], dim=-1)
            ], dim=-2)

    `f` takes a tensor with shape (N,2) and returns a tensor with shape (N,3,2) where N is the batch size.

    >>> # Point at which to compute the divergence
    >>> x = torch.tensor([1.0, 2.0])  # shape (2,)
    >>> # Compute the divergence at x
    >>> D = div_at(f, x)  # shape (3,)
    >>> D.shape
    torch.Size([3])
    >>> D
    tensor([ 3.,  1., 15.])
    """
    J: torch.Tensor = nabla_at(
        f,
        x,
        create_graph=create_graph,
    )  # has shape (b1,...,a1,...,a1,...)
    # memorize the original shape
    a_shape = x.shape
    # check that the leading dimensions match
    if J.shape[-len(a_shape) * 2 :] != a_shape + a_shape:
        raise ValueError(
            "The output of f must have the same leading dimensions as the input x for divergence computation."
        )
    b_shape = J.shape[: -2 * len(a_shape)]
    # reshape J to isolate the leading dimensions
    J = J.reshape(
        *b_shape, x.numel(), x.numel()
    )  # has shape (*b_shape,numel_a, numel_a)
    # compute trace over the first two dimensions
    return torch.einsum("...ii->...", J)  # has shape b_shape
