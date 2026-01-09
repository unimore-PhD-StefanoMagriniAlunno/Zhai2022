import torch
from typing import Callable, List, Tuple
from zhai2022.differential.functions import nabla_at, div_at


class Differential:
    class Nabla:
        def __init__(
            self,
            f: Callable[[torch.Tensor], torch.Tensor],
            n: int,
            create_graph: bool = False,
        ) -> None:
            """
            Initialize the Nabla class with a function. This class is a callable object.

            Parameters
            ----------
            f : Callable[[torch.Tensor], torch.Tensor]
                A function for which to compute the Jacobian. It takes a tensor with shape (N,a1,...) and returns a tensor with shape (N,b1,...)
            n : int
                The order of the Differential to compute.
            create_graph : bool, optional
                If True, the computation graph will be constructed, allowing for higher-order Differentials. Default is False.

            Raises
            ------
                ValueError
                    If the order of Differential n is a negative integer.

            Examples
            --------
            >>> # Let `diff` be a Differential object
            >>> # Define a function f
            >>> def f(x):
            ...     return torch.stack([x**2, x**3], dim=-2)
            >>> # Get the Nabla callable object
            >>> nabla_f = diff(f)
            """
            if n < 0:
                raise ValueError(
                    "Order of Differential n must be a non-negative integer."
                )

            self.__f = f
            self.__n = n
            self.__create_graph = create_graph

        @property
        def order(self) -> int:
            """
            Get the order of the Differential.

            Returns
            -------
            int
                The order of the Differential.
            """
            return self.__n

        @property
        def create_graph(self) -> bool:
            """
            Get whether the computation graph is created.

            Returns
            -------
            bool
                True if the computation graph is created, False otherwise.
            """
            return self.__create_graph

        def __call__(
            self, x: torch.Tensor, full_trace: bool = False
        ) -> Tuple[torch.Tensor, ...]:
            """
            Compute the Jacobian matrix of the stored function at a given point.

            In particular, for a function :math:`f: \\mathbb{R}^{a_1,...} -> \\mathbb{R}^{b_1,...}`, this method returns :math:`\\nabla_x f(x)`.

            It is possible to compute full traces of higher-order Differentials by setting the `full_trace` argument to True.

            Parameters
            ----------
            x : torch.Tensor
                A tensor input with shape (N,a1,...) at which to compute the Jacobian. The first dimension N represents the batch size.
            full_trace : bool, optional
                If True, the function returns all intermediate Differentials up to order n. Default is False.

            Returns
            -------
            Tuple[torch.Tensor, ...]
                A tuple of tensors representing the Jacobian matrices with shape (N,b1,...,[a1,... xn-times]).
                If full_trace is True, returns a tuple of tensors representing all intermediate Differentials up to order n.
                If full_trace is False, returns only the n-th order Differential.

            Examples
            --------
            >>> # Let `nabla_f` be a Nabla callable object
            >>> # where f takes a tensor with shape (N,2) and returns a tensor with shape (N,3)
            >>> # Point at which to compute the Jacobian
            >>> x = torch.rand(5,2)  # N = 5
            >>> # Compute the Jacobian at x
            >>> J = nabla_f(x)  # shape (5,3,2)
            """
            N = x.shape[0]
            J_list: List[torch.Tensor] = []
            if self.__n == 0:
                return (self.__f(x),)
            if self.__n == 1:
                for i in range(N):
                    J_i: torch.Tensor = nabla_at(
                        self.__f, x[i], create_graph=self.__create_graph
                    )  # has shape (b1,...,a1,...)
                    J_list.append(J_i)
                J = torch.stack(J_list, dim=0)  # has shape (N,b1,...,a1,...)
                if full_trace:
                    return (self.__f(x), J)  # shapes ((N,b1,...), (N,b1,...,a1,...))
                else:
                    return (J,)  # shape (N,b1,...,a1,...)
            else:  # n > 1
                # make a Differential.Nabla object with order 1 and create_graph = True
                nabla1 = Differential.Nabla(
                    self.__f,
                    1,
                    create_graph=True,
                )
                # x -> nabla1(x) is the new callable object
                df = lambda y: nabla1(y, full_trace=False)[
                    0
                ]  # takes a tensor with shape (N,a1,...) and returns a tensor with shape (N,b1,...,a1,...)
                if full_trace:
                    return (self.__f(x),) + Differential.Nabla(
                        df,
                        self.__n - 1,
                        create_graph=self.__create_graph,
                    )(
                        x, full_trace=True
                    )  # has shape ((N,b1,...), (N,b1,...,a1...), ... (N,b1,...,a1,...,[a1,... x(n-1)-times]))
                return Differential.Nabla(
                    df,
                    self.__n - 1,
                    create_graph=self.__create_graph,
                )(
                    x, full_trace=False
                )  # has shape (b1,...,a1,...,[a1,... x(n-1)-times])

    class Div:
        def __init__(
            self,
            f: Callable[[torch.Tensor], torch.Tensor],
            n: int,
            create_graph: bool = False,
        ) -> None:
            """
            Initialize the Div class with the order of Differential. This class is a callable object.

            Parameters
            ----------
            f : Callable[[torch.Tensor], torch.Tensor]
                A function for which to compute the Divergence. It takes a tensor with shape (N,a1,...) and returns a tensor with shape (N,a1,... xn times,b1,...)
            n : int
                The order of the Differential to compute.
            create_graph : bool, optional
                If True, the computation graph will be constructed, allowing for higher-order Differentials. Default is False.

            Raises
            ------
                ValueError
                    If the order of Differential n is a negative integer.

            Examples
            --------
            >>> # Let `diff` be a Differential object
            >>> # Define a function f
            >>> def f(x):
            ...     return torch.stack([x**2, x**3], dim=-2)
            >>> # Get the Div callable object
            >>> div_f = diff.div(f)
            """
            if n < 0:
                raise ValueError(
                    "Order of Differential n must be a non-negative integer."
                )

            self.__f = f
            self.__n = n
            self.__create_graph = create_graph

        def __call__(
            self, x: torch.Tensor, full_trace: bool = False
        ) -> Tuple[torch.Tensor, ...]:
            """
            Compute the Divergence of a function at a given point.

            In particular, for a function :math:`f: \\mathbb{R}^{a_1,...} -> \\mathbb{R}^{a_1,... x(n-times),b_1,...}`, this method returns :math:`\\text{Div}^n(f)(x)`.

            It is possible to compute full traces of higher-order Divergences by setting the `full_trace` argument to True.

            Parameters
            ----------
            x : torch.Tensor
                A tensor input with shape (N,a1,...) at which to compute the Divergence
            full_trace : bool, optional
                If True, the function returns all intermediate Divergences up to order n. Default is False.

            Returns
            -------
            Tuple[torch.Tensor, ...]
                A tuple of tensors representing the Divergences with shape (N,b1,...).
                If full_trace is True, returns a tuple of tensors representing all intermediate Divergences up to order n.
                If full_trace is False, returns only the n-th order Divergence.

            Examples
            --------
            >>> # Let `div_f` be a Div callable object
            >>> # where f takes a tensor with shape (N,2,3) and returns a tensor with shape (N,4,2,3)
            >>> # Point at which to compute the Divergence
            >>> x = torch.rand(5,2,3)  # N = 5
            >>> # Compute the Divergence at x
            >>> D = div_f(x)  # shape (5,4)
            """
            if self.__n == 0:
                return (self.__f(x),)
            if self.__n == 1:
                div_list: List[torch.Tensor] = []
                N = x.shape[0]
                for i in range(N):
                    div_i: torch.Tensor = div_at(
                        self.__f, x[i], create_graph=self.__create_graph
                    )  # has shape (1,)
                    div_list.append(div_i)
                div: torch.Tensor = torch.stack(div_list, dim=0).squeeze(
                    -1
                )  # has shape (N,b1,...)
                if full_trace:
                    return (self.__f(x), div)  # shapes ((N,a1,...,b1,...), (N,b1,...))
                else:
                    return (div,)  # shape (N,b1,...)
            else:  # n > 1
                div1 = Differential.Div(
                    self.__f,
                    1,
                    create_graph=True,
                )
                # x -> div1(x) is the new callable object
                divf = lambda y: div1(y, full_trace=False)[
                    0
                ]  # takes a tensor with shape (N,a1,...) and returns a tensor with shape (N,b1,...,a1,... x(n-1)-times)
                if full_trace:
                    return (self.__f(x),) + Differential.Div(
                        divf,
                        self.__n - 1,
                        create_graph=self.__create_graph,
                    )(
                        x, full_trace=True
                    )  # has shape ((N,b1,...,a1,... xn-times), (N,b1,...,a1,... x(n-1)-times), ... (N,b1,...))
                return Differential.Div(
                    divf,
                    self.__n - 1,
                    create_graph=self.__create_graph,
                )(
                    x, full_trace=False
                )  # has shape (N,b1,...)

    def __init__(self, n: int, create_graph: bool = False) -> None:
        """
        Initialize the Differential class with the order of Differential.

        This class represent the operator :math:`\\nabla^n` and the user can obtain the Nabla and Div callable objects using the methods `__call__` and `div`, respectively.

        Parameters
        ----------
        n : int
            The order of the Differential to compute.
        create_graph : bool, optional
            If True, the computation graph will be constructed, allowing for higher-order Differentials. Default is False.

        Raises
        ------
            ValueError
                If the order of Differential n is a negative integer.
        """
        if n < 0:
            raise ValueError("Order of Differential n must be a non-negative integer.")

        self.__n = n
        self.__create_graph = create_graph

    def __call__(
        self, f: Callable[[torch.Tensor], torch.Tensor]
    ) -> "Differential.Nabla":
        return Differential.Nabla(
            f,
            self.__n,
            self.__create_graph,
        )

    def div(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "Differential.Div":
        return Differential.Div(
            f,
            self.__n,
            self.__create_graph,
        )
