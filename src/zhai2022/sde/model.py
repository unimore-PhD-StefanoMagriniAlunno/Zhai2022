import numpy as np
from typing import Callable


class Model:
    """A class representing a stochastic differential equation (SDE) model defined by its drift and diffusion functions, along with an initial state generator and the dimension of the state space.

    SDE:
        :math:`dX_t = dt\\text{drift}(X_t, t) + \\text{diffusion}(X_t, t) dW_t \\quad X_t\\in\\mathbb{R}^{d}`

        where :math:`W_t` is a standard Wiener process in :math:`\\mathbb{R}^{d}`, :math:`\\text{drift}\\in\\mathbb{R}^{d}`, :math:`\\text{diffusion}\\in\\mathbb{R}^{d\\times d}`.
    """

    def __init__(
        self: "Model",
        drift: Callable[[np.ndarray, float], np.ndarray],
        diffusion: Callable[[np.ndarray, float], np.ndarray],
        initial_state: Callable[[int], np.ndarray],
        initial_time: float,
        n_dim: int,
    ) -> None:
        """
        Initializes the Model with drift, diffusion, initial state, and dimension.

        Parameters
        ----------
            self : Model
            drift : Callable[[np.ndarray, float], np.ndarray]
                Function to compute the drift term. It takes the current state (with shape (`n_samples`, `n_dim`)) and time as input and returns the drift (with shape (`n_samples`, `n_dim`)).
            diffusion : Callable[[np.ndarray, float], np.ndarray]
                Function to compute the diffusion term. It takes the current state (with shape (`n_samples`, `n_dim`)) and time as input and returns the diffusion (with shape (`n_samples`, `n_dim`, `n_dim`)).
            initial_state : Callable[[int], np.ndarray]
                Function to generate the initial state given the number of samples.
            initial_time : float
                The initial time of the SDE.
            n_dim : int
                The dimension of the state space.

        Raises
        ------
            ValueError: `n_dim` is not a positive integer.

        Returns
        -------
            None

        Examples
        --------

        >>> # Define drift, diffusion, and initial state functions
        >>> def drift(X, t):
        >>>     return -X
        >>> def diffusion(X, t):
        >>>     return np.tile(np.eye(X.shape[1]), (X.shape[0], 1, 1))
        >>> def initial_state(n_samples):
        >>>     return np.random.randn(n_samples, 2)

        >>> # Create a Model instance
        >>> model = Model(drift, diffusion, initial_state, initial_time=0.0, n_dim=2)
        """

        if n_dim <= 0:
            raise ValueError("n_dim must be a positive integer.")

        self.__drift = drift
        self.__diffusion = diffusion
        self.__initial_state = initial_state
        self.__initial_time = initial_time
        self.__n_dim = n_dim

    def sample_initial_state(self: "Model", n_samples: int) -> np.ndarray:
        """
        Samples the initial state for a given number of samples.

        Parameters
        ----------
            self : Model
            n_samples : int
                Number of samples to generate.

        Raises
        ------
            ValueError: `n_samples` is not a positive integer.

        Returns
        -------
            np.ndarray
                Initial state with shape (n_samples, n_dim).

        Examples
        --------

        >>> # Create a Model instance (assuming drift, diffusion, and initial_state are defined)
        >>> model = Model(drift, diffusion, initial_state, n_dim=2)

        >>> # Sample initial state for 5 samples
        >>> initial_states = model.sample_initial_state(5)
        """

        if n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")

        return self.__initial_state(n_samples)

    @property
    def n_dim(self: "Model") -> int:
        """
        Returns the dimension of the state space.

        Parameters
        ----------
            self : Model

        Returns
        -------
            int
                The dimension of the state space.
        """

        return self.__n_dim

    @property
    def initial_time(self: "Model") -> float:
        """
        Returns the initial time of the SDE.

        Parameters
        ----------
            self : Model

        Returns
        -------
            float
                The initial time.
        """

        return self.__initial_time

    @property
    def drift(self: "Model") -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Returns the drift function.

        Parameters
        ----------
            self : Model

        Returns
        -------
            Callable[[np.ndarray, float], np.ndarray]
                The drift function.
        """

        return self.__drift

    @property
    def diffusion(self: "Model") -> Callable[[np.ndarray, float], np.ndarray]:
        """
        Returns the diffusion function.

        Parameters
        ----------
            self : Model

        Returns
        -------
            Callable[[np.ndarray, float], np.ndarray]
                The diffusion function.
        """

        return self.__diffusion
