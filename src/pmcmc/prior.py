from abc import ABC, abstractmethod

from jax import Array
from jax.scipy.stats import uniform
from jax.typing import ArrayLike


class Prior(ABC):
    def get_likelihood(self, theta: ArrayLike) -> Array:
        """Compute and return the prior likelihood of theta."""
        raise NotImplementedError("Subclasses must implement this method.")


class UniformPrior(Prior):
    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def get_likelihood(self, theta: ArrayLike) -> Array:
        if len(theta) != 1:
            raise ValueError(
                "UniformPrior can only accommodate 1 parameter (dispersion) at the moment. "
                "Functionality needs to be expanded to accommodate more parameters.",
                f"Your input was: {theta}",
            )
        logpdf1 = uniform.logpdf(theta[0], loc=self.min_val, scale=self.max_val)
        return logpdf1
