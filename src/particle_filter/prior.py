from abc import ABC, abstractmethod

from jax.scipy.stats import uniform
from jax.typing import ArrayLike


class Prior(ABC):
    def get_likelihood(self, theta: ArrayLike) -> float:
        """Compute and return the prior likelihood of theta."""
        raise NotImplementedError("Subclasses must implement this method.")


class UniformPrior(Prior):
    def get_likelihood(
        self, theta: ArrayLike, min_dispersion: float | int = 0.5, max_dispersion: float | int = 100
    ) -> float:
        if len(theta) != 1:
            raise ValueError(
                "UniformPrior can only accommodate 1 parameter (dispersion) at the moment. Functionality needs to be expanded to accommodate more parameters.",
                f"Your input was: {theta}"
            )
        logpdf1 = uniform.logpdf(theta[0], loc=min_dispersion, scale=max_dispersion)
        return logpdf1.item()
