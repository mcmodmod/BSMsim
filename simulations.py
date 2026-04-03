import numpy as np
from dataclasses import dataclass


@dataclass
class GeometricBrownianMotion:
    """
    Attributes:
        S0: initial value
        mu: drift coefficient
        sigma: volatility
    """

    S0: float
    mu: float
    sigma: float

    def simulate(self, T: float, N: int, seed: int | None = None) -> np.ndarray:
        """
        Parameters:
            T: total time horizon
            N: number of time steps
            seed: optional seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / N
        t = np.linspace(0, T, N + 1)
        W = np.random.normal(0, np.sqrt(dt), size=N).cumsum()
        W = np.insert(W, 0, 0.0)  # W(0) = 0
        S = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        return S
