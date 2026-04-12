import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


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

    def simulate_gbm(
        self, T: float, N: int, seed: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates geometric Brownian motion path.
        Parameters:
            T: total time
            N: number of time steps
            seed: optional seed for reproducibility
        Returns tuple(times, path)
        """
        if seed is not None:
            np.random.seed(seed)

        dt = T / N
        t = np.linspace(0, T, N + 1)
        W = np.random.normal(0, np.sqrt(dt), size=N).cumsum()
        W = np.insert(W, 0, 0.0)  # W(0) = 0
        S = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        return t, S

    def simulate_gbm_final(self, T: float, n_results: int = 1, seed: int | None = None):
        """
        Returns value of geom. Brownian motion after time T, without computing the full path.
        Parameters:
            T: total time
            n_results: optional length of output array
            seed: optional seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        Z = np.random.normal(size=n_results)
        S0 = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )
        return S0


if __name__ == "__main__":
    S0 = 120
    mu = 0.0
    gbm = GeometricBrownianMotion(S0, mu, 0.1)
    times, S_path = gbm.simulate_gbm(1, 250)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, S_path, color="k", label=r"Price Path $S(t)$")
    ax.plot(
        times,
        S0 * np.exp(mu * times),
        linestyle="--",
        color="orange",
        label=r"Drift Path $S_0 e^{\mu t}$",
    )
    ax.hlines(S0, 0, 1, linestyle="-.", color="grey", label=r"Initial Value $S_0$")
    # d = max(S0 - np.min(S_path), np.max(S_path) - S0) * (1 + 0.05)
    # ax.set_ylim(S0 - d, S0 + d)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Asset Price")
    ax.legend(fontsize=12)
    plt.show()
