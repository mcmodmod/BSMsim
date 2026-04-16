import numpy as np
import scipy.stats as ss
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

    def simulate(
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
        dt = T / N
        if seed is not None:
            rng = np.random.default_rng(seed)
            dW = rng.normal(0, np.sqrt(dt), size=N)
        else:
            dW = np.random.normal(0, np.sqrt(dt), size=N)
        W = np.concatenate(([0.0], np.cumsum(dW)))
        t = np.linspace(0, T, N + 1)
        S = self.S0 * np.exp((self.mu - 0.5 * self.sigma**2) * t + self.sigma * W)
        return t, S

    def simulate_terminal(self, T: float, n_results: int = 1, seed: int | None = None):
        """
        Returns value of geom. Brownian motion after time T, without computing the full path.
        Parameters:
            T: total time
            n_results: optional length of output array
            seed: optional seed for reproducibility
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
            Z = rng.normal(size=n_results)
        else:
            Z = np.random.normal(size=n_results)
        ST = self.S0 * np.exp(
            (self.mu - 0.5 * self.sigma**2) * T + self.sigma * np.sqrt(T) * Z
        )
        return ST


@dataclass
class BlackScholesMerton:
    """
    Attributes:
        S0: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to maturity (in years)
        r: Risk-free interest rate
        sigma: Volatility of the underlying asset
    """

    S0: float
    K: float
    T: float
    r: float
    sigma: float

    def price(self, option_type: str = "call") -> float:
        """
        Parameters:
            option_type: "call" for Call option, "put" for Put option
        """
        option_type = option_type.lower()
        if option_type == "call":
            omega = 1
        elif option_type == "put":
            omega = -1
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        d1 = (
            1
            / (self.sigma * np.sqrt(self.T))
            * (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2) * self.T)
        )
        d2 = d1 - self.sigma * np.sqrt(self.T)
        return float(
            omega
            * (
                self.S0 * ss.norm.cdf(omega * d1)
                - self.K * np.exp(-self.r * self.T) * ss.norm.cdf(omega * d2)
            )
        )


if __name__ == "__main__":
    # GBM Testing
    S0 = 120
    mu = 0.0
    sigma = 0.1
    T = 1
    N = 10
    gbm = GeometricBrownianMotion(S0, mu, sigma)
    times, S_path = gbm.simulate(T, N)
    ST = gbm.simulate_terminal(T)

    # BSM Testing
    omega = 1  # European Call Option
    r = 0.02
    sigma = 0.1
    T = 1
    K = 120
    S0 = 120
    bsm = BlackScholesMerton(S0, K, T, r, sigma)
    V_call = bsm.price("call")
    V_put = bsm.price("put")
    print(bsm)
    print(f"{V_call=}")
    print(f"{V_put=}")
