from math import sqrt, log, exp
from dataclasses import dataclass
import scipy.stats as ss


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
        omega = 1 if option_type.lower() == "call" else -1
        d1 = (
            1
            / (self.sigma * sqrt(self.T))
            * (log(self.S0 / self.K) + (self.r + self.sigma**2 / 2) * self.T)
        )
        d2 = d1 - self.sigma * sqrt(self.T)
        return float(
            omega
            * (
                self.S0 * ss.norm.cdf(omega * d1)
                - self.K * exp(-self.r * self.T) * ss.norm.cdf(omega * d2)
            )
        )


if __name__ == "__main__":
    omega = 1  # European Call Option
    r = 0.02
    sigma = 0.1
    T = 1  # Year
    K = 120
    S0 = 120
    bsm = BlackScholesMerton(S0, K, T, r, sigma)
    V_call = bsm.price("call")
    V_put = bsm.price("put")
    print(bsm)
    print(f"{V_call=}")
    print(f"{V_put=}")
