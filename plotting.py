import matplotlib.pyplot as plt
import numpy as np
from simulations import GeometricBrownianMotion
from blackscholes import BlackScholesMerton

if __name__ == "__main__":
    # plt.style.use("classic")
    plt.rcParams.update(
        {
            "text.usetex": True,
            "axes.labelsize": 20,
            # "axes.grid": True,
            "legend.fontsize": 13,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.constrained_layout.use": True,
        }
    )

    r = 0.02
    sigma = 0.1
    T = 1  # Year
    K = 120
    S0 = 120
    mu = 0.05

    bsm = BlackScholesMerton(S0, K, T, r, sigma)
    price = bsm.price("call")
    # print(f"{price=}")

    gbm = GeometricBrownianMotion(S0, mu, sigma)
    N = 500
    t = np.linspace(0, T, N + 1)

    S = gbm.simulate(T, N)
    ST = S[-1]
    cash_flow = max(ST - K, 0)
    profit = cash_flow - price
    print(f"{profit=}")

    plt.figure(figsize=(8, 6))
    if profit > 0:
        plt.vlines(1, K + price, ST, color="limegreen", linewidth=2)
    else:
        plt.vlines(1, ST, K + price, color="red", linewidth=2)
    plt.plot(t, S, color="k", label="Price Path")
    plt.hlines(K, 0, 1, linestyle="--", color="grey", label=r"Strike Price $K$")
    plt.hlines(
        K + price, 0, 1, linestyle="--", color="orange", label="Strike Price + Premium"
    )
    plt.legend()
    plt.savefig("./test.pdf")

    profits = []
    for i in range(1000):
        S = gbm.simulate(T, N)
        ST = S[-1]
        cash_flow = max(ST - K, 0)
        profits.append(cash_flow - price)

    print(f"{np.average(profits)=}")
