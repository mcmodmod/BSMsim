import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from simulations import GeometricBrownianMotion
from blackscholes import BlackScholesMerton

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)


def main():
    # --- Streamlit App ---
    st.set_page_config(page_title="GBM & BSM Simulator", layout="centered")
    st.title("Black-Scholes-Merton Simulator")
    st.subheader("Geometric Brownian Motion")

    st.write(
        r"In the following, a price path of an asset is simulated for one year from some initial price $S_0\equiv S(t=0)$ using geometric Brownian motion"
    )
    st.latex(
        r"S(t) = S_0\exp\left( \left(\mu - \frac{\sigma^2}{2} \right)t + \sigma W(t)\right),"
    )
    st.write(r"""
        where $W(t)$ is a Wiener process, and the drift $\mu$ and volatility $\sigma$ can be adjusted using the fields below.
        Since this process is inherently stochastic, a different price path will be generated each time.
        A new path can be generated either by changing the input parameters or by pressing the 'Regenerate' button below.
        """)

    left, right = st.columns(2)
    S0 = left.number_input(
        r"Initial Asset Price $S_0$", value=120.0, min_value=0.01, step=1.0
    )
    N = right.number_input(r"Number of Steps $N$", value=500, min_value=10, step=10)
    mu = right.number_input(r"Drift $\mu$", value=0.0, step=0.01, format="%.3f")
    sigma = left.number_input(
        r"Volatility $\sigma$", value=0.1, min_value=0.005, step=0.005, format="%.3f"
    )
    T = 1.0
    gbm = GeometricBrownianMotion(S0, mu, sigma)

    # Generate Price Path
    times, S_path = gbm.simulate_gbm(T, N)
    if st.button("Regenerate"):
        times, S_path = gbm.simulate_gbm(T, N)
    # Display Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, S_path, color="k", label=r"Price Path $S(t)$")
    ax.plot(
        times,
        S0 * np.exp(mu * times),
        linestyle="--",
        color="blue",
        label=r"Drift Path $S_0 e^{\mu t}$",
    )
    ax.hlines(
        S0, 0, 1, linestyle="-.", color="grey", alpha=0.8, label=r"Initial Value $S_0$"
    )
    # d = max(S0 - np.min(S_path), np.max(S_path) - S0) * (1 + 0.05)
    # ax.set_ylim(S0 - d, S0 + d)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Asset Price")
    ax.legend(fontsize=12)
    st.pyplot(fig)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    st.subheader("Black-Scholes-Merton Pricing")
    st.write(r"""
        Next, a European Put or Call option with the above quantity as its underlying asset is constructed using the user inputs below together with the volatility $\sigma$ from above.
        Its appropriate price is calculated using the Black-Scholes-Merton (BSM) equation
        """)
    st.latex(r"""
    V(\omega; T, K, \sigma, r) = \omega\left(S_0 \Phi_{0,1}(\omega d_1) - K e^{-r T} \Phi_{0,1}(\omega d_2)\right)
    """)
    st.write("with")
    st.latex(r"""
    d_1 = \frac{1}{\sigma\sqrt{T}} \left[\ln\left(S_0/K\right) + \left(r-q+\sigma^2/2\right)T\right],\qquad d_2 = d_1 - \sigma \sqrt{T},
    """)
    st.write(r"""
        where $T$ is the time to maturity, $K$ is the strike price, $r$ is the risk-free interest rate, and $\Phi_{0,1}$ is the cumulative distribution function of a standard normal distribution.
        The parameter $\omega=\pm 1$ serves to distinguish Put ($\omega=-1$) from Call ($\omega=+1$) options. 
        This way the two BSM equation can be combined into one.
        """)
    options = ["European Call", "European Put"]
    option_type = st.selectbox("Option Type", options)
    is_call = option_type == "European Call"
    T = st.number_input(
        r"Time to Maturity $T$ (years)", value=1.0, min_value=0.01, step=0.1
    )
    K = st.number_input(r"Strike Price $K$", value=120.0, min_value=0.01, step=1.0)
    r = st.number_input(r"Risk-free Rate $r$", value=0.02, step=0.01, format="%.4f")
    times, S_path = gbm.simulate_gbm(T, N)
    ST = float(S_path[-1])
    bsm = BlackScholesMerton(S0, K, T, r, sigma)
    if is_call:
        cash_flow = max(ST - K, 0)
    else:
        cash_flow = max(K - ST, 0)
    price = bsm.price("call") if is_call else bsm.price("put")
    discounted_profit = np.exp(-r * T) * cash_flow - price
    st.write(f"The appropriate price for this option is **{price:.2f}**")
    if st.button("Generate New Price Path"):
        times, S_path = gbm.simulate_gbm(T, N)
        ST = float(S_path[-1])
        if is_call:
            cash_flow = max(ST - K, 0)
        else:
            cash_flow = max(K - ST, 0)
        price = bsm.price("call") if is_call else bsm.price("put")
        discounted_profit = np.exp(-r * T) * cash_flow - price
    st.write(f"Discounted profit from this simulation: **{discounted_profit:.2f}**")

    # Display Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if discounted_profit > 0:
        ax.vlines(T, K + price, ST, color="limegreen", linewidth=2, label="Profit")
    else:
        ax.vlines(T, ST, K + price, color="red", linewidth=2, label="Loss")

    ax.plot(times, S_path, color="k", label="Price Path")
    # d = max(S0 - np.min(S_path), np.max(S_path) - S0) * (1 + 0.05)
    # ax.set_ylim(S0 - d, S0 + d)
    ax.hlines(K, 0, T, linestyle="--", color="grey", label=r"Strike Price $K$")
    ax.hlines(K + price, 0, T, linestyle="--", color="blue", label="Strike + BSM Price")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Asset Price")
    ax.legend(fontsize=10)
    st.pyplot(fig)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    st.subheader("Small Monte-Carlo Simulation")
    no_of_simulations = st.number_input(
        r"Number of Simulations", value=100_000, min_value=100, step=100
    )
    st.write(
        f"Press the button below to simulate buying this option {no_of_simulations} times at the BSM-optimal price to approximate the average profit."
    )
    st.write(
        r"""**Important note**: From now on the underlying asset is assumed to have a drift equal to the risk-free rate: $\mu=r$ (risk-neutral pricing).
        This way, the average discounted profit should be zero.
        """
    )
    mu = r
    # if st.button("Run Monte-Carlo"):
    gbm = GeometricBrownianMotion(S0, mu, sigma)
    STs = gbm.simulate_gbm_final(T, no_of_simulations)
    if is_call:
        cash_flows = np.maximum(STs - K, 0)
    else:
        cash_flows = np.maximum(K - STs, 0)
    discounted_profits = np.exp(-r * T) * cash_flows - price
    avg_profit = np.mean(discounted_profits)
    st.write(
        f"Average discounted profit over {no_of_simulations} simulations: **{avg_profit:.3f}**"
    )

    # Convergence
    st.write(
        "The convergence to zero can be checked by plotting the absolute deviation from 0 as below."
    )
    fig, ax = plt.subplots()
    cummean = np.cumsum(discounted_profits) / np.arange(1, no_of_simulations + 1)
    simulations = np.arange(1, no_of_simulations + 1)
    ax.plot(
        simulations, np.abs(cummean), color="k", label="Absolute value of mean profit"
    )
    ax.plot(
        simulations,
        1 / np.sqrt(simulations),
        linestyle="dashdot",
        color="blue",
        label=r"Expected Convergence $1/\sqrt{N}$",
    )
    ax.set_xlabel("Number of Simulations N")
    ax.set_ylabel("Error")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()
    st.pyplot(fig)
    st.write(
        "As expected, the mean converges with the inverse square-root of the number of simulations."
    )


if __name__ == "__main__":
    main()
