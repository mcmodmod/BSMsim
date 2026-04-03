import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from simulations import GeometricBrownianMotion
from blackscholes import BlackScholesMerton

plt.rcParams.update(
    {
        "text.usetex": True,
    }
)


def simulate_price_path(N, S0, mu, sigma, T=1.0):
    # Simulate One GBM Path
    gbm = GeometricBrownianMotion(S0, mu, sigma)
    times = np.linspace(0, T, N + 1)
    S_path = gbm.simulate(T, N)
    ST = S_path[-1]
    return times, S_path, ST


def main():
    # --- Streamlit App ---
    st.set_page_config(page_title="GBM & BSM Simulator", layout="centered")
    st.title("Geometric Brownian Motion")

    st.write(
        r"In the following, a price path is simulated from some initial price $S_0\equiv S(t=0)$ using geometric Brownian motion"
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
        r"Initial Stock Price $S_0$", value=120.0, min_value=0.01, step=1.0
    )
    N = right.number_input(r"Number of Steps $N$", value=250, min_value=10, step=10)
    mu = right.number_input(r"Drift $\mu$", value=0.05, step=0.01, format="%.3f")
    sigma = left.number_input(
        r"Volatility $\sigma$", value=0.1, min_value=0.005, step=0.005, format="%.3f"
    )

    # Compute Black-Scholes Call Price
    times, S_path, ST = simulate_price_path(N, S0, mu, sigma)
    if st.button("Regenerate"):
        times, S_path, ST = simulate_price_path(N, S0, mu, sigma)
    st.header("Black-Scholes Parameters")

    # Display Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, S_path, color="k", label=r"Price Path $S(t)$")
    ax.plot(
        times,
        S0 * np.exp(mu * times),
        linestyle="--",
        color="orange",
        label=r"Drift $S_0 e^{\mu t}$",
    )
    ax.hlines(S0, 0, 1, linestyle="-.", color="grey", label=r"Initial Value $S_0$")
    d = max(S0 - np.min(S_path), np.max(S_path) - S0) * (1 + 0.05)
    ax.set_ylim(S0 - d, S0 + d)
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Stock Price")
    ax.legend(fontsize=12)
    st.pyplot(fig)

    # -----------------------------------------------------

    st.title("Black-Scholes-Merton Simulator")
    st.write(r"""
        Next, a European Put or Call option with the above quantity as its underlying asset is constructed using the user inputs below together with the volatility $\sigma$ from above.
        Its appropriate price is calculated using the Black-Scholes-Merton (BSM) equation
        """)
    st.latex(r"""
    V(\omega; T, K, \sigma, r) = \omega\left(S_0 \Phi_{0,1}(\omega d_1) - S_0 e^{-r T} \Phi_{0,1}(\omega d_2)\right)
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
    options = ["Put", "Call"]
    selection = st.pills("Option Type", options, default="Put")
    T = st.number_input(
        r"Time to Maturity $T$ (years)", value=1.0, min_value=0.01, step=0.1
    )
    K = st.number_input(r"Strike Price $K$", value=120.0, min_value=0.01, step=1.0)
    r = st.number_input(r"Risk-free Rate $r$", value=0.02, step=0.01, format="%.4f")
    times, S_path, ST = simulate_price_path(N, S0, mu, sigma, T)
    bsm = BlackScholesMerton(S0, K, T, r, sigma)
    if selection == "Put":
        price = bsm.price("put")
    elif selection == "Call":
        price = bsm.price("call")
    else:
        st.write("Please select either 'Put' or 'Call' above. Falling back to 'Call'.")
        price = bsm.price("call")
    st.write(f"The appropriate price is {price:.2f}")
    cash_flow = max(ST - K, 0)
    profit = cash_flow - price

    # Display Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    # Profit coloring
    if profit > 0:
        ax.vlines(T, K + price, ST, color="limegreen", linewidth=2, label="Profit")
    else:
        ax.vlines(T, ST, K + price, color="red", linewidth=2, label="Loss")

    ax.plot(times, S_path, color="k", label="Price Path")
    # d = max(S0 - np.min(S_path), np.max(S_path) - S0) * (1 + 0.05)
    # ax.set_ylim(S0 - d, S0 + d)
    ax.hlines(K, 0, T, linestyle="--", color="grey", label=r"Strike Price $K$")
    ax.hlines(K + price, 0, T, linestyle="--", color="orange", label="Strike + Premium")
    ax.set_xlabel("Time (Years)")
    ax.set_ylabel("Stock Price")
    ax.legend(fontsize=12)
    st.pyplot(fig)
    #
    # # Display Profits
    # st.subheader("Profit Analysis")
    # st.write(f"Profit from this simulation: **{profit:.2f}**")

    # Monte Carlo Average Profit
    # if st.button("Run Monte Carlo (1000 simulations)"):
    #     profits = []
    #     for _ in range(1000):
    #         S_sim = gbm.simulate(T, N)
    #         ST_sim = S_sim[-1]
    #         cash_flow_sim = max(ST_sim - K, 0)
    #         profits.append(cash_flow_sim - price)
    #     avg_profit = np.mean(profits)
    #     st.write(f"Average profit over 1000 simulations: **{avg_profit:.2f}**")


if __name__ == "__main__":
    main()
