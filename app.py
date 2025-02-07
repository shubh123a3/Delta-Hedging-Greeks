import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import enum
import streamlit as st
import seaborn as sns

st.title("Delta Hedging of European  Option")
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsGBM(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.zeros([NoOfSteps + 1])
    X[:, 0] = np.log(S_0)
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        X[:, i + 1] = X[:, i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt
        S = np.exp(X)
    paths = {"time": time, "S": S}
    return paths

def BS_Call_Put_Option_Price(CP, S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0))
          * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)
    if CP == OptionType.CALL:
        value = stats.norm.cdf(d1) * S_0 - stats.norm.cdf(d2) * K * np.exp(-r * (T - t))
    elif CP == OptionType.PUT:
        value = stats.norm.cdf(-d2) * K * np.exp(-r * (T - t)) - stats.norm.cdf(-d1) * S_0
    return value


def BS_Delta(CP, S_0, K, sigma, t, T, r):
    # when defining a time-grid it may happen that the last grid point
    # is slightly after the maturity
    if t - T > 10e-20 and T - t < 10e-7:
        t = T
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * \
          (T - t)) / (sigma * np.sqrt(T - t))
    if CP == OptionType.CALL:
        value = stats.norm.cdf(d1)
    elif CP == OptionType.PUT:
        value = stats.norm.cdf(d1) - 1.0
    return value

def GeneratePathsMerton(NoOfPaths, NoOfSteps, S0, T, xiP, muJ, sigmaJ, r, sigma):
    # Create empty matrices for Poisson process and for compensated Poisson process
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    S = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    X[:, 0] = np.log(S0)
    S[:, 0] = S0

    # Expectation E(e^J) for J~N(muJ,sigmaJ^2)
    EeJ = np.exp(muJ + 0.5 * sigmaJ * sigmaJ)
    ZPois = np.random.poisson(xiP * dt, [NoOfPaths, NoOfSteps])
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    J = np.random.normal(muJ, sigmaJ, [NoOfPaths, NoOfSteps])
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        # making sure that samples from normal have mean 0 and variance 1
        X[:, i + 1] = X[:, i] + (r - xiP * (EeJ - 1) - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * Z[:, i] \
                      + J[:, i] * ZPois[:, i]
        time[i + 1] = time[i] + dt

    S = np.exp(X)
    paths = {"time": time, "X": X, "S": S}
    return paths

def BS_Gamma(S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    return stats.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T - t))


def BS_Vega(S_0, K, sigma, t, T, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    return S_0 * stats.norm.pdf(d1) * np.sqrt(T - t)

def GeneratePathsGBMEuler(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    S = np.zeros([NoOfPaths, NoOfSteps + 1])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    S[:, 0] = np.log(S_0)
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    X[:, 0] = np.log(S_0)
    time = np.zeros([NoOfSteps + 1])
    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        X[:, i + 1] = X[:, i] + (r - 0.5 * sigma ** 2.0) * dt + sigma * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt
    paths = {"time": time, "S": np.exp(X)}
    return paths


def EUOptionPriceFromMCPathsGeneralized(CP, S, K, T, r):
    # S is a vector of Monte Carlo samples at T
    result = np.zeros([len(K), 1])
    if CP == OptionType.CALL:
        for (idx, k) in enumerate(K):
            result[idx] = np.exp(-r * T) * np.mean(np.maximum(S - k, 0.0))
    elif CP == OptionType.PUT:
        for (idx, k) in enumerate(K):
            result[idx] = np.exp(-r * T) * np.mean(np.maximum(k - S, 0.0))
    return result


def PathWiseDelta(S0, S, K, sigma, r, T):
    temp1 = S[:, -1] > K

    return np.exp(-r * T) * np.mean(S[:, -1] / S0 * temp1)


def PathwiseVega(S0, S, sigma, K, r, T):
    temp1 = S[:, -1] > K
    temp2 = 1.0 / sigma * S[:, -1] * (np.log(S[:, -1] / S0) - (r + 0.5 * sigma ** 2.0) * T)
    return np.exp(-r * T) * np.mean(temp1 * temp2)


def enhanced_plotting_streamlit(time, S, CallM, DeltaM, PnL, path_id):


    # User selection for path


    sns.set(style="darkgrid")

    # Figure 1: Stock, Call Price, Delta, PnL
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(time, S[path_id, :], label="Stock Price", color="blue", linewidth=2)
    ax.plot(time, CallM[path_id, :], label="Call Option Price", color="green", linestyle="dashed", linewidth=2)
    ax.plot(time, DeltaM[path_id, :], label="Delta Hedging", color="red", linestyle="dotted", linewidth=2)
    ax.plot(time, PnL[path_id, :], label="PnL", color="purple", linestyle="dashdot", linewidth=2)

    # Mark key buy/sell decisions
    buy_sell_points = [i for i in range(1, len(time)) if DeltaM[path_id, i] != DeltaM[path_id, i - 1]]
    ax.scatter([time[i] for i in buy_sell_points], [PnL[path_id, i] for i in buy_sell_points],
               color='black', marker='o', label="Buy/Sell Adjustments", zorder=3)

    ax.axhline(0, color='black', linestyle='--', linewidth=1)  # Zero Profit Line
    ax.set_xlabel("Time (Years)", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title("Stock Price, Call Option, Delta Hedging & PnL", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig)

    # Figure 2: Histogram of P&L
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.hist(PnL[:, -1], bins=50, color="skyblue", edgecolor="black", alpha=0.75)
    ax2.axvline(0, color='red', linestyle='dashed', linewidth=2, label="Break-even")
    ax2.set_xlim([-0.1, 0.1])
    ax2.set_xlabel("Final PnL", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("Distribution of P&L at Expiry", fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.7)

    st.pyplot(fig2)


import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


def display_pnl_info(PnL, S, K, s0, path_id):
    st.title("PnL Information Table")

    st.write(
        f"**Path No:** {path_id}  ",
        f"S0: {s0:0.4f}  ",
        f"PnL(Tm-1): {PnL[path_id, -2]:0.4f}  ",
        f"S(tm): {S[path_id, -1]:0.4f}  ",
        f"max(S(Tm)-K,0): {np.maximum(S[path_id, -1] - K, 0.0):0.4f}  ",
        f"PnL(Tm): {PnL[path_id, -1]:0.4f}"
    )


def plot_convergence(NoOfPathsV, deltaPathWiseV, delta_Exact, vegaPathWiseV, vega_Exact):
    st.subheader("Convergence of Pathwise Greeks")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.plot(NoOfPathsV, deltaPathWiseV, '.-r', label='Pathwise Est')
    ax.plot(NoOfPathsV, delta_Exact * np.ones([len(NoOfPathsV), 1]), label='Exact')
    ax.set_xlabel('Number of Paths')
    ax.set_ylabel('Delta')
    ax.set_title('Convergence of Pathwise Delta w.r.t Number of Paths')
    ax.legend()
    st.pyplot(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.plot(NoOfPathsV, vegaPathWiseV, '.-r', label='Pathwise Est')
    ax.plot(NoOfPathsV, vega_Exact * np.ones([len(NoOfPathsV), 1]), label='Exact')
    ax.set_xlabel('Number of Paths')
    ax.set_ylabel('Vega')
    ax.set_title('Convergence of Pathwise Vega w.r.t Number of Paths')
    ax.legend()
    st.pyplot(fig)


NoOfPaths=st.sidebar.number_input("Number of Paths", min_value=1, max_value=10000, value=1000, step=1)
NoOfSteps=st.sidebar.number_input("Number of Steps", min_value=1, max_value=1000, value=252, step=1)
T=st.sidebar.slider("Time to Maturity", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
r=st.sidebar.slider("Risk-free Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
sigma=st.sidebar.slider("Volatility", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
s0=st.sidebar.number_input("Initial Stock Price", min_value=0.0, max_value=100000.0, value=1.0, step=10.0)
K=st.sidebar.number_input("Strike Price", min_value=0.0, max_value=100000.0, value=0.95, step=10.0)
K=[K]
CP=st.sidebar.radio("Option Type", [OptionType.CALL,OptionType.PUT], index=0)
random_seed=st.sidebar.number_input("Random Seed", min_value=0, max_value=100000, value=7, step=1)
path_id = st.sidebar.slider("Select Path ID", min_value=0, max_value=NoOfPaths - 1, value=13)
# Parameters
model=st.radio("model", ["Black-Scholes", "Black-Scholes with Jumps", "Black-Scholes with PathWise Delta vega"], index=0)

if model=="Black-Scholes":
    st.subheader("Black-Scholes")
    np.random.seed(random_seed)
    Paths = GeneratePathsGBM(NoOfPaths, NoOfSteps, T, r, sigma, s0)
    time = Paths["time"]
    S = Paths["S"]
    # handy lambda function
    C = lambda t, K, S0: BS_Call_Put_Option_Price(CP, S0, K, sigma, t, T, r)
    Delta = lambda t, K, S0: BS_Delta(CP, S0, K, sigma, t, T, r)

    # setting up inital portfolio
    PnL = np.zeros([NoOfPaths, NoOfSteps + 1])
    delta_init = Delta(0.0, K, s0)
    PnL[:, 0] = C(0.0, K, s0) - delta_init * s0
    CallM = np.zeros([NoOfPaths, NoOfSteps + 1])
    CallM[:, 0] = C(0.0, K, s0)
    DeltaM = np.zeros([NoOfPaths, NoOfSteps + 1])
    DeltaM[:, 0] = Delta(0, K, s0)

    for i in range(1, NoOfSteps + 1):
        dt = time[i] - time[i - 1]
        delta_old = Delta(time[i - 1], K, S[:, i - 1])
        delta_curr = Delta(time[i], K, S[:, i])

        PnL[:, i] = PnL[:, i - 1] * np.exp(r * dt) - (delta_curr - delta_old) * S[:, i]
        CallM[:, i] = C(time[i], K, S[:, i])
        DeltaM[:, i] = delta_curr
        # final payment of in the money option
    PnL[:, -1] = PnL[:, -1] - np.maximum(S[:, -1] - K, 0) + DeltaM[:, -1] * S[:, -1]


    enhanced_plotting_streamlit(time, S, CallM, DeltaM, PnL, path_id)

    # Analysis for each path

    st.write(
                    f"**Path ID:** {i:2d}  ",
                    f"PnL(t_0): {PnL[0, 0]:0.4f}  ",
                    f"PnL(Tm-1): {PnL[i, -2]:0.4f}  ",
                    f"S(t_m): {S[i, -1]:0.4f}  ",
                    f"max(S(tm)-K,0): {np.max(S[i, -1] - K, 0):0.4f}  ",
                    f"PnL(t_m): {PnL[i, -1]:0.4f}"
                )
elif model=="Black-Scholes with Jumps":
    st.subheader("Black-Scholes with Jumps")
    xiP=st.sidebar.slider("Jump Intensity", min_value=0.0, max_value=1.0, value=1.0, step=0.01)
    muJ=st.sidebar.slider("Jump Mean", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
    sigmaJ=st.sidebar.slider("Jump Volatility", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    np.random.seed(random_seed)
    Paths = GeneratePathsMerton(NoOfPaths, NoOfSteps, s0, T, xiP, muJ, sigmaJ, r, sigma)
    time = Paths["time"]
    S = Paths["S"]

    # Setting up some handy lambdas
    C = lambda t, K, S0: BS_Call_Put_Option_Price(CP, S0, K, sigma, t, T, r)
    Delta = lambda t, K, S0: BS_Delta(CP, S0, K, sigma, t, T, r)

    # Setting up initial portfolio
    PnL = np.zeros([NoOfPaths, NoOfSteps + 1])
    delta_init = Delta(0.0, K, s0)
    PnL[:, 0] = C(0.0, K, s0) - delta_init * s0

    CallM = np.zeros([NoOfPaths, NoOfSteps + 1])
    CallM[:, 0] = C(0.0, K, s0)
    DeltaM = np.zeros([NoOfPaths, NoOfSteps + 1])
    DeltaM[:, 0] = Delta(0, K, s0)

    for i in range(1, NoOfSteps + 1):
        dt = time[i] - time[i - 1]
        delta_old = Delta(time[i - 1], K, S[:, i - 1])
        delta_curr = Delta(time[i], K, S[:, i])

        PnL[:, i] = PnL[:, i - 1] * np.exp(r * dt) - (delta_curr - delta_old) * S[:, i]  # PnL
        CallM[:, i] = C(time[i], K, S[:, i])
        DeltaM[:, i] = Delta(time[i], K, S[:, i])

    # Final transaction, payment of the option (if in the money) and selling the hedge
    PnL[:, -1] = PnL[:, -1] - np.maximum(S[:, -1] - K, 0) + DeltaM[:, -1] * S[:, -1]

    #plotting
    enhanced_plotting_streamlit(time, S, CallM, DeltaM, PnL, path_id)
    st.write(
        f"**Path No:** {path_id}  ",
        f"S0: {s0:0.4f}  ",
        f"PnL(Tm-1): {PnL[path_id, -2]:0.4f}  ",
        f"S(tm): {S[path_id, -1]:0.4f}  ",
        f"max(S(Tm)-K,0): {np.maximum(S[path_id, -1] - K, 0.0):0.4f}  ",
        f"PnL(Tm): {PnL[path_id, -1]:0.4f}"
    )
elif model=="Black-Scholes with PathWise Delta vega":
    st.subheader("Black-Scholes with PathWise Delta vega")
    S0=s0
    t = 0.0
    k=np.array([S0])
    delta_Exact = BS_Delta(CP, S0, k, sigma, t, T, r)
    vega_Exact = BS_Vega(S0, k, sigma, t, T, r)

    NoOfPathsV = np.round(np.linspace(5, 1000, 50))
    deltaPathWiseV = np.zeros(len(NoOfPathsV))
    vegaPathWiseV = np.zeros(len(NoOfPathsV))

    for (idx, nPaths) in enumerate(NoOfPathsV):
        #st.write('Running simulation with {0} paths'.format(nPaths))
        np.random.seed(random_seed)
        paths1 = GeneratePathsGBMEuler(int(nPaths), NoOfSteps, T, r, sigma, S0)
        S = paths1["S"]
        delta_pathwise = PathWiseDelta(S0, S, k, sigma, r, T)
        deltaPathWiseV[idx] = delta_pathwise

        vega_pathwise = PathwiseVega(S0, S, sigma, k, r, T)
        vegaPathWiseV[idx] = vega_pathwise

    plot_convergence(NoOfPathsV, deltaPathWiseV, delta_Exact, vegaPathWiseV, vega_Exact)












