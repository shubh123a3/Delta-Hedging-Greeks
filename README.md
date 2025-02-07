Below is a very detailed README file in Markdown format that includes theory, mathematical derivations in LaTeX, and a full description of the project.

---

```markdown
# Delta-Hedging-Greeks

Delta-Hedging-Greeks is a Streamlit application that demonstrates delta hedging strategies and computes option Greeks using both analytical formulas (via the Black-Scholes model and Merton jump-diffusion) and pathwise sensitivity methods. The app is modular, with each component organized into separate files and then combined into one interactive interface.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Theoretical Background](#theoretical-background)
  - [Black-Scholes Model](#black-scholes-model)
  - [Delta Hedging](#delta-hedging)
  - [Jump-Diffusion Models](#jump-diffusion-models)
  - [Pathwise Sensitivity for Greeks](#pathwise-sensitivity-for-greeks)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Detailed Explanation of Files](#detailed-explanation-of-files)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project provides an interactive Streamlit app for simulating delta hedging strategies using both the classical Black-Scholes framework and jump-diffusion models. In addition, it computes option Greeks (Delta, Gamma, Vega) via analytical methods and pathwise sensitivity using Monte Carlo simulation.

---

## Features

- **Interactive Simulations:** Visualize asset paths, option prices, and hedging performance in real time.
- **Black-Scholes Model:** Price European options and compute Greeks analytically.
- **Jump-Diffusion Model:** Incorporate jumps into asset dynamics using the Merton model.
- **Pathwise Sensitivity:** Estimate Delta and Vega from simulated paths.
- **Modular Codebase:** Organized into separate files for models, sensitivity analysis, and utilities.
- **Streamlit Interface:** User-friendly dashboard for exploring different scenarios.

---

## Theoretical Background

### Black-Scholes Model

The Black-Scholes model assumes that the underlying asset follows a Geometric Brownian Motion (GBM):

$$
dS_t = r S_t\, dt + \sigma S_t\, dW_t,
$$

where:
- \( S_t \) is the asset price at time \( t \),
- \( r \) is the risk-free rate,
- \( \sigma \) is the volatility,
- \( dW_t \) is the increment of a Wiener process.

The solution to the GBM is given by:

$$
S_T = S_0 \exp\left(\left(r - \frac{1}{2}\sigma^2\right)T + \sigma W_T\right).
$$

The Black-Scholes formula for a European call option is:

$$
C = S_0 N(d_1) - K e^{-rT} N(d_2),
$$

with

$$
d_1 = \frac{\ln\left(\frac{S_0}{K}\right) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T},
$$

and where \( N(\cdot) \) is the cumulative distribution function of the standard normal distribution.

### Delta Hedging

Delta hedging involves offsetting the directional risk of an option by trading the underlying asset. The **Delta** of an option is defined as:

$$
\Delta = \frac{\partial C}{\partial S_0}.
$$

For a call option in the Black-Scholes model, this simplifies to:

$$
\Delta_{\text{call}} = N(d_1).
$$

This value represents the sensitivity of the option price to a small change in the underlying asset price.

### Jump-Diffusion Models

The Merton jump-diffusion model extends GBM by incorporating random jumps:

$$
dS_t = S_t \left( r\, dt + \sigma\, dW_t + \left(e^J - 1\right)dN_t \right),
$$

where:
- \( N_t \) is a Poisson process with intensity \( \xi \),
- \( J \) is a normally distributed jump size with mean \( \mu_J \) and variance \( \sigma_J^2 \).

The terminal asset price becomes:

$$
S_T = S_0 \exp\left[\left(r - \xi\left(e^{\mu_J + \frac{1}{2}\sigma_J^2} - 1\right) - \frac{1}{2}\sigma^2\right)T + \sigma W_T + \sum_{i=1}^{N_T} J_i\right].
$$

### Pathwise Sensitivity for Greeks

Pathwise methods estimate the Greeks by differentiating the payoff function along each simulated path.

- **Pathwise Delta:**

  $$ 
  \Delta \approx e^{-rT} \frac{1}{M} \sum_{i=1}^{M} \frac{S_T^{(i)}}{S_0} \cdot \mathbf{1}_{\{S_T^{(i)} > K\}},
  $$

  where \( \mathbf{1}_{\{S_T^{(i)} > K\}} \) is an indicator function that is 1 if \( S_T^{(i)} > K \) and 0 otherwise.

- **Pathwise Vega:**

  $$
  \text{Vega} \approx e^{-rT} \frac{1}{M} \sum_{i=1}^{M} \mathbf{1}_{\{S_T^{(i)} > K\}} \cdot \frac{S_T^{(i)}}{\sigma} \left( \ln\left(\frac{S_T^{(i)}}{S_0}\right) - \left(r + \frac{1}{2}\sigma^2\right)T \right).
  $$

These estimators help verify the analytical Greek values and assess the sensitivity of option prices to underlying parameters.

---

## Project Structure

```
Delta-Hedging-Greeks/
├── app.py                    # Main Streamlit application
├── bs_model.py               # Black-Scholes pricing and Greeks
├── jump_model.py             # Merton jump-diffusion model implementation
├── pathwise_greeks.py        # Pathwise sensitivity estimation for Greeks
├── utils.py                  # Utility functions for simulation and plotting
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/shubh123a3/Delta-Hedging-Greeks.git
   cd Delta-Hedging-Greeks
   ```

2. **Install Dependencies**

   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

To run the Streamlit app, execute:

```bash
streamlit run app.py
```

This command will open a new browser window (or tab) with the interactive interface. You can adjust parameters, simulate asset paths, and observe the delta hedging strategy and Greek calculations in real time.

---

## Detailed Explanation of Files

### `app.py`
- **Purpose:** The main entry point for the Streamlit application.
- **Functionality:** Combines modules for Black-Scholes pricing, jump-diffusion simulation, and pathwise Greek estimation into one interactive dashboard.
- **User Interaction:** Provides sliders and input fields to modify model parameters and view real-time results.

### `bs_model.py`
- **Purpose:** Contains functions for pricing European options using the Black-Scholes model.
- **Key Functions:**
  - **`BS_Call_Put_Option_Price`**: Implements the Black-Scholes pricing formula.
  - **`BS_Delta`**, **`BS_Gamma`**, **`BS_Vega`**: Compute the analytical Greeks.

### `jump_model.py`
- **Purpose:** Implements the Merton jump-diffusion model.
- **Key Functions:**
  - Simulate asset paths with jumps.
  - Price options under jump conditions.

### `pathwise_greeks.py`
- **Purpose:** Provides methods for estimating Greeks using pathwise sensitivity.
- **Key Functions:**
  - **`PathwiseDelta`**: Estimates delta from simulated paths.
  - **`PathwiseVega`**: Estimates vega from simulated paths.

### `utils.py`
- **Purpose:** Utility functions for simulation, data manipulation, and plotting.
- **Usage:** Shared among the other modules to keep the code DRY (Don't Repeat Yourself).

---

## Contributing

Contributions are welcome! If you have suggestions or bug fixes, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

Special thanks to all contributors in the quantitative finance community for their invaluable resources and discussions, which helped shape this project.

---

Happy hedging and exploring Greeks! 🚀
```

---

Feel free to adjust or expand this README further as your project evolves. This document provides a comprehensive guide to the theory, mathematics, code structure, and usage instructions, ensuring that both users and fellow developers understand the purpose and inner workings of the project.
