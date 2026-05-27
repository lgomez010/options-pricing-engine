# Options Pricing Engine

A multi-model derivatives pricing library implementing Black-Scholes, binomial trees, finite-difference PDE solvers, and Monte Carlo methods — with extensions to stochastic volatility (Heston) and rough volatility (rough Bergomi). Built to compare analytical, numerical, and simulation-based approaches on equal footing.

## Key Results

<!-- TODO: Embed convergence plots and pricing comparison figures after implementation -->

| Method | European Call Price | Runtime (ms) | Error vs. Analytical |
|--------|-------------------|-------------|---------------------|
| Black-Scholes (analytical) | — | — | — |
| Binomial tree (N=1000) | — | — | — |
| Crank-Nicolson PDE | — | — | — |
| Monte Carlo (100k paths) | — | — | — |
| Heston (FFT) | — | — | — |

## Mathematical Background

### Risk-Neutral Valuation
The price of a European contingent claim with payoff $h(S_T)$ is the discounted expectation under the risk-neutral measure $\mathbb{Q}$:

$$V_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}[h(S_T)]$$

where $\mathbb{Q}$ is the unique equivalent martingale measure under which discounted asset prices are martingales. The existence of $\mathbb{Q}$ is equivalent to no-arbitrage (First Fundamental Theorem); its uniqueness corresponds to market completeness (Second Fundamental Theorem).

### Models Implemented
- **Black-Scholes**: Constant volatility; analytical solution via the heat equation.
- **Binomial trees (CRR)**: Discrete-time approximation converging to Black-Scholes in the continuum limit.
- **Finite differences (Crank-Nicolson)**: Direct numerical solution of the Black-Scholes PDE; handles American exercise via penalty method.
- **Monte Carlo**: Simulation of the risk-neutral SDE with variance reduction (antithetic variates, control variates).
- **Heston**: Stochastic volatility with mean-reverting variance process; priced via characteristic function and Fourier inversion.
- **Rough Bergomi**: Fractional volatility driven by fractional Brownian motion with $H < \tfrac{1}{2}$; priced via Monte Carlo.

## Quickstart

```bash
git clone https://github.com/lgomez010/options-pricing-engine.git
cd options-pricing-engine
pip install -e ".[dev,notebooks]"
pytest                           # run tests
python -m src.models.black_scholes  # quick demo
```

## Project Structure

```
options-pricing-engine/
├── src/
│   ├── models/              # Pricing models (BS, Heston, rough Bergomi)
│   │   ├── black_scholes.py
│   │   ├── heston.py
│   │   └── rough_bergomi.py
│   ├── engines/             # Numerical methods (MC, PDE, trees)
│   │   ├── monte_carlo.py
│   │   ├── pde_solver.py
│   │   └── binomial_tree.py
│   ├── greeks/              # Analytical & numerical Greeks
│   │   ├── analytical.py
│   │   └── numerical.py
│   └── utils/               # Shared helpers (payoffs, plotting)
│       └── payoffs.py
├── tests/                   # pytest suite
│   ├── test_black_scholes.py
│   ├── test_put_call_parity.py
│   └── test_convergence.py
├── notebooks/               # Exploration & presentation
│   └── comparison.ipynb
├── data/                    # Market data (see data/README.md)
├── configs/                 # Parameter configurations
├── docs/                    # Additional documentation
├── pyproject.toml
└── README.md
```

## Extensions & Limitations

### What this project demonstrates
- Rigorous comparison of analytical, numerical, and simulation approaches on identical test cases.
- Greeks computed via three methods: analytical (BS), finite-difference, and pathwise Monte Carlo.
- Convergence analysis showing each method's error-vs-compute tradeoff.

### Known limitations
- Rough Bergomi implementation uses naive Cholesky simulation of fBm, which is $O(N^3)$; hybrid schemes (e.g., Bayer-Friz-Gatheral) would improve scaling.
- No jump-diffusion models (Merton, Kou) — a natural next step.
- American option pricing via PDE only; least-squares Monte Carlo (Longstaff-Schwartz) would extend the MC engine.

### Connection to broader portfolio
This engine provides the pricing foundation for the [`volatility-surface-lab`](https://github.com/lgomez010/volatility-surface-lab) project, where these models are calibrated to market-observed implied volatility surfaces.

## License

MIT
