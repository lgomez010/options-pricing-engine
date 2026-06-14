import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.insert(0, '.')

from src.models.black_scholes import call_price, put_price
from src.models.gbm import GBMModel
from src.payoffs.european import EuropeanCallPayoff, EuropeanPutPayoff
from src.engines.pde_solver import PDESolver
from src.engines.binomial_tree import price_option as tree_price
from src.engines.monte_carlo import MonteCarloEngine


def plot_convergence():
    """plot 1: loglog convergence of PDE error vs grid spacing"""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bs = put_price(S0, K, T, r, sigma)

    n_space_values = [50, 100, 200, 400, 800]
    dxs =[]
    errors = []

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanPutPayoff(K, T)

    for n_space in n_space_values:
        solver = PDESolver(model, payoff, n_space=n_space, n_time=5000)
        pde = solver.price(S0, T)
        dxs.append(solver.dx)
        errors.append(abs(pde - bs))

    #fit slope
    log_dx = np.log(dxs)
    log_err = np.log(errors)
    slope, _ = np.polyfit(log_dx, log_err, 1)

    #reference line: error = C * dx^2, anchored at first data point
    C = errors[0] / dxs[0]**2
    dx_ref = np.linspace(dxs[0], dxs[-1], 100)
    err_ref = C * dx_ref**2

    #plot
    plt.figure(figsize=(8,5))
    plt.loglog(dxs, errors, 'o-', label=f'PDE error (slope = {slope:.2f})')
    plt.loglog(dx_ref, err_ref, '--', color='gray', label='O(dx^2) reference')
    plt.xlabel('dx (log scale)')
    plt.ylabel('Absolute error (log scale)')
    plt.title('Crank-Nicolson Spatial Convergence')
    plt.legend()
    plt.tight_layout()
    plt.savefig('notebooks/convergence_plot.png', dpi=150)
    plt.show()
    plt.close()
    print(f"Convergence slope: {slope:.2f}")


def plot_engine_comparison():
    """Plot 2: price and runtime comparison across all engines"""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanPutPayoff(K, T)

    results = {}

    #Black-Scholes (exact)
    t0 = time.time()
    bs = put_price(S0, K, T, r, sigma)
    results['Black-Scholes'] = (bs, time.time()- t0)

    #Binomial Tree
    t0 = time.time()
    tree = tree_price(S0, K, T, r, sigma, N=1000, option_type="put")
    results['Binomial Tree'] = (tree, time.time() - t0)

    #Monte Carlo
    np.random.seed(42)
    t0 = time.time()
    mc_engine = MonteCarloEngine(model, payoff, r, T)
    mc_result = mc_engine.price(n_paths=100_000, antithetic=True, control_variate=True)
    results['Monte Carlo'] = (mc_result['price'], time.time() - t0)

    #PDE solver
    t0 = time.time()
    pde_engine = PDESolver(model, payoff, n_space=400, n_time=400)
    pde = pde_engine.price(S0, T)
    results['PDE (C-N)'] = (pde, time.time() - t0)

    #print table
    print(f"\n{'Method':<20} {'Price':>10} {'Error':>12} {'Time (ms)':>10}")
    print("-" * 55)
    for name, (price, elapsed) in results.items():
        error = abs(price - bs)
        print(f"{name:<20} {price:>10.6f} {error:>12.6f} {elapsed*1000:>10.2f}")

    # Bar chart of errors (exclude BS since its error is 0)
    methods = [m for m in results if m != 'Black-Scholes']
    errors = [abs(results[m][0] - bs) for m in methods]
    times = [results[m][1] * 1000 for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(methods, errors)
    ax1.set_ylabel('Absolute error vs Black-Scholes')
    ax1.set_title('Pricing Accuracy')
    ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-3, -3))

    ax2.bar(methods, times)
    ax2.set_ylabel('Runtime (ms)')
    ax2.set_title('Computation Time')

    plt.tight_layout()
    plt.savefig('notebooks/engine_comparison.png', dpi=150)
    plt.show()
    plt.close()


def plot_exercise_boundary():
    """plot 4: American put early exercise boundary S*(t)"""

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    model = GBMModel(S0, r, sigma, T)
    put = EuropeanPutPayoff(K, T)
    solver = PDESolver(model, put, n_space=400, n_time=400)

    price = solver.price(S0, T, american=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(solver.boundary_times, solver.exercise_boundary, 'b-', linewidth=2)
    ax.axhline(y=K, color='gray', linestyle='--', alpha=0.5, label=f'Strike K = {K}')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Stock Price S*(t)')
    ax.set_title('American Put Early Exercise Boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    print(f"American put price: {price:.4f}")
    print(f"Boundary at t=0: S* = {solver.exercise_boundary[-1]:.2f}")
    print(f"Boundary at t=T: S* = {solver.exercise_boundary[0]:.2f}")
    
    plt.savefig('notebooks/exercise_boundary.png', dpi=150)
    plt.show()
    plt.close()

def plot_pde_greeks():
    """Plot 5: Delta and Gamma extracted from PDE grid"""
    from src.models.black_scholes import call_price

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K, T)
    solver = PDESolver(model, payoff, n_space=400, n_time=400)
    solver.price(S0, T)

    # extract grid data
    S = solver.S
    u = solver.u
    dx = solver.dx

    # derivatives in x via central differences (interior points only)
    du_dx = (u[2:] - u[:-2]) / (2 * dx)
    d2u_dx2 = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    S_int = S[1:-1]  # interior grid points

    # convert to S-derivatives via chain rule
    delta_pde = du_dx / S_int
    gamma_pde = (d2u_dx2 - du_dx) / S_int**2

    # analytical Black-Scholes Greeks for comparison
    from scipy.stats import norm
    d1 = (np.log(S_int / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta_bs = norm.cdf(d1)
    gamma_bs = norm.pdf(d1) / (S_int * sigma * np.sqrt(T))

    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # focus on a reasonable range around the strike
    mask = (S_int > 60) & (S_int < 140)

    ax1.plot(S_int[mask], delta_pde[mask], 'b-', linewidth=2, label='PDE')
    ax1.plot(S_int[mask], delta_bs[mask], 'r--', linewidth=1.5, label='Analytical BS')
    ax1.set_xlabel('Stock Price S')
    ax1.set_ylabel('Delta')
    ax1.set_title('Delta from PDE Grid')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(S_int[mask], gamma_pde[mask], 'b-', linewidth=2, label='PDE')
    ax2.plot(S_int[mask], gamma_bs[mask], 'r--', linewidth=1.5, label='Analytical BS')
    ax2.set_xlabel('Stock Price S')
    ax2.set_ylabel('Gamma')
    ax2.set_title('Gamma from PDE Grid')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('notebooks/pde_greeks.png', dpi=150)
    plt.show()
    plt.close()

if __name__ == "__main__":
    plot_convergence()
    plot_engine_comparison()
    plot_exercise_boundary()
    plot_pde_greeks()