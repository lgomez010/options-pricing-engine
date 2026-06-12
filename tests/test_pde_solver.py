import numpy as np
from src.models.black_scholes import call_price, put_price
from src.models.gbm import GBMModel
from src.payoffs.european import EuropeanCallPayoff, EuropeanPutPayoff
from src.engines.pde_solver import PDESolver

def test_pde_european_call():
    """PDE price matches Black-Scholes closed-form for European call."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    bs = call_price(S0, K, T, r, sigma)

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K, T)
    solver = PDESolver(model, payoff)
    pde = solver.price(S0, T)

    assert abs(pde - bs) < 0.05, f"PDE {pde:.4f} vs BS {bs:.4f}"

def test_pde_put_call_parity():
    """PDE call and put prices satisfy put-call parity."""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

    model = GBMModel(S0, r, sigma, T)

    call_payoff = EuropeanCallPayoff(K, T)
    put_payoff = EuropeanPutPayoff(K, T)

    call_pde = PDESolver(model, call_payoff).price(S0, T)
    put_pde = PDESolver(model, put_payoff).price(S0, T)

    # C - P = S - Ke^{-rT}
    lhs = call_pde - put_pde
    rhs = S0 - K * np.exp(-r * T)

    assert abs(lhs - rhs) < 0.05, f"Parity gap: {abs(lhs - rhs):.4f}"


def test_american_put_geq_european():
    """American put from PDE is at least as valuable as European put."""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanPutPayoff(K, T)

    euro = PDESolver(model, payoff).price(S0, T, american=False)
    amer = PDESolver(model, payoff).price(S0, T, american=True)

    assert amer >= euro - 1e-10, f"American {amer:.4f} < European {euro:.4f}"


def test_american_call_equals_european():
    """American call on non-dividend stock equals European call."""
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K, T)

    euro = PDESolver(model, payoff).price(S0, T, american=False)
    amer = PDESolver(model, payoff).price(S0, T, american=True)

    assert abs(amer - euro) < 0.01, f"American {amer:.4f} vs European {euro:.4f}"