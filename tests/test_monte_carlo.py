import numpy as np
from src.models.black_scholes import call_price
from src.models.gbm import GBMModel
from src.payoffs.european import EuropeanCallPayoff
from src.engines.monte_carlo import MonteCarloEngine

def test_mc_converges_to_bs():
    """Test that Monte Carlo price converges to Black_scholes price as n_paths increases"""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    n_paths = 500_000
    np.random.seed(37)

    bs = call_price(S0, K, T, r, sigma) #get analytical price

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K)
    engine = MonteCarloEngine(model, payoff, r, T)
    result = engine.price(n_paths)

    assert abs(result["price"] - bs) < 4 * result["std_error"]

def test_antithetic_reduces_variance():
    """Test that antithetic variates reduce standard error"""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    n_paths = 500_000

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K)
    engine = MonteCarloEngine(model, payoff, r, T)

    np.random.seed(37)
    plain = engine.price(n_paths, antithetic=False)

    np.random.seed(37)
    anti = engine.price(n_paths, antithetic=True)

    assert anti["std_error"] < plain["std_error"]

def test_control_variate_reduces_variance():
    """Test that control variate reduces standard error"""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05,  0.2
    n_paths = 500_000

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K)
    engine = MonteCarloEngine(model, payoff, r, T)

    np.random.seed(37)
    plain = engine.price(n_paths, control_variate=False)

    np.random.seed(37)
    control = engine.price(n_paths, control_variate=True)

    assert control["std_error"] < plain["std_error"]

def test_convergence_rate():
    """Test that the standard error decreases at rate 1/sqrt(n_paths)"""
    S0, K, T, r, sigma = 100.0, 105.0, 1.0, 0.05, 0.2
    n_path1 = 100_000
    n_path2 = 400_000

    model = GBMModel(S0, r, sigma, T)
    payoff = EuropeanCallPayoff(K)
    engine = MonteCarloEngine(model, payoff, r, T)
    np.random.seed(37)

    result1 = engine.price(n_path1)
    result2 = engine.price(n_path2)

    ratio = result1["std_error"] / result2["std_error"]
    expected_ratio = np.sqrt(n_path2 / n_path1)

    assert abs(ratio - expected_ratio) < 0.1 * expected_ratio

    