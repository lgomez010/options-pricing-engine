from src.models.black_scholes import call_price as bs_call_price
from src.engines.binomial_tree import price_option
import numpy as np

def test_crr_converges_to_bs():
    """Test that the CRR binomial tree converges to Black-Scholes price as N increases"""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    bs_price = bs_call_price(S, K, T, r, sigma)
    crr_price = price_option(S, K, T, r, sigma, 2000, "call", "european")
    assert np.isclose(crr_price, bs_price, rtol=1e-3)


def test_american_put_geq_european():
    """American put is at least as valuable as European put"""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    euro = price_option(S, K, T, r, sigma, 500, "put", "european")
    amer = price_option(S, K, T, r, sigma, 500, "put", "american")
    assert amer >= euro