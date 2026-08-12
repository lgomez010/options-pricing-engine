"""Implied volatility via Brent's method."""
import numpy as np
from scipy.optimize import brentq
from src.models.black_scholes import call_price, put_price


def implied_vol(price, S, K, T, r, option_type="call",
                lo=1e-6, hi=5.0, tol=1e-8):
    """Back out Black-Scholes implied volatility from an option price.

    Uses Brent's method (bracketed root-finder, guaranteed convergence
    for continuous functions when bracket has a sign change).

    Parameters
    ----------
    price : float
        Observed option price.
    S, K, T, r : float
        Spot, strike, maturity, risk-free rate.
    option_type : {"call", "put"}
    lo, hi : float
        Volatility bracket. Defaults handle any realistic case.
    tol : float
        Root-finder tolerance.

    Returns
    -------
    float
        Implied volatility, or np.nan if no solution in [lo, hi]
        (typically means price violates arbitrage bounds).
    """
    pricer = call_price if option_type == "call" else put_price

    def f(sigma):
        return pricer(S, K, T, r, sigma) - price

    # Check bracket has a sign change
    if f(lo) * f(hi) > 0:
        return np.nan

    return brentq(f, lo, hi, xtol=tol)