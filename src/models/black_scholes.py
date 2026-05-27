"""
Black-Scholes pricing model for European options

Implements closed-form solutions for European call and put prices
and their analytical Greeks under the Black-Scholes-Merton framework

All functions accept scalar or NumPy array inputs
"""

import numpy as np
from scipy.stats import norm

def _d1_d2(S, K, T, r, sigma):
    """compute d1 and d2 for Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility

    Returns:
    --------
    d1 : float or np.ndarray
    d2 : float or np.ndarray
    """

    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2

def call_price(S, K, T, r, sigma):
    """
    compute the price of a European call option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    price : float or np.ndarray
        price of the European call option
    """

    d1,d2 = _d1_d2(S, K, T, r, sigma)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def put_price(S, K, T, r, sigma):
    """
    compute the price of a European put option using
    Black-Scholes formula

    Parameters:
    ----------
    S: float or np.ndarray
        current spot price of the underlying asset
    K: float or np.ndarray
        strike price
    T: float or np.ndarray
        time to maturity in years
    r: float
        risk-free interest rate
    sigma: float
        volatility

    Returns:
    -------
    price: float or np.ndarray
        price of the European put option

    Note: \phi(-x) = 1 - \phi(x) for the standard normal CDF,
     so we can use norm.cdf(-d1) and norm.cdf(-d2)

    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def call_delta(S, K, T, r, sigma):
    """
    compute the delta of a European call option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    delta : float or np.ndarray
        delta of the European call option
    """

    d1, _ = _d1_d2(S, K, T, r, sigma)

    return norm.cdf(d1)

def put_delta(S, K, T, r, sigma):
    """
    compute the delta of a European put option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility

    Returns:
    --------
    delta : float or np.ndarray
        delta of the European put option
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)

    return norm.cdf(d1) - 1

def gamma(S, K, T, r, sigma):
    """
    compute the gamma of a European option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    gamma: float or np.ndarray
        gamma of European option (same for call and put)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)

    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """
    compute the vega of a European option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility

    Returns:
    --------
    vega : float or np.ndarray
        vega of European option (same for calls and puts)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)

    return S * norm.pdf(d1) * np.sqrt(T)

def call_theta(S, K, T, r, sigma):
    """
    compute the theta of a European call option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    theta : float or np.ndarray
        theta of European call option
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)

def put_theta(S, K, T, r, sigma):
    """
    compute the theta of a European put option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    theta : float or np.ndarray
        theta of European put option
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
       
    return -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)

def call_rho(S, K, T, r, sigma):
    """
    compute the rho of a European call option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    rho : float or np.ndarray
        rho of European call option
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    return K * T * np.exp(-r * T) * norm.cdf(d2)

def put_rho(S, K, T, r, sigma):
    """
    compute the rho of a European put option using
    Black-Scholes formula

    Parameters:
    -----------
    S : float or np.ndarray
        current spot price of the underlying asset
    K : float or np.ndarray
        strike price
    T : float or np.ndarray
        time to maturity in years
    r : float
        risk-free interest rate
    sigma : float
        volatility
    
    Returns:
    --------
    rho : float or np.ndarray
        rho of European put option
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
  
    return -K * T * np.exp(-r * T) * norm.cdf(-d2)


""" Demo block to test the functions with example parameters """
if __name__ == "__main__":
    params = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)

    c = call_price(**params)
    p = put_price(**params)

    print(f"Call price: {c:.4f}")
    print(f"Put price: {p:.4f}")

    # Put-call parity check
    lhs = c - p
    rhs = params["S"] - params["K"] * np.exp(-params["r"] * params["T"])
    print(f"\nPut-call parity: C-P = {lhs:.6f}")
    print(f"                 S - Ke^(-rT) = {rhs:.6f}")
    print(f"                 Difference: {abs(lhs - rhs):.2e}")
