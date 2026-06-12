"""Test for Black-Scholes pricing model"""

import numpy as np
import pytest
from src.models.black_scholes import (
    call_price,
    put_price,
    call_delta,
    put_delta,
    gamma,
    vega,
)

def test_put_call_parity():
    """Test put-call parity for European options"""
    S = 100 #spot price
    K = 100 #strike price
    T = 1 #time to maturity in years
    r = 0.05 #risk-free interest rate
    sigma = 0.2 #volatility
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)
    assert np.isclose(call - put, S - K * np.exp(-r * T))

def test_delta_relationship():
    """Test the relationship between call and put delta"""
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    call_delta_value = call_delta(S, K, T, r, sigma)
    put_delta_value = put_delta(S, K, T, r, sigma)
    assert np.isclose(call_delta_value - put_delta_value, 1.0)

def test_gamma_vega_relationship():
    """Test the relationship between gamma and vega"""
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    gamma_value = gamma(S, K, T, r, sigma)
    vega_value = vega(S, K, T, r, sigma)
    assert np.isclose(vega_value, gamma_value * S**2 * sigma * T)

# The three prices at S-h, S, S+h measure how the slope changes
# across the neighborhood around S, which is curvature (gamma)
def test_gamma_finite_difference():
    """Test gamma using finite difference approximation"""
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    h = 0.01 # small change in spot price

    gamma_analytical = gamma(S, K, T, r, sigma) #analytical gamma from formula

    #numerical gamma: bump S up and down, use the second derivative formula
    C_up = call_price(S + h, K, T, r, sigma)
    C_mid = call_price(S, K, T, r, sigma)
    C_down = call_price(S - h, K, T, r, sigma)
    gamma_numerical = (C_up -2 * C_mid + C_down) / h**2

    assert np.isclose(gamma_analytical, gamma_numerical, rtol=1e-4)
