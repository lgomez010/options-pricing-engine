"""Binomial tree (CRR) pricing engine for European and American options"""

import numpy as np

def crr_parameters(T, r, sigma, N):
    """compute the tree parameters for Cox-Ross-Rubinstein binomial tree"""
    dt = T / N #steps
    u = np.exp(sigma * np.sqrt(dt)) #up move
    d = np.exp(-sigma * np.sqrt(dt)) # down move
    p = (np.exp(r * dt) - d) / (u-d) #risk neutral probability

    return u, d, p

def price_option(S, K, T, r, sigma, N, option_type="call", exercise="european"):
    """price an option using CRR binomial tree"""
    u, d, p = crr_parameters(T, r, sigma, N)

    #terminal payoff vector
    j= np.arange(N+1) #[0,1,...,N] array
    final_prices = S * u**j * d**(N-j) #terminal stock price

    if option_type == "call":
        payoffs = np.maximum(final_prices - K, 0) #call payoff
    else:
        payoffs = np.maximum(K- final_prices, 0) # put payoff

    #backward pass thru the tree
    dt = T / N
    discount = np.exp(-r * dt)

    for step in range(N-1, -1, -1):
        # at this step, payoffs has (step + 2) elements
        # combine adjacent pairs to get (step + 1) elements
        payoffs = discount * (p * payoffs[1:] + (1-p) * payoffs[:-1])

        # for american options, compare agaisnt early exercise
        if exercise == "american":
            j = np.arange(step + 1)
            stock_prices = S * u**j * d**(step - j)
            if option_type == "call":
                early_exercise = stock_prices - K
            else: 
                early_exercise = K - stock_prices
            payoffs = np.maximum(payoffs, early_exercise)

    return payoffs[0] 

if __name__ == "__main__":
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2

    for N in [50, 100, 500, 1000, 5000]:
        price = price_option(S, K, T, r, sigma, N, "call", "european")
        print(f"N={N:>5}: {price:.6f}")
print(f"BS:      10.450584")

euro_put = price_option(S, K, T, r, sigma, 1000, "put", "european")
amer_put = price_option(S, K, T, r, sigma, 1000, "put", "american")
print(f"\nEuropean put: {euro_put:.4f}")
print(f"American put: {amer_put:.4f}")
print(f"Early exercise premium: {amer_put - euro_put:.4f}")