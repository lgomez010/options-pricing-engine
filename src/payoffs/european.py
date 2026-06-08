import numpy as np

# takes in simulated paths and returns payoffs
class EuropeanCallPayoff:
    def __init__(self, K: float):
        self.K = K

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        terminal = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(terminal - self.K, 0)

    def payoff_derivative(self, S_T: np.ndarray) -> np.ndarray:
        return (S_T > self.K).astype(float)

        