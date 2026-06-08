import numpy as np

# takes in simulated paths and returns payoffs
class EuropeanCallPayoff:
    def __init__(self, K: float):
        self.K = K

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        terminal = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(terminal - self.K, 0)
