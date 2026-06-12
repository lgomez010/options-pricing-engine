import numpy as np

# takes in simulated paths and returns payoffs
class EuropeanCallPayoff:
    def __init__(self, K: float, T: float | None = None):
        self.K = K
        self.T = T

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        terminal = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(terminal - self.K, 0)

    def payoff_derivative(self, S_T: np.ndarray) -> np.ndarray:
        return (S_T > self.K).astype(float)

    def boundary(self, S: float, t: float, r: float) -> float:
        """option value at boundary S at time t (time from today, not to expiry)"""
        if self.T is None:
            raise ValueError("Expiry T required for PDE boundary conditions")
        if S < self.K:
            return 0.0
        else:
            return S - self.K * np.exp(-r * (self.T - t))
            
class EuropeanPutPayoff:
    def __init__(self, K: float, T: float | None = None):
        self.K = K
        self.T = T

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        terminal = paths if paths.ndim == 1 else paths[:, -1]
        return np.maximum(self.K - terminal, 0)

    def payoff_derivative(self, S_T: np.ndarray) -> np.ndarray:
        return -(S_T < self.K).astype(float)

    def boundary(self, S: float, t: float, r: float) -> float:
        """option value at boundary S at time t (time from today, not to expiry)"""
        if self.T is None:
            raise ValueError("Expiry T required for PDE boundary conditions")
        if S > self.K:
            return 0.0
        else:
            return self.K * np.exp(-r * (self.T - t)) - S