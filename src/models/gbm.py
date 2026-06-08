
import numpy as np

#acts as the data container plus a simulator
#defines model (S0,r,sigma,T) then has three methods the engine will call
class GBMModel:
    def __init__(self, S0: float, r: float, sigma: float, T: float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T

    #asks "how much randomness do you need?" and returns the shape of the array of standard normals
    def random_shape(self, n_paths: int, n_steps: int) -> tuple:
        return (n_paths,) # one draw per path, exact simulation

    # takes random draws Z and uses GBM to produce terminal stock price paths
    def simulate(self, Z: np.ndarray) -> np.ndarray:
        # Z has shape (n_paths,)
        return self.S0 * np.exp(
            (self.r -0.5 * self.sigma**2) * self.T
            + self.sigma * np.sqrt(self.T) * Z
        ) 

    # the control variate is the terminal stock price, which has known expectation
    def control_variate_mean(self) -> float:
        return self.S0 * np.exp(self.r * self.T)