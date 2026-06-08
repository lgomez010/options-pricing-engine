import numpy as np

#structure of MC prices has model, payoff and engine decoupled
#engine owns the randomness and the MC mechanics
#the model converts draws to paths
#the payoff converts paths to dollars
class MonteCarloEngine:
    def __init__(self, model, payoff, r: float, T: float):
        self.model = model
        self.payoff = payoff
        self.r = r
        self.T = T
    
    def price(self, n_paths: int, n_steps: int = 1,
              antithetic: bool = False,
              control_variate: bool = False) -> dict:

        #1. ask model what shape of randomness it needs
        shape = self.model.random_shape(n_paths, n_steps)

        #2. generate standard normal draws
        Z = np.random.standard_normal(shape)

        #3. simulate paths
        paths = self.model.simulate(Z)

        #4. evaluate payoffs
        payoffs = self.payoff(paths)

        #5. if antithetic, also simulate with -Z and average
        if antithetic:
            paths_anti = self.model.simulate(-Z)
            payoffs_anti = self.payoff(paths_anti)
            payoffs = 0.5 * (payoffs + payoffs_anti)
            paths = 0.5 * (paths + paths_anti) #keep paths consistent for control variate

        #6. control variate correction
        if control_variate:
            control_mean = self.model.control_variate_mean()
            terminal_values = paths # control variate is terminal stock price
            cov = np.cov(payoffs, terminal_values)[0,1]
            var_control = np.var(terminal_values)
            if var_control > 0:
                beta = cov / var_control
                payoffs -= beta * (terminal_values - control_mean)


        #7. discount and average
        price = np.exp(-self.r * self.T) * np.mean(payoffs)
        stderr = np.exp(-self.r * self.T) * np.std(payoffs) / np.sqrt(n_paths)

        return {"price": price, "std_error": stderr}
