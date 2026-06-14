import numpy as np
from scipy.linalg import solve_banded

class PDESolver:
    """Crank-Nicolson finite difference solver for Black-Scholes PDE"""

    def __init__(self, model, payoff, n_space: int = 200, n_time: int = 200, grid_width: float = 5.0):
        self.model = model              #provides PDE coefficients r and sigma
        self.payoff = payoff            #provides terminal condition and boundary values
        self.n_space = n_space          #number of grid points in log-price direction
        self.n_time = n_time            #number of time steps from expiry to today
        self.grid_width = grid_width    #half-width of log-price grid in units of sigma*sqrt(T)

    def _setup_grid(self, S0: float, T: float):
        """set up the log_price grid and time grid for PDE solver"""
        
        #get model parameters
        r, sigma = self.model.pde_coefficients(S0, 0.0)

        #build log-price grid centered at ln(S0) with width determined by grid_width * sigma * sqrt(T)
        x_center = np.log(S0)
        x_width = self.grid_width * sigma * np.sqrt(T)
        x_min = x_center - x_width
        x_max = x_center + x_width

        #creat evenly spaced grid and compute Delta x
        self.x = np.linspace(x_min, x_max, self.n_space)
        self.dx = self.x[1] - self.x[0]

        #create time grid and compute Delta t
        self.dt = T / self.n_time

        #convert the log-grid back to stock prices (needed for terminal condition and boundary calls)
        self.S = np.exp(self.x)

        #store values
        self.r = r
        self.sigma = sigma
        self.T = T

    def _build_operator(self):
        """build tridiagonal matrices A and B for Crank-Nicolson"""
        dx = self.dx
        dt = self.dt
        mu = self.r - 0.5 * self.sigma**2 
        d = 0.5 * self.sigma**2

        #tridiagonal entries of L
        lower = -mu / (2 * dx) + d / dx**2
        main = -2 * d / dx**2 - self.r
        upper = mu / (2 * dx) + d / dx**2

        N = self.n_space

        # A = I - 0.5*dt*L
        self.A = np.zeros((3,N))
        self.A[0, :] = -0.5 * dt * upper        #upper diagnoal
        self.A[1, :] = 1 - 0.5 * dt * main      #main diagonal
        self.A[2, :] = -0.5 * dt * lower        #lower diagonal

        # B = I + 0.5*dt*L
        self.B = np.zeros((3,N))
        self.B[0, :] = 0.5 * dt * upper
        self.B[1, :] = 1 + 0.5 * dt * main
        self.B[2, :] = 0.5 * dt * lower

    def _set_terminal_condition(self):
        """set option values at expiry: u(x,T) = payoff(S)"""
        #takes array of stock prices from _setup_grid()
        #returns max(S-K,0) at each point
        self.u = self.payoff(self.S)

    def _step_backward(self, t: float):
        """one Crank-Nicolson sgtep from t+dt back to t"""

        """
        Algorithm at each step:
        1)mult. B*u^{n+1} to get RHS
        2)Apply boundary conditions to RHS
        3)Solve A*u^n = RHS for u^n
        """

        u= self.u

        #multiply B * u (tridiagonal matrix-vector product)
        rhs = np.zeros_like(u)
        rhs[1:-1] = (self.B[2, 1:-1] * u[:-2]       #lower * u_{j-1}
                    + self.B[1, 1:-1] * u[1:-1]     #main * u_j
                    + self.B[0, 1:-1] * u[2:])      #upper * u_{j+1}

        #boundary conditions from payoff
        rhs[0] = self.payoff.boundary(self.S[0], t, self.r)
        rhs[-1] = self.payoff.boundary(self.S[-1], t, self.r)

        #solve A * u_new = rhs
        self.u = solve_banded((1,1), self.A, rhs)

    def price(self, S0: float, T: float, american: bool = False):
        """
        price an option via Crank-Nicolson PDE solver
        
        Returns the option price at S0
        """

        #setup
        self._setup_grid(S0, T)
        self._build_operator()
        self._set_terminal_condition()

        boundary_prices = []
        boundary_times = []


        #step backward from expiry to today
        for n in range(self.n_time, 0, -1):
            t = (n-1) * self.dt     #time we're stepping to
            self._step_backward(t)

            # American: take max of continuation and exercise
            if american:
                exercise = self.payoff(self.S)

                # find exercise boundary
                diff = self.u - exercise 
                idx = np.argmax(diff > 0)
                if idx > 0:
                    boundary_prices.append(self.S[idx])
                    boundary_times.append(t)


                self.u = np.maximum(self.u, exercise)
        self.exercise_boundary = np.array(boundary_prices)
        self.boundary_times = np.array(boundary_times)

        # interpolate to get price at exact S0
        x0 = np.log(S0)
        price = np.interp(x0, self.x, self.u)

        return price
