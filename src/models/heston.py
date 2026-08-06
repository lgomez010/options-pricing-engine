"""Heston stochastic volatitlity model

Pricing via the characteristic function (Gatheral 2.12, stable form),
inverted with Gil-Pelaez to recover the European call price

References: 
Gatheral, The Volatiltiy Surface (2006), Ch. 2.
Albrecher et al. (2007), "The Little Heston Trap"
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad

@dataclass(frozen=True, kw_only=True)
class HestonModel:
    """Heston stochastic volatility model

    Parameters:
    -------------------------
    S0 : float
        initial stock price
    r : float
        risk-free interest rate
    v0 : float
        initial variance (volatility squared)
    kappa : float
        mean reversion speed of variance process
    theta : float
        long-term mean of variance process (bar{v} in Gatheral, eta in Albrecher)
    sigma_v : float
        volatility of variance process (vol of vol) (eta in Gatheral, lambda in Albrecher))
    rho : float
        correlation between two Brownians i.e. stock and variance processes
    """

    S0: float
    r: float
    v0: float
    kappa: float
    theta: float
    sigma_v: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0:
            raise ValueError("v0 (initial variance) must be non-negative")
        if self.theta <0:
            raise ValueError("theta (long-run variance) must be non-negative")
        if self.kappa <=0:
            raise ValueError("kappa (mean-reversion speed) must be positive")
        if self.sigma_v <= 0:
            raise ValueError("sigma_v (vol-of-vol) must be positive")
        if not (-1.0 <= self.rho <=1.0):
            raise ValueError("rho (correlation) must lie in [-1, 1]")

        feller = 2.0 * self.kappa * self.theta
        if feller < self.sigma_v ** 2:
            import warnings
            warnings.warn(
                f"Feller condition violated: 2*kappa*theta = {feller:.4f} "
                f"< sigma_v^2 = {self.sigma_v**2:.4f}. "
                f"Variance process may reach zero.",
                stacklevel=2, #makes warning point to the line where user created HestonModel and not inside __post_init__
            )

    def _d(self, u):
        """Riccati discriminant (Gatheral 2.13)

        d(u) = sqrt((rho * sigma_v * iu - kappa)^2 + sigma_v^2 (u^2 + iu))

        uses principal branch of sqrt to get stable phi_2 form (Albrecher et al. 2007)
        """

        iu = 1j * u
        return np.sqrt(
            (self.rho * self. sigma_v * iu - self.kappa) ** 2
            + self.sigma_v ** 2 * (u ** 2 +iu)
        )

    def _g(self, u):
        """Ratio g = r_- / r_+ (Gatheral 2.13)

        r_pm = (rho * sigma_v * iu - kappa +/- d) / sigma_v^2
        g = r_- / r_+ = (... - d) / (... + d)
        """
        d = self._d(u)
        beta = self.kappa - self.rho * self.sigma_v * 1j * u
        return (beta - d) / (beta + d)

    def _D(self, u, T):
        """Coefficient of v0 in the characteristic exponent (Gatheral 2.12)

        D(u,T) = r_minus * (1-exp(-d*T)) ? (1-g*exp(-d*T)
        
        where r_minus = (beta - d) / sigma_v^2, beta = rho*sigma_v*iu - kappa
        """

        d = self._d(u)
        g = self._g(u)
        beta = self.kappa - self.rho * self.sigma_v * 1j * u
        r_minus = (beta - d) / self.sigma_v**2
        return r_minus * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))    

    def _C(self, u, T):
        """Coefficient of theta in the characterisitc exponent (Gatheral 2.12)

        C(u, T) = kappa * [r_minus * T - (2/sigma_v^2) * log((1 - g*exp(-dT)) / (1 - g))]

        The phi_2 (Gatheral/Albrecher) convention guarantees the log argument
        stays in the right half-plane for all u >= 0, so principal-branch np.log
        is safe. This is the "Little Heston Trap"; see Albrecher et al. (2007).
        """

        d = self._d(u)
        g = self._g(u)
        beta = self.kappa - self.rho * self.sigma_v * 1j * u
        r_minus = (beta - d) / self.sigma_v ** 2
        return self.kappa * (
            r_minus * T - (2.0 / self.sigma_v**2) * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        
    def char_func(self, u, T):
        """Characteristic function of log(S_T) under Heston (Gatheral 2.12, phi_2 form)

        phi(u, T) = E[exp(iu * log(S_T))] = exp(iu*(log(S0) + r*T) + C(u,T)*theta + D(u,T)*v0)

        Parameters
        --------------
        u : float or complx Fourier variable
        T: float time to maturity

        Returns
        --------------
        phi : complex charateristic function value
        """

        # Special values where (beta + d) = 0 (removable singularity in _g)
        if u == 0:
            return 1.0 + 0j
        if u == -1j:
            return self.S0 * np.exp(self.r * T) + 0j
        iu = 1j * u
        return np.exp(
            iu * (np.log(self.S0) + self.r * T)
            + self._C(u, T) * self.theta
            + self._D(u, T) * self.v0
        )

    def _integrand(self, u, K, T, j):
        """Integrand for Gil-Pelaez inversion to recover P_j

        For j=2: uses phi(u, T) directly (risk-neutral measure)
        For j=1: uses phi(u -i, T) / phi(-i, T) (stock-numeraire measure)
        """
        if j == 2:
            phi = self.char_func(u, T)
        else: # j==1
            phi = self.char_func(u - 1j, T) / self.char_func(-1j, T)
        return (np.exp(-1j * u * np.log(K)) * phi / (1j * u)).real

    def price_call(self, K, T):
        """Price a European call via Gil-Pelaez inversion

        C = S0 * P1 - K * exp(-rT) *P2
        where P_j = 1/2 + (1/pi) * integral_0^inf Re[exp(-iu*log(K)) * phi_j(u) / (iu)] du

        Parameters
        --------------
        K : float strike price
        T : float time to maturity

        Returns
        --------------
        float European call option price
        """

        P1, P2 = self._compute_P1_P2(K, T)
        return self.S0 * P1 - K * np.exp(-self.r * T) * P2

    def price_put(self, K, T):
        """Price a European put via Gil-Pelaez inversion

        P = K * exp(-rT) * (1 - P2) - S0 * (1 - P1)

        Uses the same P1 and P2 probabilities as price_call. This is independent and not defined via put-call parity, 
        so parity provides a cross-check
        """
        P1, P2 = self._compute_P1_P2(K, T)
        return K * np.exp(-self.r * T) * (1.0 - P2) - self.S0 * (1.0 - P1)


    def _compute_P1_P2(self, K, T):
        """Gil-Pelaez probabilities P1, P2 for a call/put with strike K, maturity T.

        Both price_call and price_put consume these; computing them once here
        keeps put-call parity an honest cross-check rather than a tautology.
        """
        P1, _ = quad(self._integrand, 0, np.inf, args=(K, T, 1), limit=200)
        P2, _ = quad(self._integrand, 0, np.inf, args=(K, T, 2), limit=200)
        return 0.5 + P1 / np.pi, 0.5 + P2 / np.pi

