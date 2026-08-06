"""
Tests for Heston stochastic volatility model

Benchmarks against Albrecher et al. (2007), "The Little Heston Trap," Table 2 (the calibrated DAX set):

    S0 = 100, r = 0.025, v0 = 0.0175,
    kappa = 1.5768, theta = 0.0398,
    sigma_v = 0.5751, rho = -0.5711.

ATM call prices at K = 100 across maturities T = 1, 2, 5, 10 years
are taken from Albrecher Table 2.
"""
import numpy as np
import pytest
import warnings

from src.models.heston import HestonModel
from src.models import black_scholes as bs


##Albrecher Table 2 parameter set
S0, R = 100.0, 0.025
V0, KAPPA, THETA = 0.0175, 1.5768, 0.0398
SIGMA_V, RHO = 0.5751, -0.5711

ALBRECHER_ATM_PRICES = {
    1.0: 7.27,
    2.0: 11.73,
    5.0: 21.75,
    10.0: 33.84,
}

MATURITIES = [1.0, 2.0, 5.0, 10.0]
STRIKES = [80.0, 100.0, 120.0]  # ITM, ATM, OTM


#A fixture is a function that builds an object a test needs.  This says: "if any test takes an argument named 
#heston, call this function and pass the result in."
@pytest.fixture
def heston():
    """Default Heston model on the Abrecher Table 2 parameter set"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HestonModel(
            S0=S0, r=R, v0=V0, kappa=KAPPA, theta=THETA, sigma_v=SIGMA_V, rho=RHO
        )

@pytest.fixture
def heston_degenerate():
    """Heston with sigma_v -. 0, theta=v0,rho=0.  This is a Black-Scholes model with volatility sqrt(v0)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HestonModel(
            S0=S0,
            r=R,
            v0=V0,
            kappa=KAPPA,
            theta=V0,
            sigma_v=1e-4,
            rho=0.0,
        )

        
class TestBlackScholesLimit:
    """Degenerate Heston must match Black-Schoiles with sigma = sqrt(v0)"""

    @pytest.mark.parametrize("T", MATURITIES)
    @pytest.mark.parametrize("K", STRIKES)
    def test_call_matches_bs(self, heston_degenerate, K, T):
        heston_price = heston_degenerate.price_call(K, T)
        bs_price = bs.call_price(S0, K, T, R, np.sqrt(V0))
        assert heston_price == pytest.approx(bs_price, abs=1e-3)


class TestConstruction:
    """__post_inti__ enforces parameter constraints"""

    def test_negative_v0_raises(self):
        with pytest.raises(ValueError, match="v0"):
            HestonModel(S0=100, r=0.025, v0=-0.01, kappa=1.5, theta=0.04, sigma_v=0.5, rho=-0.5)

    def test_negative_theta_raises(self):
        with pytest.raises(ValueError, match='theta'):
            HestonModel(S0=100, r=0.025, v0=0.02, kappa=1.5, theta=-0.01, sigma_v=0.5, rho=-0.5)

    def test_nonpositive_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            HestonModel(S0=100, r=0.025, v0=0.02, kappa=0.0, theta=0.04, sigma_v=0.5, rho=-0.5)

    def test_nonpositive_sigma_v_raises(self):
        with pytest.raises(ValueError, match="sigma_v"):
            HestonModel(S0=100, r=0.025, v0=0.02, kappa=1.5, theta=0.04, sigma_v=-0.1, rho=-0.5)

    
    @pytest.mark.parametrize("rho", [-1.5, 1.5, 2.0])
    def test_rho_out_of_range_raises(self, rho):
        with pytest.raises(ValueError, match="rho"):
            HestonModel(S0=100, r=0.025, v0=0.02, kappa=1.5, theta=0.04, sigma_v=0.5, rho=rho)

    def test_feller_violation_warns(self):
        with pytest.warns(UserWarning, match="Feller"):
            HestonModel(S0=S0, r=R, v0=V0, kappa=KAPPA,
                        theta=THETA, sigma_v=SIGMA_V, rho=RHO)    

    def test_feller_satisfied_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            HestonModel(S0=100, r=0.025, v0=0.04, kappa=2.0,
                        theta=0.04, sigma_v=0.1, rho=-0.5)

class TestCharacteristicFunction:
    """phi(u, T) is forced to satisfy two identities by definition.

    1. phi(0, T) = 1: probability measures integrate to one.
    2. phi(-i, T) = S0 * exp(rT): risk-neutral martingale property of S_t.

    Both hold at every maturity, so we parametrize over T.
    """

    @pytest.mark.parametrize("T", MATURITIES)
    def test_phi_at_zero_exact(self, heston, T):
        """phi(0, T) = 1 + 0i exactly."""
        assert heston.char_func(0.0, T) == pytest.approx(1.0 + 0j, abs=1e-14)

    @pytest.mark.parametrize("T", MATURITIES)
    def test_phi_at_minus_i_exact(self, heston, T):
        """phi(-i, T) = S0 * exp(rT) exactly."""
        expected = S0 * np.exp(R * T)
        assert heston.char_func(-1j, T) == pytest.approx(expected, rel=1e-12)

    # Formula checks: approach the special points (bypasses short-circuits).

    @pytest.mark.parametrize("T", MATURITIES)
    def test_phi_near_zero_via_formula(self, heston, T):
        """phi(u, T) -> 1 as u -> 0 through the general Gatheral 2.12 formula."""
        phi = heston.char_func(1e-8, T)
        assert phi == pytest.approx(1.0 + 0j, abs=1e-6)

    @pytest.mark.parametrize("T", MATURITIES)
    def test_phi_near_minus_i_via_formula(self, heston, T):
        """phi(u, T) -> S0*exp(rT) as u -> -i through the general formula."""
        phi = heston.char_func(-1j + 1e-8, T)
        expected = S0 * np.exp(R * T)
        assert phi == pytest.approx(expected, rel=1e-6)

class TestBenchmarkPrices:
    """ATM call prices match Albrecher et al. (2007) Table 2 to two decimals."""

    @pytest.mark.parametrize("T,expected", list(ALBRECHER_ATM_PRICES.items()))
    def test_atm_call_matches_albrecher(self, heston, T, expected):
        price = heston.price_call(K=S0, T=T)
        assert price == pytest.approx(expected, abs=0.01)


class TestPutCallParity:
    """Put-call parity: C - P = S0 - K*exp(-rT)"""

    @pytest.mark.parametrize("T", MATURITIES)
    @pytest.mark.parametrize("K", STRIKES)
    def test_parity_holds(self, heston, K, T):
        call = heston.price_call(K, T)
        put = heston.price_put(K, T)
        lhs = call - put
        rhs = S0 - K * np.exp(-R *T)
        assert lhs == pytest.approx(rhs, abs=1e-6)
