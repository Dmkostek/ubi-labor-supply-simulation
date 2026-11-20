from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import newton

from ubi_sim import (
    SENSITIVITY_LEVELS,
    UBI_LEVELS,
    InfeasibleUBI,
    Params,
    calibrate_psi,
    gini,
    laffer_peak,
    make_population,
    optimal_hours,
    revenue_neutral_tax,
    run_scenarios,
    tax_revenue,
    utility,
)

P = Params()


@pytest.fixture(scope="module")
def pop():
    return make_population(P)


@pytest.fixture(scope="module")
def scenarios(pop):
    return run_scenarios(pop, P)


def test_population_reference_statistics(pop):
    # Reference summary statistics for seed 42.
    assert pop.wages.mean() == pytest.approx(261.26, abs=0.01)
    assert np.median(pop.wages) == pytest.approx(247.80, abs=0.01)
    assert pop.wages.min() == pytest.approx(82.58, abs=0.01)
    assert pop.wages.max() == pytest.approx(898.01, abs=0.01)


def test_baseline_revenue_reference(pop):
    # Reference value for t=0.2, B=0, seed 42.
    assert tax_revenue(pop, P.base_tax, 0.0, P) == pytest.approx(20_858_062.135, rel=1e-7)


def test_first_order_condition_holds(pop):
    tax, ubi = 0.3, 5_000.0
    m = optimal_hours(pop, tax, ubi, P) / P.T
    a = pop.wages * P.T * (1 - tax)
    foc = a / (pop.nonlabor + ubi + a * m) - P.psi * m ** (1 / P.frisch)
    assert np.max(np.abs(foc)) < 1e-9


def test_vectorised_solver_matches_independent_newton_solver(pop):
    """An independent per-agent Newton/secant solver agrees with the vectorised bisection."""

    def u(m, inc, wage, B, t):
        C = inc + B + wage * m * P.T * (1 - t)
        if C <= 0 or m < 0:
            return -np.inf
        return np.log(C) - (1 / (1 + 1 / P.frisch)) * m ** (1 + 1 / P.frisch)

    def newton_solve(inc, wage, B, t, h=1e-6):
        d = lambda m: (u(m + h, inc, wage, B, t) - u(m - h, inc, wage, B, t)) / (2 * h)
        return max(newton(d, x0=1.0, tol=1e-6, maxiter=100), 0.0) * P.T

    new = optimal_hours(pop, 0.25, 5_000.0, P)
    ref = np.array([newton_solve(pop.nonlabor[i], pop.wages[i], 5_000.0, 0.25) for i in range(25)])
    np.testing.assert_allclose(new[:25], ref, atol=0.05)


def test_hours_never_exceed_time_endowment(pop):
    for psi in (1e-4, 1e-2, 1.0):
        hours = optimal_hours(pop, 0.05, 0.0, replace(P, psi=psi))
        assert hours.max() <= P.T + 1e-9
        assert hours.min() >= 0.0


def test_no_ubi_uses_baseline_tax(scenarios):
    assert scenarios["No UBI"].tax == P.base_tax


def test_every_scenario_is_revenue_neutral(scenarios):
    for s in scenarios.values():
        assert s.revenue == pytest.approx(s.target_revenue, rel=1e-8)


def test_required_tax_rates(scenarios):
    # Reference values for seed 42.
    assert scenarios["No UBI"].tax == pytest.approx(0.2000, abs=5e-4)
    assert scenarios["UBI Low"].tax == pytest.approx(0.2532, abs=5e-4)
    assert scenarios["UBI High"].tax == pytest.approx(0.4355, abs=5e-4)


def test_scenarios_use_their_own_hours(scenarios):
    # Each scenario must be solved with its own tax rate and UBI.
    totals = [s.hours.sum() for s in scenarios.values()]
    assert totals[0] > totals[1] > totals[2]
    assert not np.allclose(scenarios["No UBI"].hours, scenarios["UBI High"].hours)


def test_utility_uses_each_scenarios_own_tax_and_hours(pop, scenarios):
    for s in scenarios.values():
        direct = utility(pop, s.hours, s.tax, s.ubi, P)
        np.testing.assert_allclose(s.utility, direct)
    # No-UBI welfare must not depend on what the UBI scenarios do.
    base = utility(pop, optimal_hours(pop, P.base_tax, 0.0, P), P.base_tax, 0.0, P)
    np.testing.assert_allclose(scenarios["No UBI"].utility, base)


def test_infeasible_ubi_raises(pop):
    with pytest.raises(InfeasibleUBI, match="infeasible"):
        revenue_neutral_tax(pop, 1e9, 1e15, P)


def test_skip_infeasible_levels(pop):
    levels = {"No UBI": 0.0, "Absurd": 1e9}
    with pytest.raises(InfeasibleUBI):
        run_scenarios(pop, P, levels)
    assert list(run_scenarios(pop, P, levels, skip_infeasible=True)) == ["No UBI"]


def test_recalibrated_high_ubi_is_infeasible(pop):
    """With a realistic labor-leisure trade-off the 20,000 UBI cannot be funded by a flat tax."""
    q = replace(P, psi=calibrate_psi(pop, P, 0.40))
    scen = run_scenarios(pop, q, SENSITIVITY_LEVELS, skip_infeasible=True)
    assert "UBI High" not in scen
    assert {"No UBI", "UBI Low", "UBI Mid"} <= set(scen)
    _, peak_revenue = laffer_peak(pop, 20_000.0, q)
    assert peak_revenue < scen["No UBI"].revenue + q.n_agents * 20_000.0


def test_ubi_raises_welfare_of_the_worst_off_and_compresses_inequality(scenarios):
    base, high = scenarios["No UBI"], scenarios["UBI High"]
    assert np.sort(high.utility)[:100].mean() > np.sort(base.utility)[:100].mean()
    assert high.utility.std() < base.utility.std()
    assert gini(high.consumption) < gini(base.consumption)


def test_invalid_tax_rejected(pop):
    with pytest.raises(ValueError):
        optimal_hours(pop, 1.0, 0.0, P)


def test_calibrate_psi_hits_target_share(pop):
    psi = calibrate_psi(pop, P, target_share=0.40)
    share = optimal_hours(pop, P.base_tax, 0.0, replace(P, psi=psi)).mean() / P.T
    assert share == pytest.approx(0.40, abs=1e-6)


def test_gini_bounds():
    assert gini(np.ones(100)) == pytest.approx(0.0, abs=1e-12)
    assert 0.9 < gini(np.r_[np.zeros(99), 1.0]) <= 1.0


def test_seed_reproducible():
    a, b = make_population(P), make_population(P)
    np.testing.assert_array_equal(a.wages, b.wages)
    c = make_population(replace(P, seed=7))
    assert not np.allclose(a.wages, c.wages)


def test_ubi_levels_unchanged():
    assert UBI_LEVELS == {"No UBI": 0.0, "UBI Low": 5_000.0, "UBI High": 20_000.0}
