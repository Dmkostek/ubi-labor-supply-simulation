"""Labor-supply model of a revenue-neutral universal basic income (UBI).

Each agent chooses the share of their time endowment ``m`` in [0, 1] to spend
working. Utility is log consumption minus a convex disutility of work::

    U(m) = ln(C) - psi * e / (1 + e) * m ** (1 + 1/e)
    C    = nonlabor + B + wage * T * m * (1 - t)

``e`` is the Frisch elasticity, ``B`` the UBI transfer and ``t`` a flat labor
income tax. The first-order condition is solved exactly (vectorised
bisection), so a whole population is solved in one call.

The government must raise the baseline tax revenue *plus* the cost of the UBI.
Tax revenue is a Laffer curve in ``t`` (it rises, peaks, then falls), so the
revenue-neutral rate is searched only on the rising branch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import brentq, minimize_scalar

UBI_LEVELS: dict[str, float] = {
    "No UBI": 0.0,
    "UBI Low": 5_000.0,
    "UBI High": 20_000.0,
}
# Adds a middle level for the sensitivity analysis.
SENSITIVITY_LEVELS: dict[str, float] = {
    "No UBI": 0.0,
    "UBI Low": 5_000.0,
    "UBI Mid": 10_000.0,
    "UBI High": 20_000.0,
}


@dataclass(frozen=True)
class Params:
    """Model parameters."""

    T: float = 400.0  # time endowment (hours)
    wage_mean: float = 260.0
    wage_std: float = 90.0
    nonlabor_mean: float = 500.0
    nonlabor_std: float = 300.0
    frisch: float = 0.5  # Frisch elasticity e
    base_tax: float = 0.20  # flat tax rate in the "No UBI" economy
    n_agents: int = 1000
    psi: float = 1.0  # scale of the disutility of work
    seed: int = 42


@dataclass(frozen=True)
class Population:
    wages: np.ndarray
    nonlabor: np.ndarray


@dataclass(frozen=True)
class Scenario:
    """Equilibrium outcome of one UBI level, with its own tax rate and hours."""

    label: str
    ubi: float
    tax: float
    hours: np.ndarray
    consumption: np.ndarray
    utility: np.ndarray
    revenue: float
    target_revenue: float


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """(mu, sigma) of the underlying normal for a lognormal with this mean/std."""
    sigma2 = np.log1p((std / mean) ** 2)
    return float(np.log(mean) - sigma2 / 2), float(np.sqrt(sigma2))


def make_population(p: Params) -> Population:
    """Draw wages and non-labor income (independent lognormals).

    A local ``RandomState`` keeps the draws reproducible without touching
    NumPy's global random state.
    """
    rng = np.random.RandomState(p.seed)
    mu_w, s_w = _lognormal_params(p.wage_mean, p.wage_std)
    mu_n, s_n = _lognormal_params(p.nonlabor_mean, p.nonlabor_std)
    wages = rng.lognormal(mean=mu_w, sigma=s_w, size=p.n_agents)
    nonlabor = rng.lognormal(mean=mu_n, sigma=s_n, size=p.n_agents)
    return Population(wages, nonlabor)


def optimal_hours(pop: Population, tax: float, ubi: float, p: Params, iters: int = 60) -> np.ndarray:
    """Utility-maximising hours worked for every agent.

    The first-order condition is ``a / (c + a m) = psi * m ** (1/e)`` with
    ``a = wage * T * (1 - t)`` and ``c = nonlabor + B``. The left side falls and
    the right side rises in ``m``, so there is exactly one root; bisection finds
    it. Hours are capped at the time endowment (``m <= 1``).
    """
    if not 0.0 <= tax < 1.0:
        raise ValueError(f"tax must be in [0, 1), got {tax}")
    a = pop.wages * p.T * (1.0 - tax)
    c = pop.nonlabor + ubi
    lo = np.zeros_like(a)
    # psi*m**(1/e) <= a/c at the root, which gives an analytic upper bound.
    hi = np.minimum(1.0, (a / (c * p.psi)) ** p.frisch)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        slope = a / (c + a * mid) - p.psi * mid ** (1.0 / p.frisch)
        positive = slope > 0.0
        lo = np.where(positive, mid, lo)
        hi = np.where(positive, hi, mid)
    return 0.5 * (lo + hi) * p.T


def consumption(pop: Population, hours: np.ndarray, tax: float, ubi: float) -> np.ndarray:
    """Consumption = non-labor income + UBI + after-tax labor income."""
    return pop.nonlabor + ubi + pop.wages * hours * (1.0 - tax)


def utility(pop: Population, hours: np.ndarray, tax: float, ubi: float, p: Params) -> np.ndarray:
    m = hours / p.T
    disutility = p.psi * p.frisch / (1.0 + p.frisch) * m ** (1.0 + 1.0 / p.frisch)
    return np.log(consumption(pop, hours, tax, ubi)) - disutility


def tax_revenue(pop: Population, tax: float, ubi: float, p: Params) -> float:
    return float(tax * np.sum(pop.wages * optimal_hours(pop, tax, ubi, p)))


class InfeasibleUBI(ValueError):
    """No flat tax rate can raise enough revenue to fund this UBI."""

    def __init__(self, ubi: float, max_revenue: float, target: float):
        super().__init__(
            f"UBI of {ubi:,.0f} is infeasible: the most revenue any flat tax can raise "
            f"is {max_revenue:,.0f}, below the required {target:,.0f}."
        )
        self.ubi, self.max_revenue, self.target = ubi, max_revenue, target


def laffer_peak(pop: Population, ubi: float, p: Params, grid: int = 60) -> tuple[float, float]:
    """(tax rate, revenue) at the top of the Laffer curve for this UBI level."""

    def neg_revenue(t: float) -> float:
        return -tax_revenue(pop, t, ubi, p)

    ts = np.linspace(1e-3, 0.999, grid)
    k = int(np.argmin([neg_revenue(t) for t in ts]))
    peak = minimize_scalar(
        neg_revenue,
        bounds=(ts[max(k - 1, 0)], ts[min(k + 1, grid - 1)]),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(peak.x), float(-peak.fun)


def revenue_neutral_tax(pop: Population, ubi: float, target: float, p: Params) -> float:
    """Flat tax rate at which revenue equals ``target``.

    Revenue is a Laffer curve, so a plain bisection on [0, 1] is invalid (the
    ends have the same sign). The peak is located first and the root is
    bracketed on the rising branch. Raises ``InfeasibleUBI`` if the target is
    above the peak.
    """
    t_peak, peak_revenue = laffer_peak(pop, ubi, p)
    if peak_revenue < target:
        raise InfeasibleUBI(ubi, peak_revenue, target)
    return float(brentq(lambda t: tax_revenue(pop, t, ubi, p) - target, 1e-6, t_peak, xtol=1e-12))


def run_scenarios(
    pop: Population,
    p: Params,
    ubi_levels: dict[str, float] | None = None,
    skip_infeasible: bool = False,
) -> dict[str, Scenario]:
    """Solve every UBI level with its own revenue-neutral tax and its own hours.

    Infeasible UBI levels raise ``InfeasibleUBI`` unless ``skip_infeasible`` is
    set, in which case they are left out of the result.
    """
    ubi_levels = UBI_LEVELS if ubi_levels is None else ubi_levels
    base_revenue = tax_revenue(pop, p.base_tax, 0.0, p)
    out: dict[str, Scenario] = {}
    for label, ubi in ubi_levels.items():
        target = base_revenue + p.n_agents * ubi
        try:
            tax = p.base_tax if ubi == 0 else revenue_neutral_tax(pop, ubi, target, p)
        except InfeasibleUBI:
            if skip_infeasible:
                continue
            raise
        hours = optimal_hours(pop, tax, ubi, p)
        out[label] = Scenario(
            label=label,
            ubi=ubi,
            tax=tax,
            hours=hours,
            consumption=consumption(pop, hours, tax, ubi),
            utility=utility(pop, hours, tax, ubi, p),
            revenue=float(tax * np.sum(pop.wages * hours)),
            target_revenue=target,
        )
    return out


def calibrate_psi(pop: Population, p: Params, target_share: float = 0.40) -> float:
    """psi such that the average agent works ``target_share`` of T at baseline.

    With psi = 1 (the default) agents work ~99.8% of their time, so the
    labor-leisure trade-off is almost absent. This gives a sensitivity check.
    """

    def gap(log_psi: float) -> float:
        q = replace(p, psi=10.0**log_psi)
        return float(optimal_hours(pop, p.base_tax, 0.0, q).mean() / p.T - target_share)

    return float(10.0 ** brentq(gap, -8.0, 8.0, xtol=1e-10))


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    return float((2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1.0) / n)


def summarize(scenarios: dict[str, Scenario], p: Params) -> list[dict]:
    """One row of headline metrics per scenario."""
    base = next(iter(scenarios.values()))
    base_hours = base.hours.sum()
    base_u = base.utility.mean()
    rows = []
    for s in scenarios.values():
        k = max(1, s.utility.size // 10)
        rows.append(
            {
                "scenario": s.label,
                "ubi": s.ubi,
                "tax_rate_pct": 100 * s.tax,
                "total_hours": s.hours.sum(),
                "mean_hours": s.hours.mean(),
                "labor_cost_index": base_hours / s.hours.sum(),
                "tax_revenue": s.revenue,
                "revenue_gap_pct": 100 * (s.revenue - s.target_revenue) / s.target_revenue,
                "mean_consumption": s.consumption.mean(),
                "gini_consumption": gini(s.consumption),
                "total_utility": s.utility.sum(),
                "mean_utility": s.utility.mean(),
                "median_utility": float(np.median(s.utility)),
                "sd_utility": s.utility.std(),
                "bottom_decile_utility": float(np.sort(s.utility)[:k].mean()),
                "mean_utility_change_vs_no_ubi": s.utility.mean() - base_u,
            }
        )
    return rows
