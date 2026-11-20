"""Figures for the UBI simulation. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from .model import Params, Population, Scenario, tax_revenue

COLORS = {"No UBI": "#4C78A8", "UBI Low": "#F58518", "UBI Mid": "#B279A2", "UBI High": "#E45756"}
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _label(s: Scenario) -> str:
    return f"{s.label} (B = {s.ubi:,.0f}, t = {100 * s.tax:.1f}%)"


def tax_rates(scenarios: dict[str, Scenario], out_dir: Path) -> Path:
    """Revenue-neutral flat tax rate for each UBI level."""
    labels = list(scenarios)
    rates = [100 * scenarios[l].tax for l in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(labels)), rates, color=[COLORS[l] for l in labels], width=0.55)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.8, f"{r:.1f}%", ha="center", fontsize=10)
    ax.set_xticks(range(len(labels)), [f"{l}\n(B = {scenarios[l].ubi:,.0f})" for l in labels])
    ax.set(ylabel="Flat tax rate (%)", title="Tax rate needed to fund baseline revenue plus the UBI", ylim=(0, 52))
    return _save(fig, out_dir, "tax_rates.png")


def wage_distribution(pop: Population, p: Params, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pop.wages, bins=40, density=True, alpha=0.75, color="#4C78A8")
    ax.axvline(pop.wages.mean(), ls="--", color="k", label=f"Mean wage: {pop.wages.mean():.1f}")
    ax.set(xlabel="Hourly wage", ylabel="Density", title=f"Simulated wages ({p.n_agents:,} agents, lognormal)")
    ax.legend()
    return _save(fig, out_dir, "wage_distribution.png")


def hours_vs_wage(pop: Population, scenarios: dict[str, Scenario], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scenarios.values():
        ax.scatter(pop.wages, s.hours, s=10, alpha=0.35, color=COLORS[s.label], label=_label(s))
    ax.set(xlabel="Hourly wage", ylabel="Hours worked", title="Labor supply by wage")
    ax.legend(markerscale=2)
    return _save(fig, out_dir, "hours_vs_wage.png")


def consumption_vs_wage(pop: Population, scenarios: dict[str, Scenario], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scenarios.values():
        ax.scatter(pop.wages, s.consumption, s=10, alpha=0.35, color=COLORS[s.label], label=_label(s))
    ax.set(xlabel="Hourly wage", ylabel="Consumption", title="Consumption by wage")
    ax.legend(markerscale=2)
    return _save(fig, out_dir, "consumption_vs_wage.png")


def _kde_panel(ax, scenarios, attr, xlabel, title):
    for s in scenarios.values():
        v = getattr(s, attr)
        xs = np.linspace(np.percentile(v, 0.5), np.percentile(v, 99.5), 300)
        ax.fill_between(xs, gaussian_kde(v)(xs), alpha=0.18, color=COLORS[s.label])
        ax.plot(xs, gaussian_kde(v)(xs), color=COLORS[s.label], label=_label(s))
    ax.set(xlabel=xlabel, ylabel="Density", title=title)
    ax.legend(fontsize=8)


def distributions(scenarios: dict[str, Scenario], out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    _kde_panel(axes[0], scenarios, "hours", "Hours worked", "Hours worked")
    _kde_panel(axes[1], scenarios, "consumption", "Consumption", "Consumption")
    _kde_panel(axes[2], scenarios, "utility", "Utility", "Utility")
    return _save(fig, out_dir, "distributions.png")


def laffer(pop: Population, p: Params, scenarios: dict[str, Scenario], out_dir: Path) -> Path:
    """Tax revenue as a function of the tax rate: why the solver brackets the peak."""
    ts = np.linspace(0.01, 0.99, 120)
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in scenarios.values():
        rev = [tax_revenue(pop, t, s.ubi, p) / 1e6 for t in ts]
        ax.plot(ts * 100, rev, color=COLORS[s.label], label=f"Revenue, B = {s.ubi:,.0f}")
        ax.axhline(s.target_revenue / 1e6, color=COLORS[s.label], ls=":", lw=1)
        ax.plot(100 * s.tax, s.target_revenue / 1e6, "o", color=COLORS[s.label])
    ax.set(xlabel="Flat tax rate (%)", ylabel="Tax revenue (millions)", title="Laffer curves and required tax rates")
    ax.legend()
    return _save(fig, out_dir, "laffer_curves.png")


def seed_robustness(delta: dict[str, dict[str, np.ndarray]], out_dir: Path) -> Path:
    """Change in mean utility vs No UBI across random populations, both calibrations."""
    fig, axes = plt.subplots(1, len(delta), figsize=(5.5 * len(delta), 4.8), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, (name, by_label) in zip(axes, delta.items()):
        keys = list(by_label)
        feasible = [by_label[k][~np.isnan(by_label[k])] for k in keys]
        ticks = []
        for k, v in zip(keys, feasible):
            share = len(v) / len(by_label[k])
            ticks.append(k if share == 1 else f"{k}\n({share:.0%} feasible)")
        bp = ax.boxplot([v if len(v) else [np.nan] for v in feasible], patch_artist=True, showfliers=False)
        ax.set_xticks(range(1, len(keys) + 1), ticks)
        for patch, k in zip(bp["boxes"], keys):
            patch.set_facecolor(COLORS[k])
            patch.set_alpha(0.6)
        ax.axhline(0, color="k", lw=1)
        ax.set(ylabel="Change in mean utility vs No UBI", title=name)
    return _save(fig, out_dir, "seed_robustness.png")
