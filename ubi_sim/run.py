"""Reproduce every number and figure in the README.

    python -m ubi_sim.run                # full run (a few minutes)
    python -m ubi_sim.run --seeds 0      # skip the multi-seed robustness check
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import plots
from .model import (
    SENSITIVITY_LEVELS,
    Params,
    calibrate_psi,
    laffer_peak,
    make_population,
    run_scenarios,
    summarize,
)

TARGET_SHARE = 0.40  # average share of time worked in the recalibrated model
CAL_NAMES = ("Baseline calibration (psi = 1)", f"Recalibrated ({TARGET_SHARE:.0%} of time worked)")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)


def _md_table(rows: list[dict], cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[1] for c in cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    def cell(v, fmt: str) -> str:
        if isinstance(v, str):
            return v
        return "n/a" if np.isnan(v) else format(v, fmt)

    body = "".join("| " + " | ".join(cell(r[k], fmt) for k, _, fmt in cols) + " |\n" for r in rows)
    return head + body


def seed_study(p: Params, seeds: range) -> dict[str, dict[str, np.ndarray]]:
    """Change in mean utility vs No UBI over many random populations, both calibrations.

    Levels that no flat tax can fund in a given population are recorded as NaN.
    """
    levels = [l for l in SENSITIVITY_LEVELS if l != "No UBI"]
    out = {n: {l: [] for l in levels} for n in CAL_NAMES}
    for seed in seeds:
        q = replace(p, seed=seed)
        pop = make_population(q)
        for name, params in zip(CAL_NAMES, (q, replace(q, psi=calibrate_psi(pop, q, TARGET_SHARE)))):
            scen = run_scenarios(pop, params, SENSITIVITY_LEVELS, skip_infeasible=True)
            base = scen["No UBI"].utility.mean()
            for l in levels:
                out[name][l].append(scen[l].utility.mean() - base if l in scen else np.nan)
    return {n: {l: np.array(v) for l, v in d.items()} for n, d in out.items()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=100, help="number of random populations for the robustness check")
    ap.add_argument("--out", type=Path, default=Path("."), help="repository root to write figures/ and results/ into")
    args = ap.parse_args(argv)
    fig_dir, res_dir = args.out / "figures", args.out / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    p = Params()
    pop = make_population(p)

    # 1) Baseline calibration (psi = 1)
    scen = run_scenarios(pop, p)
    rows = summarize(scen, p)
    _write_csv(res_dir / "summary_baseline_calibration.csv", rows)

    # 2) Recalibrated labor-leisure trade-off (sensitivity check, adds a middle level)
    psi = calibrate_psi(pop, p, TARGET_SHARE)
    p_cal = replace(p, psi=psi)
    scen_cal = run_scenarios(pop, p_cal, SENSITIVITY_LEVELS, skip_infeasible=True)
    rows_cal = summarize(scen_cal, p_cal)
    _write_csv(res_dir / "summary_recalibrated.csv", rows_cal)
    infeasible = []
    for label, ubi in SENSITIVITY_LEVELS.items():
        if label not in scen_cal:
            t_peak, rev_peak = laffer_peak(pop, ubi, p_cal)
            target = scen_cal["No UBI"].revenue + p.n_agents * ubi
            infeasible.append((label, ubi, rev_peak, target, t_peak))

    # 3) Figures
    plots.tax_rates(scen, fig_dir)
    plots.wage_distribution(pop, p, fig_dir)
    plots.hours_vs_wage(pop, scen, fig_dir)
    plots.consumption_vs_wage(pop, scen, fig_dir)
    plots.distributions(scen, fig_dir)
    plots.laffer(pop, p, scen, fig_dir)

    # 4) Robustness over random populations
    seed_rows = []
    if args.seeds > 0:
        seed_delta = seed_study(p, range(args.seeds))
        plots.seed_robustness(seed_delta, fig_dir)
        for name, by_label in seed_delta.items():
            for label, d in by_label.items():
                ok = d[~np.isnan(d)]
                seed_rows.append(
                    {
                        "calibration": name,
                        "scenario": label,
                        "share_feasible": ok.size / d.size,
                        "mean_change": float(ok.mean()) if ok.size else float("nan"),
                        "sd_change": float(ok.std(ddof=1)) if ok.size > 1 else float("nan"),
                        "share_positive": float((ok > 0).mean()) if ok.size else float("nan"),
                    }
                )
        _write_csv(res_dir / "seed_robustness.csv", seed_rows)

    # 5) Human-readable report
    cols = [
        ("scenario", "Scenario", ""),
        ("tax_rate_pct", "Tax rate (%)", ".2f"),
        ("total_hours", "Total hours", ",.0f"),
        ("labor_cost_index", "Labor cost index", ".3f"),
        ("revenue_gap_pct", "Revenue gap (%)", "+.6f"),
        ("mean_consumption", "Mean consumption", ",.0f"),
        ("gini_consumption", "Gini (consumption)", ".3f"),
        ("mean_utility", "Mean utility", ".3f"),
        ("sd_utility", "SD utility", ".3f"),
        ("bottom_decile_utility", "Bottom-decile utility", ".3f"),
        ("mean_utility_change_vs_no_ubi", "Δ mean utility", "+.4f"),
    ]
    md = (
        "# Results (generated by `python -m ubi_sim.run`)\n\n"
        f"Population: {p.n_agents} agents, seed {p.seed}.\n\n"
        "## Baseline calibration (psi = 1)\n\n" + _md_table(rows, cols) + "\n"
        f"## Recalibrated model (psi = {psi:.4g}; the average agent works {TARGET_SHARE:.0%} of the time endowment)\n\n"
        + _md_table(rows_cal, cols)
    )
    for label, ubi, rev_peak, target, t_peak in infeasible:
        md += (
            f"\n**{label} (B = {ubi:,.0f}) is infeasible**: the most revenue any flat tax can raise is "
            f"{rev_peak:,.0f} (at a {100 * t_peak:.0f}% rate), below the {target:,.0f} needed.\n"
        )
    if seed_rows:
        md += f"\n## Robustness over {args.seeds} random populations\n\n" + _md_table(
            seed_rows,
            [
                ("calibration", "Calibration", ""),
                ("scenario", "Scenario", ""),
                ("share_feasible", "Share of seeds feasible", ".0%"),
                ("mean_change", "Mean Δ utility vs No UBI", "+.4f"),
                ("sd_change", "SD across seeds", ".4f"),
                ("share_positive", "Share of seeds with Δ > 0", ".0%"),
            ],
        )
    (res_dir / "RESULTS.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
