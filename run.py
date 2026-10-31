"""Run the whole project: analysis, simulations, figures and the findings report.

    python run.py            full run (about 1 minute)
    python run.py --quick    fewer simulations, for a fast check
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from abtest import analysis, frequentist, plots, report  # noqa: E402
from abtest.data import METRICS, load  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="run fewer simulations")
    args = parser.parse_args()
    n_sims, n_perm, n_boot = (5_000, 300, 20_000) if args.quick else (50_000, 2_000, 100_000)

    for d in (RESULTS, FIGURES, REPORTS):
        d.mkdir(exist_ok=True)

    start = time.time()
    df = load()
    print(f"Loaded {len(df):,} players")

    main_res = analysis.main_analysis(df, n_boot=n_boot)
    print("Main analysis done")
    peek = analysis.peeking_studies(df, main_res, n_sims=n_sims, n_perm=n_perm)
    print("Simulations done")

    # ---- tables ----
    tests = pd.DataFrame(main_res["tests"]).T
    tests.index.name = "metric"
    tests.to_csv(RESULTS / "retention_tests.csv")
    pd.DataFrame(main_res["power"]).T.rename_axis("metric").to_csv(RESULTS / "power.csv")
    bayes_tbl = pd.DataFrame(main_res["bayes"]).T.drop(columns=["posterior_a", "posterior_b"])
    bayes_tbl.rename_axis("metric").to_csv(RESULTS / "bayesian.csv")
    peek["boundaries"].to_csv(RESULTS / "sequential_boundaries.csv", index=False)
    peek["aa"].to_csv(RESULTS / "peeking_false_positives.csv", index=False)
    peek["power"].to_csv(RESULTS / "peeking_power.csv", index=False)
    peek["fpr_vs_looks"].to_csv(RESULTS / "fpr_vs_looks.csv", index=False)

    summary = {
        "quality": main_res["quality"],
        "srm": main_res["srm"],
        "srm_worst_case": main_res["srm_worst_case"],
        "rounds": main_res["rounds"],
        "sequential_on_real_data": main_res["sequential_on_real_data"],
        "permutation_aa": peek["permutation_aa"],
        "simulation_settings": peek["settings"],
    }
    (RESULTS / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2))

    # ---- figures ----
    boot = {}
    for i, m in enumerate(METRICS):
        t = main_res["tests"][m]
        boot[m] = frequentist.bootstrap_diff(t["x_a"], t["n_a"], t["x_b"], t["n_b"], n_boot, analysis.SEED + i)
    plots.retention_bars(main_res["tests"], FIGURES / "01_retention_by_group.png")
    plots.bootstrap_hist(boot, main_res["tests"], FIGURES / "02_bootstrap_difference.png")
    plots.fpr_vs_looks(peek["fpr_vs_looks"], FIGURES / "03_peeking_false_positive_rate.png")
    plots.peeking_paths(peek["example_paths"], peek["boundaries"], FIGURES / "04_peeking_paths.png")
    plots.method_comparison(peek["aa"], peek["power"], FIGURES / "05_method_comparison.png")
    plots.posterior_plot(main_res["bayes"], main_res["tests"], FIGURES / "06_bayesian_posterior.png")
    plots.rounds_ecdf(df, FIGURES / "07_game_rounds.png")
    print("Figures saved")

    # ---- report ----
    (REPORTS / "FINDINGS.md").write_text(report.build(main_res, peek))
    print(f"Report written. Total time {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
