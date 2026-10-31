"""Figures for the report. Each function saves one PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .data import CONTROL, TREATMENT

# Palette: categorical slots 1-2 (validated pair) plus neutral text and grid tones.
BLUE = "#2a78d6"      # gate_30 (control)
ORANGE = "#eb6834"    # gate_40 (treatment)
GRAY = "#8a8985"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
GROUP_COLORS = {CONTROL: BLUE, TREATMENT: ORANGE}
GROUP_NAMES = {CONTROL: "Gate at level 30 (control)", TREATMENT: "Gate at level 40 (variant)"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 10.5,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10.5,
        "legend.frameon": False,
        "figure.dpi": 110,
    }
)


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def retention_bars(tests: dict, out: Path) -> Path:
    """Retention by group with 95% confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    metrics = ["retention_1", "retention_7"]
    names = ["Day-1 retention", "Day-7 retention"]
    width = 0.36
    x = np.arange(len(metrics))
    for j, (group, key_p, key_n) in enumerate([(CONTROL, "p_a", "n_a"), (TREATMENT, "p_b", "n_b")]):
        rates = np.array([tests[m][key_p] for m in metrics])
        ns = np.array([tests[m][key_n] for m in metrics])
        err = 1.96 * np.sqrt(rates * (1 - rates) / ns)
        pos = x + (j - 0.5) * (width + 0.02)
        ax.bar(pos, rates * 100, width, color=GROUP_COLORS[group], label=GROUP_NAMES[group], zorder=3)
        ax.errorbar(pos, rates * 100, yerr=err * 100, fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
        for xi, r in zip(pos, rates):
            ax.text(xi, r * 100 + 1.6, f"{r:.1%}", ha="center", va="bottom", color=INK, fontsize=10)
    for xi, m in zip(x, metrics):
        t = tests[m]
        ax.text(xi, 54, f"diff {t['diff'] * 100:+.2f} pts\np = {t['p_value']:.4f}", ha="center", color=INK_2, fontsize=9.5)
    ax.set_xticks(x, names)
    ax.set_ylabel("Players who came back (%)")
    ax.set_ylim(0, 62)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.86))
    ax.set_title("Moving the gate to level 40 lowered retention")
    return _save(fig, out)


def bootstrap_hist(boot: dict, tests: dict, out: Path) -> Path:
    """Bootstrap distributions of the retention difference."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, (metric, title) in zip(axes, [("retention_1", "Day-1 retention"), ("retention_7", "Day-7 retention")]):
        d = boot[metric] * 100
        ax.hist(d, bins=80, color=ORANGE, alpha=0.85, zorder=3)
        ax.axvline(0, color=INK, lw=1.2, ls="--", zorder=4)
        t = tests[metric]
        ax.axvspan(t["boot_ci_low"] * 100, t["boot_ci_high"] * 100, color=GRAY, alpha=0.18, zorder=2)
        ax.set_title(title)
        ax.set_xlabel("gate_40 minus gate_30 (percentage points)")
        ax.set_yticks([])
        ax.text(
            0.02, 0.95,
            f"95% CI [{t['boot_ci_low'] * 100:.2f}, {t['boot_ci_high'] * 100:.2f}]\n"
            f"{t['boot_share_below_zero']:.1%} of resamples < 0",
            transform=ax.transAxes, va="top", color=INK_2, fontsize=9.5,
        )
    fig.suptitle("Bootstrap: how sure are we the difference is below zero?", x=0.01, ha="left", fontweight="bold", color=INK)
    return _save(fig, out)


def fpr_vs_looks(table: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(table["looks"], table["false_positive_rate"] * 100, color=BLUE, lw=2, marker="o", ms=6, zorder=3)
    ax.fill_between(
        table["looks"],
        (table["false_positive_rate"] - 1.96 * table["se"]) * 100,
        (table["false_positive_rate"] + 1.96 * table["se"]) * 100,
        color=BLUE, alpha=0.15, lw=0,
    )
    ax.axhline(5, color=INK, ls="--", lw=1.2)
    ax.text(table["looks"].max(), 5.8, "promised 5% false positive rate", ha="right", color=INK_2, fontsize=9.5)
    row14 = table.loc[table["looks"] == 14].iloc[0]
    ax.annotate(
        f"daily for 2 weeks: {row14['false_positive_rate']:.0%}",
        (14, row14["false_positive_rate"] * 100), xytext=(22, row14["false_positive_rate"] * 100 - 8),
        arrowprops=dict(arrowstyle="-", color=INK_2), color=INK, fontsize=10,
    )
    ax.set_xscale("log")
    ax.set_xticks(table["looks"], [str(v) for v in table["looks"]])
    ax.minorticks_off()
    ax.set_xlabel("Number of times you check the results (log scale)")
    ax.set_ylabel("False positive rate (%)")
    ax.set_ylim(0, max(45, table["false_positive_rate"].max() * 100 + 5))
    ax.set_title("The more you peek, the more 'wins' are just noise")
    return _save(fig, out)


def peeking_paths(paths: np.ndarray, boundaries: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    days = boundaries["day"].to_numpy()
    crossed = (np.abs(paths) >= 1.96).any(axis=1)
    for path, hit in zip(paths, crossed):
        ax.plot(days, path, color=ORANGE if hit else GRAY, alpha=0.9 if hit else 0.35, lw=1.4 if hit else 1.0, zorder=3 if hit else 2)
    for i, sign in enumerate((1, -1)):
        ax.plot(days, sign * boundaries["naive"], color=INK, ls="--", lw=1.2, label="Naive ±1.96 every day" if i == 0 else None)
        ax.plot(days, sign * boundaries["obf"], color=BLUE, lw=2, label="O'Brien-Fleming boundary" if i == 0 else None)
    ax.plot([], [], color=ORANGE, lw=1.4, label="Crossed ±1.96 at least once (false positive)")
    ax.plot([], [], color=GRAY, lw=1.0, label="Never crossed")
    ax.legend(loc="upper right", fontsize=9, facecolor=SURFACE, framealpha=1, frameon=True, edgecolor=GRID)
    ax.set_ylim(-6, 6)
    ax.set_xlim(1, days[-1])
    ax.set_xticks(days)
    ax.set_xlabel("Day of the experiment")
    ax.set_ylabel("z-statistic")
    ax.set_title(f"{len(paths)} experiments with NO real difference: {crossed.sum()} would be called 'significant'")
    return _save(fig, out)


def method_comparison(aa: pd.DataFrame, pw: pd.DataFrame, out: Path) -> Path:
    order = ["fixed", "naive_peeking", "bayes_peeking", "bayes_peeking_95", "gsd_obf", "gsd_pocock", "msprt"]
    short = {
        "fixed": "Fixed horizon (1 look)",
        "naive_peeking": "Naive daily peeking",
        "bayes_peeking": "Bayesian peeking, 97.5%",
        "bayes_peeking_95": "Bayesian peeking, 95%",
        "gsd_obf": "Group sequential (OBF)",
        "gsd_pocock": "Group sequential (Pocock)",
        "msprt": "mSPRT",
    }
    aa = aa.set_index("method").loc[order]
    pw = pw.set_index("method").loc[order]
    valid = aa["reject_rate"] <= 0.05 + 2 * aa["reject_rate_se"]
    colors = [BLUE if v else GRAY for v in valid]
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    panels = [
        ("False positive rate\n(no real difference)", aa["reject_rate"] * 100, 5, "%"),
        ("Power\n(if the true drop is 0.82 pts)", pw["reject_rate"] * 100, None, "%"),
        ("Share of planned sample used\n(when the drop is real)", pw["avg_sample_used"] * 100, None, "%"),
    ]
    for ax, (title, vals, ref, unit) in zip(axes, panels):
        ax.barh(y, vals, color=colors, height=0.62, zorder=3)
        for yi, v in zip(y, vals):
            x_text = max(v, ref if ref is not None else 0) + 1  # keep labels clear of the reference line
            ax.text(x_text, yi, f"{v:.1f}{unit}", va="center", color=INK, fontsize=9.5)
        if ref is not None:
            ax.axvline(ref, color=INK, ls="--", lw=1.2, zorder=4)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlim(0, max(vals.max() * 1.22, 10))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [short[m] for m in order])
    fig.suptitle(
        "Gray = breaks the 5% promise (its power and speed are not trustworthy). Blue = valid.",
        x=0.01, ha="left", color=INK_2, fontsize=10.5,
    )
    return _save(fig, out)


def posterior_plot(bayes: dict, tests: dict, out: Path) -> Path:
    b = bayes["retention_7"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(0.175, 0.200, 600)
    for group, key in [(CONTROL, "posterior_a"), (TREATMENT, "posterior_b")]:
        a, bb = b[key]
        axes[0].plot(grid * 100, stats.beta.pdf(grid, a, bb), color=GROUP_COLORS[group], lw=2, label=GROUP_NAMES[group])
        axes[0].fill_between(grid * 100, stats.beta.pdf(grid, a, bb), color=GROUP_COLORS[group], alpha=0.15)
    axes[0].set_xlabel("Day-7 retention (%)")
    axes[0].set_yticks([])
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.35)  # headroom so the legend clears the curves
    axes[0].set_title("Where the true day-7 rate probably is")

    rng = np.random.default_rng(0)
    (a_a, b_a), (a_b, b_b) = b["posterior_a"], b["posterior_b"]
    diff = (rng.beta(a_b, b_b, 200_000) - rng.beta(a_a, b_a, 200_000)) * 100
    axes[1].hist(diff, bins=90, color=ORANGE, alpha=0.85, zorder=3)
    axes[1].axvline(0, color=INK, ls="--", lw=1.2, zorder=4)
    axes[1].set_xlabel("gate_40 minus gate_30 (percentage points)")
    axes[1].set_yticks([])
    axes[1].set_title(f"P(gate_30 is better) = {b['prob_a_better']:.2%}")
    return _save(fig, out)


def rounds_ecdf(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for group in (CONTROL, TREATMENT):
        r = np.sort(df.loc[df["version"] == group, "sum_gamerounds"].to_numpy())
        ax.step(r + 1, np.arange(1, r.size + 1) / r.size * 100, where="post", color=GROUP_COLORS[group], lw=2, label=GROUP_NAMES[group])
    ax.set_xscale("log")
    ax.set_xlim(1, 3_000)
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 3000],
                  ["0", "1", "4", "9", "19", "49", "99", "199", "499", "999", "2,999"])
    ax.minorticks_off()
    ax.set_xlabel("Game rounds played in the first 14 days (log scale; one player with 49,854 rounds not shown)")
    ax.set_ylabel("Players with this many rounds or fewer (%)")
    ax.legend(loc="lower right")
    ax.set_title("Engagement curves are nearly identical")
    return _save(fig, out)
