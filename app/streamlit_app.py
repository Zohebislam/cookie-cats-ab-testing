"""Cookie Cats A/B test explorer.

    streamlit run app/streamlit_app.py

Four tabs:
1. The result      what the experiment found, in plain language
2. The peeking trap   change how often you check results and watch false wins rise
3. Plan a test     how many players a new test needs
4. Glossary        every term in one sentence
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abtest import bayesian, frequentist, power, simulate  # noqa: E402
from abtest.data import group_counts, load  # noqa: E402
import abtest.plots  # noqa: E402,F401  (loads the shared chart style, overridden below)

# Dark theme. Series colors are the dark-mode steps of the project palette.
BG = "#0f1420"        # page background (matches .streamlit/config.toml)
PANEL = "#182032"     # cards and boxes
GRID = "#283149"
INK = "#f0efec"       # main text
INK_2 = "#b4b6bf"     # secondary text
GRAY = "#6f7482"
BLUE = "#3987e5"      # gate at level 30
ORANGE = "#e8743b"    # gate at level 40, highlights

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.labelcolor": INK_2,
    "axes.titlecolor": INK, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
})

st.set_page_config(page_title="Cookie Cats A/B Test", page_icon="🐱", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    .hero {background: linear-gradient(135deg, #2a1c24 0%, #182540 60%, #13203a 100%);
           border: 1px solid #283149; border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;}
    .hero h1 {font-size: 2.1rem; margin: 0 0 .4rem 0; color: #f0efec;}
    .hero p {font-size: 1.05rem; margin: 0; color: #c9cad1;}
    .verdict {border-left: 6px solid #3987e5; background: #15233d; border-radius: 10px; color: #f0efec;
              padding: 1rem 1.2rem; margin: .6rem 0 1rem 0; font-size: 1.1rem;}
    .warn {border-left: 6px solid #e8743b; background: #2a1d18; border-radius: 10px; color: #f0efec;
           padding: 1rem 1.2rem; margin: .6rem 0 1rem 0;}
    .big-number {font-size: 3.4rem; font-weight: 700; line-height: 1; margin: .2rem 0;}
    .muted {color: #b4b6bf;}
    div[data-testid="stMetric"] {background: #182032; border: 1px solid #283149;
                                 border-radius: 12px; padding: .8rem 1rem;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🐱 Should the game make players wait at level 30 or level 40?</h1>
      <p>Cookie Cats is a puzzle game. At a "gate", players must wait or pay to keep going.
      90,189 new players were randomly split: half hit the gate at level 30, half at level 40.
      This app shows what happened, and why checking results too early can fool you.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data (cached so the app stays fast)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def experiment_results():
    df = load()
    out = {}
    for metric in ("retention_1", "retention_7"):
        c = group_counts(df, metric)
        out[metric] = {**c, **frequentist.two_prop_ztest(c["x_a"], c["n_a"], c["x_b"], c["n_b"])}
        out[metric]["bayes"] = bayesian.summarize(c["x_a"], c["n_a"], c["x_b"], c["n_b"], draws=200_000)
    c7 = out["retention_7"]
    out["srm"] = frequentist.srm_test(c7["n_a"], c7["n_b"])
    return out


@st.cache_data(show_spinner=False)
def peeking_study(base_rate, n_per_group, looks, effect, sims):
    return simulate.sequential_study(base_rate, base_rate + effect, n_per_group, n_per_group,
                                     n_looks=looks, n_sims=sims, seed=42)


@st.cache_data(show_spinner=False)
def fake_paths(base_rate, n_per_group, looks, n_paths):
    return simulate.example_paths(base_rate, n_per_group, n_looks=looks, n_paths=n_paths, seed=5)


R = experiment_results()
r7, r1 = R["retention_7"], R["retention_1"]

PLAIN_NAMES = {
    "fixed": "Wait until the end, check once",
    "naive_peeking": "Check every day, stop at the first 'significant' result",
    "bayes_peeking": "Check every day, stop when 97.5% sure (Bayesian)",
    "bayes_peeking_95": "Check every day, stop when 95% sure (Bayesian)",
    "gsd_obf": "Planned check-ins, strict early (O'Brien-Fleming)",
    "gsd_pocock": "Planned check-ins, same bar each time (Pocock)",
    "msprt": "Check any time, extra cautious (mSPRT)",
}

tab_result, tab_peek, tab_plan, tab_terms = st.tabs(
    ["📊 The result", "🙈 The peeking trap", "🧮 Plan a test", "📖 Glossary"]
)

# ---------------------------------------------------------------------------
# 1. The result
# ---------------------------------------------------------------------------
with tab_result:
    st.markdown(
        f"""<div class="verdict">✅ <b>Keep the gate at level 30.</b> Moving it to level 40 meant
        fewer players were still playing a week later.</div>""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Players still playing 7 days after installing")
    c1, c2, c3 = st.columns(3)
    c1.metric("Gate at level 30 (original)", f"{r7['p_a']:.2%}")
    c2.metric("Gate at level 40 (new)", f"{r7['p_b']:.2%}", f"{r7['diff'] * 100:+.2f} pts", delta_color="normal")
    c3.metric("Chance level 30 is truly better", f"{r7['bayes']['prob_a_better']:.1%}")

    st.markdown(
        f"""<p class="muted">That's about <b>{abs(r7['diff']) * 1000:.0f} fewer returning players
        for every 1,000 who install</b>. Small, but for a game with millions of players it adds up.</p>""",
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels, rows = ["Day-1 retention", "Day-7 retention"], [r1, r7]
    x = np.arange(2)
    for j, (key, n_key, color, name) in enumerate([("p_a", "n_a", BLUE, "Gate at level 30"),
                                                   ("p_b", "n_b", ORANGE, "Gate at level 40")]):
        rates = np.array([r[key] for r in rows])
        err = 1.96 * np.sqrt(rates * (1 - rates) / np.array([r[n_key] for r in rows]))
        pos = x + (j - 0.5) * 0.38
        ax.bar(pos, rates * 100, 0.36, color=color, label=name, zorder=3)
        ax.errorbar(pos, rates * 100, yerr=err * 100, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=4)
        for xi, r in zip(pos, rates):
            ax.text(xi, r * 100 + 1.5, f"{r:.1%}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Players who came back (%)")
    ax.set_ylim(0, 55)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK)
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("#### How do we know it's not just luck?")
    a, b = st.columns(2)
    with a:
        st.markdown(
            f"""
            - **p-value = {r7['p_value']:.4f}.** If both gates were truly equal, a gap this big would
              show up by luck only about {r7['p_value'] * 1000:.0f} times in 1,000.
            - **The likely range for the true drop is {abs(r7['ci_high']) * 100:.2f} to
              {abs(r7['ci_low']) * 100:.2f} points.** That range never reaches zero.
            """
        )
    with b:
        st.markdown(
            f"""
            - **Day-1 retention** also dropped ({r1['diff'] * 100:+.2f} pts), but the test wasn't big
              enough to be sure about a change that small (p = {r1['p_value']:.3f}).
            - **Bayesian view:** {r7['bayes']['prob_a_better']:.2%} probability that level 30 is better
              at keeping players for a week.
            """
        )

    srm = R["srm"]
    st.markdown(
        f"""<div class="warn">⚠️ <b>One thing to double-check.</b> The two groups should be about
        the same size, but level 40 has {r7['n_b'] - r7['n_a']:,} more players. A fair 50/50 split
        would produce a gap this big less than 1% of the time (p = {srm['p_value']:.3f}), which can
        mean some players were logged wrong. Even in the worst case, level 40 still doesn't come
        out ahead, but the company should check its tracking.</div>""",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# 2. The peeking trap
# ---------------------------------------------------------------------------
with tab_peek:
    st.markdown("#### What if you check the results every day and stop as soon as one side 'wins'?")
    st.markdown(
        """Standard tests promise only a **5% chance of a fake win** if you look once at the end.
        Below, both versions are **exactly the same**, so every "win" is fake.
        Move the slider and watch what happens when you check more often."""
    )

    base = (r7["x_a"] + r7["x_b"]) / (r7["n_a"] + r7["n_b"])
    n_group = (r7["n_a"] + r7["n_b"]) // 2
    looks = st.slider("How many times do you check the results?", min_value=1, max_value=30, value=14,
                      help="14 = once a day for two weeks")

    with st.spinner("Running 5,000 fake experiments..."):
        null = peeking_study(base, n_group, looks, 0.0, 5_000).set_index("method")
    naive = null.loc["naive_peeking", "reject_rate"]
    color = BLUE if naive <= 0.06 else ORANGE

    left, right = st.columns([1, 2])
    with left:
        st.markdown(
            f"""<p class="muted" style="margin-bottom:0">Chance of declaring a fake winner</p>
            <div class="big-number" style="color:{color}">{naive:.0%}</div>
            <p class="muted">That's about <b>1 in {max(1, round(1 / max(naive, 1e-9)))}</b> tests "finding" a
            difference that doesn't exist.<br>The promise was 5% (1 in 20).</p>""",
            unsafe_allow_html=True,
        )
    with right:
        if looks >= 2:
            paths = fake_paths(base, n_group, looks, 30)
            days = np.arange(1, looks + 1)
            fooled = (np.abs(paths) >= 1.96).any(axis=1)
            fig, ax = plt.subplots(figsize=(7.5, 3.6))
            for p, hit in zip(paths, fooled):
                ax.plot(days, p, color=ORANGE if hit else GRAY, lw=1.5 if hit else 1, alpha=0.95 if hit else 0.55)
            ax.axhline(1.96, color=INK, ls="--", lw=1)
            ax.axhline(-1.96, color=INK, ls="--", lw=1)
            ax.text(days[-1], 2.15, "'significant' line", ha="right", color=INK_2, fontsize=9)
            ax.set_ylim(-4.5, 4.5)
            ax.set_xlabel("Check-in number")
            ax.set_ylabel("Test score (z)")
            ax.set_title(f"30 identical tests: {fooled.sum()} crossed the line at least once (orange)", fontsize=11)
            fig.tight_layout()
            st.pyplot(fig)
            st.caption("A sample of 30 tests. The big number on the left comes from 5,000.")
        else:
            st.info("With one check at the end, there's nothing to peek at. Move the slider up.")

    st.divider()
    st.markdown("#### How to check early without fooling yourself")
    st.markdown(
        """Each rule below was tried on the same simulated tests. A rule is **safe** if its fake-win
        rate stays at about 5% or lower. "Catches a real drop" uses the real Cookie Cats drop of 0.82
        points."""
    )
    with st.spinner("Testing each method..."):
        real = peeking_study(base, n_group, looks, r7["diff"], 5_000).set_index("method")
    order = ["fixed", "naive_peeking", "bayes_peeking", "bayes_peeking_95", "gsd_obf", "gsd_pocock", "msprt"]
    rows = []
    for m in order:
        fp, se = null.loc[m, "reject_rate"], null.loc[m, "reject_rate_se"]
        safe = fp <= 0.05 + 3 * se
        rows.append({
            "Safe?": "✅ Safe" if safe else "❌ Fooled",
            "How you check": PLAIN_NAMES[m],
            "Fake wins": f"{fp:.1%}",
            "Catches a real drop": f"{real.loc[m, 'reject_rate']:.0%}" if safe else "—",
            "Players needed (avg)": f"{real.loc[m, 'avg_sample_used']:.0%}" if safe else "—",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Unsafe rules show '—' for the other columns: they look fast and powerful only because "
               "they also 'find' effects that aren't there.")

    with st.expander("💡 What's the takeaway?"):
        st.markdown(
            """
            - Peeking isn't a Bayesian vs. frequentist problem. **Any** rule that says "stop as soon as
              it looks good" gets fooled more often the more you check.
            - **Planned check-ins (O'Brien-Fleming)** are the best deal here: almost as good at catching
              real changes as waiting until the end, and they usually finish early when the effect is real.
            - **mSPRT** lets you check whenever you want, but it's more cautious, so it misses more real effects.
            """
        )

# ---------------------------------------------------------------------------
# 3. Plan a test
# ---------------------------------------------------------------------------
with tab_plan:
    st.markdown("#### How many players does your test need?")
    st.markdown("Fill in the boxes. The defaults match Cookie Cats day-7 retention.")

    c1, c2 = st.columns(2)
    base_pct = c1.number_input(
        "Current rate (%)", min_value=0.5, max_value=99.0, value=19.0, step=0.5,
        help="For example, the % of players who come back after 7 days today.",
    )
    change_pts = c2.number_input(
        "Smallest change you care about (percentage points)", min_value=0.05, max_value=20.0, value=1.0, step=0.05,
        help="Smaller changes need many more players to detect.",
    )
    c3, c4, c5 = st.columns(3)
    daily = c3.number_input("New players per day", min_value=100, value=6_000, step=100,
                            help="Both groups combined.")
    target_power = c4.select_slider("How sure you want to be to catch it", options=[0.80, 0.90, 0.95], value=0.80,
                                    format_func=lambda v: f"{v:.0%}",
                                    help="This is called power. 80% is the usual choice.")
    alpha = c5.select_slider("Allowed chance of a fake win", options=[0.01, 0.05, 0.10], value=0.05,
                             format_func=lambda v: f"{v:.0%}",
                             help="Called alpha. 5% is the usual choice.")

    p0, delta = base_pct / 100, change_pts / 100
    if p0 + delta >= 1:
        st.error("The current rate plus the change has to stay below 100%.")
    else:
        n_arm = power.sample_size_per_arm(p0, delta, alpha, target_power)
        days = 2 * n_arm / daily
        m1, m2, m3 = st.columns(3)
        m1.metric("Players per group", f"{n_arm:,}")
        m2.metric("Players in total", f"{2 * n_arm:,}")
        m3.metric("Days to run", f"{days:.1f}")
        st.markdown(
            f"""<div class="verdict">You need about <b>{2 * n_arm:,} players</b> (around <b>{days:.0f}
            days</b>) to have a {target_power:.0%} chance of catching a change from {base_pct:.1f}% to
            {base_pct + change_pts:.2f}% or {base_pct - change_pts:.2f}%.</div>""",
            unsafe_allow_html=True,
        )

        changes = [c for c in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0] if p0 + c / 100 < 1]
        needed = [power.sample_size_per_arm(p0, c / 100, alpha, target_power) * 2 / daily for c in changes]
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar([f"{c} pts" for c in changes], needed, color=BLUE, width=0.6, zorder=3)
        for i, d in enumerate(needed):
            ax.text(i, d, f"{d:.0f} day" + ("" if round(d) == 1 else "s"), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_ylabel("Days needed")
        ax.set_ylim(0, max(needed) * 1.15)
        ax.set_title("Smaller changes take much longer to detect", fontsize=11)
        ax.grid(axis="x", visible=False)
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Halving the change you want to detect roughly quadruples the players you need.")

# ---------------------------------------------------------------------------
# 4. Glossary
# ---------------------------------------------------------------------------
with tab_terms:
    st.markdown("#### Every term, in one sentence")
    terms = {
        "A/B test": "Randomly split users into two groups, show each a different version, and compare results.",
        "Retention": "The share of players who came back and played 1 day or 7 days after installing.",
        "pts (percentage points)": "The plain difference between two percentages: 19.02% to 18.20% is a drop of 0.82 pts.",
        "p-value": "If there were truly no difference, how often you'd see a gap this big by luck. Under 0.05 is usually called 'significant'.",
        "Fake win (false positive)": "Declaring a winner when there's no real difference.",
        "Confidence interval": "A range that likely contains the true difference.",
        "Power": "The chance a test catches a real effect. 80% is the usual target.",
        "Bayesian analysis": "Gives the probability that one version is better, instead of a p-value.",
        "Sample ratio mismatch": "The groups aren't the sizes a fair random split should produce, which can point to a tracking bug.",
        "Peeking": "Checking results before a test ends and stopping as soon as it looks significant.",
        "A/A test": "A test where both groups get the same thing, so any 'winner' is fake.",
        "Sequential testing": "Planning check-ins ahead of time with stricter rules early on, so peeking stays safe.",
        "O'Brien-Fleming": "A sequential plan that is very strict early and close to normal at the end.",
        "Pocock": "A sequential plan with the same bar at every check-in.",
        "mSPRT": "A method that lets you check any time, as often as you like, at the cost of catching fewer real effects.",
    }
    for term, meaning in terms.items():
        st.markdown(f"**{term}**: {meaning}")
