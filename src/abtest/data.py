"""Load and validate the Cookie Cats A/B test data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "cookie_cats.csv"

CONTROL = "gate_30"    # original design: gate at level 30
TREATMENT = "gate_40"  # variant: gate moved to level 40
METRICS = ("retention_1", "retention_7")

EXPECTED_COLUMNS = {"userid", "version", "sum_gamerounds", "retention_1", "retention_7"}


def load(path: str | Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV and check that it has the shape this project expects."""
    df = pd.read_csv(path)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    groups = set(df["version"].unique())
    if groups != {CONTROL, TREATMENT}:
        raise ValueError(f"unexpected groups: {sorted(groups)}")

    for col in METRICS:
        # The CSV stores True/False; make sure they are real booleans.
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})
            if df[col].isna().any():
                raise ValueError(f"{col} has values that are not True/False")

    if (df["sum_gamerounds"] < 0).any():
        raise ValueError("sum_gamerounds has negative values")

    return df


def quality_report(df: pd.DataFrame) -> dict:
    """Basic data checks reported before any testing."""
    rounds = df["sum_gamerounds"]
    top = df.nlargest(1, "sum_gamerounds").iloc[0]
    second = rounds.nlargest(2).iloc[1]
    return {
        "rows": int(len(df)),
        "duplicate_userids": int(df["userid"].duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
        "players_control": int((df["version"] == CONTROL).sum()),
        "players_treatment": int((df["version"] == TREATMENT).sum()),
        "zero_round_players": int((rounds == 0).sum()),
        "zero_round_share": float((rounds == 0).mean()),
        "rounds_median": float(rounds.median()),
        "rounds_p99": float(rounds.quantile(0.99)),
        "rounds_max": int(top["sum_gamerounds"]),
        "rounds_max_group": str(top["version"]),
        "rounds_second_max": int(second),
        # A player who retained on day 7 but not day 1 is possible (they came back later),
        # so this is reported, not treated as an error.
        "retained_7_not_1": int((df["retention_7"] & ~df["retention_1"]).sum()),
    }


def group_counts(df: pd.DataFrame, metric: str) -> dict:
    """Successes and sample sizes for one metric, per group."""
    g = df.groupby("version")[metric]
    return {
        "x_a": int(g.sum()[CONTROL]),
        "n_a": int(g.size()[CONTROL]),
        "x_b": int(g.sum()[TREATMENT]),
        "n_b": int(g.size()[TREATMENT]),
    }
