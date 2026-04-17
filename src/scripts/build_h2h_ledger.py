"""Build a global batter–bowler H2H ledger from per-match parquet deliveries.

This is a one-time (or periodic) batch step to avoid scanning all match parquets
for every comparison query. Output is written as a Parquet file under
``data/processed/`` by default.

Example:
    python -m src.scripts.build_h2h_ledger --out data/processed/h2h_batter_bowler.parquet
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import get_project_paths, load_env

_REPO_ROOT = Path(__file__).resolve().parents[2]


_COLS = [
    "batter",
    "bowler",
    "runs_batter",
    "is_legal_delivery",
    "is_wicket",
    "wicket_player_out",
    "is_bowler_wicket",
    "is_super_over",
]


@dataclass(frozen=True, slots=True)
class _LedgerChunk:
    """Aggregated H2H fragment for a set of match files."""

    df: pd.DataFrame


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "data/processed/h2h_batter_bowler.parquet",
        help="Output parquet path",
    )
    p.add_argument(
        "--checkpoint",
        type=int,
        default=100,
        help="How many match files to merge per checkpoint",
    )
    return p.parse_args(argv)


def _agg_match_h2h(match_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batter–bowler rows for one match (super overs excluded)."""

    d = match_df.loc[:, _COLS].copy()
    d = d.loc[~d["is_super_over"].fillna(False).astype(bool)]
    d = d.loc[d["batter"].notna() & d["bowler"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["batter", "bowler", "balls", "runs", "dismissals"])
    d["legal"] = d["is_legal_delivery"].fillna(False).astype(bool)
    d["out"] = (
        d["is_wicket"].fillna(False).astype(bool)
        & d["is_bowler_wicket"].fillna(False).astype(bool)
        & (d["wicket_player_out"].astype(str) == d["batter"].astype(str))
    )
    g = d.groupby(["batter", "bowler"], as_index=False).agg(
        balls=("legal", "sum"),
        runs=("runs_batter", "sum"),
        dismissals=("out", "sum"),
    )
    return g


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Sum balls/runs/dismissals across fragments."""

    if not parts:
        return pd.DataFrame(columns=["batter", "bowler", "balls", "runs", "dismissals"])
    x = pd.concat(parts, ignore_index=True)
    if x.empty:
        return pd.DataFrame(columns=["batter", "bowler", "balls", "runs", "dismissals"])
    return (
        x.groupby(["batter", "bowler"], as_index=False)
        .agg(balls=("balls", "sum"), runs=("runs", "sum"), dismissals=("dismissals", "sum"))
        .reset_index(drop=True)
    )


def build_ledger(matches_dir: Path, checkpoint: int) -> pd.DataFrame:
    """Scan match parquets and build the global ledger."""

    files = sorted(matches_dir.glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    merged: list[pd.DataFrame] = []
    for idx, path in enumerate(files, start=1):
        df = pd.read_parquet(path, columns=_COLS)
        parts.append(_agg_match_h2h(df))
        if checkpoint > 0 and (idx % checkpoint == 0):
            merged.append(_combine(parts))
            parts = []
    merged.append(_combine(parts))
    ledger = _combine(merged)
    ledger["strike_rate"] = (
        (ledger["runs"].astype(float) * 100.0 / ledger["balls"].astype(float))
        .where(ledger["balls"].astype(float) > 0, 0.0)
        .astype(float)
    )
    return ledger.sort_values(["batter", "bowler"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    load_env()
    paths = get_project_paths()
    ns = _parse_args(argv)
    out: Path = ns.out
    out.parent.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger(paths.matches_dir, int(ns.checkpoint))
    ledger.to_parquet(out, index=False)
    print(f"Wrote H2H ledger: {out} rows={len(ledger)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

