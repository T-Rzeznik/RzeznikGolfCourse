"""Load, validate, aggregate, and append the golf data.

The per-shot table (`shots.csv`) is the source of truth. Hole and round
scorecards are *derived* here so they can never drift out of sync.
"""
from __future__ import annotations

import pandas as pd
import yaml

from . import schema


# --- Loading ---------------------------------------------------------------
def load_course() -> dict:
    with open(schema.COURSE_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def holes_frame() -> pd.DataFrame:
    """Course holes as a tidy DataFrame (one row per hole)."""
    course = load_course()
    return pd.DataFrame(course["holes"])


def load_players() -> pd.DataFrame:
    return pd.read_csv(schema.PLAYERS_FILE)


def load_rounds() -> pd.DataFrame:
    return pd.read_csv(schema.ROUNDS_FILE, dtype={"round_id": "Int64"})


def load_shots() -> pd.DataFrame:
    df = pd.read_csv(schema.SHOTS_FILE)
    if df.empty:
        return df
    df["holed"] = df["holed"].astype("boolean")
    # `mulligan` may be missing in older files; default to False.
    if "mulligan" not in df.columns:
        df["mulligan"] = False
    df["mulligan"] = df["mulligan"].fillna(False).astype("boolean")
    return df


def target_score() -> int:
    """The team total to beat (defaults to the course par total)."""
    course = load_course()["course"]
    return int(course.get("target_score", course["par_total"]))


def counting_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Shots that count toward the score (mulligan do-overs removed)."""
    return shots[~shots["mulligan"].fillna(False)]


# --- Derived scorecards ----------------------------------------------------
def hole_scores() -> pd.DataFrame:
    """One row per (round, player, hole): strokes plus par/score-to-par.

    Strokes exclude mulligan do-overs.
    """
    shots = load_shots()
    if shots.empty:
        return pd.DataFrame(
            columns=["round_id", "player_id", "hole", "strokes", "par", "to_par"]
        )
    strokes = (
        counting_shots(shots)
        .groupby(["round_id", "player_id", "hole"])
        .size()
        .reset_index(name="strokes")
    )
    pars = holes_frame()[["hole", "par"]]
    out = strokes.merge(pars, on="hole", how="left")
    out["to_par"] = out["strokes"] - out["par"]
    return out


def player_round_scores() -> pd.DataFrame:
    """One row per (round, player): individual total strokes and to_par.

    Best ball is a team game, so this is for individual analysis, not winning.
    """
    hs = hole_scores()
    if hs.empty:
        return pd.DataFrame(columns=["round_id", "player_id", "total", "to_par"])
    return (
        hs.groupby(["round_id", "player_id"])
        .agg(total=("strokes", "sum"), to_par=("to_par", "sum"))
        .reset_index()
    )


def team_hole_scores() -> pd.DataFrame:
    """Best ball: one row per (round, hole) = the best score among the players."""
    hs = hole_scores()
    if hs.empty:
        return pd.DataFrame(columns=["round_id", "hole", "team_strokes", "par", "to_par"])
    team = (
        hs.groupby(["round_id", "hole"])
        .agg(team_strokes=("strokes", "min"), par=("par", "first"))
        .reset_index()
    )
    team["to_par"] = team["team_strokes"] - team["par"]
    return team


def round_scores() -> pd.DataFrame:
    """Best-ball team result, one row per round.

    `won` = team total beat the target score (default: course par total).
    """
    ths = team_hole_scores()
    if ths.empty:
        return pd.DataFrame(
            columns=["round_id", "team_total", "to_par", "target", "won"]
        )
    out = (
        ths.groupby("round_id")
        .agg(team_total=("team_strokes", "sum"), to_par=("to_par", "sum"))
        .reset_index()
    )
    out["target"] = target_score()
    out["won"] = out["team_total"] < out["target"]
    return out


# --- Validation ------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of human-readable data problems ([] means clean)."""
    problems: list[str] = []
    shots = load_shots()
    if shots.empty:
        return problems

    valid_holes = set(holes_frame()["hole"])
    valid_players = set(load_players()["player_id"])

    bad_lies = set(shots["lie"].dropna()) - set(schema.LIES)
    if bad_lies:
        problems.append(f"Unknown lies: {sorted(bad_lies)}")
    bad_results = set(shots["result"].dropna()) - set(schema.RESULTS)
    if bad_results:
        problems.append(f"Unknown results: {sorted(bad_results)}")
    bad_holes = set(shots["hole"]) - valid_holes
    if bad_holes:
        problems.append(f"Holes not on the course: {sorted(bad_holes)}")
    bad_players = set(shots["player_id"]) - valid_players
    if bad_players:
        problems.append(f"Unknown player_ids: {sorted(bad_players)}")

    # Every (round, player, hole) must end on exactly one holed shot.
    grp = shots.groupby(["round_id", "player_id", "hole"])["holed"].sum()
    not_finished = grp[grp != 1]
    for (rid, pid, hole), n in not_finished.items():
        problems.append(
            f"Round {rid} player {pid} hole {hole}: {int(n)} holed shots (expected 1)"
        )

    # One mulligan per round for the group.
    mull = shots[shots["mulligan"].fillna(False)].groupby("round_id").size()
    for rid, n in mull[mull > 1].items():
        problems.append(f"Round {rid}: {int(n)} mulligans used (max 1 per round)")
    return problems


# --- Appending -------------------------------------------------------------
def next_id(df: pd.DataFrame, col: str) -> int:
    if df.empty or df[col].dropna().empty:
        return 1
    return int(df[col].max()) + 1


def append_rounds(rows: list[dict]) -> None:
    _append(schema.ROUNDS_FILE, rows, schema.ROUND_COLUMNS)


def append_shots(rows: list[dict]) -> None:
    _append(schema.SHOTS_FILE, rows, schema.SHOT_COLUMNS)


def _append(path, rows: list[dict], columns: list[str]) -> None:
    existing = pd.read_csv(path) if path.stat().st_size > 0 else pd.DataFrame()
    new = pd.DataFrame(rows, columns=columns)
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(path, index=False)
