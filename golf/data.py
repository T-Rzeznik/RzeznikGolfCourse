"""Load, validate, aggregate, and append the golf data.

The per-shot table (`shots.csv`) is the source of truth — one row per player per
scramble stroke. Team scorecards are *derived* here so they can never drift out
of sync. See schema.py for the scramble invariant every rule below follows from.
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
    for col in ("hole", "stroke_num", "shot_order"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("best_ball", "mulligan"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype("boolean")
    return df


def target_score() -> int:
    """The team total to beat (defaults to the course par total)."""
    course = load_course()["course"]
    return int(course.get("target_score", course["par_total"]))


def counting_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Shots that count toward the score (mulligan do-overs removed)."""
    return shots[~shots["mulligan"].fillna(False)]


def active_players(rounds: pd.DataFrame | None = None) -> dict:
    """Map round_id -> list of player_ids active that round (from rounds.csv)."""
    rounds = load_rounds() if rounds is None else rounds
    out = {}
    for row in rounds.itertuples(index=False):
        out[int(row.round_id)] = [int(p) for p in str(row.players).split("|") if p]
    return out


# --- Derived scorecards ----------------------------------------------------
def hole_scores() -> pd.DataFrame:
    """Team score per hole: one row per (round, hole).

    Scramble team score = number of strokes taken = max(stroke_num) over the
    counting shots (an all-OB re-hit bumps stroke_num, so it correctly costs a
    stroke). Mulligan do-overs are excluded.
    """
    shots = load_shots()
    if shots.empty:
        return pd.DataFrame(
            columns=["round_id", "hole", "team_strokes", "par", "to_par"]
        )
    team = (
        counting_shots(shots)
        .groupby(["round_id", "hole"])["stroke_num"]
        .max()
        .reset_index(name="team_strokes")
    )
    pars = holes_frame()[["hole", "par"]]
    out = team.merge(pars, on="hole", how="left")
    out["to_par"] = out["team_strokes"] - out["par"]
    return out


def round_scores() -> pd.DataFrame:
    """Scramble team result, one row per round.

    `won` = team total beat the target score (default: course par total).
    """
    hs = hole_scores()
    if hs.empty:
        return pd.DataFrame(
            columns=["round_id", "team_total", "to_par", "target", "won"]
        )
    out = (
        hs.groupby("round_id")
        .agg(team_total=("team_strokes", "sum"), to_par=("to_par", "sum"))
        .reset_index()
    )
    out["target"] = target_score()
    out["won"] = out["team_total"] < out["target"]
    return out


def player_contributions() -> pd.DataFrame:
    """Per (round, player): how each player contributed in the scramble.

    `shots` = counting shots hit; `best_balls` = times their ball was kept
    (includes the holing ball). The remaining columns are per-outcome counts.
    """
    shots = load_shots()
    base_cols = ["round_id", "player_id", "shots", "best_balls"]
    if shots.empty:
        return pd.DataFrame(columns=base_cols)
    cs = counting_shots(shots)
    out = (
        cs.groupby(["round_id", "player_id"])
        .agg(shots=("shot_id", "count"), best_balls=("best_ball", "sum"))
        .reset_index()
    )
    counts = (
        cs.pivot_table(index=["round_id", "player_id"], columns="outcome",
                       values="shot_id", aggfunc="count", fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    return out.merge(counts, on=["round_id", "player_id"], how="left")


# --- Validation ------------------------------------------------------------
def validate() -> list[str]:
    """Return a list of human-readable data problems ([] means clean)."""
    problems: list[str] = []
    shots = load_shots()
    if shots.empty:
        return problems

    valid_holes = set(holes_frame()["hole"])
    valid_players = set(load_players()["player_id"])
    rosters = active_players()

    bad_outcomes = set(shots["outcome"].dropna()) - set(schema.OUTCOMES)
    if bad_outcomes:
        problems.append(f"Unknown outcomes: {sorted(bad_outcomes)}")
    bad_holes = set(shots["hole"]) - valid_holes
    if bad_holes:
        problems.append(f"Holes not on the course: {sorted(bad_holes)}")
    bad_players = set(shots["player_id"]) - valid_players
    if bad_players:
        problems.append(f"Unknown player_ids: {sorted(bad_players)}")

    # One mulligan per round; each mulligan shares its slot with a real do-over.
    mull = shots[shots["mulligan"].fillna(False)]
    for rid, n in mull.groupby("round_id").size().items():
        if n > 1:
            problems.append(f"Round {rid}: {int(n)} mulligans used (max 1 per round)")
    counting = counting_shots(shots)
    for key, grp in mull.groupby(["round_id", "hole", "player_id", "stroke_num"]):
        rid, hole, pid, sn = key
        replaced = counting[
            (counting["round_id"] == rid) & (counting["hole"] == hole)
            & (counting["player_id"] == pid) & (counting["stroke_num"] == sn)
        ]
        if replaced.empty:
            problems.append(
                f"Round {rid} hole {hole} player {pid} stroke {sn}: "
                "mulligan has no do-over replacement"
            )

    # Everything below is evaluated on counting shots, per (round, hole).
    for (rid, hole), grp in counting.groupby(["round_id", "hole"]):
        roster = rosters.get(int(rid), [])
        n_players = len(roster)
        if not 1 <= n_players <= 3:
            problems.append(f"Round {rid}: {n_players} players (expected 1-3)")

        strokes = sorted(int(x) for x in grp["stroke_num"].dropna().unique())
        max_s = max(strokes)
        if strokes != list(range(1, max_s + 1)):
            problems.append(f"Round {rid} hole {hole}: stroke numbers not 1..{max_s} ({strokes})")

        # `hole` outcomes only allowed on the final stroke, and >=1 there.
        holed_strokes = sorted(int(x) for x in grp[grp["outcome"] == "hole"]["stroke_num"].unique())
        if holed_strokes != [max_s]:
            problems.append(
                f"Round {rid} hole {hole}: 'hole' outcomes on strokes {holed_strokes} "
                f"(expected only the final stroke {max_s})"
            )

        for sn, sgrp in grp.groupby("stroke_num"):
            # Every active player hits exactly once each stroke.
            counts = sgrp["player_id"].value_counts()
            missing = set(roster) - set(counts.index)
            if missing:
                problems.append(f"Round {rid} hole {hole} stroke {sn}: missing players {sorted(missing)}")
            dupes = counts[counts > 1]
            for pid, c in dupes.items():
                problems.append(f"Round {rid} hole {hole} stroke {sn}: player {pid} has {int(c)} counting shots")

            best = sgrp[sgrp["best_ball"].fillna(False)]
            all_ob = (sgrp["outcome"] == "ob").all()
            if (best["outcome"] == "ob").any():
                problems.append(f"Round {rid} hole {hole} stroke {sn}: best ball can't be an OB shot")
            if sn == max_s:
                # Final stroke: best_ball is exactly the holing ball(s).
                holing = set(sgrp[sgrp["outcome"] == "hole"]["shot_id"])
                marked = set(best["shot_id"])
                if holing != marked:
                    problems.append(
                        f"Round {rid} hole {hole} final stroke: best_ball should mark the "
                        f"holing ball(s) {sorted(holing)}, got {sorted(marked)}"
                    )
            elif all_ob:
                if len(best):
                    problems.append(f"Round {rid} hole {hole} stroke {sn}: all-OB stroke should have no best ball")
            elif len(best) != 1:
                problems.append(f"Round {rid} hole {hole} stroke {sn}: {len(best)} best balls (expected 1)")
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
