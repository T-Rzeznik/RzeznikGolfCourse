"""Feature tables for the four ML goals.

Each builder returns a tidy DataFrame ready to split into X/y. Modeling itself
(train/test, model choice, metrics) lives in the notebooks so it's easy to
iterate on; this module just guarantees consistent, leakage-free inputs.
"""
from __future__ import annotations

import pandas as pd

from . import data as gdata


def _holes_meta() -> pd.DataFrame:
    """Per-hole course attributes useful as features everywhere."""
    h = gdata.holes_frame()
    cols = ["hole", "par", "yards", "dogleg", "blind"]
    return h[[c for c in cols if c in h.columns]]


def _round_n_players() -> pd.DataFrame:
    """One row per round: how many players were on the team."""
    rounds = gdata.load_rounds()
    if rounds.empty:
        return pd.DataFrame(columns=["round_id", "n_players"])
    rounds = rounds.copy()
    rounds["n_players"] = rounds["players"].fillna("").apply(
        lambda s: len([p for p in str(s).split("|") if p])
    )
    return rounds[["round_id", "n_players"]]


# 1) SCORE PREDICTION -------------------------------------------------------
def score_prediction() -> pd.DataFrame:
    """Predict the team's strokes on a (round, hole). Target = `team_strokes`."""
    hs = gdata.hole_scores()
    if hs.empty:
        return hs
    feats = (
        hs.merge(_holes_meta(), on=["hole", "par"], how="left")
        .merge(_round_n_players(), on="round_id", how="left")
    )
    return feats  # X = [hole, par, dogleg, blind, n_players], y = team_strokes


# 2) WIN PROBABILITY --------------------------------------------------------
def win_probability() -> pd.DataFrame:
    """Scramble team result, one row per round. Target = `won` (beat target).

    Adds:
      * `n_players` in the round (more players -> better scramble score).
      * `prior_avg_to_par`: the team's mean to_par over prior rounds (expanding,
        shifted so the current round never leaks into its own feature).
    """
    rs = gdata.round_scores()
    rounds = gdata.load_rounds()
    if rs.empty:
        return rs
    rs = rs.merge(rounds[["round_id", "date", "players"]], on="round_id", how="left")
    rs["n_players"] = rs["players"].fillna("").apply(
        lambda s: len([p for p in str(s).split("|") if p])
    )
    rs = rs.sort_values(["date", "round_id"])
    rs["prior_avg_to_par"] = rs["to_par"].shift().expanding().mean()
    return rs  # X = [n_players, prior_avg_to_par], y = won


# 3) HOLE DIFFICULTY --------------------------------------------------------
def hole_difficulty() -> pd.DataFrame:
    """One row per hole: average team strokes / score-to-par and related stats."""
    hs = gdata.hole_scores()
    if hs.empty:
        return hs
    agg = (
        hs.groupby("hole")
        .agg(
            rounds_played=("team_strokes", "count"),
            avg_strokes=("team_strokes", "mean"),
            avg_to_par=("to_par", "mean"),
            std_strokes=("team_strokes", "std"),
        )
        .reset_index()
        .merge(_holes_meta(), on="hole", how="left")
        .sort_values("avg_to_par", ascending=False)
    )
    return agg


# 4) SHOT OUTCOME -----------------------------------------------------------
def shot_outcome() -> pd.DataFrame:
    """One row per counting shot. Target = `outcome` (one of six).

    With one club and no distance/lie tracking, this is purely categorical
    (player, hole, where in the hole, shot order, team size), so expect modest
    signal. Mulligan do-overs and skipped turns are excluded so we model real
    attempts.
    """
    shots = gdata.counting_shots(gdata.load_shots())
    if not shots.empty:
        shots = shots[shots["outcome"] != "skip"]
    if shots.empty:
        return shots
    pars = gdata.holes_frame()[["hole", "par"]]
    feats = (
        shots.merge(pars, on="hole", how="left")
        .merge(_round_n_players(), on="round_id", how="left")
    )
    feats["is_first"] = feats["shot_order"] == 1
    return feats  # X = [player_id, hole, par, stroke_num, shot_order, n_players], y = outcome
