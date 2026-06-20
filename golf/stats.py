"""The "brain": deterministic golf stats the chatbot narrates.

The chatbot is split into a **brain** (this module) and a **mouth** (an LLM in
`chatbot/`). The brain owns every number; the mouth may only repeat what the
brain returns. So this module is pure: it reads the data via `golf.data` /
`golf.features`, computes stats, and returns plain JSON-serializable dicts (no
pandas/numpy objects leak out). No LLM, no network, no I/O beyond the loaders.

Make-rates are **smoothed** because the dataset is tiny. A raw 1-for-1 reads as
100%, which is meaningless. We shrink toward a Beta(alpha, beta) prior:

    smoothed = (makes + alpha) / (n + alpha + beta)

With the default symmetric prior (mean 0.5, strength 2) this is Laplace
"add-one" smoothing: small samples pull toward 50/50 and the estimate converges
to the empirical rate as data grows. Every result also carries `n` and an
`uncertain` flag so the mouth can hedge honestly on thin data.
"""
from __future__ import annotations

import pandas as pd

from . import data as gdata
from . import features
from . import schema

# A shot "makes" (a good result for the team) when its outcome is one of these —
# the top two of the worst->best OUTCOMES ordering. `hole` is the literal goal;
# `good` is the clean strike that advances the scramble. Everything below is a
# weak contact (short_pop/grounder/overshoot), a bust (ob), or a non-attempt
# (skip, already excluded by features.shot_outcome). One place to tune.
MAKE_OUTCOMES = {"good", "hole"}

# Beta prior for smoothing. The prior MEAN a rate shrinks toward is the league
# average *for the same kind of shot* (computed per slice in `_league_make_rate`),
# so an inherently easy tap-in isn't dragged down by the global rate. The prior
# STRENGTH is how many league-average "phantom" attempts every player carries
# before their own shots outweigh the prior: a player needs roughly this many
# real attempts to move halfway from the league average to their own rate. Bigger
# strength => shot count matters more (this is the knob that makes volume count).
# 0.5 is only the fallback mean when the field has no comparable attempts yet.
DEFAULT_PRIOR_MEAN = 0.5
DEFAULT_PRIOR_STRENGTH = 10.0

# Below this many real attempts, flag the estimate as uncertain so the mouth hedges.
UNCERTAIN_N = 5

# compare_players needs both a real sample and a real gap before it claims an edge.
COMPARE_MIN_MARGIN = 0.05

# Derived from MAKE_OUTCOMES (in worst->best order) so the prose can't drift from
# what the brain actually counts — change the set above and this updates with it.
_MAKE_DEFINITION = (
    "a shot 'makes' when its outcome is "
    + " or ".join(f"'{o}'" for o in schema.OUTCOMES if o in MAKE_OUTCOMES)
)


# --- internals -------------------------------------------------------------
def _round(x: float, n: int = 3) -> float:
    """A plain Python float rounded for display (never a numpy scalar)."""
    return round(float(x), n)


def _smoothed(
    makes: int,
    n: int,
    prior_mean: float = DEFAULT_PRIOR_MEAN,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> dict:
    """Beta-smoothed make-rate plus the honesty metadata the mouth needs.

    Returns raw vs smoothed rate, the sample size, posterior spread, and an
    `uncertain` flag. With n == 0 the smoothed rate is exactly `prior_mean`
    (the denominator is always >= prior_strength, so never divides by zero).
    """
    makes, n = int(makes), int(n)
    alpha = prior_mean * prior_strength
    beta = (1.0 - prior_mean) * prior_strength
    a_post = makes + alpha
    b_post = (n - makes) + beta
    total = a_post + b_post
    post_mean = a_post / total
    post_var = (a_post * b_post) / (total * total * (total + 1.0))
    return {
        "n": n,
        "makes": makes,
        "raw_rate": _round(makes / n) if n else None,
        "smoothed_rate": _round(post_mean),
        "prior_mean": _round(prior_mean),  # the league rate this was shrunk toward
        "posterior_sd": _round(post_var ** 0.5),
        "uncertain": n < UNCERTAIN_N,
    }


def _players() -> pd.DataFrame:
    return gdata.load_players()


def _name_to_id(player: str) -> int:
    """Resolve a player name (case-insensitive) to a player_id.

    Raises ValueError on an unknown name; the tool layer turns that into a
    friendly message and the mouth can call `list_players` to recover.
    """
    players = _players()
    if players.empty:
        raise ValueError("No players are on record yet.")
    key = str(player).strip().lower()
    match = players[players["name"].astype(str).str.lower() == key]
    if match.empty:
        known = ", ".join(players["name"].astype(str))
        raise ValueError(f"Unknown player: {player!r}. Known players: {known}.")
    return int(match["player_id"].iloc[0])


def _id_to_name(pid: int) -> str:
    players = _players()
    match = players[players["player_id"] == pid]
    return str(match["name"].iloc[0]) if not match.empty else str(pid)


def _slice(df: pd.DataFrame, distance: str | None = None, hole: int | None = None) -> pd.DataFrame:
    """Narrow an attempts frame to one distance bucket and/or hole."""
    if df.empty:
        return df
    if distance is not None:
        df = df[df["distance"].astype("string") == distance]
    if hole is not None:
        df = df[df["hole"] == hole]
    return df


def _shots_for(player_id: int, distance: str | None = None, hole: int | None = None) -> pd.DataFrame:
    """Counting, real-attempt shots for a player, optionally filtered.

    Built on `features.shot_outcome()`, which already drops mulligan do-overs
    and `skip` turns, so the denominator is real attempts only.
    """
    df = features.shot_outcome()
    if df.empty:
        return df
    return _slice(df[df["player_id"] == player_id], distance=distance, hole=hole)


def _league_make_rate(distance: str | None = None, hole: int | None = None) -> float:
    """The make-rate over *everyone's* attempts of this same kind.

    This is the prior each per-player rate shrinks toward, so a player with few
    shots is assumed to play this shot like the field does — not like a coin
    flip, and not like the global average across all shot types. Sliced by the
    same distance/hole as the player's rate so an easy tap-in stays easy: with
    little data you read close to "how this shot plays for everyone", and only
    earn your own number once your volume outweighs the prior. Falls back to
    DEFAULT_PRIOR_MEAN when the field has no comparable attempts yet.
    """
    df = _slice(features.shot_outcome(), distance=distance, hole=hole)
    n = int(len(df))
    if not n:
        return DEFAULT_PRIOR_MEAN
    makes = int(df["outcome"].isin(MAKE_OUTCOMES).sum())
    return makes / n


def _validate_distance(distance: str | None) -> None:
    if distance is not None and distance not in schema.DISTANCES:
        raise ValueError(
            f"Unknown distance: {distance!r}. Valid distances: {', '.join(schema.DISTANCES)}."
        )


# --- public tools (1:1 with the chatbot's function declarations) -----------
def make_rate(player: str, distance: str | None = None, hole: int | None = None) -> dict:
    """How often `player` makes a shot, optionally on a distance bucket or hole.

    `distance` is one of DISTANCES (tee/long/mid/short/tap_in); `hole` is a hole
    number. Returns the smoothed make-rate with sample size and an `uncertain`
    flag. This is the core "who makes this shot" stat.
    """
    _validate_distance(distance)
    pid = _name_to_id(player)
    hole = None if hole is None else int(hole)
    df = _shots_for(pid, distance=distance, hole=hole)
    n = int(len(df))
    makes = int(df["outcome"].isin(MAKE_OUTCOMES).sum()) if n else 0
    out = {
        "player": _id_to_name(pid),
        "player_id": pid,
        "distance": distance,
        "hole": hole,
        "make_definition": _MAKE_DEFINITION,
    }
    # Shrink toward how the whole field plays this same shot, so thin samples
    # read close to the league rate and only big samples earn their own number.
    out.update(_smoothed(makes, n, prior_mean=_league_make_rate(distance, hole)))
    return out


def compare_players(player_a: str, player_b: str, distance: str | None = None) -> dict:
    """Head-to-head make-rate for two players, optionally on a distance bucket.

    Answers "is A more likely than B to make this shot?". Honest about
    confidence: `more_likely` is "tie" unless one player's smoothed rate beats
    the other by at least COMPARE_MIN_MARGIN, and `confident` is only true when
    both players also clear the uncertainty threshold.
    """
    a = make_rate(player_a, distance=distance)
    b = make_rate(player_b, distance=distance)
    diff = a["smoothed_rate"] - b["smoothed_rate"]
    if abs(diff) < COMPARE_MIN_MARGIN:
        more_likely = "tie"
    else:
        more_likely = a["player"] if diff > 0 else b["player"]
    confident = (
        more_likely != "tie"
        and not a["uncertain"]
        and not b["uncertain"]
    )
    return {
        "distance": distance,
        "player_a": a,
        "player_b": b,
        "difference": _round(diff),
        "more_likely": more_likely,
        "confident": confident,
        "make_definition": _MAKE_DEFINITION,
    }


def player_summary(player: str) -> dict:
    """Overall make-rate, a per-distance breakdown, and raw totals for a player."""
    pid = _name_to_id(player)
    name = _id_to_name(pid)

    by_distance = []
    for dist in schema.DISTANCES:
        rate = make_rate(name, distance=dist)
        if rate["n"]:  # only surface buckets the player has actually shot from
            by_distance.append(rate)

    # Aggregate raw counts across rounds from player_contributions(). Outcome
    # columns appear only when present, so guard every lookup.
    contrib = gdata.player_contributions()
    totals = {"total_shots": 0, "total_best_balls": 0,
              "outcomes": {o: 0 for o in schema.OUTCOMES}}
    if not contrib.empty:
        mine = contrib[contrib["player_id"] == pid]
        if not mine.empty:
            totals["total_shots"] = int(mine["shots"].sum())
            totals["total_best_balls"] = int(mine["best_balls"].sum())
            for o in schema.OUTCOMES:
                if o in mine.columns:
                    totals["outcomes"][o] = int(mine[o].sum())

    return {
        "player": name,
        "player_id": pid,
        "overall": make_rate(name),
        "by_distance": by_distance,
        "totals": totals,
    }


def leaderboard(distance: str | None = None, min_n: int = 1) -> dict:
    """Players ranked by smoothed make-rate (optionally for one distance bucket).

    Answers "who's best on tap-ins?". `min_n` drops players with too few
    attempts to rank; ties and thin samples stay flagged via `uncertain`.
    """
    _validate_distance(distance)
    players = _players()
    rows = []
    for name in players["name"].astype(str):
        rate = make_rate(name, distance=distance)
        if rate["n"] >= int(min_n):
            rows.append(rate)
    rows.sort(key=lambda r: r["smoothed_rate"], reverse=True)
    ranking = [
        {
            "rank": i + 1,
            "player": r["player"],
            "smoothed_rate": r["smoothed_rate"],
            "n": r["n"],
            "uncertain": r["uncertain"],
        }
        for i, r in enumerate(rows)
    ]
    return {
        "distance": distance,
        "min_n": int(min_n),
        "ranking": ranking,
        "make_definition": _MAKE_DEFINITION,
    }


def hole_difficulty(top: int | None = None) -> dict:
    """Holes ranked hardest-first by average score to par.

    Thin passthrough over `features.hole_difficulty()`. `top` limits to the N
    hardest holes. Answers "which hole is hardest?".
    """
    df = features.hole_difficulty()
    holes = []
    if not df.empty:
        for row in df.itertuples(index=False):
            std = getattr(row, "std_strokes", None)
            holes.append({
                "hole": int(row.hole),
                "par": int(row.par) if pd.notna(getattr(row, "par", None)) else None,
                "rounds_played": int(row.rounds_played),
                "avg_strokes": _round(row.avg_strokes, 2),
                "avg_to_par": _round(row.avg_to_par, 2),
                "std_strokes": _round(std, 2) if pd.notna(std) else None,
            })
    if top is not None:
        holes = holes[: int(top)]
    return {"hardest_first": holes}


def list_players() -> dict:
    """The roster, so the mouth never invents a player name."""
    players = _players()
    return {
        "players": [
            {"player_id": int(r.player_id), "name": str(r.name), "hand": str(r.hand)}
            for r in players.itertuples(index=False)
        ]
    }


def list_distances() -> dict:
    """The distance vocabulary and what a 'make' means, for the mouth's reference."""
    return {
        "distances": list(schema.DISTANCES),
        "ordering": "far -> near (tee is the tee shot, tap_in is closest)",
        "outcomes_worst_to_best": list(schema.OUTCOMES),
        "make_definition": _MAKE_DEFINITION,
    }
