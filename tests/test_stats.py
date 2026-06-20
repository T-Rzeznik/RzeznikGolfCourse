"""Tests for the chatbot's "brain" (golf/stats.py).

The brain is the trust boundary: the LLM only repeats what these functions
return, so they must be deterministic and honest about thin data. No network
here — we point the schema at small in-memory CSVs (same monkeypatch pattern as
test_pipeline.py) and assert the math.

Run from the repo root:   python -m pytest -q
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from golf import data as gdata  # noqa: E402
from golf import schema  # noqa: E402
from golf import stats  # noqa: E402


# --- Builders --------------------------------------------------------------
def shot(shot_id, rid, pid, hole, stroke, order, outcome, distance,
         best=False, mull=False, ts=0):
    return {"shot_id": shot_id, "round_id": rid, "player_id": pid, "hole": hole,
            "stroke_num": stroke, "shot_order": order, "outcome": outcome,
            "distance": distance, "best_ball": best, "mulligan": mull, "ts": ts}


PLAYERS = pd.DataFrame([
    {"player_id": 1, "name": "Tommy", "hand": "R", "notes": ""},
    {"player_id": 2, "name": "Sam", "hand": "L", "notes": ""},
])


def _round_row(rid, players, date="2026-05-01"):
    return {"round_id": rid, "date": date, "players": players,
            "ground": "dry", "wind": "calm", "client_round_id": f"cr-{rid}", "notes": ""}


def _install(monkeypatch, tmp_path, shots, rounds, players=PLAYERS):
    monkeypatch.setattr(schema, "DATA_DIR", tmp_path)
    monkeypatch.setattr(schema, "PLAYERS_FILE", tmp_path / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", tmp_path / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", tmp_path / "shots.csv")
    players.to_csv(tmp_path / "players.csv", index=False)
    rounds.to_csv(tmp_path / "rounds.csv", index=False)
    shots.to_csv(tmp_path / "shots.csv", index=False)


# A solo Tommy dataset where, off `short`, he goes good/hole/grounder/ob:
# 2 makes out of 4 real attempts (each attempt is its own hole, holed on the
# last stroke). Keeps the scramble invariant trivially (1 player per stroke).
def _tommy_short_dataset():
    rows, rounds = [], []
    sid = 1
    # outcome of the (single, holing) stroke on four separate holes/rounds
    specs = [("hole", True), ("good", True), ("grounder", True), ("ob", False)]
    for i, (oc, best) in enumerate(specs, start=1):
        # 'hole' must be on the final stroke; for non-hole finals we still need a
        # holing stroke, so model each as: stroke1 from short with `oc`, then a
        # final holing stroke. Except the pure 'hole' case holes immediately.
        rid = i
        if oc == "hole":
            rows.append(shot(sid, rid, 1, 1, 1, 1, "hole", "short", best=True)); sid += 1
        else:
            rows.append(shot(sid, rid, 1, 1, 1, 1, oc, "short", best=best)); sid += 1
            rows.append(shot(sid, rid, 1, 1, 2, 1, "hole", "short", best=True)); sid += 1
        rounds.append(_round_row(rid, "1"))
    return pd.DataFrame(rows), pd.DataFrame(rounds)


# --- smoothing math --------------------------------------------------------
def test_smoothed_empty_is_prior():
    s = stats._smoothed(makes=0, n=0)
    assert s["smoothed_rate"] == stats.DEFAULT_PRIOR_MEAN
    assert s["raw_rate"] is None
    assert s["uncertain"] is True


def test_smoothed_shrinks_one_for_one():
    # Raw 1/1 = 100%, but the prior (mean .5, strength 10) pulls it to
    # (1 + .5*10)/(1 + 10) = 6/11 = 0.545 — a lone make is barely above average.
    s = stats._smoothed(makes=1, n=1)
    assert s["raw_rate"] == 1.0
    assert s["smoothed_rate"] == 0.545
    assert s["uncertain"] is True


def test_smoothed_converges_to_empirical():
    # The stronger prior shrinks small samples hard, but with enough data the
    # estimate still converges to the empirical rate.
    s = stats._smoothed(makes=800, n=1000)
    assert abs(s["smoothed_rate"] - 0.8) < 0.01
    assert s["uncertain"] is False


def test_smoothed_shrinks_toward_the_given_prior_mean():
    # The prior mean isn't fixed at 0.5 — make_rate passes the league rate for
    # the shot. A lone make on an easy shot (league .9) stays high, not pulled
    # down to 50%: (1 + .9*10)/(1 + 10) = 10/11 = 0.909.
    s = stats._smoothed(makes=1, n=1, prior_mean=0.9)
    assert s["smoothed_rate"] == 0.909
    assert s["prior_mean"] == 0.9


# --- make definition & filters --------------------------------------------
def test_make_counts_only_good_and_hole(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    r = stats.make_rate("Tommy", distance="short")
    # The four real attempts from `short`: hole, good, grounder, ob (the extra
    # holing strokes are also from `short`, so they count as makes too).
    # makes = every good/hole shot from short; ob & grounder are not makes.
    assert r["makes"] < r["n"]              # ob/grounder drag it below 100%
    assert r["makes"] >= 2                  # at least the explicit hole + good
    assert r["raw_rate"] == round(r["makes"] / r["n"], 3)


def test_make_rate_distance_filter_narrows(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    short = stats.make_rate("Tommy", distance="short")["n"]
    tee = stats.make_rate("Tommy", distance="tee")["n"]
    overall = stats.make_rate("Tommy")["n"]
    assert tee == 0                         # nobody shot from the tee here
    assert short == overall                 # all shots were from short
    assert overall > 0


def test_case_insensitive_name(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    assert stats.make_rate("tommy")["player"] == "Tommy"


def test_unknown_player_raises(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    with pytest.raises(ValueError):
        stats.make_rate("Gandalf")


def test_unknown_distance_raises(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    with pytest.raises(ValueError):
        stats.make_rate("Tommy", distance="banana")


# --- weighting: shrink toward the league average for the same shot ---------
def _flat(spec):
    """Build (shots, rounds) from (pid, hole, distance, outcome) tuples.

    Each tuple is one independent counting attempt. stats reads attempts via
    features.shot_outcome(), which doesn't enforce the scramble invariant, so a
    flat list of shots is enough to exercise the make-rate math directly.
    """
    rows, rounds, sid = [], [], 1
    for i, (pid, hole, dist, oc) in enumerate(spec, start=1):
        rows.append(shot(sid, i, pid, hole, 1, 1, oc, dist, best=(oc in ("good", "hole"))))
        rounds.append(_round_row(i, str(pid)))
        sid += 1
    return pd.DataFrame(rows), pd.DataFrame(rounds)


def test_thin_sample_shrinks_toward_league_not_half(monkeypatch, tmp_path):
    # The field hits `short` easily (Sam 18/20). Tommy has a single short make.
    # His rate should land near the league short-rate (~0.9), not the flat 0.5.
    spec = [(2, 1, "short", "good")] * 18 + [(2, 1, "short", "ob")] * 2
    spec += [(1, 1, "short", "good")]
    _install(monkeypatch, tmp_path, *_flat(spec))
    tommy = stats.make_rate("Tommy", distance="short")
    assert tommy["n"] == 1
    assert tommy["prior_mean"] == round(19 / 21, 3)   # league short = 19/21
    assert tommy["smoothed_rate"] > 0.85              # pulled up toward the easy field


def test_volume_outweighs_the_prior(monkeypatch, tmp_path):
    # The field is weak from `short` (Sam 2/40) so the league prior is ~0.47,
    # but Tommy is good there over a big sample (40/50). His volume should keep
    # him near his own raw rate rather than dragging him down to the league.
    spec = [(2, 1, "short", "good")] * 2 + [(2, 1, "short", "ob")] * 38
    spec += [(1, 1, "short", "good")] * 40 + [(1, 1, "short", "ob")] * 10
    _install(monkeypatch, tmp_path, *_flat(spec))
    tommy = stats.make_rate("Tommy", distance="short")
    assert tommy["n"] == 50
    assert tommy["raw_rate"] == 0.8
    assert tommy["prior_mean"] < 0.55                 # league is weak here
    assert tommy["smoothed_rate"] > 0.72              # but volume keeps him high


def test_prior_is_bucket_specific(monkeypatch, tmp_path):
    # Same thin sample (1 make) in two buckets: the field makes tap-ins easily
    # but misses tees often. Tommy's tap-in should read high and his tee low —
    # proving the prior is the league rate for THAT bucket, not one global mean.
    spec  = [(2, 1, "tap_in", "good")] * 9 + [(2, 1, "tap_in", "ob")] * 1
    spec += [(2, 1, "tee", "good")] * 1 + [(2, 1, "tee", "ob")] * 9
    spec += [(1, 1, "tap_in", "good"), (1, 1, "tee", "good")]
    _install(monkeypatch, tmp_path, *_flat(spec))
    tap = stats.make_rate("Tommy", distance="tap_in")
    tee = stats.make_rate("Tommy", distance="tee")
    assert tap["n"] == tee["n"] == 1
    assert tap["smoothed_rate"] > 0.6                 # eased up by the easy field
    assert tee["smoothed_rate"] < 0.4                 # held down by the hard field
    assert tap["smoothed_rate"] - tee["smoothed_rate"] > 0.3


# --- compare_players -------------------------------------------------------
def test_compare_players_picks_leader_but_hedges(monkeypatch, tmp_path):
    # Tommy 2/2 good; Sam 0/2 (both ob). Tommy leads, but tiny n -> not confident.
    rows = [
        shot(1, 1, 1, 1, 1, 1, "good", "short", best=True),
        shot(2, 1, 1, 1, 2, 1, "hole", "short", best=True),
        shot(3, 2, 2, 1, 1, 1, "grounder", "short", best=True),
        shot(4, 2, 2, 1, 2, 1, "hole", "short", best=True),
    ]
    rounds = [_round_row(1, "1"), _round_row(2, "2")]
    _install(monkeypatch, tmp_path, pd.DataFrame(rows), pd.DataFrame(rounds))
    cmp = stats.compare_players("Tommy", "Sam", distance="short")
    assert cmp["more_likely"] == "Tommy"
    assert cmp["confident"] is False        # thin data must not claim confidence


# --- leaderboard -----------------------------------------------------------
def test_leaderboard_sorted_and_min_n(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    lb = stats.leaderboard(distance="short", min_n=1)
    assert [r["rank"] for r in lb["ranking"]] == list(range(1, len(lb["ranking"]) + 1))
    rates = [r["smoothed_rate"] for r in lb["ranking"]]
    assert rates == sorted(rates, reverse=True)
    # Sam has no `short` shots -> filtered out by min_n.
    assert all(r["player"] != "Sam" for r in lb["ranking"])


# --- hole_difficulty -------------------------------------------------------
def test_hole_difficulty_hardest_first(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    hd = stats.hole_difficulty()
    tops = [h["avg_to_par"] for h in hd["hardest_first"]]
    assert tops == sorted(tops, reverse=True)


# --- JSON-serializable (no numpy/pandas leaks into the LLM payload) --------
def test_all_results_json_serializable(monkeypatch, tmp_path):
    shots, rounds = _tommy_short_dataset()
    _install(monkeypatch, tmp_path, shots, rounds)
    payloads = [
        stats.make_rate("Tommy", distance="short"),
        stats.compare_players("Tommy", "Sam"),
        stats.player_summary("Tommy"),
        stats.leaderboard(),
        stats.hole_difficulty(),
        stats.list_players(),
        stats.list_distances(),
    ]
    for p in payloads:
        json.dumps(p)  # raises TypeError if any numpy/pandas scalar slipped through


def test_empty_data_is_safe(monkeypatch, tmp_path):
    empty_shots = pd.DataFrame(columns=schema.SHOT_COLUMNS)
    empty_rounds = pd.DataFrame(columns=schema.ROUND_COLUMNS)
    _install(monkeypatch, tmp_path, empty_shots, empty_rounds)
    r = stats.make_rate("Tommy")
    assert r["n"] == 0
    assert r["smoothed_rate"] == stats.DEFAULT_PRIOR_MEAN
    assert stats.leaderboard()["ranking"] == []
    assert stats.hole_difficulty()["hardest_first"] == []
