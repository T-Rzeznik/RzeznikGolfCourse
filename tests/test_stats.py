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
    # Raw 1/1 = 100%, but Laplace add-one gives (1+1)/(1+2) = 0.667.
    s = stats._smoothed(makes=1, n=1)
    assert s["raw_rate"] == 1.0
    assert s["smoothed_rate"] == 0.667
    assert s["uncertain"] is True


def test_smoothed_converges_to_empirical():
    s = stats._smoothed(makes=80, n=100)
    assert abs(s["smoothed_rate"] - 0.8) < 0.01
    assert s["uncertain"] is False


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
