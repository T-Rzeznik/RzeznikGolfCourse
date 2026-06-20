"""Tests for the stats-dashboard aggregation (golf/dashboard.py).

The dashboard is the visual sibling of the chat caddie: it must compute its
numbers from the same brain so the two can never disagree. These tests pin the
`overall` block's values against small in-memory datasets — same pattern as
tests/test_stats.py (monkeypatch the schema's CSV paths, then assert the derived
numbers). Frontend rendering is verified by eye, not here.

Run from the repo root:   python -m pytest -q
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from golf import schema  # noqa: E402
from golf import dashboard  # noqa: E402


PLAYERS = pd.DataFrame([
    {"player_id": 1, "name": "Tommy", "hand": "R", "notes": ""},
    {"player_id": 2, "name": "Matt", "hand": "R", "notes": ""},
])


def shot(shot_id, rid, pid, hole, stroke, order, outcome, distance, best=False, mull=False, ts=0):
    return {"shot_id": shot_id, "round_id": rid, "player_id": pid, "hole": hole,
            "stroke_num": stroke, "shot_order": order, "outcome": outcome,
            "distance": distance, "best_ball": best, "mulligan": mull, "ts": ts}


def solo_round(rid, date, strokes, pid=1, hole=1):
    """A solo round on one hole (par 3) taking `strokes` team strokes.

    Stroke 1 is a good tee shot; the final stroke holes; any middle strokes are
    good. Returns (shot rows, round row).
    """
    rows = []
    for s in range(1, strokes + 1):
        last = s == strokes
        rows.append(shot(rid * 100 + s, rid, pid, hole, s, 1,
                         "hole" if last else "good",
                         "tee" if s == 1 else "short", best=True))
    rnd = {"round_id": rid, "date": date, "players": str(pid),
           "ground": "", "wind": "", "client_round_id": f"cr-{rid}", "notes": ""}
    return rows, rnd


def two_player_round(rid, date):
    """Hole 1 (par 3), both players hit each stroke, team holes in 2.

    Tommy: good then hole (2 makes). Matt: grounder then ob (0 makes).
    """
    rows = [
        shot(rid * 100 + 1, rid, 1, 1, 1, 1, "good", "tee", best=True),
        shot(rid * 100 + 2, rid, 2, 1, 1, 2, "grounder", "tee"),
        shot(rid * 100 + 3, rid, 1, 1, 2, 1, "hole", "short", best=True),
        shot(rid * 100 + 4, rid, 2, 1, 2, 2, "ob", "short"),
    ]
    rnd = {"round_id": rid, "date": date, "players": "1|2",
           "ground": "", "wind": "", "client_round_id": f"cr-{rid}", "notes": ""}
    return rows, rnd


def install(monkeypatch, tmp_path, specs, players=PLAYERS):
    shots, rounds = [], []
    for rows, rnd in specs:
        shots += rows
        rounds.append(rnd)
    monkeypatch.setattr(schema, "DATA_DIR", tmp_path)
    monkeypatch.setattr(schema, "PLAYERS_FILE", tmp_path / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", tmp_path / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", tmp_path / "shots.csv")
    sdf = pd.DataFrame(shots) if shots else pd.DataFrame(columns=schema.SHOT_COLUMNS)
    rdf = pd.DataFrame(rounds) if rounds else pd.DataFrame(columns=schema.ROUND_COLUMNS)
    players.to_csv(tmp_path / "players.csv", index=False)
    rdf.to_csv(tmp_path / "rounds.csv", index=False)
    sdf.to_csv(tmp_path / "shots.csv", index=False)


# --- 1) best round ---------------------------------------------------------
def test_best_round_is_lowest_total_earliest_on_tie(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-02", 3),
        solo_round(2, "2026-05-01", 2),   # best: total 2, and earliest of the two 2s
        solo_round(3, "2026-05-03", 2),   # also 2 strokes, but a later date
    ])
    best = dashboard.overall()["best_round"]
    assert best["round_id"] == 2
    assert best["team_total"] == 2
    assert best["date"] == "2026-05-01"
    assert best["to_par"] == -1           # par 3, holed in 2


# --- 2) rounds played + date range -----------------------------------------
def test_rounds_played_and_date_range(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-02", 3),
        solo_round(2, "2026-05-01", 2),
        solo_round(3, "2026-05-09", 4),
    ])
    ov = dashboard.overall()
    assert ov["rounds_played"] == 3
    assert ov["date_range"] == {"first": "2026-05-01", "last": "2026-05-09"}


# --- 3) record vs the target -----------------------------------------------
def test_record_counts_wins_and_losses(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-01", 2),    # 2 < 20 -> win
        solo_round(2, "2026-05-02", 3),    # win
        solo_round(3, "2026-05-03", 21),   # 21 >= 20 -> loss
    ])
    rec = dashboard.overall()["record"]
    assert rec["wins"] == 2
    assert rec["losses"] == 1
    assert rec["win_pct"] == round(2 / 3, 3)


# --- 4) averages -----------------------------------------------------------
def test_average_total_and_to_par(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-01", 2),    # to_par -1
        solo_round(2, "2026-05-02", 3),    # to_par  0
        solo_round(3, "2026-05-03", 4),    # to_par +1
    ])
    ov = dashboard.overall()
    assert ov["avg_team_total"] == 3.0     # (2+3+4)/3
    assert ov["avg_to_par"] == 0.0         # (-1+0+1)/3


# --- 5) per-player make-rate row (shares the brain with the caddie) --------
def test_players_row_matches_make_rate_brain(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        two_player_round(1, "2026-05-01"),
        two_player_round(2, "2026-05-02"),
    ])
    from golf import stats
    rows = {p["name"]: p for p in dashboard.overall()["players"]}
    assert set(rows) == {"Tommy", "Matt"}
    for name in ("Tommy", "Matt"):
        brain = stats.make_rate(name)
        assert rows[name]["make_rate"] == brain["smoothed_rate"]
        assert rows[name]["n"] == brain["n"]
    # Tommy (all makes) ranks ahead of Matt (no makes).
    order = [p["name"] for p in dashboard.overall()["players"]]
    assert order.index("Tommy") < order.index("Matt")


# --- 6) empty data is safe and JSON-serializable ---------------------------
def test_empty_data_is_safe_and_serializable(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [])
    ov = dashboard.overall()
    assert ov["rounds_played"] == 0
    assert ov["best_round"] is None
    assert ov["date_range"] is None
    assert ov["record"] == {"wins": 0, "losses": 0, "win_pct": None}
    assert ov["avg_team_total"] is None
    assert ov["players"] == []
    json.dumps(ov)   # no pandas/numpy scalar leaks


def test_overall_is_json_serializable_with_data(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        two_player_round(1, "2026-05-01"),
        solo_round(2, "2026-05-02", 4),
    ])
    json.dumps(dashboard.overall())


# --- 8) overall chart series: score over time ------------------------------
def test_score_series_is_ordered_by_date(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-03", 4),
        solo_round(2, "2026-05-01", 2),
        solo_round(3, "2026-05-02", 3),
    ])
    ov = dashboard.overall()
    series = ov["score_series"]
    assert [p["date"] for p in series] == ["2026-05-01", "2026-05-02", "2026-05-03"]
    assert [p["team_total"] for p in series] == [2, 3, 4]
    assert ov["target"] == 19            # the dashed target line for the chart


# --- 9) overall chart series: shot-outcome mix -----------------------------
def test_outcome_mix_counts_counting_shots_in_order(monkeypatch, tmp_path):
    # two_player_round: good, hole (Tommy) + grounder, ob (Matt) -> one of each.
    install(monkeypatch, tmp_path, [two_player_round(1, "2026-05-01")])
    mix = dashboard.overall()["outcome_mix"]
    assert {m["outcome"]: m["count"] for m in mix} == {"ob": 1, "grounder": 1, "good": 1, "hole": 1}
    # worst -> best order (schema.OUTCOMES), zero-count outcomes omitted.
    assert [m["outcome"] for m in mix] == ["ob", "grounder", "good", "hole"]


# --- 10) per-hole blocks: context for all six holes ------------------------
def test_holes_returns_all_six_with_course_context(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [solo_round(1, "2026-05-01", 3, hole=1)])
    hs = dashboard.holes()
    assert [h["hole"] for h in hs] == [1, 2, 3, 4, 5, 6]
    h1 = hs[0]
    assert h1["par"] == 3
    assert h1["start"] and h1["target"]          # route from course.yaml
    assert h1["map"] and "hole_1" in h1["map"]    # hole 1 has a map image


def test_hole_averages_played_and_best_worst(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-01", 3, hole=1),   # 3 strokes, par 3 -> to_par 0
        solo_round(2, "2026-05-02", 5, hole=1),   # 5 strokes        -> to_par +2
    ])
    hs = {h["hole"]: h for h in dashboard.holes()}
    h1 = hs[1]
    assert h1["times_played"] == 2
    assert h1["avg_strokes"] == 4.0
    assert h1["avg_to_par"] == 1.0
    assert h1["best"] == 3
    assert h1["worst"] == 5
    # a hole never played still appears, with empty stats
    h2 = hs[2]
    assert h2["times_played"] == 0
    assert h2["avg_strokes"] is None
    assert h2["best"] is None


def test_hole_difficulty_rank(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-01", 5, hole=1),   # hole 1: +2 (hardest)
        solo_round(2, "2026-05-02", 3, hole=2),   # hole 2:  0
    ])
    hs = {h["hole"]: h for h in dashboard.holes()}
    assert hs[1]["difficulty_rank"] == 1
    assert hs[2]["difficulty_rank"] == 2
    assert hs[3]["difficulty_rank"] is None        # never played -> unranked


def test_hole_player_make_rate_matches_brain(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [two_player_round(1, "2026-05-01")])   # hole 1
    from golf import stats
    h1 = next(h for h in dashboard.holes() if h["hole"] == 1)
    rows = {p["name"]: p for p in h1["players"]}
    assert set(rows) == {"Tommy", "Matt"}
    for name in ("Tommy", "Matt"):
        brain = stats.make_rate(name, hole=1)
        assert rows[name]["make_rate"] == brain["smoothed_rate"]
        assert rows[name]["n"] == brain["n"]


# --- 11) per-hole chart series ---------------------------------------------
def test_hole_score_series_ordered_by_date(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-02", 4, hole=1),
        solo_round(2, "2026-05-01", 3, hole=1),
    ])
    h1 = next(h for h in dashboard.holes() if h["hole"] == 1)
    assert [p["date"] for p in h1["score_series"]] == ["2026-05-01", "2026-05-02"]
    assert [p["team_strokes"] for p in h1["score_series"]] == [3, 4]


def test_hole_score_distribution(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        solo_round(1, "2026-05-01", 3, hole=1),
        solo_round(2, "2026-05-02", 3, hole=1),
        solo_round(3, "2026-05-03", 5, hole=1),
    ])
    h1 = next(h for h in dashboard.holes() if h["hole"] == 1)
    dist = {d["strokes"]: d["count"] for d in h1["distribution"]}
    assert dist == {3: 2, 5: 1}
    # ascending by stroke count
    assert [d["strokes"] for d in h1["distribution"]] == [3, 5]


def test_hole_outcome_mix(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [two_player_round(1, "2026-05-01")])   # hole 1
    h1 = next(h for h in dashboard.holes() if h["hole"] == 1)
    assert {m["outcome"]: m["count"] for m in h1["outcome_mix"]} == \
        {"ob": 1, "grounder": 1, "good": 1, "hole": 1}


def test_payload_includes_overall_and_six_holes(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [solo_round(1, "2026-05-01", 3)])
    p = dashboard.payload()
    assert "overall" in p
    assert [h["hole"] for h in p["holes"]] == [1, 2, 3, 4, 5, 6]
    assert "players" in p
    json.dumps(p)   # whole envelope is serializable


# --- 12) Players tab: per-player blocks share the make-rate brain ----------
def test_players_block_matches_brain_and_orders_best_first(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [
        two_player_round(1, "2026-05-01"),
        two_player_round(2, "2026-05-02"),
    ])
    from golf import stats
    blocks = {p["name"]: p for p in dashboard.players()}
    assert set(blocks) == {"Tommy", "Matt"}
    for name in ("Tommy", "Matt"):
        brain = stats.make_rate(name)
        assert blocks[name]["overall"]["make_rate"] == brain["smoothed_rate"]
        assert blocks[name]["overall"]["n"] == brain["n"]
    # Tommy (all makes) sorts ahead of Matt (none) so the dropdown opens on him.
    order = [p["name"] for p in dashboard.players()]
    assert order.index("Tommy") < order.index("Matt")


def test_players_block_has_breakdowns_and_hand(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [two_player_round(1, "2026-05-01")])   # hole 1, tee+short
    tommy = next(p for p in dashboard.players() if p["name"] == "Tommy")
    # by_hole only lists holes actually played (just hole 1 here).
    assert [b["hole"] for b in tommy["by_hole"]] == [1]
    # by_distance only lists buckets the player shot from (tee then short).
    assert {b["distance"] for b in tommy["by_distance"]} == {"tee", "short"}
    # Tommy went good->hole, so his own mix is one good + one hole.
    assert {m["outcome"]: m["count"] for m in tommy["outcome_mix"]} == {"good": 1, "hole": 1}
    assert tommy["hand"] == "R"


def test_players_block_drops_players_with_no_shots(monkeypatch, tmp_path):
    # Only player 1 hits; player 2 is on the roster but never took a shot.
    install(monkeypatch, tmp_path, [solo_round(1, "2026-05-01", 3, pid=1)])
    names = [p["name"] for p in dashboard.players()]
    assert names == ["Tommy"]


def test_players_block_empty_data_is_safe(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [])
    assert dashboard.players() == []
    json.dumps(dashboard.payload())


# --- 7) the /api/stats endpoint (integration smoke) ------------------------
def test_api_stats_endpoint_returns_overall(monkeypatch, tmp_path):
    install(monkeypatch, tmp_path, [two_player_round(1, "2026-05-01")])
    from fastapi.testclient import TestClient
    from chatbot.server import app
    client = TestClient(app)
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert "overall" in body and "holes" in body
    assert body["overall"]["rounds_played"] == 1
