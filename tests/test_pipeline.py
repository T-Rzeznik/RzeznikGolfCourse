"""Tests for the data pipeline: the scramble invariant (validate), the derived
scorecards, and the importer's dedup + validate-before-write guarantees.

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


# --- Builders --------------------------------------------------------------
def shot(shot_id, rid, pid, hole, stroke, order, outcome, distance,
         best=False, mull=False, ts=0):
    return {"shot_id": shot_id, "round_id": rid, "player_id": pid, "hole": hole,
            "stroke_num": stroke, "shot_order": order, "outcome": outcome,
            "distance": distance, "best_ball": best, "mulligan": mull, "ts": ts}


def valid_solo_round():
    """One player, hole 1 (par 3): good off the tee, then holes from short range."""
    shots = pd.DataFrame([
        shot(1, 1, 1, 1, 1, 1, "good", "tee", best=True),
        shot(2, 1, 1, 1, 2, 1, "hole", "short", best=True),
    ])
    rounds = pd.DataFrame([{
        "round_id": 1, "date": "2026-05-01", "players": "1",
        "ground": "dry", "wind": "calm", "client_round_id": "cr-1", "notes": ""}])
    players = pd.DataFrame([{"player_id": 1, "name": "Tommy", "hand": "R", "notes": ""}])
    return shots, rounds, players


# --- The happy path --------------------------------------------------------
def test_valid_round_is_clean():
    shots, rounds, players = valid_solo_round()
    assert gdata.validate(shots=shots, rounds=rounds, players=players) == []


def test_blank_distance_is_allowed():
    """Logs predating distance capture leave it blank — that must validate clean."""
    shots, rounds, players = valid_solo_round()
    shots = shots.assign(distance=None)
    assert gdata.validate(shots=shots, rounds=rounds, players=players) == []


def test_generated_sample_validates(monkeypatch, tmp_path):
    """The synthetic data the whole project leans on must satisfy the invariant."""
    import scripts.generate_sample_data as gen
    sample = ROOT / "data" / "sample"
    monkeypatch.setattr(schema, "PLAYERS_FILE", sample / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", sample / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", sample / "shots.csv")
    monkeypatch.setattr(sys, "argv", ["generate_sample_data.py", "--rounds", "8"])
    gen.RNG.seed(42)
    gen.main()
    assert gdata.validate() == []


# --- Each invariant rule should be caught ----------------------------------
@pytest.mark.parametrize("mutate, needle", [
    # distance rules (the new capture)
    (lambda s: s.assign(distance=s["distance"].mask(s["stroke_num"] == 1, "mid")),
     "stroke 1 distance"),
    (lambda s: s.assign(distance=s["distance"].mask(s["stroke_num"] == 2, "tee")),
     "'tee' only valid on stroke 1"),
    (lambda s: _set(s, 2, "distance", "banana"), "Unknown distances"),
    # core scramble rules
    (lambda s: _set(s, 1, "outcome", "wormburner"), "Unknown outcomes"),
    (lambda s: _set(s, 1, "hole", 99), "Holes not on the course"),
    (lambda s: _set(s, 1, "stroke_num", 3), "stroke numbers not"),      # gap: 2,3
    (lambda s: _set(s, 1, "outcome", "hole"), "outcomes on strokes"),   # hole not on final
])
def test_validate_catches(mutate, needle):
    shots, rounds, players = valid_solo_round()
    bad = mutate(shots)
    problems = gdata.validate(shots=bad, rounds=rounds, players=players)
    assert any(needle in p for p in problems), f"expected '{needle}' in {problems}"


def test_mixed_distance_within_stroke_caught():
    # Two players: same stroke must share one distance.
    shots = pd.DataFrame([
        shot(1, 1, 1, 1, 1, 1, "good", "tee", best=True),
        shot(2, 1, 2, 1, 1, 2, "grounder", "tee"),
        shot(3, 1, 1, 1, 2, 1, "hole", "short", best=True),
        shot(4, 1, 2, 1, 2, 2, "good", "mid"),   # <- disagrees with player 1's "short"
    ])
    rounds = pd.DataFrame([{"round_id": 1, "date": "2026-05-01", "players": "1|2",
                            "ground": "", "wind": "", "client_round_id": "cr-2", "notes": ""}])
    players = pd.DataFrame([{"player_id": i, "name": n, "hand": "R", "notes": ""}
                            for i, n in [(1, "Tommy"), (2, "Alex")]])
    problems = gdata.validate(shots=shots, rounds=rounds, players=players)
    assert any("mixed distances" in p for p in problems), problems


def test_bad_conditions_caught():
    shots, rounds, players = valid_solo_round()
    rounds = rounds.assign(wind="hurricane")
    problems = gdata.validate(shots=shots, rounds=rounds, players=players)
    assert any("Unknown wind" in p for p in problems), problems


def _four_player_hole(rid="1"):
    """A clean 4-player scramble on hole 1 (par 3): all four hit each stroke,
    one best ball per stroke, the team holes on stroke 2."""
    shots = pd.DataFrame([
        shot(1, 1, 1, 1, 1, 1, "good", "tee", best=True),
        shot(2, 1, 2, 1, 1, 2, "grounder", "tee"),
        shot(3, 1, 3, 1, 1, 3, "short_pop", "tee"),
        shot(4, 1, 4, 1, 1, 4, "good", "tee"),
        shot(5, 1, 1, 1, 2, 1, "hole", "short", best=True),
        shot(6, 1, 2, 1, 2, 2, "overshoot", "short"),
        shot(7, 1, 3, 1, 2, 3, "overshoot", "short"),
        shot(8, 1, 4, 1, 2, 4, "good", "short"),
    ])
    rounds = pd.DataFrame([{"round_id": 1, "date": "2026-06-13", "players": "1|2|3|4",
                            "ground": "dry", "wind": "calm", "client_round_id": "cr-4p", "notes": ""}])
    players = pd.DataFrame([{"player_id": i, "name": n, "hand": "R", "notes": ""}
                            for i, n in [(1, "Tommy"), (2, "Matt"), (3, "Mia"), (4, "Kelsey")]])
    return shots, rounds, players


def test_four_player_round_is_clean():
    """4-player scrambles are valid (ADR 0001 raised the cap from 3 to 4)."""
    shots, rounds, players = _four_player_hole()
    assert gdata.validate(shots=shots, rounds=rounds, players=players) == []


def test_five_player_round_is_rejected():
    """The bound is 1-4: a fifth player still trips the roster check."""
    shots, rounds, players = _four_player_hole()
    shots = pd.concat([shots, pd.DataFrame([
        shot(9, 1, 5, 1, 1, 5, "good", "tee"),
        shot(10, 1, 5, 1, 2, 5, "overshoot", "short"),
    ])], ignore_index=True)
    rounds = rounds.assign(players="1|2|3|4|5")
    players = pd.concat([players, pd.DataFrame(
        [{"player_id": 5, "name": "Sam", "hand": "R", "notes": ""}])], ignore_index=True)
    problems = gdata.validate(shots=shots, rounds=rounds, players=players)
    assert any("expected 1-4" in p for p in problems), problems


# --- Derived scorecards ----------------------------------------------------
def test_scorecards(monkeypatch, tmp_path):
    shots, rounds, players = valid_solo_round()
    _write_dataset(monkeypatch, tmp_path, shots, rounds, players)
    hs = gdata.hole_scores()
    assert hs.loc[hs["hole"] == 1, "team_strokes"].iloc[0] == 2   # holed on stroke 2
    assert hs.loc[hs["hole"] == 1, "to_par"].iloc[0] == -1        # par 3
    rs = gdata.round_scores()
    assert bool(rs["won"].iloc[0]) is True                        # 2 < target 19


# --- Importer: validate-before-write + dedup -------------------------------
def test_importer_dedup_and_gating(monkeypatch, tmp_path):
    import scripts.import_log as imp
    # Point the schema at empty temp CSVs.
    _fresh_csvs(monkeypatch, tmp_path)

    payload = {
        "course": "Rzeznik Golf Course", "schema_version": schema.SCHEMA_VERSION,
        "client_round_id": "abc-123", "date": "2026-05-02",
        "ground": "dry", "wind": "calm", "notes": "",
        "players": [{"player_id": 1, "name": "Tommy"}],
        "shots": [
            {"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
             "outcome": "good", "distance": "tee", "best_ball": True, "ts": 1},
            {"player_id": 1, "hole": 1, "stroke_num": 2, "shot_order": 1,
             "outcome": "hole", "distance": "short", "best_ball": True, "ts": 2},
        ],
    }
    pfile = tmp_path / "round.json"
    pfile.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["import_log.py", str(pfile)])
    imp.main()
    assert len(gdata.load_rounds()) == 1
    assert len(gdata.load_shots()) == 2

    # Re-importing the SAME round (same client_round_id) must be a no-op.
    imp.main()
    assert len(gdata.load_rounds()) == 1, "duplicate import created a second round"
    assert len(gdata.load_shots()) == 2

    # A broken round must be REJECTED (SystemExit) and write nothing new.
    bad = dict(payload, client_round_id="bad-1")
    bad["shots"] = [dict(payload["shots"][0], distance="tee"),
                    dict(payload["shots"][1], distance="tee")]  # 'tee' on stroke 2
    bfile = tmp_path / "bad.json"
    bfile.write_text(json.dumps(bad), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["import_log.py", str(bfile)])
    with pytest.raises(SystemExit):
        imp.main()
    assert len(gdata.load_rounds()) == 1, "rejected round leaked into rounds.csv"
    assert (Path(schema.DATA_DIR) / "quarantine").exists()


def test_importer_creates_new_players(monkeypatch, tmp_path):
    """Player resolution stays the importer's job — a never-seen name is created."""
    import scripts.import_log as imp
    _fresh_csvs(monkeypatch, tmp_path)

    payload = {
        "course": "Rzeznik Golf Course", "schema_version": schema.SCHEMA_VERSION,
        "client_round_id": "newp-1", "date": "2026-05-04",
        "players": [{"player_id": 1, "name": "Riley"}],
        "shots": [
            {"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
             "outcome": "good", "distance": "tee", "best_ball": True, "ts": 1},
            {"player_id": 1, "hole": 1, "stroke_num": 2, "shot_order": 1,
             "outcome": "hole", "distance": "short", "best_ball": True, "ts": 2},
        ],
    }
    pfile = tmp_path / "newp.json"
    pfile.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["import_log.py", str(pfile)])
    imp.main()

    assert "Riley" in set(gdata.load_players()["name"])
    assert len(gdata.load_shots()) == 2


def test_importer_resolves_players_by_name_not_round_local_id(monkeypatch, tmp_path):
    """A round-local player_id must never be matched against the repo roster.

    The logger numbers players 1..n positionally per round, so "player 2" is just
    "whoever played second this round" — a different person each game. Resolving
    that local id against the roster would attribute, e.g., Mia's shots to whoever
    holds repo id 2 (Matt). Players are keyed by NAME; an unseen name is created.
    """
    import scripts.import_log as imp
    _fresh_csvs(monkeypatch, tmp_path)
    _two_players().to_csv(schema.PLAYERS_FILE, index=False)  # 1=Tommy, 2=Matt

    # Local player_id 2 is *Mia*, not Matt — the collision the bug tripped on.
    payload = {
        "course": "Rzeznik Golf Course", "schema_version": schema.SCHEMA_VERSION,
        "client_round_id": "mia-1", "date": "2026-05-06",
        "players": [{"player_id": 1, "name": "Tommy"}, {"player_id": 2, "name": "Mia"}],
        "shots": [
            {"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
             "outcome": "good", "distance": "tee", "best_ball": True, "ts": 1},
            {"player_id": 2, "hole": 1, "stroke_num": 1, "shot_order": 2,
             "outcome": "grounder", "distance": "tee", "best_ball": False, "ts": 1},
            {"player_id": 1, "hole": 1, "stroke_num": 2, "shot_order": 1,
             "outcome": "hole", "distance": "short", "best_ball": True, "ts": 2},
            {"player_id": 2, "hole": 1, "stroke_num": 2, "shot_order": 2,
             "outcome": "ob", "distance": "short", "best_ball": False, "ts": 2},
        ],
    }
    pfile = tmp_path / "mia.json"
    pfile.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["import_log.py", str(pfile)])
    imp.main()

    players = gdata.load_players()
    name_to_id = dict(zip(players["name"], players["player_id"]))
    assert "Mia" in name_to_id, "a never-seen name must be created, not collapsed onto id 2"
    mia_id = name_to_id["Mia"]
    assert mia_id != name_to_id["Matt"], "Mia must not be attributed to Matt"

    # Mia's (second-player) shots land under HER id; Matt didn't touch this round.
    shots = gdata.load_shots()
    assert (shots["player_id"] == mia_id).sum() == 2
    assert (shots["player_id"] == name_to_id["Matt"]).sum() == 0


def test_interactive_logger_lands_a_clean_round(monkeypatch, tmp_path):
    """A full solo round entered at the keyboard lands as one validated round."""
    import scripts.log_round as logr
    _fresh_csvs(monkeypatch, tmp_path)
    _solo_players().to_csv(schema.PLAYERS_FILE, index=False)

    holes = gdata.load_course()["holes"]
    # Setup answers, then per hole: stroke 1 'good' (tee, no dist prompt) + no mulligan,
    # stroke 2 distance 'short' + 'hole' + no mulligan.
    answers = ["2026-05-05", "1", "", "", ""]
    for _ in holes:
        answers += ["good", "", "short", "hole", ""]
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(feed))

    logr.main()

    assert len(gdata.load_rounds()) == 1
    assert len(gdata.load_shots()) == 2 * len(holes)   # 2 strokes/hole, solo
    assert gdata.validate() == []


def test_interactive_logger_best_ball_defaults_to_suggestion(monkeypatch, tmp_path):
    """With >1 legal ball the logger prompts but accepts a blank = the suggestion,
    so a two-player round logs without the user having to name an id each stroke."""
    import scripts.log_round as logr
    _fresh_csvs(monkeypatch, tmp_path)
    _two_players().to_csv(schema.PLAYERS_FILE, index=False)

    holes = gdata.load_course()["holes"]
    answers = ["2026-05-06", "1,2", "", "", ""]
    for _ in holes:
        # stroke 1: both 'good' (no mulligan), then a blank best-ball = take suggestion
        answers += ["good", "", "good", "", ""]
        # stroke 2 from 'short': leader holes, other 'good' -> holed, no best-ball prompt
        answers += ["short", "hole", "", "good", ""]
    feed = iter(answers)
    monkeypatch.setattr("builtins.input", lambda *a: next(feed))

    logr.main()

    assert len(gdata.load_rounds()) == 1
    assert len(gdata.load_shots()) == 4 * len(holes)   # 2 players x 2 strokes/hole
    assert gdata.validate() == []


# --- Round intake: commit_round --------------------------------------------
def _solo_players():
    return pd.DataFrame([{"player_id": 1, "name": "Tommy", "hand": "R", "notes": ""}])


def _two_players():
    return pd.DataFrame([{"player_id": i, "name": n, "hand": "R", "notes": ""}
                         for i, n in [(1, "Tommy"), (2, "Matt")]])


def _clean_solo_shots():
    """Hole 1 (par 3): good off the tee, then holed from short — as UNSHAPED rows
    (no shot_id / round_id; commit_round assigns those)."""
    return [
        {"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
         "outcome": "good", "distance": "tee", "best_ball": True, "mulligan": False, "ts": 1},
        {"player_id": 1, "hole": 1, "stroke_num": 2, "shot_order": 1,
         "outcome": "hole", "distance": "short", "best_ball": True, "mulligan": False, "ts": 2},
    ]


def test_commit_round_writes_clean_round(monkeypatch, tmp_path):
    _fresh_csvs(monkeypatch, tmp_path)
    _solo_players().to_csv(schema.PLAYERS_FILE, index=False)

    rid = gdata.commit_round(_clean_solo_shots(),
                             {"date": "2026-05-02", "client_round_id": "cr-1"})

    assert rid == 1
    assert len(gdata.load_rounds()) == 1
    shots = gdata.load_shots()
    assert len(shots) == 2
    assert set(shots["shot_id"]) == {1, 2}          # ids assigned
    assert set(shots["round_id"].dropna()) == {1}


def test_commit_round_rejects_and_writes_nothing(monkeypatch, tmp_path):
    _fresh_csvs(monkeypatch, tmp_path)
    _solo_players().to_csv(schema.PLAYERS_FILE, index=False)
    # A round that never holes violates the invariant ('hole' must be on the final stroke).
    bad = [{"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
            "outcome": "good", "distance": "tee", "best_ball": True, "mulligan": False, "ts": 1}]

    with pytest.raises(gdata.RoundRejected) as exc:
        gdata.commit_round(bad, {"date": "2026-05-02"})

    assert exc.value.problems                        # carries the problems
    assert len(gdata.load_rounds()) == 0             # nothing written
    assert len(gdata.load_shots()) == 0


def test_commit_round_assigns_non_colliding_ids(monkeypatch, tmp_path):
    _fresh_csvs(monkeypatch, tmp_path)
    _solo_players().to_csv(schema.PLAYERS_FILE, index=False)

    r1 = gdata.commit_round(_clean_solo_shots(), {"date": "2026-05-01"})
    r2 = gdata.commit_round(_clean_solo_shots(), {"date": "2026-05-02"})

    assert (r1, r2) == (1, 2)
    shots = gdata.load_shots()
    assert sorted(int(x) for x in shots["shot_id"]) == [1, 2, 3, 4]
    assert sorted({int(x) for x in shots["round_id"]}) == [1, 2]


def test_commit_round_derives_roster_from_shots(monkeypatch, tmp_path):
    _fresh_csvs(monkeypatch, tmp_path)
    _two_players().to_csv(schema.PLAYERS_FILE, index=False)
    # Hole 1: both tee off, p1's ball kept; both hole out on the final stroke.
    shots = [
        {"player_id": 1, "hole": 1, "stroke_num": 1, "shot_order": 1,
         "outcome": "good", "distance": "tee", "best_ball": True, "mulligan": False, "ts": 1},
        {"player_id": 2, "hole": 1, "stroke_num": 1, "shot_order": 2,
         "outcome": "grounder", "distance": "tee", "best_ball": False, "mulligan": False, "ts": 2},
        {"player_id": 1, "hole": 1, "stroke_num": 2, "shot_order": 1,
         "outcome": "hole", "distance": "short", "best_ball": True, "mulligan": False, "ts": 3},
        {"player_id": 2, "hole": 1, "stroke_num": 2, "shot_order": 2,
         "outcome": "hole", "distance": "short", "best_ball": True, "mulligan": False, "ts": 4},
    ]

    rid = gdata.commit_round(shots, {"date": "2026-05-03"})

    rounds = gdata.load_rounds()
    assert rounds.loc[rounds["round_id"] == rid, "players"].iloc[0] == "1|2"


# --- helpers ---------------------------------------------------------------
def _set(df, shot_id, col, val):
    df = df.copy()
    df.loc[df["shot_id"] == shot_id, col] = val
    return df


def _write_dataset(monkeypatch, tmp_path, shots, rounds, players):
    monkeypatch.setattr(schema, "DATA_DIR", tmp_path)
    monkeypatch.setattr(schema, "PLAYERS_FILE", tmp_path / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", tmp_path / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", tmp_path / "shots.csv")
    players.to_csv(tmp_path / "players.csv", index=False)
    rounds.to_csv(tmp_path / "rounds.csv", index=False)
    shots.to_csv(tmp_path / "shots.csv", index=False)


def _fresh_csvs(monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "DATA_DIR", tmp_path)
    monkeypatch.setattr(schema, "PLAYERS_FILE", tmp_path / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", tmp_path / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", tmp_path / "shots.csv")
    pd.DataFrame(columns=schema.PLAYER_COLUMNS).to_csv(tmp_path / "players.csv", index=False)
    pd.DataFrame(columns=schema.ROUND_COLUMNS).to_csv(tmp_path / "rounds.csv", index=False)
    pd.DataFrame(columns=schema.SHOT_COLUMNS).to_csv(tmp_path / "shots.csv", index=False)
