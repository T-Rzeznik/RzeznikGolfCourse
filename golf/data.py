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
    for col in ("hole", "stroke_num", "shot_order", "ts"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("best_ball", "mulligan"):
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype("boolean")
    if "distance" not in df.columns:
        df["distance"] = pd.NA
    return df


def target_score() -> int:
    """The team total to beat (defaults to the course par total)."""
    course = load_course()["course"]
    return int(course.get("target_score", course["par_total"]))


def _flag(series: pd.Series) -> pd.Series:
    """A boolean column as a real bool Series (robust to object/NA dtypes)."""
    return series.fillna(False).astype(bool)


def counting_shots(shots: pd.DataFrame) -> pd.DataFrame:
    """Shots that count toward the score (mulligan do-overs removed)."""
    return shots[~_flag(shots["mulligan"])]


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
def validate(
    shots: pd.DataFrame | None = None,
    rounds: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> list[str]:
    """Return a list of human-readable data problems ([] means clean).

    Loads from disk by default, but accepts in-memory frames so a *candidate*
    round can be validated before it's written (see scripts/import_log.py).
    """
    problems: list[str] = []
    shots = load_shots() if shots is None else shots
    if shots.empty:
        return problems

    valid_holes = set(holes_frame()["hole"])
    players = load_players() if players is None else players
    valid_players = set(players["player_id"])
    rosters = active_players(rounds)

    # Round-level conditions are optional, but if present must be in-vocabulary.
    rounds_df = load_rounds() if rounds is None else rounds
    for col, vocab in (("ground", schema.GROUND), ("wind", schema.WIND)):
        if col in rounds_df.columns:
            vals = rounds_df[col].dropna().astype(str)
            bad = set(vals[vals.str.len() > 0]) - set(vocab)
            if bad:
                problems.append(f"Unknown {col} values: {sorted(bad)}")

    bad_outcomes = set(shots["outcome"].dropna()) - set(schema.OUTCOMES)
    if bad_outcomes:
        problems.append(f"Unknown outcomes: {sorted(bad_outcomes)}")
    if "distance" in shots.columns:
        bad_dist = set(shots["distance"].dropna().astype(str)) - set(schema.DISTANCES) - {""}
        if bad_dist:
            problems.append(f"Unknown distances: {sorted(bad_dist)}")
    bad_holes = set(shots["hole"]) - valid_holes
    if bad_holes:
        problems.append(f"Holes not on the course: {sorted(bad_holes)}")
    bad_players = set(shots["player_id"]) - valid_players
    if bad_players:
        problems.append(f"Unknown player_ids: {sorted(bad_players)}")

    # One mulligan per round; each mulligan shares its slot with a real do-over.
    mull = shots[_flag(shots["mulligan"])]
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
        if not 1 <= n_players <= 4:
            problems.append(f"Round {rid}: {n_players} players (expected 1-4)")

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

            # Distance is a property of the spot, so the whole stroke shares one
            # value. Stroke 1 is always the tee; later strokes are anything but.
            if "distance" in sgrp.columns:
                dvals = {str(d) for d in sgrp["distance"].dropna() if str(d)}
                # Distance is optional: older logs predate distance capture, so a
                # blank stroke simply isn't conditioned on distance. But any value
                # that *is* present must obey the one-spot-per-stroke and tee rules.
                if len(dvals) > 1:
                    problems.append(f"Round {rid} hole {hole} stroke {sn}: mixed distances {sorted(dvals)} (one spot per stroke)")
                elif sn == 1 and dvals and dvals != {"tee"}:
                    problems.append(f"Round {rid} hole {hole}: stroke 1 distance is {sorted(dvals)} (expected 'tee')")
                elif sn != 1 and "tee" in dvals:
                    problems.append(f"Round {rid} hole {hole} stroke {sn}: distance 'tee' only valid on stroke 1")

            best = sgrp[_flag(sgrp["best_ball"])]
            # A ball that's OB or skipped can never be kept, so if every ball is
            # OB/skipped no progress is made and the team re-hits (no best ball).
            no_advance = sgrp["outcome"].isin(["ob", "skip"]).all()
            if best["outcome"].isin(["ob", "skip"]).any():
                problems.append(f"Round {rid} hole {hole} stroke {sn}: best ball can't be an OB or skipped shot")
            if sn == max_s:
                # Final stroke: best_ball is exactly the holing ball(s).
                holing = set(sgrp[sgrp["outcome"] == "hole"]["shot_id"])
                marked = set(best["shot_id"])
                if holing != marked:
                    problems.append(
                        f"Round {rid} hole {hole} final stroke: best_ball should mark the "
                        f"holing ball(s) {sorted(holing)}, got {sorted(marked)}"
                    )
            elif no_advance:
                if len(best):
                    problems.append(f"Round {rid} hole {hole} stroke {sn}: no-advance stroke (all OB/skipped) should have no best ball")
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


# --- Round intake ----------------------------------------------------------
class RoundRejected(Exception):
    """A candidate round failed the scramble invariant; nothing was written.

    Carries `.problems`, the human-readable validation messages, so the caller
    can decide what to do (quarantine, re-prompt, or raise).
    """

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def commit_round(shot_rows: list[dict], round_meta: dict,
                 players: pd.DataFrame | None = None) -> int:
    """Land one round in the data set — the single seam every writer goes through.

    Takes UNSHAPED shot rows (no ids) with resolved `player_id`s plus round
    conditions, assigns the next `round_id`/`shot_id`, derives the roster from the
    distinct players in the shots, then validates the candidate data set before
    writing. On success it appends shots and the round atomically and returns the
    new `round_id`; on failure it raises `RoundRejected` and writes nothing.

    `players` is the roster to validate against (defaults to the players on disk);
    an importer creating new players passes its candidate roster here.
    """
    rounds = load_rounds()
    shots = load_shots()
    round_id = next_id(rounds, "round_id")
    shot_id = next_id(shots, "shot_id")

    player_ids = sorted({int(r["player_id"]) for r in shot_rows})

    new_shots = []
    for r in shot_rows:
        row = {c: r.get(c) for c in schema.SHOT_COLUMNS}
        row["shot_id"] = shot_id
        row["round_id"] = round_id
        row["best_ball"] = bool(r.get("best_ball", False))
        row["mulligan"] = bool(r.get("mulligan", False))
        new_shots.append(row)
        shot_id += 1

    round_row = {
        "round_id": round_id,
        "date": round_meta.get("date", ""),
        "players": "|".join(str(p) for p in player_ids),
        "ground": round_meta.get("ground", ""),
        "wind": round_meta.get("wind", ""),
        "client_round_id": round_meta.get("client_round_id", ""),
        "notes": round_meta.get("notes", ""),
    }

    cand_shots = pd.concat(
        [shots, pd.DataFrame(new_shots, columns=schema.SHOT_COLUMNS)],
        ignore_index=True)
    cand_rounds = pd.concat(
        [rounds, pd.DataFrame([round_row], columns=schema.ROUND_COLUMNS)],
        ignore_index=True)
    problems = validate(shots=cand_shots, rounds=cand_rounds, players=players)
    mine = [p for p in problems if f"Round {round_id}" in p or "Unknown" in p]
    if mine:
        raise RoundRejected(mine)

    append_shots(new_shots)
    append_rounds([round_row])
    return round_id
