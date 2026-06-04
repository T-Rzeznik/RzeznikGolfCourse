"""Generate synthetic scramble rounds so you can exercise the analysis/ML
pipeline before you've logged real games.

    python scripts/generate_sample_data.py            # writes to data/sample/
    python scripts/generate_sample_data.py --into-data # appends to real data (careful!)

By default it writes to data/sample/ so it never touches your real CSVs.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from golf import data as gdata  # noqa: E402
from golf import schema  # noqa: E402

# Deterministic so the sample data is reproducible.
RNG = random.Random(42)

SAMPLE_PLAYERS = [(1, "Tommy", "R"), (2, "Alex", "R"), (3, "Sam", "L")]
MAX_STROKES = 8  # safety cap; force a hole-out if we somehow reach it


def outcome_weights(stroke, skill):
    """Weights over schema.OUTCOMES; `hole` grows with stroke + skill so holes end.

    schema.OUTCOMES == [ob, overshoot, grounder, short_pop, good, hole]. Tuned so
    holes take ~3 strokes (you rarely hole on the first stroke from the tee).
    """
    hole_w = max(0.05, 0.15 + 0.7 * (stroke - 1) + 0.3 * skill)
    return [0.6, 0.9, 1.4, 1.6, 1.6, hole_w]


def simulate_hole(hole, round_id, players, skills, shot_id, allow_mulligan):
    """Simulate one scramble hole for the whole team.

    Returns (rows, next_shot_id, used_mulligan).
    """
    rows = []
    used_mulligan = False
    order = players[:]
    RNG.shuffle(order)  # random starter for the hole
    stroke = 1

    def emit(pid, order_idx, outcome, best_ball, mulligan):
        nonlocal shot_id
        rows.append({
            "shot_id": shot_id, "round_id": round_id, "player_id": pid,
            "hole": hole["hole"], "stroke_num": stroke, "shot_order": order_idx,
            "outcome": outcome, "best_ball": best_ball, "mulligan": mulligan,
        })
        shot_id += 1

    while True:
        outcomes = {
            pid: RNG.choices(schema.OUTCOMES, weights=outcome_weights(stroke, skills[pid]))[0]
            for pid in order
        }
        if stroke >= MAX_STROKES:
            outcomes[order[0]] = "hole"  # terminate

        holers = [pid for pid in order if outcomes[pid] == "hole"]
        all_ob = all(outcomes[pid] == "ob" for pid in order)
        if holers:
            best = set(holers)
        elif all_ob:
            best = set()
        else:
            ranked = [(schema.OUTCOMES.index(outcomes[pid]), pid)
                      for pid in order if outcomes[pid] != "ob"]
            top = max(r for r, _ in ranked)
            best = {RNG.choice([pid for r, pid in ranked if r == top])}

        for idx, pid in enumerate(order, start=1):
            # Rarely, a player took a do-over: log the discarded mulligan first.
            if allow_mulligan and not used_mulligan and RNG.random() < 0.05:
                emit(pid, idx, RNG.choice(["ob", "overshoot"]), False, True)
                used_mulligan = True
            emit(pid, idx, outcomes[pid], pid in best, False)

        if holers:
            break
        if not all_ob:  # winner leads next stroke; all-OB keeps the same order
            leader = next(iter(best))
            order = [leader] + [p for p in players if p != leader]
        stroke += 1

    return rows, shot_id, used_mulligan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--into-data", action="store_true",
                    help="append to the real data/ CSVs instead of data/sample/")
    args = ap.parse_args()

    course = gdata.load_course()
    holes = course["holes"]
    skills = {1: 0.6, 2: 0.2, 3: -0.1}  # per-player skill offsets

    round_rows, shot_rows = [], []
    shot_id = 1
    base_date = pd.Timestamp("2026-04-01")
    for r in range(1, args.rounds + 1):
        round_id = r
        date = (base_date + pd.Timedelta(days=3 * r)).date().isoformat()
        players = [1, 2, 3]
        round_rows.append({"round_id": round_id, "date": date,
                           "players": "|".join(map(str, players)), "notes": "sample"})
        round_mulligan_used = False  # one per round for the group
        for hole in holes:
            rows, shot_id, used = simulate_hole(
                hole, round_id, players, skills, shot_id,
                allow_mulligan=not round_mulligan_used)
            round_mulligan_used = round_mulligan_used or used
            shot_rows.extend(rows)

    if args.into_data:
        pd.DataFrame(SAMPLE_PLAYERS, columns=["player_id", "name", "hand"]).assign(
            notes="").to_csv(schema.PLAYERS_FILE, index=False)
        gdata.append_rounds(round_rows)
        gdata.append_shots(shot_rows)
        print(f"Appended {len(round_rows)} rounds / {len(shot_rows)} shots to data/.")
    else:
        out_dir = schema.DATA_DIR / "sample"
        out_dir.mkdir(exist_ok=True)
        pd.DataFrame(SAMPLE_PLAYERS, columns=["player_id", "name", "hand"]).assign(
            notes="").to_csv(out_dir / "players.csv", index=False)
        pd.DataFrame(round_rows, columns=schema.ROUND_COLUMNS).to_csv(
            out_dir / "rounds.csv", index=False)
        pd.DataFrame(shot_rows, columns=schema.SHOT_COLUMNS).to_csv(
            out_dir / "shots.csv", index=False)
        print(f"Wrote sample data to {out_dir} ({len(round_rows)} rounds, "
              f"{len(shot_rows)} shots).")


if __name__ == "__main__":
    main()
