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
from golf import scramble  # noqa: E402
from golf import schema  # noqa: E402

# Deterministic so the sample data is reproducible.
RNG = random.Random(42)

SAMPLE_PLAYERS = [(1, "Tommy", "R"), (2, "Alex", "R"), (3, "Sam", "L")]
MAX_STROKES = 8  # safety cap; force a hole-out if we somehow reach it


def outcome_weights(stroke, skill):
    """Weights over schema.OUTCOMES; `hole` grows with stroke + skill so holes end.

    schema.OUTCOMES == [skip, ob, overshoot, grounder, short_pop, good, hole]. Tuned
    so holes take ~3 strokes (you rarely hole on the first stroke from the tee).
    `skip` (a player sitting out) is never auto-generated, so its weight is 0.
    """
    hole_w = max(0.05, 0.15 + 0.7 * (stroke - 1) + 0.3 * skill)
    return [0.0, 0.6, 0.9, 1.4, 1.6, 1.6, hole_w]


def stroke_distance(stroke):
    """Spot the team shoots from: tee on stroke 1, then drifting nearer the hole."""
    if stroke == 1:
        return "tee"
    buckets = ["long", "mid", "short", "tap_in"]
    center = min(len(buckets) - 1, stroke - 2)  # later strokes tend to be closer
    weights = [max(0.1, 1.0 - 0.5 * abs(i - center)) for i in range(len(buckets))]
    return RNG.choices(buckets, weights=weights)[0]


def simulate_hole(hole, players, skills, allow_mulligan):
    """Simulate one scramble hole for the whole team.

    Emits UNSHAPED shot rows (no shot_id / round_id / ts — commit_round assigns
    the ids, and main stamps ts per round). Returns (rows, used_mulligan).
    """
    rows = []
    used_mulligan = False
    order = players[:]
    RNG.shuffle(order)  # random starter for the hole
    stroke = 1
    distance = "tee"

    def emit(pid, order_idx, outcome, best_ball, mulligan):
        rows.append({
            "player_id": pid,
            "hole": hole["hole"], "stroke_num": stroke, "shot_order": order_idx,
            "outcome": outcome, "distance": distance, "best_ball": best_ball,
            "mulligan": mulligan,
        })

    while True:
        distance = stroke_distance(stroke)
        outcomes = {
            pid: RNG.choices(schema.OUTCOMES, weights=outcome_weights(stroke, skills[pid]))[0]
            for pid in order
        }
        if stroke >= MAX_STROKES:
            outcomes[order[0]] = "hole"  # terminate

        outs = [outcomes[pid] for pid in order]
        elig = scramble.suggest_best_ball(outs)  # indices into `order`
        holers = [order[i] for i, o in enumerate(outs) if o == "hole"]
        if holers:
            best = set(holers)
        elif not elig:
            best = set()  # everyone OB/skipped — no advance, re-hit the same spot
        else:
            best = {order[RNG.choice(sorted(elig))]}  # keep one of the tied-best

        for idx, pid in enumerate(order, start=1):
            # Rarely, a player took a do-over: log the discarded mulligan first.
            if allow_mulligan and not used_mulligan and RNG.random() < 0.05:
                emit(pid, idx, RNG.choice(["ob", "overshoot"]), False, True)
                used_mulligan = True
            emit(pid, idx, outcomes[pid], pid in best, False)

        if holers:
            break
        if best:  # someone advanced — they lead next stroke; all-OB keeps the order
            leader = next(iter(best))
            order = [leader] + [p for p in players if p != leader]
        stroke += 1

    return rows, used_mulligan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--into-data", action="store_true",
                    help="append to the real data/ CSVs instead of data/sample/")
    args = ap.parse_args()

    course = gdata.load_course()
    holes = course["holes"]
    skills = {1: 0.6, 2: 0.2, 3: -0.1}  # per-player skill offsets
    sample_players = pd.DataFrame(
        SAMPLE_PLAYERS, columns=["player_id", "name", "hand"]).assign(notes="")

    # Every round lands through commit_round, so the generator validates each round
    # before it's written and can never produce data that violates the invariant.
    # Default mode writes a fresh data/sample/; --into-data appends to the real CSVs.
    if not args.into_data:
        out_dir = schema.DATA_DIR / "sample"
        out_dir.mkdir(exist_ok=True)
        schema.PLAYERS_FILE = out_dir / "players.csv"
        schema.ROUNDS_FILE = out_dir / "rounds.csv"
        schema.SHOTS_FILE = out_dir / "shots.csv"
        pd.DataFrame(columns=schema.ROUND_COLUMNS).to_csv(schema.ROUNDS_FILE, index=False)
        pd.DataFrame(columns=schema.SHOT_COLUMNS).to_csv(schema.SHOTS_FILE, index=False)
    sample_players.to_csv(schema.PLAYERS_FILE, index=False)

    base_date = pd.Timestamp("2026-04-01")
    n_rounds = n_shots = 0
    for r in range(1, args.rounds + 1):
        day = base_date + pd.Timedelta(days=3 * r)
        date = day.date().isoformat()
        round_ts0 = int(day.timestamp() * 1000)  # ms at midnight of the round's date
        players = [1, 2, 3]
        # Draw conditions before simulating (keeps the RNG sequence reproducible).
        ground, wind = RNG.choice(schema.GROUND), RNG.choice(schema.WIND)

        shot_rows = []
        round_mulligan_used = False  # one per round for the group
        for hole in holes:
            rows, used = simulate_hole(
                hole, players, skills, allow_mulligan=not round_mulligan_used)
            round_mulligan_used = round_mulligan_used or used
            shot_rows.extend(rows)
        for idx, row in enumerate(shot_rows):
            row["ts"] = round_ts0 + idx * 30_000

        gdata.commit_round(shot_rows, {
            "date": date, "ground": ground, "wind": wind,
            "client_round_id": f"sample-{r}", "notes": "sample"})
        n_rounds += 1
        n_shots += len(shot_rows)

    where = "data/" if args.into_data else str(schema.DATA_DIR / "sample")
    print(f"Wrote {n_rounds} rounds / {n_shots} shots to {where}.")


if __name__ == "__main__":
    main()
