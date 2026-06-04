"""Generate synthetic rounds so you can exercise the analysis/ML pipeline
before you've logged real games.

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


def simulate_hole(hole, round_id, player_id, skill, shot_id, allow_mulligan):
    """Return (rows, shot_id, used_mulligan) for one hole (sand wedge only)."""
    par = hole["par"]
    # strokes ~ par +/- a bit, nudged by skill
    strokes = max(2, round(RNG.gauss(par + 0.4 - skill, 0.9)))
    rows = []
    used_mulligan = False

    def add(shot_num, lie, result, holed, mulligan):
        nonlocal shot_id
        rows.append({
            "shot_id": shot_id, "round_id": round_id, "player_id": player_id,
            "hole": hole["hole"], "shot_num": shot_num,
            "distance_yds": round(RNG.uniform(5, 35), 1),
            "lie": lie, "result": result, "holed": holed, "mulligan": mulligan,
        })
        shot_id += 1

    for s in range(1, strokes + 1):
        holed = s == strokes
        lie = "tee" if s == 1 else RNG.choice(["fairway", "rough", "green", "green"])
        result = "holed" if holed else RNG.choice(
            ["fairway", "rough", "green", "green", "sand", "ob"]
        )
        # Rarely, the tee shot was a do-over: log a discarded mulligan first.
        if s == 1 and allow_mulligan and not used_mulligan and RNG.random() < 0.25:
            add(1, "tee", "ob", False, True)
            used_mulligan = True
        add(s, lie, result, holed, False)
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
        for pid in players:
            for hole in holes:
                rows, shot_id, used = simulate_hole(
                    hole, round_id, pid, skills[pid], shot_id,
                    allow_mulligan=not round_mulligan_used)
                round_mulligan_used = round_mulligan_used or used
                shot_rows.extend(rows)

    if args.into_data:
        out_dir = schema.DATA_DIR
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
