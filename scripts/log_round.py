"""Interactive round logger.

Run from the repo root:   python scripts/log_round.py

Walks you hole-by-hole and shot-by-shot, then appends to data/rounds.csv and
data/shots.csv. Strokes-per-hole = number of shots you enter, so just log each
shot until the ball is holed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golf import data as gdata  # noqa: E402
from golf import schema  # noqa: E402


def prompt(text, valid=None, allow_blank=False, cast=str):
    while True:
        raw = input(text).strip()
        if not raw and allow_blank:
            return None
        if not raw:
            continue
        if valid and raw not in valid:
            print(f"  -> pick one of: {', '.join(valid)}")
            continue
        try:
            return cast(raw)
        except ValueError:
            print("  -> invalid value, try again")


def main():
    course = gdata.load_course()
    holes = {h["hole"]: h for h in course["holes"]}
    players = gdata.load_players()
    print(f"\n=== {course['course']['name']} — log a round ===")
    print("Players on file:")
    for _, p in players.iterrows():
        print(f"  {p['player_id']}: {p['name']}")

    date = prompt("Date (YYYY-MM-DD): ")
    pids = prompt("Player ids in this round (comma-separated): ")
    player_ids = [int(x) for x in pids.split(",")]
    notes = prompt("Round notes (optional): ", allow_blank=True) or ""

    round_id = gdata.next_id(gdata.load_rounds(), "round_id")
    shot_id = gdata.next_id(gdata.load_shots(), "shot_id")
    shot_rows = []
    mulligan_used = False  # one per round for the whole group

    print("\n(Only club is the sand wedge. OB = log result 'ob', then replay as the "
          "next shot. One mulligan per round: answer y when a shot was a free do-over.)")

    for pid in player_ids:
        name = players.loc[players["player_id"] == pid, "name"].iloc[0]
        print(f"\n--- {name} ---")
        for hole_num, hole in holes.items():
            print(f"\nHole {hole_num} (par {hole['par']}): {hole['start']} -> {hole['target']}")
            if hole.get("notes"):
                print(f"  note: {hole['notes']}")
            shot_num = 1
            strokes = 0
            while True:
                print(f" Shot {shot_num}:")
                dist = prompt("   distance_yds (blank if unknown): ",
                              allow_blank=True, cast=float)
                lie = prompt("   lie: ", valid=schema.LIES)
                result = prompt("   result: ", valid=schema.RESULTS)
                is_mull = False
                if not mulligan_used:
                    ans = prompt("   was this a mulligan (free do-over)? y/N: ",
                                 allow_blank=True) or "n"
                    is_mull = ans.lower().startswith("y")
                holed = result == "holed" and not is_mull
                shot_rows.append({
                    "shot_id": shot_id, "round_id": round_id, "player_id": pid,
                    "hole": hole_num, "shot_num": shot_num,
                    "distance_yds": dist, "lie": lie, "result": result,
                    "holed": holed, "mulligan": is_mull,
                })
                shot_id += 1
                shot_num += 1
                if is_mull:
                    mulligan_used = True
                    print("   (mulligan logged - doesn't count; now log the do-over)")
                else:
                    strokes += 1
                if holed:
                    print(f"   => {strokes} strokes on hole {hole_num}")
                    break

    gdata.append_rounds([{
        "round_id": round_id, "date": date,
        "players": "|".join(str(p) for p in player_ids), "notes": notes,
    }])
    gdata.append_shots(shot_rows)
    print(f"\nSaved round {round_id} ({len(shot_rows)} shots).")

    problems = gdata.validate()
    if problems:
        print("\n!! Validation warnings:")
        for p in problems:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
