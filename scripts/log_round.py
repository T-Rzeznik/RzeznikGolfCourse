"""Interactive scramble round logger (keyboard alternative to the phone app).

Run from the repo root:   python scripts/log_round.py

Drives the game stroke-by-stroke: a random player starts each hole, you enter
every player's outcome, then pick the best ball; everyone plays from there next
stroke and the player who hit it goes first. A `hole` outcome ends the hole; if
everyone goes OB the team re-hits the same spot (+1 stroke). One mulligan/round.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golf import data as gdata  # noqa: E402
from golf import schema  # noqa: E402

RNG = random.Random()


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
    names = dict(zip(players["player_id"], players["name"]))
    print(f"\n=== {course['course']['name']} — log a scramble round ===")
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
    mulligan_used = False

    print("\n(Outcomes: " + ", ".join(schema.OUTCOMES) + ". One mulligan per round: "
          "answer y when a shot was a free do-over, then re-take it.)")

    def add(pid, stroke, order_idx, outcome, best_ball, mulligan):
        nonlocal shot_id
        shot_rows.append({
            "shot_id": shot_id, "round_id": round_id, "player_id": pid,
            "hole": hole_num, "stroke_num": stroke, "shot_order": order_idx,
            "outcome": outcome, "best_ball": best_ball, "mulligan": mulligan,
        })
        shot_id += 1

    for hole_num, hole in holes.items():
        print(f"\n=== Hole {hole_num} (par {hole['par']}): {hole['start']} -> {hole['target']} ===")
        if hole.get("notes"):
            print(f"  note: {hole['notes']}")
        order = player_ids[:]
        RNG.shuffle(order)
        stroke = 1
        while True:
            spot = "the tee" if stroke == 1 else "the chosen ball"
            print(f"\n Stroke {stroke} — everyone plays from {spot}. "
                  f"Order: {', '.join(names[p] for p in order)}")
            stroke_outcomes = {}
            for idx, pid in enumerate(order, start=1):
                while True:
                    outcome = prompt(f"   {names[pid]}'s outcome: ", valid=schema.OUTCOMES)
                    if not mulligan_used:
                        ans = prompt("     mulligan (free do-over)? y/N: ",
                                     allow_blank=True) or "n"
                        if ans.lower().startswith("y"):
                            add(pid, stroke, idx, outcome, False, True)
                            mulligan_used = True
                            print("     (mulligan logged — now re-take the shot)")
                            continue
                    break
                stroke_outcomes[pid] = (idx, outcome)

            holers = [p for p in order if stroke_outcomes[p][1] == "hole"]
            all_ob = all(stroke_outcomes[p][1] == "ob" for p in order)
            best = set()
            if holers:
                best = set(holers)
            elif all_ob:
                print("   Everyone OB — re-hit from the same spot (+1 stroke).")
            else:
                non_ob = [p for p in order if stroke_outcomes[p][1] != "ob"]
                if len(non_ob) == 1:
                    best = {non_ob[0]}
                else:
                    print("   Who had the best ball?")
                    for p in non_ob:
                        print(f"     {p}: {names[p]} ({stroke_outcomes[p][1]})")
                    bid = prompt("   best ball player id: ",
                                 valid=[str(p) for p in non_ob])
                    best = {int(bid)}

            for pid in order:
                idx, outcome = stroke_outcomes[pid]
                add(pid, stroke, idx, outcome, pid in best, False)

            if holers:
                print(f"   Holed! Team took {stroke} strokes on hole {hole_num}.")
                break
            if not all_ob:
                leader = next(iter(best))
                order = [leader] + [p for p in player_ids if p != leader]
            stroke += 1

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
    else:
        print("Validation: clean.")


if __name__ == "__main__":
    main()
