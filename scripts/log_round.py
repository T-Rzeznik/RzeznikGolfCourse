"""Interactive scramble round logger (keyboard alternative to the phone app).

Run from the repo root:   python scripts/log_round.py

Drives the game stroke-by-stroke: a random player starts each hole, you enter
every player's outcome, then pick the best ball; everyone plays from there next
stroke and the player who hit it goes first. A `hole` outcome ends the hole; if
everyone goes OB the team re-hits the same spot (+1 stroke). One mulligan/round.
"""
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golf import data as gdata  # noqa: E402
from golf import scramble  # noqa: E402
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
    ground = prompt(f"Ground ({'/'.join(schema.GROUND)}, blank to skip): ",
                    valid=schema.GROUND, allow_blank=True) or ""
    wind = prompt(f"Wind ({'/'.join(schema.WIND)}, blank to skip): ",
                  valid=schema.WIND, allow_blank=True) or ""
    notes = prompt("Round notes (optional): ", allow_blank=True) or ""

    client_round_id = str(uuid.uuid4())
    shot_rows = []
    mulligan_used = False
    non_tee = [d for d in schema.DISTANCES if d != "tee"]

    print("\n(Outcomes: " + ", ".join(schema.OUTCOMES) + ". One mulligan per round: "
          "answer y when a shot was a free do-over, then re-take it.)")

    def add(pid, stroke, order_idx, outcome, best_ball, mulligan, distance):
        # Unshaped row — commit_round assigns shot_id / round_id at the end.
        shot_rows.append({
            "player_id": pid, "hole": hole_num, "stroke_num": stroke,
            "shot_order": order_idx, "outcome": outcome, "distance": distance,
            "best_ball": best_ball, "mulligan": mulligan,
            "ts": int(time.time() * 1000),
        })

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
            # Distance to target of the shared spot (stroke 1 is always the tee).
            dist = "tee" if stroke == 1 else prompt(
                f"   distance to target ({'/'.join(non_tee)}): ", valid=non_tee)
            stroke_outcomes = {}
            for idx, pid in enumerate(order, start=1):
                while True:
                    outcome = prompt(f"   {names[pid]}'s outcome: ", valid=schema.OUTCOMES)
                    if not mulligan_used:
                        ans = prompt("     mulligan (free do-over)? y/N: ",
                                     allow_blank=True) or "n"
                        if ans.lower().startswith("y"):
                            add(pid, stroke, idx, outcome, False, True, dist)
                            mulligan_used = True
                            print("     (mulligan logged — now re-take the shot)")
                            continue
                    break
                stroke_outcomes[pid] = (idx, outcome)

            outs = [stroke_outcomes[p][1] for p in order]
            elig = scramble.suggest_best_ball(outs)  # indices into `order`
            holers = [p for p in order if stroke_outcomes[p][1] == "hole"]
            # A ball that's OB/skipped can't be kept; the rest are legal to keep.
            legal = [p for p in order if stroke_outcomes[p][1] not in ("ob", "skip")]
            best = set()
            if holers:
                best = set(holers)
            elif not legal:
                print("   Everyone OB — re-hit from the same spot (+1 stroke).")
            elif len(legal) == 1:
                best = {legal[0]}  # no choice to make
            else:
                # Suggest the rank-best ball, but let the user keep any legal ball
                # (a worse outcome can be better positioned). Enter = the suggestion.
                suggested = order[min(elig)]
                print("   Who had the best ball? (Enter = suggested)")
                for p in legal:
                    star = "  <- suggested" if p == suggested else ""
                    print(f"     {p}: {names[p]} ({stroke_outcomes[p][1]}){star}")
                bid = prompt(f"   best ball player id [{suggested}]: ",
                             valid=[str(p) for p in legal], allow_blank=True)
                best = {suggested if bid is None else int(bid)}

            for pid in order:
                idx, outcome = stroke_outcomes[pid]
                add(pid, stroke, idx, outcome, pid in best, False, dist)

            if holers:
                print(f"   Holed! Team took {stroke} strokes on hole {hole_num}.")
                break
            if best:  # someone advanced — they lead next stroke; all-OB keeps order
                leader = next(iter(best))
                order = [leader] + [p for p in player_ids if p != leader]
            stroke += 1

    # Validate-before-write: commit_round refuses an invalid round outright, so a
    # mis-entry can't land on disk (unlike the old write-then-warn flow).
    try:
        round_id = gdata.commit_round(shot_rows, {
            "date": date, "ground": ground, "wind": wind,
            "client_round_id": client_round_id, "notes": notes})
    except gdata.RoundRejected as rej:
        print("\n!! Round REJECTED — nothing was saved. Problems:")
        for p in rej.problems:
            print(f"  - {p}")
        print("Fix the entries and log the round again.")
        sys.exit(1)

    print(f"\nSaved round {round_id} ({len(shot_rows)} shots). Validation: clean.")


if __name__ == "__main__":
    main()
