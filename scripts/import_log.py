"""Import a round logged in the phone web app (webapp/index.html).

The app gives you a JSON payload (email / share / copy). Bring it to the repo and:

    python scripts/import_log.py round.json      # from a saved/emailed file
    python scripts/import_log.py                 # then paste the JSON + Ctrl-Z, Enter (Windows)

It assigns global round/shot ids, creates any new players, appends to
data/rounds.csv and data/shots.csv, then runs validation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from golf import data as gdata  # noqa: E402
from golf import schema  # noqa: E402


def read_payload() -> dict:
    if len(sys.argv) > 1:
        text = Path(sys.argv[1]).read_text(encoding="utf-8")
    else:
        print("Paste the round JSON, then end input (Ctrl-Z + Enter on Windows):")
        text = sys.stdin.read()
    # Tolerate an email body prefix before the JSON object.
    text = text[text.index("{"):text.rindex("}") + 1]
    return json.loads(text)


def ensure_players(payload) -> dict:
    """Make sure every player exists in players.csv. Returns name->id map used."""
    players = gdata.load_players()
    by_id = dict(zip(players["player_id"], players["name"]))
    by_name = {n.lower(): i for i, n in by_id.items()}
    new_rows, resolved = [], {}
    next_id = (max(by_id) if len(by_id) else 0) + 1

    for p in payload["players"]:
        pid, name = p.get("player_id"), p["name"]
        if pid in by_id:
            resolved[name] = pid
        elif name.lower() in by_name:
            resolved[name] = by_name[name.lower()]
        else:
            resolved[name] = next_id
            new_rows.append({"player_id": next_id, "name": name, "hand": "", "notes": ""})
            next_id += 1

    if new_rows:
        combined = pd.concat([players, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(schema.PLAYERS_FILE, index=False)
        print(f"Added players: {[r['name'] for r in new_rows]}")
    return resolved


def main():
    payload = read_payload()
    name_to_id = ensure_players(payload)
    # Map the app's local player ids -> the repo's ids, via name.
    local_to_repo = {p.get("player_id"): name_to_id[p["name"]] for p in payload["players"]}

    round_id = gdata.next_id(gdata.load_rounds(), "round_id")
    shot_id = gdata.next_id(gdata.load_shots(), "shot_id")
    player_ids = sorted(set(local_to_repo.values()))

    shot_rows = []
    for sh in payload["shots"]:
        shot_rows.append({
            "shot_id": shot_id, "round_id": round_id,
            "player_id": local_to_repo[sh["player_id"]],
            "hole": sh["hole"], "stroke_num": sh["stroke_num"],
            "shot_order": sh.get("shot_order"),
            "outcome": sh["outcome"],
            "best_ball": bool(sh.get("best_ball", False)),
            "mulligan": bool(sh.get("mulligan", False)),
        })
        shot_id += 1

    gdata.append_rounds([{
        "round_id": round_id, "date": payload["date"],
        "players": "|".join(str(p) for p in player_ids),
        "notes": payload.get("notes", ""),
    }])
    gdata.append_shots(shot_rows)
    print(f"Imported round {round_id}: {len(shot_rows)} shots, players {player_ids}.")

    problems = gdata.validate()
    if problems:
        print("\n!! Validation warnings:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Validation: clean.")


if __name__ == "__main__":
    main()
