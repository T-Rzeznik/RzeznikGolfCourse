"""Import a round logged in the phone web app (webapp/index.html).

The app gives you a JSON payload (email / share / copy). Bring it to the repo and:

    python scripts/import_log.py round.json      # from a saved/emailed file
    python scripts/import_log.py                 # then paste the JSON + Ctrl-Z, Enter (Windows)

It assigns global round/shot ids, creates any new players, then:
  * refuses to re-import a round it has already seen (via client_round_id);
  * validates the round *before* writing, so a malformed round never lands in
    the CSVs (it's quarantined to data/quarantine/ instead);
  * appends to data/rounds.csv and data/shots.csv only once it's clean.
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
    # Tolerate an email body prefix/suffix around the JSON object.
    text = text[text.index("{"):text.rindex("}") + 1]
    return json.loads(text)


def resolve_players(payload, players: pd.DataFrame):
    """Resolve every payload player to a repo player_id, *without writing yet*.

    Returns (name_to_id, new_rows, candidate_players_df) so the caller can
    validate against the would-be roster before committing anything to disk.
    """
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
            by_name[name.lower()] = next_id
            new_rows.append({"player_id": next_id, "name": name, "hand": "", "notes": ""})
            next_id += 1

    candidate = (
        pd.concat([players, pd.DataFrame(new_rows)], ignore_index=True)
        if new_rows else players
    )
    return resolved, new_rows, candidate


def quarantine(payload: dict, tag: str) -> Path:
    qdir = schema.DATA_DIR / "quarantine"
    qdir.mkdir(exist_ok=True)
    path = qdir / f"round_{tag}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main():
    payload = read_payload()

    ver = payload.get("schema_version")
    if ver is not None and ver != schema.SCHEMA_VERSION:
        print(f"Note: payload schema_version={ver}, importer is {schema.SCHEMA_VERSION}. "
              "Importing best-effort.")

    # --- Duplicate guard: a round carries a stable client_round_id ----------
    rounds = gdata.load_rounds()
    crid = payload.get("client_round_id")
    if crid and "client_round_id" in rounds.columns and \
            crid in set(rounds["client_round_id"].dropna().astype(str)):
        print(f"Already imported (client_round_id={crid}). Nothing to do.")
        return
    if not crid:
        print("Warning: payload has no client_round_id — can't guard against a "
              "duplicate import of this round.")

    players = gdata.load_players()
    name_to_id, new_player_rows, cand_players = resolve_players(payload, players)
    local_to_repo = {p.get("player_id"): name_to_id[p["name"]] for p in payload["players"]}

    # Shape the shots (ids are assigned by commit_round) and hand the round to the
    # single intake seam. Player resolution and the duplicate guard above stay our
    # job; commit_round owns id assignment, validate-before-write, and atomic append.
    shot_rows = [{
        "player_id": local_to_repo[sh["player_id"]],
        "hole": sh["hole"], "stroke_num": sh["stroke_num"],
        "shot_order": sh.get("shot_order"),
        "outcome": sh["outcome"],
        "distance": sh.get("distance"),
        "best_ball": bool(sh.get("best_ball", False)),
        "mulligan": bool(sh.get("mulligan", False)),
        "ts": sh.get("ts"),
    } for sh in payload["shots"]]

    round_meta = {
        "date": payload["date"],
        "ground": payload.get("ground", ""), "wind": payload.get("wind", ""),
        "client_round_id": crid or "", "notes": payload.get("notes", ""),
    }

    try:
        round_id = gdata.commit_round(shot_rows, round_meta, players=cand_players)
    except gdata.RoundRejected as rej:
        path = quarantine(payload, crid or "rejected")
        print("!! Round REJECTED — not written. Problems:")
        for p in rej.problems:
            print(f"  - {p}")
        print(f"\nQuarantined the raw payload to {path}")
        print("Fix the round (or the validator) and re-import; nothing was changed.")
        sys.exit(1)

    # --- Clean: persist any new players (round + shots are already written) ---
    if new_player_rows:
        cand_players.to_csv(schema.PLAYERS_FILE, index=False)
        print(f"Added players: {[r['name'] for r in new_player_rows]}")
    player_ids = sorted(set(local_to_repo.values()))
    print(f"Imported round {round_id}: {len(shot_rows)} shots, players {player_ids}. Validation: clean.")


if __name__ == "__main__":
    main()
