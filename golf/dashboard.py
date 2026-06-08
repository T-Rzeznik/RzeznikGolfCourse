"""The stats-dashboard aggregation — the data behind the /stats page.

Pure like golf/stats.py: reads via golf.data / golf.stats, returns plain
JSON-serializable dicts (no pandas/numpy objects leak out). The /stats page only
*draws* what this produces, and it shares the make-rate brain with the chat
caddie so the two views can never disagree.
"""
from __future__ import annotations

from . import data as gdata
from . import schema
from . import stats


def _outcome_mix(shots) -> list[dict]:
    """Counts of each Outcome over counting shots, worst->best, zeros omitted.

    Excludes `skip` (a sat-out turn, not a real attempt) to match the make-rate
    denominator. Drives the shot-quality doughnut.
    """
    if shots.empty:
        return []
    counts = gdata.counting_shots(shots)["outcome"].value_counts()
    return [
        {"outcome": o, "count": int(counts[o])}
        for o in schema.OUTCOMES
        if o != "skip" and o in counts.index and int(counts[o]) > 0
    ]


def payload() -> dict:
    """The whole dashboard as one JSON-serializable envelope.

    The Overall tab uses `overall`; each Hole tab uses an entry in `holes`.
    """
    return {"overall": overall(), "holes": holes()}


def holes() -> list[dict]:
    """One block per course hole (1-6): context now; stats added below.

    Always returns all six holes (even unplayed ones) so the dashboard can render
    every tab with at least its header.
    """
    course = gdata.load_course()
    hs = gdata.hole_scores()
    shots = gdata.load_shots()
    rounds = gdata.load_rounds()
    date_by_round = dict(zip(rounds["round_id"], rounds["date"].astype(str))) if not rounds.empty else {}
    # Difficulty rank: 1 = hardest, from the same ordering the caddie uses.
    rank = {d["hole"]: i + 1 for i, d in enumerate(stats.hole_difficulty()["hardest_first"])}
    out = []
    for meta in sorted(course["holes"], key=lambda m: m["hole"]):
        h = int(meta["hole"])
        map_rel = meta.get("map")
        has_map = bool(map_rel) and (schema.ROOT / str(map_rel)).exists()
        sub = hs[hs["hole"] == h] if not hs.empty else hs
        played = int(len(sub))

        # Strokes-over-time line: one point per round on this hole, oldest first.
        series = []
        for row in sub.itertuples(index=False):
            series.append({"date": date_by_round.get(row.round_id, ""),
                           "team_strokes": int(row.team_strokes)})
        series.sort(key=lambda p: p["date"])
        # Score distribution: how often each stroke count happened, ascending.
        distribution = []
        if played:
            vc = sub["team_strokes"].value_counts().sort_index()
            distribution = [{"strokes": int(s), "count": int(c)} for s, c in vc.items()]
        hole_shots = shots[shots["hole"] == h] if not shots.empty else shots
        out.append({
            "hole": h,
            "par": int(meta["par"]),
            "start": str(meta.get("start", "")),
            "target": str(meta.get("target", "")),
            "notes": str(meta.get("notes", "")),
            "map": str(map_rel) if has_map else None,
            "times_played": played,
            "avg_strokes": round(float(sub["team_strokes"].mean()), 2) if played else None,
            "avg_to_par": round(float(sub["to_par"].mean()), 2) if played else None,
            "best": int(sub["team_strokes"].min()) if played else None,
            "worst": int(sub["team_strokes"].max()) if played else None,
            "difficulty_rank": rank.get(h),
            "players": _players_row(hole=h),
            "score_series": series,
            "distribution": distribution,
            "outcome_mix": _outcome_mix(hole_shots),
        })
    return out


def _players_row(hole: int | None = None) -> list[dict]:
    """Each player who has hit (optionally on one hole), with their smoothed
    make-rate from the brain.

    Reuses stats.make_rate so the dashboard can never disagree with the caddie.
    Players with no shots yet (n == 0) are left off rather than shown at the
    prior's 50%. Sorted best make-rate first.
    """
    players = gdata.load_players()
    rows = []
    for name in players["name"].astype(str):
        rate = stats.make_rate(name, hole=hole)
        if rate["n"] >= 1:
            rows.append({
                "player_id": rate["player_id"],
                "name": rate["player"],
                "make_rate": rate["smoothed_rate"],
                "n": rate["n"],
            })
    rows.sort(key=lambda r: r["make_rate"], reverse=True)
    return rows


def overall() -> dict:
    """The Overall tab's block: headline numbers across all rounds."""
    rs = gdata.round_scores()
    rounds = gdata.load_rounds()

    n = int(len(rs))
    wins = int(rs["won"].sum()) if not rs.empty else 0
    record = {
        "wins": wins,
        "losses": n - wins,
        "win_pct": round(wins / n, 3) if n else None,
    }

    best_round = None
    date_range = None
    score_series = []
    if not rs.empty:
        m = rs.merge(rounds[["round_id", "date"]], on="round_id", how="left")
        # Lowest team total wins; earliest date breaks ties.
        top = m.sort_values(["team_total", "date"]).iloc[0]
        best_round = {
            "round_id": int(top["round_id"]),
            "date": str(top["date"]),
            "team_total": int(top["team_total"]),
            "to_par": int(top["to_par"]),
            "won": bool(top["won"]),
        }
        dates = m["date"].dropna().astype(str)
        if not dates.empty:
            date_range = {"first": dates.min(), "last": dates.max()}
        # Score-over-time line: one point per round, oldest first.
        for row in m.sort_values(["date", "round_id"]).itertuples(index=False):
            score_series.append({
                "date": str(row.date),
                "team_total": int(row.team_total),
                "won": bool(row.won),
            })

    return {
        "rounds_played": n,
        "date_range": date_range,
        "best_round": best_round,
        "record": record,
        "avg_team_total": round(float(rs["team_total"].mean()), 2) if n else None,
        "avg_to_par": round(float(rs["to_par"].mean()), 2) if n else None,
        "players": _players_row(),
        "target": gdata.target_score(),
        "score_series": score_series,
        "hole_difficulty": stats.hole_difficulty()["hardest_first"],
        "outcome_mix": _outcome_mix(gdata.load_shots()),
    }
