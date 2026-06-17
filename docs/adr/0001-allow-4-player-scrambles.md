# ADR 0001 — Allow 4-player scrambles

- Status: Accepted
- Date: 2026-06-17

## Context

The scramble invariant (`CONTEXT.md`, enforced in `golf/data.py:validate`) capped a
team at **1–3 players**. The cap was a guess at the realistic group size, not a
constraint the scoring math depends on: every scramble rule is phrased per *active
player* and holds for any roster size ≥ 1 (each player hits once per stroke, one
best ball is kept, team score = `max(stroke_num)`).

On 2026-06-13 the first **4-player** round was logged ("First ever 4 plate game",
Tommy/Matt/Mia/Kelsey). The importer correctly refused it — the round violated the
stated invariant — and quarantined it. Nothing in the data model or scoring actually
breaks with four players; only the validator's arbitrary upper bound did.

## Decision

Raise the player-count bound to **1–4**. A four-person team is a first-class
scramble. The change is a single comparison in `validate` plus the prose/vocabulary
that mirrors it; no scoring, feature, or schema change is required.

Updated together so the docs and code stay consistent:

- `golf/data.py:validate` — `1 <= n_players <= 4` (message "expected 1-4").
- `CONTEXT.md`, `README.md`, `CLAUDE.md`, `data/course.yaml` comment — "1–4 players".
- `webapp/index.html` setup label — "best-ball scramble, 1–4".

## Consequences

- The quarantined 06-13 round imports cleanly and counts in all stats.
- If we ever want to re-tighten the cap, it lives in exactly one place in code.
- The bound stays at 4 (not unbounded) so a runaway/garbled roster is still caught;
  4 is the current real-world maximum and easy to raise again behind a new ADR.
