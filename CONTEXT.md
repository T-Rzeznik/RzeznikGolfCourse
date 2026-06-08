# Context: Rzeznik Golf Course

The domain glossary and core invariant for this project. Skills and contributors
should use these exact terms; the code already does (`golf/schema.py` is the
machine-readable source of truth, this file is the prose one).

## What this is

A stat tracker for a 6-hole backyard mini-golf course (par 20). It exists to build
a clean per-shot dataset, which is the **brain** behind a future LLM "stats caddie"
(the **mouth**) that answers natural-language questions like *"who's more likely to
make this shot?"*. The split is deliberate: the data/stats layer computes facts; the
LLM only phrases them. Never let the mouth invent numbers the brain didn't produce.

## House rules (baked into the data)

- **Sand wedge is the only legal club** — so there is no club column.
- Played as a **best-ball scramble**, 1–3 players on one team.
- **Win** = team total strokes `<` the target score (default = par total, 20).
- **One mulligan** (free do-over) per round, for the whole group.
- **Out-of-bounds (OB)** = stroke-and-distance retake (no drop); an OB ball can
  never be kept.

## Glossary

- **Round** — one play-through of the course by a team on a date (`rounds.csv`).
- **Hole** — one of the 6 holes (`course.yaml`). Holes have a `par`, and flags
  `dogleg`/`blind` that feed maps and ML features.
- **Stroke** — a team stroke index (`1..max`) within a hole. In a scramble *every
  active player hits from the same spot on the same stroke.* A hole's team score is
  `max(stroke_num)`.
- **Shot** — one player's attempt on one stroke. `shots.csv` has **one row per
  player per stroke** and is the single source of truth; scorecards are *derived*,
  never stored twice.
- **Outcome** — the controlled vocabulary, worst→best (also the quality ranking used
  to auto-suggest the best ball): `skip`, `ob`, `overshoot`, `grounder`,
  `short_pop`, `good`, `hole`.
- **Best ball** — the one shot kept after a stroke; everyone plays the next stroke
  from its spot. A `best_ball` is never an OB or skipped ball.
- **Distance** — coarse difficulty bucket for the spot the whole stroke shoots from,
  far→near: `tee`, `long`, `mid`, `short`, `tap_in`. Stroke 1 is `tee` when recorded.
  **Optional/nullable** — logs predating distance capture leave it blank, and a blank
  stroke simply isn't conditioned on distance.
- **Skip** — a player who sat out a stroke (not a real attempt). Like `ob` it can
  never be the best ball, and is excluded from the shot-outcome model.
- **Mulligan** — a discarded do-over shot (`mulligan=True`); it shares its
  `stroke_num` with the real shot and does not count.
- **Counting shots** — all shots minus mulligan do-overs; every score/validation rule
  is evaluated on these.
- **Conditions** — optional per-round `ground` (`dry`/`wet`) and `wind`
  (`calm`/`breezy`/`windy`); blank = not recorded, can't be backfilled.

## The scramble invariant

Every scoring and validation rule follows from this (enforced in `golf/data.py:validate`):

For a given `(round, hole)`, strokes are numbered `1..max` with no gaps. On each
stroke every active player has exactly one counting shot. The hole ends at the first
stroke containing a `hole` outcome (so `hole` appears only at `max`). An all-OB (or
all-skipped) stroke makes no progress and the team re-hits the same spot at
`stroke+1` — it still costs a stroke and has no best ball. Exactly one ball per
non-final stroke is the `best_ball` (none on a no-advance stroke); on the final
stroke the holing ball(s) are the `best_ball`. Team score for the hole = `max(stroke_num)`.

## Data flow

1. **Log** a round on the phone with `webapp/index.html` → exports a JSON payload.
2. **Import** with `scripts/import_log.py`: assigns global ids, creates new players,
   refuses duplicate imports (via `client_round_id`), and **validates the candidate
   round before writing** — a malformed round is quarantined to `data/quarantine/`,
   never appended.
3. **Derive** scorecards on demand in `golf/data.py` (`hole_scores`, `round_scores`,
   `player_contributions`) — always recomputed from `shots.csv`, never stored.

## Decisions

Architectural decisions live in `docs/adr/`. Respect them; if a change contradicts
one, say so explicitly rather than silently overriding it.
