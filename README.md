# Rzeznik Golf Course — stat tracker + ML

A homemade 6-hole backyard golf course, played as a **best-ball scramble**. This
repo tracks every round we play and uses the data to build ML models that predict
team scores, win probability, hole difficulty, and shot outcomes.

## The course (par 20)

| Hole | Par | Start | Target | Notes |
|------|-----|-------|--------|-------|
| 1 | 3 | Behind the pear tree | Behind the side pine | |
| 2 | 3 | Where hole 1 ends | Behind the brush well | No line of sight |
| 3 | 3 | Back-left tree, by the pine | Behind the pear tree corner | No line of sight |
| 4 | 4 | Top of the lawn by the sidewalk | Between back-left tree & porch tree | Hooks around house; **first shot lefty only** |
| 5 | 3 | Where hole 4 ends | Back-right corner, 10 ft off the small pine | Safety routing |
| 6 | 4 | Where hole 5 ends | Front tree on the Murray side (hole is left of it) | Curves around house |

Full definitions live in [`data/course.yaml`](data/course.yaml). Hole maps go in
[`maps/`](maps/README.md).

## House rules (best-ball scramble)

- **One club:** the sand wedge is the only legal club, so we don't track club choice.
- **Scramble, 1–3 players** on one team. Everyone hits from the **same spot** each
  stroke; you keep the **best ball** and all play the next stroke from there. The
  player who hit the kept ball goes first; a **random player starts each hole**.
- **Six outcomes per shot:** grounder, short pop, good, **hole**, overshoot, out of
  bounds. A `hole` ends the hole immediately. The **team score** for a hole is the
  number of strokes taken; **you "win" when the team total beats the target** (par
  total, 20) — set `target_score` in `course.yaml`.
- **One mulligan per round** for the group — a free do-over of a single player's
  shot. The discarded shot doesn't count.
- **Out of bounds:** an OB ball can't be kept. If **everyone** goes OB the team
  re-hits the same spot (it still costs a stroke). A safety rule to keep balls out
  of the neighbors' yards.
- **Pars are intentionally generous** (a par 3 where the perfect line is really a
  2) for the same reason — so players lay up instead of firing risky shots.

## How the data is structured

The **per-shot table** (`data/shots.csv`) is the single source of truth — one row
per player per stroke. Team scores are *derived* in code (team strokes for a hole =
the number of strokes taken), so they can never drift out of sync.

```
data/
  course.yaml   # the 6 holes: par, start/target, dogleg, blind, map
  players.csv   # roster
  rounds.csv    # one row per game (date, who played)
  shots.csv     # one row per player per stroke  <-- source of truth
```

Each shot records: `stroke_num` (team stroke index), `shot_order` (who hit first),
`outcome` (one of six), `best_ball` (was this ball kept), and `mulligan` (a
discarded do-over). The outcome vocabulary and the full scramble invariant are in
[`golf/schema.py`](golf/schema.py).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

## Logging a round (on your phone, on the course)

Day-to-day capture is the tap-based web app in [`webapp/index.html`](webapp/index.html):

1. **Get it on your phone once.** Easiest is GitHub Pages (after you push this
   repo): enable Pages and open `…/webapp/`. No host yet? Email `webapp/index.html`
   to yourself and open it in your phone browser. Then **Add to Home Screen** — it
   runs offline from then on.
2. **Play.** Pick the date + players; the app drives the rest. It names whose shot
   it is, you tap one of the **six outcome buttons**, and it auto-advances to the
   next player. After everyone hits it asks **who had the best ball**, then moves
   on. A `hole` ends the hole; all-OB re-hits the same spot. One **Mulligan** and an
   **Undo** are there too. Your round is saved in the browser, so a lock-screen
   won't lose it.
3. **Finish → send it to yourself** (email / share / copy).

Back at the computer, import what you sent:

```bash
python scripts/import_log.py round.json      # a saved/emailed file
python scripts/import_log.py                  # or paste the JSON, then Ctrl-Z + Enter
```

It assigns IDs, creates any new players, appends to the CSVs, and validates.

> Prefer to log from a keyboard? `python scripts/log_round.py` is an interactive
> terminal logger that does the same thing.

## Trying the pipeline before you have real data

```bash
python scripts/generate_sample_data.py     # writes synthetic data to data/sample/
```

Then open the notebook:

```bash
jupyter notebook notebooks/01_explore.ipynb
```

## The Python package

```python
from golf import data, features

data.hole_scores()          # team strokes per (round, hole)
data.round_scores()         # team total per round + won (beat target)
data.player_contributions() # per (round, player): shots, best balls, outcome counts
data.validate()             # list of data problems ([] = clean)

features.score_prediction() # X/y for predicting the team's strokes on a hole
features.win_probability()  # X/y for predicting the team beats the target
features.hole_difficulty()  # avg team strokes / score-to-par per hole
features.shot_outcome()     # X/y for predicting a shot's outcome (one of six)
```

Modeling (train/test, metrics) lives in the notebooks so it's easy to iterate;
the `golf` package just guarantees clean, leakage-free feature tables.

## Roadmap

- [ ] Add hole map images to `maps/`
- [ ] Measure and fill in `yards` for each hole in `course.yaml`
- [ ] Log real rounds
- [ ] Baseline models in `notebooks/` once enough rounds are recorded
- [ ] (Later) optional weather/conditions capture; web UI for entry
