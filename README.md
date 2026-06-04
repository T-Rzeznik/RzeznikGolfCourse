# Rzeznik Golf Course — stat tracker + ML

A homemade 6-hole backyard golf course. This repo tracks every round we play and
uses the data to build ML models that predict scores, win probability, hole
difficulty, and shot/club outcomes.

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

## House rules

- **One club:** the sand wedge is the only legal club, so we don't track club choice.
- **Best ball, 1–3 players** on one team. Each player plays their own ball; the
  team takes the best score on each hole. **You "win" when the team total beats
  the target score** (the par total, 20) — set `target_score` in `course.yaml`.
- **One mulligan per round** for the whole group — a free do-over of a single
  shot (only that one player retakes). Discarded shots don't count toward the score.
- **Out of bounds = stroke-and-distance:** you replay from the same spot (no
  dropping near the boundary). It costs a stroke. This is a safety rule to keep
  balls out of the neighbors' yards.
- **Pars are intentionally generous** (a par 3 where the perfect line is really a
  2) for the same reason — so players lay up instead of firing risky shots.

## How the data is structured

The **per-shot table** (`data/shots.csv`) is the single source of truth — one row
per shot. Strokes-per-hole is just the count of shots, so hole and round
scorecards are *derived* in code and can never drift out of sync.

```
data/
  course.yaml   # the 6 holes: par, start/target, dogleg, blind, map
  players.csv   # roster
  rounds.csv    # one row per game (date, who played)
  shots.csv     # one row per shot  <-- source of truth
```

Each shot records: `distance_yds`, `lie` (played from), `result` (ended up),
`holed`, and `mulligan` (a discarded do-over that doesn't count). Controlled
vocabularies (lies/results) are in [`golf/schema.py`](golf/schema.py).

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
2. **Play.** Pick the date + players, then tap your way through. The lie
   auto-fills (tee on shot 1, otherwise wherever the last shot ended), so most
   shots are a single tap on the result. Distance is optional. One **Mulligan**
   button (free do-over, per round) and an **Undo** are there too. OB is just the
   `ob` result — it counts as a stroke and you replay from the same spot.
   Your round is saved in the browser, so a lock-screen won't lose it.
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

data.hole_scores()          # per (round, player, hole) strokes (mulligans excluded)
data.player_round_scores()  # per-player round totals (individual analysis)
data.team_hole_scores()     # best ball: best score per hole
data.round_scores()         # best-ball team total per round + won (beat target)
data.validate()             # list of data problems ([] = clean)

features.score_prediction() # X/y for predicting a player's strokes on a hole
features.win_probability()  # X/y for predicting the team beats the target
features.hole_difficulty()  # avg score-to-par per hole
features.shot_outcome()     # X/y for predicting shot result from lie/distance/spot
```

Modeling (train/test, metrics) lives in the notebooks so it's easy to iterate;
the `golf` package just guarantees clean, leakage-free feature tables.

## Roadmap

- [ ] Add hole map images to `maps/`
- [ ] Measure and fill in `yards` for each hole in `course.yaml`
- [ ] Log real rounds
- [ ] Baseline models in `notebooks/` once enough rounds are recorded
- [ ] (Later) optional weather/conditions capture; web UI for entry
