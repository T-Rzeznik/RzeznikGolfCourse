# RzeznikGolfCourse

A stat tracker for a 6-hole backyard mini-golf course, played as a 1–3 player
best-ball **scramble** with a sand wedge. The per-shot data is the foundation for
the end goal: an LLM "stats caddie" that answers questions like *"who's more likely
to make this shot?"* See **`CONTEXT.md`** for the domain model and the scramble
invariant — read it before touching scoring, validation, or features.

## Layout

- `golf/` — the library. `schema.py` (single source of truth for columns + vocab),
  `data.py` (load / validate / derive scorecards / append), `features.py` (ML
  features), `stats.py` (the caddie "brain").
- `chatbot/` — the caddie "mouth" (Gemini); FastAPI server wires brain + mouth.
- `scripts/` — `import_log.py` (ingest a logged round), `log_round.py`,
  `generate_sample_data.py`.
- `webapp/` — the phone logger (`index.html`) that emits a round as JSON.
- `data/` — `course.yaml`, `players.csv`, `rounds.csv`, `shots.csv` (source of truth).
- `tests/` — `python -m pytest -q`.

## Common commands

- Run the tests: `python -m pytest -q`
- Import a logged round: `python scripts/import_log.py round.json`
- Regenerate sample data: `python scripts/generate_sample_data.py --rounds 8`

## Agent skills

### Issue tracker

Issues and PRDs live as GitHub issues on `T-Rzeznik/RzeznikGolfCourse` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles mapped 1:1 to GitHub labels (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
