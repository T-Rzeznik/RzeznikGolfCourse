"""Single source of truth for the data schema: file locations, columns, and
the controlled vocabularies used across data entry, validation, and ML features.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MAPS_DIR = ROOT / "maps"

COURSE_FILE = DATA_DIR / "course.yaml"
PLAYERS_FILE = DATA_DIR / "players.csv"
ROUNDS_FILE = DATA_DIR / "rounds.csv"
SHOTS_FILE = DATA_DIR / "shots.csv"

# --- Columns (the per-shot fact table is the source of truth) --------------
# We play a SCRAMBLE: every active player hits from the SAME spot each stroke,
# then the team keeps the best ball and everyone plays the next stroke from
# there. shots.csv has one row per player per stroke. Team scores are *derived*
# in data.py (never stored twice).
#
# THE SCRAMBLE INVARIANT (every scoring/validation rule follows from this):
#   For a given (round, hole), strokes are numbered 1..max with no gaps. On each
#   stroke every active player has exactly one COUNTING shot. The hole ends at
#   the first stroke containing a `hole` outcome (so `hole` appears only at max).
#   An all-OB stroke has no best ball and re-hits the same spot at stroke+1 (it
#   still costs a stroke). Exactly one ball per non-final stroke is the
#   `best_ball` (none if all-OB); on the final stroke the holing ball(s) are
#   `best_ball`. `best_ball` is never an OB ball. Team score for the hole =
#   max(stroke_num).
#
# House rules baked in:
#   * Sand wedge is the ONLY club allowed, so we don't store a club column.
#   * Out-of-bounds: a ball that goes OB can't be the best ball; if EVERYONE is
#     OB the team re-hits the same spot (that stroke still counts).
#   * One mulligan per round for the group: mark the discarded shot mulligan=True
#     (it shares its stroke_num with the do-over) and it won't count.
SHOT_COLUMNS = [
    "shot_id",     # globally unique, auto-assigned
    "round_id",
    "player_id",
    "hole",
    "stroke_num",  # team stroke index (1..max); shared by all players in a stroke
    "shot_order",  # 1..n within the stroke; 1 = who hit first (starter / best-ball hitter)
    "outcome",     # one of OUTCOMES
    "best_ball",   # True if this ball was kept for the next stroke (auto on a hole)
    "mulligan",    # True if this shot was the discarded mulligan (doesn't count)
]

PLAYER_COLUMNS = ["player_id", "name", "hand", "notes"]
ROUND_COLUMNS = ["round_id", "date", "players", "notes"]

# --- Controlled vocabularies ----------------------------------------------
# Only one club is legal on the course, so there's no club vocabulary.
CLUB = "sand wedge"

# The shot outcomes, roughly worst -> best, with `ob` as the bust. The order
# here doubles as a quality ranking for auto-suggesting the best ball.
# `skip` is special: a player who sat out a stroke (not a real attempt). Like
# `ob` it can never be the best ball, and a stroke where *every* ball is OB or
# skipped advances with no best ball (re-hit the same spot). It's excluded from
# the shot-outcome model (see features.shot_outcome).
OUTCOMES = ["skip", "ob", "overshoot", "grounder", "short_pop", "good", "hole"]
