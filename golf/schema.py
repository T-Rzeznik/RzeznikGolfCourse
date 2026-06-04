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
# Strokes-per-hole = count of shots for (round, player, hole) EXCLUDING mulligans;
# we never store it twice. Hole/round/team scores are derived in data.py.
#
# Course rules baked into the schema:
#   * Sand wedge is the ONLY club allowed, so we don't store a club column.
#   * Out-of-bounds (result == "ob") is stroke-and-distance: it counts as a
#     stroke and the next shot replays from the same lie (no drop). Just log the
#     ob shot, then the replay as the next shot_num.
#   * One mulligan per round for the group: mark the discarded shot mulligan=True
#     and it won't count toward strokes (free do-over).
SHOT_COLUMNS = [
    "shot_id",       # globally unique, auto-assigned
    "round_id",
    "player_id",
    "hole",
    "shot_num",      # 1 = tee shot, 2 = second, ...
    "distance_yds",  # distance of the shot in yards (blank if unknown)
    "lie",           # where the shot was played FROM (see LIES)
    "result",        # where the ball ended up (see RESULTS)
    "holed",         # True on the shot that finishes the hole
    "mulligan",      # True if this shot was the discarded mulligan (doesn't count)
]

PLAYER_COLUMNS = ["player_id", "name", "hand", "notes"]
ROUND_COLUMNS = ["round_id", "date", "players", "notes"]

# --- Controlled vocabularies ----------------------------------------------
# Keep these tight so categorical ML features stay clean. Add as needed.
# Only one club is legal on the course, so there's no club vocabulary.
CLUB = "sand wedge"

LIES = ["tee", "fairway", "rough", "sand", "green", "woods", "hazard"]

RESULTS = ["fairway", "rough", "green", "sand", "woods", "hazard", "ob", "holed"]
