"""Gemini function declarations + dispatch into the brain.

Each declaration maps 1:1 to a function in `golf.stats`. The model never touches
the data directly — it asks for one of these tools, the server runs the matching
brain function via DISPATCH, and the JSON result goes back to the model. The 1:1
contract (every declared tool has a dispatch entry and vice-versa) is asserted in
tests/test_tools.py.
"""
from __future__ import annotations

from google.genai import types

from golf import schema
from golf import stats

# The brain functions the mouth is allowed to call.
DISPATCH = {
    "make_rate": stats.make_rate,
    "compare_players": stats.compare_players,
    "player_summary": stats.player_summary,
    "leaderboard": stats.leaderboard,
    "hole_difficulty": stats.hole_difficulty,
    "list_players": stats.list_players,
    "list_distances": stats.list_distances,
}

_STRING = types.Type.STRING
_INTEGER = types.Type.INTEGER
_OBJECT = types.Type.OBJECT

_PLAYER = types.Schema(
    type=_STRING,
    description="A player's name from the roster (call list_players if unsure).",
)
_DISTANCE = types.Schema(
    type=_STRING,
    enum=list(schema.DISTANCES),
    description="Difficulty bucket of the spot: tee/long/mid/short/tap_in.",
)


def _decl(name, description, properties=None, required=None) -> types.FunctionDeclaration:
    params = None
    if properties:
        params = types.Schema(type=_OBJECT, properties=properties, required=required or [])
    return types.FunctionDeclaration(name=name, description=description, parameters=params)


FUNCTION_DECLARATIONS = [
    _decl(
        "make_rate",
        "How often a player makes a shot (outcome 'good' or 'hole'), optionally "
        "filtered to one distance bucket and/or hole. The core 'who makes this "
        "shot' stat. Returns smoothed rate, sample size, and an uncertainty flag.",
        {"player": _PLAYER, "distance": _DISTANCE,
         "hole": types.Schema(type=_INTEGER, description="Hole number (1-6).")},
        required=["player"],
    ),
    _decl(
        "compare_players",
        "Head-to-head make-rate for two players, optionally on one distance "
        "bucket. Answers 'is A more likely than B to make this shot?'. Returns "
        "both rates, who leads, and whether the edge is statistically confident.",
        {"player_a": _PLAYER, "player_b": _PLAYER, "distance": _DISTANCE},
        required=["player_a", "player_b"],
    ),
    _decl(
        "player_summary",
        "A player's overall make-rate, a per-distance breakdown, and raw totals "
        "(shots, best balls, outcome counts).",
        {"player": _PLAYER},
        required=["player"],
    ),
    _decl(
        "leaderboard",
        "Players ranked by smoothed make-rate, optionally for one distance "
        "bucket. Answers 'who's best on tap-ins?'.",
        {"distance": _DISTANCE,
         "min_n": types.Schema(type=_INTEGER,
                               description="Minimum shots to be ranked (default 1).")},
    ),
    _decl(
        "hole_difficulty",
        "Holes ranked hardest-first by average score to par. Answers 'which hole "
        "is hardest?'.",
        {"top": types.Schema(type=_INTEGER, description="Limit to the N hardest holes.")},
    ),
    _decl(
        "list_players",
        "The roster of known players. Call this when unsure whether a name is valid.",
    ),
    _decl(
        "list_distances",
        "The distance vocabulary, outcome ordering, and what a 'make' means.",
    ),
]

# A single Tool bundling all declarations, ready to hand to Gemini.
TOOLSET = types.Tool(function_declarations=FUNCTION_DECLARATIONS)


def run_tool(name: str, args: dict) -> dict:
    """Execute a tool by name with model-supplied args, via the brain.

    A bad argument (e.g. unknown player) is caught and returned as an `error`
    dict rather than raised, so the model can recover (apologize, call
    list_players) instead of the whole turn crashing.
    """
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**(args or {}))
    except (ValueError, TypeError) as e:
        return {"error": str(e)}
