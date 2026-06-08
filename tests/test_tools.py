"""Tests for the chatbot tool layer (chatbot/tools.py) — no network.

These guard the contract between the mouth and the brain: every tool the model
can call maps to a real brain function, dispatch actually runs it, and bad args
degrade to an error dict instead of crashing the turn. The Gemini loop itself
(gemini.answer) talks to the network and is intentionally not unit-tested here;
it's built so all the logic worth testing lives in the brain + this dispatch.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chatbot import tools  # noqa: E402
from chatbot.prompts import SYSTEM_PROMPT  # noqa: E402
from golf import schema  # noqa: E402
from golf import stats  # noqa: E402


# --- make definition has one source ----------------------------------------
def test_system_prompt_defers_make_definition_to_the_tool():
    """The mouth must source the make definition from the brain, not restate it,
    so the two can't drift apart."""
    assert "list_distances" in SYSTEM_PROMPT
    assert "'good' or 'hole'" not in SYSTEM_PROMPT  # no hard-coded literal to drift


def test_make_definition_is_derived_from_make_outcomes():
    """The make definition names exactly the make outcomes — change the set, change
    the prose."""
    definition = stats.list_distances()["make_definition"]
    for outcome in stats.MAKE_OUTCOMES:
        assert outcome in definition
    assert "grounder" not in definition          # a non-make outcome isn't named


# --- the 1:1 contract ------------------------------------------------------
def test_every_declaration_has_a_dispatch_entry():
    declared = {d.name for d in tools.FUNCTION_DECLARATIONS}
    assert declared == set(tools.DISPATCH), (
        f"declarations and DISPATCH disagree: "
        f"only declared={declared - set(tools.DISPATCH)}, "
        f"only dispatch={set(tools.DISPATCH) - declared}"
    )


def test_distance_enums_match_schema():
    # Any tool exposing a `distance` param must offer exactly the schema vocab,
    # so the model can't ask for a bucket the brain rejects.
    for d in tools.FUNCTION_DECLARATIONS:
        params = d.parameters
        if params and params.properties and "distance" in params.properties:
            assert list(params.properties["distance"].enum) == list(schema.DISTANCES)


# --- dispatch behavior -----------------------------------------------------
def _install_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(schema, "DATA_DIR", tmp_path)
    monkeypatch.setattr(schema, "PLAYERS_FILE", tmp_path / "players.csv")
    monkeypatch.setattr(schema, "ROUNDS_FILE", tmp_path / "rounds.csv")
    monkeypatch.setattr(schema, "SHOTS_FILE", tmp_path / "shots.csv")
    pd.DataFrame([{"player_id": 1, "name": "Tommy", "hand": "R", "notes": ""}]).to_csv(
        tmp_path / "players.csv", index=False)
    pd.DataFrame(columns=schema.ROUND_COLUMNS).to_csv(tmp_path / "rounds.csv", index=False)
    pd.DataFrame(columns=schema.SHOT_COLUMNS).to_csv(tmp_path / "shots.csv", index=False)


def test_run_tool_matches_brain(monkeypatch, tmp_path):
    _install_empty(monkeypatch, tmp_path)
    from golf import stats
    assert tools.run_tool("make_rate", {"player": "Tommy"}) == stats.make_rate("Tommy")


def test_run_tool_unknown_player_is_error_dict(monkeypatch, tmp_path):
    _install_empty(monkeypatch, tmp_path)
    out = tools.run_tool("make_rate", {"player": "Gandalf"})
    assert "error" in out  # caught, not raised


def test_run_tool_unknown_tool_is_error_dict():
    out = tools.run_tool("nuke_the_data", {})
    assert "error" in out


def test_run_tool_bad_arg_is_error_dict(monkeypatch, tmp_path):
    _install_empty(monkeypatch, tmp_path)
    out = tools.run_tool("make_rate", {"player": "Tommy", "distance": "banana"})
    assert "error" in out
