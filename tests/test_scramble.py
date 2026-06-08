"""Tests for the shared scramble keep-rule (golf/scramble.py).

Run from the repo root:   python -m pytest -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from golf import scramble  # noqa: E402


def test_holer_is_the_kept_ball():
    # outcomes in shot order: the holed ball ends the hole and is kept.
    assert scramble.suggest_best_ball(["good", "hole", "overshoot"]) == {1}


def test_every_holer_is_kept():
    assert scramble.suggest_best_ball(["hole", "good", "hole"]) == {0, 2}


def test_all_ob_or_skip_keeps_nothing():
    assert scramble.suggest_best_ball(["ob", "ob"]) == set()
    assert scramble.suggest_best_ball(["ob", "skip"]) == set()


def test_best_outcome_by_rank_wins():
    # good outranks grounder
    assert scramble.suggest_best_ball(["grounder", "good"]) == {1}


def test_ties_at_the_top_are_all_eligible():
    assert scramble.suggest_best_ball(["grounder", "good", "good"]) == {1, 2}


def test_ob_and_skip_are_excluded_even_when_others_are_worse():
    # the grounder is kept; the OB never can be, even though it came first.
    assert scramble.suggest_best_ball(["ob", "grounder"]) == {1}
