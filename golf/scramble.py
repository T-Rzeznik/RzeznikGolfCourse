"""The scramble keep-rule, in one place.

Given a stroke's outcomes (in shot order), decide which balls are eligible to be
kept for the next stroke. This is the *suggestion* side of the rule — the writers
(`scripts/log_round.py`, `scripts/generate_sample_data.py`) use it to pick the
best ball. `data.validate` independently *checks* legality and is deliberately
looser (a strategically worse ball may be kept), so it does not call this.

See CONTEXT.md for the scramble invariant this encodes.
"""
from . import schema

# A ball that's out of bounds or skipped can never be the kept ball.
_NO_KEEP = {"ob", "skip"}


def suggest_best_ball(outcomes: list[str]) -> set[int]:
    """Indices into `outcomes` (shot order) eligible to be the kept ball.

    - Any holed ball ends the hole: every holer is kept.
    - If every ball is OB or skipped, the stroke makes no advance: nothing kept.
    - Otherwise the balls tied at the best outcome by `schema.OUTCOMES` rank are
      eligible; the caller keeps exactly one of them.
    """
    holers = {i for i, o in enumerate(outcomes) if o == "hole"}
    if holers:
        return holers
    keepable = {i for i, o in enumerate(outcomes) if o not in _NO_KEEP}
    if not keepable:
        return set()
    top = max(schema.OUTCOMES.index(outcomes[i]) for i in keepable)
    return {i for i in keepable if schema.OUTCOMES.index(outcomes[i]) == top}
