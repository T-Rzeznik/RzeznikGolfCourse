"""The "mouth": a Gemini client that narrates the brain via function-calling.

`answer()` runs the loop: send the conversation to Gemini with the toolset, run
any function calls it requests against the brain (chatbot.tools.run_tool), feed
the results back, and repeat until the model produces a text answer. The returned
`tool_calls` list lets the caller verify the brain was actually consulted (so we
can trust the numbers weren't invented).

The Gemini API key is read once from the environment (loaded from a gitignored
.env). It lives only here, server-side — the browser never sees it.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from . import tools
from .prompts import SYSTEM_PROMPT

# override=True so a freshly-edited .env wins over any stale GEMINI_API_KEY left
# in the shell environment (the .env file is the source of truth for config).
load_dotenv(override=True)

# Cheap, reliable Flash model. (gemini-2.0-flash is retired; gemini-2.5-flash-lite
# is cheaper but frequently returns an empty completion after a tool call, so it's
# unusable for this tool-calling flow.) Override via env.
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

_EMPTY_FALLBACK = ("I pulled the numbers but couldn't phrase an answer — try asking "
                   "again, maybe a bit more specifically.")
# Generous enough that a "stats for every player" question (one tool call to list
# the roster, one summary per player, then a turn to actually write the answer)
# finishes inside the loop instead of getting cut off mid-gather. Still a hard cap
# so a misbehaving model can't loop forever.
MAX_TOOL_ROUNDS = 10

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily build the Gemini client; fail with a clear message if no key."""
    global _client
    if _client is None:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and add your "
                "key (get one at https://aistudio.google.com/apikey)."
            )
        _client = genai.Client(api_key=key)
    return _client


def _history_to_contents(history: list | None) -> list[types.Content]:
    """Convert prior [{role, text}] turns into Gemini Content objects.

    `role` is 'user' or 'bot'; Gemini expects 'user' / 'model'. Anything else is
    treated as the user speaking.
    """
    contents: list[types.Content] = []
    for turn in history or []:
        text = str(turn.get("text", "")).strip()
        if not text:
            continue
        role = "model" if turn.get("role") in ("bot", "model", "assistant") else "user"
        contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    return contents


def answer(message: str, history: list | None = None) -> dict:
    """Answer one user message, consulting the brain via tools as needed.

    Returns {"reply": str, "tool_calls": [{"name", "args", "result"}, ...]}.
    """
    client = _get_client()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[tools.TOOLSET],
        temperature=0.3,
    )

    contents = _history_to_contents(history)
    contents.append(types.Content(role="user", parts=[types.Part(text=str(message))]))

    tool_calls: list[dict] = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.models.generate_content(
            model=MODEL, contents=contents, config=config
        )

        fcalls = list(response.function_calls or [])
        if not fcalls:
            reply = (response.text or "").strip()
            # Some models occasionally return an empty completion after a tool call;
            # don't surface a blank bubble to the phone.
            return {"reply": reply or _EMPTY_FALLBACK, "tool_calls": tool_calls}

        # Record the model's tool-call turn, then run each call and answer it.
        contents.append(response.candidates[0].content)
        result_parts = []
        for call in fcalls:
            args = dict(call.args or {})
            result = tools.run_tool(call.name, args)
            tool_calls.append({"name": call.name, "args": args, "result": result})
            result_parts.append(
                types.Part.from_function_response(name=call.name, response=result)
            )
        contents.append(types.Content(role="tool", parts=result_parts))

    # Hit the round cap — force a text reply from what we've already gathered.
    # Drop the toolset AND nudge the model to answer: without the nudge Flash often
    # returns an empty completion here (it still "wants" a tool call it's no longer
    # offered), which is exactly what surfaced the "couldn't phrase an answer" reply.
    contents.append(types.Content(role="user", parts=[types.Part(text=(
        "Answer now using the tool results above — summarize what you have. "
        "Do not request any more tools."))]))
    final = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, temperature=0.3
        ),
    )
    return {"reply": (final.text or "").strip() or _EMPTY_FALLBACK, "tool_calls": tool_calls}
