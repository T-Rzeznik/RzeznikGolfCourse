"""Local FastAPI server for the stats caddie (a standalone app — the round logger
in webapp/index.html runs offline and does NOT use this server).

    python run_server.py        # binds 0.0.0.0:8000

Routes:
    GET  /          -> the caddie chat app (webapp/caddie.html)
    GET  /launch    -> laptop launch page: QR + LAN URL for the phone
    POST /api/chat  -> the caddie answers (brain in golf/stats.py)
    GET  /healthz   -> reachability check

The caddie is served from the SAME origin as /api/chat, so the phone's fetch is
same-origin and needs no CORS.
"""
from __future__ import annotations

import logging
import os
import traceback
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import launch

ROOT = Path(__file__).resolve().parent.parent
CADDIE_FILE = ROOT / "webapp" / "caddie.html"
PORT = int(os.environ.get("CHATBOT_PORT", "8000"))

log = logging.getLogger("caddie")

app = FastAPI(title="Rzeznik Golf Course — stats caddie")


@app.get("/", response_class=HTMLResponse)
def home():
    """The caddie chat app itself (phone QR points here)."""
    return FileResponse(CADDIE_FILE)


@app.get("/launch", response_class=HTMLResponse)
def launch_page():
    """Laptop landing page: QR + LAN URL to open the caddie on a phone."""
    return HTMLResponse(launch.launch_page(PORT))


class ChatTurn(BaseModel):
    role: str = "user"
    text: str = ""


class ChatRequest(BaseModel):
    message: str
    history: list[ChatTurn] | None = None


@app.get("/healthz")
def healthz():
    """Quick reachability check — open this on the phone to confirm it connects."""
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Answer one message. Imports gemini lazily so the server (and /healthz)
    start even before a key is configured; errors come back as friendly JSON."""
    if not req.message.strip():
        return {"reply": "Ask me something about the golf stats!", "tool_calls": []}
    try:
        from . import gemini
        history = [t.model_dump() for t in (req.history or [])]
        result = gemini.answer(req.message, history=history)
        return result
    except RuntimeError as e:  # missing key / config
        return JSONResponse(status_code=503, content={"reply": f"⚠️ {e}", "tool_calls": []})
    except Exception as e:  # noqa: BLE001 — never leak a stack trace to the phone
        # Log the real error server-side so it's debuggable, then hand the phone a
        # friendly, specific message (quota is the most common real-world failure).
        log.error("chat failed: %s", traceback.format_exc())
        detail = str(e)
        if "429" in detail or "RESOURCE_EXHAUSTED" in detail or "quota" in detail.lower():
            reply = ("⚠️ The Gemini API key is out of quota (HTTP 429). The free tier may be "
                     "disabled for this key's project — enable billing or use a key with quota. "
                     "Your stats are fine; only the chatbot's narration is blocked.")
            status = 429
        elif "API key" in detail or "PERMISSION_DENIED" in detail or "401" in detail or "403" in detail:
            reply = "⚠️ The Gemini API key was rejected. Check GEMINI_API_KEY in your .env."
            status = 502
        else:
            reply = (f"Something went wrong answering that ({type(e).__name__}). "
                     "Check the server console for details.")
            status = 500
        return JSONResponse(status_code=status, content={"reply": reply, "tool_calls": []})
