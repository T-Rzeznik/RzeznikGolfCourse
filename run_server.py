"""Launch the stats caddie and open its launch page.

    python run_server.py        (or just double-click Caddie.bat)

Starts the server on 0.0.0.0:<port> and pops open the laptop's /launch page (a QR
code + LAN URL so you can open the caddie on your phone). Binding 0.0.0.0 is what
lets the phone on the same wifi reach it. Closing the window stops the caddie.
"""
import os
import threading
import webbrowser

import uvicorn

PORT = int(os.environ.get("CHATBOT_PORT", "8000"))


def _open_launch_page():
    """Open the laptop browser at /launch once the server is up (best-effort)."""
    webbrowser.open(f"http://localhost:{PORT}/launch")


if __name__ == "__main__":
    print(f"Rzeznik Golf caddie -> http://localhost:{PORT}/launch")
    print("Close this window to stop the caddie.")
    # Give uvicorn ~1.5s to bind, then open the launch page.
    threading.Timer(1.5, _open_launch_page).start()
    uvicorn.run("chatbot.server:app", host="0.0.0.0", port=PORT, reload=False)
