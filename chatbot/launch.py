"""The /launch landing page: shows the phone QR code and the LAN URL.

`Caddie.bat` opens this on the laptop. Scan the QR with a phone (same wifi) to
jump straight into the caddie chat at `/`. The QR is generated server-side from
the laptop's real LAN IP each launch, so it survives the IP changing and needs no
internet of its own.
"""
from __future__ import annotations

import io
import socket

import qrcode
from qrcode.image.svg import SvgPathImage


def lan_ip() -> str:
    """Best guess at this machine's LAN IP (the one a phone on the wifi would use).

    Opens a throwaway UDP socket toward a public IP; the OS picks the outbound
    interface and we read its local address. No packets are actually sent.
    Falls back to localhost if there's no network.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _qr_svg(data: str) -> str:
    """An inline SVG QR code for `data` (no PIL dependency, no external request)."""
    buf = io.BytesIO()
    qrcode.make(data, image_factory=SvgPathImage, box_size=11, border=2).save(buf)
    svg = buf.getvalue().decode("utf-8")
    # Drop the XML prolog so it embeds cleanly inside HTML.
    return svg[svg.index("<svg"):]


def launch_page(port: int) -> str:
    """The full HTML for the laptop's launch page."""
    ip = lan_ip()
    url = f"http://{ip}:{port}/"
    qr = _qr_svg(url)
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Launch the Rzeznik Caddie</title>
<style>
  :root {{ --green:#1b5e20; --green2:#2e7d32; --green3:#43a047; --bg:#0c120c; --card:#161d16;
          --line:#33402f; --txt:#e8f0e2; --muted:#9bb08c; --accent:#7bd389; --gold:#e8c98c; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,system-ui,sans-serif; background:var(--bg); color:var(--txt);
         min-height:100vh; display:flex; align-items:center; justify-content:center; padding:24px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:22px; padding:30px 26px;
          max-width:420px; width:100%; text-align:center; box-shadow:0 12px 40px rgba(0,0,0,.5); }}
  .badge {{ width:58px; height:58px; border-radius:50%; margin:0 auto 12px;
          background:radial-gradient(circle at 32% 27%,var(--green3),var(--green) 80%);
          box-shadow:inset 0 1px 0 rgba(255,255,255,.28),0 3px 10px rgba(0,0,0,.4);
          display:flex; align-items:center; justify-content:center; font-size:28px; }}
  h1 {{ font-size:21px; margin:0 0 4px; letter-spacing:.3px; }}
  .sub {{ color:var(--muted); font-size:14px; margin-bottom:22px; }}
  .qrlabel {{ font-size:14px; color:var(--gold); font-weight:700; margin-bottom:10px; }}
  .qr {{ background:#fff; border-radius:16px; padding:14px; display:inline-block; line-height:0;
        box-shadow:0 4px 16px rgba(0,0,0,.35); }}
  .qr svg {{ width:212px; height:212px; display:block; }}
  .url {{ margin:16px 0 4px; font-size:13px; color:var(--muted); }}
  .url b {{ color:var(--txt); font-family:ui-monospace,Menlo,Consolas,monospace; font-size:14px; }}
  .open {{ display:block; margin-top:22px; padding:15px; border-radius:14px; border:1px solid var(--accent);
          background:var(--green2); color:#f3fbef; font-size:16px; font-weight:700; text-decoration:none; }}
  .open:active {{ transform:scale(.985); }}
  .hint {{ margin-top:18px; font-size:12.5px; color:var(--muted); line-height:1.5;
          border-top:1px solid var(--line); padding-top:14px; }}
  .hint b {{ color:var(--gold); }}
</style></head>
<body>
  <div class="card">
    <div class="badge">⛳</div>
    <h1>Rzeznik Caddie is running</h1>
    <div class="sub">Ask your backyard golf stats anything.</div>

    <div class="qrlabel">📱 Scan to use it on your phone</div>
    <div class="qr">{qr}</div>
    <div class="url">or open <b>{url}</b> on the same wifi</div>

    <a class="open" href="/">Open the caddie here →</a>

    <div class="hint">
      First time on your phone and it won't connect? Click <b>Allow access</b> on the
      Windows Firewall popup, and make sure the phone is on the <b>same wifi</b> as this
      laptop. Closing the launcher window stops the caddie.
    </div>
  </div>
</body></html>"""
