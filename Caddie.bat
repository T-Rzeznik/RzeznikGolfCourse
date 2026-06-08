@echo off
rem ===========================================================================
rem  Rzeznik Golf — stats caddie launcher.
rem  Double-click this to start the caddie. A launch page (with a QR code for
rem  your phone) opens automatically. Close this window to stop the caddie.
rem  Prefers a .venv if you made one; otherwise uses your system Python.
rem ===========================================================================
title Rzeznik Caddie - close this window to stop
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo.
echo   Starting the Rzeznik Golf caddie...
echo   A launch page with a phone QR code will open in your browser.
echo   Close this window when you're done to stop the caddie.
echo.

"%PY%" run_server.py

echo.
echo   Caddie stopped. Press any key to close this window.
pause >nul
