@echo off
REM Starts the PhishLens server on 127.0.0.1 and prints the address to
REM open in a browser. Run setup.bat first if you have not already.

setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo .venv was not found. Run setup.bat first.
    exit /b 1
)

echo Starting PhishLens ...
echo Once you see "Application startup complete", open this address in
echo your browser:
echo.
echo     http://127.0.0.1:8000/
echo.
echo Press Ctrl+C in this window to stop the server.
echo.

".venv\Scripts\python.exe" scripts\run_server.py --preload
endlocal
