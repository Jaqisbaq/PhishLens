@echo off
REM Sets up PhishLens from a fresh download: creates a plain venv, installs
REM dependencies, builds the test fixtures, and downloads the three pinned
REM models. Run this once. Requires Python 3.11 on PATH and an internet
REM connection. Total download is roughly 2.3 GB, so this can take a while
REM on a slow connection.

setlocal

cd /d "%~dp0"

echo Checking for Python...
python --version
if errorlevel 1 (
    echo.
    echo Python was not found on PATH. Install Python 3.11 from
    echo https://www.python.org/downloads/ and make sure to check
    echo "Add python.exe to PATH" during install, then run this again.
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
) else (
    echo .venv already exists, reusing it.
)

echo Installing dependencies (this can take several minutes) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\pip.exe" install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency install failed. See the error above.
    exit /b 1
)

echo Installing the phishlens package itself ...
".venv\Scripts\pip.exe" install -e . --no-deps
if errorlevel 1 (
    echo.
    echo Package install failed. See the error above.
    exit /b 1
)

echo Building test fixtures (synthetic, invented data only) ...
".venv\Scripts\python.exe" fixtures\make_fixtures.py

echo.
echo Downloading the three pinned models (about 2.3 GB total) ...
echo This only needs to happen once. If it fails with a Windows symlink
echo error, the script retries automatically; if it still fails, just run
echo setup.bat again.
".venv\Scripts\python.exe" scripts\download_models.py
if errorlevel 1 (
    echo.
    echo Model download failed. Check your internet connection and run
    echo setup.bat again.
    exit /b 1
)

echo.
echo Setup complete. Run run.bat to start the app.
endlocal
