@echo off
setlocal
cd /d "%~dp0"

set "PYTHONPATH=src"
set "HOST=127.0.0.1"
set "PORT=8080"
set "SUITE_DIR=artifacts/engineering_suite_2017"

REM Portable Python interpreter lookup:
REM 1. Explicit WINTELLISYT_PYTHON override
REM 2. Repository-local virtual environment (.venv)
REM 3. Active Conda environment (CONDA_PREFIX)
REM 4. Default python executable on PATH
set "PY_EXE="
if defined WINTELLISYT_PYTHON (
    if exist "%WINTELLISYT_PYTHON%" set "PY_EXE=%WINTELLISYT_PYTHON%"
)
if not defined PY_EXE (
    if exist "%~dp0.venv\Scripts\python.exe" set "PY_EXE=%~dp0.venv\Scripts\python.exe"
)
if not defined PY_EXE (
    if defined CONDA_PREFIX (
        if exist "%CONDA_PREFIX%\python.exe" set "PY_EXE=%CONDA_PREFIX%\python.exe"
    )
)
if not defined PY_EXE set "PY_EXE=python"

echo ========================================================================
echo   WINTELLISYT NiceGUI Dashboard (Local Loopback Only)
echo   Python: %PY_EXE%
echo   URL   : http://%HOST%:%PORT%/
echo   Bind  : %HOST% (No external ports opened)
echo   Press Ctrl+C in this window to stop the dashboard server.
echo ========================================================================

"%PY_EXE%" -m wintellisyt.cli ui --suite-dir "%SUITE_DIR%" --host %HOST% --port %PORT% --backend auto --show
if errorlevel 1 (
    echo.
    echo [ERROR] WINTELLISYT dashboard exited with an error.
    echo         Ensure dependencies are installed: pip install -r requirements.txt
    pause
)
endlocal
