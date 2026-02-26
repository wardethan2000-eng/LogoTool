@echo off
REM ─────────────────────────────────────────────────────────
REM  Build a standalone QuickLayer GUI executable with PyInstaller
REM ─────────────────────────────────────────────────────────

REM Always work from the directory this batch file lives in,
REM even when launched via double-click or a shortcut.
cd /d "%~dp0"

echo === QuickLayer GUI builder ===

REM Locate Python — prefer the project venv, fall back to py launcher, then PATH
set PYTHON_EXE=
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    goto :found_python
)

REM Try the "py" launcher (standard on Windows Python installs)
py --version >nul 2>&1
if %ERRORLEVEL%==0 (
    set "PYTHON_EXE=py"
    goto :found_python
)

REM Try "python" on PATH — verify it actually works (skip Windows Store alias)
python --version >nul 2>&1
if %ERRORLEVEL%==0 (
    set "PYTHON_EXE=python"
    goto :found_python
)

echo ERROR: Python not found.
echo   - Create a venv:  python -m venv .venv
echo   - Or add Python to your PATH.
pause
exit /b 1

:found_python

echo Using Python: %PYTHON_EXE%

REM 1. Make sure PyInstaller is installed
"%PYTHON_EXE%" -m pip install pyinstaller --quiet
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyInstaller.
    pause
    exit /b 1
)

REM 2. Remove stale build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM 3. Run PyInstaller from the spec file
echo Building executable ...
"%PYTHON_EXE%" -m PyInstaller logo2svg_gui.spec --noconfirm --clean
if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: PyInstaller build failed. See output above.
    pause
    exit /b 1
)

echo.
echo Done!  The executable is in:
echo   dist\QuickLayer\QuickLayer.exe
echo.
echo You can distribute the entire dist\QuickLayer\ folder,
echo or use the --onefile flag in the spec for a single EXE
echo (slower startup, but one file to ship).
pause
