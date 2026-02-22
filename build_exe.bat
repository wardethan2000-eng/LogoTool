@echo off
REM ─────────────────────────────────────────────────────────
REM  Build a standalone logo2svg GUI executable with PyInstaller
REM ─────────────────────────────────────────────────────────
echo === logo2svg GUI builder ===

set PYTHON_EXE=python
if exist ".venv\Scripts\python.exe" set PYTHON_EXE=.venv\Scripts\python.exe

REM 1. Make sure PyInstaller is installed
%PYTHON_EXE% -m pip install pyinstaller >nul 2>&1

REM 2. Remove stale build artifacts
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM 3. Run PyInstaller from the spec file
echo Building executable ...
%PYTHON_EXE% -m PyInstaller logo2svg_gui.spec --noconfirm --clean

echo.
echo Done!  The executable is in:
echo   dist\logo2svg\logo2svg.exe
echo.
echo You can distribute the entire dist\logo2svg\ folder,
echo or use the --onefile flag in the spec for a single EXE
echo (slower startup, but one file to ship).
pause
