@echo off
REM ─────────────────────────────────────────────────────────
REM  Build a standalone logo2svg GUI executable with PyInstaller
REM ─────────────────────────────────────────────────────────
echo === logo2svg GUI builder ===

REM 1. Make sure PyInstaller is installed
pip install pyinstaller >nul 2>&1

REM 2. Run PyInstaller from the spec file
echo Building executable ...
pyinstaller logo2svg_gui.spec --noconfirm

echo.
echo Done!  The executable is in:
echo   dist\logo2svg\logo2svg.exe
echo.
echo You can distribute the entire dist\logo2svg\ folder,
echo or use the --onefile flag in the spec for a single EXE
echo (slower startup, but one file to ship).
pause
