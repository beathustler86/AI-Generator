@echo off
REM === Activate Python virtual environment ===
SETLOCAL
cd /d %~dp0

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERROR] No venv found at .venv\
    pause
    EXIT /B 1
)

call ".venv\Scripts\activate.bat"
echo.
echo Venv activated. You are now inside:
where python
echo.

cmd /k
