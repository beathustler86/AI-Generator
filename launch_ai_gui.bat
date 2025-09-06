@echo off
REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Launch the app
python src\launch_gui.py

REM Pause so you can see any errors
pause