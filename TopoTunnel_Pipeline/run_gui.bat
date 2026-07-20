@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 goto run_with_py

python pipeline_gui.py
goto finished

:run_with_py
py -3 pipeline_gui.py

:finished
if errorlevel 1 pause
endlocal
