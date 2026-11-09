@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please install dependencies using README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" run_lstm.py %*
if errorlevel 1 pause
