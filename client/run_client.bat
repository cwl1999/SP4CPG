@echo off
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [client] Creating virtual environment...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m sp4cpg_client.app
pause
