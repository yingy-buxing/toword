@echo off
setlocal
pushd "%~dp0"

if "%~1"=="" (
  echo Usage: run.bat input_audio_or_video [extra transcribe.py args]
  exit /b 2
)

if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" %*
) else (
  python "%~dp0transcribe.py" %*
)
