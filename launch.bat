@echo off
setlocal enabledelayedexpansion

echo Checking Python installation...
where python >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

set VENV_DIR=venv
set REQUIREMENTS_FILE=requirements.txt

if not exist %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat

echo Installing/Updating pip...
python -m pip install --upgrade pip

echo Installing core packages...
pip install --default-timeout=100 numpy
pip install --default-timeout=100 pandas
echo Installing plotly...
pip install --default-timeout=100 plotly
echo Installing streamlit...
pip install --default-timeout=100 streamlit

echo All packages installed successfully.

echo Checking for running Streamlit instances...
echo Starting Production Scheduler...
python -m streamlit run app.py

pause