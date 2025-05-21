@echo off
echo Installing required packages...
pip install -r requirements.txt

echo Checking for running Streamlit instances...
taskkill /F /IM "streamlit.exe" /T 2>nul
timeout /t 2 /nobreak >nul

echo Starting Production Scheduler...
streamlit run app.py --server.port 8502

if errorlevel 1 (
    echo Retrying with different port...
    streamlit run app.py --server.port 8503
)

pause