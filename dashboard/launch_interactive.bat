@echo off
REM Quick launch script for the Interactive Analytics Dashboard (Windows)

echo ========================================
echo Rates Risk Library - Interactive Dashboard
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "interactive_dashboard.py" (
    echo Error: Please run this script from the dashboard directory
    echo Usage: cd dashboard ^&^& launch_interactive.bat
    exit /b 1
)

REM Check if output files exist, if not run demo
set OUTPUT_DIR=..\output
if not exist "%OUTPUT_DIR%\Sample_Trading_Book_20240115_Portfolio_Summary.csv" (
    echo Output files not found. Running demo to generate sample data...
    cd ..
    python scripts\run_demo.py --output-dir .\output
    cd dashboard
)

echo.
echo Launching Interactive Analytics Dashboard...
echo The dashboard will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

streamlit run interactive_dashboard.py
