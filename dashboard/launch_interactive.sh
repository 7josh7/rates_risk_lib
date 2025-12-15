#!/bin/bash
# Quick launch script for the Interactive Analytics Dashboard

echo "========================================"
echo "Rates Risk Library - Interactive Dashboard"
echo "========================================"
echo ""

# Check if we're in the right directory
if [ ! -f "interactive_dashboard.py" ]; then
    echo "Error: Please run this script from the dashboard directory"
    echo "Usage: cd dashboard && ./launch_interactive.sh"
    exit 1
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r requirements_interactive.txt
fi

# Check if output files exist, if not run demo
OUTPUT_DIR="../output"
if [ ! -f "$OUTPUT_DIR/Sample_Trading_Book_20240115_Portfolio_Summary.csv" ]; then
    echo "Output files not found. Running demo to generate sample data..."
    cd ..
    python scripts/run_demo.py --output-dir ./output
    cd dashboard
fi

echo ""
echo "Launching Interactive Analytics Dashboard..."
echo "The dashboard will open in your browser at http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run interactive_dashboard.py
