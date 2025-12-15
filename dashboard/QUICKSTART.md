# Quick Start Guide - Interactive Dashboard

Get up and running with the Interactive Analytics Dashboard in 5 minutes!

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Installation

### Step 1: Install the Library
```bash
# From the repository root
pip install -e .
```

### Step 2: Install Dashboard Dependencies
```bash
# From the repository root
cd dashboard
pip install -r requirements_interactive.txt
```

## Running the Dashboard

### Option 1: Quick Launch Script (Recommended)

**On Linux/Mac:**
```bash
cd dashboard
./launch_interactive.sh
```

**On Windows:**
```bash
cd dashboard
launch_interactive.bat
```

The script will:
1. Check for required dependencies
2. Generate sample data if needed
3. Launch the dashboard at http://localhost:8501

### Option 2: Manual Launch

```bash
# Step 1: Generate sample data (first time only)
cd /path/to/rates_risk_lib
python scripts/run_demo.py --output-dir ./output

# Step 2: Launch dashboard
cd dashboard
streamlit run interactive_dashboard.py
```

The dashboard will automatically open in your default browser at http://localhost:8501

## Dashboard Overview

The Interactive Analytics Dashboard provides 8 main tabs:

### 📈 Curves Tab
Visualize and analyze yield curves:
- OIS curve bootstrap results
- Treasury NSS fitted parameters
- Curve comparison (OIS vs Treasury)
- Discount factors
- Forward rates

**What to try:**
- Compare OIS and Treasury curves
- Examine the spread between curves
- View forward rate curves

### 💰 Pricing Tab
Interactive pricing calculators:
- **Bonds**: Price with custom coupon, maturity, frequency
- **Swaps**: Calculate PV, DV01, and par rates
- **Futures**: Theoretical pricing

**What to try:**
- Price a 5-year bond with 4% coupon
- Calculate par rate for a 10-year swap
- Compare prices with different maturities

### 📊 Risk Metrics Tab
Portfolio risk analysis:
- Portfolio summary (PV, DV01, convexity)
- Key Rate DV01 ladder
- Position-level risk breakdown

**What to try:**
- View the key rate ladder to see risk distribution
- Analyze which tenors have the most risk

### 🎯 VaR Analysis Tab
Value at Risk calculations:
- **Historical Simulation**: Based on historical rate moves
- **Monte Carlo**: Simulated scenarios
- **Stressed VaR**: Crisis period analysis

**What to try:**
- Compare VaR at 95% vs 99% confidence
- View P&L distribution histogram
- Adjust lookback period for historical VaR

### 📉 Scenarios Tab
Stress testing and scenario analysis:
- Standard scenarios (parallel shifts, steepeners, flatteners)
- Waterfall chart visualization
- Custom scenario builder

**What to try:**
- See impact of +100bp parallel shift
- Build a custom scenario with twist
- Compare bear flattener vs bull steepener

### 💵 P&L Attribution Tab
Decompose portfolio P&L:
- Carry
- Rolldown
- Curve moves (parallel & non-parallel)
- Convexity
- Residual

**What to try:**
- Understand which components drive P&L
- See predicted vs realized P&L
- Analyze residual (unexplained) portion

### 💧 Liquidity Risk Tab
Liquidity-adjusted VaR:
- Bid/ask spread impacts
- Holding period scaling
- Position size effects
- Instrument-specific liquidity metrics

**What to try:**
- Calculate LVaR with different holding periods
- Apply stress multiplier
- Compare base VaR vs LVaR

### 📋 Data Explorer Tab
Browse and export data:
- Market quotes (OIS, Treasury)
- Portfolio positions
- Historical rates
- Curve nodes

**What to try:**
- Export positions as CSV
- View curve node details
- Browse historical rate data

## Tips & Tricks

1. **Customize Date**: Use the sidebar to change the valuation date
2. **Interactive Charts**: All Plotly charts support zoom, pan, and hover for details
3. **Export Data**: Download position data as CSV from the Data Explorer tab
4. **Real-time Updates**: Adjust sliders and inputs to see immediate recalculations
5. **Full Screen**: Click the expand icon on any chart for full-screen view

## Keyboard Shortcuts

When the dashboard is active:
- `R` - Rerun the dashboard
- `C` - Clear cache and rerun
- `Ctrl+R` - Refresh browser
- `Ctrl+C` in terminal - Stop the server

## Troubleshooting

### Dashboard won't start
- Ensure you've installed dependencies: `pip install -r requirements_interactive.txt`
- Check Python version: `python --version` (should be 3.9+)

### No data showing
- Run the demo script first: `python scripts/run_demo.py --output-dir ./output`
- Check that CSV files exist in the `output/` directory

### Port already in use
- Stop other Streamlit instances
- Or specify a different port: `streamlit run interactive_dashboard.py --server.port 8502`

### Performance issues
- The dashboard caches curves and data automatically
- Use "Clear cache" from the hamburger menu if needed
- Reduce the number of Monte Carlo scenarios for faster calculation

## Next Steps

After exploring the dashboard:

1. **Modify Parameters**: Try different instruments, maturities, and risk scenarios
2. **Add Your Data**: Replace sample data with your own market quotes and positions
3. **Customize**: Edit `interactive_dashboard.py` to add custom visualizations
4. **Integrate**: Use the dashboard alongside the real-time monitor (`app.py`)

## Additional Resources

- Main README: `../README.md`
- Dashboard Documentation: `README.md` (in dashboard directory)
- API Reference: See main README for detailed API documentation
- Demo Scripts: `../scripts/run_demo.py` and `../scripts/run_sabr_demo.py`

## Support

For issues or questions:
1. Check the main README.md
2. Review test files in `../tests/`
3. Examine the demo scripts

## License

MIT License - See main repository LICENSE file

---

**Enjoy exploring your rates risk library!** 📊📈
