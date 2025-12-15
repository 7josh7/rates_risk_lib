# Interactive Dashboard

This directory contains interactive dashboards for visualizing the Rates Risk Library functionality.

## Available Dashboards

### 1. Real-Time Risk Monitor (`app.py`)
A Shiny-based real-time monitoring dashboard showing:
- Portfolio summary (PV, DV01, Convexity)
- Key Rate DV01 exposure
- VaR/ES metrics
- Position-level details
- Scenario analysis P&L
- Risk limit monitoring

**To run:**
```bash
cd dashboard
shiny run app.py --reload
```

### 2. Interactive Analytics Dashboard (`interactive_dashboard.py`)
A comprehensive Streamlit-based dashboard covering **all library functionality**:

#### Features:

**📈 Curves Tab**
- OIS curve bootstrap visualization
- Treasury NSS curve with fitted parameters
- Yield curve comparison (OIS vs Treasury)
- Spread analysis
- Discount factors
- Forward rate curves

**💰 Pricing Tab**
- Interactive bond pricing calculator
- Swap pricing with custom parameters
- Futures pricing
- Real-time DV01 calculation

**📊 Risk Metrics Tab**
- Portfolio risk summary
- Key Rate DV01 ladder visualization
- Convexity analysis
- Position-level risk breakdown

**🎯 VaR Analysis Tab**
- Historical Simulation VaR with distribution plots
- Monte Carlo VaR with configurable scenarios
- Stressed VaR for different crisis periods
- Expected Shortfall (ES) calculations
- Interactive confidence level selection

**📉 Scenarios Tab**
- Standard scenario impacts (parallel shifts, steepeners, flatteners)
- Waterfall chart visualization
- Custom scenario builder
- Detailed P&L breakdown by scenario

**💵 P&L Attribution Tab**
- Daily P&L decomposition:
  - Carry
  - Rolldown
  - Curve Move (Parallel)
  - Curve Move (Non-Parallel)
  - Convexity
  - Residual
- Visual breakdown charts
- Predicted vs realized P&L comparison

**💧 Liquidity Risk Tab**
- Liquidity-adjusted VaR (LVaR) calculations
- Bid/ask spread impacts
- Holding period scaling
- Position size impacts
- Stress multiplier analysis
- Instrument-specific liquidity metrics

**📋 Data Explorer Tab**
- Market quotes browser
- Portfolio positions viewer
- Historical rates data
- Curve nodes inspection
- CSV export functionality

**To run:**
```bash
cd dashboard
streamlit run interactive_dashboard.py
```

## Installation

Install dependencies for both dashboards:

```bash
# From the dashboard directory
pip install -r requirements.txt  # For Shiny dashboard
pip install -r requirements_interactive.txt  # For Streamlit dashboard

# Or install all at once
pip install shiny shinywidgets plotly streamlit pandas numpy
```

## Data Requirements

Both dashboards load data from:
- `../data/sample_quotes/` - Market data (OIS quotes, Treasury yields, historical rates)
- `../data/sample_book/` - Portfolio positions
- `../output/` - Generated risk reports (for the real-time dashboard)

To generate the required output files, run:
```bash
cd ..
python scripts/run_demo.py --output-dir ./output
```

## Usage Tips

### Streamlit Dashboard
- Use the sidebar to configure valuation date and parameters
- Navigate between tabs to explore different analytics
- Interactive charts support zoom, pan, and hover for details
- Download data as CSV from the Data Explorer tab
- Adjust sliders and inputs to see real-time updates

### Shiny Dashboard
- Automatically refreshes from CSV files in the output directory
- Shows real-time risk limits and utilization
- Color-coded alerts for limit breaches
- Filterable position tables

## Technical Notes

### Streamlit Dashboard Architecture
The interactive dashboard uses:
- **Streamlit** for the web framework
- **Plotly** for interactive visualizations
- **Pandas** for data manipulation
- **NumPy** for numerical calculations
- Integrates directly with all rateslib modules

### Performance
- Curves are cached using `@st.cache_resource`
- Data is cached using `@st.cache_data`
- Rebuilds curves only when valuation date or quotes change

### Extensibility
The dashboard is designed to be easily extended:
- Add new tabs by duplicating the tab structure
- Add new visualizations using Plotly
- Integrate additional rateslib modules as they're developed
- Customize styling via the embedded CSS

## Coverage

The interactive dashboard covers **all major library components**:

✅ Curve Construction
- OIS bootstrapping
- Nelson-Siegel-Svensson fitting
- Multiple interpolation methods

✅ Instrument Pricing
- Bond pricing (clean/dirty/accrued)
- Interest rate swaps
- Futures contracts

✅ Risk Analytics
- DV01 (parallel shift)
- Key Rate DV01 (bucketed)
- Convexity (second-order)

✅ VaR & Stress Testing
- Historical Simulation
- Monte Carlo
- Stressed VaR
- Expected Shortfall

✅ Scenario Analysis
- Standard scenarios (9 types)
- Custom scenario builder
- P&L impact calculation

✅ P&L Attribution
- Carry & rolldown
- Curve movement (parallel & non-parallel)
- Convexity effects
- Residual analysis

✅ Liquidity Risk
- LVaR calculation
- Bid/ask spreads
- Holding period adjustments
- Position size impacts

## Screenshots

(Run the dashboard to see the interactive visualizations)

## Future Enhancements

Potential additions:
- SABR volatility surface visualization (when vol data is available)
- Options Greeks calculator
- Real-time data feeds integration
- Multi-portfolio comparison
- Historical P&L tracking
- Custom reporting templates

## Support

For issues or questions:
1. Check the main README.md
2. Review the example scripts in `../scripts/`
3. Examine the test files in `../tests/`

## License

MIT License - See main repository LICENSE file
