# Rates Risk Library

A comprehensive Python library for USD yield curve construction, fixed-income instrument pricing, and risk analytics. Designed for trading desk workflows including real-time risk monitoring, P&L attribution, and regulatory risk metrics.

## Features

### Curve Construction
- **OIS Bootstrap**: Sequential bootstrap of SOFR/OIS discount curve from swap quotes
- **Treasury NSS**: Nelson-Siegel-Svensson parametric fitting for Treasury par yields
- **Interpolation**: Cubic spline (primary), linear, log-linear on discount factors

### Instrument Pricing
- **Bonds**: Coupon bonds with clean/dirty prices, accrued interest, YTM
- **Swaps**: Vanilla interest rate swaps (fixed vs floating)
- **Futures**: SOFR/Eurodollar rate futures pricing with P&L tracking
- **Swaptions**: European swaptions with SABR volatility model
- **Caplets/Floors**: Interest rate caps and floors with SABR pricing

### Volatility Modeling
- **SABR Model**: Industry-standard stochastic volatility model
- **Volatility Surface**: Multi-dimensional vol surface by expiry and tenor
- **Greeks**: Delta, vega, and parameter sensitivities
- **Multiple Conventions**: Support for Normal (basis point) and Lognormal (Black) volatilities

### Risk Metrics
- **DV01**: Dollar value of 1 basis point parallel shift
- **Key Rate DV01**: Sensitivity by maturity bucket
- **Convexity**: Second-order rate sensitivity
- **Bump-and-Reprice**: Full revaluation risk framework
- **Options Greeks**: Delta, vega, rho sensitivities for swaptions/caplets

### VaR & Stress Testing
- **Historical Simulation**: Full repricing using historical rate moves
- **Monte Carlo**: Multivariate normal simulation
- **Stressed VaR**: Predefined stress periods (COVID-2020, Rate Hike 2022)
- **Scenario Analysis**: 9 standard scenarios plus custom scenario builder
  - Parallel shifts, twists, steepeners/flatteners
  - NSS parameter tweaking (β₀-β₃, λ₁-λ₂)
  - SABR parameter stressing (σ_ATM, ν, ρ)
  - Full portfolio repricing with P&L attribution

### P&L Attribution
- Carry & rolldown decomposition
- Curve movement attribution
- Convexity P&L
- Residual/unexplained analysis

### Reporting
- Formatted console output
- CSV export for downstream systems

### Interactive Dashboards
- **Real-Time Risk Monitor**: Shiny-based dashboard for live risk monitoring
- **Interactive Analytics Dashboard**: Comprehensive Streamlit dashboard covering ALL library functionality
  - Curve visualization and analysis
  - Interactive pricing calculators (bonds, swaps, futures, swaptions, caplets)
  - SABR volatility surface visualization with implied vol curves
  - Risk metrics with visual breakdowns
  - VaR/ES analysis with distributions
  - Enhanced scenario analysis with custom builder:
    - NSS curve parameter tweaking
    - SABR parameter stressing
    - Live curve visualization
    - Full portfolio repricing with P&L attribution
  - P&L attribution decomposition (curve vs vol)
  - Liquidity risk (LVaR) calculations
  - Data explorer and export
  - **100% position coverage** - all instrument types supported

## Installation

```bash
# Clone the repository
git clone https://github.com/7josh7/rates_risk_lib.git
cd rates_risk_lib

# Install in development mode
pip install -e .

# Or install core dependencies directly
pip install numpy pandas scipy pyyaml
```

### Optional: Dashboard Dependencies

The dashboards have optional dependencies that are not part of the core library install:

```bash
cd dashboard
pip install -r requirements_interactive.txt   # Streamlit dashboard
pip install -r requirements.txt               # Shiny dashboard
```

## Quick Start

```python
from datetime import date
from rateslib.curves import OISBootstrapper, NelsonSiegelSvensson
from rateslib.pricers import BondPricer, SwapPricer
from rateslib.risk import RiskCalculator

# Build OIS curve from swap quotes
valuation_date = date(2024, 1, 15)
bootstrapper = OISBootstrapper(valuation_date)
bootstrapper.add_quote("1M", 0.0530)
bootstrapper.add_quote("3M", 0.0528)
bootstrapper.add_quote("6M", 0.0525)
bootstrapper.add_quote("1Y", 0.0520)
bootstrapper.add_quote("2Y", 0.0510)
bootstrapper.add_quote("5Y", 0.0480)
bootstrapper.add_quote("10Y", 0.0450)

result = bootstrapper.bootstrap()
ois_curve = result.curve

# Price a bond
bond_pricer = BondPricer()
price_result = bond_pricer.price(
    settlement_date=valuation_date,
    maturity_date=date(2029, 1, 15),
    coupon=0.04,  # 4% coupon
    frequency=2,  # Semi-annual
    discount_curve=ois_curve
)
print(f"Clean Price: {price_result['clean_price']:.4f}")
print(f"Dirty Price: {price_result['dirty_price']:.4f}")

# Calculate risk metrics
calculator = RiskCalculator(ois_curve)
risk = calculator.compute_bond_risk(
    settlement_date=valuation_date,
    maturity_date=date(2029, 1, 15),
    coupon=0.04,
    frequency=2,
    notional=10_000_000
)
print(f"DV01: ${risk.dv01:,.2f}")
print(f"Convexity: {risk.convexity:,.4f}")
```

## Running the Demo

A complete demo script is provided that demonstrates the full workflow:

```bash
python scripts/run_demo.py --output-dir ./output
```

This will:
1. Load sample market data (OIS quotes, Treasury yields, historical rates)
2. Build discount and Treasury curves
3. Price a sample portfolio (bonds, swaps, futures)
4. Calculate DV01 and key-rate sensitivities
5. Run Historical and Monte Carlo VaR
6. Generate scenario analysis
7. Output formatted reports and CSV files

## Interactive Dashboards

Two interactive dashboards are provided for visualization:

### 1. Real-Time Risk Monitor (Shiny)
```bash
cd dashboard
pip install -r requirements.txt
shiny run app.py --reload
```

Features:
- Real-time portfolio monitoring
- Key Rate DV01 visualization
- VaR/ES metrics display
- Risk limit tracking
- Position-level details

### 2. Interactive Analytics Dashboard (Streamlit)
```bash
cd dashboard
pip install -r requirements_interactive.txt
streamlit run interactive_dashboard.py
```

**Comprehensive coverage of ALL library functionality:**
- **📈 Curves**: OIS bootstrap, Treasury NSS, curve comparison, discount factors, forward rates
- **💰 Pricing**: Interactive calculators for bonds, swaps, and futures
- **📊 Risk Metrics**: DV01, Key Rate ladders, convexity analysis
- **🎯 VaR Analysis**: Historical, Monte Carlo, Stressed VaR with distribution plots
- **📉 Scenarios**: Standard scenarios, waterfall charts, custom scenario builder
- **💵 P&L Attribution**: Carry, rolldown, curve moves, convexity breakdown
- **💧 Liquidity Risk**: LVaR calculations, bid-ask impacts, holding period scaling
- **📋 Data Explorer**: Browse market data, positions, curve nodes, export to CSV

**Notes:**
- The dashboard includes a few intentionally simplified panels (e.g., liquidity “what-if” calculator and combined curve+vol additivity check). See `dashboard/README.md` for details.

See `dashboard/README.md` for detailed dashboard documentation.

## Technical Documentation

- Full technical write-up: `documentation.tex`

## Project Structure

```
rates_risk_lib/
├── src/
│   └── rateslib/
│       ├── __init__.py
│       ├── conventions.py     # Day count, business day conventions
│       ├── dates.py           # Tenor parsing, schedule generation
│       ├── market_state.py    # Curves + SABR surface container
│       ├── curves/
│       │   ├── __init__.py
│       │   ├── curve.py       # Curve class with interpolation
│       │   ├── interpolation.py
│       │   ├── instruments.py # Bootstrap instruments
│       │   ├── bootstrap.py   # OIS bootstrapper
│       │   └── nss.py         # Nelson-Siegel-Svensson
│       ├── pricers/
│       │   ├── __init__.py
│       │   ├── bonds.py       # Bond pricing
│       │   ├── swaps.py       # IRS pricing
│       │   ├── futures.py     # Futures pricing
│       │   └── dispatcher.py  # Unified pricing dispatcher
│       ├── options/
│       │   ├── __init__.py
│       │   ├── base_models.py # Black-Scholes foundation
│       │   ├── swaption.py    # Swaption pricing
│       │   ├── caplet.py      # Caplet/floor pricing
│       │   └── sabr_risk.py   # SABR Greeks
│       ├── vol/
│       │   ├── __init__.py
│       │   ├── sabr.py        # SABR model implementation
│       │   ├── calibration.py # SABR calibration
│       │   ├── quotes.py      # Vol quote handling
│       │   └── sabr_surface.py # Vol surface management
│       ├── portfolio/
│       │   ├── __init__.py
│       │   └── builders.py    # Trade builders + validation
│       ├── risk/
│       │   ├── __init__.py
│       │   ├── bumping.py     # Bump engine
│       │   ├── keyrate.py     # Key-rate DV01
│       │   ├── limits.py      # Risk limits and evaluation
│       │   ├── reporting.py   # Reporting engines + coverage diagnostics
│       │   ├── sensitivities.py
│       ├── pnl/
│       │   ├── __init__.py
│       │   └── attribution.py # P&L decomposition
│       ├── var/
│       │   ├── __init__.py
│       │   ├── historical.py  # Historical VaR
│       │   ├── monte_carlo.py # Monte Carlo VaR
│       │   ├── stress.py      # Stressed VaR
│       │   └── scenarios.py   # Scenario analysis
│       ├── liquidity/
│       │   └── __init__.py    # Liquidity adjustments and LVaR
│       └── reporting/
│           ├── __init__.py
│           └── risk_report.py # Report generation
├── data/
│   ├── sample_quotes/         # Market data samples
│   └── sample_book/           # Position data samples
│   └── vol_quotes.csv         # Sample vol quotes (optional)
├── dashboard/
│   ├── app.py                 # Real-time risk monitor (Shiny)
│   ├── interactive_dashboard.py  # Interactive analytics (Streamlit)
│   ├── FEATURES.md
│   ├── QUICKSTART.md
│   ├── requirements.txt       # Shiny dependencies
│   ├── requirements_interactive.txt  # Streamlit dependencies
│   ├── launch_interactive.bat
│   └── launch_interactive.sh
│   └── README.md              # Dashboard documentation
├── scripts/
│   ├── run_demo.py            # Main demo script
│   ├── run_sabr_demo.py       # SABR/options demo
│   └── run_comprehensive_demo.py  # Full system demo
├── tests/
│   ├── test_conventions.py
│   ├── test_dates.py
│   ├── test_curves.py
│   ├── test_pricers.py
│   ├── test_options.py        # Swaption/caplet tests
│   ├── test_sabr.py           # SABR model tests
│   ├── test_sabr_risk_conventions.py  # SABR Greeks tests
│   ├── test_risk.py
│   ├── test_var.py
│   ├── test_portfolio_builders.py
│   ├── test_reporting_engines.py
│   └── test_limit_reporting.py  # Risk limit tests
├── documentation.tex
├── pyproject.toml
└── README.md
```

## API Reference

### Curve Construction

#### OISBootstrapper
```python
from rateslib.curves import OISBootstrapper

bootstrapper = OISBootstrapper(valuation_date)
bootstrapper.add_quote(tenor, rate)  # e.g., "1Y", 0.05
result = bootstrapper.bootstrap()
curve = result.curve
```

#### NelsonSiegelSvensson
```python
from rateslib.curves import NelsonSiegelSvensson
import numpy as np

nss = NelsonSiegelSvensson()
nss.fit(tenors=np.array([1, 2, 5, 10, 30]), yields=np.array([...]))
curve = nss.to_curve(valuation_date, max_maturity=30)
```

### Pricers

#### BondPricer
```python
from rateslib.pricers import BondPricer

pricer = BondPricer()
result = pricer.price(settlement_date, maturity_date, coupon, frequency, discount_curve)
dv01 = pricer.dv01(settlement_date, maturity_date, coupon, frequency, discount_curve, notional)
ytm = pricer.yield_to_maturity(settlement_date, maturity_date, coupon, frequency, price)
```

#### SwapPricer
```python
from rateslib.pricers import SwapPricer

pricer = SwapPricer()
pv = pricer.present_value(valuation_date, start_date, maturity_date, fixed_rate, 
                          notional, pay_fixed, discount_curve, projection_curve)
par_rate = pricer.par_rate(valuation_date, start_date, maturity_date, 
                           discount_curve, projection_curve)
```

### Risk

#### BumpEngine
```python
from rateslib.risk import BumpEngine

engine = BumpEngine(curve)
dv01 = engine.compute_dv01(pricer_func)
convexity = engine.compute_convexity(pricer_func)
bumped_curve = engine.parallel_bump(10)  # +10bp
```

#### KeyRateEngine
```python
from rateslib.risk import KeyRateEngine

engine = KeyRateEngine(curve, tenors=['2Y', '5Y', '10Y', '30Y'])
kr_dv01 = engine.compute_key_rate_dv01(pricer_func)
```

### VaR

#### Historical Simulation
```python
from rateslib.var import HistoricalSimulation

sim = HistoricalSimulation(historical_data, lookback_days=252)
result = sim.run_simulation(base_curve, pricer_func)
print(f"VaR 95%: {result.var_95}")
print(f"ES 99%: {result.es_99}")
```

#### Scenario Analysis
```python
from rateslib.var import ScenarioEngine, STANDARD_SCENARIOS

engine = ScenarioEngine(base_curve, pricer_func)
results = engine.run_standard_scenarios()
for r in results:
    print(f"{r.scenario.name}: {r.pnl:,.0f}")
```

## Conventions

The library uses the following market conventions:

| Instrument | Day Count | Settlement | Frequency |
|------------|-----------|------------|-----------|
| OIS Swaps | ACT/360 | T+2 | Annual |
| Treasury | ACT/ACT | T+1 | Semi-annual |
| IRS (Fixed) | 30/360 | T+2 | Semi-annual |
| IRS (Float) | ACT/360 | T+2 | Quarterly |

## Testing

The library includes **177 comprehensive tests** covering all modules:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=rateslib --cov-report=html

# Run specific test file
pytest tests/test_curves.py -v

# Run SABR/options tests
pytest tests/test_sabr.py tests/test_options.py -v
```

**Test Coverage:**
- ✅ Curve construction and interpolation
- ✅ All instrument pricers (bonds, swaps, futures, options)
- ✅ SABR model and calibration
- ✅ Risk metrics and sensitivities
- ✅ VaR/stress testing
- ✅ Scenario analysis with futures support
- ✅ P&L attribution
- ✅ Options Greeks and volatility conventions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This library was designed based on common trading desk workflows and risk management practices for USD linear rates products.
