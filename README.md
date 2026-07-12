# Rates Risk Library

Rates Risk Library (`rates_risk`) is a Python library for USD rates analytics. It covers curve construction, instrument pricing, rates-option volatility modeling, desk-style risk metrics, VaR/stress testing, P&L attribution, liquidity adjustments, and reporting utilities. The repository also includes sample market data, sample books, demo scripts, and two optional dashboards.

> **Status:** experimental research library. Automated tests provide regression
> evidence, not independent model approval. See
> [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md) for the validated-versus-
> experimental inventory and open benchmark work.

## Scope

- USD-focused rates workflows
- OIS discount curve bootstrap from quote dictionaries or instrument objects
- Treasury curve fitting with Nelson-Siegel-Svensson (NSS)
- Pricing for bonds, vanilla swaps, SOFR/rate futures, swaptions, and caplets/floors
- SABR model utilities, calibration, and smile-aware option risk
- DV01, key-rate DV01, convexity, VaR/ES, scenarios, limits, P&L attribution, and liquidity-adjusted VaR
- Console and CSV reporting helpers

## Main Components

### Curves
- `Curve` for discount factors, zero rates, forward rates, and interpolation
- `bootstrap_from_quotes(...)` convenience helper for `DEPOSIT`, `OIS`, `FRA`, and `FUT/FUTURE` quote dictionaries
- `OISBootstrapper` for explicit instrument-level bootstrapping workflows
- `NelsonSiegelSvensson` for parametric Treasury fitting and curve generation

### Pricing
- `BondPricer`
- `SwapPricer`
- `FuturesPricer`
- `price_trade(...)` / `risk_trade(...)` dispatchers for portfolio-style trade dictionaries

### Volatility and Options
- `SabrModel`, `SabrCalibrator`, `build_sabr_surface(...)`
- `SwaptionPricer`
- `CapletPricer`
- `SabrOptionRisk`
- Normal (Bachelier) and lognormal (Black-style) support

### Risk and Reporting
- `BumpEngine`, `RiskCalculator`, `KeyRateEngine`
- `HistoricalSimulation`, `MonteCarloVaR`, `StressedVaR`, `ScenarioEngine`
- `PnLAttributionEngine`
- `LiquidityEngine`
- `RiskReport`, `ReportFormatter`, `export_to_csv(...)`

## Installation

### Core library

```bash
git clone https://github.com/7josh7/rates_risk_lib.git
cd rates_risk_lib
python -m pip install -e .
```

### Development and tests

```bash
python -m pip install -e .[dev]
```

### Optional dashboard dependencies

```bash
python -m pip install -e ".[all]"
```

The distribution is named `rates-risk-lib` and imports as `rates_risk`. Existing
local users of the former ambiguous `rateslib` namespace should follow
[`MIGRATION.md`](MIGRATION.md).

Core package requirements are declared in `pyproject.toml`:

- Python 3.10+
- `numpy`
- `pandas`
- `scipy`
- `pyyaml`

## Quick Start

The example below loads the sample OIS quotes shipped with the repository, bootstraps a curve, prices a bond, and computes bond risk.

```python
from datetime import date

import pandas as pd

from rates_risk.curves import bootstrap_from_quotes
from rates_risk.pricers import BondPricer
from rates_risk.risk import RiskCalculator

valuation_date = date(2024, 1, 15)

ois_quotes = pd.read_csv("data/sample_quotes/ois_quotes.csv", comment="#")
quotes = (
    ois_quotes[["instrument_type", "tenor", "rate", "day_count"]]
    .rename(columns={"rate": "quote"})
    .to_dict("records")
)

ois_curve = bootstrap_from_quotes(
    anchor_date=valuation_date,
    quotes=quotes,
)

bond_pricer = BondPricer(ois_curve)
dirty_price, clean_price, accrued = bond_pricer.price(
    settlement=valuation_date,
    maturity=date(2029, 1, 15),
    coupon_rate=0.04,
    frequency=2,
)

risk = RiskCalculator(ois_curve).compute_bond_risk(
    instrument_id="UST_5Y",
    settlement=valuation_date,
    maturity=date(2029, 1, 15),
    coupon_rate=0.04,
    notional=10_000_000,
    frequency=2,
)

print(f"Clean price: {clean_price:.4f}")
print(f"Dirty price: {dirty_price:.4f}")
print(f"Accrued interest: {accrued:.4f}")
print(f"DV01: {risk.dv01:,.2f}")
print(f"Convexity: {risk.convexity:,.2f}")
```

### Treasury NSS fit

```python
from datetime import date

import pandas as pd

from rates_risk.curves import NelsonSiegelSvensson
from rates_risk.dates import DateUtils

valuation_date = date(2024, 1, 15)
treasury_quotes = pd.read_csv("data/sample_quotes/treasury_quotes.csv", comment="#")

tenors = treasury_quotes["tenor"].map(DateUtils.tenor_to_years).to_numpy()
yields = treasury_quotes["yield"].to_numpy()

nss = NelsonSiegelSvensson(valuation_date)
nss.fit(tenors, yields)

treasury_curve = nss.to_curve(tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
print(nss.params)
```

### Portfolio-style trade dispatch

For portfolio workflows, build trade dictionaries from position rows and price them through `MarketState`.

```python
from datetime import date

import pandas as pd

from rates_risk import CurveState, MarketState, build_trade_from_position, price_trade
from rates_risk.curves import bootstrap_from_quotes

valuation_date = date(2024, 1, 15)

ois_quotes = pd.read_csv("data/sample_quotes/ois_quotes.csv", comment="#")
curve = bootstrap_from_quotes(
    valuation_date,
    ois_quotes.rename(columns={"rate": "quote"}).to_dict("records"),
)

positions = pd.read_csv("data/sample_book/positions.csv", comment="#")
trade = build_trade_from_position(positions.iloc[0], valuation_date)

market_state = MarketState(CurveState(discount_curve=curve, projection_curve=curve))
result = price_trade(trade, market_state)

print(result.instrument_type, result.pv)
```

For swaption and caplet pricing through `price_trade(...)`, include a SABR surface in `MarketState` or provide flat volatility fields on the trade.

## Demo Scripts

### End-to-end desk demo

```bash
python scripts/run_demo.py --output-dir ./output
```

This script:

1. Loads sample OIS, Treasury, historical-rate, portfolio, and optional vol data.
2. Builds discount and Treasury curves.
3. Prices the sample book.
4. Computes risk metrics and limit diagnostics.
5. Runs VaR and scenario analysis.
6. Writes reports and CSV outputs that the Shiny dashboard can consume.

### SABR and options demo

```bash
python scripts/run_sabr_demo.py
```

Demonstrates SABR parameterization, smile generation, calibration, pricing, and option risk outputs.

### Comprehensive rates plus options demo

```bash
python scripts/run_comprehensive_demo.py
```

Runs the broader rates-plus-options workflow, including options in the sample portfolio and combined market-state style analytics.

## Dashboards

Two optional dashboards live in [`dashboard/`](dashboard/):

### 1. Shiny real-time risk monitor

```bash
cd dashboard
python -m pip install -r requirements.txt
shiny run app.py --reload
```

Best used after generating `./output` files with `scripts/run_demo.py`.

### 2. Streamlit interactive analytics dashboard

```bash
cd dashboard
python -m pip install -r requirements_interactive.txt
streamlit run interactive_dashboard.py
```

The Streamlit app covers curves, pricing, risk metrics, VaR, scenarios, P&L attribution, liquidity what-if analysis, and sample-data exploration. See [`dashboard/README.md`](dashboard/README.md) and [`dashboard/QUICKSTART.md`](dashboard/QUICKSTART.md) for dashboard-specific notes.

## Sample Data

Repository data files are organized as:

- `data/sample_quotes/ois_quotes.csv`: sample OIS quote set
- `data/sample_quotes/treasury_quotes.csv`: sample Treasury par-yield inputs for NSS fitting
- `data/sample_quotes/historical_rates.csv`: historical tenor grid for VaR/stress testing
- `data/sample_book/positions.csv`: mixed sample book used by demos and dashboards
- `data/sample_book/options.csv`: standalone options sample data
- `data/vol_quotes.csv`: optional SABR calibration inputs

Historical VaR and Monte Carlo utilities accept either:

- long format with columns like `date`, `tenor`, `rate`
- wide format with a `date` column and tenor columns such as `3M`, `2Y`, `10Y`

## Position Schema Notes

The portfolio builders are intentionally strict for options. They do not infer missing option dates or types.

| Instrument | Required fields |
|------------|-----------------|
| `UST` / `BOND` | `position_id`, `instrument_type`, `maturity_date`, `notional`, `direction`; usually `coupon` |
| `IRS` / `SWAP` | `position_id`, `instrument_type`, `maturity_date`, `notional`, `direction`; usually `coupon` as fixed rate |
| `FUT` | `position_id`, `instrument_type`, `expiry_date` or `maturity_date`, `notional` or `num_contracts`; `direction` if contracts are unsigned |
| `SWAPTION` | explicit option fields are required: `expiry_date` or `expiry_tenor`, `underlying_swap_tenor`, `payer_receiver`, `position`, `strike`, `notional` |
| `CAPLET` | explicit option fields are required: `caplet_start_date`, `caplet_end_date`, `position`, `strike`, `notional` |

Notes:

- `option_type` is the preferred explicit discriminator for option rows when building trades from generic position data.
- For swaptions, expiry must be specified explicitly. The builder will not infer expiry from `maturity_date`.
- For caplets, accrual start and end dates must be provided explicitly.
- Sample column usage is documented in comments at the top of `data/sample_book/positions.csv`.

## Project Structure

```text
rates_risk_lib/
|-- src/rates_risk/
|   |-- curves/       # Curve objects, interpolation, bootstrap, NSS
|   |-- pricers/      # Bond, swap, futures, unified trade dispatch
|   |-- options/      # Swaption, caplet, SABR option risk
|   |-- vol/          # SABR model, calibration, quotes, surfaces
|   |-- risk/         # Bumping, sensitivities, key rates, limits, reporting helpers
|   |-- var/          # Historical, Monte Carlo, stressed VaR, scenarios
|   |-- pnl/          # P&L attribution
|   |-- liquidity/    # Liquidity-adjusted VaR helpers
|   |-- portfolio/    # Trade builders and portfolio diagnostics
|   |-- reporting/    # Console and CSV reporting
|   |-- market_state.py
|   |-- conventions.py
|   `-- dates.py
|-- data/
|-- scripts/
|-- dashboard/
|-- tests/
|-- documentation.tex
|-- CHANGELOG.md
`-- README.md
```

## API Reference

The extended public API inventory lives in [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md).

## Default Conventions in Code

These are the main defaults used by the core engines unless you pass custom conventions:

| Component | Default behavior |
|-----------|------------------|
| Curve zero rates | Continuously compounded |
| Curve interpolation | Cubic spline by default |
| OIS bootstrap day count | `ACT/360` |
| Treasury bond pricing | `ACT/ACT`, semi-annual coupons |
| Swap fixed leg | `ACT/360`, semi-annual payments |
| Swap floating leg | `ACT/360`, quarterly payments |

Settlement and effective dates are generally passed explicitly by the caller or set by the surrounding demo/builder workflow.

## Testing

The suite covers curves, pricing, options, risk, VaR, reporting, portfolio
builders, and optional dashboard control paths. CI reports the authoritative
collected test count so this page does not become stale as cases are added.

Install dev dependencies first:

```bash
python -m pip install -e .[dev]
```

Then run:

```bash
python -m pytest tests
python -m pytest tests --cov=rates_risk --cov-report=html
python -m pytest tests/test_curves.py -v
python -m pytest tests/test_sabr.py tests/test_options.py -v
```

## Known Limitations and Assumptions

- The library is focused on USD rates. It does not cover credit, FX, cross-currency, or multi-curve collateral frameworks beyond the provided discount/projection split.
- The sample workflows are CSV-driven. Live market data ingestion, trade capture, and real-time connectivity are not included.
- Some dashboard panels are intentionally simplified or illustrative, especially liquidity what-if views, additive combined curve-plus-vol checks, and user-driven option attribution inputs.
- The codebase has broad regression coverage, but still needs independent price/risk benchmarks and end-to-end option validation before any production-use assessment.
- The examples and dashboards are designed around the shipped sample data; if you bring your own data, field names and units need to match the expected schemas.

## Additional Documentation

- [`documentation.tex`](documentation.tex): technical write-up
- [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md): extended public API inventory
- [`CHANGELOG.md`](CHANGELOG.md): project history
- [`MODEL_VALIDATION.md`](MODEL_VALIDATION.md): current validation status and open gaps
- [`MIGRATION.md`](MIGRATION.md): package-namespace migration
- [`dashboard/README.md`](dashboard/README.md): dashboard details

## Contributing

1. Create a feature branch.
2. Make and test your changes.
3. Update documentation when behavior changes.
4. Open a pull request with a short summary of user-facing impact.

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

This library is organized around common sell-side and buy-side desk workflows for rates analytics, with sample data and demos intended to make the implementation easy to inspect and extend.
