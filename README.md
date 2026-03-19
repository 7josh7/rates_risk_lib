# Rates Risk Library

Rates Risk Library (`rateslib`) is a Python library for USD rates analytics. It covers curve construction, instrument pricing, rates-option volatility modeling, desk-style risk metrics, VaR/stress testing, P&L attribution, liquidity adjustments, and reporting utilities. The repository also includes sample market data, sample books, demo scripts, and two optional dashboards.

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
cd dashboard
python -m pip install -r requirements.txt
python -m pip install -r requirements_interactive.txt
```

Core package requirements are declared in `pyproject.toml`:

- Python 3.9+
- `numpy`
- `pandas`
- `scipy`
- `pyyaml`

## Quick Start

The example below loads the sample OIS quotes shipped with the repository, bootstraps a curve, prices a bond, and computes bond risk.

```python
from datetime import date

import pandas as pd

from rateslib.curves import bootstrap_from_quotes
from rateslib.pricers import BondPricer
from rateslib.risk import RiskCalculator

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

from rateslib.curves import NelsonSiegelSvensson
from rateslib.dates import DateUtils

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

from rateslib import CurveState, MarketState, build_trade_from_position, price_trade
from rateslib.curves import bootstrap_from_quotes

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
|-- src/rateslib/
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

The package-root API is listed first, so `from rateslib import ...` works for every symbol in those sections. Public methods and properties defined on exported classes are listed under each class. Additional public exports that are only available from submodules are listed afterward.

### Package Metadata

- `__version__`: Package version string exposed at the root module.

### Conventions and Dates

- `DayCount`: Enum of day-count rules: `ACT_360`, `ACT_365`, `ACT_ACT`, `THIRTY_360`.
  - `from_string(cls, s)`: Parse a string day-count label into the enum.
- `BusinessDayConvention`: Enum of date-adjustment rules: `MODIFIED_FOLLOWING`, `FOLLOWING`, `PRECEDING`, `UNADJUSTED`.
- `Conventions(day_count, business_day, compounding, payment_frequency, settlement_days)`: Dataclass for instrument conventions.
  - `usd_ois(cls)`: Return standard USD OIS conventions.
  - `usd_treasury(cls)`: Return standard USD Treasury conventions.
  - `usd_swap(cls)`: Return standard USD fixed-leg swap conventions.
- `year_fraction(start, end, day_count)`: Compute year fractions under the selected day-count convention.
- `DateUtils`: Static date helper class for tenor parsing, schedule generation, and date arithmetic.
  - `parse_tenor(tenor)`: Parse a tenor string like `3M` or `5Y`.
  - `add_tenor(start, tenor, holidays=None)`: Add a tenor to a date.
  - `tenor_to_years(tenor)`: Convert a tenor string to an approximate year fraction.
  - `generate_schedule(start, end, frequency, convention=BusinessDayConvention.MODIFIED_FOLLOWING, holidays=None, stub='short_front')`: Generate a standard coupon schedule.
  - `generate_ois_schedule(start, end, payment_frequency='ANNUAL', holidays=None)`: Generate an OIS-style schedule.
- `ScheduleInfo(payment_dates, accrual_starts, accrual_ends, year_fractions, day_count)`: Schedule container returned by the schedule-generation helpers.

### Curves and Interpolation

- `Curve(anchor_date, currency='USD', day_count=DayCount.ACT_365, interpolation_method='cubic_spline')`: Yield-curve object with interpolation and bump helpers.
  - `add_node(time, discount_factor)`: Add a discount-factor node by year fraction.
  - `add_node_from_date(d, discount_factor)`: Add a node using a calendar date.
  - `build()`: Rebuild the interpolator from the stored nodes.
  - `discount_factor(t)`: Return `P(0,t)`.
  - `zero_rate(t, compounding=CompoundingConvention.CONTINUOUS)`: Return the zero rate at maturity `t`.
  - `forward_rate(t1, t2, compounding=CompoundingConvention.SIMPLE)`: Return the forward rate between `t1` and `t2`.
  - `instantaneous_forward(t)`: Return the instantaneous forward rate.
  - `get_nodes()`: Return all curve nodes.
  - `get_node_times()`: Return node maturities as year fractions.
  - `get_node_dfs()`: Return node discount factors.
  - `get_node_rates()`: Return node zero rates.
  - `bump_parallel(bp)`: Return a new curve with a parallel bump in basis points.
  - `bump_node(node_index, bp)`: Return a new curve with one node bumped.
  - `bump_tenor(tenor, bp)`: Return a new curve with the node closest to a tenor bumped.
  - `copy()`: Return a deep copy of the curve.
- `OISBootstrapper(anchor_date, day_count=DayCount.ACT_360, tolerance=1e-08, interpolation_method='cubic_spline')`: Bootstraps an OIS discount curve from instrument objects.
  - `bootstrap(instruments, verify=True)`: Bootstrap a curve from deposits, OIS swaps, FRAs, or futures-style curve instruments.
- `NelsonSiegelSvensson(anchor_date, day_count=DayCount.ACT_ACT)`: Nelson-Siegel-Svensson parametric yield-curve model.
  - `yield_at(tau)`: Return fitted yield at maturity `tau`.
  - `discount_factor(tau)`: Return discount factor implied by the fitted curve.
  - `forward_rate(t1, t2)`: Return a forward rate between two maturities.
  - `instantaneous_forward(tau)`: Return the instantaneous forward rate.
  - `fit(maturities, yields, weights=None, initial_guess=None, method='L-BFGS-B')`: Fit NSS parameters to yield observations.
  - `fit_from_par_yields(tenors, par_yields, weights=None)`: Fit the model directly from par-yield inputs.
  - `to_curve(tenors=None, interpolation_method='cubic_spline')`: Convert the fitted model into a `Curve`.
  - `get_curve_data(min_tenor=0.0, max_tenor=30.0, num_points=100)`: Return sampled curve data for plotting or analysis.
- `LinearInterpolator()`: Linear interpolation implementation.
  - `fit(times, values)`: Fit the interpolator.
  - `interpolate(t)`: Interpolate with flat extrapolation.
  - `derivative(t)`: Return the piecewise-constant slope.
- `CubicSplineInterpolator()`: Natural cubic-spline interpolation.
  - `fit(times, values)`: Fit the spline coefficients.
  - `interpolate(t)`: Evaluate the spline.
  - `derivative(t)`: Return the first derivative.
  - `second_derivative(t)`: Return the second derivative.
- `LogLinearInterpolator()`: Log-linear interpolation on discount factors.
  - `fit(times, discount_factors)`: Fit the log-linear interpolator.
  - `interpolate(t)`: Interpolate the log discount factor.
  - `get_discount_factor(t)`: Return the discount factor at maturity `t`.
  - `derivative(t)`: Return the derivative of the log discount factor.

### Pricing Engines

- `BondPricer(curve, conventions=None)`: Bond pricing engine.
  - `generate_cashflows(settlement, maturity, coupon_rate, face_value=100.0, frequency=2)`: Build the bond cashflow schedule.
  - `present_value(cashflows)`: Discount supplied cashflows to present value.
  - `price(settlement, maturity, coupon_rate, face_value=100.0, frequency=2)`: Return dirty price, clean price, and accrued interest for a coupon bond.
  - `price_zero_coupon(settlement, maturity, face_value=100.0)`: Price a zero-coupon bond.
  - `yield_to_maturity(price, settlement, maturity, coupon_rate, face_value=100.0, frequency=2, is_clean=True)`: Solve for yield from price.
  - `compute_dv01(settlement, maturity, coupon_rate, face_value=100.0, frequency=2, notional=1000000)`: Compute bond DV01 by numerical bumping.
- `SwapPricer(discount_curve, projection_curve=None, fixed_conventions=None, float_conventions=None)`: Vanilla fixed-float IRS pricing engine.
  - `forward_swap_rate(expiry, tenor)`: Compute the forward par swap rate and annuity for a forward-starting swap.
  - `generate_cashflows(effective, maturity, notional, fixed_rate, pay_receive='PAY')`: Build fixed and floating leg cashflows.
  - `present_value(effective, maturity, notional, fixed_rate, pay_receive='PAY')`: Calculate swap PV.
  - `par_rate(effective, maturity)`: Return the fixed rate that sets PV to zero.
  - `dv01(effective, maturity, notional, fixed_rate, pay_receive='PAY')`: Compute swap DV01.
- `FuturesContract(contract_code, expiry, contract_size=1_000_000, tick_size=0.0025, tick_value=6.25, underlying_tenor='3M')`: Dataclass describing a rate-futures contract.
  - `sofr_3m(cls, expiry)`: Create a CME 3M SOFR futures contract specification.
  - `fed_funds_30d(cls, expiry)`: Create a CME 30D Fed Funds futures contract specification.
- `FuturesPricer(curve)`: Rate-futures pricing engine.
  - `implied_rate(expiry, underlying_tenor='3M')`: Compute the implied forward rate.
  - `price_from_rate(rate)`: Convert an implied rate to a futures price.
  - `rate_from_price(price)`: Convert a futures price to an implied rate.
  - `theoretical_price(contract)`: Return the model price for a contract specification.
  - `dv01(contract, num_contracts=1)`: Compute futures DV01 for a position.
  - `position_pv(contract, num_contracts, trade_price)`: Compute mark-to-market P&L versus trade price.

### Risk, Sensitivities, and Limits

- `BumpEngine(base_curve)`: Curve-bumping and sensitivity engine.
  - `parallel_bump(bp)`: Return a parallel-bumped curve.
  - `node_bump(node_index, bp)`: Return a curve with one node bumped.
  - `tenor_bump(tenor, bp)`: Return a curve with the node nearest a tenor bumped.
  - `custom_bump(bump_vector)`: Apply a custom bump vector across nodes.
  - `compute_dv01(pricer_func, bump_size=1.0)`: Compute DV01 via bump-and-reprice.
  - `compute_convexity(pricer_func, bump_size=1.0)`: Compute dollar convexity by second difference.
  - `compute_node_deltas(pricer_func, bump_size=1.0)`: Return node-by-node deltas.
  - `scenario_pv(pricer_func, bump_profile)`: Reprice under a custom scenario bump profile.
- `RiskCalculator(curve, bump_size=1.0)`: High-level risk engine for common linear products.
  - `compute_bond_risk(instrument_id, settlement, maturity, coupon_rate, notional, frequency=2)`: Compute bond PV, DV01, duration, convexity, and key-rate risk.
  - `compute_swap_risk(instrument_id, effective, maturity, fixed_rate, notional, pay_receive='PAY')`: Compute swap risk metrics.
  - `compute_futures_risk(instrument_id, expiry, num_contracts, underlying_tenor='3M', contract_size=1000000)`: Compute futures position risk.
  - `aggregate_portfolio(instrument_risks, as_of_date)`: Aggregate instrument-level risks into `PortfolioRisk`.
- `KeyRateEngine(curve, tenors=None, bump_size=1.0, interpolate_bumps=False)`: Key-rate duration and hedge-ratio engine.
  - `compute_key_rate_dv01(pricer_func)`: Compute key-rate DV01 across configured tenors.
  - `compute_hedge_ratios(target_dv01, hedge_instruments)`: Compute hedge ratios to neutralize key-rate exposures.
- `InstrumentRisk(instrument_id, instrument_type, pv, notional, dv01, modified_duration, convexity, key_rate_dv01)`: Dataclass holding instrument-level risk results.
  - `dv01_per_million`: Property returning DV01 per million notional.
  - `to_dict()`: Convert the result to a reporting dictionary.
- `PortfolioRisk(as_of_date, total_pv, total_dv01, total_convexity, key_rate_dv01, instrument_risks)`: Dataclass holding aggregated portfolio risk.
  - `weighted_duration`: Property returning the PV-weighted duration.
  - `to_summary_dict()`: Convert portfolio totals to a summary dictionary.
- `LimitDefinition(name, metric_key, category, warn, breach, unit='', aggregation='portfolio', magnitude='abs', comparison='max', hard=True, description='')`: Dataclass defining a risk-limit rule.
- `LimitResult(definition, value, status)`: Dataclass capturing the outcome of one limit check.
  - `as_dict`: Property returning the result as a row-style dictionary.
- `DEFAULT_LIMITS`: Default limit set covering Greeks, VaR/ES, liquidity, scenarios, and model diagnostics.
- `evaluate_limits(metrics, limits=None, status_overrides=None)`: Evaluate a metrics dictionary against a list of `LimitDefinition` rules.
- `limits_to_table(results)`: Convert `LimitResult` objects to row dictionaries for reports and dashboards.
- `compute_limit_metrics(market_state, positions_df, valuation_date)`: Build the metrics dictionary consumed by limit evaluation from market state and positions.

### VaR and Scenarios

- `HistoricalSimulation(base_curve, historical_data, pricer_func, tenors=None)`: Historical simulation VaR and ES engine.
  - `run_simulation(lookback_days=None)`: Run historical simulation VaR/ES.
  - `run_parametric_var(lookback_days=None)`: Run a quick parametric approximation from the same history.
- `MonteCarloVaR(base_curve, historical_data, pricer_func, tenors=None)`: Monte Carlo VaR and ES engine.
  - `simulate_scenarios(num_paths=10000, seed=None)`: Generate simulated rate-change paths.
  - `run_simulation(num_paths=10000, seed=42)`: Run full Monte Carlo VaR/ES.
  - `run_delta_normal_var()`: Run the closed-form delta-normal approximation.
- `StressedVaR(base_curve, historical_data, pricer_func, stress_period_start, stress_period_end)`: Stressed VaR engine.
  - `compute_stressed_var()`: Compute stressed VaR over the configured stress window.
  - `from_predefined_period(cls, base_curve, historical_data, pricer_func, period_name)`: Build a stressed-VaR engine from a named historical stress period.
- `ScenarioEngine(base_curve, pricer_func, key_rate_dv01=None)`: Scenario-analysis engine.
  - `run_scenario(scenario)`: Run one custom scenario object or shock definition.
  - `run_standard_scenarios()`: Run the predefined library scenarios.
  - `run_historical_scenario(historical_data, scenario_date, scenario_name=None)`: Apply one historical market move as a scenario.
  - `find_worst_historical_scenarios(historical_data, n=5)`: Return the worst historical scenarios by P&L.
- `STANDARD_SCENARIOS`: Dictionary of predefined curve scenarios including parallel shifts, steepeners, flatteners, twists, rallies, and sell-offs.

### PnL and Reporting

- `PnLAttributionEngine(curve_t0, curve_t1, date_t0, date_t1)`: Daily P&L attribution engine for carry, rolldown, curve moves, convexity, and residual explain.
  - `attribute_pnl(instrument_id, pricer_func_t0, pricer_func_t1, risk_t0=None)`: Attribute P&L for one instrument across two valuation dates.
- `PnLAttribution(instrument_id, date_t0, date_t1, pv_t0, pv_t1, components, key_rate_contributions)`: Dataclass holding one P&L attribution result.
  - `realized_pnl`: Property returning `pv_t1 - pv_t0`.
  - `to_dict()`: Convert the attribution result to a dictionary.
- `RiskReport(report_date, portfolio_name, sections, metadata)`: Dataclass container for multi-section risk reports.
  - `add_section(title, data, notes=None)`: Append a named section to the report.
  - `to_dict()`: Convert the full report to nested dictionaries and lists.
- `ReportFormatter(width=80, precision=2, thousands_sep=True)`: Console formatter for `RiskReport` objects and supporting tables.
  - `format_number(value, precision=None)`: Format a number for display.
  - `format_bp(value)`: Format a value in basis points.
  - `format_percent(value)`: Format a percentage.
  - `header(title)`: Render a report header line.
  - `subheader(title)`: Render a subsection header line.
  - `format_dict(data, indent=2)`: Format dictionary content.
  - `format_dataframe(df, max_rows=50)`: Format a pandas DataFrame for console output.
  - `format_report(report)`: Render the full report as text.
- `export_to_csv(report, output_dir, prefix=None)`: Write a `RiskReport` to CSV files in the target directory.

### Liquidity

- `LiquidityEngine(params=None)`: Liquidity-adjustment engine for bid/ask cost, holding-period scaling, position impact, and LVaR.
  - `compute_bid_ask_cost(dv01_by_instrument, is_stressed=False)`: Compute bid/ask cost from instrument DV01 inputs.
  - `compute_holding_period_scaling(base_var, holding_period=None)`: Scale VaR to a longer liquidation horizon.
  - `compute_position_impact(dv01_by_instrument)`: Estimate additional position-size impact.
  - `compute_lvar(base_var, dv01_by_instrument, holding_period=None, is_stressed=False)`: Combine VaR and liquidity costs into `LiquidityAdjustedVaR`.
- `LiquidityAdjustedVaR(base_var, liquidity_cost, lvar, components)`: Dataclass returned by liquidity-adjusted VaR calculations.
  - `liquidity_ratio`: Property returning `lvar / base_var`.
  - `to_dict()`: Convert the result to a dictionary.

### SABR and Volatility Surface

- `SabrParams(sigma_atm, beta, rho, nu, shift=0.0)`: Dataclass of SABR parameters.
  - `to_dict()`: Convert parameters to a dictionary.
  - `from_dict(cls, d)`: Rebuild parameters from a dictionary.
- `SabrModel()`: Core SABR model implementation, including implied vol and parameter-sensitivity helpers.
  - `implied_vol_black(F, K, T, params)`: Compute Black implied volatility with the Hagan approximation.
  - `implied_vol_normal(F, K, T, params)`: Compute normal implied volatility.
  - `alpha_from_sigma_atm(F, T, params)`: Infer alpha from ATM volatility.
  - `dalpha_dsigma_atm(F, T, params)`: Return sensitivity of alpha to ATM volatility.
  - `dalpha_dtheta(F, T, params, theta_name)`: Return sensitivity of alpha to another SABR parameter.
  - `dsigma_dF(F, K, T, params, vol_type='BLACK', hold_atm_fixed=True)`: Return sensitivity of implied vol to the forward.
  - `dsigma_drho(F, K, T, params, vol_type='BLACK')`: Return sensitivity of implied vol to `rho`.
  - `dsigma_dnu(F, K, T, params, vol_type='BLACK')`: Return sensitivity of implied vol to `nu`.
  - `smile_at_strikes(F, strikes, T, params, vol_type='BLACK')`: Evaluate a SABR smile across strikes.
  - `dsigma_dsigma_atm(F, K, T, params, vol_type='BLACK', bump_size=0.0001)`: Return sensitivity of implied vol to ATM volatility.
- `SabrCalibrator(beta=0.5, use_sigma_atm_param=True)`: SABR calibration engine for market vol quotes.
  - `fit(quotes_df, F, T, shift=0.0, vol_type='NORMAL')`: Fit SABR parameters from a normalized quote DataFrame.
  - `fit_from_vol_quotes(quotes, F, T, shift=0.0)`: Fit SABR parameters from quote dictionaries.
  - `fit_error(params, quotes_df, F, T, vol_type='NORMAL')`: Compute the calibration objective for a parameter guess.
- `VolQuote(quote_date, expiry, underlying_tenor, strike, vol, vol_type, shift=0.0)`: Dataclass representing one volatility quote.
  - `strike_value(forward)`: Convert the stored strike representation to an absolute strike level.
- `load_vol_quotes(filepath, quote_date=None)`: Load volatility quotes from CSV into the library's quote schema.
- `hagan_black_vol(F, K, T, alpha, beta, rho, nu, shift=0.0)`: Standalone Hagan Black-volatility approximation.
- `hagan_normal_vol(F, K, T, alpha, beta, rho, nu, shift=0.0)`: Standalone Hagan normal-volatility approximation.
- `normalize_vol_quotes(raw_quotes, curve_state, instrument_hint='SWAPTION')`: Normalize heterogeneous quote inputs into a canonical DataFrame.
- `SabrBucketParams(sigma_atm, nu, rho, beta, shift=0.0, diagnostics)`: Dataclass for one calibrated SABR bucket plus diagnostics.
  - `to_sabr_params()`: Convert the bucket state to plain `SabrParams`.
- `SabrSurfaceState(params_by_bucket, convention, asof=None, missing_bucket_policy='nearest')`: Bucketed SABR surface container with fallback lookup and diagnostics helpers.
  - `get_bucket_params(expiry, tenor, allow_fallback=True)`: Retrieve SABR parameters for a bucket, optionally using fallback behavior.
  - `diagnostics_table()`: Return diagnostics by bucket for downstream reporting.
- `make_bucket_key(expiry, tenor)`: Normalize an `(expiry, tenor)` bucket identifier.
- `calibrate_sabr_bucket(quotes_bucket, beta, vol_type, shift=0.0)`: Calibrate SABR parameters for one expiry-tenor bucket.
- `build_sabr_surface(normalized_quotes, curve_state, beta_policy=0.5, min_quotes_per_bucket=3)`: Calibrate a bucketed SABR surface from normalized quotes.

### Option Models and Option Risk

- `bachelier_call(F, K, T, sigma_n, df=1.0)`: Bachelier (normal) call price.
- `bachelier_put(F, K, T, sigma_n, df=1.0)`: Bachelier (normal) put price.
- `black76_call(F, K, T, sigma_b, df=1.0)`: Black'76 call price.
- `black76_put(F, K, T, sigma_b, df=1.0)`: Black'76 put price.
- `shifted_black_call(F, K, T, sigma_b, shift, df=1.0)`: Shifted-Black call price for shifted-rate settings.
- `shifted_black_put(F, K, T, sigma_b, shift, df=1.0)`: Shifted-Black put price for shifted-rate settings.
- `bachelier_greeks(F, K, T, sigma_n, df=1.0, is_call=True)`: Bachelier Greeks helper returning delta/gamma/vega-style outputs.
- `black76_greeks(F, K, T, sigma_b, df=1.0, is_call=True)`: Black'76 Greeks helper returning delta/gamma/vega-style outputs.
- `CapletPricer(discount_curve, projection_curve=None)`: Caplet and floorlet pricing engine.
  - `forward_rate(start, end)`: Compute the forward rate over an accrual period.
  - `price(F, K, T, df, vol, vol_type='NORMAL', notional=1.0, delta_t=0.25, is_cap=True, shift=0.0)`: Price a caplet or floorlet from model inputs.
  - `greeks(F, K, T, df, vol, vol_type='NORMAL', notional=1.0, delta_t=0.25, is_cap=True)`: Compute caplet or floorlet Greeks.
  - `price_from_dates(start_date, end_date, K, vol, vol_type='NORMAL', notional=1.0, is_cap=True, shift=0.0)`: Derive the forward from curves and price from dates.
  - `price_with_sabr(start_date, end_date, K, sabr_params, vol_type='NORMAL', notional=1.0, is_cap=True)`: Price using SABR-implied volatility.
- `SwaptionPricer(discount_curve, projection_curve=None, fixed_freq=2, float_freq=4)`: Swaption pricing engine.
  - `forward_swap_rate(expiry, tenor)`: Compute the forward swap rate and annuity.
  - `price(S, K, T, annuity, vol, vol_type='NORMAL', payer_receiver='PAYER', notional=1.0, shift=0.0)`: Price a payer or receiver swaption.
  - `greeks(S, K, T, annuity, vol, vol_type='NORMAL', payer_receiver='PAYER', notional=1.0)`: Compute swaption Greeks.
  - `price_from_tenor(expiry_tenor, swap_tenor, K, vol, vol_type='NORMAL', payer_receiver='PAYER', notional=1.0, shift=0.0)`: Price directly from tenor labels.
  - `price_with_sabr(expiry_tenor, swap_tenor, K, sabr_params, vol_type='NORMAL', payer_receiver='PAYER', notional=1.0)`: Price using SABR-implied volatility.
  - `par_vol(expiry_tenor, swap_tenor, market_price, vol_type='NORMAL', payer_receiver='PAYER', notional=1.0)`: Solve for implied volatility from market price.
- `SabrOptionRisk(vol_type='NORMAL')`: SABR-aware option-risk engine with smile-consistent Greeks and decomposition utilities.
  - `risk_report(F, K, T, sabr_params, annuity=1.0, is_call=True, notional=1.0)`: Generate a comprehensive SABR option-risk report.
  - `delta_decomposition(F, K, T, sabr_params, annuity=1.0, is_call=True)`: Decompose delta into backbone and sideways components.
  - `smile_risk_ladder(F, strikes, T, sabr_params, annuity=1.0, is_call=True, notional=1.0)`: Build a risk ladder across strikes.
  - `parameter_sensitivities(F, K, T, sabr_params, annuity=1.0, is_call=True, notional=1.0)`: Compute sensitivities to SABR parameters.

### Market State and Trade Dispatch

- `CurveState(discount_curve, projection_curve=None, metadata)`: Dataclass bundling discount and projection curves.
  - `forward_rate(t_start, t_end)`: Convenience helper that uses the projection curve to compute forwards.
  - `copy(discount_curve=None, projection_curve=None)`: Return a shallow copy with optional curve overrides.
- `MarketState(curve, sabr_surface=None, asof=auto_timestamp)`: Dataclass bundling curve state and optional SABR surface for pricing and risk workflows.
  - `get_sabr_params(expiry, tenor, allow_fallback=True)`: Look up SABR parameters for an expiry-tenor bucket.
  - `copy(curve=None, sabr_surface=None, asof=None)`: Return a shallow copy with optional overrides.
- `price_trade(trade, market_state)`: Root dispatcher that prices one trade dictionary and returns `PricerOutput`.
- `risk_trade(trade, market_state, method='bump')`: Root dispatcher that computes risk for one trade, including SABR-aware option risk when applicable.
- `PricerOutput(instrument_type, pv, details)`: Dataclass returned by `price_trade`.
  - `to_dict()`: Return pricing output as a flat dictionary.

### Portfolio Builders and Diagnostics

- `build_bond_trade(pos, valuation_date)`: Convert a bond/UST position row into the normalized trade schema expected by dispatchers.
- `build_swap_trade(pos, valuation_date)`: Convert a swap/IRS position row into the normalized trade schema.
- `build_swaption_trade(pos, valuation_date)`: Convert a swaption position row into the normalized trade schema. Requires explicit option fields.
- `build_caplet_trade(pos, valuation_date)`: Convert a caplet/floor position row into the normalized trade schema. Requires explicit caplet dates.
- `build_trade_from_position(pos, valuation_date, allow_legacy_options=False)`: Dispatch a generic position row to the appropriate builder.
- `price_portfolio_with_diagnostics(positions_df, market_state, valuation_date, include_options=True, allow_legacy_options=False)`: Price a portfolio while tracking coverage, failures, and successful trade-level outputs.
- `PortfolioPricingResult(total_pv, successful_trades, successful_pvs, failed_trades, total_positions)`: Dataclass containing portfolio pricing outputs and diagnostics.
  - `successful_count`: Property returning the number of successfully priced trades.
  - `failed_count`: Property returning the number of failed trades.
  - `coverage_ratio`: Property returning successful coverage as a fraction of total positions.
  - `has_failures`: Property indicating whether any trade failed.
  - `is_complete`: Property indicating whether all positions were priced successfully.
  - `get_warnings()`: Return warning strings summarizing failures and coverage gaps.
  - `to_dict()`: Convert diagnostics and totals to a dictionary.
- `TradeFailure(position_id, instrument_type, error_type, error_message, stage)`: Dataclass recording one failed build or pricing attempt.
  - `to_dict()`: Convert the failure record to a dictionary.
- `SIGN_LONG`: Builder sign constant equal to `+1.0`.
- `SIGN_SHORT`: Builder sign constant equal to `-1.0`.
- `PositionValidationError(position_id, message)`: Base exception for invalid position rows.
- `MissingFieldError(position_id, field_name, instrument_type)`: Exception raised when a required position field is missing.
- `InvalidOptionError(position_id, message)`: Exception raised for ambiguous or invalid option specifications.

### Additional Public Submodule Exports

Symbols in this appendix are public and importable from submodules such as `rateslib.curves` or `rateslib.var`, but they are not re-exported from the `rateslib` package root.

#### `rateslib.curves`

- `bootstrap_from_quotes(anchor_date, quotes, interpolation='cubic_spline')`: Convenience helper that bootstraps a curve from quote dictionaries.
- `create_flat_curve(anchor_date, rate, max_tenor_years=30.0, currency='USD')`: Create a flat yield curve.
- `BootstrapResult`: Result container returned by bootstrap workflows, including the bootstrapped curve and diagnostics.
- `Interpolator`: Abstract base class for curve interpolation.
  - `fit(self, times, values)`: Fit the interpolator to node data.
  - `interpolate(self, t)`: Interpolate at a single maturity.
  - `derivative(self, t)`: Return the first derivative.
- `CurveInstrument`: Abstract base class for bootstrap instruments.
  - `maturity_date(self, anchor)`: Return the maturity date from an anchor date.
  - `maturity_time(self, anchor)`: Return time to maturity in years.
  - `implied_discount_factor(self, anchor, prior_dfs)`: Solve for the implied discount factor from the quote.
- `Deposit`: Money-market deposit bootstrap instrument.
  - `maturity_date(self, anchor)`: Return the deposit maturity date.
  - `maturity_time(self, anchor)`: Return the deposit maturity in years.
  - `implied_discount_factor(self, anchor, prior_dfs)`: Solve for the implied discount factor from the deposit quote.
- `OISSwap`: Overnight index swap bootstrap instrument.
  - `maturity_date(self, anchor)`: Return the swap maturity date.
  - `maturity_time(self, anchor)`: Return the swap maturity in years.
  - `implied_discount_factor(self, anchor, prior_dfs)`: Solve for the implied discount factor from the swap quote.
- `FRA`: Forward-rate-agreement bootstrap instrument.
  - `start_date(self, anchor)`: Return the start of the FRA accrual period.
  - `start_time(self, anchor)`: Return the start time in years.
  - `maturity_date(self, anchor)`: Return the end date of the FRA accrual period.
  - `maturity_time(self, anchor)`: Return the maturity in years.
  - `implied_discount_factor(self, anchor, prior_dfs)`: Solve for the forward-end discount factor.
- `Future`: Interest-rate future bootstrap instrument.
  - `implied_rate(self)`: Convert the futures price quote to an implied rate.
  - `maturity_date(self, anchor)`: Return the futures expiry or settlement date.
  - `maturity_time(self, anchor)`: Return the maturity in years.
  - `implied_discount_factor(self, anchor, prior_dfs)`: Solve for the implied discount factor from the futures quote.

#### `rateslib.pricers`

- `BondCashflows`: Complete set of bond cashflows.
  - `total_coupons`: Property returning the sum of coupon payments.
  - `final_payment`: Property returning the final principal plus coupon payment.
- `SwapCashflows`: Complete set of fixed and floating leg cashflows.
  - `pv_fixed`: Property returning fixed-leg PV.
  - `pv_floating`: Property returning floating-leg PV.
  - `net_pv`: Property returning net PV.
- `price_zero_coupon_bond(curve, settlement, maturity, face_value=100.0, day_count=DayCount.ACT_ACT)`: Price a zero-coupon bond.
- `price_coupon_bond(curve, settlement, maturity, coupon_rate, face_value=100.0, frequency=2, day_count=DayCount.ACT_ACT)`: Price a coupon bond.
- `compute_accrued_interest(settlement, maturity, coupon_rate, face_value=100.0, frequency=2, day_count=DayCount.ACT_ACT)`: Compute accrued bond interest.
- `price_vanilla_swap(discount_curve, effective, maturity, notional, fixed_rate, pay_receive='PAY', projection_curve=None)`: Price a vanilla fixed-float swap.
- `compute_swap_par_rate(curve, effective, maturity)`: Compute a par swap rate from one curve.
- `price_rate_future(curve, expiry, underlying_tenor='3M')`: Price a rate future and return the implied rate and futures price.

#### `rateslib.risk`

- `BumpType`: Enum describing bump styles used by bumping utilities.
- `BumpResult`: Dataclass describing the result of a bump operation.
- `KeyRateDV01`: Dataclass holding key-rate DV01 results.
  - `to_array()`: Return the ladder as a NumPy array in tenor order.
  - `to_dict()`: Return the ladder as a dictionary.
- `STANDARD_KEY_RATE_TENORS`: Full standard key-rate tenor ladder from `3M` through `30Y`.
- `DEFAULT_KEYRATE_TENORS`: Default reporting subset of key-rate tenors.
- `EXTENDED_KEYRATE_TENORS`: Extended reporting tenor ladder used by reporting helpers.
- `CurveRiskMetrics`: Result container for portfolio curve risk with failure tracking.
  - `coverage_ratio`: Property returning successful pricing coverage.
  - `has_failures`: Property indicating whether any position failed to build or price.
  - `to_dataframe()`: Return the key-rate ladder as a DataFrame.
  - `to_dict()`: Return the metrics as a dictionary.
- `VaRCoverageInfo`: Dataclass describing portfolio coverage for VaR setup.
  - `to_dict()`: Return coverage details as a dictionary.
- `compute_curve_risk_metrics(positions_df, market_state, valuation_date, keyrate_tenors=None, bump_bp=1.0, use_explicit_builders=False)`: Compute portfolio DV01 and key-rate DV01 using bump-and-reprice.
- `build_var_portfolio_pricer(positions_df, valuation_date, market_state, include_options=True)`: Build a portfolio pricer function suitable for VaR engines and return accompanying coverage information.

#### `rateslib.var`

- `HistoricalVaRResult`: Dataclass holding historical simulation VaR results.
  - `to_dict()`: Convert the result to a dictionary.
- `MonteCarloResult`: Dataclass holding Monte Carlo VaR results.
  - `to_dict()`: Convert the result to a dictionary.
- `StressResult`: Dataclass holding stressed VaR results.
  - `to_dict()`: Convert the result to a dictionary.
- `Scenario`: Dataclass describing a curve scenario.
  - `get_parallel_equivalent()`: Return the average parallel-equivalent bump.
- `SabrShock`: Dataclass describing additive and scaling shocks to SABR parameters.
- `ScenarioResult`: Dataclass holding the result of one scenario run.
  - `to_dict()`: Convert the result to a dictionary.
- `PortfolioScenarioResult`: Dataclass holding a full portfolio scenario run with diagnostics.
  - `coverage_ratio`: Property returning successful pricing coverage.
  - `has_failures`: Property indicating whether any positions failed.
  - `to_dict()`: Convert the result to a dictionary.
- `SABR_STRESS_REGIMES`: Named combined curve-plus-SABR stress regimes such as `CALM`, `RISK_OFF`, and `CRISIS`.
- `create_custom_scenario(name, description, **tenor_bumps)`: Create a curve scenario from ad hoc tenor bumps.
- `apply_market_scenario(market_state, curve_bump_profile=None, sabr_shocks=None)`: Apply combined curve and SABR shocks to a `MarketState`.
- `apply_named_market_regime(market_state, regime)`: Apply one of the named market regimes to a `MarketState`.
- `run_single_scenario(positions_df, market_state, valuation_date, scenario, use_explicit_builders=False)`: Run one full portfolio scenario.
- `run_scenario_set(positions_df, market_state, valuation_date, scenarios=None, use_explicit_builders=False)`: Run a set of full portfolio scenarios.
- `scenarios_to_dataframe(results)`: Convert scenario results to a display-ready DataFrame.
- `compute_historical_var(base_curve, historical_data, pricer_func, confidence=0.95, lookback_days=None)`: Convenience helper for historical VaR.
- `compute_historical_es(base_curve, historical_data, pricer_func, confidence=0.95, lookback_days=None)`: Convenience helper for historical expected shortfall.
- `compute_mc_var(base_curve, historical_data, pricer_func, confidence=0.95, num_paths=10000, seed=42)`: Convenience helper for Monte Carlo VaR.

#### `rateslib.pnl`

- `PnLComponents`: Dataclass holding carry, rolldown, curve-move, convexity, and residual attribution pieces.
  - `carry_rolldown`: Property returning combined carry plus rolldown.
  - `curve_move_total`: Property returning the total curve-move contribution.
  - `predicted_total`: Property returning predicted P&L before residual.
  - `realized_total`: Property returning realized P&L including residual.
  - `to_dict()`: Convert the component breakdown to a dictionary.
- `compute_daily_pnl(curve_t0, curve_t1, date_t0, date_t1, pricer_func, instrument_id='PORTFOLIO', dv01=None, key_rate_dv01=None, convexity=None)`: Convenience helper for daily P&L attribution.
- `compute_carry_rolldown(curve, pricer_func, date_t0, holding_days=1)`: Compute carry and rolldown over a holding period.

#### `rateslib.reporting`

- `generate_risk_summary(portfolio_pv, total_dv01, total_convexity, var_95, var_99, es_95=None, es_99=None)`: Generate summary risk statistics as a dictionary.
- `generate_position_report(positions, include_greeks=True)`: Generate a position-level report as a DataFrame.
- `generate_var_report(var_results, scenarios=None, key_rate_contributions=None)`: Generate a VaR report DataFrame with optional scenario and key-rate content.

#### `rateslib.liquidity`

- `LiquidityParameters`: Dataclass controlling liquidity-adjustment assumptions such as spreads, holding period, position sizes, ADV, and stress multiplier.
- `DEFAULT_SPREADS`: Default bid/ask spread assumptions by instrument identifier or type.
- `estimate_liquidation_time(position_notional, adv, max_participation_rate=0.25)`: Estimate days needed to liquidate a position.
- `scale_var_to_horizon(one_day_var, target_horizon, decay_factor=0.94)`: Scale one-day VaR to a longer holding period using blended square-root and decay logic.

#### `rateslib.portfolio`

- `build_futures_trade(pos, valuation_date)`: Convert a futures position row into the normalized trade schema expected by the dispatchers.

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

The repository currently contains **177 test functions** across curves, pricing, options, risk, VaR, reporting, and portfolio builders.

Install dev dependencies first:

```bash
python -m pip install -e .[dev]
```

Then run:

```bash
python -m pytest tests
python -m pytest tests --cov=rateslib --cov-report=html
python -m pytest tests/test_curves.py -v
python -m pytest tests/test_sabr.py tests/test_options.py -v
```

## Known Limitations and Assumptions

- The library is focused on USD rates. It does not cover credit, FX, cross-currency, or multi-curve collateral frameworks beyond the provided discount/projection split.
- The sample workflows are CSV-driven. Live market data ingestion, trade capture, and real-time connectivity are not included.
- Some dashboard panels are intentionally simplified or illustrative, especially liquidity what-if views, additive combined curve-plus-vol checks, and user-driven option attribution inputs.
- The codebase has strong unit coverage, but the review documents in this repository still recommend additional end-to-end validation of the full option workflow before calling the platform fully production-ready.
- The examples and dashboards are designed around the shipped sample data; if you bring your own data, field names and units need to match the expected schemas.

## Additional Documentation

- [`documentation.tex`](documentation.tex): technical write-up
- [`CHANGELOG.md`](CHANGELOG.md): project history
- [`IMPLEMENTATION_REVIEW.md`](IMPLEMENTATION_REVIEW.md): implementation review notes
- [`CHECKLIST_ASSESSMENT.md`](CHECKLIST_ASSESSMENT.md): checklist-style assessment
- [`dashboard/README.md`](dashboard/README.md): dashboard details

## Contributing

1. Create a feature branch.
2. Make and test your changes.
3. Update documentation when behavior changes.
4. Open a pull request with a short summary of user-facing impact.

## License

The project metadata declares an MIT license in `pyproject.toml`.

Note: this repository does not currently include a standalone `LICENSE` file, so add one before publishing or redistributing the project externally.

## Acknowledgments

This library is organized around common sell-side and buy-side desk workflows for rates analytics, with sample data and demos intended to make the implementation easy to inspect and extend.
