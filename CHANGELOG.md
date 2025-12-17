# Changelog

All notable changes to the Rates Risk Library are documented in this file.

## [Current Version] - December 2024

### Major Features Added

#### SABR Volatility Model Integration
- **SABR Model** (`src/rateslib/vol/sabr.py`)
  - Full implementation of Hagan's SABR formulas
  - Support for both Normal (basis point) and Lognormal (Black) volatility conventions
  - Shifted SABR for negative rates
  - Parameter sensitivities (d\u03c3/dF, d\u03c3/d\u03c1, d\u03c3/d\u03bd)

- **Volatility Surface Management** (`src/rateslib/vol/sabr_surface.py`)
  - Multi-dimensional surface by expiry and tenor buckets
  - Flexible bucketing strategies (nearest neighbor, interpolation)
  - ATM vol lookup and smile generation

- **SABR Calibration** (`src/rateslib/vol/calibration.py`)
  - Market calibration from vol quotes
  - Constraint handling (correlation bounds, vol positivity)
  - Robust optimization

- **Volatility Quotes** (`src/rateslib/vol/quotes.py`)
  - Support for ATM, RR (risk reversal), BF (butterfly) quote formats
  - Strike conversion utilities
  - Market data loading from CSV

#### Options Pricing
- **Swaptions** (`src/rateslib/options/swaption.py`)
  - European swaption pricing with SABR
  - Analytical Greeks (delta, vega)
  - Payer/Receiver convention support
  - Integration with market state

- **Caplets/Floors** (`src/rateslib/options/caplet.py`)
  - Interest rate cap and floor pricing
  - SABR-based implied volatility
  - Greeks calculations
  - Multi-period caplets

- **SABR Risk** (`src/rateslib/options/sabr_risk.py`)
  - Parameter sensitivities (\u03c3_ATM, \u03bd, \u03c1)
  - Delta conventions (sticky strike, sticky delta, backbone)
  - Vega calculation with vol bumping
  - Comprehensive risk reporting

#### Futures Support
- **Enhanced Futures Pricing** (`src/rateslib/pricers/futures.py`)
  - SOFR futures contract specifications
  - Theoretical pricing vs market price
  - P&L tracking with trade price
  - Convexity adjustment
  - Expiry date validation

- **Futures in Portfolio** (`src/rateslib/var/scenarios.py`)
  - Full integration in scenario analysis
  - Proper handling of long/short positions
  - Expired contract filtering
  - Position-level P&L attribution

#### Enhanced Dashboard Features

##### SABR Visualization (Curves Tab)
- Implied volatility curves by bucket
- Volatility smile plots across strikes
- 3D heatmap of vol surface
- ATM vol levels display
- Interactive bucket selection

##### Custom Scenario Builder (Scenarios Tab)
- **NSS Parameter Tweaking**:
  - \u03b2\u2080 (level): -2.0% to +2.0%
  - \u03b2\u2081 (slope): -2.0% to +2.0%
  - \u03b2\u2082 (curvature): -2.0% to +2.0%
  - \u03b2\u2083 (2nd hump): -1.0% to +1.0%
  - \u03bb\u2081 (decay 1): 0.5 to 5.0
  - \u03bb\u2082 (decay 2): 0.5 to 10.0

- **SABR Parameter Stressing**:
  - \u03c3_ATM scale: -50% to +100%
  - \u03bd (vol of vol) scale: -50% to +100%
  - \u03c1 (correlation) shift: -0.3 to +0.3

- **Live Visualization**:
  - Base vs stressed yield curves
  - Real-time parameter updates
  - Interactive curve comparison

- **Run Custom Scenario**:
  - Full portfolio repricing
  - P&L attribution (curve vs vol)
  - Coverage metrics (100% positions)
  - Failed position diagnostics

##### Enhanced Pricing Calculators
- Swaption pricing with SABR
- Caplet/Floor pricing with SABR
- Greeks display (delta, vega)
- Volatility convention selection

##### P&L Attribution Enhancement
- Separate curve vs volatility attribution
- Volatility move component for options
- Enhanced residual analysis

### Bug Fixes & Improvements

#### Position Data Handling
- **100% Position Coverage Achievement**
  - Fixed NaN/NaT handling in position loading
  - Proper parsing of option fields (expiry_date, underlying_swap_tenor, strike)
  - Robust date conversion for futures expiry dates
  - CAPLET-specific tenor derivation from caplet_start_date and caplet_end_date

- **Enhanced `_build_trade_from_position_legacy()`**:
  - Added `get_first_valid()` helper to properly check for NaN values
  - Fixed pandas NaN handling (NaN is truthy but should be treated as None)
  - Explicit date parsing with pd.isna() checks
  - Fallback logic for missing optional fields

- **Sample Data Updates** (`data/sample_book/positions.csv`):
  - Added expiry_date for all FUT positions
  - Proper option fields for SWAPTION (expiry_date, underlying_swap_tenor, strike)
  - Complete CAPLET fields (caplet_start_date, caplet_end_date, position, strike)

#### Code Quality
- **Deprecation Warnings Fixed**:
  - Replaced `use_container_width=True` with `width="stretch"` (4 instances)
  - Updated to Streamlit 1.28+ API conventions

- **Robust Error Handling**:
  - Never silently swallow exceptions in scenario repricing
  - Detailed failure tracking with position_id and error messages
  - Coverage metrics exposed to UI for transparency

- **Date/NaT Handling**:
  - Added pd.isna() checks before date comparisons
  - Fixed "Cannot compare NaT with datetime.date object" errors
  - Proper None returns instead of NaT propagation

### Testing
- **177 Comprehensive Tests** (all passing)
  - test_sabr.py: SABR model, calibration, vol calculations
  - test_options.py: Swaption and caplet pricing
  - test_sabr_risk_conventions.py: Greeks, parameter sensitivities
  - test_var.py: Enhanced with FUT scenario tests
  - test_curves.py, test_pricers.py, test_risk.py: Existing coverage maintained

### Documentation Updates
- Updated README.md with SABR/options features
- Enhanced dashboard/README.md with new capabilities
- Updated dashboard/FEATURES.md with complete coverage summary
- Refreshed dashboard/QUICKSTART.md with new features
- Added this CHANGELOG.md

### Performance & Architecture
- Unified pricing dispatcher for all instrument types
- MarketState abstraction for curve + vol surface
- Efficient SABR surface lookups with caching
- Lazy evaluation of charts (only computed when tab active)

### Data Files
- Added vol_quotes.csv with sample SABR parameters
- Enhanced positions.csv with complete option fields
- All 12 positions now price successfully

## API Changes

### New Exports
```python
from rateslib import (
    # SABR/Vol
    SabrModel, SabrParams, SabrSurface, build_sabr_surface,
    VolQuote, normalize_vol_quotes,
    
    # Options
    SwaptionPricer, CapletPricer,
    
    # Market State
    MarketState, CurveState, SabrSurfaceState,
    
    # Pricers
    price_trade,  # Unified dispatcher
    
    # Scenarios
    run_scenario_set, apply_market_scenario, SabrShock,
)
```

### Breaking Changes
None - all changes are additive and backward compatible.

## Migration Guide

### For Existing Users

If you were using the library before SABR integration:

1. **Update dependencies**:
   ```bash
   pip install -e .  # Reinstall with updated requirements
   ```

2. **Optional: Use new SABR features**:
   ```python
   # Build vol surface
   from rateslib import build_sabr_surface, normalize_vol_quotes
   import pandas as pd
   
   vol_quotes_df = pd.read_csv('data/vol_quotes.csv')
   vol_quotes = normalize_vol_quotes(vol_quotes_df, valuation_date)
   sabr_surface = build_sabr_surface(vol_quotes, valuation_date)
   
   # Create market state
   from rateslib import MarketState, CurveState
   
   market_state = MarketState(
       curve=CurveState(discount_curve=curve, projection_curve=curve),
       sabr_surface=sabr_surface,
       asof=valuation_date
   )
   
   # Price swaption
   from rateslib import price_trade
   
   swaption_trade = {
       "instrument_type": "SWAPTION",
       "expiry_tenor": "1Y",
       "swap_tenor": "5Y",
       "strike": "ATM",
       "payer_receiver": "PAYER",
       "notional": 10_000_000,
       "vol_type": "NORMAL"
   }
   
   result = price_trade(swaption_trade, market_state)
   print(f"Swaption PV: {result.pv:,.2f}")
   print(f"Implied Vol: {result.details['implied_vol']:.4%}")
   ```

3. **Existing code continues to work** - all previous APIs unchanged

## Known Issues & Limitations

None currently identified. All 177 tests passing, 100% position coverage achieved.

## Future Roadmap

Potential enhancements under consideration:
- American options (Bermudan swaptions)
- Exotic options (barriers, digitals)
- CMS (constant maturity swap) pricing
- Real-time data feed integration
- Multi-currency support
- Historical P&L tracking over time
- Enhanced reporting templates

## Contributors

This library is actively maintained and developed for production trading desk use.

---

**For detailed API documentation, see the main README.md**
