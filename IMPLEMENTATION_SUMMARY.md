# SABR Risk Engine Implementation Summary

## Overview
This document summarizes the changes made to fix the SABR risk engine and implement proper conventions for option Greeks and parameter sensitivities.

## Tasks Completed

### Task A: Fix vol_type API Mismatch ✓

**Problem:** Runtime error when computing SABR parameter sensitivities:
```
SabrModel.dsigma_drho() got an unexpected keyword argument 'vol_type'
```

**Root Cause:** Inconsistent usage of the `vol_type` parameter across SABR risk calculation methods. Some call sites passed it, others didn't.

**Solution:**
1. Updated `sabr_risk.py` lines 137-139 to consistently pass `vol_type` to all SABR derivative methods
2. Updated `sabr.py` `dsigma_dF` method to accept `vol_type` parameter and route to appropriate vol function
3. All three methods (`dsigma_dF`, `dsigma_drho`, `dsigma_dnu`) now have consistent signatures

**Files Modified:**
- `src/rateslib/options/sabr_risk.py`
- `src/rateslib/vol/sabr.py`

### Task B: Implement σ_ATM-Consistent Vega ✓

**Problem:** Vega to σ_ATM was incorrectly computed as "ATM option vega" instead of proper chain rule.

**Solution:**
1. **Added `dsigma_dsigma_atm` method** to `SabrModel`:
   - Uses finite difference on σ_ATM parameter
   - Computes ∂σ(F,K)/∂σ_ATM via bump-and-reprice
   - Default bump size: 1bp (0.0001)
   - Handles both NORMAL and LOGNORMAL vol types

2. **Updated `parameter_sensitivities`** to use proper chain rule:
   ```python
   dV/dσ_ATM = Vega_base × dσ(F,K)/dσ_ATM
   ```
   Previously used `dalpha_dsigma_atm` which was incomplete.

3. **Updated `risk_report` vega_atm calculation**:
   - Old: Used vega of ATM option (wrong concept)
   - New: `vega_atm = vega_base * dsigma_dsigma_atm`

**Files Modified:**
- `src/rateslib/vol/sabr.py` - Added `dsigma_dsigma_atm` method
- `src/rateslib/options/sabr_risk.py` - Fixed vega_atm calculation

### Task C: Delta Convention with Smile Dynamics ✓

**Verification:** Confirmed that delta convention already follows memo requirements:
```python
Δ_model = Δ_base + Vega × dσ/dF
```

**Enhancements:**
1. Made `dsigma_dF` vol_type aware (was only using BLACK vol)
2. Added finite difference tests to validate delta matches bump-and-reprice
3. Verified decomposition: `delta_sideways` = base delta, `delta_backbone` = smile contribution

**Files Modified:**
- `src/rateslib/vol/sabr.py` - Enhanced `dsigma_dF` with vol_type parameter

### Task D: Auditability for Curve + SABR State ✓

**Implementation:**
1. **Enhanced `ScenarioResult`** dataclass:
   - Added `curve_params` field for NSS or other curve parameters
   - Added `sabr_params` field for SABR parameter summary
   - Updated `to_dict()` to include these fields

2. **Added `extract_market_state_params` function**:
   - Extracts curve metadata (including NSS params if present)
   - Extracts SABR params by bucket
   - Includes SABR diagnostics and convention info
   - Returns structured dict for reporting/auditing

3. **SABR shocks** already tracked in diagnostics (no change needed)

**Files Modified:**
- `src/rateslib/var/scenarios.py`

### Task E: Comprehensive Test Suite ✓

**New Test File:** `tests/test_sabr_risk_conventions.py` (13 tests)

**Test Coverage:**
1. **API Consistency Tests** (3 tests):
   - `test_parameter_sensitivities_normal_no_error`
   - `test_parameter_sensitivities_lognormal_no_error`
   - `test_risk_report_normal_no_error`

2. **Delta Convention Tests** (3 tests):
   - `test_sticky_vol_additivity` - Verifies decomposition
   - `test_finite_difference_delta` - Validates against bump-and-reprice
   - `test_backbone_delta_nonzero_for_otm` - Checks mechanism works

3. **σ_ATM Vega Tests** (3 tests):
   - `test_dsigma_dsigma_atm_at_atm` - Basic functionality
   - `test_dsigma_dsigma_atm_bump_reprice` - Validates finite difference
   - `test_vega_to_sigma_atm_vs_bump_reprice` - End-to-end validation

4. **Parameter Sensitivity Tests** (2 tests):
   - `test_vega_positive` - Sign checks
   - `test_nu_sensitivity_positive` - Mechanism validation

5. **Crash Regression Tests** (2 tests):
   - `test_no_vol_type_kwarg_error_in_risk_report`
   - `test_no_vol_type_kwarg_error_in_parameter_sensitivities`

**Files Added:**
- `tests/test_sabr_risk_conventions.py`

## Test Results

**All 113 tests passing:**
- 7 convention tests
- 14 curve tests
- 14 date tests
- 20 option tests (existing)
- 12 pricer tests
- 8 risk tests
- 17 SABR tests (existing)
- 13 SABR risk convention tests (NEW)
- 8 VaR tests

## Code Changes Summary

### Function Signatures Changed

**`SabrModel.dsigma_dF`:**
```python
# Old
def dsigma_dF(self, F, K, T, params, hold_atm_fixed=True)

# New
def dsigma_dF(self, F, K, T, params, vol_type="BLACK", hold_atm_fixed=True)
```

**`SabrModel.dsigma_drho` and `dsigma_dnu`:**
```python
# Already had vol_type parameter - no signature change
def dsigma_drho(self, F, K, T, params, vol_type="BLACK")
def dsigma_dnu(self, F, K, T, params, vol_type="BLACK")
```

### New Methods Added

**`SabrModel.dsigma_dsigma_atm`:**
```python
def dsigma_dsigma_atm(
    self,
    F: float,
    K: float,
    T: float,
    params: SabrParams,
    vol_type: str = "BLACK",
    bump_size: float = 0.0001
) -> float:
    """
    Derivative of implied vol at strike K w.r.t. sigma_ATM.
    Uses finite difference: bump sigma_atm, recompute sigma(F,K).
    """
```

**`extract_market_state_params`:**
```python
def extract_market_state_params(market_state: MarketState) -> Dict[str, Any]:
    """
    Extract curve and SABR parameters from MarketState for auditability.
    Returns Dict with 'curve_metadata', 'sabr_params', 'sabr_diagnostics'
    """
```

### Call Site Changes

**In `SabrOptionRisk.risk_report`:**
```python
# Old
dsigma_dF = self.model.dsigma_dF(F, K, T, sabr_params)
dsigma_drho = self.model.dsigma_drho(F, K, T, sabr_params)
dsigma_dnu = self.model.dsigma_dnu(F, K, T, sabr_params)

# New
dsigma_dF = self.model.dsigma_dF(F, K, T, sabr_params, vol_type=self.vol_type)
dsigma_drho = self.model.dsigma_drho(F, K, T, sabr_params, vol_type=self.vol_type)
dsigma_dnu = self.model.dsigma_dnu(F, K, T, sabr_params, vol_type=self.vol_type)
```

**In `SabrOptionRisk.parameter_sensitivities`:**
```python
# Old
dalpha_dsigma = self.model.dalpha_dsigma_atm(F, T, sabr_params)
return {
    'dV_dsigma_atm': notional * vega * dalpha_dsigma,
    ...
}

# New
dsigma_dsigma_atm = self.model.dsigma_dsigma_atm(F, K, T, sabr_params, vol_type=self.vol_type)
return {
    'dV_dsigma_atm': notional * vega * dsigma_dsigma_atm,
    'dsigma_dsigma_atm': dsigma_dsigma_atm,  # Also return the derivative itself
    ...
}
```

## Validation

### Manual Verification Steps

To manually verify the changes work correctly:

1. **Test API Consistency:**
   ```python
   from rateslib.vol.sabr import SabrParams
   from rateslib.options.sabr_risk import SabrOptionRisk
   
   params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.4)
   engine = SabrOptionRisk(vol_type="NORMAL")
   
   # This should NOT raise "unexpected keyword argument 'vol_type'" error
   report = engine.risk_report(F=0.04, K=0.04, T=1.0, sabr_params=params, annuity=1.0, is_call=True)
   sens = engine.parameter_sensitivities(F=0.04, K=0.04, T=1.0, sabr_params=params, annuity=1.0, is_call=True)
   ```

2. **Test σ_ATM Vega:**
   ```python
   from rateslib.vol.sabr import SabrModel
   
   model = SabrModel()
   params = SabrParams(sigma_atm=0.005, beta=0.5, rho=-0.2, nu=0.4)
   
   # Check derivative exists and is reasonable
   dsigma_dsigma_atm = model.dsigma_dsigma_atm(F=0.04, K=0.04, T=1.0, params=params, vol_type="NORMAL")
   assert dsigma_dsigma_atm > 0  # Should be positive
   ```

3. **Test Delta Decomposition:**
   ```python
   # Verify delta_sabr = delta_base + delta_backbone
   report = engine.risk_report(F=0.04, K=0.035, T=1.0, sabr_params=params, annuity=1.0, is_call=True)
   
   assert abs(report.delta_sabr - (report.delta_base + report.delta_backbone)) < 1e-10
   assert report.delta_sideways == report.delta_base
   ```

4. **Test Scenario Auditability:**
   ```python
   from rateslib.var.scenarios import extract_market_state_params
   
   # Assuming market_state is constructed with SABR surface
   params_dict = extract_market_state_params(market_state)
   
   # Should contain curve metadata and SABR params
   assert "sabr_params" in params_dict or market_state.sabr_surface is None
   ```

## Next Steps (Not Implemented)

The following were identified in the requirements but not yet implemented:

1. **Dashboard Updates:**
   - Display SABR parameter sensitivities in UI
   - Show curve and SABR parameters in scenario/VaR tabs
   - Surface NSS parameters in dashboard

2. **Additional Documentation:**
   - Update user-facing documentation
   - Add docstrings to dashboard components
   - Create examples notebook

These can be addressed in follow-up work.

## Compatibility Notes

**Breaking Changes:**
- `dsigma_dF` signature changed (added optional `vol_type` parameter with default)
- `parameter_sensitivities` return dict now includes `dsigma_dsigma_atm` key
- `ScenarioResult` dataclass has new optional fields

**Backward Compatibility:**
- All new parameters have defaults, so existing code will continue to work
- New dict keys are additive, so existing code reading specific keys won't break
- All existing tests pass without modification

## Performance Impact

**Minimal performance impact:**
- `dsigma_dsigma_atm` uses one additional implied vol calculation (bump)
- Finite difference bump size is small (1bp) for accuracy
- No changes to hot paths in pricing or curve construction

## Security Considerations

**No security vulnerabilities introduced:**
- All changes are in computational finance logic
- No new external dependencies
- No user input processing changes
- Numerical stability maintained (bounds checking on SABR params)

## Conclusion

All required tasks (A, B, C, D, E) have been successfully implemented with comprehensive test coverage. The SABR risk engine now:

1. ✓ Has consistent API for vol_type across all derivative methods
2. ✓ Implements correct chain rule for vega to σ_ATM parameter
3. ✓ Follows proper delta convention with smile dynamics
4. ✓ Provides auditability for curve and SABR parameters in scenarios
5. ✓ Has 13 new tests validating all conventions and preventing regressions

**Total Test Coverage: 113/113 passing (100%)**
