# Model validation status

**Status:** experimental quantitative-finance library / research prototype
**Version reviewed:** 0.1.0
**Last updated:** 2026-07-12

This document is the single status record for numerical validation in this
repository. It replaces earlier implementation checklists that used
"production-ready" language without independent model-validation evidence.

## What the automated suite currently establishes

- Curve, convention, schedule, pricing, option, SABR, portfolio-builder, risk,
  reporting, VaR, and dashboard code paths execute on the bundled examples.
- The suite includes analytical identities, sign and ordering checks, portfolio
  failure/coverage behavior, and dashboard-control smoke tests.
- Curve ingestion rejects non-finite quotes, missing/unknown instrument types,
  duplicate quote identities, non-finite derived nodes, and non-finite
  repricing errors.
- Negative continuously compounded rates are representable through positive
  discount factors above one.

These checks are regression evidence. They are not independent approval of the
models or evidence that the library is suitable for financial production use.

## Validated versus experimental inventory

| Area | Current evidence | Status |
|---|---|---|
| Day-count and basic date utilities | Unit tests on supported conventions | Regression-tested |
| Deposit/OIS/FRA/futures bootstrap | Repricing and input-validation tests on bundled synthetic cases | Experimental; independent market benchmark required |
| Bonds, swaps, and futures | Analytical/sign/range tests | Experimental; QuantLib or equivalent reconciliation required |
| Normal/lognormal option pricing | Formula identities and finite-difference-style checks | Experimental; independent golden cases required |
| SABR calibration and risk | Synthetic recovery/sensitivity tests | Experimental; surface-arbitrage and convention validation required |
| DV01 and key-rate risk | Bump behavior and portfolio tests | Experimental; quote-rebootstrap reconciliation required |
| Historical/Monte Carlo VaR and ES | Synthetic distribution tests | Experimental; data governance and backtesting required |
| P&L attribution and liquidity | Bundled workflow checks | Illustrative |
| Dashboards and reports | UI/load/export smoke tests | Presentation layer only |

## Open validation work

1. Reconcile curve nodes, par instruments, PV, DV01, and key-rate risk to an
   independent implementation under identical conventions.
2. Add dated market fixtures with source, units, as-of timestamp, and immutable
   expected results.
3. Validate schedules, settlement, accrued interest, stubs, calendars, and
   negative-rate behavior across adverse inputs.
4. Benchmark SABR normal/lognormal conventions, calibration residuals, boundary
   behavior, Greeks, and static-arbitrage diagnostics.
5. Add VaR input-quality controls, horizon mapping, covariance diagnostics,
   exception backtesting, and stress-period provenance.
6. Reconcile daily P&L explain to a golden portfolio including option cross
   effects and an explicitly interpreted residual.
7. Measure performance, numerical stability, test coverage, and behavior across
   the supported Python/OS matrix in CI.

## Reproduction

```bash
python -m pip install -e ".[dev,all]"
python -m pytest
python -m pytest --cov=rates_risk --cov-report=term-missing
```

The CI workflow is the canonical clean-environment check. Any benchmark output
used in documentation should record the Git revision, input hashes, dependency
versions, command, and tolerance.

## Change-control rule

Changes to curve construction, pricing formulas, risk definitions, scenario
application, VaR statistics, or P&L attribution must include:

1. a failing regression or benchmark before the fix;
2. a documented formula/convention and units;
3. numerical tolerance rationale;
4. an entry in `CHANGELOG.md`; and
5. an update to this inventory when validation status changes.
