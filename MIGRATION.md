# Package rename: `rateslib` to `rates_risk`

Version 0.1.0 uses the import namespace `rates_risk` and distribution name
`rates-risk-lib`.

The previous local package name, `rateslib`, conflicted with an unrelated and
established fixed-income project on PyPI. This repository intentionally does
not provide a `rateslib` compatibility shim because doing so would preserve the
ambiguous import.

Update imports as follows:

```python
# Before
from rateslib.curves import Curve

# After
from rates_risk.curves import Curve
```

Reinstall an existing editable environment after pulling the rename:

```bash
python -m pip uninstall -y rateslib
python -m pip install -e ".[dev]"
```

If the external PyPI `rateslib` package is also required, install it normally;
the two projects now have distinct distribution and import names.
