from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dashboard" / "app.py"


pytest.importorskip("shiny")
pytest.importorskip("shinywidgets")


def _load_dashboard_module():
    module_name = f"dashboard_app_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, APP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_shiny_dashboard_loaders_handle_missing_output(monkeypatch, tmp_path):
    dashboard_app = _load_dashboard_module()
    monkeypatch.setattr(dashboard_app, "get_output_dir", lambda: tmp_path)

    assert dashboard_app.get_latest_report_date() == "20240115"
    assert dashboard_app.load_portfolio_summary()["Total PV"] == 0
    assert dashboard_app.load_var_metrics()["Historical VaR 95%"] == 0
    assert dashboard_app.load_key_rate_dv01().empty
    assert dashboard_app.load_positions().empty
    assert dashboard_app.load_scenarios().empty


def test_shiny_dashboard_loaders_parse_populated_reports(monkeypatch, tmp_path):
    dashboard_app = _load_dashboard_module()
    monkeypatch.setattr(dashboard_app, "get_output_dir", lambda: tmp_path)

    report_date = "20240115"

    pd.DataFrame(
        [
            {
                "Total PV": 12_345_678,
                "Total DV01": 4_321,
                "Total Convexity": 98.76,
                "Number of Positions": 7,
            }
        ]
    ).to_csv(tmp_path / f"Sample_Trading_Book_{report_date}_Portfolio_Summary.csv", index=False)

    pd.DataFrame([{"Tenor": "5Y", "DV01": 123.45}]).to_csv(
        tmp_path / f"Sample_Trading_Book_{report_date}_Key_Rate_DV01.csv", index=False
    )

    pd.DataFrame(
        [
            {
                "Historical VaR 95%": 11_111,
                "Historical VaR 99%": 22_222,
                "Historical ES 95%": 33_333,
                "Historical ES 99%": 44_444,
                "Monte Carlo VaR 95%": 55_555,
                "Monte Carlo VaR 99%": 66_666,
            }
        ]
    ).to_csv(tmp_path / f"Sample_Trading_Book_{report_date}_Value_at_Risk.csv", index=False)

    pd.DataFrame(
        [
            {
                "position_id": "POS001",
                "instrument": "UST_5Y",
                "type": "Bond",
                "notional": 1_000_000,
                "pv": 999_500.25,
                "dv01": 455.29,
            }
        ]
    ).to_csv(tmp_path / f"Sample_Trading_Book_{report_date}_Position_Details.csv", index=False)

    pd.DataFrame([{"Scenario": "Parallel +100bp", "P&L": -12_345}]).to_csv(
        tmp_path / f"Sample_Trading_Book_{report_date}_Scenario_Analysis.csv", index=False
    )

    assert dashboard_app.get_latest_report_date() == report_date

    summary = dashboard_app.load_portfolio_summary()
    assert summary["Total PV"] == 12_345_678
    assert summary["Number of Positions"] == 7

    var_metrics = dashboard_app.load_var_metrics()
    assert var_metrics["Historical VaR 95%"] == 11_111
    assert var_metrics["Monte Carlo VaR 99%"] == 66_666

    key_rate = dashboard_app.load_key_rate_dv01()
    assert key_rate.to_dict("records") == [{"Tenor": "5Y", "DV01": 123.45}]

    positions = dashboard_app.load_positions()
    assert positions.loc[0, "position_id"] == "POS001"
    assert positions.loc[0, "pv"] == 999_500.25

    scenarios = dashboard_app.load_scenarios()
    assert scenarios.loc[0, "Scenario"] == "Parallel +100bp"
    assert scenarios.loc[0, "P&L"] == -12_345
