from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "dashboard" / "interactive_dashboard.py"


streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _find_widget(collection, label: str):
    for widget in collection:
        if getattr(widget, "label", None) == label:
            return widget
    raise AssertionError(f"Widget not found: {label}")


def _find_button_by_key(at, key: str):
    for widget in at.button:
        if getattr(widget, "key", None) == key:
            return widget
    raise AssertionError(f"Button not found: {key}")


def _assert_no_exceptions(at, context: str):
    assert not at.exception, f"{context} raised exceptions: {[exc.message for exc in at.exception]}"


def test_interactive_dashboard_control_paths_execute():
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()
    _assert_no_exceptions(at, "initial render")

    _find_widget(at.date_input, "Valuation Date").set_value(date(2024, 2, 1)).run()
    _assert_no_exceptions(at, "valuation date change")
    _find_widget(at.date_input, "Valuation Date").set_value(date(2024, 1, 15)).run()
    _assert_no_exceptions(at, "valuation date reset")

    _find_widget(at.checkbox, "Show All Buckets").set_value(False).run()
    _assert_no_exceptions(at, "single SABR bucket view")
    _find_widget(at.selectbox, "Select Bucket").select_index(1).run()
    _assert_no_exceptions(at, "alternate SABR bucket selection")
    _find_widget(at.checkbox, "Show All Buckets").set_value(True).run()
    _assert_no_exceptions(at, "all SABR buckets view")

    _find_widget(at.selectbox, "Select Instrument Type").set_value("Bond").run()
    _find_widget(at.selectbox, "Payment Frequency").set_value("12").run()
    _find_widget(at.selectbox, "Day Count Convention").set_value("30/360").run()
    _find_widget(at.button, "Price Bond").click().run()
    _assert_no_exceptions(at, "bond pricing")

    _find_widget(at.selectbox, "Select Instrument Type").set_value("Swap").run()
    _find_widget(at.selectbox, "Direction").set_value("RECEIVE").run()
    _find_widget(at.button, "Price Swap").click().run()
    _assert_no_exceptions(at, "swap pricing")

    _find_widget(at.selectbox, "Select Instrument Type").set_value("Futures").run()
    _find_widget(at.number_input, "Number of Contracts").set_value(25).run()
    _find_widget(at.button, "Price Futures").click().run()
    _assert_no_exceptions(at, "futures pricing")

    _find_widget(at.slider, "Strike Offset (bp)").set_value(-200).run()
    _assert_no_exceptions(at, "option strike low")
    _find_widget(at.slider, "Strike Offset (bp)").set_value(200).run()
    _assert_no_exceptions(at, "option strike high")
    _find_widget(at.slider, "Strike Offset (bp)").set_value(0).run()
    _assert_no_exceptions(at, "option strike reset")

    _find_widget(at.checkbox, "Include options in VaR (slower, more accurate)").set_value(False).run()
    _assert_no_exceptions(at, "VaR linear-only")
    for method in ["Historical Simulation", "Monte Carlo", "Stressed VaR"]:
        _find_widget(at.selectbox, "VaR Method").set_value(method).run()
        _assert_no_exceptions(at, f"VaR method {method}")
    _find_widget(at.selectbox, "Stress Period").set_value("FULL_2020_2022").run()
    _assert_no_exceptions(at, "alternate stress period")
    _find_widget(at.checkbox, "Include options in VaR (slower, more accurate)").set_value(True).run()
    _find_widget(at.selectbox, "VaR Method").set_value("Historical Simulation").run()
    _assert_no_exceptions(at, "VaR with options")

    _find_widget(at.slider, "Parallel Shift (bp)").set_value(50).run()
    _find_widget(at.slider, "Twist Magnitude (bp)").set_value(-20).run()
    _find_widget(at.slider, "2s10s Steepening (bp)").set_value(30).run()
    _find_widget(at.selectbox, "Severity").set_value("Extreme (3x)").run()
    _find_widget(at.selectbox, "Twist Pivot").set_value("10Y").run()
    _find_widget(at.slider, "β₀ (Level)").set_value(0.05).run()
    _find_widget(at.slider, "λ₁ (Decay 1)").set_value(2.0).run()
    _find_widget(at.button, "Reset to Fitted Values").click().run()
    _assert_no_exceptions(at, "NSS reset")

    _find_widget(at.slider, "σ_ATM Scale").set_value(1.2).run()
    _find_widget(at.slider, "ν Scale").set_value(1.5).run()
    _find_widget(at.slider, "ρ Shift").set_value(0.2).run()
    _find_widget(at.selectbox, "Select Bucket to Visualize").select_index(1).run()
    _assert_no_exceptions(at, "SABR stressed controls")
    _find_button_by_key(at, "run_custom_btn").click().run()
    _assert_no_exceptions(at, "custom scenario execution")
    _find_button_by_key(at, "reset_custom_btn").click().run()
    _assert_no_exceptions(at, "custom scenario reset")

    _find_widget(at.number_input, "Rate Move (bp)").set_value(-15).run()
    _find_widget(at.number_input, "Vol Move (bp)").set_value(5).run()
    _find_widget(at.number_input, "Days Passed").set_value(3).run()
    _assert_no_exceptions(at, "P&L attribution inputs")

    _find_widget(at.number_input, "Base VaR (1-day)").set_value(25_000).run()
    _find_widget(at.slider, "Holding Period (days)").set_value(5).run()
    _find_widget(at.number_input, "Avg Bid/Ask Spread (bp)").set_value(4.5).run()
    _find_widget(at.checkbox, "Apply Stress Multiplier").set_value(True).run()
    _find_widget(at.button, "Calculate LVaR").click().run()
    _assert_no_exceptions(at, "LVaR stressed")
    _find_widget(at.checkbox, "Apply Stress Multiplier").set_value(False).run()
    _find_widget(at.button, "Calculate LVaR").click().run()
    _assert_no_exceptions(at, "LVaR unstressed")

    for data_view in ["Market Quotes", "Portfolio Positions", "Historical Rates", "Curve Nodes"]:
        _find_widget(at.selectbox, "Select Data View").set_value(data_view).run()
        _assert_no_exceptions(at, f"data explorer {data_view}")
        if data_view == "Portfolio Positions":
            download_buttons = at.get("download_button")
            assert any(
                getattr(widget, "label", None) == "Download Positions as CSV" for widget in download_buttons
            )
