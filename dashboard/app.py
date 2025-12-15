"""
Rates Risk Monitoring Dashboard
===============================

A real-time risk monitoring interface for tracking trading book risk metrics.

Features:
- Portfolio summary with PV, DV01, and Convexity
- Key Rate DV01 exposure visualization
- VaR/ES metrics display
- Position-level details
- Scenario analysis P&L
- Risk limit monitoring

Usage:
    cd dashboard
    shiny run app.py
"""

import sys
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import numpy as np

# Shiny imports
from shiny import App, ui, render, reactive, req
from shinywidgets import output_widget, render_widget
import plotly.graph_objects as go
import plotly.express as px

# Add src to path for importing rateslib
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration
# =============================================================================

# Risk limits (configurable)
RISK_LIMITS = {
    "total_dv01": 10000,      # Max total DV01
    "var_95": 50000,          # Max VaR 95%
    "var_99": 100000,         # Max VaR 99%
    "single_position_dv01": 20000,  # Max DV01 per position
}

# Color scheme
COLORS = {
    "primary": "#1f77b4",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "info": "#17a2b8",
    "light": "#f8f9fa",
    "dark": "#343a40",
}


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_output_dir() -> Path:
    """Get the output directory path."""
    return Path(__file__).parent.parent / "output"


def get_latest_report_date() -> str:
    """Find the latest report date from output files."""
    output_dir = get_output_dir()
    files = list(output_dir.glob("*_Portfolio_Summary.csv"))
    if files:
        # Extract date from filename
        filename = files[0].stem
        parts = filename.split("_")
        # Find the date part (format: YYYYMMDD)
        for part in parts:
            if len(part) == 8 and part.isdigit():
                return part
    return "20240115"  # Default


def load_portfolio_summary() -> dict:
    """Load portfolio summary data."""
    output_dir = get_output_dir()
    report_date = get_latest_report_date()
    
    try:
        df = pd.read_csv(output_dir / f"Sample_Trading_Book_{report_date}_Portfolio_Summary.csv")
        return df.iloc[0].to_dict()
    except Exception as e:
        return {
            "Total PV": 0,
            "Total DV01": 0,
            "Total Convexity": 0,
            "Number of Positions": 0
        }


def load_key_rate_dv01() -> pd.DataFrame:
    """Load key rate DV01 data."""
    output_dir = get_output_dir()
    report_date = get_latest_report_date()
    
    try:
        return pd.read_csv(output_dir / f"Sample_Trading_Book_{report_date}_Key_Rate_DV01.csv")
    except Exception as e:
        return pd.DataFrame({"Tenor": [], "DV01": []})


def load_var_metrics() -> dict:
    """Load VaR and ES metrics."""
    output_dir = get_output_dir()
    report_date = get_latest_report_date()
    
    try:
        df = pd.read_csv(output_dir / f"Sample_Trading_Book_{report_date}_Value_at_Risk.csv")
        return df.iloc[0].to_dict()
    except Exception as e:
        return {
            "Historical VaR 95%": 0,
            "Historical VaR 99%": 0,
            "Historical ES 95%": 0,
            "Historical ES 99%": 0,
            "Monte Carlo VaR 95%": 0,
            "Monte Carlo VaR 99%": 0
        }


def load_positions() -> pd.DataFrame:
    """Load position details."""
    output_dir = get_output_dir()
    report_date = get_latest_report_date()
    
    try:
        return pd.read_csv(output_dir / f"Sample_Trading_Book_{report_date}_Position_Details.csv")
    except Exception as e:
        return pd.DataFrame()


def load_scenarios() -> pd.DataFrame:
    """Load scenario analysis results."""
    output_dir = get_output_dir()
    report_date = get_latest_report_date()
    
    try:
        return pd.read_csv(output_dir / f"Sample_Trading_Book_{report_date}_Scenario_Analysis.csv")
    except Exception as e:
        return pd.DataFrame({"Scenario": [], "P&L": []})


# =============================================================================
# Helper Functions
# =============================================================================

def format_currency(value: float, decimals: int = 0) -> str:
    """Format value as currency."""
    if pd.isna(value):
        return "N/A"
    return f"${value:,.{decimals}f}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with thousands separator."""
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def get_limit_status(value: float, limit: float) -> tuple:
    """Get limit status and color."""
    utilization = abs(value) / limit * 100 if limit > 0 else 0
    
    if utilization >= 100:
        return "BREACH", COLORS["danger"], utilization
    elif utilization >= 80:
        return "WARNING", COLORS["warning"], utilization
    else:
        return "OK", COLORS["success"], utilization


def create_metric_card(title: str, value: str, subtitle: str = "", status_color: str = None) -> ui.TagChild:
    """Create a metric display card."""
    border_style = f"border-left: 4px solid {status_color};" if status_color else ""
    
    return ui.div(
        ui.div(
            ui.p(title, class_="text-muted mb-1", style="font-size: 0.85rem;"),
            ui.h4(value, class_="mb-0 fw-bold"),
            ui.p(subtitle, class_="text-muted mb-0", style="font-size: 0.75rem;") if subtitle else None,
            class_="p-3"
        ),
        class_="card shadow-sm mb-3",
        style=f"background: white; {border_style}"
    )


# =============================================================================
# UI Definition
# =============================================================================

app_ui = ui.page_fluid(
    # Header
    ui.div(
        ui.div(
            ui.h3("📊 Rates Risk Monitor", class_="mb-0 text-white"),
            ui.p(
                ui.output_text("header_timestamp"),
                class_="mb-0 text-white-50",
                style="font-size: 0.85rem;"
            ),
            class_="d-flex justify-content-between align-items-center"
        ),
        class_="bg-dark p-3 mb-4"
    ),
    
    # Main content
    ui.div(
        # Row 1: Summary Metrics
        ui.h5("📈 Portfolio Summary", class_="mb-3"),
        ui.div(
            ui.div(ui.output_ui("card_total_pv"), class_="col-md-3"),
            ui.div(ui.output_ui("card_total_dv01"), class_="col-md-3"),
            ui.div(ui.output_ui("card_total_convexity"), class_="col-md-3"),
            ui.div(ui.output_ui("card_num_positions"), class_="col-md-3"),
            class_="row"
        ),
        
        # Row 2: VaR Metrics
        ui.h5("🎯 Value at Risk", class_="mb-3 mt-4"),
        ui.div(
            ui.div(ui.output_ui("card_var_95"), class_="col-md-3"),
            ui.div(ui.output_ui("card_var_99"), class_="col-md-3"),
            ui.div(ui.output_ui("card_es_95"), class_="col-md-3"),
            ui.div(ui.output_ui("card_es_99"), class_="col-md-3"),
            class_="row"
        ),
        
        # Row 3: Charts
        ui.div(
            # Key Rate DV01 Chart
            ui.div(
                ui.div(
                    ui.div(
                        ui.h6("🔑 Key Rate DV01 Exposure", class_="mb-0"),
                        class_="card-header bg-white"
                    ),
                    ui.div(
                        output_widget("plot_key_rate_dv01", height="300px"),
                        class_="card-body"
                    ),
                    class_="card shadow-sm"
                ),
                class_="col-md-6"
            ),
            # Scenario Analysis Chart
            ui.div(
                ui.div(
                    ui.div(
                        ui.h6("📉 Scenario P&L", class_="mb-0"),
                        class_="card-header bg-white"
                    ),
                    ui.div(
                        output_widget("plot_scenarios", height="300px"),
                        class_="card-body"
                    ),
                    class_="card shadow-sm"
                ),
                class_="col-md-6"
            ),
            class_="row mt-4"
        ),
        
        # Row 4: Risk Limits
        ui.h5("⚠️ Risk Limits", class_="mb-3 mt-4"),
        ui.div(
            ui.div(
                ui.div(
                    ui.div(
                        ui.h6("Limit Utilization", class_="mb-0"),
                        class_="card-header bg-white"
                    ),
                    ui.div(
                        ui.output_ui("limit_utilization_table"),
                        class_="card-body"
                    ),
                    class_="card shadow-sm"
                ),
                class_="col-12"
            ),
            class_="row"
        ),
        
        # Row 5: Position Details
        ui.h5("📋 Position Details", class_="mb-3 mt-4"),
        ui.div(
            ui.div(
                ui.div(
                    ui.div(
                        ui.h6("Position-Level Risk", class_="mb-0"),
                        class_="card-header bg-white"
                    ),
                    ui.div(
                        ui.output_data_frame("positions_table"),
                        class_="card-body"
                    ),
                    class_="card shadow-sm"
                ),
                class_="col-12"
            ),
            class_="row"
        ),
        
        # Row 6: Position DV01 Chart
        ui.div(
            ui.div(
                ui.div(
                    ui.div(
                        ui.h6("📊 Position DV01 Breakdown", class_="mb-0"),
                        class_="card-header bg-white"
                    ),
                    ui.div(
                        output_widget("plot_position_dv01", height="350px"),
                        class_="card-body"
                    ),
                    class_="card shadow-sm"
                ),
                class_="col-12"
            ),
            class_="row mt-4"
        ),
        
        class_="container-fluid px-4"
    ),
    
    # Footer
    ui.div(
        ui.p(
            "Rates Risk Library v1.0 | Data refreshed: ",
            ui.output_text("footer_timestamp", inline=True),
            class_="mb-0 text-muted text-center",
            style="font-size: 0.75rem;"
        ),
        class_="mt-4 py-3 border-top"
    ),
    
    title="Rates Risk Monitor"
)


# =============================================================================
# Server Logic
# =============================================================================

def server(input, output, session):
    """Server function for the dashboard."""
    
    # Reactive data sources
    @reactive.calc
    def portfolio_summary():
        return load_portfolio_summary()
    
    @reactive.calc
    def key_rate_dv01():
        return load_key_rate_dv01()
    
    @reactive.calc
    def var_metrics():
        return load_var_metrics()
    
    @reactive.calc
    def positions():
        return load_positions()
    
    @reactive.calc
    def scenarios():
        return load_scenarios()
    
    # Header timestamp
    @output
    @render.text
    def header_timestamp():
        report_date = get_latest_report_date()
        formatted = f"{report_date[:4]}-{report_date[4:6]}-{report_date[6:]}"
        return f"Report Date: {formatted} | Portfolio: Sample Trading Book"
    
    @output
    @render.text
    def footer_timestamp():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Portfolio Summary Cards
    @output
    @render.ui
    def card_total_pv():
        summary = portfolio_summary()
        value = summary.get("Total PV", 0)
        return create_metric_card(
            "Total PV",
            format_currency(value),
            "Mark-to-Market",
            COLORS["primary"]
        )
    
    @output
    @render.ui
    def card_total_dv01():
        summary = portfolio_summary()
        value = summary.get("Total DV01", 0)
        status, color, util = get_limit_status(value, RISK_LIMITS["total_dv01"])
        return create_metric_card(
            "Total DV01",
            format_currency(value),
            f"Limit: {format_currency(RISK_LIMITS['total_dv01'])} ({util:.0f}%)",
            color
        )
    
    @output
    @render.ui
    def card_total_convexity():
        summary = portfolio_summary()
        value = summary.get("Total Convexity", 0)
        return create_metric_card(
            "Convexity",
            format_number(value),
            "Portfolio convexity",
            COLORS["info"]
        )
    
    @output
    @render.ui
    def card_num_positions():
        summary = portfolio_summary()
        value = summary.get("Number of Positions", 0)
        return create_metric_card(
            "Positions",
            str(int(value)),
            "Active positions",
            COLORS["info"]
        )
    
    # VaR Cards
    @output
    @render.ui
    def card_var_95():
        var = var_metrics()
        value = var.get("Historical VaR 95%", 0)
        status, color, util = get_limit_status(value, RISK_LIMITS["var_95"])
        return create_metric_card(
            "Historical VaR 95%",
            format_currency(value),
            f"Limit: {format_currency(RISK_LIMITS['var_95'])} ({util:.0f}%)",
            color
        )
    
    @output
    @render.ui
    def card_var_99():
        var = var_metrics()
        value = var.get("Historical VaR 99%", 0)
        status, color, util = get_limit_status(value, RISK_LIMITS["var_99"])
        return create_metric_card(
            "Historical VaR 99%",
            format_currency(value),
            f"Limit: {format_currency(RISK_LIMITS['var_99'])} ({util:.0f}%)",
            color
        )
    
    @output
    @render.ui
    def card_es_95():
        var = var_metrics()
        value = var.get("Historical ES 95%", 0)
        return create_metric_card(
            "Expected Shortfall 95%",
            format_currency(value),
            "Conditional VaR",
            COLORS["warning"]
        )
    
    @output
    @render.ui
    def card_es_99():
        var = var_metrics()
        value = var.get("Historical ES 99%", 0)
        return create_metric_card(
            "Expected Shortfall 99%",
            format_currency(value),
            "Conditional VaR",
            COLORS["danger"]
        )
    
    # Key Rate DV01 Chart
    @output
    @render_widget
    def plot_key_rate_dv01():
        df = key_rate_dv01()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        colors = [COLORS["primary"] if v >= 0 else COLORS["danger"] for v in df["DV01"]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=df["Tenor"],
                y=df["DV01"],
                marker_color=colors,
                text=[f"${v:,.0f}" for v in df["DV01"]],
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            margin=dict(l=40, r=40, t=20, b=40),
            xaxis_title="Tenor",
            yaxis_title="DV01 ($)",
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#eee")
        )
        
        return fig
    
    # Scenario Analysis Chart
    @output
    @render_widget
    def plot_scenarios():
        df = scenarios()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Sort by P&L
        df = df.sort_values("P&L")
        colors = [COLORS["success"] if v >= 0 else COLORS["danger"] for v in df["P&L"]]
        
        fig = go.Figure(data=[
            go.Bar(
                y=df["Scenario"],
                x=df["P&L"],
                orientation="h",
                marker_color=colors,
                text=[f"${v:,.0f}" for v in df["P&L"]],
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            margin=dict(l=150, r=80, t=20, b=40),
            xaxis_title="P&L ($)",
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee", zeroline=True, zerolinecolor="#333")
        )
        
        return fig
    
    # Limit Utilization Table
    @output
    @render.ui
    def limit_utilization_table():
        summary = portfolio_summary()
        var = var_metrics()
        
        limits_data = [
            ("Total DV01", summary.get("Total DV01", 0), RISK_LIMITS["total_dv01"]),
            ("VaR 95%", var.get("Historical VaR 95%", 0), RISK_LIMITS["var_95"]),
            ("VaR 99%", var.get("Historical VaR 99%", 0), RISK_LIMITS["var_99"]),
        ]
        
        rows = []
        for name, value, limit in limits_data:
            status, color, util = get_limit_status(value, limit)
            
            # Progress bar
            progress_width = min(util, 100)
            progress_bar = ui.div(
                ui.div(
                    style=f"width: {progress_width}%; background-color: {color}; height: 100%;",
                    class_="rounded"
                ),
                class_="progress",
                style="height: 20px; background-color: #e9ecef;"
            )
            
            # Status badge
            badge_class = {
                "OK": "bg-success",
                "WARNING": "bg-warning text-dark",
                "BREACH": "bg-danger"
            }.get(status, "bg-secondary")
            
            rows.append(
                ui.tags.tr(
                    ui.tags.td(name, style="font-weight: 500;"),
                    ui.tags.td(format_currency(abs(value))),
                    ui.tags.td(format_currency(limit)),
                    ui.tags.td(progress_bar, style="width: 200px;"),
                    ui.tags.td(f"{util:.1f}%"),
                    ui.tags.td(ui.span(status, class_=f"badge {badge_class}")),
                )
            )
        
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("Risk Metric"),
                    ui.tags.th("Current"),
                    ui.tags.th("Limit"),
                    ui.tags.th("Utilization"),
                    ui.tags.th("%"),
                    ui.tags.th("Status"),
                )
            ),
            ui.tags.tbody(*rows),
            class_="table table-hover"
        )
    
    # Positions Table
    @output
    @render.data_frame
    def positions_table():
        df = positions()
        if df.empty:
            return pd.DataFrame()
        
        # Select and format columns
        display_cols = ["position_id", "instrument", "type", "notional", "pv", "dv01"]
        display_df = df[display_cols].copy()
        
        # Format numbers
        display_df["notional"] = display_df["notional"].apply(lambda x: f"${x:,.0f}")
        display_df["pv"] = display_df["pv"].apply(lambda x: f"${x:,.2f}")
        display_df["dv01"] = display_df["dv01"].apply(lambda x: f"${x:,.2f}")
        
        # Rename columns
        display_df.columns = ["Position ID", "Instrument", "Type", "Notional", "PV", "DV01"]
        
        return render.DataGrid(display_df, filters=True, height="300px")
    
    # Position DV01 Chart
    @output
    @render_widget
    def plot_position_dv01():
        df = positions()
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Sort by absolute DV01
        df = df.sort_values("dv01", key=abs, ascending=True)
        colors = [COLORS["primary"] if v >= 0 else COLORS["danger"] for v in df["dv01"]]
        
        fig = go.Figure(data=[
            go.Bar(
                y=df["instrument"],
                x=df["dv01"],
                orientation="h",
                marker_color=colors,
                text=[f"${v:,.0f}" for v in df["dv01"]],
                textposition="outside"
            )
        ])
        
        fig.update_layout(
            margin=dict(l=100, r=80, t=20, b=40),
            xaxis_title="DV01 ($)",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#eee", zeroline=True, zerolinecolor="#333")
        )
        
        return fig


# =============================================================================
# Create App
# =============================================================================

app = App(app_ui, server)


if __name__ == "__main__":
    # For development, run with:
    # shiny run dashboard/app.py --reload
    print("Starting Rates Risk Monitor...")
    print("Run with: shiny run app.py --reload")
