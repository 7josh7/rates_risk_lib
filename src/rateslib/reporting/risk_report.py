"""
Risk reporting functionality.

Provides formatted console output and CSV export for:
- Position-level risk
- Portfolio summary
- VaR/ES metrics
- Key rate durations
- P&L attribution

Per specification Section 11: Output both console (formatted) and CSV.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class ReportSection:
    """
    A section of a report.
    
    Attributes:
        title: Section title
        data: Data (DataFrame or dict)
        notes: Optional notes
    """
    title: str
    data: Union[pd.DataFrame, Dict[str, Any]]
    notes: Optional[str] = None


@dataclass
class RiskReport:
    """
    Complete risk report.
    
    Attributes:
        report_date: Date of report
        portfolio_name: Name of portfolio
        sections: List of report sections
        metadata: Additional metadata
    """
    report_date: date
    portfolio_name: str
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_section(
        self,
        title: str,
        data: Union[pd.DataFrame, Dict[str, Any]],
        notes: Optional[str] = None
    ):
        """Add a section to the report."""
        self.sections.append(ReportSection(title, data, notes))
    
    def to_dict(self) -> Dict:
        """Convert entire report to dictionary."""
        result = {
            "report_date": str(self.report_date),
            "portfolio_name": self.portfolio_name,
            "metadata": self.metadata,
            "sections": {}
        }
        
        for section in self.sections:
            if isinstance(section.data, pd.DataFrame):
                result["sections"][section.title] = section.data.to_dict(orient="records")
            else:
                result["sections"][section.title] = section.data
        
        return result


class ReportFormatter:
    """
    Formats reports for console output.
    """
    
    def __init__(
        self,
        width: int = 80,
        precision: int = 2,
        thousands_sep: bool = True
    ):
        """
        Initialize formatter.
        
        Args:
            width: Console width
            precision: Decimal precision for floats
            thousands_sep: Whether to use thousands separator
        """
        self.width = width
        self.precision = precision
        self.thousands_sep = thousands_sep
    
    def format_number(self, value: float, precision: Optional[int] = None) -> str:
        """Format a number for display."""
        p = precision if precision is not None else self.precision
        
        if abs(value) >= 1e6:
            return f"{value/1e6:,.{p}f}M"
        elif abs(value) >= 1e3:
            if self.thousands_sep:
                return f"{value:,.{p}f}"
            return f"{value:.{p}f}"
        else:
            return f"{value:.{p}f}"
    
    def format_bp(self, value: float) -> str:
        """Format basis points."""
        return f"{value:.2f}bp"
    
    def format_percent(self, value: float) -> str:
        """Format as percentage."""
        return f"{value*100:.2f}%"
    
    def header(self, title: str) -> str:
        """Create a header line."""
        return f"\n{'='*self.width}\n{title.center(self.width)}\n{'='*self.width}\n"
    
    def subheader(self, title: str) -> str:
        """Create a subheader."""
        return f"\n{'-'*self.width}\n{title}\n{'-'*self.width}\n"
    
    def format_dict(self, data: Dict[str, Any], indent: int = 2) -> str:
        """Format dictionary as key-value pairs."""
        lines = []
        pad = " " * indent
        
        for key, value in data.items():
            if isinstance(value, float):
                formatted = self.format_number(value)
            else:
                formatted = str(value)
            lines.append(f"{pad}{key}: {formatted}")
        
        return "\n".join(lines)
    
    def format_dataframe(
        self,
        df: pd.DataFrame,
        max_rows: int = 50
    ) -> str:
        """Format DataFrame for console."""
        # Configure pandas display
        with pd.option_context(
            'display.max_rows', max_rows,
            'display.width', self.width,
            'display.float_format', lambda x: self.format_number(x)
        ):
            return df.to_string()
    
    def format_report(self, report: RiskReport) -> str:
        """Format entire report for console."""
        lines = []
        
        lines.append(self.header(f"Risk Report: {report.portfolio_name}"))
        lines.append(f"Report Date: {report.report_date}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if report.metadata:
            lines.append("\nMetadata:")
            lines.append(self.format_dict(report.metadata))
        
        for section in report.sections:
            lines.append(self.subheader(section.title))
            
            if isinstance(section.data, pd.DataFrame):
                lines.append(self.format_dataframe(section.data))
            else:
                lines.append(self.format_dict(section.data))
            
            if section.notes:
                lines.append(f"\nNote: {section.notes}")
        
        lines.append(f"\n{'='*self.width}")
        lines.append("End of Report")
        
        return "\n".join(lines)


def generate_risk_summary(
    portfolio_pv: float,
    total_dv01: float,
    total_convexity: float,
    var_95: float,
    var_99: float,
    es_95: Optional[float] = None,
    es_99: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate risk summary statistics.
    
    Args:
        portfolio_pv: Portfolio present value
        total_dv01: Portfolio DV01
        total_convexity: Portfolio convexity
        var_95: 95% VaR
        var_99: 99% VaR
        es_95: 95% ES (optional)
        es_99: 99% ES (optional)
        
    Returns:
        Dictionary of summary statistics
    """
    summary = {
        "Portfolio PV": portfolio_pv,
        "Total DV01": total_dv01,
        "Total Convexity": total_convexity,
        "VaR (95%)": var_95,
        "VaR (99%)": var_99,
        "DV01 / VaR(95%)": total_dv01 / var_95 if var_95 != 0 else 0,
    }
    
    if es_95 is not None:
        summary["ES (95%)"] = es_95
    if es_99 is not None:
        summary["ES (99%)"] = es_99
    
    return summary


def generate_position_report(
    positions: List[Dict[str, Any]],
    include_greeks: bool = True
) -> pd.DataFrame:
    """
    Generate position-level report.
    
    Args:
        positions: List of position dictionaries
        include_greeks: Whether to include DV01/convexity
        
    Returns:
        DataFrame with position details
    """
    df = pd.DataFrame(positions)
    
    # Standard columns
    display_cols = ["instrument", "notional", "pv"]
    
    if include_greeks:
        if "dv01" in df.columns:
            display_cols.append("dv01")
        if "convexity" in df.columns:
            display_cols.append("convexity")
    
    # Add optional columns if present
    for col in ["yield", "spread", "duration", "mod_duration"]:
        if col in df.columns:
            display_cols.append(col)
    
    available_cols = [c for c in display_cols if c in df.columns]
    
    return df[available_cols].copy()


def generate_var_report(
    var_results: Dict[str, float],
    scenarios: Optional[Dict[str, float]] = None,
    key_rate_contributions: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Generate VaR analysis report.
    
    Args:
        var_results: Core VaR metrics
        scenarios: Scenario P&L results
        key_rate_contributions: Key rate DV01 contributions
        
    Returns:
        DataFrame with VaR details
    """
    rows = []
    
    # Core VaR metrics
    for metric, value in var_results.items():
        rows.append({
            "Category": "VaR Metrics",
            "Metric": metric,
            "Value": value
        })
    
    # Scenario results
    if scenarios:
        for scenario, pnl in scenarios.items():
            rows.append({
                "Category": "Scenarios",
                "Metric": scenario,
                "Value": pnl
            })
    
    # Key rate contributions
    if key_rate_contributions:
        for tenor, contribution in key_rate_contributions.items():
            rows.append({
                "Category": "Key Rate Contributions",
                "Metric": tenor,
                "Value": contribution
            })
    
    return pd.DataFrame(rows)


def generate_key_rate_report(
    key_rate_dv01: Dict[str, float],
    total_dv01: float
) -> pd.DataFrame:
    """
    Generate key rate DV01 report.
    
    Args:
        key_rate_dv01: DV01 by tenor
        total_dv01: Total portfolio DV01
        
    Returns:
        DataFrame with KRD breakdown
    """
    rows = []
    
    for tenor, dv01 in key_rate_dv01.items():
        pct = (dv01 / total_dv01 * 100) if total_dv01 != 0 else 0
        rows.append({
            "Tenor": tenor,
            "DV01": dv01,
            "% of Total": pct,
            "Cumulative %": 0  # Filled below
        })
    
    df = pd.DataFrame(rows)
    df["Cumulative %"] = df["% of Total"].cumsum()
    
    return df


def generate_pnl_report(
    components: Dict[str, float],
    historical_pnl: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Generate P&L attribution report.
    
    Args:
        components: P&L components
        historical_pnl: Optional historical P&L series
        
    Returns:
        DataFrame with P&L breakdown
    """
    rows = []
    
    total = sum(components.values())
    
    for component, value in components.items():
        pct = (value / total * 100) if total != 0 else 0
        rows.append({
            "Component": component,
            "P&L": value,
            "% of Total": pct
        })
    
    df = pd.DataFrame(rows)
    
    # Add total row
    total_row = pd.DataFrame([{
        "Component": "TOTAL",
        "P&L": total,
        "% of Total": 100.0
    }])
    
    df = pd.concat([df, total_row], ignore_index=True)
    
    return df


def export_to_csv(
    report: RiskReport,
    output_dir: Union[str, Path],
    prefix: Optional[str] = None
) -> List[str]:
    """
    Export report to CSV files.
    
    Creates one CSV per section.
    
    Args:
        report: RiskReport to export
        output_dir: Output directory
        prefix: Optional filename prefix
        
    Returns:
        List of created file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    prefix = prefix or report.portfolio_name.replace(" ", "_")
    date_str = report.report_date.strftime("%Y%m%d")
    
    created_files = []
    
    # Export metadata
    meta_file = output_path / f"{prefix}_{date_str}_metadata.csv"
    meta_df = pd.DataFrame([{
        "report_date": str(report.report_date),
        "portfolio_name": report.portfolio_name,
        **report.metadata
    }])
    meta_df.to_csv(meta_file, index=False)
    created_files.append(str(meta_file))
    
    # Export each section
    for section in report.sections:
        safe_title = section.title.replace(" ", "_").replace("/", "_")
        filename = output_path / f"{prefix}_{date_str}_{safe_title}.csv"
        
        if isinstance(section.data, pd.DataFrame):
            section.data.to_csv(filename, index=False)
        else:
            # Convert dict to DataFrame
            df = pd.DataFrame([section.data])
            df.to_csv(filename, index=False)
        
        created_files.append(str(filename))
    
    return created_files


def print_report(report: RiskReport, formatter: Optional[ReportFormatter] = None):
    """
    Print report to console.
    
    Args:
        report: RiskReport to print
        formatter: Optional custom formatter
    """
    fmt = formatter or ReportFormatter()
    print(fmt.format_report(report))


__all__ = [
    "ReportSection",
    "RiskReport",
    "ReportFormatter",
    "generate_risk_summary",
    "generate_position_report",
    "generate_var_report",
    "generate_key_rate_report",
    "generate_pnl_report",
    "export_to_csv",
    "print_report",
]
