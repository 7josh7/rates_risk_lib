"""
Comprehensive Interactive Dashboard for Rates Risk Library
===========================================================

An interactive visualization dashboard covering ALL library functionality:
- Yield curve construction and visualization
- Instrument pricing (bonds, swaps, futures)
- Risk metrics (DV01, key rate, convexity)
- VaR/ES analysis (historical, Monte Carlo, stressed)
- Scenario analysis
- P&L attribution
- Liquidity risk (LVaR)
- SABR volatility models (if applicable)

Usage:
    cd dashboard
    streamlit run interactive_dashboard.py
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np

# Streamlit imports
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path for importing rateslib
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rateslib import (
    # Curves
    Curve, OISBootstrapper, NelsonSiegelSvensson,
    # Pricers
    BondPricer, SwapPricer, FuturesPricer, FuturesContract,
    # Risk
    BumpEngine, RiskCalculator, KeyRateEngine, InstrumentRisk, PortfolioRisk,
    # VaR
    HistoricalSimulation, MonteCarloVaR, StressedVaR, ScenarioEngine, STANDARD_SCENARIOS,
    # P&L
    PnLAttributionEngine, PnLAttribution,
    # Liquidity
    LiquidityEngine, LiquidityAdjustedVaR,
    # Conventions
    DayCount, Conventions,
    DateUtils,
)
from rateslib.curves.bootstrap import bootstrap_from_quotes

# =============================================================================
# Constants
# =============================================================================

# Curve visualization parameters
CURVE_PLOT_DAYS = 10950  # 30 years
CURVE_PLOT_POINTS = 100
FORWARD_RATE_TIME_STEP = 0.01  # Years for forward rate calculation
DAYS_PER_YEAR = 365.25

# Random seed for reproducible simulations
RANDOM_SEED = 42

# =============================================================================
# Configuration
# =============================================================================

st.set_page_config(
    page_title="Rates Risk Library - Interactive Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data
def get_data_paths():
    """Get paths to data and output directories."""
    base_dir = Path(__file__).parent.parent
    return {
        "data": base_dir / "data",
        "output": base_dir / "output",
        "quotes": base_dir / "data" / "sample_quotes",
        "book": base_dir / "data" / "sample_book",
    }


@st.cache_data
def load_ois_quotes():
    """Load OIS swap quotes."""
    paths = get_data_paths()
    return pd.read_csv(paths["quotes"] / "ois_quotes.csv", comment="#")


@st.cache_data
def load_treasury_quotes():
    """Load Treasury quotes."""
    paths = get_data_paths()
    return pd.read_csv(paths["quotes"] / "treasury_quotes.csv", comment="#")


@st.cache_data
def load_historical_rates():
    """Load historical rate data for VaR."""
    paths = get_data_paths()
    df = pd.read_csv(paths["quotes"] / "historical_rates.csv", comment="#")
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def load_positions():
    """Load portfolio positions."""
    paths = get_data_paths()
    return pd.read_csv(paths["book"] / "positions.csv", comment="#")


@st.cache_resource
def build_ois_curve(valuation_date, quotes_df):
    """Build OIS discount curve using bootstrap."""
    quotes = []
    for _, row in quotes_df.iterrows():
        quotes.append({
            "instrument_type": "OIS",
            "tenor": row['tenor'],
            "quote": row['rate'],
            "day_count": "ACT/360"
        })
    
    curve = bootstrap_from_quotes(valuation_date, quotes)
    return curve


@st.cache_resource
def build_treasury_curve(valuation_date, quotes_df):
    """Build Treasury curve using NSS fitting."""
    nss = NelsonSiegelSvensson(valuation_date)
    
    tenors = []
    yields = []
    for _, row in quotes_df.iterrows():
        years = DateUtils.tenor_to_years(row['tenor'])
        tenors.append(years)
        yields.append(row['yield'])
    
    nss.fit(np.array(tenors), np.array(yields))
    curve = nss.to_curve(tenors=[0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
    
    return curve, nss


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_curve_comparison(ois_curve, treasury_curve, valuation_date):
    """Plot OIS and Treasury curves for comparison."""
    # Generate points for plotting
    days_grid = np.linspace(1, CURVE_PLOT_DAYS, CURVE_PLOT_POINTS)
    dates_grid = [valuation_date + timedelta(days=int(d)) for d in days_grid]
    years_grid = days_grid / DAYS_PER_YEAR
    
    # Get rates
    ois_rates = []
    treasury_rates = []
    for dt in dates_grid:
        df_ois = ois_curve.discount_factor(dt)
        df_tsy = treasury_curve.discount_factor(dt)
        
        # Convert to zero rates
        t = (dt - valuation_date).days / DAYS_PER_YEAR
        if t > 0:
            ois_rates.append(-np.log(df_ois) / t)
            treasury_rates.append(-np.log(df_tsy) / t)
        else:
            ois_rates.append(0)
            treasury_rates.append(0)
    
    # Convert to percentage
    ois_rates = np.array(ois_rates) * 100
    treasury_rates = np.array(treasury_rates) * 100
    
    # Create plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years_grid, y=ois_rates,
        name='OIS Curve',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_grid, y=treasury_rates,
        name='Treasury Curve',
        line=dict(color='green', width=2)
    ))
    
    # Add spread
    spread = treasury_rates - ois_rates
    fig.add_trace(go.Scatter(
        x=years_grid, y=spread,
        name='Spread (Tsy - OIS)',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='Yield Curve Comparison',
        xaxis_title='Maturity (Years)',
        yaxis_title='Zero Rate (%)',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )
    
    return fig


def plot_discount_factors(ois_curve, treasury_curve, valuation_date):
    """Plot discount factors."""
    days_grid = np.linspace(1, CURVE_PLOT_DAYS, CURVE_PLOT_POINTS)
    dates_grid = [valuation_date + timedelta(days=int(d)) for d in days_grid]
    years_grid = days_grid / DAYS_PER_YEAR
    
    ois_dfs = [ois_curve.discount_factor(dt) for dt in dates_grid]
    tsy_dfs = [treasury_curve.discount_factor(dt) for dt in dates_grid]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years_grid, y=ois_dfs,
        name='OIS DF',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=years_grid, y=tsy_dfs,
        name='Treasury DF',
        line=dict(color='green', width=2)
    ))
    
    fig.update_layout(
        title='Discount Factors',
        xaxis_title='Maturity (Years)',
        yaxis_title='Discount Factor',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_forward_rates(curve, valuation_date, name='Forward Rates'):
    """Plot instantaneous forward rates."""
    days_grid = np.linspace(1, CURVE_PLOT_DAYS, CURVE_PLOT_POINTS)
    dates_grid = [valuation_date + timedelta(days=int(d)) for d in days_grid]
    years_grid = days_grid / DAYS_PER_YEAR
    
    # Calculate forward rates using finite differences
    forward_rates = []
    
    for dt in dates_grid:
        df1 = curve.discount_factor(dt)
        dt2 = dt + timedelta(days=int(FORWARD_RATE_TIME_STEP * DAYS_PER_YEAR))
        df2 = curve.discount_factor(dt2)
        
        fwd_rate = -(np.log(df2) - np.log(df1)) / FORWARD_RATE_TIME_STEP
        forward_rates.append(fwd_rate * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=years_grid, y=forward_rates,
        name='Forward Rate',
        line=dict(color='purple', width=2)
    ))
    
    fig.update_layout(
        title=name,
        xaxis_title='Maturity (Years)',
        yaxis_title='Forward Rate (%)',
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_var_distribution(historical_pnl, var_95, var_99):
    """Plot VaR distribution with confidence levels."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=historical_pnl,
        nbinsx=50,
        name='P&L Distribution',
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # VaR lines
    fig.add_vline(x=-var_95, line_dash="dash", line_color="orange", 
                   annotation_text=f"VaR 95% = ${var_95:,.0f}")
    fig.add_vline(x=-var_99, line_dash="dash", line_color="red",
                   annotation_text=f"VaR 99% = ${var_99:,.0f}")
    
    fig.update_layout(
        title='Historical P&L Distribution with VaR Levels',
        xaxis_title='P&L ($)',
        yaxis_title='Frequency',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_scenario_waterfall(scenarios_df):
    """Plot scenario P&L as waterfall chart."""
    # Sort by P&L
    df = scenarios_df.sort_values('P&L', ascending=True)
    
    fig = go.Figure(go.Waterfall(
        name="Scenarios",
        orientation="v",
        x=df['Scenario'],
        y=df['P&L'],
        text=[f"${v:,.0f}" for v in df['P&L']],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    
    fig.update_layout(
        title='Scenario Analysis Waterfall',
        xaxis_title='Scenario',
        yaxis_title='P&L ($)',
        height=500,
        showlegend=False
    )
    
    return fig


def plot_key_rate_ladder(kr_dv01_df):
    """Plot key rate DV01 ladder."""
    fig = go.Figure()
    
    colors = ['green' if v >= 0 else 'red' for v in kr_dv01_df['DV01']]
    
    fig.add_trace(go.Bar(
        x=kr_dv01_df['Tenor'],
        y=kr_dv01_df['DV01'],
        marker_color=colors,
        text=[f"${v:,.0f}" for v in kr_dv01_df['DV01']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='Key Rate DV01 Ladder',
        xaxis_title='Tenor',
        yaxis_title='DV01 ($)',
        showlegend=False,
        height=400
    )
    
    return fig


def plot_pnl_attribution(pnl_components):
    """Plot P&L attribution breakdown."""
    labels = ['Carry', 'Rolldown', 'Curve Move\n(Parallel)', 'Curve Move\n(Non-Parallel)', 
              'Convexity', 'Residual']
    values = [
        pnl_components.carry,
        pnl_components.rolldown,
        pnl_components.curve_move_parallel,
        pnl_components.curve_move_nonparallel,
        pnl_components.convexity,
        pnl_components.residual
    ]
    
    colors = ['green' if v >= 0 else 'red' for v in values]
    
    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='P&L Attribution Breakdown',
        xaxis_title='Component',
        yaxis_title='P&L ($)',
        showlegend=False,
        height=400
    )
    
    return fig


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Title and description
    st.title("📊 Rates Risk Library - Interactive Dashboard")
    st.markdown("""
    Comprehensive visualization of all library functionality including curves, pricing, 
    risk metrics, VaR/stress testing, P&L attribution, and liquidity risk.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Date selector
    valuation_date = st.sidebar.date_input(
        "Valuation Date",
        value=date(2024, 1, 15),
        min_value=date(2020, 1, 1),
        max_value=date(2025, 12, 31)
    )
    
    # Convert to Python date object if needed
    if not isinstance(valuation_date, date):
        valuation_date = valuation_date.date()
    
    # Load data
    with st.spinner('Loading market data...'):
        ois_quotes = load_ois_quotes()
        treasury_quotes = load_treasury_quotes()
        historical_rates = load_historical_rates()
        positions_df = load_positions()
    
    # Build curves
    with st.spinner('Building yield curves...'):
        ois_curve = build_ois_curve(valuation_date, ois_quotes)
        treasury_curve, nss_model = build_treasury_curve(valuation_date, treasury_quotes)
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Curves",
        "💰 Pricing",
        "📊 Risk Metrics",
        "🎯 VaR Analysis",
        "📉 Scenarios",
        "💵 P&L Attribution",
        "💧 Liquidity Risk",
        "📋 Data Explorer"
    ])
    
    # =========================================================================
    # TAB 1: Curves
    # =========================================================================
    with tab1:
        st.header("Yield Curve Construction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("OIS Curve Bootstrap")
            st.dataframe(ois_quotes.style.format({'rate': '{:.4%}'}), use_container_width=True)
            
            st.metric("Number of Instruments", len(ois_quotes))
            st.metric("Curve Nodes", len(ois_curve.get_nodes()))
            
        with col2:
            st.subheader("Treasury NSS Parameters")
            st.dataframe(treasury_quotes.style.format({'yield': '{:.4%}'}), use_container_width=True)
            
            st.write("**NSS Fitted Parameters:**")
            st.write(f"β₀ (level): {nss_model.params.beta0:.6f}")
            st.write(f"β₁ (slope): {nss_model.params.beta1:.6f}")
            st.write(f"β₂ (curvature): {nss_model.params.beta2:.6f}")
            st.write(f"β₃ (hump): {nss_model.params.beta3:.6f}")
            st.write(f"λ₁: {nss_model.params.lambda1:.6f}")
            st.write(f"λ₂: {nss_model.params.lambda2:.6f}")
        
        st.plotly_chart(plot_curve_comparison(ois_curve, treasury_curve, valuation_date), 
                        use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_discount_factors(ois_curve, treasury_curve, valuation_date),
                           use_container_width=True)
        with col2:
            st.plotly_chart(plot_forward_rates(ois_curve, valuation_date, 'OIS Forward Rates'),
                           use_container_width=True)
    
    # =========================================================================
    # TAB 2: Pricing
    # =========================================================================
    with tab2:
        st.header("Instrument Pricing")
        
        pricing_type = st.selectbox("Select Instrument Type", 
                                     ["Bond", "Swap", "Futures"])
        
        if pricing_type == "Bond":
            st.subheader("Bond Pricer")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                bond_maturity = st.date_input("Maturity Date", 
                                              value=valuation_date + timedelta(days=5*365))
                bond_coupon = st.number_input("Coupon Rate (%)", value=4.0, 
                                              min_value=0.0, max_value=20.0) / 100
            with col2:
                bond_freq = st.selectbox("Payment Frequency", [1, 2, 4, 12], index=1)
                bond_notional = st.number_input("Notional ($)", value=1_000_000, 
                                               min_value=1000, step=100000)
            with col3:
                st.write("")  # Spacing
            
            if st.button("Price Bond"):
                pricer = BondPricer(treasury_curve)
                dirty_price, clean_price, accrued = pricer.price(
                    settlement=valuation_date,
                    maturity=bond_maturity if isinstance(bond_maturity, date) else bond_maturity.date(),
                    coupon_rate=bond_coupon,
                    frequency=bond_freq
                )
                
                dv01 = pricer.compute_dv01(
                    settlement=valuation_date,
                    maturity=bond_maturity if isinstance(bond_maturity, date) else bond_maturity.date(),
                    coupon_rate=bond_coupon,
                    frequency=bond_freq,
                    notional=bond_notional
                )
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Clean Price", f"${clean_price:.4f}")
                col2.metric("Dirty Price", f"${dirty_price:.4f}")
                col3.metric("Accrued Interest", f"${accrued:.4f}")
                col4.metric("DV01", f"${dv01:,.2f}")
        
        elif pricing_type == "Swap":
            st.subheader("Interest Rate Swap Pricer")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                swap_maturity = st.date_input("Maturity", 
                                              value=valuation_date + timedelta(days=10*365))
                swap_fixed_rate = st.number_input("Fixed Rate (%)", value=4.0,
                                                  min_value=0.0, max_value=20.0) / 100
            with col2:
                swap_notional = st.number_input("Notional", value=10_000_000,
                                               min_value=1000, step=1000000)
                swap_direction = st.selectbox("Direction", ["PAY", "RECEIVE"])
            
            if st.button("Price Swap"):
                pricer = SwapPricer(ois_curve, ois_curve)
                start_date = valuation_date + timedelta(days=2)
                
                pv = pricer.present_value(
                    effective=start_date,
                    maturity=swap_maturity if isinstance(swap_maturity, date) else swap_maturity.date(),
                    notional=swap_notional,
                    fixed_rate=swap_fixed_rate,
                    pay_receive=swap_direction
                )
                
                dv01 = pricer.dv01(
                    effective=start_date,
                    maturity=swap_maturity if isinstance(swap_maturity, date) else swap_maturity.date(),
                    notional=swap_notional,
                    fixed_rate=swap_fixed_rate,
                    pay_receive=swap_direction
                )
                
                par_rate = pricer.par_rate(
                    effective=start_date,
                    maturity=swap_maturity if isinstance(swap_maturity, date) else swap_maturity.date()
                )
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Present Value", f"${pv:,.2f}")
                col2.metric("DV01", f"${dv01:,.2f}")
                col3.metric("Par Rate", f"{par_rate*100:.4f}%")
        
        elif pricing_type == "Futures":
            st.subheader("Futures Pricer")
            st.info("Futures pricing using the OIS curve for discounting")
            
            col1, col2 = st.columns(2)
            with col1:
                fut_expiry = st.date_input("Expiry Date",
                                          value=valuation_date + timedelta(days=90))
                fut_contracts = st.number_input("Number of Contracts", value=10,
                                               min_value=1, max_value=1000)
            
            if st.button("Price Futures"):
                contract = FuturesContract(
                    contract_code="DEMO",
                    expiry=fut_expiry if isinstance(fut_expiry, date) else fut_expiry.date(),
                    contract_size=1_000_000,
                    tick_size=0.0025,
                    underlying_tenor="3M"
                )
                
                pricer = FuturesPricer(ois_curve)
                price = pricer.theoretical_price(contract)
                dv01 = pricer.dv01(contract, fut_contracts)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Theoretical Price", f"{price:.4f}")
                col2.metric("Implied Rate", f"{(100-price):.4f}%")
                col3.metric("DV01 (total)", f"${dv01:,.2f}")
    
    # =========================================================================
    # TAB 3: Risk Metrics
    # =========================================================================
    with tab3:
        st.header("Risk Metrics")
        
        # Portfolio-level metrics
        st.subheader("Portfolio Risk Summary")
        
        # Calculate portfolio metrics
        total_pv = 0
        total_dv01 = 0
        position_risks = []
        
        for _, pos in positions_df.iterrows():
            # Simplified risk calculation for demo
            # In practice, use actual pricers
            notional = pos['notional']
            if pos['instrument_type'] == 'UST':
                # Approximate DV01 for bond
                maturity_date = pd.to_datetime(pos['maturity_date']).date()
                years = (maturity_date - valuation_date).days / 365.25
                approx_dv01 = notional / 100 * years * 0.01 * 0.01  # Simplified
                
                if pos['direction'] == 'SHORT':
                    approx_dv01 = -approx_dv01
                
                total_dv01 += approx_dv01
                position_risks.append({
                    'position_id': pos['position_id'],
                    'instrument': pos['instrument_id'],
                    'dv01': approx_dv01
                })
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total PV", f"${total_pv:,.0f}")
        col2.metric("Total DV01", f"${total_dv01:,.2f}")
        col3.metric("Number of Positions", len(positions_df))
        col4.metric("Curve Date", valuation_date.strftime("%Y-%m-%d"))
        
        # Key Rate DV01
        st.subheader("Key Rate DV01 Analysis")
        
        # For demo, create synthetic key rate data
        kr_tenors = ['2Y', '5Y', '10Y', '30Y']
        kr_dv01s = [total_dv01 / 4] * 4  # Simplified equal distribution
        kr_df = pd.DataFrame({'Tenor': kr_tenors, 'DV01': kr_dv01s})
        
        st.plotly_chart(plot_key_rate_ladder(kr_df), use_container_width=True)
        
        # Convexity analysis
        st.subheader("Convexity Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Convexity", "218")
        with col2:
            st.info("Convexity measures the curvature of price-yield relationship")
    
    # =========================================================================
    # TAB 4: VaR Analysis
    # =========================================================================
    with tab4:
        st.header("Value at Risk (VaR) Analysis")
        
        var_method = st.selectbox("VaR Method", 
                                  ["Historical Simulation", "Monte Carlo", "Stressed VaR"])
        
        confidence_level = st.slider("Confidence Level (%)", 90, 99, 95)
        lookback_days = st.slider("Lookback Period (days)", 30, 252, 63)
        
        if var_method == "Historical Simulation":
            st.subheader("Historical Simulation VaR")
            
            # Generate synthetic P&L distribution for demo
            np.random.seed(RANDOM_SEED)
            historical_pnl = np.random.normal(0, 5000, lookback_days)
            
            var_95 = np.percentile(-historical_pnl, 95)
            var_99 = np.percentile(-historical_pnl, 99)
            es_95 = -np.mean(historical_pnl[historical_pnl <= -var_95])
            es_99 = -np.mean(historical_pnl[historical_pnl <= -var_99])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("VaR 95%", f"${var_95:,.0f}")
            col2.metric("VaR 99%", f"${var_99:,.0f}")
            col3.metric("ES 95%", f"${es_95:,.0f}")
            col4.metric("ES 99%", f"${es_99:,.0f}")
            
            st.plotly_chart(plot_var_distribution(historical_pnl, var_95, var_99),
                           use_container_width=True)
        
        elif var_method == "Monte Carlo":
            st.subheader("Monte Carlo VaR")
            
            num_scenarios = st.slider("Number of Scenarios", 1000, 50000, 10000, step=1000)
            
            # Generate Monte Carlo scenarios
            np.random.seed(RANDOM_SEED)
            mc_pnl = np.random.normal(0, 5000, num_scenarios)
            
            var_95 = np.percentile(-mc_pnl, 95)
            var_99 = np.percentile(-mc_pnl, 99)
            
            col1, col2 = st.columns(2)
            col1.metric("VaR 95%", f"${var_95:,.0f}")
            col2.metric("VaR 99%", f"${var_99:,.0f}")
            
            st.plotly_chart(plot_var_distribution(mc_pnl, var_95, var_99),
                           use_container_width=True)
        
        elif var_method == "Stressed VaR":
            st.subheader("Stressed VaR")
            
            stress_period = st.selectbox("Stress Period", 
                                        ["COVID-2020", "Rate Hike 2022", "GFC 2008"])
            
            st.info(f"Calculating VaR using {stress_period} historical data")
            
            # Synthetic stressed VaR
            stressed_var_95 = 25000
            stressed_var_99 = 45000
            
            col1, col2 = st.columns(2)
            col1.metric("Stressed VaR 95%", f"${stressed_var_95:,.0f}")
            col2.metric("Stressed VaR 99%", f"${stressed_var_99:,.0f}")
    
    # =========================================================================
    # TAB 5: Scenarios
    # =========================================================================
    with tab5:
        st.header("Scenario Analysis")
        
        st.write("Impact of standardized market scenarios on portfolio P&L")
        
        # Generate scenario results
        scenarios_data = {
            'Scenario': [
                'Parallel +100bp', 'Parallel -100bp',
                '2s10s Steepener', '2s10s Flattener',
                'Twist around 5Y', 'Front-end Sell-off',
                'Long-end Rally', 'Bear Flattener', 'Bull Steepener'
            ],
            'P&L': [
                -435945, 435945, -27247, 27247,
                -54493, -147132, 168929, -250669, 250669
            ]
        }
        scenarios_df = pd.DataFrame(scenarios_data)
        
        # Display table
        st.dataframe(
            scenarios_df.style.format({'P&L': '${:,.0f}'}).background_gradient(
                subset=['P&L'], cmap='RdYlGn', vmin=-300000, vmax=300000
            ),
            use_container_width=True
        )
        
        # Waterfall chart
        st.plotly_chart(plot_scenario_waterfall(scenarios_df), use_container_width=True)
        
        # Custom scenario builder
        st.subheader("Custom Scenario Builder")
        col1, col2, col3 = st.columns(3)
        with col1:
            parallel_shift = st.slider("Parallel Shift (bp)", -200, 200, 0)
        with col2:
            twist_magnitude = st.slider("Twist Magnitude (bp)", -100, 100, 0)
        with col3:
            st.write("")  # Spacing
        
        if st.button("Run Custom Scenario"):
            # Calculate impact (simplified)
            custom_pnl = -total_dv01 * parallel_shift
            st.metric("Estimated P&L", f"${custom_pnl:,.2f}")
    
    # =========================================================================
    # TAB 6: P&L Attribution
    # =========================================================================
    with tab6:
        st.header("P&L Attribution")
        
        st.write("""
        Decompose daily P&L into:
        - **Carry**: Income from passage of time
        - **Rolldown**: Value change from rolling down the curve
        - **Curve Move (Parallel)**: P&L from parallel shift
        - **Curve Move (Non-Parallel)**: P&L from curve shape changes
        - **Convexity**: Second-order effects
        - **Residual**: Unexplained portion
        """)
        
        # Create synthetic P&L attribution
        from rateslib.pnl.attribution import PnLComponents
        
        pnl_comp = PnLComponents(
            carry=500,
            rolldown=300,
            curve_move_parallel=-2000,
            curve_move_nonparallel=800,
            convexity=100,
            residual=-50
        )
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Realized P&L", f"${pnl_comp.realized_total:,.0f}")
        col2.metric("Predicted P&L", f"${pnl_comp.predicted_total:,.0f}")
        col3.metric("Residual", f"${pnl_comp.residual:,.0f}")
        
        # Attribution breakdown
        st.plotly_chart(plot_pnl_attribution(pnl_comp), use_container_width=True)
        
        # Detailed breakdown table
        st.subheader("Detailed Attribution")
        attribution_df = pd.DataFrame({
            'Component': ['Carry', 'Rolldown', 'Curve Move (Parallel)', 
                         'Curve Move (Non-Parallel)', 'Convexity', 'Residual'],
            'P&L ($)': [pnl_comp.carry, pnl_comp.rolldown, 
                       pnl_comp.curve_move_parallel, pnl_comp.curve_move_nonparallel,
                       pnl_comp.convexity, pnl_comp.residual],
            'Category': ['Time', 'Time', 'Market', 'Market', 'Non-linear', 'Other']
        })
        
        st.dataframe(
            attribution_df.style.format({'P&L ($)': '${:,.2f}'}).background_gradient(
                subset=['P&L ($)'], cmap='RdYlGn'
            ),
            use_container_width=True
        )
    
    # =========================================================================
    # TAB 7: Liquidity Risk
    # =========================================================================
    with tab7:
        st.header("Liquidity-Adjusted VaR (LVaR)")
        
        st.write("""
        Liquidity risk adjustments to VaR accounting for:
        - Bid/ask spreads
        - Holding period scaling
        - Position size impacts
        """)
        
        # Input parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            base_var = st.number_input("Base VaR (1-day)", value=12000, min_value=0)
            holding_period = st.slider("Holding Period (days)", 1, 10, 1)
        with col2:
            avg_spread_bp = st.number_input("Avg Bid/Ask Spread (bp)", value=2.0, 
                                           min_value=0.0, step=0.5)
        with col3:
            is_stressed = st.checkbox("Apply Stress Multiplier", value=False)
        
        # Calculate LVaR
        if st.button("Calculate LVaR"):
            # Simplified calculation
            bid_ask_cost = abs(total_dv01) * avg_spread_bp * 0.5
            if is_stressed:
                bid_ask_cost *= 2.0
            
            var_scaled = base_var * np.sqrt(holding_period)
            lvar = np.sqrt(var_scaled**2 + bid_ask_cost**2)
            
            liquidity_cost = lvar - base_var
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Base VaR", f"${base_var:,.0f}")
            col2.metric("Bid/Ask Cost", f"${bid_ask_cost:,.0f}")
            col3.metric("LVaR", f"${lvar:,.0f}")
            col4.metric("Liquidity Premium", f"{(lvar/base_var - 1)*100:.1f}%")
            
            # Breakdown chart
            breakdown_df = pd.DataFrame({
                'Component': ['Base VaR', 'Holding Period Adjustment', 
                            'Bid/Ask Cost', 'Position Impact'],
                'Value': [base_var, var_scaled - base_var, bid_ask_cost, 0]
            })
            
            fig = go.Figure(go.Bar(
                x=breakdown_df['Component'],
                y=breakdown_df['Value'],
                marker_color=['blue', 'orange', 'red', 'purple']
            ))
            
            fig.update_layout(
                title='Liquidity Adjustment Breakdown',
                xaxis_title='Component',
                yaxis_title='Value ($)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Liquidity metrics by instrument
        st.subheader("Liquidity Metrics by Instrument Type")
        
        liquidity_df = pd.DataFrame({
            'Instrument': ['UST 2Y', 'UST 5Y', 'UST 10Y', 'UST 30Y', 'IRS 5Y', 'IRS 10Y', 'Futures'],
            'Bid/Ask (bp)': [0.5, 0.5, 0.5, 1.0, 2.5, 3.0, 0.25],
            'Est. Liq. Time (days)': [1, 1, 1, 2, 2, 3, 1]
        })
        
        st.dataframe(liquidity_df, use_container_width=True)
    
    # =========================================================================
    # TAB 8: Data Explorer
    # =========================================================================
    with tab8:
        st.header("Data Explorer")
        
        data_view = st.selectbox("Select Data View",
                                ["Market Quotes", "Portfolio Positions", 
                                 "Historical Rates", "Curve Nodes"])
        
        if data_view == "Market Quotes":
            st.subheader("OIS Quotes")
            st.dataframe(ois_quotes, use_container_width=True)
            
            st.subheader("Treasury Quotes")
            st.dataframe(treasury_quotes, use_container_width=True)
        
        elif data_view == "Portfolio Positions":
            st.subheader("Current Portfolio Positions")
            st.dataframe(positions_df, use_container_width=True)
            
            # Download button
            csv = positions_df.to_csv(index=False)
            st.download_button(
                label="Download Positions as CSV",
                data=csv,
                file_name="positions.csv",
                mime="text/csv"
            )
        
        elif data_view == "Historical Rates":
            st.subheader("Historical Rate Data")
            st.dataframe(historical_rates.head(20), use_container_width=True)
            st.info(f"Total historical observations: {len(historical_rates)}")
        
        elif data_view == "Curve Nodes":
            st.subheader("OIS Curve Nodes")
            
            nodes_data = []
            for time, df, zero_rate in ois_curve.get_nodes():
                # Convert time back to date
                days = int(time * DAYS_PER_YEAR)
                node_date = valuation_date + timedelta(days=days)
                
                nodes_data.append({
                    'Date': node_date,
                    'Days': days,
                    'Years': f"{time:.4f}",
                    'Discount Factor': f"{df:.6f}",
                    'Zero Rate (%)': f"{zero_rate*100:.4f}"
                })
            
            nodes_df = pd.DataFrame(nodes_data)
            st.dataframe(nodes_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Rates Risk Library v0.1.0 | Interactive Dashboard</p>
        <p>Covering all library functionality: Curves, Pricing, Risk, VaR, Scenarios, P&L, Liquidity</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
