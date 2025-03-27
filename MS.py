#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 08:23:00 2025

@author: mathisdelooze
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime, timedelta
from textwrap import wrap
from fpdf import FPDF
import tempfile
import feedparser

# ===========================
# 1) CONFIGURATION DE LA PAGE
# ===========================
st.set_page_config(
    page_title="Fund Logic Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# ===========================
# 2) STYLES CSS PERSONNALISÉS
# ===========================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    body {
        font-family: 'Open Sans', sans-serif;
        background-color: #F8F9FA;
    }
    .header-container {
        background-color: #ffffff;
        padding: 20px 30px;
        border-bottom: 3px solid #00539b;
        margin-bottom: 20px;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .header-title {
        font-size: 32px;
        font-weight: 700;
        color: #00539b;
        margin: 0;
    }
    .header-subtitle {
        font-size: 14px;
        color: #888;
        margin: 0;
    }
    .analysis-title {
        font-size: 28px;
        font-weight: 700;
        color: #00539b;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===========================
# 3) EN-TÊTE MODERNISÉ
# ===========================
st.markdown(
    f"""
    <div class="header-container">
        <div class="header-left">
            <div>
                <h1 class="header-title">Morgan Stanley – Fund Logic</h1>
            </div>
        </div>
        <div style="text-align: right;">
            <p class="header-subtitle" style="margin-bottom: 4px;">
                Last Updated: {datetime.now().strftime('%B %d, %Y – %H:%M')}
            </p>
            <p class="header-subtitle" style="margin-top: 0;">
                Created by <a href="https://www.linkedin.com/in/mathis-de-looze/" target="_blank" style="color: #00539b; text-decoration: none; font-weight: 600;">
                    Mathis de Looze
                </a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ===========================
# 4) OPTIONS DE TICKERS
# ===========================
fund_options = {
    "S&P 500 ETF (SPY)": "SPY",
    "Nasdaq 100 ETF (QQQ)": "QQQ",
    "MSCI World ETF (URTH)": "URTH",
    "iShares EuroStoxx 50 (EXW1.DE)": "EXW1.DE",
    "iShares Clean Energy (ICLN)": "ICLN",
    "ARK Innovation ETF (ARKK)": "ARKK",
    "iShares 20+ Year Treasury Bond (TLT)": "TLT",
    "MS Inst. Growth Fund (MSGSX)": "MSGSX",
    "MS Global Opportunity Fund (MGGPX)": "MGGPX",
    "MS International Advantage Fund (MIAFX)": "MIAFX",
    "MS Developing Opportunity Fund (MDOFX)": "MDOFX"
}

benchmark_options = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq 100 (QQQ)": "QQQ",
    "MSCI World (URTH)": "URTH",
    "EuroStoxx 50 (FEZ)": "FEZ",
    "Global Agg Bonds (AGG)": "AGG",
    "10Y US Treasury (IEF)": "IEF"
}

selected_fund = st.sidebar.selectbox(
    "Select a fund",
    list(fund_options.keys()),
    index=list(fund_options.keys()).index("MS Inst. Growth Fund (MSGSX)")
)
selected_benchmark = st.sidebar.selectbox(
    "Select a benchmark",
    list(benchmark_options.keys()),
    index=list(benchmark_options.keys()).index("S&P 500 (SPY)")
)

manual_fund = st.sidebar.text_input("Or enter a custom fund ticker (optional)", "")
manual_benchmark = st.sidebar.text_input("Or enter a custom benchmark ticker (optional)", "")

fund_ticker = manual_fund.strip().upper() if manual_fund else fund_options[selected_fund]
benchmark_ticker = manual_benchmark.strip().upper() if manual_benchmark else benchmark_options[selected_benchmark]

# ===========================
# 5) DATE RANGE
# ===========================
st.sidebar.markdown("### Date Range")
end_default = datetime.today()
start_default = end_default - timedelta(days=5 * 365)

start_date = st.sidebar.date_input("Start date", start_default)
end_date = st.sidebar.date_input("End date", end_default)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if start_date >= end_date:
    st.sidebar.error("End date must be after start date.")
    st.stop()

# ===========================
# 6) BOUTON DE MISE À JOUR
# ===========================
st.sidebar.markdown("### Refresh Analysis")
st.sidebar.write("Click the button below to update all visuals and metrics.")
recalculate = st.sidebar.button("Update")

# ===========================
# 7) FONCTIONS UTILES & GLOBALES
# ===========================
def validate_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        st.sidebar.error(f"No data found for ticker: '{ticker}'. Please check the spelling.")
        return None
    return data

def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return pd.Series(dtype=float)
    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    price.name = ticker
    return price

def monthly_returns(df_prices):
    if df_prices.empty:
        return pd.Series(dtype=float)
    if isinstance(df_prices, pd.DataFrame):
        df_prices = df_prices.iloc[:, 0]
        df_prices.name = "Price"
    df_m = df_prices.resample("M").last()
    returns_m = df_m.pct_change().dropna()
    return returns_m

def performance_stats(returns_series):
    if returns_series.empty:
        return {}
    returns_series = returns_series.squeeze()
    freq = 12
    ann_return = float(returns_series.mean() * freq)
    ann_vol = float(returns_series.std() * np.sqrt(freq))
    cum_growth = (1 + returns_series).cumprod()
    best_date = pd.to_datetime(returns_series.idxmax())
    worst_date = pd.to_datetime(returns_series.idxmin())
    best_month = best_date.strftime("%B %Y")
    worst_month = worst_date.strftime("%B %Y")
    rolling_max = cum_growth.cummax()
    drawdown = (cum_growth - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": (ann_return / ann_vol) if ann_vol != 0 else np.nan,
        "best_month": best_month,
        "worst_month": worst_month,
        "cumulative_return": float(cum_growth.iloc[-1] - 1) if len(cum_growth) > 0 else np.nan,
        "max_drawdown": max_dd
    }

def compute_subperiod_stats(prices, start_sub, end_sub):
    sub_data = prices.loc[start_sub:end_sub].copy()
    if sub_data.empty:
        return {}
    sub_monthly = sub_data.resample("M").last().pct_change().dropna()
    return performance_stats(sub_monthly)

def calc_volatility_tail_risk_stats(fund_returns, bench_returns):
    fund_returns = fund_returns.dropna()
    bench_returns = bench_returns.dropna()
    def annualized_vol(ret): 
        return ret.std() * np.sqrt(12)
    def gain_vol(ret):
        pos_ret = ret[ret > 0]
        return pos_ret.std() * np.sqrt(12) if len(pos_ret) > 0 else 0
    def loss_vol(ret):
        neg_ret = ret[ret < 0]
        return neg_ret.std() * np.sqrt(12) if len(neg_ret) > 0 else 0
    def skewness(ret): 
        return ret.skew()
    def kurtosis(ret): 
        return ret.kurtosis()

    f_av = annualized_vol(fund_returns)
    b_av = annualized_vol(bench_returns)
    f_gv = gain_vol(fund_returns)
    b_gv = gain_vol(bench_returns)
    f_lv = loss_vol(fund_returns)
    b_lv = loss_vol(bench_returns)
    f_skew = skewness(fund_returns)
    b_skew = skewness(bench_returns)
    f_kurt = kurtosis(fund_returns)
    b_kurt = kurtosis(bench_returns)

    rows = [
        "Annualized Volatility",
        "Annualized Gain Volatility",
        "Annualized Loss Volatility",
        "Skewness",
        "Kurtosis"
    ]
    stats_vals = [
        [f_av, b_av],
        [f_gv, b_gv],
        [f_lv, b_lv],
        [f_skew, b_skew],
        [f_kurt, b_kurt]
    ]
    return pd.DataFrame(stats_vals, columns=["Fund", "Benchmark"], index=rows)

def style_volatility_df(df):
    pct_rows = ["Annualized Volatility", "Annualized Gain Volatility", "Annualized Loss Volatility"]
    other_rows = df.index.difference(pct_rows)
    styled = df.style.format("{:.2%}", subset=pd.IndexSlice[pct_rows, :])
    styled = styled.format("{:.2f}", subset=pd.IndexSlice[other_rows, :])
    styled = styled.set_properties(subset=["Fund"], **{'color': '#00539b', 'font-weight': 'bold'})\
                   .set_properties(subset=["Benchmark"], **{'color': '#000000', 'font-weight': 'normal'})\
                   .set_properties(**{'text-align': 'center'})
    styled = styled.set_table_styles([{'selector': '.index_name', 'props': [('font-weight', 'bold')]}])
    return styled

def efficiency_drawdown_stats(returns_series):
    """
    Calcule pour une série de rendements mensuels :
      - Average Monthly Gain (moyenne des gains positifs)
      - Average Monthly Loss (moyenne des pertes négatives)
      - Positive Months Fraction
      - Negative Months Fraction
      - Maximum Drawdown
    """
    if returns_series.empty:
        return {}
    pos = returns_series[returns_series > 0]
    neg = returns_series[returns_series < 0]
    avg_gain = pos.mean() if not pos.empty else 0
    avg_loss = neg.mean() if not neg.empty else 0
    pos_frac = len(pos) / len(returns_series)
    neg_frac = len(neg) / len(returns_series)
    cum = (1 + returns_series).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return {
        "Average Monthly Gain": avg_gain,
        "Average Monthly Loss": avg_loss,
        "Positive Months Fraction": pos_frac,
        "Negative Months Fraction": neg_frac,
        "Maximum Drawdown": max_dd
    }

def calc_efficiency_drawdown_stats(fund_returns, bench_returns):
    """
    Calcule et retourne un DataFrame avec les statistiques Efficiency and Drawdown pour Fund et Benchmark.
    """
    fund_stats = efficiency_drawdown_stats(fund_returns)
    bench_stats = efficiency_drawdown_stats(bench_returns)
    rows = [
        "Average Monthly Gain",
        "Average Monthly Loss",
        "Positive Months Fraction",
        "Negative Months Fraction",
        "Maximum Drawdown"
    ]
    data = [
        [fund_stats.get("Average Monthly Gain", np.nan), bench_stats.get("Average Monthly Gain", np.nan)],
        [fund_stats.get("Average Monthly Loss", np.nan), bench_stats.get("Average Monthly Loss", np.nan)],
        [fund_stats.get("Positive Months Fraction", np.nan), bench_stats.get("Positive Months Fraction", np.nan)],
        [fund_stats.get("Negative Months Fraction", np.nan), bench_stats.get("Negative Months Fraction", np.nan)],
        [fund_stats.get("Maximum Drawdown", np.nan), bench_stats.get("Maximum Drawdown", np.nan)]
    ]
    return pd.DataFrame(data, columns=["Fund", "Benchmark"], index=rows)

def style_efficiency_drawdown_df(df):
    pct_rows_2dec = ["Average Monthly Gain", "Average Monthly Loss", "Maximum Drawdown"]
    pct_rows_0dec = ["Positive Months Fraction", "Negative Months Fraction"]
    styled = df.style.format("{:.2%}", subset=pd.IndexSlice[pct_rows_2dec, :])
    styled = styled.format("{:.0%}", subset=pd.IndexSlice[pct_rows_0dec, :])
    styled = styled.set_properties(subset=["Fund"], **{'color': '#00539b', 'font-weight': 'bold'})\
                   .set_properties(subset=["Benchmark"], **{'color': '#000000', 'font-weight': 'normal'})\
                   .set_properties(**{'text-align': 'center'})
    styled = styled.set_table_styles([{'selector': '.index_name', 'props': [('font-weight', 'bold')]}])
    return styled

# Fonctions pour le calcul des ratios de risque
def compute_sortino_ratio(returns, target=0):
    """Calcule le ratio Sortino à partir d'une série de rendements."""
    downside = returns[returns < target]
    downside_std = downside.std() * np.sqrt(12)
    mean_ret = returns.mean() * 12
    return mean_ret / downside_std if downside_std != 0 else np.nan

def compute_treynor_ratio(portfolio_ret, benchmark_ret):
    """Calcule le ratio Treynor à partir des rendements du portefeuille et du benchmark."""
    if benchmark_ret.empty:
        return np.nan
    common_index = portfolio_ret.index.intersection(benchmark_ret.index)
    p_ret = portfolio_ret.loc[common_index]
    b_ret = benchmark_ret.loc[common_index]
    if len(p_ret) < 2:
        return np.nan
    beta = np.cov(p_ret, b_ret)[0, 1] / np.var(b_ret)
    return (p_ret.mean() * 12) / beta if beta != 0 else np.nan

def compute_summary_metrics(prices):
    monthly = monthly_returns(prices)
    cum_returns = (1 + monthly).cumprod()
    drawdowns = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
    rolling_vol = monthly.rolling(window=12).std() * np.sqrt(12)
    return {
        "cumulative_returns": cum_returns,
        "drawdowns": drawdowns,
        "rolling_volatility": rolling_vol
    }

def plot_cumulative_and_drawdowns(metrics_fund, metrics_bench):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_fund["cumulative_returns"].index,
        y=metrics_fund["cumulative_returns"] - 1,
        mode="lines",
        name="Fund Cumulative Return",
        line=dict(color="#00539b")
    ))
    fig.add_trace(go.Scatter(
        x=metrics_bench["cumulative_returns"].index,
        y=metrics_bench["cumulative_returns"] - 1,
        mode="lines",
        name="Benchmark Cumulative Return",
        line=dict(color="#555555")
    ))
    fig.add_trace(go.Scatter(
        x=metrics_fund["drawdowns"].index,
        y=metrics_fund["drawdowns"],
        mode="lines",
        name="Fund Drawdown",
        line=dict(color="red"),
        fill="tozeroy", opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=metrics_bench["drawdowns"].index,
        y=metrics_bench["drawdowns"],
        mode="lines",
        name="Benchmark Drawdown",
        line=dict(color="darkred", dash="dot", width=2),
        fill="tozeroy", opacity=0.5
    ))
    fig.update_layout(
         title="Cumulative Returns & Drawdowns",
         xaxis_title=None,
         yaxis_title="Return / Drawdown",
         height=400,
         legend=dict(orientation="h", yanchor="bottom", y=0.95, xanchor="center", x=0.5)
    )
    return fig

def plot_rolling_volatility(metrics_fund, metrics_bench):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_fund["rolling_volatility"].index,
        y=metrics_fund["rolling_volatility"],
        mode="lines",
        name="Fund Rolling Volatility",
        line=dict(color="#00539b")
    ))
    fig.add_trace(go.Scatter(
        x=metrics_bench["rolling_volatility"].index,
        y=metrics_bench["rolling_volatility"],
        mode="lines",
        name="Benchmark Rolling Volatility",
        line=dict(color="#555555")
    ))
    fig.update_layout(
         title="12-Month Rolling Volatility",
         xaxis_title=None,
         yaxis_title="Annualized Volatility",
         height=400,
         yaxis=dict(tickformat=".0%"),
         legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    return fig

def plot_return_distribution(returns_series):
    data = returns_series.dropna().values
    if len(data) == 0:
        return None
    data_perc = data * 100
    mean_perc = data_perc.mean()
    std_perc = data_perc.std()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data_perc,
        nbinsx=20,
        name="Returns distribution",
        marker_color="#00539b",
        opacity=0.75
    ))
    x_min, x_max = data_perc.min(), data_perc.max()
    x_vals = np.linspace(x_min, x_max, 200)
    n = len(data_perc)
    bin_size = (x_max - x_min) / 20
    z = (x_vals - mean_perc) / std_perc
    pdf = (1 / (std_perc * np.sqrt(2*np.pi))) * np.exp(-0.5 * z**2)
    scaled_pdf = n * bin_size * pdf

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=scaled_pdf,
        mode="lines",
        name="Normal distribution",
        line=dict(color="black")
    ))
    fig.update_layout(
        barmode="overlay",
        title="Monthly Returns Distribution",
        xaxis=dict(title=None, tickformat=".0f", ticksuffix="%"),
        yaxis=dict(title="Number of Months"),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        height=350
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white")
    return fig

def plot_cumulative_comparison(fund_cum, bench_cum):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fund_cum.index,
        y=fund_cum - 1,
        mode="lines",
        name="Fund",
        line=dict(color="#00539b")
    ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index,
        y=bench_cum - 1,
        mode="lines",
        name="Benchmark",
        line=dict(color="#555555")
    ))
    fig.update_layout(
        title="Cumulative Growth",
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5),
        height=350
    )
    return fig

def plot_monthly_bar_ms(returns_series):
    if isinstance(returns_series, pd.Series):
        df_plot = returns_series.to_frame(name="Monthly Return")
    else:
        df_plot = returns_series.copy()
        if len(df_plot.columns) == 1:
            df_plot.columns = ["Monthly Return"]
        else:
            df_plot = df_plot.iloc[:, [0]]
            df_plot.columns = ["Monthly Return"]

    df_plot["Month"] = df_plot.index.strftime("%Y-%m")
    fig = px.bar(
        df_plot,
        x="Month",
        y="Monthly Return",
        color_discrete_sequence=["#00539b"]
    )
    fig.update_layout(
        title="Monthly Returns",
        xaxis=dict(tickangle=45, title=None),
        yaxis=dict(title=None),
        height=350
    )
    fig.update_yaxes(tickformat=".2%")
    return fig

def generate_pdf_report(fig, df_abs_stats, df_metrics, weights_dict, ticker_to_name, logo_path="ms_logo.png"):
    def clean_text(text):
        return str(text).replace("–", "-").replace("’", "'").replace("‘", "'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        pio.write_image(fig, tmpfile.name, format='png', width=1200, height=800)
        img_path = tmpfile.name

    pdf = FPDF()
    pdf.add_page()
    try:
        pdf.image(logo_path, x=165, y=8, w=35)
    except:
        pass

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Portfolio Analysis Report", ln=True, align="C")

    pdf.set_font("Arial", "", 10)
    date_str = datetime.now().strftime("Generated on %B %d, %Y")
    pdf.cell(0, 10, date_str, ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    composition_lines = [
        f"{round(weight * 100, 2)}% {ticker_to_name.get(ticker, ticker)}"
        for ticker, weight in weights_dict.items()
    ]
    composition_text = "The portfolio is composed of: " + ", ".join(composition_lines) + "."
    for line in wrap(composition_text, 100):
        pdf.cell(0, 8, line, ln=True)
    pdf.ln(8)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "1. Absolute Return Statistics", ln=True)
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(80, 8, clean_text(df_abs_stats.index.name or "Period"), border=1, fill=True)
    pdf.cell(40, 8, "Portfolio", border=1, fill=True)
    pdf.cell(40, 8, "Benchmark", border=1, fill=True)
    pdf.ln(8)

    pdf.set_font("Arial", "", 10)
    for idx, row in df_abs_stats.iterrows():
        pdf.cell(80, 8, clean_text(idx), border=1)
        pdf.cell(
            40, 8,
            f"{row['Portfolio']:.2%}" if isinstance(row['Portfolio'], (float, np.floating)) else clean_text(row['Portfolio']),
            border=1
        )
        pdf.cell(
            40, 8,
            f"{row['Benchmark']:.2%}" if isinstance(row['Benchmark'], (float, np.floating)) else clean_text(row['Benchmark']),
            border=1
        )
        pdf.ln(8)
    pdf.ln(8)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "2. Cumulative Returns & Drawdowns", ln=True)
    pdf.image(img_path, x=10, y=pdf.get_y() + 5, w=190)
    pdf.ln(115)

    pdf.add_page()
    try:
        pdf.image(logo_path, x=165, y=8, w=35)
    except:
        pass

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "3. Relative & Risk-Adjusted Ratios", ln=True)
    pdf.ln(5)
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", "B", 10)
    pdf.cell(80, 8, "Metric", border=1, fill=True)
    pdf.cell(40, 8, "Portfolio", border=1, fill=True)
    pdf.cell(40, 8, "Benchmark", border=1, fill=True)
    pdf.ln(8)

    pdf.set_font("Arial", "", 10)
    for idx, row in df_metrics.iterrows():
        pdf.cell(80, 8, clean_text(idx), border=1)
        pdf.cell(40, 8, clean_text(row["Portfolio"]), border=1)
        pdf.cell(40, 8, clean_text(row["Benchmark"]), border=1)
        pdf.ln(8)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
        with open(tmp_pdf.name, "rb") as f:
            st.download_button("📄 Download Full PDF Report", data=f.read(), file_name="portfolio_report.pdf")

def plot_cumulative_and_drawdowns_portfolio(metrics_port, metrics_bench):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=metrics_port["cumulative_returns"].index,
        y=metrics_port["cumulative_returns"] - 1,
        mode="lines",
        name="Portfolio Cumulative Return",
        line=dict(color="#00539b")
    ))
    fig.add_trace(go.Scatter(
        x=metrics_bench["cumulative_returns"].index,
        y=metrics_bench["cumulative_returns"] - 1,
        mode="lines",
        name="Benchmark Cumulative Return",
        line=dict(color="#555555")
    ))
    fig.add_trace(go.Scatter(
        x=metrics_port["drawdowns"].index,
        y=metrics_port["drawdowns"],
        mode="lines",
        name="Portfolio Drawdown",
        line=dict(color="red"),
        fill="tozeroy", opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=metrics_bench["drawdowns"].index,
        y=metrics_bench["drawdowns"],
        mode="lines",
        name="Benchmark Drawdown",
        line=dict(color="darkred", dash="dot", width=2),
        fill="tozeroy", opacity=0.5
    ))
    fig.update_layout(
         title="Cumulative Returns & Drawdowns",
         xaxis_title=None,
         yaxis_title="Return / Drawdown",
         height=380,
         legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="center", x=0.5)
    )
    return fig

def get_news_rss(feed_url, limit=8):
    feed = feedparser.parse(feed_url)
    return feed.entries[:limit]

# ===========================
# 8) INITIALISATION DE SESSION_STATE
# ===========================
default_fund = "MSGSX"          # Choix par défaut pour les fonds
default_benchmark = "URTH"      # Choix par défaut pour le benchmark

if "fund_prices" not in st.session_state:
    st.session_state["fund_prices"] = pd.Series(dtype=float)
if "bench_prices" not in st.session_state:
    st.session_state["bench_prices"] = pd.Series(dtype=float)
if "fund_monthly_ret" not in st.session_state:
    st.session_state["fund_monthly_ret"] = pd.Series(dtype=float)
if "bench_monthly_ret" not in st.session_state:
    st.session_state["bench_monthly_ret"] = pd.Series(dtype=float)
if "fund_stats" not in st.session_state:
    st.session_state["fund_stats"] = {}
if "bench_stats" not in st.session_state:
    st.session_state["bench_stats"] = {}

if "initialized" not in st.session_state:
    default_fund_data = validate_data(default_fund, start_default, end_default)
    default_bench_data = validate_data(default_benchmark, start_default, end_default)
    if default_fund_data is not None and default_bench_data is not None:
        st.session_state["fund_prices"] = default_fund_data["Adj Close"] if "Adj Close" in default_fund_data else default_fund_data["Close"]
        st.session_state["fund_prices"].name = default_fund
        st.session_state["bench_prices"] = default_bench_data["Adj Close"] if "Adj Close" in default_bench_data else default_bench_data["Close"]
        st.session_state["bench_prices"].name = default_benchmark

        st.session_state["fund_monthly_ret"] = monthly_returns(st.session_state["fund_prices"])
        st.session_state["bench_monthly_ret"] = monthly_returns(st.session_state["bench_prices"])
        st.session_state["fund_stats"] = performance_stats(st.session_state["fund_monthly_ret"])
        st.session_state["bench_stats"] = performance_stats(st.session_state["bench_monthly_ret"])
        st.session_state["initialized"] = True
    else:
        st.stop()

if recalculate:
    fund_data = validate_data(fund_ticker, start_date, end_date)
    bench_data = validate_data(benchmark_ticker, start_date, end_date)
    if fund_data is not None and bench_data is not None:
        st.session_state["fund_prices"] = fund_data["Adj Close"] if "Adj Close" in fund_data else fund_data["Close"]
        st.session_state["fund_prices"].name = fund_ticker
        st.session_state["bench_prices"] = bench_data["Adj Close"] if "Adj Close" in bench_data else bench_data["Close"]
        st.session_state["bench_prices"].name = benchmark_ticker

        st.session_state["fund_monthly_ret"] = monthly_returns(st.session_state["fund_prices"])
        st.session_state["bench_monthly_ret"] = monthly_returns(st.session_state["bench_prices"])
        st.session_state["fund_stats"] = performance_stats(st.session_state["fund_monthly_ret"])
        st.session_state["bench_stats"] = performance_stats(st.session_state["bench_monthly_ret"])

fund_prices = st.session_state.get("fund_prices", pd.Series(dtype=float))
bench_prices = st.session_state.get("bench_prices", pd.Series(dtype=float))
fund_monthly_ret = st.session_state.get("fund_monthly_ret", pd.Series(dtype=float))
bench_monthly_ret = st.session_state.get("bench_monthly_ret", pd.Series(dtype=float))
fund_stats = st.session_state.get("fund_stats", {})
bench_stats = st.session_state.get("bench_stats", {})

# ===========================
# 9) ONGLETS PRINCIPAUX
# ===========================
main_tabs = st.tabs([
    "Absolute Risk & Return", 
    "Relative & Risk-Adjusted Stats", 
    "Correlation & Regression",
    "Portfolio Simulator",
    "Market News"
])

# --------------------------------
# Onglet 1: Absolute Risk & Return
# --------------------------------
with main_tabs[0]:
    inception_date_fund = fund_prices.index[0] if not fund_prices.empty else start_date
    inception_date_bench = bench_prices.index[0] if not bench_prices.empty else start_date
    inception_date = max(inception_date_fund, inception_date_bench)
    end = end_date
    last_5y = end - timedelta(days=5*365)
    last_3y = end - timedelta(days=3*365)
    last_1y = end - timedelta(days=1*365)

    def safe_stats(prices, start_sub, end_sub):
        return compute_subperiod_stats(prices, start_sub, end_sub) if not prices.empty else {}

    fund_stats_inception = safe_stats(fund_prices, inception_date, end)
    fund_stats_5y = safe_stats(fund_prices, last_5y, end)
    fund_stats_3y = safe_stats(fund_prices, last_3y, end)
    fund_stats_1y = safe_stats(fund_prices, last_1y, end)

    bench_stats_inception = safe_stats(bench_prices, inception_date, end)
    bench_stats_5y = safe_stats(bench_prices, last_5y, end)
    bench_stats_3y = safe_stats(bench_prices, last_3y, end)
    bench_stats_1y = safe_stats(bench_prices, last_1y, end)

    best_month_fund = fund_stats_inception.get("best_month", float('nan'))
    worst_month_fund = fund_stats_inception.get("worst_month", float('nan'))
    best_month_bench = bench_stats_inception.get("best_month", float('nan'))
    worst_month_bench = bench_stats_inception.get("worst_month", float('nan'))

    row_labels = [
        "Annualized Return – Since inception",
        "Annualized Return – Last 5 years",
        "Annualized Return – Last 3 years",
        "Annualized Return – Last 1 year",
        "Best Month",
        "Worst Month"
    ]
    
    stats_df = pd.DataFrame(
        [
            [fund_stats_inception.get('ann_return', 0),   bench_stats_inception.get('ann_return', 0)],
            [fund_stats_5y.get('ann_return', 0),          bench_stats_5y.get('ann_return', 0)],
            [fund_stats_3y.get('ann_return', 0),          bench_stats_3y.get('ann_return', 0)],
            [fund_stats_1y.get('ann_return', 0),          bench_stats_1y.get('ann_return', 0)],
            [best_month_fund,                             best_month_bench],
            [worst_month_fund,                            worst_month_bench]
        ],
        columns=["Fund", "Benchmark"],
        index=row_labels
    )
    stats_df.index.name = "Absolute Return Statistics"

    # ADDED: Stocker le DataFrame complet dans session_state pour le réutiliser dans l’onglet 4
    st.session_state["abs_stats_df"] = stats_df.copy()  # <--- AJOUTÉ

    stats_df_styled = stats_df.style.format({
        "Fund": lambda x: "{:.2%}".format(x) if isinstance(x, (float, np.floating)) else x,
        "Benchmark": lambda x: "{:.2%}".format(x) if isinstance(x, (float, np.floating)) else x,
    }).set_properties(subset=["Fund"], **{'color': '#00539b', 'font-weight': 'bold'})\
      .set_properties(subset=["Benchmark"], **{'color': '#000000', 'font-weight': 'normal'})\
      .set_properties(**{'text-align': 'center'})\
      .set_table_styles([{'selector': 'th.col_heading.level0.col0', 'props': [('color', '#00539b'), ('font-weight', 'bold')]}], overwrite=False)
      
    def get_cum(prices):
        if prices.empty:
            return pd.Series(dtype=float)
        monthly = monthly_returns(prices.loc[start_date:end_date])
        cum = (1 + monthly).cumprod()
        if isinstance(cum, pd.DataFrame) and not cum.empty:
            cum = cum.iloc[:, 0]
        return cum

    fund_cum = get_cum(fund_prices)
    bench_cum = get_cum(bench_prices)

    if not fund_cum.empty and not bench_cum.empty:
        fig_compare = plot_cumulative_comparison(fund_cum, bench_cum)
    else:
        fig_compare = None

    # On stocke aussi bench_monthly_ret et bench_cum pour d'autres usages
    st.session_state["bench_monthly_ret"] = bench_monthly_ret
    st.session_state["bench_cum"] = bench_cum

    container_top = st.container()
    with container_top:
        left_col, right_col = st.columns([1.7, 2])
        with left_col:
            st.markdown("    ")
            st.markdown("    ")
            st.write(stats_df_styled)
        with right_col:
            if fig_compare:
                st.plotly_chart(fig_compare, use_container_width=True, key="cumulative_chart")
            else:
                st.warning("Not enough data to plot Cumulative Growth.")
    st.markdown("---")

    if fund_monthly_ret.empty:
        st.warning("Not enough data for Fund monthly returns.")
    else:
        fig_fund_monthly = plot_monthly_bar_ms(fund_monthly_ret)
        st.plotly_chart(fig_fund_monthly, use_container_width=True, key="monthly_chart")

    st.markdown("---")
    if not fund_prices.empty and not bench_prices.empty:
        fund_summary = compute_summary_metrics(fund_prices)
        bench_summary = compute_summary_metrics(bench_prices)
        fig_returns = plot_cumulative_and_drawdowns(fund_summary, bench_summary)
        fig_vol = plot_rolling_volatility(fund_summary, bench_summary)
        col1, col_sep, col2 = st.columns([2.45, 0.1, 2])
        with col1:
            st.plotly_chart(fig_returns, use_container_width=True, key="returns_chart")
        with col_sep:
            st.markdown("<div style='height:100%; border-left:1px solid #ccc;'></div>", unsafe_allow_html=True)
        with col2:
            st.plotly_chart(fig_vol, use_container_width=True, key="vol_chart")
    else:
        st.warning("Not enough data for performance summary.")
        
    st.markdown("---")
    vol_df = calc_volatility_tail_risk_stats(fund_monthly_ret, bench_monthly_ret)
    vol_df.index.name = "Volatility and Tail Risk"
    vol_styled = style_volatility_df(vol_df)
    
    eff_df = calc_efficiency_drawdown_stats(fund_monthly_ret, bench_monthly_ret)
    eff_df.index.name = "Efficiency and Drawdown"
    eff_styled = style_efficiency_drawdown_df(eff_df)
    
    container_stats = st.container()
    with container_stats:
        col1, spacer, col2 = st.columns([4, 0.2, 2])
        with col1:
            st.markdown(
                """
                <div style="display: flex; justify-content: left; align-items: center; gap: 12px; margin-bottom: 10px;">
                    <div style="font-weight: 600; font-size: 16px;">Select distribution to display:</div>
                    <div id="radio-placeholder"></div>
                </div>
                """, unsafe_allow_html=True
            )
            selected_data = st.radio(
                label="",
                options=["Fund", "Benchmark"],
                horizontal=True,
                label_visibility="collapsed",
                key="distribution_radio"
            )
            st.markdown("---")
            fig_dist = None
            if selected_data == "Fund" and not fund_monthly_ret.empty:
                fig_dist = plot_return_distribution(fund_monthly_ret)
            elif selected_data == "Benchmark" and not bench_monthly_ret.empty:
                fig_dist = plot_return_distribution(bench_monthly_ret)
            if fig_dist is not None:
                fig_dist.update_layout(
                    margin=dict(l=20, r=20, t=30, b=0),
                    autosize=True,
                    legend=dict(orientation="v", yanchor="top", y=1.1, xanchor="right", x=1)
                )
                st.plotly_chart(fig_dist, use_container_width=True, key="distribution_chart")
            else:
                st.warning("Not enough data for distribution plot.")
        with col2:
            st.write(vol_styled)
            st.markdown("---")
            st.write(eff_styled)
    st.markdown("---")

# --------------------------------
# Onglet 2: Relative & Risk-Adjusted Stats
# --------------------------------
with main_tabs[1]:
    def compute_relative_and_risk_adjusted_stats(fund_ret, bench_ret, rf_rate=0.0):
        if fund_ret.empty or bench_ret.empty:
            return pd.DataFrame()
        fund_ret.name = "Fund"
        bench_ret.name = "Benchmark"
        aligned = pd.concat([fund_ret, bench_ret], axis=1).dropna()
        if aligned.empty:
            return pd.DataFrame()
        f = aligned["Fund"]
        b = aligned["Benchmark"]

        # Annualized
        excess_ret = (f - b).mean() * 12

        # Sharpe
        sharpe_f = (f.mean() - rf_rate / 12) / f.std() * np.sqrt(12) if f.std() != 0 else np.nan
        sharpe_b = (b.mean() - rf_rate / 12) / b.std() * np.sqrt(12) if b.std() != 0 else np.nan

        # Sortino
        downside_std_f = f[f < rf_rate / 12].std()
        downside_std_b = b[b < rf_rate / 12].std()
        sortino_f = (f.mean() - rf_rate / 12) / downside_std_f * np.sqrt(12) if downside_std_f != 0 else np.nan
        sortino_b = (b.mean() - rf_rate / 12) / downside_std_b * np.sqrt(12) if downside_std_b != 0 else np.nan

        # Beta & Treynor
        beta = np.cov(f, b)[0, 1] / np.var(b) if np.var(b) != 0 else np.nan
        treynor_f = (f.mean() * 12 - rf_rate) / beta if beta != 0 else np.nan
        treynor_b = np.nan

        # Calmar
        cum_f = (1 + f).cumprod()
        max_dd_f = ((cum_f - cum_f.cummax()) / cum_f.cummax()).min()
        calmar_f = (f.mean() * 12) / abs(max_dd_f) if max_dd_f != 0 else np.nan

        cum_b = (1 + b).cumprod()
        max_dd_b = ((cum_b - cum_b.cummax()) / cum_b.cummax()).min()
        calmar_b = (b.mean() * 12) / abs(max_dd_b) if max_dd_b != 0 else np.nan

        # Outperform ratio
        outperf_ratio = (f.align(b, join='inner')[0] > f.align(b, join='inner')[1]).mean()

        metrics = {
            "Excess Return": [excess_ret, np.nan],
            "Sharpe Ratio": [sharpe_f, sharpe_b],
            "Sortino Ratio": [sortino_f, sortino_b],
            "Treynor Ratio": [treynor_f, treynor_b],
            "Calmar Ratio": [calmar_f, calmar_b],
            "Outperform Ratio": [outperf_ratio, np.nan]
        }
        return pd.DataFrame(metrics, index=["Fund", "Benchmark"]).T

    def rolling_excess_return(fund_ret, bench_ret, window=12):
        return (fund_ret - bench_ret).rolling(window).mean() * 12

    if fund_monthly_ret.empty or bench_monthly_ret.empty:
        st.warning("Not enough data to compute statistics.")
    else:
        stats_df = compute_relative_and_risk_adjusted_stats(fund_monthly_ret, bench_monthly_ret)
        stats_df.index.name = "Relative & Risk-Adjusted Stats"

        styled_df = stats_df.style \
            .format("{:.2%}", subset=pd.IndexSlice[["Excess Return", "Outperform Ratio"], :]) \
            .format("{:.2f}", subset=pd.IndexSlice[["Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Calmar Ratio"], :]) \
            .set_properties(subset=["Fund"], **{'color': '#00539b', 'font-weight': 'bold'}) \
            .set_properties(subset=["Benchmark"], **{'color': '#000000'}) \
            .set_properties(**{'text-align': 'center'}) \
            .set_table_styles([{'selector': '.index_name', 'props': [('font-weight', 'bold')]}])

        rolling_excess = rolling_excess_return(fund_monthly_ret, bench_monthly_ret)
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=rolling_excess.index,
            y=rolling_excess.rolling(3).mean(),
            mode="lines",
            name="12M Rolling Excess Return",
            line=dict(color="#00539b")
        ))
        fig_roll.update_layout(
            height=370,
            yaxis=dict(tickformat=".0%"),
            title="12 Month Rolling Excess Returns",
            margin=dict(l=10, r=10, t=30, b=60),
            showlegend=False
        )

        container_rel = st.container()
        with container_rel:
            left_col, right_col = st.columns([1.5, 2.5])
            with left_col:
                st.markdown("  ")
                st.markdown("  ")
                st.write(styled_df)

                csv_download = stats_df.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Metrics (.csv)",
                    data=csv_download,
                    file_name="relative_risk_adjusted_stats.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with right_col:
                st.plotly_chart(fig_roll, use_container_width=True, key="rolling_excess_chart")

    st.markdown("---")
    
    perf_diff = fund_monthly_ret - bench_monthly_ret
    perf_labels = np.where(perf_diff > 0, "Outperformance", "Underperformance")
    perf_counts = pd.Series(perf_labels).value_counts()
    fig_bar = px.bar(
        perf_counts,
        x=perf_counts.index,
        y=perf_counts.values,
        labels={'x': 'Performance Type', 'y': 'Number of Months'},
        title="Monthly Outperformance vs Underperformance"
    )

# --------------------------------
# Onglet 3: Correlation & Regression
# --------------------------------
with main_tabs[2]:
    if fund_monthly_ret.empty or bench_monthly_ret.empty:
        st.warning("Not enough data to compute correlation/regression.")
    else:
        merged = pd.concat([fund_monthly_ret, bench_monthly_ret], axis=1).dropna()
        merged.columns = ["Fund", "Benchmark"]

        correlation = merged.corr().iloc[0, 1]
        upside_corr = merged[merged["Benchmark"] > 0].corr().iloc[0, 1]
        downside_corr = merged[merged["Benchmark"] < 0].corr().iloc[0, 1]

        X = sm.add_constant(merged["Benchmark"])
        y = merged["Fund"]
        model = sm.OLS(y, X).fit()
        alpha = model.params["const"] * 12
        beta = model.params["Benchmark"]
        r_squared = model.rsquared

        tracking_error = (merged["Fund"] - merged["Benchmark"]).std() * np.sqrt(12)

        reg_df = pd.DataFrame({
            "Correlation and Regression Analysis": [
                "Correlation",
                "Upside Correlation",
                "Downside Correlation",
                "Annualised Alpha",
                "Beta",
                "R-Squared",
                "Annualized Tracking Error"
            ],
            "Fund / Benchmark": [
                f"{correlation:.2f}",
                f"{upside_corr:.2f}",
                f"{downside_corr:.2f}",
                f"{alpha:.2f}",
                f"{beta:.2f}",
                f"{r_squared:.2f}",
                f"{tracking_error:.2%}"
            ]
        })
        reg_df.set_index("Correlation and Regression Analysis", inplace=True)

        reg_df_styled = reg_df.style \
            .set_properties(subset=["Fund / Benchmark"], **{"color": "#00539b", "font-weight": "bold"}) \
            .set_properties(**{"text-align": "center"}) \
            .set_table_styles([{'selector': '.index_name', 'props': [('font-weight', 'bold')]}])

        rolling_corr = fund_monthly_ret.rolling(12).corr(bench_monthly_ret)
        fig_corr = go.Figure()
        fig_corr.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr,
            mode="lines",
            name="12M Rolling Correlation",
            line=dict(color="#00539b")
        ))
        fig_corr.update_layout(
            height=300,
            title="12 Month Rolling Fund-Benchmark Correlation",
            yaxis=dict(range=[-1, 1]),
            margin=dict(l=10, r=10, t=30, b=60),
            showlegend=False
        )

        container_corr = st.container()
        with container_corr:
            left_col, right_col = st.columns([1.5, 2.5])
            with left_col:
                st.write(reg_df_styled)
            with right_col:
                st.plotly_chart(fig_corr, use_container_width=True, key="rolling_corr_chart")

        comparison_indices = {
            "S&P 500 Growth": "SPYG",
            "Russell 2000": "IWM",
            "FTSE 100": "ISF.L",
            "MSCI ACWI": "ACWI",
            "MSCI EM Asia": "EEMA"
        }
        correlation_results = {}
        for label, ticker in comparison_indices.items():
            index_prices = load_data(ticker, start_date, end_date)
            if not index_prices.empty:
                index_returns = monthly_returns(index_prices)
                common_data = pd.concat([fund_monthly_ret, index_returns], axis=1).dropna()
                if len(common_data) > 0:
                    corr_val = common_data.iloc[:, 0].corr(common_data.iloc[:, 1])
                    correlation_results[label] = corr_val

        radar_df = pd.DataFrame(dict(
            r=list(correlation_results.values()),
            theta=list(correlation_results.keys())
        ))
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["r"],
            theta=radar_df["theta"],
            fill='toself',
            name='Fund Correlation',
            line=dict(color="#00539b"),
            fillcolor="rgba(0, 83, 155, 0.3)"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Correlations to Different Indices",
            height=370
        )

        fig_scatter = px.scatter(
            merged,
            x="Benchmark",
            y="Fund",
            trendline="ols",
            labels={"Fund": "Fund", "Benchmark": "Benchmark"},
            title="Fund vs. Benchmark Monthly Returns"
        )
        fig_scatter.update_layout(
            height=370,
            yaxis_tickformat=".2%",
            xaxis_tickformat=".2%"
        )

        st.markdown("---")
        col_radar, col_scatter = st.columns([2, 2.5])
        with col_radar:
            st.plotly_chart(fig_radar, use_container_width=True, key="radar_chart")
        with col_scatter:
            st.plotly_chart(fig_scatter, use_container_width=True, key="scatter_fund_bench")

        st.markdown("---")

# --------------------------------
# Onglet 4: Portfolio Simulator
# --------------------------------
with main_tabs[3]:
    st.markdown("<h3 style='color:#00539b'>Portfolio Simulator</h3>", unsafe_allow_html=True)
    st.markdown("Define your custom asset allocation and compare it to your selected benchmark.")

    available_assets = {
        "S&P 500": "SPY",
        "Nasdaq 100": "QQQ",
        "EuroStoxx 50": "FEZ",
        "10Y Treasury": "IEF",
        "MSCI World": "URTH",
        "Emerging Markets": "EEM",
        "Clean Energy": "ICLN",
        "Gold": "GLD",
        "Crude Oil": "USO"
    }

    st.markdown("#### Select assets for your portfolio:")
    selected_assets = st.multiselect(
        label=" ",
        options=list(available_assets.keys()),
        default=["S&P 500", "10Y Treasury", "Gold"],
        key="portfolio_assets"
    )

    st.markdown("#### Weight Allocation (%)")
    weights = {}
    total_weight = 0
    if selected_assets:
        rows = (len(selected_assets) - 1) // 4 + 1
        for row in range(rows):
            cols = st.columns(4)
            for i in range(4):
                idx = row * 4 + i
                if idx < len(selected_assets):
                    asset = selected_assets[idx]
                    ticker = available_assets[asset]
                    with cols[i]:
                        weight = st.number_input(
                            label=f"{asset}",
                            min_value=0.0,
                            max_value=100.0,
                            value=round(100 / len(selected_assets), 2),
                            key=f"weight_{ticker}"
                        )
                        weights[ticker] = weight / 100
                        total_weight += weights[ticker]
    if abs(total_weight - 1) > 0.01:
        st.error("Total weights must sum to 100%.")
        st.stop()

    price_data = {}
    for name, ticker in available_assets.items():
        if ticker in weights:
            prices = load_data(ticker, start_date, end_date)
            if not prices.empty:
                price_data[ticker] = monthly_returns(prices)

    if len(price_data) < 2:
        st.warning("Need at least 2 valid assets to compute portfolio.")
        st.stop()
        
    st.markdown("---")
    df_returns = pd.concat(price_data.values(), axis=1)
    df_returns.columns = list(price_data.keys())
    df_returns.dropna(inplace=True)

    portfolio_ret = sum(weights[t] * df_returns[t] for t in weights)
    portfolio_cum = (1 + portfolio_ret).cumprod()

    # Ici, on NE recalcule PAS le benchmark : on le reprend depuis la session_state
    benchmark_ret = st.session_state["bench_monthly_ret"].reindex(df_returns.index).dropna()
    benchmark_cum = (1 + benchmark_ret).cumprod() if not benchmark_ret.empty else None

    end = end_date
    last_5y = end - timedelta(days=5*365)
    last_3y = end - timedelta(days=3*365)
    last_1y = end - timedelta(days=1*365)

    def compute_subperiod_stats_new(returns, start, end):
        sub = returns.loc[start:end]
        if sub.empty:
            return {}
        ann_return = sub.mean() * 12
        best = sub.idxmax().strftime("%B %Y")
        worst = sub.idxmin().strftime("%B %Y")
        return {"ann_return": ann_return, "best_month": best, "worst_month": worst}

    # Calcul pour le portfolio
    portfolio_stats_incep = compute_subperiod_stats_new(portfolio_ret, portfolio_ret.index[0], end)
    portfolio_stats_5y = compute_subperiod_stats_new(portfolio_ret, last_5y, end)
    portfolio_stats_3y = compute_subperiod_stats_new(portfolio_ret, last_3y, end)
    portfolio_stats_1y = compute_subperiod_stats_new(portfolio_ret, last_1y, end)

    # REMOVED: plus besoin de recalculer le benchmark ici
    # On récupère directement les stats du benchmark dans st.session_state["abs_stats_df"]
    if "abs_stats_df" in st.session_state:
        # On récupère les valeurs du benchmark telles qu'elles ont été calculées dans l’onglet 1
        abs_stats_bench = st.session_state["abs_stats_df"]["Benchmark"]
        # abs_stats_bench est une Series avec par ex. :
        #   Annualized Return – Since inception    0.1687
        #   Annualized Return – Last 5 years      0.1687
        #   ...
        #   Best Month                            November 2020
        #   Worst Month                           September 2022
    else:
        # Si jamais absent (cas rare), on met N/A
        abs_stats_bench = pd.Series({
            "Annualized Return – Since inception": np.nan,
            "Annualized Return – Last 5 years": np.nan,
            "Annualized Return – Last 3 years": np.nan,
            "Annualized Return – Last 1 year": np.nan,
            "Best Month": "N/A",
            "Worst Month": "N/A"
        })

    # On construit maintenant le DataFrame d'affichage
    abs_stats_df = pd.DataFrame(
        [
            [portfolio_stats_incep.get('ann_return', np.nan), abs_stats_bench["Annualized Return – Since inception"]],
            [portfolio_stats_5y.get('ann_return', np.nan),     abs_stats_bench["Annualized Return – Last 5 years"]],
            [portfolio_stats_3y.get('ann_return', np.nan),     abs_stats_bench["Annualized Return – Last 3 years"]],
            [portfolio_stats_1y.get('ann_return', np.nan),     abs_stats_bench["Annualized Return – Last 1 year"]],
            [portfolio_stats_incep.get('best_month', "N/A"),   abs_stats_bench["Best Month"]],
            [portfolio_stats_incep.get('worst_month', "N/A"),  abs_stats_bench["Worst Month"]]
        ],
        columns=["Portfolio", "Benchmark"],
        index=[
            "Annualized Return – Since inception",
            "Annualized Return – Last 5 years",
            "Annualized Return – Last 3 years",
            "Annualized Return – Last 1 year",
            "Best Month",
            "Worst Month"
        ]
    )
    abs_stats_df.index.name = "Absolute Return Statistics"
    
    styled_abs_stats_df = abs_stats_df.style.format({
        "Portfolio": lambda x: "{:.2%}".format(x) if isinstance(x, (float, np.floating)) else x,
        "Benchmark": lambda x: "{:.2%}".format(x) if isinstance(x, (float, np.floating)) else x,
    }).set_properties(subset=["Portfolio"], **{'color': '#00539b', 'font-weight': 'bold'})\
      .set_properties(subset=["Benchmark"], **{'color': '#000000'})
    
    col_abs_left, col_abs_right = st.columns([1.7, 2])
    with col_abs_left:
        st.markdown("   ")
        st.markdown("   ")
        st.write(styled_abs_stats_df)
    with col_abs_right:
        fig_cum = plot_cumulative_comparison(portfolio_cum, benchmark_cum)
        st.plotly_chart(fig_cum, use_container_width=True, key="cum_return")

    # --- Calcul des ratios Risk-Adjusted pour le portefeuille et le benchmark ---
    # Portfolio
    port_ann_return = portfolio_ret.mean() * 12
    port_ann_vol = portfolio_ret.std() * np.sqrt(12)
    port_max_dd = (portfolio_cum / portfolio_cum.cummax() - 1).min()
    port_sharpe = port_ann_return / port_ann_vol if port_ann_vol != 0 else np.nan
    port_sortino = compute_sortino_ratio(portfolio_ret)
    port_calmar = port_ann_return / abs(port_max_dd) if port_max_dd != 0 else np.nan
    port_treynor = compute_treynor_ratio(portfolio_ret, benchmark_ret)
    outperf_ratio = (portfolio_ret[portfolio_ret.index.intersection(benchmark_ret.index)] > benchmark_ret).mean() if not benchmark_ret.empty else np.nan

    # Benchmark (déjà dans la session ; on ne recalcule pas annual_return etc. 
    # On fait juste Sharpe / Sortino pour l’exemple)
    if not benchmark_ret.empty:
        bench_ann_return = benchmark_ret.mean() * 12
        bench_ann_vol = benchmark_ret.std() * np.sqrt(12)
        bench_cum_sim = (1 + benchmark_ret).cumprod()
        bench_max_dd = (bench_cum_sim / bench_cum_sim.cummax() - 1).min()
        bench_sharpe = bench_ann_return / bench_ann_vol if bench_ann_vol != 0 else np.nan
        bench_sortino = compute_sortino_ratio(benchmark_ret)
        bench_calmar = bench_ann_return / abs(bench_max_dd) if bench_max_dd != 0 else np.nan
    else:
        bench_sharpe = bench_sortino = bench_calmar = np.nan

    metrics_df = pd.DataFrame({
        "Relative & Risk-Adjusted Stats": [
            "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Calmar Ratio", "Outperform Ratio"
        ],
        "Portfolio": [
            f"{port_sharpe:.2f}",
            f"{port_sortino:.2f}",
            f"{port_treynor:.2f}",
            f"{port_calmar:.2f}",
            f"{outperf_ratio:.2%}"
        ],
        "Benchmark": [
            f"{bench_sharpe:.2f}",
            f"{bench_sortino:.2f}",
            "-",
            f"{bench_calmar:.2f}",
            "-"
        ]
    }).set_index("Relative & Risk-Adjusted Stats")

    styled_metrics_df = metrics_df.style.set_properties(
        subset=["Portfolio"], **{"color": "#00539b", "font-weight": "bold"}
    ).set_properties(subset=["Benchmark"], **{"color": "#000000"})\
     .set_properties(**{"text-align": "center"})

    ticker_to_name = {
        "SPY": "S&P 500 ETF",
        "QQQ": "Nasdaq 100 ETF",
        "FEZ": "EuroStoxx 50 ETF",
        "IEF": "10Y Treasury ETF",
        "URTH": "MSCI World ETF",
        "EEM": "Emerging Markets ETF",
        "ICLN": "Clean Energy ETF",
        "GLD": "Gold ETF",
        "USO": "Crude Oil ETF"
    }

    st.markdown("---")
    portfolio_summary = {
        "cumulative_returns": portfolio_cum,
        "drawdowns": portfolio_cum / portfolio_cum.cummax() - 1
    }
    benchmark_summary = {
        "cumulative_returns": benchmark_cum,
        "drawdowns": benchmark_cum / benchmark_cum.cummax() - 1
    } if benchmark_cum is not None else {
        "cumulative_returns": pd.Series(dtype=float),
        "drawdowns": pd.Series(dtype=float)
    }

    fig_cum_dd = plot_cumulative_and_drawdowns_portfolio(portfolio_summary, benchmark_summary)
    col_bot_left, col_mid, col_bot_right = st.columns([2.2, 0.25, 1.8])
    with col_bot_left:
        if fig_cum_dd:
            st.plotly_chart(fig_cum_dd, use_container_width=True, key="risk_metrics")
        else:
            st.warning("Not enough data to plot cumulative returns and drawdowns.")
    with col_bot_right:
        st.markdown("     ")
        st.markdown("     ")
        st.dataframe(styled_metrics_df)
        generate_pdf_report(
            fig=fig_cum_dd,
            df_abs_stats=abs_stats_df,
            df_metrics=metrics_df,
            weights_dict=weights,
            ticker_to_name=ticker_to_name,
        )
    st.markdown("---")

# --------------------------------
# Onglet 5: Market & Macro News
# --------------------------------
news_categories = {
    "Financial Times": {
        "World": "https://www.ft.com/world?format=rss",
        "Markets": "https://www.ft.com/markets?format=rss",
        "Companies": "https://www.ft.com/companies?format=rss",
        "Asset Management": "https://www.ft.com/asset-management?format=rss"
    },
    "CNBC": {
        "Top News": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "Markets": "https://www.cnbc.com/id/15839135/device/rss/rss.html",
        "Economy": "https://www.cnbc.com/id/20910258/device/rss/rss.html"
    },
    "Yahoo Finance": {
        "Market News": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US"
    }
}

with main_tabs[4]:
    st.markdown("<h3 style='color:#00539b'>Market & Macro News</h3>", unsafe_allow_html=True)
    st.markdown("Explore the latest macroeconomic and market headlines by source and category.")

    selected_source = st.selectbox("Choose news source:", list(news_categories.keys()))
    selected_category = st.selectbox("Select category:", list(news_categories[selected_source].keys()))
    st.markdown("---")

    feed_url = news_categories[selected_source][selected_category]
    with st.spinner("Fetching latest headlines..."):
        news_items = get_news_rss(feed_url, limit=10)

    for entry in news_items:
        st.markdown(f"##### [{entry.title}]({entry.link})")
        if hasattr(entry, 'published'):
            st.caption(entry.published)
        if hasattr(entry, 'summary'):
            st.write(entry.summary)
        st.markdown("---")
