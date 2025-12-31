# =========================================================
# FORENSIC FINANCIAL STATEMENT ANALYSIS DASHBOARD
# Accrual Manipulation + Real Earnings Management (REM)
# Proxy Models | Yahoo Finance | Audit-Grade
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(
    page_title="Forensic Earnings Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
.main { background-color: #0b1220; }
.block-container { max-width: 1250px; padding-top: 2rem; padding-bottom: 2rem; }

h1, h2, h3 { color: #E5E7EB; font-weight: 600; }

.section-header {
    margin-top: 40px;
    margin-bottom: 15px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1F2937;
}

/* KPI Cards */
.kpi-box {
    background: linear-gradient(145deg, #111827, #0f172a);
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 8px 18px rgba(0,0,0,0.45);
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-title {
    font-size: 13px;
    letter-spacing: 0.5px;
    color: #9CA3AF;
    text-transform: uppercase;
}

.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #F9FAFB;
}

.kpi-label {
    font-size: 13px;
    color: #D1D5DB;
}

.subtext {
    color: #9CA3AF;
    font-size: 15px;
    line-height: 1.6;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div style="text-align:center;">
    <h1>📊 Forensic Financial Statement Analysis Dashboard</h1>
    <p class="subtext">
        Detect earnings manipulation and <b>early management behavior</b><br>
        Accrual-based + Real Earnings Management (REM) using proxy models<br>
        Data Source: Yahoo Finance (Public Financial Statements)
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe(x):
    try:
        if pd.isna(x) or np.isinf(x):
            return None
        return float(x)
    except:
        return None

def get_col(df, names):
    for n in names:
        if n in df.columns:
            return df[n]
    return np.nan

def kpi_box(title, value, label):
    st.markdown(f"""
    <div class="kpi-box">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FETCH FINANCIAL DATA
# =========================================================
def fetch_financials(ticker):
    c = yf.Ticker(ticker)
    df = (
        c.financials.T
        .join(c.balance_sheet.T, how="inner")
        .join(c.cashflow.T, how="inner")
    )
    df.index = df.index.year
    return df.sort_index()

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def create_features(df):
    df = df.copy()

    df["Revenue"] = get_col(df, ["Total Revenue"])
    df["Net_Income"] = get_col(df, ["Net Income"])
    df["OCF"] = get_col(df, ["Total Cash From Operating Activities", "Operating Cash Flow"])
    df["Total_Assets"] = get_col(df, ["Total Assets"])
    df["Receivables"] = get_col(df, ["Net Receivables"])
    df["COGS"] = get_col(df, ["Cost Of Revenue"])

    df = df.dropna(subset=["Revenue", "Net_Income", "OCF", "Total_Assets"])

    df["Gross_Margin"] = (df["Revenue"] - df["COGS"]) / df["Revenue"]
    df["Accruals_Ratio"] = (df["Net_Income"] - df["OCF"]) / df["Total_Assets"]
    df["ROA"] = df["Net_Income"] / df["Total_Assets"]

    # REM indicators
    df["OCF_to_Revenue"] = df["OCF"] / df["Revenue"]
    df["Revenue_Growth"] = df["Revenue"].pct_change()
    df["OCF_Growth"] = df["OCF"].pct_change()
    df["Sales_Cash_Gap"] = df["Revenue_Growth"] - df["OCF_Growth"]
    df["COGS_to_Revenue"] = df["COGS"] / df["Revenue"]

    return df

# =========================================================
# FORENSIC SCORES
# =========================================================
def beneish_m_proxy(df):
    dsri = (df["Receivables"] / df["Revenue"]).pct_change().dropna()
    dsri = dsri.iloc[-1] + 1 if not dsri.empty else 1

    gmi = (df["Gross_Margin"].shift(1) / df["Gross_Margin"]).dropna()
    gmi = gmi.iloc[-1] if not gmi.empty else 1

    sgi = df["Revenue"].pct_change().dropna()
    sgi = sgi.iloc[-1] + 1 if not sgi.empty else 1

    tata = df["Accruals_Ratio"].iloc[-1]

    return safe(-4.84 + 0.92*dsri + 0.528*gmi + 0.892*sgi + 0.404*tata)

def piotroski_f_score(df):
    if len(df) < 2:
        return None

    f = 0
    f += df["Net_Income"].iloc[-1] > 0
    f += df["OCF"].iloc[-1] > 0
    f += df["ROA"].iloc[-1] > df["ROA"].iloc[-2]
    f += df["OCF"].iloc[-1] > df["Net_Income"].iloc[-1]
    f += df["Revenue"].pct_change().iloc[-1] > 0
    f += df["Gross_Margin"].iloc[-1] > df["Gross_Margin"].iloc[-2]
    f += df["Total_Assets"].pct_change().iloc[-1] <= 0

    return int(f)

def rem_risk(df):
    r1 = 1 if df["OCF_to_Revenue"].iloc[-1] < 0.10 else 0
    r2 = 1 if df["Sales_Cash_Gap"].iloc[-1] > 0.10 else 0
    r3 = 1 if df["COGS_to_Revenue"].pct_change().iloc[-1] < -0.05 else 0
    return (r1 + r2 + r3) / 3

# =========================================================
# MAIN APP
# =========================================================
ticker = st.text_input("Enter Company Ticker (e.g. TCS.NS, RELIANCE.NS, AAPL)")

if ticker:
    df = create_features(fetch_financials(ticker))

    beneish = beneish_m_proxy(df)
    fscore = piotroski_f_score(df)
    accrual = safe(df["Accruals_Ratio"].iloc[-1])
    rem = rem_risk(df)

    risks = [
        1 if beneish and beneish > -2.22 else 0,
        0 if fscore and fscore >= 7 else (0.5 if fscore and fscore >= 4 else 1),
        0 if accrual and abs(accrual) < 0.05 else (0.5 if abs(accrual) < 0.10 else 1),
        rem
    ]
    fraud_prob = round(np.mean(risks) * 100, 2)

    # =====================================================
    # KPI SCORECARD
    # =====================================================
    st.markdown('<h3 class="section-header">🧮 Forensic Scorecard</h3>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_box("Beneish M-Score", round(beneish,2), "Accrual Risk")
    with c2: kpi_box("Piotroski F-Score", fscore, "Financial Quality")
    with c3: kpi_box("Accruals Ratio", round(abs(accrual),3), "Earnings Quality")
    with c4: kpi_box("REM Score", round(rem,2), "Early Management Signal")
    with c5: kpi_box("Manipulation Probability", f"{fraud_prob}%", "Composite Risk")

    # =====================================================
    # FORENSIC VISUALS (ADDED)
    # =====================================================
    st.markdown('<h3 class="section-header">📊 Forensic Visual Diagnostics</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            px.line(df, y=["Revenue", "OCF"], title="Revenue vs Operating Cash Flow"),
            use_container_width=True
        )
    with col2:
        st.plotly_chart(
            px.bar(df, y="Accruals_Ratio", title="Accruals Ratio Trend"),
            use_container_width=True
        )

    col3, col4 = st.columns(2)
    with col3:
        rem_df = df[["OCF_to_Revenue", "Sales_Cash_Gap", "COGS_to_Revenue"]].iloc[-1]
        st.plotly_chart(
            px.bar(rem_df, title="REM Component Breakdown (Latest Year)"),
            use_container_width=True
        )
    with col4:
        st.plotly_chart(
            px.scatter(df, x="Revenue_Growth", y="Gross_Margin",
                       title="Gross Margin vs Revenue Growth",
                       trendline="ols"),
            use_container_width=True
        )

    # =====================================================
    # DATA TABLE
    # =====================================================
    st.markdown('<h3 class="section-header">📄 Historical Financial Data</h3>', unsafe_allow_html=True)
    st.dataframe(df.round(3))

    # =====================================================
    # FORENSIC CONCLUSION
    # =====================================================
    st.markdown('<h3 class="section-header">🧠 Forensic Conclusion</h3>', unsafe_allow_html=True)

    risk_band = "Low" if fraud_prob < 30 else "Moderate" if fraud_prob < 60 else "High"

    st.write(f"""
    **Overall Assessment:** **{risk_band} Earnings Manipulation Risk**

    The company shows an **estimated earnings manipulation probability of {fraud_prob}%**.

    The analysis combines:
    - **Accrual-based indicators** (Beneish M-Score, Accruals Ratio),
    - **Financial quality signals** (Piotroski F-Score),
    - **Real Earnings Management (REM)** indicators capturing early managerial behavior.

    These results represent **risk signals, not evidence of fraud**, and are intended to
    support audit planning, investment analysis, and credit evaluation.
    """)

else:
    st.info("Enter a company ticker to start analysis")
