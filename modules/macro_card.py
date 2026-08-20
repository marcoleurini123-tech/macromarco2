import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

# 7 Panieri Macro
MACRO_BASKETS = {
    "GOLDILOCKS": ["QQQ", "XLK", "XLY", "IEF", "SMH"],
    "RECESSIONE": ["TLT", "SHY", "XLU", "XLP", "GLD"],
    "STAGFLAZIONE": ["GLD", "DBC", "XLE", "TIP", "VTV"],
    "REFLAZIONE": ["XLI", "XLF", "IWM", "EEM", "DBC"],
    "DISINFLAZIONE": ["TLT", "LQD", "QQQ", "VTI", "GLD"],
    "DEBOLEZZA DOLLARO": ["EEM", "FXF", "GLD", "IXUS", "DBC"],
    "DEFLAZIONE": ["TLT", "BIL", "SHY", "XLP", "XLU"]
}

@st.cache_data(ttl=3600 * 12)
def get_macro_dominant_scenario():
    all_tickers = sorted(list({t for b in MACRO_BASKETS.values() for t in b}))
    df = yf.download(all_tickers, period="3mo", interval="1d", progress=False)['Close']
    df = df.ffill().dropna()

    perf_1m = {}
    for scenario, tickers in MACRO_BASKETS.items():
        valid = [t for t in tickers if t in df.columns]
        sub = df[valid]
        ret_mean = ((sub.iloc[-1] / sub.iloc[-21] - 1) * 100).mean()
        perf_1m[scenario] = ret_mean

    series_perf = pd.Series(perf_1m)
    dominant_scenario = series_perf.idxmax()
    
    exp_vals = np.exp(series_perf - series_perf.max())
    probabilities = (exp_vals / exp_vals.sum()) * 100
    confidence = int(probabilities[dominant_scenario])

    return dominant_scenario, confidence, series_perf

def render_quantaste_macro_card():
    dominant, confidence, _ = get_macro_dominant_scenario()
    
    st.markdown("""
        <style>
        .macro-card {
            background-color: #121927;
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid #1E293B;
            margin-bottom: 24px;
            color: #FFFFFF;
            font-family: sans-serif;
        }
        .macro-title {
            font-size: 14px;
            color: #94A3B8;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .dominant-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 14px;
            margin-bottom: 14px;
        }
        .scenario-badge {
            background-color: #F59E0B;
            color: #0F172A;
            font-weight: 800;
            padding: 6px 14px;
            border-radius: 6px;
            font-size: 15px;
            letter-spacing: 0.5px;
        }
        .confidence-val {
            font-size: 30px;
            font-weight: 800;
            color: #F59E0B;
        }
        .bar-container {
            position: relative;
            height: 6px;
            background: linear-gradient(to right, #EF4444 0%, #F59E0B 50%, #10B981 100%);
            border-radius: 3px;
            margin: 20px 0 8px 0;
        }
        .needle {
            position: absolute;
            top: -6px;
            width: 4px;
            height: 18px;
            background: #FFFFFF;
            border-radius: 2px;
            transform: translateX(-50%);
            box-shadow: 0 0 6px rgba(255,255,255,0.9);
        }
        .axis-labels {
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #64748B;
        }
        </style>
    """, unsafe_allow_html=True)

    needle_pos = min(max(confidence, 5), 95)
    st.markdown(f"""
        <div class="macro-card">
            <div class="macro-title">🌐 REGIME ECONOMICO PREDOMINANTE</div>
            <div style="font-size: 11px; color: #64748B; margin-top: 4px;">REGIME DOMINANTE (USA / GLOBALE)</div>
            <div class="dominant-row">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 18px;">🇺🇸</span>
                    <span style="font-size: 14px; font-weight: 600;">USA</span>
                    <span class="scenario-badge">{dominant}</span>
                </div>
                <div class="confidence-val">{confidence}<span style="font-size: 18px; font-weight: 600;">%</span></div>
            </div>
            <div class="bar-container">
                <div class="needle" style="left: {needle_pos}%;"></div>
            </div>
            <div class="axis-labels">
                <span>0</span>
                <span>50</span>
                <span>100</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
