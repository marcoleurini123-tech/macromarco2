import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests

# =========================================================
# CONFIGURAZIONE PIATTAFORMA
# =========================================================
st.set_page_config(
    page_title="Macro & Quant Terminal",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 1. MODULO SCENARI MACRO (7 PORTAFOGLI EOD - STILE QUANTASTE)
# =========================================================
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
def get_macro_dominant():
    all_tickers = sorted(list({t for b in MACRO_BASKETS.values() for t in b}))
    df = yf.download(all_tickers, period="6mo", interval="1d", progress=False)['Close']
    df = df.ffill().dropna()

    perf_1m = {}
    details = []
    
    for scenario, tickers in MACRO_BASKETS.items():
        valid = [t for t in tickers if t in df.columns]
        sub = df[valid]
        
        r1d = float(((sub.iloc[-1] / sub.iloc[-2] - 1) * 100).mean()) if len(sub) > 1 else 0.0
        r1w = float(((sub.iloc[-1] / sub.iloc[-6] - 1) * 100).mean()) if len(sub) > 5 else 0.0
        r1m = float(((sub.iloc[-1] / sub.iloc[-21] - 1) * 100).mean()) if len(sub) > 20 else 0.0
        r3m = float(((sub.iloc[-1] / sub.iloc[-63] - 1) * 100).mean()) if len(sub) > 62 else 0.0
        
        perf_1m[scenario] = r1m
        details.append({
            "Scenario Macro": scenario,
            "1 Giorno": f"{r1d:+.2f}%",
            "1 Settimana": f"{r1w:+.2f}%",
            "1 Mese": f"{r1m:+.2f}%",
            "3 Mesi": f"{r3m:+.2f}%"
        })

    series_perf = pd.Series(perf_1m)
    dominant_scenario = series_perf.idxmax()
    
    # Confidenza percentuale Softmax
    exp_vals = np.exp(series_perf - series_perf.max())
    probabilities = (exp_vals / exp_vals.sum()) * 100
    confidence = int(probabilities[dominant_scenario])

    df_details = pd.DataFrame(details).set_index("Scenario Macro")
    return dominant_scenario, confidence, df_details

def render_macro_card():
    dominant, confidence, df_details = get_macro_dominant()
    
    st.markdown("""
        <style>
        .macro-card {
            background-color: #121927;
            border-radius: 12px;
            padding: 20px 24px;
            border: 1px solid #1E293B;
            margin-bottom: 20px;
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
            font-size: 16px;
            letter-spacing: 0.5px;
        }
        .confidence-val {
            font-size: 32px;
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
                    <span style="font-size: 20px;">🇺🇸</span>
                    <span style="font-size: 15px; font-weight: 600;">USA</span>
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
    return df_details

# =========================================================
# 2. MODULO LIQUIDITÀ FED (FRED DATA EOD)
# =========================================================
@st.cache_data(ttl=3600 * 12)
def get_fred_liquidity_data():
    def fetch_series(series_id):
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.columns = [series_id]
        df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
        return df.dropna()

    walcl = fetch_series("WALCL")      # Total Assets (Millions)
    rrp = fetch_series("RRPONTSYD")    # Reverse Repo (Billions)
    tga = fetch_series("WTREGEN")      # TGA (Billions)
    t10y2y = fetch_series("T10Y2Y")    # 10Y-2Y Spread

    walcl_b = walcl / 1000.0           # Convert to Billions
    combined = pd.concat([walcl_b, rrp, tga, t10y2y], axis=1).ffill().dropna()
    combined.columns = ["WALCL", "RRP", "TGA", "T10Y2Y"]
    combined["Fed_Net_Liquidity"] = combined["WALCL"] - combined["RRP"] - combined["TGA"]
    return combined

def render_fed_liquidity_section():
    st.markdown("### 🏛️ Regime Macroeconomico & Liquidità Fed")
    st.caption("Monitoraggio delle condizioni monetarie: Total Assets, Reverse Repo, TGA e Yield Curve.")

    try:
        data = get_fred_liquidity_data()
        latest = data.iloc[-1]
        prev_year = data.iloc[-252] if len(data) > 252 else data.iloc[0]

        net_liq = latest["Fed_Net_Liquidity"]
        net_liq_yoy = ((net_liq / prev_year["Fed_Net_Liquidity"]) - 1) * 100
        spread_10_2 = latest["T10Y2Y"]
        total_assets = latest["WALCL"]
        rrp = latest["RRP"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Fed Net Liquidity",
                value=f"${net_liq:,.1f} B",
                delta=f"{net_liq_yoy:+.2f}% YoY"
            )
            st.metric(
                label="Total Fed Assets",
                value=f"${total_assets:,.1f} B"
            )
        with col2:
            st.metric(
                label="Spread 10Y - 2Y",
                value=f"{spread_10_2:.2f}%",
                delta="Disinversione" if spread_10_2 > 0 else "Inversione"
            )
            st.metric(
                label="Reverse Repo (RRP)",
                value=f"${rrp:,.1f} B"
            )
    except Exception:
        st.info("Dati FED in fase di sincronizzazione...")

# =========================================================
# 3. MODULO TELEGRAM & SCREENER QUANTITATIVO
# =========================================================
def send_telegram(bot_token: str, chat_id: str, message: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=8)
        return r.status_code == 200
    except Exception:
        return False

def run_screener(tickers: list, bot_token: str, chat_id: str):
    matched = []
    data = yf.download(tickers, period="6mo", interval="1d", progress=False)
    close = data['Close']
    
    for t in tickers:
        series = close[t].dropna() if isinstance(close, pd.DataFrame) else close.dropna()
        if len(series) < 50:
            continue
            
        current = float(series.iloc[-1])
        sma50 = float(series.rolling(50).mean().iloc[-1])
        prev = float(series.iloc[-2])
        
        # Protocollo di Screening: Breakout sopra la media mobile a 50 giorni
        if current > sma50 and prev <= sma50:
            matched.append({"Ticker": t, "Prezzo EOD": round(current, 2), "Segnale": "Breakout sopra SMA 50"})

    df_res = pd.DataFrame(matched)
    
    if not df_res.empty and bot_token and chat_id:
        msg = "🚨 *SEGNALI SCREENER QUANTITATIVO*\n\n"
        for _, row in df_res.iterrows():
            msg += f"• *{row['Ticker']}*: ${row['Prezzo EOD']} — `{row['Segnale']}`\n"
        send_telegram(bot_token, chat_id, msg)
        
    return df_res

# =========================================================
# INTERFACCIA PRINCIPALE
# =========================================================
st.title("Panoramica Macro e Mercati")

# 1. Card Stile Quantaste sempre in cima
df_macro_details = render_macro_card()

# 2. Selettore Moduli nella Sidebar
st.sidebar.title("🛠 Moduli & Protocolli")
modulo = st.sidebar.radio(
    "Seleziona Protocollo Attivo:",
    [
        "Panoramica Generale (Macro + Fed)",
        "Dettagli 7 Scenari Macro (ETF)",
        "Screener Azionario EOD + Alert Telegram"
    ]
)

if modulo == "Panoramica Generale (Macro + Fed)":
    render_fed_liquidity_section()

elif modulo == "Dettagli 7 Scenari Macro (ETF)":
    st.subheader("📊 Analisi Dettagliata Panieri Macro")
    if not df_macro_details.empty:
        # Tabella nativa senza dipendenze esterne
        st.dataframe(df_macro_details, use_container_width=True)

elif modulo == "Screener Azionario EOD + Alert Telegram":
    st.subheader("🎯 Monitoraggio Watchlist & Invio Telegram")
    st.caption("Esegue lo screener solo quando richiesto sui dati EOD e invia gli alert su Telegram.")
    
    col_t1, col_t2 = st.columns(2)
    bot_token = col_t1.text_input("Telegram Bot Token:", type="password", placeholder="123456789:ABC...")
    chat_id = col_t2.text_input("Telegram Chat ID / Canale:", placeholder="@tuo_canale o ID numerico")
    
    watchlist_str = st.text_area(
        "Watchlist Titoli (separati da virgola):",
        "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, INTC, NFLX"
    )
    tickers_list = [x.strip().upper() for x in watchlist_str.split(",") if x.strip()]
    
    if st.button("🚀 Esegui Protocollo Screener"):
        with st.spinner("Scaricamento dati EOD e analisi condizioni..."):
            df_signals = run_screener(tickers_list, bot_token, chat_id)
            if not df_signals.empty:
                st.success(f"Trovati {len(df_signals)} titoli conformi!")
                st.dataframe(df_signals, use_container_width=True)
                if bot_token and chat_id:
                    st.info("Notifica inviata sul canale Telegram.")
            else:
                st.warning("Nessun titolo soddisfa attualmente i criteri del protocollo.")
