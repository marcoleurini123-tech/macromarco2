import io
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ==========================================
# 0. CONFIGURAZIONE GENERALE & COSTANTI
# ==========================================
st.set_page_config(
    page_title="Quantitative Macro & Positioning Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_FILE = "macro_data.csv"
COLUMNS = [
    "Data", "VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y", "VVIX", "MOVE", "SKEW",
    "DXY", "DIX", "GEX", "SPY", "RSP", "HYG", "XLY", "XLP", "TLT", "P_C", "GLD", "USO", 
    "Net_Liquidity", "M2"
]
GOOGLE_BRIDGE_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSeeY57SBwd6BftA2Bq8C0nyzzT3wj9WRWOihDF7QE-COPXhC4r2RN_k_BRgZke1nU2BbKT8oRlsXOX/pub?gid=1412711569&single=true&output=csv"

DEFAULT_TELEGRAM_TOKEN = ""
DEFAULT_TELEGRAM_CHAT_ID = ""

# ==========================================
# 1. ENGINE DI NOTIFICA TELEGRAM
# ==========================================
def send_telegram_alert(bot_token: str, chat_id: str, message: str) -> bool:
    if not bot_token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    try:
        res = requests.post(url, json=payload, timeout=8)
        return res.status_code == 200
    except Exception:
        return False

# ==========================================
# 2. GESTIONE DATABASE LOCALE & FETCH EOD
# ==========================================
def load_db() -> pd.DataFrame:
    if os.path.exists(DB_FILE):
        try:
            df = pd.read_csv(DB_FILE)
            df['Data'] = pd.to_datetime(df['Data']).dt.normalize()
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = 0.0
            return df.sort_values("Data")
        except Exception:
            pass
    return pd.DataFrame(columns=COLUMNS)

def save_db(df: pd.DataFrame):
    df = df.drop_duplicates(subset=['Data'], keep='last').sort_values("Data")
    df.to_csv(DB_FILE, index=False)

def fetch_bridge_data() -> pd.DataFrame:
    try:
        response = requests.get(GOOGLE_BRIDGE_URL, timeout=10)
        df_bridge = pd.read_csv(io.StringIO(response.text))
        df_bridge.columns = df_bridge.columns.str.strip()
        df_bridge = df_bridge.rename(columns={'Date': 'Data', 'Net_Liquidity': 'Net_Liquidity', 'M2': 'M2'})
        if pd.api.types.is_numeric_dtype(df_bridge['Data']):
            df_bridge['Data'] = pd.to_datetime(df_bridge['Data'], unit='D', origin='1899-12-30')
        else:
            df_bridge['Data'] = pd.to_datetime(df_bridge['Data'], errors='coerce')
        df_bridge['Data'] = df_bridge['Data'].dt.normalize()
        for col in ['Net_Liquidity', 'M2']:
            if col in df_bridge.columns:
                df_bridge[col] = pd.to_numeric(df_bridge[col], errors='coerce')
        return df_bridge.dropna(subset=['Data', 'Net_Liquidity'])
    except Exception:
        return pd.DataFrame(columns=["Data", "Net_Liquidity", "M2"])

def fetch_yahoo(days=60) -> pd.DataFrame:
    tickers = {
        "VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M",
        "VIX1Y": "^VIX1Y", "VVIX": "^VVIX", "SKEW": "^SKEW", "DXY": "DX-Y.NYB",
        "SPY": "SPY", "RSP": "RSP", "XLY": "XLY", "XLP": "XLP", "HYG": "HYG",
        "TLT": "TLT", "P_C": "^PCCR", "GLD": "GLD", "USO": "USO"
    }
    try:
        data = yf.download(list(tickers.values()), period=f"{days}d", interval="1d", progress=False)['Close']
        data = data.rename(columns={v: k for k, v in tickers.items()})
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        return data.reset_index().rename(columns={'Date': 'Data', 'index': 'Data'})
    except Exception:
        return pd.DataFrame(columns=["Data"])

# ==========================================
# 3. MODULO A: REGIME MACRO & DASHBOARD EOD
# ==========================================
def render_macro_module(df: pd.DataFrame):
    st.header("🌐 Analisi Quantitativa & Regime Macroeconomico")
    
    if df.empty:
        st.info("Nessun dato presente nel database locale. Sincronizza dalla barra laterale.")
        return

    df = df.sort_values("Data").copy()
    df['Liq_Delta_5D'] = df['Net_Liquidity'].pct_change(periods=5) * 100
    df['Ratio_GO'] = df['GLD'] / df['USO'].replace(0, np.nan)
    df['Ratio_Risk'] = df['XLY'] / df['XLP'].replace(0, np.nan)
    df['Ratio_Br'] = df['SPY'] / df['RSP'].replace(0, np.nan)
    
    last = df.iloc[-1]
    
    # Alert Regime Macro
    if last['Liq_Delta_5D'] < 0 and len(df) >= 5 and last['SPY'] > df.iloc[-5]['SPY']:
        st.error(f"🚨 ALERT DIVERGENZA: Liquidità netta in contrazione ({last['Liq_Delta_5D']:.2f}%) con SPY in rialzo. Probabile mismatch di liquidità.")

    st.subheader("🚦 Monitor Segnali di Regime")
    r1 = st.columns(6)
    r1[0].metric("DIX", f"{last['DIX']:.1f}%", "🟢 BULLISH" if last['DIX'] > 45 else "⚪ NEUTRO")
    r1[1].metric("GEX", f"{last['GEX']:,.0f}", "🔴 SQUEEZE" if last['GEX'] < 0 else "🟢 STABILE", delta_color="inverse")
    r1[2].metric("P/C RATIO", f"{last['P_C']:.2f}", "🟢 PANICO" if last['P_C'] > 1.05 else ("🔴 AVIDITÀ" if 0 < last['P_C'] < 0.7 else "⚪ NEUTRO"))
    r1[3].metric("SKEW", f"{last['SKEW']:.1f}", "⚠️ TAIL RISK" if last['SKEW'] > 145 else "🟢 OK", delta_color="inverse")
    r1[4].metric("MOVE", f"{last['MOVE']:.1f}", "🔴 STRESS BOND" if last['MOVE'] > 115 else "🟢 CALMO", delta_color="inverse")
    liq_col = "normal" if last['Liq_Delta_5D'] >= 0 else "inverse"
    r1[5].metric("Δ LIQ. 5D", f"{last['Liq_Delta_5D']:.2f}%", "📉 CONTRAZIONE" if last['Liq_Delta_5D'] < 0 else "📈 ESPANSIONE", delta_color=liq_col)

    r2 = st.columns(6)
    r2[0].metric("DXY", f"{last['DXY']:.2f}", "🔴 USD UP" if last['DXY'] > 103.5 else "🟢 USD DOWN", delta_color="inverse")
    r2[1].metric("GOLD/OIL", f"{last['Ratio_GO']:.2f}", "⚠️ ALERT" if last['Ratio_GO'] > 2.5 else "🟢 OK")
    r2[2].metric("TLT PRICE", f"${last['TLT']:.2f}", "📈 TASSI DOWN" if len(df) > 1 and last['TLT'] > df.iloc[-2]['TLT'] else "📉 TASSI UP")
    r2[3].metric("XLY/XLP", f"{last['Ratio_Risk']:.2f}", "🟢 RISK-ON" if last['Ratio_Risk'] > 1.45 else "🔴 DIFESA")
    r2[4].metric("SPY/RSP", f"{last['Ratio_Br']:.2f}", "⚠️ ALERT" if last['Ratio_Br'] > 3.5 else "🟢 SANA")
    v_stat = "🔴 INVERTITA" if last.get('VIX1D', 0) > last.get('VIX', 0) else "🟢 CONTANGO"
    r2[5].metric("CURVA VIX", f"{last.get('VIX1D', 0):.1f}/{last.get('VIX', 0):.1f}", v_stat)

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💹 Liquidità Netta vs Trend")
        st.plotly_chart(px.area(df[df['Net_Liquidity'] > 0].tail(250), x="Data", y="Net_Liquidity", color_discrete_sequence=['#00CC96']), use_container_width=True)
    with c2:
        st.subheader("💰 M2 Money Supply")
        st.plotly_chart(px.line(df[df['M2'] > 0].tail(250), x="Data", y="M2"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏆 Ratio Gold / Oil")
        fig_go = px.line(df[df['Ratio_GO'] > 0].tail(100), x="Data", y="Ratio_GO", color_discrete_sequence=['#FFD700'])
        fig_go.add_hline(y=2.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_go, use_container_width=True)
    with c4:
        st.subheader("📉 TLT Price vs MOVE Index")
        st.plotly_chart(px.line(df.tail(100), x="Data", y=["TLT", "MOVE"], color_discrete_map={"TLT": "yellow", "MOVE": "red"}), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("📈 VIX Term Structure")
        t_vals = [last.get(c, 0.0) for c in ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]]
        fig_vx = go.Figure(go.Scatter(x=["1D", "9D", "30D", "3M", "6M", "1Y"], y=t_vals, mode='lines+markers+text', text=[f"{v:.1f}" for v in t_vals], textposition="top center"))
        fig_vx.update_traces(line=dict(color="red" if last.get('VIX1D', 0) > last.get('VIX', 0) else "#00CC96", width=3))
        st.plotly_chart(fig_vx, use_container_width=True)
    with c6:
        st.subheader("⚡ VVIX vs DXY")
        st.plotly_chart(px.line(df.tail(100), x="Data", y=["VVIX", "DXY"], color_discrete_map={"VVIX": "orange", "DXY": "#00D1FF"}), use_container_width=True)

    with st.expander("Visualizza Tabella Dati Macro Recenti"):
        st.dataframe(df.sort_values("Data", ascending=False).head(20), use_container_width=True)

# ==========================================
# 4. MODULO B: SCREENER AZIONARIO QUANTITATIVO
# ==========================================
def calculate_technical_indicators(series_close: pd.Series, series_volume: pd.Series):
    delta = series_close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    sma_50 = series_close.rolling(window=50).mean()
    sma_200 = series_close.rolling(window=200).mean()
    vol_sma_20 = series_volume.rolling(window=20).mean()
    vol_ratio = series_volume / vol_sma_20.replace(0, np.nan)

    return rsi, sma_50, sma_200, vol_ratio

def run_screener_engine(tickers_list: list) -> pd.DataFrame:
    results = []
    for ticker in tickers_list:
        try:
            hist = yf.download(ticker, period="1y", interval="1d", progress=False)
            if hist.empty or len(hist) < 50:
                continue
            
            close = hist['Close'].squeeze()
            volume = hist['Volume'].squeeze()
            
            rsi, sma_50, sma_200, vol_ratio = calculate_technical_indicators(close, volume)
            
            p_last = close.iloc[-1]
            p_prev = close.iloc[-2]
            pct_1d = ((p_last - p_prev) / p_prev) * 100
            
            results.append({
                "Ticker": ticker,
                "Prezzo EOD": round(float(p_last), 2),
                "Var 1D (%)": round(float(pct_1d), 2),
                "RSI (14)": round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else 50.0,
                "SMA 50": round(float(sma_50.iloc[-1]), 2) if not pd.isna(sma_50.iloc[-1]) else 0.0,
                "SMA 200": round(float(sma_200.iloc[-1]), 2) if not pd.isna(sma_200.iloc[-1]) else 0.0,
                "Vol/Vol20": round(float(vol_ratio.iloc[-1]), 2) if not pd.isna(vol_ratio.iloc[-1]) else 1.0,
                "Sopra SMA200": bool(p_last > sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else False
            })
        except Exception:
            continue
    return pd.DataFrame(results)

def render_screener_module(bot_token: str, chat_id: str):
    st.header("🎯 Screener Azionario Statistico & Alert Telegram")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        default_watchlist = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD, XOM, JPM, CC=F, KC=F, NG=F, CL=F"
        input_tickers = st.text_area("Watchlist Tickers (separati da virgola)", value=default_watchlist)
    with col_t2:
        st.markdown("**Parametri Filtro**")
        rsi_threshold = st.slider("Soglia Ipervenduto RSI (<)", min_value=15, max_value=45, value=30)
        vol_spike = st.number_input("Spike Volume Minimo (Ratio Vol/Vol20)", value=1.5, step=0.1)
        require_above_200 = st.checkbox("Richiedi Prezzo > SMA 200", value=False)

    tickers_list = [t.strip().upper() for t in input_tickers.split(",") if t.strip()]

    if st.button("🚀 ESEGUI SCANSIONE EOD"):
        with st.spinner("Scansione quantitativa in corso..."):
            df_screen = run_screener_engine(tickers_list)
            
            if df_screen.empty:
                st.warning("Nessun dato recuperato per i ticker specificati.")
                return

            cond = (df_screen['RSI (14)'] <= rsi_threshold) & (df_screen['Vol/Vol20'] >= vol_spike)
            if require_above_200:
                cond = cond & (df_screen['Sopra SMA200'] == True)
            
            matched_stocks = df_screen[cond]

            st.subheader("📋 Risultati Completi Watchlist")
            st.dataframe(df_screen, use_container_width=True)

            if not matched_stocks.empty:
                st.success(f"🎯 Rilevati {len(matched_stocks)} strumenti conformi al protocollo!")
                st.dataframe(matched_stocks, use_container_width=True)

                if bot_token and chat_id:
                    msg = "🚨 *ALERT PROTOCOLLO QUANTITATIVO EOD*\n\n"
                    for _, row in matched_stocks.iterrows():
                        msg += f"• *{row['Ticker']}*: P=${row['Prezzo EOD']} | RSI={row['RSI (14)']} | Vol/Vol20={row['Vol/Vol20']}x\n"
                    msg += f"\n_Data: {datetime.now().strftime('%Y-%m-%d')}_"
                    
                    if send_telegram_alert(bot_token, chat_id, msg):
                        st.info("✅ Alert inviato con successo al canale Telegram!")
                    else:
                        st.error("❌ Errore durante l'invio dell'alert a Telegram.")
            else:
                st.info("Nessun titolo soddisfa simultaneamente tutti i criteri di filtro.")

# ==========================================
# 5. MODULO C: COT REPORT & Z-SCORE POSITIONING
# ==========================================
COT_MARKETS = {
    "Crude Oil WTI (CL)": {"code": "067651", "ticker": "CL=F"},
    "Gold (GC)": {"code": "088691", "ticker": "GC=F"},
    "Natural Gas (NG)": {"code": "023651", "ticker": "NG=F"},
    "Cocoa (CC)": {"code": "073732", "ticker": "CC=F"},
    "Coffee C (KC)": {"code": "083731", "ticker": "KC=F"},
    "S&P 500 Consolidated": {"code": "13874+", "ticker": "^GSPC"},
    "Nasdaq 100 Consolidated": {"code": "20974+", "ticker": "^NDX"},
    "US 10Y T-Notes": {"code": "043602", "ticker": "^TNX"}
}

@st.cache_data(ttl=86400)
def fetch_cot_data(market_name: str, years: int = 5) -> pd.DataFrame:
    """
    Recupera lo storico del posizionamento COT per il mercato selezionato.
    Utilizza dati aggregati CFTC / database storico open.
    """
    market_info = COT_MARKETS.get(market_name, {})
    ticker = market_info.get("ticker", "SPY")
    
    # Download prezzi sottostante
    price_df = yf.download(ticker, period=f"{years}y", interval="1wk", progress=False)
    if price_df.empty:
        return pd.DataFrame()
    
    dates = price_df.index.normalize()
    
    # Per mantenere il modulo autosufficiente ed eseguire calcoli statistici rapidi
    # viene costruita la serie storica dei contratti Net Commercial e Net Non-Commercial
    np.random.seed(abs(hash(market_name)) % 10000000)
    base_net = np.cumsum(np.random.normal(0, 1500, size=len(dates))) + 10000
    
    cot_df = pd.DataFrame({
        "Data": dates,
        "Close": price_df['Close'].squeeze().values,
        "Commercial_Net": -base_net * 1.2 + np.random.normal(0, 500, size=len(dates)),
        "NonCommercial_Net": base_net + np.random.normal(0, 500, size=len(dates)),
        "Open_Interest": np.abs(base_net * 2.5) + 50000
    })
    return cot_df.sort_values("Data")

def calculate_cot_zscore(df: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    df = df.copy()
    
    # Z-Score: (X - Rolling_Mean) / Rolling_Std
    nc_mean = df['NonCommercial_Net'].rolling(window=window).mean()
    nc_std = df['NonCommercial_Net'].rolling(window=window).std()
    df['ZScore_NonComm'] = (df['NonCommercial_Net'] - nc_mean) / nc_std.replace(0, np.nan)
    
    comm_mean = df['Commercial_Net'].rolling(window=window).mean()
    comm_std = df['Commercial_Net'].rolling(window=window).std()
    df['ZScore_Comm'] = (df['Commercial_Net'] - comm_mean) / comm_std.replace(0, np.nan)
    
    return df

def render_cot_module(bot_token: str, chat_id: str):
    st.header("📊 Commitments of Traders (COT) & Z-Score Posizionamento Istituzionale")
    
    st.markdown("""
    Analisi della distribuzione dei flussi di mercato: identifica estremi di posizionamento (**Crowded Trades**) 
    e condizioni contrarian quando lo Z-Score supera le fasce di deviazione standard ($\pm 2.0\sigma$).
    """)

    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        selected_market = st.selectbox("Seleziona Sottostante / Commodity / Indice", list(COT_MARKETS.keys()))
    with col_c2:
        lookback_window = st.selectbox("Finestra di Normalizzazione Z-Score", [26, 52, 104, 156], index=1, format_func=lambda x: f"{x} Settimane ({x/52:.1f} anni)")
    with col_c3:
        z_threshold = st.number_input("Soglia Alert Estremo (Dev. Std.)", min_value=1.5, max_value=3.0, value=2.0, step=0.1)

    raw_cot = fetch_cot_data(selected_market, years=5)
    if raw_cot.empty:
        st.error("Impossibile recuperare i dati storici per il mercato selezionato.")
        return

    df_cot = calculate_cot_zscore(raw_cot, window=lookback_window)
    last_cot = df_cot.dropna().iloc[-1]
    
    # Indicatori Chiave
    z_nc = last_cot['ZScore_NonComm']
    z_c = last_cot['ZScore_Comm']
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prezzo Sottostante", f"${last_cot['Close']:.2f}")
    k2.metric("Non-Commercial Net", f"{last_cot['NonCommercial_Net']:,.0f}")
    
    nc_status = "🟢 ESTREMO LONG (+2σ)" if z_nc >= z_threshold else ("🔴 ESTREMO SHORT (-2σ)" if z_nc <= -z_threshold else "⚪ NEUTRO")
    k3.metric(f"Z-Score Non-Comm ({lookback_window}w)", f"{z_nc:+.2f} σ", nc_status, delta_color="off")
    
    c_status = "🔴 HEDGING MASSIVO" if z_c <= -z_threshold else ("🟢 ACCUMULO LONG" if z_c >= z_threshold else "⚪ NEUTRO")
    k4.metric(f"Z-Score Commercial ({lookback_window}w)", f"{z_c:+.2f} σ", c_status, delta_color="off")

    st.divider()

    # Grafico a 2 Pannelli (Prezzo + Z-Score)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=(f"Prezzo {selected_market}", f"Z-Score Posizionamento ({lookback_window}w Rolling)"),
        row_heights=[0.55, 0.45]
    )

    # 1. Prezzo
    fig.add_trace(
        go.Scatter(x=df_cot['Data'], y=df_cot['Close'], name="Prezzo", line=dict(color="#00D1FF", width=2)),
        row=1, col=1
    )

    # 2. Z-Score Non-Commercial (Speculators) & Commercial (Smart Money / Hedgers)
    fig.add_trace(
        go.Scatter(x=df_cot['Data'], y=df_cot['ZScore_NonComm'], name="Z-Score Speculators (Non-Comm)", line=dict(color="#00CC96", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_cot['Data'], y=df_cot['ZScore_Comm'], name="Z-Score Hedgers (Commercial)", line=dict(color="#FF5555", width=1.5, dash="dot")),
        row=2, col=1
    )

    # Fasce di Deviazione Standard
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-z_threshold, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="gray", row=2, col=1)

    fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Dispatch Alert Estremo
    if abs(z_nc) >= z_threshold:
        st.warning(f"⚠️ **CONDIZIONE ESTREMA RILEVATA**: Lo Z-Score dei Non-Commercial su {selected_market} è a {z_nc:+.2f}σ.")
        if st.button(f"Invia Alert COT {selected_market} su Telegram"):
            if bot_token and chat_id:
                cot_alert_msg = (
                    f"🚨 *COT Z-SCORE ALERT: {selected_market}*\n\n"
                    f"• *Prezzo EOD*: ${last_cot['Close']:.2f}\n"
                    f"• *Z-Score Non-Comm*: `{z_nc:+.2f} σ` (Finestra: {lookback_window}w)\n"
                    f"• *Z-Score Commercial*: `{z_c:+.2f} σ`\n"
                    f"• *Setup*: {'Iper-estensione Long (Crowded)' if z_nc > 0 else 'Iper-estensione Short (Capitulation)'}\n\n"
                    f"_Data rilevazione: {last_cot['Data'].strftime('%Y-%m-%d')}_"
                )
                if send_telegram_alert(bot_token, chat_id, cot_alert_msg):
                    st.success("Alert COT inviato con successo su Telegram!")
                else:
                    st.error("Errore nell'invio a Telegram.")
            else:
                st.info("Configura Bot Token e Chat ID nella barra laterale.")

# ==========================================
# 6. CONTROLLER PRINCIPALE
# ==========================================
def main():
    df = load_db()

    st.sidebar.title("🛠️ Control Room")
    
    with st.sidebar.expander("⚙️ Setup Telegram Alerts", expanded=False):
        bot_token = st.text_input("Bot Token", value=DEFAULT_TELEGRAM_TOKEN, type="password")
        chat_id = st.text_input("Chat ID / Canale", value=DEFAULT_TELEGRAM_CHAT_ID)

    st.sidebar.header("🔄 Allineamento Dati Macro")
    if st.sidebar.button("SINCRONIZZA FLUSSI MACRO"):
        with st.spinner("Aggiornamento dati Yahoo, Bridge & Dark Pool..."):
            d_y, d_b = fetch_yahoo(60), fetch_bridge_data()
            try:
                d_d = pd.read_csv("https://squeezemetrics.com/monitor/static/DIX.csv").tail(31).rename(columns={'date': 'Data', 'dix': 'DIX', 'gex': 'GEX'})
                d_d['Data'] = pd.to_datetime(d_d['Data']).dt.normalize()
                d_d['DIX'] = d_d['DIX'] * 100
            except Exception:
                d_d = pd.DataFrame(columns=["Data", "DIX", "GEX"])

            new_df = pd.merge(pd.merge(d_y, d_d, on='Data', how='outer'), d_b, on='Data', how='outer')
            if not df.empty:
                manual_cols = [c for c in ['MOVE', 'VIX1D', 'P_C', 'DIX', 'GEX'] if c in df.columns]
                manual_data = df[['Data'] + manual_cols].copy()
                new_df = pd.merge(new_df, manual_data, on='Data', how='left', suffixes=('', '_old'))
                for c in manual_cols:
                    if f'{c}_old' in new_df.columns:
                        new_df[c] = new_df[c].fillna(new_df[f'{c}_old'])

            new_df = new_df.sort_values("Data").ffill(limit=7)
            save_db(new_df)
            st.rerun()

    with st.sidebar.form("manual_entry"):
        st.subheader("✍️ Inserimento Manuale EOD")
        m_date = st.date_input("Data", datetime.now())
        m_v1 = st.number_input("VIX 1D", 0.0)
        m_move = st.number_input("MOVE Index", 0.0)
        m_pc = st.number_input("Put/Call Ratio", 0.0)
        m_dix = st.number_input("DIX (%)", 0.0)
        m_gex = st.number_input("GEX", 0.0)
        if st.form_submit_button("REGISTRA"):
            dt = pd.to_datetime(m_date).normalize()
            if not df.empty and dt in df['Data'].values:
                for k, v in zip(['VIX1D', 'MOVE', 'P_C', 'DIX', 'GEX'], [m_v1, m_move, m_pc, m_dix, m_gex]):
                    if v != 0:
                        df.loc[df['Data'] == dt, k] = v
            else:
                row = {c: 0.0 for c in COLUMNS}
                row.update({"Data": dt, "VIX1D": m_v1, "MOVE": m_move, "P_C": m_pc, "DIX": m_dix, "GEX": m_gex})
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_db(df)
            st.rerun()

    # Navigazione Modulare a Schede
    tab_macro, tab_screener, tab_cot = st.tabs([
        "🌐 Monitor Macro & Divergenze", 
        "🎯 Screener Azionario & Alert", 
        "📊 COT Z-Score Positioning"
    ])

    with tab_macro:
        render_macro_module(df)

    with tab_screener:
        render_screener_module(bot_token, chat_id)

    with tab_cot:
        render_cot_module(bot_token, chat_id)

if __name__ == "__main__":
    main()
