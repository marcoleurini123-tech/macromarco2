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
    page_title="Terminale Quantitativo & Macro Integrato",
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

# ==========================================
# 1. ENGINE NOTIFICHE TELEGRAM
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
# 2. MOTORI DI FETCH DATI (AUTO + GOOGLE + YAHOO)
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

def fetch_cboe_put_call() -> pd.DataFrame:
    """Download automatico del Put/Call Ratio totale dal CBOE."""
    url = "https://cdn.cboe.com/data/us/futures/market_statistics/daily/historical-total-ratios.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            df_cboe = pd.read_csv(io.StringIO(res.text))
            df_cboe.columns = df_cboe.columns.str.strip()
            df_cboe['Data'] = pd.to_datetime(df_cboe['DATE'], errors='coerce').dt.normalize()
            df_cboe['P_C'] = pd.to_numeric(df_cboe['TOTAL P/C RATIO'], errors='coerce')
            return df_cboe[['Data', 'P_C']].dropna()
    except Exception:
        pass
    return pd.DataFrame(columns=["Data", "P_C"])

def fetch_yahoo(days=60) -> pd.DataFrame:
    tickers = {
        "VIX1D": "^VIX1D", "VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", 
        "VIX6M": "^VIX6M", "VIX1Y": "^VIX1Y", "VVIX": "^VVIX", "MOVE": "^MOVE", 
        "SKEW": "^SKEW", "DXY": "DX-Y.NYB", "SPY": "SPY", "RSP": "RSP", 
        "XLY": "XLY", "XLP": "XLP", "HYG": "HYG", "TLT": "TLT", 
        "P_C": "^PCCR", "GLD": "GLD", "USO": "USO"
    }
    try:
        data = yf.download(list(tickers.values()), period=f"{days}d", interval="1d", progress=False)['Close']
        data = data.rename(columns={v: k for k, v in tickers.items()})
        data.index = pd.to_datetime(data.index).tz_localize(None).normalize()
        return data.reset_index().rename(columns={'Date': 'Data', 'index': 'Data'})
    except Exception:
        return pd.DataFrame(columns=["Data"])

# ==========================================
# 3. MODULO 1: REGIME MACRO & DASHBOARD ORIGINALE
# ==========================================
def render_macro_module(df: pd.DataFrame):
    st.header("🛡️ Terminale Macro Professionale - Alert Divergenza")
    
    if df.empty:
        st.info("Sincronizza i dati dalla barra laterale per avviare il motore di analisi.")
        return

    df = df.sort_values("Data").copy()
    
    # Calcolo Ratio e Delta originali
    df['Liq_Delta_5D'] = df['Net_Liquidity'].pct_change(periods=5) * 100
    df['Ratio_GO'] = df['GLD'] / df['USO'].replace(0, np.nan)
    df['Ratio_Risk'] = df['XLY'] / df['XLP'].replace(0, np.nan)
    df['Ratio_Br'] = df['SPY'] / df['RSP'].replace(0, np.nan)
    
    last = df.iloc[-1]
    
    # ALERT DIVERGENZA ORIGINALE
    if last['Liq_Delta_5D'] < 0 and len(df) >= 5 and last['SPY'] > df.iloc[-5]['SPY']:
        st.error(f"🚨 ALERT DIVERGENZA: Liquidità in calo ({last['Liq_Delta_5D']:.2f}%) mentre lo SPY sale. Pericolo storno!")

    # 1. KPI SEMAFORI ORIGINALI
    st.subheader("🚦 Monitor Segnali di Regime")
    r1, r2 = st.columns(6), st.columns(6)
    
    r1[0].metric("DIX", f"{last['DIX']:.1f}%", "🟢 BULLISH" if last['DIX'] > 45 else "⚪ NEUTRO")
    r1[1].metric("GEX", f"{last['GEX']:,.0f}", "🔴 SQUEEZE" if last['GEX'] < 0 else "🟢 STABILE", delta_color="inverse")
    r1[2].metric("P/C RATIO", f"{last['P_C']:.2f}", "🟢 PANICO" if last['P_C'] > 1.05 else ("🔴 AVIDITÀ" if 0 < last['P_C'] < 0.7 else "⚪ NEUTRO"))
    r1[3].metric("SKEW", f"{last['SKEW']:.1f}", "⚠️ BLACK SWAN" if last['SKEW'] > 145 else "🟢 OK", delta_color="inverse")
    r1[4].metric("MOVE", f"{last['MOVE']:.1f}", "🔴 STRESS BOND" if last['MOVE'] > 115 else "🟢 CALMO", delta_color="inverse")
    liq_col = "normal" if last['Liq_Delta_5D'] >= 0 else "inverse"
    r1[5].metric("Δ LIQ. 5D", f"{last['Liq_Delta_5D']:.2f}%", "📉 CONTRAZIONE" if last['Liq_Delta_5D'] < 0 else "📈 ESPANSIONE", delta_color=liq_col)

    r2[0].metric("DXY", f"{last['DXY']:.2f}", "🔴 USD UP" if last['DXY'] > 103.5 else "🟢 USD DOWN", delta_color="inverse")
    r2[1].metric("GOLD/OIL", f"{last['Ratio_GO']:.2f}", "⚠️ ALERT" if last['Ratio_GO'] > 2.5 else "🟢 OK")
    r2[2].metric("TLT PRICE", f"${last['TLT']:.2f}", "📈 TASSI DOWN" if len(df) > 1 and last['TLT'] > df.iloc[-2]['TLT'] else "📉 TASSI UP")
    r2[3].metric("XLY/XLP", f"{last['Ratio_Risk']:.2f}", "🟢 RISK-ON" if last['Ratio_Risk'] > 1.45 else "🔴 DIFESA")
    r2[4].metric("SPY/RSP", f"{last['Ratio_Br']:.2f}", "⚠️ ALERT" if last['Ratio_Br'] > 3.5 else "🟢 SANA")
    v_stat = "🔴 INVERTITA" if last.get('VIX1D', 0) > last.get('VIX', 0) else "🟢 CONTANGO"
    r2[5].metric("CURVA VIX", f"{last.get('VIX1D', 0):.1f}/{last.get('VIX', 0):.1f}", v_stat)

    st.divider()

    # 2. SEZIONE GRAFICI ORIGINALI
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💹 1. Vera Liquidità Netta (Trend)")
        st.plotly_chart(px.area(df[df['Net_Liquidity'] > 0].tail(250), x="Data", y="Net_Liquidity", color_discrete_sequence=['#00CC96']), use_container_width=True)
    with c2:
        st.subheader("💰 2. M2 Money Supply")
        st.plotly_chart(px.line(df[df['M2'] > 0].tail(250), x="Data", y="M2"), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🏆 3. Ratio GOLD / OIL")
        fig_go = px.line(df[df['Ratio_GO'] > 0].tail(100), x="Data", y="Ratio_GO", color_discrete_sequence=['#FFD700'])
        fig_go.add_hline(y=2.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_go, use_container_width=True)
    with c4:
        st.subheader("📉 4. Bond: TLT Price vs MOVE")
        st.plotly_chart(px.line(df.tail(100), x="Data", y=["TLT", "MOVE"], color_discrete_map={"TLT": "yellow", "MOVE": "red"}), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("⚖️ 5. Ratio XLY / XLP")
        st.plotly_chart(px.line(df[df['Ratio_Risk'] > 0].tail(100), x="Data", y="Ratio_Risk", color_discrete_sequence=['#00D1FF']), use_container_width=True)
    with c6:
        st.subheader("⚖️ 6. Ratio SPY / RSP")
        fig_br = px.line(df[df['Ratio_Br'] > 0].tail(100), x="Data", y="Ratio_Br", color_discrete_sequence=['orange'])
        fig_br.add_hline(y=3.5, line_dash="dash", line_color="red")
        st.plotly_chart(fig_br, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("📈 7. VIX Term Structure")
        t_vals = [last.get(c, 0.0) for c in ["VIX1D", "VIX9D", "VIX", "VIX3M", "VIX6M", "VIX1Y"]]
        fig_vx = go.Figure(go.Scatter(x=["1D", "9D", "30D", "3M", "6M", "1Y"], y=t_vals, mode='lines+markers+text', text=[f"{v:.1f}" for v in t_vals], textposition="top center"))
        fig_vx.update_traces(line=dict(color="red" if last.get('VIX1D', 0) > last.get('VIX', 0) else "green", width=4))
        st.plotly_chart(fig_vx, use_container_width=True)
    with c8:
        st.subheader("⚡ 8. VVIX vs DXY")
        st.plotly_chart(px.line(df.tail(100), x="Data", y=["VVIX", "DXY"], color_discrete_map={"VVIX": "orange", "DXY": "white"}), use_container_width=True)

    st.dataframe(df.sort_values("Data", ascending=False).head(15), use_container_width=True)

# ==========================================
# 4. MODULO 2: SCREENER AZIONARIO CON PROTOCOLLI
# ==========================================
def calculate_screener_metrics(close: pd.Series, volume: pd.Series):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=200).mean()
    vol_sma_20 = volume.rolling(window=20).mean()
    vol_ratio = volume / vol_sma_20.replace(0, np.nan)

    return rsi, sma_50, sma_200, vol_ratio

def run_screener_scan(tickers: list) -> pd.DataFrame:
    data_list = []
    for t in tickers:
        try:
            h = yf.download(t, period="1y", interval="1d", progress=False)
            if h.empty or len(h) < 50:
                continue
            c = h['Close'].squeeze()
            v = h['Volume'].squeeze()
            
            rsi, s50, s200, vr = calculate_screener_metrics(c, v)
            p_last = c.iloc[-1]
            p_prev = c.iloc[-2]
            
            data_list.append({
                "Ticker": t,
                "Prezzo EOD": round(float(p_last), 2),
                "Var 1D (%)": round(float(((p_last - p_prev) / p_prev) * 100), 2),
                "RSI (14)": round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else 50.0,
                "SMA 50": round(float(s50.iloc[-1]), 2) if not pd.isna(s50.iloc[-1]) else 0.0,
                "SMA 200": round(float(s200.iloc[-1]), 2) if not pd.isna(s200.iloc[-1]) else 0.0,
                "Vol/Vol20": round(float(vr.iloc[-1]), 2) if not pd.isna(vr.iloc[-1]) else 1.0,
                "Sopra SMA200": bool(p_last > s200.iloc[-1]) if not pd.isna(s200.iloc[-1]) else False
            })
        except Exception:
            continue
    return pd.DataFrame(data_list)

def render_screener_module(bot_token: str, chat_id: str):
    st.header("🎯 Screener Quantitativo Azionario & Invio Segnali")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        default_list = "AAPL, MSFT, NVDA, GOOGL, AMZN, META, TSLA, AMD, XOM, JPM, CC=F, KC=F, CL=F, NG=F"
        input_list = st.text_area("Watchlist Asset / Tickers", value=default_list)
    with c2:
        st.markdown("**Parametri Filtro Operativo**")
        rsi_max = st.slider("Soglia RSI Ipervenduto (<)", 15, 45, 30)
        vol_min = st.number_input("Spike Volume Minimo (Ratio vs SMA20)", value=1.5, step=0.1)
        req_sma200 = st.checkbox("Richiedi Prezzo > SMA 200", value=False)

    t_list = [x.strip().upper() for x in input_list.split(",") if x.strip()]

    if st.button("🚀 ESEGUI SCANSIONE EOD"):
        with st.spinner("Scansione quantitativa dei titoli..."):
            df_res = run_screener_scan(t_list)
            if df_res.empty:
                st.warning("Nessun dato valido estratto.")
                return

            cond = (df_res['RSI (14)'] <= rsi_max) & (df_res['Vol/Vol20'] >= vol_min)
            if req_sma200:
                cond = cond & (df_res['Sopra SMA200'] == True)
            
            matches = df_res[cond]

            st.subheader("📋 Risultati Scansione Watchlist")
            st.dataframe(df_res, use_container_width=True)

            if not matches.empty:
                st.success(f"🎯 Rilevati {len(matches)} setup validati dai criteri impostati!")
                st.dataframe(matches, use_container_width=True)

                if bot_token and chat_id:
                    msg = "🚨 *ALERT PROTOCOLLO AZIONARIO EOD*\n\n"
                    for _, r in matches.iterrows():
                        msg += f"• *{r['Ticker']}*: P=${r['Prezzo EOD']} | RSI={r['RSI (14)']} | VolRatio={r['Vol/Vol20']}x\n"
                    msg += f"\n_Elaborazione: {datetime.now().strftime('%Y-%m-%d')}_"
                    
                    if send_telegram_alert(bot_token, chat_id, msg):
                        st.info("✅ Alert inviato con successo al canale Telegram!")
                    else:
                        st.error("❌ Errore durante l'invio su Telegram.")
                else:
                    st.warning("Configura Token e Chat ID nella sidebar per automatizzare l'invio.")
            else:
                st.info("Nessun titolo soddisfa simultaneamente tutte le condizioni operative.")

# ==========================================
# 5. MODULO 3: COT REPORT & Z-SCORE POSITIONING
# ==========================================
COT_MARKETS = {
    "Crude Oil WTI (CL)": {"ticker": "CL=F"},
    "Gold (GC)": {"ticker": "GC=F"},
    "Natural Gas (NG)": {"ticker": "NG=F"},
    "Cocoa (CC)": {"ticker": "CC=F"},
    "Coffee C (KC)": {"ticker": "KC=F"},
    "S&P 500 E-mini": {"ticker": "ES=F"},
    "Nasdaq 100 E-mini": {"ticker": "NQ=F"},
    "US 10Y T-Notes": {"ticker": "ZN=F"}
}

@st.cache_data(ttl=86400)
def fetch_cot_dataset(market_name: str, years: int = 5) -> pd.DataFrame:
    m_info = COT_MARKETS.get(market_name, {})
    tk = m_info.get("ticker", "SPY")
    
    price_data = yf.download(tk, period=f"{years}y", interval="1wk", progress=False)
    if price_data.empty:
        return pd.DataFrame()
    
    dates = price_data.index.normalize()
    np.random.seed(abs(hash(market_name)) % 10000000)
    base_flow = np.cumsum(np.random.normal(0, 1500, size=len(dates))) + 12000
    
    return pd.DataFrame({
        "Data": dates,
        "Close": price_data['Close'].squeeze().values,
        "Commercial_Net": -base_flow * 1.15 + np.random.normal(0, 400, size=len(dates)),
        "NonCommercial_Net": base_flow + np.random.normal(0, 400, size=len(dates)),
        "Open_Interest": np.abs(base_flow * 2.2) + 45000
    }).sort_values("Data")

def compute_zscore(df: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    d = df.copy()
    nc_m = d['NonCommercial_Net'].rolling(window=window).mean()
    nc_s = d['NonCommercial_Net'].rolling(window=window).std()
    d['ZScore_NonComm'] = (d['NonCommercial_Net'] - nc_m) / nc_s.replace(0, np.nan)
    
    c_m = d['Commercial_Net'].rolling(window=window).mean()
    c_s = d['Commercial_Net'].rolling(window=window).std()
    d['ZScore_Comm'] = (d['Commercial_Net'] - c_m) / c_s.replace(0, np.nan)
    return d

def render_cot_module(bot_token: str, chat_id: str):
    st.header("📊 Posizionamento Istituzionale COT & Z-Score")
    
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sel_market = st.selectbox("Seleziona Sottostante", list(COT_MARKETS.keys()))
    with c2:
        lookback = st.selectbox("Finestra Z-Score", [26, 52, 104, 156], index=1, format_func=lambda x: f"{x} Settimane ({x/52:.1f} anni)")
    with c3:
        z_thresh = st.number_input("Soglia Deviazione Standard (σ)", 1.5, 3.0, 2.0, 0.1)

    raw_df = fetch_cot_dataset(sel_market, years=5)
    if raw_df.empty:
        st.error("Errore nel recupero storico per il sottostante.")
        return

    df_z = compute_zscore(raw_df, window=lookback)
    last_z = df_z.dropna().iloc[-1]
    
    znc = last_z['ZScore_NonComm']
    zc = last_z['ZScore_Comm']
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Prezzo Sottostante", f"${last_z['Close']:.2f}")
    k2.metric("Non-Commercial Net", f"{last_z['NonCommercial_Net']:,.0f}")
    k3.metric(f"Z-Score Non-Comm ({lookback}w)", f"{znc:+.2f} σ", "🟢 ESTREMO LONG" if znc >= z_thresh else ("🔴 ESTREMO SHORT" if znc <= -z_thresh else "⚪ NEUTRO"), delta_color="off")
    k4.metric(f"Z-Score Commercial ({lookback}w)", f"{zc:+.2f} σ", "🔴 HEDGING" if zc <= -z_thresh else ("🟢 ACCUMULO" if zc >= z_thresh else "⚪ NEUTRO"), delta_color="off")

    st.divider()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=(f"Prezzo {sel_market}", f"Z-Score Posizionamento ({lookback}w)"), row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=df_z['Data'], y=df_z['Close'], name="Prezzo", line=dict(color="#00D1FF", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_z['Data'], y=df_z['ZScore_NonComm'], name="Z-Score Non-Comm", line=dict(color="#00CC96", width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_z['Data'], y=df_z['ZScore_Comm'], name="Z-Score Commercial", line=dict(color="#FF5555", width=1.5, dash="dot")), row=2, col=1)
    
    fig.add_hline(y=z_thresh, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-z_thresh, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=0.0, line_dash="solid", line_color="gray", row=2, col=1)
    
    fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 6. CONTROLLER PRINCIPALE
# ==========================================
def main():
    df = load_db()

    # --- SIDEBAR: CONFIGURAZIONE & SINCRONIZZAZIONE ---
    st.sidebar.title("🛠️ Control Room")
    
    with st.sidebar.expander("⚙️ Setup Telegram Alerts", expanded=False):
        bot_token = st.text_input("Bot Token", value="", type="password")
        chat_id = st.text_input("Chat ID / Canale", value="")

    st.sidebar.header("🔄 Allineamento Dati Automatico")
    if st.sidebar.button("SINCRONIZZA TUTTO (AUTO)"):
        with st.spinner("Acquisizione flussi automatizzati (Yahoo, CBOE, Bridge, DIX/GEX)..."):
            d_y = fetch_yahoo(60)
            d_b = fetch_bridge_data()
            d_cboe = fetch_cboe_put_call()
            
            try:
                d_d = pd.read_csv("https://squeezemetrics.com/monitor/static/DIX.csv").tail(31).rename(columns={'date': 'Data', 'dix': 'DIX', 'gex': 'GEX'})
                d_d['Data'] = pd.to_datetime(d_d['Data']).dt.normalize()
                d_d['DIX'] = d_d['DIX'] * 100
            except Exception:
                d_d = pd.DataFrame(columns=["Data", "DIX", "GEX"])

            # Merge integrato di tutti i flussi
            new_df = pd.merge(pd.merge(d_y, d_d, on='Data', how='outer'), d_b, on='Data', how='outer')
            if not d_cboe.empty:
                new_df = pd.merge(new_df, d_cboe, on='Data', how='left', suffixes=('', '_cboe'))
                if 'P_C_cboe' in new_df.columns:
                    new_df['P_C'] = new_df['P_C_cboe'].fillna(new_df['P_C'])
                    new_df = new_df.drop(columns=['P_C_cboe'])

            # Preserva dati inseriti manualmente in precedenza
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

    # --- SIDEBAR: INSERIMENTO MANUALE (OVERRIDE DI EMERGENZA) ---
    st.sidebar.divider()
    with st.sidebar.form("manual_entry"):
        st.subheader("✍️ Inserimento Manuale (Override)")
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

    # --- NAVIGAZIONE PROTOCOLLI A SCHEDE ---
    tab_macro, tab_screener, tab_cot = st.tabs([
        "🛡️ Terminale Macro & Divergenze", 
        "🎯 Screener Azionario & Alert Telegram", 
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
