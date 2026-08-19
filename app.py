import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

# =============================================================================
# 0. CONFIGURAZIONE PAGINA
# =============================================================================
st.set_page_config(
    page_title="Macro Quant Terminal",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. CORE ENGINE: ARCHITETTURA A PLUGIN (BASE STUDY)
# =============================================================================
class BaseStudy:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def calculate(self, data, params=None):
        raise NotImplementedError

    def evaluate_signal(self, calculated_data, params=None):
        raise NotImplementedError

# =============================================================================
# 2. STUDI ANALITICI MODULARI (PLUGINS)
# =============================================================================
class StatisticalZScoreStudy(BaseStudy):
    def __init__(self):
        super().__init__("Z-Score & RSI", "Z-Score su media mobile a 252 periodi e RSI a 14")

    def calculate(self, df: pd.DataFrame, params=None) -> pd.DataFrame:
        params = params or {}
        window = params.get("zscore_window", 252)
        rsi_win = params.get("rsi_window", 14)
        df = df.copy()

        # Z-Score Prezzo rispetto alla SMA 252
        sma = df["Close"].rolling(window=window, min_periods=30).mean()
        std = df["Close"].rolling(window=window, min_periods=30).std()
        df["ZScore"] = (df["Close"] - sma) / std.replace(0, np.nan)

        # Calcolo RSI
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(rsi_win, min_periods=rsi_win).mean()
        avg_loss = loss.rolling(rsi_win, min_periods=rsi_win).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    def evaluate_signal(self, df: pd.DataFrame, params=None) -> dict:
        params = params or {}
        z_thresh = params.get("zscore_threshold", 1.8)
        latest = df.iloc[-1]
        z_val = float(latest["ZScore"]) if not pd.isna(latest["ZScore"]) else 0.0
        rsi_val = float(latest["RSI"]) if not pd.isna(latest["RSI"]) else 50.0

        is_oversold = (z_val <= -z_thresh) or (rsi_val <= 30)
        is_overbought = (z_val >= z_thresh) or (rsi_val >= 70)

        return {
            "is_triggered": is_oversold or is_overbought,
            "zscore": round(z_val, 2),
            "rsi": round(rsi_val, 2),
            "close": round(float(latest["Close"]), 2),
            "condition": "OVERSOLD" if is_oversold else ("OVERBOUGHT" if is_overbought else "NEUTRAL")
        }

class VolumeProfilePOCStudy(BaseStudy):
    def __init__(self):
        super().__init__("Volume Profile & POC", "Calcolo del Point of Control (POC) e Value Area")

    def calculate(self, df: pd.DataFrame, params=None) -> dict:
        params = params or {}
        lookback = min(len(df), params.get("vp_days", 120))
        bins = params.get("vp_bins", 40)
        sub_df = df.tail(lookback).copy()

        p_min, p_max = sub_df["Low"].min(), sub_df["High"].max()
        counts, bin_edges = np.histogram(sub_df["Close"], bins=bins, range=(p_min, p_max), weights=sub_df["Volume"])

        poc_idx = int(np.argmax(counts))
        poc_price = float((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2.0)

        target_vol = counts.sum() * 0.70
        sorted_indices = np.argsort(counts)[::-1]
        accum, va_bins = 0, []
        for idx in sorted_indices:
            accum += counts[idx]
            va_bins.append(idx)
            if accum >= target_vol:
                break

        val_price = float(bin_edges[min(va_bins)])
        vah_price = float(bin_edges[max(va_bins) + 1])

        return {
            "poc": poc_price, "vah": vah_price, "val": val_price,
            "latest_close": float(df["Close"].iloc[-1])
        }

    def evaluate_signal(self, profile_data: dict, params=None) -> dict:
        close = profile_data["latest_close"]
        poc = profile_data["poc"]
        dist_poc_pct = ((close - poc) / poc) * 100
        is_at_poc = abs(dist_poc_pct) <= 2.0

        return {
            "is_triggered": is_at_poc,
            "poc": round(poc, 2),
            "vah": round(profile_data["vah"], 2),
            "val": round(profile_data["val"], 2),
            "distance_to_poc_pct": round(dist_poc_pct, 2),
            "status": "POC_TEST_SUPPORT" if (is_at_poc and close >= poc) else ("POC_TEST_RESISTANCE" if is_at_poc else "INSIDE_RANGE")
        }

class COTPositioningStudy(BaseStudy):
    def __init__(self):
        super().__init__("COT Positioning", "Z-Score e Percentile normalizzato COT su Commercials e Speculatori")

    def calculate(self, cot_df: pd.DataFrame, params=None) -> pd.DataFrame:
        params = params or {}
        z_win = params.get("cot_zscore_window", 52)
        idx_win = params.get("cot_index_window", 156)

        df = cot_df.copy().sort_index()
        df["NonComm_Net"] = df["noncomm_long"] - df["noncomm_short"]
        df["Comm_Net"] = df["comm_long"] - df["comm_short"]

        # Z-Score Rolling
        df["ZScore_NonComm"] = (df["NonComm_Net"] - df["NonComm_Net"].rolling(z_win, min_periods=10).mean()) / df["NonComm_Net"].rolling(z_win, min_periods=10).std()
        df["ZScore_Comm"] = (df["Comm_Net"] - df["Comm_Net"].rolling(z_win, min_periods=10).mean()) / df["Comm_Net"].rolling(z_win, min_periods=10).std()

        # COT Index Percentile
        def calc_idx(s, w):
            rmin = s.rolling(w, min_periods=10).min()
            rmax = s.rolling(w, min_periods=10).max()
            denom = rmax - rmin
            return np.where(denom > 0, ((s - rmin) / denom) * 100, 50.0)

        df["COT_Index_Comm"] = calc_idx(df["Comm_Net"], idx_win)
        df["COT_Index_NonComm"] = calc_idx(df["NonComm_Net"], idx_win)
        return df

    def evaluate_signal(self, df: pd.DataFrame, params=None) -> dict:
        latest = df.iloc[-1]
        z_comm = float(latest["ZScore_Comm"]) if not pd.isna(latest["ZScore_Comm"]) else 0.0
        z_noncomm = float(latest["ZScore_NonComm"]) if not pd.isna(latest["ZScore_NonComm"]) else 0.0
        idx_comm = float(latest["COT_Index_Comm"]) if not pd.isna(latest["COT_Index_Comm"]) else 50.0

        is_accum = (z_comm >= 1.5) or (z_noncomm <= -1.2)
        is_dist = (z_comm <= -1.5) or (z_noncomm >= 1.2)

        return {
            "is_triggered": is_accum or is_dist,
            "zscore_comm": round(z_comm, 2),
            "zscore_noncomm": round(z_noncomm, 2),
            "cot_index_comm": round(idx_comm, 1),
            "comm_net": int(latest["Comm_Net"]),
            "noncomm_net": int(latest["NonComm_Net"])
        }

# =============================================================================
# 3. PROTOCOLLO COMBINATO: CONVERGENZA COT + POC
# =============================================================================
class COTPOCConvergenceProtocol:
    def __init__(self):
        self.name = "Convergenza Istituzionale COT + POC"
        self.zscore_study = StatisticalZScoreStudy()
        self.poc_study = VolumeProfilePOCStudy()
        self.cot_study = COTPositioningStudy()

    def evaluate(self, price_df: pd.DataFrame, cot_df: pd.DataFrame = None, params=None) -> dict:
        params = params or {}
        df_p = self.zscore_study.calculate(price_df, params)
        z_res = self.zscore_study.evaluate_signal(df_p, params)

        poc_data = self.poc_study.calculate(price_df, params)
        poc_res = self.poc_study.evaluate_signal(poc_data, params)

        cot_res = None
        if cot_df is not None and not cot_df.empty:
            df_c = self.cot_study.calculate(cot_df, params)
            cot_res = self.cot_study.evaluate_signal(df_c, params)

        p_z = z_res["zscore"]
        rsi = z_res["rsi"]
        dist_poc = poc_res["distance_to_poc_pct"]
        is_at_poc = abs(dist_poc) <= 2.0
        c_z_comm = cot_res["zscore_comm"] if cot_res else 0.0

        is_long = (c_z_comm >= 1.2 or not cot_res) and (p_z <= -1.2) and is_at_poc and (rsi <= 45)
        is_short = (c_z_comm <= -1.2 or not cot_res) and (p_z >= 1.2) and is_at_poc and (rsi >= 60)

        sig = "NEUTRAL"
        if is_long:
            sig = "CONVERGENCE_LONG_ACCUMULATION"
        elif is_short:
            sig = "CONVERGENCE_SHORT_DISTRIBUTION"

        return {
            "is_triggered": is_long or is_short,
            "signal_type": sig,
            "close": z_res["close"],
            "poc": poc_res["poc"],
            "vah": poc_res["vah"],
            "val": poc_res["val"],
            "distance_to_poc": f"{dist_poc:+.2f}%",
            "price_zscore": p_z,
            "rsi": rsi,
            "cot_z_comm": c_z_comm if cot_res else "N/A",
            "cot_z_noncomm": cot_res["zscore_noncomm"] if cot_res else "N/A",
            "cot_idx_comm": cot_res["cot_index_comm"] if cot_res else "N/A",
            "date": price_df.index[-1].strftime("%Y-%m-%d")
        }

# =============================================================================
# 4. TELEGRAM NOTIFIER ENGINE
# =============================================================================
class TelegramNotifier:
    @staticmethod
    def send_alert(bot_token: str, chat_id: str, ticker: str, protocol_name: str, metrics: dict) -> bool:
        if not bot_token or not chat_id:
            return False

        lines = [
            f"🚨 *TRIGGER QUANTITATIVO: {ticker}*",
            f"📋 *Protocollo:* `{protocol_name}`",
            f"📅 *Data EOD:* `{metrics.get('date', datetime.today().strftime('%Y-%m-%d'))}`",
            "───────────────────"
        ]
        for k, v in metrics.items():
            if k not in ["is_triggered", "date"]:
                formatted_k = k.replace('_', ' ').capitalize()
                lines.append(f"• *{formatted_k}:* `{v}`")

        msg = "\n".join(lines)
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        try:
            r = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=8)
            return r.status_code == 200
        except Exception:
            return False

# =============================================================================
# 5. DATA INGESTION & CACHING
# =============================================================================
@st.cache_data(ttl=3600 * 6)
def fetch_market_eod(ticker: str) -> pd.DataFrame:
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty and len(df) > 50:
            return df
    except Exception:
        pass

    # Fallback sintetico
    dates = pd.date_range(end=datetime.today(), periods=300, freq="B")
    np.random.seed(abs(hash(ticker)) % 1000)
    base_p = 100.0 + np.cumsum(np.random.randn(300) * 1.5)
    return pd.DataFrame({
        "Open": base_p, "High": base_p + 1.2, "Low": base_p - 1.2, "Close": base_p,
        "Volume": np.random.randint(500000, 2000000, 300)
    }, index=dates)

@st.cache_data(ttl=3600 * 12)
def fetch_cot_data(asset_key: str) -> pd.DataFrame:
    dates = pd.date_range(end=datetime.today(), periods=160, freq="W-FRI")
    np.random.seed(abs(hash(asset_key)) % 1000)
    return pd.DataFrame({
        "noncomm_long": np.linspace(40000, 20000, 160) + np.random.randint(-1000, 1000, 160),
        "noncomm_short": np.linspace(20000, 50000, 160) + np.random.randint(-1000, 1000, 160),
        "comm_long": np.linspace(30000, 65000, 160) + np.random.randint(-1000, 1000, 160),
        "comm_short": np.linspace(50000, 25000, 160) + np.random.randint(-1000, 1000, 160),
        "open_interest": [200000] * 160
    }, index=dates)

# =============================================================================
# 6. INTERFACCIA STREAMLIT
# =============================================================================
st.sidebar.title("🏛️ Terminale Quantitativo")
st.sidebar.markdown("---")

protocol_choice = st.sidebar.radio(
    "Seleziona Protocollo / Vista",
    [
        "🌐 1. Regime Macro & Liquidità Fed",
        "🎯 2. Studio Convergenza COT + POC",
        "📡 3. Screener EOD & Alert Telegram"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configurazione Alert Telegram")
# Lettura automatica dai secrets di Streamlit se configurati, altrimenti da input UI
default_token = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
default_chat = st.secrets.get("TELEGRAM_CHAT_ID", "")
tg_token = st.sidebar.text_input("Bot Token", value=default_token, type="password")
tg_chat = st.sidebar.text_input("Chat / Channel ID", value=default_chat)

# --- VISTA 1: REGIME MACRO ---
if "1. Regime Macro" in protocol_choice:
    st.title("🌐 Regime Macroeconomico & Liquidità Fed")
    st.caption("Monitoraggio delle condizioni monetarie globali: Total Assets, Reverse Repo, TGA e Yield Curve.")

    dates = pd.date_range(end=datetime.today(), periods=160, freq="W-FRI")
    df_macro = pd.DataFrame({
        "WALCL": np.linspace(8500, 7100, 160) + np.random.randn(160) * 15,
        "TGA": np.linspace(500, 750, 160) + np.random.randn(160) * 20,
        "RRP": np.linspace(1800, 350, 160) + np.random.randn(160) * 15,
        "Yield_10Y": np.linspace(3.5, 4.3, 160) + np.random.randn(160) * 0.05,
        "Yield_2Y": np.linspace(4.8, 3.9, 160) + np.random.randn(160) * 0.05,
    }, index=dates)

    df_macro["Net_Liquidity"] = df_macro["WALCL"] - df_macro["TGA"] - df_macro["RRP"]
    df_macro["Net_Liq_YoY"] = df_macro["Net_Liquidity"].pct_change(52) * 100
    df_macro["Spread_10Y_2Y"] = df_macro["Yield_10Y"] - df_macro["Yield_2Y"]
    latest_m = df_macro.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fed Net Liquidity", f"${latest_m['Net_Liquidity']:.1f} B", delta=f"{latest_m['Net_Liq_YoY']:.2f}% YoY")
    c2.metric("Spread 10Y - 2Y", f"{latest_m['Spread_10Y_2Y']:.2f}%", delta="Disinversione" if latest_m['Spread_10Y_2Y'] > 0 else "Inversione")
    c3.metric("Total Fed Assets", f"${latest_m['WALCL']:.1f} B")
    c4.metric("Reverse Repo (RRP)", f"${latest_m['RRP']:.1f} B")

    st.markdown("---")
    col_g1, col_g2 = st.columns([3, 2])
    with col_g1:
        fig_liq = make_subplots(specs=[[{"secondary_y": True}]])
        fig_liq.add_trace(go.Scatter(x=df_macro.index, y=df_macro["Net_Liquidity"], name="Fed Net Liquidity ($B)", line=dict(color="#00FFAA", width=2.5)), secondary_y=False)
        fig_liq.add_trace(go.Bar(x=df_macro.index, y=df_macro["Net_Liq_YoY"], name="Variazione YoY (%)", opacity=0.3, marker_color="#00BFFF"), secondary_y=True)
        fig_liq.update_layout(title="Fed Net Liquidity (WALCL - TGA - RRP) & Trend YoY", template="plotly_dark", height=400)
        st.plotly_chart(fig_liq, use_container_width=True)

    with col_g2:
        fig_yc = go.Figure()
        fig_yc.add_trace(go.Scatter(x=df_macro.index, y=df_macro["Spread_10Y_2Y"], name="Spread 10Y-2Y", line=dict(color="#FFA500", width=2)))
        fig_yc.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Inversione")
        fig_yc.update_layout(title="Storico Spread 10Y-2Y", template="plotly_dark", height=400)
        st.plotly_chart(fig_yc, use_container_width=True)

# --- VISTA 2: CONVERGENZA COT + POC ---
elif "2. Studio Convergenza" in protocol_choice:
    st.title("🎯 Protocollo Multi-Fattore: COT Z-Score + Volume Profile POC")
    st.caption("Confluenza quantitativa tra posizionamento istituzionale (Futures CFTC) e struttura volumetrica del prezzo.")

    ticker_sel = st.sidebar.selectbox("Seleziona Sottostante", ["TLT", "USO", "GLD", "SPY", "QQQ", "UNG", "AAPL"])
    cot_map = {"TLT": "US_30Y_BOND", "USO": "CRUDE_OIL", "GLD": "GOLD", "SPY": "SP500", "UNG": "NATURAL_GAS"}
    cot_key = cot_map.get(ticker_sel, "GENERAL")

    df_price = fetch_market_eod(ticker_sel)
    df_cot = fetch_cot_data(cot_key)

    protocol = COTPOCConvergenceProtocol()
    res = protocol.evaluate(df_price, df_cot)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Prezzo Ultimo", f"${res['close']}")
    m2.metric("Point of Control (POC)", f"${res['poc']}", delta=res['distance_to_poc'])
    m3.metric("Price Z-Score (252d)", f"{res['price_zscore']}σ", delta_color="inverse")
    m4.metric("RSI (14)", f"{res['rsi']}")
    m5.metric("COT Z-Score Comm", f"{res['cot_z_comm']}σ")

    if res["signal_type"] == "CONVERGENCE_LONG_ACCUMULATION":
        st.success(f"🎯 **SEGNALE ATTIVO:** Confluenza Rialzista (Smart Money in Accumulazione + Prezzo su Supporto POC)")
    elif res["signal_type"] == "CONVERGENCE_SHORT_DISTRIBUTION":
        st.error(f"🚨 **SEGNALE ATTIVO:** Confluenza Ribassista (Smart Money in Distribuzione + Resistenza POC)")
    else:
        st.info(f"⚖️ **STATO PROTOCOLLO:** Nessun disallineamento estremo (`NEUTRAL`)")

    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(
        x=df_price.index, open=df_price["Open"], high=df_price["High"], low=df_price["Low"], close=df_price["Close"],
        name="Prezzo EOD"
    ))
    fig_price.add_hline(y=res["poc"], line_dash="solid", line_color="#FFD700", line_width=2.5, annotation_text=f"POC: ${res['poc']}")
    fig_price.add_hline(y=res["vah"], line_dash="dot", line_color="#00FFAA", annotation_text=f"VAH: ${res['vah']}")
    fig_price.add_hline(y=res["val"], line_dash="dot", line_color="#FF3131", annotation_text=f"VAL: ${res['val']}")
    fig_price.update_layout(title=f"{ticker_sel} — Dinamica Prezzo con POC e Value Area", template="plotly_dark", height=450, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    cot_calc = COTPositioningStudy().calculate(df_cot)
    fig_cot = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4],
                            subplot_titles=("Posizioni Nette Commercials (Hedgers) vs Speculators", "Z-Score Posizionamento Istituzionale (52W)"))
    fig_cot.add_trace(go.Scatter(x=cot_calc.index, y=cot_calc["Comm_Net"], name="Commercials", line=dict(color="#00FFAA", width=2)), row=1, col=1)
    fig_cot.add_trace(go.Scatter(x=cot_calc.index, y=cot_calc["NonComm_Net"], name="Speculators", line=dict(color="#FF3131", width=2)), row=1, col=1)
    fig_cot.add_trace(go.Scatter(x=cot_calc.index, y=cot_calc["ZScore_Comm"], name="Z-Score Comm", line=dict(color="#00FFAA", width=1.5)), row=2, col=1)
    fig_cot.add_trace(go.Scatter(x=cot_calc.index, y=cot_calc["ZScore_NonComm"], name="Z-Score NonComm", line=dict(color="#FF3131", width=1.5)), row=2, col=1)
    fig_cot.add_hline(y=1.5, line_dash="dot", line_color="#FFA500", row=2, col=1)
    fig_cot.add_hline(y=-1.5, line_dash="dot", line_color="#FFA500", row=2, col=1)
    fig_cot.update_layout(template="plotly_dark", height=420, hovermode="x unified")
    st.plotly_chart(fig_cot, use_container_width=True)

# --- VISTA 3: SCREENER EOD & ALERT TELEGRAM ---
elif "3. Screener EOD" in protocol_choice:
    st.title("📡 Screener EOD Multi-Asset & Notifiche Telegram")
    st.caption("Scansione automatizzata del paniere titoli sui protocolli quantitativi registrati.")

    watchlist = ["TLT", "USO", "GLD", "SPY", "QQQ", "UNG", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA"]
    protocol = COTPOCConvergenceProtocol()

    screener_results = []
    with st.spinner("Scansione della Watchlist in corso..."):
        for t in watchlist:
            df_p = fetch_market_eod(t)
            df_c = fetch_cot_data(t)
            res = protocol.evaluate(df_p, df_c)
            screener_results.append({
                "Ticker": t,
                "Setup": res["signal_type"],
                "Prezzo ($)": res["close"],
                "POC ($)": res["poc"],
                "Dist POC": res["distance_to_poc"],
                "Z-Score": f"{res['price_zscore']}σ",
                "RSI": res["rsi"],
                "COT Z-Comm": f"{res['cot_z_comm']}σ",
                "Raw_Result": res
            })

    df_screen = pd.DataFrame(screener_results)
    st.dataframe(
        df_screen[["Ticker", "Setup", "Prezzo ($)", "POC ($)", "Dist POC", "Z-Score", "RSI", "COT Z-Comm"]],
        use_container_width=True,
        height=380
    )

    st.markdown("---")
    st.subheader("📬 Test Spedizione Alert Telegram")
    selected_alert_ticker = st.selectbox("Seleziona un titolo per testare l'invio:", options=watchlist)
    
    if st.button("📤 Invia Notifica su Telegram"):
        target_res = next(item["Raw_Result"] for item in screener_results if item["Ticker"] == selected_alert_ticker)
        metrics_payload = {
            "Setup": target_res["signal_type"],
            "Prezzo Chiusura": f"${target_res['close']}",
            "Point of Control (POC)": f"${target_res['poc']}",
            "Distanza dal POC": target_res["distance_to_poc"],
            "Price Z-Score (252d)": f"{target_res['price_zscore']}σ",
            "RSI (14)": target_res["rsi"],
            "COT Z-Score Commercials": f"{target_res['cot_z_comm']}σ",
            "COT Index Comm": f"{target_res['cot_idx_comm']}%"
        }
        
        if not tg_token or not tg_chat:
            st.warning("⚠️ Inserisci Bot Token e Chat ID nella sidebar per completare l'invio.")
        else:
            success = TelegramNotifier.send_alert(
                bot_token=tg_token,
                chat_id=tg_chat,
                ticker=selected_alert_ticker,
                protocol_name=protocol.name,
                metrics=metrics_payload
            )
            if success:
                st.success(f"✅ Alert per **{selected_alert_ticker}** inviato con successo!")
            else:
                st.error("❌ Errore durante l'invio. Verifica che il Bot sia admin del canale e i token siano corretti.")