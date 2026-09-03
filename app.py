"""
app.py - Sahi Key Levels LIVE (cross-timeframe validated zones + alerts)

Builds on the sahi-key-levels module (vendored below, UNCHANGED) to add:
  1. Two independently-computed zone sets per symbol: a COMPOSITE profile
     (18 trading days) and an INTRADAY profile (today's session only).
  2. Cross-timeframe validation (see zone_validation.py): a zone only
     counts as a real level if its price range shows up in BOTH profiles.
     A zone with no multi-day backing, or a composite zone today's session
     hasn't touched at all, is treated as noise and dropped.
  3. BUY/SELL alerts, edge-triggered (only logged the moment a symbol's
     signal changes, not every refresh) and gated to market hours (same
     fix already applied in hvn-lvn-scanner: an after-hours refresh pulls
     Upstox's frozen post-close quotes, which must not get logged as a
     live signal).

VENDORED FILES (copied unchanged from their source repos, per instruction
to leave the original logic untouched):
    hvn_lvn.py               <- from hvn-lvn-scanner
    sahi_style_key_levels.py <- from sahi-key-levels

THREE-TIER REFRESH MODEL (deliberate, not accidental complexity):
  - "Run Precompute" (slow, once/day): resolves instrument keys, fetches
    18 days of 5-min candles per symbol, computes the COMPOSITE zone set.
    This is the expensive step -- same reasoning as hvn-lvn-scanner's
    Precompute.
  - "Refresh Zones" (medium, every few minutes -- NOT on every quote tick):
    re-fetches ONLY today's 5-min candles per symbol (a much lighter
    historical-candle call than the 18-day Precompute fetch, but still one
    HTTP call per symbol, so this is not free -- don't wire it to run on
    every quote refresh across 200+ symbols). Recomputes the INTRADAY zone
    set, cross-validates against the cached COMPOSITE set, and logs any
    new BUY/SELL alerts.
  - "Refresh Quotes" (fast): single batch quote call for LTP/VWAP, same as
    hvn-lvn-scanner's existing fast refresh. Recomputes each symbol's
    signal against whatever zones were last computed by "Refresh Zones"
    (may be a few minutes stale) -- this keeps the Scanner table feeling
    responsive without re-fetching candles on every tick.

SETUP:
    pip install streamlit requests pandas numpy --break-system-packages
    $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
    streamlit run app.py
"""
import os
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time as dtime

import numpy as np
import pandas as pd
import requests
import streamlit as st

from hvn_lvn import build_volume_profile, find_hvn_lvn
from sahi_style_key_levels import sahi_style_key_levels
from zone_validation import cross_validated_zones, compute_zone_signal

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST)

# ---------------- Config ----------------
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
CACHE_PATH = "sahi_zones_cache.json"
ALERT_LOG_PATH = "alert_log.json"

DAILY_LOOKBACK_DAYS = 5          # just enough for PrevClose / Change%
COMPOSITE_LOOKBACK_DAYS = 18     # matches hvn-lvn-scanner's multi-day window

COMPOSITE_N_BINS = 50
INTRADAY_N_BINS = 45
MIN_PROMINENCE_PCT = 0.08
MIN_BIN_DISTANCE = 2
MAX_ZONES = 6
MIN_DISPLAY_PCT = 2.0
MIN_SIGNAL_DISTANCE_PCT = 0.5    # how far LTP must be from a validated zone to signal
MIN_VWAP_DISTANCE_PCT = 0.15     # how far LTP must be from VWAP before a bias counts as real
                                  # (found necessary live: without this, tiny VWAP wobbles of
                                  # 0.02-0.05% fired repeated BUY/SELL flips on the same symbol)

MARKET_OPEN_TIME = dtime(9, 15)   # IST - no new alerts logged before this
MARKET_CLOSE_TIME = dtime(15, 30)  # IST - no new alerts logged at/after this

# Same universe as hvn-lvn-scanner. NIFTY/BANKNIFTY handled separately
# (futures, not equity).
EQUITY_SYMBOLS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "AXISBANK",
    "KOTAKBANK", "BAJFINANCE", "BHARTIARTL", "ITC", "LT", "HINDUNILVR",
    "MARUTI", "TMPV", "TATASTEEL", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "ASIANPAINT", "WIPRO", "NTPC", "POWERGRID", "M&M", "ADANIENT",
    "ADANIPORTS", "BAJAJFINSV", "HCLTECH", "JSWSTEEL", "ONGC", "COALINDIA",
    "TECHM", "GRASIM", "DIVISLAB", "DRREDDY", "CIPLA", "EICHERMOT",
    "HEROMOTOCO", "HINDALCO", "BPCL", "BRITANNIA", "APOLLOHOSP", "SBILIFE",
    "HDFCLIFE", "INDUSINDBK", "BAJAJ-AUTO", "TATACONSUM", "UPL", "SHREECEM",
    "NESTLEIND", "VEDANTA", "GAIL", "PIDILITIND", "DLF", "GODREJCP",
    "SIEMENS", "AMBUJACEM", "BANDHANBNK", "BANKBARODA", "PNB", "CANBK",
    "IDFCFIRSTB", "FEDERALBNK", "AUROPHARMA", "BEL", "BIOCON", "CHOLAFIN",
    "COLPAL", "CONCOR", "CUMMINSIND", "DABUR", "DEEPAKNTR", "ESCORTS",
    "EXIDEIND", "GODREJPROP", "HAVELLS", "HDFCAMC", "ICICIGI", "ICICIPRULI",
    "IEX", "INDIGO", "INDUSTOWER", "IOC", "IRCTC", "JINDALSTEL", "JUBLFOOD",
    "LICHSGFIN", "LTIM", "LUPIN", "MANAPPURAM", "MARICO", "MCDOWELL-N",
    "MFSL", "MOTHERSON", "MPHASIS", "MRF", "MUTHOOTFIN", "NAUKRI",
    "NMDC", "OBEROIRLTY", "OFSS", "PAGEIND", "PEL", "PERSISTENT",
    "PETRONET", "PFC", "PIIND", "POLYCAB", "RECLTD", "SAIL", "SBICARD",
    "SRF", "SYNGENE", "TATACOMM", "TATAPOWER", "TORNTPHARM", "TRENT",
    "TVSMOTOR", "UBL", "VOLTAS", "ZEEL", "ZYDUSLIFE", "CDSL", "IRFC",
    "IDEA", "YESBANK", "SUZLON", "ZOMATO", "DMART", "JIOFIN", "PAYTM",
    "NYKAA", "POLICYBZR", "DELHIVERY", "LODHA", "PATANJALI", "ABCAPITAL",
    "ALKEM", "APLAPOLLO", "ASHOKLEY", "ASTRAL", "ATUL", "BALKRISNIND",
    "BATAINDIA", "BHARATFORG", "BHEL", "BSOFT", "CANFINHOME", "CROMPTON",
    "CUB", "DALBHARAT", "GLENMARK", "GMRINFRA", "GNFC", "GRANULES",
    "GUJGASLTD", "HAL", "HINDCOPPER", "HINDPETRO", "IBULHSGFIN", "IGL",
    "INDHOTEL", "INDIAMART", "IPCALAB", "JKCEMENT", "L&TFH", "LALPATHLAB",
    "LAURUSLABS", "M&MFIN", "METROPOLIS", "NATIONALUM", "NAVINFLUOR",
    "OIL", "PVRINOX", "RAIN", "RBLBANK", "SUNTV", "TATACHEM",
    "TATAELXSI", "TORNTPOWER", "UNIONBANK", "VBL", "WHIRLPOOL",
    "AARTIIND", "ABFRL", "ANGELONE", "APOLLOTYRE", "AUBANK", "BANKINDIA",
    "BSE", "CGPOWER", "CHAMBLFERT", "COFORGE", "COROMANDEL", "DIXON",
    "FORTIS", "GICRE", "GODFRYPHLP", "GRAPHITE", "GSPL", "HFCL",
    "HUDCO", "IIFL", "INDIACEM", "IRB", "ITI", "KALYANKJIL",
    "KEI", "LTF", "MANKIND", "MAXHEALTH", "MGL", "MOTILALOFS",
    "NBCC", "NCC", "NHPC", "PFIZER", "PGEL", "POWERINDIA",
    "PRESTIGE", "RVNL", "SJVN", "SOLARINDS", "SONACOMS", "STARHEALTH",
    "SUPREMEIND", "TIINDIA", "TITAGARH", "VEDL", "ZFCVINDIA",
]
FUTURES_SYMBOLS = ["NIFTY", "BANKNIFTY"]


def get_token():
    token = os.environ.get("UPSTOX_ACCESS_TOKEN")
    if token:
        return token.strip()
    try:
        if "UPSTOX_ACCESS_TOKEN" in st.secrets:
            return st.secrets["UPSTOX_ACCESS_TOKEN"].strip()
    except Exception:
        pass
    if os.path.exists("upstox_token.txt"):
        with open("upstox_token.txt", "r") as f:
            t = f.read().strip()
        if t and t != "PASTE_YOUR_TOKEN_HERE":
            return t
    raise RuntimeError(
        "No token found. Set $env:UPSTOX_ACCESS_TOKEN, add UPSTOX_ACCESS_TOKEN to Streamlit "
        "secrets, or create upstox_token.txt."
    )


def resolve_equity_instrument_key(symbol, token):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": symbol, "exchanges": "NSE", "segments": "EQ",
              "instrument_types": "EQ", "page_number": 1, "records": 10}
    resp = requests.get(INSTRUMENT_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("trading_symbol", "").upper() == symbol.upper()]
    return candidates[0]["instrument_key"] if candidates else None


def resolve_futures_instrument_key(name, token):
    """No expiry filter - sorts client-side by expiry (same fix already
    applied in hvn-lvn-scanner: 'current_month' keyword returns zero
    results once that month's contract expires but the calendar hasn't
    rolled over yet)."""
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"query": name, "exchanges": "NSE", "segments": "FO",
              "instrument_types": "FUT", "page_number": 1, "records": 30}
    resp = requests.get(INSTRUMENT_SEARCH_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    candidates = [inst for inst in resp.json().get("data", [])
                  if inst.get("instrument_type") == "FUT"
                  and inst.get("underlying_symbol", "").upper() == name.upper()]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x["expiry"])
    return candidates[0]["instrument_key"]


def fetch_candles(instrument_key, token, unit, interval, lookback_days):
    to_date = now_ist().strftime("%Y-%m-%d")
    from_date = (now_ist() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    candles = resp.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def compute_composite_zones(intraday_df):
    """Composite zone set from the FULL multi-day intraday_df (no date
    filtering -- composite means across all fetched days)."""
    if intraday_df.empty:
        return []
    try:
        _, shown = sahi_style_key_levels(
            intraday_df, n_bins=COMPOSITE_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def compute_intraday_zones(today_only_df):
    """Intraday zone set from a candle df already scoped to a single
    session (see fetch_today_candles below)."""
    if today_only_df.empty:
        return []
    today = today_only_df["date"].max()
    today_df = today_only_df[today_only_df["date"] == today]
    try:
        _, shown = sahi_style_key_levels(
            today_df, n_bins=INTRADAY_N_BINS, max_zones=MAX_ZONES,
            min_display_pct=MIN_DISPLAY_PCT, min_prominence_pct=MIN_PROMINENCE_PCT,
            min_bin_distance=MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown]
    except Exception:
        return []


def fetch_today_candles(instrument_key, token):
    """Lighter than the Precompute fetch: only spans yesterday->today, so
    each call returns at most ~150 5-min bars instead of 18 days' worth.
    Still one HTTP call per symbol -- see the "Refresh Zones" docstring
    note above about not wiring this to the fast quote-refresh loop."""
    df = fetch_candles(instrument_key, token, "minutes", "5", lookback_days=1)
    if df.empty:
        return df
    today = df["date"].max()
    return df[df["date"] == today]


def run_precompute(token, progress_callback=None):
    cache = {}
    all_symbols = [(s, "equity") for s in EQUITY_SYMBOLS] + [(s, "futures") for s in FUTURES_SYMBOLS]
    for i, (symbol, kind) in enumerate(all_symbols):
        try:
            key = (resolve_equity_instrument_key(symbol, token) if kind == "equity"
                   else resolve_futures_instrument_key(symbol, token))
            if key is None:
                continue
            daily_df = fetch_candles(key, token, "days", "1", DAILY_LOOKBACK_DAYS)
            intraday_df = fetch_candles(key, token, "minutes", "5", COMPOSITE_LOOKBACK_DAYS)

            prev_close = float(daily_df["close"].iloc[-1]) if not daily_df.empty else None
            composite_zones = compute_composite_zones(intraday_df)
            intraday_zones = compute_intraday_zones(intraday_df)  # seed with today's slice of what we already have

            cache[symbol] = {
                "instrument_key": key,
                "prev_close": prev_close,
                "composite_zones": composite_zones,
                "intraday_zones": intraday_zones,
                "last_signal": "-",
                "zones_updated_at": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            st.warning(f"{symbol}: precompute failed ({e}), skipping.")
        if progress_callback:
            progress_callback(i + 1, len(all_symbols), symbol)
        time.sleep(0.15)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def run_zone_refresh(cache, token, progress_callback=None):
    """The 'medium' refresh tier: re-fetches TODAY's candles per symbol
    and recomputes intraday_zones. Composite zones are left untouched
    (those only change at the next Precompute)."""
    symbols = list(cache.keys())
    for i, symbol in enumerate(symbols):
        try:
            key = cache[symbol]["instrument_key"]
            today_df = fetch_today_candles(key, token)
            cache[symbol]["intraday_zones"] = compute_intraday_zones(today_df)
            cache[symbol]["zones_updated_at"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.warning(f"{symbol}: zone refresh failed ({e}), keeping previous zones.")
        if progress_callback:
            progress_callback(i + 1, len(symbols), symbol)
        time.sleep(0.1)

    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f)
    return cache


def fetch_batch_quotes(instrument_keys, token):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    params = {"instrument_key": ",".join(instrument_keys)}
    resp = requests.get(QUOTES_URL, headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json().get("data", {})


def run_live_scan(cache, token):
    """Fast tier: one batch quote call for LTP/VWAP, signal recomputed
    against whichever zones are currently cached (may be a few minutes
    stale if 'Refresh Zones' hasn't been run recently)."""
    symbols = list(cache.keys())
    instrument_keys = [cache[s]["instrument_key"] for s in symbols]
    key_to_symbol = {cache[s]["instrument_key"]: s for s in symbols}
    quotes = fetch_batch_quotes(instrument_keys, token)

    rows = []
    signals = {}
    for quote_key, q in quotes.items():
        instrument_key = q.get("instrument_token")
        symbol = key_to_symbol.get(instrument_key)
        if not symbol:
            continue
        c = cache[symbol]
        ltp = q.get("last_price")
        vwap = q.get("average_price")
        prev_close = c.get("prev_close")

        change_pct = (round((ltp - prev_close) / prev_close * 100, 2)
                      if ltp is not None and prev_close else None)
        signal = compute_zone_signal(
            ltp, vwap, c.get("composite_zones", []), c.get("intraday_zones", []),
            min_distance_pct=MIN_SIGNAL_DISTANCE_PCT,
            min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
        )
        signals[symbol] = {"signal": signal, "ltp": ltp, "vwap": vwap}
        cache[symbol]["last_signal"] = signal

        rows.append({
            "Symbol": symbol, "PrevClose": prev_close, "LTP": ltp,
            "Change%": change_pct, "VWAP": vwap, "Signal": signal,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Symbol").reset_index(drop=True)
        df.insert(0, "S.No", range(1, len(df) + 1))
    return df, signals


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "r") as f:
            return json.load(f)
    return {"alerts": []}


def save_alert_log(log):
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def update_alert_log(alert_log, signals):
    """Edge-triggered: only logs a new entry the moment a symbol's signal
    changes to a fresh BUY/SELL state, not on every refresh it stays
    active. Gated to market hours -- an after-hours refresh pulls Upstox's
    frozen post-close LTP/VWAP, which must not get logged as a live
    signal (same bug already fixed in hvn-lvn-scanner's paper trader)."""
    now = now_ist()
    market_is_open = MARKET_OPEN_TIME <= now.time() < MARKET_CLOSE_TIME
    if not market_is_open:
        return alert_log

    last_signal = {a["symbol"]: a["signal"] for a in alert_log["alerts"] if a.get("is_latest")}
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    for symbol, data in signals.items():
        sig = data["signal"]
        if sig not in ("BUY", "SELL"):
            continue
        if last_signal.get(symbol) == sig:
            continue
        for a in alert_log["alerts"]:
            if a["symbol"] == symbol:
                a["is_latest"] = False
        alert_log["alerts"].append({
            "symbol": symbol, "signal": sig, "ltp": data["ltp"], "vwap": data["vwap"],
            "time": now_str, "is_latest": True,
        })
    return alert_log


def build_alert_display_df(alert_log):
    if not alert_log["alerts"]:
        return pd.DataFrame()
    df = pd.DataFrame(alert_log["alerts"])
    df = df[["symbol", "signal", "ltp", "vwap", "time"]]
    df.columns = ["Symbol", "Signal", "LTP", "VWAP", "Time"]
    df = df.sort_values("Time", ascending=False).reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df) + 1))
    return df


def build_zones_display_df(zones):
    if not zones:
        return pd.DataFrame()
    df = pd.DataFrame(zones)
    df = df[["price_mode", "label", "price_low", "price_high"]]
    df.columns = ["Level", "Zone %", "Range Low", "Range High"]
    return df.sort_values("Level", ascending=False).reset_index(drop=True)


# ---------------- UI (three tabs: Scanner, Key Levels, Alerts) ----------------
st.set_page_config(page_title="Sahi Key Levels LIVE", layout="wide")
st.title("Sahi Key Levels LIVE")
st.caption(
    "Cross-timeframe validated zones: a level only counts if BOTH the 18-day composite "
    "profile and today's intraday profile independently show volume clustered there."
)

col1, col2, col3 = st.columns(3)
run_precompute_clicked = col1.button("Run Precompute (slow, once/day)")
refresh_zones_clicked = col2.button("Refresh Zones (medium, every few min)")
refresh_quotes_clicked = col3.button("Refresh Quotes (fast)")

if run_precompute_clicked:
    token = get_token()
    progress_bar = st.progress(0)
    status_text = st.empty()
    def _cb(i, total, symbol):
        progress_bar.progress(i / total)
        status_text.text(f"{i}/{total}: {symbol}")
    with st.spinner("Running precompute..."):
        cache = run_precompute(token, progress_callback=_cb)
    st.success(f"Precompute done. {len(cache)} symbols cached.")

if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r") as f:
        cache = json.load(f)

    if refresh_zones_clicked:
        token = get_token()
        progress_bar = st.progress(0)
        status_text = st.empty()
        def _cb2(i, total, symbol):
            progress_bar.progress(i / total)
            status_text.text(f"{i}/{total}: {symbol}")
        with st.spinner("Refreshing intraday zones..."):
            cache = run_zone_refresh(cache, token, progress_callback=_cb2)
        st.success("Zone refresh done.")

    if refresh_quotes_clicked or refresh_zones_clicked or "last_scan_df" not in st.session_state:
        token = get_token()
        scan_df, signals = run_live_scan(cache, token)

        alert_log = load_alert_log()
        alert_log = update_alert_log(alert_log, signals)
        save_alert_log(alert_log)

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_refresh_time"] = now_ist().strftime("%H:%M:%S")
        st.session_state["alert_log"] = alert_log

    df = st.session_state.get("last_scan_df", pd.DataFrame())
    alert_log = st.session_state.get("alert_log") or load_alert_log()

    tab_scanner, tab_levels, tab_alerts = st.tabs(["Scanner", "Key Levels", "Alerts"])

    with tab_scanner:
        st.caption(f"Last refreshed: {st.session_state.get('last_refresh_time', 'never')}")
        if df.empty:
            st.write("No data yet - click Refresh Quotes.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_levels:
        st.caption(
            "Composite = 18-day profile (updates on Precompute). Intraday = today's session "
            "(updates on Refresh Zones). Only overlapping ranges across both count as validated."
        )
        symbols_with_zones = [s for s in cache if cache[s].get("composite_zones") or cache[s].get("intraday_zones")]
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            default_idx = symbols_with_zones.index("NIFTY") if "NIFTY" in symbols_with_zones else 0
            selected_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx)
            c = cache[selected_symbol]
            st.caption(f"Zones last updated: {c.get('zones_updated_at', 'never')}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Composite (18-day)**")
                st.dataframe(build_zones_display_df(c.get("composite_zones", [])),
                             use_container_width=True, hide_index=True)
            with col_b:
                st.markdown("**Intraday (today)**")
                st.dataframe(build_zones_display_df(c.get("intraday_zones", [])),
                             use_container_width=True, hide_index=True)

            val_comp, val_intra, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            st.markdown("**Validated (confirmed by both timeframes)**")
            validated_display = build_zones_display_df(val_comp)
            if validated_display.empty:
                st.write("No cross-validated zones yet.")
            else:
                st.dataframe(validated_display, use_container_width=True, hide_index=True)

    with tab_alerts:
        st.caption(
            "Logged the moment a symbol's signal changes to a fresh BUY/SELL - not repeated "
            "every refresh it stays active. Only logged during market hours (9:15-15:30 IST)."
        )
        alert_df = build_alert_display_df(alert_log)
        if alert_df.empty:
            st.write("No alerts logged yet.")
        else:
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
            csv = alert_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download alert log CSV", csv, "alert_log.csv", "text/csv")
else:
    st.info("No cache found yet - click 'Run Precompute' first.")
