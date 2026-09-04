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
  4. A Chart tab: candlesticks for today's session with composite,
     intraday, and validated zones overlaid (see candles_with_levels.py).

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
    pip install streamlit requests pandas numpy plotly --break-system-packages
    $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
    streamlit run app.py
"""
import os
import re
import json
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone, time as dtime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from hvn_lvn import build_volume_profile, find_hvn_lvn
from sahi_style_key_levels import sahi_style_key_levels
from zone_validation import cross_validated_zones, compute_zone_signal
from candles_with_levels import plot_candles_with_zones

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST)

# ---------------- Config ----------------
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
CACHE_PATH = "sahi_zones_cache.json"
ALERT_LOG_PATH = "alert_log.json"

DAILY_LOOKBACK_DAYS = 30         # needs enough history for RVOL_BASELINE_DAYS average
COMPOSITE_LOOKBACK_DAYS = 18     # matches hvn-lvn-scanner's multi-day window
RVOL_BASELINE_DAYS = 20          # prior-N-day average full-day volume, same convention as hvn-lvn-scanner
TOP_N_RVOL = 5                   # only symbols in the top N by RVOL are eligible to alert

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
AUTO_REFRESH_QUOTES_SECONDS = 60     # quotes + signal recompute cadence when auto-refresh is on
ZONE_REFRESH_EVERY_N_TICKS = 5       # also do a heavier zone refresh every Nth tick (~5 min)

MARKET_OPEN_TIME = dtime(9, 15)   # IST - no new alerts logged before this
MARKET_CLOSE_TIME = dtime(15, 30)  # IST - no new alerts logged at/after this

NEAR_ZONE_PCT = 0.3   # how close (%) LTP must be to a validated zone edge to count as "at" it

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


def fetch_intraday_candles(instrument_key, token, unit="minutes", interval="5"):
    """Upstox's historical-candle endpoint (fetch_candles above) NEVER
    includes the still-open trading day -- it only has data up through
    yesterday's final close. Today's still-forming candles require this
    separate intraday endpoint. Without this, "today's" fetch silently
    returns only yesterday's last candle, which looks like a frozen/stale
    chart rather than an obvious error."""
    url = f"https://api.upstox.com/v3/historical-candle/intraday/{instrument_key}/{unit}/{interval}"
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


def fetch_today_candles(instrument_key, token):
    """Today's session candles via the intraday endpoint. Falls back to
    the historical endpoint's last available day (e.g. before market
    open, when the intraday endpoint may return nothing yet) so the
    Chart/zone functions always get something to work with."""
    df = fetch_intraday_candles(instrument_key, token, "minutes", "5")
    if not df.empty:
        return df

    df = fetch_candles(instrument_key, token, "minutes", "5", lookback_days=1)
    if df.empty:
        return df
    latest = df["date"].max()
    return df[df["date"] == latest]


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
            avg_daily_volume = (float(daily_df["volume"].tail(RVOL_BASELINE_DAYS).mean())
                                 if len(daily_df) >= RVOL_BASELINE_DAYS else None)
            composite_zones = compute_composite_zones(intraday_df)
            intraday_zones = compute_intraday_zones(intraday_df)  # seed with today's slice of what we already have

            cache[symbol] = {
                "instrument_key": key,
                "prev_close": prev_close,
                "avg_daily_volume": avg_daily_volume,
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


def nearest_zones(ltp, validated_zones):
    """Splits validated zones into support-side (price_mode <= ltp) and
    resistance-side (price_mode > ltp), and returns whichever of each is
    CLOSEST to ltp, along with the % distance from ltp to that zone's
    near edge (price_high for support, price_low for resistance -- the
    edge price would actually touch first)."""
    support, support_dist = None, None
    resistance, resistance_dist = None, None
    for z in validated_zones:
        if ltp is None:
            break
        if z["price_mode"] <= ltp:
            dist = abs(ltp - z["price_high"]) / ltp * 100
            if support_dist is None or dist < support_dist:
                support, support_dist = z, dist
        else:
            dist = abs(z["price_low"] - ltp) / ltp * 100
            if resistance_dist is None or dist < resistance_dist:
                resistance, resistance_dist = z, dist
    return support, support_dist, resistance, resistance_dist


def build_setup_display_df(rows, zone_kind):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    label = "Support" if zone_kind == "support" else "Resistance"
    df = df[["symbol", "ltp", "vwap", "zone_level", "zone_pct", "distance_pct"]]
    df.columns = ["Symbol", "LTP", "VWAP", f"{label} level", "Zone %", "Distance %"]
    return df.sort_values("Distance %").reset_index(drop=True)


def run_live_scan(cache, token):
    """Fast tier: one batch quote call for LTP/VWAP, signal recomputed
    against whichever zones are currently cached (may be a few minutes
    stale if 'Refresh Zones' hasn't been run recently). Also computes RVOL
    (today's volume so far / prior-N-day average full-day volume) and
    ranks the top TOP_N_RVOL symbols -- only those are eligible to alert
    (see update_alert_log), since unusual volume is the conviction filter
    that keeps alerts to a handful of genuinely active names instead of
    every symbol that happens to tick across VWAP."""
    symbols = list(cache.keys())
    instrument_keys = [cache[s]["instrument_key"] for s in symbols]
    key_to_symbol = {cache[s]["instrument_key"]: s for s in symbols}
    quotes = fetch_batch_quotes(instrument_keys, token)

    rows = []
    signals = {}
    bottom_setups = []   # near support + just crossed above VWAP
    top_setups = []      # near resistance + just closed below VWAP

    for quote_key, q in quotes.items():
        instrument_key = q.get("instrument_token")
        symbol = key_to_symbol.get(instrument_key)
        if not symbol:
            continue
        c = cache[symbol]
        ltp = q.get("last_price")
        vwap = q.get("average_price")
        today_volume = q.get("volume")
        prev_close = c.get("prev_close")
        avg_daily_volume = c.get("avg_daily_volume")

        change_pct = (round((ltp - prev_close) / prev_close * 100, 2)
                      if ltp is not None and prev_close else None)
        rvol_pct = (round(today_volume / avg_daily_volume * 100, 1)
                    if today_volume is not None and avg_daily_volume else None)
        signal = compute_zone_signal(
            ltp, vwap, c.get("composite_zones", []), c.get("intraday_zones", []),
            min_distance_pct=MIN_SIGNAL_DISTANCE_PCT,
            min_vwap_distance_pct=MIN_VWAP_DISTANCE_PCT,
        )
        signals[symbol] = {"signal": signal, "ltp": ltp, "vwap": vwap, "rvol_pct": rvol_pct}
        cache[symbol]["last_signal"] = signal

        # --- Setup detection: near support/resistance + VWAP cross ---
        # "Just crossed" is edge-triggered off the PREVIOUS scan's
        # above/below state, persisted in the cache (same pattern as
        # update_alert_log's edge-triggering below) so it survives
        # across reruns instead of re-firing on every refresh.
        if ltp is not None and vwap is not None:
            val_comp, _, _ = cross_validated_zones(
                c.get("composite_zones", []), c.get("intraday_zones", [])
            )
            support, support_dist, resistance, resistance_dist = nearest_zones(ltp, val_comp)

            vwap_above_now = ltp > vwap
            prev_vwap_above = c.get("prev_vwap_above")
            crossed_up = prev_vwap_above is False and vwap_above_now
            crossed_down = prev_vwap_above is True and not vwap_above_now
            cache[symbol]["prev_vwap_above"] = vwap_above_now

            if support is not None and support_dist is not None and support_dist <= NEAR_ZONE_PCT and crossed_up:
                bottom_setups.append({
                    "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                    "zone_level": support["price_mode"],
                    "zone_pct": _pct_from_label_safe(support["label"]),
                    "distance_pct": round(support_dist, 2),
                })

            if resistance is not None and resistance_dist is not None and resistance_dist <= NEAR_ZONE_PCT and crossed_down:
                top_setups.append({
                    "symbol": symbol, "ltp": ltp, "vwap": round(vwap, 2),
                    "zone_level": resistance["price_mode"],
                    "zone_pct": _pct_from_label_safe(resistance["label"]),
                    "distance_pct": round(resistance_dist, 2),
                })

        rows.append({
            "Symbol": symbol, "PrevClose": prev_close, "LTP": ltp,
            "Change%": change_pct, "VWAP": vwap, "RVOL%": rvol_pct, "Signal": signal,
        })

    ranked = sorted(
        [(s, d["rvol_pct"]) for s, d in signals.items() if d["rvol_pct"] is not None],
        key=lambda x: x[1], reverse=True,
    )
    top_n_symbols = set(s for s, _ in ranked[:TOP_N_RVOL])

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Top5RVOL"] = df["Symbol"].isin(top_n_symbols)
        df = df.sort_values("RVOL%", ascending=False, na_position="last").reset_index(drop=True)
        df.insert(0, "S.No", range(1, len(df) + 1))
    return df, signals, top_n_symbols, bottom_setups, top_setups


def _pct_from_label_safe(label):
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        with open(ALERT_LOG_PATH, "r") as f:
            return json.load(f)
    return {"alerts": [], "last_eligible_symbols": []}


def save_alert_log(log):
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def update_alert_log(alert_log, signals, eligible_symbols):
    """Edge-triggered: only logs a new entry the moment a symbol's signal
    changes to a fresh BUY/SELL state, not on every refresh it stays
    active. Gated to market hours -- an after-hours refresh pulls Upstox's
    frozen post-close LTP/VWAP, which must not get logged as a live
    signal (same bug already fixed in hvn-lvn-scanner's paper trader).

    Also gated to eligible_symbols (the current top TOP_N_RVOL by RVOL).
    "Newly entered" (just entered the top N this cycle) is tracked via
    alert_log["last_eligible_symbols"], persisted to disk -- NOT via
    st.session_state. Session state is per-browser-session, so a page
    reload or a Streamlit Cloud reconnect resets it to empty, making
    every currently-eligible symbol look "newly entered" again and
    re-logging duplicates seconds after the original (this exact bug
    was seen live: 5 symbols logged twice, 4 seconds apart). Persisting
    to the same file the dedup check already reads from survives
    reconnects correctly."""
    now = now_ist()
    prev_eligible = set(alert_log.get("last_eligible_symbols", []))
    newly_entered = eligible_symbols - prev_eligible
    alert_log["last_eligible_symbols"] = list(eligible_symbols)

    market_is_open = MARKET_OPEN_TIME <= now.time() < MARKET_CLOSE_TIME
    if not market_is_open:
        return alert_log

    last_signal = {a["symbol"]: a["signal"] for a in alert_log["alerts"] if a.get("is_latest")}
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    for symbol, data in signals.items():
        if symbol not in eligible_symbols:
            continue
        sig = data["signal"]
        if sig not in ("BUY", "SELL"):
            continue
        is_fresh_entry = symbol in newly_entered
        if not is_fresh_entry and last_signal.get(symbol) == sig:
            continue
        for a in alert_log["alerts"]:
            if a["symbol"] == symbol:
                a["is_latest"] = False
        alert_log["alerts"].append({
            "symbol": symbol, "signal": sig, "ltp": data["ltp"], "vwap": data["vwap"],
            "rvol_pct": data.get("rvol_pct"), "time": now_str, "is_latest": True,
        })
    return alert_log


def build_alert_display_df(alert_log, price_lookup=None):
    """price_lookup: {symbol: current LTP}, from the latest Scanner scan --
    used to mark-to-market each alert's PnL% against its entry price
    (the LTP captured at the moment the alert fired)."""
    if not alert_log["alerts"]:
        return pd.DataFrame()
    price_lookup = price_lookup or {}
    df = pd.DataFrame(alert_log["alerts"])
    if "rvol_pct" not in df.columns:
        df["rvol_pct"] = None

    def _current_price(row):
        return price_lookup.get(row["symbol"])

    def _pnl(row):
        cur = row["current_ltp"]
        entry = row["ltp"]
        if cur is None or entry is None:
            return None
        if row["signal"] == "BUY":
            return round((cur - entry) / entry * 100, 2)
        elif row["signal"] == "SELL":
            return round((entry - cur) / entry * 100, 2)
        return None

    df["current_ltp"] = df.apply(_current_price, axis=1)
    df["pnl_pct"] = df.apply(_pnl, axis=1)
    df = df[["symbol", "signal", "ltp", "current_ltp", "pnl_pct", "vwap", "rvol_pct", "time"]]
    df.columns = ["Symbol", "Signal", "EntryPrice", "LTP", "PnL%", "VWAP", "RVOL%", "Time"]
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


# ---------------- UI (four tabs: Scanner, Key Levels, Chart, Alerts) ----------------
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

auto_refresh_enabled = st.checkbox(
    "Auto-refresh (quotes every 1 min, zones every 5 min) - only while this tab stays open",
    value=False,
)
auto_tick = st_autorefresh(interval=AUTO_REFRESH_QUOTES_SECONDS * 1000, key="auto_refresh_tick") if auto_refresh_enabled else None
if "last_auto_tick" not in st.session_state:
    st.session_state["last_auto_tick"] = -1
auto_quotes_due = auto_tick is not None and auto_tick != st.session_state["last_auto_tick"]
if auto_quotes_due:
    st.session_state["last_auto_tick"] = auto_tick
auto_zone_due = auto_quotes_due and auto_tick > 0 and auto_tick % ZONE_REFRESH_EVERY_N_TICKS == 0

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

    if refresh_zones_clicked or auto_zone_due:
        token = get_token()
        progress_bar = st.progress(0)
        status_text = st.empty()
        def _cb2(i, total, symbol):
            progress_bar.progress(i / total)
            status_text.text(f"{i}/{total}: {symbol}")
        with st.spinner("Refreshing intraday zones..."):
            cache = run_zone_refresh(cache, token, progress_callback=_cb2)
        st.success("Zone refresh done.")

    if refresh_quotes_clicked or refresh_zones_clicked or auto_quotes_due or auto_zone_due or "last_scan_df" not in st.session_state:
        token = get_token()
        scan_df, signals, top_n_symbols, bottom_setups, top_setups = run_live_scan(cache, token)

        alert_log = load_alert_log()
        alert_log = update_alert_log(alert_log, signals, top_n_symbols)
        save_alert_log(alert_log)

        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f)

        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_refresh_time"] = now_ist().strftime("%H:%M:%S")
        st.session_state["alert_log"] = alert_log
        st.session_state["bottom_setups"] = bottom_setups
        st.session_state["top_setups"] = top_setups

    df = st.session_state.get("last_scan_df", pd.DataFrame())
    price_lookup = dict(zip(df["Symbol"], df["LTP"])) if not df.empty else {}
    alert_log = st.session_state.get("alert_log") or load_alert_log()

    symbols_with_zones = [s for s in cache if cache[s].get("composite_zones") or cache[s].get("intraday_zones")]
    default_idx = symbols_with_zones.index("NIFTY") if "NIFTY" in symbols_with_zones else 0

    tab_scanner, tab_levels, tab_chart, tab_setups, tab_alerts = st.tabs(
        ["Scanner", "Key Levels", "Chart", "Setups", "Alerts"]
    )

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
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            selected_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="levels_symbol")
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

    with tab_chart:
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            chart_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="chart_symbol")
            c = cache[chart_symbol]
            token = get_token()

            chart_df = fetch_today_candles(c["instrument_key"], token)

            if chart_df.empty:
                st.write("No candle data yet for today.")
            else:
                val_comp, val_intra, _ = cross_validated_zones(
                    c.get("composite_zones", []), c.get("intraday_zones", [])
                )
                fig = plot_candles_with_zones(
                    chart_df,
                    composite_zones=c.get("composite_zones", []),
                    intraday_zones=c.get("intraday_zones", []),
                    validated_zones=val_comp,
                    title=f"{chart_symbol} - price with key levels",
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_setups:
        st.caption(
            f"Edge-triggered: a symbol appears only the cycle it happens, not every refresh "
            f"it stays true. 'Near' means within {NEAR_ZONE_PCT}% of the validated zone's edge."
        )
        st.markdown("**At support, just crossed above VWAP** (possible bounce)")
        bottom_df = build_setup_display_df(st.session_state.get("bottom_setups", []), "support")
        if bottom_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(bottom_df, use_container_width=True, hide_index=True)

        st.markdown("**At resistance, just closed below VWAP** (possible rejection)")
        top_df = build_setup_display_df(st.session_state.get("top_setups", []), "resistance")
        if top_df.empty:
            st.write("None this cycle.")
        else:
            st.dataframe(top_df, use_container_width=True, hide_index=True)

    with tab_alerts:
        st.caption(
            f"Logged the moment a symbol's signal changes to a fresh BUY/SELL - not repeated "
            f"every refresh it stays active. Only the top {TOP_N_RVOL} symbols by RVOL are eligible "
            f"to alert. Only logged during market hours (9:15-15:30 IST)."
        )
        alert_df = build_alert_display_df(alert_log, price_lookup)
        if alert_df.empty:
            st.write("No alerts logged yet.")
        else:
            st.dataframe(alert_df, use_container_width=True, hide_index=True)
            csv = alert_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download alert log CSV", csv, "alert_log.csv", "text/csv")
else:
    st.info("No cache found yet - click 'Run Precompute' first.")
