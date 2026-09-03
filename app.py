"""
app.py - HVN/LVN Scanner (minimal, single-page version)

ONE table for live scanning, one for paper trades. Scanner columns:
    S.No | Symbol | PrevClose | LTP | Change% | VWAP | RVOL% | Signal | SignalTime

SETUP:
    pip install streamlit requests pandas numpy --break-system-packages
    $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
    streamlit run app.py

Two-phase architecture (same reasoning as the original scanner, kept because
it's genuinely necessary, not just inherited complexity):
  - "Run Precompute" button: slow, once-per-day step. Resolves each
    symbol's instrument key, fetches daily candles (for PrevClose + RVOL
    baseline) and 5-min candles (for HVN/LVN + ATR), caches to hvn_cache.json.
  - "Refresh" button: fast, single batch quote call for LTP/VWAP/today's
    volume - this is the only thing that needs to run every few seconds.

SIGNAL LOGIC (simplified from the full live_strategy.py version):
    bias = "long" if LTP > VWAP, "short" if LTP < VWAP
    stop_distance = ATR (14-period, DAILY bars) - switched from 5-min bars
        after real trading revealed stops that tight (often under 0.5% on
        stocks that routinely move 1%+ intraday) got clipped by ordinary
        noise almost immediately. Daily ATR better matches a same-day
        holding period that can span most of the session (trades are
        force-closed at market close - see MARKET_CLOSE_TIME).
    BUY if bias=="long" AND (no known HVN above, OR the nearest one is at
        least one ATR away AND an LVN gap sits between LTP and it)
    SELL mirrors this on the downside
    Otherwise: no signal
    NOTE: this drops the order-book-imbalance confirmation that
    live_strategy.py has, since REST batch quotes don't carry depth data -
    only the WebSocket feed does. Everything else is the same reasoning.

RVOL DEFINITION USED HERE: today's cumulative volume so far / prior-20-day
average FULL-DAY volume, with NO time-of-day scaling. This will read low
early in the session and approach a real comparison by day's end - that's
expected, not a bug. (If you actually want time-of-day-matched RVOL like
the original dashboard's rvol_baseline dict, that's a bigger addition -
say so and it can be built in.)

KEY LEVELS TAB (added): Sahi-app-style collapsed volume-profile zones.
    Reuses the same build_volume_profile()/find_hvn_lvn() calls that power
    HVN/LVN above - see sahi_style_key_levels.py for the zone-segmentation
    logic. Computed once per symbol during Precompute (same cost as the
    existing HVN/LVN calc) and cached alongside it, so nothing extra runs
    on every Refresh.
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

IST = timezone(timedelta(hours=5, minutes=30))

def now_ist():
    return datetime.now(IST)

# ---------------- Config ----------------
INSTRUMENT_SEARCH_URL = "https://api.upstox.com/v2/instruments/search"
QUOTES_URL = "https://api.upstox.com/v2/market-quote/quotes"
CACHE_PATH = "hvn_cache.json"
INTRADAY_LOOKBACK_DAYS = 18
DAILY_LOOKBACK_DAYS = 30
RVOL_BASELINE_DAYS = 20
ATR_PERIOD = 14
TODAY_N_BINS = 30
COMPOSITE_N_BINS = 50
MIN_NODE_SEPARATION_BINS = 3
TRADE_LOG_PATH = "trade_log.json"
STOP_ATR_MULT = 1.25
TARGET_R_MULTIPLE = 1.75

# Key Levels tab tuning: finer bins + lower prominence threshold than the
# HVN/LVN signal logic above, so smaller-but-real clusters (e.g. a brief
# midday base) surface as their own zone instead of being merged into a
# neighbor. Tested against synthetic 4-cluster sessions: old defaults
# (n_bins=30, prominence=0.15) found 3 zones and merged the smallest
# cluster away; these settings recovered all 4.
SAHI_ZONE_N_BINS = 45
SAHI_MIN_PROMINENCE_PCT = 0.08
SAHI_MIN_BIN_DISTANCE = 2
SAHI_MAX_ZONES = 6
SAHI_MIN_DISPLAY_PCT = 2.0
MARKET_OPEN_TIME = dtime(9, 15)   # IST - no new paper-trade entries logged before this
MARKET_CLOSE_TIME = dtime(15, 30)  # IST - trades still OPEN at/after this are force-closed

# Same universe as the original scanner - trim this list if you want fewer
# symbols. NIFTY/BANKNIFTY are handled separately below (futures, not equity).
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
        pass  # st.secrets raises if no secrets.toml exists at all (fine for local dev)
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
    """No expiry filter - the 'current_month' keyword returns zero results
    once that month's contract has expired but the calendar hasn't rolled
    over yet (confirmed bug, found and fixed earlier). Sorting client-side
    by expiry and taking the earliest is more robust regardless of keyword
    edge cases."""
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


def compute_atr(df, period=ATR_PERIOD):
    if df.empty or len(df) < period + 1:
        return None
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return round(tr.tail(period).mean(), 4)


def compute_hvn_lvn(df):
    """Fixed-bin-count sizing, same fix validated earlier today (RELIANCE
    composite had 13 noisy nodes with ATR-based sizing; fixed bin count
    across each window's own range fixed it)."""
    if df.empty:
        return [], [], [], []
    today = df["date"].max()
    today_df = df[df["date"] == today]

    def _profile(candle_df, n_bins):
        try:
            price_range = candle_df["high"].max() - candle_df["low"].min()
            bin_size = max(price_range / n_bins, 0.01) if price_range > 0 else 0.01
            bins, vols = build_volume_profile(candle_df, bin_size=bin_size)
            result = find_hvn_lvn(bins, vols, min_bin_distance=MIN_NODE_SEPARATION_BINS)
            return result["hvns"], result["lvns"]
        except Exception:
            return [], []

    today_hvn, today_lvn = _profile(today_df, TODAY_N_BINS)
    composite_hvn, composite_lvn = _profile(df, COMPOSITE_N_BINS)
    return today_hvn, today_lvn, composite_hvn, composite_lvn


def compute_sahi_zones(df):
    """Sahi-app-style collapsed volume-profile zones for today's session
    only (see sahi_style_key_levels.py). Returns plain dicts, not
    KeyLevelZone objects, since this gets persisted to hvn_cache.json."""
    if df.empty:
        return []
    today = df["date"].max()
    today_df = df[df["date"] == today]
    try:
        _, shown_zones = sahi_style_key_levels(
            today_df,
            n_bins=SAHI_ZONE_N_BINS,
            max_zones=SAHI_MAX_ZONES,
            min_display_pct=SAHI_MIN_DISPLAY_PCT,
            min_prominence_pct=SAHI_MIN_PROMINENCE_PCT,
            min_bin_distance=SAHI_MIN_BIN_DISTANCE,
        )
        return [asdict(z) for z in shown_zones]
    except Exception:
        return []


def nearest_hvn_above(pool, price):
    candidates = [n["price"] for n in pool if n["price"] > price]
    return min(candidates) if candidates else None


def nearest_hvn_below(pool, price):
    candidates = [n["price"] for n in pool if n["price"] < price]
    return max(candidates) if candidates else None


def has_lvn_between(pool, low, high):
    return any(low < n["price"] < high for n in pool)


def compute_signal(ltp, vwap, atr, today_hvn, today_lvn, composite_hvn, composite_lvn):
    if ltp is None or vwap is None or atr is None or atr <= 0:
        return "-"
    hvn_pool = today_hvn + composite_hvn
    lvn_pool = today_lvn + composite_lvn

    if ltp > vwap:
        above = nearest_hvn_above(hvn_pool, ltp)
        confirms = above is None or (above - ltp >= atr and has_lvn_between(lvn_pool, ltp, above))
        return "BUY" if confirms else "-"
    elif ltp < vwap:
        below = nearest_hvn_below(hvn_pool, ltp)
        confirms = below is None or (ltp - below >= atr and has_lvn_between(lvn_pool, below, ltp))
        return "SELL" if confirms else "-"
    return "-"


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
            intraday_df = fetch_candles(key, token, "minutes", "5", INTRADAY_LOOKBACK_DAYS)

            prev_close = float(daily_df["close"].iloc[-1]) if not daily_df.empty else None
            avg_daily_volume = (float(daily_df["volume"].tail(RVOL_BASELINE_DAYS).mean())
                                 if len(daily_df) >= RVOL_BASELINE_DAYS else None)
            atr = compute_atr(daily_df)
            today_hvn, today_lvn, composite_hvn, composite_lvn = compute_hvn_lvn(intraday_df)
            sahi_zones = compute_sahi_zones(intraday_df)

            cache[symbol] = {
                "instrument_key": key, "prev_close": prev_close,
                "avg_daily_volume": avg_daily_volume, "atr": atr,
                "today_hvn": today_hvn, "today_lvn": today_lvn,
                "composite_hvn": composite_hvn, "composite_lvn": composite_lvn,
                "sahi_zones": sahi_zones,
            }
        except Exception as e:
            st.warning(f"{symbol}: precompute failed ({e}), skipping.")
        if progress_callback:
            progress_callback(i + 1, len(all_symbols), symbol)
        time.sleep(0.15)

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
    symbols = list(cache.keys())
    instrument_keys = [cache[s]["instrument_key"] for s in symbols]
    key_to_symbol = {cache[s]["instrument_key"]: s for s in symbols}
    quotes = fetch_batch_quotes(instrument_keys, token)

    rows = []
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
        signal = compute_signal(ltp, vwap, c.get("atr"),
                                 c.get("today_hvn", []), c.get("today_lvn", []),
                                 c.get("composite_hvn", []), c.get("composite_lvn", []))

        rows.append({
            "Symbol": symbol, "PrevClose": prev_close, "LTP": ltp,
            "Change%": change_pct, "VWAP": vwap, "RVOL%": rvol_pct, "Signal": signal,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Symbol").reset_index(drop=True)
        df.insert(0, "S.No", range(1, len(df) + 1))
    return df


def load_trade_log():
    if os.path.exists(TRADE_LOG_PATH):
        with open(TRADE_LOG_PATH, "r") as f:
            return json.load(f)
    return {"trades": []}


def save_trade_log(log):
    with open(TRADE_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def update_trade_log(log, scan_df, cache):
    """Two jobs, run on every refresh:
    1. Check existing OPEN trades against the latest LTP - close them out
       (STOP HIT / TARGET HIT) if price has reached either level.
    2. Log any fresh BUY/SELL signal as a new OPEN trade, skipping symbols
       that already have one open (so a signal that stays active across
       several refreshes doesn't get logged repeatedly).
    NOTE: stop/target checks only happen when the app is actually refreshed -
    if price blows through a level between visits, it's caught at whatever
    LTP is live the next time someone opens the app, not at the exact level.
    Same limitation the original dashboard's paper trader documented."""
    if scan_df.empty:
        return log

    price_lookup = scan_df.set_index("Symbol")["LTP"].to_dict()

    # Snapshot BEFORE closing anything - a trade that closes this call
    # should not immediately reopen a fresh position in the same refresh
    # (that would look like an instant, unrealistic re-entry right at the
    # old target/stop price). It becomes eligible again starting next refresh.
    open_symbols = {t["symbol"] for t in log["trades"] if t["status"] == "OPEN"}

    now = now_ist()
    for t in log["trades"]:
        if t["status"] != "OPEN":
            continue
        current_price = price_lookup.get(t["symbol"])

        # EOD square-off: this is a same-day intraday system with no other
        # exit mechanism, so an OPEN trade must be force-closed by market
        # close - otherwise ATR(14, 5-min bars), sized for roughly a
        # 70-minute move, gets applied against a holding period of many
        # hours or (if this check didn't exist) even days, which is a
        # structural mismatch, not a sizing bug. Covers two cases: a trade
        # still open from an earlier day (should never happen once this
        # runs daily, but the log is a persisted file, so it's a real
        # possibility after a gap in usage), and a trade opened today
        # that's still open once the session has actually ended.
        #
        # IMPORTANT: this check runs BEFORE the "no current price" skip
        # below (unlike the stop/target check, which legitimately needs a
        # live price to compare against) - a stale/missing quote must not
        # block the square-off, since the whole point is to guarantee the
        # position closes by day's end regardless of whether this
        # particular refresh happened to get fresh data for this symbol.
        # Falls back to entry_price (flat exit, 0% P&L) only in the rare
        # case a live price was never available - an approximation, but
        # far better than leaving the trade open indefinitely.
        entry_date = datetime.strptime(t["entry_time"], "%Y-%m-%d %H:%M:%S").date()
        is_stale_from_earlier_day = entry_date < now.date()
        is_past_close_today = entry_date == now.date() and now.time() >= MARKET_CLOSE_TIME
        if is_stale_from_earlier_day or is_past_close_today:
            t["status"] = "EOD SQUARE-OFF"
            t["exit_price"] = current_price if current_price is not None else t["entry_price"]
            t["exit_time"] = now.strftime("%Y-%m-%d %H:%M:%S")
            continue

        if current_price is None:
            continue

        if t["action"] == "BUY":
            if current_price >= t["target"]:
                t["status"], t["exit_price"] = "TARGET HIT", current_price
                t["exit_time"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            elif current_price <= t["stop"]:
                t["status"], t["exit_price"] = "STOP HIT", current_price
                t["exit_time"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
        else:  # SELL
            if current_price <= t["target"]:
                t["status"], t["exit_price"] = "TARGET HIT", current_price
                t["exit_time"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")
            elif current_price >= t["stop"]:
                t["status"], t["exit_price"] = "STOP HIT", current_price
                t["exit_time"] = now_ist().strftime("%Y-%m-%d %H:%M:%S")

    # New entries only get logged while the market is actually open. A
    # Refresh outside trading hours (e.g. checking the app in the evening,
    # as happened above - entries logged at 21:37 IST) pulls Upstox's
    # frozen post-close LTP/VWAP, which would otherwise get logged as a
    # live signal against prices that will never move again that day.
    # This does not affect the EOD square-off loop above, which must keep
    # running regardless of the current time to close out anything left
    # OPEN from earlier in the session.
    market_is_open = MARKET_OPEN_TIME <= now.time() < MARKET_CLOSE_TIME
    if not market_is_open:
        return log

    for _, row in scan_df.iterrows():
        if row["Signal"] not in ("BUY", "SELL") or row["Symbol"] in open_symbols:
            continue
        atr = cache.get(row["Symbol"], {}).get("atr")
        if not atr:
            continue
        stop_distance = atr * STOP_ATR_MULT
        entry = row["LTP"]
        if row["Signal"] == "BUY":
            stop, target = entry - stop_distance, entry + stop_distance * TARGET_R_MULTIPLE
        else:
            stop, target = entry + stop_distance, entry - stop_distance * TARGET_R_MULTIPLE
        log["trades"].append({
            "symbol": row["Symbol"], "action": row["Signal"],
            "entry_time": now_ist().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_price": entry, "stop": round(stop, 2), "target": round(target, 2),
            "status": "OPEN", "exit_price": None, "exit_time": None,
        })
    return log


def build_trade_display_df(log, scan_df):
    if not log["trades"]:
        return pd.DataFrame()
    price_lookup = scan_df.set_index("Symbol")["LTP"].to_dict() if not scan_df.empty else {}
    rows = []
    for t in log["trades"]:
        mark_price = price_lookup.get(t["symbol"]) if t["status"] == "OPEN" else t["exit_price"]
        pnl_pct = None
        if mark_price is not None:
            if t["action"] == "BUY":
                pnl_pct = round((mark_price - t["entry_price"]) / t["entry_price"] * 100, 2)
            else:
                pnl_pct = round((t["entry_price"] - mark_price) / t["entry_price"] * 100, 2)
        rows.append({
            "Symbol": t["symbol"], "Action": t["action"], "EntryTime": t["entry_time"],
            "EntryPrice": t["entry_price"], "Stop": t["stop"], "Target": t["target"],
            "Status": t["status"], "ExitPrice": t["exit_price"], "ExitTime": t["exit_time"],
            "PnL%": pnl_pct,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("EntryTime", ascending=False).reset_index(drop=True)
    df.insert(0, "S.No", range(1, len(df) + 1))
    return df


def attach_signal_times(scan_df, trade_log):
    """Adds a SignalTime column - when the currently-active signal for each
    symbol was first logged as an OPEN paper trade. Blank if there's no
    active signal right now, or if ATR was unavailable at log time (trades
    without ATR data aren't logged at all, so no timestamp exists for them)."""
    if scan_df.empty:
        return scan_df
    open_entry_times = {t["symbol"]: t["entry_time"] for t in trade_log["trades"] if t["status"] == "OPEN"}
    scan_df = scan_df.copy()
    scan_df["SignalTime"] = scan_df["Symbol"].map(open_entry_times)
    return scan_df


def build_zones_display_df(zones):
    """Formats a symbol's cached sahi_zones list (plain dicts) into a
    display table for the Key Levels tab, top price to bottom."""
    if not zones:
        return pd.DataFrame()
    df = pd.DataFrame(zones)
    df = df[["price_mode", "label", "price_low", "price_high", "color"]]
    df.columns = ["Level", "Zone %", "Range Low", "Range High", "Color"]
    df = df.sort_values("Level", ascending=False).reset_index(drop=True)
    return df


# ---------------- UI (three tabs: Scanner, Paper Trades, Key Levels) ----------------
st.set_page_config(page_title="HVN/LVN Scanner", layout="wide")
st.title("HVN/LVN Scanner")

col1, col2 = st.columns(2)
run_precompute_clicked = col1.button("Run Precompute (slow, once/day)")
refresh_clicked = col2.button("Refresh (fast)")

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

    if refresh_clicked or "last_scan_df" not in st.session_state:
        token = get_token()
        scan_df = run_live_scan(cache, token)

        trade_log = load_trade_log()
        trade_log = update_trade_log(trade_log, scan_df, cache)
        save_trade_log(trade_log)

        scan_df = attach_signal_times(scan_df, trade_log)

        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_refresh_time"] = now_ist().strftime("%H:%M:%S")
        st.session_state["trade_log"] = trade_log

    df = st.session_state.get("last_scan_df", pd.DataFrame())
    trade_log = st.session_state.get("trade_log") or load_trade_log()

    tab_scanner, tab_trades, tab_levels = st.tabs(["Scanner", "Paper Trades", "Key Levels"])

    with tab_scanner:
        st.caption(f"Last refreshed: {st.session_state.get('last_refresh_time', 'never')}")
        if df.empty:
            st.write("No data yet - click Refresh.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_trades:
        st.caption(
            "Every BUY/SELL signal is logged here automatically the first time it appears. "
            "OPEN trades are marked-to-market against the latest refresh; STOP HIT / TARGET HIT lock in "
            "once price actually reaches that level - checked only when the app is refreshed, not tick-by-tick, "
            "so a level crossed between visits is caught at the next refresh's price, not the exact level. "
            "This log persists across refreshes and the app sleeping/waking, but resets on redeploy "
            "(e.g. after a future code update is pushed) - true persistence across deploys needs an "
            "external store (Google Sheet, small database, etc.), which hasn't been built yet."
        )
        trade_df = build_trade_display_df(trade_log, df)
        if trade_df.empty:
            st.write("No trades logged yet.")
        else:
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
            csv = trade_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download trade log CSV", csv, "trade_log.csv", "text/csv")

    with tab_levels:
        st.caption(
            "Sahi-app-style collapsed volume-profile zones: today's session profile, cut at the "
            "LVN valleys already found for HVN/LVN above, each zone labeled with its % share of "
            "today's volume. Refreshed on each 'Run Precompute', not on every 'Refresh'."
        )
        symbols_with_zones = [s for s in cache if cache[s].get("sahi_zones")]
        if not symbols_with_zones:
            st.write("No zones available yet - click 'Run Precompute'.")
        else:
            default_idx = symbols_with_zones.index("NIFTY") if "NIFTY" in symbols_with_zones else 0
            selected_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx)
            zones_df = build_zones_display_df(cache[selected_symbol]["sahi_zones"])
            if zones_df.empty:
                st.write(f"No zones found for {selected_symbol}.")
            else:
                st.dataframe(zones_df, use_container_width=True, hide_index=True)
else:
    st.info("No cache found yet - click 'Run Precompute' first.")
