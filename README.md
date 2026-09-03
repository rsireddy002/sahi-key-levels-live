# sahi-key-levels-live

Live-refreshing, cross-timeframe-validated version of the Sahi-style Key
Levels zones. A level only counts as real if it shows up in BOTH:

- the **composite** profile (18 trading days), and
- **today's intraday** profile (recomputed periodically through the session)

A zone found in only one timeframe is treated as noise and dropped. When
a symbol's price has room to run before hitting a validated level (in the
direction of the VWAP bias), a BUY/SELL alert is logged.

## Vendored files (unchanged, copied from their source repos)

- `hvn_lvn.py` — from `hvn-lvn-scanner`
- `sahi_style_key_levels.py` — from `sahi-key-levels`

These are copied as-is on purpose. The original repos and their logic are
untouched; this repo only adds `zone_validation.py` (cross-timeframe
validation + signal) and a new `app.py` that orchestrates all three.

**Before running this app**, copy both vendored files in from their
source repos:

```powershell
Copy-Item ..\hvn-lvn-scanner\hvn_lvn.py .
Copy-Item ..\sahi-key-levels\sahi_style_key_levels.py .
```

## Three-tier refresh model

| Button | Speed | What it does |
|---|---|---|
| Run Precompute | slow, once/day | Resolves instrument keys, fetches 18 days of 5-min candles per symbol, computes the **composite** zone set |
| Refresh Zones | medium, every few min | Re-fetches only *today's* candles per symbol, recomputes **intraday** zones, cross-validates, logs new alerts |
| Refresh Quotes | fast | One batch quote call for LTP/VWAP; recomputes signal against whatever zones were last computed |

This split is deliberate: candle history has to be fetched per-symbol (no
batch endpoint), so re-fetching it for 200+ symbols on every quote tick
would be far too many API calls. Only "Refresh Zones" pays that cost, and
it's meant to be clicked every few minutes, not continuously.

## Setup

```powershell
pip install -r requirements.txt
$env:UPSTOX_ACCESS_TOKEN = "your_token_here"
streamlit run app.py
```

## Signal logic

```
bias = long if LTP > VWAP, short if LTP < VWAP
validated_zones = zones whose price range overlaps a zone from the OTHER timeframe

BUY  if bias == long  AND (no validated zone above LTP, OR it's >= 0.5% away)
SELL if bias == short AND (no validated zone below LTP, OR it's >= 0.5% away)
```

Alerts are edge-triggered (logged only the moment a signal changes, not
every refresh it stays active) and only logged during market hours
(9:15–15:30 IST) — an after-hours refresh pulls Upstox's frozen post-close
quotes, which must not get logged as a live signal.
