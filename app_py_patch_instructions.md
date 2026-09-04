# How to wire the chart into app.py

## 1. Add the import near the top, with your other local imports:

```python
from candles_with_levels import plot_candles_with_zones
```

## 2. Change this line:

```python
tab_scanner, tab_levels, tab_alerts = st.tabs(["Scanner", "Key Levels", "Alerts"])
```

to:

```python
tab_scanner, tab_levels, tab_chart, tab_alerts = st.tabs(["Scanner", "Key Levels", "Chart", "Alerts"])
```

## 3. Add this new block right after the `with tab_levels:` block (same indent level, sibling `with` block):

```python
with tab_chart:
    if not symbols_with_zones:
        st.write("No zones available yet - click 'Run Precompute'.")
    else:
        chart_symbol = st.selectbox("Symbol", symbols_with_zones, index=default_idx, key="chart_symbol")
        c = cache[chart_symbol]
        token = get_token()

        # today's 5-min candles for the plotted price action
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
```

Notes:
- Reuses `symbols_with_zones`, `default_idx`, `cache`, `cross_validated_zones` -- all
  already defined earlier in the file inside `tab_levels`, so no duplicate logic needed.
- Reuses `fetch_today_candles()` -- already defined in app.py, same call used for the
  "Refresh Zones" tier, so no extra API load beyond what you're already making.
- This calls `fetch_today_candles` fresh each time the tab renders/reruns. If that feels
  too chatty on repeated tab switches, cache it with `@st.cache_data(ttl=60)` on
  `fetch_today_candles` itself (affects the whole app, not just this tab).
