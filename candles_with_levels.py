"""
Candlestick chart with cross-timeframe validated zones overlaid.

Matches the zone dict shape produced by sahi_style_key_levels() in this repo:
    {"price_mode": float, "label": str (e.g. "31%"), "price_low": float, "price_high": float}

Zone *kind* (composite / intraday / validated) is NOT a field on the dict --
it's determined by which list you pass in, same as app.py's tab_levels logic.
"""

import re
import plotly.graph_objects as go

ZONE_FILL = {
    "validated": "rgba(226, 75, 74, 0.15)",   # red tint - strongest
    "composite": "rgba(56, 138, 221, 0.10)",  # blue tint
    "intraday": "rgba(186, 117, 23, 0.10)",   # amber tint
}
ZONE_LINE = {
    "validated": "#A32D2D",
    "composite": "#185FA5",
    "intraday": "#854F0B",
}


def _pct_from_label(label):
    """label is something like '31%' -- pull the number back out for the annotation."""
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def plot_candles_with_zones(df, composite_zones=None, intraday_zones=None,
                             validated_zones=None, title="Price with key levels"):
    """
    df: OHLC dataframe from fetch_candles()/fetch_today_candles() in app.py
        -- must have columns ['timestamp','open','high','low','close'].
    composite_zones / intraday_zones / validated_zones: lists of zone dicts,
        straight from cache[symbol]["composite_zones"] etc, or from
        cross_validated_zones(...) for the validated set.
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#639922",
        decreasing_line_color="#E24B4A",
        name="Price",
    ))

    x0, x1 = df["timestamp"].iloc[0], df["timestamp"].iloc[-1]

    zone_groups = [
        ("composite", composite_zones or []),
        ("intraday", intraday_zones or []),
        ("validated", validated_zones or []),
    ]

    for kind, zones in zone_groups:
        for z in zones:
            fig.add_hrect(
                y0=z["price_low"], y1=z["price_high"],
                fillcolor=ZONE_FILL[kind], line_width=0, layer="below",
            )
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=z["price_mode"], y1=z["price_mode"],
                line=dict(color=ZONE_LINE[kind], width=1, dash="dot"),
            )
            fig.add_annotation(
                x=x1, y=z["price_mode"],
                text=f"{z['price_mode']:.2f} ({_pct_from_label(z['label']):.0f}%) {kind}",
                showarrow=False, xanchor="left", font=dict(size=9),
                bgcolor="rgba(255,255,255,0.7)",
            )

    fig.update_layout(
        title=title,
        xaxis_title=None, yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=40, r=100, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
