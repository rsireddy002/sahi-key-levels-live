"""
Candlestick chart with key levels overlaid.

Validated zones are classified relative to the LAST close price:
  - zone below last close  -> SUPPORT   (green line + label)
  - zone above last close  -> RESISTANCE (red line + label)
This makes the chart instantly readable: green below price, red above --
no need to read numbers to know which is which.

Composite/intraday zones remain thin, unlabeled gray reference lines.

Matches the zone dict shape from sahi_style_key_levels():
    {"price_mode": float, "label": str (e.g. "31%"), "price_low": float, "price_high": float}
"""

import re
import plotly.graph_objects as go

SUPPORT_FILL = "rgba(99, 153, 34, 0.12)"
SUPPORT_LINE = "#3B6D11"
RESISTANCE_FILL = "rgba(226, 75, 74, 0.12)"
RESISTANCE_LINE = "#A32D2D"
REFERENCE_LINE = "rgba(136, 135, 128, 0.5)"  # faint gray, no label


def _pct_from_label(label):
    m = re.search(r"[\d.]+", str(label))
    return float(m.group()) if m else 0.0


def plot_candles_with_zones(df, composite_zones=None, intraday_zones=None,
                             validated_zones=None, title="Price with key levels"):
    """
    df: OHLC dataframe with columns ['timestamp','open','high','low','close'].
    composite_zones / intraday_zones: shown as faint reference lines only.
    validated_zones: classified as support/resistance vs last close,
        shown as shaded bands with price + zone% labels.
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
    last_close = float(df["close"].iloc[-1])

    # Faint reference lines for composite + intraday -- no fill, no label.
    for zones in [composite_zones or [], intraday_zones or []]:
        for z in zones:
            fig.add_shape(
                type="line", x0=x0, x1=x1, y0=z["price_mode"], y1=z["price_mode"],
                line=dict(color=REFERENCE_LINE, width=1, dash="dot"),
            )

    # Validated zones -- classified as support (below price) or
    # resistance (above price), colored and labeled accordingly.
    val = sorted(validated_zones or [], key=lambda z: z["price_mode"], reverse=True)
    price_span = (df["high"].max() - df["low"].min()) if len(df) > 1 else 1
    min_gap = price_span * 0.04

    placed_y = []
    for z in val:
        is_resistance = z["price_mode"] >= last_close
        fill = RESISTANCE_FILL if is_resistance else SUPPORT_FILL
        line_color = RESISTANCE_LINE if is_resistance else SUPPORT_LINE
        kind_label = "Resistance" if is_resistance else "Support"

        fig.add_hrect(
            y0=z["price_low"], y1=z["price_high"],
            fillcolor=fill, line_width=0, layer="below",
        )
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=z["price_mode"], y1=z["price_mode"],
            line=dict(color=line_color, width=2, dash="dash"),
        )

        label_y = z["price_mode"]
        for py in placed_y:
            if abs(label_y - py) < min_gap:
                label_y = py - min_gap
        placed_y.append(label_y)

        fig.add_annotation(
            x=x1, y=label_y,
            text=f"{kind_label} {z['price_mode']:.0f} ({_pct_from_label(z['label']):.0f}%)",
            showarrow=(label_y != z["price_mode"]),
            arrowhead=0, arrowwidth=1, arrowcolor=line_color,
            ax=45, ay=0,
            xanchor="left", font=dict(size=11, color=line_color),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=line_color, borderwidth=1, borderpad=3,
        )

    # marker for last close so it's obvious where "current price" sits
    fig.add_shape(
        type="line", x0=x0, x1=x1, y0=last_close, y1=last_close,
        line=dict(color="#185FA5", width=1, dash="solid"),
    )
    fig.add_annotation(
        x=x0, y=last_close, text=f"LTP {last_close:.0f}",
        showarrow=False, xanchor="right", font=dict(size=10, color="#185FA5"),
        bgcolor="rgba(255,255,255,0.9)",
    )

    fig.update_layout(
        title=title,
        xaxis_title=None, yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=60, r=150, t=40, b=30),
        showlegend=False,
    )
    return fig
