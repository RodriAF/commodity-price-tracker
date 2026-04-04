"""
Page 4 — Forecasting
"Where is it going?"
Ensemble forecast front and center, model comparison, historical percentile context.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, json, glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipeline.data_pipeline import DataPipeline
from utils.config_loader import ConfigLoader

st.set_page_config(
    page_title="Forecasting — Commodity Tracker",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:rgba(13, 15, 18, 1); color:#e2e8f0; }
.stApp { background:rgba(13, 15, 18, 1); }

.page-header { border-bottom:1px solid #2a2f3a; padding-bottom:1.2rem; margin-bottom:2rem; }
.page-title  { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600;
               color:rgba(148, 163, 184, 1); letter-spacing:0.12em; text-transform:uppercase; margin:0; }
.page-subtitle { font-size:2rem; font-weight:600; color:#f1f5f9; margin:0.25rem 0 0 0; }

.section-title { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#475569;
                 letter-spacing:0.12em; text-transform:uppercase;
                 border-bottom:1px solid #1e2330; padding-bottom:0.5rem; margin:2rem 0 1rem 0; }

.forecast-card { background:#141720; border:1px solid #1e2330; border-radius:8px;
                 padding:1.25rem 1.5rem; }
.fc-label  { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#475569;
             letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem; }
.fc-value  { font-family:'IBM Plex Mono',monospace; font-size:1.8rem; font-weight:600;
             color:#f1f5f9; line-height:1; }
.fc-delta  { font-family:'IBM Plex Mono',monospace; font-size:0.85rem; margin-top:0.3rem; }
.fc-delta.pos { color:rgba(34, 197, 94, 1); }
.fc-delta.neg { color:rgba(239, 68, 68, 1); }
.fc-delta.neu { color:#64748b; }

.confidence-badge { display:inline-block; padding:3px 10px; border-radius:3px;
                    font-family:'IBM Plex Mono',monospace; font-size:0.7rem;
                    font-weight:600; letter-spacing:0.08em; text-transform:uppercase; }
.conf-high   { background:#1a2e1a; color:#4ade80; }
.conf-medium { background:#2e2a1a; color:#fbbf24; }
.conf-low    { background:#2e1a1a; color:#f87171; }

.model-row { display:flex; align-items:center; gap:0.75rem; padding:0.6rem 0.75rem;
             border-radius:5px; margin-bottom:0.3rem; background:#141720;
             border:1px solid #1e2330; }
.model-name  { font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
               color:rgba(148, 163, 184, 1); flex:1; }
.model-mape  { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#64748b; }
.model-bar   { height:4px; border-radius:2px; background:rgba(245, 158, 11, 1); transition:width 0.3s; }
.model-bar-wrap { flex:1; max-width:80px; background:#1e2330; border-radius:2px; height:4px; }

.callout { background:#141720; border-radius:6px; border-left:3px solid rgba(51, 65, 85, 1);
           padding:0.75rem 1rem;
           font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
           color:rgba(148, 163, 184, 1); line-height:1.5; margin-bottom:0.5rem; }

div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; color:#f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important;
                                    font-size:0.68rem !important; color:#475569 !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
div[data-testid="stMetricDelta"] { font-family:'IBM Plex Mono',monospace !important; }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor='rgba(13, 15, 18, 1)', plot_bgcolor='rgba(13, 15, 18, 1)',
    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
    xaxis=dict(showgrid=False, zeroline=False, color='rgba(51, 65, 85, 1)',
               tickcolor='rgba(51, 65, 85, 1)', linecolor='#1e2330'),
    yaxis=dict(showgrid=True, gridcolor='rgba(26, 31, 46, 1)', zeroline=False,
               color='rgba(51, 65, 85, 1)', tickcolor='rgba(51, 65, 85, 1)'),
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    return DataPipeline().load_latest()

@st.cache_data(ttl=3600)
def load_forecasts():
    forecast_dir = os.path.join('data', 'forecasts')
    if not os.path.exists(forecast_dir):
        return None
    files = glob.glob(os.path.join(forecast_dir, 'forecasts_*.json'))
    if not files:
        return None
    # Load the most recently modified forecast file
    with open(max(files, key=os.path.getmtime)) as f:
        return json.load(f)

df             = load_data()
forecasts_data = load_forecasts()

if df.empty:
    st.error("No data found. Run `python automation/run_daily.py` first.")
    st.stop()

if not forecasts_data:
    st.warning("No forecasts available. Run the pipeline first.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-title">Agricultural Commodity Tracker</p>
  <p class="page-subtitle">Multi-Horizon Forecasting</p>
</div>
""", unsafe_allow_html=True)

# ── Commodity selector ────────────────────────────────────────────────────────
# Show only commodities for which a valid ensemble forecast exists
available = [k for k, v in forecasts_data['forecasts'].items()
             if 'ensemble' in v and v['ensemble'] is not None]

if not available:
    st.warning("No successful forecasts in the latest run.")
    st.stop()

commodity = st.selectbox(
    "Commodity",
    available,
    format_func=lambda x: ConfigLoader.get_commodity_info(x).get('name', x.replace('_', ' ').title())
)

results       = forecasts_data['forecasts'][commodity]
info          = ConfigLoader.get_commodity_info(commodity)
frequency     = info.get('frequency', 'monthly')
ensemble      = results.get('ensemble', {})
individual    = results.get('individual_models', {})
current_price = results.get('current_price', 0)
horizon       = len(ensemble.get('predictions', []))

# Frequency-to-pandas-offset mapping for future date generation
freq_map = {'daily': 'D', 'weekly': 'W', 'monthly': 'MS', 'quarterly': 'QS'}

def make_future_dates(last_date, frequency, horizon):
    """Generate forecast period dates starting after the last observed date."""
    return pd.date_range(
        start=last_date, periods=horizon + 1,
        freq=freq_map.get(frequency, 'MS')
    )[1:]

last_date    = df['date'].iloc[-1]
future_dates = make_future_dates(last_date, frequency, horizon)

period_label = {'daily': 'Day', 'weekly': 'Week', 'monthly': 'Month', 'quarterly': 'Quarter'}

# ── Forecast summary KPIs ─────────────────────────────────────────────────────
st.markdown('<p class="section-title">Forecast Summary</p>', unsafe_allow_html=True)

series   = df[commodity].dropna()
cur_pct  = (series < current_price).mean() * 100
conf     = ensemble.get('confidence', 'low')
next_p   = ensemble['predictions'][0] if ensemble.get('predictions') else current_price
next_chg = (next_p - current_price) / current_price * 100

kc1, kc2, kc3, kc4 = st.columns(4)

with kc1:
    st.markdown(f"""
    <div class="forecast-card">
        <div class="fc-label">Current Price</div>
        <div class="fc-value">${current_price:.2f}</div>
        <div class="fc-delta neu">{info.get('unit','')}</div>
    </div>
    """, unsafe_allow_html=True)

with kc2:
    chg_cls = 'pos' if next_chg > 0 else 'neg' if next_chg < 0 else 'neu'
    arrow   = 'Up' if next_chg > 0 else 'Down' if next_chg < 0 else 'Flat'
    st.markdown(f"""
    <div class="forecast-card">
        <div class="fc-label">Next {period_label.get(frequency,'Period')}</div>
        <div class="fc-value">${next_p:.2f}</div>
        <div class="fc-delta {chg_cls}">{arrow} {next_chg:+.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with kc3:
    conf_cls = f'conf-{conf}'
    st.markdown(f"""
    <div class="forecast-card">
        <div class="fc-label">Model Confidence</div>
        <div style="margin-top:0.5rem;">
            <span class="confidence-badge {conf_cls}">{conf.upper()}</span>
        </div>
        <div class="fc-delta neu" style="margin-top:0.5rem;">
            MAPE: {ensemble.get('avg_mape', 0):.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

with kc4:
    if   cur_pct > 80: ctx_color = 'rgba(239, 68, 68, 1)'; ctx_label = 'Historically High'
    elif cur_pct < 20: ctx_color = 'rgba(34, 197, 94, 1)'; ctx_label = 'Historically Low'
    else:              ctx_color = 'rgba(245, 158, 11, 1)'; ctx_label = 'Normal Range'
    st.markdown(f"""
    <div class="forecast-card">
        <div class="fc-label">Historical Position</div>
        <div class="fc-value" style="font-size:1.4rem;color:{ctx_color};">{cur_pct:.0f}th pct.</div>
        <div class="fc-delta neu">{ctx_label}</div>
    </div>
    """, unsafe_allow_html=True)

# ── Forecast chart ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Forecast Visualisation</p>', unsafe_allow_html=True)

# Rank individual models by MAPE on the hold-out test set
valid_models = {
    k: v for k, v in individual.items()
    if 'predictions' in v and 'metrics' in v
}
ranked = sorted(valid_models.items(), key=lambda x: x[1]['metrics'].get('mape', 999))
top3_methods = [v.get('method', k) for k, v in ranked[:3]]
all_methods  = [v.get('method', k) for k, v in ranked]

selected_models = st.multiselect(
    "Individual models to overlay",
    all_methods, default=top3_methods,
    help="The ensemble is always displayed. Select individual models to compare against it."
)

fig = go.Figure()

# Historical price series — show last 24 months for readability
lookback = min(len(df), 730)
df_plot  = df.tail(lookback)

fig.add_trace(go.Scatter(
    x=df_plot['date'], y=df_plot[commodity],
    mode='lines', name='Historical',
    line=dict(width=2.5, color='#475569'),
    hovertemplate='%{x|%d %b %Y}<br>$%{y:.2f}<extra></extra>'
))

# Marker at the last observed price point
fig.add_trace(go.Scatter(
    x=[last_date], y=[current_price],
    mode='markers', name='Current',
    marker=dict(size=10, color='#f1f5f9', symbol='circle',
                line=dict(width=2, color='rgba(13, 15, 18, 1)')),
    showlegend=False, hoverinfo='skip'
))

# Individual model traces (dotted, reduced opacity)
for k, v in ranked:
    method = v.get('method', k)
    if method not in selected_models:
        continue
    preds = v['predictions'][:horizon]
    fig.add_trace(go.Scatter(
        x=[last_date] + list(future_dates),
        y=[current_price] + preds,
        mode='lines', name=method,
        line=dict(width=1.5, dash='dot'),
        opacity=0.5,
        hovertemplate=f'<b>{method}</b><br>%{{x|%b %Y}}<br>$%{{y:.2f}}<extra></extra>'
    ))

# GARCH volatility confidence bands (if the model produced volatility forecasts)
garch_data = individual.get('garch', {})
if 'predictions' in garch_data and 'volatility_forecast' in garch_data:
    g_preds = garch_data['predictions'][:horizon]
    g_vols  = garch_data['volatility_forecast'][:horizon]
    upper   = [p * (1 + 1.645 * v) for p, v in zip(g_preds, g_vols)]
    lower   = [max(0, p * (1 - 1.645 * v)) for p, v in zip(g_preds, g_vols)]

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(reversed(future_dates)),
        y=upper + list(reversed(lower)),
        fill='toself', fillcolor='rgba(248,113,113,0.06)',
        line=dict(color='rgba(0,0,0,0)'),
        name='GARCH 90% band',
        hoverinfo='skip'
    ))

# Ensemble trace — prominent styling to distinguish from individual models
fig.add_trace(go.Scatter(
    x=[last_date] + list(future_dates),
    y=[current_price] + ensemble['predictions'],
    mode='lines+markers', name='Ensemble',
    line=dict(width=3.5, color='rgba(245, 158, 11, 1)'),
    marker=dict(size=7, color='rgba(245, 158, 11, 1)',
                line=dict(width=1.5, color='rgba(13, 15, 18, 1)')),
    hovertemplate='<b>Ensemble</b><br>%{x|%b %Y}<br>$%{y:.2f}<extra></extra>'
))

# Vertical separator at the forecast origin
fig.add_vline(
    x=pd.to_datetime(last_date),
    line_dash='dot',
    line_color='rgba(51, 65, 85, 1)',
    line_width=1.5
)

fig.add_annotation(
    x=pd.to_datetime(last_date),
    y=1,
    xref='x',
    yref='paper',
    text='Today',
    showarrow=False,
    xanchor='left',
    yanchor='bottom',
    font=dict(size=9, family='IBM Plex Mono', color='#475569')
)

fig.update_layout(
    **CHART_THEME, height=480,
    margin=dict(l=0, r=0, t=4, b=0),
    hovermode='x unified',
    legend=dict(orientation='h', y=1.07, x=0,
                font=dict(size=10, color='rgba(148, 163, 184, 1)'),
                bgcolor='rgba(0,0,0,0)'),
    yaxis_title=f"Price ({info.get('unit','')})"
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if 'volatility_forecast' in garch_data:
    st.caption("Shaded band = GARCH(1,1) 90% confidence interval (1.645σ). A wider band indicates higher modelled volatility.")

# ── Period-by-period forecast breakdown ───────────────────────────────────────
st.markdown('<p class="section-title">Period-by-Period Forecast</p>', unsafe_allow_html=True)

p_cols = st.columns(min(horizon, 6))
for i in range(horizon):
    pred  = ensemble['predictions'][i]
    chg   = (pred - current_price) / current_price * 100
    label = f"{period_label.get(frequency,'P')} {i+1}"
    chg_c = 'pos' if chg > 0 else 'neg' if chg < 0 else 'neu'
    direction = 'Up' if chg > 0 else 'Down' if chg < 0 else 'Flat'
    with p_cols[i]:
        st.markdown(f"""
        <div class="forecast-card" style="text-align:center;">
            <div class="fc-label">{label}</div>
            <div class="fc-value" style="font-size:1.3rem;">${pred:.2f}</div>
            <div class="fc-delta {chg_c}">{direction} {chg:+.2f}%</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                        color:rgba(51, 65, 85, 1);margin-top:0.3rem;">
                {future_dates[i].strftime('%d %b %Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Model performance comparison ──────────────────────────────────────────────
st.markdown('<p class="section-title">Model Performance Comparison</p>',
            unsafe_allow_html=True)
st.markdown(
    "<p class='callout'>"
    "Models are ranked by MAPE (Mean Absolute Percentage Error) on the hold-out test set. "
    "The ensemble uses an exponential weighted average of the top-performing models. "
    "Lower MAPE indicates better out-of-sample fit."
    "</p>",
    unsafe_allow_html=True
)

if ranked:
    best_mape = ranked[0][1]['metrics'].get('mape', 1)

    for i, (k, v) in enumerate(ranked):
        method = v.get('method', k)
        mape   = v['metrics'].get('mape', 0)
        mae    = v['metrics'].get('mae', 0)
        conf_m = v.get('confidence', '—').upper()
        bar_w  = max(4, min(100, (1 - (mape - best_mape) / (best_mape + 1)) * 100))
        is_top = method in top3_methods

        tag = '<span style="font-size:0.65rem;color:rgba(245, 158, 11, 1);margin-left:0.5rem">TOP</span>' if is_top else ''
        st.markdown(f"""
        <div class="model-row">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;
                         color:rgba(51, 65, 85, 1);min-width:18px;">#{i+1}</span>
            <span class="model-name">{method}{tag}</span>
            <div class="model-bar-wrap">
                <div class="model-bar" style="width:{bar_w}%;"></div>
            </div>
            <span class="model-mape">MAPE {mape:.1f}%</span>
            <span style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                         color:rgba(51, 65, 85, 1);min-width:70px;text-align:right;">
                MAE {mae:.2f}
            </span>
        </div>
        """, unsafe_allow_html=True)

# ── Historical context: where does the forecast land? ────────────────────────
st.markdown('<p class="section-title">Historical Context — Forecast Position</p>',
            unsafe_allow_html=True)

final_pred = ensemble['predictions'][-1]
fut_pct    = (series < final_pred).mean() * 100

fc1, fc2 = st.columns(2)

with fc1:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=series, nbinsx=40,
        marker_color='#1e2f3e',
        marker_line=dict(color='rgba(56, 189, 248, 0.125)', width=0.5),
        name='Historical distribution'
    ))
    # Current price reference
    fig_hist.add_vline(x=current_price, line_color='#475569',
                       line_width=1.5, line_dash='dot',
                       annotation_text='Current',
                       annotation_font=dict(size=9, family='IBM Plex Mono', color='#475569'))
    # Forecast terminal value
    fig_hist.add_vline(x=final_pred, line_color='rgba(245, 158, 11, 1)',
                       line_width=2,
                       annotation_text=f'Forecast ({fut_pct:.0f}th pct.)',
                       annotation_font=dict(size=9, family='IBM Plex Mono', color='rgba(245, 158, 11, 1)'),
                       annotation_position='top right')
    fig_hist.update_layout(
        **CHART_THEME, height=240,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        xaxis_title=info.get('unit', 'Price'),
        yaxis_title='Frequency'
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

with fc2:
    # Narrative interpretation based on the forecast's historical percentile
    if   fut_pct > 80: msg = f"The terminal forecast of ${final_pred:.2f} places the commodity at the **{fut_pct:.0f}th percentile** — historically elevated territory. Watch for potential mean reversion."
    elif fut_pct < 20: msg = f"The terminal forecast of ${final_pred:.2f} places the commodity at the **{fut_pct:.0f}th percentile** — historically low. May represent opportunity if fundamentals are supportive."
    else:              msg = f"The terminal forecast of ${final_pred:.2f} lands at the **{fut_pct:.0f}th percentile** — within the normal historical range."

    trend_dir = 'upward' if ensemble['predictions'][-1] > current_price else 'downward'
    top_m_str = ', '.join(ensemble.get('top_models', [])[:3])

    st.markdown(f"""
    <div class="forecast-card" style="height:100%;box-sizing:border-box;">
        <div class="fc-label">Interpretation</div>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.8rem;
                  color:rgba(148, 163, 184, 1);line-height:1.6;margin:0.75rem 0;">
            {msg}
        </p>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                  color:#64748b;line-height:1.5;margin:0;">
            Trend: <strong style="color:#f1f5f9">{trend_dir}</strong><br>
            Confidence: <strong style="color:#f1f5f9">{conf.upper()}</strong><br>
            Driven by: <strong style="color:#f1f5f9">{top_m_str}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Methodology ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("Methodology", expanded=False):
    st.markdown("""
<p style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:rgba(148, 163, 184, 1);line-height:1.7;">

**Models included:** ARIMA · SARIMA · Exponential Smoothing · GARCH(1,1) ·
Gradient Boosting · Random Forest · Ridge · LASSO · XGBoost

**Validation:** Hold-out test set (size adapted to data frequency).
Models are ranked by MAPE. Ensemble = exponential-weighted average of the top 3 models.

**GARCH bands:** Log-return volatility forecasts converted back to price space.
1.645σ = 90% confidence interval assuming normally distributed returns.

**Confidence levels:** HIGH = MASE < 0.8 · MEDIUM = MASE < 1.2 · LOW = MASE >= 1.2

**Disclaimer:** Forecasts are probabilistic scenarios based on historical patterns.
External shocks (geopolitical events, weather, policy changes) are not modelled.

</p>
""", unsafe_allow_html=True)