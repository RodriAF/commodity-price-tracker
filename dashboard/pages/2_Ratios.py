"""
Page 3 — Input Cost Analytics
"What does it cost to produce? Are margins under pressure?"

Architecture note — why we recalculate here:
    CommoditiesAnalytics computes cost indices and profitability ratios in
    memory during the pipeline run but does NOT persist them to DuckDB.
    Only raw prices (prices table) and signals (signals table) are stored.
    This page therefore loads raw prices, pivots them to wide format, and
    runs the same CommoditiesAnalytics pipeline to reconstruct the derived
    metrics. This is the correct pattern — derived metrics are cheap to
    recompute and should not be stored as they change with config updates.

first layout:
    1. Current environment  — plain-language regime + index value
    2. Historical context   — cost index chart (what has changed over time)
    3. Margin pressure      — which crops are under pressure and why
    4. Technical detail     — z-scores and ratio explorer (collapsed by default)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from pipeline.calculations import CommoditiesAnalytics
from utils.db import get_connection
import utils.db as db

st.set_page_config(
    page_title="Analytics — Commodity Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ------------------------------------------------------------------ #
# Styles                                                             #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background:rgba(13,15,18,1); color:#e2e8f0; }
.stApp { background:rgba(13,15,18,1); }

.page-header  { border-bottom:1px solid #2a2f3a; padding-bottom:1.2rem; margin-bottom:2rem; }
.page-title   { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600;
                color:rgba(148,163,184,1); letter-spacing:0.12em; text-transform:uppercase; margin:0; }
.page-subtitle { font-size:2rem; font-weight:600; color:#f1f5f9; margin:0.25rem 0 0 0; }

.section-title { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#475569;
                 letter-spacing:0.12em; text-transform:uppercase;
                 border-bottom:1px solid #1e2330; padding-bottom:0.5rem; margin:2rem 0 1rem 0; }

/* Regime card — coloured top border per state */
.regime-card { background:#141720; border:1px solid #1e2330; border-radius:8px;
               padding:1.25rem 1.5rem; position:relative; overflow:hidden; }
.regime-card::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.regime-card.high::before   { background:rgba(239,68,68,1); }
.regime-card.normal::before { background:rgba(245,158,11,1); }
.regime-card.low::before    { background:rgba(34,197,94,1); }
.regime-label { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#64748b;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem; }
.regime-value { font-size:1.3rem; font-weight:600; margin-bottom:0.2rem; }
.regime-value.high   { color:rgba(239,68,68,1); }
.regime-value.normal { color:rgba(245,158,11,1); }
.regime-value.low    { color:rgba(34,197,94,1); }
.regime-index { font-family:'IBM Plex Mono',monospace; font-size:1.7rem; font-weight:600;
                margin:0.5rem 0 0.1rem 0; }
.regime-index.high   { color:rgba(239,68,68,1); }
.regime-index.normal { color:rgba(245,158,11,1); }
.regime-index.low    { color:rgba(34,197,94,1); }
.regime-desc  { font-size:0.78rem; color:#64748b; line-height:1.5; margin-top:0.25rem; }

/* Margin pressure table */
.pressure-table { width:100%; border-collapse:collapse; }
.pressure-table th { font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#475569;
                     text-transform:uppercase; letter-spacing:0.08em;
                     border-bottom:1px solid #1e2330; padding:0.5rem 0.75rem; text-align:left; }
.pressure-table td { font-size:0.82rem; color:#e2e8f0; padding:0.55rem 0.75rem;
                     border-bottom:1px solid #141720; vertical-align:middle; }
.pressure-table tr:hover td { background:#141720; }
.badge { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
         font-weight:600; padding:0.2rem 0.55rem; border-radius:4px; letter-spacing:0.06em; }
.badge-pressure  { background:rgba(239,68,68,0.15); color:rgba(239,68,68,1); }
.badge-normal    { background:rgba(34,197,94,0.10); color:rgba(34,197,94,1); }
.badge-elevated  { background:rgba(245,158,11,0.12); color:rgba(245,158,11,1); }
.trend-up   { color:rgba(239,68,68,1); font-size:1rem; }
.trend-down { color:rgba(34,197,94,1); font-size:1rem; }
.trend-flat { color:#475569; font-size:1rem; }

.callout { background:#141720; border-radius:6px; border-left:3px solid rgba(51,65,85,1);
           padding:0.75rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
           color:rgba(148,163,184,1); line-height:1.5; margin-bottom:0.5rem; }

div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; color:#f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important;
                                    font-size:0.68rem !important; color:#475569 !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
</style>
""", unsafe_allow_html=True)

CHART_THEME = dict(
    paper_bgcolor='rgba(13,15,18,1)', plot_bgcolor='rgba(13,15,18,1)',
    font=dict(family='IBM Plex Mono', color='#64748b', size=10),
    xaxis=dict(showgrid=False, zeroline=False, color='rgba(51,65,85,1)',
               tickcolor='rgba(51,65,85,1)', linecolor='#1e2330'),
    yaxis=dict(showgrid=True, gridcolor='rgba(26,31,46,1)', zeroline=False,
               color='rgba(51,65,85,1)', tickcolor='rgba(51,65,85,1)'),
)

# ------------------------------------------------------------------ #
# Data Loading — always from raw prices, then recalculate            #
# ------------------------------------------------------------------ #

@st.cache_data(ttl=3600)
def load_wide_prices() -> pd.DataFrame:
    """
    Load raw prices from DuckDB and pivot to wide format.
    This is the only table that stores price data. Derived metrics
    (ratios, indices, z-scores) are always recalculated here.
    """
    try:
        with get_connection() as conn:
            df = conn.execute("SELECT date, commodity, value FROM prices ORDER BY date ASC").df()

        if df.empty:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date']).dt.normalize()

        df_wide = df.pivot_table(
            index='date',
            columns='commodity',
            values='value',
            aggfunc='last'
        ).reset_index()

        df_wide.columns.name = None
        df_wide = df_wide.sort_values('date').reset_index(drop=True)
        return df_wide

    except Exception as e:
        st.error(f"Failed to load prices from DuckDB: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def compute_analytics(df_wide: pd.DataFrame) -> dict:
    """
    Run CommoditiesAnalytics on the wide price DataFrame to produce
    cost indices, profitability ratios, z-scores, signals, and regime.
    Mirrors exactly what the pipeline does in flow.py -> analytics().
    """
    if df_wide.empty:
        return {}
    try:
        engine = CommoditiesAnalytics(df_wide)
        combined, analytics = engine.calculate_all()
        # Attach date column to combined so we can plot time series
        combined['date'] = df_wide['date'].values[:len(combined)]
        return {'combined': combined, **analytics}
    except Exception as e:
        st.error(f"Analytics computation failed: {e}")
        return {}


# ------------------------------------------------------------------ #
# Helper functions                                                   #
# ------------------------------------------------------------------ #

def _regime_label(key: str, val: str) -> tuple[str, str, str]:
    """Return (css_class, display_label, description) for a regime value."""
    meta = {
        'high_cost':  ('high',   'ABOVE AVERAGE', 'Energy inputs above historical norm — cost pressure on producers'),
        'low_cost':   ('low',    'BELOW AVERAGE', 'Energy inputs below historical norm — favourable cost conditions'),
        'normal':     ('normal', 'NORMAL',         'Energy within normal historical range'),
        'expensive':  ('high',   'ABOVE AVERAGE', 'Fertilizer costs above historical norm'),
        'cheap':      ('low',    'BELOW AVERAGE', 'Fertilizer costs below historical norm'),
    }
    return meta.get(val, ('normal', val.upper().replace('_', ' '), ''))


def _trend_arrow(series: pd.Series, window: int = 3) -> str:
    """Return a directional indicator based on recent trend."""
    recent = series.dropna().tail(window)
    if len(recent) < 2:
        return '<span class="trend-flat">—</span>'
    slope = recent.iloc[-1] - recent.iloc[0]
    if slope > 0.5:
        return '<span class="trend-up">↑</span>'
    elif slope < -0.5:
        return '<span class="trend-down">↓</span>'
    return '<span class="trend-flat">—</span>'


def _margin_status(z: float) -> tuple[str, str]:
    """Translate z-score into plain language margin status."""
    if z < -2:
        return 'badge-pressure', 'UNDER PRESSURE'
    elif z < -1:
        return 'badge-elevated', 'ELEVATED COSTS'
    elif z > 1:
        return 'badge-normal', 'STRONG MARGINS'
    return 'badge-normal', 'NORMAL'


def _crop_label(name: str) -> str:
    return name.replace('_', ' ').title()


def _input_label(name: str) -> str:
    labels = {
        'crude_oil': 'Crude Oil', 'natural_gas': 'Natural Gas',
        'diesel': 'Diesel', 'gasoline': 'Gasoline',
        'urea': 'Urea', 'dap_fertilizer': 'DAP', 'phosphate': 'Phosphate', 'potash': 'Potash',
    }
    return labels.get(name, name.replace('_', ' ').title())


# ------------------------------------------------------------------ #
# Page header                                                        #
# ------------------------------------------------------------------ #

st.markdown("""
<div class="page-header">
  <p class="page-title">Agricultural Commodity Tracker</p>
  <p class="page-subtitle">Input Cost Analytics</p>
</div>
""", unsafe_allow_html=True)

_, col_refresh = st.columns([8, 1])
with col_refresh:
    if st.button("Refresh", type="secondary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ------------------------------------------------------------------ #
# Load & compute                                                     #
# ------------------------------------------------------------------ #

df_wide = load_wide_prices()

if df_wide.empty:
    st.warning("No price data found in DuckDB. Run the pipeline first to populate the database.")
    st.stop()

analytics = compute_analytics(df_wide)

if not analytics:
    st.warning("Analytics could not be computed. Check that commodities are correctly configured.")
    st.stop()

indices       = analytics.get('cost_indices', pd.DataFrame())
profitability = analytics.get('profitability', pd.DataFrame())
zscores       = analytics.get('zscores', pd.DataFrame())
regime        = analytics.get('regime', {})
combined      = analytics.get('combined', pd.DataFrame())

# Attach dates for time series charts
if not combined.empty and 'date' not in combined.columns:
    combined.insert(0, 'date', df_wide['date'].values[:len(combined)])
if not indices.empty:
    indices['date'] = df_wide['date'].values[:len(indices)]
if not profitability.empty:
    profitability['date'] = df_wide['date'].values[:len(profitability)]

# ------------------------------------------------------------------ #
# Section 1 — Current Cost Environment (regime cards)               #
# ------------------------------------------------------------------ #

st.markdown('<p class="section-title">Current Input Cost Environment</p>', unsafe_allow_html=True)

rc1, rc2 = st.columns(2)

for col_widget, regime_key, index_col, label in [
    (rc1, 'energy',     'energy_input_cost_index', 'Energy Inputs'),
    (rc2, 'fertilizer', 'fertilizer_cost_index',   'Fertilizers'),
]:
    val = regime.get(regime_key, 'normal')
    css, display_lbl, desc = _regime_label(regime_key, val)

    # Current index value
    idx_val = None
    if not indices.empty and index_col in indices.columns:
        last_valid = indices[index_col].dropna()
        if not last_valid.empty:
            idx_val = last_valid.iloc[-1]

    # vs. 3-month-ago change
    delta_str = ""
    if not indices.empty and index_col in indices.columns and len(indices) >= 4:
        prev = indices[index_col].dropna()
        if len(prev) >= 4:
            delta = prev.iloc[-1] - prev.iloc[-4]
            sign  = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.1f} pts vs. 3 months ago"

    with col_widget:
        idx_html = ""
        if idx_val is not None and pd.notna(idx_val):
            idx_html = f'<div class="regime-index {css}">{idx_val:.1f}</div>'
            idx_html += f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.72rem;color:#475569;margin-bottom:0.3rem;">Index (100 = historical avg)</div>'
        if delta_str:
            idx_html += f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.75rem;color:rgba(148,163,184,1);">{delta_str}</div>'

        st.markdown(f"""
        <div class="regime-card {css}">
            <div class="regime-label">{label}</div>
            <div class="regime-value {css}">{display_lbl}</div>
            {idx_html}
            <div class="regime-desc" style="margin-top:0.6rem;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Section 2 — Historical Cost Index                                  #
# ------------------------------------------------------------------ #

if not indices.empty:
    st.markdown('<p class="section-title">Input Cost Index — Historical Trend</p>', unsafe_allow_html=True)
    st.markdown(
        "<p class='callout'>"
        "Each series is normalised so <strong>100 = its own long-run average</strong>. "
        "Values above 110 indicate an above-average cost environment; below 90 a below-average one. "
        "Both series are directly comparable on this scale."
        "</p>",
        unsafe_allow_html=True
    )

    fig_idx = go.Figure()
    colors  = {
        'energy_input_cost_index': 'rgba(245,158,11,1)',
        'fertilizer_cost_index':   'rgba(34,197,94,1)',
    }
    data_cols = [c for c in indices.columns if c != 'date']

    for col in data_cols:
        label = col.replace('_cost_index', '').replace('_', ' ').title()
        fig_idx.add_trace(go.Scatter(
            x=indices['date'], y=indices[col],
            mode='lines', name=label,
            line=dict(width=2.5, color=colors.get(col, 'rgba(148,163,184,1)')),
            hovertemplate=f'<b>{label}</b><br>%{{x|%b %Y}}<br>Index: %{{y:.1f}}<extra></extra>'
        ))

    for y_val, color, ann in [
        (110, 'rgba(239,68,68,0.5)',  'High cost (110)'),
        (100, 'rgba(51,65,85,1)',     'Average (100)'),
        (90,  'rgba(34,197,94,0.5)', 'Low cost (90)'),
    ]:
        fig_idx.add_hline(
            y=y_val, line_dash='dot', line_color=color, line_width=1.5,
            annotation_text=ann,
            annotation_font=dict(size=9, family='IBM Plex Mono', color=color)
        )

    fig_idx.update_layout(
        **CHART_THEME, height=320,
        margin=dict(l=0, r=0, t=4, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08, x=0,
                    font=dict(size=10, color='rgba(148,163,184,1)'),
                    bgcolor='rgba(0,0,0,0)'),
        yaxis_title="Index (100 = avg)"
    )
    st.plotly_chart(fig_idx, use_container_width=True, config={'displayModeBar': False})

# ------------------------------------------------------------------ #
# Section 3 — Margin Pressure by Crop (CEO-facing table)            #
# ------------------------------------------------------------------ #

st.markdown('<p class="section-title">Margin Pressure by Crop</p>', unsafe_allow_html=True)
st.markdown(
    "<p class='callout'>"
    "For each crop, the table shows whether current input costs are compressing margins "
    "relative to historical norms. "
    "<strong>Status</strong> is derived from how far the crop-to-input ratio deviates from its own history. "
    "The trend arrow shows the direction over the last 3 periods."
    "</p>",
    unsafe_allow_html=True
)

if not profitability.empty and not zscores.empty:
    # Identify unique crops from profitability columns
    prof_cols = [c for c in profitability.columns if c != 'date']
    crops_seen = sorted(set(c.split('_to_')[0] for c in prof_cols))

    # Build summary rows: one row per crop, worst z-score across all its input pairs
    rows_html = ""
    summary_data = []

    for crop in crops_seen:
        crop_pairs = [c for c in prof_cols if c.startswith(f'{crop}_to_')]
        if not crop_pairs:
            continue

        # Find the z-score column names (they end in _zscore in the zscores df)
        z_cols = [f'{p}_zscore' for p in crop_pairs if f'{p}_zscore' in zscores.columns]
        if not z_cols:
            continue

        # Latest z-score for each pair; pick the most negative (worst margin pressure)
        latest_zs = {col.replace('_zscore', '').replace(f'{crop}_to_', ''): zscores[col].dropna().iloc[-1]
                     for col in z_cols if not zscores[col].dropna().empty}

        if not latest_zs:
            continue

        worst_input = min(latest_zs, key=latest_zs.get)
        worst_z     = latest_zs[worst_input]

        # Trend based on the worst pair ratio itself
        worst_pair_col = f'{crop}_to_{worst_input}'
        trend_html = _trend_arrow(profitability[worst_pair_col]) if worst_pair_col in profitability.columns else '—'

        badge_cls, status_txt = _margin_status(worst_z)

        summary_data.append({
            'crop':       crop,
            'status_txt': status_txt,
            'badge_cls':  badge_cls,
            'worst_z':    worst_z,
            'worst_input': worst_input,
            'trend_html': trend_html,
            'all_zs':     latest_zs,
        })

    # Sort: worst margin pressure first (most negative z first)
    summary_data.sort(key=lambda r: r['worst_z'])

    for r in summary_data:
        other_inputs = ', '.join(
            f"{_input_label(k)} ({v:+.1f}σ)"
            for k, v in sorted(r['all_zs'].items(), key=lambda x: x[1])
            if k != r['worst_input']
        )
        detail_str = f"Worst driver: {_input_label(r['worst_input'])} ({r['worst_z']:+.1f}σ)"
        if other_inputs:
            detail_str += f"<br><span style='font-size:0.72rem;color:#475569;'>{other_inputs}</span>"

        rows_html += f"""
        <tr>
            <td><strong>{_crop_label(r['crop'])}</strong></td>
            <td><span class="badge {r['badge_cls']}">{r['status_txt']}</span></td>
            <td style="text-align:center;">{r['trend_html']}</td>
            <td style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#94a3b8;">{detail_str}</td>
        </tr>
        """

    if rows_html:
        st.markdown(f"""
        <table class="pressure-table">
            <thead>
                <tr>
                    <th>Crop</th>
                    <th>Margin Status</th>
                    <th style="text-align:center;">Trend</th>
                    <th>Key Driver</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """, unsafe_allow_html=True)
    else:
        st.info("No margin data available — not enough historical data to compute z-scores.")

else:
    st.info("Profitability data not yet available. Ensure the pipeline has run with crop and input commodities configured.")

# ------------------------------------------------------------------ #
# Section 4 — Technical detail (collapsed by default)               #
# ------------------------------------------------------------------ #

with st.expander("Technical Detail — Z-Scores & Ratio Explorer", expanded=False):
    st.markdown(
        "<p class='callout'>"
        "Z-scores measure how far each ratio deviates from its own rolling historical average. "
        "<strong>+z: crop price elevated relative to input cost (favourable margins).</strong> "
        "<strong>−z: input cost elevated relative to crop price (margin pressure).</strong> "
        "The ratio value itself is not comparable across pairs due to different units."
        "</p>",
        unsafe_allow_html=True
    )

    # ── Z-score bar chart ──────────────────────────────────────────────
    prof_cols_z = [c for c in profitability.columns if c != 'date']
    z_cols_avail = [f'{c}_zscore' for c in prof_cols_z if f'{c}_zscore' in zscores.columns]

    if z_cols_avail:
        latest_zs_all = {col.replace('_zscore', ''): zscores[col].dropna().iloc[-1]
                         for col in z_cols_avail if not zscores[col].dropna().empty}

        if latest_zs_all:
            sorted_pairs = sorted(latest_zs_all.items(), key=lambda x: x[1])
            labels  = [k.replace('_to_', ' / ').replace('_', ' ').title() for k, _ in sorted_pairs]
            z_vals  = [v for _, v in sorted_pairs]
            colors_z = ['rgba(239,68,68,1)' if abs(z) > 2 else
                        'rgba(245,158,11,1)' if abs(z) > 1 else
                        '#22477a' for z in z_vals]

            fig_z = go.Figure()
            fig_z.add_trace(go.Bar(
                x=labels, y=z_vals,
                marker_color=colors_z,
                hovertemplate='<b>%{x}</b><br>Z-Score: %{y:+.2f}σ<extra></extra>'
            ))
            for y_val, clr in [(2, 'rgba(239,68,68,0.5)'), (-2, 'rgba(239,68,68,0.5)'),
                               (1, 'rgba(245,158,11,0.5)'), (-1, 'rgba(245,158,11,0.5)')]:
                fig_z.add_hline(y=y_val, line_dash='dot', line_color=clr, line_width=1, opacity=0.7)
            fig_z.add_hline(y=0, line_color='#1e2330', line_width=1)

            chart_no_x = {k: v for k, v in CHART_THEME.items() if k != 'xaxis'}
            fig_z.update_layout(
                **chart_no_x, height=320,
                margin=dict(l=0, r=0, t=4, b=0),
                xaxis={**CHART_THEME['xaxis'], 'tickangle': -35, 'tickfont': dict(size=8)},
                yaxis_title='Z-Score (σ)',
                showlegend=False
            )
            st.plotly_chart(fig_z, use_container_width=True, config={'displayModeBar': False})
            st.caption("|z| > 2 = extreme   |z| > 1 = notable   normal range")

    st.markdown('<p class="section-title" style="margin-top:1.5rem;">Ratio Explorer</p>',
                unsafe_allow_html=True)

    if prof_cols_z:
        selected = st.selectbox(
            "Select crop / input pair",
            prof_cols_z,
            format_func=lambda x: x.replace('_to_', ' / ').replace('_', ' ').title()
        )

        if selected and selected in profitability.columns:
            series   = profitability[selected].dropna()
            mean_val = series.mean()
            std_val  = series.std()
            cur_val  = series.iloc[-1] if not series.empty else np.nan
            z_val    = (cur_val - mean_val) / std_val if std_val > 0 else 0
            pctile   = (series < cur_val).mean() * 100 if not series.empty else np.nan

            ch1, ch2 = st.columns([3, 1])

            with ch1:
                fig_r = go.Figure()
                fig_r.add_trace(go.Scatter(
                    x=profitability['date'], y=profitability[selected],
                    mode='lines', name='Ratio',
                    line=dict(width=2, color='rgba(56,189,248,1)'),
                    hovertemplate='%{x|%b %Y}<br>Ratio: %{y:.3f}<extra></extra>'
                ))
                fig_r.add_hline(
                    y=mean_val, line_dash='dash', line_color='#475569', line_width=1.5,
                    annotation_text=f"Mean {mean_val:.3f}",
                    annotation_font=dict(size=9, family='IBM Plex Mono', color='#64748b')
                )
                if std_val > 0:
                    fig_r.add_hrect(y0=mean_val - std_val,     y1=mean_val + std_val,
                                    fillcolor='rgba(34,197,94,1)', opacity=0.04, line_width=0)
                    fig_r.add_hrect(y0=mean_val - 2 * std_val, y1=mean_val + 2 * std_val,
                                    fillcolor='rgba(245,158,11,1)', opacity=0.03, line_width=0)

                chart_no_x = {k: v for k, v in CHART_THEME.items() if k != 'xaxis'}
                fig_r.update_layout(
                    **chart_no_x, height=300,
                    margin=dict(l=0, r=0, t=4, b=0),
                    showlegend=False,
                    yaxis_title="Ratio (trend only — units not comparable across pairs)",
                    xaxis={**CHART_THEME['xaxis'],
                           'rangeslider': dict(visible=True, thickness=0.04, bgcolor='#141720')}
                )
                st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})

            with ch2:
                level   = 'extreme' if abs(z_val) > 2 else 'notable' if abs(z_val) > 1 else 'normal'
                z_color = {'extreme': 'rgba(239,68,68,1)', 'notable': 'rgba(245,158,11,1)',
                           'normal': 'rgba(34,197,94,1)'}[level]

                pctile_str = f"{pctile:.0f}th" if pd.notna(pctile) else "—"

                if   pctile > 85: interp = "Crop price historically strong vs. input cost."
                elif pctile < 15: interp = "Input cost historically elevated vs. crop price — margin pressure."
                else:             interp = "Within normal historical range."

                st.markdown(f"""
                <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;
                            padding:1rem 1.25rem;margin-bottom:0.75rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                color:#64748b;text-transform:uppercase;margin-bottom:0.3rem;">Z-Score</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:1.6rem;
                                font-weight:600;color:{z_color};">{z_val:+.2f}σ</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.75rem;
                                color:#475569;margin-top:0.3rem;">Percentile: {pctile_str}</div>
                </div>
                <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;padding:1rem 1.25rem;">
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                                color:#64748b;text-transform:uppercase;margin-bottom:0.5rem;">Statistics</div>
                    <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                                color:rgba(148,163,184,1);line-height:1.8;">
                        Mean: {mean_val:.3f}<br>
                        Std:&nbsp; {std_val:.3f}<br>
                        Min:&nbsp; {series.min():.3f}<br>
                        Max:&nbsp; {series.max():.3f}
                    </div>
                </div>
                <p style="font-family:'IBM Plex Mono',monospace;font-size:0.74rem;
                          color:#64748b;margin-top:0.75rem;line-height:1.5;">{interp}</p>
                """, unsafe_allow_html=True)
    else:
        st.info("No profitability ratio data available.")