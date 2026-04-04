"""
Page 3 — Analytics
"What does it cost to produce? Are margins under pressure?"
Cost indices, profitability ratios, z-score signals. No misleading absolute values.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os, json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)
from pipeline.data_pipeline import DataPipeline
from utils.config_loader import ConfigLoader

st.set_page_config(
    page_title="Analytics — Commodity Tracker",
    page_icon="📊",
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

.regime-card { background:#141720; border:1px solid #1e2330; border-radius:8px;
               padding:1.25rem 1.5rem; position:relative; overflow:hidden; }
.regime-card::before { content:''; position:absolute; top:0;left:0;right:0; height:3px; }
.regime-card.high::before   { background:rgba(239, 68, 68, 1); }
.regime-card.normal::before { background:rgba(245, 158, 11, 1); }
.regime-card.low::before    { background:rgba(34, 197, 94, 1); }
.regime-label { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#64748b;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.4rem; }
.regime-value { font-size:1.3rem; font-weight:600; margin-bottom:0.2rem; }
.regime-value.high   { color:rgba(239, 68, 68, 1); }
.regime-value.normal { color:rgba(245, 158, 11, 1); }
.regime-value.low    { color:rgba(34, 197, 94, 1); }
.regime-desc { font-size:0.78rem; color:#64748b; line-height:1.4; }

.ratio-card { background:#141720; border:1px solid #1e2330; border-radius:8px;
              padding:1rem 1.25rem; }
.ratio-name { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b;
              text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.3rem; }
.ratio-z    { font-family:'IBM Plex Mono',monospace; font-size:1.6rem; font-weight:600;
              line-height:1; }
.ratio-z.extreme { color:rgba(239, 68, 68, 1); }
.ratio-z.notable { color:rgba(245, 158, 11, 1); }
.ratio-z.normal  { color:rgba(34, 197, 94, 1); }
.ratio-pctile { font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                color:#475569; margin-top:0.3rem; }

.callout { background:#141720; border-radius:6px; border-left:3px solid rgba(51, 65, 85, 1);
           padding:0.75rem 1rem;
           font-family:'IBM Plex Mono',monospace; font-size:0.8rem;
           color:rgba(148, 163, 184, 1); line-height:1.5; margin-bottom:0.5rem; }

div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; color:#f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important;
                                    font-size:0.68rem !important; color:#475569 !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
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
def load_ratios():
    path = os.path.join('data', 'commodity_ratios.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data(ttl=3600)
def load_signals():
    path = os.path.join('data', 'signals.json')
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        d = json.load(f)
    return d.get('signals', []), d.get('regime', {})

ratios_df       = load_ratios()
signals, regime = load_signals()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-title">Agricultural Commodity Tracker</p>
  <p class="page-subtitle">Input Cost Analytics</p>
</div>
""", unsafe_allow_html=True)

if ratios_df.empty:
    st.warning("No ratio data found. Run `python automation/run_daily.py` first.")
    st.stop()

# Separate ratio columns into indices and crop-to-input profitability ratios
ratio_cols         = [c for c in ratios_df.columns if c != 'date']
index_cols         = [c for c in ratio_cols if 'cost_index' in c]
profitability_cols = [c for c in ratio_cols if '_to_' in c]
latest             = ratios_df.iloc[-1]
means              = ratios_df[ratio_cols].mean()
stds               = ratios_df[ratio_cols].std().replace(0, np.nan)
z_now              = (ratios_df[ratio_cols].iloc[-1] - means) / stds

# ── Cost regime block ─────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Current Cost Regime</p>', unsafe_allow_html=True)

st.markdown(
    "<p class='callout'>The regime classifies the current input cost environment relative to "
    "its own history. <strong>100 = historical average.</strong> "
    "A high-cost regime indicates input costs above their long-run mean, "
    "which typically translates to margin pressure for producers.</p>",
    unsafe_allow_html=True
)

regime_meta = {
    'high_cost':  ('high',   'HIGH COST',  'Energy inputs elevated — above historical average'),
    'low_cost':   ('low',    'LOW COST',   'Energy inputs below historical average — favourable conditions'),
    'normal':     ('normal', 'NORMAL',     'Energy within historical range'),
    'expensive':  ('high',   'EXPENSIVE',  'Fertilizer above historical average'),
    'cheap':      ('low',    'CHEAP',      'Fertilizer below historical average'),
}

rc1, rc2 = st.columns(2)

for col_w, key, label_key in [(rc1, 'energy', 'energy'), (rc2, 'fertilizer', 'fert')]:
    val = regime.get(key, 'normal')
    cls, lbl, desc = regime_meta.get(val, ('normal', val.upper(), ''))

    # Retrieve the corresponding index value for display
    idx_col = 'energy_input_cost_index' if key == 'energy' else 'fertilizer_cost_index'
    idx_val = latest.get(idx_col, None)
    idx_str = f"Index: {idx_val:.1f}" if idx_val is not None and pd.notna(idx_val) else ""

    with col_w:
        st.markdown(f"""
        <div class="regime-card {cls}">
            <div class="regime-label">{'Energy Inputs' if key=='energy' else 'Fertilizers'}</div>
            <div class="regime-value {cls}">{lbl}</div>
            <div class="regime-desc">{desc}</div>
            {"<div style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;color:rgba(148, 163, 184, 1);margin-top:0.5rem;'>" + idx_str + "</div>" if idx_str else ""}
        </div>
        """, unsafe_allow_html=True)

# ── Cost index history chart ──────────────────────────────────────────────────
if index_cols:
    st.markdown('<p class="section-title">Input Cost Indices — History</p>',
                unsafe_allow_html=True)
    st.markdown(
        "<p class='callout'>These values are directly comparable: each series is normalised "
        "so 100 = its own historical average. The absolute number is meaningful "
        "only within this section of the page.</p>",
        unsafe_allow_html=True
    )

    fig_idx = go.Figure()
    colors  = {'energy_input_cost_index': 'rgba(245, 158, 11, 1)', 'fertilizer_cost_index': 'rgba(34, 197, 94, 1)'}

    for col in index_cols:
        label = col.replace('_cost_index','').replace('_',' ').title()
        fig_idx.add_trace(go.Scatter(
            x=ratios_df['date'], y=ratios_df[col],
            mode='lines', name=label,
            line=dict(width=2.5, color=colors.get(col, 'rgba(148, 163, 184, 1)')),
            hovertemplate=f'<b>{label}</b><br>%{{x|%d %b %Y}}<br>Index: %{{y:.1f}}<extra></extra>'
        ))

    # Reference bands at the historical average, high, and low thresholds
    for y_val, color, label in [
        (100, 'rgba(51, 65, 85, 1)', 'Average (100)'),
        (110, 'rgba(239, 68, 68, 0.3)', 'High (110)'),
        (90,  'rgba(34, 197, 94, 0.3)', 'Low (90)')
    ]:
        fig_idx.add_hline(y=y_val, line_dash='dot', line_color=color,
                          line_width=1.5, annotation_text=label,
                          annotation_font=dict(size=9, family='IBM Plex Mono',
                                               color=color.replace('50','')))

    fig_idx.update_layout(
        **CHART_THEME, height=320,
        margin=dict(l=0, r=0, t=4, b=0),
        hovermode='x unified',
        legend=dict(orientation='h', y=1.08, x=0,
                    font=dict(size=10, color='rgba(148, 163, 184, 1)'),
                    bgcolor='rgba(0,0,0,0)'),
        yaxis_title="Index (100 = avg)"
    )
    st.plotly_chart(fig_idx, use_container_width=True, config={'displayModeBar': False})

# ── Profitability z-score overview ────────────────────────────────────────────
if profitability_cols:
    st.markdown('<p class="section-title">Crop Profitability — Z-Score Deviation</p>',
                unsafe_allow_html=True)
    st.markdown(
        "<p class='callout'>"
        "Each ratio represents <strong>crop price / input cost</strong>. "
        "The absolute value is not comparable across ratios due to differing units. "
        "The relevant signal is the deviation from each ratio's own historical norm. "
        "<strong>Positive z: crop price elevated relative to input cost</strong> (favourable margins). "
        "<strong>Negative z: input cost elevated relative to crop price</strong> (margin pressure)."
        "</p>",
        unsafe_allow_html=True
    )

    z_vals   = z_now[profitability_cols].fillna(0)
    colors_z = ['rgba(239, 68, 68, 1)' if abs(z) > 2 else 'rgba(245, 158, 11, 1)' if abs(z) > 1 else '#22477a'
                for z in z_vals.values]
    labels_z = [c.replace('_to_', ' / ').replace('_', ' ').title()
                for c in profitability_cols]

    fig_z = go.Figure()
    fig_z.add_trace(go.Bar(
        x=labels_z, y=z_vals.values,
        marker_color=colors_z,
        hovertemplate='<b>%{x}</b><br>Z-Score: %{y:+.2f}σ<extra></extra>'
    ))

    for y_val, clr in [(2,'rgba(239, 68, 68, 0.6)'),(-2,'rgba(239, 68, 68, 0.6)'),(1,'rgba(245, 158, 11, 0.6)'),(-1,'rgba(245, 158, 11, 0.6)')]:
        fig_z.add_hline(y=y_val, line_dash='dot', line_color=clr.replace('15','60').replace('20','60'),
                        line_width=1, opacity=0.6)
    fig_z.add_hline(y=0, line_color='#1e2330', line_width=1)

    chart_theme_no_xaxis = {k: v for k, v in CHART_THEME.items() if k != 'xaxis'}

    fig_z.update_layout(
        **chart_theme_no_xaxis,
        height=320,
        margin=dict(l=0, r=0, t=4, b=0),
        xaxis={**CHART_THEME['xaxis'], "tickangle": -35, "tickfont": dict(size=8)},
        yaxis_title="Z-Score (σ)",
        showlegend=False
    )
    st.plotly_chart(fig_z, use_container_width=True, config={'displayModeBar': False})
    st.caption("|z| > 2 = extreme  |  |z| > 1 = notable  |  Normal range  |  Rolling 60-day window")

    # ── Flagged ratio cards (only ratios with |z| > 1) ────────────────────────
    notable = [(c, float(z_now[c])) for c in profitability_cols
               if pd.notna(z_now[c]) and abs(z_now[c]) > 1]
    notable.sort(key=lambda x: abs(x[1]), reverse=True)

    if notable:
        st.markdown('<p class="section-title">Flagged Ratios</p>', unsafe_allow_html=True)
        cols = st.columns(min(len(notable), 4))
        for i, (c, z) in enumerate(notable[:4]):
            pctile = (ratios_df[c].dropna() < latest[c]).mean() * 100
            level  = 'extreme' if abs(z) > 2 else 'notable'
            lbl    = c.replace('_to_', ' / ').replace('_', ' ').title()
            with cols[i]:
                st.markdown(f"""
                <div class="ratio-card">
                    <div class="ratio-name">{lbl}</div>
                    <div class="ratio-z {level}">{z:+.2f}σ</div>
                    <div class="ratio-pctile">Percentile: {pctile:.0f}th</div>
                </div>
                """, unsafe_allow_html=True)

    # ── Ratio detail explorer ──────────────────────────────────────────────────
    st.markdown('<p class="section-title">Ratio Explorer</p>', unsafe_allow_html=True)

    selected_ratio = st.selectbox(
        "Select ratio",
        profitability_cols,
        format_func=lambda x: x.replace('_to_', ' / ').replace('_', ' ').title()
    )

    if selected_ratio:
        series   = ratios_df[selected_ratio].dropna()
        mean_val = series.mean()
        std_val  = series.std()
        cur_val  = latest[selected_ratio]
        z_val    = (cur_val - mean_val) / std_val if std_val > 0 else 0
        pctile   = (series < cur_val).mean() * 100

        ch1, ch2 = st.columns([3, 1])

        with ch1:
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(
                x=ratios_df['date'], y=ratios_df[selected_ratio],
                mode='lines', name='Ratio',
                line=dict(width=2, color='rgba(56, 189, 248, 1)'),
                hovertemplate='%{x|%d %b %Y}<br>%{y:.3f}<extra></extra>'
            ))
            # Mean reference line
            fig_r.add_hline(y=mean_val, line_dash='dash', line_color='#475569',
                            line_width=1.5,
                            annotation_text=f"Mean {mean_val:.3f}",
                            annotation_font=dict(size=9, family='IBM Plex Mono', color='#64748b'))
            # 1σ and 2σ shaded bands
            fig_r.add_hrect(y0=mean_val-std_val,   y1=mean_val+std_val,
                            fillcolor='rgba(34, 197, 94, 1)', opacity=0.04, line_width=0)
            fig_r.add_hrect(y0=mean_val-2*std_val, y1=mean_val+2*std_val,
                            fillcolor='rgba(245, 158, 11, 1)', opacity=0.03, line_width=0)
            chart_theme_no_xaxis = {k: v for k, v in CHART_THEME.items() if k != 'xaxis'}

            fig_r.update_layout(
                **chart_theme_no_xaxis,
                height=300,
                margin=dict(l=0, r=0, t=4, b=0),
                showlegend=False,
                yaxis_title="Ratio (trend only)",
                xaxis={**CHART_THEME['xaxis'],
                    "rangeslider": dict(
                        visible=True,
                        thickness=0.04,
                        bgcolor='#141720'
                    )}
            )
            st.plotly_chart(fig_r, use_container_width=True, config={'displayModeBar': False})

        with ch2:
            level = 'extreme' if abs(z_val) > 2 else 'notable' if abs(z_val) > 1 else 'normal'
            z_color = {'extreme': 'rgba(239, 68, 68, 1)', 'notable': 'rgba(245, 158, 11, 1)', 'normal': 'rgba(34, 197, 94, 1)'}[level]

            st.markdown(f"""
            <div class="ratio-card" style="margin-bottom:0.75rem">
                <div class="ratio-name">Z-Score</div>
                <div class="ratio-z {level}">{z_val:+.2f}σ</div>
                <div class="ratio-pctile">Percentile: {pctile:.0f}th</div>
            </div>
            <div class="ratio-card">
                <div class="ratio-name">Statistics</div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
                            color:rgba(148, 163, 184, 1);line-height:1.8;">
                    Mean: {mean_val:.3f}<br>
                    Std:  {std_val:.3f}<br>
                    Min:  {series.min():.3f}<br>
                    Max:  {series.max():.3f}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Contextual interpretation based on historical percentile
            if   pctile > 85: interp = "Historically high — crop price strong relative to input cost."
            elif pctile < 15: interp = "Historically low — input cost elevated relative to crop price. Margin pressure likely."
            else:             interp = "Within normal historical range."

            st.markdown(
                f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.75rem;"
                f"color:#64748b;margin-top:0.75rem;line-height:1.5;'>{interp}</p>",
                unsafe_allow_html=True
            )