"""
Page 2 — Analysis
"Why is it happening? What does history say?"
Price trends with context: moving averages, percentile position, correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from pipeline.data_pipeline import DataPipeline
from utils.config_loader import ConfigLoader

st.set_page_config(
    page_title="Analysis — Commodity Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS (shared design system with Overview) ──────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: rgba(13, 15, 18, 1);
    color: #e2e8f0;
}
.stApp { background-color: rgba(13, 15, 18, 1); }

.page-header { border-bottom:1px solid #2a2f3a; padding-bottom:1.2rem; margin-bottom:2rem; }
.page-title  { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600;
               color:rgba(148, 163, 184, 1); letter-spacing:0.12em; text-transform:uppercase; margin:0; }
.page-subtitle { font-size:2rem; font-weight:600; color:#f1f5f9; margin:0.25rem 0 0 0; line-height:1.2; }

.section-title { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#475569;
                 letter-spacing:0.12em; text-transform:uppercase; border-bottom:1px solid #1e2330;
                 padding-bottom:0.5rem; margin:2rem 0 1rem 0; }

.stat-card { background:#141720; border:1px solid #1e2330; border-radius:8px;
             padding:1rem 1.25rem; }
.stat-label { font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#475569;
              letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem; }
.stat-value { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; font-weight:600; color:#f1f5f9; }
.stat-sub   { font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:#64748b; margin-top:0.2rem; }

.percentile-bar-wrap { background:#1e2330; border-radius:4px; height:6px;
                        overflow:hidden; margin-top:0.5rem; }
.percentile-bar      { height:100%; border-radius:4px; }

div[data-testid="stMetricValue"] { font-family:'IBM Plex Mono',monospace !important; color:#f1f5f9 !important; }
div[data-testid="stMetricLabel"] { font-family:'IBM Plex Mono',monospace !important;
                                    font-size:0.68rem !important; color:#475569 !important;
                                    text-transform:uppercase; letter-spacing:0.08em; }
.stSelectbox label { font-family:'IBM Plex Mono',monospace !important; font-size:0.72rem !important;
                     color:#475569 !important; text-transform:uppercase; letter-spacing:0.08em; }
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

df = load_data()

if df.empty:
    st.error("No data found. Run `python automation/run_daily.py` first.")
    st.stop()

# Exclude derived columns; retain only raw price series
base_cols = [
    c for c in df.columns
    if c != 'date' and not c.endswith(('_change_pct', '_ma', '_zscore', '_signal'))
]
categories = ConfigLoader.get_categories()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-title">Agricultural Commodity Tracker</p>
  <p class="page-subtitle">Price Analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.markdown("### Controls")

# Build category filter options
cat_options = {'All': base_cols}
for cat, items in categories.items():
    filtered = [c for c in items if c in base_cols]
    if filtered:
        cat_options[cat.replace('_', ' ').title()] = filtered

selected_cat = st.sidebar.selectbox("Category", list(cat_options.keys()))
pool = cat_options[selected_cat]

# Commodity multi-select within the chosen category
selected = st.sidebar.multiselect(
    "Commodities",
    pool,
    default=pool[:min(3, len(pool))],
    format_func=lambda x: ConfigLoader.get_commodity_info(x).get('name', x.replace('_',' ').title())
)

st.sidebar.markdown("---")
normalize  = st.sidebar.checkbox("Normalize to base 100", value=False)
show_ma    = st.sidebar.checkbox("Show moving average", value=True)
show_z     = st.sidebar.checkbox("Show z-score panel", value=True)
time_range = st.sidebar.select_slider(
    "Time window",
    options=['3M', '6M', '1Y', '2Y', 'All'],
    value='2Y'
)

# Filter the dataset to the selected time window
cutoff_map = {'3M': 90, '6M': 180, '1Y': 365, '2Y': 730, 'All': 99999}
cutoff_days = cutoff_map[time_range]
df_view = df[df['date'] >= df['date'].max() - pd.Timedelta(days=cutoff_days)].copy()

if not selected:
    st.info("Select at least one commodity from the sidebar.")
    st.stop()

latest = df_view.iloc[-1]

# ── Current Values strip ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">Current Values</p>', unsafe_allow_html=True)

cols = st.columns(min(len(selected), 5))
for i, c in enumerate(selected[:5]):
    with cols[i]:
        info  = ConfigLoader.get_commodity_info(c)
        cur   = latest[c]
        z_col = f"{c}_zscore"
        z     = latest[z_col] if z_col in df_view.columns else None
        pctile = (df[c].dropna() < cur).mean() * 100

        # Colour-code the percentile bar by historical position
        if   pctile > 80: bar_color = 'rgba(239, 68, 68, 1)'
        elif pctile < 20: bar_color = 'rgba(34, 197, 94, 1)'
        else:             bar_color = 'rgba(245, 158, 11, 1)'

        z_str = f"{z:+.2f}σ" if z is not None and pd.notna(z) else "—"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">{info.get('name', c.replace('_',' ').title())}</div>
            <div class="stat-value">{cur:.2f}</div>
            <div class="stat-sub">{info.get('unit','')}</div>
            <div class="stat-sub" style="margin-top:0.4rem;">
                Percentile: <strong style="color:{bar_color}">{pctile:.0f}th</strong>
                &nbsp;·&nbsp; Z: <strong>{z_str}</strong>
            </div>
            <div class="percentile-bar-wrap">
                <div class="percentile-bar"
                     style="width:{pctile:.0f}%;background:{bar_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Price trend chart ─────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Price Trends</p>', unsafe_allow_html=True)

ACCENT_COLORS = ['rgba(245, 158, 11, 1)', 'rgba(34, 197, 94, 1)', 'rgba(56, 189, 248, 1)', '#a78bfa', '#f87171', '#34d399']

fig = go.Figure()

for idx, c in enumerate(selected):
    info  = ConfigLoader.get_commodity_info(c)
    color = ACCENT_COLORS[idx % len(ACCENT_COLORS)]
    name  = info.get('name', c.replace('_', ' ').title())

    # Rebase to 100 at the start of the window if normalisation is active
    y = (df_view[c] / df_view[c].dropna().iloc[0] * 100) if normalize else df_view[c]

    fig.add_trace(go.Scatter(
        x=df_view['date'], y=y,
        mode='lines', name=name,
        line=dict(width=2, color=color),
        hovertemplate=f'<b>{name}</b><br>%{{x|%d %b %Y}}<br>%{{y:.2f}}<extra></extra>'
    ))

    if show_ma:
        ma_col = f"{c}_ma"
        if ma_col in df_view.columns:
            y_ma = (df_view[ma_col] / df_view[c].dropna().iloc[0] * 100) if normalize else df_view[ma_col]
            fig.add_trace(go.Scatter(
                x=df_view['date'], y=y_ma,
                mode='lines', name=f"{name} MA",
                line=dict(width=1, color=color, dash='dot'),
                opacity=0.45,
                hoverinfo='skip', showlegend=False
            ))

fig.update_layout(
    **CHART_THEME,
    height=420,
    margin=dict(l=0, r=0, t=12, b=0),
    hovermode='x unified',
    legend=dict(
        orientation='h', y=1.06, x=0,
        bgcolor='rgba(0,0,0,0)',
        font=dict(size=10, color='rgba(148, 163, 184, 1)')
    ),
    yaxis_title="Index (base 100)" if normalize else "Price",
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

if normalize:
    st.caption("All series rebased to 100 at the start of the selected window.")

# ── Z-Score panel (single commodity only) ────────────────────────────────────
if show_z and len(selected) == 1:
    c = selected[0]
    z_col = f"{c}_zscore"
    if z_col in df_view.columns:
        st.markdown('<p class="section-title">Z-Score — Statistical Context</p>',
                    unsafe_allow_html=True)

        fig_z = go.Figure()
        z_series = df_view[z_col]

        # Colour bars by deviation threshold
        colors_z = ['rgba(239, 68, 68, 1)' if abs(z) > 2 else 'rgba(245, 158, 11, 1)' if abs(z) > 1 else 'rgba(51, 65, 85, 1)'
                    for z in z_series]

        fig_z.add_trace(go.Bar(
            x=df_view['date'], y=z_series,
            marker_color=colors_z, name='Z-Score',
            hovertemplate='%{x|%d %b %Y}<br>Z: %{y:.2f}σ<extra></extra>'
        ))

        # Reference lines at ±1σ and ±2σ
        for y_val, color in [(2,'rgba(239, 68, 68, 1)30'), (-2,'rgba(239, 68, 68, 1)30'), (1,'rgba(245, 158, 11, 1)20'), (-1,'rgba(245, 158, 11, 1)20')]:
            fig_z.add_hline(y=y_val, line_dash='dot', line_color=color, line_width=1)
        fig_z.add_hline(y=0, line_color='#1e2330', line_width=1)

        fig_z.update_layout(
            **CHART_THEME,
            height=180,
            margin=dict(l=0, r=0, t=4, b=0),
            showlegend=False,
            yaxis_title="σ"
        )
        st.plotly_chart(fig_z, use_container_width=True, config={'displayModeBar': False})
        st.caption("|z| > 2 = extreme  |  |z| > 1 = notable  |  Z computed on a rolling window adapted to data frequency.")

# ── Historical price distribution ─────────────────────────────────────────────
if len(selected) == 1:
    c = selected[0]
    info = ConfigLoader.get_commodity_info(c)
    name = info.get('name', c.replace('_', ' ').title())
    cur  = latest[c]

    st.markdown('<p class="section-title">Historical Distribution</p>', unsafe_allow_html=True)

    series = df[c].dropna()
    pctile = (series < cur).mean() * 100

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=series,
        nbinsx=40,
        marker_color='#1e2f3e',
        marker_line=dict(color='rgba(56, 189, 248, 1)30', width=0.5),
        name='Historical'
    ))
    # Current price marker
    fig_hist.add_vline(
        x=cur, line_color='rgba(245, 158, 11, 1)', line_width=2,
        annotation_text=f"Now ({pctile:.0f}th pct.)",
        annotation_font=dict(color='rgba(245, 158, 11, 1)', size=10, family='IBM Plex Mono'),
        annotation_position="top right"
    )
    # Median reference line
    fig_hist.add_vline(
        x=series.median(), line_color='rgba(51, 65, 85, 1)', line_dash='dot', line_width=1,
        annotation_text="Median",
        annotation_font=dict(color='#475569', size=9, family='IBM Plex Mono'),
        annotation_position="top left"
    )

    fig_hist.update_layout(
        **CHART_THEME,
        height=220,
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        xaxis_title=info.get('unit', 'Price'),
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})

    # Contextual callout based on percentile position
    if   pctile > 85: msg = f"Currently at the **{pctile:.0f}th percentile** — historically elevated. Above 85% of all observations since {df['date'].min().year}."
    elif pctile < 15: msg = f"Currently at the **{pctile:.0f}th percentile** — historically low. Below 85% of all observations since {df['date'].min().year}."
    else:             msg = f"Currently at the **{pctile:.0f}th percentile** — within normal historical range."
    st.markdown(
        f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.82rem;"
        f"color:rgba(148, 163, 184, 1);padding:0.75rem 1rem;background:#141720;"
        f"border-radius:6px;border-left:3px solid rgba(51, 65, 85, 1);'>{msg}</p>",
        unsafe_allow_html=True
    )

# ── Correlation matrix (two or more commodities selected) ────────────────────
if len(selected) >= 2:
    st.markdown('<p class="section-title">Correlations</p>', unsafe_allow_html=True)

    df_filled = df[selected].ffill()
    corr = df_filled.corr()
    labels = [ConfigLoader.get_commodity_info(c).get('name', c.replace('_',' ').title())
              for c in selected]

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=labels, y=labels,
        colorscale=[[0,'#1c2e3d'],[0.5,'#1e2330'],[1,'#1e3a2f']],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont=dict(size=9, family='IBM Plex Mono', color='rgba(148, 163, 184, 1)'),
        hovertemplate='%{y} x %{x}<br>r = %{z:.3f}<extra></extra>',
        colorbar=dict(
            tickfont=dict(size=8, family='IBM Plex Mono', color='#475569'),
            outlinecolor='#1e2330', thickness=10
        )
    ))

    fig_corr.update_layout(
        **CHART_THEME,
        height=max(260, len(selected) * 60),
        margin=dict(l=0, r=0, t=4, b=0),
    )

    fig_corr.update_xaxes(
        tickfont=dict(size=9, family='IBM Plex Mono', color='#64748b')
    )

    fig_corr.update_yaxes(
        tickfont=dict(size=9, family='IBM Plex Mono', color='#64748b')
    )
    st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})

    # Rank all unique pairs by absolute correlation
    pairs = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            pairs.append((labels[i], labels[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    if pairs:
        pc1, pc2 = st.columns(2)
        with pc1:
            st.markdown('<p class="section-title" style="margin-top:0">Strongest Positive</p>',
                        unsafe_allow_html=True)
            for a, b, r in [p for p in pairs if p[2] > 0][:3]:
                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:rgba(148, 163, 184, 1);"
                    f"margin:0.3rem 0'>{a} x {b} "
                    f"<span style='color:rgba(34, 197, 94, 1);font-weight:600'>{r:+.3f}</span></p>",
                    unsafe_allow_html=True)
        with pc2:
            st.markdown('<p class="section-title" style="margin-top:0">Strongest Negative</p>',
                        unsafe_allow_html=True)
            for a, b, r in [p for p in reversed(pairs) if p[2] < 0][:3]:
                st.markdown(
                    f"<p style='font-family:IBM Plex Mono,monospace;font-size:0.8rem;color:rgba(148, 163, 184, 1);"
                    f"margin:0.3rem 0'>{a} x {b} "
                    f"<span style='color:rgba(239, 68, 68, 1);font-weight:600'>{r:+.3f}</span></p>",
                    unsafe_allow_html=True)

# ── Descriptive statistics table ──────────────────────────────────────────────
st.markdown('<p class="section-title">Descriptive Statistics</p>', unsafe_allow_html=True)

stat_rows = []
for c in selected:
    info   = ConfigLoader.get_commodity_info(c)
    series = df_view[c].dropna()
    cur    = latest[c]
    pctile = (df[c].dropna() < cur).mean() * 100
    stat_rows.append({
        'Commodity': info.get('name', c.replace('_', ' ').title()),
        'Freq':      info.get('frequency', '—').title()[:3],
        'Current':   round(cur, 2),
        'Pctile':    round(pctile, 1),
        'Mean':      round(series.mean(), 2),
        'Std':       round(series.std(), 2),
        'Min':       round(series.min(), 2),
        'Max':       round(series.max(), 2),
    })

stats_df = pd.DataFrame(stat_rows)

def style_pctile(val):
    if   val > 80: return 'color:rgba(239, 68, 68, 1);font-weight:600'
    elif val < 20: return 'color:rgba(34, 197, 94, 1);font-weight:600'
    return 'color:rgba(245, 158, 11, 1)'

st.dataframe(
    stats_df.style
        .map(style_pctile, subset=['Pctile'])
        .format({'Current': '{:.2f}', 'Mean': '{:.2f}', 'Std': '{:.2f}',
                 'Min': '{:.2f}', 'Max': '{:.2f}', 'Pctile': '{:.1f}%'}),
    hide_index=True,
    use_container_width=True
)