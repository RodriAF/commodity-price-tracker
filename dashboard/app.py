"""
Page 1 — Overview
"What is happening right now?"
Entry point: regime, top signals, commodity snapshot.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.data_pipeline import DataPipeline
from utils.config_loader import ConfigLoader

st.set_page_config(
    page_title="Overview — Commodity Tracker",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: rgba(13, 15, 18, 1);
    color: #e2e8f0;
}

.stApp { background-color: rgba(13, 15, 18, 1); }

/* ── Page header ── */
.page-header {
    border-bottom: 1px solid #2a2f3a;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.page-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(148, 163, 184, 1);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}
.page-subtitle {
    font-size: 2rem;
    font-weight: 600;
    color: #f1f5f9;
    margin: 0.25rem 0 0 0;
    line-height: 1.2;
}

/* ── Category cards ── */
.category-card {
    background: #141720;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1.1rem 1.3rem;
    position: relative;
    overflow: hidden;
    height: 100%;
}
.category-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.category-card.energy_input::before  { background: rgba(245, 158, 11, 1); }
.category-card.crop::before          { background: rgba(34, 197, 94, 1); }
.category-card.fertilizer::before    { background: rgba(56, 189, 248, 1); }
.category-card.livestock::before     { background: rgba(239, 68, 68, 1); }
.category-card.index::before         { background: rgba(168, 85, 247, 1); }
.category-card.economic::before      { background: rgba(148, 163, 184, 1); }

.category-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2330;
}
.category-header.energy_input  { color: rgba(245, 158, 11, 1); }
.category-header.crop          { color: rgba(34, 197, 94, 1); }
.category-header.fertilizer    { color: rgba(56, 189, 248, 1); }
.category-header.livestock     { color: rgba(239, 68, 68, 1); }
.category-header.index         { color: rgba(168, 85, 247, 1); }
.category-header.economic      { color: rgba(148, 163, 184, 1); }

.commodity-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    border-bottom: 1px solid #1a1f2e;
}
.commodity-row:last-child { border-bottom: none; }
.commodity-row-name {
    font-size: 0.78rem;
    color: #94a3b8;
    flex: 1;
}
.commodity-row-price {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    color: #f1f5f9;
    text-align: right;
    min-width: 70px;
}
.commodity-row-chg {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    text-align: right;
    min-width: 60px;
    margin-left: 0.5rem;
}
.commodity-row-chg.pos { color: rgba(34, 197, 94, 1); }
.commodity-row-chg.neg { color: rgba(239, 68, 68, 1); }
.commodity-row-chg.neu { color: #64748b; }

/* ── Signal rows ── */
.signal-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin-bottom: 0.5rem;
    background: #141720;
    border-left: 3px solid transparent;
}
.signal-row.extreme { border-left-color: rgba(239, 68, 68, 1); }
.signal-row.notable { border-left-color: rgba(245, 158, 11, 1); }

.signal-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #cbd5e1;
    flex: 1;
}
.signal-z {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    font-weight: 600;
    min-width: 60px;
    text-align: right;
}
.signal-z.extreme { color: rgba(239, 68, 68, 1); }
.signal-z.notable { color: rgba(245, 158, 11, 1); }
.signal-tag {
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.signal-tag.over  { background: #1c2e1c; color: #4ade80; }
.signal-tag.under { background: #2e1c1c; color: #f87171; }

/* ── KPI strip ── */
.kpi-strip {
    background: #141720;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-delta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
}
.kpi-delta.pos { color: rgba(34, 197, 94, 1); }
.kpi-delta.neg { color: rgba(239, 68, 68, 1); }
.kpi-delta.neu { color: #64748b; }

/* ── Section titles ── */
.section-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #475569;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2330;
    padding-bottom: 0.5rem;
    margin: 2rem 0 1rem 0;
}

/* Streamlit component overrides */
div[data-testid="stMetric"] { background: transparent !important; }
div[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.5rem !important;
    color: #f1f5f9 !important;
}
div[data-testid="stMetricDelta"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #475569 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stDataFrame { background: #141720 !important; }
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    return DataPipeline().load_latest()

@st.cache_data(ttl=3600)
def load_signals():
    path = os.path.join('data', 'signals.json')
    if not os.path.exists(path):
        return [], {}
    with open(path) as f:
        d = json.load(f)
    return d.get('signals', []), d.get('regime', {})

@st.cache_data(ttl=3600)
def load_ratios():
    path = os.path.join('data', 'commodity_ratios.csv')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

df              = load_data()
signals, regime = load_signals()
ratios_df       = load_ratios()

if df.empty:
    st.error("No data found. Run `python automation/run_daily.py` first.")
    st.stop()

# Exclude derived columns; retain raw price series only
base_cols = [
    c for c in df.columns
    if c != 'date'
    and not c.endswith(('_change_pct', '_ma', '_zscore', '_signal'))
]
latest = df.iloc[-1]
prev   = df.iloc[-2] if len(df) > 1 else latest

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <p class="page-title">Agricultural Commodity Tracker</p>
  <p class="page-subtitle">Market Overview</p>
</div>
""", unsafe_allow_html=True)

last_date     = df['date'].max().strftime('%d %b %Y')
n_signals     = len(signals)
n_extreme     = sum(1 for s in signals if abs(s.get('z_score', s.get('strength', 0))) > 2)
n_commodities = len(base_cols)

# Top-level KPI strip
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Last Update", last_date)
with k2:
    st.metric("Commodities Tracked", n_commodities)
with k3:
    st.metric("Active Signals", n_signals, delta=f"{n_extreme} extreme" if n_extreme else None)
with k4:
    notable_count = sum(
        1 for c in base_cols
        if pd.notna(latest.get(f"{c}_zscore")) and abs(latest[f"{c}_zscore"]) > 1
    )
    st.metric("Notable Moves", notable_count)


# ── Price history chart ───────────────────────────────────────────────────────
st.markdown('<p class="section-title">Price History by Category</p>', unsafe_allow_html=True)

# Category display metadata: key -> (label,)
CATEGORY_META = {
    'energy_input': ('', 'Energy Inputs'),
    'crop':         ('', 'Crops'),
    'fertilizer':   ('', 'Fertilizers'),
    'livestock':    ('', 'Livestock'),
    'index':        ('', 'Indices'),
    'economic':     ('', 'Economic'),
}

# Group commodities present in the dataset by their configured category
categories_data = {}
for c in base_cols:
    info = ConfigLoader.get_commodity_info(c)
    cat  = info.get('category', 'other')
    if cat not in categories_data:
        categories_data[cat] = []
    categories_data[cat].append(c)

# Per-category colour palette
CAT_COLORS = {
    'energy_input': ['rgba(245,158,11,1)',  'rgba(251,191,36,0.7)', 'rgba(217,119,6,0.7)',  'rgba(180,83,9,0.5)',   'rgba(120,53,15,0.5)'],
    'crop':         ['rgba(34,197,94,1)',   'rgba(74,222,128,0.7)', 'rgba(22,163,74,0.7)',  'rgba(21,128,61,0.5)',  'rgba(20,83,45,0.5)'],
    'fertilizer':   ['rgba(56,189,248,1)',  'rgba(125,211,252,0.7)','rgba(14,165,233,0.7)', 'rgba(2,132,199,0.5)',  'rgba(7,89,133,0.5)'],
    'livestock':    ['rgba(239,68,68,1)',   'rgba(248,113,113,0.7)','rgba(220,38,38,0.7)'],
    'index':        ['rgba(168,85,247,1)',  'rgba(192,132,252,0.7)'],
    'economic':     ['rgba(148,163,184,1)', 'rgba(100,116,139,0.7)'],
}

# Category multi-select filter
selected_cats = st.multiselect(
    'Filter categories',
    options=list(CATEGORY_META.keys()),
    default=list(categories_data.keys()),
    format_func=lambda k: CATEGORY_META.get(k, ('', k.replace('_',' ').title()))[1]
)

fig = go.Figure()

for cat_key in (selected_cats or list(categories_data.keys())):
    colors = CAT_COLORS.get(cat_key, ['rgba(148,163,184,1)'])
    commodities_in_cat = categories_data.get(cat_key, [])

    for i, c in enumerate(commodities_in_cat):
        if c not in df.columns:
            continue
        info  = ConfigLoader.get_commodity_info(c)
        name  = info.get('name', c.replace('_', ' ').title())
        color = colors[i % len(colors)]

        series = df[['date', c]].dropna(subset=[c])
        if series.empty:
            continue

        fig.add_trace(go.Scatter(
            x=series['date'],
            y=series[c],
            mode='lines',
            name=name,
            line=dict(width=1.5, color=color),
            hovertemplate=f'<b>{name}</b><br>%{{x|%d %b %Y}}<br>%{{y:.2f}}<extra></extra>',
        ))

fig.update_layout(
    height=480,
    margin=dict(l=8, r=8, t=16, b=8),
    paper_bgcolor='#141720',
    plot_bgcolor='#141720',
    font=dict(color='#64748b', size=11, family='IBM Plex Mono'),
    legend=dict(
        orientation='h',
        y=-0.12,
        x=0,
        font=dict(size=9, color='rgba(148,163,184,1)'),
        bgcolor='rgba(0,0,0,0)',
    ),
    xaxis=dict(
        showgrid=True,
        gridcolor='rgba(30,35,48,1)',
        zeroline=False,
        tickfont=dict(size=9),
        color='#475569',
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgba(30,35,48,1)',
        zeroline=False,
        tickfont=dict(size=9),
        color='#475569',
    ),
    hovermode='x unified',
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ── Commodity category cards ──────────────────────────────────────────────────
st.markdown('<p class="section-title">Commodities by Category</p>', unsafe_allow_html=True)

def last_valid(col):
    """Return the last non-null value for a given column, or None."""
    if col not in df.columns:
        return None
    s = df[col].dropna()
    return float(s.iloc[-1]) if not s.empty else None

# Render category cards in rows of three
cat_keys = list(CATEGORY_META.keys())
for k in categories_data:
    if k not in cat_keys:
        cat_keys.append(k)
cat_keys = [k for k in cat_keys if k in categories_data]

for row_start in range(0, len(cat_keys), 3):
    row_cats = cat_keys[row_start:row_start + 3]
    cols = st.columns(len(row_cats))

    for col_widget, cat_key in zip(cols, row_cats):
        _, cat_label = CATEGORY_META.get(cat_key, ('', cat_key.replace('_', ' ').title()))
        commodities_in_cat = categories_data[cat_key]

        rows_html = ''
        for c in commodities_in_cat:
            info  = ConfigLoader.get_commodity_info(c)
            name  = info.get('name', c.replace('_', ' ').title())
            unit  = info.get('unit', '')
            price = last_valid(c)
            chg   = last_valid(f'{c}_change_pct')

            price_str = f'{price:.2f}' if price is not None else '—'

            if chg is not None:
                chg_cls  = 'pos' if chg > 0 else ('neg' if chg < 0 else 'neu')
                chg_sign = '+' if chg > 0 else ''
                chg_str  = f'{chg_sign}{chg:.2f}%'
            else:
                chg_cls, chg_str = 'neu', '—'

            rows_html += f"""
            <div class="commodity-row">
                <span class="commodity-row-name">{name}</span>
                <span class="commodity-row-price">{price_str}</span>
                <span class="commodity-row-chg {chg_cls}">{chg_str}</span>
            </div>
            """

        with col_widget:
            st.markdown(f"""
            <div class="category-card {cat_key}">
                <div class="category-header {cat_key}">{cat_label}</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

# ── Active signals ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Active Signals — Statistical Anomalies (|z| > 1)</p>',
            unsafe_allow_html=True)

# Derive live signals from the most recent row of the dataset
live_signals = []
latest_row = df.iloc[-1]

for c in base_cols:
    z_col = f"{c}_zscore"
    if z_col in df.columns:
        z = latest_row.get(z_col)
        if pd.notna(z) and abs(z) > 1:
            info = ConfigLoader.get_commodity_info(c)
            live_signals.append({
                'commodity': c,
                'name': info.get('name', c.replace('_', ' ').title()),
                'z': float(z),
                'type': 'overvalued' if z > 0 else 'undervalued'
            })

# Sort by absolute z-score, most significant first
live_signals.sort(key=lambda x: abs(x['z']), reverse=True)

if live_signals:
    sig_col1, sig_col2 = st.columns(2)
    half = (len(live_signals) + 1) // 2

    for col_widget, chunk in [(sig_col1, live_signals[:half]), (sig_col2, live_signals[half:])]:
        with col_widget:
            for s in chunk:
                level     = 'extreme' if abs(s['z']) > 2 else 'notable'
                tag       = 'over'    if s['type'] == 'overvalued' else 'under'
                tag_label = 'ABOVE NORM' if s['type'] == 'overvalued' else 'BELOW NORM'
                st.markdown(f"""
                <div class="signal-row {level}">
                    <span class="signal-name">{s['name']}</span>
                    <span class="signal-tag {tag}">{tag_label}</span>
                    <span class="signal-z {level}">{s['z']:+.2f}σ</span>
                </div>
                """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="padding:1.5rem;background:#141720;border-radius:8px;
                border:1px solid #1e2330;color:#64748b;
                font-family:'IBM Plex Mono',monospace;font-size:0.85rem;">
        No anomalous signals — all commodities within normal statistical range.
    </div>
    """, unsafe_allow_html=True)

# ── Commodity snapshot table ──────────────────────────────────────────────────
st.markdown('<p class="section-title">Commodity Snapshot</p>', unsafe_allow_html=True)

rows = []
for c in base_cols:
    info = ConfigLoader.get_commodity_info(c)
    freq = info.get('frequency', 'monthly')

    def last_valid_snap(col):
        if col not in df.columns:
            return None
        s = df[col].dropna()
        return float(s.iloc[-1]) if not s.empty else None

    cur = last_valid_snap(c)
    chg = last_valid_snap(f'{c}_change_pct')
    z   = last_valid_snap(f'{c}_zscore')

    if z is not None:
        if   abs(z) > 2: sig = 'Extreme'
        elif abs(z) > 1: sig = 'Notable'
        else:            sig = 'Normal'
    else:
        sig = '—'

    rows.append({
        'Commodity': info.get('name', c.replace('_', ' ').title()),
        'Category':  info.get('category', '—').replace('_', ' ').title(),
        'Freq':      freq.title()[:3],
        'Price':     round(cur, 2) if cur is not None else None,
        'Chg %':     round(chg, 2) if chg is not None else None,
        'Z-Score':   round(z,   2) if z   is not None else None,
        'Signal':    sig,
    })

# Sort by absolute z-score descending
snap_df = pd.DataFrame(rows).sort_values('Z-Score', key=lambda x: x.abs(), ascending=False)

def color_z(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ''
    if   abs(val) > 2: return 'color: rgba(239, 68, 68, 1); font-weight:600'
    elif abs(val) > 1: return 'color: rgba(245, 158, 11, 1); font-weight:600'
    else:              return 'color: #4ade80'

def color_chg(val):
    if val > 0:   return 'color: #4ade80'
    elif val < 0: return 'color: #f87171'
    return ''

st.dataframe(
    snap_df.style
        .applymap(color_z,   subset=['Z-Score'])
        .applymap(color_chg, subset=['Chg %'])
        .format({
            'Price':   lambda x: f'{x:.2f}' if x is not None and not np.isnan(x) else '—',
            'Chg %':   lambda x: f'{x:+.2f}%' if x is not None and not np.isnan(x) else '—',
            'Z-Score': lambda x: f'{x:+.2f}' if x is not None and not np.isnan(x) else '—',
        }),
    hide_index=True,
    use_container_width=True,
    height=420
)

st.markdown(
    "<p style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;"
    "color:rgba(51, 65, 85, 1);margin-top:0.5rem;'>"
    f"Data source: FRED API · {last_date} · "
    f"{n_commodities} series tracked</p>",
    unsafe_allow_html=True
)