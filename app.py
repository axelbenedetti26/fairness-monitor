import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fairness Monitor — Credit Scoring",
    page_icon="⚖️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Fraunces:ital,opsz,wght@0,9..144,500..700;1,9..144,500..700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }

    /* header bar */
    .fm-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 1.4rem;
    }
    .fm-brand { display: flex; align-items: center; gap: 0.7rem; }
    .fm-monogram {
        width: 34px; height: 34px;
        border: 1px solid #c9a24b;
        display: flex; align-items: center; justify-content: center;
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-size: 1.1rem;
        color: #c9a24b;
    }
    .fm-crumbs {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #7c8798;
    }
    .fm-timestamp {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.05em;
        color: #6b7688;
        text-align: right;
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.6rem;
        font-weight: 600;
        color: #f2f5f9;
        margin: 0.3rem 0 0.3rem 0;
        line-height: 1.08;
        letter-spacing: -0.015em;
    }
    .hero-title .accent {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 600;
        color: #c9a24b;
    }

    .hero-subtitle {
        color: #8a96a8;
        font-size: 0.98rem;
        margin-bottom: 1.4rem;
        max-width: 640px;
    }

    /* rectangular badges with a marker square, sit top-right of header */
    .tag-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .tag {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.75rem;
        border: 1px solid #2a3644;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #b7c0cd;
    }
    .tag .sq { width: 8px; height: 8px; display: inline-block; }
    .tag.priv { border-color: #c9a24b; }
    .tag.priv .sq { background: #c9a24b; }
    .tag.under .sq { border: 1px solid #7c8798; }

    .section-title {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.35rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
        color: #eef1f6;
    }
    .section-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        color: #7c8798;
        text-transform: uppercase;
    }
    .section-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.4rem;
        border-bottom: 1px solid #212a37;
        padding-bottom: 0.6rem;
        margin: 1.6rem 0 1rem 0;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 6px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
        transition: border-color 0.2s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(201, 162, 75, 0.4) !important;
    }

    /* step tracker: square numbered/check badges */
    .step-track {
        display: flex;
        gap: 1.6rem;
        flex-wrap: wrap;
        margin: 0.2rem 0 1.6rem 0;
    }
    .step-item { display: flex; align-items: center; gap: 0.7rem; }
    .step-box {
        width: 30px; height: 30px;
        border: 1px solid #2a3644;
        display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #5c6779;
        flex-shrink: 0;
    }
    .step-item.done .step-box { border-color: #4ade80; color: #4ade80; }
    .step-item.active .step-box { border-color: #c9a24b; color: #c9a24b; }
    .step-status {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #5c6779;
        display: block;
    }
    .step-item.done .step-status { color: #4ade80; }
    .step-item.active .step-status { color: #c9a24b; }
    .step-label { font-size: 0.9rem; font-weight: 600; color: #cfd6e0; }
    .step-item:not(.done):not(.active) .step-label { color: #5c6779; }

    /* gauge card chrome */
    .status-bar { height: 3px; margin: -1rem -1rem 0.8rem -1rem; }
    .gauge-head { display: flex; align-items: flex-start; justify-content: space-between; margin-bottom: 0.1rem; }
    .gauge-name { font-weight: 700; font-size: 0.98rem; color: #eef1f6; }
    .gauge-sub { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #7c8798; letter-spacing: 0.04em; }

    .pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border: 1px solid;
        padding: 0.16rem 0.4rem;
        white-space: nowrap;
    }
    .pill.ok { color: #4ade80; border-color: #4ade80; }
    .pill.watch { color: #d9bd7a; border-color: #d9bd7a; }
    .pill.breach { color: #f87171; border-color: #f87171; }

    .inconclusive-mark {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        color: #7c8798;
        border-bottom: 1px dotted #7c8798;
        cursor: help;
    }

    .stat-strip {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        color: #97a2b2;
        margin-top: 0.8rem;
        padding-top: 0.7rem;
        border-top: 1px solid #1c232e;
    }
    .stat-strip b { color: #d7dde5; font-weight: 600; }

    .rt-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .rt-table th {
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #7c8798;
        font-weight: 600;
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid #263140;
    }
    .rt-table th:first-child, .rt-table th:nth-child(2) { text-align: left; }
    .rt-table td {
        text-align: right;
        padding: 0.5rem 0.6rem;
        border-bottom: 1px solid #1a212c;
        font-family: 'JetBrains Mono', monospace;
        font-variant-numeric: tabular-nums;
    }
    .rt-table td:first-child, .rt-table td:nth-child(2) {
        text-align: left;
        font-family: 'Inter', sans-serif;
    }
    .rt-table tr:hover td { background: rgba(201, 162, 75, 0.05); }
    .rt-table td.breach { color: #f87171; }
    .rt-table caption-note { color: #7c8798; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_PRED_PATH = "monitor_predictions.csv"
DEFAULT_SHAP_PATH = "shap_importance.csv"
N_BOOT = 200  # bootstrap resamples for confidence intervals

ATTR_CONFIG = {
    "Sex": {
        "column": "sex",
        "priv_code": 2,
        "priv_label": "Women",
        "under_code": 1,
        "under_label": "Men",
    },
    "Age": {
        "column": "age_group",
        "priv_code": "core",
        "priv_label": "Core age (25-65)",
        "under_code": "edge",
        "under_label": "Edge age (<25 or >65)",
    },
}

# Every entry: key -> (label, caption, null_value, value_format, is_ratio)
METRIC_INFO = {
    "SP_diff": (
        "Statistical parity diff",
        "How different the share of predicted positive outcomes is between groups.",
        0.0, "{:.3f}", False,
    ),
    "TPR_diff": (
        "TPR diff (Equal Opportunity)",
        "How different the share of correctly approved good payers is between groups.",
        0.0, "{:.3f}", False,
    ),
    "FPR_diff": (
        "FPR diff (EOd)",
        "How different the share of cases wrongly flagged as risk is between groups.",
        0.0, "{:.3f}", False,
    ),
    "DI": (
        "Disparate impact (four-fifths ratio)",
        "Underprivileged approval rate as a share of the privileged rate. 1.00 = parity, ≥ 0.80 = within the four-fifths tolerance.",
        1.0, "{:.2f}", True,
    ),
}
METRIC_CAPTIONS = {v[0]: v[1] for v in METRIC_INFO.values()}
METRIC_KEYS = {v[0]: k for k, v in METRIC_INFO.items()}

# Heuristic thresholds for the diff-style metrics (no regulatory number exists).
HEURISTIC_GREEN = 0.03
HEURISTIC_RED = 0.07
# Distance-from-parity thresholds for Disparate Impact, anchored on the
# four-fifths (80%) adverse-impact rule: DI < 0.80 is the classic breach line.
DI_GREEN = 0.20  # |1 - DI| < 0.20  <=>  DI > 0.80
DI_RED = 0.40


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    preds = pd.read_csv(DEFAULT_PRED_PATH)
    try:
        shap_df = pd.read_csv(DEFAULT_SHAP_PATH)
    except FileNotFoundError:
        shap_df = None
    return preds, shap_df


def _rates(y_true, y_pred):
    pos_rate = y_pred.mean() if len(y_pred) else np.nan
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    tpr = y_pred[pos_mask].mean() if pos_mask.sum() else np.nan
    fpr = y_pred[neg_mask].mean() if neg_mask.sum() else np.nan
    return pos_rate, tpr, fpr


def fairness_metrics(df, y_true_col, y_pred_col, group_col, priv_code, under_code):
    """SP/TPR/FPR differences plus Disparate Impact (four-fifths ratio).
    Positive diff = underprivileged group is worse off. DI < 1 = same."""
    priv = df[df[group_col].astype(str) == str(priv_code)]
    under = df[df[group_col].astype(str) == str(under_code)]

    priv_pos, priv_tpr, priv_fpr = _rates(priv[y_true_col].values, priv[y_pred_col].values)
    under_pos, under_tpr, under_fpr = _rates(under[y_true_col].values, under[y_pred_col].values)

    di = under_pos / priv_pos if priv_pos else np.nan

    return {
        "SP_diff": under_pos - priv_pos,
        "TPR_diff": under_tpr - priv_tpr,
        "FPR_diff": under_fpr - priv_fpr,
        "DI": di,
        "priv_pos_rate": priv_pos,
    }


def bootstrap_ci(df, y_true_col, y_pred_col, group_col, priv_code, under_code, n_boot=N_BOOT, seed=42):
    """95% bootstrap CI for each metric, resampling each group independently."""
    rng = np.random.default_rng(seed)
    priv = df[df[group_col].astype(str) == str(priv_code)]
    under = df[df[group_col].astype(str) == str(under_code)]

    yt_p, yp_p = priv[y_true_col].values, priv[y_pred_col].values
    yt_u, yp_u = under[y_true_col].values, under[y_pred_col].values
    n_p, n_u = len(yt_p), len(yt_u)

    sp_boot, tpr_boot, fpr_boot, di_boot = [], [], [], []
    if n_p > 0 and n_u > 0:
        for _ in range(n_boot):
            idx_p = rng.integers(0, n_p, n_p)
            idx_u = rng.integers(0, n_u, n_u)
            pp, tp, fp = _rates(yt_p[idx_p], yp_p[idx_p])
            pu, tu, fu = _rates(yt_u[idx_u], yp_u[idx_u])
            sp_boot.append(pu - pp)
            tpr_boot.append(tu - tp)
            fpr_boot.append(fu - fp)
            di_boot.append(pu / pp if pp else np.nan)

    def ci(vals):
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return (np.nan, np.nan)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "SP_diff_ci": ci(sp_boot),
        "TPR_diff_ci": ci(tpr_boot),
        "FPR_diff_ci": ci(fpr_boot),
        "DI_ci": ci(di_boot),
    }


def metric_thresholds(metric_key, priv_pos_rate):
    """Return (green, red) cutoffs in 'distance from the ideal value' units."""
    if metric_key == "DI":
        return DI_GREEN, DI_RED
    if metric_key == "SP_diff" and priv_pos_rate and not np.isnan(priv_pos_rate):
        green = 0.2 * priv_pos_rate  # four-fifths rule, translated to an absolute gap
        return green, green * 1.75
    return HEURISTIC_GREEN, HEURISTIC_RED


def distance(metric_key, value):
    null_value = METRIC_INFO[metric_key][2]
    return abs(value - null_value)


def status(metric_key, value, ci_low, ci_high, green_cut, red_cut):
    d = distance(metric_key, value)
    if d < green_cut:
        color, label = "#4ade80", "ok"
    elif d < red_cut:
        color, label = "#d9bd7a", "watch"
    else:
        color, label = "#f87171", "breach"
    null_value = METRIC_INFO[metric_key][2]
    inconclusive = not (np.isnan(ci_low) or np.isnan(ci_high)) and ci_low <= null_value <= ci_high
    return color, label, inconclusive


PILL_TEXT = {"ok": "WITHIN TOLERANCE", "watch": "MONITOR", "breach": "BREACH"}
RADAR_COLORS = ["#c9a24b", "#5b8ac2", "#8a5fd6", "#4ade80", "#f87171"]


def make_radar(current_df, models, metric_keys=("SP_diff", "TPR_diff", "FPR_diff")):
    labels = {"SP_diff": "SP diff", "TPR_diff": "EO diff", "FPR_diff": "FPR diff"}
    theta = [labels[k] for k in metric_keys] + [labels[metric_keys[0]]]
    fig = go.Figure()
    for i, model in enumerate(models):
        row = current_df[current_df["model"] == model]
        if row.empty:
            continue
        vals = [abs(row[k].values[0]) for k in metric_keys]
        vals.append(vals[0])
        color = RADAR_COLORS[i % len(RADAR_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=vals, theta=theta, name=model,
                line={"color": color, "width": 2},
                fill="toself", fillcolor=color, opacity=0.7,
            )
        )
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "color": "#5c6779", "gridcolor": "#1c232e"},
            "angularaxis": {"color": "#cfd6e0", "gridcolor": "#1c232e"},
        },
        showlegend=True,
        legend={"orientation": "h", "y": -0.12, "font": {"color": "#cfd6e0", "size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#cfd6e0", "family": "Inter, sans-serif"},
        height=360,
        margin=dict(l=50, r=50, t=20, b=40),
    )
    return fig


def make_gauge(metric_key, value, color, baseline_value=None):
    is_ratio = METRIC_INFO[metric_key][4]
    display_value = value if is_ratio else abs(value)
    axis_max = 1.4 if is_ratio else max(0.15, abs(value) * 1.3, 0.05)
    fmt = ".2f" if is_ratio else ".3f"

    indicator_kwargs = dict(
        mode="gauge+number",
        value=display_value,
        number={"valueformat": fmt, "font": {"size": 32}},
        gauge={
            "axis": {"range": [0, axis_max], "tickwidth": 1, "tickcolor": "#3a4454"},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [{"range": [0, axis_max], "color": "rgba(255,255,255,0.03)"}],
        },
    )
    if baseline_value is not None:
        ref = baseline_value if is_ratio else abs(baseline_value)
        indicator_kwargs["mode"] = "gauge+number+delta"
        indicator_kwargs["delta"] = {
            "reference": ref,
            "valueformat": fmt,
            "decreasing": {"color": "#f87171" if is_ratio else "#4ade80"},
            "increasing": {"color": "#4ade80" if is_ratio else "#f87171"},
            "font": {"size": 14},
        }
    fig = go.Figure(go.Indicator(**indicator_kwargs))
    fig.update_layout(
        height=190,
        margin=dict(l=16, r=16, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef1f6", "family": "JetBrains Mono, monospace"},
    )
    return fig


# ----------------------------------------------------------------------------
# Sidebar — data source + filters
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Upload your own CSV")
st.sidebar.caption("Columns: model, dataset, y_true, y_pred, sex, age_group")
uploaded = st.sidebar.file_uploader("Upload", type="csv", label_visibility="collapsed")
st.sidebar.caption("200MB per file • CSV")

custom_data = uploaded is not None

if custom_data:
    monitor_data = pd.read_csv(uploaded)
    shap_df = None
    st.sidebar.success("Using your uploaded file.")
else:
    monitor_data, shap_df = load_default_data()
    st.sidebar.caption("Using the preloaded experiment data.")

st.sidebar.markdown("#### Filters")

attr_name = st.sidebar.selectbox("Protected attribute", list(ATTR_CONFIG.keys()))
attr = ATTR_CONFIG[attr_name]

available_datasets = list(monitor_data["dataset"].unique())
dataset_choice = st.sidebar.radio("Dataset", available_datasets, index=0)

available_models = list(monitor_data["model"].unique())
selected_models = st.sidebar.multiselect("Models", available_models, default=available_models)

metric_name = st.sidebar.selectbox("Metric", list(METRIC_KEYS.keys()), index=3)
metric_key = METRIC_KEYS[metric_name]

# ----------------------------------------------------------------------------
# Header bar
# ----------------------------------------------------------------------------
now = datetime.datetime.now()
ref_id = f"FRM-{now.year}-{now.timetuple().tm_yday:03d}"
st.markdown(
    f"""
    <div class="fm-header">
        <div class="fm-brand">
            <div class="fm-monogram">FM</div>
            <div class="fm-crumbs">Credit Risk &nbsp;·&nbsp; Fairness Monitoring &nbsp;·&nbsp; Ref {ref_id}</div>
        </div>
        <div class="fm-timestamp">Last evaluation<br>{now.strftime('%d %b %Y · %H:%M')}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-title">Fairness Monitor <span class="accent">Console</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Point-in-time comparison of model predictions before and after bias '
    "mitigation. Reads predictions, ground truth and the protected attribute only — never the raw "
    "input features.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="tag-row" style="margin-bottom:1.6rem;">
        <span class="tag priv"><span class="sq"></span>Privileged &nbsp;·&nbsp; {attr['priv_label']}</span>
        <span class="tag under"><span class="sq"></span>Underprivileged &nbsp;·&nbsp; {attr['under_label']}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Step tracker
# ----------------------------------------------------------------------------
steps = ["Data Ingestion", "Bias Detection", "Mitigation", "Validation & Sign-off"]
is_synthetic = dataset_choice != available_datasets[0]
statuses = ["done", "done", "done" if is_synthetic else "active", "active" if is_synthetic else ""]
status_words = {"done": "Complete", "active": "In Progress", "": "Pending"}

with st.container(border=True):
    html = '<div class="step-track">'
    for i, (label, st_cls) in enumerate(zip(steps, statuses)):
        mark = "✓" if st_cls == "done" else f"{i + 1:02d}"
        cls = f"step-item {st_cls}".strip()
        html += (
            f'<div class="{cls}"><div class="step-box">{mark}</div>'
            f'<div><span class="step-status">{status_words[st_cls]}</span>'
            f'<span class="step-label">{label}</span></div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Compute results (point estimates + bootstrap CI)
# ----------------------------------------------------------------------------
rows = []
for model in selected_models:
    for ds in available_datasets:
        sub = monitor_data[(monitor_data["model"] == model) & (monitor_data["dataset"] == ds)]
        if sub.empty:
            continue
        m = fairness_metrics(sub, "y_true", "y_pred", attr["column"], attr["priv_code"], attr["under_code"])
        ci = bootstrap_ci(sub, "y_true", "y_pred", attr["column"], attr["priv_code"], attr["under_code"])
        rows.append({"model": model, "dataset": ds, **m, **ci})

results_df = pd.DataFrame(rows)
baseline_ds = available_datasets[0]
current = results_df[results_df["dataset"] == dataset_choice]
baseline_current = results_df[results_df["dataset"] == baseline_ds]

breach_count = 0
if not current.empty:
    green_cut, red_cut = metric_thresholds(metric_key, current["priv_pos_rate"].mean())
    for model in selected_models:
        r = current[current["model"] == model]
        if not r.empty:
            ci_low, ci_high = r[f"{metric_key}_ci"].values[0]
            _, label, _ = status(metric_key, r[metric_key].values[0], ci_low, ci_high, green_cut, red_cut)
            if label == "breach":
                breach_count += 1

st.markdown(
    f"""
    <div class="section-row">
        <span class="section-title">{METRIC_INFO[metric_key][0]} by model</span>
        <span class="section-meta">{breach_count} of {len(selected_models)} models in breach</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(METRIC_INFO[metric_key][1])

# ----------------------------------------------------------------------------
# Gauges
# ----------------------------------------------------------------------------
if not current.empty:
    green_cut, red_cut = metric_thresholds(metric_key, current["priv_pos_rate"].mean())
    cols = st.columns(len(selected_models)) if selected_models else []
    for col, model in zip(cols, selected_models):
        row = current[current["model"] == model]
        with col:
            with st.container(border=True):
                if not row.empty:
                    value = row[metric_key].values[0]
                    ci_low, ci_high = row[f"{metric_key}_ci"].values[0]
                    color, label, inconclusive = status(metric_key, value, ci_low, ci_high, green_cut, red_cut)
                    st.markdown(f'<div class="status-bar" style="background:{color};"></div>', unsafe_allow_html=True)

                    inc_html = (
                        f' <span class="inconclusive-mark" title="95% CI [{ci_low:.3f}, {ci_high:.3f}]">n.s.</span>'
                        if inconclusive else ""
                    )
                    st.markdown(
                        f'<div class="gauge-head">'
                        f'<div><div class="gauge-name">{model}{inc_html}</div>'
                        f'<div class="gauge-sub">{metric_key.replace("_", " ")}</div></div>'
                        f'<span class="pill {label}">{PILL_TEXT[label]}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    baseline_value = None
                    if dataset_choice != baseline_ds:
                        base_row = baseline_current[baseline_current["model"] == model]
                        if not base_row.empty:
                            baseline_value = base_row[metric_key].values[0]
                    st.plotly_chart(make_gauge(metric_key, value, color, baseline_value), width="stretch")

                    other_keys = [k for k in ("SP_diff", "TPR_diff", "FPR_diff", "DI") if k != metric_key][:2]
                    stat_html = "".join(
                        f'<span>{k.replace("_", " ")} <b>{row[k].values[0]:{METRIC_INFO[k][3][2:-1]}}</b></span>'
                        for k in other_keys
                    )
                    st.markdown(f'<div class="stat-strip">{stat_html}</div>', unsafe_allow_html=True)
                else:
                    st.info(f"No data for {model}.")

# ----------------------------------------------------------------------------
# Multi-metric comparison + results table, side by side
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="section-row"><span class="section-title">Metric profile &amp; evaluation results</span>'
    '<span class="section-meta">Normalised gap · smaller is fairer</span></div>',
    unsafe_allow_html=True,
)

col_radar, col_table = st.columns([2, 3])

with col_radar:
    with st.container(border=True):
        if not current.empty and selected_models:
            st.plotly_chart(make_radar(current, selected_models), width="stretch")
        else:
            st.info("Select at least one model.")

with col_table:
    with st.container(border=True):
        display_df = results_df.copy()
        for col in ["SP_diff", "TPR_diff", "FPR_diff", "DI"]:
            display_df[col] = display_df[col].round(4)

        rows_html = ""
        for _, r in display_df.iterrows():
            ci_low, ci_high = r[f"{metric_key}_ci"]
            g_cut, r_cut = metric_thresholds(metric_key, r["priv_pos_rate"])
            _, row_label, _ = status(metric_key, r[metric_key], ci_low, ci_high, g_cut, r_cut)
            cls = "breach" if row_label == "breach" else ""
            rows_html += (
                "<tr>"
                f"<td>{r['model']}</td><td>{r['dataset']}</td>"
                f"<td class='{cls if metric_key == 'SP_diff' else ''}'>{r['SP_diff']:.4f}</td>"
                f"<td class='{cls if metric_key == 'TPR_diff' else ''}'>{r['TPR_diff']:.4f}</td>"
                f"<td class='{cls if metric_key == 'FPR_diff' else ''}'>{r['FPR_diff']:.4f}</td>"
                f"<td class='{cls if metric_key == 'DI' else ''}'>{r['DI']:.2f}</td>"
                "</tr>"
            )
        table_html = (
            "<table class='rt-table'><thead><tr>"
            "<th>Model</th><th>Dataset</th><th>SP diff</th><th>EO diff</th><th>FPR diff</th><th>DI</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("Figures in breach on the active metric shown in breach colour.")

        csv_bytes = (
            display_df.drop(columns=["priv_pos_rate", "SP_diff_ci", "TPR_diff_ci", "FPR_diff_ci", "DI_ci"])
            .to_csv(index=False)
            .encode("utf-8")
        )
        st.download_button("Download report (CSV)", csv_bytes, file_name="fairness_report.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# SHAP proxy check
# ----------------------------------------------------------------------------
if shap_df is not None:
    with st.expander("Proxy check (SHAP)"):
        st.dataframe(shap_df, width="stretch", hide_index=True)
        st.caption("Features with high importance and high correlation with the protected attribute are proxy candidates.")
elif custom_data:
    with st.expander("Proxy check (SHAP)"):
        st.info("SHAP proxy check is only available for the preloaded experiment data, not for uploaded CSVs.")

# ----------------------------------------------------------------------------
# Technical notes
# ----------------------------------------------------------------------------
with st.expander("Technical notes — methodology, thresholds, limitations"):
    st.markdown(
        f"""
**Thresholds.** The Disparate Impact tolerance (≥ 0.80) is the four-fifths rule from EEOC adverse-impact
guidance — US employment law, not lending regulation, used here as a practical reference point. The SP diff
green cutoff ({HEURISTIC_GREEN:.2f} as a fallback, or a per-dataset value derived from the same rule) follows
the same logic. TPR/FPR diff cutoffs ({HEURISTIC_GREEN:.2f} / {HEURISTIC_RED:.2f}) are literature heuristics
with no regulatory backing.

**Significance.** "n.s." next to a model means the 95% bootstrap confidence interval (200 resamples) for
that gap includes the no-disparity value — the color still reflects the point estimate, but treat it as
directional, not proven.

**Scope.**
- Bias is synthetically injected (15% label flipping), not organically occurring — this shows the monitor's
  sensitivity to a known pattern, not validation against real-world bias.
- This is a point-in-time comparison between two static runs, not continuous production monitoring.
- No access control, audit trail, or versioning — this is a proof-of-concept dashboard.
        """
    )
