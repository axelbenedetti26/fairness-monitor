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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .eyebrow {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        color: #c9a24b;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #f2f5f9;
        margin: 0 0 0.5rem 0;
        line-height: 1.15;
    }
    .hero-title .accent {
        font-family: 'Fraunces', serif;
        font-style: italic;
        font-weight: 600;
        color: #c9a24b;
    }

    .hero-subtitle {
        color: #8a96a8;
        font-size: 1rem;
        line-height: 1.5;
        max-width: 640px;
        margin-bottom: 1.8rem;
    }

    .tag-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 2rem; }
    .tag {
        padding: 0.45rem 0.9rem;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.02);
        font-size: 0.85rem;
        color: #b7c0cd;
    }
    .tag b { color: #eef1f6; font-weight: 600; }
    .tag.priv { border-color: rgba(201, 162, 75, 0.4); }
    .tag.priv b { color: #d9bd7a; }

    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #eef1f6;
        margin: 2.4rem 0 1rem 0;
    }
    .section-sub { color: #7c8798; font-size: 0.9rem; margin-top: -0.6rem; margin-bottom: 1.2rem; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    .step-row {
        display: flex;
        gap: 2.2rem;
        flex-wrap: wrap;
        padding: 0.4rem 0;
    }
    .step-item { display: flex; align-items: center; gap: 0.8rem; }
    .step-box {
        width: 32px; height: 32px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem;
        color: #5c6779;
        flex-shrink: 0;
    }
    .step-item.done .step-box { border-color: #4ade80; color: #4ade80; background: rgba(74, 222, 128, 0.08); }
    .step-item.active .step-box { border-color: #c9a24b; color: #c9a24b; background: rgba(201, 162, 75, 0.08); }
    .step-status { font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; color: #5c6779; display: block; }
    .step-item.done .step-status { color: #4ade80; }
    .step-item.active .step-status { color: #c9a24b; }
    .step-label { font-size: 0.95rem; font-weight: 600; color: #cfd6e0; }
    .step-item:not(.done):not(.active) .step-label { color: #5c6779; }

    .gauge-rail { border-left: 4px solid; padding-left: 1rem; margin: -0.2rem 0 1rem -1rem; }
    .gauge-head { display: flex; align-items: center; justify-content: space-between; }
    .gauge-name { font-weight: 700; font-size: 1rem; color: #eef1f6; }

    .pill {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border-radius: 5px;
        padding: 0.25rem 0.55rem;
        white-space: nowrap;
    }
    .pill.ok { color: #1a2e22; background: #4ade80; }
    .pill.watch { color: #2e2410; background: #d9bd7a; }
    .pill.breach { color: #2e1414; background: #f87171; }

    .inconclusive-mark {
        font-size: 0.72rem;
        color: #7c8798;
        border-bottom: 1px dotted #7c8798;
        cursor: help;
        margin-left: 0.4rem;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        color: #97a2b2;
        margin-top: 1rem;
        padding-top: 0.9rem;
        border-top: 1px solid rgba(255, 255, 255, 0.07);
    }
    .stat-row b { color: #d7dde5; }

    .rt-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .rt-table th {
        text-align: right;
        font-size: 0.72rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #7c8798;
        font-weight: 600;
        padding: 0.6rem 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .rt-table th:first-child, .rt-table th:nth-child(2) { text-align: left; }
    .rt-table td {
        text-align: right;
        padding: 0.6rem 0.7rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    }
    .rt-table td:first-child, .rt-table td:nth-child(2) { text-align: left; }
    .rt-table tr:hover td { background: rgba(201, 162, 75, 0.04); }
    .rt-table td.breach { color: #f87171; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_PRED_PATH = "monitor_predictions.csv"
DEFAULT_SHAP_PATH = "shap_importance.csv"
N_BOOT = 200

ATTR_CONFIG = {
    "Sex": {"column": "sex", "priv_code": 2, "priv_label": "Women", "under_code": 1, "under_label": "Men"},
    "Age": {
        "column": "age_group", "priv_code": 1, "priv_label": "Core age (25-65)",
        "under_code": 0, "under_label": "Edge age (<25 or >65)",
    },
}

# key -> (label, caption, null_value, format, is_ratio). Positive class = default (1).
METRIC_INFO = {
    "SP_diff": ("Statistical parity diff", "How different the share predicted to default (default = 1) is between groups.", 0.0, "{:.3f}", False),
    "TPR_diff": ("TPR gap (default = 1)", "How different the share of true defaulters correctly flagged is between groups.", 0.0, "{:.3f}", False),
    "FPR_diff": ("FPR gap (default = 1)", "How different the share of non-defaulters wrongly flagged as default is between groups.", 0.0, "{:.3f}", False),
    "DI": ("Disparate impact (four-fifths ratio)", "Underprivileged default-prediction rate as a share of the privileged rate. 1.00 = parity.", 1.0, "{:.2f}", True),
}
METRIC_CAPTIONS = {v[0]: v[1] for v in METRIC_INFO.values()}
METRIC_KEYS = {v[0]: k for k, v in METRIC_INFO.items()}

HEURISTIC_GREEN = 0.03
HEURISTIC_RED = 0.07
DI_GREEN = 0.20
DI_RED = 0.40


# ----------------------------------------------------------------------------
# Data loading + metrics
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
    if metric_key == "DI":
        return DI_GREEN, DI_RED
    if metric_key == "SP_diff" and priv_pos_rate and not np.isnan(priv_pos_rate):
        green = 0.2 * priv_pos_rate
        return green, green * 1.75
    return HEURISTIC_GREEN, HEURISTIC_RED


def distance(metric_key, value):
    return abs(value - METRIC_INFO[metric_key][2])


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


PILL_TEXT = {"ok": "Within tolerance", "watch": "Monitor", "breach": "Breach"}
RADAR_COLORS = ["#c9a24b", "#5b8ac2", "#8a5fd6", "#4ade80", "#f87171"]


def make_radar(current_df, models, metric_keys=("SP_diff", "TPR_diff", "FPR_diff")):
    labels = {"SP_diff": "SP diff", "TPR_diff": "TPR gap", "FPR_diff": "FPR gap"}
    theta = [labels[k] for k in metric_keys] + [labels[metric_keys[0]]]
    fig = go.Figure()
    for i, model in enumerate(models):
        row = current_df[current_df["model"] == model]
        if row.empty:
            continue
        vals = [abs(row[k].values[0]) for k in metric_keys]
        vals.append(vals[0])
        color = RADAR_COLORS[i % len(RADAR_COLORS)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=theta, name=model,
            line={"color": color, "width": 2}, fill="toself", fillcolor=color, opacity=0.65,
        ))
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "color": "#5c6779", "gridcolor": "rgba(255,255,255,0.06)"},
            "angularaxis": {"color": "#cfd6e0", "gridcolor": "rgba(255,255,255,0.06)"},
        },
        showlegend=True,
        legend={"orientation": "h", "y": -0.12, "font": {"color": "#cfd6e0", "size": 12}},
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
            "reference": ref, "valueformat": fmt,
            "decreasing": {"color": "#f87171" if is_ratio else "#4ade80"},
            "increasing": {"color": "#4ade80" if is_ratio else "#f87171"},
            "font": {"size": 14},
        }
    fig = go.Figure(go.Indicator(**indicator_kwargs))
    fig.update_layout(
        height=190,
        margin=dict(l=16, r=16, t=6, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#eef1f6", "family": "Inter, sans-serif"},
    )
    return fig


# ----------------------------------------------------------------------------
# Sidebar
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
# Hero
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Credit Risk &amp; Fairness Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Fairness Monitor <span class="accent">Console</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Point-in-time comparison of model predictions before and after bias '
    "mitigation. Reads predictions, ground truth and the protected attribute only — never the raw "
    "input features.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="tag-row">
        <span class="tag">Privileged &nbsp;<b>{attr['priv_label']}</b></span>
        <span class="tag priv">Underprivileged &nbsp;<b>{attr['under_label']}</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Step tracker
# ----------------------------------------------------------------------------
steps = ["Data ingestion", "Bias detection", "Mitigation", "Validation & sign-off"]
is_synthetic = dataset_choice != available_datasets[0]
statuses = ["done", "done", "done" if is_synthetic else "active", "active" if is_synthetic else ""]
status_words = {"done": "Complete", "active": "In progress", "": "Pending"}

html = '<div class="step-row">'
for i, (label, st_cls) in enumerate(zip(steps, statuses)):
    mark = "✓" if st_cls == "done" else str(i + 1)
    cls = f"step-item {st_cls}".strip()
    html += (
        f'<div class="{cls}"><div class="step-box">{mark}</div>'
        f'<div><span class="step-status">{status_words[st_cls]}</span>'
        f'<span class="step-label">{label}</span></div></div>'
    )
html += "</div>"
st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Compute results
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

st.markdown(f'<div class="section-title">{METRIC_INFO[metric_key][0]} by model</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">{METRIC_INFO[metric_key][1]} &nbsp;·&nbsp; '
    f'{breach_count} of {len(selected_models)} models in breach</div>',
    unsafe_allow_html=True,
)

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

                    inc_html = f'<span class="inconclusive-mark" title="95% CI [{ci_low:.3f}, {ci_high:.3f}]">n.s.</span>' if inconclusive else ""
                    st.markdown(
                        f'<div class="gauge-rail" style="border-color:{color};">'
                        f'<div class="gauge-head"><span class="gauge-name">{model}{inc_html}</span>'
                        f'<span class="pill {label}">{PILL_TEXT[label]}</span></div></div>',
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
                    st.markdown(f'<div class="stat-row">{stat_html}</div>', unsafe_allow_html=True)
                else:
                    st.info(f"No data for {model}.")

# ----------------------------------------------------------------------------
# Radar + table
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">Metric profile &amp; evaluation results</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Normalised gap across all three difference metrics — smaller is fairer.</div>', unsafe_allow_html=True)

col_radar, col_table = st.columns([2, 3], gap="large")

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
            is_breach = row_label == "breach"
            rows_html += (
                "<tr>"
                f"<td>{r['model']}</td><td>{r['dataset']}</td>"
                f"<td class='{'breach' if is_breach and metric_key == 'SP_diff' else ''}'>{r['SP_diff']:.4f}</td>"
                f"<td class='{'breach' if is_breach and metric_key == 'TPR_diff' else ''}'>{r['TPR_diff']:.4f}</td>"
                f"<td class='{'breach' if is_breach and metric_key == 'FPR_diff' else ''}'>{r['FPR_diff']:.4f}</td>"
                f"<td class='{'breach' if is_breach and metric_key == 'DI' else ''}'>{r['DI']:.2f}</td>"
                "</tr>"
            )
        table_html = (
            "<table class='rt-table'><thead><tr>"
            "<th>Model</th><th>Dataset</th><th>SP diff</th><th>TPR gap</th><th>FPR gap</th><th>DI</th>"
            f"</tr></thead><tbody>{rows_html}</tbody></table>"
        )
        st.markdown(table_html, unsafe_allow_html=True)
        st.caption("Figures in breach on the active metric are highlighted.")

        csv_bytes = (
            display_df.drop(columns=["priv_pos_rate", "SP_diff_ci", "TPR_diff_ci", "FPR_diff_ci", "DI_ci"])
            .to_csv(index=False).encode("utf-8")
        )
        st.download_button("Download report (CSV)", csv_bytes, file_name="fairness_report.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# SHAP + technical notes
# ----------------------------------------------------------------------------
if shap_df is not None:
    with st.expander("Proxy check (SHAP)"):
        st.dataframe(shap_df, width="stretch", hide_index=True)
        st.caption("Exploratory: features with high importance and high correlation with the protected attribute are proxy candidates. Not a causal explanation or a guarantee against proxy discrimination.")
elif custom_data:
    with st.expander("Proxy check (SHAP)"):
        st.info("SHAP proxy check is only available for the preloaded experiment data, not for uploaded CSVs.")

with st.expander("Technical notes — methodology, thresholds, limitations"):
    st.markdown(
        f"""
**Thresholds.** The Disparate Impact tolerance (≥ 0.80) is the four-fifths rule from EEOC adverse-impact
guidance — US employment law, not lending regulation, used here as a practical reference point. TPR/FPR
gap cutoffs ({HEURISTIC_GREEN:.2f} / {HEURISTIC_RED:.2f}) are literature heuristics with no regulatory backing.

**Significance.** "n.s." next to a model means the 95% bootstrap confidence interval ({N_BOOT} resamples) for
that gap includes the no-disparity value (zero for differences, one for the disparate impact ratio) — the
color still reflects the point estimate, but treat it as directional. A non-significant result is not proof
that no disparity exists.

**Scope.** Bias is synthetically injected (15% label flipping on both groups), not organically occurring.
The positive class is default (1). This view shows a single representative run (one random seed); the
statistical claims of the thesis rest on the full ten-seed analysis, not on this demonstration. This is a
point-in-time, proof-of-concept dashboard — no access control, audit trail, or versioning.
        """
    )
