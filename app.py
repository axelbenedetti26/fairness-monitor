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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Fraunces:opsz,wght@9..144,500..700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .eyebrow {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.2em;
        color: #c9a24b;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.7rem;
        font-weight: 600;
        font-optical-sizing: auto;
        color: #f2f5f9;
        margin-bottom: 0.25rem;
        line-height: 1.1;
        border-left: 4px solid #c9a24b;
        padding-left: 0.8rem;
        letter-spacing: -0.01em;
    }

    .section-title {
        font-family: 'Fraunces', serif;
    }

    .hero-subtitle {
        color: #93a0b3;
        font-size: 1rem;
        margin-bottom: 1.1rem;
        padding-left: 1rem;
    }

    .snapshot-inline {
        color: #6b7688;
        font-size: 0.76rem;
        letter-spacing: 0.03em;
        margin: 0 0 1.6rem 1rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        color: #dde3ec;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important;
        animation: fadeIn 0.4s ease-in-out;
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        border-color: rgba(58, 110, 165, 0.45) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .step-track {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        position: relative;
        margin: 0.8rem 0 1.8rem 0;
        padding: 0 2%;
    }
    .step-track::before {
        content: '';
        position: absolute;
        top: 8px;
        left: 7%;
        right: 7%;
        height: 1px;
        background: #263140;
        z-index: 0;
    }
    .step {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        flex: 1;
    }
    .step .dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #1a212c;
        border: 2px solid #2a3644;
    }
    .step.active .dot {
        background: #3a6ea5;
        border-color: #3a6ea5;
    }
    .step .label {
        font-size: 0.78rem;
        color: #74808f;
        margin-top: 0.5rem;
        text-align: center;
        max-width: 120px;
    }
    .step.active .label {
        color: #dde3ec;
        font-weight: 600;
    }

    .badge {
        display: inline-block;
        padding: 0.32rem 0.8rem;
        border-radius: 4px;
        background: rgba(201, 162, 75, 0.1);
        border: 1px solid rgba(201, 162, 75, 0.4);
        color: #d9bd7a;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: 0.01em;
    }

    .status-bar {
        height: 4px;
        margin: -1rem -1rem 0.9rem -1rem;
        border-radius: 8px 8px 0 0;
    }

    .best-tag, .warn-tag, .flat-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-radius: 3px;
        padding: 0.14rem 0.45rem;
        margin-left: 0.5rem;
    }
    .best-tag {
        color: #1a2e22;
        background: #4ade80;
        border: 1px solid #4ade80;
    }
    .warn-tag {
        color: #2e1414;
        background: #f87171;
        border: 1px solid #f87171;
    }
    .flat-tag {
        color: #93a0b3;
        background: rgba(147, 160, 179, 0.12);
        border: 1px solid rgba(147, 160, 179, 0.28);
    }

    .inconclusive-mark {
        font-size: 0.7rem;
        color: #93a0b3;
        border-bottom: 1px dotted #93a0b3;
        cursor: help;
    }

    .ci-note {
        font-size: 0.74rem;
        color: #6b7688;
        text-align: center;
        margin-top: -0.4rem;
    }

    .rt-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    .rt-table th {
        text-align: right;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #74808f;
        font-weight: 600;
        padding: 0.5rem 0.7rem;
        border-bottom: 1px solid #263140;
    }
    .rt-table th:first-child, .rt-table th:nth-child(2) { text-align: left; }
    .rt-table td {
        text-align: right;
        padding: 0.55rem 0.7rem;
        border-bottom: 1px solid #1a212c;
        font-variant-numeric: tabular-nums;
    }
    .rt-table td:first-child, .rt-table td:nth-child(2) { text-align: left; }
    .rt-table tr:hover td { background: rgba(58, 110, 165, 0.06); }
    .rt-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 0.4rem; }
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

METRIC_CAPTIONS = {
    "Statistical parity diff": "How different the share of predicted positive outcomes is between groups.",
    "TPR diff (Equal Opportunity)": "How different the share of correctly approved good payers is between groups.",
    "FPR diff (EOd)": "How different the share of cases wrongly flagged as risk is between groups.",
}

METRIC_KEYS = {
    "Statistical parity diff": "SP_diff",
    "TPR diff (Equal Opportunity)": "TPR_diff",
    "FPR diff (EOd)": "FPR_diff",
}

# Fixed heuristic thresholds used for TPR diff / FPR diff, where no single
# regulatory numeric standard exists. These are common cutoffs seen in the
# fairness literature, not a legal requirement.
HEURISTIC_GREEN = 0.03
HEURISTIC_RED = 0.07


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
    """Statistical parity, TPR and FPR differences between privileged and
    underprivileged groups. Positive value = underprivileged group is worse off.
    Also returns the privileged group's raw positive rate, needed to translate
    the four-fifths (80%) adverse-impact rule into an absolute-difference
    threshold for statistical parity."""
    priv = df[df[group_col].astype(str) == str(priv_code)]
    under = df[df[group_col].astype(str) == str(under_code)]

    priv_pos, priv_tpr, priv_fpr = _rates(priv[y_true_col].values, priv[y_pred_col].values)
    under_pos, under_tpr, under_fpr = _rates(under[y_true_col].values, under[y_pred_col].values)

    return {
        "SP_diff": under_pos - priv_pos,
        "TPR_diff": under_tpr - priv_tpr,
        "FPR_diff": under_fpr - priv_fpr,
        "priv_pos_rate": priv_pos,
    }


def bootstrap_ci(df, y_true_col, y_pred_col, group_col, priv_code, under_code, n_boot=N_BOOT, seed=42):
    """95% bootstrap confidence interval for each metric, resampling each
    group independently with replacement. Used to flag when an observed gap
    is not statistically distinguishable from zero."""
    rng = np.random.default_rng(seed)
    priv = df[df[group_col].astype(str) == str(priv_code)]
    under = df[df[group_col].astype(str) == str(under_code)]

    yt_p, yp_p = priv[y_true_col].values, priv[y_pred_col].values
    yt_u, yp_u = under[y_true_col].values, under[y_pred_col].values
    n_p, n_u = len(yt_p), len(yt_u)

    sp_boot, tpr_boot, fpr_boot = [], [], []
    if n_p > 0 and n_u > 0:
        for _ in range(n_boot):
            idx_p = rng.integers(0, n_p, n_p)
            idx_u = rng.integers(0, n_u, n_u)
            pp, tp, fp = _rates(yt_p[idx_p], yp_p[idx_p])
            pu, tu, fu = _rates(yt_u[idx_u], yp_u[idx_u])
            sp_boot.append(pu - pp)
            tpr_boot.append(tu - tp)
            fpr_boot.append(fu - fp)

    def ci(vals):
        arr = np.array(vals, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return (np.nan, np.nan)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {"SP_ci": ci(sp_boot), "TPR_ci": ci(tpr_boot), "FPR_ci": ci(fpr_boot)}


def metric_threshold(metric_key, priv_pos_rate):
    """Return (green_cutoff, red_cutoff) for a metric. For statistical parity,
    the green cutoff approximates the four-fifths (80%) adverse-impact rule
    translated into an absolute-difference threshold using the privileged
    group's observed positive rate. For TPR/FPR diff, no equivalent
    regulatory rule exists, so fixed literature-heuristic cutoffs are used."""
    if metric_key == "SP_diff" and priv_pos_rate and not np.isnan(priv_pos_rate):
        green = 0.2 * priv_pos_rate
        return green, green * 1.75
    return HEURISTIC_GREEN, HEURISTIC_RED


def status_color(value, ci_low, ci_high, green_cutoff, red_cutoff):
    """Color always reflects the magnitude of the point estimate (green/amber/
    red). Statistical significance is reported separately as a flag, not by
    flattening the color — a small gap can still be worth flagging as
    directional even when the CI hasn't ruled out zero."""
    v = abs(value)
    if v < green_cutoff:
        color = "#22c55e"
    elif v < red_cutoff:
        color = "#d97706"
    else:
        color = "#dc2626"
    inconclusive = not (np.isnan(ci_low) or np.isnan(ci_high)) and ci_low <= 0 <= ci_high
    return color, inconclusive


RADAR_COLORS = ["#c9a24b", "#3a6ea5", "#8a5fd6", "#4ade80", "#f87171"]


def make_radar(current_df, models, metric_keys=("SP_diff", "TPR_diff", "FPR_diff")):
    labels = {"SP_diff": "SP diff", "TPR_diff": "TPR diff", "FPR_diff": "FPR diff"}
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
                r=vals,
                theta=theta,
                name=model,
                line={"color": color, "width": 2},
                fill="toself",
                fillcolor=color,
                opacity=0.75,
            )
        )
    fig.update_layout(
        polar={
            "bgcolor": "rgba(0,0,0,0)",
            "radialaxis": {"visible": True, "color": "#6b7688", "gridcolor": "#263140"},
            "angularaxis": {"color": "#dde3ec", "gridcolor": "#263140"},
        },
        showlegend=True,
        legend={"orientation": "h", "y": -0.1, "font": {"color": "#dde3ec"}},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dde3ec", "family": "Inter, sans-serif"},
        height=380,
        margin=dict(l=60, r=60, t=30, b=40),
    )
    return fig


def make_gauge(value, baseline_value=None, color="#3a6ea5"):
    indicator_kwargs = dict(
        mode="gauge+number",
        value=abs(value),
        number={"valueformat": ".3f", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 0.15], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 0.03], "color": "rgba(34,197,94,0.10)"},
                {"range": [0.03, 0.07], "color": "rgba(217,119,6,0.10)"},
                {"range": [0.07, 0.15], "color": "rgba(220,38,38,0.10)"},
            ],
        },
    )
    if baseline_value is not None:
        indicator_kwargs["mode"] = "gauge+number+delta"
        indicator_kwargs["delta"] = {
            "reference": abs(baseline_value),
            "valueformat": ".3f",
            "decreasing": {"color": "#16a34a"},
            "increasing": {"color": "#dc2626"},
            "font": {"size": 16},
        }
    fig = go.Figure(go.Indicator(**indicator_kwargs))
    fig.update_layout(
        height=210,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dde3ec", "family": "Inter, sans-serif"},
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

metric_name = st.sidebar.selectbox("Metric", list(METRIC_CAPTIONS.keys()), index=2)
metric_key = METRIC_KEYS[metric_name]

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">Credit Risk &amp; Fairness Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Fairness Monitor — Credit Scoring</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Receives model predictions, true labels, and the protected attribute. '
    "Never sees the input features.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="snapshot-inline">Point-in-time comparison of two model runs — not a continuous production feed.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        f'<span class="badge">Privileged: {attr["priv_label"]}</span> '
        f'&nbsp; <span class="badge">Underprivileged: {attr["under_label"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(METRIC_CAPTIONS[metric_name])

    steps = ["Inject bias", "Monitor detects", "FairFinGAN cleans", "Monitor confirms"]
    active_flags = [True, True, dataset_choice != available_datasets[0], dataset_choice != available_datasets[0]]
    html = '<div class="step-track">'
    for step, active in zip(steps, active_flags):
        cls = "step active" if active else "step"
        html += f'<div class="{cls}"><div class="dot"></div><div class="label">{step}</div></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Compute results (point estimates + bootstrap CI + dynamic thresholds)
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
improved_count, total_count = 0, 0
if len(available_datasets) > 1:
    other_ds = available_datasets[1]
    for model in selected_models:
        base_row = results_df[(results_df["model"] == model) & (results_df["dataset"] == baseline_ds)]
        other_row = results_df[(results_df["model"] == model) & (results_df["dataset"] == other_ds)]
        if not base_row.empty and not other_row.empty:
            total_count += 1
            if abs(other_row[metric_key].values[0]) < abs(base_row[metric_key].values[0]):
                improved_count += 1

st.markdown(
    f'<div class="section-title">The monitor detects disparity — and regression — across {len(selected_models)} model'
    f'{"s" if len(selected_models) != 1 else ""}</div>',
    unsafe_allow_html=True,
)
if dataset_choice != baseline_ds and total_count > 0:
    st.caption(
        f"{improved_count} of {total_count} models improved on this metric after FairFinGAN cleaning; "
        f"{total_count - improved_count} regressed. Both are reported — this tool is built to catch either."
    )

# ----------------------------------------------------------------------------
# Gauges
# ----------------------------------------------------------------------------
current = results_df[results_df["dataset"] == dataset_choice]
baseline_current = results_df[results_df["dataset"] == baseline_ds]

if not current.empty:
    green_cut, red_cut = metric_threshold(metric_key, current["priv_pos_rate"].mean())
    cols = st.columns(len(selected_models)) if selected_models else []
    for col, model in zip(cols, selected_models):
        row = current[current["model"] == model]
        with col:
            with st.container(border=True):
                if not row.empty:
                    value = row[metric_key].values[0]
                    ci_low, ci_high = row[f"{metric_key.split('_')[0]}_ci"].values[0]
                    color, inconclusive = status_color(value, ci_low, ci_high, green_cut, red_cut)
                    st.markdown(f'<div class="status-bar" style="background:{color};"></div>', unsafe_allow_html=True)

                    tag_html = ""
                    if dataset_choice != baseline_ds:
                        base_row = baseline_current[baseline_current["model"] == model]
                        if not base_row.empty:
                            base_val = base_row[metric_key].values[0]
                            if abs(value) < abs(base_val):
                                tag_html = '<span class="best-tag">Improved</span>'
                            elif abs(value) > abs(base_val):
                                tag_html = '<span class="warn-tag">Regressed</span>'
                            else:
                                tag_html = '<span class="flat-tag">No change</span>'

                    ci_html = ""
                    if inconclusive:
                        ci_html = (
                            f' <span class="inconclusive-mark" title="95% CI [{ci_low:.3f}, {ci_high:.3f}] '
                            'crosses zero">not significant</span>'
                        )

                    st.markdown(
                        f'<span style="font-weight:600;">{model}</span>{tag_html}{ci_html}',
                        unsafe_allow_html=True,
                    )

                    baseline_value = None
                    if dataset_choice != baseline_ds:
                        base_row = baseline_current[baseline_current["model"] == model]
                        if not base_row.empty:
                            baseline_value = base_row[metric_key].values[0]
                    st.plotly_chart(
                        make_gauge(value, baseline_value=baseline_value, color=color),
                        width="stretch",
                    )
                else:
                    st.info(f"No data for {model}.")

    threshold_note = (
        f"Green cutoff ({green_cut:.3f}) follows the four-fifths (80%) adverse-impact rule, translated to an "
        "absolute gap using the privileged group's observed rate."
        if metric_key == "SP_diff"
        else f"{HEURISTIC_GREEN:.2f} / {HEURISTIC_RED:.2f} are literature-heuristic cutoffs — no metric-specific "
        "regulatory standard exists for this one."
    )
    with st.expander("Methodology: thresholds & significance"):
        st.caption(threshold_note + " See *Limitations and scope* below for the full caveat.")
        st.caption(
            "\"not significant\" next to a model means the 95% bootstrap confidence interval for that gap "
            "crosses zero — the color still reflects the point estimate, but treat it as directional, not proven."
        )

# ----------------------------------------------------------------------------
# Multi-metric comparison (radar)
# ----------------------------------------------------------------------------
if not current.empty and len(selected_models) >= 1:
    st.markdown('<div class="section-title">Multi-metric comparison</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.plotly_chart(make_radar(current, selected_models), width="stretch")
        st.caption("Smaller shape = less disparity across all three metrics at once. Values are absolute gaps.")

# ----------------------------------------------------------------------------
# Results table + download
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">Results table</div>', unsafe_allow_html=True)
with st.container(border=True):
    display_df = results_df.copy()
    for col in ["SP_diff", "TPR_diff", "FPR_diff"]:
        display_df[col] = display_df[col].round(4)

    rows_html = ""
    for _, r in display_df.iterrows():
        ci_low, ci_high = r[f"{metric_key.split('_')[0]}_ci"]
        g_cut, r_cut = metric_threshold(metric_key, r["priv_pos_rate"])
        dot_color, _inconclusive = status_color(r[metric_key], ci_low, ci_high, g_cut, r_cut)
        rows_html += (
            "<tr>"
            f"<td><span class='rt-dot' style='background:{dot_color};'></span>{r['model']}</td>"
            f"<td>{r['dataset']}</td>"
            f"<td>{r['SP_diff']:.4f}</td>"
            f"<td>{r['TPR_diff']:.4f}</td>"
            f"<td>{r['FPR_diff']:.4f}</td>"
            "</tr>"
        )
    table_html = (
        "<table class='rt-table'><thead><tr>"
        "<th>Model</th><th>Dataset</th><th>SP diff</th><th>TPR diff</th><th>FPR diff</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(f"Row color reflects the selected metric ({metric_name}), including the confidence-interval check.")

    csv_bytes = display_df.drop(columns=["priv_pos_rate", "SP_ci", "TPR_ci", "FPR_ci"]).to_csv(index=False).encode("utf-8")
    st.download_button("Download report (CSV)", csv_bytes, file_name="fairness_report.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# SHAP proxy check
# ----------------------------------------------------------------------------
if shap_df is not None:
    with st.expander("View proxy check (SHAP)"):
        st.dataframe(shap_df, width="stretch", hide_index=True)
        st.caption("Features with high importance and high correlation with the protected attribute are proxy candidates.")
elif custom_data:
    with st.expander("View proxy check (SHAP)"):
        st.info("SHAP proxy check is only available for the preloaded experiment data, not for uploaded CSVs.")

# ----------------------------------------------------------------------------
# Limitations — stated explicitly rather than left implicit
# ----------------------------------------------------------------------------
with st.expander("Limitations and scope"):
    st.markdown(
        """
- **Bias is synthetically injected** (15% label flipping), not organically occurring. This demonstrates the
  monitor's sensitivity to a known, controlled pattern — it has not been validated against real-world,
  naturally-arising bias, which is typically subtler and structurally different.
- **This is a point-in-time comparison**, not continuous production monitoring. It compares two static
  datasets (biased vs. synthetic); it does not track drift over time.
- **Thresholds are a practical reference, not a legal standard.** The four-fifths rule (statistical parity)
  comes from US employment discrimination law, not lending regulation. TPR/FPR diff cutoffs are literature
  heuristics with no official regulatory backing.
- **Confidence intervals are bootstrap estimates** (200 resamples) computed from the available predictions,
  not a substitute for a full statistical power analysis.
- **No access control, audit trail, or versioning.** This is a proof-of-concept dashboard, not a
  production model-risk-management system.
        """
    )
