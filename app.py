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
# Custom styling (fonts, hero text, cards, animated step tracker)
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #6b8cae;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #e8edf4;
        margin-bottom: 0.2rem;
        line-height: 1.15;
        border-left: 4px solid #3a6ea5;
        padding-left: 0.7rem;
    }

    .hero-subtitle {
        color: #8a96a8;
        font-size: 0.98rem;
        margin-bottom: 1.6rem;
        padding-left: 0.9rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        color: #dde3ec;
    }

    /* cards: quiet fade in on load, subtle lift on hover (no bounce/tilt) */
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

    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(58, 110, 165, 0.5); }
        70%  { box-shadow: 0 0 0 6px rgba(58, 110, 165, 0); }
        100% { box-shadow: 0 0 0 0 rgba(58, 110, 165, 0); }
    }

    /* step tracker */
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
        transition: all 0.3s ease;
    }
    .step.active .dot {
        background: #3a6ea5;
        border-color: #3a6ea5;
        animation: pulse 2.4s infinite;
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
        padding: 0.3rem 0.75rem;
        border-radius: 4px;
        background: rgba(58, 110, 165, 0.14);
        border: 1px solid rgba(58, 110, 165, 0.3);
        color: #a9c2dc;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.6rem;
        letter-spacing: 0.01em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_PRED_PATH = "monitor_predictions.csv"
DEFAULT_SHAP_PATH = "shap_importance.csv"

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


def fairness_metrics(df, y_true_col, y_pred_col, group_col, priv_code, under_code):
    """Statistical parity, TPR and FPR differences between privileged and
    underprivileged groups. Positive value = underprivileged group is worse off."""
    results = {}
    for label, code in [("priv", priv_code), ("under", under_code)]:
        sub = df[df[group_col] == code]
        y_true = sub[y_true_col]
        y_pred = sub[y_pred_col]

        pos_rate = (y_pred == 1).mean() if len(sub) else np.nan

        pos_mask = y_true == 1
        neg_mask = y_true == 0
        tpr = (y_pred[pos_mask] == 1).mean() if pos_mask.sum() else np.nan
        fpr = (y_pred[neg_mask] == 1).mean() if neg_mask.sum() else np.nan

        results[f"pos_rate_{label}"] = pos_rate
        results[f"TPR_{label}"] = tpr
        results[f"FPR_{label}"] = fpr

    return {
        "SP_diff": results["pos_rate_under"] - results["pos_rate_priv"],
        "TPR_diff": results["TPR_under"] - results["TPR_priv"],
        "FPR_diff": results["FPR_under"] - results["FPR_priv"],
    }


def gauge_color(value):
    v = abs(value)
    if v < 0.03:
        return "#22c55e"
    if v < 0.07:
        return "#f59e0b"
    return "#ef4444"


def make_gauge(value, title, baseline_value=None):
    color = gauge_color(value)
    indicator_kwargs = dict(
        mode="gauge+number",
        value=abs(value),
        number={"valueformat": ".3f", "font": {"size": 34}},
        gauge={
            "axis": {"range": [0, 0.15], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 0.03], "color": "rgba(34,197,94,0.12)"},
                {"range": [0.03, 0.07], "color": "rgba(217,119,6,0.12)"},
                {"range": [0.07, 0.15], "color": "rgba(220,38,38,0.12)"},
            ],
        },
    )
    # When comparing against the biased baseline, show a real delta with an
    # up/down arrow — smaller disparity than baseline is an improvement (green).
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
        height=230,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dde3ec", "family": "Inter, sans-serif"},
        title={"text": title, "font": {"size": 16}},
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
# Compute results
# ----------------------------------------------------------------------------
rows = []
for model in selected_models:
    for ds in available_datasets:
        sub = monitor_data[(monitor_data["model"] == model) & (monitor_data["dataset"] == ds)]
        if sub.empty:
            continue
        m = fairness_metrics(sub, "y_true", "y_pred", attr["column"], attr["priv_code"], attr["under_code"])
        rows.append({"model": model, "dataset": ds, **m})

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
    f'<div class="section-title">The monitor detects disparity in {len(selected_models)} model'
    f'{"s" if len(selected_models) != 1 else ""}</div>',
    unsafe_allow_html=True,
)
if dataset_choice != baseline_ds and total_count > 0:
    st.caption(f"{improved_count} of {total_count} models improved on this metric after FairFinGAN cleaning.")

# ----------------------------------------------------------------------------
# Gauges
# ----------------------------------------------------------------------------
current = results_df[results_df["dataset"] == dataset_choice]
baseline_current = results_df[results_df["dataset"] == baseline_ds]
if not current.empty:
    cols = st.columns(len(selected_models)) if selected_models else []
    for col, model in zip(cols, selected_models):
        row = current[current["model"] == model]
        with col:
            with st.container(border=True):
                if not row.empty:
                    value = row[metric_key].values[0]
                    baseline_value = None
                    if dataset_choice != baseline_ds:
                        base_row = baseline_current[baseline_current["model"] == model]
                        if not base_row.empty:
                            baseline_value = base_row[metric_key].values[0]
                    st.plotly_chart(
                        make_gauge(value, model, baseline_value=baseline_value),
                        use_container_width=True,
                    )
                else:
                    st.info(f"No data for {model}.")

# ----------------------------------------------------------------------------
# Results table + download
# ----------------------------------------------------------------------------
st.markdown('<div class="section-title">Results table</div>', unsafe_allow_html=True)
with st.container(border=True):
    display_df = results_df.copy()
    for col in ["SP_diff", "TPR_diff", "FPR_diff"]:
        display_df[col] = display_df[col].round(4)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download report (CSV)", csv_bytes, file_name="fairness_report.csv", mime="text/csv")

# ----------------------------------------------------------------------------
# SHAP proxy check
# ----------------------------------------------------------------------------
if shap_df is not None:
    with st.expander("View proxy check (SHAP)"):
        st.dataframe(shap_df, use_container_width=True, hide_index=True)
        st.caption("Features with high importance and high correlation with the protected attribute are proxy candidates.")
elif custom_data:
    with st.expander("View proxy check (SHAP)"):
        st.info("SHAP proxy check is only available for the preloaded experiment data, not for uploaded CSVs.")
