import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Fairness Monitor — Credit Scoring",
    page_icon="⚖️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@1,600&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .eyebrow {
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.14em;
        color: #c9a24b; text-transform: uppercase; margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 2.6rem; font-weight: 700; color: #f2f5f9;
        margin: 0 0 0.6rem 0; line-height: 1.12;
    }
    .hero-title .accent {
        font-family: 'Fraunces', serif; font-style: italic;
        font-weight: 600; color: #c9a24b;
    }
    .hero-subtitle {
        color: #8a96a8; font-size: 1.05rem; line-height: 1.55;
        max-width: 640px; margin-bottom: 2.2rem;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #eef1f6;
        margin: 2.2rem 0 0.4rem 0;
    }
    .section-sub { color: #7c8798; font-size: 0.92rem; margin-bottom: 1.3rem; }

    .flow-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0 2rem 0; }
    .flow-card {
        flex: 1; min-width: 150px;
        border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
        padding: 1.1rem 1.2rem; background: rgba(255,255,255,0.02);
    }
    .flow-step {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #c9a24b; margin-bottom: 0.5rem;
    }
    .flow-name { font-size: 1rem; font-weight: 600; color: #eef1f6; margin-bottom: 0.35rem; }
    .flow-desc { font-size: 0.85rem; color: #8a96a8; line-height: 1.4; }

    .tag-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .tag {
        padding: 0.45rem 0.9rem; border-radius: 6px;
        border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.02);
        font-size: 0.85rem; color: #b7c0cd;
    }
    .tag b { color: #eef1f6; font-weight: 600; }
    .tag.priv { border-color: rgba(201,162,75,0.4); }
    .tag.priv b { color: #d9bd7a; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        border-color: rgba(255,255,255,0.08) !important;
    }

    .pill {
        font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em;
        text-transform: uppercase; border-radius: 5px; padding: 0.25rem 0.6rem;
    }
    .pill.ok { color: #1a2e22; background: #4ade80; }
    .pill.watch { color: #2e2410; background: #d9bd7a; }
    .pill.breach { color: #2e1414; background: #f87171; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
PRED_PATH = "monitor_predictions.csv"
SHAP_PATH = "shap_importance.csv"
ALLCOND_PATH = "all_conditions_final.csv"
RW_PATH = "reweighing_results_holdout_v2.csv"
N_BOOT = 200

ATTR_CONFIG = {
    "Sex": {"column": "sex", "priv_code": 2, "priv_label": "Women", "under_code": 1, "under_label": "Men"},
    "Age": {"column": "age_group", "priv_code": 1, "priv_label": "Core age (25-65)", "under_code": 0, "under_label": "Edge age (<25 or >65)"},
}

MODEL_ORDER = ["LogReg", "SVM", "kNN", "MLP", "RandomForest", "GradBoost", "XGBoost", "LightGBM", "CatBoost"]
MODEL_DESC = {
    "LogReg": "Logistic regression — a linear model, the classic credit-scoring baseline.",
    "SVM": "Linear support vector machine — another linear classifier.",
    "kNN": "k-nearest neighbors — classifies by the closest examples in the data.",
    "MLP": "A small neural network with two hidden layers.",
    "RandomForest": "An ensemble of decision trees trained on random subsets.",
    "GradBoost": "Gradient boosting — trees built sequentially to correct each other.",
    "XGBoost": "An optimized gradient-boosting library, strong on tabular data.",
    "LightGBM": "A fast gradient-boosting framework.",
    "CatBoost": "A gradient-boosting library designed to handle categorical data.",
}

# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
@st.cache_data
def load_predictions():
    return pd.read_csv(PRED_PATH)

@st.cache_data
def load_shap():
    try:
        return pd.read_csv(SHAP_PATH)
    except FileNotFoundError:
        return None

@st.cache_data
def load_allcond():
    try:
        return pd.read_csv(ALLCOND_PATH)
    except FileNotFoundError:
        return None

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Fairness Monitor")
st.sidebar.caption("Credit scoring · APW thesis")

page = st.sidebar.radio(
    "Navigate",
    ["Home", "Model", "Compare", "Overview", "Methodology"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# ============================================================================
# PAGE: HOME
# ============================================================================
if page == "Home":
    st.markdown('<div class="eyebrow">Credit Risk &amp; Fairness Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Fairness Monitor <span class="accent">Console</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Checks whether a credit scoring model treats demographic groups '
        "fairly, before and after bias mitigation. Reads predictions and the protected attribute only — "
        "never the raw input features.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-row">
            <div class="flow-card">
                <div class="flow-step">Step 1</div>
                <div class="flow-name">Model</div>
                <div class="flow-desc">Pick a model and read its fairness metrics, before and after mitigation.</div>
            </div>
            <div class="flow-card">
                <div class="flow-step">Step 2</div>
                <div class="flow-name">Compare</div>
                <div class="flow-desc">Line models up side by side and rank them from fairest to least fair.</div>
            </div>
            <div class="flow-card">
                <div class="flow-step">Step 3</div>
                <div class="flow-name">Overview</div>
                <div class="flow-desc">See the full picture: both mitigation methods across both metrics.</div>
            </div>
            <div class="flow-card">
                <div class="flow-step">Reference</div>
                <div class="flow-name">Methodology</div>
                <div class="flow-desc">Dataset, metrics, thresholds and how the experiment was run.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Use the menu on the left to move between sections. Start with **Model**.")

# ============================================================================
# PAGE: MODEL (placeholder — Fase 2)
# ============================================================================
elif page == "Model":
    st.markdown('<div class="section-title">Model audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick one model and read its fairness metrics in detail.</div>', unsafe_allow_html=True)
    st.info("Coming in the next step.")

# ============================================================================
# PAGE: COMPARE (placeholder)
# ============================================================================
elif page == "Compare":
    st.markdown('<div class="section-title">Compare models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Line models up side by side.</div>', unsafe_allow_html=True)
    st.info("Coming in a later step.")

# ============================================================================
# PAGE: OVERVIEW (placeholder)
# ============================================================================
elif page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Both methods, both metrics, at a glance.</div>', unsafe_allow_html=True)
    st.info("Coming in a later step.")

# ============================================================================
# PAGE: METHODOLOGY (placeholder)
# ============================================================================
elif page == "Methodology":
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the experiment was built.</div>', unsafe_allow_html=True)
    st.info("Coming in a later step.")
