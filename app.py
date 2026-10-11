import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fairness Monitor — Credit Scoring", page_icon="⚖️", layout="wide")

# ----------------------------------------------------------------------------
# Styling — editorial / technical report
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .masthead {
        border-bottom: 1px solid rgba(255,255,255,0.12);
        padding-bottom: 1.4rem; margin-bottom: 2.2rem;
    }
    .masthead .kicker {
        font-size: 0.72rem; letter-spacing: 0.18em; text-transform: uppercase;
        color: #8a96a8; margin-bottom: 1rem;
    }
    .masthead h1 {
        font-family: 'Newsreader', serif; font-weight: 500;
        font-size: 3rem; line-height: 1.05; color: #f2f5f9;
        margin: 0 0 1rem 0; letter-spacing: -0.01em;
    }
    .masthead .standfirst {
        font-family: 'Newsreader', serif; font-size: 1.25rem; font-weight: 400;
        line-height: 1.5; color: #aab4c2; max-width: 680px; font-style: italic;
    }

    .lede {
        font-size: 1rem; line-height: 1.7; color: #c3cbd6;
        max-width: 680px; margin-bottom: 2.6rem;
    }

    .contents-label {
        font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
        color: #8a96a8; margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 0.6rem;
    }
    .toc-item {
        display: flex; align-items: baseline; gap: 1.2rem;
        padding: 0.9rem 0; border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .toc-num {
        font-family: 'Newsreader', serif; font-size: 1.4rem; color: #5c6779;
        min-width: 2rem; font-style: italic;
    }
    .toc-body { flex: 1; }
    .toc-name { font-size: 1.05rem; font-weight: 600; color: #eef1f6; margin-bottom: 0.2rem; }
    .toc-desc { font-size: 0.9rem; color: #8a96a8; line-height: 1.45; }

    .section-title {
        font-family: 'Newsreader', serif; font-size: 1.9rem; font-weight: 500;
        color: #eef1f6; margin: 0 0 0.3rem 0;
    }
    .section-sub { color: #8a96a8; font-size: 0.95rem; margin-bottom: 1.6rem; }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 8px !important; border-color: rgba(255,255,255,0.08) !important;
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
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.markdown("### Fairness Monitor")
st.sidebar.caption("Credit scoring · APW thesis")
page = st.sidebar.radio("Navigate", ["Home", "Model", "Compare", "Overview", "Methodology"], label_visibility="collapsed")
st.sidebar.markdown("---")

# ============================================================================
# HOME
# ============================================================================
if page == "Home":
    st.markdown(
        """
        <div class="masthead">
            <div class="kicker">Credit Risk · Model Governance · Fairness</div>
            <h1>A monitor for fairness<br>in credit scoring models</h1>
            <div class="standfirst">When a model decides who gets credit, fairness is not one number.
            It is several, and they do not always agree.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="lede">
        This console examines how a credit scoring model treats demographic groups, before and after
        bias mitigation. It works the way an auditor would: it sees only the model's predictions and the
        protected attribute, never the raw features the model was trained on. The point is not to produce
        a single verdict, but to make visible where different fairness measures, and different mitigation
        methods, disagree.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="contents-label">Contents</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="toc-item">
            <div class="toc-num">I</div>
            <div class="toc-body">
                <div class="toc-name">Model</div>
                <div class="toc-desc">A single model, read in detail: each fairness metric before and after mitigation, with a plain reading of what it means.</div>
            </div>
        </div>
        <div class="toc-item">
            <div class="toc-num">II</div>
            <div class="toc-body">
                <div class="toc-name">Compare</div>
                <div class="toc-desc">Models placed side by side and ranked, from the fairest to the least fair on a chosen metric.</div>
            </div>
        </div>
        <div class="toc-item">
            <div class="toc-num">III</div>
            <div class="toc-body">
                <div class="toc-name">Overview</div>
                <div class="toc-desc">The full picture: two mitigation methods across two fairness metrics, and where they pull in opposite directions.</div>
            </div>
        </div>
        <div class="toc-item">
            <div class="toc-num">IV</div>
            <div class="toc-body">
                <div class="toc-name">Methodology</div>
                <div class="toc-desc">The dataset, the injected bias, the metrics and thresholds, and how the experiment was run.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif page == "Model":
    st.markdown('<div class="section-title">Model audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick one model and read its fairness metrics in detail.</div>', unsafe_allow_html=True)
    st.caption("Coming in the next step.")

elif page == "Compare":
    st.markdown('<div class="section-title">Compare models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Models placed side by side.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")

elif page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Two methods, two metrics, at a glance.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")

elif page == "Methodology":
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the experiment was built.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")
