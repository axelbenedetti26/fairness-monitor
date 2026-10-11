import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fairness Monitor — Credit Scoring", page_icon="⚖️", layout="wide")

# ----------------------------------------------------------------------------
# Styling — light, minimal: white/soft-grey bg, blue text, compact cards
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { background: #f6f8fb; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-h { font-size: 1.5rem; font-weight: 700; color: #1f3044; margin: 0; }
    .page-sub { font-size: 0.82rem; color: #8795a6; letter-spacing: 0.06em;
                text-transform: uppercase; margin: 0.15rem 0 1.6rem 0; }

    .mini {
        background: #ffffff; border: 1px solid #e7ecf2; border-radius: 8px;
        padding: 1rem 1.2rem; height: 100%;
    }
    .mini-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
                  color: #8795a6; margin-bottom: 0.4rem; }
    .mini-value { font-size: 1.6rem; font-weight: 700; color: #2b5c8a; line-height: 1.1; }
    .mini-note { font-size: 0.8rem; color: #8795a6; margin-top: 0.2rem; }

    .block {
        background: #ffffff; border: 1px solid #e7ecf2; border-radius: 8px;
        padding: 1.2rem 1.4rem; margin-top: 1.1rem;
    }
    .block-h { font-size: 0.95rem; font-weight: 700; color: #1f3044; margin-bottom: 0.9rem; }
    .lede { font-size: 0.9rem; line-height: 1.6; color: #52606e; }

    .row { display: flex; gap: 0.9rem; padding: 0.7rem 0; border-top: 1px solid #f0f3f7; }
    .row:first-of-type { border-top: none; }
    .row-n { color: #2b5c8a; font-weight: 700; font-size: 0.9rem; min-width: 1.5rem; }
    .row-h { font-size: 0.9rem; font-weight: 600; color: #1f3044; }
    .row-d { font-size: 0.82rem; color: #7a8694; margin-top: 0.1rem; line-height: 1.4; }

    .section-title { font-size: 1.4rem; font-weight: 700; color: #1f3044; margin: 0 0 0.2rem 0; }
    .section-sub { color: #8795a6; font-size: 0.88rem; margin-bottom: 1.3rem; }

    .pill { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; border-radius: 4px; padding: 0.22rem 0.55rem; }
    .pill.ok { color: #fff; background: #3f9b63; }
    .pill.watch { color: #4a3a10; background: #e0a93b; }
    .pill.breach { color: #fff; background: #d85b5b; }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    try: return pd.read_csv(SHAP_PATH)
    except FileNotFoundError: return None
@st.cache_data
def load_allcond():
    try: return pd.read_csv(ALLCOND_PATH)
    except FileNotFoundError: return None

st.sidebar.markdown("### Fairness Monitor")
st.sidebar.caption("Credit scoring · APW thesis")
page = st.sidebar.radio("Navigate", ["Home", "Model", "Compare", "Overview", "Methodology"], label_visibility="collapsed")
st.sidebar.markdown("---")

# ============================================================================
# HOME
# ============================================================================
if page == "Home":
    st.markdown('<div class="page-h">Fairness Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Credit scoring · Model governance</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3, gap="small")
    with a:
        st.markdown('<div class="mini"><div class="mini-label">Models</div><div class="mini-value">9</div><div class="mini-note">linear, trees, kNN, neural net</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="mini"><div class="mini-label">Methods</div><div class="mini-value">2</div><div class="mini-note">FairFinGAN · reweighting</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="mini"><div class="mini-label">Metrics</div><div class="mini-value">2</div><div class="mini-note">statistical parity · equalized odds</div></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="block">
            <div class="block-h">What this is</div>
            <div class="lede">A fairness check for credit scoring models. It reads only the model's
            predictions and the protected attribute, never the raw features, and shows where different
            fairness measures and mitigation methods disagree.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="block">
            <div class="block-h">Sections</div>
            <div class="row"><div class="row-n">I</div><div><div class="row-h">Model</div><div class="row-d">One model in detail, before and after mitigation.</div></div></div>
            <div class="row"><div class="row-n">II</div><div><div class="row-h">Compare</div><div class="row-d">Models side by side, ranked by fairness.</div></div></div>
            <div class="row"><div class="row-n">III</div><div><div class="row-h">Overview</div><div class="row-d">Both methods across both metrics.</div></div></div>
            <div class="row"><div class="row-n">IV</div><div><div class="row-h">Methodology</div><div class="row-d">Data, metrics and how it was run.</div></div></div>
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
    st.markdown('<div class="section-sub">Models side by side.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")
elif page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Two methods, two metrics.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")
elif page == "Methodology":
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the experiment was built.</div>', unsafe_allow_html=True)
    st.caption("Coming in a later step.")
