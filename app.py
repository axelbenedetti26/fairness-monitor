import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fairness Monitor — Credit Scoring", page_icon="⚖️", layout="wide")

# ----------------------------------------------------------------------------
# Styling — management-dashboard look: dark navy bg, white cards, icon headers
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .stApp { background: #16293d; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* page masthead (dark strip) */
    .masthead { margin-bottom: 1.6rem; }
    .masthead h1 {
        font-size: 2.1rem; font-weight: 800; color: #ffffff;
        margin: 0; letter-spacing: 0.01em; text-transform: uppercase;
    }
    .masthead .sub { font-size: 0.95rem; color: #9fb0c3; margin-top: 0.2rem; letter-spacing: 0.08em; }

    /* white card */
    .card {
        background: #ffffff; border-radius: 6px; padding: 0;
        margin-bottom: 1.3rem; overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.18);
    }
    .card-head {
        display: flex; align-items: center; gap: 0.8rem;
        padding: 1rem 1.3rem; border-bottom: 1px solid #eef1f4;
    }
    .card-icon {
        width: 34px; height: 34px; border-radius: 50%;
        background: #2d6a9f; color: #fff;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .card-title { font-size: 1.15rem; font-weight: 700; color: #1f2d3d; }
    .card-body { padding: 1.2rem 1.3rem; }

    .lede { font-size: 0.98rem; line-height: 1.65; color: #44515f; }

    /* contents list inside a white card */
    .toc-row {
        display: flex; align-items: baseline; gap: 1rem;
        padding: 0.85rem 0; border-bottom: 1px solid #eef1f4;
    }
    .toc-row:last-child { border-bottom: none; }
    .toc-badge {
        width: 26px; height: 26px; border-radius: 5px; flex-shrink: 0;
        background: #2d6a9f; color: #fff; font-size: 0.8rem; font-weight: 700;
        display: flex; align-items: center; justify-content: center;
    }
    .toc-text { flex: 1; }
    .toc-h { font-size: 1rem; font-weight: 700; color: #1f2d3d; margin-bottom: 0.1rem; }
    .toc-d { font-size: 0.88rem; color: #5d6b7a; line-height: 1.4; }

    /* status chips */
    .chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; }
    .chip {
        padding: 0.5rem 1rem; border-radius: 5px; font-size: 0.9rem;
        font-weight: 600; color: #fff;
    }
    .chip.blue { background: #2d6a9f; }
    .chip.amber { background: #e0a93b; }

    .section-title { font-size: 1.5rem; font-weight: 800; color: #ffffff; margin: 0 0 0.3rem 0; }
    .section-sub { color: #9fb0c3; font-size: 0.95rem; margin-bottom: 1.4rem; }

    .pill { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; border-radius: 5px; padding: 0.25rem 0.6rem; }
    .pill.ok { color: #fff; background: #4caf72; }
    .pill.watch { color: #3a2e10; background: #e0a93b; }
    .pill.breach { color: #fff; background: #e05a5a; }
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
    try: return pd.read_csv(SHAP_PATH)
    except FileNotFoundError: return None
@st.cache_data
def load_allcond():
    try: return pd.read_csv(ALLCOND_PATH)
    except FileNotFoundError: return None

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
            <h1>Fairness Monitor</h1>
            <div class="sub">CREDIT SCORING · MODEL GOVERNANCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 2], gap="medium")

    with c1:
        st.markdown(
            """
            <div class="card">
                <div class="card-head">
                    <div class="card-icon">●</div>
                    <div class="card-title">What this is</div>
                </div>
                <div class="card-body">
                    <div class="lede">
                    This console examines how a credit scoring model treats demographic groups, before and
                    after bias mitigation. It works the way an auditor would: it sees only the model's
                    predictions and the protected attribute, never the raw features. The goal is not a single
                    verdict, but to show where different fairness measures and different mitigation methods
                    disagree.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="card">
                <div class="card-head">
                    <div class="card-icon">◆</div>
                    <div class="card-title">At a glance</div>
                </div>
                <div class="card-body">
                    <div class="chip-row">
                        <span class="chip blue">9 models</span>
                        <span class="chip blue">2 methods</span>
                        <span class="chip amber">2 metrics</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="card">
            <div class="card-head">
                <div class="card-icon">≡</div>
                <div class="card-title">Sections</div>
            </div>
            <div class="card-body">
                <div class="toc-row">
                    <div class="toc-badge">I</div>
                    <div class="toc-text">
                        <div class="toc-h">Model</div>
                        <div class="toc-d">A single model, read in detail: each fairness metric before and after mitigation, with a plain reading of what it means.</div>
                    </div>
                </div>
                <div class="toc-row">
                    <div class="toc-badge">II</div>
                    <div class="toc-text">
                        <div class="toc-h">Compare</div>
                        <div class="toc-d">Models placed side by side and ranked, from fairest to least fair on a chosen metric.</div>
                    </div>
                </div>
                <div class="toc-row">
                    <div class="toc-badge">III</div>
                    <div class="toc-text">
                        <div class="toc-h">Overview</div>
                        <div class="toc-d">The full picture: two mitigation methods across two fairness metrics, and where they pull in opposite directions.</div>
                    </div>
                </div>
                <div class="toc-row">
                    <div class="toc-badge">IV</div>
                    <div class="toc-text">
                        <div class="toc-h">Methodology</div>
                        <div class="toc-d">The dataset, the injected bias, the metrics and thresholds, and how the experiment was run.</div>
                    </div>
                </div>
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
