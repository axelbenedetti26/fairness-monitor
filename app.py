import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Fairness Monitor — Credit Scoring", page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    .stApp { background: #f6f8fb; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .page-h { font-size: 1.5rem; font-weight: 700; color: #1f3044; margin: 0; }
    .page-sub { font-size: 0.82rem; color: #8795a6; letter-spacing: 0.06em; text-transform: uppercase; margin: 0.15rem 0 1.6rem 0; }

    .mini { background: #ffffff; border: 1px solid #e7ecf2; border-radius: 8px; padding: 1rem 1.2rem; height: 100%; }
    .mini-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #8795a6; margin-bottom: 0.4rem; }
    .mini-value { font-size: 1.6rem; font-weight: 700; color: #2b5c8a; line-height: 1.1; }
    .mini-note { font-size: 0.8rem; color: #8795a6; margin-top: 0.2rem; }

    .block { background: #ffffff; border: 1px solid #e7ecf2; border-radius: 8px; padding: 1.2rem 1.4rem; margin-top: 1.1rem; }
    .block-h { font-size: 0.95rem; font-weight: 700; color: #1f3044; margin-bottom: 0.9rem; }
    .lede { font-size: 0.9rem; line-height: 1.6; color: #52606e; }

    .row { display: flex; gap: 0.9rem; padding: 0.7rem 0; border-top: 1px solid #f0f3f7; }
    .row:first-of-type { border-top: none; }
    .row-n { color: #2b5c8a; font-weight: 700; font-size: 0.9rem; min-width: 1.5rem; }
    .row-h { font-size: 0.9rem; font-weight: 600; color: #1f3044; }
    .row-d { font-size: 0.82rem; color: #7a8694; margin-top: 0.1rem; line-height: 1.4; }

    .section-title { font-size: 1.4rem; font-weight: 700; color: #1f3044; margin: 0 0 0.2rem 0; }
    .section-sub { color: #8795a6; font-size: 0.88rem; margin-bottom: 1.3rem; }

    /* metric card */
    .mcard { background:#fff; border:1px solid #e7ecf2; border-radius:8px; padding:1.1rem 1.2rem; height:100%; }
    .mcard-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:0.2rem; }
    .mcard-name { font-size:0.9rem; font-weight:700; color:#1f3044; }
    .mcard-vals { display:flex; align-items:baseline; gap:0.5rem; margin:0.5rem 0 0.2rem 0; }
    .mcard-after { font-size:1.7rem; font-weight:700; color:#2b5c8a; }
    .mcard-before { font-size:0.85rem; color:#8795a6; text-decoration:line-through; }
    .mcard-delta { font-size:0.8rem; font-weight:600; }
    .mcard-delta.good { color:#3f9b63; }
    .mcard-delta.bad { color:#d85b5b; }
    .mcard-exp { font-size:0.8rem; color:#7a8694; line-height:1.45; margin-top:0.6rem; }

    .pill { font-size:0.62rem; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; border-radius:4px; padding:0.2rem 0.5rem; }
    .pill.ok { color:#fff; background:#3f9b63; }
    .pill.watch { color:#4a3a10; background:#e0a93b; }
    .pill.breach { color:#fff; background:#d85b5b; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRED_PATH = "monitor_predictions.csv"
SHAP_PATH = "shap_importance.csv"
ALLCOND_PATH = "all_conditions_final.csv"
RW_PATH = "reweighing_results_holdout_v2.csv"

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
    "RandomForest": "An ensemble of decision trees on random subsets.",
    "GradBoost": "Gradient boosting — trees built sequentially to correct each other.",
    "XGBoost": "An optimized gradient-boosting library, strong on tabular data.",
    "LightGBM": "A fast gradient-boosting framework.",
    "CatBoost": "A gradient-boosting library designed for categorical data.",
}
METRICS = [
    ("SP",  "Statistical parity",  "Gap in how often each group is predicted to default. 0 = both groups flagged at the same rate."),
    ("TPR", "TPR gap",             "Gap in correctly flagged true defaulters between groups. 0 = equal true-positive rates."),
    ("FPR", "FPR gap",             "Gap in non-defaulters wrongly flagged between groups. 0 = equal false-positive rates."),
]
GREEN, RED = 0.03, 0.07

@st.cache_data
def load_predictions(): return pd.read_csv(PRED_PATH)
@st.cache_data
def load_shap():
    try: return pd.read_csv(SHAP_PATH)
    except FileNotFoundError: return None

def _rates(y_true, y_pred):
    pos = y_pred.mean() if len(y_pred) else np.nan
    tpr = y_pred[y_true == 1].mean() if (y_true == 1).sum() else np.nan
    fpr = y_pred[y_true == 0].mean() if (y_true == 0).sum() else np.nan
    return pos, tpr, fpr

def model_metrics(df, col, priv, under):
    p = df[df[col] == priv]; u = df[df[col] == under]
    pp, pt, pf = _rates(p["y_true"].values, p["y_pred"].values)
    up, ut, uf = _rates(u["y_true"].values, u["y_pred"].values)
    return {"SP": up - pp, "TPR": ut - pt, "FPR": uf - pf}

def status_pill(val):
    d = abs(val)
    if d < GREEN: return "ok", "Within tolerance"
    if d < RED:   return "watch", "Monitor"
    return "breach", "Breach"

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
    a.markdown('<div class="mini"><div class="mini-label">Models</div><div class="mini-value">9</div><div class="mini-note">linear, trees, kNN, neural net</div></div>', unsafe_allow_html=True)
    b.markdown('<div class="mini"><div class="mini-label">Methods</div><div class="mini-value">2</div><div class="mini-note">FairFinGAN · reweighting</div></div>', unsafe_allow_html=True)
    c.markdown('<div class="mini"><div class="mini-label">Metrics</div><div class="mini-value">2</div><div class="mini-note">statistical parity · equalized odds</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="block"><div class="block-h">What this is</div><div class="lede">A fairness check for credit scoring models. It reads only the model\'s predictions and the protected attribute, never the raw features, and shows where different fairness measures and mitigation methods disagree.</div></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block"><div class="block-h">Sections</div>'
        '<div class="row"><div class="row-n">I</div><div><div class="row-h">Model</div><div class="row-d">One model in detail, before and after mitigation.</div></div></div>'
        '<div class="row"><div class="row-n">II</div><div><div class="row-h">Compare</div><div class="row-d">Models side by side, ranked by fairness.</div></div></div>'
        '<div class="row"><div class="row-n">III</div><div><div class="row-h">Overview</div><div class="row-d">Both methods across both metrics.</div></div></div>'
        '<div class="row"><div class="row-n">IV</div><div><div class="row-h">Methodology</div><div class="row-d">Data, metrics and how it was run.</div></div></div>'
        '</div>', unsafe_allow_html=True)

# ============================================================================
# MODEL
# ============================================================================
elif page == "Model":
    preds = load_predictions()
    st.markdown('<div class="section-title">Model audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Pick one model and read its fairness before and after mitigation.</div>', unsafe_allow_html=True)

    cc1, cc2 = st.columns([2, 1], gap="medium")
    model = cc1.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index("XGBoost"))
    attr_name = cc2.selectbox("Protected attribute", list(ATTR_CONFIG.keys()))
    attr = ATTR_CONFIG[attr_name]

    st.caption(MODEL_DESC.get(model, ""))

    orig = preds[(preds.model == model) & (preds.dataset == "original")]
    synt = preds[(preds.model == model) & (preds.dataset == "synthetic")]
    m_orig = model_metrics(orig, attr["column"], attr["priv_code"], attr["under_code"])
    m_synt = model_metrics(synt, attr["column"], attr["priv_code"], attr["under_code"])

    cols = st.columns(3, gap="small")
    for col, (key, label, expl) in zip(cols, METRICS):
        before = m_orig[key]; after = m_synt[key]
        delta = abs(after) - abs(before)          # negative = fairer
        cls, txt = status_pill(after)
        dgood = "good" if delta < 0 else "bad"
        dsign = "▼" if delta < 0 else "▲"
        with col:
            st.markdown(
                f"""
                <div class="mcard">
                    <div class="mcard-top">
                        <span class="mcard-name">{label}</span>
                        <span class="pill {cls}">{txt}</span>
                    </div>
                    <div class="mcard-vals">
                        <span class="mcard-after">{after:+.3f}</span>
                        <span class="mcard-before">{before:+.3f}</span>
                    </div>
                    <div class="mcard-delta {dgood}">{dsign} {abs(delta):.3f} vs original</div>
                    <div class="mcard-exp">{expl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # bar chart: original vs synthetic, absolute gaps
    fig = go.Figure()
    labels = [m[1] for m in METRICS]
    fig.add_trace(go.Bar(name="Original", x=labels, y=[abs(m_orig[k]) for k, _, _ in METRICS], marker_color="#9fb6cc"))
    fig.add_trace(go.Bar(name="After FairFinGAN", x=labels, y=[abs(m_synt[k]) for k, _, _ in METRICS], marker_color="#2b5c8a"))
    fig.update_layout(
        barmode="group", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#52606e", "family": "Inter"},
        legend={"orientation": "h", "y": 1.15},
        yaxis={"title": "absolute gap (closer to 0 = fairer)", "gridcolor": "#eef1f5"},
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.markdown('<div class="block-h" style="margin-top:1.4rem;">Before vs after mitigation</div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

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
