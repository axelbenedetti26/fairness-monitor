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
    .note { background:#fff8e6; border:1px solid #f0dfb0; border-radius:8px; padding:0.8rem 1rem; font-size:0.85rem; color:#5c4a1a; margin-top:1rem; line-height:1.5; }
    .concl { background:#ffffff; border-left:4px solid #2b5c8a; border-radius:6px; padding:0.8rem 1rem; font-size:0.88rem; color:#34414f; margin-top:1rem; line-height:1.55; }
    .mcard-delta.flat { color:#8795a6; }
    </style>
    """,
    unsafe_allow_html=True,
)

PRED_PATH = "monitor_predictions.csv"
SHAP_PATH = "shap_importance.csv"
ALLCOND_PATH = "all_conditions_final.csv"
RW_PATH = "reweighing_results_holdout_v2.csv"
DEMO_SEED = 1

ATTR_CONFIG = {
    "Sex": {"column": "sex", "priv_code": 2, "priv_label": "Women", "under_code": 1, "under_label": "Men"},
    "Age": {"column": "age_group", "priv_code": 1, "priv_label": "Core age (25-65)", "under_code": 0, "under_label": "Edge age (<25 or >65)"},
}
MODEL_ORDER = ["LogReg", "SVM", "kNN", "MLP", "RandomForest", "GradBoost", "XGBoost", "LightGBM", "CatBoost"]
RW_MODELS = ["LogReg", "RandomForest", "GradBoost", "XGBoost", "LightGBM", "CatBoost"]
MITIGATIONS = {"FairFinGAN": "synthetic", "Reweighting": "reweighed"}
MODEL_DESC = {
    "LogReg": "Logistic regression — a linear model, the classic credit-scoring baseline.",
    "SVM": "Linear support vector machine — another linear classifier.",
    "kNN": "k-nearest neighbors — classifies by the closest examples in the data.",
    "MLP": "A small neural network (multilayer perceptron).",
    "RandomForest": "An ensemble of decision trees on random subsets.",
    "GradBoost": "Gradient boosting — trees built sequentially to correct each other.",
    "XGBoost": "An optimized gradient-boosting library, strong on tabular data.",
    "LightGBM": "A fast gradient-boosting framework.",
    "CatBoost": "A gradient-boosting library designed for categorical data.",
}
METRICS = [
    ("SP",  "Statistical parity",  "Difference in how often each group is flagged as a likely defaulter. 0 = both groups flagged at the same rate."),
    ("TPR", "TPR gap",             "Difference in the share of actual defaulters correctly flagged. 0 = equal true-positive rates."),
    ("FPR", "FPR gap",             "Difference in the share of non-defaulters wrongly flagged. 0 = equal false-positive rates."),
]
METRIC_COL = {"SP": "statistical_parity_diff", "TPR": "TPR_diff (EOd)", "FPR": "FPR_diff (EOd)"}
RW_SCOPE = "Reweighting was evaluated on logistic regression and the five tree ensembles only, as a scope choice of the study."
EPS = 0.001    # changes below 0.1 percentage points are shown as "no change"

@st.cache_data
def load_predictions(): return pd.read_csv(PRED_PATH)
@st.cache_data
def load_shap():
    try: return pd.read_csv(SHAP_PATH)
    except FileNotFoundError: return None
@st.cache_data
def load_allcond():
    try: return pd.read_csv(ALLCOND_PATH)
    except FileNotFoundError: return None
@st.cache_data
def load_rw():
    try: return pd.read_csv(RW_PATH)
    except FileNotFoundError: return None

def _rates(y_true, y_pred):
    pos = y_pred.mean() if len(y_pred) else np.nan
    tpr = y_pred[y_true == 1].mean() if (y_true == 1).sum() else np.nan
    fpr = y_pred[y_true == 0].mean() if (y_true == 0).sum() else np.nan
    return pos, tpr, fpr

def model_metrics(df, col, priv, under):
    """Signed gaps, unprivileged minus privileged (for sex: men minus women)."""
    p = df[df[col] == priv]; u = df[df[col] == under]
    pp, pt, pf = _rates(p["y_true"].values, p["y_pred"].values)
    up, ut, uf = _rates(u["y_true"].values, u["y_pred"].values)
    return {"SP": up - pp, "TPR": ut - pt, "FPR": uf - pf}

def perf(df):
    y, yp = df["y_true"].values, df["y_pred"].values
    acc = (y == yp).mean()
    tpr = yp[y == 1].mean(); tnr = 1 - yp[y == 0].mean()
    return acc, (tpr + tnr) / 2

def status_pill(val, green, red):
    d = abs(val)
    if d < green: return "ok", "Within tolerance"
    if d < red:   return "watch", "Monitor"
    return "breach", "Breach"

# ---- filtros que se conservan entre secciones ----
for k, v in {"model": "XGBoost", "attr": "Sex", "mit": "FairFinGAN", "tol": 0.03, "cmp_models": MODEL_ORDER, "cmp_metric": "SP"}.items():
    st.session_state.setdefault(k, v)
def _keep(name):
    st.session_state[name] = st.session_state["_" + name]

st.sidebar.markdown("### Fairness Monitor")
st.sidebar.caption("Credit scoring · APW thesis")
page = st.sidebar.radio("Navigate", ["Home", "Model", "Compare", "Overview", "Methodology"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"Model view: single run (seed {DEMO_SEED}). Compare and Overview: ten seeds.")

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
    st.markdown('<div class="block"><div class="block-h">What this is</div><div class="lede">A fairness check for credit scoring models. It reads three things: the model\'s predictions, the observed outcomes used as reference labels, and the protected attribute. It never sees the input features. It shows where different fairness measures and mitigation methods disagree.</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="note"><b>Simulated bias.</b> The data contain a bias injected on purpose before the train-test split (15% of labels flipped in each sex group), so the test labels are also perturbed. All results describe behaviour under this controlled scenario, not real-world fairness.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block"><div class="block-h">Sections</div>'
        f'<div class="row"><div class="row-n">I</div><div><div class="row-h">Model</div><div class="row-d">One model in detail, before and after mitigation (single run, seed {DEMO_SEED}).</div></div></div>'
        '<div class="row"><div class="row-n">II</div><div><div class="row-h">Compare</div><div class="row-d">Models side by side, ranked by fairness (mean of ten seeds).</div></div></div>'
        '<div class="row"><div class="row-n">III</div><div><div class="row-h">Overview</div><div class="row-d">Both methods across both metrics (ten seeds).</div></div></div>'
        '<div class="row"><div class="row-n">IV</div><div><div class="row-h">Methodology</div><div class="row-d">Data, metrics and how it was run.</div></div></div>'
        '</div>', unsafe_allow_html=True)

# ============================================================================
# MODEL
# ============================================================================
elif page == "Model":
    preds = load_predictions()
    st.markdown('<div class="section-title">Model audit</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Pick one model and read its fairness before and after mitigation · single run, seed {DEMO_SEED} test set</div>', unsafe_allow_html=True)
    cc1, cc2, cc3 = st.columns([2, 1, 1], gap="medium")
    model = cc1.selectbox("Model", MODEL_ORDER, index=MODEL_ORDER.index(st.session_state.model), key="_model", on_change=_keep, args=("model",))
    attr_name = cc2.selectbox("Protected attribute", list(ATTR_CONFIG), index=list(ATTR_CONFIG).index(st.session_state.attr), key="_attr", on_change=_keep, args=("attr",))
    mit = cc3.selectbox("Mitigation", list(MITIGATIONS), index=list(MITIGATIONS).index(st.session_state.mit), key="_mit", on_change=_keep, args=("mit",))
    attr = ATTR_CONFIG[attr_name]
    st.caption(MODEL_DESC.get(model, ""))
    st.caption(f"Gaps are **{attr['under_label']} minus {attr['priv_label']}** (unprivileged minus privileged). "
               f"A positive value means the rate is higher for {attr['under_label'].lower()}; a negative value, higher for {attr['priv_label'].lower()}.")
    green_cut = st.slider("Tolerance (risk appetite) — absolute gap below this is acceptable",
                          min_value=0.01, max_value=0.15, value=float(st.session_state.tol), step=0.01, format="%.2f",
                          key="_tol", on_change=_keep, args=("tol",))
    red_cut = green_cut * 2.3
    st.caption(f"{green_cut:.2f} = {green_cut*100:.0f} percentage points. Within tolerance: |gap| below {green_cut:.2f} · "
               f"Monitor: from {green_cut:.2f} to {red_cut:.3f} (2.3 × tolerance) · Breach: above {red_cut:.3f}. The status depends on the tolerance you choose.")

    orig = preds[(preds.model == model) & (preds.dataset == "original")]
    mitg = preds[(preds.model == model) & (preds.dataset == MITIGATIONS[mit])]
    if mitg.empty:
        st.markdown(f'<div class="note"><b>Not evaluated.</b> {RW_SCOPE} Choose FairFinGAN, or one of those models.</div>', unsafe_allow_html=True)
    else:
        m_orig = model_metrics(orig, attr["column"], attr["priv_code"], attr["under_code"])
        m_mit = model_metrics(mitg, attr["column"], attr["priv_code"], attr["under_code"])
        cols = st.columns(3, gap="small")
        better, worse, out = [], [], []
        for col, (key, label, expl) in zip(cols, METRICS):
            before = m_orig[key]; after = m_mit[key]
            delta = abs(after) - abs(before)          # negative = fairer
            cls, txt = status_pill(after, green_cut, red_cut)
            if abs(delta) < EPS:
                dcls, dtxt = "flat", "no change vs original"
            elif delta < 0:
                dcls, dtxt = "good", f"▼ {abs(delta):.3f} smaller gap vs original"; better.append(label)
            else:
                dcls, dtxt = "bad", f"▲ {abs(delta):.3f} larger gap vs original"; worse.append(label)
            if cls != "ok": out.append(f"{label} ({txt.lower()})")
            with col:
                st.markdown(
                    f"""
                    <div class="mcard">
                        <div class="mcard-top"><span class="mcard-name">{label}</span><span class="pill {cls}">{txt}</span></div>
                        <div class="mcard-vals"><span class="mcard-after">{after:+.3f}</span><span class="mcard-before">{before:+.3f}</span></div>
                        <div class="mcard-delta {dcls}">{dtxt}</div>
                        <div class="mcard-exp">{expl} Signed gap after {mit}; the struck-through value is the original model.</div>
                    </div>
                    """, unsafe_allow_html=True)

        acc_o, bacc_o = perf(orig); acc_m, bacc_m = perf(mitg)
        p1, p2 = st.columns(2, gap="small")
        p1.markdown(f'<div class="mini" style="margin-top:0.9rem;"><div class="mini-label">Accuracy</div><div class="mini-value">{acc_m:.3f}</div><div class="mini-note">original {acc_o:.3f} · change {acc_m-acc_o:+.3f}</div></div>', unsafe_allow_html=True)
        p2.markdown(f'<div class="mini" style="margin-top:0.9rem;"><div class="mini-label">Balanced accuracy</div><div class="mini-value">{bacc_m:.3f}</div><div class="mini-note">original {bacc_o:.3f} · change {bacc_m-bacc_o:+.3f}</div></div>', unsafe_allow_html=True)

        parts = []
        parts.append(f"<b>Smaller gap:</b> {', '.join(better) if better else 'none'}.")
        parts.append(f"<b>Larger gap:</b> {', '.join(worse) if worse else 'none'}.")
        parts.append(f"<b>Outside tolerance after mitigation:</b> {', '.join(out) if out else 'none'}.")
        parts.append(f"<b>Cost:</b> accuracy {acc_m-acc_o:+.3f}, balanced accuracy {bacc_m-bacc_o:+.3f}.")
        st.markdown(f'<div class="concl"><b>In this run ({model}, {mit}, {attr_name.lower()}):</b> ' + " ".join(parts) +
                    ' One run can differ from the ten-seed average shown in Compare and Overview.</div>', unsafe_allow_html=True)

        fig = go.Figure()
        labels = [m[1] for m in METRICS]
        fig.add_trace(go.Bar(name="Original", x=labels, y=[abs(m_orig[k]) for k, _, _ in METRICS], marker_color="#9fb6cc"))
        fig.add_trace(go.Bar(name=f"After {mit}", x=labels, y=[abs(m_mit[k]) for k, _, _ in METRICS], marker_color="#2b5c8a"))
        fig.update_layout(barmode="group", height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#52606e", "family": "Inter"}, legend={"orientation": "h", "y": 1.15},
                          yaxis={"title": "absolute gap (closer to 0 = fairer)", "gridcolor": "#eef1f5"},
                          margin=dict(l=10, r=10, t=30, b=10))
        st.markdown('<div class="block-h" style="margin-top:1.4rem;">Before vs after mitigation (absolute gaps)</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

        lines = ["FAIRNESS MODEL CARD", "===================",
                 f"Model: {model}", MODEL_DESC.get(model, ""), f"Mitigation: {mit}",
                 f"Data: single run, seed {DEMO_SEED} test set",
                 f"Protected attribute: {attr_name} (privileged: {attr['priv_label']}, unprivileged: {attr['under_label']})",
                 f"Tolerance (risk appetite): {green_cut:.2f} (monitor above {green_cut:.2f}, breach above {red_cut:.3f})", "",
                 f"FAIRNESS METRICS (signed gaps, {attr['under_label']} minus {attr['priv_label']}; status uses the absolute value)",
                 f"{'Metric':<22}{'Original':>10}{'After ' + mit:>20}{'Status':>18}"]
        for key, label, _ in METRICS:
            b = m_orig[key]; a = m_mit[key]
            cls, txt = status_pill(a, green_cut, red_cut)
            lines.append(f"{label:<22}{b:>+10.3f}{a:>+20.3f}{txt:>18}")
        lines += ["", f"Accuracy: {acc_o:.3f} -> {acc_m:.3f}   Balanced accuracy: {bacc_o:.3f} -> {bacc_m:.3f}", "",
                  "Note: single representative run (one seed). Statistical claims rest on the full ten-seed analysis.",
                  "Positive class = default (1). The labels contain an injected bias (controlled scenario).",
                  "Generated by Fairness Monitor (APW thesis, POLIMI GSOM)."]
        st.download_button("Download model card", "\n".join(lines).encode("utf-8"),
                           file_name=f"model_card_{model}_{mit}.txt", mime="text/plain")

# ============================================================================
# COMPARE (ten seeds)
# ============================================================================
elif page == "Compare":
    ac, rw = load_allcond(), load_rw()
    st.markdown('<div class="section-title">Compare models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Models ranked by fairness after mitigation · mean of ten seeds · protected attribute: sex</div>', unsafe_allow_html=True)
    cc1, cc2 = st.columns([2, 1], gap="medium")
    sel_models = cc1.multiselect("Models", MODEL_ORDER, default=st.session_state.cmp_models, key="_cmp_models", on_change=_keep, args=("cmp_models",))
    mit = cc2.selectbox("Mitigation", list(MITIGATIONS), index=list(MITIGATIONS).index(st.session_state.mit), key="_mit_c", on_change=lambda: st.session_state.update(mit=st.session_state["_mit_c"]))
    metric_pick = st.radio("Rank by", ["SP", "TPR", "FPR"], index=["SP", "TPR", "FPR"].index(st.session_state.cmp_metric), horizontal=True,
                           format_func=lambda k: {"SP": "Statistical parity", "TPR": "TPR gap", "FPR": "FPR gap"}[k],
                           key="_cmp_metric", on_change=_keep, args=("cmp_metric",))
    st.caption("Values are absolute gaps (men minus women, sign removed), averaged over the ten seeds. Ten-seed results are available for sex only.")
    if ac is None or rw is None:
        st.info("Ten-seed results not available.")
    elif not sel_models:
        st.info("Select at least one model.")
    else:
        mcol = METRIC_COL[metric_pick]
        src = ac if mit == "FairFinGAN" else rw
        cond = MITIGATIONS[mit]
        rows, missing = [], []
        for m in sel_models:
            s = src[(src.model == m) & (src.dataset == cond)][mcol].abs()
            o = ac[(ac.model == m) & (ac.dataset == "original")][mcol].abs()
            if s.empty: missing.append(m); continue
            rows.append({"model": m, "after": s.mean(), "before": o.mean()})
        if missing:
            st.markdown(f'<div class="note"><b>Not evaluated:</b> {", ".join(missing)}. {RW_SCOPE}</div>', unsafe_allow_html=True)
        if rows:
            rank = pd.DataFrame(rows).sort_values("after")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=rank["model"], x=rank["before"], name="Original", orientation="h", marker_color="#9fb6cc"))
            fig.add_trace(go.Bar(y=rank["model"], x=rank["after"], name=f"After {mit}", orientation="h", marker_color="#2b5c8a"))
            fig.update_layout(barmode="group", height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"color": "#52606e", "family": "Inter"}, legend={"orientation": "h", "y": 1.08},
                              xaxis={"title": "absolute gap, mean of ten seeds (closer to 0 = fairer)", "gridcolor": "#eef1f5"},
                              yaxis={"autorange": "reversed"}, margin=dict(l=10, r=10, t=30, b=10))
            st.markdown('<div class="block-h" style="margin-top:1rem;">Ranking — fairest at the top</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"On {metric_pick} after {mit}: **{rank.iloc[0]['model']}** is the fairest on average, **{rank.iloc[-1]['model']}** the least fair.")

            figv = go.Figure()
            for m in rank["model"]:
                seedvals = src[(src.model == m) & (src.dataset == cond)][mcol].abs()
                figv.add_trace(go.Scatter(x=seedvals.values, y=[m] * len(seedvals), mode="markers",
                                          marker={"color": "#2b5c8a", "size": 7, "opacity": 0.45}, showlegend=False))
                figv.add_trace(go.Scatter(x=[seedvals.mean()], y=[m], mode="markers",
                                          marker={"color": "#c0504d", "size": 12, "symbol": "line-ns-open"}, showlegend=False))
            figv.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font={"color": "#52606e", "family": "Inter"},
                               xaxis={"title": f"|{metric_pick} gap| after {mit}, per seed", "gridcolor": "#eef1f5"},
                               yaxis={"autorange": "reversed"}, margin=dict(l=10, r=10, t=20, b=10))
            st.markdown('<div class="block-h" style="margin-top:1.6rem;">Variability across seeds</div>', unsafe_allow_html=True)
            st.caption("Each blue dot is one of the ten runs; the red mark is the mean. Wide spread means the effect is less consistent across runs.")
            st.plotly_chart(figv, use_container_width=True)

# ============================================================================
# OVERVIEW (ten seeds)
# ============================================================================
elif page == "Overview":
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Both mitigation methods, both metrics, all models · mean of ten seeds · protected attribute: sex</div>', unsafe_allow_html=True)
    ac = load_allcond(); rw = load_rw()
    if ac is None or rw is None:
        st.info("Overview data not available.")
    else:
        rw = rw.copy()
        rw["EOd_max"] = np.maximum(rw["TPR_diff (EOd)"].abs(), rw["FPR_diff (EOd)"].abs())
        def delta(src, cond, metric):
            vals = []
            for m in MODEL_ORDER:
                o = ac[(ac.model == m) & (ac.dataset == "original")][metric].abs().mean()
                s = src[(src.model == m) & (src.dataset == cond)][metric].abs()
                vals.append((s.mean() - o) if len(s) else np.nan)
            return vals
        cols_data = {
            "FairFinGAN<br>SP":  delta(ac, "synthetic", "statistical_parity_diff"),
            "FairFinGAN<br>EOd": delta(ac, "synthetic", "EOd_max"),
            "Reweighting<br>SP":  delta(rw, "reweighed", "statistical_parity_diff"),
            "Reweighting<br>EOd": delta(rw, "reweighed", "EOd_max"),
        }
        z = [[cols_data[c][i] for c in cols_data] for i in range(len(MODEL_ORDER))]
        text = [[("Not evaluated" if np.isnan(v) else f"{v:+.3f}") for v in row] for row in z]
        zplot = [[(0.0 if np.isnan(v) else v) for v in row] for row in z]
        fig = go.Figure(go.Heatmap(z=zplot, x=list(cols_data.keys()), y=MODEL_ORDER, text=text, texttemplate="%{text}",
                                   textfont={"size": 12}, colorscale=[[0, "#2b5c8a"], [0.5, "#f4f6f9"], [1, "#c0504d"]],
                                   zmid=0, colorbar={"title": "Δ |gap| vs<br>original"}, hoverongaps=False))
        fig.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#52606e", "family": "Inter"}, yaxis={"autorange": "reversed"},
                          margin=dict(l=10, r=10, t=20, b=10))
        st.markdown('<div class="block-h">Effect of each method on each metric</div>', unsafe_allow_html=True)
        st.caption("Change in the absolute gap with respect to the original model. Blue = fairer after mitigation · Red = less fair. "
                   "Reweighting narrows the statistical parity gap but widens the equalized odds gap in every model it is applied to. " + RW_SCOPE)
        st.plotly_chart(fig, use_container_width=True)

        pts = []
        for m in MODEL_ORDER:
            o = ac[(ac.model == m) & (ac.dataset == "original")]
            s = ac[(ac.model == m) & (ac.dataset == "synthetic")]
            r = rw[(rw.model == m) & (rw.dataset == "reweighed")]
            pts.append({"model": m, "ox": o["statistical_parity_diff"].abs().mean(), "oy": o["accuracy"].mean(),
                        "sx": s["statistical_parity_diff"].abs().mean(), "sy": s["accuracy"].mean(),
                        "rx": r["statistical_parity_diff"].abs().mean() if len(r) else np.nan,
                        "ry": r["accuracy"].mean() if len(r) else np.nan})
        tf = pd.DataFrame(pts); tr = tf.dropna(subset=["rx"])
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=tf["ox"], y=tf["oy"], mode="markers+text", text=tf["model"], textposition="top center",
                                  name="Original", marker={"color": "#9fb6cc", "size": 11}, textfont={"size": 9, "color": "#8795a6"}))
        fig2.add_trace(go.Scatter(x=tf["sx"], y=tf["sy"], mode="markers+text", text=tf["model"], textposition="bottom center",
                                  name="After FairFinGAN", marker={"color": "#2b5c8a", "size": 11}, textfont={"size": 9, "color": "#2b5c8a"}))
        fig2.add_trace(go.Scatter(x=tr["rx"], y=tr["ry"], mode="markers", name="After reweighting",
                                  marker={"color": "#6a9a6a", "size": 10, "symbol": "square"}))
        fig2.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font={"color": "#52606e", "family": "Inter"}, legend={"orientation": "h", "y": 1.08},
                           xaxis={"title": "|Statistical parity|, mean of ten seeds  (left = fairer)", "gridcolor": "#eef1f5"},
                           yaxis={"title": "Accuracy  (higher = better)", "gridcolor": "#eef1f5"},
                           margin=dict(l=10, r=10, t=30, b=10))
        st.markdown('<div class="block-h" style="margin-top:1.6rem;">Fairness vs accuracy trade-off</div>', unsafe_allow_html=True)
        st.caption("FairFinGAN moves most models towards a smaller gap at a cost in accuracy; reweighting barely changes accuracy.")
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# METHODOLOGY
# ============================================================================
elif page == "Methodology":
    st.markdown('<div class="section-title">Methodology</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How the experiment was built.</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="block"><div class="block-h">Dataset</div><div class="lede">Default of Credit Card Clients (Taiwan, UCI) — 30,000 clients, 23 features.
        The positive class is <b>default (1)</b>, so true and false positive rates refer to predicting default, not credit approval.
        Protected attributes: sex (privileged: women; unprivileged: men) and age (privileged: core age 25-65; unprivileged: under 25 or over 65).</div></div>
        <div class="block"><div class="block-h">Injected bias</div><div class="lede">The raw data shows little disparity on sex, so a controlled bias was injected
        before splitting: 15% of labels flipped in each group, worsening the unprivileged group and improving the privileged one. This is a controlled
        perturbation to study how methods respond, not real-world bias. Because it is injected before the split, the test labels are also perturbed and
        all metrics are measured relative to this simulated scenario.</div></div>
        <div class="block"><div class="block-h">Models and mitigation</div><div class="lede">Nine classifiers (logistic regression, linear SVM, k-NN, a small neural
        network, and five tree ensembles), with hyperparameters selected by cross-validation inside each seed on the training data only. Two pre-processing
        mitigation methods are compared: <b>FairFinGAN</b>, a GAN that generates a synthetic training set with a fairness objective (all nine models), and
        <b>reweighting</b> (AIF360), applied to logistic regression and the five tree ensembles as a scope choice.</div></div>
        <div class="block"><div class="block-h">Metrics</div><div class="lede">Statistical parity difference, the TPR gap and the FPR gap (the two parts of
        equalized odds), computed as unprivileged minus privileged, and an aggregate equalized-odds value, max(|TPR gap|, |FPR gap|). Rankings and the
        overview use absolute values: closer to zero = fairer.</div></div>
        <div class="block"><div class="block-h">Evaluation</div><div class="lede">Ten repeated random 80/20 holdouts. All conditions are evaluated on the same
        held-out test set, with the generator trained only on the training partition. The Model view shows a single representative run (seed 1); Compare and
        Overview use all ten seeds. Effects in the thesis are paired per-seed differences with standard errors corrected for overlapping splits.</div></div>
        <div class="block"><div class="block-h">Thresholds and scope</div><div class="lede">The tolerance is a risk appetite chosen by the analyst: 0.03 means
        3 percentage points. The four-fifths rule is a practical reference only (US employment guidance, not EU lending law). SHAP is an exploratory proxy
        check, not a causal explanation or a guarantee against proxy discrimination.</div></div>
        """, unsafe_allow_html=True)
    shap_df = load_shap()
    if shap_df is not None:
        with st.expander("Proxy check (SHAP) — top features"):
            st.caption("Tuned XGBoost trained on the original data, seed 1, explained on its test set.")
            st.dataframe(shap_df.head(10), use_container_width=True, hide_index=True)
            st.caption("Exploratory: features with high importance and high correlation with the protected attribute are proxy candidates.")
