import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

st.set_page_config(page_title="Fairness Monitor - Credit Scoring", layout="wide")

st.title("Fairness monitor — credit scoring")
st.caption("Receives model predictions (y_pred), true labels (y_true), and the protected attribute. Never sees the input features.")

ATTR_CONFIG = {
    "Sex": {"col": "sex", "priv": 2, "under": 1, "priv_label": "women", "under_label": "men"},
    "Age": {"col": "age_group", "priv": "core", "under": "edge", "priv_label": "25-65 years old", "under_label": "under 25 or over 65"},
}

METRIC_CAPTIONS = {
    "sp": "How different the approval rate (no default) is between the two groups.",
    "tpr": "How different the share of correctly identified good payers is between groups.",
    "fpr": "How different the share of cases wrongly flagged as risk is between groups.",
}

@st.cache_data
def load_default_data():
    preds = pd.read_csv("monitor_predictions.csv")
    shap_imp = pd.read_csv("shap_importance.csv")
    return preds, shap_imp

st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your own CSV (columns: model, dataset, y_true, y_pred, sex, age_group)", type="csv")

if uploaded is not None:
    data = pd.read_csv(uploaded)
    st.sidebar.success("Using your uploaded file.")
else:
    data, shap_data = load_default_data()
    st.sidebar.caption("Using the preloaded experiment data.")

st.sidebar.header("Filters")
attr_name = st.sidebar.selectbox("Protected attribute", list(ATTR_CONFIG.keys()))
attr = ATTR_CONFIG[attr_name]
dataset_choice = st.sidebar.radio("Dataset", ["Biased", "Synthetic"])
models = sorted(data['model'].unique())
selected_models = st.sidebar.multiselect("Models", models, default=models)
metric_key = st.sidebar.selectbox("Metric", ["sp", "tpr", "fpr"], format_func=lambda k: {"sp": "Statistical parity diff", "tpr": "TPR diff (EOd)", "fpr": "FPR diff (EOd)"}[k])

st.info(f"Privileged group: {attr['priv_label']}. Underprivileged group: {attr['under_label']} (based on observed default rate).")
st.caption(METRIC_CAPTIONS[metric_key])

steps_cols = st.columns(4)
step_labels = ["Inject bias", "Monitor detects", "FairFinGAN cleans", "Monitor confirms"]
active_step = 1 if dataset_choice == "Biased" else 3
for i, (col, label) in enumerate(zip(steps_cols, step_labels)):
    marker = "●" if i <= active_step else "○"
    color = "#185FA5" if i <= active_step else "#888780"
    col.markdown(f"<div style='text-align:center; color:{color};'>{marker}<br><span style='font-size:11px;'>{label}</span></div>", unsafe_allow_html=True)

def fairness_metrics(y_true, y_pred, sensitive, priv_val, underpriv_val):
    def rates(mask):
        yt = np.asarray(y_true)[mask]; yp = np.asarray(y_pred)[mask]
        if mask.sum() == 0:
            return np.nan, np.nan, np.nan
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        ppr = (yp == 1).mean()
        return tpr, fpr, ppr
    sensitive = np.asarray(sensitive)
    mask_priv = (sensitive == priv_val)
    mask_under = (sensitive == underpriv_val)
    tpr_p, fpr_p, ppr_p = rates(mask_priv)
    tpr_u, fpr_u, ppr_u = rates(mask_under)
    return {"sp": ppr_p - ppr_u, "tpr": tpr_p - tpr_u, "fpr": fpr_p - fpr_u}

results = []
for model in selected_models:
    subset = data[(data['model'] == model) & (data['dataset'] == dataset_choice)]
    if len(subset) == 0:
        continue
    m = fairness_metrics(subset['y_true'], subset['y_pred'], subset[attr['col']], attr['priv'], attr['under'])
    m['Model'] = model
    results.append(m)

results_df = pd.DataFrame(results)

improved = 0
if dataset_choice == "Synthetic":
    for model in selected_models:
        base_subset = data[(data['model'] == model) & (data['dataset'] == 'Biased')]
        base_m = fairness_metrics(base_subset['y_true'], base_subset['y_pred'], base_subset[attr['col']], attr['priv'], attr['under'])
        cur = results_df[results_df['Model'] == model][metric_key].values[0]
        if abs(cur) < abs(base_m[metric_key]):
            improved += 1

if dataset_choice == "Biased":
    st.subheader(f"The monitor detects disparity in {len(selected_models)} models")
else:
    st.subheader(f"The monitor confirms: {improved} of {len(selected_models)} models improved")

cols = st.columns(len(selected_models)) if selected_models else []
for col, model in zip(cols, selected_models):
    val = results_df[results_df['Model'] == model][metric_key].values[0]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=abs(val),
        number={'valueformat': '.3f'},
        gauge={
            'axis': {'range': [0, 0.15]},
            'bar': {'color': "#185FA5"},
            'steps': [
                {'range': [0, 0.03], 'color': "#EAF3DE"},
                {'range': [0.03, 0.07], 'color': "#FAEEDA"},
                {'range': [0.07, 0.15], 'color': "#FCEBEB"},
            ],
        },
        title={'text': model, 'font': {'size': 14}}
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    col.plotly_chart(fig, use_container_width=True)

st.subheader("Results table")
st.dataframe(results_df.rename(columns={"sp": "SP diff", "tpr": "TPR diff", "fpr": "FPR diff"}))

csv_export = results_df.to_csv(index=False).encode('utf-8')
st.download_button("Download report (CSV)", csv_export, "fairness_report.csv", "text/csv")

with st.expander("View proxy check (SHAP)"):
    if uploaded is None:
        corr_col = 'corr_with_SEX' if attr_name == "Sex" else 'corr_with_AGE'
        st.dataframe(shap_data[['feature', 'mean_abs_shap', corr_col]].sort_values('mean_abs_shap', ascending=False))
        st.caption("Features with high importance and high correlation with the protected attribute are proxy candidates.")
    else:
        st.caption("The SHAP check is only available for the preloaded experiment data.")
