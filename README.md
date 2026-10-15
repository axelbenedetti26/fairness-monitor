# Fairness Monitor — Credit Scoring

Prototype of a fairness monitoring tool for credit scoring models, built as part of an Academic Project Work (APW) at POLIMI GSOM (MSc FinTech, Finance and Digital Innovation). Thesis: *Fairness Monitoring and Bias Mitigation in AI Credit Scoring*.

## What it does

Gaps are computed as unprivileged minus privileged (for sex: men minus women). The labels contain a bias injected on purpose, so all results describe a controlled scenario.

The monitor evaluates whether a credit scoring model treats demographic groups (sex, age) fairly, using statistical parity and equalized odds (reported as the true positive rate gap and the false positive rate gap, with default = 1 as the positive class). It reads the model's predictions, the observed labels (as reference) and the protected attribute, never the input features, which keeps the monitoring role separate from the modelling role.

- **Model**: one model before and after mitigation (FairFinGAN or reweighting) on a single run (seed 1), with signed gaps, a status indicator against an adjustable tolerance, accuracy and balanced accuracy, a short summary and a downloadable model card.
- **Compare**: models ranked by fairness after either mitigation method (mean of ten seeds), with the spread of each metric across the seeds.
- **Overview**: the effect of both mitigation methods on both metrics for every model (ten seeds, sex), and the fairness-accuracy trade-off.
- **Methodology**: data, metrics and scope, with a SHAP-based proxy check presented as an exploratory diagnostic.

## Experiment

Default of Credit Card Clients dataset (Taiwan, UCI), with a controlled label bias by sex (15% of labels flipped in each group before any split). Nine classifiers, with hyperparameters selected by cross-validation inside each of ten repeated 80/20 splits. Two pre-processing mitigation methods: **FairFinGAN** (all nine models) and **reweighting** with AIF360 (logistic regression and the five tree ensembles). All conditions are evaluated on the same held-out test set, and effects are summarised by paired per-seed differences, with standard errors corrected for overlapping splits (Nadeau and Bengio).

## Key findings

- Reweighting narrows the statistical parity gap and widens the equalized odds gap in all six models it is applied to; the improvement is consistent in XGBoost, LightGBM and CatBoost.
- FairFinGAN narrows the equalized odds gap of all nine models and the statistical parity gap of the tree ensembles, but mainly by detecting fewer defaulters in both groups, with a consistent loss of ROC AUC.
- Most effects are small, and their consistency depends on how uncertainty is estimated.

## Files

- `app.py`: Streamlit app
- `monitor_predictions.csv`: test-set predictions of one representative run (seed 1) for the original, FairFinGAN and reweighting conditions. The reweighting predictions were regenerated outside the main experiment with the same data, split and hyperparameters; they match the experiment exactly for five models and in 99.7% of cases for CatBoost (library version).
- `all_conditions_final.csv`, `reweighing_results_holdout_v2.csv`: per-seed results (ten seeds)
- `paired_differences_by_seed.csv`: paired effects with uncorrected and corrected standard errors
- `shap_importance.csv`: SHAP proxy check (tuned XGBoost, seed 1)
