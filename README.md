# fairness-monitor

Fairness monitoring prototype for credit scoring. APW thesis, POLIMI GSOM.

# Fairness Monitor — Credit Scoring

Prototype of a fairness monitoring tool for credit scoring models, built as part of an Academic Project Work (APW) at POLIMI GSOM (MSc FinTech, Finance and Digital Innovation).

## What it does

The monitor evaluates whether a credit scoring model treats demographic groups (sex, age) fairly, using statistical parity and equalized odds (reported as the true positive rate gap and false positive rate gap, with default = 1 as the positive class). It receives only the model's predictions and the protected attribute — it never sees the underlying input features, which keeps the monitoring role separate from the modeling role.

For each model the tool reports the fairness metrics with a traffic-light status against a configurable tolerance, bootstrap confidence intervals to flag whether a disparity is distinguishable from zero, and a SHAP-based proxy check presented as an exploratory diagnostic.

## Experiment

The study uses the Default of Credit Card Clients dataset (Taiwan, UCI). Controlled bias was injected into the labels (15% label flipping on both groups) to simulate a biased lending scenario with a known, measurable disparity.

Nine classifiers were trained and evaluated: Logistic Regression, Linear SVM, k-Nearest Neighbors, a Multilayer Perceptron, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost.

Two pre-processing mitigation methods were compared:
- **FairFinGAN** — a fairness-aware GAN that generates a de-biased synthetic training set, evaluated on all nine models.
- **Reweighting (AIF360)** — instance weighting, evaluated on six of the models (logistic regression and the five tree ensembles) as a scope choice for the comparison.

Each configuration was evaluated across ten repeated random holdouts, with the generator trained only on the training partition and all conditions evaluated on the same held-out real test set. Effects are summarized by paired per-seed differences, so that consistent effects can be distinguished from seed-to-seed variation.

## Key findings

- FairFinGAN's improvement on statistical parity is consistent for the linear models, k-NN and the MLP; its effect on the tree ensembles is inconclusive.
- Reweighting consistently improves statistical parity on most tree models while consistently worsening equalized odds on the same models (random forest is an exception).
- The two methods move statistical parity and equalized odds in different directions, which is the practical case for monitoring multiple metrics rather than one.

## Note on the deployed demo

The deployed app shows a single representative run (one random seed) and is a demonstration of the monitoring workflow. The statistical claims of the thesis rest on the full ten-seed analysis, not on the single run shown in the app.

## Attribution

The synthetic data generation engine is forked, unmodified, from FairFinGAN (Le Quy et al.): https://github.com/tailequy/FairFinGAN

## Author

Axel Benedetti
