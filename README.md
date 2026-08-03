# fairness-monitor
Fairness monitoring prototype for credit scoring. APW thesis, POLIMI GSOM

# Fairness monitor — credit scoring

Prototype of a fairness monitoring tool for credit scoring models, built as part of an Academic Project Work (APW) at POLIMI GSOM (MSc FinTech, Finance and Digital Innovation).

## What it does

The monitor evaluates whether a credit scoring model treats different demographic groups (sex, age) fairly, using equalized odds and statistical parity. It only receives model predictions and the protected attribute — it never sees the underlying input features.

## Experiment

Using the Default of Credit Card Clients dataset (Taiwan, UCI/Kaggle), controlled bias was injected into the data to simulate an unfair model. Three classifiers (Logistic Regression, XGBoost, SVM) were trained and evaluated with the monitor. A synthetic, fairness-aware dataset was then generated with FairFinGAN to test whether retraining on it reduced the detected disparity.

## Attribution

The synthetic data generation engine is forked, unmodified, from FairFinGAN (Le Quy et al., 2026): https://github.com/tailequy/FairFinGAN

## Author

Axel Benedetti
