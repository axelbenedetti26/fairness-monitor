# Experiment notebooks

Code for the experiment reported in the thesis *Fairness Monitoring and Bias Mitigation in AI Credit Scoring* (POLIMI GSOM). The notebooks run on Google Colab and read and write their files in Google Drive (`MyDrive/tesis_backup/`).

## Order of execution

1. `01_stageA_FairFinGAN.ipynb` (GPU runtime). Loads and cleans the data, injects the label bias (once, seed 42), and for each seed 1 to 10 trains FairFinGAN on the training partition and saves the synthetic training set (`tesis_backup/v4/synth_seedN.csv`). It contains the FairFinGAN code released by the authors (https://github.com/tailequy/FairFinGAN), used without changes.
2. `02_stageB_seeds1-4.ipynb`, `03_stageB_seeds5-7.ipynb`, `04_stageB_seeds8-10.ipynb` (CPU runtime, can run in parallel). For each seed: hyperparameter search by cross-validation on that seed's training partition (randomised search, 12 configurations, stratified 3-fold, ROC AUC); training of the nine classifiers on the original, synthetic and reweighted data with the selected configuration; evaluation on the common test set. Results go to `tesis_backup/v5/`.
3. The first stage B notebook (seeds 1 to 4) waits for the other two and then builds the summaries, the comparison files, the monitoring tool's prediction file, the SHAP analysis and the figures.

## Main outputs

- `all_conditions_v5.csv`: per-seed results for every model and condition
- `paired_deltas_v5.csv`: paired differences with uncorrected and Nadeau-Bengio corrected standard errors
- `best_params_all_seeds.csv`, `cv_results_seedN.csv`: hyperparameter search results
- `data_rates_all.csv`: default rate by sex in the training data of each condition
- `monitor_predictions_seed1_v5.csv`, `shap_importance_v5.csv`: inputs for the monitoring tool
