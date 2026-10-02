# Explainable Machine Learning for Detecting Financial Reporting Anomalies: Pilot Benchmark

MScFE 690 Capstone, WorldQuant University.

This repository contains the pilot code that accompanies the project proposal. The pilot applies the three-tier benchmark of the
project (statistical models, supervised machine learning and unsupervised anomaly detection, with an explainability layer) to the
public dataset of Bao, Ke, Li, Yu and Zhang (2020). The EDGAR/XBRL panel that the full project will build is not part of the pilot.

## What the pilot does

| Step | Description |
|---|---|
| Data | 146,045 U.S. firm-years, fiscal years 1990 to 2014, with 964 AAER-labelled fraud firm-years from 412 cases (Bao et al. 2020) |
| Common sample | Only firm-years for which every model has all of its inputs, so that all models are scored on identical observations |
| Split | Walk-forward. Train on fiscal years 1991 to t-2 and test on year t, for t = 2003 to 2008 |
| Serial fraud | Training labels of AAER cases that reach the test year are recoded to 0 (Bao et al. 2020). Two robustness variants are included |
| Statistical models | Altman Z-score, Beneish M-score, Dechow F-score (model 1, logistic regression on seven ratios) |
| Supervised models | Random Forest and XGBoost on 7 ratios and on 42 variables, and RUSBoost on 28 raw items |
| Unsupervised model | Isolation Forest on 14 ratios, trained without labels. Deep Isolation Forest is planned for the full project |
| Metrics | ROC-AUC, PR-AUC, precision and share of frauds found in the top decile, with bootstrap 95% intervals (500 replicates) |
| Explainability | SHAP (TreeExplainer) for XGBoost, compared with red-flag variables fixed before the SHAP step was run |
| Replication check | RUSBoost on the full sample, compared with the figures reported by Bao et al. (2020) |

## How to run

```bash
pip install -r requirements.txt
python pilot.py
```

The notebook `Capstone_Pilot_Benchmark.ipynb` contains the same code with its outputs and can be opened in Jupyter or Google Colab.
The dataset is included in the repository. If the file is missing, the code downloads it from the public repository of Bao et al.
(https://github.com/JarFraud/FraudDetection). Random seeds are fixed. A full run takes roughly 15 to 20 minutes on a two-core machine.
Small numerical differences can occur across library versions and machines.

## Folder contents

* `pilot.py` and `Capstone_Pilot_Benchmark.ipynb`: the analysis.
* `data_FraudDetection_JAR2020.csv`: dataset of Bao et al. (2020).
* `figures/`: Figure 1 (labels by fiscal year), Figure 2 (performance by model), Figure 3 (SHAP).
* `results/summary_metrics.csv`: mean over the six test years with bootstrap intervals.
* `results/paired_differences.csv`: paired bootstrap differences between models.
* `results/metrics_by_test_year.csv`: every metric, model and test year.
* `results/replication_rusboost_full_sample.csv`: replication check.
* `results/shap_importance_*.csv`: mean absolute SHAP values.
* `results/common_sample_by_year.csv`: firm-years kept in the common sample.

## Known limitations of the pilot

* Hyperparameters are fixed. Time-series cross-validation is planned for the full project.
* The Beneish M-score is approximate. SG&A is not in the dataset, so the SGAI index is set to its neutral value of 1.
* The ratios supplied by Bao et al. were winsorised on the full sample by the original authors, which is a minor source of look-ahead.
* Each test year contains 22 to 61 fraud firm-years, so the confidence intervals are wide.
* SHAP values come from one model (test year 2008). The full project will aggregate across years and add Random Forest.

## Contributors

See `CONTRIBUTORS.md`.

## Data and reference

Bao, Yang, Bin Ke, Bin Li, Y. Julia Yu, and Jie Zhang. "Detecting Accounting Fraud in Publicly Traded U.S. Firms Using a Machine
Learning Approach." *Journal of Accounting Research*, vol. 58, no. 1, 2020, pp. 199-235. Data and code: https://github.com/JarFraud/FraudDetection
