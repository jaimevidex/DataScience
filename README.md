# Data Science Project — Profiling to Forecasting

A complete data-science pipeline over two very different datasets, following the **CRISP-DM** lifecycle end to end: profiling, preparation, feature engineering, classification and time-series forecasting.

Built for the *Data Science* course at MEIC-A, Instituto Superior Técnico.

## The two datasets

| Dataset | Domain | Why it is interesting |
|---|---|---|
| **NYC arrests** | Security | High cardinality, heavy skew, strong seasonality |
| **European financial distress** | Economy | Class imbalance, many correlated indicators |

Working with both is deliberate: techniques that look good on one fail on the other, and the report compares them.

## Pipeline

```
Data_Profiling/       → Data_Preparation/ → Feature_Engineering/ → Classification/ → Forecasting/
```

| Stage | What happens | Scripts |
|---|---|---|
| **Profiling** | Granularity, distribution, sparsity, dimensionality studies | `Data_Profiling/processing_*.py` |
| **Preparation** | Missing values, outliers, discretisation, scaling, balancing | `Data_Preparation/processing_*.py` |
| **Feature engineering** | Selection, extraction, generation, low-variance and redundant filters | `Feature_Engineering/*.py` |
| **Classification** | KNN, Decision Trees, Random Forest, Gradient Boosting, Naive Bayes, MLP | `Classification/processing_*.py` |
| **Forecasting** | Stationarity, aggregation, smoothing, differentiation, scaling | `Forecasting/Analysis/`, `Forecasting/Transformation/` |

Each stage is split by dataset (`security` / `economical`), and `config.py` + `dslabs_functions.py` provide the shared plotting and evaluation helpers.

## Report

`ds_latex/main.pdf` is the written report: methodology, figures for every study, model comparison tables and conclusions. Every `Images/` folder holds the plots referenced there.

## Run

Each stage is a standalone script with a `config.py` beside it:

```bash
cd Data_Profiling
python processing_granularity_security.py
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `imbalanced-learn`, `statsmodels`.

## Notes

- CSVs are not committed — the course datasets are large and distributed separately.
- The `Images/` folders are committed on purpose: they are the evidence behind each decision in the report.
- `aExportImages/` regenerates the figures used in the LaTeX document.
