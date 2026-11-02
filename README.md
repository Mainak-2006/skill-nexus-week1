# Week 1 — Pandas, Data Cleaning & Encoding

Two deliverables using the Kaggle Titanic dataset (891 passengers × 12 columns).

## Contents

```
WEEK1/
├── requirements.txt
├── ASSIGNMENT 1/
│   ├── assignment1.py                  # Tasks 1-3: load, impute, encode
│   ├── data/titanic.csv
│   ├── outputs/
│   │   ├── assignment1_summary.txt     # info(), describe(), missing audit
│   │   ├── titanic_encoded.csv         # 891 × 19, no missing values
│   │   └── task2_age_imputation.png
│   └── README.md
└── Titanic Survival Prediction – Data Cleaning Project/
    ├── titanic_mini_project.ipynb      # Notebook walkthrough (executed)
    ├── titanic_mini_project.html       # Rendered notebook
    ├── titanic_data_cleaning.py        # Full cleaning + EDA pipeline
    ├── data/titanic.csv
    ├── cleaned_titanic.csv             # 891 × 16, no missing values
    ├── outputs/
    │   ├── age_distribution.png
    │   └── cleaning_report.txt
    └── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
cd "ASSIGNMENT 1" && python assignment1.py
cd "../Titanic Survival Prediction – Data Cleaning Project" && python titanic_data_cleaning.py
```

Or open the mini-project notebook (outputs already saved):

```bash
cd "Titanic Survival Prediction – Data Cleaning Project"
jupyter notebook titanic_mini_project.ipynb
```

Both scripts use `matplotlib.use("Agg")`, so they work on servers and in VS Code
without a display, and write figures to their `outputs/` folders.

## Summary of results

| Item | Result |
| --- | --- |
| Dataset | 891 rows, 12 columns, 0 duplicates |
| Missing values (raw) | Cabin 687 (77.1%), Age 177 (19.9%), Embarked 2 (0.2%), Fare 1 |
| Age imputation | median = 28.0 (mean 29.70 is inflated by outliers) |
| Fare imputation | median = 14.45 |
| Embarked imputation | mode = `S` |
| Cabin | dropped (too sparse, identifier-like) |
| Sex encoding | LabelEncoder: female = 0, male = 1 |
| Embarked encoding | LabelEncoder + OneHotEncoder (`Embarked_Q`, `Embarked_S`) |
| Final cleaned CSV | `Titanic Survival Prediction – Data Cleaning Project/cleaned_titanic.csv` |
