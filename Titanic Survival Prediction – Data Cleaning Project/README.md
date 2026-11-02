# Mini Project 1 — Titanic Survival Prediction: Data Cleaning

End-to-end cleaning of the Kaggle Titanic dataset (891 passengers) as a
preparation step for survival modelling.

## Files

| Path | Description |
| --- | --- |
| `titanic_mini_project.ipynb` | **Jupyter notebook** with the full walkthrough, explanations and saved outputs |
| `titanic_mini_project.html` | Rendered HTML copy of the executed notebook (view without Jupyter) |
| `titanic_data_cleaning.py` | Full pipeline: audit → clean → encode → visualise → export |
| `data/titanic.csv` | Raw dataset |
| `cleaned_titanic.csv` | **Final cleaned dataset** (891 × 16, 0 missing values) |
| `outputs/age_distribution.png` | 4-panel age distribution figure |
| `outputs/cleaning_report.txt` | Log of every decision and the before/after values |

## Run

### Notebook (recommended walkthrough)

```bash
pip install -r ../requirements.txt jupyter
jupyter notebook titanic_mini_project.ipynb
```

Open the notebook from **inside this folder** so the relative paths resolve, then
run all cells. The notebook checks out executed with no errors and already
contains its outputs.

### Script

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python titanic_data_cleaning.py
```

## Pipeline

**1. Audit**
- Shape: 891 × 12. No duplicate rows.
- Missing: `Cabin` 687 (77.1%), `Age` 177 (19.9%), `Embarked` 2 (0.2%).
- `Fare` has one missing value in this Kaggle file, filled with the median.

**2. Clean missing data**
- `Age` → **median (28.0)**: the distribution is right-skewed and mean imputation
  would pull values toward the few elderly passengers.
- `Fare` → median (14.45), for the same robustness reason.
- `Embarked` → **mode (`S`)**; mean/median are meaningless for a categorical column.
- `Cabin` → **dropped**, because 77% is too much to impute and the column is an
  identifier rather than a measurable feature.

**3. Feature engineering**
- `FamilySize = SibSp + Parch + 1`
- `IsAlone = 1` when `FamilySize == 1`

**4. Encoding**
- `Sex` → `LabelEncoder`: `female = 0`, `male = 1` (binary, order-free).
- `Embarked` → `LabelEncoder` (`Embarked_Label`: C=0, Q=1, S=2) and
  `OneHotEncoder(drop="first")` → `Embarked_Q`, `Embarked_S`.
  The dropped `C` column is the reference category, which avoids the
  dummy-variable trap in linear models.
- `PassengerId`, `Name`, `Ticket` keep their original form as identifiers and
  should be excluded from model features.

**5. Visualisation** — `outputs/age_distribution.png`
- Age histogram with KDE, with the imputation median marked.
- Box plot to expose the elderly outliers (the reason median was chosen).
- Age distribution split by survival.
- Age distribution split by sex (using the decoded labels for readability).

**6. Export**
- `cleaned_titanic.csv`: 891 rows × 16 columns, **0 missing values**.

## Final columns

```
PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket,
Fare, Embarked, FamilySize, IsAlone, Embarked_Label, Embarked_Q, Embarked_S
```

## Quick check

```python
import pandas as pd
df = pd.read_csv("cleaned_titanic.csv")
print(df.shape, df.isna().sum().sum())   # (891, 16) 0
print(df[["Sex", "Age", "Embarked_Q", "Embarked_S"]].head())
```
