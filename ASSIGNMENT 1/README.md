# Assignment 1 — Pandas, Missing Data & Encoding

Titanic dataset (891 rows × 12 columns) used for all three tasks.

## Files

| File | Description |
| --- | --- |
| `assignment1.py` | Runs all three tasks end to end |
| `data/titanic.csv` | Raw Kaggle Titanic dataset |
| `outputs/assignment1_summary.txt` | Full console output (`info`, `describe`, missing-value audit, encodings) |
| `outputs/titanic_encoded.csv` | Cleaned + encoded dataset (891 × 19) |
| `outputs/task2_age_imputation.png` | Age before vs. after median imputation |

## Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python assignment1.py
```

## What each task does

**Task 1 — Load and summarize**
- `pd.read_csv()` into a DataFrame.
- `.info()` shows the dtypes and non-null counts (also captured into the summary file).
- `.describe()` for numeric columns and `describe(include="object")` for text columns.
- Custom audit of missing values and percentages.

**Task 2 — Missing data with mean / median**
- `Age`: 177 missing (19.87%). Right-skewed, so **median (28.0)** is the robust choice.
- `Fare`: 1 missing → median (14.45).
- `Embarked`: 2 missing → **mode** (`S`); mean/median are undefined for text.
- `Cabin`: 687 missing (77.1%) → dropped, because imputing an identifier adds no information.
- The script also builds mean-imputed and median-imputed copies and prints how the
  mean shifts (29.70 → 29.36), demonstrating that mean imputation distorts a skewed column.

**Task 3 — Encoding**
- `LabelEncoder` on `Sex` → `female = 0`, `male = 1` (`Sex_Label`).
- `LabelEncoder` on `Embarked` → `C = 0`, `Q = 1`, `S = 2` (`Embarked_Label`).
- `OneHotEncoder` on `Embarked` + `Pclass` → six 0/1 columns.
  One-hot is preferred for nominal data because it does not imply an order;
  label encoding is compact but implies `C < Q < S`, which is meaningless.

## Result

The encoded dataset has **no missing values** and is ready for a model:

```python
df = pd.read_csv("outputs/titanic_encoded.csv")
```
