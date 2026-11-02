"""
Assignment 1 - Pandas data loading, missing-value imputation and categorical encoding.

Run with:  python "assignment1.py"
Outputs:   outputs/assignment1_summary.txt
           outputs/titanic_encoded.csv
"""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "titanic.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

report_lines: list[str] = []


def log(title: str, body: object = "") -> None:
    """Print a section and keep it for the saved summary report."""
    text = f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{body}"
    print(text)
    report_lines.append(text)


# ---------------------------------------------------------------------------
# Task 1 - Load the dataset and summarise basic statistics
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

log("TASK 1 | DATASET LOADED", f"File   : {DATA_PATH.name}\nShape  : {df.shape[0]} rows x {df.shape[1]} columns")

log("TASK 1 | FIRST 5 ROWS", df.head().to_string())

info_buffer = io.StringIO()
df.info(buf=info_buffer)
log("TASK 1 | df.info()", info_buffer.getvalue())

log("TASK 1 | df.describe() - numeric columns", df.describe().round(3).to_string())

text_columns = df[[c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]]
log(
    "TASK 1 | df.describe() - categorical columns",
    text_columns.describe().to_string(),
)

log(
    "TASK 1 | MISSING VALUES PER COLUMN",
    pd.DataFrame(
        {
            "missing": df.isna().sum(),
            "percent": (df.isna().mean() * 100).round(2),
        }
    )
    .query("missing > 0")
    .sort_values("missing", ascending=False)
    .to_string(),
)

# ---------------------------------------------------------------------------
# Task 2 - Handle missing data with mean / median imputation
# ---------------------------------------------------------------------------
numeric_missing = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isna().any()]
log("TASK 2 | NUMERIC COLUMNS WITH MISSING VALUES", ", ".join(numeric_missing))

# Age has 177 missing values and a right-skewed distribution, so the median is
# the more robust choice. Fare is complete in this Kaggle file; the median fill
# is kept as a safeguard for other versions of the dataset.
age_mean = df["Age"].mean()
age_median = df["Age"].median()
fare_median = df["Fare"].median()

log(
    "TASK 2 | IMPUTATION VALUES",
    f"Age   mean = {age_mean:.2f} | median = {age_median:.2f} "
    f"(mean > median -> right skew, outliers pull the mean up)\n"
    f"Fare  median = {fare_median:.2f} (no missing values in this file)",
)

df_mean = df.copy()
df_mean["Age"] = df_mean["Age"].fillna(age_mean)
df_mean["Fare"] = df_mean["Fare"].fillna(df_mean["Fare"].mean())

df_median = df.copy()
df_median["Age"] = df_median["Age"].fillna(age_median)
df_median["Fare"] = df_median["Fare"].fillna(fare_median)

comparison = pd.DataFrame(
    {
        "Age mean - original": [df["Age"].mean()],
        "Age mean - mean-imputed": [df_mean["Age"].mean()],
        "Age mean - median-imputed": [df_median["Age"].mean()],
        "Age median - original": [df["Age"].median()],
        "Age median - median-imputed": [df_median["Age"].median()],
    }
).round(3)
log("TASK 2 | EFFECT OF IMPUTATION ON AGE", comparison.T.to_string())

df_clean = df.copy()
df_clean["Age"] = df_clean["Age"].fillna(age_median)
df_clean["Fare"] = df_clean["Fare"].fillna(fare_median)
df_clean["Embarked"] = df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0])
# Cabin is 77% empty and is an identifier rather than a measurable feature, so
# imputing it would fabricate information. Drop it instead.
df_clean = df_clean.drop(columns=["Cabin"])

remaining = df_clean.isna().sum()
remaining = remaining[remaining > 0]
log(
    "TASK 2 | REMAINING MISSING AFTER IMPUTATION",
    remaining.to_string() if len(remaining) else "none - dataset is complete",
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df["Age"].dropna(), bins=30, kde=True, ax=axes[0], color="#4c72b0")
axes[0].axvline(age_mean, color="#c44e52", linestyle="--", label=f"mean {age_mean:.1f}")
axes[0].axvline(age_median, color="#55a868", linestyle="-", label=f"median {age_median:.1f}")
axes[0].set_title("Age before imputation")
axes[0].legend()
sns.histplot(df_clean["Age"], bins=30, kde=True, ax=axes[1], color="#55a868")
axes[1].axvline(age_median, color="#333333", linestyle="--", label=f"median {age_median:.1f}")
axes[1].set_title("Age after median imputation")
axes[1].legend()
fig.suptitle("Task 2 - Missing value handling")
fig.tight_layout()
fig.savefig(OUT_DIR / "task2_age_imputation.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Task 3 - Encode categorical variables
# ---------------------------------------------------------------------------
sex_encoder = LabelEncoder()
df_clean["Sex_Label"] = sex_encoder.fit_transform(df_clean["Sex"])
log(
    "TASK 3 | LabelEncoder ON Sex",
    f"classes = {list(sex_encoder.classes_)}\n"
    + df_clean[["Sex", "Sex_Label"]].drop_duplicates().to_string(index=False),
)

embarked_encoder = LabelEncoder()
df_clean["Embarked_Label"] = embarked_encoder.fit_transform(df_clean["Embarked"])
log(
    "TASK 3 | LabelEncoder ON Embarked",
    f"classes = {list(embarked_encoder.classes_)}\n"
    + df_clean[["Embarked", "Embarked_Label"]].drop_duplicates().to_string(index=False),
)
onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded_array = onehot.fit_transform(df_clean[["Embarked", "Pclass"]])
encoded_cols = onehot.get_feature_names_out(["Embarked", "Pclass"])
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_clean.index)

log(
    "TASK 3 | OneHotEncoder ON Embarked + Pclass",
    f"new columns = {list(encoded_cols)}\n\n{encoded_df.head().to_string()}",
)

df_encoded = pd.concat([df_clean, encoded_df], axis=1)
output_path = OUT_DIR / "titanic_encoded.csv"
df_encoded.to_csv(output_path, index=False)
log("TASK 3 | ENCODED DATASET SAVED", f"{output_path}\nShape: {df_encoded.shape}")

log(
    "TASK 3 | LabelEncoder vs OneHotEncoder",
    "LabelEncoder -> one integer column, compact; implies an order, so use for\n"
    "                binary columns (Sex) or tree models.\n"
    "OneHotEncoder -> one 0/1 column per category, no false order; use for\n"
    "                nominal columns (Embarked, Pclass) with linear models.\n"
    "                drop='first' avoids the dummy-variable trap.",
)

(OUT_DIR / "assignment1_summary.txt").write_text("\n".join(report_lines), encoding="utf-8")
print(f"\nSummary written to {OUT_DIR / 'assignment1_summary.txt'}")
