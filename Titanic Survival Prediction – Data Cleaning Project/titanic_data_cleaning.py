"""
Mini Project 1 - Titanic Survival Prediction: Data Cleaning Project

Steps
  1. Load the raw Titanic dataset and audit it.
  2. Clean missing values (Age, Fare, Embarked) and drop unusable columns.
  3. Encode the Sex and Embarked columns.
  4. Visualise the age distribution (Matplotlib / Seaborn).
  5. Save the cleaned dataset as a new CSV.

Run with:  python titanic_data_cleaning.py
Outputs:   cleaned_titanic.csv
           outputs/age_distribution.png
           outputs/cleaning_report.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: write figures without a display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "titanic.csv"
OUT_DIR = BASE_DIR / "outputs"
CLEANED_PATH = BASE_DIR / "cleaned_titanic.csv"
OUT_DIR.mkdir(exist_ok=True)

AGGREGATE_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

report: list[str] = []


def log(title: str, body: object = "") -> None:
    text = f"\n{'=' * 78}\n{title}\n{'=' * 78}\n{body}"
    print(text)
    report.append(text)


# ---------------------------------------------------------------------------
# 1. Load and audit
# ---------------------------------------------------------------------------
df = pd.read_csv(RAW_PATH)
duplicates_found = int(df.duplicated().sum())
log("1. RAW DATA", f"Shape: {df.shape}\n\n{df.head().to_string()}")

missing = pd.DataFrame(
    {
        "missing": df.isna().sum(),
        "percent": (df.isna().mean() * 100).round(2),
    }
).sort_values("missing", ascending=False)
log(
    "1. MISSING VALUE AUDIT",
    missing.query("missing > 0").to_string()
    + f"\n\nDuplicate rows: {duplicates_found}",
)

# ---------------------------------------------------------------------------
# 2. Clean missing data
# ---------------------------------------------------------------------------
df = df.drop_duplicates().copy()
age_median = df["Age"].median()
fare_median = df["Fare"].median()
embarked_mode = df["Embarked"].mode()[0]
fare_missing = int(df["Fare"].isna().sum())
df["Age"] = df["Age"].fillna(age_median)
df["Fare"] = df["Fare"].fillna(fare_median)
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

# Cabin is ~77% empty, so it cannot be imputed reliably (and a missing cabin is
# itself informative: most of those passengers were in 3rd class).
df = df.drop(columns=["Cabin"])

# Derived features that make the dataset more useful downstream.
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

log(
    "2. CLEANING DECISIONS",
    f"Duplicates removed      : {duplicates_found}\n"
    f"Age imputed with median : {age_median:.1f}\n"
    f"Fare imputed with median: {fare_median:.2f} "
    f"({'no missing in this file' if fare_missing == 0 else f'{fare_missing} missing'})\n"
    f"Embarked imputed (mode) : {embarked_mode}\n"
    f"Dropped column          : Cabin (~77% missing)\n"
    f"Added columns           : FamilySize, IsAlone\n\n"
    f"Missing values left     : {int(df.isna().sum().sum())}",
)
log("2. MISSING VALUES AFTER CLEANING", df.isna().sum().to_string())

# ---------------------------------------------------------------------------
# 3. Encode Sex and Embarked
# ---------------------------------------------------------------------------
sex_text = df["Sex"].copy()  # keep readable labels for the plots below

sex_encoder = LabelEncoder()
df["Sex"] = sex_encoder.fit_transform(df["Sex"])  # female=0, male=1

embarked_encoder = LabelEncoder()
df["Embarked_Label"] = embarked_encoder.fit_transform(df["Embarked"])  # compact codes

onehot = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
embarked_onehot = pd.DataFrame(
    onehot.fit_transform(df[["Embarked"]]),
    columns=onehot.get_feature_names_out(["Embarked"]),
    index=df.index,
)
df = pd.concat([df, embarked_onehot], axis=1)

log(
    "3. ENCODING",
    f"LabelEncoder Sex      -> {dict(zip(sex_encoder.classes_, sex_encoder.transform(sex_encoder.classes_)))}\n"
    f"LabelEncoder Embarked -> {dict(zip(embarked_encoder.classes_, embarked_encoder.transform(embarked_encoder.classes_)))}\n"
    f"OneHotEncoder         -> {list(embarked_onehot.columns)} "
    f"(drop='first' avoids collinearity)\n\n"
    + df[["Sex", "Embarked", "Embarked_Label", *embarked_onehot.columns]].drop_duplicates().to_string(index=False),
)

# Keep PassengerId/Name/Ticket as identifiers (not model features) in the CSV.
log(
    "3. FINAL COLUMNS",
    f"Shape: {df.shape}\n" + ", ".join(df.columns),
)

# ---------------------------------------------------------------------------
# 4. Visualise the age distribution
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="notebook")
plot_df = df.copy()
plot_df["Sex_text"] = sex_text  # decoded labels make the plots readable

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Titanic - Age Distribution", fontsize=16, fontweight="bold")

sns.histplot(data=plot_df, x="Age", bins=30, kde=True, color="#4c72b0", ax=axes[0, 0])
axes[0, 0].axvline(age_median, color="#c44e52", linestyle="--", label=f"median = {age_median:.1f}")
axes[0, 0].set_title("Age histogram with KDE")
axes[0, 0].legend()

sns.boxplot(data=plot_df, x="Age", color="#dd8452", ax=axes[0, 1])
axes[0, 1].set_title("Age box plot (outlier check)")

sns.histplot(
    data=plot_df, x="Age", hue="Survived", bins=30, kde=True, multiple="stack",
    palette={0: "#c44e52", 1: "#55a868"}, ax=axes[1, 0],
)
axes[1, 0].set_title("Age distribution by survival")
axes[1, 0].legend(title="Survived", labels=["0 - died", "1 - survived"])

sns.histplot(
    data=plot_df, x="Age", hue="Sex_text", bins=30, kde=True, multiple="layer",
    palette={"female": "#c44e52", "male": "#4c72b0"}, alpha=0.55, ax=axes[1, 1],
)
axes[1, 1].set_title("Age distribution by sex")
axes[1, 1].legend(title="Sex", labels=["female", "male"])

fig.tight_layout()
plot_path = OUT_DIR / "age_distribution.png"
fig.savefig(plot_path, dpi=150)
plt.close(fig)
log("4. VISUALISATION SAVED", str(plot_path))

# ---------------------------------------------------------------------------
# 5. Save the cleaned dataset
# ---------------------------------------------------------------------------
df.to_csv(CLEANED_PATH, index=False)
log(
    "5. CLEANED DATASET SAVED",
    f"{CLEANED_PATH}\nShape: {df.shape[0]} rows x {df.shape[1]} columns\n"
    f"Missing values: {int(df.isna().sum().sum())}",
)

(OUT_DIR / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
print(f"\nReport written to {OUT_DIR / 'cleaning_report.txt'}")
