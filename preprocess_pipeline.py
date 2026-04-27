"""
╔══════════════════════════════════════════════════════════════════════╗
║         LOAN PREDICTION — FULL PREPROCESSING PIPELINE               ║
║                                                                      ║
║  Steps:                                                              ║
║    1.  Initial Inspection                                            ║
║    2.  Handle Missing Values (drop >40%, median/mode imputation)     ║
║    3.  Clean Inconsistent Data (standardize text, emp_length)        ║
║    4.  Outlier Detection & Treatment (IQR capping)                   ║
║    5.  Feature Encoding (ordinal + one-hot)                          ║
║    6.  Feature Scaling (StandardScaler)                              ║
║    7.  Feature Selection (correlation matrix)                        ║
║    8.  Target Variable Processing (binary 0/1)                       ║
║    9.  Train-Test Split (80/20, stratified)                          ║
║    10. Final Output                                                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH   = "./data/accepted_2007_to_2018Q4.csv.gz"
OUTPUT_DIR      = "./cleaned"
RANDOM_STATE    = 42
TEST_SIZE       = 0.20
MISSING_DROP_THRESHOLD = 0.40   # Drop columns with >40% missing
IQR_MULTIPLIER  = 3.0           # Conservative capping (3×IQR)
CORR_THRESHOLD  = 0.92          # Drop one of a pair if |r| > threshold

# Features to load (limits RAM on the 2M-row dataset)
FEATURES = [
    "loan_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
    "emp_length", "home_ownership", "annual_inc", "verification_status",
    "purpose", "dti", "delinq_2yrs", "earliest_cr_line",
    "fico_range_low", "fico_range_high", "inq_last_6mths", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc",
    "application_type", "mort_acc", "pub_rec_bankruptcies",
    "loan_status",          # target source
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────
def banner(step: int, title: str) -> None:
    line = "═" * 64
    print(f"\n{line}\n  STEP {step}: {title}\n{line}")


def stat(label: str, value) -> None:
    print(f"  {label:<45s} {value}")


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — INITIAL INSPECTION
# ══════════════════════════════════════════════════════════════════════
banner(1, "Initial Inspection")
t0 = time.time()

if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{RAW_DATA_PATH}'.\n"
        "Download: https://www.kaggle.com/datasets/wordsforthewise/lending-club\n"
        "Place 'accepted_2007_to_2018Q4.csv.gz' inside ./data/"
    )

# Load only required columns to save memory
df = pd.read_csv(
    RAW_DATA_PATH,
    usecols=FEATURES,
    compression="gzip",
    low_memory=False,
)
stat("Load time", f"{time.time() - t0:.1f}s")
stat("Raw shape (rows × cols)", str(df.shape))

# Identify numerical vs categorical columns (before target processing)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
cat_cols = [c for c in cat_cols if c != "loan_status"]

print(f"\n  Numerical features ({len(num_cols)}):")
for c in num_cols:
    print(f"    • {c}  [{df[c].dtype}]")

print(f"\n  Categorical features ({len(cat_cols)}):")
for c in cat_cols:
    print(f"    • {c}  (unique={df[c].nunique()})")

print("\n  First 5 rows:")
print(df.head().to_string())


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — HANDLE MISSING VALUES
# ══════════════════════════════════════════════════════════════════════
banner(2, "Handle Missing Values")

# 2a. Drop columns with > 40% missing
missing_pct = df.isnull().mean()
drop_cols = missing_pct[missing_pct > MISSING_DROP_THRESHOLD].index.tolist()
if drop_cols:
    print(f"\n  Dropping {len(drop_cols)} columns with >{MISSING_DROP_THRESHOLD*100:.0f}% missing:")
    for c in drop_cols:
        print(f"    ✗ {c}  ({missing_pct[c]*100:.1f}% missing)")
    df.drop(columns=drop_cols, inplace=True)
else:
    print("  No columns exceed the 40% missing threshold.")

# 2b. Show remaining missing values
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
print(f"\n  Remaining columns with missing values ({len(missing)}):")
for col, cnt in missing.items():
    print(f"    {col:<35s}  {cnt:>8,}  ({cnt/len(df)*100:.2f}%)")

# Refresh column type lists after potential drops
num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
obj_cols  = df.select_dtypes(include=["object"]).columns.tolist()
obj_cols  = [c for c in obj_cols if c not in ("loan_status",)]

# 2c. Median imputation for numerical columns
print("\n  Median imputation (numerical):")
for col in num_cols:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        med = df[col].median()
        df[col].fillna(med, inplace=True)
        print(f"    [{col}]  filled {n_null:,} NaN → median={med:.4f}")

# 2d. Mode imputation for categorical columns
print("\n  Mode imputation (categorical):")
for col in obj_cols:
    n_null = df[col].isnull().sum()
    if n_null > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"    [{col}]  filled {n_null:,} NaN → mode='{mode_val}'")

# 2e. Final safety drop
rows_before = len(df)
df.dropna(inplace=True)
stat("\nRows dropped (residual NaN)", f"{rows_before - len(df):,}")
stat("Shape after missing-value handling", str(df.shape))


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — CLEAN INCONSISTENT DATA
# ══════════════════════════════════════════════════════════════════════
banner(3, "Clean Inconsistent Data")

# 3a. Strip whitespace from all string columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.strip()

# 3b. Standardise home_ownership (e.g. 'NONE', 'ANY' → 'OTHER')
print("\n  [home_ownership] unique values:", df["home_ownership"].unique().tolist())
df["home_ownership"] = df["home_ownership"].replace({"NONE": "OTHER", "ANY": "OTHER"})
print("  → Merged NONE/ANY into OTHER")

# 3c. Standardise verification_status
print("\n  [verification_status] unique:", df["verification_status"].unique().tolist())
# Already clean in LendingClub — just title-case for consistency
df["verification_status"] = df["verification_status"].str.title()

# 3d. Parse 'term'  " 36 months" → 36 (int)
df["term"] = (
    df["term"]
    .str.replace("months", "", regex=False)
    .str.strip()
    .astype(float)
    .astype(int)
)
print(f"\n  [term] Parsed to int months. Unique: {sorted(df['term'].unique())}")

# 3e. Convert emp_length → ordinal integer 0-10
emp_map = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4,  "5 years": 5, "6 years": 6, "7 years": 7,
    "8 years": 8,  "9 years": 9, "10+ years": 10,
}
df["emp_length"] = df["emp_length"].map(emp_map)
# Fill unmapped (NaN after map) with median
med_emp = df["emp_length"].median()
df["emp_length"].fillna(med_emp, inplace=True)
print(f"\n  [emp_length] Converted to ordinal 0-10 (NaN → median={med_emp})")

# 3f. earliest_cr_line → credit_history_years
ref_date = pd.Timestamp("2018-12-01")
df["earliest_cr_line"] = pd.to_datetime(
    df["earliest_cr_line"], format="%b-%Y", errors="coerce"
)
df["credit_history_years"] = (ref_date - df["earliest_cr_line"]).dt.days / 365.25
df.drop("earliest_cr_line", axis=1, inplace=True)
df["credit_history_years"].fillna(df["credit_history_years"].median(), inplace=True)
print("  [earliest_cr_line] → credit_history_years (float)")

# 3g. Merge FICO range into single score
df["fico_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
df.drop(["fico_range_low", "fico_range_high"], axis=1, inplace=True)
print("  [fico_range_low/high] → fico_score (average)")

# 3h. Remove special characters from remaining object columns
for col in df.select_dtypes("object").columns:
    df[col] = df[col].str.replace(r"[^A-Za-z0-9 _/\-]", "", regex=True).str.strip()

stat("\nShape after data cleaning", str(df.shape))


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — OUTLIER DETECTION & TREATMENT (IQR Capping)
# ══════════════════════════════════════════════════════════════════════
banner(4, "Outlier Detection & Treatment (IQR Capping)")

# Identify continuous numeric columns (exclude binary & encoded ordinals)
skip_outlier = {"term", "emp_length", "delinq_2yrs", "inq_last_6mths",
                "open_acc", "pub_rec", "total_acc", "mort_acc",
                "pub_rec_bankruptcies"}

continuous_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in skip_outlier
]

outlier_report = {}
for col in continuous_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - IQR_MULTIPLIER * IQR
    upper = Q3 + IQR_MULTIPLIER * IQR

    n_out = ((df[col] < lower) | (df[col] > upper)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lower=lower, upper=upper)
        outlier_report[col] = {"outliers_capped": int(n_out), "lower": round(lower, 4), "upper": round(upper, 4)}
        print(f"  [{col:<30s}]  capped {n_out:>7,} rows  [{lower:.2f}, {upper:.2f}]")

stat("\nTotal columns treated for outliers", len(outlier_report))
stat("Shape unchanged", str(df.shape))


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — FEATURE ENCODING
# ══════════════════════════════════════════════════════════════════════
banner(5, "Feature Encoding")

# 5a. Ordinal encoding: grade (A→1 … G→7)
grade_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
df["grade_num"] = df["grade"].map(grade_map).astype(float)
df.drop("grade", axis=1, inplace=True)
print("  [grade] Ordinal encoded → grade_num  (A=1 … G=7)")

# 5b. Ordinal encoding: sub_grade (A1→1 … G5→35)
sub_grades = sorted(df["sub_grade"].dropna().unique())
sub_grade_map = {sg: i + 1 for i, sg in enumerate(sub_grades)}
df["sub_grade_num"] = df["sub_grade"].map(sub_grade_map).astype(float)
df.drop("sub_grade", axis=1, inplace=True)
print(f"  [sub_grade] Ordinal encoded → sub_grade_num  ({sub_grades[0]}=1 … {sub_grades[-1]}={len(sub_grades)})")

# 5c. One-hot encoding for nominal categoricals
nominal_cols = [
    c for c in df.select_dtypes("object").columns
    if c != "loan_status"
]
print(f"\n  One-hot encoding {len(nominal_cols)} nominal columns: {nominal_cols}")
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True, dtype=np.int8)
# Sanitise column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(r"[\s/\-]+", "_", regex=True)
    .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
)
stat("\nShape after encoding", str(df.shape))


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — FEATURE SCALING
# ══════════════════════════════════════════════════════════════════════
banner(6, "Feature Scaling (StandardScaler)")

# Build feature matrix (target not yet separated — loan_status still present)
# Separate target first
TARGET_RAW_VALS = {"Fully Paid", "Charged Off"}
df = df[df["loan_status"].isin(TARGET_RAW_VALS)].copy()
y_raw = df["loan_status"]
X = df.drop("loan_status", axis=1)
feature_names = X.columns.tolist()

# Scale only continuous numeric cols (skip binary/one-hot/ordinal-small)
binary_cols  = [c for c in X.columns if X[c].nunique() <= 2]
scale_cols   = [c for c in X.columns if c not in binary_cols and X[c].dtype != np.int8]

scaler = StandardScaler()
X[scale_cols] = scaler.fit_transform(X[scale_cols])

stat("Total features", len(feature_names))
stat("Scaled (continuous)", len(scale_cols))
stat("Unscaled (binary/ordinal)", len(binary_cols))

print("\n  Sample post-scaling statistics:")
for col in scale_cols[:6]:
    print(f"    {col:<35s}  mean={X[col].mean():.4f}  std={X[col].std():.4f}")
if len(scale_cols) > 6:
    print(f"    … and {len(scale_cols)-6} more")


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — FEATURE SELECTION (Correlation Matrix)
# ══════════════════════════════════════════════════════════════════════
banner(7, "Feature Selection (Correlation Matrix)")

# 7a. Remove near-zero variance features
vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
low_var_cols = [feature_names[i] for i, s in enumerate(vt.get_support()) if not s]
if low_var_cols:
    X.drop(columns=low_var_cols, inplace=True)
    print(f"  Removed {len(low_var_cols)} near-zero-variance columns: {low_var_cols}")
else:
    print("  No near-zero-variance columns found.")

# 7b. Remove highly correlated features
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] > CORR_THRESHOLD)]
if to_drop_corr:
    print(f"\n  Dropping {len(to_drop_corr)} highly correlated features (|r|>{CORR_THRESHOLD}):")
    for col in to_drop_corr:
        partner = upper_tri[col][upper_tri[col] > CORR_THRESHOLD].idxmax()
        print(f"    ✗ {col:<30s}  |r|={upper_tri[col].max():.3f}  with '{partner}'")
    X.drop(columns=to_drop_corr, inplace=True)
else:
    print("  No highly correlated features to drop.")

stat("\nFinal feature count after selection", X.shape[1])
stat("Shape of X", str(X.shape))


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — TARGET VARIABLE PROCESSING
# ══════════════════════════════════════════════════════════════════════
banner(8, "Target Variable Processing")

# Binary encode: Fully Paid=1, Charged Off=0
y = (y_raw == "Fully Paid").astype(np.int8)

paid    = int(y.sum())
default = int(len(y) - paid)
ratio   = paid / len(y)

print(f"\n  Target distribution:")
print(f"    Fully Paid  (1): {paid:>10,}  ({ratio*100:.2f}%)")
print(f"    Charged Off (0): {default:>10,}  ({(1-ratio)*100:.2f}%)")
print(f"\n  Class imbalance ratio: {max(paid,default)/min(paid,default):.2f}:1")
print("  → Will use stratified split in Step 9.")

assert set(y.unique()) == {0, 1}, "Target must be binary 0/1!"
stat("\nTarget dtype", str(y.dtype))
stat("No missing values in target", str(y.isnull().sum() == 0))


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — TRAIN-TEST SPLIT (80/20 Stratified)
# ══════════════════════════════════════════════════════════════════════
banner(9, "Train-Test Split (80/20, Stratified)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,        # preserves class ratio in both splits
)

stat("X_train shape", str(X_train.shape))
stat("X_test  shape", str(X_test.shape))
stat("y_train class ratio (1s)", f"{y_train.mean()*100:.2f}%")
stat("y_test  class ratio (1s)", f"{y_test.mean()*100:.2f}%")


# ══════════════════════════════════════════════════════════════════════
# STEP 10 — FINAL OUTPUT
# ══════════════════════════════════════════════════════════════════════
banner(10, "Final Output")

# Verify no missing values remain
assert X_train.isnull().sum().sum() == 0, "NaN found in X_train!"
assert X_test.isnull().sum().sum()  == 0, "NaN found in X_test!"
assert y_train.isnull().sum()       == 0, "NaN found in y_train!"
assert y_test.isnull().sum()        == 0, "NaN found in y_test!"
print("\n  ✓ Zero missing values in all splits.")

# Save train/test sets
train_df = X_train.copy(); train_df["target"] = y_train.values
test_df  = X_test.copy();  test_df["target"]  = y_test.values

train_path = os.path.join(OUTPUT_DIR, "train.csv")
test_path  = os.path.join(OUTPUT_DIR, "test.csv")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,   index=False)

# Save scaler & feature names
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
with open(scaler_path, "wb") as f:
    pickle.dump({"scaler": scaler, "scale_cols": scale_cols}, f)

feat_path = os.path.join(OUTPUT_DIR, "feature_names.json")
final_features = X_train.columns.tolist()
with open(feat_path, "w") as f:
    json.dump({
        "features": final_features,
        "total": len(final_features),
        "scale_cols": scale_cols,
        "binary_cols": [c for c in binary_cols if c in final_features],
    }, f, indent=2)

# Save outlier report
with open(os.path.join(OUTPUT_DIR, "outlier_report.json"), "w") as f:
    json.dump(outlier_report, f, indent=2)

# Save full summary
summary = {
    "X_train_shape": list(X_train.shape),
    "X_test_shape":  list(X_test.shape),
    "n_features":    len(final_features),
    "target_distribution": {
        "train_paid_ratio": round(float(y_train.mean()), 4),
        "test_paid_ratio":  round(float(y_test.mean()), 4),
    },
    "pipeline_steps": [
        "1. Initial inspection",
        "2. Drop >40% missing cols; median/mode imputation",
        "3. Text standardisation; emp_length→int; fico merge; credit_history_years",
        "4. IQR capping (3×IQR) on continuous features",
        "5. Ordinal encoding (grade, sub_grade); one-hot (home_ownership, purpose, etc.)",
        "6. StandardScaler on continuous features",
        "7. VarianceThreshold + correlation-matrix feature selection",
        "8. Binary target: Fully Paid=1, Charged Off=0",
        "9. 80/20 stratified train-test split",
        "10. Zero-NaN verified output",
    ],
}
with open(os.path.join(OUTPUT_DIR, "pipeline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n  Artifacts saved to '{OUTPUT_DIR}/':")
print(f"    • train.csv              ({X_train.shape[0]:,} rows)")
print(f"    • test.csv               ({X_test.shape[0]:,} rows)")
print(f"    • scaler.pkl             (fitted StandardScaler)")
print(f"    • feature_names.json     ({len(final_features)} features)")
print(f"    • outlier_report.json    ({len(outlier_report)} columns treated)")
print(f"    • pipeline_summary.json  (full run report)")

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  PIPELINE COMPLETE — Dataset is model-ready                         ║
║                                                                      ║
║  X_train : {str(X_train.shape):<20s}  (80% of clean data)          ║
║  X_test  : {str(X_test.shape):<20s}  (20% of clean data)           ║
║  Features: {len(final_features):<20d}                                       ║
║  Target  : binary 0/1  (stratified)                                 ║
║                                                                      ║
║  Ready for: XGBoost · Random Forest · Neural Networks               ║
╚══════════════════════════════════════════════════════════════════════╝
""")
