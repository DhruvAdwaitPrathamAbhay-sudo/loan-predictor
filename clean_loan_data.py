"""
╔═══════════════════════════════════════════════════════════════════╗
║   LENDING CLUB LOAN DATA — CLEANING & PREPROCESSING PIPELINE    ║
║   Dataset: https://www.kaggle.com/datasets/wordsforthewise/      ║
║                                                                   ║
║   Steps:                                                          ║
║     1. Load raw dataset                                           ║
║     2. Filter to binary target (Fully Paid / Charged Off)         ║
║     3. Select relevant features                                   ║
║     4. Fill missing values (median / mode / constant)             ║
║     5. Encode categorical variables (ordinal + one-hot)           ║
║     6. Normalize numerical features (StandardScaler)              ║
║     7. Save cleaned dataset                                       ║
╚═══════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
import os
import json
import time

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION — Edit these paths as needed
# ─────────────────────────────────────────────────────────────────
RAW_DATA_PATH = './data/accepted_2007_to_2018Q4.csv.gz'
OUTPUT_DIR = './cleaned'
RANDOM_STATE = 42

# Features to keep from the raw dataset
FEATURES = [
    'loan_amnt', 'term', 'int_rate', 'installment', 'grade', 'sub_grade',
    'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
    'purpose', 'dti', 'delinq_2yrs', 'earliest_cr_line',
    'fico_range_low', 'fico_range_high', 'inq_last_6mths', 'open_acc',
    'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
    'application_type', 'mort_acc', 'pub_rec_bankruptcies',
]


def print_header(title):
    """Pretty print a section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_stat(label, value):
    """Pretty print a statistic."""
    print(f"  {label:.<45s} {value}")


# ═════════════════════════════════════════════════════════════════
# STEP 1: LOAD RAW DATA
# ═════════════════════════════════════════════════════════════════
print_header("STEP 1: Loading Raw Dataset")
start = time.time()

if not os.path.exists(RAW_DATA_PATH):
    print(f"\n  ERROR: Dataset not found at '{RAW_DATA_PATH}'")
    print(f"  Please download it from:")
    print(f"    https://www.kaggle.com/datasets/wordsforthewise/lending-club")
    print(f"\n  Then place the 'accepted_2007_to_2018Q4.csv.gz' file in ./data/")
    print(f"\n  Quick download with opendatasets:")
    print(f"    pip install opendatasets")
    print(f"    python -c \"import opendatasets as od; od.download('https://www.kaggle.com/datasets/wordsforthewise/lending-club')\"")
    exit(1)

df = pd.read_csv(RAW_DATA_PATH, compression='gzip', low_memory=False)
print_stat("Raw rows", f"{df.shape[0]:,}")
print_stat("Raw columns", f"{df.shape[1]}")
print_stat("Load time", f"{time.time() - start:.1f}s")


# ═════════════════════════════════════════════════════════════════
# STEP 2: CREATE BINARY TARGET
# ═════════════════════════════════════════════════════════════════
print_header("STEP 2: Creating Binary Target Variable")

print("\n  Loan status distribution (raw):")
for status, count in df['loan_status'].value_counts().items():
    print(f"    {status:.<40s} {count:>10,}")

# Keep only Fully Paid and Charged Off
df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
df['target'] = (df['loan_status'] == 'Fully Paid').astype(int)

paid = df['target'].sum()
default = len(df) - paid
print_stat("\nFully Paid (target=1)", f"{paid:,} ({paid/len(df)*100:.1f}%)")
print_stat("Charged Off (target=0)", f"{default:,} ({default/len(df)*100:.1f}%)")
print_stat("Total rows after filter", f"{len(df):,}")


# ═════════════════════════════════════════════════════════════════
# STEP 3: SELECT FEATURES
# ═════════════════════════════════════════════════════════════════
print_header("STEP 3: Selecting Features")

df = df[FEATURES + ['target']].copy()
print_stat("Selected features", f"{len(FEATURES)}")
print_stat("Shape", f"{df.shape}")


# ═════════════════════════════════════════════════════════════════
# STEP 4: FILL MISSING VALUES
# ═════════════════════════════════════════════════════════════════
print_header("STEP 4: Filling Missing Values")

# Show missing values before cleaning
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_pct', ascending=False)

print("\n  Missing values before cleaning:")
for col, row in missing_df.iterrows():
    print(f"    {col:.<35s} {row['missing_count']:>10,} ({row['missing_pct']:>5.2f}%)")

# --- 4a. Parse 'term' column (e.g., " 36 months" -> 36) ---
df['term'] = df['term'].str.strip().str.replace(' months', '', regex=False).astype(float)
print("\n  [term] Parsed string to numeric (months)")

# --- 4b. Map 'emp_length' to ordinal numeric ---
emp_map = {
    '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3,
    '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
    '8 years': 8, '9 years': 9, '10+ years': 10
}
df['emp_length'] = df['emp_length'].map(emp_map)
print(f"  [emp_length] Mapped to ordinal 0-10 (NaN count: {df['emp_length'].isnull().sum():,})")

# --- 4c. Parse 'earliest_cr_line' -> credit history in years ---
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%Y', errors='coerce')
df['credit_history_years'] = (pd.Timestamp('2018-12-01') - df['earliest_cr_line']).dt.days / 365.25
df.drop('earliest_cr_line', axis=1, inplace=True)
print(f"  [earliest_cr_line] Converted to 'credit_history_years' (float)")

# --- 4d. Combine FICO range into single score ---
df['fico_score'] = (df['fico_range_low'] + df['fico_range_high']) / 2
df.drop(['fico_range_low', 'fico_range_high'], axis=1, inplace=True)
print(f"  [fico_range_low/high] Combined into 'fico_score' (average)")

# --- 4e. Fill numeric columns with median ---
median_fill_cols = ['emp_length', 'revol_util', 'dti', 'credit_history_years']
for col in median_fill_cols:
    median_val = df[col].median()
    filled = df[col].isnull().sum()
    df[col].fillna(median_val, inplace=True)
    print(f"  [{col}] Filled {filled:,} NaN with median = {median_val:.2f}")

# --- 4f. Fill count-based columns with 0 (no records = 0) ---
zero_fill_cols = ['mort_acc', 'pub_rec_bankruptcies']
for col in zero_fill_cols:
    filled = df[col].isnull().sum()
    df[col].fillna(0, inplace=True)
    print(f"  [{col}] Filled {filled:,} NaN with 0")

# --- 4g. Drop any remaining rows with NaN ---
rows_before = len(df)
df.dropna(inplace=True)
rows_dropped = rows_before - len(df)
print(f"\n  Dropped {rows_dropped:,} rows with remaining NaN ({rows_dropped/rows_before*100:.2f}%)")
print_stat("Rows after missing value handling", f"{len(df):,}")


# ═════════════════════════════════════════════════════════════════
# STEP 5: ENCODE CATEGORICAL VARIABLES
# ═════════════════════════════════════════════════════════════════
print_header("STEP 5: Encoding Categorical Variables")

# --- 5a. Feature Engineering (ratio features) ---
df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
df['revol_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)
df['high_utilization'] = (df['revol_util'] > 80).astype(int)
print("  Created engineered features:")
print("    - loan_to_income (loan amount / annual income)")
print("    - installment_to_income (monthly installment / monthly income)")
print("    - revol_to_income (revolving balance / annual income)")
print("    - high_utilization (1 if revolving utilization > 80%)")

# --- 5b. Ordinal encoding for grade & sub_grade ---
grade_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['grade_num'] = df['grade'].map(grade_map)
print(f"\n  [grade] Ordinal encoded: A=1 ... G=7")

sub_grades = sorted(df['sub_grade'].dropna().unique())
sub_grade_map = {sg: i + 1 for i, sg in enumerate(sub_grades)}
df['sub_grade_num'] = df['sub_grade'].map(sub_grade_map)
print(f"  [sub_grade] Ordinal encoded: {sub_grades[0]}=1 ... {sub_grades[-1]}={len(sub_grades)}")

df.drop(['grade', 'sub_grade'], axis=1, inplace=True)

# --- 5c. One-hot encoding for remaining categorical columns ---
cat_cols = ['home_ownership', 'verification_status', 'purpose', 'application_type']
print(f"\n  One-hot encoding columns: {cat_cols}")

for col in cat_cols:
    print(f"    [{col}] categories: {df[col].nunique()} -> {list(df[col].value_counts().head(5).index)}")

df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')

print(f"\n  Shape after encoding: {df.shape}")


# ═════════════════════════════════════════════════════════════════
# STEP 6: NORMALIZE NUMERICAL FEATURES
# ═════════════════════════════════════════════════════════════════
print_header("STEP 6: Normalizing Numerical Features (StandardScaler)")

# Separate target from features
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

# Identify numeric columns for scaling (exclude binary/one-hot columns)
binary_cols = [c for c in X.columns if X[c].nunique() <= 2]
numeric_cols = [c for c in X.columns if c not in binary_cols]

print(f"  Total features: {len(feature_names)}")
print(f"  Numeric features to scale: {len(numeric_cols)}")
print(f"  Binary features (not scaled): {len(binary_cols)}")

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"\n  Scaling applied (mean=0, std=1) to {len(numeric_cols)} features")
print(f"\n  Sample scaled statistics:")
for col in numeric_cols[:5]:
    print(f"    {col:.<35s} mean={X[col].mean():.4f}  std={X[col].std():.4f}")
if len(numeric_cols) > 5:
    print(f"    ... and {len(numeric_cols) - 5} more")


# ═════════════════════════════════════════════════════════════════
# STEP 7: SAVE CLEANED DATA
# ═════════════════════════════════════════════════════════════════
print_header("STEP 7: Saving Cleaned Dataset")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save cleaned feature matrix + target
df_cleaned = X.copy()
df_cleaned['target'] = y.values
output_path = os.path.join(OUTPUT_DIR, 'loan_data_cleaned.csv')
df_cleaned.to_csv(output_path, index=False)
file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print_stat("Cleaned dataset", f"{output_path} ({file_size_mb:.1f} MB)")

# Save feature names
feature_path = os.path.join(OUTPUT_DIR, 'feature_names.json')
with open(feature_path, 'w') as f:
    json.dump({
        'all_features': feature_names,
        'numeric_features': numeric_cols,
        'binary_features': binary_cols,
        'total_count': len(feature_names)
    }, f, indent=2)
print_stat("Feature names", feature_path)

# Save scaler for inference
import pickle
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump({'scaler': scaler, 'numeric_cols': numeric_cols}, f)
print_stat("Scaler", scaler_path)

# Save a summary report
summary = {
    'original_rows': int(pd.read_csv(RAW_DATA_PATH, compression='gzip', nrows=0).shape[0]),
    'filtered_rows': int(len(df)),
    'features': len(feature_names),
    'numeric_scaled': len(numeric_cols),
    'binary_unscaled': len(binary_cols),
    'target_distribution': {
        'fully_paid': int(y.sum()),
        'charged_off': int(len(y) - y.sum()),
        'paid_ratio': round(float(y.mean()), 4),
    },
    'missing_value_strategy': {
        'emp_length': 'median',
        'revol_util': 'median',
        'dti': 'median',
        'credit_history_years': 'median',
        'mort_acc': 'zero',
        'pub_rec_bankruptcies': 'zero',
        'remaining_nulls': 'dropped',
    },
    'encoding_strategy': {
        'grade': 'ordinal (A=1..G=7)',
        'sub_grade': 'ordinal (A1=1..G5=35)',
        'home_ownership': 'one-hot (drop_first)',
        'verification_status': 'one-hot (drop_first)',
        'purpose': 'one-hot (drop_first)',
        'application_type': 'one-hot (drop_first)',
    },
    'normalization': 'StandardScaler (mean=0, std=1) on numeric features',
}
summary_path = os.path.join(OUTPUT_DIR, 'cleaning_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print_stat("Cleaning summary", summary_path)


# ═════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════
print_header("CLEANING COMPLETE")
print(f"""
  Input:  {RAW_DATA_PATH}
  Output: {OUTPUT_DIR}/
    - loan_data_cleaned.csv   (cleaned & normalized dataset)
    - feature_names.json      (feature metadata)
    - scaler.pkl              (fitted StandardScaler for inference)
    - cleaning_summary.json   (full cleaning report)

  Pipeline Summary:
    - Filtered to binary target (Fully Paid / Charged Off)
    - Filled missing values (median for continuous, 0 for counts)
    - Encoded categoricals (ordinal for grades, one-hot for others)
    - Engineered 4 ratio features
    - Normalized {len(numeric_cols)} numeric features with StandardScaler
    - Final shape: {df_cleaned.shape}

  Ready for model training!
""")
