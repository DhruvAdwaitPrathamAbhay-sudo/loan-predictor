from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier


def _infer_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "target",
        "label",
        "y",
        "loan_status",
        "loanstatus",
        "default",
        "is_default",
        "approved",
        "loan_approved",
    ]
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_to_actual:
            return lower_to_actual[cand]
    # last resort: last column if binary
    for c in reversed(df.columns.tolist()):
        u = pd.unique(df[c].dropna())
        if len(u) == 2:
            return c
    raise ValueError("Could not infer target column.")


def _align_to_columns(X: pd.DataFrame, ref_columns: list[str]) -> pd.DataFrame:
    X2 = X.copy()
    missing = [c for c in ref_columns if c not in X2.columns]
    for c in missing:
        X2[c] = 0
    extra = [c for c in X2.columns if c not in ref_columns]
    if extra:
        X2 = X2.drop(columns=extra)
    return X2[ref_columns]


def _prediction_sanity(proba: np.ndarray) -> dict[str, float]:
    return {
        "proba_min": float(np.min(proba)),
        "proba_max": float(np.max(proba)),
        "proba_mean": float(np.mean(proba)),
        "proba_std": float(np.std(proba)),
        "pred_pos_rate@0.5": float(np.mean(proba >= 0.5)),
        "pred_pos_rate@0.2": float(np.mean(proba >= 0.2)),
        "pred_pos_rate@0.8": float(np.mean(proba >= 0.8)),
    }


def _hallucination_like_checks(
    X_train: pd.DataFrame, X_test: pd.DataFrame, proba_test: np.ndarray
) -> dict[str, object]:
    """
    For tabular classifiers, "hallucination" isn't the right term.
    These checks estimate whether predictions are unstable/unrealistic:
    - distribution shift (simple mean/std deltas)
    - prediction extremeness (too many near-0 or near-1 probabilities)
    """
    checks: dict[str, object] = {}

    # Distribution shift summary (numeric columns only)
    num_cols = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    if num_cols:
        tr = X_train[num_cols]
        te = X_test[num_cols]
        shift = {}
        # sample a subset if extremely wide
        cols = num_cols[:5000]
        mean_delta = (te[cols].mean() - tr[cols].mean()).abs()
        std_delta = (te[cols].std(ddof=0) - tr[cols].std(ddof=0)).abs()
        shift["mean_delta_median"] = float(mean_delta.median())
        shift["mean_delta_p95"] = float(mean_delta.quantile(0.95))
        shift["std_delta_median"] = float(std_delta.median())
        shift["std_delta_p95"] = float(std_delta.quantile(0.95))
        checks["distribution_shift_summary"] = shift

    # Prediction extremeness
    p = proba_test
    checks["prediction_extremeness"] = {
        "pct_p<=0.01": float(np.mean(p <= 0.01)),
        "pct_p<=0.001": float(np.mean(p <= 0.001)),
        "pct_p>=0.99": float(np.mean(p >= 0.99)),
        "pct_p>=0.999": float(np.mean(p >= 0.999)),
    }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate an XGBoost JSON model on cleaned CSVs.")
    parser.add_argument("--model", required=True, help="Path to xgb_improved.json")
    parser.add_argument("--train", required=True, help="Path to cleaned train.csv (must include target)")
    parser.add_argument("--test", required=True, help="Path to cleaned test.csv (must include target)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for metrics.")
    args = parser.parse_args()

    model_path = Path(args.model)
    train_path = Path(args.train)
    test_path = Path(args.test)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    target = _infer_target_column(train)
    if target not in test.columns:
        raise ValueError(f"Target '{target}' not found in test columns.")

    X_train = train.drop(columns=[target])
    y_train = train[target].astype(int)
    X_test = test.drop(columns=[target])
    y_test = test[target].astype(int)

    model = XGBClassifier()
    model.load_model(str(model_path))

    X_test_aligned = _align_to_columns(X_test, list(X_train.columns))
    X_train_aligned = _align_to_columns(X_train, list(X_train.columns))

    proba = model.predict_proba(X_test_aligned)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(X_train.shape[1]),
        "target": target,
        "threshold": float(args.threshold),
        "accuracy": float(accuracy_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "sanity": _prediction_sanity(proba),
    }

    checks = _hallucination_like_checks(X_train_aligned, X_test_aligned, proba)

    print("=== Metrics ===")
    print(json.dumps(metrics, indent=2))

    print("\n=== Confusion matrix ===")
    print(confusion_matrix(y_test, pred))

    print("\n=== Classification report ===")
    print(classification_report(y_test, pred, digits=4))

    print("\n=== Robustness / 'hallucination-like' checks ===")
    print(json.dumps(checks, indent=2))

    # Suggest best F1 threshold (optional quick scan)
    prec, rec, thr = precision_recall_curve(y_test, proba)
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_thr = float(thr[best_idx - 1]) if best_idx > 0 and best_idx - 1 < len(thr) else 0.5
    print("\n=== Threshold suggestion ===")
    print(json.dumps({"best_f1": float(np.nanmax(f1)), "best_f1_threshold": best_thr}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

