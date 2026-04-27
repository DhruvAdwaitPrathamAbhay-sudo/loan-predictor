"""
Loan Approval Prediction API
─────────────────────────────
A FastAPI backend that accepts user-friendly loan application inputs,
transforms them into the 29 model features (one-hot encoding + scaling),
and returns a rich prediction response for the React frontend.
"""

from __future__ import annotations

import json
import os
import pickle
from contextlib import asynccontextmanager
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────
SCALER_PATH = "./cleaned/scaler.pkl"
FEATURES_PATH = "./cleaned/feature_names.json"
MODEL_PATH = "xgb_improved.json"

# Optimal threshold found during evaluation (best F1 = 0.891)
APPROVAL_THRESHOLD = 0.5


# ─────────────────────────────────────────────────────────
# APPLICATION STATE
# ─────────────────────────────────────────────────────────
class AppState:
    """Holds loaded ML artifacts so they are available across requests."""
    model: Optional[xgb.XGBClassifier] = None
    scaler_info: Optional[dict] = None
    feature_metadata: Optional[dict] = None


state = AppState()


# ─────────────────────────────────────────────────────────
# LIFESPAN (modern replacement for on_event("startup"))
# ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all ML artifacts once at startup."""
    # Load scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            state.scaler_info = pickle.load(f)
        print(f"[OK] Scaler loaded from {SCALER_PATH}")
    else:
        print(f"[WARN] Scaler not found at {SCALER_PATH}")

    # Load feature metadata
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            state.feature_metadata = json.load(f)
        print(f"[OK] Feature metadata loaded ({state.feature_metadata['total']} features)")
    else:
        print(f"[WARN] Feature metadata not found at {FEATURES_PATH}")

    # Load XGBoost model
    if os.path.exists(MODEL_PATH):
        state.model = xgb.XGBClassifier()
        state.model.load_model(MODEL_PATH)
        print(f"[OK] XGBoost model loaded from {MODEL_PATH}")
    else:
        print(f"[WARN] Model not found at {MODEL_PATH}")

    yield  # Application runs here

    # Cleanup (if needed)
    print("Shutting down...")


# ─────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Approval Prediction API",
    description="AI-powered loan approval predictions using XGBoost. "
                "Accepts user-friendly inputs and handles all feature "
                "engineering internally.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────
# ENUMS — Used for dropdowns on the frontend
# ─────────────────────────────────────────────────────────
class HomeOwnership(str, Enum):
    RENT = "RENT"
    OWN = "OWN"
    MORTGAGE = "MORTGAGE"


class VerificationStatus(str, Enum):
    NOT_VERIFIED = "Not Verified"
    SOURCE_VERIFIED = "Source Verified"
    VERIFIED = "Verified"


class LoanPurpose(str, Enum):
    DEBT_CONSOLIDATION = "debt_consolidation"
    CREDIT_CARD = "credit_card"
    HOME_IMPROVEMENT = "home_improvement"
    MAJOR_PURCHASE = "major_purchase"
    MEDICAL = "medical"
    SMALL_BUSINESS = "small_business"
    OTHER = "other"


class ApplicationType(str, Enum):
    INDIVIDUAL = "Individual"
    JOINT = "Joint App"


# ─────────────────────────────────────────────────────────
# PYDANTIC SCHEMA — User-friendly inputs
# ─────────────────────────────────────────────────────────
class LoanApplication(BaseModel):
    """
    Accepts human-readable inputs from the frontend multi-step form.
    The backend handles all one-hot encoding and scaling internally.
    """

    # ── Step 1: Loan Details ──
    loan_amnt: float = Field(
        ..., ge=500, le=40000,
        description="Loan amount in USD ($500 – $40,000)",
        json_schema_extra={"example": 15000.0},
    )
    term: int = Field(
        ...,
        description="Loan term: 36 or 60 months",
        json_schema_extra={"example": 36},
    )
    int_rate: float = Field(
        ..., ge=1.0, le=35.0,
        description="Interest rate as a percentage (1% – 35%)",
        json_schema_extra={"example": 12.5},
    )
    purpose: LoanPurpose = Field(
        ...,
        description="Purpose of the loan",
        json_schema_extra={"example": "debt_consolidation"},
    )
    application_type: ApplicationType = Field(
        default=ApplicationType.INDIVIDUAL,
        description="Individual or Joint application",
        json_schema_extra={"example": "Individual"},
    )

    # ── Step 2: Personal & Financial Profile ──
    annual_inc: float = Field(
        ..., ge=1000,
        description="Annual gross income in USD",
        json_schema_extra={"example": 75000.0},
    )
    emp_length: int = Field(
        ..., ge=0, le=10,
        description="Employment length in years (0–10, where 10 means 10+)",
        json_schema_extra={"example": 5},
    )
    home_ownership: HomeOwnership = Field(
        ...,
        description="Home ownership status",
        json_schema_extra={"example": "MORTGAGE"},
    )
    dti: float = Field(
        ..., ge=0, le=60,
        description="Debt-to-Income ratio",
        json_schema_extra={"example": 18.5},
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.NOT_VERIFIED,
        description="Income verification status",
        json_schema_extra={"example": "Not Verified"},
    )

    # ── Step 3: Credit History ──
    fico_score: float = Field(
        ..., ge=300, le=850,
        description="FICO credit score (300–850)",
        json_schema_extra={"example": 710.0},
    )
    credit_history_years: float = Field(
        ..., ge=0,
        description="Length of credit history in years",
        json_schema_extra={"example": 15.5},
    )
    delinq_2yrs: float = Field(
        default=0.0, ge=0,
        description="Number of 30+ day delinquencies in past 2 years",
        json_schema_extra={"example": 0.0},
    )
    inq_last_6mths: float = Field(
        default=0.0, ge=0,
        description="Credit inquiries in last 6 months",
        json_schema_extra={"example": 1.0},
    )
    open_acc: float = Field(
        ..., ge=0,
        description="Number of open credit accounts",
        json_schema_extra={"example": 10.0},
    )
    pub_rec: float = Field(
        default=0.0, ge=0,
        description="Number of derogatory public records",
        json_schema_extra={"example": 0.0},
    )
    revol_bal: float = Field(
        ..., ge=0,
        description="Total revolving credit balance ($)",
        json_schema_extra={"example": 12000.0},
    )
    revol_util: float = Field(
        ..., ge=0, le=150,
        description="Revolving line utilization rate (%)",
        json_schema_extra={"example": 45.5},
    )
    total_acc: float = Field(
        ..., ge=0,
        description="Total number of credit lines ever opened",
        json_schema_extra={"example": 25.0},
    )
    mort_acc: float = Field(
        default=0.0, ge=0,
        description="Number of mortgage accounts",
        json_schema_extra={"example": 1.0},
    )
    pub_rec_bankruptcies: float = Field(
        default=0.0, ge=0,
        description="Number of public record bankruptcies",
        json_schema_extra={"example": 0.0},
    )


# ─────────────────────────────────────────────────────────
# FEATURE ENGINEERING — Convert friendly inputs → 29 model features
# ─────────────────────────────────────────────────────────
def _encode_application(app_data: LoanApplication) -> dict:
    """
    Transform a user-friendly LoanApplication into a flat dict
    of the 29 features the XGBoost model expects.
    """
    row = {
        # Continuous features
        "loan_amnt": app_data.loan_amnt,
        "term": app_data.term,
        "int_rate": app_data.int_rate,
        "emp_length": app_data.emp_length,
        "annual_inc": app_data.annual_inc,
        "dti": app_data.dti,
        "delinq_2yrs": app_data.delinq_2yrs,
        "inq_last_6mths": app_data.inq_last_6mths,
        "open_acc": app_data.open_acc,
        "pub_rec": app_data.pub_rec,
        "revol_bal": app_data.revol_bal,
        "revol_util": app_data.revol_util,
        "total_acc": app_data.total_acc,
        "mort_acc": app_data.mort_acc,
        "pub_rec_bankruptcies": app_data.pub_rec_bankruptcies,
        "credit_history_years": app_data.credit_history_years,
        "fico_score": app_data.fico_score,
        # One-hot: Home Ownership
        "home_ownership_OWN": int(app_data.home_ownership == HomeOwnership.OWN),
        "home_ownership_RENT": int(app_data.home_ownership == HomeOwnership.RENT),
        # One-hot: Verification Status
        "verification_status_Source_Verified": int(
            app_data.verification_status == VerificationStatus.SOURCE_VERIFIED
        ),
        "verification_status_Verified": int(
            app_data.verification_status == VerificationStatus.VERIFIED
        ),
        # One-hot: Loan Purpose
        "purpose_credit_card": int(app_data.purpose == LoanPurpose.CREDIT_CARD),
        "purpose_debt_consolidation": int(app_data.purpose == LoanPurpose.DEBT_CONSOLIDATION),
        "purpose_home_improvement": int(app_data.purpose == LoanPurpose.HOME_IMPROVEMENT),
        "purpose_major_purchase": int(app_data.purpose == LoanPurpose.MAJOR_PURCHASE),
        "purpose_medical": int(app_data.purpose == LoanPurpose.MEDICAL),
        "purpose_other": int(app_data.purpose == LoanPurpose.OTHER),
        "purpose_small_business": int(app_data.purpose == LoanPurpose.SMALL_BUSINESS),
        # One-hot: Application Type
        "application_type_Joint_App": int(
            app_data.application_type == ApplicationType.JOINT
        ),
    }
    return row


def _scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the fitted scaler to the feature DataFrame.
    The scaler was trained on columns that may include extras
    (installment, grade_num, sub_grade_num) not in the final
    model features. We create a temporary DataFrame with all
    scaler columns, transform, then copy back the relevant ones.
    """
    scaler = state.scaler_info["scaler"]
    scale_cols = state.scaler_info["scale_cols"]
    final_features = state.feature_metadata["features"]

    # Build a temporary DataFrame with every column the scaler expects
    df_scale = pd.DataFrame(0.0, index=[0], columns=scale_cols)

    # Fill in the columns we actually have
    for col in scale_cols:
        if col in df.columns:
            df_scale[col] = df[col].values

    # Transform
    scaled_values = scaler.transform(df_scale)
    df_scaled = pd.DataFrame(scaled_values, columns=scale_cols)

    # Write back only the columns present in the final feature set
    for col in scale_cols:
        if col in final_features:
            df[col] = df_scaled[col].values

    return df


# ─────────────────────────────────────────────────────────
# RISK ANALYSIS — Generate insights for the results page
# ─────────────────────────────────────────────────────────
def _generate_risk_factors(app_data: LoanApplication, probability: float) -> list[dict]:
    """
    Return a list of human-readable risk/positive factors so the
    frontend can display helpful insights to the user.
    """
    factors = []

    # FICO Score
    if app_data.fico_score >= 740:
        factors.append({"type": "positive", "label": "Excellent Credit Score",
                        "detail": f"Your FICO score of {int(app_data.fico_score)} is excellent."})
    elif app_data.fico_score >= 670:
        factors.append({"type": "neutral", "label": "Good Credit Score",
                        "detail": f"Your FICO score of {int(app_data.fico_score)} is in the good range."})
    else:
        factors.append({"type": "negative", "label": "Low Credit Score",
                        "detail": f"Your FICO score of {int(app_data.fico_score)} is below the good threshold (670)."})

    # DTI
    if app_data.dti > 35:
        factors.append({"type": "negative", "label": "High Debt-to-Income Ratio",
                        "detail": f"A DTI of {app_data.dti}% is considered high risk."})
    elif app_data.dti <= 20:
        factors.append({"type": "positive", "label": "Low Debt-to-Income Ratio",
                        "detail": f"A DTI of {app_data.dti}% shows manageable debt levels."})

    # Interest Rate
    if app_data.int_rate > 20:
        factors.append({"type": "negative", "label": "High Interest Rate",
                        "detail": f"An interest rate of {app_data.int_rate}% significantly increases default risk."})
    elif app_data.int_rate <= 10:
        factors.append({"type": "positive", "label": "Favorable Interest Rate",
                        "detail": f"An interest rate of {app_data.int_rate}% is well within manageable limits."})

    # Income vs Loan Amount
    income_ratio = app_data.loan_amnt / app_data.annual_inc if app_data.annual_inc > 0 else 999
    if income_ratio > 0.5:
        factors.append({"type": "negative", "label": "High Loan-to-Income Ratio",
                        "detail": f"The loan is {income_ratio:.0%} of your annual income."})
    elif income_ratio <= 0.15:
        factors.append({"type": "positive", "label": "Strong Income Coverage",
                        "detail": f"The loan is only {income_ratio:.0%} of your annual income."})

    # Delinquencies
    if app_data.delinq_2yrs > 0:
        factors.append({"type": "negative", "label": "Recent Delinquencies",
                        "detail": f"You have {int(app_data.delinq_2yrs)} delinquency record(s) in the past 2 years."})

    # Public Records
    if app_data.pub_rec > 0 or app_data.pub_rec_bankruptcies > 0:
        factors.append({"type": "negative", "label": "Public Records on File",
                        "detail": "Public records or bankruptcies negatively affect approval chances."})

    # Employment
    if app_data.emp_length >= 5:
        factors.append({"type": "positive", "label": "Stable Employment",
                        "detail": f"{app_data.emp_length}+ years of employment history."})
    elif app_data.emp_length <= 1:
        factors.append({"type": "neutral", "label": "Short Employment History",
                        "detail": "Limited employment history may affect risk assessment."})

    # Revolving Utilization
    if app_data.revol_util > 80:
        factors.append({"type": "negative", "label": "High Credit Utilization",
                        "detail": f"Revolving utilization of {app_data.revol_util}% is very high."})
    elif app_data.revol_util <= 30:
        factors.append({"type": "positive", "label": "Low Credit Utilization",
                        "detail": f"Revolving utilization of {app_data.revol_util}% is healthy."})

    return factors


def _get_risk_grade(probability: float) -> dict:
    """Map approval probability to a letter grade for the frontend gauge."""
    if probability >= 0.85:
        return {"grade": "A", "label": "Very Low Risk", "color": "#10B981"}
    elif probability >= 0.70:
        return {"grade": "B", "label": "Low Risk", "color": "#34D399"}
    elif probability >= 0.55:
        return {"grade": "C", "label": "Moderate Risk", "color": "#FBBF24"}
    elif probability >= 0.40:
        return {"grade": "D", "label": "Elevated Risk", "color": "#F97316"}
    else:
        return {"grade": "F", "label": "High Risk", "color": "#EF4444"}


# ─────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    """Health check — used by the frontend to verify the backend is reachable."""
    return {
        "status": "online",
        "model_loaded": state.model is not None,
        "scaler_loaded": state.scaler_info is not None,
        "features_loaded": state.feature_metadata is not None,
        "version": "2.0.0",
    }


@app.get("/api/form-options")
def form_options():
    """
    Returns all valid dropdown/select options for the frontend form.
    The frontend can call this once on mount to dynamically populate
    its select menus, ensuring they always match what the backend accepts.
    """
    return {
        "home_ownership": [
            {"value": e.value, "label": e.value.title()} for e in HomeOwnership
        ],
        "verification_status": [
            {"value": e.value, "label": e.value} for e in VerificationStatus
        ],
        "purpose": [
            {"value": e.value, "label": e.value.replace("_", " ").title()} for e in LoanPurpose
        ],
        "application_type": [
            {"value": e.value, "label": e.value} for e in ApplicationType
        ],
        "term": [
            {"value": 36, "label": "36 months (3 years)"},
            {"value": 60, "label": "60 months (5 years)"},
        ],
        "ranges": {
            "loan_amnt": {"min": 500, "max": 40000, "step": 500},
            "int_rate": {"min": 1.0, "max": 35.0, "step": 0.1},
            "annual_inc": {"min": 1000, "max": 500000, "step": 1000},
            "emp_length": {"min": 0, "max": 10, "step": 1},
            "dti": {"min": 0, "max": 60, "step": 0.1},
            "fico_score": {"min": 300, "max": 850, "step": 1},
            "credit_history_years": {"min": 0, "max": 50, "step": 0.5},
            "delinq_2yrs": {"min": 0, "max": 20, "step": 1},
            "inq_last_6mths": {"min": 0, "max": 15, "step": 1},
            "open_acc": {"min": 0, "max": 50, "step": 1},
            "pub_rec": {"min": 0, "max": 10, "step": 1},
            "revol_bal": {"min": 0, "max": 200000, "step": 500},
            "revol_util": {"min": 0, "max": 150, "step": 0.1},
            "total_acc": {"min": 0, "max": 100, "step": 1},
            "mort_acc": {"min": 0, "max": 20, "step": 1},
            "pub_rec_bankruptcies": {"min": 0, "max": 5, "step": 1},
        },
    }


@app.post("/api/predict")
def predict(application: LoanApplication):
    """
    Core prediction endpoint.

    Accepts user-friendly inputs → encodes them to 29 model features
    → scales continuous columns → runs XGBoost inference → returns
    a rich response with approval decision, probability, risk grade,
    and human-readable risk factors.
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure xgb_improved.json is present.",
        )
    if state.scaler_info is None or state.feature_metadata is None:
        raise HTTPException(
            status_code=503,
            detail="Scaler or feature metadata not loaded.",
        )

    # 1. Encode user-friendly inputs → 29 raw features
    encoded = _encode_application(application)
    final_features = state.feature_metadata["features"]
    df = pd.DataFrame([encoded])[final_features]

    # 2. Scale continuous features
    df = _scale_features(df)

    # 3. Predict
    probability = float(state.model.predict_proba(df)[0][1])
    approved = probability >= APPROVAL_THRESHOLD

    # 4. Build rich response for the frontend
    risk_grade = _get_risk_grade(probability)
    risk_factors = _generate_risk_factors(application, probability)
    
    # 5. SHAP Explainability via XGBoost Built-in
    try:
        # Convert df to DMatrix
        dmatrix = xgb.DMatrix(df)
        
        # pred_contribs=True returns SHAP values + base value. Shape: (n_samples, n_features + 1)
        shap_values_with_base = state.model.get_booster().predict(dmatrix, pred_contribs=True)
        
        feature_impacts = []
        # The last element is the base_value, so we only zip up to [:-1]
        for feature_name, shap_val in zip(final_features, shap_values_with_base[0][:-1]):
            feature_impacts.append({"feature": feature_name, "impact": float(shap_val)})
            
        # Sort to find the most negative impacts (factors pushing towards rejection)
        feature_impacts.sort(key=lambda x: x["impact"])
        
        # Human-readable mapping for features
        feature_names_mapping = {
            "dti": "Debt-to-Income Ratio",
            "fico_score": "FICO Credit Score",
            "int_rate": "Interest Rate",
            "annual_inc": "Annual Income",
            "loan_amnt": "Loan Amount",
            "emp_length": "Employment Length",
            "revol_util": "Revolving Utilization",
            "term": "Loan Term",
            "mort_acc": "Mortgage Accounts",
            "home_ownership_RENT": "Renting Home",
            "purpose_debt_consolidation": "Debt Consolidation Purpose",
            "credit_history_years": "Length of Credit History",
            "delinq_2yrs": "Recent Delinquencies",
            "inq_last_6mths": "Recent Credit Inquiries",
            "pub_rec": "Public Records"
        }
        
        # Add the top 3 negative SHAP factors if not approved (or just the most impactful ones generally)
        added_factors = 0
        for item in feature_impacts:
            if added_factors >= 3:
                break
            # Only consider it a negative risk factor if the SHAP impact is significantly negative
            if item["impact"] < -0.1:
                feature_key = item["feature"]
                friendly_name = feature_names_mapping.get(feature_key, feature_key.replace("_", " ").title())
                
                # Check if we already have a static risk factor for this to avoid duplicates
                already_exists = any(friendly_name.lower() in factor["label"].lower() for factor in risk_factors)
                
                if not already_exists:
                    risk_factors.append({
                        "type": "negative",
                        "label": f"AI Risk Factor: {friendly_name}",
                        "detail": f"The model identified this feature as a significant negative factor reducing your approval chances."
                    })
                    added_factors += 1

    except Exception as e:
        print(f"[WARN] Error calculating SHAP values: {e}")

    return {
        "approved": approved,
        "approval_probability": round(probability, 4),
        "risk_grade": risk_grade,
        "message": "Loan Approved!" if approved else "Loan Denied -- risk threshold not met.",
        "risk_factors": risk_factors,
        "summary": {
            "loan_amount": application.loan_amnt,
            "term_months": application.term,
            "interest_rate": application.int_rate,
            "purpose": application.purpose.value.replace("_", " ").title(),
            "fico_score": application.fico_score,
            "annual_income": application.annual_inc,
        },
    }
