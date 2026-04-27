from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import json
import os

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predicts if a user will fully pay or default on a loan using an XGBoost model.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# GLOBALS & LOAD ARTIFACTS
# ---------------------------------------------------------
SCALER_PATH = "./cleaned/scaler.pkl"
FEATURES_PATH = "./cleaned/feature_names.json"
# We expect the user to place the model downloaded from Colab here
MODEL_PATH = "xgb_improved.json"

scaler_info = None
feature_metadata = None
model = None

@app.on_event("startup")
def load_artifacts():
    global scaler_info, feature_metadata, model
    
    # 1. Load Scaler
    if not os.path.exists(SCALER_PATH):
        print(f"Warning: {SCALER_PATH} not found.")
    else:
        with open(SCALER_PATH, "rb") as f:
            scaler_info = pickle.load(f)
            
    # 2. Load Feature Metadata
    if not os.path.exists(FEATURES_PATH):
        print(f"Warning: {FEATURES_PATH} not found.")
    else:
        with open(FEATURES_PATH, "r") as f:
            feature_metadata = json.load(f)

    # 3. Load XGBoost Model
    if os.path.exists(MODEL_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        print("XGBoost Model loaded successfully.")
    else:
        print(f"Warning: {MODEL_PATH} not found! Please download it from Colab and place it in the root folder.")

# ---------------------------------------------------------
# PYDANTIC SCHEMA
# ---------------------------------------------------------
# This schema exactly matches the 29 features our pipeline selected
class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., example=15000.0)
    term: int = Field(..., example=36, description="36 or 60 months")
    int_rate: float = Field(..., example=12.5)
    emp_length: int = Field(..., example=5, description="0 to 10 years")
    annual_inc: float = Field(..., example=75000.0)
    dti: float = Field(..., example=18.5)
    delinq_2yrs: float = Field(..., example=0.0)
    inq_last_6mths: float = Field(..., example=1.0)
    open_acc: float = Field(..., example=10.0)
    pub_rec: float = Field(..., example=0.0)
    revol_bal: float = Field(..., example=12000.0)
    revol_util: float = Field(..., example=45.5)
    total_acc: float = Field(..., example=25.0)
    mort_acc: float = Field(..., example=1.0)
    pub_rec_bankruptcies: float = Field(..., example=0.0)
    credit_history_years: float = Field(..., example=15.5)
    fico_score: float = Field(..., example=710.0)
    
    # Categorical One-Hot Flags (1=Yes, 0=No)
    home_ownership_OWN: int = Field(..., example=0)
    home_ownership_RENT: int = Field(..., example=0)  # If both are 0, implies MORTGAGE/OTHER
    verification_status_Source_Verified: int = Field(..., example=0)
    verification_status_Verified: int = Field(..., example=1)
    purpose_credit_card: int = Field(..., example=0)
    purpose_debt_consolidation: int = Field(..., example=1)
    purpose_home_improvement: int = Field(..., example=0)
    purpose_major_purchase: int = Field(..., example=0)
    purpose_medical: int = Field(..., example=0)
    purpose_other: int = Field(..., example=0)
    purpose_small_business: int = Field(..., example=0)
    application_type_Joint_App: int = Field(..., example=0)

# ---------------------------------------------------------
# INFERENCE ENDPOINT
# ---------------------------------------------------------
@app.post("/predict")
def predict_loan_approval(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model file (xgb_improved.json) not found on server.")
    
    # 1. Convert input to dict matching the exact order of `feature_metadata["features"]`
    input_data = application.dict()
    final_features = feature_metadata["features"]
    
    # Create DataFrame (1 row) ensuring columns are exactly as model expects
    df = pd.DataFrame([input_data])[final_features]
    
    # 2. Scale continuous features
    scaler = scaler_info["scaler"]
    # Only scale the columns that were originally scaled and survived feature selection
    cols_to_scale = [c for c in scaler_info["scale_cols"] if c in final_features]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    # 3. Predict
    # XGBoost handles DataFrame inputs nicely if feature names match
    probability = model.predict_proba(df)[0][1]  # Probability of class 1 (Fully Paid)
    prediction = int(probability > 0.5)
    
    return {
        "approved": bool(prediction == 1),
        "approval_probability": round(float(probability), 4),
        "message": "Loan Approved!" if prediction == 1 else "Loan Denied due to risk threshold.",
        "input_used": df.iloc[0].to_dict()  # Shows the scaled inputs used
    }

@app.get("/health")
def health_check():
    return {
        "status": "online", 
        "model_loaded": model is not None,
        "scaler_loaded": scaler_info is not None
    }
