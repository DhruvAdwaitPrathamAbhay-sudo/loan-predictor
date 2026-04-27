# Loan Approval Prediction System

## Overview

The **Loan Approval Prediction System** is a full-stack machine learning application designed to predict whether a borrower will fully pay or default on a loan. It leverages a trained XGBoost classifier to assess risk based on user inputs.

The project is broken down into three main components:
1. **Machine Learning Pipeline**: Scripts for data cleaning, preprocessing, feature engineering, and model training using the Lending Club dataset.
2. **FastAPI Backend**: A high-performance Python server that loads the trained XGBoost model and exposes a RESTful API for inference.
3. **React Frontend**: A modern web interface built with React and Vite, allowing users to interact with the model via a clean, user-friendly form.

## Project Structure

```
loan-predictor/
├── data/                       # Raw Lending Club dataset (not tracked in Git)
├── cleaned/                    # Cleaned data, scalers, and feature metadata
│   ├── scaler.pkl              # Pickled sklearn Scaler used in training
│   └── feature_names.json      # JSON metadata defining expected feature order
├── frontend/                   # React + Vite frontend application
├── app.py                      # FastAPI application for inference
├── clean_loan_data.py          # Script for initial data cleaning
├── preprocess_pipeline.py      # Data preprocessing and feature engineering
├── evaluate_xgb.py             # Script for evaluating the trained XGBoost model
├── requirements.txt            # Python dependencies
├── xgb_improved.json           # Final trained XGBoost model
└── .gitignore                  # Git ignore rules
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Node.js (for the frontend)
- Git

### 1. Backend Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd "loan predictor"
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(We recommend using a virtual environment like `venv` or `conda`)*

3. **Ensure Artifacts are Present:**
   Ensure that `xgb_improved.json` is located in the root directory and that `scaler.pkl` and `feature_names.json` exist in the `cleaned/` directory.

4. **Run the FastAPI Server:**
   ```bash
   uvicorn app:app --reload
   ```
   The backend API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### 2. Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node dependencies:**
   ```bash
   npm install
   ```

3. **Run the React development server:**
   ```bash
   npm run dev
   ```
   The frontend will be available at the local URL provided by Vite (usually `http://localhost:5173`).

## Machine Learning Pipeline
If you wish to retrain the model:
1. Ensure the raw dataset is downloaded and available in the `data/` folder.
2. Run `clean_loan_data.py` to handle missing values and filter outliers.
3. Run `preprocess_pipeline.py` to scale continuous features, one-hot encode categorical features, and output the scaler object.
4. The training step generates the `xgb_improved.json` model artifact, which is then loaded by the FastAPI server in `app.py`.

## API Endpoints

- **`GET /health`**: Health check to verify if the server is running and the model/scaler are loaded.
- **`POST /predict`**: Accepts a JSON payload of loan application data (amount, term, interest rate, employment length, etc.) and returns a prediction (`approved`: true/false, `approval_probability`, and a message).

## Technologies Used
- **Backend**: Python, FastAPI, Uvicorn, Pandas, Scikit-learn, XGBoost
- **Frontend**: React, Vite, JavaScript, HTML/CSS
- **Machine Learning**: XGBoost Classifier

## License
MIT
