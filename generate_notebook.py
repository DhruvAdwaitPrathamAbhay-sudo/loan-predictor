import json
import collections

cells = []

def md(lines):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines})

def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in text.strip().split('\n')]})

md([
    "# Loan Prediction Model (End-to-End)\n",
    "This notebook covers loading data, preprocessing, simple models, advanced models, evaluation, and hyperparameter tuning."
])

md(["## STEP 1: Setup Colab"])
code("!pip install pandas numpy scikit-learn matplotlib seaborn -q")

md(["## STEP 2: Upload Dataset\n", "Use any loan dataset (e.g., from Kaggle)."])
code("""from google.colab import files
uploaded = files.upload()

import pandas as pd
# Read the first uploaded csv file
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)
df.head()""")

md(["## STEP 3: Data Preprocessing\n", "Fill missing values, encode categorical variables, and prepare the dataset."])
code("""# Fill missing values
df.fillna(method='ffill', inplace=True)

# Encode categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in df.select_dtypes(include=['object']):
    df[col] = le.fit_transform(df[col])""")

md(["## STEP 4: Feature Selection\n", "Note: Make sure 'Loan_Status' matches the exact column name in your dataset!"])
code("""X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']""")

md(["## STEP 5: Train Model (Start Simple)\n", "Logistic Regression baseline."])
code("""from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))""")

md(["## STEP 6: Use Better Model (Random Forest)"])
code("""from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))""")

md(["## STEP 7: Evaluation\n", "Confusion Matrix & Classification Report"])
code("""from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\\nClassification Report:")
print(classification_report(y_test, y_pred_rf))""")

md(["## STEP 8: Improve Model\n", "Hyperparameter tuning using GridSearchCV."])
code("""from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate best model
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("Tuned Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))""")

md(["## STEP 9: Save Model"])
code("""import pickle
pickle.dump(best_rf, open('loan_model.pkl', 'wb'))

from google.colab import files
files.download('loan_model.pkl')
print("Model saved and downloaded successfully!")""")

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open("Loan_Prediction_Colab.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Created Loan_Prediction_Colab.ipynb")
