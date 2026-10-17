import pandas as pd
import numpy as np
import pickle
from pathlib import Path

import sys

# Automatically finds your project root and adds it to Python's search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Now this import will work reliably every time
from src.feature import clean_raw_data, engineer_features, preprocess_features

def predict_new_applicants():
    """
    Loads saved XGBoost model, preprocesses new applicant data,
    generates credit scores, and applies lending decision rules.
    """
    current_dir = Path(__file__).resolve().parent
    artifacts_dir = current_dir.parent / "models"
    
    # 1. Load saved model and training feature columns
    model_path = artifacts_dir / "xgb_credit_model.pkl"
    features_path = artifacts_dir / "xgb_features.pkl"
    
    if not model_path.exists() or not features_path.exists():
        print("❌ Error: Saved XGBoost artifacts not found. Run training first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    with open(features_path, "rb") as f:
        train_features = pickle.load(f)
        
    print("Successfully loaded XGBoost model and feature schema.")

    # 2. Load new applicant data (mocked using your raw dataset)
    data_path = current_dir.parent / "data" / "credit_score.csv"
    
    new_applicants = pd.read_csv(data_path).head() # Scoring first 20 applicants
    new_applicants = preprocess_features(new_applicants, train_features)
    new_applicants.info()
    print(f"Scoring {len(new_applicants)} new loan applicants...")

    # 3. Preprocess new data (One-Hot Encoding & Column Alignment)
    # This ensures new data matches the exact columns the model was trained on
    X_new = pd.get_dummies(new_applicants, drop_first=True)
    X_new = X_new.reindex(columns=train_features, fill_value=0)

    # 4. Generate Probability of Default
    y_pred_proba = model.predict_proba(X_new)[:, 1] # Probability of default (p)

    # 5. Convert Probability to Traditional Credit Score
    base_score = 600
    base_odds = 50.0
    pdo = 50.0

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    epsilon = 1e-6
    p_clipped = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    odds_default = p_clipped / (1 - p_clipped)
    scores = offset - factor * np.log(odds_default)

    # Attach results to dataframe
    results = new_applicants.copy()
    results['Probability_of_Default'] = np.round(y_pred_proba, 4)
    results['Credit_Score'] = np.round(scores).astype(int)

    # 6. Apply Automated Business Decision Rules
    def assign_loan_decision(score):
        if score >= 700:
            return "Auto-Approve"
        elif 550 <= score < 700:
            return "Manual Review (Underwriter)"
        else:
            return "Auto-Reject"

    results['loan_decision'] = results['Credit_Score'].apply(assign_loan_decision)

    # 7. Output and Export Results
    print("\n--- XGBoost Credit Scoring & Decisions ---")
    print(results[['Credit_Score', 'Probability_of_Default', 'loan_decision']].head(10))

    output_path = current_dir.parent / "data" / "xgb_scored_output.csv"
    results.to_csv(output_path, index=False)
    print(f"\n✅ Scored output saved successfully to: {output_path}")

if __name__ == "__main__":
    predict_new_applicants()