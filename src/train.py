import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb
# Import your custom functions from feature.py
import sys

# Automatically finds your project root and adds it to Python's search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Now this import will work reliably every time
from src.feature import clean_raw_data, engineer_features, preprocess_features

from src.evaluate import evaluate_and_save_model


#import evaluate_model

def train_credit_model():
    # 1. Set up paths
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    data_path = root_dir / "data" / "credit_score.csv"
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(data_path)

    df = preprocess_features(df)
    #df.info()

    # 2. Prepare Features (X) and Target (y)
    target_col = 'default'  # Assuming 'default' is the target variable
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle categorical variables via one-hot encoding


    # 3. Train-Test Split (Stratified to maintain default ratio)
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Train the XGBoost Model
    print("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate model performance
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"✅ Model Training Complete! ROC-AUC Score on Test Set: {auc:.4f}")

    # 5. Save Model and Feature Schema Artifacts for Production Inference
    model_path = models_dir / "xgb_credit_model.pkl"
    features_path = models_dir / "xgb_features.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(features_path, "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    print(f"📁 Model saved to: {model_path}")
    print(f"📁 Feature schema saved to: {features_path}")

    # After training your model and creating X_test, y_test:
    evaluate_and_save_model(model, X_test, y_test, output_filename="xgb_performance.json")

if __name__ == "__main__":
    train_credit_model()