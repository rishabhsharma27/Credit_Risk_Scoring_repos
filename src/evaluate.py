import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

def calculate_ks_statistic(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = np.max(tpr - fpr)
    return ks

def evaluate_and_save_model(model, X_test, y_test, output_filename="model_metrics.json"):
    """
    Computes evaluation metrics and saves them to a file in the reports/ directory.
    """
    # 1. Generate predictions and metrics
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    gini = (2 * auc) - 1
    ks = calculate_ks_statistic(y_test, y_pred_proba)
    
    metrics = {
        "ROC_AUC": round(float(auc), 4),
        "Gini_Coefficient": round(float(gini), 4),
        "KS_Statistic": round(float(ks), 4)
    }
    
    print("\n--- Model Evaluation Report ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 2. Set up reports directory and save file
    current_dir = Path(__file__).resolve().parent
    reports_dir = current_dir.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    file_path = reports_dir / output_filename
    with open(file_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"✅ Evaluation metrics saved successfully to: {file_path}")
    return metrics