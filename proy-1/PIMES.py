import argparse
import sys
from typing import Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

MODEL_FILE = "pimes_model.pkl"

# agregar el requirements.txt, sino pip install numpy pandas scikit-learn matplotlib joblib

def load_and_align_data(features_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads feature and label datasets, ensuring strict index alignment."""
    try:
        X = pd.read_csv(features_path, sep="\t", index_col=0)
        Y = pd.read_csv(labels_path, sep="\t", index_col=0)
        
        common_indices = X.index.intersection(Y.index)
        if common_indices.empty:
            raise ValueError("No matching drug entries found between datasets.")
            
        return X.loc[common_indices], Y.loc[common_indices]
    except Exception as e:
        sys.exit(f"Data loading error: {str(e)}")


def compute_multilabel_roc(Y_true: np.ndarray, Y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Flattens multi-label targets and predictions to calculate micro-averaged ROC/AUC."""
    fpr, tpr, _ = roc_curve(Y_true.ravel(), Y_proba.ravel())
    macro_auc = auc(fpr, tpr)
    return fpr, tpr, macro_auc


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    """Generates and displays the ROC curve plot."""
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) - PIMES")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_training_mode(features_path: str, labels_path: str) -> None:
    """Handles cross-validation, metric visualization, and final model serialization."""
    X, Y = load_and_align_data(features_path, labels_path)
    
    # Shallow MLP architecture with early stopping for faster and stable convergence
    model = MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=1000, 
        early_stopping=True, 
        random_state=42
    )
    
    # 5-fold cross-validation returning a single 2D matrix of shape (n_samples, n_outputs)
    predictions_proba = cross_val_predict(model, X, Y, cv=5, method="predict_proba")
    
    fpr, tpr, roc_auc = compute_multilabel_roc(Y.values, predictions_proba)
    print(f"AUC: {roc_auc:.4f}")
    
    plot_roc_curve(fpr, tpr, roc_auc)
    
    # Train on full dataset before saving
    model.fit(X, Y)
    
    payload = {
        "model": model,
        "features_names": X.columns.tolist(),
        "labels_names": Y.columns.tolist()
    }
    joblib.dump(payload, MODEL_FILE)


def run_prediction_mode(test_features_path: str, target_effects_path: str) -> None:
    """Predicts side-effect probabilities for new drugs and outputs sorted results."""
    try:
        payload = joblib.load(MODEL_FILE)
        model = payload["model"]
        trained_labels = payload["labels_names"]
        trained_features = payload["features_names"]
    except FileNotFoundError:
        sys.exit(f"Model file '{MODEL_FILE}' not found. Run training (-e) first.")

    X_test = pd.read_csv(test_features_path, sep="\t", index_col=0)
    X_test = X_test.reindex(columns=trained_features, fill_value=0)
    
    with open(target_effects_path, "r", encoding="utf-8") as file:
        requested_effects = [line.strip() for line in file if line.strip()]

    # Predict probabilities directly into a 2D matrix
    proba_matrix = model.predict_proba(X_test)
    predictions_df = pd.DataFrame(proba_matrix, index=X_test.index, columns=trained_labels)
    
    valid_effects = [effect for effect in requested_effects if effect in trained_labels]
    if not valid_effects:
        sys.exit("None of the requested effects match the trained model targets.")

    # Filter and format output sorted downward by probability
    for drug in predictions_df.index:
        print(f"Drug: {drug}")
        sorted_effects = predictions_df.loc[drug, valid_effects].sort_values(ascending=False)
        for effect, probability in sorted_effects.items():
            print(f"  {effect}: {probability:.6f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PIMES: Drug Side-Effect Predictor Using Shallow Neural Networks")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    mode_group.add_argument("-e", nargs=2, metavar=("MEDICAMENTS_FILE", "EFFECTS_FILE"))
    mode_group.add_argument("-p", nargs=2, metavar=("TEST_MEDS_FILE", "TARGET_EFFECTS_FILE"))
    
    args = parser.parse_args()
    
    if args.e:
        run_training_mode(args.e[0], args.e[1])
    elif args.p:
        run_prediction_mode(args.p[0], args.p[1])


if __name__ == "__main__":
    main()