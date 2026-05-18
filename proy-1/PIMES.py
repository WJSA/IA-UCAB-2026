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


def load_and_align_data(features_path: str, labels_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los conjuntos de datos de características y etiquetas, garantizando una alineación estricta de los índices."""
    try:
        X = pd.read_csv(features_path, sep="\t", index_col=0)
        Y = pd.read_csv(labels_path, sep="\t", index_col=0)
        
        common_indices = X.index.intersection(Y.index)
        if common_indices.empty:
            raise ValueError("No se encontraron medicamentos comunes entre ambos conjuntos de datos.")
            
        return X.loc[common_indices], Y.loc[common_indices]
    except Exception as e:
        sys.exit(f"Error al cargar los datos: {str(e)}")


def compute_multilabel_roc(Y_true: np.ndarray, Y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Aclana las matrices multietiqueta para calcular la curva ROC y el AUC global (Micro-average)."""
    fpr, tpr, _ = roc_curve(Y_true.ravel(), Y_proba.ravel())
    macro_auc = auc(fpr, tpr)
    return fpr, tpr, macro_auc


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> None:
    """Genera y despliega en pantalla la gráfica de la curva ROC."""
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Curva ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) - PIMES")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_training_mode(features_path: str, labels_path: str) -> None:
    """Gestiona la validación cruzada, visualización de métricas y la serialización final del modelo."""
    X, Y = load_and_align_data(features_path, labels_path)
    
    # Arquitectura Perceptrón Multicapa (MLP) poco profunda con parada temprana
    model = MLPClassifier(
        hidden_layer_sizes=(100,), 
        max_iter=1000, 
        early_stopping=True, 
        random_state=42
    )
    
    # Validación cruzada de 5 pliegues que retorna la matriz bidimensional de probabilidades
    predictions_proba = cross_val_predict(model, X, Y, cv=5, method="predict_proba")
    
    fpr, tpr, roc_auc = compute_multilabel_roc(Y.values, predictions_proba)
    print(f"AUC: {roc_auc:.4f}")
    
    plot_roc_curve(fpr, tpr, roc_auc)
    
    # Entrenamiento final con el 100% de los datos antes de exportar
    model.fit(X, Y)
    
    payload = {
        "model": model,
        "features_names": X.columns.tolist(),
        "labels_names": Y.columns.tolist()
    }
    joblib.dump(payload, MODEL_FILE)


def run_prediction_mode(test_features_path: str, target_effects_path: str) -> None:
    """Predice las probabilidades de efectos secundarios para nuevos fármacos y muestra los resultados ordenados."""
    try:
        payload = joblib.load(MODEL_FILE)
        model = payload["model"]
        trained_labels = payload["labels_names"]
        trained_features = payload["features_names"]
    except FileNotFoundError:
        sys.exit(f"Archivo de modelo '{MODEL_FILE}' no encontrado. Ejecute primero el entrenamiento (-e).")

    X_test = pd.read_csv(test_features_path, sep="\t", index_col=0)
    X_test = X_test.reindex(columns=trained_features, fill_value=0)
    
    with open(target_effects_path, "r", encoding="utf-8") as file:
        requested_effects = [line.strip() for line in file if line.strip()]

    # Predicción directa de probabilidades en formato matricial
    proba_matrix = model.predict_proba(X_test)
    predictions_df = pd.DataFrame(proba_matrix, index=X_test.index, columns=trained_labels)
    
    valid_effects = [effect for effect in requested_effects if effect in trained_labels]
    if not valid_effects:
        sys.exit("Ninguno de los efectos solicitados coincide con los objetivos del modelo entrenado.")

    # Filtrado y formateo de la salida ordenada de forma descendente por probabilidad
    for drug in predictions_df.index:
        print(f"Drug: {drug}")
        sorted_effects = predictions_df.loc[drug, valid_effects].sort_values(ascending=False)
        for effect, probability in sorted_effects.items():
            print(f"  {effect}: {probability:.6f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="PIMES: Predictor de Efectos Secundarios mediante Redes Neuronales")
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