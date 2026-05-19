import sys
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier

# archivo para guardar el modelo
MODEL_FILE = "pimes_model.pkl"

class PIMES:
    def __init__(self):
        # inicializar modelo de red neuronal superficial
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(100,), 
            max_iter=1000, 
            early_stopping=True, 
            random_state=42
        )

    def cargar_datos(self, archivo_med, archivo_efectos):
        # leer los datos tabulares
        try:
            X = pd.read_csv(archivo_med, sep="\t", index_col=0)
            Y = pd.read_csv(archivo_efectos, sep="\t", index_col=0)
            
            # verificar indices comunes
            comunes = X.index.intersection(Y.index)
            if comunes.empty:
                print("Error: No se encontraron medicamentos comunes.")
                sys.exit(1)
                
            return X.loc[comunes], Y.loc[comunes]
        except Exception as e:
            print(f"Error cargando datos: {str(e)}")
            sys.exit(1)

    def entrenar(self, archivo_med, archivo_efectos):
        X, Y = self.cargar_datos(archivo_med, archivo_efectos)
        
        # validacion cruzada (5 folds)
        predicciones = cross_val_predict(self.modelo, X, Y, cv=5, method="predict_proba")
        
        # calculo de ROC y AUC global (micro-average para multietiqueta)
        y_true = Y.values.ravel()
        y_prob = predicciones.ravel()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        print(f"AUC: {roc_auc:.4f}")
        
        # mostrar grafica
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Falsos Positivos (FPR)")
        plt.ylabel("Verdaderos Positivos (TPR)")
        plt.title("Curva ROC - PIMES")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # entrenar con todos los datos y guardar
        self.modelo.fit(X, Y)
        datos_guardar = {
            "modelo": self.modelo,
            "features": X.columns.tolist(),
            "labels": Y.columns.tolist()
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(datos_guardar, f)

    def predecir(self, archivo_med, archivo_efectos):
        try:
            with open(MODEL_FILE, "rb") as f:
                datos_guardados = pickle.load(f)
            self.modelo = datos_guardados["modelo"]
            etiquetas_entrenadas = datos_guardados["labels"]
            caracteristicas_entrenadas = datos_guardados["features"]
        except FileNotFoundError:
            print(f"Error: Modelo '{MODEL_FILE}' no encontrado. Entrena primero con -e.")
            sys.exit(1)

        # cargar nuevos datos
        X_test = pd.read_csv(archivo_med, sep="\t", index_col=0)
        X_test = X_test.reindex(columns=caracteristicas_entrenadas, fill_value=0)
        
        # leer efectos a predecir
        with open(archivo_efectos, "r", encoding="utf-8") as f:
            efectos_solicitados = [linea.strip() for linea in f if linea.strip()]

        # predecir probabilidades
        matriz_proba = self.modelo.predict_proba(X_test)
        df_predicciones = pd.DataFrame(matriz_proba, index=X_test.index, columns=etiquetas_entrenadas)
        
        efectos_validos = [ef for ef in efectos_solicitados if ef in etiquetas_entrenadas]
        if not efectos_validos:
            print("Error: Ningun efecto solicitado coincide con los del modelo.")
            sys.exit(1)

        # mostrar resultados ordenados
        for med in df_predicciones.index:
            print(f"Drug: {med}")
            efectos_ordenados = df_predicciones.loc[med, efectos_validos].sort_values(ascending=False)
            for efecto, prob in efectos_ordenados.items():
                print(f"  {efecto}: {prob:.6f}")
            print()


def main():
    # uso: python3 PIMES.py -[e|p] medicamentos.txt efectos.txt
    if len(sys.argv) != 4:
        print("Uso: python3 PIMES.py -[e|p] <archivo-medicamentos> <archivo-efectosSecundarios>")
        sys.exit(1)
        
    modo = sys.argv[1]
    archivo_med = sys.argv[2]
    archivo_efectos = sys.argv[3]
    
    sistema = PIMES()
    
    if modo == "-e":
        sistema.entrenar(archivo_med, archivo_efectos)
    elif modo == "-p":
        sistema.predecir(archivo_med, archivo_efectos)
    else:
        print("Modo invalido. Usa -e para entrenamiento o -p para prediccion.")
        sys.exit(1)

if __name__ == "__main__":
    main()