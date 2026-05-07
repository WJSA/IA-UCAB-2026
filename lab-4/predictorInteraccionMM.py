import sys
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Valida argumento de entrada
if len(sys.argv) < 2:
    print("Uso: python predictorInteraccionMM.py <drugBankID>")
    sys.exit(1)

drug_id_objetivo = sys.argv[1]

df_meds = pd.read_csv('lista_medicamentos.txt', sep=' ', header=None, names=['CompoundNum', 'drugBankID'])
df_meds['CompoundID'] = 'CID' + df_meds['CompoundNum'].astype(str).str.zfill(9)

# Valida si el medicamento existe
if drug_id_objetivo not in df_meds['drugBankID'].values:
    print(f"Error: Medicamento {drug_id_objetivo} no encontrado.")
    sys.exit(1)

# Carga de matrices (alineadas por CompoundID)
df_dist = pd.read_csv('matriz_de_distancias.csv', index_col=0)
df_dist = df_dist.loc[df_meds['CompoundID']]

df_inter = pd.read_csv('matriz_interaciones_medicamento_medicamento.csv', index_col=0)
df_inter = df_inter.loc[df_meds['CompoundID']]

features_dist = df_dist.values.astype('float32')
target_inter = df_inter.values.astype('float32')

# Evita data leakage aislando el target
idx_target = df_meds[df_meds['drugBankID'] == drug_id_objetivo].index[0]
X_train = np.delete(features_dist, idx_target, axis=0)
Y_train = np.delete(target_inter, idx_target, axis=0)

# MLP: 3 capas ocultas según instrucciones
model = Sequential([
    Dense(256, activation='relu', input_dim=548),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(548, activation='sigmoid')
])

# LR bajo por matriz dispersa
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005))
model.fit(X_train, Y_train, epochs=60, batch_size=16, verbose=0)

# Predicción sobre caso no visto
input_vector = features_dist[idx_target].reshape(1, -1)
y_prob = model.predict(input_vector, verbose=0)[0]

y_true = target_inter[idx_target]
y_pred_bin = (y_prob > 0.5).astype(int)

# Maneja divisiones por cero en métricas
precision = precision_score(y_true, y_pred_bin, zero_division=0)
recall = recall_score(y_true, y_pred_bin, zero_division=0)
f1 = f1_score(y_true, y_pred_bin, zero_division=0)

# Maneja target con una sola clase
try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")

for cid, prob in zip(df_dist.columns, y_prob):
    print(f"{cid}, {prob:.4f}")