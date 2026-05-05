import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df_meds = pd.read_csv('lista_medicamentos.txt', sep=' ', header=None, names=['CompoundNum', 'drugBankID'])
### print(df_meds.head())
df_meds['CompoundID'] = 'CID' + df_meds['CompoundNum'].astype(str).str.zfill(9)
### print(df_meds.head())

df_dist = pd.read_csv('matriz_de_distancias.csv', index_col=0)
df_dist = df_dist.loc[df_meds['CompoundID']]
### print(df_dist.head())

df_inter = pd.read_csv('matriz_interaciones_medicamento_medicamento.csv', index_col=0)
df_inter = df_inter.loc[df_meds['CompoundID']]
### print(df_inter.head())

drug_id_objetivo = sys.argv[1]
### print(drug_id_objetivo)

X = df_dist.values.astype('float32')
Y = df_inter.values.astype('float32')

model = Sequential([
    Dense(256, activation='relu', input_dim=548), # Capa oculta 1
    Dense(128, activation='relu'),                # Capa oculta 2
    Dense(64, activation='relu'),                 # Capa oculta 3
    Dense(548, activation='sigmoid')              # Capa de salida (probabilidades independientes)
])

optimizer = Adam(learning_rate=0.0005)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.fit(X, Y, epochs=60, batch_size=16, verbose=0)

idx_target = df_meds[df_meds['drugBankID'] == drug_id_objetivo].index[0]
input_vector = X[idx_target].reshape(1, -1)

y_prob = model.predict(input_vector, verbose=0)[0]
y_true = Y[idx_target]
y_pred_bin = (y_prob > 0.5).astype(int)

precision = precision_score(y_true, y_pred_bin, zero_division=0)
recall = recall_score(y_true, y_pred_bin, zero_division=0)
f1 = f1_score(y_true, y_pred_bin, zero_division=0)

try:
    auc = roc_auc_score(y_true, y_prob)
except ValueError:
    auc = 0.0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")
print(f"AUC: {auc:.4f}")

compound_ids = df_dist.columns
for cid, prob in zip(compound_ids, y_prob):
    print(f"{cid}, {prob:.4f}")