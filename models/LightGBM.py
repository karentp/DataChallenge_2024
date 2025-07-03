# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 18:03:45 2025

@author: karsanchez
"""

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

# === Ruta base del proyecto ===
BASE_PATH = r"C:\\Users\\karsanchez\\OneDrive - Compañía Nacional de Fuerza y Luz\\Escritorio\\CD"

# === Cargar conjuntos de datos ===
train_X = pd.read_csv(os.path.join(BASE_PATH, "X_train_reg_new.csv")).drop(columns=["id"])
train_y = pd.read_csv(os.path.join(BASE_PATH, "y_train_reg_new.csv"))["trq_margin_wins"]

test_X = pd.read_csv(os.path.join(BASE_PATH, "X_test_reg_new.csv")).drop(columns=["id"])
test_y = pd.read_csv(os.path.join(BASE_PATH, "y_test_reg_new.csv"))["trq_margin_wins"]

# === Entrenamiento del modelo ===
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(train_X, train_y)

# === Predicción ===
y_pred_test = model.predict(test_X)

# === Métricas ===
mse_test = mean_squared_error(test_y, y_pred_test)
mae_test = mean_absolute_error(test_y, y_pred_test)
r2_test = r2_score(test_y, y_pred_test)
bias_test = np.mean(test_y - y_pred_test)

print("\nEvaluación en Test (torque - con winsorizing):")
print(f"MSE:   {mse_test:.4f}")
print(f"MAE:   {mae_test:.4f}")
print(f"R²:    {r2_test:.4f}")
print(f"Bias:  {bias_test:.4f}")

# === Gráficas ===
errores_test = test_y - y_pred_test

# 1. Reales vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(test_y, y_pred_test, alpha=0.7, color='seagreen', label='Predicciones')
plt.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'r--', label='Ideal (y=x)')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Valores Reales vs.  Predichos (Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Histograma de errores
plt.figure(figsize=(8, 5))
plt.hist(errores_test, bins=18, color="royalblue", edgecolor="black")
plt.title("Distribución de errores en test")
plt.xlabel("Error (real - predicción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Errores vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_pred_test, errores_test, alpha=0.7, color='')
plt.axhline(y=0, color='r', linestyle='--', label='Error = 0')
plt.xlabel("Valores Predichos")
plt.ylabel("Error (Real - Predicción)")
plt.title("Errores vs. Valores Predichos (Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Importancia de variables
feature_names = train_X.columns
importances = model.feature_importances_

feat_imp_df = pd.DataFrame({
    'Variable': feature_names,
    'Importancia': importances
}).sort_values(by='Importancia', ascending=True)

plt.figure(figsize=(8,6))
plt.barh(feat_imp_df['Variable'], feat_imp_df['Importancia'], color='cornflowerblue')
plt.xlabel("Importancia")
plt.title("Importancia de variables - LightGBM")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
