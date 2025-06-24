# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:41:18 2025

@author: karsanchez
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 17:10:22 2025
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

# === Cargar de archivos ===
train_X = pd.read_csv(os.path.join(BASE_PATH, "X_train_reg_with_winsorzing.csv")).drop(columns=["id"])
train_y = pd.read_csv(os.path.join(BASE_PATH, "y_train_reg_with_winsorzing.csv"))["trq_margin_wins"]
x_val = pd.read_csv(os.path.join(BASE_PATH, "X_val_reg_with_winsorzing.csv")).drop(columns=["id"])
y_val = pd.read_csv(os.path.join(BASE_PATH, "y_val_reg_with_winsorzing.csv"))["trq_margin_wins"]
x_test = pd.read_csv(os.path.join(BASE_PATH, "X_test_reg_with_winsorzing.csv")).drop(columns=["id"])
y_test = pd.read_csv(os.path.join(BASE_PATH, "y_test_reg_with_winsorzing.csv"))["trq_margin_wins"]

# === Entrenamiento del modelo solo con datos de entrenamiento ===
model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42)
model.fit(train_X, train_y)

# === Predicciones ===
y_pred_val = model.predict(x_val)   # Validación
y_pred_test = model.predict(x_test) # Prueba

# === Métricas en Validación ===
mse_val = mean_squared_error(y_val, y_pred_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
r2_val = r2_score(y_val, y_pred_val)
bias_val = np.mean(y_val - y_pred_val)

# === Métricas en Test ===
mse_test = mean_squared_error(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)
bias_test = np.mean(y_test - y_pred_test)

# === Mostrar métricas ===
print("\nEvaluación en Validación:")
print(f"MSE:   {mse_val:.4f}")
print(f"MAE:   {mae_val:.4f}")
print(f"R²:    {r2_val:.4f}")
print(f"Bias:  {bias_val:.4f}")

print("\nEvaluación final en Test:")
print(f"MSE:   {mse_test:.4f}")
print(f"MAE:   {mae_test:.4f}")
print(f"R²:    {r2_test:.4f}")
print(f"Bias:  {bias_test:.4f}")

# === Gráficas para Validación ===

# 1. Valores Reales vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred_val, alpha=0.7, color='seagreen', label='Predicciones')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal (y=x)')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("LightGBM - Valores Reales vs. Predichos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Histograma de errores
errores_val = y_val - y_pred_val
plt.figure(figsize=(8, 5))
plt.hist(errores_val, bins=20, color="royalblue", edgecolor="black")
plt.title("Distribución de errores en el conjunto de validación")
plt.xlabel("Error (valor real - predicción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Errores vs. Valores Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_pred_val, errores_val, alpha=0.7, color='darkorange')
plt.axhline(y=0, color='r', linestyle='--', label='Error = 0')
plt.xlabel("Valores Predichos")
plt.ylabel("Error (Real - Predicho)")
plt.title("Errores vs. Valores Predichos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Importancia de variables ===
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
