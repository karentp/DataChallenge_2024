# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:57:25 2025

@author: AMD
"""


import pandas as pd
import matplotlib.pyplot as plt
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Cargar conjuntos de datos
# Datos de entrenamiento (excluyendo el id)
train_X = pd.read_csv("X_train_reg_with_winsorzing.csv").drop(columns=["id"])  
# Etiquetas de entrenamiento
train_y = pd.read_csv("y_train_reg_with_winsorzing.csv")["trq_margin_wins"]    

# Datos de validación
x_val = pd.read_csv("X_val_reg_with_winsorzing.csv").drop(columns=["id"])      
# Etiquetas de validación
y_val = pd.read_csv("y_val_reg_with_winsorzing.csv")["trq_margin_wins"]        

# Datos de prueba
test_X = pd.read_csv("X_test_reg_with_winsorzing.csv").drop(columns=["id"])   
# Etiquetas de prueba 
test_y = pd.read_csv("y_test_reg_with_winsorzing.csv")["trq_margin_wins"]      

# Entrenar el modelo con los datos de entrenamiento
model = NGBRegressor(n_estimators=500, validation_fraction=0.0,verbose=True)
model.fit(train_X, train_y)

# Predecir sobre el conjunto de validación
y_pred_val = model.predict(x_val)

# Calcular el error (MSE) en el conjunto de validación
mse_val = mean_squared_error(y_val, y_pred_val)
print(f"MSE en validación: {mse_val:.4f}")


# Calcular el MSE, MAE y R^2 en el conjunto de validación

mae_val = mean_absolute_error(y_val, y_pred_val)
print(f"MAE en validación: {mae_val:.4f}")

r2_val = r2_score(y_val, y_pred_val)
print(f"R^2 en validación: {r2_val:.4f}")

# Calcular los errores individuales ( valor real menos la predicción)
errores = y_val - y_pred_val

# Graficar la distribución de los errores con un histograma
plt.figure(figsize=(8, 5))
plt.hist(errores, bins=20, color="royalblue", edgecolor="black")
plt.title("Distribución de errores en el conjunto de validación")
plt.xlabel("Error (valor real - predicción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafica Valores Reales vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_pred_val, alpha=0.7, color='steelblue', label='Predicciones')
# Líneas de referencia para observar la igualdad ideal
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', label='Ideal (y=x)')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Valores Reales vs. Predichos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafica Errores vs. Valores Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_pred_val, errores, alpha=0.7, color='darkorange')
plt.axhline(y=0, color='r', linestyle='--', label='Error = 0')
plt.xlabel("Valores Predichos")
plt.ylabel("Error (Valor Real - Predicción)")
plt.title("Errores vs. Valores Predichos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Fin del código

