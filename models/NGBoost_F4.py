# Code Jose Pablo Porras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ngboost import NGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Cargar conjuntos de datos
# Datos de entrenamiento (excluyendo el id)
train_X = pd.read_csv("X_train_reg_new.csv").drop(columns=["id"])
# Datos de prueba trq_margin, trq_margin_wins
train_y = pd.read_csv("y_train_reg_new.csv")["trq_margin_wins"]
# Datos de entrenamiento (excluyendo el id)
test_X  = pd.read_csv("X_test_reg_new.csv").drop(columns=["id"])
# Datos de prueba trq_margin, trq_margin_wins
test_y  = pd.read_csv("y_test_reg_new.csv")["trq_margin_wins"]

# Entrenar el modelo con los datos de entrenamiento
model = NGBRegressor(n_estimators=500, validation_fraction=0.0, verbose=True)
model.fit(train_X, train_y)

# Predecir y evaluar sobre test
y_pred = model.predict(test_X)
mse  = mean_squared_error(test_y, y_pred)
mae  = mean_absolute_error(test_y, y_pred)
r2   = r2_score(test_y, y_pred)
bias = np.mean(y_pred - test_y)

print(f"MSE test:  {mse:.4f}")
print(f"MAE test:  {mae:.4f}")
print(f"R² test:   {r2:.4f}")
print(f"Bias test: {bias:.4f}")

# Graficar la distribución de los errores con un histograma
errores = test_y - y_pred
plt.figure(figsize=(8,5))
plt.hist(errores, bins=18, color="royalblue", edgecolor="black")
plt.title("Distribución de errores en test")
plt.xlabel("Error (real - predicción)")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafica Valores Reales vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(test_y, y_pred, alpha=0.7, color='steelblue', label='Predicciones')
plt.plot([test_y.min(), test_y.max()],
         [test_y.min(), test_y.max()],
         'r--', label='Ideal (y=x)')
plt.xlabel("Valores Reales")
plt.ylabel("Valores Predichos")
plt.title("Valores Reales vs. Predichos (Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafica Errores vs. Valores Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_pred, errores, alpha=0.7, color='darkorange')
plt.axhline(y=0, color='r', linestyle='--', label='Error = 0')
plt.xlabel("Valores Predichos")
plt.ylabel("Error (Real - Predicción)")
plt.title("Errores vs. Valores Predichos (Test)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Fin del codigo 






