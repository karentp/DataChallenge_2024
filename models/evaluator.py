import pandas as pd

def print_predictions(name, preds):
    print(f"Predicciones de {name} (cuenta por clase):")
    print(pd.Series(preds).value_counts())
