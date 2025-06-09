from data.loader import load_data, prepare_data
from models.random_forest_model import train_rf
from models.xgboost_model import train_xgb
from models.evaluator import print_predictions
from utils.plotter import plot_feature_importance
from config import OUTPUT_PATH
import pandas as pd

def main():
    X_train, y_train, X_val, X_test = load_data()
    X, y, X_val_clean, X_test_clean = prepare_data(X_train, y_train, X_val, X_test)

    # Random Forest
    rf_model = train_rf(X, y)
    y_val_pred_rf = rf_model.predict(X_val_clean)
    y_test_pred_rf = rf_model.predict(X_test_clean)
    print_predictions("validación - RF", y_val_pred_rf)
    print_predictions("test - RF", y_test_pred_rf)
    plot_feature_importance(rf_model.feature_importances_, X.columns, "Importancia - Random Forest")

    # XGBoost
    xgb_model = train_xgb(X, y)
    y_val_pred_xgb = xgb_model.predict(X_val_clean)
    y_test_pred_xgb = xgb_model.predict(X_test_clean)
    plot_feature_importance(xgb_model.feature_importances_, X.columns, "Importancia - XGBoost")

    # Guardar predicciones
    pred_df = pd.DataFrame({
        "id": X_test["id"],
        "faulty_pred_rf": y_test_pred_rf,
        "faulty_pred_xgb": y_test_pred_xgb
    })
    pred_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Archivo '{OUTPUT_PATH}' guardado correctamente.")

if __name__ == "__main__":
    main()
