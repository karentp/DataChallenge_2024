import pandas as pd
from . import paths

def load_data():
    X_train = pd.read_csv(paths.X_TRAIN_PATH)
    y_train = pd.read_csv(paths.Y_TRAIN_PATH)
    X_val = pd.read_csv(paths.X_VAL_PATH)
    X_test = pd.read_csv(paths.X_TEST_PATH)
    return X_train, y_train, X_val, X_test

def prepare_data(X_train, y_train, X_val, X_test):
    df_train = X_train.merge(y_train, on="id")
    X = df_train.drop(columns=["id", "faulty", "trq_margin", "trq_measured"])
    y = df_train["faulty"]

    X_val_clean = X_val.drop(columns=["id", "trq_measured"])
    X_test_clean = X_test.drop(columns=["id", "trq_measured"])

    return X, y, X_val_clean, X_test_clean
