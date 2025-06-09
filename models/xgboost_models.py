from xgboost import XGBClassifier

def train_xgb(X, y):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X, y)
    return model
