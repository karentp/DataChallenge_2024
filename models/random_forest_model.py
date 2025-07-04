from sklearn.ensemble import RandomForestClassifier

def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model
