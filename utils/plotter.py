import matplotlib.pyplot as plt

def plot_feature_importance(importances, feature_names, title):
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importancia")
    plt.title(title)
    plt.tight_layout()
    plt.show()
