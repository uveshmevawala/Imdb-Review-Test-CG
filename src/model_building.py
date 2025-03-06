# src/model_building.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import yaml

def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_model(X, y):
    config = load_config()
    model_params = config.get("model", {})
    test_size = model_params.get("test_size", 0.2)
    random_state = model_params.get("random_state", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    clf = LogisticRegression(C=model_params.get("hyperparameters", {}).get("C", 1.0),
                             max_iter=model_params.get("hyperparameters", {}).get("max_iter", 100))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.4f}")
    return clf, acc

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    # Example dummy run
    import numpy as np
    X_dummy = np.array([[0.1, 0.2, 0.2, 0.3], [0.3, 0.4, 0.4, 0.5],[0.3, 0.4, 0.5, 0.6], [0.3, 0.7, 0.4, 0.9]])
    y_dummy = [0, 1, 0, 1]
    model, acc = train_model(X_dummy, y_dummy)
    save_model(model, "./models/movie_review_model.pkl")
