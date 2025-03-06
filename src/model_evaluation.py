# src/model_evaluation.py
import joblib
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model(model_path: str, vectorizer_path: str, X_test, y_test):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    # If X_test is raw text, convert it using the vectorizer
    if isinstance(X_test[0], str):
        X_test_transformed = vectorizer.transform(X_test)
    else:
        X_test_transformed = X_test

    print(X_test_transformed)
    y_pred = model.predict(X_test_transformed)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    return cm, report

if __name__ == "__main__":
    # Dummy evaluation
    X_test = ["Great movie"]
    y_test = [1]
    evaluate_model("./models/movie_review_model.pkl", "./models/tfidf_vectorizer.pkl", X_test, y_test)
