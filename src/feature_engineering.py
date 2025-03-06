# src/feature_engineering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml
import joblib

def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def build_tfidf_features(df: pd.DataFrame, text_column: str):
    config = load_config()
    max_features = config.get("feature_engineering", {}).get("tfidf_max_features", 500)
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(df[text_column])
    return vectorizer, features

def save_vectorizer(vectorizer, path: str):
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to {path}")

if __name__ == "__main__":
    import pandas as pd
    df = pd.DataFrame({"review": ["Great movie", "Not good"]})
    vectorizer, features = build_tfidf_features(df, "review")
    save_vectorizer(vectorizer, "./models/tfidf_vectorizer.pkl")
