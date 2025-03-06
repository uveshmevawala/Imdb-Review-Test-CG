# src/register_model.py
import mlflow
import joblib
import yaml
import os
import dagshub

def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def register_model(model_path: str, vectorizer_path: str, experiment_name="IMDB_Review_Test", run_name="register_model"):
    # Initialize MLflow experiment
    mlflow.set_tracking_uri("https://dagshub.com/uveshmevawala/Imdb-Review-Test-CG.mlflow")  # Replace with your Dagshub MLflow URI
    dagshub.init(repo_owner='uveshmevawala', repo_name='Imdb-Review-Test-CG', mlflow=True)
    print("MLFlow Connection Done")
    mlflow.set_experiment(experiment_name=experiment_name)
    print("MLFlow Experiment Setup Done")
    
    with mlflow.start_run():
        # Log parameters if any
        mlflow.log_param("model_type", "Logistic Regression")
        
        # Log artifacts: model and vectorizer
        mlflow.log_artifact(model_path, artifact_path="model")
        mlflow.log_artifact(vectorizer_path, artifact_path="vectorizer")
        
        # Optionally, log metrics or additional info
        mlflow.log_metric("accuracy", 0.88)  # Replace with actual metric
        
        # Register model in MLflow Model Registry (if needed)
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "IMDB_Sentiment_Model")
        print("Model registered!")

# mlflow.autolog()
    
if __name__ == "__main__":
    register_model("./models/movie_review_model.pkl", "./models/tfidf_vectorizer.pkl")
