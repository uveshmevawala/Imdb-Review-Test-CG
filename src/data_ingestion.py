# src/data_ingestion.py
import pandas as pd
import gcsfs
import yaml
import os
from google.cloud import storage

from dotenv import load_dotenv
import os

# Load environment variables from .env file in the project root
load_dotenv()

def load_config(config_path: str = "./config.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ingest_data() -> pd.DataFrame:
    # config = load_config()
    bucket_name = os.getenv("BUCKET_NAME")  # e.g., "gs://your-bucket-name"
    data_path = os.getenv("DATA_PATH")      # e.g., "data/imdb_reviews.csv"
    
    full_path = f"{bucket_name}/{data_path}"
    fs = gcsfs.GCSFileSystem(project=os.getenv("GCP_PROJECT"))
    
    with fs.open(full_path, "r", encoding="utf8") as f:
        df = pd.read_csv(f)
    return df

if __name__ == "__main__":
    df = ingest_data()
    print("Data shape:", df.shape)
