# flask_app/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import uvicorn

# Define the request model
class ReviewRequest(BaseModel):
    review: str

# Load the trained model and vectorizer
MODEL_PATH = os.getenv("MODEL_PATH", "./models/movie_review_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "./models/tfidf_vectorizer.pkl")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError("Failed to load model or vectorizer") from e

app = FastAPI()

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    review_text = request.review
    # Transform the input text
    features = vectorizer.transform([review_text])
    prediction = model.predict(features)
    label = "Positive" if prediction[0] == 1 else "Negative"
    return {"sentiment": label}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
