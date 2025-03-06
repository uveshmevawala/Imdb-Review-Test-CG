# src/data_preprocessing.py
import re
import pandas as pd

def clean_text(text: str, lowercase: bool = True, remove_punctuation: bool = True) -> str:
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text

def preprocess_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

if __name__ == "__main__":
    sample_df = pd.DataFrame({"review": ["This movie was GREAT!", "I did not enjoy it..."]})
    processed_df = preprocess_data(sample_df, "review")
    print(processed_df)
