"""
add_sentiment.py

Runs comments from data/clean_and_annotated/ through a RoBERTa-based
sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest) and
outputs new CSVs with a sentiment_score column inserted before the bot label.

Output columns: comment, sentiment_score, label
Output location: data/sentiment_annotated/
"""

import os
import pandas as pd
from transformers import pipeline

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR  = os.path.join(BASE_DIR, "data", "clean_and_annotated")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "sentiment_annotated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_CHARS  = 512 #Roberta's token limit

print(f"Loading model: {MODEL_NAME}")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    top_k=None,
    truncation=True,
    max_length=512,
)
print("Model loaded.\n")


def process_file(filename: str) -> None:
    input_path  = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    print(f"Processing: {filename}")

    #Load the files
    df = pd.read_csv(input_path, header=None, names=["comment", "label"])
    total = len(df)
    print(f"  Rows loaded: {total}\n")

    #Score in batches
    comments = df["comment"].fillna("").tolist()
    scores   = []

    for i in range(0, total, BATCH_SIZE):
        batch = [str(c)[:MAX_CHARS] for c in comments[i : i + BATCH_SIZE]]
        results = sentiment_pipeline(batch)
        for j, result in enumerate(results):
            s = {item["label"]: item["score"] for item in result}
            scores.append(round(s.get("positive", 0.0) - s.get("negative", 0.0), 6))
            comment_idx = i + j + 1
            preview = str(comments[i + j])[:60].replace("\n", " ")
            print(f"  [{comment_idx}/{total}] {preview}")

    print()

    #Add the sentiment scores
    df.insert(1, "sentiment_score", scores)

    df.to_csv(output_path, index=False, header=False)
    print(f"  Saved to: {output_path}\n")


if __name__ == "__main__":
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"No CSV files found in {INPUT_DIR}")
    else:
        for fname in csv_files:
            process_file(fname)
        print("All files processed.")
