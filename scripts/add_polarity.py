"""
add_polarity.py

Reads sentiment-annotated CSVs from data/sentiment_annotated/ and adds a
polarity_score column — how much a comment's sentiment deviates from the
mean sentiment of its video.

    polarity_score = sentiment_score - mean(sentiment_score for all comments in file)

This is mean-centered deviation: a positive value means the comment is more
positive than average for that video; negative means more negative than average.

Input columns:  comment, sentiment_score, label
Output columns: comment, sentiment_score, polarity_score, label
Output location: data/sentiment_annotated/ (overwrites input files in-place)
"""

import os
import pandas as pd

#Paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "sentiment_annotated")


def process_file(filename: str) -> None:
    filepath = os.path.join(DATA_DIR, filename)

    print(f"Processing: {filename}")

    #Assuming a pre-sentiment annotated file
    df = pd.read_csv(filepath, header=None, names=["comment", "sentiment_score", "label"])
    print(f"  Rows loaded: {len(df)}")

    #the mean sentiment acts as the video's baseline. Comments are scored in comparison with this
    mean_sentiment = df["sentiment_score"].mean()
    print(f"  Mean sentiment score: {mean_sentiment:.6f}")

    #subtract the mean so polarity=0 means "exactly average for this video"
    df["polarity_score"] = (df["sentiment_score"] - mean_sentiment).round(6)

    #reorder so polarity sits right next to sentiment before the label
    df = df[["comment", "sentiment_score", "polarity_score", "label"]]

    df.to_csv(filepath, index=False, header=False)
    print(f"  polarity_score added and saved to: {filepath}\n")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        print("Please run add_sentiment.py first to generate sentiment-annotated CSVs.")
    else:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

        if not csv_files:
            print(f"No CSV files found in {DATA_DIR}")
            print("Please run add_sentiment.py first.")
        else:
            for fname in csv_files:
                process_file(fname)
            print("All files processed.")
