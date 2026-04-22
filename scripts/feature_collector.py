"""
feature_collector.py

Reads comment + bot-label CSVs from data/clean_and_annotated/ and aggregates all
processed rows into one output file:

1. Sentiment: RoBERTa sentiment scores (same model as add_sentiment.py).
2. Polarity: mean-centered deviation of sentiment within the file (same as add_polarity.py).
3. POS: spaCy coarse POS counts per comment (same bucketing as the former part_of_speech_tagger).

Writes one headered feature CSV to data/feature_data/ with columns:
text, token_count, noun_count, verb_count, adj_count, adv_count, pron_count,
det_count, other_count, sentiment_score, polarity_score, is_bot_annotation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import spacy
import torch
from transformers import pipeline

MAX_LINE_LENGTH = 1000
MIN_LINE_LENGTH = 3
POS_CATEGORIES = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "OTHER")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, "data", "clean_and_annotated")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "feature_data")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_BATCH_SIZE = 32
MAX_CHARS = 512
SPACY_PIPE_BATCH = 64


def _get_nlp() -> spacy.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exception:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exception


def load_clean_csv(filepath: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(filepath, header=None, names=["comment", "label"])


def _get_sentiment_pipeline():
    # Fail fast with a clear backend error if torch is unavailable.
    _ = torch.__version__
    print(f"Loading model: {MODEL_NAME}")
    pipe = pipeline(
        "sentiment-analysis",
        model=MODEL_NAME,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    print("Model loaded.\n")
    return pipe


def add_sentiment_scores(df: pd.DataFrame, sentiment_pipe) -> pd.DataFrame:
    out = df.copy()
    comments = out["comment"].fillna("").astype(str).tolist()
    total = len(comments)
    scores: list[float] = []

    for i in range(0, total, SENTIMENT_BATCH_SIZE):
        batch = [str(c)[:MAX_CHARS] for c in comments[i : i + SENTIMENT_BATCH_SIZE]]
        results = sentiment_pipe(batch)
        for j, result in enumerate(results):
            label_scores = {item["label"]: item["score"] for item in result}
            scores.append(
                round(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0), 6)
            )
            comment_idx = i + j + 1
            preview = str(comments[i + j])[:60].replace("\n", " ")
            print(f"  [{comment_idx}/{total}] {preview}")

    out["sentiment_score"] = scores
    return out


def add_polarity_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mean_sentiment = out["sentiment_score"].mean()
    print(f"  Mean sentiment score: {mean_sentiment:.6f}")
    out["polarity_score"] = (out["sentiment_score"] - mean_sentiment).round(6)
    return out


def tag_lines(lines: list[str], batch_size: int = SPACY_PIPE_BATCH) -> list[list[str]]:
    nlp = _get_nlp()
    tagged_lines: list[list[str]] = []
    for doc in nlp.pipe(lines, batch_size=batch_size):
        tagged_lines.append([token.pos_ for token in doc if not token.is_space])
    return tagged_lines


def _bucket_tag(tag: str) -> str:
    if tag == "NOUN":
        return "NOUN"
    if tag in {"VERB", "AUX"}:
        return "VERB"
    if tag == "ADJ":
        return "ADJ"
    if tag == "ADV":
        return "ADV"
    if tag in {"PRON", "PROPN"}:
        return "PRON"
    if tag in {"DET", "NUM"}:
        return "DET"
    return "OTHER"


def count_tag_features(tagged_lines: list[list[str]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for line_tags in tagged_lines:
        counts = {category: 0 for category in POS_CATEGORIES}
        for tag in line_tags:
            counts[_bucket_tag(tag)] += 1
        rows.append([counts[category] for category in POS_CATEGORIES])
    return rows


def add_pos_feature_columns(df: pd.DataFrame, comment_col: str = "comment") -> pd.DataFrame:
    out = df.copy()
    comments = out[comment_col].fillna("").astype(str).tolist()
    tagged = tag_lines(comments)
    count_matrix = count_tag_features(tagged)
    count_df = pd.DataFrame(
        count_matrix,
        columns=[
            "noun_count",
            "verb_count",
            "adj_count",
            "adv_count",
            "pron_count",
            "det_count",
            "other_count",
        ],
    )
    out["token_count"] = count_df.sum(axis=1).astype(int)
    for col in count_df.columns:
        out[col] = count_df[col].astype(int)
    return out


def build_feature_frame(df: pd.DataFrame, sentiment_pipe) -> pd.DataFrame:
    featured = add_sentiment_scores(df, sentiment_pipe)
    featured = add_polarity_scores(featured)
    featured = add_pos_feature_columns(featured)
    featured = featured.rename(columns={"comment": "text", "label": "is_bot_annotation"})
    return featured[
        [
            "text",
            "token_count",
            "noun_count",
            "verb_count",
            "adj_count",
            "adv_count",
            "pron_count",
            "det_count",
            "other_count",
            "sentiment_score",
            "polarity_score",
            "is_bot_annotation",
        ]
    ]


def process_file(filename: str, sentiment_pipe) -> pd.DataFrame:
    input_path = os.path.join(INPUT_DIR, filename)

    print(f"Processing: {filename}")
    df = load_clean_csv(input_path)
    print(f"  Rows loaded: {len(df)}\n")

    featured = build_feature_frame(df, sentiment_pipe)
    return featured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect POS, sentiment, and polarity features into one aggregated CSV."
    )
    parser.add_argument(
        "--output-filename",
        default="combined_feature_data.csv",
        help="Output CSV filename written under data/feature_data/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(INPUT_DIR):
        print(f"Directory not found: {INPUT_DIR}")
    else:
        csv_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".csv"))
        if not csv_files:
            print(f"No CSV files found in {INPUT_DIR}")
        else:
            sentiment_pipeline = _get_sentiment_pipeline()
            all_feature_frames: list[pd.DataFrame] = []
            for fname in csv_files:
                all_feature_frames.append(process_file(fname, sentiment_pipeline))

            combined = pd.concat(all_feature_frames, ignore_index=True)
            Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, args.output_filename)
            combined.to_csv(output_path, index=False)
            print(f"Saved {len(combined)} rows to: {output_path}")
            print("All files processed.")
