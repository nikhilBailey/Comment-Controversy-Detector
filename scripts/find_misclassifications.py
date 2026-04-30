from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import pandas as pandas

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_FEATURE_CSV = os.path.join(
    BASE_DIR, "data", "feature_data", "combined_feature_data.csv"
)

FEATURE_COLUMN_ORDER = [
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
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a saved model over combined_feature_data.csv and split "
                    "into correctly-classified and misclassified output files."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a joblib-saved sklearn pipeline (e.g. svm_rbf_pca_refit_full.joblib).",
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path(DEFAULT_FEATURE_CSV),
        help="Path to the combined feature CSV (default: data/feature_data/combined_feature_data.csv).",
    )
    parser.add_argument(
        "--misclassified-output",
        type=Path,
        required=True,
        help="Where to write the misclassified rows.",
    )
    parser.add_argument(
        "--correct-output",
        type=Path,
        required=True,
        help="Where to write the correctly classified rows.",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    pipeline = joblib.load(args.model_path)

    print(f"Loading data:  {args.feature_csv}")
    data = pandas.read_csv(args.feature_csv)
    feature_matrix = data[FEATURE_COLUMN_ORDER].to_numpy()
    true_labels = data["is_bot_annotation"].astype(int).to_numpy()

    predictions = pipeline.predict(feature_matrix).astype(int)
    probability_matrix = pipeline.predict_proba(feature_matrix)
    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1) if 1 in classes else probability_matrix.shape[1] - 1
    bot_probabilities = probability_matrix[:, positive_index].round(6)

    annotated = data.copy()
    annotated["predicted_label"] = predictions
    annotated["bot_probability"] = bot_probabilities

    correct_mask = predictions == true_labels
    correct_rows = annotated[correct_mask].copy()
    misclassified_rows = annotated[~correct_mask].copy()

    args.correct_output.parent.mkdir(parents=True, exist_ok=True)
    args.misclassified_output.parent.mkdir(parents=True, exist_ok=True)
    correct_rows.to_csv(args.correct_output, index=False)
    misclassified_rows.to_csv(args.misclassified_output, index=False)

    total = len(annotated)
    correct_count = len(correct_rows)
    wrong_count = len(misclassified_rows)
    false_positives = int(((predictions == 1) & (true_labels == 0)).sum())
    false_negatives = int(((predictions == 0) & (true_labels == 1)).sum())
    print(f"Total rows:        {total}")
    print(f"Correct:           {correct_count} ({correct_count / total:.1%})")
    print(f"Misclassified:     {wrong_count} ({wrong_count / total:.1%})")
    print(f"  False positives: {false_positives} (predicted bot, actually human)")
    print(f"  False negatives: {false_negatives} (predicted human, actually bot)")
    print(f"Wrote correct:     {args.correct_output}")
    print(f"Wrote misclassed:  {args.misclassified_output}")


if __name__ == "__main__":
    main()
