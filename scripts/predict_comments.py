from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pandas

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from data_annotating import save_stripped_lines  # noqa: E402
from feature_collector import (  # noqa: E402
    _get_sentiment_pipeline,
    add_pos_feature_columns,
    add_polarity_scores,
    add_sentiment_scores,
)

BASE_DIR = os.path.dirname(SCRIPTS_DIR)

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


def clean_raw_csv_to_text_lines(raw_csv_path: Path, cleaned_text_path: Path) -> None:
    """Run the same cleaning pipeline used at training time, in unannotated mode."""
    save_stripped_lines(raw_csv_path, cleaned_text_path, annotated=False)


def load_cleaned_lines_as_dataframe(cleaned_text_path: Path) -> pandas.DataFrame:
    with cleaned_text_path.open(encoding="utf-8") as cleaned_file:
        comments = [line.rstrip("\n") for line in cleaned_file if line.strip()]
    return pandas.DataFrame({"comment": comments})


def build_features_for_inference(comments_dataframe: pandas.DataFrame) -> pandas.DataFrame:
    sentiment_pipeline_instance = _get_sentiment_pipeline()
    featured = add_sentiment_scores(comments_dataframe, sentiment_pipeline_instance)
    featured = add_polarity_scores(featured)
    featured = add_pos_feature_columns(featured)
    featured = featured.rename(columns={"comment": "text"})
    output_columns = ["text", *FEATURE_COLUMN_ORDER]
    return featured[output_columns]


def run_predictions(
    featured_dataframe: pandas.DataFrame,
    model_pipeline,
) -> pandas.DataFrame:
    feature_matrix = featured_dataframe[FEATURE_COLUMN_ORDER].to_numpy()
    predictions = model_pipeline.predict(feature_matrix)

    output = featured_dataframe.copy()
    output["predicted_label"] = predictions.astype(int)

    if hasattr(model_pipeline, "predict_proba"):
        probability_matrix = model_pipeline.predict_proba(feature_matrix)
        classes = list(model_pipeline.named_steps["classifier"].classes_)
        positive_index = classes.index(1) if 1 in classes else probability_matrix.shape[1] - 1
        output["bot_probability"] = probability_matrix[:, positive_index].round(6)
    else:
        output["bot_probability"] = float("nan")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean raw comments, engineer features, and predict bot labels using a saved model.",
    )
    parser.add_argument(
        "--raw-csv",
        type=Path,
        required=True,
        help="Path to the raw comment CSV (commenterID,commentBody,date_posted format).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to a joblib-saved sklearn pipeline (e.g. svm_rbf_pca_refit_full.joblib).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to write the predictions CSV (features + predicted_label + bot_probability).",
    )
    parser.add_argument(
        "--keep-cleaned-file",
        type=Path,
        default=None,
        help="Optional path; if provided, the intermediate cleaned text file is saved here.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {args.raw_csv}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    print(f"Loading model from: {args.model_path}")
    model_pipeline = joblib.load(args.model_path)
    print("Model loaded.\n")

    cleaned_target = args.keep_cleaned_file
    cleaned_target_was_temporary = cleaned_target is None
    if cleaned_target_was_temporary:
        temporary_directory = tempfile.mkdtemp(prefix="predict_comments_")
        cleaned_target = Path(temporary_directory) / "cleaned.txt"

    print(f"Cleaning raw CSV: {args.raw_csv}")
    clean_raw_csv_to_text_lines(args.raw_csv, cleaned_target)
    comments_dataframe = load_cleaned_lines_as_dataframe(cleaned_target)
    print(f"  Cleaned rows: {len(comments_dataframe)}\n")

    if comments_dataframe.empty:
        raise RuntimeError("Cleaning produced zero usable comments. Check the input file.")

    featured_dataframe = build_features_for_inference(comments_dataframe)
    print(f"\nFeatures built for {len(featured_dataframe)} rows.")

    predictions_dataframe = run_predictions(featured_dataframe, model_pipeline)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_dataframe.to_csv(args.output_csv, index=False)
    print(f"Predictions written to: {args.output_csv}")

    bot_count = int(predictions_dataframe["predicted_label"].sum())
    total_count = len(predictions_dataframe)
    print(f"  Bots predicted: {bot_count}/{total_count} ({bot_count / total_count:.1%})")


if __name__ == "__main__":
    main()
