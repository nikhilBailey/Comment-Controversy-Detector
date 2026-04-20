from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable
import argparse

import emoji
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lingua import LanguageDetectorBuilder

from data_annotating import iter_logical_records, strip_outer_fields


def iter_comments_from_csv(path: str | Path) -> Iterable[str]:
    csv_path = Path(path)
    with csv_path.open(encoding="utf-8", newline="") as f:
        for record in iter_logical_records(f, annotated=False):
            comment, _ = strip_outer_fields(record, annotated=False)
            if comment:
                yield comment


def count_character_percentages(comments: Iterable[str]) -> tuple[float, float, int]:
    """
    Return:
    - percent of non-ASCII characters over all characters
    - percent of emoji characters over all characters
    - total character count
    """
    total_chars = 0
    non_ascii_chars = 0
    emoji_chars = 0

    for comment in comments:
        total_chars += len(comment)
        non_ascii_chars += sum(1 for ch in comment if ord(ch) > 127)
        emoji_chars += sum(
            match["match_end"] - match["match_start"] for match in emoji.emoji_list(comment)
        )

    if total_chars == 0:
        return 0.0, 0.0, 0

    non_ascii_pct = (non_ascii_chars / total_chars) * 100
    emoji_pct = (emoji_chars / total_chars) * 100
    return non_ascii_pct, emoji_pct, total_chars


def emoji_percentage_per_comment(comments: Iterable[str]) -> list[float]:
    """Return emoji percentage for each individual comment."""
    per_comment_percentages: list[float] = []

    for comment in comments:
        total_chars = len(comment)
        if total_chars == 0:
            per_comment_percentages.append(0.0)
            continue

        emoji_chars = sum(
            match["match_end"] - match["match_start"] for match in emoji.emoji_list(comment)
        )
        per_comment_percentages.append((emoji_chars / total_chars) * 100)

    return per_comment_percentages


def detect_languages(comments: Iterable[str]) -> Counter[str]:
    """Detect language per datapoint and return language counts."""
    detector = LanguageDetectorBuilder.from_all_languages().build()
    counts: Counter[str] = Counter()

    for comment in comments:
        detected = detector.detect_language_of(comment.strip())
        language_name = detected.name if detected is not None else "UNKNOWN"
        counts[language_name] += 1

    return counts


def plot_language_distribution(
    language_counts: Counter[str], top_n: int = 15, *, figures_dir: Path
) -> None:
    """Save a bar chart of the most frequent detected languages."""
    most_common = language_counts.most_common(top_n)
    if not most_common:
        print("No language data to plot.")
        return

    labels = [name for name, _ in most_common]
    values = [count for _, count in most_common]

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "language_distribution.png"

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.title("Language Distribution Across Corpus")
    plt.xlabel("Detected Language")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def plot_percent_below_emoji_threshold(
    emoji_percentages: list[float],
    max_threshold: int = 100,
    *,
    figures_dir: Path,
) -> None:
    """
    Save a line chart: percent of comments with emoji% <= n for each n.
    X-axis: n (%) from 0..max_threshold
    Y-axis: percent of comments with emoji% <= n
    """
    if not emoji_percentages:
        print("No emoji-per-comment data to plot.")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "emoji_percent_below_threshold.png"

    thresholds = list(range(0, max_threshold + 1))
    total = len(emoji_percentages)
    cumulative_percentages = [
        (sum(1 for value in emoji_percentages if value <= n) / total) * 100
        for n in thresholds
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, cumulative_percentages)
    plt.title("Comments With Emoji % At or Below Threshold n")
    plt.xlabel("Emoji Threshold n (%)")
    plt.ylabel("Comments At or Below Threshold (%)")
    plt.ylim(0, 100)
    plt.xlim(0, max_threshold)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute non-ASCII and emoji percentages over the full corpus and "
            "plot language distribution."
        )
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["iran_war.csv", "mr_beast.csv"],
        help="CSV files to include in corpus analysis.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top detected languages to plot.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("figures"),
        help="Directory to write PNG figures (created if missing).",
    )
    args = parser.parse_args()

    all_comments: list[str] = []
    for file_path in args.files:
        all_comments.extend(iter_comments_from_csv(file_path))

    non_ascii_pct, emoji_pct, total_chars = count_character_percentages(all_comments)
    language_counts = detect_languages(all_comments)
    per_comment_emoji_percentages = emoji_percentage_per_comment(all_comments)

    print(f"Files analyzed: {', '.join(args.files)}")
    print(f"Total comments: {len(all_comments)}")
    print(f"Total characters: {total_chars}")
    print(f"Percent non-ASCII characters: {non_ascii_pct:.2f}%")
    print(f"Percent emoji characters: {emoji_pct:.2f}%")
    print("\nTop detected languages:")
    for language, count in language_counts.most_common(args.top_n):
        print(f"- {language}: {count}")

    plot_language_distribution(
        language_counts, top_n=args.top_n, figures_dir=args.figures_dir
    )
    plot_percent_below_emoji_threshold(
        per_comment_emoji_percentages, figures_dir=args.figures_dir
    )


if __name__ == "__main__":
    main()
