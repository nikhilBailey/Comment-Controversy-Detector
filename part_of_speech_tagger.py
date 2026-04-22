import spacy
import csv
from pathlib import Path


MAX_LINE_LENGTH = 1000
MIN_LINE_LENGTH = 3
POS_CATEGORIES = ("NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "OTHER")


def _get_nlp() -> spacy.Language:
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exception:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is required. "
            "Install it with: python -m spacy download en_core_web_sm"
        ) from exception

def import_data(filepaths: list[str]) -> list[str]:
    lines: list[str] = []
    for filepath in filepaths:
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if is_valid_line(line):
                    lines.append(line)
    return lines

def split_annotations(lines: list[str]) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []

    for line in lines:
        if "," not in line:
            continue
        text, label_text = line.rsplit(",", 1)
        label_text = label_text.strip()
        if label_text not in {"0", "1"}:
            continue
        text = text.strip()
        if is_valid_line(text):
            texts.append(text)
            labels.append(int(label_text))

    return texts, labels


def is_valid_line(line: str) -> bool:
    if not line:
        return False
    if len(line) < MIN_LINE_LENGTH or len(line) > MAX_LINE_LENGTH:
        return False
    if not line.isascii():
        return False
    # "[e]" is the only allowed square-bracket token.
    bracket_sanitized = line.replace("[e]", "")
    if "[" in bracket_sanitized or "]" in bracket_sanitized:
        return False
    return True

def tag_lines(lines: list[str]) -> list[list[str]]:
    nlp = _get_nlp()
    tagged_lines: list[list[str]] = []
    for doc in nlp.pipe(lines):
        tagged_lines.append([token.pos_ for token in doc if not token.is_space])
    return tagged_lines


def count_tag_features(tagged_lines: list[list[str]]) -> list[list[int]]:
    def bucket_tag(tag: str) -> str:
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

    rows: list[list[int]] = []
    for line_tags in tagged_lines:
        counts = {category: 0 for category in POS_CATEGORIES}
        for tag in line_tags:
            counts[bucket_tag(tag)] += 1
        rows.append([counts[category] for category in POS_CATEGORIES])
    return rows


def write_featured_data(
    texts: list[str],
    row_tag_counts: list[list[int]],
    labels: list[int],
    output_path: str,
) -> None:
    headers = [
        "text",
        "token_count",
        "noun_count",
        "verb_count",
        "adj_count",
        "adv_count",
        "pron_count",
        "det_count",
        "other_count",
        "is_bot_annotation",
    ]

    output = Path(output_path)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for text, label, counts in zip(texts, labels, row_tag_counts):
            token_count = sum(counts)
            writer.writerow([text, token_count, *counts, label])

if __name__ == "__main__":

    lines = import_data(["data/clean_and_annotated/beast_cleaned.csv", "data/clean_and_annotated/iran_war_cleaned.csv"])
    texts, labels = split_annotations(lines)
    tagged_lines = tag_lines(texts)
    row_tag_counts = count_tag_features(tagged_lines)

    write_featured_data(texts, row_tag_counts, labels, "data/feature_data/beast_iran_pos_tagged.csv")
