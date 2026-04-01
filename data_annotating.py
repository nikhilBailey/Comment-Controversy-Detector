"""
Extract the comment field from id,comment,timestamp rows. Records may span
multiple physical lines; a row is complete when it ends with an ISO-8601 UTC
timestamp (comma + YYYY-MM-DDTHH:MM:SSZ).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import TextIO
import re

import emoji
from lingua import Language, LanguageDetectorBuilder

_language_detector = None

# Lingua is unreliable on very short strings
LINGUA_MIN_CHARS = 10

# End of a logical row: comma then UTC Zulu timestamp
_ROW_END_SUFFIX = re.compile(r",\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s*$")


def iter_logical_records(text_stream: TextIO) -> Iterator[str]:
    """
    Buffer physical lines until the accumulated text ends with a timestamp field.

    Each yielded string is one full CSV row (may contain embedded newlines).
    """
    parts: list[str] = []
    for line in text_stream:
        parts.append(line)
        if _ROW_END_SUFFIX.search("".join(parts)):
            yield "".join(parts)
            parts.clear()
    if parts:
        yield "".join(parts)


def _normalize_newlines_to_space(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text)


def _unquote_csv_field(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text


def _get_language_detector():
    global _language_detector
    if _language_detector is None:
        _language_detector = LanguageDetectorBuilder.from_all_languages().build()
    return _language_detector


def strip_outer_fields(record: str) -> str:
    """
    From one full logical row, take text between the first
    comma (after channel id) and the last comma (before timestamp). Commas inside
    the comment are removed. Carriage returns and newlines become spaces.
    Double-quoted CSV fields are unquoted when wrapped in ``"``.
    """
    record = record.rstrip("\r\n")
    first = record.find(",")
    if first == -1:
        return ""
    last = record.rfind(",")
    if last <= first:
        return ""
    middle = record[first + 1 : last]
    middle = _normalize_newlines_to_space(middle)
    middle = _unquote_csv_field(middle)
    return middle.replace(",", "")


def save_stripped_lines(input_path: str | Path, output_path: str | Path) -> None:
    """
    Read logical CSV rows (buffered until a trailing ``,YYYY-MM-DDTHH:MM:SSZ``),
    Cleanse all data by stripping emojis, square brackets, and non-ASCII characters.
    Remove all non-English lines and stripped lines with less than 3 characters.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8", newline="") as f_in, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        for record in iter_logical_records(f_in):

            stripped_line = strip_outer_fields(record)
            stripped_line = remove_square_brackets(stripped_line)
            data_line = remove_emojis(stripped_line, do_replace=True).strip()
        
            lingua_line = remove_emojis(stripped_line, do_replace=False).strip()

            if len(lingua_line) >= LINGUA_MIN_CHARS and not is_in_english(lingua_line):
                continue

            data_line = data_line.encode("ascii", "ignore").decode("ascii")

            if len(data_line) < 3:
                continue

            f_out.write(data_line + "\n")


def remove_square_brackets(text: str) -> str:
    return re.sub(r'\[.*?\]', '', text)


def remove_emojis(text: str, do_replace: bool = True) -> str:
    if do_replace:
        return emoji.replace_emoji(text, replace="[e]")
    else:
        return emoji.replace_emoji(text, replace="")


def is_in_english(text: str) -> bool:
    """
    Uses ``lingua-language-detector`` (pretrained n-gram models over many
    languages)
    """
    text = text.strip()
    if not text:
        return False
    detected = _get_language_detector().detect_language_of(text)
    return detected == Language.ENGLISH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Strip fields/commas, remove emojis, keep English lines only; save to a new file."
    )
    parser.add_argument("input", type=Path, help="Source file path")
    parser.add_argument("output", type=Path, help="Destination file path")
    args = parser.parse_args()
    save_stripped_lines(args.input, args.output)
