"""
Extract the comment field from id,comment,timestamp rows (optional trailing
label ,0 or ,1 when using ``-a``). Records may span multiple physical lines; a
row is complete when it ends with a timestamp field, or with ``timestamp,label``
in annotated mode.
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

# End of a logical row: comma then UTC Zulu timestamp (no label)
_ROW_END_SUFFIX = re.compile(r",\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s*$")

# Annotated rows: timestamp then comma then 0 or 1 (groups: timestamp, label)
_ROW_END_ANNOTATED = re.compile(
    r",(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z),([01])\s*$"
)


def iter_logical_records(text_stream: TextIO, *, annotated: bool = False) -> Iterator[str]:
    """
    Buffer physical lines until the accumulated text ends with a row terminator.

    Each yielded string is one full CSV row (may contain newline characters).
    """
    parts: list[str] = []
    end_pat = _ROW_END_ANNOTATED if annotated else _ROW_END_SUFFIX
    first_line = True
    for line in text_stream:
        if first_line:
            first_line = False
            if annotated and not re.search(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", line
            ):
                continue
        parts.append(line)
        if end_pat.search("".join(parts)):
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


def strip_outer_fields(
    record: str, *, annotated: bool = False
) -> tuple[str, str | None]:
    """
    From one full logical row, extract the comment body between channel id and
    timestamp. Commas inside the comment are removed. Newlines become spaces.
    Double-quoted CSV fields are unquoted when wrapped in ``"``.
    """
    record = record.rstrip("\r\n")
    if annotated:
        m = _ROW_END_ANNOTATED.search(record)
        if not m:
            return "", None
        label = m.group(2)
        head = record[: m.start()]
    else:
        label = None
        last = record.rfind(",")
        if last == -1:
            return "", None
        head = record[:last]

    first = head.find(",")
    if first == -1:
        return "", label
    middle = head[first + 1 :]
    middle = _normalize_newlines_to_space(middle)
    middle = _unquote_csv_field(middle)
    return middle.replace(",", ""), label


def save_stripped_lines(
    input_path: str | Path,
    output_path: str | Path,
    *,
    annotated: bool = False,
) -> None:
    """
    Read logical CSV rows (buffered until a trailing timestamp).
    Cleanse square brackets, emojis, and non-ASCII characters, filter by English / length.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    with input_path.open(encoding="utf-8", newline="") as f_in, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_out:
        for record in iter_logical_records(f_in, annotated=annotated):

            stripped_line, label = strip_outer_fields(record, annotated=annotated)
            stripped_line = remove_square_brackets(stripped_line)
            data_line = remove_emojis(stripped_line, do_replace=True).strip()

            lingua_line = remove_emojis(stripped_line, do_replace=False).strip()

            if len(lingua_line) >= LINGUA_MIN_CHARS and not is_in_english(lingua_line):
                continue

            data_line = data_line.encode("ascii", "ignore").decode("ascii")

            if len(data_line) < 3:
                continue

            if annotated:
                if label is None:
                    continue
                f_out.write(data_line + "," + label + "\n")
            else:
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
    parser.add_argument(
        "-a",
        "--annotated",
        action="store_true",
        help="Input has trailing ,0/,1 labels; rows end with TIMESTAMP,0|1. Reattach label after cleansing.",
    )
    args = parser.parse_args()
    save_stripped_lines(args.input, args.output, annotated=args.annotated)
