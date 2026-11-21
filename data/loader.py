"""Read the public EN/BG corpus using only Python's standard library."""

import gzip
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
LANGUAGES = ("bg", "en")
SPLITS = ("train", "validation")


def iter_rows(split: str):
    """Yield {'lang': ..., 'text': ...} records in published order."""
    if split not in SPLITS:
        raise ValueError("split must be train or validation")
    with gzip.open(DATA_DIR / f"{split}.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def load_rows(split: str) -> list[dict[str, str]]:
    """Load the interleaved rows of one split into RAM."""
    return list(iter_rows(split))


def iter_texts(split: str, lang: str):
    """Yield the passages for one language in their original order."""
    if lang not in LANGUAGES:
        raise ValueError("lang must be bg or en")
    for row in iter_rows(split):
        if row["lang"] == lang:
            yield row["text"]


def load_texts(split: str, lang: str) -> list[str]:
    """Load one split and language into a list."""
    return list(iter_texts(split, lang))


def load_split(split: str) -> dict[str, list[str]]:
    """Load one split as {'bg': [...], 'en': [...]} without mixing languages."""
    groups = {lang: [] for lang in LANGUAGES}
    for row in iter_rows(split):
        groups[row["lang"]].append(row["text"])
    return groups
