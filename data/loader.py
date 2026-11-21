"""Read the public EN/BG corpus using only Python's standard library."""

import gzip
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
LANGUAGES = ("bg", "en")
SPLITS = ("train", "validation")


def iter_texts(split: str, lang: str):
    """Yield rows in file order, without the line terminator."""
    if split not in SPLITS or lang not in LANGUAGES:
        raise ValueError("split must be train or validation; lang must be bg or en")
    with gzip.open(DATA_DIR / f"{split}.{lang}.txt.gz", "rt", encoding="utf-8") as f:
        for line in f:
            yield line.removesuffix("\n")


def load_texts(split: str, lang: str) -> list[str]:
    """Load one split and language into a list."""
    return list(iter_texts(split, lang))


def load_split(split: str) -> dict[str, list[str]]:
    """Load one split as {'bg': [...], 'en': [...]} without mixing languages."""
    return {lang: load_texts(split, lang) for lang in LANGUAGES}
