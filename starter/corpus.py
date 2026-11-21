"""Use a deterministic, balanced-by-row-count slice of the public corpus."""

from itertools import islice

from data.loader import LANGUAGES, iter_rows
from grader.tokenlab import MAX_CORPUS_BYTES, MAX_DOCUMENTS


def rows_by_language(split, rows):
    """Take up to rows documents per language, in the published order."""
    corpus = {lang: [] for lang in LANGUAGES}
    for row in islice(iter_rows(split), rows * len(LANGUAGES)):
        corpus[row["lang"]].append(row["text"])
    return corpus


def quick_split(split, rows):
    """Keep the same number of documents per language within scorer limits."""
    corpus = {lang: [] for lang in LANGUAGES}
    used = 0
    records = iter_rows(split)
    # Adjacent records are one BG/EN pair. Keep or drop them together.
    pairs = zip(records, records, strict=True)
    for pair in islice(pairs, min(rows, MAX_DOCUMENTS // len(LANGUAGES))):
        pair_bytes = sum(len(row["text"].encode("utf-8")) for row in pair)
        if used + pair_bytes > MAX_CORPUS_BYTES:
            break
        for row in pair:
            corpus[row["lang"]].append(row["text"])
        used += pair_bytes
    return corpus
