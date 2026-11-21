"""Use a deterministic, balanced-by-row-count slice of the public corpus."""

from itertools import islice

from data.loader import LANGUAGES, iter_texts
from grader.tokenlab import MAX_CORPUS_BYTES, MAX_DOCUMENTS


def rows_by_language(split, rows):
    """Take up to rows documents per language, in the published order."""
    return {lang: list(islice(iter_texts(split, lang), rows)) for lang in LANGUAGES}


def quick_split(split, rows):
    """Keep the same number of documents per language within scorer limits."""
    corpus = {lang: [] for lang in LANGUAGES}
    used = 0
    for pair in zip(*(iter_texts(split, lang) for lang in LANGUAGES)):
        pair_bytes = sum(len(doc.encode("utf-8")) for doc in pair)
        if used + pair_bytes > MAX_CORPUS_BYTES:
            break
        for lang, doc in zip(LANGUAGES, pair):
            corpus[lang].append(doc)
        used += pair_bytes
        if len(corpus[LANGUAGES[0]]) >= min(rows, MAX_DOCUMENTS // len(LANGUAGES)):
            break
    return corpus
