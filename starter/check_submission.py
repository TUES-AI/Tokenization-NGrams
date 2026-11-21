"""Check a TokenLab model.json offline with the scorer on tiny EN/BG examples."""

import argparse
from pathlib import Path
import sys

from grader.tokenlab import ArtifactError, MAX_ARTIFACT_BYTES, MAX_COUNT, MAX_COUNT_ROWS, load_artifact, score

# Invented sentences: no corpus downloads or private grading data are needed.
REFERENCE = {"en": ["A cat sat. A cat ran."], "bg": ["Котка спи. Котка тича."]}
HELDOUT = {"en": ["A cat ran."], "bg": ["Котка тича."]}

# These hints explain scorer errors; the scorer alone decides whether a file is valid.
HINTS = {
    "artifact must be bytes and no larger than 6 MiB": (
        "Save compact JSON or reduce the number of vocabulary entries, merges, or n-gram rows."
    ),
    "invalid UTF-8 artifact JSON": "Save one UTF-8 JSON object without a BOM. Check quotes and commas.",
    "duplicate JSON object key": "Use each field name only once within an object.",
    "missing or unexpected artifact fields": (
        "Check field names: use version and tokenizer, with optional ngram. "
        "If you include ngram, its counts object needs both en and bg lists."
    ),
    "unsupported artifact version": "Use the integer 1 for version.",
    "unknown tokenizer type": 'Set tokenizer.type to "byte", "word", or "bpe".',
    "BPE pairs must reference two existing IDs": "Use byte IDs 0 to 255 or IDs from earlier merge rows.",
    "invalid ngram IDs or count": (
        "Use at least one ID and no more than ngram.order IDs per row, all from your tokenizer. "
        f"Use an integer count from 1 to {MAX_COUNT:,}."
    ),
    "duplicate ngram": "Keep just one count row per ID sequence in each language.",
    "ngram counts exceed the row limit": f"Keep at most {MAX_COUNT_ROWS:,} count rows across en and bg.",
}


def main():
    parser = argparse.ArgumentParser(prog="python -m starter.check_submission", description=__doc__)
    parser.add_argument("artifact", type=Path, help="path to your UTF-8 model JSON file")
    args = parser.parse_args()

    try:
        with args.artifact.open("rb") as handle:
            raw = handle.read(MAX_ARTIFACT_BYTES + 1)
        metadata = score(load_artifact(raw), REFERENCE, HELDOUT)["metadata"]
    except OSError as error:
        print(f"Could not read the file: {error.strerror}. Check the path and permissions.", file=sys.stderr)
        return 1
    except ArtifactError as error:
        message = str(error)
        print(f"Invalid artifact: {message}.", file=sys.stderr)
        if message in HINTS:
            print(f"Fix: {HINTS[message]}", file=sys.stderr)
        print("See docs/FORMAT.md for the schema and examples.", file=sys.stderr)
        return 1

    order = metadata["submitted_order"]
    counts = "no submitted n-gram model" if order is None else f"order-{order} submitted n-gram model"
    mode = "" if metadata["lossless"] else " (casefold, not lossless)"
    print(f"Valid locally: {metadata['tokenizer']} tokenizer{mode}; {counts}.")
    print("Scorer ran on tiny English and Bulgarian examples. Nothing was submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
