"""Build a TokenLab artifact from public training text only."""

import argparse
import json
from pathlib import Path

from grader.tokenlab import ArtifactError, build_tokenizer, load_artifact, score
from starter.corpus import rows_by_language


def train_word(training, size):
    # TODO: Learn one shared vocabulary from all distinct whitespace-delimited
    # train words. If they exceed size, keep the most frequent, with stable ties.
    # Return an ordered list of strings. See docs/FORMAT.md.
    raise NotImplementedError("TODO: train_word")


def train_bpe(training, size):
    # TODO: Learn up to size byte-pair merges, without crossing documents.
    # Return a list of [left_id, right_id] pairs in merge order.
    raise NotImplementedError("TODO: train_bpe")


def train_ngrams(training, encode, order):
    # TODO: Encode each document separately. Keep BG and EN counts separate.
    # Return {"order": order, "counts": {"en": [...], "bg": [...]}}.
    raise NotImplementedError("TODO: train_ngrams")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("byte", "word", "bpe"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32, help="training rows per language")
    parser.add_argument("--size", type=int, default=128, help="word entries or BPE merges")
    parser.add_argument("--order", type=int, choices=range(1, 6), help="include submitted n-grams")
    parser.add_argument("--tokenizer-from", type=Path, help="reuse an earlier tokenizer-only JSON with --order")
    args = parser.parse_args()
    if args.rows < 1 or args.size < 0:
        parser.error("--rows must be positive and --size must be nonnegative")
    if args.tokenizer_from and (args.order is None or args.tokenizer_from.resolve() == args.out.resolve()):
        parser.error("--tokenizer-from requires --order and a different output file")

    empty = {"en": [], "bg": []}
    tokenizer = {"type": args.kind}
    payload = {"version": 1, "tokenizer": tokenizer}
    try:
        if args.kind != "byte" or args.order is not None:
            training = rows_by_language("train", args.rows)
        if args.tokenizer_from:
            baseline = load_artifact(args.tokenizer_from.read_bytes())
            score(baseline, empty, empty)  # Check schema without training or evaluation.
            if "ngram" in baseline or baseline["tokenizer"]["type"] != args.kind:
                parser.error("--tokenizer-from must name a tokenizer-only JSON of the same kind")
            payload = baseline
        elif args.kind == "word":
            tokenizer["vocab"] = train_word(training, args.size)
        elif args.kind == "bpe":
            tokenizer["merges"] = train_bpe(training, args.size)
        if args.order is not None:
            encoder = build_tokenizer(payload["tokenizer"])
            payload["ngram"] = train_ngrams(training, encoder.encode, args.order)
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        score(load_artifact(raw), empty, empty)  # Enforce upload size and schema before writing.
    except NotImplementedError as error:
        parser.error(f"{error} in starter/build.py")
    except (OSError, ArtifactError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(raw)
    print(f"{args.out}: {len(raw)} bytes of JSON")


if __name__ == "__main__":
    main()
