"""Compare artifacts on a small slice of public BG and EN Wikipedia."""

import argparse
from pathlib import Path

from grader.tokenlab import ArtifactError, load_artifact, score
from starter.corpus import quick_split

# Tiny format examples only; these tables were not learned from Wikipedia.
EXAMPLES = {
    "example-word": {"version": 1, "tokenizer": {"type": "word", "vocab": ["the", "и"]}},
    "example-bpe": {"version": 1, "tokenizer": {"type": "bpe", "merges": [[208, 176]]}},
}
COLUMNS = (
    "original_bytes", "token_count", "stream_bits", "vocab_bits", "total_bits",
    "reference_accuracy", "reference_bpb", "submitted_accuracy", "submitted_bpb",
)


def cell(value):
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    parser.add_argument("--rows", type=int, default=8, help="max rows per language per split")
    parser.add_argument("--examples", action="store_true", help="compare with two tiny untrained tables")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")

    reference = quick_split("train", args.rows)
    heldout = quick_split("validation", args.rows)
    print("Public Wikipedia rows per language: "
          f"train={len(reference['bg'])}, validation={len(heldout['bg'])}")
    results = []
    for path in args.artifacts:
        try:
            results.append((str(path), score(load_artifact(path.read_bytes()), reference, heldout)))
        except (OSError, ArtifactError) as error:
            parser.error(f"{path}: {error}")
    if args.examples:
        results.extend((name, score(payload, reference, heldout)) for name, payload in EXAMPLES.items())

    print("\t".join(("artifact", "scope", "lossless", *COLUMNS)))
    for name, report in results:
        for scope in ("overall", "bg", "en"):
            metrics = report[scope]
            print("\t".join((name, scope, str(report["metadata"]["lossless"]),
                             *(cell(metrics[key]) for key in COLUMNS))))


if __name__ == "__main__":
    main()
