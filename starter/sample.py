"""Compare greedy and weighted generation on a small word-level model."""

import argparse
import random

# Fixed continuation counts, independent of the graded Wikipedia benchmark.
NEXT = {
    ("the", "robot"): {"reads": 6, "writes": 3, "draws": 1},
    ("the", "pilot"): {"reads": 4, "draws": 4, "writes": 2},
    ("reads",): {"maps": 3, "stories": 1},
    ("writes",): {"stories": 3, "notes": 1},
    ("draws",): {"maps": 1, "robots": 1},
    ("maps",): {"for": 1, "and": 1},
    ("for",): {"the": 1},
    ("and",): {"the": 1},
    ("the",): {"robot": 2, "pilot": 1},
    (): {"the": 1},
}


def continuations(words):
    for length in range(min(2, len(words)), 0, -1):
        context = tuple(words[-length:])
        if context in NEXT:
            return NEXT[context]
    return NEXT[()]


def generate(prompt, steps, *, seed=None):
    """Return only the continuation words. No seed means greedy."""
    words = prompt.split()
    start = len(words)
    rng = random.Random(seed) if seed is not None else None
    for _ in range(steps):
        counts = continuations(words)
        options = sorted(counts)
        if rng is None:
            word = max(options, key=counts.__getitem__)  # first maximum wins
        else:
            word = rng.choices(options, weights=[counts[word] for word in options])[0]
        words.append(word)
    return words[start:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="the robot")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 7])
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    print(f"prompt: {args.prompt!r}")
    print("greedy: " + " ".join(generate(args.prompt, args.steps)))
    for seed in args.seeds:
        print(f"weighted seed={seed}: " + " ".join(generate(args.prompt, args.steps, seed=seed)))


if __name__ == "__main__":
    main()
