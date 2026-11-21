"""Smoke-test the learner commands from the repository root."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from data.loader import load_split
from grader.tokenlab import MAX_CORPUS_BYTES, MAX_DOCUMENTS, load_artifact, score
from starter import build
from starter.corpus import quick_split, rows_by_language

ROOT = Path(__file__).resolve().parents[1]


class StarterWorkflowTest(unittest.TestCase):
    def cli(self, module, *args, check=True):
        return subprocess.run(
            [sys.executable, "-m", f"starter.{module}", *map(str, args)],
            cwd=ROOT, capture_output=True, text=True, check=check, timeout=30,
        )

    def test_readme_byte_build_and_quick_comparison(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "output" / "byte.json"
            self.cli("build", "byte", "--out", artifact)
            self.assertEqual(json.loads(artifact.read_text(encoding="utf-8")), {
                "version": 1, "tokenizer": {"type": "byte"},
            })
            report = self.cli("evaluate", "--examples", artifact).stdout.splitlines()
        self.assertIn("train=8, validation=8", report[0])
        rows = list(csv.DictReader(report[1:], delimiter="\t"))
        self.assertEqual({(row["artifact"], row["scope"]) for row in rows}, {
            (name, scope) for name in (str(artifact), "example-word", "example-bpe")
            for scope in ("overall", "bg", "en")
        })
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(float(row["reference_bpb"]) > 0 for row in rows))
        self.assertTrue(all(row["submitted_bpb"] == "-" for row in rows))
        overall = {row["artifact"]: row for row in rows if row["scope"] == "overall"}
        self.assertEqual(len({row["original_bytes"] for row in overall.values()}), 1)
        self.assertNotEqual(overall[str(artifact)]["total_bits"], overall["example-bpe"]["total_bits"])

    def test_evaluator_accepts_multiple_files_with_optional_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline, counted = (Path(temp) / name for name in ("byte.json", "ngram.json"))
            self.cli("build", "byte", "--out", baseline)
            payload = json.loads(baseline.read_text(encoding="utf-8"))
            payload["ngram"] = {"order": 1, "counts": {"en": [[[97], 2]], "bg": []}}
            counted.write_text(json.dumps(payload), encoding="utf-8")
            report = self.cli("evaluate", baseline, counted, "--rows", 2).stdout.splitlines()
        self.assertIn("train=2, validation=2", report[0])
        rows = [row for row in csv.DictReader(report[1:], delimiter="\t") if row["scope"] == "overall"]
        self.assertEqual([row["artifact"] for row in rows], [str(baseline), str(counted)])
        self.assertEqual(rows[0]["submitted_bpb"], "-")
        self.assertGreater(float(rows[1]["submitted_bpb"]), 0)

    def test_all_tokenizers_can_be_reused_in_a_paired_count_artifact(self):
        # Fixture results for the student's TODOs, not trainer implementations.
        vocab, merges = ["и", "the"], [[208, 176]]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pairs = []
            for kind in ("byte", "word", "bpe"):
                baseline, counted = (root / f"{kind}{suffix}.json" for suffix in ("", "-ngram"))
                args = ["build", kind, "--rows", "32", "--size", "4096" if kind == "word" else "128"]
                with patch.object(sys, "argv", [*args, "--out", str(baseline)]), \
                     patch.object(build, "train_word", return_value=vocab), \
                     patch.object(build, "train_bpe", return_value=merges):
                    build.main()
                counts = {"order": 3, "counts": {"en": [[[97], 2]], "bg": [[[208], 3]]}}
                count_args = ["build", kind, "--rows", "32", "--order", "3",
                              "--tokenizer-from", str(baseline), "--out", str(counted)]
                with patch.object(sys, "argv", count_args), \
                     patch.object(build, "train_ngrams", return_value=counts) as trainer:
                    build.main()
                training, encode, order = trainer.call_args.args
                self.assertEqual({lang: len(rows) for lang, rows in training.items()}, {"bg": 32, "en": 32})
                self.assertEqual(order, 3)
                expected = {
                    "byte": list("the и а".encode("utf-8")),
                    "word": [257, 32, 256, 32, 208, 176],
                    "bpe": [116, 104, 101, 32, 208, 184, 32, 256],
                }
                self.assertEqual(encode("the и а"), expected[kind])
                stage_one = load_artifact(baseline.read_bytes())
                stage_two = load_artifact(counted.read_bytes())
                self.assertEqual(stage_one["tokenizer"], stage_two["tokenizer"])
                self.assertEqual(stage_two["ngram"], counts)
                self.assertNotIn("ngram", stage_one)
                pairs.append((baseline, counted))
            report = self.cli("evaluate", "--rows", 32, *(path for pair in pairs for path in pair)).stdout.splitlines()
            rows = {row["artifact"]: row for row in csv.DictReader(report[1:], delimiter="\t") if row["scope"] == "overall"}
            self.assertEqual(len(rows), 6)
            for baseline, counted in pairs:
                first, second = rows[str(baseline)], rows[str(counted)]
                self.assertEqual(first["reference_bpb"], second["reference_bpb"])
                self.assertEqual(first["total_bits"], second["total_bits"])
                self.assertEqual(first["submitted_bpb"], "-")
                self.assertGreater(float(second["submitted_bpb"]), 0)

    def test_hand_counted_trigram_parity_and_bigram_comparison(self):
        # Count rows by hand for the two tiny training documents, not via a trainer.
        train = {"en": ["aba"], "bg": ["а"]}
        validation = {"en": ["aba"], "bg": ["б"]}
        en = [[[97], 2], [[98], 1], [[97, 98], 1], [[98, 97], 1]]
        bg = [[[208], 1], [[176], 1], [[208, 176], 1]]
        payload = {"version": 1, "tokenizer": {"type": "byte"},
                   "ngram": {"order": 3, "counts": {"en": en + [[[97, 98, 97], 1]], "bg": bg}}}
        trigram = score(payload, train, validation)
        payload["ngram"] = {"order": 2, "counts": {"en": en, "bg": bg}}
        bigram = score(payload, train, validation)
        for scope in ("overall", "en", "bg"):
            for metric in ("accuracy", "bpb"):
                self.assertEqual(trigram[scope][f"submitted_{metric}"], trigram[scope][f"reference_{metric}"])
        self.assertNotEqual(trigram["overall"]["submitted_bpb"], bigram["overall"]["submitted_bpb"])

    def test_invalid_build_does_not_write_an_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            artifact = Path(temp) / "invalid.json"
            result = self.cli("build", "byte", "--out", artifact, "--rows", 0, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--rows must be positive", result.stderr)
            self.assertFalse(artifact.exists())

    def test_starter_uses_original_language_prefixes(self):
        for split in ('train', 'validation'):
            with self.subTest(split=split):
                published = load_split(split)
                expected = {lang: docs[:32] for lang, docs in published.items()}
                self.assertEqual(rows_by_language(split, 32), expected)
                self.assertEqual(quick_split(split, 32), expected)
                first_pair = sum(len(docs[0].encode('utf-8')) for docs in published.values())
                with patch('starter.corpus.MAX_CORPUS_BYTES', first_pair):
                    self.assertEqual(quick_split(split, 32), {lang: docs[:1] for lang, docs in published.items()})

    def test_quick_split_stays_within_scorer_budget(self):
        for split in ("train", "validation"):
            with self.subTest(split=split):
                data = quick_split(split, 10_000)
                self.assertEqual(len(data["bg"]), len(data["en"]))
                self.assertLessEqual(sum(map(len, data.values())), MAX_DOCUMENTS)
                self.assertLessEqual(sum(len(doc.encode("utf-8")) for docs in data.values()
                                         for doc in docs), MAX_CORPUS_BYTES)
                self.assertEqual({lang: docs[:2] for lang, docs in data.items()}, quick_split(split, 2))

    def test_sampling_lab_is_repeatable_and_seeds_can_diverge(self):
        args = ("--prompt", "the robot", "--steps", 8, "--seeds", 1, 2, 7)
        output = self.cli("sample", *args).stdout
        self.assertEqual(output, self.cli("sample", *args).stdout)
        prompt, greedy, *weighted = output.splitlines()
        self.assertEqual(prompt, "prompt: 'the robot'")
        self.assertTrue(greedy.startswith("greedy: reads "))
        self.assertEqual([row.split(":")[0] for row in weighted],
                         ["weighted seed=1", "weighted seed=2", "weighted seed=7"])
        samples = [row.split(": ", 1)[1] for row in (greedy, *weighted)]
        self.assertTrue(all(len(sample.split()) == 8 for sample in samples))
        self.assertGreater(len(set(samples[1:])), 1)
        self.assertIn("greedy: the\n", self.cli("sample", "--prompt", "a stranger", "--steps", 1).stdout)


if __name__ == "__main__":
    unittest.main()
