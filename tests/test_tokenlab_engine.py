import itertools
import json
import math
import unittest
from fractions import Fraction

from grader import tokenlab


def artifact(kind="byte", **spec):
    return {"version": 1, "tokenizer": {"type": kind, **spec}}


def corpus(en=(), bg=()):
    return {"en": list(en), "bg": list(bg)}


class TokenizerTests(unittest.TestCase):
    def test_utf8_and_whitespace_word_roundtrip(self):
        text = "\tЗдравей!\u00a0е\u0301🙂  a?\n"
        word_vocab = ["Здравей!", "a", "é"]
        byte = tokenlab.build_tokenizer(artifact()["tokenizer"])
        word = tokenlab.build_tokenizer(artifact("word", vocab=word_vocab)["tokenizer"])
        self.assertEqual(byte.encode(text), list(text.encode("utf-8")))
        ids = word.encode(text)
        self.assertEqual(ids, [9, 256] + list("\u00a0е\u0301🙂  a?\n".encode("utf-8")))
        decoded = b"".join(bytes([i]) if i < 256 else word_vocab[i - 256].encode("utf-8") for i in ids)
        self.assertEqual(decoded.decode("utf-8"), text)

    def test_literal_word_boundaries_preserve_original_bytes(self):
        boundaries = "]🙂\\"
        vocab = ["a", "Бé"]
        word = tokenlab.build_tokenizer(artifact("word", vocab=vocab, boundaries=boundaries)["tokenizer"])
        text = "]a🙂]Бé\u00a0β\\a\n"
        ids = word.encode(text)
        self.assertEqual(ids, [93, 256, *"🙂]".encode("utf-8"), 257,
                               *"\u00a0β\\".encode("utf-8"), 256, 10])
        decoded = b"".join(bytes([i]) if i < 256 else vocab[i - 256].encode("utf-8") for i in ids)
        self.assertEqual(decoded, text.encode("utf-8"))
        self.assertEqual(word.encode(""), [])
        self.assertEqual(word.encode(boundaries), list(boundaries.encode("utf-8")))

    def test_word_id_width_is_not_16_bit(self):
        vocab = [f"w{i}" for i in range(65_281)]
        payload = artifact("word", vocab=vocab)
        self.assertEqual(tokenlab.build_tokenizer(payload["tokenizer"]).encode(vocab[-1]), [65_536])
        result = tokenlab.score(payload, corpus(), corpus(en=[vocab[-1]]))
        self.assertEqual(result["en"]["stream_bits"], 40)  # 2 framing + 3 payload bytes.

    def test_bpe_rank_ties_and_reference_implementation(self):
        merges = [[97, 98], [97, 97], [257, 256], [256, 97], [258, 97], [98, 98]]
        bpe = tokenlab.build_tokenizer(artifact("bpe", merges=merges)["tokenizer"])
        ranks = {tuple(pair): i for i, pair in enumerate(merges)}
        self.assertEqual(bpe.encode("aaa"), [257, 97])
        self.assertEqual(bpe.encode("aaab"), [258])
        for symbols in itertools.chain.from_iterable(itertools.product("ab ", repeat=n) for n in range(7)):
            text = "".join(symbols)
            ids = list(text.encode("utf-8"))
            while True:
                candidates = [(ranks[pair], i) for i, pair in enumerate(zip(ids, ids[1:])) if pair in ranks]
                if not candidates:
                    break
                rank, at = min(candidates)
                ids[at:at + 2] = [256 + rank]
            self.assertEqual(bpe.encode(text), ids, text)

    def test_bpe_utf8_and_no_exponential_expansion(self):
        bpe = tokenlab.build_tokenizer({"type": "bpe", "merges": [[208, 145], [256, 32]]})
        self.assertEqual(bpe.encode("Б Б"), [257, 256])
        # Table is legal all the way to ID 65535, but later rows denote strings
        # that cannot possibly fit in an input. Validation must not expand them.
        merges = [[97, 97]] + [[i, i] for i in range(256, 255 + tokenlab.MAX_MERGES)]
        bpe = tokenlab.build_tokenizer({"type": "bpe", "merges": merges})
        self.assertEqual(bpe.encode("aaa"), [256, 97])
        self.assertEqual(bpe.encode("a" * tokenlab.MAX_DOCUMENT_BYTES), [271])


class ScorerTests(unittest.TestCase):
    def test_shape_and_empty_cases(self):
        result = tokenlab.score(artifact(), corpus(), corpus(en=["a", ""]))
        self.assertEqual(set(result), {"overall", "en", "bg", "metadata"})
        self.assertEqual(result["metadata"], {
            "version": 1, "tokenizer": "byte", "lossless": True, "submitted_order": None,
        })
        self.assertEqual(result["en"], {
            "original_bytes": 1, "token_count": 1,
            "stream_bits": 32, "vocab_bits": 16, "total_bits": 48,
            "reference_accuracy": 0, "reference_bpb": 8,
            "submitted_accuracy": None, "submitted_bpb": None,
        })
        self.assertIsNone(result["bg"]["reference_bpb"])
        self.assertIsNone(result["bg"]["reference_accuracy"])
        self.assertEqual(result["bg"]["stream_bits"], 8)
        self.assertEqual(result["overall"]["stream_bits"], 40)
        self.assertEqual(result["overall"]["total_bits"], 56)

    def test_canonical_vocabulary_width_and_document_framing(self):
        result = tokenlab.score(artifact("word", vocab=["é"]), corpus(), corpus(en=["é"]))
        self.assertEqual(result["en"]["vocab_bits"], 48)
        self.assertEqual(result["en"]["stream_bits"], 32)
        self.assertEqual(result["en"]["total_bits"], 80)
        self.assertAlmostEqual(result["en"]["reference_bpb"], math.log2(257) / 2)
        result = tokenlab.score(artifact("word", vocab=["é"]), corpus(), corpus(en=["é", "é"]))
        self.assertEqual(result["en"]["stream_bits"], 48)  # 3 framing bytes, 18 bits padded once.
        result = tokenlab.score(artifact(), corpus(), corpus(en=["a" * 128] + [""] * 127))
        self.assertEqual(result["en"]["stream_bits"], 8 * (2 + 2 + 127 + 128))
        result = tokenlab.score(artifact("bpe", merges=[[97, 98]]), corpus(), corpus(en=["a", "b", "ab"]))
        self.assertEqual(result["en"]["token_count"], 3)
        self.assertEqual(result["en"]["stream_bits"], 80)
        self.assertEqual(result["en"]["vocab_bits"], 56)

    def test_word_boundary_charge_and_default_compatibility(self):
        reference = corpus(en=["aa🙂β"], bg=["βaa"])
        heldout = corpus(en=["aa🙂aa"], bg=["aa"])
        baseline = tokenlab.score(artifact("word", vocab=["aa"]), reference, heldout)
        empty = artifact("word", vocab=["aa"], boundaries="")
        self.assertEqual(baseline, tokenlab.score(empty, reference, heldout))
        self.assertEqual(baseline["overall"]["vocab_bits"], 48)
        self.assertEqual(baseline["en"]["token_count"], 8)

        extended = artifact("word", vocab=["aa"], boundaries="🙂β")
        result = tokenlab.score(extended, reference, heldout)
        reordered = tokenlab.load_artifact(json.dumps(artifact("word", vocab=["aa"], boundaries="β🙂")).encode())
        self.assertEqual(result, tokenlab.score(reordered, reference, heldout))
        self.assertTrue(result["metadata"]["lossless"])
        self.assertEqual(result["en"]["token_count"], 6)
        self.assertEqual(result["en"]["original_bytes"], 8)
        self.assertEqual(result["en"]["stream_bits"], 72)  # 2 framing + 7 padded payload bytes.
        for scope, total in (("en", 176), ("bg", 136), ("overall", 208)):
            self.assertEqual(result[scope]["vocab_bits"], 104)  # 6 baseline + 1 length + 6 UTF-8 bytes.
            self.assertEqual(result[scope]["total_bits"], total)

        boundaries = "".join(chr(0x1F600 + i) for i in range(32))
        max_spec = artifact("word", vocab=[], boundaries=boundaries)["tokenizer"]
        tokenizer = tokenlab.build_tokenizer(max_spec)
        self.assertEqual(tokenizer.vocab_bits, 8 * (3 + 2 + 128))  # ULEB128 byte length is two bytes.
        self.assertEqual(tokenizer.encode(boundaries), list(boundaries.encode("utf-8")))

    def test_casefold_runs_before_boundary_matching_and_stays_lossy(self):
        payload = artifact("word", vocab=["a", "b"], boundaries="x", normalization="casefold")
        tokenizer = tokenlab.build_tokenizer(payload["tokenizer"])
        self.assertEqual(tokenizer.encode("AXB"), [256, 120, 257])
        result = tokenlab.score(payload, corpus(), corpus(en=["AXB"]))
        self.assertFalse(result["metadata"]["lossless"])
        self.assertEqual(result["en"]["token_count"], 3)
        self.assertIsNone(result["overall"]["vocab_bits"])

    def test_reference_math_and_weighted_totals(self):
        result = tokenlab.score(artifact(), corpus(en=["aa"]), corpus(en=["aa"], bg=["ББ"]))
        p_a = (2 + 1 / 256) / 3
        loss = -math.log2(p_a) - math.log2((1 + p_a) / 2)
        self.assertAlmostEqual(result["en"]["reference_bpb"], loss / 2)
        self.assertEqual(result["en"]["reference_accuracy"], 1)
        self.assertAlmostEqual(result["overall"]["reference_bpb"], (loss + 4 * 8) / 6)
        self.assertAlmostEqual(result["overall"]["reference_accuracy"], 2 / 6)

    def test_reference_trigram_and_training_and_eval_boundaries(self):
        split = tokenlab.score(artifact(), corpus(en=["a", "b"]), corpus(en=["ab"]))
        joined = tokenlab.score(artifact(), corpus(en=["ab"]), corpus(en=["ab"]))
        self.assertLess(joined["en"]["reference_bpb"], split["en"]["reference_bpb"])
        self.assertEqual(joined["en"]["reference_accuracy"], 1)
        eval_split = tokenlab.score(artifact(), corpus(en=["ab"]), corpus(en=["a", "b"]))
        self.assertEqual(eval_split["en"]["reference_accuracy"], 0.5)
        trigram = tokenlab.score(artifact(), corpus(en=["abc", "xbd"]), corpus(en=["xbd"]))
        self.assertEqual(trigram["en"]["reference_accuracy"], 2 / 3)

    def test_submitted_fifth_order_empty_tables_and_suffix_ties(self):
        payload = artifact()
        payload["ngram"] = {"order": 5, "counts": {"en": [[list(b"abcde"), 1]], "bg": []}}
        result = tokenlab.score(payload, corpus(), corpus(en=["abcde", "abcde"], bg=["\0"]))
        self.assertEqual(result["metadata"]["submitted_order"], 5)
        self.assertEqual(result["en"]["submitted_accuracy"], 0.2)
        self.assertEqual(result["bg"]["submitted_accuracy"], 1)
        self.assertEqual(result["bg"]["submitted_bpb"], 8)
        self.assertAlmostEqual(result["en"]["submitted_bpb"], (32 - math.log2((1 + 1 / 256) / 2)) / 5)
        payload["ngram"]["counts"]["en"] = [[[97], 8], [[98], 1], [[99, 97], 2], [[99, 98], 2]]
        result = tokenlab.score(payload, corpus(), corpus(en=["ca", "cb"]))
        self.assertEqual(result["en"]["submitted_accuracy"], 0.25)
        self.assertEqual(result["en"]["reference_accuracy"], 0)

    def test_model_probabilities_are_exactly_normalized(self):
        model = tokenlab._CountModel(5, 256)
        model.train(list(b"ababbab"))
        for text in [b"", b"a", b"bb", b"abbb", b"zzzz"]:
            context = b"".join(tokenlab._ID.pack(i) for i in text)
            outcomes = [model._probability_and_best(context, i) for i in range(256)]
            probabilities = [Fraction(num, den) for num, den, _ in outcomes]
            self.assertEqual(sum(probabilities), 1)
            self.assertGreater(min(probabilities), 0)
            self.assertEqual({best for _, _, best in outcomes}, {probabilities.index(max(probabilities))})

    def test_lossy_experiments_still_use_original_bytes(self):
        payload = artifact("word", vocab=["strasse"], normalization="casefold")
        payload["ngram"] = {"order": 1, "counts": corpus()}
        result = tokenlab.score(payload, corpus(), corpus(en=["Straße"]))
        self.assertFalse(result["metadata"]["lossless"])
        for region in ("en", "bg", "overall"):
            for key in ("vocab_bits", "stream_bits", "total_bits"):
                self.assertIsNone(result[region][key])
        self.assertEqual(result["en"]["original_bytes"], 7)
        self.assertEqual(result["en"]["token_count"], 1)
        self.assertAlmostEqual(result["en"]["reference_bpb"], math.log2(257) / 7)
        self.assertEqual(result["en"]["submitted_bpb"], result["en"]["reference_bpb"])


class BoundsTests(unittest.TestCase):
    def test_json_bounds_and_strict_parser(self):
        for raw in (b" " * (tokenlab.MAX_ARTIFACT_BYTES + 1), b"[" * 9 + b"]" * 9,
                    b"[" + b"0," * 1_100_000 + b"0]", b'{"a":1,"\\u0061":2}',
                    b'{"a":12345678901}', b'{"a":1e2}', b'{"a":NaN}', b"[]", b'"\xff"'):
            with self.assertRaises(tokenlab.ArtifactError):
                tokenlab.load_artifact(raw)
        payload = artifact("word", vocab=['a"\\' + '[' * 40 + '}'])
        serialized = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.assertEqual(tokenlab.load_artifact(serialized), payload)
        self.assertEqual(tokenlab.score(payload, corpus(), corpus()),
                         tokenlab.score(tokenlab.load_artifact(serialized), corpus(), corpus()))

    def test_artifact_schema_and_limits(self):
        for payload in (
            {"version": True, "tokenizer": {"type": "byte"}},
            {**artifact(), "run_code": "not permitted"},
            artifact("byte", vocab=[]),
            artifact("word", vocab=["word", "word"]),
            artifact("word", vocab=["a\u00a0b"]),
            artifact("word", vocab=[""]),
            artifact("word", vocab=["я" * 129]),
            artifact("word", vocab=["\ud800"]),
            artifact("word", vocab=[""] * (tokenlab.MAX_WORDS + 1)),
            artifact("bpe", merges=[[256, 0]]),
            artifact("bpe", merges=[[True, 0]]),
            artifact("bpe", merges=[[97, 98], [97, 98]]),
            artifact("bpe", merges=[[0, 0]] * (tokenlab.MAX_MERGES + 1)),
        ):
            with self.assertRaises(tokenlab.ArtifactError):
                tokenlab.score(payload, corpus(), corpus())

    def test_word_boundary_validation(self):
        for payload in (
            artifact("word", vocab=[], boundaries=None),
            artifact("word", vocab=[], boundaries=["🙂"]),
            artifact("word", vocab=[], boundaries="🙂🙂"),
            artifact("word", vocab=[], boundaries="\u00a0"),
            artifact("word", vocab=[], boundaries="\ud800"),
            artifact("word", vocab=[], boundaries="".join(chr(0x1F600 + i) for i in range(33))),
            artifact("word", vocab=["a🙂b"], boundaries="🙂"),
            artifact("byte", boundaries=""),
            artifact("bpe", merges=[], boundaries=""),
        ):
            with self.subTest(payload=payload), self.assertRaises(tokenlab.ArtifactError):
                tokenlab.score(payload, corpus(), corpus())

    def test_submission_ngram_validation(self):
        def check(order, rows):
            payload = artifact()
            payload["ngram"] = {"order": order, "counts": corpus(en=rows)}
            with self.assertRaises(tokenlab.ArtifactError):
                tokenlab.score(payload, corpus(), corpus())

        check(6, [])
        check(1, [[[0, 0], 1]])
        check(5, [[[256], 1]])
        check(5, [[[], 1]])
        check(5, [[[0], True]])
        check(5, [[[0], tokenlab.MAX_COUNT + 1]])
        check(5, [[[0], 0]])
        check(5, [[[0], 1], [[0], 2]])
        check(5, [[[0], 1]] * (tokenlab.MAX_COUNT_ROWS + 1))

    def test_invalid_benchmark_inputs_are_configuration_errors(self):
        for reference in (
            {"en": []}, corpus(en=["\ud800"]), corpus(en=["é" * (tokenlab.MAX_DOCUMENT_BYTES // 2 + 1)]),
            corpus(en=[""] * (tokenlab.MAX_DOCUMENTS + 1)),
            corpus(en=["a" * tokenlab.MAX_DOCUMENT_BYTES] * 2, bg=["b"]),
        ):
            with self.assertRaises(ValueError) as err:
                tokenlab.score(artifact(), reference, corpus())
            self.assertNotIsInstance(err.exception, tokenlab.ArtifactError)
