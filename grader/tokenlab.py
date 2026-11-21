"""TokenLab v1: deterministic, data-only tokenizer and count-model benchmark."""

from collections import Counter
from heapq import heapify, heappop, heappush
import json
import math
import re
import struct

VERSION = 1
LANGUAGES = ("en", "bg")
MAX_ARTIFACT_BYTES = 6 * 1024 * 1024
MAX_MERGES = 65536 - 256
MAX_WORDS = 131_072
MAX_WORD_BYTES = 256
MAX_WORD_BOUNDARIES = 32
MAX_ORDER = 5
MAX_COUNT_ROWS = 100_000
MAX_COUNT = 1_000_000_000
MAX_DOCUMENTS = 4096  # per split across both languages
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_CORPUS_BYTES = 128 * 1024  # per split across both languages

_WORD = re.compile(r"\S+")
_JSON_MARKS = re.compile(rb'[{}\[\],"\\]')
_ID = struct.Struct("<I")  # Internal LM keys, not stream serialization.


class ArtifactError(ValueError):
    """Invalid submission data. Messages do not include any submitted values."""


def load_artifact(raw: bytes) -> dict:
    """Parse bounded UTF-8 JSON before passing it to score for schema validation."""
    if type(raw) is not bytes or len(raw) > MAX_ARTIFACT_BYTES:
        raise ArtifactError("artifact must be bytes and no larger than 6 MiB")

    # Bound nesting and structural entries before json.loads can allocate containers.
    depth = entries = 0
    quoted = False
    escaped_at = -1
    for match in _JSON_MARKS.finditer(raw):
        char = match[0]
        if quoted:
            if match.start() == escaped_at:
                continue
            if char == b"\\":
                escaped_at = match.end()
            elif char == b'"':
                quoted = False
        elif char == b'"':
            quoted = True
            entries += 1
        elif char in (b"[", b"{"):
            depth += 1
            entries += 1
            if depth > 8:
                raise ArtifactError("artifact JSON is too deeply nested")
        elif char in (b"]", b"}"):
            depth -= 1
            if depth < 0:
                raise ArtifactError("invalid artifact JSON")
        elif char == b",":
            entries += 1
        if entries > 1_100_000:
            raise ArtifactError("artifact JSON has too many entries")

    def object_pairs(pairs):
        obj = dict(pairs)
        if len(obj) != len(pairs):
            raise ArtifactError("duplicate JSON object key")
        return obj

    def invalid_number(_):
        raise ArtifactError("artifact numbers must be bounded integers")

    def integer(value):
        if len(value) > 10:
            invalid_number(value)
        return int(value)

    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=object_pairs,
            parse_int=integer, parse_float=invalid_number, parse_constant=invalid_number,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ArtifactError("invalid UTF-8 artifact JSON") from None
    if type(payload) is not dict:
        raise ArtifactError("artifact must be a JSON object")
    return payload


def _fields(obj, required, optional=()):
    if (type(obj) is not dict or len(obj) > len(required) + len(optional)
            or any(key not in obj for key in required)
            or any(key not in required and key not in optional for key in obj)):
        raise ArtifactError("missing or unexpected artifact fields")


def _int_in(value, low, high):
    return type(value) is int and low <= value <= high


def _uleb_bytes(value):
    return max(1, (value.bit_length() + 6) // 7)


def _merge_key(left, right):
    return (left << 16) | right


class _Tokenizer:
    def __init__(self, kind, width, vocab_size, vocab_bytes, *, words=None, boundary_spaces=None,
                 ranks=None, lossless=True):
        self.kind = kind
        self.width = width
        self.vocab_size = vocab_size
        self.vocab_bits = vocab_bytes * 8
        self.lossless = lossless
        self._words = words
        self._boundary_spaces = boundary_spaces
        self._ranks = ranks

    def encode(self, text: str) -> list[int]:
        """Encode a single document; no model or document boundary IDs are inserted."""
        raw = _document_bytes(text)
        if self.kind == "byte":
            return list(raw)
        if self.kind == "bpe":
            return _bpe(raw, self._ranks)
        if not self.lossless:
            text = text.casefold()
        # Mark extra separators only in the scan view. One space per code point
        # keeps match offsets aligned with the original (or casefolded) text.
        scan = text.translate(self._boundary_spaces) if self._boundary_spaces else text
        tokens = []
        end = 0
        for match in _WORD.finditer(scan):
            tokens.extend(text[end:match.start()].encode("utf-8"))
            word = match[0]
            token = self._words.get(word)
            if token is None:
                tokens.extend(word.encode("utf-8"))
            else:
                tokens.append(token)
            end = match.end()
        tokens.extend(text[end:].encode("utf-8"))
        return tokens


def _bpe(raw, ranks):
    ids = list(raw)
    if not ranks or len(ids) < 2:
        return ids
    heap = []
    for left in range(len(ids) - 1):
        rank = ranks.get(_merge_key(ids[left], ids[left + 1]))
        if rank is not None:
            heap.append((rank, left))
    if not heap:
        return ids
    heapify(heap)
    before = list(range(-1, len(ids) - 1))
    after = list(range(1, len(ids))) + [-1]
    while heap:
        rank, left = heappop(heap)
        right = after[left]
        # Merges change neighboring pairs; stale heap entries require no update.
        if right == -1 or ranks.get(_merge_key(ids[left], ids[right])) != rank:
            continue
        ids[left] = 256 + rank
        successor = after[right]
        after[left], after[right] = successor, -1
        if successor != -1:
            before[successor] = left
            candidate = ranks.get(_merge_key(ids[left], ids[successor]))
            if candidate is not None:
                heappush(heap, (candidate, left))
        predecessor = before[left]
        if predecessor != -1:
            candidate = ranks.get(_merge_key(ids[predecessor], ids[left]))
            if candidate is not None:
                heappush(heap, (candidate, predecessor))
    result = []
    left = 0
    while left != -1:
        result.append(ids[left])
        left = after[left]
    return result


def build_tokenizer(tokenizer_spec: dict) -> _Tokenizer:
    """Validate a v1 tokenizer spec and construct its deterministic encoder."""
    if type(tokenizer_spec) is not dict:
        raise ArtifactError("tokenizer must be an object")
    kind = tokenizer_spec.get("type")
    if kind == "byte":
        _fields(tokenizer_spec, ("type",))
        return _Tokenizer(kind, 8, 256, 2)
    if kind == "word":
        _fields(tokenizer_spec, ("type", "vocab"), ("normalization", "boundaries"))
        normalization = tokenizer_spec.get("normalization", "none")
        if normalization not in ("none", "casefold"):
            raise ArtifactError("unsupported word normalization")
        boundaries = tokenizer_spec.get("boundaries", "")
        if type(boundaries) is not str or len(boundaries) > MAX_WORD_BOUNDARIES:
            raise ArtifactError("word boundaries must be a string of at most 32 code points")
        boundary_chars = set(boundaries)
        if len(boundary_chars) != len(boundaries) or any(c.isspace() for c in boundaries):
            raise ArtifactError("word boundaries must be unique non-whitespace code points")
        vocab = tokenizer_spec["vocab"]
        if type(vocab) is not list or len(vocab) > MAX_WORDS:
            raise ArtifactError("invalid word vocabulary size")
        vocab_bytes = 2 + _uleb_bytes(len(vocab))
        if boundaries:
            try:
                boundary_bytes = "".join(sorted(boundaries)).encode("utf-8")
            except UnicodeEncodeError:
                raise ArtifactError("word boundaries must be valid Unicode") from None
            # The explicit-boundaries word tag replaces the default tag.
            vocab_bytes += _uleb_bytes(len(boundary_bytes)) + len(boundary_bytes)
        words = {}
        for word in vocab:
            if type(word) is not str or len(word) > MAX_WORD_BYTES or not _WORD.fullmatch(word):
                raise ArtifactError("vocabulary entries must be nonempty single words of at most 256 UTF-8 bytes")
            if boundary_chars and not boundary_chars.isdisjoint(word):
                raise ArtifactError("vocabulary entries must not contain word boundaries")
            try:
                size = len(word.encode("utf-8"))
            except UnicodeEncodeError:
                raise ArtifactError("vocabulary entries must be valid Unicode") from None
            if size > MAX_WORD_BYTES or word in words:
                raise ArtifactError("duplicate or oversized vocabulary word")
            words[word] = 256 + len(words)
            vocab_bytes += _uleb_bytes(size) + size
            if vocab_bytes > MAX_ARTIFACT_BYTES:
                raise ArtifactError("word vocabulary is too large")
        return _Tokenizer(kind, (255 + len(words)).bit_length(), 256 + len(words), vocab_bytes,
                          words=words, boundary_spaces=str.maketrans(dict.fromkeys(boundaries, " ")),
                          lossless=normalization == "none")
    if kind == "bpe":
        _fields(tokenizer_spec, ("type", "merges"))
        merges = tokenizer_spec["merges"]
        if type(merges) is not list or len(merges) > MAX_MERGES:
            raise ArtifactError("invalid BPE merge table size")
        ranks = {}
        for rank, pair in enumerate(merges):
            if (type(pair) is not list or len(pair) != 2
                    or not all(_int_in(i, 0, 255 + rank) for i in pair)):
                raise ArtifactError("BPE pairs must reference two existing IDs")
            key = _merge_key(*pair)
            if key in ranks:
                raise ArtifactError("duplicate BPE pair")
            ranks[key] = rank
        return _Tokenizer(kind, 16, 256 + len(ranks), 2 + _uleb_bytes(len(ranks)) + 4 * len(ranks),
                          ranks=ranks)
    raise ArtifactError("unknown tokenizer type")


def _document_bytes(text):
    if type(text) is not str or len(text) > MAX_DOCUMENT_BYTES:
        raise ValueError("expected a bounded corpus document string")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("corpus documents must be valid Unicode") from None
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("corpus document exceeds the UTF-8 byte limit")
    return raw


def _corpus_sizes(corpus):
    if type(corpus) is not dict or len(corpus) != len(LANGUAGES) or any(k not in corpus for k in LANGUAGES):
        raise ValueError("corpus must have en and bg document lists")
    sizes = {}
    total_docs = total_bytes = 0
    for lang in LANGUAGES:
        docs = corpus[lang]
        if type(docs) is not list:
            raise ValueError("corpus language must be a document list")
        total_docs += len(docs)
        if total_docs > MAX_DOCUMENTS:
            raise ValueError("corpus exceeds the document limit")
        sizes[lang] = 0
        for doc in docs:
            size = len(_document_bytes(doc))
            sizes[lang] += size
            total_bytes += size
            if total_bytes > MAX_CORPUS_BYTES:
                raise ValueError("corpus exceeds the UTF-8 byte limit")
    return sizes


class _CountModel:
    def __init__(self, order, vocab_size):
        self.history_bytes = _ID.size * (order - 1)
        self.vocab_size = vocab_size
        self.counts = {}
        self.totals = Counter()
        self.best = {}

    def add(self, context, token, count):
        row = self.counts.setdefault(context, {})
        row[token] = row.get(token, 0) + count
        self.totals[context] += count

    def _advance(self, context, token):
        if self.history_bytes == 0:
            return b""
        return (context + _ID.pack(token))[-self.history_bytes:]

    def _suffixes(self, context):
        return (context[i:] for i in range(len(context), -1, -_ID.size))

    def train(self, tokens):
        context = b""
        for token in tokens:
            for suffix in self._suffixes(context):
                self.add(suffix, token, 1)
            context = self._advance(context, token)

    def _ratio(self, rows, token):
        numerator, denominator = 1, self.vocab_size
        for row, total in rows:
            numerator += row.get(token, 0) * denominator
            denominator *= total + 1
        return numerator, denominator

    def _probability_and_best(self, context, token):
        active = [suffix for suffix in self._suffixes(context) if suffix in self.counts]
        rows = [(self.counts[suffix], self.totals[suffix]) for suffix in active]
        numerator, denominator = self._ratio(rows, token)
        if not active:
            return numerator, denominator, 0
        longest = active[-1]
        best = self.best.get(longest)
        if best is None:
            # An integer count difference of one dominates a strictly subunit
            # difference in backoff probability. Compare suffix odds only at ties.
            row = rows[-1][0]
            largest = max(row.values())
            tied = (candidate for candidate, count in row.items() if count == largest)
            lower = rows[:-1]
            best = max(tied, key=lambda i: (self._ratio(lower, i)[0], -i))
            self.best[longest] = best
        return numerator, denominator, best

    def evaluate(self, tokens):
        context = b""
        hits = 0

        def losses():
            nonlocal context, hits
            for token in tokens:
                num, den, best = self._probability_and_best(context, token)
                hits += token == best
                context = self._advance(context, token)
                if 2 * num > den:
                    # Avoid rounding a near-certain target's probability to 1.
                    yield math.log1p((den - num) / num) / math.log(2)
                else:
                    yield math.log2(den / num)

        loss = math.fsum(losses())
        return hits, loss


def _load_models(spec, vocab_size):
    _fields(spec, ("order", "counts"))
    order = spec["order"]
    if not _int_in(order, 1, MAX_ORDER):
        raise ArtifactError("ngram order must be between 1 and 5")
    _fields(spec["counts"], LANGUAGES)
    models = {}
    total_rows = 0
    for lang in LANGUAGES:
        rows = spec["counts"][lang]
        if type(rows) is not list:
            raise ArtifactError("ngram counts must be lists of rows")
        total_rows += len(rows)
        if total_rows > MAX_COUNT_ROWS:
            raise ArtifactError("ngram counts exceed the row limit")
        model = models[lang] = _CountModel(order, vocab_size)
        for row in rows:
            if type(row) is not list or len(row) != 2:
                raise ArtifactError("expected an ngram row of [ids, count]")
            gram, count = row
            if (type(gram) is not list or not 1 <= len(gram) <= order
                    or not all(_int_in(i, 0, vocab_size - 1) for i in gram)
                    or not _int_in(count, 1, MAX_COUNT)):
                raise ArtifactError("invalid ngram IDs or count")
            context = b"".join(_ID.pack(i) for i in gram[:-1])
            token = gram[-1]
            if token in model.counts.get(context, ()):
                raise ArtifactError("duplicate ngram")
            model.add(context, token, count)
    return order, models


class _Totals:
    def __init__(self):
        self.original_bytes = self.tokens = self.stream_bits = self.reference_hits = 0
        self.submitted_hits = 0
        self.reference_loss = self.submitted_loss = 0.0

    def combine(self, other):
        for key, value in vars(other).items():
            setattr(self, key, getattr(self, key) + value)

    def metrics(self, tokenizer, submitted):
        return {
            "original_bytes": self.original_bytes,
            "token_count": self.tokens,
            "stream_bits": self.stream_bits if tokenizer.lossless else None,
            "vocab_bits": tokenizer.vocab_bits if tokenizer.lossless else None,
            "total_bits": self.stream_bits + tokenizer.vocab_bits if tokenizer.lossless else None,
            "reference_accuracy": self.reference_hits / self.tokens if self.tokens else None,
            "reference_bpb": self.reference_loss / self.original_bytes if self.original_bytes else None,
            "submitted_accuracy": self.submitted_hits / self.tokens if submitted and self.tokens else None,
            "submitted_bpb": self.submitted_loss / self.original_bytes if submitted and self.original_bytes else None,
        }


def score(payload: dict, reference: dict[str, list[str]], heldout: dict[str, list[str]]) -> dict:
    """Score a v1 artifact without I/O, student code, or heldout model updates."""
    _fields(payload, ("version", "tokenizer"), ("ngram",))
    if not _int_in(payload["version"], VERSION, VERSION):
        raise ArtifactError("unsupported artifact version")
    tokenizer = build_tokenizer(payload["tokenizer"])
    order, models = (None, {})
    if "ngram" in payload:
        order, models = _load_models(payload["ngram"], tokenizer.vocab_size)
    _corpus_sizes(reference)
    sizes = _corpus_sizes(heldout)
    overall = _Totals()
    result = {"overall": None}
    for lang in LANGUAGES:
        trained = _CountModel(3, tokenizer.vocab_size)
        for doc in reference[lang]:
            trained.train(tokenizer.encode(doc))
        totals = _Totals()
        totals.original_bytes = sizes[lang]
        framing_bytes = _uleb_bytes(len(heldout[lang]))
        for doc in heldout[lang]:
            tokens = tokenizer.encode(doc)
            totals.tokens += len(tokens)
            framing_bytes += _uleb_bytes(len(tokens))
            hits, loss = trained.evaluate(tokens)
            totals.reference_hits += hits
            totals.reference_loss += loss
            if order is not None:
                hits, loss = models[lang].evaluate(tokens)
                totals.submitted_hits += hits
                totals.submitted_loss += loss
        # One byte-aligned token block per language, with no inter-document padding.
        totals.stream_bits = 8 * (framing_bytes + (totals.tokens * tokenizer.width + 7) // 8)
        result[lang] = totals.metrics(tokenizer, order is not None)
        overall.combine(totals)
    result["overall"] = overall.metrics(tokenizer, order is not None)
    result["metadata"] = {"version": VERSION, "tokenizer": tokenizer.kind,
                          "lossless": tokenizer.lossless, "submitted_order": order}
    return result
