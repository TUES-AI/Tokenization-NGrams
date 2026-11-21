# TokenLab artifact format, version 1

The grader accepts one UTF-8 JSON file, not Python code, a notebook, or an archive. Train or select your tokenizer and optional counts locally on the public training text. Use public validation text for local evaluation; the hosted grader uses a separate heldout split. Documents are separate strings, and the grader preserves all characters, including any line breaks inside a document.

## Tokenizer

Every artifact needs integer `version: 1` and one `tokenizer`. Unknown fields and duplicate JSON object keys are invalid. Field names and type names are case-sensitive. All IDs are nonnegative integers; JSON booleans do not count as integers.

### UTF-8 bytes

```json
{"version":1,"tokenizer":{"type":"byte"}}
```

Each byte of the original UTF-8 text is its own ID in `0..255`. For example, `AБ` becomes `[65, 208, 145]`. No vocabulary entries or special tokens are required.

### Words with byte fallback

```json
{"version":1,"tokenizer":{"type":"word","vocab":["a","Здравей!"]}}
```

By default, a word is a maximal run of non-whitespace characters (Python Unicode `\S+`, using the same whitespace definition as `str.isspace()`). The list is ordered: `vocab[i]` is ID `256+i`. A listed word becomes one token. Every unlisted word and every separator is emitted as its original UTF-8 bytes (`0..255`). This preserves leading/trailing spaces, tabs, newlines, unknown words, and exact Unicode spelling, with no separate `<unk>` token.

Optional `boundaries` is a string of up to 32 distinct non-whitespace Unicode code points, each treated as an additional separator. The string specifies a set of literal code points, not a pattern; its order has no effect. Words are maximal runs between separators, and vocabulary entries cannot contain any separator. Omit `boundaries` or use the empty string for the same whitespace-only baseline and bit cost.

With the example vocabulary, `a\tБ! ` becomes `[256, 9, 208, 145, 33, 32]`; `a!` does not match `a`. Decode by concatenating literal bytes for IDs below 256 and UTF-8 bytes of the corresponding vocabulary entries for higher IDs, then decoding the result as UTF-8. Each entry must be a unique, nonempty single word of at most 256 UTF-8 bytes. IDs are not restricted to 16 bits: the required width is `bit_length(255 + len(vocab))`, at least 8. Even fallback bytes use this width in the packed stream.

Word tokenizers may set `"normalization":"casefold"` to experiment with Python Unicode case folding before splitting and looking up words. The vocabulary and any extra separators are matched literally against the folded text. The default `"none"` never alters the text being encoded. A `casefold` run is unranked for **all** metrics, even if the specific text happens to round-trip. Its compression metrics are unavailable; its token count and LM measurements are experiment feedback only.

### Byte-pair encoding (BPE)

```json
{"version":1,"tokenizer":{"type":"bpe","merges":[[97,98],[256,99]]}}
```

Start each document as UTF-8 bytes. Merge row `i` creates ID `256+i`; both input IDs must already exist (`0..255+i`), and ordered pairs must be unique. While any adjacent pair has a rank in the table, merge the pair with the lowest row index, and for equal rank merge the leftmost one first. With the example, `abc abc` becomes `[257, 32, 257]`. Merges may span bytes of a Unicode character, spaces, or punctuation, but not document boundaries. Decode each merged ID by traversing its left and right child IDs in order to yield the original bytes. BPE uses 16-bit IDs even for a small table. The scorer stores pair ranks, not recursively expanded token strings, and uses a heap of candidate positions so it does not rescan the document for each merge.

## Optional submitted n-gram counts

```json
{
  "version":1,
  "tokenizer":{"type":"byte"},
  "ngram":{"order":5,"counts":{"en":[[[97],7],[[97,98],4]],"bg":[]}}
}
```

`order` is an integer from 1 to 5. `counts` must contain exactly `en` and `bg`, each a list of `[ids, count]` rows. An ID list is a contiguous n-gram, between 1 and `order` IDs long, all within your tokenizer's alphabet. Counts must be positive integers. Rows may be in any order; an n-gram may appear only once per language. Include counts explicitly at each order you want to use. The grader does not derive shorter n-grams from longer rows, fill a missing language's table, or update submitted counts on the heldout text. Empty lists and omission of `ngram` are both allowed; omission leaves both submitted metrics empty. To produce ordinary count models locally, encode each training document independently, then count every contiguous n-gram up to your chosen order separately in each language.

## How the grader measures size

The benchmark defines a canonical packed layout so whitespace in the JSON and how Unicode is escaped in it cannot change compression scores. This measures a fixed-width stream with the tokenizer's table, not a trained entropy coder and not a compressed JSON upload.

1. **Vocabulary bits:** the table begins with a one-byte version (`1`) and a one-byte tokenizer tag (`0` byte, `1` default word, `2` BPE, `3` word with nonempty `boundaries`). The byte table ends there. For tag `3`, immediately after the header store the unsigned LEB128 UTF-8 byte length of the extra separators, then their UTF-8 bytes in ascending code point order. No separator field is stored for tag `1`. Word and BPE tables then store their entry count as unsigned LEB128. In word order, store each word as an unsigned LEB128 UTF-8 byte length followed by those bytes. In BPE rank order, store each pair as two unsigned big-endian 16-bit IDs. No other metadata is required to reconstruct the tokenizer. Submitted n-gram tables are not used to decode and are not charged here.
2. **Stream bits:** a language section stores the number of documents, followed by the token count of each document, all as unsigned LEB128. Then it stores the document token IDs in order, packed most significant bit first at the tokenizer's fixed width (byte: 8, BPE: 16, word: width defined above). Pad the end of the language section to a byte boundary with zero bits; there is no per-document token padding. Counts preserve boundaries, including empty documents. Each decoder knows its report scope: overall stores `en` then `bg`. Each section's document count is present even if it is zero. Unsigned LEB128 uses seven value bits per byte, low groups first, with the high bit set except in the last byte; zero occupies one byte.
3. **Total bits:** stream bits plus vocabulary bits. Each single-language score charges the table it would need by itself. Overall adds both stream sections and charges the shared table once.

Example: the word vocabulary `["é"]` and one English document `é` cost 48 vocabulary bits (2 header bytes, entry count, length, 2 UTF-8 bytes). The document encodes as ID `256`, so its English stream costs 32 bits (2 count bytes and 9 token bits rounded up to 16). The English total is 80 bits. Text size is 2 original bytes even though it becomes one token.

## How the grader evaluates prediction

The **reference** score trains a fresh trigram model on the same reference documents for every artifact, after encoding them with that artifact's tokenizer. The reference and optional submitted models are evaluated separately. For both, each language has its own count table. History resets before every document, with no BOS/EOS tokens. Prediction uses the previous **actual** tokens, not the model's earlier guesses.

Let `h` be the available context up to `order-1` tokens, `c(h,x)` the count of the context followed by target `x`, and `C(h)` the sum of those counts. The probability adds one unit of lower-order probability mass:

```text
P(x | h) = (c(h,x) + P(x | suffix(h))) / (C(h) + 1)
```

`suffix(h)` drops the oldest token; for the empty history the lower-order distribution is uniform on **all** tokenizer IDs. Missing counts are zero, so an unseen context backs off entirely without losing probability mass. This always gives positive probabilities that sum to one. The reported accuracy is the fraction of target tokens equal to the highest-probability ID; exact ties go to the smallest ID. The model's loss is `sum(-log2(P(actual_token | history)))`. For example, a byte reference consisting only of `aa` gives the first `a` probability `(2+1/256)/3`, and the second `a` probability `(1+(2+1/256)/3)/2`.

`BPB` is this loss divided by the heldout text's **original UTF-8 byte count**, not its token count or character count. Overall accuracy divides total correct predictions by total tokens; overall BPB divides total loss by total original bytes. Neither averages the English and Bulgarian scores. Zero denominators yield `null`.

**How to interpret the scores:** A one-token word can replace many byte predictions, while an unfamiliar word can require many. Token counts hide the cost of a larger vocabulary, and token accuracy can favor frequent or easy units (such as spaces and predictable UTF-8 continuation bytes) without predicting more meaningful text. BPB measures likelihood of the deterministic encoded sequence per original byte, allowing a better comparison of lossless tokenizers. The count model can still place mass on alternative token sequences or sequences that do not decode to valid text. BPB estimates idealized model coding cost; it excludes vocabulary, boundaries, and model storage, unlike the fixed-width tokenizer size calculation above. For case-folded experiments, BPB measures transformed text against the original byte denominator and is not comparable to a lossless score.

## Limits and local API

| Item | Limit |
| --- | --- |
| UTF-8 JSON file | 6 MiB (6,291,456 bytes), no BOM; at most 8 nested objects/arrays |
| Word types | 131,072 ordered entries; 256 UTF-8 bytes per entry |
| Extra word separators | 32 distinct non-whitespace Unicode code points |
| BPE merges | 65,280 (at most 65,536 total IDs including bytes) |
| Submitted n-grams | 100,000 rows across both languages; each count at most 1,000,000,000 |
| Local reference/heldout input | Both use exactly `{"en":[...],"bg":[...]}`; each split at most 4,096 documents and 128 KiB UTF-8 total, with at most 64 KiB UTF-8 per document |

These corpus limits are for bounded grading and local smoke runs. To exercise the same scoring procedure locally, use a deterministic subset of public training rows as `reference` and public validation rows as `heldout`. You can still train the submitted tokenizer and counts on the full training data; if the count table exceeds the artifact limits, choose which rows to keep.

`grader.tokenlab.load_artifact(raw_bytes)` checks JSON size and structure; `score(payload, reference, heldout)` validates the schema and returns metrics for `overall`, `en`, and `bg`. Each contains `original_bytes`, `token_count`, `stream_bits`, `vocab_bits`, `total_bits`, `reference_accuracy`, `reference_bpb`, `submitted_accuracy`, and `submitted_bpb`. `metadata` includes `version`, `tokenizer`, `lossless`, and `submitted_order`. The scorer operates on plain data and never imports or executes submission code. For local count training, `build_tokenizer(payload["tokenizer"]).encode(document)` produces the same IDs the scorer uses.
