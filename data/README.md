# English and Bulgarian Wikipedia passages

One passage per line, one passage per article. Each split has the same number of rows in both languages. Train and validation share no article IDs within either language edition and no duplicate passages after case, punctuation and whitespace normalization across both editions.

| Split | Download | Rows | Rows per language |
| --- | --- | ---: | ---: |
| Train | [`train.jsonl.gz`](train.jsonl.gz) | 16,380 | 8,190 |
| Validation | [`validation.jsonl.gz`](validation.jsonl.gz) | 3,060 | 1,530 |

Both files together are about 2.5 MB compressed and 6.9 MB as UTF-8 JSONL. Each LF-terminated line is a JSON object with exactly `lang` (`bg` or `en`) and `text` (the passage). Rows alternate Bulgarian, English, starting with Bulgarian in each split. The migration preserves the texts, split membership and order within each language. These are sampled encyclopedia passages, not a representative sample of either language. Topic or translated-content overlap is possible. There is no tokenizer preprocessing to apply before choosing your own approach.

```python
from data.loader import iter_rows, load_rows

for row in iter_rows("train"):
    lang, text = row["lang"], row["text"]  # stream one record at a time
validation = load_rows("validation")  # list of records in RAM, in file order
```

Run from the repository root. The loader needs no third-party packages. See the [license and article credits](DATA_LICENSE.md) and [manifest](manifest.json). The previous per-language archives are temporarily retained for older download links; use only the two JSONL files above for the assignment.

## Rebuild the distributed files

Building is optional. It downloads about 645 MB and needs PyArrow; using the existing gzip files does not. The source is `wikimedia/wikipedia` (`20231101.bg` and `20231101.en`) at revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`.

From the repository root, with `curl` and `uv`:

```sh
revision=b04c8d1ceb2f5cd4588862100d08de323dccfbaa
source="https://huggingface.co/datasets/wikimedia/wikipedia/resolve/$revision"
mkdir -p wiki-source
curl --fail --location "$source/20231101.bg/train-00000-of-00003.parquet" -o wiki-source/bg.parquet
curl --fail --location "$source/20231101.en/train-00000-of-00041.parquet" -o wiki-source/en.parquet
uv run --no-project --python 3.12.13 --with pyarrow==22.0.0 python -B data/build.py \
    --source-dir wiki-source --output-dir rebuilt-data
```

The builder verifies the SHA-256 of each input shard. It finds sentence-like passages in the first eight nonempty paragraphs, normalizes text, drops obvious markup and link/contact artifacts, and checks that most letters match the expected script. It picks one passage per article by hash, deduplicates across both languages, and ranks articles by hash within three length bands (120–199, 200–279, 280–359 Unicode code points). Per language and band, the first 2,730 articles go to train and the next 510 to validation. A separate hash fixes the order within each language before alternating rows. Attribution counts rows within each split and language, not across the combined file.

[`manifest.json`](manifest.json) records counts, byte sizes and SHA-256 hashes for the gzip files and their decompressed UTF-8 bytes. To check a split's `by_language` hash, filter by the chosen `lang`, take the decoded `text` of each matching row in order, and append one LF to each text before encoding to UTF-8. These hashes match the original corpus partitioned by language. The canonical build used CPython 3.12.13, PyArrow 22.0.0, Unicode 15.0.0 and zlib 1.3.2. Gzip has no embedded filename or timestamp; if a different zlib version emits different compressed bytes, compare the uncompressed hashes.
