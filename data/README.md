# English and Bulgarian Wikipedia passages

One passage per line, one passage per article. Each split has the same number of rows in both languages. Train and validation share no article IDs within either language edition and no duplicate passages after case, punctuation and whitespace normalization across both editions.

| Split | Rows per language | Bulgarian | English |
| --- | ---: | --- | --- |
| Train | 8,190 | [`train.bg.txt.gz`](train.bg.txt.gz) | [`train.en.txt.gz`](train.en.txt.gz) |
| Validation | 1,530 | [`validation.bg.txt.gz`](validation.bg.txt.gz) | [`validation.en.txt.gz`](validation.en.txt.gz) |

All four files total about 2.2 MB compressed and 6.4 MB as UTF-8 text. These are sampled encyclopedia passages, not a representative sample of either language. Topic or translated-content overlap is possible. There is no tokenizer preprocessing to apply before choosing your own approach.

```python
from data.loader import iter_texts, load_split, load_texts

bg_train = load_texts("train", "bg")  # list[str]
validation = load_split("validation")  # {"bg": [...], "en": [...]}
for text in iter_texts("train", "en"):
    pass  # stream one passage at a time
```

Run from the repository root. The loader needs no third-party packages. The four gzip files have no header or language tags; line endings are LF. See [data license and credits](DATA_LICENSE.md) and [the artifact manifest](manifest.json) for per-file counts, sizes and SHA-256 hashes.

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

The builder verifies the SHA-256 of each input shard. It finds sentence-like passages in the first eight nonempty paragraphs, normalizes text, drops obvious markup and link/contact artifacts, and checks that most letters match the expected script. It picks one passage per article by hash, deduplicates across both languages, and ranks articles by hash within three length bands (120–199, 200–279, 280–359 Unicode code points). Per language and band, the first 2,730 articles go to train and the next 510 to validation. A separate hash fixes row order. Attribution uses those exact row numbers.

[`manifest.json`](manifest.json) contains hashes of the gzip files and their uncompressed UTF-8 bytes. The canonical build used CPython 3.12.13, PyArrow 22.0.0, Unicode 15.0.0 and zlib 1.3.2. Gzip has no embedded filename or timestamp; if a different zlib version emits different compressed bytes, compare the uncompressed hashes.
