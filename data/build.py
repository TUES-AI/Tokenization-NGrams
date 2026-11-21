"""Rebuild the public EN/BG corpus from pinned Wikipedia Parquet shards."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path

REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
SOURCES = {
    "en": ("20231101.en/train-00000-of-00041.parquet", "382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677"),
    "bg": ("20231101.bg/train-00000-of-00003.parquet", "7be045e443daa2f516400ef115bf1e21f2e2c4698664d4935364f66037564f01"),
}
AT_COLUMNS = ("split", "lang", "row", "page_id", "title", "article_url")
LANG_SCRIPT = {"en": re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿĀ-ž]"), "bg": re.compile(r"[А-Яа-яЍѝ]")}
SENTENCE_SPACE = re.compile(r'''(?<=[.!?…]) (?=[„“"'‘(]?[A-ZА-ЯЍ0-9])''')
END = re.compile(r'''[.!?…][”"’')»]*$''')
ARTIFACT = re.compile(r"[\[\]{}<>|=�]|https?://|www\.|\b[\w.+-]+@[\w.-]+\.\w+", re.IGNORECASE)
EMPTY_PARENS = re.compile(r"\(\s*[:;,/–—-]*\s*\)")
QUOTAS = {"train": 2730, "validation": 510}  # per length bin and language


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rank(domain, lang, value):
    return hashlib.sha256(f"wiki-en-bg-v1\0{domain}\0{lang}\0{value}".encode()).digest()


def dedup_key(row):
    return "".join(c for c in unicodedata.normalize("NFKC", row).casefold() if c.isalnum())


def band(row):
    return (len(row) - 120) // 80


def pick_row(lang, page_id, text):
    """Choose one 120–359-character passage of whole sentence-like units."""
    choices = set()
    for paragraph in [p for p in text[:48_000].split("\n\n") if p.strip()][:8]:
        paragraph = unicodedata.normalize("NFC", " ".join(paragraph.split()))
        paragraph = re.sub(r" +([.,;:!?])", r"\1", EMPTY_PARENS.sub("", paragraph))
        paragraph = " ".join(paragraph.split())
        sentences = SENTENCE_SPACE.split(paragraph)
        for i in range(len(sentences)):
            row = ""
            for sentence in sentences[i:i + 4]:
                row = f"{row} {sentence}".strip()
                if len(row) >= 360:
                    break
                if len(row) < 120 or not END.search(row) or ARTIFACT.search(row):
                    continue
                if any(unicodedata.category(c)[0] == "C" for c in row):
                    continue
                letters = sum(c.isalpha() for c in row)
                if letters * 10 < len(row) * 6 or 10 * len(LANG_SCRIPT[lang].findall(row)) < 9 * letters:
                    continue
                if row.count("(") != row.count(")"):
                    continue
                choices.add(row)
    return min(choices, key=lambda r: (rank("passage", lang, f"{page_id}\0{r}"), r), default=None)


def candidates(path, lang):
    # PyArrow is only needed for rebuilding; students consume plain gzip.
    from pyarrow.parquet import ParquetFile

    with ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=256, columns=["id", "title", "url", "text"]):
            for rec in batch.to_pylist():
                row = pick_row(lang, rec["id"], rec["text"])
                if row:
                    yield rec["id"], rec["title"], rec["url"], row


def gzip_write(path, payload):
    with path.open("wb") as f, gzip.GzipFile(filename="", fileobj=f, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(payload)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def build(paths, out):
    selections = {}
    seen_rows = set()
    for lang, path in paths.items():
        ranked = sorted(candidates(path, lang), key=lambda rec: (rank("article", lang, rec[0]), rec[0]))
        bins = [[], [], []]
        seen_ids = set()
        for rec in ranked:
            page_id, _, _, row = rec
            bucket = bins[band(row)]
            if len(bucket) == sum(QUOTAS.values()):
                continue
            key = dedup_key(row)
            if page_id in seen_ids or key in seen_rows:
                continue
            seen_ids.add(page_id)
            seen_rows.add(key)
            bucket.append(rec)
        if any(len(bucket) != sum(QUOTAS.values()) for bucket in bins):
            raise ValueError(f"insufficient {lang} articles: {[len(bucket) for bucket in bins]}")
        start = 0
        for split, count in QUOTAS.items():
            chosen = [rec for bucket in bins for rec in bucket[start:start + count]]
            selections[split, lang] = sorted(chosen, key=lambda rec: (rank("order", lang, rec[0]), rec[0]))
            start += count

    out.mkdir(parents=True, exist_ok=True)
    attr = io.StringIO(newline="")
    writer = csv.writer(attr, delimiter="\t", lineterminator="\n")
    writer.writerow(AT_COLUMNS)
    manifest = {}
    for (split, lang), chosen in sorted(selections.items()):
        rows = []
        for number, (page_id, title, url, row) in enumerate(chosen, 1):
            writer.writerow((split, lang, number, page_id, title, url))
            rows.append(row)
        filename = f"{split}.{lang}.txt.gz"
        payload = ("\n".join(rows) + "\n").encode("utf-8")
        gzip_write(out / filename, payload)
        manifest[filename] = {
            "articles": len(rows), "rows": len(rows), "utf8_bytes": len(payload),
            "mean_code_points_per_row": round(sum(map(len, rows)) / len(rows), 2),
            "sha256_plaintext": sha256(payload), "sha256_gzip": digest(out / filename),
            "gzip_bytes": (out / filename).stat().st_size,
        }
    payload = attr.getvalue().encode("utf-8")
    gzip_write(out / "attribution.tsv.gz", payload)
    manifest["attribution.tsv.gz"] = {
        "rows_excluding_header": sum(map(len, selections.values())),
        "utf8_bytes": len(payload), "sha256_plaintext": sha256(payload),
        "sha256_gzip": digest(out / "attribution.tsv.gz"),
        "gzip_bytes": (out / "attribution.tsv.gz").stat().st_size,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True, help="Pinned shards as bg.parquet and en.parquet")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {lang: args.source_dir / f"{lang}.parquet" for lang in SOURCES}
    for lang, path in paths.items():
        if digest(path) != SOURCES[lang][1]:
            parser.error(f"source checksum mismatch: {path}")
    build(paths, args.output_dir)


if __name__ == "__main__":
    main()
