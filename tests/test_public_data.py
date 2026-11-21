"""Public dataset integrity; do not require a Wikipedia dump or network access."""
import csv
import gzip
import hashlib
import io
import json
import tempfile
import unicodedata
import unittest
from pathlib import Path
from urllib.parse import urlparse

from data.build import write_corpus
from data.loader import iter_rows, iter_texts, load_rows, load_split, load_texts


DATA = Path(__file__).resolve().parents[1] / 'data'
# Plaintext checksums from the original per-language corpus and article credits.
LEGACY_SHA256 = {
    ('train', 'bg'): 'c8d997867d4064e462ab038952b5598fc72c0fbd90007ae5b692ada97edb0da6',
    ('train', 'en'): 'bc1ab0108d09b62932840072f0554f983362f6e81300e4484f6d33ea5f6a13fb',
    ('validation', 'bg'): '28a0f8a0e80015687584e52fa5abf4f5c4ac73875af26599a0555e37711f5630',
    ('validation', 'en'): '9f795334cd12f235adfa962696dbba9dae807a3cf54533802c8e1b8daf31fc17',
}
ATTRIBUTION = 'attribution.tsv.gz'
ATTRIBUTION_SHA256 = '964a5231d65fcd0e618cd047972f58bc402a02b6bd719e66e2ddb8eaf77394a4'


def sha256(content):
    return hashlib.sha256(content).hexdigest()


class PublicCorpus(unittest.TestCase):
    def test_manifest_and_article_lineage(self):
        manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(set(manifest), {'train.jsonl.gz', 'validation.jsonl.gz', ATTRIBUTION})
        payloads = {}
        for name, record in manifest.items():
            blob = (DATA / name).read_bytes()
            content = payloads[name] = gzip.decompress(blob)
            self.assertEqual(len(blob), record['gzip_bytes'])
            self.assertEqual(len(content), record['utf8_bytes'])
            self.assertEqual(sha256(blob), record['sha256_gzip'])
            self.assertEqual(sha256(content), record['sha256_plaintext'])

        def unique_object(pairs):
            row = dict(pairs)
            self.assertEqual(len(row), len(pairs), 'Duplicate JSON key')
            return row

        seen_passages = set()
        documents = {}
        for split, expected in (('train', 8190), ('validation', 1530)):
            name = f'{split}.jsonl.gz'
            content = payloads[name].decode('utf-8')
            self.assertTrue(content.endswith('\n'))
            self.assertNotIn('\r', content)
            lines = content[:-1].split('\n')
            self.assertEqual(len(lines), 2 * expected)
            self.assertEqual(len(lines), manifest[name]['rows'])
            self.assertEqual(set(manifest[name]['by_language']), {'bg', 'en'})
            groups = {'bg': [], 'en': []}
            records = []
            for index, line in enumerate(lines):
                row = json.loads(line, object_pairs_hook=unique_object)
                self.assertIsInstance(row, dict)
                self.assertEqual(set(row), {'lang', 'text'})
                lang = ('bg', 'en')[index % 2]
                self.assertEqual(row['lang'], lang)
                text = row['text']
                self.assertIs(type(text), str)
                self.assertTrue(text.strip() == text and 120 <= len(text) <= 359)
                key = ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())
                self.assertNotIn(key, seen_passages)
                seen_passages.add(key)
                groups[lang].append(text)
                records.append(row)
            self.assertEqual(list(iter_rows(split)), records)
            self.assertEqual(load_rows(split), records)
            self.assertEqual(load_split(split), groups)
            for lang, rows in groups.items():
                key = split, lang
                documents[key] = rows
                record = manifest[name]['by_language'][lang]
                plaintext = ('\n'.join(rows) + '\n').encode('utf-8')
                self.assertEqual(len(rows), expected)
                self.assertEqual(len(rows), record['rows'])
                self.assertEqual(len(plaintext), record['utf8_bytes'])
                self.assertEqual(round(sum(map(len, rows)) / len(rows), 2), record['mean_code_points_per_row'])
                self.assertEqual(sha256(plaintext), LEGACY_SHA256[key])
                self.assertEqual(sha256(plaintext), record['sha256_plaintext'])
                self.assertEqual(list(iter_texts(split, lang)), rows)
                self.assertEqual(load_texts(split, lang), rows)

        self.assertEqual(sha256(payloads[ATTRIBUTION]), ATTRIBUTION_SHA256)
        with io.StringIO(payloads[ATTRIBUTION].decode('utf-8'), newline='') as source:
            lineage = csv.DictReader(source, delimiter='\t')
            selections = {key: [] for key in documents}
            seen_ids = set()
            for entry in lineage:
                lang, page_id = entry['lang'], entry['page_id']
                self.assertNotIn((lang, page_id), seen_ids, 'One article appeared in multiple public rows.')
                seen_ids.add((lang, page_id))
                self.assertEqual(urlparse(entry['article_url']).hostname, f'{lang}.wikipedia.org')
                key = entry['split'], lang
                selected = selections[key]
                row = int(entry['row'])
                self.assertEqual(row, len(selected) + 1)
                selected.append((page_id, entry['title'], entry['article_url'], documents[key][row - 1]))
        self.assertEqual({key: len(rows) for key, rows in selections.items()},
                         {key: len(rows) for key, rows in documents.items()})
        self.assertEqual(len(seen_ids), manifest[ATTRIBUTION]['rows_excluding_header'])

        # Exercise the writer without rescanning Wikipedia. Different zlib versions
        # may emit different compressed bytes, so compare decompressed payloads.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_corpus(out, selections)
            self.assertEqual({p.name for p in out.iterdir()}, {*manifest, 'manifest.json'})
            rebuilt = json.loads((out / 'manifest.json').read_bytes())
            self.assertEqual(set(rebuilt), set(manifest))
            for name, record in manifest.items():
                blob = (out / name).read_bytes()
                self.assertEqual(gzip.decompress(blob), payloads[name])
                self.assertEqual(rebuilt[name], {**record, 'gzip_bytes': len(blob), 'sha256_gzip': sha256(blob)})
