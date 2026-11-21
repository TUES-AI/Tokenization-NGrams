"""Public dataset integrity; do not require a Wikipedia dump or network access."""
import csv
import gzip
import hashlib
import json
import unicodedata
import unittest
from pathlib import Path
from urllib.parse import urlparse


DATA = Path(__file__).resolve().parents[1] / 'data'


class PublicCorpus(unittest.TestCase):
    def test_manifest_and_article_lineage(self):
        manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
        seen = set()
        seen_passages = set()
        row_totals = {}
        for name, record in manifest.items():
            blob = (DATA / name).read_bytes()
            content = gzip.decompress(blob)
            self.assertEqual(len(blob), record['gzip_bytes'])
            self.assertEqual(len(content), record['utf8_bytes'])
            self.assertEqual(hashlib.sha256(blob).hexdigest(), record['sha256_gzip'])
            self.assertEqual(hashlib.sha256(content).hexdigest(), record['sha256_plaintext'])
        for lang in ('bg', 'en'):
            for split, expected in (('train', 8190), ('validation', 1530)):
                name = f'{split}.{lang}.txt.gz'
                with gzip.open(DATA / name, mode='rt', encoding='utf-8') as source:
                    rows = source.read().splitlines()
                self.assertEqual(len(rows), expected)
                self.assertEqual(len(rows), manifest[name]['rows'])
                self.assertTrue(all(row.strip() == row and 120 <= len(row) <= 359 for row in rows))
                for row in rows:
                    key = ''.join(c for c in unicodedata.normalize('NFKC', row).casefold() if c.isalnum())
                    self.assertNotIn(key, seen_passages)
                    seen_passages.add(key)
                row_totals[(lang, split)] = len(rows)
        with gzip.open(DATA / 'attribution.tsv.gz', mode='rt', encoding='utf-8', newline='') as source:
            lineage = csv.DictReader(source, delimiter='\t')
            positions = set()
            for entry in lineage:
                key = (entry['lang'], entry['page_id'])
                self.assertNotIn(key, seen, 'One Wikipedia article appeared in multiple public rows.')
                seen.add(key)
                source_url = urlparse(entry['article_url'])
                self.assertEqual(source_url.hostname, f'{key[0]}.wikipedia.org')
                row = int(entry['row'])
                self.assertTrue(1 <= row <= row_totals[key[0], entry['split']])
                position = (key[0], entry['split'], row)
                self.assertNotIn(position, positions)
                positions.add(position)
        self.assertEqual(len(positions), sum(row_totals.values()))
        self.assertEqual(len(positions), manifest['attribution.tsv.gz']['rows_excluding_header'])
