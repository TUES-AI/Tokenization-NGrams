# Helping a student submit

Start with [docs/SUBMISSION.md](docs/SUBMISSION.md) for the upload flow and site errors. For a specific file, run `python -m starter.check_submission "path/to/model.json"` from the repo root. Diagnose the first reported problem against [docs/FORMAT.md](docs/FORMAT.md); edit the student's trainer/exporter rather than changing the grader to accept the file.

`grader/tokenlab.py` contains the actual scoring logic. Use `starter.evaluate` on public rows for metric questions. A passing local check cannot confirm a school session, publish a result, or reproduce the hidden score. Leave the authenticated upload and publication to the student in their own browser. Do not request cookies, login codes, passwords, or private test data.
