# Contributing

Thank you for considering a contribution to `gaussian-hsmm`.

## Before opening a change

- Search existing issues to avoid duplicating work.
- For a substantial algorithm or API change, open an issue describing the statistical behavior, proposed interface, and validation strategy before implementing it.
- Never include private data, credentials, or proprietary datasets in an issue or test.

## Development setup

```bash
git clone https://github.com/SuryaThirukonda/gaussian-hsmm.git
cd gaussian-hsmm
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

Activate `.venv` using the command appropriate for your shell before installing dependencies.

## Quality checks

Run these before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy src/gaussian_hsmm
pytest --cov=gaussian_hsmm
python -m build
python -m twine check dist/*
```

Use `ruff format .` and `ruff check --fix .` for safe mechanical formatting fixes.

## Testing statistical changes

Algorithm changes need more than a smoke test. Include at least one of:

- a small case verified by brute-force enumeration;
- data simulated from known state/duration parameters;
- comparison against an independent, cited implementation;
- a regression test reproducing a reported failure.

Tests must use deterministic seeds. Assertions should focus on identifiable quantities because hidden-state labels can be permuted.

## Pull requests

- Keep the change focused.
- Add or update tests.
- Update public docstrings, README/API documentation, and the `Unreleased` changelog section when behavior changes.
- Explain numerical and statistical tradeoffs in the pull-request description.
- Avoid changing public behavior solely to make a benchmark look better.

## Versioning

- Patch: compatible bug fixes and documentation.
- Minor: backward-compatible features, and pre-1.0 API refinements documented in the changelog.
- Major: incompatible public-API changes after 1.0.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
