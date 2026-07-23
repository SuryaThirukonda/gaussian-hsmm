# Release process

This document describes a future release process. Following it locally does not publish anything unless an explicit upload command is run with valid credentials.

## One-time preparation

1. Create the GitHub repository and update all URLs in `pyproject.toml`, README, changelog, citation metadata, and community files.
2. Confirm the PyPI distribution name is available.
3. Enable branch protection and require the CI workflow.
4. Configure private vulnerability reporting.
5. Create PyPI/TestPyPI accounts with two-factor authentication.

## Prepare a version

1. Ensure `main` is clean and CI passes.
2. Move entries from `Unreleased` into a dated version in `CHANGELOG.md`.
3. Update `__version__` in `src/gaussian_hsmm/__init__.py`.
4. Update `CITATION.cff`.
5. Run all local checks:

```bash
ruff check .
ruff format --check .
mypy src/gaussian_hsmm
pytest --cov=gaussian_hsmm
python -m build
python -m twine check dist/*
```

6. Inspect both archives. Install the wheel into a new virtual environment and run an import/smoke test.
7. Commit the release preparation and tag it using `vX.Y.Z`.

## TestPyPI rehearsal

Use TestPyPI before the first production release. This is intentionally not automated in the repository.

```bash
python -m twine upload --repository testpypi dist/*
```

Install it in a clean environment without resolving dependencies from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ --no-deps gaussian-hsmm==X.Y.Z
```

## Production publishing

Production publishing is a separate, explicit maintainer action. Prefer PyPI Trusted Publishing from a protected GitHub environment rather than storing a long-lived API token. Do not add an automatic publish workflow until the repository and protected environment exist and have been reviewed.

After publication, verify the PyPI metadata, hashes, source links, wheel installation, import path, and displayed README.
