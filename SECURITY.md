# Security policy

## Supported versions

Before the first public release, only the current `main` branch is supported. After publication, this table should be updated for every maintained release line.

| Version | Supported |
|---|---|
| `main` / latest | Yes |
| Older snapshots | No |

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability.

Use GitHub's private vulnerability reporting feature after the repository is created. Until then, contact the maintainer privately using a verified contact method added to the repository profile.

Include:

- affected version or commit;
- minimal reproduction;
- expected impact;
- whether untrusted model/data files are required;
- suggested mitigation, if known.

The maintainer should acknowledge a report within seven days and coordinate disclosure after a fix is available.

## Security scope

This package processes numeric arrays and does not execute model-supplied code. Nevertheless:

- never load untrusted pickle/joblib model files;
- treat NumPy arrays from unknown sources as untrusted input;
- keep NumPy, SciPy, scikit-learn, and hmmlearn updated;
- install only distributions published by the expected PyPI project and verify the project/source links.
