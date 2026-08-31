# Result artifacts

`pilot/` and the three failed-pilot folders are generated locally and ignored.
`final/` contains the frozen configuration, split manifest and hashes,
per-seed metrics, all held-out evaluation trajectories, a learning-curve SVG,
the title-board replay used by the slides, and `analysis.json` with automated
receipt checks.

The committed final evidence is intentionally compact. Intermediate
checkpoints and per-update logs remain ignored; the one selected checkpoint is
published under `artifacts/`.

No result should be cited without its matching `config.sha256`,
`split_manifest.json`, `summary.json`, and `analysis.json`.
