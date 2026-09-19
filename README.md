# Litter Physics

Local, two-scale simulation of pine-pellet sifting litter boxes. Research fixtures
resolve small material interactions; a reduced household model will reproduce
visits and maintenance. Neither synthetic fixtures nor appealing renders establish
predictive accuracy for a real household.

Status: foundation and response-table validation library. The compiled extension,
development gate and bounded interpolation work; numerical scenes, calibration and
replay are being implemented, not yet integrated. See [the plan](docs/plan.md),
[implementation status](docs/status.md), [measurement protocol](docs/measurements.md)
and [response-table limits](docs/response-tables.md).

Requirements: Rust 1.98+, Python 3.12-3.14, `uv`, macOS or a compatible CPU platform.

```sh
uv sync --locked
uv run litter-physics --version
make gate
```

Private measurements belong in ignored `measurements/`; generated runs belong in
ignored `outputs/`. Do not commit household videos or credentials.

Development: document, test, commit, and push each milestone. Claude implementation
agents work on bounded components; independent review and integrated verification
remain mandatory. Numerical and physical validation are separate release gates.
