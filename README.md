# Litter Physics

Local, two-scale simulation of pine-pellet sifting litter boxes. Research fixtures
resolve small material interactions; a reduced household model will reproduce
visits and maintenance. Neither synthetic fixtures nor appealing renders establish
predictive accuracy for a real household.

Status: foundation only. The compiled extension and development gate work; physics,
calibration, and replay are planned, not yet implemented. See [the plan](docs/plan.md)
and [implementation status](docs/status.md).

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
