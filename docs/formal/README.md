# Formal simulation manuscript

- Read [`simulation.pdf`](simulation.pdf); edit [`simulation.tex`](simulation.tex) and
  [`sections/`](sections/).
- Scientific baseline: `9a66a559ed478629d514756662a8fd98a8f52f50`. Source links are pinned;
  this is a historical derivation/audit, not an automatically updated description of HEAD.
- Includes DEM/contact, household transport, paste/liquid constitutive mathematics,
  MLS-MPM/APIC transfers, rigid coupling, signed energy diagnostics, calibration,
  response interpolation, worked calculations and equation-to-code navigation.
- Method diagrams are TikZ; the interpolation-kernel plot is analytic; refinement
  plots use archived **actual native results**, including failed criteria.
- No measured household validation or full physical energy-closure claim is made.

Build from the repository root:

```sh
uv sync --locked
brew install tectonic  # macOS; Tectonic 0.17.0 used for the checked-in PDF
make manuscript
uv run python docs/formal/build.py --check
```

Tectonic downloads/caches its TeX resources on first use, so the first compilation
requires network access. `make manuscript` verifies the evidence hash, generates
CSV/LaTeX tables and dimensional calculations, then compiles `simulation.pdf`.
It rejects overfull boxes, unresolved references/citations and missing characters.
It does not run the simulation or overwrite its archived evidence.

The PDF and generated figure/table inputs are intentionally tracked for immediate
reading and review. Compiler logs/intermediates are ignored. `--check` is read-only
and checks generated text, **not PDF byte identity or PDF freshness**. Rebuild and
inspect the PDF after changing prose, equations or figures. PDF bytes can depend on
compiler/bundle/font versions; generated numeric inputs are deterministic.

`tests/test_formal.py` is part of `make gate`: it checks source identity, re-evaluates
archived numerical verdicts, checks dimensional examples and plotting units, and
exercises stale/tampered-data and compiler failure paths.
`core/tests/formal_examples.rs` checks the worked paste return, isolated return-energy
gap, liquid EOS, first-order volume update and inverted-volume rejection against the
real Rust kernel. CI does not require a TeX installation; PDF compilation and visual
inspection are a separate documentation gate.

Evidence:

- [`data/refinement-report.json`](data/refinement-report.json) is a byte-for-byte copy
  of `outputs/refinement-energy-ledgers/study_report.json`, created
  `2026-09-19T09:10:10+00:00` from a clean `9a66a55` checkout.
- SHA-256: `aa6030d7ea1af0919343d3c4732855bce3eecc3c69310a7babc116dc0015bce8`.
- It retains request/study hashes, native/Python hashes, dependency/hardware identity,
  all six case metrics and unresolved verdicts. Its absolute original `run_dir` paths
  are provenance, not portable links. Full original segment artifacts are not bundled
  with the manuscript; rerun the study to produce local artifacts.
- A fresh six-case run during manuscript preparation reproduced every archived native
  observable and refinement verdict exactly; only runtime/provenance metadata differed.
- Inputs at the pinned source are `examples/research_slump.yaml` and
  `examples/verification_slump.yaml`. Source meanings and exact input values are
  pinned by the report and the manuscript; future HEAD changes can alter a rerun.

Reproduce the experiment at the scientific baseline in a separate checkout/environment:

```sh
uv run litter-physics verify examples/verification_slump.yaml \
  --out outputs/reproduce-formal-slump
```

Expected baseline exit status: **1**, complete but unresolved. Existing output
directories are refused. A new report must never silently replace the archive merely
to make a plot look better; update the manuscript baseline and explain any rebaseline.

Known model distinctions surfaced by this derivation:

- Household swelling scales radius and inertia but not sphere-centre offsets;
  occupancy uses capped dry reference volume, not resolved wet porosity.
- The fitted exponential breakup fraction is not the household runtime's linear
  accumulated-damage/whole-pellet threshold law. A fitted rate alone does not close
  this model discrepancy.
- Research spatial refinement co-refines grid spacing, particle quadrature and
  jitter amplitude; reported height/spread use initial-spacing surface corrections.
- Household and research rigid bodies use different body-axis/state conventions.
- The energy residual omits physical/numerical terms and is not physical closure.
