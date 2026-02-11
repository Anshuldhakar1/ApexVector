## agents.md

### Objective
Validate whether **shared boundaries + Gaussian smoothing** can produce gap-free, flat-color SVGs from `./img0.jpg` in the `apexvector` repository, without rewriting the main pipeline.

### Operating Mode
You are running a validation spike. Prefer measurement and falsification over feature building.

### Musts
- Use git frequently; commit after every phase and every meaningful test result.
- Keep changes isolated under `validation_spike/` unless a tiny hook is unavoidable.
- Record every command executed and every numeric result in `validation_spike/RESULTS.md`.
- Keep outputs deterministic where possible (fixed RNG seeds for quantization).

### Must Nots
- Do not refactor the production pipeline as part of the spike.
- Do not “fix” issues by adding a background rectangle; background must remain transparent unless explicitly required by test.
- Do not introduce gradients; all fills must be solid colors.

### Pass/Fail Criteria
A phase is PASS only if the documented numeric criteria are met (gap pixels, coverage %, dropout counts). Visual-only judgment is not sufficient.

### Debugging Guidance
If a test fails:
1. Identify which invariant failed (gap, dropout, coverage, topology).
2. Add a small diagnostic artifact (e.g., missing-pixel overlay) under `validation_spike/artifacts/`.
3. Re-run only the relevant phase first.
4. Only then re-run the full chain.

### Git Practices (Strict)
- Work on a fresh branch named `spike/validate-shared-boundaries-<name>-<yyyymmdd>`.
- No force pushes.
- Commit messages must include the phase and result, e.g.:
  - `spike: phase4 coverage 92% FAIL`
  - `spike: phase5 gaps 0.03% PASS`

### Inputs
- `./img0.jpg` at repo root (provided by user)

### Outputs
- `validation_spike/RESULTS.md`
- `validation_spike/artifacts/*`
- Phase scripts under `validation_spike/tests/`