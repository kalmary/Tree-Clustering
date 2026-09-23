# TreeClustering Rebuild Plan

**Goal:** Incrementally split tree/shrub instance segmentation and the raycloudtools integration into testable units while preserving `TreeSegmRay`, label semantics, and direct script use.

**Root-facing contract:** `TreeSegmRay.from_config(...).segment(points, labels, rays=...)` returns point-aligned instance IDs and initial species labels.

**Design:** `../../../docs/rebuild.md`

**Branch requirement:** Perform all rebuild work on `development`. Verify the active branch first and request explicit approval before creating or switching it.

## Task 1: Establish the uv project

- [x] Create `.python-version`, `pyproject.toml`, and `uv.lock` for Python 3.12.
- [x] Replace the requirements freeze with direct dependencies derived from imports.
- [x] Define headless `basic` and `test` including `basic`, `pytest`, `matplotlib`, and `pyvista`.
- [x] Keep visualization packages out of the normal clustering import path.
- [x] Omit PyTorch because no retained TreeClustering code imports it.
- [x] Verify clean basic/test syncs, pytest collection, and current public imports.

## Task 2: Protect array and label contracts

- [ ] Retain and expand tests for empty/no-tree input, tree/shrub merging, ID offsets, `-1` sentinels, dtypes, overflow, and shape mismatch.
- [ ] Characterize accepted points, labels, and ray layouts and their validation errors.
- [ ] Test config parsing and all root-used defaults.
- [ ] Pin deterministic connected-component, floating-cluster, trunk-filter, and label-reduction behavior.
- [ ] Test imports and direct/module entry execution from supported working directories.

## Task 3: Isolate raycloudtools

- [ ] Define a small backend boundary for executable discovery, Docker lifecycle, PLY I/O, command execution, and parsed results.
- [ ] Use a deterministic fake backend for default tests.
- [ ] Preserve current command arguments, ray enabling rules, temporary data, and failure propagation.
- [ ] Guarantee cleanup of temporary files and owned containers on success, failure, and interruption.
- [ ] Keep Docker startup privileged operations out of import and construction paths.

## Task 4: Split array_processing_RE.py incrementally

- [ ] Keep `TreeSegmRay` and the old module as the compatibility facade.
- [ ] Extract config/state validation first.
- [ ] Extract ray preparation and external backend adaptation.
- [ ] Extract pure clustering/post-processing groups only after their existing tests pass independently.
- [ ] Keep each extraction behavior-neutral; do not retune thresholds or replace algorithms.

## Task 5: Normalize utilities and scripts

- [ ] Replace generic `utils` imports with package-relative imports.
- [ ] Keep plotting utilities optional to the basic path.
- [ ] Keep retained inspection, saving, and visualization scripts directly executable and module-executable.
- [ ] Resolve configs and outputs without accidental current-working-directory dependence.
- [x] Remove the retired LLM/GPT classification and PDF-report workflows and their exclusive dependencies.

## Task 6: Verify

- [ ] Run the existing algorithmic unit suite after each extraction.
- [ ] Run invocation/import tests in both standalone and BRIK contexts.
- [ ] Run a ray-disabled in-process fixture with no Docker dependency.
- [ ] Run an explicit Docker/raycloudtools smoke test with representative ray data.
- [ ] Run root integration for tree-only, shrub-only, mixed, absent-ray, and real-ray inputs.

## Completion Gate

The project resolves with uv; default tests do not require Docker or GUI access; `TreeSegmRay` remains compatible; temporary/external-resource ownership is explicit; invocation modes and the Docker smoke test pass.
