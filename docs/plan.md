# Tree Clustering Import Repair Plan

**Goal:** Repair package and standalone-tool imports without changing tree or
shrub instance segmentation or raycloudtools behavior.

**Root-facing contract:**
`TreeSegmRay.from_config(...).segment(points, labels, rays=...)` retains its
current arrays, labels, IDs, ordering, and failure behavior.

**Parent plan:** `../../../docs/plan.md`

## Frozen behavior

- Do not split or rewrite `TreeSegmRay`, clustering, connected components,
  floating-cluster handling, trunk filtering, label reduction, ray preparation,
  Docker commands, or temporary-file behavior.
- Preserve configuration keys, thresholds, labels, sentinels, output naming,
  and ray enabling rules.
- Keep every current `__main__` tool independently executable from this project.

## Completed foundation

- [x] Establish the Python 3.12 uv project and dependency groups.
- [x] Remove the retired LLM/PDF workflow and its exclusive dependencies.
- [x] Keep visualization outside the normal clustering import path.

## Task 1: Protect imports and invocations

- [ ] Characterize `TreeSegmRay` import, ray-disabled processing, and existing
  error propagation before changing imports.
- [x] Inventory every `__main__` guard without removing any executable.
- [ ] Add paired direct/module `--help` tests for operational and diagnostic
  tools.
- [ ] Verify parent-root import resolves this project's `TreeSegmRay`.

## Task 2: Repair package imports

- [ ] Replace ambiguous top-level `utils` imports with package-relative imports
  in reusable modules.
- [ ] Remove broad `ImportError` fallback selection so errors raised inside a
  valid dependency propagate unchanged.
- [ ] Prevent `utils.__init__` from loading Laspy persistence or plotting when
  only ray construction is imported.
- [ ] Keep direct-script bootstrapping limited to thin entry-point code.

## Task 3: Preserve standalone tools

- [ ] Replace developer-specific hardcoded inputs with explicit CLI arguments.
- [ ] Keep LAZ inspection/saving and visualization tools available under
  `dev`, without adding their dependencies to the root-facing import path.
- [ ] Verify all declared commands directly and with `python -m` from this
  project root.

## Task 4: Verify

- [ ] Run import-boundary, invocation, and complete deterministic tests.
- [ ] Run ray-disabled fixtures without Docker.
- [ ] Verify parent-root processing with existing tree/shrub fixtures.
- [ ] Run Docker/raycloudtools and visualization smoke tests separately when
  their environments are available.

## Completion gate

`TreeSegmRay` imports only its real root-facing dependencies, every retained
tool works independently, internal import failures are not hidden, and all
clustering outputs remain unchanged.
