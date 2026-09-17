# Engineering Guidelines

## Priorities

Produce the smallest correct change that satisfies the request. Optimize for correctness, clarity, maintainability, and compatibility with existing behavior. Every changed line must have a reason connected to the task.

When instructions conflict, follow this order:

1. The user's explicit request.
2. Repository-specific conventions and documented requirements.
3. The guidelines in this file.

## Understand Before Editing

- Read the relevant code, tests, configuration, and documentation before proposing a change.
- Trace the current behavior and identify the actual cause of a problem. Do not patch symptoms without understanding them.
- State assumptions when the repository does not provide enough evidence. Ask before making a choice that would materially change behavior or scope.
- For multi-step work, use a short plan in which each step has a concrete verification method.

## Design Data First

- Start with the data model, state transitions, inputs, outputs, ownership, and invariants.
- Prefer data structures that make valid states and the common path easy to express.
- Avoid accumulating conditionals to compensate for a poor representation. Improve the representation when doing so is within scope.
- Keep data flow explicit. Avoid hidden mutation, unnecessary shared state, and unclear ownership.
- Use classes, inheritance, patterns, or framework layers only when they provide a concrete benefit over simpler structures and functions.

## Prefer Simple, Direct Code

- Write the simplest implementation that is clearly correct.
- Do not add speculative abstractions, extension points, configuration, or error handling for unsupported hypothetical cases.
- Avoid clever code when ordinary control flow communicates the behavior more clearly.
- Reuse an existing abstraction when it fits. Do not force a task into an abstraction that makes the common case harder to understand.
- Choose descriptive, general names based on purpose. Do not rename unrelated identifiers or over-elaborate names.
- Add comments only when they explain intent, constraints, or a non-obvious decision. Do not comment that merely restate the code. In general safe commenting code is best. Proper naming of its parts should usually suffice, unless there is a fancy solution, hard to understand without comments.
- Do not add references to AI tools, generated code, prompts, or the development conversation.

## Keep Changes Surgical

- Change only files and lines required for the requested outcome.
- Match the repository's existing style, architecture, formatting, and dependency choices.
- Do not perform drive-by refactors, formatting sweeps, dependency upgrades, or unrelated cleanup.
- Remove imports, variables, helpers, and files made unused by the current change. Leave pre-existing dead code alone unless the user asks to address it.
- If an unrelated defect is discovered, report it separately instead of expanding the patch.
- Do not modify generated or vendored files unless the project workflow explicitly requires it.

## Preserve Existing Contracts

- Treat public APIs, file formats, configuration, command output, persisted data, and established workflows as contracts.
- Preserve backward compatibility unless the user explicitly requests a breaking change and accepts its consequences.
- Before changing a script or one of its functions, read the corresponding file documentation in `docs/documentation/` and treat its documented behavior as a requirement.
- When changing a script or function, preserve its input data and output exactly, including accepted formats, schemas, types, names, ordering, and externally observable behavior, unless the user explicitly requests a contract change.
- Consider boundary cases, malformed input, partial failure, retries, concurrency, and resource cleanup when they are relevant to the actual system.
- Never hide failures with broad exception handling, silent fallbacks, arbitrary retries, or default values that obscure invalid state.
- Do not weaken validation, authorization, security controls, or data integrity to make a test pass.

## Performance Must Be Evidence-Based

- Prioritize correctness and clarity unless performance is a stated requirement or a measured problem.
- Do not claim an optimization without a representative benchmark, profile, or other reproducible evidence.
- Consider algorithmic complexity, allocations, I/O, memory locality, and contention where they materially affect the workload.
- Keep hot paths direct, but do not trade readability for hypothetical gains.

## Verification

- Define success in observable, testable terms before declaring the work complete.
- Run the narrowest relevant checks first, then broader checks when the risk justifies them.
- Add or update tests for changed behavior and important regressions when a test suite exists.
- Test externally visible behavior rather than implementation details where practical.
- Use the repository's documented commands. For Python projects, prefer the existing `uv` environment, such as `uv run <command>` or `source .venv/bin/activate` followed by the project command.
- Report exactly what was verified. If a check could not be run, explain why and identify the remaining uncertainty.
- Do not describe work as complete when it is known to be broken or unverified.

## Review and Communication

- Review the final change for correctness, scope, regressions, security, and accidental artifacts.
- Support technical conclusions with code references, test output, measurements, or reproducible behavior.
- Be direct and specific about flaws in code or design, but remain professional and never attack a person.
- Distinguish confirmed facts from assumptions and recommendations.
- Explain the outcome and important tradeoffs concisely; avoid narrating routine tool use.

## Version Control

- Never push, pull, switch branches, merge, rebase, reset, stage files, or run any other Git operation without the user's explicit approval.
- Read-only access to `.git` does not grant permission to run Git commands.
- Do not leave tool-specific metadata, temporary files, debug output, or other traces unrelated to the requested result in the repository.

## Testing code

- Unit and integration tests should be written and run for new code.
- Unit tests for given functions should be written in their original files.
- Integration tests should be written in the `tests/` directory, outside of `src/`.
- Always run tests, once new code is written.
- If tests fail, explain why and identify the remaining uncertainty.
- We use pytest for testing. No need for complex testing structure.

## Editing documentation

- As an LLM you are able to edit documentation when requested.
- You should do this withour use of extra python/ any other extra scripts.
- You can use script when file format doesn't support editing directly.