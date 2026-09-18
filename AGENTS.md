# Repository guidance

## Scientific scope

- Keep this package a deterministic, CPU-only two-population point model unless the task explicitly changes its scope.
- Preserve the state order `[E, I]`, `source_to_target` coupling names, and the stable CSV column order `time,E,I`.
- Treat the model equation and response candidates as provisional. Do not silently choose a candidate, normalize or clamp a response, or settle an open scientific question from `NEXT_STEPS.md`.

## Julia

- Target Julia 1.10 and follow Julia naming conventions: `snake_case` functions, `CamelCase` types, and `!` for functions that mutate an argument.
- Prefer multiple dispatch, small composable functions, explicit data flow, and immutable typed parameter structs. Avoid non-`const` globals and overly restrictive concrete argument types.
- Keep hot numerical paths type-stable and allocation-conscious, especially `point_rhs!` and `point_jacobian!`; optimize further only with benchmark evidence.
- Preserve generic numeric behavior where practical: use `zero`, `one`, promotion, and `isapprox` rather than assuming `Float64` or exact floating-point equality.
- Add docstrings for public APIs, export them deliberately from `src/FailureOfInhibition2025.jl`, and add focused `Test` cases for normal, boundary, and invalid inputs.

## Dependencies and checks

- Never manually edit `Project.toml` or `Manifest.toml` when an equivalent `Pkg` command or API exists. Use `Pkg.add`, `Pkg.rm`, `Pkg.update`, `Pkg.compat`, `Pkg.resolve`, and `Pkg.instantiate` as appropriate, allowing Pkg to update both files together.
- Do not hand-edit `Manifest.toml`. Keep it synchronized and tracked for reproducibility.
- Run the suite with `julia --project=. -e 'using Pkg; Pkg.test()'`.
