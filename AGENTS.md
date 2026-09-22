# Repository guidance

## Scientific scope

- Keep this package a deterministic, CPU-only two-population point model unless the task explicitly changes its scope.
- Preserve the state order `[E, I]`, `source_to_target` coupling names, and the stable CSV column order `time,E,I`.
- Document current model assumptions and numerical conventions in `docs/model.md`. When behavior changes, update the implementation, tests, and documentation together.

## Julia

- Target Julia 1.10 and follow Julia naming conventions: `snake_case` functions, `CamelCase` types, and `!` for functions that mutate an argument.
- Prefer multiple dispatch, small composable functions, explicit data flow, and immutable typed parameter structs. Avoid non-`const` globals and overly restrictive concrete argument types.
- Program to Julia interfaces: accept `AbstractArray` or weaker capabilities when sufficient, and preserve input element types, axes, and storage conventions with generic constructors such as `similar` where appropriate.
- Use `eachindex` for elementwise traversal, `axes` when dimension-specific indices matter, and `firstindex`/`lastindex` for endpoints. Do not assume one-based or linear indexing unless it is an explicit, documented contract; keep fixed `[E, I]` indexing local to this model's stated two-state interface.
- Check compatible sizes or axes at API boundaries when multiple arrays are traversed together. Use `@views` to avoid unintended slice copies, and use `@inbounds` only in small, performance-critical loops whose bounds have been established and tested.
- Keep hot numerical paths type-stable and allocation-conscious, especially `point_rhs!` and `point_jacobian!`; optimize further only with benchmark evidence.
- Preserve generic numeric behavior where practical: use `zero`, `one`, promotion, and `isapprox` rather than assuming `Float64` or exact floating-point equality.
- Prefer fused broadcasting and preallocated mutating kernels when they make data flow clear; avoid abstractly typed fields and containers in performance-sensitive code.
- Add docstrings for public APIs, export them deliberately from `src/FailureOfInhibition2025.jl`, and add focused `Test` cases for normal, boundary, and invalid inputs.

## Dependencies and checks

- Never manually edit `Project.toml` or `Manifest.toml` when an equivalent `Pkg` command or API exists. Use `Pkg.add`, `Pkg.rm`, `Pkg.update`, `Pkg.compat`, `Pkg.resolve`, and `Pkg.instantiate` as appropriate, allowing Pkg to update both files together.
- Do not hand-edit `Manifest.toml`. Keep it synchronized and tracked for reproducibility.
- Run the suite with `julia --project=. -e 'using Pkg; Pkg.test()'`.
