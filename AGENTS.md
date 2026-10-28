# Repository guidance

## Scientific scope

- Keep this package a deterministic, CPU-only two-population point model unless the task explicitly changes its scope.
- Preserve the fixed two-state indexing contract `[E, I]`, `source_to_target` coupling names, and the stable CSV column order `time,E,I`.
- Document current model assumptions and numerical conventions in `docs/model.md`. When behavior changes, update the implementation, tests, and documentation together.

## Julia

- Target Julia 1.10; use `snake_case` for functions and `CamelCase` for types.
- Prefer small composable functions, explicit data flow, and immutable typed parameter structs.
- Treat `point_rhs!` and `point_jacobian!` as performance-sensitive numerical paths.
- Document public APIs and export them deliberately from `src/FailureOfInhibition2025.jl`.

## Dependencies and checks

- Keep `Manifest.toml` and `plotting/Manifest.toml` tracked for reproducibility.
- Run the root and plotting checks documented in `README.md` for final verification.
- When including a focused root test file, include `test/test_support.jl` if its helpers are needed.
