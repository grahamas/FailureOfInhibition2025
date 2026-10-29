# Basin overlays for nullcline figures

Acceptance: default shared-classification basin colours on [0,0.5]^2; optional
validated saddle stable manifolds; reproducible artifacts; benchmarked defaults;
root and plotting checks and independent review with findings resolved.

| ID | Work | Depends on | Evidence/status |
| --- | --- | --- | --- |
| numerics | Dense persistent Attractors mapper and stable-manifold tracing | — | Complete; cache consensus, known-sink seeding, rollback reviewed |
| figures | Cache, packet integrity, overlays and legacy compatibility | numerics | Complete; standalone evidence generated and checked |
| benchmark | Reuse/scaling measurements and separatrix default | numerics | 534 comparable checks agreed; 244 neutral/unassessed; default traces enabled |
| verification | Focused, root, plotting, artifact and visual checks | figures, benchmark | Passed: root 3541 assertions, plotting 19, independent artifacts 31; final variants visually inspected |
| review | Independent numerical and final diff review | verification | Independent source and benchmark review complete; no unresolved substantive finding |
| fixes | Resolve substantive findings and renew checks | review | Fixed provisional cells, false transient discovery, boundary reuse, Float32 refinement, degenerate segment verification |

Decisions: sample initial conditions only in [0,0.5]^2, allow full physical-domain
trajectories; use Attractors.jl with dense shared state; leave unmatched attractors
unresolved; enable separatrices by default only if cheaper than grid generation
on both mechanism conditions in three warmed runs. No scientific roles changed.

Benchmark evidence: `output/nullcline_basins_benchmark_20261029/`; median 100²
grid costs 0.613 s (monotone) and 0.600 s (failure), manifold costs 0.0061 s and
0.0109 s. Independent review confirmed all 534 comparable points agreed; 244
neutral points were explicitly unassessed. One manifold branch per model failed
convergence and is retained as incomplete, with no displayed line.

Delivery is isolated in `workspaces/nullcline-basins`; only this session's
hash-verified source edits were removed from `main`. Its unrelated documentation
edits were preserved. Root tests ran on byte-identical source before transfer.
Logs are retained in `output/nullcline_basins_checks_20261029/`.

Verification commands: `Pkg.instantiate(); Pkg.test(;allow_reresolve=false)` in
the root environment; `plotting/test/runtests.jl` in the plotting environment;
`verify_mechanism_evidence` on the standalone artifact. Julia 1.10.12 with a
writable temporary depot/cache and offline mode was used for checks.

Post-transfer verification passed in the isolated workspace: all 598 focused
assertions (basin reuse and boundary handling, saddle manifolds, figure
contracts, cache identity/tamper handling, legacy enrichment, and degenerate
curve distance) plus repeated independent artifact verification. The final
source diff passed whitespace review. Independent review found no unresolved
substantive defect after the recorded fixes.
