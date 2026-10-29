# Fixed-point completeness certificates

`certify_equilibria` exhaustively accounts for equilibria of one fixed model
with constant tonic inputs in the closed domain `[E,I] ∈ [0,1/2]²`. A successful
certificate proves the equilibrium count and encloses every root. It says
nothing about local stability, periodic attractors, basins, or biological roles.
`find_equilibria` remains ordinary multistart discovery and always reports
`CompletenessNotCertified`.

```julia
using FailureOfInhibition2025
certificate = certify_equilibria(model; options=CertificationOptions())
certificate.status             # :certified or :unresolved
certificate.equilibrium_count  # integer only for complete coverage
certificate.roots              # enclosures and uniqueness witnesses
certificate.nodes              # coverage and unresolved reasons
write_equilibrium_certificate("certificate.toml", certificate)
loaded = read_equilibrium_certificate("certificate.toml")
verify_equilibrium_certificate(loaded) # recomputes proof predicates
```

## Mathematical contract

The inputs are the exact stored numeric model values, including rounded values
created during model construction. They are not uncertain parameters or exact
versions of their original decimal spelling. Both supported inhibitory responses
(`LogisticResponse` and `FailureOfInhibitionResponse`) are supported. Drives must
be `NoDrive` or a `PiecewiseConstantDrive` with no pulses. Construct a constant
model explicitly rather than asking certification to select a time snapshot.

The certified equations are the analytic dimensionless balances
`g_X = -X + (1-X) F_X(u_X)`. Positive timescales do not change their zeros.
The interval evaluator implements those formulas independently of the ordinary
floating-point kernel: rounded numerical saturation at zero or one does not
redefine the analytic response. For finite supported parameters, `0 < F_X < 1`,
so `g_X(0)>0` and `g_X(1/2)<0`. Consequently genuine equilibria are interior to
the prescribed square, even when an ordinary floating-point root prints as
zero or one half. The certifier never searches the larger physical square.

## Proof and resource limits

Coverage uses interval range exclusion, Krawczyk exclusion/contraction and
existence/uniqueness tests, followed by subdivision where needed. A coverage
record retains every split, contraction, root assignment and unresolved leaf.
Root assignment requires a proof of containment in a uniqueness witness;
nearby numerical coordinates are not a deduplication rule. Distinct root
enclosures must be disjoint. Unresolved identity prevents a complete count.

All accepted evaluations require guaranteed, valid intervals and appropriate
decorations. Contraction bounds are computed outward. Rigorous rounding must
remain enabled throughout a call; do not modify IntervalArithmetic's global
configuration concurrently with certification. The implementation is serial.

The default budgets are 100,000 processed boxes, depth 64, 1,000,000 evaluations,
200,000 records, 60 seconds, and 1024-bit maximum precision. Time checks occur
between operations, so this is a cooperative deadline, not a hard process
termination guarantee. Compilation and individual arithmetic calls can exceed
it. Limits preserve unresolved coverage and never produce a successful count.
Singular or multiple roots can remain unresolved even with substantial budgets.

Float64 interval calculations handle ordinary regions; higher precision is
reserved for difficult regions. Exact input values are preserved when precision
changes. Smaller tolerances or higher precision do not themselves prove a root.

## Artifacts and verification

The versioned TOML representation records exact binary parameters and interval
endpoints, options, domain, coverage, witnesses, numerical environment and source
provenance. The equilibrium-problem hash omits positive timescales and drive
interpretation; the full model hash preserves them. Numeric identity hashes are
type sensitive. Restart Julia after source or environment edits: the certifier
rejects disk files that differ from the loaded context. Reading a saved status
does not validate it. Verification recomputes
proof predicates, coverage accounting and root distinctness, and checks
provenance. Retain the matching source and tracked package environment with a
certificate for later replay.

Verification shares the interval arithmetic library and mathematical evaluator
with certification; it is a separate proof replay, not an independent formal
proof assistant. Source hashes establish identity, not correctness. Tests and
independent review address the implementation's correctness assumptions.

## Cost and development checks

Cost depends on subdivision and conditioning, not just the number of roots.
Cheap range exclusion precedes Jacobian work; contraction reduces subdivision;
parameter constants and response calculations are reused. Bounded fixture
benchmarks report warmed runtime, allocation, box and evaluation counts, and
precision usage separately from compilation. No parameter study or dataset
batch is part of this machinery's validation.

Run the root `Pkg.test()` and plotting suite documented in the repository README.
The certification tests include known root structures, saturation, shared faces,
degeneracy, exhausted budgets, invalid interval evaluations, and altered proof
artifacts. A floating-point multistart count is never the sole test oracle.

A Julia 1.10.12 engineering run measured the following small, fixed fixtures
(best of three warmed calls, including provenance hashing):

| Fixture | Roots | Boxes | Evaluations | Precision | Warm time | Allocations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncoupled FoI | 1 | 1 | 4 | 53 bits | 1.4 ms | 203 kB |
| Excitatory feedback | 3 | 31 | 150 | 53 bits | 3.4 ms | 1.03 MB |
| Saturated logistic | 1 | 1 | 8 | 256 bits | 2.1 ms | 389 kB |

First calls took 0.9–3.7 seconds including compilation. Disabling contraction
in the three-root fixture raised boxes from 31 to 47 and evaluations from 150
to 201; disabling range exclusion raised evaluations to 162. Millisecond timing
variation did not establish a range-exclusion speedup. These fixtures establish
bounded engineering cost, not expected runtime for a dataset or nearly singular
parameters. Reproduce with `julia --project=. scripts/benchmark_certification.jl`.

The public verifier returns `true` only for a complete valid certificate.
Unresolved artifacts remain serializable for inspection. Timing, depth, and
search counters are diagnostic history; proof replay validates the mathematical
coverage, not that a prior execution actually obeyed its recorded budgets.
