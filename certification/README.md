# Fixed-parameter interval certificates

`certify_attractors.jl` uses [IntervalArithmetic.jl](https://juliaintervals.github.io/IntervalArithmetic.jl/stable/)
to cover the physical `[E,I] ∈ [0,1]²` square for one frozen, zero-drive
configuration. It evaluates the documented vector field and its Jacobian
independently of the fast scalar model kernel. Each accepted box is either
root-free by interval range exclusion or contains exactly one root by a
strict Krawczyk inclusion with contraction. The output retains every
excluded, root-containing, and unresolved box. Root classification uses
interval trace and determinant signs.

The count of **all** minimal attracting limit sets is released only if the
root cover is complete, every root is hyperbolic, the vector field points
strictly inward on the physical boundary, and interval subdivision proves
strictly negative divergence throughout the square. The latter is a
sufficient planar Bendixson condition excluding periodic and closed
separatrix orbits. Together with the Poincaré-Bendixson theorem, those gates
leave the attracting equilibria as the only attractors. Failure of any gate
produces `not_certified` and no exact attractor count. In particular, a
certified equilibrium count does not itself count all attractors.

The two configured cases currently have complete interval equilibrium
covers: Figure 3 has seven equilibria, and Figure 4 rising has five. Each
has three locally attracting equilibria. The global divergence gate fails
for both, so **neither exact attractor count is certified**. A cycle or
other attracting invariant set still requires a separate global argument.

Run from the repository root with a new output directory:

```bash
julia --project=certification certification/certify_attractors.jl --case figure3 --output output/figure3_certificate
julia --project=certification certification/certify_attractors.jl --case figure4_rising --output output/figure4_certificate
julia --project=certification certification/test/runtests.jl
```

The separate `certification/Project.toml` and `Manifest.toml` make the
interval dependency explicit without changing the main model environment.
`metadata.toml`, the exact input configuration, a source snapshot, and
SHA-256 checksums accompany each output. `roots.csv` gives enclosing boxes
and local classifications; `unresolved_boxes.csv` must be empty to claim
complete equilibrium coverage. `metadata.toml` also records the exact-count
status and a replay command for the source snapshot.

The constructed model parameters are interpreted as their exact binary
`Float64` values, including the rounded product `tau_e * tau_ratio`. This is
the parameter convention for these certificates, not an assertion about
uncertainty in physical parameters. All interval computations are required
to retain IntervalArithmetic's enclosure guarantee; a lost guarantee is an
error. The current driver deliberately accepts only `NoDrive` cases. A
certificate for a nonzero frozen baseline or a different parameter cell
requires an explicit extension and its own run.
