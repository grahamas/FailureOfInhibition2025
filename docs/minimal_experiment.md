# Minimal synthetic experiment

This is a workflow validation using the matched example in
[`experiments/minimal.toml`](../experiments/minimal.toml), not a scientific
parameter selection or biological regime comparison. The numerical contract
and artifact schemas are in [`model.md`](model.md).

## Reproduction

From the repository root, using Julia 1.10:

```sh
julia --project=. scripts/run_minimal_experiment.jl \
    --config experiments/minimal.toml --output output/minimal
julia --project=. scripts/run_minimal_experiment.jl \
    --config experiments/minimal.toml --output output/minimal-repeat
julia --project=output/minimal/source \
    output/minimal/source/scripts/run_minimal_experiment.jl \
    --config output/minimal/config.toml --output output/minimal-replay
```

Each output directory must be absent or empty. Generated artifacts live under
the ignored `output/` directory; the configuration and runner are tracked.
The run records its revision and local modifications and archives the actual
working source, so an uncommitted run does not rely on the revision alone.

The reported run used Julia 1.10.12, one Julia thread, and eight BLAS threads
with `LBTConfig([ILP64] libopenblas64_.so)`. Its configuration SHA-256 is
`9691cfbf6dc74b25bbb81ff2cf7281e31563c6435ddbc42ce612dbe553af7475`.
`metadata.toml` records individual source hashes and environment details;
`checksums.toml` records all 55 artifacts other than itself.

## Observations

All 12 trajectory cases completed successfully. Each of the 12 equilibrium
contexts produced one admissible, locally attracting representative from 25
seeds; all 300 nonlinear attempts returned solver success. These are roots
found, not evidence that other roots or attractors are absent. Every context
retains `CompletenessNotCertified`.

At zero input, the discovered equilibria were:

| Condition | E | I | Eigenvalues (ms^-1) |
| --- | ---: | ---: | --- |
| Control | 0.3636444543 | 0.3208191962 | -1.023182738, -0.624818329 |
| Failure of inhibition | 0.3779018672 | 0.2252518327 | -0.870661569, -0.643480176 |

The table below reports the largest dimensionless balance-residual norm over
both terminal windows. Classification also requires all other criteria in
the diagnostic contract; this residual alone is insufficient.

| Variant | Condition | Protocol | Maximum balance residual | Sampled-window classification |
| --- | --- | --- | ---: | --- |
| Default, 20 ms | Control | Baseline | 6.49265e-4 | Unresolved |
| Default, 20 ms | Failure of inhibition | Baseline | 6.41203e-4 | Unresolved |
| Default, 20 ms | Control | Pulsed | 3.35254e-3 | Unresolved |
| Default, 20 ms | Failure of inhibition | Pulsed | 2.98438e-3 | Unresolved |
| Refined, 20 ms | Control | Baseline | 6.49265e-4 | Unresolved |
| Refined, 20 ms | Failure of inhibition | Baseline | 6.41203e-4 | Unresolved |
| Refined, 20 ms | Control | Pulsed | 3.35254e-3 | Unresolved |
| Refined, 20 ms | Failure of inhibition | Pulsed | 2.98438e-3 | Unresolved |
| Extended, 40 ms | Control | Baseline | 2.42204e-9 | Equilibrium-compatible |
| Extended, 40 ms | Failure of inhibition | Baseline | 1.64578e-9 | Equilibrium-compatible |
| Extended, 40 ms | Control | Pulsed | 1.25850e-8 | Unresolved |
| Extended, 40 ms | Failure of inhibition | Pulsed | 7.83773e-9 | Equilibrium-compatible |

Both 20-ms variants fail the coordinate-range, balance-residual, and
equilibrium-distance criteria over windows `[10,15]` and `[15,20]` ms.
The extended pulsed control fails only the `1e-8` balance threshold over
windows `[30,35]` and `[35,40]` ms. That outcome remains unresolved; the
threshold was not adjusted to obtain a label. The three compatible cases
describe only the saved windows and do not establish asymptotic attraction
or permanent recovery. No periodic orbit or biological label is inferred.

## Validation

The complete package suite passed 887 assertions. The repeated experiment
matched all artifact hashes, and the archived-source replay matched all 41
configuration and numerical artifacts byte for byte. Every listed checksum
was verified in all three outputs. These repeatability checks apply to the
recorded environment; cross-platform bitwise identity is not asserted.

The refinement changes ODE tolerances and sampling together. Its unchanged
20-ms labels do not isolate those two effects, prove search completeness, or
replace the later numerical and model-sensitivity studies in the roadmap.
