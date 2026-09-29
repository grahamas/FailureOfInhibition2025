# Input-dependent high-E/low-I state and permanent release

The permanent-release study finds the requested numerical behavior. Starting
from `three_sinks_descending`, change only `e_to_e` from 17 to **4**.
The parameters are `(e_to_e,i_to_e,e_to_i,i_to_i)=(4,9,19,4)`,
`theta_off=8`, `tau_E=7.8 ms`, `tau_I=34.32 ms`, `a_E=a_I=5`,
`theta_E=1.5`, and `theta_on=4`. These are exploratory parameters.

| Condition | Observed `(E,I)` | Evidence |
| --- | --- | --- |
| Initial, `B_E=B_I=0` | `(0.000558673915, 2.17350226e-9)` | Attracting equilibrium; complete zero-input equilibrium cover |
| After 5000 ms at `B_E=8, B_I=0` | `(0.5, 0.000558673970)` | Two-window compatibility with the driven high-E/low-I equilibrium |
| 5000 ms after permanent release to zero | `(0.000558673913, 2.17350226e-9)` | Return to the original equilibrium within recorded tolerances |
| Input held at 8 for the same follow-up | High E / low I | Control from the identical actual switching state |

The high-E/low-I role remains provisional. Neither that name nor finite
recovery assigns a biological state or certifies asymptotic behavior.

## Earlier evidence

The adaptive rescue scan restored the original tonic input after a finite
withdrawal and assessed the destination in that restored-input system.
Its named seizure source followed a zero-input reference. Unmatched sinks
were sampled but could not contribute to its seizure-rescue summary.
That result does not exclude a state sustained by input that disappears
when input stays off.

The previous animation correctly used `0 -> 8 -> 0`; its chosen parameters
retained zero-input high-activity equilibria. It illustrates those cases,
not all parameters. Its descending exemplar reached the seizure coordinate
role; the other two reached the high-E/high-I herald role.

## Protocol and results

`scripts/run_input_release_study.jl` reads `experiments/input_release.toml`.
For each of four exemplars it varies `e_to_e=0:0.25:24`, with other parameters
fixed, and independently searches at `B_E=0` and `B_E=8`, always with `B_I=0`.
Each search uses a 21-by-21 sharper-domain grid plus default seeds. Every
root, stability result, and solver attempt is retained in its input context.

Existing high-E/low-I reference coordinates initialize tracking in the
**driven** system, then follow decreasing `e_to_e`. No zero-input counterpart
is required. Lost or ambiguous driven matches remain unavailable. The
two-sink exemplar has no configured seizure reference; its paired searches
are retained without assigning one. Root indices are local to each context.

Direct release from a driven equilibrium screens candidates. A complete
trial then starts at the lowest-E zero-input sink, establishes the driven
source, and compares permanent release with continued input. Both follow-ups
start from the actual end-of-induction state, without snapping to a root.
Success requires induction at the tracked source, recovery to the original
starting sink, and a kept-on control at the tracked source. Other destinations,
integration failures, and unresolved diagnostics remain explicit.

Each phase uses 5000, 10000, and 20000 ms horizons as needed. Two terminal
100-ms windows with at least 21 samples each must satisfy coordinate
tolerance `1e-6` and balance tolerance `1e-8`; local attraction is checked
separately. Integration uses `abstol=reltol=1e-10`, `domain_atol=1e-8`,
and `maxiters=1000000`.

All **388** paired parameter cells completed. **17** descending-family cells
at sampled couplings `0:0.25:4` passed the complete trial. The four-sink and
ascending families had respectively 17 and 20 direct-release screen
positives, but none passed the complete trial. Release from a supplied state
does not establish that the specified input induces it. No grid observation
proves an exact threshold or excludes a narrower unsampled interval.

Selection prefers the largest successful recurrent coupling, with catalogue
order breaking ties. The selected coupling 4 passed independent 21-by-21 and
41-by-41 confirmations with tenfold tighter numerical solver tolerances.
Continuation in `B_E` found a fold candidate near `0.1086`; this is numerical
branch evidence, not a certified bifurcation.

The zero-input interval run resolved every box and certified **exactly one
equilibrium**, locally attracting. This excludes another zero-input
high-E/low-I equilibrium for these parameters. The global divergence gate
remains unresolved, so the count of all attractors stays `not_certified`.
The driven equilibrium search retains `CompletenessNotCertified`.

## Reproduction

Run with Julia 1.10 from the working copy, using new output directories:

```sh
julia --project=. scripts/run_input_release_study.jl --output output/input_release
julia --project=certification certification/certify_attractors.jl --config output/input_release/zero_input_certificate_config.toml --case three_sinks_descending --output output/input_release_certificate
julia --project=plotting scripts/render_input_release.jl output/input_release output/input_release_animation
```

`--smoke` tests the descending family at couplings 6 and 4; it does not
reproduce the full scan. Retained local artifacts are:

- `output/input_release_20260928/`: scan, paired contexts, phase attempts,
  confirmations, selected trajectories, and continuation.
- `output/input_release_certificate_20260928/`: complete zero-input
  equilibrium cover and separately reported global gate.
- `output/input_release_animation_20260928/`: MP4, final frame, renderer
  source, dependency manifests, and input checksum reference.

Archives retain source snapshots and checksums. Study metadata gives the
replay command. The renderer verifies input checksums before reading data.
Animation plots emphasize the first 400 ms of each phase while the clock
advances through the full 5000-ms horizons; equilibrium markers change
with the currently applied input.
