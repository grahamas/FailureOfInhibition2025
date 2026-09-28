# Low-ratio equilibrium coexistence: finite results

The completed local run in `output/low_ratio_coexistence` tested whether reducing
`r = tau_I/tau_E` makes the intermediate failure-of-inhibition equilibrium
locally attracting alongside the three attracting equilibria found at the
manuscript examples. It used the two already declared `e_to_i` by `theta_off`
planes, `r` from 0.2 to 4.4, `tau_E = 7.8 ms`, zero drives, and the approved
equal-slope response. These are exploratory model parameters, not physiological
calibration. The artifact is ignored by version control; its `config.toml`,
`tetrastability.toml`, `source/`, `metadata.toml`, and `checksums.toml` preserve
the exact run inputs and output integrity.

## Stability calculation

At a fixed equilibrium, the balance Jacobian `Dg = [a b; c d]` does not change
with positive time constants. The ODE Jacobian has trace
`(a + d/r)/tau_E` and determinant `(a*d-b*c)/(tau_E^2*r)`. A resolved
equilibrium is locally attracting when the determinant is positive and the
trace is negative. At both manuscript examples the intermediate root has
`a > 0`, `d < 0`, and positive determinant. Its linear attracting side is
`r < -d/a`; at equality, linear stability is unresolved. This calculation
does not establish a Hopf bifurcation or a periodic orbit.

## Finite-grid findings

All 1,650 declared cells completed. The 11-by-11 screen found four locally
attracting roots at `r = 0.2` in 164 cells and in none at `r = 4.4`.
Each of those 164 cells had a witness inside its screened attracting interval
with four distinct roots matched across independent 11-by-11, 21-by-21, and
41-by-41 searches. Every witness met the recorded residual, root-separation,
and negative spectral-margin checks; no search execution or unresolved-nearby
failure was recorded. The other 1,486 cells had no four-root candidate in the
finite screen. These are sampled-cell counts, not prevalence estimates or
continuous parameter-region boundaries.

| Plane | Sampled cells | Witness-confirmed cells | `e_to_i` extent among those cells | `theta_off` extent among those cells | Screened upper ratio bound among those cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| Figure 3 couplings `(17, 9, *, 4)` | 825 | 48 | 19–27 | 8–12 | 0.303–0.398 |
| Figure 4 couplings `(19, 13, *, 6)` | 825 | 116 | 18–28 | 6.75–12 | 0.393–0.443 |

The parameter extents in this table are minima and maxima of sampled positive
cells. They do not imply that every intervening cell qualifies. The upper
ratio bounds come from the screen Jacobians; the three-grid qualification is
for a recorded witness inside each interval, not for every ratio up to its
boundary. Full cell statuses are in `cells.csv`, witness results in
`intervals.csv`, and every discovered root and nonlinear attempt in
`roots.csv`, `attempts.csv`, and `contexts/`.

At `e_to_i = 19` and `theta_off = 8`, the intermediate roots give:

| Plane | Intermediate `(E, I)` | Linear trace-zero ratio | Central-root eigenvalues at `r = 0.2` (`ms^-1`) | Central-root eigenvalues at `r = 4.4` (`ms^-1`) |
| --- | --- | ---: | --- | --- |
| Figure 3 | `(0.311014590, 0.425138163)` | 0.302967422 | `-0.430723 ± 1.486884i` | `0.073367`, `1.484659` |
| Figure 4 | `(0.329934172, 0.367295168)` | 0.430209231 | `-1.063831 ± 2.147919i` | `0.174941`, `1.492786` |

At ratios 0.002 below and above the calculated threshold, independently
recomputed central-root spectral abscissae were respectively `-0.00556` and
`+0.00549 ms^-1` for Figure 3, and `-0.00432` and `+0.00428 ms^-1` for Figure 4.
The other three attracting roots at each example remain locally attracting
at `r = 0.2`. This supports **four discovered locally attracting equilibria**
at each lower-ratio example, without identifying any as functional or
pathological activity.

## Verification and limits

The run used a clean frozen source revision and completed with 164 confirmed
witnesses and zero failed cells. Independent recomputation from the response
equations matched at least four roots across all three grids in every positive
cell; the maximum balance residual was `1.64e-14`, and the maximum difference
in recomputed balance-Jacobian entries or determinant was `1.28e-13`.
Central finite differences at the two manuscript examples agreed with the
archived trace-zero ratios within `3.4e-10`. All 2,008 files listed in the
artifact manifest passed SHA-256 verification. An archived-source smoke replay
reproduced `anchors.csv` exactly and reproduced both example interval rows
apart from full-grid cell identifiers.

Every search retains `CompletenessNotCertified`. A finite screen without four
roots does not prove their absence; a confirmed witness does not establish an
exact attractor count, basin size, a continuous coexistence region, periodic
dynamics, a biological state, or a manuscript claim. The claim catalogue
remains draft, and manuscript revision remains author-controlled.

To reproduce the local study from a clean checkout, run
`julia --project=. scripts/run_low_ratio_coexistence.jl --output output/low_ratio_coexistence_new`.
The run writes only to a new empty output directory. Its archived source can
also replay the same command using the path recorded in `metadata.toml`.
