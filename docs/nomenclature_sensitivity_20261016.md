# Activity-based names: sensitivity and dynamical distinctions

## Decisions and result

The author specified that seizure and herald must not require another
equilibrium. Rest means both E and I near zero; seizure means elevated E,
distinct from rest, and near-zero I on the inhibitory failure arm. Herald
means high E with preserved inhibitory activity. Intermediate states should
receive descriptive names rather than be forced into one of those endpoints.

The reanalysis supports separating **activity name, geometry, and dynamics**.
Low-activity attracting states exist on the middle E arm, and their local
stability depends on inhibitory feedback. Conversely, the old first-arm rest
rule admits states with almost maximal I. Neither an arm number nor the
presence of another sink is an adequate activity definition.

No numerical boundary, revised production classifier, biological label,
manuscript text, or manuscript figure is adopted by this study.

## What was evaluated

The verified audit retains 215,657 roots across 77,455 archived contexts.
Sensitivity counts use two separate cohorts of attracting, non-near-singular
roots: 74,674 screen records and 60,559 detailed-geometry records. All other
stability categories remain in the source inventory and the inventory summary.
The two cohorts overlap in scientific contexts; their counts must not be added
as distinct states or interpreted as parameter-space prevalence.

For each cohort, exact cumulative distributions retain every distinct E, I,
and max(E,I) value. The 101 displayed empirical quantiles use observed values
without interpolation. Independent two-boundary surfaces evaluate rest,
seizure, and herald membership. A separate 729-combination grid varies all six
boundaries independently. Overlapping memberships and gaps remain explicit.
The packet records membership support for every included root and candidate
memberships for all 84 intervention destinations under both cohort grids.

The proposed predicates are:

| Candidate description | Activity/geometry predicate |
| --- | --- |
| Rest | E ≤ rest-E boundary and I ≤ rest-I boundary |
| Seizure | E > seizure-E boundary, I ≤ seizure-I boundary, descending inhibitory response |
| Herald | E ≥ herald-E boundary and I ≥ herald-I boundary, on either inhibitory response side |
| High-E, partially failed inhibition | E ≥ herald-E boundary, descending response, seizure-I boundary < I < herald-I boundary |

Stability is separate from these memberships. The sensitivity figures select
attracting nonsingular roots; the predicate itself can describe an unstable
equilibrium without claiming it is an attractor. Multiple roots may share a
description. Circuits without E turning points are included. A peak or
ascending inhibitory response cannot satisfy the seizure or partial-failure
predicate. Numerical derivative underflow does not substitute for response-side
geometry: the audit supplies side from total input relative to the response peak.

The term "preserved inhibition" here is operationalized by I occupancy, as
requested. It does not establish inhibitory efficacy across different coupling
strengths or a physiological firing-rate threshold.

## Rest on the middle arm is present and dynamically distinct

The smallest max(E,I) among attracting middle-arm records is 0.01005085 in
the screen and 0.01083982 in the detailed study. Representative states are:

| Cohort / circuit | E | I | Original local role | J_EE (ms⁻¹) | Slowest linear decay (ms) |
| --- | ---: | ---: | --- | ---: | ---: |
| Screen / figure3_joint_7 | 0.01005085 | 0.00511805 | Unassigned | +0.00441946 | 40.67 |
| Detailed / three_sinks_ascending | 0.01083982 | 0.00097913 | Active | +0.00096657 | 150.03 |

Both are on the ascending inhibitory response. All 5,120 screen and 7,059
detailed attracting middle-arm records have positive J_EE. Reconstructing the
Jacobians from the recorded model parameters reproduces every retained spectral
abscissa exactly in the same floating-point computation, across all 48,381
lower/middle-arm records. This is a reconstruction check, not an independent
solver or certification of equilibrium existence.

At equilibrium, the implemented logistic E response gives

```math
J_{EE} = \frac{a_E s_E E(1-2E)-1}{\tau_E(1-E)},
```

where a_E is recurrent E coupling and s_E the response slope. The middle E arm
lies between the zeros of its numerator. Thus an E perturbation grows locally
when I is held fixed, although coupled inhibitory feedback can stabilize the
full equilibrium. This is a meaningful local distinction from the lower arm.
It does not establish different basin sizes, switching thresholds, finite
perturbation recovery, or a uniform ordering of relaxation times. The plotted
decay-time ranges overlap.

Readable occupancy probes give the following results. Requested boundaries
are snapped independently to the nearest observed E and I coordinates;
the exact values and all arm/old-role counts are retained in `activity_probes`.
These are sensitivity probes, not chosen definitions.

| Requested ceiling for both E and I | Screen members | Screen middle-arm | Detailed members | Detailed middle-arm |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 7,086 | 0 | 2,515 | 0 |
| 0.01 | 16,741 | 0 | 8,697 | 0 |
| 0.02 | 17,089 | 12 | 9,581 | 58 |
| 0.05 | 17,647 | 116 | 10,741 | 268 |

At the approximately 0.02 boundaries, 11 of the 12 screen middle-arm records
and 56 of the 58 detailed middle-arm records were previously called active.
The proposed rest definition therefore requires an explicit decision about the
rest/active boundary. It cannot be implemented solely by filling old U labels.

**Recommendation:** use "rest, lower E arm" and "rest, middle E arm" if both
meet the ultimately selected activity criterion. "Inhibition-stabilized rest"
is a supported local qualifier for the middle-arm cases. Defer "para-rest"
unless a distinct functional claim beyond this local mechanism is intended.
Describe low-E/high-I states separately, for example as inhibition-dominated
quiescence; they fail the agreed both-populations-quiet definition.

## Seizure, herald, and intermediate inhibition

Existing seizure labels already span I from numerical zero to 0.04239 in the
screen and 0.02707 in the detailed study. Existing herald labels include both
inhibitory response sides: 4,288 of 5,378 screen herald records and 7,191 of
7,592 detailed herald records are descending-response states. A rule equating
every descending-response state with seizure would erase this distinction.

For an E minimum near 0.02, changing the candidate seizure I ceiling produces:

| Requested I ceiling | Screen members | Of these, old seizures | Detailed members | Of these, old seizures |
| --- | ---: | ---: | ---: | ---: |
| 0.001 | 33,458 | 15,142 | 30,365 | 17,285 |
| 0.01 | 35,627 | 16,912 | 31,215 | 17,887 |
| 0.05 | 35,988 | 17,156 | 31,811 | 18,413 |

These use the observed-coordinate probes, with exact boundaries in the packet.
The E minimum also matters: all three rows include 74 screen and 121 detailed
records previously called rest. They satisfy the descending-response condition
and the proposed activity inequalities, but this study does not promote them
to seizures. The author must decide how far above resting E constitutes
"elevated," including whether an explicit intermediate E interval is needed.

For herald, an E floor near 0.45 and I floor near 0.30 includes 9,667 screen
and 8,116 detailed records. The screen set includes 51 old active records;
the detailed set includes all 7,592 old herald records and 524 old unassigned
records. Raising the I floor near 0.45 reduces those totals to 8,489 and 7,246.
There is no evidence here that one of these numerical boundaries is the
physiological definition. Preserved inhibitory occupancy and response-side
geometry should remain separately visible.

**Recommendation:** use seizure for the low-I failure endpoint under an explicit
E floor; herald for high E with retained I; and "high-E, partially failed
inhibition" for descending-response states between the chosen I boundaries.
Keep elevated-E/low-I ascending states descriptive: low I alone does not
establish inhibitory failure. Branch identity remains separate from local
membership, and continuation through an unstable segment is not switching.

## Interpretation of sensitivity and intervention labels

The 729 scenarios use independent empirical percentile sets: rest-E and
seizure-E {5,15,25}; rest-I and seizure-I {25,50,75}; herald-E {35,50,65};
herald-I {50,75,90}. The broad ranges deliberately expose overlaps and gaps;
several probes are inappropriate as eventual definitions of near-zero or high
activity. In particular, many E quantiles equal 0.5 and some I quantiles are
extremely small. Percentile ranks are not biological thresholds. The readable
occupancy probes supplement them, with both requested and actual values saved.

Across this grid, overlapping memberships range from zero to 14,418 screen
records and zero to 11,604 detailed records. Such overlaps are a property of
independent trial boundaries, not evidence for simultaneous biological regimes.
Gaps can include ordinary active states, intermediate-E states, or other
descriptive states. No priority rule silently resolves either case.

The intervention comparisons retain original local roles, reported roles, and
label origin for all 84 settings, including the 15 inherited coordinate-match
names. They apply each trial definition to the actual destination coordinates.
A fraction of scenarios naming a destination is sensitivity to the declared
grid, not a classification probability. No intervention was rerun.

## Review artifacts and remaining decisions

The gallery is `output/nomenclature_sensitivity_figures_20261016_final/index.html`.
It contains four figures in three groups: coordinate distributions; candidate
boundary surfaces plus membership/intervention sensitivity; and low-activity
dynamics. Full tables and replay commands are in
`reproducibility/nomenclature_sensitivity_20261016/README.md`.

Before changing the classifier, choose the two rest ceilings, seizure E floor
and I ceiling, and herald E/I floors. Decide whether a gap in E is intentional,
how active overlaps should be resolved, and whether middle-arm rest warrants
the inhibition-stabilized qualifier. The current production interface maps a
name to a single root; implementation of revised classes must instead preserve
multiple members and distinguish any separately chosen control-protocol target.

No new simulations, stability promotions, branch confirmations, or physiological
normalizations were introduced. The earlier 29 unmatched continuation checks
remain unresolved and are retained with the copied branch evidence.

## Verification and visual review

The independent verifier passed 2,126 aggregate comparisons: every one of the
1,458 six-boundary scenarios, 600 direct surface checks, and all 68 readable
occupancy probes. It also checked all 135,233 root membership-support records,
all 122,472 intervention/scenario records and their original names, exact
cumulative distributions, and complete identity coverage without duplicates.
The full package suite passed 5,026 assertions in 9m20.9s, including the 31
boundary/counting assertions that also passed in a focused run. The plotting
suite passed 207 assertions.

All four figures were rendered as PNG/PDF/SVG and visually inspected. The
low-activity figure was rerendered with shared comparison scales and stronger
marker contrast, then inspected again, including a reduced-size PDF raster.
The other three final PNGs were byte-identical to the reviewed previews. All
four SVGs parsed, and the rendering manifest and current renderer/input hashes
were verified. The old audit packet, October 6 figure packet, and original
figure-output manifests verified unchanged. Final source review found no
whitespace errors or conflicts.
