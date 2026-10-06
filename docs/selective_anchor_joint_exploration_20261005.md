# Selective-withdrawal anchor: switching and threshold response, 5 October 2026

**6 October replay status:** The switching, induction, and threshold protocols
below were rerun on the merged source after the exact-domain phase handoff
correction. Their destination labels and repeated-switching outcomes agreed
with the original run. The tracked [reference summary](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml)
now records the merged-source hashes and endpoints; the new detailed run is
in [local output](../output/selective_anchor_joint_postmerge_20261006/summary.toml).
The tonic-E scan also completed on the merged source with the same sampled
brackets and destination labels. The earlier paired-withdrawal batch uses the
historical region-selection rule described in the
[selection audit](input_response_selection_audit_20261005.md).

## Result

The selective-withdrawal parameterization now has a measured ordinary switch
at its paired-control baseline: `(e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6)`,
`theta_off=8`, `tau_I/tau_E=0.2`, and `(B_E,B_I)=(0.35,0)`. A 100 ms E-input
increase of 0.5 sends rest to active; a 100 ms I-input increase of 3 sends
active to rest. Julia integration repeats rest → active → rest twice, starting
each phase from the actual previous endpoint. Separate 100 ms pulses at the
same setting send rest to herald with E increase 1 and active to seizure with
I increase 4.5. Each pulse restores the original tonic input.

At the same baseline, increasing `theta_off` from 8 to 8.75 sends the original
seizure source to herald. The same E and I switching pulses still complete two
rest → active → rest cycles at 8.75. In the sampled threshold sweep, the
original seizure source remains seizure at 8.25 and 8.5 and reaches herald at
8.75, 9, 10, and 12. Thus the observed change lies between the sampled 8.5
and 8.75 values; this is not a certified transition threshold. The parameter
change redirects high activity to another high-activity state, not to active
or rest.

The existing paired-control evidence at `theta_off=8` remains separate:
complete E-input withdrawal sends herald to active while seizure persists,
with held-input controls staying at their original sources. [Herald](../output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1/root_5/sustained/confirmations.csv)
and [seizure](../output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1/root_7/sustained/confirmations.csv)
confirmation records support that result. Its finite-pulse brackets are in
the [input-response results](input_response_results.md#finite-withdrawal-can-control-herald-after-the-input-returns).

## Tonic E induction and cessation

A separate [Julia tonic-E replay](../reproducibility/selective_tonic_e_release_20261005/README.md)
starts from rest and active separately at `B_E=0.35` for each of the seven
failure thresholds above. The input is raised and held until a finite-window
destination is classified; the *actual* on-input endpoint is then released to
`B_E=0.35`, `0.17`, or `0`, with a matched kept-on control. All other parameters
and `B_I=0` stay fixed. This is a sustained-input protocol, distinct from the
100 ms induction pulses and from withdrawal of an already established herald
source.

At every sampled failure threshold, both starting states reach herald under
tonic E and then reach active after release to `0.17` or `0`. The first
41-by-41-confirmed qualifying input is absolute `B_E,on=1.2328125`, an increase
of `0.8828125` over the starting baseline. The adjacent lower sample
`1.22890625` does not qualify. Thus the observed onset is bracketed by
`1.22890625 < B_E,on <= 1.2328125`; the same bracket appeared for both
starting states and both lower return inputs at each listed threshold. No
sampled on input through `B_E,on=16` both induced herald and reached active
after release to `0.35`. At the first qualifying on input, release to `0.35`
returned to herald at all seven thresholds.

| `theta_off` | Seizure source at `B_E,on=1.2328125` | Cessation from that source |
| ---: | --- | --- |
| 8, 8.25, 8.5 | Discovered and 41-by-41 checked; kept-on control remains seizure | Seizure after release to `0.35`, `0.17`, and `0` |
| 8.75, 9, 10, 12 | No seizure source identified by the 41-by-41 search | No seizure cessation trial available |

The [reviewed reference summary](../reproducibility/selective_tonic_e_release_20261005/reference_summary.toml)
retains all seven threshold rows; the [local merged-source run](../output/selective_tonic_e_release_postmerge_20261006/summary.toml)
retains 147 sampled on-input points per threshold and the source-specific
induction, control, and release outcomes. These are sampled onset brackets and
finite-window destinations, not exact thresholds or evidence that no narrower
response interval exists. An unavailable seizure source is not a claim of its
absence.

## Evidence and decision boundary

The [Julia replay](../reproducibility/selective_anchor_joint_20261005/README.md)
and its [reference summary](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml)
record the exact pulse settings, sampled threshold values, Julia version, and
source hashes. The local [merged-source run output](../output/selective_anchor_joint_postmerge_20261006/summary.toml)
retains trajectory and diagnostic files. Baseline and threshold-8.75
equilibria were searched on 41-by-41 grids; the other threshold samples used
21-by-21 grids. Each reported destination met the existing equilibrium
compatibility diagnostic within a 5,000 ms follow-up, under tenfold tighter
solver tolerances. The two-cycle sequences use the same strict checks.

This adds one tested parameterization that combines ordinary switching,
separate access to both high-state coordinate roles, and the previously
reported paired input-control result. The paired result remains historical
until its response protocol is replayed on the corrected source. Its threshold
change also preserves measured switching while redirecting the original
seizure source. It is a
candidate for a unified paper example, subject to author selection. The
threshold-changed setting has not been tested for withdrawal from its
already established herald source at `B_E=0.35`; the tonic E onset/release
measurement answers a different question. No parameter neighborhood of
switching has been established.
All roles remain provisional, with `CompletenessNotCertified`; these finite
trajectories do not establish asymptotic rescue or biological interpretation.
