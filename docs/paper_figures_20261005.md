# Selective-anchor figure packet — 5 October 2026

**6 October reconciliation:** The switching/threshold and tonic-E source
replays were rerun after the merged exact-domain phase-handoff correction.
Their sampled roles, destinations, brackets, and repeated-switching outcomes
agreed with the original records. The figure data bundle was rebuilt from
those merged-source runs. Supplement S2 still draws from the frozen two-input
response batch and is labeled historical; its responses have not been rerun
under the corrected region-selection rule.

These figures present one tested point-model parameterization across the main
argument: `(e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6)`,
`tau_I/tau_E=0.2`, and baseline `(B_E,B_I)=(0.35,0)`. The plots use the
[portable Julia data bundle](../reproducibility/paper_figures_20261005/README.md)
and the [reviewed switching/intervention](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml)
and [tonic-release](../reproducibility/selective_tonic_e_release_20261005/reference_summary.toml)
records. The reviewed renders are in local
`output/paper_figures_postmerge_20261006/`; that ignored directory is not part of the
reproducible source packet. Manuscript captions and interpretation remain for
author review.

## Main figures and candidate captions

**Figure 1 — Mechanism and state arrangement.** An ordered failure threshold
produces a nonmonotone inhibitory response. At the selected baseline input, the
numerical search discovered rest, active, herald, and seizure coordinate roles
among four attracting equilibria, together with three other roots. The
monotone-inhibition ordering result concerns *two equilibria under the same
parameters and fixed input*: higher E with lower I is excluded there. The
figure does not present the numerical search as a complete root census or the
coordinate roles as biological classifications.

**Figure 2 — Ordinary switching and access.** A 100 ms E-input pulse of
amplitude `0.5` sends rest to active, and a 100 ms I-input pulse of amplitude
`3` sends active to rest. The full sequence repeats twice from each actual
previous endpoint. Separate 100 ms pulses send rest to herald with E amplitude
`1` and active to seizure with I amplitude `4.5`. Each pulse returns to the
baseline tonic input. The endpoint panel displays the complete two-cycle
sequence; short traces display early dynamics, while classifications use the
recorded 5,000 ms follow-up. Rest–active switching is an operational model
behavior, not a demonstrated cortical computation.

**Figure 3 — Tonic input control distinguishes source states.** At
`theta_off=8`, held absolute `B_E,on=1.2328125` takes baseline rest and active
separately to herald. Releasing the actual herald endpoint to `B_E=0.17` or
`0` reaches active; returning to `0.35` or keeping the on input remains at
herald. A separate seizure source discovered at the same on input remains
seizure under all three releases and its held-on control. That seizure source
was identified for `theta_off=8`, `8.25`, and `8.5`; none was identified at the
same on input for `8.75`, `9`, `10`, or `12`, so those four values have no seizure
cessation trial. These are finite-window outcomes. The sampled herald onset is
bracketed by `1.22890625 < B_E,on <= 1.2328125`, not an exact threshold.

**Figure 4 — A threshold intervention with ordinary switching retained.**
Raising `theta_off` from `8` to `8.75` sends the *original* seizure source to
herald, while the unchanged switching pulses complete two rest–active–rest
cycles at the new setting. At `8`, `8.25`, and `8.5`, the original source
remains seizure; at `8.75`, `9`, `10`, and `12`, it reaches herald. Searches
discovered seven roots/four sinks at the lower samples and five roots/three
sinks at the higher samples. This sampled change does not locate or certify a
bifurcation, and redirection to herald is not recovery to active or rest.

## Supplement and paper boundary

**S1** shows every measured tonic-on input for rest and active across the seven
thresholds, with a zoom around the sampled onset and the full `0.35–16` range.
Each dot is a sampled source/input outcome; no cells are filled or
interpolated. The plot describes this tonic protocol at fixed coupling and
time-scale ratio. It is not a continuous parameter-region map and does not
measure ordinary switching at every plotted input.

**S2** presents the earlier finite E-withdrawal pulse observations from
already established herald and seizure states at `B_E=0.35`. Pulse input is
restored to `0.35` afterward. This protocol is distinct from Figure 3's
sustained induction at `B_E,on=1.2328125` followed by release to a possibly
lower tonic input. Its source roles, amplitude, duration, and destinations
must remain separate in the manuscript and Methods. These frozen-batch rows
remain historical until current-source response protocols are rerun.

The four-figure sequence is sufficient for a bounded existence-and-control
example in this deterministic point model. It does not support a general
claim about parameter-space robustness, an exact bifurcation mechanism,
physiological calibration, seizure elimination, or a clinical intervention.
The time-scale ratio `0.2` is exploratory and differs from the old manuscript
ratio `4.4`. A stronger parameter-region or bifurcation claim would need new
measurements; the main figures use merged-source replays while S2 retains
historical frozen-batch observations.

## Review checks

The data builder checks the retained joint and tonic summary hashes, every
consumed trajectory and tonic point against a reviewed artifact digest, the
frozen response archive manifest and selected file hashes, source endpoints
and destinations, and all 147 input points at each of seven thresholds. The
renderer checks the portable bundle hashes, model and plotting sources, and
all four reference files, and refuses an existing output directory. Final
review should confirm the six figures in PDF, SVG, and PNG at manuscript scale
and retain the provenance record written beside them. The plotted roles remain
provisional and all equilibrium searches retain `CompletenessNotCertified`.
