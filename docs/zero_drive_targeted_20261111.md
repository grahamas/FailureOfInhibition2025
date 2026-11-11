# Targeted zero-drive pilot: readiness gate stopped the run

The second pilot attempted all three readiness controls and stopped because
the cusp control remained numerically unresolved. BT and Hopf passed. None of
the 19 targeted investigations ran, so this packet adds no recovery results,
transition localizations, continuation extensions, or local adjacency map.
`protocol_complete` is false and `CompletenessNotCertified` is retained.

## Execution and scope

The [frozen 22-job protocol](zero_drive_targeted_protocol.md) was implemented
and independently reviewed in [PR #160](https://github.com/grahamas/FailureOfInhibition2025/pull/160).
The run used merged commit `269e24d05d6f56e965aa65ca688cb1e631a46054`.
Its controls ran in order; the failed readiness gate left the two diagnoses,
four recovery refinements, three multiroot recoveries, three transition
investigations, and seven paired continuation extensions pending.
These are unattempted investigations, not negative scientific results.

Worker setup took 157.404 seconds; control execution took 36.590 seconds.
The new packet occupied 10,088,448 bytes after report generation and before
visual-review files. Together with the immutable first packet's 54,382,650
bytes, the combined footprint was 64,471,098 bytes, below the
5,000,000,000-byte ceiling.
Checkpoint storage values were measured before final report writes; they
are not the final directory footprint. Review files are additional owned
artifacts, also subject to the combined ceiling. After independent visual review
and the accompanying `OUTCOME.md` note, the measured combined footprint was
67,821,683 bytes (about 67.82 MB, or 1.36% of the ceiling).

## Why the cusp control was unresolved

The targeted protocol used event tolerance `1e-9`; the first cusp control
used an effective event cap of `1e-8`. At the same analytic cusp seed, the coarse finite-difference event
residual was `1.1327235446e-9`. The stricter run therefore took one Newton
step, changing `E` from `0.25` to `0.24999999994690356`.

| Check at the corrected tuple | Saved value | Outcome |
| --- | ---: | --- |
| Balance residual | `0` | Passed |
| Coarse event residual | `9.2518585385e-14` | Passed |
| Independent half-step quadratic residual | `1.0616877747e-9` | Above `1e-9` |
| Null-vector residual | `2.2204460493e-16` | Passed |
| Cubic coefficient refinement | Passed | Passed |

Independent arithmetic on the saved tuple gives an analytic quadratic
coefficient of approximately `1.1327e-9`; at the original analytic seed it
is approximately machine zero. This supports a finite-difference bias in the
coarse correction equation: correction reduced that equation's residual while
moving away from the cusp tested by the refined calculation. It does not
indicate that the known cusp is absent, and it is not a solver exception.
The independent refinement gate correctly prevented numerical validation.

## Remaining decision

The original 19 investigations remain open. Before another run, choose and
review a cusp-validation numerical policy that can satisfy its own independent
checks at the prescribed tolerance. Improving derivative accuracy is preferable
to weakening the threshold solely to pass this control. Any change needs
bounded analytic-control regression tests and a separately frozen run; preserve
this failed-readiness packet and do not resume it with changed source or policy.
No replacement run or tolerance relaxation was performed.

## Retained evidence

Local packet: `local_audits/zero_drive_targeted_20261111/`.
Read `OUTCOME.md` first: it explains the failed gate and distinguishes planned
work from execution in the original generated report. `index.html` shows
control status and unattempted cohorts; the empty
`targeted_graphs.toml` contains no inferred adjacency evidence. Raw control
results retain both correction attempts and independent validation diagnostics.

| Artifact | SHA-256 |
| --- | --- |
| `state.toml` | `1ccf6d45d0aede435e28e96f4cf570d848302a2db1e30c43106f416701550603` |
| `targeted_manifest.toml` | `127fb89006ebec21e63ba96ec2ed20d158ee4347a815d18a2e2c5dbbe0f1b573` |
| `targeted_summary.toml` | `898bcc5ec0f1f055aa344c97f9dafe4fd8d744c6b5c53b83f3735f363230c393` |
| `targeted_graphs.toml` | `f11d919c4ab4163a44afe0a527c93b8ace331b63cd43ef0d96ac45148f6da77b` |
| `index.html` | `c39b82c61fd5e64a6352b9944519476cbd122fb914d8129504bb4e410b5a17ed` |
