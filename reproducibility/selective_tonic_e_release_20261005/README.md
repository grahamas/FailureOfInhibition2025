# Selective anchor tonic-E induction and release

This Julia replay holds `(e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6)`,
`tau_I/tau_E=0.2`, and `B_I=0` fixed. At each `theta_off` in
`[8, 8.25, 8.5, 8.75, 9, 10, 12]`, it starts separately from rest and active
at `B_E=0.35`, raises absolute tonic `B_E`, and follows the actual on-input
endpoint. It then releases that endpoint to `B_E=0.35`, `0.17`, and `0`, with a
matched kept-on control. Where a seizure source is discovered at the first
confirmed qualifying input of a sampled success interval, it tests that source
under the same three releases and its own kept-on control.

Run from the working-copy root with Julia 1.10:

```sh
julia --project=. reproducibility/selective_tonic_e_release_20261005/replay.jl output/selective_tonic_e_release_20261005
```

The output directory must not exist. If interrupted, use the same command with
`--resume`; the replay checks the source and protocol hashes before reusing
completed input points. `metadata.toml` records the protocol and source hashes,
`points/` records every sampled input, `theta/` records confirmed brackets and
seizure comparisons, and `summary.toml` collects the seven thresholds. The
generated `reference_summary.toml` is compact; its reviewed copy is
[tracked here](reference_summary.toml). The tracked reference was regenerated
on 6 October against the merged source after the exact-domain phase handoff
correction. All seven sampled brackets and source/destination labels agreed
with the original run; four source hashes changed.

Each input is held until the existing finite-window equilibrium diagnostic
resolves it or exhausts 5,000, 10,000, and 20,000 ms follow-ups. Searches use
independent on/off equilibria at their actual input. A 21-by-21 grid screens
inputs; qualifying bracket endpoints are repeated with 41-by-41 grids and
tenfold tighter solver tolerances. The scan samples every 0.25 input unit and
its interval midpoint, then refines observed changes and unresolved intervals
to width 0.01. It starts over `B_E=0.35–8` and extends to `16` only if a
source/return combination lacks a qualifying sample by `8`.

The reported values are sampled brackets. Equal outcomes at sampled neighbors
do not exclude narrow unsampled intervals; a failed or unresolved measurement
does not establish absence. Roles are provisional, and the study retains
`CompletenessNotCertified`. It does not certify an exact threshold, asymptotic
recovery, or a biological state identity.
