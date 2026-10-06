# Selective anchor joint-behavior replay

Run from the repository working copy with Julia 1.10 and its recorded project
environment:

```sh
julia --project=. reproducibility/selective_anchor_joint_20261005/replay.jl output/selective_anchor_joint_replay_new
```

The output path must not exist. The replay uses the Julia point model and the
existing narrative study's equilibrium, pulse, and finite-window diagnostic
rules. It fixes the selective-withdrawal setting at
`(e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6)`, `theta_off=8`,
`tau_I/tau_E=0.2`, and `(B_E,B_I)=(0.35,0)`.

It searches the baseline on a 41-by-41 grid, applies 100 ms pulses with the
same tonic input restored afterward, and repeats rest-to-active-to-rest twice
from each actual endpoint. It also tests rest-to-herald and active-to-seizure
pulses. For `theta_off` values 8, 8.25, 8.5, 8.75, 9, 10, and 12, it follows
the original seizure source after the parameter change and checks the same
switching pulses. The 8.75 setting uses a 41-by-41 grid and repeats the full
two-cycle switching sequence; the other threshold settings use 21-by-21 grids.
Solver tolerances are tenfold tighter than the narrative protocol's defaults.

`summary.toml` records role correspondence, sampled root and sink counts,
destinations, pulse protocols, Julia version, and source hashes. The output
also contains the repeated-switching trajectories, high-state induction
trajectories, and the threshold-8.75 seizure-source trajectory. The
[reference summary](reference_summary.toml) records the 6 October replay on the
merged source after the exact-domain phase handoff correction. The original
5 October run remains in the local archive. Its reported destinations and
switching outcomes agreed with the merged-source replay; source hashes and
two induction endpoints changed.

These are finite-window observations at sampled settings. The replay does not
certify all attractors, a continuous threshold boundary, a biological state
identity, or switching throughout a parameter region. The paired E-withdrawal
result at the unmodified anchor is retained in the earlier input-response
study; this replay does not repeat it after the threshold change.
