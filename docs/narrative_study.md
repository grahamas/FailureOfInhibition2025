# Activity, persistence, and selective control

This study implements the author-selected narrative plan with the approved
point model. Rest–active switching is an operational behavior, not validation
of cortical computation. Herald and seizure remain distinct provisional
coordinate roles. No feedback controller, coupled population, spatial model,
or manuscript change is introduced.

## Protocol

`scripts/run_narrative_study.jl` uses `experiments/narrative_study.toml`.
It searches the two existing coupling families with deterministic Halton
samples in recurrent excitation 0–24, E-to-I recruitment 12–28, failure
threshold 6–12, time-constant ratio 0.2–4.4, and tonic E input 0–8. Other
model quantities retain the existing conventions; tonic I input is zero.
Each family receives 1,024 initial joint samples. The budget expands to
4,096 per family only if no initial candidate qualifies. Established anchors
are included explicitly, with finer positive-baseline probes near zero.

Discovery is independent at each input. Role assignment never requires a
zero-input counterpart. All roots and solver attempts are retained, including
unassigned roots. The lowest-E sink is called rest only when separated from
additional sinks. The active candidate is on the intermediate E-nullcline
arm (positive E self-derivative) and rising inhibitory response. This avoids
relabeling a saturated herald state as active merely because its E coordinate
is slightly smaller than the seizure candidate's. Herald and seizure are
distinct high-E, respectively higher-I and lower-I coordinate roles.
Ambiguous assignments remain unavailable. Branch correspondence uses unique
coordinate matches with collision rejection, not persistent root indices.

A candidate qualifies only if input pulses produce rest → active → rest
twice, using actual endpoints between phases, and either rest or active can
be driven to the seizure candidate. Each pulse restores the same tonic input.
Positive E, positive I, and temporary withdrawal of available E input are
tested separately; total afferent input never becomes negative. The amplitude
cap is 8 and the duration grid is 1, 2, 5, 10, 20, 50, 100, 200 ms.
Amplitude intervals include midpoint probes and are refined where sampled
destinations differ. Equal-outcome intervals can still contain missed islands.

Selection prefers a positive tonic input, then coexistence with a herald
candidate, then the deterministic case ID. It does not prefer an expected
herald–seizure recovery ordering. Selected switching and induction witnesses
are repeated with 21×21 and 41×41 equilibrium grids and tenfold tighter
solver tolerances.

## Two reduction measurements

The same positive-baseline model supplies both herald and seizure sources.
Recovery targets are rest and intermediate active; a transition between the
two high-activity roles is recorded separately.

| Protocol | What changes | Destination context |
| --- | --- | --- |
| Permanent withdrawal | `B_E` becomes `B_E - ΔB_E`, with `0 ≤ ΔB_E ≤ B_E` | Independently searched reduced-input system |
| Direct displacement | Initial state becomes `(E - ΔE, I)`, with `0 ≤ ΔE ≤ E` | Original model and original tonic input |

Both protocols sample the whole allowed interval with at least 32 subintervals
and interior probes. Withdrawal also respects the absolute amplitude spacing;
outcome changes receive finer refinement. State-displacement refinement uses
width `1e-4`. Neither protocol assumes monotone success. Reports give smallest
observed successes and adjacent sampled outcome brackets, not exact necessary
minima. Input units and state-coordinate units are not interchangeable.

The successful boundary samples, interval endpoints, and zero controls are
reintegrated with tighter tolerances. When an induction witness exists, the
reduction is repeated from its actual endpoint and compared with an
unchanged-input continuation from that identical endpoint. Direct displacement
is a basin experiment; it is not itself an afferent-drive intervention.

## Intervention and neighborhood studies

Each intervention changes one parameter: failure threshold through 12,
recruitment or recurrent excitation through 60% of baseline, or inhibitory
output through 150%. Rest/active pulse maps are remeasured, switching is
repeated, and trajectories from all baseline sources show where they go after
the parameter changes. Activity coordinates remain separate from switching
and induction outcomes. No weighted functional score is calculated.

The local input/recurrent-excitation map includes equilibria, the original
switching and induction witnesses, and full-withdrawal outcomes. Map steps
must be positive and widths nonnegative; intervention axes must be nonempty
and model-safe. These settings are checked before screening. Continuation
retains attracting and unstable branch segments. Neighbor tests use
`e_to_e ±0.25`, `e_to_i ±0.5`, `theta_off ±0.25`, and time-constant ratio
`±0.2` within the declared bounds. Neighbor switching tests reuse the selected
stimuli: failure does not establish that another stimulus could not succeed.

## Numerical limits and reproduction

Follow-up escalates through 5,000, 10,000, and 20,000 ms. Two terminal 100-ms
windows must satisfy the existing equilibrium-compatibility diagnostics and
the destination must be locally attracting. Solver failures and unresolved
diagnostics are retained. These are finite-window observations, not global
attractor counts, proof of asymptotic recovery, or physiological calibration.

```sh
julia --project=. scripts/run_narrative_study.jl --output output/narrative_new
julia --project=plotting scripts/render_narrative_study.jl output/narrative_new output/narrative_figures_new
```

Stages are `screen`, `confirm`, `recovery`, `interventions`, `map`, `robustness`,
and `all`. `--case` filters screening case IDs by substring. `--smoke` uses two
joint samples per family, shorter follow-up, and two pulse durations; it is
not a scientific replication. Rerunning a stage verifies source/configuration
identity and completed-unit checksums before resuming. Changed source or
configuration requires a new output directory. Archived source supports replay
when the working copy has moved on. Metadata records each completed stage
invocation, including its stage, case filter, and smoke mode, in replay order.
The selected confirmation grid is written to `selected.toml` and used by the
renderer; older artifacts use their archived configuration to identify it.

The full retained evidence is in `output/narrative_final_20260928`. Figures
and the standalone report are in `output/narrative_figures_20260928`. The first
exploratory run and incomplete smoke runs are retained separately and are not
the final evidence. Results and their manuscript use are summarized in
[narrative_results.md](narrative_results.md).
