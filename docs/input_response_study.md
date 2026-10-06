# Two-input response characterization

## What the study characterizes

A parameterization is a fixed set of population responses, couplings, and time
constants. Its response depends on both tonic inputs, the starting state, and
the intervention protocol. This study retains those dependencies instead of
assigning a single rescuable/unrescuable label.

The study has an equilibrium/stability atlas, source-specific sustained-input
maps, equilibrated input sweeps, finite pulses, and direct E-displacement
measurements. Herald and seizure remain distinct provisional coordinate roles.
Recovery in these measurements means a rest or intermediate-active equilibrium;
validated oscillatory destinations are reported separately. No spatial or
feedback model, biological calibration, or manuscript revision is implied.

## Exact input chart

For the existing equations, define `H_X(u) = F_X(u)/(1+F_X(u))`. A pair of total
population inputs `(u,v)` gives

```
E = H_E(u)                     I = H_I(v)
B_E = u - a H_E(u) + b H_I(v)
B_I = v - c H_E(u) + d H_I(v)
```

Here `(a,b,c,d)=(e_to_e,i_to_e,e_to_i,i_to_i)`. The chart covers every finite-input
equilibrium, including both sides of the inhibitory response. The sampled
numerical atlas is not a complete enumeration. Negative chart inputs are
excluded from the declared afferent-drive study, not clipped into zero input.

The determinant of the balance Jacobian equals the input-chart determinant
multiplied by `(1+F_E)(1+F_I)`. This identifies the same singular set. For the
logistic E response, `H_E' = s_E E(1-2E)`, allowing direct quadratic seed formulas
for both singular and trace-zero curves. The runner refines these formulas in
total I input and retains numerical fold/Hopf candidates. It does not classify
degenerate intersections or certify bifurcations. Candidate Hopf points require
positive determinant. Timescales change stability and phase-plane flow, while
leaving the equilibrium chart unchanged.

Local susceptibility is computed as `-J^(-1) ∂f/∂B` using the original ODE.
Ill-conditioned inversions remain unresolved. Independent root searches,
necessary planar index checks, and existing axis continuation provide checks
on discovery. An index sum of one does not establish completeness.

## Domain and sampling

Each external-input axis begins at 0–16 and doubles independently until uniform
response-tail bounds hold over `[0,1/2] × [0,I_max]`, the invariant box containing
the source equilibria and tested input-driven trajectories. `I_max` is derived
from the maximum of the approved inhibitory response. At the E upper edge,
`F_E >= 1-1e-6`; at the I upper edge, the descending response satisfies
`F_I <= 1e-6`. A configurable 12-doubling budget retains an unresolved flag if
these bounds are not reached. Tail bounds do not certify uniqueness, absence of
cycles, or absence of further pulse-response changes.

Root seeds combine the chart, standard independent seeds, and a state grid.
A failed planar index check triggers a 21×21 fallback. Representative inputs
are independently checked at 41×41. All attempts, roots, stability results,
and unresolved outcomes are retained. Input cells are sampled at corners and
centers in breadth-first order. Changed or unresolved cells refine toward
width 0.01 within explicit evaluation budgets. Homogeneous samples do not
exclude narrower unsampled islands.

Critical-set offsets and known baseline inputs are visited before the lattice.
Their observed signatures also enter adaptive cell refinement, so a cell
containing a different pre-sampled signature cannot be labeled homogeneous
solely from its five lattice probes. A final pass checks each homogeneous cell
against points discovered while neighboring cells were refined. Budget-exhausted
and mixed cells remain unresolved observations rather than established regions.
Connected components of the resolved sampled-cell adjacency graph define
observed regimes. Select one representative by largest distance from sampled
critical curves and rectangle edges, breaking ties by input coordinates. This
Euclidean distance is only a sampling rule, not a control cost. Isolated points
in unresolved cells remain probes rather than established connected regions.
All known baseline inputs are retained as additional representatives.

Role classification uses E-nullcline arm geometry and the full total inhibitory
input, including B_I. High-state names require relative inhibitory separation.
States without a unique role remain available as numbered equilibrium sources.
Nearby branch correspondences reject ties and collisions; root numbers are
local to an input context. Neither discovery nor source availability requires
a zero-input counterpart.

## Dynamic protocols

Every representative is probed from a deterministic 3×3 state grid. Persistent
recurrence seeds the existing periodic shooting and Floquet checks; recurrence
alone is not a validated periodic orbit. Attracting cycles supply eight source
phases, with intermediate phases added when the sampled sustained destination
maps differ. Phase sampling remains finite.

For each source, sustained changes are sampled across the full declared input
rectangle. Held-input controls and independently confirmed destination witnesses
retain trajectories. E withdrawal and I stimulation have separate coordinates;
Pareto tables retain nondominated sampled successful pairs without a weighted
cost. Transitions between herald and seizure are not recovery.

Finite pulses use the four axial input directions plus combined E withdrawal
and I stimulation at 1, 2, 5, 10, 20, 50, 100, and 200 ms. Total inputs remain
nonnegative and the original baseline is restored after each pulse. Pulse maps
retain actual offset states. Successful destination witnesses are repeated at
tighter tolerances. All line/cell budget limits remain visible.

Input sweeps begin from each source, traverse both directions, and carry actual
terminal states. Changed outcomes trigger step halving toward input width 0.01.
These finite-settling sweeps are a specified hysteresis protocol, not a proof of
an infinitely slow limit. Follow-up horizons are 5,000, 10,000, and
20,000 ms with two terminal diagnostic windows. At each horizon, an unresolved
equilibrium diagnostic triggers a 1,000 ms periodic-seed check. A numerically
validated attracting cycle can terminate the horizon search; otherwise the
longer horizons are retained. The seed requires at least five upward crossings
and less than 2% spread in the last four crossing intervals; slower or irregular
recurrence can remain unresolved under this protocol. Solver failures and unresolved
settling are retained rather than assigned a destination.

This study sets the existing solver's `domain_atol` to zero. Switching protocols
reuse terminal states as new initial states, whose public contract requires
exact membership in `[0,1]^2`. Allowing a tiny negative solver endpoint would
make that restart invalid. Adaptive domain rejection enforces this contract,
and saved times are solver stops to avoid negative interpolation roundoff near
the inhibitory tail. No endpoint is clipped or projected onto an equilibrium.

Direct E displacement holds I and both tonic inputs fixed. Herald and seizure
are compared only at shared inputs with both source roles available. Smallest
observed successful reductions are reported at sampled width 1e-4. Their ratio
is unavailable if either measurement is missing; it is not an input-cost ratio.

## Parameter comparisons and execution

The eight anchors comprise the four original exemplars, the narrative example,
selective positive-input withdrawal, the threshold-to-active counterexample,
and the weak-recurrence input-release example. Local perturbations use offsets
0.25 in recurrent excitation, 0.5 in recruitment, 0.25 in failure threshold,
and 0.2 in timescale ratio, within the previous declared ranges. Each coupling
family also receives 64 deterministic joint Halton samples. Neighbor intervals
with changed structural or fixed-protocol behavior receive midpoint refinement,
subject to 512 additional parameterizations per family.

The joint screens hold `(b,d)` at `(9,4)` or `(13,6)` and vary `a` over 0–24,
`c` over 12–28, failure threshold over 6–12, and `tau_I/tau_E` over 0.2–4.4.
The configuration must name these four distinct expansion axes; unsupported or
repeated keys fail before any case runs. Time-scale ratio bounds must be
positive, and failure-threshold bounds must exceed the onset threshold in
every anchor model.
Both response slopes remain 5, the E threshold is 1.5, the inhibitory onset is 4,
and `tau_E` is 7.8 ms. These are the declared study domains, not a survey of
every model parameter or a biological calibration.

Expansion screening includes the geometry and explicit fixed input/pulse probes.
Detailed maps are then run for the first deterministic representative of each
observed coexistence/response signature. Sharing a sampled signature does not
prove two parameterizations behaviorally equivalent. The case order, source
snapshots, budgets, and configurations make selection replayable.
Only cells with a homogeneous observed signature connect sampled inputs into
a region. The [dated selection audit](input_response_selection_audit_20261005.md)
records how the mixed-cell graph and cell-label corrections affect retained
historical observations and what remains unrerun.

The new scripts do not change the model or public drive API. Run from the
working-copy root with the existing Julia project:

```sh
JULIA_NUM_THREADS=2 julia --project=. scripts/run_input_response_study.jl --output output/input_response_new
```

The study writes checked CSV and TOML observations. The earlier exploratory
Python renderer and its HTML atlas have been retired from the active source;
the dated local reports remain historical outputs, not a fresh-checkout
rendering step. Paper figures have their own Julia renderer and evidence bundle.

Runner stages are `geometry`, `responses`, `expand`, and `all`; `--case` filters
anchor IDs. `--smoke` reduces numerical budgets and durations and is not a
scientific run. Checkpoints verify source/configuration identity and completed
unit checksums. A changed source requires a new output directory; archived
source supports exact replay. Independent cases run at most two at a time.
Scientific scans are local/on-demand; CI tests the machinery and fixtures.
Metadata retains the completed stage invocations in replay order, including
case filters and smoke mode.

## Lossless storage of completed searches

Detailed geometry records can exceed available disk space during a large run.
Between runner invocations, the optional command
`julia --project=. scripts/archive_input_response_contexts.jl OUTPUT` packs completed
`geometry/contexts/` directories into `geometry/contexts.tar.gz`. Detailed cases
must first finish their independent geometry confirmations; screen cases can
be packed once their geometry checkpoint is complete. When the runner has
written a top-level `checksums.toml`, the archiver verifies that manifest
before changing files and regenerates it atomically after successful packing.

Every original file is checksum-verified inside the archive before its unpacked
copy is removed. The archive retains the original `done.toml` at
`_checkpoint/done.toml`; `context_archive.json` records the conversion and script
hash. The replacement geometry checkpoint verifies the archive and remaining
tables, so the existing runner can resume without unpacking the search records.
Extract `contexts/` members into the geometry directory when inspecting raw
search attempts. Summary tables, confirmations and response trajectories remain
unpacked. Packing is a storage operation and changes no numerical observations.

The storage checks are included in the Julia package suite. The Julia tool
verifies archives written by the earlier Python tool without rewriting them.
