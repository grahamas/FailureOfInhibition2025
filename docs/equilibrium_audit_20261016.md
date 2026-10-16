# What the unassigned equilibria are

## Question and scope

This audit investigates the refreshed 232-case screen and eight detailed
circuits underlying the October 6 figures. It separates numerical evidence,
intrinsic equilibrium geometry, contextual role assignment, and correspondence
to a state in a different context. Existing role assignments and manuscript
figures are unchanged. The older historical studies are outside this audit.

The central finding is that **unassigned is an outcome of the naming rules,
not a dynamical state type**. Several geometrically distinct attracting states
share that label. Some are continuations of named states; others expose the
one-candidate-per-name restriction or a prerequisite of the classifier.

## What the existing classifier requires

`scripts/input_response_models.jl` assigns roles only to attracting,
non-near-singular roots. It first requires two turning points of the logistic
E-nullcline (`e_to_e * slope_e > 8`). It then identifies lower, middle, and
upper arms using those turning points.

- Rest requires exactly one attracting root on the lower arm.
- Active requires exactly one attracting middle-arm root on the ascending
  inhibitory response.
- Seizure requires exactly one attracting upper-arm root on the descending
  inhibitory response with another attracting equilibrium having greater I
  and no greater E, within the existing coordinate margins.
- Herald requires an already identified seizure and exactly one distinct
  upper-arm root with greater I.

These are contextual predicates, not independent biological definitions.
Periodic orbits do not participate in their comparison sets. The audit uses
the existing predicates and margins unchanged; its additional descriptors
record E-nullcline arm and position relative to the inhibitory response peak.

## Findings that could change the figures

### The equilibrium beside the periodic orbit is an established upper descending sink

At weak-circuit input `(B_E,B_I)=(2,1)`, the attracting equilibrium is
`(E,I)=(0.4999990682,0.0000037272)`. It occupies the upper E-nullcline arm
and descending inhibitory response. It lacks the higher-I attracting
equilibrium required by the seizure predicate. The coexisting validated cycle
does not satisfy that predicate's equilibrium comparison.

Continuation in I input connects it to contexts where the local classifier
assigns seizure. The recorded time-scale scan provides an additional controlled
comparison: equilibrium coordinates are fixed while stability of the comparison
equilibrium changes. The upper state remains attracting throughout the scan.
This distinguishes a naming change from disappearance of the upper state.
It does not identify the periodic orbit's birth or make a biological seizure
classification.

Specifically, the upper root is named seizure at recorded ratios 0.2–2.4,
and unassigned from the trace-zero candidate near 2.43950 through 4.4.
Its coordinates are identical across all those records. The loss of the
comparison sink, rather than any change in the upper root's coordinates,
accounts for this labeling boundary.

**Figure implication:** the phase-control destination in Figure 4 should be
explained by its measured coordinates and geometry, rather than presented as
an otherwise unknown equilibrium. A broader seizure label remains an author
decision about the comparison requirement.

### Recruitment interventions reveal recognizable branches and hidden label provenance

In the reference circuit, a representative unassigned destination at
`E→I=14.25` has approximately `(E,I)=(0.49971,0.46096)` and lies on the
upper/ascending geometry. Continuation in recruitment connects this branch
to locally labeled herald contexts. In the contrast circuit, the representative
at `E→I=13.3` has approximately `(0.49993,0.41279)`; its continuation reaches
locally labeled active contexts. Neither connection by itself demonstrates
recovery of ordinary switching.

The existing intervention analysis also has a documented fallback: when a
destination has no local role, a unique nearby root correspondence may inherit
a role from the original context. Consequently, a reported herald destination
can be locally unassigned. The audit retains both labels and the matching
origin, rather than changing either retroactively.

Across the 84 recorded intervention settings, 61 destination labels are local,
15 are inherited from the original context, and eight retain numbered local
equilibrium identities. These counts describe the existing implementation;
the inherited labels are not new results from this audit.

**Figure implication:** separate destination coordinates, local role eligibility,
and inherited branch names. An unchanged word can conceal a change in how that
word was assigned.

### A single name can have multiple legitimate candidates

| Representative | Attracting states relevant to ambiguity | Why the current name is unavailable |
| --- | --- | --- |
| Weak circuit, input `(0,8.8125)` | Approximately `(0,0.49933)` and `(0.000157,0.02803)` | Both are on the lower E-nullcline arm; the rule demands one rest candidate. |
| Reference, input `(0.425,2.3125)` | Approximately `(0.01211,0.000670)` and `(0.07656,0.05441)` | Both are on the middle arm with positive inhibitory-response slope; the rule demands one active candidate. |
| Four-sink circuit, input `(0.07421875,0)` | Approximately `(0.50000,0.44772)` and `(0.50000,0.000559)` | Both upper descending roots satisfy the relative-coordinate seizure predicate. |

These are separate roots in the retained numerical records, not a duplicate
name for one coordinate. Their local stability and continuation are checked
in the follow-up package. The high-I lower-arm state illustrates direct
inhibitory drive suppressing E activity; a lower E coordinate alone does not
distinguish it from a low-I resting configuration.

**Figure implication:** input maps can contain several states with the same
geometric descriptor. A descriptor must permit multiplicity and retain local
root identity; replacing `U` with a single new role would conceal this structure.

### Some circuits are excluded before any individual root is considered

The no-turning-points cases fail the classifier's global prerequisite, even
when a root is attracting and has otherwise ordinary coordinates. Their
unassigned status does not indicate a failed equilibrium solve. Continuation
under input changes preserves that prerequisite failure because recurrent
excitation is fixed.
Absence of E-nullcline turning points does not imply a unique equilibrium:
the inhibitory response can still support multiple branches.

**Figure implication:** a naming scheme built around a folded E-nullcline does
not cover every circuit in the study's declared parameter envelope.

## Classification proposals for author review

| Proposal | Benefit | Scientific choice that remains |
| --- | --- | --- |
| Add intrinsic descriptors alongside current names | Explains every displayed U without changing biological terminology; supports multiple roots per descriptor. | Whether these descriptors should become the main map legend or supporting labels. |
| Track branch identity separately from local roles | Makes names lost through changing comparison states explicit; distinguishes continuation from coordinate matching. | Whether a branch may retain a biological name after the local defining conditions cease to hold. |
| Revise the comparison requirement for seizure | Could name the upper descending sink coexisting with a cycle. | Whether geometry alone, comparison with another equilibrium, or comparison with any attractor defines the intended role. |
| Permit multiple candidates per role | Represents genuine multiplicity rather than collapsing it into U. | Whether rest/active/herald/seizure are classes or uniquely designated states within a context. |
| Rework the no-turning-points policy | Extends descriptive coverage across the existing envelope. | Which terms remain meaningful when lower/middle/upper E arms do not exist. |

The least committal presentation change is to add descriptors and label
provenance first. None of the proposed rule changes has been implemented.

## Evidence package and numerical limits

The full context inventory has 77,455 recorded contexts and 215,657 root
records, including repeated parameter/input contexts from different study
stages. Of these, 120,452 are unassigned and 40,482 are attracting unassigned
roots. Counts are separated by source stage in the package. They are neither
unique biological states nor estimates of parameter-space prevalence.

The inventory verifies archived geometry checkpoints and contents, reproduces
all stored local role assignments, and retains source digests. Time-scale
spectrum observations reuse fixed root coordinates and are tabulated separately
from the archived-context counts. Pulse and intervention labels are audited
without rerunning their control protocols.

Eleven deterministic seeds cover every observed reason/geometry combination
among attracting, non-near-singular unassigned roots, including the prioritized
cycle and intervention cases. Continuation follows both parameter directions,
retains unstable segments, and records candidate events and every termination.
Default and half-maximum-step runs each have a 500-accepted-step budget per
direction. Independent 41×41 searches check seeds, endpoints, and selected
classification transitions; independently written equations and automatic
differentiation check continued-point residuals and Jacobians.

An arclength path through an unstable branch is not an accessible trajectory.
Finite searches and branch traces retain `CompletenessNotCertified`.
Uncovered refinement points, differing search results, and candidate
bifurcations remain explicit limitations, not silently resolved identities.

Replay instructions and verification results accompany
`reproducibility/equilibrium_audit_20261016/`.

### Verification results

- All 173 independent 41×41 root-search comparisons agreed, and every selected
  continuation root was rediscovered.
- Across 7,355 default/refined continuation-point records, the largest residual
  under independently written ODE equations was `6.373e-11`; the largest
  Jacobian discrepancy against automatic differentiation of those equations
  was `4.619e-14`.
- The initial comparison matched 142 of 173 checked points under step
  refinement. Two apparent disagreements near folds were resolved by retaining
  both endpoint seeds of each intersecting refined segment: the first
  correction had selected the neighboring root. All original checks and
  corrective attempts remain in the packet.
- The other 29 points lack a suitable matching refined segment. They remain
  unconfirmed under this comparison. Each headline named-branch connection has
  at least one separately searched, refinement-agreeing named context; the
  report does not promote the entire branch to uniform confirmation.
- For each continuation variant, 27 directions terminated at parameter bounds
  and nine at minimum step size. These are numerical stopping reasons, not
  identified physical endpoints or certified bifurcations. Recruitment was
  continued within the declared 12–19 envelope, including off-grid values.
- The full package suite passed 4,988 assertions. The subsequently added fold
  comparison regression passed seven focused assertions; the full suite was
  not rerun after adding that regression.
- The plotting suite passed 207 assertions. The portable inventory round-trips
  exactly to all 215,657 root records and 77,455 context records; its file set,
  content hashes, and recorded scientific-source hashes were verified.
- All five diagnostic figures were rendered and visually reviewed. Figures
  1–3 were reviewed again after correcting title/tick spacing; Figures 4–5
  were byte-identical to their reviewed previews. Figure 5 was also inspected
  as a reduced-size PDF rasterization. The final gallery is
  `output/equilibrium_audit_figures_20261016/index.html`.
- The original October 6 figure packet and rendered-figure manifests were
  reverified unchanged. Existing role assignments and manuscript figures
  were not edited.

The packet also retains all 860 recorded time-scale spectrum observations
with derived local roles, and pulse-destination translations at all sixteen
recorded cycle phases. Their counts remain separate from equilibrium-context
and independent-search counts.

## Decision summary

| Question | Current status | Next author decision |
| --- | --- | --- |
| What is the phase-control equilibrium? | Attracting upper/descending state; its missing name follows from the comparison-sink requirement. | Whether that requirement belongs in the intended seizure definition. |
| What are the recruitment destinations? | Characterized upper branches, connected numerically to named contexts; switching remains a separate measurement. | Which descriptive or branch-based labels to use in the intervention figure. |
| Why do maps have multiple unassigned roots? | Some are genuinely distinct candidates for the same name; other circuits fail the turning-point prerequisite. | Whether roles should permit multiplicity and cover circuits without E folds. |
| Are plotted intervention names all local classifications? | No: 15 of 84 are inherited coordinate matches, documented separately. | Whether to retain inherited names and how to disclose them. |
| Is every continuation segment equally confirmed? | No: 29 comparison points remain unmatched under the bounded refinement check. | No relabeling decision depends on treating those points as confirmed. |
