# Targeted second zero-drive pilot

This protocol investigates specific gaps from the [first pilot](zero_drive_pilot_20261111.md).
It is an outcome-selected diagnostic study, not a new held-out validation.
Its deliverable is a local adjacency map backed by saved parameter points,
tracked roots, and investigated transition brackets. Neither equal signatures
nor successful endpoint recovery establish a complete connected region.

## Frozen targets and execution

Freeze the source packet inventory, selected payloads, numerical policies, and
their hashes before running any second-pilot jobs. The first packet remains
unchanged. All IDs below refer to that packet; new execution IDs are separate.

| Jobs | Purpose | Source IDs |
| ---: | --- | --- |
| 3 | Original model readiness controls | BT, cusp, Hopf-to-periodic controls |
| 2 | Reconstruct failed inventories and diagnose solver exceptions | 414, 509 |
| 4 | Refine the original base-to-target routes | 929, 941, 953, 959 |
| 3 | Test multiroot target recovery | Held-out points 872, 876, 877 |
| 3 | Investigate observed inventory/stability-change brackets | Paths 946, 953, 954 |
| 7 | Extend both continuation directions | Audits 584, 585, 598, 608, 613, 627, 637 |

There are exactly 22 scheduled jobs, including 14 continuation directions in
seven paired jobs. Attempt all three readiness controls as a batch; any failed
or unresolved control blocks dispatch of the 19 targeted investigations. Jobs generate no
automatic discovery or recursive followups. Unresolved results and exhausted
limits are retained outcomes, not reasons to rerun until success.

Preserve the four original recovery routes. For the three multiroot targets,
choose the nearest base using the first pilot's parameter-only distance and
frozen 67-base pool, before second-pilot outcomes are available. Target-origin
reverse tracking is a distinct diagnostic and cannot count as base-origin
recovery. No alternative detour route or certification campaign is included.

## Numerical policies

Use the tracked Julia environment and original model conventions. Independently
validate accepted equilibria with balance tolerance `1e-11` and the physical
domain. Preserve matching/diagnostic thresholds: duplicate roots `1e-7`,
endpoint and reverse-return matching `1e-6`, large corrected step `0.1`.
These are declared numerical policies, not mathematical constants.

For the two failed inventories, run grids of 5, 9, and 13 points per state
coordinate, preserving the known equilibrium-chart seed for case 509. Retain
all seed outcomes. A singular linear solve is an explicitly rejected attempt;
it must not discard successful attempts or imply a singular equilibrium.
Record applicable analytic decoupling/uniqueness checks separately.

For each recovery investigation, run nested uniform grids of 33, 65, and 129
values of the affine path coordinate `t` from zero to one. Retain each level's
forward base-origin and reverse target-origin attempts. Assess coverage from
the finest level without deleting earlier diagnostics. Obtain grid-9 point
inventories; cross-check the three multiroot target inventories with grid 13.
If track-derived seeds augment an inventory, identify that dependency rather
than calling the result independent. Report the observed endpoint-root union
and keep unmatched or ambiguous branches visible.

Recovery 959 is uncoupled throughout its path. Its explicit unique attracting
equilibrium supplies an analytic cross-check of the numerical tracking.
Triangular structure on path 941 supplies separate scalar fold-window
diagnostics; these do not replace investigation of the original straight path.
For an exactly triangular `B=D=0` path, scalar turn thresholds may be checked
on the original 16 intervals where both endpoint values of `A` exceed eight.
Retain endpoint zeros explicitly; bisect a sign-changing threshold equation
for at most 64 iterations, stopping at residual or affine width `1e-12`.
These scalar-window diagnostics are separately labeled, not certified event
or branch-identity evidence.

## Transition and continuation limits

Investigate all six original change brackets: path 946 samples 4–5, 5–6, and
13–14; path 953 samples 13–14 and 14–15; path 954 samples 5–6. Refine the
widest remaining observed-change interval first, breaking ties by left `t`.
Keep both changed children when a midpoint reveals two changes. Allow at most
12 total subdivisions per original bracket, stopping at affine width `1e-4`
or a resource limit. This is not a depth-12 binary tree. Record corresponding
parameter spans. Equal endpoint signatures do not exclude hidden paired
crossings; same-signature child intervals retain that limitation.

Use grid-9 inventories and grid-13 cross-checks at the sides of every terminal
claimed change bracket. Disagreement or unresolved classification makes the
connection unknown. Each investigation job has a hard maximum of 256 distinct
inventory locations; repeated evaluation at the same location remains
identifiable. Recovery and localization of path 953 are separate jobs with
separate counters, not one shared mutable cache.

Event correction must evaluate the exact affine parameter tuple `p(t)` and
keep `t` inside its observed bracket. Independently recheck saved-tuple balance
and event conditions, using event tolerance `1e-9`. Permit at most four seeds
per condition per terminal bracket and 64 corrector attempts per path. A
determinant-zero solution alone is a fold candidate. A numerically validated
generic fold additionally needs a simple zero eigenvalue and resolved,
step-refined quadratic and path-transversality coefficients. Trace-zero
solutions remain candidates unless the additional required conditions are
actually checked. No such local event establishes a nexus or global coverage.

For reproducibility, use derivative steps `1e-3` and `5e-4`, coefficient
magnitude floor `1e-5`, relative refinement tolerance `0.02`, and matching
coefficient signs. Require `abs(trace(J)) > 1e-7`, smallest singular value
`<= 1e-7`, largest singular value `> 1e-7`, and normalized null-vector
residuals `<= 1e-7` for the numerical simple-zero check. Save the quadratic
coefficient and affine-path transversality at both steps. Central path
derivative probes must stay in `0 <= t <= 1`; otherwise classification remains
unresolved. Corrector attempts use at most 16 iterations with the existing
Newton/line-search policy. Any separate correction-refinement attempt counts
toward the same 64-attempt cap. These tests establish numerical local evidence,
not certified nondegeneracy.
Repeated successful estimates may be grouped provisionally by condition kind,
complete-linkage affine distance `<= 1e-6`, and state infinity-norm distance
`<= 1e-6`. Retain every raw estimate and correction attempt, and use an actual
saved estimate as representative rather than averaging. Such grouping does
not prove event identity or exclude unresolved nearby events.

For continuation, preserve each audit's original first-root seed, `A` bounds,
scales, tight tolerances, and other options. Increase the maximum from 24 to
96 steps in both directions. Compare the stored prefix in both state and `A`
with tolerance `1e-10`; retain diagnostic disagreements. Tangents were not
saved in the first packet and cannot be compared. Termination may change;
reaching 96 steps remains an unresolved frontier rather than saturation.

## Adjacency semantics and storage

Nodes represent saved parameter-point inventories and separate root/event
witnesses. Edges distinguish root tracking, sampled inventory preservation,
observed change brackets, and event-candidate links with retained side
inventories. Equal signatures alone create no edge. Missing or failed
intermediate evidence means unknown, not disconnected. Display compression
must preserve every underlying sample ID and status, including excursions
such as 3 → 5 → 3 → 1.
A sampled inventory-preserving link requires every declared intermediate
inventory to be present, nonempty, resolved, and in agreement; a surviving
root track or matching endpoints alone does not satisfy that condition.

The combined first- and second-pilot owned artifacts must remain below
5,000,000,000 bytes, including reports, temporary files, and run-owned caches.
Measure and reserve the immutable first packet's footprint before allocating
the second output directory. Inventory symlink metadata without following it
and stale browser-socket metadata without opening the socket;
executable input references must resolve to verified regular files. Reuse
strict worker isolation, bounded transport, and reserved checkpoint/report
writes. Verify source-packet and manifest hashes on creation, resume, and
report generation.

Operational defaults are 600 seconds for worker setup, 900 seconds per job,
and a six-hour coordinator guard; the finite protocol defines scientific
completion. Code and protocol receive independent review and relevant checks
before launch. The final artifact receives scientific, integrity, and visual
review. Retain `CompletenessNotCertified` and publish unresolved limitations.
