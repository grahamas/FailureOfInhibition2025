# Bounded atlas completion protocol

This approved protocol produces an audited numerical atlas over shells 1–4,
with an explicit record of what was examined and what remains unresolved.
Completion means finishing this finite protocol and reviewing its artifacts.
It does not certify parameter-space coverage, absence of additional nexuses,
or completeness of equilibrium or attractor discovery. Every packet retains
`CompletenessNotCertified`.

Implementation status: the numerical calibration has passed independent
scientific review. The replacement pilot, three exploration rounds, and fresh
validation have not run. Executable delivery, launch, and final artifact review
remain separate acceptance gates.

## Dependencies and acceptance

| ID | Required outcome | Depends on |
| --- | --- | --- |
| A1 | Analytic state tensors, independent calibration and numerical error analysis | None |
| A2 | Independently reviewed frozen numerical policy and passing readiness controls | A1 |
| B0 | Reviewed campaign implementation, root and plotting checks, clean PR and merge | A2 |
| B1 | New replacement targeted packet, preserving both historical packets | B0 |
| B2 | Three prescribed exploration rounds, with explicit attempted/resolved status | B1 readiness |
| B3 | Frozen bases and independent forward validation | B2 |
| C1 | Reviewed scientific, integrity and presentation evidence; merged outcome documentation | B3 |

Unresolved numerical results remain evidence in the atlas. A failed readiness
gate, storage stop, or unattempted mandatory job is an incomplete protocol,
not scientific completion. Routine numerical repairs are authorized; changes
to the model, scientific scope, or storage ceiling require a new decision.

## Numerical readiness

Implement analytic second and third state derivatives for the supported
reduced logistic and difference-of-logistics model, preserving original-time
factors and existing normal-form conventions. Correction and classification
use consistent derivatives. Independent high-precision and finite-difference
refinement checks assess their accuracy; generic polynomial fixtures remain.

Calibration includes exact and perturbed cusp and BT controls, an actual-model
Hopf coefficient control, negative degeneracies, the saved failed cusp tuple, and the original 512 training
tuples. Report derivative error, correction residual, conditioning, and
classification uncertainty separately. Reassess balance, event, coefficient,
and matching thresholds separately from these errors and negative controls.
Do not increase thresholds merely to accept the cusp. Independent scientific
review has delegated authority to approve the numerical policy. Freeze its
metadata and evidence before launching the exact campaign controls.

## Replacement targeted protocol

Create a new packet for the original 22-job protocol: three readiness controls
followed by the original 19 investigations. Preserve target and route selection
and tracking, localization, and continuation limits. Only reviewed numerical
policy and implementation changes differ. Never resume the historical failed
packet using changed source.

## Three exploration rounds

Each round adds 512 points: four shells, two charts, 16 coupling masks, and four
accepted indices per stream. Freeze SHA-256-ranked selections and a protocol
seed/version before observing results. The accepted-index pools are 257–512,
513–768, and 769–1024 for rounds one, two, and three respectively.

For each round:

- Compute grid-9 inventories and analytic positive-timescale partitions for
  every discovered root. Audit one predetermined point per stream with grid 13
  (128 audits).
- Within each shell/chart group, take the first two eligible proposals for each
  of continuation, nexus, and periodic investigation, ordered by stable source
  ID and proposal order (at most 48 proposals).
- For each validated nexus, examine one bounded section and admissible positive
  and negative transverse probes along remaining geometry directions at
  normalized offsets 0.01 and 0.005. Record unexamined directions and unresolved
  sides explicitly. Followups are limited to these prescribed probes.
- Select four component-pair paths per shell/chart group: the nearest two
  same-signature and nearest two different-signature pairs, using the existing
  minimum distance between eligible observed endpoints in distinct components,
  stable-ID ties, and skipping previously examined endpoint pairs. Different
  endpoints define different straight paths even when their components are the
  same. Select at most one pair per component pair within a stage. Apply the
  existing 129-sample guards and bounded localization.

Report novelty and uncertainty by round descriptively. They are not
probabilities of missing regions. Charts have different projected domains;
coverage tables must preserve the chart distinction.

## Frozen bases and fresh validation

After round three, freeze one stable-ID representative per explicitly connected
sampled component. Equal structural signatures alone do not join components.
Retain event and transition witnesses separately.

Select 256 fresh validation points, two per stream, by SHA-256 ranking from
accepted indices 1025–1280. Freeze parameter-only routes to the representatives
before observing validation outcomes. Compute grid-9 and grid-13 inventories
and forward recovery for every point, including multiroot cases. Report forward
coverage, sampled structural preservation, transitions, and uncertainty
separately. Do not add validation cases to the frozen bases and then count them
as successful independent recovery.

## Storage, provenance and delivery

The combined ceiling is 5,000,000,000 bytes, including both historical packets,
new results, temporary files, reports, and run-owned caches. Reserve the
historical footprints and checkpoint/report capacity. Forecast remaining
storage from actual serialized evidence before each stage. A storage stop
leaves the protocol incomplete; it does not authorize deleting evidence or
silently shrinking the study.

The versioned manifest binds source, numerical policy, selections, stage rules,
and storage accounting. Preserve strict worker isolation and atomic
checkpoints. Store raw evidence once and reference it from reports.

Deliver a browsable typed evidence graph, frozen-base table and rationales,
shell/chart/mask coverage ledger, per-round novelty, and unresolved frontier.
Show planned, attempted, and resolved counts prominently. Independently review
scientific conclusions, integrity, and presentation. Merge executable changes
only after clean review and checks, then run the finite campaign; deliver
results and current planning documentation through the same review loop.

## Commands and review binding

From a checkout containing the reviewed implementation, create a new calibration
directory with the existing locked Julia environment:

```sh
julia --startup-file=no --project=. scripts/zero_drive_atlas.jl calibrate CALIBRATION \
  --source-packet FIRST_PILOT --prior-targeted-packet HISTORICAL_TARGETED \
  --storage-bytes 5000000000
```

The calibration reserves at most 256 MiB within the combined allowance and writes
its policy, input provenance, numerical source hashes, and evaluated evidence.
Passing calibration is necessary but insufficient: independent scientific review
must inspect the saved evidence and record a passing `review.toml` bound to the
policy and calibration inventory. The campaign gate checks that binding, the
numerical implementation, and the historical biased-cusp regression control.

After executable code is reviewed and merged, launch a new campaign directory:

```sh
julia --startup-file=no --project=. scripts/zero_drive_atlas.jl campaign CAMPAIGN \
  --source-packet FIRST_PILOT --prior-targeted-packet HISTORICAL_TARGETED \
  --numerical-policy CALIBRATION/policy.toml --storage-bytes 5000000000
```

The historical packets and calibration directory must remain immutable during
the campaign. Their allocated/logical footprints are reserved before creating
the new packet; strict worker isolation prevents worker writes to those inputs.
External mutation invalidates provenance and is outside the immutable-input
assumption. Reports can be regenerated with `report CAMPAIGN`. Operational time
guards can be resumed with `resume CAMPAIGN --budget-seconds SECONDS`, retaining
the cumulative finite job ceiling. A storage stop remains an incomplete
frontier and cannot be cleared by an ordinary resume.
