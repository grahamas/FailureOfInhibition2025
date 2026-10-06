# Two-input region selection audit, 5 October 2026

The six-case replay below checked the first graph correction. A subsequent
cell-consistency correction is described at the end of this note; the six-case
artifact does not certify selections under the current source.

The region graph previously joined equal-signature samples inside every
`boundary_bracket` cell. That cell status records mixed observed signatures, so
it cannot establish a homogeneous connection. The corrected selector joins
samples only within `sampled_homogeneous` cells; unresolved and mixed cells do
not supply region edges. A focused mixed-cell regression now checks this rule.

This read-only audit recomputed both selection rules from the retained
`output/input_response_20260929_v3/{anchors,expansion,representatives}/*/geometry`
tables. It matched the original `summary.toml` selections in all 306 cases.
The corrected graph partitions differ in 77 cases (8 anchors, no expansion
screens, and 69 detailed representatives). Eleven cases change their raw
component representatives. After retaining the explicitly configured
baselines, six detailed cases lose seven automatically selected inputs:

| Detailed case | Previously selected `(B_E, B_I)` no longer selected |
| --- | --- |
| `figure3_joint_37` | `(0.4296875, 0.00390625)` |
| `figure3_joint_40` | `(0.90625, 0.0625)` |
| `figure4_joint_40` | `(0.90625, 0.0546875)`, `(0.91015625, 0.0078125)` |
| `four_sinks_central_e_to_e_17.25` | `(0.015625, 0)` |
| `four_sinks_central_tau_ratio_0.4_midpoint` | `(0.0078125, 0)` |
| `selective_withdrawal_e_to_e_16.75` | `(0.425, 0.0078125)` |

No new input is selected by this comparison. The removed inputs remain
historical measurements, but two ratio rows in the retained study summary
(`four_sinks_central_e_to_e_17.25` and
`four_sinks_central_tau_ratio_0.4_midpoint`) came from these inputs. Any
aggregate that uses those rows still describes the old selection rule. The
check does not certify every generated figure or downstream summary.

The read-only comparison above reused archived samples and distances. A
separate geometry-only Julia replay then reran all six affected cases with the
first graph correction and their archived `parameters.toml` records. It reproduced
the seven removed inputs and selected no additions. In every case,
`inputs.csv`, `cells.csv`, `critical.csv`, `equilibria.csv`, `lineage.csv`, and
`bounds.toml` matched the frozen output byte for byte. That replay's
SHA-256 manifest verified all 73,459 local files. The
`selection_comparison.toml` SHA-256 is
`3d5440a00c7615c4ea4161cad47cad28a2e33f78a7a5381129b36b1960acd2e8`.
Its manifest has SHA-256
`cd8c8214a5ad27ff75fd3867e278886f194a1aa89606366fa4ef078097fe6bbb`.
The six-case replay lives outside the working copies at
`local_audits/region_selection_replay_final_20261005/`; its metadata identifies
the custom geometry-only script and does not advertise a full-study replay.
Response protocols were not rerun. The frozen full output and its checksums
remain unchanged. Scientific interpretation of the revised selection remains
pending. The local, untracked
read-only audit script and per-case JSON are in
`retired_tools/region_selection_audit_20261005/` outside the Jujutsu working
copies. The source archive's `checksums.toml` had SHA-256
`b01a2a09ab651de6a729381e83a9f4e2549c9d2e184bbb33465a7abd4f3408be`
when this comparison was made.

## Cell-consistency audit, 6 October 2026

The atlas originally classified a cell from five adaptive probes, while its
region graph considered every retained context inside the cell. A critical-set
offset sampled before the atlas, or a point sampled later in a neighboring
cell, could therefore contradict a `sampled_homogeneous` label. The current
source refines around known offsets and checks final cell labels against all
observed points before building region edges. A cell with conflicting observed
signatures cannot establish a homogeneous connection.

A read-only pass over the 306 archived geometries found 98 such cells in 69
cases. In 39 cells across 34 cases, a pre-atlas sample supplied the conflicting
signature. If only these archived labels are conservatively changed to mixed
and the component rule above is applied, 28 cases change their effective
representative sets: 30 old inputs are removed and 14 are added across one
anchor, three expansion screens, and 24 detailed cases. This is a bounded
comparison of retained observations, not the outcome of a fresh 306-case
study. New refinement can add observations and change that count. The
read-only scripts and per-case JSON are retained outside JJ at
`retired_tools/region_selection_audit_20261005/`. Response protocols and
downstream summaries have not been rerun under this correction; the frozen
study remains historical and revised scientific interpretation is pending.

A fresh geometry-only replay after the cell-consistency correction checked
one case from each affected category:

| Case | Removed selected inputs | Added selected inputs |
| --- | --- | --- |
| Anchor `two_sinks_ascending` | `(0.25, 7)`, `(4, 5.5)`, `(4.193612787581321, 5.570574093348656)` | `(0.203125, 7.5)`, `(0.203125, 7.5625)`, `(0.25, 2)` |
| Screen `figure3_joint_32` | `(0.75, 12)` | `(1, 12)` |
| Detailed `figure3_joint_40` | `(0.90625, 0.0625)` | None |

The exact records are also in the local `selection_comparison.toml` at
`local_audits/known_sample_replay_20261006/`. In each case, `bounds.toml` and
`critical.csv` matched the frozen archive byte for byte; the sampled tables
changed. An independent pass found zero homogeneous cells contradicted by
retained points in these three outputs and verified all 25,090 SHA-256 manifest
entries. The manifest SHA-256 is
`54fae453e99de4e410829e57e1ff1f8b18b578022b386f93c6a0f959a4b0aad2`.
The artifact includes its custom replay script and explicitly marks full-study
replay incomplete. These three cases do not determine selections for the other
303 cases.
