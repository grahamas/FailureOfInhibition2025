# Two-input region selection audit, 5 October 2026

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

The comparison reuses archived samples and distances; it is not a fresh
numerical run under the corrected source. The frozen output and its checksums
remain unchanged. New source hashes require a separate output directory for
any future rerun, and scientific interpretation of the revised selection
remains pending. The local, untracked audit script and per-case JSON are in
`retired_tools/region_selection_audit_20261005/` outside the Jujutsu working
copies. The source archive's `checksums.toml` had SHA-256
`b01a2a09ab651de6a729381e83a9f4e2549c9d2e184bbb33465a7abd4f3408be`
when this comparison was made.
