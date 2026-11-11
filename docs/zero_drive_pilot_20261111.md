# Zero-drive pilot: 11 November 2026

The finite pilot supports continuing the atlas effort, but does not yet justify
treating its 67 candidate base signatures as a complete set of connected
regions. Numerical inventories were repeatable on the audited subset, and
tracked paths recovered 60 of 64 assessed held-out targets. All 64 targets had
one observed equilibrium; recovery of the three multiroot held-out cases was
not assessed. Storage was not the limiting resource.

## Protocol and execution

The [frozen protocol](zero_drive_atlas.md) was delivered in
[PR #158](https://github.com/grahamas/FailureOfInhibition2025/pull/158), merged as
`1006a59a5d4a21db418024cbc9912e1047af35c0`. The run used reviewed source commit
`4c9cb29b616734412b64d05770936e31dfc98202` and retained its executable source
snapshot, configuration, payloads, and result hashes.

All 947 mandatory jobs and 12 available training followups ran. The other 12
followup slots were optional capacity, not missing mandatory work.

| Role | Assigned | Completed | Failed | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Model controls | 3 | 3 | 0 | 0 |
| Training points | 512 | 510 | 2 | 0 |
| Training followups | 12 | 11 | 0 | 1 |
| Paired numerical audits | 112 | 112 | 0 | 0 |
| Held-out points | 256 | 256 | 0 | 0 |
| Recovery paths | 64 | 60 | 0 | 4 |

`protocol_complete` means the finite schedule finished; it does not erase the
failed or unresolved results. Setup took 145.004 seconds and execution took
495.289 seconds. The retained run occupied 50,237,440 bytes before independent
review files, under the author's 5,000,000,000-byte allowance. Report storage
figures are explicitly measured before report regeneration; observed peaks are
sampled measurements, distinct from reserved capacity and enforced write limits.

## What the evidence supports

All 112 saved-training repeats and 224 grid/tolerance comparisons agreed on
the saved numerical inventories. Of the audited points, 111 had one observed
root and one had three. This supports repeatability on that subset; the two
training failures remain recorded singular-matrix solver failures.

Increasing continuation depth from 12 to 24 added 308 points on 40 of the 224
directional branches, without new candidate events or stability changes.
However, 14 branches still stopped at the step limit. Shared-prefix agreement
therefore does not establish that continuation depth is sufficient.

The frozen base pool contains 67 sampled structural signatures. Ten of the 256
held-out points introduced nine signatures absent from the training pool. These
are finite signature differences, not nine proven new connected regions.

Recovery was assessed for 64 of the 256 held-out points. Sixty targets were
recovered by corrected-root tracking; independent endpoint inventories were
obtained for all 64. Only 11 paths both recovered the target and preserved the
sampled signature. The other 49 successful paths changed signature. Coupling
zero-mask changes occur on 42 of the 64 paths, so signature changes must not be
counted directly as bifurcations.

Three paths changed sampled root count or stability. In particular,
`job-000946` observed a 3 → 5 → 3 → 1 root sequence and recovered its target.
`job-000953` observed 1 → 3 → 1 and remained unresolved; `job-000954` observed
3 → 1 and recovered its target. These sampled inventories identify places for
further investigation, not established continuous bifurcation sequences.

All 64 assessed targets had one observed root. The three held-out points with
three observed roots were among the 192 targets without recovery checks. The
60/64 result cannot be extrapolated to multiroot target recovery or to the full
parameter space.

## Remaining questions and recommended allocation

The four unresolved recoveries retain specific tracking evidence:

| Recovery job | Held-out target | Unresolved evidence |
| --- | --- | --- |
| `job-000929` | `job-000769` | Large corrected step at sample 13 |
| `job-000941` | `job-000821` | Forward correction unresolved at sample 4 |
| `job-000953` | `job-000865` | Forward correction unresolved at sample 15 |
| `job-000959` | `job-000895` | Large corrected steps at samples 2 and 3 |

The next scientific allocation should address these paths, multiroot targets,
the step-limited continuation branches, and the two solver failures before a
larger uniform expansion. This is a recommendation for author review, not an
automatic follow-on campaign. The unresolved cusp followup also remains
unvalidated. The atlas retains `CompletenessNotCertified`: neither missing
regions/nexuses nor attractor completeness have been excluded.

## Retained evidence

The local run is at
`local_audits/zero_drive_pilot_20261111/` under the project container. Its
`index.html` links the evidence tables; `pilot_summary.toml` records the decision
flags and denominators; `state.toml` indexes all 959 retained results. The raw
packet is retained locally rather than duplicated in the repository.

| Artifact | SHA-256 |
| --- | --- |
| `state.toml` | `9eed3cad7b6e3caeaa979078a35fb684e2bfc3d8d4063025b8b0e3ed31edafdd` |
| `pilot_summary.toml` | `bfb6393a1897d5fe06327a6ecab407b80cc6b0f93333a7e02d6ada7e40fc8aa3` |
| `index.html` | `f6ce24a0030779d709640bda776f0395a6551ea9a736549673be1b2fde751e8f` |

Before launch, the fresh root suite passed 7,314 tests and plotting passed 19;
PR review and CI were clean. Independent post-run review verified all 29 source
hashes, 959 payload/result/request bindings, and the recorded configuration.
All 3,087 saved root records passed independent physical-domain and balance
checks. No pilot worker remained, and run-owned runtime directories contained
no cache files. Rendered report disclosures, navigation, recovery-table
scrolling, and the summary download passed interaction checks.

The scientific packet occupies 50,237,440 bytes. A final integrity-review
snapshot including presentation-review artifacts measured 54,378,554 bytes,
about 1.09% of the 5 GB allowance. These artifact checks are distinct from the
implementation tests and from any completeness claim.
