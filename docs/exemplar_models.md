# Provisional mathematical exemplar models

The four cases in [`experiments/exemplar_models.toml`](../experiments/exemplar_models.toml)
are proposed reference configurations for studying basin geometry and finite
drive protocols. Their identifiers describe equilibrium structure, not
biological states. They do not replace package defaults or approve any
manuscript claim. All use the same zero-drive failure-of-inhibition response,
`tau_E = 7.8 ms`, slopes `a_E = a_I = 5`, and thresholds
`theta_E = 1.5`, `theta_on = 4`.

| Identifier | `(e_to_e, i_to_e, e_to_i, i_to_i)` | `theta_off` | `r = tau_I/tau_E` | Certified equilibria | Locally attracting equilibria | Distinguishing feature |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `two_sinks_ascending` | `(17, 9, 12, 4)` | 6 | 4.4 | 3 | 2 | A low-activity sink and a high-E/high-I sink on the ascending response branch; no high-E/low-I equilibrium in the complete cover. |
| `three_sinks_descending` | `(17, 9, 19, 4)` | 8 | 4.4 | 7 | 3 | Both high-E sinks have a negative inhibitory response derivative; the intermediate root repels. |
| `four_sinks_central` | `(17, 9, 19, 4)` | 8 | 0.2 | 7 | **4** | The same seven equilibrium coordinates as `three_sinks_descending`, but the intermediate root near `(0.311, 0.425)` becomes attracting. |
| `three_sinks_ascending` | `(19, 13, 14, 6)` | 6 | 4.4 | 5 | 3 | The high-E/high-I sink has a positive inhibitory response derivative; no intermediate-E root appears in the complete equilibrium cover. |

`three_sinks_descending` and `four_sinks_central` isolate a time-constant
effect: changing only `r` changes local stability without moving the
equilibria. `two_sinks_ascending` is a smaller equilibrium set with no
high-E/low-I root. `three_sinks_ascending` has a high-E/low-I sink as well
as a high-E/high-I sink on the ascending inhibitory response
branch. This set spans two, three, and four locally attracting equilibria,
and both signs of the inhibitory slope at a high-E/high-I sink. The
historical Figure 3 and Figure 4 names are retained only as provenance in
the configuration.

Evaluating `u_I = e_to_i*E - i_to_i*I` over the certified high-E/high-I root
enclosures gives `[4.24714, 4.25358]` for `two_sinks_ascending`, below its
response midpoint `5`; `[7.70886, 7.70937]` for `three_sinks_descending` and
`four_sinks_central`, above their response midpoint `6`, and
`[4.30002, 4.30117]` for `three_sinks_ascending`, below its midpoint `5`.
The newly attracting intermediate root in `four_sinks_central` has
`u_I` in `[4.20806, 4.20938]`, on the ascending side of its response.

## Evidence and limits

The interval driver completed the physical `[E,I] ∈ [0,1]²` equilibrium
cover for all four cases with zero unresolved root boxes. The local
classification uses certified Jacobian trace and determinant signs. Local
artifacts under `output/` retain the exact configuration, root enclosures,
excluded boxes, source snapshot, replay command, and SHA-256 checksums:

- `output/exemplar_cert_three_sinks_descending_20260928/`
- `output/exemplar_cert_two_sinks_ascending_20260928/`
- `output/exemplar_cert_four_sinks_central_20260928/`
- `output/exemplar_cert_three_sinks_ascending_20260928/`

The descending three-sink and ascending three-sink root CSVs are
byte-identical to the earlier Figure 3 and Figure 4 rising-case
certificates. The four-sink case also agrees with the
independent 11-by-11, 21-by-21, and 41-by-41 finite searches reported in
[`low_ratio_coexistence_results.md`](low_ratio_coexistence_results.md).
The numerical study found four matched locally attracting roots at low
ratio; the interval result adds complete equilibrium coverage for this fixed
parameter cell. All files in each new artifact passed checksum validation;
an archived-source replay of the four-sink case reproduced its root and
excluded-box CSVs byte-for-byte.

**None of these is an exact count of all attractors.** The global
cycle-exclusion gate did not pass in any case, so each certificate records
`not_certified` for that count. Basin areas and directional stability
thresholds have not been measured at full resolution for these exemplars.

**Execution update (2026-09-28).** The adaptive finite-drive scan completed
for all four exemplars and their in-bounds one-at-a-time parameter neighbors.
Its retained local archive is `output/adaptive_rescue_exemplars_20260928/`;
the E-withdrawal analysis is
`output/tonic_e_withdrawal_findings_2026-09-28.html`. No sampled trial from
the tracked high-E/low-I source reached a configured rescue target. This is
a finite-window observation, not an absence claim. The separate uniform
two-case tonic scan has no retained full-run artifact.

In particular, an ascending inhibitory response at the
`three_sinks_ascending` high-E/high-I sink does not establish
that it is rescuable. The positive-increment obstruction from
[`theory_notes.md`](theory_notes.md) applies to the two high-E sinks of the
`three_sinks_descending` and `four_sinks_central` cases under its stated
assumptions; it does not address tonic E withdrawal.

To rerun a fixed-case certificate, select an identifier and a new output
directory, for example:

```bash
julia --project=certification certification/certify_attractors.jl \
  --config experiments/exemplar_models.toml \
  --case four_sinks_central \
  --output output/four_sinks_central_recheck
```

The original basin and tonic runner still uses its two-case configuration
and three coordinate roles. The separate
[adaptive exemplar drive runner](basin_rescue.md#adaptive-exemplar-drive-study)
tracks all four attracting equilibria for finite-drive comparisons. It does
not measure basin geometry, so a complete basin comparison of
`four_sinks_central` remains open. This catalogue supplies parameter cells
and local-equilibrium evidence, not an intervention outcome by itself.
