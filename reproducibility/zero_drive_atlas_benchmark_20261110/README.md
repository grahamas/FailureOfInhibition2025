# Zero-drive atlas benchmark

This packet records the bounded benchmark of source commit
`0fb69fa41cc0d1f78c8c5379d18cab38c68f1469` using Julia 1.10.12.
It completed 200 jobs and stopped at the configured job limit, with 331
queued jobs pending. Setup took 19.843 seconds and measured execution took
42.180 seconds. All six available job kinds were exercised.

The run saved 325 parameter-point records, 20 examined paths, and 233
provisional components of the sampled connection graph. These are not 233
proven distinct parameter regions: 231 components are singletons. The 281
base-case role rows refer to 239 unique points; 48 rows describe local
neighborhood witnesses. Equal signatures alone do not establish connectivity.

The saved BT tuple and its coefficients, all saved roots, and the local event
curves were independently checked. The section contains 24 fold candidates,
12 Hopf candidates, and 12 trace-zero points without a resolved positive
determinant. Periodic work in this benchmark is a synthetic solver fixture;
the model's queued periodic follow-up was not executed. Certification applies
to one fixed parameter tuple. The atlas remains `CompletenessNotCertified`.

## Costs and scope

The discovery median was 0.0214 seconds and p95 was 0.0425 seconds across
195 jobs. Each other kind has only one observation: certification 3.63 seconds,
continuation 2.21 seconds, nexus investigation 11.54 seconds, periodic fixture
5.76 seconds, and section investigation 2.24 seconds. These include cold-handler
compilation. The rates in `timings.csv` are arithmetic extrapolations, not
measured sustained throughput or estimates of difficult parameter neighborhoods.

The benchmark first attempts each available kind before weighted scheduling.
Its job limit was reached well before the 20-minute measured-time cap. It
does not determine a sufficient pilot budget, nor establish absence of missing
regions, nexuses, equilibria, or attractors. The larger pilot awaits the
author's separate budget choice.

## Verify and rebuild the report

`evidence.tar.gz` contains the exact configuration, checkpoint, 200 raw result
files, and 27 frozen source files. Worker transport duplicates and the generated
HTML are omitted. The checkpoint retains source, configuration, payload, and
result hashes; `generated_outputs.sha256` records the accepted rendered outputs.
Independent review verified these hashes and reconstructed the component and
timing summaries.

From this directory, on Linux with Julia 1.10.12:

```sh
sha256sum -c checksums.sha256
atlas_replay_dir=$(mktemp -d)
tar -xzf evidence.tar.gz -C "$atlas_replay_dir"
cd "$atlas_replay_dir/benchmark"
julia --startup-file=no --project=source source/scripts/zero_drive_atlas.jl report .
sha256sum -c generated_outputs.sha256
```

Open the resulting `index.html`. Report generation reads saved evidence and
performs no numerical search. Running new scientific jobs requires instantiating
the frozen project and an explicit computational budget; use the
[atlas workflow](../../docs/zero_drive_atlas.md) for the commands and limits.
