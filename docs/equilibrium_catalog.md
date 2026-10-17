# Compact equilibrium catalog

The catalog records equilibrium geometry, activity, and local stability
independently of biological names. It is a lossless projection of the relevant
fields from the frozen equilibrium audit, not a new numerical search.

The full current catalog contains **215,657 equilibria in 77,455 contexts**.
Its measured size is **11,267,520 bytes (11.27 MB, 10.75 MiB)**:

| Component | Compressed CSV files | Bytes |
| --- | ---: | ---: |
| Equilibrium descriptors | 22 | 6,706,399 |
| Context dictionary | 8 | 4,443,822 |
| Parameters, schema, index, checksums | — | 117,299 |
| Total | 30 | 11,267,520 |

This is 45.84% of the old compressed root inventory's 24,579,554 bytes. The
size gate includes all catalog files and requires the total to remain below
half that source inventory. At the current context/root ratio, the measured
size extrapolates to approximately 52 MB per million roots; this is an estimate,
not a guarantee for different numerical data or compression patterns.

All generated catalog and example files are under ignored `output/` directories.
Only the implementation, tests, and documentation enter version control. Frozen
audit, sensitivity, and manuscript-figure packets remain unchanged.

## Root record: schema version 1

| Field | Decoded type | Meaning |
| --- | --- | --- |
| `context_id` | UInt32 | Reference to the context dictionary |
| `root_id` | UInt16 | Original root index within that context |
| `E_arm` | UInt8 | Existing E-nullcline arm convention |
| `I_branch` | UInt8 | Inhibitory response branch |
| `E`, `I` | Float64 | Original equilibrium occupancies |
| `lambda_max_real` | Float64 | Largest eigenvalue real part, in ms⁻¹ |
| `lambda_min_real` | Float64 | Other eigenvalue real part, in ms⁻¹ |
| `omega` | Float64 | Nonnegative absolute imaginary part, in ms⁻¹ |
| `stability` | UInt8 | Original numerical stability classification |
| `near_singular` | Bool | Original balance-Jacobian near-singularity flag |

The transport is ordinary CSV compressed with deterministic gzip. Integer
codes reduce repeated text; the reader materializes the listed types. Numeric
CSV values retain Float64 precision. Display formatting does not change storage.

Codebooks are stored once in `catalog.toml` and checked by the reader:

| Quantity | Codes |
| --- | --- |
| E arm | 0: no E turning points; 1: lower; 2: middle; 3: upper; 4: turning point; 255: unresolved |
| I response branch | 0: unresolved; 1: ascending; 2: descending; 3: peak |
| Stability | 0: StabilityUnresolved; 1: Attracting; 2: Saddle; 3: Repelling |

The I code is explicitly a **response-function branch**, not a claim to number
every fold of the inhibitory nullcline. E code 0 inherits the existing audit's
`no_turning_points` convention; it does not imply global equilibrium uniqueness.
Source numerical conventions remain in force; the catalog does not infer new
turning-point tolerances or reinterpret unresolved classifications.

### Spectrum and uncertainty

For two real eigenvalues, `omega == 0` and the eigenvalues are the two stored
real parts. For a complex conjugate pair, both stored real parts agree and the
eigenvalues are `lambda_max_real ± im * omega`. This retains the full spectrum
without duplicating the conjugate imaginary component. The encoder rejects
nonfinite values and nonconjugate complex pairs rather than silently discarding
information. Eigenvectors and Jacobians are not stored in this compact record.

A largest real part alone could not distinguish a saddle from a repeller;
the second real part is therefore retained. The source stability and
near-singular flags are kept separately instead of recomputed from a displayed
or rounded eigenvalue. In particular, a numerically unresolved equilibrium is
not promoted to stable by the reader. Imaginary zero is represented canonically
as +0.0; every imaginary zero in the current source already has that form.

## Contexts and identity

Each context appears once, with its original context string, scope, case ID,
E/I inputs, optional parameter-axis override, root count, source location, and
source SHA-256. The case parameter registry is stored once in `parameters.toml`.
Effective parameters are the case parameters plus the explicit context override;
E/I inputs are returned separately. The current detailed-case numeric parameters
agree with the source registry, and the audit's intervention contexts retain
their changed parameter and value explicitly.

Context IDs are assigned by lexicographically sorting original context strings.
The root ID is the existing within-context root index. These are catalog-local
identifiers, not continuation identities. An external annotation should retain
the catalog manifest hash together with `(context_id, root_id)`; do not join
unrelated runs by numeric IDs alone.

Biological labels and candidate-cutoff memberships belong in separate derived
tables keyed by this identity. Coordinates and spectra need not be repeated
for every naming rule. Existing source labels can be recovered from the frozen
audit through the context string and original root index, but are not columns
of this catalog.

## Build, verify, and render

Run from the working-copy root with the recorded Julia 1.10 environments:

```sh
julia --project=. scripts/build_equilibrium_catalog.jl
julia --project=. scripts/verify_equilibrium_catalog.jl output/equilibrium_catalog_20261016 reproducibility/equilibrium_audit_20261016/data
julia --project=plotting scripts/render_equilibrium_catalog.jl output/equilibrium_catalog_20261016 output/equilibrium_catalog_examples_20261016
```

The default builder targets the frozen October 16 audit and writes
`output/equilibrium_catalog_20261016`. To replay into a new directory:

```sh
julia --project=. scripts/build_equilibrium_catalog.jl reproducibility/equilibrium_audit_20261016/data output/equilibrium_catalog_replay
```

Builders and renderers refuse existing destinations. The builder validates the
source manifest, parameter-registry digest, row counts, and context identities.
The output has an exact-file-set SHA-256 manifest, versioned schema/codebooks,
chunk index, source identity, and recorded implementation hashes. The reader
validates the manifest, schema, codebooks, parameter digest, duplicate roots,
dangling references, and per-context counts before returning a catalog. It does not need
the original audit files to read a completed catalog. Source location strings
are provenance and are never opened by the reader.

## Reader interface

These are script-layer interfaces, not changes to the model package API:

```julia
include("scripts/equilibrium_catalog.jl")
using .EquilibriumCatalog

catalog = EquilibriumCatalog.read_catalog("output/equilibrium_catalog_20261016")
EquilibriumCatalog.foreach_descriptor(catalog) do d
    eigenvalues = EquilibriumCatalog.spectrum(d)
    readable = EquilibriumCatalog.expand(d)
    ctx = EquilibriumCatalog.context(catalog, d)
    inputs_and_parameters = EquilibriumCatalog.context_parameters(catalog, d)
    # Derive proposed names here, without modifying the catalog.
end
```

`foreach_descriptor` loads one compressed root chunk at a time. The initial
reader validation retains context records and a root-identity set in memory;
it is not a promise of constant-memory validation at arbitrary scale.
`descriptor_text(d)` provides a short display such as
`E2/I1 | E=0.01005, I=0.005118 | Attracting; alpha=-0.02459 ms^-1`.
The display is abbreviated; use `spectrum(d)` to retrieve both eigenvalues.

## Evidence and review

The independent source comparison checks the expanded values directly against
every audit row, without invoking the encoder. It passed for all 215,657 roots
and 77,455 contexts, including exact Float64 coordinates and eigenvalues,
geometry codes, stability, near-singularity flags, and source identities.
No labels, formulas, trajectories, or numerical stability decisions changed.

The focused suite covers real and conjugate spectra, saddles and repellers,
unresolved/near-singular states, response peaks, turning points, nonfolded and
unresolved geometry, signed-zero coordinates, explicit parameter overrides,
invalid spectra/codes, duplicate/dangling records, unsupported schemas, and
manifest tampering. The first test run exposed a vector-versus-tuple comparison
in the CSV-column validator; converting the column-name container to a tuple
fixed that check without relaxing the required field order.

The example gallery is `output/equilibrium_catalog_examples_20261016/index.html`.
Its table is a representation check, not a new scientific figure or a claim
about the frequency of the illustrated equilibrium types.

The full package suite passed all 5,074 assertions in 9m10.0s, including the
48 catalog assertions, which also passed separately. The example PNG and reduced-size PDF raster
were visually inspected; the SVG parsed successfully. Catalog/example manifests
and current source hashes were checked. The original audit, sensitivity packet,
October 6 figure packet, and manuscript-figure output manifests verified
unchanged. The generated catalog and example paths were confirmed Git-ignored.
