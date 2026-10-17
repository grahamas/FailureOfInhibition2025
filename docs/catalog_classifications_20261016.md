# Catalog classifications and exploratory figures

## Recommended representation

Keep **activity, geometry, and local dynamics as separate descriptors**. Add a
biological endpoint proposal only when its explicit activity/geometry predicate
is satisfied. Use descriptive activity bands for other equilibria. Multiple
roots in one context can have the same description or candidate name.

This first comparison proposes numerical boundaries for inspection, not final
physiological definitions. The production classifier and manuscript figures
are unchanged. No new equilibrium search or trajectory simulation was needed.
All 215,657 catalog roots in 77,455 contexts have derived annotations, including
unstable roots; figures that describe attracting equilibria explicitly select
source `Attracting` status and exclude near-singular roots.

## Three complementary classifications

### 1. Proposed endpoint names

The reference policy uses low E/I = 0.01, high E = 0.45, and retained I = 0.30.
These are readable trial boundaries: 0.01 is 2% of the occupancy ceiling 0.5,
0.45 is 90%, and 0.30 is 60%. They are not fitted thresholds or physiological
firing-rate levels. In particular, retained I occupancy is not a measurement
of effective inhibition across different coupling strengths.

| Proposal | Predicate under the reference policy |
| --- | --- |
| Rest candidate | E ≤ 0.01 and I ≤ 0.01, on any E arm or I response branch |
| Seizure candidate | E > 0.01, I ≤ 0.01, descending I response |
| Herald candidate | E ≥ 0.45, I ≥ 0.30, without a response-side requirement |
| High-E, intermediate-I failure branch | E ≥ 0.45, 0.01 < I < 0.30, descending I response |
| Descriptive remainder | Retain activity bands, geometry, and spectrum |

The ordered boundaries make these predicates disjoint. Rest uses both quiet
populations. Seizure requires neither E saturation nor a higher-I companion;
herald requires no lower-I companion. E arm never gates an endpoint label.
A low-I ascending response is described as low I without assuming failure.
The fourth category is deliberately descriptive; "partially failed inhibition"
would require care about what activity and response slope do and do not measure.

**Do not equate the descriptive remainder with active.** It includes ordinary
intermediate activity, inhibition-dominated low-E states, and other geometry.
An operational active definition remains a distinct author decision. Nor does
an endpoint-like unstable equilibrium become an attractor: stability and
near-singularity remain separate fields in the catalog.

### 2. Exhaustive activity bands

For each population, low is at or below its low cutoff, high is at or above
its high cutoff, and intermediate is strictly between them. The Cartesian
product gives nine descriptions, such as:

- low E / low I;
- low E / high I (an inhibition-dominated activity pattern);
- intermediate E / intermediate I;
- high E / intermediate I;
- high E / low I.

Append the I response side and E arm when needed. This retains a complete
coordinate description without inventing nine biological regimes. The bands
still depend on cutoffs; they are not the threshold-free representation.

### 3. Partial geometry and dynamics

These need no new activity cutoff:

- E arm, particularly whether an attracting root is on the middle arm;
- ascending, peak, or descending I response;
- source stability and near-singularity status;
- real or complex spectrum, together with both real parts and |Im λ|.

For example, an attracting middle-E-arm root conveys inhibitory stabilization
under the existing model, while a complex attracting spectrum conveys damped
local oscillatory modes. Neither establishes a limit cycle, basin size, pulse
accessibility, or a finite perturbation response. Descending I response alone
cannot distinguish herald from seizure.

## Sensitivity comparison

Three declared policies test narrower and broader endpoint definitions:

| Policy | Low E | Low I | High E | Retained I |
| --- | ---: | ---: | ---: | ---: |
| Narrow endpoints | 0.005 | 0.001 | 0.475 | 0.45 |
| Reference | 0.01 | 0.01 | 0.45 | 0.30 |
| Broad endpoints | 0.02 | 0.02 | 0.40 | 0.20 |

The names describe the overall intent, not nested membership sets: changing
low E also changes the seizure floor. All four boundaries change together,
so between-policy differences cannot isolate a causal effect of one cutoff.
The earlier six-boundary sensitivity packet remains the broader comparison.

The screen and detailed geometry are separate, overlapping cohorts of
purposefully sampled contexts, not independent populations. Counts below are
attracting, nonsingular root **records**, not prevalence or distinct branches.

| Cohort / policy | Rest? | Seizure? | Herald? | High-E / intermediate-I failure | Descriptive |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screen / narrow | 13,316 | 34,133 | 8,477 | 2,823 | 15,925 |
| Screen / reference | 16,741 | 35,716 | 9,667 | 361 | 12,189 |
| Screen / broad | 17,089 | 35,854 | 9,733 | 134 | 11,864 |
| Detailed / narrow | 5,960 | 31,039 | 7,233 | 2,199 | 14,128 |
| Detailed / reference | 8,696 | 31,394 | 8,116 | 596 | 11,757 |
| Detailed / broad | 9,581 | 31,698 | 8,116 | 113 | 11,051 |

67,248 of 74,674 screen records (90.1%) and 54,722 of 60,559 detailed records
(90.4%) retain the same candidate code under all three policies. This includes
records remaining descriptive. It is agreement over these three choices,
not a confidence score or robustness over every possible boundary. A cautious
alternative is to display named endpoints only when all three proposals agree,
and otherwise retain the descriptive bands and a boundary-sensitive flag.

The main weakness is the intermediate-I endpoint description: its counts vary
strongly with the selected low-I and retained-I boundaries. Rest also varies
appreciably. The broad rest threshold admits some middle-arm roots; the
reference one does not. Keep that dynamical distinction visible rather than
choosing the rest threshold to enforce an E-arm restriction.

## Figure gallery and scientific purpose

Generated gallery: `output/catalog_classification_figures_20261016_review/index.html`.
Each figure is available as PNG, PDF, and SVG. These are exploratory comparisons,
not seven proposed main manuscript figures.

| Figure | What it contributes |
| --- | --- |
| 01 Activity and candidates | Raw 120 × 120 occupancy density next to proposed endpoints, separately for screen and detail cohorts |
| 02 Input candidate maps | Figure 2 style maps of discovered sink count and separate rest/seizure/herald presence; coexistence remains visible |
| 03 Reference raw maps | Minimum/maximum E, I, and spectral abscissa across all eligible roots at each input |
| 04 Weak recurrence raw maps | Same six measurements and identical scales for the second Figure 2 circuit |
| 05 Partial geometry and dynamics | Three overlapping input maps: middle E arm, descending I response, complex attracting spectrum |
| 06 Low activity and spectra | Low-activity zoom by E arm/I branch, plus both eigenvalue real parts for all roots, including unstable ones |
| 07 Policies and activity bands | Endpoint sensitivity and exact counts in the complete nine-band activity description |

Input maps use `detail_geometry` for `selective_withdrawal` and
`input_dependent`, the circuits in the current Figure 2. Every dot is a measured
context. The two circuits differ in multiple parameters; their contrast does
not isolate the effect of recurrent excitation. No interpolation or adaptive-cell
filling is used; white space is unmeasured. Gray in presence panels means no
matching eligible root found.
Discovery remains `CompletenessNotCertified`.

In raw maps, extrema of E, I, and spectral abscissa are computed independently.
They need not refer to the same root. They are repertoire envelopes, not a
trajectory, branch identity, or one selected equilibrium. There is at least
one eligible sink in every displayed context for these two circuits, so the
no-sink marker is not used in this particular render. The data format and
renderer still distinguish a missing sink from zero activity.

**Recommendation for the next Figure 2:** use the sink-count/input view with
one or two mechanistic layers from Figure 05; retain the raw activity comparison
as support for the chosen names. The full six-envelope maps are valuable for
inspection, but several panels are nearly uniform and do not each earn a main
manuscript panel. Do not replace the existing measured periodic-branch evidence
with the complex-eigenvalue map: they measure different dynamics.

## Reproduction and storage

```sh
julia --project=. scripts/catalog_classifications.jl output/equilibrium_catalog_20261016 output/catalog_classifications_20261016_v2
python3 scripts/verify_catalog_classifications.py output/equilibrium_catalog_20261016 output/catalog_classifications_20261016_v2
julia --project=plotting scripts/render_catalog_classifications.jl output/equilibrium_catalog_20261016 output/catalog_classifications_20261016_v2 output/catalog_classification_figures_20261016_review
```

Use new output directories when replaying. The compact catalog remains the
source of coordinates and spectra. `annotations.csv.gz` stores IDs, three
candidate codes, the reference activity-band code, and agreement. Context
summaries store counts and continuous extrema; they do not duplicate every
root's coordinates. Policy values, codebooks, source hashes, and cohort choices
are recorded in `study.toml`; derived and rendered packets have exact file-set
SHA-256 manifests. All generated data remain under ignored `output/` paths.

The derived packet is **3,647,460 bytes (3.65 MB)**. The original 11.27 MB
catalog is unchanged. The root annotation table itself is only 645,507 bytes;
most added storage is the per-context summary table. Numerical labels can be
regenerated from that catalog;
no new simulation or enlargement of the underlying root schema is required.

## Verification and review

The 65 focused assertions passed. They cover ordered/invalid policies, exact
boundaries, response peaks/unresolved branches, nonfolded seizure candidates,
middle-arm rest, companion-independent naming, stability separation, duplicate
labels within one context, and missing extrema for empty sink sets.

An independent Python standard-library verifier reproduced all 215,657 derived
annotations, all 77,455 context summaries and their continuous extrema, and
every cohort/policy/stability count directly from catalog records. It imports
no Julia classifier code. The renderer also checks every annotation against
its source policy and verifies catalog/derived-packet identity before plotting.

Two negative checks changed a candidate code and a continuous context extremum,
then recomputed the packet hashes. The independent verifier rejected both;
these checks exercise semantic validation beyond manifest integrity.

All seven final figures were visually inspected, including reduced-size PDF
rasters. All seven SVGs parsed. Review retained shared comparison scales,
explicit sampled points, and visible threshold/aggregation explanations; it
added distinct endpoint symbols and improved activity-table text contrast.
The gallery and image exports total 8,844,748 bytes. Superseded generated drafts
were removed after the final outputs verified. Catalog, audit, sensitivity,
original figure-data, and original rendered-figure manifests verified unchanged.

The full Julia package suite passed **5,139/5,139 assertions in 9m09.4s**,
including the final 65 classification assertions. The new CairoMakie renderer
executed successfully in the plotting environment; the separate existing
plotting test suite was not rerun. `git diff --check` passed. Generated packet
and gallery paths are ignored, and no dependencies or production model APIs
changed.
