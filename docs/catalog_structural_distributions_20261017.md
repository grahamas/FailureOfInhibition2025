# Occupancy distributions within structural categories

## Scientific framing

Ask **what E and I activities occur within an E arm**, then condition further
on the inhibitory response side. These are catalog descriptors, so this view
requires no proposed rest, active, herald, or seizure thresholds.

The strongest descriptive result is that **E-arm identity separates E activity
much more clearly than it separates I activity**. Both lower and upper E arms
contain very low and very high I. Splitting by inhibitory response side helps,
but descending response still contains qualitatively different I occupancies.

All numbers below describe **attracting, nonsingular root records** unless
explicitly stated otherwise. Each record has equal weight within its structural
group. Adaptive input sampling, repeated circuit contexts, and overlapping
screen/detail cohorts preclude interpreting these fractions as physiological
prevalence, basin size, or parameter-space volume. The two cohorts are not pooled.

## 1. What does an E arm tell us about activity?

The screen cohort contains 74,674 eligible records. Values below are medians
with the central 90% interval (5th–95th observed-value percentiles), rounded
for reading. Occupancy runs from 0 to 0.5. These intervals summarize variation
in sampled records; they are not confidence intervals.

| E geometry | n | E median [central 90%] | I median [central 90%] |
| --- | ---: | --- | --- |
| Lower arm | 22,494 | 0.000790 [<10⁻¹², 0.00779] | 3.32×10⁻⁷ [<10⁻¹², 0.4993] |
| Middle arm | 5,120 | 0.2347 [0.0340, 0.3468] | 0.2584 [0.0402, 0.4493] |
| Upper arm | 44,750 | ≈0.5 [0.4987, 0.5] | 1.95×10⁻⁶ [<10⁻¹², 0.4999] |
| No E turning points | 2,310 | 0.0646 [<10⁻¹², 0.5] | 7.70×10⁻⁷ [<10⁻¹², 0.5000] |

Thus lower-arm E is usually small, while its I distribution includes a high-I
component. It cannot define both-populations-quiet rest on its own. Middle-arm
roots span substantial E and I activity, but also extend toward low activity.
Upper-arm E is highly concentrated near its ceiling, while its I distribution
spans almost the whole allowed range.

The detailed cohort shows the same broad separation: median E is 0.00235,
0.2310, and 0.5 on lower, middle, and upper arms, respectively; median I on the
middle arm is 0.2739. The 60,559 eligible detailed records come from a smaller,
purposefully selected circuit set. This is a descriptive cross-cohort comparison,
not independent replication or a matched causal comparison.

Arm occupancy boundaries depend on circuit parameters. For example, the
screen lower-arm E maximum is 0.1537 despite its much smaller 95th percentile;
upper-arm E starts at 0.3553 in the pooled screen. Pooled ranges must not become
universal arm thresholds.

## 2. What does inhibitory response side add?

The upper E arm is the clearest example:

| Cohort / I response | n | Median I | Central 90% of I | I < 0.01 | I ≥ 0.49 |
| --- | ---: | ---: | --- | ---: | ---: |
| Screen / ascending | 4,293 | 0.4850 | 0.4183–0.5000 | 0% | 44.9% |
| Screen / descending | 40,457 | 1.70×10⁻⁷ | <10⁻¹²–0.4997 | 86.2% | 10.7% |
| Detailed / ascending | 401 | 0.4768 | 0.4510–0.4963 | 0% | 26.4% |
| Detailed / descending | 39,391 | 1.11×10⁻⁵ | <10⁻¹²–0.4998 | 78.9% | 12.7% |

The 0.01 and 0.49 bounds here are the first and last display-bin boundaries,
not adopted biological definitions. The near-zero median of the descending
category hides a second concentration near maximal I. Histogram and cumulative
views expose that structure; a mean or median alone does not.

On the lower arm, both response sides also include low- and high-I records.
The middle-arm attracting group is different: all its records are on the
ascending I response in this catalog, and most lie between the I extremes.

**Classification implication:** keep E arm, I response side, and actual I
activity separately visible. A descending response is a structural failure-side
descriptor; it cannot on its own identify the low-I endpoint. A lower E arm
cannot on its own identify rest.

## 3. Which roots were excluded by selecting attracting equilibria?

The stability figure uses all recorded roots within each structural category.
All 14,553 screen and 11,958 detailed middle-E-arm/descending-I-response roots
are saddles. Their absence from the attracting activity plots therefore reflects
the stability selection, not a lack of recorded equilibria in that category.

Middle-arm/ascending-response roots include attracting, saddle, and repelling
records. Lower- and upper-arm descending categories also contain saddles.
Every recorded lower- or upper-arm ascending root in these cohorts is attracting.
These are finite inventory statements; no global stability theorem or exhaustive
search is claimed. Numerical discovery remains `CompletenessNotCertified`.

## Plots and tables

Gallery: `output/catalog_structural_figures_20261017_final/index.html`.

1. **Typical activity by structure:** median, central 50%, and central 90%
   intervals for E and I within each E-arm/I-response pair. Missing eligible
   groups are explicitly marked; every group has its own count.
2. **Cumulative occupancy by E arm:** exact within-arm empirical distributions,
   answering "what fraction has occupancy at or below x?" Full 0–0.5 axes
   include exact zeros and the ceiling.
3. **I histograms by arm and response side:** each distribution sums to 100%
   separately; 50 bins of width 0.01 show endpoint concentrations. No-E-fold
   records are omitted from this one figure and retained elsewhere.
4. **Stability within structural categories:** within-category proportions and
   counts for all roots, explaining the selection behind the first three plots.
5–6. **Joint E–I histograms:** screen and detailed cohorts separately. E is
   horizontal and I vertical; columns show all E geometry and individual lower,
   middle, and upper E arms. Rows pool I response sides or select ascending or
   descending response. Each nonempty panel sums to 100%, with identical
   100 × 100 bins and shared logarithmic percentage color scales across both
   figures. White bins contain no sampled records; empty groups are labeled.
   Unlike separate E/I summaries, these preserve the pairing of E and I.

`quantiles.csv` supplies 108 E/I summaries for joint structural categories,
E-arm marginals, I-response marginals, and whole-cohort distributions, with both all-root and eligible-root
subsets. `histograms.csv.gz` contains their 5,400 bin counts. `stability.csv`
contains 27 structural summaries, including near-singular counts. Marginal
rows overlap joint rows and must not be summed as independent records.

`joint_histograms.csv.gz` stores 80,194 nonempty two-dimensional bins; zero
bins are implicit. Both all-root and eligible-root counts are retained.

The derived data packet is **301,962 bytes (0.302 MB)**. Full empirical curves are rendered
directly from the existing catalog rather than stored as another root table.
PDF, SVG, and PNG exports are provided; all generated files remain ignored.

## Reproduce and verify

```sh
julia --project=. scripts/catalog_structural_distributions.jl output/equilibrium_catalog_20261016 output/catalog_structural_distributions_20261017_v2
python3 scripts/verify_structural_distributions.py output/equilibrium_catalog_20261016 output/catalog_structural_distributions_20261017_v2
julia --project=plotting scripts/render_structural_distributions.jl output/equilibrium_catalog_20261016 output/catalog_structural_distributions_20261017_v2 output/catalog_structural_figures_20261017_final
```

Use fresh output destinations on replay. Catalog and derived-packet manifests
are checked, and analysis, quantile-helper, catalog-reader, and renderer hashes
are recorded. Quantiles use observed values at rank max(1, ceil(q×n)), without
interpolation. Empty groups have missing quantiles, not zero activity.

The 21 focused assertions passed, including exact-zero/ceiling handling,
next-representable-value bin boundaries, tied empirical CDF jumps, empty groups,
invalid occupancies, order invariance, and stability/near-singularity filtering.
An independent Python verifier reproduced all 108 distributions, all 5,400
one-dimensional bins, all 80,194 nonempty two-dimensional bins, and all 27
stability summaries directly from source catalog rows. Joint-bin tests also
check E/I axis orientation, boundary inclusion, empty input, and invalid input.

The full Julia package suite passed **5,160/5,160 assertions** in 9m14.7s.
All six final figures rendered in PNG, PDF, and SVG; PNGs were visually
inspected for layout, endpoint visibility, labels, and normalization semantics.
All six SVGs parsed successfully. Exact manifests and current source hashes
were independently checked for the catalog, summary packet, and figure gallery.
An intermediate renderer call failed because of extra filename arguments;
that call was corrected and the complete final render succeeded. The separate
plotting test suite was not run; the renderer itself was executed end to end.
