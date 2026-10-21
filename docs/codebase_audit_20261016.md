# Codebase audit, 16 October 2026

## Decision boundary

The current forward path is the 6 October five-figure candidate design. Its
parameter registry, root and plotting environments, numerical pipeline,
renderer, and scientific limits stay in the current tree. Generated candidate
measurements are excluded from this PR; historical source and documents
remain recoverable through Jujutsu/Git history. Only selected final results
need an active packet outside version history.
This audit did not select final figures, change model equations or scientific
labels, or certify attractor completeness.

The audit workspace began at change `olxzyxqq`, the clean five-figure change.
The separate unfinished equilibrium audit in `main/` was not incorporated.

## Findings and changes

| Finding | Evidence and action |
| --- | --- |
| Active scripts have historical names | The five-figure screen and detail runners transitively include 15 scripts, including input-response, narrative, input-release, basin, and minimal-experiment modules. Retained all 15 and their relevant tests; their source snapshots and hashes remain part of fresh-run provenance. |
| Superseded standalone branches crowded the active tree | Retired 22 scripts, 17 dedicated tests, seven configurations, the old certification directory, the claims catalogue, and four earlier reproducibility directories. Removed the Figure-5b-only Hopf/topology module and exports. Kept `adaptive_rescue.toml` because the active input-release config loads its role records. No active runner or remaining test uses the retired files. |
| Paper handoffs and old result notes obscured the current study | Retired 20 superseded documents; rewrote the README and next steps around the current workflow; kept the model reference, analytical constraints, and current figure study. Removed stale references to generated, untracked render files and the historical repair script. |
| Unused direct dependencies increased setup cost | Removed root `ForwardDiff` and plotting `Base64` and root-package dependencies with Julia Pkg. Transitive dependencies may remain where other packages require them; both manifests stay tracked. |

The retained package still supplies equilibrium discovery, local stability,
continuation, periodic-orbit shooting, pulse experiments, and basin measurement.
The historical five-figure measurements and plotted arrays were not changed by
this cleanup.
The deep script include chain is a maintainability risk, but changing it would
alter source identities and require a fresh large study. No such refactor was
needed for this cleanup.

## Reproducibility boundary

A fresh checkout can run a new screen, detail study, independent verification,
packet build, and render from retained scripts, configuration, registry, and
tracked environments. The full 232-case numerical study was not rerun for this
audit, so that path is supported by retained code and focused tests rather than
a new end-to-end result. The original study's independent checks and visual
review remain historical evidence in the study note and locally archived run.
Final-result selection and any new scientific claims remain with the author.

## Verification

- Julia 1.10.12 root `Pkg.test()` passed 2,948/2,948 assertions; the plotting
  suite passed 6/6. The first two root attempts exposed a test still importing
  a retired pulse runner and an input-release config that loads adaptive role
  records. Those dependencies were corrected before the passing run.
- The locally archived candidate packet's 173 manifest entries matched its
  exact file set and SHA-256 digests. A fresh render produced five PNG/PDF/SVG
  sets; its 17 manifest entries verified. All five PNGs were visually inspected
  and were byte-identical to the prior local render.
- Static checks found no missing retained Julia includes, local documentation
  links, or literal script/config paths. `git diff --check` passed. Certification
  tests were retired with that component. CI and a full numerical recomputation
  were not run here.

## Follow-up, 17 October 2026

PR review found that the initial packet builder did not bind independent
verification to the exact completed detail stages, then that selection was not
bound to the screen contents. Both paths now check the relevant checksum
manifests and reject changed inputs. The candidate packet was rebuilt from the
original study's archived numerical outputs and source snapshots using the
current builder and verifier. A further review found that the verification
summary did not bind its trajectory report; it now records and checks that
report's digest. The screen selection and 14 detail-stage markers were checked,
and all 176 independent comparisons retained their numerical values. The
packet's 173 entries and the fresh render's 17 entries verify; all five PNGs
are byte-identical to the prior render. The full 232-case study was
not rerun. These changes refresh provenance for the candidate packet without
promoting it to a final scientific result.

The provenance refresh used the locally archived source tree: all 30 screen
and 31 detail source hashes matched their original metadata. The current
checkout deliberately rejects those historical outputs because their source
hashes predate pruning. Its supported path is a new study from current sources,
not a rebuild of the historical numerical run.

## Correction, 21 October 2026

The candidate packet was generated output, so retaining its 174 files in the
active tree and PR history overstated the reproducibility requirement. The PR
branch was rebuilt without those files. Packet artifacts are outside version
control, and current output paths are ignored. The parameter registry stays
because it is an input to a new study; the final packet will be frozen only
after author selection.
