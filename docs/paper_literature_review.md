# Literature review for the paper contribution

Prepared 30 September 2026 for author review, alongside
[Paper evidence and author handoff](paper_evidence.md). This document records
the targeted literature review and proposed comparisons; it does not revise
the manuscript or settle the contribution on the author's behalf.

The literature supports a more specific contribution question: how inhibitory
failure changes the relationship between ordinary switching, distinct
high-activity states, and the interventions that move the circuit between them.
Recent papers already address dysfunction-specific treatment and preservation
of normal activity. The opportunity is to establish the additional relationship
between these properties within the same circuit and explain where it holds
across parameter space.

## Closest recent comparison

Kamaraj and Szuromi's *Modelling dysfunction-specific interventions for seizure
termination in epilepsy* appeared online in December 2025 and has a 2026 volume
citation. It is a plausible match for the highly relevant 2025 paper discussed
previously, but that identification remains unconfirmed.
[Primary paper](https://www.nature.com/articles/s41540-025-00632-9).

The authors modify Wilson–Cowan dynamics to study hyperexcitation, inhibitory
transmitter depletion, and depolarizing GABA. Intervention effectiveness depends
on the dysfunction. They also examine disruption of normal activity and propose
an intervention intended to reduce that disruption. Their seizure-termination
analysis focuses on eliminating the seizure attractor through parameter changes.

**Proposed comparison:** different high-activity states within the same circuit
may require different inputs. Our study can distinguish interventions that leave
an attractor's basin from those that remove the attractor, and test whether
ordinary rest–active switching remains possible. General claims that treatment
should depend on dysfunction or preserve normal activity would overlap with
this paper; the additional comparison needs explicit evidence.

## Recent experimental and modeling literature

The findings column summarizes the cited work. The implications column contains
proposed interpretations for this project, rather than conclusions attributed
to those authors.

| Paper | Reported finding | Proposed implication for this paper |
| --- | --- | --- |
| **Proskurina, Ergina and Zaitsev, 2025 — Interneuron-Driven Ictogenesis in the 4-Aminopyridine Model: Depolarization Block and Potassium Accumulation Initiate Seizure-like Activity.** [Primary paper](https://doi.org/10.3390/ijms26146812) | In mouse entorhinal–hippocampal slices, strong interneuron recruitment precedes depolarization block and pyramidal recruitment, alongside extracellular potassium accumulation. | Provides recent experimental motivation for inhibition failing under intense drive. The experiment involves interacting cellular and ionic mechanisms; our model can isolate consequences of a declining inhibitory response without treating that response as the entire biological mechanism. |
| **Lemaire et al., 2025 — Depolarization block induction via slow NaV1.1 inactivation in Dravet syndrome.** [Primary paper](https://www.nature.com/articles/s41598-025-17468-2) | A conductance-based interneuron model connects altered slow sodium-channel inactivation to failure to sustain firing during prolonged stimulation. | Supports distinguishing weak inhibition from inhibition that initially responds and subsequently fails. Input duration and the timescale of failure become meaningful comparison points; the cellular model does not directly calibrate our population failure threshold. |
| **Chiang et al., 2025 — State-dependent effects of responsive neurostimulation depend on seizure localization.** [Primary paper](https://academic.oup.com/brain/article/148/2/521/7721060) | A retrospective clinical analysis finds that stimulation effects depend on seizure-risk state and localization. The states describe longer-term risk fluctuations. | State-dependent stimulation is an established research direction. Our contribution could explain a particular circuit mechanism. The clinical risk states should remain distinct from the model's instantaneous high-activity states. |
| **Páscoa dos Santos and Verschure, 2025 — Excitatory-inhibitory homeostasis and bifurcation control in the Wilson–Cowan model of cortical dynamics.** [Primary paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012723) | Different homeostatic mechanisms affect maintenance of firing rates, oscillatory frequencies, and position relative to bifurcations as inputs change. | Preserving normal function needs an operational definition. A measured rest–active switching test could provide a functional benchmark beyond retaining a low-activity equilibrium. |
| **Godin et al., 2025 — Control of Inhibition-Stabilized Oscillations in Wilson-Cowan Networks with Homeostatic Plasticity.** [Primary paper](https://doi.org/10.3390/e27020215) | Studies inhibitory stimulation, paradoxical responses, oscillations, and homeostatic plasticity in inhibition-stabilized networks. | A decrease in inhibitory activity following inhibitory stimulation does not uniquely identify depolarization block. Explain the mechanism through the response curve and circuit dynamics, rather than through the response direction alone. |
| **Liu, De Schutter and Li, 2024 — GABA-Induced Seizure-Like Events Caused by Multi-ionic Interactive Dynamics.** [Primary paper](https://www.eneuro.org/content/11/10/ENEURO.0308-24.2024) | An ionic model shows how chloride, bicarbonate, and potassium interactions can make GABAergic activity participate in seizure-like dynamics. | Distinguish failure of inhibitory neurons to fire from failure or reversal of their synaptic effect. Similar intervention outcomes can arise through different mechanisms. |
| **Duan et al., 2026 — Depolarization block paradoxically drives surges of neurotransmitter release during seizure activity.** [Primary article record and abstract](https://pubmed.ncbi.nlm.nih.gov/42113533/) | Experiments report suppressed spiking alongside substantial glutamate and dopamine release during depolarization block. | State clearly whether the inhibitory variable represents firing or effective inhibitory output. The paper does not directly establish this phenomenon for inhibitory GABA release, but it shows why firing and transmitter output should not be silently equated. |

An additional human comparison is Merricks et al.'s *Aberrant fast spiking
interneuronal activity precedes seizure transitions in humans*. It connects
changes in putative fast-spiking interneuron activity to impending seizure
transitions. The version located is a **2024 preprint**; retain that status
unless a subsequent publication is verified.
[Preprint](https://www.medrxiv.org/content/10.1101/2024.01.26.24301821v1.full).

## Earlier work that needs a direct comparison

Meijer et al. (2015), *Modeling Focal Epileptic Activity in the Wilson–Cowan
Model with Depolarization Block*, introduced a nonmonotonic population response,
motivated it through heterogeneous firing-onset and block thresholds, and
obtained an additional high-excitation, low-inhibition equilibrium. Both the
response construction and the additional state therefore need a precise
comparison with our derivation. A mathematical advance would need to be
identified in its assumptions, generality, or consequences.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4385301/).

Călin, Ilie and Akerman (2021), *Disrupting Epileptiform Activity by Preventing
Parvalbumin Interneuron Depolarization Block*, experimentally showed that brief
hyperpolarizing interventions could reduce parvalbumin-interneuron block,
sustain firing, and disrupt epileptiform activity. Preventing inhibitory block
as an intervention principle thus has direct experimental precedent. Our
proposed comparison is to explain intervention success within a circuit that
also supports ordinary switching and competing high-activity states.
[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8580142/).

These comparisons complement the existing discussion of Kim and Nykamp (2017),
Tryba et al. (2019), Liou et al. (2020), and Agopyan-Miu et al. (2023) in
[paper_evidence.md](paper_evidence.md#contribution-and-biological-positioning).
That earlier comparison remains relevant, especially for coexistence,
input-driven transitions, spatial interpretation, and tissue-specific dynamics.

## Candidate contribution question

For author consideration:

> Under what conditions can the same circuit support ordinary rest–active
> switching and distinct high-activity states, and how does inhibitory failure
> change the inputs required to enter and leave those states?

Three connected deliverables could support this argument.

1. **A precise mathematical distinction.** Establish what the ordering result
   rules out under its stated assumptions and what the fire-then-fail response
   makes possible. Compare the derivation directly with Meijer and the dynamics
   with the existing predecessors. Identify the added result explicitly.
2. **A map of intervention requirements.** Compare excitatory withdrawal,
   inhibitory input, and finite-duration perturbations across paired states
   under common parameters and baseline inputs. Distinguish leaving an
   attractor's basin from eliminating the attractor. Keep direct state
   displacement separate from realizable input perturbations.
3. **A functional tradeoff.** Measure whether successful intervention preserves
   actual rest–active switching in the same parameter context, and identify
   where that compatibility changes. Recovery in one context and preserved
   switching in another should remain separate observations.

The detailed search can locate regions where these properties occur together
and identify boundaries worth explaining. An unsuccessful intervention at one
parameter setting is a point on that map, rather than a conclusion about what
the model can support. Completed cases can guide focused follow-ups around
promising regimes and contrasting boundaries.

## Reading and review order

1. **Kamaraj and Szuromi:** compare the treatment objective, dysfunctions,
   attractor removal, and definition of preserved normal activity. Confirm
   whether this is the paper recalled from the previous discussion.
2. **Meijer, alongside Kim and Nykamp:** compare the response derivation,
   equilibrium structure, and input-driven transitions with the ordering
   result and proposed state repertoire.
3. **Călin, Proskurina, and Lemaire:** connect the population mechanism and
   intervention directions to experimental and cellular evidence, preserving
   the distinction between firing failure and effective synaptic inhibition.
4. **The functional and clinical comparisons:** use the homeostasis and
   stimulation studies to make the chosen normal-function benchmark and
   meaning of state dependence explicit.

The contribution statement should follow this comparison and the detailed-case
results. The present document proposes questions and evidence requirements;
it does not assert that the combination has already been established or that
its priority has been exhaustively certified.

## Review scope

This targeted review covers recent primary literature from 2024–2026 and the
closest earlier comparisons identified during the search. Sources were checked
through primary-paper text, PDFs, indexed primary-paper text, or article
abstracts on 30 September 2026. Each entry links to the paper or its primary
article record. The review is intended to guide positioning and source reading,
not to substitute for an exhaustive systematic review.
