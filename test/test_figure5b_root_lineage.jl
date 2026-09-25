using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "scripts", "figure5b_root_lineage.jl"))
const Lineage = Figure5bRootLineage

function lineage_model(; ratio=4.4)
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=7.8 * ratio,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=8.0)),
        coupling=PointCoupling(e_to_e=19.0, i_to_e=13.0,
            e_to_i=19.0, i_to_i=6.0), drive=NoDrive())
end

function lineage_seeds(model, points)
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=points)
        for i in range(0.0, upper_i; length=points)])
    return unique!(seeds)
end

function lineage_search_copy(search; equilibria=search.equilibria,
    unresolved_nearby=search.unresolved_nearby)
    return EquilibriumSearchResult(search.model, search.frozen_model,
        search.frozen_drive, search.source_time, search.options,
        search.stability_options, search.attempts, equilibria,
        unresolved_nearby, search.completeness)
end

@testset "Figure-5b root lineage contract" begin
    model = lineage_model()
    searches = Tuple(find_equilibria(model;
        seeds=lineage_seeds(model, points)) for points in Lineage.GRID_POINTS)
    baseline = Lineage.build_root_tracks(searches)
    @test baseline.qualified
    @test baseline.root_counts == (7, 7, 7)
    @test length(baseline.tracks) == 7
    @test all(track -> all(isfinite, track.residuals), baseline.tracks)
    safe_displacement = baseline.minimum_separation / 4

    central = classify_figure5b_topology(searches).central_state
    seeded = Lineage.seed_lineage(searches, central)
    @test seeded.accepted
    @test seeded.anchor !== nothing
    @test seeded.anchor.origin_model_identity == seeded.anchor.model_identity
    @test seeded.corrected_match !== nothing
    @test seeded.anchor.states ==
        seeded.anchor.source_tracks[seeded.anchor.followed_source_index]
    @test seeded.reasons isa Tuple
    @test seeded.tracks.tracks isa Tuple
    @test seeded.corrected_state == Tuple(central)

    reordered = (lineage_search_copy(searches[1];
            equilibria=reverse(searches[1].equilibria)),
        lineage_search_copy(searches[2];
            equilibria=searches[2].equilibria[[2, 1, 3, 4, 5, 6, 7]]),
        lineage_search_copy(searches[3];
            equilibria=reverse(searches[3].equilibria)))
    @test Lineage.build_root_tracks(reordered).qualified
    same = Lineage.transition_lineage(seeded.anchor, reordered, central;
        displacement_atol=safe_displacement, predicted_state=central,
        predictor_atol=0.01)
    @test same.accepted
    @test same.prior_match == same.predictor_match == same.corrected_match
    @test same.anchor.states == seeded.anchor.states
    @test same.reciprocal_match == seeded.anchor.followed_source_index
    @test same.source_anchor === seeded.anchor
    @test same.predicted_state == Tuple(central)

    # A five-root intermediate refinement is admissible when all grids agree.
    keep = [1, 2, 3, 4, 5]
    five = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[keep]) for search in searches)
    five_tracks = Lineage.build_root_tracks(five)
    @test five_tracks.qualified
    @test five_tracks.root_counts == (5, 5, 5)
    five_seed = Lineage.seed_lineage(five, five[3].equilibria[1].state)
    @test five_seed.accepted
    @test Lineage.transition_lineage(five_seed.anchor, five,
        five[3].equilibria[1].state; displacement_atol=2e-6).accepted

    # A 7-to-5 transition is allowed when the followed central root survives.
    central_index = seeded.corrected_match
    other_indices = [i for i in 1:7 if i != central_index]
    removed = (first(other_indices), last(other_indices))
    survivor_indices = [i for i in 1:7 if !(i in removed)]
    survivor = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[survivor_indices]) for search in searches)
    survivor_transition = Lineage.transition_lineage(seeded.anchor, survivor,
        central; displacement_atol=safe_displacement)
    @test survivor_transition.accepted
    @test survivor_transition.anchor !== nothing
    @test length(survivor_transition.anchor.source_tracks) == 5
    @test survivor_transition.anchor.origin_model_identity ==
        seeded.anchor.origin_model_identity
    @test length(unique(survivor_transition.destination_source_matches)) == 5

    # Losing the central root cannot be masked by a surviving neighbor.
    nearest_other = argmin([i == central_index ? Inf :
        norm(searches[3].equilibria[i].state .- central)
        for i in 1:7])
    lost_indices = [i for i in 1:7 if i != central_index &&
        i != (central_index == 1 ? 7 : 1)]
    lost_five = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[lost_indices]) for search in searches)
    lost_five_transition = Lineage.transition_lineage(seeded.anchor, lost_five,
        searches[3].equilibria[nearest_other].state; displacement_atol=2.0)
    @test !lost_five_transition.accepted
    @test :tracked_root_lost in lost_five_transition.reasons
    @test lost_five_transition.anchor === nothing

    # Independent review's one-root reproduction: forward proximity alone
    # used to accept this surviving neighbor as the vanished central root.
    one_neighbor = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[nearest_other:nearest_other])
        for search in searches)
    neighbor_distance = norm(searches[3].equilibria[nearest_other].state .- central)
    disappeared = Lineage.transition_lineage(seeded.anchor, one_neighbor,
        searches[3].equilibria[nearest_other].state;
        displacement_atol=max(neighbor_distance + 0.01, 0.168))
    @test !disappeared.accepted
    @test :tracked_root_lost in disappeared.reasons
    @test disappeared.prior_match == 1
    @test disappeared.reciprocal_match == nearest_other
    @test disappeared.anchor === nothing

    # With source roots at x=0 and x=.03 and one destination at x=.01,
    # reciprocal nearest matching alone wrongly favors the vanished x=0 root.
    central_track = baseline.tracks[central_index].states
    shifted(offset) = ntuple(j -> (central_track[j][1] + offset,
        central_track[j][2]), 3)
    close_source = Lineage.RootLineageAnchor(
        seeded.anchor.origin_model_identity, seeded.anchor.model_identity,
        shifted(-0.01), (shifted(-0.01), shifted(0.02)), 1)
    central_only = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[central_index:central_index])
        for search in searches)
    close_disappearance = Lineage.transition_lineage(close_source,
        central_only, central; displacement_atol=0.05)
    @test !close_disappearance.accepted
    @test close_disappearance.prior_match == 1
    @test close_disappearance.reciprocal_match == 1
    @test :source_isolation_unresolved in close_disappearance.reasons
    @test close_disappearance.anchor === nothing

    # Even reciprocal matching is unresolved if source neighbors are within
    # coordinate uncertainty of the same destination root.
    close = ntuple(j -> (seeded.anchor.states[j][1] + 5e-7,
        seeded.anchor.states[j][2]), 3)
    forged = Lineage.RootLineageAnchor(seeded.anchor.origin_model_identity,
        seeded.anchor.model_identity, seeded.anchor.states,
        (seeded.anchor.source_tracks..., close),
        seeded.anchor.followed_source_index)
    reciprocal_ambiguous = Lineage.transition_lineage(forged, searches,
        central; displacement_atol=safe_displacement)
    @test !reciprocal_ambiguous.accepted
    @test :reciprocal_root_ambiguous in reciprocal_ambiguous.reasons
    @test reciprocal_ambiguous.anchor === nothing

    # A solved neighbor cannot substitute for the followed root.
    other = first(root for root in searches[3].equilibria
        if norm(root.state .- central) > 0.01)
    switched = Lineage.transition_lineage(seeded.anchor, searches,
        other.state; displacement_atol=safe_displacement)
    @test !switched.accepted
    @test :corrected_root_switch in switched.reasons
    @test switched.anchor === nothing

    # A broad bound loses isolation even when reciprocal nearest matches agree.
    broad = Lineage.transition_lineage(seeded.anchor, searches, central;
        displacement_atol=2.0)
    @test !broad.accepted
    @test :source_isolation_unresolved in broad.reasons
    @test :destination_isolation_unresolved in broad.reasons
    @test broad.reciprocal_match == seeded.anchor.followed_source_index

    bad_predictor = Lineage.transition_lineage(seeded.anchor, searches, central;
        displacement_atol=safe_displacement, predicted_state=other.state,
        predictor_atol=0.01)
    @test !bad_predictor.accepted
    @test :predictor_root_switch in bad_predictor.reasons

    unequal = (searches[1], searches[2], lineage_search_copy(searches[3];
        equilibria=searches[3].equilibria[1:6]))
    @test :root_count_mismatch in Lineage.build_root_tracks(unequal).reasons
    missing = (searches[1], searches[2], lineage_search_copy(searches[3];
        equilibria=vcat(searches[3].equilibria[1:6],
            searches[3].equilibria[6:6])))
    @test :ambiguous_refinement_match in Lineage.build_root_tracks(missing).reasons
    @test :insufficient_root_separation in Lineage.build_root_tracks(missing).reasons

    nearby = (searches[1], searches[2], lineage_search_copy(searches[3];
        unresolved_nearby=[[1, 2]]))
    @test :unresolved_nearby_roots in Lineage.build_root_tracks(nearby).reasons
    wrong_schedule = (searches[1], searches[1], searches[3])
    @test :refinement_schedule_mismatch in
        Lineage.build_root_tracks(wrong_schedule).reasons
    altered_model = PointModelParameters(excitatory=model.excitatory,
        inhibitory=model.inhibitory,
        coupling=PointCoupling(e_to_e=19.1, i_to_e=13.0,
            e_to_i=19.0, i_to_i=6.0), drive=NoDrive())
    altered_search = find_equilibria(altered_model;
        seeds=lineage_seeds(altered_model, 41))
    @test :model_context_mismatch in
        Lineage.build_root_tracks((searches[1], searches[2], altered_search)).reasons

    root = searches[3].equilibria[1]
    corrupt = Equilibrium(root.state .+ [2e-5, 0.0], root.balance_residual,
        root.balance_jacobian, root.near_singular,
        root.representative_attempt, root.member_attempts, root.stability)
    corrupt_search = lineage_search_copy(searches[3];
        equilibria=vcat([corrupt], searches[3].equilibria[2:end]))
    corrupt_tracks = Lineage.build_root_tracks((searches[1], searches[2],
        corrupt_search); options=Lineage.RootLineageOptions(
        coordinate_atol=3e-5, minimum_root_separation=1e-4))
    @test :root_quality_failure in corrupt_tracks.reasons
    @test !Lineage.seed_lineage((searches[1], searches[2], corrupt_search),
        central; options=Lineage.RootLineageOptions(
            coordinate_atol=3e-5,
            minimum_root_separation=1e-4)).accepted

    # The neutral Hopf root has unresolved local stability but valid identity.
    critical = hopf_diagnostics(model, central).critical_ratio
    neutral_model = lineage_model(ratio=critical)
    neutral_searches = Tuple(find_equilibria(neutral_model;
        seeds=lineage_seeds(neutral_model, points))
        for points in Lineage.GRID_POINTS)
    neutral_tracks = Lineage.build_root_tracks(neutral_searches)
    @test neutral_tracks.qualified
    @test any(track -> all(==(StabilityUnresolved), track.classifications),
        neutral_tracks.tracks)
    neutral_seed = Lineage.seed_lineage(neutral_searches, central)
    @test neutral_seed.accepted
end
