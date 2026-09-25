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
    unresolved_nearby=search.unresolved_nearby,
    frozen_model=search.frozen_model,
    frozen_drive=search.frozen_drive,
    attempts=search.attempts,
    options=search.options,
    stability_options=search.stability_options)
    return EquilibriumSearchResult(search.model, frozen_model,
        frozen_drive, search.source_time, options,
        stability_options, attempts, equilibria,
        unresolved_nearby, search.completeness)
end

function lineage_root_copy(root; state=root.state,
    balance_residual=root.balance_residual,
    stability=root.stability,
    representative_attempt=root.representative_attempt,
    member_attempts=root.member_attempts)
    return Equilibrium(state, balance_residual, root.balance_jacobian,
        root.near_singular, representative_attempt, member_attempts, stability)
end

function lineage_stability_copy(stability; jacobian=stability.jacobian,
    eigenvalues=stability.eigenvalues, trace=stability.trace,
    determinant=stability.determinant,
    spectral_abscissa=stability.spectral_abscissa,
    thresholds=stability.thresholds,
    classification=stability.classification,
    geometry=stability.geometry)
    return LocalStabilityResult(jacobian, eigenvalues, trace, determinant,
        spectral_abscissa, thresholds, classification, geometry)
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
    @test_throws ArgumentError Lineage.RootLineageOptions(
        -1.0, 1e-5, 1e-9, 1e-9, 1e-8)
    @test_throws ArgumentError Lineage.RootLineageOptions(
        1e-5, 1e-5, 1e-9, 1e-9, 1e-8)
    @test_throws ArgumentError Lineage.RootLineageOptions(
        NaN, 1e-5, 1e-9, 1e-9, 1e-8)
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

    # An unrelated new pair does not break a uniquely followed root.
    expanded = Lineage.transition_lineage(five_seed.anchor, searches,
        five[3].equilibria[1].state; displacement_atol=2e-6)
    @test expanded.accepted
    @test expanded.anchor !== nothing
    @test length(expanded.anchor.source_tracks) == 7
    @test count(isnothing, expanded.destination_source_matches) == 2
    @test length(unique(filter(!isnothing,
        expanded.destination_source_matches))) == 5

    # Equal counts can conceal one vanished and one newly discovered root.
    swapped = Tuple(lineage_search_copy(search;
        equilibria=search.equilibria[[1, 2, 3, 4, 6]]) for search in searches)
    equal_count_switch = Lineage.transition_lineage(five_seed.anchor, swapped,
        five[3].equilibria[1].state; displacement_atol=2e-6)
    @test !equal_count_switch.accepted
    @test :destination_source_lost in equal_count_switch.reasons
    @test equal_count_switch.anchor === nothing

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

    # Stored autonomous context must come from the declared source model.
    forged_model = lineage_search_copy(searches[3];
        frozen_model=altered_model)
    @test :frozen_context_mismatch in
        Lineage.build_root_tracks((searches[1], searches[2], forged_model)).reasons
    forged_drive = lineage_search_copy(searches[3];
        frozen_drive=(0.1, 0.0))
    @test :frozen_context_mismatch in
        Lineage.build_root_tracks((searches[1], searches[2], forged_drive)).reasons
    policy = searches[3].options
    invalid_equilibrium_policy = EquilibriumOptions(
        policy.solver_abstol, policy.solver_reltol, policy.residual_atol,
        policy.domain_atol, Inf, policy.singular_atol,
        policy.singular_rtol, policy.maxiters)
    invalid_stability_policy = StabilityOptions(Inf,
        searches[3].stability_options.spectral_rtol)
    for record in (
        lineage_search_copy(searches[3]; options=invalid_equilibrium_policy),
        lineage_search_copy(searches[3];
            stability_options=invalid_stability_policy))
        result = Lineage.build_root_tracks((searches[1], searches[2], record))
        @test !result.qualified
        @test :invalid_numerical_policy in result.reasons
    end

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

    # A plausible coordinate cannot carry fabricated residual or spectrum.
    forged_residual = lineage_root_copy(root;
        balance_residual=root.balance_residual .+ 1e-3)
    forged_stabilities = (
        lineage_stability_copy(root.stability;
            eigenvalues=root.stability.eigenvalues .+ (1e-3 + 0im)),
        lineage_stability_copy(root.stability;
            trace=root.stability.trace + 1e-3),
        lineage_stability_copy(root.stability;
            determinant=root.stability.determinant + 1e-3),
        lineage_stability_copy(root.stability;
            spectral_abscissa=root.stability.spectral_abscissa + 1e-3),
        lineage_stability_copy(root.stability;
            thresholds=root.stability.thresholds .+ 1e-3),
        lineage_stability_copy(root.stability;
            trace=NaN),
    )
    for forged_root in (forged_residual,
        (lineage_root_copy(root; stability=stability)
            for stability in forged_stabilities)...)
        record = lineage_search_copy(searches[3]; equilibria=vcat(
            [forged_root], searches[3].equilibria[2:end]))
        result = Lineage.build_root_tracks((searches[1], searches[2], record))
        @test !result.qualified
        @test :root_quality_failure in result.reasons
    end

    # Root representatives and members must name admissible candidates
    # for this actual root, not merely in-range attempt indices.
    invalid_index = lineage_root_copy(root; representative_attempt=0)
    unrelated_member = lineage_root_copy(root; member_attempts=vcat(
        root.member_attempts,
        searches[3].equilibria[2].representative_attempt))
    for forged_root in (invalid_index, unrelated_member)
        record = lineage_search_copy(searches[3]; equilibria=vcat(
            [forged_root], searches[3].equilibria[2:end]))
        @test :root_attempt_failure in
            Lineage.build_root_tracks((searches[1], searches[2], record)).reasons
    end
    representative = searches[3].attempts[root.representative_attempt]
    failed_attempt = EquilibriumAttempt(representative.seed,
        representative.candidate, representative.solver_status, false,
        representative.solver_residual, representative.balance_residual,
        representative.residual_norm, representative.balance_jacobian,
        representative.near_singular, representative.validation,
        representative.reasons)
    attempts = copy(searches[3].attempts)
    attempts[root.representative_attempt] = failed_attempt
    failed_record = lineage_search_copy(searches[3]; attempts)
    @test Lineage.build_root_tracks((searches[1], searches[2],
        failed_record)).qualified

    # Discovery deduplicates in maximum-coordinate distance. A diagonal
    # member may exceed the same Euclidean radius, meet only the search
    # residual tolerance, and have an unsuccessful solver return code.
    loose_policy = EquilibriumOptions(residual_atol=1e-5)
    loose_searches = Tuple(lineage_search_copy(search; options=loose_policy)
        for search in searches)
    member_root = first(candidate for candidate in loose_searches[3].equilibria
        if length(candidate.member_attempts) >= 2)
    @test length(member_root.member_attempts) >= 2
    member_index = first(index for index in member_root.member_attempts
        if index != member_root.representative_attempt)
    prior_member = loose_searches[3].attempts[member_index]
    offset = 0.75loose_policy.dedup_atol
    diagonal = member_root.state .+ [offset, offset]
    @test maximum(abs, diagonal .- member_root.state) <
        loose_policy.dedup_atol
    @test norm(diagonal .- member_root.state) > loose_policy.dedup_atol
    diagonal_residual = zeros(2)
    diagonal_jacobian = zeros(2, 2)
    point_balance!(diagonal_residual, diagonal, model, 0.0)
    point_balance_jacobian!(diagonal_jacobian, diagonal, model, 0.0)
    @test maximum(abs, diagonal_residual) > 1e-9
    @test maximum(abs, diagonal_residual) < loose_policy.residual_atol
    diagonal_member = EquilibriumAttempt(prior_member.seed, diagonal,
        :MaxIters, false, diagonal_residual, diagonal_residual,
        maximum(abs, diagonal_residual), diagonal_jacobian, false,
        AdmissibleCandidate, Symbol[])
    loose_attempts = copy(loose_searches[3].attempts)
    loose_attempts[member_index] = diagonal_member
    diagonal_record = lineage_search_copy(loose_searches[3];
        attempts=loose_attempts)
    @test Lineage.build_root_tracks((loose_searches[1], loose_searches[2],
        diagonal_record)).qualified

    # A permissive classifier cannot hide resolved real parts behind an
    # unresolved label and thereby qualify a non-neutral root lineage.
    permissive_searches = Tuple(find_equilibria(model;
        seeds=lineage_seeds(model, points),
        stability_options=StabilityOptions(spectral_atol=100.0))
        for points in Lineage.GRID_POINTS)
    @test all(search -> all(root -> root.stability.classification ==
        StabilityUnresolved, search.equilibria), permissive_searches)
    @test any(root -> maximum(abs ∘ real,
        root.stability.eigenvalues) > 1e-8,
        permissive_searches[3].equilibria)
    permissive_tracks = Lineage.build_root_tracks(permissive_searches)
    @test !permissive_tracks.qualified
    @test :root_quality_failure in permissive_tracks.reasons
    @test !Lineage.seed_lineage(permissive_searches, central).accepted

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
