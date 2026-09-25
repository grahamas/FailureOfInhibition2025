using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "scripts", "figure5b_curve_traversal.jl"))
const Traverse = Figure5bCurveTraversal
const TraverseSeeds = Traverse.Seeds

function traverse_model(; ratio=0.5)
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=7.8ratio,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=8.0)),
        coupling=PointCoupling(e_to_e=18.5, i_to_e=13.0,
            e_to_i=19.0, i_to_i=7.0), drive=NoDrive())
end

@testset "Figure-5b curve traversal numerical gates" begin
    T = Traverse
    options = T.CurveTraversalOptions()
    for kwargs in ((; initial_step=0.0), (; rank_atol=Inf),
            (; max_steps=true), (; max_retries=-1),
            (; progress_fraction=1.0), (; minimum_step=0.1))
        @test_throws ArgumentError T.CurveTraversalOptions(; kwargs...)
    end
    @test_throws ArgumentError T.CurveTraversalOptions(0.01, 0.02,
        0.05, 1, 1, 1e-8, 1e-8, 1e-10, 0.1, 0.75, 1e-5,
        1e-8, 1e-6)

    scales = (1.0, 1.0, 1.0, 1.0)
    linear(z) = [z[1], z[2], z[3] + z[4] - 1.0]
    tangent = T._tangent(linear, (0.0, 0.0, 0.5, 0.5),
        scales, options.rank_atol)
    @test tangent.accepted
    @test norm(collect(tangent.scaled)) ≈ 1.0
    @test collect(T._tangent(linear, (0.0, 0.0, 0.5, 0.5), scales,
        options.rank_atol; prior=Tuple(-collect(tangent.scaled))).scaled) ≈
        -collect(tangent.scaled)
    rank_bad(z) = [z[1], z[1], z[3] + z[4]]
    @test !T._tangent(rank_bad, (0.0, 0.0, 0.5, 0.5),
        scales, options.rank_atol).accepted

    fold(z) = [z[1], z[2], z[3] - z[4]^2]
    at_fold = T._tangent(fold, (0.0, 0.0, 0.0, 0.0),
        scales, options.rank_atol)
    @test at_fold.accepted
    @test abs(at_fold.actual[3]) < 1e-9
    @test abs(at_fold.actual[4]) > 0.9
    @test isempty(T._ray_hits((0.0, 0.0, 0.0, 0.0), at_fold,
        ((0.0, 1.0), (-1.0, 1.0)), 0.01, options.boundary_atol))
    along_edge(z) = [z[1], z[2], z[3]]
    edge_tangent = T._tangent(along_edge, (0.0, 0.0, 0.0, 0.0),
        scales, options.rank_atol)
    @test edge_tangent.accepted
    @test isempty(T._ray_hits((0.0, 0.0, 0.0, 0.0), edge_tangent,
        ((0.0, 1.0), (-1.0, 1.0)), 0.01, options.boundary_atol))
    turnback(z) = [z[1], z[2], z[3] - (1 - z[4]^2)]
    before_touch = T._tangent(turnback, (0.0, 0.0, 0.99, -0.1),
        scales, options.rank_atol; prior=(0.0, 0.0, 0.5, 0.5))
    at_touch = T._tangent(turnback, (0.0, 0.0, 1.0, 0.0),
        scales, options.rank_atol; prior=before_touch.scaled)
    @test T._outward_transverse(before_touch, 1, 1,
        options.transversality_atol)
    @test !T._outward_transverse(at_touch, 1, 1,
        options.transversality_atol)
    corner_tangent = T.CurveTangent(true, (), (0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1 / sqrt(2), 1 / sqrt(2)), 1.0, nothing, nothing)
    @test length(T._ray_hits((0.0, 0.0, 0.5, 0.5), corner_tangent,
        ((0.0, 1.0), (0.0, 1.0)), 1.0, 1e-8)) == 2
    @test !T._mark_ambiguous(T.CurveBoundaryAttempt(1, 0.1, 1,
        1.0, 0.1, (0.0, 0.0, 1.0, 1.0), nothing,
        (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0), 0.0, 1.0,
        0.1, true, (), nothing, nothing, nothing, nothing,
        nothing)).accepted

    two_components(z) = [z[1], z[2], (z[3] - 0.2) * (z[3] - 0.8)]
    current = (0.0, 0.0, 0.2, 0.0)
    predictor = (0.0, 0.0, 0.21, 0.0)
    second_component = (0.0, 0.0, 0.8, 0.0)
    @test maximum(abs, two_components(collect(current))) == 0
    @test maximum(abs, two_components(collect(second_component))) < 1e-15
    progress, correction, reasons = T._geometry(current, predictor,
        second_component, corner_tangent, scales, 0.01, options)
    @test progress > 0
    @test correction > options.correction_fraction * 0.01
    @test :nonlocal_predictor_correction in reasons
    @test :nonlocal_current_advance in reasons
    _, _, stagnant = T._geometry(current, predictor, current,
        corner_tangent, scales, 0.01, options)
    @test :stagnant_or_backward_correction in stagnant
    loop(z) = [z[1], z[2], z[3]^2 + z[4]^2 - 1]
    first_loop = (0.0, 0.0, 1.0, 0.0)
    near_return = (0.0, 0.0, cos(1e-6), sin(1e-6))
    @test abs(loop(collect(first_loop))[3]) < 1e-12
    @test abs(loop(collect(near_return))[3]) < 1e-12
    @test T._revisited((first_loop,), near_return, scales,
        options.revisit_atol)
    @test !T._revisited((first_loop,), (0.0, 0.0, 0.0, 1.0),
        scales, options.revisit_atol)
end

@testset "Figure-5b curve traversal verified FoI seed" begin
    model = traverse_model()
    center = solve_equilibrium(model, [0.36, 0.39]).attempt.candidate
    seeds = TraverseSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((18.3, 18.7), (6.8, 7.2)))
    @test seeds.status == :qualified_seeds
    @test 1 in seeds.qualified_indices
    @test !Traverse._verify_seed(seeds, true).accepted
    @test !Traverse._verify_seed(seeds, 2).accepted
    forged_membership = TraverseSeeds.CurveSeedResult(seeds.context,
        seeds.options, seeds.lineage_options, seeds.origin, seeds.attempts,
        seeds.qualifications, (2,), seeds.unresolved_methods,
        :qualified_seeds)
    @test :seed_record_mismatch in
        Traverse._verify_seed(forged_membership, 2).reasons
    verified = Traverse._verify_seed(seeds, 1)
    @test verified.accepted
    @test verified.seed.source == :pr3_seed
    @test verified.seed.original_unresolved_methods ==
        seeds.unresolved_methods
    @test verified.seed.anchor.model_identity ==
        Traverse.Lineage._model_identity(
            verified.seed.source_evidence[2].searches[1])
    options = Traverse.CurveTraversalOptions(initial_step=1e-3,
        minimum_step=1e-5, maximum_step=1e-3, max_steps=1)
    result = Traverse._continue_verified(verified, options)
    @test result.seed_verification.accepted
    @test isempty(result.initial_reasons)
    @test length(result.directions) == 2
    @test result.status in (:finite_two_boundary_segment,
        :traversal_unresolved)
    @test all(direction -> !isempty(direction.points), result.directions)
    @test all(direction -> length(direction.points) == 2 &&
        only(direction.steps).accepted, result.directions)
    @test all(direction -> all(point -> point.hopf.qualified,
        direction.points), result.directions)
    @test all(direction -> all(attempt -> attempt.accepted ||
        !isempty(attempt.reasons), direction.steps), result.directions)
    stale_search = (model, policy) -> seeds.origin.searches
    stale = Traverse._continue_verified(verified, options;
        search_function=stale_search)
    @test stale.status == :seed_unresolved
    @test :search_context_mismatch in stale.initial_reasons
    failed_solve = (residual, initial, bounds, policy) ->
        TraverseSeeds.SeedSolveResult(false, :injected_failure,
            Tuple(initial), Inf, (), "injected")
    failed_options = Traverse.CurveTraversalOptions(initial_step=1e-3,
        minimum_step=1e-5, maximum_step=1e-3,
        max_steps=1, max_retries=1)
    failed = Traverse._continue_verified(verified, failed_options;
        solve_function=failed_solve)
    @test failed.status == :traversal_unresolved
    @test all(direction -> length(direction.steps) == 2 &&
        all(attempt -> !attempt.accepted &&
            attempt.raw.status == :injected_failure &&
            :corrector_unresolved in attempt.reasons,
            direction.steps), failed.directions)

    swapped = TraverseSeeds._context(model, (:i_to_i, :e_to_e),
        ((6.8, 7.2), (18.3, 18.7)))
    swapped_point = (verified.seed.point[1:2]...,
        verified.seed.point[4], verified.seed.point[3])
    @test Traverse.Lineage._point_model_identity(
        TraverseSeeds._model(swapped, swapped_point[3:4])) ==
        Traverse.Lineage._point_model_identity(
            TraverseSeeds._model(verified.seed.context,
                verified.seed.point[3:4]))
    @test Traverse._tangent(z -> TraverseSeeds._residual(swapped, z),
        swapped_point, (0.1, 0.1, 0.4, 0.4), 1e-10).accepted

    edge_verified = Traverse._verify_seed(seeds, 4)
    @test edge_verified.accepted
    edge_initial, edge_reasons, _ = Traverse._initial_point(
        edge_verified.seed, options, Traverse.Axis._searches)
    @test isempty(edge_reasons)
    edge_hits = Tuple((sign, hit) for sign in (-1, 1) for hit in
        Traverse._ray_hits(edge_initial.coordinates,
            Traverse._signed_point(edge_initial, sign).tangent,
            edge_verified.seed.context.bounds, options.initial_step,
            options.boundary_atol) if hit[3] == 0)
    @test length(edge_hits) == 1
    if !isempty(edge_hits)
        sign, hit = only(edge_hits)
        rejected = Traverse._edge_attempt(edge_verified.seed,
            Traverse._signed_point(edge_initial, sign), sign,
            options.initial_step, hit, options, failed_solve,
            Traverse.Axis._searches)
        @test !rejected.accepted
        @test rejected.raw.status == :injected_failure
        @test rejected.tangent === nothing
        @test :edge_solver_unresolved in rejected.reasons
        wrong_value = hit[2] ==
            edge_verified.seed.context.bounds[hit[1]][1] ?
            edge_verified.seed.context.bounds[hit[1]][2] :
            edge_verified.seed.context.bounds[hit[1]][1]
        wrong_edge = Traverse._edge_attempt(edge_verified.seed,
            Traverse._signed_point(edge_initial, sign), sign,
            options.initial_step, (hit[1], wrong_value, hit[3]),
            options, failed_solve, Traverse.Axis._searches)
        @test !wrong_edge.accepted
        @test wrong_edge.raw === nothing
        @test :wrong_edge_ray in wrong_edge.reasons
    end
end
