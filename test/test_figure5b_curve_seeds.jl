using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "scripts", "figure5b_curve_seeds.jl"))
const CurveSeeds = Figure5bCurveSeeds
const CurveAxis = CurveSeeds.Axis
const CurveLineage = CurveSeeds.Lineage

function curve_seed_model(; ratio=0.5, e_to_e=18.5, i_to_i=7.0)
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=7.8ratio,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=8.0)),
        coupling=PointCoupling(e_to_e=e_to_e, i_to_e=13.0,
            e_to_i=19.0, i_to_i=i_to_i), drive=NoDrive())
end

@testset "Figure-5b finite curve seed qualification" begin
    model = curve_seed_model()
    center = solve_equilibrium(model, [0.36, 0.39]).attempt.candidate
    box = ((18.3, 18.7), (6.8, 7.2))
    result = CurveSeeds.seed_trace_zero_curve(model, (:e_to_e, :i_to_i),
        center; parameter_bounds=box)
    @test result.origin.accepted
    @test result.context.pair == (:e_to_e, :i_to_i)
    @test result.context.bounds == box
    @test result.options isa CurveSeeds.CurveSeedOptions
    @test result.lineage_options isa CurveLineage.RootLineageOptions
    @test result.origin.topology.qualified
    @test result.origin.lineage.accepted
    @test result.status == :qualified_seeds
    @test result.unresolved_methods == (:edge_1_lower, :edge_1_upper)
    @test length(result.attempts) == 5
    @test length(result.qualifications) == 5
    @test first(result.attempts).method == :interior_kkt
    @test result.qualifications[1].accepted
    @test result.qualifications[1].topology.qualified
    @test result.qualifications[1].topology.tracks.root_counts == (7, 7, 7)
    @test result.qualifications[1].rank.qualified
    @test result.qualifications[1].homotopy.qualified
    @test all(trial -> trial.accepted || !isempty(trial.reasons),
        result.qualifications[1].homotopy.trials)
    @test all(index -> result.qualifications[index].accepted,
        result.qualified_indices)
    @test result.qualifications[4].accepted
    @test result.qualifications[5].accepted
    @test any(iteration -> iteration.status == :proposal_out_of_bounds,
        result.attempts[2].solve.iterations)
    @test result.attempts[2].solve.status == :backtrack_unresolved

    context = CurveSeeds._context(model, (:e_to_e, :i_to_i), box)
    replay = CurveSeeds._attempts(context, result.origin,
        CurveSeeds.CurveSeedOptions(), CurveSeeds._solve_bounded)
    @test Tuple((attempt.method, attempt.solve.status,
        attempt.candidate, attempt.solve.residual_norm) for attempt in replay) ==
        Tuple((attempt.method, attempt.solve.status,
        attempt.candidate, attempt.solve.residual_norm)
            for attempt in result.attempts)

    reversed = CurveSeeds._context(model, (:i_to_i, :e_to_e),
        (box[2], box[1]))
    @test CurveLineage._point_model_identity(CurveSeeds._model(context,
        (18.47, 7.01))) == CurveLineage._point_model_identity(
        CurveSeeds._model(reversed, (7.01, 18.47)))
    @test CurveSeeds._regularity(reversed,
        (result.attempts[1].candidate[1:2]...,
            result.attempts[1].candidate[4],
            result.attempts[1].candidate[3]),
        CurveSeeds.CurveSeedOptions()).qualified

    slope_atol = CurveAxis._topology_options(
        CurveLineage.RootLineageOptions()).slope_atol
    @test slope_atol == Figure5bTopologyOptions().slope_atol
    @test CurveSeeds._rising_slopes([1.0 -1.0; 1.0 -1.0],
        slope_atol) == (1.0, 1.0)
    @test CurveSeeds._rising_slopes([1.0 -1e-10; 1.0 -1.0],
        slope_atol) === nothing
    @test CurveSeeds._rising_slopes([1e-10 -1.0; 1.0 -1.0],
        slope_atol) === nothing
    relaxed_trace = CurveSeeds.CurveSeedOptions(
        axis_options=CurveAxis.AxisOptions(trace_atol=1.0))
    nonneutral = CurveSeeds._neutral_topology(result.origin.searches,
        result.origin.central_state, relaxed_trace,
        CurveLineage.RootLineageOptions())
    @test !nonneutral.qualified
    @test :central_not_neutral in nonneutral.reasons
    @test :central_trace_unresolved in nonneutral.reasons

    for kwargs in ((; solver_atol=0.0), (; rank_atol=Inf),
            (; state_scales=(0.1, false)), (; maxiters=true),
            (; homotopy_min_fraction=1.0))
        @test_throws ArgumentError CurveSeeds.CurveSeedOptions(; kwargs...)
    end
    @test_throws ArgumentError CurveSeeds.CurveSeedOptions(
        CurveAxis.AxisOptions(), -1.0, 20, 8, 1e-10, 1e-5,
        (0.1, 0.1), 1e-4, 64, 1e-6)
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :e_to_e), center; parameter_bounds=box)
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((13.0, 18.7), box[2]))
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((18.3, 18.3), box[2]))
    rounded_below = big"13.999999999999999999999999"
    rounded_above = big"24.000000000000000000000001"
    @test Float64(rounded_below) == 14.0
    @test Float64(rounded_above) == 24.0
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((rounded_below, 24.0), box[2]))
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((14.0, rounded_above), box[2]))
    rounded_past_origin = big"18.500000000000000000000001"
    rounded_before_origin = big"18.499999999999999999999999"
    @test Float64(rounded_past_origin) == 18.5
    @test Float64(rounded_before_origin) == 18.5
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((rounded_past_origin, 18.7), box[2]))
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((18.3, rounded_before_origin), box[2]))
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((big"18.5", rounded_past_origin), box[2]))
    outward_lower = big"18.300000000000000000000001"
    outward_upper = big"18.699999999999999999999999"
    @test Float64(outward_lower) == 18.3
    @test Float64(outward_upper) == 18.7
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((outward_lower, outward_upper), box[2]))
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center;
        parameter_bounds=((18.3f0, 18.7f0), box[2]))
    outside_frozen = PointModelParameters(excitatory=model.excitatory,
        inhibitory=model.inhibitory,
        coupling=PointCoupling(e_to_e=18.5, i_to_e=100.0,
            e_to_i=19.0, i_to_i=7.0), drive=NoDrive())
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(
        outside_frozen, (:e_to_e, :i_to_i), center;
        parameter_bounds=box)
    outside_threshold = PointModelParameters(excitatory=model.excitatory,
        inhibitory=PopulationParameters(timescale=3.9,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=13.0)),
        coupling=model.coupling, drive=NoDrive())
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(
        outside_threshold, (:e_to_e, :i_to_i), center;
        parameter_bounds=box)
    drifted_coupling = PointCoupling(
        e_to_e=big"18.50000000000000000001", i_to_e=big"13.0",
        e_to_i=big"19.0", i_to_i=big"7.0")
    drifted_origin = PointModelParameters(excitatory=model.excitatory,
        inhibitory=model.inhibitory, coupling=drifted_coupling,
        drive=NoDrive())
    @test_throws ArgumentError CurveSeeds.seed_trace_zero_curve(
        drifted_origin, (:e_to_e, :i_to_i), center;
        parameter_bounds=box)

    wrong_center = CurveSeeds.seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), (0.5, 0.5); parameter_bounds=box)
    @test wrong_center.status == :origin_unresolved
    @test :supplied_center_mismatch in wrong_center.origin.reasons
    @test isempty(wrong_center.attempts)

    wrong_context_search = (candidate_model, axis_options) ->
        result.origin.searches
    wrong_context = CurveSeeds._seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center; parameter_bounds=box,
        search_function=wrong_context_search)
    @test wrong_context.origin.accepted
    @test isempty(wrong_context.qualified_indices)
    @test :search_model_mismatch in wrong_context.qualifications[1].reasons

    injected_failure = (args...) -> error("injected seed solver failure")
    failed = CurveSeeds._seed_trace_zero_curve(model,
        (:e_to_e, :i_to_i), center; parameter_bounds=box,
        search_function=wrong_context_search,
        solve_function=injected_failure)
    @test length(failed.attempts) == 5
    @test all(attempt -> attempt.solve.status == :solver_exception &&
        occursin("injected seed solver failure", attempt.solve.error),
        failed.attempts)
    @test all(qualification -> !qualification.accepted,
        failed.qualifications)
    @test failed.status == :search_unresolved
    @test failed.unresolved_methods == Tuple(attempt.method for attempt in
        failed.attempts)

    short_policy = CurveSeeds.CurveSeedOptions(homotopy_max_trials=2,
        homotopy_min_fraction=0.1)
    foreign = curve_seed_model(i_to_i=6.0)
    foreign_solve = (candidate_model, state, axis_options) ->
        solve_equilibrium(foreign, collect(state);
            options=axis_options.equilibrium_options,
            stability_options=axis_options.stability_options)
    foreign_path = CurveSeeds._homotopy(context, result.origin,
        result.attempts[1].candidate, short_policy,
        CurveLineage.RootLineageOptions(), CurveAxis._searches,
        foreign_solve)
    @test !foreign_path.qualified
    @test all(trial -> :local_solve_context_mismatch in trial.reasons &&
        trial.local_solve !== nothing, foreign_path.trials)
    other_root = first(root for root in result.origin.searches[3].equilibria
        if norm(root.state .- collect(result.origin.central_state)) > 0.01)
    wrong_lineage = CurveLineage.seed_lineage(result.origin.searches,
        other_root.state)
    @test wrong_lineage.accepted
    forged_origin = CurveSeeds.CurveOriginEvidence(true, (),
        result.origin.supplied_state, result.origin.central_state,
        result.origin.topology, wrong_lineage, result.origin.searches,
        nothing)
    switched_path = CurveSeeds._homotopy(context, forged_origin,
        result.attempts[1].candidate, short_policy,
        CurveLineage.RootLineageOptions(), CurveAxis._searches,
        CurveAxis._solve)
    @test !switched_path.qualified
    @test any(trial -> trial.lineage !== nothing &&
        !trial.lineage.accepted, switched_path.trials)

    # Raw duplicate corner attempts are retained; only qualified seeds dedup.
    corner_solver = (residual, start, bounds, options) -> begin
        values = length(start) == 7 ? Tuple(start) :
            (start[1], start[2], bounds[3][1])
        CurveSeeds.SeedSolveResult(true, :converged, values, 0.0, (), nothing)
    end
    corners = CurveSeeds._attempts(context, result.origin,
        CurveSeeds.CurveSeedOptions(), corner_solver)
    @test length(corners) == 5
    @test corners[2].candidate == corners[4].candidate
    @test corners[2].candidate[3:4] ==
        (box[1][1], box[2][1])
    corner_checks = Tuple(CurveSeeds._qualify(context, result.origin,
        attempt, CurveSeeds.CurveSeedOptions(),
        CurveLineage.RootLineageOptions(), CurveAxis._searches,
        CurveAxis._solve) for attempt in corners)
    @test length(corner_checks) == 5
    @test all(qualification -> !qualification.accepted, corner_checks)
    # Qualification precedes deduplication: retain two raw accepted records.
    duplicate_attempts = (result.attempts[1], result.attempts[1])
    duplicate_checks = (result.qualifications[1], result.qualifications[1])
    @test length(duplicate_attempts) == length(duplicate_checks) == 2
    @test CurveSeeds._qualified_indices(duplicate_attempts,
        duplicate_checks, context, CurveSeeds.CurveSeedOptions()) == (1,)
end

@testset "Curve numerical seeds are not scientific qualification" begin
    options = CurveSeeds.CurveSeedOptions()
    # Neither coordinate axis through (0.6,0.6) reaches p+q=0.4 inside
    # [0,1]^2, but the finite stationary-distance KKT attempt finds (0.2,0.2).
    F(z) = [z[1] - 0.25, z[2] - 0.5, z[3] + z[4] - 0.4]
    z0 = [0.25, 0.5, 0.6, 0.6]
    KKT(value) = begin
        z = view(value, 1:4)
        jacobian = CurveSeeds.ForwardDiff.jacobian(F, z)
        vcat(F(z), z .- z0 .+ transpose(jacobian) * view(value, 5:7))
    end
    solved = CurveSeeds._solve_bounded(KKT, vcat(z0, zeros(3)),
        ((-Inf, Inf), (-Inf, Inf), (0.0, 1.0), (0.0, 1.0),
            (-Inf, Inf), (-Inf, Inf), (-Inf, Inf)), options)
    @test solved.converged
    @test collect(solved.candidate[3:4]) ≈ [0.2, 0.2] atol=1e-9
    @test isempty(findall(p -> abs(F([0.25, 0.5, p, 0.6])[3]) < 1e-8,
        0.0:0.1:1.0))
    @test isempty(findall(q -> abs(F([0.25, 0.5, 0.6, q])[3]) < 1e-8,
        0.0:0.1:1.0))

    hidden = z -> [z[1], 1e-7z[2], 100(z[3] - 19.0)^3]
    hidden_rank = CurveSeeds._multiscale_rank(hidden,
        (0.0, 0.0, 19.0, 6.0), ((18.99, 19.01), (5.99, 6.01)),
        options)
    @test !hidden_rank.qualified
    @test hidden_rank.minimum_singular_value ≈ 1e-7
    @test :rank_multiscale_unresolved in hidden_rank.reasons
    narrow_rank = CurveSeeds._multiscale_rank(
        z -> [z[1], z[2], z[3] - 19.0],
        (0.0, 0.0, 19.0, 6.0),
        ((19.0, 19.0 + 1e-13), (5.99, 6.01)), options)
    @test !narrow_rank.qualified
    @test :rank_probe_unresolved in narrow_rank.reasons
    bounded = z -> begin
        19.0 <= z[3] <= 19.01 && 6.0 <= z[4] <= 6.01 ||
            error("rank probe escaped box")
        [z[1], z[2], z[3] + z[4] - 25.0]
    end
    bounded_rank = CurveSeeds._multiscale_rank(bounded,
        (0.0, 0.0, 19.0, 6.0), ((19.0, 19.01), (6.0, 6.01)), options)
    @test bounded_rank.qualified

    anchor = curve_seed_model(ratio=4.4, e_to_e=19.0, i_to_i=6.0)
    historical = CurveSeeds.seed_trace_zero_curve(anchor,
        (:e_to_e, :i_to_i), (0.329934171847263, 0.367295168265613);
        parameter_bounds=((14.0, 24.0), (0.0, 10.0)))
    @test historical.origin.accepted
    @test historical.status == :search_unresolved
    @test !isempty(historical.unresolved_methods)
    @test length(historical.attempts) == 5
    @test any(index -> historical.attempts[index].solve.converged &&
        :root_count_mismatch in historical.qualifications[index].reasons &&
        historical.qualifications[index].topology.tracks.root_counts ==
            (5, 5, 5), eachindex(historical.attempts))
end
