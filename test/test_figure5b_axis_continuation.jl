using LinearAlgebra: norm

include(joinpath(@__DIR__, "..", "scripts", "figure5b_axis_continuation.jl"))
const Axis5b = Figure5bAxisContinuation
const AxisLineage = Axis5b.Figure5bRootLineage

function axis_anchor_model(; e_to_i=19.0, theta_off=8.0)
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=7.8,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=7.8 * 4.4,
            response=FailureOfInhibitionResponse(slope=5.0,
                onset_threshold=4.0, failure_threshold=theta_off)),
        coupling=PointCoupling(e_to_e=19.0, i_to_e=13.0,
            e_to_i=e_to_i, i_to_i=6.0), drive=NoDrive())
end

@testset "Figure-5b axis continuation contracts" begin
    model = axis_anchor_model()
    factory = parameter -> model
    options = Axis5b.AxisOptions(initial_step_fraction=0.5,
        minimum_step_fraction=0.25, maximum_step_fraction=0.5,
        max_steps=2, max_retries=1)
    searches = Axis5b._searches(model, options)
    topology = classify_figure5b_topology(searches)
    @test topology.qualified
    center = Axis5b._state(topology.central_state)
    seeded = AxisLineage.seed_lineage(searches, center)
    @test seeded.accepted
    root_search = (model, options) -> searches
    synthetic_trace = (model, state, parameter) -> parameter - 19.0
    positive_trace = (model, state, parameter) -> parameter - 20.0

    for kwargs in ((; initial_step_fraction=0.0),
            (; minimum_step_fraction=0.02, initial_step_fraction=0.01),
            (; trace_atol=Inf), (; max_steps=0))
        @test_throws ArgumentError Axis5b.AxisOptions(; kwargs...)
    end
    @test_throws ArgumentError Axis5b.AxisOptions(-0.01, 0.005,
        0.01, 2, 1, 0.05, 1e-8, 1e-8, 1e-10, 1e-5, 30,
        EquilibriumOptions(), StabilityOptions())
    @test_throws ArgumentError Axis5b.trace_zero_axis(factory, 19.0;
        parameter_bounds=(19.0, 19.0))
    @test_throws ArgumentError Axis5b.trace_zero_axis(factory, 19.0;
        parameter_bounds=(-floatmax(Float64), floatmax(Float64)))

    # The default fractions can traverse either half of an authorized
    # width-16 axis well before the default step budget is exhausted.
    defaults = Axis5b.AxisOptions()
    scaled = Axis5b._scaled_steps(defaults, (11.0, 27.0))
    @test scaled.initial ≈ 0.16
    @test scaled.minimum ≈ 0.00016
    @test scaled.maximum ≈ 0.8
    for (direction, bound) in ((-1, 11.0), (1, 27.0))
        parameter, step, accepted_steps = 19.0, scaled.initial, 0
        while parameter != bound && accepted_steps < defaults.max_steps
            parameter = Axis5b._axis_target(parameter, bound, step, direction)
            step = Axis5b._grown_step(step, scaled.maximum)
            accepted_steps += 1
        end
        @test parameter == bound
        @test accepted_steps < defaults.max_steps
    end

    # Both exact bounds are required for a finite exhausted path.
    quiet = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, trace_function=positive_trace)
    @test quiet.seed.accepted
    @test quiet.negative.termination == :exact_bound
    @test quiet.positive.termination == :exact_bound
    @test quiet.finite_exhausted
    @test length(quiet.negative.trials) == 1
    @test length(quiet.positive.trials) == 1
    @test all(trial -> trial.accepted,
        (quiet.negative.trials..., quiet.positive.trials...))
    @test all(trial -> trial.local_solve !== nothing &&
        trial.local_solve.validation == AdmissibleCandidate &&
        trial.local_solve.solver_status isa Symbol,
        (quiet.negative.trials..., quiet.positive.trials...))

    failed_solve = (model, state, options) -> error("injected local solve failure")
    failed = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, solve_function=failed_solve,
        trace_function=positive_trace)
    @test !failed.finite_exhausted
    @test failed.negative.termination in (:minimum_step, :max_retries)
    @test !isempty(failed.negative.trials)
    @test all(trial -> :trial_exception in trial.reasons,
        failed.negative.trials)
    @test all(trial -> trial.error !== nothing, failed.negative.trials)
    @test all(trial -> trial.local_solve === nothing, failed.negative.trials)

    rejected_solve = (model, state, options) -> begin
        solved = solve_equilibrium(model, collect(state);
            options=options.equilibrium_options,
            stability_options=options.stability_options)
        attempt = solved.attempt
        rejected = EquilibriumAttempt(attempt.seed, attempt.candidate,
            :injected_rejection, false, attempt.solver_residual,
            attempt.balance_residual, attempt.residual_norm,
            attempt.balance_jacobian, attempt.near_singular,
            RejectedCandidate, [:injected_failure])
        EquilibriumSolveResult(solved.model, solved.frozen_model,
            solved.frozen_drive, solved.source_time, solved.options,
            solved.stability_options, rejected, solved.stability)
    end
    rejected = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, solve_function=rejected_solve,
        trace_function=positive_trace)
    rejected_trial = first(rejected.negative.trials)
    @test :local_equilibrium_unresolved in rejected_trial.reasons
    @test rejected_trial.error === nothing
    @test rejected_trial.local_solve !== nothing
    @test rejected_trial.local_solve.validation == RejectedCandidate
    @test rejected_trial.local_solve.solver_status == :injected_rejection
    @test !rejected_trial.local_solve.solver_success
    @test rejected_trial.local_solve.reasons == (:injected_failure,)
    @test rejected_trial.local_solve.candidate isa Tuple

    wrong_model = axis_anchor_model(theta_off=12.0)
    wrong_seed = Axis5b.trace_zero_axis(parameter -> wrong_model, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options)
    @test !wrong_seed.seed.accepted
    @test :root_count_mismatch in wrong_seed.seed.reasons
    @test wrong_seed.status == :seed_unresolved
    @test isempty(wrong_seed.negative.trials)
    bad_seed_search = (model, options) -> error("injected seed search failure")
    seed_exception = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=bad_seed_search)
    @test :seed_exception in seed_exception.seed.reasons
    @test occursin("injected seed search failure", seed_exception.seed.error)

    policy_options = Axis5b.AxisOptions(initial_step_fraction=0.5,
        minimum_step_fraction=0.25, maximum_step_fraction=0.5, max_steps=1,
        max_retries=1,
        equilibrium_options=EquilibriumOptions(residual_atol=1e-8))
    policy_mismatch = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=policy_options,
        search_function=root_search)
    @test !policy_mismatch.seed.accepted
    @test :search_policy_mismatch in policy_mismatch.seed.reasons

    wrong_searches = Axis5b._searches(wrong_model, options)
    calls = Ref(0)
    wrong_context_search = (model, options) -> begin
        calls[] += 1
        calls[] == 1 ? searches : wrong_searches
    end
    context = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=wrong_context_search,
        trace_function=positive_trace)
    @test context.seed.accepted
    @test !context.finite_exhausted
    @test :search_model_mismatch in first(context.negative.trials).reasons

    foreign_solve = (model, state, options) -> solve_equilibrium(
        wrong_model, collect(state);
        options=options.equilibrium_options,
        stability_options=options.stability_options)
    foreign = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, solve_function=foreign_solve,
        trace_function=positive_trace)
    @test !foreign.finite_exhausted
    @test :local_solve_context_mismatch in
        first(foreign.negative.trials).reasons

    # A local Newton solve onto another genuine root cannot advance the axis.
    other = first(root for root in searches[3].equilibria
        if norm(root.state .- collect(center)) > 0.01)
    jump_solve = (model, state, options) -> solve_equilibrium(model,
        other.state; options=options.equilibrium_options,
        stability_options=options.stability_options)
    jump_options = Axis5b.AxisOptions(initial_step_fraction=0.5,
        minimum_step_fraction=0.25, maximum_step_fraction=0.5, max_steps=1,
        max_retries=1, state_step_atol=2.0)
    jumped = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=jump_options,
        search_function=root_search, solve_function=jump_solve,
        trace_function=positive_trace)
    @test !jumped.finite_exhausted
    @test !first(jumped.negative.trials).accepted
    @test first(jumped.negative.trials).lineage !== nothing
    @test :source_isolation_unresolved in
        first(jumped.negative.trials).lineage.reasons

    # A root-count change is handled by the shared assignment contract.
    track(x) = ((Float64(x), 0.0), (Float64(x), 0.0),
        (Float64(x), 0.0))
    assignment_reasons = Symbol[]
    assignment = AxisLineage._constellation_assignment(
        (track(1), track(2), track(3)), (track(1), track(3)),
        0.01, 1e-6, assignment_reasons)
    @test assignment == [1, 3]
    @test isempty(assignment_reasons)

    left = Axis5b.AxisPoint(18.99, center, -0.01, seeded.anchor)
    right = Axis5b.AxisPoint(19.01, center, 0.01, seeded.anchor)
    candidate = Axis5b.AugmentedCandidate((center..., 19.0), :converged, 1)
    corrected = Axis5b._validate_correction(factory, left, right,
        candidate, (18.99, 19.01), options,
        AxisLineage.RootLineageOptions(), root_search, synthetic_trace)
    @test corrected.accepted
    @test corrected.status == :accepted
    @test corrected.corrector_status == :converged
    @test corrected.corrector_iterations == 1
    @test corrected.from_left.accepted && corrected.from_right.accepted
    @test corrected.minimum_singular_value > options.rank_atol
    @test corrected.point.parameter == 19.0

    # Every augmented finite-difference probe stays in the accepted bracket.
    bracket_probes = Float64[]
    guarded_factory = parameter -> begin
        push!(bracket_probes, Float64(parameter))
        18.99 <= parameter <= 19.01 || error("outside bracket")
        model
    end
    guarded = Axis5b._newton_correct(guarded_factory, synthetic_trace,
        left, right, (18.98, 19.02), options)
    @test guarded.status == :converged
    guarded_residual = z -> Axis5b._augmented_residual(guarded_factory,
        synthetic_trace, z)
    for parameter in (18.99, 19.0, 19.01)
        @test Axis5b._augmented_jacobian(guarded_residual,
            Float64[center..., parameter], (18.99, 19.01), 1e-3) !== nothing
    end
    @test !isempty(bracket_probes)
    @test all(parameter -> 18.99 <= parameter <= 19.01,
        bracket_probes)
    stalled = Axis5b._newton_correct(factory,
        (model, state, parameter) -> 1.0, left, right,
        (18.99, 19.01), options)
    @test stalled.status == :rank_unresolved
    @test stalled.iterations == 0

    out_of_bracket = Axis5b._validate_correction(factory, left, right,
        Axis5b.AugmentedCandidate((center..., 19.02), :converged, 1),
        (18.98, 19.02), options, AxisLineage.RootLineageOptions(),
        root_search, synthetic_trace)
    @test !out_of_bracket.accepted
    @test :correction_out_of_bracket in out_of_bracket.reasons
    out_of_bounds = Axis5b._validate_correction(factory, left, right,
        Axis5b.AugmentedCandidate((center..., 19.02), :converged, 1),
        (18.99, 19.01), options, AxisLineage.RootLineageOptions(),
        root_search, synthetic_trace)
    @test :correction_out_of_bounds in out_of_bounds.reasons

    rank_failure = Axis5b._validate_correction(factory, left, right,
        candidate, (18.99, 19.01), options,
        AxisLineage.RootLineageOptions(), root_search,
        (model, state, parameter) -> 0.0)
    @test !rank_failure.accepted
    @test :correction_rank_unresolved in rank_failure.reasons
    @test rank_failure.corrector_status == :converged
    @test rank_failure.corrector_iterations == 1

    # A single finite-difference scale falsely reports rank for these
    # stationary trace zeros; the multiscale gate must leave them unresolved.
    cubic_trace = (model, state, parameter) -> (parameter - 19.0)^3
    cubic_correction = Axis5b._validate_correction(factory, left, right,
        candidate, (18.99, 19.01), options,
        AxisLineage.RootLineageOptions(), root_search, cubic_trace)
    @test !cubic_correction.accepted
    @test cubic_correction.minimum_singular_value > options.rank_atol
    @test :correction_rank_unresolved in cubic_correction.reasons
    cubic_singleton = Axis5b._validate_zero_endpoint(factory,
        Axis5b.AxisPoint(19.0, center, 0.0, seeded.anchor),
        (18.99, 19.01), options, cubic_trace)
    @test !cubic_singleton.accepted
    @test cubic_singleton.minimum_singular_value > options.rank_atol
    @test :zero_endpoint_rank_unresolved in cubic_singleton.reasons

    square_trace = (model, state, parameter) -> (parameter - 19.0)^2
    bound_point = Axis5b.AxisPoint(19.0, center, 0.0, seeded.anchor)
    square_singleton = Axis5b._validate_zero_endpoint(factory,
        bound_point, (19.0, 19.01), options, square_trace)
    @test !square_singleton.accepted
    @test square_singleton.minimum_singular_value > options.rank_atol
    @test :zero_endpoint_rank_unresolved in square_singleton.reasons
    square_correction = Axis5b._validate_correction(factory, bound_point,
        right, candidate, (19.0, 19.01), options,
        AxisLineage.RootLineageOptions(), root_search, square_trace)
    @test !square_correction.accepted
    @test square_correction.minimum_singular_value > options.rank_atol
    @test :correction_rank_unresolved in square_correction.reasons

    # An interval narrower than the requested finest h must not alias all
    # three probes to the same clipped width at a bound.
    narrow_bounds = (19.0, 19.0 + 1e-6)
    narrow_right = Axis5b.AxisPoint(narrow_bounds[2], center, 1e-12,
        seeded.anchor)
    narrow_square_singleton = Axis5b._validate_zero_endpoint(factory,
        bound_point, narrow_bounds, options, square_trace)
    @test !narrow_square_singleton.accepted
    @test :zero_endpoint_rank_unresolved in narrow_square_singleton.reasons
    narrow_square_correction = Axis5b._validate_correction(factory,
        bound_point, narrow_right, candidate, narrow_bounds, options,
        AxisLineage.RootLineageOptions(), root_search, square_trace)
    @test !narrow_square_correction.accepted
    @test :correction_rank_unresolved in narrow_square_correction.reasons
    narrow_linear_singleton = Axis5b._validate_zero_endpoint(factory,
        bound_point, narrow_bounds, options, synthetic_trace)
    @test narrow_linear_singleton.accepted
    narrow_linear_correction = Axis5b._validate_correction(factory,
        bound_point, narrow_right, candidate, narrow_bounds, options,
        AxisLineage.RootLineageOptions(), root_search, synthetic_trace)
    @test narrow_linear_correction.accepted

    # Comparing singular values alone misses this cubic derivative because
    # the smaller, fixed balance mode pins all three minimum values at 1e-7.
    hidden_cubic = z -> Float64[z[1], 1e-7z[2],
        100(z[3] - 19.0)^3]
    hidden_rank, hidden_stable = Axis5b._stable_augmented_rank(hidden_cubic,
        Float64[0.0, 0.0, 19.0], (18.99, 19.01), options)
    @test hidden_rank ≈ 1e-7
    @test !hidden_stable

    wrong_root = Axis5b._validate_correction(factory, left, right,
        Axis5b.AugmentedCandidate((Tuple(other.state)..., 19.0),
            :converged, 1), (18.99, 19.01), jump_options,
        AxisLineage.RootLineageOptions(), root_search, synthetic_trace)
    @test !wrong_root.accepted
    @test :left_lineage_unresolved in wrong_root.reasons
    @test :right_lineage_unresolved in wrong_root.reasons

    other_seed = AxisLineage.seed_lineage(searches, other.state)
    @test other_seed.accepted
    other_right = Axis5b.AxisPoint(19.01, Tuple(other.state), 0.01,
        other_seed.anchor)
    one_sided = Axis5b._validate_correction(factory, left, other_right,
        candidate, (18.99, 19.01), jump_options,
        AxisLineage.RootLineageOptions(), root_search, synthetic_trace)
    @test !one_sided.accepted
    @test one_sided.from_left.accepted
    @test !one_sided.from_right.accepted
    @test :right_lineage_unresolved in one_sided.reasons

    # A zero-trace seed is a singleton location, with rank probed inside
    # authorized bounds; no zero-width augmented bracket is invented.
    zero = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, trace_function=synthetic_trace)
    @test zero.seed.accepted
    @test zero.seed_zero !== nothing
    @test zero.seed_zero.accepted
    @test zero.seed_zero.corrector_status == :not_run_singleton
    @test zero.seed_zero.corrector_iterations == 0
    @test isempty(zero.negative.brackets)
    @test isempty(zero.positive.brackets)
    @test zero.finite_exhausted
    @test Axis5b._has_crossing(left, right, options.trace_atol)
    singleton_point = Axis5b.AxisPoint(19.0, center, 0.0, seeded.anchor)
    @test !Axis5b._has_crossing(singleton_point, right,
        options.trace_atol)

    endpoint_trace = (model, state, parameter) -> parameter - 19.01
    zero_at_bound = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, trace_function=endpoint_trace)
    @test zero_at_bound.seed_zero === nothing
    @test length(zero_at_bound.positive.zero_endpoints) == 1
    @test only(zero_at_bound.positive.zero_endpoints).accepted
    @test only(zero_at_bound.positive.zero_endpoints).corrector_status ==
        :not_run_singleton
    @test isempty(zero_at_bound.positive.brackets)

    failed_corrector = (factory, trace, left, right, bounds, options) ->
        Axis5b.AugmentedCandidate((NaN, NaN, NaN), :rank_unresolved, 7)
    crossing_trace = (model, state, parameter) -> parameter - 19.005
    unresolved = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, trace_function=crossing_trace,
        correct_function=failed_corrector)
    @test !unresolved.finite_exhausted
    @test unresolved.positive.termination == :exact_bound
    @test !first(unresolved.positive.brackets).correction.accepted
    @test :corrector_unresolved in
        first(unresolved.positive.brackets).correction.reasons
    @test first(unresolved.positive.brackets).correction.corrector_status ==
        :rank_unresolved
    @test first(unresolved.positive.brackets).correction.corrector_iterations == 7

    throwing_corrector = (args...) -> error("injected corrector failure")
    thrown = Axis5b._trace_zero_axis(factory, 19.0;
        parameter_bounds=(18.99, 19.01), axis_options=options,
        search_function=root_search, trace_function=crossing_trace,
        correct_function=throwing_corrector)
    thrown_correction = first(thrown.positive.brackets).correction
    @test thrown_correction.corrector_status == :corrector_exception
    @test thrown_correction.corrector_iterations == 0
    @test occursin("injected corrector failure", thrown_correction.error)

    # An exception inside Newton retains the last accepted iterate and its
    # iteration count, unlike an external corrector that never returned.
    root_hits = Ref(0)
    late_throwing_trace = (model, state, parameter) -> begin
        if abs(parameter - 19.005) < 1e-10
            root_hits[] += 1
            root_hits[] >= 2 && error("late Newton trace failure")
        end
        parameter - 19.005
    end
    partial = Axis5b._newton_correct(factory, late_throwing_trace,
        left, right, (18.99, 19.01), options)
    @test partial.status == :corrector_exception
    @test partial.iterations >= 1
    @test partial.candidate[3] ≈ 19.005
    @test occursin("late Newton trace failure", partial.error)
    partial_correction = Axis5b._validate_correction(factory, left, right,
        partial, (18.99, 19.01), options,
        AxisLineage.RootLineageOptions(), root_search, synthetic_trace)
    @test !partial_correction.accepted
    @test partial_correction.corrector_iterations == partial.iterations
    @test partial_correction.error == partial.error

    # One small genuine axis interval exercises discovery and local solves.
    real_factory = parameter -> axis_anchor_model(e_to_i=parameter)
    smoke = Axis5b.trace_zero_axis(real_factory, 19.0;
        parameter_bounds=(18.999, 19.001),
        axis_options=Axis5b.AxisOptions(initial_step_fraction=0.5,
            minimum_step_fraction=0.05, maximum_step_fraction=0.5,
            max_steps=1, max_retries=1))
    @test smoke.seed.accepted
    @test length(smoke.negative.trials) >= 1
    @test length(smoke.positive.trials) >= 1
    @test all(trial -> trial.target_parameter >= 18.999 &&
        trial.target_parameter <= 19.001,
        (smoke.negative.trials..., smoke.positive.trials...))
end
