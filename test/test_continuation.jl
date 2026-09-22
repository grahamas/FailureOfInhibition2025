function continuation_fold_model(parameter)
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=3 + log(2))),
        inhibitory=PopulationParameters(timescale=2.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(e_to_e=12.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
        drive=PiecewiseConstantDrive(baseline=(parameter, 0.0), pulses=(),
            interpretation=AbstractIntervention),
    )
end

@testset "Pseudo-arclength traverses two folds of an approved point model" begin
    result = continue_equilibria(continuation_fold_model, [0.465, 1 / 3], 0.0;
        parameter_bounds=(-1.0, 1.0),
        options=ContinuationOptions(initial_step=0.02, maximum_step=0.04,
            minimum_step=1e-7, max_steps=300))
    @test result.completeness == CompletenessNotCertified
    @test result.initial_solve.attempt.validation == AdmissibleCandidate
    @test result.negative.direction == -1
    @test result.positive.direction == 1
    @test result.negative.termination == :parameter_boundary
    @test result.positive.termination == :parameter_boundary
    points = result.negative.points
    folds = filter(candidate -> candidate.kind == :fold, result.negative.candidates)
    @test length(folds) == 2
    @test any(point -> point.stability.classification == Attracting, points)
    @test any(point -> point.stability.classification == Saddle, points)
    @test points[end].state[1] < 0.05
    @test points[end].parameter < -0.99
    # The independently derived equilibrium curve is
    # B(E) = 3 + log(2) + log(E/(1-2E)) - 12E.
    # Its two folds satisfy E=(3 +/- sqrt(3))/12.
    exact_fold_states = [(3 + sqrt(3)) / 12, (3 - sqrt(3)) / 12]
    for (candidate, expected_e) in zip(folds, exact_fold_states)
        first_e = points[candidate.first_point].state[1]
        second_e = points[candidate.second_point].state[1]
        @test min(first_e, second_e) <= expected_e <= max(first_e, second_e)
    end
    for branch in (result.negative, result.positive)
        @test count(attempt -> attempt.accepted, branch.attempts) == length(branch.points) - 1
        for point in branch.points
            e, i = point.state
            analytical_parameter = 3 + log(2) + log(e / (1 - 2e)) - 12e
            @test point.parameter ≈ analytical_parameter atol=1e-8
            @test i ≈ 1 / 3 atol=1e-10
            @test point.equilibrium_attempt.validation == AdmissibleCandidate
            @test point.equilibrium_attempt.residual_norm <= result.equilibrium_options.residual_atol
            @test -1 <= point.parameter <= 1
            @test sum(abs2, point.tangent) ≈ 1 atol=1e-12
            @test drive_value(point.model.drive, 0.0)[1] == point.parameter
        end
    end
end

@testset "Time-constant continuation retains numerical Hopf screening evidence" begin
    # Both populations have the exact equilibrium 0.2, response 0.25.
    # The balance Jacobian is [1.75 -3; 3 -1.25], independently giving
    # trace(Df)=1.75-1.25/tau_I and a possible Hopf at tau_I=5/7.
    factory = tau_i -> PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=5.0, threshold=1.5)),
        inhibitory=PopulationParameters(timescale=tau_i,
            response=LogisticResponse(slope=5.0, threshold=4.0)),
        coupling=PointCoupling(e_to_e=4.0, i_to_e=4.0, e_to_i=4.0, i_to_i=0.0),
        drive=PiecewiseConstantDrive(
            baseline=(1.5 - log(3) / 5, 4.0 - log(3) / 5 - 0.8),
            pulses=(), interpretation=AfferentExcitation),
    )
    result = continue_equilibria(factory, [0.2, 0.2], 0.6;
        parameter_bounds=(0.5, 1.0),
        options=ContinuationOptions(maximum_step=0.04))
    points = result.positive.points
    candidates = filter(candidate -> candidate.kind == :hopf, result.positive.candidates)
    @test length(candidates) == 1
    bracket = only(candidates)
    @test points[bracket.first_point].parameter < 5 / 7 < points[bracket.second_point].parameter
    @test points[1].stability.classification == Attracting
    @test points[end].stability.classification == Repelling
    @test all(point -> point.state ≈ [0.2, 0.2], points)
    @test all(point -> isapprox(point.stability.trace, 1.75 - 1.25 / point.parameter; atol=1e-12), points)
    @test result.completeness == CompletenessNotCertified
end

@testset "Exact neutral seed remains a Hopf candidate endpoint" begin
    factory = tau_i -> PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        inhibitory=PopulationParameters(timescale=tau_i,
            response=LogisticResponse(slope=1.0, threshold=4.0)),
        coupling=PointCoupling(e_to_e=12.0, i_to_e=12.0, e_to_i=12.0, i_to_i=0.0),
    )
    result = continue_equilibria(factory, [1 / 3, 1 / 3], 3.0;
        parameter_bounds=(2.0, 4.0), options=ContinuationOptions(max_steps=2))
    @test iszero(result.initial_solve.stability.trace)
    @test result.initial_solve.stability.classification == StabilityUnresolved
    for branch in (result.negative, result.positive)
        candidate = only(filter(candidate -> candidate.kind == :hopf, branch.candidates))
        @test candidate.first_point == 1
        @test candidate.second_point == 2
    end
end

@testset "Continuation retains retries and incomplete branches" begin
    result = continue_equilibria(continuation_fold_model, [0.465, 1 / 3], 0.0;
        parameter_bounds=(-1.0, 1.0),
        options=ContinuationOptions(initial_step=0.8, maximum_step=0.8,
            minimum_step=1e-7, max_steps=30, max_corrector_iters=3))
    @test !isempty(result.negative.points)
    @test any(attempt -> !attempt.accepted, result.negative.attempts)
    @test all(attempt -> attempt.accepted == (attempt.status == :corrector_converged),
        result.negative.attempts)
    @test all(attempt -> attempt.accepted ?
        attempt.equilibrium_attempt.validation == AdmissibleCandidate : true,
        result.negative.attempts)
    limited = continue_equilibria(continuation_fold_model, [0.465, 1 / 3], 0.0;
        parameter_bounds=(-1.0, 1.0), options=ContinuationOptions(max_steps=1))
    @test limited.negative.termination == :step_limit
    @test limited.positive.termination == :step_limit
    @test length(limited.negative.points) == 2

    # No parameter dependence and a singular state Jacobian lose full row
    # rank at the exact scalar fold. Retain the independently solved seed;
    # there is no tangent along which to assert a continuation result.
    fold_e = (3 + sqrt(3)) / 12
    fold_parameter = 3 + log(2) + log(fold_e / (1 - 2fold_e)) - 12fold_e
    constant_fold_factory = _ -> continuation_fold_model(fold_parameter)
    singular = continue_equilibria(constant_fold_factory, [fold_e, 1 / 3], 0.0;
        parameter_bounds=(-1.0, 1.0))
    @test singular.initial_solve.attempt.validation == AdmissibleCandidate
    @test singular.negative.termination == :initial_tangent_unresolved
    @test isempty(singular.negative.points)
end

@testset "Continuation validates inputs and autonomous factories" begin
    for kwargs in (
        (; initial_step=0.0), (; minimum_step=-1.0), (; maximum_step=Inf),
        (; initial_step=0.001, minimum_step=0.01), (; max_steps=0),
        (; max_corrector_iters=1.5), (; parameter_difference_step=NaN),
        (; state_scales=(1.0, 0.0)), (; state_scales=[1.0, 1.0]),
        (; parameter_scale=-2.0), (; rank_rtol=true), (; max_steps=true),
    )
        @test_throws ArgumentError ContinuationOptions(; kwargs...)
    end
    for bounds in ((1.0, -1.0), (0.0, 0.0), (0.0, Inf), (1.0, 2.0))
        @test_throws ArgumentError continue_equilibria(continuation_fold_model,
            [0.465, 1 / 3], 0.0; parameter_bounds=bounds)
    end
    @test_throws ArgumentError continue_equilibria(continuation_fold_model,
        [0.465, 1 / 3], NaN; parameter_bounds=(-1.0, 1.0))
    @test_throws ArgumentError continue_equilibria(continuation_fold_model,
        [0.465], 0.0; parameter_bounds=(-1.0, 1.0))
    @test_throws ArgumentError continue_equilibria(_ -> nothing,
        [0.465, 1 / 3], 0.0; parameter_bounds=(-1.0, 1.0))
    pulsed_factory = parameter -> begin
        model = continuation_fold_model(parameter)
        PointModelParameters(excitatory=model.excitatory, inhibitory=model.inhibitory,
            coupling=model.coupling, drive=PiecewiseConstantDrive(
                baseline=(0.0, 0.0),
                pulses=(DrivePulse(onset=0.0, offset=1.0, increment=(1.0, 0.0)),),
                interpretation=AfferentExcitation))
    end
    @test_throws ArgumentError continue_equilibria(pulsed_factory,
        [0.465, 1 / 3], 0.0; parameter_bounds=(-1.0, 1.0))
end
