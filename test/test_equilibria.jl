using LinearAlgebra: Diagonal, I

independent_logistic(slope, threshold, input) =
    inv(1 + exp(-slope * (input - threshold)))

function independent_response(response_parameters::LogisticResponse, input)
    return independent_logistic(
        response_parameters.slope,
        response_parameters.threshold,
        input,
    )
end

function independent_response(response_parameters::FailureOfInhibitionResponse, input)
    return independent_logistic(
        response_parameters.slope,
        response_parameters.onset_threshold,
        input,
    ) - independent_logistic(
        response_parameters.slope,
        response_parameters.failure_threshold,
        input,
    )
end

function independent_balance(model, state, time)
    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(model.drive, time)
    coupling = model.coupling
    excitatory_input = excitatory_drive + coupling.e_to_e * excitatory_activity -
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive + coupling.e_to_i * excitatory_activity -
                       coupling.i_to_i * inhibitory_activity
    excitatory_rate = independent_response(model.excitatory.response, excitatory_input)
    inhibitory_rate = independent_response(model.inhibitory.response, inhibitory_input)
    return [
        -excitatory_activity + (1 - excitatory_activity) * excitatory_rate,
        -inhibitory_activity + (1 - inhibitory_activity) * inhibitory_rate,
    ]
end

function finite_difference_balance_jacobian(model, state, time; step=1.0e-7)
    jacobian = zeros(2, 2)
    for column in 1:2
        upper = copy(state)
        lower = copy(state)
        upper[column] += step
        lower[column] -= step
        jacobian[:, column] .= (
            independent_balance(model, upper, time) .-
            independent_balance(model, lower, time)
        ) ./ (2step)
    end
    return jacobian
end

function zero_coupling_model(; inhibitory_response, timescales=(2.0, 5.0), drive)
    return PointModelParameters(
        excitatory=PopulationParameters(
            timescale=timescales[1],
            response=LogisticResponse(slope=2.0, threshold=0.3),
        ),
        inhibitory=PopulationParameters(
            timescale=timescales[2],
            response=inhibitory_response,
        ),
        coupling=PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
        drive=drive,
    )
end

function three_root_model(; timescales=(1.0, 1.0))
    return PointModelParameters(
        excitatory=PopulationParameters(
            timescale=timescales[1],
            response=LogisticResponse(slope=1.0, threshold=3 + log(2)),
        ),
        inhibitory=PopulationParameters(
            timescale=timescales[2],
            response=LogisticResponse(slope=1.0, threshold=0.0),
        ),
        coupling=PointCoupling(e_to_e=12.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
    )
end

@testset "Balance residual and analytical Jacobian" begin
    drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=(),
        interpretation=AfferentExcitation,
    )
    state = [0.31, 0.22]
    for model in (
        synthetic_model(drive=drive, excitatory_timescale=1.0e-12, inhibitory_timescale=1.0e12),
        synthetic_foi_model(drive=drive),
    )
        balance = zeros(2)
        derivative = zeros(2)
        balance_jacobian = zeros(2, 2)
        ode_jacobian = zeros(2, 2)
        @test point_balance!(balance, state, model, 0.0) === nothing
        point_rhs!(derivative, state, model, 0.0)
        @test balance ≈ independent_balance(model, state, 0.0)
        @test derivative ≈ balance ./
              [model.excitatory.timescale, model.inhibitory.timescale]

        @test point_balance_jacobian!(balance_jacobian, state, model, 0.0) ===
              balance_jacobian
        point_jacobian!(ode_jacobian, state, model, 0.0)
        @test balance_jacobian ≈ finite_difference_balance_jacobian(model, state, 0.0) atol=1.0e-7 rtol=1.0e-6
        @test balance_jacobian ≈
              Diagonal([model.excitatory.timescale, model.inhibitory.timescale]) *
              ode_jacobian
    end

    @test_throws ArgumentError point_balance!(zeros(3), state, synthetic_model(), 0.0)
    @test_throws ArgumentError point_balance_jacobian!(zeros(3, 3), state, synthetic_model(), 0.0)
end

@testset "Analytic zero-coupling equilibria" begin
    drive = PiecewiseConstantDrive(
        baseline=(0.7, 0.6),
        pulses=(),
        interpretation=AfferentExcitation,
    )
    control_response = LogisticResponse(slope=1.5, threshold=0.2)
    failure_response = FailureOfInhibitionResponse(
        slope=1.5,
        onset_threshold=0.2,
        failure_threshold=1.0,
    )
    for inhibitory_response in (control_response, failure_response)
        model = zero_coupling_model(
            inhibitory_response=inhibitory_response,
            drive=drive,
        )
        result = solve_equilibrium(model, [0.2, 0.2])
        excitatory_rate = independent_logistic(2.0, 0.3, 0.7)
        inhibitory_rate = independent_response(inhibitory_response, 0.6)
        expected = [
            excitatory_rate / (1 + excitatory_rate),
            inhibitory_rate / (1 + inhibitory_rate),
        ]
        expected_jacobian = [
            -(1 + excitatory_rate) / 2.0 0.0
            0.0 -(1 + inhibitory_rate) / 5.0
        ]
        @test result.attempt.solver_status == :Success
        @test result.attempt.solver_success
        @test result.attempt.validation == AdmissibleCandidate
        @test result.attempt.candidate ≈ expected atol=1.0e-11
        @test result.stability.jacobian ≈ expected_jacobian atol=1.0e-11
        @test result.stability.classification == Attracting
        @test result.frozen_drive == (0.7, 0.6)
        @test isnothing(result.source_time)
    end

    no_drive_model = zero_coupling_model(
        inhibitory_response=control_response,
        drive=NoDrive(),
    )
    no_drive_result = solve_equilibrium(no_drive_model, [0.2, 0.2])
    @test no_drive_result.attempt.validation == AdmissibleCandidate
    @test no_drive_result.frozen_drive == (0.0, 0.0)

    reference_state = first(
        solve_equilibrium(
            zero_coupling_model(
                inhibitory_response=control_response,
                timescales=(1.0, 1.0),
                drive=drive,
            ),
            [0.2, 0.2],
        ).attempt.candidate,
        2,
    )
    for timescales in ((1.0e-12, 1.0e12), (1.0e12, 1.0e-12))
        separated = solve_equilibrium(
            zero_coupling_model(
                inhibitory_response=control_response,
                timescales=timescales,
                drive=drive,
            ),
            [0.2, 0.2],
        )
        @test separated.attempt.validation == AdmissibleCandidate
        @test separated.attempt.candidate ≈ reference_state atol=1.0e-11
    end
end

@testset "Deterministic multiple-root discovery" begin
    model = three_root_model()
    expected_states = [
        [0.03536009083997240946, 1 / 3],
        [0.25, 1 / 3],
        [0.46463990916002759054, 1 / 3],
    ]
    seeds = [[e, i] for (e, i) in (
        (0.01, 0.2),
        (0.05, 0.4),
        (0.20, 0.3),
        (0.25, 1 / 3),
        (0.30, 0.35),
        (0.45, 0.2),
        (0.49, 0.4),
        (0.25, 1 / 3),
    )]
    original_seeds = deepcopy(seeds)
    result = find_equilibria(model; seeds=seeds)
    permuted = find_equilibria(model; seeds=reverse(seeds))

    @test seeds == original_seeds
    @test length(result.attempts) == length(seeds)
    @test length(result.equilibria) == 3
    @test result.completeness == CompletenessNotCertified
    @test isempty(result.unresolved_nearby)
    @test [equilibrium.stability.classification for equilibrium in result.equilibria] ==
          [Attracting, Saddle, Attracting]
    for (equilibrium, expected) in zip(result.equilibria, expected_states)
        @test equilibrium.state ≈ expected atol=1.0e-10
        @test equilibrium.state == result.attempts[equilibrium.representative_attempt].candidate
        @test equilibrium.representative_attempt in equilibrium.member_attempts
    end
    for (first_result, second_result) in zip(result.equilibria, permuted.equilibria)
        @test first_result.state ≈ second_result.state atol=1.0e-12
        @test first_result.stability.classification == second_result.stability.classification
    end
    @test sum(length, getfield.(result.equilibria, :member_attempts)) == length(seeds)

    default_result = find_equilibria(model)
    @test length(default_result.equilibria) == 3
    for (equilibrium, expected) in zip(default_result.equilibria, expected_states)
        @test equilibrium.state ≈ expected atol=1.0e-9
    end

    chained_seeds = [
        expected_states[1],
        expected_states[2],
        expected_states[3],
        expected_states[1],
        expected_states[2],
    ]
    chained = find_equilibria(
        model;
        seeds=chained_seeds,
        options=EquilibriumOptions(dedup_atol=0.23),
    )
    @test length(chained.unresolved_nearby) == 1
    @test length(only(chained.unresolved_nearby)) == length(chained_seeds)
    @test length(chained.equilibria) == 2
    @test sum(length, getfield.(chained.equilibria, :member_attempts)) ==
          length(chained_seeds)
    @test any(length(equilibrium.member_attempts) > 1 for equilibrium in chained.equilibria)


    nearby_model = PointModelParameters(
        excitatory=PopulationParameters(
            timescale=1.0,
            response=LogisticResponse(
                slope=1.0,
                threshold=3.2780552699158847,
            ),
        ),
        inhibitory=model.inhibitory,
        coupling=model.coupling,
    )
    nearby_roots = [
        0.10550741520369928,
        0.10581761686817375,
        0.4799259356049484,
    ]
    nearby_result = find_equilibria(
        nearby_model;
        seeds=[[root, 1 / 3] for root in nearby_roots],
        options=EquilibriumOptions(dedup_atol=1.0e-5),
    )
    @test length(nearby_result.equilibria) == 3
    @test [equilibrium.state[1] for equilibrium in nearby_result.equilibria] ≈
          nearby_roots atol=1.0e-10
end

@testset "Frozen drive snapshots" begin
    drive = PiecewiseConstantDrive(
        baseline=(-0.2, 0.1),
        pulses=(
            DrivePulse(onset=1.0, offset=3.0, increment=(0.7, -0.4)),
            DrivePulse(onset=2.0, offset=4.0, increment=(-0.3, 0.8)),
        ),
        interpretation=AbstractIntervention,
    )
    model = zero_coupling_model(
        inhibitory_response=LogisticResponse(slope=1.5, threshold=0.2),
        drive=drive,
    )
    snapshots = (
        (-1.0, (-0.2, 0.1)),
        (1.0, (0.5, -0.3)),
        (2.0, (0.2, 0.5)),
        (3.0, (-0.5, 0.9)),
        (4.0, (-0.2, 0.1)),
    )
    for (time, frozen_drive) in snapshots
        result = solve_equilibrium(model, [0.2, 0.2]; snapshot_time=time)
        excitatory_rate = independent_logistic(2.0, 0.3, frozen_drive[1])
        inhibitory_rate = independent_logistic(1.5, 0.2, frozen_drive[2])
        expected = [
            excitatory_rate / (1 + excitatory_rate),
            inhibitory_rate / (1 + inhibitory_rate),
        ]
        @test all(isapprox.(result.frozen_drive, frozen_drive))
        @test result.source_time == time
        @test result.frozen_model.excitatory === model.excitatory
        @test result.frozen_model.inhibitory === model.inhibitory
        @test result.frozen_model.coupling === model.coupling
        @test result.attempt.candidate ≈ expected atol=1.0e-10
    end
    @test_throws ArgumentError solve_equilibrium(model, [0.2, 0.2])
    @test_throws ArgumentError find_equilibria(model)
    @test_throws ArgumentError solve_equilibrium(model, [0.2, 0.2]; snapshot_time=Inf)
end

@testset "Validation, failure retention, and storage" begin
    @test_throws ArgumentError EquilibriumOptions(residual_atol=-1.0)
    @test_throws ArgumentError EquilibriumOptions(domain_atol=Inf)
    @test_throws ArgumentError EquilibriumOptions(maxiters=0)

    model = synthetic_model()
    @test_throws ArgumentError solve_equilibrium(model, (0.1, 0.2))
    @test_throws ArgumentError solve_equilibrium(model, [0.1])
    @test_throws ArgumentError solve_equilibrium(model, [0.1, "bad"])
    @test_throws ArgumentError solve_equilibrium(model, [NaN, 0.2])
    @test_throws ArgumentError solve_equilibrium(model, BigFloat[0.1, 0.2])

    integer_result = solve_equilibrium(model, [0, 0])
    @test integer_result.attempt.seed isa Vector{Float64}

    float32_model = synthetic_model(
        excitatory_response=LogisticResponse(slope=2.0f0, threshold=0.1f0),
        inhibitory_response=LogisticResponse(slope=1.5f0, threshold=0.2f0),
        excitatory_timescale=2.0f0,
        inhibitory_timescale=1.5f0,
        coupling=PointCoupling(
            e_to_e=1.2f0,
            i_to_e=0.6f0,
            e_to_i=0.7f0,
            i_to_i=0.4f0,
        ),
    )
    float32_options = EquilibriumOptions(
        solver_abstol=1.0f-6,
        solver_reltol=1.0f-5,
        residual_atol=1.0f-5,
        domain_atol=1.0f-6,
        dedup_atol=1.0f-5,
        singular_atol=1.0f-6,
        singular_rtol=1.0f-5,
    )
    float32_result = solve_equilibrium(
        float32_model,
        Float32[0.1, 0.2];
        options=float32_options,
        stability_options=StabilityOptions(
            spectral_atol=1.0f-6,
            spectral_rtol=1.0f-5,
        ),
    )
    @test float32_result.attempt.seed isa Vector{Float32}
    @test float32_result.attempt.candidate isa Vector{Float32}
    @test float32_result.stability.jacobian isa Matrix{Float32}

    parent_seed = [9.0, 0.1, 0.2, 9.0]
    seed_view = @view parent_seed[2:3]
    view_copy = copy(parent_seed)
    view_result = solve_equilibrium(model, seed_view)
    @test parent_seed == view_copy
    @test axes(view_result.attempt.seed) == (Base.OneTo(2),)
    @test view_result.attempt.seed !== seed_view

    outside_trial = solve_equilibrium(model, [10.0, 10.0]; options=EquilibriumOptions(maxiters=1))
    @test outside_trial.attempt.solver_status == :MaxIters
    @test !outside_trial.attempt.solver_success
    @test length(outside_trial.attempt.candidate) == 2

    empty_result = find_equilibria(model; seeds=Vector{Float64}[])
    @test isempty(empty_result.attempts)
    @test isempty(empty_result.equilibria)
    @test empty_result.completeness == CompletenessNotCertified

    context = FailureOfInhibition2025._frozen_point_context(model, nothing)
    nonfinite = FailureOfInhibition2025._validate_candidate(
        [0.1, 0.2],
        [NaN, 0.2],
        :Synthetic,
        false,
        [NaN, 0.0],
        context.frozen_model,
        EquilibriumOptions(),
    )
    @test nonfinite.validation == RejectedCandidate
    @test :nonfinite_candidate in nonfinite.reasons

    slow_model = zero_coupling_model(
        inhibitory_response=LogisticResponse(slope=1.0, threshold=0.0),
        timescales=(1.0e300, 1.0e300),
        drive=NoDrive(),
    )
    slow_derivative = zeros(2)
    point_rhs!(slow_derivative, [0.49, 0.49], slow_model, 0.0)
    @test maximum(abs, slow_derivative) < 1.0e-299
    large_balance = FailureOfInhibition2025._validate_candidate(
        [0.49, 0.49],
        [0.49, 0.49],
        :Synthetic,
        true,
        zeros(2),
        slow_model,
        EquilibriumOptions(),
    )
    @test large_balance.validation == RejectedCandidate
    @test :large_balance_residual in large_balance.reasons
end

@testset "Boundary ambiguity and deduplication policy" begin
    tail_response = LogisticResponse(slope=1.0, threshold=1000.0)
    tail_model = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0, response=tail_response),
        inhibitory=PopulationParameters(timescale=1.0, response=tail_response),
        coupling=PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
    )
    boundary_options = EquilibriumOptions(residual_atol=1.0e-7, domain_atol=1.0e-8)
    ambiguous = FailureOfInhibition2025._validate_candidate(
        [-5.0e-9, 0.0],
        [-5.0e-9, 0.0],
        :Synthetic,
        true,
        zeros(2),
        tail_model,
        boundary_options,
    )
    outside = FailureOfInhibition2025._validate_candidate(
        [-2.0e-8, 0.0],
        [-2.0e-8, 0.0],
        :Synthetic,
        true,
        zeros(2),
        tail_model,
        boundary_options,
    )
    @test ambiguous.validation == BoundaryAmbiguousCandidate
    @test ambiguous.candidate[1] == -5.0e-9
    @test outside.validation == RejectedCandidate
    @test :outside_physical_domain in outside.reasons

    saturated_response = LogisticResponse(slope=1.0, threshold=-1000.0)
    saturated_model = PointModelParameters(
        excitatory=PopulationParameters(timescale=1.0, response=saturated_response),
        inhibitory=PopulationParameters(timescale=1.0, response=saturated_response),
        coupling=PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
    )
    upper_ambiguous = FailureOfInhibition2025._validate_candidate(
        [0.5 + 5.0e-9, 0.5],
        [0.5 + 5.0e-9, 0.5],
        :Synthetic,
        true,
        zeros(2),
        saturated_model,
        boundary_options,
    )
    upper_outside = FailureOfInhibition2025._validate_candidate(
        [0.5 + 2.0e-8, 0.5],
        [0.5 + 2.0e-8, 0.5],
        :Synthetic,
        true,
        zeros(2),
        saturated_model,
        boundary_options,
    )
    @test upper_ambiguous.validation == BoundaryAmbiguousCandidate
    @test upper_outside.validation == RejectedCandidate
    @test :outside_equilibrium_bounds in upper_outside.reasons

    function synthetic_attempt(state)
        return EquilibriumAttempt(
            copy(state),
            copy(state),
            :Synthetic,
            true,
            zeros(2),
            zeros(2),
            0.0,
            Matrix{Float64}(I, 2, 2),
            false,
            AdmissibleCandidate,
            Symbol[],
        )
    end
    attempts = [
        synthetic_attempt([0.10, 0.20]),
        synthetic_attempt([0.175, 0.20]),
        synthetic_attempt([0.25, 0.20]),
        synthetic_attempt([0.10, 0.20]),
    ]
    component = only(
        FailureOfInhibition2025._connected_candidate_components(
            attempts,
            collect(eachindex(attempts)),
            0.1,
        ),
    )
    @test component == [1, 2, 3, 4]
    @test FailureOfInhibition2025._component_diameter(attempts, component) > 0.1
    groups = FailureOfInhibition2025._partition_complete_linkage(
        attempts,
        component,
        0.1,
    )
    @test length(groups) == 2
    @test sort!(vcat(groups...)) == component
    @test any(Set(group) == Set([1, 2, 4]) for group in groups)
end

@testset "Default coverage and matched-pair API usage demonstration — not a scientific result" begin
    control_seeds = default_equilibrium_seeds(synthetic_model())
    @test length(control_seeds) == 25
    @test first(control_seeds) == [0.0, 0.0]
    @test last(control_seeds) == [0.5, 0.5]

    failure_model = synthetic_foi_model()
    failure_seeds = default_equilibrium_seeds(failure_model)
    response_parameters = failure_model.inhibitory.response
    maximum_response = tanh(
        response_parameters.slope *
        (response_parameters.failure_threshold - response_parameters.onset_threshold) / 4,
    )
    expected_upper = maximum_response / (1 + maximum_response)
    @test maximum(seed[1] for seed in failure_seeds) == 0.5
    @test maximum(seed[2] for seed in failure_seeds) ≈ expected_upper

    models = synthetic_matched_models()
    seeds = [[0.0, 0.0], [0.25, 0.2], [0.5, 0.4]]
    control = find_equilibria(models.control; seeds=seeds)
    failure = find_equilibria(models.failure_of_inhibition; seeds=seeds)
    @test control.completeness == failure.completeness == CompletenessNotCertified
    @test models.control.excitatory === models.failure_of_inhibition.excitatory
    @test models.control.coupling === models.failure_of_inhibition.coupling
    @test !isempty(control.equilibria)
    @test !isempty(failure.equilibria)
end
