function direct_rhs_values(model, state, time)
    excitatory_activity, inhibitory_activity = state
    excitatory_drive, inhibitory_drive = drive_value(model.drive, time)
    coupling = model.coupling
    excitatory_input = excitatory_drive +
                       coupling.e_to_e * excitatory_activity -
                       coupling.i_to_e * inhibitory_activity
    inhibitory_input = inhibitory_drive +
                       coupling.e_to_i * excitatory_activity -
                       coupling.i_to_i * inhibitory_activity
    excitatory_rate = response(model.excitatory.response, excitatory_input)
    inhibitory_rate = response(model.inhibitory.response, inhibitory_input)
    return [
        (-excitatory_activity + (1 - excitatory_activity) * excitatory_rate) /
        model.excitatory.timescale,
        (-inhibitory_activity + (1 - inhibitory_activity) * inhibitory_rate) /
        model.inhibitory.timescale,
    ]
end

@testset "Typed point-model parameters" begin
    model = synthetic_model()
    @test model.excitatory isa PopulationParameters
    @test model.inhibitory isa PopulationParameters
    @test model.excitatory.timescale == 2.0
    @test model.inhibitory.timescale == 1.5
    @test model.coupling.e_to_e == 1.2
    @test model.coupling.i_to_e == 0.6
    @test model.coupling.e_to_i == 0.7
    @test model.coupling.i_to_i == 0.4

    response_parameters = LogisticResponse(slope=1.0, threshold=0.0)
    @test PopulationParameters(timescale=1, response=response_parameters).timescale == 1
    for invalid_timescale in (0.0, -1.0, Inf, -Inf, NaN)
        @test_throws ArgumentError PopulationParameters(
            timescale=invalid_timescale,
            response=response_parameters,
        )
    end
    @test_throws ArgumentError PopulationParameters(
        timescale="1.0",
        response=response_parameters,
    )
    @test_throws MethodError PopulationParameters(1.0, response_parameters)
    @test_throws MethodError PopulationParameters(
        decay=1.0,
        saturation=1.0,
        timescale=1.0,
        response=response_parameters,
    )
    @test_throws MethodError PopulationParameters(timescale=1.0, response=identity)

    @test PointCoupling(1, 2.0, 3, 4).i_to_e == 2.0
    @test PointCoupling(e_to_e=0, i_to_e=0, e_to_i=0, i_to_i=0).i_to_i == 0
    for invalid_weight in (-1.0, Inf, -Inf, NaN)
        @test_throws ArgumentError PointCoupling(
            e_to_e=invalid_weight,
            i_to_e=0.2,
            e_to_i=0.3,
            i_to_i=0.4,
        )
        @test_throws ArgumentError PointCoupling(invalid_weight, 0.2, 0.3, 0.4)
    end
    @test_throws ArgumentError PointCoupling("1", 0.2, 0.3, 0.4)

    failure_response = FailureOfInhibitionResponse(
        slope=1.5,
        onset_threshold=0.2,
        failure_threshold=1.0,
    )
    @test synthetic_model(inhibitory_response=failure_response) isa PointModelParameters

    comparison_candidate = RectifiedZeroedLogisticResponse(slope=1.0, threshold=0.0)
    comparison_population = PopulationParameters(
        timescale=1.0,
        response=comparison_candidate,
    )
    failure_population = PopulationParameters(timescale=1.0, response=failure_response)
    @test_throws ArgumentError PointModelParameters(
        excitatory=comparison_population,
        inhibitory=model.inhibitory,
        coupling=model.coupling,
    )
    @test_throws ArgumentError PointModelParameters(
        excitatory=model.excitatory,
        inhibitory=comparison_population,
        coupling=model.coupling,
    )
    @test_throws ArgumentError PointModelParameters(
        failure_population,
        model.inhibitory,
        model.coupling,
        NoDrive(),
    )
    @test_throws TypeError PointModelParameters(
        excitatory=model.excitatory,
        inhibitory=model.inhibitory,
        coupling=model.coupling,
        drive=identity,
    )
end

@testset "Supported point-model equation" begin
    overlapping_drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=[
            DrivePulse(onset=1.0, offset=3.0, increment=(0.4, -0.1)),
            DrivePulse(onset=1.5, offset=2.5, increment=(0.2, 0.3)),
        ],
        interpretation=AbstractIntervention,
    )
    models = (
        synthetic_model(drive=overlapping_drive),
        synthetic_foi_model(drive=overlapping_drive),
    )
    state = [0.2, 0.3]

    for model in models, time in (0.0, 1.25, 2.0, 3.0)
        derivative = zeros(2)
        @test point_rhs!(derivative, state, model, time) === nothing
        @test derivative ≈ direct_rhs_values(model, state, time)
    end

    model = first(models)
    excitatory_input = 0.1 + 0.4 + 0.2 +
                       model.coupling.e_to_e * state[1] -
                       model.coupling.i_to_e * state[2]
    inhibitory_input = 0.2 - 0.1 + 0.3 +
                       model.coupling.e_to_i * state[1] -
                       model.coupling.i_to_i * state[2]
    expected = [
        (-state[1] + (1 - state[1]) * response(model.excitatory.response, excitatory_input)) /
        model.excitatory.timescale,
        (-state[2] + (1 - state[2]) * response(model.inhibitory.response, inhibitory_input)) /
        model.inhibitory.timescale,
    ]
    derivative = zeros(2)
    point_rhs!(derivative, state, model, 2.0)
    @test derivative ≈ expected

    outside_state = [-0.2, 1.3]
    outside_derivative = zeros(2)
    point_rhs!(outside_derivative, outside_state, model, 2.0)
    @test outside_derivative ≈ direct_rhs_values(model, outside_state, 2.0)
    @test outside_derivative != direct_rhs_values(model, clamp.(outside_state, 0.0, 1.0), 2.0)

    @test_throws ArgumentError point_rhs!(zeros(1, 2), state, model, 0.0)
    @test_throws ArgumentError point_rhs!(zeros(2), reshape(state, 1, 2), model, 0.0)
    @test_throws ArgumentError point_rhs!(zeros(3), [0.1, 0.2, 0.3], model, 0.0)
end

@testset "Physical-domain boundary directions" begin
    for model in (synthetic_model(), synthetic_foi_model())
        derivative = zeros(2)

        point_rhs!(derivative, [0.0, 0.3], model, 0.0)
        @test derivative[1] > 0

        point_rhs!(derivative, [0.2, 0.0], model, 0.0)
        @test derivative[2] > 0

        point_rhs!(derivative, [1.0, 0.3], model, 0.0)
        @test derivative[1] == -1 / model.excitatory.timescale

        point_rhs!(derivative, [0.2, 1.0], model, 0.0)
        @test derivative[2] == -1 / model.inhibitory.timescale

        point_rhs!(derivative, [0.5, 0.3], model, 0.0)
        @test derivative[1] <= 0
    end

    slope = 1.5
    onset_threshold = 0.2
    failure_threshold = 1.0
    maximum = tanh(slope * (failure_threshold - onset_threshold) / 4)
    inhibitory_bound = maximum / (1 + maximum)

    failure_model = synthetic_foi_model()
    derivative = zeros(2)
    point_rhs!(derivative, [0.2, inhibitory_bound], failure_model, 0.0)
    @test derivative[2] <= 10eps(Float64)

    midpoint = (onset_threshold + failure_threshold) / 2
    maximum_drive = PiecewiseConstantDrive(
        baseline=(0.0, midpoint),
        pulses=DrivePulse[],
        interpretation=AfferentExcitation,
    )
    maximum_model = synthetic_model(
        inhibitory_response=FailureOfInhibitionResponse(
            slope=slope,
            onset_threshold=onset_threshold,
            failure_threshold=failure_threshold,
        ),
        coupling=PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0),
        drive=maximum_drive,
    )
    point_rhs!(derivative, [0.2, inhibitory_bound], maximum_model, 0.0)
    @test derivative[2] ≈ 0.0 atol=10eps(Float64)
end
