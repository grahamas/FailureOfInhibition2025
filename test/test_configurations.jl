@testset "Matched point-model configurations" begin
    excitatory_response = LogisticResponse(slope=2.0, threshold=0.1)
    inhibitory_response = LogisticResponse(slope=1.5, threshold=0.2)
    excitatory = PopulationParameters(timescale=2.0, response=excitatory_response)
    inhibitory_control =
        PopulationParameters(timescale=1.5, response=inhibitory_response)
    coupling = PointCoupling(
        e_to_e=1.2,
        i_to_e=0.6,
        e_to_i=0.7,
        i_to_i=0.4,
    )
    drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=[DrivePulse(onset=1.0, offset=2.0, increment=(0.3, 0.4))],
        interpretation=AfferentExcitation,
    )
    original_pulses = drive.pulses

    models = matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory_control,
        failure_threshold=1.1,
        coupling=coupling,
        drive=drive,
    )

    @test keys(models) == (:control, :failure_of_inhibition)
    @test models.control.excitatory === excitatory
    @test models.failure_of_inhibition.excitatory === excitatory
    @test models.control.inhibitory === inhibitory_control
    @test models.control.coupling === coupling
    @test models.failure_of_inhibition.coupling === coupling
    @test models.control.drive === drive
    @test models.failure_of_inhibition.drive === drive

    inhibitory_failure = models.failure_of_inhibition.inhibitory
    @test inhibitory_failure !== inhibitory_control
    @test inhibitory_failure.timescale == inhibitory_control.timescale
    @test inhibitory_failure.response isa FailureOfInhibitionResponse
    @test inhibitory_failure.response.slope == inhibitory_response.slope
    @test inhibitory_failure.response.onset_threshold == inhibitory_response.threshold
    @test inhibitory_failure.response.failure_threshold == 1.1

    failing_logistic = LogisticResponse(
        slope=inhibitory_response.slope,
        threshold=inhibitory_failure.response.failure_threshold,
    )
    for input in (-10.0, -0.5, 0.2, 0.65, 1.1, 3.0, 10.0)
        expected = response(inhibitory_response, input) - response(failing_logistic, input)
        @test response(inhibitory_failure.response, input) ≈ expected atol=2eps(Float64)
    end

    @test excitatory.response === excitatory_response
    @test inhibitory_control.response === inhibitory_response
    @test drive.pulses == original_pulses
    @test drive.baseline == (0.1, 0.2)
end

@testset "Matched model RHS comparison" begin
    excitatory = PopulationParameters(
        timescale=2.0,
        response=LogisticResponse(slope=2.0, threshold=0.1),
    )
    inhibitory_control = PopulationParameters(
        timescale=1.5,
        response=LogisticResponse(slope=1.5, threshold=0.2),
    )
    coupling = PointCoupling(e_to_e=0.0, i_to_e=0.0, e_to_i=0.0, i_to_i=0.0)
    state = [0.25, 0.25]

    negligible_models = matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory_control,
        failure_threshold=100.0,
        coupling=coupling,
    )
    control_derivative = zeros(2)
    failure_derivative = zeros(2)
    point_rhs!(control_derivative, state, negligible_models.control, 0.0)
    point_rhs!(failure_derivative, state, negligible_models.failure_of_inhibition, 0.0)
    @test failure_derivative[1] == control_derivative[1]
    @test failure_derivative[2] ≈ control_derivative[2] atol=1.0e-14 rtol=0.0

    active_drive = PiecewiseConstantDrive(
        baseline=(0.0, 1.1),
        pulses=DrivePulse[],
        interpretation=AfferentExcitation,
    )
    active_models = matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory_control,
        failure_threshold=1.1,
        coupling=coupling,
        drive=active_drive,
    )
    point_rhs!(control_derivative, state, active_models.control, 0.0)
    point_rhs!(failure_derivative, state, active_models.failure_of_inhibition, 0.0)
    @test failure_derivative[1] == control_derivative[1]
    @test failure_derivative[2] < control_derivative[2]
end

@testset "Matched model validation" begin
    logistic = LogisticResponse(slope=1.5, threshold=0.2)
    excitatory = PopulationParameters(timescale=2.0, response=logistic)
    inhibitory_control = PopulationParameters(timescale=1.5, response=logistic)
    coupling = PointCoupling(e_to_e=1.2, i_to_e=0.6, e_to_i=0.7, i_to_i=0.4)
    failure_response = FailureOfInhibitionResponse(
        logistic;
        failure_threshold=1.0,
    )
    invalid_population = PopulationParameters(timescale=1.0, response=failure_response)

    @test_throws ArgumentError matched_point_models(
        excitatory=invalid_population,
        inhibitory_control=inhibitory_control,
        failure_threshold=1.0,
        coupling=coupling,
    )
    @test_throws ArgumentError matched_point_models(
        excitatory=excitatory,
        inhibitory_control=invalid_population,
        failure_threshold=1.0,
        coupling=coupling,
    )
    @test_throws ArgumentError matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory_control,
        failure_threshold=logistic.threshold,
        coupling=coupling,
    )
    @test_throws ArgumentError matched_point_models(
        excitatory=excitatory,
        inhibitory_control=inhibitory_control,
        failure_threshold=Inf,
        coupling=coupling,
    )
end
