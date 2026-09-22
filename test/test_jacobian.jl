function finite_difference_jacobian(model, state, time; step=1.0e-7)
    jacobian = zeros(2, 2)
    plus = zeros(2)
    minus = zeros(2)
    for column in 1:2
        upper = copy(state)
        lower = copy(state)
        upper[column] += step
        lower[column] -= step
        point_rhs!(plus, upper, model, time)
        point_rhs!(minus, lower, model, time)
        jacobian[:, column] .= (plus .- minus) ./ (2step)
    end
    return jacobian
end

@testset "Jacobian scales before output conversion" begin
    flat_response = LogisticResponse(slope=1.0, threshold=0.0)
    model = synthetic_model(
        excitatory_response=flat_response,
        inhibitory_response=flat_response,
        excitatory_timescale=1.0e20,
        inhibitory_timescale=2.0e20,
        coupling=PointCoupling(1.0e40, 1.0e40, 1.0e40, 1.0e40),
    )
    state = zeros(2)
    jacobian = zeros(Float32, 2, 2)
    balance_jacobian = zeros(2, 2)
    expected = Float32[
        (-1.5 + 0.25e40) / 1.0e20 -0.25e40 / 1.0e20
        0.25e40 / 2.0e20 (-1.5 - 0.25e40) / 2.0e20
    ]
    @test point_jacobian!(jacobian, state, model, 0.0) === jacobian
    @test all(isfinite, jacobian)
    @test jacobian ≈ expected rtol=eps(Float32)
    @test point_balance_jacobian!(balance_jacobian, state, model, 0.0) === balance_jacobian
    @test all(>(floatmax(Float32)), abs.(balance_jacobian))

    tail_response = LogisticResponse(slope=1.0, threshold=110.0)
    model = synthetic_model(
        excitatory_response=tail_response,
        inhibitory_response=tail_response,
        excitatory_timescale=1.0e-40,
        inhibitory_timescale=2.0e-40,
        coupling=PointCoupling(0.0, 1.0, 1.0, 0.0),
    )
    tail_slope = exp(-110.0) / (1 + exp(-110.0))^2
    point_jacobian!(jacobian, state, model, 0.0)
    @test jacobian[1, 2] ≈ Float32(-tail_slope / 1.0e-40) rtol=eps(Float32)
    @test jacobian[2, 1] ≈ Float32(tail_slope / 2.0e-40) rtol=eps(Float32)
    @test jacobian[1, 2] < 0 < jacobian[2, 1]
end

function manual_point_jacobian(model, state, time)
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
    excitatory_slope = response_derivative(model.excitatory.response, excitatory_input)
    inhibitory_slope = response_derivative(model.inhibitory.response, inhibitory_input)

    j11 = (-1 - excitatory_rate +
           (1 - excitatory_activity) * excitatory_slope * coupling.e_to_e) /
          model.excitatory.timescale
    j12 = (-(1 - excitatory_activity) * excitatory_slope * coupling.i_to_e) /
          model.excitatory.timescale
    j21 = ((1 - inhibitory_activity) * inhibitory_slope * coupling.e_to_i) /
          model.inhibitory.timescale
    j22 = (-1 - inhibitory_rate -
           (1 - inhibitory_activity) * inhibitory_slope * coupling.i_to_i) /
          model.inhibitory.timescale
    return [j11 j12; j21 j22]
end

@testset "Jacobian mixed and generic numeric types" begin
    for model in (synthetic_model(), synthetic_foi_model())
        for state in (Float32[0.2, 0.3], [0.2, 0.3], BigFloat[0.2, 0.3])
            expected = manual_point_jacobian(model, state, 0.0)
            for T in (Float32, Float64, BigFloat)
                jacobian = zeros(T, 2, 2)
                point_jacobian!(jacobian, state, model, 0.0)
                @test jacobian ≈ T.(expected) rtol=8eps(T)
            end
        end
    end
end

@testset "Analytical Jacobian" begin
    drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=[
            DrivePulse(onset=1.0, offset=3.0, increment=(0.4, 0.1)),
            DrivePulse(onset=1.5, offset=2.5, increment=(0.2, 0.3)),
        ],
        interpretation=AfferentExcitation,
    )
    matched = synthetic_matched_models(drive=drive)
    state = [0.4, 0.2]

    for model in (matched.control, matched.failure_of_inhibition)
        for time in (0.0, 1.0, 2.0, 2.5, 3.0)
            analytical = zeros(2, 2)
            @test point_jacobian!(analytical, state, model, time) === analytical
            @test analytical ≈ manual_point_jacobian(model, state, time)
            @test analytical ≈ finite_difference_jacobian(model, state, time) atol = 1.0e-7 rtol = 1.0e-6
        end
    end

    outside_state = [-0.15, 1.2]
    outside_jacobian = zeros(2, 2)
    point_jacobian!(outside_jacobian, outside_state, matched.control, 2.0)
    @test outside_jacobian ≈ manual_point_jacobian(matched.control, outside_state, 2.0)

    model = matched.control
    @test_throws ArgumentError point_jacobian!(zeros(4), state, model, 0.0)
    @test_throws ArgumentError point_jacobian!(zeros(3, 3), state, model, 0.0)
    @test_throws ArgumentError point_jacobian!(zeros(2, 2), reshape(state, 1, 2), model, 0.0)
end

@testset "Failure-response maximum Jacobian" begin
    slope = 1.5
    onset_threshold = 0.2
    failure_threshold = 1.0
    midpoint = (onset_threshold + failure_threshold) / 2
    state = [0.2, 0.3]
    coupling = PointCoupling(e_to_e=1.2, i_to_e=0.6, e_to_i=0.7, i_to_i=0.4)
    inhibitory_baseline = midpoint - coupling.e_to_i * state[1] +
                          coupling.i_to_i * state[2]
    drive = PiecewiseConstantDrive(
        baseline=(0.0, inhibitory_baseline),
        pulses=DrivePulse[],
        interpretation=AfferentExcitation,
    )
    model = synthetic_model(
        inhibitory_response=FailureOfInhibitionResponse(
            slope=slope,
            onset_threshold=onset_threshold,
            failure_threshold=failure_threshold,
        ),
        coupling=coupling,
        drive=drive,
    )

    inhibitory_input = drive_value(drive, 0.0)[2] +
                       coupling.e_to_i * state[1] - coupling.i_to_i * state[2]
    @test inhibitory_input ≈ midpoint
    @test response_derivative(model.inhibitory.response, inhibitory_input) ≈ 0.0 atol = 10eps(Float64)

    jacobian = zeros(2, 2)
    point_jacobian!(jacobian, state, model, 0.0)
    inhibitory_rate = response(model.inhibitory.response, inhibitory_input)
    @test jacobian[2, 1] ≈ 0.0 atol = 10eps(Float64)
    @test jacobian[2, 2] ≈ (-1 - inhibitory_rate) / model.inhibitory.timescale atol = 10eps(Float64)
    @test jacobian ≈ finite_difference_jacobian(model, state, 0.0) atol = 1.0e-7 rtol = 1.0e-6
end
