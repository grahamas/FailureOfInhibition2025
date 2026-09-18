@testset "Typed point model" begin
    model = synthetic_model()
    @test model.excitatory isa PopulationParameters
    @test model.inhibitory isa PopulationParameters
    @test model.coupling.e_to_e == 1.2
    @test model.coupling.i_to_e == -0.6
    @test model.coupling.e_to_i == 0.7
    @test model.coupling.i_to_i == -0.4

    state = [0.2, 0.3]
    derivative = zeros(2)
    point_rhs!(derivative, state, model, 0.0)

    excitatory_input = 1.2 * state[1] - 0.6 * state[2]
    inhibitory_input = 0.7 * state[1] - 0.4 * state[2]
    expected_excitatory = (
        -0.7 * state[1] +
        1.1 * (1.0 - state[1]) * response(model.excitatory.response, excitatory_input)
    ) / 2.0
    expected_inhibitory = (
        -0.9 * state[2] +
        0.8 * (1.0 - state[2]) * response(model.inhibitory.response, inhibitory_input)
    ) / 1.5
    @test derivative ≈ [expected_excitatory, expected_inhibitory]

    drive = PiecewiseConstantDrive(
        baseline=(0.1, -0.2),
        amplitude=(0.5, 0.7),
        windows=[(1.0, 2.0), (4.0, 5.0)],
    )
    @test drive_value(drive, 0.0) == (0.1, -0.2)
    @test all(isapprox.(drive_value(drive, 1.0), (0.6, 0.5)))
    @test all(isapprox.(drive_value(drive, 2.0), (0.6, 0.5)))
    @test drive_value(drive, 3.0) == (0.1, -0.2)
    @test drive_value(NoDrive(), 3.0) == (0.0, 0.0)

    driven_model = synthetic_model(drive=drive)
    before = zeros(2)
    during = zeros(2)
    point_rhs!(before, state, driven_model, 0.0)
    point_rhs!(during, state, driven_model, 1.5)
    @test before != during

    @test_throws ArgumentError point_rhs!(zeros(1, 2), state, model, 0.0)
    @test_throws ArgumentError point_rhs!(zeros(2), reshape(state, 1, 2), model, 0.0)
    @test_throws ArgumentError point_rhs!(zeros(3), [0.1, 0.2, 0.3], model, 0.0)
    @test_throws ArgumentError PiecewiseConstantDrive(amplitude=(1.0, 1.0), windows=[(2.0, 1.0)])
    @test_throws ArgumentError PiecewiseConstantDrive(amplitude=(1.0, Inf), windows=[])
    @test_throws ArgumentError PiecewiseConstantDrive(amplitude=1.0, windows=[])
    @test_throws ArgumentError PiecewiseConstantDrive(amplitude=(1.0, 1.0), windows=[(1.0,)])

    generated_windows = ((start, start + 0.5) for start in (1.0, 3.0))
    generated_drive = PiecewiseConstantDrive(amplitude=(0.2, 0.4), windows=generated_windows)
    @test drive_value(generated_drive, 1.25) == (0.2, 0.4)
    @test drive_value(generated_drive, 2.0) == (0.0, 0.0)
    @test_throws ArgumentError PopulationParameters(
        decay=1.0,
        saturation=1.0,
        timescale=0.0,
        response=LogisticResponse(slope=1.0, threshold=0.0),
    )
    @test_throws TypeError PopulationParameters(
        decay=1.0,
        saturation=1.0,
        timescale=1.0,
        response=identity,
    )
    @test_throws TypeError PointModelParameters(
        excitatory=model.excitatory,
        inhibitory=model.inhibitory,
        coupling=model.coupling,
        drive=identity,
    )
end
