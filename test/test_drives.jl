@testset "Point-model drives" begin
    @test drive_value(NoDrive(), 3.0) == (0.0, 0.0)

    endpoint_drive = PiecewiseConstantDrive(
        baseline=(0.25, 0.5),
        pulses=[DrivePulse(onset=1.0, offset=2.0, increment=(0.75, 1.5))],
        interpretation=AfferentExcitation,
    )
    @test drive_value(endpoint_drive, 0.999) == (0.25, 0.5)
    @test drive_value(endpoint_drive, 1.0) == (1.0, 2.0)
    @test drive_value(endpoint_drive, prevfloat(2.0)) == (1.0, 2.0)
    @test drive_value(endpoint_drive, 2.0) == (0.25, 0.5)
    @test endpoint_drive.pulses isa Tuple
    @test_throws MethodError push!(
        endpoint_drive.pulses,
        DrivePulse(onset=0.0, offset=1.0, increment=(-1.0, -2.0)),
    )

    pulses = [
        DrivePulse(onset=0.0, offset=3.0, increment=(1.0, 2.0)),
        DrivePulse(onset=1.0, offset=4.0, increment=(10.0, 20.0)),
        DrivePulse(onset=2.0, offset=3.0, increment=(100.0, 200.0)),
    ]
    overlap_drive = PiecewiseConstantDrive(
        baseline=(0.5, 1.0),
        pulses=pulses,
        interpretation=AfferentExcitation,
    )
    @test drive_value(overlap_drive, 0.5) == (1.5, 3.0)
    @test drive_value(overlap_drive, 1.5) == (11.5, 23.0)
    @test drive_value(overlap_drive, 2.5) == (111.5, 223.0)
    @test drive_value(overlap_drive, 3.0) == (10.5, 21.0)

    reversed_drive = PiecewiseConstantDrive(
        baseline=(0.5, 1.0),
        pulses=reverse(pulses),
        interpretation=AfferentExcitation,
    )
    for time in (-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 4.0)
        @test drive_value(overlap_drive, time) == drive_value(reversed_drive, time)
    end

    deduplicated_drive = PiecewiseConstantDrive(
        pulses=(
            DrivePulse(onset=0, offset=2, increment=(1, 0)),
            DrivePulse(onset=1, offset=2, increment=(0, 1)),
            DrivePulse(onset=2, offset=3, increment=(1, 1)),
        ),
        interpretation=AfferentExcitation,
    )
    @test FailureOfInhibition2025.drive_transition_times(deduplicated_drive) == [0, 1, 2, 3]
    @test isempty(FailureOfInhibition2025.drive_transition_times(NoDrive()))

    abstract_drive = PiecewiseConstantDrive(
        baseline=(-1.0, 0.5),
        pulses=[DrivePulse(onset=0.0, offset=1.0, increment=(-2.0, -1.0))],
        interpretation=AbstractIntervention,
    )
    @test drive_value(abstract_drive, -1.0) == (-1.0, 0.5)
    @test drive_value(abstract_drive, 0.5) == (-3.0, -0.5)

    nonnegative_with_decrement = PiecewiseConstantDrive(
        baseline=(1.0, 2.0),
        pulses=(
            DrivePulse(onset=0.0, offset=2.0, increment=(-0.75, -1.5)),
            DrivePulse(onset=1.0, offset=3.0, increment=(0.5, -0.25)),
        ),
        interpretation=AfferentExcitation,
    )
    @test drive_value(nonnegative_with_decrement, 0.5) == (0.25, 0.5)
    @test drive_value(nonnegative_with_decrement, 1.5) == (0.75, 0.25)

    invalid_segment = [
        DrivePulse(onset=0.0, offset=2.0, increment=(1.0, 0.0)),
        DrivePulse(onset=1.0, offset=3.0, increment=(-1.0, 0.0)),
    ]
    @test_throws ArgumentError PiecewiseConstantDrive(
        baseline=(0.5, 0.5),
        pulses=invalid_segment,
        interpretation=AfferentExcitation,
    )
    @test_throws ArgumentError PiecewiseConstantDrive(
        baseline=(-0.1, 0.5),
        pulses=DrivePulse[],
        interpretation=AfferentExcitation,
    )

    @test_throws ArgumentError DrivePulse(onset=1.0, offset=1.0, increment=(1.0, 1.0))
    @test_throws ArgumentError DrivePulse(onset=2.0, offset=1.0, increment=(1.0, 1.0))
    @test_throws ArgumentError DrivePulse(onset=Inf, offset=2.0, increment=(1.0, 1.0))
    @test_throws ArgumentError DrivePulse(onset=0.0, offset=2.0, increment=(1.0, NaN))
    @test_throws ArgumentError DrivePulse(onset=0.0, offset=2.0, increment=1.0)
    @test_throws ArgumentError PiecewiseConstantDrive(
        baseline=(0.0, Inf),
        pulses=DrivePulse[],
        interpretation=AbstractIntervention,
    )
    @test_throws ArgumentError PiecewiseConstantDrive(
        pulses=[(0.0, 1.0)],
        interpretation=AbstractIntervention,
    )
end
