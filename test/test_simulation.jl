using CSV
using SciMLBase

@testset "Deterministic CPU simulation and CSV output" begin
    model = synthetic_model()
    initial_state = [0.1, 0.15]
    options = (saveat=0.1, abstol=1.0e-10, reltol=1.0e-10)

    first_solution = solve_point_model(initial_state, (0.0, 2.0), model; options...)
    second_solution = solve_point_model(initial_state, (0.0, 2.0), model; options...)

    @test SciMLBase.successful_retcode(first_solution)
    @test first_solution.t == second_solution.t
    @test first_solution.u == second_solution.u
    @test all(state -> state isa AbstractVector && length(state) == 2, first_solution.u)

    mktempdir() do directory
        first_path = joinpath(directory, "first.csv")
        second_path = joinpath(directory, "second.csv")
        @test write_trajectory_csv(first_path, first_solution) == first_path
        @test write_trajectory_csv(second_path, second_solution) == second_path
        @test read(first_path) == read(second_path)

        rows = collect(CSV.File(first_path))
        @test propertynames(first(rows)) == [:time, :E, :I]
        @test length(rows) == length(first_solution.t)
        @test rows[1].time == first_solution.t[1]
        @test rows[1].E == first_solution.u[1][1]
        @test rows[1].I == first_solution.u[1][2]
    end

    @test_throws ArgumentError solve_point_model(reshape(initial_state, 1, 2), (0.0, 1.0), model)
    @test_throws ArgumentError solve_point_model(initial_state, (1.0, 0.0), model)
    @test_throws ArgumentError solve_point_model(initial_state, (0.0, Inf), model)
end

@testset "Simulation domain validation" begin
    model = synthetic_model()

    for initial_state in ([0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0])
        solution = solve_point_model(initial_state, (0.0, 0.1), model)
        @test SciMLBase.successful_retcode(solution)
    end
    integer_solution = solve_point_model([0, 0], (0.0, 0.1), model)
    @test SciMLBase.successful_retcode(integer_solution)
    @test eltype(first(integer_solution.u)) === Float64

    bigfloat_solution = solve_point_model(BigFloat[0, 0], (0.0, 0.1), model)
    @test SciMLBase.successful_retcode(bigfloat_solution)
    @test eltype(first(bigfloat_solution.u)) === BigFloat

    zero_tolerance_solution = solve_point_model(
        [0.0, 1.0],
        (0.0, 0.1),
        model;
        domain_atol=0.0,
    )
    @test SciMLBase.successful_retcode(zero_tolerance_solution)

    @test_throws ArgumentError solve_point_model([0.1, "not real"], (0.0, 1.0), model)
    @test_throws DomainError solve_point_model([NaN, 0.1], (0.0, 1.0), model)
    @test_throws DomainError solve_point_model([Inf, 0.1], (0.0, 1.0), model)
    @test_throws DomainError solve_point_model([-eps(), 0.1], (0.0, 1.0), model)
    @test_throws DomainError solve_point_model([1.0 + eps(), 0.1], (0.0, 1.0), model)

    @test_throws ArgumentError solve_point_model(
        [0.1, 0.2],
        (0.0, 1.0),
        model;
        domain_atol=-eps(),
    )
    @test_throws ArgumentError solve_point_model(
        [0.1, 0.2],
        (0.0, 1.0),
        model;
        domain_atol=Inf,
    )
    @test_throws ArgumentError solve_point_model(
        [0.1, 0.2],
        (0.0, 1.0),
        model;
        domain_atol=NaN,
    )
    @test_throws ArgumentError solve_point_model(
        [0.1, 0.2],
        (0.0, 1.0),
        model;
        domain_atol="wide",
    )

    atol = 1.0e-8
    @test !FailureOfInhibition2025._state_outside_domain([-atol, 1.0 + atol], atol)
    @test FailureOfInhibition2025._state_outside_domain(
        [-2atol, 1.0 + atol],
        atol,
    )
    @test FailureOfInhibition2025._state_outside_domain([0.2, NaN], atol)

    valid_solution = (t=[0.0, 1.0], u=[[0.0, 1.0], [-atol, 1.0 + atol]])
    @test isnothing(
        FailureOfInhibition2025._postcheck_solution_domain(valid_solution, atol),
    )
    invalid_solution = (t=[0.0, 1.0], u=[[0.0, 1.0], [0.2, 1.0 + 2atol]])
    @test_throws DomainError FailureOfInhibition2025._postcheck_solution_domain(
        invalid_solution,
        atol,
    )
end

@testset "Drive transitions are integrated and saved" begin
    drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=[DrivePulse(onset=0.35, offset=0.85, increment=(0.3, 0.1))],
        interpretation=AfferentExcitation,
    )
    model = synthetic_model(drive=drive)

    interval_solution = solve_point_model(
        [0.1, 0.15],
        (0.0, 1.0),
        model;
        saveat=0.5,
        tstops=[0.5, 0.85, 0.5],
    )
    @test interval_solution.t == [0.0, 0.35, 0.5, 0.85, 1.0]

    iterable_solution = solve_point_model(
        [0.1, 0.15],
        (0.0, 1.0),
        model;
        saveat=[0.1, 0.5, 0.9, 0.5],
        tstops=(0.6, 0.35),
    )
    @test iterable_solution.t == [0.1, 0.35, 0.5, 0.85, 0.9]

    default_solution = solve_point_model([0.1, 0.15], (0.0, 1.0), model)
    @test 0.35 in default_solution.t
    @test 0.85 in default_solution.t
    @test length(default_solution.t) > 4

    endpoint_drive = PiecewiseConstantDrive(
        baseline=(0.1, 0.2),
        pulses=[DrivePulse(onset=0.0, offset=1.0, increment=(0.3, 0.1))],
        interpretation=AfferentExcitation,
    )
    endpoint_solution = solve_point_model(
        [0.1, 0.15],
        (0.0, 1.0),
        synthetic_model(drive=endpoint_drive);
        saveat=[0.5],
        save_start=false,
        save_end=false,
    )
    @test endpoint_solution.t == [0.0, 0.5, 1.0]

    numeric_saveat_solution = solve_point_model(
        [0.1, 0.15],
        (0.0, 1.0),
        synthetic_model();
        saveat=0.4,
        save_start=false,
        save_end=false,
    )
    @test numeric_saveat_solution.t == [0.4, 0.8]
end
