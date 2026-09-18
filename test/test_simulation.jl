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
