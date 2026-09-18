using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve
import CSV

"""
    solve_point_model(initial_state, time_span, parameters; solver=Tsit5(), kwargs...)

Solve the provisional point model on the CPU. `initial_state` must be a
two-element vector ordered `[E, I]`.
"""
function solve_point_model(
    initial_state,
    time_span,
    parameters::PointModelParameters;
    solver=Tsit5(),
    kwargs...,
)
    _require_point_state(initial_state, "initial_state")
    time_span isa Tuple && length(time_span) == 2 ||
        throw(ArgumentError("time_span must be a (start, stop) tuple"))
    start_time, stop_time = time_span
    start_time isa Real && stop_time isa Real ||
        throw(ArgumentError("time_span bounds must be real"))
    isfinite(start_time) && isfinite(stop_time) ||
        throw(ArgumentError("time_span bounds must be finite"))
    start_time < stop_time || throw(ArgumentError("time_span start must precede stop"))

    problem = ODEProblem(point_rhs!, collect(initial_state), time_span, parameters)
    return solve(problem, solver; kwargs...)
end

"""
    write_trajectory_csv(filename, solution)

Write a point-model trajectory with the stable column order `time,E,I`.
"""
function write_trajectory_csv(filename::AbstractString, solution)
    all(state -> state isa AbstractVector && length(state) == 2, solution.u) ||
        throw(ArgumentError("solution states must be two-element vectors ordered [E, I]"))
    table = (
        time=collect(solution.t),
        E=[state[1] for state in solution.u],
        I=[state[2] for state in solution.u],
    )
    CSV.write(filename, table)
    return filename
end
