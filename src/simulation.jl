using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve
import CSV

"""
    solve_point_model(initial_state, time_span, parameters;
                      solver=Tsit5(), domain_atol=1e-8,
                      tstops=(), saveat=(), kwargs...)

Solve the point model on the CPU. `initial_state` must be a
finite two-element real vector ordered `[E, I]` and lie exactly in `[0, 1]^2`.
Adaptive steps and returned states are rejected outside
`[-domain_atol, 1 + domain_atol]^2`; states are never clipped or projected.

Pulse transitions in the model drive are added to `tstops` and to the saved
times. Numeric `saveat` intervals are expanded before those transition times
are merged into the output schedule.
"""
function solve_point_model(
    initial_state,
    time_span,
    parameters::PointModelParameters;
    solver=Tsit5(),
    domain_atol=1.0e-8,
    tstops=(),
    saveat=(),
    kwargs...,
)
    _validate_initial_state(initial_state)
    _validate_domain_atol(domain_atol)
    time_span isa Tuple && length(time_span) == 2 ||
        throw(ArgumentError("time_span must be a (start, stop) tuple"))
    start_time, stop_time = time_span
    start_time isa Real && stop_time isa Real ||
        throw(ArgumentError("time_span bounds must be real"))
    isfinite(start_time) && isfinite(stop_time) ||
        throw(ArgumentError("time_span bounds must be finite"))
    start_time < stop_time || throw(ArgumentError("time_span start must precede stop"))

    transitions = filter(
        time -> start_time <= time <= stop_time,
        drive_transition_times(parameters.drive),
    )
    merged_tstops = _sorted_unique_times(
        vcat(_collect_time_values(tstops, "tstops"), transitions),
    )
    caller_save_times, save_every_step = _save_times(saveat, time_span)
    merged_save_times = _sorted_unique_times(vcat(caller_save_times, transitions))

    problem = ODEProblem(
        point_rhs!,
        collect(initial_state),
        time_span,
        parameters;
        isoutofdomain=(state, _, _) -> _state_outside_domain(state, domain_atol),
    )

    options = (; kwargs...)
    if save_every_step && !isempty(merged_save_times)
        options = merge((save_everystep=true, save_start=true, save_end=true), options)
    end
    start_time in transitions && (options = merge(options, (save_start=true,)))
    stop_time in transitions && (options = merge(options, (save_end=true,)))

    solution = if isempty(merged_save_times)
        solve(problem, solver; tstops=merged_tstops, options...)
    else
        solve(
            problem,
            solver;
            tstops=merged_tstops,
            saveat=merged_save_times,
            options...,
        )
    end
    _postcheck_solution_domain(solution, domain_atol)
    return solution
end

function _validate_initial_state(initial_state)
    _require_point_state(initial_state, "initial_state")
    all(value -> value isa Real, initial_state) ||
        throw(ArgumentError("initial_state coordinates must be real"))
    for value in initial_state
        if !isfinite(value) || !(zero(value) <= value <= one(value))
            throw(
                DomainError(
                    value,
                    "initial_state coordinates must be finite and lie in [0, 1]",
                ),
            )
        end
    end
    return initial_state
end

function _validate_domain_atol(domain_atol)
    domain_atol isa Real || throw(ArgumentError("domain_atol must be real"))
    isfinite(domain_atol) || throw(ArgumentError("domain_atol must be finite"))
    domain_atol >= zero(domain_atol) ||
        throw(ArgumentError("domain_atol must be nonnegative"))
    return domain_atol
end

function _state_outside_domain(state, domain_atol)
    lower_bound = -domain_atol
    upper_bound = one(domain_atol) + domain_atol
    values = state isa Real ? (state,) : state
    return any(
        value -> !(value isa Real) || !isfinite(value) ||
                 value < lower_bound || value > upper_bound,
        values,
    )
end

function _postcheck_solution_domain(solution, domain_atol)
    for (time, state) in zip(solution.t, solution.u)
        _state_outside_domain(state, domain_atol) || continue
        throw(
            DomainError(
                (time=time, state=state),
                "point-model solution left [-domain_atol, 1 + domain_atol]^2",
            ),
        )
    end
    return nothing
end

function _collect_time_values(times, name)
    if times isa Real
        values = [times]
    else
        applicable(iterate, times) ||
            throw(ArgumentError("$name must be a real value or an iterable of real values"))
        values = collect(times)
    end
    all(value -> value isa Real && isfinite(value), values) ||
        throw(ArgumentError("$name values must be finite and real"))
    return values
end

function _save_times(saveat, (start_time, stop_time))
    if saveat isa Real
        isfinite(saveat) && saveat > zero(saveat) ||
            throw(ArgumentError("numeric saveat must be finite and positive"))
        times = collect(start_time:saveat:stop_time)
        isempty(times) && push!(times, start_time)
        last(times) == stop_time || push!(times, stop_time)
        return times, false
    end

    times = _collect_time_values(saveat, "saveat")
    return times, isempty(times)
end

function _sorted_unique_times(times)
    sort!(times)
    unique!(times)
    return times
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
