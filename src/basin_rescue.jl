"""Numerical basin sampling policy. All geometric outputs are finite-resolution observations."""
struct BasinMeasurementOptions{P<:PulseExperimentOptions}
    grid_points::Int
    ray_samples::Int
    ray_refinements::Int
    angles::Int
    pulse_options::P
end

function BasinMeasurementOptions(; grid_points=17, ray_samples=16,
    ray_refinements=10, angles=32, pulse_options=PulseExperimentOptions())
    for (name, value, minimum) in (("grid_points", grid_points, 2),
            ("ray_samples", ray_samples, 2), ("ray_refinements", ray_refinements, 0),
            ("angles", angles, 4))
        value isa Integer && !(value isa Bool) && minimum <= value <= typemax(Int) ||
            throw(ArgumentError("$name must be an integer of at least $minimum"))
    end
    pulse_options isa PulseExperimentOptions ||
        throw(ArgumentError("pulse_options must be PulseExperimentOptions"))
    return BasinMeasurementOptions(Int(grid_points), Int(ray_samples),
        Int(ray_refinements), Int(angles), pulse_options)
end

"""
    basin_destination(model, equilibria, state; options=PulseExperimentOptions())

Return finite-window compatibility with a discovered attracting equilibrium.
The result never asserts an asymptotic destination or search completeness.
"""
function basin_destination(model::PointModelParameters, equilibria, state;
    options=PulseExperimentOptions())
    options isa PulseExperimentOptions ||
        throw(ArgumentError("options must be PulseExperimentOptions"))
    _pulse_context(model, equilibria)
    _validate_initial_state(state)
    attempts = NamedTuple[]
    for horizon in options.followup_times
        window = options.diagnostic_options.window_duration
        samples = options.diagnostic_options.min_samples
        times = _sorted_unique_times(vcat([zero(horizon)],
            collect(range(horizon - 2window, horizon - window; length=samples)),
            collect(range(horizon - window, horizon; length=samples))))
        try
            solution = solve_point_model(state, (zero(horizon), horizon), model;
                saveat=times, save_everystep=false, dense=false,
                abstol=options.abstol, reltol=options.reltol,
                domain_atol=options.domain_atol, maxiters=options.maxiters)
            diagnostic = diagnose_trajectory(solution, model; equilibria,
                options=options.diagnostic_options)
            matched = diagnostic.matched_equilibrium
            attracting = matched !== nothing &&
                equilibria.equilibria[matched].stability.classification == Attracting
            status = !diagnostic.integration_success ? :integration_failed :
                diagnostic.classification == EquilibriumCompatible && attracting ?
                :compatible : :unresolved
            push!(attempts, (horizon=horizon, status=status,
                destination=status == :compatible ? matched : nothing,
                reasons=copy(diagnostic.reasons)))
            status == :unresolved || break
        catch error
            error isa InterruptException && rethrow()
            push!(attempts, (horizon=horizon, status=:integration_failed,
                destination=nothing, reasons=[:integration_exception]))
            break
        end
    end
    final = last(attempts)
    return (status=final.status, destination=final.destination, attempts=attempts)
end

function _basin_ray_limit(state, direction)
    limits = eltype(state)[]
    for coordinate in 1:2
        component = direction[coordinate]
        if component > 0
            push!(limits, (one(state[coordinate]) - state[coordinate]) / component)
        elseif component < 0
            push!(limits, -state[coordinate] / component)
        end
    end
    return minimum(limits)
end

function _basin_ray_exit(classify, state, direction, options)
    maximum_radius = _basin_ray_limit(state, direction)
    maximum_radius > 0 || return (status=:domain_edge, lower=missing,
        upper=missing, domain_limit=maximum_radius)
    previous = zero(maximum_radius)
    for index in 1:options.ray_samples
        radius = maximum_radius * index / options.ray_samples
        label = classify(state .+ radius .* direction)
        if label != :source
            lower, upper = previous, radius
            label == :other || return (status=:unresolved, lower=lower,
                upper=upper, domain_limit=maximum_radius)
            for _ in 1:options.ray_refinements
                middle = (lower + upper) / 2
                midpoint_label = classify(state .+ middle .* direction)
                midpoint_label == :unresolved && return (status=:unresolved,
                    lower=lower, upper=upper, domain_limit=maximum_radius)
                if midpoint_label == :source
                    lower = middle
                else
                    upper = middle
                end
            end
            return (status=:observed_exit, lower=lower, upper=upper,
                domain_limit=maximum_radius)
        end
        previous = radius
    end
    return (status=:no_exit_sampled, lower=missing, upper=missing,
        domain_limit=maximum_radius)
end

function _measure_basin_geometry(classify, state, options)
    source = classify(state)
    source == :source || throw(ArgumentError("source must classify as its own basin"))
    n = options.grid_points
    T = eltype(state)
    unit = one(T)
    counts = Dict(:source => 0, :other => 0, :unresolved => 0)
    for e_index in 1:n, i_index in 1:n
        point = T[(T(e_index) - unit / 2) / n, (T(i_index) - unit / 2) / n]
        label = classify(point)
        label in keys(counts) || throw(ArgumentError("invalid basin classifier label"))
        counts[label] += 1
    end
    total = n^2
    area = (observed_fraction=counts[:source] / total,
        possible_fraction_upper=(counts[:source] + counts[:unresolved]) / total,
        source_samples=counts[:source], other_samples=counts[:other],
        unresolved_samples=counts[:unresolved], total_samples=total,
        cell_width=1 / n)
    directions = ((:positive_E, T[unit, 0]), (:negative_E, T[-unit, 0]),
        (:positive_I, T[0, unit]), (:negative_I, T[0, -unit]))
    directional = Dict(name => _basin_ray_exit(classify, state, direction, options)
        for (name, direction) in directions)
    radial = [_basin_ray_exit(classify, state,
        T[cospi(2T(index) / options.angles), sinpi(2T(index) / options.angles)], options)
        for index in 0:(options.angles - 1)]
    observed = filter(result -> result.status == :observed_exit, radial)
    nearest = isempty(observed) ?
        (status=any(result -> result.status == :unresolved, radial) ?
            :unresolved : :no_exit_sampled, lower=missing, upper=missing) :
        (status=any(result -> result.status == :unresolved &&
            result.lower < minimum(item.upper for item in observed), radial) ?
            :unresolved : :observed_exit,
         lower=minimum(item.lower for item in observed),
         upper=minimum(item.upper for item in observed))
    return (area=area, directional=directional, euclidean=nearest,
        angular_samples=options.angles, options=options)
end

"""
    measure_basin(model, equilibria, source_index; options=BasinMeasurementOptions())

Estimate a discovered attracting equilibrium's basin fraction in `[0,1]^2`,
its four signed-axis exit distances, and nearest sampled Euclidean exit.
All values are conditional on finite-window equilibrium compatibility.
The sampled area fractions are not mathematical area bounds.
"""
function measure_basin(model::PointModelParameters, equilibria, source_index;
    options=BasinMeasurementOptions())
    options isa BasinMeasurementOptions ||
        throw(ArgumentError("options must be BasinMeasurementOptions"))
    _pulse_context(model, equilibria)
    source_index isa Integer && !(source_index isa Bool) &&
        1 <= source_index <= length(equilibria.equilibria) ||
        throw(ArgumentError("source_index must identify an equilibrium"))
    source = equilibria.equilibria[source_index]
    source.stability.classification == Attracting ||
        throw(ArgumentError("source equilibrium must be locally attracting"))
    function classify(state)
        outcome = basin_destination(model, equilibria, state;
            options=options.pulse_options)
        return outcome.status == :compatible ?
            (outcome.destination == source_index ? :source : :other) : :unresolved
    end
    return merge((source_index=source_index, source_state=copy(source.state)),
        _measure_basin_geometry(classify, source.state, options))
end

"""
    run_tonic_rescue_trial(model, equilibria, initial_state;
        e_reduction, i_increment, duration, options=PulseExperimentOptions())

Temporarily change tonic `(B_E,B_I)` to `(B_E-e_reduction,B_I+i_increment)`.
Both totals remain nonnegative, and the original baseline resumes afterward.
The destination is a finite-window observation in the original autonomous model.
"""
function run_tonic_rescue_trial(model::PointModelParameters, equilibria, initial_state;
    e_reduction, i_increment, duration, options=PulseExperimentOptions())
    options isa PulseExperimentOptions ||
        throw(ArgumentError("options must be PulseExperimentOptions"))
    baseline = _pulse_context(model, equilibria)
    _validate_initial_state(initial_state)
    all(value -> value >= 0, baseline) ||
        throw(ArgumentError("tonic protocol requires nonnegative baseline totals"))
    for (name, value, positive) in (("e_reduction", e_reduction, false),
            ("i_increment", i_increment, false), ("duration", duration, true))
        value isa Real && !(value isa Bool) && isfinite(value) &&
            (positive ? value > 0 : value >= 0) ||
            throw(ArgumentError("$name must be finite and $(positive ? "positive" : "nonnegative")"))
    end
    e_reduction <= baseline[1] ||
        throw(ArgumentError("E reduction cannot exceed the tonic E baseline"))
    increment = (-float(e_reduction), float(i_increment))
    drive = PiecewiseConstantDrive(baseline=baseline,
        pulses=(DrivePulse(onset=0.0, offset=float(duration), increment=increment),),
        interpretation=AfferentExcitation)
    driven = PointModelParameters(excitatory=model.excitatory,
        inhibitory=model.inhibitory, coupling=model.coupling, drive=drive)
    attempts = NamedTuple[]
    for followup in options.followup_times
        stop = float(duration) + followup
        times = _pulse_save_times(float(duration), followup, options)
        try
            solution = solve_point_model(initial_state, (0.0, stop), driven;
                saveat=times, save_everystep=false, dense=false,
                abstol=options.abstol, reltol=options.reltol,
                domain_atol=options.domain_atol, maxiters=options.maxiters)
            diagnostics = diagnose_trajectory(solution, driven;
                equilibria, options=options.diagnostic_options)
            matched = diagnostics.matched_equilibrium
            attracting = matched !== nothing &&
                equilibria.equilibria[matched].stability.classification == Attracting
            status = !diagnostics.integration_success ? :integration_failed :
                diagnostics.classification == EquilibriumCompatible && attracting ?
                :compatible : :unresolved
            push!(attempts, (followup_time=followup, status=status,
                destination=status == :compatible ? matched : nothing,
                reasons=copy(diagnostics.reasons),
                solver_status=Symbol(string(solution.retcode))))
            status == :unresolved || break
        catch error
            error isa InterruptException && rethrow()
            push!(attempts, (followup_time=followup, status=:integration_failed,
                destination=nothing, reasons=[:integration_exception],
                solver_status=:exception))
            break
        end
    end
    final = last(attempts)
    return (initial_state=collect(float.(initial_state)),
        baseline=baseline, e_reduction=float(e_reduction), i_increment=float(i_increment),
        duration=float(duration), total_E=baseline[1] - e_reduction,
        total_I=baseline[2] + i_increment,
        integrated_E=-float(e_reduction * duration),
        integrated_I=float(i_increment * duration), status=final.status,
        destination=final.destination, attempts=attempts)
end
