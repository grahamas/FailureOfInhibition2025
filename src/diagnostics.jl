"""Finite-window observations; neither value is an attractor or biological label."""
@enum TrajectoryClassification begin
    EquilibriumCompatible
    TrajectoryUnresolved
end

"""
    DiagnosticOptions(; window_duration=5.0, coordinate_atol=1e-6,
                        balance_atol=1e-8, min_samples=3)

Numerical criteria for two consecutive terminal observation windows.
`window_duration` is positive and measured in ms; the nonnegative
`coordinate_atol` bounds both coordinate ranges and infinity distances to
discovered equilibria. `balance_atol` bounds the dimensionless balance residual,
not the dimensional ODE derivative. Each closed window requires at least
`min_samples >= 2` saved samples, including both endpoints. The shared boundary
is counted in both windows. These are sampled numerical criteria, not
biological thresholds or guarantees about unsampled times.
"""
struct DiagnosticOptions{T<:AbstractFloat}
    window_duration::T
    coordinate_atol::T
    balance_atol::T
    min_samples::Int

    function DiagnosticOptions(
        window_duration::T, coordinate_atol::T, balance_atol::T, min_samples::Int,
    ) where {T<:AbstractFloat}
        all(isfinite, (window_duration, coordinate_atol, balance_atol)) ||
            throw(ArgumentError("diagnostic durations and tolerances must be finite"))
        window_duration > zero(T) ||
            throw(ArgumentError("window_duration must be positive"))
        coordinate_atol >= zero(T) && balance_atol >= zero(T) ||
            throw(ArgumentError("diagnostic tolerances must be nonnegative"))
        min_samples >= 2 || throw(ArgumentError("min_samples must be at least two"))
        return new{T}(window_duration, coordinate_atol, balance_atol, min_samples)
    end
end

function DiagnosticOptions(;
    window_duration=5.0, coordinate_atol=1.0e-6, balance_atol=1.0e-8,
    min_samples=3,
)
    raw_values = (window_duration, coordinate_atol, balance_atol)
    all(value -> value isa Real, raw_values) ||
        throw(ArgumentError("diagnostic durations and tolerances must be real"))
    min_samples isa Integer && 2 <= min_samples <= typemax(Int) ||
        throw(ArgumentError("min_samples must be an integer of at least two"))
    values = promote(float.(raw_values)...)
    return DiagnosticOptions(values..., Int(min_samples))
end

"""
    TrajectoryDiagnostics

Conservative diagnostics for saved trajectory samples. `window_bounds`,
`sample_counts`, E/I `means` and `ranges`, and `max_balance_residuals` contain
one entry per terminal window. Means are arithmetic sample means, not time
averages. `equilibrium_distances` contains the maximum coordinate infinity
distance over both sampled windows to each discovered equilibrium.
`matched_equilibrium` identifies a unique coordinate match even when other
criteria leave `classification` unresolved. Unavailable metrics are `NaN`.

`integration_success` requires a successful return code and, when
`solution.prob.tspan` exists, reaching its requested final time. `reasons`
retains all detected issues. `options` retains the applied numerical policy;
the physical-domain tolerance is `equilibria.options.domain_atol`. Spectral
stability and search completeness remain in the supplied equilibrium result.
`EquilibriumCompatible` does not imply attraction, asymptotic convergence,
periodicity, completeness, or a biological interpretation.
"""
struct TrajectoryDiagnostics{T<:AbstractFloat,O<:DiagnosticOptions}
    classification::TrajectoryClassification
    reasons::Vector{Symbol}
    window_bounds::NTuple{2,NTuple{2,T}}
    sample_counts::NTuple{2,Int}
    means::NTuple{2,Vector{T}}
    ranges::NTuple{2,Vector{T}}
    max_balance_residuals::NTuple{2,T}
    equilibrium_distances::Vector{T}
    matched_equilibrium::Union{Nothing,Int}
    integration_success::Bool
    options::O
end

function _same_diagnostic_parameters(first_model, second_model)
    same_inhibitory_family =
        (first_model.inhibitory.response isa LogisticResponse) ==
        (second_model.inhibitory.response isa LogisticResponse)
    return same_inhibitory_family &&
           _model_numeric_values(first_model) == _model_numeric_values(second_model)
end

function _diagnostic_samples(solution, model, equilibria, options)
    all(name -> hasproperty(solution, name), (:t, :u, :retcode)) ||
        throw(ArgumentError("solution must provide t, u, and retcode"))
    solution.t isa AbstractVector && solution.u isa AbstractVector ||
        throw(ArgumentError("solution.t and solution.u must be vectors"))
    length(solution.t) == length(solution.u) ||
        throw(ArgumentError("solution times and states must have equal lengths"))
    all(time -> time isa Real, solution.t) ||
        throw(ArgumentError("solution times must be real"))
    T = promote_type(_model_float_type(model), typeof(options.window_duration))
    for value in equilibria.frozen_drive
        T = promote_type(T, typeof(float(value)))
    end
    for time in solution.t
        T = promote_type(T, typeof(float(time)))
    end
    for state in solution.u
        _require_point_state(state, "solution state")
        all(value -> value isa Real, state) ||
            throw(ArgumentError("solution state coordinates must be real"))
        for value in state
            T = promote_type(T, typeof(float(value)))
        end
    end
    return T.(collect(solution.t)), [T.(collect(state)) for state in solution.u], T
end

function _diagnostic_integration_success(solution, times, reasons)
    applicable(successful_retcode, solution.retcode) ||
        throw(ArgumentError("solution.retcode must support SciMLBase.successful_retcode"))
    success = successful_retcode(solution.retcode)
    success || push!(reasons, :integration_failed)
    if hasproperty(solution, :prob) && hasproperty(solution.prob, :tspan)
        span = solution.prob.tspan
        span isa Tuple && length(span) == 2 &&
            all(value -> value isa Real && isfinite(value), span) &&
            first(span) < last(span) ||
            throw(ArgumentError("solution.prob.tspan must contain ordered finite bounds"))
        if isempty(times) || !isfinite(last(times)) || last(times) != last(span)
            push!(reasons, :incomplete_integration)
            success = false
        end
        if any(time -> isfinite(time) && !(first(span) <= time <= last(span)), times)
            push!(reasons, :times_outside_integration_span)
            success = false
        end
    end
    return success
end

function _validate_diagnostic_solution_context(solution, model, times)
    hasproperty(solution, :prob) && hasproperty(solution.prob, :p) || return nothing
    source = solution.prob.p
    source isa PointModelParameters || return nothing
    _same_diagnostic_parameters(model, source) ||
        throw(ArgumentError("solution population and coupling parameters must match model"))
    interval = if hasproperty(solution.prob, :tspan)
        solution.prob.tspan
    elseif !isempty(times) && all(isfinite, times)
        extrema(times)
    else
        return nothing
    end
    start_time, stop_time = interval
    check_times = vcat(
        [start_time, stop_time],
        drive_transition_times(source.drive),
        drive_transition_times(model.drive),
    )
    all(check_times) do time
        !(start_time <= time <= stop_time) ||
            drive_value(source.drive, time) == drive_value(model.drive, time)
    end || throw(ArgumentError("solution drive protocol must match model over the integration interval"))
    return nothing
end

function _diagnostic_constant_drive(model, bounds, frozen_model, reasons)
    start_time, stop_time = first(first(bounds)), last(last(bounds))
    reference = drive_value(model.drive, start_time)
    transitions = drive_transition_times(model.drive)
    constant = all(transitions) do time
        !(start_time <= time <= stop_time) || drive_value(model.drive, time) == reference
    end
    constant || push!(reasons, :changing_drive)
    reference == drive_value(frozen_model.drive, start_time) ||
        push!(reasons, :frozen_drive_mismatch)
    return nothing
end

"""
    diagnose_trajectory(solution, model; equilibria, options=DiagnosticOptions())

Compare two closed terminal windows of saved `[E, I]` states with the
admissible equilibria in an `EquilibriumSearchResult`. A compatible observation
requires successful complete integration; finite, strictly increasing times;
finite states within the physical domain up to the equilibrium search's
`domain_atol`; sampled window endpoints and enough samples; constant drive
throughout both windows, including the final endpoint; small coordinate
ranges and dimensionless residuals; and a unique sufficiently close root
without near-singular or unresolved-nearby root uncertainty. Every threshold
comparison is inclusive. All drive transitions are checked, including those
between samples; simultaneous transitions with unchanged totals are allowed.

Malformed inputs/options or differing population/coupling parameters between
`model` and the search context throw `ArgumentError`. When `solution.prob.p` is
a point model, its population/coupling parameters and effective drive protocol
must also match `model` throughout the requested integration interval (or the
saved interval when no `prob.tspan` is available). Context mismatches throw
`ArgumentError`. Numerical failures and
unmet evidence criteria return `TrajectoryUnresolved` with reasons. When the
solution has no `prob.tspan`, its last saved time defines the observation end;
completion of a longer requested interval cannot then be established. The
caller is responsible for model provenance when `solution.prob.p` is absent.
No interpolation, asymptotic or periodic-orbit inference, or biological
classification is performed. A stationary saddle can be equilibrium-compatible.
"""
function diagnose_trajectory(
    solution, model::PointModelParameters;
    equilibria, options=DiagnosticOptions(),
)
    options isa DiagnosticOptions || throw(ArgumentError("options must be DiagnosticOptions"))
    equilibria isa EquilibriumSearchResult ||
        throw(ArgumentError("equilibria must be an EquilibriumSearchResult"))
    _same_diagnostic_parameters(model, equilibria.model) &&
        _same_diagnostic_parameters(model, equilibria.frozen_model) ||
        throw(ArgumentError("model population and coupling parameters must match the equilibrium context"))
    times, states, T = _diagnostic_samples(solution, model, equilibria, options)
    reasons = Symbol[]
    integration_success = _diagnostic_integration_success(solution, times, reasons)
    _validate_diagnostic_solution_context(solution, model, times)
    nan = convert(T, NaN)
    bounds = ((nan, nan), (nan, nan))
    counts = (0, 0)
    means = (fill(nan, 2), fill(nan, 2))
    ranges = (fill(nan, 2), fill(nan, 2))
    residual_maxima = [nan, nan]
    distances = fill(nan, length(equilibria.equilibria))
    matched = nothing

    isempty(times) && push!(reasons, :empty_solution)
    finite_times = all(isfinite, times)
    finite_times || push!(reasons, :nonfinite_time)
    ordered_times = all(pair -> first(pair) < last(pair), zip(times, Iterators.drop(times, 1)))
    ordered_times || push!(reasons, :unordered_times)
    finite_states = all(state -> all(isfinite, state), states)
    finite_states || push!(reasons, :nonfinite_state)
    domain_atol = equilibria.options.domain_atol
    any(state -> any(value -> isfinite(value) &&
        !( -domain_atol <= value <= one(value) + domain_atol), state), states) &&
        push!(reasons, :outside_physical_domain)

    if !isempty(times) && finite_times && ordered_times
        stop = last(times)
        middle = stop - options.window_duration
        start = middle - options.window_duration
        bounds = ((start, middle), (middle, stop))
        first(times) <= start || push!(reasons, :insufficient_window_coverage)
        indices = ntuple(2) do window
            lower, upper = bounds[window]
            findall(time -> lower <= time <= upper, times)
        end
        counts = map(length, indices)
        all(count -> count >= options.min_samples, counts) ||
            push!(reasons, :insufficient_samples)
        all(bound -> bound in times, (start, middle, stop)) ||
            push!(reasons, :unsampled_window_boundary)
        _diagnostic_constant_drive(model, bounds, equilibria.frozen_model, reasons)

        residual = Vector{T}(undef, 2)
        for window in 1:2
            selected = indices[window]
            isempty(selected) && continue
            all(index -> all(isfinite, states[index]), selected) || continue
            for coordinate in 1:2
                values = [states[index][coordinate] for index in selected]
                means[window][coordinate] = sum(value / length(values) for value in values)
                ranges[window][coordinate] = maximum(values) - minimum(values)
            end
            residual_maxima[window] = zero(T)
            for index in selected
                point_balance!(residual, states[index], model, times[index])
                if !all(isfinite, residual)
                    residual_maxima[window] = nan
                    push!(reasons, :nonfinite_balance_residual)
                    break
                end
                residual_maxima[window] = max(residual_maxima[window], maximum(abs, residual))
            end
        end
        any(window -> any(value -> isfinite(value) && value > options.coordinate_atol,
                          window), ranges) && push!(reasons, :large_coordinate_range)
        any(value -> isfinite(value) && value > options.balance_atol, residual_maxima) &&
            push!(reasons, :large_balance_residual)

        selected = union(indices...)
        if !isempty(selected) && all(index -> all(isfinite, states[index]), selected)
            for (equilibrium_index, equilibrium) in enumerate(equilibria.equilibria)
                distances[equilibrium_index] = maximum(selected) do index
                    maximum(abs.(states[index] .- equilibrium.state))
                end
            end
        end
        matches = findall(distance -> isfinite(distance) && distance <= options.coordinate_atol,
                          distances)
        if length(matches) == 1
            matched = only(matches)
            equilibrium = equilibria.equilibria[matched]
            equilibrium.near_singular && push!(reasons, :near_singular_equilibrium)
            any(component -> any(index -> index in component, equilibrium.member_attempts),
                equilibria.unresolved_nearby) && push!(reasons, :unresolved_nearby_equilibria)
        elseif isempty(matches)
            push!(reasons, isempty(equilibria.equilibria) ?
                  :no_discovered_equilibria : :no_matching_equilibrium)
        else
            push!(reasons, :ambiguous_equilibrium_match)
        end
    end
    unique!(reasons)
    classification = isempty(reasons) ? EquilibriumCompatible : TrajectoryUnresolved
    return TrajectoryDiagnostics(
        classification, reasons, bounds, counts, means, ranges,
        Tuple(residual_maxima), distances, matched, integration_success, options,
    )
end
