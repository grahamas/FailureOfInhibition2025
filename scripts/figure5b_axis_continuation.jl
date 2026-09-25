module Figure5bAxisContinuation

using FailureOfInhibition2025
using LinearAlgebra: norm, opnorm, svdvals

include("figure5b_root_lineage.jl")
import .Figure5bRootLineage: RootLineageAnchor, RootTransitionEvidence,
    RootLineageOptions, seed_lineage, transition_lineage

const State = NTuple{2,Float64}
const AugmentedState = NTuple{3,Float64}

"""Numerical limits for one parameter-axis trace-zero search."""
struct AxisOptions
    initial_step_fraction::Float64
    minimum_step_fraction::Float64
    maximum_step_fraction::Float64
    max_steps::Int
    max_retries::Int
    state_step_atol::Float64
    balance_atol::Float64
    trace_atol::Float64
    rank_atol::Float64
    finite_difference_step::Float64
    correction_maxiters::Int
    equilibrium_options::EquilibriumOptions
    stability_options::StabilityOptions

    function AxisOptions(initial_step_fraction::Float64, minimum_step_fraction::Float64,
        maximum_step_fraction::Float64, max_steps::Int, max_retries::Int,
        state_step_atol::Float64, balance_atol::Float64,
        trace_atol::Float64, rank_atol::Float64,
        finite_difference_step::Float64, correction_maxiters::Int,
        equilibrium_options::EquilibriumOptions,
        stability_options::StabilityOptions)
        values = (initial_step_fraction, minimum_step_fraction, maximum_step_fraction,
            state_step_atol, balance_atol, trace_atol, rank_atol,
            finite_difference_step)
        all(x -> isfinite(x) && x > 0, values) ||
            throw(ArgumentError("axis steps and tolerances must be finite and positive"))
        minimum_step_fraction <= initial_step_fraction <= maximum_step_fraction ||
            throw(ArgumentError("axis step order is invalid"))
        all(>(0), (max_steps, max_retries, correction_maxiters)) ||
            throw(ArgumentError("axis iteration limits must be positive"))
        Figure5bRootLineage._valid_search_policy((;
            options=equilibrium_options,
            stability_options=stability_options)) ||
            throw(ArgumentError("axis equilibrium or stability policy is invalid"))
        return new(values[1], values[2], values[3], max_steps,
            max_retries, values[4], values[5], values[6], values[7],
            values[8], correction_maxiters, equilibrium_options,
            stability_options)
    end
end

function AxisOptions(; initial_step_fraction=0.01, minimum_step_fraction=1e-5,
    maximum_step_fraction=0.05, max_steps=200, max_retries=8,
    state_step_atol=0.05, balance_atol=1e-8, trace_atol=1e-8,
    rank_atol=1e-10, finite_difference_step=1e-5,
    correction_maxiters=30, equilibrium_options=EquilibriumOptions(),
    stability_options=StabilityOptions())
    scalars = (initial_step_fraction, minimum_step_fraction, maximum_step_fraction, state_step_atol,
        balance_atol, trace_atol, rank_atol, finite_difference_step)
    all(x -> x isa Real && !(x isa Bool) && isfinite(x) && x > 0,
        scalars) || throw(ArgumentError("axis steps and tolerances must be positive finite reals"))
    values = Float64.(scalars)
    all(x -> isfinite(x) && x > 0, values) ||
        throw(ArgumentError("axis steps and tolerances must remain finite in Float64"))
    values[2] <= values[1] <= values[3] ||
        throw(ArgumentError("minimum_step_fraction <= initial_step_fraction <= maximum_step_fraction is required"))
    all(x -> x isa Integer && !(x isa Bool) && x > 0,
        (max_steps, max_retries, correction_maxiters)) ||
        throw(ArgumentError("axis iteration limits must be positive integers"))
    equilibrium_options isa EquilibriumOptions &&
        stability_options isa StabilityOptions ||
        throw(ArgumentError("axis equilibrium and stability options have wrong types"))
    return AxisOptions(values[1], values[2], values[3], Int(max_steps),
        Int(max_retries), values[4], values[5], values[6], values[7],
        values[8], Int(correction_maxiters), equilibrium_options,
        stability_options)
end

struct AxisPoint
    parameter::Float64
    state::State
    trace::Float64
    anchor::RootLineageAnchor
end

struct AxisSeed
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    point::Union{Nothing,AxisPoint}
    topology::Union{Nothing,Figure5bTopologyResult}
    lineage::Union{Nothing,RootTransitionEvidence}
    error::Union{Nothing,String}
end

"""Immutable snapshot of a local equilibrium solve, including rejected solves."""
struct AxisLocalSolveEvidence
    candidate::Tuple{Vararg{Float64}}
    validation::CandidateValidation
    solver_status::Symbol
    solver_success::Bool
    residual_norm::Float64
    near_singular::Bool
    reasons::Tuple{Vararg{Symbol}}
end

struct AxisTrial
    direction::Int
    source_parameter::Float64
    target_parameter::Float64
    step::Float64
    retry::Int
    accepted::Bool
    status::Symbol
    point::Union{Nothing,AxisPoint}
    local_solve::Union{Nothing,AxisLocalSolveEvidence}
    lineage::Union{Nothing,RootTransitionEvidence}
    reasons::Tuple{Vararg{Symbol}}
    error::Union{Nothing,String}
end

struct AugmentedCandidate
    candidate::AugmentedState
    status::Symbol
    iterations::Int
    error::Union{Nothing,String}
end

AugmentedCandidate(candidate::AugmentedState, status::Symbol, iterations::Int) =
    AugmentedCandidate(candidate, status, iterations, nothing)

struct AxisCorrection
    accepted::Bool
    status::Symbol
    corrector_status::Symbol
    corrector_iterations::Int
    candidate::AugmentedState
    balance_residual::State
    trace_residual::Float64
    minimum_singular_value::Float64
    from_left::Union{Nothing,RootTransitionEvidence}
    from_right::Union{Nothing,RootTransitionEvidence}
    point::Union{Nothing,AxisPoint}
    reasons::Tuple{Vararg{Symbol}}
    error::Union{Nothing,String}
end

struct TraceBracket
    left::AxisPoint
    right::AxisPoint
    correction::AxisCorrection
end

struct AxisZeroEndpoint
    point::AxisPoint
    accepted::Bool
    corrector_status::Symbol
    corrector_iterations::Int
    minimum_singular_value::Float64
    reasons::Tuple{Vararg{Symbol}}
    error::Union{Nothing,String}
end

struct AxisDirection
    direction::Int
    points::Tuple{Vararg{AxisPoint}}
    trials::Tuple{Vararg{AxisTrial}}
    brackets::Tuple{Vararg{TraceBracket}}
    zero_endpoints::Tuple{Vararg{AxisZeroEndpoint}}
    termination::Symbol
end

struct TraceZeroAxisResult
    seed::AxisSeed
    seed_zero::Union{Nothing,AxisZeroEndpoint}
    negative::AxisDirection
    positive::AxisDirection
    status::Symbol
    finite_exhausted::Bool
end

_state(value) = Figure5bRootLineage._state(value)

function _model(model_at_parameter, parameter)
    model = model_at_parameter(parameter)
    model isa PointModelParameters ||
        throw(ArgumentError("model_at_parameter must return PointModelParameters"))
    model.drive isa NoDrive ||
        throw(ArgumentError("axis continuation requires an autonomous zero-drive model"))
    return model
end

function _seeds(model, grid)
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=grid)
        for i in range(0.0, upper_i; length=grid)])
    return unique!(seeds)
end

function _searches(model, options)
    return Tuple(find_equilibria(model; seeds=_seeds(model, grid),
        options=options.equilibrium_options,
        stability_options=options.stability_options)
        for grid in Figure5bRootLineage.GRID_POINTS)
end

_solve(model, state, options) = solve_equilibrium(model, collect(state);
    options=options.equilibrium_options,
    stability_options=options.stability_options)

function _trace(model, state, parameter)
    jacobian = zeros(Float64, 2, 2)
    point_jacobian!(jacobian, collect(state), model, 0.0)
    return jacobian[1, 1] + jacobian[2, 2]
end

function _context_matches(model, searches)
    return searches isa Tuple && length(searches) == 3 &&
        all(search -> search isa EquilibriumSearchResult &&
            isequal(Figure5bRootLineage._point_model_identity(search.model),
                Figure5bRootLineage._point_model_identity(model)), searches)
end

function _search_policy_matches(searches, options)
    return searches isa Tuple && length(searches) == 3 &&
        all(search -> search isa EquilibriumSearchResult &&
            isequal(search.options, options.equilibrium_options) &&
            isequal(search.stability_options, options.stability_options),
            searches)
end

function _solve_context_matches(solved, model, options)
    return Figure5bRootLineage._frozen_context_valid(solved) &&
        isequal(Figure5bRootLineage._point_model_identity(solved.model),
            Figure5bRootLineage._point_model_identity(model)) &&
        isequal(solved.options, options.equilibrium_options) &&
        isequal(solved.stability_options, options.stability_options)
end

function _topology_options(lineage_options)
    return Figure5bTopologyOptions(
        coordinate_match_atol=lineage_options.coordinate_atol,
        minimum_root_separation=lineage_options.minimum_root_separation,
        residual_atol=lineage_options.residual_atol,
        jacobian_atol=lineage_options.jacobian_atol,
        spectral_margin=lineage_options.spectral_margin)
end

function _balance(model, state)
    residual = zeros(Float64, 2)
    point_balance!(residual, collect(state), model, 0.0)
    return (residual[1], residual[2])
end

function _displacement_bound(source, target, options, lineage_options)
    distance = hypot(source[1] - target[1], source[2] - target[2])
    return distance <= options.state_step_atol ?
        max(distance + 2lineage_options.coordinate_atol,
            2lineage_options.coordinate_atol) : nothing
end

function _seed(model_at_parameter, parameter, options, lineage_options,
    search_function, trace_function)
    reasons = Symbol[]
    model = _model(model_at_parameter, parameter)
    searches = search_function(model, options)
    _context_matches(model, searches) || push!(reasons, :search_model_mismatch)
    _search_policy_matches(searches, options) ||
        push!(reasons, :search_policy_mismatch)
    topology = nothing
    lineage = nothing
    if isempty(reasons)
        topology = classify_figure5b_topology(searches;
            options=_topology_options(lineage_options))
        topology.qualified || append!(reasons, topology.reasons)
    end
    if isempty(reasons)
        lineage = seed_lineage(searches, topology.central_state;
            options=lineage_options)
        lineage.accepted || append!(reasons, lineage.reasons)
    end
    point = nothing
    if isempty(reasons)
        state = _state(topology.central_state)
        trace = Float64(trace_function(model, state, parameter))
        residual = _balance(model, state)
        isfinite(trace) && all(isfinite, residual) &&
            maximum(abs, residual) <= options.balance_atol ||
            push!(reasons, :seed_observation_unresolved)
        isempty(reasons) && (point = AxisPoint(parameter, state, trace,
            lineage.anchor))
    end
    unique!(reasons)
    return AxisSeed(isempty(reasons), Tuple(reasons), point, topology,
        lineage, nothing)
end

function _trial(model_at_parameter, source::AxisPoint, target_parameter,
    direction, step, retry, options, lineage_options, search_function,
    solve_function, trace_function)
    reasons = Symbol[]
    lineage = nothing
    point = nothing
    local_solve = nothing
    error = nothing
    status = :unresolved
    try
        model = _model(model_at_parameter, target_parameter)
        solved = solve_function(model, source.state, options)
        solved isa EquilibriumSolveResult ||
            throw(ArgumentError("local solve must return EquilibriumSolveResult"))
        attempt = solved.attempt
        local_solve = AxisLocalSolveEvidence(Tuple(Float64.(attempt.candidate)),
            attempt.validation, attempt.solver_status, attempt.solver_success,
            Float64(attempt.residual_norm), attempt.near_singular,
            Tuple(attempt.reasons))
        _solve_context_matches(solved, model, options) ||
            push!(reasons, :local_solve_context_mismatch)
        if !isempty(reasons)
            nothing
        elseif attempt.validation != AdmissibleCandidate ||
                !isfinite(attempt.residual_norm) ||
                attempt.residual_norm > options.balance_atol ||
                attempt.near_singular
            push!(reasons, :local_equilibrium_unresolved)
        else
            state = _state(attempt.candidate)
            bound = _displacement_bound(source.state, state, options,
                lineage_options)
            bound === nothing && push!(reasons, :state_step_exceeded)
            searches = search_function(model, options)
            _context_matches(model, searches) ||
                push!(reasons, :search_model_mismatch)
            _search_policy_matches(searches, options) ||
                push!(reasons, :search_policy_mismatch)
            if isempty(reasons)
                lineage = transition_lineage(source.anchor, searches, state;
                    options=lineage_options, displacement_atol=bound)
                lineage.accepted || append!(reasons, lineage.reasons)
            end
            if isempty(reasons)
                trace = Float64(trace_function(model, state, target_parameter))
                residual = _balance(model, state)
                isfinite(trace) && all(isfinite, residual) &&
                    maximum(abs, residual) <= options.balance_atol ||
                    push!(reasons, :trial_observation_unresolved)
                isempty(reasons) && (point = AxisPoint(target_parameter,
                    state, trace, lineage.anchor))
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :trial_exception)
        error = sprint(showerror, caught)
    end
    status = isempty(reasons) ? :accepted : first(reasons)
    return AxisTrial(direction, source.parameter, target_parameter, step,
        retry, point !== nothing, status, point, local_solve, lineage,
        Tuple(unique(reasons)), error)
end

function _augmented_residual(model_at_parameter, trace_function, z)
    model = _model(model_at_parameter, z[3])
    state = (z[1], z[2])
    balance = _balance(model, state)
    return Float64[balance[1], balance[2],
        trace_function(model, state, z[3])]
end

function _augmented_jacobian(residual_function, z, probe_bounds, step;
    parameter_h=nothing)
    probe_bounds[1] <= z[3] <= probe_bounds[2] || return nothing
    base = residual_function(z)
    all(isfinite, base) || return nothing
    jacobian = zeros(Float64, 3, 3)
    for column in 1:3
        h = column == 3 && parameter_h !== nothing ? parameter_h :
            step * max(1.0, abs(z[column]))
        left, right = copy(z), copy(z)
        if column == 3
            left_room = z[3] - probe_bounds[1]
            right_room = probe_bounds[2] - z[3]
            if min(left_room, right_room) >= h
                left[column] = max(probe_bounds[1], z[3] - h)
                right[column] = min(probe_bounds[2], z[3] + h)
                jacobian[:, column] .= (residual_function(right) .-
                    residual_function(left)) ./
                    (right[column] - left[column])
            elseif right_room >= left_room && right_room > 0
                h = min(h, right_room)
                h > 32eps(Float64) * max(1.0, abs(z[3])) || return nothing
                right[column] = min(probe_bounds[2], z[3] + h)
                jacobian[:, column] .= (residual_function(right) .- base) ./
                    (right[column] - z[3])
            elseif left_room > 0
                h = min(h, left_room)
                h > 32eps(Float64) * max(1.0, abs(z[3])) || return nothing
                left[column] = max(probe_bounds[1], z[3] - h)
                jacobian[:, column] .= (base .- residual_function(left)) ./
                    (z[3] - left[column])
            else
                return nothing
            end
        else
            left[column] -= h
            right[column] += h
            jacobian[:, column] .= (residual_function(right) .-
                residual_function(left)) ./ (2h)
        end
    end
    return all(isfinite, jacobian) ? jacobian : nothing
end

function _stable_augmented_rank(residual_function, z, probe_bounds, options)
    # A single finite-difference scale can manufacture rank at a stationary
    # trace zero (central cubic or one-sided quadratic). Require full-Jacobian
    # convergence as the difference scale is quartered.
    parameter = z[3]
    left_room = parameter - probe_bounds[1]
    right_room = probe_bounds[2] - parameter
    largest_room = max(left_room, right_room)
    requested_h = options.finite_difference_step * max(1.0, abs(parameter))
    base_h = min(requested_h, largest_room / 2)
    resolution = 32eps(Float64) * max(1.0, abs(parameter))
    isfinite(base_h) && base_h / 4 > resolution || return (NaN, false)
    # Verify actual floating-point probes contract. Merely requesting three
    # scales is insufficient when a narrow bracket clips them to one point.
    side = right_room >= left_room ? 1.0 : -1.0
    widths = ntuple(index -> abs((parameter +
        side * base_h * 2.0^(-(index - 1))) - parameter), 3)
    widths[1] > 1.5widths[2] > 2resolution &&
        widths[2] > 1.5widths[3] > resolution || return (NaN, false)
    jacobians = Matrix{Float64}[]
    for fraction in (1.0, 0.5, 0.25)
        jacobian = _augmented_jacobian(residual_function, z, probe_bounds,
            fraction * options.finite_difference_step;
            parameter_h=fraction * base_h)
        jacobian === nothing && return (NaN, false)
        push!(jacobians, jacobian)
    end
    fine = minimum(svdvals(jacobians[3]))
    # A smaller, fixed balance singular mode can mask a vanishing parameter
    # derivative if only singular values are compared. Bound the full matrix
    # perturbation relative to the finest smallest singular value instead.
    stable = isfinite(fine) && fine > options.rank_atol &&
        opnorm(jacobians[1] - jacobians[2]) <= 0.1fine &&
        opnorm(jacobians[2] - jacobians[3]) <= 0.1fine
    return (fine, stable)
end

function _newton_correct(model_at_parameter, trace_function, left,
    right, bounds, options)
    lower, upper = minmax(left.parameter, right.parameter)
    z = Float64[(left.state[1] + right.state[1]) / 2,
        (left.state[2] + right.state[2]) / 2,
        (lower + upper) / 2]
    residual = point -> _augmented_residual(model_at_parameter,
        trace_function, point)
    last_iteration = 0
    status = :max_iterations
    error = nothing
    try
        for iteration in 0:options.correction_maxiters
            last_iteration = iteration
            values = residual(z)
            if all(isfinite, values) &&
                    maximum(abs, values[1:2]) <= options.balance_atol &&
                    abs(values[3]) <= options.trace_atol
                return AugmentedCandidate(Tuple(z), :converged, iteration)
            end
            iteration == options.correction_maxiters && break
            jacobian = _augmented_jacobian(residual, z, (lower, upper),
                options.finite_difference_step)
            if jacobian === nothing
                status = :jacobian_unresolved
                break
            end
            singular = svdvals(jacobian)
            if !(minimum(singular) > options.rank_atol)
                status = :rank_unresolved
                break
            end
            delta = try
                jacobian \ values
            catch caught
                caught isa InterruptException && rethrow()
                status = :linear_solve_unresolved
                error = sprint(showerror, caught)
                break
            end
            accepted_step = false
            for backtrack in 0:12
                factor = 2.0^(-backtrack)
                proposal = z .- factor .* delta
                lower <= proposal[3] <= upper || continue
                candidate_residual = residual(proposal)
                if all(isfinite, candidate_residual) &&
                        maximum(abs, candidate_residual) < maximum(abs, values)
                    z = proposal
                    accepted_step = true
                    break
                end
            end
            if !accepted_step
                status = :backtrack_unresolved
                break
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        return AugmentedCandidate(Tuple(z), :corrector_exception,
            last_iteration, sprint(showerror, caught))
    end
    return AugmentedCandidate(Tuple(z), status, last_iteration, error)
end

function _correction_failure(candidate, reasons; balance=(NaN, NaN),
    trace=NaN, rank=NaN, from_left=nothing, from_right=nothing,
    error=candidate.error)
    unique!(reasons)
    return AxisCorrection(false, first(reasons), candidate.status,
        candidate.iterations, candidate.candidate,
        balance, trace, rank, from_left, from_right, nothing,
        Tuple(reasons), error)
end

function _validate_correction(model_at_parameter, left::AxisPoint,
    right::AxisPoint, candidate::AugmentedCandidate, bounds, options,
    lineage_options, search_function, trace_function)
    reasons = Symbol[]
    z = candidate.candidate
    lower, upper = minmax(left.parameter, right.parameter)
    candidate.status == :converged || push!(reasons, :corrector_unresolved)
    all(isfinite, z) || push!(reasons, :nonfinite_correction)
    bounds[1] <= z[3] <= bounds[2] ||
        push!(reasons, :correction_out_of_bounds)
    lower <= z[3] <= upper || push!(reasons, :correction_out_of_bracket)
    !isempty(reasons) && return _correction_failure(candidate, reasons)
    balance = (NaN, NaN)
    trace = NaN
    rank = NaN
    from_left = from_right = nothing
    error = nothing
    try
        model = _model(model_at_parameter, z[3])
        balance = _balance(model, (z[1], z[2]))
        trace = Float64(trace_function(model, (z[1], z[2]), z[3]))
        all(isfinite, balance) && isfinite(trace) &&
            maximum(abs, balance) <= options.balance_atol &&
            abs(trace) <= options.trace_atol ||
            push!(reasons, :correction_residual_unresolved)
        residual = point -> _augmented_residual(model_at_parameter,
            trace_function, point)
        rank, stable_rank = _stable_augmented_rank(residual, collect(z),
            (lower, upper), options)
        stable_rank ||
            push!(reasons, :correction_rank_unresolved)
        if isempty(reasons)
            searches = search_function(model, options)
            _context_matches(model, searches) ||
                push!(reasons, :search_model_mismatch)
            _search_policy_matches(searches, options) ||
                push!(reasons, :search_policy_mismatch)
            if isempty(reasons)
                for (endpoint, side) in ((left, :left), (right, :right))
                    bound = _displacement_bound(endpoint.state, (z[1], z[2]),
                        options, lineage_options)
                    if bound === nothing
                        push!(reasons, Symbol("$(side)_state_step_exceeded"))
                        continue
                    end
                    check = transition_lineage(endpoint.anchor, searches,
                        (z[1], z[2]); options=lineage_options,
                        displacement_atol=bound)
                    side == :left ? (from_left = check) : (from_right = check)
                    check.accepted || push!(reasons,
                        Symbol("$(side)_lineage_unresolved"))
                end
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :correction_exception)
        error = sprint(showerror, caught)
    end
    if !isempty(reasons)
        return _correction_failure(candidate, reasons; balance, trace, rank,
            from_left, from_right, error)
    end
    from_left.anchor.states == from_right.anchor.states ||
        push!(reasons, :endpoint_lineage_disagreement)
    !isempty(reasons) && return _correction_failure(candidate, reasons;
        balance, trace, rank, from_left, from_right)
    point = AxisPoint(z[3], (z[1], z[2]), trace, from_left.anchor)
    return AxisCorrection(true, :accepted, candidate.status,
        candidate.iterations, z, balance, trace, rank,
        from_left, from_right, point, (), nothing)
end

function _validate_zero_endpoint(model_at_parameter, point::AxisPoint,
    bounds, options, trace_function)
    reasons = Symbol[]
    rank = NaN
    error = nothing
    try
        model = _model(model_at_parameter, point.parameter)
        balance = _balance(model, point.state)
        trace = Float64(trace_function(model, point.state, point.parameter))
        all(isfinite, balance) && isfinite(trace) &&
            maximum(abs, balance) <= options.balance_atol &&
            abs(trace) <= options.trace_atol ||
            push!(reasons, :zero_endpoint_residual_unresolved)
        residual = z -> _augmented_residual(model_at_parameter,
            trace_function, z)
        rank, stable_rank = _stable_augmented_rank(residual,
            Float64[point.state..., point.parameter], bounds, options)
        stable_rank ||
            push!(reasons, :zero_endpoint_rank_unresolved)
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :zero_endpoint_exception)
        error = sprint(showerror, caught)
    end
    unique!(reasons)
    # A sampled trace-zero point is checked directly; no augmented solver runs.
    return AxisZeroEndpoint(point, isempty(reasons), :not_run_singleton, 0, rank,
        Tuple(reasons), error)
end

_trace_sign(value, tolerance) = abs(value) <= tolerance ? 0 : sign(value)

function _has_crossing(left, right, tolerance)
    left_sign = _trace_sign(left.trace, tolerance)
    right_sign = _trace_sign(right.trace, tolerance)
    # Zero endpoints are validated as singleton locations, once per point.
    return left_sign != 0 && right_sign != 0 && left_sign != right_sign
end

function _bracket(model_at_parameter, left, right, bounds, options,
    lineage_options, search_function, trace_function, correct_function)
    candidate = try
        correct_function(model_at_parameter, trace_function, left, right,
            bounds, options)
    catch caught
        caught isa InterruptException && rethrow()
        return TraceBracket(left, right, _correction_failure(
            AugmentedCandidate((NaN, NaN, NaN), :corrector_exception, 0),
            Symbol[:corrector_exception]; error=sprint(showerror, caught)))
    end
    candidate isa AugmentedCandidate ||
        return TraceBracket(left, right, _correction_failure(
            AugmentedCandidate((NaN, NaN, NaN), :invalid_corrector_result, 0),
            Symbol[:invalid_corrector_result]))
    return TraceBracket(left, right, _validate_correction(model_at_parameter,
        left, right, candidate, bounds, options, lineage_options,
        search_function, trace_function))
end

function _scaled_steps(options::AxisOptions, bounds)
    width = bounds[2] - bounds[1]
    values = (options.initial_step_fraction * width,
        options.minimum_step_fraction * width,
        options.maximum_step_fraction * width)
    all(x -> isfinite(x) && x > 0, values) ||
        throw(ArgumentError("scaled axis steps must be finite and positive"))
    return (initial=values[1], minimum=values[2], maximum=values[3])
end

_axis_target(parameter, destination, step, direction) =
    direction < 0 ? max(destination, parameter - step) :
    min(destination, parameter + step)

_grown_step(step, maximum) = min(maximum, 1.25step)

function _direction(model_at_parameter, seed::AxisPoint, direction, bounds,
    options, lineage_options, search_function, solve_function,
    trace_function, correct_function)
    destination = direction < 0 ? bounds[1] : bounds[2]
    points = AxisPoint[seed]
    trials = AxisTrial[]
    brackets = TraceBracket[]
    zero_endpoints = AxisZeroEndpoint[]
    scaled_steps = _scaled_steps(options, bounds)
    step = scaled_steps.initial
    retries = 0
    termination = :max_steps
    while length(points) - 1 < options.max_steps
        source = last(points)
        if source.parameter == destination
            termination = :exact_bound
            break
        end
        target = _axis_target(source.parameter, destination, step, direction)
        trial = _trial(model_at_parameter, source, target, direction, step,
            retries, options, lineage_options, search_function, solve_function,
            trace_function)
        push!(trials, trial)
        if !trial.accepted
            retries += 1
            step /= 2
            if retries > options.max_retries
                termination = :max_retries
                break
            elseif step < scaled_steps.minimum
                termination = :minimum_step
                break
            end
            continue
        end
        retries = 0
        push!(points, trial.point)
        if abs(trial.point.trace) <= options.trace_atol
            push!(zero_endpoints, _validate_zero_endpoint(
                model_at_parameter, trial.point, bounds, options,
                trace_function))
        end
        if _has_crossing(source, trial.point, options.trace_atol)
            push!(brackets, _bracket(model_at_parameter, source, trial.point,
                bounds, options, lineage_options, search_function,
                trace_function, correct_function))
        end
        if target == destination
            termination = :exact_bound
            break
        end
        step = _grown_step(step, scaled_steps.maximum)
    end
    return AxisDirection(direction, Tuple(points), Tuple(trials),
        Tuple(brackets), Tuple(zero_endpoints), termination)
end

"""
    trace_zero_axis(model_at_parameter, initial_parameter; parameter_bounds,
        axis_options=AxisOptions(), lineage_options=RootLineageOptions())

Search one authorized parameter axis from a genuine seven-root Figure-5b
seed. Every accepted step and trace-zero correction must pass the shared
three-grid lineage contract. `finite_exhausted` describes only this bounded
numerical path, never model-wide absence or completeness.
"""
function trace_zero_axis(model_at_parameter, initial_parameter;
    parameter_bounds, axis_options=AxisOptions(),
    lineage_options=RootLineageOptions())
    return _trace_zero_axis(model_at_parameter, initial_parameter;
        parameter_bounds, axis_options, lineage_options,
        search_function=_searches, solve_function=_solve,
        trace_function=_trace, correct_function=_newton_correct)
end

function _trace_zero_axis(model_at_parameter, initial_parameter;
    parameter_bounds, axis_options=AxisOptions(),
    lineage_options=RootLineageOptions(), search_function=_searches,
    solve_function=_solve, trace_function=_trace,
    correct_function=_newton_correct)
    axis_options isa AxisOptions || throw(ArgumentError("axis_options must be AxisOptions"))
    lineage_options isa RootLineageOptions ||
        throw(ArgumentError("lineage_options must be RootLineageOptions"))
    initial_parameter isa Real && !(initial_parameter isa Bool) &&
        isfinite(initial_parameter) ||
        throw(ArgumentError("initial_parameter must be finite and real"))
    parameter_bounds isa Tuple && length(parameter_bounds) == 2 &&
        all(value -> value isa Real && !(value isa Bool) && isfinite(value),
            parameter_bounds) ||
        throw(ArgumentError("parameter_bounds must be two finite reals"))
    bounds = Tuple(Float64.(parameter_bounds))
    all(isfinite, bounds) && bounds[1] < bounds[2] ||
        throw(ArgumentError("parameter_bounds must remain strictly ordered in Float64"))
    _scaled_steps(axis_options, bounds)
    parameter = Float64(initial_parameter)
    isfinite(parameter) && bounds[1] <= parameter <= bounds[2] ||
        throw(ArgumentError("initial_parameter must lie within parameter_bounds"))
    seed = try
        _seed(model_at_parameter, parameter, axis_options,
            lineage_options, search_function, trace_function)
    catch caught
        caught isa InterruptException && rethrow()
        AxisSeed(false, (:seed_exception,), nothing, nothing, nothing,
            sprint(showerror, caught))
    end
    if !seed.accepted
        empty_negative = AxisDirection(-1, (), (), (), (), :seed_unresolved)
        empty_positive = AxisDirection(1, (), (), (), (), :seed_unresolved)
        return TraceZeroAxisResult(seed, nothing,
            empty_negative, empty_positive,
            :seed_unresolved, false)
    end
    seed_zero = abs(seed.point.trace) <= axis_options.trace_atol ?
        _validate_zero_endpoint(model_at_parameter, seed.point,
            bounds, axis_options, trace_function) : nothing
    negative = _direction(model_at_parameter, seed.point, -1, bounds,
        axis_options, lineage_options, search_function, solve_function,
        trace_function, correct_function)
    positive = _direction(model_at_parameter, seed.point, 1, bounds,
        axis_options, lineage_options, search_function, solve_function,
        trace_function, correct_function)
    exhausted = negative.termination == :exact_bound &&
        positive.termination == :exact_bound &&
        (seed_zero === nothing || seed_zero.accepted) &&
        all(bracket -> bracket.correction.accepted,
            (negative.brackets..., positive.brackets...)) &&
        all(endpoint -> endpoint.accepted,
            (negative.zero_endpoints..., positive.zero_endpoints...))
    return TraceZeroAxisResult(seed, seed_zero, negative, positive,
        exhausted ? :finite_exhausted : :unresolved, exhausted)
end

end # module
