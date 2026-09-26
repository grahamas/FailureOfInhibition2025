module Figure5bCurveSeeds

using FailureOfInhibition2025
using ForwardDiff
using LinearAlgebra: norm, opnorm, svdvals

include("figure5b_axis_continuation.jl")
const Axis = Figure5bAxisContinuation
const Lineage = Axis.Figure5bRootLineage

const State = NTuple{2,Float64}
const CurveState = NTuple{4,Float64}
const ParameterBox = NTuple{2,NTuple{2,Float64}}
const MAX_NEUTRAL_TRACE_ATOL = 1e-7
const FINITE_REJECTION_REASONS = (:root_count_mismatch,
    :outer_stability_pattern_mismatch, :outer_attractor_count,
    :outer_saddle_count, :central_not_neutral)
const AUTHORIZED = (e_to_e=(14.0, 24.0), i_to_e=(6.0, 18.0),
    e_to_i=(12.0, 28.0), i_to_i=(0.0, 10.0), theta_off=(6.0, 12.0))

_trace_zero_atol(axis_options, lineage_options) =
    min(axis_options.trace_atol, MAX_NEUTRAL_TRACE_ATOL,
        2lineage_options.spectral_margin)

"""Validated numerical policy for finite, bounded curve-seed discovery."""
struct CurveSeedOptions
    axis_options::Axis.AxisOptions
    solver_atol::Float64
    maxiters::Int
    max_backtracks::Int
    rank_atol::Float64
    finite_difference_step::Float64
    state_scales::State
    homotopy_min_fraction::Float64
    homotopy_max_trials::Int
    dedup_atol::Float64

    function CurveSeedOptions(axis_options::Axis.AxisOptions,
        solver_atol::Float64, maxiters::Int, max_backtracks::Int,
        rank_atol::Float64, finite_difference_step::Float64,
        state_scales::State, homotopy_min_fraction::Float64,
        homotopy_max_trials::Int, dedup_atol::Float64)
        values = (solver_atol, rank_atol, finite_difference_step,
            state_scales..., homotopy_min_fraction, dedup_atol)
        all(x -> isfinite(x) && x > 0, values) ||
            throw(ArgumentError("curve tolerances and scales must be finite and positive"))
        homotopy_min_fraction < 1 ||
            throw(ArgumentError("homotopy_min_fraction must be below one"))
        maxiters > 0 && max_backtracks > 0 && homotopy_max_trials > 0 ||
            throw(ArgumentError("curve iteration limits must be positive"))
        return new(axis_options, solver_atol, maxiters, max_backtracks,
            rank_atol, finite_difference_step, state_scales,
            homotopy_min_fraction, homotopy_max_trials, dedup_atol)
    end
end

function CurveSeedOptions(; axis_options=Axis.AxisOptions(),
    solver_atol=1e-8, maxiters=60, max_backtracks=16, rank_atol=1e-10,
    finite_difference_step=1e-5, state_scales=(0.1, 0.1),
    homotopy_min_fraction=1e-4, homotopy_max_trials=64, dedup_atol=1e-6)
    axis_options isa Axis.AxisOptions ||
        throw(ArgumentError("axis_options must be AxisOptions"))
    state_scales isa Tuple && length(state_scales) == 2 ||
        throw(ArgumentError("state_scales must contain E and I scales"))
    raw = (solver_atol, rank_atol, finite_difference_step,
        state_scales..., homotopy_min_fraction, dedup_atol)
    all(x -> x isa Real && !(x isa Bool) && isfinite(x) && x > 0, raw) ||
        throw(ArgumentError("curve tolerances and scales must be positive reals"))
    all(x -> x isa Integer && !(x isa Bool) && 0 < x <= typemax(Int),
        (maxiters, max_backtracks, homotopy_max_trials)) ||
        throw(ArgumentError("curve iteration limits must be positive integers"))
    values = Float64.(raw)
    return CurveSeedOptions(axis_options, values[1], Int(maxiters),
        Int(max_backtracks), values[2], values[3],
        (values[4], values[5]), values[6], Int(homotopy_max_trials),
        values[7])
end

struct CurveContext
    base_model::PointModelParameters
    pair::NTuple{2,Symbol}
    bounds::ParameterBox
    origin_parameters::State
end

struct CurveOriginEvidence
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    supplied_state::State
    central_state::Union{Nothing,State}
    topology::Union{Nothing,Figure5bTopologyResult}
    lineage::Union{Nothing,Lineage.RootTransitionEvidence}
    searches::Union{Nothing,Tuple}
    error::Union{Nothing,String}
end

struct SeedSolverIteration
    coordinates::Tuple{Vararg{Float64}}
    residual_norm::Float64
    damping::Float64
    status::Symbol
    error::Union{Nothing,String}
end

struct SeedSolveResult
    converged::Bool
    status::Symbol
    candidate::Tuple{Vararg{Float64}}
    residual_norm::Float64
    iterations::Tuple{Vararg{SeedSolverIteration}}
    error::Union{Nothing,String}
end

struct CurveSeedAttempt
    method::Symbol
    fixed_axis::Union{Nothing,Int}
    fixed_value::Union{Nothing,Float64}
    solve::SeedSolveResult
    candidate::CurveState
end

struct NeutralTopologyEvidence
    qualified::Bool
    reasons::Tuple{Vararg{Symbol}}
    tracks::Lineage.RootTrackEvidence
    central_track::Union{Nothing,Int}
    traces::NTuple{3,Float64}
    determinants::NTuple{3,Float64}
    frequencies::NTuple{3,Float64}
    slopes::NTuple{3,State}
end

struct CurveHomotopyTrial
    source_fraction::Float64
    target_fraction::Float64
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    local_solve::Union{Nothing,Axis.AxisLocalSolveEvidence}
    searches::Union{Nothing,Tuple}
    lineage::Union{Nothing,Lineage.RootTransitionEvidence}
    error::Union{Nothing,String}
end

struct CurveHomotopyEvidence
    qualified::Bool
    reasons::Tuple{Vararg{Symbol}}
    trials::Tuple{Vararg{CurveHomotopyTrial}}
    terminal_anchor::Union{Nothing,Lineage.RootLineageAnchor}
end

struct CurveRankEvidence
    qualified::Bool
    reasons::Tuple{Vararg{Symbol}}
    minimum_singular_value::Float64
    coarse_difference::Float64
    fine_difference::Float64
    independent_difference::Float64
    parameter_probe_widths::NTuple{2,NTuple{3,Float64}}
    error::Union{Nothing,String}
end

struct CurveSeedQualification
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    candidate::CurveState
    residual::NTuple{3,Float64}
    minimum_singular_value::Float64
    rank::Union{Nothing,CurveRankEvidence}
    determinant::Float64
    frequency::Float64
    searches::Union{Nothing,Tuple}
    topology::Union{Nothing,NeutralTopologyEvidence}
    homotopy::Union{Nothing,CurveHomotopyEvidence}
    error::Union{Nothing,String}
end

struct CurveSeedResult
    context::CurveContext
    options::CurveSeedOptions
    lineage_options::Lineage.RootLineageOptions
    origin::CurveOriginEvidence
    attempts::Tuple{Vararg{CurveSeedAttempt}}
    qualifications::Tuple{Vararg{CurveSeedQualification}}
    qualified_indices::Tuple{Vararg{Int}}
    unresolved_methods::Tuple{Vararg{Symbol}}
    status::Symbol
end

_parameter(model, name::Symbol) = name == :theta_off ?
    model.inhibitory.response.failure_threshold : getproperty(model.coupling, name)

function _context(base_model, pair, parameter_bounds)
    base_model isa PointModelParameters && base_model.drive isa NoDrive &&
        base_model.excitatory.response isa LogisticResponse &&
        base_model.inhibitory.response isa FailureOfInhibitionResponse ||
        throw(ArgumentError("curve seeds require an autonomous FoI point model"))
    all(name -> begin
        value = _parameter(base_model, name)
        lower, upper = getproperty(AUTHORIZED, name)
        value isa Real && !(value isa Bool) && isfinite(value) &&
            lower <= value <= upper
    end, keys(AUTHORIZED)) ||
        throw(ArgumentError("all five base parameters must be in the authorized domain"))
    pair isa Tuple && length(pair) == 2 && all(x -> x isa Symbol, pair) &&
        pair[1] != pair[2] && all(x -> x in keys(AUTHORIZED), pair) ||
        throw(ArgumentError("pair must contain distinct authorized parameter names"))
    parameter_bounds isa Tuple && length(parameter_bounds) == 2 ||
        throw(ArgumentError("parameter_bounds must contain two intervals"))
    bounds = ntuple(index -> begin
        interval = parameter_bounds[index]
        interval isa Tuple && length(interval) == 2 &&
            all(x -> x isa Float64 && isfinite(x), interval) ||
            throw(ArgumentError("parameter intervals require finite Float64 endpoints"))
        authorized = getproperty(AUTHORIZED, pair[index])
        authorized[1] <= interval[1] < interval[2] <= authorized[2] ||
            throw(ArgumentError("raw parameter interval exceeds authorized bounds"))
        interval[1] <= _parameter(base_model, pair[index]) <= interval[2] ||
            throw(ArgumentError("base parameter lies outside raw interval"))
        values = interval
        all(isfinite, values) &&
            authorized[1] <= values[1] < values[2] <= authorized[2] ||
            throw(ArgumentError("parameter interval exceeds authorized bounds"))
        values
    end, 2)
    origin = ntuple(index -> Float64(_parameter(base_model, pair[index])), 2)
    all(index -> bounds[index][1] <= origin[index] <= bounds[index][2], 1:2) ||
        throw(ArgumentError("base parameters must lie in the authorized box"))
    context = CurveContext(base_model, pair, bounds, origin)
    isequal(Lineage._point_model_identity(_model(context, origin)),
        Lineage._point_model_identity(base_model)) ||
        throw(ArgumentError("Float64 origin changes the base model identity"))
    return context
end

function _model(context::CurveContext, values)
    parameter(name) = name == context.pair[1] ? values[1] :
        name == context.pair[2] ? values[2] : _parameter(context.base_model, name)
    updated_coupling = PointCoupling(e_to_e=parameter(:e_to_e),
        i_to_e=parameter(:i_to_e), e_to_i=parameter(:e_to_i),
        i_to_i=parameter(:i_to_i))
    old_response = context.base_model.inhibitory.response
    response = FailureOfInhibitionResponse(slope=old_response.slope,
        onset_threshold=old_response.onset_threshold,
        failure_threshold=parameter(:theta_off))
    inhibitory = PopulationParameters(
        timescale=context.base_model.inhibitory.timescale,
        response=response)
    return PointModelParameters(excitatory=context.base_model.excitatory,
        inhibitory=inhibitory, coupling=updated_coupling, drive=NoDrive())
end

function _residual(context::CurveContext, z)
    model = _model(context, (z[3], z[4]))
    residual = similar(z, 3)
    point_balance!(view(residual, 1:2), view(z, 1:2), model,
        zero(eltype(z)))
    jacobian = zeros(eltype(z), 2, 2)
    point_jacobian!(jacobian, view(z, 1:2), model, zero(eltype(z)))
    residual[3] = jacobian[1, 1] + jacobian[2, 2]
    return residual
end

function _in_bounds(z, bounds)
    return all(index -> bounds[index][1] <= z[index] <= bounds[index][2],
        eachindex(bounds))
end

function _solve_bounded(residual_function, initial, bounds, options)
    z = Float64.(initial)
    records = SeedSolverIteration[]
    _in_bounds(z, bounds) || return SeedSolveResult(false, :initial_out_of_bounds,
        Tuple(z), Inf, (SeedSolverIteration(Tuple(z), Inf, 0.0,
            :initial_out_of_bounds, nothing),), nothing)
    status = :maximum_iterations
    error = nothing
    residual_norm = Inf
    for iteration in 0:options.maxiters
        values = try
            Float64.(residual_function(z))
        catch caught
            caught isa InterruptException && rethrow()
            error = sprint(showerror, caught)
            status = :residual_exception
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, error))
            break
        end
        residual_norm = all(isfinite, values) ? maximum(abs, values) : Inf
        if residual_norm <= options.solver_atol
            status = :converged
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, nothing))
            break
        end
        if iteration == options.maxiters
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, nothing))
            break
        end
        jacobian = try
            ForwardDiff.jacobian(residual_function, z)
        catch caught
            caught isa InterruptException && rethrow()
            error = sprint(showerror, caught)
            status = :jacobian_exception
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, error))
            break
        end
        if !all(isfinite, jacobian) ||
                minimum(svdvals(jacobian)) <= options.rank_atol
            status = :solver_rank_unresolved
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, nothing))
            break
        end
        delta = try
            jacobian \ values
        catch caught
            caught isa InterruptException && rethrow()
            error = sprint(showerror, caught)
            status = :linear_solve_exception
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, error))
            break
        end
        accepted = false
        for backtrack in 0:options.max_backtracks
            damping = 2.0^(-backtrack)
            proposal = z .- damping .* delta
            if !_in_bounds(proposal, bounds)
                push!(records, SeedSolverIteration(Tuple(proposal), Inf,
                    damping, :proposal_out_of_bounds, nothing))
                continue
            end
            proposed = try
                Float64.(residual_function(proposal))
            catch caught
                caught isa InterruptException && rethrow()
                error = sprint(showerror, caught)
                push!(records, SeedSolverIteration(Tuple(proposal), Inf,
                    damping, :proposal_exception, error))
                continue
            end
            proposed_norm = all(isfinite, proposed) ? maximum(abs, proposed) : Inf
            if proposed_norm < residual_norm
                push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                    damping, :accepted_step, nothing))
                z = proposal
                accepted = true
                break
            end
            push!(records, SeedSolverIteration(Tuple(proposal), proposed_norm,
                damping, :proposal_not_improved, nothing))
        end
        if !accepted
            status = :backtrack_unresolved
            push!(records, SeedSolverIteration(Tuple(z), residual_norm,
                0.0, status, error))
            break
        end
    end
    return SeedSolveResult(status == :converged, status, Tuple(z),
        residual_norm, Tuple(records), error)
end

function _safe_solve(solve_function, residual_function, initial, bounds, options)
    try
        solved = solve_function(residual_function, initial, bounds, options)
        solved isa SeedSolveResult && length(solved.candidate) == length(initial) ||
            return SeedSolveResult(false, :invalid_solver_result,
                Tuple(Float64.(initial)), Inf, (), "invalid solver result")
        return solved
    catch caught
        caught isa InterruptException && rethrow()
        return SeedSolveResult(false, :solver_exception,
            Tuple(Float64.(initial)), Inf, (), sprint(showerror, caught))
    end
end

function _origin(context, supplied, options, lineage_options, search_function)
    reasons = Symbol[]
    topology = nothing
    lineage = nothing
    searches = nothing
    central = nothing
    error = nothing
    try
        model = _model(context, context.origin_parameters)
        searches = search_function(model, options.axis_options)
        Axis._context_matches(model, searches) ||
            push!(reasons, :origin_model_context_mismatch)
        Axis._search_policy_matches(searches, options.axis_options) ||
            push!(reasons, :origin_search_policy_mismatch)
        if isempty(reasons)
            topology = classify_figure5b_topology(searches;
                options=Axis._topology_options(lineage_options))
            topology.qualified || append!(reasons, topology.reasons)
        end
        if isempty(reasons)
            central = Lineage._state(topology.central_state)
            norm(collect(supplied) .- collect(central)) <=
                lineage_options.coordinate_atol ||
                push!(reasons, :supplied_center_mismatch)
        end
        if isempty(reasons)
            lineage = Lineage.seed_lineage(searches, central;
                options=lineage_options)
            lineage.accepted || append!(reasons, lineage.reasons)
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :origin_exception)
        error = sprint(showerror, caught)
    end
    unique!(reasons)
    return CurveOriginEvidence(isempty(reasons), Tuple(reasons), supplied,
        central, topology, lineage, searches, error)
end

function _kkt(context, z0, scales, value)
    z = view(value, 1:4)
    multipliers = view(value, 5:7)
    residual = _residual(context, z)
    jacobian = ForwardDiff.jacobian(point -> _residual(context, point), z)
    stationarity = (z .- z0) ./ (scales .^ 2) .+
        transpose(jacobian) * multipliers
    return vcat(residual, stationarity)
end

function _attempts(context, origin, options, solve_function)
    z0 = Float64[origin.central_state...,
        context.origin_parameters...]
    scales = Float64[options.state_scales...,
        context.bounds[1][2] - context.bounds[1][1],
        context.bounds[2][2] - context.bounds[2][1]]
    interior_bounds = ((-Inf, Inf), (-Inf, Inf), context.bounds...,
        (-Inf, Inf), (-Inf, Inf), (-Inf, Inf))
    interior = _safe_solve(solve_function,
        value -> _kkt(context, z0, scales, value),
        vcat(z0, zeros(3)), interior_bounds, options)
    attempts = CurveSeedAttempt[CurveSeedAttempt(:interior_kkt, nothing,
        nothing, interior, Tuple(Float64.(interior.candidate[1:4])))]
    for fixed_axis in 1:2, fixed_value in context.bounds[fixed_axis]
        free_axis = 3 - fixed_axis
        start = Float64[origin.central_state...,
            context.origin_parameters[free_axis]]
        free_bounds = ((-Inf, Inf), (-Inf, Inf), context.bounds[free_axis])
        edge_residual = value -> begin
            z = similar(value, 4)
            z[1], z[2] = value[1], value[2]
            z[fixed_axis + 2] = fixed_value
            z[free_axis + 2] = value[3]
            _residual(context, z)
        end
        solved = _safe_solve(solve_function, edge_residual, start,
            free_bounds, options)
        candidate = zeros(Float64, 4)
        candidate[1:2] .= solved.candidate[1:2]
        candidate[fixed_axis + 2] = fixed_value
        candidate[free_axis + 2] = solved.candidate[3]
        method = Symbol("edge_$(fixed_axis)_$(fixed_value == context.bounds[fixed_axis][1] ? "lower" : "upper")")
        push!(attempts, CurveSeedAttempt(method, fixed_axis, fixed_value,
            solved, Tuple(candidate)))
    end
    return Tuple(attempts)
end

function _rising_slopes(balance, slope_atol)
    abs(balance[1, 2]) > slope_atol &&
        abs(balance[2, 2]) > slope_atol || return nothing
    slopes = (-balance[1, 1] / balance[1, 2],
        -balance[2, 1] / balance[2, 2])
    return all(x -> isfinite(x) && x > slope_atol, slopes) ? slopes : nothing
end

function _neutral_topology(searches, state, options, lineage_options)
    tracks = Lineage.build_root_tracks(searches; options=lineage_options)
    reasons = Symbol[tracks.reasons...]
    tracks.root_counts == (7, 7, 7) || push!(reasons, :root_count_mismatch)
    distances = [maximum(hypot(root[1] - state[1], root[2] - state[2])
        for root in track.states) for track in tracks.tracks]
    matches = findall(x -> x <= lineage_options.coordinate_atol, distances)
    length(matches) == 1 || push!(reasons, :central_root_not_unique)
    central_index = length(matches) == 1 ? only(matches) : nothing
    traces = fill(NaN, 3)
    determinants = fill(NaN, 3)
    frequencies = fill(NaN, 3)
    slopes = [(NaN, NaN) for _ in 1:3]
    if tracks.qualified && central_index !== nothing &&
            tracks.root_counts == (7, 7, 7)
        slope_atol = Axis._topology_options(lineage_options).slope_atol
        for grid in 1:3
            for (index, track) in enumerate(tracks.tracks)
                if index != central_index &&
                        track.classifications[grid] ∉ (Attracting, Saddle)
                    push!(reasons, :outer_stability_pattern_mismatch)
                end
            end
            count(index -> index != central_index &&
                tracks.tracks[index].classifications[grid] == Attracting,
                eachindex(tracks.tracks)) == 3 ||
                push!(reasons, :outer_attractor_count)
            count(index -> index != central_index &&
                tracks.tracks[index].classifications[grid] == Saddle,
                eachindex(tracks.tracks)) == 3 ||
                push!(reasons, :outer_saddle_count)
            model = searches[grid].frozen_model
            center = collect(tracks.tracks[central_index].states[grid])
            ode = zeros(Float64, 2, 2)
            point_jacobian!(ode, center, model, 0.0)
            traces[grid] = ode[1, 1] + ode[2, 2]
            determinants[grid] = ode[1, 1] * ode[2, 2] -
                ode[1, 2] * ode[2, 1]
            radicand = determinants[grid] - traces[grid]^2 / 4
            frequencies[grid] = radicand > 0 ? sqrt(radicand) : NaN
            tracks.tracks[central_index].classifications[grid] ==
                StabilityUnresolved ||
                push!(reasons, :central_not_neutral)
            abs(traces[grid]) <= _trace_zero_atol(options.axis_options,
                lineage_options) ||
                push!(reasons, :central_trace_unresolved)
            determinants[grid] > options.rank_atol &&
                isfinite(frequencies[grid]) &&
                frequencies[grid] > lineage_options.spectral_margin ||
                push!(reasons, :central_imaginary_pair_unresolved)
            balance = zeros(Float64, 2, 2)
            point_balance_jacobian!(balance, center, model, 0.0)
            rising = _rising_slopes(balance, slope_atol)
            if rising === nothing
                push!(reasons, :central_rising_arms_unresolved)
            else
                slopes[grid] = rising
            end
        end
    end
    unique!(reasons)
    return NeutralTopologyEvidence(isempty(reasons), Tuple(reasons), tracks,
        central_index, Tuple(traces), Tuple(determinants),
        Tuple(frequencies), Tuple(slopes))
end

function _rank_probe_widths(z, bounds, options)
    widths = ntuple(axis -> begin
        coordinate = z[axis + 2]
        left_room = coordinate - bounds[axis][1]
        right_room = bounds[axis][2] - coordinate
        base_h = min(options.finite_difference_step *
            max(1.0, abs(coordinate)), max(left_room, right_room) / 2)
        side = right_room >= left_room ? 1.0 : -1.0
        ntuple(level -> abs((coordinate + side * base_h *
            2.0^(-(level - 1))) - coordinate), 3)
    end, 2)
    valid = all(axis -> begin
        coordinate = z[axis + 2]
        resolution = 32eps(Float64) * max(1.0, abs(coordinate))
        values = widths[axis]
        all(isfinite, values) && values[1] > 1.5values[2] >
            2resolution && values[2] > 1.5values[3] > resolution
    end, 1:2)
    return widths, valid
end

function _finite_difference_jacobian(residual_function, z, bounds,
    options, parameter_widths, level)
    base = Float64.(residual_function(z))
    all(isfinite, base) || return nothing
    jacobian = zeros(Float64, 3, 4)
    for column in 1:4
        left, right = copy(z), copy(z)
        if column <= 2
            h = options.finite_difference_step *
                max(1.0, abs(z[column])) * 2.0^(-(level - 1))
            left[column] -= h
            right[column] += h
            denominator = right[column] - left[column]
            denominator > 0 || return nothing
            jacobian[:, column] .= (residual_function(right) .-
                residual_function(left)) ./ denominator
        else
            axis = column - 2
            h = parameter_widths[axis][level]
            lower, upper = bounds[axis]
            if z[column] - lower >= h && upper - z[column] >= h
                left[column] -= h
                right[column] += h
                denominator = right[column] - left[column]
                denominator > 0 || return nothing
                jacobian[:, column] .= (residual_function(right) .-
                    residual_function(left)) ./ denominator
            elseif upper - z[column] >= h
                right[column] += h
                denominator = right[column] - z[column]
                denominator > 0 || return nothing
                jacobian[:, column] .= (residual_function(right) .- base) ./
                    denominator
            elseif z[column] - lower >= h
                left[column] -= h
                denominator = z[column] - left[column]
                denominator > 0 || return nothing
                jacobian[:, column] .= (base .- residual_function(left)) ./
                    denominator
            else
                return nothing
            end
        end
    end
    return all(isfinite, jacobian) ? jacobian : nothing
end

function _multiscale_rank(residual_function, candidate, bounds, options)
    nan_widths = ((NaN, NaN, NaN), (NaN, NaN, NaN))
    z = collect(candidate)
    _in_bounds(z[3:4], bounds) || return CurveRankEvidence(false,
        (:rank_out_of_bounds,), NaN, NaN, NaN, NaN, nan_widths, nothing)
    widths, distinct = _rank_probe_widths(z, bounds, options)
    distinct || return CurveRankEvidence(false, (:rank_probe_unresolved,),
        NaN, NaN, NaN, NaN, widths, nothing)
    bounded_residual = value -> begin
        primal_parameters = (ForwardDiff.value(value[3]),
            ForwardDiff.value(value[4]))
        _in_bounds(primal_parameters, bounds) ||
            throw(ArgumentError("rank probe left the parameter box"))
        residual_function(value)
    end
    jacobians = Matrix{Float64}[]
    try
        for level in 1:3
            jacobian = _finite_difference_jacobian(bounded_residual, z,
                bounds, options, widths, level)
            jacobian === nothing && return CurveRankEvidence(false,
                (:rank_probe_unresolved,), NaN, NaN, NaN, NaN, widths,
                nothing)
            push!(jacobians, jacobian)
        end
        independent = ForwardDiff.jacobian(bounded_residual, z)
        singular = minimum(svdvals(jacobians[3]))
        coarse_difference = opnorm(jacobians[1] - jacobians[2])
        fine_difference = opnorm(jacobians[2] - jacobians[3])
        independent_difference = opnorm(independent - jacobians[3])
        reasons = Symbol[]
        isfinite(singular) && singular > options.rank_atol ||
            push!(reasons, :rank_deficient)
        all(isfinite, (coarse_difference, fine_difference,
            independent_difference)) &&
            coarse_difference <= 0.1singular &&
            fine_difference <= 0.1singular &&
            independent_difference <= 0.1singular ||
            push!(reasons, :rank_multiscale_unresolved)
        return CurveRankEvidence(isempty(reasons), Tuple(reasons), singular,
            coarse_difference, fine_difference, independent_difference,
            widths, nothing)
    catch caught
        caught isa InterruptException && rethrow()
        return CurveRankEvidence(false, (:rank_exception,), NaN, NaN,
            NaN, NaN, widths, sprint(showerror, caught))
    end
end

_regularity(context, candidate, options) = _multiscale_rank(
    value -> _residual(context, value), candidate, context.bounds, options)

function _local_snapshot(solved::EquilibriumSolveResult)
    attempt = solved.attempt
    return Axis.AxisLocalSolveEvidence(Tuple(Float64.(attempt.candidate)),
        attempt.validation, attempt.solver_status, attempt.solver_success,
        Float64(attempt.residual_norm), attempt.near_singular,
        Tuple(attempt.reasons))
end

function _homotopy(context, origin, candidate, options, lineage_options,
    search_function, local_solve_function)
    current_fraction = 0.0
    step = 1.0
    state = origin.central_state
    anchor = origin.lineage.anchor
    trials = CurveHomotopyTrial[]
    terminal_reasons = Symbol[]
    while current_fraction < 1 && length(trials) < options.homotopy_max_trials
        target_fraction = min(1.0, current_fraction + step)
        parameters = ntuple(index -> context.origin_parameters[index] +
            target_fraction * (candidate[index + 2] -
                context.origin_parameters[index]), 2)
        reasons = Symbol[]
        local_solve = nothing
        searches = nothing
        lineage = nothing
        error = nothing
        try
            model = _model(context, parameters)
            solved = local_solve_function(model, state, options.axis_options)
            solved isa EquilibriumSolveResult ||
                throw(ArgumentError("local solve must return EquilibriumSolveResult"))
            local_solve = _local_snapshot(solved)
            Axis._solve_context_matches(solved, model, options.axis_options) ||
                push!(reasons, :local_solve_context_mismatch)
            attempt = solved.attempt
            attempt.validation == AdmissibleCandidate &&
                isfinite(attempt.residual_norm) &&
                attempt.residual_norm <= options.axis_options.balance_atol &&
                !attempt.near_singular ||
                push!(reasons, :local_root_unresolved)
            next_state = all(isfinite, attempt.candidate) &&
                length(attempt.candidate) == 2 ?
                Lineage._state(attempt.candidate) : nothing
            next_state === nothing && push!(reasons, :invalid_local_state)
            if isempty(reasons)
                displacement = Axis._displacement_bound(state, next_state,
                    options.axis_options, lineage_options)
                displacement === nothing && push!(reasons, :state_step_exceeded)
                searches = search_function(model, options.axis_options)
                Axis._context_matches(model, searches) ||
                    push!(reasons, :search_model_mismatch)
                Axis._search_policy_matches(searches, options.axis_options) ||
                    push!(reasons, :search_policy_mismatch)
                if isempty(reasons)
                    lineage = Lineage.transition_lineage(anchor, searches,
                        next_state; options=lineage_options,
                        displacement_atol=displacement)
                    lineage.accepted || append!(reasons, lineage.reasons)
                end
            end
            if isempty(reasons) && target_fraction == 1 &&
                    hypot(next_state[1] - candidate[1],
                        next_state[2] - candidate[2]) >
                    lineage_options.coordinate_atol
                push!(reasons, :terminal_candidate_root_mismatch)
            end
            if isempty(reasons)
                state = next_state
                anchor = lineage.anchor
            end
        catch caught
            caught isa InterruptException && rethrow()
            push!(reasons, :homotopy_exception)
            error = sprint(showerror, caught)
        end
        unique!(reasons)
        accepted = isempty(reasons)
        push!(trials, CurveHomotopyTrial(current_fraction, target_fraction,
            accepted, Tuple(reasons), local_solve, searches, lineage, error))
        if accepted
            current_fraction = target_fraction
            step = min(1.0 - current_fraction, 1.5step)
        else
            step /= 2
            if step < options.homotopy_min_fraction ||
                    current_fraction + step == current_fraction
                append!(terminal_reasons, reasons)
                push!(terminal_reasons, :homotopy_minimum_step)
                break
            end
        end
    end
    if current_fraction < 1 && isempty(terminal_reasons)
        push!(terminal_reasons, :homotopy_maximum_trials)
    end
    unique!(terminal_reasons)
    qualified = current_fraction == 1 && isempty(terminal_reasons)
    return CurveHomotopyEvidence(qualified, Tuple(terminal_reasons),
        Tuple(trials), qualified ? anchor : nothing)
end

function _qualify(context, origin, attempt, options, lineage_options,
    search_function, local_solve_function)
    candidate = attempt.candidate
    reasons = Symbol[]
    residual = (NaN, NaN, NaN)
    minimum_singular = NaN
    rank = nothing
    determinant = NaN
    frequency = NaN
    searches = nothing
    topology = nothing
    homotopy = nothing
    error = nothing
    attempt.solve.converged || push!(reasons, :seed_solver_unresolved)
    all(isfinite, candidate) || push!(reasons, :nonfinite_seed_candidate)
    _in_bounds(candidate[3:4], context.bounds) ||
        push!(reasons, :seed_out_of_bounds)
    if isempty(reasons)
        try
            if attempt.method == :interior_kkt
                z0 = Float64[origin.central_state...,
                    context.origin_parameters...]
                scales = Float64[options.state_scales...,
                    context.bounds[1][2] - context.bounds[1][1],
                    context.bounds[2][2] - context.bounds[2][1]]
                stationarity = Float64.(_kkt(context, z0, scales,
                    collect(attempt.solve.candidate)))
                all(isfinite, stationarity) &&
                    maximum(abs, stationarity) <= options.solver_atol ||
                    push!(reasons, :seed_kkt_unresolved)
            end
            values = Float64.(_residual(context, collect(candidate)))
            residual = Tuple(values)
            all(isfinite, values) && maximum(abs, values[1:2]) <=
                options.axis_options.balance_atol &&
                abs(values[3]) <= _trace_zero_atol(options.axis_options,
                    lineage_options) ||
                push!(reasons, :seed_residual_unresolved)
            model = _model(context, candidate[3:4])
            ode = zeros(Float64, 2, 2)
            point_jacobian!(ode, collect(candidate[1:2]), model, 0.0)
            determinant = ode[1, 1] * ode[2, 2] - ode[1, 2] * ode[2, 1]
            radicand = determinant - (ode[1, 1] + ode[2, 2])^2 / 4
            frequency = radicand > 0 ? sqrt(radicand) : NaN
            determinant > options.rank_atol && isfinite(frequency) &&
                frequency > lineage_options.spectral_margin ||
                push!(reasons, :seed_hopf_pair_unresolved)
            rank = _regularity(context, candidate, options)
            minimum_singular = rank.minimum_singular_value
            rank.qualified || append!(reasons, rank.reasons)
            if isempty(reasons)
                searches = search_function(model, options.axis_options)
                Axis._context_matches(model, searches) ||
                    push!(reasons, :search_model_mismatch)
                Axis._search_policy_matches(searches, options.axis_options) ||
                    push!(reasons, :search_policy_mismatch)
                if isempty(reasons)
                    topology = _neutral_topology(searches,
                        candidate[1:2], options, lineage_options)
                    topology.qualified || append!(reasons, topology.reasons)
                end
            end
            if isempty(reasons)
                homotopy = _homotopy(context, origin, candidate, options,
                    lineage_options, search_function, local_solve_function)
                homotopy.qualified || append!(reasons, homotopy.reasons)
            end
            if isempty(reasons) &&
                    homotopy.terminal_anchor.states !=
                    topology.tracks.tracks[topology.central_track].states
                push!(reasons, :origin_central_lineage_mismatch)
            end
        catch caught
            caught isa InterruptException && rethrow()
            push!(reasons, :seed_qualification_exception)
            error = sprint(showerror, caught)
        end
    end
    unique!(reasons)
    return CurveSeedQualification(isempty(reasons), Tuple(reasons),
        candidate, residual, minimum_singular, rank, determinant,
        frequency, searches, topology, homotopy, error)
end

function _qualified_indices(attempts, qualifications, context, options)
    scales = (options.state_scales...,
        context.bounds[1][2] - context.bounds[1][1],
        context.bounds[2][2] - context.bounds[2][1])
    selected = Int[]
    for index in eachindex(attempts)
        qualifications[index].accepted || continue
        candidate = qualifications[index].candidate
        any(previous -> norm(collect((candidate .-
            qualifications[previous].candidate) ./ scales)) <=
            options.dedup_atol, selected) || push!(selected, index)
    end
    return Tuple(selected)
end

function _unresolved_methods(attempts, qualifications)
    return Tuple(attempts[index].method for index in eachindex(attempts)
        if !attempts[index].solve.converged ||
            attempts[index].solve.error !== nothing ||
            qualifications[index].error !== nothing ||
            any(reason -> reason ∉ FINITE_REJECTION_REASONS,
                qualifications[index].reasons))
end

"""
    seed_trace_zero_curve(base_model, pair, central_state; parameter_bounds, ...)

Attempt a bounded interior stationary-distance KKT solve and all four rectangle edges from
one independently confirmed Figure-5b central root. Each numerical candidate
is separately qualified by fresh neutral topology, independent observations,
regularity, and reciprocal root-lineage homotopy. No qualified seed is a
finite local negative only when every numerical method completed and had a
resolved rejection; otherwise the search status is unresolved. Neither status
certifies absence of trace-zero or Hopf curves.
"""
function seed_trace_zero_curve(base_model, pair, central_state;
    parameter_bounds, options=CurveSeedOptions(),
    lineage_options=Lineage.RootLineageOptions())
    return _seed_trace_zero_curve(base_model, pair, central_state;
        parameter_bounds, options, lineage_options,
        search_function=Axis._searches,
        solve_function=_solve_bounded,
        local_solve_function=Axis._solve)
end

function _seed_trace_zero_curve(base_model, pair, central_state;
    parameter_bounds, options=CurveSeedOptions(),
    lineage_options=Lineage.RootLineageOptions(),
    search_function=Axis._searches, solve_function=_solve_bounded,
    local_solve_function=Axis._solve)
    options isa CurveSeedOptions ||
        throw(ArgumentError("options must be CurveSeedOptions"))
    lineage_options isa Lineage.RootLineageOptions ||
        throw(ArgumentError("lineage_options must be RootLineageOptions"))
    supplied = Lineage._state(central_state)
    context = _context(base_model, pair, parameter_bounds)
    origin = _origin(context, supplied, options, lineage_options,
        search_function)
    origin.accepted || return CurveSeedResult(context, options,
        lineage_options, origin, (), (), (), (),
        :origin_unresolved)
    attempts = _attempts(context, origin, options, solve_function)
    qualifications = Tuple(_qualify(context, origin, attempt, options,
        lineage_options, search_function, local_solve_function)
        for attempt in attempts)
    indices = _qualified_indices(attempts, qualifications, context, options)
    unresolved_methods = _unresolved_methods(attempts, qualifications)
    status = !isempty(indices) ? :qualified_seeds :
        isempty(unresolved_methods) ? :no_qualified_seed :
        :search_unresolved
    return CurveSeedResult(context, options, lineage_options, origin,
        attempts, qualifications, indices, unresolved_methods, status)
end

end # module
