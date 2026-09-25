module Figure5bCurveTraversal

using FailureOfInhibition2025
using ForwardDiff
using LinearAlgebra: norm, dot, opnorm, svd, svdvals, Diagonal

include("figure5b_curve_seeds.jl")
const Seeds = Figure5bCurveSeeds
const Axis = Seeds.Axis
const Lineage = Seeds.Lineage
const State = Seeds.State
const CurveState = Seeds.CurveState

"""Validated policy for a finite, normalized pseudo-arclength trace-zero traversal."""
struct CurveTraversalOptions
    initial_step::Float64
    minimum_step::Float64
    maximum_step::Float64
    max_steps::Int
    max_retries::Int
    correction_atol::Float64
    phase_atol::Float64
    rank_atol::Float64
    progress_fraction::Float64
    correction_fraction::Float64
    revisit_atol::Float64
    boundary_atol::Float64
    transversality_atol::Float64
    function CurveTraversalOptions(initial_step::Float64,
        minimum_step::Float64, maximum_step::Float64, max_steps::Int,
        max_retries::Int, correction_atol::Float64, phase_atol::Float64,
        rank_atol::Float64, progress_fraction::Float64,
        correction_fraction::Float64, revisit_atol::Float64,
        boundary_atol::Float64, transversality_atol::Float64)
        values = (initial_step, minimum_step, maximum_step, correction_atol,
            phase_atol, rank_atol, progress_fraction, correction_fraction,
            revisit_atol, boundary_atol, transversality_atol)
        all(value -> isfinite(value) && value > 0, values) &&
            minimum_step <= initial_step <= maximum_step &&
            progress_fraction < 1 && correction_fraction < 1 &&
            max_steps > 0 && max_retries >= 0 ||
            throw(ArgumentError("invalid curve traversal policy"))
        revisit_atol < progress_fraction * minimum_step ||
            throw(ArgumentError("revisit_atol must stay strictly below the " *
                "smallest forward progress a legal step can make, or a " *
                "perfectly forward step reads as a revisit"))
        new(initial_step, minimum_step, maximum_step, max_steps, max_retries,
            correction_atol, phase_atol, rank_atol, progress_fraction,
            correction_fraction, revisit_atol, boundary_atol,
            transversality_atol)
    end
end

function CurveTraversalOptions(; initial_step=0.01, minimum_step=1e-5,
    maximum_step=0.05, max_steps=200, max_retries=8,
    correction_atol=1e-8, phase_atol=1e-8, rank_atol=1e-10,
    progress_fraction=0.1, correction_fraction=0.75,
    revisit_atol=1e-7, boundary_atol=1e-8,
    transversality_atol=1e-6)
    raw = (initial_step, minimum_step, maximum_step, correction_atol,
        phase_atol, rank_atol, progress_fraction, correction_fraction,
        revisit_atol, boundary_atol, transversality_atol)
    all(value -> value isa Real && !(value isa Bool) && isfinite(value) &&
        value > 0, raw) || throw(ArgumentError("curve tolerances must be positive finite reals"))
    all(value -> value isa Integer && !(value isa Bool) &&
        0 <= value <= typemax(Int), (max_steps, max_retries)) ||
        throw(ArgumentError("curve limits must be nonnegative integers"))
    return CurveTraversalOptions(Float64.(raw[1:3])..., Int(max_steps),
        Int(max_retries), Float64.(raw[4:end])...)
end

struct VerifiedCurveSeed{S}
    context::Seeds.CurveContext
    source::Symbol
    source_index::Int
    point::CurveState
    options::Seeds.CurveSeedOptions
    lineage_options::Lineage.RootLineageOptions
    original_unresolved_methods::Tuple{Vararg{Symbol}}
    source_evidence::S
    anchor::Lineage.RootLineageAnchor
end

struct SeedVerification
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    seed::Union{Nothing,VerifiedCurveSeed}
    error::Union{Nothing,String}
end

struct CurveTangent
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    actual::CurveState
    scaled::CurveState
    scaled_minimum_singular::Float64
    rank::Union{Nothing,Seeds.CurveRankEvidence}
    error::Union{Nothing,String}
end

struct CurvePoint
    coordinates::CurveState
    tangent::CurveTangent
    residual::NTuple{3,Float64}
    anchor::Lineage.RootLineageAnchor
    searches::Tuple
    transition::Union{Nothing,Lineage.RootTransitionEvidence}
    hopf::Seeds.NeutralTopologyEvidence
end

struct CurveStepAttempt
    direction::Int
    step::Float64
    retry::Int
    predictor::CurveState
    raw::Union{Nothing,Seeds.SeedSolveResult}
    candidate::CurveState
    residual::NTuple{3,Float64}
    phase::Float64
    augmented_minimum_singular::Float64
    progress::Float64
    correction_distance::Float64
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    searches::Union{Nothing,Tuple}
    transition::Union{Nothing,Lineage.RootTransitionEvidence}
    tangent::Union{Nothing,CurveTangent}
    error::Union{Nothing,String}
end

struct CurveBoundaryAttempt
    direction::Int
    step::Float64
    edge_axis::Int
    edge_value::Float64
    ray_distance::Float64
    predictor::CurveState
    raw::Union{Nothing,Seeds.SeedSolveResult}
    candidate::CurveState
    residual::NTuple{3,Float64}
    phase::Float64
    edge_minimum_singular::Float64
    progress::Float64
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    searches::Union{Nothing,Tuple}
    transition::Union{Nothing,Lineage.RootTransitionEvidence}
    tangent::Union{Nothing,CurveTangent}
    hopf::Union{Nothing,Seeds.NeutralTopologyEvidence}
    error::Union{Nothing,String}
end

"""
A validated parameter-box exit: the corrected state, the lineage anchor it was
accepted against, its neutrality evidence, and the box face it crossed. Carried
as a distinct field so a consumer never has to rediscover it inside a
`boundaries` tuple that also holds rejected and ambiguous attempts.
"""
struct QualifiedEndpoint
    coordinates::CurveState
    anchor::Lineage.RootLineageAnchor
    transition::Union{Nothing,Lineage.RootTransitionEvidence}
    hopf::Seeds.NeutralTopologyEvidence
    edge_axis::Int
    edge_value::Float64
end

struct CurveDirectionResult
    direction::Int
    points::Tuple{Vararg{CurvePoint}}
    steps::Tuple{Vararg{CurveStepAttempt}}
    boundaries::Tuple{Vararg{CurveBoundaryAttempt}}
    endpoint::Union{Nothing,QualifiedEndpoint}
    unqualified_points::Int
    termination::Symbol
end

const BOUNDARY_TERMINATIONS = (:qualified_boundary, :unqualified_boundary)

_qualified_endpoint(attempt::CurveBoundaryAttempt) =
    QualifiedEndpoint(attempt.candidate, attempt.transition.anchor,
        attempt.transition, attempt.hopf, attempt.edge_axis,
        attempt.edge_value)

_neutral_endpoint(attempt::CurveBoundaryAttempt) =
    attempt.hopf !== nothing && attempt.hopf.qualified

struct TraceZeroCurveResult
    seed_verification::SeedVerification
    options::CurveTraversalOptions
    initial_reasons::Tuple{Vararg{Symbol}}
    initial_error::Union{Nothing,String}
    directions::Tuple{Vararg{CurveDirectionResult}}
    status::Symbol
end

_nan_state() = (NaN, NaN, NaN, NaN)
_nan_residual() = (NaN, NaN, NaN)
_scales(seed::VerifiedCurveSeed) = (seed.options.state_scales...,
    seed.context.bounds[1][2] - seed.context.bounds[1][1],
    seed.context.bounds[2][2] - seed.context.bounds[2][1])
_scaled_distance(left, right, scales) = norm((collect(left) .-
    collect(right)) ./ collect(scales))
_revisited(visited, candidate, scales, tolerance) =
    any(old -> _scaled_distance(old, candidate, scales) <= tolerance,
        visited)

function _seed_method_valid(attempt, context)
    candidate = attempt.candidate
    solve = attempt.solve
    if attempt.method == :interior_kkt
        return attempt.fixed_axis === nothing &&
            attempt.fixed_value === nothing && length(solve.candidate) == 7 &&
            candidate == solve.candidate[1:4]
    end
    for axis in 1:2, (side, value) in enumerate(context.bounds[axis])
        method = Symbol("edge_$(axis)_$(side == 1 ? "lower" : "upper")")
        if attempt.method == method
            free = 3 - axis
            return attempt.fixed_axis == axis &&
                attempt.fixed_value == value &&
                length(solve.candidate) == 3 &&
                candidate[axis + 2] == value &&
                candidate[free + 2] == solve.candidate[3] &&
                candidate[1:2] == solve.candidate[1:2]
        end
    end
    return false
end

function _verify_seed(result::Seeds.CurveSeedResult, index;
    search_function=Axis._searches, local_solve_function=Axis._solve)
    index isa Integer && !(index isa Bool) && 1 <= index <=
        length(result.attempts) || return SeedVerification(false,
        (:invalid_seed_index,), nothing, nothing)
    reasons = Symbol[]
    result.status == :qualified_seeds || push!(reasons, :seed_status_unqualified)
    index in result.qualified_indices || push!(reasons, :seed_not_qualified_member)
    length(result.attempts) == length(result.qualifications) == 5 ||
        push!(reasons, :seed_record_shape_mismatch)
    !isempty(reasons) && return SeedVerification(false, Tuple(reasons),
        nothing, nothing)
    attempt = result.attempts[index]
    original = result.qualifications[index]
    original.accepted && original.candidate == attempt.candidate &&
        _seed_method_valid(attempt, result.context) ||
        push!(reasons, :seed_record_mismatch)
    context = nothing
    origin = nothing
    qualification = nothing
    error = nothing
    try
        context = Seeds._context(result.context.base_model,
            result.context.pair, result.context.bounds)
        context.pair == result.context.pair &&
            context.bounds == result.context.bounds &&
            context.origin_parameters == result.context.origin_parameters &&
            isequal(Lineage._point_model_identity(context.base_model),
                Lineage._point_model_identity(result.context.base_model)) ||
            push!(reasons, :seed_context_mismatch)
        if isempty(reasons)
            origin = Seeds._origin(context, result.origin.supplied_state,
                result.options, result.lineage_options, search_function)
            origin.accepted || push!(reasons, :fresh_origin_unresolved)
        end
        if isempty(reasons)
            qualification = Seeds._qualify(context, origin, attempt,
                result.options, result.lineage_options, search_function,
                local_solve_function)
            qualification.accepted || push!(reasons, :fresh_seed_unresolved)
        end
        if isempty(reasons) && (qualification.candidate != original.candidate ||
                qualification.homotopy.terminal_anchor.states !=
                    qualification.topology.tracks.tracks[
                        qualification.topology.central_track].states)
            push!(reasons, :fresh_seed_identity_mismatch)
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :seed_verification_exception)
        error = sprint(showerror, caught)
    end
    if !isempty(reasons)
        return SeedVerification(false, Tuple(unique(reasons)), nothing, error)
    end
    qualification.homotopy.terminal_anchor.model_identity ==
        Lineage._model_identity(qualification.searches[1]) &&
        qualification.homotopy.terminal_anchor.origin_model_identity ==
        Lineage._model_identity(origin.searches[1]) ||
        return SeedVerification(false, (:seed_anchor_identity_mismatch,),
            nothing, nothing)
    verified = VerifiedCurveSeed(context, :pr3_seed, Int(index),
        attempt.candidate, result.options, result.lineage_options,
        result.unresolved_methods, (origin, qualification),
        qualification.homotopy.terminal_anchor)
    return SeedVerification(true, (), verified, nothing)
end

function _tangent(residual_function, point, scales, rank_atol;
    prior=nothing, rank=nothing)
    reasons = Symbol[]
    actual = _nan_state()
    scaled = _nan_state()
    minimum_singular = NaN
    error = nothing
    try
        jacobian = ForwardDiff.jacobian(residual_function, collect(point))
        decomposition = svd(jacobian * Diagonal(collect(scales)); full=true)
        minimum_singular = minimum(decomposition.S)
        isfinite(minimum_singular) && minimum_singular > rank_atol &&
            (rank === nothing || rank.qualified) ||
            push!(reasons, :curve_rank_unresolved)
        if isempty(reasons)
            direction = collect(decomposition.V[:, end])
            direction ./= norm(direction)
            if prior === nothing
                chosen = abs(direction[3]) >= abs(direction[4]) ? 3 : 4
                abs(direction[chosen]) > rank_atol ||
                    push!(reasons, :parameter_tangent_unresolved)
                if isempty(reasons) && direction[chosen] < 0
                    direction .*= -1
                end
            else
                alignment = dot(direction, collect(prior))
                abs(alignment) > rank_atol ||
                    push!(reasons, :tangent_orientation_unresolved)
                if isempty(reasons) && alignment < 0
                    direction .*= -1
                end
            end
            if isempty(reasons)
                scaled = Tuple(direction)
                actual = Tuple(direction .* collect(scales))
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :tangent_exception)
        error = sprint(showerror, caught)
    end
    return CurveTangent(isempty(reasons), Tuple(reasons), actual, scaled,
        minimum_singular, rank, error)
end

_phase(candidate, predictor, tangent, scales) = dot((collect(candidate) .-
    collect(predictor)) ./ collect(scales), collect(tangent.scaled))

_outward_transverse(tangent, axis, side, tolerance) =
    tangent.accepted && side * tangent.actual[axis + 2] > tolerance

function _geometry(current, predictor, candidate, tangent, scales, step,
    options; allow_zero=false)
    correction = _scaled_distance(candidate, predictor, scales)
    displacement = _scaled_distance(candidate, current, scales)
    progress = dot((collect(candidate) .- collect(current)) ./ collect(scales),
        collect(tangent.scaled))
    reasons = Symbol[]
    isfinite(correction) && correction <= max(options.correction_atol,
        options.correction_fraction * step) ||
        push!(reasons, :nonlocal_predictor_correction)
    isfinite(displacement) && displacement <=
        step * (1 + options.correction_fraction) + options.correction_atol ||
        push!(reasons, :nonlocal_current_advance)
    threshold = allow_zero ? -options.correction_atol :
        options.progress_fraction * step
    isfinite(progress) && progress >= threshold ||
        push!(reasons, :stagnant_or_backward_correction)
    return progress, correction, Tuple(reasons)
end

function _ray_hits(current, tangent, bounds, step, tolerance)
    hits = Tuple{Int,Float64,Float64}[]
    for axis in 1:2
        speed = tangent.actual[axis + 2]
        abs(speed) > tolerance || continue
        value = speed > 0 ? bounds[axis][2] : bounds[axis][1]
        distance = (value - current[axis + 2]) / speed
        -tolerance <= distance <= step + tolerance &&
            push!(hits, (axis, value, max(0.0, distance)))
    end
    isempty(hits) && return ()
    nearest = minimum(hit[3] for hit in hits)
    return Tuple(hit for hit in hits if abs(hit[3] - nearest) <= tolerance)
end

function _first_ray_hit(current, tangent, bounds, step, tolerance, hit)
    axis, value, distance = hit
    axis in 1:2 && all(isfinite, (value, distance)) &&
        0 <= distance <= step + tolerance || return false
    return any(first -> first[1] == axis && first[2] == value &&
        abs(first[3] - distance) <= tolerance,
        _ray_hits(current, tangent, bounds, step, tolerance))
end

function _search_transition(seed, source_anchor, point, predictor,
    search_function)
    reasons = Symbol[]
    searches = nothing
    transition = nothing
    hopf = nothing
    error = nothing
    try
        model = Seeds._model(seed.context, point[3:4])
        searches = search_function(model, seed.options.axis_options)
        Axis._context_matches(model, searches) ||
            push!(reasons, :search_context_mismatch)
        Axis._search_policy_matches(searches, seed.options.axis_options) ||
            push!(reasons, :search_policy_mismatch)
        if isempty(reasons)
            source = source_anchor.states[3]
            # The displacement bound is shared with the axis component so the
            # two integrations cannot drift on the one numeric limit both must
            # agree on.
            bound = Axis._displacement_bound(source, point[1:2],
                seed.options.axis_options, seed.lineage_options)
            bound === nothing && push!(reasons, :state_step_exceeded)
            predictor_error = hypot(point[1] - predictor[1],
                point[2] - predictor[2])
            predictor_atol = max(predictor_error +
                2seed.lineage_options.coordinate_atol,
                2seed.lineage_options.coordinate_atol)
            predictor_error <= seed.options.axis_options.state_step_atol ||
                push!(reasons, :state_predictor_nonlocal)
            if isempty(reasons)
                transition = Lineage.transition_lineage(source_anchor,
                    searches, point[1:2]; options=seed.lineage_options,
                    displacement_atol=bound,
                    predicted_state=predictor[1:2],
                    predictor_atol=predictor_atol)
                transition.accepted || append!(reasons, transition.reasons)
            end
            # Hopf topology is recorded separately from root-lineage acceptance.
            hopf = Seeds._neutral_topology(searches, point[1:2],
                seed.options, seed.lineage_options)
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :search_transition_exception)
        error = sprint(showerror, caught)
    end
    return searches, transition, hopf, Tuple(unique(reasons)), error
end

function _initial_point(seed, options, search_function)
    reasons = Symbol[]
    point = seed.point
    residual = _nan_residual()
    rank = nothing
    tangent = nothing
    searches = nothing
    transition = nothing
    hopf = nothing
    error = nothing
    try
        values = Tuple(Float64.(Seeds._residual(seed.context, collect(point))))
        residual = values
        all(isfinite, values) && maximum(abs, values[1:2]) <=
            seed.options.axis_options.balance_atol &&
            abs(values[3]) <= min(options.correction_atol,
                seed.options.axis_options.trace_atol,
                Seeds.MAX_NEUTRAL_TRACE_ATOL) ||
            push!(reasons, :seed_residual_unresolved)
        rank = Seeds._regularity(seed.context, point, seed.options)
        rank.qualified || append!(reasons, rank.reasons)
        if isempty(reasons)
            tangent = _tangent(z -> Seeds._residual(seed.context, z),
                point, _scales(seed), options.rank_atol; rank)
            tangent.accepted || append!(reasons, tangent.reasons)
        end
        if isempty(reasons)
            searches, transition, hopf, tracking_reasons, error =
                _search_transition(seed, seed.anchor, point, point,
                    search_function)
            append!(reasons, tracking_reasons)
            searches !== nothing && seed.anchor.model_identity !=
                Lineage._model_identity(searches[1]) &&
                push!(reasons, :seed_anchor_model_mismatch)
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :seed_point_exception)
        error = sprint(showerror, caught)
    end
    if !isempty(reasons)
        return nothing, Tuple(unique(reasons)), error
    end
    return CurvePoint(point, tangent, residual, transition.anchor,
        searches, transition, hopf), (), nothing
end

function _solver_options(seed, options)
    return Seeds.CurveSeedOptions(axis_options=seed.options.axis_options,
        solver_atol=min(seed.options.solver_atol,
            options.correction_atol, options.phase_atol),
        maxiters=seed.options.maxiters,
        max_backtracks=seed.options.max_backtracks,
        rank_atol=min(seed.options.rank_atol, options.rank_atol),
        finite_difference_step=seed.options.finite_difference_step,
        state_scales=seed.options.state_scales,
        homotopy_min_fraction=seed.options.homotopy_min_fraction,
        homotopy_max_trials=seed.options.homotopy_max_trials,
        dedup_atol=seed.options.dedup_atol)
end

function _corrector_residual(seed, predictor, tangent, scales)
    return z -> vcat(Seeds._residual(seed.context, z),
        _phase(z, predictor, tangent, scales))
end

function _corrector(seed, predictor, tangent, scales, options, solve_function)
    augmented = _corrector_residual(seed, predictor, tangent, scales)
    bounds = ((-Inf, Inf), (-Inf, Inf), seed.context.bounds...)
    Seeds._in_bounds(predictor[3:4], seed.context.bounds) ||
        return Seeds.SeedSolveResult(false, :predictor_out_of_bounds,
            predictor, Inf, (), nothing)
    return Seeds._safe_solve(solve_function, augmented,
        collect(predictor), bounds, _solver_options(seed, options))
end

function _step_candidate(seed, current, predictor, direction, step, retry,
    options, solve_function, search_function, visited)
    scales = _scales(seed)
    raw = _corrector(seed, predictor, current.tangent, scales,
        options, solve_function)
    candidate = Tuple(raw.candidate)
    reasons = Symbol[]
    residual = _nan_residual()
    phase = NaN
    augmented_minimum_singular = NaN
    progress = NaN
    correction_distance = NaN
    searches = nothing
    transition = nothing
    tangent = nothing
    hopf = nothing
    error = raw.error
    raw.converged || push!(reasons, :corrector_unresolved)
    try
        if length(candidate) != 4 || !all(isfinite, candidate)
            push!(reasons, :invalid_correction_candidate)
            candidate = _nan_state()
        else
            phase = _phase(candidate, predictor, current.tangent, scales)
            isfinite(phase) && abs(phase) <= options.phase_atol ||
                push!(reasons, :phase_unresolved)
            if Seeds._in_bounds(candidate[3:4], seed.context.bounds)
                values = Float64.(Seeds._residual(seed.context,
                    collect(candidate)))
                residual = Tuple(values)
                all(isfinite, values) && maximum(abs, values[1:2]) <=
                    seed.options.axis_options.balance_atol &&
                    abs(values[3]) <= min(options.correction_atol,
                        seed.options.axis_options.trace_atol,
                        Seeds.MAX_NEUTRAL_TRACE_ATOL) ||
                    push!(reasons, :independent_residual_unresolved)
                jacobian = ForwardDiff.jacobian(
                    _corrector_residual(seed, predictor, current.tangent,
                        scales), collect(candidate))
                augmented_minimum_singular = minimum(svdvals(jacobian))
                isfinite(augmented_minimum_singular) &&
                    augmented_minimum_singular > options.rank_atol ||
                    push!(reasons, :augmented_rank_unresolved)
            else
                push!(reasons, :candidate_out_of_bounds)
            end
            progress, correction_distance, geometry_reasons = _geometry(
                current.coordinates, predictor, candidate,
                current.tangent, scales, step, options)
            append!(reasons, geometry_reasons)
            _revisited(visited, candidate, scales,
                options.revisit_atol) &&
                push!(reasons, :curve_revisit)
            if Seeds._in_bounds(candidate[3:4], seed.context.bounds)
                rank = Seeds._regularity(seed.context, candidate, seed.options)
                rank.qualified || append!(reasons, rank.reasons)
            end
            if isempty(reasons)
                tangent = _tangent(z -> Seeds._residual(seed.context, z),
                    candidate, scales, options.rank_atol;
                    prior=current.tangent.scaled, rank)
                tangent.accepted || append!(reasons, tangent.reasons)
            end
            if isempty(reasons)
                searches, transition, hopf, tracking_reasons,
                    tracking_error = _search_transition(seed, current.anchor,
                    candidate, predictor, search_function)
                append!(reasons, tracking_reasons)
                error = tracking_error
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :step_validation_exception)
        error = sprint(showerror, caught)
    end
    unique!(reasons)
    accepted = isempty(reasons)
    attempt = CurveStepAttempt(direction, step, retry, predictor, raw,
        candidate, residual, phase, augmented_minimum_singular,
        progress, correction_distance, accepted, Tuple(reasons), searches,
        transition, tangent, error)
    point = accepted ? CurvePoint(candidate, tangent, residual,
        transition.anchor, searches, transition, hopf) : nothing
    return attempt, point
end

function _edge_residual(seed, axis, value)
    free = 3 - axis
    return coordinates -> begin
        z = similar(coordinates, 4)
        z[1], z[2] = coordinates[1], coordinates[2]
        z[axis + 2] = value
        z[free + 2] = coordinates[3]
        Seeds._residual(seed.context, z)
    end
end

function _edge_candidate(coordinates, axis, value)
    length(coordinates) == 3 || return _nan_state()
    free = 3 - axis
    candidate = zeros(Float64, 4)
    candidate[1:2] .= coordinates[1:2]
    candidate[axis + 2] = value
    candidate[free + 2] = coordinates[3]
    return Tuple(candidate)
end

function _edge_attempt(seed, current, direction, step, hit,
    options, solve_function, search_function)
    axis, value, distance = hit
    scales = _scales(seed)
    ray = Tuple(collect(current.coordinates) .+
        distance .* collect(current.tangent.actual))
    if !_first_ray_hit(current.coordinates, current.tangent,
            seed.context.bounds, step, options.boundary_atol, hit)
        return CurveBoundaryAttempt(direction, step, Int(axis), Float64(value),
            Float64(distance), ray, nothing, _nan_state(),
            _nan_residual(), NaN, NaN, NaN, false,
            (:wrong_edge_ray,), nothing, nothing, nothing, nothing,
            nothing)
    end
    free = 3 - axis
    residual_function = _edge_residual(seed, axis, value)
    initial = [ray[1], ray[2], ray[free + 2]]
    bounds = ((-Inf, Inf), (-Inf, Inf), seed.context.bounds[free])
    raw = Seeds._in_bounds(ray[3:4], seed.context.bounds) ?
        Seeds._safe_solve(solve_function, residual_function,
            initial, bounds, _solver_options(seed, options)) :
        Seeds.SeedSolveResult(false, :edge_ray_out_of_bounds,
            Tuple(initial), Inf, (), nothing)
    candidate = _edge_candidate(raw.candidate, axis, value)
    reasons = Symbol[]
    raw.converged || push!(reasons, :edge_solver_unresolved)
    residual = _nan_residual()
    phase = NaN
    edge_minimum_singular = NaN
    progress = NaN
    searches = nothing
    transition = nothing
    edge_tangent = nothing
    hopf = nothing
    error = raw.error
    try
        if !all(isfinite, candidate)
            push!(reasons, :invalid_edge_candidate)
        else
            phase = _phase(candidate, ray, current.tangent, scales)
            isfinite(phase) && abs(phase) <= options.phase_atol ||
                push!(reasons, :edge_phase_unresolved)
            if Seeds._in_bounds(candidate[3:4], seed.context.bounds)
                values = Float64.(Seeds._residual(seed.context,
                    collect(candidate)))
                residual = Tuple(values)
                all(isfinite, values) && maximum(abs, values[1:2]) <=
                    seed.options.axis_options.balance_atol &&
                    abs(values[3]) <= min(options.correction_atol,
                        seed.options.axis_options.trace_atol,
                        Seeds.MAX_NEUTRAL_TRACE_ATOL) ||
                    push!(reasons, :edge_residual_unresolved)
            else
                push!(reasons, :edge_other_parameter_out_of_bounds)
            end
            side = value == seed.context.bounds[axis][2] ? 1.0 : -1.0
            _outward_transverse(current.tangent, axis, side,
                options.transversality_atol) ||
                push!(reasons, :edge_not_outward_transverse)
            progress, _, geometry_reasons = _geometry(
                current.coordinates, ray, candidate, current.tangent,
                scales, distance, options; allow_zero=distance == 0)
            append!(reasons, geometry_reasons)
            if distance == 0 && _scaled_distance(candidate,
                    current.coordinates, scales) > options.correction_atol
                push!(reasons, :edge_start_nonlocal)
            end
            if Seeds._in_bounds(candidate[3:4], seed.context.bounds)
                edge_jacobian = ForwardDiff.jacobian(residual_function,
                    [candidate[1], candidate[2], candidate[free + 2]])
                edge_minimum_singular = minimum(svdvals(edge_jacobian))
                finite_minimum, stable = Axis._stable_augmented_rank(
                    residual_function,
                    [candidate[1], candidate[2], candidate[free + 2]],
                    seed.context.bounds[free], seed.options.axis_options)
                independent = Axis._augmented_jacobian(residual_function,
                    [candidate[1], candidate[2], candidate[free + 2]],
                    seed.context.bounds[free],
                    seed.options.axis_options.finite_difference_step)
                stable && isfinite(edge_minimum_singular) &&
                    edge_minimum_singular > options.rank_atol &&
                    independent !== nothing &&
                    opnorm(edge_jacobian - independent) <=
                        0.1min(edge_minimum_singular, finite_minimum) ||
                    push!(reasons, :edge_rank_unresolved)
                curve_rank = Seeds._regularity(seed.context, candidate,
                    seed.options)
                curve_rank.qualified || append!(reasons, curve_rank.reasons)
            end
            if isempty(reasons)
                edge_tangent = _tangent(z -> Seeds._residual(seed.context, z),
                    candidate, scales, options.rank_atol;
                    prior=current.tangent.scaled, rank=curve_rank)
                edge_tangent.accepted || append!(reasons,
                    edge_tangent.reasons)
                if edge_tangent.accepted &&
                        !_outward_transverse(edge_tangent, axis, side,
                            options.transversality_atol)
                    push!(reasons, :corrected_edge_not_outward_transverse)
                end
            end
            if isempty(reasons)
                searches, transition, hopf, tracking_reasons,
                    tracking_error = _search_transition(seed, current.anchor,
                    candidate, ray, search_function)
                append!(reasons, tracking_reasons)
                error = tracking_error
            end
        end
    catch caught
        caught isa InterruptException && rethrow()
        push!(reasons, :edge_validation_exception)
        error = sprint(showerror, caught)
    end
    unique!(reasons)
    return CurveBoundaryAttempt(direction, step, axis, value, distance,
        ray, raw, candidate, residual, phase, edge_minimum_singular,
        progress, isempty(reasons), Tuple(reasons), searches, transition,
        edge_tangent, hopf, error)
end

function _mark_ambiguous(attempt::CurveBoundaryAttempt)
    return CurveBoundaryAttempt(attempt.direction, attempt.step,
        attempt.edge_axis, attempt.edge_value, attempt.ray_distance,
        attempt.predictor, attempt.raw, attempt.candidate,
        attempt.residual, attempt.phase, attempt.edge_minimum_singular,
        attempt.progress, false,
        (attempt.reasons..., :ambiguous_corner_exit), attempt.searches,
        attempt.transition, attempt.tangent, attempt.hopf, attempt.error)
end

_unqualified_count(points) =
    count(point -> !point.hopf.qualified, points)

function _signed_point(point, sign)
    tangent = point.tangent
    directed = CurveTangent(tangent.accepted, tangent.reasons,
        Tuple(sign .* collect(tangent.actual)),
        Tuple(sign .* collect(tangent.scaled)),
        tangent.scaled_minimum_singular, tangent.rank, tangent.error)
    return CurvePoint(point.coordinates, directed, point.residual,
        point.anchor, point.searches, point.transition, point.hopf)
end

function _direction(seed, initial, direction, options,
    solve_function, search_function)
    points = CurvePoint[_signed_point(initial, direction)]
    attempts = CurveStepAttempt[]
    boundaries = CurveBoundaryAttempt[]
    visited = CurveState[initial.coordinates]
    step = options.initial_step
    termination = :maximum_steps_unresolved
    direction_result(termination, endpoint) = CurveDirectionResult(direction,
        Tuple(points), Tuple(attempts), Tuple(boundaries), endpoint,
        _unqualified_count(points), termination)
    while length(points) - 1 < options.max_steps
        current = last(points)
        accepted = false
        for retry in 0:options.max_retries
            hits = _ray_hits(current.coordinates, current.tangent,
                seed.context.bounds, step, options.boundary_atol)
            if !isempty(hits)
                edge_attempts = Tuple(_edge_attempt(seed, current, direction,
                    step, hit, options, solve_function, search_function)
                    for hit in hits)
                if length(hits) > 1
                    append!(boundaries, _mark_ambiguous.(edge_attempts))
                else
                    append!(boundaries, edge_attempts)
                end
                # A validated exit is reported as qualified only when its own
                # neutrality evidence agrees. Otherwise the geometry is real but
                # the point is not an established Hopf candidate, which is a
                # distinct recorded outcome rather than a qualified boundary.
                if length(hits) == 1 && only(edge_attempts).accepted
                    exit = only(edge_attempts)
                    return direction_result(
                        _neutral_endpoint(exit) ? :qualified_boundary :
                            :unqualified_boundary,
                        _qualified_endpoint(exit))
                end
            else
                predictor = Tuple(collect(current.coordinates) .+
                    step .* collect(current.tangent.actual))
                attempt, point = _step_candidate(seed, current,
                    predictor, direction, step, retry, options,
                    solve_function, search_function, visited)
                push!(attempts, attempt)
                if point !== nothing
                    push!(points, point)
                    push!(visited, point.coordinates)
                    step = min(options.maximum_step, 1.25step)
                    accepted = true
                    break
                end
                # A revisit is only conclusive on its own. Co-occurring failures
                # mean the step never resolved, which is not evidence of a loop.
                if :curve_revisit in attempt.reasons && length(attempt.reasons) == 1
                    return direction_result(:revisit_unresolved, nothing)
                end
            end
            step /= 2
            if step < options.minimum_step ||
                    current.coordinates .+ step .* current.tangent.actual ==
                    current.coordinates
                return direction_result(:minimum_step_unresolved, nothing)
            end
        end
        if !accepted
            termination = :retries_exhausted_unresolved
            break
        end
    end
    return direction_result(termination, nothing)
end

function _continue_verified(verification::SeedVerification,
    options::CurveTraversalOptions;
    solve_function=Seeds._solve_bounded,
    search_function=Axis._searches)
    verification.accepted || return TraceZeroCurveResult(verification,
        options, verification.reasons, verification.error, (),
        :seed_unresolved)
    seed = verification.seed
    initial, reasons, error = _initial_point(seed, options,
        search_function)
    initial === nothing && return TraceZeroCurveResult(verification,
        options, reasons, error, (), :seed_unresolved)
    directions = Tuple(_direction(seed, initial, direction, options,
        solve_function, search_function) for direction in (-1, 1))
    # A two-boundary segment is only claimed when both orientations reached a
    # validated exit whose own neutrality evidence agrees. Validated exits that
    # are not neutrality-qualified are reported separately, and a segment with
    # even one unqualified point is downgraded so no consumer can read the
    # status as a Figure-5b Hopf claim.
    segment = all(result -> result.termination in BOUNDARY_TERMINATIONS,
        directions)
    neutral_segment = segment &&
        all(result -> result.termination == :qualified_boundary &&
            result.unqualified_points == 0, directions)
    status = if neutral_segment
        :finite_two_boundary_segment
    elseif segment
        :finite_two_unqualified_boundary_segment
    else
        :traversal_unresolved
    end
    return TraceZeroCurveResult(verification, options, (), nothing,
        directions, status)
end

"""
    traverse_verified_seed(verified, options=CurveTraversalOptions())

Continue a traversal from an already-verified seed token. Callers must obtain
`verified` from a gate that freshly requalifies the seed, its model, and its
lineage; this entry deliberately performs no qualification of its own.
"""
traverse_verified_seed(verified::VerifiedCurveSeed,
    options::CurveTraversalOptions=CurveTraversalOptions()) =
    _continue_verified(SeedVerification(true, (), verified, nothing), options)

"""
    trace_zero_curve(seed_result, qualified_index; options=CurveTraversalOptions())

Traverse both orientations of a freshly requalified PR3 trace-zero seed.
Only two independently validated fixed-edge exits yield a finite two-boundary
segment; neither that result nor any sampled point certifies a complete curve.
An exit counts as qualified only when its own neutrality evidence agrees, and
a segment containing any unqualified point is reported as
`:finite_two_unqualified_boundary_segment` rather than as a Hopf segment.
"""
function trace_zero_curve(seed_result::Seeds.CurveSeedResult,
    qualified_index; options=CurveTraversalOptions())
    options isa CurveTraversalOptions ||
        throw(ArgumentError("options must be CurveTraversalOptions"))
    verification = _verify_seed(seed_result, qualified_index)
    return _continue_verified(verification, options)
end

end # module
