using LinearAlgebra: dot, norm, svd

"""
    ContinuationOptions(; initial_step=0.03, minimum_step=1e-5,
                        maximum_step=0.1, max_steps=500, max_corrector_iters=12,
                        corrector_atol=1e-10, parameter_difference_step=1e-5,
                        state_scales=(1.0, 1.0), parameter_scale=1.0,
                        rank_atol=1e-12, rank_rtol=1e-10)

Numerical policies for equilibrium pseudo-arclength continuation. Arclength
uses coordinates `(E/state_scales[1], I/state_scales[2], p/parameter_scale)`;
these scales are numerical conventions, not model normalizations. Parameter
derivatives use finite differences with physical increment
`parameter_difference_step * max(1, abs(p))`, centered when possible and
one-sided at bounds. State derivatives are analytical. `corrector_atol`
bounds both the dimensionless balance residual and scaled arclength residual.
`max_steps` limits accepted steps separately in each direction; failed
correctors halve the step until `minimum_step`. No bifurcation certification
or branch-completeness claim follows from these tolerances.
"""
struct ContinuationOptions{T<:AbstractFloat}
    initial_step::T
    minimum_step::T
    maximum_step::T
    max_steps::Int
    max_corrector_iters::Int
    corrector_atol::T
    parameter_difference_step::T
    state_scales::NTuple{2,T}
    parameter_scale::T
    rank_atol::T
    rank_rtol::T
end

function ContinuationOptions(;
    initial_step=0.03,
    minimum_step=1e-5,
    maximum_step=0.1,
    max_steps=500,
    max_corrector_iters=12,
    corrector_atol=1e-10,
    parameter_difference_step=1e-5,
    state_scales=(1.0, 1.0),
    parameter_scale=1.0,
    rank_atol=1e-12,
    rank_rtol=1e-10,
)
    state_scales isa Tuple && length(state_scales) == 2 ||
        throw(ArgumentError("state_scales must be a tuple of two positive numbers"))
    raw = (initial_step, minimum_step, maximum_step, corrector_atol,
        parameter_difference_step, state_scales..., parameter_scale, rank_atol, rank_rtol)
    all(value -> value isa Real && !(value isa Bool) && isfinite(value) && value > 0, raw) ||
        throw(ArgumentError("continuation scales, steps, and tolerances must be finite and positive"))
    minimum_step <= initial_step <= maximum_step ||
        throw(ArgumentError("steps must satisfy minimum_step <= initial_step <= maximum_step"))
    for (name, value) in (("max_steps", max_steps), ("max_corrector_iters", max_corrector_iters))
        value isa Integer && !(value isa Bool) && value > 0 ||
            throw(ArgumentError("$name must be a positive integer"))
    end
    values = promote(float.(raw)...)
    return ContinuationOptions(values[1], values[2], values[3], Int(max_steps),
        Int(max_corrector_iters), values[4], values[5], (values[6], values[7]),
        values[8], values[9], values[10])
end

"""
    ContinuationAttempt

One attempted augmented Newton correction, including failures and retries.
`predictor` and `candidate` contain physical `[E, I, parameter]` coordinates;
`step` and `arclength_residual` use the scaled arclength convention.
`equilibrium_attempt`, when present, independently revalidates the candidate
balance residual and domain. `status` reports the corrector/acceptance outcome
and `accepted` distinguishes retained branch points from failed attempts.
"""
struct ContinuationAttempt{T<:AbstractFloat}
    direction::Int
    step::T
    predictor::Vector{T}
    candidate::Vector{T}
    iterations::Int
    arclength_residual::T
    status::Symbol
    accepted::Bool
    equilibrium_attempt::Union{Nothing,EquilibriumAttempt{T}}
end

"""
    ContinuationPoint

An admissible independently validated equilibrium and its original-time
local stability. `model` retains the autonomous model at `parameter`.
`tangent` is a unit vector in the documented scaled coordinates, oriented
along this branch's traversal; it can reverse its parameter component at a
fold. Neither the point nor its stability certifies a global attractor.
"""
struct ContinuationPoint{T<:AbstractFloat,M}
    parameter::T
    state::Vector{T}
    tangent::Vector{T}
    model::M
    equilibrium_attempt::EquilibriumAttempt{T}
    stability::LocalStabilityResult{T}
end

"""
    ContinuationCandidate

A numerical `:fold` or `:hopf` candidate bracketed by consecutive indices in
one branch's `points`. A fold candidate is a parameter-tangent sign change
or a zero tangent endpoint; a Hopf candidate is a trace sign change or a
zero trace endpoint with positive determinants and
resolved complex spectra at both endpoints. These are screening evidence,
not located, classified, or certified bifurcations or periodic solutions.
"""
struct ContinuationCandidate
    kind::Symbol
    first_point::Int
    second_point::Int
end

"""
    ContinuationBranch

Points in traversal order, every attempted correction, candidate brackets,
and a termination status. Both directions include the initial point when
its tangent is resolved. `:parameter_boundary` means the next step reaches
the requested bound within the minimum arclength step; an exact endpoint
is not guaranteed. `:step_limit` and `:minimum_step` retain partial results.
"""
struct ContinuationBranch{T<:AbstractFloat,M}
    direction::Int
    points::Vector{ContinuationPoint{T,M}}
    attempts::Vector{ContinuationAttempt{T}}
    candidates::Vector{ContinuationCandidate}
    termination::Symbol
end

"""
    EquilibriumContinuationResult

The model factory, initial local equilibrium solve, physical parameter
bounds, numerical policies, and separate `negative` and `positive` branch
traversals. Direction denotes the initial parameter direction; subsequent
folds may reverse it. `completeness` is always `CompletenessNotCertified`.
No branch switching or periodic-orbit continuation is performed.
"""
struct EquilibriumContinuationResult{F,S,T<:AbstractFloat,M,O,EO,SO}
    model_at_parameter::F
    initial_solve::S
    initial_parameter::T
    parameter_bounds::NTuple{2,T}
    options::O
    equilibrium_options::EO
    stability_options::SO
    negative::ContinuationBranch{T,M}
    positive::ContinuationBranch{T,M}
    completeness::SearchCompleteness
end

function _continuation_model(factory, parameter, ::Type{M}) where {M}
    model = factory(parameter)
    model isa M || throw(ArgumentError("model_at_parameter must return a consistent PointModelParameters type"))
    return _frozen_point_context(model, nothing).frozen_model
end

function _continuation_system(factory, z, scales, bounds, ::Type{M}) where {M}
    T = eltype(z)
    parameter = z[3] * scales[3]
    bounds[1] <= parameter <= bounds[2] || return nothing
    model = _continuation_model(factory, parameter, M)
    state = T[z[1] * scales[1], z[2] * scales[2]]
    residual = zeros(T, 2)
    jacobian = zeros(T, 2, 2)
    point_balance!(residual, state, model, zero(T))
    point_balance_jacobian!(jacobian, state, model, zero(T))
    return (; model, state, parameter, residual, jacobian)
end

function _continuation_augmented_jacobian(factory, system, scales, bounds, options)
    T = eltype(system.state)
    parameter = system.parameter
    increment = T(options.parameter_difference_step) * max(one(T), abs(parameter))
    # Use a symmetric difference throughout the interior; only a true bound
    # uses a one-sided difference. Every factory evaluation stays in bounds.
    available = min(parameter - bounds[1], bounds[2] - parameter)
    if available > zero(T)
        increment = min(increment, available)
        lower, upper = parameter - increment, parameter + increment
    else
        lower = max(bounds[1], parameter - increment)
        upper = min(bounds[2], parameter + increment)
    end
    upper > lower || return nothing
    minus_model = _continuation_model(factory, lower, typeof(system.model))
    plus_model = _continuation_model(factory, upper, typeof(system.model))
    minus_residual, plus_residual = zeros(T, 2), zeros(T, 2)
    point_balance!(minus_residual, system.state, minus_model, zero(T))
    point_balance!(plus_residual, system.state, plus_model, zero(T))
    parameter_derivative = (plus_residual .- minus_residual) ./ (upper - lower)
    augmented = hcat(system.jacobian, parameter_derivative)
    for column in axes(augmented, 2)
        augmented[:, column] .*= scales[column]
    end
    return all(isfinite, augmented) ? augmented : nothing
end

function _continuation_tangent(augmented, options, previous, direction)
    augmented === nothing && return nothing
    decomposition = svd(augmented; full=true)
    threshold = options.rank_atol + options.rank_rtol * maximum(decomposition.S)
    minimum(decomposition.S) > threshold || return nothing
    tangent = decomposition.V[:, end]
    if previous === nothing
        # Starting exactly at a parameter fold has no resolved +/- parameter
        # orientation. Retain the seed solve and report this as unresolved.
        abs(tangent[3]) > options.rank_atol || return nothing
        tangent[3] * direction < 0 && (tangent .*= -1)
    elseif dot(tangent, previous) < 0
        tangent .*= -1
    end
    return tangent
end

function _continuation_correct(factory, predictor, tangent, scales, bounds, options, ::Type{M}) where {M}
    z = copy(predictor)
    T = eltype(z)
    for iteration in 0:options.max_corrector_iters
        system = _continuation_system(factory, z, scales, bounds, M)
        arc = dot(z .- predictor, tangent)
        system === nothing && return (z, iteration, arc, :parameter_out_of_bounds)
        all(isfinite, system.residual) || return (z, iteration, arc, :nonfinite_residual)
        if max(maximum(abs, system.residual), abs(arc)) <= options.corrector_atol
            return (z, iteration, arc, :corrector_converged)
        end
        iteration == options.max_corrector_iters &&
            return (z, iteration, arc, :corrector_iteration_limit)
        augmented = _continuation_augmented_jacobian(factory, system, scales, bounds, options)
        augmented === nothing && return (z, iteration, arc, :parameter_derivative_unresolved)
        bordered = vcat(augmented, transpose(tangent))
        singular_values = svdvals(bordered)
        minimum(singular_values) > options.rank_atol + options.rank_rtol * maximum(singular_values) ||
            return (z, iteration, arc, :corrector_rank_unresolved)
        correction = bordered \ vcat(system.residual, arc)
        all(isfinite, correction) || return (z, iteration, arc, :nonfinite_correction)
        z .-= correction
    end
    error("unreachable corrector state")
end

function _continuation_candidates!(candidates, points)
    length(points) >= 2 || return
    left, right = points[end - 1], points[end]
    if left.tangent[3] * right.tangent[3] <= 0 &&
       !(iszero(left.tangent[3]) && iszero(right.tangent[3]))
        push!(candidates, ContinuationCandidate(:fold, length(points) - 1, length(points)))
    end
    first_stability, second_stability = left.stability, right.stability
    if first_stability.geometry == ComplexConjugateSpectrum &&
       second_stability.geometry == ComplexConjugateSpectrum &&
       first_stability.determinant > 0 && second_stability.determinant > 0 &&
       first_stability.trace * second_stability.trace <= 0 &&
       !(iszero(first_stability.trace) && iszero(second_stability.trace))
        push!(candidates, ContinuationCandidate(:hopf, length(points) - 1, length(points)))
    end
    return nothing
end

function _continue_equilibrium_branch(factory, initial_solve, initial_parameter, bounds,
    options, equilibrium_options, stability_options, direction)
    T = typeof(initial_parameter)
    M = typeof(initial_solve.frozen_model)
    points = ContinuationPoint{T,M}[]
    attempts = ContinuationAttempt{T}[]
    candidates = ContinuationCandidate[]
    finish(status) = ContinuationBranch(direction, points, attempts, candidates, status)
    initial_solve.attempt.validation == AdmissibleCandidate ||
        return finish(:initial_equilibrium_unresolved)
    scales = T[options.state_scales..., options.parameter_scale]
    z = vcat(initial_solve.attempt.candidate, initial_parameter) ./ scales
    system = _continuation_system(factory, z, scales, bounds, M)
    augmented = _continuation_augmented_jacobian(factory, system, scales, bounds, options)
    tangent = _continuation_tangent(augmented, options, nothing, direction)
    tangent === nothing && return finish(:initial_tangent_unresolved)
    push!(points, ContinuationPoint(initial_parameter, copy(initial_solve.attempt.candidate),
        tangent, system.model, initial_solve.attempt, initial_solve.stability))
    step = T(options.initial_step)

    for _ in 1:options.max_steps
        while true
            # Stop short of a bound with explicit status. Shrinking here does
            # not replace pseudo-arclength correction or assume monotonic p.
            if tangent[3] != 0
                bound = tangent[3] > 0 ? bounds[2] : bounds[1]
                distance = (bound / scales[3] - z[3]) / tangent[3]
                if distance <= options.minimum_step
                    return finish(:parameter_boundary)
                end
                step = min(step, T(0.95) * distance)
            end
            step >= options.minimum_step || return finish(:minimum_step)
            predictor = z .+ step .* tangent
            candidate, iterations, arc, status = _continuation_correct(
                factory, predictor, tangent, scales, bounds, options, M)
            attempt = nothing
            next_tangent = nothing
            next_system = _continuation_system(factory, candidate, scales, bounds, M)
            if next_system !== nothing && all(isfinite, candidate)
                attempt = _validate_candidate(
                    Vector{T}((predictor .* scales)[1:2]), copy(next_system.state), status,
                    status == :corrector_converged, copy(next_system.residual),
                    next_system.model, equilibrium_options)
            end
            if status == :corrector_converged
                if attempt === nothing || attempt.validation != AdmissibleCandidate
                    status = :equilibrium_validation_failed
                elseif norm(candidate .- predictor) > step
                    status = :excessive_corrector_distance
                else
                    augmented = _continuation_augmented_jacobian(factory, next_system, scales, bounds, options)
                    next_tangent = _continuation_tangent(augmented, options, tangent, direction)
                    if next_tangent === nothing
                        status = :tangent_rank_unresolved
                    elseif dot(next_tangent, tangent) < T(0.5)
                        status = :excessive_tangent_rotation
                    end
                end
            end
            accepted = status == :corrector_converged
            push!(attempts, ContinuationAttempt(direction, step, predictor .* scales,
                candidate .* scales, iterations, arc, status, accepted, attempt))
            if accepted
                context = _frozen_point_context(next_system.model, nothing)
                stability = _stability_for_context(context, next_system.state, stability_options)
                push!(points, ContinuationPoint(next_system.parameter, copy(next_system.state),
                    next_tangent, next_system.model, attempt, stability))
                _continuation_candidates!(candidates, points)
                z, tangent = candidate, next_tangent
                step = min(T(options.maximum_step), step * T(1.25))
                break
            end
            step /= T(2)
        end
    end
    return finish(:step_limit)
end

"""
    continue_equilibria(model_at_parameter, initial_state, initial_parameter;
                       parameter_bounds, options=ContinuationOptions(),
                       equilibrium_options=EquilibriumOptions(),
                       stability_options=StabilityOptions())

Continue one equilibrium branch in both initial parameter directions using
an analytical state Jacobian, explicit finite-difference parameter column,
and an augmented Newton pseudo-arclength corrector. This can traverse folds
and unstable equilibrium segments. `model_at_parameter(p)` must return an
autonomous `PointModelParameters` of consistent type throughout the finite
ordered bounds. All factory evaluations stay within those bounds. Factory
errors propagate as input errors; nonconvergence retains numerical attempts.

The initial state is locally solved first, and every corrected state receives
independent balance/domain validation and original-time local stability.
Failed initial solves, unresolved tangents, step limits, and minimum-step
failures return partial evidence with explicit branch termination statuses.
Fold/Hopf sign-change brackets remain numerical candidates. No completeness,
branch switching, periodic solutions, or biological regime labels are claimed.
Only Float32 and Float64 equilibrium analysis is supported.
"""
function continue_equilibria(factory, initial_state, initial_parameter;
    parameter_bounds,
    options=ContinuationOptions(),
    equilibrium_options=EquilibriumOptions(),
    stability_options=StabilityOptions(),
)
    options isa ContinuationOptions || throw(ArgumentError("options must be ContinuationOptions"))
    equilibrium_options isa EquilibriumOptions || throw(ArgumentError("equilibrium_options must be EquilibriumOptions"))
    stability_options isa StabilityOptions || throw(ArgumentError("stability_options must be StabilityOptions"))
    initial_parameter isa Real && isfinite(initial_parameter) ||
        throw(ArgumentError("initial_parameter must be finite and real"))
    parameter_bounds isa Tuple && length(parameter_bounds) == 2 ||
        throw(ArgumentError("parameter_bounds must be a (lower, upper) tuple"))
    all(value -> value isa Real && isfinite(value), parameter_bounds) ||
        throw(ArgumentError("parameter bounds must be finite and real"))
    parameter_bounds[1] < parameter_bounds[2] || throw(ArgumentError("parameter bounds must be strictly ordered"))
    parameter_bounds[1] <= initial_parameter <= parameter_bounds[2] ||
        throw(ArgumentError("initial_parameter must lie within parameter bounds"))
    _validate_raw_seed(initial_state)
    model = factory(float(initial_parameter))
    model isa PointModelParameters || throw(ArgumentError("model_at_parameter must return PointModelParameters"))
    context = _frozen_point_context(model, nothing)
    _, model_type = _normalize_one_seed(initial_state, context)
    T = _require_supported_analysis_type(promote_type(model_type, typeof(float(initial_parameter)),
        map(value -> typeof(float(value)), parameter_bounds)...))
    bounds = Tuple(T.(parameter_bounds))
    parameter = T(initial_parameter)
    normalized_model = factory(parameter)
    normalized_model isa PointModelParameters || throw(ArgumentError("model_at_parameter must return PointModelParameters"))
    initial_solve = solve_equilibrium(normalized_model, T.(collect(initial_state));
        options=equilibrium_options, stability_options=stability_options)
    negative = _continue_equilibrium_branch(factory, initial_solve, parameter, bounds,
        options, equilibrium_options, stability_options, -1)
    positive = _continue_equilibrium_branch(factory, initial_solve, parameter, bounds,
        options, equilibrium_options, stability_options, 1)
    return EquilibriumContinuationResult(factory, initial_solve, parameter, bounds,
        options, equilibrium_options, stability_options, negative, positive, CompletenessNotCertified)
end
