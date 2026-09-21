using LinearAlgebra: svdvals
using SciMLBase: NonlinearFunction, NonlinearProblem, solve, successful_retcode
using SimpleNonlinearSolve: SimpleTrustRegion

"""Validation states retained for raw equilibrium candidates."""
@enum CandidateValidation begin
    AdmissibleCandidate
    BoundaryAmbiguousCandidate
    RejectedCandidate
end

"""Search-completeness status. Numerical multistart discovery is not a certificate."""
@enum SearchCompleteness begin
    CompletenessNotCertified
end

"""
    EquilibriumOptions(; kwargs...)

Numerical policies for local and multistart equilibrium solves. Solver and
independent balance-residual tolerances are dimensionless; `domain_atol` and
`dedup_atol` use activity-coordinate units. `singular_atol` and
`singular_rtol` flag a poorly conditioned dimensionless balance Jacobian.
`maxiters` is the per-seed nonlinear iteration limit. Defaults and scaling
rules are documented in `docs/model.md`.
"""
struct EquilibriumOptions{T<:AbstractFloat}
    solver_abstol::T
    solver_reltol::T
    residual_atol::T
    domain_atol::T
    dedup_atol::T
    singular_atol::T
    singular_rtol::T
    maxiters::Int
end

function EquilibriumOptions(;
    solver_abstol=1.0e-12,
    solver_reltol=1.0e-10,
    residual_atol=1.0e-9,
    domain_atol=1.0e-8,
    dedup_atol=1.0e-7,
    singular_atol=1.0e-10,
    singular_rtol=1.0e-8,
    maxiters=100,
)
    raw_tolerances = (
        solver_abstol,
        solver_reltol,
        residual_atol,
        domain_atol,
        dedup_atol,
        singular_atol,
        singular_rtol,
    )
    all(value -> value isa Real, raw_tolerances) ||
        throw(ArgumentError("equilibrium tolerances must be real"))
    all(isfinite, raw_tolerances) ||
        throw(ArgumentError("equilibrium tolerances must be finite"))
    all(value -> value >= zero(value), raw_tolerances) ||
        throw(ArgumentError("equilibrium tolerances must be nonnegative"))
    maxiters isa Integer && maxiters > 0 ||
        throw(ArgumentError("maxiters must be a positive integer"))
    tolerances = promote(float.(raw_tolerances)...)
    return EquilibriumOptions(tolerances..., Int(maxiters))
end

"""
    EquilibriumAttempt

Diagnostics from one nonlinear solve. `seed` is a normalized copy and
`candidate` is the raw solver state. `solver_status`, `solver_success`, and
`solver_residual` retain the solver outcome; `balance_residual`, its infinity
`residual_norm`, and `balance_jacobian` are independently recomputed.
`near_singular` applies the documented singular-value test. Solver success is
reported separately from `validation`, and `reasons` retains every validation
issue without clipping the candidate.
"""
struct EquilibriumAttempt{T<:AbstractFloat}
    seed::Vector{T}
    candidate::Vector{T}
    solver_status::Symbol
    solver_success::Bool
    solver_residual::Vector{T}
    balance_residual::Vector{T}
    residual_norm::T
    balance_jacobian::Matrix{T}
    near_singular::Bool
    validation::CandidateValidation
    reasons::Vector{Symbol}
end

"""
    EquilibriumSolveResult

Context and diagnostics for one local equilibrium solve. `model` is the
source model; `frozen_model` and `frozen_drive` are the autonomous system and
its `(E, I)` input. `source_time` records an explicit protocol snapshot when
supplied. `options` and `stability_options` retain the numerical policy.
`attempt` is always present; `stability` is present only for an admissible,
independently validated candidate.
"""
struct EquilibriumSolveResult{M,F,D,S,T<:AbstractFloat,O,SO}
    model::M
    frozen_model::F
    frozen_drive::D
    source_time::S
    options::O
    stability_options::SO
    attempt::EquilibriumAttempt{T}
    stability::Union{Nothing,LocalStabilityResult{T}}
end

"""
    Equilibrium

A deterministically selected, actual validated solver candidate. `state`,
`balance_residual`, `balance_jacobian`, and `near_singular` are copied from
`representative_attempt`; the state is never an average. `member_attempts`
records all duplicate attempts represented by that candidate, and `stability`
contains its original-time Jacobian diagnostics.
"""
struct Equilibrium{T<:AbstractFloat}
    state::Vector{T}
    balance_residual::Vector{T}
    balance_jacobian::Matrix{T}
    near_singular::Bool
    representative_attempt::Int
    member_attempts::Vector{Int}
    stability::LocalStabilityResult{T}
end

"""
    EquilibriumSearchResult

Full diagnostics from deterministic multistart discovery. `model`,
`frozen_model`, `frozen_drive`, and `source_time` identify the source and
autonomous systems; `options` and `stability_options` retain the numerical
policy. `attempts` includes solver failures and rejected candidates, while
`equilibria` contains admissible representatives sorted by `(E, I)`.
`unresolved_nearby` records tolerance-chain components whose diameter prevents
a single collapse; their candidate outputs are partitioned into
complete-linkage subgroups so repeated attempts are still deduplicated.
`completeness` is always `CompletenessNotCertified`, including when no
equilibrium was found.
"""
struct EquilibriumSearchResult{M,F,D,S,T<:AbstractFloat,O,SO}
    model::M
    frozen_model::F
    frozen_drive::D
    source_time::S
    options::O
    stability_options::SO
    attempts::Vector{EquilibriumAttempt{T}}
    equilibria::Vector{Equilibrium{T}}
    unresolved_nearby::Vector{Vector{Int}}
    completeness::SearchCompleteness
end

struct _FrozenPointContext{M,F,D,S}
    model::M
    frozen_model::F
    frozen_drive::D
    source_time::S
end

_response_numeric_values(response::LogisticResponse) =
    (response.slope, response.threshold)
_response_numeric_values(response::FailureOfInhibitionResponse) =
    (response.slope, response.onset_threshold, response.failure_threshold)

function _model_numeric_values(model::PointModelParameters)
    coupling = model.coupling
    return (
        model.excitatory.timescale,
        _response_numeric_values(model.excitatory.response)...,
        model.inhibitory.timescale,
        _response_numeric_values(model.inhibitory.response)...,
        coupling.e_to_e,
        coupling.i_to_e,
        coupling.e_to_i,
        coupling.i_to_i,
    )
end

function _model_float_type(model::PointModelParameters)
    values = float.(_model_numeric_values(model))
    return promote_type(map(typeof, values)...)
end

function _validate_snapshot_time(snapshot_time)
    snapshot_time === nothing && return nothing
    snapshot_time isa Real || throw(ArgumentError("snapshot_time must be real"))
    isfinite(snapshot_time) || throw(ArgumentError("snapshot_time must be finite"))
    return snapshot_time
end

function _frozen_point_context(model::PointModelParameters, snapshot_time)
    _validate_snapshot_time(snapshot_time)
    drive = model.drive
    has_pulses = drive isa PiecewiseConstantDrive && !isempty(drive.pulses)
    has_pulses && snapshot_time === nothing && throw(
        ArgumentError("pulsed drives require an explicit finite snapshot_time"),
    )

    if drive isa NoDrive
        T = _model_float_type(model)
        frozen_drive = (zero(T), zero(T))
        return _FrozenPointContext(model, model, frozen_drive, snapshot_time)
    elseif isempty(drive.pulses)
        return _FrozenPointContext(model, model, drive.baseline, snapshot_time)
    end

    frozen_drive = drive_value(drive, snapshot_time)
    autonomous_drive = PiecewiseConstantDrive(
        baseline=frozen_drive,
        pulses=(),
        interpretation=drive.interpretation,
    )
    frozen_model = PointModelParameters(
        excitatory=model.excitatory,
        inhibitory=model.inhibitory,
        coupling=model.coupling,
        drive=autonomous_drive,
    )
    return _FrozenPointContext(model, frozen_model, frozen_drive, snapshot_time)
end

function _equilibrium_upper_bounds(model::PointModelParameters, ::Type{T}) where {T}
    excitatory_upper = one(T) / convert(T, 2)
    response_parameters = model.inhibitory.response
    inhibitory_upper = if response_parameters isa LogisticResponse
        one(T) / convert(T, 2)
    else
        slope = convert(T, response_parameters.slope)
        separation = convert(
            T,
            response_parameters.failure_threshold - response_parameters.onset_threshold,
        )
        maximum_response = tanh(slope * separation / convert(T, 4))
        maximum_response / (one(T) + maximum_response)
    end
    return (excitatory_upper, inhibitory_upper)
end

function _validate_raw_seed(seed)
    seed isa AbstractVector ||
        throw(ArgumentError("each seed must be a vector ordered [E, I]"))
    length(seed) == 2 || throw(ArgumentError("each seed must contain exactly E and I"))
    all(value -> value isa Real, seed) ||
        throw(ArgumentError("seed coordinates must be real"))
    all(isfinite, seed) || throw(ArgumentError("seed coordinates must be finite"))
    return seed
end

function _analysis_type(context::_FrozenPointContext, floating_seeds)
    values = Any[float.(_model_numeric_values(context.frozen_model))...]
    append!(values, float.(context.frozen_drive))
    for seed in floating_seeds
        append!(values, seed)
    end
    T = promote_type(map(typeof, values)...)
    return _require_supported_analysis_type(T)
end

function _normalize_seed_collection(seeds, context::_FrozenPointContext)
    applicable(iterate, seeds) ||
        throw(ArgumentError("seeds must be an iterable of [E, I] vectors"))
    raw_seeds = collect(seeds)
    floating_seeds = Vector[]
    for seed in raw_seeds
        _validate_raw_seed(seed)
        push!(floating_seeds, collect(float.(seed)))
    end
    T = _analysis_type(context, floating_seeds)
    normalized = Vector{Vector{T}}(undef, length(floating_seeds))
    for index in eachindex(floating_seeds)
        normalized[index] = Vector{T}(floating_seeds[index])
    end
    return normalized, T
end

function _normalize_one_seed(seed, context::_FrozenPointContext)
    seeds, T = _normalize_seed_collection((seed,), context)
    return only(seeds), T
end

"""
    default_equilibrium_seeds(model)

Return a deterministic 5-by-5 tensor grid over the model's sharper
equilibrium rectangle. The grid includes edges and interior points and is
default search coverage only; it does not certify completeness.
"""
function default_equilibrium_seeds(model::PointModelParameters)
    context = _frozen_point_context(model, nothing)
    T = _require_supported_analysis_type(_model_float_type(context.frozen_model))
    upper_e, upper_i = _equilibrium_upper_bounds(context.frozen_model, T)
    e_values = range(zero(T), upper_e; length=5)
    i_values = range(zero(T), upper_i; length=5)
    return [T[e, i] for e in e_values for i in i_values]
end

function _balance_solver!(residual, state, model)
    return point_balance!(residual, state, model, zero(eltype(state)))
end

function _balance_solver_jacobian!(jacobian, state, model)
    return point_balance_jacobian!(jacobian, state, model, zero(eltype(state)))
end

function _range_status(value, lower, upper, tolerance)
    value < lower - tolerance || value > upper + tolerance ? :outside :
    value < lower || value > upper ? :ambiguous : :inside
end

function _validate_candidate(
    seed::Vector{T},
    candidate::Vector{T},
    solver_status,
    solver_success,
    solver_residual::Vector{T},
    model::PointModelParameters,
    options::EquilibriumOptions,
) where {T<:AbstractFloat}
    nan_value = convert(T, NaN)
    balance_residual = fill(nan_value, 2)
    balance_jacobian = fill(nan_value, 2, 2)
    reasons = Symbol[]

    if !all(isfinite, candidate)
        push!(reasons, :nonfinite_candidate)
        return EquilibriumAttempt(
            seed,
            candidate,
            solver_status,
            solver_success,
            solver_residual,
            balance_residual,
            nan_value,
            balance_jacobian,
            false,
            RejectedCandidate,
            reasons,
        )
    end

    point_balance!(balance_residual, candidate, model, zero(T))
    if !all(isfinite, balance_residual)
        push!(reasons, :nonfinite_balance_residual)
        return EquilibriumAttempt(
            seed,
            candidate,
            solver_status,
            solver_success,
            solver_residual,
            balance_residual,
            nan_value,
            balance_jacobian,
            false,
            RejectedCandidate,
            reasons,
        )
    end

    residual_norm = maximum(abs, balance_residual)
    residual_tolerance = convert(T, options.residual_atol)
    residual_norm <= residual_tolerance || push!(reasons, :large_balance_residual)

    point_balance_jacobian!(balance_jacobian, candidate, model, zero(T))
    near_singular = false
    if all(isfinite, balance_jacobian)
        singular_values = svdvals(balance_jacobian)
        singular_threshold = convert(T, options.singular_atol) +
                             convert(T, options.singular_rtol) * maximum(singular_values)
        near_singular = minimum(singular_values) <= singular_threshold
    else
        push!(reasons, :nonfinite_balance_jacobian)
    end

    domain_tolerance = convert(T, options.domain_atol)
    upper_e, upper_i = _equilibrium_upper_bounds(model, T)
    physical_statuses = (
        _range_status(candidate[1], zero(T), one(T), domain_tolerance),
        _range_status(candidate[2], zero(T), one(T), domain_tolerance),
    )
    equilibrium_statuses = (
        _range_status(candidate[1], zero(T), upper_e, domain_tolerance),
        _range_status(candidate[2], zero(T), upper_i, domain_tolerance),
    )
    :outside in physical_statuses && push!(reasons, :outside_physical_domain)
    :outside in equilibrium_statuses && push!(reasons, :outside_equilibrium_bounds)

    hard_rejection = any(
        reason -> reason in (
            :large_balance_residual,
            :nonfinite_balance_jacobian,
            :outside_physical_domain,
            :outside_equilibrium_bounds,
        ),
        reasons,
    )
    ambiguous = :ambiguous in physical_statuses || :ambiguous in equilibrium_statuses
    if ambiguous
        :ambiguous in physical_statuses && push!(reasons, :physical_boundary_ambiguous)
        :ambiguous in equilibrium_statuses &&
            push!(reasons, :equilibrium_boundary_ambiguous)
    end
    validation = hard_rejection ? RejectedCandidate :
                 ambiguous ? BoundaryAmbiguousCandidate : AdmissibleCandidate

    return EquilibriumAttempt(
        seed,
        candidate,
        solver_status,
        solver_success,
        solver_residual,
        balance_residual,
        residual_norm,
        balance_jacobian,
        near_singular,
        validation,
        reasons,
    )
end

function _solve_equilibrium_attempt(
    model::PointModelParameters,
    seed::Vector{T},
    options::EquilibriumOptions,
) where {T<:AbstractFloat}
    nonlinear_function = NonlinearFunction(
        _balance_solver!;
        jac=_balance_solver_jacobian!,
    )
    problem = NonlinearProblem(nonlinear_function, copy(seed), model)
    solution = solve(
        problem,
        SimpleTrustRegion();
        abstol=convert(T, options.solver_abstol),
        reltol=convert(T, options.solver_reltol),
        maxiters=options.maxiters,
    )
    candidate = Vector{T}(solution.u)
    solver_residual = Vector{T}(solution.resid)
    status = Symbol(string(solution.retcode))
    return _validate_candidate(
        seed,
        candidate,
        status,
        successful_retcode(solution.retcode),
        solver_residual,
        model,
        options,
    )
end

function _stability_for_context(
    context::_FrozenPointContext,
    state::Vector{T},
    options::StabilityOptions,
) where {T<:AbstractFloat}
    jacobian = Matrix{T}(undef, 2, 2)
    point_jacobian!(jacobian, state, context.frozen_model, zero(T))
    return classify_local_stability(jacobian; options=options)
end

"""
    solve_equilibrium(model, seed; snapshot_time=nothing,
                      options=EquilibriumOptions(),
                      stability_options=StabilityOptions())

Run one explicit `SimpleTrustRegion` solve of the dimensionless balance
equations with their analytical Jacobian. Trial evaluations may leave the
physical domain. Acceptance is based on an independent balance-residual and
domain validation, not on the solver return code alone.
"""
function solve_equilibrium(
    model::PointModelParameters,
    seed;
    snapshot_time=nothing,
    options=EquilibriumOptions(),
    stability_options=StabilityOptions(),
)
    options isa EquilibriumOptions ||
        throw(ArgumentError("options must be EquilibriumOptions"))
    stability_options isa StabilityOptions ||
        throw(ArgumentError("stability_options must be StabilityOptions"))
    context = _frozen_point_context(model, snapshot_time)
    normalized_seed, T = _normalize_one_seed(seed, context)
    attempt = _solve_equilibrium_attempt(context.frozen_model, normalized_seed, options)
    stability = attempt.validation == AdmissibleCandidate ?
                _stability_for_context(context, attempt.candidate, stability_options) : nothing
    frozen_drive = Tuple(T.(context.frozen_drive))
    return EquilibriumSolveResult(
        context.model,
        context.frozen_model,
        frozen_drive,
        context.source_time,
        options,
        stability_options,
        attempt,
        stability,
    )
end

_candidate_distance(first::EquilibriumAttempt, second::EquilibriumAttempt) =
    maximum(abs.(first.candidate .- second.candidate))

function _connected_candidate_components(
    attempts::Vector{EquilibriumAttempt{T}},
    indices::Vector{Int},
    tolerance::T,
) where {T<:AbstractFloat}
    remaining = Set(indices)
    components = Vector{Vector{Int}}()
    while !isempty(remaining)
        start = minimum(remaining)
        delete!(remaining, start)
        component = Int[start]
        frontier = Int[start]
        while !isempty(frontier)
            current = popfirst!(frontier)
            neighbors = sort!(
                Int[index for index in remaining if
                    _candidate_distance(attempts[current], attempts[index]) <= tolerance],
            )
            for neighbor in neighbors
                delete!(remaining, neighbor)
                push!(component, neighbor)
                push!(frontier, neighbor)
            end
        end
        sort!(component)
        push!(components, component)
    end
    return components
end

function _component_diameter(attempts, component)
    diameter = zero(eltype(first(attempts).candidate))
    for first_index in component, second_index in component
        diameter = max(
            diameter,
            _candidate_distance(attempts[first_index], attempts[second_index]),
        )
    end
    return diameter
end

function _representative_index(attempts, component)
    return argmin(component) do index
        _candidate_order_key(attempts[index])
    end
end

_candidate_order_key(attempt) = (
    attempt.residual_norm,
    attempt.candidate[1],
    attempt.candidate[2],
    attempt.seed[1],
    attempt.seed[2],
)

function _partition_complete_linkage(attempts, component, tolerance)
    ordered = sort(component; by=index -> _candidate_order_key(attempts[index]))
    groups = Vector{Vector{Int}}()
    for index in ordered
        group_index = findfirst(groups) do group
            all(
                member -> _candidate_distance(attempts[index], attempts[member]) <= tolerance,
                group,
            )
        end
        if isnothing(group_index)
            push!(groups, Int[index])
        else
            push!(groups[group_index], index)
        end
    end
    foreach(sort!, groups)
    sort!(groups; by=first)
    return groups
end

function _equilibrium_from_attempt(context, attempts, representative, members, options)
    attempt = attempts[representative]
    stability = _stability_for_context(context, attempt.candidate, options)
    return Equilibrium(
        copy(attempt.candidate),
        copy(attempt.balance_residual),
        copy(attempt.balance_jacobian),
        attempt.near_singular,
        representative,
        copy(members),
        stability,
    )
end

"""
    find_equilibria(model; seeds=nothing, snapshot_time=nothing,
                    options=EquilibriumOptions(),
                    stability_options=StabilityOptions())

Run deterministic multistart equilibrium discovery. When `seeds` is `nothing`,
use the documented 5-by-5 sharper-rectangle grid. Validated roots are
deduplicated by coordinate tolerance using actual candidates, sorted by
`(E, I)`, and locally classified. All attempts and failures are retained.
Search completeness is never certified, and an empty result does not show
that no equilibrium exists.
"""
function find_equilibria(
    model::PointModelParameters;
    seeds=nothing,
    snapshot_time=nothing,
    options=EquilibriumOptions(),
    stability_options=StabilityOptions(),
)
    options isa EquilibriumOptions ||
        throw(ArgumentError("options must be EquilibriumOptions"))
    stability_options isa StabilityOptions ||
        throw(ArgumentError("stability_options must be StabilityOptions"))
    context = _frozen_point_context(model, snapshot_time)
    supplied_seeds = seeds === nothing ? default_equilibrium_seeds(context.frozen_model) : seeds
    normalized_seeds, T = _normalize_seed_collection(supplied_seeds, context)
    attempts = EquilibriumAttempt{T}[
        _solve_equilibrium_attempt(context.frozen_model, seed, options) for
        seed in normalized_seeds
    ]

    admissible = Int[
        index for index in eachindex(attempts) if
        attempts[index].validation == AdmissibleCandidate
    ]
    components = _connected_candidate_components(
        attempts,
        admissible,
        convert(T, options.dedup_atol),
    )
    equilibria = Equilibrium{T}[]
    unresolved_nearby = Vector{Vector{Int}}()
    for component in components
        if _component_diameter(attempts, component) <= convert(T, options.dedup_atol)
            representative = _representative_index(attempts, component)
            push!(
                equilibria,
                _equilibrium_from_attempt(
                    context,
                    attempts,
                    representative,
                    component,
                    stability_options,
                ),
            )
        else
            push!(unresolved_nearby, copy(component))
            subgroups = _partition_complete_linkage(
                attempts,
                component,
                convert(T, options.dedup_atol),
            )
            for subgroup in subgroups
                representative = _representative_index(attempts, subgroup)
                push!(
                    equilibria,
                    _equilibrium_from_attempt(
                        context,
                        attempts,
                        representative,
                        subgroup,
                        stability_options,
                    ),
                )
            end
        end
    end
    sort!(equilibria; by=equilibrium -> (equilibrium.state[1], equilibrium.state[2]))
    sort!(unresolved_nearby; by=first)

    frozen_drive = Tuple(T.(context.frozen_drive))
    return EquilibriumSearchResult(
        context.model,
        context.frozen_model,
        frozen_drive,
        context.source_time,
        options,
        stability_options,
        attempts,
        equilibria,
        unresolved_nearby,
        CompletenessNotCertified,
    )
end
