using LinearAlgebra: det, dot, norm, svd, svdvals

"""
    PeriodicContinuationOptions(; kwargs...)

Numerical policies for pseudo-arclength continuation of planar periodic
orbits. The augmented physical coordinates are
`[state[1], state[2], log(period), parameter]`; arclength uses those
coordinates divided by the corresponding state, log-period, and parameter
scales. Parameter derivatives are finite differences whose evaluations stay
inside `parameter_bounds`. Failed correctors are retained and halve the step.

These policies do not certify a complete branch or classify its endpoint as a
Hopf, fold of cycles, homoclinic orbit, or any other bifurcation.
"""
struct PeriodicContinuationOptions
    initial_step::Float64
    minimum_step::Float64
    maximum_step::Float64
    max_steps::Int
    max_corrector_iters::Int
    corrector_atol::Float64
    parameter_difference_step::Float64
    state_scales::NTuple{2,Float64}
    log_period_scale::Float64
    parameter_scale::Float64
    rank_atol::Float64
    rank_rtol::Float64
    tangent_alignment_min::Float64
end

function PeriodicContinuationOptions(;
    initial_step=0.02,
    minimum_step=1e-5,
    maximum_step=0.1,
    max_steps=250,
    max_corrector_iters=12,
    corrector_atol=1e-8,
    parameter_difference_step=1e-5,
    state_scales=(1.0, 1.0),
    log_period_scale=1.0,
    parameter_scale=1.0,
    rank_atol=1e-12,
    rank_rtol=1e-10,
    tangent_alignment_min=0.25,
)
    state_scales isa Tuple && length(state_scales) == 2 ||
        throw(ArgumentError("state_scales must be a tuple of two positive numbers"))
    positive = (initial_step, minimum_step, maximum_step, corrector_atol,
        parameter_difference_step, state_scales..., log_period_scale,
        parameter_scale, rank_atol, rank_rtol)
    converted = map(value -> _periodic_float64(value,
        "periodic-continuation scale, step, or tolerance"), positive)
    all(>(0), converted) || throw(ArgumentError(
        "periodic-continuation scales, steps, and tolerances must be positive"))
    converted[2] <= converted[1] <= converted[3] ||
        throw(ArgumentError("steps must satisfy minimum_step <= initial_step <= maximum_step"))
    tangent_alignment = _periodic_float64(tangent_alignment_min,
        "tangent_alignment_min")
    0 <= tangent_alignment < 1 ||
        throw(ArgumentError("tangent_alignment_min must be finite and lie in [0, 1)"))
    for (name, value) in (("max_steps", max_steps),
                          ("max_corrector_iters", max_corrector_iters))
        value isa Integer && !(value isa Bool) && value > 0 ||
            throw(ArgumentError("$name must be a positive integer"))
    end
    return PeriodicContinuationOptions(
        converted[1], converted[2], converted[3], Int(max_steps),
        Int(max_corrector_iters), converted[4], converted[5],
        (converted[6], converted[7]), converted[8], converted[9], converted[10],
        converted[11], tangent_alignment,
    )
end

"""
    PeriodicContinuationAttempt

One retained pseudo-arclength correction, including failed retries.
`predictor` and `candidate` use physical
`[state[1], state[2], log(period), parameter]` coordinates. Corrector residuals
precede independent shooting. Post-shoot residuals are recomputed from the
stored orbit against the original phase and arclength equations; accepted
attempts require both post-shoot residuals to satisfy `corrector_atol`.
"""
struct PeriodicContinuationAttempt
    direction::Int
    step::Float64
    predictor::Vector{Float64}
    candidate::Vector{Float64}
    iterations::Int
    corrector_arclength_residual::Float64
    corrector_constraint_residual::Float64
    post_shoot_arclength_residual::Float64
    post_shoot_constraint_residual::Float64
    status::Symbol
    accepted::Bool
    orbit::Union{Nothing,PeriodicOrbitResult}
end

"""
    PeriodicContinuationPoint

One independently re-shot orbit on a periodic branch. `tangent` is a unit
vector in scaled `[state, log(period), parameter]` coordinates. The stored
observables are coordinate half-ranges, signed phase-plane area, a finite
primitive-period alias screen, and a planar divergence/Floquet cross-check.
"""
struct PeriodicContinuationPoint
    parameter::Float64
    state::Vector{Float64}
    period::Float64
    tangent::Vector{Float64}
    orbit::PeriodicOrbitResult
    component_half_ranges::Vector{Float64}
    signed_area::Float64
    primitive_period_check::NamedTuple
    divergence_check::NamedTuple
end

"""
    PeriodicContinuationBranch

Accepted points, all corrector attempts, indices at which the oriented
parameter tangent reverses, and an explicit termination status. A reversal is
geometric branch evidence only; it is not a classified fold of cycles.
"""
struct PeriodicContinuationBranch
    direction::Int
    points::Vector{PeriodicContinuationPoint}
    attempts::Vector{PeriodicContinuationAttempt}
    parameter_reversals::Vector{Int}
    termination::Symbol
end

"""
    PeriodicContinuationResult

The initial shooting result and two pseudo-arclength traversals. `negative`
and `positive` name the initial parameter orientation only; either branch can
reverse parameter direction. Partial and unresolved results are retained.
"""
struct PeriodicContinuationResult{F,R,J,P}
    model_at_parameter::F
    rhs!::R
    jacobian!::J
    parameters_at_parameter::P
    initial_state::Vector{Float64}
    initial_period::Float64
    initial_parameter::Float64
    parameter_bounds::NTuple{2,Float64}
    options::PeriodicContinuationOptions
    periodic_options::PeriodicOrbitOptions
    initial_orbit::PeriodicOrbitResult
    negative::PeriodicContinuationBranch
    positive::PeriodicContinuationBranch
end

"""Return half of each sampled coordinate range of a validated orbit."""
function periodic_orbit_half_ranges(result::PeriodicOrbitResult)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("orbit observables require a numerically validated periodic orbit"))
    return result.amplitudes ./ 2
end

"""
    periodic_orbit_distances(result, center)

Return the minimum, root-mean-square, and maximum Euclidean distances from
the sampled orbit to a supplied two-dimensional center.
"""
function periodic_orbit_distances(result::PeriodicOrbitResult, center)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("orbit observables require a numerically validated periodic orbit"))
    center_values = _periodic_seed(center)
    states = @view result.states[1:end-1]
    distances = [norm(state .- center_values) for state in states]
    return (minimum=minimum(distances), rms=sqrt(sum(abs2, distances) / length(distances)),
        maximum=maximum(distances))
end

function _periodic_signed_area(states)
    anchor = first(states)
    local_states = [state .- anchor for state in states]
    local_origin = first(local_states) .* 0
    for state in local_states
        local_origin .+= state ./ length(local_states)
    end
    area = 0.0
    for index in eachindex(local_states)
        left = local_states[index] .- local_origin
        right = local_states[mod1(index + 1, length(local_states))] .- local_origin
        area += left[1] * right[2] - right[1] * left[2]
    end
    return area / 2
end

"""Return the signed shoelace area of a validated sampled planar orbit."""
function periodic_orbit_signed_area(result::PeriodicOrbitResult)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("orbit observables require a numerically validated periodic orbit"))
    return _periodic_signed_area(result.states)
end

"""
    periodic_orbit_winding(result, center; center_atol=1e-10,
                           initial_samples=256, max_samples=4096,
                           max_segment_angle=pi/2,
                           angle_agreement_atol=1e-3)

Numerically unwrap the orbit angle around `center`. Each segment must satisfy
an angular-resolution bound estimated from dense-solution derivatives, and a
doubled grid must agree before a winding is reported. Sampling doubles up to
`max_samples`; center crossings, unresolved angular resolution, or grid
disagreement return `resolved=false` and `winding=nothing`.
"""
function _periodic_winding_sample(result, center, samples, center_atol,
    max_segment_angle)
    phases = collect(range(0.0, 1.0; length=samples + 1))
    offsets = [Vector{Float64}(result.solution(phase)[1:2]) .- center for phase in phases]
    minimum_distance = Inf
    angle_change = 0.0
    maximum_resolved_angle = 0.0
    for index in 1:samples
        left, right = offsets[index], offsets[index + 1]
        segment = right .- left
        segment_norm_squared = sum(abs2, segment)
        fraction = iszero(segment_norm_squared) ? 0.0 :
            clamp(-dot(left, segment) / segment_norm_squared, 0.0, 1.0)
        minimum_distance = min(minimum_distance, norm(left .+ fraction .* segment))
        minimum_distance <= center_atol && return (resolved=false,
            reason=:center_crossing, winding=nothing, angle_change=NaN,
            minimum_distance, maximum_segment_angle=Inf, samples)
        angle_change += atan(left[1] * right[2] - left[2] * right[1], dot(left, right))
        for phase in (phases[index], (phases[index] + phases[index + 1]) / 2,
                      phases[index + 1])
            offset = Vector{Float64}(result.solution(phase)[1:2]) .- center
            derivative = result.solution(phase, Val{1})[1:2]
            radius_squared = sum(abs2, offset)
            radius_squared > center_atol^2 || return (resolved=false,
                reason=:center_crossing, winding=nothing, angle_change=NaN,
                minimum_distance=min(minimum_distance, sqrt(radius_squared)),
                maximum_segment_angle=Inf, samples)
            angular_rate = abs(offset[1] * derivative[2] - offset[2] * derivative[1]) /
                radius_squared
            maximum_resolved_angle = max(maximum_resolved_angle, angular_rate / samples)
        end
    end
    winding_value = round(Int, angle_change / (2pi))
    integer_resolved = abs(angle_change / (2pi) - winding_value) <= 1e-3
    angular_resolved = maximum_resolved_angle <= max_segment_angle
    resolved = integer_resolved && angular_resolved
    reason = !angular_resolved ? :angular_resolution :
        (!integer_resolved ? :noninteger_angle_change : :resolved)
    return (resolved, reason, winding=resolved ? winding_value : nothing,
        angle_change, minimum_distance,
        maximum_segment_angle=maximum_resolved_angle, samples)
end

function periodic_orbit_winding(result::PeriodicOrbitResult, center;
    center_atol=1e-10, initial_samples=256, max_samples=4096,
    max_segment_angle=pi / 2, angle_agreement_atol=1e-3)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("orbit observables require a numerically validated periodic orbit"))
    center_atol = _periodic_float64(center_atol, "center_atol")
    center_atol > 0 || throw(ArgumentError("center_atol must be positive"))
    for (value, name) in ((initial_samples, "initial_samples"),
                          (max_samples, "max_samples"))
        value isa Integer && !(value isa Bool) && value >= 8 ||
            throw(ArgumentError("$name must be an integer of at least eight"))
    end
    initial_samples <= max_samples ||
        throw(ArgumentError("initial_samples must not exceed max_samples"))
    max_segment_angle = _periodic_float64(max_segment_angle, "max_segment_angle")
    angle_agreement_atol = _periodic_float64(angle_agreement_atol,
        "angle_agreement_atol")
    0 < max_segment_angle < pi && angle_agreement_atol > 0 ||
        throw(ArgumentError("angular winding tolerances must be positive and max_segment_angle below pi"))
    center_values = _periodic_seed(center)
    samples = Int(initial_samples)
    while true
        coarse = _periodic_winding_sample(result, center_values, samples,
            center_atol, max_segment_angle)
        doubled_samples = 2samples
        doubled_samples <= max_samples || return merge(coarse,
            (resolved=false, reason=coarse.reason == :resolved ?
                :doubled_grid_unavailable : coarse.reason, winding=nothing))
        fine = _periodic_winding_sample(result, center_values, doubled_samples,
            center_atol, max_segment_angle)
        if coarse.resolved && fine.resolved && coarse.winding == fine.winding &&
           abs(coarse.angle_change - fine.angle_change) <= angle_agreement_atol
            return merge(fine, (reason=:resolved,))
        end
        doubled_samples == max_samples && return merge(fine,
            (resolved=false,
             reason=!fine.resolved ? fine.reason : :sampling_disagreement,
             winding=nothing))
        samples = doubled_samples
    end
end

"""
    periodic_orbit_primitive_check(result; max_divisor=8, alias_atol=1e-5)

Screen whether the stored period is an integer multiple of a shorter sampled
period. This finite divisor check is numerical evidence, not a proof that the
period is primitive.
"""
function periodic_orbit_primitive_check(result::PeriodicOrbitResult;
    max_divisor=8, alias_atol=1e-5)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("primitive-period screening requires a numerically validated periodic orbit"))
    max_divisor isa Integer && !(max_divisor isa Bool) && max_divisor >= 2 ||
        throw(ArgumentError("max_divisor must be an integer of at least two"))
    alias_atol = _periodic_float64(alias_atol, "alias_atol")
    alias_atol > 0 || throw(ArgumentError("alias_atol must be positive"))
    phases = collect(range(0.0, 1.0; length=257))[1:end-1]
    minimum_alias_distance = Inf
    for divisor in 2:Int(max_divisor)
        shift = 1 / divisor
        distances = Float64[]
        for phase in phases
            left = result.solution(phase)[1:2]
            right = result.solution(mod(phase + shift, 1.0))[1:2]
            push!(distances, norm(left .- right))
        end
        discrepancy = maximum(distances)
        minimum_alias_distance = min(minimum_alias_distance, discrepancy)
        if discrepancy <= alias_atol
            return (primitive=false, alias_divisor=divisor,
                minimum_alias_distance=minimum_alias_distance, alias_atol=alias_atol)
        end
    end
    return (primitive=true, alias_divisor=nothing,
        minimum_alias_distance=minimum_alias_distance, alias_atol=alias_atol)
end

"""
    phase_invariant_orbit_equivalence(left, right; samples=256,
                                      phase_refinement_iters=32,
                                      state_atol=1e-5, period_rtol=1e-5)

Compare validated orbits after a coarse cyclic phase search followed by a
continuous golden-section refinement of the best phase interval. The returned
distance is the smallest maximum Euclidean sample mismatch found.
"""
function phase_invariant_orbit_equivalence(left::PeriodicOrbitResult,
    right::PeriodicOrbitResult; samples=256, phase_refinement_iters=32,
    state_atol=1e-5, period_rtol=1e-5)
    left.validation == NumericallyValidatedPeriodicOrbit &&
        right.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("orbit equivalence requires numerically validated periodic orbits"))
    samples isa Integer && !(samples isa Bool) && samples >= 8 ||
        throw(ArgumentError("samples must be an integer of at least eight"))
    phase_refinement_iters isa Integer && !(phase_refinement_iters isa Bool) &&
        phase_refinement_iters > 0 ||
        throw(ArgumentError("phase_refinement_iters must be a positive integer"))
    state_atol = _periodic_float64(state_atol, "state_atol")
    period_rtol = _periodic_float64(period_rtol, "period_rtol")
    state_atol > 0 && period_rtol > 0 ||
        throw(ArgumentError("orbit-equivalence tolerances must be positive"))
    period_difference = abs(left.period - right.period) / max(left.period, right.period)
    phases = collect(0:(Int(samples) - 1)) ./ Int(samples)
    left_states = [left.solution(phase)[1:2] for phase in phases]
    function phase_distance(shift)
        distance = 0.0
        for index in eachindex(phases)
            distance = max(distance, norm(left_states[index] .-
                right.solution(mod(phases[index] + shift, 1.0))[1:2]))
        end
        return distance
    end
    best_distance, best_index = Inf, 0
    for index in 0:(Int(samples) - 1)
        distance = phase_distance(index / Int(samples))
        if distance < best_distance
            best_distance, best_index = distance, index
        end
    end
    spacing = 1 / Int(samples)
    lower = best_index * spacing - spacing
    upper = best_index * spacing + spacing
    inverse_phi = (sqrt(5.0) - 1) / 2
    left_probe = upper - inverse_phi * (upper - lower)
    right_probe = lower + inverse_phi * (upper - lower)
    left_distance = phase_distance(mod(left_probe, 1.0))
    right_distance = phase_distance(mod(right_probe, 1.0))
    for _ in 1:Int(phase_refinement_iters)
        if left_distance <= right_distance
            upper, right_probe, right_distance = right_probe, left_probe, left_distance
            left_probe = upper - inverse_phi * (upper - lower)
            left_distance = phase_distance(mod(left_probe, 1.0))
        else
            lower, left_probe, left_distance = left_probe, right_probe, right_distance
            right_probe = lower + inverse_phi * (upper - lower)
            right_distance = phase_distance(mod(right_probe, 1.0))
        end
    end
    refined_shift = mod((lower + upper) / 2, 1.0)
    refined_distance = phase_distance(refined_shift)
    if refined_distance < best_distance
        best_distance = refined_distance
        best_shift = refined_shift
    else
        best_shift = best_index / Int(samples)
    end
    equivalent = period_difference <= period_rtol && best_distance <= state_atol
    return (equivalent=equivalent, distance=best_distance,
        phase_shift=best_shift, period_difference=period_difference)
end

"""
    periodic_orbit_divergence_check(result, jacobian!, parameters=result.model;
                                    atol=1e-4)

Cross-check the planar Floquet product against `exp(integral(trace(J)) dt)`.
This is an independent sampled quadrature of the divergence identity, not an
additional integration or a rigorous error bound.
"""
function periodic_orbit_divergence_check(result::PeriodicOrbitResult, jacobian!,
    parameters=result.model; atol=1e-4)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        throw(ArgumentError("divergence checking requires a numerically validated periodic orbit"))
    atol = _periodic_float64(atol, "atol")
    atol > 0 || throw(ArgumentError("atol must be positive"))
    traces = Float64[]
    matrix = zeros(2, 2)
    for state in result.states
        jacobian!(matrix, state, parameters)
        all(isfinite, matrix) || return (resolved=false, divergence_multiplier=NaN,
            monodromy_determinant=det(result.monodromy),
            transverse_multiplier=real(result.transverse_multiplier), discrepancy=Inf)
        push!(traces, matrix[1, 1] + matrix[2, 2])
    end
    integral = 0.0
    for index in 1:(length(traces) - 1)
        integral += (traces[index] + traces[index + 1]) *
            (result.times[index + 1] - result.times[index]) / 2
    end
    divergence_multiplier = exp(integral)
    determinant = det(result.monodromy)
    transverse = real(result.transverse_multiplier)
    discrepancy = maximum(abs.((divergence_multiplier - determinant,
        divergence_multiplier - transverse, determinant - transverse)))
    return (resolved=isfinite(discrepancy) && discrepancy <= atol,
        divergence_multiplier=divergence_multiplier,
        monodromy_determinant=determinant, transverse_multiplier=transverse,
        discrepancy=discrepancy)
end

function _pc_parameters(factory, parameter, ::Type{P}) where {P}
    parameters = factory(parameter)
    parameters isa P ||
        throw(ArgumentError("parameters_at_parameter must return a consistent type"))
    return parameters
end

function _pc_system(rhs!, jacobian!, factory, z, scales, bounds, options,
    periodic_options, domain, reference, phase_direction, ::Type{P}) where {P}
    parameter = z[4] * scales[4]
    bounds[1] <= parameter <= bounds[2] || return nothing
    log_period = z[3] * scales[3]
    period = exp(log_period)
    isfinite(period) && periodic_options.min_period <= period <= periodic_options.max_period ||
        return nothing
    state = Float64[z[1] * scales[1], z[2] * scales[2]]
    domain !== nothing && any(value -> !(domain[1] <= value <= domain[2]), state) &&
        return nothing
    parameters = _pc_parameters(factory, parameter, P)
    solution, complete = _periodic_integrate(rhs!, jacobian!, parameters, state,
        period, periodic_options, 1.0, domain)
    complete || return (; state, period, parameter, parameters, solution,
        complete=false, residual=fill(NaN, 3), equation_residual=Inf)
    endpoint = solution.u[end]
    residual = Float64[endpoint[1] - state[1], endpoint[2] - state[2],
        dot(state .- reference, phase_direction)]
    return (; state, period, parameter, parameters, solution, complete=true,
        residual, equation_residual=maximum(abs, residual))
end

function _pc_parameter_stencil(parameter, bounds, options)
    nominal = options.parameter_difference_step * max(1.0, abs(parameter))
    lower_room = parameter - bounds[1]
    upper_room = bounds[2] - parameter
    lower_room >= 0 && upper_room >= 0 || return nothing
    if lower_room >= nominal && upper_room >= nominal
        lower = max(bounds[1], parameter - nominal)
        upper = min(bounds[2], parameter + nominal)
    elseif upper_room >= nominal
        lower, upper = parameter, min(bounds[2], parameter + nominal)
    elseif lower_room >= nominal
        lower, upper = max(bounds[1], parameter - nominal), parameter
    else
        # A balanced narrow interval still supports a useful centered
        # difference. Near a bound, prefer the wider one-sided span instead
        # of collapsing a centered stencil to ulp scale.
        centered_half_width = min(lower_room, upper_room)
        if centered_half_width > 0 &&
           2centered_half_width >= max(lower_room, upper_room)
            lower = parameter - centered_half_width
            upper = parameter + centered_half_width
        elseif upper_room >= lower_room
            lower, upper = parameter, bounds[2]
        else
            lower, upper = bounds[1], parameter
        end
    end
    scale = max(1.0, abs(parameter), abs(bounds[1]), abs(bounds[2]))
    all(isfinite, (lower, upper)) && upper > lower || return nothing
    span = upper - lower
    isfinite(span) && span > 16eps(scale) || return nothing
    return lower, upper
end

function _pc_constraint_jacobian(rhs!, jacobian!, factory, system, scales, bounds,
    options, periodic_options, domain, phase_direction, ::Type{P}) where {P}
    system.complete || return nothing
    endpoint = system.solution.u[end]
    monodromy = reshape(endpoint[3:6], 2, 2)
    end_rhs = zeros(2)
    rhs!(end_rhs, @view(endpoint[1:2]), system.parameters)
    all(isfinite, end_rhs) || return nothing
    stencil = _pc_parameter_stencil(system.parameter, bounds, options)
    stencil === nothing && return nothing
    lower, upper = stencil
    endpoints = Vector{Vector{Float64}}()
    for trial_parameter in (lower, upper)
        parameters = _pc_parameters(factory, trial_parameter, P)
        solution, complete = _periodic_integrate(rhs!, jacobian!, parameters,
            system.state, system.period, periodic_options, 1.0, domain)
        complete || return nothing
        push!(endpoints, Vector{Float64}(solution.u[end][1:2]))
    end
    parameter_derivative = (endpoints[2] .- endpoints[1]) ./ (upper - lower)
    jacobian = zeros(3, 4)
    jacobian[1:2, 1:2] .= monodromy
    jacobian[1, 1] -= 1
    jacobian[2, 2] -= 1
    jacobian[1:2, 3] .= system.period .* end_rhs
    jacobian[1:2, 4] .= parameter_derivative
    jacobian[3, 1:2] .= phase_direction
    for column in axes(jacobian, 2)
        jacobian[:, column] .*= scales[column]
    end
    return all(isfinite, jacobian) ? jacobian : nothing
end

function _pc_tangent(jacobian, options, previous, direction)
    jacobian === nothing && return nothing
    decomposition = svd(jacobian; full=true)
    threshold = options.rank_atol + options.rank_rtol * maximum(decomposition.S)
    minimum(decomposition.S) > threshold || return nothing
    tangent = decomposition.V[:, end]
    if previous === nothing
        abs(tangent[4]) > options.rank_atol || return nothing
        tangent[4] * direction < 0 && (tangent .*= -1)
    elseif dot(tangent, previous) < 0
        tangent .*= -1
    end
    return tangent
end

function _pc_correct(rhs!, jacobian!, factory, predictor, tangent, scales, bounds,
    options, periodic_options, domain, reference, phase_direction, ::Type{P}) where {P}
    z = copy(predictor)
    last_system = nothing
    for iteration in 0:options.max_corrector_iters
        system = _pc_system(rhs!, jacobian!, factory, z, scales, bounds, options,
            periodic_options, domain, reference, phase_direction, P)
        arc = dot(z .- predictor, tangent)
        system === nothing && return (z, iteration, arc, Inf,
            :augmented_coordinates_out_of_bounds, last_system)
        last_system = system
        system.complete || return (z, iteration, arc, Inf,
            :incomplete_integration, system)
        residual_norm = max(system.equation_residual, abs(arc))
        residual_norm <= options.corrector_atol &&
            return (z, iteration, arc, system.equation_residual,
                :corrector_converged, system)
        iteration == options.max_corrector_iters &&
            return (z, iteration, arc, system.equation_residual,
                :corrector_iteration_limit, system)
        constraint_jacobian = _pc_constraint_jacobian(rhs!, jacobian!, factory,
            system, scales, bounds, options, periodic_options, domain,
            phase_direction, P)
        constraint_jacobian === nothing && return (z, iteration, arc,
            system.equation_residual, :constraint_jacobian_unresolved, system)
        bordered = vcat(constraint_jacobian, transpose(tangent))
        singular_values = svdvals(bordered)
        minimum(singular_values) > options.rank_atol +
            options.rank_rtol * maximum(singular_values) ||
            return (z, iteration, arc, system.equation_residual,
                :corrector_rank_unresolved, system)
        correction = bordered \ vcat(system.residual, arc)
        all(isfinite, correction) || return (z, iteration, arc,
            system.equation_residual, :nonfinite_correction, system)
        accepted = false
        for backtrack in 0:10
            trial = z .- (0.5^backtrack) .* correction
            trial_system = _pc_system(rhs!, jacobian!, factory, trial, scales,
                bounds, options, periodic_options, domain, reference,
                phase_direction, P)
            (trial_system === nothing || !trial_system.complete) && continue
            trial_arc = dot(trial .- predictor, tangent)
            trial_norm = max(trial_system.equation_residual, abs(trial_arc))
            if trial_norm < residual_norm
                z = trial
                accepted = true
                break
            end
        end
        accepted || return (z, iteration, arc, system.equation_residual,
            :corrector_line_search_failed, system)
    end
    error("unreachable periodic corrector state")
end

function _pc_phase_direction(rhs!, state, parameters, options)
    direction = zeros(2)
    rhs!(direction, state, parameters)
    all(isfinite, direction) && norm(direction) > options.phase_speed_atol ||
        return nothing
    return direction ./ norm(direction)
end

function _pc_point(parameter, orbit, tangent, jacobian!, parameters)
    return PeriodicContinuationPoint(parameter, copy(orbit.initial_state), orbit.period,
        copy(tangent), orbit, periodic_orbit_half_ranges(orbit),
        periodic_orbit_signed_area(orbit), periodic_orbit_primitive_check(orbit),
        periodic_orbit_divergence_check(orbit, jacobian!, parameters))
end

function _pc_parameter_orientation(component, options)
    tolerance = options.rank_atol + options.rank_rtol
    return abs(component) <= tolerance ? 0 : (component < 0 ? -1 : 1)
end

function _pc_record_parameter_orientation!(reversals, point_index,
    last_orientation, component, options)
    orientation = _pc_parameter_orientation(component, options)
    orientation == 0 && return last_orientation
    last_orientation != 0 && orientation != last_orientation &&
        push!(reversals, point_index)
    return orientation
end

function _pc_continue_branch(rhs!, jacobian!, factory, initial_orbit, initial_parameter,
    bounds, options, periodic_options, domain, ::Type{P}, direction) where {P}
    points = PeriodicContinuationPoint[]
    attempts = PeriodicContinuationAttempt[]
    reversals = Int[]
    finish(status) = PeriodicContinuationBranch(direction, points, attempts, reversals, status)
    initial_orbit.validation == NumericallyValidatedPeriodicOrbit ||
        return finish(:initial_orbit_unresolved)
    scales = Float64[options.state_scales..., options.log_period_scale,
        options.parameter_scale]
    parameter = Float64(initial_parameter)
    parameters = _pc_parameters(factory, parameter, P)
    phase_direction = _pc_phase_direction(rhs!, initial_orbit.initial_state,
        parameters, periodic_options)
    phase_direction === nothing && return finish(:initial_phase_unresolved)
    z = vcat(initial_orbit.initial_state, log(initial_orbit.period), parameter) ./ scales
    system = _pc_system(rhs!, jacobian!, factory, z, scales, bounds, options,
        periodic_options, domain, initial_orbit.initial_state, phase_direction, P)
    constraint_jacobian = system === nothing ? nothing : _pc_constraint_jacobian(
        rhs!, jacobian!, factory, system, scales, bounds, options,
        periodic_options, domain, phase_direction, P)
    tangent = _pc_tangent(constraint_jacobian, options, nothing, direction)
    tangent === nothing && return finish(:initial_tangent_unresolved)
    push!(points, _pc_point(parameter, initial_orbit, tangent, jacobian!, parameters))
    last_parameter_orientation = _pc_parameter_orientation(tangent[4], options)
    step = options.initial_step

    for _ in 1:options.max_steps
        while true
            if tangent[4] != 0
                bound = tangent[4] > 0 ? bounds[2] : bounds[1]
                distance = (bound / scales[4] - z[4]) / tangent[4]
                distance <= options.minimum_step && return finish(:parameter_boundary)
                step = min(step, 0.95 * distance)
            end
            step >= options.minimum_step || return finish(:minimum_step)
            predictor = z .+ step .* tangent
            candidate, iterations, corrector_arc, corrector_constraint,
                status, corrected_system =
                _pc_correct(rhs!, jacobian!, factory, predictor, tangent, scales,
                    bounds, options, periodic_options, domain,
                    points[end].state, phase_direction, P)
            candidate_physical = candidate .* scales
            orbit = nothing
            next_tangent = nothing
            next_parameters = nothing
            post_shoot_arc = NaN
            post_shoot_constraint = NaN
            if status == :corrector_converged && corrected_system !== nothing
                next_parameters = corrected_system.parameters
                orbit = _solve_periodic_orbit(rhs!, jacobian!, next_parameters,
                    corrected_system.state, corrected_system.period,
                    periodic_options, domain)
                if orbit.validation != NumericallyValidatedPeriodicOrbit
                    status = :orbit_validation_failed
                else
                    refined = vcat(orbit.initial_state, log(orbit.period),
                        corrected_system.parameter) ./ scales
                    candidate = refined
                    candidate_physical = candidate .* scales
                    post_shoot_system = _pc_system(rhs!, jacobian!, factory,
                        refined, scales, bounds, options, periodic_options, domain,
                        points[end].state, phase_direction, P)
                    post_shoot_arc = dot(refined .- predictor, tangent)
                    post_shoot_constraint = post_shoot_system === nothing ||
                        !post_shoot_system.complete ? Inf :
                        post_shoot_system.equation_residual
                    if max(abs(post_shoot_arc), post_shoot_constraint) >
                       options.corrector_atol
                        status = :post_shoot_constraint_mismatch
                    elseif norm(refined .- predictor) > 2step
                        status = :excessive_corrector_distance
                    else
                        next_phase = _pc_phase_direction(rhs!, orbit.initial_state,
                            next_parameters, periodic_options)
                        next_system = next_phase === nothing ? nothing : _pc_system(
                            rhs!, jacobian!, factory, refined, scales, bounds, options,
                            periodic_options, domain, orbit.initial_state, next_phase, P)
                        next_jacobian = next_system === nothing ? nothing :
                            _pc_constraint_jacobian(rhs!, jacobian!, factory,
                                next_system, scales, bounds, options,
                                periodic_options, domain, next_phase, P)
                        next_tangent = _pc_tangent(next_jacobian, options, tangent, direction)
                        if next_phase === nothing
                            status = :phase_reference_unresolved
                        elseif next_tangent === nothing
                            status = :tangent_rank_unresolved
                        elseif dot(next_tangent, tangent) < options.tangent_alignment_min
                            status = :excessive_tangent_rotation
                        else
                            phase_direction = next_phase
                        end
                    end
                end
            end
            accepted = status == :corrector_converged
            push!(attempts, PeriodicContinuationAttempt(direction, step,
                predictor .* scales, candidate_physical, iterations,
                corrector_arc, corrector_constraint, post_shoot_arc,
                post_shoot_constraint, status, accepted, orbit))
            if accepted
                z, tangent = candidate, next_tangent
                push!(points, _pc_point(z[4] * scales[4], orbit, tangent,
                    jacobian!, next_parameters))
                last_parameter_orientation = _pc_record_parameter_orientation!(
                    reversals, length(points), last_parameter_orientation,
                    tangent[4], options)
                step = min(options.maximum_step, 1.25step)
                break
            end
            step /= 2
        end
    end
    return finish(:step_limit)
end

function _pc_validate_inputs(initial_state, initial_period, initial_parameter,
    parameter_bounds, options, periodic_options)
    options isa PeriodicContinuationOptions ||
        throw(ArgumentError("options must be PeriodicContinuationOptions"))
    periodic_options isa PeriodicOrbitOptions ||
        throw(ArgumentError("periodic_options must be PeriodicOrbitOptions"))
    state = _periodic_seed(initial_state)
    period = _periodic_float64(initial_period, "initial_period")
    periodic_options.min_period <= period <= periodic_options.max_period ||
        throw(ArgumentError("initial_period must lie within the periodic-orbit bounds"))
    parameter = _periodic_float64(initial_parameter, "initial_parameter")
    parameter_bounds isa Tuple && length(parameter_bounds) == 2 ||
        throw(ArgumentError("parameter_bounds must be a (lower, upper) tuple"))
    bounds = Tuple(_periodic_float64(value, "parameter bound") for value in parameter_bounds)
    bounds[1] < bounds[2] ||
        throw(ArgumentError("parameter bounds must be strictly ordered"))
    bounds[1] <= parameter <= bounds[2] ||
        throw(ArgumentError("initial_parameter must lie within parameter bounds"))
    return state, period, parameter, bounds
end

"""
    continue_periodic_orbit(model_at_parameter, initial_state, initial_period,
                            initial_parameter; parameter_bounds,
                            options=PeriodicContinuationOptions(),
                            periodic_options=PeriodicOrbitOptions())

Continue an independently validated planar periodic orbit in both initial
parameter directions using pseudo-arclength shooting in
`[state, log(period), parameter]`. The factory must return autonomous
`PointModelParameters` of a consistent type and every model is constrained to
the physical `[0,1]^2` domain. All failed corrections, retries, tangent
reversals, boundary stops, and unresolved terminations are retained.
"""
function continue_periodic_orbit(model_at_parameter, initial_state,
    initial_period, initial_parameter; parameter_bounds,
    options=PeriodicContinuationOptions(),
    periodic_options=PeriodicOrbitOptions())
    _validate_initial_state(initial_state)
    state, period, parameter, bounds = _pc_validate_inputs(initial_state,
        initial_period, initial_parameter, parameter_bounds, options,
        periodic_options)
    raw_initial_model = model_at_parameter(parameter)
    raw_initial_model isa PointModelParameters ||
        throw(ArgumentError("model_at_parameter must return PointModelParameters"))
    model_type = typeof(raw_initial_model)
    validate_model = model -> begin
        model isa model_type || throw(ArgumentError(
            "model_at_parameter must return a consistent PointModelParameters type"))
        model.drive isa NoDrive ||
            (model.drive isa PiecewiseConstantDrive && isempty(model.drive.pulses)) ||
            throw(ArgumentError("periodic continuation requires globally constant drives"))
        foreach(value -> _periodic_float64(value, "model parameter"),
            (_model_numeric_values(model)..., drive_value(model.drive, 0.0)...))
        return model
    end
    model_factory = value -> validate_model(model_at_parameter(value))
    initial_model = validate_model(raw_initial_model)
    rhs! = (du, u, model) -> point_rhs!(du, u, model, 0.0)
    jacobian! = (matrix, u, model) -> point_jacobian!(matrix, u, model, 0.0)
    initial_orbit = _solve_periodic_orbit(rhs!, jacobian!, initial_model, state,
        period, periodic_options, (0.0, 1.0))
    negative = _pc_continue_branch(rhs!, jacobian!, model_factory, initial_orbit,
        parameter, bounds, options, periodic_options, (0.0, 1.0), model_type, -1)
    positive = _pc_continue_branch(rhs!, jacobian!, model_factory, initial_orbit,
        parameter, bounds, options, periodic_options, (0.0, 1.0), model_type, 1)
    return PeriodicContinuationResult(model_at_parameter, rhs!, jacobian!,
        model_factory, state, period, parameter, bounds, options,
        periodic_options, initial_orbit, negative, positive)
end

"""
    continue_periodic_orbit(rhs!, jacobian!, parameters_at_parameter,
                            initial_state, initial_period, initial_parameter;
                            parameter_bounds, options, periodic_options)

Callback overload for autonomous planar systems. `rhs!` and `jacobian!` have
the same signatures as `solve_periodic_orbit`; `parameters_at_parameter(p)`
must return a consistent supported callback-parameter type. Factory errors
propagate as input errors. A failed finite continuation does not establish
absence of a periodic branch.
"""
function continue_periodic_orbit(rhs!, jacobian!, parameters_at_parameter,
    initial_state, initial_period, initial_parameter; parameter_bounds,
    options=PeriodicContinuationOptions(),
    periodic_options=PeriodicOrbitOptions())
    state, period, parameter, bounds = _pc_validate_inputs(initial_state,
        initial_period, initial_parameter, parameter_bounds, options,
        periodic_options)
    raw_initial_parameters = parameters_at_parameter(parameter)
    _periodic_parameter_check(raw_initial_parameters)
    parameter_type = typeof(raw_initial_parameters)
    parameter_factory = value -> begin
        parameters = parameters_at_parameter(value)
        parameters isa parameter_type || throw(ArgumentError(
            "parameters_at_parameter must return a consistent type"))
        _periodic_parameter_check(parameters)
        return parameters
    end
    initial_parameters = raw_initial_parameters
    initial_orbit = _solve_periodic_orbit(rhs!, jacobian!, initial_parameters,
        state, period, periodic_options, nothing)
    negative = _pc_continue_branch(rhs!, jacobian!, parameter_factory,
        initial_orbit, parameter, bounds, options, periodic_options, nothing,
        parameter_type, -1)
    positive = _pc_continue_branch(rhs!, jacobian!, parameter_factory,
        initial_orbit, parameter, bounds, options, periodic_options, nothing,
        parameter_type, 1)
    return PeriodicContinuationResult(parameters_at_parameter, rhs!, jacobian!,
        parameter_factory, state, period, parameter, bounds, options,
        periodic_options, initial_orbit, negative, positive)
end
