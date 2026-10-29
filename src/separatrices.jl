using LinearAlgebra: eigen, norm
using SciMLBase: ContinuousCallback, CallbackSet, terminate!, successful_retcode

"""
    SeparatrixOptions(; seed_offset=1e-5, maximum_time=20000,
        equilibrium_radius=1e-6, abstol=1e-10, reltol=1e-8,
        tightening_factor=0.1, display_span=0.5, display_pixels=512,
        geometry_atol=display_span/display_pixels, maxiters=1000000,
        max_points=20000)

Numerical policy for saddle stable-manifold observations. Each signed branch is
traced twice, using `seed_offset` and half that offset; the second integration
also tightens both ODE tolerances. `geometry_atol` defaults to one pixel in a
`display_span`-wide display. Tracing always uses the full physical `[0,1]^2`
domain. These observations do not certify a basin boundary or completeness.
"""
struct SeparatrixOptions{T<:AbstractFloat}
    seed_offset::T
    maximum_time::T
    equilibrium_radius::T
    abstol::T
    reltol::T
    tightening_factor::T
    geometry_atol::T
    maxiters::Int
    max_points::Int
end

function SeparatrixOptions(; seed_offset=1e-5, maximum_time=20000,
    equilibrium_radius=1e-6, abstol=1e-10, reltol=1e-8,
    tightening_factor=0.1, display_span=0.5, display_pixels=512,
    geometry_atol=nothing, maxiters=1000000, max_points=20000)
    for (name, value) in (("display_pixels", display_pixels),
        ("maxiters", maxiters), ("max_points", max_points))
        minimum = name == "maxiters" ? 1 : 2
        value isa Integer && !(value isa Bool) && minimum <= value <= typemax(Int) ||
            throw(ArgumentError("$name must be an integer of at least $minimum"))
    end
    display_span isa Real && !(display_span isa Bool) &&
        isfinite(display_span) && display_span > 0 ||
        throw(ArgumentError("display_span must be finite and positive"))
    tolerance = geometry_atol === nothing ? display_span / display_pixels : geometry_atol
    values = (seed_offset, maximum_time, equilibrium_radius, abstol, reltol,
        tightening_factor, tolerance)
    all(value -> value isa Real && !(value isa Bool) && isfinite(value) && value > 0,
        values) || throw(ArgumentError("separatrix scales and tolerances must be finite and positive"))
    seed_offset < 1 || throw(ArgumentError("seed_offset must be less than one"))
    tightening_factor < 1 || throw(ArgumentError("tightening_factor must be less than one"))
    promoted = promote(float.(values)...)
    return SeparatrixOptions(promoted..., Int(maxiters), Int(max_points))
end

function _separatrix_direction(stability)
    decomposition = eigen(stability.jacobian)
    index = findfirst(value -> real(value) < 0, decomposition.values)
    index === nothing && throw(ArgumentError("saddle has no negative eigenvalue"))
    direction = real.(decomposition.vectors[:, index])
    direction ./= norm(direction)
    pivot = argmax(abs.(direction))
    direction[pivot] < 0 && (direction .*= -1)
    return direction, real(decomposition.values[index])
end

function _separatrix_reverse_rhs!(output, state, model, time)
    point_rhs!(output, state, model, zero(time))
    output .*= -one(eltype(output))
    return nothing
end

# Dense interpolation resolves both curvature and long spatial chords. The
# physical event state is retained as computed, with no boundary projection.
function _separatrix_curve(solution, root, options)
    T = eltype(root)
    points = Vector{T}[copy(root), copy(first(solution.u))]
    tolerance = T(options.geometry_atol) / 16
    spacing = T(options.geometry_atol) * 2
    complete = Ref(true)
    function append_interval(left_time, left, right_time, right, depth)
        if length(points) >= options.max_points
            complete[] = false
            return
        end
        middle_time = (left_time + right_time) / 2
        middle = solution(middle_time)
        needs_split = norm(right - left) > spacing ||
            norm(middle - (left + right) / 2) > tolerance
        if needs_split && depth < 24 && left_time < middle_time < right_time
            append_interval(left_time, left, middle_time, middle, depth + 1)
            complete[] && append_interval(middle_time, middle, right_time, right, depth + 1)
        elseif needs_split
            complete[] = false
        else
            push!(points, copy(right))
        end
    end
    for index in 2:length(solution.t)
        append_interval(solution.t[index - 1], solution.u[index - 1],
            solution.t[index], solution.u[index], 0)
        complete[] || break
    end
    return points, complete[]
end

function _trace_separatrix_attempt(model, equilibria, root_index, direction,
    sign, offset, factor, options)
    root = equilibria.equilibria[root_index].state
    T = eltype(root)
    seed = root .+ T(sign) * T(offset) .* direction
    # Keep the refined solve above precision floors, raising the coarse solve
    # when necessary so that the two attempts still differ in ODE tolerance.
    refinement = T(options.tightening_factor)
    abstol = max(T(options.abstol), (eps(T) / 8) / refinement) * T(factor)
    reltol = max(T(options.reltol), (eps(T) * 8) / refinement) * T(factor)
    points = Vector{T}[]
    reason = Ref(:time_budget)
    destination = Ref{Union{Nothing,Int}}(nothing)
    finish(success, points, time, message=nothing) =
        (seed_offset=T(offset), abstol=abstol, reltol=reltol,
         successful=success, termination_reason=reason[],
         terminal_equilibrium=destination[], analysis_time=time,
         points=points, message=message)
    if !all(value -> isfinite(value) && value > 0, (abstol, reltol))
        reason[] = :tolerance_unresolved
        return finish(false, points, zero(T))
    end
    if !all(value -> isfinite(value) && 0 <= value <= 1, seed)
        reason[] = :seed_outside_domain
        return finish(false, points, zero(T))
    end
    if seed == root
        reason[] = :seed_unresolved
        return finish(false, points, zero(T))
    end
    others = [index for index in eachindex(equilibria.equilibria) if index != root_index]
    radius = T(options.equilibrium_radius)
    distance(state, index) = norm(state - equilibria.equilibria[index].state)
    near = findfirst(index -> distance(seed, index) <= radius, others)
    if near !== nothing
        reason[] = :seed_near_equilibrium
        destination[] = others[near]
        return finish(false, points, zero(T))
    end
    boundary(state, _, _) = min(state[1], state[2], 1 - state[1], 1 - state[2])
    function stop_boundary!(integrator)
        reason[] = :physical_boundary
        terminate!(integrator)
    end
    boundary_callback = ContinuousCallback(boundary, nothing, stop_boundary!;
        abstol=eps(T), reltol=zero(T), save_positions=(false, true))
    callback = if isempty(others)
        boundary_callback
    else
        nearest(state, _, _) = minimum(distance(state, index) for index in others) - radius
        function stop_equilibrium!(integrator)
            reason[] = :equilibrium_neighborhood
            destination[] = others[argmin([distance(integrator.u, index) for index in others])]
            terminate!(integrator)
        end
        CallbackSet(boundary_callback,
            ContinuousCallback(nearest, nothing, stop_equilibrium!;
                abstol=eps(T), reltol=zero(T), save_positions=(false, true)))
    end
    try
        # No isoutofdomain rejection: it would prevent the event solver from
        # bracketing the physical boundary and stall integration at the edge.
        problem = ODEProblem(_separatrix_reverse_rhs!, seed,
            (zero(T), T(options.maximum_time)), model)
        solution = solve(problem, Tsit5(); callback, abstol, reltol,
            maxiters=options.maxiters, dense=true, save_everystep=true,
            save_start=true, save_end=true, verbose=false)
        if !successful_retcode(solution) || any(state -> !all(isfinite, state), solution.u)
            reason[] = :integration_failed
            return finish(false, points, last(solution.t), string(solution.retcode))
        end
        boundary_error = T(32) * eps(T)
        if any(state -> any(value -> !(-boundary_error <= value <= 1 + boundary_error), state), solution.u)
            reason[] = :domain_event_failed
            return finish(false, points, last(solution.t))
        end
        points, resolved = _separatrix_curve(solution, root, options)
        resolved || (reason[] = :point_budget)
        if any(state -> any(value -> !(-boundary_error <= value <= 1 + boundary_error), state), points)
            reason[] = :domain_event_failed
        end
        success = resolved && reason[] in (:physical_boundary, :equilibrium_neighborhood)
        return finish(success, points, last(solution.t))
    catch error
        error isa InterruptException && rethrow()
        reason[] = :integration_exception
        return finish(false, points, zero(T), sprint(showerror, error))
    end
end

# Compare ordered curves at equal normalized arc length. Including the saddle
# makes the two seed offsets share the same geometrical starting point.
function _separatrix_disagreement(first_curve, second_curve, tolerance)
    function arc_lengths(points)
        result = zeros(eltype(first(points)), length(points))
        for index in 2:length(points)
            result[index] = result[index - 1] + norm(points[index] - points[index - 1])
        end
        return result
    end
    first_arc, second_arc = arc_lengths(first_curve), arc_lengths(second_curve)
    count = max(2, ceil(Int, max(last(first_arc), last(second_arc)) / (tolerance / 2)))
    positions = [1, 1]
    function interpolate(points, arc, position, fraction)
        target = last(arc) * fraction
        while position < length(arc) - 1 && arc[position + 1] < target
            position += 1
        end
        width = arc[position + 1] - arc[position]
        weight = iszero(width) ? zero(width) : (target - arc[position]) / width
        return points[position] + weight * (points[position + 1] - points[position]), position
    end
    largest = zero(eltype(first(first_curve)))
    for index in 0:count
        fraction = index / count
        left, positions[1] = interpolate(first_curve, first_arc, positions[1], fraction)
        right, positions[2] = interpolate(second_curve, second_arc, positions[2], fraction)
        largest = max(largest, norm(left - right))
    end
    return largest
end

"""
    trace_separatrices(model, equilibria; options=SeparatrixOptions())

Trace both stable-manifold branches of each resolved saddle by integrating the
negated autonomous vector field in positive analysis time. The autonomous model
and equilibrium context must match, as for pulse experiments. Boundary crossings
are located by continuous events; states are never clipped or projected.

Return a vector of branch records with `root_index`, `sign`, `stable_eigenvector`,
`stable_eigenvalue`, `converged`, `status`, `termination_reason`, `attempts`, and
`geometric_disagreement`. `points` (ordered `[E,I]` states), `E`, and `I` contain
only branches whose two attempts terminate successfully at the same boundary or
equilibrium neighborhood and agree within `geometry_atol`. Failed or incomplete
branches have empty display arrays; their attempted curves remain in `attempts`.
Effective ODE tolerances are retained in each attempt; precision floors raise
the coarse tolerances when needed to preserve an actual tightened comparison.
Boundary event coordinates may differ from 0 or 1 by floating-point roundoff.
No attracting-basin interpretation or completeness certification is implied.
"""
function trace_separatrices(model::PointModelParameters, equilibria;
    options=SeparatrixOptions())
    options isa SeparatrixOptions || throw(ArgumentError("options must be SeparatrixOptions"))
    _pulse_context(model, equilibria)
    branches = NamedTuple[]
    for (root_index, equilibrium) in enumerate(equilibria.equilibria)
        equilibrium.stability.classification == Saddle || continue
        direction, eigenvalue = _separatrix_direction(equilibrium.stability)
        for sign in (-1, 1)
            attempts = [_trace_separatrix_attempt(model, equilibria, root_index,
                direction, sign, options.seed_offset, one(options.tightening_factor), options),
                _trace_separatrix_attempt(model, equilibria, root_index,
                direction, sign, options.seed_offset / 2, options.tightening_factor, options)]
            both_successful = all(attempt -> attempt.successful, attempts)
            same_end = attempts[1].termination_reason == attempts[2].termination_reason &&
                attempts[1].terminal_equilibrium == attempts[2].terminal_equilibrium
            disagreement = both_successful ? _separatrix_disagreement(
                attempts[1].points, attempts[2].points, options.geometry_atol) : missing
            converged = both_successful && same_end && disagreement <= options.geometry_atol
            status = converged ? :converged : !both_successful ? :incomplete :
                !same_end ? :termination_disagreement : :geometric_disagreement
            points = converged ? attempts[2].points : Vector{eltype(equilibrium.state)}[]
            push!(branches, (root_index=root_index, sign=sign,
                root_state=copy(equilibrium.state), stable_eigenvector=copy(direction),
                stable_eigenvalue=eigenvalue, converged=converged, status=status,
                termination_reason=attempts[2].termination_reason,
                terminal_equilibrium=attempts[2].terminal_equilibrium,
                geometric_disagreement=disagreement, geometry_atol=options.geometry_atol,
                points=points, E=[point[1] for point in points],
                I=[point[2] for point in points], attempts=attempts))
        end
    end
    return branches
end
