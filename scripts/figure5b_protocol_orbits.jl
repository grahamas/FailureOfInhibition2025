"""Figure 5b protocol functions included by run_figure5b_protocol.jl."""
function _section_return(model, state, period, section_normal, section_tangent,
    epsilon, intervals, options)
    initial = state .+ epsilon .* section_tangent
    all(value -> 0 <= value <= 1, initial) || return nothing
    problem = ODEProblem(point_rhs!, initial, (0.0, 1.5period), model;
        isoutofdomain=(u, _, _) -> any(value -> !isfinite(value) || value < 0 || value > 1, u))
    solution = solve(problem, Tsit5(); abstol=options.ode_abstol,
        reltol=options.ode_reltol, maxiters=options.ode_maxiters, dense=true,
        save_everystep=true, verbose=false)
    successful_retcode(solution) || return nothing
    section_value(time) = dot(section_normal, solution(time) .- state)
    times = range(0.5period, 1.5period; length=intervals + 1)
    left_time, left_value = first(times), section_value(first(times))
    rhs = zeros(2)
    for right_time in Iterators.drop(times, 1)
        right_value = section_value(right_time)
        if left_value <= 0 <= right_value || right_value <= 0 <= left_value
            lower, upper = left_time, right_time
            lower_value = left_value
            for _ in 1:60
                midpoint = (lower + upper) / 2
                midpoint_value = section_value(midpoint)
                if lower_value * midpoint_value <= 0
                    upper = midpoint
                else
                    lower, lower_value = midpoint, midpoint_value
                end
            end
            crossing_time = (lower + upper) / 2
            crossing = Vector{Float64}(solution(crossing_time))
            point_rhs!(rhs, crossing, model, crossing_time)
            if dot(section_normal, rhs) > 0
                return (coordinate=dot(section_tangent, crossing .- state),
                    time=crossing_time)
            end
        end
        left_time, left_value = right_time, right_value
    end
    return nothing
end

"""Independent finite-difference Poincare return multiplier."""
function poincare_multiplier(result, config)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        return (resolved=false, multiplier=NaN, values=Float64[], rows=NamedTuple[])
    model = result.model
    state = result.initial_state
    velocity = zeros(2)
    point_rhs!(velocity, state, model, 0.0)
    norm(velocity) > config.shooting_options.phase_speed_atol ||
        return (resolved=false, multiplier=NaN, values=Float64[], rows=NamedTuple[])
    normal = velocity ./ norm(velocity)
    tangent = [-normal[2], normal[1]]
    values, rows = Float64[], NamedTuple[]
    for step in config.independent.poincare_steps
        positive = _section_return(model, state, result.period, normal, tangent,
            step, config.independent.poincare_intervals, config.shooting_options)
        negative = _section_return(model, state, result.period, normal, tangent,
            -step, config.independent.poincare_intervals, config.shooting_options)
        if positive === nothing || negative === nothing
            push!(rows, (; step, positive_return=NaN, negative_return=NaN,
                positive_time=NaN, negative_time=NaN, multiplier=NaN))
            continue
        end
        multiplier = (positive.coordinate - negative.coordinate) / (2step)
        push!(values, multiplier)
        push!(rows, (; step, positive_return=positive.coordinate,
            negative_return=negative.coordinate, positive_time=positive.time,
            negative_time=negative.time, multiplier))
    end
    resolved = length(values) == length(config.independent.poincare_steps) &&
        maximum(values) - minimum(values) <= config.independent.poincare_plateau_atol
    return (resolved, multiplier=resolved ? sum(values) / length(values) : NaN,
        values, rows)
end

function orbit_acceptance_reasons(; numerically_validated, attracting, primitive,
    equivalent, jacobian_resolved, divergence_resolved, poincare_resolved,
    poincare_difference, multiplier_atol, windings, central_index)
    reasons = Symbol[]
    numerically_validated || push!(reasons, :tight_replay_unresolved)
    attracting || push!(reasons, :orbit_not_attracting)
    primitive || push!(reasons, :nonprimitive_period)
    equivalent || push!(reasons, :tight_replay_mismatch)
    jacobian_resolved || push!(reasons, :jacobian_mismatch)
    divergence_resolved || push!(reasons, :divergence_multiplier_mismatch)
    poincare_resolved || push!(reasons, :poincare_unresolved)
    poincare_resolved && poincare_difference > multiplier_atol &&
        push!(reasons, :poincare_multiplier_mismatch)
    if length(windings) == 0 || !all(row -> row.resolved, windings)
        push!(reasons, :winding_unresolved)
    elseif central_index === nothing || !(1 <= central_index <= length(windings))
        push!(reasons, :central_equilibrium_unresolved)
    else
        abs(windings[central_index].winding) == 1 ||
            push!(reasons, :wrong_central_winding)
        all(index -> index == central_index || windings[index].winding == 0,
            eachindex(windings)) || push!(reasons, :encloses_other_equilibrium)
    end
    return unique(reasons)
end

function validate_orbit(result, baseline, roots, config)
    reasons = Symbol[]
    if result.validation != NumericallyValidatedPeriodicOrbit
        push!(reasons, :tight_replay_unresolved)
        result.stability == PeriodicOrbitAttracting || push!(reasons, :orbit_not_attracting)
        return (accepted=false, reasons, primitive=nothing,
            equivalence=nothing, divergence=nothing, jacobian=nothing,
            poincare=nothing, windings=NamedTuple[])
    end
    primitive = periodic_orbit_primitive_check(result)
    equivalence = phase_invariant_orbit_equivalence(baseline, result;
        state_atol=config.independent.orbit_equivalence_atol,
        period_rtol=config.independent.period_rtol)
    jacobian = jacobian_crosscheck(result.model, result.states[1:end-1], config)
    jacobian_callback = (matrix, state, model) -> point_jacobian!(matrix, state, model, 0.0)
    divergence = periodic_orbit_divergence_check(result, jacobian_callback, result.model;
        atol=config.independent.multiplier_atol)
    poincare = poincare_multiplier(result, config)
    windings = NamedTuple[]
    for (index, root) in enumerate(roots)
        winding = periodic_orbit_winding(result, root.state;
            center_atol=config.independent.winding_center_atol)
        push!(windings, (; equilibrium=index, E=root.state[1], I=root.state[2],
            resolved=winding.resolved, reason=string(winding.reason),
            winding=isnothing(winding.winding) ? 0 : winding.winding,
            minimum_distance=winding.minimum_distance, samples=winding.samples))
    end
    repellers = filter(index -> roots[index].stability.classification == Repelling,
        eachindex(roots))
    central = length(repellers) == 1 ? only(repellers) : nothing
    reasons = orbit_acceptance_reasons(
        numerically_validated=true,
        attracting=result.stability == PeriodicOrbitAttracting,
        primitive=primitive.primitive, equivalent=equivalence.equivalent,
        jacobian_resolved=jacobian.resolved,
        divergence_resolved=divergence.resolved,
        poincare_resolved=poincare.resolved,
        poincare_difference=poincare.resolved ?
            abs(poincare.multiplier - real(result.transverse_multiplier)) : Inf,
        multiplier_atol=config.independent.multiplier_atol,
        windings=windings, central_index=central)
    return (accepted=isempty(reasons), reasons, primitive, equivalence,
        divergence, jacobian, poincare, windings)
end

function write_tight_validation_artifact(output, selected, result, validation,
    center, p; stage_error=nothing)
    if result === nothing || validation === nothing
        Evidence.write_toml(joinpath(output, "tight_validation.toml"), Dict(
            "status" => stage_error === nothing ? "not_run" : "exception",
            "selected_attempt" => selected === nothing ? "not_available" : selected,
            "error" => stage_error === nothing ? "not_available" : stage_error))
        return nothing
    end
    validated = result.validation == NumericallyValidatedPeriodicOrbit
    half_ranges = validated ? periodic_orbit_half_ranges(result) : [NaN, NaN]
    distances = validated ? periodic_orbit_distances(result, center) :
        (minimum=NaN, rms=NaN, maximum=NaN)
    area = validated ? periodic_orbit_signed_area(result) : NaN
    radius = validated ? modal_radius(result, center, p) : NaN
    Evidence.write_toml(joinpath(output, "tight_validation.toml"), Dict(
        "status" => "completed", "selected_attempt" => selected,
        "accepted" => validation.accepted,
        "reasons" => string.(validation.reasons),
        "validation" => result.validation, "stability" => result.stability,
        "initial_state" => result.initial_state, "period" => result.period,
        "component_half_ranges" => half_ranges,
        "center_distances" => distances, "signed_area" => area,
        "modal_radius" => radius,
        "closure_residual" => result.closure_residual,
        "phase_residual" => result.phase_residual,
        "equation_residual" => result.equation_residual,
        "refinement_difference" => result.refinement_difference,
        "integration_success" => result.integration_success,
        "floquet_multipliers" => result.floquet_multipliers,
        "transverse_multiplier" => result.transverse_multiplier,
        "monodromy" => result.monodromy,
        "primitive_period" => validation.primitive,
        "phase_equivalence" => validation.equivalence,
        "divergence_check" => validation.divergence,
        "jacobian_check" => validation.jacobian === nothing ? "not_available" :
            Dict("resolved" => validation.jacobian.resolved,
                "maximum_discrepancy" => validation.jacobian.maximum_discrepancy),
        "poincare_check" => validation.poincare,
        "windings" => validation.windings))
    return nothing
end

function modal_radius(result, center, p)
    radii = [abs(dot(p, state .- center)) for state in result.states[1:end-1]]
    return sqrt(sum(abs2, radii) / length(radii))
end

branch_divergence_observation(point) = (
    divergence_resolved=point.divergence_check.resolved,
    divergence_multiplier=point.divergence_check.divergence_multiplier,
    divergence_discrepancy=point.divergence_check.discrepancy)

branch_primitive_observation(point) = (
    primitive=point.primitive_period_check.primitive,
    primitive_alias=point.primitive_period_check.alias_divisor)

function branch_evidence_resolved(row)
    return hasproperty(row, :validated) && row.validated === true &&
        hasproperty(row, :attracting) && row.attracting === true &&
        hasproperty(row, :divergence_resolved) && row.divergence_resolved === true &&
        hasproperty(row, :primitive) && row.primitive === true
end

function branch_observations(continuation, center, p, roots, config)
    observations = Dict{Symbol,Vector{NamedTuple}}()
    for (branch_name, branch) in ((:negative, continuation.negative),
            (:positive, continuation.positive))
        rows = NamedTuple[]
        for (index, point) in enumerate(branch.points)
            radius = modal_radius(point.orbit, center, p)
            saddle_distances = [periodic_orbit_distances(point.orbit, root.state).minimum
                for root in roots if root.stability.classification == Saddle]
            boundary_distance = minimum(vcat(
                [minimum(state) for state in point.orbit.states],
                [1 - maximum(state) for state in point.orbit.states]))
            divergence = branch_divergence_observation(point)
            primitive = branch_primitive_observation(point)
            push!(rows, (; branch=string(branch_name), point=index,
                ratio=point.parameter, modal_radius=radius,
                radius_squared=radius^2, period=point.period,
                transverse_multiplier=real(point.orbit.transverse_multiplier),
                validated=point.orbit.validation == NumericallyValidatedPeriodicOrbit,
                attracting=point.orbit.stability == PeriodicOrbitAttracting,
                divergence...,
                primitive...,
                saddle_distance=isempty(saddle_distances) ? Inf : minimum(saddle_distances),
                boundary_distance))
        end
        observations[branch_name] = rows
    end
    return observations
end

function endpoint_classification(termination, observations, reversals,
    critical_ratio, linear_period, expected_boundary, config)
    reasons = Symbol[]
    isempty(observations) && return (classification=:unresolved,
        reasons=[:no_accepted_points], hopf_compatible=false,
        fold_compatible=false, homoclinic_compatible=false,
        state_boundary_compatible=false)
    last_point = last(observations)
    monotone_toward_hopf = all(observations[index + 1].ratio < observations[index].ratio
        for index in 1:(length(observations) - 1))
    no_reversal = isempty(reversals)
    validated = all(row -> hasproperty(row, :validated) && row.validated === true,
        observations)
    attracting = all(row -> hasproperty(row, :attracting) && row.attracting === true,
        observations)
    divergence_resolved = all(row -> hasproperty(row, :divergence_resolved) &&
        row.divergence_resolved === true, observations)
    primitive = all(row -> hasproperty(row, :primitive) && row.primitive === true,
        observations)
    evidence_resolved = all(branch_evidence_resolved, observations)
    boundary_matches = expected_boundary !== nothing &&
        abs(last_point.ratio - expected_boundary) <= config.independent.hopf_ratio_atol
    hopf_compatible = termination == :parameter_boundary && no_reversal &&
        monotone_toward_hopf && evidence_resolved && boundary_matches &&
        0 < last_point.ratio - critical_ratio <= config.independent.hopf_ratio_atol &&
        last_point.modal_radius <= config.independent.hopf_radius_atol &&
        abs(last_point.period - linear_period) / linear_period <=
            config.independent.hopf_period_rtol

    real_terminal_status = termination in (:parameter_boundary, :minimum_step, :step_limit)
    terminal_reversal = length(observations) - 1
    fold_compatible = real_terminal_status && length(observations) >= 3 &&
        terminal_reversal in reversals &&
        all(row -> branch_evidence_resolved(row) &&
            abs(row.transverse_multiplier - 1) <=
            config.independent.fold_multiplier_atol, observations[(end - 2):end])
    terminal_count = min(3, length(observations))
    terminal_segment = observations[(end - terminal_count + 1):end]
    terminal_validated = terminal_count >= 3 &&
        all(branch_evidence_resolved, terminal_segment)
    terminal_ratio_decreasing = terminal_validated &&
        all(terminal_segment[index + 1].ratio < terminal_segment[index].ratio
            for index in 1:(terminal_count - 1))
    terminal_ratio_increasing = terminal_validated &&
        all(terminal_segment[index + 1].ratio > terminal_segment[index].ratio
            for index in 1:(terminal_count - 1))
    terminal_monotone = terminal_ratio_decreasing || terminal_ratio_increasing
    robust_period_growth = terminal_monotone &&
        last_point.period >= config.independent.homoclinic_period_factor *
            first(observations).period &&
        all(terminal_segment[index + 1].period > terminal_segment[index].period
            for index in 1:(terminal_count - 1))
    robust_saddle_approach = terminal_monotone &&
        last_point.saddle_distance <=
            config.independent.homoclinic_saddle_distance_atol &&
        all(terminal_segment[index + 1].saddle_distance <
            terminal_segment[index].saddle_distance
            for index in 1:(terminal_count - 1))
    homoclinic_compatible = real_terminal_status && robust_period_growth &&
        robust_saddle_approach
    state_boundary_compatible = real_terminal_status && terminal_monotone &&
        last_point.boundary_distance <= config.independent.boundary_state_atol &&
        all(terminal_segment[index + 1].boundary_distance <=
            terminal_segment[index].boundary_distance
            for index in 1:(terminal_count - 1))
    boundary = termination == :parameter_boundary
    classification = hopf_compatible ? :hopf_compatible :
        fold_compatible ? :fold_compatible :
        homoclinic_compatible ? :homoclinic_or_heteroclinic_compatible :
        state_boundary_compatible ? :state_boundary_compatible :
        boundary ? :boundary : :unresolved
    !monotone_toward_hopf && push!(reasons, :not_monotone_toward_hopf)
    !no_reversal && !fold_compatible && push!(reasons, :unresolved_parameter_reversal)
    !validated && push!(reasons, :unvalidated_branch_point)
    !attracting && push!(reasons, :nonattracting_branch_orbit)
    !divergence_resolved && push!(reasons, :divergence_multiplier_unresolved)
    !primitive && push!(reasons, :nonprimitive_branch_orbit)
    termination in (:minimum_step, :step_limit, :initial_tangent_unresolved,
        :initial_phase_unresolved, :initial_orbit_unresolved) &&
        push!(reasons, :unsuccessful_termination)
    classification == :boundary && push!(reasons, :nonhopf_parameter_boundary)
    classification == :unresolved && isempty(reasons) && push!(reasons, :endpoint_unresolved)
    return (; classification, reasons=unique(reasons), hopf_compatible,
        fold_compatible, homoclinic_compatible, state_boundary_compatible)
end

function branch_fit(continuation, observations, critical_ratio, config)
    eligible = NamedTuple[]
    for (branch_name, branch) in ((:negative, continuation.negative),
            (:positive, continuation.positive))
        rows = observations[branch_name]
        cutoff = isempty(branch.parameter_reversals) ? length(rows) :
            max(0, first(branch.parameter_reversals) - 1)
        segment = rows[1:cutoff]
        monotone = length(segment) >= 2 && all(segment[index + 1].ratio <
            segment[index].ratio for index in 1:(length(segment) - 1))
        candidates = monotone ? filter(row -> 0 < row.ratio - critical_ratio <=
            config.independent.fit_offset_max, segment) : NamedTuple[]
        length(candidates) >= config.independent.fit_minimum_points &&
            push!(eligible, (; branch_name, candidates,
                endpoint_offset=last(candidates).ratio - critical_ratio))
    end
    if isempty(eligible)
        return (resolved=false, branch="not_available", intercept=NaN, slope=NaN,
            r_squared=NaN, points=0, rows=NamedTuple[])
    end
    selected = first(sort(eligible; by=item -> item.endpoint_offset))
    candidates = selected.candidates
    x = [row.ratio - critical_ratio for row in candidates]
    y = [row.radius_squared for row in candidates]
    design = hcat(ones(length(x)), x)
    coefficients = design \ y
    fitted = design * coefficients
    total = sum(abs2, y .- sum(y) / length(y))
    residual = sum(abs2, y .- fitted)
    r_squared = iszero(total) ? NaN : 1 - residual / total
    return (resolved=all(isfinite, coefficients) && isfinite(r_squared),
        branch=string(selected.branch_name), intercept=coefficients[1],
        slope=coefficients[2], r_squared,
        points=length(candidates), rows=candidates)
end

function fit_acceptance_assessment(fit, endpoints, diagnostics, config)
    reasons = Symbol[]
    if fit === nothing || !fit.resolved
        return (accepted=false, branch=nothing, endpoint=nothing,
            nearest=nothing, reasons=[:fit_unresolved])
    end
    branch = try
        Symbol(fit.branch)
    catch
        nothing
    end
    endpoint = branch === nothing ? nothing : get(endpoints, branch, nothing)
    branch === nothing && push!(reasons, :fit_branch_invalid)
    endpoint === nothing && push!(reasons, :fit_endpoint_branch_missing)
    endpoint !== nothing && fit.branch != string(branch) &&
        push!(reasons, :fit_endpoint_branch_mismatch)
    nearest = isempty(fit.rows) ? nothing : last(fit.rows)
    fit_evidence_resolved = !isempty(fit.rows) &&
        all(branch_evidence_resolved, fit.rows)
    if !fit_evidence_resolved
        all(row -> hasproperty(row, :validated) && row.validated === true,
            fit.rows) || push!(reasons, :fit_orbit_validation_unresolved)
        all(row -> hasproperty(row, :attracting) && row.attracting === true,
            fit.rows) || push!(reasons, :fit_nonattracting_orbit)
        all(row -> hasproperty(row, :divergence_resolved) &&
            row.divergence_resolved === true, fit.rows) ||
            push!(reasons, :fit_divergence_unresolved)
        all(row -> hasproperty(row, :primitive) && row.primitive === true,
            fit.rows) || push!(reasons, :fit_nonprimitive_orbit)
    end
    fit.points >= config.independent.fit_minimum_points ||
        push!(reasons, :insufficient_fit_points)
    abs(fit.intercept) <= config.independent.fit_intercept_atol ||
        push!(reasons, :fit_intercept_too_large)
    fit.r_squared >= config.independent.fit_r_squared_minimum ||
        push!(reasons, :fit_r_squared_too_small)
    isapprox(fit.slope, diagnostics.modal_radius_squared_slope;
        rtol=config.independent.fit_slope_rtol, atol=0.0) ||
        push!(reasons, :fit_slope_disagreement)
    endpoint !== nothing && endpoint.hopf_compatible ||
        push!(reasons, :endpoint_not_hopf_compatible)
    if nearest === nothing
        push!(reasons, :nearest_hopf_point_missing)
    else
        0 < nearest.ratio - diagnostics.critical_ratio <=
            config.independent.hopf_ratio_atol ||
            push!(reasons, :endpoint_ratio_not_near_hopf)
        nearest.modal_radius <= config.independent.hopf_radius_atol ||
            push!(reasons, :endpoint_radius_not_near_zero)
        abs(nearest.period - diagnostics.linear_period) /
            diagnostics.linear_period <= config.independent.hopf_period_rtol ||
            push!(reasons, :endpoint_period_disagreement)
        nearest.modal_radius == minimum(row.modal_radius for row in fit.rows) ||
            push!(reasons, :radius_not_decreasing_to_endpoint)
        all(fit.rows[index + 1].modal_radius < fit.rows[index].modal_radius
            for index in 1:(length(fit.rows) - 1)) ||
            push!(reasons, :radius_not_strictly_decreasing_to_endpoint)
    end
    unique!(reasons)
    return (accepted=isempty(reasons), branch, endpoint, nearest, reasons)
end
