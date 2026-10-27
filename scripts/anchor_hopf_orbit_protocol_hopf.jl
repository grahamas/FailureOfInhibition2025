"""Anchor Hopf/orbit protocol functions included by run_anchor_hopf_orbit_protocol.jl."""
function finite_difference_jacobian(model, state, step)
    step = number(step, "finite-difference step"; positive=true)
    jacobian = zeros(Float64, 2, 2)
    plus, minus = zeros(2), zeros(2)
    for column in 1:2
        left, right = copy(state), copy(state)
        left[column] -= step
        right[column] += step
        point_rhs!(minus, left, model, 0.0)
        point_rhs!(plus, right, model, 0.0)
        jacobian[:, column] .= (plus .- minus) ./ (2step)
    end
    return jacobian
end

function numerical_trace_zero(config, state)
    lower, upper = config.ratios.hopf_bracket
    step = last(config.independent.jacobian_steps)
    trace_at(ratio) = begin
        matrix = finite_difference_jacobian(candidate_model(config, ratio), state, step)
        matrix[1, 1] + matrix[2, 2]
    end
    lower_trace, upper_trace = trace_at(lower), trace_at(upper)
    lower_trace * upper_trace <= 0 || return (resolved=false, ratio=NaN,
        trace=NaN, iterations=0, rows=[(; iteration=0, lower, upper,
            lower_trace, upper_trace, midpoint=NaN, midpoint_trace=NaN)])
    rows = NamedTuple[]
    midpoint, midpoint_trace = NaN, NaN
    for iteration in 1:config.independent.trace_maxiters
        midpoint = (lower + upper) / 2
        midpoint_trace = trace_at(midpoint)
        push!(rows, (; iteration, lower, upper, lower_trace, upper_trace,
            midpoint, midpoint_trace))
        if abs(upper - lower) <= config.independent.trace_ratio_atol
            return (resolved=true, ratio=midpoint, trace=midpoint_trace,
                iterations=iteration, rows)
        end
        if lower_trace * midpoint_trace <= 0
            upper, upper_trace = midpoint, midpoint_trace
        else
            lower, lower_trace = midpoint, midpoint_trace
        end
    end
    return (resolved=false, ratio=midpoint, trace=midpoint_trace,
        iterations=config.independent.trace_maxiters, rows)
end

"""Return deterministically phased right/adjoint Hopf eigenvectors."""
function hopf_basis(model, state, frequency)
    matrix = zeros(2, 2)
    point_jacobian!(matrix, state, model, 0.0)
    decomposition = eigen(complex.(matrix))
    q_index = argmin(abs.(decomposition.values .- im * frequency))
    q = decomposition.vectors[:, q_index]
    q ./= norm(q)
    pivot = argmax(abs.(q))
    q .*= exp(-im * angle(q[pivot]))
    real(q[pivot]) < 0 && (q .*= -1)
    adjoint_decomposition = eigen(adjoint(complex.(matrix)))
    p_index = argmin(abs.(adjoint_decomposition.values .+ im * frequency))
    p0 = adjoint_decomposition.vectors[:, p_index]
    p = p0 / conj(dot(p0, q))
    return (q=ComplexF64.(q), p=ComplexF64.(p), matrix,
        eigenvalue=ComplexF64(decomposition.values[q_index]),
        normalization=ComplexF64(dot(p, q)))
end

function seed_schedule(config, diagnostics, basis, central_state; smoke=false)
    phases = smoke ? config.smoke.phase_fractions : config.seeding.phase_fractions
    amplitudes = smoke ? config.smoke.amplitude_factors : config.seeding.amplitude_factors
    periods = smoke ? config.smoke.period_factors : config.seeding.period_factors
    target_ratios = unique(vcat(diagnostics.critical_ratio .+ config.seeding.ratio_offsets,
        config.ratios.orbit_anchor))
    rows = NamedTuple[]
    attempt = 0
    for ratio in target_ratios, phase in phases, amplitude_factor in amplitudes,
            period_factor in periods
        attempt += 1
        modal_radius = sqrt(max(0.0,
            diagnostics.modal_radius_squared_slope * (ratio - diagnostics.critical_ratio))) *
            amplitude_factor
        displacement = 2 .* real.(basis.q .* (modal_radius * cis(2pi * phase)))
        state = Float64.(central_state .+ displacement)
        push!(rows, (; attempt_id="orbit_$(lpad(attempt, 4, '0'))", ratio, phase,
            amplitude_factor, period_factor, modal_radius,
            seed_E=state[1], seed_I=state[2],
            period_guess=diagnostics.linear_period * period_factor))
    end
    return rows
end

function cluster_validated_attempts(attempts, config)
    validated = filter(attempt -> attempt.result !== nothing &&
        attempt.result.validation == NumericallyValidatedPeriodicOrbit, attempts)
    clusters = Vector{Vector{Any}}()
    for attempt in validated
        assigned = false
        for cluster in clusters
            comparison = phase_invariant_orbit_equivalence(
                first(cluster).result, attempt.result;
                state_atol=config.independent.orbit_equivalence_atol,
                period_rtol=config.independent.period_rtol)
            if comparison.equivalent
                push!(cluster, attempt)
                assigned = true
                break
            end
        end
        assigned || push!(clusters, Any[attempt])
    end
    sort!(clusters; by=cluster -> (-length(cluster), first(cluster).attempt_id))
    return clusters
end

function qualifying_cluster(clusters, config)
    for cluster in clusters
        phases = unique(attempt.phase for attempt in cluster)
        periods = unique(attempt.period_factor for attempt in cluster)
        if length(phases) >= config.independent.minimum_phase_seeds &&
                length(periods) >= config.independent.minimum_period_seeds
            return cluster
        end
    end
    return nothing
end

function tighter_periodic_options(config)
    source = config.shooting_options
    factor = config.independent.tighter_factor
    return PeriodicOrbitOptions(
        ode_abstol=source.ode_abstol * factor,
        ode_reltol=source.ode_reltol * factor,
        shooting_atol=source.shooting_atol * factor,
        validation_atol=source.validation_atol * factor,
        amplitude_atol=source.amplitude_atol,
        phase_speed_atol=source.phase_speed_atol,
        floquet_atol=source.floquet_atol * factor,
        refinement_factor=source.refinement_factor,
        min_period=source.min_period, max_period=source.max_period,
        maxiters=source.maxiters, ode_maxiters=source.ode_maxiters,
        samples=config.independent.tight_samples)
end

function jacobian_crosscheck(model, states, config)
    state_values = states isa AbstractVector && length(states) == 2 &&
        all(value -> value isa Real, states) ? [states] : collect(states)
    rows = NamedTuple[]
    discrepancies = Float64[]
    analytical = zeros(2, 2)
    for (sample, state) in enumerate(state_values), step in config.independent.jacobian_steps
        point_jacobian!(analytical, state, model, 0.0)
        finite_difference = finite_difference_jacobian(model, state, step)
        discrepancy = maximum(abs, analytical .- finite_difference)
        push!(discrepancies, discrepancy)
        push!(rows, (; sample, step, discrepancy,
            analytical_11=analytical[1, 1], analytical_12=analytical[1, 2],
            analytical_21=analytical[2, 1], analytical_22=analytical[2, 2],
            fd_11=finite_difference[1, 1], fd_12=finite_difference[1, 2],
            fd_21=finite_difference[2, 1], fd_22=finite_difference[2, 2]))
    end
    return (resolved=all(isfinite, discrepancies) &&
        maximum(discrepancies) <= config.independent.jacobian_atol,
        maximum_discrepancy=maximum(discrepancies), rows)
end
