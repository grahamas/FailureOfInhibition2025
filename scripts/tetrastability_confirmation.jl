"""Tetrastability protocol functions included by run_tetrastability_search.jl."""
function _cell_is_candidate(result)
    !isnothing(result) &&
        count(eq -> eq.stability.classification == Attracting, result.equilibria) >=
            CANDIDATE_MINIMUM
end

function _candidate_reference(item)
    result = item.result
    compact_result = (; equilibria=result.equilibria,
        unresolved_nearby=result.unresolved_nearby)
    return (; search_id=item.search_id, model=item.model, result=compact_result)
end

function screen_cells!(state, config, cells, output)
    _append_rows(joinpath(output, "samples.csv"), cells)
    for cell in cells
        state.cells[cell.cell_id] = cell
        for condition in CONDITIONS
            item = _run_cached_search!(state, config, cell, condition, output)
            summary = _search_summary(item.search_id, cell, condition,
                config.screen_grid_points, item.result)
            _append_rows(joinpath(output, "searches.csv"), [summary])
            if condition == :failure_of_inhibition && _cell_is_candidate(item.result)
                state.discovery_candidates[cell.cell_id] = cell
                state.candidate_results[cell.cell_id] = _candidate_reference(item)
            end
        end
    end
    return state
end

function refined_options(options, factor)
    return EquilibriumOptions(
        solver_abstol=options.solver_abstol * factor,
        solver_reltol=options.solver_reltol * factor,
        residual_atol=options.residual_atol * factor,
        domain_atol=options.domain_atol,
        dedup_atol=options.dedup_atol,
        singular_atol=options.singular_atol * factor,
        singular_rtol=options.singular_rtol * factor,
        maxiters=2options.maxiters,
    )
end

function refined_options(options::StabilityOptions, factor)
    return StabilityOptions(spectral_atol=options.spectral_atol * factor,
        spectral_rtol=options.spectral_rtol * factor)
end

function _independent_observation(model, equilibrium)
    T = eltype(equilibrium.state)
    residual = zeros(T, 2)
    balance_jacobian = zeros(T, 2, 2)
    ode_jacobian = zeros(T, 2, 2)
    point_balance!(residual, equilibrium.state, model, zero(T))
    point_balance_jacobian!(balance_jacobian, equilibrium.state, model, zero(T))
    point_jacobian!(ode_jacobian, equilibrium.state, model, zero(T))
    return (; state=Tuple(equilibrium.state),
        attracting=equilibrium.stability.classification == Attracting,
        near_singular=equilibrium.near_singular,
        residual_norm=maximum(abs, equilibrium.balance_residual),
        recomputed_residual_norm=maximum(abs, residual),
        balance_jacobian_delta=maximum(abs, balance_jacobian .- equilibrium.balance_jacobian),
        ode_jacobian_delta=maximum(abs, ode_jacobian .- equilibrium.stability.jacobian),
        spectral_abscissa=equilibrium.stability.spectral_abscissa)
end

_state_distance(first, second) = maximum(abs.(collect(first.state) .- collect(second.state)))

function unique_root_match(reference, references, candidates, tolerance)
    return Evidence.unique_reciprocal_match(reference, references, candidates,
        tolerance, _state_distance)
end

function _minimum_separation(roots)
    length(roots) < 2 && return NaN
    return minimum(_state_distance(roots[first_index], roots[second_index])
        for first_index in 1:length(roots)-1 for second_index in first_index+1:length(roots))
end

function confirmation_assessment(reference_roots, refined_roots_by_grid,
    unresolved_by_grid, config)
    reference_attracting = findall(root -> root.attracting, reference_roots)
    tracks = NamedTuple[]
    for reference_index in reference_attracting
        matches = Dict{Int,Int}()
        for grid in config.confirmation_grid_points
            match = unique_root_match(reference_roots[reference_index], reference_roots,
                refined_roots_by_grid[grid], config.confirmation.match_atol)
            isnothing(match) || (matches[grid] = match)
        end
        length(matches) == length(config.confirmation_grid_points) || continue
        observations = [reference_roots[reference_index];
            [refined_roots_by_grid[grid][matches[grid]] for grid in config.confirmation_grid_points]]
        quality = all(root -> root.attracting && !root.near_singular &&
            root.recomputed_residual_norm <= config.equilibrium_options.residual_atol &&
            root.balance_jacobian_delta <= config.equilibrium_options.residual_atol &&
            root.ode_jacobian_delta <= config.equilibrium_options.residual_atol &&
            root.spectral_abscissa <= -config.confirmation.spectral_margin, observations)
        push!(tracks, (; track_id=length(tracks) + 1, reference_index, matches,
            observations, quality))
    end
    quality_tracks = filter(track -> track.quality, tracks)
    separations = Float64[]
    push!(separations, _minimum_separation(
        [reference_roots[track.reference_index] for track in quality_tracks]))
    for grid in config.confirmation_grid_points
        push!(separations, _minimum_separation(
            [refined_roots_by_grid[grid][track.matches[grid]] for track in quality_tracks]))
    end
    separation = any(isnan, separations) ? NaN : minimum(separations)
    separation_required = config.confirmation.separation_factor *
        config.equilibrium_options.dedup_atol
    reasons = String[]
    length(quality_tracks) >= CANDIDATE_MINIMUM ||
        push!(reasons, "fewer_than_four_uniquely_matched_attracting_roots")
    any(values(unresolved_by_grid)) && push!(reasons, "unresolved_nearby_roots")
    (!isnan(separation) && separation > separation_required) ||
        push!(reasons, "insufficient_root_separation")
    observations = [root for track in tracks for root in track.observations]
    metrics = isempty(observations) ?
        (; worst_spectral_abscissa=NaN, maximum_recomputed_residual=NaN,
            maximum_balance_jacobian_delta=NaN, maximum_ode_jacobian_delta=NaN) :
        (; worst_spectral_abscissa=maximum(root.spectral_abscissa for root in observations),
            maximum_recomputed_residual=maximum(root.recomputed_residual_norm for root in observations),
            maximum_balance_jacobian_delta=maximum(root.balance_jacobian_delta for root in observations),
            maximum_ode_jacobian_delta=maximum(root.ode_jacobian_delta for root in observations))
    return (; confirmed=isempty(reasons), tracks, quality_tracks,
        matched_attracting_roots=length(quality_tracks), minimum_separation=separation,
        reasons, metrics...)
end

function _confirmation_search(state, config, cell, condition, grid_points, candidate_id, output)
    state.confirmation_search_counter[] += 1
    search_id = "$(candidate_id)_$(condition)_grid$(grid_points)"
    model = getproperty(models_for_cell(config, cell), condition)
    equilibrium_options = refined_options(config.equilibrium_options,
        config.confirmation.tolerance_factor)
    stability_options = refined_options(config.stability_options,
        config.confirmation.tolerance_factor)
    seeds = Map.deterministic_seeds(model, grid_points)
    result = try
        find_equilibria(model; seeds, options=equilibrium_options,
            stability_options=stability_options)
    catch error
        error isa InterruptException && rethrow()
        Evidence.write_toml(joinpath(output, "contexts", "confirmations", search_id * ".toml"),
            Dict("status" => "execution_failed", "error" => Evidence.error_record(error),
                "requested_seeds" => seeds, "model" => model))
        return (; search_id, model, result=nothing)
    end
    Evidence.write_toml(joinpath(output, "contexts", "confirmations", search_id * ".toml"),
        merge(Dict("search_id" => search_id, "condition" => string(condition),
            "grid_points" => grid_points, "model" => model,
            "requested_seeds" => seeds), Evidence.context_record(result)))
    _append_rows(joinpath(output, "attempts.csv"),
        [merge((; search_id), Base.structdiff(Evidence.summary_attempt(search_id, index, attempt),
            NamedTuple{(:context_id,)})) for (index, attempt) in enumerate(result.attempts)])
    _append_rows(joinpath(output, "equilibria.csv"),
        _equilibrium_rows(search_id, condition, grid_points, model, result))
    return (; search_id, model, result)
end

function confirm_candidate!(state, config, cell, output)
    state.confirmation_counter[] += 1
    candidate_id = "candidate_$(lpad(state.confirmation_counter[], 4, '0'))"
    reference_item = state.candidate_results[cell.cell_id]
    reference_result = reference_item.result
    refined = Dict{Tuple{Symbol,Int},Any}()
    for grid in config.confirmation_grid_points, condition in CONDITIONS
        item = _confirmation_search(state, config, cell, condition, grid, candidate_id, output)
        refined[(condition, grid)] = item
        isnothing(item.result) && push!(state.failures, item.search_id)
        summary = _search_summary(item.search_id, cell, condition, grid, item.result)
        _append_rows(joinpath(output, "searches.csv"), [summary])
    end
    foi_complete = all(grid -> !isnothing(refined[(:failure_of_inhibition, grid)].result),
        config.confirmation_grid_points)
    control_complete = all(grid -> !isnothing(refined[(:control, grid)].result),
        config.confirmation_grid_points)
    if foi_complete
        reference_roots = [_independent_observation(reference_item.model, equilibrium)
            for equilibrium in reference_result.equilibria]
        refined_roots = Dict(grid => [_independent_observation(
            refined[(:failure_of_inhibition, grid)].model, equilibrium)
            for equilibrium in refined[(:failure_of_inhibition, grid)].result.equilibria]
            for grid in config.confirmation_grid_points)
        unresolved = Dict(0 => !isempty(reference_result.unresolved_nearby))
        for grid in config.confirmation_grid_points
            unresolved[grid] = !isempty(refined[(:failure_of_inhibition, grid)].result.unresolved_nearby)
        end
        assessment = confirmation_assessment(reference_roots, refined_roots, unresolved, config)
    else
        assessment = (; confirmed=false, tracks=NamedTuple[], quality_tracks=NamedTuple[],
            matched_attracting_roots=0, minimum_separation=NaN,
            reasons=["failure_of_inhibition_confirmation_failed"],
            worst_spectral_abscissa=NaN, maximum_recomputed_residual=NaN,
            maximum_balance_jacobian_delta=NaN, maximum_ode_jacobian_delta=NaN)
    end
    control_complete || push!(assessment.reasons, "matched_control_confirmation_failed")
    confirmed = assessment.confirmed && control_complete
    status = confirmed ? "confirmed" : "not_confirmed"
    attracting_count(item) = isnothing(item.result) ? -1 :
        count(eq -> eq.stability.classification == Attracting, item.result.equilibria)
    row = (; candidate_id, cell_id=cell.cell_id, e_to_e=cell.e_to_e,
        i_to_e=cell.i_to_e, e_to_i=cell.e_to_i, i_to_i=cell.i_to_i,
        theta_off=cell.theta_off,
        discovery_attracting=attracting_count(reference_item),
        grid_21_attracting=haskey(refined, (:failure_of_inhibition, 21)) ?
            attracting_count(refined[(:failure_of_inhibition, 21)]) : -1,
        grid_41_attracting=haskey(refined, (:failure_of_inhibition, 41)) ?
            attracting_count(refined[(:failure_of_inhibition, 41)]) : -1,
        matched_attracting_roots=assessment.matched_attracting_roots,
        minimum_separation=assessment.minimum_separation,
        worst_spectral_abscissa=assessment.worst_spectral_abscissa,
        maximum_recomputed_residual=assessment.maximum_recomputed_residual,
        maximum_balance_jacobian_delta=assessment.maximum_balance_jacobian_delta,
        maximum_ode_jacobian_delta=assessment.maximum_ode_jacobian_delta,
        confirmation_status=status, reasons=join(unique(assessment.reasons), ";"))
    _append_rows(joinpath(output, "candidates.csv"), [row])

    match_rows = NamedTuple[]
    if foi_complete
        grids = [config.screen_grid_points; config.confirmation_grid_points]
        items = Dict(config.screen_grid_points => reference_item,
            (grid => refined[(:failure_of_inhibition, grid)] for grid in config.confirmation_grid_points)...)
        for track in assessment.tracks
            root_indices = Dict(config.screen_grid_points => track.reference_index,
                (grid => track.matches[grid] for grid in config.confirmation_grid_points)...)
            for grid in grids
                item = items[grid]
                equilibrium_index = root_indices[grid]
                observation = _independent_observation(item.model,
                    item.result.equilibria[equilibrium_index])
                reference = _independent_observation(reference_item.model,
                    reference_result.equilibria[track.reference_index])
                push!(match_rows, (; candidate_id, root_track=track.track_id, grid_points=grid,
                    search_id=item.search_id, equilibrium=equilibrium_index,
                    match_distance=_state_distance(reference, observation),
                    E=observation.state[1], I=observation.state[2],
                    attracting=observation.attracting, near_singular=observation.near_singular,
                    residual_norm=observation.residual_norm,
                    recomputed_residual_norm=observation.recomputed_residual_norm,
                    balance_jacobian_delta=observation.balance_jacobian_delta,
                    ode_jacobian_delta=observation.ode_jacobian_delta,
                    spectral_abscissa=observation.spectral_abscissa,
                    quality_pass=track.quality))
            end
        end
    end
    _append_rows(joinpath(output, "root_matches.csv"), match_rows)
    return (; candidate_id, cell, row, assessment, refined, confirmed)
end
