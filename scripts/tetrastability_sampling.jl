"""Tetrastability protocol functions included by run_tetrastability_search.jl."""
function radical_inverse(index, base)
    index isa Integer && !(index isa Bool) && index > 0 ||
        throw(ArgumentError("Halton index must be a positive integer"))
    _is_prime(base) || throw(ArgumentError("Halton base must be prime"))
    value = 0.0
    factor = inv(Float64(base))
    current = Int(index)
    while current > 0
        current, digit = divrem(current, base)
        value += digit * factor
        factor /= base
    end
    return value
end

function _axis_value(axis, unit)
    return axis.lower + unit * (axis.upper - axis.lower)
end

function halton_cell(config, index; stage="initial")
    units = ntuple(dimension -> radical_inverse(index, config.halton.bases[dimension]),
        length(SEARCH_AXES))
    values = ntuple(dimension -> _axis_value(getproperty(config.axes,
        SEARCH_AXES[dimension]), units[dimension]), length(SEARCH_AXES))
    return (; cell_id="halton_$(lpad(index, 4, '0'))", family="halton", stage,
        plane="", e_to_i_index=0, theta_off_index=0, halton_index=index,
        halton_e_to_e=units[1], halton_i_to_e=units[2], halton_e_to_i=units[3],
        halton_i_to_i=units[4], halton_theta_off=units[5],
        e_to_e=values[1], i_to_e=values[2], e_to_i=values[3], i_to_i=values[4],
        theta_off=values[5])
end

function halton_cells(config, indices; stage="initial")
    return [halton_cell(config, index; stage) for index in indices]
end

function extension_indices(config, has_screen_candidate; smoke=false)
    smoke && return 1:0
    has_screen_candidate && return 1:0
    config.halton.maximum_points <= config.halton.initial_points && return 1:0
    return config.halton.initial_points + 1:config.halton.maximum_points
end

function _plane_values(axis, smoke_values, smoke)
    smoke && return smoke_values
    return collect(range(axis.lower; step=axis.plane_step,
        length=round(Int, (axis.upper - axis.lower) / axis.plane_step) + 1))
end

function plane_cells(config; smoke=false)
    e_values = _plane_values(config.axes.e_to_i, config.smoke_e_to_i, smoke)
    threshold_values = _plane_values(config.axes.theta_off, config.smoke_theta_off, smoke)
    rows = NamedTuple[]
    for plane in config.planes, (e_index, e_to_i) in enumerate(e_values),
        (threshold_index, theta_off) in enumerate(threshold_values)
        push!(rows, (; cell_id="plane_$(plane.name)_e$(lpad(e_index, 2, '0'))_t$(lpad(threshold_index, 2, '0'))",
            family="plane", stage="plane", plane=plane.name, e_to_i_index=e_index,
            theta_off_index=threshold_index, halton_index=0,
            halton_e_to_e=NaN, halton_i_to_e=NaN, halton_e_to_i=NaN,
            halton_i_to_i=NaN, halton_theta_off=NaN,
            e_to_e=plane.e_to_e, i_to_e=plane.i_to_e, e_to_i,
            i_to_i=plane.i_to_i, theta_off))
    end
    return rows
end

function models_for_cell(config, cell)
    coupling = PointCoupling(e_to_e=cell.e_to_e, i_to_e=cell.i_to_e,
        e_to_i=cell.e_to_i, i_to_i=cell.i_to_i)
    return matched_point_models(excitatory=config.excitatory,
        inhibitory_control=config.inhibitory_control, failure_threshold=cell.theta_off,
        coupling=coupling)
end

function search_key(cell, condition)
    base = (condition, cell.e_to_e, cell.i_to_e, cell.e_to_i, cell.i_to_i)
    return condition == :control ? base : (base..., cell.theta_off)
end

function _append_rows(path, rows)
    isempty(rows) || CSV.write(path, rows; append=true)
    return nothing
end

function _initialize_artifacts(output)
    for (name, columns) in (
        ("samples.csv", SAMPLE_COLUMNS), ("searches.csv", SEARCH_COLUMNS),
        ("attempts.csv", ATTEMPT_COLUMNS), ("equilibria.csv", EQUILIBRIUM_COLUMNS),
        ("root_matches.csv", ROOT_MATCH_COLUMNS), ("candidates.csv", CANDIDATE_COLUMNS),
        ("branches.csv", BRANCH_COLUMNS),
        ("continuation_points.csv", CONTINUATION_POINT_COLUMNS),
        ("continuation_attempts.csv", CONTINUATION_ATTEMPT_COLUMNS),
        ("continuation_candidates.csv", CONTINUATION_CANDIDATE_COLUMNS),
    )
        Evidence.write_rows(joinpath(output, name), NamedTuple[], columns)
    end
end

function _search_summary(search_id, cell, condition, grid_points, result)
    found = isnothing(result) ? [] : result.equilibria
    count_stability(classification) = isnothing(result) ? -1 :
        count(eq -> eq.stability.classification == classification, found)
    return (; search_id, cell_id=cell.cell_id, stage=cell.stage,
        condition=string(condition), grid_points,
        status=isnothing(result) ? "execution_failed" : "completed",
        discovered_equilibria=isnothing(result) ? -1 : length(found),
        attracting_equilibria=count_stability(Attracting),
        saddle_equilibria=count_stability(Saddle),
        repelling_equilibria=count_stability(Repelling),
        stability_unresolved=count_stability(StabilityUnresolved),
        near_singular_equilibria=isnothing(result) ? -1 : count(eq -> eq.near_singular, found),
        unresolved_nearby_groups=isnothing(result) ? -1 : length(result.unresolved_nearby),
        attempts=isnothing(result) ? -1 : length(result.attempts),
        rejected_attempts=isnothing(result) ? -1 :
            count(attempt -> attempt.validation == RejectedCandidate, result.attempts),
        completeness=isnothing(result) ? "not_available" : string(result.completeness))
end

function _equilibrium_rows(search_id, condition, grid_points, model, result)
    isnothing(result) && return NamedTuple[]
    return [begin
        summary = Evidence.summary_equilibrium(search_id, index, equilibrium)
        observed = Map.equilibrium_observations(model, equilibrium)
        (; search_id, equilibrium=index, condition=string(condition), grid_points,
            E=summary.E, I=summary.I, residual_norm=summary.residual_norm,
            near_singular=summary.near_singular,
            representative_attempt=summary.representative_attempt,
            member_attempts=summary.member_attempts, stability=summary.stability,
            geometry=summary.geometry, eigenvalue_1_real=summary.eigenvalue_1_real,
            eigenvalue_1_imaginary=summary.eigenvalue_1_imaginary,
            eigenvalue_2_real=summary.eigenvalue_2_real,
            eigenvalue_2_imaginary=summary.eigenvalue_2_imaginary,
            spectral_abscissa=summary.spectral_abscissa, u_I=observed.u_I,
            F_I=observed.F_I, F_I_prime=observed.F_I_prime,
            inhibitory_response_midpoint=observed.inhibitory_response_midpoint,
            inhibitory_branch=observed.inhibitory_branch,
            descending_response_branch=observed.descending_response_branch)
    end for (index, equilibrium) in enumerate(result.equilibria)]
end

function _run_cached_search!(state, config, cell, condition, output)
    key = search_key(cell, condition)
    cacheable = condition == :control && cell.family == "plane"
    if cacheable && haskey(state.cache, key)
        return state.cache[key]
    end
    state.search_counter[] += 1
    search_id = "search_$(lpad(state.search_counter[], 6, '0'))"
    model = getproperty(models_for_cell(config, cell), condition)
    seeds = Map.deterministic_seeds(model, config.screen_grid_points)
    result = try
        find_equilibria(model; seeds, options=config.equilibrium_options,
            stability_options=config.stability_options)
    catch error
        error isa InterruptException && rethrow()
        Evidence.write_toml(joinpath(output, "contexts", "searches", search_id * ".toml"),
            Dict("status" => "execution_failed", "error" => Evidence.error_record(error),
                "requested_seeds" => seeds, "model" => model))
        nothing
    end
    if !isnothing(result)
        Evidence.write_toml(joinpath(output, "contexts", "searches", search_id * ".toml"),
            merge(Dict("search_id" => search_id, "cell_id" => cell.cell_id,
                "condition" => string(condition), "grid_points" => config.screen_grid_points,
                "model" => model, "requested_seeds" => seeds),
                Evidence.context_record(result)))
    end
    isnothing(result) && push!(state.failures, search_id)
    if !isnothing(result)
        _append_rows(joinpath(output, "attempts.csv"),
            [merge((; search_id), Base.structdiff(Evidence.summary_attempt(search_id, index, attempt),
                NamedTuple{(:context_id,)})) for (index, attempt) in enumerate(result.attempts)])
        _append_rows(joinpath(output, "equilibria.csv"),
            _equilibrium_rows(search_id, condition, config.screen_grid_points, model, result))
    end
    item = (; search_id, model, result)
    cacheable && (state.cache[key] = item)
    return item
end
