"""Anchor Hopf/orbit protocol functions included by run_anchor_hopf_orbit_protocol.jl."""
function candidate_model(config, ratio)
    ratio = number(ratio, "ratio"; positive=true)
    c = config.candidate
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=c.tau_e,
            response=LogisticResponse(slope=c.excitatory_slope, threshold=c.theta_e)),
        inhibitory=PopulationParameters(timescale=c.tau_e * ratio,
            response=FailureOfInhibitionResponse(slope=c.inhibitory_slope,
                onset_threshold=c.theta_on, failure_threshold=c.theta_off)),
        coupling=PointCoupling(e_to_e=c.e_to_e, i_to_e=c.i_to_e,
            e_to_i=c.e_to_i, i_to_i=c.i_to_i), drive=NoDrive())
end

function refinement_seeds(model, grid_points)
    grid_points = positive_integer(grid_points, "grid_points")
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=grid_points)
        for i in range(0.0, upper_i; length=grid_points)])
    return unique!(seeds)
end

function equilibrium_refinements(config, ratio)
    model = candidate_model(config, ratio)
    return Tuple(find_equilibria(model; seeds=refinement_seeds(model, points),
        options=config.equilibrium_options, stability_options=config.stability_options)
        for points in config.root_grids)
end

function unique_root_mapping(reference, candidate, tolerance)
    length(reference) == length(candidate) || return nothing
    distances = [norm(left.state .- right.state) for left in reference, right in candidate]
    mapping = Int[]
    for row in axes(distances, 1)
        matches = findall(value -> value <= tolerance, @view distances[row, :])
        length(matches) == 1 || return nothing
        push!(mapping, only(matches))
    end
    length(unique(mapping)) == length(candidate) || return nothing
    return mapping
end

function below_hopf_assessment(searches, options)
    reasons = Symbol[]
    all(search -> length(search.equilibria) == 7, searches) ||
        push!(reasons, :root_count_mismatch)
    all(search -> isempty(search.unresolved_nearby), searches) ||
        push!(reasons, :unresolved_nearby_roots)
    for search in searches
        for root in search.equilibria
            root.near_singular && push!(reasons, :near_singular_root)
            maximum(abs, root.balance_residual) <= options.residual_atol ||
                push!(reasons, :root_residual_too_large)
            real_parts = real.(root.stability.eigenvalues)
            if root.stability.classification == Attracting
                all(value -> value < -options.spectral_margin, real_parts) ||
                    push!(reasons, :insufficient_spectral_margin)
            elseif root.stability.classification == Saddle
                minimum(real_parts) < -options.spectral_margin &&
                    maximum(real_parts) > options.spectral_margin ||
                    push!(reasons, :insufficient_spectral_margin)
            end
        end
        if length(search.equilibria) > 1
            separation = minimum(norm(search.equilibria[left].state .-
                search.equilibria[right].state)
                for left in 1:(length(search.equilibria) - 1)
                for right in (left + 1):length(search.equilibria))
            separation >= options.minimum_root_separation ||
                push!(reasons, :insufficient_root_separation)
        end
    end
    mappings = Any[nothing, nothing]
    if all(search -> length(search.equilibria) == 7, searches)
        mappings[1] = unique_root_mapping(searches[1].equilibria,
            searches[2].equilibria, options.coordinate_match_atol)
        mappings[2] = unique_root_mapping(searches[1].equilibria,
            searches[3].equilibria, options.coordinate_match_atol)
        any(isnothing, mappings) && push!(reasons, :ambiguous_root_matching)
        if all(mapping -> !isnothing(mapping), mappings)
            reference = searches[1].equilibria
            for (offset, mapping) in enumerate(mappings)
                candidate = searches[offset + 1].equilibria
                for track in eachindex(reference)
                    reference[track].stability.classification ==
                        candidate[mapping[track]].stability.classification ||
                        push!(reasons, :stability_track_mismatch)
                end
            end
        end
        for search in searches
            counts = (count(root -> root.stability.classification == Attracting,
                search.equilibria), count(root -> root.stability.classification == Saddle,
                search.equilibria))
            counts == (4, 3) || push!(reasons, :stability_pattern_mismatch)
        end
    end
    unique!(reasons)
    return (qualified=isempty(reasons), reasons, mappings)
end
