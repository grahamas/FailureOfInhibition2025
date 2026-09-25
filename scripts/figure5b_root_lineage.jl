module Figure5bRootLineage

using FailureOfInhibition2025
using LinearAlgebra: eigvals, norm, svdvals

const GRID_POINTS = (11, 21, 41)
const State = NTuple{2,Float64}
const TrackStates = NTuple{3,State}

"""Numerical policy for root identity; step and predictor bounds belong to each transition."""
struct RootLineageOptions
    coordinate_atol::Float64
    minimum_root_separation::Float64
    residual_atol::Float64
    jacobian_atol::Float64
    spectral_margin::Float64

    function RootLineageOptions(coordinate_atol::Float64,
        minimum_root_separation::Float64, residual_atol::Float64,
        jacobian_atol::Float64, spectral_margin::Float64)
        values = (coordinate_atol, minimum_root_separation, residual_atol,
            jacobian_atol, spectral_margin)
        all(value -> isfinite(value) && value > 0, values) ||
            throw(ArgumentError("lineage tolerances must be finite and positive"))
        coordinate_atol < minimum_root_separation / 2 ||
            throw(ArgumentError("coordinate_atol must be less than half minimum_root_separation"))
        return new(values...)
    end
end

function RootLineageOptions(; coordinate_atol=1e-6,
    minimum_root_separation=1e-5, residual_atol=1e-9,
    jacobian_atol=1e-9, spectral_margin=1e-8)
    values = (coordinate_atol, minimum_root_separation, residual_atol,
        jacobian_atol, spectral_margin)
    all(value -> value isa Real && !(value isa Bool) && isfinite(value) &&
        value > 0, values) || throw(ArgumentError("lineage tolerances must be finite positive real numbers"))
    converted = Float64.(values)
    all(value -> isfinite(value) && value > 0, converted) ||
        throw(ArgumentError("lineage tolerances must remain positive in Float64"))
    return RootLineageOptions(converted...)
end

"""One root matched uniquely across the three deterministic grid searches."""
struct RootTrack
    states::TrackStates
    classifications::NTuple{3,StabilityClassification}
    residuals::NTuple{3,Float64}
    balance_jacobian_errors::NTuple{3,Float64}
    ode_jacobian_errors::NTuple{3,Float64}
    minimum_singular_values::NTuple{3,Float64}
end

"""Immutable identity token retaining every source root for reciprocal matching."""
struct RootLineageAnchor
    origin_model_identity::Tuple
    model_identity::Tuple
    states::TrackStates
    source_tracks::Tuple{Vararg{TrackStates}}
    followed_source_index::Int
end

"""Audit of a three-grid root search; `qualified` never implies completeness."""
struct RootTrackEvidence
    qualified::Bool
    reasons::Tuple{Vararg{Symbol}}
    model_identity::Union{Nothing,Tuple}
    tracks::Tuple{Vararg{RootTrack}}
    root_counts::NTuple{3,Int}
    unresolved_counts::NTuple{3,Int}
    minimum_separation::Float64
end

"""Audit of one transition. Only `accepted=true` carries a new anchor."""
struct RootTransitionEvidence
    accepted::Bool
    reasons::Tuple{Vararg{Symbol}}
    source_anchor::Union{Nothing,RootLineageAnchor}
    anchor::Union{Nothing,RootLineageAnchor}
    tracks::RootTrackEvidence
    corrected_state::State
    predicted_state::Union{Nothing,State}
    prior_distances::Tuple{Vararg{Float64}}
    reciprocal_distances::Tuple{Vararg{Float64}}
    predictor_distances::Tuple{Vararg{Float64}}
    corrected_distances::Tuple{Vararg{Float64}}
    source_isolation::Float64
    destination_isolation::Float64
    destination_source_matches::Tuple{Vararg{Union{Nothing,Int}}}
    prior_match::Union{Nothing,Int}
    reciprocal_match::Union{Nothing,Int}
    predictor_match::Union{Nothing,Int}
    corrected_match::Union{Nothing,Int}
end

function _state(value)
    value isa AbstractVector || value isa Tuple ||
        throw(ArgumentError("state must be an [E, I] vector or tuple"))
    length(value) == 2 || throw(ArgumentError("state must contain E and I"))
    all(x -> x isa Real && !(x isa Bool) && isfinite(x), value) ||
        throw(ArgumentError("state must be finite and real"))
    state = (Float64(value[1]), Float64(value[2]))
    all(isfinite, state) || throw(ArgumentError("state must remain finite in Float64"))
    return state
end

_distance(left::State, right::State) = hypot(left[1] - right[1], left[2] - right[2])

_discovery_distance(left::State, right::State) =
    max(abs(left[1] - right[1]), abs(left[2] - right[2]))

_track_distance(left::TrackStates, right::TrackStates) =
    maximum(_distance(left[j], right[j]) for j in 1:3)

_track_separation(left::TrackStates, right::TrackStates) =
    minimum(_distance(left[j], right[j]) for j in 1:3)

function _constellation_separation(tracks)
    length(tracks) < 2 && return Inf
    return minimum(_track_separation(tracks[i], tracks[j])
        for i in 1:(length(tracks) - 1) for j in (i + 1):length(tracks))
end

function _model_identity(search)
    model = search.frozen_model
    source = search.model
    return (typeof(source.excitatory.response), typeof(source.inhibitory.response),
        FailureOfInhibition2025._model_numeric_values(source),
        typeof(source.drive), repr(source.drive),
        typeof(model.excitatory.response), typeof(model.inhibitory.response),
        FailureOfInhibition2025._model_numeric_values(model),
        typeof(model.drive), repr(model.drive),
        Tuple(search.frozen_drive), search.source_time)
end

function _point_model_identity(model)
    return (typeof(model.excitatory.response),
        typeof(model.inhibitory.response),
        FailureOfInhibition2025._model_numeric_values(model),
        typeof(model.drive), repr(model.drive))
end

function _frozen_context_valid(search)
    context = try
        FailureOfInhibition2025._frozen_point_context(
            search.model, search.source_time)
    catch error
        error isa InterruptException && rethrow()
        return false
    end
    return isequal(_point_model_identity(context.frozen_model),
        _point_model_identity(search.frozen_model)) &&
        isequal(Tuple(context.frozen_drive), Tuple(search.frozen_drive)) &&
        isequal(context.source_time, search.source_time)
end

function _valid_search_policy(search)
    search.options isa EquilibriumOptions &&
        search.stability_options isa StabilityOptions || return false
    equilibrium = search.options
    stability = search.stability_options
    values = (equilibrium.solver_abstol, equilibrium.solver_reltol,
        equilibrium.residual_atol, equilibrium.domain_atol,
        equilibrium.dedup_atol, equilibrium.singular_atol,
        equilibrium.singular_rtol, stability.spectral_atol,
        stability.spectral_rtol)
    return all(value -> isfinite(value) && value >= 0, values) &&
        equilibrium.maxiters > 0
end

function _minimum_separation(roots)
    length(roots) < 2 && return Inf
    return minimum(_distance(_state(roots[i].state), _state(roots[j].state))
        for i in 1:(length(roots) - 1) for j in (i + 1):length(roots))
end

function _unique_mapping(reference, candidate, tolerance)
    length(reference) == length(candidate) || return nothing
    mapping = Int[]
    for root in reference
        matches = findall(other -> _distance(_state(root.state),
            _state(other.state)) <= tolerance, candidate)
        length(matches) == 1 || return nothing
        push!(mapping, only(matches))
    end
    length(unique(mapping)) == length(candidate) || return nothing
    return mapping
end

function _attempts_valid(search, root)
    members = root.member_attempts
    n = length(search.attempts)
    isempty(members) && return false
    length(unique(members)) == length(members) || return false
    all(index -> 1 <= index <= n, members) || return false
    root.representative_attempt in members || return false
    representative = search.attempts[root.representative_attempt]
    isequal(representative.candidate, root.state) || return false
    isequal(representative.balance_residual, root.balance_residual) || return false
    isequal(representative.balance_jacobian, root.balance_jacobian) || return false
    representative.near_singular == root.near_singular || return false
    for index in members
        attempt = search.attempts[index]
        attempt.validation == AdmissibleCandidate || return false
        length(attempt.candidate) == 2 &&
            length(attempt.balance_residual) == 2 || return false
        all(isfinite, attempt.candidate) &&
            all(isfinite, attempt.balance_residual) || return false
        _discovery_distance(_state(attempt.candidate), _state(root.state)) <=
            search.options.dedup_atol || return false
        residual = zeros(Float64, 2)
        point_balance!(residual, attempt.candidate, search.frozen_model, 0.0)
        all(isfinite, residual) || return false
        residual_norm = maximum(abs, residual)
        residual_norm <= search.options.residual_atol &&
            isfinite(attempt.residual_norm) &&
            abs(residual_norm - attempt.residual_norm) <=
                search.options.residual_atol &&
            maximum(abs, residual .- attempt.balance_residual) <=
                search.options.residual_atol || return false
    end
    return all(_discovery_distance(_state(search.attempts[left].candidate),
        _state(search.attempts[right].candidate)) <= search.options.dedup_atol
        for left in members for right in members)
end

function _stored_stability_valid(root, recomputed, options)
    stored = root.stability
    length(stored.eigenvalues) == 2 && length(stored.thresholds) == 2 ||
        return false
    all(isfinite, stored.eigenvalues) && all(isfinite, stored.thresholds) &&
        all(isfinite, stored.jacobian) &&
        all(isfinite, (stored.trace, stored.determinant,
            stored.spectral_abscissa)) || return false
    tolerance = options.jacobian_atol
    return maximum(abs, stored.eigenvalues .- recomputed.eigenvalues) <=
            tolerance &&
        maximum(abs, stored.thresholds .- recomputed.thresholds) <=
            tolerance &&
        abs(stored.trace - recomputed.trace) <= tolerance &&
        abs(stored.determinant - recomputed.determinant) <= tolerance &&
        abs(stored.spectral_abscissa - recomputed.spectral_abscissa) <=
            tolerance &&
        stored.classification == recomputed.classification &&
        stored.geometry == recomputed.geometry
end

function _quality(search, root, options)
    state = _state(root.state)
    if length(root.balance_residual) != 2 ||
            size(root.balance_jacobian) != (2, 2) ||
            size(root.stability.jacobian) != (2, 2)
        return (; qualified=false, residual_norm=Inf, balance_error=Inf,
            ode_error=Inf, minimum_singular=NaN)
    end
    residual = zeros(Float64, 2)
    balance = zeros(Float64, 2, 2)
    ode = zeros(Float64, 2, 2)
    point_balance!(residual, collect(state), search.frozen_model, 0.0)
    point_balance_jacobian!(balance, collect(state), search.frozen_model, 0.0)
    point_jacobian!(ode, collect(state), search.frozen_model, 0.0)
    residual_norm = maximum(abs, residual)
    balance_error = maximum(abs, balance .- root.balance_jacobian)
    ode_error = maximum(abs, ode .- root.stability.jacobian)
    if !all(isfinite, residual) || !all(isfinite, balance) ||
            !all(isfinite, ode) || !all(isfinite, root.balance_jacobian) ||
            !all(isfinite, root.stability.jacobian) ||
            !all(isfinite, root.balance_residual)
        return (; qualified=false, residual_norm, balance_error, ode_error,
            minimum_singular=NaN)
    end
    singular_values = svdvals(balance)
    minimum_singular = minimum(singular_values)
    singular_threshold = search.options.singular_atol +
        search.options.singular_rtol * maximum(singular_values)
    if !all(isfinite, singular_values)
        return (; qualified=false, residual_norm, balance_error, ode_error,
            minimum_singular)
    end
    recomputed = classify_local_stability(ode;
        options=search.stability_options)
    classification = recomputed.classification
    spectral = real.(eigvals(ode))
    spectral_ok = if classification == Attracting
        all(value -> value < -options.spectral_margin, spectral)
    elseif classification == Repelling
        all(value -> value > options.spectral_margin, spectral)
    elseif classification == Saddle
        minimum(spectral) < -options.spectral_margin &&
            maximum(spectral) > options.spectral_margin
    else
        # A trace-zero Hopf candidate may be unresolved, but a permissive
        # classifier must not admit roots far from the neutral spectrum.
        classification == StabilityUnresolved &&
            all(value -> abs(value) <= options.spectral_margin, spectral)
    end
    qualified = all(isfinite, residual) && all(isfinite, balance) &&
        all(isfinite, ode) && residual_norm <= options.residual_atol &&
        maximum(abs, root.balance_residual) <= options.residual_atol &&
        maximum(abs, residual .- root.balance_residual) <=
            options.residual_atol &&
        balance_error <= options.jacobian_atol &&
        ode_error <= options.jacobian_atol && !root.near_singular &&
        isfinite(minimum_singular) && minimum_singular > singular_threshold &&
        _stored_stability_valid(root, recomputed, options) &&
        spectral_ok
    return (; qualified, residual_norm, balance_error, ode_error,
        minimum_singular)
end

"""
    build_root_tracks(searches; options=RootLineageOptions())

Validate a single model context and the exact 11/21/41 seed schedules, then
match every discovered root bijectively across grids. Root counts may vary
between parameter points, but must agree within this refinement set.
"""
function build_root_tracks(searches; options=RootLineageOptions())
    options isa RootLineageOptions ||
        throw(ArgumentError("options must be RootLineageOptions"))
    searches isa Tuple && length(searches) == 3 &&
        all(search -> search isa EquilibriumSearchResult, searches) ||
        throw(ArgumentError("searches must be three EquilibriumSearchResult refinements"))
    root_counts = Tuple(length(search.equilibria) for search in searches)
    unresolved_counts = Tuple(length(search.unresolved_nearby) for search in searches)
    reasons = Symbol[]
    identities = Tuple(_model_identity(search) for search in searches)
    all(==(identities[1]), identities) || push!(reasons, :model_context_mismatch)
    all(_frozen_context_valid, searches) ||
        push!(reasons, :frozen_context_mismatch)
    all(search -> isequal(search.options, searches[1].options) &&
        isequal(search.stability_options, searches[1].stability_options),
        searches) || push!(reasons, :numerical_policy_mismatch)
    policy_valid = all(_valid_search_policy, searches)
    policy_valid || push!(reasons, :invalid_numerical_policy)
    all(FailureOfInhibition2025._matches_refinement_schedule(search, grid)
        for (search, grid) in zip(searches, GRID_POINTS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(iszero, unresolved_counts) || push!(reasons, :unresolved_nearby_roots)
    all(==(root_counts[1]), root_counts) && root_counts[1] > 0 ||
        push!(reasons, :root_count_mismatch)
    separation = minimum(_minimum_separation(search.equilibria) for search in searches)
    separation >= options.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)
    policy_valid || return RootTrackEvidence(false, Tuple(reasons),
        identities[1], (), root_counts, unresolved_counts, separation)
    claimed_attempts = Int[]
    for search in searches
        empty!(claimed_attempts)
        for root in search.equilibria
            _attempts_valid(search, root) ||
                push!(reasons, :root_attempt_failure)
            append!(claimed_attempts, root.member_attempts)
        end
        length(unique(claimed_attempts)) == length(claimed_attempts) ||
            push!(reasons, :root_attempt_overlap)
    end
    tracks = RootTrack[]
    if !(:root_count_mismatch in reasons)
        reference = sort(copy(searches[1].equilibria);
            by=root -> Tuple(root.state))
        second = _unique_mapping(reference, searches[2].equilibria,
            options.coordinate_atol)
        third = _unique_mapping(reference, searches[3].equilibria,
            options.coordinate_atol)
        if second === nothing || third === nothing
            push!(reasons, :ambiguous_refinement_match)
        else
            for i in eachindex(reference)
                roots = (reference[i], searches[2].equilibria[second[i]],
                    searches[3].equilibria[third[i]])
                qualities = Tuple(_quality(searches[j], roots[j], options)
                    for j in 1:3)
                all(quality -> quality.qualified, qualities) ||
                    push!(reasons, :root_quality_failure)
                classifications = Tuple(root.stability.classification for root in roots)
                all(==(classifications[1]), classifications) ||
                    push!(reasons, :classification_flip)
                push!(tracks, RootTrack(Tuple(_state(root.state) for root in roots),
                    classifications,
                    Tuple(quality.residual_norm for quality in qualities),
                    Tuple(quality.balance_error for quality in qualities),
                    Tuple(quality.ode_error for quality in qualities),
                    Tuple(quality.minimum_singular for quality in qualities)))
            end
        end
    end
    unique!(reasons)
    return RootTrackEvidence(isempty(reasons), Tuple(reasons), identities[1],
        Tuple(tracks), root_counts, unresolved_counts, separation)
end

function _match(distances, bound, uncertainty, lost, ambiguous, reasons)
    if isempty(distances) || minimum(distances) > bound
        push!(reasons, lost)
        return nothing
    end
    nearest = argmin(distances)
    sorted = sort(distances)
    length(sorted) > 1 && sorted[2] - sorted[1] <= 2uncertainty &&
        push!(reasons, ambiguous)
    return nearest
end

function _constellation_assignment(source_tracks, target_states, displacement,
    uncertainty, reasons)
    matches = Union{Nothing,Int}[]
    source_count = length(source_tracks)
    target_count = length(target_states)
    for target in target_states
        distances = [_track_distance(source, target) for source in source_tracks]
        if source_count < target_count && minimum(distances) > displacement
            # Only a net increase permits a destination without a source.
            push!(matches, nothing)
        else
            push!(matches, _match(distances, displacement, uncertainty,
                :destination_source_lost, :destination_source_ambiguous, reasons))
        end
    end
    matched = filter(!isnothing, matches)
    required = min(source_count, target_count)
    if length(matched) != required || length(unique(matched)) != required
        push!(reasons, source_count < target_count ?
            :source_destination_assignment_unresolved :
            :destination_source_assignment_unresolved)
    end
    return matches
end

"""Seed a lineage from an independently validated three-grid root search."""
function seed_lineage(searches, corrected_state; options=RootLineageOptions())
    tracks = build_root_tracks(searches; options)
    corrected = _state(corrected_state)
    reasons = Symbol[tracks.reasons...]
    distances = [maximum(_distance(state, corrected) for state in track.states)
        for track in tracks.tracks]
    match = tracks.qualified ? _match(distances, options.coordinate_atol,
        options.coordinate_atol,
        :corrected_root_lost, :corrected_root_ambiguous, reasons) : nothing
    anchor = isempty(reasons) ? RootLineageAnchor(tracks.model_identity,
        tracks.model_identity, tracks.tracks[match].states,
        Tuple(track.states for track in tracks.tracks), match) : nothing
    return RootTransitionEvidence(anchor !== nothing, Tuple(reasons), nothing,
        anchor, tracks, corrected, nothing, (), (), (), Tuple(distances),
        Inf, tracks.minimum_separation, (), nothing, nothing, nothing, match)
end

"""
    transition_lineage(anchor, searches, corrected_state; ...)

Accept a new root only if its target track is the unique nearest match to the
followed source and that target's unique nearest source is the same root.
The optional predictor and corrected Newton root must identify that track.
The displacement bound must be strictly below half the local source and
destination root separation after coordinate uncertainty. This is
conservative numerical tracking, not proof of global root identity.
"""
function transition_lineage(anchor::RootLineageAnchor, searches, corrected_state;
    options=RootLineageOptions(), displacement_atol,
    predicted_state=nothing, predictor_atol=nothing)
    displacement_atol isa Real && !(displacement_atol isa Bool) &&
        isfinite(displacement_atol) && displacement_atol > 0 ||
        throw(ArgumentError("displacement_atol must be finite and positive"))
    displacement = Float64(displacement_atol)
    isfinite(displacement) && displacement > 0 || throw(ArgumentError(
        "displacement_atol must remain finite and positive in Float64"))
    if predicted_state !== nothing
        predictor_atol isa Real && !(predictor_atol isa Bool) &&
            isfinite(predictor_atol) && predictor_atol > 0 ||
            throw(ArgumentError("predictor_atol must be finite and positive"))
    end
    predictor = predictor_atol === nothing ? nothing : Float64(predictor_atol)
    predictor !== nothing && !(isfinite(predictor) && predictor > 0) &&
        throw(ArgumentError("predictor_atol must remain finite and positive in Float64"))
    corrected = _state(corrected_state)
    predicted = predicted_state === nothing ? nothing : _state(predicted_state)
    tracks = build_root_tracks(searches; options)
    reasons = Symbol[tracks.reasons...]
    valid_source = 1 <= anchor.followed_source_index <= length(anchor.source_tracks) &&
        anchor.states == anchor.source_tracks[anchor.followed_source_index]
    valid_source || push!(reasons, :invalid_source_anchor)
    target_states = Tuple(track.states for track in tracks.tracks)
    source_isolation = valid_source ? minimum((_track_separation(anchor.states,
        source) for (index, source) in enumerate(anchor.source_tracks)
        if index != anchor.followed_source_index); init=Inf) : NaN
    destination_isolation = _constellation_separation(target_states)
    2(displacement + options.coordinate_atol) < source_isolation ||
        push!(reasons, :source_isolation_unresolved)
    2(displacement + options.coordinate_atol) < destination_isolation ||
        push!(reasons, :destination_isolation_unresolved)
    prior_distances = [_track_distance(anchor.states, target)
        for target in target_states]
    reciprocal_distances = Float64[]
    predictor_distances = predicted === nothing ? Float64[] :
        [maximum(_distance(predicted, state) for state in track.states)
            for track in tracks.tracks]
    corrected_distances = [maximum(_distance(corrected, state)
        for state in track.states) for track in tracks.tracks]
    destination_source_matches = Union{Nothing,Int}[]
    prior_match = reciprocal_match = predictor_match = corrected_match = nothing
    if tracks.qualified && valid_source
        prior_match = _match(prior_distances, displacement,
            options.coordinate_atol,
            :prior_root_lost, :prior_root_ambiguous, reasons)
        if prior_match !== nothing
            target = tracks.tracks[prior_match].states
            reciprocal_distances = [maximum(_distance(source[j], target[j])
                for j in 1:3) for source in anchor.source_tracks]
            reciprocal_match = _match(reciprocal_distances, displacement,
                options.coordinate_atol, :reciprocal_root_lost,
                :reciprocal_root_ambiguous, reasons)
            reciprocal_match !== nothing &&
                reciprocal_match != anchor.followed_source_index &&
                push!(reasons, :tracked_root_lost)
        end
        destination_source_matches = _constellation_assignment(
            anchor.source_tracks, target_states, displacement,
            options.coordinate_atol, reasons)
        if predicted !== nothing
            predictor_match = _match(predictor_distances,
                predictor, options.coordinate_atol, :predictor_root_lost,
                :predictor_root_ambiguous, reasons)
        end
        corrected_match = _match(corrected_distances, options.coordinate_atol,
            options.coordinate_atol,
            :corrected_root_lost, :corrected_root_ambiguous, reasons)
        prior_match !== nothing && corrected_match !== nothing &&
            prior_match != corrected_match && push!(reasons, :corrected_root_switch)
        prior_match !== nothing && predictor_match !== nothing &&
            prior_match != predictor_match && push!(reasons, :predictor_root_switch)
    end
    unique!(reasons)
    new_anchor = isempty(reasons) ? RootLineageAnchor(
        anchor.origin_model_identity, tracks.model_identity,
        tracks.tracks[prior_match].states,
        Tuple(track.states for track in tracks.tracks), prior_match) : nothing
    return RootTransitionEvidence(new_anchor !== nothing, Tuple(reasons),
        anchor, new_anchor, tracks, corrected, predicted, Tuple(prior_distances),
        Tuple(reciprocal_distances), Tuple(predictor_distances),
        Tuple(corrected_distances), source_isolation, destination_isolation,
        Tuple(destination_source_matches), prior_match, reciprocal_match,
        predictor_match, corrected_match)
end

end # module
