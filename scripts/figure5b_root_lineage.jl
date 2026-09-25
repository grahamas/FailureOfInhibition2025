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
    converted[1] < converted[2] / 2 || throw(ArgumentError(
        "coordinate_atol must be less than half minimum_root_separation"))
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
        Tuple(search.frozen_drive), search.source_time)
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

function _quality(search, root, options)
    state = _state(root.state)
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
            !all(isfinite, root.stability.jacobian)
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
    classification = classify_local_stability(ode;
        options=search.stability_options).classification
    spectral = real.(eigvals(ode))
    spectral_ok = if classification == Attracting
        all(value -> value < -options.spectral_margin, spectral)
    elseif classification == Repelling
        all(value -> value > options.spectral_margin, spectral)
    elseif classification == Saddle
        minimum(spectral) < -options.spectral_margin &&
            maximum(spectral) > options.spectral_margin
    else
        # A trace-zero Hopf candidate may legitimately be unresolved here.
        classification == StabilityUnresolved
    end
    qualified = all(isfinite, residual) && all(isfinite, balance) &&
        all(isfinite, ode) && residual_norm <= options.residual_atol &&
        balance_error <= options.jacobian_atol &&
        ode_error <= options.jacobian_atol && !root.near_singular &&
        isfinite(minimum_singular) && minimum_singular > singular_threshold &&
        classification == root.stability.classification && spectral_ok
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
    all(search -> isequal(search.options, searches[1].options) &&
        isequal(search.stability_options, searches[1].stability_options),
        searches) || push!(reasons, :numerical_policy_mismatch)
    all(FailureOfInhibition2025._matches_refinement_schedule(search, grid)
        for (search, grid) in zip(searches, GRID_POINTS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(iszero, unresolved_counts) || push!(reasons, :unresolved_nearby_roots)
    all(==(root_counts[1]), root_counts) && root_counts[1] > 0 ||
        push!(reasons, :root_count_mismatch)
    separation = minimum(_minimum_separation(search.equilibria) for search in searches)
    separation >= options.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)
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
        if length(anchor.source_tracks) != length(target_states)
            for target in target_states
                distances = [_track_distance(source, target)
                    for source in anchor.source_tracks]
                push!(destination_source_matches, _match(distances,
                    displacement, options.coordinate_atol,
                    :destination_source_lost, :destination_source_ambiguous,
                    reasons))
            end
            matched = filter(!isnothing, destination_source_matches)
            length(matched) == length(target_states) &&
                length(unique(matched)) == length(target_states) ||
                push!(reasons, :destination_source_assignment_unresolved)
        end
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
