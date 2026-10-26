"""Deterministic finite search for four locally attracting FoI equilibria."""
module TetrastabilityExperiment

using FailureOfInhibition2025
import CSV
import TOML

include("coexistence_core.jl")
const Map = CoexistenceExperiment
const Evidence = Map.MinimalExperiment

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const SEARCH_AXES = (:e_to_e, :i_to_e, :e_to_i, :i_to_i, :theta_off)
const HALTON_BASES = (2, 3, 5, 7, 11)
const CANDIDATE_MINIMUM = 4
const CONDITIONS = (:control, :failure_of_inhibition)

const SAMPLE_COLUMNS = (
    :cell_id, :family, :stage, :plane, :e_to_i_index, :theta_off_index,
    :halton_index, :halton_e_to_e, :halton_i_to_e, :halton_e_to_i,
    :halton_i_to_i, :halton_theta_off, :e_to_e, :i_to_e, :e_to_i,
    :i_to_i, :theta_off,
)
const SEARCH_COLUMNS = (
    :search_id, :cell_id, :stage, :condition, :grid_points, :status,
    :discovered_equilibria, :attracting_equilibria, :saddle_equilibria,
    :repelling_equilibria, :stability_unresolved, :near_singular_equilibria,
    :unresolved_nearby_groups, :attempts, :rejected_attempts, :completeness,
)
const ATTEMPT_COLUMNS = (
    :search_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I,
    :solver_status, :solver_success, :residual_norm, :validation,
    :near_singular, :reasons,
)
const EQUILIBRIUM_COLUMNS = (
    :search_id, :equilibrium, :condition, :grid_points, :E, :I,
    :residual_norm, :near_singular, :representative_attempt, :member_attempts,
    :stability, :geometry, :eigenvalue_1_real, :eigenvalue_1_imaginary,
    :eigenvalue_2_real, :eigenvalue_2_imaginary, :spectral_abscissa,
    :u_I, :F_I, :F_I_prime, :inhibitory_response_midpoint,
    :inhibitory_branch, :descending_response_branch,
)
const ROOT_MATCH_COLUMNS = (
    :candidate_id, :root_track, :grid_points, :search_id, :equilibrium,
    :match_distance, :E, :I, :attracting, :near_singular, :residual_norm,
    :recomputed_residual_norm, :balance_jacobian_delta, :ode_jacobian_delta,
    :spectral_abscissa, :quality_pass,
)
const CANDIDATE_COLUMNS = (
    :candidate_id, :cell_id, :e_to_e, :i_to_e, :e_to_i, :i_to_i,
    :theta_off, :discovery_attracting, :grid_21_attracting,
    :grid_41_attracting, :matched_attracting_roots, :minimum_separation,
    :worst_spectral_abscissa, :maximum_recomputed_residual,
    :maximum_balance_jacobian_delta, :maximum_ode_jacobian_delta,
    :confirmation_status, :reasons,
)
const BRANCH_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :points, :attempts,
    :screening_candidates, :termination,
)
const CONTINUATION_POINT_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :point, :parameter,
    :E, :I, :residual_norm, :tangent_parameter, :stability,
    :spectral_abscissa, :trace, :determinant,
)
const CONTINUATION_ATTEMPT_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :attempt, :step,
    :iterations, :arclength_residual, :status, :accepted,
)
const CONTINUATION_CANDIDATE_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :kind,
    :first_point, :second_point, :lower, :upper,
)

include("tetrastability_config.jl")

include("tetrastability_sampling.jl")

include("tetrastability_confirmation.jl")

include("tetrastability_continuation.jl")

include("tetrastability_artifacts.jl")

function run_experiment(config_path::AbstractString, output_dir::AbstractString;
    smoke=false, require_canonical=true)
    config = load_config(config_path; require_canonical)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    mkpath(joinpath(output, "contexts", "searches"))
    mkpath(joinpath(output, "contexts", "confirmations"))
    mkpath(joinpath(output, "contexts", "continuations"))
    _initialize_artifacts(output)
    metadata = _archive_provenance(config_path, output, config, smoke)
    state = _state()

    screen_cells!(state, config, plane_cells(config; smoke), output)
    initial_count = smoke ? config.smoke_halton_points : config.halton.initial_points
    screen_cells!(state, config, halton_cells(config, 1:initial_count; stage="initial"), output)
    extension_run = false
    extension = extension_indices(config, !isempty(state.discovery_candidates); smoke)
    if !isempty(extension)
        extension_run = true
        screen_cells!(state, config, halton_cells(config, extension; stage="extension"), output)
    end

    confirmed_count = 0
    for cell_id in sort!(collect(keys(state.discovery_candidates)))
        confirmation = confirm_candidate!(state, config,
            state.discovery_candidates[cell_id], output)
        confirmation.confirmed && (confirmed_count += 1)
        continue_candidate!(state, config, confirmation, output; smoke)
        delete!(state.candidate_results, cell_id)
    end
    metadata["execution_success"] = isempty(state.failures)
    metadata["failed_contexts"] = state.failures
    metadata["extension_run"] = extension_run
    metadata["extension_trigger"] = extension_run ?
        "no FoI screen-positive candidate in planes or Halton indices 1:$(config.halton.initial_points)" :
        "not triggered"
    metadata["last_halton_index"] = smoke ? initial_count :
        extension_run ? config.halton.maximum_points : config.halton.initial_points
    metadata["parameter_cells"] = length(state.cells)
    unique_searches = state.search_counter[] + state.confirmation_search_counter[]
    metadata["unique_search_contexts"] = unique_searches
    metadata["screen_positive_cells"] = length(state.discovery_candidates)
    metadata["confirmed_cells"] = confirmed_count
    metadata["scientific_outcome"] = confirmed_count > 0 ?
        "at least one sampled cell has at least four confirmed discovered locally attracting FoI equilibria" :
        "no sampled cell met the confirmation rule; this finite result is not an absence proof"
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success=isempty(state.failures), extension_run,
        parameter_cells=length(state.cells), unique_searches,
        screen_positive=length(state.discovery_candidates), confirmed=confirmed_count)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "tetrastability.toml")
    output = nothing
    smoke = false
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--smoke"
            smoke = true
            index += 1
            continue
        elseif option in ("--config", "--output")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            value = args[index + 1]
            startswith(value, "--") && throw(ArgumentError("$option requires a value"))
            option == "--config" ? (config_path = value) : (output = value)
            index += 2
            continue
        end
        throw(ArgumentError("unknown option: $option"))
    end
    isnothing(output) && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output; smoke)
    println("Tetrastability search: $(result.parameter_cells) cells, " *
        "$(result.screen_positive) screen-positive, $(result.confirmed) confirmed; " *
        "execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end
