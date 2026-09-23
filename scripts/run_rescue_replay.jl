"""Coordinate-identified, claim-specific replay of the Figure 3 rescue counterexample."""
module RescueReplay

using FailureOfInhibition2025
import TOML
include("run_pulse_experiment.jl")
using .PulseExperiment

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const CANONICAL_CONFIG = joinpath(REPOSITORY_ROOT, "experiments", "rescue_replay.toml")
const BASE_KEYS = ("schema_version", "description", "model", "search", "equilibrium",
    "stability", "diagnostics", "pulses")
const RESCUE_KEYS = ("condition", "target", "source_state", "destination_state",
    "coordinate_atol", "cap_recheck_duration", "recheck_followup_time",
    "recheck_abstol", "recheck_reltol")

function _load_config(path::AbstractString; require_canonical::Bool)
    raw = TOML.parsefile(path)
    PulseExperiment.require_keys(raw, (BASE_KEYS..., "rescue"), "configuration")
    if require_canonical
        canonical = TOML.parsefile(CANONICAL_CONFIG)
        raw == canonical || throw(ArgumentError(
            "claim-specific rescue replay configuration must match experiments/rescue_replay.toml exactly"))
    end
    rescue = raw["rescue"]
    PulseExperiment.require_keys(rescue, RESCUE_KEYS, "rescue")
    base_raw = deepcopy(raw)
    delete!(base_raw, "rescue")
    base = mktempdir() do directory
        base_path = joinpath(directory, "pulse.toml")
        PulseExperiment.write_toml(base_path, base_raw)
        PulseExperiment.load_config(base_path)
    end
    rescue["condition"] == "failure_of_inhibition" ||
        throw(ArgumentError("rescue.condition must be failure_of_inhibition"))
    rescue["target"] == "negative_E" ||
        throw(ArgumentError("rescue.target must be negative_E"))
    base.pulse_options.targets == [:negative_E] ||
        throw(ArgumentError("pulses.targets must contain only negative_E"))
    source_state = collect(PulseExperiment.MinimalExperiment.pair(
        rescue["source_state"], "rescue.source_state"))
    destination_state = collect(PulseExperiment.MinimalExperiment.pair(
        rescue["destination_state"], "rescue.destination_state"))
    all(x -> 0 <= x <= 1, source_state) && all(x -> 0 <= x <= 1, destination_state) ||
        throw(ArgumentError("rescue source and destination states must lie in [0,1]^2"))
    source_state != destination_state ||
        throw(ArgumentError("rescue source and destination coordinates must differ"))
    coordinate_atol = PulseExperiment.finite_number(
        rescue["coordinate_atol"], "rescue.coordinate_atol"; positive=true)
    cap_recheck_duration = PulseExperiment.finite_number(
        rescue["cap_recheck_duration"], "rescue.cap_recheck_duration"; positive=true)
    cap_recheck_duration in base.pulse_options.durations ||
        throw(ArgumentError("rescue.cap_recheck_duration must be a configured pulse duration"))
    recheck_followup_time = PulseExperiment.finite_number(
        rescue["recheck_followup_time"], "rescue.recheck_followup_time"; positive=true)
    recheck_followup_time >= 2base.pulse_options.diagnostic_options.window_duration ||
        throw(ArgumentError("rescue.recheck_followup_time must cover both diagnostic windows"))
    recheck_abstol = PulseExperiment.finite_number(
        rescue["recheck_abstol"], "rescue.recheck_abstol"; positive=true)
    recheck_reltol = PulseExperiment.finite_number(
        rescue["recheck_reltol"], "rescue.recheck_reltol"; positive=true)
    recheck_abstol < base.pulse_options.abstol &&
        recheck_reltol < base.pulse_options.reltol ||
        throw(ArgumentError("rescue recheck tolerances must both be tighter than the primary tolerances"))
    selection = (; condition=:failure_of_inhibition, target=:negative_E,
        source_state, destination_state, coordinate_atol, cap_recheck_duration,
        recheck_followup_time, recheck_abstol, recheck_reltol)
    return merge(base, (; raw, selection))
end

load_config(path::AbstractString) = _load_config(path; require_canonical=true)

function matching_attracting_equilibrium(search, coordinates, atol, label)
    matches = [(index, equilibrium) for (index, equilibrium) in enumerate(search.equilibria)
        if equilibrium.stability.classification == Attracting &&
           maximum(abs.(equilibrium.state .- coordinates)) <= atol]
    length(matches) == 1 || throw(ArgumentError(
        "$label coordinates must match exactly one discovered attracting equilibrium; found $(length(matches))"))
    return only(matches)
end

function recheck_options(options, selection)
    return PulseExperimentOptions(amplitudes=[first(options.amplitudes)],
        durations=[first(options.durations)], targets=[selection.target],
        followup_times=[selection.recheck_followup_time],
        diagnostic_options=options.diagnostic_options, refinement_levels=0,
        duration_refinement_levels=0, retain_trajectories=options.retain_trajectories,
        trajectory_saveat=options.trajectory_saveat, abstol=selection.recheck_abstol,
        reltol=selection.recheck_reltol, domain_atol=options.domain_atol,
        maxiters=options.maxiters)
end

function selected_rechecks(result, source_id, destination_id, selection)
    selected = Dict{Tuple{Float64,Float64},Int}()
    cap_index = findfirst(result.trials) do trial
        trial.duration == selection.cap_recheck_duration &&
            trial.amplitude == maximum(result.options.amplitudes)
    end
    cap_index === nothing && throw(ArgumentError("configured cap recheck trial is missing"))
    cap_trial = result.trials[cap_index]
    selected[(Float64(cap_trial.duration), Float64(cap_trial.amplitude))] = cap_index
    for boundary in result.boundaries
        outcomes = Set(filter(!isnothing, (boundary.lower_outcome, boundary.upper_outcome)))
        outcomes == Set((source_id, destination_id)) || continue
        for index in (boundary.lower_trial, boundary.upper_trial)
            trial = result.trials[index]
            selected[(Float64(trial.duration), Float64(trial.amplitude))] = index
        end
    end
    return sort!(collect(selected); by=first)
end

function duration_observations(result, source_id, destination_id)
    observations = Dict{String,Any}[]
    for duration in sort!(unique(trial.duration for trial in result.trials))
        trials = filter(trial -> trial.duration == duration, result.trials)
        push!(observations, Dict(
            "duration" => duration,
            "destination_observed" => any(trial -> trial.outcome_equilibrium == destination_id, trials),
            "source_observed" => any(trial -> trial.outcome_equilibrium == source_id, trials),
            "resolved_trials" => count(trial -> trial.status == :compatible, trials),
            "unresolved_trials" => count(trial -> trial.status == :unresolved, trials),
            "integration_failed_trials" => count(trial -> trial.status == :integration_failed, trials),
            "maximum_tested_amplitude" => maximum(trial.amplitude for trial in trials),
        ))
    end
    return observations
end

function protocol_change_brackets(observations, rechecks=NamedTuple[])
    brackets = Dict{String,Any}[]
    for (lower, upper) in zip(observations, Iterators.drop(observations, 1))
        lower["destination_observed"] == upper["destination_observed"] && continue
        lower_unresolved = lower["unresolved_trials"]
        upper_unresolved = upper["unresolved_trials"]
        lower_failures = lower["integration_failed_trials"]
        upper_failures = upper["integration_failed_trials"]
        supporting_rechecks = filter(rechecks) do row
            row.duration == lower["duration"] || row.duration == upper["duration"]
        end
        rechecks_agree = all(row -> row.matches_primary, supporting_rechecks)
        eligible = lower_unresolved == 0 && upper_unresolved == 0 &&
            lower_failures == 0 && upper_failures == 0
        eligible &= rechecks_agree
        status = if lower_unresolved != 0 || upper_unresolved != 0 ||
                lower_failures != 0 || upper_failures != 0
            "censored"
        elseif !rechecks_agree
            "recheck_disagreement"
        else
            "fully_resolved"
        end
        push!(brackets, Dict(
            "lower_duration" => lower["duration"],
            "upper_duration" => upper["duration"],
            "lower_destination_observed" => lower["destination_observed"],
            "upper_destination_observed" => upper["destination_observed"],
            "lower_unresolved_trials" => lower_unresolved,
            "upper_unresolved_trials" => upper_unresolved,
            "lower_integration_failed_trials" => lower_failures,
            "upper_integration_failed_trials" => upper_failures,
            "supporting_recheck_count" => length(supporting_rechecks),
            "supporting_rechecks_agree" => rechecks_agree,
            "claim_evidence_eligible" => eligible,
            "evidence_status" => status,
            "interpretation" => "finite sampled protocol-change bracket; not an exact minimum duration",
        ))
    end
    return brackets
end

function write_primary_rows(rows, output)
    PulseExperiment.write_rows(joinpath(output, "trials.csv"), rows.trials,
        (:condition, :trial_id, :initial_id, :initial_E, :initial_I, :initial_provenance,
         :target, :amplitude, :duration, :duration_over_tau_e, :duration_over_tau_i,
         :integrated_E, :integrated_I, :absolute_input_cost, :status,
         :outcome_equilibrium, :final_followup_time, :attempts))
    PulseExperiment.write_rows(joinpath(output, "followups.csv"), rows.followups,
        (:condition, :trial_id, :attempt, :followup_time, :status, :solver_status,
         :outcome_equilibrium, :diagnostic_classification, :integration_success,
         :reasons, :error))
    PulseExperiment.write_rows(joinpath(output, "tails.csv"), rows.tails,
        (:condition, :trial_id, :attempt, :window, :lower_time, :upper_time, :samples,
         :E_mean, :E_minimum, :E_maximum, :I_mean, :I_minimum, :I_maximum,
         :u_I_mean, :u_I_minimum, :u_I_maximum, :F_I_prime_mean,
         :F_I_prime_minimum, :F_I_prime_maximum))
    PulseExperiment.write_rows(joinpath(output, "boundaries.csv"), rows.boundaries,
        (:condition, :initial_id, :target, :duration, :duration_over_tau_e,
         :duration_over_tau_i, :lower_amplitude, :upper_amplitude, :lower_trial,
         :upper_trial, :lower_outcome, :upper_outcome, :lower_status, :upper_status, :kind))
    PulseExperiment.write_rows(joinpath(output, "duration_refinements.csv"),
        rows.duration_refinements,
        (:condition, :initial_id, :target, :level, :lower_duration,
         :lower_duration_over_tau_e, :lower_duration_over_tau_i,
         :inserted_duration, :inserted_duration_over_tau_e,
         :inserted_duration_over_tau_i, :upper_duration,
         :upper_duration_over_tau_e, :upper_duration_over_tau_i, :reasons))
    PulseExperiment.write_rows(joinpath(output, "duration_brackets.csv"),
        rows.duration_brackets,
        (:condition, :initial_id, :target, :lower_duration, :upper_duration,
         :lower_duration_over_tau_e, :upper_duration_over_tau_e,
         :lower_duration_over_tau_i, :upper_duration_over_tau_i, :reasons,
         :lower_destinations, :upper_destinations, :lower_boundaries,
         :upper_boundaries, :lower_unresolved, :upper_unresolved,
         :lower_integration_failure, :upper_integration_failure))
end

function _run_experiment(config, config_path::AbstractString, output_dir::AbstractString)
    selection = config.selection
    model = config.models.failure_of_inhibition
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    for directory in ("contexts", "diagnostics", "trajectories", "rechecks")
        mkpath(joinpath(output, directory))
    end
    metadata = PulseExperiment.pulse_provenance(config_path, output; smoke=false)
    relative = "scripts/run_rescue_replay.jl"
    cp(joinpath(REPOSITORY_ROOT, relative), joinpath(output, "source", relative))
    metadata["source_sha256"][relative] = PulseExperiment.file_hash(
        joinpath(output, "source", relative))
    canonical_relative = "experiments/rescue_replay.toml"
    canonical_archive = joinpath(output, "source", canonical_relative)
    mkpath(dirname(canonical_archive))
    cp(config_path, canonical_archive)
    metadata["source_sha256"][canonical_relative] =
        PulseExperiment.file_hash(canonical_archive)
    metadata["purpose"] = "coordinate-identified finite rescue replay; no exact duration threshold or biological claim"
    metadata["condition"] = string(selection.condition)
    metadata["target"] = string(selection.target)
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_rescue_replay.jl --config config.toml --output replay"
    metadata["artifact_schema"]["initial_states"] =
        "one coordinate-selected locally attracting Figure 3 FoI equilibrium"
    metadata["artifact_schema"]["claim_summary"] =
        "coordinate matches, per-duration observations, and finite protocol-change brackets; censored brackets are not claim evidence"
    metadata["artifact_schema"]["rechecks"] =
        "10-ms amplitude cap plus both endpoints of every sampled source-to-destination boundary, re-integrated for 20 seconds at tighter tolerances"

    search = find_equilibria(model; seeds=PulseExperiment.pulse_seeds(model, config.grid_points),
        options=config.equilibrium_options, stability_options=config.stability_options)
    source_id, source = matching_attracting_equilibrium(search, selection.source_state,
        selection.coordinate_atol, "source")
    destination_id, destination = matching_attracting_equilibrium(search,
        selection.destination_state, selection.coordinate_atol, "destination")
    source_id != destination_id || throw(ArgumentError("source and destination matched the same equilibrium"))
    context = PulseExperiment.context_record(search)
    PulseExperiment.write_toml(joinpath(output, "contexts", "failure_of_inhibition.toml"), context)
    equilibria = [merge(PulseExperiment.summary_equilibrium("failure_of_inhibition", index, equilibrium),
        (u_I=model.coupling.e_to_i * equilibrium.state[1] - model.coupling.i_to_i * equilibrium.state[2],
         F_I_prime=response_derivative(model.inhibitory.response,
            model.coupling.e_to_i * equilibrium.state[1] - model.coupling.i_to_i * equilibrium.state[2])))
        for (index, equilibrium) in enumerate(search.equilibria)]
    attempts = [PulseExperiment.summary_attempt("failure_of_inhibition", index, attempt)
        for (index, attempt) in enumerate(search.attempts)]
    PulseExperiment.write_rows(joinpath(output, "equilibria.csv"), equilibria,
        (:context_id, :equilibrium, :E, :I, :residual_norm, :near_singular,
         :representative_attempt, :member_attempts, :stability, :geometry,
         :eigenvalue_1_real, :eigenvalue_1_imaginary, :eigenvalue_2_real,
         :eigenvalue_2_imaginary, :spectral_abscissa, :u_I, :F_I_prime))
    PulseExperiment.write_rows(joinpath(output, "attempts.csv"), attempts,
        (:context_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I,
         :solver_status, :solver_success, :residual_norm, :validation,
         :near_singular, :reasons))

    initial = (id="coordinate_selected_source", state=copy(source.state),
        provenance="unique attracting equilibrium within configured coordinate tolerance")
    result = run_pulse_experiments(model; equilibria=search,
        options=config.pulse_options, initial_states=[initial])
    rows = PulseExperiment.save_condition(result, :failure_of_inhibition, output)
    write_primary_rows(rows, output)

    rechecks = NamedTuple[]
    options = recheck_options(config.pulse_options, selection)
    for ((duration, amplitude), primary_index) in selected_rechecks(
        result, source_id, destination_id, selection)
        primary = result.trials[primary_index]
        recheck = run_pulse_trial(model, search, source.state; target=selection.target,
            amplitude, duration, options, initial_id=initial.id,
            initial_provenance=initial.provenance)
        recheck_id = "recheck_$(length(rechecks) + 1)"
        PulseExperiment.write_toml(joinpath(output, "rechecks", "$recheck_id.toml"),
            (duration, amplitude, primary_status=primary.status,
             primary_outcome=primary.outcome_equilibrium, result=recheck))
        push!(rechecks, (; recheck_id, duration, amplitude,
            primary_status=string(primary.status),
            primary_outcome=PulseExperiment.missing_id(primary.outcome_equilibrium),
            recheck_status=string(recheck.status),
            recheck_outcome=PulseExperiment.missing_id(recheck.outcome_equilibrium),
            matches_primary=primary.status == recheck.status &&
                primary.outcome_equilibrium == recheck.outcome_equilibrium,
            followup_time=last(recheck.attempts).followup_time))
    end
    PulseExperiment.write_rows(joinpath(output, "rechecks.csv"), rechecks,
        (:recheck_id, :duration, :amplitude, :primary_status, :primary_outcome,
         :recheck_status, :recheck_outcome, :matches_primary, :followup_time))

    observations = duration_observations(result, source_id, destination_id)
    brackets = protocol_change_brackets(observations, rechecks)
    summary = Dict{String,Any}(
        "schema_version" => 1,
        "condition" => string(selection.condition),
        "target" => string(selection.target),
        "source" => Dict("requested_state" => selection.source_state,
            "matched_equilibrium" => source_id, "matched_state" => source.state,
            "stability" => string(source.stability.classification)),
        "destination" => Dict("requested_state" => selection.destination_state,
            "matched_equilibrium" => destination_id, "matched_state" => destination.state,
            "stability" => string(destination.stability.classification)),
        "duration_observations" => observations,
        "protocol_change_brackets" => brackets,
        "recheck_count" => length(rechecks),
        "all_rechecks_match_primary" => all(row -> row.matches_primary, rechecks),
        "interpretation" => "finite sampled evidence only; no exact minimum duration, global reachability, permanence, or biological rescue inference",
    )
    PulseExperiment.write_toml(joinpath(output, "claim_summary.toml"), summary)
    success = all(trial -> trial.status != :integration_failed, result.trials) &&
        all(row -> row.recheck_status != "integration_failed", rechecks)
    metadata["execution_success"] = success
    metadata["trial_count"] = length(result.trials)
    metadata["recheck_count"] = length(rechecks)
    metadata["source_equilibrium"] = source_id
    metadata["destination_equilibrium"] = destination_id
    PulseExperiment.write_toml(joinpath(output, "metadata.toml"), metadata)
    PulseExperiment.artifact_checksums(output)
    return (; success, result, rechecks, summary)
end

"""Run the canonical finite 10/15/20-ms replay without assigning an exact rescue threshold."""
function run_experiment(config_path::AbstractString, output_dir::AbstractString)
    config = load_config(config_path)
    return _run_experiment(config, config_path, output_dir)
end

function main(args=ARGS)
    config = joinpath(REPOSITORY_ROOT, "experiments", "rescue_replay.toml")
    output = nothing
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        option in ("--config", "--output") || throw(ArgumentError("unknown option: $option"))
        index < length(args) || throw(ArgumentError("$option requires a value"))
        option == "--config" ? (config = args[index + 1]) : (output = args[index + 1])
        index += 2
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config, output)
    println("Wrote $(length(result.result.trials)) rescue trials and $(length(result.rechecks)) rechecks; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(RescueReplay.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
