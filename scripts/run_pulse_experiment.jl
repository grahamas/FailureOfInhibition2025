"""Explicit pulse experiments at an author-specified autonomous parameter point."""
module PulseExperiment

using FailureOfInhibition2025
import CSV
import TOML
include("run_minimal_experiment.jl")
using .MinimalExperiment: require_keys, finite_number, positive_integer,
    options_from, write_toml, write_rows, archive_provenance, file_hash,
    context_record, summary_attempt, summary_equilibrium, artifact_checksums, error_record

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

function load_config(path)
    raw = TOML.parsefile(path)
    require_keys(raw, ("schema_version", "description", "model", "search",
        "equilibrium", "stability", "diagnostics", "pulses"), "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema_version"))
    raw["description"] isa AbstractString || throw(ArgumentError("description must be a string"))
    require_keys(raw["model"], ("excitatory", "inhibitory", "coupling"), "model")
    e, i = raw["model"]["excitatory"], raw["model"]["inhibitory"]
    require_keys(e, ("timescale", "slope", "threshold"), "model.excitatory")
    require_keys(i, ("timescale", "slope", "threshold", "failure_threshold"), "model.inhibitory")
    population(table) = PopulationParameters(
        timescale=finite_number(table["timescale"], "timescale"; positive=true),
        response=LogisticResponse(slope=finite_number(table["slope"], "slope"; positive=true),
            threshold=finite_number(table["threshold"], "threshold")))
    coupling = options_from(raw["model"]["coupling"], PointCoupling,
        ("e_to_e", "i_to_e", "e_to_i", "i_to_i"), "model.coupling")
    models = matched_point_models(excitatory=population(e), inhibitory_control=population(i),
        failure_threshold=finite_number(i["failure_threshold"], "failure_threshold"),
        coupling=coupling, drive=NoDrive())
    require_keys(raw["search"], ("grid_points",), "search")
    grid_points = positive_integer(raw["search"]["grid_points"], "search.grid_points")
    grid_points >= 2 || throw(ArgumentError("search.grid_points must be at least two"))
    equilibrium_options = options_from(raw["equilibrium"], EquilibriumOptions,
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium"; integer_key="maxiters")
    stability_options = options_from(raw["stability"], StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")
    diagnostic_options = options_from(raw["diagnostics"], DiagnosticOptions,
        ("window_duration", "coordinate_atol", "balance_atol", "min_samples"),
        "diagnostics"; integer_key="min_samples")
    pulses = raw["pulses"]
    require_keys(pulses, ("amplitudes", "durations", "targets", "followup_times",
        "refinement_levels", "retain_trajectories", "trajectory_saveat", "abstol",
        "reltol", "domain_atol", "maxiters"), "pulses")
    axis = pulses["amplitudes"]
    require_keys(axis, ("start", "stop", "step"), "pulses.amplitudes")
    start = finite_number(axis["start"], "amplitudes.start"; nonnegative=true)
    stop = finite_number(axis["stop"], "amplitudes.stop"; nonnegative=true)
    step = finite_number(axis["step"], "amplitudes.step"; positive=true)
    start <= stop || throw(ArgumentError("amplitudes.start must not exceed stop"))
    amplitudes = collect(start:step:stop)
    last(amplitudes) == stop || push!(amplitudes, stop)
    pulses["targets"] isa AbstractVector && all(x -> x isa String, pulses["targets"]) ||
        throw(ArgumentError("pulses.targets must be an array of strings"))
    pulse_options = PulseExperimentOptions(amplitudes=amplitudes,
        durations=pulses["durations"], targets=Symbol.(pulses["targets"]),
        followup_times=pulses["followup_times"], diagnostic_options=diagnostic_options,
        refinement_levels=pulses["refinement_levels"],
        retain_trajectories=pulses["retain_trajectories"],
        trajectory_saveat=pulses["trajectory_saveat"], abstol=pulses["abstol"],
        reltol=pulses["reltol"], domain_atol=pulses["domain_atol"], maxiters=pulses["maxiters"])
    return (; raw, models, grid_points, equilibrium_options, stability_options, pulse_options)
end

function smoke_options(options)
    return PulseExperimentOptions(amplitudes=unique([first(options.amplitudes), last(options.amplitudes)]),
        durations=unique([first(options.durations), last(options.durations)]),
        targets=options.targets, followup_times=options.followup_times,
        diagnostic_options=options.diagnostic_options, refinement_levels=0,
        retain_trajectories=options.retain_trajectories,
        trajectory_saveat=options.trajectory_saveat, abstol=options.abstol,
        reltol=options.reltol, domain_atol=options.domain_atol, maxiters=options.maxiters)
end

function pulse_provenance(config_path, output; smoke)
    metadata = archive_provenance(config_path, output)
    relative = "scripts/run_pulse_experiment.jl"
    cp(joinpath(REPOSITORY_ROOT, relative), joinpath(output, "source", relative))
    metadata["source_sha256"][relative] = file_hash(joinpath(output, "source", relative))
    metadata["purpose"] = "explicit pulse transitions; finite-window equilibrium compatibility; no biological labels"
    metadata["smoke"] = smoke
    metadata["equilibrium_seed_policy"] = "configured uniform grid on analytical sharper rectangle plus default seeds; discovery completeness not certified"
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_pulse_experiment.jl --config config.toml --output replay" *
        (smoke ? " --smoke" : "")
    metadata["artifact_schema"] = Dict("version" => 1,
        "trajectory_columns" => ["time", "E", "I"],
        "toml_unavailable_value" => "not_available",
        "outcome_equilibrium" => "local equilibrium ID only; compatibility is finite-window",
        "initial_states" => "all discovered admissible locally attracting equilibria",
        "trials" => "one row per condition, initial equilibrium, target, amplitude, duration",
        "followups" => "every attempted observation horizon, including failures and unresolved diagnostics",
        "tails" => "two closed sampled diagnostic windows with E, I, u_I and F_I_prime summaries",
        "boundaries" => "all adjacent differing resolved outcomes and unresolved brackets; no monotonicity assumption",
        "cost_units" => "effective-input units times ms; not biological energy")
    return metadata
end

function pulse_seeds(model, grid_points)
    bounds = FailureOfInhibition2025._equilibrium_upper_bounds(model, Float64)
    return vcat([[e, i] for e in range(0.0, bounds[1]; length=grid_points)
        for i in range(0.0, bounds[2]; length=grid_points)], default_equilibrium_seeds(model))
end

missing_id(x) = x === nothing ? missing : x

function save_condition(result, condition, output)
    trials, followups, tails, boundaries = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    for (index, trial) in enumerate(result.trials)
        trial_id = "$(condition)_$index"
        final = last(trial.attempts)
        push!(trials, (condition=string(condition), trial_id=trial_id,
            initial_id=trial.initial_id, initial_E=trial.initial_state[1], initial_I=trial.initial_state[2],
            initial_provenance=trial.initial_provenance, target=string(trial.target),
            amplitude=trial.amplitude, duration=trial.duration,
            integrated_E=trial.integrated_E, integrated_I=trial.integrated_I,
            absolute_input_cost=trial.absolute_input_cost, status=string(trial.status),
            outcome_equilibrium=missing_id(trial.outcome_equilibrium),
            final_followup_time=final.followup_time, attempts=length(trial.attempts)))
        for (attempt_index, attempt) in enumerate(trial.attempts)
            diagnostic = attempt.diagnostics
            attempt_id = "$(trial_id)_$attempt_index"
            write_toml(joinpath(output, "diagnostics", "$attempt_id.toml"),
                (trial_id=trial_id, attempt=attempt_index,
                 initial_id=trial.initial_id, initial_state=trial.initial_state,
                 target=trial.target, amplitude=trial.amplitude, duration=trial.duration,
                 result=attempt))
            result.options.retain_trajectories && !isempty(attempt.times) &&
                write_trajectory_csv(joinpath(output, "trajectories", "$attempt_id.csv"),
                    (t=attempt.times, u=attempt.states))
            push!(followups, (condition=string(condition), trial_id=trial_id, attempt=attempt_index,
                followup_time=attempt.followup_time, status=string(attempt.status),
                solver_status=string(attempt.solver_status),
                outcome_equilibrium=missing_id(attempt.outcome_equilibrium),
                diagnostic_classification=diagnostic === nothing ? "not_available" : string(diagnostic.classification),
                integration_success=diagnostic === nothing ? false : diagnostic.integration_success,
                reasons=join(string.(attempt.reasons), ";"),
                error=attempt.error === nothing ? "" : attempt.error.message))
            attempt.tail === nothing && continue
            for (window, tail) in enumerate(attempt.tail)
                push!(tails, (condition=string(condition), trial_id=trial_id, attempt=attempt_index,
                    window=window, lower_time=tail.bounds[1], upper_time=tail.bounds[2],
                    samples=tail.samples, E_mean=tail.E.mean, E_minimum=tail.E.minimum,
                    E_maximum=tail.E.maximum, I_mean=tail.I.mean, I_minimum=tail.I.minimum,
                    I_maximum=tail.I.maximum, u_I_mean=tail.u_I.mean,
                    u_I_minimum=tail.u_I.minimum, u_I_maximum=tail.u_I.maximum,
                    F_I_prime_mean=tail.F_I_prime.mean,
                    F_I_prime_minimum=tail.F_I_prime.minimum, F_I_prime_maximum=tail.F_I_prime.maximum))
            end
        end
    end
    for boundary in result.boundaries
        push!(boundaries, (condition=string(condition), initial_id=boundary.initial_id,
            target=string(boundary.target), duration=boundary.duration,
            lower_amplitude=boundary.lower_amplitude, upper_amplitude=boundary.upper_amplitude,
            lower_trial="$(condition)_$(boundary.lower_trial)",
            upper_trial="$(condition)_$(boundary.upper_trial)",
            lower_outcome=missing_id(boundary.lower_outcome),
            upper_outcome=missing_id(boundary.upper_outcome), kind=string(boundary.kind)))
    end
    return (; trials, followups, tails, boundaries)
end

"""Validate, snapshot and execute; refuse to overwrite any nonempty output directory."""
function run_experiment(config_path, output_dir; smoke=false)
    config = load_config(config_path)
    options = smoke ? smoke_options(config.pulse_options) : config.pulse_options
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    mkpath(output)
    for directory in ("contexts", "diagnostics", "trajectories")
        mkpath(joinpath(output, directory))
    end
    metadata = pulse_provenance(config_path, output; smoke=smoke)
    all_trials, all_followups, all_tails, all_boundaries =
        NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    all_equilibria, all_attempts = NamedTuple[], NamedTuple[]
    contexts, failures = Dict{String,Any}(), String[]
    for condition in (:control, :failure_of_inhibition)
        model = getproperty(config.models, condition)
        try
            search = find_equilibria(model; seeds=pulse_seeds(model, config.grid_points),
                options=config.equilibrium_options, stability_options=config.stability_options)
            contexts[string(condition)] = context_record(search)
            write_toml(joinpath(output, "contexts", "$condition.toml"), context_record(search))
            append!(all_attempts, [summary_attempt(string(condition), index, attempt)
                for (index, attempt) in enumerate(search.attempts)])
            append!(all_equilibria, [merge(summary_equilibrium(string(condition), index, equilibrium),
                (u_I=model.coupling.e_to_i * equilibrium.state[1] - model.coupling.i_to_i * equilibrium.state[2],
                 F_I_prime=response_derivative(model.inhibitory.response,
                    model.coupling.e_to_i * equilibrium.state[1] - model.coupling.i_to_i * equilibrium.state[2])))
                for (index, equilibrium) in enumerate(search.equilibria)])
            result = run_pulse_experiments(model; equilibria=search, options=options)
            rows = save_condition(result, condition, output)
            append!(all_trials, rows.trials)
            append!(all_followups, rows.followups)
            append!(all_tails, rows.tails)
            append!(all_boundaries, rows.boundaries)
            println("$condition: $(length(search.equilibria)) discovered equilibria, $(length(result.initial_states)) attracting starts, $(length(rows.trials)) pulse trials")
        catch error
            error isa InterruptException && rethrow()
            push!(failures, string(condition))
            record = Dict("status" => "failed", "error" => error_record(error))
            contexts[string(condition)] = record
            write_toml(joinpath(output, "contexts", "$(condition)_error.toml"), record)
        end
    end
    success = isempty(failures) && all(row -> row.status != "integration_failed", all_trials)
    metadata["execution_success"] = success
    metadata["failed_contexts"] = failures
    metadata["effective_pulse_options"] = options
    metadata["equilibrium_options"] = config.equilibrium_options
    metadata["stability_options"] = config.stability_options
    metadata["grid_points"] = config.grid_points
    metadata["trial_count"] = length(all_trials)
    metadata["unresolved_trial_count"] = count(row -> row.status == "unresolved", all_trials)
    write_toml(joinpath(output, "metadata.toml"), metadata)
    write_rows(joinpath(output, "trials.csv"), all_trials,
        (:condition, :trial_id, :initial_id, :initial_E, :initial_I, :initial_provenance,
         :target, :amplitude, :duration, :integrated_E, :integrated_I, :absolute_input_cost,
         :status, :outcome_equilibrium, :final_followup_time, :attempts))
    write_rows(joinpath(output, "followups.csv"), all_followups,
        (:condition, :trial_id, :attempt, :followup_time, :status, :solver_status,
         :outcome_equilibrium, :diagnostic_classification, :integration_success, :reasons, :error))
    write_rows(joinpath(output, "tails.csv"), all_tails,
        (:condition, :trial_id, :attempt, :window, :lower_time, :upper_time, :samples,
         :E_mean, :E_minimum, :E_maximum, :I_mean, :I_minimum, :I_maximum,
         :u_I_mean, :u_I_minimum, :u_I_maximum, :F_I_prime_mean, :F_I_prime_minimum, :F_I_prime_maximum))
    write_rows(joinpath(output, "boundaries.csv"), all_boundaries,
        (:condition, :initial_id, :target, :duration, :lower_amplitude, :upper_amplitude,
         :lower_trial, :upper_trial, :lower_outcome, :upper_outcome, :kind))
    write_rows(joinpath(output, "equilibria.csv"), all_equilibria,
        (:context_id, :equilibrium, :E, :I, :residual_norm, :near_singular,
         :representative_attempt, :member_attempts, :stability, :geometry,
         :eigenvalue_1_real, :eigenvalue_1_imaginary, :eigenvalue_2_real,
         :eigenvalue_2_imaginary, :spectral_abscissa, :u_I, :F_I_prime))
    write_rows(joinpath(output, "attempts.csv"), all_attempts,
        (:context_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I,
         :solver_status, :solver_success, :residual_norm, :validation, :near_singular, :reasons))
    artifact_checksums(output)
    return (; success, trials=all_trials, failures)
end

function main(args=ARGS)
    config = joinpath(REPOSITORY_ROOT, "experiments", "pulses.toml")
    output, smoke = nothing, false
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--smoke"
            smoke = true
            index += 1
        elseif option in ("--config", "--output")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            option == "--config" ? (config = args[index + 1]) : (output = args[index + 1])
            index += 2
        else
            throw(ArgumentError("unknown option: $option"))
        end
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config, output; smoke=smoke)
    println("Wrote $(length(result.trials)) pulse trials; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(PulseExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
