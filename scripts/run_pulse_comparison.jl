"""Explicit pulse protocols at separately identified parameter configurations."""
module PulseComparison

import CSV
import TOML
include(joinpath(@__DIR__, "run_pulse_experiment.jl"))
using .PulseExperiment.MinimalExperiment: write_toml, archive_provenance,
    artifact_checksums, file_hash, error_record

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const CASE_NAMES = ("baseline", "theta_off_10", "theta_off_12", "e_to_i_60_percent",
    "e_to_e_60_percent", "i_to_e_150_percent", "figure4_exploration")

function validate_case_names(cases)
    cases isa AbstractString && throw(ArgumentError("cases must be a collection of names"))
    names = collect(cases)
    !isempty(names) || throw(ArgumentError("at least one case is required"))
    all(name -> name isa AbstractString && name in CASE_NAMES, names) ||
        throw(ArgumentError("case names must belong to $(join(CASE_NAMES, ", "))"))
    length(unique(names)) == length(names) || throw(ArgumentError("case names must be unique"))
    return String.(names)
end

"""
Derive one explicitly named parameter configuration without mutating its baseline.

Interventions change exactly one inhibitory threshold or coupling magnitude.
`figure4_exploration` sets its stated couplings plus an explicitly chosen
`e_to_i=19`, `theta_off=8`; it is a separate exploration anchor, not an
individually varied intervention. Pulse protocols and all remaining parameters
are inherited unchanged.
"""
function derived_config(raw, name::AbstractString)
    name in CASE_NAMES || throw(ArgumentError("unknown comparison case: $name"))
    derived = deepcopy(raw)
    coupling = derived["model"]["coupling"]
    inhibitory = derived["model"]["inhibitory"]
    if name == "theta_off_10"
        inhibitory["failure_threshold"] = 10.0
    elseif name == "theta_off_12"
        inhibitory["failure_threshold"] = 12.0
    elseif name == "e_to_i_60_percent"
        coupling["e_to_i"] *= 0.6
    elseif name == "e_to_e_60_percent"
        coupling["e_to_e"] *= 0.6
    elseif name == "i_to_e_150_percent"
        coupling["i_to_e"] *= 1.5
    elseif name == "figure4_exploration"
        coupling["e_to_e"] = 19.0
        coupling["i_to_e"] = 13.0
        coupling["e_to_i"] = 19.0
        coupling["i_to_i"] = 6.0
        inhibitory["failure_threshold"] = 8.0
    end
    derived["description"] = "Pulse comparison case $name; " *
        (name == "figure4_exploration" ?
            "separate Figure 4 exploration anchor, with e_to_i=19 and theta_off=8 specified for this experiment; " :
            "individual parameter comparison; ") *
        "equilibrium IDs are local to this case, with no biological or cross-case branch labels. " * raw["description"]
    return derived
end

function case_parameters(raw)
    coupling, e, i = raw["model"]["coupling"], raw["model"]["excitatory"], raw["model"]["inhibitory"]
    return (; e_to_e=coupling["e_to_e"], i_to_e=coupling["i_to_e"],
        e_to_i=coupling["e_to_i"], i_to_i=coupling["i_to_i"],
        tau_e=e["timescale"], tau_i=i["timescale"], a_e=e["slope"], a_i=i["slope"],
        theta_e=e["threshold"], theta_on=i["threshold"], theta_off=i["failure_threshold"])
end

function comparison_provenance(config_path, output, names; smoke)
    metadata = archive_provenance(config_path, output)
    for relative in ("scripts/run_pulse_comparison.jl", "scripts/run_pulse_experiment.jl")
        destination = joinpath(output, "source", relative)
        cp(joinpath(REPOSITORY_ROOT, relative), destination)
        metadata["source_sha256"][relative] = file_hash(destination)
    end
    metadata["purpose"] = "explicit pulse comparisons across independent parameter cases"
    metadata["case_names"] = names
    metadata["smoke"] = smoke
    metadata["equilibrium_seed_policy"] = "each case uses PulseExperiment with its configured grid and default seeds independently"
    metadata["branch_correspondence"] = "not_assigned; equilibrium IDs are local to each case and condition"
    metadata["biological_interpretation"] = "not_assigned"
    metadata["absence_inference"] = "a missing discovered equilibrium or pulse transition does not certify absence or global unreachability"
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_pulse_comparison.jl --config config.toml --output replay" *
        join([" --case $name" for name in names]) * (smoke ? " --smoke" : "")
    metadata["artifact_schema"] = Dict("version" => 1,
        "cases" => "one row per parameter configuration with exact parameters, trial counts and elapsed execution seconds",
        "configs" => "complete derived configurations; the root config.toml is the original baseline",
        "case_directories" => "independent full PulseExperiment artifacts with local equilibrium IDs and continuous states",
        "unavailable_count" => "-1 if a case throws before returning trial results")
    return metadata
end

"""
Execute a named subset of pulse comparisons and retain complete per-case artifacts.

All configurations are validated before the absent output root is created. The
configured pulse protocol applies to every discovered locally attracting
equilibrium in each matched control/FoI case. Per-case failures remain explicit;
later independent cases continue. This runner does not infer branch identities
or active/seizure roles across parameter changes.
"""
function run_experiment(config_path::AbstractString, output_dir::AbstractString;
    cases=CASE_NAMES, smoke=false)
    names = validate_case_names(cases)
    base = PulseExperiment.load_config(config_path)
    configs = [derived_config(base.raw, name) for name in names]
    mktempdir() do directory
        for (name, raw) in zip(names, configs)
            path = joinpath(directory, "$name.toml")
            write_toml(path, raw)
            PulseExperiment.load_config(path)
        end
    end
    output = abspath(output_dir)
    ispath(output) && throw(ArgumentError("comparison output root must be absent"))
    mkpath(joinpath(output, "configs"))
    metadata = comparison_provenance(config_path, output, names; smoke)
    rows = NamedTuple[]
    success = true
    for (name, raw) in zip(names, configs)
        derived_path = joinpath(output, "configs", "$name.toml")
        write_toml(derived_path, raw)
        trial_count, unresolved_count, failed_trial_count = -1, -1, -1
        status, case_success, error_message = "completed", false, ""
        started = time_ns()
        try
            result = PulseExperiment.run_experiment(derived_path, joinpath(output, name); smoke)
            case_success = result.success
            trial_count = length(result.trials)
            unresolved_count = count(row -> row.status == "unresolved", result.trials)
            failed_trial_count = count(row -> row.status == "integration_failed", result.trials)
            case_success || (status = "execution_failed")
        catch error
            error isa InterruptException && rethrow()
            status = "execution_failed"
            failure = error_record(error)
            error_message = failure["message"]
            write_toml(joinpath(output, "$(name)_error.toml"), failure)
        end
        elapsed_seconds = (time_ns() - started) / 1.0e9
        success &= case_success
        push!(rows, merge((; case_name=name,
            case_kind=name == "baseline" ? "baseline" :
                name == "figure4_exploration" ? "separate_anchor" : "individual_intervention",
            config_path="configs/$name.toml", results_directory=name,
            status, execution_success=case_success, trial_count, unresolved_count,
            failed_trial_count, elapsed_seconds, error_message), case_parameters(raw)))
        # Keep a readable case index even if a later run is interrupted.
        CSV.write(joinpath(output, "cases.csv"), rows)
        println("$name: $trial_count trials; execution_success=$case_success; elapsed_seconds=$(round(elapsed_seconds; digits=3))")
    end
    metadata["execution_success"] = success
    metadata["case_count"] = length(rows)
    metadata["trial_count"] = sum(row.trial_count for row in rows if row.trial_count >= 0)
    metadata["derived_config_sha256"] = Dict(name => file_hash(joinpath(output, "configs", "$name.toml")) for name in names)
    write_toml(joinpath(output, "metadata.toml"), metadata)
    artifact_checksums(output)
    return (; success, cases=rows)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "pulses.toml")
    output, smoke = nothing, false
    cases = String[]
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option != "--case" && option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--smoke"
            smoke = true
        elseif option in ("--config", "--output", "--case")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            index += 1
            if option == "--case"
                push!(cases, args[index])
            elseif option == "--config"
                config_path = args[index]
            else
                output = args[index]
            end
        else
            throw(ArgumentError("unknown option: $option"))
        end
        index += 1
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output;
        cases=isempty(cases) ? CASE_NAMES : cases, smoke)
    println("Wrote $(length(result.cases)) parameter cases; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(PulseComparison.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
