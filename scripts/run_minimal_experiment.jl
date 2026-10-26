"""Reproducible synthetic experiment; no biological regime labels are inferred."""
module MinimalExperiment

using FailureOfInhibition2025
using SciMLBase: successful_retcode
import CSV
using LinearAlgebra: BLAS
import SHA
import TOML

include("study_support.jl")
using .StudySupport: require_keys, finite_number, pair, positive_integer, options_from,
    toml_value, write_toml, error_record, source_files, file_hash, git_output,
    archive_provenance, context_record, summary_attempt, summary_equilibrium,
    write_rows, artifact_checksums

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const CONDITIONS = (:control, :failure_of_inhibition)
const PROTOCOLS = (:baseline, :pulsed)

"""Validate the complete version-1 configuration without creating output files."""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    require_keys(raw, ("schema_version", "initial_state", "model", "drive",
        "equilibrium", "stability", "diagnostics", "variants"), "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema_version"))
    initial_state = collect(pair(raw["initial_state"], "initial_state"))
    all(x -> 0 <= x <= 1, initial_state) ||
        throw(ArgumentError("initial_state must lie in [0,1]^2"))
    require_keys(raw["model"], ("excitatory", "inhibitory", "coupling"), "model")
    excitatory = raw["model"]["excitatory"]
    inhibitory = raw["model"]["inhibitory"]
    require_keys(excitatory, ("timescale", "slope", "threshold"), "model.excitatory")
    require_keys(inhibitory, ("timescale", "slope", "threshold", "failure_threshold"),
        "model.inhibitory")
    population(table) = PopulationParameters(
        timescale=finite_number(table["timescale"], "timescale"; positive=true),
        response=LogisticResponse(
            slope=finite_number(table["slope"], "slope"; positive=true),
            threshold=finite_number(table["threshold"], "threshold"),
        ),
    )
    e_population, i_population = population(excitatory), population(inhibitory)
    failure_threshold = finite_number(inhibitory["failure_threshold"], "failure_threshold")
    coupling_keys = ("e_to_e", "i_to_e", "e_to_i", "i_to_i")
    coupling = options_from(raw["model"]["coupling"], PointCoupling, coupling_keys,
        "model.coupling")
    drive = raw["drive"]
    require_keys(drive, ("baseline", "interpretation", "pulses"), "drive")
    drive["interpretation"] == "AfferentExcitation" || throw(ArgumentError(
        "the synthetic experiment requires the AfferentExcitation interpretation",
    ))
    baseline = pair(drive["baseline"], "drive.baseline")
    drive["pulses"] isa AbstractVector || throw(ArgumentError("drive.pulses must be an array"))
    pulses = map(drive["pulses"]) do pulse
        require_keys(pulse, ("onset", "offset", "increment"), "pulse")
        DrivePulse(onset=finite_number(pulse["onset"], "pulse.onset"),
            offset=finite_number(pulse["offset"], "pulse.offset"),
            increment=pair(pulse["increment"], "pulse.increment"))
    end
    models = Dict(protocol => matched_point_models(
        excitatory=e_population, inhibitory_control=i_population,
        failure_threshold=failure_threshold, coupling=coupling,
        drive=PiecewiseConstantDrive(baseline=baseline,
            pulses=protocol == :pulsed ? pulses : (), interpretation=AfferentExcitation),
    ) for protocol in PROTOCOLS)
    equilibrium_options = options_from(raw["equilibrium"], EquilibriumOptions,
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium"; integer_key="maxiters")
    stability_options = options_from(raw["stability"], StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")
    diagnostic_options = options_from(raw["diagnostics"], DiagnosticOptions,
        ("window_duration", "coordinate_atol", "balance_atol", "min_samples"),
        "diagnostics"; integer_key="min_samples")
    raw["variants"] isa AbstractVector && !isempty(raw["variants"]) ||
        throw(ArgumentError("variants must be a nonempty array"))
    variants = map(raw["variants"]) do variant
        require_keys(variant, ("name", "time_span", "saveat", "abstol", "reltol",
            "domain_atol", "maxiters"), "variant")
        name = variant["name"]
        name isa AbstractString && occursin(r"^[a-z][a-z0-9_]*$", name) ||
            throw(ArgumentError("variant name must contain lowercase letters, digits, underscores"))
        time_span = pair(variant["time_span"], "variant.time_span")
        time_span[1] < time_span[2] || throw(ArgumentError("variant time_span must increase"))
        time_span[2] - time_span[1] >= 2diagnostic_options.window_duration ||
            throw(ArgumentError("variant time_span must cover both diagnostic windows"))
        all(pulse -> time_span[1] <= pulse.onset < pulse.offset <= time_span[2], pulses) ||
            throw(ArgumentError("each variant must include the complete pulse protocol"))
        (name=String(name), time_span=time_span,
         saveat=finite_number(variant["saveat"], "variant.saveat"; positive=true),
         abstol=finite_number(variant["abstol"], "variant.abstol"; positive=true),
         reltol=finite_number(variant["reltol"], "variant.reltol"; positive=true),
         domain_atol=finite_number(variant["domain_atol"], "variant.domain_atol"; nonnegative=true),
         maxiters=positive_integer(variant["maxiters"], "variant.maxiters"))
    end
    length(unique(v.name for v in variants)) == length(variants) ||
        throw(ArgumentError("variant names must be unique"))
    first(variants).name == "default" || throw(ArgumentError("the first variant must be default"))
    all(v -> v.time_span[1] == first(variants).time_span[1], variants) ||
        throw(ArgumentError("variants must have the same start time"))
    return (; raw, initial_state, models, equilibrium_options, stability_options,
        diagnostic_options, variants)
end

# TOML has no null or complex scalar. Missing objects use an explicit string;
# spectra retain separate real/imaginary components, and matrices use rows.
function search_contexts(config, output)
    searches = Dict{String,Any}()
    contexts = Dict{Tuple{Symbol,Symbol},Vector{String}}()
    attempts, equilibria = NamedTuple[], NamedTuple[]
    failures = String[]
    start_time = first(config.variants).time_span[1]
    for protocol in PROTOCOLS, condition in CONDITIONS
        model = getproperty(config.models[protocol], condition)
        transitions = FailureOfInhibition2025.drive_transition_times(model.drive)
        source_times = protocol == :baseline ? [start_time] : sort!(unique(vcat(start_time, transitions)))
        ids = String[]
        for (segment, source_time) in enumerate(source_times)
            context_id = "$(condition)_$(protocol)_$(segment)"
            push!(ids, context_id)
            base_record = Dict{String,Any}("context_id" => context_id,
                "condition" => string(condition), "protocol" => string(protocol),
                "segment" => segment, "segment_start" => source_time,
                "segment_end" => segment < length(source_times) ? source_times[segment + 1] : "unbounded",
                "options" => config.equilibrium_options,
                "stability_options" => config.stability_options,
                "completeness" => CompletenessNotCertified)
            snapshot_time = protocol == :baseline ? nothing : source_time
            try
                result = find_equilibria(model; snapshot_time,
                    options=config.equilibrium_options,
                    stability_options=config.stability_options)
                searches[context_id] = result
                merge!(base_record, context_record(result))
                append!(attempts, [summary_attempt(context_id, i, a) for (i, a) in enumerate(result.attempts)])
                append!(equilibria, [summary_equilibrium(context_id, i, e) for (i, e) in enumerate(result.equilibria)])
            catch error
                error isa InterruptException && rethrow()
                push!(failures, context_id)
                context = FailureOfInhibition2025._frozen_point_context(model, snapshot_time)
                merge!(base_record, Dict("status" => "execution_failed",
                    "error" => error_record(error), "source_time" => snapshot_time,
                    "frozen_drive" => context.frozen_drive,
                    "requested_seeds" => default_equilibrium_seeds(context.frozen_model),
                    "attempts_status" => "search threw; per-seed results were not returned"))
            end
            write_toml(joinpath(output, "contexts", context_id * ".toml"), base_record)
        end
        contexts[(condition, protocol)] = ids
    end
    return (; searches, contexts, attempts, equilibria, failures)
end

function save_times(variant, options)
    start_time, end_time = variant.time_span
    middle = end_time - options.window_duration
    boundaries = [middle - options.window_duration, middle, end_time]
    return sort!(unique(vcat(collect(start_time:variant.saveat:end_time), boundaries)))
end

function run_case(config, variant, condition, protocol, search_data, output)
    case_id = "$(variant.name)_$(condition)_$(protocol)"
    ids = search_data.contexts[(condition, protocol)]
    terminal_context = last(ids)
    model = getproperty(config.models[protocol], condition)
    trajectory_path, diagnostic_path = "", ""
    status, solver_status, classification = "completed", "not_available", "TrajectoryUnresolved"
    integration_success = false
    reasons = String[]
    error_type, error_message = "", ""
    diagnostic_record = Dict{String,Any}("case_id" => case_id,
        "variant" => variant, "condition" => string(condition), "protocol" => string(protocol),
        "context_ids" => ids, "terminal_context" => terminal_context,
        "options" => config.diagnostic_options,
        "periodic_orbit_status" => "unvalidated",
        "biological_interpretation" => "not_assigned")
    try
        solution = solve_point_model(config.initial_state, variant.time_span, model;
            saveat=save_times(variant, config.diagnostic_options),
            abstol=variant.abstol, reltol=variant.reltol,
            domain_atol=variant.domain_atol, maxiters=variant.maxiters)
        solver_status = string(solution.retcode)
        integration_success = successful_retcode(solution.retcode)
        trajectory_path = "trajectories/$(case_id).csv"
        write_trajectory_csv(joinpath(output, trajectory_path), solution)
        diagnostic_record["solver_status"] = solver_status
        if haskey(search_data.searches, terminal_context)
            diagnostics = diagnose_trajectory(solution, model;
                equilibria=search_data.searches[terminal_context], options=config.diagnostic_options)
            diagnostic_record["diagnostics"] = diagnostics
            integration_success = diagnostics.integration_success
            classification = string(diagnostics.classification)
            append!(reasons, string.(diagnostics.reasons))
        else
            push!(reasons, "terminal_equilibrium_search_failed")
        end
        if !integration_success
            status = "execution_failed"
            push!(reasons, "integration_failed")
        end
        if any(id -> !haskey(search_data.searches, id), ids)
            status = "execution_failed"
            push!(reasons, "equilibrium_search_failed")
        end
    catch error
        error isa InterruptException && rethrow()
        status = "execution_failed"
        failure = error_record(error)
        error_type, error_message = failure["type"], failure["message"]
        diagnostic_record["error"] = failure
        push!(reasons, "execution_exception")
    end
    unique!(reasons)
    diagnostic_record["status"] = status
    diagnostic_record["integration_success"] = integration_success
    diagnostic_record["classification"] = classification
    diagnostic_record["reasons"] = reasons
    diagnostic_path = "diagnostics/$(case_id).toml"
    write_toml(joinpath(output, diagnostic_path), diagnostic_record)
    return (; case_id, variant=variant.name, condition=string(condition), protocol=string(protocol),
        status, solver_status, integration_success, diagnostic_classification=classification,
        reasons=join(reasons, ";"), terminal_context, context_ids=join(ids, ";"),
        trajectory_path, diagnostic_path, execution_error_type=error_type,
        execution_error_message=error_message)
end

function comparison_rows(cases)
    rows = NamedTuple[]
    for case in cases
        case.variant == "default" && continue
        reference = only(filter(row -> row.variant == "default" &&
            row.condition == case.condition && row.protocol == case.protocol, cases))
        push!(rows, (; condition=case.condition, protocol=case.protocol,
            reference_case=reference.case_id, comparison_case=case.case_id,
            reference_status=reference.status, comparison_status=case.status,
            reference_classification=reference.diagnostic_classification,
            comparison_classification=case.diagnostic_classification,
            classification_changed=reference.diagnostic_classification != case.diagnostic_classification,
            reasons_changed=reference.reasons != case.reasons))
    end
    return rows
end


"""
    run_experiment(config_path, output_dir)

Run all configured numerical variants for matched control/FoI baseline/pulsed
cases. Validate before writing, require an absent or empty output directory,
and return `(success, cases)`. Unresolved diagnostics and nonlinear solver
failures retained as search attempts do not themselves indicate an execution
failure. Exceptions or unsuccessful ODE integration do.
"""
function run_experiment(config_path::AbstractString, output_dir::AbstractString)
    config = load_config(config_path)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be an absent or empty directory"))
    mkpath(output)
    for directory in ("contexts", "diagnostics", "trajectories")
        mkpath(joinpath(output, directory))
    end
    metadata = archive_provenance(config_path, output)
    search_data = search_contexts(config, output)
    cases = [run_case(config, variant, condition, protocol, search_data, output)
        for variant in config.variants for protocol in PROTOCOLS for condition in CONDITIONS]
    success = isempty(search_data.failures) && all(row -> row.status == "completed", cases)
    metadata["execution_success"] = success
    metadata["failed_search_contexts"] = search_data.failures
    metadata["equilibrium_options"] = config.equilibrium_options
    metadata["stability_options"] = config.stability_options
    metadata["diagnostic_options"] = config.diagnostic_options
    metadata["variants"] = config.variants
    write_toml(joinpath(output, "metadata.toml"), metadata)
    CSV.write(joinpath(output, "cases.csv"), cases)
    write_rows(joinpath(output, "attempts.csv"), search_data.attempts,
        (:context_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I, :solver_status,
         :solver_success, :residual_norm, :validation, :near_singular, :reasons))
    write_rows(joinpath(output, "equilibria.csv"), search_data.equilibria,
        (:context_id, :equilibrium, :E, :I, :residual_norm, :near_singular,
         :representative_attempt, :member_attempts, :stability, :geometry,
         :eigenvalue_1_real, :eigenvalue_1_imaginary, :eigenvalue_2_real,
         :eigenvalue_2_imaginary, :spectral_abscissa))
    write_rows(joinpath(output, "comparisons.csv"), comparison_rows(cases),
        (:condition, :protocol, :reference_case, :comparison_case, :reference_status,
         :comparison_status, :reference_classification, :comparison_classification,
         :classification_changed, :reasons_changed))
    artifact_checksums(output)
    return (; success, cases)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "minimal.toml")
    output = nothing
    length(args) % 2 == 0 || throw(ArgumentError("use --config FILE --output DIRECTORY"))
    seen = Set{String}()
    for index in 1:2:length(args)
        option, value = args[index], args[index + 1]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--config"
            config_path = value
        elseif option == "--output"
            output = value
        else
            throw(ArgumentError("unknown option: $option"))
        end
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output)
    println("Wrote $(length(result.cases)) cases; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(MinimalExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
