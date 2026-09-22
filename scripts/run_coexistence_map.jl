"""Deterministic equilibrium discovery over the proposed manuscript parameter plane."""
module CoexistenceExperiment

using FailureOfInhibition2025
import CSV
import TOML

include(joinpath(@__DIR__, "run_minimal_experiment.jl"))
using .MinimalExperiment: require_keys, finite_number, positive_integer, options_from,
    write_toml, write_rows, file_hash, artifact_checksums, context_record,
    summary_attempt, summary_equilibrium, error_record

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const CONDITIONS = (:control, :failure_of_inhibition)

function axis_values(table, name)
    require_keys(table, ("minimum", "maximum", "step"), name)
    lower = finite_number(table["minimum"], "$name.minimum"; nonnegative=true)
    upper = finite_number(table["maximum"], "$name.maximum"; nonnegative=true)
    step = finite_number(table["step"], "$name.step"; positive=true)
    upper >= lower || throw(ArgumentError("$name.maximum must be at least its minimum"))
    intervals = (upper - lower) / step
    isfinite(intervals) && intervals < typemax(Int) ||
        throw(ArgumentError("$name has too many intervals"))
    count = round(Int, intervals)
    isapprox(intervals, count; atol=1e-10, rtol=1e-12) ||
        throw(ArgumentError("$name.step must divide its extent"))
    return collect(range(lower; step, length=count + 1))
end

function smoke_values(values, axis, name)
    values isa AbstractVector && !isempty(values) ||
        throw(ArgumentError("$name must be a nonempty array"))
    result = [finite_number(value, name) for value in values]
    length(unique(result)) == length(result) ||
        throw(ArgumentError("$name values must be unique"))
    # Return actual grid coordinates rather than a nearby floating-point spelling.
    indices = [findfirst(x -> isapprox(x, value; atol=1e-12, rtol=1e-12), axis)
        for value in result]
    all(!isnothing, indices) || throw(ArgumentError("$name must be a subset of its full axis"))
    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("$name values must identify distinct full-axis coordinates"))
    return sort!(axis[[Int(index) for index in indices]])
end

"""Validate the complete configuration before creating artifacts. Times are milliseconds."""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    require_keys(raw, ("schema_version", "model", "anchors", "axes", "search", "smoke",
        "equilibrium", "stability"), "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema_version"))
    require_keys(raw["model"], ("excitatory", "inhibitory"), "model")
    population(table, name) = begin
        require_keys(table, ("timescale", "slope", "threshold"), name)
        PopulationParameters(timescale=finite_number(table["timescale"], "$name.timescale";
            positive=true), response=LogisticResponse(
            slope=finite_number(table["slope"], "$name.slope"; positive=true),
            threshold=finite_number(table["threshold"], "$name.threshold")))
    end
    excitatory = population(raw["model"]["excitatory"], "model.excitatory")
    inhibitory_control = population(raw["model"]["inhibitory"], "model.inhibitory")
    raw["anchors"] isa AbstractVector && !isempty(raw["anchors"]) ||
        throw(ArgumentError("anchors must be a nonempty array"))
    anchors = map(raw["anchors"]) do anchor
        require_keys(anchor, ("name", "e_to_e", "i_to_e", "i_to_i"), "anchor")
        name = anchor["name"]
        name isa AbstractString && occursin(r"^[a-z][a-z0-9_]*$", name) ||
            throw(ArgumentError("anchor names require lowercase letters, digits, underscores"))
        (; name=String(name),
            e_to_e=finite_number(anchor["e_to_e"], "anchor.e_to_e"; nonnegative=true),
            i_to_e=finite_number(anchor["i_to_e"], "anchor.i_to_e"; nonnegative=true),
            i_to_i=finite_number(anchor["i_to_i"], "anchor.i_to_i"; nonnegative=true))
    end
    length(unique(anchor.name for anchor in anchors)) == length(anchors) ||
        throw(ArgumentError("anchor names must be unique"))
    require_keys(raw["axes"], ("e_to_i", "failure_threshold"), "axes")
    e_to_i_values = axis_values(raw["axes"]["e_to_i"], "axes.e_to_i")
    failure_threshold_values = axis_values(raw["axes"]["failure_threshold"], "axes.failure_threshold")
    first(failure_threshold_values) > inhibitory_control.response.threshold ||
        throw(ArgumentError("failure thresholds must exceed the inhibitory onset threshold"))
    require_keys(raw["search"], ("seed_grid_points",), "search")
    seed_grid_points = positive_integer(raw["search"]["seed_grid_points"], "search.seed_grid_points")
    seed_grid_points > 5 || throw(ArgumentError("seed_grid_points must supplement the default 5-by-5 grid"))
    require_keys(raw["smoke"], ("e_to_i", "failure_threshold"), "smoke")
    smoke_e_to_i_values = smoke_values(raw["smoke"]["e_to_i"], e_to_i_values, "smoke.e_to_i")
    smoke_failure_threshold_values = smoke_values(raw["smoke"]["failure_threshold"],
        failure_threshold_values, "smoke.failure_threshold")
    equilibrium_options = options_from(raw["equilibrium"], EquilibriumOptions,
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium"; integer_key="maxiters")
    stability_options = options_from(raw["stability"], StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")
    return (; raw, excitatory, inhibitory_control, anchors, e_to_i_values,
        failure_threshold_values, smoke_e_to_i_values, smoke_failure_threshold_values,
        seed_grid_points, equilibrium_options, stability_options)
end

"""Build zero-drive control/FoI models with identical populations and coupling magnitudes."""
function models_at(config, anchor, e_to_i, failure_threshold)
    if anchor isa Union{AbstractString,Symbol}
        index = findfirst(value -> value.name == string(anchor), config.anchors)
        isnothing(index) && throw(ArgumentError("unknown anchor: $anchor"))
        anchor = config.anchors[index]
    end
    return matched_point_models(excitatory=config.excitatory,
        inhibitory_control=config.inhibitory_control,
        failure_threshold=finite_number(failure_threshold, "failure_threshold"),
        coupling=PointCoupling(e_to_e=anchor.e_to_e, i_to_e=anchor.i_to_e,
            e_to_i=finite_number(e_to_i, "e_to_i"; nonnegative=true), i_to_i=anchor.i_to_i))
end

"""Union the default 5-by-5 seeds with a denser grid over the same sharper rectangle."""
function deterministic_seeds(model, grid_points=11)
    count = positive_integer(grid_points, "grid_points")
    count > 5 || throw(ArgumentError("grid_points must exceed five"))
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=count)
        for i in range(0.0, upper_i; length=count)])
    return unique!(seeds)
end

"""Continuous equilibrium observations; the branch refers to the response, not biology."""
function equilibrium_observations(model, equilibrium)
    E, I = equilibrium.state
    u_I = model.coupling.e_to_i * E - model.coupling.i_to_i * I
    response_parameters = model.inhibitory.response
    F_I = response(response_parameters, u_I)
    F_I_prime = response_derivative(response_parameters, u_I)
    midpoint = response_parameters isa FailureOfInhibitionResponse ?
        (response_parameters.onset_threshold + response_parameters.failure_threshold) / 2 : NaN
    # Equal-slope FoI symmetry determines the branch even if the derivative underflows.
    inhibitory_branch = isnan(midpoint) ? "monotone" :
        u_I < midpoint ? "ascending" : u_I > midpoint ? "descending" : "midpoint"
    return (; u_I, F_I, F_I_prime, inhibitory_response_midpoint=midpoint,
        inhibitory_branch, descending_response_branch=inhibitory_branch == "descending")
end

function archive_provenance(config_path, output, config, smoke)
    metadata = MinimalExperiment.archive_provenance(config_path, output)
    relative = "scripts/run_coexistence_map.jl"
    destination = joinpath(output, "source", relative)
    cp(joinpath(REPOSITORY_ROOT, relative), destination)
    metadata["source_sha256"][relative] = file_hash(destination)
    merge!(metadata, Dict("purpose" => "proposed manuscript parameter exploration; no physiological calibration",
        "experiment" => "coexistence_map", "smoke" => smoke,
        "equilibrium_seed_policy" => "union of default 5-by-5 and $(config.seed_grid_points)-by-$(config.seed_grid_points) sharper-rectangle grids",
        "control_cache_policy" => "one search per anchor and e_to_i; explicit matched cell rows at every failure threshold",
        "time_unit" => "millisecond", "baseline_drive" => [0.0, 0.0],
        "count_interpretation" => "discovered admissible candidates; no certificate of completeness, absence, or distinct exact equilibrium count",
        "completeness" => CompletenessNotCertified,
        "biological_interpretation" => "not_assigned", "periodic_orbit_status" => "not_tested",
        "continuation_status" => "not_performed_by_this_grid_runner",
        "equilibrium_options" => config.equilibrium_options,
        "stability_options" => config.stability_options,
        "replay_from_artifact_directory" => "julia --project=source source/scripts/run_coexistence_map.jl --config config.toml --output replay" * (smoke ? " --smoke" : ""),
        "artifact_schema" => Dict("version" => 1,
            "map" => "one row per anchor, e_to_i, failure_threshold and condition",
            "attempts" => "all returned attempts per unique search context, including rejected candidates",
            "equilibria" => "all discovered admissible equilibria per matched cell, including unstable equilibria",
            "contexts" => "complete search diagnostics; repeated control cells reference the same context",
            "missing_counts" => "negative one means search execution failed",
            "branch" => "FoI effective input relative to the equal-slope response midpoint; not a biological label",
            "toml_unavailable_value" => "not_available")))
    return metadata
end

function run_search(config, model, context_id, output)
    seeds = deterministic_seeds(model, config.seed_grid_points)
    record = Dict{String,Any}("context_id" => context_id, "model" => model,
        "requested_seeds" => seeds, "biological_interpretation" => "not_assigned",
        "periodic_orbit_status" => "not_tested", "completeness" => CompletenessNotCertified)
    result = try
        find_equilibria(model; seeds, options=config.equilibrium_options,
            stability_options=config.stability_options)
    catch error
        error isa InterruptException && rethrow()
        merge!(record, Dict("status" => "execution_failed", "error" => error_record(error),
            "attempts_status" => "search threw; per-seed results were not returned"))
        nothing
    end
    !isnothing(result) && merge!(record, context_record(result))
    write_toml(joinpath(output, "contexts", context_id * ".toml"), record)
    return result
end

"""
    run_experiment(config_path, output_dir; smoke=false)

Discover equilibria for both configured anchors and matched monotone controls.
Retain rejected attempts, unstable roots, local stability and continuous response
observations. Counts refer to discovered admissible candidates; they do not certify
completeness, absence, or the number of distinct exact equilibria.
The smoke slice uses the same seed and numerical policies as the full plane.
"""
function run_experiment(config_path::AbstractString, output_dir::AbstractString; smoke::Bool=false)
    config = load_config(config_path)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be an absent or empty directory"))
    mkpath(joinpath(output, "contexts"))
    metadata = archive_provenance(config_path, output, config, smoke)
    e_values = smoke ? config.smoke_e_to_i_values : config.e_to_i_values
    threshold_values = smoke ? config.smoke_failure_threshold_values : config.failure_threshold_values
    cells, attempts, equilibria = NamedTuple[], NamedTuple[], NamedTuple[]
    failures = String[]
    for anchor in config.anchors, (e_index, e_to_i) in enumerate(e_values)
        control_context = "$(anchor.name)_e$(e_index)_control"
        control = models_at(config, anchor, e_to_i, first(threshold_values)).control
        control_result = run_search(config, control, control_context, output)
        if isnothing(control_result)
            push!(failures, control_context)
        else
            append!(attempts, [summary_attempt(control_context, i, attempt)
                for (i, attempt) in enumerate(control_result.attempts)])
        end
        for (threshold_index, failure_threshold) in enumerate(threshold_values)
            models = models_at(config, anchor, e_to_i, failure_threshold)
            base = (; anchor=anchor.name, e_to_e=anchor.e_to_e, i_to_e=anchor.i_to_e,
                e_to_i, i_to_i=anchor.i_to_i, failure_threshold,
                high_E_failure_guide=e_to_i - 2failure_threshold)
            for condition in CONDITIONS
                cell_id = "$(anchor.name)_e$(e_index)_f$(threshold_index)_$(condition)"
                model = getproperty(models, condition)
                context_id = condition == :control ? control_context : cell_id
                result = condition == :control ? control_result : run_search(config, model, context_id, output)
                if condition != :control
                    if isnothing(result)
                        push!(failures, context_id)
                    else
                        append!(attempts, [summary_attempt(context_id, i, attempt)
                            for (i, attempt) in enumerate(result.attempts)])
                    end
                end
                found = isnothing(result) ? [] : result.equilibria
                for (index, equilibrium) in enumerate(found)
                    push!(equilibria, merge((; cell_id, condition=string(condition)), base,
                        summary_equilibrium(context_id, index, equilibrium),
                        equilibrium_observations(model, equilibrium)))
                end
                n = predicate -> isnothing(result) ? -1 : count(predicate, found)
                push!(cells, merge((; cell_id, context_id, condition=string(condition)), base,
                    (; status=isnothing(result) ? "execution_failed" : "completed",
                        discovered_equilibria=isnothing(result) ? -1 : length(found),
                        attracting_equilibria=n(eq -> eq.stability.classification == Attracting),
                        saddle_equilibria=n(eq -> eq.stability.classification == Saddle),
                        repelling_equilibria=n(eq -> eq.stability.classification == Repelling),
                        stability_unresolved=n(eq -> eq.stability.classification == StabilityUnresolved),
                        attracting_descending_equilibria=n(eq -> eq.stability.classification == Attracting &&
                            equilibrium_observations(model, eq).descending_response_branch),
                        attempts=isnothing(result) ? -1 : length(result.attempts),
                        rejected_attempts=isnothing(result) ? -1 : count(a -> a.validation == RejectedCandidate, result.attempts),
                        boundary_ambiguous_attempts=isnothing(result) ? -1 : count(a -> a.validation == BoundaryAmbiguousCandidate, result.attempts),
                        completeness=string(CompletenessNotCertified),
                        periodic_orbit_status="not_tested", biological_interpretation="not_assigned")))
            end
        end
    end
    success = isempty(failures)
    metadata["execution_success"] = success
    metadata["failed_search_contexts"] = failures
    metadata["e_to_i_values"] = e_values
    metadata["failure_threshold_values"] = threshold_values
    metadata["cell_count"] = length(cells)
    metadata["unique_search_count"] = length(config.anchors) * length(e_values) * (1 + length(threshold_values))
    write_toml(joinpath(output, "metadata.toml"), metadata)
    CSV.write(joinpath(output, "map.csv"), cells)
    write_rows(joinpath(output, "attempts.csv"), attempts,
        (:context_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I, :solver_status,
         :solver_success, :residual_norm, :validation, :near_singular, :reasons))
    write_rows(joinpath(output, "equilibria.csv"), equilibria,
        (:cell_id, :condition, :anchor, :e_to_e, :i_to_e, :e_to_i, :i_to_i, :failure_threshold,
         :high_E_failure_guide, :context_id, :equilibrium, :E, :I, :residual_norm, :near_singular,
         :representative_attempt, :member_attempts, :stability, :geometry,
         :eigenvalue_1_real, :eigenvalue_1_imaginary, :eigenvalue_2_real,
         :eigenvalue_2_imaginary, :spectral_abscissa, :u_I, :F_I, :F_I_prime,
         :inhibitory_response_midpoint, :inhibitory_branch, :descending_response_branch))
    artifact_checksums(output)
    return (; success, cells)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "coexistence.toml")
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
            continue
        end
        option in ("--config", "--output") || throw(ArgumentError("unknown option: $option"))
        index < length(args) || throw(ArgumentError("$option requires a value"))
        value = args[index + 1]
        startswith(value, "--") && throw(ArgumentError("$option requires a value"))
        option == "--config" ? (config_path = value) : (output = value)
        index += 2
    end
    isnothing(output) && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output; smoke)
    println("Wrote $(length(result.cells)) matched cells; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(CoexistenceExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
