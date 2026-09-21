"""Independent matched-model intervention and targeted robustness equilibrium searches."""
module InterventionExperiment

using FailureOfInhibition2025
import CSV
import TOML

include(joinpath(@__DIR__, "run_minimal_experiment.jl"))
using .MinimalExperiment: require_keys, finite_number, positive_integer, options_from,
    write_toml, write_rows, archive_provenance, artifact_checksums, file_hash,
    error_record, context_record, summary_attempt, summary_equilibrium

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const PARAMETER_NAMES = (:e_to_e, :i_to_e, :e_to_i, :i_to_i, :tau_e, :tau_i,
    :a_e, :a_i, :theta_e, :theta_on, :theta_off)
const INTERVENTION_AXES = (:theta_off, :e_to_i, :e_to_e, :i_to_e)
const ROBUSTNESS_AXES = (:e_to_e, :i_to_e, :i_to_i, :tau_ratio, :a_e, :a_i,
    :theta_e, :theta_on)
const JOINT_BASES = (2, 3, 5, 7, 11, 13, 17, 19)
const CONDITIONS = (:control, :failure_of_inhibition)

"""Construct matched control/FoI models for a cell, optionally with a pulse drive."""
function models_for(cell; drive=NoDrive())
    p = cell.parameters
    return matched_point_models(
        excitatory=PopulationParameters(timescale=p.tau_e,
            response=LogisticResponse(slope=p.a_e, threshold=p.theta_e)),
        inhibitory_control=PopulationParameters(timescale=p.tau_i,
            response=LogisticResponse(slope=p.a_i, threshold=p.theta_on)),
        failure_threshold=p.theta_off,
        coupling=PointCoupling(e_to_e=p.e_to_e, i_to_e=p.i_to_e,
            e_to_i=p.e_to_i, i_to_i=p.i_to_i), drive=drive)
end

function changed_parameters(parameters, axis, value)
    if axis == :tau_ratio
        return merge(parameters, (tau_i=parameters.tau_e * value,))
    end
    return merge(parameters, NamedTuple{(axis,)}((value,)))
end

function validate_axes(raw_axes, allowed, baseline, name)
    raw_axes isa AbstractVector && !isempty(raw_axes) ||
        throw(ArgumentError("$name must be a nonempty array of tables"))
    axes = map(raw_axes) do raw
        require_keys(raw, ("name", "values"), name)
        raw["name"] isa AbstractString || throw(ArgumentError("axis name must be a string"))
        axis = Symbol(raw["name"])
        axis in allowed || throw(ArgumentError("unsupported $name axis: $axis"))
        raw["values"] isa AbstractVector && !isempty(raw["values"]) ||
            throw(ArgumentError("$name.$axis.values must be a nonempty array"))
        values = [finite_number(v, "$name.$axis.values") for v in raw["values"]]
        length(unique(values)) == length(values) ||
            throw(ArgumentError("$name.$axis.values must be unique"))
        for value in values
            models_for((parameters=changed_parameters(baseline, axis, value),))
        end
        (; name=axis, values)
    end
    length(unique(a.name for a in axes)) == length(axes) ||
        throw(ArgumentError("$name axis names must be unique"))
    return axes
end

"""Validate the exploration configuration before creating any output files."""
function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    require_keys(raw, ("schema_version", "baseline", "interventions", "robustness",
        "search", "equilibrium", "stability"), "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema_version"))
    require_keys(raw["baseline"], string.(PARAMETER_NAMES), "baseline")
    baseline = NamedTuple{PARAMETER_NAMES}(Tuple(finite_number(raw["baseline"][string(k)],
        "baseline.$k") for k in PARAMETER_NAMES))
    models_for((parameters=baseline,))
    interventions = validate_axes(raw["interventions"], INTERVENTION_AXES, baseline,
        "interventions")
    require_keys(raw["robustness"], ("axes", "joint_samples"), "robustness")
    robustness = validate_axes(raw["robustness"]["axes"], ROBUSTNESS_AXES, baseline,
        "robustness.axes")
    joint_samples = positive_integer(raw["robustness"]["joint_samples"],
        "robustness.joint_samples")
    require_keys(raw["search"], ("grid_points",), "search")
    grid_points = positive_integer(raw["search"]["grid_points"], "search.grid_points")
    grid_points >= 2 || throw(ArgumentError("search.grid_points must be at least 2"))
    equilibrium_options = options_from(raw["equilibrium"], EquilibriumOptions,
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium"; integer_key="maxiters")
    stability_options = options_from(raw["stability"], StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")
    config = (; raw, baseline, interventions, robustness, joint_samples, grid_points,
        equilibrium_options, stability_options)
    # Also validate joint combinations, since independently admissible axes can interact.
    for cell in robustness_cells(config)
        models_for(cell)
    end
    return config
end

function axis_cells(baseline, axes, family; smoke=false)
    cells = NamedTuple[]
    for axis in axes
        indices = smoke ? unique([firstindex(axis.values), lastindex(axis.values)]) :
            eachindex(axis.values)
        for index in indices
            value = axis.values[index]
            parameters = changed_parameters(baseline, axis.name, value)
            push!(cells, (; id="$(family)_$(axis.name)_$(lpad(index, 3, '0'))",
                family, axis=string(axis.name), value, parameters))
        end
    end
    return cells
end

"""Enumerate independent baseline and individual-intervention cells."""
function intervention_cells(config; smoke=false)
    baseline = (id="baseline", family="baseline", axis="none", value=0.0,
        parameters=config.baseline)
    return vcat([baseline], axis_cells(config.baseline, config.interventions,
        "intervention"; smoke))
end

function radical_inverse(index::Integer, base::Integer)
    value, weight = 0.0, inv(Float64(base))
    while index > 0
        index, digit = divrem(index, base)
        value += digit * weight
        weight /= base
    end
    return value
end

"""
Enumerate targeted one-axis slices and deterministic joint samples.

Joint sample `n` chooses node `1 + floor(length(values)*radical_inverse(n,base))`
from each configured axis. Bases are the first eight primes in configured axis
order. These are reproducible exploration nodes, not physiological distributions.
"""
function robustness_cells(config; smoke=false)
    cells = axis_cells(config.baseline, config.robustness, "robustness_slice"; smoke)
    for index in 1:(smoke ? min(2, config.joint_samples) : config.joint_samples)
        parameters = config.baseline
        for (axis_index, axis) in enumerate(config.robustness)
            node = min(length(axis.values), 1 + floor(Int,
                length(axis.values) * radical_inverse(index, JOINT_BASES[axis_index])))
            parameters = changed_parameters(parameters, axis.name, axis.values[node])
        end
        push!(cells, (; id="robustness_joint_$(lpad(index, 3, '0'))",
            family="robustness_joint", axis="joint", value=Float64(index), parameters))
    end
    return cells
end

function deterministic_seeds(model, grid_points)
    seeds = default_equilibrium_seeds(model)
    upper_e = maximum(seed[1] for seed in seeds)
    upper_i = maximum(seed[2] for seed in seeds)
    append!(seeds, ([e, i] for e in range(0.0, upper_e; length=grid_points)
        for i in range(0.0, upper_i; length=grid_points)))
    return unique!(seeds)
end

"""Continuous equilibrium observations and mathematical inhibitory-response branch."""
function observations(model, state)
    E, I = state
    coupling = model.coupling
    drive_e, drive_i = drive_value(model.drive, 0.0)
    u_E = coupling.e_to_e * E - coupling.i_to_e * I + drive_e
    u_I = coupling.e_to_i * E - coupling.i_to_i * I + drive_i
    inhibitory = model.inhibitory.response
    branch = if inhibitory isa FailureOfInhibitionResponse
        midpoint = inhibitory.onset_threshold / 2 + inhibitory.failure_threshold / 2
        u_I < midpoint ? "ascending" : u_I > midpoint ? "descending" : "midpoint"
    else
        "monotone"
    end
    return (; u_E, u_I, F_I=response(inhibitory, u_I),
        F_I_prime=response_derivative(inhibitory, u_I), inhibitory_response_branch=branch)
end

function archive_run(config_path, output, config; smoke, robustness)
    metadata = archive_provenance(config_path, output)
    relative = "scripts/run_intervention_experiment.jl"
    destination = joinpath(output, "source", relative)
    cp(joinpath(REPOSITORY_ROOT, relative), destination)
    metadata["source_sha256"][relative] = file_hash(destination)
    metadata["purpose"] = "independent matched-model intervention and targeted robustness equilibrium discovery"
    metadata["smoke"] = smoke
    metadata["robustness_enabled"] = robustness
    metadata["equilibrium_seed_policy"] = "union of default 5-by-5 grid and configured $(config.grid_points)-by-$(config.grid_points) sharper-rectangle grid; fresh seeds per cell"
    metadata["equilibrium_options"] = config.equilibrium_options
    metadata["stability_options"] = config.stability_options
    metadata["joint_sampling"] = Dict("rule" => "select configured node using 1 + floor(node_count * radical_inverse(sample_index, prime_base))",
        "sample_indices" => "1 through joint_samples", "prime_bases" => collect(JOINT_BASES),
        "axis_order" => string.([axis.name for axis in config.robustness]),
        "physiological_distribution" => "not_assigned")
    metadata["biological_interpretation"] = "not_assigned"
    metadata["branch_correspondence"] = "not_assigned; cell-local equilibrium indices do not identify branches across cells"
    metadata["periodic_orbit_status"] = "not_tested"
    metadata["induction_rescue_thresholds"] = "not_tested; run explicit pulse experiments at selected cells"
    metadata["absence_inference"] = "no discovery does not establish absence of an equilibrium or attractor"
    metadata["replay_from_artifact_directory"] = "julia --project=source source/scripts/run_intervention_experiment.jl --config config.toml --output replay" *
        (smoke ? " --smoke" : "") * (robustness ? " --robustness" : "")
    metadata["artifact_schema"] = Dict("version" => 1,
        "cells" => "one row per independently searched parameter cell and condition",
        "equilibria" => "one row per locally validated equilibrium, with continuous observations and local stability",
        "attempts" => "one row per seed including rejected or unresolved solves",
        "contexts" => "complete search result and parameters; completeness never certified",
        "unavailable_count" => "-1 when a search throws before results are available",
        "toml_unavailable_value" => "not_available")
    return metadata
end

"""
Run individual interventions and optional robustness slices/joint samples.

All roots, including saddles and repellers, are retained. Outputs report local
equilibrium evidence without assigning functional/seizure labels or matching
branches by nearest distance. Existing nonempty output directories are refused.
"""
function run_experiment(config_path::AbstractString, output_dir::AbstractString;
    smoke=false, robustness=false)
    config = load_config(config_path)
    cells = intervention_cells(config; smoke)
    robustness && append!(cells, robustness_cells(config; smoke))
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be an absent or empty directory"))
    mkpath(joinpath(output, "contexts"))
    metadata = archive_run(config_path, output, config; smoke, robustness)
    cell_rows, attempts, equilibria = NamedTuple[], NamedTuple[], NamedTuple[]
    success = true
    for cell in cells
        models = models_for(cell)
        for condition in CONDITIONS
            model = getproperty(models, condition)
            context_id = "$(cell.id)_$(condition)"
            seeds = deterministic_seeds(model, config.grid_points)
            status, discovered, attracting, descending = "completed", 0, 0, 0
            failed_attempts, error_message = 0, ""
            record = Dict{String,Any}("context_id" => context_id,
                "cell" => cell, "condition" => string(condition), "requested_seeds" => seeds,
                "biological_interpretation" => "not_assigned",
                "branch_correspondence" => "not_assigned",
                "completeness" => CompletenessNotCertified)
            try
                result = find_equilibria(model; seeds, options=config.equilibrium_options,
                    stability_options=config.stability_options)
                merge!(record, context_record(result))
                discovered = length(result.equilibria)
                failed_attempts = count(attempt -> !attempt.solver_success, result.attempts)
                append!(attempts, [summary_attempt(context_id, i, attempt)
                    for (i, attempt) in enumerate(result.attempts)])
                for (index, equilibrium) in enumerate(result.equilibria)
                    observed = observations(model, equilibrium.state)
                    is_attracting = equilibrium.stability.classification == Attracting
                    attracting += is_attracting
                    descending += is_attracting && observed.inhibitory_response_branch == "descending"
                    push!(equilibria, merge(summary_equilibrium(context_id, index, equilibrium),
                        (; cell_id=cell.id, condition=string(condition)), observed))
                end
            catch error
                error isa InterruptException && rethrow()
                success = false
                status = "execution_failed"
                discovered, attracting, descending, failed_attempts = -1, -1, -1, -1
                failure = error_record(error)
                error_message = failure["message"]
                record["error"] = failure
                record["status"] = status
            end
            write_toml(joinpath(output, "contexts", context_id * ".toml"), record)
            push!(cell_rows, merge((; context_id, cell_id=cell.id, family=cell.family,
                axis=cell.axis, value=cell.value, condition=string(condition), status,
                discovered_equilibria=discovered, locally_attracting_equilibria=attracting,
                attracting_descending_response=descending, failed_attempts,
                completeness="CompletenessNotCertified", error_message), cell.parameters))
        end
    end
    CSV.write(joinpath(output, "cells.csv"), cell_rows)
    write_rows(joinpath(output, "attempts.csv"), attempts,
        (:context_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I, :solver_status,
         :solver_success, :residual_norm, :validation, :near_singular, :reasons))
    write_rows(joinpath(output, "equilibria.csv"), equilibria,
        (:context_id, :equilibrium, :E, :I, :residual_norm, :near_singular,
         :representative_attempt, :member_attempts, :stability, :geometry,
         :eigenvalue_1_real, :eigenvalue_1_imaginary, :eigenvalue_2_real,
         :eigenvalue_2_imaginary, :spectral_abscissa, :cell_id, :condition,
         :u_E, :u_I, :F_I, :F_I_prime, :inhibitory_response_branch))
    metadata["execution_success"] = success
    metadata["cell_count"] = length(cells)
    metadata["search_count"] = length(cell_rows)
    write_toml(joinpath(output, "metadata.toml"), metadata)
    artifact_checksums(output)
    return (; success, cells=cell_rows, equilibria)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "interventions.toml")
    output, smoke, robustness = nothing, false, false
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option in ("--smoke", "--robustness")
            option == "--smoke" ? (smoke = true) : (robustness = true)
        elseif option in ("--config", "--output")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            index += 1
            option == "--config" ? (config_path = args[index]) : (output = args[index])
        else
            throw(ArgumentError("unknown option: $option"))
        end
        index += 1
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output; smoke, robustness)
    println("Wrote $(length(result.cells)) matched-condition searches; execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(InterventionExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
