module ContinuationExperiment

using FailureOfInhibition2025
import CSV
include("run_coexistence_map.jl")
const Map = CoexistenceExperiment
const Evidence = Map.MinimalExperiment

"""Continue all discovered representative roots along both primary axes.

The supplied coexistence configuration determines the models, seed coverage,
equilibrium tolerances, and parameter bounds. Both traversal directions and
unstable roots are retained. Candidate brackets are numerical screening
evidence; they are not certified bifurcations or proof of complete coverage.
"""
function run_experiment(config_path, output_dir; smoke=false)
    config = Map.load_config(config_path)
    first(config.e_to_i_values) <= 19.0 <= last(config.e_to_i_values) &&
        first(config.failure_threshold_values) <= 8.0 <= last(config.failure_threshold_values) ||
        throw(ArgumentError("representative continuation requires e_to_i=19 and failure_threshold=8 within configured bounds"))
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    mkpath(joinpath(output, "branches"))
    metadata = Evidence.archive_provenance(config_path, output)
    for name in ("run_coexistence_map.jl", "run_continuation_experiment.jl")
        relative = joinpath("scripts", name)
        destination = joinpath(output, "source", relative)
        cp(joinpath(@__DIR__, name), destination)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
    metadata["purpose"] = "representative equilibrium branch continuation; no completeness certification"
    metadata["smoke"] = smoke
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_continuation_experiment.jl --config config.toml --output replay" *
        (smoke ? " --smoke" : "")
    metadata["artifact_schema"] = Dict("points" => "branch-local traversal indices and raw state/spectral evidence",
        "branches" => "all corrector attempts, candidate brackets, termination, and source models")
    options = ContinuationOptions(max_steps=smoke ? 8 : 500)
    metadata["continuation_options"] = options
    metadata["equilibrium_seed_policy"] = "union of default 5-by-5 and configured sharper-rectangle grids"
    rows = NamedTuple[]
    summaries = NamedTuple[]
    success = true
    for anchor in config.anchors, condition in (:control, :failure_of_inhibition)
        model = getproperty(Map.models_at(config, anchor, 19.0, 8.0), condition)
        seeds = Map.deterministic_seeds(model, config.seed_grid_points)
        search = find_equilibria(model; seeds, options=config.equilibrium_options,
            stability_options=config.stability_options)
        Evidence.write_toml(joinpath(output, "branches", "$(anchor.name)_$(condition)_search.toml"),
            Evidence.context_record(search))
        for axis in (:e_to_i, :failure_threshold)
            bounds = extrema(axis === :e_to_i ? config.e_to_i_values : config.failure_threshold_values)
            initial_parameter = axis === :e_to_i ? 19.0 : 8.0
            factory = p -> getproperty(Map.models_at(config, anchor,
                axis === :e_to_i ? p : 19.0, axis === :failure_threshold ? p : 8.0), condition)
            for (root_id, equilibrium) in enumerate(search.equilibria)
                id = "$(anchor.name)_$(condition)_$(axis)_root$(root_id)"
                try
                    result = continue_equilibria(factory, equilibrium.state, initial_parameter;
                        parameter_bounds=bounds, options,
                        equilibrium_options=config.equilibrium_options,
                        stability_options=config.stability_options)
                    # Factories are executable closures, not serializable provenance.
                    # Every branch point retains its actual model and initial context.
                    Evidence.write_toml(joinpath(output, "branches", id * ".toml"),
                        Dict("initial_solve" => result.initial_solve,
                            "initial_parameter" => result.initial_parameter,
                            "parameter_bounds" => result.parameter_bounds,
                            "options" => result.options,
                            "negative" => result.negative, "positive" => result.positive,
                            "completeness" => result.completeness))
                    for branch in (result.negative, result.positive)
                        push!(summaries, (; id, anchor=anchor.name, condition=string(condition),
                            axis=string(axis), root_id, direction=branch.direction,
                            points=length(branch.points), attempts=length(branch.attempts),
                            candidates=length(branch.candidates), termination=string(branch.termination)))
                        for (index, point) in enumerate(branch.points)
                            e, i = point.state
                            u_i = point.model.coupling.e_to_i * e - point.model.coupling.i_to_i * i
                            stability = point.stability
                            push!(rows, (; id, direction=branch.direction, point=index,
                                parameter=point.parameter, E=e, I=i, u_I=u_i,
                                F_I_prime=response_derivative(point.model.inhibitory.response, u_i),
                                residual_norm=point.equilibrium_attempt.residual_norm,
                                tangent_parameter=point.tangent[3],
                                stability=string(stability.classification),
                                spectral_abscissa=stability.spectral_abscissa,
                                trace=stability.trace, determinant=stability.determinant))
                        end
                    end
                catch error
                    error isa InterruptException && rethrow()
                    success = false
                    Evidence.write_toml(joinpath(output, "branches", id * "_failure.toml"),
                        Evidence.error_record(error))
                end
            end
        end
    end
    Evidence.write_rows(joinpath(output, "points.csv"), rows,
        (:id, :direction, :point, :parameter, :E, :I, :u_I, :F_I_prime,
         :residual_norm, :tangent_parameter, :stability, :spectral_abscissa, :trace, :determinant))
    Evidence.write_rows(joinpath(output, "branches.csv"), summaries,
        (:id, :anchor, :condition, :axis, :root_id, :direction, :points, :attempts, :candidates, :termination))
    metadata["execution_success"] = success
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success, branches=length(summaries), points=length(rows))
end

function main(args=ARGS)
    config = joinpath(@__DIR__, "..", "experiments", "coexistence.toml")
    output = joinpath(@__DIR__, "..", "output", "continuation")
    smoke = false
    index = 1
    while index <= length(args)
        if args[index] == "--smoke"
            smoke = true
        elseif args[index] in ("--config", "--output")
            index < length(args) || throw(ArgumentError("missing value for $(args[index])"))
            args[index] == "--config" ? (config = args[index + 1]) : (output = args[index + 1])
            index += 1
        else
            throw(ArgumentError("unknown argument $(args[index])"))
        end
        index += 1
    end
    result = run_experiment(config, output; smoke)
    println("Continuation: $(result.branches) traversals, $(result.points) points; success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(ContinuationExperiment.main())
end
