"""Shared validation, serialization, and provenance for finite study runners."""
module StudySupport

using FailureOfInhibition2025
using LinearAlgebra: BLAS
import CSV
import SHA
import TOML

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

function require_keys(table, keys, name)
    table isa AbstractDict || throw(ArgumentError("$name must be a table"))
    Set(Base.keys(table)) == Set(keys) || throw(ArgumentError(
        "$name requires exactly these keys: $(join(sort(collect(keys)), ", "))",
    ))
end

function finite_number(value, name; positive=false, nonnegative=false)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("$name must be a finite real number"))
    converted = Float64(value)
    isfinite(converted) || throw(ArgumentError("$name must be representable as Float64"))
    positive && converted <= 0 && throw(ArgumentError("$name must be positive"))
    nonnegative && converted < 0 && throw(ArgumentError("$name must be nonnegative"))
    return converted
end

function pair(value, name)
    value isa AbstractVector && length(value) == 2 ||
        throw(ArgumentError("$name must contain exactly two numbers"))
    return Tuple(finite_number(x, name) for x in value)
end

function positive_integer(value, name)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) ||
        throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end

function options_from(table, constructor, keys, name; integer_key=nothing)
    require_keys(table, keys, name)
    entries = Pair{Symbol,Any}[]
    for key in keys
        value = key == integer_key ? positive_integer(table[key], "$name.$key") :
                finite_number(table[key], "$name.$key"; nonnegative=true)
        push!(entries, Symbol(key) => value)
    end
    return constructor(; entries...)
end
toml_value(x::Union{Symbol,Enum}) = string(x)
toml_value(::Nothing) = "not_available"
toml_value(x::Complex) = Dict("real" => real(x), "imaginary" => imag(x))
toml_value(x::AbstractMatrix) = [toml_value(collect(row)) for row in eachrow(x)]
toml_value(x::Union{AbstractVector,Tuple}) = [toml_value(v) for v in x]
toml_value(x::Union{Real,AbstractString}) = x
toml_value(x::AbstractDict) = Dict(string(k) => toml_value(v) for (k, v) in x)
toml_value(x) = Dict(string(k) => toml_value(getproperty(x, k)) for k in propertynames(x))

function write_toml(path, data)
    open(path, "w") do io
        TOML.print(io, toml_value(data); sorted=true)
    end
end

error_record(error) = Dict("type" => string(typeof(error)), "message" => sprint(showerror, error))

function source_files(root)
    files = ["Project.toml", "Manifest.toml", "scripts/run_minimal_experiment.jl", "scripts/study_support.jl"]
    for (directory, _, names) in walkdir(joinpath(root, "src"))
        for name in names
            endswith(name, ".jl") && push!(files, relpath(joinpath(directory, name), root))
        end
    end
    isfile(joinpath(root, "docs/model.md")) && push!(files, "docs/model.md")
    return sort!(files)
end

file_hash(path) = bytes2hex(SHA.sha256(read(path)))

"""Archive explicit repository-relative sources and record their copied hashes."""
function archive_sources!(metadata, output, relatives; root=REPOSITORY_ROOT,
    force=false, missing_only=false)
    for relative in relatives
        relative isa AbstractString && !isabspath(relative) &&
            !(".." in splitpath(relative)) || throw(ArgumentError("invalid source path: $relative"))
        destination = joinpath(output, "source", relative)
        mkpath(dirname(destination))
        if !missing_only || !isfile(destination)
            cp(joinpath(root, relative), destination; force)
        end
        metadata["source_sha256"][relative] = file_hash(destination)
    end
    return metadata
end

function git_output(root, arguments...)
    try
        return strip(read(`git --no-optional-locks -C $root $arguments`, String))
    catch error
        return "unavailable: " * sprint(showerror, error)
    end
end

function archive_provenance(config_path, output)
    files = source_files(REPOSITORY_ROOT)
    hashes = Dict{String,String}()
    archive_sources!(Dict("source_sha256" => hashes), output, files)
    cp(config_path, joinpath(output, "config.toml"))
    return Dict(
        "schema_version" => 1,
        "purpose" => "synthetic workflow validation; no biological regime inference",
        "julia_version" => string(VERSION), "architecture" => string(Sys.ARCH),
        "kernel" => string(Sys.KERNEL), "julia_threads" => Threads.nthreads(),
        "blas_configuration" => string(BLAS.get_config()),
        "blas_threads" => BLAS.get_num_threads(),
        "ode_solver" => "OrdinaryDiffEqTsit5.Tsit5",
        "equilibrium_solver" => "SimpleNonlinearSolve.SimpleTrustRegion",
        "equilibrium_seed_policy" => "default deterministic 5-by-5 sharper-rectangle grid",
        "git_revision" => git_output(REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "git_status_porcelain" => git_output(REPOSITORY_ROOT, "status", "--porcelain=v1", "--untracked-files=all"),
        "source_sha256" => hashes,
        "config_sha256" => file_hash(joinpath(output, "config.toml")),
        "replay_from_artifact_directory" => "julia --project=source source/scripts/run_minimal_experiment.jl --config config.toml --output replay",
        "artifact_schema" => Dict("version" => 1,
            "trajectory_columns" => ["time", "E", "I"],
            "toml_unavailable_value" => "not_available",
            "matrices" => "array of rows", "complex_values" => "real and imaginary tables",
            "contexts" => "complete search attempts, deduplication, equilibria and stability",
            "cases" => "one row per condition, protocol and numerical variant",
            "comparisons" => "each non-default case compared with its default counterpart"),
    )
end

function context_record(result)
    return Dict("status" => "completed", "source_time" => result.source_time,
        "frozen_drive" => result.frozen_drive, "options" => result.options,
        "stability_options" => result.stability_options, "attempts" => result.attempts,
        "equilibria" => result.equilibria, "unresolved_nearby" => result.unresolved_nearby,
        "completeness" => result.completeness)
end

function summary_attempt(context_id, index, attempt)
    return (; context_id, attempt=index, seed_E=attempt.seed[1], seed_I=attempt.seed[2],
        candidate_E=attempt.candidate[1], candidate_I=attempt.candidate[2],
        solver_status=string(attempt.solver_status), solver_success=attempt.solver_success,
        residual_norm=attempt.residual_norm, validation=string(attempt.validation),
        near_singular=attempt.near_singular, reasons=join(string.(attempt.reasons), ";"))
end

function summary_equilibrium(context_id, index, equilibrium)
    stability = equilibrium.stability
    return (; context_id, equilibrium=index, E=equilibrium.state[1], I=equilibrium.state[2],
        residual_norm=maximum(abs, equilibrium.balance_residual),
        near_singular=equilibrium.near_singular,
        representative_attempt=equilibrium.representative_attempt,
        member_attempts=join(equilibrium.member_attempts, ";"),
        stability=string(stability.classification), geometry=string(stability.geometry),
        eigenvalue_1_real=real(stability.eigenvalues[1]),
        eigenvalue_1_imaginary=imag(stability.eigenvalues[1]),
        eigenvalue_2_real=real(stability.eigenvalues[2]),
        eigenvalue_2_imaginary=imag(stability.eigenvalues[2]),
        spectral_abscissa=stability.spectral_abscissa)
end
function write_rows(path, rows, empty_columns)
    if isempty(rows)
        CSV.write(path, NamedTuple{Tuple(empty_columns)}(Tuple(String[] for _ in empty_columns)))
    else
        CSV.write(path, rows)
    end
end
function artifact_checksums(output)
    hashes = Dict{String,String}()
    for (directory, _, files) in walkdir(output), filename in files
        path = joinpath(directory, filename)
        relative = relpath(path, output)
        relative == "checksums.toml" && continue
        hashes[relative] = file_hash(path)
    end
    write_toml(joinpath(output, "checksums.toml"), Dict("schema_version" => 1,
        "algorithm" => "SHA-256", "files" => hashes))
end

"""Match attracting roots by Euclidean state distance, rejecting ties and collisions."""
function match_attracting_roles(search, references, tolerance)
    K = keytype(typeof(references))
    choices = Dict{K,Union{Nothing,Int}}()
    reasons = Dict{K,Symbol}()
    for (role, reference) in references
        if reference === nothing
            choices[role], reasons[role] = nothing, :tracking_lost
            continue
        end
        candidates = [(index, hypot(root.state[1] - reference[1],
                root.state[2] - reference[2]))
            for (index, root) in enumerate(search.equilibria)
            if root.stability.classification == Attracting]
        sort!(candidates; by=last)
        if isempty(candidates) || first(candidates)[2] > tolerance
            choices[role], reasons[role] = nothing, :not_matched
        elseif length(candidates) > 1 && candidates[2][2] - candidates[1][2] < 1e-6
            choices[role], reasons[role] = nothing, :ambiguous
        else
            choices[role], reasons[role] = first(candidates)[1], :matched
        end
    end
    selected = [index for index in values(choices) if index !== nothing]
    for role in keys(choices)
        index = choices[role]
        if index !== nothing && count(==(index), selected) > 1
            choices[role], reasons[role] = nothing, :ambiguous
        end
    end
    return choices, reasons
end

"""Accept a root only when coordinate matching is unique in both directions."""
function unique_reciprocal_match(reference, references, candidates, tolerance, distance)
    matches = findall(candidate -> distance(reference, candidate) <= tolerance, candidates)
    length(matches) == 1 || return nothing
    selected = only(matches)
    count(other -> distance(other, candidates[selected]) <= tolerance,
        references) == 1 || return nothing
    return selected
end

end
