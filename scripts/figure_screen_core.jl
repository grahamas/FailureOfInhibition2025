"""Refresh the frozen paper parameter screen without launching a full response atlas."""
module FigureScreen
import TOML, CSV, SHA
include("input_response_core.jl")
include("archive_input_response_contexts.jl")
const S = InputResponseStudy
const A = InputResponseArchives
const ROOT = normpath(joinpath(@__DIR__, ".."))

function registry_path(config)
    relative = config.raw["redesign"]["registry"]
    relative isa AbstractString && !isempty(relative) && !isabspath(relative) &&
        normpath(relative) == relative && !(".." in splitpath(relative)) ||
        throw(ArgumentError("registry must be a repository-relative file path"))
    source = joinpath(ROOT, relative)
    isfile(source) && !islink(source) &&
        !(".." in splitpath(relpath(realpath(source), realpath(ROOT)))) ||
        throw(ArgumentError("registry must be a regular file inside the repository"))
    return relative
end

function registry(config)
    relative = registry_path(config)
    data = TOML.parsefile(joinpath(ROOT, relative))
    data["schema_version"] == 1 || error("unsupported case registry")
    cases = data["cases"]
    ids = [p["id"] for p in cases]
    length(ids) == length(unique(ids)) || error("duplicate case identifiers")
    for p in cases
        id = p["id"]
        id isa AbstractString && occursin(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$", id) ||
            throw(ArgumentError("case id must be a single safe path component"))
        S.M.model(p)
        p["category"] in ("anchors", "expansion") || error("invalid category")
        for b in p["baselines"]
            length(b) == 2 || error("invalid baseline")
            S.M.model(p,b...)
        end
    end
    cases
end

function initialize(config_path, cfg, output)
    relative_registry = registry_path(cfg)
    metadata = S.initialize(config_path,cfg,output;stage="geometry")
    for relative in ("scripts/run_figure_screen.jl", "scripts/figure_screen_core.jl",
                     "scripts/archive_input_response_contexts.jl",
                     relative_registry)
        hash = S.N.Evidence.file_hash(joinpath(ROOT,relative))
        old = get(metadata["source_sha256"],relative,hash)
        old == hash || error("screen source changed: $relative")
        destination = joinpath(output,"source",relative)
        mkpath(dirname(destination))
        isfile(destination) || cp(joinpath(ROOT,relative),destination)
        S.N.Evidence.file_hash(destination) == hash || error("archived source mismatch")
        metadata["source_sha256"][relative] = hash
    end
    metadata["purpose"] = "Current-source geometry and fixed-protocol refresh of a frozen case list"
    metadata["replay_from_artifact_directory"] = "julia --project=source source/scripts/run_figure_screen.jl config.toml replay"
    S.write_record(joinpath(output,"metadata.toml"),metadata)
    metadata
end

function run(config_path, output; limit=nothing)
    cfg=S.load_config(config_path); cases=registry(cfg)
    limit === nothing || (cases=first(cases,min(limit,length(cases))))
    output=abspath(output); metadata=initialize(config_path,cfg,output)
    rows=S.parallel_cases(cases) do p
        directory=joinpath(output,p["category"],p["id"])
        geometry=S.geometry(p,cfg,joinpath(directory,"geometry");screen=true)
        behavior=S.behavioral_screen(p,cfg,directory)
        # These are freshly generated screen contexts; the historical archive is untouched.
        A.archive_geometry(joinpath(directory,"geometry"))
        S.N.complete_unit(directory)
        println("screen ",p["id"]," inputs=",geometry["inputs"],
                " trials=",behavior["observations"]); flush(stdout)
        (;id=p["id"],category=p["category"],inputs=geometry["inputs"],
          unresolved_cells=geometry["budget_unresolved_cells"],observations=behavior["observations"])
    end
    CSV.write(joinpath(output,"screen.csv"),rows)
    metadata["completed"]=limit===nothing
    metadata["completed_cases"]=length(rows)
    S.write_record(joinpath(output,"metadata.toml"),metadata)
    S.N.Evidence.artifact_checksums(output)
    output
end
end
