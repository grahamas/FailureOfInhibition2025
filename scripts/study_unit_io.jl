"""Shared checkpoint and replay command handling for staged studies."""
module StudyUnitIO
import TOML
include("study_support.jl")
const Evidence = StudySupport

function write_record(path,data)
    mkpath(dirname(path))
    Evidence.write_toml(path*".tmp",data)
    mv(path*".tmp",path;force=true)
end

function checked_unit(dir)
    marker=joinpath(dir,"done.toml")
    isfile(marker) || return false
    files=TOML.parsefile(marker)["files"]
    for (relative,hash) in files
        isabspath(relative) || ".." in splitpath(relative) ? throw(ArgumentError("invalid manifest path")) : nothing
        isfile(joinpath(dir,relative)) && Evidence.file_hash(joinpath(dir,relative))==hash ||
            throw(ArgumentError("checkpoint checksum mismatch: $relative"))
    end
    return true
end

function complete_unit(dir)
    files=Dict{String,String}()
    for (root,_,names) in walkdir(dir), name in names
        name in ("done.toml","done.toml.tmp") && continue
        files[relpath(joinpath(root,name),dir)]=Evidence.file_hash(joinpath(root,name))
    end
    write_record(joinpath(dir,"done.toml"),Dict("files"=>files))
end

shell_quote(value::AbstractString) = "'" * replace(value, "'" => "'\\''") * "'"

function replay_command(script; stage="all", case_filter=nothing, smoke=false)
    command="julia --project=source source/scripts/$script --config config.toml --output replay --stage $stage"
    case_filter===nothing || (command *= " --case " * shell_quote(case_filter))
    smoke && (command *= " --smoke")
    command
end

function complete_replay!(metadata, script; stage, case_filter, smoke)
    commands=get!(metadata,"replay_invocations",String[])
    push!(commands,replay_command(script;stage,case_filter,smoke))
    metadata["replay_from_artifact_directory"]=join(commands," && ")
    metadata
end

end
