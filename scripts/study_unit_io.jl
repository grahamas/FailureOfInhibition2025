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
    islink(dir) && throw(ArgumentError("checkpoint directory is a symbolic link"))
    marker=joinpath(dir,"done.toml")
    isfile(marker) || return false
    islink(marker) && throw(ArgumentError("checkpoint marker is a symbolic link"))
    files=TOML.parsefile(marker)["files"]
    files isa AbstractDict || throw(ArgumentError("checkpoint files must be a table"))
    present=Set{String}()
    children=String[]
    for (root,dirs,names) in walkdir(dir)
        any(name->islink(joinpath(root,name)),dirs) &&
            throw(ArgumentError("checkpoint contains a symbolic link"))
        for name in names
            path=joinpath(root,name)
            islink(path) && throw(ArgumentError("checkpoint contains a symbolic link"))
            isfile(path) || throw(ArgumentError("checkpoint contains a non-file"))
            relative=relpath(path,dir)
            name == "done.toml.tmp" && throw(ArgumentError("unfinished checkpoint marker: $relative"))
            if name == "done.toml"
                path == marker || push!(children,root)
            else
                push!(present,relative)
            end
        end
    end
    present == Set(keys(files)) || throw(ArgumentError("checkpoint file set mismatch"))
    for (relative,hash) in files
        relative isa AbstractString && !isabspath(relative) &&
            !(".." in splitpath(relative)) && normpath(relative)==relative ||
            throw(ArgumentError("invalid manifest path"))
        isfile(joinpath(dir,relative)) && Evidence.file_hash(joinpath(dir,relative))==hash ||
            throw(ArgumentError("checkpoint checksum mismatch: $relative"))
    end
    for child in children
        checked_unit(child) || throw(ArgumentError("incomplete nested checkpoint: $child"))
    end
    return true
end

function complete_unit(dir)
    islink(dir) && throw(ArgumentError("checkpoint directory is a symbolic link"))
    files=Dict{String,String}()
    for (root,dirs,names) in walkdir(dir)
        any(name->islink(joinpath(root,name)),dirs) &&
            throw(ArgumentError("checkpoint contains a symbolic link"))
        for name in names
            path=joinpath(root,name)
            islink(path) && throw(ArgumentError("checkpoint contains a symbolic link"))
            isfile(path) || throw(ArgumentError("checkpoint contains a non-file"))
            name == "done.toml.tmp" && throw(ArgumentError("unfinished checkpoint marker: $path"))
            if name == "done.toml"
                root == dir || checked_unit(root) ||
                    throw(ArgumentError("incomplete nested checkpoint: $root"))
                continue
            end
            files[relpath(path,dir)]=Evidence.file_hash(path)
        end
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
