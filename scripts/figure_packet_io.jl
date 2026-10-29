"""Portable figure data integrity; paths and complete file sets are checked."""
module FigurePacketIO
using SHA, TOML
digest(path)=bytes2hex(open(SHA.sha256,path))
function checked_name(name)
    name isa AbstractString && !isempty(name) && !isabspath(name) &&
        !occursin('\\',name) && !occursin(':',name) &&
        all(p->!isempty(p) && p!="." && p!="..",split(name,'/')) || error("unsafe packet path")
    name
end
portable_name(name,separator)=checked_name(replace(name,separator=>'/'))
function files(directory)
    isdir(directory) && !islink(directory) || error("packet must be a regular directory")
    result=Dict{String,String}()
    separator=Sys.iswindows() ? '\\' : '/'
    for (dir,folders,names) in walkdir(directory;follow_symlinks=false)
        any(islink(joinpath(dir,n)) for n in vcat(folders,names)) && error("packet symlink")
        for name in names
            path=joinpath(dir,name);isfile(path) || error("nonregular packet file")
            relative=portable_name(relpath(path,directory),separator)
            relative=="checksums.toml" && continue
            result[checked_name(relative)]=digest(path)
        end
    end
    result
end
"""Write a deterministic SHA-256 manifest for every packet file, atomically."""
function write_manifest(directory)
    data=Dict("algorithm"=>"SHA-256","files"=>files(directory))
    path=joinpath(directory,"checksums.toml")
    temporary,io=mktemp(directory)
    try
        TOML.print(io,data;sorted=true);close(io);mv(temporary,path;force=true)
    finally
        isopen(io) && close(io);isfile(temporary) && rm(temporary)
    end
    path
end
"""Verify exact file-set equality and hashes; return the parsed manifest."""
function verify_manifest(directory)
    path=joinpath(directory,"checksums.toml")
    isfile(path) && !islink(path) || error("missing regular packet manifest")
    data=TOML.parsefile(path)
    get(data,"algorithm",nothing)=="SHA-256" || error("unsupported digest")
    expected=get(data,"files",nothing)
    expected isa AbstractDict || error("missing file map")
    for (name,hash) in expected
        checked_name(name)
        hash isa AbstractString && occursin(r"^[0-9a-f]{64}$",hash) || error("invalid digest")
    end
    files(directory)==expected || error("packet file set or content changed")
    data
end

"""Hash completed-stage markers after checking their exact file sets and contents."""
function completed_stage_manifests(directory, stages, separator)
    result=Dict{String,String}()
    for stage_name in stages
        relative=portable_name(stage_name,separator)
        haskey(result,relative) && error("duplicate completed stage: $relative")
        stage=joinpath(directory,relative)
        isdir(stage) && !islink(stage) || error("missing regular completed stage: $relative")
        marker=joinpath(stage,"done.toml")
        isfile(marker) && !islink(marker) || error("missing regular stage manifest: $relative")
        source=get(TOML.parsefile(marker),"files",nothing)
        source isa AbstractDict || error("missing stage file map: $relative")
        recorded=Dict{String,String}()
        for (name,hash) in source
            canonical=portable_name(name,separator)
            haskey(recorded,canonical) && error("duplicate stage path: $relative/$canonical")
            hash isa AbstractString && occursin(r"^[0-9a-f]{64}$",hash) || error("invalid stage digest: $relative/$name")
            recorded[canonical]=hash
        end
        actual=Dict{String,String}()
        for (root,folders,names) in walkdir(stage;follow_symlinks=false)
            any(islink(joinpath(root,name)) for name in vcat(folders,names)) && error("stage symlink: $relative")
            for name in names
                path=joinpath(root,name)
                file=portable_name(relpath(path,stage),separator)
                file=="done.toml" && continue
                isfile(path) || error("nonregular stage file: $relative")
                actual[file]=digest(path)
            end
        end
        actual==recorded || error("stage file set or content changed: $relative")
        result[relative]=digest(marker)
    end
    result
end
completed_stage_manifests(directory,stages)=
    completed_stage_manifests(directory,stages,Sys.iswindows() ? '\\' : '/')
end
