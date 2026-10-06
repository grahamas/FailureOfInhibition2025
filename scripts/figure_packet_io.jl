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
function files(directory)
    isdir(directory) && !islink(directory) || error("packet must be a regular directory")
    result=Dict{String,String}()
    for (dir,folders,names) in walkdir(directory;follow_symlinks=false)
        any(islink(joinpath(dir,n)) for n in vcat(folders,names)) && error("packet symlink")
        for name in names
            path=joinpath(dir,name);isfile(path) || error("nonregular packet file")
            relative=replace(relpath(path,directory),'\\'=>'/')
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
end
