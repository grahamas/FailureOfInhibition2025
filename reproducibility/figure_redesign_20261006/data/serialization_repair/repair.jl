"""Apply the single storage-only driver repair to an interrupted figure study."""
module RepairFigureSerialization
import TOML, SHA
const ROOT=normpath(joinpath(@__DIR__,".."))
digest(path)=bytes2hex(open(SHA.sha256,path))

function repair(output)
    relative="scripts/run_figure_details.jl"
    archived=joinpath(output,"source",relative)
    current=joinpath(ROOT,relative)
    old=read(archived,String);new=read(current,String)
    replacement="                attempts=[Dict(string(k)=>(k==:orbit ? (a.orbit===nothing ? nothing : orbit_record(a.orbit)) : getproperty(a,k)) for k in propertynames(a)) for a in branch.attempts]\n                S.write_record(joinpath(dir,\"attempts_\$(branch.direction).toml\"),Dict(\"attempts\"=>attempts))"
    original="                S.write_record(joinpath(dir,\"attempts_\$(branch.direction).toml\"),Dict(\"attempts\"=>branch.attempts))"
    occursin(replacement,new) && replace(new,replacement=>original)==old ||
        error("not the exact storage-only repair; numerical changes require a new study")
    metadata_path=joinpath(output,"metadata.toml")
    metadata=TOML.parsefile(metadata_path)
    for (file,hash) in metadata["source_sha256"]
        digest(joinpath(output,"source",file))==hash || error("archived source mismatch: $file")
        file==relative || digest(joinpath(ROOT,file))==hash || error("other source changed: $file")
    end
    history=joinpath(output,"serialization_repair")
    ispath(history) && error("repair history already exists")
    mkpath(history)
    cp(archived,joinpath(history,"previous_driver.jl"))
    cp(metadata_path,joinpath(history,"previous_metadata.toml"))
    cp(@__FILE__,joinpath(history,"repair.jl"))
    record=Dict("reason"=>"Avoid recursive serialization of solver internals in continuation attempts",
        "numerical_identity"=>"Exact source equality after reversing the sole attempt-record serialization replacement",
        "previous_driver_sha256"=>digest(archived),"current_driver_sha256"=>digest(current),
        "previous_metadata_sha256"=>digest(metadata_path),"repair_sha256"=>digest(@__FILE__))
    open(joinpath(history,"record.toml"),"w") do io;TOML.print(io,record;sorted=true);end
    cp(current,archived;force=true)
    metadata["source_sha256"][relative]=digest(current)
    metadata["serialization_repair"]=record
    open(metadata_path*".tmp","w") do io;TOML.print(io,metadata;sorted=true);end
    mv(metadata_path*".tmp",metadata_path;force=true)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: repair_figure_serialization.jl INTERRUPTED_DETAILS")
    RepairFigureSerialization.repair(ARGS[1])
end
