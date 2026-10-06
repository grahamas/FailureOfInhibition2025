"""Apply narrowly specified storage or replay-metadata repairs to a figure study."""
module RepairFigureSerialization
import TOML, SHA
const ROOT=normpath(joinpath(@__DIR__,".."))
digest(path)=bytes2hex(open(SHA.sha256,path))

function repair(output;kind="serialization")
    relative="scripts/run_figure_details.jl"
    archived=joinpath(output,"source",relative)
    current=joinpath(ROOT,relative)
    old=read(archived,String);new=read(current,String)
    if kind=="serialization"
        replacement="                attempts=[Dict(string(k)=>(k==:orbit ? (a.orbit===nothing ? nothing : orbit_record(a.orbit)) : getproperty(a,k)) for k in propertynames(a)) for a in branch.attempts]\n                S.write_record(joinpath(dir,\"attempts_\$(branch.direction).toml\"),Dict(\"attempts\"=>attempts))"
        original="                S.write_record(joinpath(dir,\"attempts_\$(branch.direction).toml\"),Dict(\"attempts\"=>branch.attempts))"
        reason="Avoid recursive serialization of solver internals in continuation attempts"
    elseif kind=="replay"
        replacement="    metadata[\"replay_from_artifact_directory\"]=\"julia --project=source source/scripts/run_figure_details.jl config.toml SCREEN_OUTPUT replay STAGE\"\n"
        original=""
        reason="Replace the inherited screen command with the detail-stage replay command; SCREEN_OUTPUT and STAGE are placeholders"
    else
        error("unsupported repair")
    end
    occursin(replacement,new) && replace(new,replacement=>original)==old ||
        error("not the exact storage-only repair; numerical changes require a new study")
    metadata_path=joinpath(output,"metadata.toml")
    metadata=TOML.parsefile(metadata_path)
    for (file,hash) in metadata["source_sha256"]
        digest(joinpath(output,"source",file))==hash || error("archived source mismatch: $file")
        file==relative || digest(joinpath(ROOT,file))==hash || error("other source changed: $file")
    end
    record_key=kind*"_repair"
    history=joinpath(output,record_key)
    ispath(history) && error("repair history already exists")
    mkpath(history)
    cp(archived,joinpath(history,"previous_driver.jl"))
    cp(metadata_path,joinpath(history,"previous_metadata.toml"))
    cp(@__FILE__,joinpath(history,"repair.jl"))
    record=Dict("reason"=>reason,
        "numerical_identity"=>"Exact source equality after reversing the sole named nonnumerical replacement",
        "previous_driver_sha256"=>digest(archived),"current_driver_sha256"=>digest(current),
        "previous_metadata_sha256"=>digest(metadata_path),"repair_sha256"=>digest(@__FILE__))
    open(joinpath(history,"record.toml"),"w") do io;TOML.print(io,record;sorted=true);end
    cp(current,archived;force=true)
    metadata["source_sha256"][relative]=digest(current)
    metadata[record_key]=record
    open(metadata_path*".tmp","w") do io;TOML.print(io,metadata;sorted=true);end
    mv(metadata_path*".tmp",metadata_path;force=true)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (1,2) || error("usage: repair_figure_serialization.jl DETAILS [serialization|replay]")
    RepairFigureSerialization.repair(ARGS[1];kind=length(ARGS)==2 ? ARGS[2] : "serialization")
end
