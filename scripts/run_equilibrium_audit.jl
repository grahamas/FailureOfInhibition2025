"""Inventory verified current-study equilibria and choose bounded follow-up seeds."""
module EquilibriumAuditStudy
using FailureOfInhibition2025, LinearAlgebra, Tar
import CSV,TOML,SHA
include("equilibrium_role_audit.jl")
include("archive_input_response_contexts.jl")
include("figure_packet_io.jl")
const A=EquilibriumRoleAudit
const IOChecks=FigurePacketIO
const ROOT=normpath(joinpath(@__DIR__,".."))
digest(bytes::Vector{UInt8})=bytes2hex(SHA.sha256(bytes))
write_record(path,value)=open(io->TOML.print(io,value;sorted=true),path,"w")

function verify_sources(directory)
    meta=TOML.parsefile(joinpath(directory,"metadata.toml"))
    get(meta,"completed",false) || error("incomplete source study")
    for (name,hash) in meta["source_sha256"]
        IOChecks.digest(joinpath(ROOT,name))==hash || error("changed scientific source: $name")
    end
    meta
end

function append_context!(inventory,contexts,params,scope,id,key,p,be,bi,records,roles,source,hash;cycles=0,axis="",value=NaN)
    model=A.M.model(p,be,bi);rows,computed=A.explain(model,records)
    computed==roles || error("role reproduction mismatch: $key")
    uid=scope*"/"*id*"/"*key
    haskey(params,uid) && error("duplicate context $uid")
    params[uid]=Dict("parameters"=>p,"B_E"=>be,"B_I"=>bi,"source"=>source,"source_sha256"=>hash)
    push!(contexts,(;context=uid,scope,id,B_E=be,B_I=bi,roots=length(rows),
        unassigned=count(r->r.role=="unassigned",rows),unassigned_sinks=count(r->r.role=="unassigned" && r.stability=="Attracting",rows),
        cycles,axis,value,source,source_sha256=hash))
    for row in rows
        push!(inventory,merge((;context=uid,scope,id,B_E=be,B_I=bi,cycles,axis,value,source_sha256=hash),row))
    end
end

function geometry!(inventory,contexts,params,p,folder,scope,sources)
    InputResponseArchives.verify_checkpoint(folder)
    archive=joinpath(folder,"contexts.tar.gz")
    InputResponseArchives.verify_archive(archive)
    sources[relpath(archive,ROOT)]=IOChecks.digest(archive)
    headers=Tar.list(`gzip -dc $archive`)
    grouped=Dict{String,Dict{String,Any}}()
    open(`tar -xOzf $archive`,"r") do stream
        for h in headers
            bytes=read(stream,h.size)
            startswith(h.path,"contexts/") || continue
            key=dirname(h.path);name=basename(h.path)
            entry=get!(() -> Dict{String,Any}(),grouped,key)
            if name=="context.toml"
                entry["records"]=A.retained_context(String(copy(bytes)))["equilibria"]
                entry["hash"]=digest(bytes)
            elseif name in ("roles.toml","input.toml")
                entry[name]=TOML.parse(String(bytes))
            end
        end
    end
    for key in sort(collect(keys(grouped)))
        e=grouped[key];b=e["input.toml"]
        append_context!(inventory,contexts,params,scope,p["id"],key,p,b["B_E"],b["B_I"],e["records"],e["roles.toml"],
            relpath(archive,ROOT)*"::"*key*"/context.toml",e["hash"])
    end
    expected=sum(r.roots for r in CSV.File(joinpath(folder,"inputs.csv")))
    actual=sum(length(e["records"]) for e in values(grouped))
    expected==actual || error("geometry root total mismatch")
    length(grouped)==length(CSV.File(joinpath(folder,"inputs.csv"))) || error("geometry context count mismatch")
    println(scope," ",p["id"],": ",length(grouped)," contexts / ",actual," roots");flush(stdout)
end

function raw_context!(inventory,contexts,params,folder,p,scope,id,key,sources;cycles=0,axis="",value=NaN)
    source=joinpath(folder,"context.toml");bytes=read(source);hash=digest(bytes)
    sources[relpath(source,ROOT)]=hash
    b=TOML.parsefile(joinpath(folder,"input.toml"));roles=TOML.parsefile(joinpath(folder,"roles.toml"))
    for name in ("input.toml","roles.toml")
        path=joinpath(folder,name);sources[relpath(path,ROOT)]=IOChecks.digest(path)
    end
    append_context!(inventory,contexts,params,scope,id,key,p,b["B_E"],b["B_I"],A.retained_context(String(bytes))["equilibria"],roles,
        relpath(source,ROOT),hash;cycles,axis,value)
end

sortkey(r)=(r.id,r.B_E,r.B_I,r.E,r.I,r.context,r.root)
groupkey(r)=r.reasons*"::"*r.descriptor
function select_seeds(inventory;limit=24)
    candidates=sort(filter(r->r.role=="unassigned" && r.stability=="Attracting" && !r.near_singular,inventory);by=sortkey)
    selected=NamedTuple[];used=Set{Tuple{String,Int}}();covered=Set{String}()
    function take(r,reason)
        length(selected)>=limit && return
        (r.context,r.root) in used && return
        push!(used,(r.context,r.root));push!(covered,groupkey(r))
        push!(selected,merge((;seed=length(selected)+1,selection_reason=reason),r))
    end
    for r in filter(r->r.scope=="baseline" && r.cycles>0,candidates)
        take(r,"equilibrium coexisting with validated cycle")
    end
    # Blocks are contiguous in all sampled intervention settings, not only U rows.
    interventions=filter(r->r.scope=="intervention",inventory)
    for (id,axis) in sort(unique((r.id,r.axis) for r in interventions))
        allvalues=sort(unique(r.value for r in interventions if r.id==id && r.axis==axis))
        localrows=filter(r->r.scope=="intervention" && r.id==id && r.axis==axis,candidates)
        for key in sort(unique(groupkey(r) for r in localrows))
            group=filter(r->groupkey(r)==key,localrows);blocks=Vector{Vector{NamedTuple}}();block=NamedTuple[];previous=0
            for value in allvalues
                here=filter(r->r.value==value,group)
                if isempty(here)
                    isempty(block) || push!(blocks,block);block=NamedTuple[]
                else
                    append!(block,here)
                end
            end
            isempty(block) || push!(blocks,block)
            for block in blocks
                values=sort(unique(r.value for r in block));mid=values[cld(length(values),2)]
                take(first(sort(filter(r->r.value==mid,block);by=sortkey)),"midpoint of contiguous sampled intervention descriptor block")
            end
        end
    end
    for pool in (filter(r->r.scope=="detail_geometry" && r.id in ("selective_withdrawal","input_dependent"),candidates),candidates)
        for r in pool
            groupkey(r) in covered || take(r,"uncovered reason and geometry combination")
        end
    end
    selected,setdiff(Set(groupkey(r) for r in candidates),covered)
end

function inventory(screen,details,output)
    ispath(output) && error("audit output already exists")
    verify_sources(screen);verify_sources(details);mkpath(output)
    cases=TOML.parsefile(joinpath(ROOT,"reproducibility/figure_redesign_20261006/cases.toml"))["cases"]
    selected=TOML.parsefile(joinpath(details,"selection.toml"))["cases"]
    byid=Dict(p["id"]=>p for p in cases)
    inv=NamedTuple[];contexts=NamedTuple[];params=Dict{String,Any}();sources=Dict{String,String}()
    for p in cases
        geometry!(inv,contexts,params,p,joinpath(screen,p["category"],p["id"],"geometry"),"screen",sources)
    end
    for p in selected
        folder=joinpath(details,"cases",p["id"])
        geometry!(inv,contexts,params,p,joinpath(folder,"geometry"),"detail_geometry",sources)
        for (dir,_,names) in walkdir(folder)
            "context.toml" in names && "input.toml" in names && "roles.toml" in names || continue
            base=startswith(basename(dir),"baseline_")
            cycles=base ? get(TOML.parsefile(joinpath(dir,"summary.toml")),"validated_cycles",0) : 0
            raw_context!(inv,contexts,params,dir,p,base ? "baseline" : "geometry_confirmation",p["id"],relpath(dir,folder),sources;cycles)
        end
    end
    for id in ("selective_withdrawal","threshold_to_active")
        folder=joinpath(details,"interventions",id);p=byid[id]
        InputResponseArchives.verify_checkpoint(folder)
        for r in CSV.File(joinpath(folder,"summary.csv"))
            q=merge(p,Dict(r.axis=>r.value));key=r.axis*"_"*string(r.value)
            raw_context!(inv,contexts,params,joinpath(folder,key),q,"intervention",id,key,sources;axis=r.axis,value=r.value)
        end
        raw_context!(inv,contexts,params,folder,p,"intervention_baseline",id,"original",sources)
    end
    for p in selected
        for (dir,_,names) in walkdir(joinpath(details,"timescales"))
            startswith(basename(dir),p["id"]*"_") && "context.toml" in names || continue
            raw_context!(inv,contexts,params,dir,p,"timescale_baseline",p["id"],basename(dir),sources)
        end
    end
    raw_context!(inv,contexts,params,joinpath(details,"sequence"),byid["selective_withdrawal"],"sequence","selective_withdrawal","original",sources)
    CSV.write(joinpath(output,"inventory.csv"),inv);CSV.write(joinpath(output,"contexts.csv"),contexts)
    seeds,uncovered=select_seeds(inv);CSV.write(joinpath(output,"seeds.csv"),seeds)
    write_record(joinpath(output,"seed_parameters.toml"),Dict("seeds"=>[merge(Dict("seed"=>r.seed,"context"=>r.context,"root"=>r.root),params[r.context]) for r in seeds]))
    summary=NamedTuple[]
    for scope in sort(unique(r.scope for r in inv)), key in sort(unique(groupkey(r) for r in inv if r.scope==scope && r.role=="unassigned"))
        rows=filter(r->r.scope==scope && r.role=="unassigned" && groupkey(r)==key,inv)
        push!(summary,(;scope,key,roots=length(rows),sinks=count(r->r.stability=="Attracting",rows),contexts=length(unique(r.context for r in rows)),cases=length(unique(r.id for r in rows))))
    end
    CSV.write(joinpath(output,"summary.csv"),summary)
    source_code=Dict(name=>IOChecks.digest(joinpath(@__DIR__,name)) for name in ("run_equilibrium_audit.jl","equilibrium_role_audit.jl"))
    write_record(joinpath(output,"audit.toml"),Dict("inventory_completed"=>true,"roots"=>length(inv),"contexts"=>length(contexts),
        "unassigned"=>count(r->r.role=="unassigned",inv),"unassigned_sinks"=>count(r->r.role=="unassigned" && r.stability=="Attracting",inv),
        "selected_seeds"=>length(seeds),"uncovered_combinations"=>sort(collect(uncovered)),"sources"=>sources,"source_code"=>source_code,
        "completeness"=>"CompletenessNotCertified","screen"=>relpath(screen,ROOT),"details"=>relpath(details,ROOT)))
    IOChecks.write_manifest(output)
    println("Inventory complete: ",length(inv)," roots, ",length(contexts)," contexts, ",length(seeds)," seeds")
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: run_equilibrium_audit.jl SCREEN DETAILS NEW_OUTPUT")
    EquilibriumAuditStudy.inventory(abspath.(ARGS)...)
end
