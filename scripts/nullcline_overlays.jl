"""Portable basin/manifold evidence for the fixed-input mechanism panels."""
module NullclineOverlays
import CSV, TOML, SHA
const ROOT=normpath(joinpath(@__DIR__,".."))
const CONDITIONS=("monotone","failure")
digest(path)=bytes2hex(open(SHA.sha256,path))
record(x::Symbol)=string(x)
record(::Nothing)="automatic"
record(::Missing)="unavailable"
record(x::AbstractMatrix)=[record(collect(row)) for row in eachrow(x)]
record(x::Union{AbstractString,Number,Bool})=x
record(x::Union{Tuple,AbstractVector})=record.(collect(x))
record(x::NamedTuple)=Dict(string(k)=>record(v) for (k,v) in pairs(x))
record(x::AbstractDict)=Dict(string(k)=>record(v) for (k,v) in x)
record(x)=Dict(string(k)=>record(getfield(x,k)) for k in fieldnames(typeof(x)))
function write_record(path,value)
    mkpath(dirname(path));open(path,"w") do io;TOML.print(io,record(value);sorted=true);end
end
function checked_complete_unit(folder)
    marker=joinpath(folder,"done.toml")
    isfile(marker) || return false
    expected=TOML.parsefile(marker)["files"];actual=Dict{String,String}()
    for (directory,_,files) in walkdir(folder), file in files
        file=="done.toml" && continue
        path=joinpath(directory,file)
        actual[relpath(path,folder)]=digest(path)
    end
    actual==expected || error("mechanism checkpoint file set or content changed")
    true
end
function source_identity()
    paths=String["Project.toml","Manifest.toml","scripts/nullcline_overlays.jl",
        "scripts/run_figure_details.jl","scripts/input_response_models.jl"]
    for (directory,_,files) in walkdir(joinpath(ROOT,"src")), file in files
        endswith(file,".jl") && push!(paths,relpath(joinpath(directory,file),ROOT))
    end
    Dict(path=>digest(joinpath(ROOT,path)) for path in sort(paths))
end
function configuration(cfg;basins=nothing,separatrices=nothing)
    section=get(cfg.raw,"nullcline_overlays",Dict{String,Any}())
    (;basins=basins===nothing ? get(section,"basins",true) : basins,
      separatrices=separatrices===nothing ? get(section,"separatrices",true) : separatrices,
      basin=get(section,"basin",Dict{String,Any}()),
      separatrix=get(section,"separatrix",Dict{String,Any}()))
end
function options_from(::Type{T},values;api=nothing) where T
    allowed=Set(string.(fieldnames(T)))
    startswith(string(T),"SeparatrixOptions") && union!(allowed,("display_span","display_pixels"))
    all(in(allowed),keys(values)) || error("unsupported $(T) overlay setting")
    kwargs=Pair{Symbol,Any}[]
    for (key,value) in values
        key=="bounds" && (value=Tuple(Tuple(Float64.(a)) for a in value))
        if key=="pulse_options" && value isa AbstractDict
            api===nothing && error("pulse_options decoding needs the scientific API")
            value=options_from(api.PulseExperimentOptions,value;api)
        elseif key=="diagnostic_options" && value isa AbstractDict
            api===nothing && error("diagnostic_options decoding needs the scientific API")
            value=options_from(api.DiagnosticOptions,value;api)
        elseif key=="targets"
            value=Symbol.(value)
        end
        push!(kwargs,Symbol(key)=>value)
    end
    T(;kwargs...)
end
function identity(p,cfg,settings,roots)
    Dict("schema_version"=>1,"parameters"=>record(p),"baseline"=>[.35,0.],
         "configuration"=>record(cfg.raw),"generation"=>record(settings),
         "roots"=>record(roots),"source_sha256"=>source_identity(),
         "limits"=>"Finite-window equilibrium compatibility; unresolved trajectories remain neutral. No completeness certification.")
end
function basin_rows(condition,result)
    [(;condition,E_index=e,I_index=i,E=result.E[e],I=result.I[i],
       label=result.labels[e,i],status=string(result.statuses[e,i]))
     for e in eachindex(result.E) for i in eachindex(result.I)]
end
function generate(folder,p,cfg,models,searches,roots;basins=nothing,separatrices=nothing,api)
    settings=configuration(cfg;basins,separatrices)
    expected=identity(p,cfg,settings,roots)
    path=joinpath(folder,"overlays","metadata.toml")
    if isfile(path)
        checked_complete_unit(folder) || error("incomplete overlay checkpoint; use a new output")
        existing=TOML.parsefile(path)
        existing["identity"]==expected || error("overlay identity changed; use a new output")
        return existing
    end
    mkpath(dirname(path));metadata=Dict{String,Any}("identity"=>expected,"conditions"=>Dict{String,Any}())
    for condition in CONDITIONS
        model=models[condition];search=searches[condition];entry=Dict{String,Any}()
        if settings.basins
            options=options_from(api.BasinGridOptions,settings.basin;api)
            result=api.basin_grid(model,search;options)
            CSV.write(joinpath(dirname(path),"$(condition)_basins.csv"),basin_rows(condition,result))
            entry["basin_options"]=record(result.options)
            entry["basin_statistics"]=record(result.statistics)
            entry["shape"]=[length(result.E),length(result.I)]
        end
        if settings.separatrices
            options=options_from(api.SeparatrixOptions,settings.separatrix;api)
            result=api.trace_separatrices(model,search;options)
            entry["separatrix_options"]=record(options)
            write_separatrices(dirname(path),condition,result,entry)
        end
        metadata["conditions"][condition]=entry
    end
    write_record(path,metadata);metadata
end

function write_separatrices(folder,condition,branches,entry)
    records=Dict{String,Any}[];traces=NamedTuple[]
    for (index,branch) in enumerate(branches)
        metadata=Dict(string(k)=>record(v) for (k,v) in pairs(branch) if k ∉ (:points,:E,:I,:attempts))
        metadata["attempts"]=[merge(Dict(string(k)=>record(v) for (k,v) in pairs(a) if k!=:points),
            Dict("point_count"=>length(a.points))) for a in branch.attempts]
        metadata["display_points"]=length(branch.points)
        push!(records,metadata)
        for (attempt,points) in vcat([(0,branch.points)],[(k,a.points) for (k,a) in enumerate(branch.attempts)])
            for (point,state) in enumerate(points)
                push!(traces,(;branch=index,attempt,point,E=state[1],I=state[2]))
            end
        end
    end
    entry["separatrices"]=records
    # A typed empty table retains its header when the finite root search found no saddles.
    table=isempty(traces) ? (;branch=Int[],attempt=Int[],point=Int[],E=Float64[],I=Float64[]) : traces
    CSV.write(joinpath(folder,"$(condition)_separatrices.csv"),table)
end
function read_separatrices(folder,condition,roots,entry)
    records=entry["separatrices"]
    traces=collect(CSV.File(joinpath(folder,"$(condition)_separatrices.csv");stringtype=String))
    expected=sort([(r.root,s) for r in roots if r.stability=="Saddle" for s in (-1,1)])
    sort([(r["root_index"],r["sign"]) for r in records])==expected || error("separatrix saddle/sign coverage changed")
    result=NamedTuple[]
    for (index,branch) in enumerate(records)
        root=roots[branch["root_index"]]
        branch["root_state"]==[root.E,root.I] || error("separatrix root identity changed")
        display=filter(r->r.branch==index && r.attempt==0,traces)
        length(display)==branch["display_points"] || error("separatrix display shape changed")
        all(r.point==k for (k,r) in enumerate(display)) || error("separatrix point ordering changed")
        all(r->isfinite(r.E) && isfinite(r.I) && -1e-8<=r.E<=1+1e-8 && -1e-8<=r.I<=1+1e-8,traces) || error("separatrix left physical domain")
        branch["converged"]==(branch["status"]=="converged") || error("separatrix convergence status changed")
        if branch["converged"]
            length(display)>=2 || error("converged separatrix has no display geometry")
            branch["geometric_disagreement"]<=branch["geometry_atol"] || error("separatrix convergence tolerance changed")
            all(a->a["successful"],branch["attempts"]) || error("converged separatrix has failed attempt")
            a,b=branch["attempts"]
            (a["termination_reason"],a["terminal_equilibrium"])==(b["termination_reason"],b["terminal_equilibrium"]) || error("separatrix attempt endpoints disagree")
        else
            isempty(display) || error("unconverged separatrix has display geometry")
        end
        for attempt in 1:2
            group=filter(r->r.branch==index && r.attempt==attempt,traces)
            length(group)==branch["attempts"][attempt]["point_count"] || error("separatrix attempt shape changed")
            all(r.point==k for (k,r) in enumerate(group)) || error("separatrix attempt ordering changed")
            if !isempty(group)
                length(group)>=2 || error("separatrix attempt lacks its launch seed")
                [first(group).E,first(group).I]==[root.E,root.I] || error("separatrix attempt saddle changed")
                seed=[root.E,root.I]+branch["sign"]*branch["attempts"][attempt]["seed_offset"]*branch["stable_eigenvector"]
                max(abs(group[2].E-seed[1]),abs(group[2].I-seed[2]))<1e-10 || error("separatrix seed/sign changed")
            end
        end
        push!(result,(;E=getproperty.(display,:E),I=getproperty.(display,:I),converged=branch["converged"],metadata=branch))
    end
    all(r->1<=r.branch<=length(records) && r.attempt in 0:2,traces) || error("unknown separatrix trace")
    result
end

"""Read and validate explicit grid indices without relying on CSV row order."""
function read_basins(folder,condition,roots,entry)
    shape=Int.(entry["shape"]);length(shape)==2 && all(>=(2),shape) || error("invalid basin shape")
    rows=collect(CSV.File(joinpath(folder,"$(condition)_basins.csv");stringtype=String))
    length(rows)==prod(shape) || error("basin cell count changed")
    E=fill(NaN,shape[1]);I=fill(NaN,shape[2]);labels=fill(-1,Tuple(shape));statuses=fill("",Tuple(shape))
    for row in rows
        row.condition==condition || error("basin condition changed")
        e=row.E_index;i=row.I_index
        1<=e<=shape[1] && 1<=i<=shape[2] || error("basin index outside grid")
        labels[e,i]==-1 || error("duplicate basin cell")
        0<=row.E<=.5 && 0<=row.I<=.5 || error("basin coordinate outside displayed bounds")
        (isnan(E[e]) || E[e]==row.E) && (isnan(I[i]) || I[i]==row.I) || error("inconsistent basin coordinates")
        E[e]=row.E;I[i]=row.I
        0<=row.label<=length(roots) || error("invalid basin root label")
        if row.label>0
            roots[row.label].stability=="Attracting" || error("basin label is not an attracting root")
            row.status=="compatible" || error("resolved basin has incompatible status")
        else
            row.status!="compatible" || error("compatible basin lacks a root")
        end
        labels[e,i]=row.label;statuses[e,i]=row.status
    end
    E==collect(range(0.,.5;length=shape[1])) && I==collect(range(0.,.5;length=shape[2])) ||
        error("basin grid changed from displayed [0,0.5] bounds")
    entry["basin_options"]["grid_points"]==shape[1]==shape[2] || error("basin settings disagree with grid")
    entry["basin_options"]["bounds"]==[[0.,.5],[0.,.5]] || error("basin settings bounds changed")
    (;E,I,labels,statuses)
end
end
