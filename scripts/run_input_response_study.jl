"""Bounded two-input equilibrium, stability, and transition characterization."""
module InputResponseStudy
using FailureOfInhibition2025, LinearAlgebra
import CSV,TOML,SHA,SciMLBase
include("input_response_models.jl")
include("run_narrative_study.jl")
const M=InputResponseModels
const N=NarrativeStudy
const IR=N.IR
const ROOT=normpath(joinpath(@__DIR__,".."))
const write_record=N.write_record
const source_files=vcat(N.SOURCE_FILES,["scripts/input_response_models.jl","scripts/run_input_response_study.jl"])

function load_config(path;smoke=false)
    raw=TOML.parsefile(path)
    raw["schema_version"]===1 || throw(ArgumentError("unsupported schema"))
    for key in ("initial_input_bound","tail_tolerance","input_width","chart_step","displacement_width")
        M.number(raw[key],key;positive=true)
    end
    for key in ("maximum_expansions","geometry_budget","screen_budget","transition_budget","line_budget",
        "root_grid","confirmation_grid","state_probe_grid","halton_per_family","maximum_additional_per_family",
        "phase_count","phase_confirmation_count")
        N.Evidence.positive_integer(raw[key],key)
    end
    raw["root_grid"]>=2 && raw["state_probe_grid"]>=2 || throw(ArgumentError("grid too small"))
    all(raw[k]>=5 for k in ("geometry_budget","screen_budget","transition_budget")) || throw(ArgumentError("rectangle budgets must allow five probes"))
    raw["phase_confirmation_count"]==2raw["phase_count"] || throw(ArgumentError("phase confirmation count must double the initial count"))
    for key in ("durations","followup_times","input_knots")
        v=raw[key]
        !isempty(v) && issorted(v) && length(unique(v))==length(v) || throw(ArgumentError("invalid $key"))
        foreach(x->M.number(x,key;positive=key!="input_knots"),v)
    end
    first(raw["input_knots"])==0 || throw(ArgumentError("input grid must begin at zero"))
    d=raw["diagnostics"]
    diag=DiagnosticOptions(window_duration=d["window_duration"],coordinate_atol=d["coordinate_atol"],
        balance_atol=d["balance_atol"],min_samples=d["min_samples"])
    horizons=Float64.(raw["followup_times"])
    options=PulseExperimentOptions(amplitudes=[0.],durations=raw["durations"],followup_times=horizons,
        diagnostic_options=diag,abstol=d["abstol"],reltol=d["reltol"],domain_atol=d["domain_atol"],maxiters=d["maxiters"])
    expansion=raw["expansion"]
    length(expansion["axes"])==length(expansion["lower"])==length(expansion["upper"])==length(expansion["offsets"])==4 ||
        throw(ArgumentError("invalid expansion axes"))
    for (lo,hi,step) in zip(expansion["lower"],expansion["upper"],expansion["offsets"])
        M.number(lo,"lower");M.number(hi,"upper");M.number(step,"offset";positive=true)
        lo<hi || throw(ArgumentError("unordered expansion bounds"))
    end
    if smoke
        raw["geometry_budget"]=70;raw["screen_budget"]=40;raw["transition_budget"]=40
        raw["line_budget"]=40;raw["chart_step"]=.2;raw["durations"]=[10.,100.]
    end
    (;raw,smoke,diagnostics=diag,horizons,abstol=options.abstol,reltol=options.reltol,
        domain_atol=options.domain_atol,maxiters=options.maxiters)
end

function anchors()
    cases=TOML.parsefile(joinpath(ROOT,"experiments/exemplar_models.toml"))["cases"]
    result=Dict{String,Any}[]
    for p in cases
        push!(result,merge(p,Dict("id"=>p["name"],"family"=>p["i_to_e"]==9 ? "figure3" : "figure4",
            "baselines"=>[[0.,0.]])))
    end
    base=Dict{String,Any}("family"=>"figure4","e_to_e"=>19.,"i_to_e"=>13.,"e_to_i"=>19.,
        "i_to_i"=>6.,"theta_off"=>8.,"tau_ratio"=>.2)
    push!(result,merge(base,Dict("id"=>"narrative","baselines"=>[[.015625,0.]])))
    push!(result,merge(base,Dict("id"=>"selective_withdrawal","e_to_e"=>17.,"baselines"=>[[.35,0.],[.1,0.]])))
    push!(result,merge(base,Dict("id"=>"threshold_to_active","e_to_e"=>16.,"baselines"=>[[.5,0.]])))
    push!(result,merge(first(result),Dict("id"=>"input_dependent","e_to_e"=>4.,"baselines"=>[[8.,0.],[0.,0.]])))
    result
end

function initialize(config_path,cfg,output;stage="all",case_filter=nothing)
    path=joinpath(output,"metadata.toml")
    if isfile(path)
        data=TOML.parsefile(path)
        data["config_sha256"]==N.Evidence.file_hash(config_path) && data["smoke"]==cfg.smoke ||
            throw(ArgumentError("configuration changed; use a new output"))
        for (file,hash) in data["source_sha256"]
            N.Evidence.file_hash(joinpath(ROOT,file))==hash && N.Evidence.file_hash(joinpath(output,"source",file))==hash ||
                throw(ArgumentError("source identity changed: $file"))
        end
        return data
    end
    isdir(output) && !isempty(readdir(output)) && throw(ArgumentError("nonempty unrecognized output"))
    mkpath(output);data=N.Evidence.archive_provenance(config_path,output)
    data["purpose"]="bounded two-input response characterization; no biological certification"
    data["replay_invocations"]=String[]
    data["replay_from_artifact_directory"]=N.replay_command("run_input_response_study.jl";
        stage,case_filter,smoke=cfg.smoke)
    for file in vcat(source_files,["experiments/exemplar_models.toml"])
        target=joinpath(output,"source",file);mkpath(dirname(target));cp(joinpath(ROOT,file),target;force=true)
        data["source_sha256"][file]=N.Evidence.file_hash(target)
    end
    data["smoke"]=cfg.smoke;data["completed"]=false
    data["claim_limits"]="numerical equilibrium/critical sets; finite-window transitions; no biological or global completeness certification"
    write_record(path,data);data
end

function chart_seeds(m,be,bi;n=257)
    c=m.coupling;c.e_to_i>0 || return default_equilibrium_seeds(m)
    _,imax=M.state_bounds(m)
    vs=collect(range(bi-c.i_to_i*imax,bi+c.e_to_i/2;length=n))
    f=m.inhibitory.response
    append!(vs,filter(v->first(vs)<=v<=last(vs),collect(f.onset_threshold-2:.025:f.failure_threshold+2)))
    sort!(unique!(vs))
    value(v)=begin
        i=M.occupancy(f,v);e=(v+c.i_to_i*i-bi)/c.e_to_i
        if !(0<e<.5);return (NaN,[e,i]);end
        residual=-e+(1-e)*response(m.excitatory.response,be+c.e_to_e*e-c.i_to_e*i)
        (residual,[e,i])
    end
    seeds=Vector{Float64}[]
    for (a,b) in zip(vs,vs[2:end])
        fa,xa=value(a);fb,xb=value(b)
        isfinite(fa) && abs(fa)<1e-5 && push!(seeds,xa)
        isfinite(fa) && isfinite(fb) && signbit(fa)!=signbit(fb) || continue
        for _ in 1:40
            mid=(a+b)/2;fm,_=value(mid)
            isfinite(fm) || break
            if signbit(fm)==signbit(fa);a=mid;fa=fm;else;b=mid;end
        end
        _,x=value((a+b)/2);push!(seeds,x)
    end
    seeds
end

function context(p,be,bi,cfg;tight=false,grid=cfg.raw["root_grid"])
    m=M.model(p,be,bi)
    seeds=vcat(chart_seeds(m,be,bi),default_equilibrium_seeds(m),
        [[e,i] for e in range(0,.5;length=grid) for i in range(0,M.state_bounds(m)[2];length=grid)])
    options=tight ? EquilibriumOptions(residual_atol=1e-11,solver_reltol=1e-11) : EquilibriumOptions()
    search=find_equilibria(m;seeds,options)
    # An inward planar field has index one. A mismatch is a useful discovery
    # alarm, not a sufficient test of root completeness when it does match.
    index_sum(s)=sum(x.stability.classification==Saddle ? -1 :
        x.stability.classification in (Attracting,Repelling) ? 1 : 0 for x in s.equilibria;init=0)
    if index_sum(search)!=1 && grid<21
        return context(p,be,bi,cfg;tight,grid=21)
    end
    (;p,be,bi,model=m,search,roles=M.roles(m,search),
        discovery_status=index_sum(search)==1 ? "index_consistent_not_complete" : "index_unresolved")
end

label(ctx,k)=something(findfirst(==(k),ctx.roles),"unassigned")
signature(ctx)=join(sort([string(x.stability.classification)*":"*label(ctx,k) for (k,x) in enumerate(ctx.search.equilibria)]),"|")*
    (ctx.discovery_status=="index_unresolved" ? "|index_unresolved" : "")
context_id(be,bi)=string(be)*"_"*string(bi)

function save_context(dir,ctx)
    mkpath(dir)
    write_record(joinpath(dir,"context.toml"),N.Evidence.context_record(ctx.search))
    write_record(joinpath(dir,"roles.toml"),ctx.roles)
    write_record(joinpath(dir,"input.toml"),(;B_E=ctx.be,B_I=ctx.bi))
end

function critical_curves(p,bounds,cfg)
    m=M.model(p);vrange=M.total_input_bounds(m,bounds)[2]
    rows=NamedTuple[]
    for kind in ("fold","hopf")
        cache=Dict{Float64,Any}()
        at(v)=get!(() -> filter(r->0<=r.B_E<=bounds.B_E && 0<=r.B_I<=bounds.B_I,M.critical_states(m,v,kind)),cache,v)
        function refine(a,b,depth=0)
            left,right=at(a),at(b);middle=at((a+b)/2)
            isempty(left) && isempty(right) && isempty(middle) && return
            depth>=14 && return
            changed=length(left)!=length(right) || length(left)!=length(middle)
            wide=any(l->any(r->l.sheet==r.sheet && max(abs(l.B_E-r.B_E),abs(l.B_I-r.B_I))>cfg.raw["input_width"],right),left)
            if changed || wide
                refine(a,(a+b)/2,depth+1);refine((a+b)/2,b,depth+1)
            end
        end
        vs=unique(vcat(collect(vrange[1]:cfg.raw["chart_step"]:vrange[2]),[vrange[2]]))
        for (a,b) in zip(vs,vs[2:end]);refine(a,b);end
        for v in sort(collect(keys(cache)));append!(rows,cache[v]);end
    end
    rows
end

function input_axis(cfg,upper;extra=Float64[])
    sort!(unique(vcat(filter(x->x<=upper,Float64.(cfg.raw["input_knots"])),[upper],
        filter(x->0<=x<=upper,Float64.(extra)),collect(16.:16.:upper))))
end

function geometry(p,cfg,dir;screen=false)
    if N.checked_unit(dir)
        return TOML.parsefile(joinpath(dir,"summary.toml"))
    end
    mkpath(dir);write_record(joinpath(dir,"parameters.toml"),p)
    bounds=M.input_bounds(M.model(p);initial=cfg.raw["initial_input_bound"],epsilon=cfg.raw["tail_tolerance"],
        max_expansions=cfg.raw["maximum_expansions"])
    write_record(joinpath(dir,"bounds.toml"),bounds)
    curves=critical_curves(p,bounds,cfg)
    isempty(curves) || CSV.write(joinpath(dir,"critical.csv"),curves)
    contexts=Dict{Tuple{Float64,Float64},Any}()
    function at(be,bi)
        get!(contexts,(Float64(be),Float64(bi))) do
            ctx=context(p,be,bi,cfg)
            save_context(joinpath(dir,"contexts",context_id(be,bi)),ctx)
            ctx
        end
    end
    # Known points and critical-set offsets are visited before the adaptive
    # lattice, so a finite budget cannot suppress the anchor observations.
    baselines=p["baselines"]
    for b in baselines;at(b...);end
    stride=max(1,cld(length(curves),screen ? 8 : 40))
    for r in curves[1:stride:end],sign in (-1,1)
        at(clamp(r.B_E+sign*.02,0,bounds.B_E),clamp(r.B_I+sign*.02,0,bounds.B_I))
    end
    xs=input_axis(cfg,bounds.B_E;extra=first.(baselines));ys=input_axis(cfg,bounds.B_I;extra=last.(baselines))
    atlas=M.sample_rectangle(at,xs,ys;width=cfg.raw["input_width"],key=signature,
        max_evaluations=cfg.raw[screen ? "screen_budget" : "geometry_budget"])
    CSV.write(joinpath(dir,"cells.csv"),atlas.leaves)
    rows=NamedTuple[];rootrows=NamedTuple[]
    for ((be,bi),ctx) in sort(collect(contexts);by=first)
        push!(rows,(;B_E=be,B_I=bi,roots=length(ctx.search.equilibria),sinks=length(IR.sink_indices(ctx.search)),
            signature=signature(ctx),roles=join(sort(collect(keys(ctx.roles))),","),discovery_status=ctx.discovery_status))
        for (k,r) in enumerate(ctx.search.equilibria)
            gain=M.susceptibility(ctx.model,r.state)
            push!(rootrows,(;B_E=be,B_I=bi,root=k,E=r.state[1],I=r.state[2],role=label(ctx,k),
                stability=string(r.stability.classification),gain_status=gain.status,
                dE_dBE=gain.gain[1,1],dE_dBI=gain.gain[1,2],dI_dBE=gain.gain[2,1],dI_dBI=gain.gain[2,2]))
        end
    end
    CSV.write(joinpath(dir,"inputs.csv"),rows);CSV.write(joinpath(dir,"equilibria.csv"),rootrows)
    lineage=NamedTuple[]
    for axis in (1,2)
        fixed_values=sort(unique(k[3-axis] for k in keys(contexts)))
        for fixed in fixed_values
            ordered=sort(filter(k->k[3-axis]==fixed,collect(keys(contexts)));by=k->k[axis])
            for (left,right) in zip(ordered,ordered[2:end])
                right[axis]-left[axis]<=.25 || continue
                pairs=M.correspondence(contexts[left].search,contexts[right].search)
                for (old,new) in sort(collect(pairs);by=first)
                    push!(lineage,(;axis,from_E=left[1],from_I=left[2],from_root=old,
                        to_E=right[1],to_I=right[2],to_root=new,status="unique_nearby_match"))
                end
            end
        end
    end
    isempty(lineage) || CSV.write(joinpath(dir,"lineage.csv"),lineage)
    representatives=observed_representatives(atlas,contexts,curves,bounds)
    representatives=unique(vcat([Float64.(b) for b in baselines],representatives))
    summary=Dict("id"=>p["id"],"inputs"=>length(rows),"critical_points"=>length(curves),
        "budget_unresolved_cells"=>count(x->x.status=="budget_unresolved",atlas.leaves),
        "signatures"=>sort(unique(r.signature for r in rows)),"representatives"=>representatives,
        "screen"=>screen,"completeness"=>"CompletenessNotCertified")
    write_record(joinpath(dir,"summary.toml"),summary);N.complete_unit(dir)
    summary
end

"""Select one representative per component supported by sampled homogeneous cells."""
function observed_representatives(atlas,contexts,curves,bounds)
    keys_sorted=sort(collect(keys(contexts)));adj=Dict(k=>Tuple{Float64,Float64}[] for k in keys_sorted)
    for cell in atlas.leaves
        cell.status=="sampled_homogeneous" || continue
        points=filter(k->haskey(contexts,k),[(cell.x0,cell.y0),(cell.x1,cell.y0),(cell.x0,cell.y1),(cell.x1,cell.y1),((cell.x0+cell.x1)/2,(cell.y0+cell.y1)/2)])
        append!(points,filter(k->cell.x0<=k[1]<=cell.x1 && cell.y0<=k[2]<=cell.y1,keys_sorted))
        unique!(points)
        for a in points,b in points
            a!=b && signature(contexts[a])==signature(contexts[b]) && push!(adj[a],b)
        end
    end
    visited=Set{Tuple{Float64,Float64}}();representatives=Vector{Float64}[]
    for seed in keys_sorted
        seed in visited && continue
        isempty(adj[seed]) && continue # unresolved/disconnected probes are not established regions
        queue=[seed];component=Tuple{Float64,Float64}[];push!(visited,seed)
        while !isempty(queue)
            x=pop!(queue);push!(component,x)
            for y in adj[x]
                y in visited && continue
                push!(visited,y);push!(queue,y)
            end
        end
        # Euclidean input distance is only a representative-selection rule,
        # never a control cost. Include rectangle edges as boundaries.
        distance(x)=min(x[1],bounds.B_E-x[1],x[2],bounds.B_I-x[2],
            isempty(curves) ? Inf : minimum(hypot(x[1]-r.B_E,x[2]-r.B_I) for r in curves))
        sort!(component;by=x->(-distance(x),x))
        push!(representatives,collect(first(component)))
    end
    representatives
end

function confirm_geometry(p,cfg,geometry_dir)
    dir=joinpath(dirname(geometry_dir),"geometry_confirmation")
    N.checked_unit(dir) && return
    mkpath(dir);summary=TOML.parsefile(joinpath(geometry_dir,"summary.toml"))
    confirmations=NamedTuple[]
    for (index,b) in enumerate(summary["representatives"])
        ctx=context(p,b...,cfg;tight=true,grid=cfg.raw["confirmation_grid"])
        save_context(joinpath(dir,"input_$index"),ctx)
        original=TOML.parsefile(joinpath(geometry_dir,"contexts",context_id(b...),"context.toml"))
        oldstates=[r["state"] for r in original["equilibria"]]
        consistent=length(oldstates)==length(ctx.search.equilibria) && all(x->any(y->maximum(abs.(x-y.state))<1e-6,ctx.search.equilibria),oldstates)
        push!(confirmations,(;B_E=b[1],B_I=b[2],consistent,roots=length(ctx.search.equilibria)))
    end
    CSV.write(joinpath(dir,"independent_searches.csv"),confirmations)
    # Axis continuations cross-check the chart using the existing independent
    # predictor/corrector. Every stop reason and candidate remains retained.
    b=first(p["baselines"]);ctx=context(p,b...,cfg;grid=21)
    bounds=TOML.parsefile(joinpath(geometry_dir,"bounds.toml"))
    for axis in (1,2), (index,root) in enumerate(ctx.search.equilibria)
        factory=x->M.model(p,axis==1 ? x : b[1],axis==2 ? x : b[2])
        result=continue_equilibria(factory,root.state,b[axis];parameter_bounds=(0.,bounds[axis==1 ? "B_E" : "B_I"]),
            options=ContinuationOptions(max_steps=600))
        write_record(joinpath(dir,"axis_$(axis)_root_$(index).toml"),result)
    end
    N.complete_unit(dir)
end

function periodic_candidate(ctx,initial,cfg;horizon=5000.)
    solution=solve_point_model(initial,(0.,horizon),ctx.model;saveat=.5,
        tstops=cfg.domain_atol==0 ? collect(0.:.5:horizon) : (),
        save_everystep=false,dense=false,abstol=cfg.abstol,reltol=cfg.reltol,
        domain_atol=cfg.domain_atol,maxiters=cfg.maxiters)
    SciMLBase.successful_retcode(solution) || return (;status="integration_failed",orbit=nothing,solution)
    indices=findall(t->t>=horizon-1000.,solution.t)
    es=[solution.u[k][1] for k in indices]
    maximum(es)-minimum(es)>1e-5 || return (;status="no_nonconstant_seed",orbit=nothing,solution)
    level=(maximum(es)+minimum(es))/2;times=Float64[];states=Vector{Float64}[]
    for k in first(indices):last(indices)-1
        l,r=solution.u[k][1],solution.u[k+1][1]
        l<level<=r || continue
        fraction=(level-l)/(r-l)
        push!(times,solution.t[k]+fraction*(solution.t[k+1]-solution.t[k]))
        push!(states,solution.u[k]+fraction*(solution.u[k+1]-solution.u[k]))
    end
    length(times)>=5 || return (;status="insufficient_recurrence",orbit=nothing,solution)
    periods=diff(times[end-4:end]);period=sum(periods)/length(periods)
    (maximum(periods)-minimum(periods))/period<.02 || return (;status="irregular_recurrence",orbit=nothing,solution)
    orbit=solve_periodic_orbit(ctx.model,last(states),period)
    (;status=string(orbit.validation),orbit,solution)
end

function destination(ctx,phase;reference=nothing)
    phase.status=="compatible" || return phase.status
    role=label(ctx,phase.destination)
    if role=="unassigned" && reference!==nothing
        matches=M.correspondence(reference.search,ctx.search;atol=.05)
        old=findfirst(==(phase.destination),matches)
        old===nothing || (role=label(reference,old))
    end
    role=="unassigned" ? "equilibrium_$(phase.destination)" : role
end

function observe(ctx,initial,cfg;reference=nothing,retain=false,tight=false,cycles=false)
    attempts=NamedTuple[];phase=nothing;dest="unresolved";status="unresolved";orbit=nothing
    # Apply the same stopping policy to equilibria and attracting cycles.
    # Unresolved observations still restart at every configured longer horizon.
    for horizon in (cycles ? cfg.horizons : [last(cfg.horizons)])
        local_cfg=cycles ? merge(cfg,(;horizons=[horizon])) : cfg
        phase=IR.observe_phase(ctx.model,ctx.search,initial,local_cfg;retain,tight,
            stop_at_saved_times=cfg.domain_atol==0)
        append!(attempts,phase.attempts)
        dest=destination(ctx,phase;reference);status=phase.status;orbit=nothing
        if cycles && status=="unresolved"
            candidate=periodic_candidate(ctx,phase.final,cfg;horizon=1000.)
            orbit=candidate.orbit
            if orbit!==nothing && orbit.validation==NumericallyValidatedPeriodicOrbit && orbit.stability==PeriodicOrbitAttracting
                status="periodic_compatible";dest="oscillatory"
            end
        end
        status!="unresolved" && break
    end
    phase=merge(phase,(;attempts))
    (;status,destination=dest,final=phase.final,phase,orbit,
        recovery=status=="compatible" && dest in ("rest","active"))
end

function save_observation(dir,name,result)
    mkpath(dir);IR.save_phase(dir,name,result.phase)
    write_record(joinpath(dir,name*"_summary.toml"),(;result.status,result.destination,result.recovery))
    if result.orbit!==nothing
        orbit=result.orbit
        write_record(joinpath(dir,name*"_orbit.toml"),Dict(string(k)=>getproperty(orbit,k) for k in propertynames(orbit) if k!=:solution))
    end
end

function pulse(ctx,initial,be,bi,duration,cfg;retain=false,tight=false)
    # Explicit constant-input solve followed by baseline continuation; no
    # half-open endpoint ambiguity and no snapping to a destination root.
    driven=M.model(ctx.p,be,bi)
    times=retain ? collect(range(0,duration;length=101)) : [0.,duration]
    sol=solve_point_model(initial,(0.,duration),driven;saveat=times,
        tstops=cfg.domain_atol==0 ? times : (),
        dense=false,save_everystep=false,abstol=cfg.abstol/(tight ? 10 : 1),reltol=cfg.reltol/(tight ? 10 : 1),
        domain_atol=cfg.domain_atol,maxiters=cfg.maxiters)
    if !SciMLBase.successful_retcode(sol)
        return (;status="integration_failed",destination="integration_failed",recovery=false,final=last(sol.u),phase=nothing,orbit=nothing,pulse_solution=sol)
    end
    merge(observe(ctx,last(sol.u),cfg;reference=ctx,retain,tight,cycles=true),(;pulse_solution=sol))
end

function source_states(ctx,cfg,dir)
    sources=NamedTuple[];probes=NamedTuple[];orbits=Any[]
    for k in IR.sink_indices(ctx.search)
        push!(sources,(;id="root_$k",role=label(ctx,k),initial=copy(ctx.search.equilibria[k].state),phase=0.))
    end
    emax,imax=M.state_bounds(ctx.model)
    for (k,initial) in enumerate([[e,i] for e in range(0,emax;length=cfg.raw["state_probe_grid"])
        for i in range(0,imax;length=cfg.raw["state_probe_grid"])])
        result=observe(ctx,initial,cfg;cycles=true)
        push!(probes,(;seed=k,E=initial[1],I=initial[2],status=result.status,destination=result.destination))
        if result.orbit!==nothing
            save_observation(joinpath(dir,"probes"),string(k),result)
            o=result.orbit
            if o.validation==NumericallyValidatedPeriodicOrbit && o.stability==PeriodicOrbitAttracting &&
                all(q->abs(q.period-o.period)>1e-4*o.period || maximum(abs.(q.amplitudes-o.amplitudes))>1e-4,orbits)
                push!(orbits,o)
            end
        elseif result.status!="compatible"
            save_observation(joinpath(dir,"probes"),string(k),result)
        end
    end
    mkpath(dir);CSV.write(joinpath(dir,"probes.csv"),probes)
    for (k,o) in enumerate(orbits)
        phases=collect(0:cfg.raw["phase_count"]-1)./cfg.raw["phase_count"]
        for (fraction,state) in zip(phases,periodic_orbit_phases(o,phases))
            push!(sources,(;id="cycle_$(k)_phase_$(fraction)",role="oscillatory",initial=state,phase=fraction))
        end
    end
    (;sources,orbits)
end

function transition_map(ctx,source,cfg,bounds,dir;getcontext)
    if N.checked_unit(dir);return TOML.parsefile(joinpath(dir,"summary.toml"));end
    mkpath(dir)
    control=observe(ctx,source.initial,cfg;reference=ctx,retain=true,cycles=source.role=="oscillatory")
    save_observation(dir,"held_input",control)
    rows=NamedTuple[]
    function evaluate(be,bi)
        target=getcontext(be,bi)
        result=observe(target,source.initial,cfg;reference=ctx,cycles=true)
        push!(rows,(;source=source.id,source_role=source.role,phase=source.phase,
            baseline_E=ctx.be,baseline_I=ctx.bi,B_E=be,B_I=bi,status=result.status,
            destination=result.destination,recovery=result.recovery,E=result.final[1],I=result.final[2]))
        result
    end
    xs=input_axis(cfg,bounds.B_E;extra=[ctx.be]);ys=input_axis(cfg,bounds.B_I;extra=[ctx.bi])
    sampled=M.sample_rectangle(evaluate,xs,ys;width=cfg.raw["input_width"],
        key=x->(x.status,x.destination),max_evaluations=cfg.raw["transition_budget"])
    CSV.write(joinpath(dir,"destinations.csv"),rows);CSV.write(joinpath(dir,"cells.csv"),sampled.leaves)
    controls=filter(r->r.recovery && r.B_E<=ctx.be && r.B_I>=ctx.bi,rows)
    pareto=M.nondominated([(;E_withdrawal=ctx.be-r.B_E,I_stimulation=r.B_I-ctx.bi,destination=r.destination) for r in controls])
    isempty(pareto) || CSV.write(joinpath(dir,"pareto.csv"),pareto)
    # Independently confirm each destination category, each Pareto point, and
    # the pure E-withdrawal endpoints; retain actual trajectories.
    chosen=Tuple{Float64,Float64}[(ctx.be,ctx.bi),(0.,ctx.bi)]
    for dest in unique(r.destination for r in rows)
        candidates=sort(filter(r->r.destination==dest,rows);by=r->(r.B_E,r.B_I))
        push!(chosen,(first(candidates).B_E,first(candidates).B_I))
    end
    append!(chosen,[(ctx.be-r.E_withdrawal,ctx.bi+r.I_stimulation) for r in pareto])
    confirmations=NamedTuple[]
    for (k,(be,bi)) in enumerate(unique(chosen))
        target=context(ctx.p,be,bi,cfg;tight=true,grid=cfg.raw["confirmation_grid"])
        result=observe(target,source.initial,cfg;reference=ctx,retain=true,tight=true,cycles=true)
        save_observation(joinpath(dir,"confirmation"),string(k),result)
        push!(confirmations,(;B_E=be,B_I=bi,status=result.status,destination=result.destination,recovery=result.recovery))
    end
    CSV.write(joinpath(dir,"confirmations.csv"),confirmations)
    summary=Dict("source"=>source.id,"role"=>source.role,"observations"=>length(rows),
        "destinations"=>sort(unique(r.destination for r in rows)),"recoveries"=>count(r->r.recovery,rows),
        "phase_signature"=>bytes2hex(SHA.sha256(join(["$(r.B_E),$(r.B_I),$(r.status),$(r.destination)" for r in sort(rows;by=r->(r.B_E,r.B_I))],"\n"))),
        "budget_unresolved_cells"=>count(x->x.status=="budget_unresolved",sampled.leaves))
    write_record(joinpath(dir,"summary.toml"),summary);N.complete_unit(dir);summary
end

function pulse_maps(ctx,source,cfg,bounds,dir)
    N.checked_unit(dir) && return
    mkpath(dir);rows=NamedTuple[];boundaries=NamedTuple[];unresolved=NamedTuple[]
    key(x)=(x.status,x.destination)
    function trial(be,bi,duration,axis,amplitude)
        result=pulse(ctx,source.initial,be,bi,duration,cfg)
        push!(rows,(;source=source.id,source_role=source.role,phase=source.phase,axis,amplitude,duration,
            B_E=be,B_I=bi,status=result.status,destination=result.destination,recovery=result.recovery,
            switch_E=last(result.pulse_solution.u)[1],switch_I=last(result.pulse_solution.u)[2]))
        result
    end
    for duration in cfg.raw["durations"],axis in ("E_increase","E_withdrawal","I_increase","I_withdrawal")
        cap=axis=="E_increase" ? bounds.B_E-ctx.be : axis=="E_withdrawal" ? ctx.be :
            axis=="I_increase" ? bounds.B_I-ctx.bi : ctx.bi
        inputs(a)=axis=="E_increase" ? (ctx.be+a,ctx.bi) : axis=="E_withdrawal" ? (ctx.be-a,ctx.bi) :
            axis=="I_increase" ? (ctx.be,ctx.bi+a) : (ctx.be,ctx.bi-a)
        f(a)=trial(inputs(a)...,duration,axis,a)
        sample=M.sample_line(f,0.,cap;step=max(min(1.,cap/8),eps()),width=cfg.raw["input_width"],key,
            max_evaluations=cfg.raw["line_budget"])
        for (a,b) in sample.unresolved;push!(unresolved,(;axis,duration,lower=a,upper=b));end
        amounts=sort(collect(keys(sample.cache)))
        for (a,b) in zip(amounts,amounts[2:end])
            key(sample.cache[a])==key(sample.cache[b]) && continue
            push!(boundaries,(;axis,duration,lower=a,upper=b,lower_destination=sample.cache[a].destination,
                upper_destination=sample.cache[b].destination,status=b-a<=cfg.raw["input_width"] ? "sampled_bracket" : "budget_unresolved"))
        end
    end
    # Joint withdrawal/stimulation costs remain two-dimensional.
    for duration in cfg.raw["durations"]
        ctx.be>0 || continue
        evaluate(a,b)=trial(ctx.be-a,ctx.bi+b,duration,"joint",a)
        sampled=M.sample_rectangle(evaluate,[0.,ctx.be/2,ctx.be],[0.,(bounds.B_I-ctx.bi)/2,bounds.B_I-ctx.bi];
            width=cfg.raw["input_width"],key,max_evaluations=cfg.raw["transition_budget"])
        CSV.write(joinpath(dir,"joint_$(duration)_cells.csv"),sampled.leaves)
        wins=[(;E_withdrawal=a,I_stimulation=b,destination=x.destination) for ((a,b),x) in sampled.cache if x.recovery]
        isempty(wins) || CSV.write(joinpath(dir,"joint_$(duration)_pareto.csv"),M.nondominated(wins))
    end
    CSV.write(joinpath(dir,"pulses.csv"),rows)
    isempty(boundaries) || CSV.write(joinpath(dir,"boundaries.csv"),boundaries)
    isempty(unresolved) || CSV.write(joinpath(dir,"unresolved_intervals.csv"),unresolved)
    # Confirm the lexicographically first witness for each observed destination.
    for dest in sort(unique(r.destination for r in rows))
        dest in ("integration_failed","unresolved") && continue
        row=first(sort(filter(r->r.destination==dest,rows);by=r->(r.duration,r.B_E,r.B_I)))
        result=pulse(ctx,source.initial,row.B_E,row.B_I,row.duration,cfg;retain=true,tight=true)
        result.phase===nothing && continue
        save_observation(joinpath(dir,"confirmation"),dest,result)
        write_trajectory_csv(joinpath(dir,"confirmation",dest*"_pulse.csv"),result.pulse_solution)
    end
    N.complete_unit(dir)
end

function displacement(ctx,cfg,dir)
    N.checked_unit(dir) && return
    mkpath(dir);rows=NamedTuple[];summary=Dict{String,Any}()
    for source in ("herald","seizure")
        haskey(ctx.roles,source) || continue
        initial=ctx.search.equilibria[ctx.roles[source]].state
        f(a)=observe(ctx,[initial[1]-a,initial[2]],cfg;reference=ctx)
        sample=M.sample_line(f,0.,initial[1];step=initial[1]/32,width=cfg.raw["displacement_width"],
            key=x->(x.status,x.destination),max_evaluations=cfg.raw["line_budget"])
        success=sort([a for (a,x) in sample.cache if x.recovery])
        summary[source]=isempty(success) ? "not_observed" : first(success)
        for (a,x) in sort(collect(sample.cache);by=first)
            push!(rows,(;source,reduction=a,status=x.status,destination=x.destination,recovery=x.recovery))
        end
        if !isempty(success)
            confirmed=observe(ctx,[initial[1]-first(success),initial[2]],cfg;reference=ctx,retain=true,tight=true)
            save_observation(dir,source,confirmed)
        end
    end
    if all(k->haskey(summary,k) && summary[k] isa Real && summary[k]>0,("herald","seizure"))
        summary["seizure_over_herald"]=summary["seizure"]/summary["herald"]
    else
        summary["seizure_over_herald"]="unavailable"
    end
    isempty(rows) || CSV.write(joinpath(dir,"displacement.csv"),rows)
    write_record(joinpath(dir,"summary.toml"),summary);N.complete_unit(dir)
end

function hysteresis(ctx,source,cfg,bounds,dir;getcontext)
    N.checked_unit(dir) && return
    mkpath(dir);rows=NamedTuple[]
    for axis in ("B_E","B_I")
        base=axis=="B_E" ? ctx.be : ctx.bi;cap=axis=="B_E" ? bounds.B_E : bounds.B_I
        at(x)=axis=="B_E" ? getcontext(x,ctx.bi) : getcontext(ctx.be,x)
        # Each direction begins at the declared source, then carries actual
        # endpoints. Halving restarts from the same pre-step state.
        for first_direction in (-1,1)
            initial=copy(source.initial);current=base
            previous=source.role=="unassigned" ? "equilibrium" : source.role
            for target in (first_direction<0 ? (0.,cap,base) : (cap,0.,base))
                step=min(1.,abs(target-current));count=0
                while abs(target-current)>1e-12 && count<cfg.raw["line_budget"]
                    next=current+sign(target-current)*min(step,abs(target-current))
                    trial=observe(at(next),initial,cfg;reference=ctx,cycles=true)
                    if trial.destination!=previous && abs(next-current)>cfg.raw["input_width"]
                        step/=2;continue
                    end
                    push!(rows,(;axis,first_direction,input=next,previous_input=current,
                        status=trial.status,destination=trial.destination,E=trial.final[1],I=trial.final[2]))
                    trial.status=="integration_failed" && break
                    current=next;initial=trial.final;previous=trial.destination;count+=1
                    step=min(1.,max(cfg.raw["input_width"],step*2))
                end
                abs(target-current)>1e-12 && push!(rows,(;axis,first_direction,input=current,previous_input=current,
                    status="budget_unresolved",destination="unresolved",E=initial[1],I=initial[2]))
            end
        end
    end
    isempty(rows) || CSV.write(joinpath(dir,"sweeps.csv"),rows)
    N.complete_unit(dir)
end

function characterize(p,cfg,dir;detailed=true,screen=false)
    g=geometry(p,cfg,joinpath(dir,"geometry");screen)
    detailed || return g
    confirm_geometry(p,cfg,joinpath(dir,"geometry"))
    bounds=M.input_bounds(M.model(p);initial=cfg.raw["initial_input_bound"],epsilon=cfg.raw["tail_tolerance"],max_expansions=cfg.raw["maximum_expansions"])
    reps=g["representatives"]
    for (index,b) in enumerate(reps)
        base_dir=joinpath(dir,"responses","baseline_$index")
        N.checked_unit(base_dir) && continue
        ctx=context(p,b...,cfg;tight=true,grid=cfg.raw["confirmation_grid"])
        save_context(base_dir,ctx)
        cache=Dict{Tuple{Float64,Float64},Any}((ctx.be,ctx.bi)=>ctx)
        getcontext(be,bi)=get!(() -> context(p,be,bi,cfg),cache,(Float64(be),Float64(bi)))
        discovery=source_states(ctx,cfg,base_dir)
        summaries=Dict{String,Any}[]
        for source in discovery.sources
            sub=joinpath(base_dir,source.id)
            push!(summaries,transition_map(ctx,source,cfg,bounds,joinpath(sub,"sustained");getcontext))
            pulse_maps(ctx,source,cfg,bounds,joinpath(sub,"pulses"))
            hysteresis(ctx,source,cfg,bounds,joinpath(sub,"hysteresis");getcontext)
        end
        # Refine phase coverage whenever a cycle's sampled phases differ in
        # observed destination repertoire. Intermediate phases are additional
        # observations, not an assertion of phase completeness.
        for (k,orbit) in enumerate(discovery.orbits)
            group=filter(x->startswith(x["source"],"cycle_$(k)_"),summaries)
            if length(unique(x["phase_signature"] for x in group))>1
                count=cfg.raw["phase_confirmation_count"]
                for fraction in collect(1:2:count-1)./count
                    initial=only(periodic_orbit_phases(orbit,[fraction]))
                    source=(;id="cycle_$(k)_phase_$(fraction)",role="oscillatory",initial,phase=fraction)
                    sub=joinpath(base_dir,source.id)
                    push!(summaries,transition_map(ctx,source,cfg,bounds,joinpath(sub,"sustained");getcontext))
                    pulse_maps(ctx,source,cfg,bounds,joinpath(sub,"pulses"))
                end
            end
        end
        displacement(ctx,cfg,joinpath(base_dir,"displacement"))
        write_record(joinpath(base_dir,"summary.toml"),Dict("B_E"=>ctx.be,"B_I"=>ctx.bi,"sources"=>summaries,
            "validated_cycles"=>length(discovery.orbits)))
        N.complete_unit(base_dir)
        println(p["id"]," baseline ",index,"/",length(reps)," sources=",length(discovery.sources));flush(stdout)
    end
    g
end

function expansion_cases(cfg,known)
    e=cfg.raw["expansion"];result=Dict{String,Any}[]
    for p in known,(axis,lo,hi,step) in zip(e["axes"],e["lower"],e["upper"],e["offsets"]),sign in (-1,1)
        value=p[axis]+sign*step
        lo<=value<=hi || continue
        push!(result,merge(p,Dict(axis=>value,"id"=>p["id"]*"_"*axis*"_"*string(value),"parent"=>p["id"],"changed_axis"=>axis)))
    end
    for family in ("figure3","figure4"),n in 1:cfg.raw["halton_per_family"]
        p=copy(first(filter(p->p["family"]==family,known)))
        for (axis,lo,hi,base) in zip(e["axes"],e["lower"],e["upper"],(2,3,5,7))
            p[axis]=lo+N.NarrativeModels.radical_inverse(n,base)*(hi-lo)
        end
        p["id"]="$(family)_joint_$n";p["baselines"]=[[0.,0.],[.35,0.],[1.,1.]]
        push!(result,p)
    end
    result
end

function behavioral_screen(p,cfg,dir)
    path=joinpath(dir,"behavior.toml")
    isfile(path) && return TOML.parsefile(path)
    rows=NamedTuple[];regimes=String[]
    for b in p["baselines"]
        ctx=context(p,b...,cfg;grid=11)
        local_rows=NamedTuple[]
        for k in IR.sink_indices(ctx.search)
            initial=ctx.search.equilibria[k].state
            for (name,be,bi) in (("withdraw_E",0.,ctx.bi),("half_E",ctx.be/2,ctx.bi),
                ("increase_I_1",ctx.be,ctx.bi+1),("increase_I_4",ctx.be,ctx.bi+4))
                target=context(p,be,bi,cfg)
                result=observe(target,initial,cfg;reference=ctx,cycles=true)
                push!(local_rows,(;B_E=ctx.be,B_I=ctx.bi,source=label(ctx,k),source_index=k,
                    protocol=name,destination=result.destination,status=result.status))
            end
            for (name,be,bi) in (("pulse_E_1",ctx.be+1,ctx.bi),("pulse_I_1",ctx.be,ctx.bi+1))
                result=pulse(ctx,initial,be,bi,100.,cfg)
                push!(local_rows,(;B_E=ctx.be,B_I=ctx.bi,source=label(ctx,k),source_index=k,
                    protocol=name,destination=result.destination,status=result.status))
            end
        end
        append!(rows,local_rows)
        transitions=join(sort(unique("$(r.source):$(r.protocol):$(r.status):$(r.destination)" for r in local_rows)),"|")
        push!(regimes,signature(ctx)*";"*transitions)
    end
    isempty(rows) || CSV.write(joinpath(dir,"behavior.csv"),rows)
    data=Dict("regimes"=>sort(unique(regimes)),"observations"=>length(rows),"purpose"=>"fixed-protocol screening, not absence of other transitions")
    write_record(path,data);data
end

function parallel_cases(f,items)
    # Two independent cases at a time limits memory while retaining deterministic
    # result ordering. All mutations stay inside each case's artifact directory.
    results=Any[]
    for first_index in 1:2:length(items)
        group=items[first_index:min(first_index+1,length(items))]
        jobs=[Threads.@spawn f(item) for item in group]
        append!(results,fetch.(jobs))
    end
    results
end

function run_study(config_path,output;stage="all",case_filter=nothing,smoke=false)
    stage in ("all","geometry","responses","expand") || throw(ArgumentError("unknown stage"))
    cfg=load_config(config_path;smoke);output=abspath(output)
    metadata=initialize(config_path,cfg,output;stage,case_filter)
    known=anchors();summaries=Dict{String,Any}[]
    selected_anchors=filter(p->case_filter===nothing || occursin(case_filter,p["id"]),known)
    append!(summaries,parallel_cases(selected_anchors) do p
        g=characterize(p,cfg,joinpath(output,"anchors",p["id"]);detailed=stage in ("all","responses"))
        println("characterized ",p["id"]);flush(stdout);g
    end)
    if stage in ("all","expand")
        cases=expansion_cases(cfg,known);smoke && (cases=cases[1:2])
        registry=Dict(p["id"]=>p for p in vcat(known,cases))
        signatures=Dict(s["id"]=>join(s["signatures"],";") for s in summaries)
        behavior_signatures=Dict{String,String}()
        for p in known
            base_dir=joinpath(output,"anchors",p["id"]);mkpath(base_dir)
            behavior_signatures[p["id"]]=join(behavioral_screen(p,cfg,base_dir)["regimes"],";")
        end
        budgets=Dict("figure3"=>0,"figure4"=>0);representatives=Dict{String,String}()
        index=1
        while index<=length(cases)
            p=cases[index];index+=1;family=p["family"]
            budgets[family]>=cfg.raw["maximum_additional_per_family"] && continue
            g=characterize(p,cfg,joinpath(output,"expansion",p["id"]);detailed=false,screen=true)
            push!(summaries,g);budgets[family]+=1
            sig=join(g["signatures"],";");signatures[p["id"]]=sig
            behavior=behavioral_screen(p,cfg,joinpath(output,"expansion",p["id"]))
            behavior_signatures[p["id"]]=join(behavior["regimes"],";")
            for regime in behavior["regimes"]
                get!(representatives,regime,p["id"])
            end
            changed=haskey(p,"parent") && (get(signatures,p["parent"],sig)!=sig ||
                get(behavior_signatures,p["parent"],"")!=behavior_signatures[p["id"]])
            if changed && !get(p,"refined",false)
                parent=registry[p["parent"]];axis=p["changed_axis"]
                midpoint=(p[axis]+parent[axis])/2
                refined=merge(p,Dict(axis=>midpoint,"id"=>p["id"]*"_midpoint","refined"=>true))
                registry[refined["id"]]=refined;push!(cases,refined)
            end
            println("expansion ",p["id"]," ",budgets);flush(stdout)
        end
        write_record(joinpath(output,"expansion_selection.toml"),Dict("representatives"=>representatives,"budgets"=>budgets,
            "selection"=>"first deterministic baseline coexistence and fixed-protocol response representative; this does not equate all parameterizations in that class"))
        representative_ids=sort(unique(collect(values(representatives))))
        parallel_cases(representative_ids) do id
            characterize(registry[id],cfg,joinpath(output,"representatives",id);detailed=true)
        end
    end
    write_record(joinpath(output,"summary.toml"),Dict("cases"=>summaries))
    metadata["last_stage"]=stage;metadata["case_filter"]=something(case_filter,"all")
    metadata["completed"]=stage=="all" && case_filter===nothing
    N.complete_replay!(metadata,"run_input_response_study.jl";stage,case_filter,smoke)
    write_record(joinpath(output,"metadata.toml"),metadata);N.Evidence.artifact_checksums(output)
    output
end

function main(args=ARGS)
    opts=Dict{String,String}();smoke=false;k=1
    while k<=length(args)
        if args[k]=="--smoke";smoke=true;k+=1;continue;end
        args[k] in ("--config","--output","--stage","--case") && k<length(args) || throw(ArgumentError("invalid argument"))
        haskey(opts,args[k]) && throw(ArgumentError("duplicate argument"))
        opts[args[k]]=args[k+1];k+=2
    end
    haskey(opts,"--output") || throw(ArgumentError("--output required"))
    run_study(get(opts,"--config",joinpath(ROOT,"experiments/input_response.toml")),opts["--output"];
        stage=get(opts,"--stage","all"),case_filter=get(opts,"--case",nothing),smoke)
end
end
if abspath(PROGRAM_FILE)==@__FILE__;InputResponseStudy.main();end
