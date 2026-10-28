"""Question-led measurements for the point-model figure redesign."""
module FigureDetails
using FailureOfInhibition2025, LinearAlgebra
import CSV, TOML
include("run_figure_screen.jl")
const F=FigureScreen
const S=F.S
const ROOT=F.ROOT

function initialize(config_path, output)
    cfg=S.load_config(config_path)
    metadata=F.initialize(config_path,cfg,output)
    relative="scripts/run_figure_details.jl"
    hash=S.N.Evidence.file_hash(joinpath(ROOT,relative))
    get(metadata["source_sha256"],relative,hash)==hash || error("detail source changed; use a new output")
    S.N.Evidence.archive_sources!(metadata, output, (relative,); missing_only=true)
    metadata["source_sha256"][relative]==hash || error("archived detail source changed")
    metadata["purpose"]="Targeted figure measurements; finite discovery, no completeness certification"
    metadata["replay_from_artifact_directory"]="julia --project=source source/scripts/run_figure_details.jl config.toml SCREEN_OUTPUT replay STAGE"
    S.write_record(joinpath(output,"metadata.toml"),metadata)
    cfg,metadata
end

function expected_selection(cfg,screen,config_sha256)
    all_cases=F.registry(cfg)
    metadata=TOML.parsefile(joinpath(screen,"metadata.toml"))
    get(metadata,"completed",false) || error("screen is incomplete")
    get(metadata,"config_sha256",nothing)==config_sha256 || error("screen configuration differs from detail study")
    F.A.verify_study_manifest(screen)
    for (relative,hash) in metadata["source_sha256"]
        S.N.Evidence.file_hash(joinpath(ROOT,relative))==hash || error("screen source is stale: $relative")
    end
    for p in all_cases
        S.N.checked_unit(joinpath(screen,p["category"],p["id"])) || error("incomplete screen case")
    end
    byid=Dict(p["id"]=>p for p in all_cases)
    preferred=[("selective_withdrawal","reference switching and selective control"),
        ("narrative","contrast with small available tonic E"),
        ("input_dependent","weak recurrence, input dependence, and periodic-source lead"),
        ("threshold_to_active","intervention destination contrast"),
        ("input_dependent_e_to_e_4.25","nearby contrast in paired withdrawal"),
        ("figure4_joint_29","higher-timescale selective-withdrawal lead")]
    selected=Dict{String,Any}[];covered=Set{String}()
    regimes(p)=TOML.parsefile(joinpath(screen,p["category"],p["id"],"behavior.toml"))["regimes"]
    for (id,reason) in preferred
        haskey(byid,id) || error("missing predeclared contrast $id")
        p=copy(byid[id]);p["selection_reason"]=reason
        push!(selected,p);union!(covered,regimes(p))
    end
    for p in all_cases
        length(selected)>=cfg.raw["redesign"]["detailed_limit"] && break
        any(q->q["id"]==p["id"],selected) && continue
        observed=regimes(p)
        all(in(covered),observed) && continue
        q=copy(p);q["selection_reason"]="first remaining registry case with a new measured screen signature"
        push!(selected,q);union!(covered,observed)
    end
    record=Dict("cases"=>selected,"screen_metadata_sha256"=>S.N.Evidence.file_hash(joinpath(screen,"metadata.toml")),
        "screen_manifest_sha256"=>S.N.Evidence.file_hash(joinpath(screen,"checksums.toml")),
        "selection"=>"Six predeclared contrast leads, then deterministic uncovered screen signatures; not prevalence")
    record
end

function select_cases(cfg,screen,output,config_sha256)
    expected=expected_selection(cfg,screen,config_sha256)
    path=joinpath(output,"selection.toml")
    if isfile(path)
        TOML.parsefile(path)==expected || error("selection differs from the frozen registry and checked screen")
    else
        S.write_record(path,expected)
    end
    expected["cases"]
end

function confirmed_context(p,b,cfg,dir)
    ctx=S.context(p,b...,cfg;tight=true,grid=cfg.raw["confirmation_grid"])
    independent=find_equilibria(ctx.model;seeds=[[e,i] for e in range(0,.5;length=41)
        for i in range(0,S.M.state_bounds(ctx.model)[2];length=41)],
        options=EquilibriumOptions(residual_atol=1e-11,solver_reltol=1e-11))
    consistent=length(independent.equilibria)==length(ctx.search.equilibria) &&
        all(r->any(q->maximum(abs.(r.state-q.state))<1e-7,independent.equilibria),ctx.search.equilibria)
    S.save_context(dir,ctx)
    S.write_record(joinpath(dir,"independent.toml"),(;consistent,search=S.N.Evidence.context_record(independent)))
    consistent || error("independent root search disagrees: $(p["id"]) $b")
    ctx
end

function target_baselines(p)
    values=copy(p["baselines"])
    p["id"]=="input_dependent" && push!(values,[2.,1.])
    unique(values)
end

function characterize_case(p,cfg,output)
    dir=joinpath(output,"cases",p["id"])
    S.N.checked_unit(dir) && return
    S.geometry(p,cfg,joinpath(dir,"geometry"))
    S.confirm_geometry(p,cfg,joinpath(dir,"geometry"))
    checks=CSV.File(joinpath(dir,"geometry_confirmation","independent_searches.csv"))
    all(r->r.consistent,checks) || error("geometry confirmation disagreement")
    F.A.archive_geometry(joinpath(dir,"geometry"))
    baselines=target_baselines(p)
    for (index,b) in enumerate(baselines)
        sub=joinpath(dir,"baseline_$index")
        S.N.checked_unit(sub) && continue
        ctx=confirmed_context(p,b,cfg,sub)
        bounds=S.M.input_bounds(ctx.model;initial=cfg.raw["initial_input_bound"],
            epsilon=cfg.raw["tail_tolerance"],max_expansions=cfg.raw["maximum_expansions"])
        discovery=S.source_states(ctx,cfg,sub)
        cache=Dict{Tuple{Float64,Float64},Any}((ctx.be,ctx.bi)=>ctx)
        at(be,bi)=get!(() -> S.context(p,be,bi,cfg),cache,(Float64(be),Float64(bi)))
        summaries=Dict{String,Any}[]
        for source in discovery.sources
            folder=joinpath(sub,source.id)
            push!(summaries,S.transition_map(ctx,source,cfg,bounds,joinpath(folder,"sustained");getcontext=at))
            S.pulse_maps(ctx,source,cfg,bounds,joinpath(folder,"pulses"))
            # Equilibrium sweeps carry endpoints; cycles retain phase-specific pulse/sustained maps.
            source.role=="oscillatory" || S.hysteresis(ctx,source,cfg,bounds,joinpath(folder,"hysteresis");getcontext=at)
        end
        for (k,orbit) in enumerate(discovery.orbits)
            group=filter(x->startswith(x["source"],"cycle_$(k)_"),summaries)
            if length(unique(x["phase_signature"] for x in group))>1
                count=cfg.raw["phase_confirmation_count"]
                for fraction in collect(1:2:count-1)./count
                    source=(;id="cycle_$(k)_phase_$(fraction)",role="oscillatory",
                        initial=only(periodic_orbit_phases(orbit,[fraction])),phase=fraction)
                    folder=joinpath(sub,source.id)
                    push!(summaries,S.transition_map(ctx,source,cfg,bounds,joinpath(folder,"sustained");getcontext=at))
                    S.pulse_maps(ctx,source,cfg,bounds,joinpath(folder,"pulses"))
                end
            end
            CSV.write(joinpath(sub,"orbit_$k.csv"),[(;time=t,E=x[1],I=x[2]) for (t,x) in zip(orbit.times,orbit.states)])
            S.write_record(joinpath(sub,"orbit_$k.toml"),orbit_record(orbit))
        end
        S.displacement(ctx,cfg,joinpath(sub,"displacement"))
        S.write_record(joinpath(sub,"summary.toml"),Dict("B_E"=>ctx.be,"B_I"=>ctx.bi,
            "sources"=>summaries,"validated_cycles"=>length(discovery.orbits)))
        S.N.complete_unit(sub)
        println("detailed ",p["id"]," baseline ",index,"/",length(baselines));flush(stdout)
    end
    S.write_record(joinpath(dir,"summary.toml"),Dict("baselines"=>baselines,"parameters"=>p))
    S.N.complete_unit(dir)
end

orbit_record(o)=(;period=o.period,validation=string(o.validation),stability=string(o.stability),
    amplitudes=o.amplitudes,closure_residual=o.closure_residual,phase_residual=o.phase_residual,
    equation_residual=o.equation_residual,transverse_real=real(o.transverse_multiplier),
    transverse_imag=imag(o.transverse_multiplier),refinement_difference=o.refinement_difference,
    reasons=string.(o.reasons))

function mechanism(p,cfg,output)
    dir=joinpath(output,"mechanism");S.N.checked_unit(dir) && return
    ref=S.context(p,.35,0.,cfg;tight=true,grid=41)
    mkpath(dir);response_rows=NamedTuple[];nullclines=NamedTuple[];roots=NamedTuple[]
    for (condition,control) in (("monotone",true),("failure",false))
        model=S.M.model(p,.35,0.;control)
        search=find_equilibria(model;seeds=[[e,i] for e in range(0,.5;length=41) for i in range(0,.5;length=41)])
        roles=S.M.roles(model,search)
        for (k,r) in enumerate(search.equilibria)
            push!(roots,(;condition,root=k,E=r.state[1],I=r.state[2],stability=string(r.stability.classification),
                role=something(findfirst(==(k),roles),"unassigned")))
        end
        for input in range(-1.,13.;length=701)
            push!(response_rows,(;condition,input,response=response(model.inhibitory.response,input)))
        end
        c=model.coupling
        for input in range(-c.i_to_e+.35,c.e_to_e+.35;length=6001)
            e=S.M.occupancy(model.excitatory.response,input)
            i=(c.e_to_e*e+.35-input)/c.i_to_e
            0<=i<=1 && push!(nullclines,(;condition,population="E",input,E=e,I=i))
        end
        for input in range(-c.i_to_i,c.e_to_i;length=6001)
            i=S.M.occupancy(model.inhibitory.response,input)
            e=(input+c.i_to_i*i)/c.e_to_i
            0<=e<=1 && push!(nullclines,(;condition,population="I",input,E=e,I=i))
        end
    end
    CSV.write(joinpath(dir,"responses.csv"),response_rows)
    CSV.write(joinpath(dir,"nullclines.csv"),nullclines);CSV.write(joinpath(dir,"roots.csv"),roots)
    star=ref.search.equilibria[ref.roles["seizure"]].state
    inhibitory_input=p["e_to_i"]*star[1]-p["i_to_i"]*star[2]
    S.write_record(joinpath(dir,"restriction.toml"),Dict("E_star"=>star[1],"I_star"=>star[2],
        "inhibitory_input"=>inhibitory_input,"response_peak_input"=>(get(p,"theta_on",4.)+p["theta_off"])/2,
        "descending_branch_condition"=>inhibitory_input>=(get(p,"theta_on",4.)+p["theta_off"])/2,
        "drive_condition"=>"Both drive increments are nonnegative relative to baseline throughout the protocol"))
    S.N.complete_unit(dir)
end

"""Pick the midpoint of the widest observed successful interval, then remeasure it."""
function interior_pulse(rows,destination;duration=100.)
    candidates=NamedTuple[]
    for axis in ("E_increase","I_increase","E_withdrawal","I_withdrawal")
        ordered=sort(filter(r->r.axis==axis && r.duration==duration,rows);by=r->r.amplitude)
        start=nothing
        for (i,r) in enumerate(ordered)
            success=r.status=="compatible" && r.destination==destination
            if success
                start===nothing && (start=i)
                if i==length(ordered) || !(ordered[i+1].status=="compatible" && ordered[i+1].destination==destination)
                    lo=ordered[start].amplitude;hi=r.amplitude
                    hi>lo && push!(candidates,(;axis,amplitude=(lo+hi)/2,duration,width=hi-lo))
                    start=nothing
                end
            else
                start=nothing
            end
        end
    end
    isempty(candidates) && return nothing
    first(sort(candidates;by=x->(-x.width,x.axis,x.amplitude)))
end

input_for(ctx,w)=w.axis=="E_increase" ? (ctx.be+w.amplitude,ctx.bi) :
    w.axis=="I_increase" ? (ctx.be,ctx.bi+w.amplitude) :
    w.axis=="E_withdrawal" ? (ctx.be-w.amplitude,ctx.bi) : (ctx.be,ctx.bi-w.amplitude)

function sequence(p,cfg,output)
    dir=joinpath(output,"sequence");S.N.checked_unit(dir) && return
    ctx=confirmed_context(p,[.35,0.],cfg,dir)
    route=["rest","active","rest","herald","active","rest","active","seizure"]
    all(haskey(ctx.roles,r) for r in unique(route)) || error("reference sequence roles unavailable")
    initial=copy(ctx.search.equilibria[ctx.roles[first(route)]].state)
    trace=NamedTuple[];inputs=NamedTuple[];events=NamedTuple[];clock=0.
    function append_solution(sol,be,bi)
        for (t,x) in zip(sol.t,sol.u)
            !isempty(trace) && clock+t==last(trace).time && continue
            push!(trace,(;time=clock+t,E=x[1],I=x[2]))
        end
        push!(inputs,(;time=clock,B_E=be,B_I=bi))
        clock+=last(sol.t)
        push!(inputs,(;time=clock,B_E=be,B_I=bi))
    end
    pre=solve_point_model(initial,(0.,50.),ctx.model;saveat=0.:.5:50.,tstops=0.:.5:50.,
        dense=false,save_everystep=false,abstol=cfg.abstol/10,reltol=cfg.reltol/10,domain_atol=0.)
    append_solution(pre,ctx.be,ctx.bi);initial=copy(last(pre.u))
    for (index,(source,destination)) in enumerate(zip(route,route[2:end]))
        path=joinpath(output,"cases",p["id"],"baseline_1","root_$(ctx.roles[source])","pulses","pulses.csv")
        rows=collect(CSV.File(path));w=interior_pulse(rows,destination)
        w===nothing && error("no interior pulse for $source -> $destination; inspect retained pulse map")
        be,bi=input_for(ctx,w)
        trial=S.pulse(ctx,initial,be,bi,w.duration,cfg;retain=true,tight=true)
        trial.status=="compatible" && trial.destination==destination || error("connected route did not qualify at event $index")
        onset=clock;start=copy(initial)
        maximum(abs.(first(trial.pulse_solution.u)-initial))<=1e-12 || error("state reset at pulse onset")
        maximum(abs.(last(trial.pulse_solution.u)-first(trial.phase.solution.u)))<=1e-12 || error("state reset at restoration")
        append_solution(trial.pulse_solution,be,bi);offset=clock
        append_solution(trial.phase.solution,ctx.be,ctx.bi)
        push!(events,(;index,source,destination,axis=w.axis,amplitude=w.amplitude,duration=w.duration,
            onset,offset,followup_end=clock,start_E=start[1],start_I=start[2],
            final_E=trial.final[1],final_I=trial.final[2],status=trial.status))
        S.save_observation(dir,"event_$index",trial)
        initial=copy(trial.final)
    end
    CSV.write(joinpath(dir,"trajectory.csv"),trace);CSV.write(joinpath(dir,"inputs.csv"),inputs)
    CSV.write(joinpath(dir,"events.csv"),events)
    S.write_record(joinpath(dir,"summary.toml"),Dict("route"=>route,"parameters"=>p,
        "B_E"=>ctx.be,"B_I"=>ctx.bi,"state_handoffs"=>"actual endpoints; no snapping or resets"))
    S.N.complete_unit(dir)
end

function timescales(p,b,cfg,output)
    dir=joinpath(output,"timescales",p["id"]*"_"*S.context_id(b...))
    S.N.checked_unit(dir) && return
    ctx=confirmed_context(p,b,cfg,dir);rows=NamedTuple[];candidates=NamedTuple[]
    ratios=collect(range(.2,4.4;length=85))
    for (k,root) in enumerate(ctx.search.equilibria)
        balance=zeros(2,2);point_balance_jacobian!(balance,root.state,ctx.model,0.)
        a,d=balance[1,1],balance[2,2]
        r=a==0 ? NaN : -d/a
        if isfinite(r) && .2<=r<=4.4 && det(balance)>0
            push!(ratios,r)
            push!(candidates,(;root=k,ratio=r,E=root.state[1],I=root.state[2],status="linear_trace_zero_candidate"))
        end
    end
    for ratio in sort(unique(ratios)), (k,root) in enumerate(ctx.search.equilibria)
        m=S.M.model(merge(p,Dict("tau_ratio"=>ratio)),b...);j=zeros(2,2)
        point_jacobian!(j,root.state,m,0.);spectrum=eigvals(j);alpha=maximum(real.(spectrum))
        stability=det(j)<0 ? "Saddle" : abs(alpha)<1e-8 ? "Unresolved" : alpha<0 ? "Attracting" : "Repelling"
        push!(rows,(;ratio,root=k,E=root.state[1],I=root.state[2],stability,alpha,determinant=det(j)))
    end
    CSV.write(joinpath(dir,"equilibria.csv"),rows)
    isempty(candidates) || CSV.write(joinpath(dir,"trace_candidates.csv"),candidates)
    seeds=NamedTuple[];branches=NamedTuple[];stops=NamedTuple[];continued=false
    for ratio in sort(unique(vcat([.2,.4,.5,1.,2.,4.4],p["tau_ratio"])))
        q=merge(p,Dict("tau_ratio"=>ratio));changed=S.context(q,b...,cfg)
        central=filter(r->.01<r.state[1]<.49 && r.stability.classification==Repelling,changed.search.equilibria)
        initials=isempty(central) ? [[.25,.25]] : [r.state.+[.001,0.] for r in central]
        for initial in initials
            candidate=S.periodic_candidate(changed,initial,cfg)
            push!(seeds,(;ratio,initial_E=initial[1],initial_I=initial[2],status=candidate.status))
            o=candidate.orbit
            o===nothing && continue
            o.validation==NumericallyValidatedPeriodicOrbit || continue
            folder=joinpath(dir,"seed_$(ratio)");mkpath(folder)
            S.write_record(joinpath(folder,"orbit.toml"),orbit_record(o))
            CSV.write(joinpath(folder,"orbit.csv"),[(;time=t,E=x[1],I=x[2]) for (t,x) in zip(o.times,o.states)])
            continued && continue
            factory=r->S.M.model(merge(p,Dict("tau_ratio"=>r)),b...)
            continuation=continue_periodic_orbit(factory,o.initial_state,o.period,ratio;parameter_bounds=(.2,4.4),
                options=PeriodicContinuationOptions(max_steps=250,state_scales=(.1,.1)))
            for branch in (continuation.negative,continuation.positive)
                push!(stops,(;direction=branch.direction,termination=string(branch.termination),points=length(branch.points)))
                attempts=[Dict(string(k)=>(k==:orbit ? (a.orbit===nothing ? nothing : orbit_record(a.orbit)) : getproperty(a,k)) for k in propertynames(a)) for a in branch.attempts]
                S.write_record(joinpath(dir,"attempts_$(branch.direction).toml"),Dict("attempts"=>attempts))
                for (index,point) in enumerate(branch.points)
                    orbit=point.orbit
                    push!(branches,(;direction=branch.direction,index,ratio=point.parameter,period=orbit.period,
                        E_min=minimum(x[1] for x in orbit.states),E_max=maximum(x[1] for x in orbit.states),
                        I_min=minimum(x[2] for x in orbit.states),I_max=maximum(x[2] for x in orbit.states),
                        validation=string(orbit.validation),stability=string(orbit.stability),closure=orbit.closure_residual,
                        multiplier=abs(orbit.transverse_multiplier)))
                end
            end
            continued=true
        end
    end
    CSV.write(joinpath(dir,"seeds.csv"),seeds)
    isempty(branches) || CSV.write(joinpath(dir,"cycles.csv"),branches)
    isempty(stops) || CSV.write(joinpath(dir,"stops.csv"),stops)
    S.write_record(joinpath(dir,"parameters.toml"),merge(p,Dict("B_E"=>b[1],"B_I"=>b[2])))
    S.N.complete_unit(dir)
end

function switching(ctx,cfg,witnesses;initial=nothing,output=nothing)
    legs=Dict{String,Any}[]
    function finish(status)
        if output!==nothing
            mkpath(output)
            S.write_record(joinpath(output,"summary.toml"),Dict("status"=>status,"legs"=>legs))
        end
        status
    end
    all(haskey(ctx.roles,r) for r in ("rest","active")) || return finish("sources_unavailable")
    state=initial===nothing ? copy(ctx.search.equilibria[ctx.roles["rest"]].state) : copy(initial)
    for cycle in 1:2, (leg,(w,destination)) in enumerate(zip(witnesses,("active","rest")))
        be,bi=input_for(ctx,w);trial=S.pulse(ctx,state,be,bi,w.duration,cfg)
        if output!==nothing
            name="cycle_$(cycle)_leg_$(leg)"
            mkpath(output)
            write_trajectory_csv(joinpath(output,name*"_pulse.csv"),trial.pulse_solution)
            trial.phase===nothing || S.save_observation(output,name,trial)
            push!(legs,Dict("name"=>name,"cycle"=>cycle,"leg"=>leg,
                "axis"=>w.axis,"amplitude"=>w.amplitude,"duration"=>w.duration,
                "B_E"=>be,"B_I"=>bi,"status"=>trial.status,"destination"=>trial.destination,
                "final_E"=>trial.final[1],"final_I"=>trial.final[2]))
        end
        if trial.status!="compatible" || trial.destination!=destination
            return finish(trial.destination)
        end
        state=copy(trial.final)
    end
    finish("retained")
end

function interventions(p,b,cfg,output)
    dir=joinpath(output,"interventions",p["id"]);S.N.checked_unit(dir) && return
    baseline=confirmed_context(p,b,cfg,dir)
    haskey(baseline.roles,"seizure") || error("intervention source unavailable")
    source=baseline.search.equilibria[baseline.roles["seizure"]].state
    fixed=[(;axis="E_increase",amplitude=.5,duration=100.),(;axis="I_increase",amplitude=3.,duration=100.)]
    axes=[("theta_off",sort(unique(vcat(p["theta_off"],collect(6.:.25:12.))))),
        ("e_to_e",sort(unique(vcat(p["e_to_e"],filter(x->0<=x<=24,p["e_to_e"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6]))))),
        ("e_to_i",sort(unique(vcat(p["e_to_i"],filter(x->12<=x<=28,p["e_to_i"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6])))))]
    rows=NamedTuple[]
    for (axis,values) in axes, value in values
        sub=joinpath(dir,axis*"_"*string(value));q=merge(p,Dict(axis=>value))
        if S.N.checked_unit(sub)
            append!(rows,[(; (Symbol(k)=>v for (k,v) in r)...) for r in TOML.parsefile(joinpath(sub,"summary.toml"))["rows"]]);continue
        end
        ctx=confirmed_context(q,b,cfg,sub)
        response=S.observe(ctx,source,cfg;reference=baseline,retain=true,tight=true,cycles=true)
        S.save_observation(sub,"original_seizure",response)
        fixed_status=switching(ctx,cfg,fixed;output=joinpath(sub,"fixed_switching"))
        remeasured=fixed_status;up=NaN;down=NaN;duration_up=NaN;duration_down=NaN
        if fixed_status!="retained" && all(haskey(ctx.roles,r) for r in ("rest","active"))
            found=Any[]
            for (role,destination) in (("rest","active"),("active","rest"))
                source_state=ctx.search.equilibria[ctx.roles[role]].state
                trials=NamedTuple[];witness=nothing
                for duration in cfg.raw["durations"]
                    for direction in ("E_increase","I_increase"), amplitude in 0.:.25:8.
                        w=(;axis=direction,amplitude,duration)
                        be,bi=input_for(ctx,w);trial=S.pulse(ctx,source_state,be,bi,duration,cfg)
                        push!(trials,(;source=role,axis=direction,amplitude,duration,status=trial.status,destination=trial.destination))
                    end
                    witness=interior_pulse(trials,destination;duration)
                    witness===nothing || break
                end
                CSV.write(joinpath(sub,"$(role)_remeasure.csv"),trials)
                push!(found,witness)
            end
            if all(!isnothing,found)
                remeasured=switching(ctx,cfg,found;output=joinpath(sub,"remeasured_switching"))
                up=found[1].amplitude;down=found[2].amplitude
                duration_up=found[1].duration;duration_down=found[2].duration
            else
                remeasured="not_demonstrated_within_protocol"
            end
        end
        row=(;axis,value,destination=response.destination,status=response.status,fixed_switching=fixed_status,
            remeasured_switching=remeasured,up_amplitude=up,down_amplitude=down,duration_up,duration_down,
            roots=length(ctx.search.equilibria),roles=join(sort(collect(keys(ctx.roles))),","),
            final_E=response.final[1],final_I=response.final[2])
        S.write_record(joinpath(sub,"summary.toml"),Dict("rows"=>[row]));S.N.complete_unit(sub);push!(rows,row)
        println("intervention ",p["id"]," ",axis,"=",value," -> ",response.destination," switching ",fixed_status,"/",remeasured);flush(stdout)
    end
    CSV.write(joinpath(dir,"summary.csv"),rows);S.N.complete_unit(dir)
end

function paired_controls(cases,cfg,output)
    dir=joinpath(output,"paired_controls");S.N.checked_unit(dir) && return
    mkpath(dir);rows=NamedTuple[];comparisons=NamedTuple[]
    for p in cases, (index,b) in enumerate(target_baselines(p))
        b[1]>0 || continue
        ctx=S.context(p,b...,cfg;tight=true,grid=41)
        all(haskey(ctx.roles,r) for r in ("herald","seizure")) || continue
        sources=[ctx.search.equilibria[ctx.roles[r]].state for r in ("herald","seizure")]
        released=S.context(p,0.,b[2],cfg;tight=true,grid=41)
        controls=[S.observe(ctx,x,cfg;reference=ctx,retain=true,tight=true,cycles=true) for x in sources]
        outcomes=[S.observe(released,x,cfg;reference=ctx,retain=true,tight=true,cycles=true) for x in sources]
        sub=joinpath(dir,p["id"]*"_$(index)");mkpath(sub)
        S.save_context(sub,ctx)
        S.save_context(joinpath(sub,"released"),released)
        for (k,role) in enumerate(("herald","seizure"))
            S.save_observation(sub,role*"_held",controls[k]);S.save_observation(sub,role*"_release",outcomes[k])
            controls[k].status=="compatible" && controls[k].destination==role || error("held control changed")
        end
        push!(comparisons,(;id=p["id"],baseline=index,B_E=b[1],B_I=b[2],tau_ratio=p["tau_ratio"],
            herald=outcomes[1].destination,seizure=outcomes[2].destination,
            herald_status=outcomes[1].status,seizure_status=outcomes[2].status))
        p["id"]=="selective_withdrawal" && index==1 || continue
        for duration in cfg.raw["durations"]
            function at(amplitude)
                pair=[S.pulse(ctx,x,b[1]-amplitude,b[2],duration,cfg) for x in sources]
                row_index=length(rows)+1
                traces=joinpath(sub,"pulse_trials","row_$row_index");mkpath(traces)
                for (k,role) in enumerate(("herald","seizure"))
                    write_trajectory_csv(joinpath(traces,role*"_pulse.csv"),pair[k].pulse_solution)
                    pair[k].phase===nothing || S.save_observation(traces,role,pair[k])
                end
                push!(rows,(;id=p["id"],baseline=index,amplitude,duration,
                    herald=pair[1].destination,seizure=pair[2].destination,
                    herald_status=pair[1].status,seizure_status=pair[2].status))
                (pair[1].status,pair[1].destination,pair[2].status,pair[2].destination)
            end
            sample=S.M.sample_line(at,0.,b[1];step=b[1]/8,width=cfg.raw["input_width"],
                key=identity,max_evaluations=cfg.raw["line_budget"])
            S.write_record(joinpath(sub,"pulse_intervals_$(duration).toml"),Dict("unresolved"=>sample.unresolved))
        end
    end
    CSV.write(joinpath(dir,"sustained.csv"),comparisons)
    CSV.write(joinpath(dir,"pulses.csv"),rows);S.N.complete_unit(dir)
end

function run(config_path,screen,output,stage;case_id=nothing)
    cfg,metadata=initialize(config_path,output)
    cases=select_cases(cfg,screen,output,metadata["config_sha256"])
    reference=only(filter(p->p["id"]==cfg.raw["redesign"]["reference"],cases))
    if stage=="cases"
        chosen=case_id===nothing ? cases : filter(p->p["id"]==case_id,cases)
        isempty(chosen) && error("unknown selected case")
        S.parallel_cases(chosen) do p
            characterize_case(p,cfg,output)
        end
    elseif stage=="sequence"
        sequence(reference,cfg,output)
    elseif stage=="mechanism"
        mechanism(reference,cfg,output)
    elseif stage=="timescales"
        timescales(reference,[.35,0.],cfg,output)
        weak=only(filter(p->p["id"]=="input_dependent",cases))
        timescales(weak,[2.,1.],cfg,output)
    elseif stage=="interventions"
        for id in ("selective_withdrawal","threshold_to_active")
            p=only(filter(p->p["id"]==id,cases));interventions(p,first(p["baselines"]),cfg,output)
        end
    elseif stage=="paired"
        paired_controls(cases,cfg,output)
    elseif stage=="finalize"
        required=vcat([joinpath("cases",p["id"]) for p in cases],
            ["sequence","mechanism","paired_controls","interventions/selective_withdrawal","interventions/threshold_to_active",
             "timescales/selective_withdrawal_0.35_0.0","timescales/input_dependent_2.0_1.0"])
        all(S.N.checked_unit(joinpath(output,p)) for p in required) || error("required detail stage incomplete")
        metadata["completed"]=true;metadata["completed_stages"]=required
        S.write_record(joinpath(output,"metadata.toml"),metadata)
    else
        error("unknown stage: $stage")
    end
    S.N.Evidence.artifact_checksums(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS) in (4,5) || error("usage: run_figure_details.jl CONFIG SCREEN OUTPUT STAGE [CASE]")
    FigureDetails.run(ARGS[1:4]...;case_id=length(ARGS)==5 ? ARGS[5] : nothing)
end
