"""Independent equations for checking the figure trajectories; no package RHS calls."""
module VerifyFigureRedesign
using SciMLBase, OrdinaryDiffEqTsit5
import CSV,TOML,SHA
include("figure_packet_io.jl")
const IOChecks=FigurePacketIO
digest(path)=bytes2hex(open(SHA.sha256,path))
sigmoid(x)=x>=0 ? inv(1+exp(-x)) : exp(x)/(1+exp(x))
function independent_rhs!(du,x,p,t)
    q,be,bi=p; e,i=x
    ue=q["e_to_e"]*e-q["i_to_e"]*i+be
    ui=q["e_to_i"]*e-q["i_to_i"]*i+bi
    se=get(q,"slope_e",5.);si=get(q,"slope_i",5.)
    fe=sigmoid(se*(ue-get(q,"theta_e",1.5)))
    on=get(q,"theta_on",4.);off=q["theta_off"]
    # Algebraically the equal-slope difference, evaluated without subtracting saturated sigmoids.
    fi=sigmoid(si*(ui-on))*sigmoid(si*(off-ui))*(-expm1(-si*(off-on)))
    tau=get(q,"tau_e",7.8)
    du[1]=(-e+(1-e)*fe)/tau
    du[2]=(-i+(1-i)*fi)/(tau*q["tau_ratio"])
end
function integrate(initial,p,be,bi,times)
    problem=ODEProblem(independent_rhs!,copy(initial),(0.,last(times)),(p,be,bi))
    solution=solve(problem,Tsit5();abstol=1e-12,reltol=1e-12,saveat=times,tstops=times,
        save_everystep=false,dense=false,maxiters=2_000_000,
        isoutofdomain=(x,p,t)->any(y->y<0 || y>1,x))
    SciMLBase.successful_retcode(solution) || error("independent integration failed")
    solution
end
function compare(path,p,be,bi,root)
    rows=collect(CSV.File(path));times=Float64.(getproperty.(rows,:time))
    solution=integrate([first(rows).E,first(rows).I],p,be,bi,times)
    length(solution.u)==length(rows) || error("independent sample count changed")
    coordinate_error=maximum(max(abs(x[1]-r.E),abs(x[2]-r.I)) for (x,r) in zip(solution.u,rows))
    (;file=relpath(path,root),maximum_coordinate_error=coordinate_error,passed=coordinate_error<1e-6)
end
function verify_observation(details,folder,name,context_dir,reference_dir,p,baseline,status,destination,source;final=nothing)
    file=joinpath(folder,name*".csv")
    trace=collect(CSV.File(file))
    observed=TOML.parsefile(joinpath(folder,name*"_summary.toml"))
    phase=TOML.parsefile(joinpath(folder,name*".toml"))
    observed["status"]==status && observed["destination"]==destination ||
        error("reported observation outcome changed: $folder/$name")
    phase["status"]==(status=="periodic_compatible" ? "unresolved" : status) ||
        error("observation phase status changed: $folder/$name")
    initial_error=max(abs(first(trace).E-source[1]),abs(first(trace).I-source[2]))
    endpoint=[last(trace).E,last(trace).I]
    final_error=maximum(abs.(endpoint.-phase["final"]))
    final===nothing || (final_error=max(final_error,maximum(abs.(endpoint.-final))))
    destination_error=0.
    if status=="compatible"
        roots=TOML.parsefile(joinpath(context_dir,"context.toml"))["equilibria"]
        index=phase["destination"]
        index isa Integer && 1<=index<=length(roots) || error("invalid observation destination root: $folder/$name")
        target=roots[index]["state"]
        destination_error=maximum(abs.(endpoint.-target))
        roles=TOML.parsefile(joinpath(context_dir,"roles.toml"))
        assigned=[role for (role,k) in roles if k==index]
        if !isempty(assigned)
            destination==only(assigned) || error("observation destination role changed: $folder/$name")
        elseif destination!="equilibrium_$index"
            reference_roles=TOML.parsefile(joinpath(reference_dir,"roles.toml"))
            reference=TOML.parsefile(joinpath(reference_dir,"context.toml"))["equilibria"]
            haskey(reference_roles,destination) || error("unknown observation destination: $folder/$name")
            old=reference[reference_roles[destination]]["state"]
            maximum(abs.(target.-old))<.05 || error("observation destination correspondence changed: $folder/$name")
        end
    elseif status=="periodic_compatible"
        destination=="oscillatory" || error("periodic observation destination changed: $folder/$name")
        destination_error=verify_periodic_orbit(folder,name,endpoint,p,baseline)
    else
        destination==status || error("unresolved observation destination changed: $folder/$name")
    end
    replay=compare(file,p,baseline...,details)
    discrepancy=max(initial_error,final_error,destination_error,replay.maximum_coordinate_error)
    (;file=replay.file,maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6)
end
verify_intervention(details,sub,p,baseline,row,source)=verify_observation(details,sub,
    "original_seizure",sub,dirname(sub),p,baseline,row.status,row.destination,source;
    final=(row.final_E,row.final_I))
function verify_switching(details,sub,p,baseline,protocol,expected_status)
    folder=joinpath(sub,protocol)
    record=TOML.parsefile(joinpath(folder,"summary.toml"))
    record["status"]==expected_status || error("switching status changed: $folder")
    legs=get(record,"legs",Any[])
    expected_status=="retained" && length(legs)!=4 && error("retained switching lacks two complete cycles: $folder")
    expected_status=="sources_unavailable" && !isempty(legs) && error("unavailable switching has recorded legs: $folder")
    length(legs)>4 && error("switching has more than two cycles: $folder")
    expected_status!="sources_unavailable" && isempty(legs) && error("switching attempt is missing: $folder")
    isempty(legs) && return NamedTuple[]
    roles=TOML.parsefile(joinpath(sub,"roles.toml"))
    context=TOML.parsefile(joinpath(sub,"context.toml"))
    state=Float64.(context["equilibria"][roles["rest"]]["state"])
    checks=NamedTuple[]
    for (index,leg) in enumerate(legs)
        cycle=cld(index,2);number=isodd(index) ? 1 : 2
        name="cycle_$(cycle)_leg_$(number)"
        destination=number==1 ? "active" : "rest"
        leg["name"]==name && leg["cycle"]==cycle && leg["leg"]==number ||
            error("switching leg identity changed: $folder")
        succeeded=leg["status"]=="compatible" && leg["destination"]==destination
        if index==length(legs) && expected_status!="retained"
            !succeeded && leg["destination"]==expected_status || error("switching failure changed: $folder")
        else
            succeeded || error("switching successful leg changed: $folder")
        end
        axis=leg["axis"];amplitude=leg["amplitude"]
        if protocol=="fixed_switching"
            fixed_axis,fixed_amplitude=number==1 ? ("E_increase",.5) : ("I_increase",3.)
            (axis,amplitude,leg["duration"])==(fixed_axis,fixed_amplitude,100.) ||
                error("fixed switching protocol changed: $folder/$name")
        end
        axis in ("E_increase","I_increase") || error("unsupported switching input axis: $axis")
        be=baseline[1]+(axis=="E_increase" ? amplitude : 0.)
        bi=baseline[2]+(axis=="I_increase" ? amplitude : 0.)
        max(abs(be-leg["B_E"]),abs(bi-leg["B_I"]))<1e-12 || error("switching input changed: $folder")
        isfile(joinpath(folder,name*".csv")) || error("switching baseline continuation unavailable: $folder/$name")
        check=verify_pulse_trace(details,folder,name,sub,p,baseline,be,bi,state,leg["duration"],
            leg["status"],leg["destination"])
        followup=collect(CSV.File(joinpath(folder,name*".csv")))
        final_error=max(abs(last(followup).E-leg["final_E"]),abs(last(followup).I-leg["final_I"]))
        discrepancy=max(final_error,check.maximum_coordinate_error)
        push!(checks,(;file=check.file,maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6))
        state=[last(followup).E,last(followup).I]
    end
    checks
end
function orbit_curve_distance(point,states)
    length(states)>=2 || error("periodic orbit has too few samples")
    minimum(begin
        a=states[index];b=states[index+1]
        de=b[1]-a[1];di=b[2]-a[2]
        fraction=clamp(((point[1]-a[1])*de+(point[2]-a[2])*di)/(de*de+di*di),0.,1.)
        max(abs(point[1]-a[1]-fraction*de),abs(point[2]-a[2]-fraction*di))
    end for index in 1:length(states)-1)
end
function verify_periodic_orbit(folder,name,endpoint,p,baseline)
    orbit=TOML.parsefile(joinpath(folder,name*"_orbit.toml"))
    orbit["validation"]=="NumericallyValidatedPeriodicOrbit" &&
        orbit["stability"]=="PeriodicOrbitAttracting" || error("orbit validation changed: $folder/$name")
    length(orbit["times"])==length(orbit["states"]) || error("orbit samples changed: $folder/$name")
    independent=integrate(orbit["initial_state"],p,baseline...,orbit["times"])
    discrepancy=maximum(max(abs(x[1]-r[1]),abs(x[2]-r[2])) for (x,r) in zip(independent.u,orbit["states"]))
    discrepancy=max(discrepancy,maximum(abs.(last(independent.u).-orbit["initial_state"])))
    maximum(x[1] for x in independent.u)-minimum(x[1] for x in independent.u)>1e-4 ||
        error("orbit has no resolved oscillation: $folder/$name")
    orbit_curve_distance(endpoint,orbit["states"])<1e-4 ||
        error("continuation misses recorded orbit: $folder/$name")
    discrepancy
end
function verify_pulse_trace(details,folder,name,context_dir,p,baseline,be,bi,initial,duration,status,destination)
    pulse_path=joinpath(folder,name*"_pulse.csv")
    followup_path=joinpath(folder,name*".csv")
    pulse=collect(CSV.File(pulse_path));followup=collect(CSV.File(followup_path))
    first(pulse).time==0 && first(followup).time==0 &&
        abs(last(pulse).time-duration)<1e-10 || error("pulse phase times changed: $folder/$name")
    observed=TOML.parsefile(joinpath(folder,name*"_summary.toml"))
    phase=TOML.parsefile(joinpath(folder,name*".toml"))
    observed["status"]==status && observed["destination"]==destination ||
        error("pulse outcome changed: $folder/$name")
    phase["status"]==(status=="periodic_compatible" ? "unresolved" : status) ||
        error("pulse continuation status changed: $folder/$name")
    start_error=max(abs(first(pulse).E-initial[1]),abs(first(pulse).I-initial[2]))
    handoff_error=max(abs(last(pulse).E-first(followup).E),abs(last(pulse).I-first(followup).I))
    final_error=max(abs(last(followup).E-phase["final"][1]),abs(last(followup).I-phase["final"][2]))
    destination_error=0.
    if status=="compatible"
        roots=TOML.parsefile(joinpath(context_dir,"context.toml"))["equilibria"]
        roles=TOML.parsefile(joinpath(context_dir,"roles.toml"))
        index=phase["destination"]
        index isa Integer && 1<=index<=length(roots) || error("invalid pulse destination root: $folder/$name")
        target=roots[index]["state"]
        destination_error=max(abs(last(followup).E-target[1]),abs(last(followup).I-target[2]))
        assigned=[role for (role,k) in roles if k==index]
        expected=isempty(assigned) ? "equilibrium_$index" : only(assigned)
        destination==expected || error("pulse destination label changed: $folder/$name")
    elseif status=="periodic_compatible"
        destination=="oscillatory" || error("periodic pulse destination changed: $folder/$name")
        endpoint=[last(followup).E,last(followup).I]
        destination_error=verify_periodic_orbit(folder,name,endpoint,p,baseline)
    else
        destination==status || error("unresolved pulse destination changed: $folder/$name")
    end
    driven=compare(pulse_path,p,be,bi,details)
    restored=compare(followup_path,p,baseline...,details)
    discrepancy=max(start_error,handoff_error,final_error,destination_error,
        driven.maximum_coordinate_error,restored.maximum_coordinate_error)
    (;file=relpath(joinpath(folder,name),details),maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6)
end
function orbit_phase_state(folder,source,p,baseline)
    capture=match(r"^cycle_(\d+)_phase_([0-9.]+)$",source)
    capture===nothing && error("unsupported cycle source: $source")
    orbit=joinpath(folder,"orbit_$(capture.captures[1])")
    record=TOML.parsefile(orbit*".toml")
    record["validation"]=="NumericallyValidatedPeriodicOrbit" &&
        record["stability"]=="PeriodicOrbitAttracting" || error("unvalidated cycle source: $source")
    fraction=parse(Float64,capture.captures[2])
    0<=fraction<1 || error("invalid cycle phase: $source")
    firstrow=first(CSV.File(orbit*".csv"))
    initial=[firstrow.E,firstrow.I]
    state=fraction==0 ? initial : last(integrate(initial,p,baseline...,[0.,fraction*record["period"]]).u)
    state,fraction
end
function verify_axial_pulse_row(details,file,index,row,source,p,baseline,roots,roles,orbit_states,horizons,durations)
    row.axis in ("E_increase","E_withdrawal","I_increase","I_withdrawal") ||
        error("unsupported plotted pulse axis: $file:$index")
    row.amplitude>=0 && row.duration in durations || error("invalid plotted pulse protocol: $file:$index")
    signed=row.axis in ("E_withdrawal","I_withdrawal") ? -row.amplitude : row.amplitude
    be=baseline[1]+(startswith(row.axis,"E_") ? signed : 0.)
    bi=baseline[2]+(startswith(row.axis,"I_") ? signed : 0.)
    be>=0 && bi>=0 && abs(row.B_E-be)<1e-10 && abs(row.B_I-bi)<1e-10 ||
        error("plotted pulse input changed: $file:$index")
    driven=last(integrate(source,p,be,bi,[0.,row.duration]).u)
    switch_error=max(abs(driven[1]-row.switch_E),abs(driven[2]-row.switch_I))
    state=driven;previous=0.
    status="unresolved";destination="unresolved"
    for horizon in horizons
        state=last(integrate(state,p,baseline...,[0.,horizon-previous]).u)
        distances=[maximum(abs.(state.-root["state"])) for root in roots]
        if minimum(distances)<1e-6
            root=argmin(distances)
            assigned=[role for (role,k) in roles if k==root]
            status="compatible"
            destination=isempty(assigned) ? "equilibrium_$root" : only(assigned)
            break
        elseif orbit_states!==nothing && orbit_curve_distance(state,orbit_states)<1e-4
            status="periodic_compatible";destination="oscillatory"
            break
        end
        previous=horizon
    end
    (row.status,row.destination)==(status,destination) ||
        error("plotted pulse outcome changed: $file:$index $(row.status)/$(row.destination) vs $status/$destination")
    (;file="$(relpath(file,details)):$index",maximum_coordinate_error=switch_error,passed=switch_error<1e-6)
end
function independent_target(p,baseline,source,axis,amplitude,duration,roots,target,horizons)
    be=baseline[1]+(axis=="E_increase" ? amplitude : 0.)
    bi=baseline[2]+(axis=="I_increase" ? amplitude : 0.)
    state=last(integrate(source,p,be,bi,[0.,duration]).u)
    previous=0.
    for horizon in horizons
        state=last(integrate(state,p,baseline...,[0.,horizon-previous]).u)
        distances=[maximum(abs.(state.-root["state"])) for root in roots]
        minimum(distances)<1e-6 && return argmin(distances)==target
        previous=horizon
    end
    false
end
function verify_negative_search(details,sub,p,baseline,config)
    roots=TOML.parsefile(joinpath(sub,"context.toml"))["equilibria"]
    roles=TOML.parsefile(joinpath(sub,"roles.toml"))
    all(haskey(roles,r) for r in ("rest","active")) || error("replacement sources unavailable: $sub")
    durations=Float64.(config["durations"])
    amplitudes=collect(0.:.25:8.)
    found=Bool[];checks=NamedTuple[]
    for (role,target) in (("rest","active"),("active","rest"))
        file=joinpath(sub,role*"_remeasure.csv")
        rows=collect(CSV.File(file))
        processed=sort(unique(r.duration for r in rows))
        isempty(processed) && error("empty replacement search: $file")
        length(processed)<=length(durations) && processed==durations[1:length(processed)] ||
            error("replacement duration prefix changed: $file")
        expected=Set((duration,axis,amplitude) for duration in processed
            for axis in ("E_increase","I_increase") for amplitude in amplitudes)
        actual=Set((r.duration,r.axis,r.amplitude) for r in rows)
        length(rows)==length(expected) && actual==expected && all(r->r.source==role,rows) ||
            error("replacement search grid changed: $file")
        source=roots[roles[role]]["state"]
        replayed=NamedTuple[]
        for r in rows
            success=independent_target(p,baseline,source,r.axis,r.amplitude,r.duration,
                roots,roles[target],config["followup_times"])
            reported=r.status=="compatible" && r.destination==target
            success==reported || error("replacement pulse outcome changed: $file $(r.axis) $(r.amplitude) $(r.duration)")
            push!(replayed,(;axis=r.axis,amplitude=r.amplitude,duration=r.duration,success))
        end
        witness=nothing
        for duration in processed
            for axis in ("E_increase","I_increase")
                ordered=sort(filter(r->r.duration==duration && r.axis==axis,replayed);by=r->r.amplitude)
                any(a.success && b.success for (a,b) in zip(ordered,ordered[2:end])) || continue
                witness=duration
                break
            end
            witness===nothing || break
        end
        expected_length=witness===nothing ? length(durations) : findfirst(==(witness),durations)
        length(processed)==expected_length || error("replacement search stopped at wrong duration: $file")
        push!(found,witness!==nothing)
        push!(checks,(;file=relpath(file,details),maximum_coordinate_error=0.,passed=true))
    end
    all(found) && error("replacement switching was demonstrable within protocol: $sub")
    checks
end
function verify_sequence(details,reference)
    sequence_dir=joinpath(details,"sequence")
    seq=collect(CSV.File(joinpath(sequence_dir,"trajectory.csv")))
    events=collect(CSV.File(joinpath(sequence_dir,"events.csv")))
    route=["rest","active","rest","herald","active","rest","active","seizure"]
    TOML.parsefile(joinpath(sequence_dir,"summary.toml"))["route"]==route || error("sequence route changed")
    length(events)==length(route)-1 || error("sequence event count changed")
    roots=TOML.parsefile(joinpath(sequence_dir,"context.toml"))["equilibria"]
    roles=TOML.parsefile(joinpath(sequence_dir,"roles.toml"))
    all(haskey(roles,role) for role in route) || error("sequence route source unavailable")
    rest=roots[roles["rest"]]["state"]
    maximum(abs.([first(seq).E,first(seq).I].-rest))<1e-6 || error("sequence does not start at rest")
    first(seq).time==0 || error("sequence time origin changed")
    pre_rows=filter(r->r.time<=first(events).onset,seq)
    pre=integrate(rest,reference,.35,0.,getproperty.(pre_rows,:time))
    pre_error=maximum(max(abs(x[1]-r.E),abs(x[2]-r.I)) for (x,r) in zip(pre.u,pre_rows))
    checks=NamedTuple[(;file="sequence/prelude",maximum_coordinate_error=pre_error,passed=pre_error<1e-7)]
    initial=last(pre.u)
    previous_end=first(events).onset
    for (index,event) in enumerate(events)
        event.index==index && event.source==route[index] && event.destination==route[index+1] &&
            event.status=="compatible" || error("sequence event route changed: $index")
        event.axis in ("E_increase","E_withdrawal","I_increase") && event.amplitude>=0 &&
            abs(event.onset-previous_end)<1e-10 && abs(event.offset-event.onset-event.duration)<1e-10 &&
            event.followup_end>event.offset || error("sequence event timing or input changed: $index")
        phase=TOML.parsefile(joinpath(sequence_dir,"event_$index.toml"))
        observed=TOML.parsefile(joinpath(sequence_dir,"event_$(index)_summary.toml"))
        phase["status"]=="compatible" && phase["destination"]==roles[event.destination] &&
            observed["status"]==event.status && observed["destination"]==event.destination ||
            error("sequence event destination changed: $index")
        source_root=roots[roles[event.source]]["state"]
        destination_root=roots[roles[event.destination]]["state"]
        source_error=maximum(abs.(initial.-source_root))
        start_error=maximum(abs.(initial.-[event.start_E,event.start_I]))
        be=.35+(event.axis=="E_increase" ? event.amplitude : event.axis=="E_withdrawal" ? -event.amplitude : 0.)
        bi=event.axis=="I_increase" ? event.amplitude : 0.
        pulse_rows=filter(r->event.onset<=r.time<=event.offset,seq)
        followup_rows=filter(r->event.offset<=r.time<=event.followup_end,seq)
        pulse=integrate(initial,reference,be,bi,[r.time-event.onset for r in pulse_rows])
        followup=integrate(last(pulse.u),reference,.35,0.,[r.time-event.offset for r in followup_rows])
        length(pulse.u)==length(pulse_rows) && length(followup.u)==length(followup_rows) || error("sequence sample count changed")
        waveform_error=maximum(max(abs(x[1]-r.E),abs(x[2]-r.I)) for (sol,rows) in
            ((pulse,pulse_rows),(followup,followup_rows)) for (x,r) in zip(sol.u,rows))
        initial=last(followup.u)
        discrepancy=max(source_error,start_error,waveform_error,
            maximum(abs.(initial.-[event.final_E,event.final_I])),
            maximum(abs.(initial.-phase["final"])),maximum(abs.(initial.-destination_root)))
        push!(checks,(;file="sequence/event_$(event.index)",maximum_coordinate_error=discrepancy,passed=discrepancy<1e-7))
        previous_end=event.followup_end
    end
    abs(last(seq).time-previous_end)<1e-10 || error("sequence endpoint changed")
    checks
end
function verify(details,output)
    ispath(output) && error("verification output exists")
    metadata=TOML.parsefile(joinpath(details,"metadata.toml"))
    config=TOML.parsefile(joinpath(details,"config.toml"))
    get(metadata,"completed",false) || error("detail study is incomplete")
    stage_manifests=IOChecks.completed_stage_manifests(details,metadata["completed_stages"])
    mkpath(output);cases=TOML.parsefile(joinpath(details,"selection.toml"))["cases"]
    byid=Dict(p["id"]=>p for p in cases);checks=NamedTuple[]
    reference=byid["selective_withdrawal"]
    append!(checks,verify_sequence(details,reference))
    for p in cases
        root=joinpath(details,"cases",p["id"])
        for baseline in filter(n->startswith(n,"baseline_"),readdir(root))
            folder=joinpath(root,baseline);input=TOML.parsefile(joinpath(folder,"input.toml"))
            for name in filter(n->startswith(n,"orbit_") && endswith(n,".csv"),readdir(folder))
                push!(checks,compare(joinpath(folder,name),p,input["B_E"],input["B_I"],details))
            end
            for source in filter(n->startswith(n,"cycle_"),readdir(folder))
                initial,fraction=orbit_phase_state(folder,source,p,(input["B_E"],input["B_I"]))
                pulses=joinpath(folder,source,"pulses")
                pulse_file=joinpath(pulses,"pulses.csv")
                trials=collect(CSV.File(pulse_file))
                all(r->r.source==source && r.phase==fraction,trials) || error("cycle pulse source changed: $source")
                if p["id"]=="input_dependent" && (input["B_E"],input["B_I"])==(2.,1.) &&
                    startswith(source,"cycle_1_phase_")
                    roots=TOML.parsefile(joinpath(folder,"context.toml"))["equilibria"]
                    roles=TOML.parsefile(joinpath(folder,"roles.toml"))
                    orbit_states=[[r.E,r.I] for r in CSV.File(joinpath(folder,"orbit_1.csv"))]
                    for (index,row) in enumerate(trials)
                        row.axis=="joint" && continue
                        row.source_role=="oscillatory" || error("plotted pulse source role changed: $pulse_file:$index")
                        push!(checks,verify_axial_pulse_row(details,pulse_file,index,row,initial,p,
                            (input["B_E"],input["B_I"]),roots,roles,orbit_states,
                            config["followup_times"],config["durations"]))
                    end
                end
                for destination in sort(unique(r.destination for r in trials))
                    destination in ("integration_failed","unresolved") && continue
                    witness=first(sort(filter(r->r.destination==destination,trials);by=r->(r.duration,r.B_E,r.B_I)))
                    confirmation=joinpath(pulses,"confirmation")
                    push!(checks,verify_pulse_trace(details,confirmation,destination,folder,p,
                        (input["B_E"],input["B_I"]),witness.B_E,witness.B_I,initial,
                        witness.duration,witness.status,witness.destination))
                end
            end
            if p["id"]=="selective_withdrawal" && (input["B_E"],input["B_I"])==(.35,0.)
                roots=TOML.parsefile(joinpath(folder,"context.toml"))["equilibria"]
                roles=TOML.parsefile(joinpath(folder,"roles.toml"))
                for role in ("rest","active")
                    source="root_$(roles[role])"
                    pulse_file=joinpath(folder,source,"pulses","pulses.csv")
                    for (index,row) in enumerate(CSV.File(pulse_file))
                        row.axis=="joint" && continue
                        row.source==source && row.source_role==role && row.phase==0 ||
                            error("plotted pulse source changed: $pulse_file:$index")
                        push!(checks,verify_axial_pulse_row(details,pulse_file,index,row,
                            roots[roles[role]]["state"],p,(input["B_E"],input["B_I"]),
                            roots,roles,nothing,config["followup_times"],config["durations"]))
                    end
                end
            end
        end
    end
    for row in CSV.File(joinpath(details,"paired_controls","sustained.csv"))
        p=byid[row.id]
        sub=joinpath(details,"paired_controls","$(row.id)_$(row.baseline)")
        input=TOML.parsefile(joinpath(sub,"input.toml"))
        (row.B_E,row.B_I,row.tau_ratio)==(input["B_E"],input["B_I"],p["tau_ratio"]) ||
            error("paired sustained baseline changed: $sub")
        roots=TOML.parsefile(joinpath(sub,"context.toml"))["equilibria"]
        roles=TOML.parsefile(joinpath(sub,"roles.toml"))
        for role in ("herald","seizure"), protocol in ("held","release")
            source=roots[roles[role]]["state"]
            status=protocol=="held" ? "compatible" : getproperty(row,Symbol(role*"_status"))
            destination=protocol=="held" ? role : getproperty(row,Symbol(role))
            context_dir=protocol=="held" ? sub : joinpath(sub,"released")
            push!(checks,verify_observation(details,sub,"$(role)_$(protocol)",context_dir,sub,p,
                (protocol=="held" ? row.B_E : 0.,row.B_I),status,destination,source))
        end
    end
    for (index,row) in enumerate(CSV.File(joinpath(details,"paired_controls","pulses.csv")))
        p=byid[row.id]
        sub=joinpath(details,"paired_controls","$(row.id)_$(row.baseline)")
        input=TOML.parsefile(joinpath(sub,"input.toml"))
        be=input["B_E"];bi=input["B_I"]
        0<=row.amplitude<=be || error("paired pulse amplitude changed: $sub")
        roots=TOML.parsefile(joinpath(sub,"context.toml"))["equilibria"]
        roles=TOML.parsefile(joinpath(sub,"roles.toml"))
        traces=joinpath(sub,"pulse_trials","row_$index")
        for role in ("herald","seizure")
            initial=roots[roles[role]]["state"]
            push!(checks,verify_pulse_trace(details,traces,role,sub,p,(be,bi),be-row.amplitude,bi,
                initial,row.duration,getproperty(row,Symbol(role*"_status")),getproperty(row,Symbol(role))))
        end
    end
    for id in ("selective_withdrawal","threshold_to_active")
        p=byid[id];b=first(p["baselines"])
        base=joinpath(details,"interventions",id)
        reference=TOML.parsefile(joinpath(base,"context.toml"))["equilibria"]
        roles=TOML.parsefile(joinpath(base,"roles.toml"))
        source=reference[roles["seizure"]]["state"]
        for row in CSV.File(joinpath(details,"interventions",id,"summary.csv"))
            q=merge(p,Dict(row.axis=>row.value))
            sub=joinpath(details,"interventions",id,"$(row.axis)_$(row.value)")
            push!(checks,verify_intervention(details,sub,q,b,row,source))
            append!(checks,verify_switching(details,sub,q,b,"fixed_switching",row.fixed_switching))
            if isfile(joinpath(sub,"remeasured_switching","summary.toml"))
                append!(checks,verify_switching(details,sub,q,b,"remeasured_switching",row.remeasured_switching))
            elseif row.fixed_switching=="retained"
                row.remeasured_switching=="retained" || error("retained fixed switching must carry through")
            elseif row.fixed_switching=="sources_unavailable"
                row.remeasured_switching=="sources_unavailable" || error("missing switching sources changed")
            else
                row.remeasured_switching=="not_demonstrated_within_protocol" ||
                    error("missing remeasured switching evidence")
                append!(checks,verify_negative_search(details,sub,q,b,config))
            end
        end
    end
    report=joinpath(output,"trajectories.csv")
    CSV.write(report,checks)
    record=Dict("passed"=>all(r->r.passed,checks),"trajectories"=>length(checks),
        "maximum_coordinate_error"=>maximum(r.maximum_coordinate_error for r in checks),
        "independence"=>"Independently written population responses and ODE; same Tsit5 integrator, 1e-12 tolerances",
        "verifier_sha256"=>digest(@__FILE__),
        "packet_io_sha256"=>digest(joinpath(@__DIR__,"figure_packet_io.jl")),
        "completed_stage_manifests_sha256"=>stage_manifests,
        "trajectories_sha256"=>digest(report),
        "details_metadata_sha256"=>digest(joinpath(details,"metadata.toml")),
        "selection_sha256"=>digest(joinpath(details,"selection.toml")))
    open(joinpath(output,"summary.toml"),"w") do io;TOML.print(io,record;sorted=true);end
    record["passed"] || error("independent equation comparison failed")
    println(record)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==2 || error("usage: verify_figure_redesign.jl DETAILS NEW_VERIFICATION_OUTPUT")
    VerifyFigureRedesign.verify(ARGS...)
end
