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
function verify_intervention(details,sub,p,baseline,row,source)
    file=joinpath(sub,"original_seizure.csv")
    trace=collect(CSV.File(file))
    observed=TOML.parsefile(joinpath(sub,"original_seizure_summary.toml"))
    phase=TOML.parsefile(joinpath(sub,"original_seizure.toml"))
    observed["status"]==row.status && observed["destination"]==row.destination ||
        error("reported intervention outcome changed: $sub")
    phase["status"]==(row.status=="periodic_compatible" ? "unresolved" : row.status) ||
        error("intervention phase status changed: $sub")
    initial_error=max(abs(first(trace).E-source[1]),abs(first(trace).I-source[2]))
    final_error=maximum(abs.([last(trace).E-row.final_E,last(trace).I-row.final_I,
        last(trace).E-phase["final"][1],last(trace).I-phase["final"][2]]))
    destination_error=0.
    if row.status=="compatible"
        roots=TOML.parsefile(joinpath(sub,"context.toml"))["equilibria"]
        index=phase["destination"]
        index isa Integer && 1<=index<=length(roots) || error("invalid intervention destination root: $sub")
        target=roots[index]["state"]
        destination_error=max(abs(last(trace).E-target[1]),abs(last(trace).I-target[2]))
        roles=TOML.parsefile(joinpath(sub,"roles.toml"))
        assigned=[name for (name,k) in roles if k==index]
        if !isempty(assigned)
            row.destination==only(assigned) || error("intervention destination role changed: $sub")
        elseif row.destination!="equilibrium_$index"
            reference_roles=TOML.parsefile(joinpath(dirname(sub),"roles.toml"))
            reference=TOML.parsefile(joinpath(dirname(sub),"context.toml"))["equilibria"]
            haskey(reference_roles,row.destination) || error("unknown intervention destination: $sub")
            old=reference[reference_roles[row.destination]]["state"]
            maximum(abs.(target.-old))<.05 || error("intervention destination correspondence changed: $sub")
        end
    end
    replay=compare(file,p,baseline...,details)
    discrepancy=max(initial_error,final_error,destination_error,replay.maximum_coordinate_error)
    (;file=replay.file,maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6)
end
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
        axis in ("E_increase","I_increase") || error("unsupported switching input axis: $axis")
        be=baseline[1]+(axis=="E_increase" ? amplitude : 0.)
        bi=baseline[2]+(axis=="I_increase" ? amplitude : 0.)
        max(abs(be-leg["B_E"]),abs(bi-leg["B_I"]))<1e-12 || error("switching input changed: $folder")
        pulse=collect(CSV.File(joinpath(folder,name*"_pulse.csv")))
        isfile(joinpath(folder,name*".csv")) || error("switching baseline continuation unavailable: $folder/$name")
        followup=collect(CSV.File(joinpath(folder,name*".csv")))
        first(pulse).time==0 && first(followup).time==0 &&
            abs(last(pulse).time-leg["duration"])<1e-10 || error("switching phase times changed: $folder")
        start_error=maximum(abs.(state.-[first(pulse).E,first(pulse).I]))
        handoff_error=max(abs(last(pulse).E-first(followup).E),abs(last(pulse).I-first(followup).I))
        final_error=max(abs(last(followup).E-leg["final_E"]),abs(last(followup).I-leg["final_I"]))
        driven=compare(joinpath(folder,name*"_pulse.csv"),p,be,bi,details)
        restored=compare(joinpath(folder,name*".csv"),p,baseline...,details)
        discrepancy=max(start_error,handoff_error,final_error,
            driven.maximum_coordinate_error,restored.maximum_coordinate_error)
        push!(checks,(;file=relpath(joinpath(folder,name),details),
            maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6))
        state=[last(followup).E,last(followup).I]
    end
    checks
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
        orbit=TOML.parsefile(joinpath(folder,name*"_orbit.toml"))
        orbit["validation"]=="NumericallyValidatedPeriodicOrbit" &&
            orbit["stability"]=="PeriodicOrbitAttracting" || error("pulse orbit validation changed: $folder/$name")
        independent=integrate(orbit["initial_state"],p,baseline...,orbit["times"])
        destination_error=maximum(max(abs(x[1]-r[1]),abs(x[2]-r[2])) for (x,r) in zip(independent.u,orbit["states"]))
        destination_error=max(destination_error,maximum(abs.(last(independent.u).-orbit["initial_state"])))
        maximum(x[1] for x in independent.u)-minimum(x[1] for x in independent.u)>1e-4 ||
            error("pulse orbit has no resolved oscillation: $folder/$name")
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
    record["validation"]=="NumericallyValidatedPeriodicOrbit" || error("unvalidated cycle source: $source")
    fraction=parse(Float64,capture.captures[2])
    0<=fraction<1 || error("invalid cycle phase: $source")
    firstrow=first(CSV.File(orbit*".csv"))
    initial=[firstrow.E,firstrow.I]
    state=fraction==0 ? initial : last(integrate(initial,p,baseline...,[0.,fraction*record["period"]]).u)
    state,fraction
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
function verify(details,output)
    ispath(output) && error("verification output exists")
    metadata=TOML.parsefile(joinpath(details,"metadata.toml"))
    config=TOML.parsefile(joinpath(details,"config.toml"))
    get(metadata,"completed",false) || error("detail study is incomplete")
    stage_manifests=IOChecks.completed_stage_manifests(details,metadata["completed_stages"])
    mkpath(output);cases=TOML.parsefile(joinpath(details,"selection.toml"))["cases"]
    byid=Dict(p["id"]=>p for p in cases);checks=NamedTuple[]
    reference=byid["selective_withdrawal"]
    seq=collect(CSV.File(joinpath(details,"sequence","trajectory.csv")))
    events=collect(CSV.File(joinpath(details,"sequence","events.csv")))
    initial=[first(seq).E,first(seq).I]
    initial=last(integrate(initial,reference,.35,0.,[0.,50.]).u)
    for event in events
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
        discrepancy=max(start_error,waveform_error,maximum(abs.(initial.-[event.final_E,event.final_I])))
        push!(checks,(;file="sequence/event_$(event.index)",maximum_coordinate_error=discrepancy,passed=discrepancy<1e-7))
    end
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
                trials=collect(CSV.File(joinpath(pulses,"pulses.csv")))
                all(r->r.source==source && r.phase==fraction,trials) || error("cycle pulse source changed: $source")
                for destination in sort(unique(r.destination for r in trials))
                    destination in ("integration_failed","unresolved") && continue
                    witness=first(sort(filter(r->r.destination==destination,trials);by=r->(r.duration,r.B_E,r.B_I)))
                    confirmation=joinpath(pulses,"confirmation")
                    push!(checks,verify_pulse_trace(details,confirmation,destination,folder,p,
                        (input["B_E"],input["B_I"]),witness.B_E,witness.B_I,initial,
                        witness.duration,witness.status,witness.destination))
                end
            end
        end
    end
    for row in CSV.File(joinpath(details,"paired_controls","sustained.csv"))
        p=byid[row.id]
        for role in ("herald","seizure"), protocol in ("held","release")
            file=joinpath(details,"paired_controls","$(row.id)_$(row.baseline)","$(role)_$(protocol).csv")
            push!(checks,compare(file,p,protocol=="held" ? row.B_E : 0.,row.B_I,details))
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
