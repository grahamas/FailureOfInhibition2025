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
inhibitory_rate(input,p,control)=control ?
    sigmoid(get(p,"slope_i",5.)*(input-get(p,"theta_on",4.))) :
    sigmoid(get(p,"slope_i",5.)*(input-get(p,"theta_on",4.)))*
    sigmoid(get(p,"slope_i",5.)*(p["theta_off"]-input))*
    (-expm1(-get(p,"slope_i",5.)*(p["theta_off"]-get(p,"theta_on",4.))))
function independent_jacobian(state,p,be,bi;control=false)
    e,i=state
    ue=p["e_to_e"]*e-p["i_to_e"]*i+be
    ui=p["e_to_i"]*e-p["i_to_i"]*i+bi
    se=get(p,"slope_e",5.);si=get(p,"slope_i",5.)
    fe=sigmoid(se*(ue-get(p,"theta_e",1.5)))
    a=sigmoid(si*(ui-get(p,"theta_on",4.)))
    b=sigmoid(si*(p["theta_off"]-ui))
    fi=control ? a : a*b*(-expm1(-si*(p["theta_off"]-get(p,"theta_on",4.))))
    dfe=se*fe*(1-fe);dfi=control ? si*a*(1-a) : si*fi*(b-a)
    tau=get(p,"tau_e",7.8);tau_i=tau*p["tau_ratio"]
    j11=(-1-fe+(1-e)*dfe*p["e_to_e"])/tau
    j12=-(1-e)*dfe*p["i_to_e"]/tau
    j21=(1-i)*dfi*p["e_to_i"]/tau_i
    j22=(-1-fi-(1-i)*dfi*p["i_to_i"])/tau_i
    (;j11,j12,j21,j22,determinant=j11*j22-j12*j21)
end
function orbit_measurements(initial,p,baseline,times)
    solution=integrate(initial,p,baseline...,times)
    states=solution.u
    trace_integral=sum(begin
        a=independent_jacobian(states[k],p,baseline...)
        b=independent_jacobian(states[k+1],p,baseline...)
        (a.j11+a.j22+b.j11+b.j22)*(times[k+1]-times[k])/2
    end for k in 1:length(times)-1)
    (;states,closure=maximum(abs.(last(states).-first(states))),
        E_min=minimum(x[1] for x in states),E_max=maximum(x[1] for x in states),
        I_min=minimum(x[2] for x in states),I_max=maximum(x[2] for x in states),
        multiplier=exp(trace_integral))
end
function verify_orbit_values(measured,record,where;attracting_only=false)
    record["validation"]=="NumericallyValidatedPeriodicOrbit" ||
        error("unvalidated plotted orbit: $where")
    measured.closure<1e-6 && measured.E_max-measured.E_min>1e-4 &&
        isfinite(measured.multiplier) || error("periodic closure failed: $where")
    stability=record["stability"]
    if stability=="PeriodicOrbitAttracting"
        measured.multiplier<1-1e-4 || error("periodic attraction failed: $where")
    elseif stability=="PeriodicOrbitRepelling" && !attracting_only
        measured.multiplier>1+1e-4 || error("periodic repulsion failed: $where")
    else
        error("unsupported plotted orbit stability: $where")
    end
    transverse=haskey(record,"transverse_multiplier") ?
        record["transverse_multiplier"] :
        Dict("real"=>record["transverse_real"],"imaginary"=>record["transverse_imag"])
    abs(measured.multiplier-transverse["real"])<1e-3 &&
        abs(transverse["imaginary"])<1e-3 || error("periodic multiplier changed: $where")
end
function verify_source_orbit(details,file,p,baseline)
    rows=collect(CSV.File(file));record=TOML.parsefile(replace(file,r"\.csv$"=>".toml"))
    times=Float64.(getproperty.(rows,:time))
    length(times)>=3 && first(times)==0 && all(diff(times).>0) &&
        abs(last(times)-record["period"])<1e-8 || error("periodic source times changed: $file")
    measured=orbit_measurements([first(rows).E,first(rows).I],p,baseline,times)
    waveform_error=maximum(max(abs(x[1]-r.E),abs(x[2]-r.I)) for (x,r) in zip(measured.states,rows))
    verify_orbit_values(measured,record,file;attracting_only=true)
    amplitude_error=max(abs(measured.E_max-measured.E_min-record["amplitudes"][1]),
        abs(measured.I_max-measured.I_min-record["amplitudes"][2]))
    discrepancy=max(waveform_error,measured.closure,amplitude_error)
    (;file=relpath(file,details),maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6)
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
function verify_switching(details,sub,p,baseline,protocol,expected_status;witnesses=nothing)
    folder=joinpath(sub,protocol)
    record=TOML.parsefile(joinpath(folder,"summary.toml"))
    record["status"]==expected_status || error("switching status changed: $folder")
    legs=get(record,"legs",Any[])
    expected_status=="retained" && length(legs)!=4 && error("retained switching lacks two complete cycles: $folder")
    expected_status=="sources_unavailable" && !isempty(legs) && error("unavailable switching has recorded legs: $folder")
    length(legs)>4 && error("switching has more than two cycles: $folder")
    expected_status!="sources_unavailable" && isempty(legs) && error("switching attempt is missing: $folder")
    roles=TOML.parsefile(joinpath(sub,"roles.toml"))
    sources_present=all(haskey(roles,role) for role in ("rest","active"))
    (expected_status=="sources_unavailable")==!sources_present ||
        error("switching source availability changed: $folder")
    isempty(legs) && return NamedTuple[]
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
        elseif protocol=="remeasured_switching"
            witnesses===nothing && error("remeasured switching search was not checked: $folder")
            witness=witnesses[number]
            (axis,amplitude,leg["duration"])==(witness.axis,witness.amplitude,witness.duration) ||
                error("remeasured switching witness changed: $folder/$name")
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
    times=Float64.(orbit["times"])
    length(times)==length(orbit["states"]) && length(times)>=3 &&
        first(times)==0 && all(diff(times).>0) &&
        abs(last(times)-orbit["period"])<1e-8 || error("orbit samples changed: $folder/$name")
    measured=orbit_measurements(orbit["initial_state"],p,baseline,times)
    verify_orbit_values(measured,orbit,"$folder/$name";attracting_only=true)
    discrepancy=maximum(max(abs(x[1]-r[1]),abs(x[2]-r[2])) for (x,r) in
        zip(measured.states,orbit["states"]))
    discrepancy=max(discrepancy,measured.closure,
        abs(measured.E_max-measured.E_min-orbit["amplitudes"][1]),
        abs(measured.I_max-measured.I_min-orbit["amplitudes"][2]))
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
function verify_remeasured_witnesses(details,sub,config)
    durations=Float64.(config["durations"]);amplitudes=collect(0.:.25:8.)
    witnesses=NamedTuple[];checks=NamedTuple[]
    for (role,target) in (("rest","active"),("active","rest"))
        file=joinpath(sub,role*"_remeasure.csv")
        rows=collect(CSV.File(file))
        processed=sort(unique(r.duration for r in rows))
        !isempty(processed) && length(processed)<=length(durations) &&
            processed==durations[1:length(processed)] ||
            error("replacement duration prefix changed: $file")
        expected=Set((duration,axis,amplitude) for duration in processed
            for axis in ("E_increase","I_increase") for amplitude in amplitudes)
        actual=Set((r.duration,r.axis,r.amplitude) for r in rows)
        length(rows)==length(expected) && actual==expected && all(r->r.source==role,rows) ||
            error("replacement search grid changed: $file")
        witness=nothing
        for duration in processed
            candidates=NamedTuple[]
            for axis in ("E_increase","I_increase")
                ordered=sort(filter(r->r.duration==duration && r.axis==axis,rows);by=r->r.amplitude)
                start=nothing
                for (index,row) in enumerate(ordered)
                    success=row.status=="compatible" && row.destination==target
                    if success
                        start===nothing && (start=index)
                        if index==length(ordered) ||
                            !(ordered[index+1].status=="compatible" && ordered[index+1].destination==target)
                            lo=ordered[start].amplitude;hi=row.amplitude
                            hi>lo && push!(candidates,(;axis,amplitude=(lo+hi)/2,duration,width=hi-lo))
                            start=nothing
                        end
                    else
                        start=nothing
                    end
                end
            end
            if !isempty(candidates)
                witness=first(sort(candidates;by=x->(-x.width,x.axis,x.amplitude)))
                break
            end
        end
        witness===nothing && error("remeasured switching has no search witness: $file")
        processed==durations[1:findfirst(==(witness.duration),durations)] ||
            error("replacement search stopped at wrong duration: $file")
        push!(witnesses,witness)
        push!(checks,(;file=relpath(file,details),maximum_coordinate_error=0.,passed=true))
    end
    witnesses,checks
end
function verify_timescale_spectrum(details,folder,p,baseline)
    roots=TOML.parsefile(joinpath(folder,"context.toml"))["equilibria"]
    path=joinpath(folder,"equilibria.csv");rows=collect(CSV.File(path))
    ratios=unique(r.ratio for r in rows)
    all(r->.2<=r<=4.4,ratios) && all(any(abs(r-grid)<1e-10 for r in ratios)
        for grid in range(.2,4.4;length=85)) || error("time-scale scan grid changed: $path")
    length(rows)==length(roots)*length(ratios) || error("time-scale spectrum rows changed: $path")
    seen=Set{Tuple{Float64,Int}}();checks=NamedTuple[]
    for (index,row) in enumerate(rows)
        1<=row.root<=length(roots) || error("invalid time-scale root: $path:$index")
        push!(seen,(row.ratio,row.root))
        q=merge(p,Dict("tau_ratio"=>row.ratio))
        state=roots[row.root]["state"]
        field=zeros(2);independent_rhs!(field,state,(q,baseline...),0.)
        j=independent_jacobian(state,q,baseline...)
        trace=j.j11+j.j22;discriminant=trace^2-4*j.determinant
        alpha=discriminant>=0 ? (trace+sqrt(discriminant))/2 : trace/2
        stability=j.determinant<0 ? "Saddle" : abs(alpha)<1e-8 ? "Unresolved" :
            alpha<0 ? "Attracting" : "Repelling"
        row.stability==stability || error("time-scale stability changed: $path:$index")
        discrepancy=maximum((abs(row.E-state[1]),abs(row.I-state[2]),
            abs(row.alpha-alpha),abs(row.determinant-j.determinant),maximum(abs.(field))))
        push!(checks,(;file="$(relpath(path,details)):$index",
            maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6))
    end
    length(seen)==length(rows) || error("duplicate time-scale spectrum row: $path")
    checks
end
function verify_cycle_row(details,path,index,row,record,initial,p,baseline)
    period=record["period"]
    period>0 && abs(row.period-period)<1e-6 &&
        row.validation==record["validation"] && row.stability==record["stability"] ||
        error("continued orbit record changed: $path:$index")
    times=collect(range(0.,period;length=257))
    measured=orbit_measurements(initial,p,baseline,times)
    verify_orbit_values(measured,record,"$path:$index")
    discrepancy=maximum((measured.closure,abs(measured.E_min-row.E_min),
        abs(measured.E_max-row.E_max),abs(measured.I_min-row.I_min),
        abs(measured.I_max-row.I_max),abs(measured.multiplier-row.multiplier),
        abs(measured.closure-row.closure)))
    (;file="$(relpath(path,details)):$index",maximum_coordinate_error=discrepancy,
        passed=discrepancy<1e-3)
end
function verify_timescale_cycles(details,folder,p,baseline)
    path=joinpath(folder,"cycles.csv")
    isfile(path) || error("plotted periodic continuation is missing: $path")
    rows=collect(CSV.File(path));stops=collect(CSV.File(joinpath(folder,"stops.csv")))
    Set(r.direction for r in rows)==Set((-1,1)) &&
        Set(r.direction for r in stops)==Set((-1,1)) ||
        error("continued branch directions changed: $path")
    seed_file=joinpath(folder,"seed_$(p["tau_ratio"])","orbit.csv")
    seed_record=TOML.parsefile(replace(seed_file,r"\.csv$"=>".toml"))
    seed_first=first(CSV.File(seed_file))
    checks=NamedTuple[verify_source_orbit(details,seed_file,p,baseline)]
    for direction in (-1,1)
        branch=filter(r->r.direction==direction,rows)
        stop=only(filter(r->r.direction==direction,stops))
        attempts=TOML.parsefile(joinpath(folder,"attempts_$direction.toml"))["attempts"]
        accepted=filter(a->a["accepted"],attempts)
        length(branch)==length(accepted)+1 && stop.points==length(branch) &&
            all(r.index==index for (index,r) in enumerate(branch)) ||
            error("continued branch point count changed: $path:$direction")
        first(branch).ratio==p["tau_ratio"] || error("continued branch seed changed: $path")
        seed_index=only(findall(r->r.direction==direction && r.index==1,rows))
        push!(checks,verify_cycle_row(details,path,seed_index,first(branch),
            seed_record,[seed_first.E,seed_first.I],p,baseline))
        for (index,attempt) in enumerate(accepted)
            row=branch[index+1];candidate=attempt["candidate"]
            abs(candidate[4]-row.ratio)<1e-10 && abs(exp(candidate[3])-row.period)<1e-6 &&
                attempt["status"]=="corrector_converged" ||
                error("accepted continuation candidate changed: $path:$direction:$index")
            q=merge(p,Dict("tau_ratio"=>row.ratio))
            row_index=only(findall(r->r.direction==direction && r.index==row.index,rows))
            push!(checks,verify_cycle_row(details,path,row_index,row,
                attempt["orbit"],candidate[1:2],q,baseline))
        end
        if stop.termination=="step_limit"
            length(accepted)==250 || error("continuation step limit changed: $path:$direction")
        elseif stop.termination=="parameter_boundary"
            bound=direction<0 ? .2 : 4.4
            abs(last(branch).ratio-bound)<1e-4 ||
                error("continuation boundary changed: $path:$direction")
        else
            error("unverified continuation stop: $path:$direction:$(stop.termination)")
        end
    end
    checks
end
function verify_hysteresis(details,folder,p,baseline,config)
    roles=TOML.parsefile(joinpath(folder,"roles.toml"))
    roots=TOML.parsefile(joinpath(folder,"context.toml"))["equilibria"]
    file=joinpath(folder,"root_$(roles["rest"])","hysteresis","sweeps.csv")
    rows=filter(r->r.axis=="B_E" && r.first_direction==-1,collect(CSV.File(file)))
    cap=TOML.parsefile(joinpath(dirname(folder),"geometry","bounds.toml"))["B_E"]
    state=roots[roles["rest"]]["state"];current=baseline[1];checks=NamedTuple[]
    index=1
    for target in (0.,cap,baseline[1])
        while abs(current-target)>1e-10
            index<=length(rows) || error("plotted hysteresis sweep ended early: $file")
            row=rows[index]
            abs(row.previous_input-current)<1e-10 &&
                0<=row.input<=cap && (row.input-current)*(target-current)>0 &&
                abs(row.input-current)<=abs(target-current)+1e-10 &&
                row.status=="compatible" || error("plotted hysteresis step changed: $file:$index")
            discrepancy=Inf
            for horizon in config["followup_times"]
                ending=last(integrate(state,p,row.input,baseline[2],[0.,horizon]).u)
                discrepancy=min(discrepancy,maximum(abs.(ending.-[row.E,row.I])))
                discrepancy<1e-6 && break
            end
            push!(checks,(;file="$(relpath(file,details)):$index",
                maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6))
            current=row.input;state=[row.E,row.I];index+=1
        end
    end
    index==length(rows)+1 || error("plotted hysteresis has extra steps: $file")
    checks
end
function verify_cell_partition(cells,bounds)
    edges=sort(unique(vcat(0.,bounds["B_E"],
        [cell.x0 for cell in cells],[cell.x1 for cell in cells])))
    for (left,right) in zip(edges,edges[2:end])
        intervals=sort([(cell.y0,cell.y1) for cell in cells
            if cell.x0<=left && cell.x1>=right];by=first)
        upper=0.
        for (lo,hi) in intervals
            lo==upper || error("geometry cells leave a gap or overlap")
            upper=hi
        end
        upper==bounds["B_I"] || error("geometry cells do not cover input domain")
    end
    true
end
function verify_geometry(details,p)
    folder=joinpath(details,"cases",p["id"],"geometry")
    inputs=collect(CSV.File(joinpath(folder,"inputs.csv")))
    roots=collect(CSV.File(joinpath(folder,"equilibria.csv")))
    cells=collect(CSV.File(joinpath(folder,"cells.csv")))
    bounds=TOML.parsefile(joinpath(folder,"bounds.toml"))
    byinput=Dict{Tuple{Float64,Float64},Vector{Any}}()
    for root in roots
        push!(get!(byinput,(root.B_E,root.B_I),Any[]),root)
    end
    length(byinput)==length(inputs) || error("geometry root/input set changed: $(p["id"])")
    signatures=Dict{Tuple{Float64,Float64},String}();checks=NamedTuple[]
    for (index,row) in enumerate(inputs)
        key=(row.B_E,row.B_I)
        0<=row.B_E<=bounds["B_E"] && 0<=row.B_I<=bounds["B_I"] &&
            !haskey(signatures,key) && haskey(byinput,key) ||
            error("geometry input changed: $(p["id"]):$index")
        group=byinput[key]
        length(group)==row.roots && all(r.root==k for (k,r) in enumerate(group)) ||
            error("geometry root count changed: $(p["id"]):$index")
        labels=String[];maximum_error=0.
        for root in group
            state=[root.E,root.I]
            field=zeros(2);independent_rhs!(field,state,(p,row.B_E,row.B_I),0.)
            j=independent_jacobian(state,p,row.B_E,row.B_I)
            trace=j.j11+j.j22;discriminant=trace^2-4*j.determinant
            alpha=discriminant>=0 ? (trace+sqrt(discriminant))/2 : trace/2
            stability=j.determinant<0 ? "Saddle" : abs(alpha)<1e-8 ? "Unresolved" :
                alpha<0 ? "Attracting" : "Repelling"
            root.stability==stability || error("geometry stability changed: $(p["id"]):$index")
            maximum_error=max(maximum_error,maximum(abs.(field)))
            push!(labels,root.stability*":"*root.role)
        end
        row.sinks==count(r->r.stability=="Attracting",group) &&
            coalesce(row.roles,"")==join(sort([r.role for r in group if r.role!="unassigned"]),",") ||
            error("geometry role counts changed: $(p["id"]):$index")
        suffix=row.discovery_status=="index_unresolved" ? "|index_unresolved" :
            row.discovery_status=="index_consistent_not_complete" ? "" :
            error("geometry discovery status changed: $(p["id"]):$index")
        signature=join(sort(labels),"|")*suffix
        row.signature==signature || error("geometry signature changed: $(p["id"]):$index")
        signatures[key]=signature
        push!(checks,(;file="$(relpath(folder,details))/inputs.csv:$index",
            maximum_coordinate_error=maximum_error,passed=maximum_error<1e-6))
    end
    for (index,cell) in enumerate(cells)
        0<=cell.x0<cell.x1<=bounds["B_E"] &&
            0<=cell.y0<cell.y1<=bounds["B_I"] ||
            error("geometry cell outside bounds: $(p["id"]):$index")
        if cell.status=="sampled_homogeneous"
            points=((cell.x0,cell.y0),(cell.x0,cell.y1),(cell.x1,cell.y0),
                (cell.x1,cell.y1),((cell.x0+cell.x1)/2,(cell.y0+cell.y1)/2))
            all(haskey(signatures,point) for point in points) &&
                length(Set(signatures[point] for point in points))==1 ||
                error("geometry fill lacks agreeing samples: $(p["id"]):$index")
        elseif cell.status ∉ ("budget_unresolved","boundary_bracket","mixed")
            error("unsupported geometry cell status: $(p["id"]):$index")
        end
    end
    verify_cell_partition(cells,bounds)
    push!(checks,(;file="$(relpath(folder,details))/cells.csv",
        maximum_coordinate_error=0.,passed=true))
    checks
end
function verify_mechanism(details,p)
    folder=joinpath(details,"mechanism");checks=NamedTuple[]
    responses=collect(CSV.File(joinpath(folder,"responses.csv")))
    nullclines=collect(CSV.File(joinpath(folder,"nullclines.csv")))
    roots=collect(CSV.File(joinpath(folder,"roots.csv")))
    reference=joinpath(details,"cases","selective_withdrawal","baseline_1")
    reference_roots=TOML.parsefile(joinpath(reference,"context.toml"))["equilibria"]
    reference_roles=TOML.parsefile(joinpath(reference,"roles.toml"))
    for (condition,control) in (("monotone",true),("failure",false))
        response_rows=filter(r->r.condition==condition,responses)
        length(response_rows)==701 || error("mechanism response grid changed: $condition")
        response_error=maximum(begin
            abs(row.input-input)<1e-10 || error("mechanism response input changed: $condition")
            abs(row.response-inhibitory_rate(input,p,control))
        end for (row,input) in zip(response_rows,range(-1.,13.;length=701)))
        push!(checks,(;file="mechanism/responses.csv:$condition",
            maximum_coordinate_error=response_error,passed=response_error<1e-10))
        for population in ("E","I")
            group=filter(r->r.condition==condition && r.population==population,nullclines)
            inputs=population=="E" ? range(-p["i_to_e"]+.35,p["e_to_e"]+.35;length=6001) :
                range(-p["i_to_i"],p["e_to_i"];length=6001)
            expected=NamedTuple[]
            for input in inputs
                if population=="E"
                    rate=sigmoid(get(p,"slope_e",5.)*(input-get(p,"theta_e",1.5)))
                    e=rate/(1+rate);i=(p["e_to_e"]*e+.35-input)/p["i_to_e"]
                else
                    rate=inhibitory_rate(input,p,control)
                    i=rate/(1+rate);e=(input+p["i_to_i"]*i)/p["e_to_i"]
                end
                0<=e<=1 && 0<=i<=1 && push!(expected,(;input,E=e,I=i))
            end
            length(group)==length(expected) || error("mechanism nullcline grid changed: $condition/$population")
            error=maximum(max(abs(row.input-value.input),abs(row.E-value.E),abs(row.I-value.I))
                for (row,value) in zip(group,expected))
            push!(checks,(;file="mechanism/nullclines.csv:$condition/$population",
                maximum_coordinate_error=error,passed=error<1e-10))
        end
        group=filter(r->r.condition==condition,roots)
        !isempty(group) && all(r.root==index for (index,r) in enumerate(group)) ||
            error("mechanism root numbering changed: $condition")
        matched_reference=Int[]
        for row in group
            state=[row.E,row.I]
            all(0<=x<=1 for x in state) || error("mechanism root outside state domain: $condition")
            ue=p["e_to_e"]*row.E-p["i_to_e"]*row.I+.35
            ui=p["e_to_i"]*row.E-p["i_to_i"]*row.I
            fe=sigmoid(get(p,"slope_e",5.)*(ue-get(p,"theta_e",1.5)))
            fi=inhibitory_rate(ui,p,control)
            tau=get(p,"tau_e",7.8)
            field=((-row.E+(1-row.E)*fe)/tau,
                (-row.I+(1-row.I)*fi)/(tau*p["tau_ratio"]))
            j=independent_jacobian(state,p,.35,0.;control)
            trace=j.j11+j.j22;discriminant=trace^2-4*j.determinant
            alpha=discriminant>=0 ? (trace+sqrt(discriminant))/2 : trace/2
            stability=j.determinant<0 ? "Saddle" : abs(alpha)<1e-8 ? "Unresolved" :
                alpha<0 ? "Attracting" : "Repelling"
            row.stability==stability || error("mechanism root stability changed: $condition/$(row.root)")
            if !control
                matching=findall(r->maximum(abs.(state.-r["state"]))<1e-5,reference_roots)
                length(matching)==1 || error("mechanism root lacks reference correspondence: $(row.root)")
                push!(matched_reference,only(matching))
                assigned=[role for (role,k) in reference_roles if k==only(matching)]
                expected_role=isempty(assigned) ? "unassigned" : only(assigned)
                row.role==expected_role || error("mechanism root role changed: $(row.root)")
            end
            discrepancy=max(abs(field[1]),abs(field[2]))
            push!(checks,(;file="mechanism/roots.csv:$condition:$(row.root)",
                maximum_coordinate_error=discrepancy,passed=discrepancy<1e-6))
        end
        !control && sort(matched_reference)!=collect(eachindex(reference_roots)) &&
            error("mechanism omits a recorded reference root")
    end
    restriction=TOML.parsefile(joinpath(folder,"restriction.toml"))
    seizure=reference_roots[reference_roles["seizure"]]["state"]
    input=p["e_to_i"]*seizure[1]-p["i_to_i"]*seizure[2]
    peak=(get(p,"theta_on",4.)+p["theta_off"])/2
    restriction["descending_branch_condition"]==(input>=peak) ||
        error("mechanism restriction condition changed")
    discrepancy=maximum((abs(restriction["E_star"]-seizure[1]),
        abs(restriction["I_star"]-seizure[2]),
        abs(restriction["inhibitory_input"]-input),
        abs(restriction["response_peak_input"]-peak)))
    push!(checks,(;file="mechanism/restriction.toml",maximum_coordinate_error=discrepancy,
        passed=discrepancy<1e-10))
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
function verify_paired_pulse_grid(rows,durations)
    expected=Set(Float64.(durations))
    actual=Set(row.duration for row in rows)
    actual==expected || error("paired pulse duration grid changed")
    all(row.id=="selective_withdrawal" && row.baseline==1 for row in rows) ||
        error("paired pulse source changed")
    length(Set((row.duration,row.amplitude) for row in rows))==length(rows) ||
        error("duplicate paired pulse protocol")
end
function expected_paired_sustained(details,cases)
    expected=Set{Tuple{String,Int}}()
    for p in cases
        baselines=copy(p["baselines"])
        p["id"]=="input_dependent" && push!(baselines,[2.,1.])
        for (index,b) in enumerate(unique(baselines))
            b[1]>0 || continue
            folder=joinpath(details,"cases",p["id"],"baseline_$index")
            input=TOML.parsefile(joinpath(folder,"input.toml"))
            (input["B_E"],input["B_I"])==(b[1],b[2]) ||
                error("selected paired baseline changed: $folder")
            roles=TOML.parsefile(joinpath(folder,"roles.toml"))
            all(haskey(roles,role) for role in ("herald","seizure")) &&
                push!(expected,(p["id"],index))
        end
    end
    expected
end
function verify_paired_sustained_grid(rows,expected)
    actual=Set((row.id,row.baseline) for row in rows)
    length(rows)==length(expected) && actual==expected ||
        error("paired sustained context set changed")
end
function verify_intervention_grid(rows,p)
    axes=(("theta_off",vcat(p["theta_off"],collect(6.:.25:12.))),
        ("e_to_e",filter(x->0<=x<=24,p["e_to_e"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6])),
        ("e_to_i",filter(x->12<=x<=28,p["e_to_i"].*[1.,.95,.9,.85,.8,.75,.7,.65,.6])))
    expected=Set((axis,value) for (axis,values) in axes for value in values)
    actual=Set((row.axis,row.value) for row in rows)
    length(rows)==length(expected) && actual==expected ||
        error("intervention parameter grid changed: $(p["id"])")
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
    append!(checks,verify_mechanism(details,reference))
    append!(checks,verify_timescale_spectrum(details,
        joinpath(details,"timescales","selective_withdrawal_0.35_0.0"),reference,(.35,0.)))
    weak=byid["input_dependent"]
    append!(checks,verify_geometry(details,reference))
    append!(checks,verify_geometry(details,weak))
    weak_timescales=joinpath(details,"timescales","input_dependent_2.0_1.0")
    append!(checks,verify_timescale_spectrum(details,weak_timescales,weak,(2.,1.)))
    append!(checks,verify_timescale_cycles(details,weak_timescales,weak,(2.,1.)))
    append!(checks,verify_hysteresis(details,
        joinpath(details,"cases","selective_withdrawal","baseline_1"),reference,(.35,0.),config))
    for p in cases
        root=joinpath(details,"cases",p["id"])
        for baseline in filter(n->startswith(n,"baseline_"),readdir(root))
            folder=joinpath(root,baseline);input=TOML.parsefile(joinpath(folder,"input.toml"))
            orbit_files=filter(n->startswith(n,"orbit_") && endswith(n,".csv"),readdir(folder))
            TOML.parsefile(joinpath(folder,"summary.toml"))["validated_cycles"]==length(orbit_files) ||
                error("plotted cycle count changed: $folder")
            for name in orbit_files
                push!(checks,verify_source_orbit(details,joinpath(folder,name),p,
                    (input["B_E"],input["B_I"])))
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
    sustained=collect(CSV.File(joinpath(details,"paired_controls","sustained.csv")))
    verify_paired_sustained_grid(sustained,expected_paired_sustained(details,cases))
    for row in sustained
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
    paired_pulses=collect(CSV.File(joinpath(details,"paired_controls","pulses.csv")))
    verify_paired_pulse_grid(paired_pulses,config["durations"])
    for (index,row) in enumerate(paired_pulses)
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
        intervention_rows=collect(CSV.File(joinpath(base,"summary.csv")))
        verify_intervention_grid(intervention_rows,p)
        for row in intervention_rows
            q=merge(p,Dict(row.axis=>row.value))
            sub=joinpath(details,"interventions",id,"$(row.axis)_$(row.value)")
            push!(checks,verify_intervention(details,sub,q,b,row,source))
            append!(checks,verify_switching(details,sub,q,b,"fixed_switching",row.fixed_switching))
            if isfile(joinpath(sub,"remeasured_switching","summary.toml"))
                witnesses,search_checks=verify_remeasured_witnesses(details,sub,config)
                append!(checks,search_checks)
                (row.up_amplitude,row.down_amplitude,row.duration_up,row.duration_down)==
                    (witnesses[1].amplitude,witnesses[2].amplitude,
                     witnesses[1].duration,witnesses[2].duration) ||
                    error("reported remeasured protocol changed: $sub")
                append!(checks,verify_switching(details,sub,q,b,"remeasured_switching",
                    row.remeasured_switching;witnesses))
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
