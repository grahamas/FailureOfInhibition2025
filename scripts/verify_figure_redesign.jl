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
function verify(details,output)
    ispath(output) && error("verification output exists")
    metadata=TOML.parsefile(joinpath(details,"metadata.toml"))
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
                pulses=joinpath(folder,source,"pulses")
                trials=collect(CSV.File(joinpath(pulses,"pulses.csv")))
                for destination in sort(unique(r.destination for r in trials))
                    destination in ("integration_failed","unresolved") && continue
                    witness=first(sort(filter(r->r.destination==destination,trials);by=r->(r.duration,r.B_E,r.B_I)))
                    confirmation=joinpath(pulses,"confirmation")
                    push!(checks,compare(joinpath(confirmation,destination*"_pulse.csv"),p,witness.B_E,witness.B_I,details))
                    push!(checks,compare(joinpath(confirmation,destination*".csv"),p,input["B_E"],input["B_I"],details))
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
    for id in ("selective_withdrawal","threshold_to_active")
        p=byid[id];b=first(p["baselines"])
        for row in CSV.File(joinpath(details,"interventions",id,"summary.csv"))
            q=merge(p,Dict(row.axis=>row.value))
            file=joinpath(details,"interventions",id,"$(row.axis)_$(row.value)","original_seizure.csv")
            push!(checks,compare(file,q,b...,details))
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
