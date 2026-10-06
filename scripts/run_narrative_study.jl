"""Reproducible bounded studies for the activity / persistence manuscript narrative."""
module NarrativeStudy
using FailureOfInhibition2025
import CSV, TOML, SHA, SciMLBase
include("narrative_models.jl")
using .NarrativeModels
include("run_input_release_study.jl")
const IR = InputReleaseStudy
const Evidence = IR.Evidence
const ROOT = normpath(joinpath(@__DIR__, ".."))
const SOURCE_FILES = ["scripts/run_narrative_study.jl", "scripts/narrative_models.jl",
    "scripts/run_input_release_study.jl", "scripts/input_release_models.jl",
    "scripts/run_basin_rescue_study.jl"]

function load_config(path; smoke=false)
    raw = TOML.parsefile(path)
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema"))
    s,p,d = raw["search"],raw["protocol"],raw["diagnostics"]
    for key in ("initial_points","maximum_points","screen_grid")
        Evidence.positive_integer(s[key],key)
    end
    s["initial_points"]<=s["maximum_points"] || throw(ArgumentError("invalid sample budget"))
    s["screen_grid"]>=2 || throw(ArgumentError("grid must be at least 2"))
    for key in ("e_to_e","e_to_i","theta_off","tau_ratio","baseline_E")
        values=s[key]
        length(values)==2 || throw(ArgumentError("invalid bounds"))
        all(x -> NarrativeModels.finite_number(x,key)>=0, values)
        values[1]<values[2] || throw(ArgumentError("unordered bounds"))
    end
    s["tau_ratio"][1]>0 && s["theta_off"][1]>4 || throw(ArgumentError("invalid model bounds"))
    s["halton_bases"] == [2,3,5,7,11] || throw(ArgumentError("unsupported Halton bases"))
    for key in ("amplitude_max","amplitude_step","transition_width","displacement_width")
        NarrativeModels.finite_number(p[key],key; positive=true)
    end
    Evidence.positive_integer(p["displacement_samples"],"displacement samples")
    for values in (p["durations"],p["followup_times"],s["confirmation_grids"])
        !isempty(values) && issorted(values) && length(unique(values))==length(values) ||
            throw(ArgumentError("axes must be nonempty and strictly increasing"))
        all(x -> NarrativeModels.finite_number(x,"axis"; positive=true)>0,values)
    end
    all(grid -> Evidence.positive_integer(grid,"confirmation grid")>=2,
        s["confirmation_grids"]) || throw(ArgumentError("confirmation grids must be at least 2"))
    baselines=s["anchor_baselines"]
    baselines isa AbstractVector && !isempty(baselines) && length(unique(baselines))==length(baselines) ||
        throw(ArgumentError("invalid anchor baselines"))
    foreach(value -> NarrativeModels.finite_number(value,"anchor baseline"),baselines)
    families=get(raw,"families",nothing)
    families isa AbstractVector && !isempty(families) &&
        all(family -> family isa AbstractDict && all(haskey(family,key)
            for key in ("name","i_to_e","i_to_i")),families) ||
        throw(ArgumentError("invalid families"))
    names=[family["name"] for family in families]
    all(name -> name in ("figure3","figure4"),names) && length(unique(names))==length(names) ||
        throw(ArgumentError("unsupported or repeated family"))
    for family in families, key in ("i_to_e","i_to_i")
        NarrativeModels.finite_number(family[key],key)
    end
    interventions=get(raw,"interventions",nothing)
    interventions isa AbstractDict && all(haskey(interventions,key)
        for key in ("fractions","output_factors","threshold_step")) ||
        throw(ArgumentError("invalid interventions"))
    NarrativeModels.finite_number(interventions["threshold_step"],"threshold step";positive=true)
    for key in ("fractions","output_factors")
        values=interventions[key]
        values isa AbstractVector && !isempty(values) && length(unique(values))==length(values) ||
            throw(ArgumentError("invalid intervention $key"))
        for value in values
            NarrativeModels.finite_number(value,key)
            key=="fractions" && value>1 && throw(ArgumentError("fraction exceeds one"))
        end
    end
    map_settings=get(raw,"map",nothing)
    map_settings isa AbstractDict && all(haskey(map_settings,key)
        for key in ("baseline_step","coupling_step","baseline_halfwidth","coupling_halfwidth")) ||
        throw(ArgumentError("invalid map settings"))
    for key in ("baseline_step","coupling_step")
        NarrativeModels.finite_number(map_settings[key],key;positive=true)
    end
    for key in ("baseline_halfwidth","coupling_halfwidth")
        NarrativeModels.finite_number(map_settings[key],key)
    end
    diag = DiagnosticOptions(window_duration=d["window_duration"],
        coordinate_atol=d["coordinate_atol"],balance_atol=d["balance_atol"],min_samples=d["min_samples"])
    horizons=smoke ? [1000.0,2000.0] : Float64.(p["followup_times"])
    options=PulseExperimentOptions(amplitudes=[0.0],durations=Float64.(p["durations"]),
        followup_times=horizons,diagnostic_options=diag,abstol=d["abstol"],reltol=d["reltol"],
        domain_atol=d["domain_atol"],maxiters=d["maxiters"])
    return (; raw,smoke,diagnostics=diag,horizons,abstol=options.abstol,reltol=options.reltol,
        domain_atol=options.domain_atol,maxiters=options.maxiters,options)
end

confirmation_grid(config)=last(config.raw["search"]["confirmation_grids"])

function write_record(path,data)
    mkpath(dirname(path))
    Evidence.write_toml(path*".tmp",data)
    mv(path*".tmp",path;force=true)
end

function checked_unit(dir)
    marker=joinpath(dir,"done.toml")
    isfile(marker) || return false
    files=TOML.parsefile(marker)["files"]
    for (relative,hash) in files
        isabspath(relative) || ".." in splitpath(relative) ? throw(ArgumentError("invalid manifest path")) : nothing
        isfile(joinpath(dir,relative)) && Evidence.file_hash(joinpath(dir,relative))==hash ||
            throw(ArgumentError("checkpoint checksum mismatch: $relative"))
    end
    return true
end

function complete_unit(dir)
    files=Dict{String,String}()
    for (root,_,names) in walkdir(dir), name in names
        name in ("done.toml","done.toml.tmp") && continue
        files[relpath(joinpath(root,name),dir)]=Evidence.file_hash(joinpath(root,name))
    end
    write_record(joinpath(dir,"done.toml"),Dict("files"=>files))
end

shell_quote(value::AbstractString) = "'" * replace(value, "'" => "'\\''") * "'"

function replay_command(script; stage="all", case_filter=nothing, smoke=false)
    command="julia --project=source source/scripts/$script --config config.toml --output replay --stage $stage"
    case_filter===nothing || (command *= " --case " * shell_quote(case_filter))
    smoke && (command *= " --smoke")
    command
end

function complete_replay!(metadata, script; stage, case_filter, smoke)
    commands=get!(metadata,"replay_invocations",String[])
    push!(commands,replay_command(script;stage,case_filter,smoke))
    metadata["replay_from_artifact_directory"]=join(commands," && ")
    metadata
end

function initialize(config_path, output, config; stage="all", case_filter=nothing)
    metadata_path=joinpath(output,"metadata.toml")
    if isfile(metadata_path)
        metadata=TOML.parsefile(metadata_path)
        metadata["config_sha256"]==Evidence.file_hash(config_path) || throw(ArgumentError("configuration changed"))
        metadata["smoke"]==config.smoke || throw(ArgumentError("smoke mode changed"))
        for (relative,hash) in metadata["source_sha256"]
            Evidence.file_hash(joinpath(ROOT,relative))==hash || throw(ArgumentError("source changed: $relative; use a new output directory"))
            Evidence.file_hash(joinpath(output,"source",relative))==hash || throw(ArgumentError("archived source changed"))
        end
        return metadata
    end
    isdir(output) && !isempty(readdir(output)) && throw(ArgumentError("nonempty unrecognized output"))
    mkpath(output)
    metadata=Evidence.archive_provenance(config_path,output)
    for relative in SOURCE_FILES
        destination=joinpath(output,"source",relative)
        mkpath(dirname(destination)); cp(joinpath(ROOT,relative),destination;force=true)
        metadata["source_sha256"][relative]=Evidence.file_hash(destination)
    end
    metadata["purpose"]="bounded rest-active switching, selective intervention, paired withdrawal/displacement"
    metadata["smoke"]=config.smoke
    metadata["replay_invocations"]=String[]
    metadata["replay_from_artifact_directory"]=replay_command("run_narrative_study.jl";
        stage,case_filter,smoke=config.smoke)
    metadata["claim_limits"]="provisional coordinate roles; finite-window destinations; sampled boundaries; no biological, global-attractor or exact-minimum certification"
    write_record(metadata_path,metadata)
    return metadata
end

function parameters(family, values)
    return Dict{String,Any}("family"=>family["name"],"i_to_e"=>family["i_to_e"],
        "i_to_i"=>family["i_to_i"],(k=>v for (k,v) in zip(("e_to_e","e_to_i","theta_off","tau_ratio","B_E"), values))...)
end

function cells(config,first_index,last_index; anchors=false)
    s=config.raw["search"]; result=Pair{String,Dict{String,Any}}[]
    for family in config.raw["families"]
        if anchors
            tuples=family["name"]=="figure3" ? [(17.,19.,8.,4.4),(17.,19.,8.,0.2),(17.,12.,6.,4.4)] : [(19.,14.,6.,4.4),(19.,19.,8.,0.2)]
            for (a,anchor) in enumerate(tuples), (b,baseline) in enumerate(s["anchor_baselines"])
                push!(result,"$(family["name"])_anchor$(a)_B$(b)"=>parameters(family,[anchor...,baseline]))
            end
        end
        for n in first_index:last_index
            values=[s[key][1]+NarrativeModels.radical_inverse(n,base)*(s[key][2]-s[key][1])
                for (key,base) in zip(("e_to_e","e_to_i","theta_off","tau_ratio","baseline_E"),s["halton_bases"])]
            push!(result,"$(family["name"])_halton$(n)"=>parameters(family,values))
        end
    end
    return result
end

function context(p,grid; tight=false, control=false)
    pair=NarrativeModels.models(p)
    model=control ? pair.control : pair.failure_of_inhibition
    search=IR.search_context(model,grid;tight)
    assignments=control ? Dict{String,Int}() : NarrativeModels.roles(model,search)
    return (; p,model,search,roles=assignments)
end

function save_context(dir,ctx)
    mkpath(dir)
    write_record(joinpath(dir,"parameters.toml"),ctx.p)
    write_record(joinpath(dir,"context.toml"),Evidence.context_record(ctx.search))
    write_record(joinpath(dir,"roles.toml"),ctx.roles)
end

function phase_row(phase,roles)
    destination=phase.destination===nothing ? "unresolved" : NarrativeModels.role_name(roles,phase.destination)
    return (; status=phase.status,destination,destination_index=something(phase.destination,0),
        E=phase.final[1],I=phase.final[2],horizon=phase.horizon)
end

"""Apply a finite input pulse, restore baseline, and retain the actual endpoint."""
function pulse(ctx,initial,target,amplitude,duration,config; retain=false,tight=false)
    baseline=ctx.p["B_E"]
    target=="withdraw_E" && amplitude>baseline && throw(ArgumentError("withdrawal exceeds baseline"))
    increment=target=="E" ? (amplitude,0.0) : target=="I" ? (0.0,amplitude) :
        target=="withdraw_E" ? (-amplitude,0.0) : throw(ArgumentError("unknown target"))
    drive=PiecewiseConstantDrive(baseline=(baseline,0.0),pulses=(DrivePulse(onset=0.0,
        offset=duration,increment=increment),),interpretation=AfferentExcitation)
    driven=PointModelParameters(excitatory=ctx.model.excitatory,inhibitory=ctx.model.inhibitory,
        coupling=ctx.model.coupling,drive=drive)
    times=retain ? collect(range(0.,duration;length=201)) : [0.,duration]
    # The endpoint starts a second solve, whose initial state must lie exactly in [0, 1]^2.
    solution=solve_point_model(initial,(0.,duration),driven;saveat=times,tstops=times,
        save_everystep=false,
        dense=false,abstol=config.abstol/(tight ? 10 : 1),reltol=config.reltol/(tight ? 10 : 1),
        domain_atol=0.0,maxiters=config.maxiters)
    switch=copy(last(solution.u))
    if !SciMLBase.successful_retcode(solution)
        return (; status="integration_failed",destination="unresolved",destination_index=0,
            E=switch[1],I=switch[2],horizon=0.,final=switch,phase=nothing,solution)
    end
    phase=IR.observe_phase(ctx.model,ctx.search,switch,config;retain,tight,handoff=true)
    return merge(phase_row(phase,ctx.roles),(;final=phase.final,phase,solution))
end

function pulse_map(ctx,config,dir; sources=sort(collect(keys(ctx.roles))))
    rows=NamedTuple[]; witnesses=Dict{String,Any}()
    mkpath(dir)
    durations=config.smoke ? [20.,200.] : config.raw["protocol"]["durations"]
    for source in sources
        source in keys(ctx.roles) || continue
        initial=ctx.search.equilibria[ctx.roles[source]].state
        for target in ("E","I","withdraw_E"), duration in durations
            cap=target=="withdraw_E" ? ctx.p["B_E"] : config.raw["protocol"]["amplitude_max"]
            function evaluate(amplitude)
                trial=pulse(ctx,initial,target,amplitude,duration,config)
                row=(; source,target,amplitude,duration,status=trial.status,
                    destination=trial.destination,E=trial.E,I=trial.I,horizon=trial.horizon,
                    switch_E=last(trial.solution.u)[1],switch_I=last(trial.solution.u)[2])
                push!(rows,row)
                write_record(joinpath(dir,"attempts","$(source)_$(target)_$(duration)_$(amplitude).toml"),
                    (; row,initial,phase=trial.phase===nothing ? nothing : IR.phase_record(trial.phase)))
                if trial.status=="compatible" && trial.destination!=source
                    key=source*"_to_"*trial.destination
                    old=get(witnesses,key,nothing)
                    if old===nothing || (amplitude,duration,target)<(old.amplitude,old.duration,old.target)
                        witnesses[key]=row
                    end
                end
                return trial
            end
            NarrativeModels.sample_line(evaluate,cap;step=config.raw["protocol"]["amplitude_step"],
                width=config.raw["protocol"]["transition_width"])
        end
    end
    mkpath(dir)
    isempty(rows) || CSV.write(joinpath(dir,"pulses.csv"),rows)
    write_record(joinpath(dir,"witnesses.toml"),witnesses)
    return (; rows,witnesses)
end

function repeat_switching(ctx,witnesses,config,dir; tight=false)
    required=("rest_to_active","active_to_rest")
    all(k -> haskey(witnesses,k),required) || return false
    initial=copy(ctx.search.equilibria[ctx.roles["rest"]].state)
    for cycle in 1:2, key in required
        witness=witnesses[key]
        getvalue(k)=witness isa AbstractDict ? witness[k] : getproperty(witness,Symbol(k))
        trial=pulse(ctx,initial,getvalue("target"),getvalue("amplitude"),getvalue("duration"),config;retain=true,tight)
        trial.phase===nothing && return false
        IR.save_phase(dir,"cycle$(cycle)_$key",trial.phase)
        write_trajectory_csv(joinpath(dir,"cycle$(cycle)_$(key)_pulse.csv"),trial.solution)
        trial.status=="compatible" && trial.destination==last(split(key,"_to_")) || return false
        initial=copy(trial.final)
    end
    return true
end

function qualify(ctx,config,dir)
    all(k -> haskey(ctx.roles,k),("rest","active","seizure")) || return Dict("qualified"=>false,"reason"=>"structural_roles_unavailable")
    probes=pulse_map(ctx,config,dir;sources=["rest","active"])
    repeat_ok=repeat_switching(ctx,probes.witnesses,config,dir)
    induction=any(k -> haskey(probes.witnesses,k),("rest_to_seizure","active_to_seizure"))
    return Dict("qualified"=>repeat_ok && induction,"repeat_switching"=>repeat_ok,
        "induction"=>induction,"reason"=>repeat_ok && induction ? "qualified" : "sampled_protocols_not_qualified")
end

function screen_stage(config,output;case_filter=nothing)
    summary=Dict{String,Any}[]
    initial=config.smoke ? 2 : config.raw["search"]["initial_points"]
    maximum=config.smoke ? 2 : config.raw["search"]["maximum_points"]
    for (lo,hi,anchors) in ((1,initial,true),(initial+1,maximum,false))
        lo>hi && continue
        !anchors && any(x -> x["qualified"],summary) && break
        for (id,p) in cells(config,lo,hi;anchors)
            case_filter!==nothing && !occursin(case_filter,id) && continue
            dir=joinpath(output,"screen",id)
            if checked_unit(dir)
                push!(summary,TOML.parsefile(joinpath(dir,"summary.toml"))); continue
            end
            ctx=context(p,config.raw["search"]["screen_grid"])
            save_context(dir,ctx)
            result=Dict{String,Any}("qualified"=>false,"repeat_switching"=>false,"induction"=>false,
                "reason"=>"structural_roles_unavailable")
            if all(k -> haskey(ctx.roles,k),("rest","active","seizure"))
                println("qualifying ",id," at B_E=",p["B_E"]); flush(stdout)
                confirmed=context(p,21)
                save_context(joinpath(dir,"grid21"),confirmed)
                merge!(result,qualify(confirmed,config,joinpath(dir,"qualification")))
            end
            row=merge(copy(p),result,Dict("id"=>id,"roots"=>length(ctx.search.equilibria),
                "sinks"=>length(IR.sink_indices(ctx.search)),"roles"=>join(sort(collect(keys(ctx.roles))),",")))
            write_record(joinpath(dir,"summary.toml"),row)
            complete_unit(dir); push!(summary,row)
            length(summary)%32==0 && (println("screened ",length(summary)," cells");flush(stdout))
        end
    end
    isempty(summary) || CSV.write(joinpath(output,"screen.csv"),summary)
    write_record(joinpath(output,"screen_summary.toml"),Dict("cells"=>length(summary),
        "qualified"=>count(x->x["qualified"],summary),"maximum_points"=>maximum,
        "case_filter"=>something(case_filter,"all")))
    return summary
end

function selected_stage(config,output)
    summaries=[TOML.parsefile(joinpath(output,"screen",id,"summary.toml"))
        for id in sort(readdir(joinpath(output,"screen"))) if checked_unit(joinpath(output,"screen",id))]
    # Prefer positive background and complete role sets, never the desired rescue ordering.
    candidates=filter(x->x["qualified"],summaries)
    sort!(candidates;by=x -> (x["B_E"]<=0,!occursin("herald",x["roles"]),x["id"]))
    for row in candidates
        p=Dict(k=>row[k] for k in ("family","e_to_e","i_to_e","e_to_i","i_to_i","theta_off","tau_ratio","B_E"))
        dir=joinpath(output,"confirmation",row["id"])
        if checked_unit(dir)
            result=TOML.parsefile(joinpath(dir,"result.toml"))
            result["qualified"] && return (p=p,id=row["id"],directory=dir)
            continue
        end
        witnesses=TOML.parsefile(joinpath(output,"screen",row["id"],"qualification","witnesses.toml"))
        ok=true
        for grid in config.raw["search"]["confirmation_grids"]
            ctx=context(p,grid;tight=true); sub=joinpath(dir,"grid$grid")
            save_context(sub,ctx)
            matched=all(k->haskey(ctx.roles,k),("rest","active","seizure"))
            repeated=matched && repeat_switching(ctx,witnesses,config,sub;tight=true)
            induced=false
            for source in ("rest","active")
                key=source*"_to_seizure"
                matched && haskey(witnesses,key) || continue
                w=witnesses[key]
                trial=pulse(ctx,ctx.search.equilibria[ctx.roles[source]].state,w["target"],w["amplitude"],w["duration"],config;retain=true,tight=true)
                if trial.phase!==nothing
                    IR.save_phase(sub,key,trial.phase)
                    write_trajectory_csv(joinpath(sub,key*"_pulse.csv"),trial.solution)
                    induced |= trial.destination=="seizure" && trial.status=="compatible"
                end
            end
            ok &= repeated && induced
        end
        write_record(joinpath(dir,"result.toml"),Dict("qualified"=>ok,"id"=>row["id"]))
        complete_unit(dir)
        ok && return (p=p,id=row["id"],directory=dir)
    end
    return nothing
end

"""Permanent input withdrawal and direct E displacement are separate experiments."""
function recovery_trial(ctx,source,amount,protocol,config;initial=nothing,retain=false,tight=false)
    amount=NarrativeModels.finite_number(amount,"reduction")
    state=initial===nothing ? copy(ctx.search.equilibria[ctx.roles[source]].state) : copy(initial)
    if protocol=="permanent_withdrawal"
        amount<=ctx.p["B_E"] || throw(ArgumentError("withdrawal exceeds baseline"))
        p=merge(ctx.p,Dict("B_E"=>ctx.p["B_E"]-amount))
        destination=context(p,tight ? confirmation_grid(config) : 21;tight)
        matched=NarrativeModels.match_roles(ctx.search,ctx.roles,destination.search)
        # Independent contextual assignments do not overwrite ambiguous or lost branch identities.
        destination=merge(destination,(;roles=matched))
    elseif protocol=="direct_E_displacement"
        amount<=state[1] || throw(ArgumentError("displacement exceeds E"))
        state[1]-=amount; destination=ctx
    else
        throw(ArgumentError("unknown reduction protocol"))
    end
    phase=IR.observe_phase(destination.model,destination.search,state,config;retain,tight)
    row=phase_row(phase,destination.roles)
    return merge(row,(;phase,context=destination,initial=state,
        recovery=row.status=="compatible" && row.destination in ("rest","active")))
end

function recovery_stage(ctx,config,dir)
    checked_unit(dir) && return
    mkpath(dir); save_context(joinpath(dir,"baseline"),ctx)
    rows=NamedTuple[]; boundaries=NamedTuple[]
    for source in ("herald","seizure")
        haskey(ctx.roles,source) || continue
        for protocol in ("permanent_withdrawal","direct_E_displacement")
            cap=protocol=="permanent_withdrawal" ? ctx.p["B_E"] : ctx.search.equilibria[ctx.roles[source]].state[1]
            relative_step=max(cap/config.raw["protocol"]["displacement_samples"],eps())
            step=protocol=="permanent_withdrawal" ? min(config.raw["protocol"]["amplitude_step"],relative_step) : relative_step
            width=protocol=="permanent_withdrawal" ? min(config.raw["protocol"]["transition_width"],relative_step/4) : config.raw["protocol"]["displacement_width"]
            evaluate(a)=recovery_trial(ctx,source,a,protocol,config)
            cache=NarrativeModels.sample_line(evaluate,cap;step,width)
            amounts=sort(collect(keys(cache)))
            for a in amounts
                trial=cache[a]
                sub=joinpath(dir,source,protocol,string(a)); mkpath(sub)
                save_context(joinpath(sub,"destination"),trial.context)
                write_record(joinpath(sub,"phase.toml"),IR.phase_record(trial.phase))
                push!(rows,(;source,protocol,reduction=a,fraction=cap>0 ? a/cap : 0.,
                    status=trial.status,destination=trial.destination,recovery=trial.recovery,
                    final_E=trial.E,final_I=trial.I,horizon=trial.horizon))
            end
            chosen=Float64[0.0,cap]
            for (a,b) in zip(amounts,amounts[2:end])
                x,y=cache[a],cache[b]
                if (x.status,x.destination)!=(y.status,y.destination)
                    push!(boundaries,(;source,protocol,lower=a,upper=b,
                        lower_status=x.status,upper_status=y.status,lower_destination=x.destination,upper_destination=y.destination))
                    append!(chosen,[a,b])
                end
            end
            successful=filter(a->cache[a].recovery,amounts)
            isempty(successful) || push!(chosen,first(successful))
            for a in unique(chosen)
                trial=recovery_trial(ctx,source,a,protocol,config;retain=true,tight=true)
                sub=joinpath(dir,source,protocol,string(a))
                IR.save_phase(sub,"confirmation",trial.phase)
                write_record(joinpath(sub,"confirmation_summary.toml"),phase_row(trial.phase,trial.context.roles))
            end
        end
    end
    isempty(rows) || CSV.write(joinpath(dir,"recovery.csv"),rows)
    isempty(boundaries) || CSV.write(joinpath(dir,"boundaries.csv"),boundaries)
    complete_unit(dir)
end

function induced_recovery_stage(ctx,config,witnesses,dir)
    checked_unit(dir) && return
    mkpath(dir);results=Dict{String,Any}()
    for source in ("herald","seizure")
        haskey(ctx.roles,source) || continue
        keys_found=[start*"_to_"*source for start in ("rest","active") if haskey(witnesses,start*"_to_"*source)]
        if isempty(keys_found)
            results[source]=Dict("status"=>"no_induction_witness");continue
        end
        key=first(keys_found);w=witnesses[key];start=first(split(key,"_to_"))
        trial=pulse(ctx,ctx.search.equilibria[ctx.roles[start]].state,w["target"],w["amplitude"],w["duration"],config;retain=true,tight=true)
        if trial.status!="compatible" || trial.destination!=source
            results[source]=Dict("status"=>"induction_not_confirmed");continue
        end
        sub=joinpath(dir,source);mkpath(sub)
        IR.save_phase(sub,"induction",trial.phase)
        write_trajectory_csv(joinpath(sub,"induction_pulse.csv"),trial.solution)
        control=IR.observe_phase(ctx.model,ctx.search,trial.final,config;retain=true,tight=true)
        IR.save_phase(sub,"control",control)
        rows=CSV.File(joinpath(dirname(dir),"recovery","recovery.csv"))
        for protocol in ("permanent_withdrawal","direct_E_displacement")
            relevant=filter(x->x.source==source && x.protocol==protocol,collect(rows))
            successful=filter(x->x.recovery,relevant)
            chosen=isempty(successful) ? maximum(x.reduction for x in relevant) : minimum(x.reduction for x in successful)
            protocol=="direct_E_displacement" && (chosen=min(chosen,trial.final[1]))
            outcome=recovery_trial(ctx,source,chosen,protocol,config;initial=trial.final,retain=true,tight=true)
            IR.save_phase(sub,protocol,outcome.phase)
            write_record(joinpath(sub,protocol*"_summary.toml"),(; reduction=chosen,
                actual_switch_state=trial.final,initial=outcome.initial,status=outcome.status,
                destination=outcome.destination,recovery=outcome.recovery))
        end
        results[source]=Dict("status"=>"induced_and_tested","control_destination"=>phase_row(control,ctx.roles).destination)
    end
    write_record(joinpath(dir,"summary.toml"),results);complete_unit(dir)
end

function intervention_stage(ctx,config,dir)
    summaries=Dict{String,Any}[]
    axes=[("theta_off",unique(vcat(ctx.p["theta_off"],collect(ctx.p["theta_off"]:config.raw["interventions"]["threshold_step"]:12.0),12.0))),
        ("e_to_i",ctx.p["e_to_i"].*config.raw["interventions"]["fractions"]),
        ("e_to_e",ctx.p["e_to_e"].*config.raw["interventions"]["fractions"]),
        ("i_to_e",ctx.p["i_to_e"].*config.raw["interventions"]["output_factors"])]
    for (axis,values) in axes, value in values
        id=axis*"_"*string(value); sub=joinpath(dir,id)
        if checked_unit(sub)
            push!(summaries,TOML.parsefile(joinpath(sub,"summary.toml")));continue
        end
        changed=context(merge(ctx.p,Dict(axis=>value)),21)
        changed=merge(changed,(;roles=NarrativeModels.match_roles(ctx.search,ctx.roles,changed.search)))
        save_context(sub,changed)
        probes=pulse_map(changed,config,sub;sources=["rest","active"])
        repeated=all(k->haskey(changed.roles,k),("rest","active")) &&
            repeat_switching(changed,probes.witnesses,config,sub)
        # Apply the parameter intervention to each actual baseline source, not a new root.
        destinations=Dict{String,Any}()
        for source in ("rest","active","herald","seizure")
            haskey(ctx.roles,source) || continue
            phase=IR.observe_phase(changed.model,changed.search,ctx.search.equilibria[ctx.roles[source]].state,config;retain=true)
            IR.save_phase(sub,"from_"*source,phase)
            destinations[source]=phase_row(phase,changed.roles)
        end
        write_record(joinpath(sub,"destinations.toml"),destinations)
        row=Dict{String,Any}("axis"=>axis,"value"=>value,"switching_preserved"=>repeated,
            "roles"=>join(sort(collect(keys(changed.roles))),","),
            "seizure_after_parameter_change"=>haskey(destinations,"seizure") ? destinations["seizure"].destination : "unavailable",
            "seizure_induced_from_rest"=>haskey(probes.witnesses,"rest_to_seizure"),
            "seizure_induced_from_active"=>haskey(probes.witnesses,"active_to_seizure"))
        for role in ("rest","active","herald","seizure"), (j,coordinate) in enumerate(("E","I"))
            row[role*"_"*coordinate]=haskey(changed.roles,role) ? changed.search.equilibria[changed.roles[role]].state[j] : "unavailable"
        end
        write_record(joinpath(sub,"summary.toml"),row);complete_unit(sub);push!(summaries,row)
        println("intervention ",id," switching=",repeated);flush(stdout)
    end
    CSV.write(joinpath(dir,"summary.csv"),summaries)
end

function map_stage(ctx,config,dir,witnesses)
    rows=Dict{String,Any}[]; p=ctx.p; settings=config.raw["map"]
    elo,ehi=max(0.,p["e_to_e"]-settings["coupling_halfwidth"]),min(24.,p["e_to_e"]+settings["coupling_halfwidth"])
    blo,bhi=max(0.,p["B_E"]-settings["baseline_halfwidth"]),min(8.,p["B_E"]+settings["baseline_halfwidth"])
    es=sort(unique(vcat(p["e_to_e"],collect(elo:settings["coupling_step"]:ehi))))
    bs=sort(unique(vcat(0.,p["B_E"],collect(blo:settings["baseline_step"]:bhi))))
    for (i,e) in enumerate(es),(j,b) in enumerate(bs)
        sub=joinpath(dir,"e$(i)_b$(j)")
        if checked_unit(sub)
            raw=TOML.parsefile(joinpath(sub,"summary.toml"))
            push!(rows,raw);continue
        end
        changed=context(merge(p,Dict("e_to_e"=>e,"B_E"=>b)),21)
        changed=merge(changed,(;roles=NarrativeModels.match_roles(ctx.search,ctx.roles,changed.search)))
        save_context(sub,changed)
        row=Dict{String,Any}("e_to_e"=>e,"B_E"=>b,"sinks"=>length(IR.sink_indices(changed.search)),
            "roles"=>join(sort(collect(keys(changed.roles))),","))
        for name in ("rest_to_active","active_to_rest","rest_to_seizure","active_to_seizure")
            source=first(split(name,"_to_"))
            row[name]="unavailable"
            haskey(witnesses,name) && haskey(changed.roles,source) || continue
            w=witnesses[name]
            if w["target"]=="withdraw_E" && w["amplitude"]>b
                row[name]="withdrawal_exceeds_baseline";continue
            end
            trial=pulse(changed,changed.search.equilibria[changed.roles[source]].state,
                w["target"],w["amplitude"],w["duration"],config)
            row[name]=trial.status=="compatible" ? trial.destination : trial.status
            trial.phase===nothing || IR.save_phase(sub,name,trial.phase)
        end
        for source in ("herald","seizure")
            row[source*"_after_full_withdrawal"]="unavailable"
            haskey(changed.roles,source) || continue
            trial=recovery_trial(changed,source,b,"permanent_withdrawal",config)
            row[source*"_after_full_withdrawal"]=trial.status=="compatible" ? trial.destination : trial.status
            IR.save_phase(sub,source*"_release",trial.phase)
        end
        write_record(joinpath(sub,"summary.toml"),row);complete_unit(sub);push!(rows,row)
    end
    CSV.write(joinpath(dir,"map.csv"),rows)
    for role in sort(collect(keys(ctx.roles)))
        file=joinpath(dir,"continuation_"*role*".toml")
        isfile(file) && continue
        factory=b->NarrativeModels.models(merge(p,Dict("B_E"=>b))).failure_of_inhibition
        result=continue_equilibria(factory,ctx.search.equilibria[ctx.roles[role]].state,p["B_E"];
            parameter_bounds=(0.,8.),options=ContinuationOptions(max_steps=600))
        write_record(file,result)
    end
end

function robustness_stage(ctx,config,witnesses,dir)
    rows=Dict{String,Any}[]
    for (key,offset,limits) in (("e_to_e",0.25,(0.,24.)),("e_to_i",0.5,(12.,28.)),
            ("theta_off",0.25,(6.,12.)),("tau_ratio",0.2,(0.2,4.4))), sign in (-1,1)
        value=ctx.p[key]+sign*offset
        limits[1]<=value<=limits[2] || continue
        sub=joinpath(dir,key*"_"*string(value))
        if checked_unit(sub)
            push!(rows,TOML.parsefile(joinpath(sub,"summary.toml")));continue
        end
        changed=context(merge(ctx.p,Dict(key=>value)),21;tight=true)
        changed=merge(changed,(;roles=NarrativeModels.match_roles(ctx.search,ctx.roles,changed.search)))
        save_context(sub,changed)
        repeated=all(k->haskey(changed.roles,k),("rest","active")) &&
            repeat_switching(changed,witnesses,config,sub;tight=true)
        induction=false
        for start in ("rest","active")
            name=start*"_to_seizure"
            haskey(witnesses,name) && haskey(changed.roles,start) || continue
            w=witnesses[name]
            w["target"]=="withdraw_E" && w["amplitude"]>changed.p["B_E"] && continue
            trial=pulse(changed,changed.search.equilibria[changed.roles[start]].state,w["target"],w["amplitude"],w["duration"],config;tight=true)
            induction |= trial.status=="compatible" && trial.destination=="seizure"
        end
        recovery_stage(changed,config,joinpath(sub,"recovery"))
        row=Dict{String,Any}("axis"=>key,"value"=>value,"same_switching_protocols"=>repeated,
            "same_induction_protocols"=>induction,"roles"=>join(sort(collect(keys(changed.roles))),","))
        write_record(joinpath(sub,"summary.toml"),row);complete_unit(sub);push!(rows,row)
    end
    isempty(rows) || CSV.write(joinpath(dir,"summary.csv"),rows)
end

function run_study(config_path,output_dir;stage="all",smoke=false,case_filter=nothing)
    stage in ("all","screen","confirm","recovery","interventions","map","robustness") || throw(ArgumentError("unknown stage"))
    config=load_config(config_path;smoke);output=abspath(output_dir)
    metadata=initialize(config_path,output,config;stage,case_filter)
    stage in ("all","screen") && screen_stage(config,output;case_filter)
    if stage=="screen"
        complete_replay!(metadata,"run_narrative_study.jl";stage,case_filter,smoke)
        metadata["last_completed_stage"]=stage;metadata["completed"]=false
        write_record(joinpath(output,"metadata.toml"),metadata)
        Evidence.artifact_checksums(output)
        return output
    end
    isdir(joinpath(output,"screen")) || throw(ArgumentError("run screen stage first"))
    selected=selected_stage(config,output)
    if selected!==nothing
        final_grid=confirmation_grid(config)
        write_record(joinpath(output,"selected.toml"),Dict("id"=>selected.id,"parameters"=>selected.p,
            "confirmation_directory"=>relpath(selected.directory,output),"final_grid"=>final_grid))
        ctx=context(selected.p,final_grid;tight=true)
        witnesses=TOML.parsefile(joinpath(output,"screen",selected.id,"qualification","witnesses.toml"))
        control=context(selected.p,final_grid;tight=true,control=true)
        save_context(joinpath(output,"matched_control"),control)
        stage in ("all","recovery") && recovery_stage(ctx,config,joinpath(output,"recovery"))
        stage in ("all","recovery") && induced_recovery_stage(ctx,config,witnesses,joinpath(output,"induced_recovery"))
        stage in ("all","interventions") && intervention_stage(ctx,config,joinpath(output,"interventions"))
        stage in ("all","map") && map_stage(ctx,config,joinpath(output,"map"),witnesses)
        stage in ("all","robustness") && robustness_stage(ctx,config,witnesses,joinpath(output,"robustness"))
    end
    metadata["selected_example"]=selected===nothing ? "unavailable" : selected.id
    metadata["last_completed_stage"]=stage
    metadata["completed"]=stage=="all" && case_filter===nothing
    complete_replay!(metadata,"run_narrative_study.jl";stage,case_filter,smoke)
    write_record(joinpath(output,"metadata.toml"),metadata)
    Evidence.artifact_checksums(output)
    return output
end

function main(args=ARGS)
    options=Dict{String,String}();smoke=false;i=1
    while i<=length(args)
        key=args[i]
        if key=="--smoke"
            smoke=true;i+=1;continue
        end
        key in ("--config","--output","--stage","--case") && i<length(args) || throw(ArgumentError("invalid argument $key"))
        haskey(options,key) && throw(ArgumentError("duplicate $key"))
        options[key]=args[i+1];i+=2
    end
    haskey(options,"--output") || throw(ArgumentError("--output is required"))
    return run_study(get(options,"--config",joinpath(ROOT,"experiments/narrative_study.toml")),options["--output"];
        stage=get(options,"--stage","all"),smoke,case_filter=get(options,"--case",nothing))
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    NarrativeStudy.main()
end
