"""Build a portable, checked plotting packet from completed figure studies."""
module FigureRedesignData
using FailureOfInhibition2025
import CSV,TOML
include("run_figure_details.jl")
include("figure_packet_io.jl")
const D=FigureDetails
const S=D.S
const IOChecks=FigurePacketIO

function compact_roots(directory)
    context=TOML.parsefile(joinpath(directory,"context.toml"))
    roles=TOML.parsefile(joinpath(directory,"roles.toml"))
    [(;root=k,E=r["state"][1],I=r["state"][2],stability=r["stability"]["classification"],
        role=something(findfirst(==(k),roles),"unassigned")) for (k,r) in enumerate(context["equilibria"])]
end

function build_mechanism(selection,output;cfg=S.load_config(joinpath(D.ROOT,"experiments/figure_redesign.toml")))
    p=only(filter(p->p["id"]=="selective_withdrawal",selection["cases"]))
    ref=S.context(p,.35,0.,cfg;tight=true,grid=41)
    mechanism=joinpath(output,"mechanism");mkpath(mechanism)
    response_rows=NamedTuple[];nullclines=NamedTuple[];roots=NamedTuple[]
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
    CSV.write(joinpath(mechanism,"responses.csv"),response_rows)
    CSV.write(joinpath(mechanism,"nullclines.csv"),nullclines);CSV.write(joinpath(mechanism,"roots.csv"),roots)
    star=ref.search.equilibria[ref.roles["seizure"]].state
    inhibitory_input=p["e_to_i"]*star[1]-p["i_to_i"]*star[2]
    S.write_record(joinpath(mechanism,"restriction.toml"),Dict("E_star"=>star[1],"I_star"=>star[2],
        "inhibitory_input"=>inhibitory_input,"response_peak_input"=>(get(p,"theta_on",4.)+p["theta_off"])/2,
        "descending_branch_condition"=>inhibitory_input>=(get(p,"theta_on",4.)+p["theta_off"])/2,
        "drive_condition"=>"Both drive increments are nonnegative relative to baseline throughout the protocol"))
end

function build(screen,details,output)
    ispath(output) && error("packet destination already exists")
    metadata=TOML.parsefile(joinpath(details,"metadata.toml"))
    get(metadata,"completed",false) || error("detail study is incomplete")
    config_path=joinpath(details,"config.toml")
    IOChecks.digest(config_path)==metadata["config_sha256"] || error("recorded configuration changed")
    cfg=S.load_config(config_path)
    cfg.raw["redesign"]["reference"]=="selective_withdrawal" || error("this figure design requires the declared reference circuit")
    verification=TOML.parsefile(joinpath(details,"verification","summary.toml"))
    get(verification,"passed",false) || error("independent verification has not passed")
    verification["verifier_sha256"]==IOChecks.digest(joinpath(D.ROOT,"scripts/verify_figure_redesign.jl")) || error("verifier source changed")
    verification["packet_io_sha256"]==IOChecks.digest(joinpath(D.ROOT,"scripts/figure_packet_io.jl")) || error("verification helper changed")
    verification["details_metadata_sha256"]==IOChecks.digest(joinpath(details,"metadata.toml")) || error("verified study identity changed")
    verification["selection_sha256"]==IOChecks.digest(joinpath(details,"selection.toml")) || error("verified selection changed")
    verification["trajectories_sha256"]==IOChecks.digest(joinpath(details,"verification","trajectories.csv")) ||
        error("verification trajectory report changed")
    verification["completed_stage_manifests_sha256"]==
        IOChecks.completed_stage_manifests(details,metadata["completed_stages"]) ||
        error("verified stage contents changed")
    for (relative,hash) in metadata["source_sha256"]
        S.N.Evidence.file_hash(joinpath(D.ROOT,relative))==hash || error("source mismatch: $relative")
    end
    selection=TOML.parsefile(joinpath(details,"selection.toml"))
    selection["screen_metadata_sha256"]==IOChecks.digest(joinpath(screen,"metadata.toml")) || error("screen identity changed")
    D.F.A.verify_study_manifest(screen)
    selection["screen_manifest_sha256"]==IOChecks.digest(joinpath(screen,"checksums.toml")) ||
        error("screen contents changed since selection")
    mkpath(output);sources=Dict{String,String}();derived=Dict{String,Any}()
    function copy_evidence(base,relative;destination=relative)
        source=joinpath(base,relative);target=joinpath(output,destination)
        mkpath(dirname(target));hash=IOChecks.digest(source);cp(source,target)
        IOChecks.digest(target)==hash || error("copy changed data")
        sources[destination]=hash
    end
    function select_rows(relative,predicate,description)
        source=joinpath(details,relative);target=joinpath(output,relative)
        original=collect(CSV.File(source));selected=filter(predicate,original)
        mkpath(dirname(target));CSV.write(target,selected)
        derived[relative]=Dict("source_sha256"=>IOChecks.digest(source),"source_rows"=>length(original),
            "retained_rows"=>length(selected),"selection"=>description)
    end
    copy_evidence(details,"selection.toml");copy_evidence(details,"metadata.toml");copy_evidence(details,"config.toml")
    for kind in ("serialization_repair","replay_repair")
        haskey(metadata,kind) || continue
        for name in ("record.toml","previous_driver.jl","previous_metadata.toml","repair.jl")
            copy_evidence(details,joinpath(kind,name))
        end
    end
    copy_evidence(details,"verification/summary.toml");copy_evidence(details,"verification/trajectories.csv")
    copy_evidence(screen,"screen.csv");copy_evidence(screen,"metadata.toml";destination="screen_metadata.toml")
    allscreen=NamedTuple[];screen_sources=Dict{String,String}()
    for p in D.F.registry(cfg)
        S.N.checked_unit(joinpath(screen,p["category"],p["id"])) || error("incomplete screen case")
        path=joinpath(screen,p["category"],p["id"],"behavior.csv")
        isfile(path) || continue
        screen_sources[relpath(path,screen)]=IOChecks.digest(path)
        for row in CSV.File(path)
            push!(allscreen,merge((;id=p["id"],e_to_e=p["e_to_e"],e_to_i=p["e_to_i"],
                theta_off=p["theta_off"],tau_ratio=p["tau_ratio"]),NamedTuple(row)))
        end
    end
    CSV.write(joinpath(output,"screen_responses.csv"),allscreen)
    derived["screen_responses.csv"]=Dict("source_sha256"=>screen_sources,
        "selection"=>"Every recorded screen response, with frozen case identifiers and parameters attached")
    baseline_rows=NamedTuple[]
    for p in selection["cases"]
        id=p["id"];root=joinpath("cases",id)
        copy_evidence(details,joinpath(root,"summary.toml"))
        copy_evidence(details,joinpath(root,"geometry_confirmation","independent_searches.csv"))
        if id in ("selective_withdrawal","input_dependent")
            for name in ("inputs.csv","cells.csv","bounds.toml","summary.toml")
                file=joinpath(root,"geometry",name)
                isfile(joinpath(details,file)) && copy_evidence(details,file)
            end
        end
        for (index,b) in enumerate(D.target_baselines(p))
            sub=joinpath(root,"baseline_$index");summary=TOML.parsefile(joinpath(details,sub,"summary.toml"))
            copy_evidence(details,joinpath(sub,"summary.toml"));copy_evidence(details,joinpath(sub,"probes.csv"))
            roots=compact_roots(joinpath(details,sub))
            CSV.write(joinpath(output,sub,"roots.csv"),roots)
            derived[joinpath(sub,"roots.csv")]=Dict("source_sha256"=>Dict(
                name=>IOChecks.digest(joinpath(details,sub,name)) for name in ("context.toml","roles.toml")),
                "selection"=>"All discovered roots; retain coordinates, local stability, and assigned role")
            push!(baseline_rows,(;id,baseline=index,B_E=b[1],B_I=b[2],tau_ratio=p["tau_ratio"],
                validated_cycles=summary["validated_cycles"],sources=length(summary["sources"])))
            for k in 1:summary["validated_cycles"], ext in ("csv","toml")
                copy_evidence(details,joinpath(sub,"orbit_$k.$ext"))
            end
            for source in summary["sources"]
                folder=joinpath(sub,source["source"])
                # Main accessibility maps use the reference; cycle panels retain all measured phases.
                if (id=="selective_withdrawal" && index==1) || source["role"]=="oscillatory"
                    for relative in ("pulses/pulses.csv","pulses/boundaries.csv","pulses/unresolved_intervals.csv",
                        "hysteresis/sweeps.csv","sustained/confirmations.csv")
                        if isfile(joinpath(details,folder,relative))
                            path=joinpath(folder,relative)
                            if relative=="pulses/pulses.csv"
                                select_rows(path,r->r.axis!="joint","All axial pulse observations; unplotted joint-input observations remain in the full study")
                            else
                                copy_evidence(details,path)
                            end
                        end
                    end
                end
            end
        end
    end
    CSV.write(joinpath(output,"baselines.csv"),baseline_rows)
    for folder in ("sequence","paired_controls","timescales","interventions")
        for (dir,_,names) in walkdir(joinpath(details,folder)), name in names
            relative=relpath(joinpath(dir,name),details)
            # Retain the plotted arrays and concise records, not nonlinear-search attempts.
            allowed=folder=="sequence" ? name in ("trajectory.csv","events.csv","inputs.csv","summary.toml") :
                folder=="paired_controls" ? dir==joinpath(details,folder) && endswith(name,".csv") :
                folder=="interventions" ? name=="summary.csv" :
                endswith(name,".csv") || name in ("parameters.toml","orbit.toml")
            if allowed
                if relative==joinpath("sequence","trajectory.csv")
                    events=collect(CSV.File(joinpath(details,"sequence","events.csv")))
                    select_rows(relative,r->any(e->e.onset-50<=r.time<=e.offset+250,events),
                        "Exact display windows: 50 ms before through 250 ms after each 100 ms pulse; omitted settled intervals remain in the full connected trajectory")
                else
                    copy_evidence(details,relative)
                end
            end
        end
    end
    build_mechanism(selection,output;cfg)
    S.write_record(joinpath(output,"packet.toml"),Dict("schema_version"=>1,"purpose"=>"Five question-led figure drafts",
        "copied_artifact_sha256"=>sources,"builder_sha256"=>IOChecks.digest(@__FILE__),
        "derived_artifacts"=>derived,
        "limits"=>"Bounded numerical discovery and finite-window control; CompletenessNotCertified; roles provisional",
        "selection"=>"See selection.toml. Geometry panels: reference and weak-recurrence input maps; all selected baselines indexed.",
        "filtering"=>"Reference axial pulse/sweep maps, axial pulses at every measured cycle phase, and exact connected-trace display windows retained. Joint pulses and omitted settled trace intervals remain in full study; derived_artifacts records the selections."))
    IOChecks.write_manifest(output);IOChecks.verify_manifest(output)
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3 || error("usage: build_figure_redesign.jl SCREEN DETAILS NEW_PACKET")
    FigureRedesignData.build(ARGS...)
end
