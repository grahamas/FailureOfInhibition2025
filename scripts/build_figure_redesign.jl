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

function checked_mechanism_evidence(evidence)
    IOChecks.verify_manifest(evidence)
    D.NullclineOverlays.checked_complete_unit(joinpath(evidence,"mechanism")) || error("mechanism evidence changed")
    record=TOML.parsefile(joinpath(evidence,"mechanism_verification","summary.toml"))
    get(record,"passed",false) || error("mechanism evidence has not passed independent verification")
    for (key,path) in (("verifier_sha256",joinpath(D.ROOT,"scripts","verify_figure_redesign.jl")),
        ("helper_sha256",joinpath(D.ROOT,"scripts","nullcline_overlays.jl")),
        ("mechanism_done_sha256",joinpath(evidence,"mechanism","done.toml")),
        ("checks_sha256",joinpath(evidence,"mechanism_verification","checks.csv")))
        record[key]==IOChecks.digest(path) || error("verified mechanism evidence changed: $key")
    end
    record
end

"""Enrich a checked historical packet with independently verified mechanism evidence."""
function with_mechanism_evidence(packet,evidence,output)
    IOChecks.verify_manifest(packet);checked_mechanism_evidence(evidence)
    ispath(output) && error("packet destination already exists")
    check_mechanism_compatibility(packet,evidence)
    cp(packet,output)
    rm(joinpath(output,"mechanism");recursive=true)
    cp(joinpath(evidence,"mechanism"),joinpath(output,"mechanism"))
    ispath(joinpath(output,"mechanism_verification")) && rm(joinpath(output,"mechanism_verification");recursive=true)
    cp(joinpath(evidence,"mechanism_verification"),joinpath(output,"mechanism_verification"))
    provenance=TOML.parsefile(joinpath(output,"packet.toml"))
    copied=provenance["copied_artifact_sha256"]
    for path in collect(keys(copied))
        startswith(path,"mechanism/") && delete!(copied,path)
    end
    for folder in ("mechanism","mechanism_verification")
        for (dir,_,names) in walkdir(joinpath(output,folder)), name in names
            relative=relpath(joinpath(dir,name),output);copied[relative]=IOChecks.digest(joinpath(output,relative))
        end
    end
    provenance["mechanism_evidence"]=Dict("parent_packet_manifest_sha256"=>IOChecks.digest(joinpath(packet,"checksums.toml")),
        "evidence_manifest_sha256"=>IOChecks.digest(joinpath(evidence,"checksums.toml")),
        "builder_sha256"=>IOChecks.digest(@__FILE__))
    S.write_record(joinpath(output,"packet.toml"),provenance)
    IOChecks.write_manifest(output);IOChecks.verify_manifest(output)
    output
end

function check_mechanism_compatibility(packet,evidence)
    if isfile(joinpath(packet,"config.toml"))
        original_cfg=TOML.parsefile(joinpath(packet,"config.toml"));replacement_cfg=TOML.parsefile(joinpath(evidence,"config.toml"))
        pop!(original_cfg,"nullcline_overlays",nothing);pop!(replacement_cfg,"nullcline_overlays",nothing)
        original_cfg==replacement_cfg || error("mechanism evidence model configuration differs from packet")
    end
    original=collect(CSV.File(joinpath(packet,"mechanism","roots.csv");stringtype=String))
    replacement=collect(CSV.File(joinpath(evidence,"mechanism","roots.csv");stringtype=String))
    length(original)==length(replacement) && all(
        a.condition==b.condition && a.root==b.root && a.role==b.role && a.stability==b.stability &&
        max(abs(a.E-b.E),abs(a.I-b.I))<1e-7 for (a,b) in zip(original,replacement)) ||
        error("mechanism evidence is incompatible with packet roots")
    true
end

function build(screen,details,output;mechanism_evidence=nothing)
    ispath(output) && error("packet destination already exists")
    metadata=TOML.parsefile(joinpath(details,"metadata.toml"))
    get(metadata,"completed",false) || error("detail study is incomplete")
    config_path=joinpath(details,"config.toml")
    IOChecks.digest(config_path)==metadata["config_sha256"] || error("recorded configuration changed")
    screen_metadata=TOML.parsefile(joinpath(screen,"metadata.toml"))
    get(screen_metadata,"config_sha256",nothing)==metadata["config_sha256"] ||
        error("screen configuration differs from detail study")
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
    mechanism_source=mechanism_evidence===nothing ? details : mechanism_evidence
    if mechanism_evidence!==nothing
        checked_mechanism_evidence(mechanism_evidence)
        check_mechanism_compatibility(details,mechanism_evidence)
    end
    for (dir,_,names) in walkdir(joinpath(mechanism_source,"mechanism")), name in names
        relative=relpath(joinpath(dir,name),mechanism_source)
        copy_evidence(mechanism_source,relative)
    end
    if mechanism_evidence!==nothing
        for name in ("summary.toml","checks.csv")
            copy_evidence(mechanism_evidence,joinpath("mechanism_verification",name))
        end
    end
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
