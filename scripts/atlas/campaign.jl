# Bounded numerical campaign. All design points are frozen before any new outcome.
const CAMPAIGN_PROTOCOL = "bounded_atlas_campaign_v1"
const CAMPAIGN_VERSION = 1
const CAMPAIGN_SELECTION_SEED = "atlas-campaign-v2"
const CAMPAIGN_COMBINED_CAP = 5_000_000_000
const CAMPAIGN_CHARTS = ("parameter", "equilibrium")
const CAMPAIGN_TERMINAL = ("completed", "unresolved", "failed", "timed_out", "interrupted", "storage_budget")

function campaign_job_bounds()
    round = Dict("inventory"=>512, "audit"=>128, "proposals"=>48,
        "nexus_sections"=>16, "geometry_probes"=>384, "component_paths"=>32)
    Dict{String,Any}("replacement"=>22, "round_count"=>3, "per_round"=>round,
        "round_maximum"=>sum(values(round)), "validation_inventory"=>256,
        "validation_recovery"=>256, "maximum_jobs"=>3894,
        "mandatory_jobs"=>2454, "recursive_followups"=>false)
end

campaign_stream(k, chart, mask) = "shell$k/$chart/mask" * lpad(string(mask),2,'0')
function campaign_selected_indices(k, chart, mask, round)
    k in 1:4 && chart in CAMPAIGN_CHARTS && mask in 0:15 && round in 1:4 ||
        throw(ArgumentError("invalid bounded campaign stream/round"))
    pool = collect((256round+1):(256round+256))
    sort!(pool; by=index->(bytes2hex(sha256("$CAMPAIGN_SELECTION_SEED|shell=$k|chart=$chart|mask=$mask|index=$index")),index))
    pool[1:(round==4 ? 2 : 4)]
end

function campaign_design(; sampler=shell_sample)
    rows = Dict{String,Any}[]
    for k in 1:4, chart in CAMPAIGN_CHARTS, mask in 0:15, round in 1:4
        indices = campaign_selected_indices(k,chart,mask,round)
        samples = [merge(sampler(k,index;chart=chart,mask=mask),
            Dict("accepted_index"=>index,"design_position"=>position)) for (position,index) in enumerate(indices)]
        push!(rows,Dict("stream"=>campaign_stream(k,chart,mask), "shell"=>k,
            "chart"=>chart,"mask"=>mask,"round"=>round,"pool_bounds"=>[256round+1,256round+256],
            "selected_indices"=>indices,"samples"=>samples,"audit_design_position"=>round==4 ? 0 : 1))
    end
    rows
end

function campaign_packet(root, role; maximum_bytes=CAMPAIGN_COMBINED_CAP)
    role in ("original_pilot","historical_targeted") || throw(ArgumentError("unsupported historical packet role"))
    root=abspath(root)
    isdir(root) && !islink(root) || throw(ArgumentError("campaign source packet must be a non-symlink directory"))
    root=realpath(root)
    inventory=targeted_inventory(root;maximum_bytes=maximum_bytes)
    packet=Dict{String,Any}("root"=>root,"role"=>role,"inventory"=>inventory,
        "inventory_hash"=>toml_digest(Dict("entries"=>inventory)),"retained_bytes"=>sum(row["bytes"] for row in inventory))
    summary_file=role=="historical_targeted" ? "targeted_summary.toml" : "pilot_summary.toml"
    for (key,relative) in (("state","state.toml"),("config","config.toml"),("summary",summary_file))
        packet[key]=TOML.parse(String(targeted_packet_file(packet,relative)))
    end
    packet["state"]["config_hash"]==toml_digest(packet["config"]) || throw(ArgumentError("historical packet configuration hash mismatch"))
    for (relative,digest) in get(packet["state"],"source_hashes",Dict())
        bytes2hex(sha256(targeted_packet_file(packet,joinpath("source",relative))))==digest || throw(ArgumentError("historical executable source hash mismatch"))
    end
    for (index,job) in enumerate(packet["state"]["jobs"])
        job["id"]=="job-"*lpad(string(index),6,'0') && job["payload_hash"]==toml_digest(job["payload"]) || throw(ArgumentError("historical canonical job binding mismatch"))
        if haskey(job,"result_hash")
            bytes=targeted_packet_file(packet,job["result_file"])
            bytes2hex(sha256(bytes))==job["result_hash"] || throw(ArgumentError("historical result hash mismatch"))
            result=TOML.parse(String(bytes))
            isequal(get(result,"executed_job",nothing),merge(job["payload"],Dict("id"=>job["id"]))) || throw(ArgumentError("historical executed job binding mismatch"))
        elseif job["status"]=="completed"
            throw(ArgumentError("historical completed job lacks bound result evidence"))
        end
    end
    # The failed second-pilot control gate is historical evidence and is retained
    # verbatim; it is deliberately not required to have completed its protocol.
    role=="original_pilot" && !get(get(packet["state"],"pilot",Dict()),"protocol_complete",false) &&
        throw(ArgumentError("original pilot must have completed its finite protocol"))
    role=="historical_targeted" && get(packet["state"],"protocol","")!=TARGETED_PROTOCOL &&
        throw(ArgumentError("historical targeted packet has the wrong protocol"))
    packet
end

function campaign_verify_packet(packet)
    inventory=targeted_inventory(packet["root"];maximum_bytes=packet["retained_bytes"])
    isequal(inventory,packet["inventory"]) && toml_digest(Dict("entries"=>inventory))==packet["inventory_hash"] ||
        throw(ArgumentError("immutable historical packet changed"))
    nothing
end

function campaign_capacity_forecast(cap, retained; measured_result_bytes=0, remaining_jobs=3894)
    cap isa Integer && !(cap isa Bool) && 0<cap<=CAMPAIGN_COMBINED_CAP || throw(ArgumentError("combined campaign cap must be in 1:5000000000"))
    retained isa Integer && retained>=0 || throw(ArgumentError("invalid historical retained bytes"))
    measured_result_bytes isa Integer && measured_result_bytes>=0 || throw(ArgumentError("invalid measured result bytes"))
    remaining_jobs isa Integer && remaining_jobs>=0 || throw(ArgumentError("invalid remaining job count"))
    residual=Base.checked_sub(Int(cap),Int(retained)); policy=storage_policy(residual)
    reserve=policy["checkpoint_reserve_bytes"]+policy["report_reserve_bytes"]+2policy["result_reserve_bytes"]+2policy["monitor_headroom_bytes"]
    forecast=Base.checked_mul(Int(measured_result_bytes),Int(remaining_jobs))
    Dict{String,Any}("combined_cap_bytes"=>Int(cap),"historical_retained_bytes"=>Int(retained),
        "new_packet_cap_bytes"=>residual,"dispatch_reserves_bytes"=>reserve,
        "measured_result_bytes"=>Int(measured_result_bytes),"remaining_jobs"=>Int(remaining_jobs),
        "projected_result_bytes"=>forecast,"projection_available"=>measured_result_bytes>0,
        "projected_capacity_sufficient"=>measured_result_bytes>0 && Base.checked_add(forecast,reserve)<=residual,
        "projection_scope"=>"advisory measured-result forecast; controlled-write reservations enforce the cap")
end

function campaign_path_contains(root,path)
    relative=relpath(path,root)
    relative=="." || first(splitpath(relative))!=".."
end

function campaign_resolved_destination(output)
    ancestor=abspath(output);suffix=String[]
    while !ispath(ancestor)
        pushfirst!(suffix,basename(ancestor));parent=dirname(ancestor)
        parent==ancestor && throw(ArgumentError("no existing output ancestor"))
        ancestor=parent
    end
    joinpath(realpath(ancestor),suffix...)
end

function campaign_validate_configuration(config,policy)
    baseline=atlas_config()
    allowed=Set(("warmup_seconds","job_timeout_seconds","pilot_guard_seconds","pilot_max_jobs","campaign_calibration_sources"))
    for (key,value) in config
        key in allowed && continue
        if key=="science"
            for (science_key,science_value) in value
                haskey(baseline["science"],science_key) || throw(ArgumentError("unsupported campaign scientific override: $science_key"))
                prescribed=science_key in ("balance_atol","event_atol") ? policy[science_key] : science_key=="seed_grid" ? 9 : baseline["science"][science_key]
                isequal(science_value,baseline["science"][science_key]) || isequal(science_value,prescribed) || throw(ArgumentError("campaign scientific limits cannot be overridden: $science_key"))
            end
        else
            haskey(baseline,key) && isequal(value,baseline[key]) || throw(ArgumentError("unsupported campaign configuration override: $key"))
        end
    end
    for key in ("warmup_seconds","job_timeout_seconds","pilot_guard_seconds")
        haskey(config,key) || continue
        value=config[key]
        value isa Real && !(value isa Bool) && isfinite(value) && value>0 || throw(ArgumentError("invalid campaign operational guard: $key"))
    end
    haskey(config,"pilot_max_jobs") && config["pilot_max_jobs"]!=3894 && throw(ArgumentError("campaign pilot_max_jobs is fixed at its finite ceiling 3894"))
    science=merge(deepcopy(baseline["science"]),Dict("seed_grid"=>9,"balance_atol"=>policy["balance_atol"],"event_atol"=>policy["event_atol"]))
    merge(config,Dict("science"=>science,"numerical_policy"=>deepcopy(policy),"pilot_max_jobs"=>3894))
end

function campaign_required_calibration_controls(calibration)
    rows=filter(row->get(row,"name","")=="failed_targeted_cusp_bias_reproduced",get(calibration,"controls",[]))
    length(rows)==1 && get(only(rows),"passed",false)===true || throw(ArgumentError("campaign calibration requires one passing failed_targeted_cusp_bias_reproduced control"))
    nothing
end

function campaign_calibration_review(policy)
    root=policy["calibration_root"]
    campaign_required_calibration_controls(TOML.parsefile(safe_atlas_path(root,"calibration.toml")))
    policy_path=safe_atlas_path(root,"policy.toml")
    isfile(policy_path) && !islink(policy_path) || throw(ArgumentError("calibrated policy must be a regular policy.toml file under its calibration root"))
    isequal(TOML.parsefile(policy_path),policy) || throw(ArgumentError("policy argument differs from its frozen calibration file"))
    path=safe_atlas_path(root,"review.toml")
    isfile(path) && !islink(path) || throw(ArgumentError("campaign requires an independent numerical review artifact"))
    review=TOML.parsefile(path)
    get(review,"status","")=="passed" && get(review,"reviewer","") isa AbstractString && !isempty(review["reviewer"]) || throw(ArgumentError("numerical review has not passed"))
    get(review,"policy_hash","")==toml_digest(policy) || throw(ArgumentError("numerical review does not bind this exact policy"))
    entries=filter(row->!(row["path"] in (".","review.toml")),targeted_inventory(root))
    get(review,"calibration_inventory_hash","")==toml_digest(Dict("entries"=>entries)) || throw(ArgumentError("numerical review does not bind this exact calibration packet"))
    Dict{String,Any}("path"=>path,"sha256"=>bytes2hex(sha256(read(path))),"record"=>review)
end

function campaign_calibration_roots(config,policy)
    roots=policy["calibration_sources"]
    isequal(get(config,"campaign_calibration_sources",roots),roots) || throw(ArgumentError("campaign calibration roots must match the reviewed numerical policy"))
    deepcopy(roots)
end

function campaign_prepare(config,pilot_root,targeted_root; policy,source_hashes,combined_cap=CAMPAIGN_COMBINED_CAP)
    policy isa AbstractDict && !isempty(policy) || throw(ArgumentError("a reviewed numerical policy must be supplied"))
    get(policy,"readiness",false)===true || throw(ArgumentError("campaign numerical calibration is not ready"))
    atlas_validate_numerical_policy(policy)
    numerical_review=campaign_calibration_review(policy)
    source_hashes isa AbstractDict && !isempty(source_hashes) || throw(ArgumentError("campaign executable source hashes are required"))
    config=campaign_validate_configuration(config,policy)
    config["campaign_calibration_sources"]=campaign_calibration_roots(config,policy)
    first_packet=targeted_source_packet(pilot_root;maximum_bytes=combined_cap)
    first_packet["role"]="original_pilot"
    second_packet=campaign_packet(targeted_root,"historical_targeted";maximum_bytes=combined_cap)
    verify_targeted_inputs(second_packet["root"],second_packet["state"],second_packet["config"])
    calibration_packets=Dict{String,Any}[]
    for root in get(config,"campaign_calibration_sources",String[])
        isdir(root) && !islink(root) || throw(ArgumentError("calibration root must be an existing non-symlink directory"))
        root=realpath(root);inventory=targeted_inventory(root;maximum_bytes=combined_cap)
        push!(calibration_packets,Dict("root"=>root,"role"=>"numerical_calibration","inventory"=>inventory,
            "inventory_hash"=>toml_digest(Dict("entries"=>inventory)),"retained_bytes"=>sum(row["bytes"] for row in inventory)))
    end
    isempty(calibration_packets) && throw(ArgumentError("campaign requires a frozen numerical calibration root"))
    roots=[packet["root"] for packet in vcat([first_packet,second_packet],calibration_packets)]
    for (index,root) in enumerate(roots), other in roots[index+1:end]
        (campaign_path_contains(root,other) || campaign_path_contains(other,root)) && throw(ArgumentError("campaign source roots must be disjoint"))
    end
    retained=foldl(Base.checked_add,(packet["retained_bytes"] for packet in vcat([first_packet,second_packet],calibration_packets));init=0)
    capacity=campaign_capacity_forecast(combined_cap,retained)
    replacements=build_targeted_jobs(first_packet,config)
    length(replacements)==22 && all(get(replacements[i],"target_role","")=="control" for i in 1:3) || throw(ArgumentError("replacement protocol must begin with its three controls"))
    # Historical config_overrides are retained as provenance, but execution uses
    # the reviewed campaign policy rather than silently reinstating old numerics.
    for (index,payload) in enumerate(replacements)
        overrides=deepcopy(get(payload,"config_overrides",Dict{String,Any}()))
        payload["historical_config_overrides"]=deepcopy(overrides)
        for key in ("balance_atol","event_atol")
            haskey(policy,key) || throw(ArgumentError("reviewed campaign policy lacks $key"))
            overrides[key]=min(get(overrides,key,policy[key]),get(get(get(policy,"stage_caps",Dict()),"targeted",Dict()),key,policy[key]),policy[key])
        end
        payload["config_overrides"]=overrides
        payload["campaign_stage"]=index<=3 ? "replacement_controls" : "replacement"
        payload["campaign_role"]="replacement";payload["campaign_round"]=0
    end
    packet_metadata(packet)=merge(Dict(key=>packet[key] for key in ("root","role","inventory","inventory_hash","retained_bytes")),
        Dict("result_kinds"=>Dict(job["result_file"]=>job["kind"] for job in packet["state"]["jobs"] if haskey(job,"result_hash"))))
    manifest=Dict{String,Any}("schema_version"=>CAMPAIGN_VERSION,"protocol"=>CAMPAIGN_PROTOCOL,
        "completeness"=>"CompletenessNotCertified","payload_schema_version"=>2,"shells"=>[1,2,3,4],"charts"=>collect(CAMPAIGN_CHARTS),"masks"=>collect(0:15),
        "selection_seed"=>CAMPAIGN_SELECTION_SEED,"selection_algorithm"=>"ascending SHA256 hex then accepted index; independent per shell/chart/mask; first 4 training and first 2 validation",
        "selection_encoding"=>"atlas-campaign-v2|shell=K|chart=CHART|mask=MASK|index=INDEX",
        "design"=>campaign_design(),"bounds"=>campaign_job_bounds(),"replacement_jobs"=>replacements,
        "source_packets"=>[packet_metadata(first_packet),packet_metadata(second_packet)],"calibration_packets"=>calibration_packets,
        "numerical_review"=>numerical_review,"numerical_policy"=>deepcopy(policy),"numerical_policy_hash"=>toml_digest(policy),"source_hashes"=>deepcopy(source_hashes),
        "capacity"=>capacity,"effective_science"=>deepcopy(config["science"]),"validation_rule"=>"Freeze component representatives and all parameter-only routes before any validation result; validation never changes the base pool",
        "scope"=>"Bounded finite numerical atlas on shells 1 through 4; no global certification or absence claims")
    foreach(campaign_verify_packet,vcat([first_packet,second_packet],calibration_packets))
    prepared=merge(config,Dict("pilot_max_jobs"=>3894,"protocol"=>CAMPAIGN_PROTOCOL,"campaign_manifest"=>manifest,"campaign_manifest_hash"=>toml_digest(manifest)))
    prepared,capacity["new_packet_cap_bytes"]
end

campaign_initial_jobs(config)=deepcopy(config["campaign_manifest"]["replacement_jobs"])
campaign_role(job)=get(job["payload"],"campaign_role","")
campaign_stage(job)=get(job["payload"],"campaign_stage","")

function campaign_sample_jobs(manifest,round; audit=false,validation=false)
    jobs=Dict{String,Any}[]
    for row in manifest["design"]
        row["round"]==round || continue
        samples=audit ? row["samples"][1:1] : row["samples"]
        for sample in samples
            role=validation ? "validation_inventory" : audit ? "audit" : "inventory"
            stage=validation ? role : "round$(round)_$role"
            kind=validation ? "campaign_validation" : audit ? "campaign_audit" : "discovery"
            payload=merge(deepcopy(sample),Dict("kind"=>kind,"queue"=>"discovery","campaign_stage"=>stage,"campaign_role"=>role,
                "campaign_round"=>round,"campaign_stream"=>row["stream"],"campaign_shell"=>row["shell"],"campaign_chart"=>row["chart"],
                "campaign_mask"=>row["mask"],"seed_grid"=>audit ? 13 : 9,"validation_no_followups"=>validation))
            push!(jobs,payload)
        end
    end
    jobs
end

# Execution inputs are exact scientific projections of hash-bound raw evidence.
# Solver attempts and other diagnostics remain once in the retained result file.
function campaign_evidence_reference(job,key,index,record)
    haskey(job,"result_hash") || throw(ArgumentError("campaign source lacks bound raw evidence"))
    Dict{String,Any}("source_job"=>job["id"],"result_file"=>job["result_file"],"result_hash"=>job["result_hash"],
        "evidence_key"=>key,"record_index"=>index,"record_id"=>get(record,"id",get(record,"point_id","")))
end

function campaign_point_input(point;reference=get(point,"source_evidence_reference",nothing),retain_root_records=false)
    input=Dict{String,Any}("parameters"=>deepcopy(point["parameters"]),"signature"=>point["signature"],
        "roots"=>retain_root_records ? deepcopy(point["roots"]) : [Dict("state"=>deepcopy(root["state"])) for root in point["roots"]])
    for key in ("id","point_id","source_job")
        haskey(point,key) && (input[key]=point[key])
    end
    reference===nothing || (input["source_evidence_reference"]=deepcopy(reference))
    input
end

function campaign_witness_input(witness)
    input=Dict{String,Any}(key=>deepcopy(witness[key]) for key in
        ("id","kind","classification","nexus_type","status","validated","source_job","source_evidence_reference") if haskey(witness,key))
    input
end

function campaign_audit_jobs(output,state,manifest,round)
    payloads=campaign_sample_jobs(manifest,round;audit=true)
    for payload in payloads
        source=only(filter(state["jobs"]) do job
            p=job["payload"]
            campaign_role(job)=="inventory" && p["campaign_round"]==round &&
                p["campaign_stream"]==payload["campaign_stream"] && p["accepted_index"]==payload["accepted_index"]
        end)
        payload["training_source_job"]=source["id"]
        payload["training_status"]=source["status"]
        result=pilot_result(output,source)
        points=result===nothing ? [] : get(get(result,"evidence",Dict()),"points",[])
        if source["status"]=="completed" && length(points)==1
            payload["training_point"]=campaign_point_input(only(points);reference=campaign_evidence_reference(source,"points",1,only(points)))
        end
    end
    payloads
end

# Signatures alone never establish graph edges: the adapter must retain clean
# forward root tracks and an independent path inventories with paired endpoint audits.
function campaign_connection_paths(records)
    filter(records["paths"]) do path
        get(path,"forward_tracking_unambiguous",false)===true &&
            get(path,"audit_agreement",false)===true && isempty(get(path,"tracking_guard_failures",["unreported"]))
    end
end

function campaign_records(output,state; before_validation=true)
    records=Dict(key=>Dict{String,Any}[] for key in ("points","paths","events","cycles","sections","audit_exclusions"))
    for job in sort(state["jobs"];by=j->j["id"])
        role=campaign_role(job)
        role in ("replacement","") && continue
        before_validation && startswith(role,"validation") && continue
        result=pilot_result(output,job)
        if role=="audit"
            rows=result===nothing ? [] : get(get(result,"evidence",Dict()),"audits",[])
            passed=job["status"]=="completed" && result!==nothing && get(result,"status","")=="completed" &&
                length(rows)==1 && get(only(rows),"status","")=="agreement_observed"
            passed || push!(records["audit_exclusions"],Dict("source_job"=>get(job["payload"],"training_source_job",""),
                "reason"=>"direct_grid9_grid13_audit_disagreement_or_unresolved"))
        end
        job["status"]=="completed" && result!==nothing || continue
        get(result,"status","")=="completed" || continue
        for key in ("points","paths","events","cycles","sections"), (index,raw) in enumerate(get(get(result,"evidence",Dict()),key,[]))
            raw isa AbstractDict || throw(ArgumentError("campaign evidence must contain records"))
            push!(records[key],merge(deepcopy(raw),Dict("source_job"=>job["id"],
                "source_evidence_reference"=>campaign_evidence_reference(job,key,index,raw),"campaign_role"=>role,
                "campaign_shell"=>get(job["payload"],"campaign_shell",0),"campaign_chart"=>get(job["payload"],"campaign_chart",""))))
        end
    end
    records
end

function campaign_proposal_jobs(output,state,round)
    chosen=Dict{String,Any}[];opportunities=Dict{String,Any}[]
    exclusions=Set(row["source_job"] for row in campaign_records(output,state)["audit_exclusions"])
    sources=sort(filter(j->campaign_role(j)=="inventory" && get(j["payload"],"campaign_round",0)==round,state["jobs"]);by=j->j["id"])
    for k in 1:4, chart in CAMPAIGN_CHARTS, kind in ("continuation","nexus","periodic")
        count=0
        for source in sources
            p=source["payload"]
            p["campaign_shell"]==k && p["campaign_chart"]==chart && source["status"]=="completed" && !(source["id"] in exclusions) || continue
            result=pilot_result(output,source);result===nothing && continue
            get(result,"status","")=="completed" || continue
            for (order,proposal) in enumerate(get(result,"followups",[]))
                get(proposal,"kind","")==kind || continue
                count==2 && break
                normalized=atlas_normalize_job(deepcopy(proposal))
                push!(chosen,merge(normalized,Dict("campaign_stage"=>"round$(round)_proposals","campaign_role"=>"proposal","campaign_round"=>round,
                    "campaign_shell"=>k,"campaign_chart"=>chart,"campaign_source_job"=>source["id"],"campaign_proposal_order"=>order,"campaign_followup_depth"=>1)))
                count+=1
            end
            count==2 && break
        end
        push!(opportunities,Dict("round"=>round,"shell"=>k,"chart"=>chart,"kind"=>kind,"selected"=>count,"quota"=>2,
            "excluded_audit_source_ids"=>[source["id"] for source in sources if source["id"] in exclusions && source["payload"]["campaign_shell"]==k && source["payload"]["campaign_chart"]==chart],
            "status"=>count==2 ? "quota_filled" : "eligible_proposals_exhausted"))
    end
    chosen,opportunities
end

function campaign_nexus_jobs(output,state,round)
    jobs=Dict{String,Any}[]
    for source in sort(state["jobs"];by=j->j["id"])
        p=source["payload"]
        campaign_role(source)=="proposal" && p["campaign_round"]==round && source["kind"]=="nexus" && source["status"]=="completed" || continue
        result=pilot_result(output,source);result===nothing && continue
        get(result,"status","")=="completed" || continue
        events=get(get(result,"evidence",Dict()),"events",[])
        length(events)==1 && get(only(events),"validated",false)===true || continue
        sections=filter(q->get(q,"kind","")=="section",get(result,"followups",[]))
        length(sections)==1 || throw(ArgumentError("validated nexus must declare exactly one bounded section"))
        section=atlas_normalize_job(deepcopy(only(sections)))
        common=Dict("campaign_stage"=>"round$(round)_nexus_sections","campaign_round"=>round,"campaign_shell"=>p["campaign_shell"],
            "campaign_chart"=>p["campaign_chart"],"campaign_source_job"=>source["id"],"campaign_followup_depth"=>2)
        push!(jobs,merge(section,common,Dict("campaign_role"=>"nexus_section")))
        for coordinate in setdiff(collect(ATLAS_COORDINATES[1:7]), section["parameter_pair"]), offset in (-0.01,-0.005,0.005,0.01)
            push!(jobs,merge(deepcopy(section),common,Dict("kind"=>"campaign_geometry_probe","campaign_role"=>"geometry_probe",
                "normalized_transverse_offset"=>offset,"transverse_coordinate"=>coordinate,
                "nexus_evidence"=>merge(Dict{String,Any}(key=>deepcopy(only(events)[key]) for key in ("validated","id") if haskey(only(events),key)),
                    Dict("source_evidence_reference"=>campaign_evidence_reference(source,"events",1,only(events)))))))
        end
    end
    jobs
end

function campaign_eligible_points(records)
    excluded=Set(row["source_job"] for row in get(records,"audit_exclusions",[]))
    points=filter(records["points"]) do point
        get(point,"campaign_role","") in ("inventory","component_path") && get(point,"audit_only",false)!==true && !(get(point,"source_job","") in excluded) &&
            !isempty(get(point,"roots",[])) && !occursin("unresolved",lowercase(get(point,"signature","unresolved")))
    end
    ids=getindex.(points,"id")
    length(unique(ids))==length(ids) || throw(ArgumentError("duplicate campaign point IDs"))
    sort(points;by=p->p["id"])
end

function campaign_path_jobs(output,state,round)
    records=campaign_records(output,state); points=campaign_eligible_points(records)
    components=atlas_region_components(points,campaign_connection_paths(records))
    lookup=Dict(p["id"]=>p for p in points)
    component_of=Dict(id=>first(component) for component in components for id in component)
    examined=Set{Tuple{String,String}}()
    for job in state["jobs"]
        campaign_role(job)=="component_path" || continue
        p=job["payload"];push!(examined,Tuple(sort([p["from"]["id"],p["to"]["id"]])))
    end
    jobs=Dict{String,Any}[];opportunities=Dict{String,Any}[]
    for k in 1:4, chart in CAMPAIGN_CHARTS, same in (true,false)
        local_points=filter(p->p["campaign_shell"]==k && p["campaign_chart"]==chart,points)
        nearest=Dict{Tuple{String,String},Tuple{Float64,String,String}}()
        for i in eachindex(local_points), j in (i+1):length(local_points)
            a,b=local_points[i],local_points[j];pair=Tuple(sort([a["id"],b["id"]]))
            pair in examined && continue
            component_of[a["id"]]==component_of[b["id"]] && continue
            (a["signature"]==b["signature"])==same || continue
            component_pair=Tuple(sort([component_of[a["id"]],component_of[b["id"]]]))
            candidate=(pilot_distance(a["parameters"],b["parameters"]),pair...)
            (!haskey(nearest,component_pair) || candidate<nearest[component_pair]) && (nearest[component_pair]=candidate)
        end
        candidates=sort!(collect(values(nearest)))
        selected=first(candidates,min(2,length(candidates)))
        for (_,a,b) in selected
            push!(examined,(a,b))
            push!(jobs,Dict("kind"=>"campaign_path","queue"=>"events","campaign_stage"=>"round$(round)_component_paths","campaign_role"=>"component_path",
                "campaign_round"=>round,"campaign_shell"=>k,"campaign_chart"=>chart,"from"=>campaign_point_input(lookup[a]),"to"=>campaign_point_input(lookup[b]),"same_signature_pair"=>same,
                "component_pair_ids"=>sort([component_of[a],component_of[b]]),
                "selection_rule"=>"Minimum pilot_distance over eligible endpoints in distinct explicit components, then stable endpoint IDs; one minimum per component pair and previously examined endpoint pairs excluded","path_samples"=>129))
        end
        push!(opportunities,Dict("round"=>round,"shell"=>k,"chart"=>chart,"same_signature"=>same,"selected"=>length(selected),"quota"=>2,
            "status"=>length(selected)==2 ? "quota_filled" : "eligible_pairs_exhausted"))
    end
    jobs,opportunities
end

function campaign_freeze_bases!(output,state,manifest)
    metadata=state["campaign"]
    get(metadata,"bases_frozen",false) && return metadata["base_selection"]
    records=campaign_records(output,state);points=campaign_eligible_points(records)
    lookup=Dict(p["id"]=>p for p in points);components=atlas_region_components(points,campaign_connection_paths(records))
    entries=[Dict("point_id"=>first(c),"members"=>c,"parameters"=>lookup[first(c)]["parameters"],"roots"=>[Dict("state"=>deepcopy(root["state"])) for root in lookup[first(c)]["roots"]],
        "source_evidence_reference"=>deepcopy(lookup[first(c)]["source_evidence_reference"]),"signature"=>lookup[first(c)]["signature"],"source_job"=>lookup[first(c)]["source_job"]) for c in components]
    witnesses=[campaign_witness_input(row) for key in ("events","cycles","sections") for row in records[key] if get(row,"validated",false)===true]
    routes=Dict{String,Any}[]
    for payload in campaign_sample_jobs(manifest,4;validation=true)
        route=Dict{String,Any}("stream"=>payload["campaign_stream"],"accepted_index"=>payload["accepted_index"],"params"=>payload["params"],
            "selection_rule"=>"Nearest frozen explicit-component base by parameter-only pilot_distance, then point_id; no validation outcomes")
        if isempty(entries)
            route["status"]="unresolved_no_eligible_base"
        else
            base=first(sort(entries;by=b->(pilot_distance(b["parameters"],payload["params"]),b["point_id"])))
            route["base_point_id"]=base["point_id"];route["status"]="route_frozen"
        end
        push!(routes,route)
    end
    selection=Dict{String,Any}("entries"=>entries,"event_witnesses"=>witnesses,"routes"=>routes,"frozen_before_validation"=>true,
        "component_rule"=>"Stable-ID representative per explicit examined signature-preserving sampled-path component; equal signatures alone do not connect points",
        "completeness"=>"CompletenessNotCertified")
    metadata["base_selection"]=selection;metadata["base_selection_hash"]=toml_digest(selection);metadata["bases_frozen"]=true
    selection
end

function campaign_recovery_jobs(output,state)
    metadata=state["campaign"];selection=metadata["base_selection"]
    toml_digest(selection)==metadata["base_selection_hash"] || throw(ArgumentError("frozen campaign routes/bases changed"))
    jobs=Dict{String,Any}[]
    for source in sort(filter(j->campaign_role(j)=="validation_inventory",state["jobs"]);by=j->j["id"])
        p=source["payload"]
        route=only(filter(r->r["stream"]==p["campaign_stream"] && r["accepted_index"]==p["accepted_index"],selection["routes"]))
        payload=Dict{String,Any}("kind"=>"campaign_recovery","queue"=>"events","campaign_stage"=>"validation_recovery","campaign_role"=>"validation_recovery",
            "campaign_round"=>4,"campaign_stream"=>p["campaign_stream"],"campaign_shell"=>p["campaign_shell"],"campaign_chart"=>p["campaign_chart"],
            "campaign_target_job"=>source["id"],"route"=>deepcopy(route),"base_selection_hash"=>metadata["base_selection_hash"],"target_parameters"=>p["params"],"path_samples"=>129)
        haskey(route,"base_point_id") && (payload["base"]=campaign_point_input(only(filter(b->b["point_id"]==route["base_point_id"],selection["entries"]))))
        result=pilot_result(output,source)
        payload["validation_inventory_status"]=source["status"]
        if result!==nothing
            points=get(get(result,"evidence",Dict()),"points",[])
            payload["validation_evidence"]=Dict("points"=>[campaign_point_input(point;
                reference=campaign_evidence_reference(source,"points",index,point),retain_root_records=true) for (index,point) in enumerate(points)])
        end
        push!(jobs,payload)
    end
    jobs
end

function campaign_phases()
    phases=["replacement_controls","replacement"]
    for round in 1:3
        append!(phases,["round$(round)_$role" for role in ("inventory","audit","proposals","nexus_sections","component_paths")])
    end
    append!(phases,["validation_inventory","validation_recovery"])
end

function campaign_phase_payloads(output,state,manifest,phase)
    if phase in ("replacement_controls","replacement")
        return Any[] # The engine registers the immutable 22 replacement jobs initially.
    elseif phase=="validation_inventory"
        campaign_freeze_bases!(output,state,manifest)
        return campaign_sample_jobs(manifest,4;validation=true)
    elseif phase=="validation_recovery"
        return campaign_recovery_jobs(output,state)
    end
    round=parse(Int,split(phase,'_')[1][6:end]);role=join(split(phase,'_')[2:end],"_")
    if role=="inventory"
        campaign_sample_jobs(manifest,round)
    elseif role=="audit"
        campaign_audit_jobs(output,state,manifest,round)
    elseif role=="proposals"
        payloads,slots=campaign_proposal_jobs(output,state,round)
        state["campaign"][phase*"_opportunities"]=slots
        payloads
    elseif role=="nexus_sections"
        campaign_nexus_jobs(output,state,round)
    else
        payloads,slots=campaign_path_jobs(output,state,round)
        state["campaign"][phase*"_opportunities"]=slots
        payloads
    end
end

function campaign_remaining_kind_bounds(manifest,state)
    bounds=Dict("discovery"=>1536,"campaign_audit"=>384,"continuation"=>48,"nexus"=>48,"periodic"=>48,
        "section"=>48,"campaign_geometry_probe"=>1152,"campaign_path"=>96,"campaign_validation"=>256,"campaign_recovery"=>256)
    for payload in manifest["replacement_jobs"]
        kind=payload["kind"];bounds[kind]=get(bounds,kind,0)+1
    end
    for job in state["jobs"]
        job["status"] in CAMPAIGN_TERMINAL || continue
        kind=job["kind"];bounds[kind]=max(0,get(bounds,kind,0)-1)
    end
    # Empty conditional slots cease to be future work only after their own
    # registration barrier and every registered dependent job have finished.
    # A completed proposal phase does not close the later nexus/probe phase.
    metadata=get(state,"campaign",Dict())
    for round in 1:3
        quotas=("proposals"=>Dict("continuation"=>16,"nexus"=>16,"periodic"=>16),
            "nexus_sections"=>Dict("section"=>16,"campaign_geometry_probe"=>384),
            "component_paths"=>Dict("campaign_path"=>32))
        for (role,kinds) in quotas
            phase="round$(round)_$role"
            get(metadata,phase*"_registered",false)===true || continue
            jobs=filter(job->campaign_stage(job)==phase,state["jobs"])
            all(job->job["status"] in CAMPAIGN_TERMINAL,jobs) || continue
            for (kind,quota) in kinds
                unused=quota-count(job->job["kind"]==kind,jobs)
                unused>=0 || throw(ArgumentError("registered campaign stage exceeds its kind quota"))
                bounds[kind]=max(0,bounds[kind]-unused)
            end
        end
    end
    bounds
end

function campaign_stage_forecast!(output,state,config,phase)
    manifest=config["campaign_manifest"]
    isfile(joinpath(output,".storage.toml")) || return nothing # pure scheduler tests have no filesystem budget
    budget=storage_budget(output)
    budget===nothing && throw(ArgumentError("campaign storage policy is missing"))
    observation=storage_snapshot!(budget)
    checkpoint_bytes=storage_serialized_size(state)
    if checkpoint_bytes>budget.checkpoint
        state["campaign"]["storage_stop"]=true;state["campaign"]["stop_reason"]="storage_budget"
        throw(StorageBudgetExceeded("storage_budget: current serialized checkpoint exceeds the versioned campaign allowance before $phase"))
    end
    measured=Dict{String,Int}();origins=Dict{String,String}()
    for packet in manifest["source_packets"], entry in packet["inventory"]
        get(entry,"kind","")=="file" || continue
        kind=get(get(packet,"result_kinds",Dict()),entry["path"],nothing)
        kind===nothing && continue
        measured[kind]=max(get(measured,kind,0),entry["bytes"]);origins[kind]="historical_same_kind"
    end
    current=Dict{String,Int}()
    for job in state["jobs"]
        haskey(job,"result_hash") || continue
        path=safe_atlas_path(output,job["result_file"]);bytes=read(path)
        bytes2hex(sha256(bytes))==job["result_hash"] || throw(ArgumentError("campaign result changed during capacity forecast"))
        kind=job["kind"];current[kind]=max(get(current,kind,0),length(bytes))
    end
    for (kind,bytes) in current
        measured[kind]=bytes;origins[kind]="current_campaign_serialization"
    end
    remaining=campaign_remaining_kind_bounds(manifest,state)
    rows=Dict{String,Any}[];projected=0;unknown=0
    for kind in sort!(collect(keys(remaining)))
        count=remaining[kind];bytes=get(measured,kind,0)
        projection=Base.checked_mul(bytes,count);projected=Base.checked_add(projected,projection)
        bytes==0 && (unknown+=count)
        push!(rows,Dict("kind"=>kind,"remaining_job_upper_bound"=>count,"measured_max_result_bytes"=>bytes,
            "estimate_source"=>get(origins,kind,"unknown_unobserved_kind"),"projected_bytes"=>projection))
    end
    reserve=budget.checkpoint+budget.report+2budget.result+2budget.headroom
    required=Base.checked_add(Base.checked_add(observation["retained_bytes"],projected),reserve)
    forecast=Dict{String,Any}("stage"=>phase,"retained_new_packet_bytes"=>observation["retained_bytes"],"kinds"=>rows,
        "serialized_checkpoint_bytes"=>checkpoint_bytes,"checkpoint_reserve_bytes"=>budget.checkpoint,
        "checkpoint_remaining_bytes"=>budget.checkpoint-checkpoint_bytes,
        "checkpoint_forecast_scope"=>"Exact current TOML snapshot; versioned reserve also covers the declared fully populated max-root stress fixture with at least 25 percent measured margin, not arbitrary diagnostic growth",
        "remaining_job_upper_bound"=>sum(values(remaining)),"unknown_kind_job_count"=>unknown,"projected_result_bytes"=>projected,"dispatch_reserves_bytes"=>reserve,
        "required_known_new_packet_bytes"=>required,"new_packet_cap_bytes"=>budget.cap,
        "measurement_scope"=>"Maximum actual saved result bytes per scientific kind, with same-kind historical fallback; unobserved kinds remain explicit unknowns",
        "projection_scope"=>"Measured kindwise forecast is not a future-size guarantee; unknown kinds are advisory and controlled-write guards always enforce the cap",
        "capacity_sufficient_for_known_estimates"=>required<=budget.cap)
    state["campaign"][phase*"_capacity_forecast"]=forecast
    if required>budget.cap
        state["campaign"]["storage_stop"]=true
        state["campaign"]["stop_reason"]="storage_budget"
        throw(StorageBudgetExceeded("storage_budget: measured kindwise result forecast plus remaining-job upper bounds and reserves exceeds campaign capacity before $phase"))
    end
    forecast
end

function campaign_storage_stopped(state)
    metadata=get(state,"campaign",Dict())
    get(metadata,"storage_stop",false)===true || get(state,"status","")=="storage_budget" ||
        occursin("storage_budget",get(state,"stop_reason","")) || startswith(get(state,"stop_reason",""),"checkpoint_capacity_") ||
        any(job->get(job,"status","")=="storage_budget",get(state,"jobs",[]))
end

function campaign_next_job!(output,state,config)
    manifest=config["campaign_manifest"]
    metadata=get!(state,"campaign",Dict{String,Any}("manifest_hash"=>config["campaign_manifest_hash"],"protocol_complete"=>false,"bases_frozen"=>false))
    metadata["manifest_hash"]==config["campaign_manifest_hash"] || throw(ArgumentError("campaign state manifest binding changed"))
    if campaign_storage_stopped(state)
        metadata["storage_stop"]=true;metadata["protocol_complete"]=false
        metadata["stage"]="storage_frontier";metadata["stop_reason"]="storage_budget"
        return nothing
    end
    if get(metadata,"bases_frozen",false)
        toml_digest(metadata["base_selection"])==metadata["base_selection_hash"] || throw(ArgumentError("frozen campaign base selection changed"))
    end
    for phase in campaign_phases()
        metadata["stage"]=phase
        key=phase*"_registered"
        if !get(metadata,key,false)
            # Verify the full immutable design at each stage boundary. Resume
            # verification independently checks its persisted binding.
            toml_digest(manifest)==config["campaign_manifest_hash"] || throw(ArgumentError("campaign manifest changed"))
            campaign_stage_forecast!(output,state,config,phase)
            payloads=campaign_phase_payloads(output,state,manifest,phase)
            for payload in payloads
                register_atlas_job!(state,payload)
            end
            length(state["jobs"])<=manifest["bounds"]["maximum_jobs"] || throw(ArgumentError("campaign exceeded its finite job bound"))
            metadata[key]=true;metadata[phase*"_registered_count"]=length(payloads)
        end
        jobs=sort(filter(job->campaign_stage(job)==phase,state["jobs"]);by=j->j["id"])
        pending=findfirst(job->job["status"]=="pending",jobs)
        pending===nothing || return jobs[pending]
        any(job->job["status"]=="running",jobs) && return nothing
        all(job->job["status"] in CAMPAIGN_TERMINAL,jobs) || throw(ArgumentError("unknown campaign job status"))
        if phase=="replacement_controls"
            ready=campaign_controls_ready(output,jobs)
            metadata["controls_ready"]=ready
            if !ready
                metadata["stage"]="blocked";metadata["stop_reason"]="control_validation_failed";return nothing
            end
        end
    end
    metadata["stage"]="complete";metadata["protocol_complete"]=true;metadata["stop_reason"]="protocol_complete"
    metadata["unresolved_jobs"]=count(job->job["status"]!="completed",state["jobs"])
    nothing
end

function campaign_controls_ready(output,jobs)
    length(jobs)==3 && all(jobs) do job
        result=pilot_result(output,job)
        rows=result===nothing ? [] : get(get(result,"evidence",Dict()),"controls",[])
        job["status"]=="completed" && result!==nothing && get(result,"status","")=="completed" &&
            length(rows)==1 && get(only(rows),"status","")=="passed"
    end
end

# Rebuild registrations from prior outcomes instead of trusting checkpoint flags,
# quotas, self-hashed payloads, or a self-asserted frozen base-selection hash.
function campaign_verify_schedule(output,state,manifest)
    jobs=state["jobs"];metadata=get(state,"campaign",Dict())
    length(jobs)>=22 || throw(ArgumentError("campaign lost replacement jobs"))
    isequal([j["payload"] for j in jobs[1:22]],manifest["replacement_jobs"]) || throw(ArgumentError("campaign replacement protocol changed"))
    shadow=Dict{String,Any}("jobs"=>deepcopy(jobs[1:22]),"campaign"=>Dict{String,Any}("bases_frozen"=>false))
    previous_terminal=true;registration_ended=false
    for phase in campaign_phases()
        registered=get(metadata,phase*"_registered",false)
        registered isa Bool || throw(ArgumentError("campaign registration flag must be Boolean"))
        if phase in ("replacement_controls","replacement")
            actual=filter(j->campaign_stage(j)==phase,jobs)
            if phase=="replacement" && registered
                campaign_controls_ready(output,filter(j->campaign_stage(j)=="replacement_controls",jobs)) || throw(ArgumentError("replacement phase bypassed controls"))
            end
            registered && !previous_terminal && throw(ArgumentError("campaign phase registration bypassed unfinished predecessors"))
            registered && get(metadata,phase*"_registered_count",-1)!=0 && throw(ArgumentError("replacement registration counter changed"))
        else
            actual=filter(j->campaign_stage(j)==phase,jobs)
            if !registered
                isempty(actual) || throw(ArgumentError("campaign jobs exist without phase registration"))
                registration_ended=true
                continue
            end
            !registration_ended && previous_terminal || throw(ArgumentError("campaign registration skipped or bypassed a phase"))
            expected=campaign_phase_payloads(output,shadow,manifest,phase)
            isequal(getindex.(actual,"payload"),expected) || throw(ArgumentError("campaign phase payloads differ from deterministic reconstruction: $phase"))
            get(metadata,phase*"_registered_count",-1)==length(expected) || throw(ArgumentError("campaign phase registration count changed"))
            if endswith(phase,"_proposals") || endswith(phase,"_component_paths")
                isequal(get(metadata,phase*"_opportunities",nothing),shadow["campaign"][phase*"_opportunities"]) || throw(ArgumentError("campaign opportunity ledger changed"))
            end
            append!(shadow["jobs"],deepcopy(actual))
        end
        previous_terminal=all(j->j["status"] in CAMPAIGN_TERMINAL,actual)
        !registered && (registration_ended=true)
    end
    isequal(getindex.(shadow["jobs"],"payload"),getindex.(jobs,"payload")) || throw(ArgumentError("campaign gained unexpected or reordered jobs"))
    if get(metadata,"protocol_complete",false)
        campaign_storage_stopped(state) && throw(ArgumentError("storage-stopped campaign cannot claim protocol completion"))
        all(phase->get(metadata,phase*"_registered",false),campaign_phases()) && all(j->j["status"] in CAMPAIGN_TERMINAL,jobs) || throw(ArgumentError("campaign completion flag bypassed unfinished work"))
    end
    reconstructed=shadow["campaign"]
    get(metadata,"bases_frozen",false)==get(reconstructed,"bases_frozen",false) || throw(ArgumentError("campaign base freeze marker bypassed the validation boundary"))
    if get(reconstructed,"bases_frozen",false)
        isequal(metadata["base_selection"],reconstructed["base_selection"]) && metadata["base_selection_hash"]==reconstructed["base_selection_hash"] ||
            throw(ArgumentError("campaign frozen bases or routes differ from prevalidation reconstruction"))
    end
    nothing
end

function campaign_verify_result(output,job)
    status=job["status"]
    status in ("pending","running",CAMPAIGN_TERMINAL...) || throw(ArgumentError("unknown campaign job status"))
    job["result_file"]==joinpath("results",job["id"]*".toml") || throw(ArgumentError("campaign result path is not canonical"))
    status=="completed" && !haskey(job,"result_hash") && throw(ArgumentError("completed campaign job lacks bound evidence"))
    path=safe_atlas_path(output,job["result_file"])
    status=="pending" && isfile(path) && throw(ArgumentError("pending campaign job has existing result evidence"))
    haskey(job,"result_hash") || return nothing
    result=pilot_result(output,job)
    result===nothing && throw(ArgumentError("campaign result is missing"))
    isequal(get(result,"executed_job",nothing),merge(job["payload"],Dict("id"=>job["id"]))) || throw(ArgumentError("campaign result is bound to a different executed job"))
    raw_status=get(result,"status","")
    raw_status in ("completed","unresolved","failed","storage_budget") || throw(ArgumentError("unknown saved campaign result status"))
    conservative_override=status=="unresolved" && startswith(get(job,"error",""),"coordinator could not validate/persist complete job evidence:")
    status==raw_status || conservative_override || throw(ArgumentError("campaign job and result statuses disagree"))
    result
end

function verify_campaign_inputs(output,state,config)
    campaign_run_marker(output,state,config) || return nothing
    manifest=config["campaign_manifest"];digest=config["campaign_manifest_hash"]
    state["config_hash"]==toml_digest(config) || throw(ArgumentError("campaign configuration binding changed"))
    toml_digest(manifest)==digest==get(state,"campaign_manifest_hash","") || throw(ArgumentError("campaign manifest binding changed"))
    toml_digest(TOML.parsefile(safe_atlas_path(output,"campaign_manifest.toml")))==digest || throw(ArgumentError("frozen campaign manifest changed"))
    manifest["numerical_policy_hash"]==toml_digest(manifest["numerical_policy"]) && isequal(config["numerical_policy"],manifest["numerical_policy"]) || throw(ArgumentError("campaign numerical policy binding changed"))
    isequal(config["science"],manifest["effective_science"]) || throw(ArgumentError("campaign effective science changed"))
    atlas_validate_numerical_policy(manifest["numerical_policy"])
    isequal(campaign_calibration_review(manifest["numerical_policy"]),manifest["numerical_review"]) || throw(ArgumentError("campaign independent numerical review binding changed"))
    packets=vcat(manifest["source_packets"],manifest["calibration_packets"])
    foreach(campaign_verify_packet,packets)
    capacity=manifest["capacity"]
    retained=foldl(Base.checked_add,(p["retained_bytes"] for p in packets);init=0)
    config["storage_bytes"]==capacity["new_packet_cap_bytes"] && retained==capacity["historical_retained_bytes"] &&
        Base.checked_add(retained,config["storage_bytes"])==capacity["combined_cap_bytes"]<=CAMPAIGN_COMBINED_CAP || throw(ArgumentError("campaign combined storage binding changed"))
    manifest["bounds"]==campaign_job_bounds() && config["pilot_max_jobs"]==3894 || throw(ArgumentError("campaign finite job bounds changed"))
    length(state["jobs"])<=3894 || throw(ArgumentError("campaign job ceiling exceeded"))
    for (index,job) in enumerate(state["jobs"])
        job["id"]=="job-"*lpad(string(index),6,'0') && job["payload_hash"]==toml_digest(job["payload"]) &&
            job["kind"]==job["payload"]["kind"] && job["queue"]==job["payload"]["queue"] || throw(ArgumentError("campaign canonical job binding changed"))
        campaign_stage(job) in campaign_phases() || throw(ArgumentError("unexpected campaign stage"))
        campaign_verify_result(output,job)
    end
    isequal(state["source_hashes"],manifest["source_hashes"]) || throw(ArgumentError("campaign source bindings changed"))
    for (relative,hash) in manifest["source_hashes"]
        bytes2hex(sha256(read(safe_atlas_path(output,joinpath("source",relative)))))==hash || throw(ArgumentError("campaign executable snapshot changed"))
    end
    campaign_verify_schedule(output,state,manifest)
    nothing
end

function campaign_run_marker(output,state,config)
    marker=get(config,"protocol","")==CAMPAIGN_PROTOCOL || get(state,"protocol","")==CAMPAIGN_PROTOCOL ||
        haskey(config,"campaign_manifest") || haskey(config,"campaign_manifest_hash") || haskey(state,"campaign") ||
        haskey(state,"campaign_manifest_hash") || ispath(joinpath(output,"campaign_manifest.toml")) || islink(joinpath(output,"campaign_manifest.toml")) ||
        any(job->haskey(get(job,"payload",Dict()),"campaign_stage"),get(state,"jobs",[]))
    marker || return false
    get(config,"protocol","")==CAMPAIGN_PROTOCOL && get(state,"protocol","")==CAMPAIGN_PROTOCOL ||
        throw(ArgumentError("campaign protocol markers are missing or inconsistent"))
    true
end

function run_campaign(output,pilot_root,targeted_root,config;combined_storage_bytes=CAMPAIGN_COMBINED_CAP,numerical_policy)
    destination=campaign_resolved_destination(output)
    ispath(destination) && throw(ArgumentError("campaign output already exists"))
    config=merge(config,Dict("campaign_calibration_sources"=>get(numerical_policy,"calibration_sources",get(config,"campaign_calibration_sources",String[]))))
    prepared,effective=campaign_prepare(config,pilot_root,targeted_root;policy=numerical_policy,
        source_hashes=atlas_source_hashes(),combined_cap=combined_storage_bytes)
    manifest=prepared["campaign_manifest"]
    for packet in vcat(manifest["source_packets"],manifest["calibration_packets"])
        campaign_path_contains(packet["root"],destination) && throw(ArgumentError("campaign output must be outside immutable source packets"))
    end
    result=run_pilot(destination,prepared;storage_bytes=effective)
    verify_campaign_inputs(destination,result,TOML.parsefile(joinpath(destination,"config.toml")))
    result
end
