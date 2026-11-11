# Immutable inputs and finite execution for the targeted second pilot.
const TARGETED_PROTOCOL = "targeted_followup_v1"
const TARGETED_COUNTS = Dict("control"=>3,"diagnosis"=>2,"recovery_refinement"=>4,
    "multiroot"=>3,"transition"=>3,"audit_extension"=>7)

function targeted_inventory(root;maximum_bytes=typemax(Int))
    islink(root) && throw(ArgumentError("source packet root must not be a symlink"))
    rows=Dict{String,Any}[]
    counted=0
    for (directory,dirs,files) in walkdir(root;follow_symlinks=false)
        paths=vcat([directory],[joinpath(directory,name) for name in files],
            [joinpath(directory,name) for name in dirs if islink(joinpath(directory,name))])
        for path in paths
            info=lstat(path)
            row=Dict{String,Any}("path"=>relpath(path,root),"bytes"=>max(Int(info.size),512Int(info.blocks)))
            counted=Base.checked_add(counted,row["bytes"])
            counted<=maximum_bytes || throw(ArgumentError("source packet exceeds its frozen storage allowance"))
            if islink(path)
                row["kind"]="symlink";row["target"]=readlink(path)
                row["sha256"]=bytes2hex(sha256(codeunits(row["target"])))
            elseif isfile(path)
                row["kind"]="file"
                row["sha256"]=open(io->bytes2hex(sha256(io)),path,"r")
            elseif isdir(path)
                row["kind"]="directory"
            elseif issocket(path)
                # Frozen browser-review sockets contain no executable input;
                # account/type them without opening or following them.
                row["kind"]="socket";row["mode"]=Int(info.mode)
            else
                throw(ArgumentError("unsupported source packet entry: $(row["path"])"))
            end
            push!(rows,row)
        end
    end
    sort!(rows;by=row->row["path"])
end

function targeted_packet_file(packet,relative)
    path=safe_atlas_path(packet["root"],relative)
    record=findfirst(row->row["path"]==relative,packet["inventory"])
    record===nothing && throw(ArgumentError("source packet file is absent from its inventory: $relative"))
    entry=packet["inventory"][record]
    entry["kind"]=="file" || throw(ArgumentError("executable source inputs must be regular files"))
    bytes=read(path)
    bytes2hex(sha256(bytes))==entry["sha256"] || throw(ArgumentError("source packet input changed: $relative"))
    bytes
end

function targeted_source_packet(path;maximum_bytes=5_000_000_000)
    root=abspath(path)
    isdir(root) && !islink(root) || throw(ArgumentError("source packet must be an existing non-symlink directory"))
    root=realpath(root)
    for relative in ("state.toml","config.toml","pilot_summary.toml")
        isfile(safe_atlas_path(root,relative)) || throw(ArgumentError("source packet is missing $relative"))
    end
    inventory=targeted_inventory(root;maximum_bytes=maximum_bytes)
    packet=Dict{String,Any}("root"=>root,"inventory"=>inventory,
        "inventory_hash"=>toml_digest(Dict("entries"=>inventory)),"retained_bytes"=>sum(row["bytes"] for row in inventory))
    for (key,relative) in (("state","state.toml"),("summary","pilot_summary.toml"),("config","config.toml"))
        packet[key]=TOML.parse(String(targeted_packet_file(packet,relative)))
    end
    state=packet["state"]
    state["mode"]=="pilot" && get(get(state,"pilot",Dict()),"protocol_complete",false)===true ||
        throw(ArgumentError("targeted inputs require a completed frozen first pilot"))
    state["config_hash"]==toml_digest(packet["config"]) || throw(ArgumentError("first pilot configuration hash mismatch"))
    for (relative,digest) in state["source_hashes"]
        bytes2hex(sha256(targeted_packet_file(packet,joinpath("source",relative))))==digest ||
            throw(ArgumentError("first pilot source snapshot hash mismatch"))
    end
    seen=Set{String}()
    for (index,job) in enumerate(state["jobs"])
        id=job["id"]
        id=="job-"*lpad(string(index),6,'0') && !(id in seen) || throw(ArgumentError("source packet job IDs are not unique/canonical"))
        push!(seen,id)
        payload=job["payload"]
        job["payload_hash"]==toml_digest(payload) || throw(ArgumentError("first pilot payload hash mismatch: $id"))
        job["kind"]==payload["kind"] && job["queue"]==get(payload,"queue","events") || throw(ArgumentError("first pilot job metadata mismatch: $id"))
        job["status"] in ("completed","unresolved","failed","timed_out","interrupted","storage_budget") || throw(ArgumentError("source packet contains unattempted/unknown job status"))
        if haskey(job,"result_hash")
            result_bytes=targeted_packet_file(packet,job["result_file"])
            bytes2hex(sha256(result_bytes))==job["result_hash"] ||
                throw(ArgumentError("first pilot result hash mismatch: $(job["id"])"))
            get(payload,"generate",false)===true && throw(ArgumentError("generated discovery source requests require explicit normalization; unsupported by this targeted pilot"))
            raw=TOML.parse(String(result_bytes))
            isequal(get(raw,"executed_job",nothing),merge(payload,Dict("id"=>id))) || throw(ArgumentError("first pilot executed job disagrees with its frozen request: $id"))
        elseif job["status"]=="completed"
            throw(ArgumentError("completed source job lacks bound evidence"))
        end
    end
    packet
end

function targeted_read_result(packet,job_id)
    jobs=filter(job->job["id"]==job_id,packet["state"]["jobs"])
    length(jobs)==1 || throw(ArgumentError("source job ID is not unique: $job_id"))
    job=only(jobs)
    haskey(job,"result_hash") || throw(ArgumentError("source job has no saved result: $job_id"))
    bytes=targeted_packet_file(packet,job["result_file"])
    bytes2hex(sha256(bytes))==job["result_hash"] || throw(ArgumentError("source result hash mismatch"))
    TOML.parse(String(bytes))
end

function targeted_config(path=nothing)
    config=atlas_config(path)
    custom=path===nothing ? Dict{String,Any}() : TOML.parsefile(path)
    haskey(custom,"job_timeout_seconds") || (config["job_timeout_seconds"]=900.0)
    haskey(custom,"pilot_guard_seconds") || (config["pilot_guard_seconds"]=21600.0)
    haskey(custom,"pilot_max_jobs") || (config["pilot_max_jobs"]=22)
    config
end

function targeted_validate_configuration(config,packet)
    baseline=atlas_config()
    allowed=Set(("warmup_seconds","job_timeout_seconds","pilot_guard_seconds","pilot_max_jobs"))
    for (key,value) in config
        key in allowed && continue
        haskey(baseline,key) && isequal(value,baseline[key]) ||
            throw(ArgumentError("targeted protocol does not permit overriding prescribed configuration: $key"))
    end
    isequal(config["science"],packet["config"]["science"]) ||
        throw(ArgumentError("targeted scientific baseline must match the frozen first pilot"))
    cap=get(config,"pilot_max_jobs",22)
    cap isa Integer && !(cap isa Bool) && 0<cap<=22 || throw(ArgumentError("targeted pilot_max_jobs must be in 1:22"))
    validate_atlas_config(config)
    guard=get(config,"pilot_guard_seconds",21600.0)
    guard isa Real && !(guard isa Bool) && isfinite(guard) && guard>0 || throw(ArgumentError("targeted pilot guard must be finite and positive"))
end

function targeted_validate_jobs(jobs,packet)
    length(jobs)==22 || throw(ArgumentError("targeted protocol requires exactly 22 frozen jobs"))
    counts=Dict(role=>0 for role in keys(TARGETED_COUNTS))
    for (index,job) in enumerate(jobs)
        role=get(job,"target_role","")
        haskey(counts,role) || throw(ArgumentError("unknown targeted role: $role"))
        counts[role]+=1
        (index<=3)==(role=="control") || throw(ArgumentError("the three readiness controls must precede all targeted jobs"))
        get(job,"queue","") in ATLAS_QUEUES || throw(ArgumentError("targeted job has invalid queue"))
        haskey(job,"kind") || throw(ArgumentError("targeted job lacks kind"))
        ids=get(job,"source_job_ids",String[])
        ids isa AbstractVector && !isempty(ids) || throw(ArgumentError("targeted job lacks source evidence IDs"))
        for id in ids
            targeted_read_result(packet,id)
        end
        toml_digest(job) # fail before creating output if any payload cannot be frozen
    end
    counts==TARGETED_COUNTS || throw(ArgumentError("targeted manifest role counts differ from the fixed protocol"))
    length(unique(toml_digest.(jobs)))==22 || throw(ArgumentError("targeted job payloads are not distinct"))
    counts
end

function targeted_prepare(config,source_path,combined_cap)
    combined_cap isa Integer && !(combined_cap isa Bool) && combined_cap>0 ||
        throw(ArgumentError("followup requires a positive integer combined --storage-bytes cap"))
    packet=targeted_source_packet(source_path;maximum_bytes=Int(combined_cap))
    targeted_validate_configuration(config,packet)
    effective=Base.checked_sub(Int(combined_cap),packet["retained_bytes"])
    storage_policy(effective) # validate the residual allowance before creating output
    jobs=build_targeted_jobs(packet,config)
    counts=targeted_validate_jobs(jobs,packet)
    final_inventory=targeted_inventory(packet["root"];maximum_bytes=packet["retained_bytes"])
    isequal(final_inventory,packet["inventory"]) || throw(ArgumentError("first packet changed while freezing targeted inputs"))
    manifest=Dict{String,Any}("schema_version"=>1,"protocol"=>TARGETED_PROTOCOL,
        "maximum_jobs"=>22,"mandatory_jobs"=>22,"counts"=>counts,"jobs"=>jobs,
        "job_payload_hashes"=>toml_digest.(jobs),
        "source_packet"=>Dict("root"=>packet["root"],"inventory"=>packet["inventory"],"inventory_hash"=>packet["inventory_hash"],"retained_bytes"=>packet["retained_bytes"]),
        "combined_storage_cap_bytes"=>Int(combined_cap),"new_packet_cap_bytes"=>effective,
        "original_scientific_baseline"=>packet["config"],
        "scientific_policy"=>Dict("prescribed"=>targeted_scientific_policy(),"baseline_science"=>config["science"],"per_job_config_overrides"=>[get(job,"config_overrides",Dict{String,Any}()) for job in jobs]),
        "completeness"=>"CompletenessNotCertified")
    prepared=merge(config,Dict("protocol"=>TARGETED_PROTOCOL,"targeted_manifest"=>manifest,"targeted_manifest_hash"=>toml_digest(manifest)))
    prepared,effective
end

function run_targeted_pilot(output,source_path,config;combined_storage_bytes=nothing)
    # Resolve the nearest existing ancestor so an output alias cannot place new
    # artifacts inside the frozen first packet through a symlinked parent.
    ancestor=abspath(output);suffix=String[]
    while !ispath(ancestor)
        pushfirst!(suffix,basename(ancestor));ancestor=dirname(ancestor)
    end
    destination=joinpath(realpath(ancestor),suffix...)
    source=isdir(source_path) ? realpath(source_path) : abspath(source_path)
    relative=relpath(destination,source)
    (relative=="." || !(first(splitpath(relative))=="..")) &&
        throw(ArgumentError("follow-up output must be outside the immutable first packet"))
    ispath(destination) && throw(ArgumentError("follow-up output already exists"))
    prepared,effective=targeted_prepare(config,source,combined_storage_bytes)
    result=run_pilot(destination,prepared;storage_bytes=effective)
    verify_targeted_inputs(destination,result,TOML.parsefile(joinpath(destination,"config.toml")))
    result
end

function targeted_initial_jobs(config)
    manifest=config["targeted_manifest"]
    toml_digest(manifest)==config["targeted_manifest_hash"] || throw(ArgumentError("targeted manifest hash mismatch"))
    deepcopy(manifest["jobs"])
end

function targeted_next_job!(output,state,config)
    manifest=config["targeted_manifest"]
    metadata=get!(state,"targeted",Dict{String,Any}())
    merge!(metadata,Dict("manifest_counts"=>manifest["counts"],"manifest_hash"=>config["targeted_manifest_hash"],
        "source_packet_root"=>manifest["source_packet"]["root"],"source_packet_inventory_hash"=>manifest["source_packet"]["inventory_hash"],
        "source_packet_retained_bytes"=>manifest["source_packet"]["retained_bytes"],"combined_storage_cap_bytes"=>manifest["combined_storage_cap_bytes"],
        "new_packet_cap_bytes"=>manifest["new_packet_cap_bytes"],"protocol_complete"=>false,"stop_reason"=>"running"))
    controls=[job for job in state["jobs"] if job["payload"]["target_role"]=="control"]
    pending_control=findfirst(job->job["status"]=="pending",controls)
    if pending_control!==nothing
        metadata["stage"]="readiness_controls"
        return controls[pending_control]
    end
    ready=all(controls) do job
        job["status"]=="completed" && haskey(job,"result_hash") || return false
        path=safe_atlas_path(output,job["result_file"])
        bytes=read(path)
        bytes2hex(sha256(bytes))==job["result_hash"] || throw(ArgumentError("targeted control result hash mismatch"))
        result=TOML.parse(String(bytes))
        evidence=get(get(result,"evidence",Dict()),"controls",[])
        result["status"]=="completed" && !isempty(evidence) && all(row->get(row,"status","")=="passed",evidence)
    end
    metadata["controls_ready"]=ready
    if !ready
        metadata["stage"]="blocked"
        metadata["stop_reason"]="control_validation_failed"
        return nothing
    end
    pending=findfirst(job->job["status"]=="pending",state["jobs"])
    if pending===nothing
        metadata["stage"]="complete";metadata["protocol_complete"]=true;metadata["stop_reason"]="protocol_complete"
        return nothing
    end
    metadata["stage"]=state["jobs"][pending]["payload"]["target_role"]
    state["jobs"][pending]
end

function targeted_run_marker(output,state,config)
    marker=haskey(config,"protocol") || haskey(state,"protocol") || haskey(config,"targeted_manifest") ||
        haskey(config,"targeted_manifest_hash") ||
        haskey(state,"targeted_manifest_hash") || haskey(state,"targeted") ||
        ispath(joinpath(output,"targeted_manifest.toml")) || islink(joinpath(output,"targeted_manifest.toml"))
    marker |= any(get(state,"jobs",[])) do job
        job isa AbstractDict || return false
        payload=get(job,"payload",Dict())
        (payload isa AbstractDict && haskey(payload,"target_role")) || startswith(string(get(job,"kind","")),"targeted_")
    end
    marker || return false
    get(config,"protocol","")==TARGETED_PROTOCOL && get(state,"protocol","")==TARGETED_PROTOCOL ||
        throw(ArgumentError("targeted protocol markers are missing or inconsistent"))
    true
end

function verify_targeted_inputs(output,state,config)
    targeted_run_marker(output,state,config) || return nothing
    manifest=config["targeted_manifest"]
    state["protocol"]==config["protocol"] && state["config_hash"]==toml_digest(config) || throw(ArgumentError("targeted configuration binding changed"))
    digest=config["targeted_manifest_hash"]
    toml_digest(manifest)==digest==get(state,"targeted_manifest_hash","") || throw(ArgumentError("targeted manifest binding changed"))
    frozen=TOML.parsefile(safe_atlas_path(output,"targeted_manifest.toml"))
    toml_digest(frozen)==digest || throw(ArgumentError("frozen targeted manifest changed"))
    inventory=targeted_inventory(manifest["source_packet"]["root"];maximum_bytes=manifest["source_packet"]["retained_bytes"])
    isequal(inventory,manifest["source_packet"]["inventory"]) || throw(ArgumentError("frozen first packet inventory changed"))
    toml_digest(Dict("entries"=>inventory))==manifest["source_packet"]["inventory_hash"] || throw(ArgumentError("frozen first packet digest changed"))
    config["storage_bytes"]==manifest["new_packet_cap_bytes"] &&
        manifest["new_packet_cap_bytes"]+manifest["source_packet"]["retained_bytes"]==manifest["combined_storage_cap_bytes"] ||
        throw(ArgumentError("combined first/second storage allowance changed"))
    length(manifest["jobs"])==22 && toml_digest.(manifest["jobs"])==manifest["job_payload_hashes"] || throw(ArgumentError("targeted payloads changed"))
    isequal(manifest["scientific_policy"]["prescribed"],targeted_scientific_policy()) || throw(ArgumentError("targeted scientific policy changed"))
    packet=deepcopy(manifest["source_packet"])
    for (key,relative) in (("state","state.toml"),("summary","pilot_summary.toml"),("config","config.toml"))
        packet[key]=TOML.parse(String(targeted_packet_file(packet,relative)))
    end
    isequal(manifest["original_scientific_baseline"],packet["config"]) &&
        isequal(manifest["scientific_policy"]["baseline_science"],config["science"]) &&
        isequal(config["science"],packet["config"]["science"]) || throw(ArgumentError("targeted scientific baseline changed"))
    isequal(manifest["scientific_policy"]["per_job_config_overrides"],
        [get(job,"config_overrides",Dict{String,Any}()) for job in manifest["jobs"]]) || throw(ArgumentError("targeted job overrides changed"))
    isequal(build_targeted_jobs(packet,config),manifest["jobs"]) || throw(ArgumentError("targeted payloads no longer match their deterministic prescribed construction"))
    for (relative,hash) in get(state,"source_hashes",Dict())
        bytes2hex(sha256(read(safe_atlas_path(output,joinpath("source",relative)))))==hash || throw(ArgumentError("targeted source snapshot changed"))
    end
    length(state["jobs"])==22 || throw(ArgumentError("targeted run gained or lost jobs"))
    for (job,payload) in zip(state["jobs"],manifest["jobs"])
        isequal(job["payload"],payload) || throw(ArgumentError("targeted scheduled payload changed"))
    end
    nothing
end
