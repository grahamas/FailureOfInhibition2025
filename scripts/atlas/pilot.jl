# Finite, declared experiment. Held-out observations never change the base pool.
const PILOT_VERSION = 1
const PILOT_TRAINING_INDICES = (1,2,3,4)
const PILOT_HELDOUT_INDICES = (5,6)
const PILOT_AUDIT_INDICES = (1,3)
const PILOT_PHASES = ("control","training","training_followup","audit","heldout","recovery")
const PILOT_AUDIT_MASKS = (0,1,3,5,7,10,15)
const PILOT_RECOVERY_MASKS = (0,5,10,15)

const PILOT_SELECTION_SEED = "pilot-v1"
function pilot_selected_indices(k,chart,mask)
    candidates=collect(1:256)
    sort!(candidates;by=index->(bytes2hex(sha256("$PILOT_SELECTION_SEED|shell=$k|chart=$chart|mask=$mask|index=$index")),index))
    candidates[1:6]
end
function pilot_marginal_holes(samples,k,mask)
    holes=String[];values=getindex.(samples,"chart_coordinates")
    for coordinate in (5,6)
        any(row->row[coordinate]<0,values) || push!(holes,"coordinate_$coordinate:no_negative_sample")
        any(row->row[coordinate]>=0,values) || push!(holes,"coordinate_$coordinate:no_nonnegative_sample")
    end
    any(row->row[7]<1,values) || push!(holes,"h:no_sample_below_one")
    any(row->row[7]>=1,values) || push!(holes,"h:no_sample_at_least_one")
    if k==4 && mask&1==0
        any(row->row[1]>8,values) || push!(holes,"A:no_sample_above_eight")
        any(row->row[1]<=8,values) || push!(holes,"A:no_sample_at_most_eight")
    end
    holes
end
function pilot_cohort_design(k,chart,mask)
    indices=pilot_selected_indices(k,chart,mask)
    samples=Dict{String,Any}[]
    for (position,index) in enumerate(indices)
        sample=shell_sample(k,index;chart=chart,mask=mask)
        push!(samples,Dict("design_position"=>position,"role"=>position<=4 ? "training" : "heldout",
            "accepted_index"=>index,"raw_sequence_index"=>sample["raw_sequence_index"],
            "chart_coordinates"=>sample["chart_coordinates"],"parameters"=>sample["params"]))
    end
    Dict("cohort"=>pilot_cohort(k,chart,mask),"shell"=>k,"chart"=>chart,"mask"=>mask,"indices"=>indices,
        "training_indices"=>indices[1:4],"heldout_indices"=>indices[5:6],"audit_indices"=>indices[[1,3]],
        "samples"=>samples,"marginal_holes"=>Dict(role=>pilot_marginal_holes(filter(sample->sample["role"]==role,samples),k,mask) for role in ("training","heldout")))
end
function pilot_support_ledger(designs)
    rows=Dict{String,Any}[]
    for role in ("training","heldout"),chart in ("parameter","equilibrium")
        samples=[sample for design in designs if design["chart"]==chart for sample in design["samples"] if sample["role"]==role]
        labels=chart=="parameter" ? ("A","B","C","D","a","b","h") : ("A","B","C","D","u","v","h")
        dimensions=Dict{String,Any}()
        for (coordinate,label) in enumerate(labels)
            values=[sample["chart_coordinates"][coordinate] for sample in samples]
            dimensions[label]=Dict("minimum"=>minimum(values),"maximum"=>maximum(values),"negative_count"=>count(<(0),values),"nonnegative_count"=>count(>=(0),values))
        end
        push!(rows,Dict("role"=>role,"chart"=>chart,"sample_count"=>length(samples),"dimensions"=>dimensions,
            "h_below_one_count"=>count(sample->sample["chart_coordinates"][7]<1,samples),
            "h_at_least_one_count"=>count(sample->sample["chart_coordinates"][7]>=1,samples),
            "A_at_most_eight_count"=>count(sample->sample["chart_coordinates"][1]<=8,samples),
            "A_above_eight_count"=>count(sample->sample["chart_coordinates"][1]>8,samples)))
    end
    rows
end
function pilot_manifest(config)
    designs=[pilot_cohort_design(k,chart,mask) for k in 1:4 for chart in ("parameter","equilibrium") for mask in 0:15]
    Dict{String,Any}("selection_algorithm"=>"SHA-256 lexicographic rank, numeric-index tie-break","selection_seed"=>PILOT_SELECTION_SEED,
        "selection_pool"=>collect(1:256),"selection_pool_bounds"=>[1,256],"selection_encoding"=>"pilot-v1|shell={k}|chart={chart}|mask={mask}|index={accepted_index}",
        "cohort_designs"=>designs,"support_ledger"=>pilot_support_ledger(designs),"protocol_version"=>PILOT_VERSION,"shells"=>collect(1:4),
        "charts"=>["parameter","equilibrium"],"coupling_masks"=>collect(0:15),
        "training_design_positions"=>collect(PILOT_TRAINING_INDICES),"heldout_design_positions"=>collect(PILOT_HELDOUT_INDICES),"audit_design_positions"=>collect(PILOT_AUDIT_INDICES),"audit_masks"=>collect(PILOT_AUDIT_MASKS),
        "recovery_masks"=>collect(PILOT_RECOVERY_MASKS),"recovery_samples"=>17,
        "counts"=>Dict("control"=>3,"training"=>512,"training_followup_quota"=>24,
            "audit"=>112,"heldout"=>256,"recovery"=>64),
        "holdout_design"=>"Disjoint deterministic hash-ranked accepted-index blocks, not independent random observations",
        "design_rationale"=>"Dispersed fixed-pool indices prevent shared high-prime prefix artifacts; seed fixed before coordinate ledger and numerical outcomes",
        "window_scope"=>"Shell coordinates refer to each chart; equilibrium-chart images do not obey parameter-chart box bounds",
        "mandatory_jobs"=>947,"maximum_jobs"=>971,"training_followup_scope"=>"At most first eligible continuation/nexus/periodic per shell/chart, one generation; not statistically representative",
        "reliability_scope"=>"Seven preselected coupling masks cover zero-count families and recovery strata, not every directional face",
        "completeness"=>"CompletenessNotCertified")
end
pilot_cohort(k,chart,mask)="shell$k/$chart/mask$mask"
function pilot_sample_job(k,chart,mask,position,role)
    index=pilot_selected_indices(k,chart,mask)[position]
    Dict{String,Any}("kind"=>"pilot_sample","queue"=>"discovery","pilot_role"=>role,
        "pilot_stage"=>role,"pilot_cohort"=>pilot_cohort(k,chart,mask),"pilot_shell"=>k,
        "pilot_chart"=>chart,"pilot_mask"=>mask,"pilot_sample_index"=>index,"pilot_design_position"=>position)
end
function pilot_initial_jobs(config)
    bt=merge(analytic_seed(),Dict("pilot_role"=>"control","pilot_stage"=>"control","pilot_control"=>"bt"))
    cusp=merge(equilibrium_chart(8.,0.,0.,0.,-log(2.),0.,log(9.)),
        Dict("kind"=>"nexus","queue"=>"events","event_type"=>"cusp","pilot_role"=>"control","pilot_stage"=>"control","pilot_control"=>"cusp"))
    hopf=merge(equilibrium_chart(12.,8.,8.,0.,-log(2.),0.,log(9.);r=2.1),
        Dict("kind"=>"periodic","queue"=>"cycles","pilot_role"=>"control","pilot_stage"=>"control","pilot_control"=>"hopf"))
    jobs=Dict{String,Any}[bt,cusp,hopf]
    for k in 1:4,chart in ("parameter","equilibrium"),mask in 0:15,index in PILOT_TRAINING_INDICES
        push!(jobs,pilot_sample_job(k,chart,mask,index,"training"))
    end
    jobs
end
pilot_role(job)=get(job["payload"],"pilot_role","")
function pilot_result(output,job)
    haskey(job,"result_hash") || return nothing
    path=safe_atlas_path(output,job["result_file"])
    isfile(path) || return nothing
    bytes2hex(sha256(read(path)))==job["result_hash"] || throw(ArgumentError("pilot result hash mismatch"))
    TOML.parsefile(path)
end
function pilot_training_points(output,state)
    points=Dict{String,Any}[]
    for job in state["jobs"]
        pilot_role(job)=="training" && job["status"]=="completed" || continue
        result=pilot_result(output,job);result===nothing && continue
        for point in get(get(result,"evidence",Dict()),"points",[])
            isempty(get(point,"roots",[])) && continue
            occursin("unresolved",lowercase(get(point,"signature","unresolved"))) && continue
            push!(points,merge(point,Dict("source_job"=>job["id"],"cohort"=>job["payload"]["pilot_cohort"])))
        end
    end
    points
end
function pilot_freeze_bases!(output,state)
    entries=Dict{String,Any}[];excluded=Dict{String,Any}[];seen=Set{String}()
    audits=Dict{Tuple{String,Int},String}()
    for job in state["jobs"]
        pilot_role(job)=="audit" || continue
        result=pilot_result(output,job)
        rows=result===nothing ? [] : get(get(result,"evidence",Dict()),"audits",[])
        comparisons=length(rows)==1 ? get(first(rows),"comparisons",[]) : []
        repeat=length(rows)==1 ? get(first(rows),"training_repeat_comparison",Dict()) : Dict()
        passed=job["status"]=="completed"&&length(comparisons)==2&&all(row->get(row,"status","")=="agreement_observed"&&get(row,"classification_resolved",false),vcat(comparisons,[repeat]))
        audits[(job["payload"]["pilot_cohort"],job["payload"]["pilot_sample_index"])]=passed ? "fixed_parameter_agreement_observed" : "disagreement_or_unresolved"
    end
    for point in pilot_training_points(output,state)
        job=only(filter(job->job["id"]==point["source_job"],state["jobs"]))
        index=job["payload"]["pilot_sample_index"];audit_key=(point["cohort"],index)
        cohort_status=get(audits,audit_key,"not_audited")
        cohort_warning=any(key[1]==point["cohort"]&&status=="disagreement_or_unresolved" for (key,status) in audits)
        directly_audited=haskey(audits,audit_key)
        if directly_audited&&cohort_status=="disagreement_or_unresolved"
            push!(excluded,Dict("point_id"=>point["id"],"cohort"=>point["cohort"],"sample_index"=>index,"reason"=>"direct_fixed_parameter_audit_disagreement_or_unresolved"))
            continue
        end
        signature=point["signature"];signature in seen && continue
        push!(seen,signature)
        push!(entries,Dict("point_id"=>point["id"],"source_job"=>point["source_job"],"cohort"=>point["cohort"],
            "sample_index"=>index,"audit_status"=>directly_audited ? cohort_status : "not_audited","cohort_audit_warning"=>cohort_warning,
            "signature"=>signature,"parameters"=>point["parameters"],"roots"=>point["roots"]))
    end
    selection=Dict{String,Any}("entries"=>entries,"excluded_candidates"=>excluded,"rule"=>"Earliest successful eligible training point per full sampled structural signature, excluding directly contradicted audits",
        "scope"=>"Candidate grouping hypothesis; equal signatures do not establish connected regions",
        "source_role"=>"training","frozen_before_heldout"=>true)
    routes=Dict{String,Any}[]
    for k in 1:4,chart in ("parameter","equilibrium"),mask in PILOT_RECOVERY_MASKS,index in PILOT_HELDOUT_INDICES
        position=index;accepted_index=pilot_selected_indices(k,chart,mask)[position]
        parameters=shell_sample(k,accepted_index;chart=chart,mask=mask)["params"]
        route=Dict{String,Any}("cohort"=>pilot_cohort(k,chart,mask),"sample_index"=>accepted_index,"design_position"=>position,"parameters"=>parameters,
            "selection_rule"=>"Parameter-only nearest frozen base; no heldout inventory or stability used")
        if !isempty(entries)
            ordered=sort(entries;by=base->(pilot_distance(base["parameters"],parameters),base["point_id"]))
            route["base_point_id"]=first(ordered)["point_id"]
        end
        push!(routes,route)
    end
    selection["recovery_routes"]=routes
    state["pilot"]["base_selection"]=selection
    state["pilot"]["base_selection_hash"]=toml_digest(selection)
    state["pilot"]["base_selection_frozen"]=true
end
function pilot_followup_jobs(output,state)
    chosen=Dict{String,Any}[];opportunities=Dict{String,Any}[]
    for k in 1:4,chart in ("parameter","equilibrium"),kind in ("continuation","nexus","periodic")
        found=false
        slot=Dict{String,Any}("shell"=>k,"chart"=>chart,"kind"=>kind,"status"=>"not_observed")
        for job in state["jobs"]
            payload=job["payload"]
            pilot_role(job)=="training" && payload["pilot_shell"]==k && payload["pilot_chart"]==chart || continue
            result=pilot_result(output,job);result===nothing && continue
            for proposal in get(result,"followups",[])
                proposal["kind"]==kind || continue
                push!(chosen,merge(proposal,Dict("pilot_role"=>"training_followup","pilot_stage"=>"training_followup",
                    "pilot_cohort"=>payload["pilot_cohort"],"pilot_parent_job"=>job["id"],"pilot_followup_depth"=>1)))
                slot["status"]="queued";slot["source_job"]=job["id"];slot["source_status"]=job["status"]
                found=true;break
            end
            found && break
        end
        push!(opportunities,slot)
    end
    get!(state,"pilot",Dict{String,Any}())["followup_opportunities"]=opportunities
    chosen
end
function pilot_distance(a,b)
    sum(((a[k]-b[k])/max(1.0,abs(a[k]),abs(b[k])))^2 for k in ATLAS_COORDINATES)
end
function pilot_recovery_jobs(output,state)
    entries=state["pilot"]["base_selection"]["entries"]
    jobs=Dict{String,Any}[]
    for heldout in state["jobs"]
        payload=heldout["payload"]
        pilot_role(heldout)=="heldout" && payload["pilot_mask"] in PILOT_RECOVERY_MASKS || continue
        recovery=Dict{String,Any}("kind"=>"pilot_recovery","queue"=>"events","pilot_role"=>"recovery",
            "pilot_stage"=>"recovery","pilot_cohort"=>payload["pilot_cohort"],"pilot_target_id"=>heldout["id"],
            "recovery_target_cohort"=>"$(payload["pilot_cohort"])/index$(payload["pilot_sample_index"])","pilot_sample_index"=>payload["pilot_sample_index"],"pilot_design_position"=>payload["pilot_design_position"],"base_selection_hash"=>state["pilot"]["base_selection_hash"])
        result=pilot_result(output,heldout)
        targets=result===nothing ? [] : get(get(result,"evidence",Dict()),"points",[])
        route=only(filter(route->route["cohort"]==payload["pilot_cohort"]&&route["sample_index"]==payload["pilot_sample_index"],state["pilot"]["base_selection"]["recovery_routes"]))
        if haskey(route,"base_point_id")
            base=only(filter(base->base["point_id"]==route["base_point_id"],entries))
            available=heldout["status"]=="completed"&&!isempty(targets)&&!isempty(first(targets)["roots"])
            target=available ? first(targets) : Dict("parameters"=>route["parameters"],"roots"=>Any[],"signature"=>"unavailable","id"=>heldout["id"])
            recovery["base"]=base;recovery["target"]=target;recovery["heldout_inventory_available"]=available
            recovery["base_target_signature_equal"]=available&&base["signature"]==target["signature"]
            recovery["selection_rule"]=route["selection_rule"]
        else
            recovery["unavailable_reason"]="no_resolved_training_base"
        end
        push!(jobs,recovery)
    end
    jobs
end
function pilot_audit_jobs(output,state)
    jobs=Dict{String,Any}[]
    for k in 1:4,chart in ("parameter","equilibrium"),mask in PILOT_AUDIT_MASKS,index in PILOT_AUDIT_INDICES
        payload=merge(pilot_sample_job(k,chart,mask,index,"audit"),Dict("kind"=>"pilot_audit","queue"=>"events","pilot_pair"=>"$(pilot_cohort(k,chart,mask))/position$index"))
        source=only(filter(job->pilot_role(job)=="training"&&job["payload"]["pilot_cohort"]==payload["pilot_cohort"]&&job["payload"]["pilot_sample_index"]==payload["pilot_sample_index"],state["jobs"]))
        payload["training_source_job"]=source["id"]
        result=pilot_result(output,source)
        points=result===nothing ? [] : get(get(result,"evidence",Dict()),"points",[])
        if source["status"]=="completed"&&!isempty(points)
            point=first(points)
            payload["training_point"]=Dict("id"=>point["id"],"parameters"=>point["parameters"],"signature"=>point["signature"],"roots"=>[Dict("state"=>root["state"]) for root in point["roots"]])
        end
        push!(jobs,payload)
    end
    jobs
end
function pilot_next_job!(output,state,config)
    protocol=get!(state,"pilot") do
        Dict{String,Any}("manifest"=>pilot_manifest(config),"protocol_version"=>PILOT_VERSION,
            "stage"=>"control","protocol_complete"=>false,"base_selection_frozen"=>false)
    end
    get!(protocol,"manifest") do
        pilot_manifest(config)
    end
    get!(protocol,"manifest_counts",protocol["manifest"]["counts"])
    get!(protocol,"base_selection_frozen",false)
    get!(protocol,"protocol_complete",false)
    if protocol["base_selection_frozen"]
        toml_digest(protocol["base_selection"])==protocol["base_selection_hash"] || throw(ArgumentError("frozen pilot base selection changed"))
    end
    for phase in PILOT_PHASES
        protocol["stage"]=phase
        registered_key=phase*"_registered"
        if phase in ("control","training")
            protocol[registered_key]=true # initial manifest is registered by the engine
        elseif !get(protocol,registered_key,false)
            payloads=if phase=="training_followup"
                pilot_followup_jobs(output,state)
            elseif phase=="audit"
                pilot_audit_jobs(output,state)
            elseif phase=="heldout"
                get(protocol,"base_selection_frozen",false) || pilot_freeze_bases!(output,state)
                [pilot_sample_job(k,chart,mask,index,"heldout") for k in 1:4 for chart in ("parameter","equilibrium") for mask in 0:15 for index in PILOT_HELDOUT_INDICES]
            else
                pilot_recovery_jobs(output,state)
            end
            for payload in payloads
                register_atlas_job!(state,payload)
            end
            protocol[registered_key]=true
            protocol[phase*"_registered_count"]=length(payloads)
        end
        jobs=filter(job->pilot_role(job)==phase,state["jobs"])
        pending=findfirst(job->job["status"]=="pending",jobs)
        pending===nothing || return jobs[pending]
        any(job->job["status"]=="running",jobs) && return nothing
        if phase=="control"
            ready=length(jobs)==3 && all(jobs) do job
                result=pilot_result(output,job)
                rows=result===nothing ? [] : get(get(result,"evidence",Dict()),"controls",[])
                job["status"]=="completed" && length(rows)==1 && first(rows)["status"]=="passed"
            end
            protocol["readiness"]=ready
            if !ready
                protocol["stop_reason"]="control_validation_failed"
                return nothing
            end
        end
    end
    protocol["stage"]="complete";protocol["protocol_complete"]=true;protocol["stop_reason"]="protocol_complete"
    nothing
end
function pilot_sample_payload(job)
    merge(shell_sample(job["pilot_shell"],job["pilot_sample_index"];chart=job["pilot_chart"],mask=job["pilot_mask"]),
        Dict("kind"=>"discovery","queue"=>"discovery","id"=>get(job,"id","pilot-sample")))
end
function pilot_search_comparison(a,b,label_a,label_b)
    first_roots=a["roots"];second_roots=b["roots"]
    used=Set{Int}();matched=0
    for root in first_roots
        distances=[i in used ? Inf : norm(root["state"]-candidate["state"],Inf) for (i,candidate) in enumerate(second_roots)]
        isempty(distances) && continue
        index=argmin(distances)
        if distances[index]<=1e-7
            push!(used,index);matched+=1
        end
    end
    inventory_match=!isempty(first_roots)&&length(first_roots)==length(second_roots)==matched
    classification_resolved=!occursin("unresolved",lowercase(a["signature"]*b["signature"]))
    agreement=inventory_match&&classification_resolved&&a["signature"]==b["signature"]
    Dict("pair"=>[label_a,label_b],"status"=>agreement ? "agreement_observed" : "disagreement_or_unresolved",
        "inventory_match"=>inventory_match,"classification_resolved"=>classification_resolved,"first_root_count"=>length(first_roots),"second_root_count"=>length(second_roots),"matched_roots"=>matched)
end
function pilot_branch_summary(curve)
    [Dict("direction"=>branch.direction,"termination"=>string(branch.termination),
        "points"=>[Dict("parameter"=>point.parameter,"state"=>point.state,"balance_residual"=>point.equilibrium_attempt.residual_norm,
            "stability"=>string(point.stability.classification)) for point in branch.points],
        "candidates"=>[Dict("kind"=>string(candidate.kind),"first_point"=>candidate.first_point,"second_point"=>candidate.second_point) for candidate in branch.candidates],
        "attempts"=>[Dict("accepted"=>attempt.accepted,"status"=>string(attempt.status),"candidate"=>attempt.candidate,"step"=>attempt.step,"iterations"=>attempt.iterations,"arclength_residual"=>attempt.arclength_residual,"equilibrium_attempt"=>attempt.equilibrium_attempt===nothing ? Dict{String,Any}() : atlas_attempt(attempt.equilibrium_attempt)) for attempt in branch.attempts]) for branch in (curve.negative,curve.positive)]
end
function execute_pilot_audit(job,config)
    payload=pilot_sample_payload(job);p=payload["params"]
    loose=merge(config,Dict("seed_grid"=>5));dense=merge(loose,Dict("seed_grid"=>9))
    tight=merge(dense,Dict("balance_atol"=>atlas_balance_tolerance(config)/100,"event_atol"=>atlas_event_tolerance(config,1.0)/100))
    extra=get(job,"pilot_chart","")=="equilibrium" ? [occupancy(payload["chart_coordinates"][5:7]...)] : Vector{Float64}[]
    labels=["grid5_loose","grid9_loose","grid9_tight"]
    variants=[Dict("label"=>label,"point"=>atlas_point(p,cfg;id="$(get(job,"id","audit"))-$label",extra_seeds=extra)) for (label,cfg) in zip(labels,(loose,dense,tight))]
    comparisons=[pilot_search_comparison(variants[1]["point"],variants[2]["point"],labels[1],labels[2]),pilot_search_comparison(variants[2]["point"],variants[3]["point"],labels[2],labels[3])]
    paired=Dict{String,Any}("status"=>"unavailable_no_baseline_root")
    roots=variants[1]["point"]["roots"]
    if !isempty(roots)
        seed=first(roots)["state"];name="A";radius=max(1.0,p[name])*0.5;bounds=(max(0.0,p[name]-radius),p[name]+radius)
        curves=[continue_equilibria(value->reduced_model(atlas_replace(p,name,value)),seed,p[name];parameter_bounds=bounds,
            options=ContinuationOptions(max_steps=depth,corrector_atol=min(1e-10,atlas_balance_tolerance(tight))),equilibrium_options=atlas_equilibrium_options(tight)) for depth in (12,24)]
        summaries=pilot_branch_summary.(curves);difference=0.0
        for (a,b) in zip(summaries[1],summaries[2]),i in 1:min(length(a["points"]),length(b["points"]))
            difference=max(difference,norm(a["points"][i]["state"]-b["points"][i]["state"],Inf))
        end
        depth_rows=Dict{String,Any}[]
        for (a,b) in zip(summaries[1],summaries[2])
            shallow=a["points"];deep=b["points"];count=min(length(shallow),length(deep))
            extension=length(deep)>length(shallow) ? deep[length(shallow)+1:end] : []
            transitions=sum((deep[index]["stability"]!=deep[index-1]["stability"] for index in max(2,length(shallow)+1):length(deep));init=0)
            push!(depth_rows,Dict("direction"=>a["direction"],"extension_point_count"=>length(extension),
                "shallow_parameter_span"=>isempty(shallow) ? Float64[] : [minimum(point["parameter"] for point in shallow),maximum(point["parameter"] for point in shallow)],
                "deep_parameter_span"=>isempty(deep) ? Float64[] : [minimum(point["parameter"] for point in deep),maximum(point["parameter"] for point in deep)],
                "new_candidate_count"=>max(0,length(b["candidates"])-length(a["candidates"])),"extension_stability_changes"=>transitions,
                "shallow_termination"=>a["termination"],"deep_termination"=>b["termination"],
                "depth_adequacy"=>b["termination"]=="step_limit" ? "unresolved_step_limit" : "bounded_termination_only"))
        end
        prefix_count=sum(min(length(a["points"]),length(b["points"])) for (a,b) in zip(summaries[1],summaries[2]))
        paired=Dict("status"=>"attempted","coordinate"=>name,"bounds"=>collect(bounds),"initial_state"=>seed,
            "shallow_steps"=>12,"deep_steps"=>24,"shallow"=>summaries[1],"deep"=>summaries[2],"shared_prefix_max_state_difference"=>difference,
            "shared_prefix_count"=>prefix_count,"comparison_tolerance"=>1e-7,
            "comparison_status"=>difference<=1e-7&&prefix_count>0 ? "shared_prefix_consistent" : "shared_prefix_disagreement_or_unresolved",
            "depth_extensions"=>depth_rows,"scope"=>"Deterministic shared-prefix consistency is not depth-insensitivity or branch completeness")
    end
    repeat=haskey(job,"training_point") ? pilot_search_comparison(job["training_point"],variants[1]["point"],"saved_training","grid5_loose") : Dict("status"=>"unavailable_training_result")
    record=Dict("cohort"=>job["pilot_cohort"],"sample_index"=>job["pilot_sample_index"],"role"=>"audit","variants"=>variants,"comparisons"=>comparisons,"training_repeat_comparison"=>repeat,"continuation_pair"=>paired,
        "scope"=>"Predeclared subset; agreement does not establish root or branch completeness")
    Dict("status"=>"completed","evidence"=>Dict("audits"=>[record]),"followups"=>Any[])
end
function pilot_match_roots(targets,candidates;tolerance=1e-6)
    # Injective matching; unresolved close-root ambiguity is never counted twice.
    matched=Dict{String,Any}[];used=Set{Int}()
    for (target_index,target) in enumerate(targets)
        eligible=[i for (i,candidate) in enumerate(candidates) if !(i in used)&&norm(target["state"]-candidate["state"],Inf)<=tolerance]
        isempty(eligible) && continue
        index=first(sort(eligible;by=i->(norm(target["state"]-candidates[i]["state"],Inf),i)))
        push!(used,index);push!(matched,Dict("target_root"=>target_index,"candidate_root"=>index,"distance"=>norm(target["state"]-candidates[index]["state"],Inf)))
    end
    matched
end
function execute_pilot_recovery(job,config)
    record=Dict{String,Any}("target_cohort"=>job["recovery_target_cohort"],"target_job_id"=>job["pilot_target_id"],
        "status"=>"unavailable_target_or_base","tracked_target_coverage"=>false,"sampled_inventory_preserved"=>false,
        "independent_endpoint_solved"=>false,"samples"=>Any[],"changes"=>Any[],"tracks"=>Any[],"continuous_coverage_certified"=>false,
        "scope"=>"Sampled corrected root tracks; no certified branch identity, continuous coverage, or substitution guarantee")
    if !haskey(job,"base") || !haskey(job,"target")
        record["unavailable_reason"]=get(job,"unavailable_reason","missing_target_or_base")
        return Dict("status"=>"unresolved","evidence"=>Dict("recoveries"=>[record]),"followups"=>Any[])
    end
    base,target=job["base"],job["target"]
    p0,p1=base["parameters"],target["parameters"]
    options=atlas_equilibrium_options(config)
    states=[copy(root["state"]) for root in base["roots"]]
    clean=trues(length(states));active=trues(length(states));attempts=Dict{String,Any}[]
    samples=Dict{String,Any}[];changes=Dict{String,Any}[];previous_parameters=p0
    for (index,t) in enumerate(range(0.0,1.0;length=17))
        p=Dict{String,Any}(key=>(1-t)*p0[key]+t*p1[key] for key in ATLAS_COORDINATES)
        corrected=Dict{Int,Vector{Float64}}()
        for track in eachindex(states)
            active[track] || continue
            previous=copy(states[track]);solve=solve_equilibrium(reduced_model(p),previous;options=options)
            attempt=solve.attempt;reasons=String[]
            admissible=attempt.validation==AdmissibleCandidate
            row=Dict{String,Any}("sample_index"=>index,"track"=>track,"forward"=>atlas_attempt(attempt),"previous_state"=>previous)
            if !admissible
                active[track]=false;clean[track]=false;push!(reasons,"forward_correction_unresolved")
            else
                candidate=copy(attempt.candidate);movement=norm(candidate-previous,Inf)
                row["movement"]=movement
                movement>0.1 && push!(reasons,"large_step_diagnostic")
                attempt.near_singular && push!(reasons,"near_singular_root")
                reverse=solve_equilibrium(reduced_model(previous_parameters),candidate;options=options)
                row["reverse"]=atlas_attempt(reverse.attempt)
                reverse_ok=reverse.attempt.validation==AdmissibleCandidate&&norm(reverse.attempt.candidate-previous,Inf)<=1e-6
                row["reverse_returned_previous"]=reverse_ok
                reverse_ok || push!(reasons,"reverse_correction_ambiguous")
                states[track]=candidate;corrected[track]=candidate
                isempty(reasons) || (clean[track]=false)
            end
            row["diagnostics"]=reasons;row["track_clean"]=clean[track];push!(attempts,row)
        end
        for first_track in keys(corrected),second_track in keys(corrected)
            first_track<second_track || continue
            if norm(corrected[first_track]-corrected[second_track],Inf)<=1e-7
                clean[first_track]=false;clean[second_track]=false
                push!(attempts,Dict("sample_index"=>index,"tracks"=>[first_track,second_track],"diagnostics"=>["duplicate_corrected_roots"],"track_clean"=>false))
            end
        end
        extra=collect(values(corrected))
        index==17 && append!(extra,[root["state"] for root in target["roots"]])
        point=atlas_point(p,config;id="$(get(job,"id","recovery"))-sample$index",extra_seeds=extra)
        push!(samples,point)
        if index>1&&samples[end-1]["signature"]!=point["signature"]
            push!(changes,Dict("first_point_id"=>samples[end-1]["id"],"second_point_id"=>point["id"],
                "first_signature"=>samples[end-1]["signature"],"second_signature"=>point["signature"]))
        end
        previous_parameters=p
    end
    observed=Dict{String,Any}[Dict("state"=>root["state"]) for root in target["roots"]]
    for root in samples[end]["roots"]
        all(existing->norm(existing["state"]-root["state"],Inf)>1e-7,observed) && push!(observed,Dict("state"=>root["state"]))
    end
    candidates=[Dict("state"=>states[index],"track"=>index) for index in eachindex(states) if active[index]&&clean[index]]
    matching=pilot_match_roots(observed,candidates)
    observed_covered=!isempty(observed)&&length(matching)==length(observed)
    available=get(job,"heldout_inventory_available",!isempty(target["roots"]))
    covered=available&&observed_covered
    preserved=isempty(changes)&&all(point->!isempty(point["roots"])&&!occursin("unresolved",lowercase(point["signature"])),samples)
    merge!(record,Dict("base_point_id"=>base["point_id"],"base_target_signature_equal"=>job["base_target_signature_equal"],
        "status"=>covered ? "sampled_tracked_target_coverage" : "tracked_coverage_unresolved",
        "tracked_target_coverage"=>covered,"tracked_observed_endpoint_coverage"=>observed_covered,"heldout_inventory_available"=>available,"sampled_inventory_preserved"=>preserved,
        "independent_endpoint_solved"=>!isempty(samples[end]["roots"]),"saved_target_root_count"=>length(target["roots"]),
        "observed_target_root_count"=>length(observed),"matched_target_roots"=>length(matching),"tracked_root_count"=>length(candidates),
        "sample_count"=>17,"intervals_without_continuous_certification"=>16,"initial_track_count"=>length(states),"samples"=>samples,"changes"=>changes,"tracks"=>attempts,"endpoint_matching"=>matching,
        "observed_target_roots"=>observed,"endpoint_clean_tracks"=>candidates,
        "diagnostic_thresholds"=>Dict("large_step"=>0.1,"reverse_return"=>1e-6,"duplicate_root"=>1e-7,"endpoint_match"=>1e-6)))
    Dict("status"=>covered ? "completed" : "unresolved","evidence"=>Dict("recoveries"=>[record]),"followups"=>Any[])
end
function execute_pilot_control(job,config)
    payload=copy(job);delete!(payload,"pilot_role")
    local_config=merge(config,Dict("seed_grid"=>5,"section_angles"=>8,"section_curve_max_steps"=>4,"periodic_max_steps"=>4))
    result=execute_job(payload,local_config)
    kind=job["pilot_control"]
    raw=get(result["evidence"],kind=="hopf" ? "cycles" : "events",[])
    passed=result["status"]=="completed"&&any(row->get(row,"validated",false),raw)
    if passed&&kind!="hopf"&&!isempty(result["followups"])
        section=first(result["followups"])
        section["id"]="$(get(job,"id","control"))-protected-section"
        neighbors=execute_job(section,local_config)
        for (key,value) in neighbors["evidence"]
            append!(get!(result["evidence"],key,Any[]),value)
        end
    end
    result["evidence"]["controls"]=[Dict("control"=>kind,"status"=>passed ? "passed" : "failed",
        "scope"=>"protected_actual_model_control","discovery_yield_eligible"=>false)]
    result["followups"]=Any[]
    result["status"]=passed ? "completed" : "unresolved"
    result
end
function execute_pilot_job(job,config)
    local_config=merge(config,Dict("seed_grid"=>5))
    get(job,"pilot_role","")=="control" && return execute_pilot_control(job,local_config)
    job["kind"]=="pilot_sample" && return execute_discovery(pilot_sample_payload(job),local_config)
    job["kind"]=="pilot_audit" && return execute_pilot_audit(job,local_config)
    job["kind"]=="pilot_recovery" && return execute_pilot_recovery(job,local_config)
    throw(ArgumentError("unknown pilot protocol job"))
end
