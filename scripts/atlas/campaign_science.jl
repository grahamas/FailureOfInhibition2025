# Scientific adapters for a finite campaign. No handler certifies completeness.
const CAMPAIGN_SCIENCE_POLICY = Dict{String,Any}(
    "discovery_grid"=>9,"audit_grid"=>13,"path_samples"=>129,
    "path_localization_split_cap"=>12,"path_corrector_cap"=>64,
    "path_inventory_location_cap"=>256,"path_width_target"=>1e-4,
    "recovery_grids"=>[33,65,129],"transverse_offsets"=>[0.01,0.005])

campaign_science_settings(config,grid=9)=merge(config,Dict{String,Any}(
    "seed_grid"=>grid,"balance_atol"=>min(1e-11,atlas_balance_tolerance(config)),"event_atol"=>min(1e-9,atlas_event_tolerance(config,1e-9)),"max_corrector_iters"=>100))
campaign_point_resolved(point)=!isempty(get(point,"roots",[])) &&
    !occursin("unresolved",lowercase(get(point,"signature","unresolved")))
function campaign_science_point(parameters,config;id,point_solver=atlas_point,extra_seeds=Vector{Float64}[])
    validate_parameters(parameters)
    try
        point_solver(parameters,config;id=id,extra_seeds=extra_seeds)
    catch err
        (err isa SingularException || err isa DomainError || err isa OverflowError || err isa LinearAlgebra.LAPACKException) || rethrow()
        Dict{String,Any}("id"=>id,"parameters"=>parameters,"roots"=>Any[],"signature"=>"unresolved_solver_failure",
            "attempts"=>Any[],"error"=>sprint(showerror,err),"completeness"=>"CompletenessNotCertified",
            "coverage"=>"failed_sampled_point")
    end
end
function campaign_science_result(key,record;points=Any[],paths=Any[],events=Any[],status="completed")
    record["completeness"]="CompletenessNotCertified"
    record["continuous_coverage_certified"]=false
    record["status"]=get(record,"status",status)
    aliases=Dict("campaign_audits"=>"audits","campaign_recoveries"=>"recoveries","campaign_validations"=>"validations","campaign_paths"=>"campaign_paths","geometry_probes"=>"geometry_probes")
    Dict{String,Any}("status"=>status,"evidence"=>Dict{String,Any}(get(aliases,key,key)=>[record],"points"=>points,"paths"=>paths,"events"=>events),"followups"=>Any[])
end
# Reject both many-to-one and one-to-many near-root ambiguities explicitly.
function campaign_unique_matching(targets,candidates;tolerance=1e-6)
    eligible=[[index for (index,candidate) in enumerate(candidates) if norm(target["state"]-candidate["state"],Inf)<=tolerance] for target in targets]
    uses=Dict{Int,Int}()
    for indices in eligible,index in indices
        uses[index]=get(uses,index,0)+1
    end
    matches=[Dict("target_root"=>index,"candidate_root"=>only(indices)) for (index,indices) in enumerate(eligible) if length(indices)==1 && uses[only(indices)]==1]
    ambiguous=[index for (index,indices) in enumerate(eligible) if length(indices)>1 || any(candidate->uses[candidate]>1,indices)]
    Dict{String,Any}("matches"=>matches,"ambiguous_target_indices"=>ambiguous,"eligible_candidates"=>eligible,
        "complete"=>!isempty(targets)&&length(matches)==length(targets)&&isempty(ambiguous),"tolerance"=>tolerance)
end
# Same-inventory roots are always kept separate. Only unique one-to-one
# cross-inventory matches may identify repeated observations.
function campaign_observed_roots(groups;tolerance=1e-7)
    observed=Dict{String,Any}[];ambiguous=false;sources=Any[]
    for (label,roots) in groups
        push!(sources,Dict("source"=>label,"roots"=>deepcopy(roots)))
        eligible=[[index for (index,candidate) in enumerate(observed) if norm(root["state"]-candidate["state"],Inf)<=tolerance] for root in roots]
        usage=Dict{Int,Int}()
        for indices in eligible,index in indices
            usage[index]=get(usage,index,0)+1
        end
        # Near roots within the same inventory are an unresolved ambiguity.
        ambiguous |= any(norm(roots[i]["state"]-roots[j]["state"],Inf)<=tolerance for i in eachindex(roots) for j in (i+1):length(roots))
        for (root,indices) in zip(roots,eligible)
            if length(indices)==1 && usage[only(indices)]==1
                push!(observed[only(indices)]["sources"],label)
            else
                ambiguous |= !isempty(indices)
                push!(observed,Dict("state"=>copy(root["state"]),"sources"=>[label]))
            end
        end
    end
    Dict{String,Any}("roots"=>observed,"ambiguous"=>ambiguous,"source_inventories"=>sources,"cross_inventory_tolerance"=>tolerance)
end
function campaign_search_comparison(first,second,label_first,label_second)
    matching=campaign_unique_matching(first["roots"],second["roots"];tolerance=1e-7)
    parameters_match=all(first["parameters"][key]==second["parameters"][key] for key in ATLAS_COORDINATES)
    inventory_match=matching["complete"] && length(first["roots"])==length(second["roots"])
    resolved=campaign_point_resolved(first)&&campaign_point_resolved(second)
    agreed=parameters_match&&inventory_match&&resolved&&first["signature"]==second["signature"]
    Dict{String,Any}("pair"=>[label_first,label_second],"status"=>agreed ? "agreement_observed" : "disagreement_or_unresolved",
        "parameters_match"=>parameters_match,"inventory_match"=>inventory_match,"classification_resolved"=>resolved,"first_root_count"=>length(first["roots"]),
        "second_root_count"=>length(second["roots"]),"matched_roots"=>length(matching["matches"]),"matching_diagnostics"=>matching)
end
function campaign_job_parameters(job)
    parameters=get(job,"parameters",get(job,"params",nothing))
    parameters===nothing && haskey(job,"training_point") && (parameters=job["training_point"]["parameters"])
    parameters===nothing && throw(ArgumentError("campaign job requires parameters"))
    validate_parameters(parameters)
end
function campaign_inventory_pair(job,config;point_solver=atlas_point)
    p=campaign_job_parameters(job);id=get(job,"id","campaign-pair")
    extra=get(job,"campaign_chart",get(job,"chart",""))=="equilibrium" && haskey(job,"chart_coordinates") ? [occupancy(job["chart_coordinates"][5:7]...)] : Vector{Float64}[]
    points=[campaign_science_point(p,campaign_science_settings(config,grid);id="$id-grid$grid",point_solver=point_solver,extra_seeds=extra) for grid in (9,13)]
    comparison=campaign_search_comparison(points[1],points[2],"independent_grid9","independent_grid13")
    record=Dict{String,Any}("source_job_id"=>get(job,"source_job_id",""),"parameters"=>p,
        "variants"=>[Dict("grid_size"=>grid,"point_id"=>point["id"]) for (grid,point) in zip((9,13),points)],
        "comparisons"=>[comparison],"comparison"=>comparison,
        "inventory_agreement"=>comparison["status"]=="agreement_observed",
        "scope"=>"Independent fixed-parameter finite seed grids; agreement does not establish root completeness")
    source=get(job,"training_point",get(job,"source_point",nothing))
    if source!==nothing
        record["training_repeat_comparison"]=campaign_search_comparison(source,points[1],"saved_training","independent_grid9")
        push!(record["comparisons"],campaign_search_comparison(source,points[2],"saved_training","independent_grid13"))
        record["inventory_agreement"] &= record["training_repeat_comparison"]["status"]=="agreement_observed"
    end
    record["inventory_agreement"] &= all(row->row["status"]=="agreement_observed",record["comparisons"])
    record["status"]=record["inventory_agreement"] ? "agreement_observed" : "disagreement_or_unresolved"
    record,points
end
function execute_campaign_audit(job,config;point_solver=atlas_point)
    record,points=campaign_inventory_pair(job,config;point_solver=point_solver)
    if !haskey(job,"training_point") || get(job,"training_status","completed")!="completed"
        record["inventory_agreement"]=false;record["status"]="disagreement_or_unresolved"
        record["unresolved_reason"]="saved_training_inventory_unavailable_or_failed"
    end
    campaign_science_result("campaign_audits",record;points=points,status=record["inventory_agreement"] ? "completed" : "unresolved")
end
function execute_campaign_validation(job,config;point_solver=atlas_point)
    record,points=campaign_inventory_pair(job,config;point_solver=point_solver)
    record["route_selection_scope"]="Frozen parameter-only route; validation inventories never select a base"
    campaign_science_result("campaign_validations",record;points=points,status=record["inventory_agreement"] ? "completed" : "unresolved")
end

function campaign_path_endpoints(job)
    left=get(job,"from",get(job,"from_point",get(job,"first",get(job,"base",nothing))))
    right=get(job,"to",get(job,"to_point",get(job,"last",get(job,"target",nothing))))
    (left===nothing || right===nothing) && throw(ArgumentError("campaign path requires from_point/to_point"))
    validate_parameters(left["parameters"]);validate_parameters(right["parameters"])
    left,right
end
function campaign_path_edge(job,left,right,points)
    ids=getindex.(points,"id")
    preserved=length(points)==129 && length(unique(ids))==129 && all(campaign_point_resolved,points) &&
        all(point->point["signature"]==points[1]["signature"],points)
    Dict{String,Any}("from"=>get(left,"point_id",get(left,"id","")),"to"=>get(right,"point_id",get(right,"id","")),
        "kind"=>preserved ? "sampled_inventory_preserved" : "unknown_connection",
        "signature_preserved"=>preserved,"validated"=>preserved,"status"=>preserved ? "completed" : "unresolved",
        "sample_ids"=>ids,"declared_sample_count"=>129,"coverage"=>"129_sample_full_inventory_path",
        "continuous_coverage_certified"=>false,"scope"=>"Finite sampled connection only; equal signature alone does not merge components")
end

function execute_campaign_path(job,config;point_solver=atlas_point,corrector=targeted_corrector,track_solver=targeted_track_level)
    left,right=campaign_path_endpoints(job);p0,p1=left["parameters"],right["parameters"]
    id=get(job,"id","campaign-path");points=Dict{String,Any}[];cache=Dict{Float64,Dict{Int,Dict{String,Any}}}()
    function inventory(t,grid=9)
        position=Float64(t)
        if !haskey(cache,position)
            length(cache)<256 || return nothing
            cache[position]=Dict{Int,Dict{String,Any}}()
        end
        haskey(cache[position],grid) && return cache[position][grid]
        point=campaign_science_point(targeted_affine_parameters(p0,p1,position),campaign_science_settings(config,grid);
            id="$id-t$position-grid$grid",point_solver=point_solver)
        point["affine_t"]=position;point["inventory_grid"]=grid;point["audit_only"]=grid==13
        point["inventory_seed_policy"]="Independent grid only; no tracked-state seeds"
        cache[position][grid]=point;push!(points,point);point
    end
    samples=[inventory(index/128) for index in 0:128]
    edge=campaign_path_edge(job,left,right,samples)
    endpoint_comparisons=[campaign_search_comparison(left,samples[1],"saved_from","grid9"),campaign_search_comparison(right,samples[end],"saved_to","grid9")]
    for t in (0.0,1.0)
        push!(endpoint_comparisons,campaign_search_comparison(inventory(t),inventory(t,13),"grid9","grid13"))
    end
    audit_agreement=all(row->row["status"]=="agreement_observed",endpoint_comparisons)
    forward=track_solver(p0,p1,left["roots"],129,"base_origin_forward",campaign_science_settings(config),getindex.(samples,"id"))
    failures=String[]
    for point in samples
        rows=filter(row->get(row,"to_point_id","")==point["id"],forward["attempts"])
        tracks=[Dict("state"=>row["forward"]["candidate"]) for row in rows if get(row,"track_clean",false)]
        matching=campaign_unique_matching(point["roots"],tracks)
        matching["complete"] && length(tracks)==length(point["roots"])==length(left["roots"]) || push!(failures,"unresolved_or_ambiguous_tracking_at:"*point["id"])
    end
    edge["forward_tracking_unambiguous"]=isempty(failures)
    edge["audit_agreement"]=audit_agreement
    edge["tracking_guard_failures"]=failures
    edge["validated"] &= audit_agreement && isempty(failures)
    edge["signature_preserved"]=edge["validated"]
    edge["status"]=edge["validated"] ? "completed" : "unresolved"
    edge["kind"]=edge["validated"] ? "sampled_inventory_preserved" : "unknown_connection"
    edge["fresh_endpoint_sample_ids"]=[samples[1]["id"],samples[end]["id"]]
    for t in (0.0,1.0),grid in (9,13)
        inventory(t,grid)["audit_only"]=true
    end
    # The graph already owns original endpoint inventories. Independent repeats
    # remain separately saved; audited equality authorizes these endpoint aliases.
    edge["sample_ids"][1]=edge["from"];edge["sample_ids"][end]=edge["to"]
    intervals=Dict{String,Any}[];terminal=Dict{String,Any}[];splits=0
    pending=[(index/128,(index+1)/128) for index in 0:127 if samples[index+1]["signature"]!=samples[index+2]["signature"]]
    while !isempty(pending)
        sort!(pending;by=pair->(-(pair[2]-pair[1]),pair[1]));a,b=popfirst!(pending)
        first,last=inventory(a),inventory(b)
        first["signature"]==last["signature"] && continue
        reason=b-a<=1e-4 ? "width_target" : splits>=12 ? "split_cap" : length(cache)>=256 ? "inventory_cap" : ""
        if isempty(reason)
            midpoint=inventory((a+b)/2)
            if midpoint!==nothing
                splits+=1;push!(pending,(a,(a+b)/2),((a+b)/2,b));continue
            end
            reason="inventory_cap"
        end
        densefirst,denselast=inventory(a,13),inventory(b,13)
        comparisons=[campaign_search_comparison(first,densefirst,"grid9","grid13"),campaign_search_comparison(last,denselast,"grid9","grid13")]
        agreement=all(row->row["status"]=="agreement_observed",comparisons)
        row=Dict{String,Any}("id"=>targeted_interval_id(job,a,b),"t_bounds"=>[a,b],"from"=>first["id"],"to"=>last["id"],
            "status"=>agreement ? "signature_change_observed" : "crosscheck_disagreement","crosscheck_agreement"=>agreement,
            "grid13_comparisons"=>comparisons,"localization_stop_reason"=>reason,"localization_width_target_met"=>b-a<=1e-4,
            "linked_event_ids"=>String[],"corrector_attempts"=>Any[],"validated"=>false,"continuous_coverage_certified"=>false)
        push!(intervals,row);push!(terminal,row)
    end
    events=Dict{String,Any}[];correctors=0
    for seed_index in 1:4,interval in terminal,kind in ("fold","trace_zero")
        correctors>=64 && break
        a,b=interval["t_bounds"];seeds=targeted_event_seeds(inventory(a),inventory(b),kind)
        length(seeds)>=seed_index || continue
        correctors+=1;coarse=corrector(p0,p1,seeds[seed_index],kind,(a,b))
        attempt=Dict{String,Any}("kind"=>kind,"seed"=>seeds[seed_index],"coarse"=>coarse);push!(interval["corrector_attempts"],attempt)
        coarse["converged"] || continue
        refined=if correctors<64
            correctors+=1;corrector(p0,p1,coarse["coordinates"],kind,(a,b);tight=true)
        else
            Dict{String,Any}("converged"=>false,"reason"=>"corrector_cap")
        end
        attempt["refined"]=refined;refined["converged"] || continue
        z=refined["coordinates"];coefficients=kind=="fold" ? targeted_fold_validation(p0,p1,z,coarse["coordinates"],interval["crosscheck_agreement"]) : Dict{String,Any}()
        validated=kind=="fold" && get(coefficients,"validated",false)
        event=Dict{String,Any}("id"=>"$id-event$(length(events)+1)","source_job_id"=>id,"source_interval_id"=>interval["id"],
            "kind"=>kind,"state"=>copy(z[1:2]),"affine_t"=>z[3],"parameters"=>targeted_affine_parameters(p0,p1,z[3]),
            "validated"=>validated,"status"=>validated ? "completed" : "candidate","coefficients"=>coefficients,
            "balance_residual"=>refined["balance_residual"],"event_condition_residual"=>refined["event_condition_residual"],
            "scope"=>kind=="fold" ? "Numerical fold on bounded affine interval" : "Trace-zero candidate only; no Hopf classification")
        push!(events,event);push!(interval["linked_event_ids"],event["id"])
    end
    for interval in terminal
        a,b=interval["t_bounds"]
        required=sum(length(targeted_event_seeds(inventory(a),inventory(b),kind)) for kind in ("fold","trace_zero"))
        attempts=interval["corrector_attempts"]
        incomplete=length(attempts)<required || any(attempt->get(get(attempt,"refined",Dict()),"reason","")=="corrector_cap",attempts)
        interval["corrector_status"]=incomplete ? "unresolved_corrector_cap" : "declared_seed_attempts_finished"
    end
    record=Dict{String,Any}("from"=>edge["from"],"to"=>edge["to"],"sample_point_ids"=>getindex.(samples,"id"),"sampled_inventory_preserved"=>edge["validated"],
        "forward_tracking"=>forward,"endpoint_comparisons"=>endpoint_comparisons,
        "endpoint_repeats"=>[Dict("original_point_id"=>get(source,"id",""),"fresh_sample_ids"=>[inventory(t,9)["id"],inventory(t,13)["id"]]) for (t,source) in ((0.,left),(1.,right))],"tracking_guard_failures"=>failures,
        "interval_ids"=>getindex.(intervals,"id"),"event_ids"=>getindex.(events,"id"),"inventory_location_count"=>length(cache),"inventory_solve_count"=>length(points),
        "localization_split_count"=>splits,"localization_split_cap"=>12,"corrector_count"=>correctors,"corrector_cap"=>64,
        "inventory_location_cap"=>256,"scope"=>"129 independent inventories plus bounded refinement of observed changes only; unchanged intervals remain uncertified")
    campaign_science_result("campaign_paths",record;points=points,paths=vcat([edge],intervals),events=events)
end

function execute_campaign_recovery(job,config;point_solver=atlas_point,track_solver=targeted_track_level)
    base=get(job,"base",nothing);target=get(job,"target",nothing)
    if target===nothing && haskey(job,"target_parameters")
        target=Dict("id"=>get(job,"campaign_target_job",""),"parameters"=>job["target_parameters"])
    end
    record=Dict{String,Any}("status"=>"unavailable_target_or_base","base_origin_target_coverage"=>false,"tracked_target_coverage"=>false)
    if base===nothing || target===nothing
        record["unavailable_reason"]=get(job,"unavailable_reason","missing_target_or_base")
        return campaign_science_result("campaign_recoveries",record;status="unresolved")
    end
    p0,p1=base["parameters"],target["parameters"];id=get(job,"id","campaign-recovery")
    points=[campaign_science_point(targeted_affine_parameters(p0,p1,key/128),campaign_science_settings(config,9);
        id="$id-inventory$key",point_solver=point_solver) for key in 0:128]
    terminal=campaign_science_point(p1,campaign_science_settings(config,13);id="$id-target-grid13",point_solver=point_solver)
    ids=getindex.(points,"id")
    # Complete every forward solve before inspecting held-out root inventories.
    forwards=[track_solver(p0,p1,base["roots"],grid,"base_origin_forward",campaign_science_settings(config),ids) for grid in (33,65,129)]
    saved=get(get(job,"validation_evidence",Dict()),"points",Any[])
    groups=[("validation_$index",point["roots"]) for (index,point) in enumerate(saved)]
    haskey(target,"roots") && push!(groups,("saved_validation",target["roots"]))
    append!(groups,[("independent_grid9",points[end]["roots"]),("independent_grid13",terminal["roots"])])
    observations=campaign_observed_roots(groups);observed=observations["roots"]
    comparison=campaign_search_comparison(points[end],terminal,"independent_grid9","independent_grid13")
    levels=Dict{String,Any}[]
    for (grid,forward) in zip((33,65,129),forwards)
        matching_record=campaign_unique_matching(observed,forward["endpoint_clean_tracks"])
        matching=matching_record["matches"]
        push!(levels,Dict{String,Any}("grid_size"=>grid,"base_origin_forward"=>forward,"forward_endpoint_matching"=>matching,
            "matching_diagnostics"=>matching_record,"base_origin_target_coverage"=>matching_record["complete"]&&!observations["ambiguous"]))
    end
    covered=levels[end]["base_origin_target_coverage"] && comparison["status"]=="agreement_observed" && get(job,"validation_inventory_status","completed")=="completed"
    merge!(record,Dict("status"=>covered ? "sampled_tracked_target_coverage" : "tracked_coverage_unresolved",
        "base_point_id"=>get(base,"point_id",get(base,"id","")),"target_point_id"=>get(target,"id",""),
        "levels"=>levels,"finest_level"=>129,"base_origin_target_coverage"=>covered,"tracked_target_coverage"=>covered,
        "observed_target_roots"=>observed,"observed_target_root_count"=>length(observed),"observed_root_diagnostics"=>observations,
        "saved_root_observation_count"=>haskey(target,"roots") ? length(target["roots"]) : sum(length(point["roots"]) for point in saved;init=0),
        "saved_inventory_root_counts"=>[Dict("point_id"=>point["id"],"root_count"=>length(point["roots"])) for point in saved],
        "matched_target_roots"=>length(levels[end]["forward_endpoint_matching"]),"initial_track_count"=>length(base["roots"]),
        "terminal_grid_comparison"=>comparison,"terminal_grid13_point_id"=>terminal["id"],"independent_point_ids"=>getindex.(points,"id"),
        "inventory_location_count"=>129,"inventory_solve_count"=>130,"route_selection_scope"=>"Frozen parameter-only base; no validation-outcome routing",
        "scope"=>"Forward base-origin tracks only; every observed endpoint root, including repelling and saddle roots, enters the denominator; branch identity remains uncertified"))
    campaign_science_result("campaign_recoveries",record;points=vcat(points,[terminal]),paths=levels[end]["base_origin_forward"]["adjacency_links"],status=covered ? "completed" : "unresolved")
end

function execute_campaign_geometry_probe(job,config;point_solver=atlas_point)
    nexus=get(job,"nexus_evidence",get(job,"nexus",job))
    parameters=campaign_job_parameters(job)
    get(nexus,"validated",false) || throw(ArgumentError("geometry probes require a validated nexus"))
    pair=String.(job["parameter_pair"]);coordinate=String(job["transverse_coordinate"])
    coordinate in ATLAS_COORDINATES[1:7] && !(coordinate in pair) || throw(ArgumentError("probe must use remaining geometry coordinate"))
    offset=job["normalized_transverse_offset"]
    offset in (-0.01,-0.005,0.005,0.01) || throw(ArgumentError("undeclared transverse offset"))
    p=copy(parameters);p[coordinate]+=offset*max(1,abs(parameters[coordinate]))
    row=Dict{String,Any}("coordinate"=>coordinate,"normalized_transverse_offset"=>offset,"parameters"=>p)
    points=Any[]
    if !atlas_parameter_admissible(p)
        row["status"]="outside_admissible_domain";row["resolved"]=false
    else
        point=campaign_science_point(p,campaign_science_settings(config);id="$(get(job,"id","geometry-probe"))-inventory",point_solver=point_solver)
        push!(points,point);row["point_id"]=point["id"];row["resolved"]=campaign_point_resolved(point)
        row["status"]=row["resolved"] ? "sampled_inventory" : "unresolved_inventory"
    end
    record=Dict{String,Any}("source_nexus_id"=>get(nexus,"id",get(job,"campaign_parent_job","")),
        "parameter_pair"=>pair,"transverse_probes"=>[row],"examined_geometry_directions"=>[coordinate],
        "unexamined_geometry_directions"=>[name for name in ATLAS_COORDINATES[1:7] if !(name in pair) && name!=coordinate],
        "unresolved_sides"=>row["resolved"] ? Any[] : [row],
        "scope"=>"One declared signed transverse probe; companion jobs examine other sides. This record does not establish complete neighborhood coverage")
    campaign_science_result("geometry_probes",record;points=points,status=row["resolved"] ? "completed" : "unresolved")
end
function execute_campaign_job(job,config;kwargs...)
    kind=job["kind"]
    kind=="campaign_audit" && return execute_campaign_audit(job,config;kwargs...)
    kind=="campaign_path" && return execute_campaign_path(job,config;kwargs...)
    kind=="campaign_validation" && return execute_campaign_validation(job,config;kwargs...)
    kind=="campaign_recovery" && return execute_campaign_recovery(job,config;kwargs...)
    kind=="campaign_geometry_probe" && return execute_campaign_geometry_probe(job,config;kwargs...)
    throw(ArgumentError("unsupported campaign scientific job: $kind"))
end
