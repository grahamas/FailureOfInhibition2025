# Outcome-selected follow-up of the immutable first pilot; not another held-out test.
function targeted_scientific_policy()
    Dict{String,Any}("balance_atol"=>1e-11,"event_atol"=>1e-9,"seed_grid"=>9,
        "max_corrector_iters"=>100,"diagnosis_grids"=>[5,9,13],"tracking_grids"=>[33,65,129],
        "terminal_crosscheck_grid"=>13,"path_inventory_budget"=>256,
        "scalar_fold_iteration_cap"=>64,"scalar_fold_residual_atol"=>1e-12,"scalar_fold_t_width"=>1e-12,
        "large_step"=>0.1,"reverse_return"=>1e-6,"duplicate_root"=>1e-7,"endpoint_match"=>1e-6)
end
function targeted_thin_point(point)
    Dict{String,Any}("id"=>point["id"],"parameters"=>deepcopy(point["parameters"]),"signature"=>point["signature"],
        "roots"=>[Dict(key=>deepcopy(root[key]) for key in ("state","stability","spectrum","residual","determinant") if haskey(root,key)) for root in point["roots"]])
end
function targeted_source_job(packet,id)
    only(filter(job->job["id"]==id,packet["state"]["jobs"]))
end
function targeted_source_point(packet,id)
    result=targeted_read_result(packet,id)
    targeted_thin_point(only(result["evidence"]["points"]))
end
function build_targeted_jobs(packet,config)
    jobs=Dict{String,Any}[]
    overrides=Dict{String,Any}("balance_atol"=>1e-11,"event_atol"=>1e-9,"seed_grid"=>9,"max_corrector_iters"=>100)
    decorate(payload,role,ids)=merge(deepcopy(payload),Dict("target_role"=>role,"source_job_ids"=>ids,
        "source_job_id"=>first(ids),"config_overrides"=>copy(overrides),"study_scope"=>"Outcome-selected diagnostic follow-up; no new held-out validation"))
    for id in ("job-000001","job-000002","job-000003")
        payload=targeted_source_job(packet,id)["payload"]
        get(payload,"pilot_role","")=="control" || throw(ArgumentError("source control mismatch"))
        push!(jobs,decorate(payload,"control",[id]))
    end
    for id in ("job-000414","job-000509")
        source=targeted_source_job(packet,id)["payload"]
        design=only(filter(row->row["cohort"]==source["pilot_cohort"],packet["state"]["pilot"]["manifest"]["cohort_designs"]))
        sample=only(filter(row->row["accepted_index"]==source["pilot_sample_index"],design["samples"]))
        payload=Dict{String,Any}("kind"=>"targeted_diagnosis","queue"=>"discovery","source_payload"=>deepcopy(source),
            "parameters"=>deepcopy(sample["parameters"]),"source_error"=>get(targeted_read_result(packet,id),"error",""))
        source["pilot_chart"]=="equilibrium" && (payload["known_chart_coordinates"]=sample["chart_coordinates"][5:7])
        push!(jobs,decorate(payload,"diagnosis",[id]))
    end
    for id in ("job-000929","job-000941","job-000953","job-000959")
        source=targeted_source_job(packet,id)["payload"]
        payload=Dict{String,Any}("kind"=>"targeted_recovery","queue"=>"events","base"=>deepcopy(source["base"]),
            "target"=>targeted_thin_point(source["target"]),"source_payload"=>deepcopy(source),"target_job_id"=>source["pilot_target_id"],
            "route_selection"=>"Original first-pilot route preserved unchanged")
        prior=only(targeted_read_result(packet,id)["evidence"]["recoveries"])
        payload["original_level17"]=Dict("grid_size"=>17,"base_origin_target_coverage"=>prior["tracked_target_coverage"],
            "sampled_inventory_preserved"=>prior["sampled_inventory_preserved"],"tracks"=>deepcopy(prior["tracks"]),
            "samples"=>targeted_thin_point.(prior["samples"]),"observed_target_root_count"=>prior["observed_target_root_count"],"matched_target_roots"=>prior["matched_target_roots"])
        push!(jobs,decorate(payload,"recovery_refinement",[id,source["pilot_target_id"]]))
    end
    bases=packet["state"]["pilot"]["base_selection"]["entries"]
    length(bases)==67 || throw(ArgumentError("targeted protocol requires the original 67-base frozen pool"))
    for id in ("job-000872","job-000876","job-000877")
        target=targeted_source_point(packet,id)
        base=first(sort(bases;by=base->(pilot_distance(base["parameters"],target["parameters"]),base["point_id"])))
        payload=Dict{String,Any}("kind"=>"targeted_recovery","queue"=>"events","base"=>deepcopy(base),"target"=>target,
            "target_job_id"=>id,"base_pool_size"=>length(bases),"route_selection"=>"Original normalized parameter-only nearest-base rule over the frozen first-pilot pool")
        push!(jobs,decorate(payload,"multiroot",[id,base["source_job"]]))
    end
    for id in ("job-000946","job-000953","job-000954")
        source=targeted_source_job(packet,id)["payload"]
        row=only(targeted_read_result(packet,id)["evidence"]["recoveries"])
        recovery=Dict("samples"=>targeted_thin_point.(row["samples"]),"changes"=>deepcopy(row["changes"]))
        push!(jobs,decorate(Dict("kind"=>"targeted_transition","queue"=>"events","source_payload"=>deepcopy(source),"source_recovery"=>recovery),"transition",[id]))
    end
    for id in ("job-000584","job-000585","job-000598","job-000608","job-000613","job-000627","job-000637")
        source=targeted_source_job(packet,id)["payload"]
        audit=only(targeted_read_result(packet,id)["evidence"]["audits"])
        push!(jobs,decorate(Dict("kind"=>"targeted_continuation","queue"=>"events","source_payload"=>deepcopy(source),
            "source_audit"=>deepcopy(audit["continuation_pair"])),"audit_extension",[id]))
    end
    length(jobs)==22 || error("targeted protocol manifest must contain exactly 22 jobs")
    jobs
end
function targeted_diagnosis_specials(p)
    result=Dict{String,Any}("scope"=>"Analytic consequences only when the exact coupling conditions match")
    if p["B"]==0&&p["C"]==0
        result["populations_decoupled"]=true
        result["E_scalar_contraction_bound"]=p["A"]/8
        result["E_uniqueness_established_by_contraction"]=p["A"]<8
        if p["D"]==0&&p["A"]<8
            result["equilibrium_unique"]=true
            result["argument"]="E=g(AE-a) has derivative bounded by A/8<1; inhibitory occupancy is constant in state"
            result["singular_solver_iterate_is_not_evidence_of_singular_equilibrium"]=true
        end
    end
    result
end
function execute_targeted_diagnosis(job,config)
    p=job["parameters"];extra=haskey(job,"known_chart_coordinates") ? [occupancy(job["known_chart_coordinates"]...)] : Vector{Float64}[]
    variants=Dict{String,Any}[]
    for grid in (5,9,13)
        point=atlas_point(p,merge(config,Dict("seed_grid"=>grid));id="$(get(job,"id","diagnosis"))-grid$grid",extra_seeds=extra)
        push!(variants,Dict("grid_size"=>grid,"point"=>point,
            "singular_solver_attempt_count"=>count(attempt->attempt["solver_status"]=="SingularException",point["attempts"])))
    end
    comparisons=[pilot_search_comparison(variants[index]["point"],variants[index+1]["point"],"grid$(variants[index]["grid_size"])","grid$(variants[index+1]["grid_size"])") for index in 1:2]
    record=Dict("source_job_id"=>job["source_job_id"],"prior_error"=>job["source_error"],"variants"=>variants,"comparisons"=>comparisons,
        "known_chart_seed_included"=>!isempty(extra),"analytic_diagnostics"=>targeted_diagnosis_specials(p),"completeness"=>"CompletenessNotCertified")
    agreed=!isempty(variants[end]["point"]["roots"])&&all(row->row["status"]=="agreement_observed",comparisons)
    record["inventory_agreement"]=agreed
    status=agreed ? "completed" : "unresolved"
    Dict("status"=>status,"evidence"=>Dict("targeted_diagnoses"=>[record]),"followups"=>Any[])
end
function targeted_union_roots(groups)
    observed=Dict{String,Any}[]
    for (label,roots) in groups,root in roots
        index=findfirst(candidate->norm(candidate["state"]-root["state"],Inf)<=1e-7,observed)
        if index===nothing
            push!(observed,Dict("state"=>copy(root["state"]),"sources"=>[label]))
        else
            label in observed[index]["sources"] || push!(observed[index]["sources"],label)
        end
    end
    observed
end
function targeted_track_level(base,target,initial,grid,origin,config,point_ids)
    forward=origin=="base_origin_forward"
    indices=forward ? collect(0:grid-1) : collect(grid-1:-1:0)
    denominator=grid-1;options=atlas_equilibrium_options(config)
    states=[copy(root["state"]) for root in initial];clean=trues(length(states));active=trues(length(states))
    attempts=Dict{String,Any}[];links=Dict{String,Any}[];previous=forward ? base : target
    previous_key=forward ? 0 : 128
    for index in indices
        t=index/denominator;key=div(index*128,denominator)
        p=Dict{String,Any}(coordinate=>(1-t)*base[coordinate]+t*target[coordinate] for coordinate in ATLAS_COORDINATES)
        corrected=Dict{Int,Vector{Float64}}();rows=Dict{Int,Dict{String,Any}}()
        for track in eachindex(states)
            active[track] || continue
            prior=copy(states[track]);was_clean=clean[track]
            attempt=solve_equilibrium(reduced_model(p),prior;options=options).attempt
            row=Dict{String,Any}("origin"=>origin,"track"=>track,"t"=>t,"dyadic_key"=>key,
                "previous_state"=>prior,"forward"=>atlas_attempt(attempt));reasons=String[]
            if attempt.validation!=AdmissibleCandidate
                active[track]=false;clean[track]=false;push!(reasons,"forward_correction_unresolved")
            else
                candidate=copy(attempt.candidate);movement=norm(candidate-prior,Inf);row["movement"]=movement
                movement>0.1 && push!(reasons,"large_step_diagnostic")
                attempt.near_singular && push!(reasons,"near_singular_root")
                reverse=solve_equilibrium(reduced_model(previous),candidate;options=options).attempt
                row["reverse_check"]=atlas_attempt(reverse)
                returned=reverse.validation==AdmissibleCandidate&&norm(reverse.candidate-prior,Inf)<=1e-6
                row["reverse_returned_previous"]=returned
                returned || push!(reasons,"reverse_correction_ambiguous")
                isempty(reasons) || (clean[track]=false)
                states[track]=candidate;corrected[track]=candidate
            end
            row["diagnostics"]=reasons;row["track_clean"]=clean[track];row["previous_track_clean"]=was_clean
            row["from_point_id"]=point_ids[previous_key+1];row["to_point_id"]=point_ids[key+1]
            rows[track]=row
        end
        for first in sort(collect(keys(corrected))),second in sort(collect(keys(corrected)))
            first<second || continue
            if norm(corrected[first]-corrected[second],Inf)<=1e-7
                for track in (first,second)
                    clean[track]=false;push!(rows[track]["diagnostics"],"duplicate_corrected_roots");rows[track]["track_clean"]=false
                end
            end
        end
        for track in sort(collect(keys(rows)))
            row=rows[track];push!(attempts,row)
            if key!=previous_key
                push!(links,Dict("from"=>row["from_point_id"],"to"=>row["to_point_id"],"kind"=>"tracked_root_link",
                    "origin"=>origin,"track"=>track,"grid_size"=>grid,"track_clean"=>row["previous_track_clean"]&&row["track_clean"],
                    "validated"=>row["previous_track_clean"]&&row["track_clean"],"status"=>row["previous_track_clean"]&&row["track_clean"] ? "clean_sampled_corrector_link" : "unresolved",
                    "from_state"=>row["previous_state"],"to_state"=>get(corrected,track,Float64[]),"continuous_coverage_certified"=>false))
            end
        end
        previous=p;previous_key=key
    end
    endpoints=[Dict("track"=>index,"state"=>states[index]) for index in eachindex(states) if active[index]&&clean[index]]
    Dict{String,Any}("origin"=>origin,"grid_size"=>grid,"initial_track_count"=>length(initial),"attempts"=>attempts,
        "endpoint_clean_tracks"=>endpoints,"adjacency_links"=>links,"branch_identity_certified"=>false)
end
function targeted_fold_threshold(A,which)
    A>8 || throw(ArgumentError("scalar fold threshold requires A>8"))
    E=(1+(which=="maximum" ? 1 : -1)*sqrt(1-8/A))/4
    A*E-log(E/(1-2E))
end
function targeted_analytic_path(base,target)
    diagnostics=Dict{String,Any}("scope"=>"Exact special cases only; not certification of the atlas or generic paths")
    if all(base[key]==0&&target[key]==0 for key in ("A","B","C","D"))
        diagnostics["uncoupled_unique_attracting_path"]=true
        diagnostics["argument"]="At every t the responses are constant in state; occupancy=(FE/(1+FE),FI/(1+FI)); J=diag(-(1+FE),-(1+FI)/r) is strictly negative"
        diagnostics["analytic_endpoint_states"]=[occupancy(-p["a"],-p["b"],p["h"]) for p in (base,target)]
    end
    if all(base[key]==0&&target[key]==0 for key in ("B","D"))
        diagnostics["triangular_E_scalar_path"]=true;events=Dict{String,Any}[]
        for interval in 0:15,which in ("minimum","maximum")
            left,right=interval/16,(interval+1)/16
            parameters=t->Dict(coordinate=>(1-t)*base[coordinate]+t*target[coordinate] for coordinate in ATLAS_COORDINATES)
            min(parameters(left)["A"],parameters(right)["A"])>8 || continue
            f=t->parameters(t)["a"]-targeted_fold_threshold(parameters(t)["A"],which)
            a,b=f(left),f(right);a*b<=0 || continue
            iterations=0;endpoint=iszero(a)||iszero(b);nonisolated=iszero(a)&&iszero(b)
            t=left
            if iszero(a)
                t=left
            elseif iszero(b)
                t=right
            else
                for iteration in 1:64
                    iterations=iteration;middle=(left+right)/2;value=f(middle)
                    if abs(value)<=1e-12 || right-left<=1e-12
                        break
                    end
                    if signbit(value)==signbit(a)
                        left=middle;a=value
                    else
                        right=middle;b=value
                    end
                end
                t=(left+right)/2
            end
            push!(events,Dict("kind"=>nonisolated ? "analytic_scalar_turn_locus_segment" : "analytic_scalar_turn_crossing_candidate","turn"=>which,"t"=>t,
                "t_bracket"=>[left,right],"shared_grid_endpoint"=>endpoint,"isolated_crossing_not_established"=>nonisolated,"threshold_residual"=>abs(f(t)),"iterations"=>iterations,"iteration_cap"=>64,
                "original_sample_bracket"=>[interval+1,interval+2],"certified"=>false))
        end
        diagnostics["scalar_fold_windows"]=events
        diagnostics["scalar_window_grouping"]="Raw original brackets retained; shared-endpoint detections are not distinct-event counts"
        diagnostics["alternative_detour_not_run"]=true
    end
    diagnostics
end
function execute_targeted_recovery(job,config)
    base,target=job["base"],job["target"];p0,p1=base["parameters"],target["parameters"]
    independent=Dict{String,Any}[]
    for key in 0:128
        t=key/128;p=Dict{String,Any}(coordinate=>(1-t)*p0[coordinate]+t*p1[coordinate] for coordinate in ATLAS_COORDINATES)
        point=atlas_point(p,merge(config,Dict("seed_grid"=>9));id="$(get(job,"id","targeted"))-independent-$key")
        point["t"]=t;point["inventory_seed_policy"]="Independent grid9 only; no tracked-state seeds"
        push!(independent,point)
    end
    terminal=job["target_role"]=="multiroot" ? atlas_point(p1,merge(config,Dict("seed_grid"=>13));id="$(get(job,"id","targeted"))-terminal-grid13") : nothing
    groups=[("saved_first_pilot",target["roots"]),("independent_grid9",independent[end]["roots"])]
    terminal===nothing || push!(groups,("independent_grid13",terminal["roots"]))
    observed=targeted_union_roots(groups)
    base_observed=targeted_union_roots([("saved_first_pilot",base["roots"]),("independent_grid9",independent[1]["roots"])])
    terminal_comparison=terminal===nothing ? Dict("status"=>"not_in_protocol") : pilot_search_comparison(independent[end],terminal,"independent_grid9","independent_grid13")
    terminal_agreement=terminal===nothing || terminal_comparison["status"]=="agreement_observed"
    ids=getindex.(independent,"id");levels=Dict{String,Any}[]
    for grid in (33,65,129)
        forward=targeted_track_level(p0,p1,base["roots"],grid,"base_origin_forward",config,ids)
        reverse=targeted_track_level(p0,p1,observed,grid,"target_origin_reverse",config,ids)
        forward_match=pilot_match_roots(observed,forward["endpoint_clean_tracks"])
        reverse_match=pilot_match_roots(base_observed,reverse["endpoint_clean_tracks"])
        push!(levels,Dict("grid_size"=>grid,"base_origin_forward"=>forward,"target_origin_reverse"=>reverse,
            "base_origin_target_coverage"=>!isempty(observed)&&length(forward_match)==length(observed),
            "reverse_target_origin_base_coverage"=>!isempty(base_observed)&&length(reverse_match)==length(base_observed),
            "forward_endpoint_matching"=>forward_match,"reverse_endpoint_matching"=>reverse_match))
    end
    edges=Dict{String,Any}[];changes=Dict{String,Any}[]
    for index in 2:129
        a,b=independent[index-1:index]
        resolved=!isempty(a["roots"])&&!isempty(b["roots"])&&!occursin("unresolved",lowercase(a["signature"]*b["signature"]))
        equal=a["signature"]==b["signature"]
        edge=Dict{String,Any}("from"=>a["id"],"to"=>b["id"],"kind"=>!resolved ? "unknown_connection" : equal ? "finite_same_signature_samples" : "sampled_inventory_change",
            "origin"=>"independent_grid9","status"=>resolved ? "sampled_endpoints_only" : "unresolved","validated"=>false,"continuous_coverage_certified"=>false)
        push!(edges,edge);equal || push!(changes,edge)
    end
    whole_path_preserved=terminal_agreement&&length(independent)==129&&length(unique(ids))==129&&all(point->!isempty(point["roots"])&&!occursin("unresolved",lowercase(point["signature"]))&&point["signature"]==independent[1]["signature"],independent)
    if whole_path_preserved
        push!(edges,Dict("from"=>first(ids),"to"=>last(ids),"kind"=>"sampled_inventory_preserved","origin"=>"independent_grid9",
            "status"=>"completed","validated"=>true,"scope"=>"finite_declared_samples_only","sample_ids"=>ids,
            "declared_sample_count"=>129,"continuous_coverage_certified"=>false))
    end
    append!(edges,levels[end]["base_origin_forward"]["adjacency_links"])
    append!(edges,levels[end]["target_origin_reverse"]["adjacency_links"])
    covered=levels[end]["base_origin_target_coverage"]
    record=Dict{String,Any}("source_job_id"=>job["source_job_id"],"target_job_id"=>job["target_job_id"],"base_point_id"=>base["point_id"],
        "levels"=>levels,"finest_level"=>129,"base_origin_target_coverage"=>covered,"sampled_inventory_preserved"=>whole_path_preserved,
        "reverse_target_origin_base_coverage"=>levels[end]["reverse_target_origin_base_coverage"],
        "independent_endpoint_solved"=>!isempty(independent[end]["roots"]),"independent_points"=>independent,
        "observed_target_roots"=>observed,"saved_target_root_count"=>length(target["roots"]),"observed_target_root_count"=>length(observed),
        "inventory_locations"=>129,"inventory_evaluations"=>terminal===nothing ? 129 : 130,"inventory_location_budget"=>256,
        "terminal_grid13_assessed"=>terminal!==nothing,"terminal_grid_comparison"=>terminal_comparison,"terminal_inventory_agreement"=>terminal_agreement,"changes"=>changes,"adjacency"=>Dict("nodes"=>independent,"edges"=>edges),
        "analytic_diagnostics"=>targeted_analytic_path(p0,p1),"scope"=>"Outcome-selected sampled tracking; target-origin reverse tracks never count as base-origin coverage; no certified continuous coverage")
    terminal===nothing || (record["terminal_grid13"]=terminal)
    record["original_level17_available"]=haskey(job,"original_level17")
    haskey(job,"original_level17") && (record["original_level17"]=job["original_level17"])
    Dict("status"=>covered&&terminal_agreement ? "completed" : "unresolved","evidence"=>Dict("targeted_recoveries"=>[record]),"followups"=>Any[])
end
function execute_targeted_job(job,config)
    effective=merge(config,Dict("balance_atol"=>1e-11,"event_atol"=>1e-9,"seed_grid"=>9,"max_corrector_iters"=>100))
    job["kind"]=="targeted_diagnosis" && return execute_targeted_diagnosis(job,effective)
    job["kind"]=="targeted_recovery" && return execute_targeted_recovery(job,effective)
    job["kind"]=="targeted_transition" && return execute_targeted_transition(job,effective)
    job["kind"]=="targeted_continuation" && return execute_targeted_continuation(job,effective)
    throw(ArgumentError("unsupported targeted job"))
end
