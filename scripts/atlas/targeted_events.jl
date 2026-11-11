# Predeclared follow-up to three saved paths and seven paired continuations.
# These handlers never read or mutate the source packet. The builder embeds the
# frozen source records, whose hashes belong to the protocol manifest.
const TARGETED_TRANSITION_BRACKETS = Dict(
    "job-000946"=>[(4,5),(5,6),(13,14)],
    "job-000953"=>[(13,14),(14,15)],
    "job-000954"=>[(5,6)])
const TARGETED_CONTINUATION_SOURCES = Set("job-"*lpad(string(i),6,'0') for i in (584,585,598,608,613,627,637))

function targeted_affine_parameters(first,last,t)
    0<=t<=1 || throw(ArgumentError("affine path coordinate must lie in [0,1]"))
    Dict{String,Any}(key=>(1-t)*first[key]+t*last[key] for key in ATLAS_COORDINATES)
end
function targeted_source_id(job)
    source=String(job["source_job_id"])
    startswith(source,"job-") || throw(ArgumentError("source job identifier required"))
    source
end
function targeted_interval_id(job,a,b)
    "$(get(job,"id","targeted"))-interval-$(string(a))-$(string(b))"
end
function targeted_event_residual(first,last,z,kind)
    parameters=targeted_affine_parameters(first,last,z[3])
    K=balance_jacobian(parameters,z[1:2])
    J=Diagonal([1.0,1/parameters["r"]])*K
    vcat(balance_value(parameters,z[1:2]),kind=="fold" ? det(K) : tr(J))
end

function targeted_fold_coefficients(first,last,z;step=1e-3,path_step=step)
    if !(0<=z[3]-path_step && z[3]+path_step<=1)
        return Dict{String,Any}("resolved"=>false,"reason"=>"centered_path_derivative_outside_frozen_path",
            "state_step"=>step,"path_step"=>path_step)
    end
    parameters=targeted_affine_parameters(first,last,z[3])
    state=copy(z[1:2]);K=balance_jacobian(parameters,state)
    J=Diagonal([1.0,1/parameters["r"]])*K
    S=svd(J);q=atlas_orient(S.V[:,end]);p=S.U[:,end]
    pairing=dot(p,q)
    abs(pairing)>1e-7 || return Dict{String,Any}("resolved"=>false,"reason"=>"unresolved_zero_eigenvalue_pairing")
    p/=pairing
    B,_=atlas_tensors(point->atlas_jac(parameters,point),state;step=step)
    plus=targeted_affine_parameters(first,last,z[3]+path_step)
    minus=targeted_affine_parameters(first,last,z[3]-path_step)
    path_derivative=(atlas_rhs(plus,state)-atlas_rhs(minus,state))/(2path_step)
    spectrum=eigvals(J)
    right_residual=norm(J*q,Inf)/norm(q,Inf)
    left_residual=norm(J'*p,Inf)/norm(p,Inf)
    simple=abs(tr(J))>1e-7 && S.S[1]>1e-7 && S.S[2]<=1e-7 &&
        right_residual<=1e-7 && left_residual<=1e-7
    Dict{String,Any}("resolved"=>simple,"quadratic"=>dot(p,B(q,q))/2,
        "path_transversality"=>dot(p,path_derivative),"singular_values"=>S.S,
        "eigenvalue_real"=>real.(spectrum),"eigenvalue_imag"=>imag.(spectrum),
        "null_residual"=>right_residual,"left_null_residual"=>left_residual,"left_right_pairing"=>pairing,
        "trace"=>tr(J),"state_step"=>step,"path_step"=>path_step,"vector_field"=>"original_time_rhs")
end

function targeted_corrector(first,last,seed,kind,bounds;tight=false)
    balance_tolerance=tight ? 1e-12 : 1e-11
    event_tolerance=tight ? 1e-10 : 1e-9
    scale=[balance_tolerance,balance_tolerance,event_tolerance]
    residual=coordinates->targeted_event_residual(first,last,coordinates,kind)./scale
    admissible=coordinates->all(isfinite,coordinates) && all(0 .<= coordinates[1:2] .<= 0.5) &&
        bounds[1]<=coordinates[3]<=bounds[2]
    coordinates,converged,history=atlas_newton(residual,seed,admissible;iterations=16,atol=1.0)
    independent=targeted_event_residual(first,last,coordinates,kind)
    accepted=converged && admissible(coordinates) && all(isfinite,independent) &&
        maximum(abs,independent[1:2])<=balance_tolerance && abs(independent[3])<=event_tolerance
    Dict{String,Any}("coordinates"=>coordinates,"converged"=>accepted,"attempts"=>history,
        "balance_residual"=>maximum(abs,independent[1:2]),"event_condition_residual"=>abs(independent[3]),
        "balance_atol"=>balance_tolerance,"event_atol"=>event_tolerance)
end

function targeted_fold_validation(first,last,z,coarse_coordinates,crosscheck_agreement)
    coarse=targeted_fold_coefficients(first,last,z)
    fine=targeted_fold_coefficients(first,last,z;step=5e-4)
    residual=targeted_event_residual(first,last,z,"fold")
    simple=get(coarse,"resolved",false)&&get(fine,"resolved",false)
    quadratic=simple&&atlas_coefficient_resolved(coarse["quadratic"],fine["quadratic"])
    transverse=simple&&atlas_coefficient_resolved(coarse["path_transversality"],fine["path_transversality"])
    accepted=all(isfinite,residual)&&maximum(abs,residual[1:2])<=1e-11&&abs(residual[3])<=1e-9&&
        norm(z-coarse_coordinates,Inf)<=1e-6&&crosscheck_agreement&&simple&&quadratic&&transverse
    Dict{String,Any}("validated"=>accepted,"coarse"=>coarse,"refined"=>fine,
        "simple_zero_refinement_passed"=>simple,"quadratic_refinement_passed"=>quadratic,
        "transversality_refinement_passed"=>transverse,"crosscheck_agreement"=>crosscheck_agreement)
end

function targeted_event_seeds(left,right,kind)
    seeds=Tuple{Float64,Vector{Float64}}[]
    for point in (left,right)
        for root in point["roots"]
            K=balance_jacobian(point["parameters"],root["state"])
            J=Diagonal([1.0,1/point["parameters"]["r"]])*K
            score=abs(kind=="fold" ? det(K) : tr(J))
            push!(seeds,(score,vcat(Float64.(root["state"]),point["affine_t"])))
        end
    end
    sort!(seeds;by=entry->(entry[1],Tuple(entry[2])))
    selected=Vector{Float64}[]
    for (_,seed) in seeds
        any(old->norm(old[1:2]-seed[1:2],Inf)<=1e-7,selected) && continue
        push!(selected,seed)
        length(selected)==4 && break
    end
    selected
end

function targeted_repeat_groups(estimates;time_tolerance=1e-6,state_tolerance=1e-6)
    memberships=Vector{Vector{Int}}()
    order=sortperm(estimates;by=estimate->(estimate["kind"],estimate["affine_t"],Tuple(estimate["state"]),estimate["id"]))
    for index in order
        estimate=estimates[index]
        eligible=findall(members->all(other->estimates[other]["kind"]==estimate["kind"] &&
            abs(estimates[other]["affine_t"]-estimate["affine_t"])<=time_tolerance &&
            norm(estimates[other]["state"]-estimate["state"],Inf)<=state_tolerance,members),memberships)
        if isempty(eligible)
            push!(memberships,[index])
        else
            push!(memberships[first(eligible)],index)
        end
    end
    groups=Dict{String,Any}[];mapping=Dict{String,String}()
    for (group_index,members) in enumerate(memberships)
        selected=sort(members;by=index->(!get(estimates[index],"validated",false),
            get(estimates[index],"balance_residual",Inf)+get(estimates[index],"event_condition_residual",Inf),estimates[index]["id"]))
        representative=estimates[first(selected)]
        group=copy(representative);group_id="$(representative["source_job_id"])-repeat-group$group_index"
        group["id"]=group_id;group["representative_estimate_id"]=representative["id"]
        group["raw_estimate_ids"]=[estimates[index]["id"] for index in members]
        group["grouping_scope"]="Provisional complete-linkage numerical repeat group; not proof of event identity or distinctness. Close unresolved multiple events are not ruled out"
        group["grouping_tolerance"]=Dict("affine_t"=>time_tolerance,"state_infinity_norm"=>state_tolerance)
        for index in members
            mapping[estimates[index]["id"]]=group_id
        end
        push!(groups,group)
    end
    groups,mapping
end

function execute_targeted_transition(job,config)
    source=targeted_source_id(job)
    haskey(TARGETED_TRANSITION_BRACKETS,source) || throw(ArgumentError("undeclared transition path"))
    original=job["source_recovery"];payload=job["source_payload"]
    first=payload["base"]["parameters"];last=payload["target"]["parameters"]
    frozen_samples=original["samples"]
    length(frozen_samples)==17 || throw(ArgumentError("source path must have seventeen frozen samples"))
    for (index,sample) in enumerate(frozen_samples)
        reconstructed=targeted_affine_parameters(first,last,(index-1)/16)
        maximum(abs(reconstructed[key]-sample["parameters"][key]) for key in ATLAS_COORDINATES)<=1e-12 ||
            throw(ArgumentError("source path samples disagree with frozen endpoints"))
    end
    cache=Dict{Float64,Dict{Int,Dict{String,Any}}}()
    points=Dict{String,Any}[];intervals=Dict{String,Any}[];terminal=Dict{String,Any}[]
    inventory_limit=256
    function inventory(t,grid)
        local position=Float64(t)
        if !haskey(cache,position)
            length(cache)<inventory_limit || return nothing
            cache[position]=Dict{Int,Dict{String,Any}}()
        end
        haskey(cache[position],grid) && return cache[position][grid]
        local parameters=targeted_affine_parameters(first,last,position)
        local nearest=argmin(abs(position-(i-1)/16) for i in 1:17)
        local extra=[Float64.(root["state"]) for root in frozen_samples[nearest]["roots"]]
        local settings=merge(config,Dict("seed_grid"=>grid,"balance_atol"=>1e-11,"event_atol"=>1e-9))
        local point=atlas_point(parameters,settings;id="$(get(job,"id",source))-t$(string(position))-g$grid",extra_seeds=extra)
        point["affine_t"]=position;point["inventory_grid"]=grid;point["source_job_id"]=source
        cache[position][grid]=point;push!(points,point)
        point
    end
    for (bracket_index,(left_index,right_index)) in enumerate(TARGETED_TRANSITION_BRACKETS[source])
        left=(left_index-1)/16;right=(right_index-1)/16
        initial_left=inventory(left,9);initial_right=inventory(right,9)
        (initial_left===nothing || initial_right===nothing) && error("initial inventory budget exhausted")
        pending=[(left,right)];split_count=0
        while !isempty(pending)
            sort!(pending;by=pair->(-(pair[2]-pair[1]),pair[1]))
            a,b=popfirst!(pending);left_point=inventory(a,9);right_point=inventory(b,9)
            (left_point===nothing || right_point===nothing) && error("cached inventory missing")
            changed=left_point["signature"]!=right_point["signature"]
            row=Dict{String,Any}("id"=>targeted_interval_id(job,a,b),"original_bracket"=>bracket_index,
                "source_sample_indices"=>[left_index,right_index],"t_bounds"=>[a,b],"width"=>b-a,
                "from"=>left_point["id"],"to"=>right_point["id"],"sample_ids"=>[left_point["id"],right_point["id"]],
                "kind"=>"inventory_sample_segment","continuous_coverage_certified"=>false,"validated"=>false,
                "first_signature"=>left_point["signature"],"second_signature"=>right_point["signature"],
                "linked_event_ids"=>String[])
            if !changed
                row["status"]="finite_same_signature_samples"
                row["scope"]="Equal inventories at these sampled endpoints do not exclude intervening changes"
                push!(intervals,row);continue
            end
            reason=b-a<=1e-4 ? "width_target" : split_count>=12 ? "split_cap" : length(cache)>=inventory_limit ? "inventory_cap" : ""
            if isempty(reason)
                midpoint=(a+b)/2;midpoint_inventory=inventory(midpoint,9)
                if midpoint_inventory!==nothing
                    split_count+=1
                    push!(pending,(a,midpoint),(midpoint,b))
                    continue
                end
                reason="inventory_cap"
            end
            left_dense=inventory(a,13);right_dense=inventory(b,13)
            comparisons=left_dense===nothing || right_dense===nothing ? Any[] :
                Any[pilot_search_comparison(left_point,left_dense,"grid9","grid13"),
                    pilot_search_comparison(right_point,right_dense,"grid9","grid13")]
            agreement=length(comparisons)==2 && all(comparison->comparison["status"]=="agreement_observed",comparisons)
            row["status"]=agreement ? "signature_change_observed" : "crosscheck_disagreement"
            row["localization_stop_reason"]=reason;row["grid13_comparisons"]=comparisons
            row["crosscheck_agreement"]=agreement;row["split_count"]=split_count
            row["localization_width_target_met"]=b-a<=1e-4
            push!(intervals,row);push!(terminal,row)
        end
    end
    events=Dict{String,Any}[];correctors=0
    # Fair first-seed passes across all retained changes precede additional
    # seeds. Every coarse and refined Newton solve consumes the common cap.
    for seed_index in 1:4,interval in terminal,kind in ("fold","trace_zero")
        correctors>=64 && break
        a,b=interval["t_bounds"];left=inventory(a,9);right=inventory(b,9)
        seeds=targeted_event_seeds(left,right,kind)
        length(seeds)>=seed_index || continue
        correctors+=1
        coarse=targeted_corrector(first,last,seeds[seed_index],kind,(a,b))
        attempt=Dict{String,Any}("kind"=>kind,"seed"=>seeds[seed_index],"coarse"=>coarse)
        push!(get!(interval,"corrector_attempts",Any[]),attempt)
        coarse["converged"] || continue
        refined=if correctors<64
            correctors+=1
            targeted_corrector(first,last,coarse["coordinates"],kind,(a,b);tight=true)
        else
            Dict{String,Any}("converged"=>false,"reason"=>"corrector_cap")
        end
        attempt["refined"]=refined
        refined["converged"] || continue
        z=refined["coordinates"];parameters=targeted_affine_parameters(first,last,z[3]);state=copy(z[1:2])
        saved_residual=targeted_event_residual(first,last,vcat(state,z[3]),kind)
        refinement_difference=norm(z-coarse["coordinates"],Inf)
        valid_residual=maximum(abs,saved_residual[1:2])<=1e-11 && abs(saved_residual[3])<=1e-9
        classification=kind=="fold" ? "unclassified_fold_candidate" : "trace_zero_candidate"
        validated=false;coefficients=Dict{String,Any}()
        if kind=="fold"
            coefficients=targeted_fold_validation(first,last,z,coarse["coordinates"],interval["crosscheck_agreement"])
            validated=valid_residual&&coefficients["validated"]
            validated && (classification="numerically_validated_fold")
        end
        K=balance_jacobian(parameters,state);J=Diagonal([1.0,1/parameters["r"]])*K
        event_id="$(get(job,"id",source))-event$(length(events)+1)"
        event=Dict{String,Any}("id"=>event_id,"source_interval_id"=>interval["id"],"source_job_id"=>source,
            "kind"=>kind,"classification"=>classification,"status"=>validated ? "completed" : "candidate",
            "validated"=>validated,"parameters"=>parameters,"state"=>state,"affine_t"=>z[3],
            "balance_residual"=>maximum(abs,saved_residual[1:2]),"event_condition_residual"=>abs(saved_residual[3]),
            "trace"=>tr(J),"determinant"=>det(J),"refinement_difference"=>refinement_difference,
            "coefficients"=>coefficients,"scope"=>kind=="fold" ? "Numerical fold checks on the frozen affine path; no neighborhood certification" :
                "Localized trace-zero condition only; no Hopf bifurcation classification")
        push!(events,event);push!(interval["linked_event_ids"],event_id)
    end
    initial_brackets=Dict{String,Any}[]
    for pair in TARGETED_TRANSITION_BRACKETS[source]
        a,b=(pair[1]-1)/16,(pair[2]-1)/16
        fresh_left=inventory(a,9);fresh_right=inventory(b,9)
        source_comparisons=[pilot_search_comparison(frozen_samples[pair[1]],fresh_left,"frozen_source","fresh_grid9"),
            pilot_search_comparison(frozen_samples[pair[2]],fresh_right,"frozen_source","fresh_grid9")]
        push!(initial_brackets,Dict("sample_indices"=>collect(pair),"t_bounds"=>[a,b],
            "source_endpoint_comparisons"=>source_comparisons,
            "source_signatures"=>[frozen_samples[pair[1]]["signature"],frozen_samples[pair[2]]["signature"]],
            "grid9_signatures"=>[fresh_left["signature"],fresh_right["signature"]],
            "source_root_counts"=>[length(frozen_samples[pair[1]]["roots"]),length(frozen_samples[pair[2]]["roots"])],
            "grid9_root_counts"=>[length(fresh_left["roots"]),length(fresh_right["roots"])],
            "grid9_signature_change_observed"=>fresh_left["signature"]!=fresh_right["signature"],
            "source_endpoint_inventories_reproduced"=>all(comparison->comparison["status"]=="agreement_observed",source_comparisons)))
    end
    for interval in terminal
        required=sum(length(targeted_event_seeds(inventory(interval["t_bounds"][1],9),
            inventory(interval["t_bounds"][2],9),kind)) for kind in ("fold","trace_zero"))
        attempts=get(interval,"corrector_attempts",Any[])
        incomplete=length(attempts)<required||any(attempt->get(get(attempt,"refined",Dict()),"reason","")=="corrector_cap",attempts)
        interval["corrector_status"]=incomplete ? "unresolved_corrector_cap" : "declared_seed_attempts_finished"
    end
    groups,group_mapping=targeted_repeat_groups(events)
    for interval in intervals
        interval["linked_estimate_ids"]=copy(interval["linked_event_ids"])
        interval["linked_event_ids"]=unique([group_mapping[id] for id in interval["linked_estimate_ids"]])
    end
    record=Dict{String,Any}("source_job_id"=>source,"source_sample_ids"=>getindex.(frozen_samples,"id"),
        "path_start"=>first,"path_end"=>last,"initial_brackets"=>initial_brackets,
        "intervals"=>intervals,"events"=>groups,"event_estimates"=>events,
        "event_estimate_count"=>length(events),"repeat_group_count"=>length(groups),
        "repeat_group_policy"=>Dict("algorithm"=>"complete_linkage","affine_t_tolerance"=>1e-6,"state_infinity_tolerance"=>1e-6,
            "representative"=>"actual best validated estimate, residual-sum then identifier tie-break; never averaged",
            "scope"=>"Numerical repeat grouping is not proof of event identity or counts of distinct events"),
        "inventory_location_count"=>length(cache),"inventory_location_cap"=>256,
        "inventory_solve_count"=>length(points),"corrector_count"=>correctors,"corrector_cap"=>64,
        "status"=>"bounded_protocol_executed","continuous_coverage_certified"=>false,
        "scope"=>"Only declared original brackets were refined; unchanged sampled intervals do not prove absence. Event nodes provisionally group numerical repeats; raw estimates and all attempts remain saved",
        "completeness"=>"CompletenessNotCertified")
    Dict{String,Any}("status"=>"completed","evidence"=>Dict("targeted_transitions"=>[record],"points"=>points,"paths"=>intervals,"events"=>groups),"followups"=>Any[])
end

function targeted_prefix_comparison(saved,fresh;tolerance=1e-10)
    oldpoints=saved["points"];newpoints=fresh["points"]
    count=min(length(oldpoints),length(newpoints));parameter_difference=0.0;state_difference=0.0
    diagnostics=Dict{String,Any}[]
    for index in 1:count
        before=oldpoints[index];after=newpoints[index]
        parameter_difference=max(parameter_difference,abs(before["parameter"]-after["parameter"]))
        state_difference=max(state_difference,norm(before["state"]-after["state"],Inf))
        push!(diagnostics,Dict("index"=>index,"saved_balance_residual"=>before["balance_residual"],
            "new_balance_residual"=>after["balance_residual"],"saved_stability"=>before["stability"],"new_stability"=>after["stability"]))
    end
    agreement=count==length(oldpoints)&&count>0&&parameter_difference<=tolerance&&state_difference<=tolerance
    Dict{String,Any}("status"=>agreement ? "shared_prefix_consistent" : "shared_prefix_disagreement_or_unresolved",
        "compared_point_count"=>count,"saved_point_count"=>length(oldpoints),
        "maximum_parameter_difference"=>parameter_difference,"maximum_state_difference"=>state_difference,
        "tolerance"=>tolerance,"diagnostics"=>diagnostics,"saved_attempts"=>saved["attempts"],
        "saved_candidates"=>saved["candidates"],"tangent_comparison"=>"unavailable_not_persisted_in_source",
        "scope"=>"Only persisted parameter/state prefixes are compared; matching deterministic prefixes do not establish depth sufficiency")
end

function execute_targeted_continuation(job,config)
    source=targeted_source_id(job)
    source in TARGETED_CONTINUATION_SOURCES || throw(ArgumentError("undeclared continuation source"))
    original=job["source_audit"];payload=job["source_payload"]
    original["deep_steps"]==24 || throw(ArgumentError("expected frozen 24-step source"))
    original["coordinate"]=="A" || throw(ArgumentError("only frozen A continuations are authorized"))
    parameters=payload["training_point"]["parameters"]
    initial=Float64.(original["initial_state"]);bounds=Tuple(Float64.(original["bounds"]))
    tight=merge(config,Dict("balance_atol"=>1e-11,"event_atol"=>1e-9,"max_corrector_iters"=>100,"max_evaluations"=>10000))
    options=ContinuationOptions(max_steps=96,corrector_atol=1e-11)
    curve=continue_equilibria(value->reduced_model(atlas_replace(parameters,"A",value)),initial,parameters["A"];
        parameter_bounds=bounds,options=options,equilibrium_options=atlas_equilibrium_options(tight))
    branches=pilot_branch_summary(curve)
    for branch in branches
        saved=only(filter(row->row["direction"]==branch["direction"],original["deep"]))
        branch["prefix_comparison"]=targeted_prefix_comparison(saved,branch)
        branch["extension_point_count"]=max(0,length(branch["points"])-length(saved["points"]))
        branch["source_point_count"]=length(saved["points"])
        branch["source_termination"]=saved["termination"]
        branch["new_candidate_count"]=count(candidate->candidate["second_point"]>length(saved["points"]),branch["candidates"])
        branch["point_scope"]="Single continued equilibrium branch; these points are not full root inventories"
        branch["finite_outcome"]=branch["termination"]=="step_limit" ? "unresolved_96_step_limit" : "bounded_termination_observed"
    end
    record=Dict{String,Any}("source_job_id"=>source,"parameters"=>parameters,"initial_state"=>initial,
        "coordinate"=>"A","bounds"=>collect(bounds),"source_steps"=>24,"extended_steps"=>96,
        "continuation_options"=>Dict(string(key)=>getfield(options,key) isa Tuple ? collect(getfield(options,key)) : getfield(options,key) for key in fieldnames(typeof(options))),
        "branches"=>branches,"status"=>"bounded_protocol_executed","continuous_coverage_certified"=>false,
        "scope"=>"Fourfold accepted-step cap extension from the original seed and bounds; branch completeness remains unresolved",
        "completeness"=>"CompletenessNotCertified")
    Dict{String,Any}("status"=>"completed","evidence"=>Dict("targeted_continuations"=>[record]),"followups"=>Any[])
end
