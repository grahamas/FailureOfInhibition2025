# Bounded campaign reader: separate from historical pilot and targeted readers.
using TOML, SHA

_campaign_payload(job) = get(job, "payload", job)
_campaign_role(job) = string(get(_campaign_payload(job), "campaign_role", "unknown"))
_campaign_round(job) = get(_campaign_payload(job), "campaign_round", get(_campaign_payload(job), "round", 0))
_campaign_digest(value) = bytes2hex(sha256(codeunits(sprint(io -> TOML.print(io, value; sorted=true)))))
_campaign_anchor(id) = "evidence-" * bytes2hex(sha256(string(id)))[1:16]

"""Read hash-bound results only. Execution completion and resolved science are separate."""
function campaign_report_data(output)
    output = realpath(output)
    for name in ("state.toml", "config.toml", "campaign_manifest.toml")
        islink(joinpath(output,name)) && throw(ArgumentError("campaign input must not be a symlink"))
    end
    state = TOML.parsefile(joinpath(output,"state.toml"))
    config = TOML.parsefile(joinpath(output,"config.toml"))
    manifest = TOML.parsefile(joinpath(output,"campaign_manifest.toml"))
    isdefined(@__MODULE__, :verify_campaign_inputs) || throw(ArgumentError("campaign reports require frozen-input verification"))
    verify_campaign_inputs(output,state,config)
    jobs = _report_records(get(state,"jobs",[]))
    results = Dict{String,Any}(); issues = Dict{String,Any}[]
    for job in jobs
        id = string(get(job,"id","")); status = string(get(job,"status","pending"))
        status in ("pending","running") && continue
        relative = get(job,"result_file","")
        path = _report_safe_file(output,relative)
        if path === nothing || !haskey(job,"result_hash")
            push!(issues,Dict("job"=>id,"status"=>"unverified_result","detail"=>"Missing hash-bound saved result")); continue
        end
        bytes = read(path)
        if bytes2hex(sha256(bytes)) != job["result_hash"]
            push!(issues,Dict("job"=>id,"status"=>"unverified_result","detail"=>"SHA-256 mismatch")); continue
        end
        result = try TOML.parse(String(bytes)) catch
            push!(issues,Dict("job"=>id,"status"=>"unreadable_result","detail"=>"Malformed TOML")); continue
        end
        results[id] = Dict("status"=>get(result,"status","unknown"),"evidence"=>get(result,"evidence",Dict()),"source_result"=>relative)
    end
    (;state,config,manifest,jobs,results,issues)
end

function _campaign_success(job,result)
    get(job,"status","")=="completed" && result !== nothing && get(result,"status","")=="completed" || return false
    if _campaign_role(job) in ("inventory","validation_inventory")
        points=_report_records(get(result["evidence"],"points",[]))
        return !isempty(points) && all(p->!isempty(get(p,"roots",[])) && !occursin("unresolved",lowercase(string(get(p,"signature","unresolved")))),points)
    end
    true
end
"""Expose declared unresolved diagnostics without changing graph eligibility."""
function _campaign_unresolved_diagnostics(job,result)
    result===nothing && return Dict{String,Any}[]
    rows=Dict{String,Any}[];seen=Set{Tuple{String,String}}()
    evidence=get(result,"evidence",Dict())
    observed_transition=any(row->get(row,"status","")=="signature_change_observed" && get(row,"crosscheck_agreement",false)===true,_report_records(get(evidence,"paths",[])))
    function record!(id,reason)
        key=(id,reason);key in seen && return
        push!(seen,key)
        push!(rows,Dict("source_job"=>string(job["id"]),"record_id"=>id,"reason"=>reason,"source_result"=>get(result,"source_result","")))
    end
    function visit(record,fallback)
        record isa AbstractDict || return
        id=string(get(record,"id",fallback))
        failures=get(record,"tracking_guard_failures",Any[])
        failures isa AbstractVector && !isempty(failures) && record!(id,"tracking_guard_failures")
        get(record,"audit_agreement",nothing)===false && record!(id,"audit_disagreement_or_unresolved")
        get(record,"forward_tracking_unambiguous",nothing)===false && record!(id,"forward_tracking_unresolved")
        stop=string(get(record,"localization_stop_reason",""))
        stop in ("split_cap","inventory_cap","corrector_cap") && record!(id,"localization_"*stop)
        corrector=string(get(record,"corrector_status",""))
        startswith(corrector,"unresolved") && record!(id,corrector)
        status=string(get(record,"status",""))
        # A different resolved signature makes the aggregate preservation edge
        # unresolved by design. Its audited transition is evidence, not failure.
        transition_only=status=="unresolved" && get(record,"kind","")=="unknown_connection" && observed_transition && get(record,"audit_agreement",false)===true && get(record,"forward_tracking_unambiguous",false)===true && failures isa AbstractVector && isempty(failures)
        !transition_only && (startswith(status,"unresolved") || endswith(status,"_unresolved") || status in ("crosscheck_disagreement","disagreement_observed","disagreement_or_unresolved","outside_admissible_domain","unavailable_target_or_base")) && record!(id,status)
        if haskey(record,"signature") && (occursin("unresolved",lowercase(string(record["signature"]))) || isempty(get(record,"roots",Any[])))
            record!(id,"unresolved_inventory")
        end
        # Follow only declared scientific summaries, not solver attempt logs:
        # an unsuccessful seed does not invalidate an otherwise resolved search.
        for key in ("intervals","comparisons","endpoint_comparisons","transverse_probes","unresolved_sides","levels")
            for (index,child) in enumerate(_report_records(get(record,key,[])))
                visit(child,"$id/$key/$index")
            end
        end
    end
    evidence=get(result,"evidence",Dict())
    for key in ("points","paths","campaign_paths","audits","validations","recoveries","geometry_probes","events")
        for (index,record) in enumerate(_report_records(get(evidence,key,[])))
            visit(record,"$key/$index")
        end
    end
    rows
end
_campaign_fully_resolved(job,result) = _campaign_success(job,result) && isempty(_campaign_unresolved_diagnostics(job,result))

function _campaign_rows(data,key; successful_only=false)
    rows=Dict{String,Any}[]
    for job in data.jobs
        result=get(data.results,string(job["id"]),nothing)
        result===nothing && continue
        successful_only && !_campaign_success(job,result) && continue
        for raw in _report_records(get(result["evidence"],key,[]))
            push!(rows,merge(Dict{String,Any}(string(k)=>v for (k,v) in raw),Dict("source_job"=>job["id"],"source_result"=>result["source_result"],"campaign_role"=>_campaign_role(job),"round"=>_campaign_round(job),"eligible_for_regions"=>_campaign_success(job,result))))
        end
    end
    rows
end

"""Compact typed graph references raw tuples instead of duplicating numerical arrays."""
function campaign_evidence_graph(data)
    points=_campaign_rows(data,"points")
    excluded=Set{String}()
    for job in data.jobs
        _campaign_role(job)=="audit" || continue
        result=get(data.results,string(job["id"]),nothing)
        audits=result===nothing ? [] : _report_records(get(result["evidence"],"audits",[]))
        passed=_campaign_success(job,result) && length(audits)==1 && get(only(audits),"status","")=="agreement_observed"
        passed || push!(excluded,string(get(_campaign_payload(job),"training_source_job","")))
    end
    for point in points
        point["source_job"] in excluded && (point["eligible_for_regions"]=false)
    end
    paths=_campaign_rows(data,"paths")
    lookup=Dict{String,Any}();nodes=Dict{String,Any}[];edges=Dict{String,Any}[]
    for point in points
        id=string(get(point,"id",""));isempty(id) && continue
        haskey(lookup,id) && throw(ArgumentError("duplicate campaign inventory ID: $id"))
        lookup[id]=point
        push!(nodes,Dict("id"=>id,"node_type"=>"inventory","signature"=>get(point,"signature","unresolved"),"root_count"=>length(get(point,"roots",[])),"audit_only"=>get(point,"audit_only",false),"eligible"=>point["eligible_for_regions"],"source_job"=>point["source_job"],"source_result"=>point["source_result"],"round"=>point["round"]))
    end
    valid_paths=Dict{String,Any}[]
    root_seen=Set{String}()
    for (index,path) in enumerate(paths)
        from=string(get(path,"from",""));to=string(get(path,"to",""))
        # Use the common component validator, which checks every intermediate
        # inventory and never infers connectivity from equal signatures.
        structural=get(path,"forward_tracking_unambiguous",false)===true && get(path,"audit_agreement",false)===true && isempty(get(path,"tracking_guard_failures",["unreported"])) && get(path,"eligible_for_regions",false) && get(path,"validated",false)===true && get(path,"signature_preserved",get(path,"sampled_inventory_preserved",false))===true && get(path,"status","")=="completed"
        ids=string.(get(path,"sample_ids",String[]))
        structural &= length(ids)>=129 && length(unique(ids))==length(ids) && !isempty(ids) && first(ids)==from && last(ids)==to && all(id->haskey(lookup,id),ids)
        structural && (structural=all(id->get(lookup[id],"eligible_for_regions",false) && !isempty(get(lookup[id],"roots",[])) && !occursin("unresolved",lowercase(string(get(lookup[id],"signature","unresolved")))) && lookup[id]["signature"]==lookup[from]["signature"],ids))
        clean_track=get(path,"kind","")=="tracked_root_link" && get(path,"validated",false)===true && get(path,"track_clean",false)===true && get(path,"status","")=="clean_sampled_corrector_link"
        if clean_track
            witnesses=String[]
            for (context,key) in ((from,"from_state"),(to,"to_state"))
                state=get(path,key,Any[])
                length(state)==2 && all(x->x isa Real && isfinite(x),state) || continue
                witness="root:"*path["source_job"]*":"*string(get(path,"origin","unspecified"))*":"*string(get(path,"track",0))*":"*context
                if !(witness in root_seen)
                    push!(root_seen,witness)
                    push!(nodes,Dict("id"=>witness,"node_type"=>"root_witness","inventory_context"=>context,"inventory_membership_asserted"=>false,"origin"=>get(path,"origin","unspecified"),"source_job"=>path["source_job"],"source_result"=>path["source_result"]))
                end
                push!(witnesses,witness)
            end
            if length(witnesses)==2
                from,to=witnesses
            else
                clean_track=false
            end
        end
        kind=clean_track ? (get(path,"origin","")=="base_origin_forward" ? "forward_root_track" : "reverse_root_track") : structural ? "sampled_structural_preservation" : get(path,"status","")=="signature_change_observed" && get(path,"crosscheck_agreement",false)===true && get(path,"eligible_for_regions",false) ? "sampled_transition" : "uncertainty"
        push!(edges,Dict("id"=>string(get(path,"id","path-$index")),"from"=>from,"to"=>to,"kind"=>kind,"sample_count"=>length(ids),"source_job"=>path["source_job"],"source_result"=>path["source_result"]))
        structural && push!(valid_paths,merge(path,Dict("signature_preserved"=>true)))
    end
    for record in _campaign_rows(data,"campaign_paths")
        comparisons=_report_records(get(record,"endpoint_comparisons",[]))
        !isempty(comparisons) && all(row->get(row,"status","")=="agreement_observed",comparisons) || continue
        for repeated in _report_records(get(record,"endpoint_repeats",[]))
            original=string(get(repeated,"original_point_id",""))
            haskey(lookup,original) || continue
            for fresh in string.(get(repeated,"fresh_sample_ids",String[]))
                haskey(lookup,fresh) || continue
                push!(edges,Dict("id"=>"audit:"*record["source_job"]*":"*fresh,"from"=>original,"to"=>fresh,"kind"=>"fixed_parameter_inventory_audit","inventory_equivalence_certified"=>false,"source_job"=>record["source_job"],"source_result"=>record["source_result"]))
            end
        end
    end
    for event in _campaign_rows(data,"events")
        id=string(get(event,"id",""));isempty(id) && continue
        event_node="event:"*id;kind=string(get(event,"kind",get(event,"event_type","unknown")))
        validated=get(event,"validated",false)===true
        label=kind=="fold" && validated ? "numerically_validated_fold_witness" : kind=="trace_zero" ? "trace_zero_candidate" : string(get(event,"classification",kind*"_candidate"))
        push!(nodes,Dict("id"=>event_node,"node_type"=>"event_witness","kind"=>kind,"validated"=>validated,"status"=>get(event,"status","candidate"),"classification"=>label,"source_job"=>event["source_job"],"source_result"=>event["source_result"]))
        for path in paths
            path["source_job"]==event["source_job"] || continue
            (id in get(path,"linked_event_ids",String[]) || get(path,"id",nothing)==get(event,"source_interval_id",nothing) && haskey(event,"source_interval_id")) || continue
            side_status=get(path,"status","")=="signature_change_observed" && get(path,"crosscheck_agreement",false)===true ? "sampled_side_context" : "unresolved_incident_interval"
            for endpoint in (string(get(path,"from","")),string(get(path,"to","")))
                haskey(lookup,endpoint) || continue
                push!(edges,Dict("id"=>"event-side:"*id*":"*endpoint,"from"=>endpoint,"to"=>event_node,"kind"=>"event_side_context","status"=>side_status,"interval_validation_inferred_from_event"=>false,"source_job"=>event["source_job"],"source_result"=>event["source_result"]))
            end
        end
    end
    # Held-out observations never enter the training components or base selection.
    training=[p for p in points if p["campaign_role"] in ("inventory","component_path") && get(p,"audit_only",false)!==true]
    components=atlas_region_components(training,valid_paths)
    Dict("nodes"=>nodes,"edges"=>edges,"sampled_components"=>components,"scope"=>"Explicit sampled evidence only; equal signatures add no edge; missing links do not prove disconnection; continuous coverage not certified")
end

function _campaign_progress(data,jobs)
    verified=[job for job in jobs if haskey(data.results,string(job["id"]))]
    Dict("planned"=>length(jobs),"attempted"=>count(j->get(j,"status","")!="pending",jobs),"execution_completed"=>count(j->get(j,"status","")=="completed",jobs),"resolved"=>count(j->_campaign_fully_resolved(j,get(data.results,string(j["id"]),nothing)),jobs),"fully_resolved"=>count(j->_campaign_fully_resolved(j,get(data.results,string(j["id"]),nothing)),jobs),"completed_with_unresolved"=>count(j->get(j,"status","")=="completed" && !_campaign_fully_resolved(j,get(data.results,string(j["id"]),nothing)),jobs),"verified_results"=>length(verified),"pending"=>count(j->get(j,"status","")=="pending",jobs),"failed_or_unresolved"=>count(j->get(j,"status","") in ("failed","unresolved","timed_out","cancelled"),jobs))
end

function campaign_report_summary(data)
    graph=campaign_evidence_graph(data)
    progress=Dict{String,Any}[];ledger=Dict{String,Any}[];rounds=Dict{String,Any}[]
    for role in sort(unique(_campaign_role.(data.jobs)))
        jobs=filter(j->_campaign_role(j)==role,data.jobs)
        push!(progress,merge(Dict("role"=>role),_campaign_progress(data,jobs)))
    end
    cohorts=Dict{Tuple{String,String,String,String},Vector{Any}}()
    for job in data.jobs
        p=_campaign_payload(job)
        key=(string(_campaign_round(job)),string(get(p,"campaign_shell",get(p,"shell","unreported"))),string(get(p,"campaign_chart",get(p,"chart","unreported"))),string(get(p,"campaign_mask",get(p,"campaign_boundary_mask",get(p,"boundary_mask","unreported")))))
        push!(get!(cohorts,key,Any[]),job)
    end
    for (key,jobs) in sort(collect(cohorts);by=first)
        push!(ledger,merge(Dict("round"=>key[1],"shell"=>key[2],"chart"=>key[3],"boundary_mask"=>key[4]),_campaign_progress(data,jobs)))
    end
    seen=Set{String}()
    points=_campaign_rows(data,"points";successful_only=true)
    for round in 1:3
        selected=[p for p in points if p["round"]==round && p["campaign_role"] in ("inventory","component_path") && get(p,"audit_only",false)!==true && !occursin("unresolved",lowercase(string(get(p,"signature","unresolved")))) && !isempty(get(p,"roots",[]))]
        signatures=Set(string(p["signature"]) for p in selected)
        novel=setdiff(signatures,seen);union!(seen,signatures)
        push!(rounds,Dict("round"=>round,"planned_discovery_points"=>512,"planned_audits"=>128,"saved_resolved_inventory_records"=>length(selected),"observed_signatures"=>length(signatures),"first_observed_signatures"=>length(novel),"interpretation"=>"Descriptive sampled novelty; not a missing-region probability or stopping guarantee"))
    end
    recoveries=Dict{String,Any}[]
    for r in _campaign_rows(data,"recoveries")
        push!(recoveries,Dict("source_job"=>r["source_job"],"source_result"=>r["source_result"],"target_job_id"=>get(_campaign_payload(only(filter(j->j["id"]==r["source_job"],data.jobs))),"campaign_target_job",get(r,"target_job_id","unreported")),"target_point_id"=>get(r,"target_point_id","unreported"),"base_point_id"=>get(r,"base_point_id","unreported"),"job_resolved"=>r["eligible_for_regions"],"forward_target_coverage"=>r["eligible_for_regions"] && get(r,"base_origin_target_coverage",false)===true,"reverse_coverage"=>get(r,"reverse_target_origin_base_coverage","unreported"),"observed_target_root_count"=>get(r,"observed_target_root_count",0),"sampled_structural_preservation"=>get(r,"sampled_inventory_preserved","not_assessed"),"terminal_inventory_agreement"=>get(get(r,"terminal_grid_comparison",Dict()),"status","")=="agreement_observed", "matched_target_roots"=>get(r,"matched_target_roots",0), "initial_track_count"=>get(r,"initial_track_count",0)))
    end
    raw_bases=get(get(data.state,"campaign",Dict()),"base_selection",Dict())
    bases=Dict{String,Any}("frozen_before_validation"=>get(raw_bases,"frozen_before_validation",false),"component_rule"=>get(raw_bases,"component_rule","unreported"),"entries"=>[Dict("point_id"=>get(b,"point_id","unreported"),"members"=>get(b,"members",String[]),"signature"=>get(b,"signature","unreported"),"source_job"=>get(b,"source_job","unreported")) for b in _report_records(get(raw_bases,"entries",[]))],"route_count"=>length(get(raw_bases,"routes",[])),"event_witness_count"=>length(get(raw_bases,"event_witnesses",[])))
    frontier=[Dict("source_job"=>j["id"],"role"=>_campaign_role(j),"status"=>get(j,"status","pending"),"detail"=>get(j,"error","No resolved, hash-verified result")) for j in data.jobs if !_campaign_success(j,get(data.results,string(j["id"]),nothing))]
    for job in data.jobs
        append!(frontier,_campaign_unresolved_diagnostics(job,get(data.results,string(job["id"]),nothing)))
    end
    Dict{String,Any}("schema_version"=>1,"completeness"=>"CompletenessNotCertified","status"=>_campaign_progress(data,data.jobs),"progress"=>progress,"coverage_ledger"=>ledger,"round_novelty"=>rounds,"bases"=>bases,"graph"=>graph,"recoveries"=>recoveries,"unresolved_frontier"=>frontier,"issues"=>data.issues,"heldout_planned"=>256,"scope"=>"Bounded three-round campaign; finite sampled inventories and sampled paths do not certify complete parameter-space coverage")
end

function _campaign_render_html(io,summary,data)
    status=summary["status"];metadata=get(data.state,"campaign",Dict())
    print(io,"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Bounded atlas evidence</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#18202a}nav{display:flex;gap:1rem;flex-wrap:wrap}.warning{border:3px solid #985400;padding:1rem;background:#fff7df}.scroll{overflow:auto;max-height:36rem}table{border-collapse:collapse;font-size:14px;width:100%}th,td{border:1px solid #bbb;padding:.4rem;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#eee}pre{white-space:pre-wrap;margin:0}section{margin:2rem 0}a{color:#075b9b}details{margin:.5rem 0}</style><h1>Bounded atlas evidence</h1><div class=warning><strong>CompletenessNotCertified</strong><p>Planned work, attempted work, execution completion, and resolved scientific evidence are separate. Fully resolved means no explicit unresolved diagnostics for the finite job; candidate classifications remain candidates. Finite samples do not establish complete parameter-space coverage.</p>""")
    print(io,"<p><strong>",status["planned"]," assigned jobs · ",status["attempted"]," attempted · ",status["resolved"]," fully resolved · ",status["completed_with_unresolved"]," completed with unresolved evidence · ",status["pending"]," pending</strong></p><p>Scheduler stage: ",_report_escape(get(metadata,"stage","unreported")),". Protocol finished: ",_report_escape(get(metadata,"protocol_complete",false)),". Stop reason: ",_report_escape(get(metadata,"stop_reason","unreported")),".</p></div>")
    print(io,"<nav><a href=#progress>Progress</a><a href=#graph>Evidence graph</a><a href=#bases>Frozen bases</a><a href=#coverage>Coverage ledger</a><a href=#novelty>Round novelty</a><a href=#recovery>Held-out recovery</a><a href=#frontier>Unresolved frontier</a></nav><p><a href=campaign_summary.toml>Compact decision data</a> · <a href=campaign_manifest.toml>Frozen bounded design</a> · <a href=state.toml>Checkpoint and raw result references</a></p>")
    print(io,"<section id=progress><h2>Planned, attempted, and resolved</h2><p>The design prescribes three rounds of 512 discovery points and 128 audits per round, bounded proposals, geometry probes and component paths, followed by frozen bases and 256 held-out inventories with recovery attempts, including observed multiroot targets. Assigned jobs appear as each stage becomes eligible; the manifest records the complete bounded design. A failed control with pending jobs is unfinished science.</p>")
    _report_table(io,summary["progress"];columns=["role","planned","attempted","execution_completed","fully_resolved","completed_with_unresolved","pending","failed_or_unresolved"],compact=true)
    print(io,"</section><section id=graph><h2>Typed evidence graph</h2><p>This browsable graph is schematic; table order, distances and areas have no parameter-space meaning. Inventory nodes, numerical event witnesses, sampled structural links, transition brackets and uncertainty remain distinct. Equal signatures alone add no edge. Every record links to its single raw source; numerical arrays remain there.</p><p>Filter graph records: <input id=graph-filter aria-label=\"Filter graph records\" type=search></p><div id=graph-records>")
    for node in summary["graph"]["nodes"]
        id=node["id"]
        print(io,"<details class=graph-record id=",_campaign_anchor(id),"><summary>",_report_escape(get(node,"node_type","unknown"))," · ",_report_escape(id)," · ",_report_escape(get(node,"signature",get(node,"classification",""))),"</summary>")
        _report_table(io,[node]);print(io,"</details>")
    end
    node_ids=Set(node["id"] for node in summary["graph"]["nodes"])
    for edge in summary["graph"]["edges"]
        print(io,"<details class=graph-record><summary>",_report_escape(edge["kind"])," · ",_report_escape(edge["from"])," → ",_report_escape(edge["to"]),"</summary><p>")
        for (key,label) in (("from","From evidence"),("to","To evidence"))
            key=="to" && print(io," · ")
            if edge[key] in node_ids
                print(io,"<a href=#",_campaign_anchor(edge[key]),">",label,"</a>")
            else
                print(io,label,": unresolved endpoint (",_report_escape(edge[key]),")")
            end
        end
        print(io,"</p>")
        _report_table(io,[edge]);print(io,"</details>")
    end
    isempty(summary["graph"]["nodes"]) && print(io,"<p>No verified saved inventory nodes or witnesses are available.</p>")
    print(io,"</div></section><section id=bases><h2>Frozen bases and inclusion rationale</h2><p>Bases are representatives of explicitly connected sampled training components. Separate components may share a signature. Held-out inventories and recoveries cannot add training bases or alter their frozen routes. An isolated sampled inventory is a singleton component, not a proven separate region.</p>")
    bases=summary["bases"]
    print(io,"<p>Bases frozen: ",_report_escape(get(metadata,"bases_frozen",false)),". Frozen before validation: ",_report_escape(get(bases,"frozen_before_validation",false)),".</p>")
    base_rows=Dict{String,Any}[]
    for base in _report_records(get(bases,"entries",[]))
        members=get(base,"members",[])
        push!(base_rows,Dict("point_id"=>get(base,"point_id","unreported"),"signature"=>get(base,"signature","unreported"),"sampled_component_members"=>length(members),"source_job"=>get(base,"source_job","unreported"),"rationale"=>length(members)>1 ? "Stable-ID representative of explicit examined sampled-path component" : "Singleton sampled component; no accepted connecting path"))
    end
    _report_table(io,base_rows;compact=true)
    print(io,"</section><section id=coverage><h2>Shell, chart, and boundary-mask coverage ledger</h2><p>The parameter chart bounds projected reduced parameters. The equilibrium chart bounds couplings, u, v and h before projection to thresholds a and b. Their projected domains differ; chart counts must not be pooled into a uniform parameter-space coverage claim. Boundary masks identify sampled zero-coupling strata; unvisited strata remain unresolved.</p>")
    _report_table(io,summary["coverage_ledger"];compact=true)
    print(io,"</section><section id=novelty><h2>Round novelty</h2><p>First observed signatures describe these finite samples. They do not estimate missing-region probability, independent region counts, or a completeness stopping criterion.</p>")
    _report_table(io,summary["round_novelty"];compact=true)
    print(io,"</section><section id=recovery><h2>Frozen-base held-out inventory and recovery</h2><p>The prescribed denominator is 256 held-out target inventories, with separately assigned recovery jobs. Forward coverage asks whether base-origin tracks cover every observed target root. Reverse tracking starts with target information and does not establish forward coverage. Independent inventories, sampled structural preservation, transition evidence, and uncertainty answer separate questions. Multiroot targets retain their full observed-root denominator. No continuous-path or root-completeness certification follows.</p>")
    _report_table(io,summary["recoveries"];compact=true)
    print(io,"</section><section id=frontier><h2>Unresolved frontier</h2><p>Pending, failed, timed-out, missing and unresolved results remain visible. Scheduler completion does not resolve these obligations.</p>")
    _report_table(io,summary["unresolved_frontier"];compact=true)
    print(io,"<h3>Evidence integrity issues</h3>")
    if isempty(summary["issues"])
        print(io,"<p>No evidence integrity issues were detected in the saved records.</p>")
    else
        _report_table(io,summary["issues"];compact=true)
    end
    print(io,"""</section><script>
document.getElementById('graph-filter').addEventListener('input',function(){
  const q=this.value.toLowerCase();
  document.querySelectorAll('.graph-record').forEach(el=>el.hidden=!el.textContent.toLowerCase().includes(q));
});
document.getElementById('graph-records').addEventListener('click',function(event){
  const link=event.target.closest('a[href^="#"]');
  if(!link)return;
  const target=document.getElementById(link.getAttribute('href').slice(1));
  if(!target)return;
  target.hidden=false;
  target.open=true;
});
</script></html>""")
end

"""Render compact campaign decisions without running any scientific jobs."""
function write_campaign_report(output)
    output=realpath(output)
    for name in ("index.html","campaign_summary.toml")
        _report_check_target(joinpath(output,name))
    end
    data=campaign_report_data(output);summary=campaign_report_summary(data)
    budget=isdefined(@__MODULE__,:storage_budget) ? storage_budget(output) : nothing
    _report_atomic_write(joinpath(output,"campaign_summary.toml");budget=budget) do io
        TOML.print(io,summary;sorted=true)
    end
    _report_atomic_write(joinpath(output,"index.html");budget=budget) do io
        _campaign_render_html(io,summary,data)
    end
    joinpath(output,"index.html")
end
