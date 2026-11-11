# Outcome-selected diagnostic follow-up. Included after the shared report helpers.
using TOML, SHA

_targeted_role(job) = string(get(_report_pilot_payload(job),"target_role","unknown"))
_targeted_anchor(id) = "targeted-" * bytes2hex(sha256(string(id)))[1:12]
_targeted_digest(value) = bytes2hex(sha256(codeunits(sprint(io->TOML.print(io,value;sorted=true)))))

function targeted_report_data(output)
    output=realpath(output)
    islink(joinpath(output,"state.toml")) && throw(ArgumentError("checkpoint must not be a symlink"))
    state=TOML.parsefile(joinpath(output,"state.toml"))
    get(state,"protocol","")=="targeted_followup_v1" || throw(ArgumentError("not a targeted follow-up packet"))
    manifest_path=_report_safe_file(output,"targeted_manifest.toml")
    manifest_path===nothing && throw(ArgumentError("missing targeted manifest"))
    islink(joinpath(output,"targeted_manifest.toml")) && throw(ArgumentError("targeted manifest must not be a symlink"))
    manifest=TOML.parsefile(manifest_path)
    recorded=get(state,"targeted_manifest_hash",get(get(state,"targeted",Dict()),"manifest_hash",nothing))
    _targeted_digest(manifest)==recorded || throw(ArgumentError("targeted manifest hash mismatch"))
    isdefined(@__MODULE__,:verify_targeted_inputs) || throw(ArgumentError("targeted reports require the frozen-input verifier"))
    begin
        config_path=_report_safe_file(output,"config.toml")
        config_path===nothing && throw(ArgumentError("missing targeted configuration"))
        verify_targeted_inputs(output,state,TOML.parsefile(config_path))
    end
    shared=atlas_report_data(output)
    return (;shared...,manifest)
end

function _targeted_point_nodes(points)
    nodes=Dict{String,Any}[];seen=Set{String}()
    for point in _report_records(points)
        id=string(get(point,"id",""));isempty(id) && continue
        id in seen && throw(ArgumentError("duplicate targeted inventory ID: $id"))
        push!(seen,id)
        roots=get(point,"roots",[])
        push!(nodes,Dict{String,Any}("id"=>id,"point_ids"=>[id],"signature"=>string(get(point,"signature","unresolved")),"root_count"=>roots isa AbstractVector ? length(roots) : 0,"parameters"=>get(point,"parameters",Dict()),"node_type"=>"inventory","inventory_scope"=>"saved finite root search; completeness not certified"))
    end
    nodes
end

function _targeted_edge_kind(edge;path_inventory_preserved=false)
    kind=string(get(edge,"kind","unknown"));origin=string(get(edge,"origin","unspecified"))
    if kind=="tracked_root_link"
        clean=get(edge,"validated",false)===true && get(edge,"track_clean",false)===true && get(edge,"status","")=="clean_sampled_corrector_link" && _targeted_valid_state(get(edge,"from_state",nothing)) && _targeted_valid_state(get(edge,"to_state",nothing))
        return clean && origin in ("base_origin_forward","forward_base_origin") ? "forward_track" : clean && origin in ("target_origin_reverse","reverse_target_origin") ? "reverse_track" : "unresolved"
    elseif kind=="sampled_inventory_preserved"
        return path_inventory_preserved && get(edge,"status","") in ("sampled_inventory_preserved","completed") ? "sampled_inventory_link" : "finite_interval"
    elseif kind in ("inventory_sample_segment","finite_same_signature_samples")
        return get(edge,"status","")=="signature_change_observed" && get(edge,"crosscheck_agreement",false)===true ? "change_bracket" : get(edge,"status","") in ("finite_same_signature_samples","sampled_endpoints_only") ? "finite_interval" : "unresolved"
    elseif kind in ("sampled_inventory_change","signature_change_bracket")
        return get(edge,"status","") in ("signature_change_observed","sampled_inventory_change","sampled_endpoints_only","completed") ? "change_bracket" : "unresolved"
    elseif kind in ("event_candidate","fold_candidate","trace_zero_candidate")
        return "event_candidate"
    end
    "unresolved"
end

"""Normalize explicit links only; matching signatures never manufacture an edge."""
function targeted_graph(record;source_job="",source_result="")
    adjacency=get(record,"adjacency",Dict());adjacency isa AbstractDict || (adjacency=Dict())
    raw_nodes=get(adjacency,"nodes",get(record,"independent_points",get(record,"points",[])))
    nodes=_targeted_point_nodes(raw_nodes)
    lookup=Dict(node["id"]=>node for node in nodes)
    raw_edges=_report_records(get(adjacency,"edges",[]))
    if isempty(raw_edges) && haskey(record,"intervals")
        raw_edges=[Dict{String,Any}(string(k)=>v for (k,v) in interval) for interval in _report_records(record["intervals"])]
    end
    path_preserved=get(record,"sampled_inventory_preserved",false)===true && length(nodes)>=129 && all(node->node["root_count"]>0 && !occursin("unresolved",lowercase(node["signature"])),nodes) && length(unique(node["signature"] for node in nodes))==1
    edges=Dict{String,Any}[]
    for (index,edge) in enumerate(raw_edges)
        from=string(get(edge,"from",""));to=string(get(edge,"to",""))
        kind=_targeted_edge_kind(edge;path_inventory_preserved=path_preserved)
        if kind=="sampled_inventory_link"
            sample_ids=get(edge,"sample_ids",[])
            valid=get(edge,"validated",false)===true && get(edge,"declared_sample_count",0)==129 && sample_ids isa AbstractVector && length(sample_ids)==129 && length(unique(sample_ids))==129 && first(sample_ids)==get(edge,"from",nothing) && last(sample_ids)==get(edge,"to",nothing) && all(id->haskey(lookup,string(id)),sample_ids)
            valid || (kind="unresolved")
        end
        endpoints_known=haskey(lookup,from)&&haskey(lookup,to)
        if kind in ("forward_track","reverse_track") && endpoints_known && !all(id->_targeted_valid_parameters(lookup[id]["parameters"]),(from,to))
            kind="unresolved"
        end
        if !endpoints_known || kind=="sampled_inventory_link" && get(get(lookup,from,Dict()),"signature","unresolved") != get(get(lookup,to,Dict()),"signature","unresolved")
            kind="unresolved"
        end
        push!(edges,Dict{String,Any}("id"=>string(get(edge,"id","$(source_job)-edge$index")),"from"=>from,"to"=>to,"kind"=>kind,"declared_kind"=>get(edge,"kind","unknown"),"status"=>get(edge,"status","unknown"),"origin"=>get(edge,"origin","unspecified"),"endpoint_inventories_saved"=>endpoints_known,"source_job"=>source_job,"source_result"=>source_result,"details"=>edge))
    end
    # Collapse only consecutive, explicitly linked runs with the same resolved
    # signature. Keep every inventory ID and every original edge in the data.
    groups=Dict{String,Any}[];members=Dict{String,String}()
    preserved=Set{Tuple{String,String}}()
    for edge in edges
        edge["kind"] in ("sampled_inventory_link","forward_track","reverse_track") || continue
        push!(preserved,(edge["from"],edge["to"]));push!(preserved,(edge["to"],edge["from"]))
        if edge["kind"]=="sampled_inventory_link"
            ids=string.(edge["details"]["sample_ids"])
            for (a,b) in zip(ids[1:end-1],ids[2:end])
                push!(preserved,(a,b));push!(preserved,(b,a))
            end
        end
    end
    for (node_index,node) in enumerate(nodes)
        can_join=length(groups)>1 && node_index<length(nodes) && (last(groups[end]["point_ids"]),node["id"]) in preserved && groups[end]["signature"]==node["signature"] && !occursin("unresolved",lowercase(node["signature"])) && node["root_count"]>0 && groups[end]["root_count"]==node["root_count"]
        if can_join
            push!(groups[end]["point_ids"],node["id"])
        else
            push!(groups,merge(copy(node),Dict("id"=>"$(source_job)-group$(length(groups)+1)","point_ids"=>copy(node["point_ids"]))))
        end
        members[node["id"]]=groups[end]["id"]
    end
    witnesses,witness_edges=_targeted_witnesses(record,nodes,edges;source_job=source_job,source_result=source_result)
    Dict{String,Any}("source_job"=>source_job,"source_result"=>source_result,"inventory_nodes"=>nodes,"witness_nodes"=>witnesses,"witness_edges"=>witness_edges,"nodes"=>vcat(nodes,witnesses),"display_groups"=>groups,"membership"=>members,"edges"=>edges,"scope"=>"Schematic evidence graph; distances and areas have no parameter-space meaning. A missing or unresolved link does not establish disconnection.")
end

function _targeted_records(data,key)
    rows=Dict{String,Any}[]
    for job in data.jobs
        id=string(get(job,"id",""));saved=get(data.pilot_results,id,nothing)
        saved===nothing && continue
        allowed=get(Dict("targeted_diagnoses"=>("diagnosis",),"targeted_recoveries"=>("recovery_refinement","multiroot"),"targeted_transitions"=>("transition",),"targeted_continuations"=>("audit_extension",),"controls"=>("control",)),key,())
        _targeted_role(job) in allowed || continue
        for record in _report_records(get(saved["evidence"],key,[]))
            push!(rows,merge(Dict{String,Any}(string(k)=>v for (k,v) in record),Dict("source_job"=>id,"source_result"=>saved["source_result"],"target_role"=>_targeted_role(job),"job_status"=>get(job,"status","unknown"),"result_status"=>saved["status"])))
        end
    end
    rows
end

function targeted_report_summary(data)
    diagnoses=_targeted_records(data,"targeted_diagnoses")
    recoveries=_targeted_records(data,"targeted_recoveries")
    transitions=_targeted_records(data,"targeted_transitions")
    continuations=_targeted_records(data,"targeted_continuations")
    graphs=Dict{String,Any}[]
    for record in vcat(recoveries,transitions)
        job=record["source_job"]
        point_rows=[point for point in data.records["points"] if string(get(point,"source_job",""))==job]
        prepared=merge(Dict("points"=>point_rows),record)
        push!(graphs,targeted_graph(prepared;source_job=job,source_result=record["source_result"]))
    end
    recovery_rows=Dict{String,Any}[]
    for record in recoveries
        levels=_report_records(get(record,"levels",[]))
        grids=[get(level,"grid_size",0) for level in levels]
        level_complete=length(grids)==3 && all(grid->grid isa Integer,grids) && sort(grids)==[33,65,129] && get(record,"finest_level",0)==129
        forward=get(record,"base_origin_target_coverage",false)===true && record["job_status"]=="completed" && record["result_status"]=="completed" && level_complete && get(last(levels),"base_origin_target_coverage",false)===true && get(record,"independent_endpoint_solved",false)===true && get(record,"terminal_inventory_agreement",false)===true && get(record,"observed_target_root_count",0) isa Integer && get(record,"observed_target_root_count",0)>0
        push!(recovery_rows,Dict("source_job"=>record["source_job"],"source_result"=>record["source_result"],"role"=>record["target_role"],"base_point_id"=>get(record,"base_point_id","unreported"),"target_job_id"=>get(record,"target_job_id","unreported"),"accepted_forward_coverage"=>forward,"forward_base_origin_target_coverage"=>get(record,"base_origin_target_coverage",false)===true,"reported_forward_coverage"=>get(record,"base_origin_target_coverage","unreported"),"finest_grid"=>get(record,"finest_level","unreported"),"independent_endpoint_solved"=>get(record,"independent_endpoint_solved","unreported"),"reverse_target_origin_base_coverage"=>isempty(levels) ? "unreported" : get(last(levels),"reverse_target_origin_base_coverage","unreported"),"level_count"=>length(levels),"declared_levels_complete"=>level_complete,"terminal_inventory_agreement"=>get(record,"terminal_inventory_agreement","unreported"),"job_status"=>record["job_status"]))
    end
    roles=sort!(unique!([_targeted_role(job) for job in data.jobs]))
    progress=Dict{String,Any}[]
    for role in roles
        jobs=[job for job in data.jobs if _targeted_role(job)==role]
        push!(progress,Dict("role"=>role,"assigned"=>length(jobs),"completed"=>count(j->get(j,"status","")=="completed",jobs),"verified_results"=>count(j->haskey(data.pilot_results,string(get(j,"id",""))),jobs),"unresolved_or_unexecuted"=>count(j->get(j,"status","")!="completed",jobs)))
    end
    Dict{String,Any}("schema_version"=>1,"protocol"=>"targeted_followup_v1","study_scope"=>"Outcome-selected diagnostic follow-up to the first pilot; not a fresh held-out validation sample or a domain-completeness study","completeness"=>"CompletenessNotCertified","progress"=>progress,"diagnoses"=>diagnoses,"recoveries"=>recovery_rows,"transitions"=>transitions,"continuations"=>continuations,"graphs"=>graphs,"route_denominator"=>length([job for job in data.jobs if _targeted_role(job) in ("recovery_refinement","multiroot")]),"evaluated_recovery_routes"=>length(recovery_rows),"forward_covered_routes"=>count(row->row["accepted_forward_coverage"],recovery_rows),"unique_evaluated_target_ids"=>length(Set(row["target_job_id"] for row in recovery_rows if row["target_job_id"]!="unreported")),"issues"=>data.issues)
end

function _targeted_map_svg(io,graph)
    groups=graph["display_groups"];isempty(groups) && return print(io,"<p>No saved inventory nodes are available. Missing evidence does not establish disconnected regions.</p>")
    important=Set{String}([first(groups)["id"],last(groups)["id"]])
    for edge in graph["edges"]
        if edge["kind"] in ("change_bracket","event_candidate") || !isempty(get(edge["details"],"linked_event_ids",[]))
            for id in (edge["from"],edge["to"])
                group=get(graph["membership"],id,nothing);group===nothing || push!(important,group)
            end
        end
    end
    for (a,b) in zip(groups[1:end-1],groups[2:end])
        if a["signature"]!=b["signature"] || a["root_count"]!=b["root_count"]
            push!(important,a["id"]);push!(important,b["id"])
        end
    end
    shown=if length(groups)<=36
        groups
    else
        chosen=copy(important)
        remaining=max(0,36-length(chosen))
        candidates=[group for group in groups if !(group["id"] in chosen)]
        for index in unique(round.(Int,range(1,max(1,length(candidates));length=max(1,remaining))))
            remaining>0 && !isempty(candidates) && push!(chosen,candidates[index]["id"])
        end
        [group for group in groups if group["id"] in chosen]
    end
    positions=Dict(group["id"]=>(110+210*((index-1)%4),60+110*((index-1)÷4)) for (index,group) in enumerate(shown))
    height=120+110*((length(shown)-1)÷4)
    print(io,"<svg viewBox=\"0 0 900 ",height,"\" style=\"width:100%;max-width:1100px;background:#fff\" role=img aria-label=\"Observed local inventory adjacency, schematic layout\"><defs><marker id=\"",_targeted_anchor(graph["source_job"]),"-arrow\" markerWidth=8 markerHeight=8 refX=7 refY=4 orient=auto><path d=\"M0 0L8 4L0 8Z\" fill=\"#4b5866\"/></marker></defs>")
    rendered=Set{Tuple{String,String,String}}()
    for edge in graph["edges"]
        from=get(graph["membership"],edge["from"],"");to=get(graph["membership"],edge["to"],"")
        from==to && continue
        haskey(positions,from) && haskey(positions,to) || continue
        key=(from,to,edge["kind"]);key in rendered && continue;push!(rendered,key)
        x1,y1=positions[from];x2,y2=positions[to]
        dx=x2-x1;dy=y2-y1;boundary=max(abs(dx)/86,abs(dy)/33)
        x1+=dx/boundary;y1+=dy/boundary;x2-=dx/boundary;y2-=dy/boundary
        kind=edge["kind"]
        color=kind=="forward_track" ? "#096ba0" : kind=="reverse_track" ? "#8b4c9c" : kind=="change_bracket" ? "#b26112" : kind=="sampled_inventory_link" ? "#237548" : "#7d8790"
        dash=kind=="reverse_track" ? "8 4" : kind in ("unresolved","finite_interval") ? "2 5" : kind=="change_bracket" ? "10 3" : "none"
        marker=kind in ("forward_track","reverse_track") ? " marker-end=\"url(#$(_targeted_anchor(graph["source_job"]))-arrow)\"" : ""
        # Curvature separates overlapping forward/reverse observations. Geometry
        # is solely a readable schematic, never a parameter-distance estimate.
        offset=kind=="reverse_track" ? -26 : kind=="forward_track" ? 18 : 0
        mx=(x1+x2)/2;my=(y1+y2)/2+offset
        print(io,"<path d=\"M",x1," ",y1,"Q",mx," ",my," ",x2," ",y2,"\" fill=none stroke=\"",color,"\" stroke-width=2 stroke-dasharray=\"",dash,"\"",marker,"><title>",_report_escape(kind),": ",_report_escape(edge["status"]),"</title></path>")
        kind in ("unresolved","finite_interval") && print(io,"<text x=\"",mx,"\" y=\"",my-6,"\" font-size=14 fill=\"#626b73\">?</text>")
        kind=="event_candidate" && print(io,"<text x=\"",mx,"\" y=\"",my-6,"\" font-size=16>◇</text>")
        kind=="change_bracket" && print(io,"<text x=\"",mx,"\" y=\"",my-6,"\" font-size=16 fill=\"#9d5105\">△</text>")
    end
    event_index=0
    for witness in graph["witness_nodes"]
        witness["node_type"]=="event_witness" || continue
        contexts=[link for link in graph["witness_edges"] if link["kind"]=="event_side_context" && link["to"]==witness["id"]]
        points=[positions[group] for link in contexts for group in [get(graph["membership"],link["from"],"")] if haskey(positions,group)]
        isempty(points) && continue
        event_index+=1
        x=sum(first.(points))/length(points);y=sum(last.(points))/length(points)-38-14*((event_index-1)%2)
        for link in contexts
            group=get(graph["membership"],link["from"],"");haskey(positions,group) || continue
            sx,sy=positions[group]
            print(io,"<path d=\"M",sx," ",sy,"L",x," ",y,"\" fill=none stroke=\"#667581\" stroke-dasharray=\"3 4\"><title>",_report_escape(link["status"]),"; event context, not inventory membership</title></path>")
        end
        print(io,"<a href=\"#",_targeted_anchor(graph["source_job"]),"\"><polygon points=\"",x,",",y-8," ",x+8,",",y," ",x,",",y+8," ",x-8,",",y,"\" fill=white stroke=\"#53455c\"><title>",_report_escape(witness["saved_event_id"]),"; ",_report_escape(witness["classification"]),"; state=",_report_escape(witness["state"]),"</title></polygon><text x=\"",x+10,"\" y=\"",y-6,"\" font-size=12>E",event_index,"</text></a>")
    end
    for (index,group) in enumerate(shown)
        x,y=positions[group["id"]]
        unresolved=group["root_count"]==0 || occursin("unresolved",lowercase(group["signature"]))
        print(io,"<a href=\"#",_targeted_anchor(graph["source_job"]),"\"><rect x=\"",x-80,"\" y=\"",y-24,"\" width=160 height=54 rx=7 fill=\"",unresolved ? "#fff1df" : "#ecf3f7","\" stroke=\"#516b7b\"/>")
        print(io,"<text x=\"",x,"\" y=\"",y-3,"\" text-anchor=middle font-size=14>",length(group["point_ids"])>1 ? "Inventory run " : "Inventory ",index,"</text><text x=\"",x,"\" y=\"",y+15,"\" text-anchor=middle font-size=12>",length(group["point_ids"])," saved IDs; ",group["root_count"]," found roots",unresolved ? "; unresolved" : "","</text><title>",_report_escape(join(group["point_ids"],", ")),"</title></a>")
    end
    root_witnesses=[node for node in graph["witness_nodes"] if node["node_type"]=="root_witness"]
    print(io,"</svg><p>",length(root_witnesses)," saved individual root witness tuples and ",count(node->node["node_type"]=="event_witness",graph["witness_nodes"])," saved event witness tuples are retained separately from inventories. Inventory arrows summarize sampled root links. Dotted event-side links identify incident sampled intervals; they do not infer interval validation or inventory membership. Root-witness parameter contexts remain separate in the graph data.</p><p>",_report_escape(graph["scope"])," Collapsed runs retain every saved ID and original link in the downloadable graph data; shared signatures alone create no link.</p>")
    if !isempty(root_witnesses)
        print(io,"<details><summary>Individual root witnesses: first ",min(12,length(root_witnesses))," of ",length(root_witnesses)," saved tuples</summary><p>These are individual corrected roots, not complete inventories. All tuples and their source links remain in targeted_graphs.toml.</p>")
        rows=[Dict("origin"=>node["origin"],"track"=>node["track"],"inventory_context"=>node["inventory_context"],"state"=>string(node["state"]),"source_result"=>node["source_result"]) for node in first(root_witnesses,min(12,length(root_witnesses)))]
        _report_table(io,rows;compact=true);print(io,"</details>")
    end
    length(groups)>length(shown) && print(io,"<p>Preview shows ",length(shown)," of ",length(groups)," display nodes, preserving every change/event-incident node; repetitive context nodes are omitted only from this preview. Graph data retain every inventory, witness, and link.</p>")
    missing=count(edge->!edge["endpoint_inventories_saved"],graph["edges"])
    missing>0 && print(io,"<p>",missing," attempted links lack saved endpoint inventories and remain unresolved in the data; no placeholder is treated as an observed inventory.</p>")
end

function _targeted_compact_diagnosis(record)
    variants=Dict{String,Any}[]
    for variant in _report_records(get(record,"variants",[]))
        point=get(variant,"point",Dict());point isa AbstractDict || (point=Dict())
        roots=_report_records(get(point,"roots",[]))
        residuals=[root["residual"] for root in roots if get(root,"residual",nothing) isa Real && isfinite(root["residual"])]
        push!(variants,Dict("grid_size"=>get(variant,"grid_size","unreported"),"point_id"=>get(point,"id","unreported"),"found_roots"=>length(roots),"signature"=>get(point,"signature","unresolved"),"max_saved_residual"=>isempty(residuals) ? "unreported" : maximum(residuals),"singular_solver_attempt_count"=>get(variant,"singular_solver_attempt_count","unreported")))
    end
    Dict("source_job"=>record["source_job"],"source_result"=>record["source_result"],"source_job_id"=>get(record,"source_job_id","unreported"),"known_chart_seed_included"=>get(record,"known_chart_seed_included","unreported"),"variants"=>variants,"comparisons"=>get(record,"comparisons",[]),"analytic_diagnostics"=>get(record,"analytic_diagnostics",Dict()))
end

function _targeted_compact_continuation(record)
    branches=Dict{String,Any}[]
    for branch in _report_records(get(record,"branches",[]))
        push!(branches,Dict("direction"=>get(branch,"direction","unreported"),"termination"=>get(branch,"termination","unknown"),"accepted_samples"=>length(get(branch,"points",[])),"attempt_count"=>length(get(branch,"attempts",[])),"candidate_rows"=>length(get(branch,"candidates",[])),"extension_point_count"=>get(branch,"extension_point_count","unreported"),"prefix_comparison"=>get(branch,"prefix_comparison",Dict()),"finite_outcome"=>get(branch,"finite_outcome","unknown"),"point_scope"=>"single continued equilibrium; not a full inventory"))
    end
    Dict("source_job"=>record["source_job"],"source_result"=>record["source_result"],"source_job_id"=>get(record,"source_job_id","unreported"),"coordinate"=>get(record,"coordinate","unreported"),"bounds"=>get(record,"bounds",[]),"branches"=>branches)
end

function _targeted_compact_transition(record)
    keep=("id","from","to","t_bounds","width","original_bracket","status","kind","crosscheck_agreement","localization_stop_reason","localization_width_target_met","linked_event_ids","corrector_status")
    intervals=[Dict(string(k)=>v for (k,v) in row if string(k) in keep) for row in _report_records(get(record,"intervals",[]))]
    events=[Dict(string(k)=>v for (k,v) in row if string(k) in ("id","source_interval_id","event_type","kind","event_condition_residual","trace","determinant","classification","status","validated","parameters","state","affine_t","balance_residual","unscaled_balance_residual","event_residual","representative_estimate_id","raw_estimate_ids","grouping_scope","grouping_tolerance")) for row in _report_records(get(record,"events",[]))]
    Dict("source_job"=>record["source_job"],"source_result"=>record["source_result"],"source_job_id"=>get(record,"source_job_id","unreported"),"initial_brackets"=>get(record,"initial_brackets",[]),"intervals"=>intervals,"events"=>events,"inventory_location_count"=>get(record,"inventory_location_count","unreported"),"inventory_location_cap"=>get(record,"inventory_location_cap",256),"event_row_count"=>length(events),"raw_event_estimate_count"=>length(get(record,"event_estimates",[])),"event_count_scope"=>"Provisional repeat groups retain every raw estimate ID and use an actual representative; neither count establishes distinct physical bifurcations")
end

function _targeted_branch_plot(io,record)
    for branch in _report_records(get(record,"branches",[]))
        rows=Tuple{Float64,Float64,String}[]
        for point in _report_records(get(branch,"points",[]))
            state=get(point,"state",[]);parameter=get(point,"parameter",nothing)
            parameter isa Real && state isa AbstractVector && length(state)==2 && state[1] isa Real || continue
            push!(rows,(Float64(parameter),Float64(state[1]),string(get(point,"stability","unresolved"))))
        end
        _report_scatter(io,rows,"$(get(record,"coordinate","parameter")) (dimensionless)","E occupancy; one continued root","Direction $(get(branch,"direction","unreported")); termination $(get(branch,"termination","unknown")). Markers are accepted continuation samples, not full equilibrium inventories or certified continuous branches.")
    end
end

function _targeted_render_html(io,summary,data)
    print(io,"""<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Targeted zero-drive follow-up</title><style>body{font:16px/1.5 system-ui;color:#192f3b;background:#f7f9fb;max-width:1250px;margin:auto;padding:24px}h1,h2{line-height:1.2}a{color:#075f92}.notice{background:#fff1d9;border-left:4px solid #ad6810;padding:12px}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}th,td{border:1px solid #d7e0e6;padding:8px;text-align:left;vertical-align:top}th{background:#e9f0f5}pre{font:13px/1.4 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;max-width:45em}.compact th{white-space:nowrap}.compact td pre{min-width:9ch}section{margin:30px 0}summary{cursor:pointer;font-weight:600}.legend{background:#fff;padding:12px;border:1px solid #d7e0e6}.empty{color:#586d78}</style><h1>Targeted zero-drive follow-up</h1><p class=notice><strong>CompletenessNotCertified.</strong> Outcome-selected diagnostics of earlier unresolved and changed cases. This is not a fresh held-out validation sample. Missing or failed evidence does not establish disconnected parameter regions.</p><nav><a href=#map>Local adjacency</a> · <a href=#recovery>Recovery evidence</a> · <a href=#diagnoses>Diagnosis</a> · <a href=#transitions>Transition brackets</a> · <a href=#continuation>Continuation extensions</a> · <a href=#storage>Storage and provenance</a></nav><p><a href=targeted_summary.toml>Decision data</a> · <a href=targeted_graphs.toml>All map nodes and links</a> · <a href=targeted_manifest.toml>Frozen 22-job manifest</a></p>""")
    _report_table(io,summary["progress"];columns=["role","assigned","verified_results","completed","unresolved_or_unexecuted"],compact=true)
    print(io,"<section id=map><h2>Observed local adjacency</h2><p>Each box represents a saved finite root inventory, or an explicitly linked run sharing a signature. Every saved ID remains in the graph data. The layout is schematic: distances, box areas, and neighboring placement have no parameter-space meaning. Equal signatures alone add no edge.</p><div class=legend><strong>Legend:</strong> solid blue arrow: clean sampled base-origin root-corrector links; dashed purple arrow: clean sampled target-origin reverse links; green line: explicitly checked preservation over all 129 declared inventory samples; orange dashed line with △: observed sampled inventory-change bracket; gray dotted line with ?: finite-only or unresolved connection; ◇: numerical event candidate. None certifies a continuous path. Reverse success and independent root finding do not count as base-origin forward coverage.</div>")
    for graph in summary["graphs"]
        print(io,"<h3><a href=#",_targeted_anchor(graph["source_job"]),">",_report_escape(graph["source_job"]),"</a></h3>")
        _targeted_map_svg(io,graph)
    end
    isempty(summary["graphs"]) && print(io,"<p>No verified graph evidence is available yet; planned work remains unresolved.</p>")
    print(io,"</section><section id=recovery><h2>Forward, reverse, and independent evidence</h2><p>",summary["forward_covered_routes"]," of ",summary["route_denominator"]," prescribed routes have completed forward coverage with their required terminal checks; ",summary["evaluated_recovery_routes"]," route records concern ",summary["unique_evaluated_target_ids"]," distinct target IDs. Route counts are not distinct-region counts.</p>")
    _report_table(io,summary["recoveries"];compact=true)
    for record in _targeted_records(data,"targeted_recoveries")
        print(io,"<details id=",_targeted_anchor(record["source_job"]),"><summary>",_report_escape(record["source_job"]),": refinement levels and diagnostics</summary><p><a href=\"",_report_escape(record["source_result"]),"\">Full saved tracking attempts and independent inventories</a></p>")
        levels=Dict{String,Any}[]
        for level in _report_records(get(record,"levels",[]))
            forward=get(level,"base_origin_forward",Dict());reverse=get(level,"target_origin_reverse",Dict())
            push!(levels,Dict("grid_size"=>get(level,"grid_size","unreported"),"forward_base_target_coverage"=>get(level,"base_origin_target_coverage","unreported"),"reverse_target_base_coverage"=>get(level,"reverse_target_origin_base_coverage","unreported"),"forward_clean_endpoints"=>length(get(forward,"endpoint_clean_tracks",[])),"reverse_clean_endpoints"=>length(get(reverse,"endpoint_clean_tracks",[])),"forward_attempts"=>length(get(forward,"attempts",[])),"reverse_attempts"=>length(get(reverse,"attempts",[]))))
        end
        _report_table(io,levels;compact=true)
        print(io,"<p>Forward base-origin tracks and reverse target-origin tracks have different starting information. Independent inventories are searched without seeding from tracked roots. Finer sampled agreement does not prove branch identity or rule out missed roots.</p><h4>Terminal grid cross-check</h4>")
        _report_table(io,[Dict("grid13_assessed"=>get(record,"terminal_grid13_assessed",false),"terminal_inventory_agreement"=>get(record,"terminal_inventory_agreement","unreported"),"comparison"=>get(record,"terminal_grid_comparison",Dict()))])
        print(io,"<h4>Analytic diagnostics</h4>");_report_table(io,_report_records(get(record,"analytic_diagnostics",[]));compact=true)
        print(io,"</details>")
    end
    print(io,"</section><section id=diagnoses><h2>Failed-case diagnosis</h2><p>The 5×5, 9×9, and 13×13 grid comparisons include the declared common known chart seed where recorded. Agreement is therefore not three independent grid discoveries. These fixed-parameter method comparisons add no parameter-space adjacency edge.</p>")
    for record in summary["diagnoses"]
        print(io,"<details id=",_targeted_anchor(record["source_job"]),"><summary>",_report_escape(record["source_job"]),"</summary><p><a href=\"",_report_escape(record["source_result"]),"\">Raw diagnosis evidence</a></p>")
        _report_table(io,[Dict("known_chart_seed_included"=>record["known_chart_seed_included"])])
        _report_table(io,record["variants"];compact=true);_report_table(io,_report_records(record["comparisons"]);compact=true)
        _report_table(io,_report_records(record["analytic_diagnostics"]);compact=true);print(io,"</details>")
    end
    print(io,"</section><section id=transitions><h2>Changed intervals and event candidates</h2><p>Only the six prescribed original brackets are refined: at most 12 splits per original bracket, width target 10⁻⁴, and 256 inventory locations per path. Equal sampled endpoints do not exclude intervening changes. Corrector rows may repeat the same physical event and are not a count of distinct bifurcations.</p>")
    for record in summary["transitions"]
        print(io,"<details id=",_targeted_anchor(record["source_job"]),"><summary>",_report_escape(record["source_job"]),": sampled brackets and localizations</summary><p><a href=\"",_report_escape(record["source_result"]),"\">Raw bracket, cross-check, and event evidence</a></p><h4>Original changes versus fresh inventory</h4>")
        _report_table(io,_report_records(record["initial_brackets"]);compact=true)
        _report_table(io,record["intervals"];compact=true);_report_table(io,record["events"];compact=true)
        print(io,"<p>",record["event_row_count"]," provisional candidate groups from ",record["raw_event_estimate_count"]," raw successful estimates. ",_report_escape(record["event_count_scope"]),".</p></details>")
    end
    print(io,"</section><section id=continuation><h2>Seven paired continuation extensions</h2><p>Fourteen directions extend the saved seed and parameter bounds to a 96-step cap. These points are individual continued roots, so they are shown separately from inventory adjacency. Shared-prefix agreement does not establish depth saturation; failed or step-limited termination remains an open frontier.</p>")
    raw_continuations=_targeted_records(data,"targeted_continuations")
    for record in summary["continuations"]
        print(io,"<details id=",_targeted_anchor(record["source_job"]),"><summary>",_report_escape(record["source_job"]),"</summary><p><a href=\"",_report_escape(record["source_result"]),"\">Raw continuation points, correctors, and candidates</a></p>")
        _report_table(io,record["branches"];compact=true)
        raw=only(filter(row->row["source_job"]==record["source_job"],raw_continuations));_targeted_branch_plot(io,raw)
        print(io,"</details>")
    end
    print(io,"</section><section id=storage><h2>Storage, controls, and unresolved evidence</h2><p>The first packet remains immutable. Combined accounting reserves the verified first packet before allocating the new packet; regenerable reports use the same controlled writer and reserves as scientific evidence.</p>")
    _report_table(io,[get(summary,"storage",Dict("status"=>"unrecorded; no cap claim"))])
    print(io,"<h3>Method controls, separate from discovery</h3>");_report_table(io,get(summary,"controls",[]);compact=true)
    print(io,"<h3>Unresolved and deferred work</h3>");_report_table(io,summary["issues"];compact=true)
    print(io,"<p><a href=state.toml>Checkpoint</a> · <a href=config.toml>Configuration</a> · <a href=targeted_manifest.toml>Source hashes, baseline, and scientific policy</a></p></section></html>")
end

function write_targeted_report(output)
    output=realpath(output)
    for filename in ("index.html","targeted_summary.toml","targeted_graphs.toml")
        _report_check_target(joinpath(output,filename))
    end
    data=targeted_report_data(output)
    summary=targeted_report_summary(data)
    summary["diagnoses"]=_targeted_compact_diagnosis.(summary["diagnoses"])
    summary["transitions"]=_targeted_compact_transition.(summary["transitions"])
    summary["continuations"]=_targeted_compact_continuation.(summary["continuations"])
    summary["controls"]=_targeted_records(data,"controls")
    budget=isdefined(@__MODULE__,:storage_budget) ? storage_budget(output) : nothing
    isfile(joinpath(output,".storage.toml")) && budget===nothing && throw(ArgumentError("missing storage writer for governed targeted report"))
    metadata=get(data.state,"targeted",Dict())
    if budget!==nothing
        previous=get(get(data.state,"storage",Dict()),"peak_observed_bytes",0)
        previous isa Integer && previous>=0 && (budget.peak=max(budget.peak,previous))
        storage=storage_snapshot!(budget)
        reserved=data.manifest["source_packet"]["retained_bytes"]
        summary["storage"]=merge(storage,Dict("measurement_phase"=>"before_report_regeneration","first_packet_reserved_bytes"=>reserved,"combined_retained_bytes"=>reserved+storage["retained_bytes"],"combined_peak_observed_bytes"=>reserved+storage["peak_observed_bytes"],"combined_cap_bytes"=>data.manifest["combined_storage_cap_bytes"],"new_packet_cap_bytes"=>data.manifest["new_packet_cap_bytes"]))
    end
    _report_atomic_write(joinpath(output,"targeted_graphs.toml");budget=budget) do io
        TOML.print(io,Dict("scope"=>"Local sampled inventory evidence, not region connectivity certification","graphs"=>summary["graphs"]);sorted=true)
    end
    _report_atomic_write(joinpath(output,"targeted_summary.toml");budget=budget) do io
        TOML.print(io,Dict(string(k)=>v for (k,v) in summary if string(k)!="graphs");sorted=true)
    end
    _report_atomic_write(joinpath(output,"index.html");budget=budget) do io
        _targeted_render_html(io,summary,data)
    end
    joinpath(output,"index.html")
end

_targeted_valid_parameters(parameters) = parameters isa AbstractDict && !isempty(parameters) && all(value->value isa Real && isfinite(value),values(parameters))
_targeted_valid_state(state) = state isa AbstractVector && length(state)==2 && all(value->value isa Real && isfinite(value),state)

function _targeted_witnesses(record,inventories,edges;source_job="",source_result="")
    nodes=Dict{String,Any}[];links=Dict{String,Any}[];seen=Set{String}()
    lookup=Dict(node["id"]=>node for node in inventories)
    for edge in edges
        edge["kind"] in ("forward_track","reverse_track") || continue
        raw=edge["details"];root_ids=String[]
        for (point_id,state_key) in ((edge["from"],"from_state"),(edge["to"],"to_state"))
            haskey(lookup,point_id) || continue
            state=get(raw,state_key,nothing);_targeted_valid_state(state) || continue
            track=get(raw,"track","unreported")
            id="root:$source_job:$(edge["origin"]):$track:$point_id:$(_targeted_digest(Dict("state"=>state)))"
            if !(id in seen)
                push!(seen,id)
                push!(nodes,Dict("id"=>id,"node_type"=>"root_witness","state"=>state,"parameters"=>lookup[point_id]["parameters"],"inventory_context"=>point_id,"origin"=>edge["origin"],"track"=>track,"validated_sampled_corrector"=>true,"source_job"=>source_job,"source_result"=>source_result,"scope"=>"One saved corrected root tuple; parameter context does not assert membership in the independent root inventory"))
                push!(links,Dict("from"=>id,"to"=>point_id,"kind"=>"parameter_context","inventory_membership_asserted"=>false,"status"=>"saved_same_parameter_context"))
            end
            push!(root_ids,id)
        end
        length(root_ids)==2 && push!(links,Dict("from"=>root_ids[1],"to"=>root_ids[2],"kind"=>edge["kind"],"origin"=>edge["origin"],"source_edge_id"=>edge["id"],"status"=>edge["status"]))
    end
    for event in _report_records(get(record,"events",[]))
        event_id=string(get(event,"id",""));state=get(event,"state",nothing);parameters=get(event,"parameters",nothing)
        isempty(event_id) && continue
        _targeted_valid_state(state) && _targeted_valid_parameters(parameters) || continue
        id="event:$source_job:$event_id"
        id in seen && throw(ArgumentError("duplicate event witness ID"));push!(seen,id)
        push!(nodes,Dict("id"=>id,"saved_event_id"=>event_id,"node_type"=>"event_witness","state"=>state,"parameters"=>parameters,"kind"=>get(event,"kind",get(event,"event_type","unreported")),"classification"=>get(event,"classification","unclassified_candidate"),"validated"=>get(event,"validated",false),"affine_t"=>get(event,"affine_t","unreported"),"event_condition_residual"=>get(event,"event_condition_residual","unreported"),"raw_estimate_ids"=>get(event,"raw_estimate_ids",[]),"source_job"=>source_job,"source_result"=>source_result,"scope"=>"Saved numerical event/root tuple; not a full equilibrium inventory or proof of a distinct bifurcation"))
        for edge in edges
            event_id in get(edge["details"],"linked_event_ids",[]) || continue
            # Incident evidence is judged from this interval, independently of
            # the representative candidate's validation on another interval.
            status=edge["kind"]=="change_bracket" ? "sampled_side_context" : "unresolved_incident_interval"
            for inventory_id in (edge["from"],edge["to"])
                haskey(lookup,inventory_id) || continue
                push!(links,Dict("from"=>inventory_id,"to"=>id,"kind"=>"event_side_context","status"=>status,"source_edge_id"=>edge["id"],"interval_validation_inferred_from_event"=>false))
            end
        end
    end
    nodes,links
end
