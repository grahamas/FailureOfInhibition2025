# Included inside ZeroDriveAtlas. Uses only TOML and Julia Base.
using TOML
using SHA

_report_escape(x) = replace(string(x), '&' => "&amp;", '<' => "&lt;", '>' => "&gt;", '"' => "&quot;", '\'' => "&#39;")
_report_csv(x) = "\"" * replace(string(x), '"' => "\"\"") * "\""
_report_records(x) = x isa AbstractVector ? [v for v in x if v isa AbstractDict] : x isa AbstractDict ? [x] : AbstractDict[]

function _report_safe_file(output_dir, relative)
    relative isa AbstractString || return nothing
    isabspath(relative) && return nothing
    candidate = normpath(joinpath(output_dir, relative))
    root = realpath(output_dir)
    isfile(candidate) || return nothing
    resolved = realpath(candidate)
    (resolved == root || startswith(resolved, root * string(Base.Filesystem.path_separator))) || return nothing
    return resolved
end

"""Read only checkpoint-referenced evidence; malformed or absent results stay visible."""
function atlas_report_data(output_dir)
    islink(joinpath(output_dir, "state.toml")) && throw(ArgumentError("checkpoint must not be a symbolic link"))
    state = TOML.parsefile(joinpath(output_dir, "state.toml"))
    jobs = _report_records(get(state, "jobs", []))
    records = Dict(k => Dict{String,Any}[] for k in ("points", "paths", "events", "cycles", "certifications", "sections"))
    pilot_results = Dict{String,Any}()
    issues = Dict{String,Any}[]
    for job in jobs
        status = string(get(job, "status", "unknown"))
        status == "completed" || push!(issues, Dict("job" => get(job, "id", ""), "status" => status, "detail" => get(job, "error", "")))
        # Deferred jobs have assigned filenames but no result is expected yet.
        # Display their deferred status once rather than adding a missing-file
        # issue (or consuming a stale partial result from a running worker).
        status in ("pending", "running") && continue
        relative = get(job, "result_file", "")
        if isempty(string(relative))
            status == "completed" && push!(issues, Dict("job"=>get(job,"id",""), "status"=>"missing_result", "detail"=>"Completed job has no result reference"))
            continue
        end
        path = _report_safe_file(output_dir, relative)
        if path === nothing
            push!(issues, Dict("job" => get(job, "id", ""), "status" => "missing_result", "detail" => string(relative)))
            continue
        end
        recorded_hash = get(job, "result_hash", nothing)
        if !(recorded_hash isa AbstractString)
            push!(issues, Dict("job"=>get(job,"id",""), "status"=>"unverified_result", "detail"=>"Result has no recorded SHA-256 hash; scientific evidence excluded"))
            continue
        end
        raw_result = read(path)
        if bytes2hex(sha256(raw_result)) != recorded_hash
            push!(issues, Dict("job"=>get(job,"id",""), "status"=>"unverified_result", "detail"=>"Result SHA-256 mismatch; scientific evidence excluded"))
            continue
        end
        result = try
            # Parse the same bytes whose hash was checked, avoiding a second
            # read that could consume a changed file after verification.
            TOML.parse(String(raw_result))
        catch err
            push!(issues, Dict("job" => get(job, "id", ""), "status" => "unreadable_result", "detail" => sprint(showerror, err)))
            continue
        end
        evidence = get(result, "evidence", Dict())
        evidence isa AbstractDict || continue
        payload = get(job, "payload", job)
        if payload isa AbstractDict && (haskey(payload, "pilot_role") || haskey(payload, "target_role"))
            pilot_results[string(get(job,"id",""))] = Dict("evidence"=>evidence,"status"=>get(result,"status","unknown"),"source_result"=>relative)
        end
        # Failed jobs may have useful raw evidence, but it cannot establish atlas nodes.
        begin
            for key in keys(records), item in _report_records(get(evidence, key, []))
                row = Dict{String,Any}(string(k) => v for (k,v) in item)
                row["source_job"] = get(job, "id", "")
                row["source_result"] = relative
                row["eligible_for_regions"] = status == "completed" && get(result,"status","") == "completed"
                if payload isa AbstractDict && haskey(payload,"pilot_role")
                    row["pilot_role"] = payload["pilot_role"]
                    row["pilot_cohort"] = get(payload,"pilot_cohort","")
                end
                push!(records[key], row)
            end
        end
    end
    return (; state, jobs, records, issues, pilot_results)
end

"""Components require explicit examined paths with preserved complete signatures."""
function atlas_region_components(points, paths)
    nodes = Dict{String,Any}()
    for point in points
        id = string(get(point, "id", ""))
        isempty(id) && continue
        get(point,"eligible_for_regions",true) === true || continue
        roots = get(point, "roots", nothing)
        roots isa AbstractVector && !isempty(roots) || continue
        signature = get(point, "signature", nothing)
        signature isa AbstractString && !isempty(signature) && !occursin("unresolved", lowercase(signature)) || continue
        haskey(nodes, id) && throw(ArgumentError("duplicate atlas point id: $id"))
        nodes[id] = point
    end
    adjacency = Dict(id => String[] for id in keys(nodes))
    for path in paths
        a, b = string(get(path, "from", "")), string(get(path, "to", ""))
        haskey(nodes, a) && haskey(nodes, b) || continue
        get(path, "validated", false) === true || continue
        get(path, "signature_preserved", false) === true || continue
        get(path, "status", "") == "completed" || continue
        # An inventory-checked path includes its intermediate sampled points.
        # Reject the entire connection if any declared sample is missing or
        # ineligible; silently falling back to endpoints would discard evidence.
        sample_ids = get(path, "sample_ids", nothing)
        sample_ids isa AbstractVector || continue
        ids = string.(sample_ids)
        # The current connection protocol inventories both endpoints and three
        # distinct interior parameter samples. Endpoint agreement alone is not
        # sufficient evidence for a region connection.
        length(ids) >= 5 && length(unique(ids)) == length(ids) && first(ids) == a && last(ids) == b || continue
        all(id -> haskey(nodes, id), ids) || continue
        signature = get(nodes[a], "signature", nothing)
        signature !== nothing && all(id -> get(nodes[id], "signature", nothing) == signature, ids) || continue
        for (left, right) in zip(ids[1:end-1], ids[2:end])
            push!(adjacency[left], right); push!(adjacency[right], left)
        end
    end
    components = Vector{Vector{String}}()
    unseen = Set(keys(nodes))
    while !isempty(unseen)
        start = minimum(unseen)
        queue = [start]; delete!(unseen, start)
        component = String[]
        while !isempty(queue)
            current = popfirst!(queue); push!(component, current)
            for neighbor in sort(adjacency[current])
                neighbor in unseen || continue
                delete!(unseen, neighbor); push!(queue, neighbor)
            end
        end
        push!(components, sort(component))
    end
    return components
end

function _report_table(io, rows; columns=nothing, compact=false)
    isempty(rows) && return print(io, "<p class=empty>No records established in this run.</p>")
    if compact && length(rows) > 200
        print(io,"<p>Showing 200 of ",length(rows)," saved records; the checkpoint links the complete raw evidence.</p>")
        rows = rows[1:200]
    end
    columns === nothing && (columns = sort!(unique!(String[string(k) for row in rows for k in keys(row)])))
    print(io,compact ? "<div class=scroll><table class=compact><thead><tr>" : "<div class=scroll><table><thead><tr>")
    for key in columns; print(io, "<th>", _report_escape(key), "</th>"); end
    print(io, "</tr></thead><tbody>")
    for row in rows
        print(io, "<tr>")
        for key in columns
            value = get(row, key, "")
            if key == "source_result" && value isa AbstractString
                print(io, "<td><a href=\"",_report_escape(value),"\">Raw evidence</a></td>")
            elseif value isa AbstractDict || value isa AbstractVector
                if compact
                    print(io,"<td>",value isa AbstractVector ? string(length(value)," records") : string(length(value)," fields"),"; see raw evidence</td>")
                else
                    print(io,"<td><details><summary>",value isa AbstractVector ? string(length(value)," records") : "Details","</summary><pre>",_report_escape(value),"</pre></details></td>")
                end
            else
                print(io, "<td><pre>", _report_escape(value), "</pre></td>")
            end
        end
        print(io, "</tr>")
    end
    print(io, "</tbody></table></div>")
end

function _report_check_target(path)
    islink(path) && throw(ArgumentError("report output must not be a symbolic link: $path"))
    ispath(path) && !isfile(path) && throw(ArgumentError("report output must be a regular file: $path"))
    nothing
end

"""Replace one generated file atomically, without following an existing target."""
function _report_atomic_write(writer, path; budget=nothing)
    _report_check_target(path)
    if budget !== nothing
        return storage_atomic_write(path,writer;budget=budget,category="report")
    end
    temporary, io = mktemp(dirname(path))
    try
        writer(io)
        flush(io)
        close(io)
        _report_check_target(path)
        # Base.Filesystem.rename falls back to copy/remove on failure. Use the
        # POSIX operation directly so failure cannot become a non-atomic copy.
        result = ccall(:rename, Cint, (Cstring, Cstring), temporary, path)
        Base.systemerror("atomic report rename", result != 0)
    finally
        isopen(io) && close(io)
        ispath(temporary) && rm(temporary)
    end
    nothing
end

function _report_write_csv(path, columns, rows; budget=nothing)
    _report_atomic_write(path;budget=budget) do io
        println(io, join(_report_csv.(columns), ','))
        for row in rows; println(io, join([_report_csv(get(row, c, "")) for c in columns], ',')); end
    end
end

function atlas_timing_rows(jobs)
    rows = Dict{String,Any}[]
    for kind in sort!(unique!([string(get(job, "kind", "unknown")) for job in jobs]))
        selected = [j for j in jobs if string(get(j, "kind", "unknown")) == kind]
        timings = sort!([Float64(j["elapsed_seconds"]) for j in selected if get(j, "status", "") in ("completed", "unresolved", "failed") && get(j, "elapsed_seconds", nothing) isa Real && isfinite(j["elapsed_seconds"]) && j["elapsed_seconds"] >= 0])
        n = length(timings)
        median = n == 0 ? "unmeasured" : isodd(n) ? timings[(n+1)÷2] : (timings[n÷2]+timings[n÷2+1])/2
        p95 = n == 0 ? "unmeasured" : timings[max(1, ceil(Int, 0.95n))]
        push!(rows, Dict("kind" => kind, "attempted" => count(j -> get(j,"status","") ∉ ("pending", "running"), selected), "uncensored_samples" => n, "completed_samples"=>count(j->get(j,"status","")=="completed",selected), "timeout_samples"=>count(j->get(j,"status","")=="timed_out",selected), "projected_jobs_per_hour_at_median"=>median isa Real && median>0 ? 3600/median : "unmeasured",
            "median_seconds" => median, "p95_seconds" => p95, "failures_or_unresolved" => count(j -> get(j,"status","") ∈ ("failed", "timed_out", "unresolved"), selected)))
    end
    rows
end

"""Render the saved atlas evidence without launching scientific computations."""
function write_report(output_dir)
    output_dir = realpath(output_dir)
    # Validate the complete output set before replacing even the first file.
    for filename in ("index.html", "base_cases.csv", "timings.csv", "coverage.toml")
        _report_check_target(joinpath(output_dir, filename))
    end
    data = atlas_report_data(output_dir)
    state, jobs, records, issues = data.state, data.jobs, data.records, data.issues
    config_path = joinpath(output_dir, "config.toml")
    islink(config_path) && throw(ArgumentError("report configuration must not be a symbolic link"))
    config = isfile(config_path) ? TOML.parsefile(config_path) : Dict{String,Any}()
    targeted_active = if isdefined(@__MODULE__, :targeted_run_marker)
        targeted_run_marker(output_dir, state, config)
    else
        any(key -> haskey(config, key), ("protocol", "targeted_manifest", "targeted_manifest_hash")) ||
        any(key -> haskey(state, key), ("protocol", "targeted_manifest_hash", "targeted")) ||
        ispath(joinpath(output_dir, "targeted_manifest.toml")) ||
        islink(joinpath(output_dir, "targeted_manifest.toml")) ||
        any(job -> haskey(get(job, "payload", Dict()), "target_role") ||
            startswith(string(get(job, "kind", "")), "targeted_"), jobs)
    end
    if targeted_active
        isdefined(@__MODULE__, :write_targeted_report) &&
            isdefined(@__MODULE__, :verify_targeted_inputs) ||
            throw(ArgumentError("targeted reports require the report module and frozen-input verifier"))
        return write_targeted_report(output_dir)
    end
    pilot_active = haskey(state,"pilot") || any(job -> get(_report_pilot_payload(job),"pilot_role","") != "",jobs)
    if isdefined(@__MODULE__, :storage_budget)
        budget = storage_budget(output_dir)
    else
        isfile(joinpath(output_dir,".storage.toml")) && throw(ArgumentError("storage-governed reports require the storage writer"))
        budget = nothing
    end
    pilot_summary = pilot_active ? atlas_pilot_summary(data) : nothing
    if pilot_active
        # Protected method controls have their own pilot evidence section and
        # must not enter discovery-region counts or candidate-base tables.
        records = Dict(key => [row for row in rows if get(row,"pilot_role","") != "control"] for (key,rows) in records)
    end
    if pilot_active
        _report_check_target(joinpath(output_dir,"pilot_summary.toml"))
        if budget !== nothing
            previous_peak=get(get(state,"storage",Dict()),"peak_observed_bytes",0)
            previous_peak isa Integer && previous_peak>=0 && (budget.peak=max(budget.peak,previous_peak))
            pilot_summary["storage"] = merge(storage_snapshot!(budget),Dict("measurement_phase"=>"before_report_regeneration"))
        end
    end
    components = atlas_region_components(records["points"], records["paths"])
    bases = Dict{String,Any}[]
    for (i, component) in enumerate(components)
        push!(bases, Dict("kind"=>pilot_active ? "observed_graph_representative" : "region_representative", "id"=>first(component), "component"=>i, "members"=>join(component, ";"),
            "evidence"=>"Observed connections only; lexicographically first point; path ledger supplies connections."))
    end
    known_points = Set(string(get(point,"id","")) for point in records["points"])
    witness_rows = Dict{Tuple{String,String},Dict{String,Any}}()
    for key in ("events", "cycles"), item in records[key]
        get(item,"validated",false) === true || continue
        if key == "cycles"
            kind = "periodic_witness"
            ids = get(item,"witness_point_ids",String[])
        elseif get(item,"kind","") in ("local_neighborhood_witness", "section_side_witnesses")
            kind = "local_neighborhood_witness"
            ids = get(item,"witness_point_ids",String[])
        else
            # A numerically validated nexus or nearby section sample is not
            # itself an observed bracket on two sides of a transition.
            kind = "transition_witness"
            ids = get(item,"bracket_point_ids",String[])
            ids isa AbstractVector && length(unique(ids)) >= 2 || continue
        end
        ids isa AbstractVector || continue
        all(id -> string(id) in known_points, ids) || continue
        for raw_id in ids
            id = string(raw_id)
            witness_rows[(kind,id)] = Dict("kind"=>kind,"id"=>id,"component"=>"","members"=>"","evidence"=>get(item,"source_job",""))
        end
    end
    append!(bases,[witness_rows[key] for key in sort!(collect(keys(witness_rows)))])
    _report_write_csv(joinpath(output_dir,"base_cases.csv"), ["kind","id","component","members","evidence"], bases;budget=budget)
    timings = atlas_timing_rows(jobs)
    _report_write_csv(joinpath(output_dir,"timings.csv"), ["kind","attempted","uncensored_samples","completed_samples","timeout_samples","median_seconds","p95_seconds","projected_jobs_per_hour_at_median","failures_or_unresolved"], timings;budget=budget)
    coverage = Dict("completeness"=>"CompletenessNotCertified", "jobs"=>length(jobs), "pending_jobs"=>count(j->get(j,"status","") ∈ ("pending","running"),jobs),
        "observed_points"=>length(records["points"]), "observed_components"=>length(components), "examined_paths"=>length(records["paths"]), "issues"=>length(issues),
        "jobs_with_issues"=>length(Set(string(get(issue,"job","")) for issue in issues)),
        "base_case_role_rows"=>length(bases), "unique_base_case_points"=>length(Set(string(row["id"]) for row in bases)),
        "frontier"=>"Unexamined finite parameters, unsampled strata and paths, unbounded shells, excluded h=0/r=0/infinite limits, and unsearched attractors remain unresolved.")
    _report_atomic_write(joinpath(output_dir,"coverage.toml");budget=budget) do io; TOML.print(io,coverage;sorted=true); end
    destination = joinpath(output_dir,"index.html")
    _report_atomic_write(destination;budget=budget) do io
        print(io, """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zero-drive parameter atlas</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#182b39;background:#f7f9fb;max-width:1250px;margin:auto;padding:24px}h1,h2{line-height:1.2}nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#075c8c}.notice{border-left:4px solid #b16b11;background:#fff2db;padding:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d9e0e6;padding:8px;text-align:left;vertical-align:top}th{background:#e9f0f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:13px/1.4 ui-monospace,monospace;max-width:38em}.empty{color:#506575}.compact th{white-space:nowrap}.compact td pre{min-width:10ch}section{margin:32px 0}summary{cursor:pointer;font-weight:600}</style>
<h1>Zero-drive parameter atlas</h1><p class=notice><strong>CompletenessNotCertified.</strong> This report records finite numerical evidence. Sparse samples do not establish filled parameter regions; equal signatures do not establish connectivity.</p>
<nav><a href="#coverage">Coverage</a><a href="#bases">Base cases</a><a href="#points">Points</a><a href="#paths">Paths</a><a href="#events">Transitions</a><a href="#cycles">Cycles</a><a href="#certifications">Point certificates</a><a href="#issues">Unresolved</a><a href="#timing">Costs</a><a href="#provenance">Provenance</a></nav>
<section id=coverage><h2>Coverage and frontier</h2>""")
        pilot_active && _report_pilot_html(io,pilot_summary)
        _report_table(io, [coverage])
        print(io,"<p>Stop reason: ",_report_escape(get(state,"stop_reason","unspecified")),". Mode: ",_report_escape(get(state,"mode","unspecified")),". A budget or job limit ends this protocol; it does not establish absence of additional structures.</p></section><section id=bases><h2>Base cases and connections</h2><p>Representatives belong to provisional components of explicitly examined, signature-preserving connections. Path validity is numerical and scoped to saved evidence. Local neighborhood samples, observed transition brackets, and periodic witnesses retain separate roles. Role rows may share a point; the coverage table reports unique base-case points separately.</p><p><a href=base_cases.csv>Download base cases</a></p>")
        _report_table(io,bases;compact=pilot_active)
        print(io,"</section>")
        print(io,"<section id=plots><h2>Saved branch and section samples</h2><p>Markers are saved numerical samples, without interpolation or filled parameter regions. Root ordering does not imply branch identity.</p>")
        _report_scientific_plots(io,records;compact=pilot_active)
        print(io,"</section>")
        for key in ("points","paths","events","cycles","certifications")
            title = Dict("points"=>"Parameter points and quantitative readouts","paths"=>"Examined branches and paths","events"=>"Transitions and nexus investigations","cycles"=>"Periodic-orbit evidence","certifications"=>"Fixed-parameter certificates")[key]
            print(io,"<section id=",key,"><h2>",title,"</h2>")
            key == "certifications" && print(io,"<p>A certificate applies to equilibria at its exact parameter tuple over the state rectangle. It does not certify parameter neighborhoods, stability, or all attractors.</p>")
            key == "events" && print(io,"<p>Read the event status and nondegeneracy evidence. Node–focus boundaries and projected curve intersections are not automatically bifurcations or organizing nexuses.</p>")
            _report_table(io,records[key];compact=pilot_active); print(io,"</section>")
        end
        print(io,"<section id=issues><h2>Unresolved and deferred work</h2>"); _report_table(io,issues;compact=pilot_active)
        print(io,"</section><section id=timing><h2>Measured costs</h2><p>Setup/warm-up wall time: ",_report_escape(get(state,"setup_seconds","unrecorded"))," seconds. Measured run wall time: ",_report_escape(get(state,"elapsed_seconds","unrecorded"))," seconds. Compilation within individual first jobs may remain included.</p><p>Uncensored terminal-job wall times include completed, unresolved, and failed jobs, grouped by kind. Timeouts are counted separately as censored observations. Jobs-per-hour projections divide 3600 by the observed median; they are throughput illustrations, not campaign cost guarantees. Small sample sizes, startup overhead, and difficult parameter neighborhoods limit extrapolation. Choose the pilot budget after reviewing these measurements.</p><p><a href=timings.csv>Download timings</a></p>"); _report_table(io,timings)
        print(io,"</section><section id=provenance><h2>Saved provenance</h2><p><a href=state.toml>Checkpoint and result references</a> · <a href=config.toml>Run configuration</a> · <a href=coverage.toml>Coverage ledger</a></p><details><summary>Checkpoint metadata</summary><pre>")
        metadata = Dict(string(k)=>v for (k,v) in state if k != "jobs")
        print(io,pilot_active ? "See state.toml for complete checkpoint metadata; raw data are linked rather than duplicated in this regenerable report." : _report_escape(sprint(tio->TOML.print(tio,metadata;sorted=true))),"</pre></details></section></html>")
    end
    if pilot_active
        _report_atomic_write(joinpath(output_dir,"pilot_summary.toml");budget=budget) do io
            TOML.print(io,pilot_summary;sorted=true)
        end
    end
    return destination
end

write_atlas_report(output_dir) = write_report(output_dir)

function _report_scatter(io, rows, xlabel, ylabel, caption)
    isempty(rows) && return
    finite = [r for r in rows if isfinite(r[1]) && isfinite(r[2])]
    isempty(finite) && return
    xs, ys = first.(finite), getindex.(finite,2)
    xmin,xmax=extrema(xs); ymin,ymax=extrema(ys)
    dx=xmax-xmin; dy=ymax-ymin
    dx == 0 && (dx=1.0; xmin-=0.5; xmax+=0.5)
    dy == 0 && (dy=1.0; ymin-=0.5; ymax+=0.5)
    xmap(x)=65+485*(x-xmin)/dx; ymap(y)=270-230*(y-ymin)/dy
    print(io,"<figure><svg viewBox=\"0 0 600 330\" role=img aria-label=\"",_report_escape(caption),"\" style=\"width:100%;max-width:700px;background:white\"><path d=\"M65 35V270H560\" fill=none stroke=\"#394d5b\"/>")
    for r in finite
        x,y=xmap(r[1]),ymap(r[2]);event=length(r)>=3 ? r[3] : "sample"
        if event == "fold_candidate" || event == "hopf_candidate"
            print(io,"<path d=\"M",x-4," ",y-4,"l8 8m0 -8l-8 8\" stroke=\"#a44213\"><title>",_report_escape(event),"</title></path>")
        elseif event == "trace_zero_without_positive_determinant"
            print(io,"<polygon data-marker=\"trace-zero-no-positive-determinant\" points=\"",x,",",y-4," ",x+4,",",y," ",x,",",y+4," ",x-4,",",y,"\" fill=\"none\" stroke=\"#626b73\"><title>Trace-zero point without a resolved positive determinant</title></polygon>")
        else
            print(io,"<circle cx=\"",x,"\" cy=\"",y,"\" r=2.8 fill=\"#086b9a\"><title>",_report_escape(event),"</title></circle>")
        end
    end
    for (x,y,label) in ((65,290,string(round(xmin;sigdigits=4))),(535,290,string(round(xmax;sigdigits=4))),(5,270,string(round(ymin;sigdigits=4))),(5,45,string(round(ymax;sigdigits=4))),(250,320,xlabel),(65,20,ylabel))
        print(io,"<text x=\"",x,"\" y=\"",y,"\" font-size=13>",_report_escape(label),"</text>")
    end
    print(io,"</svg><figcaption>",_report_escape(caption),"</figcaption></figure>")
end

function _report_scientific_plots(io,records;compact=false)
    job_ids = sort!(unique!([string(get(p,"source_job","")) for p in records["points"]]))
    if compact
        print(io,"<p>Pilot preview: at most six job projections and three sections; complete saved records remain linked in the checkpoint.</p>")
        job_ids = first(job_ids,min(6,length(job_ids)))
    end
    for job in job_ids
        points=[p for p in records["points"] if string(get(p,"source_job",""))==job && get(p,"parameters",nothing) isa AbstractDict]
        length(points)>=2 || continue
        coordinates=[k for k in ("A","B","C","D","a","b","h","r") if all(p->get(p["parameters"],k,nothing) isa Real,points) && length(unique(p["parameters"][k] for p in points))>1]
        isempty(coordinates) && continue
        coordinate=first(coordinates)
        rows=Tuple{Float64,Float64,String}[]
        for p in points,root in get(p,"roots",[])
            state=get(root,"state",[]);length(state)==2 || continue
            push!(rows,(Float64(p["parameters"][coordinate]),Float64(state[1]),string(get(root,"stability","unresolved"))))
        end
        varied = join(coordinates, ", ")
        _report_scatter(io,rows,"$coordinate (dimensionless)","E (occupancy)","Job $job: projection of saved equilibrium samples; varying parameter coordinates: $varied. Other coordinates may vary; this is not necessarily a fixed-parameter slice. Unconnected markers do not assign root identity.")
    end
    sections = compact ? first(records["sections"],min(3,length(records["sections"]))) : records["sections"]
    for section in sections
        pair=get(section,"parameter_pair",[]);length(pair)==2 || continue
        rows=Tuple{Float64,Float64,String}[]
        for sample in get(section,"section_samples",[])
            get(sample,"status","")=="sampled" || continue
            p=get(sample,"parameters",Dict());all(k->get(p,k,nothing) isa Real,pair) || continue
            push!(rows,(Float64(p[pair[1]]),Float64(p[pair[2]]),"section sample"))
        end
        for curve in get(section,"transition_curves",[]), sample in get(curve,"points",[])
            p=get(sample,"parameters",Dict());all(k->get(p,k,nothing) isa Real,pair) || continue
            push!(rows,(Float64(p[pair[1]]),Float64(p[pair[2]]),string(get(sample,"classification","transition candidate"))))
        end
        _report_scatter(io,rows,"$(pair[1]) (dimensionless)","$(pair[2]) (dimensionless)","Job $(get(section,"source_job","")): local two-parameter section; blue circles are probes, crosses are fold/Hopf candidates, gray diamonds are trace-zero points without a resolved positive determinant. No filled region or continuous-curve claim.")
    end
end

_report_pilot_payload(job) = get(job,"payload",job) isa AbstractDict ? get(job,"payload",job) : Dict{String,Any}()

function _report_pilot_bases(metadata)
    selection=get(metadata,"base_selection",nothing)
    selection isa AbstractDict || return Dict{String,Any}("hash_verified"=>false,"status"=>"missing_or_malformed_frozen_selection","entries"=>Dict{String,Any}[],"recovery_routes"=>Dict{String,Any}[])
    stream=IOBuffer();TOML.print(stream,selection;sorted=true)
    verified=bytes2hex(sha256(take!(stream)))==get(metadata,"base_selection_hash",nothing)
    entries=[Dict(string(k)=>v for (k,v) in entry if string(k) in ("point_id","source_job","cohort","signature","parameters","audit_status","cohort_audit_warning")) for entry in _report_records(get(selection,"entries",[]))]
    routes=[Dict(string(k)=>v for (k,v) in route if string(k)!="parameters" && !(v isa AbstractDict) && !(v isa AbstractVector)) for route in _report_records(get(selection,"recovery_routes",[]))]
    Dict{String,Any}("hash_verified"=>verified,"frozen_before_heldout"=>get(selection,"frozen_before_heldout",false),"rule"=>get(selection,"rule","unreported"),"scope"=>get(selection,"scope","unreported"),"source_role"=>get(selection,"source_role","unreported"),"entries"=>entries,"recovery_routes"=>routes,"excluded_candidates"=>_report_records(get(selection,"excluded_candidates",[])))
end

function _report_pilot_rows(data, key)
    rows=Dict{String,Any}[]
    expected_role=get(Dict("controls"=>"control","audits"=>"audit","recoveries"=>"recovery"),key,"")
    for id in sort!(collect(keys(data.pilot_results)))
        saved=data.pilot_results[id]
        job=only(filter(job->string(get(job,"id",""))==id,data.jobs))
        get(_report_pilot_payload(job),"pilot_role","")==expected_role || continue
        for raw in _report_records(get(saved["evidence"],key,[]))
            push!(rows,merge(Dict{String,Any}(string(k)=>v for (k,v) in raw),Dict("source_job"=>id,"source_result"=>saved["source_result"],"pilot_sample_index"=>get(_report_pilot_payload(job),"pilot_sample_index","unreported"),"pilot_design_position"=>get(_report_pilot_payload(job),"pilot_design_position","unreported"),"pilot_pair"=>get(_report_pilot_payload(job),"pilot_pair","unreported"),"verified_completed"=>get(job,"status","")=="completed" && get(saved,"status","")=="completed")))
        end
    end
    rows
end

function _report_pilot_changes(row)
    samples=get(row,"samples",[])
    samples isa AbstractVector || return Dict{String,Any}[]
    indices=Dict(string(get(sample,"id",""))=>index for (index,sample) in enumerate(samples) if sample isa AbstractDict)
    changes=Dict{String,Any}[]
    for change in _report_records(get(row,"changes",[]))
        compact=Dict{String,Any}(string(k)=>v for (k,v) in change if string(k) in ("first_point_id","second_point_id","first_signature","second_signature"))
        for (idkey,fractionkey) in (("first_point_id","first_path_fraction"),("second_point_id","second_path_fraction"))
            index=get(indices,string(get(compact,idkey,"")),nothing)
            compact[fractionkey]=index===nothing || length(samples)<2 ? "unavailable" : (index-1)/(length(samples)-1)
        end
        push!(changes,compact)
    end
    changes
end

function _report_pilot_design(manifest,jobs)
    ledger=_report_records(get(manifest,"cohort_designs",[]))
    pool=get(manifest,"selection_pool",Int[])
    pool_valid=pool isa AbstractVector && !isempty(pool) && all(x->x isa Integer && x>0,pool) && length(unique(pool))==length(pool)
    compact=[Dict{String,Any}(string(k)=>v for (k,v) in row if string(k)!="samples") for row in ledger]
    if haskey(manifest,"cohort_designs")
        mapping=Dict{String,Any}();verified=!isempty(ledger)
        for row in ledger
            cohort=string(get(row,"cohort",""))
            if isempty(cohort) || haskey(mapping,cohort)
                verified=false
            else
                mapping[cohort]=row
            end
            train=get(row,"training_indices",nothing);held=get(row,"heldout_indices",nothing);indices=get(row,"indices",nothing)
            valid=train isa AbstractVector && held isa AbstractVector && indices isa AbstractVector && !isempty(train) && !isempty(held) && all(x->x isa Integer && x>0,indices) && length(unique(indices))==length(indices) && indices==vcat(train,held) && isempty(intersect(train,held))
            verified &= valid && pool_valid && all(index->index in pool,indices)
        end
        for job in jobs
            payload=_report_pilot_payload(job);role=get(payload,"pilot_role","")
            role in ("training","heldout","audit") || continue
            row=get(mapping,string(get(payload,"pilot_cohort","")),nothing)
            if row===nothing
                verified=false;continue
            end
            index=get(payload,"pilot_sample_index",nothing);position=get(payload,"pilot_design_position",nothing)
            indices=get(row,"indices",[])
            allowed=get(row,role=="heldout" ? "heldout_indices" : role=="audit" ? "audit_indices" : "training_indices",[])
            valid=indices isa AbstractVector && allowed isa AbstractVector && position isa Integer && 1<=position<=length(indices) && indices[position]==index && index in allowed
            verified &= valid
        end
        training=Int[];heldout=Int[] # accepted indices are cohort-local, not global blocks
    else
        training=get(manifest,"training_indices",Int[]);heldout=get(manifest,"heldout_indices",Int[])
        verified=training isa AbstractVector && heldout isa AbstractVector && !isempty(training) && !isempty(heldout) && isempty(intersect(training,heldout))
        training isa AbstractVector || (training=Int[])
        heldout isa AbstractVector || (heldout=Int[])
        verified &= all(jobs) do job
            payload=_report_pilot_payload(job);role=get(payload,"pilot_role","")
            role=="training" ? get(payload,"pilot_sample_index",nothing) in training : role=="heldout" ? get(payload,"pilot_sample_index",nothing) in heldout : true
        end
    end
    Dict{String,Any}("verified"=>verified,"cohort_designs"=>compact,"training_indices"=>training isa AbstractVector ? training : Int[],"heldout_indices"=>heldout isa AbstractVector ? heldout : Int[],"selection_algorithm"=>get(manifest,"selection_algorithm",haskey(manifest,"cohort_designs") ? "unreported" : "legacy declared accepted-index blocks"),"selection_seed"=>get(manifest,"selection_seed","unreported"),"selection_pool"=>pool isa AbstractVector ? pool : Int[],"selection_pool_size"=>pool isa AbstractVector ? length(pool) : 0,"selection_pool_bounds"=>get(manifest,"selection_pool_bounds",Int[]),"selection_encoding"=>get(manifest,"selection_encoding","unreported"),"support_ledger"=>_report_records(get(manifest,"support_ledger",[])),"validation_scope"=>"Declared cohort-local role/index separation and assigned job design-position membership; no statistical independence or representativeness claim")
end

function _report_pilot_display_indices(rows)
    [Dict(string(k)=>(v isa AbstractVector && all(x->x isa Integer,v) ? join(v,", ") : v) for (k,v) in row) for row in rows]
end

"""Finite pilot accounting; completed jobs alone never imply scientific success."""
function atlas_pilot_summary(data)
    metadata=get(data.state,"pilot",Dict{String,Any}())
    metadata isa AbstractDict || (metadata=Dict{String,Any}())
    manifest=get(metadata,"manifest_counts",Dict{String,Any}())
    manifest isa AbstractDict || (manifest=Dict{String,Any}())
    protocol_manifest=get(metadata,"manifest",Dict{String,Any}())
    protocol_manifest isa AbstractDict || (protocol_manifest=Dict{String,Any}())
    design=_report_pilot_design(protocol_manifest,data.jobs)
    training_indices=design["training_indices"];heldout_indices=design["heldout_indices"];separated=design["verified"]
    roles=("control","training","training_followup","audit","heldout","recovery")
    aliases=Dict("control"=>("control","controls"),"training"=>("training",),"training_followup"=>("training_followup_quota","training_followup","training_followups"),"audit"=>("audit","audits"),"heldout"=>("heldout",),"recovery"=>("recovery","recoveries"))
    cohorts=Dict{String,Any}[];role_totals=Dict{String,Any}[]
    for role in roles
        selected=[job for job in data.jobs if get(_report_pilot_payload(job),"pilot_role","")==role]
        planned=nothing
        for key in aliases[role]
            value=get(manifest,key,nothing)
            value isa Integer && value>=0 && (planned=Int(value);break)
        end
        row=Dict{String,Any}("role"=>role,"assigned_jobs"=>length(selected),"verified_results"=>count(j->haskey(data.pilot_results,string(get(j,"id",""))),selected),"planned_jobs"=>planned===nothing ? "not_recorded" : planned,"quota_semantics"=>role=="training_followup" ? "optional_capacity" : "mandatory")
        for status in ("completed","unresolved","failed","timed_out","pending","running")
            row[status]=count(j->get(j,"status","")==status,selected)
        end
        row["not_yet_assigned"]=planned===nothing ? "unknown" : max(0,planned-length(selected))
        push!(role_totals,row)
        names=sort!(unique!([string(get(_report_pilot_payload(j),"pilot_cohort","unspecified")) for j in selected]))
        for name in names
            group=[j for j in selected if string(get(_report_pilot_payload(j),"pilot_cohort","unspecified"))==name]
            indices=sort!(unique!([Int(v) for j in group for v in [get(_report_pilot_payload(j),"pilot_sample_index",nothing)] if v isa Integer]))
            push!(cohorts,Dict("role"=>role,"cohort"=>name,"assigned"=>length(group),"sample_indices"=>indices,"completed"=>count(j->get(j,"status","")=="completed",group),"verified_results"=>count(j->haskey(data.pilot_results,string(get(j,"id",""))),group),"unresolved_or_unexecuted"=>count(j->get(j,"status","")!="completed",group)))
        end
    end
    controls=_report_pilot_rows(data,"controls")
    control_rows=[Dict("control"=>get(row,"control","unspecified"),"status"=>get(row,"status","unknown"),"scope"=>get(row,"scope","unknown"),"verified_completed"=>row["verified_completed"],"source_job"=>row["source_job"],"source_result"=>row["source_result"]) for row in controls]
    audits=_report_pilot_rows(data,"audits");audit_rows=Dict{String,Any}[]
    for row in audits
        comparisons=_report_records(get(row,"comparisons",[]))
        depth=get(row,"continuation_pair",Dict())
        depth isa AbstractDict || (depth=Dict())
        statuses=[string(get(item,"status","unknown")) for item in comparisons]
        extensions=_report_records(get(depth,"depth_extensions",[]))
        repeat=get(row,"training_repeat_comparison",Dict());repeat isa AbstractDict || (repeat=Dict())
        prefix_count=get(depth,"shared_prefix_count",0)
        prefix_difference=get(depth,"shared_prefix_max_state_difference",nothing)
        prefix_tolerance=get(depth,"comparison_tolerance",nothing)
        prefix_verified=prefix_count isa Integer && prefix_count>0 && prefix_difference isa Real && isfinite(prefix_difference) && prefix_tolerance isa Real && isfinite(prefix_tolerance) && prefix_tolerance>0 && 0<=prefix_difference<=prefix_tolerance
        push!(audit_rows,Dict("cohort"=>get(row,"cohort","unspecified"),"sample_index"=>row["pilot_sample_index"],"design_position"=>row["pilot_design_position"],"pair_id"=>row["pilot_pair"],"resolved_signature_disagreement_comparisons"=>count(item->get(item,"status","")!="agreement_observed" && get(item,"inventory_match",false)===true && get(item,"classification_resolved",false)===true,comparisons),"inventory_disagreement_comparisons"=>count(item->get(item,"inventory_match",nothing)===false,comparisons),"inventory_comparison_unknown"=>count(item->!(get(item,"inventory_match",nothing) isa Bool),comparisons),"unresolved_classification_comparisons"=>count(item->get(item,"classification_resolved",nothing)===false,comparisons),"training_repeat_inventory_match"=>get(repeat,"inventory_match","unreported"),"paired_comparisons"=>length(statuses),"resolved_classification_comparisons"=>count(item->get(item,"classification_resolved",false)===true,comparisons),"training_repeat_status"=>get(repeat,"status","unreported"),"training_repeat_classification_resolved"=>get(repeat,"classification_resolved",false),"extension_point_count"=>sum(get(item,"extension_point_count",0) for item in extensions;init=0),"extension_candidate_count"=>sum(get(item,"new_candidate_count",0) for item in extensions;init=0),"extension_stability_changes"=>sum(get(item,"extension_stability_changes",0) for item in extensions;init=0),"deep_step_limit_directions"=>count(item->get(item,"deep_termination","")=="step_limit",extensions),"deep_unsuccessful_or_unknown_terminations"=>count(item->get(item,"deep_termination","") ∉ ("step_limit","parameter_boundary"),extensions),"shallow_unsuccessful_or_unknown_terminations"=>count(item->get(item,"shallow_termination","") ∉ ("step_limit","parameter_boundary"),extensions),"depth_extensions"=>extensions,"depth_extension_records"=>length(extensions),"shared_prefix_criterion_verified"=>prefix_verified,"shared_prefix_count"=>prefix_count,"agreement_observed"=>count(==("agreement_observed"),statuses),"disagreement_or_unresolved"=>count(!=("agreement_observed"),statuses),"unknown_comparison_statuses"=>count(status->status ∉ ("agreement_observed","disagreement_or_unresolved"),statuses),"verified_completed"=>row["verified_completed"],"depth_comparison_status"=>get(depth,"comparison_status","unresolved_missing_comparison_criterion"),"shared_prefix_max_state_difference"=>get(depth,"shared_prefix_max_state_difference","unreported"),"comparison_tolerance"=>get(depth,"comparison_tolerance","unreported"),"source_job"=>row["source_job"],"source_result"=>row["source_result"]))
    end
    recoveries=_report_pilot_rows(data,"recoveries");recovery_rows=Dict{String,Any}[]
    for row in recoveries
        changes=get(row,"changes",[])
        target_id=string(get(row,"target_job_id",""))
        targets=filter(job->string(get(job,"id",""))==target_id && get(_report_pilot_payload(job),"pilot_role","")=="heldout",data.jobs)
        target_verified=length(targets)==1 && get(first(targets),"status","")=="completed" && haskey(data.pilot_results,target_id) && get(data.pilot_results[target_id],"status","")=="completed"
        samples=get(row,"samples",[])
        row_evidence_complete=row["verified_completed"] && target_verified && get(row,"status","")=="sampled_tracked_target_coverage" && get(row,"tracked_observed_endpoint_coverage",false)===true && get(row,"heldout_inventory_available",false)===true && samples isa AbstractVector && length(samples)>=17 && get(row,"independent_endpoint_solved",false)===true && get(row,"saved_target_root_count",0) isa Integer && get(row,"saved_target_root_count",0)>0 && get(row,"observed_target_root_count",0) isa Integer && get(row,"matched_target_roots",-1)==get(row,"observed_target_root_count",0) && get(row,"observed_target_root_count",0)>0 && get(row,"saved_target_root_count",0)<=get(row,"observed_target_root_count",0)
        push!(recovery_rows,Dict("target_cohort"=>get(row,"target_cohort","unspecified"),"target_job_id"=>get(row,"target_job_id","unspecified"),"target_sample_index"=>length(targets)==1 ? get(_report_pilot_payload(first(targets)),"pilot_sample_index","unreported") : "unreported","target_design_position"=>length(targets)==1 ? get(_report_pilot_payload(first(targets)),"pilot_design_position","unreported") : "unreported","base_point_id"=>get(row,"base_point_id","unavailable"),"base_signature_match_after_routing"=>get(row,"base_target_signature_equal",get(row,"base_signature_eligible","unreported")),"status"=>get(row,"status","unknown"),"tracked_target_coverage"=>get(row,"tracked_target_coverage",false)===true && row_evidence_complete,"reported_tracked_target_coverage"=>get(row,"tracked_target_coverage",false)===true,"heldout_result_verified"=>target_verified,"endpoint_coverage_evidence_complete"=>row_evidence_complete,"heldout_inventory_available"=>get(row,"heldout_inventory_available",target_verified),"tracked_observed_endpoint_coverage"=>get(row,"tracked_observed_endpoint_coverage","unreported"),"independent_endpoint_solved"=>get(row,"independent_endpoint_solved",false)===true,"sampled_inventory_preserved"=>get(row,"sampled_inventory_preserved",false)===true,"saved_target_roots"=>get(row,"saved_target_root_count","unreported"),"observed_target_roots"=>get(row,"observed_target_root_count","unreported"),"matched_target_roots"=>get(row,"matched_target_roots","unreported"),"structural_changes_observed"=>changes isa AbstractVector ? length(changes) : "unreported","sampled_signature_change_brackets"=>_report_pilot_changes(row),"sampled_points"=>get(row,"samples",nothing) isa AbstractVector ? length(row["samples"]) : 0,"source_job"=>row["source_job"],"source_result"=>row["source_result"]))
    end
    reasons=String[];actions=String[]
    bases=_report_pilot_bases(metadata)
    if !separated
        push!(actions,"reassess");push!(reasons,"training_heldout_index_separation_missing_or_violated")
    end
    if !bases["hash_verified"] || get(bases,"frozen_before_heldout",false)!==true
        push!(actions,"reassess");push!(reasons,"frozen_training_base_selection_missing_unverified_or_not_declared_frozen_before_heldout")
    end
    for (role,rows) in (("control",control_rows),("audit",audit_rows),("recovery",recovery_rows))
        expected=[string(job["id"]) for job in data.jobs if get(_report_pilot_payload(job),"pilot_role","")==role]
        if any(id->count(row->row["source_job"]==id,rows)!=1,expected)
            push!(actions,"reassess");push!(reasons,"missing_or_duplicate_$(role)_scientific_evidence")
        end
    end
    # The protocol handler supplies control validation statuses. Unknown statuses
    # remain unresolved rather than being interpreted from job completion.
    passed_controls=count(r->r["verified_completed"] && get(r,"status","")=="passed",control_rows)
    failed_controls=count(r->get(r,"status","")=="failed",control_rows)
    failed_controls>0 && (push!(actions,"repair");push!(reasons,"known_control_validation_failed: investigate the method; failed periodic-control shooting is not evidence of cycle absence"))
    any(r->!r["verified_completed"] || r["paired_comparisons"]<2 || r["unknown_comparison_statuses"]>0 || r["inventory_disagreement_comparisons"]>0 || r["resolved_signature_disagreement_comparisons"]>0 || r["inventory_comparison_unknown"]>0 || r["training_repeat_inventory_match"]!==true || r["training_repeat_inventory_match"]===true && r["training_repeat_classification_resolved"]===true && r["training_repeat_status"]!="agreement_observed",audit_rows) && (push!(actions,"reassess");push!(reasons,"fixed_parameter_inventory_disagreement_or_missing_comparison_evidence"))
    any(r->r["inventory_disagreement_comparisons"]==0 && r["inventory_comparison_unknown"]==0 && (r["unresolved_classification_comparisons"]>0 || r["resolved_classification_comparisons"]<r["paired_comparisons"] || r["training_repeat_classification_resolved"]!==true),audit_rows) && (push!(actions,"reallocate");push!(reasons,"resolved_inventory_matching_with_classification_uncertainty_requires_targeted_boundary_characterization; not_automatically_a_solver_defect"))
    any(r->r["depth_comparison_status"]!="shared_prefix_consistent" || !r["shared_prefix_criterion_verified"] || r["depth_extension_records"]<2,audit_rows) && (push!(actions,"reassess");push!(reasons,"continuation_shared_prefix_consistency_not_resolved"))
    any(r->r["deep_step_limit_directions"]>0 || r["extension_candidate_count"]>0 || r["extension_stability_changes"]>0,audit_rows) && (push!(actions,"reallocate");push!(reasons,"deeper_continuation_adds_transition_evidence_or_retains_step_limited_frontier; shared_prefix_consistency_is_not_depth_saturation"))
    any(r->r["deep_unsuccessful_or_unknown_terminations"]>0 || r["shallow_unsuccessful_or_unknown_terminations"]>0,audit_rows) && (push!(actions,"reassess");push!(reasons,"continuation_termination_unsuccessful_or_unknown; prefix_consistency_does_not_resolve_a_corrector_failure"))
    any(r->!r["tracked_target_coverage"],recovery_rows) && (push!(actions,"reassess");push!(reasons,"base_to_target_tracking_has_coverage_gap_or_ambiguity"))
    any(issue->get(issue,"status","") in ("unverified_result","unreadable_result","missing_result"),data.issues) && (push!(actions,"repair");push!(reasons,"missing_or_unverified_scientific_evidence"))
    incomplete=any(row->row["planned_jobs"]=="not_recorded" || row["not_yet_assigned"]!="unknown" && row["not_yet_assigned"]>0 || row["completed"]<row["assigned_jobs"],filter(row->row["role"]!="training_followup",role_totals))
    any(row->row["role"]=="training_followup" && row["completed"]<row["assigned_jobs"],role_totals) && (push!(actions,"reallocate");push!(reasons,"assigned_optional_training_followup_unresolved; unused_capacity_is_optional_but_assigned_results_remain_a_frontier"))
    any(row->row["planned_jobs"] isa Integer && row["assigned_jobs"]>row["planned_jobs"],role_totals) && (push!(actions,"reassess");push!(reasons,"assigned_job_count_exceeds_declared_role_quota"))
    incomplete && (push!(actions,"reallocate");push!(reasons,"finite_protocol_has_unexecuted_or_unresolved_work; use recorded cohort denominators before allocating more exploration"))
    passed_controls<length(control_rows) && (push!(actions,"reassess");push!(reasons,"one_or_more_control_outcomes_not_resolved_as_passed"))
    expected_controls=first(filter(row->row["role"]=="control",role_totals))
    if isempty(control_rows) || expected_controls["verified_results"]<expected_controls["assigned_jobs"]
        push!(actions,"reassess");push!(reasons,"protected_controls_not_fully_evaluated")
    end
    recovery_role=first(filter(row->row["role"]=="recovery",role_totals))
    evaluated_targets=length(Set(row["target_job_id"] for row in recovery_rows if row["target_job_id"]!="unspecified"))
    tracked_targets=length(Set(row["target_job_id"] for row in recovery_rows if row["tracked_target_coverage"]))
    if evaluated_targets<length(recovery_rows)
        push!(actions,"reassess");push!(reasons,"duplicate_or_missing_recovery_target_identity; cohorts_with_multiple_heldout_indices_must_keep_targets_distinct")
    end
    if recovery_role["planned_jobs"] isa Integer && recovery_role["completed"]==recovery_role["planned_jobs"] && (evaluated_targets!=recovery_role["planned_jobs"] || tracked_targets<recovery_role["planned_jobs"])
        push!(actions,"reassess");push!(reasons,"completed_recovery_job_count_does_not_establish_planned_unique_target_coverage")
    end
    if isempty(actions) && !isempty(audit_rows) && !isempty(recovery_rows)
        push!(actions,"explore");push!(reasons,"completed_finite_protocol_checks_support_considering_another_bounded_exploration; this is not parameter_domain_completeness")
    elseif isempty(actions)
        push!(actions,"reassess");push!(reasons,"scientific_checks_missing; job_completion_alone_is_insufficient")
    end
    Dict{String,Any}("schema_version"=>1,"completeness"=>"CompletenessNotCertified","phase"=>get(metadata,"stage",get(metadata,"phase","unreported")),"role_totals"=>role_totals,"cohorts"=>cohorts,"sampling_design"=>design,"holdout_design"=>get(protocol_manifest,"holdout_design","unreported"),"training_indices"=>training_indices isa AbstractVector ? training_indices : Int[],"heldout_indices"=>heldout_indices isa AbstractVector ? heldout_indices : Int[],"index_separation_verified"=>separated,"frozen_base_selection"=>bases,"controls"=>control_rows,"paired_audits"=>audit_rows,"recoveries"=>recovery_rows,"recovery_planned_targets"=>recovery_role["planned_jobs"],"recovery_evaluated_targets"=>evaluated_targets,"recovery_records"=>length(recovery_rows),"recovery_tracked_targets"=>tracked_targets,"recovery_unresolved_or_unexecuted_targets"=>recovery_role["planned_jobs"] isa Integer ? max(0,recovery_role["planned_jobs"]-tracked_targets) : "unknown_planned_denominator","recommendations"=>sort!(unique!(actions)),"reason_codes"=>sort!(unique!(reasons)),"interpretation"=>"Controls test methods and are separate from discovery. Base compression is a frozen structural-signature grouping hypothesis, not a region-connectivity proof. Base-to-target routing uses parameter distance; held-out signature matches are diagnostics after selection, not routing criteria. Sampled straight-path signature changes are retained observations, not automatic recovery failures or established bifurcation discoveries. Singleton components do not count as newly discovered regions. Recommendations require author judgment and authorize no automatic follow-on run.")
end

function _report_pilot_html(io,summary)
    print(io,"<section id=pilot><h2>Bounded pilot: decisions and open questions</h2><p>",_report_escape(summary["interpretation"]),"</p><p><a href=pilot_summary.toml>Download pilot decision summary</a></p><p>Suggested actions: <strong>",_report_escape(join(summary["recommendations"],", ")),"</strong>. These are evidence flags for review.</p>")
    _report_table(io,[Dict("reason"=>reason) for reason in summary["reason_codes"]])
    print(io,"<p>Protected controls are excluded from discovery graph counts below. Generic observed graph representatives are sampling bookkeeping, not novelty counts or the frozen compression-test base pool.</p><h3>Frozen compression-test bases</h3><p>The checkpoint's pre-held-out base-selection ledger defines this hypothesis; held-out points cannot retroactively become its bases.</p>")
    _report_table(io,[Dict(string(k)=>v for (k,v) in summary["frozen_base_selection"] if string(k) ∉ ("entries","recovery_routes","excluded_candidates"))])
    _report_table(io,summary["frozen_base_selection"]["entries"];compact=true)
    print(io,"<details><summary>Candidates excluded by direct fixed-parameter audits</summary>")
    _report_table(io,get(summary["frozen_base_selection"],"excluded_candidates",[]);compact=true);print(io,"</details>")
    print(io,"<details><summary>Pre-held-out parameter-only recovery routes</summary>")
    _report_table(io,summary["frozen_base_selection"]["recovery_routes"];compact=true);print(io,"</details>")
    print(io,"<h3>Protected cohorts and denominators</h3>");_report_table(io,summary["role_totals"];columns=["role","quota_semantics","planned_jobs","assigned_jobs","verified_results","completed","unresolved","failed","timed_out","pending","not_yet_assigned"],compact=true)
    print(io,"<p>Cohort labels preserve shell, chart, and zero-coupling mask. Counts describe this finite design, not coverage of the unbounded domain. The recorded selection design is deterministic; it does not establish statistical independence or representative parameter-space coverage.</p>")
    design=summary["sampling_design"]
    if isempty(design["cohort_designs"])
        _report_table(io,_report_pilot_display_indices([Dict("declared_design"=>summary["holdout_design"],"training_indices"=>summary["training_indices"],"heldout_indices"=>summary["heldout_indices"],"index_separation_verified"=>summary["index_separation_verified"])]))
    else
        _report_table(io,[Dict("declared_design"=>summary["holdout_design"],"index_separation_verified"=>summary["index_separation_verified"],"index_scope"=>"cohort-specific accepted indices; see frozen ledger")])
    end
    _report_table(io,_report_pilot_display_indices([Dict(string(k)=>v for (k,v) in design if string(k) ∉ ("cohort_designs","support_ledger","training_indices","heldout_indices","selection_pool"))]))
    print(io,"<details><summary>Frozen cohort-specific accepted-index ledger</summary>")
    _report_table(io,_report_pilot_display_indices(design["cohort_designs"]);compact=true);print(io,"</details>")
    print(io,"<details><summary>Declared coordinate-support ledger</summary><p>Marginal support descriptions characterize this finite design; they are not coverage or independence guarantees.</p>")
    _report_table(io,design["support_ledger"];compact=false);print(io,"</details>")
    _report_table(io,_report_pilot_display_indices(summary["cohorts"]);compact=true)
    print(io,"<p>Inventory and tolerance audits test fixed-parameter root searches and resolved classifications, including repeated saved training results. They do not separately establish event-localization sensitivity. Shared-prefix consistency checks deterministic continuation overlap; extra points, candidates, stability changes, and step-limit terminations describe the remaining depth frontier, not branch completeness.</p>")
    for (key,title) in (("controls","Method controls (not discovery)"),("paired_audits","Paired grid, tolerance, and depth checks"),("recoveries","Sampled straight-path base-to-held-out recovery"))
        print(io,"<h3>",title,"</h3>");_report_table(io,summary[key];compact=true)
    end
    brackets=Dict{String,Any}[]
    for recovery in summary["recoveries"],change in recovery["sampled_signature_change_brackets"]
        push!(brackets,merge(change,Dict("target_job_id"=>recovery["target_job_id"],"source_result"=>recovery["source_result"])))
    end
    print(io,"<details><summary>Sampled signature-change brackets along straight paths</summary><p>Path fraction 0 is the frozen base and 1 is the held-out target in the recorded dimensionless coordinates. These are adjacent sampled signature changes, which can include unresolved classifications; they are not automatically localized bifurcations or continuous-path certificates.</p>")
    _report_table(io,brackets;compact=true);print(io,"</details>")
    heldout_total=only(filter(row->row["role"]=="heldout",summary["role_totals"]))["planned_jobs"]
    print(io,"<p>Recovery subset: ",_report_escape(summary["recovery_planned_targets"])," prespecified targets out of ",_report_escape(heldout_total)," planned held-out points. Target denominator: ",_report_escape(summary["recovery_planned_targets"])," planned; ",summary["recovery_evaluated_targets"]," evaluated; ",summary["recovery_tracked_targets"]," with observed tracked target coverage; ",_report_escape(summary["recovery_unresolved_or_unexecuted_targets"])," unresolved or unexecuted. Structural changes along sampled paths are retained explicitly; endpoint agreement does not certify a continuous path or all equilibria.</p>")
    if haskey(summary,"storage")
        print(io,"<h3>Storage guard and measured footprint</h3>");_report_table(io,[summary["storage"]])
    else
        print(io,"<p>Storage footprint and cap were not recorded by an available storage policy; no storage-bound claim is made in this summary.</p>")
    end
    print(io,"<p>The HTML and summary are regenerable derivatives. Full raw attempts are linked rather than duplicated here. Storage measurements are explicitly timed snapshots of the run-owned tree under the recorded enforcement policy, not a whole-system filesystem quota.</p></section>")
end
