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
    issues = Dict{String,Any}[]
    for job in jobs
        status = string(get(job, "status", "unknown"))
        status == "completed" || push!(issues, Dict("job" => get(job, "id", ""), "status" => status, "detail" => get(job, "error", "")))
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
        # Failed jobs may have useful raw evidence, but it cannot establish atlas nodes.
        begin
            for key in keys(records), item in _report_records(get(evidence, key, []))
                row = Dict{String,Any}(string(k) => v for (k,v) in item)
                row["source_job"] = get(job, "id", "")
                row["source_result"] = relative
                row["eligible_for_regions"] = status == "completed" && get(result,"status","") == "completed"
                push!(records[key], row)
            end
        end
    end
    return (; state, jobs, records, issues)
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

function _report_table(io, rows; columns=nothing)
    isempty(rows) && return print(io, "<p class=empty>No records established in this run.</p>")
    columns === nothing && (columns = sort!(unique!(String[string(k) for row in rows for k in keys(row)])))
    print(io, "<div class=scroll><table><thead><tr>")
    for key in columns; print(io, "<th>", _report_escape(key), "</th>"); end
    print(io, "</tr></thead><tbody>")
    for row in rows
        print(io, "<tr>")
        for key in columns
            value = get(row, key, "")
            if key == "source_result" && value isa AbstractString
                print(io, "<td><a href=\"",_report_escape(value),"\">Raw evidence</a></td>")
            elseif value isa AbstractDict || value isa AbstractVector
                print(io,"<td><details><summary>",value isa AbstractVector ? string(length(value)," records") : "Details","</summary><pre>",_report_escape(value),"</pre></details></td>")
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
function _report_atomic_write(writer, path)
    _report_check_target(path)
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

function _report_write_csv(path, columns, rows)
    _report_atomic_write(path) do io
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
    components = atlas_region_components(records["points"], records["paths"])
    bases = Dict{String,Any}[]
    for (i, component) in enumerate(components)
        push!(bases, Dict("kind"=>"region_representative", "id"=>first(component), "component"=>i, "members"=>join(component, ";"),
            "evidence"=>"Observed connections only; lexicographically first point; path ledger supplies connections."))
    end
    for (key, kind) in (("events", "transition_witness"), ("cycles", "periodic_witness")), item in records[key]
        get(item,"validated",false) === true || continue
        for id in get(item,"witness_point_ids",String[])
            push!(bases, Dict("kind"=>kind,"id"=>id,"component"=>"","members"=>"","evidence"=>get(item,"source_job","")))
        end
    end
    _report_write_csv(joinpath(output_dir,"base_cases.csv"), ["kind","id","component","members","evidence"], bases)
    timings = atlas_timing_rows(jobs)
    _report_write_csv(joinpath(output_dir,"timings.csv"), ["kind","attempted","uncensored_samples","completed_samples","timeout_samples","median_seconds","p95_seconds","projected_jobs_per_hour_at_median","failures_or_unresolved"], timings)
    coverage = Dict("completeness"=>"CompletenessNotCertified", "jobs"=>length(jobs), "pending_jobs"=>count(j->get(j,"status","") ∈ ("pending","running"),jobs),
        "observed_points"=>length(records["points"]), "observed_components"=>length(components), "examined_paths"=>length(records["paths"]), "issues"=>length(issues),
        "frontier"=>"Unexamined finite parameters, unsampled strata and paths, unbounded shells, excluded h=0/r=0/infinite limits, and unsearched attractors remain unresolved.")
    _report_atomic_write(joinpath(output_dir,"coverage.toml")) do io; TOML.print(io,coverage;sorted=true); end
    destination = joinpath(output_dir,"index.html")
    _report_atomic_write(destination) do io
        print(io, """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zero-drive parameter atlas</title>
<style>body{font:16px/1.5 system-ui,sans-serif;color:#182b39;background:#f7f9fb;max-width:1250px;margin:auto;padding:24px}h1,h2{line-height:1.2}nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#075c8c}.notice{border-left:4px solid #b16b11;background:#fff2db;padding:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d9e0e6;padding:8px;text-align:left;vertical-align:top}th{background:#e9f0f5}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:13px/1.4 ui-monospace,monospace;max-width:38em}.empty{color:#506575}section{margin:32px 0}summary{cursor:pointer;font-weight:600}</style>
<h1>Zero-drive parameter atlas</h1><p class=notice><strong>CompletenessNotCertified.</strong> This report records finite numerical evidence. Sparse samples do not establish filled parameter regions; equal signatures do not establish connectivity.</p>
<nav><a href="#coverage">Coverage</a><a href="#bases">Base cases</a><a href="#points">Points</a><a href="#paths">Paths</a><a href="#events">Transitions</a><a href="#cycles">Cycles</a><a href="#certifications">Point certificates</a><a href="#issues">Unresolved</a><a href="#timing">Costs</a><a href="#provenance">Provenance</a></nav>
<section id=coverage><h2>Coverage and frontier</h2>""")
        _report_table(io, [coverage])
        print(io,"<p>Stop reason: ",_report_escape(get(state,"stop_reason","unspecified")),". Mode: ",_report_escape(get(state,"mode","unspecified")),". A budget or job limit ends this protocol; it does not establish absence of additional structures.</p></section><section id=bases><h2>Base cases and connections</h2><p>Representatives belong to provisional components of explicitly examined, signature-preserving connections. Path validity is numerical and scoped to saved evidence. Transition and periodic witnesses are listed separately.</p><p><a href=base_cases.csv>Download base cases</a></p>")
        _report_table(io,bases)
        print(io,"</section>")
        print(io,"<section id=plots><h2>Saved branch and section samples</h2><p>Markers are saved numerical samples, without interpolation or filled parameter regions. Root ordering does not imply branch identity.</p>")
        _report_scientific_plots(io,records)
        print(io,"</section>")
        for key in ("points","paths","events","cycles","certifications")
            title = Dict("points"=>"Parameter points and quantitative readouts","paths"=>"Examined branches and paths","events"=>"Transitions and nexus investigations","cycles"=>"Periodic-orbit evidence","certifications"=>"Fixed-parameter certificates")[key]
            print(io,"<section id=",key,"><h2>",title,"</h2>")
            key == "certifications" && print(io,"<p>A certificate applies to equilibria at its exact parameter tuple over the state rectangle. It does not certify parameter neighborhoods, stability, or all attractors.</p>")
            key == "events" && print(io,"<p>Read the event status and nondegeneracy evidence. Node–focus boundaries and projected curve intersections are not automatically bifurcations or organizing nexuses.</p>")
            _report_table(io,records[key]); print(io,"</section>")
        end
        print(io,"<section id=issues><h2>Unresolved and deferred work</h2>"); _report_table(io,issues)
        print(io,"</section><section id=timing><h2>Measured costs</h2><p>Setup/warm-up wall time: ",_report_escape(get(state,"setup_seconds","unrecorded"))," seconds. Measured run wall time: ",_report_escape(get(state,"elapsed_seconds","unrecorded"))," seconds. Compilation within individual first jobs may remain included.</p><p>Uncensored terminal-job wall times include completed, unresolved, and failed jobs, grouped by kind. Timeouts are counted separately as censored observations. Jobs-per-hour projections divide 3600 by the observed median; they are throughput illustrations, not campaign cost guarantees. Small sample sizes, startup overhead, and difficult parameter neighborhoods limit extrapolation. Choose the pilot budget after reviewing these measurements.</p><p><a href=timings.csv>Download timings</a></p>"); _report_table(io,timings)
        print(io,"</section><section id=provenance><h2>Saved provenance</h2><p><a href=state.toml>Checkpoint and result references</a> · <a href=config.toml>Run configuration</a> · <a href=coverage.toml>Coverage ledger</a></p><details><summary>Checkpoint metadata</summary><pre>")
        metadata = Dict(string(k)=>v for (k,v) in state if k != "jobs")
        print(io,_report_escape(sprint(tio->TOML.print(tio,metadata;sorted=true))),"</pre></details></section></html>")
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
        else
            print(io,"<circle cx=\"",x,"\" cy=\"",y,"\" r=2.8 fill=\"#086b9a\"><title>",_report_escape(event),"</title></circle>")
        end
    end
    for (x,y,label) in ((65,290,string(round(xmin;sigdigits=4))),(535,290,string(round(xmax;sigdigits=4))),(5,270,string(round(ymin;sigdigits=4))),(5,45,string(round(ymax;sigdigits=4))),(250,320,xlabel),(65,20,ylabel))
        print(io,"<text x=\"",x,"\" y=\"",y,"\" font-size=13>",_report_escape(label),"</text>")
    end
    print(io,"</svg><figcaption>",_report_escape(caption),"</figcaption></figure>")
end

function _report_scientific_plots(io,records)
    for job in sort!(unique!([string(get(p,"source_job","")) for p in records["points"]]))
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
        _report_scatter(io,rows,"$coordinate (dimensionless)","E (occupancy)","Job $job: saved equilibrium samples; unconnected markers do not assign root identity.")
    end
    for section in records["sections"]
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
        _report_scatter(io,rows,"$(pair[1]) (dimensionless)","$(pair[2]) (dimensionless)","Job $(get(section,"source_job","")): local two-parameter section; circles are probes, crosses are fold/Hopf candidates. No filled region or continuous-curve claim.")
    end
end
