"""Replayable adaptive, finite-window drive study for the exemplar models."""
module AdaptiveRescueStudy

using FailureOfInhibition2025
import CSV
import SHA
import TOML
include("run_basin_rescue_study.jl")
using .BasinRescueStudy: model_for, search_model
using .BasinRescueStudy.MinimalExperiment: archive_provenance,
    artifact_checksums, context_record, file_hash, write_rows, write_toml

const ROOT = normpath(joinpath(@__DIR__, ".."))
const PARAMETERS = ("e_to_i", "theta_off", "tau_ratio")
const ROLE_NAMES = ("quiescent", "seizure", "active_mid", "herald")
const TRIAL_COLUMNS = [:case, :cell, :baseline_E, :source, :source_equilibrium,
    :source_E, :source_I, :duration, :protocol, :E_reduction, :I_increment,
    :total_E, :total_I, :integrated_E, :integrated_I, :status,
    :destination_equilibrium, :destination_role, :transition, :rescue,
    :final_followup, :attempts]
const BOUNDARY_COLUMNS = [:case, :cell, :baseline_E, :source, :duration,
    :axis, :fixed_amplitude, :lower_amplitude, :upper_amplitude,
    :lower_status, :upper_status, :lower_destination, :upper_destination,
    :censored]
const BRANCH_COLUMNS = [:case, :cell, :baseline_E, :role, :status,
    :equilibrium, :E, :I, :completeness]

function finite(value, name; lower=-Inf, strict=false)
    value isa Real && !(value isa Bool) && isfinite(value) &&
        (strict ? value > lower : value >= lower) ||
        throw(ArgumentError("$name must be finite and $(strict ? "greater than" : "at least") $lower"))
    return Float64(value)
end

function integer(value, name; minimum=1)
    value isa Integer && !(value isa Bool) && value >= minimum ||
        throw(ArgumentError("$name must be an integer of at least $minimum"))
    return Int(value)
end

function load_config(path)
    raw = TOML.parsefile(path)
    get(raw, "schema_version", nothing) == 1 ||
        throw(ArgumentError("unsupported adaptive rescue schema_version"))
    exemplar_path = normpath(joinpath(dirname(path), raw["exemplars"]))
    exemplars = TOML.parsefile(exemplar_path)
    exemplars["schema_version"] == 1 || throw(ArgumentError("unsupported exemplar schema"))
    cases = exemplars["cases"]
    names = [case["name"] for case in cases]
    length(names) == length(unique(names)) || throw(ArgumentError("duplicate exemplar name"))
    scan = raw["scan"]
    stops = [finite(x, "stage stop"; lower=0, strict=true) for x in scan["stage_stops"]]
    stops == [2.0, 4.0, 8.0] || throw(ArgumentError("stage stops must be [2,4,8]"))
    step = finite(scan["coarse_amplitude_step"], "coarse step"; lower=0, strict=true)
    baseline_step = finite(scan["baseline_probe_step"], "baseline probe step"; lower=0, strict=true)
    baseline_refine = finite(scan["baseline_refinement_step"], "baseline refinement step"; lower=0, strict=true)
    width = finite(scan["transition_width"], "transition width"; lower=0, strict=true)
    step == 1.0 && baseline_step == 0.5 && baseline_refine == 0.25 &&
        width == 0.0625 || throw(ArgumentError("unsupported adaptive sampling resolution"))
    durations = [finite(x, "duration"; lower=0, strict=true) for x in scan["durations"]]
    !isempty(durations) && issorted(durations) && length(unique(durations)) == length(durations) ||
        throw(ArgumentError("durations must be unique and increasing"))
    grid_points = integer(scan["equilibrium_grid_points"], "equilibrium grid"; minimum=2)
    tolerance = finite(scan["branch_match_atol"], "branch tolerance"; lower=0, strict=true)
    targets = Set(String.(raw["rescue_targets"]))
    !isempty(targets) && issubset(targets, Set(("quiescent", "active_mid", "herald"))) ||
        throw(ArgumentError("invalid rescue target set"))
    references = Dict{String,Dict{String,Vector{Float64}}}()
    for entry in raw["roles"]
        name = String(entry["case"])
        name in names && !haskey(references, name) ||
            throw(ArgumentError("unknown or duplicate role case: $name"))
        roles = Dict{String,Vector{Float64}}()
        for role in ROLE_NAMES
            haskey(entry, role) || continue
            state = entry[role]
            state isa Vector && length(state) == 2 ||
                throw(ArgumentError("$name.$role must be a two-state coordinate"))
            roles[role] = [finite(x, "$name.$role"; lower=0) for x in state]
            all(x -> x <= 1, roles[role]) ||
                throw(ArgumentError("$name.$role must be in [0,1]^2"))
        end
        haskey(roles, "quiescent") || throw(ArgumentError("$name lacks quiescent role"))
        references[name] = roles
    end
    Set(keys(references)) == Set(names) || throw(ArgumentError("roles must cover every exemplar"))
    sensitivity = raw["sensitivity"]
    for name in PARAMETERS
        lower = finite(sensitivity["$(name)_minimum"], "$name minimum";
            lower=0, strict=name == "tau_ratio")
        upper = finite(sensitivity["$(name)_maximum"], "$name maximum"; lower=lower, strict=true)
        finite(sensitivity["$(name)_offset"], "$name offset"; lower=0, strict=true)
        all(case -> lower <= case[name] <= upper, cases) ||
            throw(ArgumentError("exemplar $name is outside sensitivity bounds"))
    end
    diag = raw["diagnostics"]
    diagnostic_options = DiagnosticOptions(window_duration=diag["window_duration"],
        coordinate_atol=diag["coordinate_atol"], balance_atol=diag["balance_atol"],
        min_samples=diag["min_samples"])
    pulse_options = PulseExperimentOptions(amplitudes=[0.0], durations=durations,
        targets=[:I], followup_times=diag["followup_times"],
        diagnostic_options=diagnostic_options, refinement_levels=0, abstol=diag["abstol"],
        reltol=diag["reltol"], domain_atol=diag["domain_atol"],
        maxiters=diag["maxiters"])
    return (; raw, cases, references, targets, sensitivity, scan, stops,
        durations, grid_points, tolerance, pulse_options, exemplar_path)
end

function parameter_cells(case, sensitivity)
    nominal = Dict(name => Float64(case[name]) for name in PARAMETERS)
    cells = [(name="nominal", values=nominal, parameter="none", direction="none")]
    for parameter in PARAMETERS, (direction, sign) in (("minus", -1), ("plus", 1))
        lower = sensitivity["$(parameter)_minimum"]
        upper = sensitivity["$(parameter)_maximum"]
        value = clamp(nominal[parameter] + sign * sensitivity["$(parameter)_offset"], lower, upper)
        value == nominal[parameter] && continue
        values = copy(nominal)
        values[parameter] = value
        push!(cells, (name="$(parameter)_$(direction)", values,
            parameter, direction))
    end
    return cells
end

function match_roles(search, references, tolerance)
    matches = Dict{String,Union{Nothing,Int}}()
    reasons = Dict{String,String}()
    for (role, reference) in references
        if reference === nothing
            matches[role], reasons[role] = nothing, "tracking_lost"
            continue
        end
        candidates = [(index, hypot(root.state[1] - reference[1],
                root.state[2] - reference[2]))
            for (index, root) in enumerate(search.equilibria)
            if root.stability.classification == Attracting]
        sort!(candidates; by=last)
        if isempty(candidates) || first(candidates)[2] > tolerance
            matches[role], reasons[role] = nothing, "not_matched"
        elseif length(candidates) > 1 && candidates[2][2] - candidates[1][2] < 1e-6
            matches[role], reasons[role] = nothing, "ambiguous"
        else
            matches[role], reasons[role] = first(candidates)[1], "matched"
        end
    end
    used = [index for index in values(matches) if index !== nothing]
    for role in keys(matches)
        index = matches[role]
        if index !== nothing && count(==(index), used) > 1
            matches[role], reasons[role] = nothing, "ambiguous"
        end
    end
    return matches, reasons
end

function tracked_contexts(case, cell, reference_roles, baselines, grid_points, tolerance)
    references = Dict{String,Union{Nothing,Vector{Float64}}}(
        role => copy(state) for (role, state) in reference_roles)
    contexts = Dict{Float64,Any}()
    for baseline in baselines
        model = model_for(case, baseline; e_to_i=cell.values["e_to_i"],
            theta_off=cell.values["theta_off"], tau_ratio=cell.values["tau_ratio"])
        search = search_model(model, grid_points)
        matches, reasons = match_roles(search, references, tolerance)
        contexts[baseline] = (; model, search, matches, reasons)
        for role in keys(references)
            index = matches[role]
            references[role] = index === nothing ? nothing : copy(search.equilibria[index].state)
        end
    end
    return contexts
end

function source_roles(context)
    inverse = Dict(index => role for (role, index) in context.matches if index !== nothing)
    return [(index=index, role=get(inverse, index, "untracked_$index"))
        for (index, root) in enumerate(context.search.equilibria)
        if root.stability.classification == Attracting]
end

outcome_key(trial) = (trial.status, trial.destination)

"""Sample a 1-D line without assuming monotonic outcomes."""
function sample_line!(evaluate, start, stop, coarse_step, width)
    points = collect(start:coarse_step:stop)
    last(points) < stop && push!(points, stop)
    for (left, right) in zip(points, points[2:end])
        evaluate(left)
        evaluate((left + right) / 2) # guard for an interior island
    end
    evaluate(stop)
    function refine(left, right)
        right - left <= width && return
        middle = (left + right) / 2
        left_key, mid_key, right_key = outcome_key(evaluate(left)),
            outcome_key(evaluate(middle)), outcome_key(evaluate(right))
        if left_key != mid_key || mid_key != right_key || left_key[1] != :compatible
            refine(left, middle)
            refine(middle, right)
        end
    end
    for (left, right) in zip(points, points[2:end])
        refine(left, right)
    end
    return nothing
end

"""Sample corners and center; split only cells with changed or unresolved outcomes."""
function sample_cell!(evaluate, e0, e1, i0, i1, width)
    em, im = (e0 + e1) / 2, (i0 + i1) / 2
    keys = [outcome_key(evaluate(e, i)) for (e, i) in
        ((e0, i0), (e0, i1), (e1, i0), (e1, i1), (em, im))]
    all(key -> key == first(keys) && key[1] == :compatible, keys) && return
    e1 - e0 <= width && i1 - i0 <= width && return
    eparts = e1 - e0 > width ? ((e0, em), (em, e1)) : ((e0, e1),)
    iparts = i1 - i0 > width ? ((i0, im), (im, i1)) : ((i0, i1),)
    for (left_e, right_e) in eparts, (left_i, right_i) in iparts
        sample_cell!(evaluate, left_e, right_e, left_i, right_i, width)
    end
    return nothing
end

function amplitude_axis(stop, step)
    values = collect(0.0:step:stop)
    last(values) < stop && push!(values, stop)
    return values
end

function sampled_boundaries(cache, case_name, cell_name, baseline, role, duration)
    rows = NamedTuple[]
    by_e = Dict{Float64,Vector{Float64}}()
    by_i = Dict{Float64,Vector{Float64}}()
    for (e, i) in keys(cache)
        push!(get!(by_e, e, Float64[]), i)
        push!(get!(by_i, i, Float64[]), e)
    end
    for (axis, groups) in (("I", by_e), ("E_reduction", by_i))
        for (fixed, values) in groups
            sort!(values)
            for (lower, upper) in zip(values, values[2:end])
                left = axis == "I" ? cache[(fixed, lower)] : cache[(lower, fixed)]
                right = axis == "I" ? cache[(fixed, upper)] : cache[(upper, fixed)]
                outcome_key(left) == outcome_key(right) && continue
                push!(rows, (case=case_name, cell=cell_name, baseline_E=baseline,
                    source=role, duration, axis, fixed_amplitude=fixed,
                    lower_amplitude=lower, upper_amplitude=upper,
                    lower_status=string(left.status), upper_status=string(right.status),
                    lower_destination=left.destination === nothing ? missing : left.destination,
                    upper_destination=right.destination === nothing ? missing : right.destination,
                    censored=left.status != :compatible || right.status != :compatible))
            end
        end
    end
    return rows
end

function trial_row(case_name, cell_name, baseline, source, duration, trial, context, targets)
    destination = trial.destination
    inverse = Dict(index => role for (role, index) in context.matches if index !== nothing)
    destination_role = destination === nothing ? "unresolved" :
        get(inverse, destination, "untracked_$destination")
    protocol = trial.e_reduction == 0 ? "positive_I" : "tonic_E_withdrawal"
    rescue = source.role == "seizure" && trial.status == :compatible &&
        destination_role in targets && destination != source.index
    return (case=case_name, cell=cell_name, baseline_E=baseline,
        source=source.role, source_equilibrium=source.index,
        source_E=trial.initial_state[1], source_I=trial.initial_state[2],
        duration, protocol, E_reduction=trial.e_reduction,
        I_increment=trial.i_increment, total_E=trial.total_E,
        total_I=trial.total_I, integrated_E=trial.integrated_E,
        integrated_I=trial.integrated_I, status=string(trial.status),
        destination_equilibrium=destination === nothing ? missing : destination,
        destination_role, transition=trial.status == :compatible &&
            destination != source.index, rescue,
        final_followup=last(trial.attempts).followup_time,
        attempts=length(trial.attempts))
end

function resource_snapshot(path)
    available = try
        line = only(filter(x -> startswith(x, "MemAvailable:"), readlines("/proc/meminfo")))
        parse(Int, split(line)[2]) * 1024
    catch
        Sys.free_memory()
    end
    parent = isdir(path) ? path : dirname(path)
    disk_kib = parse(Int, split(split(read(`df -Pk $parent`, String), '\n')[2])[4])
    return (ram_available=available, disk_available=disk_kib * 1024)
end

function require_resources(path; min_ram=2_000_000_000, min_disk=5_000_000_000)
    snapshot = resource_snapshot(path)
    snapshot.ram_available >= min_ram ||
        throw(ArgumentError("available RAM below 2 GB; completed contexts can be resumed"))
    snapshot.disk_available >= min_disk ||
        throw(ArgumentError("available disk below 5 GB; completed contexts can be resumed"))
    return snapshot
end

function unit_status(rows, role, target_available)
    role == "seizure" || return "not_applicable"
    target_available || return "target_unavailable"
    isempty(rows) && return "not_sampled"
    any(row -> row.rescue, rows) && return "observed_rescue"
    any(row -> row.status != "compatible", rows) && return "unresolved"
    return "not_observed"
end

function unit_dir(output, case_name, cell_name, baseline, role)
    code = replace(string(baseline), "." => "p")
    return joinpath(output, "units", case_name, cell_name, "B_$code", role)
end

# Rename within the destination directory so an interrupted write never publishes
# a partial completion marker. Unmarked units are recomputed on resume.
function write_completion_marker(path, data; writer=write_toml)
    temporary, io = mktemp(dirname(path))
    close(io)
    try
        writer(temporary, data)
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary)
    end
    return nothing
end

function verify_resume_sources(metadata, exemplar_path; root=ROOT)
    for (relative, expected) in metadata["source_sha256"]
        source = relative == "experiments/exemplar_models.toml" ?
            exemplar_path : joinpath(root, relative)
        isfile(source) && file_hash(source) == expected ||
            throw(ArgumentError("source changed or missing since checkpoint: $relative"))
    end
    return nothing
end

function run_unit!(output, config, case_name, cell_name, baseline, context, source;
    smoke=false)
    directory = unit_dir(output, case_name, cell_name, baseline, source.role)
    marker = joinpath(directory, "done.toml")
    if isfile(marker)
        done = TOML.parsefile(marker)
        all(name -> isfile(joinpath(directory, name)) &&
            file_hash(joinpath(directory, name)) == done["sha256"][name],
            ("trials.csv", "boundaries.csv")) ||
            throw(ArgumentError("checkpoint checksum mismatch: $directory"))
        return done["presence"]
    end
    ispath(directory) && rm(directory; recursive=true, force=true)
    mkpath(directory)
    rows, boundaries = NamedTuple[], NamedTuple[]
    max_amplitude = smoke ? 1.0 : last(config.stops)
    durations = smoke ? config.durations[1:1] : config.durations
    width = smoke ? 0.25 : config.scan["transition_width"]
    state = context.search.equilibria[source.index].state
    for duration in durations
        cache = Dict{Tuple{Float64,Float64},Any}()
        function evaluate(e, i)
            return get!(cache, (e, i)) do
                trial = run_tonic_rescue_trial(context.model, context.search, state;
                    e_reduction=e, i_increment=i, duration,
                    options=config.pulse_options)
                push!(rows, trial_row(case_name, cell_name, baseline, source,
                    duration, trial, context, config.targets))
                trial
            end
        end
        # Visit all three stages even when earlier stages show no change.
        previous = 0.0
        for stop in (smoke ? [max_amplitude] : config.stops)
            sample_line!(i -> evaluate(0.0, i), previous, stop,
                config.scan["coarse_amplitude_step"], width)
            if baseline > 0
                e_axis = amplitude_axis(baseline, config.scan["coarse_amplitude_step"])
                i_axis = collect(previous:config.scan["coarse_amplitude_step"]:stop)
                last(i_axis) < stop && push!(i_axis, stop)
                for (e0, e1) in zip(e_axis, e_axis[2:end]),
                    (i0, i1) in zip(i_axis, i_axis[2:end])
                    sample_cell!(evaluate, e0, e1, i0, i1, width)
                end
            end
            previous = stop
            require_resources(output)
        end
        append!(boundaries, sampled_boundaries(cache, case_name, cell_name,
            baseline, source.role, duration))
    end
    trial_path, boundary_path = joinpath(directory, "trials.csv"),
        joinpath(directory, "boundaries.csv")
    write_rows(trial_path, rows, TRIAL_COLUMNS)
    write_rows(boundary_path, boundaries, BOUNDARY_COLUMNS)
    target_available = any(role -> haskey(context.matches, role) &&
        context.matches[role] !== nothing, config.targets)
    positive = unit_status(filter(row -> row.protocol == "positive_I", rows),
        source.role, target_available)
    tonic = baseline == 0 ? "not_applicable" :
        unit_status(filter(row -> row.protocol == "tonic_E_withdrawal", rows),
            source.role, target_available)
    presence = Dict("positive_I" => positive, "tonic_E_withdrawal" => tonic,
        "positive_I_transition" => any(row -> row.protocol == "positive_I" &&
            row.transition, rows),
        "tonic_E_withdrawal_transition" => any(row ->
            row.protocol == "tonic_E_withdrawal" && row.transition, rows),
        "positive_I_trials" => count(row -> row.protocol == "positive_I", rows),
        "tonic_E_withdrawal_trials" => count(row -> row.protocol == "tonic_E_withdrawal", rows),
        "boundary_count" => length(boundaries))
    write_completion_marker(marker, Dict("presence" => presence,
        "sha256" => Dict("trials.csv" => file_hash(trial_path),
            "boundaries.csv" => file_hash(boundary_path))))
    return presence
end

function baseline_signature(output, case_name, cell_name, baseline, context)
    signatures = String[]
    for source in source_roles(context)
        marker = joinpath(unit_dir(output, case_name, cell_name, baseline,
            source.role), "done.toml")
        isfile(marker) || continue
        presence = TOML.parsefile(marker)["presence"]
        push!(signatures, "$(source.role):$(presence["positive_I"]):$(presence["tonic_E_withdrawal"]):$(presence["positive_I_transition"]):$(presence["tonic_E_withdrawal_transition"])")
    end
    return join(sort!(signatures), "|")
end

function write_context_artifacts!(output, config, case, cell, baseline, context)
    context_dir = joinpath(output, "contexts", case["name"], cell.name)
    mkpath(context_dir)
    context_path = joinpath(context_dir, "B_$(replace(string(baseline), "." => "p")).toml")
    # Contexts are reconstructed on every run; replace any interrupted write.
    write_toml(context_path, context_record(context.search))
    branch_path = joinpath(context_dir,
        "B_$(replace(string(baseline), "." => "p"))_branches.csv")
    branches = NamedTuple[]
    for role in sort!(collect(keys(config.references[case["name"]])))
        index = context.matches[role]
        root = index === nothing ? nothing : context.search.equilibria[index]
        push!(branches, (case=case["name"], cell=cell.name,
            baseline_E=baseline, role, status=context.reasons[role],
            equilibrium=index === nothing ? missing : index,
            E=root === nothing ? missing : root.state[1],
            I=root === nothing ? missing : root.state[2],
            completeness=string(context.search.completeness)))
    end
    write_rows(branch_path, branches, BRANCH_COLUMNS)
end

function run_context!(output, config, case, cell, baseline, context; smoke=false)
    require_resources(output)
    write_context_artifacts!(output, config, case, cell, baseline, context)
    for source in source_roles(context)
        run_unit!(output, config, case["name"], cell.name, baseline, context,
            source; smoke)
    end
end

function _append_csv(destination, source)
    rows = CSV.File(source)
    isempty(rows) && return
    CSV.write(destination, rows; append=isfile(destination))
end

function aggregate!(output)
    trials_path = joinpath(output, "trials.csv")
    boundaries_path = joinpath(output, "boundaries.csv")
    branches_path = joinpath(output, "branches.csv")
    isfile(trials_path) && rm(trials_path)
    isfile(boundaries_path) && rm(boundaries_path)
    isfile(branches_path) && rm(branches_path)
    presence_rows = NamedTuple[]
    for (directory, _, filenames) in walkdir(joinpath(output, "units"))
        "done.toml" in filenames || continue
        relative = splitpath(relpath(directory, joinpath(output, "units")))
        case_name, cell_name, baseline_code, role = relative
        baseline = parse(Float64, replace(baseline_code[3:end], "p" => "."))
        done = TOML.parsefile(joinpath(directory, "done.toml"))["presence"]
        _append_csv(trials_path, joinpath(directory, "trials.csv"))
        _append_csv(boundaries_path, joinpath(directory, "boundaries.csv"))
        for protocol in ("positive_I", "tonic_E_withdrawal")
            push!(presence_rows, (case=case_name, cell=cell_name,
                baseline_E=baseline, source=role, protocol,
                status=done[protocol],
                observed_transition=done["$(protocol)_transition"],
                trials=done["$(protocol)_trials"]))
        end
    end
    !isfile(trials_path) && write_rows(trials_path, NamedTuple[], TRIAL_COLUMNS)
    !isfile(boundaries_path) && write_rows(boundaries_path, NamedTuple[], BOUNDARY_COLUMNS)
    sort!(presence_rows; by=r -> (r.case, r.cell, r.baseline_E, r.source, r.protocol))
    write_rows(joinpath(output, "presence.csv"), presence_rows,
        [:case, :cell, :baseline_E, :source, :protocol, :status,
            :observed_transition, :trials])
    for (directory, _, filenames) in walkdir(joinpath(output, "contexts"))
        for filename in sort!(filter(name -> endswith(name, "_branches.csv"), filenames))
            _append_csv(branches_path, joinpath(directory, filename))
        end
    end
    !isfile(branches_path) && write_rows(branches_path, NamedTuple[], BRANCH_COLUMNS)
    return length(presence_rows)
end

function run_experiment(config_path, output_dir; case_filter=nothing,
    cell_filter=nothing, smoke=false)
    config = load_config(config_path)
    output = abspath(output_dir)
    ispath(output) && !isdir(output) && throw(ArgumentError("output must be a directory"))
    mkpath(output)
    snapshot = require_resources(output)
    metadata_path = joinpath(output, "metadata.toml")
    if isfile(metadata_path)
        metadata = TOML.parsefile(metadata_path)
        metadata["config_sha256"] == file_hash(config_path) ||
            throw(ArgumentError("resume config differs from archived config"))
        metadata["smoke"] == smoke || throw(ArgumentError("resume smoke mode differs"))
        metadata["case_filter"] == (case_filter === nothing ? "all" : case_filter) &&
            metadata["cell_filter"] == (cell_filter === nothing ? "all" : cell_filter) ||
            throw(ArgumentError("resume case or cell filter differs"))
        verify_resume_sources(metadata, config.exemplar_path)
    else
        isempty(readdir(output)) || throw(ArgumentError("output is not an adaptive checkpoint"))
        metadata = archive_provenance(config_path, output)
        for relative in ("scripts/run_basin_rescue_study.jl",
            "scripts/run_adaptive_rescue_study.jl")
            destination = joinpath(output, "source", relative)
            cp(joinpath(ROOT, relative), destination; force=true)
            metadata["source_sha256"][relative] = file_hash(destination)
        end
        exemplar_destination = joinpath(output, "source", "experiments", "exemplar_models.toml")
        mkpath(dirname(exemplar_destination))
        cp(config.exemplar_path, exemplar_destination)
        metadata["source_sha256"]["experiments/exemplar_models.toml"] =
            file_hash(exemplar_destination)
        cp(config.exemplar_path, joinpath(output, basename(config.exemplar_path)))
        metadata["purpose"] = "adaptive finite-window drive observations"
        metadata["smoke"] = smoke
        metadata["case_filter"] = case_filter === nothing ? "all" : case_filter
        metadata["cell_filter"] = cell_filter === nothing ? "all" : cell_filter
        metadata["ram_available_preflight"] = snapshot.ram_available
        metadata["disk_available_preflight"] = snapshot.disk_available
        metadata["claim_limits"] = "sampled finite-window destinations only; no exact attractor count, asymptotic rescue proof, or exclusion of narrow unsampled islands"
        metadata["replay_from_artifact_directory"] =
            "julia --project=source source/scripts/run_adaptive_rescue_study.jl --config config.toml --output replay" *
            (smoke ? " --smoke" : "") *
            (case_filter === nothing ? "" : " --case $case_filter") *
            (cell_filter === nothing ? "" : " --cell $cell_filter")
        write_toml(metadata_path, metadata)
    end
    baselines = smoke ? collect(0.0:0.25:0.5) : collect(0.0:0.25:8.0)
    initial = smoke ? [0.0, 0.5] : collect(0.0:0.5:8.0)
    selected_cases = case_filter === nothing ? config.cases :
        filter(case -> case["name"] == case_filter, config.cases)
    isempty(selected_cases) && throw(ArgumentError("unknown case filter"))
    for case in selected_cases
        cells = smoke ? parameter_cells(case, config.sensitivity)[1:1] :
            parameter_cells(case, config.sensitivity)
        cell_filter !== nothing && filter!(cell -> cell.name == cell_filter, cells)
        isempty(cells) && throw(ArgumentError("unknown cell filter for $(case["name"])"))
        for cell in cells
            contexts = tracked_contexts(case, cell,
                config.references[case["name"]], baselines,
                smoke ? 5 : config.grid_points, config.tolerance)
            for baseline in initial
                run_context!(output, config, case, cell, baseline,
                    contexts[baseline]; smoke)
            end
            if !smoke
                extra = Float64[]
                for (lower, upper) in zip(initial, initial[2:end])
                    left = baseline_signature(output, case["name"], cell.name,
                        lower, contexts[lower])
                    right = baseline_signature(output, case["name"], cell.name,
                        upper, contexts[upper])
                    left != right && push!(extra, (lower + upper) / 2)
                end
                for baseline in extra
                    run_context!(output, config, case, cell, baseline,
                        contexts[baseline]; smoke)
                end
            end
        end
    end
    aggregate!(output)
    artifact_checksums(output)
    return output
end

function main(args=ARGS)
    arguments = Dict{String,String}()
    smoke = false
    index = 1
    while index <= length(args)
        if args[index] == "--smoke"
            smoke && throw(ArgumentError("duplicate --smoke"))
            smoke = true
            index += 1
            continue
        end
        key = args[index]
        key in ("--config", "--output", "--case", "--cell") &&
            index < length(args) && !haskey(arguments, key) ||
            throw(ArgumentError("unknown, duplicate, or incomplete option: $key"))
        arguments[key] = args[index + 1]
        index += 2
    end
    haskey(arguments, "--output") || throw(ArgumentError("--output is required"))
    path = get(arguments, "--config", joinpath(ROOT, "experiments", "adaptive_rescue.toml"))
    output = run_experiment(path, arguments["--output"];
        case_filter=get(arguments, "--case", nothing),
        cell_filter=get(arguments, "--cell", nothing), smoke)
    println("Wrote adaptive rescue observations to $output")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(AdaptiveRescueStudy.main())
end
