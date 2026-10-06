"""Bounded tonic-E induction and release at the selective point-model anchor."""
module SelectiveTonicERelease

import TOML
include(joinpath(@__DIR__, "..", "..", "scripts", "run_narrative_study.jl"))
const NS = NarrativeStudy
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const THETAS = (8.0, 8.25, 8.5, 8.75, 9.0, 10.0, 12.0)
const RETURNS = (0.35, 0.17, 0.0)
const SOURCES = ("rest", "active")
const BASE_E = 0.35
const STEP = 0.25
const WIDTH = 0.01
const POINT_BUDGET = 4096
const SOURCE_PATHS = vcat(
    [joinpath("src", name) for name in sort(readdir(joinpath(ROOT, "src"))) if endswith(name, ".jl")],
    ["scripts/run_narrative_study.jl", "scripts/narrative_models.jl",
     "scripts/run_input_release_study.jl", "scripts/input_release_models.jl",
     "scripts/run_basin_rescue_study.jl", "scripts/run_minimal_experiment.jl",
     "experiments/narrative_study.toml", "Project.toml", "Manifest.toml"])

base_parameters(theta, b) = Dict{String, Any}(
    "family" => "figure4", "e_to_e" => 17.0, "i_to_e" => 13.0,
    "e_to_i" => 19.0, "i_to_i" => 6.0, "theta_off" => theta,
    "tau_ratio" => 0.2, "B_E" => b)
input_key(b) = replace(string(round(b; digits=9)), "." => "p")
return_key(b) = "B_" * input_key(b)
theta_key(theta) = "theta_" * input_key(theta)

"""Return every sampled success run, retaining the preceding sampled outcome."""
function success_runs(points, source, return_input)
    axis = sort(collect(keys(points)))
    isempty(axis) && return Dict{String, Any}[]
    key = return_key(return_input)
    success(b) = get(get(points[b]["sources"][source], "qualifies", Dict()), key, false)
    result = Dict{String, Any}[]
    i = 1
    while i <= length(axis)
        if !success(axis[i])
            i += 1
            continue
        end
        first_success = i
        while i < length(axis) && success(axis[i+1])
            i += 1
        end
        previous = first_success == 1 ? axis[1] : axis[first_success-1]
        push!(result, Dict{String, Any}(
            "lower_sample" => previous, "first_success" => axis[first_success],
            "last_success" => axis[i],
            "next_sample" => i == length(axis) ? axis[i] : axis[i+1],
            "at_domain_start" => first_success == 1,
            "at_domain_end" => i == length(axis),
            "lower_status" => points[previous]["sources"][source]["status"]))
        i += 1
    end
    return result
end

"""A success requires induction, its kept-on control, and off recovery."""
function qualifies(induction, control, release)
    return induction["status"] == "compatible" && induction["destination"] == "herald" &&
        control["status"] == "compatible" && control["destination"] == "herald" &&
        release["status"] == "compatible" && release["destination"] == "active"
end

function combine_roles(direct_roles, matched)
    direct = copy(direct_roles)
    for (role, index) in matched
        if haskey(direct, role) && direct[role] != index
            delete!(direct, role)
        elseif !any(other != role && candidate == index for (other, candidate) in direct)
            direct[role] = index
        end
    end
    collisions = [index for index in values(direct)
        if count(==(index), values(direct)) > 1]
    filter!(pair -> !(pair.second in collisions), direct)
    return direct
end

function roles_with_match(reference, discovered)
    matched = NS.NarrativeModels.match_roles(reference.search, reference.roles, discovered.search)
    roles = combine_roles(discovered.roles, matched)
    return merge(discovered, (; roles))
end

function context(theta, b, grid, reference)
    discovered = NS.context(base_parameters(theta, b), grid; tight=true)
    return reference === nothing ? discovered : roles_with_match(reference, discovered)
end

function phase_summary(phase, roles)
    row = NS.phase_row(phase, roles)
    return Dict{String, Any}("status" => row.status, "destination" => row.destination,
        "destination_index" => row.destination_index, "final" => collect(phase.final),
        "horizon" => row.horizon,
        "solver_status" => isempty(phase.attempts) ? "none" : last(phase.attempts).solver_status)
end

function unavailable(reason)
    return Dict{String, Any}("status" => reason, "destination" => "unresolved")
end

function build_contexts(theta, reference)
    baseline = context(theta, BASE_E, 41, reference)
    off = Dict{Float64, Any}(BASE_E => baseline)
    previous = baseline
    for b in (0.17, 0.0)
        previous = context(theta, b, 41, previous)
        off[b] = previous
    end
    return baseline, off
end

function source_trial(source, baseline, on, off, config)
    haskey(baseline.roles, source) || return Dict{String, Any}(
        "status" => "source_unavailable", "qualifies" => Dict{String, Bool}(),
        "release" => Dict{String, Any}())
    initial = copy(baseline.search.equilibria[baseline.roles[source]].state)
    induction = NS.IR.observe_phase(on.model, on.search, initial, config; tight=true)
    on_row = phase_summary(induction, on.roles)
    result = Dict{String, Any}("status" => on_row["status"],
        "initial" => initial, "switch_state" => collect(induction.final),
        "induction" => on_row, "qualifies" => Dict{String, Bool}(),
        "release" => Dict{String, Any}())
    on_row["status"] == "compatible" && on_row["destination"] == "herald" || return result
    control = NS.IR.observe_phase(on.model, on.search, induction.final, config; tight=true)
    control_row = phase_summary(control, on.roles)
    result["control"] = control_row
    for b in RETURNS
        key = return_key(b)
        released = NS.IR.observe_phase(off[b].model, off[b].search, induction.final, config; tight=true)
        row = phase_summary(released, off[b].roles)
        result["release"][key] = row
        result["qualifies"][key] = qualifies(on_row, control_row, row)
    end
    return result
end

function evaluate_point(theta, b, baseline, off, config; grid=21)
    on = context(theta, b, grid, baseline)
    sources = Dict(source => source_trial(source, baseline, on, off, config) for source in SOURCES)
    return Dict{String, Any}("theta_off" => theta, "B_E_on" => b,
        "on_grid" => grid, "on_roots" => length(on.search.equilibria),
        "on_sinks" => length(NS.IR.sink_indices(on.search)),
        "on_roles" => on.roles, "sources" => sources)
end

function signature(point)
    return [(source, get(point["sources"][source], "induction", unavailable("unrun"))["status"],
        get(point["sources"][source], "induction", unavailable("unrun"))["destination"],
        get(point["sources"][source], "control", unavailable("unrun"))["status"],
        get(point["sources"][source], "control", unavailable("unrun"))["destination"],
        [(key, get(point["sources"][source]["release"], key, unavailable("unrun"))["status"],
            get(point["sources"][source]["release"], key, unavailable("unrun"))["destination"])
            for key in return_key.(RETURNS)]) for source in SOURCES]
end

function uncertain(point)
    for source in SOURCES
        row = point["sources"][source]
        row["status"] == "source_unavailable" && continue
        on = row["induction"]
        on["status"] != "compatible" && return true
        if on["destination"] == "herald"
            get(row["control"], "status", "unresolved") != "compatible" && return true
            any(get(release, "status", "unresolved") != "compatible"
                for release in values(row["release"])) && return true
        end
    end
    return false
end

function source_hashes()
    return Dict(path => NS.Evidence.file_hash(joinpath(ROOT, path)) for path in SOURCE_PATHS)
end

function metadata()
    return Dict{String, Any}("julia_version" => string(VERSION),
        "purpose" => "selective anchor tonic E induction and cessation",
        "theta_off" => collect(THETAS), "return_B_E" => collect(RETURNS),
        "sources" => collect(SOURCES), "baseline_B_E" => BASE_E,
        "step" => STEP, "width" => WIDTH, "point_budget_per_theta" => POINT_BUDGET,
        "initial_upper_B_E" => 8.0, "extension_upper_B_E" => 16.0,
        "replay_sha256" => NS.Evidence.file_hash(@__FILE__),
        "source_sha256" => source_hashes())
end

function initialize(output; resume=false)
    path = joinpath(output, "metadata.toml")
    expected = metadata()
    if resume
        isfile(path) || error("cannot resume without metadata: $path")
        TOML.parsefile(path) == expected || error("source or protocol changed; use a new output directory")
    else
        ispath(output) && error("output path already exists: $output")
        mkpath(output)
        NS.write_record(path, expected)
    end
end

function point_path(output, theta, b)
    return joinpath(output, "points", theta_key(theta), "B_" * input_key(b) * ".toml")
end

function point_at!(points, theta, b, baseline, off, config, output)
    b = round(Float64(b); digits=9)
    if haskey(points, b)
        return points[b]
    end
    path = point_path(output, theta, b)
    row = if isfile(path)
        TOML.parsefile(path)
    else
        trial = evaluate_point(theta, b, baseline, off, config)
        NS.write_record(path, trial)
        trial
    end
    points[b] = row
    println("theta=", theta, " B_E_on=", b, " rest=",
        get(row["sources"]["rest"], "induction", unavailable("unrun"))["destination"],
        " active=", get(row["sources"]["active"], "induction", unavailable("unrun"))["destination"])
    flush(stdout)
    return row
end

function sample_range!(points, theta, lower, upper, baseline, off, config, output)
    axis = collect(lower:STEP:upper)
    last(axis) < upper && push!(axis, upper)
    at(b) = point_at!(points, theta, b, baseline, off, config, output)
    at(first(axis))
    exhausted = Ref(false)
    function refine(a, b)
        if length(points) >= POINT_BUDGET
            exhausted[] = true
            return
        end
        mid = round((a+b)/2; digits=9)
        middle = at(mid)
        b-a <= WIDTH && return
        left, right = at(a), at(b)
        if signature(left) != signature(right) || signature(left) != signature(middle) ||
            uncertain(left) || uncertain(middle) || uncertain(right)
            refine(a, mid)
            refine(mid, b)
        end
    end
    for (a, b) in zip(axis, axis[2:end])
        at(b)
        refine(a, b)
    end
    return exhausted[]
end

function needs_extension(points)
    for source in SOURCES, b in RETURNS
        any(get(get(row["sources"][source], "qualifies", Dict()), return_key(b), false)
            for row in values(points)) || return true
    end
    return false
end

function seizure_trial(theta, b, baseline, off, config)
    on = context(theta, b, 41, baseline)
    haskey(on.roles, "seizure") || return Dict{String, Any}("status" => "source_unavailable",
        "B_E_on" => b, "on_grid" => 41)
    initial = copy(on.search.equilibria[on.roles["seizure"]].state)
    control = NS.IR.observe_phase(on.model, on.search, initial, config; tight=true)
    releases = Dict{String, Any}()
    for off_b in RETURNS
        released = NS.IR.observe_phase(off[off_b].model, off[off_b].search, initial, config; tight=true)
        releases[return_key(off_b)] = phase_summary(released, off[off_b].roles)
    end
    return Dict{String, Any}("status" => "tested", "B_E_on" => b,
        "on_grid" => 41, "initial" => initial,
        "control" => phase_summary(control, on.roles), "release" => releases)
end

function summarize_theta(theta, points, baseline, off, config, output, exhausted)
    results = Dict{String, Any}[]
    seizure = Dict{String, Any}()
    for source in SOURCES, off_b in RETURNS
        runs = success_runs(points, source, off_b)
        for run in runs
            lower, upper = run["lower_sample"], run["first_success"]
            lower_confirmed = evaluate_point(theta, lower, baseline, off, config; grid=41)
            upper_confirmed = evaluate_point(theta, upper, baseline, off, config; grid=41)
            key = return_key(off_b)
            lower_ok = get(lower_confirmed["sources"][source]["qualifies"], key, false)
            upper_ok = get(upper_confirmed["sources"][source]["qualifies"], key, false)
            run["confirmed"] = upper_ok && (run["at_domain_start"] || !lower_ok)
            run["confirmation_grid"] = 41
            run["lower_confirmation"] = lower_confirmed["sources"][source]
            run["upper_confirmation"] = upper_confirmed["sources"][source]
            if run["confirmed"]
                bkey = input_key(upper)
                seizure[bkey] = get!(()->seizure_trial(theta, upper, baseline, off, config), seizure, bkey)
            end
        end
        push!(results, Dict{String, Any}("source" => source,
            "return_B_E" => off_b, "runs" => runs,
            "qualified_samples" => count(row ->
                get(row["sources"][source]["qualifies"], return_key(off_b), false), values(points))))
    end
    summary = Dict{String, Any}("theta_off" => theta, "sample_count" => length(points),
        "sampled_max_B_E" => maximum(keys(points)), "point_budget_exhausted" => exhausted,
        "baseline_roles" => baseline.roles,
        "off_roles" => Dict(return_key(b) => off[b].roles for b in RETURNS),
        "thresholds" => results, "seizure" => seizure)
    NS.write_record(joinpath(output, "theta", theta_key(theta) * ".toml"), summary)
    return summary
end

function reference_summary(summary)
    rows = Dict{String, Any}[]
    for theta in summary["thresholds"]
        thresholds = Dict{String, Any}[]
        for result in theta["thresholds"]
            brackets = [Dict{String, Any}(
                "lower_sample" => run["lower_sample"],
                "first_success" => run["first_success"],
                "minimum_delta_B_E" => run["first_success"] - BASE_E,
                "last_success" => run["last_success"],
                "confirmed_41_grid" => run["confirmed"],
                "at_domain_start" => run["at_domain_start"],
                "at_domain_end" => run["at_domain_end"])
                for run in result["runs"]]
            push!(thresholds, Dict{String, Any}(
                "source" => result["source"], "return_B_E" => result["return_B_E"],
                "qualified_samples" => result["qualified_samples"],
                "brackets" => brackets))
        end
        seizure = Dict{String, Any}()
        for (input, trial) in theta["seizure"]
            seizure[input] = Dict{String, Any}(
                "status" => trial["status"],
                "control" => get(get(trial, "control", Dict()), "destination", "not_tested"),
                "release" => Dict(key => phase["destination"]
                    for (key, phase) in get(trial, "release", Dict())))
        end
        push!(rows, Dict{String, Any}(
            "theta_off" => theta["theta_off"],
            "sample_count" => theta["sample_count"],
            "sampled_max_B_E" => theta["sampled_max_B_E"],
            "point_budget_exhausted" => theta["point_budget_exhausted"],
            "thresholds" => thresholds, "seizure" => seizure))
    end
    return Dict{String, Any}("metadata" => summary["metadata"],
        "thresholds" => rows, "limits" => summary["limits"])
end

function run(output; resume=false)
    initialize(output; resume)
    config = NS.load_config(joinpath(ROOT, "experiments", "narrative_study.toml"))
    reference = context(8.0, BASE_E, 41, nothing)
    summaries = Dict{String, Any}[]
    for theta in THETAS
        baseline, off = build_contexts(theta, reference)
        points = Dict{Float64, Any}()
        exhausted = sample_range!(points, theta, BASE_E, 8.0, baseline, off, config, output)
        if needs_extension(points)
            exhausted |= sample_range!(points, theta, 8.0, 16.0, baseline, off, config, output)
        end
        push!(summaries, summarize_theta(theta, points, baseline, off, config, output, exhausted))
        println("completed theta=", theta, " points=", length(points),
            " budget_exhausted=", exhausted); flush(stdout)
    end
    summary = Dict{String, Any}("metadata" => metadata(), "thresholds" => summaries,
        "limits" => "Sampled input brackets and finite-window outcomes; no exact threshold, attractor completeness, asymptotic rescue, or biological certification")
    NS.write_record(joinpath(output, "summary.toml"), summary)
    NS.write_record(joinpath(output, "reference_summary.toml"), reference_summary(summary))
    return summary
end

function main(args)
    length(args) in (1, 2) || error("usage: julia --project=. replay.jl OUTPUT [--resume]")
    resume = length(args) == 2
    resume && args[2] != "--resume" && error("unknown option: $(args[2])")
    result = run(abspath(args[1]); resume)
    println("Julia tonic-E replay complete: ", length(result["thresholds"]), " failure thresholds")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    SelectiveTonicERelease.main(ARGS)
end
