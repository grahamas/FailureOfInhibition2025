"""Build a portable, checked figure-data bundle from the reviewed selective runs."""
module PaperFigureData

using CSV, TOML, SHA
include(joinpath(@__DIR__, "..", "reproducibility", "selective_tonic_e_release_20261005", "replay.jl"))
const TER = SelectiveTonicERelease
const NS = TER.NS
const ROOT = normpath(joinpath(@__DIR__, ".."))
const BUNDLE = joinpath(ROOT, "reproducibility", "paper_figures_20261005")
const RESPONSE_REFERENCE = joinpath(BUNDLE, "response_reference.toml")
const SOURCE_ARTIFACTS = joinpath(BUNDLE, "source_artifacts.toml")

hashfile(path) = bytes2hex(SHA.sha256(read(path)))
portable_path(path) = replace(path, '\\' => '/')
function require_equal(actual, expected, label)
    actual == expected || error("figure-data source differs: $label")
end
function artifact_digest(directory, relative_files)
    length(relative_files) == length(unique(portable_path.(relative_files))) ||
        error("duplicate source artifact")
    record = IOBuffer()
    for relative in sort(relative_files; by=portable_path)
        (isabspath(relative) || ".." in splitpath(relative)) &&
            error("unsafe source artifact path: $relative")
        path = joinpath(directory, relative)
        isfile(path) || error("missing source artifact: $relative")
        print(record, portable_path(relative), '\0', hashfile(path), '\n')
    end
    return bytes2hex(SHA.sha256(take!(record)))
end
function check_numerical_environment(joint, tonic, environment)
    version = environment["julia_version"]
    for (label, actual) in (("current Julia", string(VERSION)),
            ("joint Julia", joint["julia_version"]),
            ("tonic Julia", tonic["metadata"]["julia_version"]))
        require_equal(actual, version, label)
    end
    hashes = environment["source_sha256"]
    required = Set(("Project.toml", "Manifest.toml",
        "plotting/Project.toml", "plotting/Manifest.toml"))
    require_equal(Set(keys(hashes)), required, "numerical environment file list")
    for relative in required
        digest = hashes[relative]
        require_equal(hashfile(joinpath(ROOT, relative)), digest,
            "numerical environment: $relative")
        if relative in ("Project.toml", "Manifest.toml")
            require_equal(joint["source_sha256"][relative], digest,
                "joint environment: $relative")
            require_equal(tonic["metadata"]["source_sha256"][relative], digest,
                "tonic environment: $relative")
        end
    end
    return true
end
function check_artifacts(directory, relative_files, reference, label)
    require_equal(length(relative_files), reference["files"], "$label file count")
    require_equal(artifact_digest(directory, relative_files), reference["sha256"],
        "$label artifacts")
end
function copy_checked(source, destination)
    mkpath(dirname(destination))
    cp(source, destination)
    require_equal(hashfile(destination), hashfile(source), source)
end
function compact_trace(path)
    rows = collect(CSV.File(path))
    kept = filter(row -> row.time <= 500.0, rows)
    last(rows).time > 500.0 && push!(kept, last(rows))
    CSV.write(path, kept)
end
function check_phase(phase, roles, expected, label)
    row = TER.phase_summary(phase, roles)
    require_equal(row["status"], expected["status"], "$label status")
    require_equal(row["destination"], expected["destination"], "$label destination")
    maximum(abs.(row["final"] .- expected["final"])) < 1e-7 ||
        error("figure-data endpoint differs: $label")
    return row
end
function save_checked(out, name, model, search, initial, roles, expected, config)
    phase = NS.IR.observe_phase(model, search, initial, config; retain=true, tight=true)
    check_phase(phase, roles, expected, name)
    NS.IR.save_phase(out, name, phase)
    compact_trace(joinpath(out, name * ".csv"))
    return phase
end
function root_rows(context)
    [Dict{String, Any}(
        "E" => root.state[1], "I" => root.state[2],
        "stability" => string(root.stability.classification),
        "role" => NS.NarrativeModels.role_name(context.roles, i))
        for (i, root) in enumerate(context.search.equilibria)]
end
function check_joint_traces(joint_dir, setting, context)
    previous = collect(context.search.equilibria[context.roles["rest"]].state)
    for cycle in 1:2, (phase, destination) in
            (("rest_to_active", "active"), ("active_to_rest", "rest"))
        stem = joinpath(joint_dir, setting, "cycle$(cycle)_$(phase)")
        pulse = collect(CSV.File(stem * "_pulse.csv"))
        followup = collect(CSV.File(stem * ".csv"))
        state(row) = [row.E, row.I]
        maximum(abs.(state(first(pulse)) .- previous)) < 1e-7 ||
            error("switching source discontinuity: $stem")
        maximum(abs.(state(last(pulse)) .- state(first(followup)))) < 1e-7 ||
            error("switching pulse/follow-up discontinuity: $stem")
        expected = context.search.equilibria[context.roles[destination]].state
        maximum(abs.(state(last(followup)) .- expected)) < 1e-7 ||
            error("switching endpoint differs: $stem")
        previous = state(last(followup))
    end
end

function check_response_archive(response_dir, anchor)
    archive = normpath(joinpath(response_dir, "..", "..", "..", ".."))
    relative_base = joinpath("anchors", "selective_withdrawal", "responses", "baseline_1")
    require_equal(normpath(response_dir), joinpath(archive, relative_base), "response archive layout")
    expected = TOML.parsefile(RESPONSE_REFERENCE)["files"]
    manifest = TOML.parsefile(joinpath(archive, "checksums.toml"))
    require_equal(manifest["algorithm"], "SHA-256", "response checksum algorithm")
    for (relative, digest) in expected
        path = joinpath(archive, relative)
        isfile(path) || error("missing response archive file: $relative")
        require_equal(get(manifest["files"], relative, nothing), digest,
            "response manifest: $relative")
        require_equal(hashfile(path), digest, "response archive: $relative")
    end
    parameters = TOML.parsefile(joinpath(archive, "anchors", "selective_withdrawal",
        "geometry", "parameters.toml"))
    for key in ("family", "e_to_e", "i_to_e", "e_to_i", "i_to_i", "tau_ratio", "theta_off")
        require_equal(parameters[key], anchor[key], "response parameter: $key")
    end
    input = TOML.parsefile(joinpath(response_dir, "input.toml"))
    require_equal(input["B_E"], anchor["B_E"], "response baseline E input")
    require_equal(input["B_I"], 0.0, "response baseline I input")
    roles = TOML.parsefile(joinpath(response_dir, "roles.toml"))
    require_equal(roles["herald"], 5, "response herald root")
    require_equal(roles["seizure"], 7, "response seizure root")
    return hashfile(RESPONSE_REFERENCE)
end

function build(joint_dir, tonic_dir, response_dir, destination)
    ispath(destination) && error("output path already exists: $destination")
    joint_ref = joinpath(ROOT, "reproducibility", "selective_anchor_joint_20261005", "reference_summary.toml")
    tonic_ref = joinpath(ROOT, "reproducibility", "selective_tonic_e_release_20261005", "reference_summary.toml")
    require_equal(hashfile(joinpath(joint_dir, "summary.toml")), hashfile(joint_ref), "joint summary")
    require_equal(hashfile(joinpath(tonic_dir, "reference_summary.toml")), hashfile(tonic_ref), "tonic summary")
    joint = TOML.parsefile(joint_ref)
    tonic = TOML.parsefile(tonic_ref)
    response_ref_hash = check_response_archive(response_dir, joint["baseline_parameters"])
    require_equal(joint["baseline_repeat_switching"], true, "baseline switching")
    require_equal(only(filter(row -> row["theta_off"] == 8.75,
        joint["threshold_interventions"]))["repeat_switching"], true, "threshold switching")
    joint_files = String[]
    for setting in ("baseline_switching", "theta_off_8.75_switching")
        for cycle in 1:2, phase in ("rest_to_active", "active_to_rest"), suffix in ("_pulse", "")
            push!(joint_files, joinpath(setting, "cycle$(cycle)_$(phase)$(suffix).csv"))
        end
    end
    append!(joint_files, ["induce_rest_to_herald_pulse.csv", "induce_rest_to_herald.csv",
        "induce_active_to_seizure_pulse.csv", "induce_active_to_seizure.csv",
        "theta_off_8.75_from_seizure.csv"])
    point_files = Dict(theta => filter(endswith(".toml"),
        readdir(joinpath(tonic_dir, "points", TER.theta_key(theta)); join=true))
        for theta in TER.THETAS)
    tonic_files = [relpath(file, tonic_dir) for theta in TER.THETAS
        for file in point_files[theta]]
    push!(tonic_files, joinpath("theta", "theta_8p0.toml"))
    source_artifacts = TOML.parsefile(SOURCE_ARTIFACTS)
    check_numerical_environment(joint, tonic, source_artifacts["environment"])
    check_artifacts(joint_dir, joint_files, source_artifacts["joint"], "joint")
    check_artifacts(tonic_dir, tonic_files, source_artifacts["tonic"], "tonic")

    mkpath(destination)
    traces = joinpath(destination, "traces")
    mkpath(traces)
    for relative in joint_files
        destination_trace = joinpath(traces, relative)
        copy_checked(joinpath(joint_dir, relative), destination_trace)
        compact_trace(destination_trace)
    end

    config = NS.load_config(joinpath(ROOT, "experiments", "narrative_study.toml"))
    baseline, off = TER.build_contexts(8.0, nothing)
    on = TER.context(8.0, 1.2328125, 41, baseline)
    length(baseline.search.equilibria) == 7 || error("baseline root count changed")
    length(on.search.equilibria) == 5 || error("on-input root count changed")
    point = TOML.parsefile(joinpath(tonic_dir, "points", "theta_8p0", "B_1p2328125.toml"))
    point["B_E_on"] == 1.2328125 || error("wrong tonic input point")
    selected = Dict{String, Any}()
    for source in ("rest", "active")
        source_row = point["sources"][source]
        initial = baseline.search.equilibria[baseline.roles[source]].state
        induction = save_checked(traces, "tonic_$(source)_induction",
            on.model, on.search, initial, on.roles, source_row["induction"], config)
        selected[source] = Dict("initial" => collect(initial),
            "switch_state" => collect(induction.final), "induction" => source_row["induction"],
            "control" => source_row["control"], "release" => source_row["release"])
        if source == "rest"
            save_checked(traces, "tonic_herald_held", on.model, on.search,
                induction.final, on.roles, source_row["control"], config)
            for b in TER.RETURNS
                key = TER.return_key(b)
                save_checked(traces, "tonic_herald_release_$(key)",
                    off[b].model, off[b].search, induction.final, off[b].roles,
                    source_row["release"][key], config)
            end
        end
    end
    seizure_record = TOML.parsefile(joinpath(tonic_dir, "theta", "theta_8p0.toml"))["seizure"]["1p2328125"]
    seizure_initial = on.search.equilibria[on.roles["seizure"]].state
    maximum(abs.(seizure_initial .- seizure_record["initial"])) < 1e-7 ||
        error("seizure source differs")
    save_checked(traces, "tonic_seizure_held", on.model, on.search,
        seizure_initial, on.roles, seizure_record["control"], config)
    for b in TER.RETURNS
        key = TER.return_key(b)
        save_checked(traces, "tonic_seizure_release_$(key)",
            off[b].model, off[b].search, seizure_initial, off[b].roles,
            seizure_record["release"][key], config)
    end
    selected["seizure"] = seizure_record

    changed = TER.context(8.75, 0.35, 41, baseline)
    length(changed.search.equilibria) == 5 || error("changed root count differs")
    check_joint_traces(joint_dir, "baseline_switching", baseline)
    check_joint_traces(joint_dir, "theta_off_8.75_switching", changed)
    rows = NamedTuple[]
    for theta in TER.THETAS
        length(point_files[theta]) == 147 || error("tonic point count differs at $theta")
        for file in point_files[theta]
            point_row = TOML.parsefile(file)
            point_row["theta_off"] == theta || error("tonic threshold mismatch")
            for source in TER.SOURCES
                result = point_row["sources"][source]
                induction = get(result, "induction", TER.unavailable("unrun"))
                control = get(result, "control", TER.unavailable("unrun"))
                releases = result["release"]
                push!(rows, (
                    theta_off=theta, B_E_on=point_row["B_E_on"], source=source,
                    induction_status=induction["status"], induction_destination=induction["destination"],
                    control_status=control["status"], control_destination=control["destination"],
                    release_0p35_status=get(releases, "B_0p35", TER.unavailable("unrun"))["status"],
                    release_0p35_destination=get(releases, "B_0p35", TER.unavailable("unrun"))["destination"],
                    release_0p17_status=get(releases, "B_0p17", TER.unavailable("unrun"))["status"],
                    release_0p17_destination=get(releases, "B_0p17", TER.unavailable("unrun"))["destination"],
                    release_0p0_status=get(releases, "B_0p0", TER.unavailable("unrun"))["status"],
                    release_0p0_destination=get(releases, "B_0p0", TER.unavailable("unrun"))["destination"]))
            end
        end
    end
    sort!(rows; by=row -> (row.theta_off, row.B_E_on, row.source))
    CSV.write(joinpath(destination, "tonic_samples.csv"), rows)
    length(rows) == 7 * 147 * 2 || error("tonic sample table incomplete")

    for (role, index) in (("herald", 5), ("seizure", 7))
        source = joinpath(response_dir, "root_$index", "pulses", "pulses.csv")
        pulses = filter(row -> row.axis == "E_withdrawal",
            collect(CSV.File(source)))
        isempty(pulses) && error("no established-state withdrawal pulses for $role")
        CSV.write(joinpath(destination, "$(role)_withdrawal_pulses.csv"), pulses)
        confirmations = joinpath(response_dir, "root_$index", "sustained", "confirmations.csv")
        copy_checked(confirmations, joinpath(destination, "$(role)_sustained.csv"))
    end
    check_artifacts(joint_dir, joint_files, source_artifacts["joint"], "joint")
    check_artifacts(tonic_dir, tonic_files, source_artifacts["tonic"], "tonic")
    record = Dict{String, Any}(
        "anchor" => joint["baseline_parameters"],
        "baseline_roots" => root_rows(baseline), "threshold_8p75_roots" => root_rows(changed),
        "tonic_on_roots" => root_rows(on), "selected_tonic" => selected,
        "samples_per_threshold" => 147, "sample_rows" => length(rows),
        "julia_version" => string(VERSION),
        "source_sha256" => Dict(path => hashfile(joinpath(ROOT, path)) for path in
            ("Project.toml", "Manifest.toml", "plotting/Project.toml",
             "plotting/Manifest.toml", "scripts/build_paper_figure_data.jl",
             "scripts/run_narrative_study.jl",
             "scripts/narrative_models.jl", "scripts/render_paper_figures.jl",
             "experiments/narrative_study.toml",
             "reproducibility/selective_tonic_e_release_20261005/replay.jl",
             "src/FailureOfInhibition2025.jl", "src/responses.jl", "src/drives.jl",
             "src/point_model.jl", "src/stability.jl", "src/equilibria.jl",
             "src/configurations.jl", "src/simulation.jl")),
        "reference_sha256" => Dict("joint" => hashfile(joint_ref), "tonic" => hashfile(tonic_ref),
            "response" => response_ref_hash, "source_artifacts" => hashfile(SOURCE_ARTIFACTS)),
        "limits" => "Finite-window trajectories and sampled input/threshold points; no bifurcation or completeness claim")
    NS.write_record(joinpath(destination, "data.toml"), record)
    files = [portable_path(relpath(joinpath(dir, file), destination))
        for (dir, _, filenames) in walkdir(destination)
        for file in filenames if file != "checksums.toml"]
    NS.write_record(joinpath(destination, "checksums.toml"),
        Dict("files" => Dict(file => hashfile(joinpath(destination, file)) for file in files)))
    return length(rows)
end

function main(args)
    length(args) == 4 || error("usage: julia --project=. scripts/build_paper_figure_data.jl JOINT_RUN TONIC_RUN RESPONSE_BASELINE NEW_BUNDLE")
    build(abspath.(args)...)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    PaperFigureData.main(ARGS)
end
