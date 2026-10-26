"""Independent on/off equilibrium discovery and permanent E-input release."""
module InputReleaseStudy

using FailureOfInhibition2025
import CSV
import TOML
include("input_release_models.jl")
using .InputReleaseModels: release_models
include("input_release_core.jl")
const Evidence = InputReleaseCore.Evidence
const ROOT = normpath(joinpath(@__DIR__, ".."))

function load_config(path)
    raw = TOML.parsefile(path)
    Evidence.require_keys(raw, ("schema_version", "exemplars", "roles", "on_input",
        "e_to_e_min", "e_to_e_max", "e_to_e_step", "grid_points", "confirmation_grids",
        "branch_match_atol", "horizons", "diagnostics", "solver"), "release config")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported release schema"))
    number(x, name; positive=false) = Evidence.finite_number(x, name;
        positive, nonnegative=true)
    lower = number(raw["e_to_e_min"], "e_to_e_min")
    upper = number(raw["e_to_e_max"], "e_to_e_max")
    lower < upper || throw(ArgumentError("e_to_e bounds must increase"))
    step = number(raw["e_to_e_step"], "e_to_e_step"; positive=true)
    axis = collect(lower:step:upper)
    last(axis) == upper || throw(ArgumentError("e_to_e step must divide bounds"))
    on_input = number(raw["on_input"], "on_input"; positive=true)
    grid = Evidence.positive_integer(raw["grid_points"], "grid_points")
    grids = [Evidence.positive_integer(x, "confirmation grid") for x in raw["confirmation_grids"]]
    grid >= 2 && !isempty(grids) && all(>=(grid), grids) ||
        throw(ArgumentError("invalid discovery/confirmation grids"))
    tolerance = number(raw["branch_match_atol"], "branch_match_atol"; positive=true)
    diag = raw["diagnostics"]
    diagnostics = DiagnosticOptions(window_duration=number(diag["window_duration"], "window"; positive=true),
        coordinate_atol=number(diag["coordinate_atol"], "coordinate tolerance"; positive=true),
        balance_atol=number(diag["balance_atol"], "balance tolerance"; positive=true),
        min_samples=Evidence.positive_integer(diag["min_samples"], "min_samples"))
    horizons = [number(x, "horizon"; positive=true) for x in raw["horizons"]]
    !isempty(horizons) && issorted(horizons) && length(unique(horizons)) == length(horizons) &&
        first(horizons) >= 2diagnostics.window_duration || throw(ArgumentError("invalid horizons"))
    solver = raw["solver"]
    abstol = number(solver["abstol"], "abstol"; positive=true)
    reltol = number(solver["reltol"], "reltol"; positive=true)
    domain_atol = number(solver["domain_atol"], "domain_atol")
    maxiters = Evidence.positive_integer(solver["maxiters"], "maxiters")
    exemplar_path = normpath(joinpath(dirname(path), raw["exemplars"]))
    role_path = normpath(joinpath(dirname(path), raw["roles"]))
    cases = TOML.parsefile(exemplar_path)["cases"]
    roles = Dict(row["case"] => get(row, "seizure", nothing)
        for row in TOML.parsefile(role_path)["roles"])
    # Validate all model cells' constant parameters before creating artifacts.
    for case in cases
        release_models(case; e_to_e=lower, on_input)
    end
    return (; raw, axis=reverse(axis), on_input, grid, grids, tolerance,
        diagnostics, horizons, abstol, reltol, domain_atol, maxiters,
        exemplar_path, role_path, cases, roles)
end

using .InputReleaseCore: search_context, sink_indices, match_source, observe_phase,
    phase_record, qualifies_trial, full_trial, save_phase, save_trial


function archive_inputs(config_path, config, output; smoke)
    metadata = Evidence.archive_provenance(config_path, output)
    paths = ["scripts/input_release_models.jl", "scripts/run_input_release_study.jl",
        "scripts/input_release_core.jl"]
    for relative in paths
        target = joinpath(output, "source", relative)
        cp(joinpath(ROOT, relative), target)
        metadata["source_sha256"][relative] = Evidence.file_hash(target)
    end
    for (name, source) in (("exemplars", config.exemplar_path), ("roles", config.role_path))
        relative = config.raw[name]
        # Keep replay dependencies beside config.toml; reject paths escaping the archive.
        isabspath(relative) || ".." in splitpath(relative) ?
            throw(ArgumentError("archive config paths must be relative without parent traversal")) : nothing
        target = joinpath(output, relative)
        mkpath(dirname(target))
        cp(source, target)
        metadata[name * "_sha256"] = Evidence.file_hash(target)
    end
    metadata["purpose"] = "input-dependent numerical seizure role and permanent zero-input release"
    metadata["equilibrium_seed_policy"] = "independent paired searches: configured grid plus default seeds"
    metadata["smoke"] = smoke
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_input_release_study.jl --config config.toml --output replay" * (smoke ? " --smoke" : "")
    metadata["artifact_schema"] = Dict("scan" => "all paired parameter cells and source availability",
        "contexts" => "all equilibrium attempts and local stability at both inputs",
        "selected" => "confirmed induction, actual switch state, permanent release and kept-on control")
    return metadata
end

function save_continuation(output, case, e_to_e, config, source_state)
    base = release_models(case; e_to_e, on_input=config.on_input).off
    factory = b -> PointModelParameters(excitatory=base.excitatory, inhibitory=base.inhibitory,
        coupling=base.coupling, drive=PiecewiseConstantDrive(baseline=(Float64(b), 0.0),
            pulses=(), interpretation=AfferentExcitation))
    result = continue_equilibria(factory, source_state, config.on_input;
        parameter_bounds=(0.0, config.on_input), options=ContinuationOptions(max_steps=600))
    Evidence.write_toml(joinpath(output, "continuation.toml"),
        (; result.initial_solve, result.initial_parameter, result.parameter_bounds,
            result.options, result.negative, result.positive, result.completeness))
    rows = [(direction=branch.direction, point=i, B_E=point.parameter,
        E=point.state[1], I=point.state[2], stability=string(point.stability.classification),
        spectral_abscissa=point.stability.spectral_abscissa)
        for branch in (result.negative, result.positive) for (i, point) in enumerate(branch.points)]
    Evidence.write_rows(joinpath(output, "continuation.csv"), rows,
        [:direction, :point, :B_E, :E, :I, :stability, :spectral_abscissa])
    return result
end

function confirm_candidate(output, case, e_to_e, reference, config)
    models = release_models(case; e_to_e, on_input=config.on_input)
    successful = true
    last_trial = nothing
    last_searches = nothing
    for grid in config.grids
        searches = (off=search_context(models.off, grid; tight=true),
            on=search_context(models.on, grid; tight=true))
        source = match_source(searches.on, reference, config.tolerance)
        sinks = sink_indices(searches.off)
        source === nothing || isempty(sinks) ? (successful=false; break) : nothing
        start = first(sort(sinks; by=i -> searches.off.equilibria[i].state[1]))
        trial = full_trial(models, searches, source, start, config; retain=true, tight=true)
        directory = joinpath(output, "grid_$grid")
        save_trial(directory, trial)
        for phase in (:on, :off)
            Evidence.write_toml(joinpath(directory, "$(phase)_context.toml"),
                Evidence.context_record(getproperty(searches, phase)))
        end
        successful &= trial.success
        last_trial, last_searches = trial, searches
    end
    return (; successful, trial=last_trial, searches=last_searches, models)
end

function run_study(config_path, output_dir; smoke=false)
    config = load_config(config_path)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    mkpath(output)
    metadata = archive_inputs(config_path, config, output; smoke)
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    rows = NamedTuple[]
    candidates = NamedTuple[]
    cases = smoke ? config.cases[1:1] : config.cases
    axis = smoke ? [6.0, 4.0] : config.axis
    for case in cases
        reference = get(config.roles, case["name"], nothing)
        for e_to_e in axis
            id = case["name"] * "_e_" * replace(string(e_to_e), "." => "p")
            directory = joinpath(output, "cells", id)
            mkpath(directory)
            models = release_models(case; e_to_e, on_input=config.on_input)
            searches = (off=search_context(models.off, config.grid),
                on=search_context(models.on, config.grid))
            for phase in (:off, :on)
                Evidence.write_toml(joinpath(directory, "$(phase)_context.toml"),
                    Evidence.context_record(getproperty(searches, phase)))
            end
            source = match_source(searches.on, reference, config.tolerance)
            reference = source === nothing ? nothing : copy(searches.on.equilibria[source].state)
            off_sinks = sink_indices(searches.off)
            start = isempty(off_sinks) ? nothing : first(sort(off_sinks;
                by=i -> searches.off.equilibria[i].state[1]))
            screen_status, screen_destination = "source_unavailable", 0
            induction_status, recovery_status, control_status = "not_run", "not_run", "not_run"
            success = false
            if source !== nothing && start !== nothing
                screen = observe_phase(models.off, searches.off, reference, config)
                save_phase(directory, "direct_release", screen)
                screen_status = screen.status
                screen_destination = something(screen.destination, 0)
                if screen.status == "compatible" && screen.destination == start
                    trial = full_trial(models, searches, source, start, config)
                    save_trial(joinpath(directory, "full_trial"), trial)
                    induction_status = trial.induction.status
                    recovery_status = trial.recovery.status
                    control_status = trial.control.status
                    success = trial.success
                    success && push!(candidates, (; case, e_to_e, reference=copy(reference)))
                end
            elseif source !== nothing
                screen_status = "target_unavailable"
            end
            push!(rows, (case=case["name"], e_to_e, off_roots=length(searches.off.equilibria),
                off_sinks=length(off_sinks), on_roots=length(searches.on.equilibria),
                on_sinks=length(sink_indices(searches.on)), source=something(source, 0),
                source_E=reference === nothing ? missing : reference[1],
                source_I=reference === nothing ? missing : reference[2],
                start=something(start, 0), screen_status, screen_destination,
                induction_status, recovery_status, control_status, success))
            CSV.write(joinpath(output, "scan.csv"), rows)
            println(id, " off/on sinks=", length(off_sinks), "/", length(sink_indices(searches.on)),
                " recovery=", success)
        end
    end
    # Prefer the largest positive recurrent coupling; ties follow catalogue order.
    sort!(candidates; by=x -> -x.e_to_e, alg=Base.Sort.MergeSort)
    selected = nothing
    for candidate in candidates
        directory = joinpath(output, "confirmations", candidate.case["name"] * "_" * string(candidate.e_to_e))
        confirmation = confirm_candidate(directory, candidate.case, candidate.e_to_e, candidate.reference, config)
        confirmation.successful || continue
        trial = confirmation.trial
        selected = Dict("case" => candidate.case["name"], "e_to_e" => candidate.e_to_e,
            "on_input" => config.on_input, "initial_state" => trial.initial,
            "driven_state" => confirmation.searches.on.equilibria[trial.source].state,
            "switch_state" => trial.induction.final, "final_state" => trial.recovery.final,
            "on_duration" => trial.induction.horizon, "off_duration" => trial.recovery.horizon,
            "confirmation_directory" => relpath(directory, output),
            "final_grid" => last(config.grids), "success" => true,
            "completeness" => "CompletenessNotCertified")
        Evidence.write_toml(joinpath(output, "selected.toml"), selected)
        certificate_case = copy(candidate.case)
        certificate_case["e_to_e"] = candidate.e_to_e
        Evidence.write_toml(joinpath(output, "zero_input_certificate_config.toml"),
            Dict("schema_version" => 1, "cases" => [certificate_case]))
        try
            save_continuation(output, candidate.case, candidate.e_to_e, config, selected["driven_state"])
        catch error
            error isa InterruptException && rethrow()
            Evidence.write_toml(joinpath(output, "continuation_error.toml"), Evidence.error_record(error))
        end
        break
    end
    metadata["completed"] = true
    metadata["cells"] = length(rows)
    metadata["observed_recovery_cells"] = count(row -> row.success, rows)
    metadata["confirmed_example"] = selected !== nothing
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; rows, selected)
end

function main(args=ARGS)
    options = Dict{String,String}()
    smoke = false
    i = 1
    while i <= length(args)
        key = args[i]
        if key == "--smoke"
            smoke = true; i += 1; continue
        end
        key in ("--config", "--output") && i < length(args) || throw(ArgumentError("invalid argument $key"))
        haskey(options, key) && throw(ArgumentError("duplicate $key"))
        options[key] = args[i+1]; i += 2
    end
    haskey(options, "--output") || throw(ArgumentError("--output is required"))
    config = get(options, "--config", joinpath(ROOT, "experiments/input_release.toml"))
    run_study(config, options["--output"]; smoke)
    return 0
end
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(InputReleaseStudy.main())
end
