"""Replayable finite basin geometry and tonic-drive pulse study."""
module BasinRescueStudy

using FailureOfInhibition2025
import CSV
import TOML
include("run_minimal_experiment.jl")
using .MinimalExperiment: write_toml, write_rows, archive_provenance,
    file_hash, artifact_checksums, context_record

const ROOT = normpath(joinpath(@__DIR__, ".."))
const ROLES = (:low_activity, :high_e_low_i, :high_e_high_i)

function _finite(value, name; minimum=-Inf, strict=false)
    value isa Real && !(value isa Bool) && isfinite(value) &&
        (strict ? value > minimum : value >= minimum) ||
        throw(ArgumentError("$name must be finite and $(strict ? "greater than" : "at least") $minimum"))
    return Float64(value)
end

function _integer(value, name; minimum=0)
    value isa Integer && !(value isa Bool) && minimum <= value <= typemax(Int) ||
        throw(ArgumentError("$name must be an integer of at least $minimum"))
    return Int(value)
end

function _axis(first, last, step)
    values = collect(first:step:last)
    isempty(values) && push!(values, first)
    values[end] < last && push!(values, last)
    return values
end

function load_config(path)
    raw = TOML.parsefile(path)
    raw["schema_version"] == 1 || throw(ArgumentError("unsupported schema_version"))
    scan = raw["scan"]
    for name in ("baseline_start", "baseline_stop", "baseline_step",
        "e_reduction_step", "i_amplitude_start", "i_amplitude_stop",
        "i_amplitude_step", "branch_match_atol")
        _finite(scan[name], name; minimum=0, strict=endswith(name, "step") || name == "branch_match_atol")
    end
    scan["baseline_start"] <= scan["baseline_stop"] ||
        throw(ArgumentError("baseline range is reversed"))
    scan["i_amplitude_start"] <= scan["i_amplitude_stop"] ||
        throw(ArgumentError("I amplitude range is reversed"))
    _integer(scan["equilibrium_grid_points"], "equilibrium_grid_points"; minimum=2)
    _integer(scan["amplitude_refinement_levels"], "amplitude_refinement_levels")
    durations = scan["durations"]
    !isempty(durations) && all(value -> value isa Real && !(value isa Bool) &&
        isfinite(value) && value > 0, durations) && issorted(durations) &&
        length(unique(durations)) == length(durations) ||
        throw(ArgumentError("durations must be positive, unique, and increasing"))
    diagnostics = raw["diagnostics"]
    diagnostic_options = DiagnosticOptions(
        window_duration=diagnostics["window_duration"],
        coordinate_atol=diagnostics["coordinate_atol"],
        balance_atol=diagnostics["balance_atol"],
        min_samples=diagnostics["min_samples"])
    pulse_options = PulseExperimentOptions(amplitudes=[0.0], durations=durations,
        targets=[:I], followup_times=diagnostics["followup_times"],
        diagnostic_options=diagnostic_options, refinement_levels=0,
        abstol=diagnostics["abstol"], reltol=diagnostics["reltol"],
        domain_atol=diagnostics["domain_atol"], maxiters=diagnostics["maxiters"])
    geometry = raw["geometry"]
    basin_options = BasinMeasurementOptions(
        grid_points=geometry["grid_points"], ray_samples=geometry["ray_samples"],
        ray_refinements=geometry["ray_refinements"], angles=geometry["angles"],
        pulse_options=pulse_options)
    sensitivity = raw["sensitivity"]
    for name in ("e_to_i", "theta_off", "tau_ratio")
        lower = _finite(sensitivity["$(name)_minimum"], "$name minimum";
            minimum=0, strict=name == "tau_ratio")
        upper = _finite(sensitivity["$(name)_maximum"], "$name maximum";
            minimum=lower, strict=true)
        lower < upper || throw(ArgumentError("invalid $name sensitivity bounds"))
    end
    cases = raw["cases"]
    length(cases) == 2 && Set(case["name"] for case in cases) ==
        Set(("figure3", "figure4_rising")) ||
        throw(ArgumentError("configuration must provide the two selected cases"))
    for case in cases
        for name in ("e_to_e", "i_to_e", "e_to_i", "i_to_i", "theta_off",
            "tau_e", "tau_ratio", "slope_e", "slope_i", "theta_e", "theta_on")
            _finite(case[name], "$(case["name"]).$name"; minimum=0,
                strict=name in ("tau_e", "tau_ratio", "slope_e", "slope_i"))
        end
        case["theta_on"] < case["theta_off"] ||
            throw(ArgumentError("theta_on must precede theta_off"))
        for role in ROLES
            state = case[string(role)]
            state isa Vector && length(state) == 2 &&
                all(value -> value isa Real && !(value isa Bool) &&
                    isfinite(value) && 0 <= value <= 1, state) ||
                throw(ArgumentError("$(case["name"]).$role must be a physical E/I state"))
        end
    end
    return (; raw, scan, cases, pulse_options, basin_options, sensitivity)
end

function model_for(case, baseline; e_to_i=case["e_to_i"],
    theta_off=case["theta_off"], tau_ratio=case["tau_ratio"])
    drive = PiecewiseConstantDrive(baseline=(Float64(baseline), 0.0), pulses=(),
        interpretation=AfferentExcitation)
    return matched_point_models(
        excitatory=PopulationParameters(timescale=case["tau_e"],
            response=LogisticResponse(slope=case["slope_e"],
                threshold=case["theta_e"])),
        inhibitory_control=PopulationParameters(
            timescale=case["tau_e"] * tau_ratio,
            response=LogisticResponse(slope=case["slope_i"],
                threshold=case["theta_on"])),
        failure_threshold=theta_off,
        coupling=PointCoupling(e_to_e=case["e_to_e"], i_to_e=case["i_to_e"],
            e_to_i=e_to_i, i_to_i=case["i_to_i"]),
        drive=drive).failure_of_inhibition
end

function search_model(model, grid_points)
    upper = FailureOfInhibition2025._equilibrium_upper_bounds(model, Float64)
    seeds = vcat([[e, i] for e in range(0.0, upper[1]; length=grid_points)
        for i in range(0.0, upper[2]; length=grid_points)],
        default_equilibrium_seeds(model))
    return find_equilibria(model; seeds)
end

function match_roles(search, references, tolerance)
    choices = Dict{Symbol,Union{Nothing,Int}}()
    reasons = Dict{Symbol,Symbol}()
    for role in ROLES
        reference = references[role]
        if reference === nothing
            choices[role], reasons[role] = nothing, :tracking_lost
            continue
        end
        candidates = [(index, hypot(root.state[1] - reference[1],
                root.state[2] - reference[2]))
            for (index, root) in enumerate(search.equilibria)
            if root.stability.classification == Attracting]
        sort!(candidates; by=last)
        if isempty(candidates) || first(candidates)[2] > tolerance
            choices[role], reasons[role] = nothing, :not_matched
        elseif length(candidates) > 1 && candidates[2][2] - candidates[1][2] < 1e-6
            choices[role], reasons[role] = nothing, :ambiguous
        else
            choices[role], reasons[role] = first(candidates)[1], :matched
        end
    end
    selected = filter(value -> !isnothing(value), collect(values(choices)))
    for role in ROLES
        index = choices[role]
        if index !== nothing && count(==(index), selected) > 1
            choices[role], reasons[role] = nothing, :ambiguous
        end
    end
    return choices, reasons
end

function _outcome_key(trial)
    return (trial.status, trial.destination)
end

function _trial_row(case_name, baseline, role, trial, target)
    final = last(trial.attempts)
    return (case=case_name, baseline_E=baseline, source=string(role),
        source_E=trial.initial_state[1], source_I=trial.initial_state[2],
        target=target === nothing ? "not_available" : string(target),
        E_reduction=trial.e_reduction, I_increment=trial.i_increment,
        duration=trial.duration, total_E=trial.total_E, total_I=trial.total_I,
        integrated_E=trial.integrated_E, integrated_I=trial.integrated_I,
        status=string(trial.status), destination=trial.destination === nothing ? missing : trial.destination,
        rescue=trial.status == :compatible && trial.destination == target,
        final_followup=final.followup_time, attempts=length(trial.attempts))
end

function _boundary_row(case_name, baseline, role, duration, axis,
    fixed_value, lower, upper, left, right)
    return (case=case_name, baseline_E=baseline, source=string(role),
        duration=duration, axis=string(axis), fixed_amplitude=fixed_value,
        lower_amplitude=lower, upper_amplitude=upper,
        lower_status=string(left.status), upper_status=string(right.status),
        lower_destination=left.destination === nothing ? missing : left.destination,
        upper_destination=right.destination === nothing ? missing : right.destination,
        censored=left.status != :compatible || right.status != :compatible)
end

function _append(path, rows)
    isempty(rows) && return
    CSV.write(path, rows; append=isfile(path))
end

function _metadata(config_path, output, mode, case_name, smoke, parameter_values)
    metadata = archive_provenance(config_path, output)
    relative = "scripts/run_basin_rescue_study.jl"
    destination = joinpath(output, "source", relative)
    cp(joinpath(ROOT, relative), destination)
    metadata["source_sha256"][relative] = file_hash(destination)
    metadata["purpose"] = "finite basin and nonnegative-total-drive protocol observations"
    metadata["mode"] = mode
    metadata["case"] = case_name
    metadata["smoke"] = smoke
    metadata["parameter_values"] = parameter_values
    flags = join((" --$(replace(name, "_" => "-")) $(value)"
        for (name, value) in sort!(collect(parameter_values); by=first)), "")
    metadata["replay_from_artifact_directory"] =
        "julia --project=source source/scripts/run_basin_rescue_study.jl --config config.toml --mode $mode --case $case_name --output replay" *
        (smoke ? " --smoke" : "") * flags
    metadata["claim_limits"] = "no exact attractor count, asymptotic basin proof, biological label, or exact pulse threshold"
    return metadata
end

function run_experiment(config_path, output_dir; mode="basin", case_name="figure3",
    smoke=false, e_to_i=nothing, theta_off=nothing, tau_ratio=nothing)
    config = load_config(config_path)
    mode in ("basin", "tonic") || throw(ArgumentError("mode must be basin or tonic"))
    case = only(filter(item -> item["name"] == case_name, config.cases))
    actual = Dict{String,Float64}()
    for (name, value) in (("e_to_i", e_to_i), ("theta_off", theta_off),
        ("tau_ratio", tau_ratio))
        value === nothing && continue
        lower = config.sensitivity["$(name)_minimum"]
        upper = config.sensitivity["$(name)_maximum"]
        actual[name] = _finite(value, name; minimum=lower)
        actual[name] <= upper || throw(ArgumentError("$name exceeds the approved exploratory bound"))
    end
    parameter_values = Dict(name => get(actual, name, Float64(case[name]))
        for name in ("e_to_i", "theta_off", "tau_ratio"))
    parameter_values["theta_off"] > case["theta_on"] ||
        throw(ArgumentError("theta_off must exceed theta_on"))
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    mkpath(output)
    mkpath(joinpath(output, "contexts"))
    metadata = _metadata(config_path, output, mode, case_name, smoke, parameter_values)
    scan = config.scan
    baselines = _axis(scan["baseline_start"], scan["baseline_stop"],
        scan["baseline_step"])
    smoke && (baselines = unique([first(baselines), min(1.0, last(baselines))]))
    mode == "basin" && (baselines = [first(baselines)])
    references = Dict{Symbol,Union{Nothing,Vector{Float64}}}(
        role => Float64.(case[string(role)]) for role in ROLES)
    branch_rows = NamedTuple[]
    rescue_presence = Dict(role => Dict{Float64,Symbol}() for role in ROLES)
    trials_path = joinpath(output, "trials.csv")
    boundaries_path = joinpath(output, "boundaries.csv")
    for baseline in baselines
        model = model_for(case, baseline;
            e_to_i=parameter_values["e_to_i"],
            theta_off=parameter_values["theta_off"],
            tau_ratio=parameter_values["tau_ratio"])
        grid_points = smoke ? 5 : scan["equilibrium_grid_points"]
        search = search_model(model, grid_points)
        write_toml(joinpath(output, "contexts", "baseline_$(baseline).toml"),
            context_record(search))
        matches, reasons = match_roles(search, references, scan["branch_match_atol"])
        for role in ROLES
            index = matches[role]
            push!(branch_rows, (case=case_name, baseline_E=baseline,
                role=string(role), status=string(reasons[role]),
                equilibrium=index === nothing ? missing : index,
                E=index === nothing ? missing : search.equilibria[index].state[1],
                I=index === nothing ? missing : search.equilibria[index].state[2],
                inhibitory_input=index === nothing ? missing :
                    model.coupling.e_to_i * search.equilibria[index].state[1] -
                    model.coupling.i_to_i * search.equilibria[index].state[2],
                completeness=string(search.completeness)))
            if index !== nothing
                references[role] = copy(search.equilibria[index].state)
            else
                references[role] = nothing
            end
        end
        if mode == "basin"
            options = smoke ? BasinMeasurementOptions(grid_points=2, ray_samples=2,
                ray_refinements=1, angles=4,
                pulse_options=config.pulse_options) : config.basin_options
            area_rows, distance_rows = NamedTuple[], NamedTuple[]
            for role in ROLES
                index = matches[role]
                index === nothing && continue
                result = measure_basin(model, search, index; options)
                area = result.area
                push!(area_rows, (case=case_name, baseline_E=baseline, role=string(role),
                    observed_fraction=area.observed_fraction,
                    possible_fraction_upper=area.possible_fraction_upper,
                    source_samples=area.source_samples, other_samples=area.other_samples,
                    unresolved_samples=area.unresolved_samples,
                    total_samples=area.total_samples, cell_width=area.cell_width))
                for name in (:positive_E, :negative_E, :positive_I, :negative_I)
                    item = result.directional[name]
                    push!(distance_rows, (case=case_name, baseline_E=baseline,
                        role=string(role), measure=string(name), status=string(item.status),
                        lower=item.lower, upper=item.upper, domain_limit=item.domain_limit))
                end
                item = result.euclidean
                push!(distance_rows, (case=case_name, baseline_E=baseline,
                    role=string(role), measure="euclidean", status=string(item.status),
                    lower=item.lower, upper=item.upper, domain_limit=missing))
            end
            write_rows(joinpath(output, "area.csv"), area_rows,
                [:case, :baseline_E, :role, :observed_fraction,
                 :possible_fraction_upper, :source_samples, :other_samples,
                 :unresolved_samples, :total_samples, :cell_width])
            write_rows(joinpath(output, "distances.csv"), distance_rows,
                [:case, :baseline_E, :role, :measure, :status, :lower, :upper,
                 :domain_limit])
        else
            target = matches[:low_activity]
            durations = smoke ? [first(scan["durations"])] : scan["durations"]
            i_values = smoke ? [0.0, 1.0] :
                _axis(scan["i_amplitude_start"], scan["i_amplitude_stop"],
                    scan["i_amplitude_step"])
            e_values = smoke ? unique([0.0, baseline]) :
                _axis(0.0, baseline, scan["e_reduction_step"])
            levels = smoke ? 0 : scan["amplitude_refinement_levels"]
            for role in (:high_e_low_i, :high_e_high_i)
                source = matches[role]
                if source === nothing || target === nothing
                    rescue_presence[role][baseline] = :branch_unavailable
                    continue
                end
                rows, boundary_rows = NamedTuple[], NamedTuple[]
                found, incomplete = false, false
                for duration in durations
                    cache = Dict{Tuple{Float64,Float64},Any}()
                    function evaluate(e_reduction, i_increment)
                        key = (e_reduction, i_increment)
                        return get!(cache, key) do
                            trial = run_tonic_rescue_trial(model, search,
                                search.equilibria[source].state;
                                e_reduction, i_increment, duration,
                                options=config.pulse_options)
                            push!(rows, _trial_row(case_name, baseline, role, trial, target))
                            trial
                        end
                    end
                    for e in e_values, i in i_values
                        evaluate(e, i)
                    end
                    current_e, current_i = copy(e_values), copy(i_values)
                    for _ in 1:levels
                        new_points = Tuple{Float64,Float64}[]
                        for e in current_e, (left, right) in zip(current_i, current_i[2:end])
                            _outcome_key(evaluate(e, left)) !=
                                _outcome_key(evaluate(e, right)) &&
                                push!(new_points, (e, (left + right) / 2))
                        end
                        for i in current_i, (left, right) in zip(current_e, current_e[2:end])
                            _outcome_key(evaluate(left, i)) !=
                                _outcome_key(evaluate(right, i)) &&
                                push!(new_points, ((left + right) / 2, i))
                        end
                        for (e, i) in unique(new_points)
                            evaluate(e, i)
                        end
                        current_e = sort!(unique(vcat(current_e, first.(new_points))))
                        current_i = sort!(unique(vcat(current_i, last.(new_points))))
                    end
                    for e in sort!(unique(first.(collect(keys(cache)))))
                        sampled = sort!([i for (sample_e, i) in keys(cache) if sample_e == e])
                        for (left, right) in zip(sampled, sampled[2:end])
                            lower, upper = cache[(e, left)], cache[(e, right)]
                            _outcome_key(lower) == _outcome_key(upper) && continue
                            push!(boundary_rows, _boundary_row(case_name, baseline,
                                role, duration, :I, e, left, right, lower, upper))
                        end
                    end
                    for i in sort!(unique(last.(collect(keys(cache)))))
                        sampled = sort!([e for (e, sample_i) in keys(cache) if sample_i == i])
                        for (left, right) in zip(sampled, sampled[2:end])
                            lower, upper = cache[(left, i)], cache[(right, i)]
                            _outcome_key(lower) == _outcome_key(upper) && continue
                            push!(boundary_rows, _boundary_row(case_name, baseline,
                                role, duration, :E_reduction, i, left, right,
                                lower, upper))
                        end
                    end
                    found |= any(row -> row.rescue, rows)
                    incomplete |= any(row -> row.status != "compatible", rows)
                end
                _append(trials_path, rows)
                _append(boundaries_path, boundary_rows)
                rescue_presence[role][baseline] = found ? :observed_rescue :
                    incomplete ? :unresolved : :not_observed
            end
        end
    end
    write_rows(joinpath(output, "branches.csv"), branch_rows,
        [:case, :baseline_E, :role, :status, :equilibrium, :E, :I,
         :inhibitory_input, :completeness])
    if mode == "tonic"
        if !isfile(trials_path)
            write_rows(trials_path, NamedTuple[], [:case, :baseline_E, :source,
                :source_E, :source_I, :target, :E_reduction, :I_increment,
                :duration, :total_E, :total_I, :integrated_E, :integrated_I,
                :status, :destination, :rescue, :final_followup, :attempts])
        end
        if !isfile(boundaries_path)
            write_rows(boundaries_path, NamedTuple[], [:case, :baseline_E, :source,
                :duration, :axis, :fixed_amplitude, :lower_amplitude,
                :upper_amplitude, :lower_status, :upper_status,
                :lower_destination, :upper_destination, :censored])
        end
        presence_rows, onset_rows = NamedTuple[], NamedTuple[]
        for role in (:high_e_low_i, :high_e_high_i)
            for baseline in baselines
                push!(presence_rows, (case=case_name, source=string(role),
                    baseline_E=baseline, status=string(rescue_presence[role][baseline])))
            end
            for (lower, upper) in zip(baselines, baselines[2:end])
                first_status = rescue_presence[role][lower]
                second_status = rescue_presence[role][upper]
                first_status != second_status || continue
                push!(onset_rows, (case=case_name, source=string(role),
                    lower_baseline_E=lower, upper_baseline_E=upper,
                    lower_status=string(first_status), upper_status=string(second_status),
                    censored=first_status in (:unresolved, :branch_unavailable) ||
                        second_status in (:unresolved, :branch_unavailable)))
            end
        end
        write_rows(joinpath(output, "rescue_presence.csv"), presence_rows,
            [:case, :source, :baseline_E, :status])
        write_rows(joinpath(output, "onset_brackets.csv"), onset_rows,
            [:case, :source, :lower_baseline_E, :upper_baseline_E,
             :lower_status, :upper_status, :censored])
    end
    write_toml(joinpath(output, "metadata.toml"), metadata)
    artifact_checksums(output)
    return (output=output, cases=length(baselines))
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
        key in ("--config", "--output", "--mode", "--case", "--e-to-i",
            "--theta-off", "--tau-ratio") &&
            index < length(args) && !haskey(arguments, key) ||
            throw(ArgumentError("unknown, duplicate, or incomplete option: $key"))
        arguments[key] = args[index + 1]
        index += 2
    end
    haskey(arguments, "--output") || throw(ArgumentError("--output is required"))
    config = get(arguments, "--config", joinpath(ROOT, "experiments", "basin_rescue.toml"))
    function numeric_option(name)
        haskey(arguments, name) || return nothing
        value = tryparse(Float64, arguments[name])
        value === nothing && throw(ArgumentError("$name must be numeric"))
        return value
    end
    result = run_experiment(config, arguments["--output"];
        mode=get(arguments, "--mode", "basin"),
        case_name=get(arguments, "--case", "figure3"), smoke,
        e_to_i=numeric_option("--e-to-i"),
        theta_off=numeric_option("--theta-off"),
        tau_ratio=numeric_option("--tau-ratio"))
    println("Wrote $(result.cases) baseline contexts to $(result.output)")
    return 0
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(BasinRescueStudy.main())
end
