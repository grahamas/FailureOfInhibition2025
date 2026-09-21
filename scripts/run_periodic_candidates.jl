"""Representative basin screening followed by independent periodic shooting."""
module PeriodicCandidateExperiment

using FailureOfInhibition2025
include("run_coexistence_map.jl")
const Map = CoexistenceExperiment
const Evidence = Map.MinimalExperiment

function screening_policy(smoke)
    return (
        horizons_ms=smoke ? [5000.0] : [5000.0, 10000.0, 20000.0],
        saveat_ms=1.0, terminal_window_ms=2000.0,
        amplitude_minimum=1e-4, minimum_upward_crossings=4,
        maximum_relative_period_spread=0.1,
        ode_abstol=1e-10, ode_reltol=1e-9, ode_maxiters=1_000_000,
        diagnostic_options=DiagnosticOptions(window_duration=1000.0,
            coordinate_atol=1e-6, balance_atol=1e-8, min_samples=3),
        periodic_options=PeriodicOrbitOptions(),
    )
end

function requested_cases(config, smoke)
    names = [anchor.name for anchor in config.anchors]
    all(name -> name in names, ("figure3", "figure4_exploration")) ||
        throw(ArgumentError("the representative screen requires figure3 and figure4_exploration anchors"))
    cases = smoke ? [(anchor="figure3", e_to_i=19.0, failure_threshold=8.0,
                     condition=:failure_of_inhibition, seed=[0.25, 0.25])] :
        [(anchor=anchor, e_to_i=19.0, failure_threshold=8.0,
              condition=condition, seed=[e, i])
        for anchor in ("figure3", "figure4_exploration")
        for condition in (:control, :failure_of_inhibition)
        for e in (0.0, 0.25, 0.5) for i in (0.0, 0.25, 0.5)]
    # Targeted neighboring points use a single off-equilibrium initial state;
    # their coverage is deliberately distinct from the baseline basin grids.
    if !smoke
        for (anchor, e_to_i, failure_threshold) in (
            ("figure3", 16.0, 8.0), ("figure3", 22.0, 10.0),
            ("figure4_exploration", 16.0, 10.0), ("figure4_exploration", 22.0, 8.0),
        ), condition in (:control, :failure_of_inhibition)
            push!(cases, (; anchor, e_to_i, failure_threshold, condition, seed=[0.25, 0.25]))
        end
    end
    for case in cases
        case.e_to_i in config.e_to_i_values && case.failure_threshold in config.failure_threshold_values ||
            throw(ArgumentError("representative points must lie on the configured parameter plane"))
    end
    return cases
end

"""A finite-trace recurrence is only a seed; it never validates an orbit."""
function recurrence_seed(solution, policy)
    first_index = findfirst(time -> time >= last(solution.t) - policy.terminal_window_ms, solution.t)
    indices = first_index:length(solution.t)
    e_values = [solution.u[index][1] for index in indices]
    amplitude = maximum(e_values) - minimum(e_values)
    amplitude > policy.amplitude_minimum || return (
        status="no_seed_small_E_range", amplitude=amplitude,
        crossings=Float64[], seed=Float64[], period_guess=NaN, relative_period_spread=NaN,
    )
    level = (maximum(e_values) + minimum(e_values)) / 2
    crossing_times = Float64[]
    crossing_states = Vector{Float64}[]
    for index in first(indices):last(indices)-1
        left, right = solution.u[index][1], solution.u[index + 1][1]
        if left < level <= right
            fraction = (level - left) / (right - left)
            push!(crossing_times, solution.t[index] + fraction * (solution.t[index + 1] - solution.t[index]))
            push!(crossing_states, solution.u[index] .+ fraction .* (solution.u[index + 1] .- solution.u[index]))
        end
    end
    if length(crossing_times) < policy.minimum_upward_crossings
        return (status="no_seed_insufficient_crossings", amplitude=amplitude,
            crossings=crossing_times, seed=Float64[], period_guess=NaN, relative_period_spread=NaN)
    end
    intervals = diff(crossing_times[end-policy.minimum_upward_crossings+1:end])
    period_guess = sum(intervals) / length(intervals)
    spread = (maximum(intervals) - minimum(intervals)) / period_guess
    status = spread <= policy.maximum_relative_period_spread ?
        "shooting_seed_only" : "no_seed_irregular_recurrence"
    return (; status, amplitude, crossings=crossing_times, seed=last(crossing_states),
        period_guess, relative_period_spread=spread)
end

function periodic_record(result)
    # Dense ODE solutions contain executable closures. Archive only explicit
    # numerical fields, plus the caller's original seed and period elsewhere.
    return Dict(string(name) => getproperty(result, name)
        for name in propertynames(result) if name != :solution)
end

function run_case(case, index, model, search, output, policy)
    case_id = "case$(index)"
    stages = NamedTuple[]
    execution_success = true
    for horizon in policy.horizons_ms
        stage_id = "$(case_id)_$(Int(horizon))ms"
        record = Dict{String,Any}("case_id" => case_id, "request" => case,
            "time_span_ms" => [0.0, horizon], "biological_interpretation" => "not_assigned",
            "periodic_attractor_reachability" => "not_established")
        classification = "TrajectoryUnresolved"
        recurrence_status = "not_attempted"
        periodic_validation = "not_attempted"
        periodic_stability = "not_available"
        period = NaN
        integration_success = false
        error_message = ""
        try
            solution = solve_point_model(case.seed, (0.0, horizon), model;
                saveat=policy.saveat_ms, abstol=policy.ode_abstol,
                reltol=policy.ode_reltol, maxiters=policy.ode_maxiters)
            write_trajectory_csv(joinpath(output, "trajectories", stage_id * ".csv"), solution)
            diagnostic = diagnose_trajectory(solution, model; equilibria=search,
                options=policy.diagnostic_options)
            record["diagnostics"] = diagnostic
            record["solver_retcode"] = string(solution.retcode)
            integration_success = diagnostic.integration_success
            classification = string(diagnostic.classification)
            if integration_success && diagnostic.classification != EquilibriumCompatible
                recurrence = recurrence_seed(solution, policy)
                record["recurrence_screen"] = recurrence
                recurrence_status = recurrence.status
                if recurrence.status == "shooting_seed_only"
                    periodic = solve_periodic_orbit(model, recurrence.seed, recurrence.period_guess;
                        options=policy.periodic_options)
                    record["periodic_shooting"] = periodic_record(periodic)
                    periodic_validation = string(periodic.validation)
                    periodic_stability = string(periodic.stability)
                    period = periodic.period
                    if !isempty(periodic.times)
                        write_trajectory_csv(joinpath(output, "orbits", stage_id * ".csv"),
                            (t=periodic.times, u=periodic.states))
                    end
                end
            else
                recurrence_status = integration_success ?
                    "not_attempted_equilibrium_compatible" : "not_attempted_incomplete_integration"
            end
            record["terminal_state"] = last(solution.u)
            record["terminal_u_I"] = model.coupling.e_to_i * last(solution.u)[1] -
                model.coupling.i_to_i * last(solution.u)[2]
            record["terminal_F_I_prime"] = response_derivative(model.inhibitory.response,
                record["terminal_u_I"])
        catch error
            error isa InterruptException && rethrow()
            execution_success = false
            record["error"] = Evidence.error_record(error)
            error_message = sprint(showerror, error)
        end
        execution_success &= integration_success
        record["recurrence_status"] = recurrence_status
        record["periodic_validation"] = periodic_validation
        Evidence.write_toml(joinpath(output, "cases", stage_id * ".toml"), record)
        push!(stages, (; case_id, stage_id, anchor=case.anchor, condition=string(case.condition),
            e_to_i=case.e_to_i, failure_threshold=case.failure_threshold,
            initial_E=case.seed[1], initial_I=case.seed[2], horizon_ms=horizon,
            integration_success, classification, recurrence_status, periodic_validation,
            periodic_stability, period, error_message))
        classification == "EquilibriumCompatible" && break
        integration_success || break
    end
    return (; execution_success, stages)
end

"""Run 36 baseline basin starts and eight selected neighborhood starts.

The 3-by-3 basin grid covers `[0, 0.5]^2`; it is finite sampling, not complete
basin coverage. Ambiguous traces are re-integrated from the original state to
10 and 20 seconds. Every saved trajectory, screen rejection, and returned
shooting result is retained. Validated shooting attempts are not deduplicated
into an orbit count, since phase and nonprimitive-period equivalence are not
certified by this workflow.
"""
function run_experiment(config_path, output_dir; smoke=false)
    config = Map.load_config(config_path)
    cases = requested_cases(config, smoke)
    policy = screening_policy(smoke)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    for directory in ("contexts", "cases", "trajectories", "orbits")
        mkpath(joinpath(output, directory))
    end
    metadata = Evidence.archive_provenance(config_path, output)
    for name in ("run_coexistence_map.jl", "run_periodic_candidates.jl")
        relative = joinpath("scripts", name)
        destination = joinpath(output, "source", relative)
        cp(joinpath(@__DIR__, name), destination)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
    merge!(metadata, Dict("purpose" => "finite basin screen and numerical periodic candidate validation",
        "time_unit" => "millisecond", "smoke" => smoke, "baseline_drive" => [0.0, 0.0],
        "biological_interpretation" => "not_assigned", "completeness" => CompletenessNotCertified,
        "equilibrium_seed_policy" => "union of default and configured sharper-rectangle grids",
        "equilibrium_options" => config.equilibrium_options,
        "stability_options" => config.stability_options,
        "screening_policy" => policy,
        "trajectory_retention" => "all saved states at 1 ms spacing for every integration horizon",
        "extension_policy" => "repeat original initial-value problem at 10 and 20 seconds while equilibrium diagnostics stay unresolved",
        "cycle_count_policy" => "report attempts only; no deduplication or absence certification",
        "replay_from_artifact_directory" =>
            "julia --project=source source/scripts/run_periodic_candidates.jl --config config.toml --output replay" *
            (smoke ? " --smoke" : ""),
        "artifact_schema" => Dict("cases" => "one row per trajectory horizon; recurrence and shooting statuses are separate",
            "contexts" => "complete equilibrium searches and exact model contexts",
            "case_details" => "diagnostics, original recurrence seed/period, all final shooting evidence except executable dense solution",
            "trajectories" => "time,E,I; original time in milliseconds",
            "orbits" => "corrected sampled shooting trajectories; may be rejected, inspect validation field")))
    Evidence.write_toml(joinpath(output, "requested_cases.toml"), Dict("cases" => cases))
    searches = Dict{Tuple{String,Float64,Float64,Symbol},Any}()
    rows = NamedTuple[]
    success = true
    for (index, case) in enumerate(cases)
        key = (case.anchor, case.e_to_i, case.failure_threshold, case.condition)
        model = getproperty(Map.models_at(config, case.anchor, case.e_to_i, case.failure_threshold), case.condition)
        if !haskey(searches, key)
            context_id = "$(case.anchor)_e$(case.e_to_i)_f$(case.failure_threshold)_$(case.condition)"
            searches[key] = Map.run_search(config, model, context_id, output)
        end
        if searches[key] === nothing
            success = false
            Evidence.write_toml(joinpath(output, "cases", "case$(index)_failure.toml"),
                Dict("request" => case, "status" => "equilibrium_search_failed"))
            continue
        end
        result = run_case(case, index, model, searches[key], output, policy)
        success &= result.execution_success
        append!(rows, result.stages)
    end
    Evidence.write_rows(joinpath(output, "cases.csv"), rows,
        (:case_id, :stage_id, :anchor, :condition, :e_to_i, :failure_threshold,
         :initial_E, :initial_I, :horizon_ms, :integration_success, :classification,
         :recurrence_status, :periodic_validation, :periodic_stability, :period, :error_message))
    metadata["execution_success"] = success
    metadata["requested_trajectories"] = length(cases)
    metadata["integration_stages"] = length(rows)
    metadata["validated_shooting_attempts"] = count(row -> row.periodic_validation == "NumericallyValidatedPeriodicOrbit", rows)
    metadata["shooting_attempts"] = count(row -> row.periodic_validation != "not_attempted", rows)
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success, trajectories=length(cases), stages=length(rows),
        shooting_attempts=metadata["shooting_attempts"], validated_attempts=metadata["validated_shooting_attempts"])
end

function main(args=ARGS)
    config = joinpath(@__DIR__, "..", "experiments", "coexistence.toml")
    output = joinpath(@__DIR__, "..", "output", "periodic_candidates")
    smoke = false
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--smoke"
            smoke = true
        elseif option in ("--config", "--output")
            index < length(args) || throw(ArgumentError("missing value for $option"))
            option == "--config" ? (config = args[index + 1]) : (output = args[index + 1])
            index += 1
        else
            throw(ArgumentError("unknown option: $option"))
        end
        index += 1
    end
    result = run_experiment(config, output; smoke)
    println("Periodic screen: $(result.trajectories) starts, $(result.stages) stages, $(result.shooting_attempts) shooting attempts; success=$(result.success)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(PeriodicCandidateExperiment.main())
end
