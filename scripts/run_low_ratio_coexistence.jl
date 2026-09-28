"""Finite search for locally attracting FoI equilibria as the time-constant ratio varies."""
module LowRatioCoexistenceExperiment

using FailureOfInhibition2025
using LinearAlgebra: det, eigvals
import CSV
import TOML

include("run_tetrastability_search.jl")
const BaseStudy = TetrastabilityExperiment
const Map = BaseStudy.Map
const Evidence = BaseStudy.Evidence
const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

const CELL_COLUMNS = (:cell_id, :plane, :e_to_i, :theta_off, :status,
    :discovered_roots, :screen_intervals, :confirmed_intervals, :unresolved_nearby,
    :completeness)
const INTERVAL_COLUMNS = (:cell_id, :plane, :e_to_i, :theta_off,
    :ratio_lower, :ratio_upper, :witness_ratio, :screen_attracting,
    :confirmed_attracting, :minimum_separation, :worst_spectral_abscissa,
    :status, :reasons)
const ANCHOR_COLUMNS = (:plane, :e_to_i, :theta_off, :status, :E, :I,
    :critical_ratio, :eigenvalue_1_min_real, :eigenvalue_1_min_imaginary,
    :eigenvalue_2_min_real, :eigenvalue_2_min_imaginary,
    :eigenvalue_1_max_real, :eigenvalue_1_max_imaginary,
    :eigenvalue_2_max_real, :eigenvalue_2_max_imaginary,
    :confirmed_interval, :completeness)
const SEARCH_COLUMNS = BaseStudy.SEARCH_COLUMNS
const ATTEMPT_COLUMNS = (:context_id, :attempt, :seed_E, :seed_I,
    :candidate_E, :candidate_I, :solver_status, :solver_success, :residual_norm,
    :validation, :near_singular, :reasons)
const ROOT_COLUMNS = (:search_id, :cell_id, :grid_points, :equilibrium,
    :E, :I, :residual_norm, :near_singular, :baseline_stability,
    :balance_ee, :balance_ii, :balance_determinant, :critical_ratio,
    :attracting_at_minimum, :attracting_at_maximum,
    :minimum_spectral_abscissa, :maximum_spectral_abscissa,
    :u_I, :F_I_prime, :inhibitory_branch)

function load_config(path::AbstractString)
    raw = TOML.parsefile(path)
    BaseStudy.require_exact_keys(raw, ("schema_version", "base_config", "ratios"),
        "low-ratio configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("unsupported schema_version"))
    base_name = raw["base_config"]
    base_name isa AbstractString && basename(base_name) == base_name &&
        !isempty(base_name) || throw(ArgumentError("base_config must be a filename beside the configuration"))
    base_path = joinpath(dirname(path), base_name)
    base = BaseStudy.load_config(base_path)
    ratios = BaseStudy.require_exact_keys(raw["ratios"], ("minimum", "maximum"),
        "ratios")
    lower = BaseStudy.finite_number(ratios["minimum"], "ratios.minimum"; positive=true)
    upper = BaseStudy.finite_number(ratios["maximum"], "ratios.maximum"; positive=true)
    lower < upper || throw(ArgumentError("ratios.minimum must be below ratios.maximum"))
    lower == 0.2 && upper == base.fixed.tau_ratio ||
        throw(ArgumentError("this protocol requires the approved ratio interval [0.2, 4.4]"))
    return (; raw, base, base_path, lower, upper)
end

function model_at(config, cell, ratio; condition=:failure_of_inhibition)
    base = config.base
    inhibitory_control = PopulationParameters(timescale=base.fixed.tau_e * ratio,
        response=base.inhibitory_control.response)
    ratio_config = merge(base, (; inhibitory_control))
    return getproperty(BaseStudy.models_for_cell(ratio_config, cell), condition)
end

"""Return the original-time linear spectrum at a fixed equilibrium."""
function spectrum(balance_jacobian, tau_e, ratio)
    matrix = [balance_jacobian[1, 1] balance_jacobian[1, 2];
        balance_jacobian[2, 1] / ratio balance_jacobian[2, 2] / ratio] / tau_e
    return eigvals(matrix)
end

function attracting(balance_jacobian, tau_e, ratio; margin=0.0)
    determinant = det(balance_jacobian)
    isfinite(determinant) && determinant > 0 || return false
    values = spectrum(balance_jacobian, tau_e, ratio)
    return all(value -> isfinite(real(value)) && real(value) < -margin, values)
end

critical_ratio(balance_jacobian) = balance_jacobian[1, 1] == 0 ? NaN :
    -balance_jacobian[2, 2] / balance_jacobian[1, 1]

"""Open ratio segments with at least four locally attracting discovered roots."""
function candidate_intervals(jacobians, tau_e, lower, upper)
    cuts = Float64[lower, upper]
    for jacobian in jacobians
        ratio = critical_ratio(jacobian)
        det(jacobian) > 0 && isfinite(ratio) && lower < ratio < upper &&
            push!(cuts, ratio)
    end
    unique!(sort!(cuts))
    intervals = NamedTuple[]
    for index in 1:(length(cuts) - 1)
        left, right = cuts[index], cuts[index + 1]
        witness = left + (right - left) / 2
        left < witness < right || continue
        indices = findall(jacobian -> attracting(jacobian, tau_e, witness), jacobians)
        length(indices) >= 4 && push!(intervals,
            (; lower=left, upper=right, witness, indices))
    end
    return intervals
end

_distance(left, right) = max(abs(left[1] - right[1]), abs(left[2] - right[2]))

function unique_match(reference, references, candidates, tolerance)
    matches = findall(candidate -> _distance(reference.state, candidate.state) <= tolerance,
        candidates)
    length(matches) == 1 || return nothing
    selected = only(matches)
    count(other -> _distance(other.state, candidates[selected].state) <= tolerance,
        references) == 1 || return nothing
    return selected
end

function root_quality(equilibrium, model, ratio, config)
    state = equilibrium.state
    residual = zeros(Float64, 2)
    balance_jacobian = zeros(Float64, 2, 2)
    point_balance!(residual, state, model, 0.0)
    point_balance_jacobian!(balance_jacobian, state, model, 0.0)
    options = config.base
    spectral = maximum(real, spectrum(balance_jacobian,
        options.fixed.tau_e, ratio))
    return (; state=Tuple(state),
        quality=!equilibrium.near_singular &&
            maximum(abs, residual) <= options.equilibrium_options.residual_atol &&
            maximum(abs, balance_jacobian .- equilibrium.balance_jacobian) <=
                options.equilibrium_options.residual_atol &&
            isfinite(spectral) && spectral <= -options.confirmation.spectral_margin,
        spectral)
end

function confirm_interval(config, cell, interval, searches)
    reasons = String[]
    any(isnothing, searches) && return (; confirmed=false, matched=0,
        separation=NaN, worst_spectral=NaN,
        reasons=["confirmation_search_failed"])
    any(search -> !isempty(search.unresolved_nearby), searches) &&
        push!(reasons, "unresolved_nearby_roots")
    model = model_at(config, cell, interval.witness)
    roots = [[root_quality(eq, model, interval.witness, config)
        for eq in search.equilibria] for search in searches]
    references = first(roots)
    quality_tracks = Vector{Int}[]
    worst_spectral = -Inf
    for index in interval.indices
        matches = [index]
        for candidates in roots[2:end]
            match = unique_match(references[index], references, candidates,
                config.base.confirmation.match_atol)
            isnothing(match) && break
            push!(matches, match)
        end
        length(matches) == length(roots) || continue
        observations = [roots[grid][matches[grid]] for grid in eachindex(roots)]
        all(observation -> observation.quality, observations) || continue
        push!(quality_tracks, matches)
        worst_spectral = max(worst_spectral,
            maximum(observation.spectral for observation in observations))
    end
    length(quality_tracks) >= 4 || push!(reasons, "fewer_than_four_matched_quality_roots")
    separation = if length(quality_tracks) >= 2
        minimum(_distance(roots[grid][left[grid]].state,
            roots[grid][right[grid]].state)
            for grid in eachindex(roots)
            for (left_index, left) in enumerate(quality_tracks)
            for right in quality_tracks[left_index + 1:end])
    else
        NaN
    end
    required = config.base.confirmation.separation_factor *
        config.base.equilibrium_options.dedup_atol
    isfinite(separation) && separation > required ||
        push!(reasons, "insufficient_root_separation")
    return (; confirmed=isempty(reasons), matched=length(quality_tracks),
        separation, worst_spectral=isempty(quality_tracks) ? NaN : worst_spectral,
        reasons)
end

function append_rows(path, rows)
    isempty(rows) || CSV.write(path, rows; append=true)
    return nothing
end

function search!(config, cell, grid_points, output; refined=false)
    search_id = "$(cell.cell_id)_grid$(grid_points)"
    model = model_at(config, cell, config.lower)
    seeds = Map.deterministic_seeds(model, grid_points)
    equilibrium_options = refined ? BaseStudy.refined_options(
        config.base.equilibrium_options, config.base.confirmation.tolerance_factor) :
        config.base.equilibrium_options
    stability_options = refined ? BaseStudy.refined_options(
        config.base.stability_options, config.base.confirmation.tolerance_factor) :
        config.base.stability_options
    result = try
        find_equilibria(model; seeds, options=equilibrium_options,
            stability_options=stability_options)
    catch error
        error isa InterruptException && rethrow()
        Evidence.write_toml(joinpath(output, "contexts", search_id * ".toml"),
            Dict("status" => "execution_failed", "error" => Evidence.error_record(error),
                "model" => model, "requested_seeds" => seeds))
        nothing
    end
    if !isnothing(result)
        Evidence.write_toml(joinpath(output, "contexts", search_id * ".toml"),
            merge(Dict("search_id" => search_id, "cell_id" => cell.cell_id,
                "grid_points" => grid_points, "model" => model,
                "requested_seeds" => seeds), Evidence.context_record(result)))
        append_rows(joinpath(output, "attempts.csv"),
            [Evidence.summary_attempt(search_id, index, attempt)
                for (index, attempt) in enumerate(result.attempts)])
        append_rows(joinpath(output, "roots.csv"), [begin
            jacobian = eq.balance_jacobian
            at_min = spectrum(jacobian, config.base.fixed.tau_e, config.lower)
            at_max = spectrum(jacobian, config.base.fixed.tau_e, config.upper)
            observed = Map.equilibrium_observations(model, eq)
            (; search_id, cell_id=cell.cell_id, grid_points, equilibrium=index,
                E=eq.state[1], I=eq.state[2],
                residual_norm=maximum(abs, eq.balance_residual),
                near_singular=eq.near_singular,
                baseline_stability=string(eq.stability.classification),
                balance_ee=jacobian[1, 1], balance_ii=jacobian[2, 2],
                balance_determinant=det(jacobian),
                critical_ratio=critical_ratio(jacobian),
                attracting_at_minimum=attracting(jacobian,
                    config.base.fixed.tau_e, config.lower),
                attracting_at_maximum=attracting(jacobian,
                    config.base.fixed.tau_e, config.upper),
                minimum_spectral_abscissa=maximum(real, at_min),
                maximum_spectral_abscissa=maximum(real, at_max),
                u_I=observed.u_I, F_I_prime=observed.F_I_prime,
                inhibitory_branch=observed.inhibitory_branch)
        end for (index, eq) in enumerate(result.equilibria)])
    end
    append_rows(joinpath(output, "searches.csv"),
        [BaseStudy._search_summary(search_id, cell, :failure_of_inhibition,
            grid_points, result)])
    return result
end

function anchor_row(config, cell, result, confirmed)
    default = (; plane=cell.plane, e_to_i=cell.e_to_i,
        theta_off=cell.theta_off, status="unresolved", E=NaN, I=NaN,
        critical_ratio=NaN, eigenvalue_1_min_real=NaN,
        eigenvalue_1_min_imaginary=NaN, eigenvalue_2_min_real=NaN,
        eigenvalue_2_min_imaginary=NaN, eigenvalue_1_max_real=NaN,
        eigenvalue_1_max_imaginary=NaN, eigenvalue_2_max_real=NaN,
        eigenvalue_2_max_imaginary=NaN, confirmed_interval=confirmed,
        completeness=string(CompletenessNotCertified))
    isnothing(result) && return default
    candidates = [eq for eq in result.equilibria if
        attracting(eq.balance_jacobian, config.base.fixed.tau_e, config.lower) &&
        !attracting(eq.balance_jacobian, config.base.fixed.tau_e, config.upper) &&
        det(eq.balance_jacobian) > 0 &&
        config.lower < critical_ratio(eq.balance_jacobian) < config.upper]
    length(candidates) == 1 || return default
    eq = only(candidates)
    minimum = spectrum(eq.balance_jacobian, config.base.fixed.tau_e, config.lower)
    maximum = spectrum(eq.balance_jacobian, config.base.fixed.tau_e, config.upper)
    return merge(default, (; status=confirmed ? "confirmed_local_stability" : "screen_only",
        E=eq.state[1], I=eq.state[2],
        critical_ratio=critical_ratio(eq.balance_jacobian),
        eigenvalue_1_min_real=real(minimum[1]),
        eigenvalue_1_min_imaginary=imag(minimum[1]),
        eigenvalue_2_min_real=real(minimum[2]),
        eigenvalue_2_min_imaginary=imag(minimum[2]),
        eigenvalue_1_max_real=real(maximum[1]),
        eigenvalue_1_max_imaginary=imag(maximum[1]),
        eigenvalue_2_max_real=real(maximum[2]),
        eigenvalue_2_max_imaginary=imag(maximum[2])))
end

function run_experiment(config_path::AbstractString, output_dir::AbstractString; smoke=false)
    config = load_config(config_path)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or empty"))
    mkpath(joinpath(output, "contexts"))
    metadata = Evidence.archive_provenance(config_path, output)
    cp(config.base_path, joinpath(output, basename(config.base_path)))
    for name in ("run_coexistence_map.jl", "run_tetrastability_search.jl",
        "run_low_ratio_coexistence.jl")
        relative = joinpath("scripts", name)
        destination = joinpath(output, "source", relative)
        cp(joinpath(REPOSITORY_ROOT, relative), destination)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
    metadata["base_config_sha256"] = Evidence.file_hash(
        joinpath(output, basename(config.base_path)))
    merge!(metadata, Dict("experiment" => "low_ratio_coexistence",
        "purpose" => "finite local equilibrium stability across tau_I/tau_E",
        "ratio_bounds" => [config.lower, config.upper],
        "equilibrium_seed_policy" => "default 5-by-5 union with 11-by-11 screen; tighter 21-by-21 and 41-by-41 confirmation",
        "completeness" => CompletenessNotCertified,
        "biological_interpretation" => "not_assigned",
        "periodic_orbit_status" => "not_tested",
        "smoke" => smoke,
        "boundary_policy" => "critical ratios are open linear-stability boundaries; no Hopf certification",
        "artifact_schema" => Dict("version" => 1,
            "cells" => "one finite sampled plane cell with screen and confirmation status",
            "intervals" => "open screen ratio segments with at least four attracting roots; confirmation applies to the recorded witness only",
            "roots" => "every discovered equilibrium with balance Jacobian and endpoint spectra",
            "searches" => "all completed and failed grid searches",
            "attempts" => "every nonlinear attempt in each search",
            "contexts" => "full search attempts, equilibria, numerical options and failures",
            "anchors" => "two manuscript examples with intermediate-root endpoint spectra"),
        "replay_from_artifact_directory" => "julia --project=source source/scripts/run_low_ratio_coexistence.jl --config config.toml --output replay" * (smoke ? " --smoke" : "")))
    for (file, columns) in (("cells.csv", CELL_COLUMNS),
        ("intervals.csv", INTERVAL_COLUMNS), ("anchors.csv", ANCHOR_COLUMNS),
        ("searches.csv", SEARCH_COLUMNS), ("attempts.csv", ATTEMPT_COLUMNS),
        ("roots.csv", ROOT_COLUMNS))
        Evidence.write_rows(joinpath(output, file), NamedTuple[], columns)
    end
    cells = BaseStudy.plane_cells(config.base; smoke)
    failures = String[]
    confirmed_cells = 0
    screen_positive = 0
    for cell in cells
        screen = search!(config, cell, config.base.screen_grid_points, output)
        intervals = isnothing(screen) ? NamedTuple[] : candidate_intervals(
            [eq.balance_jacobian for eq in screen.equilibria],
            config.base.fixed.tau_e, config.lower, config.upper)
        !isempty(intervals) && (screen_positive += 1)
        refined = isempty(intervals) ? Any[] : [search!(config, cell, grid, output;
            refined=true) for grid in config.base.confirmation_grid_points]
        confirmed_count = 0
        for interval in intervals
            assessment = confirm_interval(config, cell, interval, [screen; refined])
            assessment.confirmed && (confirmed_count += 1)
            append_rows(joinpath(output, "intervals.csv"),
                [(; cell_id=cell.cell_id, plane=cell.plane, e_to_i=cell.e_to_i,
                    theta_off=cell.theta_off, ratio_lower=interval.lower,
                    ratio_upper=interval.upper, witness_ratio=interval.witness,
                    screen_attracting=length(interval.indices),
                    confirmed_attracting=assessment.matched,
                    minimum_separation=assessment.separation,
                    worst_spectral_abscissa=assessment.worst_spectral,
                    status=assessment.confirmed ? "witness_confirmed" : "unresolved",
                    reasons=join(assessment.reasons, ";"))])
        end
        confirmed_count > 0 && (confirmed_cells += 1)
        any(isnothing, [screen; refined]) && push!(failures, cell.cell_id)
        screen_uncertain = !isnothing(screen) && (
            !isempty(screen.unresolved_nearby) ||
            any(eq -> eq.near_singular ||
                eq.stability.classification == StabilityUnresolved, screen.equilibria))
        status = isnothing(screen) ? "execution_failed" :
            screen_uncertain ? "screen_unresolved" :
            isempty(intervals) ? "screen_resolved_no_candidate" :
            confirmed_count > 0 ? "witness_confirmed" : "candidate_unresolved"
        append_rows(joinpath(output, "cells.csv"),
            [(; cell_id=cell.cell_id, plane=cell.plane, e_to_i=cell.e_to_i,
                theta_off=cell.theta_off, status,
                discovered_roots=isnothing(screen) ? -1 : length(screen.equilibria),
                screen_intervals=length(intervals), confirmed_intervals=confirmed_count,
                unresolved_nearby=isnothing(screen) ? -1 : length(screen.unresolved_nearby),
                completeness=string(CompletenessNotCertified))])
        if cell.e_to_i == 19.0 && cell.theta_off == 8.0
            append_rows(joinpath(output, "anchors.csv"),
                [anchor_row(config, cell, screen, confirmed_count > 0)])
        end
    end
    metadata["execution_success"] = isempty(failures)
    metadata["failed_cells"] = failures
    metadata["parameter_cells"] = length(cells)
    metadata["screen_positive_cells"] = screen_positive
    metadata["confirmed_cells"] = confirmed_cells
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success=isempty(failures), cells=length(cells), screen_positive,
        confirmed_cells)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "low_ratio_coexistence.toml")
    output = nothing
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
            index < length(args) || throw(ArgumentError("$option requires a value"))
            index += 1
            option == "--config" ? (config_path = args[index]) : (output = args[index])
        else
            throw(ArgumentError("unknown option: $option"))
        end
        index += 1
    end
    isnothing(output) && throw(ArgumentError("--output is required"))
    result = run_experiment(config_path, output; smoke)
    println("Low-ratio study: $(result.cells) cells, $(result.screen_positive) screen-positive, $(result.confirmed_cells) confirmed")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(LowRatioCoexistenceExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
