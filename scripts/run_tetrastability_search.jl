"""Deterministic finite search for four locally attracting FoI equilibria."""
module TetrastabilityExperiment

using FailureOfInhibition2025
import CSV
import TOML

include("run_coexistence_map.jl")
const Map = CoexistenceExperiment
const Evidence = Map.MinimalExperiment

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const SEARCH_AXES = (:e_to_e, :i_to_e, :e_to_i, :i_to_i, :theta_off)
const HALTON_BASES = (2, 3, 5, 7, 11)
const CANDIDATE_MINIMUM = 4
const CONDITIONS = (:control, :failure_of_inhibition)

const SAMPLE_COLUMNS = (
    :cell_id, :family, :stage, :plane, :e_to_i_index, :theta_off_index,
    :halton_index, :halton_e_to_e, :halton_i_to_e, :halton_e_to_i,
    :halton_i_to_i, :halton_theta_off, :e_to_e, :i_to_e, :e_to_i,
    :i_to_i, :theta_off,
)
const SEARCH_COLUMNS = (
    :search_id, :cell_id, :stage, :condition, :grid_points, :status,
    :discovered_equilibria, :attracting_equilibria, :saddle_equilibria,
    :repelling_equilibria, :stability_unresolved, :near_singular_equilibria,
    :unresolved_nearby_groups, :attempts, :rejected_attempts, :completeness,
)
const ATTEMPT_COLUMNS = (
    :search_id, :attempt, :seed_E, :seed_I, :candidate_E, :candidate_I,
    :solver_status, :solver_success, :residual_norm, :validation,
    :near_singular, :reasons,
)
const EQUILIBRIUM_COLUMNS = (
    :search_id, :equilibrium, :condition, :grid_points, :E, :I,
    :residual_norm, :near_singular, :representative_attempt, :member_attempts,
    :stability, :geometry, :eigenvalue_1_real, :eigenvalue_1_imaginary,
    :eigenvalue_2_real, :eigenvalue_2_imaginary, :spectral_abscissa,
    :u_I, :F_I, :F_I_prime, :inhibitory_response_midpoint,
    :inhibitory_branch, :descending_response_branch,
)
const ROOT_MATCH_COLUMNS = (
    :candidate_id, :root_track, :grid_points, :search_id, :equilibrium,
    :match_distance, :E, :I, :attracting, :near_singular, :residual_norm,
    :recomputed_residual_norm, :balance_jacobian_delta, :ode_jacobian_delta,
    :spectral_abscissa, :quality_pass,
)
const CANDIDATE_COLUMNS = (
    :candidate_id, :cell_id, :e_to_e, :i_to_e, :e_to_i, :i_to_i,
    :theta_off, :discovery_attracting, :grid_21_attracting,
    :grid_41_attracting, :matched_attracting_roots, :minimum_separation,
    :worst_spectral_abscissa, :maximum_recomputed_residual,
    :maximum_balance_jacobian_delta, :maximum_ode_jacobian_delta,
    :confirmation_status, :reasons,
)
const BRANCH_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :points, :attempts,
    :screening_candidates, :termination,
)
const CONTINUATION_POINT_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :point, :parameter,
    :E, :I, :residual_norm, :tangent_parameter, :stability,
    :spectral_abscissa, :trace, :determinant,
)
const CONTINUATION_ATTEMPT_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :attempt, :step,
    :iterations, :arclength_residual, :status, :accepted,
)
const CONTINUATION_CANDIDATE_COLUMNS = (
    :candidate_id, :root_track, :axis, :direction, :kind,
    :first_point, :second_point, :lower, :upper,
)

function require_exact_keys(table, expected_keys, label)
    table isa AbstractDict || throw(ArgumentError("$label must be a table"))
    missing = filter(key -> !haskey(table, key), expected_keys)
    isempty(missing) || throw(ArgumentError(
        "$label is missing required keys: $(join(missing, ", "))"))
    extra = setdiff(collect(keys(table)), collect(expected_keys))
    isempty(extra) || throw(ArgumentError(
        "$label has unknown keys: $(join(sort!(extra), ", "))"))
    return table
end

function finite_number(value, label; positive=false, nonnegative=false)
    return Map.finite_number(value, label; positive, nonnegative)
end

function pairwise_bound(table, label; step=false)
    expected = step ? ("minimum", "maximum", "plane_step") : ("minimum", "maximum")
    require_exact_keys(table, expected, label)
    lower = finite_number(table["minimum"], "$label.minimum"; nonnegative=true)
    upper = finite_number(table["maximum"], "$label.maximum"; nonnegative=true)
    upper > lower || throw(ArgumentError("$label.maximum must exceed its minimum"))
    plane_step = step ? finite_number(table["plane_step"], "$label.plane_step"; positive=true) : NaN
    if step
        intervals = (upper - lower) / plane_step
        isapprox(intervals, round(Int, intervals); atol=1e-10, rtol=1e-12) ||
            throw(ArgumentError("$label.plane_step must divide its extent"))
    end
    return (; lower, upper, plane_step)
end

function _is_prime(value)
    value isa Integer && !(value isa Bool) && value > 1 || return false
    return all(divisor -> value % divisor != 0, 2:isqrt(value))
end

function _canonical_check(config)
    config.fixed == (tau_e=7.8, tau_ratio=4.4, a_e=5.0, a_i=5.0,
        theta_e=1.5, theta_on=4.0) ||
        throw(ArgumentError("fixed model parameters differ from the canonical protocol"))
    config.planes == [
        (name="figure3", e_to_e=17.0, i_to_e=9.0, i_to_i=4.0),
        (name="figure4_exploration", e_to_e=19.0, i_to_e=13.0, i_to_i=6.0),
    ] || throw(ArgumentError("planes differ from the canonical protocol"))
    expected_bounds = Dict(
        :e_to_e => (14.0, 24.0), :i_to_e => (6.0, 18.0),
        :e_to_i => (12.0, 28.0), :i_to_i => (0.0, 10.0),
        :theta_off => (6.0, 12.0),
    )
    all(axis -> (config.axes[axis].lower, config.axes[axis].upper) == expected_bounds[axis],
        SEARCH_AXES) || throw(ArgumentError("axis bounds differ from the canonical protocol"))
    config.axes.e_to_i.plane_step == 0.5 && config.axes.theta_off.plane_step == 0.25 ||
        throw(ArgumentError("plane steps differ from the canonical protocol"))
    config.halton.initial_points == 1024 && config.halton.maximum_points == 4096 &&
        config.halton.bases == HALTON_BASES ||
        throw(ArgumentError("Halton schedule differs from the canonical protocol"))
    config.screen_grid_points == 11 && config.confirmation_grid_points == [21, 41] ||
        throw(ArgumentError("seed-grid policy differs from the canonical protocol"))
    (config.smoke_halton_points, config.smoke_e_to_i, config.smoke_theta_off) ==
        (2, [19.0], [8.0]) ||
        throw(ArgumentError("smoke schedule differs from the canonical protocol"))
    equilibrium = config.equilibrium_options
    (equilibrium.solver_abstol, equilibrium.solver_reltol,
        equilibrium.residual_atol, equilibrium.domain_atol,
        equilibrium.dedup_atol, equilibrium.singular_atol,
        equilibrium.singular_rtol, equilibrium.maxiters) ==
        (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-10, 1e-8, 100) ||
        throw(ArgumentError("equilibrium policy differs from the canonical protocol"))
    stability = config.stability_options
    (stability.spectral_atol, stability.spectral_rtol) == (1e-10, 1e-8) ||
        throw(ArgumentError("stability policy differs from the canonical protocol"))
    config.confirmation == (tolerance_factor=0.1, match_atol=1e-6,
        separation_factor=100.0, spectral_margin=1e-8) ||
        throw(ArgumentError("confirmation policy differs from the canonical protocol"))
    config.continuation == (initial_step=0.01, minimum_step=1e-5,
        maximum_step=0.03, max_steps=500, smoke_max_steps=3,
        max_corrector_iters=12, corrector_atol=1e-10,
        parameter_difference_step=1e-5, rank_atol=1e-12, rank_rtol=1e-10) ||
        throw(ArgumentError("continuation policy differs from the canonical protocol"))
    return config
end

"""Load and fail-closed validate the tetrastability protocol."""
function load_config(path::AbstractString; require_canonical=true)
    raw = TOML.parsefile(path)
    require_exact_keys(raw, ("schema_version", "fixed", "planes", "axes", "halton",
        "search", "smoke", "equilibrium", "stability", "confirmation", "continuation"),
        "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("schema_version must be integer 1"))

    fixed_table = require_exact_keys(raw["fixed"],
        ("tau_e", "tau_ratio", "a_e", "a_i", "theta_e", "theta_on"), "fixed")
    fixed = (;
        tau_e=finite_number(fixed_table["tau_e"], "fixed.tau_e"; positive=true),
        tau_ratio=finite_number(fixed_table["tau_ratio"], "fixed.tau_ratio"; positive=true),
        a_e=finite_number(fixed_table["a_e"], "fixed.a_e"; positive=true),
        a_i=finite_number(fixed_table["a_i"], "fixed.a_i"; positive=true),
        theta_e=finite_number(fixed_table["theta_e"], "fixed.theta_e"),
        theta_on=finite_number(fixed_table["theta_on"], "fixed.theta_on"),
    )

    raw["planes"] isa AbstractVector && !isempty(raw["planes"]) ||
        throw(ArgumentError("planes must be a nonempty array"))
    planes = map(raw["planes"]) do table
        require_exact_keys(table, ("name", "e_to_e", "i_to_e", "i_to_i"), "plane")
        name = table["name"]
        name isa AbstractString && occursin(r"^[a-z][a-z0-9_]*$", name) ||
            throw(ArgumentError("plane.name must be a lowercase identifier"))
        (; name=String(name),
            e_to_e=finite_number(table["e_to_e"], "plane.e_to_e"; nonnegative=true),
            i_to_e=finite_number(table["i_to_e"], "plane.i_to_e"; nonnegative=true),
            i_to_i=finite_number(table["i_to_i"], "plane.i_to_i"; nonnegative=true))
    end
    allunique(plane.name for plane in planes) || throw(ArgumentError("plane names must be unique"))

    axes_table = require_exact_keys(raw["axes"], string.(SEARCH_AXES), "axes")
    axes = (;
        e_to_e=pairwise_bound(axes_table["e_to_e"], "axes.e_to_e"),
        i_to_e=pairwise_bound(axes_table["i_to_e"], "axes.i_to_e"),
        e_to_i=pairwise_bound(axes_table["e_to_i"], "axes.e_to_i"; step=true),
        i_to_i=pairwise_bound(axes_table["i_to_i"], "axes.i_to_i"),
        theta_off=pairwise_bound(axes_table["theta_off"], "axes.theta_off"; step=true),
    )
    axes.theta_off.lower > fixed.theta_on ||
        throw(ArgumentError("theta_off bounds must remain above theta_on"))

    halton_table = require_exact_keys(raw["halton"],
        ("initial_points", "maximum_points", "bases"), "halton")
    initial_points = Map.positive_integer(halton_table["initial_points"], "halton.initial_points")
    maximum_points = Map.positive_integer(halton_table["maximum_points"], "halton.maximum_points")
    maximum_points >= initial_points ||
        throw(ArgumentError("halton.maximum_points must be at least initial_points"))
    bases_raw = halton_table["bases"]
    bases_raw isa AbstractVector && length(bases_raw) == length(SEARCH_AXES) &&
        all(_is_prime, bases_raw) && allunique(bases_raw) ||
        throw(ArgumentError("halton.bases must be five distinct primes"))
    bases = Tuple(Int.(bases_raw))
    halton = (; initial_points, maximum_points, bases)

    search_table = require_exact_keys(raw["search"],
        ("screen_grid_points", "confirmation_grid_points"), "search")
    screen_grid_points = Map.positive_integer(search_table["screen_grid_points"],
        "search.screen_grid_points")
    screen_grid_points > 5 ||
        throw(ArgumentError("search.screen_grid_points must exceed the default 5-by-5 grid"))
    confirmation_raw = search_table["confirmation_grid_points"]
    confirmation_raw isa AbstractVector && !isempty(confirmation_raw) ||
        throw(ArgumentError("search.confirmation_grid_points must be a nonempty array"))
    confirmation_grid_points = [Map.positive_integer(value,
        "search.confirmation_grid_points") for value in confirmation_raw]
    allunique(confirmation_grid_points) && issorted(confirmation_grid_points) &&
        all(>(screen_grid_points), confirmation_grid_points) ||
        throw(ArgumentError("confirmation grids must be unique, sorted and exceed the screen grid"))

    smoke_table = require_exact_keys(raw["smoke"],
        ("halton_points", "e_to_i", "theta_off"), "smoke")
    smoke_halton_points = Map.positive_integer(smoke_table["halton_points"], "smoke.halton_points")
    smoke_halton_points <= initial_points ||
        throw(ArgumentError("smoke.halton_points must not exceed the initial Halton batch"))
    axis_values(axis) = collect(range(axis.lower; step=axis.plane_step,
        length=round(Int, (axis.upper - axis.lower) / axis.plane_step) + 1))
    function smoke_values(values, axis, label)
        values isa AbstractVector && !isempty(values) ||
            throw(ArgumentError("$label must be a nonempty array"))
        parsed = [finite_number(value, label) for value in values]
        allunique(parsed) || throw(ArgumentError("$label must be unique"))
        available = axis_values(axis)
        all(value -> any(candidate -> isapprox(value, candidate; atol=1e-12, rtol=1e-12),
            available), parsed) || throw(ArgumentError("$label must lie on its plane grid"))
        return sort!(parsed)
    end
    smoke_e_to_i = smoke_values(smoke_table["e_to_i"], axes.e_to_i, "smoke.e_to_i")
    smoke_theta_off = smoke_values(smoke_table["theta_off"], axes.theta_off, "smoke.theta_off")

    equilibrium_table = require_exact_keys(raw["equilibrium"],
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium")
    equilibrium_options = Map.options_from(equilibrium_table, EquilibriumOptions,
        ("solver_abstol", "solver_reltol", "residual_atol", "domain_atol", "dedup_atol",
         "singular_atol", "singular_rtol", "maxiters"), "equilibrium";
        integer_key="maxiters")
    stability_table = require_exact_keys(raw["stability"],
        ("spectral_atol", "spectral_rtol"), "stability")
    stability_options = Map.options_from(stability_table, StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")

    confirmation_table = require_exact_keys(raw["confirmation"],
        ("tolerance_factor", "match_atol", "separation_factor", "spectral_margin"),
        "confirmation")
    tolerance_factor = finite_number(confirmation_table["tolerance_factor"],
        "confirmation.tolerance_factor"; positive=true)
    tolerance_factor < 1 || throw(ArgumentError("confirmation.tolerance_factor must be below one"))
    confirmation = (;
        tolerance_factor,
        match_atol=finite_number(confirmation_table["match_atol"],
            "confirmation.match_atol"; positive=true),
        separation_factor=finite_number(confirmation_table["separation_factor"],
            "confirmation.separation_factor"; positive=true),
        spectral_margin=finite_number(confirmation_table["spectral_margin"],
            "confirmation.spectral_margin"; positive=true),
    )
    confirmation.match_atol > equilibrium_options.dedup_atol ||
        throw(ArgumentError("confirmation.match_atol must exceed the discovery dedup_atol"))

    continuation_table = require_exact_keys(raw["continuation"],
        ("initial_step", "minimum_step", "maximum_step", "max_steps", "smoke_max_steps",
         "max_corrector_iters", "corrector_atol", "parameter_difference_step",
         "rank_atol", "rank_rtol"), "continuation")
    continuation = (;
        initial_step=finite_number(continuation_table["initial_step"],
            "continuation.initial_step"; positive=true),
        minimum_step=finite_number(continuation_table["minimum_step"],
            "continuation.minimum_step"; positive=true),
        maximum_step=finite_number(continuation_table["maximum_step"],
            "continuation.maximum_step"; positive=true),
        max_steps=Map.positive_integer(continuation_table["max_steps"], "continuation.max_steps"),
        smoke_max_steps=Map.positive_integer(continuation_table["smoke_max_steps"],
            "continuation.smoke_max_steps"),
        max_corrector_iters=Map.positive_integer(continuation_table["max_corrector_iters"],
            "continuation.max_corrector_iters"),
        corrector_atol=finite_number(continuation_table["corrector_atol"],
            "continuation.corrector_atol"; positive=true),
        parameter_difference_step=finite_number(continuation_table["parameter_difference_step"],
            "continuation.parameter_difference_step"; positive=true),
        rank_atol=finite_number(continuation_table["rank_atol"],
            "continuation.rank_atol"; positive=true),
        rank_rtol=finite_number(continuation_table["rank_rtol"],
            "continuation.rank_rtol"; positive=true),
    )
    continuation.minimum_step <= continuation.initial_step <= continuation.maximum_step ||
        throw(ArgumentError("continuation steps must satisfy minimum <= initial <= maximum"))

    excitatory = PopulationParameters(timescale=fixed.tau_e,
        response=LogisticResponse(slope=fixed.a_e, threshold=fixed.theta_e))
    inhibitory_control = PopulationParameters(timescale=fixed.tau_e * fixed.tau_ratio,
        response=LogisticResponse(slope=fixed.a_i, threshold=fixed.theta_on))
    config = (; raw, fixed, planes, axes, halton, screen_grid_points,
        confirmation_grid_points, smoke_halton_points, smoke_e_to_i, smoke_theta_off,
        equilibrium_options, stability_options, confirmation, continuation,
        excitatory, inhibitory_control)
    return require_canonical ? _canonical_check(config) : config
end

"""Unscrambled one-indexed radical inverse."""
function radical_inverse(index, base)
    index isa Integer && !(index isa Bool) && index > 0 ||
        throw(ArgumentError("Halton index must be a positive integer"))
    _is_prime(base) || throw(ArgumentError("Halton base must be prime"))
    value = 0.0
    factor = inv(Float64(base))
    current = Int(index)
    while current > 0
        current, digit = divrem(current, base)
        value += digit * factor
        factor /= base
    end
    return value
end

function _axis_value(axis, unit)
    return axis.lower + unit * (axis.upper - axis.lower)
end

function halton_cell(config, index; stage="initial")
    units = ntuple(dimension -> radical_inverse(index, config.halton.bases[dimension]),
        length(SEARCH_AXES))
    values = ntuple(dimension -> _axis_value(getproperty(config.axes,
        SEARCH_AXES[dimension]), units[dimension]), length(SEARCH_AXES))
    return (; cell_id="halton_$(lpad(index, 4, '0'))", family="halton", stage,
        plane="", e_to_i_index=0, theta_off_index=0, halton_index=index,
        halton_e_to_e=units[1], halton_i_to_e=units[2], halton_e_to_i=units[3],
        halton_i_to_i=units[4], halton_theta_off=units[5],
        e_to_e=values[1], i_to_e=values[2], e_to_i=values[3], i_to_i=values[4],
        theta_off=values[5])
end

function halton_cells(config, indices; stage="initial")
    return [halton_cell(config, index; stage) for index in indices]
end

function extension_indices(config, has_screen_candidate; smoke=false)
    smoke && return 1:0
    has_screen_candidate && return 1:0
    config.halton.maximum_points <= config.halton.initial_points && return 1:0
    return config.halton.initial_points + 1:config.halton.maximum_points
end

function _plane_values(axis, smoke_values, smoke)
    smoke && return smoke_values
    return collect(range(axis.lower; step=axis.plane_step,
        length=round(Int, (axis.upper - axis.lower) / axis.plane_step) + 1))
end

function plane_cells(config; smoke=false)
    e_values = _plane_values(config.axes.e_to_i, config.smoke_e_to_i, smoke)
    threshold_values = _plane_values(config.axes.theta_off, config.smoke_theta_off, smoke)
    rows = NamedTuple[]
    for plane in config.planes, (e_index, e_to_i) in enumerate(e_values),
        (threshold_index, theta_off) in enumerate(threshold_values)
        push!(rows, (; cell_id="plane_$(plane.name)_e$(lpad(e_index, 2, '0'))_t$(lpad(threshold_index, 2, '0'))",
            family="plane", stage="plane", plane=plane.name, e_to_i_index=e_index,
            theta_off_index=threshold_index, halton_index=0,
            halton_e_to_e=NaN, halton_i_to_e=NaN, halton_e_to_i=NaN,
            halton_i_to_i=NaN, halton_theta_off=NaN,
            e_to_e=plane.e_to_e, i_to_e=plane.i_to_e, e_to_i,
            i_to_i=plane.i_to_i, theta_off))
    end
    return rows
end

function models_for_cell(config, cell)
    coupling = PointCoupling(e_to_e=cell.e_to_e, i_to_e=cell.i_to_e,
        e_to_i=cell.e_to_i, i_to_i=cell.i_to_i)
    return matched_point_models(excitatory=config.excitatory,
        inhibitory_control=config.inhibitory_control, failure_threshold=cell.theta_off,
        coupling=coupling)
end

function search_key(cell, condition)
    base = (condition, cell.e_to_e, cell.i_to_e, cell.e_to_i, cell.i_to_i)
    return condition == :control ? base : (base..., cell.theta_off)
end

function _append_rows(path, rows)
    isempty(rows) || CSV.write(path, rows; append=true)
    return nothing
end

function _initialize_artifacts(output)
    for (name, columns) in (
        ("samples.csv", SAMPLE_COLUMNS), ("searches.csv", SEARCH_COLUMNS),
        ("attempts.csv", ATTEMPT_COLUMNS), ("equilibria.csv", EQUILIBRIUM_COLUMNS),
        ("root_matches.csv", ROOT_MATCH_COLUMNS), ("candidates.csv", CANDIDATE_COLUMNS),
        ("branches.csv", BRANCH_COLUMNS),
        ("continuation_points.csv", CONTINUATION_POINT_COLUMNS),
        ("continuation_attempts.csv", CONTINUATION_ATTEMPT_COLUMNS),
        ("continuation_candidates.csv", CONTINUATION_CANDIDATE_COLUMNS),
    )
        Evidence.write_rows(joinpath(output, name), NamedTuple[], columns)
    end
end

function _search_summary(search_id, cell, condition, grid_points, result)
    found = isnothing(result) ? [] : result.equilibria
    count_stability(classification) = isnothing(result) ? -1 :
        count(eq -> eq.stability.classification == classification, found)
    return (; search_id, cell_id=cell.cell_id, stage=cell.stage,
        condition=string(condition), grid_points,
        status=isnothing(result) ? "execution_failed" : "completed",
        discovered_equilibria=isnothing(result) ? -1 : length(found),
        attracting_equilibria=count_stability(Attracting),
        saddle_equilibria=count_stability(Saddle),
        repelling_equilibria=count_stability(Repelling),
        stability_unresolved=count_stability(StabilityUnresolved),
        near_singular_equilibria=isnothing(result) ? -1 : count(eq -> eq.near_singular, found),
        unresolved_nearby_groups=isnothing(result) ? -1 : length(result.unresolved_nearby),
        attempts=isnothing(result) ? -1 : length(result.attempts),
        rejected_attempts=isnothing(result) ? -1 :
            count(attempt -> attempt.validation == RejectedCandidate, result.attempts),
        completeness=isnothing(result) ? "not_available" : string(result.completeness))
end

function _equilibrium_rows(search_id, condition, grid_points, model, result)
    isnothing(result) && return NamedTuple[]
    return [begin
        summary = Evidence.summary_equilibrium(search_id, index, equilibrium)
        observed = Map.equilibrium_observations(model, equilibrium)
        (; search_id, equilibrium=index, condition=string(condition), grid_points,
            E=summary.E, I=summary.I, residual_norm=summary.residual_norm,
            near_singular=summary.near_singular,
            representative_attempt=summary.representative_attempt,
            member_attempts=summary.member_attempts, stability=summary.stability,
            geometry=summary.geometry, eigenvalue_1_real=summary.eigenvalue_1_real,
            eigenvalue_1_imaginary=summary.eigenvalue_1_imaginary,
            eigenvalue_2_real=summary.eigenvalue_2_real,
            eigenvalue_2_imaginary=summary.eigenvalue_2_imaginary,
            spectral_abscissa=summary.spectral_abscissa, u_I=observed.u_I,
            F_I=observed.F_I, F_I_prime=observed.F_I_prime,
            inhibitory_response_midpoint=observed.inhibitory_response_midpoint,
            inhibitory_branch=observed.inhibitory_branch,
            descending_response_branch=observed.descending_response_branch)
    end for (index, equilibrium) in enumerate(result.equilibria)]
end

function _run_cached_search!(state, config, cell, condition, output)
    key = search_key(cell, condition)
    cacheable = condition == :control && cell.family == "plane"
    if cacheable && haskey(state.cache, key)
        return state.cache[key]
    end
    state.search_counter[] += 1
    search_id = "search_$(lpad(state.search_counter[], 6, '0'))"
    model = getproperty(models_for_cell(config, cell), condition)
    seeds = Map.deterministic_seeds(model, config.screen_grid_points)
    result = try
        find_equilibria(model; seeds, options=config.equilibrium_options,
            stability_options=config.stability_options)
    catch error
        error isa InterruptException && rethrow()
        Evidence.write_toml(joinpath(output, "contexts", "searches", search_id * ".toml"),
            Dict("status" => "execution_failed", "error" => Evidence.error_record(error),
                "requested_seeds" => seeds, "model" => model))
        nothing
    end
    if !isnothing(result)
        Evidence.write_toml(joinpath(output, "contexts", "searches", search_id * ".toml"),
            merge(Dict("search_id" => search_id, "cell_id" => cell.cell_id,
                "condition" => string(condition), "grid_points" => config.screen_grid_points,
                "model" => model, "requested_seeds" => seeds),
                Evidence.context_record(result)))
    end
    isnothing(result) && push!(state.failures, search_id)
    if !isnothing(result)
        _append_rows(joinpath(output, "attempts.csv"),
            [merge((; search_id), Base.structdiff(Evidence.summary_attempt(search_id, index, attempt),
                NamedTuple{(:context_id,)})) for (index, attempt) in enumerate(result.attempts)])
        _append_rows(joinpath(output, "equilibria.csv"),
            _equilibrium_rows(search_id, condition, config.screen_grid_points, model, result))
    end
    item = (; search_id, model, result)
    cacheable && (state.cache[key] = item)
    return item
end

function _cell_is_candidate(result)
    !isnothing(result) &&
        count(eq -> eq.stability.classification == Attracting, result.equilibria) >=
            CANDIDATE_MINIMUM
end

function _candidate_reference(item)
    result = item.result
    compact_result = (; equilibria=result.equilibria,
        unresolved_nearby=result.unresolved_nearby)
    return (; search_id=item.search_id, model=item.model, result=compact_result)
end

function screen_cells!(state, config, cells, output)
    _append_rows(joinpath(output, "samples.csv"), cells)
    for cell in cells
        state.cells[cell.cell_id] = cell
        for condition in CONDITIONS
            item = _run_cached_search!(state, config, cell, condition, output)
            summary = _search_summary(item.search_id, cell, condition,
                config.screen_grid_points, item.result)
            _append_rows(joinpath(output, "searches.csv"), [summary])
            if condition == :failure_of_inhibition && _cell_is_candidate(item.result)
                state.discovery_candidates[cell.cell_id] = cell
                state.candidate_results[cell.cell_id] = _candidate_reference(item)
            end
        end
    end
    return state
end

function refined_options(options, factor)
    return EquilibriumOptions(
        solver_abstol=options.solver_abstol * factor,
        solver_reltol=options.solver_reltol * factor,
        residual_atol=options.residual_atol * factor,
        domain_atol=options.domain_atol,
        dedup_atol=options.dedup_atol,
        singular_atol=options.singular_atol * factor,
        singular_rtol=options.singular_rtol * factor,
        maxiters=2options.maxiters,
    )
end

function refined_options(options::StabilityOptions, factor)
    return StabilityOptions(spectral_atol=options.spectral_atol * factor,
        spectral_rtol=options.spectral_rtol * factor)
end

function _independent_observation(model, equilibrium)
    T = eltype(equilibrium.state)
    residual = zeros(T, 2)
    balance_jacobian = zeros(T, 2, 2)
    ode_jacobian = zeros(T, 2, 2)
    point_balance!(residual, equilibrium.state, model, zero(T))
    point_balance_jacobian!(balance_jacobian, equilibrium.state, model, zero(T))
    point_jacobian!(ode_jacobian, equilibrium.state, model, zero(T))
    return (; state=Tuple(equilibrium.state),
        attracting=equilibrium.stability.classification == Attracting,
        near_singular=equilibrium.near_singular,
        residual_norm=maximum(abs, equilibrium.balance_residual),
        recomputed_residual_norm=maximum(abs, residual),
        balance_jacobian_delta=maximum(abs, balance_jacobian .- equilibrium.balance_jacobian),
        ode_jacobian_delta=maximum(abs, ode_jacobian .- equilibrium.stability.jacobian),
        spectral_abscissa=equilibrium.stability.spectral_abscissa)
end

_state_distance(first, second) = maximum(abs.(collect(first.state) .- collect(second.state)))

function unique_root_match(reference, references, candidates, tolerance)
    matches = findall(candidate -> _state_distance(reference, candidate) <= tolerance, candidates)
    length(matches) == 1 || return nothing
    candidate_index = only(matches)
    reciprocal = count(other -> _state_distance(other, candidates[candidate_index]) <= tolerance,
        references)
    return reciprocal == 1 ? candidate_index : nothing
end

function _minimum_separation(roots)
    length(roots) < 2 && return NaN
    return minimum(_state_distance(roots[first_index], roots[second_index])
        for first_index in 1:length(roots)-1 for second_index in first_index+1:length(roots))
end

function confirmation_assessment(reference_roots, refined_roots_by_grid,
    unresolved_by_grid, config)
    reference_attracting = findall(root -> root.attracting, reference_roots)
    tracks = NamedTuple[]
    for reference_index in reference_attracting
        matches = Dict{Int,Int}()
        for grid in config.confirmation_grid_points
            match = unique_root_match(reference_roots[reference_index], reference_roots,
                refined_roots_by_grid[grid], config.confirmation.match_atol)
            isnothing(match) || (matches[grid] = match)
        end
        length(matches) == length(config.confirmation_grid_points) || continue
        observations = [reference_roots[reference_index];
            [refined_roots_by_grid[grid][matches[grid]] for grid in config.confirmation_grid_points]]
        quality = all(root -> root.attracting && !root.near_singular &&
            root.recomputed_residual_norm <= config.equilibrium_options.residual_atol &&
            root.balance_jacobian_delta <= config.equilibrium_options.residual_atol &&
            root.ode_jacobian_delta <= config.equilibrium_options.residual_atol &&
            root.spectral_abscissa <= -config.confirmation.spectral_margin, observations)
        push!(tracks, (; track_id=length(tracks) + 1, reference_index, matches,
            observations, quality))
    end
    quality_tracks = filter(track -> track.quality, tracks)
    separations = Float64[]
    push!(separations, _minimum_separation(
        [reference_roots[track.reference_index] for track in quality_tracks]))
    for grid in config.confirmation_grid_points
        push!(separations, _minimum_separation(
            [refined_roots_by_grid[grid][track.matches[grid]] for track in quality_tracks]))
    end
    separation = any(isnan, separations) ? NaN : minimum(separations)
    separation_required = config.confirmation.separation_factor *
        config.equilibrium_options.dedup_atol
    reasons = String[]
    length(quality_tracks) >= CANDIDATE_MINIMUM ||
        push!(reasons, "fewer_than_four_uniquely_matched_attracting_roots")
    any(values(unresolved_by_grid)) && push!(reasons, "unresolved_nearby_roots")
    (!isnan(separation) && separation > separation_required) ||
        push!(reasons, "insufficient_root_separation")
    observations = [root for track in tracks for root in track.observations]
    metrics = isempty(observations) ?
        (; worst_spectral_abscissa=NaN, maximum_recomputed_residual=NaN,
            maximum_balance_jacobian_delta=NaN, maximum_ode_jacobian_delta=NaN) :
        (; worst_spectral_abscissa=maximum(root.spectral_abscissa for root in observations),
            maximum_recomputed_residual=maximum(root.recomputed_residual_norm for root in observations),
            maximum_balance_jacobian_delta=maximum(root.balance_jacobian_delta for root in observations),
            maximum_ode_jacobian_delta=maximum(root.ode_jacobian_delta for root in observations))
    return (; confirmed=isempty(reasons), tracks, quality_tracks,
        matched_attracting_roots=length(quality_tracks), minimum_separation=separation,
        reasons, metrics...)
end

function _confirmation_search(state, config, cell, condition, grid_points, candidate_id, output)
    state.confirmation_search_counter[] += 1
    search_id = "$(candidate_id)_$(condition)_grid$(grid_points)"
    model = getproperty(models_for_cell(config, cell), condition)
    equilibrium_options = refined_options(config.equilibrium_options,
        config.confirmation.tolerance_factor)
    stability_options = refined_options(config.stability_options,
        config.confirmation.tolerance_factor)
    seeds = Map.deterministic_seeds(model, grid_points)
    result = try
        find_equilibria(model; seeds, options=equilibrium_options,
            stability_options=stability_options)
    catch error
        error isa InterruptException && rethrow()
        Evidence.write_toml(joinpath(output, "contexts", "confirmations", search_id * ".toml"),
            Dict("status" => "execution_failed", "error" => Evidence.error_record(error),
                "requested_seeds" => seeds, "model" => model))
        return (; search_id, model, result=nothing)
    end
    Evidence.write_toml(joinpath(output, "contexts", "confirmations", search_id * ".toml"),
        merge(Dict("search_id" => search_id, "condition" => string(condition),
            "grid_points" => grid_points, "model" => model,
            "requested_seeds" => seeds), Evidence.context_record(result)))
    _append_rows(joinpath(output, "attempts.csv"),
        [merge((; search_id), Base.structdiff(Evidence.summary_attempt(search_id, index, attempt),
            NamedTuple{(:context_id,)})) for (index, attempt) in enumerate(result.attempts)])
    _append_rows(joinpath(output, "equilibria.csv"),
        _equilibrium_rows(search_id, condition, grid_points, model, result))
    return (; search_id, model, result)
end

function confirm_candidate!(state, config, cell, output)
    state.confirmation_counter[] += 1
    candidate_id = "candidate_$(lpad(state.confirmation_counter[], 4, '0'))"
    reference_item = state.candidate_results[cell.cell_id]
    reference_result = reference_item.result
    refined = Dict{Tuple{Symbol,Int},Any}()
    for grid in config.confirmation_grid_points, condition in CONDITIONS
        item = _confirmation_search(state, config, cell, condition, grid, candidate_id, output)
        refined[(condition, grid)] = item
        isnothing(item.result) && push!(state.failures, item.search_id)
        summary = _search_summary(item.search_id, cell, condition, grid, item.result)
        _append_rows(joinpath(output, "searches.csv"), [summary])
    end
    foi_complete = all(grid -> !isnothing(refined[(:failure_of_inhibition, grid)].result),
        config.confirmation_grid_points)
    control_complete = all(grid -> !isnothing(refined[(:control, grid)].result),
        config.confirmation_grid_points)
    if foi_complete
        reference_roots = [_independent_observation(reference_item.model, equilibrium)
            for equilibrium in reference_result.equilibria]
        refined_roots = Dict(grid => [_independent_observation(
            refined[(:failure_of_inhibition, grid)].model, equilibrium)
            for equilibrium in refined[(:failure_of_inhibition, grid)].result.equilibria]
            for grid in config.confirmation_grid_points)
        unresolved = Dict(0 => !isempty(reference_result.unresolved_nearby))
        for grid in config.confirmation_grid_points
            unresolved[grid] = !isempty(refined[(:failure_of_inhibition, grid)].result.unresolved_nearby)
        end
        assessment = confirmation_assessment(reference_roots, refined_roots, unresolved, config)
    else
        assessment = (; confirmed=false, tracks=NamedTuple[], quality_tracks=NamedTuple[],
            matched_attracting_roots=0, minimum_separation=NaN,
            reasons=["failure_of_inhibition_confirmation_failed"],
            worst_spectral_abscissa=NaN, maximum_recomputed_residual=NaN,
            maximum_balance_jacobian_delta=NaN, maximum_ode_jacobian_delta=NaN)
    end
    control_complete || push!(assessment.reasons, "matched_control_confirmation_failed")
    confirmed = assessment.confirmed && control_complete
    status = confirmed ? "confirmed" : "not_confirmed"
    attracting_count(item) = isnothing(item.result) ? -1 :
        count(eq -> eq.stability.classification == Attracting, item.result.equilibria)
    row = (; candidate_id, cell_id=cell.cell_id, e_to_e=cell.e_to_e,
        i_to_e=cell.i_to_e, e_to_i=cell.e_to_i, i_to_i=cell.i_to_i,
        theta_off=cell.theta_off,
        discovery_attracting=attracting_count(reference_item),
        grid_21_attracting=haskey(refined, (:failure_of_inhibition, 21)) ?
            attracting_count(refined[(:failure_of_inhibition, 21)]) : -1,
        grid_41_attracting=haskey(refined, (:failure_of_inhibition, 41)) ?
            attracting_count(refined[(:failure_of_inhibition, 41)]) : -1,
        matched_attracting_roots=assessment.matched_attracting_roots,
        minimum_separation=assessment.minimum_separation,
        worst_spectral_abscissa=assessment.worst_spectral_abscissa,
        maximum_recomputed_residual=assessment.maximum_recomputed_residual,
        maximum_balance_jacobian_delta=assessment.maximum_balance_jacobian_delta,
        maximum_ode_jacobian_delta=assessment.maximum_ode_jacobian_delta,
        confirmation_status=status, reasons=join(unique(assessment.reasons), ";"))
    _append_rows(joinpath(output, "candidates.csv"), [row])

    match_rows = NamedTuple[]
    if foi_complete
        grids = [config.screen_grid_points; config.confirmation_grid_points]
        items = Dict(config.screen_grid_points => reference_item,
            (grid => refined[(:failure_of_inhibition, grid)] for grid in config.confirmation_grid_points)...)
        for track in assessment.tracks
            root_indices = Dict(config.screen_grid_points => track.reference_index,
                (grid => track.matches[grid] for grid in config.confirmation_grid_points)...)
            for grid in grids
                item = items[grid]
                equilibrium_index = root_indices[grid]
                observation = _independent_observation(item.model,
                    item.result.equilibria[equilibrium_index])
                reference = _independent_observation(reference_item.model,
                    reference_result.equilibria[track.reference_index])
                push!(match_rows, (; candidate_id, root_track=track.track_id, grid_points=grid,
                    search_id=item.search_id, equilibrium=equilibrium_index,
                    match_distance=_state_distance(reference, observation),
                    E=observation.state[1], I=observation.state[2],
                    attracting=observation.attracting, near_singular=observation.near_singular,
                    residual_norm=observation.residual_norm,
                    recomputed_residual_norm=observation.recomputed_residual_norm,
                    balance_jacobian_delta=observation.balance_jacobian_delta,
                    ode_jacobian_delta=observation.ode_jacobian_delta,
                    spectral_abscissa=observation.spectral_abscissa,
                    quality_pass=track.quality))
            end
        end
    end
    _append_rows(joinpath(output, "root_matches.csv"), match_rows)
    return (; candidate_id, cell, row, assessment, refined, confirmed)
end

function _replace_axis(cell, axis, value)
    return merge(cell, NamedTuple{(axis,)}((value,)))
end

function _continuation_options(config, axis; smoke=false)
    table = config.continuation
    width = getproperty(config.axes, axis).upper - getproperty(config.axes, axis).lower
    return ContinuationOptions(initial_step=table.initial_step,
        minimum_step=table.minimum_step, maximum_step=table.maximum_step,
        max_steps=smoke ? table.smoke_max_steps : table.max_steps,
        max_corrector_iters=table.max_corrector_iters,
        corrector_atol=table.corrector_atol,
        parameter_difference_step=table.parameter_difference_step,
        parameter_scale=width, rank_atol=table.rank_atol, rank_rtol=table.rank_rtol)
end

function continue_candidate!(state, config, confirmation, output; smoke=false)
    confirmation.confirmed || return
    cell = confirmation.cell
    final_grid = last(config.confirmation_grid_points)
    final_item = confirmation.refined[(:failure_of_inhibition, final_grid)]
    equilibrium_options = refined_options(config.equilibrium_options,
        config.confirmation.tolerance_factor)
    stability_options = refined_options(config.stability_options,
        config.confirmation.tolerance_factor)
    for track in confirmation.assessment.quality_tracks
        track_index = track.track_id
        root_index = track.matches[final_grid]
        initial_state = final_item.result.equilibria[root_index].state
        for axis in SEARCH_AXES
            bounds_table = getproperty(config.axes, axis)
            bounds = (bounds_table.lower, bounds_table.upper)
            initial_parameter = getproperty(cell, axis)
            factory = parameter -> models_for_cell(config,
                _replace_axis(cell, axis, parameter)).failure_of_inhibition
            branch_id = "$(confirmation.candidate_id)_root$(track_index)_$(axis)"
            try
                result = continue_equilibria(factory, initial_state, initial_parameter;
                    parameter_bounds=bounds, options=_continuation_options(config, axis; smoke),
                    equilibrium_options, stability_options)
                Evidence.write_toml(joinpath(output, "contexts", "continuations", branch_id * ".toml"),
                    Dict("candidate_id" => confirmation.candidate_id,
                        "root_track" => track_index, "axis" => string(axis),
                        "initial_parameter" => initial_parameter, "parameter_bounds" => bounds,
                        "initial_solve" => result.initial_solve,
                        "negative" => result.negative, "positive" => result.positive,
                        "options" => result.options, "completeness" => result.completeness))
                for branch in (result.negative, result.positive)
                    _append_rows(joinpath(output, "branches.csv"), [(;
                        candidate_id=confirmation.candidate_id, root_track=track_index,
                        axis=string(axis), direction=branch.direction,
                        points=length(branch.points), attempts=length(branch.attempts),
                        screening_candidates=length(branch.candidates),
                        termination=string(branch.termination))])
                    point_rows = [begin
                        point = branch.points[index]
                        (; candidate_id=confirmation.candidate_id, root_track=track_index,
                            axis=string(axis), direction=branch.direction, point=index,
                            parameter=point.parameter, E=point.state[1], I=point.state[2],
                            residual_norm=point.equilibrium_attempt.residual_norm,
                            tangent_parameter=point.tangent[3],
                            stability=string(point.stability.classification),
                            spectral_abscissa=point.stability.spectral_abscissa,
                            trace=point.stability.trace, determinant=point.stability.determinant)
                    end for index in eachindex(branch.points)]
                    _append_rows(joinpath(output, "continuation_points.csv"), point_rows)
                    attempt_rows = [(; candidate_id=confirmation.candidate_id,
                        root_track=track_index, axis=string(axis), direction=branch.direction,
                        attempt=index, step=attempt.step, iterations=attempt.iterations,
                        arclength_residual=attempt.arclength_residual,
                        status=string(attempt.status), accepted=attempt.accepted)
                        for (index, attempt) in enumerate(branch.attempts)]
                    _append_rows(joinpath(output, "continuation_attempts.csv"), attempt_rows)
                    screening_rows = [begin
                        first_point = branch.points[candidate.first_point]
                        second_point = branch.points[candidate.second_point]
                        (; candidate_id=confirmation.candidate_id, root_track=track_index,
                            axis=string(axis), direction=branch.direction,
                            kind=string(candidate.kind), first_point=candidate.first_point,
                            second_point=candidate.second_point,
                            lower=min(first_point.parameter, second_point.parameter),
                            upper=max(first_point.parameter, second_point.parameter))
                    end for candidate in branch.candidates]
                    _append_rows(joinpath(output, "continuation_candidates.csv"), screening_rows)
                end
            catch error
                error isa InterruptException && rethrow()
                push!(state.failures, branch_id)
                Evidence.write_toml(joinpath(output, "contexts", "continuations",
                    branch_id * "_failure.toml"), Evidence.error_record(error))
            end
        end
    end
end

function _archive_provenance(config_path, output, config, smoke)
    metadata = Evidence.archive_provenance(config_path, output)
    for name in ("run_coexistence_map.jl", "run_tetrastability_search.jl")
        relative = joinpath("scripts", name)
        destination = joinpath(output, "source", relative)
        cp(joinpath(REPOSITORY_ROOT, relative), destination)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
    merge!(metadata, Dict(
        "purpose" => "finite equilibrium tetrastability discovery and confirmation; no biological interpretation",
        "experiment" => "tetrastability_search", "smoke" => smoke,
        "fixed_model" => config.fixed, "search_axes" => string.(SEARCH_AXES),
        "baseline_drive" => [0.0, 0.0],
        "axis_bounds" => Dict(string(axis) => [getproperty(config.axes, axis).lower,
            getproperty(config.axes, axis).upper] for axis in SEARCH_AXES),
        "halton_bases" => collect(config.halton.bases),
        "halton_boundary_policy" => "unscrambled positive-index Halton points lie strictly inside continuous bounds",
        "initial_halton_points" => config.halton.initial_points,
        "maximum_halton_points" => config.halton.maximum_points,
        "candidate_rule" => "FoI search with at least four discovered locally attracting equilibria",
        "equilibrium_seed_policy" => "union of the default 5-by-5 grid with the configured 11-by-11 discovery grid; independent 21-by-21 and 41-by-41 confirmation grids",
        "confirmation_rule" => "at least four uniquely coordinate-matched attracting roots across independent 11, 21 and 41 searches with declared quality margins",
        "recomputation_scope" => "balance residuals and Jacobians are recomputed after search through the same package kernels; this detects artifact inconsistency, not a shared formula defect",
        "matched_control_policy" => "matched control retained at every cell and confirmation grid; it is not part of candidate eligibility",
        "continuation_policy" => "five independent one-parameter continuations from every confirmed FoI root; not multiparameter continuation",
        "completeness" => CompletenessNotCertified,
        "biological_interpretation" => "not_assigned",
        "absence_claim" => "not_permitted",
        "prevalence_claim" => "not_permitted",
        "interruption_policy" => "partial artifacts are retained for diagnosis but are not resumable; restart the protocol into a new empty output directory",
        "replay_from_artifact_directory" => "julia --project=source source/scripts/run_tetrastability_search.jl --config config.toml --output replay" * (smoke ? " --smoke" : ""),
        "artifact_schema" => Dict("version" => 1,
            "samples" => "one row per declared plane or Halton parameter cell",
            "searches" => "one row per cell, condition and seed-grid search, including failures",
            "attempts" => "all nonlinear attempts streamed once per unique search context",
            "equilibria" => "all discovered admissible roots with response and local-spectrum observations",
            "root_matches" => "coordinate tracks across discovery and confirmation grids",
            "candidates" => "every screen-positive FoI cell and its confirmation outcome",
            "continuation" => "all retained branch points, corrector attempts and fold/Hopf screening brackets"),
    ))
    return metadata
end

function _state()
    return (cache=Dict{Any,Any}(), search_counter=Ref(0), cells=Dict{String,Any}(),
        candidate_results=Dict{String,Any}(),
        discovery_candidates=Dict{String,Any}(), confirmation_counter=Ref(0),
        confirmation_search_counter=Ref(0),
        failures=String[])
end

"""Run the finite search. No candidate is a successful scientific outcome."""
function run_experiment(config_path::AbstractString, output_dir::AbstractString;
    smoke=false, require_canonical=true)
    config = load_config(config_path; require_canonical)
    output = abspath(output_dir)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    mkpath(joinpath(output, "contexts", "searches"))
    mkpath(joinpath(output, "contexts", "confirmations"))
    mkpath(joinpath(output, "contexts", "continuations"))
    _initialize_artifacts(output)
    metadata = _archive_provenance(config_path, output, config, smoke)
    state = _state()

    screen_cells!(state, config, plane_cells(config; smoke), output)
    initial_count = smoke ? config.smoke_halton_points : config.halton.initial_points
    screen_cells!(state, config, halton_cells(config, 1:initial_count; stage="initial"), output)
    extension_run = false
    extension = extension_indices(config, !isempty(state.discovery_candidates); smoke)
    if !isempty(extension)
        extension_run = true
        screen_cells!(state, config, halton_cells(config, extension; stage="extension"), output)
    end

    confirmed_count = 0
    for cell_id in sort!(collect(keys(state.discovery_candidates)))
        confirmation = confirm_candidate!(state, config,
            state.discovery_candidates[cell_id], output)
        confirmation.confirmed && (confirmed_count += 1)
        continue_candidate!(state, config, confirmation, output; smoke)
        delete!(state.candidate_results, cell_id)
    end
    metadata["execution_success"] = isempty(state.failures)
    metadata["failed_contexts"] = state.failures
    metadata["extension_run"] = extension_run
    metadata["extension_trigger"] = extension_run ?
        "no FoI screen-positive candidate in planes or Halton indices 1:$(config.halton.initial_points)" :
        "not triggered"
    metadata["last_halton_index"] = smoke ? initial_count :
        extension_run ? config.halton.maximum_points : config.halton.initial_points
    metadata["parameter_cells"] = length(state.cells)
    unique_searches = state.search_counter[] + state.confirmation_search_counter[]
    metadata["unique_search_contexts"] = unique_searches
    metadata["screen_positive_cells"] = length(state.discovery_candidates)
    metadata["confirmed_cells"] = confirmed_count
    metadata["scientific_outcome"] = confirmed_count > 0 ?
        "at least one sampled cell has at least four confirmed discovered locally attracting FoI equilibria" :
        "no sampled cell met the confirmation rule; this finite result is not an absence proof"
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success=isempty(state.failures), extension_run,
        parameter_cells=length(state.cells), unique_searches,
        screen_positive=length(state.discovery_candidates), confirmed=confirmed_count)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "tetrastability.toml")
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
            index += 1
            continue
        elseif option in ("--config", "--output")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            value = args[index + 1]
            startswith(value, "--") && throw(ArgumentError("$option requires a value"))
            option == "--config" ? (config_path = value) : (output = value)
            index += 2
            continue
        end
        throw(ArgumentError("unknown option: $option"))
    end
    isnothing(output) && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, output; smoke)
    println("Tetrastability search: $(result.parameter_cells) cells, " *
        "$(result.screen_positive) screen-positive, $(result.confirmed) confirmed; " *
        "execution_success=$(result.success)")
    return result.success ? 0 : 1
end

end


if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(TetrastabilityExperiment.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
