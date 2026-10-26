"""Tetrastability protocol functions included by run_tetrastability_search.jl."""
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
