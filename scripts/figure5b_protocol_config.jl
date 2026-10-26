"""Figure 5b protocol functions included by run_figure5b_protocol.jl."""
function number_vector(value, label; positive=false, nonnegative=false,
    minimum_length=1)
    value isa AbstractVector && length(value) >= minimum_length ||
        throw(ArgumentError("$label must contain at least $minimum_length numbers"))
    return [number(item, label; positive, nonnegative) for item in value]
end

function integer_vector(value, label; minimum_length=1)
    value isa AbstractVector && length(value) >= minimum_length ||
        throw(ArgumentError("$label must contain at least $minimum_length integers"))
    return [positive_integer(item, label) for item in value]
end

function pair(value, label; positive=false)
    values = number_vector(value, label; positive, minimum_length=2)
    length(values) == 2 || throw(ArgumentError("$label must contain exactly two numbers"))
    values[1] < values[2] || throw(ArgumentError("$label must be strictly increasing"))
    return Tuple(values)
end

function parse_options(table, constructor, keys, label; integer_keys=())
    require_exact_keys(table, keys, label)
    entries = Pair{Symbol,Any}[]
    for key in keys
        value = key in integer_keys ? positive_integer(table[key], "$label.$key") :
            number(table[key], "$label.$key"; nonnegative=true)
        push!(entries, Symbol(key) => value)
    end
    return constructor(; entries...)
end

function _canonical_check(config)
    config.candidate == (tau_e=7.8, excitatory_slope=5.0,
        inhibitory_slope=5.0, theta_e=1.5, theta_on=4.0, theta_off=8.0,
        e_to_e=19.0, i_to_e=13.0, e_to_i=19.0, i_to_i=6.0) ||
        throw(ArgumentError("candidate differs from the frozen Figure-5b anchor"))
    config.ratios == (manuscript=4.4, below_hopf=0.4, orbit_anchor=0.5,
        hopf_bracket=(0.4, 0.5), continuation_bounds=(0.2, 8.0)) ||
        throw(ArgumentError("ratio schedule differs from the canonical protocol"))
    config.root_grids == collect(ROOT_GRIDS) ||
        throw(ArgumentError("root grids differ from 11, 21 and 41"))
    config.seeding == (ratio_offsets=[0.002, 0.01],
        phase_fractions=[0.0, 0.25, 0.5, 0.75],
        amplitude_factors=[0.75, 1.0, 1.25],
        period_factors=[0.9, 1.0, 1.1]) ||
        throw(ArgumentError("Hopf seed schedule differs from the canonical protocol"))
    e = config.equilibrium_options
    (e.solver_abstol, e.solver_reltol, e.residual_atol, e.domain_atol,
        e.dedup_atol, e.singular_atol, e.singular_rtol, e.maxiters) ==
        (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-10, 1e-8, 100) ||
        throw(ArgumentError("equilibrium policy differs from the canonical protocol"))
    s = config.stability_options
    (s.spectral_atol, s.spectral_rtol) == (1e-10, 1e-8) ||
        throw(ArgumentError("stability policy differs from the canonical protocol"))
    t = config.topology_options
    (t.coordinate_match_atol, t.minimum_root_separation, t.residual_atol,
        t.jacobian_atol, t.slope_atol, t.spectral_margin) ==
        (1e-6, 1e-5, 1e-9, 1e-9, 1e-8, 1e-8) ||
        throw(ArgumentError("topology policy differs from the canonical protocol"))
    h = config.hopf_options
    (h.balance_residual_atol, h.diagonal_atol, h.determinant_atol,
        h.transversality_atol, h.lyapunov_atol, h.lyapunov_agreement_atol,
        h.lyapunov_agreement_rtol, h.fd_plateau_atol, h.fd_plateau_rtol,
        h.fd_steps) == (1e-10, 1e-10, 1e-10, 1e-8, 1e-7, 1e-4, 1e-4,
        1e-4, 1e-4, (5e-4, 7.5e-4, 1e-3)) ||
        throw(ArgumentError("Hopf diagnostic policy differs from the canonical protocol"))
    p = config.shooting_options
    (p.ode_abstol, p.ode_reltol, p.shooting_atol, p.validation_atol,
        p.amplitude_atol, p.phase_speed_atol, p.floquet_atol,
        p.refinement_factor, p.min_period, p.max_period, p.maxiters,
        p.ode_maxiters, p.samples) == (1e-10, 1e-10, 1e-9, 1e-6,
        1e-5, 1e-10, 1e-4, 0.1, 1e-3, 100.0, 30, 1_000_000, 257) ||
        throw(ArgumentError("periodic shooting policy differs from the canonical protocol"))
    c = config.continuation_options
    (c.initial_step, c.minimum_step, c.maximum_step, c.max_steps,
        c.max_corrector_iters, c.corrector_atol, c.parameter_difference_step,
        c.state_scales, c.log_period_scale, c.parameter_scale, c.rank_atol,
        c.rank_rtol, c.tangent_alignment_min) == (0.002, 1e-6, 0.05, 500,
        16, 1e-8, 1e-5, (0.1, 0.1), 1.0, 1.0, 1e-12, 1e-10, 0.25) ||
        throw(ArgumentError("periodic continuation policy differs from the canonical protocol"))
    i = config.independent
    (i.tighter_factor, i.tight_samples, i.trace_ratio_atol,
        i.trace_agreement_atol, i.trace_maxiters, i.jacobian_steps,
        i.jacobian_atol, i.poincare_steps, i.poincare_intervals,
        i.poincare_plateau_atol, i.multiplier_atol,
        i.orbit_equivalence_atol, i.period_rtol, i.minimum_phase_seeds,
        i.minimum_period_seeds, i.winding_center_atol, i.fit_offset_max,
        i.fit_minimum_points, i.fit_slope_rtol, i.fit_intercept_atol,
        i.fit_r_squared_minimum, i.hopf_radius_atol, i.hopf_period_rtol,
        i.hopf_target_offset, i.hopf_ratio_atol, i.fold_multiplier_atol,
        i.homoclinic_period_factor, i.homoclinic_saddle_distance_atol,
        i.boundary_state_atol) ==
        (0.1, 513, 1e-10, 1e-8, 80, [1e-5, 5e-6, 2.5e-6], 1e-6,
        [1e-5, 5e-6, 2.5e-6], 1024, 2e-3, 5e-3, 1e-5, 1e-5,
        2, 2, 1e-8, 0.05, 4, 0.25, 1e-4, 0.9, 0.02, 0.05,
        1e-5, 5e-4, 2e-2, 5.0, 2e-2, 1e-4) ||
        throw(ArgumentError("independent-validation policy differs from the canonical protocol"))
    config.smoke == (phase_fractions=[0.0, 0.5], amplitude_factors=[1.0],
        period_factors=[0.9, 1.1], continuation_steps=1,
        hopf_continuation_steps=8) ||
        throw(ArgumentError("smoke policy differs from the canonical protocol"))
    return config
end

"""Load and fail-closed validate the version-1 Figure-5b protocol."""
function load_config(path::AbstractString; require_canonical=true)
    raw = TOML.parsefile(path)
    require_exact_keys(raw, ("schema_version", "candidate", "ratios", "search",
        "equilibrium", "stability", "topology", "hopf", "seeding", "shooting",
        "continuation", "independent", "smoke"), "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("schema_version must be integer 1"))

    candidate_keys = ("tau_e", "excitatory_slope", "inhibitory_slope", "theta_e",
        "theta_on", "theta_off", "e_to_e", "i_to_e", "e_to_i", "i_to_i")
    candidate_table = require_exact_keys(raw["candidate"], candidate_keys, "candidate")
    candidate_values = [number(candidate_table[key], "candidate.$key";
        positive=key in ("tau_e", "excitatory_slope", "inhibitory_slope"),
        nonnegative=key in ("e_to_e", "i_to_e", "e_to_i", "i_to_i"))
        for key in candidate_keys]
    candidate = NamedTuple{Symbol.(candidate_keys)}(Tuple(candidate_values))
    candidate.theta_on < candidate.theta_off ||
        throw(ArgumentError("candidate.theta_on must precede theta_off"))

    ratios_table = require_exact_keys(raw["ratios"], ("manuscript", "below_hopf",
        "orbit_anchor", "hopf_bracket", "continuation_bounds"), "ratios")
    ratios = (;
        manuscript=number(ratios_table["manuscript"], "ratios.manuscript"; positive=true),
        below_hopf=number(ratios_table["below_hopf"], "ratios.below_hopf"; positive=true),
        orbit_anchor=number(ratios_table["orbit_anchor"], "ratios.orbit_anchor"; positive=true),
        hopf_bracket=pair(ratios_table["hopf_bracket"], "ratios.hopf_bracket"; positive=true),
        continuation_bounds=pair(ratios_table["continuation_bounds"],
            "ratios.continuation_bounds"; positive=true))
    ratios.hopf_bracket[1] <= ratios.below_hopf < ratios.orbit_anchor <=
        ratios.hopf_bracket[2] || throw(ArgumentError("Hopf bracket must contain protocol ratios"))
    ratios.continuation_bounds[1] < ratios.below_hopf < ratios.manuscript <
        ratios.continuation_bounds[2] || throw(ArgumentError("continuation bounds must contain ratios"))

    search_table = require_exact_keys(raw["search"], ("root_grids",), "search")
    root_grids = integer_vector(search_table["root_grids"], "search.root_grids";
        minimum_length=3)
    length(unique(root_grids)) == length(root_grids) && issorted(root_grids) ||
        throw(ArgumentError("root grids must be unique and sorted"))

    equilibrium_keys = ("solver_abstol", "solver_reltol", "residual_atol",
        "domain_atol", "dedup_atol", "singular_atol", "singular_rtol", "maxiters")
    equilibrium_options = parse_options(raw["equilibrium"], EquilibriumOptions,
        equilibrium_keys, "equilibrium"; integer_keys=("maxiters",))
    stability_keys = ("spectral_atol", "spectral_rtol")
    stability_options = parse_options(raw["stability"], StabilityOptions,
        stability_keys, "stability")
    topology_keys = ("coordinate_match_atol", "minimum_root_separation", "residual_atol",
        "jacobian_atol", "slope_atol", "spectral_margin")
    topology_options = parse_options(raw["topology"], Figure5bTopologyOptions,
        topology_keys, "topology")

    hopf_keys = ("balance_residual_atol", "diagonal_atol", "determinant_atol",
        "transversality_atol", "lyapunov_atol", "lyapunov_agreement_atol",
        "lyapunov_agreement_rtol", "fd_plateau_atol", "fd_plateau_rtol", "fd_steps")
    hopf_table = require_exact_keys(raw["hopf"], hopf_keys, "hopf")
    fd_steps = Tuple(number_vector(hopf_table["fd_steps"], "hopf.fd_steps";
        positive=true, minimum_length=3))
    hopf_options = HopfDiagnosticOptions(; fd_steps,
        (Symbol(key) => number(hopf_table[key], "hopf.$key"; positive=true)
            for key in hopf_keys if key != "fd_steps")...)

    seeding_table = require_exact_keys(raw["seeding"], ("ratio_offsets",
        "phase_fractions", "amplitude_factors", "period_factors"), "seeding")
    seeding = (;
        ratio_offsets=number_vector(seeding_table["ratio_offsets"],
            "seeding.ratio_offsets"; positive=true),
        phase_fractions=number_vector(seeding_table["phase_fractions"],
            "seeding.phase_fractions"; nonnegative=true),
        amplitude_factors=number_vector(seeding_table["amplitude_factors"],
            "seeding.amplitude_factors"; positive=true),
        period_factors=number_vector(seeding_table["period_factors"],
            "seeding.period_factors"; positive=true))
    all(value -> 0 <= value < 1, seeding.phase_fractions) ||
        throw(ArgumentError("phase fractions must lie in [0,1)"))
    allunique(seeding.phase_fractions) || throw(ArgumentError("phase fractions must be unique"))

    shooting_keys = ("ode_abstol", "ode_reltol", "shooting_atol", "validation_atol",
        "amplitude_atol", "phase_speed_atol", "floquet_atol", "refinement_factor",
        "min_period", "max_period", "maxiters", "ode_maxiters", "samples")
    shooting_options = parse_options(raw["shooting"], PeriodicOrbitOptions,
        shooting_keys, "shooting";
        integer_keys=("maxiters", "ode_maxiters", "samples"))

    continuation_keys = ("initial_step", "minimum_step", "maximum_step", "max_steps",
        "max_corrector_iters", "corrector_atol", "parameter_difference_step",
        "state_scales", "log_period_scale", "parameter_scale", "rank_atol", "rank_rtol",
        "tangent_alignment_min")
    continuation_table = require_exact_keys(raw["continuation"], continuation_keys,
        "continuation")
    # State scales are two positive coordinates, not ordered bounds.
    state_scales = Tuple(number_vector(continuation_table["state_scales"],
        "continuation.state_scales"; positive=true, minimum_length=2))
    length(state_scales) == 2 || throw(ArgumentError("state_scales must have length two"))
    continuation_options = PeriodicContinuationOptions(; state_scales,
        (Symbol(key) => (key in ("max_steps", "max_corrector_iters") ?
            positive_integer(continuation_table[key], "continuation.$key") :
            number(continuation_table[key], "continuation.$key"; nonnegative=true))
            for key in continuation_keys if key != "state_scales")...)

    independent_keys = ("tighter_factor", "tight_samples", "trace_ratio_atol",
        "trace_agreement_atol",
        "trace_maxiters", "jacobian_steps", "jacobian_atol", "poincare_steps",
        "poincare_intervals", "poincare_plateau_atol", "multiplier_atol",
        "orbit_equivalence_atol", "period_rtol", "minimum_phase_seeds",
        "minimum_period_seeds", "winding_center_atol", "fit_offset_max",
        "fit_minimum_points", "fit_slope_rtol", "fit_intercept_atol",
        "fit_r_squared_minimum", "hopf_radius_atol", "hopf_period_rtol",
        "hopf_target_offset", "hopf_ratio_atol", "fold_multiplier_atol",
        "homoclinic_period_factor", "homoclinic_saddle_distance_atol",
        "boundary_state_atol")
    independent_table = require_exact_keys(raw["independent"], independent_keys,
        "independent")
    independent = (;
        tighter_factor=number(independent_table["tighter_factor"],
            "independent.tighter_factor"; positive=true),
        tight_samples=positive_integer(independent_table["tight_samples"],
            "independent.tight_samples"),
        trace_ratio_atol=number(independent_table["trace_ratio_atol"],
            "independent.trace_ratio_atol"; positive=true),
        trace_agreement_atol=number(independent_table["trace_agreement_atol"],
            "independent.trace_agreement_atol"; positive=true),
        trace_maxiters=positive_integer(independent_table["trace_maxiters"],
            "independent.trace_maxiters"),
        jacobian_steps=number_vector(independent_table["jacobian_steps"],
            "independent.jacobian_steps"; positive=true, minimum_length=3),
        jacobian_atol=number(independent_table["jacobian_atol"],
            "independent.jacobian_atol"; positive=true),
        poincare_steps=number_vector(independent_table["poincare_steps"],
            "independent.poincare_steps"; positive=true, minimum_length=3),
        poincare_intervals=positive_integer(independent_table["poincare_intervals"],
            "independent.poincare_intervals"),
        poincare_plateau_atol=number(independent_table["poincare_plateau_atol"],
            "independent.poincare_plateau_atol"; positive=true),
        multiplier_atol=number(independent_table["multiplier_atol"],
            "independent.multiplier_atol"; positive=true),
        orbit_equivalence_atol=number(independent_table["orbit_equivalence_atol"],
            "independent.orbit_equivalence_atol"; positive=true),
        period_rtol=number(independent_table["period_rtol"],
            "independent.period_rtol"; positive=true),
        minimum_phase_seeds=positive_integer(independent_table["minimum_phase_seeds"],
            "independent.minimum_phase_seeds"),
        minimum_period_seeds=positive_integer(independent_table["minimum_period_seeds"],
            "independent.minimum_period_seeds"),
        winding_center_atol=number(independent_table["winding_center_atol"],
            "independent.winding_center_atol"; positive=true),
        fit_offset_max=number(independent_table["fit_offset_max"],
            "independent.fit_offset_max"; positive=true),
        fit_minimum_points=positive_integer(independent_table["fit_minimum_points"],
            "independent.fit_minimum_points"),
        fit_slope_rtol=number(independent_table["fit_slope_rtol"],
            "independent.fit_slope_rtol"; positive=true),
        fit_intercept_atol=number(independent_table["fit_intercept_atol"],
            "independent.fit_intercept_atol"; positive=true),
        fit_r_squared_minimum=number(independent_table["fit_r_squared_minimum"],
            "independent.fit_r_squared_minimum"; positive=true),
        hopf_radius_atol=number(independent_table["hopf_radius_atol"],
            "independent.hopf_radius_atol"; positive=true),
        hopf_period_rtol=number(independent_table["hopf_period_rtol"],
            "independent.hopf_period_rtol"; positive=true),
        hopf_target_offset=number(independent_table["hopf_target_offset"],
            "independent.hopf_target_offset"; positive=true),
        hopf_ratio_atol=number(independent_table["hopf_ratio_atol"],
            "independent.hopf_ratio_atol"; positive=true),
        fold_multiplier_atol=number(independent_table["fold_multiplier_atol"],
            "independent.fold_multiplier_atol"; positive=true),
        homoclinic_period_factor=number(independent_table["homoclinic_period_factor"],
            "independent.homoclinic_period_factor"; positive=true),
        homoclinic_saddle_distance_atol=number(
            independent_table["homoclinic_saddle_distance_atol"],
            "independent.homoclinic_saddle_distance_atol"; positive=true),
        boundary_state_atol=number(independent_table["boundary_state_atol"],
            "independent.boundary_state_atol"; positive=true))
    independent.tighter_factor < 1 || throw(ArgumentError("tighter_factor must be below one"))
    independent.tight_samples > shooting_options.samples ||
        throw(ArgumentError("tight_samples must exceed shooting samples"))
    independent.fit_r_squared_minimum <= 1 ||
        throw(ArgumentError("fit_r_squared_minimum must not exceed one"))
    independent.hopf_target_offset < independent.hopf_ratio_atol ||
        throw(ArgumentError("hopf_target_offset must be below hopf_ratio_atol"))

    smoke_table = require_exact_keys(raw["smoke"], ("phase_fractions",
        "amplitude_factors", "period_factors", "continuation_steps",
        "hopf_continuation_steps"), "smoke")
    smoke = (;
        phase_fractions=number_vector(smoke_table["phase_fractions"],
            "smoke.phase_fractions"; nonnegative=true),
        amplitude_factors=number_vector(smoke_table["amplitude_factors"],
            "smoke.amplitude_factors"; positive=true),
        period_factors=number_vector(smoke_table["period_factors"],
            "smoke.period_factors"; positive=true),
        continuation_steps=positive_integer(smoke_table["continuation_steps"],
            "smoke.continuation_steps"),
        hopf_continuation_steps=positive_integer(
            smoke_table["hopf_continuation_steps"],
            "smoke.hopf_continuation_steps"))
    all(value -> value in seeding.phase_fractions, smoke.phase_fractions) ||
        throw(ArgumentError("smoke phases must be a subset of the full schedule"))
    all(value -> value in seeding.amplitude_factors, smoke.amplitude_factors) ||
        throw(ArgumentError("smoke amplitudes must be a subset of the full schedule"))
    all(value -> value in seeding.period_factors, smoke.period_factors) ||
        throw(ArgumentError("smoke periods must be a subset of the full schedule"))

    config = (; raw, candidate, ratios, root_grids, equilibrium_options,
        stability_options, topology_options, hopf_options, seeding, shooting_options,
        continuation_options, independent, smoke)
    return require_canonical ? _canonical_check(config) : config
end

"""Construct the exact frozen candidate at a supplied timescale ratio."""
