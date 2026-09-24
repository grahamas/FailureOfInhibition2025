"""Clean, candidate-specific Figure-5b Hopf and periodic-orbit protocol."""
module Figure5bProtocol

using FailureOfInhibition2025
using LinearAlgebra: BLAS, det, dot, eigen, eigvals, norm
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode
import CSV
import SHA
import TOML

include("run_minimal_experiment.jl")
const Evidence = MinimalExperiment

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const ROOT_GRIDS = (11, 21, 41)

require_exact_keys(table, keys, label) = begin
    table isa AbstractDict || throw(ArgumentError("$label must be a table"))
    missing = filter(key -> !haskey(table, key), keys)
    isempty(missing) || throw(ArgumentError("$label is missing: $(join(missing, ", "))"))
    extra = setdiff(collect(Base.keys(table)), collect(keys))
    isempty(extra) || throw(ArgumentError("$label has unknown keys: $(join(sort!(extra), ", "))"))
    table
end

number(value, label; positive=false, nonnegative=false) =
    Evidence.finite_number(value, label; positive, nonnegative)
positive_integer(value, label) = Evidence.positive_integer(value, label)

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
function candidate_model(config, ratio)
    ratio = number(ratio, "ratio"; positive=true)
    c = config.candidate
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=c.tau_e,
            response=LogisticResponse(slope=c.excitatory_slope, threshold=c.theta_e)),
        inhibitory=PopulationParameters(timescale=c.tau_e * ratio,
            response=FailureOfInhibitionResponse(slope=c.inhibitory_slope,
                onset_threshold=c.theta_on, failure_threshold=c.theta_off)),
        coupling=PointCoupling(e_to_e=c.e_to_e, i_to_e=c.i_to_e,
            e_to_i=c.e_to_i, i_to_i=c.i_to_i), drive=NoDrive())
end

function refinement_seeds(model, grid_points)
    grid_points = positive_integer(grid_points, "grid_points")
    seeds = default_equilibrium_seeds(model)
    upper_e, upper_i = last(seeds)
    append!(seeds, [[e, i] for e in range(0.0, upper_e; length=grid_points)
        for i in range(0.0, upper_i; length=grid_points)])
    return unique!(seeds)
end

function equilibrium_refinements(config, ratio)
    model = candidate_model(config, ratio)
    return Tuple(find_equilibria(model; seeds=refinement_seeds(model, points),
        options=config.equilibrium_options, stability_options=config.stability_options)
        for points in config.root_grids)
end

function unique_root_mapping(reference, candidate, tolerance)
    length(reference) == length(candidate) || return nothing
    distances = [norm(left.state .- right.state) for left in reference, right in candidate]
    mapping = Int[]
    for row in axes(distances, 1)
        matches = findall(value -> value <= tolerance, @view distances[row, :])
        length(matches) == 1 || return nothing
        push!(mapping, only(matches))
    end
    length(unique(mapping)) == length(candidate) || return nothing
    return mapping
end

function below_hopf_assessment(searches, options)
    reasons = Symbol[]
    all(search -> length(search.equilibria) == 7, searches) ||
        push!(reasons, :root_count_mismatch)
    all(search -> isempty(search.unresolved_nearby), searches) ||
        push!(reasons, :unresolved_nearby_roots)
    for search in searches
        for root in search.equilibria
            root.near_singular && push!(reasons, :near_singular_root)
            maximum(abs, root.balance_residual) <= options.residual_atol ||
                push!(reasons, :root_residual_too_large)
            real_parts = real.(root.stability.eigenvalues)
            if root.stability.classification == Attracting
                all(value -> value < -options.spectral_margin, real_parts) ||
                    push!(reasons, :insufficient_spectral_margin)
            elseif root.stability.classification == Saddle
                minimum(real_parts) < -options.spectral_margin &&
                    maximum(real_parts) > options.spectral_margin ||
                    push!(reasons, :insufficient_spectral_margin)
            end
        end
        if length(search.equilibria) > 1
            separation = minimum(norm(search.equilibria[left].state .-
                search.equilibria[right].state)
                for left in 1:(length(search.equilibria) - 1)
                for right in (left + 1):length(search.equilibria))
            separation >= options.minimum_root_separation ||
                push!(reasons, :insufficient_root_separation)
        end
    end
    mappings = Any[nothing, nothing]
    if all(search -> length(search.equilibria) == 7, searches)
        mappings[1] = unique_root_mapping(searches[1].equilibria,
            searches[2].equilibria, options.coordinate_match_atol)
        mappings[2] = unique_root_mapping(searches[1].equilibria,
            searches[3].equilibria, options.coordinate_match_atol)
        any(isnothing, mappings) && push!(reasons, :ambiguous_root_matching)
        if all(mapping -> !isnothing(mapping), mappings)
            reference = searches[1].equilibria
            for (offset, mapping) in enumerate(mappings)
                candidate = searches[offset + 1].equilibria
                for track in eachindex(reference)
                    reference[track].stability.classification ==
                        candidate[mapping[track]].stability.classification ||
                        push!(reasons, :stability_track_mismatch)
                end
            end
        end
        for search in searches
            counts = (count(root -> root.stability.classification == Attracting,
                search.equilibria), count(root -> root.stability.classification == Saddle,
                search.equilibria))
            counts == (4, 3) || push!(reasons, :stability_pattern_mismatch)
        end
    end
    unique!(reasons)
    return (qualified=isempty(reasons), reasons, mappings)
end

function finite_difference_jacobian(model, state, step)
    step = number(step, "finite-difference step"; positive=true)
    jacobian = zeros(Float64, 2, 2)
    plus, minus = zeros(2), zeros(2)
    for column in 1:2
        left, right = copy(state), copy(state)
        left[column] -= step
        right[column] += step
        point_rhs!(minus, left, model, 0.0)
        point_rhs!(plus, right, model, 0.0)
        jacobian[:, column] .= (plus .- minus) ./ (2step)
    end
    return jacobian
end

function numerical_trace_zero(config, state)
    lower, upper = config.ratios.hopf_bracket
    step = last(config.independent.jacobian_steps)
    trace_at(ratio) = begin
        matrix = finite_difference_jacobian(candidate_model(config, ratio), state, step)
        matrix[1, 1] + matrix[2, 2]
    end
    lower_trace, upper_trace = trace_at(lower), trace_at(upper)
    lower_trace * upper_trace <= 0 || return (resolved=false, ratio=NaN,
        trace=NaN, iterations=0, rows=[(; iteration=0, lower, upper,
            lower_trace, upper_trace, midpoint=NaN, midpoint_trace=NaN)])
    rows = NamedTuple[]
    midpoint, midpoint_trace = NaN, NaN
    for iteration in 1:config.independent.trace_maxiters
        midpoint = (lower + upper) / 2
        midpoint_trace = trace_at(midpoint)
        push!(rows, (; iteration, lower, upper, lower_trace, upper_trace,
            midpoint, midpoint_trace))
        if abs(upper - lower) <= config.independent.trace_ratio_atol
            return (resolved=true, ratio=midpoint, trace=midpoint_trace,
                iterations=iteration, rows)
        end
        if lower_trace * midpoint_trace <= 0
            upper, upper_trace = midpoint, midpoint_trace
        else
            lower, lower_trace = midpoint, midpoint_trace
        end
    end
    return (resolved=false, ratio=midpoint, trace=midpoint_trace,
        iterations=config.independent.trace_maxiters, rows)
end

"""Return deterministically phased right/adjoint Hopf eigenvectors."""
function hopf_basis(model, state, frequency)
    matrix = zeros(2, 2)
    point_jacobian!(matrix, state, model, 0.0)
    decomposition = eigen(complex.(matrix))
    q_index = argmin(abs.(decomposition.values .- im * frequency))
    q = decomposition.vectors[:, q_index]
    q ./= norm(q)
    pivot = argmax(abs.(q))
    q .*= exp(-im * angle(q[pivot]))
    real(q[pivot]) < 0 && (q .*= -1)
    adjoint_decomposition = eigen(adjoint(complex.(matrix)))
    p_index = argmin(abs.(adjoint_decomposition.values .+ im * frequency))
    p0 = adjoint_decomposition.vectors[:, p_index]
    p = p0 / conj(dot(p0, q))
    return (q=ComplexF64.(q), p=ComplexF64.(p), matrix,
        eigenvalue=ComplexF64(decomposition.values[q_index]),
        normalization=ComplexF64(dot(p, q)))
end

function seed_schedule(config, diagnostics, basis, central_state; smoke=false)
    phases = smoke ? config.smoke.phase_fractions : config.seeding.phase_fractions
    amplitudes = smoke ? config.smoke.amplitude_factors : config.seeding.amplitude_factors
    periods = smoke ? config.smoke.period_factors : config.seeding.period_factors
    target_ratios = unique(vcat(diagnostics.critical_ratio .+ config.seeding.ratio_offsets,
        config.ratios.orbit_anchor))
    rows = NamedTuple[]
    attempt = 0
    for ratio in target_ratios, phase in phases, amplitude_factor in amplitudes,
            period_factor in periods
        attempt += 1
        modal_radius = sqrt(max(0.0,
            diagnostics.modal_radius_squared_slope * (ratio - diagnostics.critical_ratio))) *
            amplitude_factor
        displacement = 2 .* real.(basis.q .* (modal_radius * cis(2pi * phase)))
        state = Float64.(central_state .+ displacement)
        push!(rows, (; attempt_id="orbit_$(lpad(attempt, 4, '0'))", ratio, phase,
            amplitude_factor, period_factor, modal_radius,
            seed_E=state[1], seed_I=state[2],
            period_guess=diagnostics.linear_period * period_factor))
    end
    return rows
end

function cluster_validated_attempts(attempts, config)
    validated = filter(attempt -> attempt.result !== nothing &&
        attempt.result.validation == NumericallyValidatedPeriodicOrbit, attempts)
    clusters = Vector{Vector{Any}}()
    for attempt in validated
        assigned = false
        for cluster in clusters
            comparison = phase_invariant_orbit_equivalence(
                first(cluster).result, attempt.result;
                state_atol=config.independent.orbit_equivalence_atol,
                period_rtol=config.independent.period_rtol)
            if comparison.equivalent
                push!(cluster, attempt)
                assigned = true
                break
            end
        end
        assigned || push!(clusters, Any[attempt])
    end
    sort!(clusters; by=cluster -> (-length(cluster), first(cluster).attempt_id))
    return clusters
end

function qualifying_cluster(clusters, config)
    for cluster in clusters
        phases = unique(attempt.phase for attempt in cluster)
        periods = unique(attempt.period_factor for attempt in cluster)
        if length(phases) >= config.independent.minimum_phase_seeds &&
                length(periods) >= config.independent.minimum_period_seeds
            return cluster
        end
    end
    return nothing
end

function tighter_periodic_options(config)
    source = config.shooting_options
    factor = config.independent.tighter_factor
    return PeriodicOrbitOptions(
        ode_abstol=source.ode_abstol * factor,
        ode_reltol=source.ode_reltol * factor,
        shooting_atol=source.shooting_atol * factor,
        validation_atol=source.validation_atol * factor,
        amplitude_atol=source.amplitude_atol,
        phase_speed_atol=source.phase_speed_atol,
        floquet_atol=source.floquet_atol * factor,
        refinement_factor=source.refinement_factor,
        min_period=source.min_period, max_period=source.max_period,
        maxiters=source.maxiters, ode_maxiters=source.ode_maxiters,
        samples=config.independent.tight_samples)
end

function jacobian_crosscheck(model, states, config)
    state_values = states isa AbstractVector && length(states) == 2 &&
        all(value -> value isa Real, states) ? [states] : collect(states)
    rows = NamedTuple[]
    discrepancies = Float64[]
    analytical = zeros(2, 2)
    for (sample, state) in enumerate(state_values), step in config.independent.jacobian_steps
        point_jacobian!(analytical, state, model, 0.0)
        finite_difference = finite_difference_jacobian(model, state, step)
        discrepancy = maximum(abs, analytical .- finite_difference)
        push!(discrepancies, discrepancy)
        push!(rows, (; sample, step, discrepancy,
            analytical_11=analytical[1, 1], analytical_12=analytical[1, 2],
            analytical_21=analytical[2, 1], analytical_22=analytical[2, 2],
            fd_11=finite_difference[1, 1], fd_12=finite_difference[1, 2],
            fd_21=finite_difference[2, 1], fd_22=finite_difference[2, 2]))
    end
    return (resolved=all(isfinite, discrepancies) &&
        maximum(discrepancies) <= config.independent.jacobian_atol,
        maximum_discrepancy=maximum(discrepancies), rows)
end

function _section_return(model, state, period, section_normal, section_tangent,
    epsilon, intervals, options)
    initial = state .+ epsilon .* section_tangent
    all(value -> 0 <= value <= 1, initial) || return nothing
    problem = ODEProblem(point_rhs!, initial, (0.0, 1.5period), model;
        isoutofdomain=(u, _, _) -> any(value -> !isfinite(value) || value < 0 || value > 1, u))
    solution = solve(problem, Tsit5(); abstol=options.ode_abstol,
        reltol=options.ode_reltol, maxiters=options.ode_maxiters, dense=true,
        save_everystep=true, verbose=false)
    successful_retcode(solution) || return nothing
    section_value(time) = dot(section_normal, solution(time) .- state)
    times = range(0.5period, 1.5period; length=intervals + 1)
    left_time, left_value = first(times), section_value(first(times))
    rhs = zeros(2)
    for right_time in Iterators.drop(times, 1)
        right_value = section_value(right_time)
        if left_value <= 0 <= right_value || right_value <= 0 <= left_value
            lower, upper = left_time, right_time
            lower_value = left_value
            for _ in 1:60
                midpoint = (lower + upper) / 2
                midpoint_value = section_value(midpoint)
                if lower_value * midpoint_value <= 0
                    upper = midpoint
                else
                    lower, lower_value = midpoint, midpoint_value
                end
            end
            crossing_time = (lower + upper) / 2
            crossing = Vector{Float64}(solution(crossing_time))
            point_rhs!(rhs, crossing, model, crossing_time)
            if dot(section_normal, rhs) > 0
                return (coordinate=dot(section_tangent, crossing .- state),
                    time=crossing_time)
            end
        end
        left_time, left_value = right_time, right_value
    end
    return nothing
end

"""Independent finite-difference Poincare return multiplier."""
function poincare_multiplier(result, config)
    result.validation == NumericallyValidatedPeriodicOrbit ||
        return (resolved=false, multiplier=NaN, values=Float64[], rows=NamedTuple[])
    model = result.model
    state = result.initial_state
    velocity = zeros(2)
    point_rhs!(velocity, state, model, 0.0)
    norm(velocity) > config.shooting_options.phase_speed_atol ||
        return (resolved=false, multiplier=NaN, values=Float64[], rows=NamedTuple[])
    normal = velocity ./ norm(velocity)
    tangent = [-normal[2], normal[1]]
    values, rows = Float64[], NamedTuple[]
    for step in config.independent.poincare_steps
        positive = _section_return(model, state, result.period, normal, tangent,
            step, config.independent.poincare_intervals, config.shooting_options)
        negative = _section_return(model, state, result.period, normal, tangent,
            -step, config.independent.poincare_intervals, config.shooting_options)
        if positive === nothing || negative === nothing
            push!(rows, (; step, positive_return=NaN, negative_return=NaN,
                positive_time=NaN, negative_time=NaN, multiplier=NaN))
            continue
        end
        multiplier = (positive.coordinate - negative.coordinate) / (2step)
        push!(values, multiplier)
        push!(rows, (; step, positive_return=positive.coordinate,
            negative_return=negative.coordinate, positive_time=positive.time,
            negative_time=negative.time, multiplier))
    end
    resolved = length(values) == length(config.independent.poincare_steps) &&
        maximum(values) - minimum(values) <= config.independent.poincare_plateau_atol
    return (resolved, multiplier=resolved ? sum(values) / length(values) : NaN,
        values, rows)
end

function orbit_acceptance_reasons(; numerically_validated, attracting, primitive,
    equivalent, jacobian_resolved, divergence_resolved, poincare_resolved,
    poincare_difference, multiplier_atol, windings, central_index)
    reasons = Symbol[]
    numerically_validated || push!(reasons, :tight_replay_unresolved)
    attracting || push!(reasons, :orbit_not_attracting)
    primitive || push!(reasons, :nonprimitive_period)
    equivalent || push!(reasons, :tight_replay_mismatch)
    jacobian_resolved || push!(reasons, :jacobian_mismatch)
    divergence_resolved || push!(reasons, :divergence_multiplier_mismatch)
    poincare_resolved || push!(reasons, :poincare_unresolved)
    poincare_resolved && poincare_difference > multiplier_atol &&
        push!(reasons, :poincare_multiplier_mismatch)
    if length(windings) == 0 || !all(row -> row.resolved, windings)
        push!(reasons, :winding_unresolved)
    elseif central_index === nothing || !(1 <= central_index <= length(windings))
        push!(reasons, :central_equilibrium_unresolved)
    else
        abs(windings[central_index].winding) == 1 ||
            push!(reasons, :wrong_central_winding)
        all(index -> index == central_index || windings[index].winding == 0,
            eachindex(windings)) || push!(reasons, :encloses_other_equilibrium)
    end
    return unique(reasons)
end

function validate_orbit(result, baseline, roots, config)
    reasons = Symbol[]
    if result.validation != NumericallyValidatedPeriodicOrbit
        push!(reasons, :tight_replay_unresolved)
        result.stability == PeriodicOrbitAttracting || push!(reasons, :orbit_not_attracting)
        return (accepted=false, reasons, primitive=nothing,
            equivalence=nothing, divergence=nothing, jacobian=nothing,
            poincare=nothing, windings=NamedTuple[])
    end
    primitive = periodic_orbit_primitive_check(result)
    equivalence = phase_invariant_orbit_equivalence(baseline, result;
        state_atol=config.independent.orbit_equivalence_atol,
        period_rtol=config.independent.period_rtol)
    jacobian = jacobian_crosscheck(result.model, result.states[1:end-1], config)
    jacobian_callback = (matrix, state, model) -> point_jacobian!(matrix, state, model, 0.0)
    divergence = periodic_orbit_divergence_check(result, jacobian_callback, result.model;
        atol=config.independent.multiplier_atol)
    poincare = poincare_multiplier(result, config)
    windings = NamedTuple[]
    for (index, root) in enumerate(roots)
        winding = periodic_orbit_winding(result, root.state;
            center_atol=config.independent.winding_center_atol)
        push!(windings, (; equilibrium=index, E=root.state[1], I=root.state[2],
            resolved=winding.resolved, reason=string(winding.reason),
            winding=isnothing(winding.winding) ? 0 : winding.winding,
            minimum_distance=winding.minimum_distance, samples=winding.samples))
    end
    repellers = filter(index -> roots[index].stability.classification == Repelling,
        eachindex(roots))
    central = length(repellers) == 1 ? only(repellers) : nothing
    reasons = orbit_acceptance_reasons(
        numerically_validated=true,
        attracting=result.stability == PeriodicOrbitAttracting,
        primitive=primitive.primitive, equivalent=equivalence.equivalent,
        jacobian_resolved=jacobian.resolved,
        divergence_resolved=divergence.resolved,
        poincare_resolved=poincare.resolved,
        poincare_difference=poincare.resolved ?
            abs(poincare.multiplier - real(result.transverse_multiplier)) : Inf,
        multiplier_atol=config.independent.multiplier_atol,
        windings=windings, central_index=central)
    return (accepted=isempty(reasons), reasons, primitive, equivalence,
        divergence, jacobian, poincare, windings)
end

function write_tight_validation_artifact(output, selected, result, validation,
    center, p; stage_error=nothing)
    if result === nothing || validation === nothing
        Evidence.write_toml(joinpath(output, "tight_validation.toml"), Dict(
            "status" => stage_error === nothing ? "not_run" : "exception",
            "selected_attempt" => selected === nothing ? "not_available" : selected,
            "error" => stage_error === nothing ? "not_available" : stage_error))
        return nothing
    end
    validated = result.validation == NumericallyValidatedPeriodicOrbit
    half_ranges = validated ? periodic_orbit_half_ranges(result) : [NaN, NaN]
    distances = validated ? periodic_orbit_distances(result, center) :
        (minimum=NaN, rms=NaN, maximum=NaN)
    area = validated ? periodic_orbit_signed_area(result) : NaN
    radius = validated ? modal_radius(result, center, p) : NaN
    Evidence.write_toml(joinpath(output, "tight_validation.toml"), Dict(
        "status" => "completed", "selected_attempt" => selected,
        "accepted" => validation.accepted,
        "reasons" => string.(validation.reasons),
        "validation" => result.validation, "stability" => result.stability,
        "initial_state" => result.initial_state, "period" => result.period,
        "component_half_ranges" => half_ranges,
        "center_distances" => distances, "signed_area" => area,
        "modal_radius" => radius,
        "closure_residual" => result.closure_residual,
        "phase_residual" => result.phase_residual,
        "equation_residual" => result.equation_residual,
        "refinement_difference" => result.refinement_difference,
        "integration_success" => result.integration_success,
        "floquet_multipliers" => result.floquet_multipliers,
        "transverse_multiplier" => result.transverse_multiplier,
        "monodromy" => result.monodromy,
        "primitive_period" => validation.primitive,
        "phase_equivalence" => validation.equivalence,
        "divergence_check" => validation.divergence,
        "jacobian_check" => validation.jacobian === nothing ? "not_available" :
            Dict("resolved" => validation.jacobian.resolved,
                "maximum_discrepancy" => validation.jacobian.maximum_discrepancy),
        "poincare_check" => validation.poincare,
        "windings" => validation.windings))
    return nothing
end

function modal_radius(result, center, p)
    radii = [abs(dot(p, state .- center)) for state in result.states[1:end-1]]
    return sqrt(sum(abs2, radii) / length(radii))
end

branch_divergence_observation(point) = (
    divergence_resolved=point.divergence_check.resolved,
    divergence_multiplier=point.divergence_check.divergence_multiplier,
    divergence_discrepancy=point.divergence_check.discrepancy)

branch_primitive_observation(point) = (
    primitive=point.primitive_period_check.primitive,
    primitive_alias=point.primitive_period_check.alias_divisor)

function branch_evidence_resolved(row)
    return hasproperty(row, :validated) && row.validated === true &&
        hasproperty(row, :attracting) && row.attracting === true &&
        hasproperty(row, :divergence_resolved) && row.divergence_resolved === true &&
        hasproperty(row, :primitive) && row.primitive === true
end

function branch_observations(continuation, center, p, roots, config)
    observations = Dict{Symbol,Vector{NamedTuple}}()
    for (branch_name, branch) in ((:negative, continuation.negative),
            (:positive, continuation.positive))
        rows = NamedTuple[]
        for (index, point) in enumerate(branch.points)
            radius = modal_radius(point.orbit, center, p)
            saddle_distances = [periodic_orbit_distances(point.orbit, root.state).minimum
                for root in roots if root.stability.classification == Saddle]
            boundary_distance = minimum(vcat(
                [minimum(state) for state in point.orbit.states],
                [1 - maximum(state) for state in point.orbit.states]))
            divergence = branch_divergence_observation(point)
            primitive = branch_primitive_observation(point)
            push!(rows, (; branch=string(branch_name), point=index,
                ratio=point.parameter, modal_radius=radius,
                radius_squared=radius^2, period=point.period,
                transverse_multiplier=real(point.orbit.transverse_multiplier),
                validated=point.orbit.validation == NumericallyValidatedPeriodicOrbit,
                attracting=point.orbit.stability == PeriodicOrbitAttracting,
                divergence...,
                primitive...,
                saddle_distance=isempty(saddle_distances) ? Inf : minimum(saddle_distances),
                boundary_distance))
        end
        observations[branch_name] = rows
    end
    return observations
end

function endpoint_classification(termination, observations, reversals,
    critical_ratio, linear_period, expected_boundary, config)
    reasons = Symbol[]
    isempty(observations) && return (classification=:unresolved,
        reasons=[:no_accepted_points], hopf_compatible=false,
        fold_compatible=false, homoclinic_compatible=false,
        state_boundary_compatible=false)
    last_point = last(observations)
    monotone_toward_hopf = all(observations[index + 1].ratio < observations[index].ratio
        for index in 1:(length(observations) - 1))
    no_reversal = isempty(reversals)
    validated = all(row -> hasproperty(row, :validated) && row.validated === true,
        observations)
    attracting = all(row -> hasproperty(row, :attracting) && row.attracting === true,
        observations)
    divergence_resolved = all(row -> hasproperty(row, :divergence_resolved) &&
        row.divergence_resolved === true, observations)
    primitive = all(row -> hasproperty(row, :primitive) && row.primitive === true,
        observations)
    evidence_resolved = all(branch_evidence_resolved, observations)
    boundary_matches = expected_boundary !== nothing &&
        abs(last_point.ratio - expected_boundary) <= config.independent.hopf_ratio_atol
    hopf_compatible = termination == :parameter_boundary && no_reversal &&
        monotone_toward_hopf && evidence_resolved && boundary_matches &&
        0 < last_point.ratio - critical_ratio <= config.independent.hopf_ratio_atol &&
        last_point.modal_radius <= config.independent.hopf_radius_atol &&
        abs(last_point.period - linear_period) / linear_period <=
            config.independent.hopf_period_rtol

    real_terminal_status = termination in (:parameter_boundary, :minimum_step, :step_limit)
    terminal_reversal = length(observations) - 1
    fold_compatible = real_terminal_status && length(observations) >= 3 &&
        terminal_reversal in reversals &&
        all(row -> branch_evidence_resolved(row) &&
            abs(row.transverse_multiplier - 1) <=
            config.independent.fold_multiplier_atol, observations[(end - 2):end])
    terminal_count = min(3, length(observations))
    terminal_segment = observations[(end - terminal_count + 1):end]
    terminal_validated = terminal_count >= 3 &&
        all(branch_evidence_resolved, terminal_segment)
    terminal_ratio_decreasing = terminal_validated &&
        all(terminal_segment[index + 1].ratio < terminal_segment[index].ratio
            for index in 1:(terminal_count - 1))
    terminal_ratio_increasing = terminal_validated &&
        all(terminal_segment[index + 1].ratio > terminal_segment[index].ratio
            for index in 1:(terminal_count - 1))
    terminal_monotone = terminal_ratio_decreasing || terminal_ratio_increasing
    robust_period_growth = terminal_monotone &&
        last_point.period >= config.independent.homoclinic_period_factor *
            first(observations).period &&
        all(terminal_segment[index + 1].period > terminal_segment[index].period
            for index in 1:(terminal_count - 1))
    robust_saddle_approach = terminal_monotone &&
        last_point.saddle_distance <=
            config.independent.homoclinic_saddle_distance_atol &&
        all(terminal_segment[index + 1].saddle_distance <
            terminal_segment[index].saddle_distance
            for index in 1:(terminal_count - 1))
    homoclinic_compatible = real_terminal_status && robust_period_growth &&
        robust_saddle_approach
    state_boundary_compatible = real_terminal_status && terminal_monotone &&
        last_point.boundary_distance <= config.independent.boundary_state_atol &&
        all(terminal_segment[index + 1].boundary_distance <=
            terminal_segment[index].boundary_distance
            for index in 1:(terminal_count - 1))
    boundary = termination == :parameter_boundary
    classification = hopf_compatible ? :hopf_compatible :
        fold_compatible ? :fold_compatible :
        homoclinic_compatible ? :homoclinic_or_heteroclinic_compatible :
        state_boundary_compatible ? :state_boundary_compatible :
        boundary ? :boundary : :unresolved
    !monotone_toward_hopf && push!(reasons, :not_monotone_toward_hopf)
    !no_reversal && !fold_compatible && push!(reasons, :unresolved_parameter_reversal)
    !validated && push!(reasons, :unvalidated_branch_point)
    !attracting && push!(reasons, :nonattracting_branch_orbit)
    !divergence_resolved && push!(reasons, :divergence_multiplier_unresolved)
    !primitive && push!(reasons, :nonprimitive_branch_orbit)
    termination in (:minimum_step, :step_limit, :initial_tangent_unresolved,
        :initial_phase_unresolved, :initial_orbit_unresolved) &&
        push!(reasons, :unsuccessful_termination)
    classification == :boundary && push!(reasons, :nonhopf_parameter_boundary)
    classification == :unresolved && isempty(reasons) && push!(reasons, :endpoint_unresolved)
    return (; classification, reasons=unique(reasons), hopf_compatible,
        fold_compatible, homoclinic_compatible, state_boundary_compatible)
end

function branch_fit(continuation, observations, critical_ratio, config)
    eligible = NamedTuple[]
    for (branch_name, branch) in ((:negative, continuation.negative),
            (:positive, continuation.positive))
        rows = observations[branch_name]
        cutoff = isempty(branch.parameter_reversals) ? length(rows) :
            max(0, first(branch.parameter_reversals) - 1)
        segment = rows[1:cutoff]
        monotone = length(segment) >= 2 && all(segment[index + 1].ratio <
            segment[index].ratio for index in 1:(length(segment) - 1))
        candidates = monotone ? filter(row -> 0 < row.ratio - critical_ratio <=
            config.independent.fit_offset_max, segment) : NamedTuple[]
        length(candidates) >= config.independent.fit_minimum_points &&
            push!(eligible, (; branch_name, candidates,
                endpoint_offset=last(candidates).ratio - critical_ratio))
    end
    if isempty(eligible)
        return (resolved=false, branch="not_available", intercept=NaN, slope=NaN,
            r_squared=NaN, points=0, rows=NamedTuple[])
    end
    selected = first(sort(eligible; by=item -> item.endpoint_offset))
    candidates = selected.candidates
    x = [row.ratio - critical_ratio for row in candidates]
    y = [row.radius_squared for row in candidates]
    design = hcat(ones(length(x)), x)
    coefficients = design \ y
    fitted = design * coefficients
    total = sum(abs2, y .- sum(y) / length(y))
    residual = sum(abs2, y .- fitted)
    r_squared = iszero(total) ? NaN : 1 - residual / total
    return (resolved=all(isfinite, coefficients) && isfinite(r_squared),
        branch=string(selected.branch_name), intercept=coefficients[1],
        slope=coefficients[2], r_squared,
        points=length(candidates), rows=candidates)
end

function fit_acceptance_assessment(fit, endpoints, diagnostics, config)
    reasons = Symbol[]
    if fit === nothing || !fit.resolved
        return (accepted=false, branch=nothing, endpoint=nothing,
            nearest=nothing, reasons=[:fit_unresolved])
    end
    branch = try
        Symbol(fit.branch)
    catch
        nothing
    end
    endpoint = branch === nothing ? nothing : get(endpoints, branch, nothing)
    branch === nothing && push!(reasons, :fit_branch_invalid)
    endpoint === nothing && push!(reasons, :fit_endpoint_branch_missing)
    endpoint !== nothing && fit.branch != string(branch) &&
        push!(reasons, :fit_endpoint_branch_mismatch)
    nearest = isempty(fit.rows) ? nothing : last(fit.rows)
    fit_evidence_resolved = !isempty(fit.rows) &&
        all(branch_evidence_resolved, fit.rows)
    if !fit_evidence_resolved
        all(row -> hasproperty(row, :validated) && row.validated === true,
            fit.rows) || push!(reasons, :fit_orbit_validation_unresolved)
        all(row -> hasproperty(row, :attracting) && row.attracting === true,
            fit.rows) || push!(reasons, :fit_nonattracting_orbit)
        all(row -> hasproperty(row, :divergence_resolved) &&
            row.divergence_resolved === true, fit.rows) ||
            push!(reasons, :fit_divergence_unresolved)
        all(row -> hasproperty(row, :primitive) && row.primitive === true,
            fit.rows) || push!(reasons, :fit_nonprimitive_orbit)
    end
    fit.points >= config.independent.fit_minimum_points ||
        push!(reasons, :insufficient_fit_points)
    abs(fit.intercept) <= config.independent.fit_intercept_atol ||
        push!(reasons, :fit_intercept_too_large)
    fit.r_squared >= config.independent.fit_r_squared_minimum ||
        push!(reasons, :fit_r_squared_too_small)
    isapprox(fit.slope, diagnostics.modal_radius_squared_slope;
        rtol=config.independent.fit_slope_rtol, atol=0.0) ||
        push!(reasons, :fit_slope_disagreement)
    endpoint !== nothing && endpoint.hopf_compatible ||
        push!(reasons, :endpoint_not_hopf_compatible)
    if nearest === nothing
        push!(reasons, :nearest_hopf_point_missing)
    else
        0 < nearest.ratio - diagnostics.critical_ratio <=
            config.independent.hopf_ratio_atol ||
            push!(reasons, :endpoint_ratio_not_near_hopf)
        nearest.modal_radius <= config.independent.hopf_radius_atol ||
            push!(reasons, :endpoint_radius_not_near_zero)
        abs(nearest.period - diagnostics.linear_period) /
            diagnostics.linear_period <= config.independent.hopf_period_rtol ||
            push!(reasons, :endpoint_period_disagreement)
        nearest.modal_radius == minimum(row.modal_radius for row in fit.rows) ||
            push!(reasons, :radius_not_decreasing_to_endpoint)
        all(fit.rows[index + 1].modal_radius < fit.rows[index].modal_radius
            for index in 1:(length(fit.rows) - 1)) ||
            push!(reasons, :radius_not_strictly_decreasing_to_endpoint)
    end
    unique!(reasons)
    return (accepted=isempty(reasons), branch, endpoint, nearest, reasons)
end

function write_rows(path, rows, columns)
    if isempty(rows)
        names = Tuple(columns)
        CSV.write(path, NamedTuple{names}(Tuple(String[] for _ in names)))
    else
        CSV.write(path, rows)
    end
    return path
end

function write_orbit(path, result)
    CSV.write(path, (; time=result.times,
        E=[state[1] for state in result.states],
        I=[state[2] for state in result.states]))
end

function normalize_accepted_revision(value)
    value isa AbstractString ||
        throw(ArgumentError("accepted revision must be a 40-character hexadecimal commit"))
    revision = lowercase(String(value))
    occursin(r"^[0-9a-f]{40}$", revision) ||
        throw(ArgumentError("accepted revision must be a 40-character hexadecimal commit"))
    return revision
end

function provenance_assessment(expected_revision, actual_revision, status_porcelain,
    head_name)
    expected = expected_revision === nothing ? nothing :
        normalize_accepted_revision(expected_revision)
    actual = actual_revision isa AbstractString ? lowercase(String(actual_revision)) :
        string(actual_revision)
    detached = head_name == "HEAD"
    clean = status_porcelain == ""
    actual_resolved = occursin(r"^[0-9a-f]{40}$", actual)
    revision_match = expected !== nothing && actual_resolved && expected == actual
    reasons = Symbol[]
    expected === nothing && push!(reasons, :accepted_revision_missing)
    actual_resolved || push!(reasons, :actual_revision_unresolved)
    revision_match || push!(reasons, :revision_mismatch)
    detached || push!(reasons, :head_not_detached)
    clean || push!(reasons, :working_tree_dirty)
    unique!(reasons)
    return (eligible=isempty(reasons), expected_revision=expected,
        actual_revision=actual, detached, head_name=String(head_name), clean,
        status_porcelain=String(status_porcelain), revision_match, reasons)
end

function capture_provenance(expected_revision)
    actual_revision = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    status_porcelain = Evidence.git_output(REPOSITORY_ROOT, "status", "--porcelain=v1",
        "--untracked-files=all")
    head_name = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    return provenance_assessment(expected_revision, actual_revision, status_porcelain,
        head_name)
end

scientific_evidence_eligible(smoke, replay_mode, provenance_eligible,
    execution_success, failure_injection, canonical_config) =
    !smoke && !replay_mode && provenance_eligible && execution_success &&
    failure_injection === nothing && canonical_config

supercritical_hopf_candidate(numerical_four_attractor,
    manuscript_topology_qualified, diagnostics, hopf_agreement,
    below_hopf_qualified, central_below_attracting, fit_acceptance,
    stage_errors) =
    numerical_four_attractor && manuscript_topology_qualified &&
    diagnostics.resolved &&
    diagnostics.classification == :supercritical_candidate && hopf_agreement &&
    below_hopf_qualified && central_below_attracting && fit_acceptance &&
    isempty(stage_errors)

function validate_replay_parent(parent_directory, config_path)
    parent = realpath(parent_directory)
    isdir(parent) || throw(ArgumentError("replay parent must be an artifact directory"))
    checksums_path = joinpath(parent, "checksums.toml")
    metadata_path = joinpath(parent, "metadata.toml")
    isfile(checksums_path) && isfile(metadata_path) || throw(ArgumentError(
        "replay parent must contain checksums.toml and metadata.toml"))
    manifest = TOML.parsefile(checksums_path)
    manifest["algorithm"] == "SHA-256" ||
        throw(ArgumentError("replay parent checksum algorithm must be SHA-256"))
    files = manifest["files"]
    expected_paths = Set(String.(keys(files)))
    actual_paths = Set(relpath(joinpath(root, filename), parent)
        for (root, _, names) in walkdir(parent) for filename in names
        if relpath(joinpath(root, filename), parent) != "checksums.toml")
    expected_paths == actual_paths || throw(ArgumentError(
        "replay parent checksum manifest does not exactly cover the artifact"))
    for relative in sort!(collect(expected_paths))
        Evidence.file_hash(joinpath(parent, relative)) == files[relative] ||
            throw(ArgumentError("replay parent checksum mismatch: $relative"))
    end
    metadata = TOML.parsefile(metadata_path)
    archived_config = joinpath(parent, "config.toml")
    realpath(config_path) == archived_config || throw(ArgumentError(
        "replay config must be the parent artifact config.toml"))
    Evidence.file_hash(archived_config) == metadata["config_sha256"] ||
        throw(ArgumentError("replay parent config snapshot hash mismatch"))
    for (relative, expected_hash) in metadata["source_sha256"]
        snapshot = joinpath(parent, "source", relative)
        isfile(snapshot) && Evidence.file_hash(snapshot) == expected_hash ||
            throw(ArgumentError("replay parent source snapshot mismatch: $relative"))
    end
    runner_relative = joinpath("scripts", "run_figure5b_protocol.jl")
    archived_runner = joinpath(parent, "source", runner_relative)
    isfile(archived_runner) || throw(ArgumentError("replay parent lacks archived runner"))
    Evidence.file_hash(@__FILE__) == Evidence.file_hash(archived_runner) ||
        throw(ArgumentError("executing runner differs from the archived runner"))
    haskey(metadata, "smoke") && metadata["smoke"] isa Bool ||
        throw(ArgumentError("replay parent metadata lacks a Boolean smoke mode"))
    return (parent, metadata, smoke=metadata["smoke"],
        checksums_sha256=Evidence.file_hash(checksums_path),
        metadata_sha256=Evidence.file_hash(metadata_path),
        config_sha256=Evidence.file_hash(archived_config),
        runner_sha256=Evidence.file_hash(archived_runner))
end

function captured_call(action)
    try
        return (result=action(), error=nothing)
    catch error
        error isa InterruptException && rethrow()
        return (result=nothing, error=Evidence.error_record(error))
    end
end

function record_stage_error!(metadata, stage, error_record)
    metadata["stage_status"][stage] = "failed"
    metadata["stage_errors"][stage] = error_record
    metadata["execution_success"] = false
    return metadata
end

function finalize_artifact!(output, metadata)
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return metadata
end

function equilibrium_artifacts(searches_by_ratio, output)
    search_rows, attempt_rows, equilibrium_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    root_match_rows, unresolved_rows = NamedTuple[], NamedTuple[]
    for ratio in sort!(collect(keys(searches_by_ratio)))
        searches = searches_by_ratio[ratio]
        reference = first(searches).equilibria
        for (grid_index, (grid, search)) in enumerate(zip(ROOT_GRIDS, searches))
            search_id = "ratio_$(replace(string(ratio), "." => "p"))_grid_$grid"
            push!(search_rows, (; search_id, ratio, grid_points=grid,
                status="completed", discovered_equilibria=length(search.equilibria),
                attracting=count(root -> root.stability.classification == Attracting,
                    search.equilibria),
                saddles=count(root -> root.stability.classification == Saddle,
                    search.equilibria),
                repelling=count(root -> root.stability.classification == Repelling,
                    search.equilibria),
                unresolved=count(root -> root.stability.classification == StabilityUnresolved,
                    search.equilibria), attempts=length(search.attempts),
                unresolved_nearby=length(search.unresolved_nearby),
                completeness=string(search.completeness)))
            for (index, attempt) in enumerate(search.attempts)
                push!(attempt_rows, (; search_id, attempt=index,
                    seed_E=attempt.seed[1], seed_I=attempt.seed[2],
                    candidate_E=attempt.candidate[1], candidate_I=attempt.candidate[2],
                    solver_status=string(attempt.solver_status),
                    solver_success=attempt.solver_success,
                    solver_residual_E=attempt.solver_residual[1],
                    solver_residual_I=attempt.solver_residual[2],
                    balance_residual_E=attempt.balance_residual[1],
                    balance_residual_I=attempt.balance_residual[2],
                    residual_norm=attempt.residual_norm,
                    balance_jacobian_11=attempt.balance_jacobian[1, 1],
                    balance_jacobian_12=attempt.balance_jacobian[1, 2],
                    balance_jacobian_21=attempt.balance_jacobian[2, 1],
                    balance_jacobian_22=attempt.balance_jacobian[2, 2],
                    validation=string(attempt.validation),
                    near_singular=attempt.near_singular,
                    reasons=join(string.(attempt.reasons), ";")))
            end
            mapping = grid_index == 1 ? collect(eachindex(reference)) :
                unique_root_mapping(reference, search.equilibria, 1e-6)
            for (index, root) in enumerate(search.equilibria)
                stability = root.stability
                push!(equilibrium_rows, (; search_id, equilibrium=index,
                    E=root.state[1], I=root.state[2],
                    balance_residual_E=root.balance_residual[1],
                    balance_residual_I=root.balance_residual[2],
                    residual_norm=maximum(abs, root.balance_residual),
                    balance_jacobian_11=root.balance_jacobian[1, 1],
                    balance_jacobian_12=root.balance_jacobian[1, 2],
                    balance_jacobian_21=root.balance_jacobian[2, 1],
                    balance_jacobian_22=root.balance_jacobian[2, 2],
                    near_singular=root.near_singular,
                    representative_attempt=root.representative_attempt,
                    member_attempts=join(root.member_attempts, ";"),
                    stability=string(stability.classification),
                    geometry=string(stability.geometry),
                    eigenvalue_1_real=real(stability.eigenvalues[1]),
                    eigenvalue_1_imaginary=imag(stability.eigenvalues[1]),
                    eigenvalue_2_real=real(stability.eigenvalues[2]),
                    eigenvalue_2_imaginary=imag(stability.eigenvalues[2]),
                    spectral_abscissa=stability.spectral_abscissa,
                    ode_jacobian_11=stability.jacobian[1, 1],
                    ode_jacobian_12=stability.jacobian[1, 2],
                    ode_jacobian_21=stability.jacobian[2, 1],
                    ode_jacobian_22=stability.jacobian[2, 2],
                    trace=stability.trace, determinant=stability.determinant,
                    threshold_1=stability.thresholds[1],
                    threshold_2=stability.thresholds[2]))
            end
            for (group, members) in enumerate(search.unresolved_nearby),
                    attempt_index in members
                push!(unresolved_rows, (; search_id, group,
                    attempt=attempt_index))
            end
            if mapping !== nothing
                for track in eachindex(reference)
                    matched = search.equilibria[mapping[track]]
                    push!(root_match_rows, (; ratio, root_track=track, grid_points=grid,
                        equilibrium=mapping[track], E=matched.state[1], I=matched.state[2],
                        distance=norm(reference[track].state .- matched.state),
                        stability=string(matched.stability.classification)))
                end
            end
        end
    end
    write_rows(joinpath(output, "equilibrium_searches.csv"), search_rows,
        (:search_id, :ratio, :grid_points, :status))
    write_rows(joinpath(output, "equilibrium_attempts.csv"), attempt_rows,
        (:search_id, :attempt))
    write_rows(joinpath(output, "equilibria.csv"), equilibrium_rows,
        (:search_id, :equilibrium))
    write_rows(joinpath(output, "root_matches.csv"), root_match_rows,
        (:ratio, :root_track, :grid_points))
    write_rows(joinpath(output, "unresolved_nearby.csv"), unresolved_rows,
        (:search_id, :group, :attempt))
end

function cross_ratio_artifacts(searches_by_ratio, reference_ratio, tolerance, output)
    reference = searches_by_ratio[reference_ratio][3].equilibria
    rows = NamedTuple[]
    for ratio in sort!(collect(keys(searches_by_ratio)))
        candidate = searches_by_ratio[ratio][3].equilibria
        mapping = unique_root_mapping(reference, candidate, tolerance)
        if mapping === nothing
            push!(rows, (; reference_ratio, ratio, root_track=0,
                equilibrium=0, matched=false, distance=NaN,
                reference_E=NaN, reference_I=NaN, E=NaN, I=NaN,
                stability="not_available"))
        else
            for track in eachindex(reference)
                matched = candidate[mapping[track]]
                push!(rows, (; reference_ratio, ratio, root_track=track,
                    equilibrium=mapping[track], matched=true,
                    distance=norm(reference[track].state .- matched.state),
                    reference_E=reference[track].state[1],
                    reference_I=reference[track].state[2],
                    E=matched.state[1], I=matched.state[2],
                    stability=string(matched.stability.classification)))
            end
        end
    end
    write_rows(joinpath(output, "cross_ratio_matches.csv"), rows,
        (:reference_ratio, :ratio, :root_track, :matched))
    return rows
end

function write_topology_artifact(output, manuscript, anchor, below,
    central_below_attracting)
    topology_record(result) = Dict(
        "qualified" => result.qualified,
        "reasons" => string.(result.reasons),
        "central_track" => result.central_track,
        "central_state" => result.central_state,
        "nullcline_slopes" => result.nullcline_slopes,
        "minimum_root_separation" => result.minimum_root_separation,
        "refinement_grid_points" => collect(result.refinement_grid_points),
        "root_tracks" => result.root_tracks)
    Evidence.write_toml(joinpath(output, "topology.toml"), Dict(
        "manuscript_ratio" => topology_record(manuscript),
        "orbit_anchor_ratio" => topology_record(anchor),
        "below_hopf" => Dict("qualified" => below.qualified,
            "reasons" => string.(below.reasons),
            "central_equilibrium_attracting" => central_below_attracting)))
end

function continuation_options(config; smoke=false)
    source = config.continuation_options
    return PeriodicContinuationOptions(
        initial_step=source.initial_step, minimum_step=source.minimum_step,
        maximum_step=source.maximum_step,
        max_steps=smoke ? config.smoke.continuation_steps : source.max_steps,
        max_corrector_iters=source.max_corrector_iters,
        corrector_atol=source.corrector_atol,
        parameter_difference_step=source.parameter_difference_step,
        state_scales=source.state_scales,
        log_period_scale=source.log_period_scale,
        parameter_scale=source.parameter_scale,
        rank_atol=source.rank_atol, rank_rtol=source.rank_rtol,
        tangent_alignment_min=source.tangent_alignment_min)
end

function hopf_continuation_options(config; smoke=false)
    source = config.continuation_options
    return PeriodicContinuationOptions(
        initial_step=min(0.02, source.maximum_step),
        minimum_step=source.minimum_step,
        maximum_step=source.maximum_step,
        max_steps=smoke ? config.smoke.hopf_continuation_steps : source.max_steps,
        max_corrector_iters=source.max_corrector_iters,
        corrector_atol=source.corrector_atol,
        parameter_difference_step=source.parameter_difference_step,
        state_scales=source.state_scales,
        log_period_scale=source.log_period_scale,
        parameter_scale=source.parameter_scale,
        rank_atol=source.rank_atol, rank_rtol=source.rank_rtol,
        tangent_alignment_min=source.tangent_alignment_min)
end

function _archive_provenance(config_path, output, config, smoke, provenance,
    canonical_config, replay_info)
    metadata = Evidence.archive_provenance(config_path, output)
    relative = joinpath("scripts", "run_figure5b_protocol.jl")
    destination = joinpath(output, "source", relative)
    cp(joinpath(REPOSITORY_ROOT, relative), destination)
    metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    candidate_text = join((string(key) * "=" * string(getproperty(config.candidate, key))
        for key in propertynames(config.candidate)), "\n")
    metadata["git_revision"] = provenance.actual_revision
    metadata["git_status_porcelain"] = provenance.status_porcelain
    replay_mode = replay_info !== nothing
    merge!(metadata, Dict(
        "purpose" => "candidate-specific Figure-5b Hopf and periodic-orbit numerical validation",
        "experiment" => "figure5b_hopf_protocol", "smoke" => smoke,
        "replay_mode" => replay_mode,
        "scientific_acceptance_enabled" => !smoke && !replay_mode && canonical_config,
        "canonical_config_validated" => canonical_config,
        "accepted_revision_expected" => isnothing(provenance.expected_revision) ?
            "not_supplied" : provenance.expected_revision,
        "accepted_revision_actual" => provenance.actual_revision,
        "accepted_revision_match" => provenance.revision_match,
        "detached_head" => provenance.detached,
        "head_name" => provenance.head_name,
        "clean_working_tree" => provenance.clean,
        "provenance_eligible" => provenance.eligible,
        "provenance_reasons" => string.(provenance.reasons),
        "replay_parent" => replay_mode ? replay_info.parent : "not_applicable",
        "replay_parent_checksums_sha256" => replay_mode ?
            replay_info.checksums_sha256 : "not_applicable",
        "replay_parent_metadata_sha256" => replay_mode ?
            replay_info.metadata_sha256 : "not_applicable",
        "replay_parent_config_sha256" => replay_mode ?
            replay_info.config_sha256 : "not_applicable",
        "replay_parent_runner_sha256" => replay_mode ?
            replay_info.runner_sha256 : "not_applicable",
        "candidate" => config.candidate,
        "candidate_sha256" => bytes2hex(SHA.sha256(candidate_text)),
        "baseline_drive" => [0.0, 0.0],
        "root_grid_policy" => config.root_grids,
        "equilibrium_seed_policy" => "union of default deterministic seeds with independent 11-by-11, 21-by-21 and 41-by-41 grids",
        "seed_ordering" => "ratio, phase fraction, amplitude factor, period factor",
        "orbit_selection_policy" => "first attempt in the largest qualifying phase-equivalence cluster; attempt identifier breaks ties",
        "completeness" => CompletenessNotCertified,
        "absence_claim" => "not_permitted", "prevalence_claim" => "not_permitted",
        "exact_attractor_count_claim" => "not_permitted",
        "biological_interpretation" => "not_assigned",
        "reachability_role" => "not_run; existence and local attraction are separate outputs",
        "interruption_policy" => "partial artifacts are diagnostic and not resumable; restart into a new empty output directory",
        "replay_from_artifact_directory" => "julia --project=source source/scripts/run_figure5b_protocol.jl --config config.toml --output ../replay --replay-parent .",
        "artifact_schema" => Dict("version" => 1,
            "equilibria" => "all equilibrium attempts, roots and cross-grid coordinate matches",
            "hopf" => "analytical diagnostics plus independent finite-difference trace bisection",
            "orbits" => "all shooting outcomes and sampled validated trajectories",
            "validation" => "primitive-period, winding, Jacobian, divergence, Poincare and tight-replay checks",
            "continuation" => "all accepted points and failed correctors; endpoint classifications remain compatibility statements")))
    return metadata, provenance
end

function _continuation_artifacts(output, continuation, center, p, roots,
    critical_ratio, linear_period, expected_radius_slope, config;
    prefix="", expected_hopf_boundary=nothing)
    branch_rows, point_rows, attempt_rows, winding_rows =
        NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    directory_name = isempty(prefix) ? "continuation" : prefix * "continuation"
    mkpath(joinpath(output, "orbits", directory_name))
    observations = branch_observations(continuation, center, p, roots, config)
    endpoints = Dict{Symbol,Any}()
    for (name, branch) in ((:negative, continuation.negative),
            (:positive, continuation.positive))
        endpoint = endpoint_classification(branch.termination, observations[name],
            branch.parameter_reversals, critical_ratio, linear_period,
            expected_hopf_boundary, config)
        endpoints[name] = endpoint
        push!(branch_rows, (; branch=string(name), direction=branch.direction,
            points=length(branch.points), attempts=length(branch.attempts),
            parameter_reversals=join(branch.parameter_reversals, ";"),
            termination=string(branch.termination),
            endpoint_classification=string(endpoint.classification),
            endpoint_reasons=join(string.(endpoint.reasons), ";"),
            hopf_compatible=endpoint.hopf_compatible,
            fold_compatible=endpoint.fold_compatible,
            homoclinic_compatible=endpoint.homoclinic_compatible,
            state_boundary_compatible=endpoint.state_boundary_compatible))
        for (index, point) in enumerate(branch.points)
            orbit = point.orbit
            distances = periodic_orbit_distances(orbit, center)
            observation = observations[name][index]
            push!(point_rows, (; branch=string(name), point=index,
                ratio=point.parameter, initial_E=point.state[1], initial_I=point.state[2],
                period=point.period, tangent_E=point.tangent[1],
                tangent_I=point.tangent[2], tangent_log_period=point.tangent[3],
                tangent_parameter=point.tangent[4],
                E_half_range=point.component_half_ranges[1],
                I_half_range=point.component_half_ranges[2], signed_area=point.signed_area,
                distance_minimum=distances.minimum, distance_rms=distances.rms,
                distance_maximum=distances.maximum,
                modal_radius=observation.modal_radius,
                closure_residual=orbit.closure_residual,
                phase_residual=orbit.phase_residual,
                equation_residual=orbit.equation_residual,
                refinement_waveform=orbit.refinement_difference.waveform,
                refinement_period=orbit.refinement_difference.period,
                refinement_monodromy=orbit.refinement_difference.monodromy,
                validation=string(orbit.validation), stability=string(orbit.stability),
                floquet_1_real=real(orbit.floquet_multipliers[1]),
                floquet_1_imaginary=imag(orbit.floquet_multipliers[1]),
                floquet_2_real=real(orbit.floquet_multipliers[2]),
                floquet_2_imaginary=imag(orbit.floquet_multipliers[2]),
                transverse_multiplier_real=real(orbit.transverse_multiplier),
                transverse_multiplier_imaginary=imag(orbit.transverse_multiplier),
                primitive=point.primitive_period_check.primitive,
                primitive_alias=string(point.primitive_period_check.alias_divisor),
                divergence_resolved=point.divergence_check.resolved,
                divergence_multiplier=point.divergence_check.divergence_multiplier,
                divergence_discrepancy=point.divergence_check.discrepancy))
            for (root_index, root) in enumerate(roots)
                winding = periodic_orbit_winding(orbit, root.state;
                    center_atol=config.independent.winding_center_atol)
                push!(winding_rows, (; branch=string(name), point=index,
                    equilibrium=root_index, E=root.state[1], I=root.state[2],
                    resolved=winding.resolved, reason=string(winding.reason),
                    winding=isnothing(winding.winding) ? 0 : winding.winding,
                    minimum_distance=winding.minimum_distance, samples=winding.samples))
            end
            write_orbit(joinpath(output, "orbits", directory_name,
                "$(name)_$(lpad(index, 4, '0')).csv"), orbit)
        end
        for (index, attempt) in enumerate(branch.attempts)
            orbit = attempt.orbit
            push!(attempt_rows, (; branch=string(name), attempt=index,
                direction=attempt.direction, step=attempt.step, iterations=attempt.iterations,
                predictor_E=attempt.predictor[1], predictor_I=attempt.predictor[2],
                predictor_log_period=attempt.predictor[3],
                predictor_parameter=attempt.predictor[4],
                candidate_E=attempt.candidate[1], candidate_I=attempt.candidate[2],
                candidate_log_period=attempt.candidate[3],
                candidate_parameter=attempt.candidate[4],
                corrector_arclength_residual=attempt.corrector_arclength_residual,
                corrector_constraint_residual=attempt.corrector_constraint_residual,
                post_shoot_arclength_residual=attempt.post_shoot_arclength_residual,
                post_shoot_constraint_residual=attempt.post_shoot_constraint_residual,
                status=string(attempt.status), accepted=attempt.accepted,
                orbit_present=orbit !== nothing,
                orbit_validation=orbit === nothing ? "not_available" : string(orbit.validation),
                orbit_stability=orbit === nothing ? "not_available" : string(orbit.stability),
                orbit_period=orbit === nothing ? NaN : orbit.period,
                orbit_closure=orbit === nothing ? NaN : orbit.closure_residual,
                orbit_phase=orbit === nothing ? NaN : orbit.phase_residual,
                orbit_equation=orbit === nothing ? NaN : orbit.equation_residual,
                orbit_transverse_real=orbit === nothing ? NaN :
                    real(orbit.transverse_multiplier),
                orbit_transverse_imaginary=orbit === nothing ? NaN :
                    imag(orbit.transverse_multiplier)))
        end
    end
    write_rows(joinpath(output, prefix * "continuation_branches.csv"), branch_rows,
        (:branch, :direction, :termination))
    write_rows(joinpath(output, prefix * "continuation_points.csv"), point_rows,
        (:branch, :point, :ratio))
    write_rows(joinpath(output, prefix * "continuation_attempts.csv"), attempt_rows,
        (:branch, :attempt, :status))
    write_rows(joinpath(output, prefix * "continuation_windings.csv"), winding_rows,
        (:branch, :point, :equilibrium, :resolved, :winding))
    fit = branch_fit(continuation, observations, critical_ratio, config)
    slope_agreement = fit.resolved && isapprox(fit.slope, expected_radius_slope;
        rtol=config.independent.fit_slope_rtol, atol=0.0)
    Evidence.write_toml(joinpath(output, prefix * "branch_fit.toml"), Dict(
        "resolved" => fit.resolved, "branch" => fit.branch,
        "intercept" => fit.intercept, "slope" => fit.slope,
        "r_squared" => fit.r_squared, "points" => fit.points,
        "slope_agreement" => slope_agreement,
        "interpretation" => "one finite monotone pre-reversal branch segment; not a complete branch certificate"))
    return (; fit, endpoints, observations)
end

function write_empty_continuation_artifacts(output, prefix, reason)
    write_rows(joinpath(output, prefix * "continuation_branches.csv"), NamedTuple[],
        (:branch, :direction, :termination))
    write_rows(joinpath(output, prefix * "continuation_points.csv"), NamedTuple[],
        (:branch, :point, :ratio))
    write_rows(joinpath(output, prefix * "continuation_attempts.csv"), NamedTuple[],
        (:branch, :attempt, :status))
    write_rows(joinpath(output, prefix * "continuation_windings.csv"), NamedTuple[],
        (:branch, :point, :equilibrium, :resolved, :winding))
    Evidence.write_toml(joinpath(output, prefix * "branch_fit.toml"), Dict(
        "resolved" => false, "reason" => reason))
end

"""Run the frozen clean protocol; smoke mode never enables scientific acceptance."""
function run_experiment(config_path::AbstractString, output_dir::AbstractString;
    smoke=false, require_canonical=true, accepted_revision=nothing,
    replay_parent=nothing, failure_injection=nothing)
    smoke isa Bool || throw(ArgumentError("smoke must be Bool"))
    output = abspath(output_dir)
    if replay_parent !== nothing
        islink(output) && throw(ArgumentError(
            "replay output must not be an existing symbolic link"))
        parent = realpath(replay_parent)
        output_parent = realpath(dirname(output))
        output = joinpath(output_parent, basename(output))
        output != parent && output_parent == dirname(parent) ||
            throw(ArgumentError("replay output must be a sibling of the parent artifact"))
    end
    replay_info = replay_parent === nothing ? nothing :
        validate_replay_parent(replay_parent, config_path)
    replay_info !== nothing && (smoke = replay_info.smoke)
    failure_injection === nothing || failure_injection in
        (:equilibria, :topology, :hopf_diagnostics, :hopf_basis, :orbit_seeds,
         :tight_validation, :continuation, :hopf_continuation) ||
        throw(ArgumentError("unknown failure injection"))
    config = load_config(config_path; require_canonical)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    provenance = capture_provenance(accepted_revision)
    mkpath(joinpath(output, "orbits", "shooting"))
    current_stage = "provenance_archive"
    metadata = Dict{String,Any}(
        "experiment" => "figure5b_hopf_protocol", "smoke" => smoke,
        "replay_mode" => replay_info !== nothing,
        "stage_status" => Dict{String,Any}(),
        "stage_errors" => Dict{String,Any}())
    try
    metadata, provenance = _archive_provenance(config_path, output, config, smoke,
        provenance, require_canonical, replay_info)
    metadata["failure_injection"] = failure_injection === nothing ?
        "not_active" : string(failure_injection)
    metadata["stage_status"] = Dict{String,Any}()
    metadata["stage_errors"] = Dict{String,Any}()
    if replay_info === nothing && !smoke && !provenance.eligible
        metadata["execution_success"] = false
        metadata["evidence_eligible"] = false
        metadata["run_status"] = "provenance_ineligible"
        metadata["stage_status"]["provenance"] = "ineligible"
        metadata["provenance_failure"] = Dict(
            "type" => "ProvenanceEligibilityError",
            "message" => join(string.(provenance.reasons), ";"))
        metadata["numerical_four_attractor_coexistence"] =
            "not_executed_provenance_ineligible"
        metadata["supercritical_hopf_conclusion"] =
            "not_executed_provenance_ineligible"
        metadata["scientific_outcome"] =
            "protocol not executed because accepted-revision provenance failed closed"
        finalize_artifact!(output, metadata)
        return (; success=false, smoke, topology_manuscript=false,
            topology_anchor=false, below_hopf=false, hopf=false, orbit=false,
            scientific_acceptance=false)
    end
    metadata["stage_status"]["provenance"] = replay_info !== nothing ?
        "verified_parent_replay" : smoke ? "not_required_smoke" : "passed"

    current_stage = "equilibria"
    failure_injection == :equilibria && error("injected equilibrium-stage failure")
    ratios = (config.ratios.below_hopf, config.ratios.orbit_anchor,
        config.ratios.manuscript)
    searches = Dict(ratio => equilibrium_refinements(config, ratio) for ratio in ratios)
    equilibrium_artifacts(searches, output)
    cross_ratio_artifacts(searches, config.ratios.manuscript,
        config.topology_options.coordinate_match_atol, output)
    metadata["stage_status"]["equilibria"] = "completed"
    current_stage = "topology"
    failure_injection == :topology && error("injected topology-stage failure")
    topology_manuscript = classify_figure5b_topology(searches[config.ratios.manuscript];
        options=config.topology_options)
    topology_anchor = classify_figure5b_topology(searches[config.ratios.orbit_anchor];
        options=config.topology_options)
    below = below_hopf_assessment(searches[config.ratios.below_hopf],
        config.topology_options)
    central_state = topology_manuscript.central_state
    central_state === nothing && (central_state = topology_anchor.central_state)
    central_state === nothing && throw(ErrorException("central equilibrium was not resolved"))
    below_central_matches = findall(root -> norm(root.state .- central_state) <=
        config.topology_options.coordinate_match_atol,
        searches[config.ratios.below_hopf][3].equilibria)
    central_below_attracting = length(below_central_matches) == 1 &&
        searches[config.ratios.below_hopf][3].equilibria[
            only(below_central_matches)].stability.classification == Attracting
    write_topology_artifact(output, topology_manuscript, topology_anchor, below,
        central_below_attracting)
    metadata["stage_status"]["topology"] = "completed"

    current_stage = "hopf_diagnostics"
    failure_injection == :hopf_diagnostics &&
        error("injected Hopf-diagnostic-stage failure")
    diagnostics = hopf_diagnostics(candidate_model(config, config.ratios.manuscript),
        central_state; options=config.hopf_options)
    trace = numerical_trace_zero(config, central_state)
    hopf_agreement = diagnostics.resolved && trace.resolved &&
        abs(diagnostics.critical_ratio - trace.ratio) <=
            config.independent.trace_agreement_atol
    Evidence.write_toml(joinpath(output, "hopf.toml"), Dict(
        "analytical" => diagnostics, "numerical_trace_ratio" => trace.ratio,
        "numerical_trace" => trace.trace, "numerical_trace_resolved" => trace.resolved,
        "location_agreement" => hopf_agreement,
        "statement" => "local Hopf diagnostics do not establish a periodic orbit"))
    write_rows(joinpath(output, "hopf_trace.csv"), trace.rows,
        (:iteration, :lower, :upper, :midpoint, :midpoint_trace))
    metadata["stage_status"]["hopf_diagnostics"] = "completed"

    current_stage = "hopf_basis"
    failure_injection == :hopf_basis && error("injected Hopf-basis-stage failure")
    basis = hopf_basis(candidate_model(config, diagnostics.critical_ratio),
        central_state, diagnostics.frequency)
    current_stage = "orbit_seeds"
    failure_injection == :orbit_seeds && error("injected orbit-seed-stage failure")
    schedule = seed_schedule(config, diagnostics, basis, central_state; smoke)
    attempts = Any[]
    attempt_rows = NamedTuple[]
    for seed in schedule
        model = candidate_model(config, seed.ratio)
        result, error_type, error_message = try
            (solve_periodic_orbit(model, [seed.seed_E, seed.seed_I], seed.period_guess;
                options=config.shooting_options), "", "")
        catch error
            error isa InterruptException && rethrow()
            (nothing, string(typeof(error)), sprint(showerror, error))
        end
        record = merge(seed, (; result))
        push!(attempts, record)
        validated = result !== nothing &&
            result.validation == NumericallyValidatedPeriodicOrbit
        push!(attempt_rows, (; seed.attempt_id, seed.ratio, seed.phase,
            seed.amplitude_factor, seed.period_factor, seed.modal_radius,
            seed.seed_E, seed.seed_I, seed.period_guess,
            status=result === nothing ? "exception" : string(result.validation),
            validated,
            stability=result === nothing ? "not_available" : string(result.stability),
            period=result === nothing ? NaN : result.period,
            closure_residual=result === nothing ? NaN : result.closure_residual,
            phase_residual=result === nothing ? NaN : result.phase_residual,
            equation_residual=result === nothing ? NaN : result.equation_residual,
            transverse_multiplier=result === nothing ? NaN : real(result.transverse_multiplier),
            reasons=result === nothing ? "" : join(string.(result.reasons), ";"),
            error_type, error_message))
        validated && write_orbit(joinpath(output, "orbits", "shooting",
            seed.attempt_id * ".csv"), result)
    end
    write_rows(joinpath(output, "orbit_attempts.csv"), attempt_rows,
        (:attempt_id, :ratio, :status))
    metadata["stage_status"]["orbit_shooting"] = "completed"

    anchor_attempts = filter(attempt -> attempt.ratio == config.ratios.orbit_anchor,
        attempts)
    clusters = cluster_validated_attempts(anchor_attempts, config)
    cluster_rows = NamedTuple[]
    for (cluster_index, cluster) in enumerate(clusters), attempt in cluster
        push!(cluster_rows, (; cluster=cluster_index, attempt_id=attempt.attempt_id,
            members=length(cluster), phase=attempt.phase,
            period_factor=attempt.period_factor))
    end
    write_rows(joinpath(output, "orbit_clusters.csv"), cluster_rows,
        (:cluster, :attempt_id, :members))
    metadata["stage_status"]["orbit_clustering"] = "completed"
    selected_cluster = qualifying_cluster(clusters, config)
    representative = selected_cluster === nothing ? nothing : first(selected_cluster)

    tight_result = nothing
    validation = nothing
    tight_error = nothing
    current_stage = "tight_validation"
    if representative !== nothing
        tight_stage = captured_call() do
            failure_injection == :tight_validation &&
                error("injected tight-validation-stage failure")
            result = solve_periodic_orbit(candidate_model(config,
                config.ratios.orbit_anchor), representative.result.initial_state,
                representative.result.period; options=tighter_periodic_options(config))
            roots = searches[config.ratios.orbit_anchor][3].equilibria
            checked = validate_orbit(result, representative.result, roots, config)
            return (; result, validation=checked)
        end
        if tight_stage.error === nothing
            tight_result = tight_stage.result.result
            validation = tight_stage.result.validation
            tight_result.validation == NumericallyValidatedPeriodicOrbit &&
                write_orbit(joinpath(output, "orbits", "tight_replay.csv"), tight_result)
            metadata["stage_status"]["tight_validation"] = "completed"
        else
            tight_error = tight_stage.error
            record_stage_error!(metadata, "tight_validation", tight_error)
        end
    else
        metadata["stage_status"]["tight_validation"] = "skipped_no_qualifying_cluster"
    end
    write_tight_validation_artifact(output,
        representative === nothing ? nothing : representative.attempt_id,
        tight_result, validation, central_state, basis.p; stage_error=tight_error)
    validation_rows = [validation === nothing ?
        (; selected="not_available", accepted=false, reasons="no_qualifying_cluster",
            validation="not_available", stability="not_available", period=NaN,
            transverse_multiplier=NaN) :
        (; selected=representative.attempt_id, accepted=validation.accepted,
            reasons=join(string.(validation.reasons), ";"),
            validation=string(tight_result.validation), stability=string(tight_result.stability),
            period=tight_result.period,
            transverse_multiplier=real(tight_result.transverse_multiplier))]
    write_rows(joinpath(output, "orbit_validations.csv"), validation_rows,
        (:selected, :accepted, :reasons))
    write_rows(joinpath(output, "jacobian_checks.csv"),
        validation === nothing || validation.jacobian === nothing ? NamedTuple[] :
            validation.jacobian.rows, (:step, :discrepancy))
    write_rows(joinpath(output, "poincare_checks.csv"),
        validation === nothing || validation.poincare === nothing ? NamedTuple[] :
            validation.poincare.rows, (:step, :multiplier))
    write_rows(joinpath(output, "winding_checks.csv"),
        validation === nothing ? NamedTuple[] : validation.windings,
        (:equilibrium, :resolved, :winding))

    roots_anchor = searches[config.ratios.orbit_anchor][3].equilibria
    continuation = nothing
    continuation_evidence = nothing
    hopf_continuation = nothing
    hopf_evidence = nothing
    fit = nothing
    if validation !== nothing && validation.accepted
        current_stage = "continuation"
        continuation_stage = captured_call() do
            failure_injection == :continuation &&
                error("injected continuation-stage failure")
            result = continue_periodic_orbit(
                ratio -> candidate_model(config, ratio), tight_result.initial_state,
                tight_result.period, config.ratios.orbit_anchor;
                parameter_bounds=config.ratios.continuation_bounds,
                options=continuation_options(config; smoke),
                periodic_options=tighter_periodic_options(config))
            evidence = _continuation_artifacts(output, result, central_state, basis.p,
                roots_anchor, diagnostics.critical_ratio, diagnostics.linear_period,
                diagnostics.modal_radius_squared_slope, config)
            return (; result, evidence)
        end
        if continuation_stage.error === nothing
            continuation = continuation_stage.result.result
            continuation_evidence = continuation_stage.result.evidence
            metadata["stage_status"]["continuation"] = "completed"
        else
            record_stage_error!(metadata, "continuation", continuation_stage.error)
            write_empty_continuation_artifacts(output, "", "continuation_exception")
        end

        current_stage = "hopf_continuation"
        hopf_boundary = diagnostics.critical_ratio +
            config.independent.hopf_target_offset
        hopf_stage = captured_call() do
            failure_injection == :hopf_continuation &&
                error("injected Hopf-continuation-stage failure")
            result = continue_periodic_orbit(
                ratio -> candidate_model(config, ratio), tight_result.initial_state,
                tight_result.period, config.ratios.orbit_anchor;
                parameter_bounds=(hopf_boundary, config.ratios.continuation_bounds[2]),
                options=hopf_continuation_options(config; smoke),
                periodic_options=tighter_periodic_options(config))
            evidence = _continuation_artifacts(output, result, central_state, basis.p,
                roots_anchor, diagnostics.critical_ratio, diagnostics.linear_period,
                diagnostics.modal_radius_squared_slope, config;
                prefix="hopf_", expected_hopf_boundary=hopf_boundary)
            return (; result, evidence)
        end
        if hopf_stage.error === nothing
            hopf_continuation = hopf_stage.result.result
            hopf_evidence = hopf_stage.result.evidence
            fit = hopf_evidence.fit
            metadata["stage_status"]["hopf_continuation"] = "completed"
        else
            record_stage_error!(metadata, "hopf_continuation", hopf_stage.error)
            write_empty_continuation_artifacts(output, "hopf_",
                "hopf_continuation_exception")
        end
    else
        write_empty_continuation_artifacts(output, "", "orbit_validation_not_accepted")
        write_empty_continuation_artifacts(output, "hopf_",
            "orbit_validation_not_accepted")
        metadata["stage_status"]["continuation"] = "skipped_orbit_not_accepted"
        metadata["stage_status"]["hopf_continuation"] = "skipped_orbit_not_accepted"
    end

    numerical_four_attractor = validation !== nothing && validation.accepted &&
        topology_anchor.qualified
    fit_assessment = fit_acceptance_assessment(fit,
        hopf_evidence === nothing ? Dict{Symbol,Any}() : hopf_evidence.endpoints,
        diagnostics, config)
    hopf_endpoint = fit_assessment.endpoint
    fit_acceptance = fit_assessment.accepted
    supercritical_candidate = supercritical_hopf_candidate(
        numerical_four_attractor, topology_manuscript.qualified, diagnostics,
        hopf_agreement, below.qualified, central_below_attracting,
        fit_acceptance, metadata["stage_errors"])
    clean_provenance = provenance.eligible
    execution_success = isempty(metadata["stage_errors"])
    replay_mode = replay_info !== nothing
    evidence_eligible = scientific_evidence_eligible(smoke, replay_mode,
        provenance.eligible, execution_success, failure_injection, require_canonical)
    metadata["execution_success"] = execution_success
    metadata["run_status"] = execution_success ?
        (replay_mode ? "replay_completed_evidence_ineligible" : "completed") :
        "execution_failed"
    metadata["clean_provenance"] = clean_provenance
    metadata["evidence_eligible"] = evidence_eligible
    metadata["topology_manuscript_qualified"] = topology_manuscript.qualified
    metadata["topology_anchor_qualified"] = topology_anchor.qualified
    metadata["below_hopf_four_equilibria_qualified"] = below.qualified
    metadata["central_equilibrium_attracting_below_hopf"] = central_below_attracting
    metadata["hopf_diagnostics_resolved"] = diagnostics.resolved
    metadata["hopf_location_agreement"] = hopf_agreement
    metadata["qualifying_orbit_cluster"] = selected_cluster !== nothing
    metadata["orbit_validation_accepted"] = validation !== nothing && validation.accepted
    metadata["branch_fit_accepted"] = fit_acceptance
    metadata["branch_fit_branch"] = fit_assessment.branch === nothing ?
        "not_available" : string(fit_assessment.branch)
    metadata["branch_fit_reasons"] = string.(fit_assessment.reasons)
    metadata["hopf_endpoint_classification"] = hopf_endpoint === nothing ?
        "not_available" : string(hopf_endpoint.classification)
    metadata["hopf_endpoint_reasons"] = hopf_endpoint === nothing ? String[] :
        string.(hopf_endpoint.reasons)
    ineligible_label = !execution_success ? "execution_failed" :
        replay_mode ? "replay_evidence_ineligible" :
        smoke ? "disabled_in_smoke" :
        !require_canonical ? "noncanonical_configuration_evidence_ineligible" :
        "ineligible_dirty_or_unresolved_provenance"
    metadata["numerical_four_attractor_coexistence"] = !evidence_eligible ?
        ineligible_label : numerical_four_attractor
    metadata["supercritical_hopf_conclusion"] = !evidence_eligible ?
        ineligible_label : supercritical_candidate
    metadata["scientific_outcome"] = !execution_success ?
        "execution_failed; no affirmative scientific disposition is permitted" :
        replay_mode ?
            "artifact-local replay completed; replay is always evidence-ineligible" :
        smoke ? "smoke execution only; no scientific acceptance" :
        !require_canonical ?
            "canonical configuration validation was disabled; no scientific acceptance" :
        !clean_provenance ?
            "numerical execution used dirty or unresolved source provenance and is not scientific evidence" :
        numerical_four_attractor ?
            "three confirmed locally attracting equilibria and one numerically validated attracting orbit were found at ratio 0.5; exact attractor count is not claimed" :
            "candidate-specific acceptance gates were not all met; this is not evidence of absence"
    finalize_artifact!(output, metadata)
    return (; success=execution_success, smoke,
        topology_manuscript=topology_manuscript.qualified,
        topology_anchor=topology_anchor.qualified, below_hopf=below.qualified,
        hopf=diagnostics.resolved && hopf_agreement,
        orbit=validation !== nothing && validation.accepted,
        scientific_acceptance=evidence_eligible && numerical_four_attractor)
    catch error
        error isa InterruptException && rethrow()
        get!(metadata, "stage_status", Dict{String,Any}())
        get!(metadata, "stage_errors", Dict{String,Any}())
        record_stage_error!(metadata, current_stage, Evidence.error_record(error))
        metadata["execution_success"] = false
        metadata["run_status"] = "execution_failed"
        metadata["evidence_eligible"] = false
        metadata["numerical_four_attractor_coexistence"] = "execution_failed"
        metadata["supercritical_hopf_conclusion"] = "execution_failed"
        metadata["scientific_outcome"] =
            "execution_failed before scientific or provenance disposition"
        finalize_artifact!(output, metadata)
        return (; success=false, smoke, topology_manuscript=false,
            topology_anchor=false, below_hopf=false, hopf=false, orbit=false,
            scientific_acceptance=false)
    end
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "figure5b_hopf.toml")
    output = nothing
    smoke = false
    accepted_revision = nothing
    replay_parent = nothing
    seen = Set{String}()
    index = 1
    while index <= length(args)
        option = args[index]
        option in seen && throw(ArgumentError("duplicate option: $option"))
        push!(seen, option)
        if option == "--smoke"
            smoke = true
            index += 1
        elseif option in ("--config", "--output", "--accepted-revision",
                "--replay-parent")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            value = args[index + 1]
            startswith(value, "--") && throw(ArgumentError("$option requires a value"))
            if option == "--config"
                config_path = value
            elseif option == "--output"
                output = value
            elseif option == "--accepted-revision"
                accepted_revision = normalize_accepted_revision(value)
            else
                replay_parent = value
            end
            index += 2
        else
            throw(ArgumentError("unknown option: $option"))
        end
    end
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    !smoke && replay_parent === nothing && accepted_revision === nothing &&
        throw(ArgumentError(
        "--accepted-revision COMMIT is required outside smoke mode"))
    result = run_experiment(config_path, output; smoke, accepted_revision,
        replay_parent)
    println("Figure-5b protocol: topology=$(result.topology_anchor), " *
        "hopf=$(result.hopf), orbit=$(result.orbit), smoke=$(result.smoke)")
    return result.success ? 0 : 1
end

end


if abspath(PROGRAM_FILE) == @__FILE__
    exit(Figure5bProtocol.main())
end
