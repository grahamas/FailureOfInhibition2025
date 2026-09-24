"""Targeted fixed-ratio Figure-5b Hopf and periodic-orbit search."""
module Figure5bRatio44Search

using FailureOfInhibition2025
using ForwardDiff
using LinearAlgebra: BLAS, Diagonal, I, det, dot, eigen, eigvals, norm, svd, svdvals
using SciMLBase: NonlinearFunction, NonlinearProblem, solve, successful_retcode
using SimpleNonlinearSolve: SimpleTrustRegion
import CSV
import SHA
import TOML

include("run_minimal_experiment.jl")
include("run_tetrastability_search.jl")
include("run_figure5b_protocol.jl")
const Evidence = MinimalExperiment
const Tetrastability = TetrastabilityExperiment
const Figure5b = Figure5bProtocol

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTPUT_IGNORE_PROBE = ".figure5b_ratio44_write_probe"
const SEARCH_AXES = (:e_to_e, :i_to_e, :e_to_i, :i_to_i, :theta_off)
const CURVE_PAIRS = ((:e_to_e, :i_to_i), (:e_to_i, :i_to_e),
    (:theta_off, :e_to_i))
const ROOT_GRIDS = (11, 21, 41)

function require_exact_keys(table, expected, label)
    table isa AbstractDict || throw(ArgumentError("$label must be a table"))
    missing = filter(key -> !haskey(table, key), expected)
    isempty(missing) || throw(ArgumentError("$label is missing: $(join(missing, ", "))"))
    extra = setdiff(collect(keys(table)), collect(expected))
    isempty(extra) || throw(ArgumentError("$label has unknown keys: $(join(sort!(extra), ", "))"))
    return table
end

function finite_number(value, label; positive=false, nonnegative=false)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("$label must be a finite real number"))
    positive && value <= 0 && throw(ArgumentError("$label must be positive"))
    nonnegative && value < 0 && throw(ArgumentError("$label must be nonnegative"))
    return Float64(value)
end

function positive_integer(value, label)
    value isa Integer && !(value isa Bool) && value > 0 ||
        throw(ArgumentError("$label must be a positive integer"))
    return Int(value)
end

function number_vector(value, label; positive=false, nonnegative=false,
    length_required=nothing)
    value isa AbstractVector && !isempty(value) ||
        throw(ArgumentError("$label must be a nonempty array"))
    result = [finite_number(item, label; positive, nonnegative) for item in value]
    isnothing(length_required) || length(result) == length_required ||
        throw(ArgumentError("$label must have length $length_required"))
    return result
end

function _options(table, constructor, keys, label; integers=())
    require_exact_keys(table, keys, label)
    pairs = Pair{Symbol,Any}[]
    for key in keys
        value = key in integers ? positive_integer(table[key], "$label.$key") :
            finite_number(table[key], "$label.$key"; nonnegative=true)
        push!(pairs, Symbol(key) => value)
    end
    return constructor(; pairs...)
end

function _canonical_check(config)
    config.seed_policy == (experiment="tetrastability_search",
        accepted_revision="b365f02d433bdad86069dc5c52dbbfeb1e2ae98e",
        initial_halton_points=1024, maximum_halton_points=4096,
        last_halton_index=4096, parameter_cells=5746,
        planes=["figure3", "figure4_exploration"]) ||
        throw(ArgumentError("seed-artifact policy differs from the canonical protocol"))
    config.fixed == (tau_e=7.8, tau_ratio=4.4, excitatory_slope=5.0,
        inhibitory_slope=5.0, theta_e=1.5, theta_on=4.0) ||
        throw(ArgumentError("fixed model differs from the manuscript-ratio protocol"))
    expected_bounds = Dict(:e_to_e => (14.0, 24.0), :i_to_e => (6.0, 18.0),
        :e_to_i => (12.0, 28.0), :i_to_i => (0.0, 10.0),
        :theta_off => (6.0, 12.0))
    config.bounds == expected_bounds ||
        throw(ArgumentError("authorized parameter bounds differ from the canonical protocol"))
    config.path_axes == SEARCH_AXES ||
        throw(ArgumentError("one-axis path schedule must contain the five authorized axes"))
    config.curve_pairs == CURVE_PAIRS ||
        throw(ArgumentError("curve schedule differs from the three authorized pairs"))
    config.root_grids == ROOT_GRIDS ||
        throw(ArgumentError("root refinement schedule must be 11, 21 and 41"))
    e = config.equilibrium_options
    (e.solver_abstol, e.solver_reltol, e.residual_atol, e.domain_atol,
        e.dedup_atol, e.singular_atol, e.singular_rtol, e.maxiters) ==
        (1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-10, 1e-8, 100) ||
        throw(ArgumentError("equilibrium policy differs from the canonical protocol"))
    s = config.stability_options
    (s.spectral_atol, s.spectral_rtol) == (1e-10, 1e-8) ||
        throw(ArgumentError("stability policy differs from the canonical protocol"))
    t = config.topology
    Tuple(getproperty(t, name) for name in propertynames(t)) ==
        (1e-6, 1e-5, 1e-9, 1e-9, 1e-8, 1e-8, 1e-7, 1e-8) ||
        throw(ArgumentError("topology policy differs from the canonical protocol"))
    h = config.hopf_options
    (h.balance_residual_atol, h.diagonal_atol, h.determinant_atol,
        h.transversality_atol, h.lyapunov_atol, h.lyapunov_agreement_atol,
        h.lyapunov_agreement_rtol, h.fd_plateau_atol, h.fd_plateau_rtol,
        h.fd_steps) == (1e-10, 1e-10, 1e-10, 1e-8, 1e-7, 1e-4, 1e-4,
        1e-4, 1e-4, (5e-4, 7.5e-4, 1e-3)) ||
        throw(ArgumentError("Hopf policy differs from the canonical protocol"))
    config.augmented == (solver_abstol=1e-11, solver_reltol=1e-10,
        residual_atol=1e-8, maxiters=100, initial_step_fraction=0.01,
        minimum_step_fraction=1e-5, maximum_step_fraction=0.05, max_steps=200,
        max_retries=8, dedup_atol=1e-6, rank_atol=1e-10,
        state_step_atol=0.05) ||
        throw(ArgumentError("axis and augmented-solve policy differs from canonical"))
    config.curve_continuation == (initial_step=0.01, minimum_step=1e-5,
        maximum_step=0.05, max_steps=300, max_corrector_iters=16,
        max_retries=8, corrector_atol=1e-8, rank_atol=1e-10,
        state_scales=(0.1, 0.1)) ||
        throw(ArgumentError("curve-continuation policy differs from canonical"))
    config.directional == (fd_steps=[1e-3, 5e-4, 2.5e-4],
        plateau_atol=1e-4, plateau_rtol=0.05, offset=0.01,
        state_match_factor=10.0) ||
        throw(ArgumentError("directional policy differs from canonical"))
    p = config.shooting.shooting_options
    (config.shooting.maximum_hopf_points, config.shooting.phase_fractions,
        config.shooting.amplitude_factors, config.shooting.period_factors,
        p.ode_abstol, p.ode_reltol, p.shooting_atol, p.validation_atol,
        p.amplitude_atol, p.phase_speed_atol, p.floquet_atol, p.refinement_factor,
        p.min_period, p.max_period, p.maxiters, p.ode_maxiters, p.samples) ==
        (32, [0.0, 0.25, 0.5, 0.75], [0.75, 1.0, 1.25], [0.9, 1.0, 1.1],
        1e-10, 1e-10, 1e-9, 1e-6, 1e-5, 1e-10, 1e-4, 0.1, 1e-3,
        100.0, 30, 1_000_000, 257) ||
        throw(ArgumentError("periodic-shooting policy differs from canonical"))
    i = config.independent
    (i.tighter_factor, i.tight_samples, i.jacobian_steps, i.jacobian_atol,
        i.poincare_steps, i.poincare_intervals, i.poincare_plateau_atol,
        i.multiplier_atol, i.orbit_equivalence_atol, i.period_rtol,
        i.minimum_phase_seeds, i.minimum_period_seeds, i.winding_center_atol) ==
        (0.1, 513, [1e-5, 5e-6, 2.5e-6], 1e-6,
        [1e-5, 5e-6, 2.5e-6], 1024, 2e-3, 5e-3, 1e-5, 1e-5,
        2, 2, 1e-8) ||
        throw(ArgumentError("independent-validation policy differs from canonical"))
    config.smoke == (maximum_seeds=1, path_steps=1, curve_steps=1,
        shooting=false) || throw(ArgumentError("smoke policy differs from canonical"))
    return config
end

"""Load and fail-closed validate the fixed-ratio search configuration."""
function load_config(path::AbstractString; require_canonical=true)
    raw = TOML.parsefile(path)
    top_keys = ("schema_version", "seed_artifact", "fixed", "axes", "paths",
        "curves", "search", "equilibrium", "stability", "topology", "hopf",
        "augmented", "curve_continuation", "directional", "shooting",
        "independent", "smoke")
    require_exact_keys(raw, top_keys, "configuration")
    raw["schema_version"] === 1 || throw(ArgumentError("schema_version must be integer 1"))

    seed_keys = ("experiment", "accepted_revision", "initial_halton_points",
        "maximum_halton_points", "last_halton_index", "parameter_cells", "planes")
    seed = require_exact_keys(raw["seed_artifact"], seed_keys, "seed_artifact")
    seed["experiment"] isa AbstractString || throw(ArgumentError("seed_artifact.experiment must be text"))
    seed["accepted_revision"] isa AbstractString && length(seed["accepted_revision"]) == 40 ||
        throw(ArgumentError("seed_artifact.accepted_revision must be a full revision hash"))
    seed["planes"] isa AbstractVector && all(x -> x isa AbstractString, seed["planes"]) ||
        throw(ArgumentError("seed_artifact.planes must contain names"))
    seed_policy = (experiment=String(seed["experiment"]),
        accepted_revision=String(seed["accepted_revision"]),
        initial_halton_points=positive_integer(seed["initial_halton_points"],
            "seed_artifact.initial_halton_points"),
        maximum_halton_points=positive_integer(seed["maximum_halton_points"],
            "seed_artifact.maximum_halton_points"),
        last_halton_index=positive_integer(seed["last_halton_index"],
            "seed_artifact.last_halton_index"),
        parameter_cells=positive_integer(seed["parameter_cells"],
            "seed_artifact.parameter_cells"), planes=String.(seed["planes"]))

    fixed_keys = ("tau_e", "tau_ratio", "excitatory_slope", "inhibitory_slope",
        "theta_e", "theta_on")
    fixed_table = require_exact_keys(raw["fixed"], fixed_keys, "fixed")
    fixed = NamedTuple{Symbol.(fixed_keys)}(Tuple(finite_number(fixed_table[key],
        "fixed.$key"; positive=key in ("tau_e", "tau_ratio", "excitatory_slope",
            "inhibitory_slope")) for key in fixed_keys))

    axes_table = require_exact_keys(raw["axes"], string.(SEARCH_AXES), "axes")
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for axis in SEARCH_AXES
        table = require_exact_keys(axes_table[string(axis)], ("minimum", "maximum"),
            "axes.$axis")
        lower = finite_number(table["minimum"], "axes.$axis.minimum"; nonnegative=true)
        upper = finite_number(table["maximum"], "axes.$axis.maximum"; nonnegative=true)
        lower < upper || throw(ArgumentError("axes.$axis must be strictly increasing"))
        bounds[axis] = (lower, upper)
    end

    paths = require_exact_keys(raw["paths"], ("axes",), "paths")["axes"]
    paths isa AbstractVector && all(x -> x isa AbstractString, paths) ||
        throw(ArgumentError("paths.axes must be an array of names"))
    path_axes = Tuple(Symbol.(paths))
    all(axis -> axis in SEARCH_AXES, path_axes) && allunique(path_axes) ||
        throw(ArgumentError("paths.axes contains an unknown or duplicate axis"))

    raw_curves = raw["curves"]
    raw_curves isa AbstractVector && !isempty(raw_curves) ||
        throw(ArgumentError("curves must be a nonempty array of tables"))
    curve_pairs = Tuple(begin
        table = require_exact_keys(item, ("first", "second"), "curves")
        first_axis, second_axis = Symbol(table["first"]), Symbol(table["second"])
        first_axis in SEARCH_AXES && second_axis in SEARCH_AXES && first_axis != second_axis ||
            throw(ArgumentError("curve axes must be distinct authorized axes"))
        (first_axis, second_axis)
    end for item in raw_curves)

    search = require_exact_keys(raw["search"], ("root_grids",), "search")
    grids = search["root_grids"]
    grids isa AbstractVector && length(grids) == 3 ||
        throw(ArgumentError("search.root_grids must have length three"))
    root_grids = Tuple(positive_integer(value, "search.root_grids") for value in grids)
    root_grids == ROOT_GRIDS || throw(ArgumentError(
        "search.root_grids must be exactly 11, 21 and 41"))

    equilibrium_keys = ("solver_abstol", "solver_reltol", "residual_atol",
        "domain_atol", "dedup_atol", "singular_atol", "singular_rtol", "maxiters")
    equilibrium_options = _options(raw["equilibrium"], EquilibriumOptions,
        equilibrium_keys, "equilibrium"; integers=("maxiters",))
    stability_options = _options(raw["stability"], StabilityOptions,
        ("spectral_atol", "spectral_rtol"), "stability")

    topology_keys = ("coordinate_match_atol", "minimum_root_separation",
        "residual_atol", "jacobian_atol", "slope_atol", "spectral_margin",
        "neutral_trace_atol", "neutral_determinant_atol")
    topology_table = require_exact_keys(raw["topology"], topology_keys, "topology")
    topology = NamedTuple{Symbol.(topology_keys)}(Tuple(finite_number(
        topology_table[key], "topology.$key"; positive=true) for key in topology_keys))
    topology.coordinate_match_atol < topology.minimum_root_separation / 2 ||
        throw(ArgumentError("topology coordinate match tolerance is too large"))
    topology_options = Figure5bTopologyOptions(
        coordinate_match_atol=topology.coordinate_match_atol,
        minimum_root_separation=topology.minimum_root_separation,
        residual_atol=topology.residual_atol, jacobian_atol=topology.jacobian_atol,
        slope_atol=topology.slope_atol, spectral_margin=topology.spectral_margin)

    hopf_keys = ("balance_residual_atol", "diagonal_atol", "determinant_atol",
        "transversality_atol", "lyapunov_atol", "lyapunov_agreement_atol",
        "lyapunov_agreement_rtol", "fd_plateau_atol", "fd_plateau_rtol", "fd_steps")
    hopf_table = require_exact_keys(raw["hopf"], hopf_keys, "hopf")
    fd_steps = Tuple(number_vector(hopf_table["fd_steps"], "hopf.fd_steps";
        positive=true))
    hopf_options = HopfDiagnosticOptions(; fd_steps,
        (Symbol(key) => finite_number(hopf_table[key], "hopf.$key"; positive=true)
            for key in hopf_keys if key != "fd_steps")...)

    augmented_keys = ("solver_abstol", "solver_reltol", "residual_atol", "maxiters",
        "initial_step_fraction", "minimum_step_fraction", "maximum_step_fraction",
        "max_steps", "max_retries", "dedup_atol", "rank_atol", "state_step_atol")
    augmented_table = require_exact_keys(raw["augmented"], augmented_keys, "augmented")
    augmented = NamedTuple{Symbol.(augmented_keys)}(Tuple(key in ("maxiters", "max_steps",
        "max_retries") ? positive_integer(augmented_table[key], "augmented.$key") :
        finite_number(augmented_table[key], "augmented.$key"; positive=true)
        for key in augmented_keys))
    augmented.minimum_step_fraction <= augmented.initial_step_fraction <=
        augmented.maximum_step_fraction || throw(ArgumentError("invalid augmented step ordering"))

    curve_keys = ("initial_step", "minimum_step", "maximum_step", "max_steps",
        "max_corrector_iters", "max_retries", "corrector_atol", "rank_atol",
        "state_scales")
    curve_table = require_exact_keys(raw["curve_continuation"], curve_keys,
        "curve_continuation")
    curve_continuation = (initial_step=finite_number(curve_table["initial_step"],
            "curve_continuation.initial_step"; positive=true),
        minimum_step=finite_number(curve_table["minimum_step"],
            "curve_continuation.minimum_step"; positive=true),
        maximum_step=finite_number(curve_table["maximum_step"],
            "curve_continuation.maximum_step"; positive=true),
        max_steps=positive_integer(curve_table["max_steps"], "curve_continuation.max_steps"),
        max_corrector_iters=positive_integer(curve_table["max_corrector_iters"],
            "curve_continuation.max_corrector_iters"),
        max_retries=positive_integer(curve_table["max_retries"],
            "curve_continuation.max_retries"),
        corrector_atol=finite_number(curve_table["corrector_atol"],
            "curve_continuation.corrector_atol"; positive=true),
        rank_atol=finite_number(curve_table["rank_atol"],
            "curve_continuation.rank_atol"; positive=true),
        state_scales=Tuple(number_vector(curve_table["state_scales"],
            "curve_continuation.state_scales"; positive=true, length_required=2)))
    curve_continuation.minimum_step <= curve_continuation.initial_step <=
        curve_continuation.maximum_step || throw(ArgumentError("invalid curve step ordering"))

    directional_table = require_exact_keys(raw["directional"], ("fd_steps",
        "plateau_atol", "plateau_rtol", "offset", "state_match_factor"), "directional")
    directional = (fd_steps=number_vector(directional_table["fd_steps"],
            "directional.fd_steps"; positive=true),
        plateau_atol=finite_number(directional_table["plateau_atol"],
            "directional.plateau_atol"; positive=true),
        plateau_rtol=finite_number(directional_table["plateau_rtol"],
            "directional.plateau_rtol"; positive=true),
        offset=finite_number(directional_table["offset"],
            "directional.offset"; positive=true),
        state_match_factor=finite_number(directional_table["state_match_factor"],
            "directional.state_match_factor"; positive=true))
    issorted(directional.fd_steps; rev=true) && allunique(directional.fd_steps) ||
        throw(ArgumentError("directional.fd_steps must be unique and decreasing"))

    shooting_keys = ("maximum_hopf_points", "phase_fractions", "amplitude_factors",
        "period_factors", "ode_abstol", "ode_reltol", "shooting_atol",
        "validation_atol", "amplitude_atol", "phase_speed_atol", "floquet_atol",
        "refinement_factor", "min_period", "max_period", "maxiters", "ode_maxiters",
        "samples")
    shooting_table = require_exact_keys(raw["shooting"], shooting_keys, "shooting")
    shooting_options = PeriodicOrbitOptions(;
        (Symbol(key) => (key in ("maxiters", "ode_maxiters", "samples") ?
            positive_integer(shooting_table[key], "shooting.$key") :
            finite_number(shooting_table[key], "shooting.$key"; positive=true))
            for key in shooting_keys if !(key in ("maximum_hopf_points", "phase_fractions",
                "amplitude_factors", "period_factors")))...)
    shooting = (maximum_hopf_points=positive_integer(shooting_table["maximum_hopf_points"],
            "shooting.maximum_hopf_points"),
        phase_fractions=number_vector(shooting_table["phase_fractions"],
            "shooting.phase_fractions"; nonnegative=true),
        amplitude_factors=number_vector(shooting_table["amplitude_factors"],
            "shooting.amplitude_factors"; positive=true),
        period_factors=number_vector(shooting_table["period_factors"],
            "shooting.period_factors"; positive=true), shooting_options)
    all(value -> value < 1, shooting.phase_fractions) && allunique(shooting.phase_fractions) ||
        throw(ArgumentError("shooting phase fractions must be unique in [0,1)"))

    independent_keys = ("tighter_factor", "tight_samples", "jacobian_steps",
        "jacobian_atol", "poincare_steps", "poincare_intervals",
        "poincare_plateau_atol", "multiplier_atol", "orbit_equivalence_atol",
        "period_rtol", "minimum_phase_seeds", "minimum_period_seeds",
        "winding_center_atol")
    independent_table = require_exact_keys(raw["independent"], independent_keys,
        "independent")
    independent = (tighter_factor=finite_number(independent_table["tighter_factor"],
            "independent.tighter_factor"; positive=true),
        tight_samples=positive_integer(independent_table["tight_samples"],
            "independent.tight_samples"),
        jacobian_steps=number_vector(independent_table["jacobian_steps"],
            "independent.jacobian_steps"; positive=true),
        jacobian_atol=finite_number(independent_table["jacobian_atol"],
            "independent.jacobian_atol"; positive=true),
        poincare_steps=number_vector(independent_table["poincare_steps"],
            "independent.poincare_steps"; positive=true),
        poincare_intervals=positive_integer(independent_table["poincare_intervals"],
            "independent.poincare_intervals"),
        poincare_plateau_atol=finite_number(independent_table["poincare_plateau_atol"],
            "independent.poincare_plateau_atol"; positive=true),
        multiplier_atol=finite_number(independent_table["multiplier_atol"],
            "independent.multiplier_atol"; positive=true),
        orbit_equivalence_atol=finite_number(independent_table["orbit_equivalence_atol"],
            "independent.orbit_equivalence_atol"; positive=true),
        period_rtol=finite_number(independent_table["period_rtol"],
            "independent.period_rtol"; positive=true),
        minimum_phase_seeds=positive_integer(independent_table["minimum_phase_seeds"],
            "independent.minimum_phase_seeds"),
        minimum_period_seeds=positive_integer(independent_table["minimum_period_seeds"],
            "independent.minimum_period_seeds"),
        winding_center_atol=finite_number(independent_table["winding_center_atol"],
            "independent.winding_center_atol"; positive=true))

    smoke_table = require_exact_keys(raw["smoke"], ("maximum_seeds", "path_steps",
        "curve_steps", "shooting"), "smoke")
    smoke_table["shooting"] isa Bool || throw(ArgumentError("smoke.shooting must be Boolean"))
    smoke = (maximum_seeds=positive_integer(smoke_table["maximum_seeds"],
            "smoke.maximum_seeds"),
        path_steps=positive_integer(smoke_table["path_steps"], "smoke.path_steps"),
        curve_steps=positive_integer(smoke_table["curve_steps"], "smoke.curve_steps"),
        shooting=smoke_table["shooting"])

    config = (; raw, seed_policy, fixed, bounds, path_axes, curve_pairs, root_grids,
        equilibrium_options, stability_options, topology, topology_options, hopf_options,
        augmented, curve_continuation, directional, shooting, independent, smoke)
    return require_canonical ? _canonical_check(config) : config
end

function _verify_checksums(root)
    path = joinpath(root, "checksums.toml")
    isfile(path) || throw(ArgumentError("seed artifact lacks checksums.toml"))
    table = TOML.parsefile(path)
    require_exact_keys(table, ("schema_version", "algorithm", "files"),
        "seed checksums")
    table["schema_version"] === 1 || throw(ArgumentError("seed checksum schema mismatch"))
    table["algorithm"] == "SHA-256" || throw(ArgumentError("seed checksum algorithm mismatch"))
    files = table["files"]
    files isa AbstractDict || throw(ArgumentError("seed checksum files must be a table"))
    actual = String[]
    for (directory, _, names) in walkdir(root), name in names
        relative = relpath(joinpath(directory, name), root)
        relative == "checksums.toml" || push!(actual, relative)
    end
    sort!(actual)
    expected = sort!(String.(collect(keys(files))))
    actual == expected || throw(ArgumentError("seed checksum coverage mismatch; extra=" *
        join(setdiff(actual, expected), ",") * "; missing=" *
        join(setdiff(expected, actual), ",")))
    for relative in expected
        Evidence.file_hash(joinpath(root, relative)) == files[relative] ||
            throw(ArgumentError("seed checksum mismatch: $relative"))
    end
    return Evidence.file_hash(path)
end

function verify_seed_artifact(path::AbstractString, config; require_canonical=true)
    root = abspath(path)
    isdir(root) || throw(ArgumentError("seed artifact must be a directory"))
    required = ("metadata.toml", "config.toml", "samples.csv", "searches.csv",
        "equilibria.csv", "checksums.toml")
    all(name -> isfile(joinpath(root, name)), required) ||
        throw(ArgumentError("seed artifact is missing required files"))
    checksum_hash = _verify_checksums(root)
    metadata = TOML.parsefile(joinpath(root, "metadata.toml"))
    get(metadata, "schema_version", nothing) === 1 ||
        throw(ArgumentError("seed metadata schema mismatch"))
    get(metadata, "experiment", nothing) == config.seed_policy.experiment ||
        throw(ArgumentError("seed artifact experiment mismatch"))
    get(metadata, "execution_success", false) === true ||
        throw(ArgumentError("seed artifact did not complete successfully"))
    get(metadata, "smoke", true) === false ||
        throw(ArgumentError("seed artifact must not be a smoke run"))
    isempty(get(metadata, "failed_contexts", ["missing"])) ||
        throw(ArgumentError("seed artifact retains failed contexts"))
    get(metadata, "git_status_porcelain", "unavailable") == "" ||
        throw(ArgumentError("seed artifact source provenance is dirty or unavailable"))
    if require_canonical
        get(metadata, "git_revision", nothing) == config.seed_policy.accepted_revision ||
            throw(ArgumentError("seed artifact revision mismatch"))
        get(metadata, "extension_run", false) === true ||
            throw(ArgumentError("seed artifact did not run the canonical Halton extension"))
        get(metadata, "last_halton_index", nothing) == config.seed_policy.last_halton_index ||
            throw(ArgumentError("seed artifact stopped before Halton index 4096"))
        get(metadata, "parameter_cells", nothing) == config.seed_policy.parameter_cells ||
            throw(ArgumentError("seed artifact parameter-cell count mismatch"))
    end
    archived = Tetrastability.load_config(joinpath(root, "config.toml");
        require_canonical=require_canonical)
    if require_canonical
        archived.halton.initial_points == config.seed_policy.initial_halton_points ||
            throw(ArgumentError("seed initial Halton schedule mismatch"))
        archived.halton.maximum_points == config.seed_policy.maximum_halton_points ||
            throw(ArgumentError("seed maximum Halton schedule mismatch"))
        [plane.name for plane in archived.planes] == config.seed_policy.planes ||
            throw(ArgumentError("seed plane schedule mismatch"))
    end
    return (; root, metadata, archived, checksum_hash,
        config_hash=Evidence.file_hash(joinpath(root, "config.toml")),
        source_hashes=get(metadata, "source_sha256", Dict{String,Any}()))
end

function _rowdict(path, key)
    result = Dict{String,NamedTuple}()
    for row in CSV.File(path)
        result[string(getproperty(row, key))] = NamedTuple(row)
    end
    return result
end

function _float_bits(value)
    return string(reinterpret(UInt64, Float64(value)); base=16, pad=16)
end

function canonical_proposal_direction(direction)
    names = propertynames(direction)
    isempty(names) && throw(ArgumentError("proposal direction must not be empty"))
    all(name -> name in SEARCH_AXES, names) ||
        throw(ArgumentError("proposal direction contains an unknown axis"))
    values = Float64[]
    for axis in SEARCH_AXES
        if hasproperty(direction, axis)
            value = getproperty(direction, axis)
            value isa Real && !(value isa Bool) && isfinite(value) ||
                throw(ArgumentError(
                    "proposal direction must contain finite real values"))
            push!(values, Float64(value))
        else
            push!(values, 0.0)
        end
    end
    magnitude = norm(values)
    isfinite(magnitude) && magnitude > 0 ||
        throw(ArgumentError("proposal direction must have nonzero finite norm"))
    values ./= magnitude
    first_nonzero = findfirst(!iszero, values)
    values[first_nonzero] < 0 && (values .*= -1)
    values = [iszero(value) ? 0.0 : value for value in values]
    return NamedTuple{SEARCH_AXES}(Tuple(values))
end

function hypothesis_hash(parameters, fixed, source_hash, config_hash, parent, method;
    state=nothing, direction=nothing)
    io = IOBuffer()
    for name in SEARCH_AXES
        print(io, name, "=", _float_bits(getproperty(parameters, name)), ";")
    end
    for name in propertynames(fixed)
        print(io, name, "=", _float_bits(getproperty(fixed, name)), ";")
    end
    if state !== nothing
        length(state) == 2 || throw(ArgumentError("hypothesis state must contain E and I"))
        print(io, "E=", _float_bits(state[1]), ";I=", _float_bits(state[2]), ";")
    end
    if direction !== nothing
        canonical = canonical_proposal_direction(direction)
        for axis in SEARCH_AXES
            print(io, "direction_", axis, "=", _float_bits(getproperty(canonical, axis)), ";")
        end
    end
    print(io, "source=", source_hash, ";config=", config_hash,
        ";parent=", parent, ";method=", method)
    return bytes2hex(SHA.sha256(take!(io)))
end

function normalized_seed_manifest(seed_artifact, config; maximum=nothing)
    samples = _rowdict(joinpath(seed_artifact.root, "samples.csv"), :cell_id)
    searches = collect(CSV.File(joinpath(seed_artifact.root, "searches.csv")))
    equilibria = collect(CSV.File(joinpath(seed_artifact.root, "equilibria.csv")))
    source_hash = get(seed_artifact.source_hashes, "scripts/run_tetrastability_search.jl",
        seed_artifact.checksum_hash)
    manifest = NamedTuple[]
    for search in searches
        string(search.condition) == "failure_of_inhibition" || continue
        search.status == "completed" || continue
        search.discovered_equilibria == 7 || continue
        search.attracting_equilibria == 3 || continue
        search.saddle_equilibria == 3 || continue
        search.repelling_equilibria == 1 || continue
        haskey(samples, string(search.cell_id)) || continue
        root_rows = filter(row -> string(row.search_id) == string(search.search_id), equilibria)
        central = filter(row -> string(row.stability) == "Repelling", root_rows)
        length(central) == 1 || continue
        cell = samples[string(search.cell_id)]
        parameters = NamedTuple{SEARCH_AXES}(Tuple(Float64(getproperty(cell, axis))
            for axis in SEARCH_AXES))
        parent = "seed-artifact:" * string(search.search_id)
        identifier = hypothesis_hash(parameters, config.fixed, string(source_hash),
            seed_artifact.config_hash, parent, "seven_root_seed")
        push!(manifest, (; hypothesis_id=identifier, parent_hypothesis=parent,
            cell_id=string(search.cell_id), search_id=string(search.search_id), parameters,
            central_state=(Float64(only(central).E), Float64(only(central).I))))
    end
    sort!(manifest; by=item -> (Tuple(getproperty(item.parameters, axis)
        for axis in SEARCH_AXES), item.search_id))
    unique_manifest = NamedTuple[]
    seen = Set{NTuple{5,UInt64}}()
    for item in manifest
        key = Tuple(reinterpret(UInt64, getproperty(item.parameters, axis)) for axis in SEARCH_AXES)
        key in seen && continue
        push!(seen, key)
        push!(unique_manifest, item)
    end
    isnothing(maximum) || resize!(unique_manifest, min(length(unique_manifest), maximum))
    return unique_manifest
end

function model_at(config, parameters)
    fixed = config.fixed
    excitatory = PopulationParameters(timescale=fixed.tau_e,
        response=LogisticResponse(slope=fixed.excitatory_slope,
            threshold=fixed.theta_e))
    inhibitory = PopulationParameters(timescale=fixed.tau_e * fixed.tau_ratio,
        response=FailureOfInhibitionResponse(slope=fixed.inhibitory_slope,
            onset_threshold=fixed.theta_on, failure_threshold=parameters.theta_off))
    coupling = PointCoupling(e_to_e=parameters.e_to_e, i_to_e=parameters.i_to_e,
        e_to_i=parameters.e_to_i, i_to_i=parameters.i_to_i)
    return PointModelParameters(excitatory=excitatory, inhibitory=inhibitory,
        coupling=coupling, drive=NoDrive())
end

replace_parameter(parameters, axis, value) = merge(parameters, NamedTuple{(axis,)}((value,)))

function trace_at(model, state)
    jacobian = zeros(Float64, 2, 2)
    point_jacobian!(jacobian, state, model, 0.0)
    return jacobian[1, 1] + jacobian[2, 2]
end

function equilibrium_refinements(model, config)
    searches = EquilibriumSearchResult[]
    for grid in config.root_grids
        seeds = Figure5b.refinement_seeds(model, grid)
        push!(searches, find_equilibria(model; seeds,
            options=config.equilibrium_options,
            stability_options=config.stability_options))
    end
    return Tuple(searches)
end

function _minimum_separation(equilibria)
    length(equilibria) < 2 && return Inf
    return minimum(norm(equilibria[left].state .- equilibria[right].state)
        for left in 1:(length(equilibria) - 1) for right in (left + 1):length(equilibria))
end

function _unique_mapping(reference, candidate, tolerance)
    length(reference) == length(candidate) || return nothing
    distances = [norm(left.state .- right.state) for left in reference, right in candidate]
    mapping = Int[]
    for row in axes(distances, 1)
        matches = findall(value -> value <= tolerance, @view distances[row, :])
        length(matches) == 1 || return nothing
        push!(mapping, only(matches))
    end
    length(unique(mapping)) == length(mapping) || return nothing
    all(column -> count(value -> value <= tolerance, @view distances[:, column]) == 1,
        axes(distances, 2)) || return nothing
    return mapping
end

function _root_quality(search, equilibrium, config; neutral=false)
    residual = zeros(Float64, 2)
    balance_jacobian = zeros(Float64, 2, 2)
    ode_jacobian = zeros(Float64, 2, 2)
    point_balance!(residual, equilibrium.state, search.frozen_model, 0.0)
    point_balance_jacobian!(balance_jacobian, equilibrium.state, search.frozen_model, 0.0)
    point_jacobian!(ode_jacobian, equilibrium.state, search.frozen_model, 0.0)
    quality = maximum(abs, residual) <= config.topology.residual_atol &&
        maximum(abs, balance_jacobian .- equilibrium.balance_jacobian) <=
            config.topology.jacobian_atol && !equilibrium.near_singular
    values = eigvals(ode_jacobian)
    if neutral
        quality &= abs(sum(values)) <= config.topology.neutral_trace_atol
        quality &= real(det(ode_jacobian)) > config.topology.neutral_determinant_atol
        quality &= maximum(abs, imag.(values)) > config.topology.spectral_margin
    elseif equilibrium.stability.classification == Attracting
        quality &= all(real.(values) .< -config.topology.spectral_margin)
    elseif equilibrium.stability.classification == Repelling
        quality &= all(real.(values) .> config.topology.spectral_margin)
    elseif equilibrium.stability.classification == Saddle
        parts = sort(real.(values))
        quality &= first(parts) < -config.topology.spectral_margin < last(parts)
    else
        quality = false
    end
    return (; quality, residual=maximum(abs, residual), balance_jacobian, ode_jacobian,
        trace=real(sum(values)), determinant=real(det(ode_jacobian)), eigenvalues=values)
end

"""Fail-closed seven-root gate specialized to the neutral Hopf center."""
function neutral_hopf_topology(searches, hopf_state, config)
    searches isa Tuple && length(searches) == 3 ||
        throw(ArgumentError("neutral topology requires three searches"))
    reasons = Symbol[]
    all(search -> FailureOfInhibition2025._same_search_context(first(searches), search),
        searches) || push!(reasons, :model_context_mismatch)
    all(FailureOfInhibition2025._matches_refinement_schedule(search, grid)
        for (search, grid) in zip(searches, ROOT_GRIDS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(search -> isempty(search.unresolved_nearby), searches) ||
        push!(reasons, :unresolved_nearby_roots)
    all(search -> length(search.equilibria) == 7, searches) ||
        push!(reasons, :root_count_mismatch)
    minimum_separation = minimum(_minimum_separation(search.equilibria) for search in searches)
    minimum_separation >= config.topology.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)
    tracks = NamedTuple[]
    central_index = nothing
    if all(search -> length(search.equilibria) == 7, searches)
        reference = searches[1].equilibria
        mappings = [_unique_mapping(reference, searches[index].equilibria,
            config.topology.coordinate_match_atol) for index in 2:3]
        any(isnothing, mappings) && push!(reasons, :ambiguous_root_matching)
        if all(!isnothing, mappings)
            distances = [norm(root.state .- hopf_state) for root in reference]
            nearest = argmin(distances)
            distances[nearest] <= config.topology.coordinate_match_atol ||
                push!(reasons, :central_root_not_matched)
            central_index = nearest
            for index in eachindex(reference)
                roots = (reference[index], searches[2].equilibria[mappings[1][index]],
                    searches[3].equilibria[mappings[2][index]])
                qualities = [_root_quality(searches[j], roots[j], config;
                    neutral=index == nearest) for j in 1:3]
                all(row -> row.quality, qualities) || push!(reasons,
                    index == nearest ? :central_neutral_quality_failure : :outer_root_quality_failure)
                push!(tracks, (; root_track=index,
                    states=Tuple(Tuple(root.state) for root in roots),
                    classifications=Tuple(root.stability.classification for root in roots),
                    traces=Tuple(row.trace for row in qualities),
                    determinants=Tuple(row.determinant for row in qualities)))
            end
            for refinement in 1:3
                classifications = [track.classifications[refinement] for track in tracks]
                outer = [classification for (index, classification) in enumerate(classifications)
                    if index != nearest]
                count(==(Attracting), outer) == 3 || push!(reasons, :outer_attractor_count)
                count(==(Saddle), outer) == 3 || push!(reasons, :outer_saddle_count)
                classifications[nearest] == StabilityUnresolved ||
                    push!(reasons, :central_not_neutral)
            end
            central = searches[3].equilibria[mappings[2][nearest]]
            jacobian = zeros(Float64, 2, 2)
            point_balance_jacobian!(jacobian, central.state, searches[3].frozen_model, 0.0)
            if abs(jacobian[1, 2]) <= config.topology.slope_atol ||
                    abs(jacobian[2, 2]) <= config.topology.slope_atol
                push!(reasons, :unresolved_nullcline_slope)
            else
                slopes = (-jacobian[1, 1] / jacobian[1, 2],
                    -jacobian[2, 1] / jacobian[2, 2])
                all(value -> isfinite(value) && value > config.topology.slope_atol,
                    slopes) || push!(reasons, :nonrising_nullcline_arm)
            end
        end
    end
    unique!(reasons)
    return (; qualified=isempty(reasons), reasons, central_index, tracks,
        minimum_separation)
end

function augmented_residual(config, base_parameters, axis, z)
    parameters = replace_parameter(base_parameters, axis, z[3])
    model = model_at(config, parameters)
    residual = similar(z, 3)
    balance = view(residual, 1:2)
    point_balance!(balance, view(z, 1:2), model, zero(eltype(z)))
    jacobian = zeros(eltype(z), 2, 2)
    point_jacobian!(jacobian, view(z, 1:2), model, zero(eltype(z)))
    residual[3] = jacobian[1, 1] + jacobian[2, 2]
    return residual
end

function _damped_newton(residual_function, seed, options)
    candidate = Float64.(seed)
    last_residual = fill(NaN, length(candidate))
    for iteration in 1:options.maxiters
        residual = try Float64.(residual_function(candidate)) catch error
            return (; success=false, candidate, residual=last_residual, residual_norm=Inf,
                status="fallback_exception", error=Evidence.error_record(error))
        end
        last_residual = residual
        residual_norm = all(isfinite, residual) ? maximum(abs, residual) : Inf
        residual_norm <= options.residual_atol && return (; success=true, candidate,
            residual, residual_norm, status="fallback_converged", error=nothing)
        all(isfinite, residual) || break
        jacobian = try ForwardDiff.jacobian(residual_function, candidate) catch error
            return (; success=false, candidate, residual, residual_norm,
                status="fallback_jacobian_exception", error=Evidence.error_record(error))
        end
        scale = max(1.0, maximum(abs, jacobian))
        accepted = false
        for damping in scale^2 .* (1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
            step = try
                -(jacobian' * jacobian + damping * I) \ (jacobian' * residual)
            catch
                continue
            end
            all(isfinite, step) || continue
            alpha = 1.0
            while alpha >= 2.0^-20
                trial = candidate .+ alpha .* step
                trial_residual = try Float64.(residual_function(trial)) catch; fill(NaN, length(candidate)) end
                if all(isfinite, trial_residual) && maximum(abs, trial_residual) < residual_norm
                    candidate = trial
                    accepted = true
                    break
                end
                alpha /= 2
            end
            accepted && break
        end
        accepted || break
    end
    residual = try Float64.(residual_function(candidate)) catch; last_residual end
    residual_norm = all(isfinite, residual) ? maximum(abs, residual) : Inf
    return (; success=residual_norm <= options.residual_atol, candidate, residual,
        residual_norm, status="fallback_unresolved", error=nothing)
end

function _solve_square(residual_function, seed, options)
    f! = (residual, z, _) -> copyto!(residual, residual_function(z))
    j! = (matrix, z, _) -> copyto!(matrix, ForwardDiff.jacobian(residual_function, z))
    problem = NonlinearProblem(NonlinearFunction(f!; jac=j!), Float64.(seed), nothing)
    primary_error = nothing
    solution = try
        solve(problem, SimpleTrustRegion(); abstol=options.solver_abstol,
            reltol=options.solver_reltol, maxiters=options.maxiters)
    catch error
        error isa InterruptException && rethrow()
        primary_error = Evidence.error_record(error)
        nothing
    end
    solution === nothing && return merge(_damped_newton(residual_function, seed, options),
        (; primary_error))
    candidate = Float64.(solution.u)
    residual = try
        Float64.(residual_function(candidate))
    catch
        fill(NaN, length(candidate))
    end
    residual_norm = all(isfinite, residual) ? maximum(abs, residual) : Inf
    success = successful_retcode(solution.retcode) && all(isfinite, candidate) &&
        residual_norm <= options.residual_atol
    success && return (; success, candidate, residual, residual_norm,
        status=string(solution.retcode), error=nothing, primary_error)
    fallback = _damped_newton(residual_function, candidate, options)
    return merge(fallback, (; primary_error=Dict("type" => "primary_unresolved",
        "message" => "retcode=$(solution.retcode), residual_norm=$residual_norm")))
end

"""Injection-friendly square solve of balance plus trace along one parameter."""
function solve_augmented_trace_zero(residual_function, seed;
    solver_abstol=1e-11, solver_reltol=1e-10, residual_atol=1e-8, maxiters=100)
    options = (; solver_abstol=finite_number(solver_abstol, "solver_abstol"; positive=true),
        solver_reltol=finite_number(solver_reltol, "solver_reltol"; positive=true),
        residual_atol=finite_number(residual_atol, "residual_atol"; positive=true),
        maxiters=positive_integer(maxiters, "maxiters"))
    length(seed) == 3 && all(value -> value isa Real && isfinite(value), seed) ||
        throw(ArgumentError("augmented seed must contain three finite numbers"))
    result = _solve_square(residual_function, seed, options)
    rank = if result.success
        values = svdvals(ForwardDiff.jacobian(residual_function, result.candidate))
        minimum(values)
    else
        0.0
    end
    return merge(result, (; minimum_singular_value=rank))
end

function augmented_trace_zero_eligibility(result, bounds, bracket, rank_atol)
    bracket === nothing || (length(bracket) == 2 && all(isfinite, bracket) &&
        bracket[1] <= bracket[2]) || throw(ArgumentError(
        "augmented bracket must contain two ordered finite bounds"))
    in_bounds = result.success && bounds[1] <= result.candidate[3] <= bounds[2]
    in_bracket = result.success && (bracket === nothing ||
        bracket[1] <= result.candidate[3] <= bracket[2])
    full_rank = result.success && result.minimum_singular_value > rank_atol
    return (; success=result.success && in_bounds && in_bracket && full_rank,
        in_bounds, in_bracket, full_rank)
end

function solve_augmented_trace_zero(config, parameters, axis, seed; bracket=nothing)
    result = solve_augmented_trace_zero(z -> augmented_residual(config, parameters, axis, z),
        seed; solver_abstol=config.augmented.solver_abstol,
        solver_reltol=config.augmented.solver_reltol,
        residual_atol=config.augmented.residual_atol,
        maxiters=config.augmented.maxiters)
    eligibility = augmented_trace_zero_eligibility(result, config.bounds[axis], bracket,
        config.augmented.rank_atol)
    return merge(result, eligibility)
end

function _equilibrium_step(config, parameters, state)
    model = model_at(config, parameters)
    solved = solve_equilibrium(model, collect(state); options=config.equilibrium_options,
        stability_options=config.stability_options)
    accepted = solved.attempt.validation == AdmissibleCandidate &&
        solved.attempt.residual_norm <= config.equilibrium_options.residual_atol &&
        !solved.attempt.near_singular
    return (; accepted, solved, model,
        state=Tuple(solved.attempt.candidate),
        trace=accepted ? trace_at(model, solved.attempt.candidate) : NaN)
end

function _axis_root_quality(search, equilibrium, config)
    residual = zeros(Float64, 2)
    jacobian = zeros(Float64, 2, 2)
    point_balance!(residual, equilibrium.state, search.frozen_model, 0.0)
    point_balance_jacobian!(jacobian, equilibrium.state, search.frozen_model, 0.0)
    return maximum(abs, residual) <= config.topology.residual_atol &&
        maximum(abs, jacobian .- equilibrium.balance_jacobian) <=
            config.topology.jacobian_atol && !equilibrium.near_singular &&
        all(isfinite, jacobian)
end

function _axis_root_tracks(searches, config)
    reasons = Symbol[]
    searches isa Tuple && length(searches) == 3 ||
        return (; qualified=false, tracks=NamedTuple[],
            reasons=[:invalid_refinement_set], root_counts=Int[],
            unresolved_counts=Int[], minimum_separation=NaN)
    all(search -> search isa EquilibriumSearchResult, searches) ||
        return (; qualified=false, tracks=NamedTuple[],
            reasons=[:invalid_refinement_type], root_counts=Int[],
            unresolved_counts=Int[], minimum_separation=NaN)
    root_counts = [length(search.equilibria) for search in searches]
    unresolved_counts = [length(search.unresolved_nearby) for search in searches]
    all(search -> FailureOfInhibition2025._same_search_context(first(searches), search),
        searches) || push!(reasons, :model_context_mismatch)
    all(FailureOfInhibition2025._matches_refinement_schedule(search, grid)
        for (search, grid) in zip(searches, ROOT_GRIDS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(iszero, unresolved_counts) || push!(reasons, :unresolved_nearby_roots)
    all(==(7), root_counts) || push!(reasons, :root_count_mismatch)
    minimum_separation = minimum(_minimum_separation(search.equilibria)
        for search in searches)
    minimum_separation >= config.topology.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)
    tracks = NamedTuple[]
    if all(==(7), root_counts)
        reference = searches[1].equilibria
        mappings = [_unique_mapping(reference, searches[index].equilibria,
            config.topology.coordinate_match_atol) for index in 2:3]
        any(isnothing, mappings) && push!(reasons, :ambiguous_root_matching)
        if all(!isnothing, mappings)
            for index in eachindex(reference)
                roots = (reference[index], searches[2].equilibria[mappings[1][index]],
                    searches[3].equilibria[mappings[2][index]])
                all(_axis_root_quality(searches[refinement], roots[refinement], config)
                    for refinement in 1:3) || push!(reasons, :root_quality_failure)
                push!(tracks, (; root_track=index,
                    states=Tuple(Tuple(root.state) for root in roots)))
            end
        end
    end
    unique!(reasons)
    return (; qualified=isempty(reasons), tracks, reasons, root_counts,
        unresolved_counts, minimum_separation)
end

function unique_axis_track_match(previous_state, solved_state, track_states;
    coordinate_atol, displacement_tolerance)
    previous = Float64.(previous_state)
    solved = Float64.(solved_state)
    tracks = [[Float64.(state) for state in track] for track in track_states]
    reasons = Symbol[]
    length(previous) == 2 && length(solved) == 2 &&
        all(track -> length(track) == 3 && all(state -> length(state) == 2, track),
            tracks) || return (; matched=false, central_track=nothing,
                solved_track=nothing, branch_distances=Float64[],
                solved_distances=Float64[], nearest_gap=NaN,
                reasons=[:invalid_branch_state_dimension])
    all(isfinite, previous) && all(isfinite, solved) &&
        all(track -> all(state -> all(isfinite, state), track), tracks) ||
        return (; matched=false, central_track=nothing, solved_track=nothing,
            branch_distances=Float64[], solved_distances=Float64[], nearest_gap=NaN,
            reasons=[:nonfinite_branch_state])
    branch_distances = [maximum(norm(state .- previous) for state in track)
        for track in tracks]
    branch_matches = findall(distance -> distance <= displacement_tolerance,
        branch_distances)
    length(branch_matches) == 1 || push!(reasons,
        isempty(branch_matches) ? :central_track_lost : :central_track_ambiguous)
    central_track = length(branch_matches) == 1 ? only(branch_matches) : nothing
    ordered = sort(branch_distances)
    nearest_gap = length(ordered) >= 2 ? ordered[2] - ordered[1] : Inf
    nearest_gap > 2coordinate_atol || push!(reasons,
        :central_track_proximity_unresolved)
    central_track !== nothing && central_track != argmin(branch_distances) &&
        push!(reasons, :central_track_not_unique_nearest)
    solved_distances = [norm(last(track) .- solved) for track in tracks]
    solved_matches = findall(distance -> distance <= coordinate_atol, solved_distances)
    length(solved_matches) == 1 || push!(reasons,
        isempty(solved_matches) ? :solved_root_not_matched :
        :solved_root_match_ambiguous)
    solved_track = length(solved_matches) == 1 ? only(solved_matches) : nothing
    central_track !== nothing && solved_track !== nothing &&
        central_track != solved_track && push!(reasons, :solved_root_track_mismatch)
    return (; matched=isempty(reasons), central_track, solved_track,
        branch_distances, solved_distances, nearest_gap, reasons)
end

function _axis_trial_identity(config, parameters, previous_state, solved_state)
    searches = equilibrium_refinements(model_at(config, parameters), config)
    tracks = _axis_root_tracks(searches, config)
    track_states = [track.states for track in tracks.tracks]
    if !tracks.qualified
        return (; matched=false, central_track=nothing, solved_track=nothing,
            branch_distances=Float64[], solved_distances=Float64[], nearest_gap=NaN,
            reasons=tracks.reasons, root_counts=tracks.root_counts,
            unresolved_counts=tracks.unresolved_counts,
            minimum_root_separation=tracks.minimum_separation, track_states)
    end
    matched = unique_axis_track_match(previous_state, solved_state, track_states;
        coordinate_atol=config.topology.coordinate_match_atol,
        displacement_tolerance=config.augmented.state_step_atol)
    return merge(matched, (; root_counts=tracks.root_counts,
        unresolved_counts=tracks.unresolved_counts,
        minimum_root_separation=tracks.minimum_separation, track_states))
end

function _empty_axis_identity(reason)
    return (; matched=false, central_track=nothing, solved_track=nothing,
        branch_distances=Float64[], solved_distances=Float64[], nearest_gap=NaN,
        reasons=[reason], root_counts=Int[], unresolved_counts=Int[],
        minimum_root_separation=NaN, track_states=Tuple[])
end

"""Continue one central equilibrium in both directions and retain every failure."""
function continue_axis_path(config, seed, axis; smoke=false,
    equilibrium_step_function=_equilibrium_step,
    augmented_solver=solve_augmented_trace_zero,
    identity_check_function=_axis_trial_identity)
    bounds = config.bounds[axis]
    extent = bounds[2] - bounds[1]
    initial_step = config.augmented.initial_step_fraction * extent
    minimum_step = config.augmented.minimum_step_fraction * extent
    maximum_step = config.augmented.maximum_step_fraction * extent
    max_steps = smoke ? config.smoke.path_steps : config.augmented.max_steps
    points, attempts, brackets, locations = NamedTuple[], NamedTuple[], NamedTuple[], NamedTuple[]
    start = try
        equilibrium_step_function(config, seed.parameters, seed.central_state)
    catch error
        error isa InterruptException && rethrow()
        record = Evidence.error_record(error)
        return (; points, attempts=[(; direction=0, attempt=1, step=0.0,
            retry=0, parameter=getproperty(seed.parameters, axis), accepted=false,
            status="seed_equilibrium_exception", candidate_E=NaN, candidate_I=NaN,
            residual_norm=Inf, solver_status="exception", reasons="",
            identity_central_track=0, identity_solved_track=0,
            identity_nearest_gap=NaN, identity_branch_distances=Float64[],
            identity_solved_distances=Float64[], identity_root_counts="",
            identity_unresolved_counts="", identity_minimum_root_separation=NaN,
            identity_track_states=Tuple[],
            error_type=record["type"], error_message=record["message"])],
            brackets, locations, termination="seed_equilibrium_failed")
    end
    start.accepted || return (; points, attempts=[(; direction=0, attempt=1, step=0.0,
        retry=0, parameter=getproperty(seed.parameters, axis), accepted=false,
        status="seed_equilibrium_failed", candidate_E=start.state[1],
        candidate_I=start.state[2], residual_norm=start.solved.attempt.residual_norm,
        solver_status=string(start.solved.attempt.solver_status),
        reasons=join(string.(start.solved.attempt.reasons), ";"),
        identity_central_track=0, identity_solved_track=0,
        identity_nearest_gap=NaN, identity_branch_distances=Float64[],
        identity_solved_distances=Float64[], identity_root_counts="",
        identity_unresolved_counts="", identity_minimum_root_separation=NaN,
        identity_track_states=Tuple[], error_type="",
        error_message="")],
        brackets, locations, termination="seed_equilibrium_failed")
    push!(points, (; direction=0, point=0, parameter=getproperty(seed.parameters, axis),
        E=start.state[1], I=start.state[2], trace=start.trace,
        classification=string(start.solved.stability.classification)))
    terminations = String[]
    for direction in (-1, 1)
        destination = direction < 0 ? bounds[1] : bounds[2]
        parameter = getproperty(seed.parameters, axis)
        state = start.state
        previous_trace = start.trace
        step = initial_step
        retries = 0
        accepted_points = 0
        termination = "maximum_steps"
        while accepted_points < max_steps
            if parameter == destination
                termination = "parameter_bound"
                break
            end
            target = direction < 0 ? max(destination, parameter - step) :
                min(destination, parameter + step)
            trial_parameters = replace_parameter(seed.parameters, axis, target)
            trial = try
                equilibrium_step_function(config, trial_parameters, state)
            catch error
                error isa InterruptException && rethrow()
                (; accepted=false, state=state, trace=NaN,
                    solved=nothing, model=nothing, error=Evidence.error_record(error))
            end
            identity_error = nothing
            identity = if trial.accepted
                try
                    identity_check_function(config, trial_parameters, state, trial.state)
                catch error
                    error isa InterruptException && rethrow()
                    identity_error = Evidence.error_record(error)
                    _empty_axis_identity(:root_identity_exception)
                end
            else
                _empty_axis_identity(:equilibrium_rejected)
            end
            accepted = trial.accepted && identity.matched
            solver_reasons = trial.solved === nothing ? Symbol[] :
                Symbol.(trial.solved.attempt.reasons)
            combined_reasons = unique(vcat(solver_reasons, identity.reasons))
            push!(attempts, (; direction, attempt=length(attempts) + 1,
                parameter=target, step, retry=retries, accepted,
                status=!trial.accepted ? "equilibrium_failed" :
                    identity.matched ? "accepted" : "root_identity_failed",
                candidate_E=trial.state[1], candidate_I=trial.state[2],
                residual_norm=trial.solved === nothing ? Inf :
                    trial.solved.attempt.residual_norm,
                solver_status=trial.solved === nothing ? "exception" :
                    string(trial.solved.attempt.solver_status),
                reasons=join(string.(combined_reasons), ";"),
                identity_central_track=something(identity.central_track, 0),
                identity_solved_track=something(identity.solved_track, 0),
                identity_nearest_gap=identity.nearest_gap,
                identity_branch_distances=identity.branch_distances,
                identity_solved_distances=identity.solved_distances,
                identity_root_counts=join(identity.root_counts, ";"),
                identity_unresolved_counts=join(identity.unresolved_counts, ";"),
                identity_minimum_root_separation=identity.minimum_root_separation,
                identity_track_states=identity.track_states,
                error_type=identity_error !== nothing ? identity_error["type"] :
                    hasproperty(trial, :error) ? trial.error["type"] : "",
                error_message=identity_error !== nothing ? identity_error["message"] :
                    hasproperty(trial, :error) ? trial.error["message"] : ""))
            if !accepted
                retries += 1
                step /= 2
                if step < minimum_step || retries > config.augmented.max_retries
                    termination = retries > config.augmented.max_retries ?
                        "maximum_retries" : "minimum_step_after_failure"
                    break
                end
                continue
            end
            retries = 0
            accepted_points += 1
            push!(points, (; direction, point=accepted_points, parameter=target,
                E=trial.state[1], I=trial.state[2], trace=trial.trace,
                classification=string(trial.solved.stability.classification)))
            if isfinite(previous_trace) && previous_trace * trial.trace <= 0
                bracket = (; direction, lower=min(parameter, target), upper=max(parameter, target),
                    first_trace=previous_trace, second_trace=trial.trace)
                push!(brackets, bracket)
                augmented_seed = [(state[1] + trial.state[1]) / 2,
                    (state[2] + trial.state[2]) / 2, (parameter + target) / 2]
                solved = augmented_solver(config, seed.parameters, axis,
                    augmented_seed; bracket=(bracket.lower, bracket.upper))
                retained_error = solved.error === nothing &&
                    hasproperty(solved, :primary_error) ? solved.primary_error : solved.error
                candidate_parameters = solved.success ? replace_parameter(seed.parameters,
                    axis, solved.candidate[3]) : seed.parameters
                push!(locations, (; axis, direction, success=solved.success,
                    E=solved.candidate[1], I=solved.candidate[2],
                    parameter=solved.candidate[3], residual_norm=solved.residual_norm,
                    minimum_singular_value=solved.minimum_singular_value,
                    in_bounds=solved.in_bounds, in_bracket=solved.in_bracket,
                    full_rank=solved.full_rank,
                    parameters=candidate_parameters, status=solved.status,
                    error_type=retained_error === nothing ? "" : retained_error["type"],
                    error_message=retained_error === nothing ? "" : retained_error["message"]))
            end
            parameter, state, previous_trace = target, trial.state, trial.trace
            if target == destination
                termination = "parameter_bound"
                break
            end
            step = min(maximum_step, step * 1.25)
        end
        push!(terminations, "$direction:$termination")
    end
    return (; points, attempts, brackets, locations, termination=join(terminations, ";"))
end

axis_branch_exhausted(termination, unresolved_locations) =
    termination == "-1:parameter_bound;1:parameter_bound" &&
    iszero(unresolved_locations)

axis_path_exhausted(result) = axis_branch_exhausted(result.termination,
    count(location -> !location.success, result.locations))

function curve_residual(config, base_parameters, pair, z)
    parameters = replace_parameter(replace_parameter(base_parameters, pair[1], z[3]),
        pair[2], z[4])
    model = model_at(config, parameters)
    residual = similar(z, 3)
    point_balance!(view(residual, 1:2), view(z, 1:2), model, zero(eltype(z)))
    jacobian = zeros(eltype(z), 2, 2)
    point_jacobian!(jacobian, view(z, 1:2), model, zero(eltype(z)))
    residual[3] = jacobian[1, 1] + jacobian[2, 2]
    return residual
end

function _curve_scales(config, pair)
    return [config.curve_continuation.state_scales...,
        config.bounds[pair[1]][2] - config.bounds[pair[1]][1],
        config.bounds[pair[2]][2] - config.bounds[pair[2]][1]]
end

function _curve_tangent(residual_function, point, scales, rank_atol;
    orientation=nothing)
    jacobian = ForwardDiff.jacobian(residual_function, point)
    decomposition = svd(jacobian * Diagonal(scales); full=true)
    length(decomposition.S) == 3 && minimum(decomposition.S) > rank_atol || return nothing
    scaled = Vector{Float64}(decomposition.V[:, end])
    scaled ./= norm(scaled)
    tangent = scales .* scaled
    if orientation === nothing
        first_nonzero = findfirst(value -> abs(value) > 100eps(Float64), scaled)
        !isnothing(first_nonzero) && scaled[first_nonzero] < 0 &&
            (scaled .*= -1; tangent .*= -1)
    elseif dot(scaled, orientation ./ scales) < 0
        scaled .*= -1
        tangent .*= -1
    end
    return (; actual=tangent, scaled, jacobian,
        minimum_singular_value=minimum(decomposition.S))
end

function _curve_corrector(residual_function, predictor, tangent, scales, options)
    augmented = z -> vcat(residual_function(z),
        dot((z .- predictor) ./ scales, tangent.scaled))
    solver_options = (; solver_abstol=options.corrector_atol / 100,
        solver_reltol=options.corrector_atol / 10,
        residual_atol=options.corrector_atol, maxiters=options.max_corrector_iters)
    return _solve_square(augmented, predictor, solver_options)
end

"""Injection-friendly pseudo-arclength continuation of a three-equation curve."""
function continue_trace_zero_curve(residual_function, initial_point;
    scales=ones(4), bounds=fill((-Inf, Inf), 4), initial_step=0.01,
    minimum_step=1e-5, maximum_step=0.05, max_steps=100,
    max_corrector_iters=16, max_retries=8, corrector_atol=1e-8,
    rank_atol=1e-10, corrector_function=_curve_corrector)
    length(initial_point) == 4 && length(scales) == 4 && length(bounds) == 4 ||
        throw(ArgumentError("curve point, scales and bounds must have length four"))
    all(value -> value isa Real && isfinite(value), initial_point) ||
        throw(ArgumentError("curve initial point must be finite"))
    all(value -> value isa Real && isfinite(value) && value > 0, scales) ||
        throw(ArgumentError("curve scales must be finite and positive"))
    minimum_step <= initial_step <= maximum_step ||
        throw(ArgumentError("invalid curve step ordering"))
    options = (; corrector_atol, max_corrector_iters, rank_atol)
    points, attempts = NamedTuple[], NamedTuple[]
    initial = Float64.(initial_point)
    residual = Float64.(residual_function(initial))
    maximum(abs, residual) <= corrector_atol ||
        return (; points, attempts, termination="initial_residual", reversals=0)
    tangent = _curve_tangent(residual_function, initial, Float64.(scales), rank_atol)
    isnothing(tangent) && return (; points, attempts, termination="initial_rank", reversals=0)
    push!(points, (; point=0, coordinates=Tuple(initial), residual_norm=maximum(abs, residual),
        tangent=Tuple(tangent.actual), minimum_singular_value=tangent.minimum_singular_value))
    reversal_count = 0
    terminations = String[]
    for direction in (-1, 1)
        current = copy(initial)
        current_tangent = direction .* tangent.actual
        scaled_tangent = direction .* tangent.scaled
        step = initial_step
        termination = "maximum_steps"
        accepted = 0
        while accepted < max_steps
            succeeded = false
            corrected = nothing
            last_failure_out_of_bounds = false
            for retry in 0:max_retries
                predictor = current .+ step .* current_tangent
                corrected = corrector_function(residual_function, predictor,
                    (; actual=current_tangent, scaled=scaled_tangent), Float64.(scales), options)
                in_bounds = corrected.success && all(bounds[index][1] <=
                    corrected.candidate[index] <= bounds[index][2] for index in 1:4)
                last_failure_out_of_bounds = corrected.success && !in_bounds
                retained_error = corrected.error === nothing &&
                    hasproperty(corrected, :primary_error) ? corrected.primary_error :
                    corrected.error
                push!(attempts, (; direction, attempt=length(attempts) + 1,
                    point=accepted + 1, retry, step, accepted=corrected.success && in_bounds,
                    solver_success=corrected.success, in_bounds,
                    residual_norm=corrected.residual_norm, status=corrected.status,
                    error_type=retained_error === nothing ? "" : retained_error["type"],
                    error_message=retained_error === nothing ? "" : retained_error["message"]))
                if corrected.success && in_bounds
                    succeeded = true
                    break
                end
                step /= 2
                step < minimum_step && break
            end
            if !succeeded
                termination = last_failure_out_of_bounds && step < minimum_step ?
                    "parameter_bound" : step < minimum_step ? "minimum_step" :
                    "corrector_failure"
                break
            end
            new_tangent = _curve_tangent(residual_function, corrected.candidate,
                Float64.(scales), rank_atol; orientation=current_tangent)
            if isnothing(new_tangent)
                termination = "rank_failure"
                break
            end
            dot(new_tangent.actual[3:4], current_tangent[3:4]) < 0 &&
                (reversal_count += 1)
            accepted += 1
            push!(points, (; point=direction * accepted,
                coordinates=Tuple(corrected.candidate),
                residual_norm=corrected.residual_norm,
                tangent=Tuple(new_tangent.actual),
                minimum_singular_value=new_tangent.minimum_singular_value))
            current = corrected.candidate
            current_tangent = new_tangent.actual
            scaled_tangent = new_tangent.scaled
            step = min(maximum_step, step * 1.25)
        end
        push!(terminations, "$direction:$termination")
    end
    sort!(points; by=point -> point.point)
    return (; points, attempts, termination=join(terminations, ";"),
        reversals=reversal_count)
end

"""Find a deterministic scaled closest point on a three-equation curve."""
function closest_point_curve_seed(residual_function, seed, scales;
    solver_abstol=1e-11, solver_reltol=1e-10, residual_atol=1e-8, maxiters=100)
    length(seed) == 4 && length(scales) == 4 ||
        throw(ArgumentError("closest-point seed and scales must have length four"))
    initial = vcat(Float64.(seed), zeros(3))
    function kkt(value)
        z = view(value, 1:4)
        lambda = view(value, 5:7)
        residual = residual_function(z)
        jacobian = ForwardDiff.jacobian(residual_function, z)
        stationarity = (z .- seed) ./ (Float64.(scales) .^ 2) + jacobian' * lambda
        return vcat(residual, stationarity)
    end
    options = (; solver_abstol, solver_reltol, residual_atol, maxiters)
    solved = _solve_square(kkt, initial, options)
    retained_error = solved.error === nothing && hasproperty(solved, :primary_error) ?
        solved.primary_error : solved.error
    return (; success=solved.success, candidate=solved.candidate[1:4],
        multipliers=solved.candidate[5:7], residual_norm=solved.residual_norm,
        status=solved.status, error=retained_error)
end

function bounded_closest_curve_seed(residual_function, seed, scales, parameter_bounds;
    solver_abstol=1e-11, solver_reltol=1e-10, residual_atol=1e-8, maxiters=100)
    length(parameter_bounds) == 2 ||
        throw(ArgumentError("parameter_bounds must contain two pairs"))
    interior = closest_point_curve_seed(residual_function, seed, scales;
        solver_abstol, solver_reltol, residual_atol, maxiters)
    attempts = NamedTuple[]
    candidates = NamedTuple[]
    interior_in_bounds = interior.success && all(parameter_bounds[index][1] <=
        interior.candidate[index + 2] <= parameter_bounds[index][2] for index in 1:2)
    push!(attempts, (; method="interior_kkt", fixed_axis=0, fixed_value=NaN,
        success=interior.success, in_bounds=interior_in_bounds,
        residual_norm=interior.residual_norm, status=interior.status,
        error_type=interior.error === nothing ? "" : interior.error["type"],
        error_message=interior.error === nothing ? "" : interior.error["message"]))
    interior_in_bounds && push!(candidates, (; method="interior_kkt",
        point=interior.candidate))
    solver_options = (; solver_abstol, solver_reltol, residual_atol, maxiters)
    for fixed_index in 1:2, fixed_value in parameter_bounds[fixed_index]
        free_index = 3 - fixed_index
        initial = [seed[1], seed[2], seed[free_index + 2]]
        function edge_residual(value)
            point = similar(value, 4)
            fill!(point, zero(eltype(value)))
            point[1], point[2] = value[1], value[2]
            point[fixed_index + 2] = fixed_value
            point[free_index + 2] = value[3]
            return residual_function(point)
        end
        solved = _solve_square(edge_residual, initial, solver_options)
        point = Float64[solved.candidate[1], solved.candidate[2], 0.0, 0.0]
        point[fixed_index + 2] = fixed_value
        point[free_index + 2] = solved.candidate[3]
        in_bounds = solved.success && all(parameter_bounds[index][1] <=
            point[index + 2] <= parameter_bounds[index][2] for index in 1:2)
        method = "boundary_$(fixed_index)_$(_float_bits(fixed_value))"
        retained_error = solved.error === nothing && hasproperty(solved, :primary_error) ?
            solved.primary_error : solved.error
        push!(attempts, (; method, fixed_axis=fixed_index, fixed_value,
            success=solved.success, in_bounds, residual_norm=solved.residual_norm,
            status=solved.status,
            error_type=retained_error === nothing ? "" : retained_error["type"],
            error_message=retained_error === nothing ? "" : retained_error["message"]))
        in_bounds && push!(candidates, (; method, point))
    end
    sort!(candidates; by=item -> (sum(((item.point .- seed) ./ scales).^2), item.method))
    selected = isempty(candidates) ? nothing : first(candidates)
    return (; success=selected !== nothing,
        candidate=selected === nothing ? Float64.(seed) : selected.point,
        method=selected === nothing ? "unresolved" : selected.method,
        attempts, interior)
end

function _curve_seed_candidates(config, seed, pair, axis_locations)
    candidates = NamedTuple[]
    scales = _curve_scales(config, pair)
    for location in axis_locations
        location.success || continue
        location.axis in pair || continue
        parameters = location.parameters
        point = [location.E, location.I, getproperty(parameters, pair[1]),
            getproperty(parameters, pair[2])]
        push!(candidates, (; method="axis_solution", point,
            parent=string(location.axis)))
    end
    base_point = [seed.central_state[1], seed.central_state[2],
        getproperty(seed.parameters, pair[1]), getproperty(seed.parameters, pair[2])]
    closest = bounded_closest_curve_seed(z -> curve_residual(config, seed.parameters, pair, z),
        base_point, scales, [config.bounds[pair[1]], config.bounds[pair[2]]];
        solver_abstol=config.augmented.solver_abstol,
        solver_reltol=config.augmented.solver_reltol,
        residual_atol=config.augmented.residual_atol,
        maxiters=config.augmented.maxiters)
    closest.success && push!(candidates, (; method="closest_point:" * closest.method,
        point=closest.candidate,
        parent=seed.hypothesis_id))
    sort!(candidates; by=item -> (item.method, Tuple(item.point)))
    unique_candidates = NamedTuple[]
    for item in candidates
        any(other -> norm((item.point .- other.point) ./ scales) <=
            config.augmented.dedup_atol, unique_candidates) || push!(unique_candidates, item)
    end
    return (; candidates=unique_candidates, closest)
end

function continue_curve_pair(config, seed, pair, axis_locations; smoke=false)
    seeds = _curve_seed_candidates(config, seed, pair, axis_locations)
    branches = NamedTuple[]
    scales = _curve_scales(config, pair)
    bounds = [(-Inf, Inf), (-Inf, Inf), config.bounds[pair[1]], config.bounds[pair[2]]]
    max_steps = smoke ? config.smoke.curve_steps : config.curve_continuation.max_steps
    for (index, candidate) in enumerate(seeds.candidates)
        residual_function = z -> curve_residual(config, seed.parameters, pair, z)
        branch = continue_trace_zero_curve(residual_function, candidate.point;
            scales, bounds, initial_step=config.curve_continuation.initial_step,
            minimum_step=config.curve_continuation.minimum_step,
            maximum_step=config.curve_continuation.maximum_step,
            max_steps, max_corrector_iters=config.curve_continuation.max_corrector_iters,
            max_retries=config.curve_continuation.max_retries,
            corrector_atol=config.curve_continuation.corrector_atol,
            rank_atol=config.curve_continuation.rank_atol)
        push!(branches, (; branch=index, method=candidate.method, result=branch))
    end
    return (; closest=seeds.closest, branches)
end

curve_seed_search_unresolved(result) = !result.closest.success

function unresolved_curve_branch_record(curve_id, hypothesis_id, pair, result)
    curve_seed_search_unresolved(result) || return nothing
    return (; branch_id="$curve_id:unresolved", hypothesis_id,
        first_axis=string(pair[1]), second_axis=string(pair[2]),
        method="closest_point:unresolved", points=0,
        attempts=length(result.closest.attempts), reversals=0,
        termination="closest_seed_failed")
end

function _hopf_local_gate(model, state, config)
    state_values = collect(state)
    balance = zeros(Float64, 2)
    jacobian = zeros(Float64, 2, 2)
    point_balance!(balance, state_values, model, 0.0)
    point_jacobian!(jacobian, state_values, model, 0.0)
    values = eigvals(jacobian)
    singular = minimum(svdvals(jacobian))
    reasons = Symbol[]
    maximum(abs, balance) <= config.augmented.residual_atol ||
        push!(reasons, :balance_residual)
    abs(real(sum(values))) <= config.topology.neutral_trace_atol ||
        push!(reasons, :trace_residual)
    real(det(jacobian)) > config.topology.neutral_determinant_atol ||
        push!(reasons, :nonpositive_determinant)
    maximum(abs, imag.(values)) > config.topology.spectral_margin ||
        push!(reasons, :not_simple_imaginary)
    singular > config.augmented.rank_atol || push!(reasons, :singular_jacobian)
    return (; qualified=isempty(reasons), reasons, balance_residual=maximum(abs, balance),
        trace=real(sum(values)), determinant=real(det(jacobian)), eigenvalues=values,
        minimum_singular_value=singular)
end

function _equivalent_proposal_direction(left, right, tolerance)
    left_has_direction = hasproperty(left, :direction)
    right_has_direction = hasproperty(right, :direction)
    left_has_direction == right_has_direction || return false
    left_has_direction || return true
    left_canonical = canonical_proposal_direction(left.direction)
    right_canonical = canonical_proposal_direction(right.direction)
    left_values = Float64[getproperty(left_canonical, axis) for axis in SEARCH_AXES]
    right_values = Float64[getproperty(right_canonical, axis) for axis in SEARCH_AXES]
    return norm(left_values .- right_values) <= tolerance
end

function deduplicate_trace_zero_locations(locations, tolerance)
    priority = Dict("axis_solution" => 1, "curve_seed" => 2,
        "curve_endpoint" => 3, "curve_reversal" => 4, "curve_arclength" => 5)
    sorted = sort!(collect(locations); by=item -> (
        get(priority, item.proposal_stratum, 99), Tuple(item.state),
        Tuple(getproperty(item.parameters, axis) for axis in SEARCH_AXES), item.source_id))
    representatives = NamedTuple[]
    links = NamedTuple[]
    scales = [0.1, 0.1, 10.0, 12.0, 16.0, 10.0, 6.0]
    for item in sorted
        item_values = [item.state[1], item.state[2],
            (getproperty(item.parameters, axis) for axis in SEARCH_AXES)...]
        match = findfirst(other -> begin
            other_values = [other.state[1], other.state[2],
                (getproperty(other.parameters, axis) for axis in SEARCH_AXES)...]
            norm((item_values .- other_values) ./ scales) <= tolerance &&
                _equivalent_proposal_direction(item, other, tolerance)
        end, representatives)
        if isnothing(match)
            push!(representatives, item)
            match = length(representatives)
        end
        push!(links, (; source_id=item.source_id, representative=match,
            duplicate=item !== representatives[match]))
    end
    return (; representatives, links)
end

function select_shooting_proposals(locations, maximum)
    maximum isa Integer && !(maximum isa Bool) && maximum > 0 ||
        throw(ArgumentError("maximum shooting proposals must be a positive integer"))
    required_strata = Set(("axis_solution", "curve_seed", "curve_endpoint",
        "curve_reversal"))
    selected = Set(item.source_id for item in locations
        if item.proposal_stratum in required_strata)
    remaining = sort!([item for item in locations if !(item.source_id in selected)];
        by=item -> (item.proposal_group, item.arclength_index, item.source_id))
    slots = max(0, maximum - length(selected))
    if slots >= length(remaining)
        union!(selected, (item.source_id for item in remaining))
    elseif slots > 0
        indices = slots == 1 ? [cld(length(remaining), 2)] :
            unique(round.(Int, range(1, length(remaining); length=slots)))
        union!(selected, (remaining[index].source_id for index in indices))
    end
    rows = [(; source_id=item.source_id, proposal_stratum=item.proposal_stratum,
        selected=item.source_id in selected,
        reason=item.source_id in selected ? (item.proposal_stratum in required_strata ?
            "required_stratum" : "arclength_spaced_sample") :
            "not_selected_after_stratification") for item in locations]
    return (; selected, rows, requested_maximum=maximum,
        effective_count=length(selected), required_count=count(row ->
            row.reason == "required_stratum", rows))
end

"""Estimate fixed-ratio directional eigenvalue transversality from trace samples."""
function directional_transversality(trace_function, steps;
    plateau_atol=1e-4, plateau_rtol=0.05)
    derivatives = Float64[]
    rows = NamedTuple[]
    errors = NamedTuple[]
    for step in steps
        positive, positive_error = try
            Float64(trace_function(step)), nothing
        catch error
            NaN, Evidence.error_record(error)
        end
        negative, negative_error = try
            Float64(trace_function(-step)), nothing
        catch error
            NaN, Evidence.error_record(error)
        end
        positive_error === nothing || push!(errors, (; step=Float64(step), side="positive",
            error_type=positive_error["type"], message=positive_error["message"]))
        negative_error === nothing || push!(errors, (; step=Float64(step), side="negative",
            error_type=negative_error["type"], message=negative_error["message"]))
        derivative = isfinite(positive) && isfinite(negative) ?
            (positive - negative) / (2step) : NaN
        beta = derivative / 2
        push!(derivatives, beta)
        push!(rows, (; step=Float64(step), positive_trace=positive,
            negative_trace=negative, beta,
            positive_error_type=positive_error === nothing ? "" : positive_error["type"],
            positive_error_message=positive_error === nothing ? "" : positive_error["message"],
            negative_error_type=negative_error === nothing ? "" : negative_error["type"],
            negative_error_message=negative_error === nothing ? "" : negative_error["message"]))
    end
    finite = filter(isfinite, derivatives)
    center = isempty(finite) ? NaN : sum(finite) / length(finite)
    spread = isempty(finite) ? Inf : maximum(finite) - minimum(finite)
    tolerance = plateau_atol + plateau_rtol * (isfinite(center) ? abs(center) : 0.0)
    resolved = length(finite) == length(steps) && spread <= tolerance &&
        abs(center) > tolerance
    return (; resolved, beta=resolved ? center : NaN, values=derivatives, rows, errors,
        spread, tolerance)
end

function curve_parameter_normal(tangent, pair, config)
    length(tangent) == 4 || throw(ArgumentError("curve tangent must have length four"))
    parameter_tangent = Float64[tangent[3], tangent[4]]
    norm(parameter_tangent) > 0 || return nothing
    normal = [-parameter_tangent[2], parameter_tangent[1]]
    normal ./= norm(normal)
    first_nonzero = findfirst(value -> abs(value) > 100eps(Float64), normal)
    !isnothing(first_nonzero) && normal[first_nonzero] < 0 && (normal .*= -1)
    return NamedTuple{pair}(Tuple(normal))
end

function _offset_parameters(parameters, direction, delta)
    result = parameters
    for axis in propertynames(direction)
        result = replace_parameter(result, axis,
            getproperty(result, axis) + delta * getproperty(direction, axis))
    end
    return result
end

function _direction_in_bounds(config, parameters, direction, delta)
    shifted = _offset_parameters(parameters, direction, delta)
    return all(axis -> config.bounds[axis][1] <= getproperty(shifted, axis) <=
        config.bounds[axis][2], propertynames(direction))
end

function _directional_state_tangent(config, parameters, state, direction)
    step = minimum(config.directional.fd_steps)
    _direction_in_bounds(config, parameters, direction, step) &&
        _direction_in_bounds(config, parameters, direction, -step) ||
        return (; resolved=false, tangent=fill(NaN, 2), step,
            minimum_singular_value=NaN, reasons=[:predictor_out_of_bounds])
    model = model_at(config, parameters)
    state_vector = Float64.(collect(state))
    balance_jacobian = zeros(Float64, 2, 2)
    point_balance_jacobian!(balance_jacobian, state_vector, model, 0.0)
    singular_values = svdvals(balance_jacobian)
    minimum_singular_value = minimum(singular_values)
    isfinite(minimum_singular_value) &&
        minimum_singular_value > config.augmented.rank_atol ||
        return (; resolved=false, tangent=fill(NaN, 2), step,
            minimum_singular_value, reasons=[:predictor_singular_balance_jacobian])
    positive = zeros(Float64, 2)
    negative = zeros(Float64, 2)
    point_balance!(positive, state_vector,
        model_at(config, _offset_parameters(parameters, direction, step)), 0.0)
    point_balance!(negative, state_vector,
        model_at(config, _offset_parameters(parameters, direction, -step)), 0.0)
    parameter_derivative = (positive .- negative) ./ (2step)
    tangent = -(balance_jacobian \ parameter_derivative)
    all(isfinite, tangent) || return (; resolved=false, tangent, step,
        minimum_singular_value, reasons=[:predictor_nonfinite_tangent])
    return (; resolved=true, tangent, step, minimum_singular_value,
        reasons=Symbol[])
end

function _directional_root_tracks(searches, config)
    reasons = Symbol[]
    searches isa Tuple && length(searches) == 3 ||
        return (; qualified=false, tracks=NamedTuple[], reasons=[:invalid_refinement_set],
            root_counts=Int[], unresolved_counts=Int[], minimum_separation=NaN)
    all(search -> search isa EquilibriumSearchResult, searches) ||
        return (; qualified=false, tracks=NamedTuple[], reasons=[:invalid_refinement_type],
            root_counts=Int[], unresolved_counts=Int[], minimum_separation=NaN)
    root_counts = [length(search.equilibria) for search in searches]
    unresolved_counts = [length(search.unresolved_nearby) for search in searches]
    all(search -> FailureOfInhibition2025._same_search_context(first(searches), search),
        searches) || push!(reasons, :model_context_mismatch)
    all(FailureOfInhibition2025._matches_refinement_schedule(search, grid)
        for (search, grid) in zip(searches, ROOT_GRIDS)) ||
        push!(reasons, :refinement_schedule_mismatch)
    all(iszero, unresolved_counts) || push!(reasons, :unresolved_nearby_roots)
    all(==(7), root_counts) || push!(reasons, :root_count_mismatch)
    minimum_separation = minimum(_minimum_separation(search.equilibria)
        for search in searches)
    minimum_separation >= config.topology.minimum_root_separation ||
        push!(reasons, :insufficient_root_separation)
    tracks = NamedTuple[]
    if all(==(7), root_counts)
        reference = searches[1].equilibria
        mappings = [_unique_mapping(reference, searches[index].equilibria,
            config.topology.coordinate_match_atol) for index in 2:3]
        any(isnothing, mappings) && push!(reasons, :ambiguous_root_matching)
        if all(!isnothing, mappings)
            for index in eachindex(reference)
                roots = (reference[index], searches[2].equilibria[mappings[1][index]],
                    searches[3].equilibria[mappings[2][index]])
                qualities = [_root_quality(searches[j], roots[j], config)
                    for j in 1:3]
                all(row -> row.quality, qualities) ||
                    push!(reasons, :root_quality_failure)
                traces = Tuple(row.trace for row in qualities)
                all(isfinite, traces) &&
                    maximum(traces) - minimum(traces) <=
                        config.topology.neutral_trace_atol ||
                    push!(reasons, :refinement_trace_disagreement)
                push!(tracks, (; root_track=index,
                    states=Tuple(Tuple(root.state) for root in roots),
                    classifications=Tuple(root.stability.classification for root in roots),
                    residuals=Tuple(row.residual for row in qualities), traces))
            end
        end
    end
    unique!(reasons)
    return (; qualified=isempty(reasons), tracks, reasons, root_counts,
        unresolved_counts, minimum_separation)
end

function unique_directional_track_match(reference_states, predicted_state, solved_state,
    track_states; coordinate_atol, predictor_atol, displacement_tolerance)
    reference = [Float64.(state) for state in reference_states]
    predicted = Float64.(predicted_state)
    solved = Float64.(solved_state)
    states = [[Float64.(state) for state in track] for track in track_states]
    reasons = Symbol[]
    length(reference) == 3 && all(state -> length(state) == 2, reference) &&
        length(predicted) == 2 && length(solved) == 2 &&
        all(track -> length(track) == 3 && all(state -> length(state) == 2, track),
            states) ||
        return (; matched=false, predicted_track=nothing, solved_track=nothing,
            central_track=nothing, branch_distances=Float64[],
            predicted_distances=Float64[], solved_distances=Float64[], nearest_gap=NaN,
            reasons=[:invalid_branch_state_dimension])
    all(state -> all(isfinite, state), reference) && all(isfinite, predicted) &&
        all(isfinite, solved) &&
        all(track -> all(state -> all(isfinite, state), track), states) ||
        return (; matched=false, predicted_track=nothing, solved_track=nothing,
            central_track=nothing, branch_distances=Float64[],
            predicted_distances=Float64[], solved_distances=Float64[], nearest_gap=NaN,
            reasons=[:nonfinite_branch_state])
    branch_distances = [maximum(norm(track[index] .- reference[index])
        for index in 1:3) for track in states]
    branch_matches = findall(distance -> distance <= displacement_tolerance,
        branch_distances)
    length(branch_matches) == 1 || push!(reasons,
        isempty(branch_matches) ? :central_track_lost : :central_track_ambiguous)
    central_track = length(branch_matches) == 1 ? only(branch_matches) : nothing
    ordered = sort(branch_distances)
    nearest_gap = length(ordered) >= 2 ? ordered[2] - ordered[1] : Inf
    nearest_gap > 2coordinate_atol || push!(reasons,
        :central_track_proximity_unresolved)
    central_track !== nothing && central_track != argmin(branch_distances) &&
        push!(reasons, :central_track_not_unique_nearest)
    predicted_distances = [maximum(norm(state .- predicted) for state in track)
        for track in states]
    solved_distances = [norm(last(track) .- solved) for track in states]
    predicted_matches = findall(distance -> distance <= predictor_atol,
        predicted_distances)
    solved_matches = findall(distance -> distance <= coordinate_atol, solved_distances)
    length(predicted_matches) == 1 || push!(reasons,
        isempty(predicted_matches) ? :predicted_track_lost :
        :predicted_track_ambiguous)
    length(solved_matches) == 1 || push!(reasons,
        isempty(solved_matches) ? :solved_root_not_matched : :solved_root_match_ambiguous)
    predicted_track = length(predicted_matches) == 1 ? only(predicted_matches) : nothing
    solved_track = length(solved_matches) == 1 ? only(solved_matches) : nothing
    central_track !== nothing && predicted_track !== nothing &&
        central_track != predicted_track && push!(reasons, :predicted_track_mismatch)
    central_track !== nothing && solved_track !== nothing &&
        central_track != solved_track && push!(reasons, :solved_root_track_mismatch)
    return (; matched=isempty(reasons), central_track, predicted_track, solved_track,
        branch_distances, predicted_distances, solved_distances, nearest_gap, reasons)
end

function _directional_branch_match(config, reference_states, predicted_state,
    solved_state, searches, delta)
    tracks = _directional_root_tracks(searches, config)
    predictor_tolerance = max(config.topology.coordinate_match_atol,
        config.directional.state_match_factor * abs(delta)^2)
    displacement_tolerance = max(config.topology.coordinate_match_atol,
        config.directional.state_match_factor * abs(delta))
    if !tracks.qualified
        return (; matched=false, predicted_state=Float64.(predicted_state),
            predictor_tolerance, displacement_tolerance, central_track=nothing,
            predicted_track=nothing, solved_track=nothing, branch_distances=Float64[],
            predicted_distances=Float64[], solved_distances=Float64[],
            nearest_gap=NaN, matched_states=Tuple[], matched_traces=Float64[],
            matched_trace=NaN,
            reasons=tracks.reasons, root_counts=tracks.root_counts,
            unresolved_counts=tracks.unresolved_counts,
            minimum_root_separation=tracks.minimum_separation,
            track_states=Tuple[])
    end
    track_states = [track.states for track in tracks.tracks]
    matched = unique_directional_track_match(reference_states, predicted_state,
        solved_state, track_states;
        coordinate_atol=config.topology.coordinate_match_atol,
        predictor_atol=predictor_tolerance,
        displacement_tolerance=displacement_tolerance)
    selected = matched.central_track
    matched_states = selected === nothing ? Tuple[] : tracks.tracks[selected].states
    matched_traces = selected === nothing ? Float64[] :
        collect(tracks.tracks[selected].traces)
    matched_trace = matched.matched ? last(matched_traces) : NaN
    return merge(matched, (; predicted_state, predictor_tolerance,
        displacement_tolerance, matched_states, matched_traces, matched_trace,
        root_counts=tracks.root_counts, unresolved_counts=tracks.unresolved_counts,
        minimum_root_separation=tracks.minimum_separation, track_states))
end

_directional_sample_trace(branch, _solver_trace) =
    branch.matched && isfinite(branch.matched_trace) ? branch.matched_trace : NaN

function fixed_ratio_lyapunov_assessment(diagnostics, options)
    reasons = Symbol[]
    c1 = diagnostics.c1
    lyapunov_ad = diagnostics.lyapunov_ad
    lyapunov_fd = diagnostics.lyapunov_fd
    fd_values = diagnostics.lyapunov_fd_values
    frequency = diagnostics.frequency
    frequency isa Real && isfinite(frequency) && frequency > 0 ||
        push!(reasons, :lyapunov_frequency_unresolved)
    c1 isa Complex && isfinite(c1) || push!(reasons, :lyapunov_c1_unresolved)
    lyapunov_ad isa Real && isfinite(lyapunov_ad) ||
        push!(reasons, :lyapunov_ad_unresolved)
    lyapunov_fd isa Real && isfinite(lyapunov_fd) ||
        push!(reasons, :lyapunov_fd_unresolved)
    length(fd_values) == length(options.fd_steps) && all(isfinite, fd_values) ||
        push!(reasons, :lyapunov_fd_samples_unresolved)
    if length(fd_values) == length(options.fd_steps) && all(isfinite, fd_values)
        center = sum(fd_values) / length(fd_values)
        spread = maximum(fd_values) - minimum(fd_values)
        spread <= options.fd_plateau_atol + options.fd_plateau_rtol * abs(center) ||
            push!(reasons, :lyapunov_fd_no_plateau)
    end
    if lyapunov_ad isa Real && isfinite(lyapunov_ad) &&
            lyapunov_fd isa Real && isfinite(lyapunov_fd)
        abs(lyapunov_ad) > options.lyapunov_atol &&
            abs(lyapunov_fd) > options.lyapunov_atol ||
            push!(reasons, :lyapunov_near_zero)
        sign(lyapunov_ad) == sign(lyapunov_fd) ||
            push!(reasons, :lyapunov_sign_disagreement)
        isapprox(lyapunov_ad, lyapunov_fd;
            atol=options.lyapunov_agreement_atol,
            rtol=options.lyapunov_agreement_rtol) ||
            push!(reasons, :lyapunov_magnitude_disagreement)
    end
    if c1 isa Complex && isfinite(c1)
        abs(real(c1)) > options.lyapunov_atol ||
            push!(reasons, :degenerate_c1)
        if frequency isa Real && isfinite(frequency) && frequency > 0 &&
                lyapunov_ad isa Real && isfinite(lyapunov_ad)
            isapprox(real(c1) / frequency, lyapunov_ad;
                atol=options.lyapunov_agreement_atol,
                rtol=options.lyapunov_agreement_rtol) ||
                push!(reasons, :lyapunov_c1_ad_disagreement)
        end
    end
    unique!(reasons)
    return (; resolved=isempty(reasons), reasons, c1, lyapunov_ad, lyapunov_fd,
        lyapunov_fd_values=fd_values)
end

function assess_directional_hopf(config, parameters, state, direction, reference_track)
    norm_direction = sqrt(sum(getproperty(direction, axis)^2
        for axis in propertynames(direction)))
    isfinite(norm_direction) && norm_direction > 0 ||
        return (; resolved=false, beta=NaN, values=Float64[], rows=NamedTuple[],
            spread=Inf, tolerance=NaN, rho_squared_slope=NaN,
            c1_real=NaN, diagnostics=nothing, reasons=[:invalid_normal],
            execution_errors=NamedTuple[])
    normalized = NamedTuple{propertynames(direction)}(Tuple(getproperty(direction, axis) /
        norm_direction for axis in propertynames(direction)))
    model = model_at(config, parameters)
    diagnostics = hopf_diagnostics(model, collect(state); options=config.hopf_options)
    lyapunov = fixed_ratio_lyapunov_assessment(diagnostics, config.hopf_options)
    reference_states = hasproperty(reference_track, :states) ? reference_track.states : Tuple[]
    if length(reference_states) != 3 ||
            !all(root -> length(root) == 2 && all(isfinite, root), reference_states)
        return (; resolved=false, beta=NaN, values=Float64[], rows=NamedTuple[],
            errors=NamedTuple[], spread=Inf, tolerance=NaN, rho_squared_slope=NaN,
            c1_real=diagnostics.c1 === nothing ? NaN : real(diagnostics.c1),
            diagnostics, lyapunov, reasons=[:invalid_reference_central_track],
            direction=normalized, samples=NamedTuple[], reference_states,
            predictor=nothing, execution_errors=NamedTuple[])
    end
    predictor = _directional_state_tangent(config, parameters, state, normalized)
    if !predictor.resolved
        return (; resolved=false, beta=NaN, values=Float64[], rows=NamedTuple[],
            errors=NamedTuple[], spread=Inf, tolerance=NaN, rho_squared_slope=NaN,
            c1_real=diagnostics.c1 === nothing ? NaN : real(diagnostics.c1),
            diagnostics, lyapunov,
            reasons=unique(vcat([:directional_predictor_unresolved],
                predictor.reasons)), direction=normalized, samples=NamedTuple[],
            reference_states, predictor, execution_errors=NamedTuple[])
    end
    samples = NamedTuple[]
    function sampled_trace(delta)
        if !_direction_in_bounds(config, parameters, normalized, delta)
            push!(samples, (; delta, accepted=false, matched=false, E=NaN, I=NaN,
                state_distance=Inf, state_tolerance=NaN, predicted_E=NaN,
                predicted_I=NaN, predictor_tolerance=NaN, predicted_track=0,
                solved_track=0, minimum_root_separation=NaN, root_counts="",
                unresolved_counts="", trace=NaN, residual_norm=Inf,
                solver_status="out_of_bounds", reasons="out_of_bounds"))
            return NaN
        end
        shifted = _offset_parameters(parameters, normalized, delta)
        solved = _equilibrium_step(config, shifted, state)
        distance = norm(collect(solved.state) .- collect(state))
        tolerance = max(config.topology.coordinate_match_atol,
            config.directional.state_match_factor * abs(delta))
        branch = if solved.accepted
            searches = equilibrium_refinements(model_at(config, shifted), config)
            predicted_state = Float64.(state) .+ delta .* predictor.tangent
            _directional_branch_match(config, reference_states, predicted_state,
                solved.state, searches, delta)
        else
            predicted_state = Float64.(state) .+ delta .* predictor.tangent
            (; matched=false, predicted_state,
                predictor_tolerance=max(config.topology.coordinate_match_atol,
                    config.directional.state_match_factor * abs(delta)^2),
                displacement_tolerance=tolerance, central_track=nothing,
                predicted_track=nothing, solved_track=nothing,
                branch_distances=Float64[], predicted_distances=Float64[],
                solved_distances=Float64[], nearest_gap=NaN,
                matched_states=Tuple[], matched_traces=Float64[], matched_trace=NaN,
                track_states=Tuple[],
                minimum_root_separation=NaN, root_counts=Int[],
                unresolved_counts=Int[], reasons=[:equilibrium_step_rejected])
        end
        matched = solved.accepted && branch.matched
        sample_reasons = unique(vcat(Symbol.(solved.solved.attempt.reasons),
            branch.reasons))
        push!(samples, (; delta, accepted=solved.accepted, matched,
            E=solved.state[1], I=solved.state[2], state_distance=distance,
            state_tolerance=tolerance, predicted_E=branch.predicted_state[1],
            predicted_I=branch.predicted_state[2],
            predictor_tolerance=branch.predictor_tolerance,
            displacement_tolerance=branch.displacement_tolerance,
            central_track=something(branch.central_track, 0),
            predicted_track=something(branch.predicted_track, 0),
            solved_track=something(branch.solved_track, 0),
            nearest_gap=branch.nearest_gap,
            minimum_root_separation=branch.minimum_root_separation,
            root_counts=join(branch.root_counts, ";"),
            unresolved_counts=join(branch.unresolved_counts, ";"),
            matched_states=branch.matched_states,
            matched_traces=branch.matched_traces,
            branch_distances=branch.branch_distances,
            predicted_distances=branch.predicted_distances,
            solved_distances=branch.solved_distances,
            trace=matched ? branch.matched_trace : NaN,
            residual_norm=solved.solved.attempt.residual_norm,
            solver_status=string(solved.solved.attempt.solver_status),
            reasons=join(string.(sample_reasons), ";")))
        return _directional_sample_trace(branch, solved.trace)
    end
    estimate = directional_transversality(sampled_trace, config.directional.fd_steps;
        plateau_atol=config.directional.plateau_atol,
        plateau_rtol=config.directional.plateau_rtol)
    execution_errors = [(; stage="directional_trace_$(row.side)", attempt=0,
        error_type=row.error_type, message="step=$(row.step): $(row.message)")
        for row in estimate.errors]
    reasons = Symbol[]
    isempty(estimate.errors) || push!(reasons, :directional_trace_exception)
    all(sample -> sample.matched, samples) || push!(reasons, :directional_branch_mismatch)
    estimate.resolved || push!(reasons, :directional_transversality_unresolved)
    lyapunov.resolved || append!(reasons, lyapunov.reasons)
    c1_real = diagnostics.c1 === nothing ? NaN : real(diagnostics.c1)
    rho_squared_slope = isempty(reasons) ? -estimate.beta / c1_real : NaN
    return merge(estimate, (; resolved=isempty(reasons), rho_squared_slope,
        c1_real, diagnostics, lyapunov, reasons=unique(reasons),
        direction=normalized, samples, predictor,
        reference_states, execution_errors))
end

function _periodic_validation_config(config)
    return (; shooting_options=config.shooting.shooting_options,
        independent=config.independent)
end

function _tighter_periodic_options(config)
    return Figure5b.tighter_periodic_options(_periodic_validation_config(config))
end

function _shooting_seed_schedule(config, state, basis, radius, period; smoke=false)
    phases = smoke ? config.shooting.phase_fractions[1:1] : config.shooting.phase_fractions
    amplitudes = smoke ? config.shooting.amplitude_factors[1:1] :
        config.shooting.amplitude_factors
    periods = smoke ? config.shooting.period_factors[1:1] : config.shooting.period_factors
    schedule = NamedTuple[]
    for phase in phases, amplitude_factor in amplitudes, period_factor in periods
        angle = 2pi * phase
        displacement = 2 .* real.(basis.q .* (radius * amplitude_factor * cis(angle)))
        push!(schedule, (; phase, amplitude_factor, period_factor,
            initial_state=Float64.(state .+ displacement),
            period_guess=period * period_factor))
    end
    return schedule
end

function _mapped_shooting_center(topology)
    result = topology.topology
    track = result.central_track
    center = result.central_state
    track === nothing && return nothing
    center === nothing && return nothing
    track in eachindex(result.root_tracks) || return nothing
    mapped = result.root_tracks[track][3]
    length(center) == 2 && all(isfinite, center) && mapped == center || return nothing
    return Float64.(center)
end

function ratio44_orbit_acceptance_reasons(; validation_reasons=Symbol[],
    fixed_attractors, phase_seeds, period_seeds, minimum_phase_seeds=2,
    minimum_period_seeds=2, transverse_multiplier, floquet_atol=1e-4)
    reasons = Symbol.(validation_reasons)
    fixed_attractors == 3 || push!(reasons, :three_fixed_attractors_not_confirmed)
    length(unique(phase_seeds)) >= minimum_phase_seeds ||
        push!(reasons, :insufficient_phase_seeds)
    length(unique(period_seeds)) >= minimum_period_seeds ||
        push!(reasons, :insufficient_period_seeds)
    isfinite(real(transverse_multiplier)) && isfinite(imag(transverse_multiplier)) ||
        push!(reasons, :nonfinite_transverse_multiplier)
    isfinite(abs(transverse_multiplier)) &&
        abs(abs(transverse_multiplier) - 1) > floquet_atol ||
        push!(reasons, :near_neutral_periodic_stability)
    return unique(reasons)
end

function _shoot_orbit_set(config, model, topology, hopf_state, diagnostics,
    directional, signed_offset; smoke=false)
    attempts = NamedTuple[]
    execution_errors = NamedTuple[]
    diagnostics.frequency === nothing && return (; attempts, accepted=nothing,
        reasons=[:frequency_unresolved], execution_errors)
    rho2 = directional.rho_squared_slope * signed_offset
    rho2 > 0 || return (; attempts, accepted=nothing,
        reasons=[:normal_form_radius_not_positive], execution_errors)
    basis = Figure5b.hopf_basis(model_at(config,
        _offset_parameters(topology.parameters, directional.direction, -signed_offset)),
        hopf_state, diagnostics.frequency)
    radius = sqrt(rho2)
    center = _mapped_shooting_center(topology)
    center === nothing && return (; attempts, accepted=nothing,
        reasons=[:central_repeller_unresolved], execution_errors)
    schedule = _shooting_seed_schedule(config, center, basis, radius,
        diagnostics.linear_period; smoke)
    for (index, seed) in enumerate(schedule)
        result = try
            solve_periodic_orbit(model, seed.initial_state, seed.period_guess;
                options=config.shooting.shooting_options)
        catch error
            error isa InterruptException && rethrow()
            push!(attempts, (; attempt_id="shoot_$(lpad(index, 4, '0'))",
                phase=seed.phase, amplitude_factor=seed.amplitude_factor,
                period_factor=seed.period_factor, result=nothing,
                error=Evidence.error_record(error)))
            record = Evidence.error_record(error)
            push!(execution_errors, (; stage="shooting_attempt", attempt=index,
                error_type=record["type"], message=record["message"]))
            continue
        end
        push!(attempts, (; attempt_id="shoot_$(lpad(index, 4, '0'))",
            phase=seed.phase, amplitude_factor=seed.amplitude_factor,
            period_factor=seed.period_factor, result, error=nothing))
    end
    clusters = Figure5b.cluster_validated_attempts(attempts,
        _periodic_validation_config(config))
    cluster = Figure5b.qualifying_cluster(clusters, _periodic_validation_config(config))
    cluster === nothing && return (; attempts, accepted=nothing,
        reasons=[:multiple_seed_cluster_unresolved], execution_errors)
    baseline = first(cluster).result
    tighter = try
        solve_periodic_orbit(model, baseline.initial_state, baseline.period;
            options=_tighter_periodic_options(config))
    catch error
        error isa InterruptException && rethrow()
        record = Evidence.error_record(error)
        push!(execution_errors, (; stage="tight_replay", attempt=0,
            error_type=record["type"], message=record["message"]))
        return (; attempts, accepted=nothing, reasons=[:tight_replay_exception],
            tight_error=record, execution_errors)
    end
    validation = Figure5b.validate_orbit(tighter, baseline,
        topology.searches[3].equilibria, _periodic_validation_config(config))
    acceptance_reasons = ratio44_orbit_acceptance_reasons(
        validation_reasons=validation.reasons, fixed_attractors=3,
        phase_seeds=[attempt.phase for attempt in cluster],
        period_seeds=[attempt.period_factor for attempt in cluster],
        minimum_phase_seeds=config.independent.minimum_phase_seeds,
        minimum_period_seeds=config.independent.minimum_period_seeds,
        transverse_multiplier=tighter.transverse_multiplier,
        floquet_atol=config.shooting.shooting_options.floquet_atol)
    return (; attempts, accepted=isempty(acceptance_reasons) ?
        (; baseline, tighter, validation, cluster_size=length(cluster), radius,
            signed_offset) : nothing,
        reasons=acceptance_reasons, validation, execution_errors)
end

function qualify_and_shoot(config, location; smoke=false, allow_shooting=true)
    execution_errors = NamedTuple[]
    model = model_at(config, location.parameters)
    local_gate = _hopf_local_gate(model, location.state, config)
    local_gate.qualified || return (; local_gate, neutral=nothing, directional=nothing,
        offsets=NamedTuple[], accepted=NamedTuple[], reasons=local_gate.reasons,
        execution_errors)
    searches = try
        equilibrium_refinements(model, config)
    catch error
        error isa InterruptException && rethrow()
        record = Evidence.error_record(error)
        push!(execution_errors, (; stage="neutral_search", attempt=0,
            error_type=record["type"], message=record["message"]))
        return (; local_gate, neutral=nothing, directional=nothing,
            offsets=NamedTuple[], accepted=NamedTuple[], reasons=[:neutral_search_exception],
            error=record, execution_errors)
    end
    neutral = neutral_hopf_topology(searches, location.state, config)
    neutral.qualified || return (; local_gate, neutral, neutral_searches=searches,
        directional=nothing,
        offsets=NamedTuple[], accepted=NamedTuple[], reasons=neutral.reasons,
        execution_errors)
    neutral.central_index !== nothing &&
        neutral.central_index in eachindex(neutral.tracks) ||
        return (; local_gate, neutral, neutral_searches=searches, directional=nothing,
            offsets=NamedTuple[], accepted=NamedTuple[],
            reasons=[:invalid_reference_central_track], execution_errors)
    reference_track = neutral.tracks[neutral.central_index]
    directional = assess_directional_hopf(config, location.parameters, location.state,
        location.direction, reference_track)
    append!(execution_errors, directional.execution_errors)
    directional.resolved || return (; local_gate, neutral, neutral_searches=searches,
        directional,
        offsets=NamedTuple[], accepted=NamedTuple[], reasons=directional.reasons,
        execution_errors)
    offsets, accepted = NamedTuple[], NamedTuple[]
    do_shooting = allow_shooting && (!smoke || config.smoke.shooting)
    for sign in (-1, 1)
        delta = sign * config.directional.offset
        if !_direction_in_bounds(config, location.parameters, directional.direction, delta)
            push!(offsets, (; sign, delta, status="out_of_bounds", topology=nothing,
                shooting=nothing))
            continue
        end
        parameters = _offset_parameters(location.parameters, directional.direction, delta)
        offset_model = model_at(config, parameters)
        offset_searches = try
            equilibrium_refinements(offset_model, config)
        catch error
            error isa InterruptException && rethrow()
            record = Evidence.error_record(error)
            push!(execution_errors, (; stage="offset_search_$sign", attempt=0,
                error_type=record["type"], message=record["message"]))
            push!(offsets, (; sign, delta, status="search_exception", topology=nothing,
                shooting=nothing, error=record))
            continue
        end
        topology = classify_figure5b_topology(offset_searches;
            options=config.topology_options)
        wrapper = (; topology, searches=offset_searches, parameters)
        if !topology.qualified
            push!(offsets, (; sign, delta, status="ordinary_topology_failed",
                topology=wrapper, shooting=nothing))
            continue
        end
        shooting = do_shooting ? _shoot_orbit_set(config, offset_model, wrapper,
            location.state, directional.diagnostics, directional, delta; smoke) : nothing
        shooting === nothing || append!(execution_errors, shooting.execution_errors)
        push!(offsets, (; sign, delta, status=do_shooting ? "shooting_completed" :
            "shooting_disabled", topology=wrapper, shooting))
        if shooting !== nothing && shooting.accepted !== nothing
            push!(accepted, (; sign, delta, parameters, evidence=shooting.accepted))
        end
    end
    return (; local_gate, neutral, neutral_searches=searches, directional, offsets, accepted,
        reasons=isempty(accepted) ? [:no_validated_orbit] : Symbol[], execution_errors)
end

function _periodic_summary(result)
    result === nothing && return nothing
    return (; initial_state=Tuple(result.initial_state), period=result.period,
        amplitudes=Tuple(result.amplitudes), closure_residual=result.closure_residual,
        phase_residual=result.phase_residual, equation_residual=result.equation_residual,
        transverse_multiplier=result.transverse_multiplier,
        floquet_multipliers=result.floquet_multipliers,
        refinement_difference=result.refinement_difference,
        integration_success=result.integration_success,
        validation=result.validation, stability=result.stability, reasons=result.reasons,
        times=result.times, states=Tuple.(result.states))
end

function _shooting_summary(shooting)
    shooting === nothing && return nothing
    attempts = [(; attempt_id=attempt.attempt_id, phase=attempt.phase,
        amplitude_factor=attempt.amplitude_factor, period_factor=attempt.period_factor,
        result=_periodic_summary(attempt.result), error=attempt.error)
        for attempt in shooting.attempts]
    accepted = shooting.accepted === nothing ? nothing :
        (; baseline=_periodic_summary(shooting.accepted.baseline),
            tighter=_periodic_summary(shooting.accepted.tighter),
            validation=shooting.accepted.validation,
            cluster_size=shooting.accepted.cluster_size,
            radius=shooting.accepted.radius,
            signed_offset=shooting.accepted.signed_offset)
    return (; attempts, accepted, shooting.reasons)
end

function _qualification_record(location, result)
    offsets = NamedTuple[]
    for offset in result.offsets
        topology = offset.topology === nothing ? nothing :
            (; topology=offset.topology.topology,
                searches=[Evidence.context_record(search)
                    for search in offset.topology.searches],
                parameters=offset.topology.parameters)
        push!(offsets, (; sign=offset.sign, delta=offset.delta, status=offset.status,
            topology, shooting=_shooting_summary(offset.shooting),
            error=hasproperty(offset, :error) ? offset.error : nothing))
    end
    return (; source_id=location.source_id, parameters=location.parameters,
        state=location.state, direction=location.direction,
        local_gate=result.local_gate,
        neutral=hasproperty(result, :neutral) ? result.neutral : nothing,
        neutral_searches=hasproperty(result, :neutral_searches) ?
            [Evidence.context_record(search) for search in result.neutral_searches] : Any[],
        directional=hasproperty(result, :directional) ? result.directional : nothing,
        offsets, accepted_count=length(result.accepted), reasons=result.reasons,
        execution_errors=hasproperty(result, :execution_errors) ?
            result.execution_errors : NamedTuple[],
        error=hasproperty(result, :error) ? result.error : nothing)
end

function trace_zero_hypothesis_identity(location, fixed, source_hash, config_hash)
    direction = canonical_proposal_direction(location.direction)
    identifier = hypothesis_hash(location.parameters, fixed, source_hash, config_hash,
        location.parent_hypothesis, "trace_zero:$(location.method)";
        state=location.state, direction)
    record = Dict("hypothesis_id" => identifier,
        "parent_hypothesis" => location.parent_hypothesis,
        "method" => location.method, "parameters" => location.parameters,
        "state" => location.state, "direction" => direction,
        "source_id" => location.source_id)
    return (; identifier, direction, record)
end

function _write_hypothesis(output, identifier, record)
    path = joinpath(output, "contexts", "hypotheses", identifier * ".toml")
    ispath(path) && throw(ArgumentError("hypothesis record already exists: $identifier"))
    Evidence.write_toml(path, record)
    return path
end

function _archive_provenance(config_path, output, config, seed, seed_path, smoke)
    metadata = Evidence.archive_provenance(config_path, output)
    for relative in (joinpath("scripts", "run_coexistence_map.jl"),
        joinpath("scripts", "run_tetrastability_search.jl"),
        joinpath("scripts", "run_figure5b_protocol.jl"),
        joinpath("scripts", "run_figure5b_ratio44_search.jl"),
        joinpath("experiments", "tetrastability.toml"),
        joinpath("experiments", "figure5b_hopf.toml"))
        source = joinpath(REPOSITORY_ROOT, relative)
        isfile(source) || continue
        destination = joinpath(output, "source", relative)
        mkpath(dirname(destination))
        cp(source, destination; force=true)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
    merge!(metadata, Dict(
        "experiment" => "figure5b_ratio44_search",
        "purpose" => "finite targeted trace-zero and periodic-orbit search at tau_I/tau_E=4.4; no biological interpretation",
        "smoke" => smoke,
        "fixed_ratio" => config.fixed.tau_ratio,
        "zero_drive" => true,
        "seed_artifact_sha256" => seed.checksum_hash,
        "seed_artifact_config_sha256" => seed.config_hash,
        "seed_artifact_revision" => get(seed.metadata, "git_revision", "unavailable"),
        "seed_artifact_path_at_run" => abspath(seed_path),
        "seed_artifact_requirement" => "claim-grade discovery requires the independently retained upstream artifact; numerical replay uses this artifact's verified normalized manifest",
        "search_axes" => string.(SEARCH_AXES),
        "curve_pairs" => [[string(pair[1]), string(pair[2])] for pair in CURVE_PAIRS],
        "directional_transversality" => "fixed-ratio equilibrium-following derivative normal to the trace-zero set; never the timescale-ratio derivative",
        "normal_form_radius_law" => "rho_squared_slope=-beta_normal/real(c1)",
        "completeness" => CompletenessNotCertified,
        "absence_claim" => "not_permitted",
        "prevalence_claim" => "not_permitted",
        "biological_interpretation" => "not_assigned",
        "interruption_policy" => "partial artifacts retain completed contexts but are not resumable; restart into a new empty output directory",
        "replay_from_source" => "julia --project=source source/scripts/run_figure5b_ratio44_search.jl --config config.toml --replay-artifact ARTIFACT_DIRECTORY --output REPLAY_DIRECTORY" * (smoke ? " --smoke" : ""),
        "artifact_schema" => Dict("version" => 1,
            "seed_manifest" => "normalized unique seven-root seeds from the verified artifact",
            "seed_topologies" => "fresh 11/21/41 Figure-5b classifications",
            "axis" => "all path points, retries, brackets and augmented solves",
            "curves" => "all pseudo-arclength points, corrector attempts, reversals and terminations",
            "trace_zero" => "deduplicated locations with provenance-preserving links",
            "hopf" => "neutral topology, fixed-ratio directional transversality and local coefficient evidence",
            "shooting" => "gated attempts, tight validation and all rejected outcomes")))
    return metadata
end

function _initialize_output(output)
    mkpath(joinpath(output, "contexts", "hypotheses"))
    mkpath(joinpath(output, "contexts", "seed_topologies"))
    mkpath(joinpath(output, "contexts", "axis"))
    mkpath(joinpath(output, "contexts", "curves"))
    mkpath(joinpath(output, "contexts", "hopf"))
end

function _manifest_rows(manifest)
    return [(; hypothesis_id=item.hypothesis_id,
        parent_hypothesis=item.parent_hypothesis, cell_id=item.cell_id,
        search_id=item.search_id, e_to_e=item.parameters.e_to_e,
        i_to_e=item.parameters.i_to_e, e_to_i=item.parameters.e_to_i,
        i_to_i=item.parameters.i_to_i, theta_off=item.parameters.theta_off,
        central_E=item.central_state[1], central_I=item.central_state[2]) for item in manifest]
end

function _manifest_from_csv(path)
    manifest = NamedTuple[]
    for row in CSV.File(path)
        parameters = (e_to_e=Float64(row.e_to_e), i_to_e=Float64(row.i_to_e),
            e_to_i=Float64(row.e_to_i), i_to_i=Float64(row.i_to_i),
            theta_off=Float64(row.theta_off))
        push!(manifest, (; hypothesis_id=string(row.hypothesis_id),
            parent_hypothesis=string(row.parent_hypothesis), cell_id=string(row.cell_id),
            search_id=string(row.search_id), parameters,
            central_state=(Float64(row.central_E), Float64(row.central_I))))
    end
    return manifest
end

function scientific_provenance_assessment(; current_revision, status, head_name,
    accepted_revision, smoke=false, replay=false)
    reasons = Symbol[]
    smoke && push!(reasons, :smoke_mode)
    replay && push!(reasons, :normalized_manifest_replay)
    accepted_revision === nothing && push!(reasons, :accepted_revision_missing)
    unavailable(value) = !(value isa AbstractString) || startswith(value, "unavailable:")
    unavailable(current_revision) && push!(reasons, :revision_unavailable)
    unavailable(status) && push!(reasons, :status_unavailable)
    unavailable(head_name) && push!(reasons, :head_state_unavailable)
    !unavailable(status) && !isempty(status) && push!(reasons, :dirty_checkout)
    !unavailable(head_name) && head_name != "HEAD" && push!(reasons, :attached_head)
    accepted_revision !== nothing && !unavailable(current_revision) &&
        current_revision != accepted_revision && push!(reasons, :accepted_revision_mismatch)
    unique!(reasons)
    return (; eligible=isempty(reasons), reasons)
end

function combined_provenance_assessment(initial, final)
    reasons = unique(vcat(initial.reasons, final.reasons))
    return (; eligible=initial.eligible && final.eligible, reasons)
end

scientific_result_outcome(numerical_outcome, failures, provenance) =
    !isempty(failures) ? "incomplete_execution" :
    !provenance.eligible ? "evidence_ineligible" : numerical_outcome

scientific_acceptance_enabled(failures, provenance) =
    isempty(failures) && provenance.eligible

function finite_search_numerical_outcome(trace_zero_count, accepted_count,
    qualified_unattempted, unexhausted_branches, unresolved_seed_topologies)
    accepted_count > 0 && return "completed_with_validated_ratio44_orbit"
    unresolved = qualified_unattempted > 0 || unexhausted_branches > 0 ||
        unresolved_seed_topologies > 0
    trace_zero_count == 0 && !unresolved &&
        return "completed_no_trace_zero_solution"
    unresolved && return "completed_unresolved_targeted_search"
    return "completed_no_validated_ratio44_orbit"
end

function _canonical_destination(path::AbstractString)
    absolute = abspath(path)
    cursor = absolute
    suffix = String[]
    while !ispath(cursor) && !islink(cursor)
        parent = dirname(cursor)
        parent == cursor && throw(ArgumentError(
            "output destination has no existing filesystem ancestor"))
        pushfirst!(suffix, basename(cursor))
        cursor = parent
    end
    resolved = try
        realpath(cursor)
    catch error
        throw(ArgumentError("output destination ancestor cannot be resolved: " *
            sprint(showerror, error)))
    end
    !isempty(suffix) && !isdir(resolved) && throw(ArgumentError(
        "output destination ancestor is not a directory"))
    return normpath(joinpath(resolved, suffix...))
end

function _inside_directory(path, root)
    relative = relpath(path, root)
    return relative == "." || !(relative == ".." || startswith(relative, "../") ||
        startswith(relative, "..\\"))
end

function _has_symlink_component(path, root)
    relative = relpath(path, root)
    _inside_directory(path, root) || return false
    cursor = root
    for component in splitpath(relative)
        component in ("", ".") && continue
        cursor = joinpath(cursor, component)
        islink(cursor) && return true
        ispath(cursor) || break
    end
    return false
end

function output_destination_assessment(output_dir::AbstractString;
    repository_root::AbstractString=REPOSITORY_ROOT)
    requested_output = normpath(abspath(output_dir))
    requested_root = normpath(abspath(repository_root))
    lexical_inside = _inside_directory(requested_output, requested_root)
    if lexical_inside && _has_symlink_component(requested_output, requested_root)
        return (; output=requested_output, inside_checkout=true, git_ignored=false,
            eligible=false, reasons=[:output_symlink_component])
    end
    output = _canonical_destination(output_dir)
    root = _canonical_destination(repository_root)
    isdir(root) || throw(ArgumentError("repository root must be a directory"))
    inside_checkout = _inside_directory(output, root)
    if !inside_checkout
        return (; output, inside_checkout, git_ignored=nothing,
            eligible=true, reasons=Symbol[])
    end
    probe = joinpath(output, OUTPUT_IGNORE_PROBE)
    relative_probe = relpath(probe, root)
    git_ignored = try
        success(pipeline(`git --no-optional-locks -C $root check-ignore --quiet -- $relative_probe`;
            stdout=devnull, stderr=devnull))
    catch
        false
    end
    reasons = git_ignored ? Symbol[] : [:output_not_git_ignored]
    return (; output, inside_checkout, git_ignored,
        eligible=isempty(reasons), reasons)
end

function require_output_destination(output_dir::AbstractString;
    repository_root::AbstractString=REPOSITORY_ROOT)
    assessment = output_destination_assessment(output_dir; repository_root)
    assessment.eligible || throw(ArgumentError(
        "output inside the checkout must be git-ignored before the run starts"))
    return assessment
end

function require_output_outside_artifact(output::AbstractString,
    artifact_root::AbstractString)
    canonical_output = _canonical_destination(output)
    canonical_artifact = _canonical_destination(artifact_root)
    _inside_directory(canonical_output, canonical_artifact) && throw(ArgumentError(
        "output must be outside the seed/source artifact"))
    return (; output=canonical_output, artifact=canonical_artifact)
end

function verified_replay_manifest(path, config_path, smoke)
    root = abspath(path)
    isdir(root) || throw(ArgumentError("replay artifact must be a directory"))
    _verify_checksums(root)
    metadata_path = joinpath(root, "metadata.toml")
    manifest_path = joinpath(root, "seed_manifest.csv")
    isfile(metadata_path) && isfile(manifest_path) ||
        throw(ArgumentError("replay artifact lacks metadata or normalized manifest"))
    metadata = TOML.parsefile(metadata_path)
    get(metadata, "experiment", nothing) == "figure5b_ratio44_search" ||
        throw(ArgumentError("replay artifact experiment mismatch"))
    get(metadata, "execution_success", false) === true ||
        throw(ArgumentError("replay artifact execution was incomplete"))
    get(metadata, "smoke", nothing) === smoke ||
        throw(ArgumentError("replay smoke mode must match the source artifact"))
    get(metadata, "config_sha256", nothing) == Evidence.file_hash(config_path) ||
        throw(ArgumentError("replay configuration hash mismatch"))
    manifest = _manifest_from_csv(manifest_path)
    seed = (checksum_hash=String(metadata["seed_artifact_sha256"]),
        config_hash=String(metadata["seed_artifact_config_sha256"]),
        metadata=Dict("git_revision" => get(metadata, "seed_artifact_revision", "unavailable")),
        source_hashes=Dict{String,String}())
    return (; root, metadata, manifest, seed)
end

"""Run the finite fixed-ratio hypothesis program. A zero result is successful."""
function run_experiment(config_path::AbstractString,
    seed_artifact_path::Union{Nothing,AbstractString}, output_dir::AbstractString;
    smoke=false, require_canonical=true, replay_artifact=nothing,
    accepted_revision=nothing)
    config = load_config(config_path; require_canonical)
    replay_mode = replay_artifact !== nothing
    if replay_mode
        replay = verified_replay_manifest(replay_artifact, config_path, smoke)
        seed_artifact = replay.seed
        manifest = replay.manifest
        provenance_seed_path = replay.root
    else
        seed_artifact_path === nothing &&
            throw(ArgumentError("seed artifact is required outside replay mode"))
        seed_artifact = verify_seed_artifact(seed_artifact_path, config; require_canonical)
        manifest = normalized_seed_manifest(seed_artifact, config;
            maximum=smoke ? config.smoke.maximum_seeds : nothing)
        provenance_seed_path = seed_artifact_path
    end
    output_policy = require_output_destination(output_dir)
    output = output_policy.output
    require_output_outside_artifact(output, provenance_seed_path)
    ispath(output) && (!isdir(output) || !isempty(readdir(output))) &&
        throw(ArgumentError("output must be absent or an empty directory"))
    accepted_revision === nothing || (accepted_revision isa AbstractString &&
        occursin(r"^[0-9a-f]{40}$", accepted_revision)) ||
        throw(ArgumentError("accepted_revision must be a full lowercase revision hash"))
    current_revision = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    current_status = Evidence.git_output(REPOSITORY_ROOT, "status", "--porcelain=v1",
        "--untracked-files=all")
    head_name = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    provenance = scientific_provenance_assessment(current_revision=current_revision,
        status=current_status, head_name=head_name,
        accepted_revision=accepted_revision, smoke=smoke, replay=replay_mode)
    !smoke && !replay_mode && !provenance.eligible && throw(ArgumentError(
        "claim-grade run requires clean detached accepted revision: " *
        join(string.(provenance.reasons), ",")))
    _initialize_output(output)
    metadata = _archive_provenance(config_path, output, config, seed_artifact,
        provenance_seed_path, smoke)
    metadata["replay_from_normalized_manifest"] = replay_mode
    metadata["accepted_revision"] = something(accepted_revision, "not_supplied")
    metadata["head_name"] = head_name
    metadata["initial_git_revision"] = current_revision
    metadata["initial_git_status_porcelain"] = current_status
    metadata["initial_head_name"] = head_name
    metadata["initial_scientific_provenance_reasons"] = string.(provenance.reasons)
    metadata["output_inside_checkout"] = output_policy.inside_checkout
    metadata["output_git_ignored"] = output_policy.inside_checkout ?
        output_policy.git_ignored : "not_applicable"
    protocol_source_hash = Evidence.file_hash(joinpath(REPOSITORY_ROOT, "scripts",
        "run_figure5b_ratio44_search.jl"))
    protocol_config_hash = Evidence.file_hash(config_path)
    Evidence.write_rows(joinpath(output, "seed_manifest.csv"), _manifest_rows(manifest),
        (:hypothesis_id, :parent_hypothesis, :cell_id, :search_id, :e_to_e,
            :i_to_e, :e_to_i, :i_to_i, :theta_off, :central_E, :central_I))

    seed_rows, axis_branch_rows, axis_point_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    axis_attempt_rows, bracket_rows, curve_branch_rows = NamedTuple[], NamedTuple[], NamedTuple[]
    curve_point_rows, curve_attempt_rows, raw_locations = NamedTuple[], NamedTuple[], NamedTuple[]
    failures, accepted_rows = NamedTuple[], NamedTuple[]
    qualifying_seeds = 0
    unresolved_seed_topologies = 0

    for seed in manifest
        _write_hypothesis(output, seed.hypothesis_id, Dict(
            "hypothesis_id" => seed.hypothesis_id,
            "parent_hypothesis" => seed.parent_hypothesis,
            "method" => "fresh_seven_root_replay", "parameters" => seed.parameters,
            "source_hash" => get(seed_artifact.source_hashes,
                "scripts/run_tetrastability_search.jl", seed_artifact.checksum_hash),
            "config_hash" => seed_artifact.config_hash))
        model = model_at(config, seed.parameters)
        searches = try
            equilibrium_refinements(model, config)
        catch error
            error isa InterruptException && rethrow()
            unresolved_seed_topologies += 1
            push!(failures, (; context=seed.hypothesis_id, stage="seed_topology",
                error_type=string(typeof(error)), message=sprint(showerror, error)))
            push!(seed_rows, (; hypothesis_id=seed.hypothesis_id, qualified=false,
                unresolved=true, reasons="execution_exception", central_E=NaN,
                central_I=NaN))
            continue
        end
        Evidence.write_toml(joinpath(output, "contexts", "seed_topologies",
            seed.hypothesis_id * ".toml"), Dict("searches" =>
                [Evidence.context_record(search) for search in searches]))
        topology = classify_figure5b_topology(searches; options=config.topology_options)
        topology_unresolved = !topology.qualified
        topology_unresolved && (unresolved_seed_topologies += 1)
        push!(seed_rows, (; hypothesis_id=seed.hypothesis_id,
            qualified=topology.qualified, unresolved=topology_unresolved,
            reasons=join(string.(topology.reasons), ";"),
            central_E=topology.central_state === nothing ? NaN : topology.central_state[1],
            central_I=topology.central_state === nothing ? NaN : topology.central_state[2]))
        topology.qualified || continue
        qualifying_seeds += 1
        replayed_seed = merge(seed, (; central_state=Tuple(topology.central_state)))
        seed_axis_locations = NamedTuple[]
        for axis in config.path_axes
            branch_id = hypothesis_hash(seed.parameters, config.fixed,
                protocol_source_hash, protocol_config_hash,
                seed.hypothesis_id, "axis:$axis")
            result = try
                continue_axis_path(config, replayed_seed, axis; smoke)
            catch error
                error isa InterruptException && rethrow()
                push!(failures, (; context=branch_id, stage="axis:$axis",
                    error_type=string(typeof(error)), message=sprint(showerror, error)))
                continue
            end
            Evidence.write_toml(joinpath(output, "contexts", "axis", branch_id * ".toml"),
                Dict("axis" => axis, "parent_hypothesis" => seed.hypothesis_id,
                    "result" => result))
            push!(axis_branch_rows, (; branch_id, hypothesis_id=seed.hypothesis_id,
                axis=string(axis), points=length(result.points),
                attempts=length(result.attempts), brackets=length(result.brackets),
                locations=length(result.locations),
                unresolved_locations=count(location -> !location.success,
                    result.locations), termination=result.termination))
            append!(axis_point_rows, [(; branch_id, axis=string(axis), row...) for row in result.points])
            append!(axis_attempt_rows, [(; branch_id, axis=string(axis), row...) for row in result.attempts])
            append!(bracket_rows, [(; branch_id, axis=string(axis), row...) for row in result.brackets])
            for (index, location) in enumerate(result.locations)
                source_id = "$branch_id:location:$index"
                direction = NamedTuple{(axis,)}((1.0,))
                item = (; source_id, parent_hypothesis=seed.hypothesis_id,
                    method="axis", proposal_stratum="axis_solution",
                    proposal_group=branch_id, arclength_index=Float64(index),
                    parameters=location.parameters,
                    state=(location.E, location.I), direction, success=location.success,
                    residual_norm=location.residual_norm)
                push!(seed_axis_locations, merge(location, (; source_id)))
                location.success && push!(raw_locations, item)
            end
        end

        for pair in config.curve_pairs
            curve_id = hypothesis_hash(seed.parameters, config.fixed,
                protocol_source_hash, protocol_config_hash,
                seed.hypothesis_id, "curve:$(pair[1]):$(pair[2])")
            result = try
                continue_curve_pair(config, replayed_seed, pair, seed_axis_locations; smoke)
            catch error
                error isa InterruptException && rethrow()
                push!(failures, (; context=curve_id,
                    stage="curve:$(pair[1]):$(pair[2])",
                    error_type=string(typeof(error)), message=sprint(showerror, error)))
                continue
            end
            Evidence.write_toml(joinpath(output, "contexts", "curves", curve_id * ".toml"),
                Dict("pair" => pair, "parent_hypothesis" => seed.hypothesis_id,
                    "closest_point" => result.closest,
                    "branches" => [(; branch=branch.branch, method=branch.method,
                        result=branch.result) for branch in result.branches]))
            unresolved_curve = unresolved_curve_branch_record(curve_id,
                seed.hypothesis_id, pair, result)
            unresolved_curve === nothing || push!(curve_branch_rows, unresolved_curve)
            for branch in result.branches
                branch_id = "$curve_id:$(branch.branch)"
                push!(curve_branch_rows, (; branch_id, hypothesis_id=seed.hypothesis_id,
                    first_axis=string(pair[1]), second_axis=string(pair[2]),
                    method=branch.method, points=length(branch.result.points),
                    attempts=length(branch.result.attempts),
                    reversals=branch.result.reversals,
                    termination=branch.result.termination))
                branch_points = branch.result.points
                for (point_index, point) in enumerate(branch_points)
                    z = point.coordinates
                    parameters = replace_parameter(replace_parameter(seed.parameters,
                        pair[1], z[3]), pair[2], z[4])
                    normal = curve_parameter_normal(point.tangent, pair, config)
                    source_id = "$branch_id:point:$(point.point)"
                    current_parameter_tangent = collect(point.tangent[3:4])
                    previous_parameter_tangent = point_index > 1 ?
                        collect(branch_points[point_index - 1].tangent[3:4]) : Float64[]
                    parameter_reversal = point_index > 1 && any(index ->
                        current_parameter_tangent[index] * previous_parameter_tangent[index] < 0 ||
                        abs(current_parameter_tangent[index]) <= 1e-10,
                        eachindex(current_parameter_tangent))
                    proposal_stratum = point.point == 0 ? "curve_seed" :
                        point_index in (1, length(branch_points)) ? "curve_endpoint" :
                        parameter_reversal ? "curve_reversal" : "curve_arclength"
                    push!(curve_point_rows, (; branch_id, point=point.point,
                        E=z[1], I=z[2], first_parameter=z[3], second_parameter=z[4],
                        residual_norm=point.residual_norm,
                        minimum_singular_value=point.minimum_singular_value))
                    normal === nothing || push!(raw_locations, (; source_id,
                        parent_hypothesis=seed.hypothesis_id, method="curve",
                        proposal_stratum, proposal_group=branch_id,
                        arclength_index=Float64(point.point),
                        parameters, state=(z[1], z[2]), direction=normal,
                        success=true, residual_norm=point.residual_norm))
                end
                append!(curve_attempt_rows, [(; branch_id, row...) for row in branch.result.attempts])
            end
        end
    end

    deduplicated = deduplicate_trace_zero_locations(raw_locations,
        config.augmented.dedup_atol)
    proposals = select_shooting_proposals(deduplicated.representatives,
        config.shooting.maximum_hopf_points)
    proposal_reason = Dict(row.source_id => row for row in proposals.rows)
    location_rows = NamedTuple[]
    link_rows = deduplicated.links
    accepted_count = 0
    qualified_unattempted = 0
    for (index, location) in enumerate(deduplicated.representatives)
        identity = trace_zero_hypothesis_identity(location, config.fixed,
            protocol_source_hash, protocol_config_hash)
        identifier = identity.identifier
        _write_hypothesis(output, identifier, identity.record)
        candidate = merge(location, (; source_id=identifier))
        allow_shooting = location.source_id in proposals.selected
        proposal = proposal_reason[location.source_id]
        result = try
            qualify_and_shoot(config, candidate; smoke, allow_shooting)
        catch error
            error isa InterruptException && rethrow()
            push!(failures, (; context=identifier, stage="hopf_qualification",
                error_type=string(typeof(error)), message=sprint(showerror, error)))
            push!(location_rows, (; hypothesis_id=identifier, source_id=location.source_id,
                method=location.method, proposal_stratum=location.proposal_stratum,
                shooting_selected=allow_shooting, selection_reason=proposal.reason,
                qualified=false, accepted_orbits=0,
                reasons="execution_exception"))
            continue
        end
        Evidence.write_toml(joinpath(output, "contexts", "hopf", identifier * ".toml"),
            _qualification_record(candidate, result))
        for error in result.execution_errors
            push!(failures, (; context=identifier, stage=error.stage,
                error_type=error.error_type, message=error.message))
        end
        accepted_count += length(result.accepted)
        neutral_qualified = hasproperty(result, :neutral) && result.neutral !== nothing &&
            result.neutral.qualified
        neutral_qualified && !allow_shooting && (qualified_unattempted += 1)
        push!(location_rows, (; hypothesis_id=identifier, source_id=location.source_id,
            method=location.method, proposal_stratum=location.proposal_stratum,
            shooting_selected=allow_shooting, selection_reason=proposal.reason,
            qualified=neutral_qualified,
            accepted_orbits=length(result.accepted),
            reasons=join(string.(result.reasons), ";")))
        for orbit in result.accepted
            push!(accepted_rows, (; hypothesis_id=identifier, sign=orbit.sign,
                delta=orbit.delta, e_to_e=orbit.parameters.e_to_e,
                i_to_e=orbit.parameters.i_to_e, e_to_i=orbit.parameters.e_to_i,
                i_to_i=orbit.parameters.i_to_i, theta_off=orbit.parameters.theta_off,
                period=orbit.evidence.tighter.period,
                transverse_multiplier_real=real(orbit.evidence.tighter.transverse_multiplier),
                transverse_multiplier_imaginary=imag(orbit.evidence.tighter.transverse_multiplier)))
        end
    end

    Evidence.write_rows(joinpath(output, "seed_topologies.csv"), seed_rows,
        (:hypothesis_id, :qualified, :unresolved, :reasons, :central_E, :central_I))
    Evidence.write_rows(joinpath(output, "axis_branches.csv"), axis_branch_rows,
        (:branch_id, :hypothesis_id, :axis, :points, :attempts, :brackets, :locations,
            :unresolved_locations, :termination))
    Evidence.write_rows(joinpath(output, "axis_points.csv"), axis_point_rows,
        (:branch_id, :axis, :direction, :point, :parameter, :E, :I, :trace,
            :classification))
    Evidence.write_rows(joinpath(output, "axis_attempts.csv"), axis_attempt_rows,
        (:branch_id, :axis, :direction, :attempt, :parameter, :step, :retry,
            :accepted, :status, :candidate_E, :candidate_I, :residual_norm,
            :solver_status, :reasons, :identity_central_track,
            :identity_solved_track, :identity_nearest_gap, :identity_root_counts,
            :identity_unresolved_counts, :identity_minimum_root_separation,
            :error_type, :error_message))
    Evidence.write_rows(joinpath(output, "axis_brackets.csv"), bracket_rows,
        (:branch_id, :axis, :direction, :lower, :upper, :first_trace, :second_trace))
    Evidence.write_rows(joinpath(output, "curve_branches.csv"), curve_branch_rows,
        (:branch_id, :hypothesis_id, :first_axis, :second_axis, :method, :points,
            :attempts, :reversals, :termination))
    Evidence.write_rows(joinpath(output, "curve_points.csv"), curve_point_rows,
        (:branch_id, :point, :E, :I, :first_parameter, :second_parameter,
            :residual_norm, :minimum_singular_value))
    Evidence.write_rows(joinpath(output, "curve_attempts.csv"), curve_attempt_rows,
        (:branch_id, :direction, :attempt, :point, :retry, :step, :accepted,
            :solver_success, :in_bounds, :residual_norm, :status, :error_type,
            :error_message))
    Evidence.write_rows(joinpath(output, "trace_zero_locations.csv"), location_rows,
        (:hypothesis_id, :source_id, :method, :proposal_stratum, :shooting_selected,
            :selection_reason, :qualified, :accepted_orbits, :reasons))
    Evidence.write_rows(joinpath(output, "trace_zero_links.csv"), link_rows,
        (:source_id, :representative, :duplicate))
    Evidence.write_rows(joinpath(output, "shooting_proposals.csv"), proposals.rows,
        (:source_id, :proposal_stratum, :selected, :reason))
    Evidence.write_rows(joinpath(output, "accepted_orbits.csv"), accepted_rows,
        (:hypothesis_id, :sign, :delta, :e_to_e, :i_to_e, :e_to_i, :i_to_i,
            :theta_off, :period, :transverse_multiplier_real,
            :transverse_multiplier_imaginary))
    Evidence.write_rows(joinpath(output, "failures.csv"), failures,
        (:context, :stage, :error_type, :message))

    final_revision = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "HEAD")
    final_status = Evidence.git_output(REPOSITORY_ROOT, "status", "--porcelain=v1",
        "--untracked-files=all")
    final_head_name = Evidence.git_output(REPOSITORY_ROOT, "rev-parse", "--abbrev-ref", "HEAD")
    final_provenance = scientific_provenance_assessment(current_revision=final_revision,
        status=final_status, head_name=final_head_name,
        accepted_revision=accepted_revision, smoke=smoke, replay=replay_mode)
    combined_provenance = combined_provenance_assessment(provenance, final_provenance)
    metadata["final_git_revision"] = final_revision
    metadata["final_git_status_porcelain"] = final_status
    metadata["final_head_name"] = final_head_name
    metadata["final_scientific_provenance_reasons"] =
        string.(final_provenance.reasons)
    metadata["scientific_provenance_reasons"] =
        string.(combined_provenance.reasons)

    exhausted_termination(value) = value == "-1:parameter_bound;1:parameter_bound"
    unexhausted_axis = count(row -> !axis_branch_exhausted(row.termination,
        row.unresolved_locations),
        axis_branch_rows)
    unexhausted_curves = count(row -> !exhausted_termination(row.termination),
        curve_branch_rows)
    unexhausted_branches = unexhausted_axis + unexhausted_curves
    numerical_outcome = finite_search_numerical_outcome(
        length(deduplicated.representatives), accepted_count, qualified_unattempted,
        unexhausted_branches, unresolved_seed_topologies)
    outcome = scientific_result_outcome(numerical_outcome, failures,
        combined_provenance)
    metadata["execution_success"] = isempty(failures)
    metadata["failed_contexts"] = [row.context for row in failures]
    metadata["seed_manifest_count"] = length(manifest)
    metadata["qualifying_seed_count"] = qualifying_seeds
    metadata["unresolved_seed_topologies"] = unresolved_seed_topologies
    metadata["raw_trace_zero_locations"] = length(raw_locations)
    metadata["deduplicated_trace_zero_locations"] = length(deduplicated.representatives)
    metadata["validated_orbits"] = accepted_count
    metadata["shooting_proposals_selected"] = proposals.effective_count
    metadata["shooting_required_strata"] = proposals.required_count
    metadata["qualified_unattempted_hopf_points"] = qualified_unattempted
    metadata["unexhausted_branches"] = unexhausted_branches
    metadata["outcome"] = outcome
    metadata["numerical_outcome"] = numerical_outcome
    metadata["scientific_outcome"] = outcome * ": " * numerical_outcome *
        "; finite targeted search only, with no absence or prevalence inference"
    metadata["scientific_acceptance_enabled"] =
        scientific_acceptance_enabled(failures, combined_provenance)
    metadata["smoke_scientific_acceptance"] = smoke ? "disabled" : "not_applicable"
    Evidence.write_toml(joinpath(output, "metadata.toml"), metadata)
    Evidence.artifact_checksums(output)
    return (; success=isempty(failures), outcome, seeds=length(manifest),
        qualifying_seeds, trace_zero=length(deduplicated.representatives),
        validated_orbits=accepted_count)
end

function main(args=ARGS)
    config_path = joinpath(REPOSITORY_ROOT, "experiments", "figure5b_ratio44_search.toml")
    seed_artifact = nothing
    replay_artifact = nothing
    accepted_revision = nothing
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
        elseif option in ("--config", "--seed-artifact", "--replay-artifact",
                "--accepted-revision", "--output")
            index < length(args) || throw(ArgumentError("$option requires a value"))
            value = args[index + 1]
            startswith(value, "--") && throw(ArgumentError("$option requires a value"))
            if option == "--config"
                config_path = value
            elseif option == "--seed-artifact"
                seed_artifact = value
            elseif option == "--replay-artifact"
                replay_artifact = value
            elseif option == "--accepted-revision"
                accepted_revision = value
            else
                output = value
            end
            index += 2
        else
            throw(ArgumentError("unknown option: $option"))
        end
    end
    (seed_artifact === nothing) == (replay_artifact === nothing) && throw(ArgumentError(
        "provide exactly one of --seed-artifact or --replay-artifact"))
    output === nothing && throw(ArgumentError("--output DIRECTORY is required"))
    result = run_experiment(config_path, seed_artifact, output; smoke, replay_artifact,
        accepted_revision)
    println("outcome=$(result.outcome); trace_zero=$(result.trace_zero); validated_orbits=$(result.validated_orbits)")
    return result.success ? 0 : 1
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    try
        exit(Figure5bRatio44Search.main())
    catch error
        error isa InterruptException && rethrow()
        showerror(stderr, error)
        println(stderr)
        exit(1)
    end
end
