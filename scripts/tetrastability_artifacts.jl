"""Tetrastability protocol functions included by run_tetrastability_search.jl."""
function _archive_provenance(config_path, output, config, smoke)
    metadata = Evidence.archive_provenance(config_path, output)
    Evidence.archive_sources!(metadata, output,
        (joinpath("scripts", name) for name in ("run_coexistence_map.jl", "coexistence_core.jl",
        "run_tetrastability_search.jl", "tetrastability_core.jl",
        "tetrastability_config.jl", "tetrastability_sampling.jl",
        "tetrastability_confirmation.jl", "tetrastability_continuation.jl",
        "tetrastability_artifacts.jl")))
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
