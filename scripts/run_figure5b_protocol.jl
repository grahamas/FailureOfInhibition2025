"""Clean, candidate-specific Figure-5b Hopf and periodic-orbit protocol."""
module Figure5bProtocol

using FailureOfInhibition2025
using LinearAlgebra: BLAS, det, dot, eigen, eigvals, norm
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode
import CSV
import SHA
import TOML

include("study_support.jl")
const MinimalExperiment = StudySupport
const Evidence = MinimalExperiment

const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const ROOT_GRIDS = (11, 21, 41)
const PROTOCOL_SOURCES = ("run_figure5b_protocol.jl", "study_support.jl",
    "figure5b_protocol_config.jl", "figure5b_protocol_topology.jl",
    "figure5b_protocol_hopf.jl", "figure5b_protocol_orbits.jl",
    "figure5b_protocol_evidence.jl", "figure5b_protocol_artifacts.jl")

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

include("figure5b_protocol_config.jl")

include("figure5b_protocol_topology.jl")

include("figure5b_protocol_hopf.jl")

include("figure5b_protocol_orbits.jl")

include("figure5b_protocol_evidence.jl")

include("figure5b_protocol_artifacts.jl")

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
