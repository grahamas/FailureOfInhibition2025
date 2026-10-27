"""Anchor Hopf/orbit protocol functions included by run_anchor_hopf_orbit_protocol.jl."""
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
    runner_relative = joinpath("scripts", "run_anchor_hopf_orbit_protocol.jl")
    archived_runner = joinpath(parent, "source", runner_relative)
    isfile(archived_runner) || throw(ArgumentError("replay parent lacks archived runner"))
    Evidence.file_hash(joinpath(REPOSITORY_ROOT, runner_relative)) ==
        Evidence.file_hash(archived_runner) ||
        throw(ArgumentError("executing runner differs from the archived runner"))
    for name in PROTOCOL_SOURCES
        relative = joinpath("scripts", name)
        current = joinpath(REPOSITORY_ROOT, relative)
        isfile(current) && get(metadata["source_sha256"], relative, nothing) ==
            Evidence.file_hash(current) ||
            throw(ArgumentError("executing protocol source differs from the archived source: $relative"))
    end
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
