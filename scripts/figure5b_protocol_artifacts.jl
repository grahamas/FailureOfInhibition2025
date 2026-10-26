"""Figure 5b protocol functions included by run_figure5b_protocol.jl."""
function _archive_provenance(config_path, output, config, smoke, provenance,
    canonical_config, replay_info)
    metadata = Evidence.archive_provenance(config_path, output)
    for name in PROTOCOL_SOURCES
        relative = joinpath("scripts", name)
        destination = joinpath(output, "source", relative)
        isfile(destination) || cp(joinpath(REPOSITORY_ROOT, relative), destination)
        metadata["source_sha256"][relative] = Evidence.file_hash(destination)
    end
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
