include(joinpath(@__DIR__, "..", "scripts", "run_figure5b_ratio44_search.jl"))

const Ratio44 = Figure5bRatio44Search

function ratio44_fixture_artifact(directory, config; seed=true)
    mkpath(directory)
    cp(joinpath(@__DIR__, "..", "experiments", "tetrastability.toml"),
        joinpath(directory, "config.toml"))
    sample_columns = Ratio44.Tetrastability.SAMPLE_COLUMNS
    samples = seed ? [(; cell_id="anchor", family="halton", stage="extension",
        plane="not_applicable", e_to_i_index=0, theta_off_index=0, halton_index=4096,
        halton_e_to_e=0.5, halton_i_to_e=0.5, halton_e_to_i=0.5,
        halton_i_to_i=0.5, halton_theta_off=0.5, e_to_e=19.0, i_to_e=13.0,
        e_to_i=19.0, i_to_i=6.0, theta_off=8.0)] : NamedTuple[]
    Ratio44.Evidence.write_rows(joinpath(directory, "samples.csv"), samples,
        sample_columns)
    searches = seed ? [(; search_id="search_001232", cell_id="anchor",
        stage="extension", condition="failure_of_inhibition", grid_points=11,
        status="completed", discovered_equilibria=7, attracting_equilibria=3,
        saddle_equilibria=3, repelling_equilibria=1, stability_unresolved=0,
        near_singular_equilibria=0, unresolved_nearby_groups=0, attempts=146,
        rejected_attempts=0, completeness="CompletenessNotCertified")] : NamedTuple[]
    Ratio44.Evidence.write_rows(joinpath(directory, "searches.csv"), searches,
        Ratio44.Tetrastability.SEARCH_COLUMNS)
    roots = seed ? [
        (0.02, 0.01, "Attracting"), (0.10, 0.12, "Saddle"),
        (0.20, 0.22, "Attracting"), (0.329934171847263, 0.367295168265613, "Repelling"),
        (0.50, 0.48, "Saddle"), (0.65, 0.58, "Attracting"),
        (0.75, 0.65, "Saddle")
    ] : Tuple{Float64,Float64,String}[]
    equilibria = [(; search_id="search_001232", equilibrium=index,
        condition="failure_of_inhibition", grid_points=11, E=root[1], I=root[2],
        residual_norm=1e-14, near_singular=false, representative_attempt=index,
        member_attempts="[$index]", stability=root[3], geometry="not_used",
        eigenvalue_1_real=0.0, eigenvalue_1_imaginary=0.0,
        eigenvalue_2_real=0.0, eigenvalue_2_imaginary=0.0,
        spectral_abscissa=0.0, u_I=0.0, F_I=0.0, F_I_prime=0.0,
        inhibitory_response_midpoint=6.0, inhibitory_branch="not_used",
        descending_response_branch=false) for (index, root) in enumerate(roots)]
    Ratio44.Evidence.write_rows(joinpath(directory, "equilibria.csv"), equilibria,
        Ratio44.Tetrastability.EQUILIBRIUM_COLUMNS)
    metadata = Dict(
        "schema_version" => 1, "experiment" => "tetrastability_search",
        "execution_success" => true, "smoke" => false,
        "failed_contexts" => String[], "git_status_porcelain" => "",
        "git_revision" => config.seed_policy.accepted_revision,
        "extension_run" => true,
        "last_halton_index" => config.seed_policy.last_halton_index,
        "parameter_cells" => config.seed_policy.parameter_cells,
        "source_sha256" => Dict("scripts/run_tetrastability_search.jl" => repeat("a", 64)))
    Ratio44.Evidence.write_toml(joinpath(directory, "metadata.toml"), metadata)
    Ratio44.Evidence.artifact_checksums(directory)
    return directory
end

@testset "Fixed-ratio configuration and artifact gate" begin
    path = joinpath(@__DIR__, "..", "experiments", "figure5b_ratio44_search.toml")
    config = Ratio44.load_config(path)
    @test config.fixed.tau_ratio == 4.4
    @test config.path_axes == Ratio44.SEARCH_AXES
    @test config.curve_pairs == Ratio44.CURVE_PAIRS
    @test config.seed_policy.maximum_halton_points == 4096
    @test config.seed_policy.parameter_cells == 5746

    mktempdir() do directory
        for mutate in (
            raw -> raw["schema_version"] = true,
            raw -> raw["unexpected"] = true,
            raw -> raw["fixed"]["tau_ratio"] = 4.5,
            raw -> raw["axes"]["e_to_e"]["minimum"] = NaN,
            raw -> raw["paths"]["axes"] = ["e_to_e", "e_to_e"],
            raw -> raw["curves"][1]["second"] = "e_to_e",
            raw -> raw["smoke"]["shooting"] = 1,
            raw -> raw["augmented"]["max_steps"] = 199,
            raw -> raw["curve_continuation"]["initial_step"] = 0.02,
            raw -> raw["directional"]["offset"] = 0.02,
            raw -> raw["shooting"]["maximum_hopf_points"] = 31,
            raw -> raw["independent"]["tight_samples"] = 257,
            raw -> raw["hopf"]["lyapunov_atol"] = 1e-6,
        )
            raw = deepcopy(config.raw)
            mutate(raw)
            invalid = joinpath(directory, "invalid.toml")
            Ratio44.Evidence.write_toml(invalid, raw)
            @test_throws ArgumentError Ratio44.load_config(invalid)
        end

        alternate = deepcopy(config.raw)
        alternate["search"]["root_grids"] = [9, 19, 39]
        alternate_path = joinpath(directory, "alternate_grids.toml")
        Ratio44.Evidence.write_toml(alternate_path, alternate)
        expected_grid_error = "ArgumentError: search.root_grids must be exactly 11, 21 and 41"
        load_error = try
            Ratio44.load_config(alternate_path; require_canonical=false)
            nothing
        catch error
            error
        end
        @test load_error isa ArgumentError
        @test load_error isa ArgumentError && sprint(showerror, load_error) == expected_grid_error
        rejected_output = joinpath(directory, "alternate_grid_output")
        run_error = try
            Ratio44.run_experiment(alternate_path, nothing, rejected_output;
                smoke=true, require_canonical=false)
            nothing
        catch error
            error
        end
        @test run_error isa ArgumentError
        @test run_error isa ArgumentError && sprint(showerror, run_error) == expected_grid_error
        @test !ispath(rejected_output)

        artifact = ratio44_fixture_artifact(joinpath(directory, "artifact"), config)
        verified = Ratio44.verify_seed_artifact(artifact, config)
        @test verified.metadata["execution_success"]
        manifest = Ratio44.normalized_seed_manifest(verified, config)
        @test length(manifest) == 1
        @test manifest == Ratio44.normalized_seed_manifest(verified, config)
        @test manifest[1].parameters.theta_off == 8.0
        @test manifest[1].hypothesis_id == Ratio44.hypothesis_hash(
            manifest[1].parameters, config.fixed, repeat("a", 64),
            verified.config_hash, "seed-artifact:search_001232", "seven_root_seed")

        checksums = Ratio44.TOML.parsefile(joinpath(artifact, "checksums.toml"))
        checksums["files"]["samples.csv"] = repeat("0", 64)
        Ratio44.Evidence.write_toml(joinpath(artifact, "checksums.toml"), checksums)
        @test_throws ArgumentError Ratio44.verify_seed_artifact(artifact, config)

        for (name, mutate) in (
            ("revision", metadata -> metadata["git_revision"] = repeat("0", 40)),
            ("dirty", metadata -> metadata["git_status_porcelain"] = "M source.jl"),
            ("smoke", metadata -> metadata["smoke"] = true),
            ("schema", metadata -> metadata["schema_version"] = true),
            ("schedule", metadata -> metadata["last_halton_index"] = 1024),
        )
            bad = ratio44_fixture_artifact(joinpath(directory, "bad_$name"), config)
            metadata = Ratio44.TOML.parsefile(joinpath(bad, "metadata.toml"))
            mutate(metadata)
            Ratio44.Evidence.write_toml(joinpath(bad, "metadata.toml"), metadata)
            Ratio44.Evidence.artifact_checksums(bad)
            @test_throws ArgumentError Ratio44.verify_seed_artifact(bad, config)
        end
    end
end

@testset "Augmented trace-zero and curve continuation" begin
    config = Ratio44.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_ratio44_search.toml"))
    axis_residual(z) = [z[1] - 0.25, z[2] - 0.5, z[3] - 3.0]
    solved = Ratio44.solve_augmented_trace_zero(axis_residual, [0.2, 0.4, 2.8])
    @test solved.success
    @test solved.candidate ≈ [0.25, 0.5, 3.0] atol=1e-10
    @test solved.minimum_singular_value > 0.9
    eligible_candidate = (; success=true, candidate=[0.25, 0.5, 3.0],
        minimum_singular_value=1.0)
    bracketed = Ratio44.augmented_trace_zero_eligibility(eligible_candidate,
        (0.0, 4.0), (2.5, 3.5), 1e-8)
    @test bracketed.success && bracketed.in_bounds && bracketed.in_bracket &&
        bracketed.full_rank
    escaped = Ratio44.augmented_trace_zero_eligibility(eligible_candidate,
        (0.0, 4.0), (2.0, 2.5), 1e-8)
    @test !escaped.success && escaped.in_bounds && !escaped.in_bracket
    outside_domain = Ratio44.augmented_trace_zero_eligibility(
        merge(eligible_candidate, (; candidate=[0.25, 0.5, 5.0])),
        (0.0, 4.0), (2.0, 6.0), 1e-8)
    @test !outside_domain.success && !outside_domain.in_bounds &&
        outside_domain.in_bracket
    raw_failure = Ratio44.augmented_trace_zero_eligibility(
        merge(eligible_candidate, (; success=false)), (0.0, 4.0),
        (2.5, 3.5), 1e-8)
    @test !raw_failure.success && !raw_failure.in_bounds &&
        !raw_failure.in_bracket && !raw_failure.full_rank
    rank_deficient = Ratio44.augmented_trace_zero_eligibility(
        merge(eligible_candidate, (; minimum_singular_value=1e-12)),
        (0.0, 4.0), (2.5, 3.5), 1e-8)
    @test !rank_deficient.success && !rank_deficient.full_rank

    curve_residual(z) = [z[1] - 0.25, z[2] - 0.5, z[3] + 2z[4] - 1.0]
    closest = Ratio44.closest_point_curve_seed(curve_residual,
        [0.25, 0.5, 0.0, 0.0], ones(4))
    @test closest.success
    @test closest.candidate[3] + 2closest.candidate[4] ≈ 1.0 atol=1e-8
    @test closest.candidate[3:4] ≈ [0.2, 0.4] atol=1e-6
    branch = Ratio44.continue_trace_zero_curve(curve_residual, closest.candidate;
        scales=ones(4), bounds=[(0.0, 1.0), (0.0, 1.0), (-2.0, 2.0),
            (-2.0, 2.0)], max_steps=3)
    @test length(branch.points) == 7
    @test all(point -> abs(point.coordinates[3] + 2point.coordinates[4] - 1) < 1e-8,
        branch.points)
    @test isempty(filter(attempt -> !attempt.accepted, branch.attempts))

    valid_curve_identity = function (_current, _predictor, corrected, _step)
        states = ((Tuple(corrected[1:2]), Tuple(corrected[1:2]),
            Tuple(corrected[1:2])),)
        return (; matched=true, central_track=1, solved_track=1,
            branch_distances=[0.0], solved_distances=[0.0], nearest_gap=Inf,
            reasons=Symbol[], root_counts=[7, 7, 7], unresolved_counts=[0, 0, 0],
            minimum_root_separation=0.1, track_states=states)
    end
    tracked_branch = Ratio44.continue_trace_zero_curve(curve_residual,
        closest.candidate; scales=ones(4), bounds=[(0.0, 1.0), (0.0, 1.0),
            (-2.0, 2.0), (-2.0, 2.0)], max_steps=1,
        identity_check_function=valid_curve_identity)
    @test length(tracked_branch.points) == 3
    @test all(attempt -> attempt.accepted && attempt.is_local &&
        attempt.identity_checked && attempt.identity_matched,
        tracked_branch.attempts)

    multicomponent_residual(z) = [z[1] - 0.25, z[2] - 0.5, z[3]^2 - 1.0]
    component_jump(_residual, _predictor, _tangent, _scales, _options) =
        (; success=true, candidate=[0.25, 0.5, -1.0, 0.0],
            residual_norm=0.0, status="converged_other_component", error=nothing)
    jumped_component = Ratio44.continue_trace_zero_curve(multicomponent_residual,
        [0.25, 0.5, 1.0, 0.0]; scales=ones(4),
        bounds=[(0.0, 1.0), (0.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)],
        initial_step=0.01, minimum_step=0.005, maximum_step=0.01,
        max_steps=1, max_retries=0, corrector_function=component_jump)
    @test all(attempt -> !attempt.accepted && !attempt.is_local &&
        attempt.status == "nonlocal_correction", jumped_component.attempts)
    @test !Ratio44.curve_branch_exhausted(jumped_component.termination)

    function rejected_curve_identity(reason; central_track=nothing, solved_track=nothing)
        return (_current, _predictor, _corrected, _step) ->
            (; matched=false, central_track, solved_track,
                branch_distances=[0.001, 0.006], solved_distances=[0.005, 0.0],
                nearest_gap=0.005, reasons=[reason], root_counts=[7, 7, 7],
                unresolved_counts=[0, 0, 0], minimum_root_separation=0.005,
                track_states=Tuple[])
    end
    for (reason, central_track, solved_track) in (
            (:solved_root_track_mismatch, 1, 2),
            (:central_track_ambiguous, nothing, 2),
            (:central_track_lost, nothing, nothing))
        rejected_branch = Ratio44.continue_trace_zero_curve(curve_residual,
            closest.candidate; scales=ones(4),
            bounds=[(0.0, 1.0), (0.0, 1.0), (-2.0, 2.0), (-2.0, 2.0)],
            initial_step=0.01, minimum_step=0.005, maximum_step=0.01,
            max_steps=1, max_retries=0,
            identity_check_function=rejected_curve_identity(reason;
                central_track, solved_track))
        @test all(attempt -> !attempt.accepted && attempt.is_local &&
            attempt.status == "root_identity_failed" &&
            occursin(string(reason), attempt.identity_reasons),
            rejected_branch.attempts)
        @test !Ratio44.curve_branch_exhausted(rejected_branch.termination)
        @test length(rejected_branch.points) == 1
    end

    rank_bad(z) = [z[1], z[1], z[3] + z[4]]
    failed = Ratio44.continue_trace_zero_curve(rank_bad, zeros(4))
    @test failed.termination == "initial_rank"

    retry_calls = Ref(0)
    function stale_boundary_corrector(_residual, predictor, _tangent, _scales, _options)
        retry_calls[] += 1
        retry_calls[] == 1 && return (; success=true,
            candidate=[0.25, 0.5, 2.0, 2.0], residual_norm=0.0,
            status="out_of_bounds", error=nothing)
        return (; success=false, candidate=copy(predictor), residual_norm=Inf,
            status="failed", error=nothing)
    end
    stale_boundary = Ratio44.continue_trace_zero_curve(curve_residual,
        closest.candidate; scales=ones(4),
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        initial_step=0.1, minimum_step=0.04, maximum_step=0.1,
        max_steps=1, max_retries=2,
        corrector_function=stale_boundary_corrector)
    @test !occursin("parameter_bound", stale_boundary.termination)
    @test occursin("minimum_step", stale_boundary.termination)
    always_boundary(_residual, _predictor, _tangent, _scales, _options) = (; success=true,
        candidate=[0.25, 0.5, 2.0, 2.0], residual_norm=0.0,
        status="out_of_bounds", error=nothing)
    true_boundary = Ratio44.continue_trace_zero_curve(curve_residual,
        closest.candidate; scales=ones(4),
        bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        initial_step=0.1, minimum_step=0.04, maximum_step=0.1,
        max_steps=1, max_retries=2, corrector_function=always_boundary)
    @test true_boundary.termination == "-1:parameter_bound;1:parameter_bound"

    unresolved_closest = (; closest=(; success=false, attempts=[(; attempt=1,)]),
        branches=NamedTuple[])
    @test Ratio44.curve_seed_search_unresolved(unresolved_closest)
    unresolved_curve_row = Ratio44.unresolved_curve_branch_record("curve", "seed",
        (:e_to_e, :i_to_i), unresolved_closest)
    @test unresolved_curve_row.termination == "closest_seed_failed"
    @test unresolved_curve_row.attempts == 1
    failed_closest_with_axis_branch = (; closest=unresolved_closest.closest,
        branches=[(; branch=1,)])
    @test Ratio44.curve_seed_search_unresolved(failed_closest_with_axis_branch)
    retained_closest_failure = Ratio44.unresolved_curve_branch_record(
        "curve", "seed", (:e_to_e, :i_to_i), failed_closest_with_axis_branch)
    @test retained_closest_failure.termination == "closest_seed_failed"
    @test retained_closest_failure.branch_id == "curve:unresolved"

    items = [(; source_id="b", state=(0.3, 0.4), proposal_stratum="curve_arclength",
        proposal_group="g", arclength_index=1.0,
        parameters=(e_to_e=19.0, i_to_e=13.0,
            e_to_i=19.0, i_to_i=6.0, theta_off=8.0)),
        (; source_id="a", state=(0.3 + 1e-10, 0.4), proposal_stratum="axis_solution",
            proposal_group="a", arclength_index=1.0,
            parameters=(e_to_e=19.0 + 1e-9, i_to_e=13.0,
                e_to_i=19.0, i_to_i=6.0, theta_off=8.0)),
        (; source_id="c", state=(0.5, 0.6), proposal_stratum="curve_endpoint",
            proposal_group="g", arclength_index=2.0,
            parameters=(e_to_e=19.0, i_to_e=13.0,
                e_to_i=19.0, i_to_i=6.0, theta_off=8.0))]
    dedup = Ratio44.deduplicate_trace_zero_locations(items, 1e-6)
    @test length(dedup.representatives) == 2
    @test length(dedup.links) == 3
    @test count(row -> row.duplicate, dedup.links) == 1
    directional_items = [
        (; source_id="curve-x", parent_hypothesis="parent", method="curve",
            state=(0.3, 0.4), proposal_stratum="curve_seed", proposal_group="g",
            arclength_index=0.0, direction=(e_to_e=1.0, i_to_i=0.0),
            parameters=items[1].parameters),
        (; source_id="curve-direction", parent_hypothesis="parent", method="curve",
            state=(0.3, 0.4),
            proposal_stratum="curve_seed", proposal_group="g",
            arclength_index=0.0, direction=(e_to_e=0.0, i_to_i=1.0),
            parameters=items[1].parameters),
        (; source_id="curve-opposite", parent_hypothesis="parent", method="curve",
            state=(0.3, 0.4),
            proposal_stratum="curve_seed", proposal_group="g",
            arclength_index=0.0, direction=(e_to_e=0.0, i_to_i=-1.0),
            parameters=items[1].parameters)]
    directional_dedup = Ratio44.deduplicate_trace_zero_locations(
        directional_items, 1e-6)
    @test length(directional_dedup.representatives) == 2
    @test any(item -> item.source_id == "curve-direction",
        directional_dedup.representatives)
    identities = [Ratio44.trace_zero_hypothesis_identity(item, config.fixed,
        repeat("a", 64), repeat("b", 64))
        for item in directional_dedup.representatives]
    @test length(unique(identity.identifier for identity in identities)) == 2
    positive_y = Ratio44.trace_zero_hypothesis_identity(directional_items[2],
        config.fixed, repeat("a", 64), repeat("b", 64))
    negative_y = Ratio44.trace_zero_hypothesis_identity(directional_items[3],
        config.fixed, repeat("a", 64), repeat("b", 64))
    @test positive_y.identifier == negative_y.identifier
    @test positive_y.direction == negative_y.direction
    mktempdir() do artifact
        mkpath(joinpath(artifact, "contexts", "hypotheses"))
        for identity in identities
            Ratio44._write_hypothesis(artifact, identity.identifier, identity.record)
        end
        files = readdir(joinpath(artifact, "contexts", "hypotheses"))
        @test length(files) == 2
        records = [Ratio44.TOML.parsefile(joinpath(artifact, "contexts",
            "hypotheses", file)) for file in files]
        @test all(haskey(record, "direction") for record in records)
        @test length(unique(record["hypothesis_id"] for record in records)) == 2
    end
    proposals = Ratio44.select_shooting_proposals([
        (; source_id="axis", proposal_stratum="axis_solution", proposal_group="a",
            arclength_index=0.0),
        (; source_id="seed", proposal_stratum="curve_seed", proposal_group="g",
            arclength_index=0.0),
        (; source_id="end", proposal_stratum="curve_endpoint", proposal_group="g",
            arclength_index=3.0),
        (; source_id="reverse", proposal_stratum="curve_reversal", proposal_group="g",
            arclength_index=2.0),
        (; source_id="sample1", proposal_stratum="curve_arclength", proposal_group="g",
            arclength_index=1.0),
        (; source_id="sample2", proposal_stratum="curve_arclength", proposal_group="g",
            arclength_index=2.5)], 5)
    @test all(id -> id in proposals.selected, ("axis", "seed", "end", "reverse"))
    @test proposals.effective_count == 5
    @test count(row -> !row.selected, proposals.rows) == 1

    previous = (0.0, 0.0)
    intended_track = ((0.001, 0.0), (0.001, 0.0), (0.001, 0.0))
    nearby_track = ((0.006, 0.0), (0.006, 0.0), (0.006, 0.0))
    distant_track = ((0.2, 0.0), (0.2, 0.0), (0.2, 0.0))
    positive_identity = Ratio44.unique_axis_track_match(previous, [0.001, 0.0],
        [intended_track, distant_track]; coordinate_atol=1e-6,
        displacement_tolerance=0.01)
    @test positive_identity.matched
    @test positive_identity.central_track == positive_identity.solved_track == 1
    nearby_jump = Ratio44.unique_axis_track_match(previous, [0.006, 0.0],
        [intended_track, nearby_track]; coordinate_atol=1e-6,
        displacement_tolerance=0.01)
    @test !nearby_jump.matched
    @test :central_track_ambiguous in nearby_jump.reasons
    @test nearby_jump.solved_track == 2
    lost_identity = Ratio44.unique_axis_track_match(previous, [0.2, 0.0],
        [distant_track]; coordinate_atol=1e-6,
        displacement_tolerance=0.01)
    @test !lost_identity.matched
    @test :central_track_lost in lost_identity.reasons

    synthetic_attempt = (; residual_norm=0.0, solver_status=:synthetic,
        reasons=Symbol[])
    synthetic_axis_step = function (_config, parameters, _state)
        parameter = parameters.e_to_e
        return (; accepted=true, state=(0.3, 0.4), trace=parameter - 23.975,
            solved=(; attempt=synthetic_attempt,
                stability=(; classification=Repelling)), model=nothing)
    end
    accepted_identity = function (_config, _parameters, _previous, solved)
        states = ((Tuple(solved), Tuple(solved), Tuple(solved)),)
        return (; matched=true, central_track=1, solved_track=1,
            branch_distances=[0.0], solved_distances=[0.0], nearest_gap=Inf,
            reasons=Symbol[], root_counts=[7, 7, 7], unresolved_counts=[0, 0, 0],
            minimum_root_separation=0.1, track_states=states)
    end
    accepted_augmented = function (_config, parameters, axis, seed; bracket=nothing)
        parameter = sum(bracket) / 2
        return (; success=true, candidate=[seed[1], seed[2], parameter],
            residual_norm=0.0, minimum_singular_value=1.0, in_bounds=true,
            in_bracket=true, full_rank=true, status="synthetic", error=nothing,
            parameters=Ratio44.replace_parameter(parameters, axis, parameter))
    end
    axis_seed = (; hypothesis_id="synthetic-axis",
        parameters=(e_to_e=23.95, i_to_e=13.0, e_to_i=19.0, i_to_i=6.0,
            theta_off=8.0), central_state=(0.3, 0.4))
    bounded_axis = Ratio44.continue_axis_path(config, axis_seed, :e_to_e;
        smoke=true, equilibrium_step_function=synthetic_axis_step,
        augmented_solver=accepted_augmented,
        identity_check_function=accepted_identity)
    @test any(point -> point.direction == 1 && point.parameter == 24.0,
        bounded_axis.points)
    @test any(bracket -> bracket.direction == 1 && bracket.upper == 24.0,
        bounded_axis.brackets)
    @test occursin("1:parameter_bound", bounded_axis.termination)

    crossing_axis_step = function (_config, parameters, _state)
        parameter = parameters.e_to_e
        return (; accepted=true, state=(0.3, 0.4), trace=parameter - 20.0,
            solved=(; attempt=synthetic_attempt,
                stability=(; classification=Repelling)), model=nothing)
    end
    failed_augmented(_config, _parameters, _axis, seed; bracket=nothing) =
        (; success=false, candidate=[seed[1], seed[2], sum(bracket) / 2],
            residual_norm=1.0, minimum_singular_value=0.0, in_bounds=true,
            in_bracket=true, full_rank=false, status="rank_deficient", error=nothing)
    exhaustive_seed = merge(axis_seed, (; parameters=merge(axis_seed.parameters,
        (; e_to_e=19.0))))
    unresolved_axis = Ratio44.continue_axis_path(config, exhaustive_seed, :e_to_e;
        equilibrium_step_function=crossing_axis_step,
        augmented_solver=failed_augmented,
        identity_check_function=accepted_identity)
    @test unresolved_axis.termination == "-1:parameter_bound;1:parameter_bound"
    @test any(location -> !location.success && !location.full_rank,
        unresolved_axis.locations)
    @test !Ratio44.axis_path_exhausted(unresolved_axis)

    ambiguous_identity = function (_config, _parameters, _previous, solved)
        tracks = (((0.001, 0.0), (0.001, 0.0), Tuple(solved)),
            ((0.006, 0.0), (0.006, 0.0), Tuple(solved)))
        return (; matched=false, central_track=nothing, solved_track=2,
            branch_distances=[0.001, 0.006], solved_distances=[0.005, 0.0],
            nearest_gap=0.005, reasons=[:central_track_ambiguous],
            root_counts=[7, 7, 7], unresolved_counts=[0, 0, 0],
            minimum_root_separation=0.005, track_states=tracks)
    end
    rejected_identity = Ratio44.continue_axis_path(config, axis_seed, :e_to_e;
        smoke=true, equilibrium_step_function=synthetic_axis_step,
        augmented_solver=accepted_augmented,
        identity_check_function=ambiguous_identity)
    @test any(attempt -> attempt.status == "root_identity_failed" &&
        occursin("central_track_ambiguous", attempt.reasons),
        rejected_identity.attempts)
    @test isempty(filter(attempt -> attempt.accepted, rejected_identity.attempts))
end

@testset "Directional fixed-ratio transversality" begin
    config = Ratio44.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_ratio44_search.toml"))
    estimated = Ratio44.directional_transversality(delta -> 6delta,
        [1e-3, 5e-4, 2.5e-4])
    @test estimated.resolved
    @test estimated.beta ≈ 3.0 atol=1e-12
    unresolved = Ratio44.directional_transversality(delta -> delta + delta^2 / 1e-6,
        [1e-3, 5e-4, 2.5e-4]; plateau_atol=1e-12, plateau_rtol=1e-12)
    # Symmetric differences remove even terms, so inject a step-dependent failure.
    @test unresolved.resolved
    failed = Ratio44.directional_transversality(delta -> abs(delta) < 3e-4 ? NaN : delta,
        [1e-3, 5e-4, 2.5e-4])
    @test !failed.resolved
    exceptional = Ratio44.directional_transversality(delta ->
        delta > 0 ? error("injected trace failure") : delta, [1e-3])
    @test !exceptional.resolved
    @test only(exceptional.errors).side == "positive"
    @test occursin("injected trace failure", only(exceptional.errors).message)

    reference = ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    central = ((0.001, 0.0), (0.001, 0.0), (0.001, 0.0))
    distant = ((0.02, 0.0), (0.02, 0.0), (0.02, 0.0))
    valid = Ratio44.unique_directional_track_match(reference, [0.001, 0.0],
        [0.001, 0.0], [central, distant]; coordinate_atol=1e-6,
        predictor_atol=1e-5, displacement_tolerance=0.01)
    @test valid.matched
    @test valid.central_track == valid.predicted_track == valid.solved_track == 1
    permuted = Ratio44.unique_directional_track_match(reference, [0.001, 0.0],
        [0.001, 0.0], [distant, central]; coordinate_atol=1e-6,
        predictor_atol=1e-5, displacement_tolerance=0.01)
    @test permuted.matched
    @test permuted.central_track == permuted.predicted_track == permuted.solved_track == 2

    nearby = ((0.006, 0.0), (0.006, 0.0), (0.006, 0.0))
    jumped = Ratio44.unique_directional_track_match(reference, [0.001, 0.0],
        [0.006, 0.0], [central, nearby]; coordinate_atol=1e-6,
        predictor_atol=1e-5, displacement_tolerance=0.01)
    @test !jumped.matched
    @test :central_track_ambiguous in jumped.reasons
    @test jumped.solved_track == 2

    wrong_root = Ratio44.unique_directional_track_match(reference, [0.001, 0.0],
        [0.02, 0.0], [central, distant]; coordinate_atol=1e-6,
        predictor_atol=1e-5, displacement_tolerance=0.01)
    @test !wrong_root.matched
    @test :solved_root_track_mismatch in wrong_root.reasons

    @test Ratio44._directional_sample_trace(
        (; matched=true, matched_trace=1.25), -999.0) == 1.25
    @test isnan(Ratio44._directional_sample_trace(
        (; matched=false, matched_trace=1.25), 1.25))

    lyapunov_only = (; resolved=false,
        reasons=[:unresolved_transversality,
            :underflowed_modal_radius_squared_slope],
        frequency=1.0, c1=-1e-3 + 0im, lyapunov_ad=-1e-3,
        lyapunov_fd=-1.0e-3,
        lyapunov_fd_values=fill(-1.0e-3,
            length(config.hopf_options.fd_steps)))
    independent_lyapunov = Ratio44.fixed_ratio_lyapunov_assessment(
        lyapunov_only, config.hopf_options)
    @test independent_lyapunov.resolved
    @test isempty(independent_lyapunov.reasons)
    sign_bad = Ratio44.fixed_ratio_lyapunov_assessment(
        merge(lyapunov_only, (; lyapunov_fd=1e-3,
            lyapunov_fd_values=fill(1e-3,
                length(config.hopf_options.fd_steps)))), config.hopf_options)
    @test !sign_bad.resolved
    @test :lyapunov_sign_disagreement in sign_bad.reasons
end

@testset "Scientific provenance eligibility" begin
    revision = repeat("a", 40)
    accepted = Ratio44.scientific_provenance_assessment(current_revision=revision,
        status="", head_name="HEAD", accepted_revision=revision)
    @test accepted.eligible
    @test :attached_head in Ratio44.scientific_provenance_assessment(
        current_revision=revision, status="", head_name="main",
        accepted_revision=revision).reasons
    @test :accepted_revision_mismatch in Ratio44.scientific_provenance_assessment(
        current_revision=revision, status="", head_name="HEAD",
        accepted_revision=repeat("b", 40)).reasons
    @test :dirty_checkout in Ratio44.scientific_provenance_assessment(
        current_revision=revision, status="M file", head_name="HEAD",
        accepted_revision=revision).reasons
    @test :revision_unavailable in Ratio44.scientific_provenance_assessment(
        current_revision="unavailable: no git", status="unavailable: no git",
        head_name="unavailable: no git", accepted_revision=revision).reasons

    final_dirty = Ratio44.scientific_provenance_assessment(
        current_revision=revision, status="?? unexpected", head_name="HEAD",
        accepted_revision=revision)
    combined = Ratio44.combined_provenance_assessment(accepted, final_dirty)
    @test !combined.eligible
    @test :dirty_checkout in combined.reasons
    @test Ratio44.scientific_result_outcome(
        "completed_no_trace_zero_solution", NamedTuple[], combined) ==
        "evidence_ineligible"
    @test !Ratio44.scientific_acceptance_enabled(NamedTuple[], combined)
    @test Ratio44.finite_search_numerical_outcome(0, 0, 0, 0, 0) ==
        "completed_no_trace_zero_solution"
    @test Ratio44.finite_search_numerical_outcome(0, 0, 0, 1, 0) ==
        "completed_unresolved_targeted_search"
    @test Ratio44.finite_search_numerical_outcome(0, 0, 0, 0, 1) ==
        "completed_unresolved_targeted_search"
    @test Ratio44.finite_search_numerical_outcome(1, 0, 0, 0, 0) ==
        "completed_no_validated_ratio44_orbit"
end


@testset "Output destination provenance guard" begin
    mktempdir() do directory
        repository = joinpath(directory, "repository")
        mkpath(repository)
        run(`git init --quiet $repository`)
        write(joinpath(repository, ".gitignore"),
            "output/*\n!output/keep/\n")

        ignored = joinpath(repository, "output", "accepted", "run")
        @test !ispath(ignored)
        accepted = Ratio44.output_destination_assessment(ignored;
            repository_root=repository)
        @test accepted.eligible
        @test accepted.inside_checkout
        @test accepted.git_ignored === true
        @test !ispath(ignored)

        unignored = joinpath(repository, "unignored", "run")
        @test_throws ArgumentError Ratio44.require_output_destination(unignored;
            repository_root=repository)
        @test !ispath(unignored)

        negated = joinpath(repository, "output", "keep", "run")
        @test_throws ArgumentError Ratio44.require_output_destination(negated;
            repository_root=repository)
        @test !ispath(negated)

        outside = joinpath(directory, "outside", "run")
        external = Ratio44.require_output_destination(outside;
            repository_root=repository)
        @test external.eligible
        @test !external.inside_checkout
        @test external.git_ignored === nothing
        @test !ispath(outside)

        if !Sys.iswindows()
            inside_parent = joinpath(repository, "through_link")
            mkpath(inside_parent)
            outside_link = joinpath(directory, "link_into_repository")
            symlink(inside_parent, outside_link)
            linked_output = joinpath(outside_link, "run")
            linked = Ratio44.output_destination_assessment(linked_output;
                repository_root=repository)
            @test linked.inside_checkout
            @test !linked.eligible
            @test_throws ArgumentError Ratio44.require_output_destination(linked_output;
                repository_root=repository)
            @test !ispath(linked_output)

            external_parent = joinpath(directory, "external_parent")
            mkpath(external_parent)
            inside_link = joinpath(repository, "link_outside_repository")
            symlink(external_parent, inside_link)
            escaped_output = joinpath(inside_link, "run")
            escaped = Ratio44.output_destination_assessment(escaped_output;
                repository_root=repository)
            @test escaped.inside_checkout
            @test !escaped.eligible
            @test :output_symlink_component in escaped.reasons
            @test_throws ArgumentError Ratio44.require_output_destination(escaped_output;
                repository_root=repository)
            @test !ispath(joinpath(external_parent, "run"))
        end

        nongit = joinpath(directory, "not-a-repository")
        mkpath(nongit)
        failed_check = Ratio44.output_destination_assessment(
            joinpath(nongit, "output"); repository_root=nongit)
        @test failed_check.inside_checkout
        @test !failed_check.eligible
        @test failed_check.git_ignored === false
    end
end

@testset "Injected fixed-ratio orbit acceptance" begin
    accepted = Ratio44.ratio44_orbit_acceptance_reasons(
        fixed_attractors=3, phase_seeds=[0.0, 0.25], period_seeds=[0.9, 1.1],
        transverse_multiplier=0.4 + 0im)
    @test isempty(accepted)
    rejected = Ratio44.ratio44_orbit_acceptance_reasons(
        validation_reasons=[:nonprimitive_period], fixed_attractors=2,
        phase_seeds=[0.0], period_seeds=[1.0], transverse_multiplier=0.99999 + 0im)
    @test :nonprimitive_period in rejected
    @test :three_fixed_attractors_not_confirmed in rejected
    @test :insufficient_phase_seeds in rejected
    @test :insufficient_period_seeds in rejected
    @test :near_neutral_periodic_stability in rejected
end

@testset "Neutral seven-root Hopf fixture" begin
    config = Ratio44.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_ratio44_search.toml"))
    figure5b = Ratio44.Figure5b.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_hopf.toml"))
    state = [0.329934171847263, 0.367295168265613]
    diagnostics = hopf_diagnostics(Ratio44.Figure5b.candidate_model(figure5b, 0.5), state)
    @test Ratio44.fixed_ratio_lyapunov_assessment(
        diagnostics, config.hopf_options).resolved
    basis = Ratio44.Figure5b.hopf_basis(
        Ratio44.Figure5b.candidate_model(figure5b, diagnostics.critical_ratio), state,
        diagnostics.frequency)
    schedule = Ratio44._shooting_seed_schedule(config, state, basis, 0.01,
        diagnostics.linear_period)
    @test all(item -> eltype(item.initial_state) == Float64 &&
        all(isfinite, item.initial_state), schedule)
    searches = Ratio44.Figure5b.equilibrium_refinements(figure5b,
        diagnostics.critical_ratio)
    topology = Ratio44.neutral_hopf_topology(searches, state, config)
    @test topology.qualified
    @test isempty(topology.reasons)
    @test topology.central_index == 3

    trace_at_ratio = function (ratio)
        jacobian = zeros(2, 2)
        point_jacobian!(jacobian, state,
            Ratio44.Figure5b.candidate_model(figure5b, ratio), 0.0)
        return jacobian[1, 1] + jacobian[2, 2]
    end
    trace_step = 1e-5
    trace_derivative = (trace_at_ratio(diagnostics.critical_ratio + trace_step) -
        trace_at_ratio(diagnostics.critical_ratio - trace_step)) / (2trace_step)
    inside_target = 0.5config.topology.neutral_trace_atol
    inside_ratio = diagnostics.critical_ratio + inside_target / trace_derivative
    inside_searches = Ratio44.Figure5b.equilibrium_refinements(figure5b,
        inside_ratio)
    inside_topology = Ratio44.neutral_hopf_topology(inside_searches, state, config)
    @test inside_topology.qualified
    @test all(classification -> classification != StabilityUnresolved,
        inside_topology.tracks[inside_topology.central_index].classifications)
    @test all(trace -> config.topology.spectral_margin < abs(trace) <=
        config.topology.neutral_trace_atol,
        inside_topology.tracks[inside_topology.central_index].traces)

    outside_target = 1.5config.topology.neutral_trace_atol
    outside_ratio = diagnostics.critical_ratio + outside_target / trace_derivative
    outside_searches = Ratio44.Figure5b.equilibrium_refinements(figure5b,
        outside_ratio)
    outside_topology = Ratio44.neutral_hopf_topology(outside_searches, state, config)
    @test !outside_topology.qualified
    @test :central_neutral_quality_failure in outside_topology.reasons
    @test :central_trace_outside_neutral_tolerance in outside_topology.reasons
    wrong_center = Ratio44.neutral_hopf_topology(searches, state .+ 0.1, config)
    @test !wrong_center.qualified
    @test :central_root_not_matched in wrong_center.reasons

    ordinary_searches = Ratio44.Figure5b.equilibrium_refinements(figure5b,
        figure5b.ratios.manuscript)
    ordinary_topology = classify_figure5b_topology(ordinary_searches;
        options=config.topology_options)
    @test ordinary_topology.qualified
    third = ordinary_searches[3]
    permutation = vcat(2:length(third.equilibria), 1)
    permuted_third = EquilibriumSearchResult(third.model, third.frozen_model,
        third.frozen_drive, third.source_time, third.options, third.stability_options,
        third.attempts, third.equilibria[permutation], third.unresolved_nearby,
        third.completeness)
    permuted_searches = (ordinary_searches[1], ordinary_searches[2], permuted_third)
    permuted_topology = classify_figure5b_topology(permuted_searches;
        options=config.topology_options)
    @test permuted_topology.qualified
    directional_tracks = Ratio44._directional_root_tracks(permuted_searches, config)
    @test directional_tracks.qualified
    @test length(directional_tracks.tracks) == 7
    @test all(track -> all(isfinite, track.traces), directional_tracks.tracks)
    axis_tracks = Ratio44._axis_root_tracks(permuted_searches, config)
    @test axis_tracks.qualified
    @test length(axis_tracks.tracks) == 7
    @test permuted_topology.central_state ≈ ordinary_topology.central_state atol=1e-12
    wrapper = (; topology=permuted_topology, searches=permuted_searches,
        parameters=nothing)
    mapped_center = Ratio44._mapped_shooting_center(wrapper)
    @test mapped_center ≈ permuted_topology.central_state atol=1e-12
    unrelated = permuted_third.equilibria[permuted_topology.central_track].state
    @test maximum(abs, unrelated .- mapped_center) > 1e-3
    mapped_schedule = Ratio44._shooting_seed_schedule(config, mapped_center, basis,
        0.01, diagnostics.linear_period)
    first_seed = first(mapped_schedule)
    displacement = 2 .* real.(basis.q .* (0.01 * first_seed.amplitude_factor *
        cis(2pi * first_seed.phase)))
    @test first_seed.initial_state ≈ mapped_center .+ displacement atol=1e-12
    @test maximum(abs, first_seed.initial_state .-
        (unrelated .+ displacement)) > 1e-3
end

@testset "Real-anchor bounded gates" begin
    config = Ratio44.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_ratio44_search.toml"))
    parameters = (e_to_e=19.0, i_to_e=13.0, e_to_i=19.0, i_to_i=6.0,
        theta_off=8.0)
    seed = (hypothesis_id="anchor", parameters,
        central_state=(0.329934171847263, 0.367295168265613))
    axis = Ratio44.continue_axis_path(config, seed, :e_to_e; smoke=true)
    @test isempty(axis.brackets)
    @test any(attempt -> attempt.accepted, axis.attempts)
    @test all(attempt -> !attempt.accepted ||
        (attempt.identity_central_track != 0 &&
            attempt.identity_central_track == attempt.identity_solved_track),
        axis.attempts)
    curve = Ratio44._curve_seed_candidates(config, seed, (:e_to_e, :i_to_i),
        NamedTuple[])
    @test curve.closest.success
    @test config.bounds[:e_to_e][1] <= curve.closest.candidate[3] <=
        config.bounds[:e_to_e][2]
    @test config.bounds[:i_to_i][1] <= curve.closest.candidate[4] <=
        config.bounds[:i_to_i][2]
    candidate_parameters = merge(parameters,
        (; e_to_e=curve.closest.candidate[3], i_to_i=curve.closest.candidate[4]))
    candidate_state = curve.closest.candidate[1:2]
    model = Ratio44.model_at(config, candidate_parameters)
    @test Ratio44._hopf_local_gate(model, candidate_state, config).qualified
    searches = Ratio44.equilibrium_refinements(model, config)
    topology = Ratio44.neutral_hopf_topology(searches, candidate_state, config)
    @test !topology.qualified
    @test :root_count_mismatch in topology.reasons
end

@testset "Finite zero-candidate smoke artifact and replay" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "figure5b_ratio44_search.toml")
    config = Ratio44.load_config(config_path)
    mktempdir() do directory
        artifact = ratio44_fixture_artifact(joinpath(directory, "seed"), config; seed=false)
        blocked_output = joinpath(@__DIR__, "ratio44_unignored_guard_output")
        @test !ispath(blocked_output)
        @test_throws ArgumentError Ratio44.run_experiment(config_path, artifact,
            blocked_output; smoke=true)
        @test !ispath(blocked_output)
        nested_seed_output = joinpath(artifact, "nested", "run")
        @test_throws ArgumentError Ratio44.run_experiment(config_path, artifact,
            nested_seed_output; smoke=true)
        @test !ispath(nested_seed_output)
        first_output = joinpath(directory, "first")
        second_output = joinpath(directory, "second")
        first = Ratio44.run_experiment(config_path, artifact, first_output; smoke=true)
        second = Ratio44.run_experiment(config_path, artifact, second_output; smoke=true)
        @test first.success && second.success
        @test first.outcome == "evidence_ineligible"
        @test first.trace_zero == first.validated_orbits == 0
        for filename in ("seed_manifest.csv", "seed_topologies.csv", "axis_branches.csv",
            "curve_branches.csv", "trace_zero_locations.csv", "accepted_orbits.csv",
            "failures.csv")
            @test read(joinpath(first_output, filename)) == read(joinpath(second_output, filename))
        end
        metadata = Ratio44.TOML.parsefile(joinpath(first_output, "metadata.toml"))
        @test metadata["execution_success"]
        @test metadata["numerical_outcome"] == "completed_no_trace_zero_solution"
        @test metadata["smoke_scientific_acceptance"] == "disabled"
        @test metadata["absence_claim"] == "not_permitted"
        @test occursin("no absence", metadata["scientific_outcome"])
        checksums = Ratio44.TOML.parsefile(joinpath(first_output, "checksums.toml"))["files"]
        @test all(relative -> Ratio44.Evidence.file_hash(joinpath(first_output, relative)) ==
            checksums[relative], keys(checksums))
        rm(artifact; recursive=true)
        replay_output = joinpath(directory, "replay")
        replay_command = `$(Base.julia_cmd()) --startup-file=no --project=source source/scripts/run_figure5b_ratio44_search.jl --config config.toml --replay-artifact . --output $replay_output --smoke`
        cd(first_output) do
            run(replay_command)
        end
        @test read(joinpath(first_output, "seed_manifest.csv")) ==
            read(joinpath(replay_output, "seed_manifest.csv"))
        @test read(joinpath(first_output, "trace_zero_locations.csv")) ==
            read(joinpath(replay_output, "trace_zero_locations.csv"))
        @test Ratio44._verify_checksums(first_output) isa String
        @test_throws ArgumentError Ratio44.run_experiment(config_path, nothing,
            joinpath(first_output, "nested"); smoke=true, replay_artifact=first_output)
        @test_throws ArgumentError Ratio44.run_experiment(config_path, nothing,
            first_output; smoke=true)
    end
end

@testset "Bounded real-anchor runner smoke and archived replay" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "figure5b_ratio44_search.toml")
    config = Ratio44.load_config(config_path)
    mktempdir() do directory
        upstream = ratio44_fixture_artifact(joinpath(directory, "seed"), config; seed=true)
        output = joinpath(directory, "anchor")
        result = Ratio44.run_experiment(config_path, upstream, output; smoke=true)
        @test result.success
        @test result.seeds == result.qualifying_seeds == 1
        @test length(collect(Ratio44.CSV.File(joinpath(output, "axis_branches.csv")))) == 5
        @test !isempty(collect(Ratio44.CSV.File(joinpath(output, "curve_branches.csv"))))
        @test !isempty(collect(Ratio44.CSV.File(joinpath(output, "curve_points.csv"))))
        @test !isempty(collect(Ratio44.CSV.File(joinpath(output,
            "trace_zero_locations.csv"))))
        @test isempty(collect(Ratio44.CSV.File(joinpath(output, "failures.csv"))))
        metadata = Ratio44.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test metadata["outcome"] == "evidence_ineligible"
        @test metadata["numerical_outcome"] == "completed_unresolved_targeted_search"
        rm(upstream; recursive=true)
        replay = joinpath(directory, "anchor_replay")
        Ratio44.run_experiment(joinpath(output, "config.toml"), nothing, replay;
            smoke=true, replay_artifact=output)
        for filename in ("seed_manifest.csv", "seed_topologies.csv", "axis_branches.csv",
            "axis_points.csv", "axis_attempts.csv", "curve_branches.csv",
            "curve_points.csv", "curve_attempts.csv", "trace_zero_locations.csv",
            "shooting_proposals.csv", "accepted_orbits.csv", "failures.csv")
            @test read(joinpath(output, filename)) == read(joinpath(replay, filename))
        end
        @test Ratio44._verify_checksums(output) isa String
    end
end

@testset "Fixed-ratio CLI validation" begin
    @test_throws ArgumentError Ratio44.main(String[])
    @test_throws ArgumentError Ratio44.main(["--seed-artifact", "x"])
    @test_throws ArgumentError Ratio44.main(["--smoke", "--smoke"])
    @test_throws ArgumentError Ratio44.main(["--unknown"])
end
