include(joinpath(@__DIR__, "..", "scripts", "run_figure5b_protocol.jl"))
using LinearAlgebra: norm
import TOML

const Figure5b = Figure5bProtocol

@testset "Canonical Figure-5b protocol configuration" begin
    path = joinpath(@__DIR__, "..", "experiments", "figure5b_hopf.toml")
    config = Figure5b.load_config(path)
    @test config.candidate == (tau_e=7.8, excitatory_slope=5.0,
        inhibitory_slope=5.0, theta_e=1.5, theta_on=4.0, theta_off=8.0,
        e_to_e=19.0, i_to_e=13.0, e_to_i=19.0, i_to_i=6.0)
    @test config.ratios.manuscript == 4.4
    @test config.ratios.below_hopf == 0.4
    @test config.ratios.orbit_anchor == 0.5
    @test config.ratios.continuation_bounds == (0.2, 8.0)
    @test config.root_grids == [11, 21, 41]
    @test config.continuation_options.state_scales == (0.1, 0.1)
    @test config.smoke.phase_fractions == [0.0, 0.5]
    @test config.smoke.period_factors == [0.9, 1.1]

    model = Figure5b.candidate_model(config, 0.5)
    @test model.excitatory.timescale == 7.8
    @test model.inhibitory.timescale == 3.9
    @test model.coupling == PointCoupling(19.0, 13.0, 19.0, 6.0)
    @test model.inhibitory.response.failure_threshold == 8.0
    @test model.drive isa NoDrive
    @test drive_value(model.drive, 0.0) == (0.0, 0.0)

    @test_throws ArgumentError Figure5b.main(String[])
    @test_throws ArgumentError Figure5b.main(["--smoke", "--smoke"])
    @test_throws ArgumentError Figure5b.main(["--output"])
    @test_throws ArgumentError Figure5b.main(["--output", "unused"])
    @test_throws ArgumentError Figure5b.normalize_accepted_revision("abc")
    @test_throws ArgumentError Figure5b.main(["--unknown"])

    mktempdir() do directory
        for mutate in (
            raw -> raw["schema_version"] = true,
            raw -> raw["unexpected"] = true,
            raw -> raw["candidate"]["tau_e"] = true,
            raw -> raw["ratios"]["hopf_bracket"] = [0.5, 0.4],
            raw -> raw["search"]["root_grids"] = [11, 41, 21],
            raw -> raw["seeding"]["phase_fractions"] = [0.0, 1.0],
            raw -> raw["shooting"]["samples"] = true,
            raw -> raw["continuation"]["state_scales"] = [0.1],
            raw -> raw["independent"]["tighter_factor"] = 1.0,
            raw -> raw["smoke"]["phase_fractions"] = [0.125],
        )
            raw = deepcopy(config.raw)
            mutate(raw)
            invalid = joinpath(directory, "invalid.toml")
            Figure5b.Evidence.write_toml(invalid, raw)
            @test_throws ArgumentError Figure5b.load_config(invalid;
                require_canonical=false)
        end
        noncanonical = deepcopy(config.raw)
        noncanonical["candidate"]["e_to_i"] = 19.5
        changed = joinpath(directory, "changed.toml")
        Figure5b.Evidence.write_toml(changed, noncanonical)
        @test_throws ArgumentError Figure5b.load_config(changed)
        @test Figure5b.load_config(changed; require_canonical=false).candidate.e_to_i == 19.5
    end
end

@testset "Figure-5b topology, trace and deterministic Hopf seeds" begin
    config = Figure5b.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_hopf.toml"))
    searches = Figure5b.equilibrium_refinements(config, config.ratios.manuscript)
    topology = classify_figure5b_topology(searches; options=config.topology_options)
    @test topology.qualified
    @test topology.central_state ≈ [0.329934171847263, 0.367295168265613] atol=1e-10
    diagnostics = hopf_diagnostics(Figure5b.candidate_model(config, 4.4),
        topology.central_state; options=config.hopf_options)
    @test diagnostics.resolved
    trace = Figure5b.numerical_trace_zero(config, topology.central_state)
    @test trace.resolved
    @test trace.ratio ≈ diagnostics.critical_ratio atol=config.independent.trace_agreement_atol

    basis = Figure5b.hopf_basis(
        Figure5b.candidate_model(config, diagnostics.critical_ratio),
        topology.central_state, diagnostics.frequency)
    @test norm(basis.q) ≈ 1.0 atol=1e-12
    @test basis.normalization ≈ 1.0 + 0im atol=1e-12
    @test imag(basis.q[argmax(abs.(basis.q))]) ≈ 0.0 atol=1e-12
    @test real(basis.q[argmax(abs.(basis.q))]) > 0
    first_schedule = Figure5b.seed_schedule(config, diagnostics, basis,
        topology.central_state)
    second_schedule = Figure5b.seed_schedule(config, diagnostics, basis,
        topology.central_state)
    @test first_schedule == second_schedule
    @test length(first_schedule) == 3 * 4 * 3 * 3
    @test first_schedule[1].attempt_id == "orbit_0001"
    @test first_schedule[1].ratio ≈ diagnostics.critical_ratio + 0.002
    @test first_schedule[end].ratio == 0.5
    @test length(Figure5b.seed_schedule(config, diagnostics, basis,
        topology.central_state; smoke=true)) == 12

    below = Figure5b.equilibrium_refinements(config, config.ratios.below_hopf)
    assessment = Figure5b.below_hopf_assessment(below, config.topology_options)
    @test assessment.qualified
    @test all(search -> count(root -> root.stability.classification == Attracting,
        search.equilibria) == 4, below)
    mapping = Figure5b.unique_root_mapping(searches[1].equilibria,
        below[1].equilibria, config.topology_options.coordinate_match_atol)
    @test mapping !== nothing

    fake_stability(classification) = (; classification,
        eigenvalues=classification == Attracting ? ComplexF64[-2, -1] :
            ComplexF64[-1, 1])
    fake_roots(classifications) = [(; state=[0.1index, 0.05index],
        near_singular=false, balance_residual=zeros(2),
        stability=fake_stability(classification))
        for (index, classification) in enumerate(classifications)]
    reference_classes = [fill(Attracting, 4); fill(Saddle, 3)]
    flipped_classes = copy(reference_classes)
    flipped_classes[4], flipped_classes[5] = flipped_classes[5], flipped_classes[4]
    fake_searches = ((; equilibria=fake_roots(reference_classes),
            unresolved_nearby=Vector{Vector{Int}}()),
        (; equilibria=fake_roots(flipped_classes),
            unresolved_nearby=Vector{Vector{Int}}()),
        (; equilibria=fake_roots(reference_classes),
            unresolved_nearby=Vector{Vector{Int}}()))
    flipped = Figure5b.below_hopf_assessment(fake_searches,
        config.topology_options)
    @test !flipped.qualified
    @test :stability_track_mismatch in flipped.reasons
end

@testset "Accepted revision and endpoint gates fail closed" begin
    revision = repeat("a", 40)
    accepted = Figure5b.provenance_assessment(revision, revision, "", "HEAD")
    @test accepted.eligible
    @test accepted.detached
    @test accepted.revision_match
    @test accepted.status_porcelain == ""
    @test Figure5b.scientific_evidence_eligible(false, false, true, true,
        nothing, true)
    @test !Figure5b.scientific_evidence_eligible(false, false, true, true,
        nothing, false)
    diagnostics = (resolved=true, classification=:supercritical_candidate)
    @test Figure5b.supercritical_hopf_candidate(true, true, diagnostics,
        true, true, true, true, Dict{String,Any}())
    @test !Figure5b.supercritical_hopf_candidate(true, false, diagnostics,
        true, true, true, true, Dict{String,Any}())
    for (actual, status, head, reason) in (
        (revision, " M file", "HEAD", :working_tree_dirty),
        (revision, "", "codex/branch", :head_not_detached),
        (repeat("b", 40), "", "HEAD", :revision_mismatch),
    )
        result = Figure5b.provenance_assessment(revision, actual, status, head)
        @test !result.eligible
        @test reason in result.reasons
    end
    @test :accepted_revision_missing in Figure5b.provenance_assessment(
        nothing, revision, "", "HEAD").reasons

    mktempdir() do directory
        config_path = joinpath(@__DIR__, "..", "experiments", "figure5b_hopf.toml")
        config = Figure5b.load_config(config_path)
        snapshot_status = " M preexisting-review-fixture"
        snapshot = Figure5b.provenance_assessment(revision, revision,
            snapshot_status, "codex/review-fixture")
        metadata, archived = Figure5b._archive_provenance(config_path,
            joinpath(directory, "archive"), config, false, snapshot, true, nothing)
        @test archived == snapshot
        @test metadata["git_revision"] == revision
        @test metadata["git_status_porcelain"] == snapshot_status
        @test metadata["head_name"] == "codex/review-fixture"
        @test !metadata["clean_working_tree"]
        @test !metadata["provenance_eligible"]
        @test sort(metadata["provenance_reasons"]) ==
            sort(string.([:head_not_detached, :working_tree_dirty]))
    end

    mktempdir() do directory
        output = joinpath(directory, "provenance_failure")
        result = Figure5b.run_experiment(joinpath(@__DIR__, "..", "experiments",
            "figure5b_hopf.toml"), output; accepted_revision=repeat("f", 40))
        @test !result.success
        metadata = TOML.parsefile(joinpath(output, "metadata.toml"))
        @test !metadata["execution_success"]
        @test metadata["accepted_revision_expected"] == repeat("f", 40)
        @test metadata["stage_status"]["provenance"] == "ineligible"
        @test metadata["run_status"] == "provenance_ineligible"
        @test isfile(joinpath(output, "checksums.toml"))
    end

    config = Figure5b.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_hopf.toml"))
    r_h, period = 0.43, 3.84
    observation(ratio, radius, multiplier=0.5; validated=true, attracting=true,
        saddle_distance=1.0, boundary_distance=0.1, period_value=period,
        branch="negative", divergence_resolved=true,
        divergence_multiplier=multiplier, divergence_discrepancy=0.0,
        primitive=true, primitive_alias=0) =
        (; ratio, modal_radius=radius, radius_squared=radius^2,
            period=period_value, transverse_multiplier=multiplier, validated, attracting,
            divergence_resolved, divergence_multiplier, divergence_discrepancy,
            primitive, primitive_alias,
            saddle_distance, boundary_distance, branch, point=1)
    divergence_observation = Figure5b.branch_divergence_observation((;
        divergence_check=(resolved=false, divergence_multiplier=0.4,
            discrepancy=0.2)))
    @test divergence_observation == (divergence_resolved=false,
        divergence_multiplier=0.4, divergence_discrepancy=0.2)
    primitive_observation = Figure5b.branch_primitive_observation((;
        primitive_period_check=(primitive=false, alias_divisor=2)))
    @test primitive_observation == (primitive=false, primitive_alias=2)
    hopf_rows = [observation(0.50, 0.03), observation(0.46, 0.015),
        observation(r_h + config.independent.hopf_target_offset, 0.001)]
    hopf = Figure5b.endpoint_classification(:parameter_boundary, hopf_rows,
        Int[], r_h, period, r_h + config.independent.hopf_target_offset, config)
    @test hopf.classification == :hopf_compatible
    @test hopf.hopf_compatible
    repelling_hopf_rows = copy(hopf_rows)
    repelling_hopf_rows[end] = merge(repelling_hopf_rows[end], (; attracting=false))
    repelling_hopf = Figure5b.endpoint_classification(:parameter_boundary,
        repelling_hopf_rows, Int[], r_h, period,
        r_h + config.independent.hopf_target_offset, config)
    @test !repelling_hopf.hopf_compatible
    @test :nonattracting_branch_orbit in repelling_hopf.reasons
    unresolved_hopf_rows = copy(hopf_rows)
    unresolved_hopf_rows[2] = merge(unresolved_hopf_rows[2],
        (; divergence_resolved=false))
    unresolved_hopf = Figure5b.endpoint_classification(:parameter_boundary,
        unresolved_hopf_rows, Int[], r_h, period,
        r_h + config.independent.hopf_target_offset, config)
    @test !unresolved_hopf.hopf_compatible
    @test :divergence_multiplier_unresolved in unresolved_hopf.reasons
    nonprimitive_hopf_rows = copy(hopf_rows)
    nonprimitive_hopf_rows[2] = merge(nonprimitive_hopf_rows[2],
        (; primitive=false, primitive_alias=2))
    nonprimitive_hopf = Figure5b.endpoint_classification(:parameter_boundary,
        nonprimitive_hopf_rows, Int[], r_h, period,
        r_h + config.independent.hopf_target_offset, config)
    @test !nonprimitive_hopf.hopf_compatible
    @test :nonprimitive_branch_orbit in nonprimitive_hopf.reasons
    unresolved = Figure5b.endpoint_classification(:minimum_step, hopf_rows,
        Int[], r_h, period, nothing, config)
    @test unresolved.classification == :unresolved
    @test :unsuccessful_termination in unresolved.reasons
    reversal_only = Figure5b.endpoint_classification(:step_limit, hopf_rows,
        [2], r_h, period, nothing, config)
    @test reversal_only.classification == :unresolved
    fold_rows = [observation(0.50, 0.03, 1.0),
        observation(0.48, 0.025, 1.0), observation(0.49, 0.026, 1.0)]
    fold = Figure5b.endpoint_classification(:step_limit, fold_rows,
        [2], r_h, period, nothing, config)
    @test fold.classification == :fold_compatible
    @test fold.fold_compatible
    unresolved_fold_rows = copy(fold_rows)
    unresolved_fold_rows[2] = merge(unresolved_fold_rows[2],
        (; divergence_resolved=false))
    unresolved_fold = Figure5b.endpoint_classification(:step_limit,
        unresolved_fold_rows, [2], r_h, period, nothing, config)
    @test !unresolved_fold.fold_compatible
    nonprimitive_fold_rows = copy(fold_rows)
    nonprimitive_fold_rows[2] = merge(nonprimitive_fold_rows[2],
        (; primitive=false, primitive_alias=2))
    nonprimitive_fold = Figure5b.endpoint_classification(:step_limit,
        nonprimitive_fold_rows, [2], r_h, period, nothing, config)
    @test !nonprimitive_fold.fold_compatible
    earlier_fold_rows = vcat(fold_rows,
        [observation(0.47, 0.03, 0.5), observation(0.45, 0.03, 0.5)])
    earlier_fold = Figure5b.endpoint_classification(:step_limit,
        earlier_fold_rows, [2], r_h, period, nothing, config)
    @test !earlier_fold.fold_compatible
    @test earlier_fold.classification == :unresolved
    @test :unresolved_parameter_reversal in earlier_fold.reasons
    final_point_reversal = Figure5b.endpoint_classification(:step_limit,
        fold_rows, [length(fold_rows)], r_h, period, nothing, config)
    @test !final_point_reversal.fold_compatible
    @test final_point_reversal.classification == :unresolved
    @test :unresolved_parameter_reversal in final_point_reversal.reasons
    earlier_fold_then_homoclinic_rows = vcat(fold_rows,
        [observation(0.47, 0.04, 0.5; saddle_distance=0.1,
             period_value=2period),
         observation(0.45, 0.05, 0.5; saddle_distance=0.05,
             period_value=4period),
         observation(0.43, 0.06, 0.5; saddle_distance=0.01,
             period_value=6period)])
    earlier_fold_then_homoclinic = Figure5b.endpoint_classification(
        :minimum_step, earlier_fold_then_homoclinic_rows, [2], r_h, period,
        nothing, config)
    @test !earlier_fold_then_homoclinic.fold_compatible
    @test earlier_fold_then_homoclinic.classification ==
        :homoclinic_or_heteroclinic_compatible
    homoclinic_rows = [observation(0.5, 0.03; saddle_distance=0.1,
            period_value=period),
        observation(0.47, 0.04; saddle_distance=0.05,
            period_value=2period),
        observation(0.45, 0.05; saddle_distance=0.01,
            period_value=6period)]
    homoclinic = Figure5b.endpoint_classification(:minimum_step,
        homoclinic_rows, Int[], r_h, period, nothing, config)
    @test homoclinic.classification == :homoclinic_or_heteroclinic_compatible
    unresolved_homoclinic_rows = copy(homoclinic_rows)
    unresolved_homoclinic_rows[2] = merge(unresolved_homoclinic_rows[2],
        (; divergence_resolved=false))
    unresolved_homoclinic = Figure5b.endpoint_classification(:minimum_step,
        unresolved_homoclinic_rows, Int[], r_h, period, nothing, config)
    @test !unresolved_homoclinic.homoclinic_compatible
    nonprimitive_homoclinic_rows = copy(homoclinic_rows)
    nonprimitive_homoclinic_rows[2] = merge(nonprimitive_homoclinic_rows[2],
        (; primitive=false, primitive_alias=2))
    nonprimitive_homoclinic = Figure5b.endpoint_classification(:minimum_step,
        nonprimitive_homoclinic_rows, Int[], r_h, period, nothing, config)
    @test !nonprimitive_homoclinic.homoclinic_compatible
    increasing_homoclinic_rows = [observation(0.45, 0.03;
            saddle_distance=0.1, period_value=period, branch="positive"),
        observation(0.47, 0.04; saddle_distance=0.05,
            period_value=2period, branch="positive"),
        observation(0.5, 0.05; saddle_distance=0.01,
            period_value=6period, branch="positive")]
    increasing_homoclinic = Figure5b.endpoint_classification(:minimum_step,
        increasing_homoclinic_rows, Int[], r_h, period, nothing, config)
    @test increasing_homoclinic.classification ==
        :homoclinic_or_heteroclinic_compatible
    state_boundary_rows = [observation(0.5, 0.03; boundary_distance=0.1),
        observation(0.47, 0.04; boundary_distance=0.01),
        observation(0.45, 0.05; boundary_distance=1e-5)]
    state_boundary = Figure5b.endpoint_classification(:step_limit,
        state_boundary_rows, Int[], r_h, period, nothing, config)
    @test state_boundary.classification == :state_boundary_compatible
    unresolved_state_boundary_rows = copy(state_boundary_rows)
    unresolved_state_boundary_rows[2] = merge(unresolved_state_boundary_rows[2],
        (; divergence_resolved=false))
    unresolved_state_boundary = Figure5b.endpoint_classification(:step_limit,
        unresolved_state_boundary_rows, Int[], r_h, period, nothing, config)
    @test !unresolved_state_boundary.state_boundary_compatible
    nonprimitive_state_boundary_rows = copy(state_boundary_rows)
    nonprimitive_state_boundary_rows[2] = merge(
        nonprimitive_state_boundary_rows[2],
        (; primitive=false, primitive_alias=2))
    nonprimitive_state_boundary = Figure5b.endpoint_classification(:step_limit,
        nonprimitive_state_boundary_rows, Int[], r_h, period, nothing, config)
    @test !nonprimitive_state_boundary.state_boundary_compatible
    increasing_state_boundary_rows = [observation(0.45, 0.03;
            boundary_distance=0.1, branch="positive"),
        observation(0.47, 0.04; boundary_distance=0.01, branch="positive"),
        observation(0.5, 0.05; boundary_distance=1e-5, branch="positive")]
    increasing_state_boundary = Figure5b.endpoint_classification(:step_limit,
        increasing_state_boundary_rows, Int[], r_h, period, nothing, config)
    @test increasing_state_boundary.classification == :state_boundary_compatible
    boundary = Figure5b.endpoint_classification(:parameter_boundary,
        [observation(0.5, 0.03), observation(0.2, 0.04)], Int[], r_h,
        period, 0.2, config)
    @test boundary.classification == :boundary

    fit_rows = [observation(r_h + offset, sqrt(0.1offset))
        for offset in (0.04, 0.03, 0.02, 0.01)]
    disconnected = observation(0.09, 0.5)
    observations = Dict(:negative => vcat(fit_rows, [disconnected]),
        :positive => [observation(0.5, 0.03), observation(0.55, 0.04)])
    continuation = (negative=(parameter_reversals=[5],),
        positive=(parameter_reversals=Int[],))
    fit = Figure5b.branch_fit(continuation, observations, r_h, config)
    @test fit.resolved
    @test fit.branch == "negative"
    @test fit.points == 4
    @test all(row -> row.ratio > r_h, fit.rows)
    mismatched = merge(fit, (; branch="positive"))
    assessment = Figure5b.fit_acceptance_assessment(mismatched,
        Dict(:negative => hopf),
        (critical_ratio=r_h, linear_period=period,
            modal_radius_squared_slope=fit.slope), config)
    @test !assessment.accepted
    @test :fit_endpoint_branch_missing in assessment.reasons

    near_offsets = (4e-4, 3e-4, 2e-4, 1e-4)
    accepted_fit_rows = [observation(r_h + offset, sqrt(0.1offset))
        for offset in near_offsets]
    fit_observations = Dict(:negative => accepted_fit_rows,
        :positive => [observation(0.5, 0.03), observation(0.55, 0.04)])
    fit_continuation = (negative=(parameter_reversals=Int[],),
        positive=(parameter_reversals=Int[],))
    accepted_fit = Figure5b.branch_fit(fit_continuation, fit_observations,
        r_h, config)
    accepted_endpoint = Figure5b.endpoint_classification(:parameter_boundary,
        accepted_fit_rows, Int[], r_h, period, last(accepted_fit_rows).ratio,
        config)
    fit_diagnostics = (critical_ratio=r_h, linear_period=period,
        modal_radius_squared_slope=0.1)
    accepted_assessment = Figure5b.fit_acceptance_assessment(accepted_fit,
        Dict(:negative => accepted_endpoint), fit_diagnostics, config)
    @test accepted_assessment.accepted

    bumped_rows = copy(accepted_fit.rows)
    bumped_radius = 1.1 * bumped_rows[2].modal_radius
    bumped_rows[3] = merge(bumped_rows[3],
        (; modal_radius=bumped_radius, radius_squared=bumped_radius^2))
    bumped_fit = merge(accepted_fit, (; rows=bumped_rows))
    bumped_assessment = Figure5b.fit_acceptance_assessment(bumped_fit,
        Dict(:negative => accepted_endpoint), fit_diagnostics, config)
    @test !bumped_assessment.accepted
    @test :radius_not_strictly_decreasing_to_endpoint in bumped_assessment.reasons

    repelling_fit_rows = copy(accepted_fit_rows)
    repelling_fit_rows[2] = merge(repelling_fit_rows[2], (; attracting=false))
    repelling_fit = Figure5b.branch_fit(fit_continuation,
        Dict(:negative => repelling_fit_rows,
            :positive => fit_observations[:positive]), r_h, config)
    repelling_fit_assessment = Figure5b.fit_acceptance_assessment(
        repelling_fit, Dict(:negative => accepted_endpoint), fit_diagnostics,
        config)
    @test !repelling_fit_assessment.accepted
    @test :fit_nonattracting_orbit in repelling_fit_assessment.reasons

    unresolved_fit_rows = copy(accepted_fit_rows)
    unresolved_fit_rows[2] = merge(unresolved_fit_rows[2],
        (; divergence_resolved=false))
    unresolved_fit = Figure5b.branch_fit(fit_continuation,
        Dict(:negative => unresolved_fit_rows,
            :positive => fit_observations[:positive]), r_h, config)
    @test unresolved_fit.resolved
    @test unresolved_fit.points == length(unresolved_fit_rows)
    @test any(row -> !row.divergence_resolved, unresolved_fit.rows)
    unresolved_fit_assessment = Figure5b.fit_acceptance_assessment(
        unresolved_fit, Dict(:negative => accepted_endpoint), fit_diagnostics,
        config)
    @test !unresolved_fit_assessment.accepted
    @test :fit_divergence_unresolved in unresolved_fit_assessment.reasons

    nonprimitive_fit_rows = copy(accepted_fit_rows)
    nonprimitive_fit_rows[2] = merge(nonprimitive_fit_rows[2],
        (; primitive=false, primitive_alias=2))
    nonprimitive_fit = Figure5b.branch_fit(fit_continuation,
        Dict(:negative => nonprimitive_fit_rows,
            :positive => fit_observations[:positive]), r_h, config)
    @test nonprimitive_fit.resolved
    @test nonprimitive_fit.points == length(nonprimitive_fit_rows)
    @test any(row -> !row.primitive && row.primitive_alias == 2,
        nonprimitive_fit.rows)
    nonprimitive_fit_assessment = Figure5b.fit_acceptance_assessment(
        nonprimitive_fit, Dict(:negative => accepted_endpoint), fit_diagnostics,
        config)
    @test !nonprimitive_fit_assessment.accepted
    @test :fit_nonprimitive_orbit in nonprimitive_fit_assessment.reasons

    supplied_rows = [(; ratio=row.ratio, modal_radius=row.modal_radius,
        radius_squared=row.radius_squared, period=row.period,
        validated=row.validated, divergence_resolved=row.divergence_resolved,
        divergence_multiplier=row.divergence_multiplier,
        divergence_discrepancy=row.divergence_discrepancy)
        for row in accepted_fit.rows]
    supplied_fit = merge(accepted_fit, (; rows=supplied_rows))
    supplied_assessment = Figure5b.fit_acceptance_assessment(supplied_fit,
        Dict(:negative => accepted_endpoint), fit_diagnostics, config)
    @test !supplied_assessment.accepted
    @test :fit_nonprimitive_orbit in supplied_assessment.reasons

    legacy_divergence_rows = [(; ratio=row.ratio,
        modal_radius=row.modal_radius, radius_squared=row.radius_squared,
        period=row.period, validated=row.validated, primitive=row.primitive,
        primitive_alias=row.primitive_alias) for row in accepted_fit.rows]
    legacy_divergence_fit = merge(accepted_fit,
        (; rows=legacy_divergence_rows))
    legacy_divergence_assessment = Figure5b.fit_acceptance_assessment(
        legacy_divergence_fit, Dict(:negative => accepted_endpoint),
        fit_diagnostics, config)
    @test !legacy_divergence_assessment.accepted
    @test :fit_divergence_unresolved in legacy_divergence_assessment.reasons
end

@testset "Injected stage failures remain checksummed" begin
    failure = Figure5b.captured_call(() -> error("injected continuation failure"))
    @test failure.result === nothing
    @test occursin("injected continuation failure", failure.error["message"])
    mktempdir() do output
        metadata = Dict{String,Any}(
            "stage_status" => Dict{String,Any}(),
            "stage_errors" => Dict{String,Any}(),
            "execution_success" => true)
        Figure5b.record_stage_error!(metadata, "continuation", failure.error)
        Figure5b.finalize_artifact!(output, metadata)
        parsed = TOML.parsefile(joinpath(output, "metadata.toml"))
        @test !parsed["execution_success"]
        @test parsed["stage_status"]["continuation"] == "failed"
        checksums = TOML.parsefile(joinpath(output, "checksums.toml"))["files"]
        @test haskey(checksums, "metadata.toml")
        @test checksums["metadata.toml"] == Figure5b.Evidence.file_hash(
            joinpath(output, "metadata.toml"))
    end
end

@testset "Non-git full-mode replay is evidence-ineligible" begin
    mktempdir() do directory
        parent = joinpath(directory, "parent")
        mkpath(joinpath(parent, "source", "scripts"))
        config_source = joinpath(@__DIR__, "..", "experiments", "figure5b_hopf.toml")
        runner_source = joinpath(@__DIR__, "..", "scripts", "run_figure5b_protocol.jl")
        cp(config_source, joinpath(parent, "config.toml"))
        cp(runner_source, joinpath(parent, "source", "scripts",
            "run_figure5b_protocol.jl"))
        metadata = Dict(
            "smoke" => false,
            "config_sha256" => Figure5b.Evidence.file_hash(
                joinpath(parent, "config.toml")),
            "source_sha256" => Dict("scripts/run_figure5b_protocol.jl" =>
                Figure5b.Evidence.file_hash(joinpath(parent, "source", "scripts",
                    "run_figure5b_protocol.jl"))))
        Figure5b.Evidence.write_toml(joinpath(parent, "metadata.toml"), metadata)
        replay_target = joinpath(parent, "replay-target")
        mkpath(replay_target)
        Figure5b.Evidence.artifact_checksums(parent)
        replay_link = joinpath(directory, "replay-link")
        symlink(replay_target, replay_link; dir_target=true)
        @test_throws ArgumentError Figure5b.run_experiment(
            joinpath(parent, "config.toml"), replay_link;
            replay_parent=parent, failure_injection=:equilibria)
        @test isempty(readdir(replay_target))
        link_revalidated = Figure5b.validate_replay_parent(parent,
            joinpath(parent, "config.toml"))
        @test link_revalidated.parent == realpath(parent)
        @test_throws ArgumentError Figure5b.run_experiment(
            joinpath(parent, "config.toml"), joinpath(parent, "replay");
            replay_parent=parent, failure_injection=:equilibria)
        replay_output = joinpath(directory, "replay")
        result = Figure5b.run_experiment(joinpath(parent, "config.toml"),
            replay_output; replay_parent=parent, failure_injection=:equilibria)
        @test !result.success
        @test !result.scientific_acceptance
        replay_metadata = TOML.parsefile(joinpath(replay_output, "metadata.toml"))
        @test replay_metadata["replay_mode"]
        @test !replay_metadata["evidence_eligible"]
        @test replay_metadata["run_status"] == "execution_failed"
        @test replay_metadata["numerical_four_attractor_coexistence"] ==
            "execution_failed"
        @test replay_metadata["replay_parent_checksums_sha256"] ==
            Figure5b.Evidence.file_hash(joinpath(parent, "checksums.toml"))
        @test isfile(joinpath(replay_output, "checksums.toml"))
        revalidated = Figure5b.validate_replay_parent(parent,
            joinpath(parent, "config.toml"))
        @test revalidated.parent == abspath(parent)
    end
end

@testset "Figure-5b orbit acceptance fails closed" begin
    resolved_windings = [(; resolved=true, winding=value) for value in (0, 1, 0)]
    common = (numerically_validated=true, attracting=true, primitive=true,
        equivalent=true, jacobian_resolved=true, divergence_resolved=true,
        poincare_resolved=true, poincare_difference=0.0, multiplier_atol=1e-3,
        windings=resolved_windings, central_index=2)
    @test isempty(Figure5b.orbit_acceptance_reasons(; common...))
    cases = (
        ((; primitive=false), :nonprimitive_period),
        ((; equivalent=false), :tight_replay_mismatch),
        ((; attracting=false), :orbit_not_attracting),
        ((; jacobian_resolved=false), :jacobian_mismatch),
        ((; poincare_resolved=false), :poincare_unresolved),
        ((; poincare_difference=0.1), :poincare_multiplier_mismatch),
        ((; windings=[(; resolved=true, winding=0),
            (; resolved=true, winding=0), (; resolved=true, winding=0)]),
            :wrong_central_winding),
        ((; windings=[(; resolved=true, winding=1),
            (; resolved=true, winding=1), (; resolved=true, winding=0)]),
            :encloses_other_equilibrium),
        ((; windings=[(; resolved=false, winding=0)]), :winding_unresolved),
    )
    for (change, expected) in cases
        values = merge(common, change)
        @test expected in Figure5b.orbit_acceptance_reasons(; values...)
    end

    fake_attempt(id, phase, period) = (; attempt_id=id, phase,
        period_factor=period, result=nothing)
    qualifying = [fake_attempt("a", 0.0, 0.9), fake_attempt("b", 0.25, 0.9),
        fake_attempt("c", 0.0, 1.1), fake_attempt("d", 0.25, 1.1)]
    config = Figure5b.load_config(joinpath(@__DIR__, "..", "experiments",
        "figure5b_hopf.toml"))
    @test Figure5b.qualifying_cluster([qualifying], config) === qualifying
    @test Figure5b.qualifying_cluster([qualifying[1:2]], config) === nothing
end

@testset "Figure-5b exact-candidate smoke artifact and replay" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "figure5b_hopf.toml")
    mktempdir() do directory
        output = joinpath(directory, "output")
        result = Figure5b.run_experiment(config_path, output; smoke=true)
        @test result.success
        @test result.smoke
        @test result.topology_manuscript
        @test result.topology_anchor
        @test result.below_hopf
        @test result.hopf
        @test result.orbit
        @test !result.scientific_acceptance
        metadata = TOML.parsefile(joinpath(output, "metadata.toml"))
        @test metadata["smoke"]
        @test !metadata["scientific_acceptance_enabled"]
        @test metadata["numerical_four_attractor_coexistence"] == "disabled_in_smoke"
        @test metadata["completeness"] == "CompletenessNotCertified"
        @test metadata["stage_status"]["tight_validation"] == "completed"
        @test metadata["stage_status"]["continuation"] == "completed"
        @test haskey(metadata["source_sha256"], "scripts/run_figure5b_protocol.jl")
        checksums = TOML.parsefile(joinpath(output, "checksums.toml"))["files"]
        @test all(relative -> Figure5b.Evidence.file_hash(joinpath(output, relative)) ==
            checksums[relative], keys(checksums))

        deterministic_paths = sort([relpath(joinpath(root, filename), output)
            for (root, _, files) in walkdir(output) for filename in files
            if relpath(joinpath(root, filename), output) ∉
                ("metadata.toml", "checksums.toml")])
        replay_output = joinpath(directory, "replay")
        replay = `$(Base.julia_cmd()) --project=source source/scripts/run_figure5b_protocol.jl --config config.toml --output ../replay --replay-parent .`
        @test success(Cmd(replay; dir=output))
        for relative in deterministic_paths
            @test read(joinpath(output, relative)) ==
                read(joinpath(replay_output, relative))
        end
        revalidated = Figure5b.validate_replay_parent(output,
            joinpath(output, "config.toml"))
        @test revalidated.parent == abspath(output)
    end
end
