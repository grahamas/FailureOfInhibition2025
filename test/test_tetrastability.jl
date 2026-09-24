include(joinpath(@__DIR__, "..", "scripts", "run_tetrastability_search.jl"))

const Tetrastability = TetrastabilityExperiment

function synthetic_tetrastability_root(state; attracting=true, near_singular=false,
    residual=1e-12, spectral=-0.01)
    return (; state=Tuple(state), attracting, near_singular,
        residual_norm=residual, recomputed_residual_norm=residual,
        balance_jacobian_delta=0.0, ode_jacobian_delta=0.0,
        spectral_abscissa=spectral)
end

@testset "Canonical tetrastability configuration and Halton design" begin
    path = joinpath(@__DIR__, "..", "experiments", "tetrastability.toml")
    config = Tetrastability.load_config(path)
    @test config.fixed.tau_e == 7.8
    @test config.fixed.tau_ratio == 4.4
    @test Tetrastability.SEARCH_AXES ==
        (:e_to_e, :i_to_e, :e_to_i, :i_to_i, :theta_off)
    @test config.halton.bases == (2, 3, 5, 7, 11)
    @test config.halton.initial_points == 1024
    @test config.halton.maximum_points == 4096
    @test config.screen_grid_points == 11
    @test config.confirmation_grid_points == [21, 41]

    first_cell = Tetrastability.halton_cell(config, 1)
    units = (first_cell.halton_e_to_e, first_cell.halton_i_to_e,
        first_cell.halton_e_to_i, first_cell.halton_i_to_i,
        first_cell.halton_theta_off)
    @test all(isapprox(observed, expected) for (observed, expected) in
        zip(units, (1 / 2, 1 / 3, 1 / 5, 1 / 7, 1 / 11)))
    @test first_cell.e_to_e == 19.0
    @test first_cell.i_to_e == 10.0
    @test first_cell.e_to_i == 15.2
    @test first_cell.i_to_i ≈ 10 / 7
    @test first_cell.theta_off ≈ 72 / 11
    @test all(index -> 0 < Tetrastability.radical_inverse(index, 2) < 1, 1:100)
    @test Tetrastability.halton_cells(config, 1:1024) ==
        Tetrastability.halton_cells(config, 1:4096)[1:1024]
    @test length(unique(cell.cell_id for cell in Tetrastability.halton_cells(config, 1:4096))) == 4096
    @test_throws ArgumentError Tetrastability.radical_inverse(0, 2)
    @test_throws ArgumentError Tetrastability.radical_inverse(true, 2)
    @test_throws ArgumentError Tetrastability.radical_inverse(1, 4)

    planes = Tetrastability.plane_cells(config)
    @test length(planes) == 2 * 33 * 25 == 1650
    @test Set(cell.plane for cell in planes) == Set(("figure3", "figure4_exploration"))
    @test extrema(cell.e_to_i for cell in planes) == (12.0, 28.0)
    @test extrema(cell.theta_off for cell in planes) == (6.0, 12.0)
    @test collect(Tetrastability.extension_indices(config, false)) == collect(1025:4096)
    @test isempty(Tetrastability.extension_indices(config, true))
    @test isempty(Tetrastability.extension_indices(config, false; smoke=true))

    models = Tetrastability.models_for_cell(config, first_cell)
    @test models.control.coupling === models.failure_of_inhibition.coupling
    @test models.control.excitatory === models.failure_of_inhibition.excitatory
    @test models.control.drive === models.failure_of_inhibition.drive
    @test models.control.inhibitory.timescale == models.failure_of_inhibition.inhibitory.timescale == 34.32
    @test models.control.inhibitory.response.threshold == 4.0
    @test models.failure_of_inhibition.inhibitory.response.failure_threshold == first_cell.theta_off
    @test drive_value(models.control.drive, 0.0) == (0.0, 0.0)

    @test_throws ArgumentError Tetrastability.main(String[])
    @test_throws ArgumentError Tetrastability.main(["--smoke", "--smoke"])
    @test_throws ArgumentError Tetrastability.main(["--output"])
    @test_throws ArgumentError Tetrastability.main(["--unknown"])

    mktempdir() do directory
        for mutate in (
            raw -> raw["schema_version"] = true,
            raw -> raw["unexpected"] = true,
            raw -> raw["fixed"]["tau_ratio"] = 4.5,
            raw -> raw["axes"]["theta_off"]["minimum"] = 4.0,
            raw -> raw["halton"]["bases"] = [2, 3, 5, 7, 7],
            raw -> raw["halton"]["initial_points"] = true,
            raw -> raw["search"]["screen_grid_points"] = 5,
            raw -> raw["search"]["confirmation_grid_points"] = [41, 21],
            raw -> raw["confirmation"]["tolerance_factor"] = 1.0,
            raw -> raw["continuation"]["minimum_step"] = 0.1,
        )
            raw = deepcopy(config.raw)
            mutate(raw)
            invalid = joinpath(directory, "invalid.toml")
            Tetrastability.Evidence.write_toml(invalid, raw)
            output = joinpath(directory, "must_not_exist")
            @test_throws ArgumentError Tetrastability.run_experiment(invalid, output)
            @test !ispath(output)
        end
        noncanonical = deepcopy(config.raw)
        noncanonical["halton"]["initial_points"] = 2
        noncanonical["halton"]["maximum_points"] = 4
        noncanonical_path = joinpath(directory, "noncanonical.toml")
        Tetrastability.Evidence.write_toml(noncanonical_path, noncanonical)
        @test_throws ArgumentError Tetrastability.load_config(noncanonical_path)
        @test Tetrastability.load_config(noncanonical_path; require_canonical=false).halton.maximum_points == 4

        invalid_grid = deepcopy(noncanonical)
        invalid_grid["search"]["screen_grid_points"] = 5
        invalid_grid_path = joinpath(directory, "invalid_grid.toml")
        Tetrastability.Evidence.write_toml(invalid_grid_path, invalid_grid)
        @test_throws ArgumentError Tetrastability.load_config(invalid_grid_path;
            require_canonical=false)
    end
end

@testset "Tetrastability candidate qualification" begin
    config = Tetrastability.load_config(joinpath(@__DIR__, "..", "experiments",
        "tetrastability.toml"))
    roots = [synthetic_tetrastability_root((0.05 + 0.1index, 0.02 + 0.08index))
        for index in 0:3]
    refined_21 = [synthetic_tetrastability_root((root.state[1] + 1e-9,
        root.state[2] - 1e-9)) for root in reverse(roots)]
    refined_41 = [synthetic_tetrastability_root((root.state[1] - 1e-9,
        root.state[2] + 1e-9)) for root in (roots[2], roots[4], roots[1], roots[3])]
    refined = Dict(21 => refined_21, 41 => refined_41)
    unresolved = Dict(0 => false, 21 => false, 41 => false)
    assessment = Tetrastability.confirmation_assessment(roots, refined, unresolved, config)
    @test assessment.confirmed
    @test assessment.matched_attracting_roots == 4
    @test assessment.minimum_separation > 1e-5

    three = roots[1:3]
    failed = Tetrastability.confirmation_assessment(three,
        Dict(21 => refined_21[1:3], 41 => refined_41[1:3]), unresolved, config)
    @test !failed.confirmed
    @test "fewer_than_four_uniquely_matched_attracting_roots" in failed.reasons

    stability_flip = deepcopy(refined)
    stability_flip[41][1] = merge(stability_flip[41][1], (; attracting=false))
    failed = Tetrastability.confirmation_assessment(roots, stability_flip, unresolved, config)
    @test !failed.confirmed

    ambiguous = deepcopy(refined)
    push!(ambiguous[21], synthetic_tetrastability_root(roots[1].state))
    failed = Tetrastability.confirmation_assessment(roots, ambiguous, unresolved, config)
    @test !failed.confirmed

    unresolved_copy = copy(unresolved)
    unresolved_copy[41] = true
    failed = Tetrastability.confirmation_assessment(roots, refined, unresolved_copy, config)
    @test !failed.confirmed
    @test "unresolved_nearby_roots" in failed.reasons

    fake_result(count) = (; equilibria=[(; stability=(; classification=Attracting))
        for _ in 1:count])
    @test Tetrastability._cell_is_candidate(fake_result(4))
    @test !Tetrastability._cell_is_candidate(fake_result(3))
    @test !Tetrastability._cell_is_candidate(nothing)
    compact = Tetrastability._candidate_reference((; search_id="fixture", model=nothing,
        result=(; equilibria=roots, unresolved_nearby=Any[], attempts=fill(nothing, 100))))
    @test !hasproperty(compact.result, :attempts)
    @test compact.result.equilibria === roots

    cell = Tetrastability.halton_cell(config, 1)
    for axis in Tetrastability.SEARCH_AXES
        changed = Tetrastability._replace_axis(cell, axis, getproperty(cell, axis) + 0.01)
        @test getproperty(changed, axis) == getproperty(cell, axis) + 0.01
        @test all(other -> other == axis || getproperty(changed, other) == getproperty(cell, other),
            Tetrastability.SEARCH_AXES)
    end
end

@testset "Reduced tetrastability artifact run" begin
    canonical = Tetrastability.load_config(joinpath(@__DIR__, "..", "experiments",
        "tetrastability.toml"))
    mktempdir() do directory
        raw = deepcopy(canonical.raw)
        raw["planes"] = raw["planes"][1:1]
        raw["halton"]["initial_points"] = 2
        raw["halton"]["maximum_points"] = 4
        raw["search"]["screen_grid_points"] = 6
        raw["smoke"]["halton_points"] = 1
        path = joinpath(directory, "config.toml")
        Tetrastability.Evidence.write_toml(path, raw)
        first_output = joinpath(directory, "first")
        second_output = joinpath(directory, "second")
        first_run = Tetrastability.run_experiment(path, first_output;
            smoke=true, require_canonical=false)
        second_run = Tetrastability.run_experiment(path, second_output;
            smoke=true, require_canonical=false)
        @test first_run.success && second_run.success
        @test first_run.parameter_cells == 2
        @test first_run.unique_searches == 4
        @test first_run.screen_positive == 0
        @test !first_run.extension_run
        @test_throws ArgumentError Tetrastability.run_experiment(path, first_output;
            smoke=true, require_canonical=false)
        for filename in ("samples.csv", "searches.csv", "attempts.csv", "equilibria.csv",
            "candidates.csv", "root_matches.csv", "branches.csv")
            @test read(joinpath(first_output, filename)) == read(joinpath(second_output, filename))
        end
        searches = collect(Tetrastability.CSV.File(joinpath(first_output, "searches.csv")))
        @test length(searches) == 4
        @test Set(row.condition for row in searches) ==
            Set(("control", "failure_of_inhibition"))
        @test all(row -> row.completeness == "CompletenessNotCertified", searches)
        metadata = Tetrastability.TOML.parsefile(joinpath(first_output, "metadata.toml"))
        @test metadata["execution_success"]
        @test metadata["confirmed_cells"] == 0
        @test metadata["absence_claim"] == "not_permitted"
        @test occursin("not resumable", metadata["interruption_policy"])
        @test occursin("not an absence proof", metadata["scientific_outcome"])
        @test haskey(metadata["source_sha256"], "scripts/run_tetrastability_search.jl")
        checksums = Tetrastability.TOML.parsefile(joinpath(first_output, "checksums.toml"))["files"]
        @test all(relative -> Tetrastability.Evidence.file_hash(
            joinpath(first_output, relative)) == checksums[relative], keys(checksums))
        @test Set(keys(checksums)) == Set(relpath(joinpath(root, file), first_output)
            for (root, _, files) in walkdir(first_output) for file in files
            if file != "checksums.toml")

    end
end

@testset "Canonical artifact replay" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "tetrastability.toml")
    mktempdir() do directory
        output = joinpath(directory, "output")
        result = Tetrastability.run_experiment(config_path, output; smoke=true)
        @test result.success
        replay_command = `$(Base.julia_cmd()) --project=source source/scripts/run_tetrastability_search.jl --config config.toml --output replay --smoke`
        replay_succeeded = success(Cmd(replay_command; dir=output))
        @test replay_succeeded
        if replay_succeeded
            for filename in ("config.toml", "samples.csv", "searches.csv", "attempts.csv",
                "equilibria.csv", "candidates.csv", "root_matches.csv", "branches.csv",
                "continuation_points.csv", "continuation_attempts.csv",
                "continuation_candidates.csv")
                @test read(joinpath(output, filename)) ==
                    read(joinpath(output, "replay", filename))
            end
        end
    end
end

@testset "Reduced extension schedule" begin
    canonical = Tetrastability.load_config(joinpath(@__DIR__, "..", "experiments",
        "tetrastability.toml"))
    mktempdir() do directory
        raw = deepcopy(canonical.raw)
        raw["planes"] = raw["planes"][1:1]
        raw["axes"]["e_to_i"] =
            Dict("minimum" => 19.0, "maximum" => 19.5, "plane_step" => 0.5)
        raw["axes"]["theta_off"] =
            Dict("minimum" => 8.0, "maximum" => 8.25, "plane_step" => 0.25)
        raw["halton"]["initial_points"] = 2
        raw["halton"]["maximum_points"] = 4
        raw["search"]["screen_grid_points"] = 6
        config_path = joinpath(directory, "extension.toml")
        Tetrastability.Evidence.write_toml(config_path, raw)
        output = joinpath(directory, "output")
        result = Tetrastability.run_experiment(config_path, output;
            require_canonical=false)
        @test result.success
        @test result.extension_run
        @test result.parameter_cells == 8
        @test result.unique_searches == 14
        @test result.screen_positive == 0
        samples = collect(Tetrastability.CSV.File(joinpath(output, "samples.csv")))
        halton = filter(row -> row.family == "halton", samples)
        @test [row.halton_index for row in halton] == collect(1:4)
        @test [row.stage for row in halton] ==
            ["initial", "initial", "extension", "extension"]
        searches = collect(Tetrastability.CSV.File(joinpath(output, "searches.csv")))
        @test length(searches) == 16
        @test count(row -> row.condition == "control" && row.stage == "plane", searches) == 4
        @test length(unique(row.search_id for row in searches
            if row.condition == "control" && row.stage == "plane")) == 2
        metadata = Tetrastability.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test metadata["last_halton_index"] == 4
        @test metadata["extension_run"]
    end
end

@testset "Candidate confirmation orchestration" begin
    config = Tetrastability.load_config(joinpath(@__DIR__, "..", "experiments",
        "tetrastability.toml"))
    cell = only(filter(candidate -> candidate.plane == "figure3" &&
        candidate.e_to_i == 19.0 && candidate.theta_off == 8.0,
        Tetrastability.plane_cells(config)))
    model = Tetrastability.models_for_cell(config, cell).failure_of_inhibition
    reference = find_equilibria(model;
        seeds=Tetrastability.Map.deterministic_seeds(model, config.screen_grid_points),
        options=config.equilibrium_options, stability_options=config.stability_options)
    mktempdir() do output
        mkpath(joinpath(output, "contexts", "searches"))
        mkpath(joinpath(output, "contexts", "confirmations"))
        mkpath(joinpath(output, "contexts", "continuations"))
        Tetrastability._initialize_artifacts(output)
        state = Tetrastability._state()
        state.candidate_results[cell.cell_id] =
            (; search_id="discovery_fixture", model, result=reference)
        confirmation = Tetrastability.confirm_candidate!(state, config, cell, output)
        @test confirmation.candidate_id == "candidate_0001"
        @test state.confirmation_counter[] == 1
        @test state.confirmation_search_counter[] ==
            length(Tetrastability.CONDITIONS) * length(config.confirmation_grid_points)
        searches = collect(Tetrastability.CSV.File(joinpath(output, "searches.csv")))
        @test length(searches) == state.confirmation_search_counter[]
        @test Set(row.condition for row in searches) ==
            Set(("control", "failure_of_inhibition"))
        @test Set(row.grid_points for row in searches) == Set((21, 41))
        @test all(row -> row.status == "completed", searches)
        @test length(collect(Tetrastability.CSV.File(
            joinpath(output, "candidates.csv")))) == 1
    end
end

@testset "Candidate continuation artifact path" begin
    config = Tetrastability.load_config(joinpath(@__DIR__, "..", "experiments",
        "tetrastability.toml"))
    cell = only(filter(candidate -> candidate.plane == "figure3" &&
        candidate.e_to_i == 19.0 && candidate.theta_off == 8.0,
        Tetrastability.plane_cells(config)))
    model = Tetrastability.models_for_cell(config, cell).failure_of_inhibition
    search = find_equilibria(model;
        seeds=Tetrastability.Map.deterministic_seeds(model,
            last(config.confirmation_grid_points)),
        options=Tetrastability.refined_options(config.equilibrium_options,
            config.confirmation.tolerance_factor),
        stability_options=Tetrastability.refined_options(config.stability_options,
            config.confirmation.tolerance_factor))
    item = (; search_id="grid41_fixture", model, result=search)
    track = (; track_id=1, matches=Dict(41 => 1))
    confirmation = (; confirmed=true, cell,
        refined=Dict((:failure_of_inhibition, 41) => item),
        assessment=(; quality_tracks=[track]), candidate_id="candidate_fixture")
    mktempdir() do output
        mkpath(joinpath(output, "contexts", "searches"))
        mkpath(joinpath(output, "contexts", "confirmations"))
        mkpath(joinpath(output, "contexts", "continuations"))
        Tetrastability._initialize_artifacts(output)
        state = Tetrastability._state()
        Tetrastability.continue_candidate!(state, config, confirmation, output; smoke=true)
        @test isempty(state.failures)
        branches = collect(Tetrastability.CSV.File(joinpath(output, "branches.csv")))
        @test length(branches) == 2length(Tetrastability.SEARCH_AXES)
        @test Set(row.axis for row in branches) == Set(string.(Tetrastability.SEARCH_AXES))
        @test Set(row.direction for row in branches) == Set((-1, 1))
        @test all(row -> row.points > 0 && row.attempts > 0, branches)
        @test length(readdir(joinpath(output, "contexts", "continuations"))) ==
            length(Tetrastability.SEARCH_AXES)
        @test !isempty(collect(Tetrastability.CSV.File(
            joinpath(output, "continuation_points.csv"))))
        @test !isempty(collect(Tetrastability.CSV.File(
            joinpath(output, "continuation_attempts.csv"))))
    end
end
