include(joinpath(@__DIR__, "..", "scripts", "run_coexistence_map.jl"))

@testset "Manuscript coexistence configuration and matched models" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "coexistence.toml")
    config = CoexistenceExperiment.load_config(config_path)
    @test config.e_to_i_values == collect(12.0:0.5:28.0)
    @test config.failure_threshold_values == collect(6.0:0.25:12.0)
    @test [anchor.name for anchor in config.anchors] == ["figure3", "figure4_exploration"]
    @test config.smoke_e_to_i_values == [19.0]
    @test config.smoke_failure_threshold_values == [8.0, 10.0, 12.0]
    for anchor in config.anchors
        models = CoexistenceExperiment.models_at(config, anchor.name, 19.0, 8.0)
        @test models.control.excitatory === models.failure_of_inhibition.excitatory
        @test models.control.coupling === models.failure_of_inhibition.coupling
        @test models.control.drive === models.failure_of_inhibition.drive
        @test models.control.excitatory.timescale == 7.8
        @test models.control.inhibitory.timescale == 34.32
        @test models.control.inhibitory.response.threshold == 4.0
        @test models.failure_of_inhibition.inhibitory.response.failure_threshold == 8.0
        @test drive_value(models.control.drive, 0.0) == (0.0, 0.0)
        @test models.control.coupling.e_to_e == anchor.e_to_e
        @test models.control.coupling.i_to_e == anchor.i_to_e
        @test models.control.coupling.i_to_i == anchor.i_to_i
        for model in models
            seeds = CoexistenceExperiment.deterministic_seeds(model, config.seed_grid_points)
            @test length(seeds) > config.seed_grid_points^2
            @test length(unique(seeds)) == length(seeds)
            @test all(seed -> seed in seeds, default_equilibrium_seeds(model))
        end
    end
    @test_throws ArgumentError CoexistenceExperiment.models_at(config, "missing", 19, 8)
    @test_throws ArgumentError CoexistenceExperiment.main(String[])
    @test_throws ArgumentError CoexistenceExperiment.main(["--smoke", "--smoke"])
    @test_throws ArgumentError CoexistenceExperiment.main(["--config"])
    @test_throws ArgumentError CoexistenceExperiment.main(["--unknown"])
    mktempdir() do directory
        path = joinpath(directory, "invalid.toml")
        for invalidate in (
            raw -> raw["schema_version"] = true,
            raw -> raw["model"]["excitatory"]["timescale"] = 0.0,
            raw -> raw["model"]["inhibitory"]["slope"] = Inf,
            raw -> raw["axes"]["e_to_i"]["step"] = 0.0,
            raw -> raw["axes"]["e_to_i"]["step"] = 0.7,
            raw -> raw["axes"]["failure_threshold"]["minimum"] = 4.0,
            raw -> raw["anchors"][1]["name"] = "../outside",
            raw -> raw["anchors"][2]["name"] = "figure3",
            raw -> raw["anchors"][1]["i_to_e"] = -1.0,
            raw -> raw["smoke"]["e_to_i"] = [19.1],
            raw -> raw["smoke"]["e_to_i"] = [19.0, 19.0 + 1e-13],
            raw -> raw["smoke"]["failure_threshold"] = [8.0, 8.0],
            raw -> raw["search"]["seed_grid_points"] = 5,
            raw -> raw["search"]["seed_grid_points"] = true,
            raw -> raw["equilibrium"]["maxiters"] = true,
            raw -> raw["stability"]["spectrall_atol"] = 1e-10,
        )
            raw = deepcopy(config.raw)
            invalidate(raw)
            CoexistenceExperiment.write_toml(path, raw)
            output = joinpath(directory, "must_not_exist")
            @test_throws ArgumentError CoexistenceExperiment.run_experiment(path, output; smoke=true)
            @test !ispath(output)
        end
    end
end

@testset "Figure 3 discovery and response-branch evidence" begin
    config = CoexistenceExperiment.load_config(joinpath(@__DIR__, "..", "experiments", "coexistence.toml"))
    models = CoexistenceExperiment.models_at(config, :figure3, 19, 8)
    result = find_equilibria(models.failure_of_inhibition;
        seeds=CoexistenceExperiment.deterministic_seeds(models.failure_of_inhibition, config.seed_grid_points),
        options=config.equilibrium_options, stability_options=config.stability_options)
    attracting = filter(eq -> eq.stability.classification == Attracting, result.equilibria)
    # Independently supplied approximate anchor coordinates; these are discovery
    # regressions and make no claim that other equilibria or attractors are absent.
    for state in ([0.00058, 0.0], [0.5, 0.44772], [0.5, 0.00056])
        @test any(eq -> maximum(abs.(eq.state .- state)) < 1e-4, attracting)
    end
    @test any(eq -> eq.stability.classification == Saddle, result.equilibria)
    @test any(eq -> eq.stability.classification == Repelling, result.equilibria)
    @test result.completeness == CompletenessNotCertified
    for equilibrium in result.equilibria
        observed = CoexistenceExperiment.equilibrium_observations(models.failure_of_inhibition, equilibrium)
        @test observed.u_I ≈ 19equilibrium.state[1] - 4equilibrium.state[2]
        @test observed.F_I_prime ≈ response_derivative(models.failure_of_inhibition.inhibitory.response, observed.u_I)
        @test observed.descending_response_branch == (observed.u_I > 6.0)
    end
end

@testset "Coexistence artifacts, control caching, and retained failures" begin
    mktempdir() do directory
        config = CoexistenceExperiment.load_config(joinpath(@__DIR__, "..", "experiments", "coexistence.toml"))
        raw = deepcopy(config.raw)
        raw["anchors"] = raw["anchors"][1:1]
        raw["smoke"]["failure_threshold"] = [8.0, 12.0]
        path = joinpath(directory, "config.toml")
        CoexistenceExperiment.write_toml(path, raw)
        first_output, second_output = joinpath(directory, "first"), joinpath(directory, "second")
        first_run = CoexistenceExperiment.run_experiment(path, first_output; smoke=true)
        second_run = CoexistenceExperiment.run_experiment(path, second_output; smoke=true)
        @test first_run.success && second_run.success
        @test length(first_run.cells) == 4
        @test length(readdir(joinpath(first_output, "contexts"))) == 3
        control_cells = filter(row -> row.condition == "control", first_run.cells)
        @test length(unique(row.context_id for row in control_cells)) == 1
        @test all(row -> row.completeness == "CompletenessNotCertified" &&
            row.periodic_orbit_status == "not_tested" && row.biological_interpretation == "not_assigned", first_run.cells)
        @test_throws ArgumentError CoexistenceExperiment.run_experiment(path, first_output; smoke=true)
        for file in ("map.csv", "equilibria.csv", "attempts.csv")
            @test read(joinpath(first_output, file)) == read(joinpath(second_output, file))
        end
        metadata = CoexistenceExperiment.TOML.parsefile(joinpath(first_output, "metadata.toml"))
        @test metadata["unique_search_count"] == 3
        @test metadata["cell_count"] == 4
        @test metadata["smoke"]
        @test occursin("--smoke", metadata["replay_from_artifact_directory"])
        @test haskey(metadata["source_sha256"], "scripts/run_coexistence_map.jl")
        @test haskey(metadata["source_sha256"], "scripts/run_minimal_experiment.jl")
        checksums = CoexistenceExperiment.TOML.parsefile(joinpath(first_output, "checksums.toml"))
        @test all(relative -> CoexistenceExperiment.file_hash(joinpath(first_output, relative)) ==
            checksums["files"][relative], keys(checksums["files"]))
        contexts = [CoexistenceExperiment.TOML.parsefile(joinpath(first_output, "contexts", file))
            for file in readdir(joinpath(first_output, "contexts"))]
        @test all(context -> length(context["attempts"]) == length(context["requested_seeds"]), contexts)
        @test all(context -> all(attempt -> haskey(attempt, "balance_jacobian") &&
            haskey(attempt, "solver_residual") && haskey(attempt, "validation"), context["attempts"]), contexts)
        # Solver nonconvergence remains inspectable and is distinct from a runner exception.
        raw["equilibrium"]["maxiters"] = 1
        CoexistenceExperiment.write_toml(path, raw)
        failed_output = joinpath(directory, "limited")
        limited_run = CoexistenceExperiment.run_experiment(path, failed_output; smoke=true)
        @test limited_run.success
        @test any(cell -> cell.rejected_attempts > 0, limited_run.cells)
        attempts = CoexistenceExperiment.CSV.File(joinpath(failed_output, "attempts.csv"))
        @test any(attempt -> !attempt.solver_success, attempts)
        @test any(attempt -> attempt.validation == "RejectedCandidate", attempts)
    end
end
