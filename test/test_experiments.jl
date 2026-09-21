include(joinpath(@__DIR__, "..", "scripts", "run_minimal_experiment.jl"))

@testset "Synthetic experiment configuration" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "minimal.toml")
    config = MinimalExperiment.load_config(config_path)
    @test length(config.variants) == 3
    @test config.initial_state == [0.1, 0.15]
    @test [v.name for v in config.variants] == ["default", "refined", "extended"]
    for protocol in (:baseline, :pulsed)
        models = config.models[protocol]
        @test models.control.excitatory === models.failure_of_inhibition.excitatory
        @test models.control.coupling === models.failure_of_inhibition.coupling
        @test models.control.drive === models.failure_of_inhibition.drive
        @test models.control.inhibitory.timescale == models.failure_of_inhibition.inhibitory.timescale
        @test models.failure_of_inhibition.inhibitory.response.failure_threshold == 1.1
    end
    @test drive_value(config.models[:baseline].control.drive, 4.0) == (0.0, 0.0)
    @test drive_value(config.models[:pulsed].control.drive, 4.0) == (0.25, 0.10)
    fractional_variant = merge(first(config.variants), (time_span=(0.0, 20.2), saveat=0.3))
    fractional_options = DiagnosticOptions(window_duration=0.1)
    fractional_middle = fractional_variant.time_span[2] - fractional_options.window_duration
    fractional_start = fractional_middle - fractional_options.window_duration
    @test fractional_start in MinimalExperiment.save_times(fractional_variant, fractional_options)
    @test fractional_middle in MinimalExperiment.save_times(fractional_variant, fractional_options)
    @test_throws ArgumentError MinimalExperiment.main(String[])
    @test_throws ArgumentError MinimalExperiment.main(["--unknown", "value"])

    mktempdir() do directory
        path = joinpath(directory, "invalid.toml")
        invalid_configs = [
            raw -> raw["schema_version"] = 2,
            raw -> raw["initial_state"] = [-0.1, 0.2],
            raw -> raw["model"]["excitatory"]["timescale"] = Inf,
            raw -> raw["model"]["coupling"]["i_to_e"] = -0.5,
            raw -> raw["drive"]["pulses"][1]["offset"] = 1.0,
            raw -> raw["diagnostics"]["min_samples"] = 1,
            raw -> raw["diagnostics"]["window_duration"] = 0.0,
            raw -> raw["variants"][1]["saveat"] = 0.0,
            raw -> raw["variants"][1]["maxiters"] = true,
            raw -> raw["variants"][1]["time_span"] = [0.0, 5.0],
            raw -> raw["variants"][2]["name"] = "default",
            raw -> raw["variants"][1]["name"] = "../elsewhere",
            raw -> raw["variants"][1]["abstoll"] = 1e-6,
        ]
        for invalidate in invalid_configs
            raw = deepcopy(config.raw)
            invalidate(raw)
            MinimalExperiment.write_toml(path, raw)
            output = joinpath(directory, "must_not_exist")
            @test_throws ArgumentError MinimalExperiment.run_experiment(path, output)
            @test !ispath(output)
        end
    end
end

@testset "Reproducible experiment artifacts and retained failures" begin
    mktempdir() do directory
        config_path = joinpath(@__DIR__, "..", "experiments", "minimal.toml")
        config = MinimalExperiment.load_config(config_path)
        raw = deepcopy(config.raw)
        raw["variants"] = raw["variants"][1:1]
        # This spacing does not divide the observation-window boundaries.
        raw["variants"][1]["saveat"] = 0.3
        test_config = joinpath(directory, "test.toml")
        MinimalExperiment.write_toml(test_config, raw)
        first_output = joinpath(directory, "first")
        second_output = joinpath(directory, "second")
        first_run = MinimalExperiment.run_experiment(test_config, first_output)
        second_run = MinimalExperiment.run_experiment(test_config, second_output)
        @test first_run.success && second_run.success
        @test length(first_run.cases) == 4
        @test all(case -> case.status == "completed", first_run.cases)
        @test read(joinpath(first_output, "config.toml")) == read(test_config)
        @test_throws ArgumentError MinimalExperiment.run_experiment(test_config, first_output)

        for table in ("cases.csv", "attempts.csv", "equilibria.csv", "comparisons.csv")
            @test read(joinpath(first_output, table)) == read(joinpath(second_output, table))
        end
        @test length(readdir(joinpath(first_output, "contexts"))) == 12
        for subdirectory in ("contexts", "diagnostics", "trajectories")
            for filename in readdir(joinpath(first_output, subdirectory))
                @test read(joinpath(first_output, subdirectory, filename)) ==
                      read(joinpath(second_output, subdirectory, filename))
            end
        end
        for case in first_run.cases
            trajectory_file = joinpath(first_output, case.trajectory_path)
            @test first(readlines(trajectory_file)) == "time,E,I"
            trajectory = MinimalExperiment.CSV.File(trajectory_file)
            @test all(boundary -> boundary in trajectory.time, (10.0, 15.0, 20.0))
            @test first(trajectory).E == 0.1 && first(trajectory).I == 0.15
            @test all(row -> 0 <= row.E <= 1 && 0 <= row.I <= 1, trajectory)
            record = MinimalExperiment.TOML.parsefile(joinpath(first_output, case.diagnostic_path))
            @test record["periodic_orbit_status"] == "unvalidated"
            @test record["biological_interpretation"] == "not_assigned"
            @test record["diagnostics"]["window_bounds"] == [[10.0, 15.0], [15.0, 20.0]]
            @test record["terminal_context"] == case.terminal_context
        end
        for condition in ("control", "failure_of_inhibition")
            for (index, time) in enumerate([0.0, 2.0, 4.0, 5.0, 7.0])
                search = MinimalExperiment.TOML.parsefile(joinpath(first_output,
                    "contexts", "$(condition)_pulsed_$(index).toml"))
                @test search["source_time"] == time
                @test search["frozen_drive"] == collect(drive_value(config.models[:pulsed].control.drive, time))
                @test search["completeness"] == "CompletenessNotCertified"
                @test length(search["attempts"]) == 25
                @test all(attempt -> haskey(attempt, "balance_jacobian") &&
                    haskey(attempt, "solver_residual") && haskey(attempt, "reasons"), search["attempts"])
                @test all(root -> haskey(root["stability"], "jacobian") &&
                    haskey(root["stability"], "eigenvalues") && haskey(root, "member_attempts"), search["equilibria"])
            end
        end
        metadata = MinimalExperiment.TOML.parsefile(joinpath(first_output, "metadata.toml"))
        @test metadata["execution_success"]
        @test haskey(metadata, "git_revision") && haskey(metadata, "git_status_porcelain")
        @test haskey(metadata["source_sha256"], "src/diagnostics.jl")
        checksums = MinimalExperiment.TOML.parsefile(joinpath(first_output, "checksums.toml"))
        @test all(relative -> MinimalExperiment.file_hash(joinpath(first_output, relative)) ==
            checksums["files"][relative], keys(checksums["files"]))
        @test read(joinpath(first_output, "source", "scripts", "run_minimal_experiment.jl")) ==
              read(joinpath(@__DIR__, "..", "scripts", "run_minimal_experiment.jl"))

        raw["equilibrium"]["maxiters"] = 1
        raw["variants"][1]["maxiters"] = 1
        MinimalExperiment.write_toml(test_config, raw)
        failed_output = joinpath(directory, "failed")
        failed_run = MinimalExperiment.run_experiment(test_config, failed_output)
        @test !failed_run.success
        @test length(failed_run.cases) == 4
        @test all(case -> case.status == "execution_failed", failed_run.cases)
        attempts = MinimalExperiment.CSV.File(joinpath(failed_output, "attempts.csv"))
        @test length(attempts) == 300
        @test any(attempt -> !attempt.solver_success, attempts)
        @test any(attempt -> attempt.validation == "RejectedCandidate", attempts)
        for case in failed_run.cases
            @test case.diagnostic_classification == "TrajectoryUnresolved"
            @test !case.integration_success
            @test isfile(joinpath(failed_output, case.trajectory_path))
            record = MinimalExperiment.TOML.parsefile(joinpath(failed_output, case.diagnostic_path))
            @test record["status"] == "execution_failed"
            @test "integration_failed" in record["reasons"]
        end
    end
end
