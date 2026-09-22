include(joinpath(@__DIR__, "..", "scripts", "run_intervention_experiment.jl"))

@testset "Intervention and robustness exploration contracts" begin
    experiment = InterventionExperiment
    config = experiment.load_config(joinpath(@__DIR__, "..", "experiments", "interventions.toml"))
    cells = experiment.intervention_cells(config)
    baseline = first(cells)
    @test length(cells) == 45
    @test length(experiment.intervention_cells(config; smoke=true)) == 9
    @test baseline.parameters.tau_i / baseline.parameters.tau_e ≈ 4.4
    @test Set(cell.axis for cell in cells[2:end]) ==
        Set(["theta_off", "e_to_i", "e_to_e", "i_to_e"])
    for cell in cells
        model_pair = experiment.models_for(cell)
        @test model_pair.control.coupling === model_pair.failure_of_inhibition.coupling
        @test model_pair.control.excitatory === model_pair.failure_of_inhibition.excitatory
        @test model_pair.control.inhibitory.timescale == model_pair.failure_of_inhibition.inhibitory.timescale
        @test drive_value(model_pair.control.drive, 0.0) == (0.0, 0.0)
        @test model_pair.failure_of_inhibition.inhibitory.response.slope == cell.parameters.a_i
        if cell.family == "intervention"
            @test all(k -> string(k) == cell.axis ||
                getproperty(cell.parameters, k) == getproperty(baseline.parameters, k),
                propertynames(baseline.parameters))
        end
    end
    @test last(filter(c -> c.axis == "theta_off", cells)).value == 12.0
    @test last(filter(c -> c.axis == "e_to_i", cells)).value ≈ 0.6baseline.parameters.e_to_i
    @test last(filter(c -> c.axis == "e_to_e", cells)).value ≈ 0.6baseline.parameters.e_to_e
    @test last(filter(c -> c.axis == "i_to_e", cells)).value ≈ 1.5baseline.parameters.i_to_e
    robust = experiment.robustness_cells(config)
    @test robust == experiment.robustness_cells(config)
    @test length(robust) == 63
    joint = filter(c -> c.family == "robustness_joint", robust)
    @test length(joint) == 24
    @test length(unique(c.parameters for c in joint)) == 24
    @test experiment.radical_inverse(1, 2) == 0.5
    @test experiment.radical_inverse(2, 2) == 0.25
    @test experiment.radical_inverse(3, 2) == 0.75
    for cell in joint, axis in config.robustness
        value = axis.name == :tau_ratio ? cell.parameters.tau_i / cell.parameters.tau_e :
            getproperty(cell.parameters, axis.name)
        @test any(node -> isapprox(value, node), axis.values)
    end

    model = experiment.models_for(baseline).failure_of_inhibition
    seeds = experiment.deterministic_seeds(model, config.grid_points)
    @test all(seed -> seed in seeds, default_equilibrium_seeds(model))
    @test length(seeds) >= config.grid_points^2
    @test all(seed -> all(x -> 0 <= x <= 0.5, seed), seeds)
    # Both high-E roots occupy the descending response. That fact cannot choose a
    # seizure label or certify preserved functional activity.
    @test experiment.observations(model, [0.5, 0.448]).inhibitory_response_branch == "descending"
    @test experiment.observations(model, [0.5, 0.00056]).inhibitory_response_branch == "descending"
    @test experiment.observations(model, [0.00058, 0.0]).inhibitory_response_branch == "ascending"
    @test experiment.observations(experiment.models_for(baseline).control,
        [0.5, 0.00056]).inhibitory_response_branch == "monotone"

    # Time constants scale the Jacobian but must not move the balance roots.
    search = find_equilibria(model; seeds, options=config.equilibrium_options)
    @test any(e -> isapprox(e.state, [0.5, 0.447721]; atol=1e-5) &&
        e.stability.classification == Attracting, search.equilibria)
    @test any(e -> isapprox(e.state, [0.5, 0.00056]; atol=1e-5) &&
        e.stability.classification == Attracting, search.equilibria)
    @test search.completeness == CompletenessNotCertified
    for ratio in (0.5, 8.0)
        modified = experiment.models_for((parameters=experiment.changed_parameters(
            baseline.parameters, :tau_ratio, ratio),)).failure_of_inhibition
        for root in search.equilibria
            original_balance, modified_balance = zeros(2), zeros(2)
            point_balance!(original_balance, root.state, model, 0.0)
            point_balance!(modified_balance, root.state, modified, 0.0)
            @test original_balance == modified_balance
            original_jacobian, modified_jacobian = zeros(2, 2), zeros(2, 2)
            point_jacobian!(original_jacobian, root.state, model, 0.0)
            point_jacobian!(modified_jacobian, root.state, modified, 0.0)
            @test original_jacobian[1, :] == modified_jacobian[1, :]
            @test original_jacobian[2, :] * model.inhibitory.timescale ≈
                modified_jacobian[2, :] * modified.inhibitory.timescale
        end
    end

    mktempdir() do directory
        path = joinpath(directory, "invalid.toml")
        for invalidate in (
            raw -> raw["baseline"]["theta_off"] = 3.0,
            raw -> raw["baseline"]["a_i"] = false,
            raw -> raw["interventions"][1]["name"] = "unsupported",
            raw -> raw["interventions"][1]["values"] = [8.0, 8.0],
            raw -> raw["robustness"]["axes"][4]["values"] = [0.0],
            raw -> raw["robustness"]["joint_samples"] = 0,
            raw -> raw["search"]["grid_points"] = 1,
            raw -> raw["equilibrium"]["residual_atol"] = -1.0,
        )
            raw = deepcopy(config.raw)
            invalidate(raw)
            experiment.write_toml(path, raw)
            output = joinpath(directory, "must_not_exist")
            @test_throws ArgumentError experiment.run_experiment(path, output)
            @test !ispath(output)
        end
    end
    @test_throws ArgumentError experiment.main(String[])
    @test_throws ArgumentError experiment.main(["--output"])
    @test_throws ArgumentError experiment.main(["--smoke", "--smoke"])
    @test_throws ArgumentError experiment.main(["--unknown"])
end

@testset "Independent intervention artifact replay and retained search failures" begin
    experiment = InterventionExperiment
    config = experiment.load_config(joinpath(@__DIR__, "..", "experiments", "interventions.toml"))
    mktempdir() do directory
        raw = deepcopy(config.raw)
        raw["interventions"] = [Dict("name" => "theta_off", "values" => [8.0, 12.0])]
        raw["search"]["grid_points"] = 5
        path = joinpath(directory, "test.toml")
        experiment.write_toml(path, raw)
        outputs = [joinpath(directory, "first"), joinpath(directory, "second")]
        results = [experiment.run_experiment(path, output; smoke=true) for output in outputs]
        @test all(result -> result.success, results)
        @test all(result -> length(result.cells) == 6, results)
        @test_throws ArgumentError experiment.run_experiment(path, first(outputs))
        for filename in ("cells.csv", "equilibria.csv", "attempts.csv")
            @test read(joinpath(outputs[1], filename)) == read(joinpath(outputs[2], filename))
        end
        for filename in readdir(joinpath(first(outputs), "contexts"))
            context = experiment.TOML.parsefile(joinpath(first(outputs), "contexts", filename))
            @test context["completeness"] == "CompletenessNotCertified"
            @test context["biological_interpretation"] == "not_assigned"
            @test length(context["attempts"]) == 25
        end
        metadata = experiment.TOML.parsefile(joinpath(first(outputs), "metadata.toml"))
        @test metadata["smoke"]
        @test !metadata["robustness_enabled"]
        @test metadata["periodic_orbit_status"] == "not_tested"
        @test metadata["biological_interpretation"] == "not_assigned"
        @test occursin("--smoke", metadata["replay_from_artifact_directory"])
        @test haskey(metadata["source_sha256"], "scripts/run_intervention_experiment.jl")
        @test haskey(metadata["source_sha256"], "scripts/run_minimal_experiment.jl")
        hashes = experiment.TOML.parsefile(joinpath(first(outputs), "checksums.toml"))["files"]
        @test all(relative -> experiment.file_hash(joinpath(first(outputs), relative)) ==
            hashes[relative], keys(hashes))

        raw["equilibrium"]["maxiters"] = 1
        experiment.write_toml(path, raw)
        failed = experiment.run_experiment(path, joinpath(directory, "limited"))
        # Nonlinear failures are evidence retained per seed, not false absence claims.
        @test any(cell -> cell.failed_attempts > 0, failed.cells)
        @test all(cell -> cell.completeness == "CompletenessNotCertified", failed.cells)
        attempts = experiment.CSV.File(joinpath(directory, "limited", "attempts.csv"))
        @test length(attempts) == 150
        @test any(attempt -> !attempt.solver_success, attempts)
    end
end
