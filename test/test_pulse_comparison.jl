include(joinpath(@__DIR__, "..", "scripts", "run_pulse_comparison.jl"))

@testset "Pulse comparison parameter isolation" begin
    comparison = PulseComparison
    path = joinpath(@__DIR__, "..", "experiments", "pulses.toml")
    config = comparison.PulseExperiment.load_config(path)
    original = deepcopy(config.raw)
    baseline = comparison.case_parameters(config.raw)
    changed = Dict("theta_off_10" => (:theta_off, 10.0),
        "theta_off_12" => (:theta_off, 12.0),
        "e_to_i_60_percent" => (:e_to_i, 11.4),
        "e_to_e_60_percent" => (:e_to_e, 10.2),
        "i_to_e_150_percent" => (:i_to_e, 13.5))
    for name in comparison.CASE_NAMES
        derived = comparison.derived_config(config.raw, name)
        parameters = comparison.case_parameters(derived)
        @test config.raw == original
        for section in ("pulses", "diagnostics", "equilibrium", "stability", "search")
            @test derived[section] == original[section]
        end
        if name == "baseline"
            @test parameters == baseline
        elseif haskey(changed, name)
            axis, value = changed[name]
            @test getproperty(parameters, axis) ≈ value
            @test all(key -> key == axis ||
                getproperty(parameters, key) == getproperty(baseline, key), propertynames(baseline))
        else
            @test parameters.e_to_e == 19.0
            @test parameters.i_to_e == 13.0
            @test parameters.i_to_i == 6.0
            @test parameters.e_to_i == 19.0
            @test parameters.theta_off == 8.0
            @test occursin("separate Figure 4 exploration anchor", derived["description"])
        end
    end
    @test_throws ArgumentError comparison.derived_config(config.raw, "unsupported")
    @test_throws ArgumentError comparison.validate_case_names(String[])
    @test_throws ArgumentError comparison.validate_case_names(["baseline", "baseline"])
    @test_throws ArgumentError comparison.validate_case_names(["../baseline"])
    @test_throws ArgumentError comparison.validate_case_names("baseline")
    @test_throws ArgumentError comparison.main(String[])
    @test_throws ArgumentError comparison.main(["--case"])
    @test_throws ArgumentError comparison.main(["--smoke", "--smoke"])
    @test_throws ArgumentError comparison.main(["--unknown"])
end

@testset "Pulse comparison artifacts and matched independent cases" begin
    comparison = PulseComparison
    config = comparison.PulseExperiment.load_config(joinpath(@__DIR__, "..", "experiments", "pulses.toml"))
    mktempdir() do directory
        raw = deepcopy(config.raw)
        raw["search"]["grid_points"] = 5
        raw["pulses"]["amplitudes"] = Dict("start" => 0.0, "stop" => 0.25, "step" => 0.25)
        raw["pulses"]["durations"] = [1.0]
        raw["pulses"]["targets"] = ["E"]
        raw["pulses"]["followup_times"] = [500.0]
        raw["pulses"]["refinement_levels"] = 0
        path = joinpath(directory, "test.toml")
        comparison.write_toml(path, raw)
        output = joinpath(directory, "comparison")
        result = comparison.run_experiment(path, output;
            cases=["baseline", "theta_off_12"], smoke=true)
        @test result.success
        @test length(result.cases) == 2
        @test all(row -> row.trial_count > 0 && row.elapsed_seconds >= 0, result.cases)
        @test_throws ArgumentError comparison.run_experiment(path, output)
        invalid_output = joinpath(directory, "invalid")
        @test_throws ArgumentError comparison.run_experiment(path, invalid_output; cases=["bad"])
        @test !ispath(invalid_output)
        empty_output = mkpath(joinpath(directory, "empty"))
        @test_throws ArgumentError comparison.run_experiment(path, empty_output; cases=["baseline"])
        for row in result.cases
            derived_path = joinpath(output, row.config_path)
            derived = comparison.PulseExperiment.load_config(derived_path)
            @test derived.models.control.coupling === derived.models.failure_of_inhibition.coupling
            @test derived.models.control.excitatory === derived.models.failure_of_inhibition.excitatory
            @test isfile(joinpath(output, row.results_directory, "boundaries.csv"))
            @test isfile(joinpath(output, row.results_directory, "equilibria.csv"))
            @test read(derived_path) == read(joinpath(output, row.results_directory, "config.toml"))
        end
        metadata = comparison.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test metadata["smoke"]
        @test metadata["case_names"] == ["baseline", "theta_off_12"]
        @test metadata["biological_interpretation"] == "not_assigned"
        @test occursin("IDs are local", metadata["branch_correspondence"])
        @test occursin("--case baseline --case theta_off_12 --smoke", metadata["replay_from_artifact_directory"])
        @test haskey(metadata["source_sha256"], "scripts/run_pulse_comparison.jl")
        @test haskey(metadata["source_sha256"], "scripts/run_pulse_experiment.jl")
        hashes = comparison.TOML.parsefile(joinpath(output, "checksums.toml"))["files"]
        @test all(relative -> comparison.file_hash(joinpath(output, relative)) ==
            hashes[relative], keys(hashes))
    end
end
