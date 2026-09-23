include(joinpath(@__DIR__, "..", "scripts", "run_rescue_replay.jl"))

@testset "Rescue replay configuration and coordinate selection" begin
    path = joinpath(@__DIR__, "..", "experiments", "rescue_replay.toml")
    config = RescueReplay.load_config(path)
    @test config.selection.condition == :failure_of_inhibition
    @test config.selection.target == :negative_E
    @test config.pulse_options.durations == [10.0, 15.0, 20.0]
    @test config.pulse_options.targets == [:negative_E]
    @test config.pulse_options.followup_times == [5000.0, 10000.0, 20000.0]
    @test config.selection.recheck_followup_time == 20000.0

    equilibrium(state) = (state=state,
        stability=(classification=Attracting,))
    unique_search = (equilibria=[equilibrium([0.2, 0.3])],)
    index, matched = RescueReplay.matching_attracting_equilibrium(
        unique_search, [0.2, 0.3], 1e-8, "source")
    @test index == 1
    @test matched.state == [0.2, 0.3]
    @test_throws ArgumentError RescueReplay.matching_attracting_equilibrium(
        unique_search, [0.8, 0.8], 1e-8, "source")
    duplicate_search = (equilibria=[equilibrium([0.2, 0.3]), equilibrium([0.2, 0.3])],)
    @test_throws ArgumentError RescueReplay.matching_attracting_equilibrium(
        duplicate_search, [0.2, 0.3], 1e-8, "source")

    mktempdir() do temporary
        for mutate in (
            raw -> (raw["pulses"]["durations"] = [10.0, 20.0]),
            raw -> (raw["pulses"]["followup_times"] = [5000.0, 10000.0]),
            raw -> (raw["pulses"]["refinement_levels"] = 1),
            raw -> (raw["rescue"]["recheck_abstol"] = raw["pulses"]["abstol"]),
            raw -> (raw["rescue"]["recheck_reltol"] = raw["pulses"]["reltol"]),
        )
            raw = deepcopy(config.raw)
            mutate(raw)
            changed = joinpath(temporary, "changed.toml")
            RescueReplay.PulseExperiment.write_toml(changed, raw)
            @test_throws ArgumentError RescueReplay.load_config(changed)
        end
        for (key, primary_key) in (("recheck_abstol", "abstol"),
                ("recheck_reltol", "reltol"))
            raw = deepcopy(config.raw)
            raw["rescue"][key] = raw["pulses"][primary_key]
            changed = joinpath(temporary, "equal_$key.toml")
            RescueReplay.PulseExperiment.write_toml(changed, raw)
            @test_throws ArgumentError RescueReplay._load_config(
                changed; require_canonical=false)
        end
    end
end

@testset "Rescue replay selections and censored brackets" begin
    trials = [
        (duration=10.0, amplitude=8.0),
        (duration=15.0, amplitude=3.0),
        (duration=15.0, amplitude=3.5),
    ]
    boundaries = [
        (lower_outcome=7, upper_outcome=1, lower_trial=2, upper_trial=3),
        (lower_outcome=7, upper_outcome=1, lower_trial=2, upper_trial=3),
    ]
    result = (trials=trials, boundaries=boundaries, options=(amplitudes=[0.0, 8.0],))
    selected = RescueReplay.selected_rechecks(result, 7, 1,
        (cap_recheck_duration=10.0,))
    @test selected == [(10.0, 8.0) => 1, (15.0, 3.0) => 2, (15.0, 3.5) => 3]

    resolved = [
        Dict("duration" => 10.0, "destination_observed" => false,
            "unresolved_trials" => 0, "integration_failed_trials" => 0),
        Dict("duration" => 15.0, "destination_observed" => true,
            "unresolved_trials" => 0, "integration_failed_trials" => 0),
    ]
    bracket = only(RescueReplay.protocol_change_brackets(resolved))
    @test bracket["claim_evidence_eligible"]
    @test bracket["evidence_status"] == "fully_resolved"
    censored = deepcopy(resolved)
    censored[1]["unresolved_trials"] = 1
    bracket = only(RescueReplay.protocol_change_brackets(censored))
    @test !bracket["claim_evidence_eligible"]
    @test bracket["evidence_status"] == "censored"
    @test bracket["lower_unresolved_trials"] == 1
    disagreement = [(duration=15.0, matches_primary=false,)]
    bracket = only(RescueReplay.protocol_change_brackets(resolved, disagreement))
    @test !bracket["claim_evidence_eligible"]
    @test !bracket["supporting_rechecks_agree"]
    @test bracket["evidence_status"] == "recheck_disagreement"
end

@testset "Focused rescue replay writes one-condition coordinate evidence" begin
    config = RescueReplay.load_config(
        joinpath(@__DIR__, "..", "experiments", "rescue_replay.toml"))
    mktempdir() do temporary
        raw = deepcopy(config.raw)
        raw["search"]["grid_points"] = 5
        raw["diagnostics"]["window_duration"] = 3.0
        raw["diagnostics"]["min_samples"] = 3
        raw["pulses"]["amplitudes"] = Dict("start" => 0.0, "stop" => 0.0, "step" => 0.25)
        raw["pulses"]["durations"] = [1.0]
        raw["pulses"]["followup_times"] = [50.0]
        raw["pulses"]["refinement_levels"] = 0
        raw["rescue"]["cap_recheck_duration"] = 1.0
        raw["rescue"]["recheck_followup_time"] = 50.0
        config_path = joinpath(temporary, "rescue.toml")
        RescueReplay.PulseExperiment.write_toml(config_path, raw)
        output = joinpath(temporary, "output")
        fixture = RescueReplay._load_config(config_path; require_canonical=false)
        result = RescueReplay._run_experiment(fixture, config_path, output)
        @test result.success
        @test length(result.result.trials) == 1
        trial = only(result.result.trials)
        @test trial.target == :negative_E
        @test trial.duration == 1.0
        @test trial.initial_state ≈ config.selection.source_state atol=1e-6
        @test length(result.rechecks) == 1
        @test only(result.rechecks).duration == 1.0
        @test only(result.rechecks).amplitude == 0.0

        trials = collect(RescueReplay.PulseExperiment.CSV.File(joinpath(output, "trials.csv")))
        @test length(trials) == 1
        @test only(trials).condition == "failure_of_inhibition"
        @test only(trials).target == "negative_E"
        @test !isfile(joinpath(output, "contexts", "control.toml"))
        @test isfile(joinpath(output, "claim_summary.toml"))
        @test isfile(joinpath(output, "rechecks.csv"))
        summary_text = read(joinpath(output, "claim_summary.toml"), String)
        @test occursin("matched_state", summary_text)
        @test !occursin("minimum_duration", summary_text)
        metadata = RescueReplay.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test metadata["condition"] == "failure_of_inhibition"
        @test metadata["target"] == "negative_E"
        @test occursin("run_rescue_replay.jl", metadata["replay_from_artifact_directory"])
        @test haskey(metadata["source_sha256"], "scripts/run_rescue_replay.jl")
        @test haskey(metadata["source_sha256"], "experiments/rescue_replay.toml")
        archived_contract = joinpath(output, "source", "experiments", "rescue_replay.toml")
        @test isfile(archived_contract)
        @test RescueReplay.TOML.parsefile(archived_contract) ==
            RescueReplay.TOML.parsefile(config_path)
        @test occursin("coordinate-selected", metadata["artifact_schema"]["initial_states"])
        @test haskey(metadata["artifact_schema"], "claim_summary")
        @test haskey(metadata["artifact_schema"], "rechecks")
        hashes = RescueReplay.TOML.parsefile(joinpath(output, "checksums.toml"))["files"]
        @test all(relative -> RescueReplay.PulseExperiment.file_hash(
            joinpath(output, relative)) == hashes[relative], keys(hashes))
        replay_command = `$(Base.julia_cmd()) --project=source source/scripts/run_rescue_replay.jl --config config.toml --output replay`
        @test success(Cmd(replay_command; dir=output))
        @test isfile(joinpath(output, "replay", "claim_summary.toml"))
        @test_throws ArgumentError RescueReplay._run_experiment(
            fixture, config_path, output)
    end
end

@testset "Rescue replay CLI validation" begin
    @test_throws ArgumentError RescueReplay.main(String[])
    @test_throws ArgumentError RescueReplay.main(["--output"])
    @test_throws ArgumentError RescueReplay.main(["--unknown", "value"])
    @test_throws ArgumentError RescueReplay.main(["--output", "one", "--output", "two"])
end
