function pulse_test_model(; timescale=1.0, recurrent=0.0, threshold=0.0, drive=NoDrive())
    return PointModelParameters(
        excitatory=PopulationParameters(timescale=timescale,
            response=LogisticResponse(slope=1.0, threshold=threshold)),
        inhibitory=PopulationParameters(timescale=timescale,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(recurrent, 0.0, 0.0, 0.0), drive=drive)
end

@testset "Pulse experiment policy validation" begin
    options = PulseExperimentOptions()
    @test options.amplitudes == collect(0.0:0.25:8.0)
    @test options.followup_times == [5000.0, 10000.0, 20000.0]
    @test options.durations == [1, 2, 5, 10, 20, 50, 100, 200]
    @test options.duration_refinement_levels == 0
    @test !options.retain_trajectories
    for kwargs in ((; amplitudes=[]), (; amplitudes=[0, 0]), (; amplitudes=[1, 0]),
        (; amplitudes=[-1, 0]), (; durations=[0]), (; followup_times=[100, 1000]),
        (; followup_times=[200, Inf]), (; targets=[:E, :E]), (; targets=[:unknown]),
        (; refinement_levels=-1), (; refinement_levels=true),
        (; duration_refinement_levels=-1), (; duration_refinement_levels=true), (; maxiters=0),
        (; trajectory_saveat=0), (; abstol=NaN), (; retain_trajectories=1))
        @test_throws ArgumentError PulseExperimentOptions(; kwargs...)
    end
end

@testset "Pulse withdrawal, cost, and conservative destination" begin
    model = pulse_test_model()
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    equilibrium = only(search.equilibria)
    options = PulseExperimentOptions(amplitudes=[0.0, 1.0], durations=[2.0],
        followup_times=[50.0, 100.0],
        diagnostic_options=DiagnosticOptions(window_duration=3.0, min_samples=7),
        refinement_levels=0)
    for (target, costs) in ((:E, (2.0, 0.0)), (:I, (0.0, 2.0)),
                            (:equal, (2.0, 2.0)), (:negative_E, (-2.0, 0.0)))
        trial = run_pulse_trial(model, search, equilibrium.state;
            target=target, amplitude=1.0, duration=2.0, options=options)
        @test trial.status == :compatible
        @test trial.outcome_equilibrium == 1
        @test (trial.integrated_E, trial.integrated_I) == costs
        @test trial.absolute_input_cost == sum(abs, costs)
        @test length(trial.attempts) == 1
        attempt = only(trial.attempts)
        @test attempt.diagnostics.window_bounds == ((46.0, 49.0), (49.0, 52.0))
        @test attempt.diagnostics.sample_counts == (7, 7)
        @test [0.0, 2.0, 46.0, 49.0, 52.0] ⊆ attempt.times
        @test length(attempt.times) == 15
        @test all(tail -> tail.E.mean ≈ 1 / 3, attempt.tail)
        @test all(tail -> tail.I.mean ≈ 1 / 3, attempt.tail)
        @test all(tail -> tail.u_I.minimum == 0.0, attempt.tail)
        @test all(tail -> tail.F_I_prime.mean ≈ 0.25, attempt.tail)
    end
    zero_trial = run_pulse_trial(model, search, equilibrium.state;
        target=:E, amplitude=0, duration=2, options=options)
    @test zero_trial.status == :compatible
    @test zero_trial.absolute_input_cost == 0
    batch = run_pulse_experiments(model; equilibria=search, options=options)
    @test length(batch.initial_states) == 1
    @test length(batch.trials) == 8
    @test all(t -> t.outcome_equilibrium == 1, batch.trials)
    @test isempty(batch.boundaries)

    full_options = PulseExperimentOptions(amplitudes=[0], durations=[2],
        followup_times=[50], diagnostic_options=options.diagnostic_options,
        retain_trajectories=true, trajectory_saveat=1)
    full = run_pulse_trial(model, search, equilibrium.state;
        target=:equal, amplitude=0, duration=2, options=full_options)
    @test length(only(full.attempts).times) > length(only(zero_trial.attempts).times)
    @test collect(0.0:1.0:52.0) ⊆ only(full.attempts).times
    @test full.status == :compatible

    for kwargs in ((; target=:other, amplitude=1, duration=1),
        (; target=:E, amplitude=-1, duration=1), (; target=:E, amplitude=1, duration=0))
        @test_throws ArgumentError run_pulse_trial(model, search, equilibrium.state;
            options=options, kwargs...)
    end
    @test_throws ArgumentError run_pulse_trial(pulse_test_model(timescale=2), search,
        equilibrium.state; target=:E, amplitude=0, duration=1, options=options)
    pulsed = pulse_test_model(drive=PiecewiseConstantDrive(baseline=(0.0, 0.0),
        pulses=(DrivePulse(onset=0, offset=1, increment=(1.0, 0.0)),),
        interpretation=AfferentExcitation))
    @test_throws ArgumentError run_pulse_trial(pulsed, search, equilibrium.state;
        target=:E, amplitude=0, duration=1, options=options)
end

@testset "Pulse unresolved extensions and numerical failures remain evidence" begin
    model = pulse_test_model(timescale=1000)
    search = find_equilibria(model; seeds=[[1 / 3, 1 / 3]])
    options = PulseExperimentOptions(amplitudes=[0.0, 1.0], durations=[1.0], targets=[:E],
        followup_times=[10.0, 20.0, 40.0],
        diagnostic_options=DiagnosticOptions(window_duration=2.0, min_samples=5),
        refinement_levels=2)
    initial = (id="supplied_transient", state=[0.1, 0.1], provenance="test transient, no attractor claim")
    result = run_pulse_experiments(model; equilibria=search, options=options, initial_states=[initial])
    @test length(result.trials) == 2
    @test all(trial -> trial.status == :unresolved, result.trials)
    @test all(trial -> length(trial.attempts) == 3, result.trials)
    @test all(trial -> [a.followup_time for a in trial.attempts] == [10, 20, 40], result.trials)
    @test only(result.boundaries).kind == :unresolved
    @test only(result.boundaries).lower_outcome === nothing
    @test :large_balance_residual in last(first(result.trials).attempts).reasons
    @test_throws ArgumentError run_pulse_experiments(model; equilibria=search,
        options=options, initial_states=[initial, initial])

    failure_options = PulseExperimentOptions(amplitudes=[8], durations=[200], targets=[:E],
        followup_times=[5000], maxiters=1)
    failed = run_pulse_trial(model, search, [0.1, 0.1]; target=:E,
        amplitude=8, duration=200, options=failure_options)
    @test failed.status == :integration_failed
    @test failed.outcome_equilibrium === nothing
    @test !isempty(only(failed.attempts).reasons)

    saddle_model = pulse_test_model(recurrent=12.0, threshold=3 + log(2))
    saddle_search = find_equilibria(saddle_model; seeds=[[0.25, 1 / 3]])
    @test only(saddle_search.equilibria).stability.classification == Saddle
    stationary = run_pulse_trial(saddle_model, saddle_search,
        only(saddle_search.equilibria).state; target=:E, amplitude=0, duration=1,
        options=PulseExperimentOptions(followup_times=[10],
            diagnostic_options=DiagnosticOptions(window_duration=2)))
    @test stationary.status == :unresolved
    @test stationary.outcome_equilibrium === nothing
    @test :matched_equilibrium_not_attracting in only(stationary.attempts).reasons
    @test isempty(run_pulse_experiments(saddle_model; equilibria=saddle_search,
        options=options).initial_states)
end

@testset "Amplitude refinement resolves separate transition intervals" begin
    # Two disjoint transitions cannot be represented by one monotone threshold.
    evaluate = amplitude -> (amplitude=amplitude, status=:compatible,
        outcome_equilibrium=amplitude < 0.3 || amplitude >= 1.7 ? 1 : 2)
    trials = [evaluate(a) for a in (0.0, 1.0, 2.0)]
    indices = [1, 2, 3]
    FailureOfInhibition2025._refine_pulse_amplitudes!(trials, indices, evaluate, 3)
    brackets = [(trials[l].amplitude, trials[r].amplitude)
        for (l, r) in zip(indices, Iterators.drop(indices, 1))
        if trials[l].outcome_equilibrium != trials[r].outcome_equilibrium]
    @test length(brackets) == 2
    @test first(brackets)[1] < 0.3 <= first(brackets)[2]
    @test last(brackets)[1] < 1.7 <= last(brackets)[2]
    @test all(pair -> pair[2] - pair[1] <= 0.125, brackets)
    @test FailureOfInhibition2025._pulse_boundary_kind(
        (status=:unresolved, outcome_equilibrium=nothing,),
        (status=:compatible, outcome_equilibrium=1,)) == :unresolved
    @test FailureOfInhibition2025._pulse_boundary_kind(
        (status=:integration_failed, outcome_equilibrium=nothing,),
        (status=:compatible, outcome_equilibrium=1,)) == :integration_failed
end

@testset "Duration refinement uses topology rather than amplitude location" begin
    signature(destinations; boundaries=(), unresolved=false, failed=false) =
        (destinations=Tuple(destinations), boundary_sequence=Tuple(boundaries),
         unresolved_present=unresolved, integration_failure_present=failed)
    transition(lower, upper; kind=:outcome_change) =
        (kind=kind, lower_outcome=lower, upper_outcome=upper,
         lower_status=lower === nothing ? kind : :compatible,
         upper_status=upper === nothing ? kind : :compatible)

    same = signature((1,); boundaries=(transition(1, 2),))
    durations = [10.0, 20.0]
    signatures = Dict(10.0 => same, 20.0 => same)
    evaluations = Float64[]
    refinements = FailureOfInhibition2025._refine_pulse_durations!(
        durations, signatures, duration -> (push!(evaluations, duration); same), 1)
    @test isempty(refinements)
    @test isempty(evaluations)
    @test durations == [10.0, 20.0]

    lower = signature((1,))
    upper = signature((1, 2); boundaries=(transition(1, 2),))
    durations = [10.0, 20.0]
    signatures = Dict(10.0 => lower, 20.0 => upper)
    refinements = FailureOfInhibition2025._refine_pulse_durations!(
        durations, signatures, _ -> upper, 1)
    @test durations == [10.0, 15.0, 20.0]
    @test only(refinements).inserted_duration == 15.0
    @test only(refinements).level == 1
    @test :destination_set in only(refinements).reasons
    @test :transition_presence in only(refinements).reasons
    @test :boundary_count in only(refinements).reasons
    brackets = FailureOfInhibition2025._pulse_duration_brackets(durations, signatures)
    @test length(brackets) == 1
    @test (only(brackets).lower_duration, only(brackets).upper_duration) == (10.0, 15.0)

    reversed = signature((1, 2); boundaries=(transition(2, 1),))
    @test FailureOfInhibition2025._pulse_duration_difference_reasons(upper, reversed) ==
        [:boundary_order]
    unresolved = signature((1, 2); boundaries=(transition(1, nothing; kind=:unresolved),),
        unresolved=true)
    reasons = FailureOfInhibition2025._pulse_duration_difference_reasons(upper, unresolved)
    @test :boundary_order in reasons
    @test :transition_presence in reasons
    @test :unresolved_presence in reasons
    no_boundary_unresolved = signature((1, 2); unresolved=true)
    reasons = FailureOfInhibition2025._pulse_duration_difference_reasons(
        no_boundary_unresolved, unresolved)
    @test !(:transition_presence in reasons)
    failed = signature((1, 2);
        boundaries=(transition(1, nothing; kind=:integration_failed),), failed=true)
    reasons = FailureOfInhibition2025._pulse_duration_difference_reasons(unresolved, failed)
    @test :unresolved_presence in reasons
    @test :integration_failure_presence in reasons
end

include(joinpath(@__DIR__, "..", "scripts", "run_pulse_experiment.jl"))

@testset "Nonempty duration records serialize with normalized times" begin
    model = PointModelParameters(
        excitatory=PopulationParameters(timescale=2.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        inhibitory=PopulationParameters(timescale=4.0,
            response=LogisticResponse(slope=1.0, threshold=0.0)),
        coupling=PointCoupling(0.0, 0.0, 0.0, 0.0), drive=NoDrive())
    lower_signature = (destinations=(1,), boundary_sequence=(),
        unresolved_present=false, integration_failure_present=false)
    upper_signature = (destinations=(1, 2),
        boundary_sequence=((kind=:outcome_change, lower_outcome=1, upper_outcome=2,
            lower_status=:compatible, upper_status=:compatible),),
        unresolved_present=false, integration_failure_present=false)
    result = (model=model, trials=PulseTrialResult[], boundaries=NamedTuple[],
        duration_refinements=[(initial_id="source", target=:negative_E, level=1,
            lower_duration=10.0, inserted_duration=15.0, upper_duration=20.0,
            reasons=(:destination_set, :transition_presence, :boundary_count))],
        duration_brackets=[(initial_id="source", target=:negative_E,
            lower_duration=10.0, upper_duration=15.0,
            reasons=(:destination_set, :transition_presence, :boundary_count),
            lower_signature, upper_signature)])
    mktempdir() do temporary
        rows = PulseExperiment.save_condition(result, :failure_of_inhibition, temporary)
        refinement = only(rows.duration_refinements)
        @test refinement.lower_duration_over_tau_e == 5.0
        @test refinement.inserted_duration_over_tau_i == 3.75
        @test refinement.upper_duration_over_tau_e == 10.0
        bracket = only(rows.duration_brackets)
        @test bracket.lower_duration_over_tau_i == 2.5
        @test bracket.upper_duration_over_tau_e == 7.5
        @test !bracket.lower_integration_failure

        refinement_path = joinpath(temporary, "duration_refinements.csv")
        bracket_path = joinpath(temporary, "duration_brackets.csv")
        PulseExperiment.write_rows(refinement_path, rows.duration_refinements,
            propertynames(refinement))
        PulseExperiment.write_rows(bracket_path, rows.duration_brackets,
            propertynames(bracket))
        @test length(collect(PulseExperiment.CSV.File(refinement_path))) == 1
        @test length(collect(PulseExperiment.CSV.File(bracket_path))) == 1
    end
end

@testset "Pulse runner configuration, snapshots, and overwrite protection" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "pulses.toml")
    config = PulseExperiment.load_config(config_path)
    @test config.models.failure_of_inhibition.coupling == PointCoupling(17.0, 9.0, 19.0, 4.0)
    @test config.models.failure_of_inhibition.excitatory.timescale == 7.8
    @test config.pulse_options.duration_refinement_levels == 1
    @test PulseExperiment.smoke_options(config.pulse_options).amplitudes == [0.0, 8.0]
    @test PulseExperiment.smoke_options(config.pulse_options).duration_refinement_levels == 0
    mktempdir() do temporary
        raw = deepcopy(config.raw)
        for invalid_version in (true, 1.0)
            invalid = deepcopy(raw)
            invalid["schema_version"] = invalid_version
            invalid_version_path = joinpath(temporary, "invalid-version-$(repr(invalid_version)).toml")
            PulseExperiment.write_toml(invalid_version_path, invalid)
            @test_throws ArgumentError PulseExperiment.load_config(invalid_version_path)
        end
        legacy = deepcopy(raw)
        legacy["schema_version"] = 1
        delete!(legacy["pulses"], "duration_refinement_levels")
        legacy_path = joinpath(temporary, "legacy.toml")
        PulseExperiment.write_toml(legacy_path, legacy)
        @test PulseExperiment.load_config(legacy_path).pulse_options.duration_refinement_levels == 0

        invalid_v2 = deepcopy(raw)
        delete!(invalid_v2["pulses"], "duration_refinement_levels")
        invalid_path = joinpath(temporary, "invalid-v2.toml")
        PulseExperiment.write_toml(invalid_path, invalid_v2)
        @test_throws ArgumentError PulseExperiment.load_config(invalid_path)

        raw["model"]["excitatory"] = Dict("timescale" => 1.0, "slope" => 1.0, "threshold" => 0.0)
        raw["model"]["inhibitory"] = Dict("timescale" => 1.0, "slope" => 1.0,
            "threshold" => 0.0, "failure_threshold" => 8.0)
        raw["model"]["coupling"] = Dict(name => 0.0 for name in ("e_to_e", "i_to_e", "e_to_i", "i_to_i"))
        raw["search"]["grid_points"] = 3
        raw["diagnostics"]["window_duration"] = 3.0
        raw["diagnostics"]["min_samples"] = 3
        raw["pulses"]["amplitudes"] = Dict("start" => 0.0, "stop" => 0.0, "step" => 0.25)
        raw["pulses"]["durations"] = [1.0]
        raw["pulses"]["followup_times"] = [50.0]
        small_config = joinpath(temporary, "small.toml")
        PulseExperiment.write_toml(small_config, raw)
        output = joinpath(temporary, "result")
        result = PulseExperiment.run_experiment(small_config, output)
        @test result.success
        @test length(result.trials) == 8
        @test all(trial -> trial.status == "compatible", result.trials)
        @test isfile(joinpath(output, "trials.csv"))
        @test isfile(joinpath(output, "tails.csv"))
        @test isfile(joinpath(output, "boundaries.csv"))
        @test isfile(joinpath(output, "duration_refinements.csv"))
        @test isfile(joinpath(output, "duration_brackets.csv"))
        @test isfile(joinpath(output, "source", "scripts", "run_pulse_experiment.jl"))
        @test isfile(joinpath(output, "source", "scripts", "run_minimal_experiment.jl"))
        metadata = PulseExperiment.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test occursin("run_pulse_experiment.jl", metadata["replay_from_artifact_directory"])
        @test !occursin("synthetic", metadata["purpose"])
        @test metadata["trial_count"] == 8
        @test metadata["duration_refinement_count"] == 0
        trials = PulseExperiment.CSV.File(joinpath(output, "trials.csv")) |> collect
        @test all(row -> row.duration_over_tau_e == 1.0, trials)
        @test all(row -> row.duration_over_tau_i == 1.0, trials)
        @test !isempty(PulseExperiment.TOML.parsefile(joinpath(output, "checksums.toml"))["files"])
        @test_throws ArgumentError PulseExperiment.run_experiment(small_config, output)
    end
end
