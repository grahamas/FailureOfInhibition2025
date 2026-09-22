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
    @test !options.retain_trajectories
    for kwargs in ((; amplitudes=[]), (; amplitudes=[0, 0]), (; amplitudes=[1, 0]),
        (; amplitudes=[-1, 0]), (; durations=[0]), (; followup_times=[100, 1000]),
        (; followup_times=[200, Inf]), (; targets=[:E, :E]), (; targets=[:unknown]),
        (; refinement_levels=-1), (; refinement_levels=true), (; maxiters=0),
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
    evaluate = amplitude -> (amplitude=amplitude,
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
        (outcome_equilibrium=nothing,), (outcome_equilibrium=1,)) == :unresolved
end

include(joinpath(@__DIR__, "..", "scripts", "run_pulse_experiment.jl"))

@testset "Pulse runner configuration, snapshots, and overwrite protection" begin
    config_path = joinpath(@__DIR__, "..", "experiments", "pulses.toml")
    config = PulseExperiment.load_config(config_path)
    @test config.models.failure_of_inhibition.coupling == PointCoupling(17.0, 9.0, 19.0, 4.0)
    @test config.models.failure_of_inhibition.excitatory.timescale == 7.8
    @test PulseExperiment.smoke_options(config.pulse_options).amplitudes == [0.0, 8.0]
    mktempdir() do temporary
        raw = deepcopy(config.raw)
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
        @test isfile(joinpath(output, "source", "scripts", "run_pulse_experiment.jl"))
        @test isfile(joinpath(output, "source", "scripts", "run_minimal_experiment.jl"))
        metadata = PulseExperiment.TOML.parsefile(joinpath(output, "metadata.toml"))
        @test occursin("run_pulse_experiment.jl", metadata["replay_from_artifact_directory"])
        @test !occursin("synthetic", metadata["purpose"])
        @test metadata["trial_count"] == 8
        @test !isempty(PulseExperiment.TOML.parsefile(joinpath(output, "checksums.toml"))["files"])
        @test_throws ArgumentError PulseExperiment.run_experiment(small_config, output)
    end
end
