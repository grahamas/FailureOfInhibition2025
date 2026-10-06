include(joinpath(@__DIR__, "..", "scripts", "run_input_release_study.jl"))

@testset "Permanent release protocol" begin
    study = InputReleaseStudy
    config = study.load_config(joinpath(@__DIR__, "..", "experiments", "input_release.toml"))
    @test length(config.axis) == 97
    @test extrema(config.axis) == (0.0, 24.0)
    config = merge(config, (; horizons=[1000.0, 2000.0]))
    case = first(config.cases)
    models = study.release_models(case; e_to_e=4.0)
    # Compare the paired helper with the established constructor, independently.
    changed = merge(case, Dict("e_to_e" => 4.0))
    established = study.BasinRescueStudy.model_for(changed, 8.0)
    for state in ([0.01, 0.02], [0.3, 0.4], [0.5, 0.001])
        a, b = zeros(2), zeros(2)
        point_rhs!(a, state, models.on, 0.0)
        point_rhs!(b, state, established, 0.0)
        @test a ≈ b
    end
    searches = (on=study.search_context(models.on, 11), off=study.search_context(models.off, 11))
    source = study.match_source(searches.on, config.roles[case["name"]], config.tolerance)
    @test source !== nothing
    @test study.match_source(searches.off, searches.on.equilibria[source].state, config.tolerance) === nothing
    start = only(study.sink_indices(searches.off))
    trial = study.full_trial(models, searches, source, start, config; retain=true)
    @test trial.success
    @test all(x -> 0 <= x <= 1, trial.induction.final)
    @test trial.recovery.solution.u[1] == trial.induction.final
    @test trial.control.solution.u[1] == trial.induction.final
    @test trial.recovery.destination == start
    @test trial.control.destination == source
    @test trial.recovery.final[1] < 0.001
    @test trial.control.final[1] > 0.49
    mismatched = diagnose_trajectory(trial.recovery.solution, models.off;
        equilibria=searches.on, options=config.diagnostics)
    @test mismatched.classification == TrajectoryUnresolved
    @test :frozen_drive_mismatch in mismatched.reasons

    # Independent single-solve rectangular pulse must reproduce segmented release.
    options = PulseExperimentOptions(followup_times=[trial.recovery.horizon],
        diagnostic_options=config.diagnostics, abstol=config.abstol, reltol=config.reltol)
    pulse = run_pulse_trial(models.off, searches.off, trial.initial;
        target=:E, amplitude=8.0, duration=trial.induction.horizon, options)
    @test pulse.status == :compatible
    @test pulse.outcome_equilibrium == start
    @test last(last(pulse.attempts).states) ≈ trial.recovery.final atol=1e-8

    for status in ("unresolved", "integration_failed")
        bad = merge(trial.recovery, (; status, destination=nothing))
        @test !study.qualifies_trial(source, start, trial.induction, bad, trial.control)
        bad_on = merge(trial.induction, (; status, destination=nothing))
        @test !study.qualifies_trial(source, start, bad_on, trial.recovery, trial.control)
    end
    wrong_source = merge(trial.induction, (; destination=source+1))
    @test !study.qualifies_trial(source, start, wrong_source, trial.recovery, trial.control)
    limited = merge(config, (; maxiters=1))
    failure = study.observe_phase(models.on, searches.on, trial.initial, limited)
    @test failure.status == "integration_failed"
    @test failure.destination === nothing

    persistent_models = study.release_models(case; e_to_e=6.0)
    persistent_search = study.search_context(persistent_models.off, 11)
    persistent = study.observe_phase(persistent_models.off, persistent_search,
        searches.on.equilibria[source].state, config)
    @test persistent.status == "compatible"
    @test persistent.final[1] > 0.49
end
